// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

use dynamo_agent_rt::{
    BoxFuture, ToolExecutionFailure, ToolExecutionRequest, ToolExecutionResult, ToolExecutor,
    ToolFailureDisposition, ToolFailurePolicy,
};
use reqwest::header::{ACCEPT, HeaderValue};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::sync::Semaphore;
use url::Url;

const PROVIDER_MAX_QUERY_BYTES: usize = 400;
const PROVIDER_MAX_QUERY_WORDS: usize = 50;
const PROVIDER_MAX_RESULTS: u8 = 20;
const DEFAULT_MAX_RESULTS: u8 = 5;
const DEFAULT_RESPONSE_BYTES: usize = 1024 * 1024;
const DEFAULT_CONCURRENCY: usize = 16;
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(10);
const BRAVE_WEB_SEARCH_ENDPOINT: &str = "https://api.search.brave.com/res/v1/web/search";
const BRAVE_API_VERSION: &str = "2023-01-01";
const WEB_SEARCH_CONNECTOR: &str = "web_search";
const WEB_SEARCH_OPERATION: &str = "search";

/// Model-visible arguments for the read-only `web_search` tool.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WebSearchArguments {
    pub query: String,
    pub count: u8,
    pub freshness: Option<WebSearchFreshness>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawWebSearchArguments {
    query: String,
    #[serde(default)]
    count: Option<u8>,
    #[serde(default)]
    freshness: Option<WebSearchFreshness>,
}

impl WebSearchArguments {
    pub fn from_value(
        arguments: serde_json::Value,
        max_results: u8,
    ) -> Result<Self, WebSearchArgumentsError> {
        let arguments: RawWebSearchArguments = serde_json::from_value(arguments)
            .map_err(|error| WebSearchArgumentsError::Schema(error.to_string()))?;
        let query = arguments.query.trim().to_owned();
        if query.is_empty() {
            return Err(WebSearchArgumentsError::EmptyQuery);
        }
        let query_bytes = query.len();
        let query_words = query.split_whitespace().count();
        if query_bytes > PROVIDER_MAX_QUERY_BYTES || query_words > PROVIDER_MAX_QUERY_WORDS {
            return Err(WebSearchArgumentsError::QueryTooLarge {
                actual_bytes: query_bytes,
                actual_words: query_words,
            });
        }
        let count = arguments.count.unwrap_or(max_results);
        if count == 0 || count > max_results {
            return Err(WebSearchArgumentsError::InvalidCount {
                count,
                max: max_results,
            });
        }
        Ok(Self {
            query,
            count,
            freshness: arguments.freshness,
        })
    }

    pub fn effective_count(&self) -> u8 {
        self.count
    }
}

/// Coarse recency filters intentionally exposed to the model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WebSearchFreshness {
    Day,
    Week,
    Month,
    Year,
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum WebSearchArgumentsError {
    #[error("web-search arguments do not match the required schema: {0}")]
    Schema(String),
    #[error("web-search query must not be empty")]
    EmptyQuery,
    #[error(
        "web-search query contains {actual_bytes} bytes and {actual_words} words; maximum is 400 bytes and 50 words"
    )]
    QueryTooLarge {
        actual_bytes: usize,
        actual_words: usize,
    },
    #[error("web-search result count {count} is outside the configured range 1..={max}")]
    InvalidCount { count: u8, max: u8 },
}

/// Provider-neutral result persisted in the tool journal and sent to the model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WebSearchOutput {
    pub query: String,
    /// Search results are external, untrusted model input.
    pub content_is_untrusted: bool,
    pub results: Vec<WebSearchResult>,
    pub more_results_available: bool,
}

/// Citation-bearing normalized web result.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WebSearchResult {
    pub title: String,
    pub url: String,
    pub snippet: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub published: Option<String>,
}

/// Deployment-owned Brave Search settings. The API credential never enters a
/// [`ToolExecutionRequest`] or durable tool journal.
#[derive(Clone)]
pub struct BraveWebSearchProfile {
    api_key: HeaderValue,
    endpoint: Url,
    country: String,
    search_language: String,
    max_results: u8,
    max_response_bytes: usize,
    concurrency: usize,
    timeout: Duration,
}

impl BraveWebSearchProfile {
    pub fn new(api_key: impl AsRef<str>) -> Result<Self, WebSearchConfigError> {
        let api_key = api_key.as_ref();
        if api_key.is_empty() {
            return Err(WebSearchConfigError::EmptyApiKey);
        }
        let mut api_key =
            HeaderValue::from_str(api_key).map_err(|_| WebSearchConfigError::InvalidApiKey)?;
        api_key.set_sensitive(true);
        Ok(Self {
            api_key,
            endpoint: Url::parse(BRAVE_WEB_SEARCH_ENDPOINT)
                .map_err(|error| WebSearchConfigError::InvalidEndpoint(error.to_string()))?,
            country: "US".to_owned(),
            search_language: "en".to_owned(),
            max_results: DEFAULT_MAX_RESULTS,
            max_response_bytes: DEFAULT_RESPONSE_BYTES,
            concurrency: DEFAULT_CONCURRENCY,
            timeout: DEFAULT_TIMEOUT,
        })
    }

    pub fn with_limits(
        mut self,
        max_results: u8,
        max_response_bytes: usize,
        concurrency: usize,
        timeout: Duration,
    ) -> Result<Self, WebSearchConfigError> {
        if max_results == 0 || max_results > PROVIDER_MAX_RESULTS {
            return Err(WebSearchConfigError::InvalidMaxResults(max_results));
        }
        if max_response_bytes == 0 {
            return Err(WebSearchConfigError::InvalidResponseLimit);
        }
        if concurrency == 0 {
            return Err(WebSearchConfigError::InvalidConcurrency);
        }
        if timeout.is_zero() {
            return Err(WebSearchConfigError::InvalidTimeout);
        }
        self.max_results = max_results;
        self.max_response_bytes = max_response_bytes;
        self.concurrency = concurrency;
        self.timeout = timeout;
        Ok(self)
    }

    pub fn with_locale(
        mut self,
        country: impl Into<String>,
        search_language: impl Into<String>,
    ) -> Result<Self, WebSearchConfigError> {
        let country = country.into().to_ascii_uppercase();
        let search_language = search_language.into().to_ascii_lowercase();
        if country.len() != 2 || !country.bytes().all(|byte| byte.is_ascii_uppercase()) {
            return Err(WebSearchConfigError::InvalidCountry(country));
        }
        if !(2..=8).contains(&search_language.len())
            || !search_language
                .bytes()
                .all(|byte| byte.is_ascii_lowercase() || byte == b'-')
        {
            return Err(WebSearchConfigError::InvalidSearchLanguage(search_language));
        }
        self.country = country;
        self.search_language = search_language;
        Ok(self)
    }

    #[cfg(test)]
    fn with_test_endpoint(mut self, endpoint: Url) -> Self {
        self.endpoint = endpoint;
        self
    }
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum WebSearchConfigError {
    #[error("Brave Search API key must not be empty")]
    EmptyApiKey,
    #[error("Brave Search API key is not a valid HTTP header value")]
    InvalidApiKey,
    #[error("Brave Search endpoint is invalid: {0}")]
    InvalidEndpoint(String),
    #[error("Brave Search max_results must be in 1..=20, got {0}")]
    InvalidMaxResults(u8),
    #[error("Brave Search response byte limit must be nonzero")]
    InvalidResponseLimit,
    #[error("Brave Search concurrency must be nonzero")]
    InvalidConcurrency,
    #[error("Brave Search timeout must be nonzero")]
    InvalidTimeout,
    #[error("Brave Search country must be a two-letter code, got {0:?}")]
    InvalidCountry(String),
    #[error("Brave Search language code is invalid: {0:?}")]
    InvalidSearchLanguage(String),
    #[error("web-search profile name must not be empty")]
    EmptyProfileName,
    #[error("duplicate web-search profile {0:?}")]
    DuplicateProfile(String),
    #[error("failed to construct the Brave Search HTTP client: {0}")]
    HttpClient(String),
}

struct ProfileRuntime {
    config: BraveWebSearchProfile,
    concurrency: Arc<Semaphore>,
}

/// Bounded read-only ToolExecutor for Brave's Web Search endpoint.
pub struct BraveWebSearchExecutor {
    client: reqwest::Client,
    profiles: HashMap<String, ProfileRuntime>,
}

impl BraveWebSearchExecutor {
    pub fn new(
        profiles: impl IntoIterator<Item = (String, BraveWebSearchProfile)>,
    ) -> Result<Self, WebSearchConfigError> {
        let client = reqwest::Client::builder()
            .https_only(true)
            .redirect(reqwest::redirect::Policy::none())
            .user_agent("dynamo-agent-tools/0.1")
            .build()
            .map_err(|error| WebSearchConfigError::HttpClient(error.to_string()))?;
        Self::from_parts(profiles, client)
    }

    fn from_parts(
        profiles: impl IntoIterator<Item = (String, BraveWebSearchProfile)>,
        client: reqwest::Client,
    ) -> Result<Self, WebSearchConfigError> {
        let mut seen = HashSet::new();
        let mut runtimes = HashMap::new();
        for (name, config) in profiles {
            if name.trim().is_empty() {
                return Err(WebSearchConfigError::EmptyProfileName);
            }
            if !seen.insert(name.clone()) {
                return Err(WebSearchConfigError::DuplicateProfile(name));
            }
            runtimes.insert(
                name,
                ProfileRuntime {
                    concurrency: Arc::new(Semaphore::new(config.concurrency)),
                    config,
                },
            );
        }
        Ok(Self {
            client,
            profiles: runtimes,
        })
    }

    async fn execute_inner(
        &self,
        request: &ToolExecutionRequest,
    ) -> Result<ToolExecutionResult, BraveWebSearchError> {
        if request.connector != WEB_SEARCH_CONNECTOR || request.operation != WEB_SEARCH_OPERATION {
            return Err(BraveWebSearchError::UnsupportedRoute {
                connector: request.connector.clone(),
                operation: request.operation.clone(),
            });
        }
        let profile = self
            .profiles
            .get(&request.profile)
            .ok_or_else(|| BraveWebSearchError::UnknownProfile(request.profile.clone()))?;
        let arguments =
            WebSearchArguments::from_value(request.arguments.clone(), profile.config.max_results)?;
        let output = tokio::time::timeout(profile.config.timeout, async {
            let _permit = profile
                .concurrency
                .acquire()
                .await
                .map_err(|_| BraveWebSearchError::ExecutorClosed)?;
            self.search(&profile.config, &arguments).await
        })
        .await
        .map_err(|_| BraveWebSearchError::Timeout)??;
        Ok(ToolExecutionResult {
            output: serde_json::to_value(output).map_err(BraveWebSearchError::Normalize)?,
        })
    }

    async fn search(
        &self,
        profile: &BraveWebSearchProfile,
        arguments: &WebSearchArguments,
    ) -> Result<WebSearchOutput, BraveWebSearchError> {
        let count = arguments.effective_count().to_string();
        let mut query = vec![
            ("q", arguments.query.as_str()),
            ("count", count.as_str()),
            ("country", profile.country.as_str()),
            ("search_lang", profile.search_language.as_str()),
            ("safesearch", "strict"),
        ];
        let freshness = arguments.freshness.map(brave_freshness);
        if let Some(freshness) = freshness {
            query.push(("freshness", freshness));
        }
        let mut response = self
            .client
            .get(profile.endpoint.clone())
            .header(ACCEPT, "application/json")
            .header("api-version", BRAVE_API_VERSION)
            .header("x-subscription-token", profile.api_key.clone())
            .query(&query)
            .send()
            .await
            .map_err(BraveWebSearchError::Transport)?;
        let status = response.status();
        if !status.is_success() {
            return Err(BraveWebSearchError::ProviderStatus(status.as_u16()));
        }
        if response
            .content_length()
            .is_some_and(|length| length > profile.max_response_bytes as u64)
        {
            return Err(BraveWebSearchError::ResponseTooLarge {
                limit_bytes: profile.max_response_bytes,
            });
        }
        let mut body = Vec::new();
        while let Some(chunk) = response
            .chunk()
            .await
            .map_err(BraveWebSearchError::Transport)?
        {
            if body.len().saturating_add(chunk.len()) > profile.max_response_bytes {
                return Err(BraveWebSearchError::ResponseTooLarge {
                    limit_bytes: profile.max_response_bytes,
                });
            }
            body.extend_from_slice(&chunk);
        }
        let response: BraveSearchResponse =
            serde_json::from_slice(&body).map_err(BraveWebSearchError::Decode)?;
        Ok(normalize_response(
            arguments,
            response,
            profile.configured_result_limit(arguments),
        ))
    }
}

impl BraveWebSearchProfile {
    fn configured_result_limit(&self, arguments: &WebSearchArguments) -> usize {
        usize::from(arguments.effective_count().min(self.max_results))
    }
}

impl ToolExecutor for BraveWebSearchExecutor {
    type Error = BraveWebSearchError;

    fn execute(
        &self,
        request: ToolExecutionRequest,
    ) -> BoxFuture<'_, Result<ToolExecutionResult, Self::Error>> {
        Box::pin(async move { self.execute_inner(&request).await })
    }

    fn lookup<'a>(
        &'a self,
        request: &'a ToolExecutionRequest,
    ) -> BoxFuture<'a, Result<Option<ToolExecutionResult>, Self::Error>> {
        // Web search is explicitly read-only. Re-executing a journaled request
        // after restart cannot duplicate an external side effect.
        Box::pin(async move { self.execute_inner(request).await.map(Some) })
    }
}

#[derive(Debug, Error)]
pub enum BraveWebSearchError {
    #[error("unsupported tool route {connector}/{operation}")]
    UnsupportedRoute {
        connector: String,
        operation: String,
    },
    #[error("unknown web-search profile {0:?}")]
    UnknownProfile(String),
    #[error(transparent)]
    InvalidArguments(#[from] WebSearchArgumentsError),
    #[error("web-search executor is shutting down")]
    ExecutorClosed,
    #[error("web-search execution timed out")]
    Timeout,
    #[error("web-search provider transport failed: {0}")]
    Transport(reqwest::Error),
    #[error("web-search provider returned HTTP {0}")]
    ProviderStatus(u16),
    #[error("web-search provider response exceeded {limit_bytes} bytes")]
    ResponseTooLarge { limit_bytes: usize },
    #[error("web-search provider response was invalid: {0}")]
    Decode(serde_json::Error),
    #[error("web-search output normalization failed: {0}")]
    Normalize(serde_json::Error),
}

#[derive(Debug, Clone, Copy, Default)]
pub struct BraveWebSearchFailurePolicy;

impl ToolFailurePolicy<BraveWebSearchError> for BraveWebSearchFailurePolicy {
    fn classify(&self, error: &BraveWebSearchError) -> ToolFailureDisposition {
        let (code, message, retryable) = match error {
            BraveWebSearchError::UnsupportedRoute { .. } => (
                "unsupported_tool_route",
                "The web-search tool route is not configured",
                false,
            ),
            BraveWebSearchError::UnknownProfile(_) => (
                "unknown_tool_profile",
                "The web-search deployment profile is not configured",
                false,
            ),
            BraveWebSearchError::InvalidArguments(_) => (
                "invalid_tool_arguments",
                "The web-search arguments are invalid",
                false,
            ),
            BraveWebSearchError::ExecutorClosed => (
                "tool_unavailable",
                "The web-search executor is unavailable",
                true,
            ),
            BraveWebSearchError::Timeout => {
                ("tool_timeout", "The web-search request timed out", true)
            }
            BraveWebSearchError::Transport(_) => (
                "provider_transport",
                "The web-search provider could not be reached",
                true,
            ),
            BraveWebSearchError::ProviderStatus(status) => (
                "provider_status",
                "The web-search provider rejected the request",
                *status == 429 || *status >= 500,
            ),
            BraveWebSearchError::ResponseTooLarge { .. } => (
                "provider_response_too_large",
                "The web-search provider response exceeded deployment limits",
                false,
            ),
            BraveWebSearchError::Decode(_) | BraveWebSearchError::Normalize(_) => (
                "provider_response_invalid",
                "The web-search provider returned an invalid response",
                false,
            ),
        };
        ToolFailureDisposition::Failed(ToolExecutionFailure {
            code: code.to_owned(),
            message: message.to_owned(),
            retryable,
        })
    }
}

fn brave_freshness(freshness: WebSearchFreshness) -> &'static str {
    match freshness {
        WebSearchFreshness::Day => "pd",
        WebSearchFreshness::Week => "pw",
        WebSearchFreshness::Month => "pm",
        WebSearchFreshness::Year => "py",
    }
}

#[derive(Debug, Deserialize)]
struct BraveSearchResponse {
    #[serde(default)]
    query: Option<BraveQuery>,
    #[serde(default)]
    web: Option<BraveWebResults>,
}

#[derive(Debug, Deserialize)]
struct BraveQuery {
    #[serde(default)]
    more_results_available: bool,
}

#[derive(Debug, Deserialize)]
struct BraveWebResults {
    #[serde(default)]
    results: Vec<BraveWebResult>,
}

#[derive(Debug, Deserialize)]
struct BraveWebResult {
    title: String,
    url: String,
    #[serde(default)]
    description: String,
    #[serde(default)]
    age: Option<String>,
}

fn normalize_response(
    arguments: &WebSearchArguments,
    response: BraveSearchResponse,
    result_limit: usize,
) -> WebSearchOutput {
    let results = response
        .web
        .map(|web| web.results)
        .unwrap_or_default()
        .into_iter()
        .filter_map(normalize_result)
        .take(result_limit)
        .collect();
    WebSearchOutput {
        query: arguments.query.clone(),
        content_is_untrusted: true,
        results,
        more_results_available: response
            .query
            .is_some_and(|query| query.more_results_available),
    }
}

fn normalize_result(result: BraveWebResult) -> Option<WebSearchResult> {
    let url = Url::parse(&result.url).ok()?;
    if !matches!(url.scheme(), "http" | "https")
        || !url.username().is_empty()
        || url.password().is_some()
    {
        return None;
    }
    Some(WebSearchResult {
        title: truncate_utf8(result.title.trim(), 512),
        url: truncate_utf8(url.as_str(), 2048),
        snippet: truncate_utf8(result.description.trim(), 4096),
        published: result.age.map(|age| truncate_utf8(age.trim(), 128)),
    })
}

fn truncate_utf8(value: &str, max_bytes: usize) -> String {
    if value.len() <= max_bytes {
        return value.to_owned();
    }
    let mut boundary = max_bytes;
    while !value.is_char_boundary(boundary) {
        boundary -= 1;
    }
    value[..boundary].to_owned()
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeSet, HashMap};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    use axum::extract::{Query, State};
    use axum::http::{HeaderMap, StatusCode};
    use axum::response::{IntoResponse, Response};
    use axum::routing::get;
    use axum::{Json, Router};
    use dynamo_agent_rt::{
        AuthorizationScope, Blake3ToolIdempotencyKeys, IdempotencyKey, OpenAiResponses, ResponseId,
        RuntimeAuthorization, RuntimeLimits, RuntimeToolCall, ToolExecutionRequest, ToolExecutor,
        ToolFailureDisposition, ToolFailurePolicy, ToolIdempotencyKeyProvider, ToolJournal,
        ToolJournalState, ToolRunner,
    };
    use dynamo_agent_rt_store::SqliteStore;
    use serde_json::json;
    use tokio::net::TcpListener;
    use url::Url;

    use super::{
        BraveWebSearchError, BraveWebSearchExecutor, BraveWebSearchFailurePolicy,
        BraveWebSearchProfile, WebSearchArguments, WebSearchArgumentsError, WebSearchFreshness,
        WebSearchOutput,
    };

    #[derive(Clone, Default)]
    struct FakeProviderState {
        calls: Arc<AtomicUsize>,
    }

    async fn fake_brave_search(
        State(state): State<FakeProviderState>,
        Query(query): Query<HashMap<String, String>>,
        headers: HeaderMap,
    ) -> Response {
        state.calls.fetch_add(1, Ordering::SeqCst);
        if headers
            .get("x-subscription-token")
            .and_then(|value| value.to_str().ok())
            != Some("deployment-secret")
            || headers
                .get("api-version")
                .and_then(|value| value.to_str().ok())
                != Some("2023-01-01")
        {
            return StatusCode::UNAUTHORIZED.into_response();
        }
        if query.get("q").map(String::as_str) == Some("slow") {
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        if query.get("q").map(String::as_str) == Some("overloaded") {
            return StatusCode::TOO_MANY_REQUESTS.into_response();
        }
        if query.get("q").map(String::as_str) == Some("oversize") {
            return Json(json!({
                "web": {"results": [{
                    "title": "large",
                    "url": "https://example.com/large",
                    "description": "x".repeat(4096)
                }]}
            }))
            .into_response();
        }
        if query.get("q").map(String::as_str) != Some("dynamo agents")
            || query.get("count").map(String::as_str) != Some("2")
            || query.get("country").map(String::as_str) != Some("CA")
            || query.get("search_lang").map(String::as_str) != Some("en")
            || query.get("safesearch").map(String::as_str) != Some("strict")
            || query.get("freshness").map(String::as_str) != Some("pw")
        {
            return StatusCode::BAD_REQUEST.into_response();
        }
        Json(json!({
            "query": {"more_results_available": true},
            "web": {"results": [
                {
                    "title": "Dynamo agent runtime",
                    "url": "https://example.com/agent-runtime",
                    "description": "A bounded result",
                    "age": "1 day ago"
                },
                {
                    "title": "unsafe URL",
                    "url": "javascript:alert(1)",
                    "description": "must be dropped"
                },
                {
                    "title": "Second safe result",
                    "url": "http://example.org/two",
                    "description": "another result"
                }
            ]}
        }))
        .into_response()
    }

    async fn fake_executor(
        timeout: Duration,
        response_limit: usize,
    ) -> (
        BraveWebSearchExecutor,
        FakeProviderState,
        tokio::task::JoinHandle<()>,
    ) {
        let state = FakeProviderState::default();
        let app = Router::new()
            .route("/res/v1/web/search", get(fake_brave_search))
            .with_state(state.clone());
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        let endpoint = Url::parse(&format!("http://{address}/res/v1/web/search")).unwrap();
        let profile = BraveWebSearchProfile::new("deployment-secret")
            .unwrap()
            .with_limits(5, response_limit, 2, timeout)
            .unwrap()
            .with_locale("ca", "EN")
            .unwrap()
            .with_test_endpoint(endpoint);
        let client = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .unwrap();
        let executor =
            BraveWebSearchExecutor::from_parts([("default".to_owned(), profile)], client).unwrap();
        (executor, state, server)
    }

    fn tool_request(query: &str, count: Option<u8>) -> ToolExecutionRequest {
        let mut arguments = json!({
            "query": query,
            "freshness": "week"
        });
        if let Some(count) = count {
            arguments["count"] = json!(count);
        }
        ToolExecutionRequest {
            response_id: ResponseId::from("resp-1"),
            call_id: "call-1".to_owned(),
            connector: "web_search".to_owned(),
            operation: "search".to_owned(),
            profile: "default".to_owned(),
            arguments,
            scope: AuthorizationScope {
                tenant_id: "tenant-a".to_owned(),
                principal_id: "principal-a".to_owned(),
            },
            idempotency_key: IdempotencyKey::from("tool-key-1"),
            attempt: 0,
        }
    }

    #[tokio::test]
    async fn brave_executor_is_bounded_normalized_and_recoverable() {
        let (executor, state, server) = fake_executor(Duration::from_secs(1), 32 * 1024).await;
        let request = tool_request("dynamo agents", Some(2));

        let result = executor.execute(request.clone()).await.unwrap();
        let output: WebSearchOutput = serde_json::from_value(result.output.clone()).unwrap();
        assert!(output.content_is_untrusted);
        assert!(output.more_results_available);
        assert_eq!(output.results.len(), 2);
        assert_eq!(output.results[0].title, "Dynamo agent runtime");
        assert_eq!(output.results[0].url, "https://example.com/agent-runtime");
        assert_eq!(output.results[1].url, "http://example.org/two");

        let recovered = executor.lookup(&request).await.unwrap().unwrap();
        assert_eq!(recovered, result);
        assert_eq!(state.calls.load(Ordering::SeqCst), 2);
        server.abort();
    }

    #[tokio::test]
    async fn provider_timeout_status_and_body_limits_are_typed() {
        let (executor, _, server) = fake_executor(Duration::from_millis(10), 512).await;
        assert!(matches!(
            executor.execute(tool_request("slow", None)).await,
            Err(BraveWebSearchError::Timeout)
        ));
        assert!(matches!(
            executor.execute(tool_request("overloaded", None)).await,
            Err(BraveWebSearchError::ProviderStatus(429))
        ));
        assert!(matches!(
            executor.execute(tool_request("oversize", None)).await,
            Err(BraveWebSearchError::ResponseTooLarge { limit_bytes: 512 })
        ));
        server.abort();
    }

    #[tokio::test]
    async fn started_search_recovers_through_durable_sqlite_journal() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("agent-tools.sqlite");
        let journal = SqliteStore::<OpenAiResponses>::open(&path).unwrap();
        let (executor, state, server) = fake_executor(Duration::from_secs(1), 32 * 1024).await;
        let response_id = ResponseId::from("resp-recovery");
        let call = RuntimeToolCall {
            call_id: "call-recovery".to_owned(),
            connector: "web_search".to_owned(),
            operation: "search".to_owned(),
            profile: "default".to_owned(),
            arguments: json!({
                "query": "dynamo agents",
                "count": 2,
                "freshness": "week"
            }),
        };
        let scope = AuthorizationScope {
            tenant_id: "tenant-a".to_owned(),
            principal_id: "principal-a".to_owned(),
        };
        let idempotency_key = Blake3ToolIdempotencyKeys.idempotency_key(&response_id, &call, 0);
        let execution_request = ToolExecutionRequest {
            response_id: response_id.clone(),
            call_id: call.call_id.clone(),
            connector: call.connector.clone(),
            operation: call.operation.clone(),
            profile: call.profile.clone(),
            arguments: call.arguments.clone(),
            scope: scope.clone(),
            idempotency_key: idempotency_key.clone(),
            attempt: 0,
        };
        journal.claim(execution_request).await.unwrap();
        let runner = ToolRunner::new(
            journal.clone(),
            executor,
            Blake3ToolIdempotencyKeys,
            BraveWebSearchFailurePolicy,
        );
        let authorization = RuntimeAuthorization {
            scope: scope.clone(),
            permitted_connectors: BTreeSet::from(["web_search".to_owned()]),
            limits: RuntimeLimits::default(),
        };

        let result = runner
            .run(&response_id, call, &authorization, 0)
            .await
            .unwrap();
        assert_eq!(result.result.output["content_is_untrusted"], true);
        assert_eq!(state.calls.load(Ordering::SeqCst), 1);

        drop(runner);
        drop(journal);
        let reopened = SqliteStore::<OpenAiResponses>::open(&path).unwrap();
        let record = reopened
            .load(&dynamo_agent_rt::ToolJournalKey {
                scope,
                idempotency_key,
            })
            .await
            .unwrap()
            .unwrap();
        assert_eq!(record.state, ToolJournalState::Completed);
        assert_eq!(record.result.unwrap().output, result.result.output);
        server.abort();
    }

    #[tokio::test]
    #[ignore = "requires BRAVE_SEARCH_API_KEY and external network access"]
    async fn live_brave_search() {
        let api_key = std::env::var("BRAVE_SEARCH_API_KEY")
            .expect("BRAVE_SEARCH_API_KEY is required for the ignored live test");
        let profile = BraveWebSearchProfile::new(api_key).unwrap();
        let executor = BraveWebSearchExecutor::new([("default".to_owned(), profile)]).unwrap();

        let result = executor
            .execute(tool_request("NVIDIA Dynamo inference", Some(3)))
            .await
            .unwrap();
        let output: WebSearchOutput = serde_json::from_value(result.output).unwrap();
        assert!(output.content_is_untrusted);
        assert!(!output.results.is_empty());
        assert!(
            output
                .results
                .iter()
                .all(|result| result.url.starts_with("http://")
                    || result.url.starts_with("https://"))
        );
    }

    #[test]
    fn read_only_failure_policy_never_reports_unknown_side_effects() {
        let disposition =
            BraveWebSearchFailurePolicy.classify(&BraveWebSearchError::ProviderStatus(429));
        assert!(matches!(
            disposition,
            ToolFailureDisposition::Failed(failure)
                if failure.code == "provider_status" && failure.retryable
        ));
    }

    #[test]
    fn arguments_are_trimmed_and_bounded_by_deployment_policy() {
        let arguments = WebSearchArguments::from_value(
            json!({"query": "  dynamo agents  ", "count": 3, "freshness": "week"}),
            5,
        )
        .unwrap();

        assert_eq!(arguments.query, "dynamo agents");
        assert_eq!(arguments.effective_count(), 3);
        assert_eq!(arguments.freshness, Some(WebSearchFreshness::Week));
    }

    #[test]
    fn model_cannot_exceed_result_policy_or_add_provider_fields() {
        assert_eq!(
            WebSearchArguments::from_value(json!({"query": "x", "count": 6}), 5).unwrap_err(),
            WebSearchArgumentsError::InvalidCount { count: 6, max: 5 }
        );
        assert!(matches!(
            WebSearchArguments::from_value(
                json!({"query": "x", "endpoint": "https://attacker.invalid"}),
                5
            ),
            Err(WebSearchArgumentsError::Schema(_))
        ));
    }

    #[test]
    fn provider_query_limits_are_enforced_before_dispatch() {
        let too_many_words = (0..51).map(|_| "word").collect::<Vec<_>>().join(" ");
        assert!(matches!(
            WebSearchArguments::from_value(json!({"query": too_many_words}), 5),
            Err(WebSearchArgumentsError::QueryTooLarge { .. })
        ));
        assert_eq!(
            WebSearchArguments::from_value(json!({"query": "  "}), 5).unwrap_err(),
            WebSearchArgumentsError::EmptyQuery
        );
    }
}
