// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Narrow, deployment-configured outbound MCP execution.

#![cfg_attr(docsrs, feature(doc_cfg))]

mod client;

use std::fmt;
use std::time::Duration;

use thiserror::Error;
use url::{Host, Url};

pub use client::{McpToolExecutor, McpToolExecutorError, McpToolFailurePolicy};

const DEFAULT_CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
const DEFAULT_CALL_TIMEOUT: Duration = Duration::from_secs(30);
const DEFAULT_MAX_OUTPUT_BYTES: usize = 1024 * 1024;
const DEFAULT_MAX_SSE_EVENT_BYTES: usize = 1024 * 1024;
const DEFAULT_MAX_LIST_PAGES: usize = 16;
const DEFAULT_MAX_LIST_TOOLS: usize = 256;
const DEFAULT_MAX_CONCURRENCY: usize = 16;
const MAX_NAME_BYTES: usize = 128;
const MAX_DESCRIPTION_BYTES: usize = 4096;

/// Bearer credential kept outside tool requests and durable state.
#[derive(Clone, PartialEq, Eq)]
pub struct McpBearerToken(String);

impl McpBearerToken {
    pub fn new(value: impl Into<String>) -> Result<Self, McpConfigError> {
        let value = value.into();
        if value.is_empty() {
            return Err(McpConfigError::EmptyBearerToken);
        }
        if value.bytes().any(|byte| byte.is_ascii_control()) {
            return Err(McpConfigError::InvalidBearerToken);
        }
        Ok(Self(value))
    }

    pub(crate) fn expose(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for McpBearerToken {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("McpBearerToken([REDACTED])")
    }
}

/// Deployment-owned descriptor for one model-visible, read-only MCP tool.
#[derive(Debug, Clone, PartialEq)]
pub struct McpToolDefinition {
    public_name: String,
    remote_name: String,
    description: String,
    input_schema: serde_json::Value,
    timeout: Duration,
    max_output_bytes: usize,
}

impl McpToolDefinition {
    pub fn new(
        public_name: impl Into<String>,
        remote_name: impl Into<String>,
        description: impl Into<String>,
        input_schema: serde_json::Value,
    ) -> Result<Self, McpConfigError> {
        let definition = Self {
            public_name: public_name.into(),
            remote_name: remote_name.into(),
            description: description.into(),
            input_schema,
            timeout: DEFAULT_CALL_TIMEOUT,
            max_output_bytes: DEFAULT_MAX_OUTPUT_BYTES,
        };
        definition.validate()?;
        Ok(definition)
    }

    pub fn with_limits(
        mut self,
        timeout: Duration,
        max_output_bytes: usize,
    ) -> Result<Self, McpConfigError> {
        self.timeout = timeout;
        self.max_output_bytes = max_output_bytes;
        self.validate()?;
        Ok(self)
    }

    pub fn public_name(&self) -> &str {
        &self.public_name
    }

    pub fn remote_name(&self) -> &str {
        &self.remote_name
    }

    pub fn description(&self) -> &str {
        &self.description
    }

    pub fn input_schema(&self) -> &serde_json::Value {
        &self.input_schema
    }

    pub(crate) fn timeout(&self) -> Duration {
        self.timeout
    }

    pub(crate) fn max_output_bytes(&self) -> usize {
        self.max_output_bytes
    }

    fn validate(&self) -> Result<(), McpConfigError> {
        validate_name("public", &self.public_name)?;
        validate_name("remote", &self.remote_name)?;
        if self.description.len() > MAX_DESCRIPTION_BYTES {
            return Err(McpConfigError::DescriptionTooLarge {
                tool: self.public_name.clone(),
                max_bytes: MAX_DESCRIPTION_BYTES,
            });
        }
        let schema =
            self.input_schema
                .as_object()
                .ok_or_else(|| McpConfigError::InvalidInputSchema {
                    tool: self.public_name.clone(),
                    reason: "root must be a JSON object".to_owned(),
                })?;
        if schema.get("type").and_then(serde_json::Value::as_str) != Some("object") {
            return Err(McpConfigError::InvalidInputSchema {
                tool: self.public_name.clone(),
                reason: "root type must be object".to_owned(),
            });
        }
        if schema
            .get("additionalProperties")
            .and_then(serde_json::Value::as_bool)
            != Some(false)
        {
            return Err(McpConfigError::InvalidInputSchema {
                tool: self.public_name.clone(),
                reason: "root additionalProperties must be false".to_owned(),
            });
        }
        if self.timeout.is_zero() {
            return Err(McpConfigError::InvalidToolTimeout(self.public_name.clone()));
        }
        if self.max_output_bytes == 0 {
            return Err(McpConfigError::InvalidToolOutputLimit(
                self.public_name.clone(),
            ));
        }
        Ok(())
    }
}

/// One trusted MCP server and its fixed deployment-owned allowlist.
#[derive(Debug, Clone)]
pub struct McpClientConfig {
    endpoint: Url,
    bearer_token: Option<McpBearerToken>,
    tools: Vec<McpToolDefinition>,
    connect_timeout: Duration,
    max_sse_event_bytes: usize,
    max_list_pages: usize,
    max_list_tools: usize,
    max_concurrency: usize,
}

impl McpClientConfig {
    pub fn new(
        endpoint: Url,
        tools: impl IntoIterator<Item = McpToolDefinition>,
    ) -> Result<Self, McpConfigError> {
        let config = Self {
            endpoint,
            bearer_token: None,
            tools: tools.into_iter().collect(),
            connect_timeout: DEFAULT_CONNECT_TIMEOUT,
            max_sse_event_bytes: DEFAULT_MAX_SSE_EVENT_BYTES,
            max_list_pages: DEFAULT_MAX_LIST_PAGES,
            max_list_tools: DEFAULT_MAX_LIST_TOOLS,
            max_concurrency: DEFAULT_MAX_CONCURRENCY,
        };
        config.validate(false)?;
        Ok(config)
    }

    /// Allows plaintext HTTP only when the endpoint resolves syntactically to loopback.
    pub fn new_for_loopback(
        endpoint: Url,
        tools: impl IntoIterator<Item = McpToolDefinition>,
    ) -> Result<Self, McpConfigError> {
        let config = Self {
            endpoint,
            bearer_token: None,
            tools: tools.into_iter().collect(),
            connect_timeout: DEFAULT_CONNECT_TIMEOUT,
            max_sse_event_bytes: DEFAULT_MAX_SSE_EVENT_BYTES,
            max_list_pages: DEFAULT_MAX_LIST_PAGES,
            max_list_tools: DEFAULT_MAX_LIST_TOOLS,
            max_concurrency: DEFAULT_MAX_CONCURRENCY,
        };
        config.validate(true)?;
        Ok(config)
    }

    pub fn with_bearer_token(mut self, token: McpBearerToken) -> Self {
        self.bearer_token = Some(token);
        self
    }

    pub fn with_limits(
        mut self,
        connect_timeout: Duration,
        max_sse_event_bytes: usize,
        max_list_pages: usize,
        max_list_tools: usize,
        max_concurrency: usize,
    ) -> Result<Self, McpConfigError> {
        self.connect_timeout = connect_timeout;
        self.max_sse_event_bytes = max_sse_event_bytes;
        self.max_list_pages = max_list_pages;
        self.max_list_tools = max_list_tools;
        self.max_concurrency = max_concurrency;
        let allow_loopback = self.endpoint.scheme() == "http";
        self.validate(allow_loopback)?;
        Ok(self)
    }

    pub fn endpoint(&self) -> &Url {
        &self.endpoint
    }

    pub fn tools(&self) -> &[McpToolDefinition] {
        &self.tools
    }

    pub(crate) fn bearer_token(&self) -> Option<&McpBearerToken> {
        self.bearer_token.as_ref()
    }

    pub(crate) fn connect_timeout(&self) -> Duration {
        self.connect_timeout
    }

    pub(crate) fn max_sse_event_bytes(&self) -> usize {
        self.max_sse_event_bytes
    }

    pub(crate) fn max_list_pages(&self) -> usize {
        self.max_list_pages
    }

    pub(crate) fn max_list_tools(&self) -> usize {
        self.max_list_tools
    }

    pub(crate) fn max_concurrency(&self) -> usize {
        self.max_concurrency
    }

    fn validate(&self, allow_http_loopback: bool) -> Result<(), McpConfigError> {
        validate_endpoint(&self.endpoint, allow_http_loopback)?;
        if self.tools.is_empty() {
            return Err(McpConfigError::EmptyToolAllowlist);
        }
        let mut public_names = std::collections::HashSet::new();
        let mut remote_names = std::collections::HashSet::new();
        for tool in &self.tools {
            tool.validate()?;
            if !public_names.insert(tool.public_name.clone()) {
                return Err(McpConfigError::DuplicatePublicTool(
                    tool.public_name.clone(),
                ));
            }
            if !remote_names.insert(tool.remote_name.clone()) {
                return Err(McpConfigError::DuplicateRemoteTool(
                    tool.remote_name.clone(),
                ));
            }
        }
        if self.connect_timeout.is_zero() {
            return Err(McpConfigError::InvalidConnectTimeout);
        }
        if self.max_sse_event_bytes == 0 {
            return Err(McpConfigError::InvalidSseEventLimit);
        }
        if self.max_list_pages == 0 {
            return Err(McpConfigError::InvalidListPageLimit);
        }
        if self.max_list_tools == 0 || self.max_list_tools < self.tools.len() {
            return Err(McpConfigError::InvalidListToolLimit {
                configured: self.tools.len(),
                limit: self.max_list_tools,
            });
        }
        if self.max_concurrency == 0 {
            return Err(McpConfigError::InvalidConcurrency);
        }
        Ok(())
    }
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum McpConfigError {
    #[error("MCP bearer token must not be empty")]
    EmptyBearerToken,
    #[error("MCP bearer token contains invalid control characters")]
    InvalidBearerToken,
    #[error(
        "MCP endpoint must use HTTPS; plaintext HTTP is allowed only for explicit loopback development"
    )]
    InsecureEndpoint,
    #[error("MCP endpoint must not contain user information, a query, or a fragment")]
    EndpointContainsUnsafeComponents,
    #[error("MCP endpoint must have a host")]
    EndpointMissingHost,
    #[error("MCP tool allowlist must not be empty")]
    EmptyToolAllowlist,
    #[error("MCP {kind} tool name must contain 1..={MAX_NAME_BYTES} bytes: {name:?}")]
    InvalidToolName { kind: &'static str, name: String },
    #[error("MCP tool {tool:?} description exceeds {max_bytes} bytes")]
    DescriptionTooLarge { tool: String, max_bytes: usize },
    #[error("MCP tool {tool:?} has an invalid input schema: {reason}")]
    InvalidInputSchema { tool: String, reason: String },
    #[error("MCP tool {0:?} timeout must be nonzero")]
    InvalidToolTimeout(String),
    #[error("MCP tool {0:?} output limit must be nonzero")]
    InvalidToolOutputLimit(String),
    #[error("duplicate public MCP tool {0:?}")]
    DuplicatePublicTool(String),
    #[error("duplicate remote MCP tool {0:?}")]
    DuplicateRemoteTool(String),
    #[error("MCP connection timeout must be nonzero")]
    InvalidConnectTimeout,
    #[error("MCP SSE event limit must be nonzero")]
    InvalidSseEventLimit,
    #[error("MCP tools/list page limit must be nonzero")]
    InvalidListPageLimit,
    #[error("MCP tools/list limit {limit} is smaller than the {configured} configured tools")]
    InvalidListToolLimit { configured: usize, limit: usize },
    #[error("MCP concurrency must be nonzero")]
    InvalidConcurrency,
    #[error("failed to construct the MCP HTTP client: {0}")]
    HttpClient(String),
}

fn validate_name(kind: &'static str, name: &str) -> Result<(), McpConfigError> {
    if name.trim().is_empty()
        || name.len() > MAX_NAME_BYTES
        || name.bytes().any(|byte| byte.is_ascii_control())
    {
        return Err(McpConfigError::InvalidToolName {
            kind,
            name: name.to_owned(),
        });
    }
    Ok(())
}

fn validate_endpoint(endpoint: &Url, allow_http_loopback: bool) -> Result<(), McpConfigError> {
    if !endpoint.username().is_empty()
        || endpoint.password().is_some()
        || endpoint.query().is_some()
        || endpoint.fragment().is_some()
    {
        return Err(McpConfigError::EndpointContainsUnsafeComponents);
    }
    let host = endpoint.host().ok_or(McpConfigError::EndpointMissingHost)?;
    let is_loopback = match host {
        Host::Ipv4(address) => address.is_loopback(),
        Host::Ipv6(address) => address.is_loopback(),
        Host::Domain(domain) => domain.eq_ignore_ascii_case("localhost"),
    };
    match endpoint.scheme() {
        "https" => Ok(()),
        "http" if allow_http_loopback && is_loopback => Ok(()),
        _ => Err(McpConfigError::InsecureEndpoint),
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use serde_json::json;
    use url::Url;

    use super::{McpBearerToken, McpClientConfig, McpConfigError, McpToolDefinition};

    fn tool() -> McpToolDefinition {
        McpToolDefinition::new(
            "search",
            "search",
            "Search",
            json!({
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
                "additionalProperties": false
            }),
        )
        .unwrap()
    }

    #[test]
    fn bearer_token_debug_is_redacted() {
        let token = McpBearerToken::new("top-secret").unwrap();
        assert_eq!(format!("{token:?}"), "McpBearerToken([REDACTED])");
        assert!(!format!("{token:?}").contains("top-secret"));
    }

    #[test]
    fn production_endpoint_requires_https() {
        let error =
            McpClientConfig::new(Url::parse("http://mcp.example.com/api").unwrap(), [tool()])
                .unwrap_err();
        assert_eq!(error, McpConfigError::InsecureEndpoint);
    }

    #[test]
    fn loopback_http_requires_explicit_constructor() {
        let endpoint = Url::parse("http://127.0.0.1:8080/mcp").unwrap();
        assert!(McpClientConfig::new(endpoint.clone(), [tool()]).is_err());
        assert!(McpClientConfig::new_for_loopback(endpoint, [tool()]).is_ok());
    }

    #[test]
    fn strict_object_schema_is_required() {
        let error = McpToolDefinition::new("search", "search", "Search", json!({"type": "object"}))
            .unwrap_err();
        assert!(matches!(error, McpConfigError::InvalidInputSchema { .. }));
    }

    #[test]
    fn zero_limits_are_rejected() {
        let error = tool().with_limits(Duration::ZERO, 1).unwrap_err();
        assert_eq!(
            error,
            McpConfigError::InvalidToolTimeout("search".to_owned())
        );
    }

    #[test]
    fn endpoint_cannot_smuggle_credentials() {
        let endpoint = Url::parse("https://user:pass@mcp.example.com/api?token=secret").unwrap();
        let error = McpClientConfig::new(endpoint, [tool()]).unwrap_err();
        assert_eq!(error, McpConfigError::EndpointContainsUnsafeComponents);
    }
}
