// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde::{Deserialize, Serialize};
use thiserror::Error;

const PROVIDER_MAX_QUERY_BYTES: usize = 400;
const PROVIDER_MAX_QUERY_WORDS: usize = 50;

/// Model-visible arguments for the read-only `web_search` tool.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WebSearchArguments {
    pub query: String,
    #[serde(default)]
    pub count: Option<u8>,
    #[serde(default)]
    pub freshness: Option<WebSearchFreshness>,
}

impl WebSearchArguments {
    pub fn from_value(
        arguments: serde_json::Value,
        max_results: u8,
    ) -> Result<Self, WebSearchArgumentsError> {
        let mut arguments: Self = serde_json::from_value(arguments)
            .map_err(|error| WebSearchArgumentsError::Schema(error.to_string()))?;
        arguments.query = arguments.query.trim().to_owned();
        if arguments.query.is_empty() {
            return Err(WebSearchArgumentsError::EmptyQuery);
        }
        let query_bytes = arguments.query.len();
        let query_words = arguments.query.split_whitespace().count();
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
        arguments.count = Some(count);
        Ok(arguments)
    }

    pub fn effective_count(&self) -> u8 {
        self.count.expect("validated web-search count")
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

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{WebSearchArguments, WebSearchArgumentsError, WebSearchFreshness};

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
