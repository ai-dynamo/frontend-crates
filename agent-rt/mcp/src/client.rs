// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use dynamo_agent_rt::{
    BoxFuture, ToolExecutionFailure, ToolExecutionRequest, ToolExecutionResult, ToolExecutor,
    ToolFailureDisposition, ToolFailurePolicy,
};
use rmcp::model::{
    CallToolRequestParams, CallToolResponse, ClientInfo, ContentBlock, Implementation,
    PaginatedRequestParams, ProtocolVersion,
};
use rmcp::service::RunningService;
use rmcp::transport::StreamableHttpClientTransport;
use rmcp::transport::streamable_http_client::StreamableHttpClientTransportConfig;
use rmcp::{ClientLifecycleMode, ClientServiceExt, RoleClient};
use thiserror::Error;
use tokio::sync::{Mutex, Semaphore};

use crate::{McpClientConfig, McpConfigError, McpToolDefinition, validate_name};

const MCP_CONNECTOR: &str = "mcp";
const CLIENT_NAME: &str = "dynamo-agent-rt-mcp";

type McpClient = RunningService<RoleClient, ClientInfo>;

/// Outbound executor for one trusted MCP server and one deployment profile.
pub struct McpToolExecutor {
    profile: String,
    config: McpClientConfig,
    tools: HashMap<String, McpToolDefinition>,
    http_client: reqwest::Client,
    concurrency: Arc<Semaphore>,
    client: Mutex<Option<Arc<McpClient>>>,
}

impl McpToolExecutor {
    pub fn new(
        profile: impl Into<String>,
        config: McpClientConfig,
    ) -> Result<Self, McpConfigError> {
        let profile = profile.into();
        validate_name("profile", &profile)?;
        let http_client = reqwest::Client::builder()
            .connect_timeout(config.connect_timeout())
            .redirect(reqwest::redirect::Policy::none())
            .user_agent(concat!("dynamo-agent-rt-mcp/", env!("CARGO_PKG_VERSION")))
            .build()
            .map_err(|error| McpConfigError::HttpClient(error.to_string()))?;
        let tools = config
            .tools()
            .iter()
            .cloned()
            .map(|tool| (tool.remote_name().to_owned(), tool))
            .collect();
        Ok(Self {
            profile,
            concurrency: Arc::new(Semaphore::new(config.max_concurrency())),
            config,
            tools,
            http_client,
            client: Mutex::new(None),
        })
    }

    pub fn profile(&self) -> &str {
        &self.profile
    }

    pub fn tools(&self) -> &[McpToolDefinition] {
        self.config.tools()
    }

    async fn execute_inner(
        &self,
        request: &ToolExecutionRequest,
    ) -> Result<ToolExecutionResult, McpToolExecutorError> {
        if request.connector != MCP_CONNECTOR || request.profile != self.profile {
            return Err(McpToolExecutorError::UnsupportedRoute {
                connector: request.connector.clone(),
                profile: request.profile.clone(),
            });
        }
        let tool = self
            .tools
            .get(&request.operation)
            .ok_or_else(|| McpToolExecutorError::ToolNotAllowed(request.operation.clone()))?;
        let arguments = request
            .arguments
            .as_object()
            .cloned()
            .ok_or(McpToolExecutorError::InvalidArguments)?;

        tokio::time::timeout(tool.timeout(), async {
            let _permit = self
                .concurrency
                .acquire()
                .await
                .map_err(|_| McpToolExecutorError::ExecutorClosed)?;
            let client = self.client().await?;
            let params =
                CallToolRequestParams::new(tool.remote_name().to_owned()).with_arguments(arguments);
            let response = client
                .call_tool_once(params)
                .await
                .map_err(|error| McpToolExecutorError::Call(error.to_string()))?;
            let CallToolResponse::Complete(result) = response else {
                return Err(McpToolExecutorError::UnsupportedContinuation);
            };
            normalize_result(result, tool.max_output_bytes())
        })
        .await
        .map_err(|_| McpToolExecutorError::Timeout)?
    }

    async fn client(&self) -> Result<Arc<McpClient>, McpToolExecutorError> {
        let mut slot = self.client.lock().await;
        if let Some(client) = slot.as_ref()
            && !client.is_closed()
            && !client.peer().is_transport_closed()
        {
            return Ok(Arc::clone(client));
        }

        let client = tokio::time::timeout(self.config.connect_timeout(), self.connect())
            .await
            .map_err(|_| McpToolExecutorError::ConnectTimeout)??;
        let client = Arc::new(client);
        *slot = Some(Arc::clone(&client));
        Ok(client)
    }

    async fn connect(&self) -> Result<McpClient, McpToolExecutorError> {
        let mut transport_config = StreamableHttpClientTransportConfig::with_uri(
            self.config.endpoint().as_str().to_owned(),
        )
        .max_sse_event_size(self.config.max_sse_event_bytes())
        .reinit_on_expired_session(true);
        if let Some(token) = self.config.bearer_token() {
            transport_config = transport_config.auth_header(token.expose().to_owned());
        }
        let transport =
            StreamableHttpClientTransport::with_client(self.http_client.clone(), transport_config);
        let mut client_info = ClientInfo::default();
        client_info.client_info = Implementation::new(CLIENT_NAME, env!("CARGO_PKG_VERSION"));
        let mut client = client_info
            .serve_with_lifecycle(
                transport,
                ClientLifecycleMode::Auto {
                    preferred_versions: vec![ProtocolVersion::V_2026_07_28],
                    legacy_version: Some(ProtocolVersion::V_2025_11_25),
                },
            )
            .await
            .map_err(|error| McpToolExecutorError::Connect(error.to_string()))?;
        if let Err(error) = self.verify_allowlist(&client).await {
            let _ = client
                .close_with_timeout(self.config.connect_timeout())
                .await;
            return Err(error);
        }
        tracing::debug!(
            endpoint = %self.config.endpoint(),
            profile = %self.profile,
            tools = self.tools.len(),
            "connected to configured MCP server"
        );
        Ok(client)
    }

    async fn verify_allowlist(&self, client: &McpClient) -> Result<(), McpToolExecutorError> {
        let mut advertised = HashMap::new();
        let mut seen_cursors = HashSet::new();
        let mut cursor = None;

        for page in 0..self.config.max_list_pages() {
            let params = cursor
                .clone()
                .map(|cursor| PaginatedRequestParams::default().with_cursor(Some(cursor)));
            let result = client
                .list_tools(params)
                .await
                .map_err(|error| McpToolExecutorError::ListTools(error.to_string()))?;
            if advertised.len().saturating_add(result.tools.len()) > self.config.max_list_tools() {
                return Err(McpToolExecutorError::TooManyAdvertisedTools {
                    limit: self.config.max_list_tools(),
                });
            }
            for tool in result.tools {
                let name = tool.name.into_owned();
                let schema = serde_json::Value::Object(tool.input_schema.as_ref().clone());
                if advertised.insert(name.clone(), schema).is_some() {
                    return Err(McpToolExecutorError::DuplicateAdvertisedTool(name));
                }
            }
            let Some(next_cursor) = result.next_cursor else {
                for tool in self.tools.values() {
                    let actual = advertised.get(tool.remote_name()).ok_or_else(|| {
                        McpToolExecutorError::MissingAdvertisedTool(tool.remote_name().to_owned())
                    })?;
                    if actual != tool.input_schema() {
                        return Err(McpToolExecutorError::SchemaMismatch(
                            tool.remote_name().to_owned(),
                        ));
                    }
                }
                return Ok(());
            };
            if !seen_cursors.insert(next_cursor.clone()) {
                return Err(McpToolExecutorError::RepeatedListCursor);
            }
            cursor = Some(next_cursor);
            if page + 1 == self.config.max_list_pages() {
                return Err(McpToolExecutorError::TooManyListPages {
                    limit: self.config.max_list_pages(),
                });
            }
        }
        Err(McpToolExecutorError::TooManyListPages {
            limit: self.config.max_list_pages(),
        })
    }
}

impl ToolExecutor for McpToolExecutor {
    type Error = McpToolExecutorError;

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
        // The initial MCP contract admits only deployment-classified read-only
        // tools. Re-execution cannot duplicate an external side effect.
        Box::pin(async move { self.execute_inner(request).await.map(Some) })
    }
}

fn normalize_result(
    result: rmcp::model::CallToolResult,
    max_output_bytes: usize,
) -> Result<ToolExecutionResult, McpToolExecutorError> {
    let mut text = Vec::with_capacity(result.content.len());
    for block in result.content {
        match block {
            ContentBlock::Text(content) => text.push(content.text),
            _ => return Err(McpToolExecutorError::UnsupportedContent),
        }
    }
    let output = result
        .structured_content
        .unwrap_or_else(|| match text.len() {
            0 => serde_json::Value::Null,
            1 => serde_json::Value::String(text.pop().expect("one text item")),
            _ => {
                serde_json::Value::Array(text.into_iter().map(serde_json::Value::String).collect())
            }
        });
    let output_bytes = serde_json::to_vec(&output)
        .map_err(|error| McpToolExecutorError::Normalize(error.to_string()))?
        .len();
    if output_bytes > max_output_bytes {
        return Err(McpToolExecutorError::OutputTooLarge {
            actual_bytes: output_bytes,
            limit_bytes: max_output_bytes,
        });
    }
    Ok(ToolExecutionResult {
        output,
        is_error: result.is_error.unwrap_or(false),
    })
}

#[derive(Debug, Error)]
pub enum McpToolExecutorError {
    #[error("unsupported MCP tool route {connector}/{profile}")]
    UnsupportedRoute { connector: String, profile: String },
    #[error("MCP tool {0:?} is not in the deployment allowlist")]
    ToolNotAllowed(String),
    #[error("MCP tool arguments must be a JSON object")]
    InvalidArguments,
    #[error("MCP executor is shutting down")]
    ExecutorClosed,
    #[error("MCP connection timed out")]
    ConnectTimeout,
    #[error("MCP connection failed: {0}")]
    Connect(String),
    #[error("MCP tools/list failed: {0}")]
    ListTools(String),
    #[error("MCP server advertised more than {limit} tools")]
    TooManyAdvertisedTools { limit: usize },
    #[error("MCP server returned more than {limit} tools/list pages")]
    TooManyListPages { limit: usize },
    #[error("MCP server repeated a tools/list cursor")]
    RepeatedListCursor,
    #[error("MCP server advertised duplicate tool {0:?}")]
    DuplicateAdvertisedTool(String),
    #[error("MCP server did not advertise configured tool {0:?}")]
    MissingAdvertisedTool(String),
    #[error("MCP server schema does not match configured tool {0:?}")]
    SchemaMismatch(String),
    #[error("MCP tool execution timed out")]
    Timeout,
    #[error("MCP tools/call failed: {0}")]
    Call(String),
    #[error("MCP tool requested an unsupported continuation or task")]
    UnsupportedContinuation,
    #[error("MCP tool returned unsupported non-text content")]
    UnsupportedContent,
    #[error("MCP tool output contains {actual_bytes} bytes; limit is {limit_bytes}")]
    OutputTooLarge {
        actual_bytes: usize,
        limit_bytes: usize,
    },
    #[error("MCP tool output normalization failed: {0}")]
    Normalize(String),
}

/// Failure mapping for the initial read-only MCP contract.
#[derive(Debug, Clone, Copy, Default)]
pub struct McpToolFailurePolicy;

impl ToolFailurePolicy<McpToolExecutorError> for McpToolFailurePolicy {
    fn classify(&self, error: &McpToolExecutorError) -> ToolFailureDisposition {
        let (code, message, retryable) = match error {
            McpToolExecutorError::UnsupportedRoute { .. }
            | McpToolExecutorError::ToolNotAllowed(_) => (
                "unsupported_tool_route",
                "The MCP tool route is not configured",
                false,
            ),
            McpToolExecutorError::InvalidArguments => (
                "invalid_tool_arguments",
                "The MCP tool arguments are invalid",
                false,
            ),
            McpToolExecutorError::ExecutorClosed => (
                "tool_unavailable",
                "The MCP tool executor is unavailable",
                true,
            ),
            McpToolExecutorError::ConnectTimeout | McpToolExecutorError::Timeout => {
                ("tool_timeout", "The MCP tool request timed out", true)
            }
            McpToolExecutorError::Connect(_)
            | McpToolExecutorError::ListTools(_)
            | McpToolExecutorError::Call(_) => (
                "mcp_transport",
                "The configured MCP server could not complete the request",
                true,
            ),
            McpToolExecutorError::TooManyAdvertisedTools { .. }
            | McpToolExecutorError::TooManyListPages { .. }
            | McpToolExecutorError::RepeatedListCursor
            | McpToolExecutorError::DuplicateAdvertisedTool(_)
            | McpToolExecutorError::MissingAdvertisedTool(_)
            | McpToolExecutorError::SchemaMismatch(_) => (
                "mcp_descriptor_mismatch",
                "The configured MCP server does not match its deployment descriptor",
                false,
            ),
            McpToolExecutorError::UnsupportedContinuation
            | McpToolExecutorError::UnsupportedContent
            | McpToolExecutorError::OutputTooLarge { .. }
            | McpToolExecutorError::Normalize(_) => (
                "mcp_response_invalid",
                "The MCP server returned an unsupported response",
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

#[cfg(test)]
mod tests {
    use rmcp::model::{CallToolResult, ContentBlock};
    use serde_json::json;

    use super::{McpToolExecutorError, normalize_result};

    #[test]
    fn structured_content_is_preferred_and_error_is_preserved() {
        let result = CallToolResult::structured_error(json!({"reason": "denied"}));
        let normalized = normalize_result(result, 1024).unwrap();
        assert_eq!(normalized.output, json!({"reason": "denied"}));
        assert!(normalized.is_error);
    }

    #[test]
    fn multiple_text_blocks_remain_distinct() {
        let result = CallToolResult::success(vec![
            ContentBlock::text("first"),
            ContentBlock::text("second"),
        ]);
        let normalized = normalize_result(result, 1024).unwrap();
        assert_eq!(normalized.output, json!(["first", "second"]));
        assert!(!normalized.is_error);
    }

    #[test]
    fn output_limit_applies_after_normalization() {
        let result = CallToolResult::success(vec![ContentBlock::text("too large")]);
        let error = normalize_result(result, 4).unwrap_err();
        assert!(matches!(error, McpToolExecutorError::OutputTooLarge { .. }));
    }

    #[test]
    fn binary_content_is_rejected() {
        let result = CallToolResult::success(vec![ContentBlock::image("AAAA", "image/png")]);
        let error = normalize_result(result, 1024).unwrap_err();
        assert!(matches!(error, McpToolExecutorError::UnsupportedContent));
    }
}
