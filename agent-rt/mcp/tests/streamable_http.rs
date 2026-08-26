// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Json, Router};
use dynamo_agent_rt::{
    AuthorizationScope, IdempotencyKey, ResponseId, ToolExecutionRequest, ToolExecutor,
};
use dynamo_agent_rt_mcp::{
    McpBearerToken, McpClientConfig, McpToolDefinition, McpToolExecutor, McpToolExecutorError,
};
use serde_json::{Value, json};
use tokio::net::TcpListener;
use tokio::task::JoinHandle;
use url::Url;

#[derive(Clone)]
struct FixtureState {
    pages: Arc<Vec<Vec<Value>>>,
    call_result: Arc<Value>,
    call_delay: Duration,
    methods: Arc<Mutex<Vec<String>>>,
    authorization: Arc<Mutex<Vec<Option<String>>>>,
    call_count: Arc<AtomicUsize>,
}

struct Fixture {
    endpoint: Url,
    state: FixtureState,
    task: JoinHandle<()>,
}

impl Drop for Fixture {
    fn drop(&mut self) {
        self.task.abort();
    }
}

impl Fixture {
    async fn start(pages: Vec<Vec<Value>>, call_result: Value, call_delay: Duration) -> Self {
        let state = FixtureState {
            pages: Arc::new(pages),
            call_result: Arc::new(call_result),
            call_delay,
            methods: Arc::new(Mutex::new(Vec::new())),
            authorization: Arc::new(Mutex::new(Vec::new())),
            call_count: Arc::new(AtomicUsize::new(0)),
        };
        let app = Router::new()
            .route("/mcp", post(handle_mcp))
            .with_state(state.clone());
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let task = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        Self {
            endpoint: Url::parse(&format!("http://{address}/mcp")).unwrap(),
            state,
            task,
        }
    }
}

async fn handle_mcp(
    State(state): State<FixtureState>,
    headers: HeaderMap,
    Json(request): Json<Value>,
) -> Response {
    let method = request
        .get("method")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned();
    state.methods.lock().unwrap().push(method.clone());
    state.authorization.lock().unwrap().push(
        headers
            .get("authorization")
            .and_then(|value| value.to_str().ok())
            .map(str::to_owned),
    );
    let id = request.get("id").cloned().unwrap_or(Value::Null);
    let result = match method.as_str() {
        "server/discover" => json!({
            "resultType": "complete",
            "supportedVersions": ["2026-07-28"],
            "capabilities": {"tools": {}},
            "ttlMs": 0,
            "cacheScope": "private"
        }),
        "tools/list" => {
            let cursor = request.pointer("/params/cursor").and_then(Value::as_str);
            let page = cursor
                .and_then(|cursor| cursor.strip_prefix("page-"))
                .and_then(|page| page.parse::<usize>().ok())
                .unwrap_or(0);
            let Some(tools) = state.pages.get(page) else {
                return rpc_error(id, -32602, "invalid fixture cursor");
            };
            let next_cursor = (page + 1 < state.pages.len()).then(|| format!("page-{}", page + 1));
            json!({
                "resultType": "complete",
                "tools": tools,
                "nextCursor": next_cursor,
                "ttlMs": 0,
                "cacheScope": "private"
            })
        }
        "tools/call" => {
            state.call_count.fetch_add(1, Ordering::SeqCst);
            tokio::time::sleep(state.call_delay).await;
            state.call_result.as_ref().clone()
        }
        _ => return rpc_error(id, -32601, "method not found"),
    };
    (
        StatusCode::OK,
        Json(json!({"jsonrpc": "2.0", "id": id, "result": result})),
    )
        .into_response()
}

fn rpc_error(id: Value, code: i64, message: &str) -> Response {
    (
        StatusCode::OK,
        Json(json!({
            "jsonrpc": "2.0",
            "id": id,
            "error": {"code": code, "message": message}
        })),
    )
        .into_response()
}

fn schema() -> Value {
    json!({
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
        "additionalProperties": false
    })
}

fn advertised_tool(name: &str, input_schema: Value) -> Value {
    json!({
        "name": name,
        "description": "Fixture search",
        "inputSchema": input_schema
    })
}

fn definition(timeout: Duration, max_output_bytes: usize) -> McpToolDefinition {
    McpToolDefinition::new("search", "remote_search", "Fixture search", schema())
        .unwrap()
        .with_limits(timeout, max_output_bytes)
        .unwrap()
}

fn request() -> ToolExecutionRequest {
    ToolExecutionRequest {
        response_id: ResponseId::from("resp-1"),
        call_id: "call-1".to_owned(),
        connector: "mcp".to_owned(),
        operation: "remote_search".to_owned(),
        profile: "fixture".to_owned(),
        arguments: json!({"query": "dynamo"}),
        scope: AuthorizationScope {
            tenant_id: "tenant-a".to_owned(),
            principal_id: "principal-a".to_owned(),
        },
        idempotency_key: IdempotencyKey::from("tool-key-1"),
        attempt: 0,
    }
}

fn make_executor(
    fixture: &Fixture,
    definition: McpToolDefinition,
) -> Result<McpToolExecutor, dynamo_agent_rt_mcp::McpConfigError> {
    let config = McpClientConfig::new_for_loopback(fixture.endpoint.clone(), [definition])?
        .with_bearer_token(McpBearerToken::new("fixture-secret")?);
    McpToolExecutor::new("fixture", config)
}

#[tokio::test]
async fn calls_allowlisted_tool_with_auth_and_preserves_completed_error() {
    let fixture = Fixture::start(
        vec![vec![advertised_tool("remote_search", schema())]],
        json!({
            "resultType": "complete",
            "content": [{"type": "text", "text": "denied"}],
            "structuredContent": {"reason": "denied"},
            "isError": true
        }),
        Duration::ZERO,
    )
    .await;
    let executor = make_executor(&fixture, definition(Duration::from_secs(1), 1024)).unwrap();

    let result = executor.execute(request()).await.unwrap();

    assert_eq!(result.output, json!({"reason": "denied"}));
    assert!(result.is_error);
    assert_eq!(
        fixture.state.methods.lock().unwrap().as_slice(),
        ["server/discover", "tools/list", "tools/call"]
    );
    assert!(
        fixture
            .state
            .authorization
            .lock()
            .unwrap()
            .iter()
            .all(|header| header.as_deref() == Some("Bearer fixture-secret"))
    );
}

#[tokio::test]
async fn verifies_a_paginated_descriptor_once_for_reused_client() {
    let fixture = Fixture::start(
        vec![
            vec![advertised_tool("unrelated", schema())],
            vec![advertised_tool("remote_search", schema())],
        ],
        json!({
            "resultType": "complete",
            "content": [{"type": "text", "text": "ok"}],
            "isError": false
        }),
        Duration::ZERO,
    )
    .await;
    let executor = make_executor(&fixture, definition(Duration::from_secs(1), 1024)).unwrap();

    executor.execute(request()).await.unwrap();
    executor.execute(request()).await.unwrap();

    let methods = fixture.state.methods.lock().unwrap();
    assert_eq!(
        methods
            .iter()
            .filter(|method| *method == "server/discover")
            .count(),
        1
    );
    assert_eq!(
        methods
            .iter()
            .filter(|method| *method == "tools/list")
            .count(),
        2
    );
    assert_eq!(
        methods
            .iter()
            .filter(|method| *method == "tools/call")
            .count(),
        2
    );
}

#[tokio::test]
async fn schema_mismatch_fails_before_call_dispatch() {
    let fixture = Fixture::start(
        vec![vec![advertised_tool(
            "remote_search",
            json!({
                "type": "object",
                "properties": {"q": {"type": "string"}},
                "additionalProperties": false
            }),
        )]],
        json!({"resultType": "complete", "content": []}),
        Duration::ZERO,
    )
    .await;
    let executor = make_executor(&fixture, definition(Duration::from_secs(1), 1024)).unwrap();

    let error = executor.execute(request()).await.unwrap_err();

    assert!(matches!(error, McpToolExecutorError::SchemaMismatch(name) if name == "remote_search"));
    assert_eq!(fixture.state.call_count.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn executor_timeout_is_bounded_and_read_only_lookup_reexecutes() {
    let fixture = Fixture::start(
        vec![vec![advertised_tool("remote_search", schema())]],
        json!({
            "resultType": "complete",
            "content": [{"type": "text", "text": "ok"}]
        }),
        Duration::from_millis(100),
    )
    .await;
    let executor = make_executor(&fixture, definition(Duration::from_millis(20), 1024)).unwrap();

    let error = executor.execute(request()).await.unwrap_err();
    assert!(matches!(error, McpToolExecutorError::Timeout));

    let fixture = Fixture::start(
        vec![vec![advertised_tool("remote_search", schema())]],
        json!({
            "resultType": "complete",
            "content": [{"type": "text", "text": "recovered"}]
        }),
        Duration::ZERO,
    )
    .await;
    let executor = make_executor(&fixture, definition(Duration::from_secs(1), 1024)).unwrap();
    let request = request();
    let result = executor.lookup(&request).await.unwrap().unwrap();
    assert_eq!(result.output, json!("recovered"));
    assert_eq!(fixture.state.call_count.load(Ordering::SeqCst), 1);
}
