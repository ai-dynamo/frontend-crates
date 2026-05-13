//! POST /v1/tool-parse -- Dedicated dynamo-parsers demo endpoint
//!
//! Accepts raw text + optional parser name + tool definitions, returns parsed tool calls.
//! This is NOT part of any standard API -- it exists to showcase dynamo-parsers directly.

use axum::Json;
use axum::response::{IntoResponse, Response};
use axum::http::StatusCode;
use serde::{Deserialize, Serialize};

use dynamo_parsers::tool_calling::{
    ToolDefinition, detect_and_parse_tool_call,
    parsers::get_available_tool_parsers,
};

#[derive(Deserialize)]
pub struct ToolParseRequest {
    /// The raw text that may contain tool calls
    pub text: String,
    /// Parser name (e.g. "hermes", "llama3_json", "mistral"). Defaults to "default".
    pub parser: Option<String>,
    /// Tool definitions to validate against
    pub tools: Option<Vec<ToolDef>>,
}

#[derive(Deserialize)]
pub struct ToolDef {
    pub name: String,
    pub parameters: Option<serde_json::Value>,
}

#[derive(Serialize)]
pub struct ToolParseResponse {
    pub tool_calls: Vec<ParsedToolCall>,
    pub remaining_content: Option<String>,
    pub available_parsers: Vec<String>,
}

#[derive(Serialize)]
pub struct ParsedToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub tp: String,
    pub function: ParsedFunction,
}

#[derive(Serialize)]
pub struct ParsedFunction {
    pub name: String,
    pub arguments: String,
}

pub async fn handler(Json(req): Json<ToolParseRequest>) -> Response {
    let tool_defs: Option<Vec<ToolDefinition>> = req.tools.map(|tools| {
        tools
            .into_iter()
            .map(|t| ToolDefinition {
                name: t.name,
                parameters: t.parameters,
            })
            .collect()
    });

    let parser_str = req.parser.as_deref();
    let tools_ref = tool_defs.as_deref();

    match detect_and_parse_tool_call(&req.text, parser_str, tools_ref).await {
        Ok((calls, remaining)) => {
            let parsed: Vec<ParsedToolCall> = calls
                .into_iter()
                .map(|tc| ParsedToolCall {
                    id: tc.id,
                    tp: "function".to_string(),
                    function: ParsedFunction {
                        name: tc.function.name,
                        arguments: tc.function.arguments,
                    },
                })
                .collect();

            let resp = ToolParseResponse {
                tool_calls: parsed,
                remaining_content: remaining,
                available_parsers: get_available_tool_parsers()
                    .into_iter()
                    .map(String::from)
                    .collect(),
            };
            Json(resp).into_response()
        }
        Err(e) => {
            let body = serde_json::json!({
                "error": e.to_string(),
                "available_parsers": get_available_tool_parsers(),
            });
            (StatusCode::BAD_REQUEST, Json(body)).into_response()
        }
    }
}
