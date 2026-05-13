//! POST /v1/tokenize, /v1/detokenize -- dynamo-tokenizers demo endpoints
//!
//! Round-trip text <-> token IDs using the tokenizer loaded at startup.

use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::{Deserialize, Serialize};

use crate::engine::AppState;

#[derive(Deserialize)]
pub struct TokenizeRequest {
    pub text: String,
    #[serde(default)]
    pub add_special_tokens: bool,
}

#[derive(Serialize)]
pub struct TokenizeResponse {
    pub token_ids: Vec<u32>,
    pub count: usize,
}

#[derive(Deserialize)]
pub struct DetokenizeRequest {
    pub token_ids: Vec<u32>,
    #[serde(default)]
    pub skip_special_tokens: bool,
}

#[derive(Serialize)]
pub struct DetokenizeResponse {
    pub text: String,
}

pub async fn encode(State(state): State<AppState>, Json(req): Json<TokenizeRequest>) -> Response {
    // `add_special_tokens` is accepted for API compat but the tokenizer's default behavior is used.
    let _ = req.add_special_tokens;
    match state.encode(&req.text) {
        Some(ids) => Json(TokenizeResponse {
            count: ids.len(),
            token_ids: ids,
        })
        .into_response(),
        None => no_tokenizer(),
    }
}

pub async fn decode(State(state): State<AppState>, Json(req): Json<DetokenizeRequest>) -> Response {
    match state.decode(&req.token_ids, req.skip_special_tokens) {
        Some(text) => Json(DetokenizeResponse { text }).into_response(),
        None => no_tokenizer(),
    }
}

fn no_tokenizer() -> Response {
    let body = serde_json::json!({
        "error": "no tokenizer loaded",
        "hint": "set TOKENIZER_PATH=/path/to/tokenizer.json before starting the server",
    });
    (StatusCode::SERVICE_UNAVAILABLE, Json(body)).into_response()
}
