// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! POST /v1/render -- Dedicated dynamo-renderer demo endpoint
//!
//! Accepts a standard OpenAI chat-completion request body and returns the
//! prompt string produced by applying the model's chat template to it. This is
//! NOT part of any standard API -- it exists to showcase dynamo-renderer's
//! `apply_chat_template` rendering (the *encode* side of serving).
//!
//! Requires the server to have been started with a `tokenizer_config.json`
//! carrying a `chat_template` (via `--model`, or `--chat-template-config`).

use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::Serialize;

use dynamo_protocols::types::CreateChatCompletionRequest;
use dynamo_renderer::PromptFormatter;

use crate::engine::AppState;

#[derive(Serialize)]
pub struct RenderResponse {
    /// The fully-rendered prompt string ready to feed a tokenizer / engine.
    pub prompt: String,
    pub model: String,
}

pub async fn handler(
    State(state): State<AppState>,
    Json(req): Json<CreateChatCompletionRequest>,
) -> Response {
    let Some(PromptFormatter::OAI(formatter)) = state.formatter else {
        let body = serde_json::json!({
            "error": "no chat template loaded; start the server with --model or \
                      --chat-template-config pointing at a tokenizer_config.json \
                      that carries a chat_template",
        });
        return (StatusCode::SERVICE_UNAVAILABLE, Json(body)).into_response();
    };

    let model = req.model.clone();
    match formatter.render(&req) {
        Ok(prompt) => Json(RenderResponse { prompt, model }).into_response(),
        Err(e) => {
            let body = serde_json::json!({ "error": e.to_string() });
            (StatusCode::BAD_REQUEST, Json(body)).into_response()
        }
    }
}
