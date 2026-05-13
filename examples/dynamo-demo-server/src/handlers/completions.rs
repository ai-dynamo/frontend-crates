//! POST /v1/completions -- OpenAI Completions API
//!
//! Supports both streaming (SSE) and non-streaming responses.

use std::convert::Infallible;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response, sse::{Event, KeepAlive, Sse}};
use futures::stream::Stream;

use dynamo_protocols::types::*;

use crate::echo;
use crate::engine::AppState;

pub async fn handler(
    State(state): State<AppState>,
    Json(req): Json<CreateCompletionRequest>,
) -> Response {
    let model = req.model.clone();
    let text = match echo::extract_completion_text(&req.prompt) {
        Ok(t) => t,
        Err(e) => return (StatusCode::BAD_REQUEST, e).into_response(),
    };

    let id = format!("cmpl-{}", uuid::Uuid::new_v4());
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs() as u32;

    if req.stream == Some(true) {
        let stream = make_stream(id, model, created, text);
        Sse::new(stream).keep_alive(KeepAlive::default()).into_response()
    } else {
        let echo_text = format!("[echo] {text}");
        let resp = CreateCompletionResponse {
            id,
            object: "text_completion".to_string(),
            created,
            model,
            choices: vec![Choice {
                index: 0,
                text: echo_text.clone(),
                finish_reason: Some(CompletionFinishReason::Stop),
                logprobs: None,
            }],
            usage: Some({
                let prompt_tokens = state.count_tokens(&text);
                let completion_tokens = state.count_tokens(&echo_text);
                CompletionUsage {
                    prompt_tokens,
                    completion_tokens,
                    total_tokens: prompt_tokens + completion_tokens,
                    prompt_tokens_details: None,
                    completion_tokens_details: None,
                }
            }),
            system_fingerprint: None,
        };
        Json(resp).into_response()
    }
}

fn make_stream(
    id: String,
    model: String,
    created: u32,
    text: String,
) -> impl Stream<Item = Result<Event, Infallible>> {
    async_stream::stream! {
        let echo_text = format!("[echo] {text}");
        for word in echo_text.split_inclusive(' ') {
            let chunk = CreateCompletionResponse {
                id: id.clone(),
                object: "text_completion".to_string(),
                created,
                model: model.clone(),
                choices: vec![Choice {
                    index: 0,
                    text: word.to_string(),
                    finish_reason: None,
                    logprobs: None,
                }],
                usage: None,
                system_fingerprint: None,
            };
            yield Ok(Event::default().data(serde_json::to_string(&chunk).unwrap()));
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }

        // Final chunk
        let final_chunk = CreateCompletionResponse {
            id: id.clone(),
            object: "text_completion".to_string(),
            created,
            model: model.clone(),
            choices: vec![Choice {
                index: 0,
                text: String::new(),
                finish_reason: Some(CompletionFinishReason::Stop),
                logprobs: None,
            }],
            usage: None,
            system_fingerprint: None,
        };
        yield Ok(Event::default().data(serde_json::to_string(&final_chunk).unwrap()));
        yield Ok(Event::default().data("[DONE]"));
    }
}
