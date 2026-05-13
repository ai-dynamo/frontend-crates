//! POST /v1/messages -- Anthropic Messages API
//!
//! Supports both streaming (SSE) and non-streaming responses.
//! Anthropic SSE uses event type fields (message_start, content_block_delta, etc.)

use std::convert::Infallible;

use axum::Json;
use axum::response::{IntoResponse, Response, sse::{Event, KeepAlive, Sse}};
use futures::stream::Stream;

use dynamo_protocols::types::anthropic::*;

use crate::echo;

pub async fn handler(Json(req): Json<AnthropicCreateMessageRequest>) -> Response {
    let model = req.model.clone();
    let text = echo::extract_anthropic_text(&req.messages);
    let echo_text = format!("[echo] {text}");
    let id = format!("msg_{}", uuid::Uuid::new_v4().simple());

    if req.stream {
        let stream = make_stream(id, model, echo_text);
        Sse::new(stream).keep_alive(KeepAlive::default()).into_response()
    } else {
        let resp = AnthropicMessageResponse {
            id,
            object_type: "message".to_string(),
            role: "assistant".to_string(),
            content: vec![AnthropicResponseContentBlock::Text {
                text: echo_text.clone(),
                citations: None,
            }],
            model,
            stop_reason: Some(AnthropicStopReason::EndTurn),
            stop_sequence: None,
            usage: AnthropicUsage {
                input_tokens: text.len() as u32 / 4,
                output_tokens: echo_text.len() as u32 / 4,
                cache_creation_input_tokens: None,
                cache_read_input_tokens: None,
            },
        };
        Json(resp).into_response()
    }
}

fn make_stream(
    id: String,
    model: String,
    text: String,
) -> impl Stream<Item = Result<Event, Infallible>> {
    async_stream::stream! {
        // message_start
        let msg_start = AnthropicStreamEvent::MessageStart {
            message: AnthropicMessageResponse {
                id: id.clone(),
                object_type: "message".to_string(),
                role: "assistant".to_string(),
                content: vec![],
                model: model.clone(),
                stop_reason: None,
                stop_sequence: None,
                usage: AnthropicUsage {
                    input_tokens: text.len() as u32 / 4,
                    output_tokens: 0,
                    cache_creation_input_tokens: None,
                    cache_read_input_tokens: None,
                },
            },
        };
        yield Ok(Event::default()
            .event("message_start")
            .data(serde_json::to_string(&msg_start).unwrap()));

        // content_block_start
        let block_start = AnthropicStreamEvent::ContentBlockStart {
            index: 0,
            content_block: AnthropicResponseContentBlock::Text {
                text: String::new(),
                citations: None,
            },
        };
        yield Ok(Event::default()
            .event("content_block_start")
            .data(serde_json::to_string(&block_start).unwrap()));

        // content_block_delta -- stream word by word
        for word in text.split_inclusive(' ') {
            let delta = AnthropicStreamEvent::ContentBlockDelta {
                index: 0,
                delta: AnthropicDelta::TextDelta {
                    text: word.to_string(),
                },
            };
            yield Ok(Event::default()
                .event("content_block_delta")
                .data(serde_json::to_string(&delta).unwrap()));
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }

        // content_block_stop
        let block_stop = AnthropicStreamEvent::ContentBlockStop { index: 0 };
        yield Ok(Event::default()
            .event("content_block_stop")
            .data(serde_json::to_string(&block_stop).unwrap()));

        // message_delta
        let msg_delta = AnthropicStreamEvent::MessageDelta {
            delta: AnthropicMessageDeltaBody {
                stop_reason: Some(AnthropicStopReason::EndTurn),
                stop_sequence: None,
            },
            usage: AnthropicUsage {
                input_tokens: 0,
                output_tokens: text.len() as u32 / 4,
                cache_creation_input_tokens: None,
                cache_read_input_tokens: None,
            },
        };
        yield Ok(Event::default()
            .event("message_delta")
            .data(serde_json::to_string(&msg_delta).unwrap()));

        // message_stop
        let msg_stop = AnthropicStreamEvent::MessageStop {};
        yield Ok(Event::default()
            .event("message_stop")
            .data(serde_json::to_string(&msg_stop).unwrap()));
    }
}
