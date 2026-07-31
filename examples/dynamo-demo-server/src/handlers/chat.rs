// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! POST /v1/chat/completions -- OpenAI Chat Completions API
//!
//! Supports both streaming (SSE) and non-streaming responses.
//! When tools are provided, demonstrates inline tool-call parsing via dynamo-parsers.

use std::convert::Infallible;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::Json;
use axum::extract::State;
use axum::response::{
    IntoResponse, Response,
    sse::{Event, KeepAlive, Sse},
};
use futures::stream::Stream;

use dynamo_protocols::types::*;

use crate::echo;
use crate::engine::AppState;

pub async fn handler(
    State(state): State<AppState>,
    Json(req): Json<CreateChatCompletionRequest>,
) -> Response {
    let model = req.model.clone();
    let text = echo::extract_chat_text(&req.messages);
    let id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs() as u32;

    // If tools are provided, generate a synthetic tool call and parse it
    let tool_call_result = if let Some(tools) = &req.tools {
        if let Some(first_tool) = tools.first() {
            let tool_name = &first_tool.function.name;
            // Produce a synthetic tool-call block that dynamo-parsers can parse
            let synthetic = format!(
                r#"<tool_call>{{"name": "{tool_name}", "arguments": {{"input": "{text}"}}}}</tool_call>"#
            );
            let tool_defs: Vec<dynamo_parsers::tool_calling::ToolDefinition> = tools
                .iter()
                .map(|t| dynamo_parsers::tool_calling::ToolDefinition {
                    name: t.function.name.clone(),
                    parameters: t.function.parameters.clone(),
                    strict: None,
                })
                .collect();
            match dynamo_parsers::tool_calling::detect_and_parse_tool_call(
                &synthetic,
                Some("hermes"),
                Some(&tool_defs),
            )
            .await
            {
                Ok((calls, _content)) if !calls.is_empty() => Some(calls),
                _ => None,
            }
        } else {
            None
        }
    } else {
        None
    };

    if req.stream == Some(true) {
        let stream = make_stream(id, model, created, text, tool_call_result);
        Sse::new(stream)
            .keep_alive(KeepAlive::default())
            .into_response()
    } else {
        let tool_calls = tool_call_result.map(|calls| {
            calls
                .into_iter()
                .map(|tc| ChatCompletionMessageToolCall {
                    id: tc.id,
                    r#type: FunctionType::Function,
                    function: FunctionCall {
                        name: tc.function.name,
                        arguments: tc.function.arguments,
                    },
                })
                .collect::<Vec<_>>()
        });

        let (content, finish) = if tool_calls.is_some() {
            (None, Some(FinishReason::ToolCalls))
        } else {
            (
                Some(ChatCompletionMessageContent::Text(format!("[echo] {text}"))),
                Some(FinishReason::Stop),
            )
        };

        #[allow(deprecated)]
        let resp = CreateChatCompletionResponse {
            id,
            object: "chat.completion".to_string(),
            created,
            model,
            choices: vec![ChatChoice {
                index: 0,
                message: ChatCompletionResponseMessage {
                    role: Role::Assistant,
                    content,
                    tool_calls,
                    refusal: None,
                    function_call: None,
                    audio: None,
                    reasoning_content: None,
                },
                finish_reason: finish,
                logprobs: None,
            }],
            usage: Some({
                let echo_text = format!("[echo] {text}");
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
            service_tier: None,
        };
        Json(resp).into_response()
    }
}

fn make_stream(
    id: String,
    model: String,
    created: u32,
    text: String,
    tool_calls: Option<Vec<dynamo_parsers::tool_calling::ToolCallResponse>>,
) -> impl Stream<Item = Result<Event, Infallible>> {
    async_stream::stream! {
        // First chunk: role
        let first = make_stream_chunk(&id, &model, created, None, Some(Role::Assistant), None, None);
        yield Ok(Event::default().data(serde_json::to_string(&first).unwrap()));

        if let Some(calls) = tool_calls {
            // Stream tool calls
            for (i, tc) in calls.into_iter().enumerate() {
                let chunk_tc = ChatCompletionMessageToolCallChunk {
                    index: i as u32,
                    id: Some(tc.id),
                    r#type: Some(FunctionType::Function),
                    function: Some(FunctionCallStream {
                        name: Some(tc.function.name),
                        arguments: Some(tc.function.arguments),
                    }),
                };
                let chunk = make_stream_chunk_with_tool_call(&id, &model, created, vec![chunk_tc], None);
                yield Ok(Event::default().data(serde_json::to_string(&chunk).unwrap()));
            }
            // Final chunk with finish_reason
            let final_chunk = make_stream_chunk(&id, &model, created, None, None, Some(FinishReason::ToolCalls), None);
            yield Ok(Event::default().data(serde_json::to_string(&final_chunk).unwrap()));
        } else {
            // Stream content tokens (word by word for demo)
            let echo_text = format!("[echo] {text}");
            for word in echo_text.split_inclusive(' ') {
                let chunk = make_stream_chunk(
                    &id, &model, created,
                    Some(ChatCompletionMessageContent::Text(word.to_string())),
                    None, None, None,
                );
                yield Ok(Event::default().data(serde_json::to_string(&chunk).unwrap()));
                tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            }
            // Final chunk with finish_reason
            let final_chunk = make_stream_chunk(&id, &model, created, None, None, Some(FinishReason::Stop), None);
            yield Ok(Event::default().data(serde_json::to_string(&final_chunk).unwrap()));
        }

        // Sentinel
        yield Ok(Event::default().data("[DONE]"));
    }
}

fn make_stream_chunk(
    id: &str,
    model: &str,
    created: u32,
    content: Option<ChatCompletionMessageContent>,
    role: Option<Role>,
    finish_reason: Option<FinishReason>,
    usage: Option<CompletionUsage>,
) -> CreateChatCompletionStreamResponse {
    CreateChatCompletionStreamResponse {
        id: id.to_string(),
        object: "chat.completion.chunk".to_string(),
        created,
        model: model.to_string(),
        choices: vec![ChatChoiceStream {
            index: 0,
            delta: ChatCompletionStreamResponseDelta {
                content,
                role,
                tool_calls: None,
                function_call: None,
                refusal: None,
                reasoning_content: None,
            },
            finish_reason,
            logprobs: None,
        }],
        usage,
        system_fingerprint: None,
        service_tier: None,
    }
}

fn make_stream_chunk_with_tool_call(
    id: &str,
    model: &str,
    created: u32,
    tool_calls: Vec<ChatCompletionMessageToolCallChunk>,
    finish_reason: Option<FinishReason>,
) -> CreateChatCompletionStreamResponse {
    CreateChatCompletionStreamResponse {
        id: id.to_string(),
        object: "chat.completion.chunk".to_string(),
        created,
        model: model.to_string(),
        choices: vec![ChatChoiceStream {
            index: 0,
            delta: ChatCompletionStreamResponseDelta {
                content: None,
                role: None,
                tool_calls: Some(tool_calls),
                function_call: None,
                refusal: None,
                reasoning_content: None,
            },
            finish_reason,
            logprobs: None,
        }],
        usage: None,
        system_fingerprint: None,
        service_tier: None,
    }
}
