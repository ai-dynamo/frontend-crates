// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Based on https://github.com/64bit/async-openai/ by Himanshu Neema
// Original Copyright (c) 2022 Himanshu Neema
// Licensed under MIT License (see ATTRIBUTIONS-Rust.md)
//
// Modifications Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
// Licensed under Apache 2.0

use dynamo_protocols::types::{
    ChatCompletionRequestAssistantMessage, ChatCompletionRequestMessage,
    ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestUserMessageArgs,
    CreateChatCompletionRequest, CreateChatCompletionRequestArgs, ReasoningContent,
};

#[tokio::test]
async fn chat_types_serde() {
    let request: CreateChatCompletionRequest = CreateChatCompletionRequestArgs::default()
        .messages([
            ChatCompletionRequestSystemMessageArgs::default()
                .content("your are a calculator")
                .build()
                .unwrap()
                .into(),
            ChatCompletionRequestUserMessageArgs::default()
                .content("what is the result of 1+1")
                .build()
                .unwrap()
                .into(),
        ])
        .build()
        .unwrap();
    // serialize the request
    let serialized = serde_json::to_string(&request).unwrap();
    // deserialize the request
    let deserialized: CreateChatCompletionRequest = serde_json::from_str(&serialized).unwrap();
    assert_eq!(request, deserialized);
}

#[test]
fn assistant_reasoning_content_accepts_canonical_and_alias_fields() {
    let cases = [
        (
            r#"{"reasoning_content":"thinking"}"#,
            ReasoningContent::Text("thinking".to_string()),
        ),
        (
            r#"{"reasoning":"thinking"}"#,
            ReasoningContent::Text("thinking".to_string()),
        ),
        (
            r#"{"reasoning_content":["before","after"]}"#,
            ReasoningContent::Segments(vec!["before".to_string(), "after".to_string()]),
        ),
        (
            r#"{"reasoning":["before","after"]}"#,
            ReasoningContent::Segments(vec!["before".to_string(), "after".to_string()]),
        ),
    ];

    for (json, expected) in cases {
        let message: ChatCompletionRequestAssistantMessage = serde_json::from_str(json).unwrap();
        assert_eq!(message.reasoning_content, Some(expected));
    }
}

#[test]
fn chat_request_accepts_reasoning_alias_for_assistant_message() {
    let request: CreateChatCompletionRequest = serde_json::from_value(serde_json::json!({
        "messages": [{
            "role": "assistant",
            "content": null,
            "reasoning": "thinking"
        }],
        "model": "test-model"
    }))
    .unwrap();

    let [ChatCompletionRequestMessage::Assistant(message)] = request.messages.as_slice() else {
        panic!("expected one assistant message");
    };
    assert_eq!(
        message.reasoning_content,
        Some(ReasoningContent::Text("thinking".to_string()))
    );
}

#[test]
fn assistant_reasoning_content_serializes_with_canonical_field() {
    let message: ChatCompletionRequestAssistantMessage =
        serde_json::from_str(r#"{"reasoning":["before","after"]}"#).unwrap();

    assert_eq!(
        serde_json::to_value(message).unwrap(),
        serde_json::json!({"reasoning_content": ["before", "after"]})
    );
}

#[test]
fn assistant_reasoning_content_rejects_both_field_names() {
    let error = serde_json::from_str::<ChatCompletionRequestAssistantMessage>(
        r#"{"reasoning_content":"canonical","reasoning":"alias"}"#,
    )
    .unwrap_err();

    assert!(
        error
            .to_string()
            .contains("duplicate field `reasoning_content`"),
        "unexpected error: {error}"
    );
}
