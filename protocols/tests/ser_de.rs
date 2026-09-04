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
    ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestUserMessageArgs,
    CreateChatCompletionRequest, CreateChatCompletionRequestArgs,
};
use serde_json::json;

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

/// Kimi Code CLI sends `prompt_cache_key` on every request. It is preserved
/// verbatim and omitted when absent; nothing acts on it yet.
#[test]
fn prompt_cache_key_round_trips_and_is_omitted_when_absent() {
    let body = json!({
        "model": "kimi-k3",
        "prompt_cache_key": "sess_8f3a",
        "messages": [{"role": "user", "content": "hi"}]
    });

    let request: CreateChatCompletionRequest = serde_json::from_value(body).unwrap();
    assert_eq!(request.prompt_cache_key.as_deref(), Some("sess_8f3a"));

    let serialized = serde_json::to_value(&request).unwrap();
    assert_eq!(serialized["prompt_cache_key"], json!("sess_8f3a"));
    let round_trip: CreateChatCompletionRequest = serde_json::from_value(serialized).unwrap();
    assert_eq!(request, round_trip);

    let plain: CreateChatCompletionRequest = serde_json::from_value(json!({
        "model": "kimi-k3",
        "messages": [{"role": "user", "content": "hi"}]
    }))
    .unwrap();
    assert!(
        serde_json::to_value(&plain)
            .unwrap()
            .get("prompt_cache_key")
            .is_none()
    );
}

/// Dynamic tool system messages must carry a list; scalars and objects are
/// rejected at deserialization rather than reaching a renderer.
#[test]
fn system_message_tools_must_be_an_array() {
    for bad in [json!("lookup"), json!({"name": "lookup"}), json!(1)] {
        let body = json!({
            "model": "kimi-k3",
            "messages": [{"role": "system", "tools": bad}]
        });
        assert!(
            serde_json::from_value::<CreateChatCompletionRequest>(body).is_err(),
            "tools={bad} must not deserialize"
        );
    }
}
