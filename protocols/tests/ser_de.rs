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
    ChatCompletionToolType, CreateChatCompletionRequest, CreateChatCompletionRequestArgs,
    ThinkingConfig, ThinkingObject, ThinkingType,
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

/// Kimi Code CLI sends `prompt_cache_key` on every request; Moonshot's hosted
/// `$web_search` is declared with `type: "builtin_function"`.
#[test]
fn prompt_cache_key_and_builtin_function_round_trip() {
    let body = json!({
        "model": "kimi-k3",
        "prompt_cache_key": "sess_8f3a",
        "safety_identifier": "user_42",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{
            "type": "builtin_function",
            "function": {"name": "$web_search"}
        }]
    });

    let request: CreateChatCompletionRequest = serde_json::from_value(body).unwrap();
    assert_eq!(request.prompt_cache_key.as_deref(), Some("sess_8f3a"));
    assert_eq!(request.safety_identifier.as_deref(), Some("user_42"));
    assert_eq!(
        request.tools.as_ref().unwrap()[0].r#type,
        ChatCompletionToolType::BuiltinFunction
    );

    let serialized = serde_json::to_value(&request).unwrap();
    assert_eq!(serialized["tools"][0]["type"], json!("builtin_function"));
    assert_eq!(serialized["prompt_cache_key"], json!("sess_8f3a"));

    let round_trip: CreateChatCompletionRequest = serde_json::from_value(serialized).unwrap();
    assert_eq!(request, round_trip);

    let plain: CreateChatCompletionRequest = serde_json::from_value(json!({
        "model": "kimi-k3",
        "messages": [{"role": "user", "content": "hi"}]
    }))
    .unwrap();
    let plain_json = serde_json::to_value(&plain).unwrap();
    assert!(plain_json.get("prompt_cache_key").is_none());
    assert!(plain_json.get("safety_identifier").is_none());
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

/// Kimi Code sends `thinking: {type, effort, keep}`. Only `type` is kept for
/// now; `effort` / `keep` are accepted and dropped (see `ThinkingObject`).
#[test]
fn thinking_accepts_object_and_bool_and_drops_moonshot_sub_fields() {
    let request: CreateChatCompletionRequest = serde_json::from_value(json!({
        "model": "kimi-k3",
        "messages": [{"role": "user", "content": "hi"}],
        "thinking": {"type": "enabled", "effort": "high", "keep": "all"}
    }))
    .unwrap();
    assert_eq!(
        request.thinking,
        Some(ThinkingConfig::Object(ThinkingObject {
            r#type: Some(ThinkingType::Enabled),
        }))
    );
    assert_eq!(request.thinking.as_ref().unwrap().enabled(), Some(true));
    let serialized = serde_json::to_value(&request).unwrap();
    assert_eq!(serialized["thinking"], json!({"type": "enabled"}));

    let as_bool: CreateChatCompletionRequest = serde_json::from_value(json!({
        "model": "kimi-k3",
        "messages": [{"role": "user", "content": "hi"}],
        "thinking": false
    }))
    .unwrap();
    assert_eq!(as_bool.thinking, Some(ThinkingConfig::Enabled(false)));
    assert_eq!(as_bool.thinking.as_ref().unwrap().enabled(), Some(false));

    let adaptive: CreateChatCompletionRequest = serde_json::from_value(json!({
        "model": "MiniMax-M3",
        "messages": [{"role": "user", "content": "hi"}],
        "thinking": {"type": "adaptive"}
    }))
    .unwrap();
    assert_eq!(adaptive.thinking.as_ref().unwrap().enabled(), None);

    let absent: CreateChatCompletionRequest = serde_json::from_value(json!({
        "model": "kimi-k3",
        "messages": [{"role": "user", "content": "hi"}]
    }))
    .unwrap();
    assert!(
        serde_json::to_value(&absent)
            .unwrap()
            .get("thinking")
            .is_none()
    );
}
