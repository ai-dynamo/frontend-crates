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
    ChatCompletionRequestMessage, ChatCompletionRequestSystemMessageArgs,
    ChatCompletionRequestSystemMessageContent, ChatCompletionRequestUserMessageArgs,
    ChatCompletionToolType, CreateChatCompletionRequest, CreateChatCompletionRequestArgs,
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

/// Kimi Code CLI wire shape: `prompt_cache_key` on every request, a content-less
/// system message carrying dynamically loaded tools, a `builtin_function` tool,
/// and a `partial` assistant prefill.
#[test]
fn moonshot_kimi_request_fields_round_trip() {
    let body = json!({
        "model": "kimi-k3",
        "prompt_cache_key": "sess_8f3a",
        "safety_identifier": "user_42",
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "system", "tools": [{
                "type": "function",
                "function": {"name": "read_file", "parameters": {"type": "object"}}
            }]},
            {"role": "user", "content": "```json\n"},
            {"role": "assistant", "content": "{\"answer\":", "partial": true}
        ],
        "tools": [{
            "type": "builtin_function",
            "function": {"name": "$web_search"}
        }]
    });

    let request: CreateChatCompletionRequest = serde_json::from_value(body.clone()).unwrap();
    assert_eq!(request.prompt_cache_key.as_deref(), Some("sess_8f3a"));
    assert_eq!(request.safety_identifier.as_deref(), Some("user_42"));

    let ChatCompletionRequestMessage::System(dynamic) = &request.messages[1] else {
        panic!("expected system message");
    };
    assert_eq!(
        dynamic.content,
        ChatCompletionRequestSystemMessageContent::Text(String::new())
    );
    assert_eq!(dynamic.tools.as_ref().map(Vec::len), Some(1));

    let ChatCompletionRequestMessage::Assistant(prefill) = &request.messages[3] else {
        panic!("expected assistant message");
    };
    assert_eq!(prefill.partial, Some(true));

    let tools = request.tools.as_ref().unwrap();
    assert_eq!(tools[0].r#type, ChatCompletionToolType::BuiltinFunction);

    let serialized = serde_json::to_value(&request).unwrap();
    assert_eq!(serialized["tools"][0]["type"], json!("builtin_function"));
    assert_eq!(serialized["messages"][3]["partial"], json!(true));
    assert!(serialized["messages"][0].get("tools").is_none());
    assert!(serialized["messages"][0].get("partial").is_none());

    let round_trip: CreateChatCompletionRequest = serde_json::from_value(serialized).unwrap();
    assert_eq!(request, round_trip);
}
