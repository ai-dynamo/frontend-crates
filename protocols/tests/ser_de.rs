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
    ChatCompletionRequestToolMessageArgs, ChatCompletionRequestUserMessageArgs,
    CreateChatCompletionRequest, CreateChatCompletionRequestArgs, ImageDetail, ImageUrl,
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

#[test]
fn system_message_tools_must_be_an_array() {
    let body = json!({
        "model": "kimi-k3",
        "messages": [{"role": "system", "tools": {"name": "lookup"}}]
    });
    assert!(serde_json::from_value::<CreateChatCompletionRequest>(body).is_err());
}

#[test]
fn tool_message_name_round_trips_and_is_omitted_when_absent() {
    let body = json!({
        "model": "test",
        "messages": [{
            "role": "tool", "tool_call_id": "call_1", "name": "lookup_image", "content": "ok"
        }]
    });

    let request: CreateChatCompletionRequest = serde_json::from_value(body.clone()).unwrap();
    let ChatCompletionRequestMessage::Tool(tool) = &request.messages[0] else {
        panic!("expected a tool message");
    };
    assert_eq!(tool.name.as_deref(), Some("lookup_image"));
    assert_eq!(
        serde_json::to_value(&request).unwrap()["messages"][0],
        body["messages"][0]
    );

    let plain: CreateChatCompletionRequest = serde_json::from_value(json!({
        "model": "test",
        "messages": [{"role": "tool", "tool_call_id": "call_1", "content": "ok"}]
    }))
    .unwrap();
    let ChatCompletionRequestMessage::Tool(tool) = &plain.messages[0] else {
        panic!("expected a tool message");
    };
    assert_eq!(tool.name, None);
    assert!(
        serde_json::to_value(&plain).unwrap()["messages"][0]
            .get("name")
            .is_none()
    );

    let built = ChatCompletionRequestToolMessageArgs::default()
        .tool_call_id("call_1")
        .content("ok")
        .name("lookup_image")
        .build()
        .unwrap();
    assert_eq!(built.name.as_deref(), Some("lookup_image"));
}

#[test]
fn image_detail_accepts_every_sglang_level_in_user_and_tool_content() {
    // SGLang's request model is `Optional[Literal["auto","low","medium","high"]] = "auto"`:
    // every listed value and an explicit null must survive the typed parse unchanged.
    for detail in ["auto", "low", "medium", "high", "original"]
        .map(|level| json!(level))
        .into_iter()
        .chain([serde_json::Value::Null])
    {
        for role in ["user", "tool"] {
            let mut message = json!({
                "role": role,
                "content": [{"type": "image_url", "image_url": {
                    "url": "https://example.com/a.png", "detail": detail
                }}]
            });
            if role == "tool" {
                message["tool_call_id"] = json!("call_1");
                message["name"] = json!("lookup_image");
            }
            let body = json!({"model": "test", "messages": [message]});
            let request: CreateChatCompletionRequest =
                serde_json::from_value(body.clone()).unwrap();
            assert_eq!(
                serde_json::to_value(&request).unwrap()["messages"][0],
                body["messages"][0],
                "role={role} detail={detail}"
            );
        }
    }
    let request: CreateChatCompletionRequest = serde_json::from_value(json!({
        "model": "test", "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}
        ]}]
    }))
    .unwrap();
    assert_eq!(
        serde_json::to_value(&request).unwrap()["messages"][0]["content"][0]["image_url"]["detail"],
        "auto"
    );
    assert!(
        serde_json::from_value::<ImageUrl>(json!({
            "url": "https://example.com/a.png", "detail": "ultra"
        }))
        .is_err()
    );
}

#[test]
fn image_detail_defaults_to_auto_only_when_omitted() {
    let omitted: ImageUrl =
        serde_json::from_value(json!({"url": "https://example.com/a.png"})).unwrap();
    assert_eq!(omitted.detail, Some(ImageDetail::Auto));
    assert_eq!(serde_json::to_value(&omitted).unwrap()["detail"], "auto");

    let explicit_null: ImageUrl =
        serde_json::from_value(json!({"url": "https://example.com/a.png", "detail": null}))
            .unwrap();
    assert_eq!(explicit_null.detail, None);
    assert_eq!(
        serde_json::to_value(&explicit_null).unwrap()["detail"],
        serde_json::Value::Null
    );

    let medium: ImageUrl =
        serde_json::from_value(json!({"url": "https://example.com/a.png", "detail": "medium"}))
            .unwrap();
    assert_eq!(medium.detail, Some(ImageDetail::Medium));
}

#[test]
fn image_detail_converts_from_upstream() {
    use async_openai::types::chat::ImageDetail as Upstream;
    for (upstream, local) in [
        (Upstream::Auto, ImageDetail::Auto),
        (Upstream::Low, ImageDetail::Low),
        (Upstream::High, ImageDetail::High),
        (Upstream::Original, ImageDetail::Original),
    ] {
        assert_eq!(ImageDetail::from(upstream), local);
    }
}
