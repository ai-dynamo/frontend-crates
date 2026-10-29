// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_chat_parsing::ParserVersion;
use dynamo_chat_parsing::aggregator::{AggregatorOptions, DeltaAggregator};
use dynamo_chat_parsing::guided::GuidedToolConstraint;
use dynamo_chat_parsing::reasoning_off::ChatStreamRequest;
use dynamo_protocols::types::{
    ChatCompletionMessageContent, CreateChatCompletionRequest, CreateChatCompletionResponse,
};
use serde_json::{Value, json};

async fn aggregate(args: Option<Value>) -> (CreateChatCompletionResponse, AggregatorOptions) {
    let request: CreateChatCompletionRequest = serde_json::from_value(json!({
        "model": "gpt-oss", "messages": [{"role": "user", "content": "hi"}]
    }))
    .expect("request");
    let args = args.map(|args| serde_json::from_value(args).expect("args"));
    let view = ChatStreamRequest {
        request: &request,
        chat_template_args: args.as_ref(),
        client_thinking_field: false,
        explicit_guided_decoding: false,
    };
    let options = AggregatorOptions::for_request(
        &view,
        Some("harmony"),
        Some("gpt_oss"),
        None,
        GuidedToolConstraint::None,
        Vec::new(),
        ParserVersion::Auto,
    );
    let chunk = serde_json::from_value(json!({
        "id": "reply", "created": 0, "model": "gpt-oss", "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": {
            "role": "assistant",
            "content": "<|start|>assistant<|channel|>analysis<|message|>Private analysis.<|end|><|start|>assistant<|channel|>final<|message|>42<|return|>"
        }, "finish_reason": "stop"}]
    }))
    .expect("chunk");
    let mut actual = DeltaAggregator::new();
    actual.push(chunk, &[]);
    (actual.finish(&options).await.expect("aggregate"), options)
}

#[tokio::test]
async fn gpt_oss_without_a_thinking_alias_keeps_analysis_separate() {
    let (actual, options) = aggregate(None).await;
    assert!(!options.reasoning_disabled);
    let message = &actual.choices[0].message;
    assert_eq!(
        message.content,
        Some(ChatCompletionMessageContent::Text("42".into()))
    );
    assert_eq!(
        message.reasoning_content.as_deref(),
        Some("Private analysis.")
    );
}

#[tokio::test]
async fn gpt_oss_options_keep_analysis_out_of_the_answer() {
    let (actual, options) = aggregate(Some(json!({"enable_thinking": false}))).await;
    let message = &actual.choices[0].message;
    // Dynamo 37f2f0a712, preprocessor.rs:7170: GPT-OSS cannot disable reasoning.
    assert_eq!(
        (&message.content, message.reasoning_content.as_deref()),
        (
            &Some(ChatCompletionMessageContent::Text("42".into())),
            Some("Private analysis.")
        ),
        "the handler must keep analysis separate even when an unsupported thinking alias is false; options: {options:?}"
    );
}
