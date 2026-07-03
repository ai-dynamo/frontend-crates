// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_parsers::tool_calling::jail::{Annotated, JailedStream};
use dynamo_parsers::tool_calling::{config::ToolCallConfig, json::try_tool_call_parse_basic_json};
use dynamo_protocols::types::{
    ChatChoiceStream, ChatCompletionMessageContent, ChatCompletionStreamResponseDelta,
    CreateChatCompletionStreamResponse, Role,
};
use futures::{StreamExt, stream};

fn chunk(content: impl Into<String>) -> Annotated<CreateChatCompletionStreamResponse> {
    #[allow(deprecated)]
    let choice = ChatChoiceStream {
        index: 0,
        delta: ChatCompletionStreamResponseDelta {
            role: Some(Role::Assistant),
            content: Some(ChatCompletionMessageContent::Text(content.into())),
            tool_calls: None,
            function_call: None,
            refusal: None,
            reasoning_content: None,
        },
        finish_reason: None,
        logprobs: None,
    };
    Annotated {
        data: Some(CreateChatCompletionStreamResponse {
            id: "incremental-jail-test".to_string(),
            choices: vec![choice],
            created: 0,
            model: "test-model".to_string(),
            system_fingerprint: None,
            object: "chat.completion.chunk".to_string(),
            usage: None,
            service_tier: None,
        }),
        id: None,
        event: None,
        comment: None,
        error: None,
    }
}

async fn run(
    parser: &str,
    chunks: impl IntoIterator<Item = &'static str>,
) -> Vec<Annotated<CreateChatCompletionStreamResponse>> {
    let chunks: Vec<_> = chunks.into_iter().map(chunk).collect();
    JailedStream::builder()
        .tool_call_parser(parser)
        .build()
        .apply_with_finish_reason(stream::iter(chunks))
        .collect()
        .await
}

fn tool_calls(
    responses: &[Annotated<CreateChatCompletionStreamResponse>],
) -> Vec<(String, String)> {
    responses
        .iter()
        .filter_map(|response| response.data.as_ref())
        .flat_map(|response| response.choices.iter())
        .filter_map(|choice| choice.delta.tool_calls.as_ref())
        .flatten()
        .filter_map(|call| call.function.as_ref())
        .filter_map(|function| {
            Some((
                function.name.clone()?,
                function.arguments.clone().unwrap_or_default(),
            ))
        })
        .collect()
}

fn content(responses: &[Annotated<CreateChatCompletionStreamResponse>]) -> String {
    responses
        .iter()
        .filter_map(|response| response.data.as_ref())
        .flat_map(|response| response.choices.iter())
        .filter_map(|choice| choice.delta.content.as_ref())
        .filter_map(|content| match content {
            ChatCompletionMessageContent::Text(text) => Some(text.as_str()),
            ChatCompletionMessageContent::Parts(_) => None,
        })
        .collect()
}

#[tokio::test]
async fn invalid_balanced_candidate_does_not_pin_later_valid_call() {
    let responses = run(
        "llama3_json",
        [
            r#"<|python_tag|>{"name": }"#,
            r#"<|python_tag|>{"name":"get_time","arguments":{}}"#,
        ],
    )
    .await;

    let calls = tool_calls(&responses);
    assert_eq!(calls.len(), 1, "later valid JSON must trigger revalidation");
    assert_eq!(calls[0].0, "get_time");
}

#[tokio::test]
async fn argumentless_wrapped_calls_keep_existing_behavior() {
    for parser in ["hermes", "qwen25"] {
        let responses = run(parser, [r#"<tool_call>{"name":"get_time"}</tool_call>"#]).await;
        let calls = tool_calls(&responses);
        assert_eq!(calls.len(), 1, "{parser} argument-less call was dropped");
        assert_eq!(calls[0].0, "get_time");
        assert_eq!(calls[0].1, "{}");
    }
}

#[tokio::test]
async fn split_mistral_close_marker_never_leaks() {
    let responses = run(
        "mistral",
        [
            r#"[TOOL_CALLS][{"name":"get_time","arguments":{}}]"#,
            "[/TOOL_CA",
            "LLS]",
            " terminé 🧪",
        ],
    )
    .await;

    assert_eq!(tool_calls(&responses).len(), 1);
    let text = content(&responses);
    assert!(
        !text.contains("[/TOOL_CALLS]"),
        "close marker leaked: {text:?}"
    );
    assert_eq!(text, " terminé 🧪");
}

#[tokio::test]
async fn overlapping_end_markers_choose_longest_boundary() {
    let responses: Vec<_> = JailedStream::builder()
        .tool_call_parser("hermes")
        .jail_end_sequences(["</tool", "</tool_call>"])
        .build()
        .apply_with_finish_reason(stream::iter([chunk(
            "<tool_call>not-json</tool_call>visible",
        )]))
        .collect()
        .await;

    assert_eq!(content(&responses), "visible");
}

#[tokio::test]
async fn markerless_followup_calls_rejail_from_parser_capabilities() {
    let cases = [
        (
            "llama3_json",
            r#"<|python_tag|>{"name":"first","arguments":{}}"#,
            r#"{"name":"second","arguments":{}}"#,
        ),
        (
            "phi4",
            r#"functools[{"name":"first","arguments":{}}]"#,
            r#"[{"name":"second","arguments":{}}]"#,
        ),
    ];

    for (parser, first, second) in cases {
        let responses = run(parser, [first, second]).await;
        let names: Vec<_> = tool_calls(&responses)
            .into_iter()
            .map(|(name, _)| name)
            .collect();
        assert_eq!(names, ["first", "second"], "{parser} markerless followup");
    }
}

#[tokio::test]
async fn kimi_missing_section_end_still_recovers_at_eof() {
    let responses = run(
        "kimi_k2",
        [
            "<|tool_calls_section_begin|>",
            r#"<|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{"location":"Paris"}<|tool_call_end|>"#,
        ],
    )
    .await;

    let calls = tool_calls(&responses);
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].0, "get_weather");
    assert!(!content(&responses).contains("<|tool_call"));
}

#[test]
fn raw_null_arguments_remain_present() {
    let ToolCallConfig { parser_config, .. } = ToolCallConfig::hermes();
    let dynamo_parsers::tool_calling::config::ParserConfig::Json(config) = parser_config else {
        unreachable!();
    };
    let (calls, _) = try_tool_call_parse_basic_json(
        r#"<tool_call>{"name":"nullable","arguments":null}</tool_call>"#,
        &config,
        None,
    )
    .unwrap();

    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.arguments, "null");
}
