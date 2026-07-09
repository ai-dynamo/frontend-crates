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
            reasoning: None,
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

fn content_chunks(responses: &[Annotated<CreateChatCompletionStreamResponse>]) -> Vec<String> {
    responses
        .iter()
        .filter_map(|response| response.data.as_ref())
        .flat_map(|response| response.choices.iter())
        .filter_map(|choice| choice.delta.content.as_ref())
        .filter_map(|content| match content {
            ChatCompletionMessageContent::Text(text) => Some(text.clone()),
            ChatCompletionMessageContent::Parts(_) => None,
        })
        .filter(|text| !text.is_empty())
        .collect()
}

#[tokio::test]
async fn marker_only_jail_completes_even_when_default_parser_accepts_body() {
    let responses: Vec<_> = JailedStream::builder()
        .jail_start_sequence("<jail>")
        .jail_end_sequence("</jail>")
        .build()
        .apply_with_finish_reason(stream::iter([
            chunk(r#"<jail><TOOLCALL>[{"name":"get_time","arguments":{}}]</TOOLCALL></jail>"#),
            chunk(" trailing prose"),
        ]))
        .collect()
        .await;

    assert_eq!(tool_calls(&responses).len(), 1);
    assert!(content(&responses).ends_with(" trailing prose"));
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
async fn unbalanced_candidate_resynchronizes_at_later_start_marker() {
    for poisoned in [r#"<|python_tag|>{""#, r#"<|python_tag|>{{{"#] {
        let responses = run(
            "llama3_json",
            [
                poisoned,
                r#"<|python_tag|>{"name":"get_time","arguments":{}}"#,
                " all done!",
            ],
        )
        .await;

        let calls = tool_calls(&responses);
        assert_eq!(calls.len(), 1, "poisoned prefix: {poisoned:?}");
        assert_eq!(calls[0].0, "get_time");
        assert_eq!(content(&responses), " all done!");
    }
}

#[tokio::test]
async fn deepseek_bare_calls_complete_before_trailing_prose() {
    let cases = [
        (
            "deepseek_v3",
            concat!(
                "<｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n",
                "```json\n{\"location\":\"Paris\"}\n```\n",
                "<｜tool▁call▁end｜>"
            ),
        ),
        (
            "deepseek_v3_1",
            concat!(
                "<｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>",
                "{\"location\":\"Paris\"}<｜tool▁call▁end｜>"
            ),
        ),
    ];

    for (parser, call) in cases {
        let responses = run(parser, [call, " And here is some trailing prose."]).await;
        let calls = tool_calls(&responses);
        assert_eq!(calls.len(), 1, "bare call was dropped for {parser}");
        assert_eq!(calls[0].0, "get_weather");
        assert_eq!(content(&responses), " And here is some trailing prose.");
    }
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
async fn validated_mistral_candidate_retries_authoritative_boundary() {
    let responses = run(
        "mistral",
        [
            r#"[TOOL_CALLS][{"name":"get_time","arguments":{}}]"#,
            " trailing prose",
        ],
    )
    .await;

    assert_eq!(tool_calls(&responses).len(), 1);
    assert_eq!(content(&responses), " trailing prose");
}

#[tokio::test]
async fn overlapping_end_markers_choose_longest_boundary() {
    let responses: Vec<_> = JailedStream::builder()
        .tool_call_parser("hermes")
        .jail_end_sequences(["</tool_call>", "</tool"])
        .build()
        .apply_with_finish_reason(stream::iter([chunk(
            "<tool_call>not-json</tool_call>visible",
        )]))
        .collect()
        .await;

    assert_eq!(content(&responses), "visible");
}

#[tokio::test]
async fn configured_end_marker_order_has_priority_over_text_position() {
    let responses: Vec<_> = JailedStream::builder()
        .jail_start_sequence("<jail>")
        .jail_end_sequences(["<late>", "</jail>"])
        .build()
        .apply_with_finish_reason(stream::iter([chunk(
            "<jail>inside</jail>outside<late>tail",
        )]))
        .collect()
        .await;

    let chunks = content_chunks(&responses);
    assert_eq!(chunks.len(), 2);
    assert_eq!(chunks[0], "<jail>inside</jail>outside<late>");
    assert_eq!(chunks[1], "tail");
}

#[tokio::test]
async fn manual_end_marker_override_disables_json_boundary_exit() {
    let responses: Vec<_> = JailedStream::builder()
        .tool_call_parser("llama3_json")
        .jail_end_sequence("[[END]]")
        .build()
        .apply_with_finish_reason(stream::iter([
            chunk(r#"<|python_tag|>{"name":"get_time","arguments":{}}"#),
            chunk("[[END]]"),
            chunk(" after"),
        ]))
        .collect()
        .await;

    assert_eq!(tool_calls(&responses).len(), 1);
    assert_eq!(content(&responses), " after");
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
