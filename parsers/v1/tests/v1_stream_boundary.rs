// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tests for the parser-name rules, the jail boundary, and the stream item
//! seam. The first five are moved from dynamo `lib/llm` with the code
//! (`src/preprocessor.rs` and `tests/test_streaming_tool_parsers.rs`).

use dynamo_parsers::chat_stream::ChatStreamItem;
use dynamo_parsers::reasoning::stream::{
    ReasoningStreamOptions, guided_json_reasoning_start_token, parse_reasoning_content_from_stream,
    strip_leading_reasoning_start_from_stream, supports_reasoning_before_guided_json,
};
use dynamo_parsers::tool_calling::jail::{
    Annotated, JailChoiceCountError, apply_jail_terminal_error_contract, apply_tool_calling_jail,
    validate_jail_nvext_choice_count,
};
use dynamo_protocols::types::{
    ChatCompletionMessageContent, ChatCompletionToolChoiceOption,
    CreateChatCompletionStreamResponse,
};
use futures::{StreamExt, stream};

type Chunk = Annotated<CreateChatCompletionStreamResponse>;

fn chunk(choices: serde_json::Value) -> Chunk {
    Annotated {
        data: Some(
            serde_json::from_value(serde_json::json!({"id": "test",
                "object": "chat.completion.chunk", "created": 0, "model": "test",
                "choices": choices}))
            .unwrap(),
        ),
        id: None,
        event: None,
        comment: None,
        error: None,
    }
}

fn text_chunk(index: u32, text: &str) -> Chunk {
    chunk(serde_json::json!(
        [{"index": index, "delta": {"content": text}, "finish_reason": null}]
    ))
}

fn error_item(message: &str) -> Chunk {
    Annotated {
        data: None,
        id: None,
        event: Some("error".to_string()),
        comment: None,
        error: Some(message.to_string()),
    }
}

fn has_tool_calls(out: &[Chunk]) -> bool {
    out.iter().any(|item| {
        item.data.as_ref().is_some_and(|data| {
            data.choices.iter().any(|choice| {
                choice
                    .delta
                    .tool_calls
                    .as_ref()
                    .is_some_and(|c| !c.is_empty())
            })
        })
    })
}

/// Verifies which force-reasoning parsers use guided-output shape detection.
#[test]
fn test_reasoning_before_guided_json_parser_allowlist() {
    for parser in [
        "deepseek_r1",
        "deepseek_v3",
        "deepseek_v3_1",
        "deepseek_v3_2",
        "step3",
        "kimi_k25",
        "mistral",
        "minimax_m2",
        "nemotron_nano",
        "nemotron3",
        "nemotron_v3",
    ] {
        assert!(
            supports_reasoning_before_guided_json(Some(parser)),
            "{parser} should inspect guided output shape"
        );
    }

    assert!(
        !supports_reasoning_before_guided_json(Some("minimax_append_think")),
        "minimax_append_think must retain the guided-JSON bypass"
    );
}

/// Verifies parser-specific openers that can be confused with guided JSON.
#[test]
fn test_guided_json_reasoning_start_tokens() {
    assert_eq!(
        guided_json_reasoning_start_token(Some("mistral")),
        Some("[THINK]")
    );
    assert_eq!(guided_json_reasoning_start_token(Some("nemotron_v3")), None);
}

#[test]
fn jail_rejects_multiple_choices_for_choice_specific_nvext() {
    let engine_data = vec!["engine_data".to_string()];
    let request_level = vec!["timing".to_string(), "worker_id".to_string()];

    assert_eq!(
        validate_jail_nvext_choice_count(2, Some(&engine_data), true),
        Err(JailChoiceCountError {
            field: "engine_data".to_string()
        })
    );
    assert!(validate_jail_nvext_choice_count(1, Some(&engine_data), true).is_ok());
    assert!(validate_jail_nvext_choice_count(2, Some(&request_level), true).is_ok());
    // A route without the jail (unified, pass-through) accepts it.
    assert!(validate_jail_nvext_choice_count(2, Some(&engine_data), false).is_ok());
    // dynamo returns this text in its HTTP 400 body.
    assert_eq!(
        JailChoiceCountError {
            field: "stop_reason".to_string()
        }
        .to_string(),
        "legacy tool-call parsing requires n = 1 when nvext.extra_fields requests choice-specific field `stop_reason`"
    );
}

#[tokio::test]
async fn test_minimax_m3_end_of_stream_flush_skips_finished_choices() {
    /// `(index, content, finish_reason set)` per choice, chunk by chunk.
    async fn placement(backend: Vec<Chunk>) -> Vec<Vec<(u32, String, bool)>> {
        parse_reasoning_content_from_stream(
            stream::iter(backend),
            "minimax_m3".to_string(),
            ReasoningStreamOptions::default(),
        )
        .filter_map(|item| async move { item.data })
        .map(|data| {
            data.choices
                .iter()
                .map(|choice| {
                    let content = match choice.delta.content.as_ref() {
                        Some(ChatCompletionMessageContent::Text(text)) => text.clone(),
                        _ => String::new(),
                    };
                    (choice.index, content, choice.finish_reason.is_some())
                })
                .collect()
        })
        .collect()
        .await
    }
    let world = text_chunk(1, "World");
    let hello = text_chunk(0, "Hello");
    let stop = chunk(serde_json::json!(
        [{"index": 0, "delta": {}, "finish_reason": "stop"}]
    ));

    // Choice 0 finished on its terminal chunk, so the end-of-stream flush
    // carries only choice 1.
    assert_eq!(
        placement(vec![world.clone(), hello.clone(), stop]).await,
        vec![
            vec![(1, String::new(), false)],
            vec![(0, String::new(), false)],
            vec![(0, "Hello".to_string(), true)],
            vec![(1, "World".to_string(), false)],
        ]
    );

    // With neither finished, one flush chunk carries both, in index order.
    assert_eq!(
        placement(vec![world, hello]).await,
        vec![
            vec![(1, String::new(), false)],
            vec![(0, String::new(), false)],
            vec![
                (0, "Hello".to_string(), false),
                (1, "World".to_string(), false)
            ],
        ]
    );
}

fn deepseek_v4_partial_invoke_chunk() -> Chunk {
    text_chunk(
        0,
        "<｜DSML｜tool_calls>\n\
<｜DSML｜invoke name=\"get_weather\">\n\
<｜DSML｜parameter name=\"location\" string=\"true\">NYC</｜DSML｜parameter>\n\
</｜DSML｜invoke>",
    )
}

async fn deepseek_v4_jail(input: Vec<Chunk>) -> Vec<Chunk> {
    apply_jail_terminal_error_contract(stream::iter(input), |s| {
        apply_tool_calling_jail(Some("deepseek_v4".to_string()), None, None, false, s)
    })
    .collect()
    .await
}

// Control for the test below: at a clean end of stream with no finish reason,
// the jail does finalize the finished invoke into a tool call.
#[tokio::test]
async fn test_deepseek_v4_finalize_recovers_at_bare_eof_with_no_finish_reason() {
    let out = deepseek_v4_jail(vec![deepseek_v4_partial_invoke_chunk()]).await;
    assert!(
        has_tool_calls(&out),
        "a clean EOF (no error) must still recover the finished invoke; got: {out:#?}"
    );
}

// The same partial invoke, but the stream ends in an error. The contract must
// yield exactly that error, last and once, and no tool call built by the jail's
// end-of-stream finalize, because the request never completed.
#[tokio::test]
async fn test_deepseek_v4_upstream_errors_suppress_jail_finalize() {
    for message in ["invalid argument", "engine shutdown", "disconnected"] {
        let out = deepseek_v4_jail(vec![
            deepseek_v4_partial_invoke_chunk(),
            error_item(message),
        ])
        .await;

        let last = out
            .last()
            .expect("the terminal error must reach the caller");
        assert!(last.is_error(), "the error must be last; got: {out:#?}");
        assert_eq!(last.error.as_deref(), Some(message));
        assert_eq!(out.iter().filter(|a| a.is_error()).count(), 1);
        assert!(
            !has_tool_calls(&out),
            "an upstream error must suppress the jail's finalize; got: {out:#?}"
        );
    }
}

#[tokio::test]
async fn jail_yields_transport_error_after_finished_choice() {
    let stop = chunk(serde_json::json!(
        [{"index": 0, "delta": {}, "finish_reason": "stop"}]
    ));
    let out: Vec<Chunk> = apply_jail_terminal_error_contract(
        stream::iter(vec![stop, error_item("transport failed")]),
        |s| {
            apply_tool_calling_jail(
                Some("hermes".to_string()),
                Some(ChatCompletionToolChoiceOption::Required),
                None,
                false,
                s,
            )
        },
    )
    .collect()
    .await;

    assert_eq!(out.iter().filter(|a| a.is_error()).count(), 1);
    assert!(out.last().is_some_and(|a| a.is_error()));
}

/// An item with a field this crate does not know, as dynamo's item carries
/// metrics and `nvext` beside the chunk.
#[derive(Clone, Debug)]
struct MeteredItem {
    item: Chunk,
    chunk_tokens: Option<u32>,
}

impl ChatStreamItem for MeteredItem {
    fn is_error(&self) -> bool {
        self.item.is_error()
    }
    fn chunk(&self) -> Option<&CreateChatCompletionStreamResponse> {
        self.item.chunk()
    }
    fn chunk_mut(&mut self) -> Option<&mut CreateChatCompletionStreamResponse> {
        self.item.chunk_mut()
    }
    fn scrub_synthetic(&mut self) {
        self.item.scrub_synthetic();
        self.chunk_tokens = None;
    }
}

fn metered(item: Chunk, chunk_tokens: u32) -> MeteredItem {
    MeteredItem {
        item,
        chunk_tokens: Some(chunk_tokens),
    }
}

// The stages keep the caller's per-chunk fields on each chunk they pass on,
// and scrub them on the synthetic chunk that a flush builds at end of stream.
// MiniMax M3 holds marker-free text until the end, so its flush builds one.
#[tokio::test]
async fn stages_keep_item_fields_and_scrub_flush_chunks() {
    let input = vec![
        metered(text_chunk(0, "plan"), 2),
        metered(text_chunk(0, " more"), 3),
    ];
    let out: Vec<MeteredItem> = parse_reasoning_content_from_stream(
        stream::iter(input),
        "minimax_m3".to_string(),
        ReasoningStreamOptions::default(),
    )
    .collect()
    .await;
    let tokens: Vec<_> = out.iter().map(|item| item.chunk_tokens).collect();
    assert_eq!(tokens, vec![Some(2), Some(3), None], "got: {out:#?}");

    let out: Vec<MeteredItem> = strip_leading_reasoning_start_from_stream(
        stream::iter(vec![metered(text_chunk(0, "<thi"), 4)]),
        "<think>",
    )
    .collect()
    .await;
    let tokens: Vec<_> = out.iter().map(|item| item.chunk_tokens).collect();
    assert_eq!(tokens, vec![Some(4), None], "got: {out:#?}");
}

// The jail still sees an end of stream after an error, so its finalize runs,
// as in dynamo. The contract drops what it emits; only the error escapes.
#[tokio::test]
async fn terminal_error_drops_jail_finalize_output() {
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    let finalized = Arc::new(AtomicUsize::new(0));
    let finalized_in = Arc::clone(&finalized);
    let out: Vec<Chunk> = apply_jail_terminal_error_contract(
        stream::iter(vec![error_item("backend failed")]),
        move |mut input| {
            async_stream::stream! {
                while let Some(item) = input.next().await {
                    yield item;
                }
                finalized_in.fetch_add(1, Ordering::SeqCst);
                yield text_chunk(0, "EOF finalize output");
            }
        },
    )
    .collect()
    .await;

    assert_eq!(out.len(), 1);
    assert!(out[0].is_error());
    assert_eq!(out[0].error.as_deref(), Some("backend failed"));
    assert_eq!(
        finalized.load(Ordering::SeqCst),
        1,
        "the jail's finalize runs once; its output is dropped"
    );
}
