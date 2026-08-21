// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Guided tool-call payloads must reach the client AS THEY ARRIVE.
//!
//! Under `tool_choice=required` or a named tool, generation is constrained to a
//! known JSON shape, so the jail does not have to buffer until the closing brace.
//! These tests assert the delivery TIMELINE, not just the final value: a buffering
//! path produces one terminal delta, a streaming path spreads them across chunks.

use dynamo_parsers::tool_calling::jail::{Annotated, JailedStream};
use dynamo_protocols::types::{
    ChatChoiceStream, ChatCompletionMessageContent, ChatCompletionStreamResponseDelta,
    CreateChatCompletionStreamResponse, Role,
};
use futures::StreamExt;
use futures::stream;

fn chunk(content: &str) -> Annotated<CreateChatCompletionStreamResponse> {
    #[allow(deprecated)]
    let choice = ChatChoiceStream {
        index: 0,
        delta: ChatCompletionStreamResponseDelta {
            role: Some(Role::Assistant),
            content: Some(ChatCompletionMessageContent::Text(content.to_string())),
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
            id: "test-id".to_string(),
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

/// Split on char boundaries so a chunk never cuts a multi-byte character.
fn split_every(payload: &str, n: usize) -> Vec<Annotated<CreateChatCompletionStreamResponse>> {
    let mut out = Vec::new();
    let mut at = 0usize;
    while at < payload.len() {
        let mut end = (at + n).min(payload.len());
        while !payload.is_char_boundary(end) {
            end += 1;
        }
        out.push(chunk(&payload[at..end]));
        at = end;
    }
    out
}

/// Tool-call delivery plus visible assistant content.
async fn drive(
    jail: JailedStream,
    chunks: Vec<Annotated<CreateChatCompletionStreamResponse>>,
) -> (usize, Vec<String>, String, String) {
    let results: Vec<_> = jail
        .apply_with_finish_reason(stream::iter(chunks))
        .collect()
        .await;
    let mut carrying = 0usize;
    let mut names = Vec::new();
    let mut args = String::new();
    let mut content = String::new();
    for r in &results {
        let Some(data) = r.data.as_ref() else {
            continue;
        };
        let mut saw = false;
        for choice in &data.choices {
            if let Some(ChatCompletionMessageContent::Text(text)) = choice.delta.content.as_ref() {
                content.push_str(text);
            }
            let Some(calls) = choice.delta.tool_calls.as_ref() else {
                continue;
            };
            saw = true;
            for call in calls {
                if let Some(f) = call.function.as_ref() {
                    if let Some(n) = f.name.as_ref() {
                        names.push(n.clone());
                    }
                    if let Some(a) = f.arguments.as_ref() {
                        args.push_str(a);
                    }
                }
            }
        }
        if saw {
            carrying += 1;
        }
    }
    (carrying, names, args, content)
}

#[tokio::test]
async fn required_streams_across_chunks_instead_of_one_burst() {
    let payload = r#"[{"name":"search","parameters":{"query":"Rust","limit":10,"note":"a longer value so there are many fragments to release"}}]"#;
    let (carrying, names, args, content) = drive(
        JailedStream::builder()
            .tool_choice_required()
            .guided_streaming(true)
            .build(),
        split_every(payload, 7),
    )
    .await;

    assert_eq!(
        names,
        vec!["search".to_string()],
        "name must ride exactly one delta"
    );
    assert_eq!(
        args,
        r#"{"query":"Rust","limit":10,"note":"a longer value so there are many fragments to release"}"#,
        "arguments must reassemble byte for byte, exactly once"
    );
    assert!(
        carrying > 3,
        "arguments must arrive across many responses, got {carrying}"
    );
    assert!(
        content.is_empty(),
        "guided JSON leaked as text: {content:?}"
    );
}

#[tokio::test]
async fn named_streams_across_chunks_instead_of_one_burst() {
    let payload = r#"{"query":"Rust","limit":10,"note":"a longer value so there are many fragments to release"}"#;
    let (carrying, names, args, content) = drive(
        JailedStream::builder()
            .tool_choice_named("search".to_string())
            .guided_streaming(true)
            .build(),
        split_every(payload, 7),
    )
    .await;

    assert_eq!(names, vec!["search".to_string()]);
    assert_eq!(
        args, payload,
        "arguments must reassemble byte for byte, exactly once"
    );
    assert!(
        carrying > 3,
        "arguments must arrive across many responses, got {carrying}"
    );
    assert!(
        content.is_empty(),
        "guided JSON leaked as text: {content:?}"
    );
}

/// The hazard: the completion path rebuilds every call in full, so a streamed call
/// must be subtracted or the client receives its arguments twice.
#[tokio::test]
async fn streamed_arguments_are_never_emitted_twice() {
    let payload = r#"[{"name":"search","parameters":{"query":"Rust"}}]"#;
    let (_, names, args, content) = drive(
        JailedStream::builder()
            .tool_choice_required()
            .guided_streaming(true)
            .build(),
        split_every(payload, 5),
    )
    .await;
    assert_eq!(
        names.len(),
        1,
        "the name must not be repeated, got {names:?}"
    );
    assert_eq!(
        args, r#"{"query":"Rust"}"#,
        "duplicated arguments: {args:?}"
    );
    assert!(
        content.is_empty(),
        "guided JSON leaked as text: {content:?}"
    );
}

/// EOF after a call was already committed.
///
/// `finalize` rebuilds calls independently of the normal completion path, so it runs
/// the same subtraction. NOTE: this test currently passes with that subtraction
/// removed - for Immediate mode the recovery parse yields no call for a truncated
/// payload, so there is nothing to duplicate. The subtraction is kept so both exits
/// share one reconciliation rule rather than relying on that. This test pins the
/// INVARIANT (arguments appear once at EOF); it is not proof the subtraction bites.
#[tokio::test]
async fn truncated_after_commit_does_not_duplicate_streamed_arguments() {
    // Committed (name closed, argument object opened) but never closed.
    let payload = r#"[{"name":"search","parameters":{"query":"Rust"}}"#;
    let (_, names, args, content) = drive(
        JailedStream::builder()
            .tool_choice_required()
            .guided_streaming(true)
            .build(),
        split_every(payload, 6),
    )
    .await;

    assert!(
        names.len() <= 1,
        "the name must not be repeated at EOF, got {names:?}"
    );
    assert!(
        !args.contains(r#"{"query":"Rust{"query":"Rust"#),
        "truncated payload duplicated its arguments: {args:?}"
    );
    assert_eq!(
        args.matches(r#""query""#).count(),
        1,
        "argument bytes must appear exactly once, got {args:?}"
    );
    assert!(
        !content.contains("query") && !content.contains("Rust"),
        "streamed argument bytes also appeared as content: {content:?}"
    );
}

/// Two guided payloads in one stream. The cursor's byte offsets and released-byte
/// counts describe one payload; carrying them across would suppress the second call's
/// arguments as "already streamed".
#[tokio::test]
async fn a_second_guided_payload_is_not_poisoned_by_the_first() {
    let first = r#"[{"name":"search","parameters":{"query":"Rust"}}]"#;
    let second = r#"[{"name":"search","parameters":{"query":"Zig"}}]"#;
    let mut chunks = split_every(first, 6);
    chunks.extend(split_every(second, 6));

    let (_, names, args, content) = drive(
        JailedStream::builder()
            .tool_choice_required()
            .guided_streaming(true)
            .build(),
        chunks,
    )
    .await;

    assert!(
        args.contains(r#""Rust""#),
        "first payload's arguments missing: {args:?}"
    );
    assert!(
        args.contains(r#""Zig""#),
        "second payload's arguments were suppressed by the first payload's ledger: {args:?}"
    );
    assert_eq!(
        names.len(),
        2,
        "expected one name per payload, got {names:?}"
    );
    assert!(
        content.is_empty(),
        "guided JSON leaked as text: {content:?}"
    );
}

#[tokio::test]
async fn named_leading_whitespace_does_not_shift_the_rebuilt_tail() {
    let payload = "  \n\t{\"query\":\"Rust\",\"limit\":10}";
    let (_, names, args, content) = drive(
        JailedStream::builder()
            .tool_choice_named("search".to_string())
            .guided_streaming(true)
            .build(),
        split_every(payload, 5),
    )
    .await;

    assert_eq!(names, vec!["search".to_string()]);
    assert_eq!(args, r#"{"query":"Rust","limit":10}"#);
    assert!(
        content.is_empty(),
        "guided JSON leaked as text: {content:?}"
    );
}

#[tokio::test]
async fn truncated_named_arguments_do_not_reappear_as_content() {
    let payload = "  {\"query\":\"Par";
    let (_, names, args, content) = drive(
        JailedStream::builder()
            .tool_choice_named("search".to_string())
            .guided_streaming(true)
            .build(),
        split_every(payload, 4),
    )
    .await;

    assert_eq!(names, vec!["search".to_string()]);
    assert_eq!(args, r#"{"query":"Par"#);
    assert!(
        content.is_empty(),
        "streamed argument bytes also appeared as content: {content:?}"
    );
}

#[tokio::test]
async fn disagreement_never_reemits_a_streamed_call() {
    let payload = r#"[{"name":"search","parameters":{"q":1},"arguments":{"different":true}}]"#;
    let (_, names, args, content) = drive(
        JailedStream::builder()
            .tool_choice_required()
            .guided_streaming(true)
            .build(),
        split_every(payload, 6),
    )
    .await;

    assert_eq!(names, vec!["search".to_string()]);
    assert_eq!(args, r#"{"q":1}"#, "the rebuilt call was emitted again");
    assert!(
        content.is_empty(),
        "guided JSON leaked as text: {content:?}"
    );
}

#[tokio::test]
async fn reconciliation_is_invariant_at_every_split() {
    let cases = [
        (
            false,
            r#"[{"parameters":{"q":1}},{"name":"b","parameters":{"x":1}}]"#,
            "b",
            r#"{"x":1}"#,
        ),
        (
            false,
            r#"[{"name":"search","parameters":{"q":1},"arguments":{"different":true}}]"#,
            "search",
            r#"{"q":1}"#,
        ),
        (
            true,
            "  \n\t{\"query\":\"Rust\",\"limit\":10}",
            "search",
            r#"{"query":"Rust","limit":10}"#,
        ),
        (true, "  {\"query\":\"Par", "search", r#"{"query":"Par"#),
    ];

    for (named, payload, expected_name, expected_arguments) in cases {
        for split in 0..=payload.len() {
            if !payload.is_char_boundary(split) {
                continue;
            }
            let chunks = if split == 0 || split == payload.len() {
                vec![chunk(payload)]
            } else {
                vec![chunk(&payload[..split]), chunk(&payload[split..])]
            };
            let builder = JailedStream::builder().guided_streaming(true);
            let jail = if named {
                builder.tool_choice_named("search".to_string()).build()
            } else {
                builder.tool_choice_required().build()
            };
            let (_, names, arguments, content) = drive(jail, chunks).await;
            assert_eq!(
                names,
                vec![expected_name.to_string()],
                "name differed at split {split} of {payload:?}"
            );
            assert_eq!(
                arguments, expected_arguments,
                "arguments differed at split {split} of {payload:?}"
            );
            assert!(
                !content.contains("query")
                    && !content.contains("Rust")
                    && !content.contains("different"),
                "streamed bytes leaked as text at split {split} of {payload:?}: {content:?}"
            );
        }
    }
}

/// Two guided payloads must not reuse tool-call indices.
///
/// `emit_completed_jail` advances `emitted_tool_calls_count` from the chunks that
/// SURVIVE subtraction. A fully streamed call leaves none, so without counting the
/// rebuilt calls the offset never moves and the second payload restarts at index 0 —
/// the client sees two different calls sharing one index.
#[tokio::test]
async fn a_second_guided_payload_gets_a_distinct_tool_index() {
    let first = r#"[{"name":"search","parameters":{"query":"Rust"}}]"#;
    let second = r#"[{"name":"search","parameters":{"query":"Zig"}}]"#;
    let mut chunks = split_every(first, 6);
    chunks.extend(split_every(second, 6));

    let results: Vec<_> = JailedStream::builder()
        .tool_choice_required()
        .guided_streaming(true)
        .build()
        .apply_with_finish_reason(stream::iter(chunks))
        .collect()
        .await;

    // Map each emitted argument fragment to the tool index it arrived on.
    let mut by_index: std::collections::BTreeMap<u32, String> = Default::default();
    for r in &results {
        let Some(data) = r.data.as_ref() else {
            continue;
        };
        for choice in &data.choices {
            let Some(calls) = choice.delta.tool_calls.as_ref() else {
                continue;
            };
            for call in calls {
                if let Some(f) = call.function.as_ref()
                    && let Some(a) = f.arguments.as_ref()
                {
                    by_index.entry(call.index).or_default().push_str(a);
                }
            }
        }
    }

    assert_eq!(
        by_index.len(),
        2,
        "two payloads must occupy two tool indices, got {by_index:?}"
    );
    assert!(
        by_index.values().any(|v| v.contains("Rust")),
        "first payload missing: {by_index:?}"
    );
    assert!(
        by_index.values().any(|v| v.contains("Zig")),
        "second payload missing: {by_index:?}"
    );
}

/// Sparse indices: the cursor advances its element index even for elements it
/// refuses to commit, so a payload whose FIRST element is uncommittable streams its
/// second element at index 1 while only ONE call is on the wire. Advancing the offset
/// by the COUNT would move it to 1 and the next payload would reuse index 1.
#[tokio::test]
async fn a_skipped_element_still_advances_the_index_offset() {
    // Element 0 has no `name`, so the cursor never commits it - but it still advances
    // its element index, so `b` streams at tool index 1 while only ONE call is on the
    // wire and completion rebuilds only one. Advancing the offset by the COUNT leaves
    // it at 1 and the next payload reuses index 1.
    let first = r#"[{"parameters":{"q":1}},{"name":"b","parameters":{"x":1}}]"#;
    let second = r#"[{"name":"c","parameters":{"y":2}}]"#;
    let mut chunks = split_every(first, 7);
    chunks.extend(split_every(second, 7));

    let results: Vec<_> = JailedStream::builder()
        .tool_choice_required()
        .guided_streaming(true)
        .build()
        .apply_with_finish_reason(stream::iter(chunks))
        .collect()
        .await;

    let mut by_index: std::collections::BTreeMap<u32, String> = Default::default();
    for r in &results {
        let Some(data) = r.data.as_ref() else {
            continue;
        };
        for choice in &data.choices {
            let Some(calls) = choice.delta.tool_calls.as_ref() else {
                continue;
            };
            for call in calls {
                if let Some(f) = call.function.as_ref()
                    && let Some(a) = f.arguments.as_ref()
                {
                    by_index.entry(call.index).or_default().push_str(a);
                }
            }
        }
    }

    let merged: Vec<_> = by_index
        .iter()
        .filter(|(_, v)| v.contains("\"x\"") && v.contains("\"y\""))
        .collect();
    assert!(
        merged.is_empty(),
        "a skipped element left the offset short and two payloads shared an index: {by_index:?}"
    );
}

#[tokio::test]
async fn a_sparse_streamed_call_is_not_rebuilt_on_a_dense_index() {
    let payload = r#"[{"parameters":{"q":1}},{"name":"b","parameters":{"x":1}}]"#;
    let results: Vec<_> = JailedStream::builder()
        .tool_choice_required()
        .guided_streaming(true)
        .build()
        .apply_with_finish_reason(stream::iter(split_every(payload, 7)))
        .collect()
        .await;

    let mut by_index: std::collections::BTreeMap<u32, String> = Default::default();
    let mut names = Vec::new();
    for result in &results {
        let Some(data) = result.data.as_ref() else {
            continue;
        };
        for choice in &data.choices {
            let Some(calls) = choice.delta.tool_calls.as_ref() else {
                continue;
            };
            for call in calls {
                if let Some(function) = call.function.as_ref() {
                    if let Some(name) = function.name.as_ref() {
                        names.push((call.index, name.clone()));
                    }
                    if let Some(arguments) = function.arguments.as_ref() {
                        by_index.entry(call.index).or_default().push_str(arguments);
                    }
                }
            }
        }
    }

    assert_eq!(names, vec![(1, "b".to_string())]);
    assert_eq!(
        by_index,
        std::collections::BTreeMap::from([(1, r#"{"x":1}"#.to_string())]),
        "the completion parser rebuilt the streamed call on another index"
    );
}
