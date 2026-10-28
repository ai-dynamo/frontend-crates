// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Source recording through the tool-call jail.
//!
//! The jail's recording entry point is crate-internal until its stream
//! boundary takes a chunk type with a source slot, so these tests live in the
//! crate. For `hermes`, `kimi_k2`, and `minimax_m3`, on every corpus stream
//! and hand-written input, at the recorded split, one character per chunk, and
//! every two-way split:
//!
//! * Same output: recording on gives the chunks that `apply` gives.
//! * Verbatim and tiling: each emitted choice's ranges, read from the input,
//!   equal its content and its reasoning.
//! * Causality: no range goes past the bytes fed so far.
//! * Order: one choice's content ranges never go back in the input.

// A list with one range is the expected value in many cases here.
#![allow(clippy::single_range_in_vec_init)]

use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use dynamo_protocols::types::{
    ChatCompletionMessageContent, CreateChatCompletionStreamResponse, FinishReason,
};
use futures::{StreamExt, stream};

use super::{Annotated, JailedStream, SourcedItem};
use crate::sources::ChoiceSources;
use crate::tool_calling::sources_tests::{RECORDING_TOOL_PARSERS, corpus, read, targeted};

type Chunk = Annotated<CreateChatCompletionStreamResponse>;

/// One input chunk: choice index, content, reasoning, finish reason.
#[derive(Clone)]
struct In {
    index: u32,
    content: Option<String>,
    reasoning: Option<String>,
    finish: Option<FinishReason>,
}

fn text(index: u32, content: &str) -> In {
    In {
        index,
        content: Some(content.to_string()),
        reasoning: None,
        finish: None,
    }
}

fn finish(index: u32) -> In {
    In {
        index,
        content: None,
        reasoning: None,
        finish: Some(FinishReason::Stop),
    }
}

fn chunk(input: &In) -> Chunk {
    let mut delta = serde_json::json!({});
    if let Some(content) = &input.content {
        delta["content"] = content.as_str().into();
    }
    if let Some(reasoning) = &input.reasoning {
        delta["reasoning_content"] = reasoning.as_str().into();
    }
    let finish = input.finish.map(|_| "stop");
    Annotated {
        data: Some(
            serde_json::from_value(serde_json::json!({
                "id": "sources", "object": "chat.completion.chunk", "created": 0, "model": "m",
                "choices": [{"index": input.index, "delta": delta, "finish_reason": finish}]
            }))
            .unwrap(),
        ),
        id: None,
        event: None,
        comment: None,
        error: None,
    }
}

/// The stock output, with tool-call ids blanked (they are random).
fn stock_shape(chunks: &[Chunk]) -> Vec<serde_json::Value> {
    chunks
        .iter()
        .map(|chunk| {
            let mut value = serde_json::to_value(chunk).unwrap();
            if let Some(choices) = value["data"]["choices"].as_array_mut() {
                for choice in choices {
                    if let Some(calls) = choice["delta"]["tool_calls"].as_array_mut() {
                        for call in calls {
                            if call.get("id").is_some_and(|id| !id.is_null()) {
                                call["id"] = "id".into();
                            }
                        }
                    }
                }
            }
            value
        })
        .collect()
}

fn jail(parser: &str) -> JailedStream {
    JailedStream::builder().tool_call_parser(parser).build()
}

async fn stock(parser: &str, inputs: &[In]) -> Vec<Chunk> {
    let chunks: Vec<_> = inputs.iter().map(chunk).collect();
    jail(parser).apply(stream::iter(chunks)).collect().await
}

/// Runs with recording on. Each output item comes with the number of input
/// items the jail had pulled when it was produced.
async fn recorded(
    parser: &str,
    inputs: &[In],
    sources: Vec<Option<Vec<ChoiceSources>>>,
) -> Vec<(SourcedItem, usize)> {
    let pulled = Arc::new(AtomicUsize::new(0));
    let counter = pulled.clone();
    let items: Vec<SourcedItem> = inputs.iter().map(chunk).zip(sources).collect();
    let input = stream::iter(items).inspect(move |_| {
        counter.fetch_add(1, Ordering::SeqCst);
    });
    let jail = jail(parser);
    assert!(jail.can_record_sources(), "{parser} cannot record");
    jail.apply_pairs(true, input)
        .map(|item| {
            let seen = pulled.load(Ordering::SeqCst);
            (item, seen)
        })
        .collect()
        .await
}

fn content_of(choice: &dynamo_protocols::types::ChatChoiceStream) -> &str {
    match &choice.delta.content {
        Some(ChatCompletionMessageContent::Text(text)) => text,
        _ => "",
    }
}

/// Checks one input against the stock jail, with recording on, using the
/// jail's own per-choice counter for offsets.
async fn check(parser: &str, inputs: &[In], label: &str) {
    let stock = stock(parser, inputs).await;
    let sources = vec![None; inputs.len()];
    let on = recorded(parser, inputs, sources).await;
    let on_chunks: Vec<Chunk> = on.iter().map(|((chunk, _), _)| chunk.clone()).collect();
    assert_eq!(
        stock_shape(&stock),
        stock_shape(&on_chunks),
        "{label}: recording changed the output"
    );

    // Per-choice input streams, and the fed length after each input item.
    let mut content_in: std::collections::BTreeMap<u32, String> = Default::default();
    let mut reasoning_in: std::collections::BTreeMap<u32, String> = Default::default();
    let mut fed_after = Vec::new();
    for input in inputs {
        content_in
            .entry(input.index)
            .or_default()
            .push_str(input.content.as_deref().unwrap_or(""));
        reasoning_in
            .entry(input.index)
            .or_default()
            .push_str(input.reasoning.as_deref().unwrap_or(""));
        fed_after.push((content_in.clone(), reasoning_in.clone()));
    }
    let mut last_end: std::collections::BTreeMap<u32, usize> = Default::default();
    for ((chunk, sources), pulled) in &on {
        let sources = sources
            .as_ref()
            .unwrap_or_else(|| panic!("{label}: no sources while recording"));
        let choices = chunk
            .data
            .as_ref()
            .map_or(&[][..], |data| &data.choices[..]);
        assert_eq!(
            sources.len(),
            choices.len(),
            "{label}: one source entry per choice"
        );
        let (content_fed, reasoning_fed) = pulled
            .checked_sub(1)
            .map(|at| fed_after[at.min(fed_after.len() - 1)].clone())
            .unwrap_or_default();
        for (choice, source) in choices.iter().zip(sources) {
            assert_eq!(source.index, choice.index, "{label}");
            let content = content_in.get(&choice.index).map_or("", String::as_str);
            let reasoning = reasoning_in.get(&choice.index).map_or("", String::as_str);
            assert_eq!(
                read(content, &source.content),
                content_of(choice),
                "{label}: content does not tile"
            );
            assert_eq!(
                read(reasoning, &source.reasoning),
                choice.delta.reasoning_content.as_deref().unwrap_or(""),
                "{label}: reasoning does not tile"
            );
            let fed = content_fed.get(&choice.index).map_or(0, String::len);
            let fed_reasoning = reasoning_fed.get(&choice.index).map_or(0, String::len);
            for range in &source.content {
                assert!(
                    range.end <= fed,
                    "{label}: {range:?} is after the {fed} bytes fed"
                );
                let last = last_end.entry(choice.index).or_default();
                assert!(
                    range.start >= *last,
                    "{label}: {range:?} goes back from {last}"
                );
                *last = range.end;
            }
            for range in &source.reasoning {
                assert!(
                    range.end <= fed_reasoning,
                    "{label}: reasoning {range:?} is after the input"
                );
            }
        }
    }
}

fn splits(input: &str, recorded: &[String]) -> Vec<Vec<String>> {
    let mut out = vec![recorded.to_vec()];
    out.push(input.chars().map(String::from).collect());
    for at in (1..input.len()).filter(|at| input.is_char_boundary(*at)) {
        out.push(vec![input[..at].to_string(), input[at..].to_string()]);
    }
    out
}

async fn sweep(parser: &str, recorded_split: &[String], label: &str) -> usize {
    let input = recorded_split.concat();
    let mut count = 0;
    for chunks in splits(&input, recorded_split) {
        let mut inputs: Vec<In> = chunks.iter().map(|c| text(0, c)).collect();
        // With a finish chunk, and ending at EOF without one.
        check(parser, &inputs, label).await;
        inputs.push(finish(0));
        check(parser, &inputs, label).await;
        count += 2;
    }
    count
}

#[tokio::test]
async fn every_corpus_stream_at_every_split() {
    let mut runs = 0;
    let corpus = corpus();
    for case in &corpus.tool {
        runs += sweep(
            &case.parser,
            &case.chunks,
            &format!("{} [{}]", case.name, case.parser),
        )
        .await;
    }
    assert!(corpus.tool.len() >= 69, "corpus shrank");
    println!(
        "{} streams, {runs} runs per recording mode",
        corpus.tool.len()
    );
}

#[tokio::test]
async fn targeted_inputs_at_every_split() {
    let mut runs = 0;
    for parser in RECORDING_TOOL_PARSERS {
        for input in targeted(parser) {
            runs += sweep(parser, &[input.to_string()], &format!("{parser} {input:?}")).await;
        }
    }
    println!("{runs} runs per recording mode");
}

/// Content ranges of each emitted chunk for choice 0, with recording on.
async fn content_ranges(parser: &str, chunks: &[&str]) -> Vec<(String, Vec<Range<usize>>)> {
    let mut inputs: Vec<In> = chunks.iter().map(|c| text(0, c)).collect();
    inputs.push(finish(0));
    let sources = vec![None; inputs.len()];
    recorded(parser, &inputs, sources)
        .await
        .into_iter()
        .flat_map(|((chunk, sources), _)| {
            let choices = chunk.data.map(|data| data.choices).unwrap_or_default();
            choices
                .into_iter()
                .zip(sources.unwrap())
                .map(|(choice, source)| (content_of(&choice).to_string(), source.content))
                .collect::<Vec<_>>()
        })
        .filter(|(content, _)| !content.is_empty())
        .collect()
}

#[tokio::test]
async fn text_beside_a_call_is_dated_by_its_own_chunk() {
    // Review finding 1: `STOP` came from the first chunk, not the closing one.
    let got = content_ranges("hermes", &["STOP {", r#""name":"f","arguments":{}}"#]).await;
    assert_eq!(got, vec![("STOP".to_string(), vec![0..4])]);
    let got = content_ranges(
        "minimax_m3",
        &[
            "STOP \t]<]minimax[>[<invoke name=\"f\">",
            "]<]minimax[>[</invoke>",
        ],
    )
    .await;
    assert_eq!(got, vec![("STOP".to_string(), vec![0..4])]);
}

#[tokio::test]
async fn a_cut_orphan_closer_keeps_the_bytes_on_both_sides() {
    // Review finding 3: the zero-call release cuts `</tool_call>` out.
    let got = content_ranges("hermes", &[r#"{"x":"a</tool_call>b"}"#]).await;
    assert_eq!(got, vec![(r#"{"x":"ab"}"#.to_string(), vec![0..7, 19..22])]);
    // Split inside the closer, with the same byte on both sides of it: the
    // stock jail releases two chunks here, and the second `b` is dated by its
    // own offset, not the first one's.
    let got = content_ranges("hermes", &[r#"{"x":"b</tool"#, r#"_call>b"}"#]).await;
    assert_eq!(
        got,
        vec![
            (r#"{"x":"b"#.to_string(), vec![0..7]),
            (r#"b"}"#.to_string(), vec![19..22]),
        ]
    );
}

#[tokio::test]
async fn trailing_text_after_a_call_keeps_its_offsets() {
    let call = r#"<tool_call>{"name":"f","arguments":{}}</tool_call>"#;
    let got = content_ranges("hermes", &["Hi ", call, " bye"]).await;
    let after = 3 + call.len();
    assert_eq!(
        got,
        vec![
            ("Hi ".to_string(), vec![0..3]),
            (" bye".to_string(), vec![after..after + 4]),
        ]
    );
}

#[tokio::test]
async fn caller_offsets_and_several_choices() {
    // Two choices, interleaved, each with its own offsets from the caller.
    let inputs = vec![
        text(0, "STOP {"),
        text(1, "plain"),
        text(0, r#""name":"f","arguments":{}}"#),
        text(1, " text"),
        finish(0),
        finish(1),
    ];
    let given = |index: u32, range: Range<usize>| {
        Some(vec![ChoiceSources {
            index,
            content: vec![range],
            reasoning: vec![],
        }])
    };
    let sources = vec![
        given(0, 100..106),
        given(1, 500..505),
        given(0, 106..132),
        given(1, 505..510),
        Some(vec![]),
        Some(vec![]),
    ];
    let mut got: Vec<(u32, String, Vec<Range<usize>>)> = Vec::new();
    for ((chunk, sources), _) in recorded("hermes", &inputs, sources).await {
        let choices = chunk.data.map(|data| data.choices).unwrap_or_default();
        for (choice, source) in choices.iter().zip(sources.unwrap()) {
            let content = content_of(choice);
            if !content.is_empty() {
                got.push((choice.index, content.to_string(), source.content));
            }
        }
    }
    assert_eq!(
        got,
        vec![
            (1, "plain".to_string(), vec![500..505]),
            (0, "STOP".to_string(), vec![100..104]),
            (1, " text".to_string(), vec![505..510]),
        ]
    );
}

#[tokio::test]
async fn reasoning_passes_through_with_its_offsets() {
    let mut first = text(0, "Hi <tool_");
    first.reasoning = Some("think".to_string());
    let mut second = text(0, r#"call>{"name":"f","arguments":{}}</tool_call>"#);
    second.reasoning = Some(" more".to_string());
    let inputs = vec![first, second, finish(0)];
    check("hermes", &inputs, "reasoning").await;
    let sources = vec![None; inputs.len()];
    let mut reasoning = Vec::new();
    for ((chunk, sources), _) in recorded("hermes", &inputs, sources).await {
        let choices = chunk.data.map(|data| data.choices).unwrap_or_default();
        for (choice, source) in choices.iter().zip(sources.unwrap()) {
            if let Some(text) = &choice.delta.reasoning_content {
                reasoning.push((text.clone(), source.reasoning));
            }
        }
    }
    assert_eq!(
        reasoning,
        vec![
            ("think".to_string(), vec![0..5]),
            (" more".to_string(), vec![5..10])
        ]
    );
}

#[tokio::test]
async fn recording_off_carries_no_sources() {
    let call = r#"STOP <tool_call>{"name":"f","arguments":{}}</tool_call> after"#;
    for parser in RECORDING_TOOL_PARSERS {
        let items: Vec<SourcedItem> = [text(0, call), finish(0)]
            .iter()
            .map(|input| (chunk(input), None))
            .collect();
        let out: Vec<SourcedItem> = jail(parser)
            .apply_pairs(false, stream::iter(items))
            .collect()
            .await;
        assert!(!out.is_empty());
        assert!(out.iter().all(|(_, sources)| sources.is_none()), "{parser}");
    }
    // A jail that cannot record ignores the request and carries none.
    let items = vec![(chunk(&text(0, "plain")), None)];
    let out: Vec<SourcedItem> = jail("glm47")
        .apply_pairs(true, stream::iter(items))
        .collect()
        .await;
    assert!(out.iter().all(|(_, sources)| sources.is_none()));
}

#[tokio::test]
async fn partial_marker_splitters_run_without_recording() {
    // `gemma4` and Harmony hold a partial call prefix through their own
    // splitters. They cannot record, so the request falls back to no ranges,
    // and the stock output stays the same.
    let cases: &[(&str, &[&str])] = &[
        (
            "gemma4",
            &["hi <|tool_c", "all>call:f{}<tool_call|>", " bye"],
        ),
        ("gemma4", &["text <|tool", "_call>"]),
        ("harmony", &["answer <", "|start|> tail <"]),
    ];
    for (parser, chunks) in cases {
        let inputs: Vec<In> = chunks
            .iter()
            .map(|c| text(0, c))
            .chain([finish(0)])
            .collect();
        let stock = stock(parser, &inputs).await;
        let items: Vec<SourcedItem> = inputs.iter().map(|input| (chunk(input), None)).collect();
        let asked: Vec<SourcedItem> = jail(parser)
            .apply_pairs(true, stream::iter(items))
            .collect()
            .await;
        assert!(
            asked.iter().all(|(_, sources)| sources.is_none()),
            "{parser}"
        );
        let asked: Vec<Chunk> = asked.into_iter().map(|(chunk, _)| chunk).collect();
        assert_eq!(
            stock_shape(&stock),
            stock_shape(&asked),
            "{parser} {chunks:?}"
        );
    }
}

#[test]
fn only_the_three_marker_parsers_record() {
    for parser in RECORDING_TOOL_PARSERS {
        assert!(jail(parser).can_record_sources(), "{parser}");
    }
    for parser in [
        "kimi_k3",
        "harmony",
        "minimax_m2",
        "gemma4",
        "glm47",
        "qwen25",
    ] {
        assert!(!jail(parser).can_record_sources(), "{parser}");
    }
    assert!(!JailedStream::builder().build().can_record_sources());
    let immediate = JailedStream::builder()
        .tool_call_parser("hermes")
        .tool_choice_required()
        .build();
    assert!(!immediate.can_record_sources());
}
