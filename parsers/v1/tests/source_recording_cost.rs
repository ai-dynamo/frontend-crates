// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Cost of source recording, per input chunk. Ignored by default.
//!
//! It uses only the public API. For a before and after comparison of the
//! stock path (recording off), run it on a base commit with the
//! `recording on` row removed, since the base has no `record_sources`:
//!
//! ```text
//! cargo test --release -p dynamo-parsers --test source_recording_cost -- --ignored --nocapture
//! ```
//!
//! Rows: the v1 reasoning parser with recording off and on, and the
//! tool-call jail (`apply`, recording off, including one clone of each input
//! chunk), over the recorded dynamo streams fed one character per chunk, as
//! the endpoint feeds one token per chunk.

use std::time::{Duration, Instant};

use dynamo_parsers::ReasoningParser;
use dynamo_parsers::reasoning::ReasoningParserType;
use dynamo_parsers::tool_calling::jail::{JailedStream, annotated::Annotated};
use dynamo_protocols::types::CreateChatCompletionStreamResponse;
use futures::{StreamExt, stream};
use serde::Deserialize;

const CORPUS: &str = include_str!("data/source_ranges_corpus.json");
const ROUNDS: usize = 15;

#[derive(Deserialize)]
struct Corpus {
    reasoning: Vec<Case>,
    tool: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    parser: String,
    chunks: Vec<String>,
}

fn chars(case: &Case) -> Vec<String> {
    case.chunks.concat().chars().map(String::from).collect()
}

/// Median over `ROUNDS` of `run`, in nanoseconds per chunk.
fn median_ns_per_chunk(chunks: usize, mut run: impl FnMut()) -> f64 {
    let mut times: Vec<Duration> = (0..ROUNDS)
        .map(|_| {
            let start = Instant::now();
            run();
            start.elapsed()
        })
        .collect();
    times.sort();
    times[ROUNDS / 2].as_nanos() as f64 / chunks as f64
}

fn reasoning_row(corpus: &Corpus, record: bool) -> f64 {
    let streams: Vec<(String, Vec<String>)> = corpus
        .reasoning
        .iter()
        .map(|case| (case.parser.clone(), chars(case)))
        .collect();
    let chunks: usize = streams.iter().map(|(_, chunks)| chunks.len()).sum();
    let mut runs = Vec::new();
    median_ns_per_chunk(chunks, || {
        for (name, stream) in &streams {
            let mut parser = ReasoningParserType::get_reasoning_parser_from_name(name);
            if record {
                assert!(parser.record_sources());
            }
            for chunk in stream {
                std::hint::black_box(parser.parse_reasoning_streaming_incremental(chunk, &[]));
                parser.take_sources(&mut runs);
                runs.clear();
            }
            std::hint::black_box(parser.finish_reasoning_stream());
        }
    })
}

fn chunk(text: &str) -> Annotated<CreateChatCompletionStreamResponse> {
    Annotated {
        data: Some(
            serde_json::from_value(serde_json::json!({
                "id": "cost", "object": "chat.completion.chunk", "created": 0, "model": "m",
                "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": null}]
            }))
            .unwrap(),
        ),
        id: None,
        event: None,
        comment: None,
        error: None,
    }
}

fn jail_row(corpus: &Corpus) -> f64 {
    let streams: Vec<(String, Vec<Annotated<CreateChatCompletionStreamResponse>>)> = corpus
        .tool
        .iter()
        .map(|case| {
            (
                case.parser.clone(),
                chars(case).iter().map(|c| chunk(c)).collect(),
            )
        })
        .collect();
    let chunks: usize = streams.iter().map(|(_, chunks)| chunks.len()).sum();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    median_ns_per_chunk(chunks, || {
        runtime.block_on(async {
            for (parser, items) in &streams {
                let jail = JailedStream::builder().tool_call_parser(parser).build();
                let out: Vec<_> = jail.apply(stream::iter(items.clone())).collect().await;
                std::hint::black_box(out);
            }
        })
    })
}

#[test]
#[ignore = "cost guard: run with --release on a quiet host"]
fn source_recording_cost() {
    let corpus: Corpus = serde_json::from_str(CORPUS).unwrap();
    println!(
        "reasoning, recording off: {:.1} ns/chunk",
        reasoning_row(&corpus, false)
    );
    println!(
        "reasoning, recording on:  {:.1} ns/chunk",
        reasoning_row(&corpus, true)
    );
    println!(
        "jail apply, recording off: {:.1} ns/chunk",
        jail_row(&corpus)
    );
}
