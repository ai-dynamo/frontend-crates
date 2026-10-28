// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Cost of the stock path (recording off), per input chunk. Ignored by default.
//!
//! It uses only API that the base commit also has, so this file and
//! `cost_harness/` run unchanged on a base commit for a before and after
//! comparison:
//!
//! ```text
//! cargo test --release -p dynamo-parsers --test source_recording_cost -- --ignored --nocapture
//! ```
//!
//! Rows: the v1 reasoning parser, and the tool-call jail's `apply` (including
//! one clone of each input chunk), over the recorded dynamo streams fed one
//! character per chunk. `source_recording_cost_on.rs` has the recording-on row.

mod cost_harness;

use cost_harness::{chars, chunk, corpus, median_ns_per_chunk};
use dynamo_parsers::ReasoningParser;
use dynamo_parsers::reasoning::ReasoningParserType;
use dynamo_parsers::tool_calling::jail::JailedStream;
use futures::{StreamExt, stream};

#[test]
#[ignore = "cost guard: run with --release on a quiet host"]
fn stock_path_cost() {
    let corpus = corpus();

    let streams: Vec<(String, Vec<String>)> = corpus
        .reasoning
        .iter()
        .map(|case| (case.parser.clone(), chars(case)))
        .collect();
    let chunks: usize = streams.iter().map(|(_, chunks)| chunks.len()).sum();
    let reasoning = median_ns_per_chunk(chunks, || {
        for (name, stream) in &streams {
            let mut parser = ReasoningParserType::get_reasoning_parser_from_name(name);
            for chunk in stream {
                std::hint::black_box(parser.parse_reasoning_streaming_incremental(chunk, &[]));
            }
            std::hint::black_box(parser.finish_reasoning_stream());
        }
    });

    let streams: Vec<(String, Vec<_>)> = corpus
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
    let jail = median_ns_per_chunk(chunks, || {
        runtime.block_on(async {
            for (parser, items) in &streams {
                let jail = JailedStream::builder().tool_call_parser(parser).build();
                let out: Vec<_> = jail.apply(stream::iter(items.clone())).collect().await;
                std::hint::black_box(out);
            }
        })
    });

    println!("reasoning, recording off: {reasoning:.1} ns/chunk");
    println!("jail apply, recording off: {jail:.1} ns/chunk");
}
