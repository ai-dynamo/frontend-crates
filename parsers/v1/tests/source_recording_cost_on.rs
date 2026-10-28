// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Cost of recording sources in the v1 reasoning parser, per input chunk.
//! Ignored by default. Compare with the reasoning row of
//! `source_recording_cost.rs` on the same commit:
//!
//! ```text
//! cargo test --release -p dynamo-parsers --test source_recording_cost_on -- --ignored --nocapture
//! ```

mod cost_harness;

use cost_harness::{chars, corpus, median_ns_per_chunk};
use dynamo_parsers::ReasoningParser;
use dynamo_parsers::reasoning::ReasoningParserType;

#[test]
#[ignore = "cost guard: run with --release on a quiet host"]
fn recording_on_cost() {
    let streams: Vec<(String, Vec<String>)> = corpus()
        .reasoning
        .iter()
        .map(|case| (case.parser.clone(), chars(case)))
        .collect();
    let chunks: usize = streams.iter().map(|(_, chunks)| chunks.len()).sum();
    let mut runs = Vec::new();
    let ns = median_ns_per_chunk(chunks, || {
        for (name, stream) in &streams {
            let mut parser = ReasoningParserType::get_reasoning_parser_from_name(name);
            assert!(parser.record_sources());
            for chunk in stream {
                std::hint::black_box(parser.parse_reasoning_streaming_incremental(chunk, &[]));
                parser.take_sources(&mut runs);
                runs.clear();
            }
            std::hint::black_box(parser.finish_reasoning_stream());
            parser.take_sources(&mut runs);
            runs.clear();
        }
    });
    println!("reasoning, recording on: {ns:.1} ns/chunk");
}
