// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Cost of the v1 route's stock path, per input chunk. Ignored by default.
//!
//! The reasoning stream, then the jail under the terminal-error contract, on
//! items with no source slot, for Rhino's three v1 families, over the recorded
//! dynamo streams fed one character per chunk. It uses only API that the
//! commit before the stages carried sources also has, so it runs unchanged
//! there for a before and after comparison:
//!
//! ```text
//! cargo test --release -p dynamo-parsers --test v1_route_cost -- --ignored --nocapture
//! ```

mod cost_harness;

use cost_harness::{chars, chunk, corpus, median_ns_per_chunk};
use dynamo_parsers::reasoning::stream::{
    ReasoningStreamOptions, parse_reasoning_content_from_stream,
};
use dynamo_parsers::tool_calling::jail::{
    apply_jail_terminal_error_contract, apply_tool_calling_jail,
};
use futures::{StreamExt, stream};

#[test]
#[ignore = "cost guard: run with --release on a quiet host"]
fn v1_route_cost() {
    let corpus = corpus();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    for (reasoning, tool) in [
        ("qwen3", "hermes"),
        ("basic", "kimi_k2"),
        ("minimax_m3", "minimax_m3"),
    ] {
        let streams: Vec<Vec<_>> = corpus
            .tool
            .iter()
            .filter(|case| case.parser == tool)
            .map(|case| chars(case).iter().map(|c| chunk(c)).collect())
            .collect();
        let chunks: usize = streams.iter().map(Vec::len).sum();
        let ns = median_ns_per_chunk(chunks, || {
            runtime.block_on(async {
                for items in &streams {
                    let reasoned = parse_reasoning_content_from_stream(
                        stream::iter(items.clone()),
                        reasoning.to_string(),
                        ReasoningStreamOptions::default(),
                    )
                    .boxed();
                    let parser = Some(tool.to_string());
                    let out: Vec<_> = apply_jail_terminal_error_contract(reasoned, move |s| {
                        apply_tool_calling_jail(parser, None, None, false, s)
                    })
                    .collect()
                    .await;
                    std::hint::black_box(out);
                }
            })
        });
        println!("v1 route {reasoning} + {tool}, stock: {ns:.1} ns/chunk");
    }
}
