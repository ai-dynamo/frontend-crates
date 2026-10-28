// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The v1 route's stock path (the reasoning stream, then the jail under the
//! terminal-error contract, on items with no source slot) must not allocate
//! more often than it did before the stages carried sources.
//!
//! The limits are the counts at the commit that merged the moved stream
//! stages with the parsers' recording (`b4abfd48`, debug build), for the
//! recorded dynamo streams fed one character per chunk. The allocation count
//! may not exceed them, and the bytes may exceed them by at most 1%. Lower
//! the limits when a change makes the route allocate less; raise them only
//! with a wall-time comparison that shows the cost is accepted.

mod cost_harness;

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

use cost_harness::{chars, chunk, corpus};
use dynamo_parsers::reasoning::stream::{
    ReasoningStreamOptions, parse_reasoning_content_from_stream,
};
use dynamo_parsers::tool_calling::jail::{
    apply_jail_terminal_error_contract, apply_tool_calling_jail,
};
use futures::{StreamExt, stream};

struct Counting;

static ALLOCS: AtomicUsize = AtomicUsize::new(0);
static BYTES: AtomicUsize = AtomicUsize::new(0);

// SAFETY: every call forwards to `System` with the same arguments.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Relaxed);
        BYTES.fetch_add(layout.size(), Relaxed);
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, Relaxed);
        BYTES.fetch_add(new_size, Relaxed);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

/// Reasoning parser, tool parser, and the allocations and bytes at `b4abfd48`.
const LIMITS: &[(&str, &str, usize, usize)] = &[
    ("qwen3", "hermes", 26_240, 3_565_397),
    ("basic", "kimi_k2", 33_604, 7_329_660),
    ("minimax_m3", "minimax_m3", 18_058, 4_279_206),
];

/// How far the bytes may exceed the limits, in percent.
const BYTES_SLACK_PERCENT: usize = 1;

#[test]
fn stock_v1_route_allocates_no_more_than_before() {
    let corpus = corpus();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    let mut over = Vec::new();
    for (reasoning, tool, max_allocs, max_bytes) in LIMITS {
        // The tool corpus holds the raw model text of each family's route.
        let streams: Vec<Vec<_>> = corpus
            .tool
            .iter()
            .filter(|case| case.parser == *tool)
            .map(|case| chars(case).iter().map(|c| chunk(c)).collect())
            .collect();
        let (allocs, bytes) = (ALLOCS.load(Relaxed), BYTES.load(Relaxed));
        runtime.block_on(async {
            for items in streams {
                let reasoned = parse_reasoning_content_from_stream(
                    stream::iter(items),
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
                drop(out);
            }
        });
        let allocs = ALLOCS.load(Relaxed) - allocs;
        let bytes = BYTES.load(Relaxed) - bytes;
        println!("{reasoning} + {tool}: {allocs} allocations, {bytes} bytes");
        if allocs > *max_allocs || bytes > max_bytes + max_bytes * BYTES_SLACK_PERCENT / 100 {
            over.push(format!(
                "{reasoning} + {tool}: {allocs} allocations (limit {max_allocs}), {bytes} bytes (limit {max_bytes})"
            ));
        }
    }
    assert!(
        over.is_empty(),
        "the stock v1 route allocates more than before: {over:?}"
    );
}
