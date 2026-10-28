// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The tool-call jail's stock path (`apply`, no source recording) must not
//! allocate more often than it did before recording existed.
//!
//! The limits are the counts that `main` (`4a161dff`, debug build) gives for
//! the same input: the recorded dynamo streams, fed one character per chunk.
//! The allocation count may not exceed them. The bytes may exceed them by at
//! most 1%: the per-choice state is 136 bytes larger (608 to 744), allocated
//! once per stream, and each output batch is collected as items with a source
//! slot (376 bytes, not 352). Neither adds an allocation. Lower the limits
//! when a change makes the jail allocate less; raise them only with a
//! wall-time comparison that shows the cost is accepted.

mod cost_harness;

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

use cost_harness::{chars, chunk, corpus};
use dynamo_parsers::tool_calling::jail::JailedStream;
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

/// Parser, and the allocations and bytes that `main` (`4a161dff`) makes.
const LIMITS: &[(&str, usize, usize)] = &[
    ("hermes", 21_091, 3_363_570),
    ("kimi_k2", 27_129, 7_112_905),
    ("minimax_m3", 17_227, 4_320_181),
];

/// How far the bytes may exceed `main`, in percent.
const BYTES_SLACK_PERCENT: usize = 1;

#[test]
fn stock_jail_allocates_no_more_than_main() {
    let corpus = corpus();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    let mut over = Vec::new();
    for (parser, max_allocs, max_bytes) in LIMITS {
        let streams: Vec<Vec<_>> = corpus
            .tool
            .iter()
            .filter(|case| case.parser == *parser)
            .map(|case| chars(case).iter().map(|c| chunk(c)).collect())
            .collect();
        let (allocs, bytes) = (ALLOCS.load(Relaxed), BYTES.load(Relaxed));
        runtime.block_on(async {
            for items in streams {
                let jail = JailedStream::builder().tool_call_parser(*parser).build();
                let out: Vec<_> = jail.apply(stream::iter(items)).collect().await;
                drop(out);
            }
        });
        let allocs = ALLOCS.load(Relaxed) - allocs;
        let bytes = BYTES.load(Relaxed) - bytes;
        println!("{parser}: {allocs} allocations, {bytes} bytes");
        if allocs > *max_allocs || bytes > max_bytes + max_bytes * BYTES_SLACK_PERCENT / 100 {
            over.push(format!(
                "{parser}: {allocs} allocations (main {max_allocs}), {bytes} bytes (main {max_bytes})"
            ));
        }
    }
    assert!(
        over.is_empty(),
        "the stock jail allocates more than main: {over:?}"
    );
}
