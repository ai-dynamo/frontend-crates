// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![allow(clippy::single_range_in_vec_init)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicIsize, Ordering::Relaxed};

use dynamo_parsers::chat_stream::SourcedChunk;
use dynamo_parsers::reasoning::stream::{
    ReasoningStreamOptions, parse_reasoning_content_from_stream,
};
use dynamo_parsers::sources::ChoiceSources;
use dynamo_parsers::tool_calling::jail::Annotated;
use futures::{StreamExt, stream};
use serde_json::json;

struct Counting;
static LIVE_BYTES: AtomicIsize = AtomicIsize::new(0);

// SAFETY: each operation forwards its unchanged arguments to System.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            LIVE_BYTES.fetch_add(layout.size() as isize, Relaxed);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        LIVE_BYTES.fetch_sub(layout.size() as isize, Relaxed);
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let ptr = unsafe { System.realloc(ptr, layout, new_size) };
        if !ptr.is_null() {
            LIVE_BYTES.fetch_add(new_size as isize - layout.size() as isize, Relaxed);
        }
        ptr
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

fn item(text: &str, at: Option<usize>) -> SourcedChunk {
    SourcedChunk {
        chunk: Annotated {
            data: Some(
                serde_json::from_value(json!({
                    "id": "retention", "object": "chat.completion.chunk", "created": 0,
                    "model": "m", "choices": [{"index": 0, "delta": {"content": text}}],
                }))
                .unwrap(),
            ),
            id: None,
            event: None,
            comment: None,
            error: None,
        },
        sources: Some(
            at.map(|at| ChoiceSources {
                index: 0,
                content: vec![at..at + text.len()],
                reasoning: vec![],
            })
            .into_iter()
            .collect(),
        ),
    }
}

async fn retained_bytes(chunks: usize, gap: bool) -> isize {
    const TEXT: &str = "<think>r</think>a";
    let before = LIVE_BYTES.load(Relaxed);
    let inputs = std::iter::once(item("x", Some(100)))
        .chain(std::iter::once(item("y", (!gap).then_some(200))))
        .chain((0..chunks).map(|i| item(TEXT, Some(1000 + i * TEXT.len()))));
    let mut output = parse_reasoning_content_from_stream(
        stream::iter(inputs),
        "qwen3".to_string(),
        ReasoningStreamOptions::default(),
    )
    .boxed();
    for i in 0..chunks + 2 {
        let item = output.next().await.unwrap();
        if gap && i > 0 {
            assert!(item.sources.as_ref().unwrap().is_empty());
        } else {
            assert_eq!(item.sources.as_ref().unwrap().len(), 1);
        }
        drop(item);
    }
    // Keep the stream open so dropping its parser cannot hide retained runs.
    let retained = LIVE_BYTES.load(Relaxed) - before;
    drop(output);
    retained
}

#[tokio::test]
#[ignore = "F4: source runs accumulate after a gap drops the mapper that drained them"]
async fn review_f4_source_gap_keeps_parser_retention_bounded() {
    retained_bytes(8, false).await;
    let control_short = retained_bytes(16, false).await;
    let control_long = retained_bytes(4096, false).await;
    println!(
        "control retained after 16 chunks: {control_short} bytes; after 4096 chunks: {control_long} bytes"
    );
    assert!(control_long - control_short <= 8192);
    let short = retained_bytes(16, true).await;
    let long = retained_bytes(4096, true).await;
    println!("retained after 16 chunks: {short} bytes; after 4096 chunks: {long} bytes");
    assert!(
        long - short <= 8192,
        "a choice with no source entries retains source runs as the stream grows: short={short}, long={long}"
    );
}
