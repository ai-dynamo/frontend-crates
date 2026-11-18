// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use dynamo_parsers_v2::{UnifiedParserOutput, create_unified_parser_for_family, encode_harmony};

struct CountingAllocator;
thread_local! {
    static COUNTING: Cell<bool> = const { Cell::new(false) };
    static ALLOCATED: Cell<usize> = const { Cell::new(0) };
    static LIVE: Cell<isize> = const { Cell::new(0) };
}

fn record(size: usize) {
    if COUNTING.try_with(Cell::get).unwrap_or(false) {
        let _ = ALLOCATED.try_with(|total| total.set(total.get() + size));
        let _ = LIVE.try_with(|total| total.set(total.get() + size as isize));
    }
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record(layout.size());
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        if COUNTING.try_with(Cell::get).unwrap_or(false) {
            let _ = LIVE.try_with(|total| total.set(total.get() - layout.size() as isize));
        }
        unsafe { System.dealloc(pointer, layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        record(layout.size());
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        record(size);
        if COUNTING.try_with(Cell::get).unwrap_or(false) {
            let _ = LIVE.try_with(|total| total.set(total.get() - layout.size() as isize));
        }
        unsafe { System.realloc(pointer, layout, size) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn measured<T>(operation: impl FnOnce() -> T) -> (T, usize) {
    ALLOCATED.with(|total| total.set(0));
    LIVE.with(|total| total.set(0));
    COUNTING.with(|active| active.set(true));
    let result = operation();
    COUNTING.with(|active| active.set(false));
    (result, ALLOCATED.with(Cell::get))
}

fn allocation_sample(prefix_bytes: usize, choices: usize, quoted: bool) -> (usize, usize) {
    let header = "<|start|>assistant to=functions.lookup<|message|>{\"q\":\"";
    let prefix = if quoted {
        format!("{header}<|start|>{}", "q".repeat(prefix_bytes))
    } else {
        format!("{header}{}", "q".repeat(prefix_bytes))
    };
    let ids = encode_harmony(&prefix).unwrap();
    let append = encode_harmony("z").unwrap();
    let close = encode_harmony("\"}<|call|>").unwrap();
    let mut parser = create_unified_parser_for_family("gpt_oss", &[]).unwrap();
    parser.initialize(&[]).unwrap();
    let mut discarded = UnifiedParserOutput::default();
    for chunk in ids.chunks(256) {
        parser.parse_token_ids_into(chunk, &mut discarded).unwrap();
        discarded.events.clear();
    }
    let continuation = parser.continuation().unwrap().unwrap();
    let (mut forks, first_append) = measured(|| {
        (0..choices)
            .map(|_| {
                let mut fork = continuation.fork().unwrap();
                fork.parse_token_ids_into(&append, &mut UnifiedParserOutput::default())
                    .unwrap();
                fork
            })
            .collect::<Vec<_>>()
    });
    let (_, completion) = measured(|| {
        for fork in &mut forks {
            fork.parse_token_ids_into(&close, &mut UnifiedParserOutput::default())
                .unwrap();
            fork.finish().unwrap();
        }
    });
    (first_append, completion)
}

#[test]
fn consumed_body_has_no_per_choice_fork_or_first_append_allocation_slope() {
    // Warm tokenizer and allocator state before comparing consumed-prefix sizes.
    allocation_sample(128, 1, false);
    let choices = 4;
    let small = allocation_sample(1024, choices, false);
    let large = allocation_sample(131_072, choices, false);
    eprintln!(
        "choices={choices} prefix=1024 first_append={} completion={}",
        small.0, small.1
    );
    eprintln!(
        "choices={choices} prefix=131072 first_append={} completion={}",
        large.0, large.1
    );
    assert!(
        large.0 <= small.0 + 4096 * choices,
        "fork/append copied consumed body: {small:?} -> {large:?}"
    );
    assert!(
        large.1 <= small.1 + 4096 * choices,
        "completion copied already emitted body: {small:?} -> {large:?}"
    );
}

#[test]
fn unresolved_quoted_control_is_shared_until_disambiguation() {
    allocation_sample(128, 1, true);
    let choices = 4;
    let small = allocation_sample(1024, choices, true);
    let large = allocation_sample(131_072, choices, true);
    eprintln!("quoted choices={choices} small={small:?} large={large:?}");
    assert!(
        large.0 <= small.0 + 4096 * choices,
        "fork/append copied ambiguous history: {small:?} -> {large:?}"
    );
    // Disambiguation must emit the unresolved literal body. Its output
    // allocation is expected to scale with the bytes that were not emitted.
}

fn retained_choices(tokens: &[u32], choices: usize) -> isize {
    let append = encode_harmony("z").unwrap();
    let mut parser = create_unified_parser_for_family("gpt_oss", &[]).unwrap();
    parser.initialize(&[]).unwrap();
    let mut discarded = UnifiedParserOutput::default();
    for chunk in tokens.chunks(256) {
        parser.parse_token_ids_into(chunk, &mut discarded).unwrap();
        discarded.events.clear();
    }
    let seed = parser.continuation().unwrap().unwrap();
    drop(parser);
    let ((retained, after_drop), allocated) = measured(|| {
        let mut forks = Vec::new();
        for _ in 0..choices {
            let mut fork = seed.fork().unwrap();
            fork.parse_token_ids_into(&append, &mut UnifiedParserOutput::default())
                .unwrap();
            forks.push(fork);
        }
        let retained = LIVE.with(Cell::get);
        drop(forks);
        (retained, LIVE.with(Cell::get))
    });
    assert_eq!(after_drop, 0, "choice-owned memory must be reclaimed");
    eprintln!(
        "tokens={} choices={choices} cohorts=1 allocated={allocated} retained={retained} after_drop={after_drop}",
        tokens.len()
    );
    retained
}

#[test]
fn retained_choice_cost_is_flat_for_original_prefix_and_128k_tokens() {
    let header = "<|start|>assistant to=functions.lookup<|channel|>commentary<|message|>{\"q\":\"";
    let body = "hello world ".repeat(65_536);
    let original = format!("{header}{}", &body[..65_608 - header.len()]);
    assert_eq!(original.len(), 65_608);
    let original = encode_harmony(&original).unwrap();
    let mut full = encode_harmony(&format!("{header}{body}")).unwrap();
    assert!(full.len() >= 131_072);
    full.truncate(131_072);
    retained_choices(&original, 4);
    let small = retained_choices(&original, 4);
    let large = retained_choices(&full, 4);
    assert!(
        large <= small + 4096 * 4,
        "consumed-prefix memory was copied per choice: {small} -> {large}"
    );
}
