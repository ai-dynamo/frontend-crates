// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use dynamo_parsers_v2::{
    Tool, UnifiedEvent, UnifiedParserExt, assemble, create_unified_parser_for_family,
};
use serde_json::json;

struct CountingAllocator;
thread_local! {
    static COUNTING: Cell<bool> = const { Cell::new(false) };
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

fn record() {
    if COUNTING.try_with(Cell::get).unwrap_or(false) {
        let _ = ALLOCATIONS.try_with(|count| count.set(count.get() + 1));
    }
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record();
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        record();
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        record();
        unsafe { System.realloc(pointer, layout, size) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn wire(family: &str, value: &str) -> String {
    match family {
        "glm47" => format!(
            "<tool_call>record<arg_key>text</arg_key><arg_value>{value}</arg_value></tool_call>"
        ),
        _ => {
            let n = "]<]minimax[>[";
            format!(
                "{n}<tool_call>{n}<invoke name=\"record\">{n}<text>{value}{n}</text>{n}</invoke>{n}</tool_call>"
            )
        }
    }
}

// Allocations to parse one call whose string-only argument holds `value`.
fn allocations(family: &str, value: &str) -> usize {
    let tools = [Tool {
        name: "record".into(),
        description: None,
        parameters: json!({"type": "object", "properties": {"text": {"type": "string"}}}),
        strict: None,
    }];
    let input = wire(family, value);
    let mut parser = create_unified_parser_for_family(family, &tools).unwrap();
    ALLOCATIONS.with(|count| count.set(0));
    COUNTING.with(|active| active.set(true));
    let mut events = parser.push(&input).unwrap();
    events.extend(parser.finish().unwrap().events);
    COUNTING.with(|active| active.set(false));
    assert_eq!(
        assemble(&events),
        [UnifiedEvent::ToolCall {
            name: "record".into(),
            arguments: json!({"text": value}),
        }],
        "{family}"
    );
    ALLOCATIONS.with(Cell::get)
}

#[test]
fn string_only_arguments_never_decode_json_text() {
    // 262,145 bytes: an array of 32,768 objects, against plain text of equal length.
    let array = format!("[{}]", vec![r#"{"x":1}"#; 32_768].join(","));
    let plain = "x".repeat(array.len());
    for family in ["glm47", "minimax_m3"] {
        allocations(family, "warm");
        let json_text = allocations(family, &array);
        let plain_text = allocations(family, &plain);
        eprintln!("{family}: json_text={json_text} plain_text={plain_text}");
        assert!(
            json_text <= plain_text + 64,
            "{family} decoded a string argument: {json_text} vs {plain_text} allocations"
        );
    }
}
