// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The heap allocations of decoding one chat request message.
//!
//! A message whose first key is `role` decodes in one pass: the only
//! allocations are the message's own strings and vectors. A message that
//! puts `role` later is buffered once, to find the role. This file holds its
//! own test binary because it counts every allocation in the process.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, Ordering};

use dynamo_protocols::types::ChatCompletionRequestMessage;

struct Counting;

static ALLOCATIONS: AtomicU64 = AtomicU64::new(0);

// SAFETY: every call goes to `System` unchanged; the counter does not touch
// the memory handed out.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

/// The allocations of decoding `json` once, after a warm-up decode.
fn allocations(json: &str) -> u64 {
    serde_json::from_str::<ChatCompletionRequestMessage>(json).expect("message");
    let before = ALLOCATIONS.load(Ordering::Relaxed);
    let message = serde_json::from_str::<ChatCompletionRequestMessage>(json).expect("message");
    let after = ALLOCATIONS.load(Ordering::Relaxed);
    drop(message);
    after - before
}

// One test only: a second test thread would allocate during the count.
#[test]
fn message_decode_allocates_only_the_message() {
    for (json, expected) in [
        // The content string.
        (r#"{"role":"user","content":"hi"}"#, 1),
        // The content and the tool call id.
        (r#"{"role":"tool","content":"ok","tool_call_id":"c1"}"#, 2),
        // The tool call vector, its id, name, and arguments, and the content.
        (
            r#"{"role":"assistant","content":"x","tool_calls":[{"id":"c1","type":"function","function":{"name":"f","arguments":"{}"}}]}"#,
            5,
        ),
        // With `role` last, the tagged enum's buffer adds its entry vector,
        // and on the assistant message the tool call's and function's too.
        // The decoder before this one made 3, 4, and 10 in either order.
        (r#"{"content":"hi","role":"user"}"#, 2),
        (r#"{"content":"ok","tool_call_id":"c1","role":"tool"}"#, 3),
        (
            r#"{"content":"x","tool_calls":[{"id":"c1","type":"function","function":{"name":"f","arguments":"{}"}}],"role":"assistant"}"#,
            9,
        ),
    ] {
        assert_eq!(allocations(json), expected, "{json}");
    }
}
