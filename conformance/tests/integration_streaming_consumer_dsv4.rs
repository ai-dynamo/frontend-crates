// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration check that a streaming consumer reconstructs the complete
//! DeepSeek-V4 DSML tool call from the v2 parser's two-frame output.
//!
//! The v2 stream parser emits one tool call as two OpenAI streaming deltas: a
//! name-only delta the moment the `<｜DSML｜invoke name="...">` header closes,
//! then an arguments-only delta when `</｜DSML｜invoke>` closes (SGLang/vLLM-style
//! name-first wire shape). `parity_toolcalling_stream` already locks the raw
//! per-chunk emit against fixtures. This test goes one step further and proves a
//! *streaming consumer* — the OpenAI client-side delta aggregation, over the real
//! `dynamo-protocols` wire types — merges those two frames back into a single
//! complete call. It deliberately does NOT use the parser's own
//! `coalesce_calls()`, so the merge it exercises is the standard wire-protocol
//! one a downstream client performs, not the parser's internal helper.
//!
//! Scope: this is the parser-plus-consumer contract — not a true
//! end-to-end test (no serving stack, no HTTP boundary, no real client; the
//! "consumer" is the in-process delta-merge helper below). A full HTTP e2e
//! against dynamo's serving stack is tracked in ai-dynamo/dynamo#10856 and is
//! blocked on wiring the v2 parser into dynamo's streaming path (dynamo currently
//! consumes only the v1 jail parser).

use dynamo_parsers_v2::{DeepSeekV4ToolStreamParser, ToolCallDelta, ToolParseResult, ToolParser};
use dynamo_protocols::types::{
    ChatCompletionMessageToolCall, ChatCompletionMessageToolCallChunk, FunctionCall,
    FunctionCallStream, FunctionType,
};
use std::collections::BTreeMap;

/// Drive the real v2 DSv4 stream parser over `chunks` and return every emitted
/// delta in arrival order — the raw on-the-wire frame sequence, not coalesced.
fn stream_frames(chunks: &[&str]) -> ToolParseResult {
    let mut parser = DeepSeekV4ToolStreamParser::new();
    let mut out = ToolParseResult::default();
    for chunk in chunks {
        out.append(parser.push(chunk).expect("push"));
    }
    out.append(parser.finish().expect("finish"));
    out
}

/// Map one parser delta onto the OpenAI streaming wire chunk a serving layer
/// would send. The serving layer mints a tool-call id and stamps the type on the
/// first frame (the one carrying the name); later argument-only frames carry no
/// id/name, only an `arguments` fragment.
fn to_wire_chunk(delta: &ToolCallDelta) -> ChatCompletionMessageToolCallChunk {
    let first_frame = delta.name.is_some();
    ChatCompletionMessageToolCallChunk {
        index: delta.tool_index as u32,
        id: first_frame.then(|| format!("call_{}", delta.tool_index)),
        r#type: first_frame.then_some(FunctionType::Function),
        function: Some(FunctionCallStream {
            name: delta.name.clone(),
            arguments: Some(delta.arguments.clone()),
        }),
    }
}

/// Aggregate a stream of OpenAI tool-call delta chunks into complete tool calls
/// the way an OpenAI client (or any SSE consumer) does: key by `index`, keep the
/// first `id`/`name` seen, and concatenate `arguments` fragments in arrival
/// order.
fn aggregate(chunks: &[ChatCompletionMessageToolCallChunk]) -> Vec<ChatCompletionMessageToolCall> {
    // (id, name, arguments) accumulated per tool-call index.
    let mut acc: BTreeMap<u32, (Option<String>, Option<String>, String)> = BTreeMap::new();
    let mut order: Vec<u32> = Vec::new();
    for c in chunks {
        let entry = acc.entry(c.index).or_insert_with(|| {
            order.push(c.index);
            (None, None, String::new())
        });
        if entry.0.is_none() {
            entry.0 = c.id.clone();
        }
        if let Some(function) = &c.function {
            if entry.1.is_none() {
                entry.1 = function.name.clone();
            }
            if let Some(args) = &function.arguments {
                entry.2.push_str(args);
            }
        }
    }
    order
        .into_iter()
        .map(|index| {
            let (id, name, arguments) = acc.remove(&index).expect("index seen");
            ChatCompletionMessageToolCall {
                id: id.unwrap_or_default(),
                r#type: FunctionType::Function,
                function: FunctionCall {
                    name: name.unwrap_or_default(),
                    arguments,
                },
            }
        })
        .collect()
}

#[test]
fn dsv4_two_frame_stream_aggregates_to_single_call() {
    // Same happy-path chunking as deepseek_v4/TOOLCALLING.streamv2.1.
    let frames = stream_frames(&[
        "<｜DSML｜tool_calls> <｜DSML｜invoke",
        " name=\"get_weather\">",
        " <｜DSML｜parameter name=\"location\" string=\"true\">",
        "NYC</｜DSML｜parameter> </｜DSML｜invoke>",
        " </｜DSML｜tool_calls>",
    ]);

    // 1. The parser emits exactly two frames: name-first (empty args), then args-only.
    assert_eq!(frames.normal_text, "");
    assert_eq!(
        frames.calls.len(),
        2,
        "expected name-first then args-only frames, got {:?}",
        frames.calls
    );
    assert_eq!(frames.calls[0].tool_index, 0);
    assert_eq!(frames.calls[0].name.as_deref(), Some("get_weather"));
    assert_eq!(frames.calls[0].arguments, "");
    assert_eq!(frames.calls[1].tool_index, 0);
    assert_eq!(frames.calls[1].name, None);
    assert_eq!(frames.calls[1].arguments, r#"{"location":"NYC"}"#);

    // 2. Map the frames onto OpenAI wire chunks and aggregate them the way a real
    //    streaming consumer does (standard merge, not the parser's coalesce_calls).
    let wire: Vec<ChatCompletionMessageToolCallChunk> =
        frames.calls.iter().map(to_wire_chunk).collect();
    let calls = aggregate(&wire);

    // 3. The consumer reconstructs one complete, well-formed call from two frames.
    assert_eq!(calls.len(), 1, "two frames must merge into one call");
    assert_eq!(calls[0].id, "call_0");
    assert!(matches!(calls[0].r#type, FunctionType::Function));
    assert_eq!(calls[0].function.name, "get_weather");
    let args: serde_json::Value =
        serde_json::from_str(&calls[0].function.arguments).expect("aggregated arguments are JSON");
    assert_eq!(args, serde_json::json!({ "location": "NYC" }));
}
