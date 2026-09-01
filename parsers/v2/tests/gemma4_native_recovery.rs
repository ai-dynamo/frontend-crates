// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Gemma 4 native-mode recovery through both public parser adapters.

use dynamo_parsers_v2::tool_calling::create_tool_parser_for_family;
use dynamo_parsers_v2::{
    Tool, ToolParseResult, UnifiedEvent, UnifiedParserExt, UnifiedParserInit, assemble,
};

#[cfg(feature = "test-utils")]
use dynamo_parsers_v2::tool_calling::gemma4::{
    boundary_examined_bytes, reset_boundary_examined_bytes,
};

fn weather_tools() -> Vec<Tool> {
    vec![Tool {
        name: "get_weather".into(),
        description: None,
        parameters: serde_json::json!({
            "type": "object",
            "properties": { "city": { "type": "string" } }
        }),
        strict: None,
    }]
}

fn chunkings(input: &str) -> Vec<Vec<&str>> {
    let mut chunks = vec![vec![input]];
    chunks.extend(
        (1..input.len())
            .filter(|&at| input.is_char_boundary(at))
            .map(|at| vec![&input[..at], &input[at..]]),
    );
    chunks
}

fn assert_tool_only_at_every_split(input: &str, expected_text: &str) {
    let tools = weather_tools();
    for chunks in chunkings(input) {
        let mut parser = create_tool_parser_for_family("gemma4", &tools).expect("Gemma parser");
        let mut output = ToolParseResult::default();
        for chunk in &chunks {
            output.append(parser.push(chunk).expect("push"));
        }
        output.append(parser.finish().expect("finish"));
        let output = output.coalesce_calls();
        assert_eq!(output.normal_text, expected_text, "chunks={chunks:?}");
        assert_eq!(output.calls.len(), 1, "chunks={chunks:?}");
        assert_eq!(output.calls[0].name.as_deref(), Some("get_weather"));
        assert_eq!(output.calls[0].arguments, r#"{"city":"NYC"}"#);
    }
}

fn assert_unified_at_every_split(input: &str, expected_text: &str) {
    let tools = weather_tools();
    let expected = if expected_text.is_empty() {
        vec![UnifiedEvent::ToolCall {
            name: "get_weather".into(),
            arguments: serde_json::json!({"city": "NYC"}),
        }]
    } else {
        vec![
            UnifiedEvent::Text {
                text: expected_text.into(),
            },
            UnifiedEvent::ToolCall {
                name: "get_weather".into(),
                arguments: serde_json::json!({"city": "NYC"}),
            },
        ]
    };

    for chunks in chunkings(input) {
        let mut parser = dynamo_parsers_v2::create_unified_parser_for_family("gemma4", &tools)
            .expect("Gemma unified parser");
        parser
            .initialize_request(UnifiedParserInit::default())
            .expect("native request");
        let mut events = Vec::new();
        for chunk in &chunks {
            events.extend(parser.push(chunk).expect("push"));
        }
        events.extend(parser.finish().expect("finish").events);
        assert_eq!(assemble(&events), expected, "chunks={chunks:?}");
    }
}

fn assert_both_adapters(input: &str, expected_text: &str) {
    assert_tool_only_at_every_split(input, expected_text);
    assert_unified_at_every_split(input, expected_text);
}

fn assert_both_adapters_at_chunk_sizes(input: &str, chunk_sizes: &[usize], expected_calls: usize) {
    let tools = weather_tools();
    for &chunk_size in chunk_sizes {
        let chunks: Vec<_> = input
            .as_bytes()
            .chunks(chunk_size)
            .map(|chunk| std::str::from_utf8(chunk).expect("ASCII Gemma fixture"))
            .collect();

        let mut tool_only = create_tool_parser_for_family("gemma4", &tools).expect("Gemma parser");
        let mut tool_output = ToolParseResult::default();
        for chunk in &chunks {
            tool_output.append(tool_only.push(chunk).expect("tool-only push"));
        }
        tool_output.append(tool_only.finish().expect("tool-only finish"));
        assert_eq!(
            tool_output.coalesce_calls().calls.len(),
            expected_calls,
            "tool-only chunk_size={chunk_size}"
        );

        let mut unified = dynamo_parsers_v2::create_unified_parser_for_family("gemma4", &tools)
            .expect("Gemma unified parser");
        unified
            .initialize_request(UnifiedParserInit::default())
            .expect("native request");
        let mut events = Vec::new();
        for chunk in &chunks {
            events.extend(unified.push(chunk).expect("unified push"));
        }
        events.extend(unified.finish().expect("unified finish").events);
        let call_count = assemble(&events)
            .into_iter()
            .filter(|event| matches!(event, UnifiedEvent::ToolCall { .. }))
            .count();
        assert_eq!(
            call_count, expected_calls,
            "unified chunk_size={chunk_size}"
        );
    }
}

#[test]
fn later_balanced_call_without_wrapper_close_recovers_at_eof_at_each_chunk_size() {
    let malformed = "<|tool_call>call:broken{note:<|\"|>unterminated";
    let later_balanced = "<|\"|>}<|tool_call>call:get_weather{city:<|\"|>NYC<|\"|>}";
    let input = format!("{malformed}{later_balanced}");
    assert_both_adapters_at_chunk_sizes(&input, &[1, 4, 16], 1);
}

#[test]
fn prose_call_prefix_does_not_capture_a_later_wrapped_call() {
    let input = concat!(
        "I will call: you tomorrow",
        "<|tool_call>call:get_weather{city:<|\"|>NYC<|\"|>}<tool_call|>",
    );
    assert_both_adapters(input, "I will call: you tomorrow");
}

#[test]
fn malformed_block_resynchronizes_to_a_later_valid_block() {
    let input = concat!(
        "<|tool_call>call:broken{city:<|\"|>Paris<|\"|>}",
        "<|tool_call>call:get_weather{city:<|\"|>NYC<|\"|>}<tool_call|>",
    );
    assert_both_adapters(input, "");
}

#[test]
fn incomplete_intermediate_block_does_not_hide_a_later_valid_block() {
    let input = concat!(
        "<|tool_call>call:broken{city:<|\"|>Paris<|\"|>}",
        "<|tool_call>call:still_broken{nested:{",
        "<|tool_call>call:get_weather{city:<|\"|>NYC<|\"|>}<tool_call|>",
    );
    assert_both_adapters(input, "");
}

#[test]
fn string_data_cannot_become_a_resynchronization_target() {
    let input = concat!(
        "<|tool_call>call:broken{note:<|\"|>",
        "<|tool_call>call:get_weather{city:<|\"|>TRAP<|\"|>}<tool_call|>",
        "<|\"|>}",
        "<|tool_call>call:get_weather{city:<|\"|>NYC<|\"|>}<tool_call|>",
    );
    assert_both_adapters(input, "");
}

#[test]
fn long_incremental_invokes_preserve_both_adapter_contracts() {
    let value = "x".repeat(32 * 1024);
    let valid = format!("<|tool_call>call:get_weather{{city:<|\"|>{value}<|\"|>}}<tool_call|>");
    let incomplete = format!("<|tool_call>call:get_weather{{city:<|\"|>{value}");
    assert_both_adapters_at_chunk_sizes(&valid, &[4, 16], 1);
    assert_both_adapters_at_chunk_sizes(&incomplete, &[4, 16], 0);
}

#[test]
fn repeated_unmatched_wrappers_recover_the_later_complete_call() {
    let malformed = "<|tool_call>call:broken{note:<|\"|>unterminated<|\"|>".repeat(256);
    let input =
        format!("{malformed}<|tool_call>call:get_weather{{city:<|\"|>NYC<|\"|>}}<tool_call|>");
    assert_both_adapters_at_chunk_sizes(&input, &[4, 16], 1);
}

/// This cannot be represented by the shared corpus: its assertion is a bound on
/// scanner work, not parser output. Exercise each exported adapter factory so a
/// future wiring change cannot bypass Gemma's request-local resynchronizer.
#[cfg(feature = "test-utils")]
#[test]
fn repeated_closers_in_an_unterminated_string_resynchronize_in_linear_time() {
    for chunk_size in [1, 7, 64] {
        let mut tool_only_scans = Vec::new();
        let mut unified_scans = Vec::new();

        for repeats in [4 * 1024, 8 * 1024] {
            let input = format!(
                "<|tool_call>call:broken{{note:<|\"|>{}",
                "<tool_call|>".repeat(repeats)
            );
            let chunks: Vec<_> = input
                .as_bytes()
                .chunks(chunk_size)
                .map(|chunk| std::str::from_utf8(chunk).expect("ASCII Gemma fixture"))
                .collect();

            reset_boundary_examined_bytes();
            let mut tool_only = create_tool_parser_for_family("gemma4", &weather_tools())
                .expect("Gemma tool-only parser");
            let mut output = ToolParseResult::default();
            for chunk in &chunks {
                output.append(tool_only.push(chunk).expect("tool-only push"));
            }
            output.append(tool_only.finish().expect("tool-only finish"));
            assert!(
                output.coalesce_calls().calls.is_empty(),
                "unterminated string cannot recover a call at chunk_size={chunk_size}"
            );
            tool_only_scans.push(boundary_examined_bytes());

            reset_boundary_examined_bytes();
            let mut unified =
                dynamo_parsers_v2::create_unified_parser_for_family("gemma4", &weather_tools())
                    .expect("Gemma unified parser");
            unified
                .initialize_request(UnifiedParserInit::default())
                .expect("native request");
            let mut events = Vec::new();
            for chunk in &chunks {
                events.extend(unified.push(chunk).expect("unified push"));
            }
            events.extend(unified.finish().expect("unified finish").events);
            assert!(
                assemble(&events).is_empty(),
                "unterminated string cannot recover unified events at chunk_size={chunk_size}"
            );
            unified_scans.push(boundary_examined_bytes());
        }

        for (adapter, scans) in [("tool-only", tool_only_scans), ("unified", unified_scans)] {
            let [at_n, at_2n] = scans.as_slice() else {
                unreachable!("the N/2N measurement always has two inputs");
            };
            assert!(
                *at_2n <= *at_n * 3,
                "{adapter} rescanned superlinearly: N={at_n}, 2N={at_2n}, chunk_size={chunk_size}"
            );
        }
    }
}
