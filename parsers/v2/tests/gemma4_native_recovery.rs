// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Gemma 4 native-mode recovery through both public parser adapters.

use dynamo_parsers_v2::tool_calling::create_tool_parser_for_family;
use dynamo_parsers_v2::{
    Tool, ToolParseResult, ToolParser, UnifiedEvent, UnifiedParser, UnifiedParserExt,
    UnifiedParserInit, assemble,
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

fn weather_echo_tools() -> Vec<Tool> {
    vec![Tool {
        name: "echo".into(),
        description: None,
        parameters: serde_json::json!({
            "type": "object",
            "properties": { "value": { "type": "string" } },
            "required": ["value"]
        }),
        strict: None,
    }]
}

fn chunkings(input: &str) -> Vec<Vec<&str>> {
    let mut chunks = vec![vec![input]];
    chunks.extend(
        (0..=input.len())
            .filter(|&at| input.is_char_boundary(at))
            .map(|at| vec![&input[..at], &input[at..]]),
    );
    chunks.push(
        input
            .char_indices()
            .map(|(at, ch)| &input[at..at + ch.len_utf8()])
            .collect(),
    );
    chunks
}

fn drive_tool(parser: &mut dyn ToolParser, chunks: &[&str]) -> ToolParseResult {
    let mut output = ToolParseResult::default();
    for chunk in chunks {
        output.append(parser.push(chunk).expect("tool push"));
    }
    output.append(parser.finish().expect("tool finish"));
    output.coalesce_calls()
}

fn drive_unified(parser: &mut dyn UnifiedParser, chunks: &[&str]) -> Vec<UnifiedEvent> {
    let mut events = Vec::new();
    for chunk in chunks {
        events.extend(parser.push(chunk).expect("unified push"));
    }
    events.extend(parser.finish().expect("unified finish").events);
    assemble(&events)
}

fn assert_adapters(
    family: &str,
    tools: &[Tool],
    chunks: &[&str],
    expected_text: &str,
    expected_calls: &[(&str, serde_json::Value)],
) {
    if family == "deepseek_v41" {
        assert!(create_tool_parser_for_family(family, tools).is_err());
    } else {
        let mut tool = create_tool_parser_for_family(family, tools).expect("tool parser");
        let output = drive_tool(tool.as_mut(), chunks);
        assert_eq!(
            output.normal_text, expected_text,
            "family={family}, chunks={chunks:?}"
        );
        let calls: Vec<_> = output
            .calls
            .iter()
            .enumerate()
            .map(|(index, call)| {
                assert_eq!(call.tool_index, index);
                assert!(call.complete);
                (
                    call.name.as_deref().expect("tool name"),
                    serde_json::from_str::<serde_json::Value>(&call.arguments).expect("arguments"),
                )
            })
            .collect();
        assert_eq!(calls, expected_calls, "family={family}, chunks={chunks:?}");
    }

    let mut unified =
        dynamo_parsers_v2::create_unified_parser_for_family(family, tools).expect("unified parser");
    unified
        .initialize_request(UnifiedParserInit::default())
        .expect("native request");
    let events = drive_unified(unified.as_mut(), chunks);
    let mut expected = Vec::new();
    if !expected_text.is_empty() {
        expected.push(UnifiedEvent::Text {
            text: expected_text.into(),
        });
    }
    expected.extend(
        expected_calls
            .iter()
            .map(|(name, arguments)| UnifiedEvent::ToolCall {
                name: (*name).into(),
                arguments: arguments.clone(),
            }),
    );
    assert_eq!(events, expected, "family={family}, chunks={chunks:?}");
}

fn assert_both_adapters(input: &str, expected_text: &str) {
    for chunks in chunkings(input) {
        assert_adapters(
            "gemma4",
            &weather_tools(),
            &chunks,
            expected_text,
            &[("get_weather", serde_json::json!({"city": "NYC"}))],
        );
    }
}

fn assert_no_calls_at_every_split(input: &str) {
    for chunks in chunkings(input) {
        assert_adapters("gemma4", &weather_tools(), &chunks, "", &[]);
    }
}

fn assert_both_adapters_at_chunk_sizes(
    input: &str,
    chunk_sizes: &[usize],
    expected_calls: &[(&str, serde_json::Value)],
) {
    for &chunk_size in chunk_sizes {
        let chunks: Vec<_> = input
            .as_bytes()
            .chunks(chunk_size)
            .map(|chunk| std::str::from_utf8(chunk).expect("ASCII Gemma fixture"))
            .collect();
        assert_adapters("gemma4", &weather_tools(), &chunks, "", expected_calls);
    }
}

#[test]
fn unified_5_4_uses_unchanged_corpus_schemas_through_available_public_adapters() {
    for family in ["gemma4", "deepseek_v41"] {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(format!(
            "../../conformance/fixtures-unified-v2/families/{family}/inputs_and_golden.yaml"
        ));
        let document: serde_yaml::Value =
            serde_yaml::from_str(&std::fs::read_to_string(path).expect("authored corpus"))
                .expect("corpus YAML");
        let case = &document["cases"]["malformed_json_then_two_valid_calls"];
        assert_eq!(case["display_id"].as_str(), Some("UNIFIED.5-4"));
        let request = &case["request"];
        let tools: Vec<Tool> =
            serde_yaml::from_value(request["tools"].clone()).expect("authored schemas");
        let input = request["input"].as_str().expect("authored input");
        for chunks in chunkings(input) {
            assert_adapters(
                family,
                &tools,
                &chunks,
                "",
                &[
                    ("echo", serde_json::json!({"value": "é"})),
                    ("echo", serde_json::json!({"value": "Café"})),
                ],
            );
        }
    }
}

#[test]
fn later_balanced_call_without_wrapper_close_recovers_at_eof_at_each_chunk_size() {
    let malformed = "<|tool_call>call:broken{note:<|\"|>unterminated";
    let later_balanced = "<|\"|>}<|tool_call>call:get_weather{city:<|\"|>NYC<|\"|>}";
    let input = format!("{malformed}{later_balanced}");
    assert_both_adapters_at_chunk_sizes(
        &input,
        &[1, 4, 16],
        &[("get_weather", serde_json::json!({"city": "NYC"}))],
    );
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
fn malformed_outer_quote_closes_after_an_incomplete_candidate_key() {
    let input = concat!(
        "<|tool_call>call:broken{note:<|\"|>outer",
        "<|tool_call>call:fake{<|\"|>}",
        "<|tool_call>call:get_weather{city:<|\"|>NYC<|\"|>}<tool_call|>",
    );
    assert_both_adapters(input, "");
    assert_both_adapters_at_chunk_sizes(
        input,
        &[1, 4, 16],
        &[("get_weather", serde_json::json!({"city": "NYC"}))],
    );
}

#[test]
fn malformed_outer_quote_keeps_a_later_candidate_after_an_incomplete_value() {
    for fake_value in ["x:", "x:[", "x:{y:"] {
        for (echo_value, value_schema, expected_value) in [
            (
                "<|\"|>é<|\"|>",
                serde_json::json!({"type": "string"}),
                serde_json::json!("é"),
            ),
            (
                "42",
                serde_json::json!({"type": "integer"}),
                serde_json::json!(42),
            ),
            (
                "true",
                serde_json::json!({"type": "boolean"}),
                serde_json::json!(true),
            ),
            (
                "null",
                serde_json::json!({"type": "null"}),
                serde_json::Value::Null,
            ),
            (
                "[42]",
                serde_json::json!({
                    "type": "array",
                    "items": {"type": "integer"}
                }),
                serde_json::json!([42]),
            ),
        ] {
            let input = format!(
                "<|tool_call>call:broken{{note:<|\"|>outer<|tool_call>call:fake{{{fake_value}<|\"|>}}<|tool_call>call:echo{{value:{echo_value}}}<tool_call|>"
            );
            let tools = [Tool {
                name: "echo".into(),
                description: None,
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": { "value": value_schema },
                    "required": ["value"]
                }),
                strict: None,
            }];
            for chunks in chunkings(&input) {
                assert_adapters(
                    "gemma4",
                    &tools,
                    &chunks,
                    "",
                    &[("echo", serde_json::json!({"value": expected_value}))],
                );
            }
        }
    }
}

#[test]
fn malformed_outer_quote_keeps_later_empty_argument_call() {
    for fake_value in ["x:", "x:[", "x:{y:"] {
        let input = format!(
            "<|tool_call>call:broken{{note:<|\"|>outer<|tool_call>call:fake{{{fake_value}<|\"|>}}<|tool_call>call:echo{{}}<tool_call|>"
        );
        let tools = [Tool {
            name: "echo".into(),
            description: None,
            parameters: serde_json::json!({ "type": "object", "properties": {} }),
            strict: None,
        }];
        for chunks in chunkings(&input) {
            assert_adapters(
                "gemma4",
                &tools,
                &chunks,
                "",
                &[("echo", serde_json::json!({}))],
            );
        }
    }
}

#[test]
fn repeated_ambiguous_outer_quotes_keep_each_later_call() {
    for repeats in [2, 3, 8] {
        let values: Vec<_> = (0..repeats).map(|index| format!("value{index}")).collect();
        let input = values
            .iter()
            .map(|value| {
                format!(
                    "<|tool_call>call:broken{{note:<|\"|>x<|tool_call>call:fake{{x:<|\"|>}}<|tool_call>call:echo{{value:<|\"|>{value}<|\"|>}}<tool_call|>"
                )
            })
            .collect::<String>();
        let expected: Vec<_> = values
            .iter()
            .map(|value| ("echo", serde_json::json!({"value": value})))
            .collect();
        if repeats <= 3 {
            for chunks in chunkings(&input) {
                assert_adapters("gemma4", &weather_echo_tools(), &chunks, "", &expected);
            }
        } else {
            assert_both_adapters_at_chunk_sizes(&input, &[1, 7, 64], &expected);
        }
    }
}

#[test]
fn empty_nested_object_keeps_array_string_value_context() {
    let input = concat!(
        "<|tool_call>call:broken{note:<|\"|>outer",
        "<|tool_call>call:get_weather{city:<|\"|>NYC<|\"|>,vals:[{},<|\"|>x<|\"|>]}",
    );
    for chunks in chunkings(input) {
        assert_adapters(
            "gemma4",
            &weather_tools(),
            &chunks,
            "",
            &[(
                "get_weather",
                serde_json::json!({"city": "NYC", "vals": [{}, "x"]}),
            )],
        );
    }
}

#[test]
fn string_data_cannot_become_a_resynchronization_target() {
    let input = concat!(
        "<|tool_call>call:broken{note:<|\"|>",
        "<|tool_call>call:get_weather{city:<|\"|>TRAP1<|\"|>}<tool_call|>more",
        "<|tool_call>call:get_weather{city:<|\"|>TRAP2<|\"|>}<tool_call|>",
        "<|\"|>}",
        "<|tool_call>call:get_weather{city:<|\"|>NYC<|\"|>}<tool_call|>",
    );
    assert_both_adapters(input, "");
}

#[test]
fn closed_quoted_string_at_eof_cannot_recover_its_marker_text() {
    let input = concat!(
        "<|tool_call>call:broken{note:<|\"|>x",
        "<|tool_call>call:get_weather{city:<|\"|>TRAP<|\"|>}<tool_call|>y<|\"|>",
    );
    assert_no_calls_at_every_split(input);
}

#[test]
fn ambiguous_balanced_call_without_wrapper_close_recovers_at_eof() {
    let input = concat!(
        "<|tool_call>call:broken{note:<|\"|>unterminated",
        "<|tool_call>call:get_weather{city:<|\"|>NYC<|\"|>}",
    );
    assert_both_adapters(input, "");
}

#[test]
fn second_marker_inside_quoted_outer_value_stays_data_until_the_quote_closes() {
    let input = concat!(
        "<|tool_call>call:broken{note:<|\"|>x",
        "<|tool_call>call:fake{open",
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
    assert_both_adapters_at_chunk_sizes(
        &valid,
        &[4, 16],
        &[("get_weather", serde_json::json!({"city": value}))],
    );
    assert_both_adapters_at_chunk_sizes(&incomplete, &[4, 16], &[]);
}

#[test]
fn repeated_unmatched_wrappers_recover_the_later_complete_call() {
    let malformed = "<|tool_call>call:broken{note:<|\"|>unterminated<|\"|>".repeat(256);
    let input =
        format!("{malformed}<|tool_call>call:get_weather{{city:<|\"|>NYC<|\"|>}}<tool_call|>");
    assert_both_adapters_at_chunk_sizes(
        &input,
        &[4, 16],
        &[("get_weather", serde_json::json!({"city": "NYC"}))],
    );
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
            let mut tool_only =
                create_tool_parser_for_family("gemma4", &weather_tools()).expect("tool parser");
            assert_eq!(
                drive_tool(tool_only.as_mut(), &chunks),
                ToolParseResult::default()
            );
            tool_only_scans.push(boundary_examined_bytes());

            reset_boundary_examined_bytes();
            let mut unified =
                dynamo_parsers_v2::create_unified_parser_for_family("gemma4", &weather_tools())
                    .expect("unified parser");
            unified
                .initialize_request(UnifiedParserInit::default())
                .expect("native request");
            assert_eq!(drive_unified(unified.as_mut(), &chunks), vec![]);
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

#[cfg(feature = "test-utils")]
#[test]
fn repeated_ambiguous_value_openers_resynchronize_in_linear_time() {
    for chunk_size in [1, 7, 64] {
        for adapter in ["tool-only", "unified"] {
            let mut scans = Vec::new();
            for repeats in [128, 256] {
                let input = format!(
                    "<|tool_call>call:broken{{note:<|\"|>outer{}",
                    "<|tool_call>call:fake{x:<|\"|>}".repeat(repeats)
                );
                let chunks: Vec<_> = input
                    .as_bytes()
                    .chunks(chunk_size)
                    .map(|chunk| std::str::from_utf8(chunk).expect("ASCII Gemma fixture"))
                    .collect();

                reset_boundary_examined_bytes();
                if adapter == "tool-only" {
                    let mut parser =
                        create_tool_parser_for_family("gemma4", &weather_tools()).expect("parser");
                    let _ = drive_tool(parser.as_mut(), &chunks);
                } else {
                    let mut parser = dynamo_parsers_v2::create_unified_parser_for_family(
                        "gemma4",
                        &weather_tools(),
                    )
                    .expect("parser");
                    parser
                        .initialize_request(UnifiedParserInit::default())
                        .expect("initialize");
                    let _ = drive_unified(parser.as_mut(), &chunks);
                }
                scans.push(boundary_examined_bytes());
            }

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

#[cfg(feature = "test-utils")]
#[test]
fn repeated_ambiguous_calls_resynchronize_in_linear_time() {
    let segment = |value: &str| {
        format!(
            "<|tool_call>call:broken{{note:<|\"|>x<|tool_call>call:fake{{x:<|\"|>}}<|tool_call>call:echo{{value:<|\"|>{value}<|\"|>}}<tool_call|>"
        )
    };
    for chunk_size in [1, 7, 64] {
        for adapter in ["tool-only", "unified"] {
            let mut scans = Vec::new();
            for repeats in [32, 64] {
                let input = (0..repeats)
                    .map(|index| segment(&format!("value{index}")))
                    .collect::<String>();
                let chunks: Vec<_> = input
                    .as_bytes()
                    .chunks(chunk_size)
                    .map(|chunk| std::str::from_utf8(chunk).expect("ASCII Gemma fixture"))
                    .collect();

                reset_boundary_examined_bytes();
                if adapter == "tool-only" {
                    let mut parser = create_tool_parser_for_family("gemma4", &weather_echo_tools())
                        .expect("parser");
                    let output = drive_tool(parser.as_mut(), &chunks);
                    assert_eq!(output.calls.len(), repeats);
                } else {
                    let mut parser = dynamo_parsers_v2::create_unified_parser_for_family(
                        "gemma4",
                        &weather_echo_tools(),
                    )
                    .expect("parser");
                    parser
                        .initialize_request(UnifiedParserInit::default())
                        .expect("initialize");
                    let output = drive_unified(parser.as_mut(), &chunks);
                    assert_eq!(output.len(), repeats);
                }
                scans.push(boundary_examined_bytes());
            }

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

#[test]
fn ambiguous_recovery_reset_cannot_contaminate_the_next_request() {
    let ambiguous = concat!(
        "<|tool_call>call:broken{note:<|\"|>unfinished",
        "<|tool_call>call:get_weather{city:<|\"|>OLD<|\"|>}<tool_call|>",
    );
    for finish_first in [false, true] {
        let mut parser =
            dynamo_parsers_v2::create_unified_parser_for_family("gemma4", &weather_tools())
                .expect("parser");
        parser
            .initialize_request(UnifiedParserInit::default())
            .expect("initialize");
        assert!(parser.push(ambiguous).expect("push").is_empty());
        if finish_first {
            assert_eq!(
                assemble(&parser.finish().expect("finish").events),
                vec![UnifiedEvent::ToolCall {
                    name: "get_weather".into(),
                    arguments: serde_json::json!({"city": "OLD"}),
                }]
            );
        }
        parser.reset();
        parser
            .initialize_request(UnifiedParserInit::default())
            .expect("reuse");
        assert_eq!(
            drive_unified(
                parser.as_mut(),
                &[
                    "fresh",
                    "<|tool_call>call:get_weather{city:<|\"|>NYC<|\"|>}<tool_call|>"
                ]
            ),
            vec![
                UnifiedEvent::Text {
                    text: "fresh".into()
                },
                UnifiedEvent::ToolCall {
                    name: "get_weather".into(),
                    arguments: serde_json::json!({"city": "NYC"})
                },
            ]
        );
    }
}
