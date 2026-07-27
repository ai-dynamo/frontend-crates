// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::{ToolCallDelta, UnifiedParserEvent};

fn weather_tools() -> Vec<Tool> {
    vec![Tool {
        name: "get_weather".to_string(),
        description: None,
        parameters: serde_json::json!({
            "type": "object",
            "properties": {
                "city": {"type": "string"}
            },
            "required": ["city"]
        }),
        strict: None,
    }]
}

fn tool_call() -> &'static str {
    concat!(
        "<tool_call>\n",
        "<function=get_weather>\n",
        "<parameter=city>Tokyo</parameter>\n",
        "</function>\n",
        "</tool_call>"
    )
}

fn parse(prefill: UnifiedParserPrefill, chunks: &[&str]) -> UnifiedParserOutput {
    let mut parser = Qwen3CoderUnifiedParser::new(&weather_tools());
    parser.initialize(prefill).unwrap();
    let mut output = UnifiedParserOutput::default();
    for chunk in chunks {
        parser.parse_into(chunk, &mut output).unwrap();
    }
    output.append(parser.finish().unwrap());
    output
}

#[test]
fn emits_reasoning_text_and_tool_call_in_order() {
    let input = format!("<think>reason</think>answer {}", tool_call());
    let output = parse(UnifiedParserPrefill::None, &[&input]);

    assert_eq!(
        output.events,
        vec![
            UnifiedParserEvent::Reasoning("reason".to_string()),
            UnifiedParserEvent::Text("answer ".to_string()),
            UnifiedParserEvent::ToolCall(ToolCallDelta {
                tool_index: 0,
                name: Some("get_weather".to_string()),
                arguments: r#"{"city":"Tokyo"}"#.to_string(),
            }),
        ]
    );
}

#[test]
fn prompt_prefill_starts_in_reasoning() {
    let input = format!("reason</think>{}", tool_call());
    let output = parse(UnifiedParserPrefill::Reasoning, &[&input]);

    assert_eq!(
        output.events,
        vec![
            UnifiedParserEvent::Reasoning("reason".to_string()),
            UnifiedParserEvent::ToolCall(ToolCallDelta {
                tool_index: 0,
                name: Some("get_weather".to_string()),
                arguments: r#"{"city":"Tokyo"}"#.to_string(),
            }),
        ]
    );
}

#[test]
fn every_byte_boundary_matches_single_chunk_output() {
    let input = format!("<think>reason</think>answer {}", tool_call());
    let expected = parse(UnifiedParserPrefill::None, &[&input]);

    for split in 1..input.len() {
        let actual = parse(
            UnifiedParserPrefill::None,
            &[&input[..split], &input[split..]],
        );
        assert_eq!(actual, expected, "split at byte {split}");
    }
}

#[test]
fn response_prefill_does_not_reinterpret_literal_tags() {
    assert_eq!(
        parse(
            UnifiedParserPrefill::Response,
            &["literal <think>tag</think>"]
        )
        .events,
        vec![UnifiedParserEvent::Text(
            "literal <think>tag</think>".to_string()
        )]
    );
}

#[test]
fn preserves_text_after_a_tool_call() {
    let input = format!("before {} after", tool_call());
    assert_eq!(
        parse(UnifiedParserPrefill::None, &[&input]).events,
        vec![
            UnifiedParserEvent::Text("before ".to_string()),
            UnifiedParserEvent::ToolCall(ToolCallDelta {
                tool_index: 0,
                name: Some("get_weather".to_string()),
                arguments: r#"{"city":"Tokyo"}"#.to_string(),
            }),
            UnifiedParserEvent::Text(" after".to_string()),
        ]
    );
}
