// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::unified::{UnifiedParserEvent, UnifiedParserPrefill};

fn parser() -> KimiK3UnifiedParser {
    KimiK3UnifiedParser::new(&[])
}

fn arg(key: &str, arg_type: &str, value: &str) -> String {
    format!("{OPEN}argument key=\"{key}\" type=\"{arg_type}\"{SEP}{value}<|close|>argument{SEP}")
}

fn call(attrs: &str, body: &str) -> String {
    format!("{OPEN}call {attrs}{SEP}{body}<|close|>call{SEP}")
}

fn complete_message(reasoning: &str, response: &str, tools: &str) -> String {
    let mut output = format!("{THINK_OPEN}{reasoning}{THINK_CLOSE}");
    output.push_str(&format!("{RESPONSE_OPEN}{response}{RESPONSE_CLOSE}"));
    if !tools.is_empty() {
        output.push_str(&format!("{TOOLS_OPEN}{tools}{TOOLS_CLOSE}"));
    }
    output.push_str(MESSAGE_CLOSE);
    output
}

fn parse_chunks(chunks: &[&str]) -> UnifiedParserOutput {
    let mut parser = parser();
    let mut output = UnifiedParserOutput::default();
    for chunk in chunks {
        output.append(parser.push(chunk).unwrap());
    }
    output.append(parser.finish().unwrap());
    output
}

fn text(output: &UnifiedParserOutput) -> String {
    output
        .events
        .iter()
        .filter_map(|event| match event {
            UnifiedParserEvent::Text(value) => Some(value.as_str()),
            _ => None,
        })
        .collect()
}

fn reasoning(output: &UnifiedParserOutput) -> String {
    output
        .events
        .iter()
        .filter_map(|event| match event {
            UnifiedParserEvent::Reasoning(value) => Some(value.as_str()),
            _ => None,
        })
        .collect()
}

fn calls(output: &UnifiedParserOutput) -> Vec<ToolCallDelta> {
    output
        .events
        .iter()
        .filter_map(|event| match event {
            UnifiedParserEvent::ToolCall(call) => Some(call.clone()),
            _ => None,
        })
        .collect()
}

#[test]
fn parses_reasoning_response_and_typed_tool_call() {
    let tool = call(
        "tool=\"get_weather\" index=\"1\"",
        &format!(
            "{}{}{}",
            arg("city", "string", "Hangzhou"),
            arg("days", "number", "3"),
            arg("metric", "boolean", "true")
        ),
    );
    let mut parser = parser();
    let output = parser
        .parse_complete(&complete_message("inspect", "calling", &tool))
        .unwrap();

    assert_eq!(reasoning(&output), "inspect");
    assert_eq!(text(&output), "calling");
    assert_eq!(
        calls(&output),
        vec![ToolCallDelta {
            tool_index: 0,
            name: Some("get_weather".to_string()),
            arguments: r#"{"city":"Hangzhou","days":3,"metric":true}"#.to_string(),
        }]
    );
    assert_eq!(parser.tool_call_id(0), Some("get_weather:0"));
}

#[test]
fn preserves_typed_argument_order_and_number_spelling() {
    let tool = call(
        "tool=\"calculate\" index=\"2\"",
        &format!(
            "{}{}{}",
            arg("z", "number", "1e+09"),
            arg("a", "string", "first"),
            arg("m", "null", "null")
        ),
    );
    let output = parser()
        .parse_complete(&format!("{TOOLS_OPEN}{tool}{TOOLS_CLOSE}"))
        .unwrap();

    assert_eq!(
        calls(&output)[0].arguments,
        r#"{"z":1000000000.0,"a":"first","m":null}"#
    );
}

#[test]
fn streams_across_every_marker_boundary() {
    let tool = call(
        "tool=\"lookup\" index=\"1\"",
        &arg("query", "string", "weather"),
    );
    let full = complete_message("reason", "answer", &tool);
    let chunks = full
        .char_indices()
        .map(|(index, character)| {
            let end = index + character.len_utf8();
            &full[index..end]
        })
        .collect::<Vec<_>>();
    let output = parse_chunks(&chunks);

    assert_eq!(reasoning(&output), "reason");
    assert_eq!(text(&output), "answer");
    assert_eq!(calls(&output).len(), 1);
    assert_eq!(calls(&output)[0].arguments, r#"{"query":"weather"}"#);
}

#[test]
fn prompt_prefill_initializes_reasoning_or_response() {
    let mut reasoning_parser = parser();
    reasoning_parser
        .initialize(UnifiedParserPrefill::Reasoning)
        .unwrap();
    let reasoning_output = reasoning_parser
        .parse_complete(&format!(
            "hidden{THINK_CLOSE}{RESPONSE_OPEN}visible{RESPONSE_CLOSE}"
        ))
        .unwrap();
    assert_eq!(reasoning(&reasoning_output), "hidden");
    assert_eq!(text(&reasoning_output), "visible");

    let mut response_parser = parser();
    response_parser
        .initialize(UnifiedParserPrefill::Response)
        .unwrap();
    let response_output = response_parser
        .parse_complete(&format!("visible{RESPONSE_CLOSE}{MESSAGE_CLOSE}"))
        .unwrap();
    assert_eq!(reasoning(&response_output), "");
    assert_eq!(text(&response_output), "visible");
}

#[test]
fn marker_free_output_falls_through() {
    let output = parse_chunks(&["plain ", "assistant ", "text"]);
    assert_eq!(
        output.events,
        vec![UnifiedParserEvent::Text("plain assistant text".into())]
    );
}

#[test]
fn tool_call_waits_for_close_marker() {
    let mut parser = parser();
    let opening = format!(
        "{TOOLS_OPEN}{OPEN}call tool=\"lookup\" index=\"1\"{SEP}{}",
        arg("query", "string", "weather")
    );
    assert!(parser.push(&opening).unwrap().events.is_empty());

    let output = parser.push(&format!("{CALL_CLOSE}{TOOLS_CLOSE}")).unwrap();
    assert_eq!(calls(&output).len(), 1);
}

#[test]
fn parses_multiple_tool_calls_with_model_ids() {
    let calls_text = format!(
        "{}{}",
        call("tool=\"first\" index=\"1\"", &arg("x", "number", "1")),
        call("tool=\"second\" index=\"2\"", &arg("y", "number", "2"))
    );
    let mut parser = parser();
    let output = parser
        .parse_complete(&format!("{TOOLS_OPEN}{calls_text}{TOOLS_CLOSE}"))
        .unwrap();

    assert_eq!(
        calls(&output)
            .iter()
            .map(|call| call.name.as_deref())
            .collect::<Vec<_>>(),
        vec![Some("first"), Some("second")]
    );
    assert_eq!(parser.tool_call_id(0), Some("first:0"));
    assert_eq!(parser.tool_call_id(1), Some("second:1"));
}

#[test]
fn raw_json_arguments_pass_through() {
    let raw = "{ \"city\": \"Paris\", \"days\": 2 }";
    let tool = call(
        "tool=\"weather\" index=\"1\"",
        &format!("{JSON_OPEN} type=\"object\"{SEP}{raw}{JSON_CLOSE}"),
    );
    let output = parser()
        .parse_complete(&format!("{TOOLS_OPEN}{tool}{TOOLS_CLOSE}"))
        .unwrap();

    assert_eq!(calls(&output)[0].arguments, raw);
}

#[test]
fn string_arguments_remain_raw_and_other_malformed_values_fall_back() {
    let tool = call(
        "tool=\"echo\" index=\"1\"",
        &format!(
            "{}{}",
            arg("literal", "string", r#"a "quote" and \ slash"#),
            arg("quirky", "number", "not-a-number")
        ),
    );
    let output = parser()
        .parse_complete(&format!("{TOOLS_OPEN}{tool}{TOOLS_CLOSE}"))
        .unwrap();

    assert_eq!(
        calls(&output)[0].arguments,
        r#"{"literal":"a \"quote\" and \\ slash","quirky":"not-a-number"}"#
    );
}

#[test]
fn attribute_values_are_unescaped() {
    let tool = call(
        "tool=\"look&amp;up&quot;now\" index=\"raw\"",
        &arg("query", "string", "x"),
    );
    let mut parser = parser();
    let output = parser
        .parse_complete(&format!("{TOOLS_OPEN}{tool}{TOOLS_CLOSE}"))
        .unwrap();

    assert_eq!(calls(&output)[0].name.as_deref(), Some("look&up\"now"));
    assert_eq!(parser.tool_call_id(0), Some("look&up\"now:raw"));
}

#[test]
fn drops_call_without_tool_name() {
    let tool = call("index=\"1\"", &arg("query", "string", "x"));
    let output = parser()
        .parse_complete(&format!("{TOOLS_OPEN}{tool}{TOOLS_CLOSE}"))
        .unwrap();
    assert!(calls(&output).is_empty());
}

#[test]
fn preserves_source_event_order() {
    let tool = call("tool=\"lookup\" index=\"1\"", &arg("q", "string", "x"));
    let output = parser()
        .parse_complete(&format!(
            "{THINK_OPEN}r{THINK_CLOSE}{RESPONSE_OPEN}t{RESPONSE_CLOSE}{TOOLS_OPEN}{tool}{TOOLS_CLOSE}"
        ))
        .unwrap();

    assert!(matches!(output.events[0], UnifiedParserEvent::Reasoning(_)));
    assert!(matches!(output.events[1], UnifiedParserEvent::Text(_)));
    assert!(matches!(output.events[2], UnifiedParserEvent::ToolCall(_)));
}

#[test]
fn ignores_epilogue_and_post_message_noise() {
    let output = parser()
        .parse_complete(&format!(
            "{RESPONSE_OPEN}answer{RESPONSE_CLOSE}noise{MESSAGE_CLOSE}leak"
        ))
        .unwrap();
    assert_eq!(text(&output), "answer");
}

#[test]
fn finish_flushes_unclosed_reasoning_and_partial_text_marker() {
    let mut reasoning_parser = parser();
    let first = reasoning_parser
        .push(&format!("{THINK_OPEN}unfinished"))
        .unwrap();
    assert_eq!(reasoning(&first), "unfinished");
    assert!(reasoning_parser.finish().unwrap().events.is_empty());

    let mut text_parser = parser();
    let first = text_parser.push("answer<|ope").unwrap();
    assert_eq!(text(&first), "answer");
    let final_output = text_parser.finish().unwrap();
    assert_eq!(text(&final_output), "<|ope");
}

#[test]
fn finish_rejects_partial_call_but_keeps_complete_calls_before_truncated_fence() {
    let mut partial = parser();
    partial
        .push(&format!(
            "{TOOLS_OPEN}{OPEN}call tool=\"x\" index=\"1\"{SEP}{}",
            arg("q", "string", "unfinished")
        ))
        .unwrap();
    assert!(partial.finish().is_err());

    let complete = call("tool=\"x\" index=\"1\"", &arg("q", "string", "done"));
    let mut truncated_outer = parser();
    let output = truncated_outer
        .push(&format!("{TOOLS_OPEN}{complete}"))
        .unwrap();
    assert_eq!(calls(&output).len(), 1);
    assert!(truncated_outer.finish().unwrap().events.is_empty());
    assert!(truncated_outer.buffer.is_empty());
    assert_eq!(truncated_outer.cursor, 0);
}

#[test]
fn malformed_attributes_fail_and_reset_recovers_uncommitted_text() {
    let input = format!("{TOOLS_OPEN}{OPEN}call tool=bad{SEP}");
    let mut parser = parser();
    assert!(parser.push(&input).is_err());
    assert_eq!(parser.reset(), format!("{OPEN}call tool=bad{SEP}"));
}

#[test]
fn cursor_compacts_long_incremental_text() {
    let mut parser = parser();
    let chunk = "x".repeat(512);
    let mut output = UnifiedParserOutput::default();
    for _ in 0..64 {
        output.append(parser.push(&chunk).unwrap());
    }

    assert_eq!(text(&output).len(), 32 * 1024);
    assert!(parser.buffer.len() < 2 * COMPACT_MIN_CONSUMED);
    assert_eq!(parser.cursor, 0);
}
