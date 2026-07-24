// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Inkling (thinkingmachines/Inkling-NVFP4) tool-call parser.
//!
//! One call: `<|message_model|>NAME<|content_invoke_tool_json|>{"name":..,"args":{..}}<|end_message|>`.
//! The JSON arguments key is `args`, not `arguments`, and a redundant `NAME` header
//! precedes the delimiter, so the generic JSON parser cannot be reused. Calls run
//! back-to-back with no separator; a headerless block still parses.

use serde_json::value::RawValue;
use uuid::Uuid;

use super::super::ToolDefinition;
use super::super::config::InklingParserConfig;
use super::super::response::{CalledFunction, ToolCallResponse, ToolCallType};

pub(crate) const MESSAGE_MODEL: &str = "<|message_model|>";
pub(crate) const INVOKE: &str = "<|content_invoke_tool_json|>";
pub(crate) const END_MESSAGE: &str = "<|end_message|>";

const MIN_PARTIAL_MARKER_LEN: usize = 3;

/// `args` is a `RawValue` to preserve the argument bytes verbatim.
#[derive(serde::Deserialize)]
struct InklingToolCall {
    name: String,
    #[serde(default)]
    args: Option<Box<RawValue>>,
}

pub fn detect_tool_call_start_inkling(chunk: &str, _config: &InklingParserConfig) -> bool {
    for marker in [MESSAGE_MODEL, INVOKE] {
        if chunk.contains(marker) || ends_with_partial_marker(chunk, marker) {
            return true;
        }
    }
    false
}

fn ends_with_partial_marker(chunk: &str, marker: &str) -> bool {
    for (i, _) in marker.char_indices().skip(1) {
        if i < MIN_PARTIAL_MARKER_LEN {
            continue;
        }
        if chunk.ends_with(&marker[..i]) {
            return true;
        }
    }
    false
}

pub fn find_tool_call_end_position_inkling(chunk: &str, _config: &InklingParserConfig) -> usize {
    chunk
        .rfind(END_MESSAGE)
        .map(|pos| pos + END_MESSAGE.len())
        .unwrap_or(chunk.len())
}

pub fn try_tool_call_parse_inkling(
    message: &str,
    config: &InklingParserConfig,
    _tools: Option<&[ToolDefinition]>,
) -> anyhow::Result<(Vec<ToolCallResponse>, Option<String>)> {
    let Some(invoke_pos) = message.find(INVOKE) else {
        return Ok((vec![], Some(message.to_string())));
    };

    // Start the stripped span at the `<|message_model|>NAME` header when present, so
    // it never leaks into normal_text.
    let header_pos = message[..invoke_pos].rfind(MESSAGE_MODEL);
    let block_start = header_pos.unwrap_or(invoke_pos);
    let mut prefix = message[..block_start].trim_end().to_string();

    let mut calls = Vec::new();
    let mut cursor = block_start;
    while let Some(rel) = message[cursor..].find(INVOKE) {
        let json_start = cursor + rel + INVOKE.len();
        let (json_str, next_cursor) = match message[json_start..].find(END_MESSAGE) {
            Some(end_rel) => {
                let end = json_start + end_rel;
                (&message[json_start..end], Some(end + END_MESSAGE.len()))
            }
            None => (&message[json_start..], None),
        };

        // Unterminated call: recover only on the finalize path; mid-stream drop it so
        // the jail doesn't claim a call before `<|end_message|>` arrives.
        if next_cursor.is_none() && !config.allow_eof_recovery {
            break;
        }

        if let Some(call) = parse_inkling_call(json_str)? {
            calls.push(call);
        }

        match next_cursor {
            Some(next) => cursor = next,
            None => break,
        }
    }

    // Header-less generation-primer block: when `add_generation_prompt` puts the
    // `<|message_model|>` primer in the prompt, the model emits its first block as
    // `NAME<|content_invoke_tool_json|>...` with no header. That leading NAME is the
    // redundant call header, not content, so drop it when it exactly matches the parsed
    // call name. Guarded on both no-header and exact-name-match: real prose (e.g.
    // "Let me check.") never equals the tool name, so it is still preserved as
    // normal_text. Matches vLLM's standalone output; in the two-stage pipeline the
    // reasoning parser reconstructs the header upstream, so this only affects the
    // tool parser run on its own.
    if header_pos.is_none()
        && let Some(first) = calls.first()
        && prefix == first.function.name
    {
        prefix.clear();
    }

    Ok((calls, Some(prefix)))
}

// Streaming deserializer so a complete object parses despite trailing bytes (missing
// delimiter); a mid-value truncation fails to parse and is dropped, never guessed.
fn parse_inkling_call(raw: &str) -> anyhow::Result<Option<ToolCallResponse>> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }

    let mut stream = serde_json::Deserializer::from_str(trimmed).into_iter::<InklingToolCall>();
    let Some(Ok(call)) = stream.next() else {
        return Ok(None);
    };
    if call.name.is_empty() {
        return Ok(None);
    }

    let arguments = match call.args {
        Some(args) => args.get().to_string(),
        None => "{}".to_string(),
    };

    Ok(Some(ToolCallResponse {
        id: format!("call-{}", Uuid::new_v4()),
        tp: ToolCallType::Function,
        function: CalledFunction {
            name: call.name,
            arguments,
        },
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn recovery_config() -> InklingParserConfig {
        InklingParserConfig {
            allow_eof_recovery: true,
        }
    }

    fn call_name_and_args(call: &ToolCallResponse) -> (String, serde_json::Value) {
        (
            call.function.name.clone(),
            serde_json::from_str(&call.function.arguments).expect("valid JSON arguments"),
        )
    }

    #[test]
    fn parses_single_call_with_header() {
        let input = r#"<|message_model|>get_weather<|content_invoke_tool_json|>{"name":"get_weather","args":{"location":"Paris","unit":"celsius"}}<|end_message|>"#;
        let (calls, normal) =
            try_tool_call_parse_inkling(input, &InklingParserConfig::default(), None).unwrap();
        assert_eq!(normal.as_deref(), Some(""));
        assert_eq!(calls.len(), 1);
        let (name, args) = call_name_and_args(&calls[0]);
        assert_eq!(name, "get_weather");
        assert_eq!(args["location"], "Paris");
        assert_eq!(args["unit"], "celsius");
    }

    #[test]
    fn parses_headerless_call() {
        let input =
            r#"<|content_invoke_tool_json|>{"name":"weather","args":{"city":"SF"}}<|end_message|>"#;
        let (calls, normal) =
            try_tool_call_parse_inkling(input, &InklingParserConfig::default(), None).unwrap();
        assert_eq!(normal.as_deref(), Some(""));
        assert_eq!(calls.len(), 1);
        let (name, args) = call_name_and_args(&calls[0]);
        assert_eq!(name, "weather");
        assert_eq!(args["city"], "SF");
    }

    #[test]
    fn parses_multiple_calls_back_to_back() {
        let input = r#"<|message_model|>get_weather<|content_invoke_tool_json|>{"name":"get_weather","args":{"location":"Paris"}}<|end_message|><|message_model|>get_weather<|content_invoke_tool_json|>{"name":"get_weather","args":{"location":"London"}}<|end_message|><|content_model_end_sampling|>"#;
        let (calls, normal) =
            try_tool_call_parse_inkling(input, &InklingParserConfig::default(), None).unwrap();
        assert_eq!(normal.as_deref(), Some(""));
        assert_eq!(calls.len(), 2);
        assert_eq!(call_name_and_args(&calls[0]).1["location"], "Paris");
        assert_eq!(call_name_and_args(&calls[1]).1["location"], "London");
    }

    #[test]
    fn keeps_prefix_prose_as_normal_text_and_strips_header() {
        let input = r#"Let me check. <|message_model|>get_weather<|content_invoke_tool_json|>{"name":"get_weather","args":{"location":"NYC"}}<|end_message|>"#;
        let (calls, normal) =
            try_tool_call_parse_inkling(input, &InklingParserConfig::default(), None).unwrap();
        assert_eq!(normal.as_deref(), Some("Let me check."));
        assert!(!normal.as_deref().unwrap().contains(MESSAGE_MODEL));
        assert_eq!(calls.len(), 1);
    }

    #[test]
    fn no_tool_call_is_plain_text() {
        let input = "Hello, how can I help you today?";
        let (calls, normal) =
            try_tool_call_parse_inkling(input, &InklingParserConfig::default(), None).unwrap();
        assert!(calls.is_empty());
        assert_eq!(normal.as_deref(), Some("Hello, how can I help you today?"));
    }

    #[test]
    fn undeclared_tool_name_is_still_surfaced() {
        let input = r#"<|message_model|>unknown_tool<|content_invoke_tool_json|>{"name":"unknown_tool","args":{"x":1}}<|end_message|>"#;
        let (calls, _) =
            try_tool_call_parse_inkling(input, &InklingParserConfig::default(), None).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "unknown_tool");
    }

    #[test]
    fn truncated_final_call_is_dropped_even_with_recovery() {
        let input = r#"<|content_invoke_tool_json|>{"name":"weather","args":{"city":"S"#;
        let (calls, normal) = try_tool_call_parse_inkling(input, &recovery_config(), None).unwrap();
        assert!(calls.is_empty());
        assert_eq!(normal.as_deref(), Some(""));
    }

    #[test]
    fn complete_body_without_end_marker_recovers_on_finalize() {
        // The closing `}` is a terminating delimiter, so a complete object with a
        // missing `<|end_message|>` fence is recovered on the finalize path.
        let input = r#"<|content_invoke_tool_json|>{"name":"weather","args":{"city":"SF"}}"#;
        let (calls, _) = try_tool_call_parse_inkling(input, &recovery_config(), None).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(call_name_and_args(&calls[0]).1["city"], "SF");
    }

    #[test]
    fn preserves_argument_byte_span() {
        let input =
            r#"<|content_invoke_tool_json|>{"name":"f","args":{"z":1,"a":2.0}}<|end_message|>"#;
        let (calls, _) =
            try_tool_call_parse_inkling(input, &InklingParserConfig::default(), None).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.arguments, r#"{"z":1,"a":2.0}"#);
    }

    #[test]
    fn headerless_generation_primer_name_is_dropped() {
        // Real e2e shape: `add_generation_prompt` consumes the `<|message_model|>`
        // primer, so the block is `NAME<|content_invoke_tool_json|>...`. The bare NAME
        // is the redundant header and must not leak into normal_text.
        let input = r#"book_flight<|content_invoke_tool_json|>{"name":"book_flight","args":{"destination":"Paris"}}<|end_message|>"#;
        let (calls, normal) =
            try_tool_call_parse_inkling(input, &InklingParserConfig::default(), None).unwrap();
        assert_eq!(normal.as_deref(), Some(""));
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "book_flight");
    }

    #[test]
    fn headerless_prefix_prose_is_kept_when_not_the_name() {
        // A header-less prefix that is NOT the tool name is real prose, kept verbatim
        // (only an exact name match is treated as the redundant header).
        let input = r#"here you go book_flight<|content_invoke_tool_json|>{"name":"book_flight","args":{}}<|end_message|>"#;
        let (calls, normal) =
            try_tool_call_parse_inkling(input, &InklingParserConfig::default(), None).unwrap();
        assert_eq!(normal.as_deref(), Some("here you go book_flight"));
        assert_eq!(calls.len(), 1);
    }
}
