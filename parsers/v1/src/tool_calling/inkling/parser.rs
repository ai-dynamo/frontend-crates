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
use super::tokens::{END_MESSAGE, END_SAMPLING, INVOKE, MESSAGE_MODEL};

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

/// True when `chunk` ends with a non-empty proper prefix of `marker`: a tool-open
/// marker split across chunks. Any prefix length counts (down to a lone `<`), so a
/// split at `<` or `<|` is still recognized as a partial start rather than emitted
/// as unrecoverable normal text.
fn ends_with_partial_marker(chunk: &str, marker: &str) -> bool {
    for (i, _) in marker.char_indices().skip(1) {
        if chunk.ends_with(&marker[..i]) {
            return true;
        }
    }
    false
}

/// Byte offset immediately after the last complete tool-call block, or `None`
/// until a JSON-complete call followed by its `<|end_message|>` fence has arrived.
///
/// Looking for the token with `rfind` is not sufficient: the same literal can
/// legally occur inside a JSON string argument. The shared helper below finds a
/// fence only after the deserializer's complete-value boundary.
pub fn find_tool_call_end_position_inkling(
    chunk: &str,
    _config: &InklingParserConfig,
) -> Option<usize> {
    find_complete_tool_call_end(chunk)
}

/// Return the end of the last JSON-complete, fenced Inkling call in `message`.
///
/// This is `pub(crate)` because the reasoning parser must use the exact same
/// boundary before forwarding a tool block to the downstream tool-call parser.
/// Keeping the boundary shared prevents the reasoning stage from truncating a
/// call on marker-looking text inside an argument.
pub(crate) fn find_complete_tool_call_end(message: &str) -> Option<usize> {
    let mut search_from = 0;
    let mut last_end = None;

    while let Some(rel) = message[search_from..].find(INVOKE) {
        let json_start = search_from + rel + INVOKE.len();
        if let Some((_, block_end)) = delimited_json_span(message, json_start) {
            last_end = Some(block_end);
            search_from = block_end;
        } else {
            // This opener is incomplete or malformed. Keep scanning so a later
            // well-formed block can still be recovered (batch.4.e), but always
            // advance at least one byte to avoid re-matching the same opener.
            search_from = json_start.min(message.len());
            if search_from == message.len() {
                break;
            }
        }
    }

    last_end
}

pub fn try_tool_call_parse_inkling(
    message: &str,
    config: &InklingParserConfig,
    _tools: Option<&[ToolDefinition]>,
) -> anyhow::Result<(Vec<ToolCallResponse>, Option<String>)> {
    let Some(invoke_pos) = message.find(INVOKE) else {
        // These are parser-owned protocol tokens even when an opener is absent.
        // Do not leak an orphan close/header/turn terminator as user-visible text.
        return Ok((vec![], Some(strip_outer_framing(message))));
    };

    // Start the stripped span at the `<|message_model|>NAME` header when present, so
    // it never leaks into normal_text.
    let header_pos = message[..invoke_pos].rfind(MESSAGE_MODEL);
    let block_start = header_pos.unwrap_or(invoke_pos);
    let mut prefix = strip_outer_framing(message[..block_start].trim_end());

    let mut calls = Vec::new();
    let mut cursor = block_start;
    while let Some(rel) = message[cursor..].find(INVOKE) {
        let json_start = cursor + rel + INVOKE.len();
        let complete_json_end = json_value_end(message, json_start);
        let delimited = delimited_json_span(message, json_start);

        // A malformed block may still be followed by a valid block. Skip through
        // its raw fence so the next loop can recover the later call. This fallback
        // is never used to claim a call: parsing below only happens when the JSON
        // value itself is complete.
        let next_cursor = delimited.map(|(_, end)| end).or_else(|| {
            if complete_json_end.is_none() {
                message[json_start..]
                    .find(END_MESSAGE)
                    .map(|rel| json_start + rel + END_MESSAGE.len())
            } else {
                None
            }
        });

        let Some(json_end) = complete_json_end else {
            match next_cursor {
                Some(next) => {
                    cursor = next;
                    continue;
                }
                None => break,
            }
        };

        // Unterminated call: recover only on the finalize path; mid-stream drop it so
        // the jail doesn't claim a call before `<|end_message|>` arrives.
        if delimited.is_none() && !config.allow_eof_recovery {
            break;
        }

        if let Some(call) = parse_inkling_call(&message[json_start..json_end])? {
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

/// Remove Inkling framing tokens owned by the tool-call layer. Content-kind
/// markers are intentionally left to the reasoning parser, which runs first in
/// the serving pipeline.
fn strip_outer_framing(text: &str) -> String {
    [MESSAGE_MODEL, END_MESSAGE, END_SAMPLING]
        .into_iter()
        .fold(text.to_string(), |out, marker| out.replace(marker, ""))
}

/// Absolute byte offset after the first complete JSON value beginning at
/// `json_start`, or `None` while the value is incomplete/malformed.
fn json_value_end(message: &str, json_start: usize) -> Option<usize> {
    let raw = message.get(json_start..)?;
    let mut stream = serde_json::Deserializer::from_str(raw).into_iter::<serde::de::IgnoredAny>();
    match stream.next() {
        Some(Ok(_)) => Some(json_start + stream.byte_offset()),
        _ => None,
    }
}

/// `(json_end, block_end)` for a complete JSON value followed only by optional
/// whitespace and the real block fence.
fn delimited_json_span(message: &str, json_start: usize) -> Option<(usize, usize)> {
    let json_end = json_value_end(message, json_start)?;
    let tail = message.get(json_end..)?;
    let whitespace = tail.len() - tail.trim_start().len();
    let fence_start = json_end + whitespace;
    message
        .get(fence_start..)?
        .starts_with(END_MESSAGE)
        .then_some((json_end, fence_start + END_MESSAGE.len()))
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
    fn end_message_position_is_none_until_fence_arrives() {
        // No `<|end_message|>` yet: an incomplete call must not look complete, so the
        // end-position is `None` (the jail keeps accumulating). Once the fence lands,
        // it returns the offset just past it.
        let cfg = InklingParserConfig::default();
        let partial = r#"<|message_model|>get_weather<|content_invoke_tool_json|>{"name":"get_weather","args":{}}"#;
        assert_eq!(find_tool_call_end_position_inkling(partial, &cfg), None);
        let complete = format!("{partial}{END_MESSAGE}");
        assert_eq!(
            find_tool_call_end_position_inkling(&complete, &cfg),
            Some(complete.len())
        );
    }

    #[test]
    fn partial_open_marker_split_at_angle_bracket_is_detected() {
        // A tool-open marker split at `<` or `<|` must still register as a partial
        // start, else those bytes stream as unrecoverable normal text.
        let cfg = InklingParserConfig::default();
        assert!(detect_tool_call_start_inkling("<", &cfg));
        assert!(detect_tool_call_start_inkling("<|", &cfg));
        assert!(detect_tool_call_start_inkling("Let me check.<", &cfg));
        // Reassembling the split chunks parses the call cleanly.
        let chunks = [
            "<",
            r#"|message_model|>get_weather<|content_invoke_tool_json|>{"name":"get_weather","args":{}}<|end_message|>"#,
        ];
        let joined = chunks.concat();
        let (calls, normal) =
            try_tool_call_parse_inkling(&joined, &InklingParserConfig::default(), None).unwrap();
        assert_eq!(normal.as_deref(), Some(""));
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
    }

    #[test]
    fn end_message_literal_inside_string_arg_does_not_truncate() {
        // `<|end_message|>` inside a JSON string argument must not be mistaken for the
        // block fence: the call parses and the argument bytes survive verbatim.
        let input = r#"<|content_invoke_tool_json|>{"name":"echo","args":{"text":"a<|end_message|>b"}}<|end_message|>"#;
        let (calls, normal) =
            try_tool_call_parse_inkling(input, &InklingParserConfig::default(), None).unwrap();
        assert_eq!(normal.as_deref(), Some(""));
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "echo");
        assert_eq!(
            calls[0].function.arguments,
            r#"{"text":"a<|end_message|>b"}"#
        );
    }

    #[test]
    fn inner_end_message_without_real_fence_is_not_complete_mid_stream() {
        let input =
            r#"<|content_invoke_tool_json|>{"name":"echo","args":{"text":"a<|end_message|>b"}}"#;
        let cfg = InklingParserConfig::default();
        assert_eq!(find_tool_call_end_position_inkling(input, &cfg), None);

        let (calls, normal) = try_tool_call_parse_inkling(input, &cfg, None).unwrap();
        assert!(calls.is_empty());
        assert_eq!(normal.as_deref(), Some(""));

        // Finalization may recover the complete JSON body without the outer fence.
        let (calls, _) = try_tool_call_parse_inkling(input, &recovery_config(), None).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(
            calls[0].function.arguments,
            r#"{"text":"a<|end_message|>b"}"#
        );
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

    #[test]
    fn orphan_outer_framing_is_stripped_without_an_invoke() {
        let input =
            "prefix <|end_message|> middle <|message_model|> suffix<|content_model_end_sampling|>";
        let (calls, normal) =
            try_tool_call_parse_inkling(input, &InklingParserConfig::default(), None).unwrap();
        assert!(calls.is_empty());
        assert_eq!(normal.as_deref(), Some("prefix  middle  suffix"));
    }
}
