// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Reference implementation:
// https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/function_call/kimik2_detector.py
// https://github.com/vllm-project/vllm/blob/main/vllm/tool_parsers/kimi_k2_tool_parser.py

use std::sync::OnceLock;

use regex::Regex;

use super::super::ToolDefinition;
use super::super::config::KimiK2ParserConfig;
use super::response::{CalledFunction, ToolCallResponse, ToolCallType};

static ID_REGEX: OnceLock<Regex> = OnceLock::new();

/// Builds the regex that captures `function_id` (e.g. `functions.get_weather:0`) and
/// `arguments` (JSON object) between the configured `call_start`, `argument_begin`, and
/// `call_end` tokens.
///
/// The `function_id` pattern `[\w.\-]+:\d+` matches the `functions.name:index` format used by
/// Kimi K2, consistent with sglang's reference implementation. The hyphen is included to
/// support function names with dashes (common in MCP tools, e.g. `mcp__portal__search-documents`).
///
/// The regex is built per-config (an owned `Regex`, not a `OnceLock`-cached
/// `&'static`) because its delimiters come from the caller's
/// `KimiK2ParserConfig`. A global cache would freeze the FIRST caller's tokens
/// and then silently parse every later config with the wrong delimiters.
fn get_tool_call_regex(config: &KimiK2ParserConfig) -> Regex {
    // Arguments capture is intentionally permissive (`.*?`) rather than
    // `\{...\}` so that truncated JSON (e.g. `{"location":"NYC` from
    // max_tokens / EOS) still matches. The downstream `serde_json::from_str`
    // is the validator: well-formed payloads parse, malformed/truncated
    // ones fall back to the raw-string arguments path.
    let pattern = format!(
        r"(?s){}\s*(?P<function_id>[\w.\-]+:\d+)\s*{}\s*(?P<arguments>.*?)\s*{}",
        regex::escape(&config.call_start),
        regex::escape(&config.argument_begin),
        regex::escape(&config.call_end),
    );
    Regex::new(&pattern).expect("Failed to compile kimi k2 tool call regex")
}

fn get_id_regex() -> &'static Regex {
    ID_REGEX.get_or_init(|| {
        Regex::new(r"^(?:functions\.)?(?P<name>[\w.\-]+):(?P<index>\d+)$")
            .expect("Failed to compile kimi k2 id regex")
    })
}

/// Format:
/// ```text
/// <|tool_calls_section_begin|>
/// <|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{"location":"NYC"}<|tool_call_end|>
/// <|tool_calls_section_end|>
/// ```
///
/// Returns (parsed_tool_calls, normal_text_content)
pub fn try_tool_call_parse_kimi_k2(
    message: &str,
    config: &KimiK2ParserConfig,
    tools: Option<&[ToolDefinition]>,
) -> anyhow::Result<(Vec<ToolCallResponse>, Option<String>)> {
    let (normal_text, tool_calls) = extract_tool_calls(message, config, tools)?;

    let normal_content = if normal_text.is_empty() {
        Some("".to_string())
    } else {
        Some(normal_text)
    };

    Ok((tool_calls, normal_content))
}

/// Find the first occurrence of any section start variant in `text[cursor..]`.
/// Returns `(relative_position, matched_token_length)` or `None`.
fn find_section_start(
    text: &str,
    cursor: usize,
    config: &KimiK2ParserConfig,
) -> Option<(usize, usize)> {
    let mut best: Option<(usize, usize)> = None;
    for variant in &config.section_start_variants {
        if let Some(pos) = text[cursor..].find(variant.as_str())
            && best.is_none_or(|(bp, _)| pos < bp)
        {
            best = Some((pos, variant.len()));
        }
    }
    best
}

/// Find the first occurrence of any section end variant in `text[from..]`.
/// Returns `(relative_position, matched_token_length)` or `None`.
fn find_section_end(
    text: &str,
    from: usize,
    config: &KimiK2ParserConfig,
) -> Option<(usize, usize)> {
    let mut best: Option<(usize, usize)> = None;
    for variant in &config.section_end_variants {
        if let Some(pos) = text[from..].find(variant.as_str())
            && best.is_none_or(|(bp, _)| pos < bp)
        {
            best = Some((pos, variant.len()));
        }
    }
    best
}

/// Extract tool calls and normal text from message.
///
/// ## Difference from Moonshot's reference implementation
///
/// The reference parser in
/// [tool_call_guidance.md](https://huggingface.co/moonshotai/Kimi-K2-Instruct/blob/main/docs/tool_call_guidance.md)
/// requires `section_end` to extract any tool calls:
///
/// ```python
/// pattern = r"<\|tool_calls_section_begin\|>(.*?)<\|tool_calls_section_end\|>"
/// tool_calls_sections = re.findall(pattern, tool_call_rsp, re.DOTALL)
/// ```
///
/// When `section_end` is missing (model hit max_tokens, EOS, or stop sequence),
/// `re.findall` returns `[]` and all complete individual tool calls are silently
/// dropped — even when individual calls have complete `call_begin` + args +
/// `call_end` markers.
///
/// This implementation treats a missing `section_end` as "section extends to
/// end-of-string", equivalent to:
///
/// ```python
/// pattern = r"<\|tool_calls_section_begin\|>(.*?)(?:<\|tool_calls_section_end\|>|$)"
/// ```
///
/// This allows recovery of complete individual tool calls from truncated output.
fn extract_tool_calls(
    text: &str,
    config: &KimiK2ParserConfig,
    tools: Option<&[ToolDefinition]>,
) -> anyhow::Result<(String, Vec<ToolCallResponse>)> {
    let mut normal_parts = Vec::new();
    let mut calls = Vec::new();
    let mut cursor = 0;

    if find_section_start(text, 0, config).is_none()
        && let Some(marker_idx) = first_orphan_kimi_marker_index(text, config)
    {
        let orphan_tail = &text[marker_idx..];
        if orphan_tail.starts_with(config.call_start.as_str()) {
            let mut recovered = parse_section_block(orphan_tail, config, tools)?;
            if !recovered.is_empty() {
                tracing::warn!(
                    why = "bare_call_recovery",
                    recovered_calls = recovered.len(),
                    recovered_bytes = orphan_tail.len(),
                    kept_prefix_bytes = marker_idx,
                    "Kimi K2 parser recovered complete call(s) without section_begin"
                );
                calls.append(&mut recovered);
                // Keep the leading text verbatim (no trim): the whitespace
                // before a recovered bare call belongs to the visible answer,
                // e.g. "Before. <|tool_call_begin|>..." must keep its trailing
                // space so the rendered content is "Before. ", not "Before.".
                return Ok((text[..marker_idx].to_string(), calls));
            }
        }
        return Ok((text[..marker_idx].trim().to_string(), calls));
    }

    while cursor < text.len() {
        if let Some((start_pos, _start_len)) = find_section_start(text, cursor, config) {
            let abs_start = cursor + start_pos;
            let gap = &text[cursor..abs_start];
            if let Some((prefix, mut recovered)) =
                recover_bare_kimi_calls_in_span(gap, config, tools)?
            {
                normal_parts.push(prefix);
                calls.append(&mut recovered);
            } else {
                normal_parts.push(gap.to_string());
            }

            // Add text before tool call section to normal parts.

            let (block, next_cursor) =
                if let Some((end_pos, end_len)) = find_section_end(text, abs_start, config) {
                    let abs_end = abs_start + end_pos + end_len;
                    (&text[abs_start..abs_end], abs_end)
                } else {
                    // No section_end found — treat rest of string as section
                    // body. Complete individual calls can still be extracted;
                    // truly truncated calls (no call_end) are ignored by
                    // parse_section_block's regex.
                    (&text[abs_start..], text.len())
                };

            if let Ok(mut parsed_calls) = parse_section_block(block, config, tools) {
                calls.append(&mut parsed_calls);
            }

            cursor = next_cursor;
        } else {
            // No more tool call sections.
            let gap = &text[cursor..];
            if let Some((prefix, mut recovered)) =
                recover_bare_kimi_calls_in_span(gap, config, tools)?
            {
                normal_parts.push(prefix);
                calls.append(&mut recovered);
            } else {
                normal_parts.push(gap.to_string());
            }
            break;
        }
    }

    let joined_normal_text = normal_parts.join("");
    let normal_text = if calls.is_empty() {
        // No tool calls parsed: this is plain content. Trim leading/trailing
        // whitespace so whitespace-only inputs collapse to "" (matches the
        // no-tool-call and empty-section fixtures).
        joined_normal_text.trim().to_string()
    } else {
        // Tool calls parsed: `normal_text` is the model text with each complete
        // tool-call block (section_begin through section_end, or to EOF when
        // section_end is missing) removed, keeping ALL surrounding natural
        // language verbatim — prefix, text BETWEEN sections, and text AFTER the
        // last section. `normal_parts` already holds exactly those gaps (the
        // section blocks were skipped, never pushed), so joining them yields the
        // model text minus tool-call markup with whitespace preserved as-is.
        // Malformed/unrecoverable blocks are handled by the bare-call recovery
        // paths above, which strip their markup without leaking it.
        joined_normal_text
    };
    Ok((normal_text, calls))
}

fn recover_bare_kimi_calls_in_span(
    span: &str,
    config: &KimiK2ParserConfig,
    tools: Option<&[ToolDefinition]>,
) -> anyhow::Result<Option<(String, Vec<ToolCallResponse>)>> {
    let Some(marker_idx) = first_orphan_kimi_marker_index(span, config) else {
        return Ok(None);
    };
    let orphan_tail = &span[marker_idx..];
    if !orphan_tail.starts_with(config.call_start.as_str()) {
        return Ok(None);
    }

    let recovered = parse_section_block(orphan_tail, config, tools)?;
    if recovered.is_empty() {
        return Ok(None);
    }

    tracing::warn!(
        why = "bare_call_gap_recovery",
        recovered_calls = recovered.len(),
        recovered_bytes = orphan_tail.len(),
        kept_prefix_bytes = marker_idx,
        "Kimi K2 parser recovered complete bare call(s) before a later section_begin"
    );
    // Keep the leading text verbatim (no trim): whitespace before a recovered
    // bare call is part of the visible answer. This prefix is pushed into
    // `normal_parts` and, because calls are non-empty, flows to the output
    // untrimmed (see the join in `extract_tool_calls`).
    Ok(Some((span[..marker_idx].to_string(), recovered)))
}

fn first_orphan_kimi_marker_index(text: &str, config: &KimiK2ParserConfig) -> Option<usize> {
    let mut best = [
        config.call_start.as_str(),
        config.call_end.as_str(),
        config.argument_begin.as_str(),
    ]
    .into_iter()
    .filter_map(|marker| text.find(marker))
    .min();

    for marker in &config.section_end_variants {
        if let Some(idx) = text.find(marker.as_str()) {
            best = Some(best.map_or(idx, |current| current.min(idx)));
        }
    }

    best
}

/// Parse a tool calls section block, extracting individual tool calls.
///
/// The block is between `<|tool_calls_section_begin|>` and `<|tool_calls_section_end|>`.
/// Each individual call is between `<|tool_call_begin|>` and `<|tool_call_end|>`.
fn parse_section_block(
    block: &str,
    config: &KimiK2ParserConfig,
    tools: Option<&[ToolDefinition]>,
) -> anyhow::Result<Vec<ToolCallResponse>> {
    let tool_call_regex = get_tool_call_regex(config);
    let id_regex = get_id_regex();

    let mut results = Vec::new();

    for cap in tool_call_regex.captures_iter(block) {
        let function_id = cap
            .name("function_id")
            .map(|m| m.as_str().trim())
            .unwrap_or("");
        let arguments_raw = cap
            .name("arguments")
            .map(|m| m.as_str().trim())
            .unwrap_or("{}");

        // Parse function ID
        let function_name = if let Some(id_cap) = id_regex.captures(function_id) {
            id_cap
                .name("name")
                .map(|m| m.as_str().to_string())
                .unwrap_or_default()
        } else {
            // Fallback: use the whole ID as the function name
            tracing::warn!(
                "Unexpected tool_call_id format: '{}', using as-is",
                function_id
            );
            function_id.to_string()
        };

        if function_name.is_empty() {
            continue;
        }

        // Validate function name against tools if provided
        if let Some(tools) = tools
            && !tools.iter().any(|t| t.name == function_name)
        {
            tracing::warn!("Tool '{}' is not defined in the tools list.", function_name);
        }

        // Validate JSON arguments
        let arguments_json = match serde_json::from_str::<serde_json::Value>(arguments_raw) {
            Ok(val) => serde_json::to_string(&val)?,
            Err(e) => {
                tracing::warn!(
                    "Failed to parse JSON arguments for tool '{}': {}. Using raw string.",
                    function_name,
                    e,
                );
                arguments_raw.to_string()
            }
        };

        // NOTE: Unlike other parsers (XML, DSML) which generate `call-{UUID}` IDs,
        // we preserve the model's native function_id (e.g., "functions.bash:0") here.
        // This matches the behavior of vllm/sglang and is required for Kimi K2 compatibility.
        let tool_call = ToolCallResponse {
            id: function_id.to_string(),
            tp: ToolCallType::Function,
            function: CalledFunction {
                name: function_name,
                arguments: arguments_json,
            },
        };

        results.push(tool_call);
    }

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Finding 1 regression: the tool-call regex must be built from the config
    /// passed on each call, never frozen by a global cache. Two configs with
    /// different delimiters must each parse with THEIR OWN tokens. A
    /// `OnceLock`-cached regex would parse the second config with the first
    /// config's delimiters and silently fail.
    #[test]
    fn test_regex_not_frozen_across_configs() {
        // Config A: default Kimi K2 delimiters.
        let config_a = KimiK2ParserConfig::default();

        // Config B: entirely different delimiters (angle-bracket style).
        let config_b = KimiK2ParserConfig {
            call_start: "[[CALL]]".to_string(),
            call_end: "[[/CALL]]".to_string(),
            argument_begin: "[[ARGS]]".to_string(),
            ..KimiK2ParserConfig::default()
        };

        let msg_a = "<|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{\"location\":\"NYC\"}<|tool_call_end|>";
        let msg_b = "[[CALL]]functions.get_time:0[[ARGS]]{\"zone\":\"UTC\"}[[/CALL]]";

        // Parse A first so its delimiters would be the ones a global cache locks in.
        let (calls_a, _) = try_tool_call_parse_kimi_k2(msg_a, &config_a, None).unwrap();
        assert_eq!(calls_a.len(), 1, "config A should parse its own delimiters");
        assert_eq!(calls_a[0].function.name, "get_weather");

        // Now parse B with B's delimiters. With a frozen cache this yields zero
        // calls (regex still expects config A's tokens).
        let (calls_b, _) = try_tool_call_parse_kimi_k2(msg_b, &config_b, None).unwrap();
        assert_eq!(
            calls_b.len(),
            1,
            "config B must parse with its OWN delimiters, not config A's frozen ones"
        );
        assert_eq!(calls_b[0].function.name, "get_time");
        assert_eq!(calls_b[0].function.arguments, "{\"zone\":\"UTC\"}");

        // And config A must STILL parse correctly afterward (both directions).
        let (calls_a2, _) = try_tool_call_parse_kimi_k2(msg_a, &config_a, None).unwrap();
        assert_eq!(calls_a2.len(), 1);
        assert_eq!(calls_a2[0].function.name, "get_weather");
    }

    /// Finding 2 regression: whitespace before a recovered bare call (no
    /// `section_begin`) belongs to the visible answer and must survive. The
    /// trailing space in "Before. " must not be trimmed away.
    #[test]
    fn test_leading_whitespace_preserved_before_bare_call() {
        let config = KimiK2ParserConfig::default();

        // Bare call (no section_begin) preceded by prose with a trailing space.
        let msg = "Before. <|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{\"location\":\"NYC\"}<|tool_call_end|>";
        let (calls, content) = try_tool_call_parse_kimi_k2(msg, &config, None).unwrap();

        assert_eq!(calls.len(), 1, "bare call should be recovered");
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(
            content.as_deref(),
            Some("Before. "),
            "trailing space before the recovered bare call must be preserved"
        );
    }

    /// Finding 2 regression for the second recovery path: a bare call in the
    /// gap BEFORE a later real section. The whitespace in the gap prose must
    /// survive into the joined normal text.
    #[test]
    fn test_leading_whitespace_preserved_before_bare_call_in_gap() {
        let config = KimiK2ParserConfig::default();

        // Bare call in a gap, followed by a real section further along.
        let msg = "Hi. <|tool_call_begin|>functions.a:0<|tool_call_argument_begin|>{}<|tool_call_end|> then <|tool_calls_section_begin|><|tool_call_begin|>functions.b:0<|tool_call_argument_begin|>{}<|tool_call_end|><|tool_calls_section_end|>";
        let (calls, content) = try_tool_call_parse_kimi_k2(msg, &config, None).unwrap();

        assert_eq!(calls.len(), 2, "both bare and sectioned calls recovered");
        assert_eq!(calls[0].function.name, "a");
        assert_eq!(calls[1].function.name, "b");
        // "Hi. " keeps its trailing space; " then " between the recovered bare
        // call and the section is preserved verbatim too.
        let content = content.unwrap();
        assert!(
            content.starts_with("Hi. "),
            "gap prose whitespace before recovered bare call must be preserved, got {content:?}"
        );
    }
}
