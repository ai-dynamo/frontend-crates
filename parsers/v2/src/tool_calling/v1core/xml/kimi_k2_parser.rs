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
use crate::tool_calling::scan::{find_first_outside_strings, json_value_end};

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
/// Deliberately a plain (non-string-aware) search, unlike `find_section_end`
/// below: `cursor` only ever lands on gap/narration text between sections
/// (never inside an open JSON tool-call argument), and narration is where a
/// stray, ordinary quotation mark is expected -- string-tracking there would
/// risk mis-toggling on the user's own prose instead of protecting anything.
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

/// Find the first occurrence of any section end variant in `text[from..]`,
/// skipping over each embedded call's own JSON argument body rather than
/// tracking quotes as one continuous scan across the whole region.
///
/// `text[from..]` spans potentially multiple calls plus narration, not one
/// call's own body, and a naive single quote-tracking pass over that whole
/// span cannot safely tell "a `call_end`-shaped byte sequence embedded as
/// DATA inside one call's own well-formed string argument" apart from "an
/// earlier call's malformed, unbalanced-quote argument having desynced
/// `in_string` for everything after it" -- both look identical as local
/// per-character state (an earlier version of this function tried resolving
/// that ambiguity with a "reset at every `call_end` occurrence" heuristic
/// and a later "prefer honest tracking, fall back to reset" refinement;
/// both were reproduced as unsound: `in_string` carries desync GLOBALLY
/// across the whole multi-call span with either approach, so a stray quote
/// in one call could silently resync at exactly the wrong position inside a
/// LATER, unrelated call's own legitimate string).
///
/// So this walks call-by-call instead, using [`json_value_end`] -- real
/// bracket/quote-aware JSON structure, not a heuristic -- to skip each
/// call's argument body:
/// - A WELL-FORMED argument's structural end is trusted completely; a
///   `call_end`-shaped byte sequence embedded in one of ITS OWN strings can
///   never be mistaken for a boundary, because `json_value_end` parses the
///   whole value rather than pattern-matching mid-string.
/// - A MALFORMED (non-JSON) argument -- not itself illegal Kimi output,
///   just a raw-string-fallback case -- uses the shared quote-aware marker
///   scan for THIS call's own boundaries. If its quotes never close, the
///   conservative raw fence comparison keeps a real section end from being
///   swallowed into the argument.
///
/// Neither path lets one call's content affect how a DIFFERENT call is
/// scanned, so a malformed call's damage is bounded to itself and a
/// well-formed call's embedded marker-looking data is never mistaken for a
/// real boundary, regardless of what came before it in the same section.
/// Returns `(relative_position, matched_token_length)` or `None`.
fn find_section_end(
    text: &str,
    from: usize,
    config: &KimiK2ParserConfig,
) -> Option<(usize, usize)> {
    let region = &text[from..];
    let mut cursor = 0usize;
    loop {
        let next_section_end = config
            .section_end_variants
            .iter()
            .filter_map(|m| {
                region[cursor..]
                    .find(m.as_str())
                    .map(|p| (cursor + p, m.len()))
            })
            .min_by_key(|&(p, _)| p);
        // Plain, non-string-aware search for the next call's argument
        // start: outside an already-open call, `argument_begin` is a real
        // structural marker, the same trust level `find_section_start`
        // above already places in the surrounding section markers.
        let next_arg_begin_start = region[cursor..]
            .find(config.argument_begin.as_str())
            .map(|p| cursor + p);

        let section_end_is_first = match (next_section_end, next_arg_begin_start) {
            (Some((end_pos, _)), Some(arg_pos)) => end_pos < arg_pos,
            (Some(_), None) => true,
            (None, _) => false,
        };
        if section_end_is_first {
            return next_section_end;
        }
        let arg_pos = next_arg_begin_start? + config.argument_begin.len();
        // `json_value_end` only proves bracket/quote nesting is balanced,
        // not that the bytes are valid JSON (`{<|tool_calls_section_end|>}`
        // balances even though its "content" is a section-end marker, not
        // JSON). Trusting a bracket-balanced-but-invalid body as this
        // call's real argument boundary means its embedded section-end
        // marker is never scanned for at all -- the call-by-call walk just
        // skips straight past it as if it were legitimate JSON content.
        // Validating here routes an invalid body through the SAME
        // malformed/raw-string fallback below (with its own intervening-
        // section-end guard) instead of trusting `json_value_end` alone;
        // mirrors the identical hoist in streaming's `kimi_invoke_end`.
        let valid_json_end = json_value_end(&region[arg_pos..]).filter(|&json_end| {
            serde_json::from_str::<serde_json::Value>(&region[arg_pos..arg_pos + json_end]).is_ok()
        });
        cursor = match valid_json_end {
            Some(json_end) => arg_pos + json_end,
            None => {
                // Malformed argument (json_value_end failed): the boundary
                // fallback below must not blindly trust a `call_end` match
                // that has a REAL section-end marker sitting before it --
                // that means this call never got its own closer and the
                // section actually ends at that marker. Without this
                // check, the malformed call's raw text (which still
                // contains the section-end bytes, since nothing skipped
                // past them) swallows the real boundary and the call
                // recovers with a corrupted argument -- streaming's
                // `kimi_invoke_end` has the identical guard on its own
                // sibling malformed-argument fallback for exactly this
                // reason (blind-audit-caught: without this, batch mode
                // recovered a call streaming correctly dropped, a real
                // streaming/batch divergence in the opposite direction).
                let after_arg = &region[arg_pos..];
                // Quote state from malformed arguments is meaningful only
                // inside this call. Bound every structural and raw fallback
                // search at the next literal opener, then restart the loop
                // there so an unmatched quote cannot hide the next call and
                // later resynchronize on that call's quotes.
                let next_call_start = after_arg.find(config.call_start.as_str());
                let current_call_region = &after_arg[..next_call_start.unwrap_or(after_arg.len())];
                let structural_call_end =
                    find_first_outside_strings(current_call_region, [config.call_end.as_str()]);
                let structural_section_end = find_first_outside_strings(
                    current_call_region,
                    config.section_end_variants.iter().map(String::as_str),
                );
                if let Some((call_pos, call_len)) = structural_call_end {
                    if let Some((section_pos, section_len)) = structural_section_end
                        && section_pos < call_pos
                    {
                        return Some((arg_pos + section_pos, section_len));
                    }
                    arg_pos + call_pos + call_len
                } else {
                    // No quote-aware call end means this malformed call has
                    // no structurally provable boundary. Do not trust a later
                    // quote-aware section match either: an unmatched quote in
                    // this call can coincidentally resynchronize on a quote in
                    // the next call, recreating cross-call state leakage.
                    // Preserve the conservative historical fallback for this
                    // shape by comparing the first raw section end and call
                    // end, so the current call is bounded before scanning the
                    // next one independently.
                    let raw_section_end = config
                        .section_end_variants
                        .iter()
                        .filter_map(|marker| {
                            current_call_region
                                .find(marker.as_str())
                                .map(|position| (position, marker.len()))
                        })
                        .min_by_key(|&(position, _)| position);
                    match current_call_region.find(config.call_end.as_str()) {
                        Some(call_pos) => match raw_section_end {
                            Some((section_pos, section_len)) if section_pos < call_pos => {
                                return Some((arg_pos + section_pos, section_len));
                            }
                            _ => arg_pos + call_pos + config.call_end.len(),
                        },
                        // No literal call_end at all -- if a section-end
                        // exists ahead, the section ends there (this malformed
                        // call is truncated inside a still-open section);
                        // otherwise genuinely nothing further to skip.
                        None => match raw_section_end {
                            Some((section_pos, section_len)) => {
                                return Some((arg_pos + section_pos, section_len));
                            }
                            None => match next_call_start {
                                Some(next) => arg_pos + next,
                                None => return None,
                            },
                        },
                    }
                }
            }
        };
    }
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
        let lazy_arguments = cap
            .name("arguments")
            .map(|m| m.as_str().trim())
            .unwrap_or("{}");

        // The capture above is lazy (`.*?`) and stops at the first literal
        // `call_end`, including a copy inside a closed quoted span. That
        // corrupts both valid JSON strings and Kimi's permissive malformed
        // raw-string fallback before the latter reaches its typing branch.
        // Reuse the same quote-aware marker owner as streaming and section
        // discovery, bounded against a later invoke opener. An unmatched
        // quote intentionally falls back to the lazy capture: no structural
        // closer can be proved in that shape, and the historical conservative
        // recovery remains the safer contract.
        let arguments_raw = cap
            .name("arguments")
            .and_then(|m| {
                let args_start = m.start();
                let argument_region = &block[args_start..];
                if let Some(json_len) = json_value_end(argument_region).filter(|&json_len| {
                    serde_json::from_str::<serde_json::Value>(&argument_region[..json_len]).is_ok()
                }) {
                    let after_json = &argument_region[json_len..];
                    let call_end = after_json.find(config.call_end.as_str())?;
                    let next_call_start = after_json.find(config.call_start.as_str());
                    next_call_start
                        .is_none_or(|next| call_end < next)
                        .then(|| argument_region[..json_len].trim())
                } else {
                    // Once JSON is malformed, quote state cannot safely span
                    // a later invoke: two adjacent calls with unmatched
                    // quotes can cancel each other's state and make the first
                    // call borrow the second call's closer. The raw opener is
                    // therefore the conservative per-call damage boundary.
                    let next_call_start = argument_region.find(config.call_start.as_str());
                    let (call_end, _) =
                        find_first_outside_strings(argument_region, [config.call_end.as_str()])?;
                    next_call_start
                        .is_none_or(|next| call_end < next)
                        .then(|| argument_region[..call_end].trim())
                }
            })
            .unwrap_or(lazy_arguments);

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

    /// Reviewer-caught regression: a MALFORMED (non-JSON) argument containing
    /// one unescaped `"` used to desync `in_string` for the rest of the scan,
    /// so the real `section_end` after it was missed entirely and
    /// `extract_tool_calls`'s "no section_end found" fallback swallowed
    /// genuinely trailing normal text into the tool section. Confirmed with a
    /// scratch repro before fixing: `find_section_end` returned `None` even
    /// though a real closer existed further in the text. Fixed by walking
    /// call-by-call and falling back to a literal `call_end` search bounded
    /// to just this one call's own span (see the function's doc comment).
    #[test]
    fn find_section_end_survives_an_unbalanced_quote_in_a_malformed_argument() {
        let config = KimiK2ParserConfig::default();
        let text = "<|tool_call_begin|>functions.run:0<|tool_call_argument_begin|>bad\"arg<|tool_call_end|><|tool_calls_section_end|>After.";
        let real_pos = text.find("<|tool_calls_section_end|>").unwrap();
        let (pos, len) = find_section_end(text, 0, &config)
            .expect("must find the real closer despite the unbalanced quote");
        assert_eq!(pos, real_pos);
        assert_eq!(&text[pos..pos + len], "<|tool_calls_section_end|>");
    }

    /// Sibling case: TWO malformed calls each with their own unbalanced quote
    /// -- proves the literal `call_end` fallback is applied independently to
    /// EVERY call's own span, not just the first.
    #[test]
    fn find_section_end_survives_two_malformed_arguments_each_with_an_unbalanced_quote() {
        let config = KimiK2ParserConfig::default();
        let text = "<|tool_call_begin|>functions.a:0<|tool_call_argument_begin|>x\"1<|tool_call_end|><|tool_call_begin|>functions.b:1<|tool_call_argument_begin|>y\"2<|tool_call_end|><|tool_calls_section_end|>After.";
        let real_pos = text.find("<|tool_calls_section_end|>").unwrap();
        let (pos, len) = find_section_end(text, 0, &config)
            .expect("must find the real closer past both malformed calls");
        assert_eq!(pos, real_pos);
        assert_eq!(&text[pos..pos + len], "<|tool_calls_section_end|>");
    }

    /// Direct unit test on the shared boundary owner itself: `find_section_end`
    /// must skip a section-end-looking byte sequence embedded inside a JSON
    /// string argument and find the REAL trailing marker instead. Before this
    /// fix, `extract_tool_calls` truncated the section at the embedded copy,
    /// so the block handed to `parse_section_block` never contained the real
    /// `call_end` and the whole call silently vanished (`tool_index` for any
    /// later call in the same section then desynced, since nothing advanced
    /// past the dropped one).
    #[test]
    fn find_section_end_skips_a_marker_embedded_in_a_json_string() {
        let config = KimiK2ParserConfig::default();
        let text = "<|tool_call_begin|>functions.run:0<|tool_call_argument_begin|>{\"cmd\":\"echo <|tool_calls_section_end|>\"}<|tool_call_end|><|tool_calls_section_end|>";
        let (pos, len) = find_section_end(text, 0, &config).expect("must find the real closer");
        assert_eq!(
            &text[pos..pos + len],
            "<|tool_calls_section_end|>",
            "must resolve to the REAL trailing marker, not the embedded copy"
        );
        assert!(
            text[..pos].ends_with("<|tool_call_end|>"),
            "the resolved position must be the marker AFTER the invoke's own call_end, \
             not the earlier embedded lookalike inside the JSON string"
        );
    }

    /// Blind-audit-caught regression IN the prior fix: an EARLIER malformed
    /// call's unbalanced quote must not affect whether a LATER, well-formed
    /// call's embedded section-end lookalike is (correctly) treated as
    /// string data. A single continuous quote-tracking pass across the whole
    /// multi-call region -- whether or not it force-resets at every
    /// `call_end` -- carries desync state across call boundaries; this test
    /// combines both single-call regressions above into one input to prove
    /// the call-by-call structural walk keeps them fully independent.
    #[test]
    fn find_section_end_is_unaffected_by_an_earlier_malformed_calls_desync() {
        let config = KimiK2ParserConfig::default();
        let text = "<|tool_call_begin|>functions.a:0<|tool_call_argument_begin|>bad\"arg<|tool_call_end|>\
                     <|tool_call_begin|>functions.b:1<|tool_call_argument_begin|>{\"cmd\":\"echo <|tool_calls_section_end|>\"}<|tool_call_end|>\
                     After.<|tool_calls_section_end|>";
        let real_pos = text.rfind("<|tool_calls_section_end|>").unwrap();
        let (pos, len) = find_section_end(text, 0, &config)
            .expect("must find the real trailing closer past both calls");
        assert_eq!(
            pos, real_pos,
            "must resolve to the REAL trailing marker after \"After.\", not the earlier \
             embedded copy inside call b's own string, and not None"
        );
        assert_eq!(&text[pos..pos + len], "<|tool_calls_section_end|>");
    }

    /// Blind-audit-caught regression: batch mode used to disagree with
    /// streaming's `kimi_invoke_end` on the identical malformed-argument
    /// shape (a real section-end marker occurring before a malformed call's
    /// only available `call_end`) -- `find_section_end` couldn't find ANY
    /// section end at all (its own scan is bounded to look for a section
    /// end only in the gap AFTER a call's own span resolves, and this
    /// malformed call's span never resolves), so `parse_section_block`
    /// treated the rest of the buffer as one long section and recovered
    /// the call with the section-end marker swallowed into its raw-string
    /// argument -- a real streaming/batch divergence in the opposite
    /// direction from the one this session's fixes were meant to close.
    /// End-to-end via the real public entry point, not just the boundary
    /// function directly, to prove the whole pipeline agrees.
    #[test]
    fn batch_mode_drops_a_malformed_call_when_a_section_end_intervenes_before_its_call_end() {
        let config = KimiK2ParserConfig::default();
        let msg = "<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{\"location\": \"unterminated<|tool_calls_section_end|><|tool_call_end|>";
        let (calls, _content) = try_tool_call_parse_kimi_k2(msg, &config, None).unwrap();
        assert!(
            calls.is_empty(),
            "batch mode must drop this call, matching streaming's kimi_invoke_end -- \
             got {calls:?}"
        );
    }

    /// Reviewer-caught regression: `parse_section_block`'s I7 override used
    /// to trust `json_value_end`'s bracket-balance alone before REPLACING
    /// the lazy regex capture's own boundary -- a bracket-balanced but
    /// JSON-GRAMMAR-INVALID body containing an embedded `call_end`-looking
    /// substring as literal (non-string) content, immediately followed by
    /// the real literal `call_end`, still passed the override and shipped
    /// `arguments` containing the embedded fake closer as swallowed text
    /// instead of falling back to the lazy capture's own (shorter, but
    /// independently-derived) boundary. Reproduced directly before fixing:
    /// unpatched code shipped `arguments: "{not<|tool_call_end|>json}"`.
    #[test]
    fn i7_override_requires_actually_valid_json_not_just_balanced_brackets() {
        let config = KimiK2ParserConfig::default();
        let msg = "<|tool_calls_section_begin|><|tool_call_begin|>functions.run:0<|tool_call_argument_begin|>{not<|tool_call_end|>json}<|tool_call_end|><|tool_calls_section_end|>";
        let (calls, _content) = try_tool_call_parse_kimi_k2(msg, &config, None).unwrap();
        assert_eq!(
            calls.len(),
            1,
            "the call must still ship, just with the lazy boundary"
        );
        assert_eq!(
            calls[0].function.arguments, "{not",
            "must fall back to the lazy capture's own boundary, not the invalid \
             bracket-balanced span that swallows the embedded fake closer"
        );
    }

    /// Sibling positive control: the SAME override still correctly extends
    /// past an embedded fake closer that's legitimate DATA inside a
    /// well-formed JSON string (the original I7 case this override exists
    /// for) -- this fix must not regress that contract.
    #[test]
    fn i7_override_still_extends_past_a_fake_closer_inside_a_well_formed_string() {
        let config = KimiK2ParserConfig::default();
        let msg = "<|tool_calls_section_begin|><|tool_call_begin|>functions.run:0<|tool_call_argument_begin|>{\"note\":\"<|tool_call_end|>\"}<|tool_call_end|><|tool_calls_section_end|>";
        let (calls, _content) = try_tool_call_parse_kimi_k2(msg, &config, None).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(
            calls[0].function.arguments, "{\"note\":\"<|tool_call_end|>\"}",
            "the override must still extend past the embedded fake closer when it's \
             genuinely inside a well-formed JSON string"
        );
    }

    /// This stays a unit-level regression because the unified fixture corpus
    /// normalizes malformed raw arguments; it cannot assert the byte-exact
    /// v1 fallback values that prove each regex capture kept its own call.
    #[test]
    fn adjacent_malformed_calls_keep_their_own_raw_argument_boundaries() {
        let config = KimiK2ParserConfig::default();
        let msg = "<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>x\"1<|tool_call_end|><|tool_call_begin|>functions.run:1<|tool_call_argument_begin|>y\"2<|tool_call_end|><|tool_calls_section_end|>";
        let (calls, _content) = try_tool_call_parse_kimi_k2(msg, &config, None).unwrap();
        assert_eq!(
            calls.len(),
            2,
            "one malformed call must not absorb the next"
        );
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[0].function.arguments, "x\"1");
        assert_eq!(calls[1].function.name, "run");
        assert_eq!(calls[1].function.arguments, "y\"2");
    }

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
