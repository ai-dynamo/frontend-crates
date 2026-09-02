// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Streaming tool-call parser for GLM-4.7 / GLM 5.1.
//!
//! GLM emits tool calls as
//!   `<tool_call>NAME<arg_key>k1</arg_key><arg_value>v1</arg_value>...</tool_call>`
//! The function name comes directly after `<tool_call>` (there is no inner
//! `<function=` marker), and there is exactly ONE call per
//! `<tool_call>...</tool_call>` block; multiple calls are multiple blocks.
//!
//! The streaming concern (buffering, chunk-split marker safety, normal_text
//! suppression) is owned here. The per-block value typing is delegated to the v1
//! batch parser `try_tool_call_parse_glm47` driven by `Glm47ParserConfig::default()`,
//! so a streamed call matches exactly what the batch parser produces. Arguments
//! are re-serialized in source `<arg_key>` order because the v1 parser builds
//! them from a `HashMap` whose key order is non-deterministic; the streaming
//! fixtures store the arguments as an exact JSON string, so order is pinned to
//! the model-emitted order (the order vLLM's Rust parser also preserves).

use crate::tool_calling::scan::reorder_arguments;
use crate::tool_calling::v1core::{Glm47ParserConfig, ToolDefinition, try_tool_call_parse_glm47};

use crate::tool_calling::traits::{Tool, ToolCallDelta, ToolParseResult, ToolParser};

const BLOCK_START: &str = "<tool_call>";
const BLOCK_END: &str = "</tool_call>";
const ARG_KEY_START: &str = "<arg_key>";
const ARG_KEY_END: &str = "</arg_key>";
const ARG_VALUE_START: &str = "<arg_value>";

/// Orphan markers that can anchor a bare call body that was emitted without a
/// leading `<tool_call>` opener (truncation / malformed framing). The first of
/// these in the buffer marks the boundary; the function name is the token
/// immediately before it. Mirrors `first_orphan_glm47_marker_index` in the v1
/// parser so streaming recovery agrees with batch recovery.
const ORPHAN_ANCHORS: [&str; 4] = [BLOCK_END, ARG_KEY_START, ARG_KEY_END, ARG_VALUE_START];

/// Stream parser for GLM-4.7 tool calls.
pub struct Glm47ToolStreamParser {
    buffer: String,
    suppress_normal_text: bool,
    next_index: usize,
    config: Glm47ParserConfig,
    tools: Vec<ToolDefinition>,
}

impl Glm47ToolStreamParser {
    pub fn new(tools: &[Tool]) -> Self {
        Self {
            buffer: String::new(),
            suppress_normal_text: false,
            next_index: 0,
            config: Glm47ParserConfig::default(),
            tools: tools.iter().map(ToolDefinition::from).collect(),
        }
    }

    fn drain(&mut self, flush: bool) -> anyhow::Result<ToolParseResult> {
        let mut out = ToolParseResult::default();

        loop {
            // A bare call body (no `<tool_call>` opener) is anchored by the first
            // orphan marker (`<arg_key>` / `</tool_call>` / ...) that precedes the
            // next wrapped opener. The function name is the identifier token
            // immediately before that anchor; prose before the name stays
            // normal_text. The v1 batch parser recovers these the same way, so
            // the Dynamo column never leaks tool markup into user-visible text.
            let wrapped_start = self.buffer.find(BLOCK_START);
            let bare = self.bare_anchor(wrapped_start);

            // A lone `</tool_call>` (BLOCK_END) with no wrapped opener before it
            // and no recoverable bare call anchoring it is a TRUE orphan close:
            // malformed markup that must never leak into normal_text. In GLM
            // `</tool_call>` doubles as the close of a bare call (it is in
            // ORPHAN_ANCHORS), so only strip it when `bare_anchor` finds no bare
            // body before it — otherwise the bare arm below recovers the call. A
            // bare body always has a function-name identifier before its first
            // orphan marker, so `bare.is_none()` reliably means "no bare call".
            // Emit the preceding prose when not suppressing, drain through the
            // marker, clear suppression, and continue. Mirrors the orphan-close
            // idiom in `minimax_m3`.
            if bare.is_none()
                && let Some(pos) = self.buffer.find(BLOCK_END)
                && wrapped_start.is_none_or(|open| pos < open)
            {
                if !self.suppress_normal_text && pos > 0 {
                    out.normal_text.push_str(&self.buffer[..pos]);
                }
                self.buffer.drain(..pos + BLOCK_END.len());
                self.suppress_normal_text = false;
                continue;
            }

            match (wrapped_start, bare) {
                // Bare anchor comes first (or there is no wrapped opener).
                // `bare_anchor` only returns `Some` when the bare body precedes
                // any wrapped opener, so this arm always wins over the wrapped
                // arm when a bare call is present.
                (_, Some(bare)) => {
                    // Surface prose preceding the bare function name.
                    if bare.name_start > 0 {
                        if !self.suppress_normal_text {
                            out.normal_text.push_str(&self.buffer[..bare.name_start]);
                        }
                        self.buffer.drain(..bare.name_start);
                    }

                    // Recover only once the bare body's `</tool_call>` close has
                    // streamed; otherwise hold the body (no leak) and wait. At
                    // EOF an unterminated bare body is dropped (truncation). The
                    // prose prefix was already drained, so the function name is
                    // now at the front of the buffer.
                    let Some(end_rel) = self.buffer.find(BLOCK_END) else {
                        if flush {
                            tracing::warn!(
                                why = "glm47_incomplete_tool_call",
                                "GLM-4.7 stream dropped incomplete bare tool call at EOF"
                            );
                            self.buffer.clear();
                        }
                        break;
                    };
                    let close = end_rel + BLOCK_END.len();
                    let bare_body = self.buffer[..close].to_string();
                    self.buffer.drain(..close);
                    self.suppress_normal_text = true;
                    // Wrap so the v1 parser takes its normal wrapped path.
                    let wrapped = format!("{BLOCK_START}{bare_body}");
                    if let Some(delta) = self.parse_block_delta(&wrapped)? {
                        tracing::warn!(
                            why = "glm47_bare_call_recovery",
                            tool_index = delta.tool_index,
                            "GLM-4.7 stream recovered a complete bare tool call without <tool_call> opener"
                        );
                        out.calls.push(delta);
                        self.next_index += 1;
                    }
                    continue;
                }
                // Wrapped opener comes first.
                (Some(start), _) => {
                    if start > 0 {
                        if !self.suppress_normal_text {
                            out.normal_text.push_str(&self.buffer[..start]);
                        }
                        self.buffer.drain(..start);
                    }

                    // Wait for the matching block end before parsing. The whole
                    // block (name + args) is value-typed by the v1 parser in one
                    // shot.
                    let Some(end_rel) = self.buffer.find(BLOCK_END) else {
                        if flush {
                            tracing::warn!(
                                why = "glm47_incomplete_tool_call",
                                "GLM-4.7 stream dropped incomplete tool call at EOF"
                            );
                            self.buffer.clear();
                        }
                        break;
                    };

                    let block_end = end_rel + BLOCK_END.len();
                    let block = self.buffer[..block_end].to_string();
                    self.buffer.drain(..block_end);

                    // A complete wrapped block is fully consumed here, so natural
                    // text after it (inter-call / trailing) is kept again: drop
                    // only the block markup and clear suppression. This matches
                    // the v1 batch parser, which strips the complete-block markup
                    // and preserves surrounding text verbatim (cases 8.b/8.c/8.d).
                    self.suppress_normal_text = false;

                    if let Some(delta) = self.parse_block_delta(&block)? {
                        out.calls.push(delta);
                        self.next_index += 1;
                    }
                    continue;
                }
                // No opener and no bare anchor: emit buffered text, but hold back
                // anything split across this chunk boundary that the next chunk
                // could turn into a tool call — a partial marker, a partial
                // orphan anchor plus the bare function name before it, or a
                // trailing bare identifier awaiting its `<arg_key>` — unless
                // flushing.
                (None, None) => {
                    let keep = if flush {
                        0
                    } else {
                        trailing_holdback_len(&self.buffer)
                    };
                    let emit_len = self.buffer.len().saturating_sub(keep);
                    if emit_len > 0 {
                        if !self.suppress_normal_text {
                            out.normal_text.push_str(&self.buffer[..emit_len]);
                        }
                        self.buffer.drain(..emit_len);
                    }
                    break;
                }
            }
        }

        Ok(out)
    }

    /// Locate a bare call anchor in the buffer: the first orphan marker (before
    /// any wrapped `<tool_call>` opener) whose preceding identifier token is a
    /// plausible function name. Returns `None` when no such anchor exists (the
    /// region is plain prose or a normal wrapped block). `wrapped_start` is the
    /// index of the next `<tool_call>` opener, so an orphan marker that belongs
    /// to a wrapped block (i.e. appears after the opener) is ignored.
    fn bare_anchor(&self, wrapped_start: Option<usize>) -> Option<BareAnchor> {
        let marker_idx = ORPHAN_ANCHORS
            .iter()
            .filter_map(|m| self.buffer.find(m))
            .min()?;
        // An orphan marker after the next wrapped opener belongs to that block.
        if wrapped_start.is_some_and(|w| w <= marker_idx) {
            return None;
        }
        let before = self.buffer[..marker_idx].trim_end();
        let name_start = before
            .char_indices()
            .rev()
            .find(|(_, ch)| ch.is_whitespace())
            .map(|(idx, ch)| idx + ch.len_utf8())
            .unwrap_or(0);
        let candidate = before[name_start..].trim();
        if candidate.is_empty()
            || !candidate
                .chars()
                .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '.'))
        {
            return None;
        }
        Some(BareAnchor { name_start })
    }

    /// Parse one complete `<tool_call>...</tool_call>` block into a delta.
    ///
    /// Delegates value typing to the v1 batch parser, then re-orders the
    /// arguments to the source `<arg_key>` order so the serialized JSON string
    /// matches the engine reference output exactly.
    fn parse_block_delta(&self, block: &str) -> anyhow::Result<Option<ToolCallDelta>> {
        let (calls, _content) = try_tool_call_parse_glm47(block, &self.config, Some(&self.tools))?;
        let Some(call) = calls.into_iter().next() else {
            return Ok(None);
        };
        let arguments = reorder_arguments(&call.function.arguments, &source_arg_key_order(block));
        Ok(Some(ToolCallDelta {
            tool_index: self.next_index,
            name: Some(call.function.name),
            arguments,
            complete: true,
        }))
    }
}

impl ToolParser for Glm47ToolStreamParser {
    fn create(tools: &[Tool]) -> anyhow::Result<Box<dyn ToolParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self::new(tools)))
    }

    fn preserve_special_tokens(&self) -> bool {
        true
    }

    fn push(&mut self, chunk: &str) -> anyhow::Result<ToolParseResult> {
        self.buffer.push_str(chunk);
        self.drain(false)
    }

    fn finish(&mut self) -> anyhow::Result<ToolParseResult> {
        self.drain(true)
    }
}

/// A located bare call anchor: `name_start` is the byte offset of the function
/// name token in the buffer (prose before it is normal_text).
#[derive(Clone, Copy)]
struct BareAnchor {
    name_start: usize,
}

/// Number of trailing bytes to withhold from normal_text at a chunk boundary so
/// a marker or bare function name split across the boundary is not leaked.
///
/// Reached only when the buffer holds no complete `<tool_call>` opener and no
/// complete orphan anchor. Three overlapping split cases must be retained:
///   * a partial `<tool_call>` opener (`<too`);
///   * a partial orphan anchor (`<arg`, `</too`, ...) — and the bare
///     function-name identifier immediately before it, since the next chunk may
///     complete the anchor and make that identifier the call name;
///   * a trailing bare identifier with no marker yet (`get_weather`), which the
///     next chunk may anchor with `<arg_key>`.
///
/// All framing markers start with `<`, so a lone trailing `<` is treated as a
/// possible orphan anchor and the preceding identifier is held too. Mirrors the
/// v1 `first_orphan_glm47_marker_index` recovery policy: emit nothing that could
/// be the prefix of an orphan anchor or a pending bare call. Held-back bytes are
/// flushed on the next chunk (or at EOF), so the concatenated normal_text is
/// unchanged — only its chunk boundaries shift.
fn trailing_holdback_len(text: &str) -> usize {
    // Longest proper-prefix of any framing marker that `text` ends with, and
    // whether that partial suffix could belong to an orphan anchor.
    let mut marker_keep = 0;
    let mut orphan_partial = false;
    for marker in [
        BLOCK_START,
        BLOCK_END,
        ARG_KEY_START,
        ARG_KEY_END,
        ARG_VALUE_START,
    ] {
        let is_orphan = marker != BLOCK_START;
        for len in 1..marker.len() {
            if text.ends_with(&marker[..len]) {
                if len > marker_keep {
                    marker_keep = len;
                    orphan_partial = is_orphan;
                } else if len == marker_keep && is_orphan {
                    orphan_partial = true;
                }
            }
        }
    }

    // Hold the preceding bare identifier when the partial suffix could be an
    // orphan anchor, or when there is no partial marker at all (a bare name
    // still awaiting its `<arg_key>`).
    if !orphan_partial && marker_keep != 0 {
        return marker_keep;
    }
    let ident_end = text.len() - marker_keep;
    let name_start = text[..ident_end]
        .char_indices()
        .rev()
        .take_while(|(_, ch)| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '.'))
        .last()
        .map(|(idx, _)| idx)
        .unwrap_or(ident_end);
    text.len() - name_start
}

/// Argument key names in the order they appear in a tool-call block.
fn source_arg_key_order(block: &str) -> Vec<String> {
    let mut names = Vec::new();
    let mut cursor = 0;
    while let Some(rel) = block[cursor..].find(ARG_KEY_START) {
        let start = cursor + rel + ARG_KEY_START.len();
        let Some(end_rel) = block[start..].find(ARG_KEY_END) else {
            break;
        };
        let name = block[start..start + end_rel].trim();
        if !name.is_empty() {
            names.push(name.to_string());
        }
        cursor = start + end_rel + ARG_KEY_END.len();
    }
    names
}

#[cfg(test)]
mod tests {
    use super::*;

    fn weather_tools() -> Vec<Tool> {
        vec![Tool {
            name: "get_weather".to_string(),
            description: None,
            parameters: serde_json::json!({
                "type": "object",
                "properties": { "location": { "type": "string" } }
            }),
            strict: None,
        }]
    }

    fn parse_chunks(tools: &[Tool], chunks: &[&str]) -> ToolParseResult {
        let mut parser = Glm47ToolStreamParser::new(tools);
        let mut out = ToolParseResult::default();
        for chunk in chunks {
            out.append(parser.push(chunk).expect("push"));
        }
        out.append(parser.finish().expect("finish"));
        out
    }

    #[test]
    fn repeated_arg_key_emits_key_once() {
        // A repeated <arg_key> must not produce duplicate keys in the arguments.
        let out = parse_chunks(
            &weather_tools(),
            &[
                "<tool_call>get_weather<arg_key>location</arg_key><arg_value>NYC</arg_value>\
               <arg_key>location</arg_key><arg_value>NYC</arg_value></tool_call>",
            ],
        );
        let merged = out.coalesce_calls();
        assert_eq!(merged.calls.len(), 1);
        let args = merged.calls[0].arguments.clone();
        assert_eq!(
            args.matches("\"location\"").count(),
            1,
            "duplicate key in arguments: {args}"
        );
    }

    #[test]
    fn emits_complete_call_on_close() {
        let out = parse_chunks(
            &weather_tools(),
            &[
                "<tool_call>get_weather<arg_key>",
                "location</arg_key>",
                "<arg_value>NYC</arg_value></tool_call>",
            ],
        );
        assert_eq!(out.normal_text, "");
        let merged = out.coalesce_calls();
        assert_eq!(merged.calls.len(), 1);
        assert_eq!(merged.calls[0].tool_index, 0);
        assert_eq!(merged.calls[0].name.as_deref(), Some("get_weather"));
        assert_eq!(merged.calls[0].arguments, r#"{"location":"NYC"}"#);
    }

    #[test]
    fn emits_two_parallel_calls() {
        let tools = vec![
            Tool {
                name: "get_weather".to_string(),
                description: None,
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": { "location": { "type": "string" } }
                }),
                strict: None,
            },
            Tool {
                name: "get_time".to_string(),
                description: None,
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": { "timezone": { "type": "string" } }
                }),
                strict: None,
            },
        ];
        let out = parse_chunks(
            &tools,
            &[
                "<tool_call>get_weather<arg_key>location</arg_key><arg_value>NYC</arg_value></tool_call>",
                "<tool_call>get_time<arg_key>timezone</arg_key><arg_value>EST</arg_value></tool_call>",
            ],
        );
        let merged = out.coalesce_calls();
        assert_eq!(merged.calls.len(), 2);
        assert_eq!(merged.calls[0].name.as_deref(), Some("get_weather"));
        assert_eq!(merged.calls[0].arguments, r#"{"location":"NYC"}"#);
        assert_eq!(merged.calls[1].tool_index, 1);
        assert_eq!(merged.calls[1].name.as_deref(), Some("get_time"));
        assert_eq!(merged.calls[1].arguments, r#"{"timezone":"EST"}"#);
    }

    #[test]
    fn preserves_prefix_text_before_block() {
        let out = parse_chunks(
            &weather_tools(),
            &[
                "Checking: ",
                "<tool",
                "_call>get_weather<arg_key>location</arg_key><arg_value>NYC</arg_value></tool_call>",
            ],
        );
        assert_eq!(out.normal_text, "Checking: ");
        assert_eq!(out.coalesce_calls().calls.len(), 1);
    }

    #[test]
    fn holds_back_partial_start_marker() {
        // The `<tool_call>` marker is split across two chunks; the partial
        // prefix must not leak as normal_text.
        let mut parser = Glm47ToolStreamParser::new(&weather_tools());
        let first = parser.push("hello <tool").expect("push");
        assert_eq!(first.normal_text, "hello ");
        let second = parser
            .push("_call>get_weather<arg_key>location</arg_key><arg_value>NYC</arg_value></tool_call>")
            .expect("push");
        assert_eq!(second.normal_text, "");
        assert_eq!(second.calls.len(), 1);
    }

    #[test]
    fn suppresses_incomplete_tool_call_at_eof() {
        let out = parse_chunks(
            &weather_tools(),
            &[
                "<tool_call>get_weather<arg_key>location</arg_key>",
                "<arg_value>NY",
            ],
        );
        assert_eq!(out.normal_text, "");
        assert!(out.calls.is_empty());
    }

    #[test]
    fn preserves_source_arg_key_order() {
        // destination, passengers, first_class is deliberately NOT alphabetical:
        // the serialized arguments must keep the model-emitted arg-key order.
        let tools = vec![Tool {
            name: "book_flight".to_string(),
            description: None,
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "destination": { "type": "string" },
                    "passengers": { "type": "integer" },
                    "first_class": { "type": "boolean" }
                }
            }),
            strict: None,
        }];
        let out = parse_chunks(
            &tools,
            &[
                "<tool_call>book_flight<arg_key>destination</arg_key><arg_value>Paris</arg_value>",
                "<arg_key>passengers</arg_key><arg_value>2</arg_value>",
                "<arg_key>first_class</arg_key><arg_value>true</arg_value></tool_call>",
            ],
        );
        let merged = out.coalesce_calls();
        assert_eq!(merged.calls.len(), 1);
        assert_eq!(
            merged.calls[0].arguments,
            r#"{"destination":"Paris","passengers":2,"first_class":true}"#
        );
    }

    #[test]
    fn function_only_call_no_args() {
        let tools = vec![Tool {
            name: "get_time".to_string(),
            description: None,
            parameters: serde_json::json!({ "type": "object", "properties": {} }),
            strict: None,
        }];
        let out = parse_chunks(&tools, &["<tool_call>get_time</tool_call>"]);
        let merged = out.coalesce_calls();
        assert_eq!(merged.calls.len(), 1);
        assert_eq!(merged.calls[0].name.as_deref(), Some("get_time"));
        assert_eq!(merged.calls[0].arguments, "{}");
    }

    // ── Bare-call recovery (no `<tool_call>` opener) — conformance 5.b/5.f/5.g.
    // The v1 batch parser recovers a complete call body emitted without the
    // outer opener; the streaming parser must do the same so tool markup never
    // leaks into the Dynamo `normal_text` column.

    #[test]
    fn recovers_bare_call_without_opener() {
        // 5.b: `NAME<arg_key>...</tool_call>` with no `<tool_call>` open.
        let out = parse_chunks(
            &weather_tools(),
            &[
                "get_weather<arg_key>location</arg_key>",
                "<arg_value>",
                "NYC</arg_value></tool_call>",
            ],
        );
        assert_eq!(
            out.normal_text, "",
            "bare body must not leak as normal_text"
        );
        let merged = out.coalesce_calls();
        assert_eq!(merged.calls.len(), 1);
        assert_eq!(merged.calls[0].name.as_deref(), Some("get_weather"));
        assert_eq!(merged.calls[0].arguments, r#"{"location":"NYC"}"#);
    }

    #[test]
    fn recovers_bare_call_after_prose_keeps_prose() {
        // 5.g: genuine prose before the bare call stays normal_text; the bare
        // body is recovered, not leaked.
        let out = parse_chunks(
            &weather_tools(),
            &[
                "I will",
                " check",
                " that. get_weather<arg_key>location</arg_key>",
                "<arg_value>NYC</arg_value>",
                "</tool_call>",
            ],
        );
        assert_eq!(out.normal_text, "I will check that. ");
        let merged = out.coalesce_calls();
        assert_eq!(merged.calls.len(), 1);
        assert_eq!(merged.calls[0].name.as_deref(), Some("get_weather"));
        assert_eq!(merged.calls[0].arguments, r#"{"location":"NYC"}"#);
    }

    #[test]
    fn recovers_bare_call_before_wrapped_call() {
        // 5.f: a bare call followed by a complete wrapped call — both recover,
        // with distinct indices.
        let out = parse_chunks(
            &weather_tools(),
            &[
                "get_weather<arg_key>location</arg_key>",
                "<arg_value>",
                "NYC</arg_value></tool_call><tool_call>",
                "get_weather<arg_key>location</arg_key>",
                "<arg_value>",
                "Boston</arg_value></tool_call>",
            ],
        );
        assert_eq!(out.normal_text, "");
        let merged = out.coalesce_calls();
        assert_eq!(merged.calls.len(), 2);
        assert_eq!(merged.calls[0].tool_index, 0);
        assert_eq!(merged.calls[0].arguments, r#"{"location":"NYC"}"#);
        assert_eq!(merged.calls[1].tool_index, 1);
        assert_eq!(merged.calls[1].arguments, r#"{"location":"Boston"}"#);
    }

    #[test]
    fn preserves_trailing_text_after_block() {
        // 8.b: trailing narration after a complete block flows into normal_text.
        let out = parse_chunks(
            &weather_tools(),
            &[
                "<tool_call>get_weather<arg_key>location</arg_key><arg_value>NYC</arg_value></tool_call>",
                " Let me know if you need more.",
            ],
        );
        assert_eq!(out.normal_text, " Let me know if you need more.");
        assert_eq!(out.coalesce_calls().calls.len(), 1);
    }

    #[test]
    fn preserves_inter_call_and_trailing_text() {
        // 8.d: narration between two complete blocks flows into normal_text;
        // both calls are emitted with distinct indices.
        let out = parse_chunks(
            &weather_tools(),
            &[
                "I will check the weather. <tool_call>get_weather<arg_key>location</arg_key><arg_value>NYC</arg_value></tool_call>",
                " Then check LA weather. <tool_call>get_weather<arg_key>location</arg_key><arg_value>LA</arg_value></tool_call>",
            ],
        );
        assert_eq!(
            out.normal_text,
            "I will check the weather.  Then check LA weather. "
        );
        let merged = out.coalesce_calls();
        assert_eq!(merged.calls.len(), 2);
        assert_eq!(merged.calls[0].arguments, r#"{"location":"NYC"}"#);
        assert_eq!(merged.calls[1].arguments, r#"{"location":"LA"}"#);
    }

    #[test]
    fn plain_text_still_flows_as_normal_text() {
        let out = parse_chunks(
            &weather_tools(),
            &["Hello, how", " can I help you", " today?"],
        );
        assert_eq!(out.normal_text, "Hello, how can I help you today?");
        assert!(out.calls.is_empty());
    }

    #[test]
    fn drops_truncated_bare_call_at_eof() {
        // Bare body with no closing `</tool_call>` before EOF: dropped (no leak,
        // no partial call).
        let out = parse_chunks(
            &weather_tools(),
            &["get_weather<arg_key>location</arg_key><arg_value>NY"],
        );
        assert_eq!(out.normal_text, "");
        assert!(out.calls.is_empty());
    }

    #[test]
    fn retains_bare_name_across_boundary_before_arg_key() {
        // Boundary falls BETWEEN the bare function name and its `<arg_key>`:
        // the name must be held (not emitted as normal_text) until the next
        // chunk supplies the anchor, or the bare call is lost.
        let mut parser = Glm47ToolStreamParser::new(&weather_tools());
        let first = parser.push("get_weather").expect("push");
        assert_eq!(
            first.normal_text, "",
            "pending bare name must not leak before its <arg_key>"
        );
        assert!(first.calls.is_empty());
        let second = parser
            .push("<arg_key>location</arg_key><arg_value>NYC</arg_value></tool_call>")
            .expect("push");
        assert_eq!(second.normal_text, "");
        let out = {
            let mut o = ToolParseResult::default();
            o.append(first);
            o.append(second);
            o.append(parser.finish().expect("finish"));
            o
        };
        let merged = out.coalesce_calls();
        assert_eq!(merged.calls.len(), 1);
        assert_eq!(merged.calls[0].name.as_deref(), Some("get_weather"));
        assert_eq!(merged.calls[0].arguments, r#"{"location":"NYC"}"#);
    }

    #[test]
    fn retains_bare_name_across_boundary_inside_arg_key() {
        // Boundary splits `<arg_key>` itself (`<arg` | `_key>`): both the
        // partial orphan anchor AND the bare name before it must be held, so
        // neither the name nor the `<arg` markup leaks into normal_text.
        let mut parser = Glm47ToolStreamParser::new(&weather_tools());
        let first = parser.push("get_weather<arg").expect("push");
        assert_eq!(
            first.normal_text, "",
            "name + partial <arg_key> must be held across the split"
        );
        assert!(first.calls.is_empty());
        let second = parser
            .push("_key>location</arg_key><arg_value>NYC</arg_value></tool_call>")
            .expect("push");
        assert_eq!(second.normal_text, "");
        let out = {
            let mut o = ToolParseResult::default();
            o.append(first);
            o.append(second);
            o.append(parser.finish().expect("finish"));
            o
        };
        let merged = out.coalesce_calls();
        assert_eq!(merged.calls.len(), 1);
        assert_eq!(merged.calls[0].name.as_deref(), Some("get_weather"));
        assert_eq!(merged.calls[0].arguments, r#"{"location":"NYC"}"#);
    }

    #[test]
    fn strips_split_orphan_close_marker() {
        // A lone `</tool_call>` (BLOCK_END) in prose with NO matching
        // `<tool_call>` opener and NO recoverable bare call before it is a true
        // orphan close (here split across a chunk boundary: `</tool` | `_call>`).
        // It must be stripped, never leaked into normal_text, and yield no calls.
        let out = parse_chunks(
            &weather_tools(),
            &["I will", " check that. ", "</tool", "_call> ok", ""],
        );
        assert_eq!(
            out.normal_text, "I will check that.  ok",
            "orphan </tool_call> leaked into normal_text: {:?}",
            out.normal_text
        );
        assert!(out.calls.is_empty(), "orphan close must not produce a call");
        assert!(
            !out.normal_text.contains("tool_call") && !out.normal_text.contains("arg_"),
            "tool markup leaked: {:?}",
            out.normal_text
        );
    }

    #[test]
    fn retains_bare_name_across_boundary_at_lone_angle() {
        // The ambiguous split `get_weather<` | `arg_key>...`: a lone trailing
        // `<` could open either `<tool_call>` or an orphan anchor, so the name
        // is held until disambiguated. Here it resolves to a bare call.
        let out = parse_chunks(
            &weather_tools(),
            &[
                "get_weather<",
                "arg_key>location</arg_key><arg_value>NYC</arg_value></tool_call>",
            ],
        );
        assert_eq!(out.normal_text, "");
        let merged = out.coalesce_calls();
        assert_eq!(merged.calls.len(), 1);
        assert_eq!(merged.calls[0].name.as_deref(), Some("get_weather"));
        assert_eq!(merged.calls[0].arguments, r#"{"location":"NYC"}"#);
    }
}
