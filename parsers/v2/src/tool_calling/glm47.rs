// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! GLM-4.7/5.x XML tool calls and their legacy ToolParser projection.
//!
//! GLM's outer `<tool_call>` block is itself the invoke: the function name is
//! followed directly by `<arg_key>`/`<arg_value>` pairs. `WrappedBlockScanner`
//! owns all buffering, recovery, and chunk-boundary handling; this module only
//! supplies the grammar and value emitter.

use crate::tool_calling::scan::{
    BareRecoveryLatch, GuidedInvokePrefix, GuidedInvokePrefixContext, InvokeBoundary,
    InvokeBoundaryFactory, InvokeEmitter, InvokeLatch, WrappedBlockScanner, WrappedBlockSpec,
    marker_prefix_suffix_len, reorder_arguments,
};
use crate::tool_calling::traits::{Tool, ToolCallDelta, ToolParseResult, ToolParser};
use crate::tool_calling::v1core::{Glm47ParserConfig, ToolDefinition, parse_glm47_invoke};

pub(crate) const BLOCK_START: &str = "<tool_call>";
pub(crate) const BLOCK_END: &str = "</tool_call>";
const ARG_KEY_START: &str = "<arg_key>";
const ARG_KEY_END: &str = "</arg_key>";
const ARG_VALUE_START: &str = "<arg_value>";
const ARG_VALUE_END: &str = "</arg_value>";

const ORPHAN_ANCHORS: [&str; 4] = [BLOCK_END, ARG_KEY_START, ARG_KEY_END, ARG_VALUE_START];

fn spec() -> WrappedBlockSpec {
    WrappedBlockSpec {
        family: "glm47",
        block_starts: vec![BLOCK_START.to_string()],
        block_ends: vec![BLOCK_END.to_string()],
        // The block opener is also the invoke opener. The scanner consumes the
        // opener as block markup and passes the body plus closer to the emitter.
        invoke_start: BLOCK_START.to_string(),
        invoke_end: BLOCK_END.to_string(),
        orphan_markers: ORPHAN_ANCHORS
            .iter()
            .map(|marker| (*marker).to_string())
            .collect(),
        holdback_markers: [
            BLOCK_START,
            BLOCK_END,
            ARG_KEY_START,
            ARG_KEY_END,
            ARG_VALUE_START,
        ]
        .into_iter()
        .map(str::to_string)
        .collect(),
        bare_recovery_latch: BareRecoveryLatch::Clear,
        invoke_latch: InvokeLatch::IfEmitted,
        invoke_boundary_factory: Some(InvokeBoundaryFactory::custom(glm47_boundary)),
        preserve_special_tokens: true,
        ..Default::default()
    }
}

pub(crate) struct Glm47Emitter {
    config: Glm47ParserConfig,
    tools: Vec<ToolDefinition>,
}

impl InvokeEmitter for Glm47Emitter {
    fn parse_invoke(
        &mut self,
        invoke: &str,
        tool_index: usize,
    ) -> anyhow::Result<Option<ToolCallDelta>> {
        let call = match parse_glm47_invoke(invoke, &self.config, Some(&self.tools)) {
            Ok(call) => call,
            Err(error) => {
                tracing::warn!(
                    why = "glm47_unparseable_invoke",
                    tool_index,
                    error = %error,
                    "GLM stream dropped a delimited invoke that failed value typing"
                );
                return Ok(None);
            }
        };
        Ok(Some(ToolCallDelta {
            tool_index,
            name: Some(call.function.name),
            arguments: reorder_arguments(&call.function.arguments, &source_arg_key_order(invoke)),
            complete: true,
        }))
    }
}

#[derive(Default)]
struct Glm47BoundaryProgress {
    cursor: usize,
    in_arg_value: bool,
    completed_arg_value: bool,
    possible_outer_end: Option<usize>,
}

impl Glm47BoundaryProgress {
    fn end(&mut self, text: &str, flush: bool) -> Option<usize> {
        while self.cursor < text.len() {
            let rest = &text[self.cursor..];
            if self.in_arg_value {
                if rest.starts_with(BLOCK_START) && self.possible_outer_end.is_some() {
                    let body = &rest[BLOCK_START.len()..];
                    match body.find(ARG_KEY_START) {
                        Some(at)
                            if !body[..at].is_empty()
                                && body[..at].chars().all(|ch| {
                                    ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '.')
                                }) =>
                        {
                            return self.possible_outer_end;
                        }
                        None if !flush => return None,
                        _ => {}
                    }
                }
                if !flush
                    && self.possible_outer_end.is_some()
                    && rest.len() < BLOCK_START.len()
                    && BLOCK_START.starts_with(rest)
                {
                    return None;
                }
                if rest.starts_with(ARG_VALUE_END) {
                    self.in_arg_value = false;
                    self.completed_arg_value = true;
                    self.possible_outer_end = None;
                    self.cursor += ARG_VALUE_END.len();
                    continue;
                }
                if rest.len() < ARG_VALUE_END.len() && ARG_VALUE_END.starts_with(rest) {
                    return flush.then_some(self.possible_outer_end).flatten();
                }
                if rest.starts_with(BLOCK_END) {
                    self.possible_outer_end
                        .get_or_insert(self.cursor + BLOCK_END.len());
                    self.cursor += BLOCK_END.len();
                    continue;
                }
                if rest.len() < BLOCK_END.len() && BLOCK_END.starts_with(rest) {
                    return flush.then_some(self.possible_outer_end).flatten();
                }
            } else {
                if rest.starts_with(ARG_VALUE_START) {
                    self.in_arg_value = true;
                    self.cursor += ARG_VALUE_START.len();
                    continue;
                }
                if rest.starts_with(BLOCK_END) {
                    return Some(self.cursor + BLOCK_END.len());
                }
                if (rest.len() < ARG_VALUE_START.len() && ARG_VALUE_START.starts_with(rest))
                    || (rest.len() < BLOCK_END.len() && BLOCK_END.starts_with(rest))
                {
                    return None;
                }
            }
            let ch = rest.chars().next().expect("cursor is before text end");
            self.cursor += ch.len_utf8();
        }
        if flush && !self.in_arg_value && self.completed_arg_value {
            Some(self.cursor)
        } else {
            flush.then_some(self.possible_outer_end).flatten()
        }
    }
}

#[derive(Default)]
struct Glm47Boundary {
    native: Glm47BoundaryProgress,
    guided: Glm47BoundaryProgress,
}

impl InvokeBoundary for Glm47Boundary {
    fn block_is_invoke(&self) -> bool {
        true
    }

    fn bare_invoke_start(&self, text: &str) -> Option<usize> {
        find_bare_invoke_start(text)
    }

    fn bare_invoke_holdback(&self, text: &str) -> usize {
        trailing_holdback_len(text)
    }

    fn accepts_bare_invoke(&self, invoke: &str) -> bool {
        if invoke.contains(ARG_KEY_START) {
            return true;
        }
        let Some(end) = Glm47BoundaryProgress::default().end(invoke, false) else {
            return false;
        };
        let invoke = &invoke[..end];
        let name = invoke.strip_suffix(BLOCK_END).unwrap_or(invoke).trim();
        !name.is_empty()
            && !name.ends_with('.')
            && name
                .chars()
                .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '.'))
    }

    fn bare_invoke_uses_eof_boundary(&self) -> bool {
        true
    }

    fn owns_guided_prefix(&self) -> bool {
        true
    }

    fn guided_invoke_at(&self, text: &str) -> Option<(usize, usize)> {
        text.find(BLOCK_START).map(|at| (at, BLOCK_START.len()))
    }

    fn is_guided_invoke_marker(&self, marker: &str) -> bool {
        marker == BLOCK_START
    }

    fn guided_prefix_append(
        &mut self,
        candidate: &str,
        _append: &str,
        context: GuidedInvokePrefixContext,
    ) -> Option<GuidedInvokePrefix> {
        let body = candidate.strip_prefix(BLOCK_START)?;
        if body.trim_start().starts_with(['{', '[']) {
            return Some(GuidedInvokePrefix::Match(BLOCK_START.len()));
        }
        if !context.outside_reasoning {
            return Some(GuidedInvokePrefix::Strip(BLOCK_START.len()));
        }
        self.guided
            .end(body, false)
            .map(|end| GuidedInvokePrefix::Strip(BLOCK_START.len() + end))
            .or(Some(GuidedInvokePrefix::Pending))
    }

    fn end_append(
        &mut self,
        candidate: &str,
        _append: &str,
        flush: bool,
        _tool_index: usize,
    ) -> Option<usize> {
        self.native.end(candidate, flush)
    }

    fn opens(&self, _text: &str, _at: usize) -> bool {
        true
    }

    fn holdback(&self, text: &str) -> usize {
        marker_prefix_suffix_len(text, [BLOCK_END, ARG_VALUE_START, ARG_VALUE_END])
    }

    fn resync(&mut self, _text: &str, _flush: bool, _tool_index: usize) -> Option<usize> {
        None
    }

    fn reset(&mut self) {
        *self = Self::default();
    }
}

fn glm47_boundary() -> Box<dyn InvokeBoundary> {
    Box::new(Glm47Boundary::default())
}

/// The one GLM scanner construction site shared by native UnifiedParser and
/// the legacy ToolParser compatibility surface.
pub(crate) fn glm47_scanner(tools: &[Tool]) -> WrappedBlockScanner<Glm47Emitter> {
    WrappedBlockScanner::new(
        spec(),
        Glm47Emitter {
            config: Glm47ParserConfig::default(),
            tools: tools.iter().map(ToolDefinition::from).collect(),
        },
    )
}

/// Compatibility projection for callers that still use the tool-only trait.
pub struct Glm47ToolStreamParser {
    scanner: WrappedBlockScanner<Glm47Emitter>,
}

impl Glm47ToolStreamParser {
    pub fn new(tools: &[Tool]) -> Self {
        Self {
            scanner: glm47_scanner(tools),
        }
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
        self.scanner.preserve_special_tokens()
    }

    fn push(&mut self, chunk: &str) -> anyhow::Result<ToolParseResult> {
        self.scanner.push(chunk)
    }

    fn finish(&mut self) -> anyhow::Result<ToolParseResult> {
        self.scanner.finish()
    }
}

fn find_bare_invoke_start(text: &str) -> Option<usize> {
    let marker_idx = ORPHAN_ANCHORS
        .iter()
        .filter_map(|marker| text.find(marker))
        .min()?;
    if text
        .find(BLOCK_START)
        .is_some_and(|wrapped| wrapped < marker_idx)
    {
        return None;
    }
    let before = text[..marker_idx].trim_end();
    let name_start = before
        .char_indices()
        .rev()
        .find(|(_, ch)| ch.is_whitespace())
        .map(|(idx, ch)| idx + ch.len_utf8())
        .unwrap_or(0);
    let candidate = before[name_start..].trim();
    (!candidate.is_empty()
        && candidate
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '.')))
    .then_some(name_start)
}

fn trailing_holdback_len(text: &str) -> usize {
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
        for length in 1..marker.len() {
            if text.ends_with(&marker[..length]) {
                if length > marker_keep {
                    marker_keep = length;
                    orphan_partial = is_orphan;
                } else if length == marker_keep && is_orphan {
                    orphan_partial = true;
                }
            }
        }
    }
    if !orphan_partial && marker_keep != 0 {
        return marker_keep;
    }
    let identifier_end = text.len() - marker_keep;
    let name_start = text[..identifier_end]
        .char_indices()
        .rev()
        .take_while(|(_, ch)| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '.'))
        .last()
        .map(|(idx, _)| idx)
        .unwrap_or(identifier_end);
    text.len() - name_start
}

fn source_arg_key_order(block: &str) -> Vec<String> {
    let mut names = Vec::new();
    let mut cursor = 0;
    while let Some(relative) = block[cursor..].find(ARG_KEY_START) {
        let start = cursor + relative + ARG_KEY_START.len();
        let Some(end) = block[start..].find(ARG_KEY_END) else {
            break;
        };
        let name = block[start..start + end].trim();
        if !name.is_empty() {
            names.push(name.to_string());
        }
        cursor = start + end + ARG_KEY_END.len();
    }
    names
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tools() -> Vec<Tool> {
        vec![
            Tool {
                name: "get_weather".into(),
                description: None,
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": { "city": { "type": "string" } }
                }),
                strict: None,
            },
            Tool {
                name: "get_time".into(),
                description: None,
                parameters: serde_json::json!({"type": "object", "properties": {}}),
                strict: None,
            },
            Tool {
                name: "run".into(),
                description: None,
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": { "cmd": { "type": "string" } }
                }),
                strict: None,
            },
        ]
    }

    fn legacy(tools: &[Tool], chunks: &[&str]) -> ToolParseResult {
        let mut parser = Glm47ToolStreamParser::new(tools);
        let mut output = ToolParseResult::default();
        for chunk in chunks {
            output.append(parser.push(chunk).expect("push"));
        }
        output.append(parser.finish().expect("finish"));
        output
    }

    #[test]
    fn large_integer_keeps_numeric_bytes_in_nullable_schemas() {
        let value = "9".repeat(400);
        for schema in [
            serde_json::json!({"type": "integer"}),
            serde_json::json!({"type": ["integer", "null"]}),
            serde_json::json!({"anyOf": [{"type": "integer"}, {"type": "null"}]}),
            serde_json::json!({"type": ["number", "null"]}),
        ] {
            let tools = vec![Tool {
                name: "inspect".into(),
                description: None,
                parameters: serde_json::json!({"properties": {"value": schema}}),
                strict: None,
            }];
            let input = format!(
                "<tool_call>inspect<arg_key>value</arg_key><arg_value>{value}</arg_value></tool_call>"
            );
            let result = legacy(&tools, &[&input]).coalesce_calls();
            assert_eq!(result.calls.len(), 1);
            // Parsing into Value here would impose the numeric limit under test.
            assert_eq!(result.calls[0].arguments, format!("{{\"value\":{value}}}"));
        }
    }

    #[test]
    fn composed_nonstring_arguments_keep_json_types() {
        for (schema, value) in [
            (
                serde_json::json!({"type": ["integer", "null"]}),
                serde_json::Value::Null,
            ),
            (
                serde_json::json!({"type": ["integer", "null"]}),
                serde_json::json!(42),
            ),
            (
                serde_json::json!({"anyOf": [{"type": "integer"}, {"type": "null"}]}),
                serde_json::json!(42),
            ),
            (
                serde_json::json!({"oneOf": [{"type": "null"}, {"type": "integer"}]}),
                serde_json::json!(42),
            ),
            (
                serde_json::json!({"type": ["number", "null"]}),
                serde_json::json!(1.25),
            ),
            (
                serde_json::json!({"type": ["boolean", "null"]}),
                serde_json::json!(false),
            ),
            (
                serde_json::json!({"allOf": [
                    {"type": ["integer", "string"]}, {"type": "integer"}
                ]}),
                serde_json::json!(7),
            ),
            (
                serde_json::json!({"allOf": [
                    {"minimum": 0}, {"type": "integer"}, {"type": ["string", "number"]}
                ]}),
                serde_json::json!(7),
            ),
            (
                serde_json::json!({"type": ["integer", "string"], "allOf": [
                    {"anyOf": [{"type": "integer"}, {"type": "null"}]}
                ]}),
                serde_json::json!(7),
            ),
        ] {
            let tools = vec![Tool {
                name: "inspect".into(),
                description: None,
                parameters: serde_json::json!({
                    "type": "object", "properties": {"value": schema}
                }),
                strict: None,
            }];
            let input = format!(
                "<tool_call>inspect<arg_key>value</arg_key><arg_value>{value}</arg_value></tool_call>"
            );
            let result = legacy(&tools, &[&input]).coalesce_calls();
            assert!(result.normal_text.is_empty());
            assert_eq!(result.calls.len(), 1);
            let arguments: serde_json::Value =
                serde_json::from_str(&result.calls[0].arguments).unwrap();
            assert_eq!(arguments, serde_json::json!({"value": value}));
        }
    }

    #[test]
    fn native_legacy_projection_parses_glm_xml() {
        let output = legacy(
            &tools(),
            &[
                "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>",
            ],
        );
        let calls = output.coalesce_calls();
        assert_eq!(calls.normal_text, "");
        assert_eq!(calls.calls[0].name.as_deref(), Some("get_weather"));
        assert_eq!(calls.calls[0].arguments, r#"{"city":"Paris"}"#);
    }

    #[test]
    fn legacy_projection_is_split_invariant() {
        let input = "before <tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call> after";
        let whole = legacy(&tools(), &[input]).coalesce_calls();
        for split in input.char_indices().map(|(at, _)| at).chain([input.len()]) {
            let split_output =
                legacy(&tools(), &[&input[..split], &input[split..]]).coalesce_calls();
            assert_eq!(split_output, whole, "split at {split}");
        }
    }

    #[test]
    fn legacy_recovers_known_no_argument_bare_call_at_every_valid_split() {
        let input = "get_time</tool_call>";
        let want = legacy(&tools(), &[input]).coalesce_calls();
        assert_eq!(want.normal_text, "");
        assert_eq!(want.calls.len(), 1);
        assert_eq!(want.calls[0].name.as_deref(), Some("get_time"));
        assert_eq!(want.calls[0].arguments, "{}");

        for split in input.char_indices().map(|(at, _)| at).chain([input.len()]) {
            assert_eq!(
                legacy(&tools(), &[&input[..split], &input[split..]]).coalesce_calls(),
                want,
                "split at {split}"
            );
        }
    }

    #[test]
    fn legacy_recovers_bare_call_with_arguments_at_every_valid_split() {
        let input = "run<arg_key>cmd</arg_key><arg_value>git status</arg_value></tool_call>";
        let want = legacy(&tools(), &[input]).coalesce_calls();
        assert_eq!(want.normal_text, "");
        assert_eq!(want.calls.len(), 1);
        assert_eq!(want.calls[0].name.as_deref(), Some("run"));
        assert_eq!(want.calls[0].arguments, r#"{"cmd":"git status"}"#);

        for split in input.char_indices().map(|(at, _)| at).chain([input.len()]) {
            assert_eq!(
                legacy(&tools(), &[&input[..split], &input[split..]]).coalesce_calls(),
                want,
                "split at {split}"
            );
        }
    }

    #[test]
    fn legacy_does_not_recover_punctuation_or_prose_before_orphan_close() {
        for input in ["get_time.</tool_call>", "Please wait café</tool_call>"] {
            let want = legacy(&tools(), &[input]).coalesce_calls();
            let expected_calls = usize::from(input.contains(BLOCK_START));
            assert_eq!(
                want.calls.len(),
                expected_calls,
                "unexpected calls for {input:?}"
            );
            let expected_text = input
                .split_once(BLOCK_END)
                .map(|(prefix, _)| prefix)
                .unwrap_or(input);
            assert_eq!(want.normal_text, expected_text);
            for split in input.char_indices().map(|(at, _)| at).chain([input.len()]) {
                assert_eq!(
                    legacy(&tools(), &[&input[..split], &input[split..]]).coalesce_calls(),
                    want,
                    "split at {split} for {input:?}"
                );
            }
        }
    }

    #[test]
    fn legacy_recovers_unknown_bare_tool_without_arguments_at_every_split() {
        let input = "foo</tool_call>";
        let want = legacy(&tools(), &[input]).coalesce_calls();
        assert_eq!(want.normal_text, "");
        assert_eq!(want.calls.len(), 1);
        assert_eq!(want.calls[0].name.as_deref(), Some("foo"));
        assert_eq!(want.calls[0].arguments, "{}");
        for split in input.char_indices().map(|(at, _)| at).chain([input.len()]) {
            assert_eq!(
                legacy(&tools(), &[&input[..split], &input[split..]]).coalesce_calls(),
                want,
                "split at {split}"
            );
        }
    }

    #[test]
    fn legacy_recovers_the_last_identifier_before_an_orphan_close() {
        let input = "Please wait</tool_call><tool_call>get_time</tool_call>";
        let want = legacy(&tools(), &[input]).coalesce_calls();
        assert_eq!(want.normal_text, "Please ");
        assert_eq!(want.calls.len(), 2);
        assert_eq!(want.calls[0].name.as_deref(), Some("wait"));
        assert_eq!(want.calls[1].name.as_deref(), Some("get_time"));
        for split in input.char_indices().map(|(at, _)| at).chain([input.len()]) {
            assert_eq!(
                legacy(&tools(), &[&input[..split], &input[split..]]).coalesce_calls(),
                want,
                "split at {split}"
            );
        }
    }

    #[test]
    fn legacy_preserves_embedded_close_and_finds_the_following_call_at_every_split() {
        let input = "<tool_call>run<arg_key>cmd</arg_key><arg_value>git log </tool_call> --oneline</arg_value></tool_call> café <tool_call>get_time</tool_call>";
        let want = legacy(&tools(), &[input]).coalesce_calls();
        assert_eq!(want.normal_text, " café ");
        assert_eq!(want.calls.len(), 2);
        assert_eq!(want.calls[0].name.as_deref(), Some("run"));
        assert_eq!(
            want.calls[0].arguments,
            r#"{"cmd":"git log </tool_call> --oneline"}"#
        );
        assert_eq!(want.calls[1].name.as_deref(), Some("get_time"));

        for split in input.char_indices().map(|(at, _)| at).chain([input.len()]) {
            assert_eq!(
                legacy(&tools(), &[&input[..split], &input[split..]]).coalesce_calls(),
                want,
                "split at {split}"
            );
        }
    }

    #[test]
    fn legacy_recovers_at_possible_outer_close_when_argument_value_never_closes() {
        let input = "<tool_call>run<arg_key>cmd</arg_key><arg_value>git log </tool_call> --oneline";
        let want = legacy(&tools(), &[input]).coalesce_calls();
        assert_eq!(want.normal_text, " --oneline");
        assert_eq!(want.calls.len(), 1);
        assert_eq!(want.calls[0].arguments, r#"{"cmd":"git log "}"#);
        for split in input.char_indices().map(|(at, _)| at).chain([input.len()]) {
            assert_eq!(
                legacy(&tools(), &[&input[..split], &input[split..]]).coalesce_calls(),
                want,
                "split at {split}"
            );
        }
    }

    #[test]
    fn legacy_preserves_close_and_open_markers_inside_an_argument() {
        let input = "<tool_call>run<arg_key>cmd</arg_key><arg_value>before </tool_call><tool_call> after</arg_value></tool_call><tool_call>get_time</tool_call>";
        let want = legacy(&tools(), &[input]).coalesce_calls();
        assert_eq!(want.calls.len(), 2);
        assert_eq!(
            want.calls[0].arguments,
            r#"{"cmd":"before </tool_call><tool_call> after"}"#
        );
        for split in input.char_indices().map(|(at, _)| at).chain([input.len()]) {
            assert_eq!(
                legacy(&tools(), &[&input[..split], &input[split..]]).coalesce_calls(),
                want,
                "split at {split}"
            );
        }
    }

    #[test]
    fn legacy_preserves_parameterless_call_shape_inside_an_argument() {
        let input = "<tool_call>run<arg_key>cmd</arg_key><arg_value>before </tool_call><tool_call>get_weather</tool_call> after</arg_value></tool_call>";
        let want = legacy(&tools(), &[input]).coalesce_calls();
        assert_eq!(want.normal_text, "");
        assert_eq!(want.calls.len(), 1);
        assert_eq!(want.calls[0].name.as_deref(), Some("run"));
        assert_eq!(
            want.calls[0].arguments,
            r#"{"cmd":"before </tool_call><tool_call>get_weather</tool_call> after"}"#
        );

        for split in input.char_indices().map(|(at, _)| at).chain([input.len()]) {
            assert_eq!(
                legacy(&tools(), &[&input[..split], &input[split..]]).coalesce_calls(),
                want,
                "split at {split}"
            );
        }
    }

    #[test]
    fn legacy_unclosed_argument_recovers_before_a_following_call() {
        for input in [
            "<tool_call>run<arg_key>cmd</arg_key><arg_value>first</tool_call><tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>",
            "run<arg_key>cmd</arg_key><arg_value>first</tool_call><tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>",
        ] {
            let want = legacy(&tools(), &[input]).coalesce_calls();
            assert_eq!(want.calls.len(), 2, "input {input:?}");
            assert_eq!(want.calls[0].arguments, r#"{"cmd":"first"}"#);
            assert_eq!(want.calls[1].name.as_deref(), Some("get_weather"));
            assert_eq!(want.calls[1].arguments, r#"{"city":"Paris"}"#);
            for split in input.char_indices().map(|(at, _)| at).chain([input.len()]) {
                assert_eq!(
                    legacy(&tools(), &[&input[..split], &input[split..]]).coalesce_calls(),
                    want,
                    "split at {split} for {input:?}"
                );
            }
        }
    }

    #[test]
    fn legacy_recovers_missing_argument_value_close_at_terminal_outer_close() {
        let input = "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</tool_call>";
        let want = legacy(&tools(), &[input]).coalesce_calls();
        assert_eq!(want.normal_text, "");
        assert_eq!(want.calls.len(), 1);
        assert_eq!(want.calls[0].arguments, r#"{"city":"Paris"}"#);
        for split in input.char_indices().map(|(at, _)| at).chain([input.len()]) {
            assert_eq!(
                legacy(&tools(), &[&input[..split], &input[split..]]).coalesce_calls(),
                want,
                "split at {split}"
            );
        }
    }

    #[test]
    fn saved_outer_closer_recovers_wrapped_and_bare_calls_at_every_split() {
        let wrapped = "<tool_call>run<arg_key>cmd</arg_key><arg_value>first</tool_call>";
        let bare = "run<arg_key>cmd</arg_key><arg_value>first</tool_call>";
        for input in [wrapped, bare] {
            let want = legacy(&tools(), &[input]).coalesce_calls();
            assert_eq!(want.normal_text, "");
            assert_eq!(want.calls.len(), 1);
            assert_eq!(want.calls[0].name.as_deref(), Some("run"));
            assert_eq!(want.calls[0].arguments, r#"{"cmd":"first"}"#);
            for split in input.char_indices().map(|(at, _)| at).chain([input.len()]) {
                assert_eq!(
                    legacy(&tools(), &[&input[..split], &input[split..]]).coalesce_calls(),
                    want,
                    "split at {split} for {input:?}"
                );
            }
        }
    }

    #[test]
    fn saved_outer_closer_survives_partial_marker_eof_without_recovering_missing_closers() {
        let wrapped = "<tool_call>run<arg_key>cmd</arg_key><arg_value>first</tool_call>";
        for marker in [ARG_VALUE_END, BLOCK_END] {
            for (at, _) in marker.char_indices().skip(1) {
                let input = format!("{wrapped}{}", &marker[..at]);
                let got = legacy(&tools(), &[&input]).coalesce_calls();
                assert_eq!(got.calls.len(), 1, "partial {marker:?} at {at}");
                assert_eq!(got.calls[0].arguments, r#"{"cmd":"first"}"#);
                assert_eq!(got.normal_text, &marker[..at]);
            }
        }

        for tail in ["", "</not_tool_call>"] {
            let input = format!("{wrapped}{tail}");
            let got = legacy(&tools(), &[&input]).coalesce_calls();
            assert_eq!(got.calls.len(), 1, "tail {tail:?}");
            assert_eq!(got.calls[0].arguments, r#"{"cmd":"first"}"#);
        }

        for input in [
            "<tool_call>run<arg_key>cmd</arg_key><arg_value>first<",
            "run<arg_key>cmd</arg_key><arg_value>first<",
        ] {
            assert!(
                legacy(&tools(), &[input]).coalesce_calls().calls.is_empty(),
                "missing real closer must not recover: {input:?}"
            );
        }
    }

    #[test]
    fn legacy_drops_malformed_block_and_keeps_following_call_at_every_split() {
        let input = "<tool_call></tool_call><tool_call>get_time</tool_call>";
        let want = legacy(&tools(), &[input]).coalesce_calls();
        assert_eq!(want.normal_text, "");
        assert_eq!(want.calls.len(), 1);
        assert_eq!(want.calls[0].name.as_deref(), Some("get_time"));
        for split in input.char_indices().map(|(at, _)| at).chain([input.len()]) {
            assert_eq!(
                legacy(&tools(), &[&input[..split], &input[split..]]).coalesce_calls(),
                want,
                "split at {split}"
            );
        }
    }

    #[test]
    fn malformed_and_eof_tool_markup_is_dropped() {
        let output = legacy(&tools(), &["visible <tool_call>get_weather<arg_key>city"]);
        assert_eq!(output.normal_text, "visible ");
        assert!(output.calls.is_empty());
    }

    #[test]
    fn complete_argument_body_recovers_without_outer_close_at_every_split() {
        let input = "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value>";
        let want = legacy(&tools(), &[input]).coalesce_calls();
        assert_eq!(want.normal_text, "");
        assert_eq!(want.calls.len(), 1);
        assert_eq!(want.calls[0].name.as_deref(), Some("get_weather"));
        assert_eq!(want.calls[0].arguments, r#"{"city":"Paris"}"#);
        for split in input.char_indices().map(|(at, _)| at).chain([input.len()]) {
            assert_eq!(
                legacy(&tools(), &[&input[..split], &input[split..]]).coalesce_calls(),
                want,
                "split at {split}"
            );
        }
    }

    #[test]
    fn bare_name_is_held_until_its_argument_marker_arrives() {
        let mut parser = Glm47ToolStreamParser::new(&tools());
        assert!(
            parser
                .push("get_weather")
                .expect("push")
                .normal_text
                .is_empty()
        );
        let mut output = parser
            .push("<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>")
            .expect("push");
        output.append(parser.finish().expect("finish"));
        let calls = output.coalesce_calls();
        assert_eq!(calls.normal_text, "");
        assert_eq!(calls.calls.len(), 1);
    }
}
