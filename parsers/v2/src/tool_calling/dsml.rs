// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! DeepSeek V4 DSML grammar and the legacy ToolParser projection.
//!
//! `WrappedBlockScanner` is the sole DSML state machine. The public
//! `DeepSeekV4ToolStreamParser` remains for compatibility, but it projects the
//! native UnifiedParser's ordered events instead of maintaining another parser.

use serde_json::{Map, Value};

use crate::tool_calling::scan::{
    BareRecoveryLatch, GuidedInvokePrefix, GuidedInvokePrefixContext, InvokeBoundary,
    InvokeBoundaryFactory, InvokeEmitter, InvokeLatch, WrappedBlockScanner, WrappedBlockSpec,
};
use crate::tool_calling::traits::{Tool, ToolCallDelta, ToolParseResult, ToolParser};
use crate::unified::{UnifiedParser, UnifiedParserExt, UnifiedParserInit};

pub(crate) const BLOCK_START: &str = "<｜DSML｜tool_calls>";
pub(crate) const BLOCK_END: &str = "</｜DSML｜tool_calls>";
pub(crate) const INVOKE_START_PREFIX: &str = "<｜DSML｜invoke name=\"";
pub(crate) const INVOKE_END: &str = "</｜DSML｜invoke>";
pub(crate) const PARAMETER_PREFIX: &str = "<｜DSML｜parameter name=";
pub(crate) const PARAMETER_END: &str = "</｜DSML｜parameter>";

/// Construct the only DeepSeek V4 DSML state machine.
pub(crate) fn deepseek_v4_scanner(_tools: &[Tool]) -> WrappedBlockScanner<DsmlEmitter> {
    WrappedBlockScanner::new(
        WrappedBlockSpec {
            family: "deepseek_v4",
            block_starts: vec![BLOCK_START.to_string()],
            block_ends: vec![BLOCK_END.to_string()],
            invoke_start: INVOKE_START_PREFIX.to_string(),
            invoke_end: INVOKE_END.to_string(),
            orphan_markers: vec![BLOCK_END.to_string(), INVOKE_END.to_string()],
            holdback_markers: vec![
                BLOCK_START.to_string(),
                BLOCK_END.to_string(),
                INVOKE_START_PREFIX.to_string(),
                INVOKE_END.to_string(),
                PARAMETER_PREFIX.to_string(),
                PARAMETER_END.to_string(),
            ],
            bare_recovery_latch: BareRecoveryLatch::Set,
            invoke_latch: InvokeLatch::IfEmitted,
            // A DSML block may omit its outer close after a complete invoke.
            drop_invoke_crossing_block_end: false,
            invoke_boundary_factory: Some(InvokeBoundaryFactory::custom(dsml_invoke_boundary)),
            preserve_special_tokens: true,
        },
        DsmlEmitter,
    )
}

pub(crate) struct DsmlEmitter;

#[cfg(test)]
std::thread_local! {
    static BOUNDARY_EXAMINED_BYTES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static BOUNDARY_RETAINED_PREFIX_COMPARISONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static BOUNDARY_COPIED_BYTES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

fn count_boundary_bytes(bytes: usize) {
    #[cfg(test)]
    BOUNDARY_EXAMINED_BYTES.with(|examined| examined.set(examined.get() + bytes));
    #[cfg(not(test))]
    let _ = bytes;
}

#[cfg(test)]
pub(crate) fn reset_boundary_examined_bytes() {
    BOUNDARY_EXAMINED_BYTES.with(|examined| examined.set(0));
    BOUNDARY_RETAINED_PREFIX_COMPARISONS.with(|comparisons| comparisons.set(0));
    BOUNDARY_COPIED_BYTES.with(|copied| copied.set(0));
}

#[cfg(test)]
pub(crate) fn boundary_examined_bytes() -> usize {
    BOUNDARY_EXAMINED_BYTES.with(std::cell::Cell::get)
}

#[cfg(test)]
pub(crate) fn boundary_retained_prefix_comparisons() -> usize {
    BOUNDARY_RETAINED_PREFIX_COMPARISONS.with(std::cell::Cell::get)
}

#[cfg(test)]
pub(crate) fn boundary_copied_bytes() -> usize {
    BOUNDARY_COPIED_BYTES.with(std::cell::Cell::get)
}

#[derive(Default)]
struct DsmlInvokeBoundary {
    candidate_len: usize,
    cursor: usize,
    mode: DsmlLexMode,
    unterminated_parameter_block_end: Option<usize>,
    guided_prefix_scanned: usize,
    guided_prefix_starts_payload: bool,
    guided_prefix_has_terminator: bool,
}

enum DsmlLexMode {
    Header {
        name_closed: bool,
    },
    Body,
    ParameterHeader {
        in_quote: bool,
    },
    ParameterValue,
    Json {
        stack: Vec<char>,
        in_string: bool,
        escape: bool,
    },
}

impl Default for DsmlLexMode {
    fn default() -> Self {
        Self::Header { name_closed: false }
    }
}

fn dsml_invoke_boundary() -> Box<dyn InvokeBoundary> {
    Box::new(DsmlInvokeBoundary::default())
}

impl InvokeBoundary for DsmlInvokeBoundary {
    fn owns_guided_prefix(&self) -> bool {
        true
    }

    fn guided_prefix_append(
        &mut self,
        candidate: &str,
        _append: &str,
        context: GuidedInvokePrefixContext,
    ) -> Option<GuidedInvokePrefix> {
        if context.followed_by_competing_marker
            || !context.outside_reasoning
            || !context.payload_is_empty
        {
            return Some(GuidedInvokePrefix::Strip(INVOKE_START_PREFIX.len()));
        }
        if self.guided_prefix_scanned == 0 {
            if !candidate.starts_with(INVOKE_START_PREFIX) {
                return Some(GuidedInvokePrefix::NoMatch);
            }
            self.guided_prefix_scanned = INVOKE_START_PREFIX.len();
            count_boundary_bytes(INVOKE_START_PREFIX.len());
        }
        while self.guided_prefix_scanned < candidate.len() {
            let ch = candidate[self.guided_prefix_scanned..].chars().next()?;
            count_boundary_bytes(ch.len_utf8());
            if self.guided_prefix_scanned == INVOKE_START_PREFIX.len() && matches!(ch, '{' | '[') {
                self.guided_prefix_starts_payload = true;
            }
            if ch == '>' {
                self.guided_prefix_has_terminator = true;
            }
            self.guided_prefix_scanned += ch.len_utf8();
        }
        Some(if self.guided_prefix_starts_payload {
            GuidedInvokePrefix::Match(INVOKE_START_PREFIX.len())
        } else if self.guided_prefix_has_terminator {
            GuidedInvokePrefix::NoMatch
        } else {
            GuidedInvokePrefix::Pending
        })
    }

    fn end_append(
        &mut self,
        candidate: &str,
        append: &str,
        flush: bool,
        _tool_index: usize,
    ) -> Option<usize> {
        if candidate.len() != self.candidate_len + append.len() {
            self.reset();
        }
        self.candidate_len = candidate.len();
        if self.cursor == 0 {
            if !candidate.starts_with(INVOKE_START_PREFIX) {
                return None;
            }
            count_boundary_bytes(INVOKE_START_PREFIX.len());
            self.cursor = INVOKE_START_PREFIX.len();
        }

        while self.cursor < candidate.len() {
            let rest = &candidate[self.cursor..];
            match &mut self.mode {
                DsmlLexMode::Header { name_closed } => {
                    let ch = rest.chars().next()?;
                    count_boundary_bytes(ch.len_utf8());
                    if ch == '"' {
                        *name_closed = true;
                    } else if ch == '>' && *name_closed {
                        self.mode = DsmlLexMode::Body;
                    }
                    self.cursor += ch.len_utf8();
                }
                DsmlLexMode::Body => {
                    for marker in [INVOKE_END, BLOCK_END, PARAMETER_PREFIX] {
                        if marker.starts_with(rest) && rest.len() < marker.len() {
                            count_boundary_bytes(rest.len());
                            return None;
                        }
                    }
                    if rest.starts_with(INVOKE_END) {
                        count_boundary_bytes(INVOKE_END.len());
                        return Some(self.cursor + INVOKE_END.len());
                    }
                    if rest.starts_with(BLOCK_END) {
                        // Return the bytes before the outer close as a malformed
                        // invoke candidate. The emitter rejects it, then the shared
                        // scanner consumes the still-buffered block close and emits
                        // any following visible text.
                        count_boundary_bytes(BLOCK_END.len());
                        return Some(self.cursor);
                    }
                    if rest.starts_with(PARAMETER_PREFIX) {
                        count_boundary_bytes(PARAMETER_PREFIX.len());
                        self.cursor += PARAMETER_PREFIX.len();
                        self.mode = DsmlLexMode::ParameterHeader { in_quote: false };
                        continue;
                    }
                    let ch = rest.chars().next()?;
                    count_boundary_bytes(ch.len_utf8());
                    if matches!(ch, '{' | '[') {
                        self.mode = DsmlLexMode::Json {
                            stack: vec![if ch == '{' { '}' } else { ']' }],
                            in_string: false,
                            escape: false,
                        };
                    }
                    self.cursor += ch.len_utf8();
                }
                DsmlLexMode::ParameterHeader { in_quote } => {
                    let ch = rest.chars().next()?;
                    count_boundary_bytes(ch.len_utf8());
                    if ch == '"' {
                        *in_quote = !*in_quote;
                    } else if ch == '>' && !*in_quote {
                        self.mode = DsmlLexMode::ParameterValue;
                    }
                    self.cursor += ch.len_utf8();
                }
                DsmlLexMode::ParameterValue => {
                    for marker in [PARAMETER_END, INVOKE_END, BLOCK_END] {
                        if marker.starts_with(rest) && rest.len() < marker.len() {
                            count_boundary_bytes(rest.len());
                            return None;
                        }
                    }
                    if rest.starts_with(PARAMETER_END) {
                        count_boundary_bytes(PARAMETER_END.len());
                        self.cursor += PARAMETER_END.len();
                        self.mode = DsmlLexMode::Body;
                        self.unterminated_parameter_block_end = None;
                        continue;
                    }
                    if rest.starts_with(INVOKE_END) {
                        let invoke_end = self.cursor + INVOKE_END.len();
                        let after_invoke = candidate[invoke_end..].trim_start();
                        if after_invoke.is_empty() {
                            if flush {
                                count_boundary_bytes(INVOKE_END.len());
                                return Some(invoke_end);
                            }
                            return None;
                        }
                        if after_invoke.starts_with(BLOCK_END) {
                            count_boundary_bytes(INVOKE_END.len());
                            return Some(invoke_end);
                        }
                        if BLOCK_END.starts_with(after_invoke) {
                            return None;
                        }
                    }
                    if rest.starts_with(BLOCK_END) {
                        count_boundary_bytes(BLOCK_END.len());
                        self.unterminated_parameter_block_end
                            .get_or_insert(self.cursor);
                        self.cursor += BLOCK_END.len();
                        continue;
                    }
                    let ch = rest.chars().next()?;
                    count_boundary_bytes(ch.len_utf8());
                    self.cursor += ch.len_utf8();
                }
                DsmlLexMode::Json {
                    stack,
                    in_string,
                    escape,
                } => {
                    if !*in_string {
                        if BLOCK_END.starts_with(rest) && rest.len() < BLOCK_END.len() {
                            count_boundary_bytes(rest.len());
                            return None;
                        }
                        if rest.starts_with(BLOCK_END) {
                            count_boundary_bytes(BLOCK_END.len());
                            return Some(self.cursor);
                        }
                    }
                    let ch = rest.chars().next()?;
                    count_boundary_bytes(ch.len_utf8());
                    if *in_string {
                        if *escape {
                            *escape = false;
                        } else if ch == '\\' {
                            *escape = true;
                        } else if ch == '"' {
                            *in_string = false;
                        }
                    } else {
                        match ch {
                            '"' => *in_string = true,
                            '{' => stack.push('}'),
                            '[' => stack.push(']'),
                            '}' | ']' if stack.pop() != Some(ch) => return None,
                            _ => {}
                        }
                    }
                    self.cursor += ch.len_utf8();
                    if stack.is_empty() {
                        self.mode = DsmlLexMode::Body;
                    }
                }
            }
        }
        flush
            .then_some(self.unterminated_parameter_block_end)
            .flatten()
    }

    fn opens(&self, text: &str, at: usize) -> bool {
        text[at..].starts_with(INVOKE_START_PREFIX)
    }

    fn holdback(&self, text: &str) -> usize {
        crate::tool_calling::scan::marker_prefix_suffix_len(
            text,
            [
                INVOKE_START_PREFIX,
                INVOKE_END,
                PARAMETER_PREFIX,
                PARAMETER_END,
            ],
        )
    }

    fn resync(&mut self, _text: &str, _flush: bool, _tool_index: usize) -> Option<usize> {
        None
    }

    fn reset(&mut self) {
        *self = Self::default();
    }
}

impl InvokeEmitter for DsmlEmitter {
    fn parse_invoke(
        &mut self,
        invoke: &str,
        tool_index: usize,
    ) -> anyhow::Result<Option<ToolCallDelta>> {
        let Some((name, header_len)) = parse_invoke_header(invoke) else {
            return Ok(None);
        };
        let Some(body) = invoke[header_len..].strip_suffix(INVOKE_END) else {
            return Ok(None);
        };
        let arguments = serde_json::to_string(&parse_parameters(body)?)?;
        Ok(Some(ToolCallDelta {
            tool_index,
            name: Some(name),
            arguments,
            complete: true,
        }))
    }

    fn parse_invoke_deltas(
        &mut self,
        invoke: &str,
        tool_index: usize,
    ) -> anyhow::Result<Option<Vec<ToolCallDelta>>> {
        let Some(delta) = self.parse_invoke(invoke, tool_index)? else {
            return Ok(None);
        };
        Ok(Some(vec![
            ToolCallDelta {
                tool_index,
                name: delta.name,
                arguments: String::new(),
                complete: false,
            },
            ToolCallDelta {
                tool_index,
                name: None,
                arguments: delta.arguments,
                complete: true,
            },
        ]))
    }
}

/// Compatibility adapter for callers that still need `ToolParseResult`.
pub struct DeepSeekV4ToolStreamParser {
    parser: Box<dyn UnifiedParser>,
    initialized: bool,
}

impl DeepSeekV4ToolStreamParser {
    pub fn new() -> Self {
        Self::new_with_tools(&[])
    }

    pub fn new_with_tools(tools: &[Tool]) -> Self {
        Self {
            parser: crate::unified::deepseek_v4::deepseek_v4_unified(tools),
            initialized: false,
        }
    }

    fn initialize_native(&mut self) -> anyhow::Result<()> {
        if !self.initialized {
            self.parser
                .initialize_request(UnifiedParserInit::native(&[]))?;
            self.initialized = true;
        }
        Ok(())
    }
}

impl Default for DeepSeekV4ToolStreamParser {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolParser for DeepSeekV4ToolStreamParser {
    fn create(tools: &[Tool]) -> anyhow::Result<Box<dyn ToolParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self::new_with_tools(tools)))
    }

    fn preserve_special_tokens(&self) -> bool {
        self.parser.preserve_special_tokens()
    }

    fn push(&mut self, chunk: &str) -> anyhow::Result<ToolParseResult> {
        self.initialize_native()?;
        Ok(ToolParseResult::from_deltas(self.parser.push(chunk)?))
    }

    fn finish(&mut self) -> anyhow::Result<ToolParseResult> {
        self.initialize_native()?;
        Ok(ToolParseResult::from_deltas(self.parser.finish()?.events))
    }
}

fn parse_invoke_header(s: &str) -> Option<(String, usize)> {
    let after_prefix = s.strip_prefix(INVOKE_START_PREFIX)?;
    let name_end = after_prefix.find('"')?;
    let name = after_prefix[..name_end].trim().to_string();
    let rest = &after_prefix[name_end + 1..];
    let gt = rest.find('>')?;
    let header_len = INVOKE_START_PREFIX.len() + name_end + 1 + gt + 1;
    Some((name, header_len))
}

fn parse_parameters(body: &str) -> anyhow::Result<Map<String, Value>> {
    let mut params = Map::new();
    let mut cursor = 0;
    while let Some(rel_start) = body[cursor..].find(PARAMETER_PREFIX) {
        let start = cursor + rel_start + PARAMETER_PREFIX.len();
        let Some(after_name_quote) = body[start..].strip_prefix('"') else {
            cursor = start;
            continue;
        };
        let Some(name_end) = after_name_quote.find('"') else {
            break;
        };
        let name = after_name_quote[..name_end].trim();
        let attrs_start = start + 1 + name_end + 1;
        let Some(header_end_rel) = body[attrs_start..].find('>') else {
            break;
        };
        let attrs = &body[attrs_start..attrs_start + header_end_rel];
        let value_start = attrs_start + header_end_rel + 1;
        let Some(value_end_rel) = body[value_start..].find(PARAMETER_END) else {
            break;
        };
        let raw_value = body[value_start..value_start + value_end_rel].trim();
        let value = if attrs.contains(r#"string="true""#) {
            Value::String(raw_value.to_string())
        } else {
            serde_json::from_str(raw_value).unwrap_or_else(|_| Value::String(raw_value.to_string()))
        };
        params.insert(name.to_string(), value);
        cursor = value_start + value_end_rel + PARAMETER_END.len();
    }
    if params.is_empty()
        && let Ok(Value::Object(object)) = serde_json::from_str::<Value>(body.trim())
    {
        return Ok(object.into_iter().collect());
    }
    Ok(params)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::unified::{
        InvalidGuidedPayloadPolicy, UnifiedParserStartingState, UnifiedToolOutputMode,
        guided_append_work, reset_guided_append_work,
    };

    fn boundary_work_for_parameter_bytes(value_len: usize) -> (usize, usize, usize) {
        reset_boundary_examined_bytes();
        let input = format!(
            "{BLOCK_START}{INVOKE_START_PREFIX}run\">{PARAMETER_PREFIX}\"payload\" string=\"true\">{}{PARAMETER_END}",
            "x".repeat(value_len)
        );
        let mut scanner = deepseek_v4_scanner(&[]);
        for ch in input.chars() {
            scanner
                .push_ordered(&ch.to_string())
                .expect("push character");
        }
        scanner.finish_ordered().expect("finish");
        (
            boundary_examined_bytes(),
            boundary_retained_prefix_comparisons(),
            boundary_copied_bytes(),
        )
    }

    #[test]
    fn public_constructor_remains_zero_argument_and_factory_accepts_tools() {
        let _parser = DeepSeekV4ToolStreamParser::new();
        let tool = Tool {
            name: "run".into(),
            description: None,
            parameters: serde_json::json!({"type": "object"}),
            strict: None,
        };
        let _tool_aware = DeepSeekV4ToolStreamParser::new_with_tools(std::slice::from_ref(&tool));
        DeepSeekV4ToolStreamParser::create(&[tool]).expect("tool-aware factory");
    }

    #[test]
    fn invoke_boundary_work_is_linear_for_one_character_chunks() {
        let small_len = 4_096;
        let large_len = small_len * 2;
        let small_work = boundary_work_for_parameter_bytes(small_len);
        let large_work = boundary_work_for_parameter_bytes(large_len);
        println!("DSML boundary work: {small_len}={small_work:?}, {large_len}={large_work:?}");

        assert!(small_work.0 >= small_len);
        assert!(large_work.0 >= large_len);
        assert_eq!(
            large_work.0 - small_work.0,
            large_len - small_len,
            "each additional parameter byte must be examined exactly once"
        );
        assert!(
            large_work.0 <= small_work.0 * 2 + 256,
            "doubling payload size must roughly double scan work: {small_work:?} -> {large_work:?}"
        );
        assert_eq!(
            small_work.1, 0,
            "retained-prefix comparisons: {small_work:?}"
        );
        assert_eq!(
            large_work.1, 0,
            "retained-prefix comparisons: {large_work:?}"
        );
        assert_eq!(small_work.2, 0, "boundary-owned copies: {small_work:?}");
        assert_eq!(large_work.2, 0, "boundary-owned copies: {large_work:?}");
    }

    #[test]
    fn invoke_boundary_accepts_append_and_candidate_replacement_without_prefix_work() {
        reset_boundary_examined_bytes();
        let mut boundary = DsmlInvokeBoundary::default();
        let first = format!("{INVOKE_START_PREFIX}one\">partial");
        assert_eq!(boundary.end_append(&first, &first, false, 0), None);

        let replacement = format!("{INVOKE_START_PREFIX}two\">body{INVOKE_END}");
        assert_eq!(
            boundary.end_append(&replacement, &replacement, false, 0),
            Some(replacement.len())
        );
        assert_eq!(boundary_retained_prefix_comparisons(), 0);
        assert_eq!(boundary_copied_bytes(), 0);
    }

    #[test]
    fn guided_append_tracking_is_constant_work_per_chunk_and_resets_for_reuse() {
        fn work(value_len: usize) -> (usize, usize, usize) {
            reset_guided_append_work();
            reset_boundary_examined_bytes();
            let input = format!("{INVOKE_START_PREFIX}{}", "x".repeat(value_len));
            let mut parser = crate::unified::deepseek_v4::deepseek_v4_unified(&[]);
            parser
                .initialize_request(UnifiedParserInit {
                    starting_state: UnifiedParserStartingState::None,
                    tool_output_mode: UnifiedToolOutputMode::GuidedJson { named_tool: None },
                    invalid_guided_payload: InvalidGuidedPayloadPolicy::RecoverAsText,
                    ..UnifiedParserInit::default()
                })
                .expect("initialize guided parser");
            for ch in input.chars() {
                parser.push(&ch.to_string()).expect("push character");
            }
            parser.finish().expect("finish partial candidate");
            let append = guided_append_work();
            assert_eq!(boundary_retained_prefix_comparisons(), 0);
            assert_eq!(boundary_copied_bytes(), 0);
            append
        }

        let small = work(4_096);
        let large = work(8_192);
        println!("guided append work: 4096={small:?}, 8192={large:?}");
        assert_eq!(small, (0, 0, 0));
        assert_eq!(large, (0, 0, 0));

        let mut parser = crate::unified::deepseek_v4::deepseek_v4_unified(&[]);
        parser
            .initialize_request(UnifiedParserInit {
                tool_output_mode: UnifiedToolOutputMode::GuidedJson { named_tool: None },
                ..UnifiedParserInit::default()
            })
            .expect("initialize guided parser");
        parser
            .push(&format!("{INVOKE_START_PREFIX}abandoned"))
            .expect("partial first request");
        parser.reset();
        let payload = r#"[{"name":"weather","arguments":{"city":"Paris"}}]"#;
        let events = parser.push(payload).expect("replacement payload");
        assert_eq!(events.len(), 1);
        parser.finish().expect("finish replacement request");
    }

    #[test]
    fn legacy_parser_projects_the_unified_dsml_events() {
        let input = "before<think>reason</think><｜DSML｜tool_calls><｜DSML｜invoke name=\"get_weather\"><｜DSML｜parameter name=\"city\" string=\"true\">Paris</｜DSML｜parameter></｜DSML｜invoke></｜DSML｜tool_calls>after";
        let split = input.find("<｜DSML｜tool_calls>").expect("DSML block");
        let mut legacy = DeepSeekV4ToolStreamParser::default();
        let mut legacy_result = legacy.push(&input[..split]).expect("legacy prefix");
        legacy_result.append(legacy.push(&input[split..]).expect("legacy suffix"));
        legacy_result.append(legacy.finish().expect("legacy finish"));

        let mut unified = crate::unified::deepseek_v4::deepseek_v4_unified(&[]);
        unified
            .initialize_request(UnifiedParserInit::native(&[]))
            .expect("initialize unified");
        let mut unified_events = unified.push(&input[..split]).expect("unified prefix");
        unified_events.extend(unified.push(&input[split..]).expect("unified suffix"));
        unified_events.extend(unified.finish().expect("unified finish").events);

        assert_eq!(ToolParseResult::from_deltas(unified_events), legacy_result);
    }

    #[test]
    fn missing_parameter_close_keeps_call_with_empty_arguments_at_every_split() {
        let input = concat!(
            "<｜DSML｜tool_calls>",
            "<｜DSML｜invoke name=\"get_weather\">",
            "<｜DSML｜parameter name=\"location\" string=\"true\">NYC",
            "</｜DSML｜invoke>",
            "</｜DSML｜tool_calls>",
        );
        let expected = ToolParseResult {
            calls: vec![ToolCallDelta {
                tool_index: 0,
                name: Some("get_weather".to_string()),
                arguments: "{}".to_string(),
                complete: true,
            }],
            normal_text: String::new(),
        };

        for split in input
            .char_indices()
            .map(|(at, _)| at)
            .chain(std::iter::once(input.len()))
        {
            let mut parser = DeepSeekV4ToolStreamParser::new();
            let mut result = parser.push(&input[..split]).expect("prefix");
            result.append(parser.push(&input[split..]).expect("suffix"));
            result.append(parser.finish().expect("finish"));
            assert_eq!(result.coalesce_calls(), expected, "split at {split}");
        }
    }

    #[test]
    fn invoke_close_inside_parameter_value_remains_data() {
        let input = concat!(
            "<｜DSML｜tool_calls>",
            "<｜DSML｜invoke name=\"run\">",
            "<｜DSML｜parameter name=\"cmd\" string=\"true\">",
            "before</｜DSML｜invoke>after",
            "</｜DSML｜parameter>",
            "</｜DSML｜invoke>",
            "</｜DSML｜tool_calls>",
        );
        let mut parser = DeepSeekV4ToolStreamParser::new();
        let mut result = parser.push(input).expect("push");
        result.append(parser.finish().expect("finish"));
        assert_eq!(
            result.coalesce_calls().calls[0].arguments,
            r#"{"cmd":"before</｜DSML｜invoke>after"}"#
        );
    }
}
