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
use std::collections::HashSet;

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
    partial: Option<PartialArguments>,
    pending_header: Option<PendingInvokeHeader>,
    #[cfg(test)]
    searched_bytes: usize,
    #[cfg(test)]
    serialized_bytes: usize,
}

struct PendingInvokeHeader {
    tool_index: usize,
    search_cursor: usize,
}

/// Cursors refer to the scanner-owned invoke, never a second input buffer.
struct PartialArguments {
    tool_index: usize,
    name: String,
    emitted_json: String,
    scan_cursor: usize,
    seen_keys: HashSet<String>,
    header: Option<PartialArgumentHeader>,
    active: Option<PartialArgument>,
    blocked: bool,
}

struct PartialArgumentHeader {
    arg_start: usize,
    key_start: usize,
    cursor: usize,
    key: Option<String>,
}

struct PartialArgument {
    arg_start: usize,
    value_cursor: usize,
    search_cursor: usize,
    // Until </arg_value> arrives, this may be the EOF-recovery outer fence.
    ambiguous_outer_close: Option<usize>,
    streamable: bool,
}

fn is_unambiguous_string(schema: &serde_json::Value) -> bool {
    schema.get("type").and_then(serde_json::Value::as_str) == Some("string")
        && [
            "const",
            "nullable",
            "$ref",
            "$dynamicRef",
            "$recursiveRef",
            "allOf",
            "anyOf",
            "oneOf",
            "not",
            "if",
            "then",
            "else",
        ]
        .iter()
        .all(|keyword| schema.get(*keyword).is_none())
}

impl InvokeEmitter for Glm47Emitter {
    fn parse_partial_invoke(
        &mut self,
        invoke: &str,
        tool_index: usize,
    ) -> anyhow::Result<Option<ToolCallDelta>> {
        if self
            .partial
            .as_ref()
            .is_some_and(|p| p.tool_index != tool_index)
            || self
                .pending_header
                .as_ref()
                .is_some_and(|header| header.tool_index != tool_index)
        {
            self.abandon_invoke();
        }
        if self.partial.is_none() {
            let header = self.pending_header.get_or_insert(PendingInvokeHeader {
                tool_index,
                search_cursor: 0,
            });
            let tail = &invoke[header.search_cursor..];
            #[cfg(test)]
            {
                self.searched_bytes += tail.len();
            }
            let Some(relative) = tail.find(ARG_KEY_START) else {
                header.search_cursor =
                    invoke.len() - marker_prefix_suffix_len(tail, [ARG_KEY_START]);
                return Ok(None);
            };
            let header_end = header.search_cursor + relative;
            let name = invoke[..header_end].trim();
            self.partial = Some(PartialArguments {
                tool_index,
                name: name.to_string(),
                emitted_json: String::new(),
                scan_cursor: header_end,
                seen_keys: HashSet::new(),
                header: None,
                active: None,
                blocked: !self.tools.iter().any(|tool| tool.name == name),
            });
            self.pending_header = None;
        }

        let partial = self.partial.as_mut().expect("initialized above");
        if partial.blocked {
            return Ok(None);
        }
        let mut arguments = String::new();
        loop {
            if let Some(active) = partial.active.as_mut() {
                let search = &invoke[active.search_cursor..];
                #[cfg(test)]
                {
                    self.searched_bytes += search.len();
                }
                let close = search
                    .find(ARG_VALUE_END)
                    .map(|at| active.search_cursor + at);
                if close.is_none() && active.ambiguous_outer_close.is_none() {
                    #[cfg(test)]
                    {
                        self.searched_bytes += search.len();
                    }
                    active.ambiguous_outer_close =
                        search.find(BLOCK_END).map(|at| active.search_cursor + at);
                }
                // Advance searches even while an ambiguous outer closer withholds output.
                active.search_cursor =
                    invoke.len() - marker_prefix_suffix_len(search, [ARG_VALUE_END, BLOCK_END]);
                if active.streamable {
                    let safe_end = close
                        .or(active.ambiguous_outer_close)
                        .unwrap_or(active.search_cursor);
                    let raw = &invoke[active.value_cursor..safe_end];
                    if !raw.is_empty() {
                        #[cfg(test)]
                        {
                            self.serialized_bytes += raw.len();
                        }
                        let encoded = serde_json::to_string(raw)?;
                        arguments.push_str(&encoded[1..encoded.len() - 1]);
                    }
                    active.value_cursor = safe_end;
                    if close.is_some() {
                        arguments.push('"');
                    }
                } else if let Some(close) = close {
                    // Reuse batch coercion only for this closed argument.
                    let single = format!(
                        "{}{}",
                        partial.name,
                        &invoke[active.arg_start..close + ARG_VALUE_END.len()]
                    );
                    let call = parse_glm47_invoke(&single, &self.config, Some(&self.tools))?;
                    let json = call.function.arguments;
                    arguments.push(if partial.emitted_json.is_empty() && arguments.is_empty() {
                        '{'
                    } else {
                        ','
                    });
                    arguments.push_str(&json[1..json.len() - 1]);
                }
                let Some(close) = close else {
                    break;
                };
                partial.scan_cursor = close + ARG_VALUE_END.len();
                partial.active = None;
                continue;
            }

            let mut header = if let Some(header) = partial.header.take() {
                header
            } else {
                let tail = &invoke[partial.scan_cursor..];
                #[cfg(test)]
                {
                    self.searched_bytes += tail.len();
                }
                let Some(relative) = tail.find(ARG_KEY_START) else {
                    partial.scan_cursor =
                        invoke.len() - marker_prefix_suffix_len(tail, [ARG_KEY_START]);
                    break;
                };
                let arg_start = partial.scan_cursor + relative;
                let key_start = arg_start + ARG_KEY_START.len();
                PartialArgumentHeader {
                    arg_start,
                    key_start,
                    cursor: key_start,
                    key: None,
                }
            };
            if header.key.is_none() {
                let tail = &invoke[header.cursor..];
                #[cfg(test)]
                {
                    self.searched_bytes += tail.len();
                }
                let Some(relative) = tail.find(ARG_KEY_END) else {
                    header.cursor = invoke.len() - marker_prefix_suffix_len(tail, [ARG_KEY_END]);
                    partial.header = Some(header);
                    break;
                };
                let key_end = header.cursor + relative;
                let raw_key = &invoke[header.key_start..key_end];
                let key = raw_key.trim();
                if key.is_empty() || raw_key.contains('<') || partial.seen_keys.contains(key) {
                    partial.blocked = true;
                    break;
                }
                header.key = Some(key.to_string());
                header.cursor = key_end + ARG_KEY_END.len();
            }
            let after_key = &invoke[header.cursor..];
            let value_header = after_key.trim_start();
            let whitespace = after_key.len() - value_header.len();
            header.cursor += whitespace;
            #[cfg(test)]
            {
                self.searched_bytes += whitespace + value_header.len().min(ARG_VALUE_START.len());
            }
            if !value_header.starts_with(ARG_VALUE_START) {
                if !ARG_VALUE_START.starts_with(value_header) {
                    partial.blocked = true;
                }
                partial.header = Some(header);
                break;
            }
            let value_start = header.cursor + ARG_VALUE_START.len();
            let key = header
                .key
                .as_deref()
                .expect("key parsed before value header");
            let streamable = self
                .tools
                .iter()
                .find(|tool| tool.name == partial.name)
                .and_then(|tool| tool.parameters.as_ref())
                .and_then(|schema| schema.get("properties"))
                .and_then(|properties| properties.get(key))
                .is_some_and(is_unambiguous_string);
            partial.seen_keys.insert(key.to_string());
            if streamable {
                arguments.push(if partial.emitted_json.is_empty() && arguments.is_empty() {
                    '{'
                } else {
                    ','
                });
                arguments.push_str(&serde_json::to_string(key)?);
                arguments.push_str(":\"");
            }
            partial.active = Some(PartialArgument {
                arg_start: header.arg_start,
                value_cursor: value_start,
                search_cursor: value_start,
                ambiguous_outer_close: None,
                streamable,
            });
        }
        if arguments.is_empty() {
            return Ok(None);
        }
        let first = partial.emitted_json.is_empty();
        partial.emitted_json.push_str(&arguments);
        Ok(Some(ToolCallDelta {
            tool_index,
            name: first.then(|| partial.name.clone()),
            arguments,
            complete: false,
        }))
    }

    fn parse_invoke(
        &mut self,
        invoke: &str,
        tool_index: usize,
    ) -> anyhow::Result<Option<ToolCallDelta>> {
        let partial = self.partial.take();
        self.pending_header = None;
        let streamed = partial.as_ref().filter(|p| !p.emitted_json.is_empty());
        let incomplete = || ToolCallDelta {
            tool_index,
            name: None,
            arguments: String::new(),
            complete: false,
        };
        // EOF recovery must not validate a truncated next argument header.
        if has_incomplete_argument_header(invoke)
            || (!invoke.ends_with(BLOCK_END) && !invoke.trim_end().ends_with(ARG_VALUE_END))
        {
            tracing::warn!(
                why = "glm47_incomplete_argument_tail",
                tool_index,
                "GLM stream dropped an invoke ending inside an argument header"
            );
            return Ok(streamed.map(|_| incomplete()));
        }
        let call = match parse_glm47_invoke(invoke, &self.config, Some(&self.tools)) {
            Ok(call) => call,
            Err(error) => {
                tracing::warn!(
                    why = "glm47_unparseable_invoke",
                    tool_index,
                    error = %error,
                    "GLM stream dropped a delimited invoke that failed value typing"
                );
                // Reserve an emitted index so a later call cannot complete its prefix.
                return Ok(streamed.map(|_| incomplete()));
            }
        };
        let arguments = reorder_arguments(&call.function.arguments, &source_arg_key_order(invoke));
        let arguments = if let Some(partial) = streamed {
            if partial.tool_index != tool_index
                || partial.name != call.function.name
                || !arguments.starts_with(&partial.emitted_json)
            {
                // Last-write-wins cannot retract already-published duplicate values.
                tracing::warn!(
                    why = "glm47_streamed_call_prefix_mismatch",
                    tool_index,
                    "streamed GLM arguments no longer match the completed call"
                );
                return Ok(Some(incomplete()));
            }
            arguments[partial.emitted_json.len()..].to_string()
        } else {
            arguments
        };
        Ok(Some(ToolCallDelta {
            tool_index,
            name: streamed.is_none().then_some(call.function.name),
            arguments,
            complete: true,
        }))
    }

    fn abandon_invoke(&mut self) {
        self.partial = None;
        self.pending_header = None;
    }

    fn reset(&mut self) {
        self.abandon_invoke();
    }
}

/// A terminal outer fence does not complete an abandoned next parameter.
/// Skip closed values so literal argument markers inside strings remain data.
/// Missing value-close recovery remains the boundary scanner's responsibility.
fn has_incomplete_argument_header(invoke: &str) -> bool {
    let mut rest = invoke.strip_suffix(BLOCK_END).unwrap_or(invoke);
    let mut after_value = false;
    loop {
        let Some(start) = rest.find(ARG_KEY_START) else {
            let tail = rest.trim();
            return after_value && !tail.is_empty() && ARG_KEY_START.starts_with(tail);
        };
        rest = &rest[start + ARG_KEY_START.len()..];
        let Some(end) = rest.find(ARG_KEY_END) else {
            return true;
        };
        rest = rest[end + ARG_KEY_END.len()..].trim_start();
        let Some(value) = rest.strip_prefix(ARG_VALUE_START) else {
            return true;
        };
        let Some(end) = value.find(ARG_VALUE_END) else {
            return false;
        };
        rest = &value[end + ARG_VALUE_END.len()..];
        after_value = true;
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
        !name.is_empty() && !name.ends_with('.') && name.chars().all(is_tool_identifier_char)
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

    fn owns_prose_invoke(&self, text: &str, at: usize) -> bool {
        let Some(body) = text[at..].strip_prefix(BLOCK_START) else {
            return false;
        };
        let name = body.split('<').next().unwrap_or(body).trim();
        !name.is_empty() && name.chars().all(is_tool_identifier_char)
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
            partial: None,
            pending_header: None,
            #[cfg(test)]
            searched_bytes: 0,
            #[cfg(test)]
            serialized_bytes: 0,
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

fn is_tool_identifier_char(ch: char) -> bool {
    ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '.')
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
    (!candidate.is_empty() && candidate.chars().all(is_tool_identifier_char)).then_some(name_start)
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
        cursor = start + end + ARG_KEY_END.len();
        let value_header = block[cursor..].trim_start();
        if !value_header.starts_with(ARG_VALUE_START) {
            continue;
        }
        if !name.is_empty() {
            names.push(name.to_string());
        }
        let value_start = block.len() - value_header.len() + ARG_VALUE_START.len();
        cursor = block[value_start..]
            .find(ARG_VALUE_END)
            .map_or(block.len(), |end| value_start + end + ARG_VALUE_END.len());
    }
    names
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_long_string_streams_before_value_and_call_close() {
        let tools = vec![Tool {
            name: "write_file".into(),
            description: None,
            parameters: serde_json::json!({
                "type": "object",
                "properties": {"content": {"type": "string"}}
            }),
            strict: None,
        }];
        let mut parser = Glm47ToolStreamParser::new(&tools);
        let mut output = parser
            .push("<tool_call>write_file<arg_key>content</arg_key><arg_value>")
            .unwrap();
        let piece = "x".repeat(4096);

        for chunk in 0..24 {
            let update = parser.push(&piece).unwrap();
            assert!(
                update.calls.iter().any(|call| !call.arguments.is_empty()),
                "chunk {chunk} produced no arguments before either closing tag"
            );
            assert!(
                update
                    .calls
                    .iter()
                    .all(|call| { call.tool_index == 0 && !call.complete })
            );
            output.append(update);
        }

        output.append(parser.push("</arg_value></tool_call>").unwrap());
        output.append(parser.finish().unwrap());
        assert_eq!(
            output
                .calls
                .iter()
                .filter(|call| call.name.is_some())
                .count(),
            1
        );
        assert_eq!(output.calls.iter().filter(|call| call.complete).count(), 1);
        let complete = output.coalesce_calls();
        assert_eq!(complete.calls.len(), 1);
        assert_eq!(complete.calls[0].name.as_deref(), Some("write_file"));
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&complete.calls[0].arguments).unwrap(),
            serde_json::json!({"content": piece.repeat(24)})
        );
    }

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

    fn file_tools() -> Vec<Tool> {
        vec![Tool {
            name: "write_file".into(),
            description: None,
            parameters: serde_json::json!({"properties": {
                "count": {"type": "integer"},
                "path": {"type": "string"},
                "content": {"type": "string"}
            }}),
            strict: None,
        }]
    }

    fn char_chunks(input: &str) -> Vec<&str> {
        input
            .char_indices()
            .map(|(at, ch)| &input[at..at + ch.len_utf8()])
            .collect()
    }

    fn batch_arguments(tools: &[Tool], input: &str) -> String {
        let definitions: Vec<_> = tools.iter().map(ToolDefinition::from).collect();
        let invoke = input.strip_prefix(BLOCK_START).unwrap_or(input);
        let call = parse_glm47_invoke(invoke, &Glm47ParserConfig::default(), Some(&definitions))
            .expect("batch parse");
        reorder_arguments(&call.function.arguments, &source_arg_key_order(invoke))
    }

    fn assert_append_only(output: ToolParseResult, expected: &str) {
        let mut arguments = String::new();
        let mut names = 0;
        let mut completions = 0;
        for delta in &output.calls {
            assert_eq!(delta.tool_index, 0);
            assert_eq!(completions, 0, "delta after completion");
            names += usize::from(delta.name.is_some());
            completions += usize::from(delta.complete);
            arguments.push_str(&delta.arguments);
            assert!(
                expected.starts_with(&arguments),
                "not a batch prefix: {arguments:?}"
            );
        }
        assert_eq!(names, 1);
        assert_eq!(completions, 1);
        assert_eq!(arguments, expected);
        let assembled = output.coalesce_calls();
        assert_eq!(assembled.calls.len(), 1);
        assert_eq!(assembled.calls[0].arguments, expected);
    }

    #[test]
    fn native_long_content_emits_repeatedly_before_both_closers() {
        let tools = file_tools();
        let mut parser = Glm47ToolStreamParser::new(&tools);
        let opener = "<tool_call>write_file<arg_key>content</arg_key><arg_value>";
        let mut output = parser.push(opener).unwrap();
        assert_eq!(output.calls[0].name.as_deref(), Some("write_file"));
        let piece = "中文 \"quote\" \\ path\n &amp; ".repeat(32);
        let encoded = serde_json::to_string(&piece).unwrap();
        for _ in 0..24 {
            let update = parser.push(&piece).unwrap();
            assert_eq!(update.calls.len(), 1);
            assert!(!update.calls[0].complete);
            assert!(update.calls[0].name.is_none());
            assert_eq!(update.calls[0].arguments, encoded[1..encoded.len() - 1]);
            output.append(update);
        }
        let close_value = parser.push(ARG_VALUE_END).unwrap();
        assert_eq!(close_value.calls[0].arguments, "\"");
        assert!(!close_value.calls[0].complete);
        output.append(close_value);
        assert!(parser.push("\n").unwrap().calls.is_empty());
        let close_call = parser.push(BLOCK_END).unwrap();
        assert_eq!(close_call.calls[0].arguments, "}");
        assert!(close_call.calls[0].name.is_none());
        assert!(close_call.calls[0].complete);
        output.append(close_call);
        output.append(parser.finish().unwrap());
        let input = format!("{opener}{}{ARG_VALUE_END}\n{BLOCK_END}", piece.repeat(24));
        assert_append_only(output, &batch_arguments(&tools, &input));
    }

    #[test]
    fn native_closed_nonstring_then_path_and_content_stream_in_source_order() {
        let tools = file_tools();
        let mut parser = Glm47ToolStreamParser::new(&tools);
        let start = "<tool_call>write_file<arg_key>count</arg_key><arg_value>2";
        assert!(parser.push(start).unwrap().calls.is_empty());
        let mut output = parser.push(ARG_VALUE_END).unwrap();
        assert_eq!(output.calls[0].arguments, "{\"count\":2");
        let path = "<arg_key>path</arg_key><arg_value>/tmp/文件.txt";
        let update = parser.push(path).unwrap();
        assert_eq!(update.calls[0].arguments, ",\"path\":\"/tmp/文件.txt");
        output.append(update);
        let content = "</arg_value><arg_key>content</arg_key><arg_value>\n";
        output.append(parser.push(content).unwrap());
        for piece in ["one ", "two\n", " three\n"] {
            let update = parser.push(piece).unwrap();
            assert_eq!(update.calls.len(), 1);
            assert!(!update.calls[0].complete);
            output.append(update);
        }
        let end = "</arg_value></tool_call>";
        output.append(parser.push(end).unwrap());
        output.append(parser.finish().unwrap());
        let input = format!("{start}{ARG_VALUE_END}{path}{content}one two\n three\n{end}");
        assert_append_only(output, &batch_arguments(&tools, &input));
    }

    #[test]
    fn native_literal_strings_are_append_safe_at_every_utf8_boundary() {
        let tools = file_tools();
        let input = concat!(
            "<tool_call>write_file<arg_key>count</arg_key><arg_value>2</arg_value>",
            "<arg_key>path</arg_key> \n<arg_value> 文件\\\".txt </arg_value>",
            "<arg_key>content</arg_key><arg_value>\n 中文\"\\\n &amp; ",
            "</tool_call> literal <x> </arg_valuX>\r\n\t ",
            "</arg_value></tool_call>"
        );
        let expected = batch_arguments(&tools, input);
        for split in input.char_indices().map(|(at, _)| at).chain([input.len()]) {
            assert_append_only(
                legacy(&tools, &[&input[..split], &input[split..]]),
                &expected,
            );
        }
        assert_append_only(legacy(&tools, &char_chunks(input)), &expected);
        let value: serde_json::Value = serde_json::from_str(&expected).unwrap();
        assert_eq!(
            value["content"],
            "\n 中文\"\\\n &amp; </tool_call> literal <x> </arg_valuX>\r\n\t "
        );
    }

    #[test]
    fn native_ambiguous_outer_close_is_withheld_until_value_close() {
        let tools = file_tools();
        let opener = "<tool_call>write_file<arg_key>content</arg_key><arg_value>prefix";
        let mut parser = Glm47ToolStreamParser::new(&tools);
        let mut output = parser.push(opener).unwrap();
        for chunk in char_chunks("</tool_call> literal text") {
            assert!(
                parser.push(chunk).unwrap().calls.is_empty(),
                "chunk {chunk:?}"
            );
        }
        let resolved = parser.push(ARG_VALUE_END).unwrap();
        assert_eq!(resolved.calls[0].arguments, "</tool_call> literal text\"");
        output.append(resolved);
        output.append(parser.push(BLOCK_END).unwrap());
        output.append(parser.finish().unwrap());
        let input = format!("{opener}</tool_call> literal text{ARG_VALUE_END}{BLOCK_END}");
        assert_append_only(output, &batch_arguments(&tools, &input));

        for suffix in ["", "</arg_", "</tool_"] {
            let input = format!("{opener}{BLOCK_END}{suffix}");
            let output = legacy(&tools, &char_chunks(&input));
            assert_eq!(output.normal_text, suffix);
            assert_append_only(output, "{\"content\":\"prefix\"}");
        }
    }

    #[test]
    fn native_closed_values_reuse_batch_coercion_before_streaming_next_string() {
        for (schema, raw) in [
            (serde_json::json!({"type": "integer"}), "9".repeat(400)),
            (serde_json::json!({"type": "boolean"}), "yes".into()),
            (serde_json::json!({"type": "array"}), " one, two ".into()),
            (
                serde_json::json!({"type": "object"}),
                "{\"nested\":[1,true]}".into(),
            ),
            (
                serde_json::json!({"type": ["integer", "null"]}),
                "null".into(),
            ),
            (serde_json::json!({}), "[1,2]".into()),
        ] {
            let mut tools = file_tools();
            tools[0].parameters["properties"]["count"] = schema;
            let start = format!(
                "<tool_call>write_file<arg_key>path</arg_key><arg_value>first</arg_value><arg_key>count</arg_key><arg_value>{raw}"
            );
            let next = "</arg_value><arg_key>content</arg_key><arg_value>still streaming";
            let mut parser = Glm47ToolStreamParser::new(&tools);
            let mut output = parser.push(&start).unwrap();
            let update = parser.push(next).unwrap();
            assert!(
                update.calls[0]
                    .arguments
                    .ends_with(",\"content\":\"still streaming")
            );
            assert!(!update.calls[0].complete);
            output.append(update);
            output.append(parser.push("</arg_value></tool_call>").unwrap());
            output.append(parser.finish().unwrap());
            let input = format!("{start}{next}</arg_value></tool_call>");
            assert_append_only(output, &batch_arguments(&tools, &input));
        }
    }

    #[test]
    fn native_ambiguous_or_missing_schema_does_not_stream_open_values() {
        for schema in [
            serde_json::json!({"type": "string", "nullable": true}),
            serde_json::json!({"type": ["string", "null"]}),
            serde_json::json!({"anyOf": [{"type": "string"}, {"type": "null"}]}),
            serde_json::json!({"type": "string", "oneOf": [{"type": "string"}]}),
            serde_json::json!({"type": "string", "allOf": [{"type": "string"}]}),
            serde_json::json!({"type": "string", "$ref": "#/$defs/text"}),
            serde_json::json!({}),
            serde_json::Value::Null,
        ] {
            for raw in ["null", "\"quoted\"", "{\"x\":1}", "\n literal &amp; \n"] {
                let mut tools = file_tools();
                tools[0].parameters["properties"]["content"] = schema.clone();
                tools[0].parameters["$defs"] = serde_json::json!({"text": {"type": "string"}});
                let open =
                    format!("<tool_call>write_file<arg_key>content</arg_key><arg_value>{raw}");
                let mut parser = Glm47ToolStreamParser::new(&tools);
                for chunk in char_chunks(&open) {
                    assert!(
                        parser.push(chunk).unwrap().calls.is_empty(),
                        "schema {schema}, raw {raw:?}"
                    );
                }
                let mut output = parser.push("</arg_value></tool_call>").unwrap();
                output.append(parser.finish().unwrap());
                assert_append_only(
                    output,
                    &batch_arguments(&tools, &format!("{open}</arg_value></tool_call>")),
                );
            }
        }
        let input = "<tool_call>write_file<arg_key>content</arg_key><arg_value>[1,2]</arg_value></tool_call>";
        for tools in [
            Vec::new(),
            vec![Tool {
                parameters: serde_json::Value::Null,
                ..file_tools().remove(0)
            }],
        ] {
            let open = input.find(ARG_VALUE_END).unwrap();
            let mut parser = Glm47ToolStreamParser::new(&tools);
            assert!(parser.push(&input[..open]).unwrap().calls.is_empty());
            let mut output = parser.push(&input[open..]).unwrap();
            output.append(parser.finish().unwrap());
            assert_append_only(output, &batch_arguments(&tools, input));
        }
    }

    #[test]
    fn native_multiple_calls_keep_indices_names_and_empty_arguments() {
        let mut tools = tools();
        tools.extend(file_tools());
        let input = concat!(
            "<tool_call>get_time</tool_call>",
            "<tool_call>write_file<arg_key>path</arg_key><arg_value></arg_value>",
            "<arg_key>content</arg_key><arg_value></arg_value></tool_call>",
            "<tool_call>get_weather<arg_key>city</arg_key><arg_value>北京</arg_value></tool_call>"
        );
        let output = legacy(&tools, &char_chunks(input));
        for index in 0..3 {
            let deltas: Vec<_> = output
                .calls
                .iter()
                .filter(|d| d.tool_index == index)
                .collect();
            assert_eq!(deltas.iter().filter(|d| d.name.is_some()).count(), 1);
            assert_eq!(deltas.iter().filter(|d| d.complete).count(), 1);
        }
        let calls = output.coalesce_calls().calls;
        assert_eq!(calls.len(), 3);
        assert_eq!(calls[0].arguments, "{}");
        assert_eq!(calls[1].arguments, "{\"path\":\"\",\"content\":\"\"}");
        assert_eq!(calls[2].arguments, "{\"city\":\"北京\"}");
    }

    #[test]
    fn native_unclosed_value_never_completes_at_eof() {
        let tools = file_tools();
        for value in [
            "",
            "partial",
            "中文\n",
            "partial</arg_value",
            "partial</tool_call",
        ] {
            let input =
                format!("<tool_call>write_file<arg_key>content</arg_key><arg_value>{value}");
            let output = legacy(&tools, &char_chunks(&input));
            assert!(!output.calls.is_empty());
            assert!(output.calls.iter().all(|d| !d.complete));
            assert!(output.coalesce_calls().calls.is_empty());
        }
    }

    #[test]
    fn native_truncated_next_argument_header_cannot_complete_previous_arguments() {
        let tools = file_tools();
        let first = "<tool_call>write_file<arg_key>count</arg_key><arg_value>2</arg_value>";
        let next = "<arg_key>content</arg_key>\n<arg_value>unclosed";
        for end in 1..=next.len() {
            let input = format!("{first}{}", &next[..end]);
            for chunks in [vec![input.as_str()], char_chunks(&input)] {
                let output = legacy(&tools, &chunks);
                assert!(output.calls.iter().all(|d| !d.complete), "EOF at {input:?}");
                assert!(output.coalesce_calls().calls.is_empty());
            }
        }
    }

    #[test]
    fn native_outer_closer_cannot_hide_truncated_next_argument() {
        let tools = file_tools();
        let first = "<tool_call>write_file<arg_key>path</arg_key><arg_value>a</arg_value>";
        for tail in [
            "<arg_key>content</arg_key></tool_call>",
            "<arg_key>content</tool_call>",
            "<arg_key></tool_call>",
            "<arg_ke</tool_call>",
            "<arg_key>content</arg_key><arg_val</tool_call>",
        ] {
            let input = format!("{first}{tail}");
            for chunks in [vec![input.as_str()], vec![first, tail], char_chunks(&input)] {
                let output = legacy(&tools, &chunks);
                assert!(
                    output.calls.iter().all(|delta| !delta.complete),
                    "truncated argument accepted as complete: {input:?}"
                );
                assert!(output.coalesce_calls().calls.is_empty());
            }
        }
    }

    #[test]
    fn native_duplicate_key_invalidates_published_prefix_without_poisoning_next_call() {
        let mut tools = tools();
        tools.extend(file_tools());
        let first = "<tool_call>write_file<arg_key>path</arg_key><arg_value>first</arg_value>";
        let duplicate = "<arg_key>path</arg_key><arg_value>changed</arg_value></tool_call>";
        let input = format!("{first}{duplicate}");
        let expected = batch_arguments(&tools, &input);
        assert_append_only(legacy(&tools, &[&input]), &expected);
        // A duplicate arriving entirely in the final chunk still reconciles.
        for chunks in [vec![first, duplicate], char_chunks(&input)] {
            let mut parser = Glm47ToolStreamParser::new(&tools);
            let mut output = ToolParseResult::default();
            for chunk in chunks {
                output.append(parser.push(chunk).unwrap());
            }
            assert!(output.calls.iter().all(|d| !d.complete));
            assert!(output.clone().coalesce_calls().calls.is_empty());
            output.append(parser.push("<tool_call>get_time</tool_call>").unwrap());
            output.append(parser.finish().unwrap());
            let calls = output.coalesce_calls().calls;
            assert_eq!(calls.len(), 1);
            assert_eq!(calls[0].tool_index, 1);
            assert_eq!(calls[0].name.as_deref(), Some("get_time"));
            assert_eq!(calls[0].arguments, "{}");
        }
    }

    #[test]
    fn native_reset_clears_partial_arguments_and_restarts_index_zero() {
        let tools = file_tools();
        let mut scanner = glm47_scanner(&tools);
        let fresh =
            "<tool_call>write_file<arg_key>path</arg_key><arg_value>new</arg_value></tool_call>";
        for abandoned in [
            "<tool_call>write_file<arg_key>content",
            "<tool_call>write_file<arg_key>content</arg_key><arg_value>old",
            "<tool_call>write_file<arg_key>content</arg_key><arg_value>old</tool_call> withheld",
            "<tool_call>write_file<arg_key>count</arg_key><arg_value>123",
            "<tool_call>write_file<arg_key>path</arg_key><arg_value>old</arg_value><arg_key>path</arg_key>",
        ] {
            scanner.push(abandoned).unwrap();
            scanner.reset();
            let mut output = ToolParseResult::default();
            for chunk in char_chunks(fresh) {
                output.append(scanner.push(chunk).unwrap());
            }
            output.append(scanner.finish().unwrap());
            assert_append_only(output, &batch_arguments(&tools, fresh));
            scanner.reset();
        }
    }

    #[test]
    fn native_literal_argument_keys_do_not_reorder_later_parameters() {
        let tools = file_tools();
        let first = "<tool_call>write_file<arg_key>path</arg_key><arg_value><arg_key>content</arg_key></arg_value><arg_key>count</arg_key><arg_value>2</arg_value>";
        let last = "<arg_key>content</arg_key><arg_value>C</arg_value></tool_call>";
        let input = format!("{first}{last}");
        let expected = r#"{"path":"<arg_key>content</arg_key>","count":2,"content":"C"}"#;
        assert_eq!(batch_arguments(&tools, &input), expected);
        assert_append_only(legacy(&tools, &[first, last]), expected);
        assert_append_only(legacy(&tools, &char_chunks(&input)), expected);
        for (split, _) in input.char_indices() {
            assert_append_only(
                legacy(&tools, &[&input[..split], &input[split..]]),
                expected,
            );
        }
    }

    #[test]
    fn native_constrained_string_typing_is_split_invariant() {
        for (property, expected, streams_open_value) in [
            (
                serde_json::json!({"type": "string", "const": "x"}),
                r#"{"content":"x"}"#,
                false,
            ),
            (
                serde_json::json!({"type": "string", "enum": ["x"]}),
                r#"{"content":"\"x\""}"#,
                true,
            ),
            (
                serde_json::json!({"type": "string", "enum": ["x", 1]}),
                r#"{"content":"\"x\""}"#,
                true,
            ),
        ] {
            let tools = vec![Tool {
                name: "write_file".into(),
                description: None,
                parameters: serde_json::json!({
                    "type": "object", "properties": {"content": property}
                }),
                strict: None,
            }];
            let input = concat!(
                "<tool_call>write_file<arg_key>content</arg_key><arg_value>",
                "\"x\"</arg_value></tool_call>"
            );
            let mut parser = Glm47ToolStreamParser::new(&tools);
            let prefix = input.strip_suffix("</arg_value></tool_call>").unwrap();
            let open = parser.push(prefix).unwrap();
            assert_eq!(
                !open.calls.is_empty(),
                streams_open_value,
                "open value, schema {property}"
            );
            for split in 0..=input.len() {
                let output = legacy(&tools, &[&input[..split], &input[split..]]).coalesce_calls();
                assert_eq!(output.calls.len(), 1, "split {split}, schema {property}");
                assert!(output.calls[0].complete, "split {split}, schema {property}");
                assert_eq!(
                    output.calls[0].arguments, expected,
                    "split {split}, schema {property}"
                );
            }
        }
    }

    #[test]
    fn native_abandon_discards_unpublished_arguments_at_reused_index() {
        let tools = file_tools();
        for name in ["write_file", "unknown"] {
            let mut emitter = Glm47Emitter {
                config: Glm47ParserConfig::default(),
                tools: tools.iter().map(ToolDefinition::from).collect(),
                partial: None,
                pending_header: None,
                searched_bytes: 0,
                serialized_bytes: 0,
            };
            let abandoned = format!(
                "{name}<arg_key>count</arg_key><arg_value>{}",
                "2".repeat(8192)
            );
            assert!(
                emitter
                    .parse_partial_invoke(&abandoned, 0)
                    .unwrap()
                    .is_none()
            );
            emitter.abandon_invoke();

            let next = "write_file<arg_key>content</arg_key><arg_value>x";
            let update = emitter.parse_partial_invoke(next, 0).unwrap().unwrap();
            assert_eq!(update.tool_index, 0);
            assert_eq!(update.name.as_deref(), Some("write_file"));
            assert_eq!(update.arguments, r#"{"content":"x"#);
            assert!(!update.complete);
            let complete = emitter
                .parse_invoke(&format!("{next}</arg_value></tool_call>"), 0)
                .unwrap()
                .unwrap();
            assert_eq!(complete.arguments, "\"}");
            assert!(complete.complete);
        }
    }

    #[test]
    fn native_function_header_search_is_linear() {
        let tools = file_tools();
        for name in ["write_file", "unknown"] {
            let mut emitter = Glm47Emitter {
                config: Glm47ParserConfig::default(),
                tools: tools.iter().map(ToolDefinition::from).collect(),
                partial: None,
                pending_header: None,
                searched_bytes: 0,
                serialized_bytes: 0,
            };
            let invoke = format!(
                "{name}{}<arg_key>content</arg_key><arg_value>x</arg_value></tool_call>",
                " ".repeat(8192)
            );
            let mut output = ToolParseResult::default();
            for end in 1..=invoke.len() {
                if let Some(delta) = emitter.parse_partial_invoke(&invoke[..end], 0).unwrap() {
                    output.calls.push(delta);
                }
            }
            output
                .calls
                .push(emitter.parse_invoke(&invoke, 0).unwrap().unwrap());
            let complete = output.coalesce_calls();
            assert_eq!(complete.calls.len(), 1);
            assert_eq!(complete.calls[0].name.as_deref(), Some(name));
            assert_eq!(complete.calls[0].arguments, r#"{"content":"x"}"#);
            assert!(
                emitter.searched_bytes <= 32 * invoke.len(),
                "searched {} bytes for {} bytes of {name} input",
                emitter.searched_bytes,
                invoke.len()
            );
        }
    }

    #[test]
    fn native_argument_header_scanning_is_linear() {
        let tools = file_tools();
        for (key, whitespace) in [
            ("content".to_string(), " ".repeat(65_536)),
            ("k".repeat(8192), "\t\n".into()),
        ] {
            let mut emitter = Glm47Emitter {
                config: Glm47ParserConfig::default(),
                tools: tools.iter().map(ToolDefinition::from).collect(),
                partial: None,
                pending_header: None,
                searched_bytes: 0,
                serialized_bytes: 0,
            };
            let invoke = format!(
                "write_file<arg_key>{key}</arg_key>{whitespace}<arg_value>x</arg_value></tool_call>"
            );
            let mut output = ToolParseResult::default();
            for end in 1..=invoke.len() {
                if let Some(delta) = emitter.parse_partial_invoke(&invoke[..end], 0).unwrap() {
                    output.calls.push(delta);
                }
            }
            output
                .calls
                .push(emitter.parse_invoke(&invoke, 0).unwrap().unwrap());
            assert_append_only(output, &batch_arguments(&tools, &invoke));
            assert!(
                emitter.searched_bytes <= 32 * invoke.len(),
                "searched {} bytes for {}",
                emitter.searched_bytes,
                invoke.len()
            );
        }
    }

    #[test]
    fn native_long_value_search_and_fragment_serialization_are_linear() {
        let tools = file_tools();
        let mut emitter = Glm47Emitter {
            config: Glm47ParserConfig::default(),
            tools: tools.iter().map(ToolDefinition::from).collect(),
            partial: None,
            pending_header: None,
            searched_bytes: 0,
            serialized_bytes: 0,
        };
        for length in [4096, 8192] {
            for marker in ["", BLOCK_END] {
                emitter.reset();
                emitter.searched_bytes = 0;
                emitter.serialized_bytes = 0;
                let value = format!("prefix{marker}{}", "x".repeat(length));
                let invoke = format!(
                    "write_file<arg_key>count</arg_key><arg_value>2</arg_value><arg_key>path</arg_key><arg_value>a.txt</arg_value><arg_key>content</arg_key><arg_value>{value}</arg_value></tool_call>"
                );
                let mut output = ToolParseResult::default();
                for end in 1..=invoke.len() {
                    if let Some(delta) = emitter.parse_partial_invoke(&invoke[..end], 0).unwrap() {
                        output.calls.push(delta);
                    }
                }
                output
                    .calls
                    .push(emitter.parse_invoke(&invoke, 0).unwrap().unwrap());
                assert_append_only(output, &batch_arguments(&tools, &invoke));
                assert!(
                    emitter.searched_bytes <= 32 * invoke.len(),
                    "searched {} bytes for {}",
                    emitter.searched_bytes,
                    invoke.len()
                );
                assert_eq!(emitter.serialized_bytes, "a.txt".len() + value.len());
                assert!(emitter.partial.is_none());
            }
        }
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
    fn literal_entities_remain_argument_data() {
        let tools = vec![Tool {
            name: "inspect".into(),
            description: None,
            parameters: serde_json::json!({
                "type": "object", "properties": {
                    "text": {"type": "string"}, "payload": {"type": "object"}
                }
            }),
            strict: None,
        }];
        let input = concat!(
            "<tool_call>inspect<arg_key>text</arg_key><arg_value>",
            "&lt;tag&gt; &amp; &quot;x&quot; &apos;y&apos; &#65; &amp;lt;",
            "</arg_value><arg_key>payload</arg_key><arg_value>",
            r#"{"text":"&quot; &amp;"}"#,
            "</arg_value></tool_call>"
        );
        let result = legacy(&tools, &[input]).coalesce_calls();
        assert!(result.normal_text.is_empty());
        assert_eq!(result.calls.len(), 1);
        let arguments: serde_json::Value =
            serde_json::from_str(&result.calls[0].arguments).unwrap();
        assert_eq!(
            arguments,
            serde_json::json!({
                "text": "&lt;tag&gt; &amp; &quot;x&quot; &apos;y&apos; &#65; &amp;lt;",
                "payload": {"text": "&quot; &amp;"}
            })
        );
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
