// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Streaming XML-ish tool-call parser for MiniMax-M3.
//!
//! MiniMax-M3 prefixes every tag with the namespace token `]<]minimax[>[` and
//! names parameters by their TAG (not a `name=` attribute):
//!   `]<]minimax[>[<tool_call>
//!    ]<]minimax[>[<invoke name="NAME">
//!    ]<]minimax[>[<KEY>value]<]minimax[>[</KEY>
//!    ]<]minimax[>[</invoke>
//!    ]<]minimax[>[</tool_call>`
//! plus a bare `]<]minimax[>[<invoke ...>...</invoke>` back-off form when the
//! outer wrapper is absent.
//!
use crate::tool_calling::scan::{
    BareRecoveryLatch, GuidedInvokePrefix, GuidedInvokePrefixContext, InvokeBoundary,
    InvokeBoundaryFactory, InvokeEmitter, InvokeLatch, ReasoningSpec, WrappedBlockScanner,
    WrappedBlockSpec, marker_prefix_suffix_len,
};
use crate::tool_calling::v1core::{MiniMaxM3ParserConfig, ToolDefinition};

use super::{GuidedRouted, ScannerUnified, UnifiedParser};
use crate::tool_calling::traits::{Tool, ToolCallDelta};
use crate::tool_calling::v1core::xml::minimax_m3_parser::{
    M3ParameterKind, parameter_kind, parameter_path_kind, parse_invoke_name,
    parse_minimax_m3_invoke,
};
use std::{collections::HashSet, sync::Arc};

/// The namespace token emitted before every M3 tag.
const NS: &str = "]<]minimax[>[";
const BLOCK_START: &str = "]<]minimax[>[<tool_call>";
const BLOCK_END: &str = "]<]minimax[>[</tool_call>";
/// Bare `<invoke` (no trailing `>`): matches both `<invoke name="...">` and the
/// malformed nameless `<invoke>` (whose parse yields no call and is dropped).
const FUNCTION_START: &str = "]<]minimax[>[<invoke";
const FUNCTION_END: &str = "]<]minimax[>[</invoke>";

fn spec(tools: Arc<[ToolDefinition]>) -> WrappedBlockSpec {
    WrappedBlockSpec {
        family: "minimax_m3",
        block_starts: vec![BLOCK_START.to_string()],
        block_ends: vec![BLOCK_END.to_string()],
        invoke_start: FUNCTION_START.to_string(),
        invoke_end: FUNCTION_END.to_string(),
        orphan_markers: vec![BLOCK_END.to_string()],
        // BLOCK_END is held back too: after a bare-invoke recovery latches
        // suppression, a split closing marker must be retained whole so the
        // orphan-close path (which clears the latch) can match it.
        holdback_markers: vec![
            BLOCK_START.to_string(),
            FUNCTION_START.to_string(),
            BLOCK_END.to_string(),
        ],
        bare_recovery_latch: BareRecoveryLatch::Set,
        invoke_latch: InvokeLatch::IfEmitted,
        drop_invoke_crossing_block_end: false,
        // Every wrapped family's markers are special tokens today.
        preserve_special_tokens: true,
        invoke_boundary_factory: Some(InvokeBoundaryFactory::Configured(Arc::new(move || {
            Box::new(M3InvokeBoundary {
                tools: tools.clone(),
                ..Default::default()
            })
        }))),
    }
}

/// The shared scanner owns channels and buffering. This cursor only locates an
/// invoke's closing tag: reserved markers inside a parameter belong to its value.
#[derive(Default)]
struct M3InvokeBoundary {
    tools: Arc<[ToolDefinition]>,
    name: Option<String>,
    cursor: usize,
    tag_scanned: usize,
    path: Vec<String>,
    literal_close: Option<String>,
    parameter_only: bool,
    fallback_end: Option<usize>,
    completed_end: Option<usize>,
    guided_scanned: usize,
    guided_name_closed: bool,
    guided_native: bool,
    #[cfg(test)]
    examined: usize,
}

impl InvokeBoundary for M3InvokeBoundary {
    fn owns_guided_prefix(&self) -> bool {
        true
    }

    fn is_guided_invoke_marker(&self, marker: &str) -> bool {
        marker == FUNCTION_START
    }

    fn guided_prefix_append(
        &mut self,
        candidate: &str,
        _append: &str,
        context: GuidedInvokePrefixContext,
    ) -> Option<GuidedInvokePrefix> {
        // Guided generation sometimes retains this unclosed native header
        // before the JSON payload. A complete header instead owns a native body.
        const PREFIX: &str = "]<]minimax[>[<invoke name=\"";
        if self.guided_native {
            return Some(GuidedInvokePrefix::NativeBody);
        }
        if self.guided_scanned == 0 {
            if PREFIX.starts_with(candidate) {
                return Some(GuidedInvokePrefix::Pending);
            }
            if !candidate.starts_with(PREFIX) {
                return Some(GuidedInvokePrefix::NoMatch);
            }
            self.guided_scanned = PREFIX.len();
        }
        while self.guided_scanned < candidate.len() {
            let ch = candidate[self.guided_scanned..].chars().next()?;
            if matches!(ch, '{' | '[') {
                let len = if self.guided_name_closed {
                    self.guided_scanned
                } else {
                    PREFIX.len()
                };
                return Some(if context.outside_reasoning {
                    GuidedInvokePrefix::Match(len)
                } else {
                    GuidedInvokePrefix::Strip(len)
                });
            }
            if context.followed_by_competing_marker || ch == '<' {
                return Some(GuidedInvokePrefix::Strip(PREFIX.len()));
            }
            if ch == '>' && self.guided_name_closed {
                self.guided_native = true;
                return Some(GuidedInvokePrefix::NativeBody);
            }
            self.guided_name_closed |= ch == '"';
            self.guided_scanned += ch.len_utf8();
        }
        Some(GuidedInvokePrefix::Pending)
    }

    fn end_append(
        &mut self,
        candidate: &str,
        _append: &str,
        flush: bool,
        _index: usize,
    ) -> Option<usize> {
        if let Some(end) = self.completed_end {
            return Some(end);
        }
        loop {
            if let Some(close) = &self.literal_close {
                let tail = &candidate[self.cursor..];
                if let Some(at) = tail.find(close) {
                    self.cursor += at + close.len();
                    self.literal_close = None;
                    self.path.pop();
                    self.fallback_end = None;
                    if self.path.is_empty() && self.parameter_only {
                        self.completed_end = Some(self.cursor);
                        return self.completed_end;
                    }
                    continue;
                }
                if self.fallback_end.is_none() {
                    self.fallback_end = tail
                        .find(FUNCTION_END)
                        .map(|at| self.cursor + at + FUNCTION_END.len());
                }
                self.cursor = candidate.len()
                    - marker_prefix_suffix_len(tail, [close.as_str(), FUNCTION_END]);
                return if flush { self.fallback_end } else { None };
            }
            let Some(relative) = candidate[self.cursor..].find(NS) else {
                break;
            };
            #[cfg(test)]
            {
                self.examined += relative + NS.len();
            }
            let start = self.cursor + relative;
            let tag_start = start + NS.len();
            if tag_start == candidate.len() {
                self.cursor = start;
                return if flush { self.fallback_end } else { None };
            }
            if !candidate[tag_start..].starts_with('<') {
                self.cursor = tag_start;
                continue;
            }
            let scan_from = self.tag_scanned.max(tag_start + 1);
            let end = candidate[scan_from..].find('>').map(|at| scan_from + at);
            // A new namespace before the terminator belongs to a new tag. The
            // preceding unterminated tag-looking bytes are argument data.
            if let Some(next) = candidate[scan_from..].find(NS).map(|at| scan_from + at)
                && end.is_none_or(|end| next < end)
            {
                self.cursor = next;
                self.tag_scanned = 0;
                continue;
            }
            #[cfg(test)]
            {
                self.examined += end.map_or(candidate.len() - scan_from, |end| end - scan_from + 1);
            }
            let Some(end) = end else {
                self.cursor = start;
                self.tag_scanned =
                    candidate.len() - marker_prefix_suffix_len(&candidate[scan_from..], [NS]);
                return flush.then_some(self.fallback_end).flatten();
            };
            self.tag_scanned = 0;
            let tag = &candidate[tag_start + 1..end];
            self.cursor = end + 1;
            if start == 0
                && let Some(attributes) = tag.strip_prefix("invoke")
            {
                self.name = parse_invoke_name(attributes);
            } else if let Some(name) = tag.strip_prefix('/') {
                if self.path.is_empty() && name == "invoke" {
                    self.completed_end = Some(self.cursor);
                    return self.completed_end;
                }
                if let Some(at) = self.path.iter().rposition(|part| part == name) {
                    self.path.truncate(at);
                    if self.path.is_empty() {
                        self.fallback_end = None;
                        if self.parameter_only {
                            self.completed_end = Some(self.cursor);
                            return self.completed_end;
                        }
                    }
                } else if name == "invoke" && self.fallback_end.is_none() {
                    self.fallback_end = Some(self.cursor);
                }
            } else if !tag.is_empty() {
                self.path.push(tag.to_string());
                if self.name.as_deref().is_some_and(|name| {
                    matches!(
                        parameter_path_kind(name, &self.path, &self.tools),
                        M3ParameterKind::LiteralString { .. }
                    )
                }) {
                    self.literal_close = Some(format!("{NS}</{tag}>"));
                }
            }
        }
        #[cfg(test)]
        {
            self.examined += candidate.len() - self.cursor;
        }
        self.cursor = candidate.len() - marker_prefix_suffix_len(&candidate[self.cursor..], [NS]);
        if flush { self.fallback_end } else { None }
    }

    fn opens(&self, _text: &str, _at: usize) -> bool {
        true
    }
    fn holdback(&self, _text: &str) -> usize {
        0
    }
    fn resync(&mut self, _text: &str, _flush: bool, _index: usize) -> Option<usize> {
        None
    }
    fn reset(&mut self) {
        *self = Self {
            tools: self.tools.clone(),
            ..Self::default()
        };
    }
}

/// Append-only argument emission, with final typing owned by the shared value converter.
struct M3Emitter {
    config: MiniMaxM3ParserConfig,
    tools: Arc<[ToolDefinition]>,
    partial: Option<PartialArguments>,
    header_scanned: usize,
    header_rejected: bool,
}

struct PartialArguments {
    name: String,
    header_end: usize,
    emitted: String,
    cursor: usize,
    tag_scanned: usize,
    seen: HashSet<String>,
    active: Option<PartialParameter>,
    blocked: bool,
}

struct PartialParameter {
    close: String,
    cursor: usize,
    start: usize,
    boundary: Option<M3InvokeBoundary>,
    streamable: bool,
}

impl InvokeEmitter for M3Emitter {
    fn parse_partial_invoke(
        &mut self,
        invoke: &str,
        tool_index: usize,
    ) -> anyhow::Result<Option<ToolCallDelta>> {
        if self.header_rejected {
            return Ok(None);
        }
        if self.partial.is_none() {
            let scan_from = self.header_scanned.max(FUNCTION_START.len());
            let Some(end) = invoke[scan_from..].find('>').map(|at| at + scan_from) else {
                self.header_scanned = invoke.len();
                return Ok(None);
            };
            let Some(name) = parse_invoke_name(&invoke[FUNCTION_START.len()..end])
                .filter(|name| !name.is_empty())
            else {
                self.header_rejected = true;
                return Ok(None);
            };
            self.partial = Some(PartialArguments {
                name,
                header_end: end + 1,
                emitted: String::new(),
                cursor: end + 1,
                tag_scanned: 0,
                seen: HashSet::new(),
                active: None,
                blocked: false,
            });
        }
        let partial = self.partial.as_mut().expect("initialized above");
        if partial.blocked {
            return Ok(None);
        }
        let mut arguments = String::new();
        loop {
            if let Some(active) = &mut partial.active {
                if !active.streamable {
                    // Reuse the schema-aware cursor so nested strings own
                    // their literal tags while this typed parameter is open.
                    let end = active
                        .boundary
                        .as_mut()
                        .expect("typed boundary")
                        .end_append(invoke, "", false, tool_index);
                    let Some(end) = end else {
                        break;
                    };
                    // Reuse the sole value converter for this completed parameter.
                    // The original header avoids re-quoting a model-supplied name.
                    let one_parameter = format!(
                        "{}{}{FUNCTION_END}",
                        &invoke[..partial.header_end],
                        &invoke[active.start..end]
                    );
                    let Some((_, typed)) =
                        parse_minimax_m3_invoke(&one_parameter, &self.config, &self.tools)?
                    else {
                        partial.blocked = true;
                        break;
                    };
                    arguments.push(if partial.emitted.is_empty() && arguments.is_empty() {
                        '{'
                    } else {
                        ','
                    });
                    arguments.push_str(&typed[1..typed.len() - 1]);
                    partial.cursor = end;
                    partial.active = None;
                    continue;
                }
                let tail = &invoke[active.cursor..];
                let close = tail.find(&active.close).map(|at| active.cursor + at);
                let end = close.unwrap_or_else(|| {
                    invoke.len() - marker_prefix_suffix_len(tail, [active.close.as_str()])
                });
                let encoded = serde_json::to_string(&invoke[active.cursor..end])?;
                arguments.push_str(&encoded[1..encoded.len() - 1]);
                active.cursor = end;
                if close.is_some() {
                    arguments.push('"');
                    partial.cursor = end + active.close.len();
                    partial.active = None;
                    continue;
                }
                break;
            }
            let tail = &invoke[partial.cursor..];
            let Some(relative) = tail.find(NS) else {
                partial.cursor = invoke.len() - marker_prefix_suffix_len(tail, [NS]);
                break;
            };
            let tag_start = partial.cursor + relative + NS.len();
            partial.cursor += relative;
            if tag_start == invoke.len() {
                break;
            }
            if !invoke[tag_start..].starts_with('<') {
                partial.cursor = tag_start;
                continue;
            }
            let scan_from = partial.tag_scanned.max(tag_start + 1);
            let Some(end) = invoke[scan_from..].find('>').map(|at| scan_from + at) else {
                partial.tag_scanned = invoke.len();
                break;
            };
            partial.tag_scanned = 0;
            let parameter = &invoke[tag_start + 1..end];
            // Strings stream while open. Other types are emitted as soon as their
            // parameter closes, using the shared converter without speculation.
            let kind = parameter_kind(&partial.name, parameter, &self.tools);
            let streamable = matches!(kind, M3ParameterKind::LiteralString { streamable: true });
            if parameter.is_empty()
                || parameter.starts_with('/')
                || !partial.seen.insert(parameter.to_string())
                || matches!(kind, M3ParameterKind::Buffered)
            {
                partial.blocked = true;
                break;
            }
            if streamable {
                arguments.push(if partial.emitted.is_empty() && arguments.is_empty() {
                    '{'
                } else {
                    ','
                });
                arguments.push_str(&serde_json::to_string(parameter)?);
                arguments.push_str(":\"");
            }
            partial.active = Some(PartialParameter {
                close: format!("{NS}</{parameter}>"),
                cursor: end + 1,
                start: tag_start - NS.len(),
                boundary: (!streamable).then(|| M3InvokeBoundary {
                    tools: self.tools.clone(),
                    name: Some(partial.name.clone()),
                    cursor: end + 1,
                    path: vec![parameter.to_string()],
                    literal_close: matches!(kind, M3ParameterKind::LiteralString { .. })
                        .then(|| format!("{NS}</{parameter}>")),
                    parameter_only: true,
                    ..Default::default()
                }),
                streamable,
            });
        }
        if arguments.is_empty() {
            return Ok(None);
        }
        let first = partial.emitted.is_empty();
        partial.emitted.push_str(&arguments);
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
        self.header_scanned = 0;
        self.header_rejected = false;
        let Some((name, arguments)) = parse_minimax_m3_invoke(invoke, &self.config, &self.tools)?
        else {
            self.partial = None;
            return Ok(None);
        };
        let partial = self.partial.take();
        let prefix = partial
            .as_ref()
            .map_or("", |partial| partial.emitted.as_str());
        if !arguments.starts_with(prefix) {
            // A repeated parameter changes the final value to an array. Bytes
            // already sent cannot be retracted; do not complete corrupted JSON.
            return Ok(Some(ToolCallDelta {
                tool_index,
                name: None,
                arguments: String::new(),
                complete: false,
            }));
        }
        Ok(Some(ToolCallDelta {
            tool_index,
            name: prefix.is_empty().then_some(name),
            arguments: arguments[prefix.len()..].to_string(),
            complete: true,
        }))
    }

    fn reset(&mut self) {
        self.partial = None;
        self.header_scanned = 0;
        self.header_rejected = false;
    }
}

/// Build the shared M3 scanner with ordered reasoning and tool events.
pub(crate) fn minimax_m3_unified(tools: &[Tool]) -> Box<dyn UnifiedParser> {
    let tools: Arc<[ToolDefinition]> = tools.iter().map(ToolDefinition::from).collect();
    Box::new(
        GuidedRouted::new(ScannerUnified::new(
            WrappedBlockScanner::new(
                spec(tools.clone()),
                M3Emitter {
                    config: MiniMaxM3ParserConfig::default(),
                    tools,
                    partial: None,
                    header_scanned: 0,
                    header_rejected: false,
                },
            )
            .with_reasoning(ReasoningSpec {
                start: "<mm:think>",
                end: "</mm:think>",
                ..Default::default()
            }),
        ))
        .with_native_tool_fallback(BLOCK_START),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn boundary_work_is_linear_for_long_headers_and_values() {
        for size in [1_024, 8_192] {
            let name = "n".repeat(size);
            let value = "v".repeat(size);
            let input = format!(
                "{FUNCTION_START} name=\"{name}\">{NS}<{name}>{value}{NS}</{name}>{FUNCTION_END}"
            );
            let mut boundary = M3InvokeBoundary::default();
            for end in 1..=input.len() {
                let result = boundary.end_append(&input[..end], &input[end - 1..end], false, 0);
                assert_eq!(result, (end == input.len()).then_some(end));
            }
            assert!(
                boundary.examined < 20 * input.len(),
                "{} bytes examined for {} input bytes",
                boundary.examined,
                input.len()
            );
            // Guided routing can probe the completed candidate again before
            // deciding whether the following bytes are narration or payload.
            assert_eq!(boundary.end_append(&input, "", false, 0), Some(input.len()));
            boundary.reset();
            assert_eq!(
                boundary.end_append(FUNCTION_START, FUNCTION_START, false, 0),
                None
            );
        }
    }
}
