// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Hermes JSON envelopes use the wrapped scanner and the guided JSON lexer.

use crate::tool_calling::scan::{
    GuidedInvokePrefix, GuidedInvokePrefixContext, InvokeBoundary, InvokeBoundaryFactory,
    InvokeEmitter, JsonValueBoundary, ReasoningSpec, WrappedBlockScanner, WrappedBlockSpec,
    marker_prefix_suffix_len,
};
use crate::tool_calling::traits::{Tool, ToolCallDelta};
use crate::unified::{
    GuidedJsonCursor, GuidedRouted, ScannerUnified, UnifiedParser, parse_required_guided_calls,
};

const START: &str = "<tool_call>";
const END: &str = "</tool_call>";

#[derive(Default)]
struct PayloadStart {
    scanned: usize,
    first: Option<char>,
}

impl PayloadStart {
    fn first(&mut self, text: &str) -> Option<char> {
        if self.first.is_none() {
            for ch in text[self.scanned..].chars() {
                #[cfg(test)]
                crate::tool_calling::kimi_progress::count_work(2, ch.len_utf8());
                self.scanned += ch.len_utf8();
                if !ch.is_whitespace() {
                    self.first = Some(ch);
                    break;
                }
            }
        }
        self.first
    }
}

#[derive(Default)]
struct HermesBoundary {
    scanned: usize,
    json: JsonValueBoundary,
    value_end: Option<usize>,
    guided_prefix: PayloadStart,
}

fn boundary() -> Box<dyn InvokeBoundary> {
    Box::<HermesBoundary>::default()
}

impl InvokeBoundary for HermesBoundary {
    fn block_is_invoke(&self) -> bool {
        true
    }

    fn end_append(
        &mut self,
        candidate: &str,
        _append: &str,
        flush: bool,
        _tool_index: usize,
    ) -> Option<usize> {
        // The block scanner passes the body; prose lookahead passes the full
        // envelope. A leading opener must advance rather than become an empty
        // resynchronization span in ProseControlState.
        if self.scanned == 0 && candidate.starts_with(START) {
            self.scanned = START.len();
        }
        while self.scanned < candidate.len() {
            let tail = &candidate[self.scanned..];
            if !self.json.inside_string() {
                if tail.starts_with(END) {
                    return Some(self.scanned + END.len());
                }
                if tail.starts_with(START) {
                    return Some(self.scanned);
                }
                if !flush && [START, END].iter().any(|marker| marker.starts_with(tail)) {
                    return None;
                }
                if self.value_end.is_some() && !tail.starts_with(char::is_whitespace) {
                    return self.value_end;
                }
            }
            let ch = tail.chars().next().expect("nonempty candidate suffix");
            self.scanned += ch.len_utf8();
            if self.json.advance(ch) {
                self.value_end = Some(self.scanned);
            }
        }
        flush.then_some(self.value_end).flatten()
    }

    fn opens(&self, _text: &str, _at: usize) -> bool {
        // Native scanning enters through block_is_invoke. Under guided decoding
        // the JSON envelope is already the constrained payload, not a second
        // native body for the router to suppress.
        false
    }

    fn owns_prose_invoke(&self, text: &str, at: usize) -> bool {
        text[at..]
            .strip_prefix(START)
            .is_some_and(|body| body.trim_start().starts_with('{'))
    }

    fn owns_guided_prefix(&self) -> bool {
        true
    }

    fn guided_prefix_append(
        &mut self,
        candidate: &str,
        _append: &str,
        _context: GuidedInvokePrefixContext,
    ) -> Option<GuidedInvokePrefix> {
        let body = candidate.strip_prefix(START)?;
        Some(match self.guided_prefix.first(body) {
            Some('{' | '[') => GuidedInvokePrefix::Match(START.len()),
            None => GuidedInvokePrefix::Pending,
            Some(_) => GuidedInvokePrefix::Strip(START.len()),
        })
    }

    fn holdback(&self, text: &str) -> usize {
        marker_prefix_suffix_len(text, [START, END])
    }

    fn resync(&mut self, _text: &str, _flush: bool, _tool_index: usize) -> Option<usize> {
        None
    }

    fn reset(&mut self) {
        *self = Self::default();
    }
}

#[derive(Default)]
struct HermesEmitter {
    cursor: GuidedJsonCursor,
    prefix: PayloadStart,
}

impl InvokeEmitter for HermesEmitter {
    fn parse_partial_invoke(
        &mut self,
        invoke: &str,
        tool_index: usize,
    ) -> anyhow::Result<Option<ToolCallDelta>> {
        let invoke = invoke.strip_prefix(START).unwrap_or(invoke);
        // The model template defines one object per envelope. Bare response JSON
        // and arrays remain visible outside an envelope, never inferred as calls.
        if self.prefix.first(invoke) != Some('{') {
            return Ok(None);
        }
        let mut fragments = Vec::new();
        self.cursor.advance(invoke, &mut fragments);
        let mut result: Option<ToolCallDelta> = None;
        for mut fragment in fragments {
            fragment.tool_index = tool_index;
            if let Some(delta) = result.as_mut() {
                delta.arguments.push_str(&fragment.arguments);
            } else {
                result = Some(fragment);
            }
        }
        Ok(result)
    }

    fn parse_invoke(
        &mut self,
        invoke: &str,
        tool_index: usize,
    ) -> anyhow::Result<Option<ToolCallDelta>> {
        let invoke = invoke.strip_prefix(START).unwrap_or(invoke);
        let body = invoke.strip_suffix(END).unwrap_or(invoke).trim();
        let call = body
            .starts_with('{')
            .then(|| parse_required_guided_calls(body))
            .flatten()
            .and_then(|mut calls| calls.pop());
        let Some(call) = call else {
            tracing::warn!(
                why = "hermes_invalid_call",
                tool_index,
                skipped_bytes = body.len(),
                "Dropped invalid Hermes envelope; provisional fragments remain incomplete"
            );
            return Ok(None);
        };
        if !invoke.ends_with(END) {
            tracing::warn!(
                why = "hermes_missing_close_recovered",
                tool_index,
                recovered_bytes = body.len(),
                "Recovered a delimiter-complete Hermes call"
            );
        }
        if let Some(committed) = self.cursor.committed().first() {
            if committed.ambiguous
                || committed.name != call.name
                || committed.released > call.arguments.len()
            {
                tracing::warn!(
                    why = "hermes_invalid_provisional_call",
                    tool_index,
                    "Hermes envelope changed after provisional output; withholding completion"
                );
                return Ok(None);
            }
            return Ok(Some(ToolCallDelta {
                tool_index,
                name: None,
                arguments: call.arguments[committed.released..].to_string(),
                complete: true,
            }));
        }
        Ok(Some(ToolCallDelta {
            tool_index,
            name: Some(call.name),
            arguments: call.arguments,
            complete: true,
        }))
    }

    fn abandon_invoke(&mut self) {
        self.cursor.reset();
        self.prefix = PayloadStart::default();
    }

    fn reset(&mut self) {
        self.abandon_invoke();
    }
}

/// Build Hermes' single native/guided, reasoning/text/tool state machine.
pub(crate) fn hermes_unified(_tools: &[Tool]) -> Box<dyn UnifiedParser> {
    let scanner = WrappedBlockScanner::new(
        WrappedBlockSpec {
            family: "hermes",
            block_starts: vec![START.into()],
            block_ends: vec![END.into()],
            invoke_start: START.into(),
            invoke_end: END.into(),
            orphan_markers: vec![END.into()],
            holdback_markers: vec![START.into(), END.into()],
            invoke_boundary_factory: Some(InvokeBoundaryFactory::custom(boundary)),
            preserve_special_tokens: true,
            ..Default::default()
        },
        HermesEmitter::default(),
    )
    .with_reasoning(ReasoningSpec {
        start: "<think>",
        end: "</think>",
        ..Default::default()
    });
    Box::new(GuidedRouted::new(ScannerUnified::new(scanner)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::unified::{UnifiedParserInit, UnifiedParserOutput, UnifiedToolOutputMode};

    fn whitespace_work(size: usize, guided: bool) -> usize {
        let mut parser = hermes_unified(&[]);
        let init = UnifiedParserInit {
            tool_output_mode: if guided {
                UnifiedToolOutputMode::GuidedJson { named_tool: None }
            } else {
                UnifiedToolOutputMode::Native
            },
            ..Default::default()
        };
        for _ in 0..2 {
            parser.initialize_request(init.clone()).unwrap();
            parser
                .parse_into(START, &mut UnifiedParserOutput::default())
                .unwrap();
            for _ in 0..size {
                parser
                    .parse_into(" ", &mut UnifiedParserOutput::default())
                    .unwrap();
            }
            parser
                .parse_into(
                    r#"{"name":"f","arguments":{}}</tool_call>"#,
                    &mut UnifiedParserOutput::default(),
                )
                .unwrap();
            parser.finish().unwrap();
            parser.reset();
        }
        crate::tool_calling::kimi_progress::work().into_iter().sum()
    }

    #[test]
    fn whitespace_prefix_work_is_linear_in_native_and_guided_modes() {
        for guided in [false, true] {
            crate::tool_calling::kimi_progress::reset_work();
            let small = whitespace_work(4_096, guided);
            crate::tool_calling::kimi_progress::reset_work();
            let large = whitespace_work(8_192, guided);
            assert!(small > 4_096);
            assert!(
                large <= 2 * small + 256,
                "guided={guided}: {small} -> {large}"
            );
        }
    }

    #[test]
    fn debug_wrapper_preserves_hermes_request_capabilities_and_each_delta() {
        use crate::unified::{DebugUnifiedParser, UnifiedParserStartingState};
        for starting_state in [
            UnifiedParserStartingState::None,
            UnifiedParserStartingState::Reasoning,
            UnifiedParserStartingState::Response,
        ] {
            for mode in [
                UnifiedToolOutputMode::Native,
                UnifiedToolOutputMode::GuidedJson { named_tool: None },
                UnifiedToolOutputMode::GuidedJson {
                    named_tool: Some("f".into()),
                },
            ] {
                let payload = match &mode {
                    UnifiedToolOutputMode::Native => {
                        r#"<tool_call>{"name":"f","arguments":{"x":"雪"}}</tool_call>"#
                    }
                    UnifiedToolOutputMode::GuidedJson { named_tool: None } => {
                        r#"{"name":"f","arguments":{"x":"雪"}}"#
                    }
                    UnifiedToolOutputMode::GuidedJson { .. } => r#"{"x":"雪"}"#,
                };
                let input = if starting_state == UnifiedParserStartingState::Reasoning {
                    format!("check</think>{payload}")
                } else {
                    payload.to_string()
                };
                for at in (0..=input.len()).filter(|at| input.is_char_boundary(*at)) {
                    let mut plain = hermes_unified(&[]);
                    let mut debug: Box<dyn UnifiedParser> = Box::new(DebugUnifiedParser {
                        family: "hermes".into(),
                        inner: hermes_unified(&[]),
                    });
                    let init = UnifiedParserInit {
                        starting_state,
                        tool_output_mode: mode.clone(),
                        ..Default::default()
                    };
                    for _ in 0..2 {
                        plain.initialize_request(init.clone()).unwrap();
                        debug.initialize_request(init.clone()).unwrap();
                        assert_eq!(
                            plain.preserve_special_tokens(),
                            debug.preserve_special_tokens()
                        );
                        assert_eq!(plain.input_capabilities(), debug.input_capabilities());
                        for chunk in [&input[..at], &input[at..]] {
                            let mut left = UnifiedParserOutput::default();
                            let mut right = UnifiedParserOutput::default();
                            plain.parse_into(chunk, &mut left).unwrap();
                            debug.parse_into(chunk, &mut right).unwrap();
                            assert_eq!(left.events, right.events);
                        }
                        assert_eq!(
                            plain.finish().unwrap().events,
                            debug.finish().unwrap().events
                        );
                        assert_eq!(plain.reset(), debug.reset());
                    }
                }
            }
        }
    }
}
