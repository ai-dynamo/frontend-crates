// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Mistral v11+/v13/v15 tool calls on the shared scanner.
//!
//! A call is `[TOOL_CALLS]NAME[ARGS]{json}`, or the fallback `[TOOL_CALLS]NAME{json}`
//! whose `{` belongs to the arguments. v11 may insert `[CALL_ID]ID` after the name.
//! Parallel calls repeat `[TOOL_CALLS]`; nothing ever closes a call or the block.
//!
//! So the block never closes before EOF. Each call ends just past its JSON value,
//! and the next one starts at the following `[TOOL_CALLS]`. Text between the two,
//! or after the last call, belongs to the previous arguments slot and is dropped,
//! as upstream vLLM drops it (vllm-project/vllm#48975). A name that is not a tool
//! identifier is not a call either, which is how the unsupported pre-v11
//! `[TOOL_CALLS] [{...}]` array and prose after a bare opener are dropped.

use crate::tool_calling::scan::{
    InvokeBoundary, InvokeBoundaryFactory, InvokeEmitter, JsonStringState, WrappedBlockScanner,
    WrappedBlockSpec,
};
use crate::tool_calling::traits::ToolCallDelta;

pub(crate) const TOOL_CALLS: &str = "[TOOL_CALLS]";
pub(crate) const ARGS: &str = "[ARGS]";
const CALL_ID: &str = "[CALL_ID]";

/// Length of a tool-call id. mistral-common's validator requires 9 alphanumerics.
const TOOL_CALL_ID_LEN: usize = 9;

fn spec() -> WrappedBlockSpec {
    WrappedBlockSpec {
        family: "mistral",
        block_starts: vec![TOOL_CALLS.to_string()],
        // No closer: the boundary ends each call, and EOF ends the block.
        block_ends: Vec::new(),
        invoke_start: TOOL_CALLS.to_string(),
        invoke_end: String::new(),
        orphan_markers: vec![ARGS.to_string(), CALL_ID.to_string()],
        holdback_markers: [TOOL_CALLS, ARGS, CALL_ID]
            .into_iter()
            .map(str::to_string)
            .collect(),
        invoke_boundary_factory: Some(InvokeBoundaryFactory::NativeOnly(mistral_boundary)),
        preserve_special_tokens: true,
        ..Default::default()
    }
}

/// An empty name stays a call so a malformed generation surfaces; anything that
/// is not an identifier is not a call.
fn is_tool_name(name: &str) -> bool {
    name.chars()
        .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '.'))
}

/// Where a call scan stands. A first call starts at its name, because the
/// scanner consumed the block opener; a later call starts at the previous
/// call's leftover text.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    /// Text left after the previous call or a rejected name, up to the next
    /// `[TOOL_CALLS]`.
    Lead,
    Name,
    CallId,
    /// After `[ARGS]`, before the first non-whitespace byte of the body.
    Args,
    /// A JSON object or array body.
    Json,
    /// A body that is not a JSON value. It runs to the next call or to EOF.
    Raw,
}

enum Step {
    /// The call ends at this offset.
    Ended(usize),
    /// A marker may be split at the end of the text.
    Pending,
    /// The text ran out before the call ended.
    Exhausted,
}

/// One resumable scan of a call. The boundary drives it over the growing
/// buffer; the emitter replays it over the delimited call to type it.
#[derive(Default)]
struct CallScan {
    phase: Option<Phase>,
    at: usize,
    name_start: usize,
    name_end: Option<usize>,
    call_id_start: Option<usize>,
    call_id_end: Option<usize>,
    body_start: Option<usize>,
    closers: Vec<char>,
    string: JsonStringState,
    /// The first `[TOOL_CALLS]` inside a JSON string of the body.
    marker_in_string: Option<usize>,
    /// Names rejected so far, so the boundary can report each one once.
    rejected: usize,
}

impl CallScan {
    fn step(&mut self, text: &str, flush: bool, after_call: bool) -> Step {
        let initial = if after_call { Phase::Lead } else { Phase::Name };
        let mut phase = *self.phase.get_or_insert(initial);
        let pending = |rest: &str, markers: &[&str]| {
            !flush && markers.iter().any(|marker| marker.starts_with(rest))
        };
        let step = loop {
            let rest = &text[self.at..];
            let Some(ch) = rest.chars().next() else {
                break Step::Exhausted;
            };
            match phase {
                Phase::Lead => {
                    if rest.starts_with(TOOL_CALLS) {
                        self.at += TOOL_CALLS.len();
                        self.name_start = self.at;
                        phase = Phase::Name;
                        continue;
                    }
                    if pending(rest, &[TOOL_CALLS]) {
                        break Step::Pending;
                    }
                }
                Phase::Name => {
                    let separator = [TOOL_CALLS, ARGS, CALL_ID]
                        .into_iter()
                        .find(|marker| rest.starts_with(marker))
                        .or((ch == '{').then_some("{"));
                    if let Some(separator) = separator {
                        let name = text[self.name_start..self.at].trim();
                        // A repeated opener before any name is one opener.
                        if separator == TOOL_CALLS && name.is_empty() {
                            self.at += TOOL_CALLS.len();
                            self.name_start = self.at;
                            continue;
                        }
                        if !is_tool_name(name) {
                            self.rejected += 1;
                            phase = Phase::Lead;
                            continue;
                        }
                        self.name_end = Some(self.at);
                        match separator {
                            TOOL_CALLS => break Step::Ended(self.at),
                            ARGS => {
                                self.at += ARGS.len();
                                self.body_start = Some(self.at);
                                phase = Phase::Args;
                            }
                            CALL_ID => {
                                self.at += CALL_ID.len();
                                self.call_id_start = Some(self.at);
                                phase = Phase::CallId;
                            }
                            _ => {
                                self.body_start = Some(self.at);
                                phase = Phase::Json;
                            }
                        }
                        continue;
                    }
                    if pending(rest, &[TOOL_CALLS, ARGS, CALL_ID]) {
                        break Step::Pending;
                    }
                }
                Phase::CallId => {
                    if rest.starts_with(TOOL_CALLS) {
                        self.call_id_end = Some(self.at);
                        break Step::Ended(self.at);
                    }
                    if rest.starts_with(ARGS) {
                        self.call_id_end = Some(self.at);
                        self.at += ARGS.len();
                        self.body_start = Some(self.at);
                        phase = Phase::Args;
                        continue;
                    }
                    if ch == '{' {
                        self.call_id_end = Some(self.at);
                        self.body_start = Some(self.at);
                        phase = Phase::Json;
                        continue;
                    }
                    if pending(rest, &[TOOL_CALLS, ARGS]) {
                        break Step::Pending;
                    }
                }
                Phase::Args => {
                    if rest.starts_with(TOOL_CALLS) {
                        break Step::Ended(self.at);
                    }
                    // A repeated separator does not open a JSON array.
                    if rest.starts_with(ARGS) {
                        self.at += ARGS.len();
                        self.body_start = Some(self.at);
                        continue;
                    }
                    if rest.starts_with(CALL_ID) {
                        self.at += CALL_ID.len();
                        self.call_id_start = Some(self.at);
                        self.body_start = None;
                        phase = Phase::CallId;
                        continue;
                    }
                    if pending(rest, &[TOOL_CALLS, ARGS, CALL_ID]) {
                        break Step::Pending;
                    }
                    if matches!(ch, '{' | '[') {
                        phase = Phase::Json;
                        continue;
                    }
                    if !ch.is_whitespace() {
                        phase = Phase::Raw;
                        continue;
                    }
                }
                Phase::Raw => {
                    if rest.starts_with(TOOL_CALLS) {
                        break Step::Ended(self.at);
                    }
                    if pending(rest, &[TOOL_CALLS]) {
                        break Step::Pending;
                    }
                }
                Phase::Json => {
                    // A marker inside a JSON string is argument data (`I7`), though
                    // it is remembered for the EOF fallback. Outside one, the next
                    // call starts and this malformed body ends.
                    if rest.starts_with(TOOL_CALLS) {
                        if !self.string.is_inside() {
                            break Step::Ended(self.at);
                        }
                        self.marker_in_string.get_or_insert(self.at);
                    } else if pending(rest, &[TOOL_CALLS]) {
                        break Step::Pending;
                    }
                    self.at += ch.len_utf8();
                    if self.string.advance(ch) {
                        continue;
                    }
                    match ch {
                        '{' => self.closers.push('}'),
                        '[' => self.closers.push(']'),
                        '}' | ']' => {
                            if self.closers.pop() != Some(ch) {
                                phase = Phase::Raw;
                            } else if self.closers.is_empty() {
                                break Step::Ended(self.at);
                            }
                        }
                        _ => {}
                    }
                    continue;
                }
            }
            self.at += ch.len_utf8();
        };
        self.phase = Some(phase);
        step
    }

    /// Where a call that is still open at EOF ends, if it is a call at all.
    fn eof_end(&mut self, text: &str) -> Option<usize> {
        match self.phase? {
            // Only the previous call's leftover text.
            Phase::Lead => None,
            // A missing quote leaves the body inside a string at EOF, because
            // every later call adds an even number of quotes. Such a body ends
            // at the first marker it swallowed, so later calls survive. With the
            // body outside a string, any marker sat in a closed string and is
            // data, so the call was truncated and is dropped (policy P2).
            Phase::Json => self.marker_in_string.filter(|_| self.string.is_inside()),
            Phase::Name => {
                let name = text[self.name_start..].trim();
                if !is_tool_name(name) {
                    self.rejected += 1;
                    return None;
                }
                (!name.is_empty()).then_some(text.len())
            }
            Phase::CallId | Phase::Args | Phase::Raw => Some(text.len()),
        }
    }

    /// The trimmed name, call id, and body of a scanned call that ends at
    /// `text.len()`.
    fn parts<'a>(&self, text: &'a str) -> (&'a str, &'a str, &'a str) {
        if self.phase == Some(Phase::Lead) {
            return ("", "", "");
        }
        let name = &text[self.name_start..self.name_end.unwrap_or(text.len())];
        let call_id = self.call_id_start.map_or("", |start| {
            &text[start..self.call_id_end.unwrap_or(text.len())]
        });
        let body = self.body_start.map_or("", |start| &text[start..]);
        (name.trim(), call_id.trim(), body.trim())
    }
}

#[derive(Default)]
struct MistralBoundary {
    scan: CallScan,
    reported: usize,
}

impl MistralBoundary {
    fn report_rejected_names(&mut self) {
        if self.scan.rejected > self.reported {
            tracing::warn!(
                why = "mistral_unrecognized_call",
                rejected = self.scan.rejected - self.reported,
                "stream dropped a call whose name is not a tool identifier"
            );
            self.reported = self.scan.rejected;
        }
    }
}

impl InvokeBoundary for MistralBoundary {
    fn block_is_invoke(&self) -> bool {
        true
    }

    /// Leftover text after the last call is the previous arguments slot, not a
    /// dropped call. A truncated JSON body still reaches the scanner's warning.
    fn block_ends_at_eof(&self) -> bool {
        self.scan.phase != Some(Phase::Json)
    }

    fn end_append(
        &mut self,
        candidate: &str,
        _append: &str,
        flush: bool,
        tool_index: usize,
    ) -> Option<usize> {
        // A nonzero index means a call already ended in this block: the block
        // cannot close before EOF, so it is the stream's only one.
        let end = match self.scan.step(candidate, flush, tool_index > 0) {
            Step::Ended(end) => Some(end),
            Step::Exhausted if flush => self.scan.eof_end(candidate),
            Step::Pending | Step::Exhausted => None,
        };
        self.report_rejected_names();
        end
    }

    fn opens(&self, _text: &str, _at: usize) -> bool {
        true
    }

    fn holdback(&self, _text: &str) -> usize {
        0
    }

    fn resync(&mut self, _text: &str, _flush: bool, _tool_index: usize) -> Option<usize> {
        None
    }

    fn reset(&mut self) {
        *self = Self::default();
    }
}

fn mistral_boundary() -> Box<dyn InvokeBoundary> {
    Box::new(MistralBoundary::default())
}

/// The first complete JSON value of a body, without the text after it, as
/// upstream vLLM keeps it (`raw_decode` in `_extract_args_json`). A value need
/// not be an object. `None` when the body does not start with one.
fn first_json_value(body: &str) -> Option<&str> {
    let mut values =
        serde_json::Deserializer::from_str(body).into_iter::<serde::de::IgnoredAny>();
    values.next()?.ok()?;
    Some(&body[..values.byte_offset()])
}

/// Types one delimited call and keeps its id: the v11 `[CALL_ID]` when it is
/// valid, otherwise a minted one.
#[derive(Default)]
pub(crate) struct MistralEmitter {
    ids: Vec<Option<String>>,
}

impl InvokeEmitter for MistralEmitter {
    fn parse_invoke(
        &mut self,
        invoke: &str,
        tool_index: usize,
    ) -> anyhow::Result<Option<ToolCallDelta>> {
        let mut scan = CallScan::default();
        scan.step(invoke, true, tool_index > 0);
        let (name, call_id, body) = scan.parts(invoke);
        if name.is_empty() && body.is_empty() {
            return Ok(None);
        }
        let body = first_json_value(body).unwrap_or(body);
        let id = if call_id.len() == TOOL_CALL_ID_LEN
            && call_id.chars().all(|ch| ch.is_ascii_alphanumeric())
        {
            call_id.to_string()
        } else {
            let mut minted = uuid::Uuid::new_v4().simple().to_string();
            minted.truncate(TOOL_CALL_ID_LEN);
            minted
        };
        if self.ids.len() <= tool_index {
            self.ids.resize(tool_index + 1, None);
        }
        self.ids[tool_index] = Some(id);
        Ok(Some(ToolCallDelta {
            tool_index,
            name: Some(name.to_string()),
            arguments: if body.is_empty() { "{}" } else { body }.to_string(),
            complete: true,
        }))
    }

    fn tool_call_id(&self, tool_index: usize) -> Option<&str> {
        self.ids.get(tool_index)?.as_deref()
    }

    fn reset(&mut self) {
        self.ids.clear();
    }
}

pub(crate) fn mistral_scanner() -> WrappedBlockScanner<MistralEmitter> {
    WrappedBlockScanner::new(spec(), MistralEmitter::default())
        .with_special_token_markers()
        .with_trimmed_reasoning_tail()
}
