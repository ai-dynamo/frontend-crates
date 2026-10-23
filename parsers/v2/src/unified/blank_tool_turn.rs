// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Whitespace-only text in a tool-only turn (issue #406).
//!
//! Nemotron frames each Qwen3-Coder call with a newline on both sides, so a turn
//! that only calls tools parsed to `Text("\n") | call | Text("\n")`. The rule here
//! follows the v1 jail's (`hold_blank_content` / `settle_blank_content` in
//! `dynamo-parsers`), with one difference: a Unified parser never sees the finish
//! reason, so it cannot keep the whitespace on a truncated (`length`,
//! `content_filter`) turn the way v1 does. A truncated turn that completed a call
//! drops it here.
//!
//! - Whitespace-only text is held until visible (non-whitespace) text arrives. It is
//!   then released unchanged in front of that text, and holding stops for the rest
//!   of the stream.
//! - At finish, held whitespace is dropped if the stream completed a tool call and
//!   showed no visible text; otherwise it is released unchanged.
//! - `reset` returns held whitespace ahead of the inner parser's unconsumed text.
//! - Holding stops at [`MAX_HELD_BLANK_BYTES`], so a model stuck emitting newlines
//!   still streams.
//!
//! Reasoning and tool-call events pass through untouched and in order. The visible
//! text channel is byte-identical whenever the turn has visible text; the one thing
//! that can move is where held whitespace sits relative to an earlier reasoning or
//! call event, because it is released in front of the visible text that decided it.
//!
//! # Why a wrapper in the factory, and only for Nemotron V3
//!
//! The rule is about the turn, not a grammar: whether whitespace is framing depends
//! on bytes that have not arrived yet (a later call, or later prose), which no single
//! scanner position can see, so it wraps whatever parser the factory builds.
//!
//! It applies to `nemotron_v3` (and its aliases) only, because that is where it was
//! measured. Nemotron's chat template trims assistant content, so a history turn
//! with the framing newlines and one with no content render the same prompt tokens,
//! and dropping them costs no prefix-cache reuse. Templates that render content
//! verbatim would lose reuse at that turn on every later request, so other families
//! keep their output unchanged (see [`applies_to`]).

use super::{
    ToolCallPolicy, UnifiedParser, UnifiedParserEvent, UnifiedParserInit,
    UnifiedParserInputCapabilities, UnifiedParserOutput,
};
use crate::tool_calling::traits::Result;

/// Whitespace held before visible text is released past this size. Matches the v1 jail.
pub(crate) const MAX_HELD_BLANK_BYTES: usize = 4096;

/// Families that get the tool-only-turn rule. Keyed by canonical family name.
pub(crate) fn applies_to(canonical_family: &str) -> bool {
    canonical_family == "nemotron_v3"
}

/// Wrap `inner` with the tool-only-turn whitespace rule if its family gets it.
pub(crate) fn wrap(
    canonical_family: &str,
    inner: Box<dyn UnifiedParser>,
) -> Box<dyn UnifiedParser> {
    if !applies_to(canonical_family) {
        return inner;
    }
    Box::new(BlankToolTurn::new(inner))
}

/// Holds whitespace-only text until the turn shows whether it is framing.
struct BlankToolTurn {
    inner: Box<dyn UnifiedParser>,
    /// Whitespace-only text not yet released.
    held: String,
    /// Holding is over for this stream: visible text arrived or the cap was reached.
    closed: bool,
    /// This stream completed a tool call. A provisional delta does not count:
    /// `assemble` drops a call that never completes (truncated, or invalidated by
    /// a duplicate parameter), and the turn then has no call to frame.
    made_tool_call: bool,
}

impl BlankToolTurn {
    fn new(inner: Box<dyn UnifiedParser>) -> Self {
        Self {
            inner,
            held: String::new(),
            closed: false,
            made_tool_call: false,
        }
    }

    /// Route one advance's events into `output`, holding whitespace-only text.
    fn filter(&mut self, events: UnifiedParserOutput, output: &mut UnifiedParserOutput) {
        for event in events {
            match event {
                UnifiedParserEvent::Text(text) if !self.closed => {
                    if text.trim().is_empty()
                        && self.held.len() + text.len() <= MAX_HELD_BLANK_BYTES
                    {
                        self.held.push_str(&text);
                        continue;
                    }
                    self.closed = true;
                    output.push_text(std::mem::take(&mut self.held));
                    output.push_text(text);
                }
                UnifiedParserEvent::ToolCall(call) => {
                    // Same rule `assemble` uses to keep a call.
                    self.made_tool_call |= call.complete;
                    output.push_call(call);
                }
                other => output.push_event(other),
            }
        }
    }

    /// Release held whitespace unchanged and stop holding.
    fn release(&mut self, output: &mut UnifiedParserOutput) {
        self.closed = true;
        output.push_text(std::mem::take(&mut self.held));
    }
}

impl UnifiedParser for BlankToolTurn {
    fn input_capabilities(&self) -> UnifiedParserInputCapabilities {
        self.inner.input_capabilities()
    }

    fn initialize_request(&mut self, init: UnifiedParserInit) -> Result<()> {
        self.inner.initialize_request(init)
    }

    fn set_tool_call_policy(&mut self, policy: ToolCallPolicy) -> Result<()> {
        self.inner.set_tool_call_policy(policy)
    }

    /// Token-id ingestion (GPT-OSS Harmony) gets the same rule as text deltas.
    fn parse_token_ids_into(
        &mut self,
        token_ids: &[u32],
        output: &mut UnifiedParserOutput,
    ) -> Result<()> {
        let mut advanced = UnifiedParserOutput::default();
        let result = self.inner.parse_token_ids_into(token_ids, &mut advanced);
        self.filter(advanced, output);
        if result.is_err() {
            self.release(output);
        }
        result
    }

    fn parse_into(&mut self, delta: &str, output: &mut UnifiedParserOutput) -> Result<()> {
        let mut advanced = UnifiedParserOutput::default();
        let result = self.inner.parse_into(delta, &mut advanced);
        self.filter(advanced, output);
        if result.is_err() {
            // The error contract keeps committed output with the caller. Held
            // whitespace was committed by the inner parser, so it goes out too.
            self.release(output);
        }
        result
    }

    fn finish(&mut self) -> Result<UnifiedParserOutput> {
        let tail = self.inner.finish()?;
        let mut filtered = UnifiedParserOutput::default();
        self.filter(tail, &mut filtered);
        if self.held.is_empty() {
            return Ok(filtered);
        }
        if self.made_tool_call {
            self.held.clear();
            self.closed = true;
            return Ok(filtered);
        }
        // Held whitespace precedes everything this finish produced.
        let mut output = UnifiedParserOutput::default();
        self.release(&mut output);
        output.append(&mut filtered);
        Ok(output)
    }

    fn reset(&mut self) -> String {
        // Held whitespace was never emitted, so it is unconsumed text like the inner
        // parser's buffer: return it first, in stream order, for the caller to recover.
        let mut remainder = std::mem::take(&mut self.held);
        self.closed = false;
        self.made_tool_call = false;
        remainder.push_str(&self.inner.reset());
        remainder
    }

    fn preserve_special_tokens(&self) -> bool {
        self.inner.preserve_special_tokens()
    }

    fn tool_call_id(&self, tool_index: usize) -> Option<&str> {
        self.inner.tool_call_id(tool_index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn applies_only_to_nemotron_v3() {
        assert!(applies_to("nemotron_v3"));
        for family in [
            "qwen3",
            "qwen3_coder",
            "deepseek_v4",
            "deepseek_v41",
            "gpt_oss",
            "beam",
        ] {
            assert!(!applies_to(family), "{family}");
        }
    }

    /// The wrapper forwards the request-scoped tool-call policy (Beam supports one);
    /// without forwarding, the trait default would reject every non-default policy.
    #[test]
    fn tool_call_policy_reaches_the_wrapped_parser() {
        let policy = ToolCallPolicy {
            argument_framing: crate::ArgumentFraming::ConstrainedJson,
            ..Default::default()
        };
        let mut wrapped = BlankToolTurn::new(super::super::beam::beam_unified(&[]));
        wrapped.set_tool_call_policy(policy).unwrap();
    }
}
