// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unified parser for the Muse Glimmer grammar: recipient-routed reasoning,
//! content and ATEM tool calls, in ONE state machine.
//!
//! ```text
//! reasoning:  <|start|>assistant to=self<|message|> … <|eom|>
//! tool call:  <|start|>assistant to=NAME<|message|><atem:invoke name="NAME"> … </atem:invoke><|eom|>
//! content:    <|start|>assistant to=user<|message|> … <|eot|>
//! ```
//!
//! This file is only the trait wiring. The scan core is the same
//! [`crate::tool_calling::muse_glimmer::MuseChannelScanner`] the tool-only
//! `MuseGlimmerToolStreamParser` runs on, so channel routing, the
//! missing-`<|eom|>` recovery, the bare-header latch, chunk-boundary holdback and
//! the end-of-stream drop stay a single implementation.
//!
//! There is deliberately no `ReasoningSpec`: Muse has no reasoning marker PAIR — a
//! dynamic `to=self<|message|>` header opens reasoning, sharing every literal byte
//! except the recipient with the content and tool channels, so routing is by
//! recipient, not marker. That is why this family does not run on
//! [`crate::unified::ScannerUnified`].
//!
//! Having no marker pair is NOT the same as having no reasoning channel, and the two
//! were conflated for a while: guided tool output was refused outright for this
//! family, which left 37 of the 81 unified conformance scenarios ungenerated. The
//! guided reader now asks [`crate::unified::GuidedReasoning`] where a thought begins
//! instead of assuming a fixed opener string, so this family supplies the header
//! resolver it already uses natively and is served guided output like any other.
//!
//! What the unified path adds is ORDER: one machine routes reasoning, content and calls
//! directly, so a thought between two calls stays between them instead of being hoisted
//! ahead of the calls the way the split path does.

use crate::tool_calling::muse_glimmer::{
    GUIDED_CLOSE_MARKERS, GUIDED_COMPETITORS, GUIDED_CONTROL_MARKERS, MuseChannelScanner,
    guided_content_header, guided_header_holdback, guided_reasoning_close, guided_reasoning_open,
    guided_routing_header, guided_stray_header, guided_strip_text, guided_turn_end, muse_scanner,
};
use crate::tool_calling::traits::{Result, Tool};
use crate::unified::{
    GuidedChannel, GuidedGrammar, GuidedReasoning, GuidedRouted, NativeUnified, UnifiedParser,
    UnifiedParserEvent, UnifiedParserOutput, UnifiedParserStartingState,
};

/// The invoke opener, in the prefix form the guided reader anchors on.
const INVOKE_START: &str = "<atem:invoke";
const INVOKE_END: &str = "</atem:invoke>";

impl NativeUnified for MuseChannelScanner {
    fn preserve_special_tokens(&self) -> bool {
        true
    }

    /// Reasoning recognised by RECIPIENT rather than by a marker pair.
    ///
    /// Always `Some`: this family does have a reasoning channel, so an explicitly
    /// prefilled thought and guided tool output are both honourable. Returning
    /// `None` here would silently reinstate the refusal this replaced.
    fn guided_reasoning(&self) -> Option<GuidedReasoning> {
        Some(GuidedReasoning::Channel(GuidedChannel {
            find_open: guided_reasoning_open,
            find_close: guided_reasoning_close,
            find_turn_end: guided_turn_end,
            find_stray: guided_stray_header,
            find_routing: guided_routing_header,
            find_transition: guided_content_header,
            holdback: guided_header_holdback,
            strip_text: guided_strip_text,
            competitors: &GUIDED_COMPETITORS,
            close_markers: &GUIDED_CLOSE_MARKERS,
        }))
    }

    fn guided_grammar(&self) -> GuidedGrammar {
        GuidedGrammar {
            control_markers: GUIDED_CONTROL_MARKERS
                .iter()
                .map(|m| m.to_string())
                .collect(),
            invoke_start: INVOKE_START.to_string(),
            invoke_end: INVOKE_END.to_string(),
            // The marker-only rules are enough here: an ATEM invoke is delimited by a
            // literal opener and closer, with no grammar-aware location rule of the
            // kind gemma4's value wrapping needs.
            invoke_scan: None,
        }
    }

    fn apply_native_init(&mut self, starting_state: UnifiedParserStartingState) {
        self.reset_stream();
        self.apply_starting_state(starting_state);
    }

    fn restore_native_state(&mut self, starting_state: UnifiedParserStartingState) {
        self.apply_starting_state(starting_state);
    }

    fn push_native(&mut self, delta: &str, output: &mut UnifiedParserOutput) -> Result<()> {
        // Through a carry the caller keeps on `Err`, not `push_ordered`'s owned
        // `Result<Vec<_>>`: the drain types arguments mid-advance (`parse_invoke`), so it
        // can commit events and THEN fail, and the trait promises those stay in `output`.
        let mut events = Vec::new();
        let result = self.push_ordered_into(delta, &mut events);
        for event in events {
            push_event(output, event);
        }
        result
    }

    fn finish_native(&mut self, output: &mut UnifiedParserOutput) -> Result<()> {
        for event in self.finish_ordered()? {
            push_event(output, event);
        }
        Ok(())
    }

    /// The trait default returns an empty string and clears nothing. Muse buffers on
    /// every path — a partial marker, an open channel, the bare-header latch and a used
    /// `next_index` — so inheriting it would report an empty carry while the scanner
    /// still held all of that. A caller on the documented recovery path would drop the
    /// held bytes and resume on a counter that re-numbers its first call onto an index
    /// the abandoned stream already dispatched.
    fn reset_native(&mut self) -> String {
        self.reset_stream()
    }
}

fn push_event(output: &mut UnifiedParserOutput, event: UnifiedParserEvent) {
    match event {
        UnifiedParserEvent::Text(text) => output.push_text(text),
        UnifiedParserEvent::Reasoning(text) => output.push_reasoning(text),
        UnifiedParserEvent::ToolCall(call) => output.push_call(call),
    }
}

/// Build the Muse Glimmer unified parser for one stream.
pub(crate) fn muse_glimmer_unified(tools: &[Tool]) -> Box<dyn UnifiedParser> {
    Box::new(GuidedRouted::new(muse_scanner(tools)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::unified::{
        InvalidGuidedPayloadPolicy, UnifiedEvent, UnifiedParserExt, UnifiedParserInit,
        UnifiedToolOutputMode, assemble,
    };

    /// The conformance harness vocabulary, so a unit test and a golden case can
    /// describe the same call.
    fn tools() -> Vec<Tool> {
        ["get_weather", "f", "g", "run"]
            .iter()
            .map(|n| Tool {
                name: n.to_string(),
                description: None,
                parameters: serde_json::json!({"type": "object", "properties": {}}),
                strict: None,
            })
            .collect()
    }

    fn events(chunks: &[&str]) -> Vec<UnifiedEvent> {
        let mut parser = muse_glimmer_unified(&tools());
        let mut deltas = Vec::new();
        for chunk in chunks {
            deltas.extend(parser.push(chunk).expect("push"));
        }
        deltas.extend(parser.finish().expect("finish"));
        assemble(&deltas)
    }

    fn batch(input: &str) -> Vec<UnifiedEvent> {
        muse_glimmer_unified(&tools())
            .parse_complete(input)
            .expect("parse_complete")
    }

    fn reasoning(text: &str) -> UnifiedEvent {
        UnifiedEvent::Reasoning { text: text.into() }
    }
    fn text(text: &str) -> UnifiedEvent {
        UnifiedEvent::Text { text: text.into() }
    }
    fn call(name: &str, arguments: serde_json::Value) -> UnifiedEvent {
        UnifiedEvent::ToolCall {
            name: name.into(),
            arguments,
        }
    }

    /// One `to=<recipient>` channel, framed and closed with `<|eom|>`.
    fn channel(recipient: &str, body: &str) -> String {
        format!("<|start|>assistant to={recipient}<|message|>{body}<|eom|>")
    }

    /// One single-parameter ATEM tool channel.
    fn tool_channel(name: &str, key: &str, value: &str) -> String {
        channel(
            name,
            &format!(
                "<atem:function_calls>\n<atem:invoke name=\"{name}\">\n\
                 <atem:parameter name=\"{key}\">{value}</atem:parameter>\n\
                 </atem:invoke>\n</atem:function_calls>"
            ),
        )
    }

    // ── Ported from the v1 reasoning parser ───────────────────────────────────

    #[test]
    fn reasoning_then_answer() {
        let out = events(&[
            " to=self<|message|>The user asks 2+2. It is 4.<|eom|>",
            "<|start|>assistant to=user<|message|>2 + 2 = 4.<|eot|>",
        ]);
        assert_eq!(
            out,
            vec![reasoning("The user asks 2+2. It is 4."), text("2 + 2 = 4.")]
        );
    }

    #[test]
    fn adjacent_reasoning_blocks_join_with_a_newline() {
        // The `\n` is the concatenation artifact of v1's single `reasoning_text`
        // field, which both engines' batch parsers also produce. Adjacent blocks
        // keep it so v1 and v2 agree on the case the corpus contains.
        let out = events(&[
            " to=self<|message|>first<|eom|>",
            "<|start|>assistant to=self<|message|>second<|eom|>",
            "<|start|>assistant to=user<|message|>done<|eot|>",
        ]);
        assert_eq!(out, vec![reasoning("first\nsecond"), text("done")]);
    }

    #[test]
    fn unframed_prose_is_text() {
        assert_eq!(events(&["plain answer"]), vec![text("plain answer")]);
    }

    #[test]
    fn empty_input_emits_nothing() {
        assert_eq!(events(&[""]), vec![]);
    }

    #[test]
    fn unterminated_reasoning_is_promoted_at_finish() {
        let out = events(&[" to=self<|message|>cut off mid thought"]);
        assert_eq!(out, vec![reasoning("cut off mid thought")]);
    }

    #[test]
    fn empty_reasoning_block_emits_no_reasoning_event() {
        let out = events(&[
            " to=self<|message|><|eom|>",
            "<|start|>assistant to=user<|message|>hi<|eot|>",
        ]);
        assert_eq!(out, vec![text("hi")]);
    }

    #[test]
    fn recipient_less_header_is_content() {
        let out = events(&["<|start|>assistant<|message|>untagged content<|eot|>"]);
        assert_eq!(out, vec![text("untagged content")]);
    }

    #[test]
    fn reasoning_ends_at_a_bare_tool_header() {
        // Invariant 1: the observed defect — the analysis channel is abandoned
        // without `<|eom|>` and the tool header follows directly. The space before
        // the recovered header stays in the body, exactly as vLLM's bounded
        // open-reasoning strip behaves.
        let out = events(&[
            " to=self<|message|>thinking to=get_weather<|message|>",
            "<atem:invoke name=\"get_weather\"><atem:parameter name=\"city\">Paris</atem:parameter></atem:invoke><|eom|>",
        ]);
        assert_eq!(
            out,
            vec![
                reasoning("thinking "),
                call("get_weather", serde_json::json!({"city": "Paris"})),
            ]
        );
    }

    #[test]
    fn quoted_bare_header_in_an_answer_is_not_a_call() {
        // Invariants 2 and 5: the recovery is reasoning-only, so a bare header
        // quoted inside a `to=user` answer stays prose. Its `<|message|>` is
        // dropped as orphan framing (`I3`), which also keeps the v1 chain — whose
        // tool parser rescans this text — from resolving the quoted header.
        let out = events(&[
            "<|start|>assistant to=user<|message|>Example: to=g<|message|>",
            "<atem:invoke name=\"g\"><atem:parameter name=\"y\">oops</atem:parameter></atem:invoke><|eot|>",
        ]);
        assert_eq!(
            out,
            vec![text(
                "Example: to=g<atem:invoke name=\"g\"><atem:parameter name=\"y\">oops</atem:parameter></atem:invoke>"
            )]
        );
    }

    #[test]
    fn atem_markup_quoted_in_reasoning_stays_reasoning() {
        // Invariant 5: `extract_invokes` runs only inside a tool channel, so there
        // is no scan-everything fallback to promote quoted markup.
        let out = events(&[
            " to=self<|message|>I could emit <atem:invoke name=\"f\"> but will not.<|eom|>",
            "<|start|>assistant to=user<|message|>ok<|eot|>",
        ]);
        assert_eq!(
            out,
            vec![
                reasoning("I could emit <atem:invoke name=\"f\"> but will not."),
                text("ok"),
            ]
        );
    }

    #[test]
    fn orphan_terminator_is_stripped() {
        // Invariant 3.
        assert_eq!(events(&["<|eom|>"]), vec![]);
        assert_eq!(
            events(&["some prose<|eom|> more"]),
            vec![text("some prose more")]
        );
    }

    #[test]
    fn prose_before_the_first_header_is_text() {
        let out = events(&["Hello. to=self<|message|>think<|eom|>"]);
        assert_eq!(out, vec![text("Hello."), reasoning("think")]);
    }

    #[test]
    fn tool_channel_cut_by_framed_header_keeps_the_answer() {
        // Invariant 4, restated for unified. v1 needed a synthetic `<|eom|>` so the
        // downstream tool parser would not absorb the answer as tool payload; here
        // the `<|start|>` boundary returns the ONE machine to Idle, from which the
        // framed header routes the answer to content. Same observable behavior, no
        // forwarding contract to keep in sync.
        let out = events(&[
            " to=get_weather<|message|><atem:function_calls>\n",
            "<atem:invoke name=\"get_weather\">\n",
            "<atem:parameter name=\"city\">Paris</atem:parameter>\n",
            "</atem:invoke>\n</atem:function_calls>",
            "<|start|>assistant to=user<|message|>Here is the weather.<|eot|>",
        ]);
        assert_eq!(
            out,
            vec![
                call("get_weather", serde_json::json!({"city": "Paris"})),
                text("Here is the weather."),
            ]
        );
    }

    #[test]
    fn committed_partial_special_token_is_dropped_at_finish() {
        let out = events(&[" to=self<|message|>thought<|eo"]);
        assert_eq!(out, vec![reasoning("thought")]);
    }

    #[test]
    fn ambiguous_angle_prefix_is_kept_at_finish() {
        // `<` / `<|` are ordinary prose as often as framing.
        let out = events(&[" to=user<|message|>a < b and a <| b"]);
        assert_eq!(out, vec![text("a < b and a <| b")]);
    }

    #[test]
    fn trailing_to_fragment_is_flushed_as_prose() {
        let out = events(&[" to=user<|message|>walk me to<|eot|>"]);
        assert_eq!(out, vec![text("walk me to")]);
    }

    #[test]
    fn crlf_reasoning_body_and_unicode_recipient_route_cleanly() {
        let out = events(&[
            " to=self<|message|>line one\r\nline two<|eom|>",
            "<|start|>assistant to=天気<|message|><atem:invoke name=\"天気\"></atem:invoke><|eom|>",
        ]);
        assert_eq!(
            out,
            vec![
                reasoning("line one\r\nline two"),
                call("天気", serde_json::json!({})),
            ]
        );
    }

    // ── Ported from parsers/v1/tests/jail.rs ──────────────────────────────────

    #[test]
    fn chained_channels_in_one_delta_emit_before_terminal() {
        // Invariant 7, and the direct replacement for v1's
        // `find_tool_call_end_position_muse_glimmer` walk: the jail needed that walk
        // to cut a span holding both chained calls, but the drain loop `continue`s
        // after each call, so one push of an interval-batched delta emits both.
        let input = format!(
            "{}{}",
            tool_channel("get_weather", "city", "Paris"),
            tool_channel("f", "x", "1")
        );
        assert_eq!(
            events(&[&input]),
            vec![
                call("get_weather", serde_json::json!({"city": "Paris"})),
                call("f", serde_json::json!({"x": 1})),
            ]
        );
    }

    #[test]
    fn a_call_emits_on_its_close_not_on_the_channel_terminator() {
        // The incrementality contract, at the level `assemble` hides: a call is
        // released the moment `</atem:invoke>` streams, and every call in one chunk
        // leaves in THAT push. Comparing assembled output cannot tell a prompt
        // delta from one withheld until `finish`.
        let calls = |d: &[UnifiedParserEvent]| {
            d.iter()
                .filter(|x| matches!(x, UnifiedParserEvent::ToolCall(_)))
                .count()
        };

        let mut parser = muse_glimmer_unified(&tools());
        let open = parser
            .push(" to=f<|message|><atem:invoke name=\"f\"><atem:parameter name=\"x\">1</atem:parameter>")
            .expect("push");
        assert_eq!(calls(&open), 0, "emitted before its close streamed");
        let closed = parser.push("</atem:invoke>").expect("push");
        assert_eq!(calls(&closed), 1, "withheld until the channel terminator");

        // Invariant 7, per push: two whole channels in one chunk emit both calls.
        let mut parser = muse_glimmer_unified(&tools());
        let both = parser
            .push(&format!(
                "{}{}",
                tool_channel("f", "x", "1"),
                tool_channel("g", "y", "2")
            ))
            .expect("push");
        assert_eq!(
            calls(&both),
            2,
            "a chained call slipped past its terminator"
        );
        let finished: Vec<_> = parser.finish().expect("finish").into_iter().collect();
        assert_eq!(calls(&finished), 0);
    }

    #[test]
    fn narration_between_channels_in_one_delta() {
        let input = format!(
            "{} Now the time. {}",
            tool_channel("get_weather", "city", "Paris"),
            tool_channel("f", "x", "1")
        );
        assert_eq!(
            events(&[&input]),
            vec![
                call("get_weather", serde_json::json!({"city": "Paris"})),
                text(" Now the time. "),
                call("f", serde_json::json!({"x": 1})),
            ]
        );
    }

    #[test]
    fn whitespace_separated_channels_in_one_delta() {
        let input = format!(
            "{}\n{}",
            tool_channel("get_weather", "city", "Paris"),
            tool_channel("f", "x", "1")
        );
        assert_eq!(
            events(&[&input]),
            vec![
                call("get_weather", serde_json::json!({"city": "Paris"})),
                text("\n"),
                call("f", serde_json::json!({"x": 1})),
            ]
        );
    }

    #[test]
    fn prefixed_channels_then_empty_chunk_order() {
        let input = format!(
            "Checking. {} {}",
            tool_channel("get_weather", "city", "Paris"),
            tool_channel("f", "x", "1")
        );
        assert_eq!(
            events(&[&input, ""]),
            vec![
                text("Checking. "),
                call("get_weather", serde_json::json!({"city": "Paris"})),
                text(" "),
                call("f", serde_json::json!({"x": 1})),
            ]
        );
    }

    #[test]
    fn missing_eom_recovers_at_finalize() {
        let out = events(&[concat!(
            "<|start|>assistant to=get_weather<|message|><atem:function_calls>\n",
            "<atem:invoke name=\"get_weather\">\n",
            "<atem:parameter name=\"city\">Paris</atem:parameter>\n",
            "</atem:invoke>\n</atem:function_calls>"
        )]);
        assert_eq!(
            out,
            vec![call("get_weather", serde_json::json!({"city": "Paris"}))]
        );
    }

    #[test]
    fn quoted_atem_in_content_is_not_a_call() {
        let out = events(&[
            "Example: <atem:function_calls><atem:invoke ",
            "name=\"g\"></atem:invoke></atem:function_calls>",
        ]);
        assert_eq!(
            out,
            vec![text(
                "Example: <atem:function_calls><atem:invoke name=\"g\"></atem:invoke></atem:function_calls>"
            )]
        );
    }

    #[test]
    fn a_special_token_in_a_parameter_value_is_data_only_until_it_ends_the_channel() {
        // `opaque: ["atem:parameter"]` in parser_families.yaml colours a parameter
        // body as argument DATA. That holds for `<|message|>`, which has no
        // body-cutting role, but NOT for the channel terminators: they close the
        // message before `</atem:invoke>` arrives, so the invoke is truncated and
        // dropped, and the residual markup surfaces as the prose it now is. Both
        // engines truncate on `<|eom|>` / `<|eot|>` the same way.
        //
        // `<|start|>` is where the two sides part: this crate reads the decoded
        // reserved token as a REAL channel switch (see the `next_channel_message`
        // rationale in the v1 parser) and drops the call, while SGLang's `muse`
        // detector keeps it as data and emits `{"x": "a<|start|>b"}`. The corpus has
        // no case for it, so nothing renders it today — this pins which side of the
        // gap Dynamo is on, so a change of mind is deliberate rather than silent.
        let call_of = |marker: &str| {
            format!(
                "<|start|>assistant to=f<|message|><atem:invoke name=\"f\">\
                 <atem:parameter name=\"x\">a{marker}b</atem:parameter></atem:invoke><|eom|>"
            )
        };
        let truncated = vec![text("b</atem:parameter></atem:invoke>")];
        for marker in ["<|eom|>", "<|eot|>", "<|start|>"] {
            assert_eq!(events(&[&call_of(marker)]), truncated, "marker {marker}");
        }
        assert_eq!(
            events(&[&call_of("<|message|>")]),
            vec![call("f", serde_json::json!({"x": "a<|message|>b"}))]
        );
    }

    #[test]
    fn an_integer_argument_past_u64_loses_precision_like_the_v1_parser() {
        // `decode_value` types a parameter with `serde_json::from_str`, which has no
        // arbitrary-precision path: an integer literal above `u64::MAX` becomes an
        // `f64`, so `18446744073709551617` serializes back as `1.8446744073709552e19`.
        // SGLang's `muse` detector keeps the digits exactly, so this is a real parity
        // gap — but it is the v1 muse parser's behaviour byte for byte, NOT something
        // the unified port introduced, and no corpus case carries such a value. Pin it
        // so closing the gap is a deliberate change to BOTH generations at once
        // (`raw_number_literal` in `parsers/v1/src/tool_calling/xml/parsed_value.rs`
        // is the pattern the XML families already use), rather than a silent drift.
        let big = "<|start|>assistant to=f<|message|><atem:invoke name=\"f\">\
                   <atem:parameter name=\"n\">18446744073709551617</atem:parameter>\
                   </atem:invoke><|eom|>";
        assert_eq!(
            events(&[big]),
            vec![call("f", serde_json::json!({"n": 1.8446744073709552e19}))]
        );
        // Everything inside i64 is exact, including past f64's contiguous range.
        let exact = big.replace("18446744073709551617", "9007199254740993");
        assert_eq!(
            events(&[&exact]),
            vec![call("f", serde_json::json!({"n": 9007199254740993i64}))]
        );
    }

    // ── New: what the split path structurally cannot express ──────────────────

    #[test]
    fn reasoning_after_a_call_keeps_its_position() {
        // The defect the unified parser exists to fix: under the split, both
        // thoughts merge into one span ahead of the call.
        let input = format!(
            "{}{}{}{}",
            channel("self", "Look it up."),
            tool_channel("get_weather", "city", "Paris"),
            channel("self", "Now answer."),
            "<|start|>assistant to=user<|message|>It's 18C.<|eot|>"
        );
        assert_eq!(
            events(&[&input]),
            vec![
                reasoning("Look it up."),
                call("get_weather", serde_json::json!({"city": "Paris"})),
                reasoning("Now answer."),
                text("It's 18C."),
            ]
        );
    }

    #[test]
    fn reasoning_call_reasoning_call_reasoning() {
        let input = format!(
            "{}{}{}{}{}",
            channel("self", "A"),
            tool_channel("f", "x", "1"),
            channel("self", "B"),
            tool_channel("g", "y", "2"),
            channel("self", "C")
        );
        assert_eq!(
            events(&[&input]),
            vec![
                reasoning("A"),
                call("f", serde_json::json!({"x": 1})),
                reasoning("B"),
                call("g", serde_json::json!({"y": 2})),
                reasoning("C"),
            ]
        );
    }

    #[test]
    fn content_before_reasoning_is_not_hoisted() {
        let input = format!(
            "{}{}{}",
            channel("user", "Hello there. "),
            channel("self", "let me recall"),
            "<|start|>assistant to=user<|message|>The capital is Paris.<|eot|>"
        );
        assert_eq!(
            events(&[&input]),
            vec![
                text("Hello there. "),
                reasoning("let me recall"),
                text("The capital is Paris."),
            ]
        );
    }

    #[test]
    fn two_thoughts_separated_by_a_call_do_not_join_with_a_newline() {
        // The counterpart to `adjacent_reasoning_blocks_join_with_a_newline`: the
        // separator is a v1 concatenation artifact, so emitting it across a call
        // would invent bytes the model never produced.
        let input = format!(
            "{}{}{}",
            channel("self", "a"),
            tool_channel("f", "x", "1"),
            channel("self", "b")
        );
        assert_eq!(
            events(&[&input]),
            vec![
                reasoning("a"),
                call("f", serde_json::json!({"x": 1})),
                reasoning("b"),
            ]
        );
    }

    #[test]
    fn an_empty_answer_between_two_thoughts_still_joins_them() {
        // The join latch tracks what was EMITTED, not what the model framed: an
        // empty `to=user` message produces no delta, so the thoughts around it are
        // still adjacent and must join. Clearing the latch on the mere presence of
        // a content channel would silently drop the separator.
        let out = events(&[
            &channel("self", "a"),
            &channel("user", ""),
            &channel("self", "b"),
        ]);
        assert_eq!(out, vec![reasoning("a\nb")]);
    }

    #[test]
    fn two_invokes_with_the_same_name_in_one_channel_stay_two_calls() {
        // `assemble` joins argument fragments by `tool_index`, so a scanner that
        // reused an index here would silently concatenate two calls into one.
        let out = events(&[concat!(
            " to=f<|message|>",
            "<atem:invoke name=\"f\"><atem:parameter name=\"x\">1</atem:parameter></atem:invoke>",
            "<atem:invoke name=\"f\"><atem:parameter name=\"x\">2</atem:parameter></atem:invoke>",
            "<|eom|>"
        )]);
        assert_eq!(
            out,
            vec![
                call("f", serde_json::json!({"x": 1})),
                call("f", serde_json::json!({"x": 2})),
            ]
        );
    }

    #[test]
    fn interstitial_text_between_two_calls() {
        let input = format!(
            "{}{}{}",
            tool_channel("f", "x", "1"),
            channel("user", " then "),
            tool_channel("g", "y", "2")
        );
        assert_eq!(
            events(&[&input]),
            vec![
                call("f", serde_json::json!({"x": 1})),
                text(" then "),
                call("g", serde_json::json!({"y": 2})),
            ]
        );
    }

    #[test]
    fn parallel_calls_keep_source_order_and_indices() {
        let input = format!(
            "{}{}",
            tool_channel("g", "y", "2"),
            tool_channel("f", "x", "1")
        );
        let mut parser = muse_glimmer_unified(&tools());
        let mut deltas = parser.push(&input).expect("push");
        deltas.extend(parser.finish().expect("finish"));
        let indices: Vec<(usize, Option<String>)> = deltas
            .iter()
            .filter_map(|d| match d {
                UnifiedParserEvent::ToolCall(c) => Some((c.tool_index, c.name.clone())),
                _ => None,
            })
            .collect();
        assert_eq!(
            indices,
            vec![(0, Some("g".into())), (1, Some("f".into()))],
            "tool_index counts up across channels, in model order"
        );
    }

    #[test]
    fn batch_and_stream_assemble_identically() {
        // I6, at the parser level: `parse_complete` routes through the same
        // push/finish lifecycle, so parity is structural.
        let input = format!(
            "{}{}{}{}",
            channel("self", "a"),
            channel("user", "Here you go: "),
            tool_channel("get_weather", "city", "Paris"),
            channel("self", "b")
        );
        let batched = batch(&input);
        assert_eq!(batched, events(&[&input]));
        assert_eq!(batched.len(), 4);
    }

    #[test]
    fn every_marker_split_across_chunks_never_leaks() {
        let input = format!(
            "{}{}{}",
            channel("self", "think"),
            tool_channel("get_weather", "city", "Paris"),
            "<|start|>assistant to=user<|message|>done<|eot|>"
        );
        let markers = [
            "<|start|>",
            "<|message|>",
            "<|eom|>",
            "<|eot|>",
            "<atem:invoke",
            "</atem:invoke>",
        ];
        for marker in markers {
            let at = input.find(marker).expect("marker present") + marker.len() / 2;
            let out = events(&[&input[..at], &input[at..]]);
            for event in &out {
                let payload = match event {
                    UnifiedEvent::Reasoning { text } | UnifiedEvent::Text { text } => text.clone(),
                    UnifiedEvent::ToolCall { .. } => continue,
                };
                for m in markers {
                    assert!(
                        !payload.contains(m),
                        "marker {m:?} leaked into {payload:?} when splitting {marker:?}"
                    );
                }
            }
            assert_eq!(
                out,
                events(&[&input]),
                "split inside {marker:?} changed the parse"
            );
        }
    }

    #[test]
    fn a_recipient_split_across_chunks_resolves() {
        let out = events(&[
            " to=self<|message|>go<|eom|><|start|>assistant to=get_",
            "weather<|message|><atem:invoke name=\"get_weather\">",
            "<atem:parameter name=\"city\">Paris</atem:parameter></atem:invoke><|eom|>",
        ]);
        assert_eq!(
            out,
            vec![
                reasoning("go"),
                call("get_weather", serde_json::json!({"city": "Paris"})),
            ]
        );
    }

    #[test]
    fn a_bare_header_split_mid_word_does_not_cut_reasoning() {
        // Invariant 2: `last_body_char` carries the whitespace anchor across the
        // chunk boundary, so `pota` + `to=f<|message|>` stays one word.
        let out = events(&[
            " to=self<|message|>weird pota",
            "to=f<|message|><atem:invoke name=\"f\"></atem:invoke><|eom|>",
        ]);
        // The `<|message|>` is stripped now that reasoning runs through `emit_reasoning`.
        // The ATEM markup beside it still is not: that is the priced cost of
        // `bare_header_pos` refusing an unanchored cut, which the strip cannot reach.
        assert_eq!(
            out,
            vec![reasoning(
                "weird potato=f<atem:invoke name=\"f\"></atem:invoke>"
            )]
        );
    }

    /// `I3` says a control marker never appears inside a text OR a reasoning payload.
    /// Both routes now honour it: the same bytes come back stripped either way.
    ///
    /// `<|start|>`, `<|eom|>` and `<|eot|>` all CUT a body, so `<|message|>` is the one
    /// marker that can reach a reader at all, which is why it is the case pinned here.
    ///
    /// This puts Dynamo AHEAD of both engines rather than level with them, deliberately:
    /// vLLM's `_CHANNEL_HEADER_RE` requires a recipient, so a bare `<|message|>` never
    /// ends a body and both its batch `_REASONING_RE` and its streaming `_classify_bodies`
    /// return `a<|message|>b`; SGLang's `MuseGlimmerDetector` appends the body to
    /// `reasoning_parts` with no strip. Expect the conformance suite to score these as
    /// divergences until the engines follow.
    #[test]
    fn an_orphan_marker_is_stripped_on_the_reasoning_route_as_well_as_the_content_one() {
        assert_eq!(
            batch(" to=self<|message|>note the <|message|> token<|eom|>"),
            vec![reasoning("note the  token")]
        );
        assert_eq!(
            batch(" to=user<|message|>note the <|message|> token<|eot|>"),
            vec![text("note the  token")]
        );

        // The two routes must agree byte for byte on identical input.
        assert_eq!(
            batch(" to=self<|message|>a<|message|>b<|eom|>")
                .into_iter()
                .chain(batch(" to=user<|message|>a<|message|>b<|eot|>"))
                .collect::<Vec<_>>(),
            vec![reasoning("ab"), text("ab")]
        );
    }

    #[test]
    fn a_marker_spliced_across_a_run_seam_also_costs_chunk_invariance() {
        // `stripped` retracts a marker its own removals splice together, but only
        // inside ONE emitted run. Where a run ENDS is therefore where the splice
        // survives, and that is not only a concatenation artifact: it moves the
        // assembled event list, so these inputs are the exception to I5 and I6.
        //
        // Whole, the prose is one run and the splice cancels to nothing.
        assert!(batch("<|mes<|eot|>sage|>").is_empty());
        // Chunked so `<|mes` is released before `sage|>` arrives, the two halves land
        // in separate runs and the marker re-forms in the reader's text.
        assert_eq!(
            events(&["<|mes", "<|eot|>", "sage|>"]),
            vec![text("<|message|>")]
        );

        // A real header between the halves splits the run in BATCH too, so this shape
        // leaks the same either way and chunk-invariance still holds for it.
        let framed = "foo<|st to=user<|message|>art|>bar<|eot|>";
        assert_eq!(batch(framed), vec![text("foo<|start|>bar")]);
        assert_eq!(events(&[framed]), batch(framed));

        // Left as is. Closing the seam means carrying a held marker prefix ACROSS a
        // consumed header or terminator and re-joining it to a different channel's
        // text, which moves event boundaries and so trades I5 for I2. Holding it in
        // batch alone would split batch from stream instead. `flush_open_text` also
        // releases a trailing `<` / `<|` on purpose, because holding those would stall
        // every `<` a model writes. Reaching this needs the model to emit a marker's
        // first half as ordinary text, then a real special token, then the second
        // half; a marker it merely quotes whole is stripped by the run it sits in.
    }

    /// A channel TRANSITION is not a channel CLOSER, and confusing them dispatched a
    /// tool call the model never made.
    ///
    /// `to=user` routes the turn to visible content. Consuming it as if it were an
    /// ordinary reasoning closer dropped the turn back into the guided JSON
    /// accumulator, so a visible answer that happened to look like a call was parsed as
    /// one. The model chose text and the client executed a tool — failing OPEN on a side
    /// effect, which is the worst outcome this parser can produce.
    ///
    /// Routing to content is one-way until the MESSAGE ends, not for the whole turn:
    /// the native scan opens a tool channel after a content message, so a closer puts
    /// the turn back where a later payload can route it again. Both halves are asserted
    /// here, because fixing only the first loses a legitimate call after `<|eom|>`.
    ///
    /// And `<|eom|>` is not `<|eot|>`. A message close leaves room for a later routed
    /// message; a TURN END does not, so bytes behind it are trailing text however much
    /// they look like a call. Treating the two as one closer re-opened the payload
    /// accumulator after the turn was already over.
    #[test]
    fn a_content_transition_keeps_the_answer_visible() {
        let call = r#"[{"name":"get_weather","arguments":{"city":"Paris"}}]"#;
        let text = |t: &str| UnifiedEvent::Text { text: t.into() };
        let reasoning = |t: &str| UnifiedEvent::Reasoning { text: t.into() };
        let weather = || UnifiedEvent::ToolCall {
            name: "get_weather".into(),
            arguments: serde_json::json!({"city": "Paris"}),
        };
        for (label, start, input, want) in [
            (
                "JSON-shaped answer from outside reasoning",
                UnifiedParserStartingState::None,
                format!("<|start|>assistant to=user<|message|>{call}"),
                vec![text(call)],
            ),
            (
                "JSON-shaped answer after an open thought",
                UnifiedParserStartingState::None,
                format!(
                    "<|start|>assistant to=self<|message|>t<|start|>assistant to=user<|message|>{call}"
                ),
                vec![reasoning("t"), text(call)],
            ),
            (
                "JSON-shaped answer from a PREFILLED thought",
                UnifiedParserStartingState::Reasoning,
                format!("t<|start|>assistant to=user<|message|>{call}"),
                vec![reasoning("t"), text(call)],
            ),
            (
                "ordinary prose is unaffected",
                UnifiedParserStartingState::None,
                "<|start|>assistant to=user<|message|>just words".to_string(),
                vec![text("just words")],
            ),
            (
                "malformed JSON stays visible too",
                UnifiedParserStartingState::None,
                r#"<|start|>assistant to=user<|message|>[{"name": "#.to_string(),
                vec![text(r#"[{"name": "#)],
            ),
            (
                // The other direction: the message ENDS, so a payload may route again.
                "a closed content message releases the turn",
                UnifiedParserStartingState::None,
                format!("<|start|>assistant to=user<|message|>ans<|eom|>{call}"),
                vec![text("ans"), weather()],
            ),
            (
                // ...but the TURN ending does not. Nothing routes after `<|eot|>`.
                "a terminal closer does not release the turn",
                UnifiedParserStartingState::None,
                format!("<|start|>assistant to=user<|message|>ans<|eot|>{call}"),
                vec![text(&format!("ans{call}"))],
            ),
        ] {
            let drive = |chunks: Vec<&str>| {
                let mut parser = muse_glimmer_unified(&tools());
                parser
                    .initialize_request(UnifiedParserInit {
                        prompt_token_ids: Vec::new(),
                        starting_state: start,
                        tool_output_mode: UnifiedToolOutputMode::GuidedJson { named_tool: None },
                        invalid_guided_payload: InvalidGuidedPayloadPolicy::RecoverAsText,
                    })
                    .expect("guided init must be accepted");
                let mut deltas = Vec::new();
                for c in chunks {
                    deltas.extend(parser.push(c).expect("push"));
                }
                deltas.extend(parser.finish().expect("finish"));
                assemble(&deltas)
            };
            assert_eq!(drive(vec![&input]), want, "{label}: whole input");
            let per_char: Vec<String> = input.chars().map(|c| c.to_string()).collect();
            assert_eq!(
                drive(per_char.iter().map(String::as_str).collect()),
                want,
                "{label}: one character at a time"
            );
            for at in 1..input.len() {
                if !input.is_char_boundary(at) {
                    continue;
                }
                assert_eq!(
                    drive(vec![&input[..at], &input[at..]]),
                    want,
                    "{label}: split at byte {at}"
                );
            }
            // Conservation: a transition must never dispatch a call from the answer.
            if !label.starts_with("a closed") {
                assert!(
                    !drive(vec![&input])
                        .iter()
                        .any(|e| matches!(e, UnifiedEvent::ToolCall { .. })),
                    "{label}: a visible answer was dispatched as a tool call"
                );
            }
        }
    }

    /// Stripping control markup is not a routing decision, and a routed turn must not
    /// hold prose back.
    ///
    /// Two consequences of tracking turn scope, each measured before being fixed:
    ///
    /// - an ORPHAN recipient-less `<|message|>` is markup that gets stripped and routes
    ///   nothing, but it was counted as a header and spent the turn's routing scope. The
    ///   next REAL bare header was then demoted and its recipient came out as visible
    ///   text — ` to=get_weather` in the user's answer.
    /// - once a payload has routed the turn, a trailing `to=`-shaped fragment can no
    ///   longer become a header, so holding it back delays bytes whose meaning is already
    ///   settled. Asserted on what `push` RETURNS, before `finish`, because that is the
    ///   only place the delay is observable.
    #[test]
    fn markup_does_not_route_and_a_routed_turn_makes_progress() {
        let call = r#"[{"name":"get_weather","arguments":{"city":"Paris"}}]"#;
        let weather = || UnifiedEvent::ToolCall {
            name: "get_weather".into(),
            arguments: serde_json::json!({"city": "Paris"}),
        };
        let guided = |start: UnifiedParserStartingState| {
            let mut parser = muse_glimmer_unified(&tools());
            parser
                .initialize_request(UnifiedParserInit {
                    prompt_token_ids: Vec::new(),
                    starting_state: start,
                    tool_output_mode: UnifiedToolOutputMode::GuidedJson { named_tool: None },
                    invalid_guided_payload: InvalidGuidedPayloadPolicy::RecoverAsText,
                })
                .expect("guided init must be accepted");
            parser
        };

        // An orphan marker leaves the scope alone, so the later bare header still routes.
        let orphan = format!("thinking<|message|>still<|eom|> to=get_weather<|message|>{call}");
        // A real stray header DOES spend it: the same trailing bare header is a quote
        // once a tool-routed header has already run the turn.
        // No payload on this one: prose ahead of the JSON means it is not a guided
        // payload, which is existing behaviour and not what this sibling is about.
        let real = " to=other<|message|>body<|eom|>I mean to=self<|message|>literal".to_string();
        for (label, start, input, want) in [
            (
                "an orphan marker does not spend routing scope",
                UnifiedParserStartingState::Reasoning,
                orphan,
                vec![
                    UnifiedEvent::Reasoning {
                        text: "thinkingstill".into(),
                    },
                    weather(),
                ],
            ),
            (
                "a real routed header does spend it",
                UnifiedParserStartingState::None,
                real,
                vec![UnifiedEvent::Text {
                    text: "bodyI mean to=selfliteral".into(),
                }],
            ),
        ] {
            let drive = |chunks: Vec<&str>| {
                let mut parser = guided(start);
                let mut deltas = Vec::new();
                for c in chunks {
                    deltas.extend(parser.push(c).expect("push"));
                }
                deltas.extend(parser.finish().expect("finish"));
                assemble(&deltas)
            };
            assert_eq!(drive(vec![&input]), want, "{label}: whole input");
            let per_char: Vec<String> = input.chars().map(|c| c.to_string()).collect();
            assert_eq!(
                drive(per_char.iter().map(String::as_str).collect()),
                want,
                "{label}: one character at a time"
            );
            for at in 1..input.len() {
                if !input.is_char_boundary(at) {
                    continue;
                }
                assert_eq!(
                    drive(vec![&input[..at], &input[at..]]),
                    want,
                    "{label}: split at byte {at}"
                );
            }
        }

        // INTERMEDIATE PROGRESS: what `push` returns, with no `finish` behind it.
        let mut routed = guided(UnifiedParserStartingState::None);
        let pushed = routed.push(&format!("{call}I mean to=self")).expect("push");
        assert_eq!(
            assemble(&pushed),
            vec![
                weather(),
                UnifiedEvent::Text {
                    text: "I mean to=self".into(),
                },
            ],
            "a routed turn held prose back until finish"
        );

        // ...and the contrast: before anything routes the turn, and inside an open
        // thought, a bare-header prefix really can still complete, so it stays held.
        for (label, start, head) in [
            (
                "unrouted",
                UnifiedParserStartingState::None,
                "I mean to=self",
            ),
            (
                "in reasoning",
                UnifiedParserStartingState::Reasoning,
                "thinking to=self",
            ),
        ] {
            let mut parser = guided(start);
            let pushed = parser.push(head).expect("push");
            let held = assemble(&pushed);
            assert!(
                held.iter().all(|e| match e {
                    UnifiedEvent::Text { text } | UnifiedEvent::Reasoning { text } =>
                        !text.contains("to=self"),
                    _ => true,
                }),
                "{label}: a still-completable bare header was released early: {held:?}"
            );
        }
    }

    /// Turn position and open channel are INDEPENDENT, and header resolution needs
    /// both.
    ///
    /// A single "bare headers still allowed" flag cannot carry them, and collapsing
    /// them broke this in both directions at once:
    ///
    /// - a guided payload routes the turn without any header doing it, so the flag
    ///   stayed permissive and a bare header the model QUOTED after the call was
    ///   promoted into a real thought;
    /// - consuming the turn's opening reasoning header closed the flag, so a real bare
    ///   tool header INSIDE the thought — the missing-terminator recovery boundary —
    ///   was demoted and its recipient words leaked into the reasoning.
    ///
    /// Both expectations are taken from the NATIVE parser on the same shapes, not
    /// authored by hand — including the separator space in `"thinking "`, which the
    /// native scan keeps because it cuts the body at the `to=`. Guided absorbed that
    /// byte, and a one-byte disagreement on identical input is still a parity failure.
    #[test]
    fn header_resolution_tracks_turn_scope_not_a_single_flag() {
        let call = r#"[{"name":"get_weather","arguments":{"city":"Paris"}}]"#;
        let weather = || UnifiedEvent::ToolCall {
            name: "get_weather".into(),
            arguments: serde_json::json!({"city": "Paris"}),
        };
        for (label, input, want) in [
            (
                "a payload routes the turn, so a later bare header is a quote",
                format!("{call}I mean to=self<|message|>literal"),
                vec![
                    weather(),
                    UnifiedEvent::Text {
                        text: "I mean to=selfliteral".into(),
                    },
                ],
            ),
            (
                "a bare tool header inside an open thought still recovers",
                format!(" to=self<|message|>thinking to=get_weather<|message|>{call}"),
                // `"thinking "` WITH the separator space, byte-for-byte what the native
                // scan emits: it cuts the body at the `to=`, so that space is the last
                // byte of the thought. Guided absorbed it and the two paths disagreed
                // by one byte, which is a parity failure however small it looks.
                vec![
                    UnifiedEvent::Reasoning {
                        text: "thinking ".into(),
                    },
                    weather(),
                ],
            ),
        ] {
            let drive = |chunks: Vec<&str>| {
                let mut parser = muse_glimmer_unified(&tools());
                parser
                    .initialize_request(UnifiedParserInit {
                        prompt_token_ids: Vec::new(),
                        starting_state: UnifiedParserStartingState::None,
                        tool_output_mode: UnifiedToolOutputMode::GuidedJson { named_tool: None },
                        invalid_guided_payload: InvalidGuidedPayloadPolicy::RecoverAsText,
                    })
                    .expect("guided init must be accepted");
                let mut deltas = Vec::new();
                for c in chunks {
                    deltas.extend(parser.push(c).expect("push"));
                }
                deltas.extend(parser.finish().expect("finish"));
                assemble(&deltas)
            };
            assert_eq!(drive(vec![&input]), want, "{label}: whole input");
            let per_char: Vec<String> = input.chars().map(|c| c.to_string()).collect();
            assert_eq!(
                drive(per_char.iter().map(String::as_str).collect()),
                want,
                "{label}: one character at a time"
            );
            for at in 1..input.len() {
                if !input.is_char_boundary(at) {
                    continue;
                }
                assert_eq!(
                    drive(vec![&input[..at], &input[at..]]),
                    want,
                    "{label}: split at byte {at}"
                );
            }
        }
    }

    /// A bare header the model QUOTED inside its visible answer is words, not a
    /// channel switch.
    ///
    /// This family resolves an unframed `to=…<|message|>` header at turn start, when
    /// the prompt has consumed `<|start|>assistant`. Once the turn has been routed,
    /// the identical bytes are something the model wrote — and the native scan demotes
    /// them, keeping the `to=…` words visible and stripping only the marker. The
    /// guided hooks were stateless, so they promoted the quote: a `to=self` quote
    /// split one answer into an answer plus a THOUGHT, and a quoted tool recipient
    /// silently deleted the words from the answer.
    ///
    /// Both paths now consult one `resolve_header_latched`, so the two request modes
    /// cannot disagree about which bytes are structural.
    #[test]
    fn a_quoted_bare_header_inside_the_answer_stays_visible() {
        let call = r#"[{"name":"get_weather","arguments":{"city":"Paris"}}]"#;
        for (label, quoted) in [
            ("a quoted reasoning recipient", "to=self"),
            ("a quoted tool recipient", "to=get_weather"),
        ] {
            let input = format!(
                "<|start|>assistant to=user<|message|>I mean {quoted}<|message|>literal<|eom|>{call}"
            );
            let want = vec![
                UnifiedEvent::Text {
                    text: format!("I mean {quoted}literal"),
                },
                UnifiedEvent::ToolCall {
                    name: "get_weather".into(),
                    arguments: serde_json::json!({"city": "Paris"}),
                },
            ];
            let drive = |chunks: Vec<&str>| {
                let mut parser = muse_glimmer_unified(&tools());
                parser
                    .initialize_request(UnifiedParserInit {
                        prompt_token_ids: Vec::new(),
                        starting_state: UnifiedParserStartingState::None,
                        tool_output_mode: UnifiedToolOutputMode::GuidedJson { named_tool: None },
                        invalid_guided_payload: InvalidGuidedPayloadPolicy::RecoverAsText,
                    })
                    .expect("guided init must be accepted");
                let mut deltas = Vec::new();
                for c in chunks {
                    deltas.extend(parser.push(c).expect("push"));
                }
                deltas.extend(parser.finish().expect("finish"));
                assemble(&deltas)
            };
            assert_eq!(drive(vec![&input]), want, "{label}: whole input");
            let per_char: Vec<String> = input.chars().map(|c| c.to_string()).collect();
            assert_eq!(
                drive(per_char.iter().map(String::as_str).collect()),
                want,
                "{label}: one character at a time"
            );
            for at in 1..input.len() {
                if !input.is_char_boundary(at) {
                    continue;
                }
                assert_eq!(
                    drive(vec![&input[..at], &input[at..]]),
                    want,
                    "{label}: split at byte {at}"
                );
            }
        }
    }

    /// A thought whose terminator never arrived, running into the guided payload
    /// through the family's own tool WRAPPER rather than straight into the JSON.
    ///
    /// The sibling of the direct case, and the one the first fix missed: recovery
    /// tested for JSON IMMEDIATELY after the routed header, so
    /// `to=NAME<|message|>[{…}]` recovered while
    /// `to=NAME<|message|><atem:function_calls>[{…}]` still emitted the payload as
    /// REASONING and dispatched nothing. qwen3 dropped the call the same way on
    /// `<think>a<tool_call>[{…}]`, which has no routed header at all — so the rule
    /// lives in the shared owner and both families run the one implementation.
    ///
    /// The narrated contrast is asserted beside it: the same markup with PROSE behind
    /// it is something the model wrote while thinking, and the span stays open.
    #[test]
    fn an_unterminated_thought_recovers_through_a_tool_wrapper() {
        let call = r#"[{"name":"get_weather","arguments":{"city":"Paris"}}]"#;
        let weather = || UnifiedEvent::ToolCall {
            name: "get_weather".into(),
            arguments: serde_json::json!({"city": "Paris"}),
        };
        let reasoning = |text: &str| UnifiedEvent::Reasoning { text: text.into() };
        for (label, input, want) in [
            (
                "wrapper between the header and the payload still recovers",
                format!(
                    "<|start|>assistant to=self<|message|>thinking\
                     <|start|>assistant to=get_weather<|message|>\
                     <atem:function_calls>{call}</atem:function_calls><|eom|>"
                ),
                vec![reasoning("thinking"), weather()],
            ),
            (
                "the same wrapper with prose behind it is narration",
                format!(
                    "<|start|>assistant to=self<|message|>thinking about \
                     <atem:function_calls> and more<|eom|>{call}"
                ),
                vec![reasoning("thinking about  and more"), weather()],
            ),
        ] {
            let drive = |chunks: Vec<&str>| {
                let mut parser = muse_glimmer_unified(&tools());
                parser
                    .initialize_request(UnifiedParserInit {
                        prompt_token_ids: Vec::new(),
                        starting_state: UnifiedParserStartingState::None,
                        tool_output_mode: UnifiedToolOutputMode::GuidedJson { named_tool: None },
                        invalid_guided_payload: InvalidGuidedPayloadPolicy::RecoverAsText,
                    })
                    .expect("guided init must be accepted");
                let mut deltas = Vec::new();
                for c in chunks {
                    deltas.extend(parser.push(c).expect("push"));
                }
                deltas.extend(parser.finish().expect("finish"));
                assemble(&deltas)
            };
            assert_eq!(drive(vec![&input]), want, "{label}: whole input");
            for at in 1..input.len() {
                if !input.is_char_boundary(at) {
                    continue;
                }
                assert_eq!(
                    drive(vec![&input[..at], &input[at..]]),
                    want,
                    "{label}: split at byte {at}"
                );
            }
        }
    }

    /// Guided framing cases an independent review found after the first round, each
    /// one measured at every split point rather than only whole and per-character.
    ///
    /// - a bare `<|message|>` the model writes mid-thought is a stray marker, not a
    ///   recipient-less content header; reading it as one ENDED the thought and sent
    ///   the rest of the reasoning to the user as the answer;
    /// - an invoke wrapper whose opener was stripped ahead of a guided payload left
    ///   its CLOSER trailing behind the call as visible text, on both families;
    /// - a tool-routed header that leads straight into the guided payload is the
    ///   missing-terminator recovery point, not narration — stripping it left the
    ///   payload inside the thought, so the call was never made and the model's
    ///   private reasoning carried the raw JSON.
    #[test]
    fn guided_framing_survives_every_split_point() {
        let call = r#"[{"name":"get_weather","arguments":{"city":"Paris"}}]"#;
        let weather = || UnifiedEvent::ToolCall {
            name: "get_weather".into(),
            arguments: serde_json::json!({"city": "Paris"}),
        };
        let reasoning = |text: &str| UnifiedEvent::Reasoning { text: text.into() };

        for (label, input, want) in [
            (
                "a bare message marker mid-thought does not end the thought",
                format!(
                    "<|start|>assistant to=self<|message|>thinking<|message|>still thinking<|eom|>{call}"
                ),
                vec![reasoning("thinkingstill thinking"), weather()],
            ),
            (
                "an invoke closer behind the payload is not visible text",
                format!("<atem:invoke name=\"get_weather\">{call}</atem:invoke>"),
                vec![weather()],
            ),
            (
                "a tool header leading into the payload recovers the missing terminator",
                format!(
                    "<|start|>assistant to=self<|message|>thinking\
                     <|start|>assistant to=get_weather<|message|>{call}<|eom|>"
                ),
                vec![reasoning("thinking"), weather()],
            ),
            (
                // The contrast case for the one above: the same header shape NARRATED,
                // with no payload behind it, stays stripped and the thought stays open.
                "a narrated tool header with no payload behind it stays narration",
                format!(
                    "<|start|>assistant to=self<|message|>I will call \
                     <|start|>assistant to=get_weather<|message|> soon<|eom|>{call}"
                ),
                vec![reasoning("I will call  soon"), weather()],
            ),
        ] {
            let drive = |chunks: Vec<&str>| {
                let mut parser = muse_glimmer_unified(&tools());
                parser
                    .initialize_request(UnifiedParserInit {
                        prompt_token_ids: Vec::new(),
                        starting_state: UnifiedParserStartingState::None,
                        tool_output_mode: UnifiedToolOutputMode::GuidedJson { named_tool: None },
                        invalid_guided_payload: InvalidGuidedPayloadPolicy::RecoverAsText,
                    })
                    .expect("guided init must be accepted");
                let mut deltas = Vec::new();
                for c in chunks {
                    deltas.extend(parser.push(c).expect("push"));
                }
                deltas.extend(parser.finish().expect("finish"));
                assemble(&deltas)
            };

            assert_eq!(drive(vec![&input]), want, "{label}: whole input");
            // EVERY valid split point, not a sample. A boundary that lands one byte
            // inside a header is exactly where the holdback rules are load-bearing, and
            // picking a few offsets by hand is how the first version passed while a
            // per-character drive still leaked.
            for at in 1..input.len() {
                if !input.is_char_boundary(at) {
                    continue;
                }
                assert_eq!(
                    drive(vec![&input[..at], &input[at..]]),
                    want,
                    "{label}: split at byte {at}"
                );
            }
        }
    }

    /// Guided mode must read the same bytes the same way the native path does.
    ///
    /// Three defects an independent review found in the first version of the guided
    /// wiring, each one a case where the two paths disagreed about identical input:
    ///
    /// - a `to=user` header inside a thought was stripped as markup instead of
    ///   ENDING the thought, so the model's visible answer came out as its private
    ///   thinking (and, with a tool recipient, the payload behind it was never read);
    /// - a native invoke whose ARGUMENT VALUE opened with `{` had its closer hidden
    ///   behind the payload bound, so only its header was stripped and the parameter
    ///   body went to the user as text with no call dispatched;
    /// - a header whose role word is not `assistant` resolves from its `to=` run, and
    ///   the `<|start|>` sitting in front of it was emitted verbatim.
    ///
    /// Held as guided-equals-native rather than as literal expectations, so the pin
    /// cannot rot into asserting whatever the guided path happens to do.
    #[test]
    fn guided_reads_channel_framing_the_way_the_native_path_does() {
        let guided = |input: &str, split: bool| {
            let mut parser = muse_glimmer_unified(&tools());
            parser
                .initialize_request(UnifiedParserInit {
                    prompt_token_ids: Vec::new(),
                    starting_state: UnifiedParserStartingState::None,
                    tool_output_mode: UnifiedToolOutputMode::GuidedJson { named_tool: None },
                    invalid_guided_payload: InvalidGuidedPayloadPolicy::RecoverAsText,
                })
                .expect("guided init must be accepted");
            let mut deltas = Vec::new();
            if split {
                for ch in input.chars() {
                    deltas.extend(parser.push(&ch.to_string()).expect("push"));
                }
            } else {
                deltas.extend(parser.push(input).expect("push"));
            }
            deltas.extend(parser.finish().expect("finish"));
            assemble(&deltas)
        };
        let native = |input: &str| {
            let mut parser = muse_glimmer_unified(&tools());
            let mut deltas = parser.push(input).expect("push");
            deltas.extend(parser.finish().expect("finish"));
            assemble(&deltas)
        };

        for (label, input) in [
            (
                "a switch to the visible channel ends the thought",
                "<|start|>assistant to=self<|message|>thinking\
                 <|start|>assistant to=user<|message|>answer",
            ),
            (
                "a role word the grammar does not know leaks no marker",
                "<|start|>wrong-role to=self<|message|>secret<|eom|>",
            ),
        ] {
            assert_eq!(guided(input, false), native(input), "{label}: whole input");
            assert_eq!(
                guided(input, true),
                native(input),
                "{label}: one character at a time"
            );
        }

        // A complete native invoke under guided decoding emits NOTHING — the mode
        // promised bare JSON and this turn carried none, so there is no call to make
        // and no markup to show. The brace inside the argument value must not be
        // mistaken for the payload.
        let native_only = concat!(
            "<|start|>assistant to=get_weather<|message|><atem:function_calls>",
            "<atem:invoke name=\"get_weather\"><atem:parameter name=\"city\">{\"x\":1}</atem:parameter>",
            "</atem:invoke></atem:function_calls><|eom|>"
        );
        assert_eq!(
            guided(native_only, false),
            vec![],
            "a native invoke with a brace-opening argument leaked under guided decoding"
        );
    }

    /// Guided tool output, on the family that had none.
    ///
    /// The guided reader used to be refused outright here, on the ground that muse has
    /// no reasoning marker PAIR. It has a reasoning CHANNEL — routed by recipient — and
    /// conflating the two skipped 37 of the 81 unified conformance scenarios for this
    /// family.
    ///
    /// Each shape is driven twice, whole and one character at a time, and the two must
    /// agree. That is the assertion that matters, not the literal events: the first
    /// working version of this parsed every shape correctly whole and, per character,
    /// released `assistant` from the middle of `<|start|>assistant to=self<|message|>`
    /// as the model's visible answer — then could no longer see the `<|start|>` it
    /// needed to look back at, so the thought opened at the wrong offset. Same bytes,
    /// two answers, decided by where the chunk boundaries fell (`I6`).
    #[test]
    fn guided_payloads_parse_the_same_whole_and_split() {
        let one_call = r#"[{"name": "get_weather", "arguments": {"city": "Paris"}}]"#;
        let call = |args: &str| UnifiedEvent::ToolCall {
            name: "get_weather".into(),
            arguments: serde_json::from_str(args).expect("golden arguments are JSON"),
        };
        for (label, named, input, want) in [
            (
                "a named choice sends the argument object alone",
                Some("get_weather"),
                r#"{"city": "Paris"}"#.to_string(),
                vec![call(r#"{"city": "Paris"}"#)],
            ),
            (
                "a required choice sends an array of envelopes",
                None,
                one_call.to_string(),
                vec![call(r#"{"city": "Paris"}"#)],
            ),
            (
                "a framed thought precedes the payload",
                None,
                format!("<|start|>assistant to=self<|message|>thinking<|eom|>{one_call}"),
                vec![
                    UnifiedEvent::Reasoning {
                        text: "thinking".into(),
                    },
                    call(r#"{"city": "Paris"}"#),
                ],
            ),
            (
                // The form the prompt leaves when it has already consumed
                // `<|start|>assistant`. A fixed opener string would not match it.
                "a BARE thought header precedes the payload",
                None,
                format!(" to=self<|message|>thinking<|eom|>{one_call}"),
                vec![
                    UnifiedEvent::Reasoning {
                        text: "thinking".into(),
                    },
                    call(r#"{"city": "Paris"}"#),
                ],
            ),
            (
                // Guided decoding constrains the payload to bare JSON, so ATEM around
                // it is markup with nothing behind it. Left in, it enters the JSON
                // buffer, breaks the parse and costs the call.
                "native ATEM markup brackets the payload",
                None,
                format!("<atem:function_calls>{one_call}</atem:function_calls>"),
                vec![call(r#"{"city": "Paris"}"#)],
            ),
        ] {
            let drive = |chunks: Vec<String>| {
                let mut parser = muse_glimmer_unified(&tools());
                parser
                    .initialize_request(UnifiedParserInit {
                        prompt_token_ids: Vec::new(),
                        starting_state: UnifiedParserStartingState::None,
                        tool_output_mode: UnifiedToolOutputMode::GuidedJson {
                            named_tool: named.map(str::to_string),
                        },
                        invalid_guided_payload: Default::default(),
                    })
                    .expect("guided init must be accepted");
                let mut deltas = Vec::new();
                for chunk in chunks {
                    deltas.extend(parser.push(&chunk).expect("push"));
                }
                deltas.extend(parser.finish().expect("finish"));
                assemble(&deltas)
            };
            let whole = drive(vec![input.clone()]);
            let split = drive(input.chars().map(|c| c.to_string()).collect());
            assert_eq!(whole, want, "{label}: whole input");
            assert_eq!(split, want, "{label}: one character at a time");
        }
    }

    #[test]
    fn a_reused_parser_reads_the_next_turn_the_way_a_fresh_one_does() {
        // `finish` cleared the buffer and the state but left every PER-TURN latch set,
        // so a second turn on the same instance was read through the first turn's tail.
        // The worst of it is not cosmetic: `allow_bare_header` stays false, the turn's
        // header-less FIRST message (the prompt consumed `<|start|>assistant`) no longer
        // resolves, and the opening TOOL CALL is lost while its raw ATEM goes to the
        // client as content — output the client must never see.
        //
        // The v1 reasoning parser resets the same latches at its own
        // `finish_reasoning_stream` and says why; this is the port catching up. Held as
        // fresh-equals-reused rather than as literal expectations, so the guard cannot
        // rot into asserting whatever the scanner happens to do.
        // `reset` between turns is the documented reuse path — the trait says one
        // instance parses exactly one choice of one request, and the shared adapter
        // rejects a push after `finish` for every family rather than for none. What
        // this pin is about survives that: the latches below are cleared by the
        // scanner's own `take_stream_state`, which both `finish` and `reset` run, so a
        // latch that leaked would still show up as reused-diverges-from-fresh.
        fn turn(parser: &mut Box<dyn UnifiedParser>, text: &str) -> Vec<UnifiedParserEvent> {
            let mut deltas = parser.push(text).expect("push");
            deltas.extend(parser.finish().expect("finish"));
            parser.reset();
            deltas
        }
        let reasoning_turn = " to=self<|message|>one<|eom|>";
        for second in [
            // The header-less first message of a turn, in each channel it can open.
            tool_channel("get_weather", "city", "Paris")
                .strip_prefix("<|start|>assistant")
                .expect("tool_channel is framed"),
            " to=self<|message|>two<|eom|>",
            " to=user<|message|>hi<|eot|>",
        ] {
            let mut reused = muse_glimmer_unified(&tools());
            turn(&mut reused, reasoning_turn);
            let got = turn(&mut reused, second);
            let want = turn(&mut muse_glimmer_unified(&tools()), second);
            // Raw deltas, not assembled events: `tool_index` is the field a stale
            // counter moves, and `assemble` keys calls by it.
            assert_eq!(got, want, "reused parser diverged on {second:?}");
        }
    }

    #[test]
    fn an_empty_thought_adds_no_separator() {
        // The adjacency newline used to be pushed when the `to=self` header resolved,
        // before the body produced anything, so a thought that turned out EMPTY still
        // added visible whitespace to the previous one. The separator is now OWED at the
        // header and paid by the first bytes that actually arrive.
        //
        // Deliberately ahead of both engines: SGLang appends "\n" on the header when
        // `_saw_reasoning_block` is set, before the body is known, and vLLM's batch
        // `_COLLAPSE_RE` rewrites the inter-block gap so `extract_reasoning` returns
        // "a\n" for this input. (vLLM's STREAM path returns "a", disagreeing with its own
        // batch path.) Expect a conformance divergence until the engines follow.
        let empty_second = concat!(
            " to=self<|message|>a<|eom|>",
            "<|start|>assistant to=self<|message|><|eom|>"
        );
        assert_eq!(events(&[empty_second]), vec![reasoning("a")]);
        assert_eq!(batch(empty_second), vec![reasoning("a")]);

        // The join itself must survive: two NON-empty adjacent thoughts still join, and
        // an empty block between two of them is transparent rather than separator-eating.
        let two_adjacent = concat!(
            " to=self<|message|>a<|eom|>",
            "<|start|>assistant to=self<|message|>b<|eom|>"
        );
        let empty_between = concat!(
            " to=self<|message|>a<|eom|>",
            "<|start|>assistant to=self<|message|><|eom|>",
            "<|start|>assistant to=self<|message|>b<|eom|>"
        );
        assert_eq!(events(&[two_adjacent]), vec![reasoning("a\nb")]);
        assert_eq!(events(&[empty_between]), vec![reasoning("a\nb")]);
    }

    #[test]
    fn reset_hands_back_the_held_bytes_and_restarts_the_stream() {
        // `CUSTOM_PARSERS.md` states the override as MANDATORY for a parser that holds
        // bytes back, and muse holds on every path. The inherited default returned "" and
        // cleared nothing, so a caller on the documented recovery path lost the held
        // header AND resumed on a used `next_index` — the next stream's first call filed
        // under an index the abandoned one had already dispatched. Held as
        // fresh-equals-recovered, matching the reuse pins above.
        let mut parser = muse_glimmer_unified(&tools());
        parser.push(&tool_channel("f", "x", "1")).expect("push");
        // Idle holds from `<|start|>` on, so the whole partial header stays buffered.
        parser.push("<|start|>assist").expect("push");
        assert_eq!(
            parser.reset(),
            "<|start|>assist",
            "reset must hand back the held bytes, not the default empty string"
        );
        assert_eq!(parser.reset(), "", "a reset parser holds nothing");

        let second = tool_channel("g", "y", "2");
        let mut got = parser.push(&second).expect("push");
        got.extend(parser.finish().expect("finish"));
        let mut fresh = muse_glimmer_unified(&tools());
        let mut want = fresh.push(&second).expect("push");
        want.extend(fresh.finish().expect("finish"));
        assert_eq!(got, want, "a reset parser diverged from a fresh one");
    }

    #[test]
    fn a_reused_parser_restarts_tool_indices_after_a_call_turn() {
        // The reuse pin above opens turn one with REASONING, which never moves
        // `next_index`, so its `tool_index` assertions cannot catch the counter reset
        // in `flush`. Open turn one with a CALL instead: `next_index` reaches 1, and
        // only the reset returns turn two's first call to index 0 — the same index a
        // fresh parser gives it. `assemble` keys arguments by `tool_index`, so a stale
        // counter here would file the second turn's call under a slot the client's
        // first call already owns. Held as fresh-equals-reused for the same reason the
        // pin above is.
        let first = tool_channel("f", "x", "1");
        let second = tool_channel("g", "y", "2");
        let mut reused = muse_glimmer_unified(&tools());
        let mut warmup = reused.push(&first).expect("push");
        warmup.extend(reused.finish().expect("finish"));
        reused.reset();
        let mut got = reused.push(&second).expect("push");
        got.extend(reused.finish().expect("finish"));
        let mut fresh = muse_glimmer_unified(&tools());
        let mut want = fresh.push(&second).expect("push");
        want.extend(fresh.finish().expect("finish"));
        assert_eq!(got, want, "reused turn diverged after a call turn");
        let indices: Vec<usize> = got
            .iter()
            .filter_map(|d| match d {
                UnifiedParserEvent::ToolCall(c) => Some(c.tool_index),
                _ => None,
            })
            .collect();
        assert_eq!(indices, vec![0], "the second turn's call is not index 0");
    }

    #[test]
    fn a_header_the_run_seam_respells_stays_content_here() {
        // The seam costs v1 more than text. There the reasoning parser hands its
        // unwrapped answer to a SECOND parser, which reads the `<|message|>` the two
        // runs spell between them as a bare turn-start header and dispatches the
        // quoted markup as a live call (pinned in the v1 reasoning parser as
        // `a_header_the_run_seam_respells_does_become_a_call`).
        //
        // One machine routes here, and it never re-reads its own content, so the
        // respelled marker is inert: the same bytes stay one Text event with no call
        // at any chunking. That is the property the split path cannot have, so pin
        // it rather than leave it to the ledger entry for the text leak above.
        let spliced = concat!(
            "<|start|>assistant to=user<|message|>Quote: to=run<|mes<|eom|>sage|>",
            "<atem:invoke name=\"run\"><atem:parameter name=\"q\">pwn</atem:parameter>",
            "</atem:invoke><|eom|>"
        );
        let want = vec![text(concat!(
            "Quote: to=run<|message|><atem:invoke name=\"run\">",
            "<atem:parameter name=\"q\">pwn</atem:parameter></atem:invoke>"
        ))];
        assert_eq!(batch(spliced), want);
        let chars: Vec<String> = spliced.chars().map(|c| c.to_string()).collect();
        assert_eq!(
            events(&chars.iter().map(String::as_str).collect::<Vec<_>>()),
            want
        );
    }

    // ── Byte-split sweep ──────────────────────────────────────────────────────

    #[test]
    fn streaming_matches_batch_at_every_split_boundary() {
        let self_tool_self_user = format!(
            "{}{}{}{}",
            channel("self", "A"),
            tool_channel("f", "x", "1"),
            channel("self", "B"),
            "<|start|>assistant to=user<|message|>done<|eot|>"
        );
        let chained_tools = format!(
            "{}{}",
            tool_channel("f", "x", "1"),
            tool_channel("g", "y", "2")
        );
        let tool_cut_by_header = format!(
            "{}{}",
            " to=f<|message|><atem:function_calls>\n<atem:invoke name=\"f\">\n</atem:invoke>",
            "<|start|>assistant to=user<|message|>kept<|eot|>"
        );
        let adjacent_thoughts = format!("{}{}", channel("self", "a"), channel("self", "b"));
        // Two invokes in ONE tool channel: both drain inside a single `tool_body_limit`,
        // so their incremental emit (indices 0 then 1) must stay split-invariant.
        let two_invokes_one_channel = concat!(
            " to=f<|message|><atem:invoke name=\"f\"><atem:parameter name=\"x\">1</atem:parameter></atem:invoke>",
            "<atem:invoke name=\"g\"><atem:parameter name=\"y\">2</atem:parameter></atem:invoke><|eom|>"
        );
        let cases = [
            " to=self<|message|>The user asks 2+2. It is 4.<|eom|><|start|>assistant to=user<|message|>2 + 2 = 4.<|eot|>",
            concat!(
                " to=self<|message|>Need the weather.<|eom|>",
                "<|start|>assistant to=get_weather<|message|><atem:function_calls>\n",
                "<atem:invoke name=\"get_weather\">\n",
                "<atem:parameter name=\"city\">Paris</atem:parameter>\n",
                "</atem:invoke>\n</atem:function_calls><|eom|>"
            ),
            " to=self<|message|>a<|eom|><|start|>assistant to=self<|message|>b<|eom|>",
            " to=user<|message|>only answer<|eot|>",
            "<|message|>bare content<|eot|>",
            " to=self<|message|>thinking to=f<|message|><atem:invoke name=\"f\"></atem:invoke><|eom|>",
            "plain answer with no framing at all",
            " to=self<|message|>unterminated reasoning tail",
            " to=user<|message|>Example: to=g<|message|><atem:invoke name=\"g\"></atem:invoke><|eot|>",
            " to=f<|message|><atem:invoke name=\"f\"></atem:invoke><|start|>assistant to=user<|message|>kept<|eot|>",
            "my assistant  to=user<|message|>x<|eot|>",
            " to=self<|message|>weird potato=f<|message|><atem:invoke name=\"f\"></atem:invoke><|eom|>",
            // The same mid-word `to=` at IDLE, where resolution is unanchored.
            "poto=f<|message|>x<|eom|>",
            // Latch spent by a closed answer: `to=` prose between messages.
            " to=user<|message|>x<|eot|>abc to=f<|message|>hi<|eot|>",
            // An empty reasoning body whose first bytes are the recovered header.
            " to=self<|message|>to=f<|message|><atem:invoke name=\"f\"></atem:invoke><|eom|>",
            // A framed cut immediately followed by a bare-looking header.
            " to=self<|message|>a<|start|> to=f<|message|><atem:invoke name=\"f\"></atem:invoke><|eom|>",
            // A header lookalike inside a parameter value.
            " to=f<|message|><atem:invoke name=\"f\"><atem:parameter name=\"q\">say to=g<|message|> please</atem:parameter></atem:invoke><|eom|>",
            &self_tool_self_user,
            &chained_tools,
            &tool_cut_by_header,
            &adjacent_thoughts,
            two_invokes_one_channel,
        ];
        for input in cases {
            let expected = batch(input);
            for split in input
                .char_indices()
                .map(|(i, _)| i)
                .chain(std::iter::once(input.len()))
            {
                assert_eq!(
                    events(&[&input[..split], &input[split..]]),
                    expected,
                    "batch/stream mismatch at byte {split} for {input:?}"
                );
            }
        }
    }
}
