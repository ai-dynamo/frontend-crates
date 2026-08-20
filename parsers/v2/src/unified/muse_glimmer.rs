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
//! dynamic `to=self<|message|>` header opens reasoning, sharing its opener with the
//! content and tool channels, so routing is by recipient, not marker. That is why this
//! family does not run on [`crate::unified::ScannerUnified`].
//!
//! What the unified path adds is ORDER: one machine routes reasoning, content and calls
//! directly, so a thought between two calls stays between them instead of being hoisted
//! ahead of the calls the way the split path does.

use crate::tool_calling::muse_glimmer::{MuseChannelScanner, muse_scanner};
use crate::tool_calling::traits::{Result, Tool};
use crate::unified::{UnifiedParser, UnifiedParserEvent, UnifiedParserOutput};

impl UnifiedParser for MuseChannelScanner {
    fn preserve_special_tokens(&self) -> bool {
        true
    }

    /// The trait default returns an empty string and clears nothing. Muse buffers on
    /// every path — a partial marker, an open channel, the bare-header latch and a used
    /// `next_index` — so inheriting it would report an empty carry while the scanner
    /// still held all of that. A caller on the documented recovery path would drop the
    /// held bytes and resume on a counter that re-numbers its first call onto an index
    /// the abandoned stream already dispatched.
    fn reset(&mut self) -> String {
        self.reset_stream()
    }

    fn parse_into(&mut self, delta: &str, output: &mut UnifiedParserOutput) -> Result<()> {
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

    fn finish(&mut self) -> Result<UnifiedParserOutput> {
        let mut output = UnifiedParserOutput::default();
        for event in self.finish_ordered()? {
            push_event(&mut output, event);
        }
        Ok(output)
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
    Box::new(muse_scanner(tools))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::unified::{UnifiedEvent, UnifiedParserExt, assemble};

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
        assert_eq!(
            out,
            vec![reasoning(
                "weird potato=f<|message|><atem:invoke name=\"f\"></atem:invoke>"
            )]
        );
    }

    #[test]
    fn an_orphan_marker_survives_on_the_reasoning_route_but_not_the_content_one() {
        // The marker in the expected string above is not incidental, so pin it on its
        // own. Every route into visible text runs the strip, and the reasoning route
        // does not. `<|start|>`, `<|eom|>` and `<|eot|>` all CUT a body, so
        // `<|message|>` is the one marker that can reach a reader, and the same bytes
        // therefore come back stripped as an answer and intact as a thought.
        assert_eq!(
            batch(" to=self<|message|>note the <|message|> token<|eom|>"),
            vec![reasoning("note the <|message|> token")]
        );
        assert_eq!(
            batch(" to=user<|message|>note the <|message|> token<|eot|>"),
            vec![text("note the  token")]
        );

        // ADOPTED as the contract, on the engine reference this comment used to say was
        // missing. vLLM PR #51655 merged 2026-08-14, so both references now exist and
        // both KEEP the marker:
        //
        //   vLLM  `_CHANNEL_HEADER_RE` is `to=<recipient><|message|>`, so a BARE
        //         `<|message|>` never ends a body; `_classify_bodies` and the batch
        //         `_REASONING_RE` (`to=self<|message|>(.*?)<|eom|>`) both return
        //         `a<|message|>b`.
        //   SGL   `MuseGlimmerDetector::_consume` appends the body verbatim to
        //         `reasoning_parts` with no strip.
        //
        // So stripping here would make Dynamo the ONLY implementation that strips, and
        // trade an I3 reading for a parity divergence on a family the conformance suite
        // scores against those two engines. I3 stays unmet on this route by agreement
        // with the engines, not by oversight; closing it belongs upstream first.
        // Stripping also would not make the mid-word case above clean, because the ATEM
        // markup beside the marker stays either way: that one is the priced cost of
        // `bare_header_pos` refusing an unanchored cut, not something the strip reaches.
        assert_eq!(
            batch(" to=self<|message|>a<|message|>b<|eom|>")
                .into_iter()
                .chain(batch(" to=user<|message|>a<|message|>b<|eot|>"))
                .collect::<Vec<_>>(),
            vec![reasoning("a<|message|>b"), text("ab")]
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
        fn turn(parser: &mut Box<dyn UnifiedParser>, text: &str) -> Vec<UnifiedParserEvent> {
            let mut deltas = parser.push(text).expect("push");
            deltas.extend(parser.finish().expect("finish"));
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
    fn an_empty_thought_still_carries_the_adjacency_newline() {
        // The separator is emitted when the `to=self` header resolves, so a thought
        // that turns out EMPTY still contributes it. That reads like a bug against the
        // empty-block contract, and it is what BOTH references do:
        //
        //   SGL   `_consume` appends "\n" on the header when `_saw_reasoning_block`,
        //         before the body is known.
        //   vLLM  batch `_COLLAPSE_RE.sub("\n", ..)` rewrites the inter-block gap, so
        //         `extract_reasoning` returns "a\n" for exactly this input.
        //
        // vLLM's STREAM path returns "a" instead, disagreeing with its own batch path,
        // which is why this is pinned against the batch parsers — the comparison this
        // family's join was written to match.
        let empty_second = concat!(
            " to=self<|message|>a<|eom|>",
            "<|start|>assistant to=self<|message|><|eom|>"
        );
        assert_eq!(events(&[empty_second]), vec![reasoning("a\n")]);
        assert_eq!(batch(empty_second), vec![reasoning("a\n")]);

        // The join this rides on: two NON-empty adjacent thoughts read as one run.
        let two_adjacent = concat!(
            " to=self<|message|>a<|eom|>",
            "<|start|>assistant to=self<|message|>b<|eom|>"
        );
        assert_eq!(events(&[two_adjacent]), vec![reasoning("a\nb")]);
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
