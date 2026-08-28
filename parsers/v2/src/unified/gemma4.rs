// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unified parser for the Gemma 4 grammar: the Gemma 4 tool grammar plus the
//! `<|channel>` reasoning channel, in ONE state machine.
//!
//! ```text
//! reasoning:  <|channel>thought\n … <channel|>
//! tool call:  <|tool_call>call:NAME{key:<|"|>value<|"|>}<tool_call|>
//! everything else: visible content
//! ```
//!
//! This file is only the grammar wiring. The scan core is the same
//! [`crate::tool_calling::scan::WrappedBlockScanner`] the tool-only
//! `Gemma4ToolStreamParser` runs on — block open/close, bare-`call:` recovery,
//! orphan-close stripping, chunk-boundary holdback and EOF recovery stay a single
//! implementation, so the unified path cannot quietly regress the tool handling
//! the tool-only suite already pins. Value typing likewise delegates to the
//! shared v1 batch parser, so a streamed argument object matches the batch one
//! exactly. The `UnifiedParser` impl itself is generic
//! ([`crate::unified::ScannerUnified`]).
//!
//! Gemma 4 was the hardest family to move because its END MARKER IS DATA: a
//! literal `<tool_call|>` inside a `<|"|>`-delimited string value is an argument
//! value, so the plain `find` every other wrapped family uses would cut the value
//! there (`I7`, case `7.b`). A model-owned incremental invoke driver resolves
//! that lexical ambiguity without putting Gemma state in the shared scanner;
//! see `tool_calling/gemma4.rs`.
//!
//! The reasoning side needed the other extension. Gemma 4's opener is a marker
//! PLUS a `thought\n` role label, and the label is OPTIONAL — folding it into
//! `start` parses the corpus but makes a bare `<|channel>` open nothing and leak
//! its markup as visible text. [`ReasoningSpec::start_label`] keeps the label
//! structural and optional (policy `P4`).
//!
//! What the unified path adds is ORDER: a thought between two calls is a second
//! thought in its own position instead of being hoisted into the first. Nesting
//! is asymmetric, exactly as for qwen3 — a call inside a thought is extracted and
//! the thought splits around it (`I3`, case `12.b`), while a channel marker
//! inside a quoted argument value is data and survives byte-exact (`I7`, case
//! `12.a`).

use crate::tool_calling::gemma4::{
    REASONING_END, REASONING_START, REASONING_START_LABEL, gemma4_scanner,
};
use crate::tool_calling::scan::ReasoningSpec;
use crate::tool_calling::traits::Tool;
use crate::unified::{GuidedRouted, ScannerUnified, UnifiedParser};

/// Build the Gemma 4 unified parser for one stream.
pub(crate) fn gemma4_unified(tools: &[Tool]) -> Box<dyn UnifiedParser> {
    // `preserve_special_tokens` must match the TOOL-ONLY parser for this same
    // grammar (`Gemma4ToolStreamParser` answers `true`): gemma4's markers ARE
    // tokenizer special tokens, so a host told `false` would strip them and the
    // calls would vanish on the unified path only.
    Box::new(GuidedRouted::new(ScannerUnified::new(
        gemma4_scanner(tools).with_reasoning(ReasoningSpec {
            start: REASONING_START,
            end: REASONING_END,
            start_label: Some(REASONING_START_LABEL),
            // Gemma 4 emits its own `<|channel>`; the template does not pre-fill
            // one, so the stream starts in visible content (policy P5).
            forced_start: false,
            preserve_special_tokens: true,
        }),
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::unified::{
        InvalidGuidedPayloadPolicy, UnifiedEvent, UnifiedParserExt, UnifiedParserInit,
        UnifiedParserStartingState, UnifiedToolOutputMode, assemble,
    };

    fn weather_tools() -> Vec<Tool> {
        vec![Tool {
            name: "get_weather".to_string(),
            description: None,
            parameters: serde_json::json!({
                "type": "object",
                "properties": { "city": { "type": "string" } }
            }),
            strict: None,
        }]
    }

    fn tool(name: &str, key: &str) -> Tool {
        Tool {
            name: name.to_string(),
            description: None,
            parameters: serde_json::json!({
                "type": "object",
                "properties": { key: { "type": "string" } }
            }),
            strict: None,
        }
    }

    fn events(tools: &[Tool], chunks: &[&str]) -> Vec<UnifiedEvent> {
        let mut parser = gemma4_unified(tools);
        let mut deltas = Vec::new();
        for chunk in chunks {
            deltas.extend(parser.push(chunk).expect("push"));
        }
        deltas.extend(parser.finish().expect("finish").events);
        assemble(&deltas)
    }

    fn configured_events(
        tools: &[Tool],
        starting_state: UnifiedParserStartingState,
        tool_output_mode: UnifiedToolOutputMode,
        chunks: &[&str],
    ) -> Vec<UnifiedEvent> {
        let mut parser = gemma4_unified(tools);
        parser
            .initialize_request(UnifiedParserInit {
                starting_state,
                tool_output_mode,
                invalid_guided_payload: InvalidGuidedPayloadPolicy::RecoverAsText,
                ..UnifiedParserInit::default()
            })
            .expect("initialize");
        let mut deltas = Vec::new();
        for chunk in chunks {
            deltas.extend(parser.push(chunk).expect("push"));
        }
        deltas.extend(parser.finish().expect("finish").events);
        assemble(&deltas)
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

    #[test]
    fn reasoning_after_a_call_keeps_its_position() {
        // The defect the unified parser exists to fix: under the split, both
        // thoughts merge into one span ahead of the call.
        let out = events(
            &weather_tools(),
            &[
                "<|channel>thought\nLook it up.<channel|>",
                "<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>}<tool_call|>",
                "<|channel>thought\nNow answer.<channel|>It's 18C.",
            ],
        );
        assert_eq!(
            out,
            vec![
                reasoning("Look it up."),
                call("get_weather", serde_json::json!({"city": "Paris"})),
                reasoning("Now answer."),
                text("It's 18C."),
            ]
        );
    }

    #[test]
    fn the_thought_role_label_is_structural_but_optional() {
        // P4: `thought\n` is tokenizer grammar, so it is stripped — but a channel
        // that opens WITHOUT it keeps every byte of the thought, including a first
        // word that merely starts like the label. Folding the label into `start`
        // would leave `<|channel>` opening nothing and leaking as text.
        assert_eq!(
            events(&weather_tools(), &["<|channel>thought\nstripped<channel|>"]),
            vec![reasoning("stripped")]
        );
        assert_eq!(
            events(&weather_tools(), &["<|channel>no label here<channel|>"]),
            vec![reasoning("no label here")]
        );
        assert_eq!(
            events(&weather_tools(), &["<|channel>thoughtful musing<channel|>"]),
            vec![reasoning("thoughtful musing")]
        );
    }

    #[test]
    fn a_label_split_across_chunks_never_leaks_and_never_eats_content() {
        // The undecided window: the buffer ends INSIDE `thought\n`, so the opener
        // cannot be classified yet and must be held rather than released as text.
        assert_eq!(
            events(
                &weather_tools(),
                &["<|channel>", "thou", "ght\nheld<channel|>"]
            ),
            vec![reasoning("held")]
        );
        assert_eq!(
            events(
                &weather_tools(),
                &["<|chan", "nel>thoug", "htful musing<channel|>"]
            ),
            vec![reasoning("thoughtful musing")]
        );
    }

    #[test]
    fn markers_split_across_chunks_never_leak() {
        // Every marker is cut in half at a chunk boundary, including the `<|"|>`
        // string delimiter and the asymmetric end marker.
        let out = events(
            &weather_tools(),
            &[
                "<|chan",
                "nel>thought\na<chann",
                "el|>go: <|tool",
                "_call>call:get_weather{city:<|",
                "\"|>Par",
                "is<|\"|",
                ">}<tool_cal",
                "l|>done",
            ],
        );
        assert_eq!(
            out,
            vec![
                reasoning("a"),
                text("go: "),
                call("get_weather", serde_json::json!({"city": "Paris"})),
                text("done"),
            ]
        );
    }

    #[test]
    fn unterminated_reasoning_is_promoted_at_finish() {
        // 10.d: not dropped, and not leaked as visible text.
        let out = events(
            &weather_tools(),
            &["<|channel>thought\nthinking but stream ends"],
        );
        assert_eq!(out, vec![reasoning("thinking but stream ends")]);
    }

    #[test]
    fn an_argument_value_may_contain_the_block_close_marker() {
        // 7.b / I7 — the case that made gemma4 Tier C. `<tool_call|>` inside a
        // `<|"|>` string is DATA, so a plain `find` of the end marker would cut
        // the value here. The invoke scan resolves the real end instead.
        let out = events(
            &[tool("run", "cmd")],
            &["<|tool_call>call:run{cmd:<|\"|>git log }<tool_call|> --oneline<|\"|>}<tool_call|>"],
        );
        assert_eq!(
            out,
            vec![call(
                "run",
                serde_json::json!({"cmd": "git log }<tool_call|> --oneline"})
            )]
        );
    }

    #[test]
    fn reasoning_marker_inside_an_argument_is_data() {
        // 12.a / I7: once a tool block is open, `<|channel>` is a value, not a
        // control token, and must survive byte-exact — role label and all.
        let out = events(
            &[tool("log", "note")],
            &[
                "<|tool_call>call:log{note:<|\"|><|channel>thought\nreconsider<channel|><|\"|>}<tool_call|>",
            ],
        );
        assert_eq!(
            out,
            vec![call(
                "log",
                serde_json::json!({"note": "<|channel>thought\nreconsider<channel|>"})
            )]
        );
    }

    #[test]
    fn reasoning_marker_inside_an_argument_is_data_at_every_split() {
        // 12.a / I7 at every chunk boundary: once a tool block is open, a
        // `<|channel>` marker inside the argument string is data, byte-exact,
        // regardless of where the stream happens to be cut.
        let input = "<|tool_call>call:log{note:<|\"|><|channel>thought\nreconsider<channel|><|\"|>}<tool_call|>";
        let expected = vec![call(
            "log",
            serde_json::json!({"note": "<|channel>thought\nreconsider<channel|>"}),
        )];
        for split in input
            .char_indices()
            .map(|(index, _)| index)
            .chain(std::iter::once(input.len()))
        {
            assert_eq!(
                events(&[tool("log", "note")], &[&input[..split], &input[split..]]),
                expected,
                "split={split}"
            );
        }
    }

    #[test]
    fn tool_call_inside_reasoning_is_extracted_and_splits_the_thought() {
        // 12.b: tool structure dominates reasoning. A call the model emits inside
        // a thought is still a real call, so it surfaces as its own event and the
        // thought splits around it. Burying it would both drop the call and leak
        // `<|tool_call>` markup into the reasoning payload (`I3`).
        let out = events(
            &weather_tools(),
            &[
                "<|channel>thought\nI should check. <|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>}<tool_call|> now answer<channel|>Done.",
            ],
        );
        assert_eq!(
            out,
            vec![
                reasoning("I should check. "),
                call("get_weather", serde_json::json!({"city": "Paris"})),
                reasoning(" now answer"),
                text("Done."),
            ]
        );
    }

    #[test]
    fn tool_call_inside_reasoning_is_extracted_at_every_split() {
        // 12.b at every chunk boundary: a call emitted inside a thought is
        // extracted as its own event and the thought splits around it,
        // regardless of where the stream happens to be cut.
        let input = "<|channel>thought\nI should check. <|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>}<tool_call|> now answer<channel|>Done.";
        let expected = vec![
            reasoning("I should check. "),
            call("get_weather", serde_json::json!({"city": "Paris"})),
            reasoning(" now answer"),
            text("Done."),
        ];
        for split in input
            .char_indices()
            .map(|(index, _)| index)
            .chain(std::iter::once(input.len()))
        {
            assert_eq!(
                events(&weather_tools(), &[&input[..split], &input[split..]]),
                expected,
                "split={split}"
            );
        }
    }

    #[test]
    fn narration_after_a_call_survives() {
        // The block IS the invoke, so closing the call must close the block. A
        // block left open would suppress every later text run as markup.
        let out = events(
            &weather_tools(),
            &["<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>}<tool_call|> Let me know."],
        );
        assert_eq!(
            out,
            vec![
                call("get_weather", serde_json::json!({"city": "Paris"})),
                text(" Let me know."),
            ]
        );
    }

    #[test]
    fn orphan_close_is_stripped_not_leaked() {
        // 5.c / I3: a `<tool_call|>` with nothing open is malformed markup.
        let out = events(&weather_tools(), &["I will check that. <tool_call|>"]);
        assert_eq!(out, vec![text("I will check that. ")]);
    }

    #[test]
    fn a_duplicate_channel_opener_inside_a_thought_is_stripped_with_its_label() {
        // I3: a second `<|channel>` while one is already open is a duplicate
        // opener, not content — and its role label goes with it, or `thought\n`
        // lands in the reasoning payload.
        let out = events(
            &weather_tools(),
            &["<|channel>thought\na<|channel>thought\nb<channel|>tail"],
        );
        assert_eq!(out, vec![reasoning("ab"), text("tail")]);
    }

    #[test]
    fn truncated_tool_call_at_eof_keeps_preceding_output() {
        // 5.a / P2: drop the unrecoverable partial call, no error, no leaked markup.
        let out = events(
            &weather_tools(),
            &[
                "<|channel>thought\nok<channel|>Checking. <|tool_call>call:get_weather{city:<|\"|>Par",
            ],
        );
        assert_eq!(out, vec![reasoning("ok"), text("Checking. ")]);
    }

    #[test]
    fn truncated_tool_call_at_eof_keeps_preceding_output_at_every_split() {
        // 5.a / P2 at every chunk boundary: the unrecoverable partial call must
        // drop cleanly and the preceding reasoning/text must survive, no matter
        // where the stream happens to be cut.
        let input =
            "<|channel>thought\nok<channel|>Checking. <|tool_call>call:get_weather{city:<|\"|>Par";
        let expected = vec![reasoning("ok"), text("Checking. ")];
        for split in input
            .char_indices()
            .map(|(index, _)| index)
            .chain(std::iter::once(input.len()))
        {
            assert_eq!(
                events(&weather_tools(), &[&input[..split], &input[split..]]),
                expected,
                "split={split}"
            );
        }
    }

    #[test]
    fn a_complete_body_missing_its_close_marker_is_recovered_at_finish() {
        // 5.b: the body balanced; only the close never streamed. v1 parity says
        // recover it rather than drop it.
        let out = events(
            &weather_tools(),
            &["<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>}"],
        );
        assert_eq!(
            out,
            vec![call("get_weather", serde_json::json!({"city": "Paris"}))]
        );
    }

    #[test]
    fn a_complete_body_missing_its_close_marker_is_recovered_at_every_split() {
        // 5.b at every chunk boundary: the body balances but the close marker
        // never streams, so `finish` must recover it regardless of split point.
        let input = "<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>}";
        let expected = vec![call("get_weather", serde_json::json!({"city": "Paris"}))];
        for split in input
            .char_indices()
            .map(|(index, _)| index)
            .chain(std::iter::once(input.len()))
        {
            assert_eq!(
                events(&weather_tools(), &[&input[..split], &input[split..]]),
                expected,
                "split={split}"
            );
        }
    }

    #[test]
    fn concatenated_calls_stay_separate_events() {
        // Gemma 4 concatenates calls with NO separator between them.
        let out = events(
            &[tool("f", "x"), tool("g", "y")],
            &[
                "<|tool_call>call:f{x:<|\"|>1<|\"|>}<tool_call|><|tool_call>call:g{y:<|\"|>2<|\"|>}<tool_call|>",
            ],
        );
        assert_eq!(
            out,
            vec![
                call("f", serde_json::json!({"x": "1"})),
                call("g", serde_json::json!({"y": "2"})),
            ]
        );
    }

    #[test]
    fn a_bare_call_without_its_opener_is_recovered_as_a_call() {
        // Gemma 4's missing-start recovery: the body is a call, not prose, so it
        // must not leak as visible text. Prose that merely contains the WORD
        // `call:` still reaches the user untouched.
        assert_eq!(
            events(
                &weather_tools(),
                &["I will check. call:get_weather{city:<|\"|>Paris<|\"|>}<tool_call|>"],
            ),
            vec![
                text("I will check. "),
                call("get_weather", serde_json::json!({"city": "Paris"})),
            ]
        );
        assert_eq!(
            events(&weather_tools(), &["I will call: you tomorrow."]),
            vec![text("I will call: you tomorrow.")]
        );
    }

    #[test]
    fn a_bare_call_without_its_opener_is_recovered_at_every_split() {
        // Gemma 4's missing-start recovery at every chunk boundary: the prose
        // prefix stays text and the bare call body recovers as a call, no matter
        // where the stream happens to be cut.
        let input = "I will check. call:get_weather{city:<|\"|>Paris<|\"|>}<tool_call|>";
        let expected = vec![
            text("I will check. "),
            call("get_weather", serde_json::json!({"city": "Paris"})),
        ];
        for split in input
            .char_indices()
            .map(|(index, _)| index)
            .chain(std::iter::once(input.len()))
        {
            assert_eq!(
                events(&weather_tools(), &[&input[..split], &input[split..]]),
                expected,
                "split={split}"
            );
        }
    }

    #[test]
    fn malformed_wrapper_recovery_is_chunk_invariant() {
        let cases = [
            (
                "<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>} trailing </tool_call>",
                Vec::new(),
            ),
            (
                "for example, call:get_weather{city:<|\"|>Paris<|\"|>}",
                vec![text("for example, ")],
            ),
        ];
        for (input, expected) in cases {
            for split in input
                .char_indices()
                .map(|(index, _)| index)
                .chain(std::iter::once(input.len()))
            {
                assert_eq!(
                    events(&weather_tools(), &[&input[..split], &input[split..]]),
                    expected,
                    "input={input:?} split={split}"
                );
            }
        }
    }

    #[test]
    fn batch_and_stream_assemble_identically() {
        // I6, at the parser level: `parse_complete` routes through the same
        // push/finish lifecycle, so parity is structural.
        let input = "<|channel>thought\na<channel|>Here you go: <|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>}<tool_call|><|channel>thought\nb<channel|>Done.";
        let batch = gemma4_unified(&weather_tools())
            .parse_complete(input)
            .expect("parse_complete");
        assert_eq!(batch, events(&weather_tools(), &[input]));
        assert_eq!(batch.len(), 5);
    }

    #[test]
    fn reasoning_prefill_classifies_leading_text_as_reasoning() {
        // 40.*: the prompt opened the channel AND wrote the role label, so the
        // stream begins mid-thought with neither marker nor label.
        let out = configured_events(
            &weather_tools(),
            UnifiedParserStartingState::Reasoning,
            UnifiedToolOutputMode::Native,
            &["hidden<chann", "el|>visible"],
        );
        assert_eq!(out, vec![reasoning("hidden"), text("visible")]);
    }

    #[test]
    fn reasoning_prefill_consumes_a_redundant_opener_with_its_label() {
        // 41.a: the backend re-emits the opener the prompt already wrote. Exactly
        // one echo is consumed — including `thought\n`, which is part of it.
        let out = configured_events(
            &weather_tools(),
            UnifiedParserStartingState::Reasoning,
            UnifiedToolOutputMode::Native,
            &[
                "<|channel>thought\nchecking weather<channel|>",
                "<|tool_call>call:get_weather{city:<|\"|>London<|\"|>}<tool_call|>",
            ],
        );
        assert_eq!(
            out,
            vec![
                reasoning("checking weather"),
                call("get_weather", serde_json::json!({"city": "London"})),
            ]
        );
    }

    #[test]
    fn response_prefill_does_not_interpret_channel_markers() {
        // 50.d — the only case where `starting_state=Response` is observable: the prompt
        // opened VISIBLE content, so this stream has no reasoning channel and the
        // channel markers are ordinary characters the user must see.
        let out = configured_events(
            &weather_tools(),
            UnifiedParserStartingState::Response,
            UnifiedToolOutputMode::Native,
            &["<|channel>thought\nliteral<channel|> then a call"],
        );
        assert_eq!(
            out,
            vec![text("<|channel>thought\nliteral<channel|> then a call")]
        );
    }

    #[test]
    fn guided_json_strips_the_channel_envelope_around_the_payload() {
        // 40.b: guided decoding is a BACKEND feature, so the payload is bare JSON
        // in every family — but the reasoning envelope around it is still the
        // family's own markers, which is all `GuidedState` needs.
        let out = configured_events(
            &weather_tools(),
            UnifiedParserStartingState::Reasoning,
            UnifiedToolOutputMode::GuidedJson { named_tool: None },
            &[
                "checking weather<channel|>[{\"name\": \"get_weather\", ",
                "\"arguments\": {\"city\": \"Paris\"}}]",
            ],
        );
        assert_eq!(
            out,
            vec![
                reasoning("checking weather"),
                call("get_weather", serde_json::json!({"city": "Paris"})),
            ]
        );
    }

    #[test]
    fn guided_json_named_choice_parses_a_bare_arguments_object() {
        // 30.a: a named choice constrains output to that tool's arguments alone.
        let out = configured_events(
            &weather_tools(),
            UnifiedParserStartingState::None,
            UnifiedToolOutputMode::GuidedJson {
                named_tool: Some("get_weather".to_string()),
            },
            &["{\"city\": ", "\"Paris\"}"],
        );
        assert_eq!(
            out,
            vec![call("get_weather", serde_json::json!({"city": "Paris"}))]
        );
    }

    #[test]
    fn guided_json_recovers_a_payload_that_is_not_a_call_as_text() {
        // 31.a / P2: no `name`, so there is nothing to dispatch. Surface the bytes
        // rather than dropping them or erroring.
        let out = configured_events(
            &weather_tools(),
            UnifiedParserStartingState::None,
            UnifiedToolOutputMode::GuidedJson { named_tool: None },
            &["{\"unexpected\": \"shape\"}"],
        );
        assert_eq!(out, vec![text("{\"unexpected\": \"shape\"}")]);
    }

    #[test]
    fn guided_native_recovery_cannot_cross_a_reasoning_closer() {
        let input = concat!(
            "<|channel>thought\ncall:broken{",
            "<channel|>",
            "<|tool_call>call:get_weather{city:<|\"|>trap<|\"|>}<tool_call|>",
            "[{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Paris\"}}]",
        );
        let expected = vec![call("get_weather", serde_json::json!({"city": "Paris"}))];
        for split in input
            .char_indices()
            .map(|(index, _)| index)
            .chain(std::iter::once(input.len()))
        {
            assert_eq!(
                configured_events(
                    &weather_tools(),
                    UnifiedParserStartingState::None,
                    UnifiedToolOutputMode::GuidedJson { named_tool: None },
                    &[&input[..split], &input[split..]],
                ),
                expected,
                "split={split}"
            );
        }
    }

    /// gemma4's opener carries a `thought\n` ROLE LABEL. It is grammar, not
    /// thought, so it must never surface as reasoning — in EITHER request mode,
    /// at ANY chunk boundary.
    ///
    /// Enumerated from the property, not from a fix: every char-boundary split
    /// of a labelled opener, crossed with native and guided. Sampling splits is
    /// what let this defect reach the corpus in the first place.
    ///
    /// Negative control: make the guided drain consume `start.len()` instead of
    /// `reasoning_opener_len(..)` and every guided row here fails with
    /// `reasoning("thought\nchecking")`.
    #[test]
    fn a_role_label_is_never_thought_in_either_request_mode() {
        let tools = weather_tools();
        let cases = [
            (
                "<|channel>thought\nchecking<channel|><|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>}<tool_call|>",
                UnifiedToolOutputMode::Native,
            ),
            (
                "<|channel>thought\nchecking<channel|>[{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}]",
                UnifiedToolOutputMode::GuidedJson { named_tool: None },
            ),
        ];
        for (input, mode) in cases {
            for split in 1..input.len() {
                if !input.is_char_boundary(split) {
                    continue;
                }
                let (head, tail) = input.split_at(split);
                let got = configured_events(
                    &tools,
                    UnifiedParserStartingState::None,
                    mode.clone(),
                    &[head, tail],
                );
                let reasoning: String = got
                    .iter()
                    .filter_map(|e| match e {
                        UnifiedEvent::Reasoning { text } => Some(text.as_str()),
                        _ => None,
                    })
                    .collect();
                assert_eq!(
                    reasoning, "checking",
                    "{mode:?} split at {split}: role label or thought text wrong; got {got:?}"
                );
            }
        }
    }
}
