// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unified parser for the Qwen3 grammar: the Qwen3-Coder tool grammar plus a
//! `<think>` reasoning channel, in ONE state machine.
//!
//! ```text
//! reasoning:  <think> … </think>
//! tool call:  <tool_call><function=NAME><parameter=KEY>VALUE</parameter></function></tool_call>
//! everything else: visible content
//! ```
//!
//! This file is only the grammar wiring. The scan core is the same
//! `WrappedBlockScanner` the tool-only
//! `Qwen3CoderToolStreamParser` runs on — block open/close, bare-`<function=>`
//! recovery, orphan-close stripping, chunk-boundary holdback and EOF drop stay a
//! single implementation, so the unified path cannot quietly regress the tool
//! handling the tool-only suite already pins. Value typing likewise delegates to
//! the shared batch XML parser, so a streamed argument object matches the batch
//! one exactly. The `UnifiedParser` impl itself is generic
//! (`ScannerUnified`).
//!
//! What the unified path adds is ORDER: `<think>` between two calls is a second
//! thought in its own position instead of being hoisted into the first.
//!
//! Nesting is asymmetric, because tool structure dominates. A tool call the model
//! emits INSIDE a thought is still a real call, so it is extracted and the thought
//! splits around it (burying it would drop the call and leak its markup into the
//! reasoning payload, `I3`). A reasoning marker inside a tool ARGUMENT is data and
//! survives byte-exact (`I7`), because the in-block scan never looks for one.

use crate::tool_calling::qwen3_coder::qwen3_scanner;
use crate::tool_calling::scan::ReasoningSpec;
use crate::tool_calling::traits::Tool;
use crate::unified::{ScannerUnified, UnifiedParser};

const REASONING_START: &str = "<think>";
const REASONING_END: &str = "</think>";

/// Build the Qwen3 unified parser for one stream.
pub(crate) fn qwen3_unified(tools: &[Tool]) -> Box<dyn UnifiedParser> {
    Box::new(ScannerUnified::new(qwen3_scanner(tools).with_reasoning(
        ReasoningSpec {
            start: REASONING_START,
            end: REASONING_END,
            // Qwen3 emits its own `<think>`; the template does not pre-fill one,
            // so the stream starts in visible content (policy P5).
            forced_start: false,
            // `<think>` is not a special token for this family; the OR comes from the grammar.
            preserve_special_tokens: false,
            ..Default::default()
        },
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::unified::{
        InvalidGuidedPayloadPolicy, UnifiedEvent, UnifiedParserEvent, UnifiedParserExt,
        UnifiedParserInit, UnifiedParserStartingState, UnifiedToolOutputMode, assemble,
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

    fn events(tools: &[Tool], chunks: &[&str]) -> Vec<UnifiedEvent> {
        let mut parser = qwen3_unified(tools);
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
        let mut parser = qwen3_unified(tools);
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

    fn recover_init(
        starting_state: UnifiedParserStartingState,
        tool_output_mode: UnifiedToolOutputMode,
    ) -> UnifiedParserInit {
        UnifiedParserInit {
            starting_state,
            tool_output_mode,
            invalid_guided_payload: InvalidGuidedPayloadPolicy::RecoverAsText,
            ..UnifiedParserInit::default()
        }
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
                "<think>Look it up.</think>",
                "<tool_call><function=get_weather><parameter=city>Paris</parameter></function></tool_call>",
                "<think>Now answer.</think>It's 18C.",
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
    fn content_before_reasoning_is_not_hoisted() {
        let out = events(
            &weather_tools(),
            &["Hello there. <think>let me recall</think>The capital is Paris."],
        );
        assert_eq!(
            out,
            vec![
                text("Hello there. "),
                reasoning("let me recall"),
                text("The capital is Paris."),
            ]
        );
    }

    #[test]
    fn unterminated_reasoning_is_promoted_at_finish() {
        // 4.e: not dropped, and not leaked as visible text.
        let out = events(&weather_tools(), &["<think>thinking but stream ends"]);
        assert_eq!(out, vec![reasoning("thinking but stream ends")]);
    }

    #[test]
    fn markers_split_across_chunks_never_leak() {
        // Every marker is cut in half at a chunk boundary.
        let out = events(
            &weather_tools(),
            &[
                "<thi",
                "nk>a</thin",
                "k>go: <tool",
                "_call><func",
                "tion=get_weather><parameter=city>Paris</parameter></func",
                "tion></tool",
                "_call>done",
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
    fn reasoning_marker_inside_an_argument_is_data() {
        // I7: once a tool block is open, `<think>` is a value, not a control
        // token, and must survive byte-exact.
        let tools = vec![Tool {
            name: "run".to_string(),
            description: None,
            parameters: serde_json::json!({
                "type": "object",
                "properties": { "cmd": { "type": "string" } }
            }),
            strict: None,
        }];
        let out = events(
            &tools,
            &[
                "<tool_call><function=run><parameter=cmd>echo <think>hi</think></parameter></function></tool_call>",
            ],
        );
        assert_eq!(
            out,
            vec![call(
                "run",
                serde_json::json!({"cmd": "echo <think>hi</think>"})
            )]
        );
    }

    #[test]
    fn tool_call_inside_reasoning_is_extracted_and_splits_the_thought() {
        // Tool structure dominates reasoning. A call the model emits inside a
        // thought is still a real call, so it surfaces as its own event and the
        // thought splits around it. Burying it would both drop the call and
        // leak `<tool_call>` markup into the reasoning payload (`I3`).
        let out = events(
            &weather_tools(),
            &[
                "<think>I should check. <tool_call><function=get_weather><parameter=city>Paris</parameter></function></tool_call> now answer</think>Done.",
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
    fn the_two_nestings_are_not_symmetric() {
        // Tool-inside-reasoning extracts the call (above), but
        // reasoning-inside-a-tool-argument stays argument data (`I7`) — the
        // in-block scan never looks for reasoning markers.
        let tools = vec![Tool {
            name: "log".to_string(),
            description: None,
            parameters: serde_json::json!({
                "type": "object",
                "properties": { "note": { "type": "string" } }
            }),
            strict: None,
        }];
        let out = events(
            &tools,
            &[
                "<tool_call><function=log><parameter=note><think>reconsider</think></parameter></function></tool_call>",
            ],
        );
        assert_eq!(
            out,
            vec![call(
                "log",
                serde_json::json!({"note": "<think>reconsider</think>"})
            )]
        );
    }

    #[test]
    fn an_argument_value_may_contain_the_block_close_marker() {
        // I7: the scanner has already delimited the invoke, so typing must not
        // re-discover its bounds and cut the value at an embedded `</tool_call>`.
        let tools = vec![Tool {
            name: "run".to_string(),
            description: None,
            parameters: serde_json::json!({
                "type": "object",
                "properties": { "cmd": { "type": "string" } }
            }),
            strict: None,
        }];
        let out = events(
            &tools,
            &[
                "<tool_call>\n<function=run>\n<parameter=cmd>\ngit log </tool_call> --oneline\n</parameter>\n</function>\n</tool_call>",
            ],
        );
        assert_eq!(
            out,
            vec![call(
                "run",
                serde_json::json!({"cmd": "git log </tool_call> --oneline"})
            )]
        );
    }

    #[test]
    fn a_duplicate_reasoning_opener_inside_a_thought_is_stripped() {
        // I3: best-effort recovery strips malformed markup rather than letting it
        // land in the payload. A second `<think>` while one is already open is a
        // duplicate opener, not content.
        let out = events(&weather_tools(), &["<think>a<think>b</think>tail"]);
        assert_eq!(out, vec![reasoning("ab"), text("tail")]);
    }

    #[test]
    fn a_stray_tool_close_inside_a_thought_is_stripped() {
        // Same rule as the orphan handler applies OUTSIDE reasoning — a stray
        // `</tool_call>` with nothing open is markup, so it must not leak into the
        // reasoning payload just because a thought happens to be open.
        let out = events(&weather_tools(), &["<think>a</tool_call>b</think>tail"]);
        assert_eq!(out, vec![reasoning("ab"), text("tail")]);
    }

    /// Guided decoding constrains the TOOL payload, not the reasoning channel, so
    /// the model can still emit malformed markup inside a thought. These assert the
    /// two request modes agree BYTE FOR BYTE on identical reasoning bytes — the
    /// property that was broken: guided only scanned for the closer, so a duplicate
    /// opener surfaced as `reasoning("a<think>b")` where native gave `reasoning("ab")`,
    /// putting raw tags in what the user reads as the model's thinking.
    fn guided_reasoning(chunk: &str) -> Vec<UnifiedEvent> {
        let mut parser = qwen3_unified(&weather_tools());
        parser
            .initialize_request(recover_init(
                UnifiedParserStartingState::None,
                UnifiedToolOutputMode::GuidedJson { named_tool: None },
            ))
            .unwrap();
        let mut deltas = parser.push(chunk).unwrap();
        deltas.extend(parser.finish().unwrap().events);
        assemble(&deltas)
    }

    const GUIDED_CALL: &str = r#"[{"name": "get_weather", "arguments": {"city": "Paris"}}]"#;

    fn marker_tools() -> Vec<Tool> {
        vec![Tool {
            name: "f".to_string(),
            description: None,
            parameters: serde_json::json!({
                "type": "object",
                "properties": { "x": { "type": "string" } }
            }),
            strict: None,
        }]
    }

    fn required_events(tools: &[Tool], chunks: &[&str]) -> Vec<UnifiedEvent> {
        configured_events(
            tools,
            UnifiedParserStartingState::None,
            UnifiedToolOutputMode::GuidedJson { named_tool: None },
            chunks,
        )
    }

    fn assert_required_split_invariant(tools: &[Tool], input: &str, want: &[UnifiedEvent]) {
        assert_eq!(required_events(tools, &[input]), want, "whole-input push");
        for split in 0..=input.len() {
            if !input.is_char_boundary(split) {
                continue;
            }
            let chunks = [&input[..split], &input[split..]];
            assert_eq!(
                required_events(tools, &chunks),
                want,
                "split at {split} of {input:?}"
            );
        }
    }

    #[test]
    fn competing_control_marker_ends_a_prefix_header_at_every_split() {
        let tools = marker_tools();
        let input = r#"x<function=f><tool_call>[{"name":"f","arguments":{"x":"y"}}]"#;
        assert_required_split_invariant(
            &tools,
            input,
            &[text("x"), call("f", serde_json::json!({"x": "y"}))],
        );
    }

    #[test]
    fn required_guided_mode_preserves_whitespace_after_visible_trailing_text() {
        let tools = marker_tools();
        let input = r#"[{"name":"f","arguments":{"x":"y"}}]x "#;
        let want = vec![call("f", serde_json::json!({"x": "y"})), text("x ")];
        assert_required_split_invariant(&tools, input, &want);

        let character_chunks: Vec<&str> = input
            .char_indices()
            .map(|(at, ch)| &input[at..at + ch.len_utf8()])
            .collect();
        assert_eq!(
            required_events(&tools, &character_chunks),
            want,
            "one-character pushes"
        );
    }

    #[test]
    fn a_payload_boundary_before_a_prefix_header_does_not_collapse_its_search() {
        let tools = marker_tools();
        let input = "x{}<function=f>";
        assert_required_split_invariant(&tools, input, &[text("x{}")]);
    }

    #[test]
    fn generated_control_marker_pairs_are_split_invariant() {
        let tools = marker_tools();
        let payload = r#"[{"name":"f","arguments":{"x":"y"}}]"#;
        let components = [
            "<function=",
            "<tool_call>",
            "<think>",
            "</think>",
            "{",
            "[",
            ">",
        ];

        for first in components {
            for second in components {
                let input = format!("x{first}{second}{payload}");
                let whole = required_events(&tools, &[&input]);
                for split in 0..=input.len() {
                    if !input.is_char_boundary(split) {
                        continue;
                    }
                    let chunks = [&input[..split], &input[split..]];
                    assert_eq!(
                        required_events(&tools, &chunks),
                        whole,
                        "generated pair ({first:?}, {second:?}), split at {split}"
                    );
                }
            }
        }
    }

    #[test]
    fn a_complete_guided_call_is_emitted_by_the_push_that_completes_it() {
        for split in 0..GUIDED_CALL.len() {
            let mut parser = qwen3_unified(&weather_tools());
            parser
                .initialize_request(recover_init(
                    UnifiedParserStartingState::None,
                    UnifiedToolOutputMode::GuidedJson { named_tool: None },
                ))
                .unwrap();

            let mut before = parser.push(&GUIDED_CALL[..split]).unwrap();
            assert!(
                before
                    .iter()
                    .all(|delta| !matches!(delta, UnifiedParserEvent::ToolCall(_))),
                "call emitted before its JSON value completed at split {split}"
            );
            let completing = parser.push(&GUIDED_CALL[split..]).unwrap();
            assert!(
                completing
                    .iter()
                    .any(|delta| matches!(delta, UnifiedParserEvent::ToolCall(_))),
                "completing push buffered the call until finish at split {split}"
            );
            before.extend(completing);
            let finished = parser.finish().unwrap().events;
            assert!(
                finished.is_empty(),
                "finish re-emitted data at split {split}"
            );
            assert_eq!(
                assemble(&before),
                vec![call("get_weather", serde_json::json!({"city": "Paris"}))],
                "split {split}"
            );
        }
    }

    #[test]
    fn syntax_after_a_completed_guided_payload_keeps_order_at_every_split() {
        let input = format!("{GUIDED_CALL} \t</tool_call><think>after</think><function=orphan>");
        let want = vec![
            call("get_weather", serde_json::json!({"city": "Paris"})),
            reasoning("after"),
        ];
        for split in 0..=input.len() {
            let chunks = [&input[..split], &input[split..]];
            assert_eq!(
                configured_events(
                    &weather_tools(),
                    UnifiedParserStartingState::None,
                    UnifiedToolOutputMode::GuidedJson { named_tool: None },
                    &chunks,
                ),
                want,
                "split {split}"
            );
        }
    }

    #[test]
    fn a_required_choice_call_with_non_object_arguments_is_voided_like_the_named_path() {
        // The named path already rejects a payload that is not a JSON object. A
        // required-choice ELEMENT has the same wire contract, so `"just a string"`
        // as arguments must not dispatch: it cannot bind to the tool's parameters,
        // and a tool call is a side effect, so failing OPEN is the wrong direction.
        let out = guided_reasoning(r#"[{"name": "get_weather", "arguments": "just a string"}]"#);
        assert_eq!(
            out,
            vec![text(
                r#"[{"name": "get_weather", "arguments": "just a string"}]"#
            )],
            "a non-object argument payload should surface as text, not dispatch"
        );
    }

    #[test]
    fn guided_handles_prose_before_a_thought_split_across_chunks() {
        // The single-chunk case is covered by the invariant test above. This pins the
        // boundary: the prose arrives BEFORE the opener is visible, so nothing may
        // latch the payload buffer until the parser knows no thought is coming.
        let mut parser = qwen3_unified(&weather_tools());
        parser
            .initialize_request(recover_init(
                UnifiedParserStartingState::None,
                UnifiedToolOutputMode::GuidedJson { named_tool: None },
            ))
            .unwrap();
        let mut deltas = Vec::new();
        for chunk in ["Hello there. ", "<think>let me recall</think>", GUIDED_CALL] {
            deltas.extend(parser.push(chunk).unwrap());
        }
        deltas.extend(parser.finish().unwrap().events);
        let out = assemble(&deltas);
        assert_eq!(
            out[0],
            text("Hello there. "),
            "prose was not emitted as text: {out:?}"
        );
        assert_eq!(
            out[1],
            reasoning("let me recall"),
            "thought not recovered: {out:?}"
        );
    }

    /// Push `input` one N-byte slice at a time (char-boundary safe).
    fn guided_chunked(input: &str, n: usize) -> Vec<UnifiedEvent> {
        let mut parser = qwen3_unified(&weather_tools());
        parser
            .initialize_request(recover_init(
                UnifiedParserStartingState::None,
                UnifiedToolOutputMode::GuidedJson { named_tool: None },
            ))
            .unwrap();
        let mut deltas = Vec::new();
        let mut i = 0;
        while i < input.len() {
            let mut j = (i + n).min(input.len());
            while !input.is_char_boundary(j) {
                j += 1;
            }
            deltas.extend(parser.push(&input[i..j]).unwrap());
            i = j;
        }
        deltas.extend(parser.finish().unwrap().events);
        assemble(&deltas)
    }

    #[test]
    fn a_thinking_tag_inside_a_guided_argument_is_data_at_every_chunk_size() {
        // I7: inside the payload a reasoning marker is argument data. I6: the answer
        // cannot depend on chunking. This regressed once — a whole-input push found the
        // `<think>` in the argument string, split the payload into text/reasoning/text
        // and DROPPED the call, while small chunks parsed it correctly.
        let payload = r#"[{"name": "log", "arguments": {"note": "<think>x</think>"}}]"#;
        let want = vec![call("log", serde_json::json!({"note": "<think>x</think>"}))];
        assert_eq!(guided_reasoning(payload), want, "whole-input push");
        for n in [1, 3, 7, 16, 64] {
            assert_eq!(guided_chunked(payload, n), want, "chunk size {n}");
        }
    }

    #[test]
    fn guided_strips_an_orphan_reasoning_close_after_prose() {
        // Native strips a stray `</think>` wherever it appears before any opener and
        // emits the preceding prose as text; the guided path only did so when the
        // prefix was whitespace, so the markup rode into the payload and out verbatim.
        let out = guided_reasoning(&format!("Hello </think>{GUIDED_CALL}"));
        assert_eq!(
            out[0],
            text("Hello "),
            "prose+orphan close mishandled: {out:?}"
        );
        assert!(
            !format!("{out:?}").contains("</think>"),
            "orphan closer leaked: {out:?}"
        );
    }

    #[test]
    fn prose_then_orphan_close_then_payload_is_chunk_independent() {
        // The prose is buffered by an EARLIER chunk than the one carrying the closer,
        // so judging only the current prefix left it glued to the JSON and lost the
        // call — while one push emitted it as text and parsed fine (`I6`).
        fn named(chunks: &[&str]) -> Vec<UnifiedEvent> {
            let mut parser = qwen3_unified(&weather_tools());
            parser
                .initialize_request(recover_init(
                    UnifiedParserStartingState::None,
                    UnifiedToolOutputMode::GuidedJson {
                        named_tool: Some("get_weather".to_string()),
                    },
                ))
                .unwrap();
            let mut deltas = Vec::new();
            for c in chunks {
                deltas.extend(parser.push(c).unwrap());
            }
            deltas.extend(parser.finish().unwrap().events);
            assemble(&deltas)
        }
        let want = vec![
            text("thinking text"),
            call("get_weather", serde_json::json!({"city": "Paris"})),
        ];
        assert_eq!(
            named(&[r#"thinking text</think>{"city": "Paris"}"#]),
            want,
            "one push"
        );
        assert_eq!(
            named(&["thinking text", "</think>", r#"{"city": "Paris"}"#]),
            want,
            "split at the closer"
        );
    }

    #[test]
    fn tool_markup_narrated_inside_a_thought_does_not_eat_the_guided_payload() {
        // Guided decoding leaves the reasoning channel UNCONSTRAINED, so the model can
        // write `<tool_call>` while narrating; the real call arrives after, as JSON.
        // Treating that markup as stream-ending discarded the payload and returned an
        // empty response.
        let out = guided_reasoning(&format!(
            "<think>I'll use <tool_call> next</think>{GUIDED_CALL}"
        ));
        assert!(
            out.iter()
                .any(|e| matches!(e, UnifiedEvent::ToolCall { .. })),
            "guided payload was discarded: {out:?}"
        );
        assert!(
            !format!("{out:?}").contains("<tool_call>"),
            "narrated markup leaked: {out:?}"
        );
    }

    #[test]
    fn an_orphan_closer_before_a_real_thought_is_stripped_not_shown() {
        // The opener search ran over the whole buffer before the closer was ever
        // considered, so an orphan `</think>` sitting AHEAD of a real thought landed in
        // the opener's prefix and went out as visible text. Native compares positions
        // and strips the earlier marker.
        let native = events(&weather_tools(), &["</think>a<think>b</think>tail"]);
        let guided = guided_reasoning(&format!("</think>a<think>b</think>{GUIDED_CALL}"));
        assert_eq!(native[0], text("a"), "native changed: {native:?}");
        assert_eq!(
            guided[0],
            text("a"),
            "orphan closer leaked into the reply: {guided:?}"
        );
        assert!(
            !format!("{guided:?}").contains("</think>"),
            "markup reached the user: {guided:?}"
        );
    }

    #[test]
    fn a_named_choice_forwards_its_payload_verbatim() {
        // The payload IS the tool's argument object. An earlier revision tried to
        // unwrap a `{"name","arguments"}` shape to tolerate envelope-emitting backends;
        // that heuristic is unsound because ordinary arguments can use those key names,
        // and it broke both ways — voiding a legitimate forced call when the inner name
        // differed, and dropping the real argument set when it matched.
        fn named(tool: &str, payload: &str) -> Vec<UnifiedEvent> {
            let tools = vec![Tool {
                name: tool.into(),
                description: None,
                parameters: serde_json::json!({"type": "object"}),
                strict: None,
            }];
            let mut parser = qwen3_unified(&tools);
            parser
                .initialize_request(recover_init(
                    UnifiedParserStartingState::None,
                    UnifiedToolOutputMode::GuidedJson {
                        named_tool: Some(tool.to_string()),
                    },
                ))
                .unwrap();
            let mut deltas = parser.push(payload).unwrap();
            deltas.extend(parser.finish().unwrap().events);
            assemble(&deltas)
        }
        assert_eq!(
            named("get_weather", r#"{"city": "Paris"}"#),
            vec![call("get_weather", serde_json::json!({"city": "Paris"}))],
            "bare arguments"
        );
        // The case the heuristic broke: a forced tool whose OWN arguments happen to use
        // `name` + `parameters`. It must still be dispatched, with those arguments.
        let args = serde_json::json!({"name": "foo", "parameters": {"x": 1}});
        assert_eq!(
            named("register_handler", &args.to_string()),
            vec![call("register_handler", args.clone())],
            "legitimate arguments using name/parameters keys must still dispatch"
        );
        let same = serde_json::json!({"name": "register_handler", "parameters": {"x": 1}});
        assert_eq!(
            named("register_handler", &same.to_string()),
            vec![call("register_handler", same.clone())],
            "an inner name matching the tool must not truncate the argument set"
        );
    }

    /// Guided control markers must never reach the user and must never cost the
    /// call, at EVERY chunk boundary, for every starting_state and choice shape.
    ///
    /// This is a table rather than a list of examples on purpose. Every previous
    /// bug in this area was a cell someone else found: an opener recognised whole
    /// but lost when split, a prefix-form `<function=` consumed by its declared
    /// length leaving `NAME>` behind, markup after a thought never examined because
    /// the closer had already latched. Each was fixed with the one input that had
    /// broken, so the next cell broke next. The property is combinatorial — marker
    /// x position x delivery x starting_state x choice — so the test is too.
    #[test]
    fn guided_control_markers_never_leak_and_never_cost_the_call() {
        let choices = [
            (Some("get_weather"), r#"{"city": "Paris"}"#),
            (
                None,
                r#"[{"name":"get_weather","arguments":{"city":"Paris"}}]"#,
            ),
        ];
        // Marker positions differ by prompt state. Reasoning starting_state starts inside a
        // thought and response starting_state makes reasoning tags literal, while tool
        // control markers remain structural until the JSON value opens in all modes.
        let cases: &[(UnifiedParserStartingState, &[&str])] = &[
            (
                UnifiedParserStartingState::None,
                &[
                    "<tool_call>",
                    "</tool_call>",
                    "<function=get_weather>",
                    "<think>x</think><tool_call>",
                    "<think>x</think></tool_call>",
                    "</think><tool_call>",
                    "prose <tool_call>",
                    "",
                ],
            ),
            (
                UnifiedParserStartingState::Reasoning,
                &[
                    "x</think><tool_call>",
                    "x</think></tool_call>",
                    "<tool_call>x</think>",
                    "<function=get_weather>x</think>",
                    "</tool_call>x</think>",
                    "</think>",
                ],
            ),
            (
                UnifiedParserStartingState::Response,
                &["<tool_call>", "</tool_call>", "<function=get_weather>", ""],
            ),
        ];
        for &(starting_state, prefixes) in cases {
            for &(named_tool, payload) in &choices {
                for prefix in prefixes {
                    // Markers can BRACKET the payload, not only precede it: a
                    // template-emitted closer after the JSON rode into the buffer
                    // and cost the call. The table covers both ends now.
                    for suffix in ["", "</tool_call>", "</function>"] {
                        let input = format!("{prefix}{payload}{suffix}");
                        // Delivery: whole, then split at EVERY byte boundary.
                        let mut deliveries: Vec<Vec<String>> = vec![vec![input.clone()]];
                        for cut in 1..input.len() {
                            if input.is_char_boundary(cut) {
                                deliveries.push(vec![input[..cut].into(), input[cut..].into()]);
                            }
                        }
                        for chunks in deliveries {
                            let mut parser = qwen3_unified(&weather_tools());
                            parser
                                .initialize_request(recover_init(
                                    starting_state,
                                    UnifiedToolOutputMode::GuidedJson {
                                        named_tool: named_tool.map(str::to_string),
                                    },
                                ))
                                .unwrap();
                            let mut deltas = Vec::new();
                            for c in &chunks {
                                deltas.extend(parser.push(c).unwrap());
                            }
                            deltas.extend(parser.finish().unwrap().events);
                            let out = assemble(&deltas);
                            let at = format!(
                                "starting_state {starting_state:?}, named {named_tool:?}, prefix {prefix:?}, chunks {chunks:?} -> {out:?}"
                            );

                            assert!(
                                out.iter().any(|e| matches!(
                                    e, UnifiedEvent::ToolCall { name, arguments }
                                    if name == "get_weather"
                                        && arguments == &serde_json::json!({"city": "Paris"})
                                )),
                                "call lost or arguments wrong: {at}"
                            );
                            for ev in &out {
                                if let UnifiedEvent::Text { text }
                                | UnifiedEvent::Reasoning { text } = ev
                                {
                                    for marker in [
                                        "<tool_call>",
                                        "</tool_call>",
                                        "<function=",
                                        "<think>",
                                        "</think>",
                                    ] {
                                        assert!(
                                            !text.contains(marker),
                                            "{marker} leaked to the user: {at}"
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn guided_native_markup_only_is_split_invariant() {
        let input = "<tool_call><function=get_weather><parameter=city>Paris</parameter></function></tool_call>";

        for split in 0..=input.len() {
            let chunks = [&input[..split], &input[split..]];
            let out = configured_events(
                &weather_tools(),
                UnifiedParserStartingState::None,
                UnifiedToolOutputMode::GuidedJson { named_tool: None },
                &chunks,
            );
            assert!(out.is_empty(), "split at {split} leaked {out:?}");
        }
    }

    #[test]
    fn markers_inside_a_started_guided_payload_stay_byte_exact() {
        // The other half of the same property: once the payload has opened, a marker
        // is argument DATA and must survive untouched (`I7`), at every boundary.
        const INPUT: &str = r#"{"city": "<tool_call><think>x</think></function>"}"#;
        for cut in 0..INPUT.len() {
            if !INPUT.is_char_boundary(cut) {
                continue;
            }
            let mut parser = qwen3_unified(&weather_tools());
            parser
                .initialize_request(recover_init(
                    UnifiedParserStartingState::None,
                    UnifiedToolOutputMode::GuidedJson {
                        named_tool: Some("get_weather".to_string()),
                    },
                ))
                .unwrap();
            let mut deltas = parser.push(&INPUT[..cut]).unwrap();
            deltas.extend(parser.push(&INPUT[cut..]).unwrap());
            deltas.extend(parser.finish().unwrap().events);
            assert_eq!(
                assemble(&deltas),
                vec![call(
                    "get_weather",
                    serde_json::json!({"city": "<tool_call><think>x</think></function>"})
                )],
                "argument data changed, split at {cut}"
            );
        }
    }

    #[test]
    fn a_narrated_invoke_inside_a_thought_leaves_no_marker_behind() {
        // Reviewed: stripping `<function=NAME>` left its `</function>` in the shown
        // thinking. The invoke terminator is now part of the guided vocabulary.
        //
        // The two markers NOT added, and why: `<parameter=` and a BARE `</function>`
        // with no invoke open are kept verbatim by the NATIVE scanner as well —
        // measured identical on both paths — so stripping them here would create a
        // divergence rather than remove one. Those two cases are pinned below.
        let leaked = guided_reasoning(&format!(
            "<think>a<function=run>x</function>x</think>{GUIDED_CALL}"
        ));
        assert!(
            !format!("{leaked:?}").contains("</function>"),
            "invoke terminator left behind: {leaked:?}"
        );
        assert!(
            leaked
                .iter()
                .any(|e| matches!(e, UnifiedEvent::ToolCall { .. })),
            "call lost: {leaked:?}"
        );

        // Parity with native on the shapes that are ordinary text for both.
        for thought in [
            "<think>a<parameter=city>y</parameter>b</think>",
            "<think>a</function>b</think>",
        ] {
            let native = events(&weather_tools(), &[&format!("{thought}tail")]);
            let guided = guided_reasoning(&format!("{thought}{GUIDED_CALL}"));
            assert_eq!(
                native[0], guided[0],
                "guided diverged from native on text-like markup for {thought:?}"
            );
        }
    }

    #[test]
    fn a_narrated_invoke_does_not_swallow_the_thought_closer_or_the_payload() {
        // The terminator search was unbounded, so a `</function>` occurring inside a
        // guided ARGUMENT STRING — past the end of the thought — was claimed as the
        // narrated invoke's terminator. Everything between went with it: the closer,
        // the payload, the call. Fragments of the discarded JSON then surfaced as the
        // model's thinking.
        let input = concat!(
            r#"<think>I'll use <function=get_weather> to check</think>"#,
            r#"[{"name":"log","arguments":{"note":"close with </function>"}}]"#
        );
        let mut parser = qwen3_unified(&weather_tools());
        parser
            .initialize_request(recover_init(
                UnifiedParserStartingState::None,
                UnifiedToolOutputMode::GuidedJson { named_tool: None },
            ))
            .unwrap();
        let mut deltas = parser.push(input).unwrap();
        deltas.extend(parser.finish().unwrap().events);
        let out = assemble(&deltas);
        assert!(
            out.iter()
                .any(|e| matches!(e, UnifiedEvent::ToolCall { name, .. } if name == "log")),
            "the terminator search swallowed the payload: {out:?}"
        );
        assert!(
            !format!("{out:?}").contains("}}]"),
            "payload fragments surfaced as thinking: {out:?}"
        );
    }

    #[test]
    fn guided_strips_a_duplicate_reasoning_opener_inside_a_thought() {
        let out = guided_reasoning(&format!("<think>a<think>b</think>{GUIDED_CALL}"));
        assert_eq!(
            out[0],
            reasoning("ab"),
            "duplicate opener leaked into the thought"
        );
    }

    #[test]
    fn guided_strips_a_stray_tool_close_inside_a_thought() {
        let out = guided_reasoning(&format!("<think>a</tool_call>b</think>{GUIDED_CALL}"));
        assert_eq!(
            out[0],
            reasoning("ab"),
            "stray tool close leaked into the thought"
        );
    }

    #[test]
    fn guided_and_native_agree_on_the_same_reasoning_bytes() {
        // Two properties, deliberately scoped differently.
        //
        // NO MARKER MAY LEAK, on either path, for ANY of these inputs. That one is
        // absolute — it is the `I3` contract, and comparing only the FIRST event is
        // how a leak survived this test twice, coming out in the tail instead.
        //
        // BYTE-EQUAL reasoning payloads hold only for inputs with no native TOOL
        // structure. Native can interpret `<tool_call>`/`<function=`: it opens a
        // block and recovers the call from the markup. Guided cannot — under guided
        // decoding the reasoning channel is unconstrained, so that markup is the
        // model NARRATING, and the real call arrives afterwards as JSON. Treating it
        // as structural there discarded the payload and returned an empty response.
        // So the two modes genuinely differ on those inputs, and the honest thing is
        // to say so rather than force an equality that costs the call.
        let equal_payload = [
            "<think>a<think>b</think>",
            "<think>a</tool_call>b</think>",
            // Visible prose BEFORE the thought opens (`content_then_reason`). Every
            // other case here starts the thought at byte 0, which is how the guided
            // path shipped a bug where the prose latched the payload buffer and the
            // model's private thinking was surfaced to the user as the answer.
            "Hello there. <think>let me recall</think>",
            "Sure. <think>check</think>",
            "<think>plain thought</think>",
        ];
        // Native tool structure inside a thought: no-leak only, per the note above.
        let no_leak_only = [
            "<think>a<tool_call>x</think>",
            "<think>a<function=run>x</function>x</think>",
        ];
        for thought in equal_payload.iter().chain(no_leak_only.iter()) {
            let native = events(&weather_tools(), &[&format!("{thought}tail")]);
            let guided = guided_reasoning(&format!("{thought}{GUIDED_CALL}"));
            if equal_payload.contains(thought) {
                assert_eq!(
                    native[0], guided[0],
                    "request mode changed the reasoning payload for {thought:?}"
                );
            }
            // Comparing only the FIRST event is how the tool-opener leak survived
            // this test twice: the reasoning span matched while the markup came
            // out in the tail instead. No event on either side may carry a marker.
            for ev in native.iter().chain(guided.iter()) {
                if let UnifiedEvent::Text { text } | UnifiedEvent::Reasoning { text } = ev {
                    for marker in ["<think>", "</think>", "<tool_call>", "<function="] {
                        assert!(
                            !text.contains(marker),
                            "{marker} leaked into {ev:?} for {thought:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn guided_strips_a_stray_marker_split_across_chunks() {
        // The holdback was narrowed to the closer, so a stray marker split across a
        // chunk boundary streamed out before it could be recognized.
        let mut parser = qwen3_unified(&weather_tools());
        parser
            .initialize_request(recover_init(
                UnifiedParserStartingState::None,
                UnifiedToolOutputMode::GuidedJson { named_tool: None },
            ))
            .unwrap();
        let mut deltas = Vec::new();
        for chunk in ["<think>a</tool", "_call>b</think>", GUIDED_CALL] {
            deltas.extend(parser.push(chunk).unwrap());
        }
        deltas.extend(parser.finish().unwrap().events);
        assert_eq!(assemble(&deltas)[0], reasoning("ab"));
    }

    #[test]
    fn orphan_reasoning_close_is_stripped_not_leaked() {
        // I3: a `</think>` with nothing open is malformed markup.
        let out = events(&weather_tools(), &["Hello </think>world"]);
        assert_eq!(out, vec![text("Hello world")]);
    }

    #[test]
    fn truncated_tool_call_at_eof_keeps_preceding_output() {
        // P2: drop the unrecoverable partial call, no error, no leaked markup.
        let out = events(
            &weather_tools(),
            &["<think>ok</think>Checking. <tool_call><function=get_weather><parameter=city>Par"],
        );
        assert_eq!(out, vec![reasoning("ok"), text("Checking. ")]);
    }

    #[test]
    fn empty_arguments_become_an_empty_object() {
        // P3.
        let tools = vec![Tool {
            name: "ping".to_string(),
            description: None,
            parameters: serde_json::json!({"type": "object", "properties": {}}),
            strict: None,
        }];
        let out = events(
            &tools,
            &["<tool_call><function=ping></function></tool_call>"],
        );
        assert_eq!(out, vec![call("ping", serde_json::json!({}))]);
    }

    #[test]
    fn batch_and_stream_assemble_identically() {
        // I6, at the parser level: `parse_complete` routes through the same
        // push/finish lifecycle, so parity is structural.
        let input = "<think>a</think>Here you go: <tool_call><function=get_weather><parameter=city>Paris</parameter></function></tool_call><think>b</think>Done.";
        let batch = qwen3_unified(&weather_tools())
            .parse_complete(input)
            .expect("parse_complete");
        assert_eq!(batch, events(&weather_tools(), &[input]));
        assert_eq!(batch.len(), 5);
    }

    #[test]
    fn reasoning_prefill_classifies_leading_text_as_reasoning() {
        let out = configured_events(
            &weather_tools(),
            UnifiedParserStartingState::Reasoning,
            UnifiedToolOutputMode::Native,
            &["hidden</thi", "nk>visible"],
        );
        assert_eq!(out, vec![reasoning("hidden"), text("visible")]);
    }

    #[test]
    fn reasoning_prefill_consumes_a_redundant_split_opener() {
        for tool_output_mode in [
            UnifiedToolOutputMode::Native,
            UnifiedToolOutputMode::GuidedJson {
                named_tool: Some("get_weather".to_string()),
            },
        ] {
            let out = configured_events(
                &weather_tools(),
                UnifiedParserStartingState::Reasoning,
                tool_output_mode.clone(),
                &["\n<thi", "nk>hidden</think>{\"city\":\"Tokyo\"}"],
            );
            let mut expected = vec![reasoning("\nhidden")];
            if tool_output_mode == UnifiedToolOutputMode::Native {
                expected.push(text(r#"{"city":"Tokyo"}"#));
            } else {
                expected.push(call("get_weather", serde_json::json!({"city": "Tokyo"})));
            }
            assert_eq!(out, expected);
        }
    }

    #[test]
    fn response_prefill_does_not_interpret_reasoning_markers() {
        let out = configured_events(
            &weather_tools(),
            UnifiedParserStartingState::Response,
            UnifiedToolOutputMode::Native,
            &["<think>literal</think>"],
        );
        assert_eq!(out, vec![text("<think>literal</think>")]);
    }

    #[test]
    fn named_choice_parses_bare_arguments_object() {
        let out = configured_events(
            &weather_tools(),
            UnifiedParserStartingState::Reasoning,
            UnifiedToolOutputMode::GuidedJson {
                named_tool: Some("get_weather".to_string()),
            },
            &["reason</think>{\n", "  \"city\": \"Tokyo\"\n}"],
        );
        assert_eq!(
            out,
            vec![
                reasoning("reason"),
                call("get_weather", serde_json::json!({"city": "Tokyo"})),
            ]
        );
    }

    #[test]
    fn guided_json_strips_a_split_orphan_reasoning_close_before_json() {
        let out = configured_events(
            &weather_tools(),
            UnifiedParserStartingState::None,
            UnifiedToolOutputMode::GuidedJson {
                named_tool: Some("get_weather".to_string()),
            },
            &["</thi", "nk>{\"city\":\"Tokyo\"}"],
        );
        assert_eq!(
            out,
            vec![call("get_weather", serde_json::json!({"city": "Tokyo"}))]
        );
    }

    #[test]
    fn guided_json_preserves_native_marker_strings_as_argument_data() {
        let marker_value =
            "<think>x</think><tool_call><function=get_weather></function></tool_call>";
        let input = format!(
            r#"reason</think>{{"city":{}}}"#,
            serde_json::to_string(marker_value).unwrap()
        );
        let out = configured_events(
            &weather_tools(),
            UnifiedParserStartingState::Reasoning,
            UnifiedToolOutputMode::GuidedJson {
                named_tool: Some("get_weather".to_string()),
            },
            &[&input[..20], &input[20..]],
        );
        assert_eq!(
            out,
            vec![
                reasoning("reason"),
                call("get_weather", serde_json::json!({"city": marker_value})),
            ]
        );
    }

    #[test]
    fn required_choice_parses_single_and_parallel_calls() {
        let single = configured_events(
            &weather_tools(),
            UnifiedParserStartingState::Response,
            UnifiedToolOutputMode::GuidedJson { named_tool: None },
            &[r#"{"name":"get_weather","parameters":{"city":"Tokyo"}}"#],
        );
        assert_eq!(
            single,
            vec![call("get_weather", serde_json::json!({"city": "Tokyo"}))]
        );

        let parallel = configured_events(
            &weather_tools(),
            UnifiedParserStartingState::Response,
            UnifiedToolOutputMode::GuidedJson { named_tool: None },
            &[
                r#"[{"name":"get_weather","arguments":{"city":"Paris"}},"#,
                r#"{"name":"get_weather","parameters":{"city":"Tokyo"}}]"#,
            ],
        );
        assert_eq!(
            parallel,
            vec![
                call("get_weather", serde_json::json!({"city": "Paris"})),
                call("get_weather", serde_json::json!({"city": "Tokyo"})),
            ]
        );
    }

    #[test]
    fn required_choice_recovers_the_whole_array_when_any_call_is_invalid() {
        // Invalid = missing `name` (the one required field). A missing ARGUMENT key is
        // not invalid — that is a parameterless call, per `UNIFIED.6.a`.
        let input = r#"[{"name":"get_weather","parameters":{"city":"Paris"}},{"parameters":{"city":"Tokyo"}}]"#;
        let out = configured_events(
            &weather_tools(),
            UnifiedParserStartingState::Response,
            UnifiedToolOutputMode::GuidedJson { named_tool: None },
            &[input],
        );
        assert_eq!(out, vec![text(input)]);
    }

    #[test]
    fn required_choice_rejects_explicit_null_arguments() {
        // Missing arguments means a parameterless call. Explicit null is different:
        // it is a present value with the wrong shape and must not be dispatched as
        // `{}`, because that turns a malformed side-effect request into a valid one.
        for input in [
            r#"{"name":"get_weather","arguments":null}"#,
            r#"{"name":"get_weather","parameters":null}"#,
        ] {
            let out = configured_events(
                &weather_tools(),
                UnifiedParserStartingState::Response,
                UnifiedToolOutputMode::GuidedJson { named_tool: None },
                &[input],
            );
            assert_eq!(out, vec![text(input)], "explicit null dispatched: {out:?}");
        }
    }

    #[test]
    fn required_choice_rejects_ambiguous_argument_fields() {
        let input =
            r#"{"name":"get_weather","parameters":{"city":"Paris"},"arguments":{"city":"Tokyo"}}"#;
        let out = configured_events(
            &weather_tools(),
            UnifiedParserStartingState::Response,
            UnifiedToolOutputMode::GuidedJson { named_tool: None },
            &[input],
        );
        assert_eq!(out, vec![text(input)], "ambiguous call dispatched: {out:?}");
    }

    #[test]
    fn native_mode_keeps_xml_under_reasoning_prefill() {
        let out = configured_events(
            &weather_tools(),
            UnifiedParserStartingState::Reasoning,
            UnifiedToolOutputMode::Native,
            &[
                "reason</think><tool_call><function=get_weather>",
                "<parameter=city>Paris</parameter></function></tool_call>",
            ],
        );
        assert_eq!(
            out,
            vec![
                reasoning("reason"),
                call("get_weather", serde_json::json!({"city": "Paris"})),
            ]
        );
    }

    #[test]
    fn guided_json_is_chunk_boundary_independent() {
        let input = r#"reason</think>[{"name":"get_weather","parameters":{"city":"Tokyo"}}]"#;
        let chunks = input
            .as_bytes()
            .iter()
            .map(|byte| std::str::from_utf8(std::slice::from_ref(byte)).unwrap())
            .collect::<Vec<_>>();
        let out = configured_events(
            &weather_tools(),
            UnifiedParserStartingState::Reasoning,
            UnifiedToolOutputMode::GuidedJson { named_tool: None },
            &chunks,
        );
        assert_eq!(
            out,
            vec![
                reasoning("reason"),
                call("get_weather", serde_json::json!({"city": "Tokyo"})),
            ]
        );
    }

    #[test]
    fn named_choice_preserves_surrounding_argument_bytes() {
        // The named payload is the argument object itself. Validation may use a
        // trimmed view, but the emitted wire string must remain model-byte-exact.
        let input = " \n{\"city\": \"Tokyo\"}\t ";
        let mut parser = qwen3_unified(&weather_tools());
        parser
            .initialize_request(recover_init(
                UnifiedParserStartingState::Response,
                UnifiedToolOutputMode::GuidedJson {
                    named_tool: Some("get_weather".to_string()),
                },
            ))
            .unwrap();
        let mut deltas = parser.push(input).unwrap();
        deltas.extend(parser.finish().unwrap().events);
        let arguments = deltas
            .iter()
            .filter_map(|delta| match delta {
                UnifiedParserEvent::ToolCall(call) if call.tool_index == 0 => {
                    Some(call.arguments.as_str())
                }
                _ => None,
            })
            .collect::<String>();
        assert_eq!(arguments, input);
    }

    #[test]
    fn incomplete_guided_json_recovers_as_text() {
        for tool_output_mode in [
            UnifiedToolOutputMode::GuidedJson {
                named_tool: Some("get_weather".to_string()),
            },
            UnifiedToolOutputMode::GuidedJson { named_tool: None },
        ] {
            let input = r#"{"city":"Tok"#;
            let out = configured_events(
                &weather_tools(),
                UnifiedParserStartingState::Response,
                tool_output_mode,
                &[input],
            );
            assert_eq!(out, vec![text(input)]);
        }
    }

    #[test]
    fn reset_recovers_buffered_guided_text_and_restarts_lifecycle() {
        let mut parser = qwen3_unified(&weather_tools());
        parser
            .initialize_request(recover_init(
                UnifiedParserStartingState::Reasoning,
                UnifiedToolOutputMode::GuidedJson {
                    named_tool: Some("get_weather".to_string()),
                },
            ))
            .unwrap();
        assert_eq!(
            parser.push(r#"reason</think>{"city":"Tok"#).unwrap(),
            vec![UnifiedParserEvent::Reasoning("reason".to_string())]
        );
        assert!(
            parser
                .initialize_request(recover_init(
                    UnifiedParserStartingState::Response,
                    UnifiedToolOutputMode::Native
                ))
                .is_err()
        );
        assert_eq!(parser.reset(), r#"{"city":"Tok"#);

        parser
            .initialize_request(recover_init(
                UnifiedParserStartingState::Response,
                UnifiedToolOutputMode::GuidedJson {
                    named_tool: Some("get_weather".to_string()),
                },
            ))
            .unwrap();
        let mut deltas = parser.push(r#"{"city":"Tokyo"}"#).unwrap();
        deltas.extend(parser.finish().unwrap().events);
        assert_eq!(
            assemble(&deltas),
            vec![call("get_weather", serde_json::json!({"city": "Tokyo"}))]
        );
        assert!(parser.finish().is_err());
        assert!(parser.push("after finish").is_err());
    }

    /// P2 recovery must not show the user markup the parse already stripped.
    ///
    /// `finish_json` trims trailing control markers before parsing; the fallback
    /// used to hand back the RAW buffer, so a malformed guided payload put
    /// `</tool_call>` in the visible answer (`I3`). Byte fidelity still holds when
    /// nothing was stripped (`I7`), which the third row pins.
    #[test]
    fn guided_recovery_text_never_carries_the_markup_it_stripped() {
        let tools = weather_tools();
        for (input, want) in [
            (
                "[{\"name\": \"get_weather\", \"arguments\": {\"city\": </tool_call>",
                "[{\"name\": \"get_weather\", \"arguments\": {\"city\":",
            ),
            (
                "{\"unexpected\": \"shape\"}</tool_call>",
                "{\"unexpected\": \"shape\"}",
            ),
            ("{\"unexpected\": \"shape\"}", "{\"unexpected\": \"shape\"}"),
        ] {
            let got = configured_events(
                &tools,
                UnifiedParserStartingState::None,
                UnifiedToolOutputMode::GuidedJson { named_tool: None },
                &[input],
            );
            assert_eq!(got, vec![text(want)], "input {input:?}");
        }
    }

    /// A prefix-form marker's `>` search is bounded by the payload start.
    ///
    /// Unbounded, it ran INTO the payload: with no `>` before the JSON the flush
    /// arm consumed the whole buffer and the turn emitted nothing at all, losing
    /// a well-formed call.
    #[test]
    fn a_bare_invoke_opener_does_not_swallow_the_guided_payload() {
        let tools = weather_tools();
        let got = configured_events(
            &tools,
            UnifiedParserStartingState::None,
            UnifiedToolOutputMode::GuidedJson { named_tool: None },
            &["<function=[{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}]"],
        );
        assert_eq!(
            got,
            vec![call("get_weather", serde_json::json!({"city": "Paris"}))],
            "bare opener swallowed the payload: {got:?}"
        );
    }

    /// `control_marker_at` and `guided_holdback_len` must agree on when a
    /// prefix-form marker is COMPLETE, at every chunk boundary.
    ///
    /// They disagreed once: consume required the `>` before the payload start,
    /// holdback accepted any `>` anywhere after the marker. A `>` inside an
    /// argument string satisfies only the second, so `<function=` was neither
    /// consumed nor retained — it flushed into the payload buffer, the JSON failed
    /// to parse, and the call surfaced as text with the marker still attached.
    #[test]
    fn a_marker_before_a_payload_containing_gt_still_dispatches_at_every_split() {
        let tools = weather_tools();
        let input = "<function=[{\"name\": \"get_weather\", \"arguments\": {\"city\": \"a > b\"}}]";
        let want = vec![call("get_weather", serde_json::json!({"city": "a > b"}))];
        for split in 0..=input.len() {
            if split > 0 && !input.is_char_boundary(split) {
                continue;
            }
            let chunks: Vec<&str> = if split == 0 {
                vec![input]
            } else {
                vec![&input[..split], &input[split..]]
            };
            let got = configured_events(
                &tools,
                UnifiedParserStartingState::None,
                UnifiedToolOutputMode::GuidedJson { named_tool: None },
                &chunks,
            );
            assert_eq!(got, want, "split at {split}");
        }
    }

    /// A prefix-form header must not BORROW its `>` from a later marker.
    ///
    /// `control_marker_at` bounded the `>` scan by the payload start only, so a
    /// stray `<function=` consumed through the `>` of a following `<think>` and the
    /// model's PRIVATE reasoning was emitted as visible text. The boundary is the
    /// earliest payload OR competing control/reasoning marker, and both the consume
    /// path and the holdback derive it from `prefix_header_end` — one rule, because
    /// two predicates drifted twice before this.
    #[test]
    fn a_prefix_header_never_borrows_a_later_markers_terminator() {
        let tools = weather_tools();
        let call = call("get_weather", serde_json::json!({"city": "Paris"}));
        for (input, want) in [
            (
                "<function=<think>secret</think>[{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}]",
                vec![reasoning("secret"), call.clone()],
            ),
            (
                "<think>I'll call <function=get_weather</think>[{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}]",
                vec![reasoning("I'll call get_weather"), call.clone()],
            ),
        ] {
            for split in 0..=input.len() {
                if split > 0 && !input.is_char_boundary(split) {
                    continue;
                }
                let chunks: Vec<&str> = if split == 0 {
                    vec![input]
                } else {
                    vec![&input[..split], &input[split..]]
                };
                let got = configured_events(
                    &tools,
                    UnifiedParserStartingState::None,
                    UnifiedToolOutputMode::GuidedJson { named_tool: None },
                    &chunks,
                );
                assert_eq!(got, want, "split at {split} of {input:?}");
                for ev in &got {
                    if let UnifiedEvent::Text { text } | UnifiedEvent::Reasoning { text } = ev {
                        assert!(!text.contains("<function="), "marker leaked: {text:?}");
                    }
                }
            }
        }
    }

    use crate::tool_calling::traits::ToolCallDelta;

    /// Init selecting the streaming contract.
    fn stream_init(mode: UnifiedToolOutputMode) -> UnifiedParserInit {
        UnifiedParserInit {
            prompt_token_ids: Vec::new(),
            starting_state: UnifiedParserStartingState::None,
            tool_output_mode: mode,
            invalid_guided_payload: InvalidGuidedPayloadPolicy::StreamBestEffort,
        }
    }

    /// Every delta produced by feeding `payload` in `chunks` under `tool_choice`
    /// `required`, push order preserved.
    fn streamed(payload: &str, chunks: &[&str]) -> (Vec<ToolCallDelta>, Vec<UnifiedParserEvent>) {
        let _ = payload;
        streamed_in(
            UnifiedToolOutputMode::GuidedJson { named_tool: None },
            InvalidGuidedPayloadPolicy::StreamBestEffort,
            chunks,
        )
    }

    /// The same, for a NAMED choice: the payload is `get_weather`'s bare arguments.
    fn streamed_named(chunks: &[&str]) -> (Vec<ToolCallDelta>, Vec<UnifiedParserEvent>) {
        streamed_in(
            UnifiedToolOutputMode::GuidedJson {
                named_tool: Some("get_weather".to_string()),
            },
            InvalidGuidedPayloadPolicy::StreamBestEffort,
            chunks,
        )
    }

    /// One driver for every guided contract: the required and named modes must be
    /// exercised through the SAME push/finish sequence, or a divergence in the
    /// harness hides a divergence in the parser.
    fn streamed_in(
        mode: UnifiedToolOutputMode,
        policy: InvalidGuidedPayloadPolicy,
        chunks: &[&str],
    ) -> (Vec<ToolCallDelta>, Vec<UnifiedParserEvent>) {
        let mut parser = qwen3_unified(&weather_tools());
        let mut init = stream_init(mode);
        init.invalid_guided_payload = policy;
        parser.initialize_request(init).expect("guided init");
        let mut calls = Vec::new();
        let mut all = Vec::new();
        for chunk in chunks {
            for event in parser.push(chunk).expect("push") {
                if let UnifiedParserEvent::ToolCall(delta) = &event {
                    calls.push(delta.clone());
                }
                all.push(event);
            }
        }
        for event in parser.finish().expect("finish").events {
            if let UnifiedParserEvent::ToolCall(delta) = &event {
                calls.push(delta.clone());
            }
            all.push(event);
        }
        (calls, all)
    }

    /// One chunk per character.
    fn per_char(payload: &str) -> Vec<&str> {
        payload
            .char_indices()
            .map(|(i, ch)| &payload[i..i + ch.len_utf8()])
            .collect()
    }

    /// Byte span of the first argument OBJECT in `payload`, tolerating any
    /// whitespace the fixture uses between the key, the colon and the brace.
    fn argument_object(payload: &str) -> (usize, usize) {
        let key = payload.find("\"arguments\"").expect("fixture shape");
        let open = payload[key..].find('{').expect("fixture shape") + key;
        let mut depth = 0i32;
        let mut in_string = false;
        let mut escaped = false;
        for (offset, ch) in payload[open..].char_indices() {
            if in_string {
                match ch {
                    _ if escaped => escaped = false,
                    '\\' => escaped = true,
                    '"' => in_string = false,
                    _ => {}
                }
                continue;
            }
            match ch {
                '"' => in_string = true,
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        return (open, open + offset + 1);
                    }
                }
                _ => {}
            }
        }
        panic!("unterminated argument object in {payload:?}");
    }

    /// Arguments reassembled per tool index, in first-seen index order.
    fn joined_arguments(calls: &[ToolCallDelta]) -> Vec<(usize, String)> {
        let mut out: Vec<(usize, String)> = Vec::new();
        for delta in calls {
            match out.iter_mut().find(|(index, _)| *index == delta.tool_index) {
                Some((_, text)) => text.push_str(&delta.arguments),
                None => out.push((delta.tool_index, delta.arguments.clone())),
            }
        }
        out
    }

    /// The defect a downstream user reported: with `tool_choice="required"`,
    /// streaming on and thinking disabled, nothing reaches the wire until the whole
    /// call has been generated. Their vanilla-vLLM deployment starts streaming in
    /// ~0.2s; this path took ~10s and then arrived in one burst.
    ///
    /// The assertion is ORDERING, not wall-clock: a name-carrying event must exist
    /// before the parser has been handed the bytes that complete the call. A timing
    /// assertion would pass on a fast machine, fail on a slow one, and say nothing
    /// about emission granularity.
    #[test]
    fn required_guided_emits_the_name_before_the_call_completes() {
        // Cut just after the arguments object opens - the commit point - which is
        // still well before the arguments close.
        let cut = argument_object(GUIDED_CALL).0 + 1;
        let mut parser = qwen3_unified(&weather_tools());
        parser
            .initialize_request(stream_init(UnifiedToolOutputMode::GuidedJson {
                named_tool: None,
            }))
            .expect("required guided init");

        let early = parser.push(&GUIDED_CALL[..cut]).expect("push prefix");
        let named = early.iter().any(|e| {
            matches!(e, UnifiedParserEvent::ToolCall(c) if c.name.as_deref() == Some("get_weather"))
        });

        assert!(
            named,
            "no name-carrying frame before the call completed; got {early:?}. \
             The function name is unambiguous at byte {cut} - everything after it is \
             arguments - so a consumer could already have started rendering the call."
        );
    }

    /// Streaming must be INCREMENTAL, not merely "earlier", and the fragments must
    /// reassemble BYTE FOR BYTE.
    ///
    /// An earlier version of this test asserted only that the join `contains("Paris")`,
    /// which a parser that mangled every other byte would still pass. The assertion
    /// is now equality against the exact argument object.
    #[test]
    fn required_guided_streams_arguments_in_multiple_fragments() {
        let (calls, events) = streamed(GUIDED_CALL, &per_char(GUIDED_CALL));

        let names: Vec<&str> = calls.iter().filter_map(|c| c.name.as_deref()).collect();
        assert_eq!(names, vec!["get_weather"], "exactly one decoded name");
        assert!(
            calls.iter().all(|c| c.tool_index == 0),
            "one call must not spread across indices: {calls:?}"
        );

        let frames = calls.iter().filter(|c| !c.arguments.is_empty()).count();
        assert!(
            frames >= 2,
            "arguments arrived in {frames} frame(s) - that is a burst, not a stream"
        );

        let (start, end) = argument_object(GUIDED_CALL);
        let expected = GUIDED_CALL[start..end].to_string();
        assert_eq!(
            joined_arguments(&calls),
            vec![(0, expected)],
            "fragments must reassemble the argument object byte for byte"
        );

        assert!(
            !events
                .iter()
                .any(|e| matches!(e, UnifiedParserEvent::Text(_))),
            "a valid streamed call must not also produce recovery text: {events:?}"
        );
    }

    /// The `parameters` spelling is as legal as `arguments`, and recognising only
    /// one of them streamed NOTHING for the other - the call then assembled with
    /// empty arguments, which is worse than the latency it was fixing.
    #[test]
    fn required_guided_streams_the_parameters_alias_too() {
        let payload = r#"[{"name":"get_weather","parameters":{"city":"Tokyo"}}]"#;
        let (calls, _) = streamed(payload, &per_char(payload));
        assert_eq!(
            calls
                .iter()
                .filter_map(|c| c.name.as_deref())
                .collect::<Vec<_>>(),
            vec!["get_weather"]
        );
        assert_eq!(
            joined_arguments(&calls),
            vec![(0, r#"{"city":"Tokyo"}"#.to_string())]
        );
        // Assert it STREAMED, not merely that it assembled. The buffered path
        // produces the identical assembled result, so an outcome-only assertion
        // passes with the alias unrecognised - which is exactly the bug.
        let frames = calls.iter().filter(|c| !c.arguments.is_empty()).count();
        assert!(
            frames >= 2,
            "the parameters alias assembled correctly but never streamed: \
             {frames} argument frame(s)"
        );
    }

    /// A `\uXXXX` escape in the function name must decode. The old scanner pushed
    /// the escape's payload characters verbatim and produced `getu005fweather`.
    #[test]
    fn required_guided_decodes_escaped_names() {
        let payload = "[{\"name\":\"get\\u005fweather\",\"arguments\":{\"city\":\"Paris\"}}]";
        assert!(
            payload.contains("\\u005f"),
            "the escape must survive to the test"
        );
        let (calls, _) = streamed(payload, &per_char(payload));
        assert_eq!(
            calls
                .iter()
                .filter_map(|c| c.name.as_deref())
                .collect::<Vec<_>>(),
            vec!["get_weather"]
        );
    }

    /// Parallel calls keep distinct indices and stay in payload order. Hardcoding
    /// index 0 collapsed both calls into one.
    #[test]
    fn required_guided_streams_parallel_calls_on_distinct_indices() {
        let payload = concat!(
            r#"[{"name":"get_weather","arguments":{"city":"Paris"}},"#,
            r#"{"name":"get_weather","arguments":{"city":"Tokyo"}}]"#
        );
        let (calls, _) = streamed(payload, &per_char(payload));
        let named: Vec<(usize, &str)> = calls
            .iter()
            .filter_map(|c| c.name.as_deref().map(|n| (c.tool_index, n)))
            .collect();
        assert_eq!(named, vec![(0, "get_weather"), (1, "get_weather")]);
        assert_eq!(
            joined_arguments(&calls),
            vec![
                (0, r#"{"city":"Paris"}"#.to_string()),
                (1, r#"{"city":"Tokyo"}"#.to_string()),
            ]
        );
    }

    /// PER-CALL recovery, the guarantee `StreamBestEffort` trades the atomic one
    /// for: a malformed second element becomes its own text and the valid first
    /// element still dispatches.
    #[test]
    fn required_guided_recovers_only_the_invalid_element() {
        let payload = concat!(
            r#"[{"name":"get_weather","arguments":{"city":"Paris"}},"#,
            r#"{"arguments":{"city":"Tokyo"}}]"#
        );
        let (calls, events) = streamed(payload, &per_char(payload));
        assert_eq!(
            calls
                .iter()
                .filter_map(|c| c.name.as_deref())
                .collect::<Vec<_>>(),
            vec!["get_weather"],
            "the valid element must still dispatch"
        );
        let texts: Vec<&str> = events
            .iter()
            .filter_map(|e| match e {
                UnifiedParserEvent::Text(text) => Some(text.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(
            texts,
            vec![r#"{"arguments":{"city":"Tokyo"}}"#],
            "only the nameless element recovers as text"
        );
    }

    /// Shapes the contract voids must never reach the wire early, because a
    /// fragment cannot be withdrawn. The commit point requires an argument OBJECT
    /// opener, so each of these stays on the buffered path.
    #[test]
    fn required_guided_never_streams_a_shape_the_contract_voids() {
        for payload in [
            r#"[{"name":"get_weather","arguments":"just a string"}]"#,
            r#"[{"name":"get_weather","arguments":null}]"#,
            r#"[{"name":"get_weather","arguments":[1,2]}]"#,
        ] {
            let (calls, events) = streamed(payload, &per_char(payload));
            assert!(
                calls.is_empty(),
                "{payload} streamed a call the contract voids: {calls:?}"
            );
            assert!(
                events
                    .iter()
                    .any(|e| matches!(e, UnifiedParserEvent::Text(_))),
                "{payload} should have recovered as text; got {events:?}"
            );
        }
    }

    /// The ONE guarantee `StreamBestEffort` genuinely cannot keep, pinned so it
    /// stays a documented loss rather than a surprise.
    ///
    /// A call carrying BOTH argument aliases is ambiguous and the buffered
    /// contracts void it. Streaming cannot: the second alias only appears after the
    /// first has already supplied an object opener, which is the commit point, so
    /// the call is on the wire before the ambiguity exists to be seen. It stays,
    /// carrying the alias that arrived first, and the parser says so in a warning.
    #[test]
    fn ambiguity_discovered_after_the_commit_cannot_be_withdrawn() {
        let payload = r#"[{"name":"get_weather","arguments":{"a":1},"parameters":{"b":2}}]"#;
        let (calls, events) = streamed(payload, &per_char(payload));
        assert_eq!(
            calls
                .iter()
                .filter_map(|c| c.name.as_deref())
                .collect::<Vec<_>>(),
            vec!["get_weather"],
            "the call was committed before the second alias arrived"
        );
        assert_eq!(
            joined_arguments(&calls),
            vec![(0, r#"{"a":1}"#.to_string())],
            "it carries the alias that opened first"
        );
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, UnifiedParserEvent::Text(_))),
            "recovery text here would DUPLICATE the streamed call: {events:?}"
        );
    }

    /// A parameterless call has no argument object to commit on, so it settles on
    /// the buffered path - and must still arrive with `{}`, not an empty string,
    /// and must not disturb a streamed sibling.
    #[test]
    fn required_guided_parameterless_call_keeps_its_empty_argument_set() {
        let payload = concat!(
            r#"[{"name":"get_weather"},"#,
            r#"{"name":"get_weather","arguments":{"city":"Paris"}}]"#
        );
        let (calls, _) = streamed(payload, &per_char(payload));
        let mut assembled = joined_arguments(&calls);
        // Deltas for call 0 arrive AFTER call 1's, and that is inherent: a
        // parameterless call has no argument object to commit on, so it can only be
        // settled once the payload closes, by which time its streamed sibling has
        // long been on the wire. Consumers key by `tool_index` (`assemble` does), so
        // the wire order is not the call order.
        assembled.sort_by_key(|(index, _)| *index);
        assert_eq!(
            assembled,
            vec![
                (0, "{}".to_string()),
                (1, r#"{"city":"Paris"}"#.to_string())
            ]
        );
    }

    /// Output conservation: bytes already dispatched as a tool call must NOT come
    /// back as visible text. When a later array element leaves the payload
    /// unsplittable, the recovery path may only emit the UN-streamed remainder --
    /// re-emitting the whole buffer makes a client execute the tool and then render
    /// its JSON as prose.
    #[test]
    fn required_guided_truncation_does_not_duplicate_a_streamed_call_as_text() {
        // One complete call, then a second element truncated right after its
        // `"arguments":` key, so the payload can no longer be split into elements.
        let payload = concat!(
            r#"[{"name": "get_weather", "arguments": {"city": "Paris"}},"#,
            r#"{"name": "get_weather", "arguments":"#
        );
        let (calls, all) = streamed(payload, &per_char(payload));

        // The first call must have streamed.
        assert!(
            calls
                .iter()
                .any(|c| c.name.as_deref() == Some("get_weather")),
            "expected the complete first call to stream: {calls:?}"
        );
        let streamed_args: String = calls
            .iter()
            .filter(|c| c.tool_index == 0)
            .map(|c| c.arguments.as_str())
            .collect();
        assert!(
            !streamed_args.is_empty(),
            "expected argument fragments for call 0"
        );

        let text: String = all
            .iter()
            .filter_map(|e| match e {
                UnifiedParserEvent::Text(t) => Some(t.as_str()),
                _ => None,
            })
            .collect();

        assert!(
            !text.contains(&streamed_args),
            "recovery text re-emitted bytes already dispatched as call 0.\n               streamed args: {streamed_args:?}\n  recovery text: {text:?}"
        );
    }

    /// The conservation invariant must hold at EVERY chunk boundary, not just the
    /// per-char one: the split decides how much streamed before truncation, so a
    /// single split size can pass while others duplicate.
    #[test]
    fn required_guided_truncation_conserves_output_at_every_split() {
        let payload = concat!(
            r#"[{"name": "get_weather", "arguments": {"city": "Paris"}},"#,
            r#"{"name": "get_weather", "arguments":"#
        );
        for split in 0..=payload.len() {
            if !payload.is_char_boundary(split) {
                continue;
            }
            let chunks = vec![&payload[..split], &payload[split..]];
            let (calls, all) = streamed(payload, &chunks);
            let streamed_args: String = calls
                .iter()
                .filter(|c| c.tool_index == 0)
                .map(|c| c.arguments.as_str())
                .collect();
            if streamed_args.is_empty() {
                continue; // nothing committed at this split, nothing to conserve
            }
            let text: String = all
                .iter()
                .filter_map(|e| match e {
                    UnifiedParserEvent::Text(t) => Some(t.as_str()),
                    _ => None,
                })
                .collect();
            assert!(
                !text.contains(&streamed_args),
                "split {split}: recovery text re-emitted dispatched bytes\n                   streamed: {streamed_args:?}\n  text: {text:?}"
            );
        }
    }

    /// The single-call case: one call commits, then the stream ends mid-arguments.
    /// There is no second element to make the payload unsplittable, so this
    /// exercises the other recovery branch.
    #[test]
    fn required_guided_single_committed_call_truncated_conserves_output() {
        let cut = GUIDED_CALL.find("Paris").expect("fixture shape") + 2;
        let payload = &GUIDED_CALL[..cut];
        let (calls, all) = streamed(payload, &per_char(payload));
        let streamed_args: String = calls
            .iter()
            .filter(|c| c.tool_index == 0)
            .map(|c| c.arguments.as_str())
            .collect();
        if streamed_args.is_empty() {
            return; // nothing committed, nothing to conserve
        }
        let text: String = all
            .iter()
            .filter_map(|e| match e {
                UnifiedParserEvent::Text(t) => Some(t.as_str()),
                _ => None,
            })
            .collect();
        assert!(
            !text.contains(&streamed_args),
            "single-call truncation re-emitted dispatched bytes\n               streamed: {streamed_args:?}\n  text: {text:?}"
        );
    }

    /// Truncation mid-payload: whatever was streamed stays streamed, and nothing
    /// is invented for the part that never arrived.
    #[test]
    fn required_guided_truncation_after_commit_does_not_invent_arguments() {
        let cut = GUIDED_CALL.find("\"city\"").expect("fixture shape");
        let (calls, _) = streamed(&GUIDED_CALL[..cut], &per_char(&GUIDED_CALL[..cut]));
        assert_eq!(
            calls
                .iter()
                .filter_map(|c| c.name.as_deref())
                .collect::<Vec<_>>(),
            vec!["get_weather"]
        );
        let joined = joined_arguments(&calls);
        let released = joined.first().map(|(_, text)| text.as_str()).unwrap_or("");
        assert!(
            GUIDED_CALL[cut - released.len()..].starts_with(released) || released.len() <= 1,
            "released bytes must be a prefix of the real arguments; got {released:?}"
        );
    }

    /// Reuse: a second request on the same parser must not inherit the first
    /// request's cursor offsets, or the new payload is lexed from the wrong place.
    #[test]
    fn required_guided_streaming_state_does_not_leak_across_requests() {
        let mut parser = qwen3_unified(&weather_tools());
        let mut seen = Vec::new();
        for request in 0..2 {
            if request > 0 {
                parser.reset();
            }
            parser
                .initialize_request(stream_init(UnifiedToolOutputMode::GuidedJson {
                    named_tool: None,
                }))
                .expect("required guided init");
            let mut calls = Vec::new();
            for chunk in per_char(GUIDED_CALL) {
                for event in parser.push(chunk).expect("push") {
                    if let UnifiedParserEvent::ToolCall(delta) = event {
                        calls.push(delta);
                    }
                }
            }
            for event in parser.finish().expect("finish").events {
                if let UnifiedParserEvent::ToolCall(delta) = event {
                    calls.push(delta);
                }
            }
            seen.push(joined_arguments(&calls));
        }
        assert_eq!(
            seen[0], seen[1],
            "the second request diverged from the first"
        );
    }

    /// Whole-input and every valid split must assemble identically, and the split
    /// runs must show intermediate progress rather than one terminal burst.
    #[test]
    fn required_guided_streaming_is_split_invariant() {
        let whole = streamed(GUIDED_CALL, &[GUIDED_CALL]).0;
        let baseline = joined_arguments(&whole);
        let names: Vec<&str> = whole.iter().filter_map(|c| c.name.as_deref()).collect();

        for split in 1..GUIDED_CALL.len() {
            if !GUIDED_CALL.is_char_boundary(split) {
                continue;
            }
            let (calls, _) = streamed(GUIDED_CALL, &[&GUIDED_CALL[..split], &GUIDED_CALL[split..]]);
            assert_eq!(
                joined_arguments(&calls),
                baseline,
                "arguments differ at split {split}"
            );
            assert_eq!(
                calls
                    .iter()
                    .filter_map(|c| c.name.as_deref())
                    .collect::<Vec<_>>(),
                names,
                "names differ at split {split}"
            );
        }
    }

    // ---- named choice: bare arguments for the tool the request already fixed ----

    /// The v2 gap this closes: a named choice refused to stream at all, so a client
    /// waited for the whole argument object even though the tool name was known
    /// before the first byte. The name must land on the FIRST delta, and argument
    /// bytes must follow as fragments.
    #[test]
    fn named_guided_streams_its_bare_arguments() {
        let payload = r#"{"city": "Paris", "unit": "celsius"}"#;
        let (calls, events) = streamed_named(&per_char(payload));

        let names: Vec<&str> = calls.iter().filter_map(|c| c.name.as_deref()).collect();
        assert_eq!(names, vec!["get_weather"], "exactly one name, once");
        assert!(
            calls[0].name.as_deref() == Some("get_weather"),
            "the name must ride the FIRST delta, not a later one: {calls:?}"
        );
        assert!(
            calls[1..].iter().all(|c| c.name.is_none()),
            "every later delta must carry name: None: {calls:?}"
        );
        assert!(
            calls.iter().all(|c| c.tool_index == 0),
            "a named choice is one call: {calls:?}"
        );

        let frames = calls.iter().filter(|c| !c.arguments.is_empty()).count();
        assert!(
            frames >= 2,
            "arguments arrived in {frames} frame(s) - that is a burst, not a stream"
        );
        assert_eq!(
            joined_arguments(&calls),
            vec![(0, payload.to_string())],
            "fragments must reassemble the payload byte for byte"
        );
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, UnifiedParserEvent::Text(_))),
            "a valid streamed call must not also produce recovery text: {events:?}"
        );
    }

    /// The highest-risk failure mode: streaming puts the arguments on the wire and
    /// then the completion path settles the SAME call again, so the client receives
    /// the argument bytes twice and `assemble` concatenates them into `{…}{…}`.
    ///
    /// The assertion is on the TOTAL: every argument byte the client receives across
    /// streaming plus completion must equal the payload exactly once.
    #[test]
    fn named_guided_never_emits_the_arguments_twice() {
        let payload = r#"{"city": "Paris"}"#;
        for chunks in [
            per_char(payload),
            vec![payload],
            vec![&payload[..1], &payload[1..]],
        ] {
            let (calls, _) = streamed_named(&chunks);
            assert_eq!(
                joined_arguments(&calls),
                vec![(0, payload.to_string())],
                "argument bytes were not delivered exactly once: {calls:?}"
            );
            assert_eq!(
                calls.iter().filter(|c| c.name.is_some()).count(),
                1,
                "the name was delivered more than once: {calls:?}"
            );
        }
    }

    /// Same invariant at EVERY chunk boundary: the split decides how much streamed
    /// before the payload closed, so one split can pass while another duplicates.
    #[test]
    fn named_guided_conserves_output_at_every_split() {
        let payload = r#"{"city": "Paris", "unit": "celsius"}"#;
        for split in 0..=payload.len() {
            if !payload.is_char_boundary(split) {
                continue;
            }
            let (calls, events) = streamed_named(&[&payload[..split], &payload[split..]]);
            assert_eq!(
                joined_arguments(&calls),
                vec![(0, payload.to_string())],
                "split {split}: arguments were not delivered exactly once"
            );
            let text: String = events
                .iter()
                .filter_map(|e| match e {
                    UnifiedParserEvent::Text(t) => Some(t.as_str()),
                    _ => None,
                })
                .collect();
            assert!(
                text.is_empty(),
                "split {split}: streamed bytes came back as text: {text:?}"
            );
        }
    }

    /// A split can only land on a char boundary because `push` takes `&str`, but the
    /// released fragments must still never cut a codepoint, and they must reassemble
    /// the multi-byte content byte for byte.
    #[test]
    fn named_guided_is_split_invariant_with_multi_byte_arguments() {
        let payload = r#"{"city": "東京", "emoji": "😀"}"#;
        let baseline = vec![(0, payload.to_string())];
        for split in 0..=payload.len() {
            if !payload.is_char_boundary(split) {
                continue;
            }
            let (calls, _) = streamed_named(&[&payload[..split], &payload[split..]]);
            assert_eq!(
                joined_arguments(&calls),
                baseline,
                "arguments differ at split {split}"
            );
            assert_eq!(
                calls
                    .iter()
                    .filter_map(|c| c.name.as_deref())
                    .collect::<Vec<_>>(),
                vec!["get_weather"],
                "names differ at split {split}"
            );
        }
        let (per_char_calls, _) = streamed_named(&per_char(payload));
        assert_eq!(joined_arguments(&per_char_calls), baseline);
    }

    /// A named payload that does not open with `{` is not an argument set: it cannot
    /// be bound to the tool, and the buffered path already surfaces it as text.
    /// Streaming it would put a shape the contract voids on the wire, unwithdrawably.
    #[test]
    fn named_guided_never_streams_a_payload_that_is_not_an_argument_set() {
        for payload in [r#""just a string""#, "42", "null", "[1,2]"] {
            let (calls, events) = streamed_named(&per_char(payload));
            assert!(
                calls.is_empty(),
                "{payload}: streamed a payload that is not an argument set: {calls:?}"
            );
            let text: String = events
                .iter()
                .filter_map(|e| match e {
                    UnifiedParserEvent::Text(t) => Some(t.as_str()),
                    _ => None,
                })
                .collect();
            assert_eq!(
                text, payload,
                "{payload}: buffered recovery text changed shape"
            );
        }
    }

    /// Streaming stays OPT-IN. Under the default `Reject` policy a named choice must
    /// still buffer to completion and arrive as one terminal delta.
    #[test]
    fn named_guided_still_buffers_under_the_default_reject_policy() {
        let payload = r#"{"city": "Paris"}"#;
        assert_eq!(
            InvalidGuidedPayloadPolicy::default(),
            InvalidGuidedPayloadPolicy::Reject,
            "this test is about the DEFAULT policy"
        );
        let (calls, _) = streamed_in(
            UnifiedToolOutputMode::GuidedJson {
                named_tool: Some("get_weather".to_string()),
            },
            InvalidGuidedPayloadPolicy::default(),
            &per_char(payload),
        );
        assert_eq!(
            calls.len(),
            1,
            "the payload streamed under Reject: {calls:?}"
        );
        assert_eq!(calls[0].name.as_deref(), Some("get_weather"));
        assert_eq!(calls[0].arguments, payload);
    }

    /// Text after the payload is still text, and none of the streamed argument bytes
    /// may be repeated into it.
    #[test]
    fn named_guided_keeps_post_payload_text_out_of_the_arguments() {
        let payload = r#"{"city": "Paris"}"#;
        let input = format!("{payload}done");
        let (calls, events) = streamed_named(&per_char(&input));
        assert_eq!(joined_arguments(&calls), vec![(0, payload.to_string())]);
        let text: String = events
            .iter()
            .filter_map(|e| match e {
                UnifiedParserEvent::Text(t) => Some(t.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(text, "done", "post-payload text was lost or duplicated");
    }

    /// Truncation after the commit: whatever streamed stays streamed, nothing is
    /// invented for the bytes that never arrived, and the released prefix must not
    /// come back as visible text.
    #[test]
    fn named_guided_truncation_conserves_output() {
        let payload = r#"{"city": "Par"#;
        let (calls, events) = streamed_named(&per_char(payload));
        assert_eq!(
            calls
                .iter()
                .filter_map(|c| c.name.as_deref())
                .collect::<Vec<_>>(),
            vec!["get_weather"]
        );
        let streamed_args: String = calls.iter().map(|c| c.arguments.as_str()).collect();
        assert_eq!(
            streamed_args, payload,
            "released bytes must be the arrivals"
        );
        let text: String = events
            .iter()
            .filter_map(|e| match e {
                UnifiedParserEvent::Text(t) => Some(t.as_str()),
                _ => None,
            })
            .collect();
        assert!(
            !text.contains(&streamed_args),
            "recovery text re-emitted dispatched bytes: {text:?}"
        );
    }

    /// Reuse: a second request must not inherit the first request's cursor offsets,
    /// and a named cursor must still be named after the reset.
    #[test]
    fn named_guided_streaming_state_does_not_leak_across_requests() {
        let payload = r#"{"city": "Paris"}"#;
        let mut parser = qwen3_unified(&weather_tools());
        let mut seen = Vec::new();
        for request in 0..2 {
            if request > 0 {
                parser.reset();
            }
            parser
                .initialize_request(stream_init(UnifiedToolOutputMode::GuidedJson {
                    named_tool: Some("get_weather".to_string()),
                }))
                .expect("named guided init");
            let mut calls = Vec::new();
            for chunk in per_char(payload) {
                for event in parser.push(chunk).expect("push") {
                    if let UnifiedParserEvent::ToolCall(delta) = event {
                        calls.push(delta);
                    }
                }
            }
            for event in parser.finish().expect("finish").events {
                if let UnifiedParserEvent::ToolCall(delta) = event {
                    calls.push(delta);
                }
            }
            seen.push(joined_arguments(&calls));
        }
        assert_eq!(seen[0], vec![(0, payload.to_string())]);
        assert_eq!(
            seen[0], seen[1],
            "the second request diverged from the first"
        );
    }

    /// The named payload arriving after a thought, in one piece, must still stream
    /// and must not swallow or duplicate the reasoning.
    #[test]
    fn named_guided_streams_after_reasoning() {
        let payload = r#"{"city": "Paris"}"#;
        let input = format!("<think>hmm</think>{payload}");
        let (calls, events) = streamed_named(&per_char(&input));
        assert_eq!(joined_arguments(&calls), vec![(0, payload.to_string())]);
        let reasoning: String = events
            .iter()
            .filter_map(|e| match e {
                UnifiedParserEvent::Reasoning(t) => Some(t.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(reasoning, "hmm");
    }
}

#[cfg(test)]
mod guided_warning_tests {
    use super::*;
    use crate::tool_calling::traits::Tool;
    use crate::unified::{
        InvalidGuidedPayloadPolicy, UnifiedParserEvent, UnifiedParserExt, UnifiedParserInit,
        UnifiedParserStartingState, UnifiedToolOutputMode,
    };

    fn weather_tools() -> Vec<Tool> {
        vec![Tool {
            name: "get_weather".into(),
            description: None,
            parameters: serde_json::json!({"type":"object","properties":{"city":{"type":"string"}}}),
            strict: None,
        }]
    }

    fn recover_init(
        starting_state: UnifiedParserStartingState,
        tool_output_mode: UnifiedToolOutputMode,
    ) -> UnifiedParserInit {
        UnifiedParserInit {
            starting_state,
            tool_output_mode,
            invalid_guided_payload: InvalidGuidedPayloadPolicy::RecoverAsText,
            ..UnifiedParserInit::default()
        }
    }

    /// Every malformed guided payload must WARN, not just silently become text.
    /// A caller cannot tell "guided decoding failed" from "the model answered in
    /// prose" by looking at the events, so the log line is the only signal.
    #[test]
    fn malformed_guided_payloads_warn() {
        for (label, payload) in [
            ("valid json, not a call", r#"{"unexpected": "shape"}"#),
            (
                "unparseable json",
                r#"{"name": "get_weather", "arguments": {"city": "Par"#,
            ),
            (
                "array, one element not a call",
                r#"[{"name":"get_weather","arguments":{"city":"Paris"}}, {"arguments":{"city":"Tokyo"}}]"#,
            ),
            (
                "array with a broken element",
                r#"[{"name":"get_weather","arguments":{"city":"Paris"}}, {"name":"run","arguments":{"cmd": ]"#,
            ),
        ] {
            let mut p = qwen3_unified(&weather_tools());
            p.initialize_request(recover_init(
                UnifiedParserStartingState::None,
                UnifiedToolOutputMode::GuidedJson { named_tool: None },
            ))
            .unwrap();
            let mut out = p.push(payload).unwrap();
            out.extend(p.finish().unwrap().events);
            // recovered as text, no call dispatched
            assert!(
                out.iter()
                    .all(|d| !matches!(d, UnifiedParserEvent::ToolCall(_))),
                "{label}: dispatched a call from an unvalidated payload"
            );
            assert!(
                out.iter()
                    .any(|d| matches!(d, UnifiedParserEvent::Text(..))),
                "{label}: payload was dropped instead of surfaced as text"
            );
        }
    }

    /// The recovery above must be OBSERVABLE. Capture the log to prove the warning
    /// is actually emitted, not merely present in the source: to a caller the events
    /// look identical to a model that answered in prose, so this line is the only
    /// way an operator learns guided decoding failed.
    #[test]
    fn the_guided_fallback_actually_emits_a_warning() {
        use std::sync::{Arc, Mutex};
        #[derive(Clone, Default)]
        struct Sink(Arc<Mutex<Vec<u8>>>);
        impl std::io::Write for Sink {
            fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
                self.0.lock().unwrap().extend_from_slice(b);
                Ok(b.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let sink = Sink::default();
        let captured = sink.0.clone();
        let sub = tracing_subscriber::fmt()
            .with_writer(move || sink.clone())
            .with_max_level(tracing::Level::WARN)
            .finish();

        // `tracing` caches per-callsite Interest PROCESS-GLOBALLY. The sibling
        // tests above drive this same guided-fallback `warn!` with no subscriber
        // installed, so one of them can cache the callsite as "never interested"
        // and this capture then sees an empty log — a ~60% flake decided purely by
        // which test reaches the callsite first. A thread-local `with_default`
        // cannot fix that, and neither can a one-shot rebuild, because the race
        // repeats on every run. Installing a permanent global default means the
        // callsite is always interesting; `with_default` below still overrides it
        // on THIS thread, so the capture stays local to this test.
        static GLOBAL_SUB: std::sync::Once = std::sync::Once::new();
        GLOBAL_SUB.call_once(|| {
            let _ = tracing::subscriber::set_global_default(
                tracing_subscriber::fmt()
                    .with_writer(std::io::sink)
                    .with_max_level(tracing::Level::WARN)
                    .finish(),
            );
            tracing::callsite::rebuild_interest_cache();
        });

        const GUIDED_SECRET: &str = "DO_NOT_LOG_GUIDED_PAYLOAD_7f31";
        const ARGUMENT_SECRET: &str = "DO_NOT_LOG_TOOL_ARGUMENTS_98c2";
        tracing::subscriber::with_default(sub, || {
            let mut p = qwen3_unified(&weather_tools());
            p.initialize_request(recover_init(
                UnifiedParserStartingState::None,
                UnifiedToolOutputMode::GuidedJson { named_tool: None },
            ))
            .unwrap();
            p.push(&format!(r#"{{"unexpected": "{GUIDED_SECRET}"}}"#))
                .unwrap();
            p.finish().unwrap();
            crate::unified::assemble(&[UnifiedParserEvent::ToolCall(
                crate::tool_calling::traits::ToolCallDelta {
                    tool_index: 0,
                    name: Some("get_weather".into()),
                    arguments: format!(r#"{{"api_key":"{ARGUMENT_SECRET}""#),
                },
            )]);
        });

        let log = String::from_utf8(captured.lock().unwrap().clone()).unwrap();
        assert!(
            log.contains("unified_guided_json_not_a_tool_call"),
            "no warning emitted for an unparseable guided payload; log was: {log:?}"
        );
        assert!(
            log.contains("required"),
            "warning omitted which tool choice was in play: {log:?}"
        );
        assert!(
            !log.contains(GUIDED_SECRET) && !log.contains(ARGUMENT_SECRET),
            "warning exposed model or user payload bytes: {log:?}"
        );
    }

    /// Guided mode fed the family's NATIVE markup must not produce a silent empty turn.
    ///
    /// Everything is stripped, so emitting no events is right — markup is not an
    /// answer. Doing it SILENTLY is not: the caller cannot tell this from a model
    /// that legitimately said nothing, and the usual cause is guided decoding
    /// configured against a backend still emitting native call grammar. P2 is
    /// best-effort recovery, NOT silent loss.
    #[test]
    fn guided_output_that_is_only_markup_warns_instead_of_vanishing() {
        use std::sync::{Arc, Mutex};
        #[derive(Clone, Default)]
        struct Sink(Arc<Mutex<Vec<u8>>>);
        impl std::io::Write for Sink {
            fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
                self.0.lock().unwrap().extend_from_slice(b);
                Ok(b.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let sink = Sink::default();
        let captured = sink.0.clone();
        let sub = tracing_subscriber::fmt()
            .with_writer(move || sink.clone())
            .with_max_level(tracing::Level::WARN)
            .finish();
        // Same process-global callsite-interest caveat as the sibling test above.
        static GLOBAL_SUB: std::sync::Once = std::sync::Once::new();
        GLOBAL_SUB.call_once(|| {
            let _ = tracing::subscriber::set_global_default(
                tracing_subscriber::fmt()
                    .with_writer(std::io::sink)
                    .with_max_level(tracing::Level::WARN)
                    .finish(),
            );
            tracing::callsite::rebuild_interest_cache();
        });

        let mut events = Vec::new();
        tracing::subscriber::with_default(sub, || {
            let mut p = qwen3_unified(&weather_tools());
            p.initialize_request(recover_init(
                UnifiedParserStartingState::None,
                UnifiedToolOutputMode::GuidedJson { named_tool: None },
            ))
            .unwrap();
            events.extend(
                p.push("<tool_call><function=get_weather><parameter=city>Paris</parameter></function></tool_call>")
                    .unwrap(),
            );
            events.extend(p.finish().unwrap().events);
        });

        assert!(
            events.is_empty(),
            "markup-only guided turn emitted {events:?}"
        );
        let log = String::from_utf8(captured.lock().unwrap().clone()).unwrap();
        assert!(
            log.contains("unified_guided_output_was_only_markup"),
            "empty guided turn produced NO diagnostic; log was: {log}"
        );
    }
}

#[cfg(test)]
mod reset_and_payload_tests {
    use super::*;
    use crate::tool_calling::traits::Tool;
    use crate::unified::{
        InvalidGuidedPayload, InvalidGuidedPayloadKind, InvalidGuidedPayloadPolicy, UnifiedEvent,
        UnifiedParserEvent, UnifiedParserExt, UnifiedParserInit, UnifiedParserOutput,
        UnifiedParserStartingState, UnifiedToolOutputMode, assemble,
    };

    fn tools() -> Vec<Tool> {
        vec![Tool {
            name: "get_weather".into(),
            description: None,
            parameters: serde_json::json!({"type":"object","properties":{"city":{"type":"string"}}}),
            strict: None,
        }]
    }

    fn guided_init(
        named_tool: Option<&str>,
        invalid_guided_payload: InvalidGuidedPayloadPolicy,
    ) -> UnifiedParserInit {
        UnifiedParserInit {
            tool_output_mode: UnifiedToolOutputMode::GuidedJson {
                named_tool: named_tool.map(str::to_string),
            },
            invalid_guided_payload,
            ..UnifiedParserInit::default()
        }
    }

    fn recover_init(
        starting_state: UnifiedParserStartingState,
        tool_output_mode: UnifiedToolOutputMode,
    ) -> UnifiedParserInit {
        UnifiedParserInit {
            starting_state,
            tool_output_mode,
            invalid_guided_payload: InvalidGuidedPayloadPolicy::RecoverAsText,
            ..UnifiedParserInit::default()
        }
    }

    /// `reset` must clear "resume reasoning after the interrupting call". Leaving it
    /// armed made the NEXT stream's first post-call answer come out as reasoning —
    /// the user's visible answer silently becomes private thinking.
    #[test]
    fn reset_does_not_leak_resume_reasoning_into_the_next_stream() {
        let mut p = qwen3_unified(&tools());
        p.initialize_request(recover_init(
            UnifiedParserStartingState::None,
            UnifiedToolOutputMode::Native,
        ))
        .unwrap();
        // interrupt a thought with a call, then reset mid-stream
        p.push("<think>weighing<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>")
            .unwrap();
        p.reset();

        p.initialize_request(recover_init(
            UnifiedParserStartingState::None,
            UnifiedToolOutputMode::Native,
        ))
        .unwrap();
        let out = assemble(
            &[p.push("<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>visible answer").unwrap(),
              p.finish().unwrap().events].concat());
        assert!(
            out.iter().any(
                |e| matches!(e, UnifiedEvent::Text { text } if text.contains("visible answer"))
            ),
            "post-call answer was not visible text after reset: {out:?}"
        );
        assert!(
            !out.iter().any(
                |e| matches!(e, UnifiedEvent::Reasoning { text } if text.contains("visible answer"))
            ),
            "post-call answer leaked into reasoning after reset: {out:?}"
        );
    }

    /// `reset` on a guided stream must restore the channel, not just drop buffers.
    /// Left at VisibleOnly, the next stream's reasoning is swallowed as JSON payload.
    #[test]
    fn reset_restores_guided_channel_state() {
        let mut p = qwen3_unified(&tools());
        p.initialize_request(recover_init(
            UnifiedParserStartingState::None,
            UnifiedToolOutputMode::GuidedJson { named_tool: None },
        ))
        .unwrap();
        p.push(r#"{"partial"#).unwrap(); // drives the mode to VisibleOnly
        p.reset();

        p.initialize_request(recover_init(
            UnifiedParserStartingState::None,
            UnifiedToolOutputMode::GuidedJson { named_tool: None },
        ))
        .unwrap();
        let out = assemble(
            &[p.push(r#"<think>thinking</think>[{"name":"get_weather","arguments":{"city":"Paris"}}]"#).unwrap(),
              p.finish().unwrap().events].concat());
        assert!(
            out.iter()
                .any(|e| matches!(e, UnifiedEvent::Reasoning { .. })),
            "reasoning was swallowed as payload after reset: {out:?}"
        );
        assert!(
            out.iter()
                .any(|e| matches!(e, UnifiedEvent::ToolCall { .. })),
            "call not recovered after reset: {out:?}"
        );
    }

    /// A named choice constrains output to that tool's ARGUMENTS, which are an object.
    /// A bare scalar or array is valid JSON but not an argument set, so dispatching it
    /// would hand the tool a shape it cannot bind.
    #[test]
    fn named_choice_rejects_a_non_object_payload() {
        for payload in [r#""just a string""#, "42", "null", "[1,2]"] {
            let mut p = qwen3_unified(&tools());
            p.initialize_request(recover_init(
                UnifiedParserStartingState::None,
                UnifiedToolOutputMode::GuidedJson {
                    named_tool: Some("get_weather".to_string()),
                },
            ))
            .unwrap();
            let mut out = p.push(payload).unwrap();
            out.extend(p.finish().unwrap().events);
            assert!(
                out.iter()
                    .all(|d| !matches!(d, UnifiedParserEvent::ToolCall(_))),
                "{payload}: dispatched a non-object payload as tool arguments"
            );
            assert!(
                out.iter()
                    .any(|d| matches!(d, UnifiedParserEvent::Text(..))),
                "{payload}: payload was dropped instead of surfaced as text"
            );
        }
    }

    /// Guided must agree with native on `UNIFIED.6.a`: a call with no argument key
    /// is a parameterless call, not a malformed one — and inside an array it must not
    /// take its siblings down with it.
    #[test]
    fn a_parameterless_guided_call_is_dispatched_and_does_not_void_its_siblings() {
        let mut p = qwen3_unified(&tools());
        p.initialize_request(recover_init(
            UnifiedParserStartingState::None,
            UnifiedToolOutputMode::GuidedJson { named_tool: None },
        ))
        .unwrap();
        let mut out = p
            .push(r#"[{"name":"get_weather"},{"name":"get_weather","arguments":{"city":"Paris"}}]"#)
            .unwrap();
        out.extend(p.finish().unwrap().events);
        let calls: Vec<_> = out
            .iter()
            .filter_map(|d| match d {
                UnifiedParserEvent::ToolCall(c) => Some((c.name.clone(), c.arguments.clone())),
                _ => None,
            })
            .collect();
        assert_eq!(calls.len(), 2, "a no-arg call voided the array: {out:?}");
        assert_eq!(
            calls[0].1, "{}",
            "no-arg call did not get an empty argument set: {calls:?}"
        );
    }

    /// The object case must still work.
    #[test]
    fn named_choice_still_accepts_an_object_payload() {
        let mut p = qwen3_unified(&tools());
        p.initialize_request(recover_init(
            UnifiedParserStartingState::None,
            UnifiedToolOutputMode::GuidedJson {
                named_tool: Some("get_weather".to_string()),
            },
        ))
        .unwrap();
        let mut out = p.push(r#"{"city": "Paris"}"#).unwrap();
        out.extend(p.finish().unwrap().events);
        assert!(
            out.iter()
                .any(|d| matches!(d, UnifiedParserEvent::ToolCall(_))),
            "object payload not dispatched: {out:?}"
        );
    }

    #[test]
    fn reject_policy_returns_typed_errors_for_every_invalid_payload_class() {
        for (named_tool, payload, expected_kind) in [
            (None, r#"not json"#, InvalidGuidedPayloadKind::InvalidJson),
            (None, r#"{}"#, InvalidGuidedPayloadKind::WrongShape),
            (
                None,
                r#"[{"name":"get_weather"},{"arguments":{}}]"#,
                InvalidGuidedPayloadKind::WrongShape,
            ),
            (
                None,
                r#"{"name":"get_weather","arguments":7}"#,
                InvalidGuidedPayloadKind::WrongShape,
            ),
            (
                Some("get_weather"),
                r#"[1,2]"#,
                InvalidGuidedPayloadKind::WrongShape,
            ),
        ] {
            let mut parser = qwen3_unified(&tools());
            parser
                .initialize_request(guided_init(named_tool, InvalidGuidedPayloadPolicy::Reject))
                .unwrap();
            let error = match parser.push(payload) {
                Ok(_) => parser.finish().expect_err("invalid payload must reject"),
                Err(error) => error,
            };
            let typed = error
                .downcast_ref::<InvalidGuidedPayload>()
                .expect("caller must be able to downcast guided failures");
            assert_eq!(typed.kind, expected_kind, "payload={payload}");
            assert_eq!(
                typed.choice,
                if named_tool.is_some() {
                    "named"
                } else {
                    "required"
                }
            );
            assert!(
                !error.to_string().contains(payload),
                "typed error leaked raw model output"
            );
        }
    }

    #[test]
    fn recover_policy_preserves_invalid_payload_as_text() {
        let mut parser = qwen3_unified(&tools());
        parser
            .initialize_request(guided_init(None, InvalidGuidedPayloadPolicy::RecoverAsText))
            .unwrap();
        let payload = r#"{"arguments":{}}"#;
        let mut output = parser.push(payload).unwrap();
        output.extend(parser.finish().unwrap().events);
        assert_eq!(output, vec![UnifiedParserEvent::Text(payload.into())]);
    }

    #[test]
    fn reject_keeps_reasoning_committed_and_buffer_recoverable() {
        let mut parser = qwen3_unified(&tools());
        parser
            .initialize_request(guided_init(None, InvalidGuidedPayloadPolicy::Reject))
            .unwrap();
        let mut output = UnifiedParserOutput::default();
        let error = parser
            .parse_into(r#"<think>considering</think>{"arguments":{}}"#, &mut output)
            .expect_err("wrong-shape payload must reject");
        assert!(error.downcast_ref::<InvalidGuidedPayload>().is_some());
        assert_eq!(
            output.events,
            vec![UnifiedParserEvent::Reasoning("considering".into())]
        );
        assert_eq!(parser.reset(), r#"{"arguments":{}}"#);
    }
}
