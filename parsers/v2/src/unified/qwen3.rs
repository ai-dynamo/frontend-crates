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
//! [`crate::tool_calling::scan::WrappedBlockScanner`] the tool-only
//! `Qwen3CoderToolStreamParser` runs on — block open/close, bare-`<function=>`
//! recovery, orphan-close stripping, chunk-boundary holdback and EOF drop stay a
//! single implementation, so the unified path cannot quietly regress the tool
//! handling the tool-only suite already pins. Value typing likewise delegates to
//! the shared batch XML parser, so a streamed argument object matches the batch
//! one exactly. The `UnifiedParser` impl itself is generic
//! ([`crate::unified::ScannerUnified`]).
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
    // `preserve_special_tokens` must match the TOOL-ONLY parser for this same
    // grammar (`Qwen3CoderToolStreamParser` answers `true`): both read the same
    // markers, so a host told `false` here would strip them and the calls would
    // vanish on the unified path only.
    Box::new(ScannerUnified::new(
        qwen3_scanner(tools).with_reasoning(ReasoningSpec {
            start: REASONING_START,
            end: REASONING_END,
            // Qwen3 emits its own `<think>`; the template does not pre-fill one,
            // so the stream starts in visible content (policy P5).
            forced_start: false,
        }),
        true,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::unified::{
        UnifiedDelta, UnifiedEvent, UnifiedParserPrefill, UnifiedToolOutputMode, assemble,
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
        deltas.extend(parser.finish().expect("finish"));
        assemble(&deltas)
    }

    fn configured_events(
        tools: &[Tool],
        prefill: UnifiedParserPrefill,
        tool_output_mode: UnifiedToolOutputMode<'_>,
        chunks: &[&str],
    ) -> Vec<UnifiedEvent> {
        let mut parser = qwen3_unified(tools);
        parser
            .initialize_with_output_mode(prefill, tool_output_mode)
            .expect("initialize");
        let mut deltas = Vec::new();
        for chunk in chunks {
            deltas.extend(parser.push(chunk).expect("push"));
        }
        deltas.extend(parser.finish().expect("finish"));
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
            .initialize_with_output_mode(
                UnifiedParserPrefill::None,
                UnifiedToolOutputMode::GuidedJson { named_tool: None },
            )
            .unwrap();
        let mut deltas = parser.push(chunk).unwrap();
        deltas.extend(parser.finish().unwrap());
        assemble(&deltas)
    }

    const GUIDED_CALL: &str = r#"[{"name": "get_weather", "arguments": {"city": "Paris"}}]"#;

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
        // The invariant, stated directly: the request mode must not change the
        // reasoning payload. Native ends with a tool call in markup, guided with the
        // same call as JSON, so only the FIRST event is comparable.
        for thought in [
            "<think>a<think>b</think>",
            "<think>a</tool_call>b</think>",
            // A tool OPENER inside a thought: native ends the span there, so guided
            // must too. Reviewed catch — the first two cases passed while this one
            // leaked `<tool_call>` into the displayed thinking on the guided path.
            "<think>a<tool_call>x</think>",
            "<think>a<function=run>x</think>",
            "<think>plain thought</think>",
        ] {
            let native = events(&weather_tools(), &[&format!("{thought}tail")]);
            let guided = guided_reasoning(&format!("{thought}{GUIDED_CALL}"));
            assert_eq!(
                native[0], guided[0],
                "request mode changed the reasoning payload for {thought:?}"
            );
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
            .initialize_with_output_mode(
                UnifiedParserPrefill::None,
                UnifiedToolOutputMode::GuidedJson { named_tool: None },
            )
            .unwrap();
        let mut deltas = Vec::new();
        for chunk in ["<think>a</tool", "_call>b</think>", GUIDED_CALL] {
            deltas.extend(parser.push(chunk).unwrap());
        }
        deltas.extend(parser.finish().unwrap());
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
            UnifiedParserPrefill::Reasoning,
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
                named_tool: Some("get_weather"),
            },
        ] {
            let out = configured_events(
                &weather_tools(),
                UnifiedParserPrefill::Reasoning,
                tool_output_mode,
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
            UnifiedParserPrefill::Response,
            UnifiedToolOutputMode::Native,
            &["<think>literal</think>"],
        );
        assert_eq!(out, vec![text("<think>literal</think>")]);
    }

    #[test]
    fn named_choice_parses_bare_arguments_object() {
        let out = configured_events(
            &weather_tools(),
            UnifiedParserPrefill::Reasoning,
            UnifiedToolOutputMode::GuidedJson {
                named_tool: Some("get_weather"),
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
            UnifiedParserPrefill::None,
            UnifiedToolOutputMode::GuidedJson {
                named_tool: Some("get_weather"),
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
            UnifiedParserPrefill::Reasoning,
            UnifiedToolOutputMode::GuidedJson {
                named_tool: Some("get_weather"),
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
            UnifiedParserPrefill::Response,
            UnifiedToolOutputMode::GuidedJson { named_tool: None },
            &[r#"{"name":"get_weather","parameters":{"city":"Tokyo"}}"#],
        );
        assert_eq!(
            single,
            vec![call("get_weather", serde_json::json!({"city": "Tokyo"}))]
        );

        let parallel = configured_events(
            &weather_tools(),
            UnifiedParserPrefill::Response,
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
            UnifiedParserPrefill::Response,
            UnifiedToolOutputMode::GuidedJson { named_tool: None },
            &[input],
        );
        assert_eq!(out, vec![text(input)]);
    }

    #[test]
    fn native_mode_keeps_xml_under_reasoning_prefill() {
        let out = configured_events(
            &weather_tools(),
            UnifiedParserPrefill::Reasoning,
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
            UnifiedParserPrefill::Reasoning,
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
    fn incomplete_guided_json_recovers_as_text() {
        for tool_output_mode in [
            UnifiedToolOutputMode::GuidedJson {
                named_tool: Some("get_weather"),
            },
            UnifiedToolOutputMode::GuidedJson { named_tool: None },
        ] {
            let input = r#"{"city":"Tok"#;
            let out = configured_events(
                &weather_tools(),
                UnifiedParserPrefill::Response,
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
            .initialize_with_output_mode(
                UnifiedParserPrefill::Reasoning,
                UnifiedToolOutputMode::GuidedJson {
                    named_tool: Some("get_weather"),
                },
            )
            .unwrap();
        assert_eq!(
            parser.push(r#"reason</think>{"city":"Tok"#).unwrap(),
            vec![UnifiedDelta::Reasoning {
                text: "reason".to_string()
            }]
        );
        assert!(parser.initialize(UnifiedParserPrefill::Response).is_err());
        assert_eq!(parser.reset(), r#"{"city":"Tok"#);

        parser
            .initialize_with_output_mode(
                UnifiedParserPrefill::Response,
                UnifiedToolOutputMode::GuidedJson {
                    named_tool: Some("get_weather"),
                },
            )
            .unwrap();
        let mut deltas = parser.push(r#"{"city":"Tokyo"}"#).unwrap();
        deltas.extend(parser.finish().unwrap());
        assert_eq!(
            assemble(&deltas),
            vec![call("get_weather", serde_json::json!({"city": "Tokyo"}))]
        );
        assert!(parser.finish().is_err());
        assert!(parser.push("after finish").is_err());
    }
}

#[cfg(test)]
mod guided_warning_tests {
    use super::*;
    use crate::tool_calling::traits::Tool;
    use crate::unified::{UnifiedDelta, UnifiedParserPrefill, UnifiedToolOutputMode};

    fn weather_tools() -> Vec<Tool> {
        vec![Tool {
            name: "get_weather".into(),
            description: None,
            parameters: serde_json::json!({"type":"object","properties":{"city":{"type":"string"}}}),
            strict: None,
        }]
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
            p.initialize_with_output_mode(
                UnifiedParserPrefill::None,
                UnifiedToolOutputMode::GuidedJson { named_tool: None },
            )
            .unwrap();
            p.push(payload).unwrap();
            let out = p.finish().unwrap();
            // recovered as text, no call dispatched
            assert!(
                out.iter().all(|d| !matches!(d, UnifiedDelta::ToolCall(_))),
                "{label}: dispatched a call from an unvalidated payload"
            );
            assert!(
                out.iter().any(|d| matches!(d, UnifiedDelta::Text { .. })),
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

        tracing::subscriber::with_default(sub, || {
            let mut p = qwen3_unified(&weather_tools());
            p.initialize_with_output_mode(
                UnifiedParserPrefill::None,
                UnifiedToolOutputMode::GuidedJson { named_tool: None },
            )
            .unwrap();
            p.push(r#"{"unexpected": "shape"}"#).unwrap();
            p.finish().unwrap();
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
    }
}

#[cfg(test)]
mod reset_and_payload_tests {
    use super::*;
    use crate::tool_calling::traits::Tool;
    use crate::unified::{
        UnifiedDelta, UnifiedEvent, UnifiedParserPrefill, UnifiedToolOutputMode, assemble,
    };

    fn tools() -> Vec<Tool> {
        vec![Tool {
            name: "get_weather".into(),
            description: None,
            parameters: serde_json::json!({"type":"object","properties":{"city":{"type":"string"}}}),
            strict: None,
        }]
    }

    /// `reset` must clear "resume reasoning after the interrupting call". Leaving it
    /// armed made the NEXT stream's first post-call answer come out as reasoning —
    /// the user's visible answer silently becomes private thinking.
    #[test]
    fn reset_does_not_leak_resume_reasoning_into_the_next_stream() {
        let mut p = qwen3_unified(&tools());
        p.initialize(UnifiedParserPrefill::None).unwrap();
        // interrupt a thought with a call, then reset mid-stream
        p.push("<think>weighing<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>")
            .unwrap();
        p.reset();

        p.initialize(UnifiedParserPrefill::None).unwrap();
        let out = assemble(
            &[p.push("<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>visible answer").unwrap(),
              p.finish().unwrap()].concat());
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
        p.initialize_with_output_mode(
            UnifiedParserPrefill::None,
            UnifiedToolOutputMode::GuidedJson { named_tool: None },
        )
        .unwrap();
        p.push(r#"{"partial"#).unwrap(); // drives the mode to VisibleOnly
        p.reset();

        p.initialize_with_output_mode(
            UnifiedParserPrefill::None,
            UnifiedToolOutputMode::GuidedJson { named_tool: None },
        )
        .unwrap();
        let out = assemble(
            &[p.push(r#"<think>thinking</think>[{"name":"get_weather","arguments":{"city":"Paris"}}]"#).unwrap(),
              p.finish().unwrap()].concat());
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
            p.initialize_with_output_mode(
                UnifiedParserPrefill::None,
                UnifiedToolOutputMode::GuidedJson {
                    named_tool: Some("get_weather"),
                },
            )
            .unwrap();
            p.push(payload).unwrap();
            let out = p.finish().unwrap();
            assert!(
                out.iter().all(|d| !matches!(d, UnifiedDelta::ToolCall(_))),
                "{payload}: dispatched a non-object payload as tool arguments"
            );
            assert!(
                out.iter().any(|d| matches!(d, UnifiedDelta::Text { .. })),
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
        p.initialize_with_output_mode(
            UnifiedParserPrefill::None,
            UnifiedToolOutputMode::GuidedJson { named_tool: None },
        )
        .unwrap();
        p.push(r#"[{"name":"get_weather"},{"name":"get_weather","arguments":{"city":"Paris"}}]"#)
            .unwrap();
        let out = p.finish().unwrap();
        let calls: Vec<_> = out
            .iter()
            .filter_map(|d| match d {
                UnifiedDelta::ToolCall(c) => Some((c.name.clone(), c.arguments.clone())),
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
        p.initialize_with_output_mode(
            UnifiedParserPrefill::None,
            UnifiedToolOutputMode::GuidedJson {
                named_tool: Some("get_weather"),
            },
        )
        .unwrap();
        p.push(r#"{"city": "Paris"}"#).unwrap();
        let out = p.finish().unwrap();
        assert!(
            out.iter().any(|d| matches!(d, UnifiedDelta::ToolCall(_))),
            "object payload not dispatched: {out:?}"
        );
    }
}
