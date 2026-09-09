// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Kimi K2 on the UnifiedParser.
//!
//! The whole family is the shared scanner plus a `ReasoningSpec`. Kimi's tool
//! grammar is a plain wrapped block (`<|tool_calls_section_begin|>` … ) and its
//! reasoning channel is `<think>`/`</think>` — the same markers qwen3 uses — so
//! this port needs NOTHING that the scanner does not already have. That is the
//! interesting result: gemma4's `invoke_scan` and `start_label` are gemma4's,
//! not a general cost of porting a family.

use crate::tool_calling::kimi_k2::kimi_k2_scanner;
use crate::tool_calling::scan::ReasoningSpec;
use crate::tool_calling::traits::Tool;
use crate::unified::{GuidedRouted, ScannerUnified, UnifiedParser};

const REASONING_START: &str = "<think>";
const REASONING_END: &str = "</think>";

/// Build the Kimi K2 unified parser for one stream.
pub(crate) fn kimi_k2_unified(tools: &[Tool]) -> Box<dyn UnifiedParser> {
    Box::new(GuidedRouted::new(ScannerUnified::new(
        kimi_k2_scanner(tools).with_reasoning(ReasoningSpec {
            start: REASONING_START,
            end: REASONING_END,
            // Kimi emits its own `<think>`; the template does not pre-fill one,
            // so the stream starts in visible content (policy P5).
            forced_start: false,
            // `<think>` is not a tokenizer special token for this family; the
            // grammar's own markers carry that requirement via the block spec.
            preserve_special_tokens: false,
            ..Default::default()
        }),
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::unified::{
        InvalidGuidedPayloadPolicy, UnifiedEvent, UnifiedParserEvent, UnifiedParserExt,
        UnifiedParserInit, UnifiedParserStartingState, UnifiedToolOutputMode, assemble,
    };

    fn tools() -> Vec<Tool> {
        vec![Tool {
            name: "get_weather".to_string(),
            description: None,
            parameters: serde_json::json!({"type":"object","properties":{"city":{"type":"string"}}}),
            strict: None,
        }]
    }

    fn run_tool() -> Vec<Tool> {
        vec![Tool {
            name: "run".to_string(),
            description: None,
            parameters: serde_json::json!({"type":"object","properties":{"cmd":{"type":"string"}}}),
            strict: None,
        }]
    }

    /// A `<|tool_call_end|>` found before any `<|tool_call_argument_begin|>`
    /// used to be trusted immediately, with no `flush` gate -- but streaming
    /// only ever appends bytes, so a legitimate `argument_begin` that just
    /// hasn't arrived yet can still turn up later and change the correct
    /// reading entirely. Same final input, split only at whether the real
    /// `argument_begin` had streamed in yet: one push dropped the call
    /// silently, two pushes leaked the raw JSON as visible text. Every split
    /// point must agree with the whole-input result.
    #[test]
    fn call_end_before_argument_begin_does_not_depend_on_the_chunk_boundary() {
        let input = "<|tool_call_begin|>functions.get_weather:0<|tool_call_end|><|tool_call_argument_begin|>{\"city\": \"NYC\"}<|tool_call_end|>";
        let want = assemble_at_every_split(&tools(), input)
            .into_iter()
            .next()
            .expect("at least one split");
        for (i, got) in assemble_at_every_split(&tools(), input)
            .into_iter()
            .enumerate()
        {
            assert_eq!(got, want, "split at byte {i}, got {got:?}, want {want:?}");
        }
    }

    /// A bare invoke that closes with its own `call_end` before EVER having
    /// an `argument_begin`, immediately followed by a real, well-formed
    /// second invoke. Model output is probabilistic and can violate its own
    /// grammar this way even though the downstream batch regex and every
    /// golden fixture require `argument_begin` unconditionally -- the
    /// streaming boundary finder still has to survive it rather than let the
    /// second invoke's `argument_begin` get matched to the first invoke's
    /// span. That merged both invokes into one string handed to the typing
    /// layer, which only ever returns its first parsed call, so the
    /// malformed first invoke silently absorbed the second invoke's bytes
    /// and the second call vanished with it. The malformed first invoke has
    /// no valid recovery (no argument section ever existed for it) and is
    /// correctly dropped; the second, valid call must still ship on its own.
    #[test]
    fn bare_close_before_argument_begin_does_not_swallow_the_next_invoke() {
        let tools = vec![tools()[0].clone(), run_tool()[0].clone()];
        let input = "<|tool_calls_section_begin|><|tool_call_begin|>functions.run:0<|tool_call_end|><|tool_call_begin|>functions.get_weather:1<|tool_call_argument_begin|>{\"city\": \"Paris\"}<|tool_call_end|><|tool_calls_section_end|>";
        let want = vec![UnifiedEvent::ToolCall {
            name: "get_weather".into(),
            arguments: serde_json::json!({"city": "Paris"}),
        }];
        for (i, got) in assemble_at_every_split(&tools, input)
            .into_iter()
            .enumerate()
        {
            assert_eq!(got, want, "split at byte {i}, got {got:?}");
        }
    }

    /// Same malformed shape, but at true EOF with no `<|tool_calls_section_end|>`
    /// ever arriving -- the second (valid) call must still recover via the
    /// same best-effort EOF path `UNIFIED.5.b` already covers, not get lost
    /// because the first invoke's bytes were merged into its span.
    #[test]
    fn bare_close_before_argument_begin_still_recovers_the_second_call_at_finish() {
        let input = "<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_end|><|tool_call_begin|>functions.run:1<|tool_call_argument_begin|>{\"cmd\": \"ls\"}";
        let mut p = kimi_k2_unified(&run_tool());
        let mut ev = p.push(input).expect("push");
        ev.extend(p.finish().expect("finish").events);
        let got = assemble(&ev);
        assert_eq!(
            got,
            vec![UnifiedEvent::ToolCall {
                name: "run".into(),
                arguments: serde_json::json!({"cmd": "ls"}),
            }],
            "got {got:?}"
        );
    }

    /// The batch-mode typing layer (`parse_section_block`) has its own
    /// raw-string fallback for an argument body that never parses as JSON --
    /// `serde_json::from_str` fails and it ships the raw text verbatim
    /// rather than rejecting the call. The streaming boundary finder used to
    /// short-circuit on the SAME shape (`json_value_end` returning `None`)
    /// and drop the whole invoke before it ever reached that fallback --
    /// this pins that the call still ships (best-effort P3, not a silent
    /// drop) once the family's own literal `call_end` is present.
    #[test]
    fn malformed_non_json_arguments_still_ship_the_call_instead_of_vanishing() {
        let input = "<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>not-json-at-all<|tool_call_end|><|tool_calls_section_end|>";
        let mut p = kimi_k2_unified(&tools());
        let mut ev = p.push(input).expect("push");
        ev.extend(p.finish().expect("finish").events);
        assert_eq!(
            assemble(&ev),
            vec![UnifiedEvent::ToolCall {
                name: "get_weather".into(),
                // P3 best-effort (assemble's own documented fallback): a raw
                // argument string that doesn't parse as JSON lands as an
                // empty object rather than being discarded -- the call
                // itself is what must survive here, not this specific value.
                arguments: serde_json::json!({}),
            }]
        );
    }

    fn assert_malformed_quoted_marker_emits_before_finish_at_every_split(marker: &str) {
        let header = "<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>";
        let arguments = format!(r#"{{"location" "München {marker} literal"}}"#);
        let input = format!("{header}{arguments}<|tool_call_end|><|tool_calls_section_end|>");
        for split in (0..=input.len()).filter(|&index| input.is_char_boundary(index)) {
            let mut parser = kimi_k2_unified(&tools());
            let mut emitted = parser.push(&input[..split]).unwrap();
            emitted.extend(parser.push(&input[split..]).unwrap());
            let calls: Vec<_> = emitted
                .iter()
                .filter_map(|event| match event {
                    UnifiedParserEvent::ToolCall(call) => Some(call),
                    _ => None,
                })
                .collect();
            assert_eq!(
                calls.len(),
                1,
                "marker {marker:?}, split at byte {split} must emit once the closer is available"
            );
            assert_eq!(calls[0].arguments, arguments);
            assert!(
                parser.finish().unwrap().events.is_empty(),
                "marker {marker:?}, split at byte {split} must not defer the call until finish"
            );
        }
    }

    fn assert_malformed_quoted_marker_emits_on_the_closing_chunk(marker: &str) {
        let arguments = format!(r#"{{"location" "München {marker} literal"}}"#);
        let mut parser = kimi_k2_unified(&tools());
        assert!(
            parser
                .push("<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>")
                .unwrap()
                .is_empty()
        );
        let emitted = parser
            .push(&format!(
                "{arguments}<|tool_call_end|><|tool_calls_section_end|>"
            ))
            .unwrap();
        assert_eq!(
            emitted,
            vec![UnifiedParserEvent::ToolCall(
                crate::tool_calling::traits::ToolCallDelta {
                    tool_index: 0,
                    name: Some("get_weather".to_string()),
                    arguments,
                    complete: true,
                }
            )]
        );
        assert!(parser.finish().unwrap().events.is_empty());
    }

    #[test]
    fn malformed_quoted_call_end_emits_on_the_closing_chunk() {
        assert_malformed_quoted_marker_emits_on_the_closing_chunk("<|tool_call_end|>");
    }

    #[test]
    fn malformed_quoted_section_end_emits_on_the_closing_chunk() {
        assert_malformed_quoted_marker_emits_on_the_closing_chunk("<|tool_calls_section_end|>");
    }

    #[test]
    fn malformed_quoted_call_end_emits_before_finish_at_every_valid_utf8_split() {
        assert_malformed_quoted_marker_emits_before_finish_at_every_split("<|tool_call_end|>");
    }

    #[test]
    fn malformed_quoted_section_end_emits_before_finish_at_every_valid_utf8_split() {
        assert_malformed_quoted_marker_emits_before_finish_at_every_split(
            "<|tool_calls_section_end|>",
        );
    }

    /// This stays a unit-level property test because the fixture corpus uses
    /// fixed delivery schedules and normalized JSON output; it cannot sweep
    /// every UTF-8 split or preserve malformed raw argument bytes.
    #[test]
    fn adjacent_malformed_calls_stay_separate_at_every_valid_utf8_split() {
        let tools = vec![tools()[0].clone(), run_tool()[0].clone()];
        let input = "<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>x\"1<|tool_call_end|><|tool_call_begin|>functions.run:1<|tool_call_argument_begin|>y\"2<|tool_call_end|><|tool_calls_section_end|>";
        let first = UnifiedParserEvent::ToolCall(crate::tool_calling::traits::ToolCallDelta {
            tool_index: 0,
            name: Some("get_weather".to_string()),
            arguments: "x\"1".to_string(),
            complete: true,
        });
        let second = UnifiedParserEvent::ToolCall(crate::tool_calling::traits::ToolCallDelta {
            tool_index: 1,
            name: Some("run".to_string()),
            arguments: "y\"2".to_string(),
            complete: true,
        });

        for split in (0..=input.len()).filter(|&index| input.is_char_boundary(index)) {
            let mut parser = kimi_k2_unified(&tools);
            let mut emitted = parser.push(&input[..split]).unwrap();
            emitted.extend(parser.push(&input[split..]).unwrap());
            assert_eq!(emitted, vec![first.clone()], "split at byte {split}");
            let finished = parser.finish().unwrap().events;
            assert_eq!(finished, vec![second.clone()], "split at byte {split}");
        }
    }

    /// Every valid UTF-8 split point of `input`, pushed as separate chunks
    /// then finished, assembled into one ordered event list. Sweeps chunk
    /// boundaries around every control marker rather than trusting one
    /// hand-picked split (gate: "derive cases from the property").
    fn assemble_at_every_split(tools: &[Tool], input: &str) -> Vec<Vec<UnifiedEvent>> {
        (0..=input.len())
            .filter(|&i| input.is_char_boundary(i))
            .map(|i| {
                let mut p = kimi_k2_unified(tools);
                let mut ev = p.push(&input[..i]).expect("push prefix");
                ev.extend(p.push(&input[i..]).expect("push suffix"));
                ev.extend(p.finish().expect("finish").events);
                assemble(&ev)
            })
            .collect()
    }

    /// `UNIFIED.5.b` (`tool_no_close`): the JSON argument body is complete but
    /// `<|tool_call_end|>` never streams before EOF. Best-effort recovery
    /// (policy P2 sibling) must still emit the call at `finish`, identically
    /// for a single push and for every chunk-split point -- not just recover
    /// at EOF and then silently regress mid-stream for some split.
    #[test]
    fn tool_no_close_recovers_the_call_at_finish_across_every_split() {
        let input = "<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{\"city\": \"Paris\"}";
        let want = vec![UnifiedEvent::ToolCall {
            name: "get_weather".into(),
            arguments: serde_json::json!({"city":"Paris"}),
        }];
        for (i, got) in assemble_at_every_split(&tools(), input)
            .into_iter()
            .enumerate()
        {
            assert_eq!(got, want, "split at byte {i}, got {got:?}");
        }
    }

    /// Negative control for the same property: a body that is genuinely
    /// truncated mid-JSON (never balances) must still drop the call, not be
    /// swept up by the new EOF-recovery path. Mirrors
    /// `tool_calling::kimi_k2::tests::suppresses_truncated_call_at_eof` at the
    /// unified layer.
    #[test]
    fn genuinely_truncated_json_still_drops_at_finish() {
        let input = "<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{\"city\": \"NY";
        let mut p = kimi_k2_unified(&tools());
        let mut ev = p.push(input).expect("push");
        ev.extend(p.finish().expect("finish").events);
        let got = assemble(&ev);
        assert_eq!(got, vec![], "truncated body must still drop, got {got:?}");
    }

    /// `UNIFIED.7.b` (`arg_marker_in_string`): a `<|tool_call_end|>`-looking
    /// byte sequence sits INSIDE the quoted JSON string argument. Invariant
    /// I7 -- it is data, preserved byte-exact, never mistaken for the real
    /// closer -- identically for a single push and every chunk-split point,
    /// including splits that land INSIDE the embedded marker itself.
    #[test]
    fn arg_marker_in_string_survives_byte_exact_across_every_split() {
        let input = "<|tool_calls_section_begin|><|tool_call_begin|>functions.run:0<|tool_call_argument_begin|>{\"cmd\": \"git log <|tool_call_end|> --oneline\"}<|tool_call_end|><|tool_calls_section_end|>";
        let want = vec![UnifiedEvent::ToolCall {
            name: "run".into(),
            arguments: serde_json::json!({"cmd": "git log <|tool_call_end|> --oneline"}),
        }];
        for (i, got) in assemble_at_every_split(&run_tool(), input)
            .into_iter()
            .enumerate()
        {
            assert_eq!(got, want, "split at byte {i}, got {got:?}");
        }
    }

    /// `UNIFIED.7.b` (`arg_marker_in_string`) with ordinary whitespace between
    /// the JSON body and the real `<|tool_call_end|>`. The byte-exact override
    /// in `parse_section_block` (v1core) used to require the closer to
    /// immediately abut the JSON (`starts_with`, no whitespace tolerance),
    /// while the regex it patches over -- and the streaming boundary finder,
    /// `kimi_invoke_end` -- both already tolerate `\s*` there. Any whitespace
    /// silently fell back to the lazy regex capture, which stops at the
    /// EMBEDDED fake closer inside the string and ships either a truncated,
    /// unparseable argument or (once `assemble`'s P3 fallback catches the
    /// unparseable string) an empty object -- losing the whole call.
    #[test]
    fn arg_marker_in_string_survives_whitespace_before_the_real_closer_across_every_split() {
        let input = "<|tool_calls_section_begin|><|tool_call_begin|>functions.run:0<|tool_call_argument_begin|>{\"cmd\": \"git log <|tool_call_end|> --oneline\"} <|tool_call_end|><|tool_calls_section_end|>";
        let want = vec![UnifiedEvent::ToolCall {
            name: "run".into(),
            arguments: serde_json::json!({"cmd": "git log <|tool_call_end|> --oneline"}),
        }];
        for (i, got) in assemble_at_every_split(&run_tool(), input)
            .into_iter()
            .enumerate()
        {
            assert_eq!(got, want, "split at byte {i}, got {got:?}");
        }
    }

    /// The corpus case `UNIFIED.11.b`: a thought, a call, a second thought, an
    /// answer. This is the ordering the split path cannot represent — it hoists
    /// both thoughts to the front and fuses them.
    #[test]
    fn thought_call_thought_answer_keeps_its_order() {
        let input = "<think>Look it up.</think><|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{\"city\": \"Paris\"}<|tool_call_end|><|tool_calls_section_end|><think>Now answer.</think>It's 18C.";
        let mut p = kimi_k2_unified(&tools());
        let mut ev = p.push(input).expect("push");
        ev.extend(p.finish().expect("finish").events);
        let got = assemble(&ev);
        assert_eq!(
            got,
            vec![
                UnifiedEvent::Reasoning {
                    text: "Look it up.".into()
                },
                UnifiedEvent::ToolCall {
                    name: "get_weather".into(),
                    arguments: serde_json::json!({"city":"Paris"})
                },
                UnifiedEvent::Reasoning {
                    text: "Now answer.".into()
                },
                UnifiedEvent::Text {
                    text: "It's 18C.".into()
                },
            ],
            "got {got:?}"
        );
    }

    /// Every valid UTF-8 split point of `input`, pushed as separate chunks
    /// then finished under GuidedJson mode with `RecoverAsText`. Mirrors
    /// `assemble_at_every_split` but for the guided path, which needs
    /// `initialize_request` before any bytes arrive.
    fn assemble_guided_at_every_split(tools: &[Tool], input: &str) -> Vec<Vec<UnifiedEvent>> {
        (0..=input.len())
            .filter(|&i| input.is_char_boundary(i))
            .map(|i| {
                let mut p = kimi_k2_unified(tools);
                p.initialize_request(UnifiedParserInit {
                    starting_state: UnifiedParserStartingState::None,
                    tool_output_mode: UnifiedToolOutputMode::GuidedJson { named_tool: None },
                    invalid_guided_payload: InvalidGuidedPayloadPolicy::RecoverAsText,
                    ..UnifiedParserInit::default()
                })
                .expect("initialize");
                let mut ev = p.push(&input[..i]).expect("push prefix");
                ev.extend(p.push(&input[i..]).expect("push suffix"));
                ev.extend(p.finish().expect("finish").events);
                assemble(&ev)
            })
            .collect()
    }

    /// Proves `tool_index: 0` is safe at `GuidedState::control_marker_at`'s
    /// two `scan.end` call sites (`unified/mod.rs`) even for the disputed
    /// "second invoke" shape, not just a lone one. Two structural facts make
    /// this safe together:
    /// (1) `kimi_invoke_opens` (`tool_calling/kimi_k2.rs`) always returns
    /// `true`, so `control_marker_at`'s scan loop returns on the FIRST
    /// `invoke_start` occurrence in whatever `haystack` it searches -- the
    /// `cursor = at + ...` advance to look for a second occurrence is
    /// structurally unreachable for Kimi, so this hook can never itself walk
    /// past a first marker to inspect a second one.
    /// (2) The result only classifies a native-looking marker inside a
    /// thought as stray markup to strip; it never types or emits a call --
    /// that happens entirely through the independent, array-indexed
    /// `GuidedJsonCursor`/`emit_completed_json` path.
    /// Concretely: both a properly closed first native marker and an
    /// incomplete second one are stripped from reasoning, while the real
    /// guided payload after both still types exactly once.
    #[test]
    fn two_native_markers_in_guided_reasoning_are_stripped_before_the_guided_payload() {
        let input = "<think>First: <|tool_call_begin|>functions.run:0<|tool_call_argument_begin|>{\"cmd\":\"ok\"}<|tool_call_end|> Second: <|tool_call_begin|>functions.other:1<|tool_call_argument_begin|>{\"cmd\":\"partial</think>[{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}]";
        let want = vec![
            UnifiedEvent::Reasoning {
                text: "First:  Second: ".into(),
            },
            UnifiedEvent::ToolCall {
                name: "get_weather".into(),
                arguments: serde_json::json!({"city": "Paris"}),
            },
        ];
        for (i, got) in assemble_guided_at_every_split(&tools(), input)
            .into_iter()
            .enumerate()
        {
            assert_eq!(got, want, "split at byte {i}, got {got:?}");
        }
    }

    /// The fixture corpus does not cover native envelopes inside guided
    /// output, so this unit property sweeps every split and checks the public
    /// event stream directly: an unrecoverable partial call is dropped at
    /// EOF rather than emitted as visible recovery text.
    #[test]
    fn incomplete_native_envelope_in_guided_mode_does_not_leak_at_finish() {
        let input =
            "<|tool_call_begin|>functions.other:1<|tool_call_argument_begin|>{\"cmd\":\"partial";
        for (i, got) in assemble_guided_at_every_split(&tools(), input)
            .into_iter()
            .enumerate()
        {
            assert!(got.is_empty(), "split at byte {i}, got {got:?}");
        }
    }

    /// `UNIFIED.guided_json_gt_in_argument_bare_opener.kimi_k2`: guided
    /// decoding constrains the payload to JSON, but the model still narrates
    /// a bare `<|tool_call_begin|>functions.` header with no `NAME:IDX`, no
    /// `argument_begin`, and no `call_end` -- nothing `kimi_invoke_end` used
    /// to treat as a resolvable boundary, so the header AND the JSON payload
    /// after it leaked as one blob of visible text instead of the header
    /// being stripped and the JSON being parsed as the call.
    #[test]
    fn bare_opener_before_guided_json_strips_the_header_and_recovers_the_call() {
        let input = "<|tool_call_begin|>functions.[{\"name\": \"get_weather\", \"arguments\": {\"city\": \"a > b\"}}]";
        let want = vec![UnifiedEvent::ToolCall {
            name: "get_weather".into(),
            arguments: serde_json::json!({"city": "a > b"}),
        }];
        for (i, got) in assemble_guided_at_every_split(&tools(), input)
            .into_iter()
            .enumerate()
        {
            assert_eq!(got, want, "split at byte {i}, got {got:?}");
        }
    }

    /// `UNIFIED.guided_json_schema_error_not_a_call_bare_opener.kimi_k2`:
    /// same bare-header shape, but the guided payload isn't a call at all
    /// (no `name`) -- policy P2, surface it as text rather than dropping it.
    #[test]
    fn bare_opener_before_non_call_guided_json_surfaces_the_payload_as_text() {
        let input = "<|tool_call_begin|>functions.{\"unexpected\": \"shape\"}";
        let want = vec![UnifiedEvent::Text {
            text: "{\"unexpected\": \"shape\"}".into(),
        }];
        for (i, got) in assemble_guided_at_every_split(&tools(), input)
            .into_iter()
            .enumerate()
        {
            assert_eq!(got, want, "split at byte {i}, got {got:?}");
        }
    }

    /// `UNIFIED.guided_json_narrated_prefix_inside_reasoning.kimi_k2`: a
    /// narrated header with a bare NAME and no `:IDX` at all
    /// (`functions.get_weather`, no colon) is prose the model wrote while
    /// still thinking, not a real invoke id -- it must survive as reasoning
    /// text, not be swallowed as control markup the way a real
    /// `functions.NAME:IDX` id would be.
    #[test]
    fn bare_name_with_no_index_inside_reasoning_survives_as_reasoning_text() {
        let input = "<think>I'll call <|tool_call_begin|>functions.get_weather</think>[{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}]";
        let want = vec![
            UnifiedEvent::Reasoning {
                text: "I'll call get_weather".into(),
            },
            UnifiedEvent::ToolCall {
                name: "get_weather".into(),
                arguments: serde_json::json!({"city": "Paris"}),
            },
        ];
        for (i, got) in assemble_guided_at_every_split(&tools(), input)
            .into_iter()
            .enumerate()
        {
            assert_eq!(got, want, "split at byte {i}, got {got:?}");
        }
    }

    /// Negative control for the same fix: a real `functions.NAME:IDX` id
    /// (colon present) inside a native invoke must still be recognized and
    /// consumed as control markup, not left dangling as text -- the tri-state
    /// id scan must not treat every bare name as "no id" indiscriminately.
    #[test]
    fn a_real_native_id_is_still_recognized_across_every_split() {
        let input = "<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{\"city\": \"Paris\"}<|tool_call_end|><|tool_calls_section_end|>";
        let want = vec![UnifiedEvent::ToolCall {
            name: "get_weather".into(),
            arguments: serde_json::json!({"city": "Paris"}),
        }];
        for (i, got) in assemble_at_every_split(&tools(), input)
            .into_iter()
            .enumerate()
        {
            assert_eq!(got, want, "split at byte {i}, got {got:?}");
        }
    }

    #[test]
    fn native_tool_call_id_survives_the_guided_router() {
        let input = "<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{\"city\": \"Paris\"}<|tool_call_end|><|tool_calls_section_end|>";
        let mut parser = kimi_k2_unified(&tools());
        parser.push(input).expect("push");

        assert_eq!(parser.tool_call_id(0), Some("functions.get_weather:0"));
        assert_eq!(parser.tool_call_id(1), None);
    }

    /// Reviewer-caught regression: the Kimi batch grammar permits `\s*`
    /// between `NAME:IDX` and `argument_begin`, but the two partial-marker
    /// holdback checks in `kimi_invoke_end` used the raw remainder verbatim
    /// -- a chunk split landing right after that permitted whitespace
    /// (`remainder == " "`) matched neither marker's prefix, so the
    /// function wrongly committed to a header-only boundary one push
    /// early, before `argument_begin` streamed in. `K2Emitter` cannot parse
    /// that header-only span, and the real call was silently lost. This
    /// sibling of the test above adds exactly one space before
    /// `argument_begin` -- absent from that test, which is why it never
    /// caught this.
    #[test]
    fn whitespace_before_argument_marker_is_split_invariant() {
        let input = "<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0 <|tool_call_argument_begin|>{\"city\": \"Paris\"}<|tool_call_end|><|tool_calls_section_end|>";
        let want = vec![UnifiedEvent::ToolCall {
            name: "get_weather".into(),
            arguments: serde_json::json!({"city": "Paris"}),
        }];
        for (i, got) in assemble_at_every_split(&tools(), input)
            .into_iter()
            .enumerate()
        {
            assert_eq!(
                got, want,
                "valid whitespace-delimited call changed at split byte {i}: {got:?}"
            );
        }
    }

    /// A first invoke that never closes (`call_end` missing) before a
    /// second invoke opens must not reach across the new opener and swallow
    /// the second call's bytes as if they belonged to the first -- that
    /// silently corrupted the first call's arguments AND dropped the second
    /// call entirely. The first is recovered best-effort (its JSON is
    /// complete); the second parses as its own independent invoke.
    #[test]
    fn unclosed_invoke_does_not_swallow_the_next_invoke_across_every_split() {
        let tools = vec![tools()[0].clone(), run_tool()[0].clone()];
        let input = "<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{\"city\": \"Paris\"}<|tool_call_begin|>functions.run:1<|tool_call_argument_begin|>{\"cmd\": \"ls\"}<|tool_call_end|><|tool_calls_section_end|>";
        let want = vec![
            UnifiedEvent::ToolCall {
                name: "get_weather".into(),
                arguments: serde_json::json!({"city": "Paris"}),
            },
            UnifiedEvent::ToolCall {
                name: "run".into(),
                arguments: serde_json::json!({"cmd": "ls"}),
            },
        ];
        for (i, got) in assemble_at_every_split(&tools, input)
            .into_iter()
            .enumerate()
        {
            assert_eq!(got, want, "split at byte {i}, got {got:?}");
        }
    }
}
