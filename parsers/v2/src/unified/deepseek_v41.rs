// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Context;
use serde_json::{Map, Value};

use crate::tool_calling::scan::{
    BareRecoveryLatch, InvokeBoundary, InvokeBoundaryFactory, InvokeEmitter, ReasoningSpec,
    WrappedBlockScanner, WrappedBlockSpec,
};
use crate::tool_calling::traits::{Tool, ToolCallDelta};
use crate::unified::{
    GuidedInvokePrefix, GuidedInvokePrefixContext, GuidedRouted, ScannerUnified, UnifiedParser,
};

const BLOCK_START: &str = "<｜DSML｜ calls>";
const BLOCK_END: &str = "</｜DSML｜ calls>";
const INVOKE_START: &str = "<｜DSML｜ invoke name=\"";
const INVOKE_END: &str = "</｜DSML｜ invoke>";
const PARAMETER_START: &str = "<｜DSML｜ parameter name=\"";
const PARAMETER_END: &str = "</｜DSML｜ parameter>";

pub(crate) fn deepseek_v41_unified(_tools: &[Tool]) -> Box<dyn UnifiedParser> {
    let spec = WrappedBlockSpec {
        family: "deepseek_v41",
        block_starts: vec![BLOCK_START.into()],
        block_ends: vec![BLOCK_END.into()],
        invoke_start: INVOKE_START.into(),
        invoke_end: INVOKE_END.into(),
        orphan_markers: vec![BLOCK_END.into()],
        holdback_markers: vec![BLOCK_START.into(), BLOCK_END.into(), INVOKE_START.into()],
        bare_recovery_latch: BareRecoveryLatch::Set,
        invoke_boundary_factory: Some(InvokeBoundaryFactory::custom(invocation_boundary)),
        preserve_special_tokens: true,
        ..Default::default()
    };
    let scanner = WrappedBlockScanner::new(spec, DeepSeekV41).with_reasoning(ReasoningSpec {
        start: "<think>",
        end: "</think>",
        preserve_special_tokens: true,
        ..Default::default()
    });
    Box::new(GuidedRouted::new(ScannerUnified::new(scanner)))
}

fn parameter_header(text: &str) -> Option<(&str, bool, &str)> {
    let (name, rest) = text.strip_prefix(PARAMETER_START)?.split_once('"')?;
    let (string, value) = rest.strip_prefix(" string=\"")?.split_once("\">")?;
    let string = match string {
        "true" => true,
        "false" => false,
        _ => return None,
    };
    Some((name, string, value))
}

#[cfg(test)]
std::thread_local! {
    static BOUNDARY_EXAMINED_BYTES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

fn count_boundary_bytes(bytes: usize) {
    #[cfg(test)]
    BOUNDARY_EXAMINED_BYTES.with(|examined| examined.set(examined.get() + bytes));
    #[cfg(not(test))]
    let _ = bytes;
}

fn find_from(text: &str, start: usize, marker: &str) -> Option<usize> {
    let suffix = &text[start..];
    count_boundary_bytes(suffix.len());
    suffix.find(marker).map(|at| start + at)
}

fn find_payload_from(text: &str, start: usize) -> Option<usize> {
    let suffix = &text[start..];
    count_boundary_bytes(suffix.len());
    suffix.find(['{', '[']).map(|at| start + at)
}

fn next_scan_start(text: &str, marker_len: usize) -> usize {
    let mut start = text.len().saturating_sub(marker_len.saturating_sub(1));
    while !text.is_char_boundary(start) {
        start -= 1;
    }
    start
}

#[derive(Default)]
enum InvocationPosition {
    #[default]
    BetweenParameters,
    ParameterHeader {
        start: usize,
        scan_from: usize,
    },
    ParameterValue {
        start: usize,
        scan_from: usize,
    },
    InvalidParameter {
        start: usize,
    },
}

#[derive(Default)]
struct DeepSeekV41InvocationBoundary {
    position: InvocationPosition,
    scan_from: usize,
    guided_prefix_scan_from: usize,
    guided_prefix_payload_at: Option<usize>,
    guided_prefix_header_end: Option<usize>,
}

impl DeepSeekV41InvocationBoundary {
    fn malformed_end(text: &str, start: usize, flush: bool) -> Option<usize> {
        flush
            .then(|| find_from(text, start, INVOKE_END))
            .flatten()
            .map(|at| at + INVOKE_END.len())
    }
}

impl InvokeBoundary for DeepSeekV41InvocationBoundary {
    fn owns_guided_prefix(&self) -> bool {
        true
    }

    fn guided_prefix_append(
        &mut self,
        candidate: &str,
        append: &str,
        context: GuidedInvokePrefixContext,
    ) -> Option<GuidedInvokePrefix> {
        let header = candidate.strip_prefix(INVOKE_START)?;
        if !context.outside_reasoning || context.followed_by_competing_marker {
            return Some(GuidedInvokePrefix::Strip(INVOKE_START.len()));
        }

        let append_start = candidate.len() - append.len();
        let scan_from = self
            .guided_prefix_scan_from
            .max(append_start.saturating_sub(INVOKE_START.len()));
        if self.guided_prefix_payload_at.is_none() {
            self.guided_prefix_payload_at = find_payload_from(header, scan_from);
        }
        if self.guided_prefix_header_end.is_none() {
            self.guided_prefix_header_end = find_from(header, scan_from, ">");
        }
        self.guided_prefix_scan_from = header.len();

        if self.guided_prefix_header_end.is_some_and(|header_end| {
            self.guided_prefix_payload_at
                .is_none_or(|payload| header_end < payload)
        }) {
            return Some(GuidedInvokePrefix::NoMatch);
        }
        if let Some(payload_at) = self.guided_prefix_payload_at {
            // A bare DSML header has no closing quote or `>` before guided JSON.
            // Stop at the payload opener: the first quote in a JSON key is payload
            // data, not the header terminator.
            return Some(if context.payload_is_empty {
                GuidedInvokePrefix::Match(INVOKE_START.len() + payload_at)
            } else {
                GuidedInvokePrefix::Strip(INVOKE_START.len() + payload_at)
            });
        }
        Some(GuidedInvokePrefix::Pending)
    }

    fn end_append(
        &mut self,
        candidate: &str,
        _append: &str,
        flush: bool,
        _tool_index: usize,
    ) -> Option<usize> {
        loop {
            match self.position {
                InvocationPosition::BetweenParameters => {
                    let close = find_from(candidate, self.scan_from, INVOKE_END);
                    let parameter = find_from(candidate, self.scan_from, PARAMETER_START);
                    match (close, parameter) {
                        (Some(close), Some(parameter)) if parameter <= close => {
                            self.position = InvocationPosition::ParameterHeader {
                                start: parameter,
                                scan_from: parameter + PARAMETER_START.len(),
                            };
                        }
                        (Some(close), _) => return Some(close + INVOKE_END.len()),
                        (None, Some(parameter)) => {
                            self.position = InvocationPosition::ParameterHeader {
                                start: parameter,
                                scan_from: parameter + PARAMETER_START.len(),
                            };
                        }
                        (None, None) => {
                            self.scan_from = next_scan_start(
                                candidate,
                                INVOKE_END.len().max(PARAMETER_START.len()),
                            );
                            return None;
                        }
                    }
                }
                InvocationPosition::ParameterHeader { start, scan_from } => {
                    let Some(header_end) = find_from(candidate, scan_from, "\">") else {
                        if flush {
                            return Self::malformed_end(candidate, start, true);
                        }
                        self.position = InvocationPosition::ParameterHeader {
                            start,
                            scan_from: next_scan_start(candidate, 2),
                        };
                        return None;
                    };
                    let Some((_, _, value)) = parameter_header(&candidate[start..]) else {
                        self.position = InvocationPosition::InvalidParameter { start };
                        continue;
                    };
                    let value_start = candidate.len() - value.len();
                    debug_assert!(value_start >= header_end + 2);
                    self.position = InvocationPosition::ParameterValue {
                        start,
                        scan_from: value_start,
                    };
                }
                InvocationPosition::ParameterValue { start, scan_from } => {
                    let Some(value_end) = find_from(candidate, scan_from, PARAMETER_END) else {
                        if flush {
                            return Self::malformed_end(candidate, start, true);
                        }
                        self.position = InvocationPosition::ParameterValue {
                            start,
                            scan_from: next_scan_start(candidate, PARAMETER_END.len()),
                        };
                        return None;
                    };
                    self.scan_from = value_end + PARAMETER_END.len();
                    self.position = InvocationPosition::BetweenParameters;
                }
                InvocationPosition::InvalidParameter { start } => {
                    return Self::malformed_end(candidate, start, flush);
                }
            }
        }
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

fn invocation_boundary() -> Box<dyn InvokeBoundary> {
    Box::new(DeepSeekV41InvocationBoundary::default())
}

struct DeepSeekV41;

impl InvokeEmitter for DeepSeekV41 {
    fn parse_invoke(
        &mut self,
        invoke: &str,
        tool_index: usize,
    ) -> anyhow::Result<Option<ToolCallDelta>> {
        let (name, body) = invoke
            .strip_prefix(INVOKE_START)
            .and_then(|text| text.split_once("\">"))
            .context("invalid DeepSeek V4.1 invocation header")?;
        anyhow::ensure!(!name.is_empty(), "empty DeepSeek V4.1 tool name");
        let mut body = body
            .strip_suffix(INVOKE_END)
            .context("incomplete DeepSeek V4.1 invocation")?;
        let mut arguments = Map::new();
        while !body.trim().is_empty() {
            let (name, string, value) = parameter_header(body.trim_start())
                .context("invalid DeepSeek V4.1 parameter header")?;
            let (raw, remainder) = value
                .split_once(PARAMETER_END)
                .context("incomplete DeepSeek V4.1 parameter")?;
            let value = if string {
                Value::String(raw.to_string())
            } else {
                serde_json::from_str(raw)?
            };
            anyhow::ensure!(
                arguments.insert(name.to_string(), value).is_none(),
                "duplicate DeepSeek V4.1 parameter {name:?}"
            );
            body = remainder;
        }
        Ok(Some(ToolCallDelta {
            tool_index,
            name: Some(name.to_string()),
            arguments: serde_json::to_string(&arguments)?,
            complete: true,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::unified::{
        InvalidGuidedPayloadPolicy, UnifiedEvent, UnifiedParserExt, UnifiedParserInit,
        UnifiedParserOutput, UnifiedParserStartingState, UnifiedToolOutputMode,
        create_unified_parser_for_family,
    };

    fn parse_chunks(input: &str, split: usize, init: UnifiedParserInit) -> UnifiedParserOutput {
        let mut parser = create_unified_parser_for_family("deepseek_v41", &[]).unwrap();
        assert!(parser.preserve_special_tokens());
        parser.initialize_request(init).unwrap();
        let mut output = UnifiedParserOutput::default();
        parser.parse_into(&input[..split], &mut output).unwrap();
        parser.parse_into(&input[split..], &mut output).unwrap();
        output.append(&mut parser.finish().unwrap());
        output
    }

    fn assert_every_split_with_init(
        input: &str,
        init: UnifiedParserInit,
        expected: Vec<UnifiedEvent>,
    ) {
        for split in (0..=input.len()).filter(|&i| input.is_char_boundary(i)) {
            assert_eq!(
                parse_chunks(input, split, init.clone()).assembled(),
                expected,
                "split {split}"
            );
        }
        let mut parser = deepseek_v41_unified(&[]);
        parser.initialize_request(init).unwrap();
        let mut output = UnifiedParserOutput::default();
        for ch in input.chars() {
            parser
                .parse_into(ch.encode_utf8(&mut [0; 4]), &mut output)
                .unwrap();
        }
        output.append(&mut parser.finish().unwrap());
        assert_eq!(output.assembled(), expected, "one character at a time");
    }

    fn assert_every_split(
        input: &str,
        state: UnifiedParserStartingState,
        expected: Vec<UnifiedEvent>,
    ) {
        assert_every_split_with_init(
            input,
            UnifiedParserInit {
                starting_state: state,
                ..Default::default()
            },
            expected,
        );
    }

    #[test]
    fn deepseek_v41_registration() {
        assert_every_split(
            "hello 世界",
            UnifiedParserStartingState::None,
            vec![UnifiedEvent::Text {
                text: "hello 世界".into(),
            }],
        );
    }

    #[test]
    fn reasoning_transition_and_multiple_calls() {
        let input = concat!(
            "Check the tools.\n</think>\n\n<｜DSML｜ calls>\n",
            "<｜DSML｜ invoke name=\"weather\">\n",
            "<｜DSML｜ parameter name=\"city\" string=\"true\">東京 &amp; \"Paris\"\\\n</｜DSML｜ parameter>\n",
            "<｜DSML｜ parameter name=\"days\" string=\"false\">3</｜DSML｜ parameter>\n",
            "<｜DSML｜ parameter name=\"options\" string=\"false\">{\"x\":[true,null]}</｜DSML｜ parameter>\n",
            "</｜DSML｜ invoke>\n<｜DSML｜ invoke name=\"done\">\n</｜DSML｜ invoke>\n",
            "</｜DSML｜ calls>",
        );
        assert_every_split(
            input,
            UnifiedParserStartingState::Reasoning,
            vec![
                UnifiedEvent::Reasoning {
                    text: "Check the tools.\n".into(),
                },
                UnifiedEvent::Text {
                    text: "\n\n".into(),
                },
                UnifiedEvent::ToolCall {
                    name: "weather".into(),
                    arguments: serde_json::json!({
                        "city": "東京 &amp; \"Paris\"\\\n", "days": 3, "options": {"x": [true, null]}
                    }),
                },
                UnifiedEvent::ToolCall {
                    name: "done".into(),
                    arguments: serde_json::json!({}),
                },
            ],
        );
    }

    #[test]
    fn tool_markup_inside_string_is_data() {
        let input = "<｜DSML｜ calls><｜DSML｜ invoke name=\"run\"><｜DSML｜ parameter name=\"text\" string=\"true\"><think>quoted</think> <｜DSML｜ calls></｜DSML｜ parameter></｜DSML｜ invoke></｜DSML｜ calls>";
        assert_every_split(
            input,
            UnifiedParserStartingState::None,
            vec![UnifiedEvent::ToolCall {
                name: "run".into(),
                arguments: serde_json::json!({"text": "<think>quoted</think> <｜DSML｜ calls>"}),
            }],
        );
    }

    #[test]
    fn closing_markers_and_whitespace_inside_strings_are_data() {
        let value = " X</｜DSML｜ calls>Y</｜DSML｜ invoke>Z\n ";
        let input = format!(
            "<｜DSML｜ calls><｜DSML｜ invoke name=\"run\"><｜DSML｜ parameter name=\"user name\" string=\"true\">{value}</｜DSML｜ parameter></｜DSML｜ invoke></｜DSML｜ calls>"
        );
        assert_every_split(
            &input,
            UnifiedParserStartingState::None,
            vec![UnifiedEvent::ToolCall {
                name: "run".into(),
                arguments: serde_json::json!({"user name":value}),
            }],
        );
    }

    #[test]
    fn incomplete_arguments_do_not_emit_calls() {
        let input = "<｜DSML｜ calls><｜DSML｜ invoke name=\"run\"><｜DSML｜ parameter name=\"text\" string=\"true\">unfinished";
        assert_every_split(input, UnifiedParserStartingState::None, vec![]);
    }

    #[test]
    fn partial_plain_markers_and_open_reasoning_survive_eof() {
        for input in ["hello <", "hello <｜DS", "ordinary text\n", "你好"] {
            assert_every_split(
                input,
                UnifiedParserStartingState::None,
                vec![UnifiedEvent::Text { text: input.into() }],
            );
            assert_every_split(
                input,
                UnifiedParserStartingState::Reasoning,
                vec![UnifiedEvent::Reasoning { text: input.into() }],
            );
        }
    }

    #[test]
    fn calls_stream_at_each_invocation_close() {
        let mut parser = deepseek_v41_unified(&[]);
        assert!(parser.push("<｜DSML｜ calls><｜DSML｜ invoke name=\"run\"><｜DSML｜ parameter name=\"text\" string=\"true\">hello").unwrap().is_empty());
        let events = parser
            .push("</｜DSML｜ parameter></｜DSML｜ invoke>")
            .unwrap();
        let output: UnifiedParserOutput = events.into_iter().collect();
        assert_eq!(
            output.assembled(),
            vec![UnifiedEvent::ToolCall {
                name: "run".into(),
                arguments: serde_json::json!({"text":"hello"}),
            }]
        );
        assert!(parser.push("</｜DSML｜ calls>").unwrap().is_empty());
    }

    #[test]
    fn invocation_requires_its_complete_closing_tag() {
        for suffix in ["", " invoke", " banana>"] {
            let input = format!("<｜DSML｜ calls><｜DSML｜ invoke name=\"run\"></｜DSML｜{suffix}");
            let mut parser = deepseek_v41_unified(&[]);
            assert!(parser.push(&input).unwrap().is_empty());
            assert!(parser.finish().unwrap().events.is_empty());
        }
    }

    #[test]
    fn invocation_boundary_scans_large_streamed_parameters_linearly() {
        let mut parser = deepseek_v41_unified(&[]);
        parser
            .push("<｜DSML｜ calls><｜DSML｜ invoke name=\"run\"><｜DSML｜ parameter name=\"text\" string=\"true\">")
            .unwrap();
        BOUNDARY_EXAMINED_BYTES.with(|examined| examined.set(0));

        let value = "x".repeat(16 * 1024);
        for byte in value.as_bytes() {
            parser
                .push(std::str::from_utf8(std::slice::from_ref(byte)).unwrap())
                .unwrap();
        }
        let events = parser
            .push("</｜DSML｜ parameter></｜DSML｜ invoke></｜DSML｜ calls>")
            .unwrap();
        let examined = BOUNDARY_EXAMINED_BYTES.with(std::cell::Cell::get);

        let output: UnifiedParserOutput = events.into_iter().collect();
        assert_eq!(
            output.assembled(),
            vec![UnifiedEvent::ToolCall {
                name: "run".into(),
                arguments: serde_json::json!({"text": value}),
            }]
        );
        assert!(
            examined < value.len() * PARAMETER_END.len() * 2,
            "boundary examined {examined} bytes for a {}-byte value",
            value.len()
        );
    }

    #[test]
    fn guided_bare_header_scans_streamed_name_linearly() {
        let mut boundary = DeepSeekV41InvocationBoundary::default();
        let context = GuidedInvokePrefixContext {
            outside_reasoning: true,
            payload_is_empty: true,
            followed_by_competing_marker: false,
        };
        let mut candidate = INVOKE_START.to_string();
        assert_eq!(
            boundary.guided_prefix_append(&candidate, INVOKE_START, context),
            Some(GuidedInvokePrefix::Pending)
        );
        BOUNDARY_EXAMINED_BYTES.with(|examined| examined.set(0));

        let name = "x".repeat(16 * 1024);
        for byte in name.bytes() {
            let append = std::str::from_utf8(std::slice::from_ref(&byte)).unwrap();
            candidate.push_str(append);
            assert_eq!(
                boundary.guided_prefix_append(&candidate, append, context),
                Some(GuidedInvokePrefix::Pending)
            );
        }
        let examined = BOUNDARY_EXAMINED_BYTES.with(std::cell::Cell::get);
        assert!(
            examined < name.len() * 4,
            "guided prefix examined {examined} bytes for a {}-byte name",
            name.len()
        );
    }

    #[test]
    fn malformed_invocation_is_an_error_without_tool_deltas() {
        for input in [
            "<｜DSML｜ calls><｜DSML｜ invoke name=\"run\"><｜DSML｜ parameter name=\"x\" string=\"true\">first</｜DSML｜ parameter><｜DSML｜ parameter name=\"x\" string=\"true\">second</｜DSML｜ parameter></｜DSML｜ invoke></｜DSML｜ calls>",
            "<｜DSML｜ calls><｜DSML｜ invoke name=\"run\"><｜DSML｜ parameter name=\"value\" string=\"false\">invalid</｜DSML｜ parameter></｜DSML｜ invoke></｜DSML｜ calls>",
        ] {
            for split in (0..=input.len()).filter(|&i| input.is_char_boundary(i)) {
                let mut parser = deepseek_v41_unified(&[]);
                let mut output = UnifiedParserOutput::default();
                let result = parser
                    .parse_into(&input[..split], &mut output)
                    .and_then(|()| parser.parse_into(&input[split..], &mut output));
                assert!(result.is_err(), "split {split}");
                assert!(output.events.is_empty(), "split {split}");
            }
        }
    }

    #[test]
    fn closed_malformed_parameter_is_an_error_at_eof() {
        for input in [
            "<｜DSML｜ calls><｜DSML｜ invoke name=\"run\"><｜DSML｜ parameter name=\"value\" string=\"maybe\">1</｜DSML｜ parameter></｜DSML｜ invoke></｜DSML｜ calls>",
            "<｜DSML｜ calls><｜DSML｜ invoke name=\"run\"><｜DSML｜ parameter name=\"value\" string=\"false\">1</｜DSML｜ invoke></｜DSML｜ calls>",
        ] {
            for split in (0..=input.len()).filter(|&i| input.is_char_boundary(i)) {
                let mut parser = deepseek_v41_unified(&[]);
                let mut output = UnifiedParserOutput::default();
                let result = parser
                    .parse_into(&input[..split], &mut output)
                    .and_then(|()| parser.parse_into(&input[split..], &mut output))
                    .and_then(|()| parser.finish().map(|_| ()));
                assert!(result.is_err(), "split {split}");
                assert!(output.events.is_empty(), "split {split}");
            }
        }
    }

    #[test]
    fn guided_output_uses_shared_decoder() {
        let mut parser = deepseek_v41_unified(&[]);
        parser
            .initialize_request(UnifiedParserInit {
                starting_state: UnifiedParserStartingState::Reasoning,
                tool_output_mode: UnifiedToolOutputMode::GuidedJson {
                    named_tool: Some("weather".into()),
                },
                ..Default::default()
            })
            .unwrap();
        assert_eq!(
            parser
                .parse_complete("checking weather</think>{\"city\":\"Paris\"}")
                .unwrap(),
            vec![
                UnifiedEvent::Reasoning {
                    text: "checking weather".into()
                },
                UnifiedEvent::ToolCall {
                    name: "weather".into(),
                    arguments: serde_json::json!({"city":"Paris"})
                },
            ]
        );
    }

    #[test]
    fn guided_bare_headers_do_not_consume_json_or_reasoning() {
        let payload = r#"[{"name":"get_weather","arguments":{"city":"Paris"}}]"#;
        let invalid_payloads = [
            (
                r#"[{"name":"get_weather","arguments":{"city": "#,
                r#"[{"name":"get_weather","arguments":{"city": "#,
            ),
            (r#"{"unexpected":"shape"}"#, r#"{"unexpected":"shape"}"#),
            (
                r#"[{"name":"get_weather","arguments":{"city":"Paris"}},{"arguments":{}}]"#,
                r#"[{"name":"get_weather","arguments":{"city":"Paris"}},{"arguments":{}}]"#,
            ),
        ];
        let init = UnifiedParserInit {
            tool_output_mode: UnifiedToolOutputMode::GuidedJson { named_tool: None },
            invalid_guided_payload: InvalidGuidedPayloadPolicy::RecoverAsText,
            ..Default::default()
        };
        for (input, expected) in invalid_payloads {
            assert_every_split_with_init(
                &format!("{INVOKE_START}{input}"),
                init.clone(),
                vec![UnifiedEvent::Text {
                    text: expected.into(),
                }],
            );
        }
        assert_every_split_with_init(&format!("{INVOKE_START}>{payload}"), init.clone(), vec![]);
        assert_every_split_with_init(
            &format!("{INVOKE_START}<think>secret</think>{payload}"),
            init.clone(),
            vec![
                UnifiedEvent::Reasoning {
                    text: "secret".into(),
                },
                UnifiedEvent::ToolCall {
                    name: "get_weather".into(),
                    arguments: serde_json::json!({"city":"Paris"}),
                },
            ],
        );
        assert_every_split_with_init(
            &format!("<think>I'll use {INVOKE_START} next</think>{payload}"),
            init.clone(),
            vec![
                UnifiedEvent::Reasoning {
                    text: "I'll use  next".into(),
                },
                UnifiedEvent::ToolCall {
                    name: "get_weather".into(),
                    arguments: serde_json::json!({"city":"Paris"}),
                },
            ],
        );
        assert_every_split_with_init(
            &format!("<think>I'll call {INVOKE_START}get_weather</think>{payload}"),
            init,
            vec![
                UnifiedEvent::Reasoning {
                    text: "I'll call get_weather".into(),
                },
                UnifiedEvent::ToolCall {
                    name: "get_weather".into(),
                    arguments: serde_json::json!({"city":"Paris"}),
                },
            ],
        );
    }

    #[test]
    fn reset_restarts_tool_indices() {
        let input =
            "<｜DSML｜ calls><｜DSML｜ invoke name=\"done\"></｜DSML｜ invoke></｜DSML｜ calls>";
        let mut parser = deepseek_v41_unified(&[]);
        let first = parser.push(input).unwrap();
        parser.finish().unwrap();
        assert!(parser.reset().is_empty());
        assert_eq!(parser.push(input).unwrap(), first);
    }
}
