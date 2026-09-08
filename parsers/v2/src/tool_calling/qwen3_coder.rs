// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Streaming XML tool-call parser for Qwen3-Coder.
//!
//! Qwen3-Coder emits tool calls as
//!   `<tool_call> <function=NAME> <parameter=KEY>value</parameter> ... </function> </tool_call>`
//! plus a bare `<function=...></function>` back-off form when the outer wrapper
//! is absent (shared with nemotron_nano).
//!
//! The streaming concern (buffering, chunk-split marker safety, normal_text
//! suppression) is owned by the shared [`scan::WrappedBlockScanner`]. The
//! per-block value typing is delegated to the vendored batch XML parser via
//! `parse_tool_call_block`, so a streamed call matches exactly what the batch
//! parser produces. Arguments are re-serialized in the
//! source parameter order because the v1 parser builds them from a `HashMap`
//! whose key order is non-deterministic; streaming fixtures store the arguments
//! as an exact JSON string, so order has to be pinned to the model-emitted
//! order (the order vLLM's Rust parser also preserves).

use crate::tool_calling::scan::{
    BareRecoveryLatch, InvokeEmitter, InvokeLatch, WrappedBlockScanner, WrappedBlockSpec,
    marker_prefix_suffix_len, reorder_arguments,
};
use crate::tool_calling::v1core::{ToolDefinition, XmlParserConfig, parse_tool_call_block};

use crate::tool_calling::traits::{Tool, ToolCallDelta, ToolParseResult, ToolParser};
use std::collections::HashSet;

const BLOCK_START: &str = "<tool_call>";
const BLOCK_END: &str = "</tool_call>";
const FUNCTION_START: &str = "<function=";
const FUNCTION_END: &str = "</function>";
const PARAMETER_START: &str = "<parameter=";

fn spec() -> WrappedBlockSpec {
    WrappedBlockSpec {
        family: "qwen3_coder",
        block_starts: vec![BLOCK_START.to_string()],
        block_ends: vec![BLOCK_END.to_string()],
        invoke_start: FUNCTION_START.to_string(),
        invoke_end: FUNCTION_END.to_string(),
        orphan_markers: vec![BLOCK_END.to_string()],
        // BLOCK_END is held back too so a split stray/orphan close (consumed
        // and dropped by the orphan-close handler once complete) never emits
        // its first half as text.
        holdback_markers: vec![
            BLOCK_START.to_string(),
            BLOCK_END.to_string(),
            FUNCTION_START.to_string(),
        ],
        bare_recovery_latch: BareRecoveryLatch::Set,
        invoke_latch: InvokeLatch::IfEmitted,
        drop_invoke_crossing_block_end: false,
        // Every wrapped family's markers are special tokens today.
        preserve_special_tokens: true,
        ..Default::default()
    }
}

/// Value-typing hook: types one complete `<function=...></function>` block and
/// re-orders the arguments to source order.
pub(crate) struct Qwen3Emitter {
    config: XmlParserConfig,
    tools: Vec<ToolDefinition>,
    partial: Option<PartialStringArgument>,
}

/// Append-only Qwen argument state. Each schema-declared string parameter can
/// stream while open; completed parameters advance `scan_cursor` so a later
/// long string keeps making progress before the function closes.
struct PartialStringArgument {
    tool_index: usize,
    name: String,
    emitted_json: String,
    scan_cursor: usize,
    seen_parameters: HashSet<String>,
    active: Option<ActiveStringParameter>,
    blocked: bool,
}

struct ActiveStringParameter {
    value_cursor: usize,
    pending_entity: String,
    trailing_whitespace: String,
    started: bool,
    opener_pending: String,
}

impl InvokeEmitter for Qwen3Emitter {
    fn parse_partial_invoke(
        &mut self,
        invoke: &str,
        tool_index: usize,
    ) -> anyhow::Result<Option<ToolCallDelta>> {
        if self.partial.is_none() {
            let Some(header_end) = invoke.find('>') else {
                return Ok(None);
            };
            let name = invoke[FUNCTION_START.len()..header_end]
                .trim()
                .trim_matches('"')
                .trim();
            if !self.tools.iter().any(|tool| tool.name == name) {
                return Ok(None);
            }
            self.partial = Some(PartialStringArgument {
                tool_index,
                name: name.to_string(),
                emitted_json: String::new(),
                scan_cursor: header_end + 1,
                seen_parameters: HashSet::new(),
                active: None,
                blocked: false,
            });
        }

        let partial = self.partial.as_mut().expect("initialized above");
        if partial.tool_index != tool_index || partial.blocked {
            return Ok(None);
        }
        let mut arguments = String::new();
        loop {
            if let Some(active) = partial.active.as_mut() {
                let value = &invoke[active.value_cursor..];
                let close = value.find("</parameter>");
                let safe_end =
                    close.unwrap_or_else(|| value.len() - qwen_partial_suffix_len(value));
                let decoded = decode_streamable_xml_text(
                    &value[..safe_end],
                    &mut active.pending_entity,
                    close.is_some(),
                );
                active.value_cursor += safe_end;
                let mut fragment = String::new();
                append_trimmed_string_fragment(active, &decoded, &mut fragment);
                if active.started && !active.opener_pending.is_empty() {
                    arguments.push_str(&active.opener_pending);
                    active.opener_pending.clear();
                }
                arguments.push_str(&fragment);
                if close.is_some() {
                    if !active.started {
                        arguments.push_str(&active.opener_pending);
                        arguments.push('"');
                        partial.scan_cursor = active.value_cursor + "</parameter>".len();
                        partial.active = None;
                        continue;
                    }
                    active.value_cursor += "</parameter>".len();
                    arguments.push('"');
                    partial.scan_cursor = active.value_cursor;
                    partial.active = None;
                    continue;
                }
                break;
            }

            let Some(relative) = invoke[partial.scan_cursor..].find(PARAMETER_START) else {
                break;
            };
            let parameter_start = partial.scan_cursor + relative;
            let parameter_header = parameter_start + PARAMETER_START.len();
            let Some(relative_end) = invoke[parameter_header..].find('>') else {
                break;
            };
            let parameter_end = parameter_header + relative_end;
            let parameter = invoke[parameter_header..parameter_end]
                .trim()
                .trim_matches('"')
                .trim();
            if partial.seen_parameters.contains(parameter) {
                partial.blocked = true;
                break;
            }
            let value_start = parameter_end + 1;
            let closed = invoke[value_start..]
                .find("</parameter>")
                .map(|relative| value_start + relative + "</parameter>".len());
            let streamable = self
                .tools
                .iter()
                .find(|tool| tool.name == partial.name)
                .and_then(|tool| tool.parameters.as_ref())
                .and_then(|schema| schema.get("properties"))
                .and_then(|properties| properties.get(parameter))
                .and_then(|schema| schema.get("type"))
                .and_then(serde_json::Value::as_str)
                == Some("string");
            if !streamable {
                let Some(_) = closed else {
                    break;
                };
                if !partial.emitted_json.is_empty() {
                    partial.blocked = true;
                    break;
                }
                partial.blocked = true;
                break;
            }
            partial.seen_parameters.insert(parameter.to_string());
            let mut opener = String::new();
            if !partial.emitted_json.is_empty() || !arguments.is_empty() {
                opener.push(',');
            } else {
                opener.push('{');
            }
            opener.push_str(&serde_json::to_string(parameter)?);
            opener.push_str(":\"");
            partial.active = Some(ActiveStringParameter {
                value_cursor: value_start,
                pending_entity: String::new(),
                trailing_whitespace: String::new(),
                started: false,
                opener_pending: opener,
            });
            continue;
        }
        if arguments.is_empty() {
            return Ok(None);
        }
        let first = partial.emitted_json.is_empty();
        partial.emitted_json.push_str(&arguments);
        Ok(Some(ToolCallDelta {
            tool_index,
            name: first.then(|| partial.name.clone()),
            arguments,
            complete: false,
        }))
    }

    fn parse_invoke(
        &mut self,
        invoke: &str,
        tool_index: usize,
    ) -> anyhow::Result<Option<ToolCallDelta>> {
        // Type this ONE invoke directly. Wrapping it back in `<tool_call>` and
        // re-entering `try_tool_call_parse_xml` made the batch parser re-run
        // block discovery, which cuts the block at the FIRST `</tool_call>` —
        // so a parameter value that legitimately contains that marker was
        // truncated (`<parameter=cmd>git log </tool_call> --oneline</parameter>`
        // typed as `git log </tool_call>`). The scanner has already delimited
        // the invoke, so re-discovering its bounds could only corrupt them.
        let calls = parse_tool_call_block(invoke, &self.config, Some(&self.tools))?;
        let Some(call) = calls.into_iter().next() else {
            return Ok(None);
        };
        let arguments =
            reorder_arguments(&call.function.arguments, &source_parameter_order(invoke));
        let partial = self.partial.take();
        let streamed = partial
            .as_ref()
            .is_some_and(|partial| !partial.emitted_json.is_empty());
        if streamed
            && partial
                .as_ref()
                .is_some_and(|partial| !arguments.starts_with(&partial.emitted_json))
        {
            // Duplicate parameters use last-write-wins in the batch parser. Once
            // the first value has reached a streaming consumer, the later value
            // cannot be retracted without assembling corrupted JSON, so leave
            // this call incomplete and let the normal coalescer discard it.
            tracing::warn!(
                why = "qwen_streamed_call_invalidated_by_duplicate_parameter",
                tool_index,
                "streamed Qwen arguments no longer match the completed call"
            );
            return Ok(Some(ToolCallDelta {
                tool_index,
                name: None,
                arguments: String::new(),
                complete: false,
            }));
        }
        let arguments = if let Some(partial) = &partial {
            if streamed
                && partial.tool_index == tool_index
                && arguments.starts_with(&partial.emitted_json)
            {
                arguments[partial.emitted_json.len()..].to_string()
            } else {
                arguments
            }
        } else {
            arguments
        };
        Ok(Some(ToolCallDelta {
            tool_index,
            // The streaming opener already supplied the name. The close carries
            // only the argument suffix, matching OpenAI delta semantics.
            name: (!streamed).then_some(call.function.name),
            arguments,
            complete: true,
        }))
    }

    fn reset(&mut self) {
        self.partial = None;
    }
}

/// Build the scan core for the Qwen3 tool grammar.
///
/// The single construction site for this grammar. The unified Qwen3 parser
/// calls it too and layers reasoning on top, so both parsers get the same
/// block/invoke markers, holdback set, recovery latches and value typing —
/// there is no second copy to drift.
pub(crate) fn qwen3_scanner(tools: &[Tool]) -> WrappedBlockScanner<Qwen3Emitter> {
    WrappedBlockScanner::new(
        spec(),
        Qwen3Emitter {
            config: XmlParserConfig::default(),
            tools: tools.iter().map(ToolDefinition::from).collect(),
            partial: None,
        },
    )
}

fn append_trimmed_string_fragment(
    active: &mut ActiveStringParameter,
    decoded: &str,
    output: &mut String,
) {
    let decoded = if active.started {
        decoded
    } else {
        decoded.trim_start()
    };
    let content_end = decoded.trim_end().len();
    if content_end == 0 {
        if active.started {
            active.trailing_whitespace.push_str(decoded);
        }
        return;
    }
    let mut fragment = std::mem::take(&mut active.trailing_whitespace);
    fragment.push_str(&decoded[..content_end]);
    output.push_str(&json_string_fragment(&fragment));
    active.started = true;
    if content_end < decoded.len() {
        active.trailing_whitespace.push_str(&decoded[content_end..]);
    }
}

/// Decode only complete entities while retaining a bounded ambiguous suffix.
fn decode_streamable_xml_text(raw: &str, pending: &mut String, flush: bool) -> String {
    const ENTITIES: [(&str, &str); 9] = [
        ("&amp;quot;", "\""),
        ("&amp;#x27;", "'"),
        ("&amp;#39;", "'"),
        ("&lt;", "<"),
        ("&gt;", ">"),
        ("&amp;", "&"),
        ("&quot;", "\""),
        ("&#x27;", "'"),
        ("&#39;", "'"),
    ];
    pending.push_str(raw);
    let mut decoded = String::new();
    let mut cursor = 0;
    while cursor < pending.len() {
        let rest = &pending[cursor..];
        if rest.starts_with('&') {
            if let Some((entity, replacement)) =
                ENTITIES.iter().find(|(entity, _)| rest.starts_with(entity))
            {
                if !flush
                    && ENTITIES
                        .iter()
                        .any(|(longer, _)| longer.len() > entity.len() && longer.starts_with(rest))
                {
                    break;
                }
                decoded.push_str(replacement);
                cursor += entity.len();
                continue;
            }
            if !flush && ENTITIES.iter().any(|(entity, _)| entity.starts_with(rest)) {
                break;
            }
        }
        let next_entity = rest
            .char_indices()
            .skip(1)
            .find(|(_, character)| *character == '&')
            .map(|(at, _)| at)
            .unwrap_or(rest.len());
        decoded.push_str(&rest[..next_entity]);
        cursor += next_entity;
    }
    pending.drain(..cursor);
    decoded
}

fn json_string_fragment(text: &str) -> String {
    let encoded = serde_json::to_string(text).expect("serializing a string cannot fail");
    encoded[1..encoded.len() - 1].to_string()
}

/// Keep a native function-close prefix out of an open parameter value without
/// exposing it to the shared guided-decoding marker vocabulary.
fn qwen_partial_suffix_len(value: &str) -> usize {
    marker_prefix_suffix_len(value, ["</parameter>", FUNCTION_END])
}

/// Stream parser for Qwen3-Coder XML tool calls.
pub struct Qwen3CoderToolStreamParser {
    scanner: WrappedBlockScanner<Qwen3Emitter>,
}

impl Qwen3CoderToolStreamParser {
    pub fn new(tools: &[Tool]) -> Self {
        Self {
            scanner: qwen3_scanner(tools),
        }
    }
}

impl ToolParser for Qwen3CoderToolStreamParser {
    fn create(tools: &[Tool]) -> anyhow::Result<Box<dyn ToolParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self::new(tools)))
    }

    fn preserve_special_tokens(&self) -> bool {
        // Delegated, not restated: the unified adapter over this same scanner must not
        // be able to answer differently for identical markup.
        self.scanner.preserve_special_tokens()
    }

    fn push(&mut self, chunk: &str) -> anyhow::Result<ToolParseResult> {
        self.scanner.push(chunk)
    }

    fn finish(&mut self) -> anyhow::Result<ToolParseResult> {
        self.scanner.finish()
    }
}

/// Parameter names in the order they appear in a function block.
fn source_parameter_order(function: &str) -> Vec<String> {
    let mut names = Vec::new();
    let mut cursor = 0;
    while let Some(rel) = function[cursor..].find(PARAMETER_START) {
        let start = cursor + rel + PARAMETER_START.len();
        let Some(header_end) = function[start..].find('>') else {
            break;
        };
        let name = function[start..start + header_end]
            .trim()
            .trim_matches('"')
            .trim();
        if !name.is_empty() {
            names.push(name.to_string());
        }
        cursor = start + header_end + 1;
    }
    names
}

#[cfg(test)]
mod tests {
    use super::*;

    fn weather_tools() -> Vec<Tool> {
        vec![Tool {
            name: "get_weather".to_string(),
            description: None,
            parameters: serde_json::json!({
                "type": "object",
                "properties": { "location": { "type": "string" } }
            }),
            strict: None,
        }]
    }

    fn create_file_tools() -> Vec<Tool> {
        vec![Tool {
            name: "create_file".to_string(),
            description: None,
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string" },
                    "content": { "type": "string" }
                }
            }),
            strict: None,
        }]
    }

    fn parse_chunks(tools: &[Tool], chunks: &[&str]) -> ToolParseResult {
        let mut parser = Qwen3CoderToolStreamParser::new(tools);
        let mut out = ToolParseResult::default();
        for chunk in chunks {
            out.append(parser.push(chunk).expect("push"));
        }
        out.append(parser.finish().expect("finish"));
        out
    }

    #[test]
    fn emits_complete_call_on_close() {
        let out = parse_chunks(
            &weather_tools(),
            &[
                "<tool_call> <function=get_weather>",
                " <parameter=location>",
                " NYC </parameter> </function>",
                " </tool_call>",
            ],
        );
        assert_eq!(out.normal_text, "");
        let out = out.coalesce_calls();
        assert_eq!(out.calls.len(), 1);
        assert_eq!(out.calls[0].tool_index, 0);
        assert_eq!(out.calls[0].name.as_deref(), Some("get_weather"));
        // Value is schema-typed (string) and trimmed, matching the v1 batch parser.
        assert_eq!(out.calls[0].arguments, r#"{"location":"NYC"}"#);
    }

    #[test]
    fn preserves_prefix_text_before_block() {
        let out = parse_chunks(
            &weather_tools(),
            &[
                "I will",
                " check the weather. <tool_call>",
                " <function=get_weather>",
                " <parameter=location>NYC</parameter> </function> </tool_call>",
            ],
        );
        assert_eq!(out.normal_text, "I will check the weather. ");
        assert_eq!(out.calls.len(), 1);
    }

    #[test]
    fn recovers_complete_bare_function() {
        let out = parse_chunks(
            &weather_tools(),
            &[
                "I will check that. <function=get_weather>",
                " <parameter=location>NYC</parameter>",
                " </function>",
            ],
        );
        assert_eq!(out.normal_text, "I will check that. ");
        let out = out.coalesce_calls();
        assert_eq!(out.calls.len(), 1);
        assert_eq!(out.calls[0].arguments, r#"{"location":"NYC"}"#);
    }

    #[test]
    fn preserves_trailing_text_after_block() {
        // 8.b: trailing narration after a complete block flows into normal_text.
        let out = parse_chunks(
            &weather_tools(),
            &[
                "<tool_call> <function=get_weather> <parameter=location>NYC</parameter> </function> </tool_call>",
                " Let me know if you need more.",
            ],
        );
        assert_eq!(out.normal_text, " Let me know if you need more.");
        assert_eq!(out.coalesce_calls().calls.len(), 1);
    }

    #[test]
    fn preserves_inter_call_and_trailing_text() {
        // 8.d: narration between two complete blocks flows into normal_text;
        // both calls are emitted with distinct indices.
        let out = parse_chunks(
            &weather_tools(),
            &[
                "I will check the weather. <tool_call> <function=get_weather> <parameter=location>NYC</parameter> </function> </tool_call>",
                " Then check LA weather. <tool_call> <function=get_weather> <parameter=location>LA</parameter> </function> </tool_call>",
            ],
        );
        assert_eq!(
            out.normal_text,
            "I will check the weather.  Then check LA weather. "
        );
        let merged = out.coalesce_calls();
        assert_eq!(merged.calls.len(), 2);
        assert_eq!(merged.calls[0].arguments, r#"{"location":"NYC"}"#);
        assert_eq!(merged.calls[1].arguments, r#"{"location":"LA"}"#);
    }

    #[test]
    fn truncated_string_function_exposes_only_an_incomplete_delta() {
        let out = parse_chunks(
            &weather_tools(),
            &[
                "<tool_call> <function=get_weather>",
                " <parameter=location> NY",
            ],
        );
        assert_eq!(out.normal_text, "");
        assert!(!out.calls.is_empty());
        assert!(out.calls.iter().all(|call| !call.complete));
        assert!(out.coalesce_calls().calls.is_empty());
    }

    #[test]
    fn parameter_header_without_any_value_never_assembles_a_call() {
        let input = "<tool_call><function=get_weather><parameter=location>";
        for split in (0..=input.len()).filter(|&index| input.is_char_boundary(index)) {
            let out = parse_chunks(&weather_tools(), &[&input[..split], &input[split..]]);
            assert!(
                out.calls.is_empty(),
                "split {split} emitted without a value"
            );
            assert!(
                out.coalesce_calls().calls.is_empty(),
                "split {split} assembled an unfinished call"
            );
        }
    }

    #[test]
    fn holds_back_split_orphan_close() {
        // A stray/orphan `</tool_call>` split across a chunk boundary with no tool
        // call open must NOT leak its first half ("</tool") into normal_text: the
        // partial close is held back until the next chunk completes the marker, at
        // which point the orphan-close handler drops it entirely.
        let out = parse_chunks(&weather_tools(), &["done </tool", "_call> ok"]);
        assert!(out.calls.is_empty());
        assert!(
            !out.normal_text.contains('<'),
            "markup fragment leaked into normal_text: {:?}",
            out.normal_text
        );
        assert_eq!(out.normal_text, "done  ok");
    }

    #[test]
    fn preserves_source_parameter_order() {
        // path, old_str, new_str, command is deliberately NOT alphabetical: the
        // serialized arguments must keep the model-emitted parameter order.
        let tools = vec![Tool {
            name: "file_editor".to_string(),
            description: None,
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string" },
                    "old_str": { "type": "string" },
                    "new_str": { "type": "string" },
                    "command": { "type": "string" }
                }
            }),
            strict: None,
        }];
        let out = parse_chunks(
            &tools,
            &[
                "<tool_call> <function=file_editor>",
                " <parameter=path>/app/x.go</parameter>",
                " <parameter=old_str>foo</parameter>",
                " <parameter=new_str>bar</parameter>",
                " <parameter=command>str_replace</parameter>",
                " </function> </tool_call>",
            ],
        );
        let out = out.coalesce_calls();
        assert_eq!(out.calls.len(), 1);
        assert_eq!(
            out.calls[0].arguments,
            r#"{"path":"/app/x.go","old_str":"foo","new_str":"bar","command":"str_replace"}"#
        );
    }

    #[test]
    fn duplicate_streamed_parameter_does_not_assemble_corrupted_arguments() {
        let input = "<tool_call><function=get_weather><parameter=location>Paris</parameter><parameter=location>London</parameter></function></tool_call>";
        let first_value = input.find("Paris").unwrap() + "Paris".len();
        let out = parse_chunks(
            &weather_tools(),
            &[&input[..first_value], &input[first_value..]],
        );
        let assembled = out.clone().coalesce_calls();
        assert!(
            assembled.calls.is_empty(),
            "streamed duplicate parameter assembled mismatched JSON: {out:?}"
        );

        let complete = parse_chunks(&weather_tools(), &[input]).coalesce_calls();
        assert_eq!(complete.calls[0].arguments, r#"{"location":"London"}"#);
    }

    fn stream_every_char(tools: &[Tool], input: &str) -> ToolParseResult {
        let mut parser = Qwen3CoderToolStreamParser::new(tools);
        let mut out = ToolParseResult::default();
        for character in input.chars() {
            out.append(parser.push(&character.to_string()).expect("push"));
        }
        out.append(parser.finish().expect("finish"));
        out
    }

    #[test]
    fn streams_string_arguments_before_function_close_at_every_char_boundary() {
        let input = "<tool_call><function=get_weather><parameter=location>Montréal \"café\" \\ path with substantial remaining content</parameter>still-open</function></tool_call>";
        let baseline = parse_chunks(&weather_tools(), &[input]).coalesce_calls();
        assert_eq!(
            baseline.calls[0].arguments,
            r#"{"location":"Montréal \"café\" \\ path with substantial remaining content"}"#
        );

        let close = input.find("</function>").unwrap();
        let mut parser = Qwen3CoderToolStreamParser::new(&weather_tools());
        let mut before_close = ToolParseResult::default();
        for character in input[..close].chars() {
            before_close.append(parser.push(&character.to_string()).expect("push"));
        }
        assert!(
            before_close.calls.iter().any(|call| call.name.is_some()),
            "name-bearing delta must arrive before </function>: {before_close:?}"
        );
        let before_parameter_close = input.find("</parameter>").unwrap();
        let mut parser = Qwen3CoderToolStreamParser::new(&weather_tools());
        let mut during_value = ToolParseResult::default();
        for character in input[..before_parameter_close - 20].chars() {
            during_value.append(parser.push(&character.to_string()).expect("push"));
        }
        let emitted_arguments: String = during_value
            .calls
            .iter()
            .map(|call| call.arguments.as_str())
            .collect();
        assert!(
            emitted_arguments.contains("café"),
            "content must stream with substantial value remaining: {during_value:?}"
        );
        let out = stream_every_char(&weather_tools(), input);
        assert_eq!(
            out.calls.iter().filter(|call| call.name.is_some()).count(),
            1,
            "Qwen must put the name on the first update only"
        );
        assert!(out.calls.last().expect("call updates").complete);
        let fragments: Vec<_> = out
            .calls
            .iter()
            .filter(|call| !call.arguments.is_empty())
            .collect();
        assert!(
            fragments.len() >= 2,
            "expected multiple argument fragments, got {fragments:?}"
        );
        assert!(
            fragments
                .iter()
                .all(|fragment| !fragment.arguments.contains("</parameter>")),
            "parameter marker leaked into arguments: {fragments:?}"
        );
        assert_eq!(out.coalesce_calls(), baseline);
    }

    #[test]
    fn streams_a_long_second_string_parameter_before_function_close() {
        let content = "A long report body that must continue arriving while the function is open.";
        let input = format!(
            "<tool_call><function=create_file><parameter=path>report.md</parameter><parameter=content>{content}</parameter></function></tool_call>"
        );
        let close = input.find("</function>").unwrap();
        let mut parser = Qwen3CoderToolStreamParser::new(&create_file_tools());
        let mut before_close = ToolParseResult::default();
        for character in input[..close].chars() {
            before_close.append(parser.push(&character.to_string()).expect("push"));
        }
        let emitted: String = before_close
            .calls
            .iter()
            .map(|call| call.arguments.as_str())
            .collect();
        assert!(emitted.contains(r#"{"path":"report.md","content":""#));
        assert!(emitted.contains("must continue arriving"));

        before_close.append(parser.push(&input[close..]).expect("close"));
        before_close.append(parser.finish().expect("finish"));
        assert_eq!(
            before_close.coalesce_calls().calls[0].arguments,
            format!(
                r#"{{"path":"report.md","content":{}}}"#,
                serde_json::to_string(content).unwrap()
            )
        );
    }

    #[test]
    fn empty_first_string_does_not_block_a_long_second_string() {
        let input = "<tool_call><function=create_file><parameter=path></parameter><parameter=content>long second value that must stream</parameter></function></tool_call>";
        let close = input.find("</function>").unwrap();
        let mut parser = Qwen3CoderToolStreamParser::new(&create_file_tools());
        let mut before_close = ToolParseResult::default();
        for character in input[..close].chars() {
            before_close.append(parser.push(&character.to_string()).expect("push"));
        }
        let emitted: String = before_close
            .calls
            .iter()
            .map(|call| call.arguments.as_str())
            .collect();
        assert!(emitted.contains("long second value that must stream"));
        before_close.append(parser.push(&input[close..]).expect("close"));
        assert_eq!(
            before_close.coalesce_calls().calls[0].arguments,
            r#"{"path":"","content":"long second value that must stream"}"#
        );
    }

    #[test]
    fn defers_non_string_parameter_until_function_close() {
        let tools = vec![Tool {
            name: "set_count".to_string(),
            description: None,
            parameters: serde_json::json!({
                "type": "object",
                "properties": { "count": { "type": "integer" } }
            }),
            strict: None,
        }];
        let mut parser = Qwen3CoderToolStreamParser::new(&tools);
        let open = "<tool_call><function=set_count><parameter=count>42</parameter>";
        assert!(parser.push(open).unwrap().calls.is_empty());
        let closed = parser.push("</function></tool_call>").unwrap();
        assert_eq!(closed.calls.len(), 1);
        assert_eq!(closed.calls[0].arguments, r#"{"count":42}"#);
    }

    #[test]
    fn defers_html_entities_until_complete_typing() {
        let input = "<tool_call><function=get_weather><parameter=location>BEGIN-&amp;-LONG-TAIL-THAT-MUST-STREAM</parameter></function></tool_call>";
        let entity = input.find('&').unwrap();
        let mut parser = Qwen3CoderToolStreamParser::new(&weather_tools());
        let mut before_entity = ToolParseResult::default();
        for character in input[..entity].chars() {
            before_entity.append(parser.push(&character.to_string()).expect("push"));
        }
        let emitted: String = before_entity
            .calls
            .iter()
            .map(|call| call.arguments.as_str())
            .collect();
        assert!(emitted.contains("BEGIN-"));

        let close = input.find("</function>").unwrap();
        for character in input[entity..close].chars() {
            before_entity.append(parser.push(&character.to_string()).expect("push"));
        }
        let emitted_before_close: String = before_entity
            .calls
            .iter()
            .map(|call| call.arguments.as_str())
            .collect();
        assert!(emitted_before_close.contains("&-LONG-TAIL-THAT-MUST-STREAM"));

        before_entity.append(parser.push(&input[close..]).expect("close"));
        before_entity.append(parser.finish().expect("finish"));
        assert_eq!(
            before_entity
                .coalesce_calls()
                .calls
                .into_iter()
                .map(|call| call.arguments)
                .collect::<String>(),
            r#"{"location":"BEGIN-&-LONG-TAIL-THAT-MUST-STREAM"}"#
        );
    }

    #[test]
    fn longer_entity_prefix_waits_for_disambiguation() {
        let input = "<tool_call><function=get_weather><parameter=location>&amp;quot;tail</parameter></function></tool_call>";
        let split = input.find("&amp;").unwrap() + "&amp;".len();
        let baseline = parse_chunks(&weather_tools(), &[input]).coalesce_calls();
        assert_eq!(
            parse_chunks(&weather_tools(), &[&input[..split], &input[split..]]).coalesce_calls(),
            baseline
        );
    }
}
