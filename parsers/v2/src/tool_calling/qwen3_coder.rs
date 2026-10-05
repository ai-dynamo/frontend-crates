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
//! `parse_qwen_invoke`, which retains Qwen literal parameter text while reusing
//! schema-directed value typing. Arguments are re-serialized in the
//! source parameter order because the v1 parser builds them from a `HashMap`
//! whose key order is non-deterministic; streaming fixtures store the arguments
//! as an exact JSON string, so order has to be pinned to the model-emitted
//! order (the order vLLM's Rust parser also preserves).

use crate::tool_calling::scan::{
    BareRecoveryLatch, InvokeBoundary, InvokeBoundaryFactory, InvokeEmitter, InvokeLatch,
    WrappedBlockScanner, WrappedBlockSpec, marker_prefix_suffix_len, reorder_arguments,
};
use crate::tool_calling::v1core::{ToolDefinition, parse_qwen_invoke};

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
        invoke_boundary_factory: Some(InvokeBoundaryFactory::NativeOnly(|| {
            Box::new(QwenInvokeBoundary::default())
        })),
    }
}

/// Cursor for the shared scanner's active invoke. Only a parameter closer
/// releases ownership of a value; function and tool markers inside it are data.
#[derive(Default)]
struct QwenInvokeBoundary {
    cursor: usize,
    in_parameter: bool,
    parameter_value_start: usize,
}

impl InvokeBoundary for QwenInvokeBoundary {
    fn end_append(
        &mut self,
        candidate: &str,
        _append: &str,
        flush: bool,
        _tool_index: usize,
    ) -> Option<usize> {
        loop {
            let tail = &candidate[self.cursor..];
            if self.in_parameter {
                if let Some(end) = tail.find("</parameter>") {
                    self.cursor += end + "</parameter>".len();
                    self.in_parameter = false;
                } else {
                    if flush {
                        return candidate[self.parameter_value_start..]
                            .find(FUNCTION_END)
                            .map(|at| self.parameter_value_start + at + FUNCTION_END.len());
                    }
                    self.cursor =
                        candidate.len() - marker_prefix_suffix_len(tail, ["</parameter>"]);
                    return None;
                }
            } else {
                let parameter = tail.find(PARAMETER_START);
                let close = tail.find(FUNCTION_END);
                if let Some(close) = close
                    && parameter.is_none_or(|parameter| close < parameter)
                {
                    return Some(self.cursor + close + FUNCTION_END.len());
                }
                if let Some(parameter) = parameter {
                    self.cursor += parameter;
                    let header_end = candidate[self.cursor..].find('>')?;
                    self.cursor += header_end + 1;
                    self.in_parameter = true;
                    self.parameter_value_start = self.cursor;
                } else {
                    self.cursor = candidate.len()
                        - marker_prefix_suffix_len(tail, [PARAMETER_START, FUNCTION_END]);
                    return None;
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

/// Value-typing hook: types one complete `<function=...></function>` block and
/// re-orders the arguments to source order.
pub(crate) struct Qwen3Emitter {
    tools: Vec<ToolDefinition>,
    partial: Option<PartialStringArgument>,
    #[cfg(test)]
    searched_bytes: usize,
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
    // Searching advances even when EOF recovery requires withholding value bytes.
    search_cursor: usize,
    ambiguous_function_close: Option<usize>,
    at_start: bool,
    pending_newline: bool,
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
                let search = &invoke[active.search_cursor..];
                #[cfg(test)]
                {
                    self.searched_bytes += search.len();
                }
                let close = search
                    .find("</parameter>")
                    .map(|offset| active.search_cursor + offset);
                // A function closer without a parameter closer is ambiguous until
                // EOF. Retain it for legacy missing-parameter-close recovery; a
                // later parameter closer confirms the bytes are literal data.
                if close.is_none() && active.ambiguous_function_close.is_none() {
                    #[cfg(test)]
                    {
                        self.searched_bytes += search.len();
                    }
                    active.ambiguous_function_close = search
                        .find(FUNCTION_END)
                        .map(|offset| active.search_cursor + offset);
                }
                active.search_cursor = invoke.len() - qwen_partial_suffix_len(search);
                let safe_end = close
                    .or(active.ambiguous_function_close)
                    .unwrap_or(active.search_cursor);
                let mut fragment = String::new();
                append_literal_string_fragment(
                    active,
                    &invoke[active.value_cursor..safe_end],
                    &mut fragment,
                );
                active.value_cursor = safe_end;
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
                .is_some_and(|schema| {
                    schema.get("type").and_then(serde_json::Value::as_str) == Some("string")
                        && schema.get("nullable").and_then(serde_json::Value::as_bool) != Some(true)
                });
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
                search_cursor: value_start,
                ambiguous_function_close: None,
                at_start: true,
                pending_newline: false,
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
        let Some(call) = parse_qwen_invoke(invoke, &self.tools)? else {
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
            tools: tools.iter().map(ToolDefinition::from).collect(),
            partial: None,
            #[cfg(test)]
            searched_bytes: 0,
        },
    )
}

/// Drop only the framing newlines. Hold the last newline until another byte
/// proves it belongs to the value, preserving all spaces and entity spellings.
fn append_literal_string_fragment(
    active: &mut ActiveStringParameter,
    raw: &str,
    output: &mut String,
) {
    if raw.is_empty() {
        return;
    }
    let raw = if active.at_start {
        active.at_start = false;
        raw.strip_prefix('\n').unwrap_or(raw)
    } else {
        raw
    };
    if raw.is_empty() {
        return;
    }
    let mut fragment = String::new();
    if active.pending_newline {
        fragment.push('\n');
    }
    active.pending_newline = raw.ends_with('\n');
    fragment.push_str(if active.pending_newline {
        &raw[..raw.len() - 1]
    } else {
        raw
    });
    output.push_str(&json_string_fragment(&fragment));
    active.started |= !fragment.is_empty();
}

fn json_string_fragment(text: &str) -> String {
    let encoded = serde_json::to_string(text).expect("serializing a string cannot fail");
    encoded[1..encoded.len() - 1].to_string()
}

/// Hold split parameter and function closers until ownership is known.
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
        let value_start = start + header_end + 1;
        let Some(close) = function[value_start..].find("</parameter>") else {
            break;
        };
        cursor = value_start + close + "</parameter>".len();
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
    fn literal_function_closer_scanning_is_linear() {
        let tools = weather_tools();
        let mut emitter = Qwen3Emitter {
            tools: tools.iter().map(ToolDefinition::from).collect(),
            partial: None,
            searched_bytes: 0,
        };
        for length in [4096, 8192] {
            emitter.reset();
            emitter.searched_bytes = 0;
            let value = format!("prefix</function>{}", "x".repeat(length));
            let invoke =
                format!("<function=get_weather><parameter=location>{value}</parameter></function>");
            let mut arguments = String::new();
            for end in 1..=invoke.len() {
                if let Some(delta) = emitter.parse_partial_invoke(&invoke[..end], 0).unwrap() {
                    arguments.push_str(&delta.arguments);
                }
            }
            if let Some(delta) = emitter.parse_invoke(&invoke, 0).unwrap() {
                arguments.push_str(&delta.arguments);
            }
            assert_eq!(
                serde_json::from_str::<serde_json::Value>(&arguments).unwrap(),
                serde_json::json!({"location": value})
            );
            assert!(
                emitter.searched_bytes <= 32 * invoke.len(),
                "searched {} bytes for {} input bytes",
                emitter.searched_bytes,
                invoke.len()
            );
        }
    }

    #[test]
    fn missing_parameter_close_recovers_at_eof_at_every_split() {
        let input = "<tool_call>\n<function=get_weather>\n<parameter=location>\nNYC\n</function>\n</tool_call>";
        for split in 0..=input.len() {
            let output = parse_chunks(&weather_tools(), &[&input[..split], &input[split..]])
                .coalesce_calls();
            assert_eq!(output.calls.len(), 1, "split {split}");
            assert_eq!(
                output.calls[0].arguments, r#"{"location":"NYC"}"#,
                "split {split}"
            );
        }
        let chunks: Vec<_> = input
            .char_indices()
            .map(|(at, c)| &input[at..at + c.len_utf8()])
            .collect();
        assert_eq!(
            parse_chunks(&weather_tools(), &chunks)
                .coalesce_calls()
                .calls[0]
                .arguments,
            r#"{"location":"NYC"}"#
        );
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
        assert_eq!(out.calls[0].arguments, r#"{"location":" NYC "}"#);
    }

    fn assert_argument_chunks(schema: serde_json::Value, raw: &str, expected: serde_json::Value) {
        use crate::unified::UnifiedParserExt;
        let mut tools = weather_tools();
        tools[0].parameters["properties"]["location"] = schema.clone();
        let input = format!(
            "<tool_call><function=get_weather><parameter=location>{raw}</parameter></function></tool_call>"
        );
        for width in [1, input.len()] {
            let chunks: Vec<_> = input
                .as_bytes()
                .chunks(width)
                .map(|chunk| std::str::from_utf8(chunk).unwrap())
                .collect();
            let output = parse_chunks(&tools, &chunks).coalesce_calls();
            assert_eq!(output.calls.len(), 1, "schema {schema}, width {width}");
            assert!(output.calls[0].complete, "schema {schema}, width {width}");
            assert_eq!(
                serde_json::from_str::<serde_json::Value>(&output.calls[0].arguments).unwrap(),
                serde_json::json!({"location": expected}),
                "schema {schema}, width {width}"
            );
            let mut parser = crate::unified::qwen3::qwen3_unified(&tools);
            let mut deltas = Vec::new();
            for chunk in &chunks {
                deltas.extend(parser.push(chunk).expect("unified push"));
            }
            deltas.extend(parser.finish().expect("unified finish").events);
            assert_eq!(
                crate::unified::assemble(&deltas),
                vec![crate::unified::UnifiedEvent::ToolCall {
                    name: "get_weather".into(),
                    arguments: serde_json::json!({"location": expected}),
                }],
                "unified schema {schema}, width {width}"
            );
        }
    }

    #[test]
    fn padded_scalar_arguments_follow_the_schema() {
        use serde_json::{Value, json};

        for (schema, raw, expected) in [
            (json!({"type": "boolean"}), " true ", json!(true)),
            (json!({"type": "boolean"}), " false ", json!(false)),
            (json!({"type": "integer"}), " 42 ", json!(42)),
            (json!({"type": "number"}), " 1.5 ", json!(1.5)),
            (json!({"type": "null"}), " null ", Value::Null),
            (json!({"type": ["integer", "null"]}), " null ", Value::Null),
            (json!({"type": ["integer", "null"]}), " 42 ", json!(42)),
            (
                json!({"anyOf": [{"type": "boolean"}, {"type": "null"}]}),
                " true ",
                json!(true),
            ),
            (
                json!({"anyOf": [{"type": "number"}, {"type": "null"}]}),
                " 1.5 ",
                json!(1.5),
            ),
            (
                json!({"type": "string", "nullable": true}),
                " null ",
                Value::Null,
            ),
            (
                json!({"type": "string", "nullable": true}),
                " 42 ",
                json!(" 42 "),
            ),
            (json!({"type": "string"}), " true ", json!(" true ")),
            (json!({"type": "string"}), " null ", json!(" null ")),
            (
                json!({"anyOf": [{"type": "string"}, {"type": "null"}]}),
                " 42 ",
                json!(" 42 "),
            ),
            (
                json!({"anyOf": [{"type": "string"}, {"type": "integer"}]}),
                " null ",
                json!(" null "),
            ),
            (json!({"type": "integer"}), " invalid ", json!(" invalid ")),
        ] {
            assert_argument_chunks(schema, raw, expected);
        }
    }

    #[test]
    fn null_text_completes_with_the_schema_selected_type() {
        for (schema, expected) in [
            (
                serde_json::json!({"type": "string"}),
                serde_json::json!("null"),
            ),
            (
                serde_json::json!({"anyOf": [{"type": "string"}, {"type": "integer"}]}),
                serde_json::json!("null"),
            ),
            (
                serde_json::json!({"type": ["string", "null"]}),
                serde_json::Value::Null,
            ),
            (
                serde_json::json!({"type": "string", "nullable": true}),
                serde_json::Value::Null,
            ),
            (
                serde_json::json!({"type": "string", "anyOf": [
                    {"type": "string"}, {"type": "null"}
                ]}),
                serde_json::json!("null"),
            ),
            (
                serde_json::json!({"type": ["string", "null"], "oneOf": [
                    {"type": "string"}, {"type": "integer"}
                ]}),
                serde_json::json!("null"),
            ),
            (
                serde_json::json!({"type": "string", "anyOf": [
                    {"minLength": 1}, {"type": "null"}
                ]}),
                serde_json::json!("null"),
            ),
            (
                serde_json::json!({"anyOf": [{"const": "null"}, {"type": "integer"}]}),
                serde_json::json!("null"),
            ),
            (
                serde_json::json!({"anyOf": [{"const": null}, {"type": "string"}]}),
                serde_json::Value::Null,
            ),
            (
                serde_json::json!({"type": "string", "enum": ["null", null]}),
                serde_json::json!("null"),
            ),
        ] {
            assert_argument_chunks(schema, "null", expected);
        }
    }

    #[test]
    fn literal_union_branches_preserve_typed_arguments() {
        for schema in [
            serde_json::json!({"anyOf": [{"const": "auto"}, {"type": "integer"}]}),
            serde_json::json!({"oneOf": [{"enum": ["auto"]}, {"type": "integer"}]}),
        ] {
            for (raw, expected) in [
                ("42", serde_json::json!(42)),
                ("auto", serde_json::json!("auto")),
            ] {
                assert_argument_chunks(schema.clone(), raw, expected);
            }
        }
    }

    #[test]
    fn numeric_literal_constraints_preserve_argument_types() {
        for keyword in ["const", "enum"] {
            for (literal, raw, types, expected) in [
                (
                    serde_json::json!(42.0),
                    "42",
                    serde_json::json!(["integer", "null"]),
                    serde_json::json!(42),
                ),
                (
                    serde_json::json!(42.5),
                    "42.5",
                    serde_json::json!(["number", "null"]),
                    serde_json::json!(42.5),
                ),
            ] {
                let mut schema = serde_json::json!({"type": types});
                schema[keyword] = if keyword == "enum" {
                    serde_json::json!([literal])
                } else {
                    literal
                };
                assert_argument_chunks(schema, raw, expected);
            }
        }
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
    fn streams_entity_spellings_as_literal_text() {
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
        assert!(emitted_before_close.contains("&amp;-LONG-TAIL-THAT-MUST-STREAM"));

        before_entity.append(parser.push(&input[close..]).expect("close"));
        before_entity.append(parser.finish().expect("finish"));
        assert_eq!(
            before_entity
                .coalesce_calls()
                .calls
                .into_iter()
                .map(|call| call.arguments)
                .collect::<String>(),
            r#"{"location":"BEGIN-&amp;-LONG-TAIL-THAT-MUST-STREAM"}"#
        );
    }

    #[test]
    fn entity_spellings_are_chunk_invariant() {
        let input = "<tool_call><function=get_weather><parameter=location>&amp;quot;tail</parameter></function></tool_call>";
        let split = input.find("&amp;").unwrap() + "&amp;".len();
        let baseline = parse_chunks(&weather_tools(), &[input]).coalesce_calls();
        assert_eq!(
            parse_chunks(&weather_tools(), &[&input[..split], &input[split..]]).coalesce_calls(),
            baseline
        );
    }
    #[test]
    fn integral_decimal_arguments_with_literal_constraints() {
        for keyword in ["const", "enum"] {
            for literal in [serde_json::json!(42), serde_json::json!(42.0)] {
                for ty in [
                    serde_json::json!(["number", "null"]),
                    serde_json::json!(["integer", "null"]),
                    serde_json::json!(["number", "string"]),
                ] {
                    let mut schema = serde_json::json!({"type":ty});
                    schema[keyword] = if keyword == "const" {
                        literal.clone()
                    } else {
                        serde_json::json!([literal])
                    };
                    for raw in ["42", "42.0", "4.2e1"] {
                        let mut tools = weather_tools();
                        tools[0].parameters["properties"]["location"] = schema.clone();
                        let input = format!(
                            "<tool_call><function=get_weather><parameter=location>{raw}</parameter></function></tool_call>"
                        );
                        for width in [1, input.len()] {
                            let chunks: Vec<_> = input
                                .as_bytes()
                                .chunks(width)
                                .map(|c| std::str::from_utf8(c).unwrap())
                                .collect();
                            let out = parse_chunks(&tools, &chunks).coalesce_calls();
                            assert_eq!(out.calls.len(), 1);
                            assert_eq!(
                                out.calls[0].arguments, r#"{"location":42}"#,
                                "{schema}, {raw}, width {width}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn integral_decimal_coercion_preserves_precision() {
        for (raw, expected) in [
            ("9007199254740993.0", "9007199254740993"),
            ("9.007199254740993e15", "9007199254740993"),
            ("-4.2E+1", "-42"),
            ("4200e-2", "42"),
            ("0.0e-400", "0"),
            ("42.0000000000000001", "\"42.0000000000000001\""),
            ("1e-400", "\"1e-400\""),
            ("42.5", "\"42.5\""),
        ] {
            let mut tools = weather_tools();
            tools[0].parameters["properties"]["location"] =
                serde_json::json!({"type":["integer","string"]});
            let input = format!(
                "<tool_call><function=get_weather><parameter=location>{raw}</parameter></function></tool_call>"
            );
            for width in [1, input.len()] {
                let chunks: Vec<_> = input
                    .as_bytes()
                    .chunks(width)
                    .map(|c| std::str::from_utf8(c).unwrap())
                    .collect();
                let out = parse_chunks(&tools, &chunks).coalesce_calls();
                assert_eq!(out.calls.len(), 1);
                assert_eq!(
                    out.calls[0].arguments,
                    format!(r#"{{"location":{expected}}}"#),
                    "{raw}, width {width}"
                );
            }
        }
    }
    #[test]
    fn large_fractional_arguments_remain_exact_json_numbers() {
        for raw in [
            "9007199254740992.5",
            "9007199254740993.1",
            "9007199254740993.25",
            "9.0071992547409925e15",
            "0.10000000000000000001",
            "-9007199254740992.5",
        ] {
            for schema in [
                serde_json::json!({"type":"number"}),
                serde_json::json!({"type":["number","null"],"const":serde_json::from_str::<serde_json::Value>(raw).unwrap()}),
                serde_json::json!({"type":["number","null"],"enum":[serde_json::from_str::<serde_json::Value>(raw).unwrap()]}),
            ] {
                let mut tools = weather_tools();
                tools[0].parameters["properties"]["location"] = schema.clone();
                let input = format!(
                    "<tool_call><function=get_weather><parameter=location>{raw}</parameter></function></tool_call>"
                );
                for width in [1, input.len()] {
                    let chunks: Vec<_> = input
                        .as_bytes()
                        .chunks(width)
                        .map(|c| std::str::from_utf8(c).unwrap())
                        .collect();
                    let out = parse_chunks(&tools, &chunks).coalesce_calls();
                    assert_eq!(out.calls.len(), 1);
                    assert_eq!(
                        out.calls[0].arguments,
                        format!(r#"{{"location":{raw}}}"#),
                        "{schema}, width {width}"
                    );
                }
            }
        }
    }
}
