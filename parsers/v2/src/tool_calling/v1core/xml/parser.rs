// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Reference implementation:
// https://github.com/sgl-project/sglang/blob/44da737770e4bcd9bfa27751f0a0751c9b5c06e1/python/sglang/srt/function_call/qwen3_coder_detector.py

use std::collections::{HashMap, HashSet};

use num_traits::ToPrimitive;
use regex::Regex;
use serde_json::Value;
use uuid::Uuid;

use super::super::ToolDefinition;
use super::super::config::XmlParserConfig;
use super::parsed_value::{
    ParsedValue, coerce_integer_literal, is_integer_literal, raw_number_literal,
};
use super::response::{CalledFunction, ToolCallResponse, ToolCallType};

/// Build a `<start>name>(body)<end>` regex pattern. When `strict` is false,
/// missing `<end>` falls back to end-of-block so truncated input still parses
/// best-effort. Strict mode requires both fences and returns no match without
/// the close tag.
fn build_block_pattern(start: &str, end: &str, strict: bool) -> String {
    let start = regex::escape(start);
    let end = regex::escape(end);
    if strict {
        format!(r"(?s){}([^>]+)>(.*?){}", start, end)
    } else {
        format!(r"(?s){}([^>]+)>(.*?)(?:{}|$)", start, end)
    }
}

/// Strip surrounding quotes from a string if present
fn strip_quotes(s: &str) -> &str {
    let trimmed = s.trim();
    // Require at least two bytes: a lone `"` or `'` satisfies both starts_with and
    // ends_with on the SAME char, and `trimmed[1..len-1]` = `trimmed[1..0]` panics.
    if trimmed.len() >= 2
        && ((trimmed.starts_with('"') && trimmed.ends_with('"'))
            || (trimmed.starts_with('\'') && trimmed.ends_with('\'')))
    {
        &trimmed[1..trimmed.len() - 1]
    } else {
        trimmed
    }
}

/// Try to parse Qwen3Coder formatted tool calls from a message.
/// Format: `<tool_call><function=name><parameter=key>value</parameter></function></tool_call>`
/// Returns (parsed_tool_calls, normal_text_content)
pub fn try_tool_call_parse_xml(
    message: &str,
    config: &XmlParserConfig,
    tools: Option<&[ToolDefinition]>,
) -> anyhow::Result<(Vec<ToolCallResponse>, Option<String>)> {
    // Qwen3-Coder-style passthrough: if the function-start token is absent
    // anywhere in the input, the reference parser returns the raw input as
    // content with no tool calls. Gated so it only fires for parsers that opt
    // in (e.g. qwen3_coder, nemotron_nano).
    if config.passthrough_when_no_function
        && !message.contains(config.function_start_token.as_str())
        && !message.contains(config.tool_call_start_token.as_str())
    {
        return Ok((vec![], Some(message.to_string())));
    }

    // Back-off: outer wrapper missing but function/invoke tags are present —
    // parse the whole input as a single tool-call block. Qwen3-Coder uses this
    // in its reference parser. MiniMax uses the same path to recover complete
    // inner invokes when the outer wrapper opener is missing.
    //
    // Streaming recovery needs the orphan outer close marker as well as the
    // inner close marker. Otherwise a split `</tool_call>` / MiniMax close can
    // arrive after the jail has already released.
    if config.is_bare_function_mode(message)
        && (message.contains(config.function_end_token.as_str()) || config.allow_eof_recovery)
        && (message.contains(config.tool_call_end_token.as_str()) || config.allow_eof_recovery)
    {
        let calls = parse_tool_call_block(message, config, tools).unwrap_or_default();
        if !calls.is_empty() {
            // Preserve narration BEFORE the first `<function=...>` tag AND any
            // text AFTER the recovered closing marker so streaming output isn't
            // dropped on the back-off path.
            let marker_idx = message
                .find(config.function_start_token.as_str())
                .unwrap_or(0);
            let text = bare_recovery_surrounding_text(message, marker_idx, config);
            return Ok((calls, Some(text)));
        }
    }

    if config.strict_match
        && !message.contains(config.tool_call_start_token.as_str())
        && let Some(prefix) = prefix_before_orphan_xml_marker(message, config)
    {
        return Ok((vec![], Some(prefix)));
    }

    let (normal_text, tool_calls) = extract_tool_calls(message, config, tools)?;

    let normal_content = if normal_text.is_empty() {
        Some("".to_string())
    } else {
        Some(normal_text)
    };

    Ok((tool_calls, normal_content))
}

/// Extract tool calls and normal text from message.
fn extract_tool_calls(
    text: &str,
    config: &XmlParserConfig,
    tools: Option<&[ToolDefinition]>,
) -> anyhow::Result<(String, Vec<ToolCallResponse>)> {
    let mut normal_text = String::new();
    let mut calls = Vec::new();
    let mut cursor = 0;

    let start_token = &config.tool_call_start_token;
    let end_token = &config.tool_call_end_token;

    while cursor < text.len() {
        // Find next tool call start.
        if let Some(start_pos) = text[cursor..].find(start_token.as_str()) {
            let abs_start = cursor + start_pos;
            let gap = &text[cursor..abs_start];
            if let Some((prefix, mut recovered_calls)) =
                recover_bare_xml_calls_in_span(gap, config, tools)?
            {
                normal_text.push_str(&prefix);
                calls.append(&mut recovered_calls);
            } else {
                normal_text.push_str(gap);
            }

            // Preserve ALL natural-language text — prefix, text between calls,
            // and text after the last call — and strip only the tool-call
            // markup (start marker through end marker). The gap before this
            // start token is appended above unconditionally; the block markup
            // itself is skipped when the cursor advances past `abs_end`.

            // Find the corresponding end token.
            if let Some(end_pos) = text[abs_start..].find(end_token.as_str()) {
                let abs_end = abs_start + end_pos + end_token.len();
                let block = &text[abs_start..abs_end];

                // Parse this tool call block.
                if let Ok(mut parsed_calls) = parse_tool_call_block(block, config, tools) {
                    calls.append(&mut parsed_calls);
                }

                cursor = abs_end;
            } else {
                // Recovery: outer end token absent (max_tokens / EOS truncation).
                // Gated on `allow_eof_recovery` so streaming early-exit doesn't
                // fire mid-stream. Strict XML families still require paired
                // inner invoke/parameter fences before a call is recovered.
                // Recovery also requires the trailing slice to contain a
                // function-start opener — structural signal that a real tool
                // call was emitted, so plain text starting with `<tool_call>`
                // is preserved verbatim.
                let block = &text[abs_start..];
                let function_start = &config.function_start_token;
                let looks_like_tool_call = block.contains(function_start.as_str())
                    || block.contains(config.parameter_start_token.as_str());
                if config.allow_eof_recovery
                    && looks_like_tool_call
                    && let Ok(mut parsed_calls) = parse_tool_call_block(block, config, tools)
                    && !parsed_calls.is_empty()
                {
                    calls.append(&mut parsed_calls);
                    break;
                }
                // No end marker and either no tool-call structure or
                // unrecoverable: a `<start_token>` with no closing markup is
                // natural text, not strippable markup, so preserve it verbatim.
                // Malformed/unrecoverable tool-call blocks keep the
                // drop-without-leak behavior (nothing appended).
                if !looks_like_tool_call {
                    normal_text.push_str(&text[abs_start..]);
                }
                break;
            }
        } else {
            // No more tool calls — preserve the trailing text after the last
            // parsed call verbatim (RULE: only tool-call markup is stripped).
            let gap = &text[cursor..];
            if let Some((prefix, mut recovered_calls)) =
                recover_bare_xml_calls_in_span(gap, config, tools)?
            {
                normal_text.push_str(&prefix);
                calls.append(&mut recovered_calls);
            } else {
                normal_text.push_str(gap);
            }
            break;
        }
    }

    let normal_text = if calls.is_empty() {
        normal_text.trim().to_string()
    } else {
        normal_text
    };
    Ok((normal_text, calls))
}

fn prefix_before_orphan_xml_marker(text: &str, config: &XmlParserConfig) -> Option<String> {
    [
        config.tool_call_end_token.as_str(),
        config.function_start_token.as_str(),
        config.function_end_token.as_str(),
        config.parameter_start_token.as_str(),
        config.parameter_end_token.as_str(),
    ]
    .into_iter()
    .filter_map(|marker| text.find(marker))
    .min()
    .map(|idx| text[..idx].trim().to_string())
}

fn recover_bare_xml_calls_in_span(
    span: &str,
    config: &XmlParserConfig,
    tools: Option<&[ToolDefinition]>,
) -> anyhow::Result<Option<(String, Vec<ToolCallResponse>)>> {
    if !config.backoff_when_no_wrapper {
        return Ok(None);
    }

    let Some(marker_idx) = span.find(config.function_start_token.as_str()) else {
        return Ok(None);
    };

    let tail = &span[marker_idx..];
    let has_inner_close = tail.contains(config.function_end_token.as_str());
    let has_outer_close = tail.contains(config.tool_call_end_token.as_str());
    if (!has_inner_close || !has_outer_close) && !config.allow_eof_recovery {
        return Ok(None);
    }

    let calls = parse_tool_call_block(tail, config, tools)?;
    if calls.is_empty() {
        return Ok(None);
    }

    tracing::warn!(
        why = "bare_function_gap_recovery",
        recovered_calls = calls.len(),
        recovered_bytes = tail.len(),
        kept_prefix_bytes = marker_idx,
        "XML recovery: recovered complete bare function block(s) before a later outer wrapper"
    );
    // Keep BOTH the prefix before the marker AND any narration after the
    // recovered close — otherwise trailing text in the gap is dropped.
    let text = bare_recovery_surrounding_text(span, marker_idx, config);
    Ok(Some((text, calls)))
}

/// Natural-language text preserved around a recovered bare-function block: the
/// prefix before the function-start marker at `marker_idx`, plus any narration
/// after the last recovered close marker (`</function>` or `</tool_call>`).
/// Both bare-call recovery paths would otherwise treat everything from the first
/// function marker onward as consumed and drop trailing text.
fn bare_recovery_surrounding_text(
    span: &str,
    marker_idx: usize,
    config: &XmlParserConfig,
) -> String {
    let prefix = &span[..marker_idx];
    let consumed_end = [
        config.function_end_token.as_str(),
        config.tool_call_end_token.as_str(),
    ]
    .into_iter()
    .filter_map(|tok| span.rfind(tok).map(|i| i + tok.len()))
    .max();
    let trailing = consumed_end.map(|end| &span[end..]).unwrap_or("");
    format!("{prefix}{trailing}")
}

/// Parse a single tool call block
/// Format: `<tool_call><function=name><parameter=key>value</parameter>...</function></tool_call>`
fn parse_tool_call_block(
    block: &str,
    config: &XmlParserConfig,
    tools: Option<&[ToolDefinition]>,
) -> anyhow::Result<Vec<ToolCallResponse>> {
    // Strict-match families (e.g. minimax_m2) require paired fences; lenient
    // families fall back to end-of-block when the close tag is missing.
    let function_regex = Regex::new(&build_block_pattern(
        &config.function_start_token,
        &config.function_end_token,
        config.strict_match,
    ))?;
    let parameter_regex = Regex::new(&build_block_pattern(
        &config.parameter_start_token,
        &config.parameter_end_token,
        config.strict_match,
    ))?;

    let mut results = Vec::new();

    // Find all function blocks.
    for func_cap in function_regex.captures_iter(block) {
        let function_name_raw = func_cap.get(1).map(|m| m.as_str().trim()).unwrap_or("");
        let function_name = strip_quotes(function_name_raw);
        let function_body = func_cap.get(2).map(|m| m.as_str()).unwrap_or("");

        if function_name.is_empty() {
            continue;
        }

        // Truncation guard: when a function block is recovered without its
        // `</function>` close (EOF / max_tokens cut) and its trailing
        // `<parameter=...>` value runs to raw end-of-input with no following
        // close marker, the value was cut off mid-stream. The lenient recovery
        // regex would capture the partial value via its `$` fallback, but an
        // unterminated value can't be trusted (e.g. "New York" may be a truncated
        // "New York City"), so drop the whole call rather than emit a guessed
        // argument. A value bounded by *any* close marker — `</parameter>`,
        // `</function>` (function terminated), or `</tool_call>` — is complete and
        // still recovered.
        let function_terminated = func_cap
            .get(0)
            .is_some_and(|m| m.as_str().contains(config.function_end_token.as_str()));
        if !function_terminated
            && let Some(open_idx) = function_body.rfind(config.parameter_start_token.as_str())
        {
            let after_value = &function_body[open_idx..];
            let value_bounded = after_value.contains(config.parameter_end_token.as_str())
                || after_value.contains(config.tool_call_end_token.as_str());
            if !value_bounded {
                continue;
            }
        }

        // Get parameter config for this function
        let param_config = get_arguments_config(function_name, tools);

        // Parse parameters from the function body.
        let mut parameters: HashMap<String, ParsedValue> = HashMap::new();

        for param_cap in parameter_regex.captures_iter(function_body) {
            let param_name_raw = param_cap.get(1).map(|m| m.as_str().trim()).unwrap_or("");
            let param_name = strip_quotes(param_name_raw);
            let param_value = param_cap.get(2).map(|m| m.as_str()).unwrap_or("");

            if !param_name.is_empty() {
                let parsed_value =
                    convert_param_value(param_value, param_name, &param_config, function_name);
                parameters.insert(param_name.to_string(), parsed_value);
            }
        }

        // Create tool call response.
        let arguments_json = serde_json::to_string(&parameters)?;

        let tool_call = ToolCallResponse {
            id: format!("call-{}", Uuid::new_v4()),
            tp: ToolCallType::Function,
            function: CalledFunction {
                name: function_name.to_string(),
                arguments: arguments_json,
            },
        };

        results.push(tool_call);
    }

    Ok(results)
}

/// Extract argument configuration for a function from the tool definitions.
/// Returns a HashMap of parameter names to their schema definitions.
fn get_arguments_config(
    func_name: &str,
    tools: Option<&[ToolDefinition]>,
) -> HashMap<String, Value> {
    let Some(tools) = tools else {
        return HashMap::new();
    };

    for tool in tools {
        if tool.name == func_name {
            if let Some(params) = &tool.parameters {
                // Try to extract "properties" from the parameters schema
                if let Some(properties) = params.get("properties") {
                    if let Some(props_obj) = properties.as_object() {
                        return props_obj
                            .iter()
                            .map(|(k, v)| (k.clone(), v.clone()))
                            .collect();
                    }
                } else if let Some(params_obj) = params.as_object() {
                    // If no "properties" field, treat the whole thing as the config
                    return params_obj
                        .iter()
                        .map(|(k, v)| (k.clone(), v.clone()))
                        .collect();
                }
            }
            return HashMap::new();
        }
    }

    tracing::warn!("Tool '{}' is not defined in the tools list.", func_name);
    HashMap::new()
}

/// Convert parameter value based on its type in the schema.
/// This matches the behavior of the Python implementation.
/// Converts a string parameter value from XML into a typed JSON Value.
///
/// # Examples
///
/// **String types:**
/// ```text
/// Input:  param_value="hello world", param_type="string"
/// Output: Value::String("hello world")
/// ```
///
/// ```text
/// Input:  param_value="42", param_type="string"
/// Output: Value::String("42")
/// ```
///
/// **Integer types:**
/// ```text
/// Input:  param_value="42", param_type="integer"
/// Output: Value::Number(42)
///
/// Input:  param_value="not_a_number", param_type="int"
/// Output: Value::String("not_a_number")  // Falls back to string with warning
/// ```
///
/// **Float/Number types:**
/// ```text
/// Input:  param_value="3.14", param_type="number"
/// Output: Value::Number(3.14)
///
/// Input:  param_value="42.0", param_type="float"
/// Output: Value::Number(42)  // Whole numbers stored as integers
/// ```
///
/// **Boolean types:**
/// ```text
/// Input:  param_value="true", param_type="boolean"
/// Output: Value::Bool(true)
///
/// Input:  param_value="yes", param_type="bool"
/// Output: Value::Bool(false)  // Falls back to false with warning
/// ```
///
/// **Complex types (objects/arrays):**
/// ```text
/// Input:  param_value='{"key": "value"}', param_type="object"
/// Output: Value::Object({"key": "value"})
///
/// Input:  param_value="[1, 2, 3]", param_type="array"
/// Output: Value::Array([1, 2, 3])
///
/// Input:  param_value="{'key': 'value'}", param_type="dict"
/// Output: Value::Object({"key": "value"})  // Uses ast.literal_eval-style parsing
/// ```
///
/// **Special cases:**
/// ```text
/// Input:  param_value="null", param_type=<any>
/// Output: Value::Null  // Handled before type checking
///
/// Input:  param_value="&lt;tag&gt;", param_type="string"
/// Output: Value::String("<tag>")  // HTML entities are unescaped
///
/// Input:  param_value="123", param_type=<undefined/not in schema>
/// Output: Value::String("123")  // Unknown params returned as strings
/// ```
///
/// # Arguments
///
/// * `param_value` - The raw string value from XML parameter
/// * `param_name` - The parameter name (used for schema lookup and error messages)
/// * `param_config` - Schema defining expected types for each parameter
/// * `func_name` - The function/tool name (used for error messages)
///
/// # Type Aliases
///
/// The function recognizes various type name aliases:
/// - Strings: "string", "str", "text", "varchar", "char", "enum"
/// - Integers: "int", "integer", "int32", "int64", "uint", "long", "short", "unsigned"
/// - Numbers: "number", "num", "float", "float32", "float64", "double"
/// - Booleans: "boolean", "bool", "binary"
/// - Objects: "object", "dict", "dictionary"
/// - Arrays: "array", "arr", "list"
fn convert_param_value(
    param_value: &str,
    param_name: &str,
    param_config: &HashMap<String, Value>,
    func_name: &str,
) -> ParsedValue {
    // HTML unescape and trim
    let param_value = html_unescape(param_value.trim());

    // Handle null
    if param_value.to_lowercase() == "null" {
        return Value::Null.into();
    }

    // Check if parameter is in config
    if !param_config.contains_key(param_name) {
        tracing::debug!(
            "Parsed parameter '{}' is not defined in the tool parameters for tool '{}', directly returning the string value.",
            param_name,
            func_name
        );
        return Value::String(param_value).into();
    }

    // Get the type from schema.
    let param_schema = param_config.get(param_name);
    let direct_type = param_schema
        .and_then(|v| v.get("type"))
        .and_then(|t| t.as_str())
        .map(|t| t.to_lowercase());

    let param_type = match direct_type {
        Some(t) => t,
        None => {
            // No scalar `type` string. The schema may still constrain the value
            // via a union: `anyOf`/`oneOf`, a `type: [..]` array, or OpenAPI
            // `nullable`. Resolve the allowed alternatives and coerce ONLY to a
            // type the union permits — a bare `42` for `anyOf:[string,null]`
            // must stay the string "42", not become the JSON number 42. When no
            // union is present, fall back to the documented string behavior.
            if let Some(schema) = param_schema {
                let allowed = collect_allowed_types(schema);
                if !allowed.is_empty() {
                    return coerce_union_value(&param_value, &allowed);
                }
            }
            "string".to_string()
        }
    };

    // The follow `match` block follows this rough pattern for each block:
    // 1. Match `param_type` against predefined string representations of each type,
    // 2. Parse the string value and convert it to the appropriate Rust JSON Value type.
    // Each branch handles a category of type aliases (e.g., "int"/"integer"/"int32" all map to i64).
    // If parsing fails, we log a warning and fall back to returning the value as a string.
    match param_type.as_str() {
        // String types: Return value as-is (already HTML-unescaped above)
        "string" | "str" | "text" | "varchar" | "char" | "enum" => {
            Value::String(param_value).into()
        }

        // Integer types: Parse as i64, fall back to string on error.
        // Matches: "int", "integer", "int32", "uint", "unsigned", "long", "short", etc.
        t if t.starts_with("int")
            || t.starts_with("uint")
            || t.starts_with("long")
            || t.starts_with("short")
            || t.starts_with("unsigned") =>
        {
            // Preserve large integers as JSON numbers. `coerce_integer_literal`
            // parses to i64 when it fits and falls back to a raw numeric literal
            // (via `serde_json::value::RawValue`) for values outside i64 range,
            // so a 21-digit argument stays a JSON number instead of a string.
            match coerce_integer_literal(&param_value) {
                Some(coerced) => coerced,
                None => {
                    tracing::warn!(
                        "Parsed value '{}' of parameter '{}' is not an integer in tool '{}', degenerating to string.",
                        param_value,
                        param_name,
                        func_name
                    );
                    Value::String(param_value).into()
                }
            }
        }

        // Float/Number types: Parse integer-looking tokens before f64 to avoid
        // precision loss above f64's exact integer range.
        // Matches: "number", "num", "float", "float32", "float64", "double", etc.
        // Note: Whole numbers (e.g., 42.0) are stored as integers for better JSON compatibility
        // when they fit in i64. Larger finite whole numbers must not be cast with `as i64`,
        // which saturates to i64::MIN/MAX and corrupts model-emitted arguments.
        t if t.starts_with("num") || t.starts_with("float") => {
            if is_integer_literal(&param_value) {
                if let Ok(int_val) = param_value.parse::<i64>() {
                    Value::Number(int_val.into()).into()
                } else if let Some(raw) = raw_number_literal(&param_value) {
                    raw
                } else {
                    Value::String(param_value).into()
                }
            } else {
                match param_value.parse::<f64>() {
                    Ok(float_val) => {
                        if float_val.fract() == 0.0 && float_val.is_finite() {
                            if let Some(int_val) = float_val.to_i64() {
                                Value::Number(int_val.into()).into()
                            } else if let Some(raw) = raw_number_literal(&param_value) {
                                raw
                            } else {
                                Value::String(param_value).into()
                            }
                        } else if let Some(num) = serde_json::Number::from_f64(float_val) {
                            Value::Number(num).into()
                        } else {
                            tracing::warn!(
                                "Parsed value '{}' of parameter '{}' is not a valid float in tool '{}', degenerating to string.",
                                param_value,
                                param_name,
                                func_name
                            );
                            Value::String(param_value).into()
                        }
                    }
                    Err(_) => {
                        tracing::warn!(
                            "Parsed value '{}' of parameter '{}' is not a float in tool '{}', degenerating to string.",
                            param_value,
                            param_name,
                            func_name
                        );
                        Value::String(param_value).into()
                    }
                }
            }
        }

        // Boolean types: Only "true" or "false" (case-insensitive) are valid.
        // Any other value defaults to false with a warning.
        "boolean" | "bool" | "binary" => {
            let lower_val = param_value.to_lowercase();
            if lower_val != "true" && lower_val != "false" {
                tracing::warn!(
                    "Parsed value '{}' of parameter '{}' is not a boolean (`true` or `false`) in tool '{}', degenerating to false.",
                    param_value,
                    param_name,
                    func_name
                );
            }
            Value::Bool(lower_val == "true").into()
        }

        // Complex types (objects/arrays): Try JSON parsing, then fall back to Python-style
        // `ast.literal_eval` (or our own barebones version of it for the purposes of this
        // parser).
        // Matches: "object", "array", "arr", "dict", "dictionary", "list", etc.
        // This handles both JSON syntax ({"a": 1}) and Python syntax ({'a': 1}).
        t if t == "object"
            || t == "array"
            || t == "arr"
            || t.starts_with("dict")
            || t.starts_with("list") =>
        {
            // Try JSON parsing first (standard JSON with double quotes).
            if let Ok(json_val) = serde_json::from_str::<Value>(&param_value) {
                return json_val.into();
            }

            tracing::warn!(
                "Parsed value '{}' of parameter '{}' cannot be parsed with json.loads in tool '{}', will try other methods to parse it.",
                param_value,
                param_name,
                func_name
            );

            // Try `ast.literal_eval` equivalent (handles Python-style single quotes, etc.).
            if let Ok(json_val) = try_literal_eval(&param_value) {
                return json_val.into();
            }

            tracing::warn!(
                "Parsed value '{}' of parameter '{}' cannot be converted via Python `ast.literal_eval()` in tool '{}', degenerating to string.",
                param_value,
                param_name,
                func_name
            );
            Value::String(param_value).into()
        }

        // Unknown/custom types: Attempt best-effort parsing via `literal_eval`.
        // This allows for flexible type names while still trying to parse structured data
        _ => {
            // Unknown type, try `literal_eval`.
            if let Ok(json_val) = try_literal_eval(&param_value) {
                return json_val.into();
            }

            tracing::warn!(
                "Parsed value '{}' of parameter '{}' cannot be converted via Python `ast.literal_eval()` in tool '{}', degenerating to string.",
                param_value,
                param_name,
                func_name
            );
            Value::String(param_value).into()
        }
    }
}

/// Coarse JSON-schema type category, used to resolve union (`anyOf`/`oneOf`/
/// `type: [..]`/`nullable`) schemas to the set of types they actually allow.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum SchemaType {
    String,
    Integer,
    Number,
    Boolean,
    Object,
    Array,
    Null,
}

/// Map a schema type name (including the aliases `convert_param_value`
/// recognizes) to its category. Unknown names return `None`.
fn categorize_type(name: &str) -> Option<SchemaType> {
    let t = name.to_lowercase();
    let t = t.as_str();
    if matches!(t, "string" | "str" | "text" | "varchar" | "char" | "enum") {
        Some(SchemaType::String)
    } else if matches!(t, "boolean" | "bool" | "binary") {
        Some(SchemaType::Boolean)
    } else if matches!(t, "null" | "none") {
        Some(SchemaType::Null)
    } else if t.starts_with("int")
        || t.starts_with("uint")
        || t.starts_with("long")
        || t.starts_with("short")
        || t.starts_with("unsigned")
    {
        Some(SchemaType::Integer)
    } else if t.starts_with("num") || t.starts_with("float") {
        Some(SchemaType::Number)
    } else if t == "object" || t.starts_with("dict") {
        Some(SchemaType::Object)
    } else if t == "array" || t == "arr" || t.starts_with("list") {
        Some(SchemaType::Array)
    } else {
        None
    }
}

/// Collect the set of types a (possibly union) schema allows, walking
/// `type` (string or array), `anyOf`/`oneOf` branches, and OpenAPI `nullable`.
fn collect_allowed_types(schema: &Value) -> HashSet<SchemaType> {
    let mut out = HashSet::new();
    collect_allowed_types_into(schema, &mut out);
    out
}

fn collect_allowed_types_into(schema: &Value, out: &mut HashSet<SchemaType>) {
    if let Some(ty) = schema.get("type") {
        if let Some(name) = ty.as_str() {
            if let Some(cat) = categorize_type(name) {
                out.insert(cat);
            }
        } else if let Some(arr) = ty.as_array() {
            for item in arr.iter().filter_map(Value::as_str) {
                if let Some(cat) = categorize_type(item) {
                    out.insert(cat);
                }
            }
        }
    }
    for key in ["anyOf", "oneOf"] {
        if let Some(options) = schema.get(key).and_then(Value::as_array) {
            for option in options {
                collect_allowed_types_into(option, out);
            }
        }
    }
    if schema.get("nullable").and_then(Value::as_bool) == Some(true) {
        out.insert(SchemaType::Null);
    }
}

/// The category a parsed JSON value belongs to (integers report as `Integer`).
fn value_category(v: &Value) -> SchemaType {
    match v {
        Value::String(_) => SchemaType::String,
        Value::Bool(_) => SchemaType::Boolean,
        Value::Number(n) => {
            if n.is_i64() || n.is_u64() {
                SchemaType::Integer
            } else {
                SchemaType::Number
            }
        }
        Value::Object(_) => SchemaType::Object,
        Value::Array(_) => SchemaType::Array,
        Value::Null => SchemaType::Null,
    }
}

fn value_matches(v: &Value, allowed: &HashSet<SchemaType>) -> bool {
    let cat = value_category(v);
    // An integer literal also satisfies a `number` constraint.
    allowed.contains(&cat) || (cat == SchemaType::Integer && allowed.contains(&SchemaType::Number))
}

/// Coerce a raw XML value to one of the types a union schema allows. Tries
/// structured (object/array) parsing only when the union permits it, then
/// integer, number, and boolean, and finally falls back to a string. A value
/// that matches none of the allowed types degenerates to a string (documented
/// behavior) rather than being force-parsed into a disallowed JSON type.
fn coerce_union_value(value: &str, allowed: &HashSet<SchemaType>) -> ParsedValue {
    // `null` is already handled by the caller before schema lookup.
    if allowed.contains(&SchemaType::Object) || allowed.contains(&SchemaType::Array) {
        if let Ok(json_val) = serde_json::from_str::<Value>(value)
            && value_matches(&json_val, allowed)
        {
            return json_val.into();
        }
        if let Ok(json_val) = try_literal_eval(value)
            && value_matches(&json_val, allowed)
        {
            return json_val.into();
        }
    }

    if allowed.contains(&SchemaType::Integer)
        && is_integer_literal(value)
        && let Some(coerced) = coerce_integer_literal(value)
    {
        return coerced;
    }

    if allowed.contains(&SchemaType::Number) {
        if is_integer_literal(value)
            && let Some(coerced) = coerce_integer_literal(value)
        {
            return coerced;
        }
        if let Ok(f) = value.parse::<f64>() {
            if f.fract() == 0.0
                && f.is_finite()
                && let Some(i) = f.to_i64()
            {
                return Value::Number(i.into()).into();
            }
            if let Some(num) = serde_json::Number::from_f64(f) {
                return Value::Number(num).into();
            }
        }
    }

    if allowed.contains(&SchemaType::Boolean) {
        let lower = value.to_lowercase();
        if lower == "true" || lower == "false" {
            return Value::Bool(lower == "true").into();
        }
    }

    Value::String(value.to_string()).into()
}

/// Try to parse a value similar to Python's ast.literal_eval.
/// This is a simplified version that handles common cases.
fn try_literal_eval(s: &str) -> Result<Value, ()> {
    // First try standard JSON
    if let Ok(val) = serde_json::from_str::<Value>(s) {
        return Ok(val);
    }

    // Try to handle Python-style literals (single quotes, True/False/None).
    // Tokenize so the keyword rewrites only touch code OUTSIDE quoted strings —
    // a global `replace` would corrupt real data (`{'message': 'True story'}`
    // must NOT become `{"message": "true story"}`).
    let normalized = normalize_python_literal(s);

    serde_json::from_str::<Value>(&normalized).map_err(|_| ())
}

/// Convert a Python-style literal into a JSON-ish string: swap single-quote
/// string delimiters for double quotes and rewrite the bareword constants
/// `True`/`False`/`None` -> `true`/`false`/`null`, but ONLY outside quoted
/// strings. String contents (including any literal `True`/`None` words, embedded
/// quotes, and backslash escapes) are preserved verbatim so real argument data
/// isn't mangled.
fn normalize_python_literal(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    // The quote char that opened the current string, if we're inside one.
    let mut string_delim: Option<char> = None;
    // Accumulates a run of identifier chars so we can match whole keywords only.
    let mut ident = String::new();

    // Flush a pending bareword, rewriting the Python constants.
    fn flush_ident(ident: &mut String, out: &mut String) {
        match ident.as_str() {
            "True" => out.push_str("true"),
            "False" => out.push_str("false"),
            "None" => out.push_str("null"),
            other => out.push_str(other),
        }
        ident.clear();
    }

    while let Some(c) = chars.next() {
        if let Some(delim) = string_delim {
            // Inside a string: copy verbatim, tracking escapes and the close.
            if c == '\\' {
                out.push(c);
                if let Some(escaped) = chars.next() {
                    out.push(escaped);
                }
            } else if c == delim {
                // Close the string; JSON always uses double quotes.
                out.push('"');
                string_delim = None;
            } else if c == '"' {
                // A literal double quote inside a single-quoted string must be
                // escaped once we re-delimit with double quotes.
                out.push_str("\\\"");
            } else {
                out.push(c);
            }
            continue;
        }

        if c.is_alphanumeric() || c == '_' {
            ident.push(c);
            continue;
        }
        if !ident.is_empty() {
            flush_ident(&mut ident, &mut out);
        }

        if c == '\'' || c == '"' {
            // Open a string; emit a double quote as the JSON delimiter.
            out.push('"');
            string_delim = Some(c);
        } else {
            out.push(c);
        }
    }
    if !ident.is_empty() {
        flush_ident(&mut ident, &mut out);
    }
    out
}

/// Safely parse a value - tries JSON, then falls back to string.
/// Mimics SGLang's `_safe_val` function in spirit.
/// NOTE: This function is deprecated and kept for reference. Use convert_param_value instead.
#[allow(dead_code)]
fn safe_parse_value(raw: &str) -> serde_json::Value {
    // HTML unescape
    let unescaped = html_unescape(raw.trim());

    if let Ok(value) = serde_json::from_str::<serde_json::Value>(&unescaped) {
        return value;
    }

    if let Ok(num) = unescaped.parse::<i64>() {
        return serde_json::Value::Number(num.into());
    }

    if let Ok(num) = unescaped.parse::<f64>()
        && let Some(num_val) = serde_json::Number::from_f64(num)
    {
        return serde_json::Value::Number(num_val);
    }

    match unescaped.to_lowercase().as_str() {
        "true" => return serde_json::Value::Bool(true),
        "false" => return serde_json::Value::Bool(false),
        "null" | "none" => return serde_json::Value::Null,
        _ => {}
    }

    // Default to string, stripping newlines from start and end.
    serde_json::Value::String(unescaped.trim_matches('\n').to_string())
}

/// Simple HTML unescape for common entities.
fn html_unescape(s: &str) -> String {
    s.replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&amp;", "&")
        .replace("&quot;", "\"")
        .replace("&#x27;", "'")
        .replace("&#39;", "'")
}

#[cfg(test)]
mod strip_quotes_tests {
    use super::strip_quotes;

    #[test]
    fn single_quote_char_does_not_panic() {
        // Regression: a lone quote matched both ends and sliced [1..0] -> panic.
        assert_eq!(strip_quotes("'"), "'");
        assert_eq!(strip_quotes("\""), "\"");
    }

    #[test]
    fn strips_matched_pairs_only() {
        assert_eq!(strip_quotes("\"hi\""), "hi");
        assert_eq!(strip_quotes("'hi'"), "hi");
        assert_eq!(strip_quotes("\"\""), "");
        assert_eq!(strip_quotes("bare"), "bare");
        assert_eq!(strip_quotes("\"mismatch'"), "\"mismatch'");
    }
}

#[cfg(test)]
mod coderabbit_fix_tests {
    use super::*;
    use serde_json::json;

    fn one_param(name: &str, schema: Value) -> HashMap<String, Value> {
        let mut m = HashMap::new();
        m.insert(name.to_string(), schema);
        m
    }

    fn ser(pv: &ParsedValue) -> String {
        serde_json::to_string(pv).unwrap()
    }

    // Finding 1: integer-schema values outside i64 range stay JSON numbers.
    #[test]
    fn large_integer_schema_value_stays_a_number() {
        let cfg = one_param("x", json!({"type": "integer"}));
        // 21 digits — well past i64::MAX. Old code degenerated this to a string.
        let pv = convert_param_value("123456789012345678901", "x", &cfg, "f");
        assert_eq!(ser(&pv), "123456789012345678901");
        assert_ne!(ser(&pv), "\"123456789012345678901\"");

        // In-range integers and non-numeric fallback still behave.
        assert_eq!(ser(&convert_param_value("42", "x", &cfg, "f")), "42");
        assert_eq!(ser(&convert_param_value("abc", "x", &cfg, "f")), "\"abc\"");
    }

    // Finding 2: Python keyword rewrites must not touch quoted string contents.
    #[test]
    fn python_keywords_inside_strings_are_preserved() {
        let v = try_literal_eval("{'message': 'True story'}").unwrap();
        assert_eq!(v, json!({"message": "True story"}));

        // Bare constants OUTSIDE strings still convert.
        let v = try_literal_eval("[True, False, None]").unwrap();
        assert_eq!(v, json!([true, false, null]));

        // Full path through convert_param_value with an object schema.
        let cfg = one_param("x", json!({"type": "object"}));
        let pv = convert_param_value("{'message': 'True story'}", "x", &cfg, "f");
        assert_eq!(ser(&pv), r#"{"message":"True story"}"#);
        // The previously-wrong "true story" must NOT appear.
        assert!(!ser(&pv).contains("true story"));
    }

    // Finding 3: union schemas coerce only to an allowed alternative.
    #[test]
    fn union_schema_coerces_to_allowed_type_only() {
        // anyOf [string, null] + "42": stays a string (was the JSON number 42).
        let cfg = one_param(
            "x",
            json!({"anyOf": [{"type": "string"}, {"type": "null"}]}),
        );
        let pv = convert_param_value("42", "x", &cfg, "f");
        assert_eq!(ser(&pv), "\"42\"");
        assert_ne!(ser(&pv), "42");

        // anyOf [integer, null] + "42": becomes a number.
        let cfg = one_param(
            "x",
            json!({"anyOf": [{"type": "integer"}, {"type": "null"}]}),
        );
        assert_eq!(ser(&convert_param_value("42", "x", &cfg, "f")), "42");

        // oneOf [object, null] + object literal: still JSON-parsed.
        let cfg = one_param(
            "x",
            json!({"oneOf": [{"type": "object"}, {"type": "null"}]}),
        );
        assert_eq!(
            ser(&convert_param_value("{\"a\": 1}", "x", &cfg, "f")),
            r#"{"a":1}"#
        );

        // type: ["number", "null"] + "3.14": becomes a float.
        let cfg = one_param("x", json!({"type": ["number", "null"]}));
        assert_eq!(ser(&convert_param_value("3.14", "x", &cfg, "f")), "3.14");

        // No alternative matches "42" -> documented string fallback.
        let cfg = one_param(
            "x",
            json!({"anyOf": [{"type": "boolean"}, {"type": "null"}]}),
        );
        assert_eq!(ser(&convert_param_value("42", "x", &cfg, "f")), "\"42\"");
    }

    fn bare_config() -> XmlParserConfig {
        XmlParserConfig {
            backoff_when_no_wrapper: true,
            allow_eof_recovery: true,
            ..XmlParserConfig::default()
        }
    }

    fn str_tool(name: &str) -> ToolDefinition {
        ToolDefinition {
            name: name.to_string(),
            parameters: Some(json!({
                "type": "object",
                "properties": {"k": {"type": "string"}},
            })),
        }
    }

    // Finding 4, path A: back-off recovery keeps trailing narration.
    #[test]
    fn backoff_recovery_preserves_trailing_text() {
        let tools = vec![str_tool("get_weather")];
        // No outer <tool_call> wrapper -> is_bare_function_mode fires (path A).
        let message = "Before<function=get_weather><parameter=k>v</parameter></function>After";
        let (calls, content) =
            try_tool_call_parse_xml(message, &bare_config(), Some(&tools)).unwrap();
        assert_eq!(calls.len(), 1);
        // Old code returned only the prefix "Before" and dropped "After".
        assert_eq!(content.as_deref(), Some("BeforeAfter"));
    }

    // Finding 4, path B: gap recovery before a later wrapper keeps trailing text.
    #[test]
    fn gap_recovery_preserves_trailing_text() {
        let tools = vec![str_tool("a"), str_tool("b")];
        // A bare function in the gap, then a real <tool_call> block. The " mid "
        // narration after the recovered </function> must survive.
        let message = "Intro<function=a><parameter=k>v</parameter></function> mid \
            <tool_call><function=b><parameter=k>v</parameter></function></tool_call>";
        let (calls, content) =
            try_tool_call_parse_xml(message, &bare_config(), Some(&tools)).unwrap();
        assert_eq!(calls.len(), 2);
        let content = content.unwrap();
        assert!(content.contains("Intro"), "prefix kept: {content:?}");
        assert!(content.contains("mid"), "trailing kept: {content:?}");
    }
}
