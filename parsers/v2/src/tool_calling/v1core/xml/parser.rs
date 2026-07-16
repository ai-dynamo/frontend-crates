// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Reference implementation:
// https://github.com/sgl-project/sglang/blob/44da737770e4bcd9bfa27751f0a0751c9b5c06e1/python/sglang/srt/function_call/qwen3_coder_detector.py

use std::collections::HashMap;

use num_traits::ToPrimitive;
use regex::Regex;
use serde_json::Value;
use uuid::Uuid;

use super::super::ToolDefinition;
use super::super::config::XmlParserConfig;
use super::parsed_value::{ParsedValue, is_integer_literal, raw_number_literal};
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
            // Preserve narration before the first `<function=...>` tag so
            // streaming output isn't dropped on the back-off path.
            let prefix = message
                .split_once(config.function_start_token.as_str())
                .map(|(p, _)| p.to_string())
                .unwrap_or_default();
            return Ok((calls, Some(prefix)));
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
    Ok(Some((span[..marker_idx].to_string(), calls)))
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
    // If a parameter uses "anyOf"/"oneOf" instead of a direct "type", there is no
    // top-level "type" key. Treat it as "object" so the value goes through JSON
    // parsing rather than being returned as a double-encoded string.
    let param_schema = param_config.get(param_name);
    let param_type = param_schema
        .and_then(|v| v.get("type"))
        .and_then(|t| t.as_str())
        .map(|t| t.to_lowercase())
        .unwrap_or_else(|| {
            if param_schema
                .map(|v| v.get("anyOf").is_some() || v.get("oneOf").is_some())
                .unwrap_or(false)
            {
                "object".to_string()
            } else {
                "string".to_string()
            }
        });

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
            match param_value.parse::<i64>() {
                Ok(int_val) => Value::Number(int_val.into()).into(),
                Err(_) => {
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

/// Try to parse a value similar to Python's ast.literal_eval.
/// This is a simplified version that handles common cases.
fn try_literal_eval(s: &str) -> Result<Value, ()> {
    // First try standard JSON
    if let Ok(val) = serde_json::from_str::<Value>(s) {
        return Ok(val);
    }

    // Try to handle Python-style literals (single quotes, True/False/None)
    let normalized = s
        .replace('\'', "\"") // Replace single quotes with double quotes
        .replace("True", "true")
        .replace("False", "false")
        .replace("None", "null");

    serde_json::from_str::<Value>(&normalized).map_err(|_| ())
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
