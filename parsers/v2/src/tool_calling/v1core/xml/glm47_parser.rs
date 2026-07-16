// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// GLM-4.7 Tool Call Parser
// Format: <tool_call>function_name<arg_key>param1</arg_key><arg_value>value1</arg_value></tool_call>
// Reference: https://huggingface.co/zai-org/GLM-4.7/blob/main/chat_template.jinja

use regex::Regex;
use serde_json::Value;
use std::collections::HashMap;
use tracing::warn;
use uuid::Uuid;

use super::super::ToolDefinition;
use super::super::config::Glm47ParserConfig;
use super::parsed_value::{ParsedValue, coerce_integer_literal};
use super::response::{CalledFunction, ToolCallResponse, ToolCallType};

/// Render a tool_call block snippet for logs. Bounded so a huge truncated
/// argument body doesn't blow up the log line; control chars are escaped
/// because raw newlines/tabs make the warning unreadable in grep/jq.
fn truncate_for_log(s: &str) -> String {
    const MAX: usize = 200;
    let mut out = String::with_capacity(MAX.min(s.len()) + 16);
    let mut bytes = 0usize;
    for ch in s.chars() {
        if bytes >= MAX {
            out.push('…');
            break;
        }
        match ch {
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c => out.push(c),
        }
        bytes += ch.len_utf8();
    }
    out
}

/// Try to parse GLM-4.7 formatted tool calls from a message.
/// Format: <tool_call>function_name<arg_key>param1</arg_key><arg_value>value1</arg_value></tool_call>
/// Returns (parsed_tool_calls, normal_text_content)
pub fn try_tool_call_parse_glm47(
    message: &str,
    config: &Glm47ParserConfig,
    tools: Option<&[ToolDefinition]>,
) -> anyhow::Result<(Vec<ToolCallResponse>, Option<String>)> {
    let (normal_text, tool_calls) = extract_tool_calls(message, config, tools)?;

    let normal_content = if normal_text.is_empty() {
        Some("".to_string())
    } else {
        Some(normal_text)
    };

    Ok((tool_calls, normal_content))
}

/// Extract tool calls and normal text from message.
///
/// `normal_text` is the model text with each complete tool-call block removed
/// (from its `<tool_call>` start through its `</tool_call>` end), keeping ALL
/// other text verbatim: the prefix before the first call, text BETWEEN calls,
/// and text AFTER the last call. Whitespace is preserved as-is. Only tool-call
/// markup is stripped — natural text is never dropped and markup never leaks
/// into normal_text. Malformed / unrecoverable blocks (missing fences, truncated
/// tails) keep the drop-without-leak behavior.
fn extract_tool_calls(
    text: &str,
    config: &Glm47ParserConfig,
    tools: Option<&[ToolDefinition]>,
) -> anyhow::Result<(String, Vec<ToolCallResponse>)> {
    let mut normal_parts = Vec::new();
    let mut calls = Vec::new();
    let mut cursor = 0;

    let start_token = &config.tool_call_start;
    let end_token = &config.tool_call_end;

    if !text.contains(start_token.as_str())
        && let Some(marker_idx) = first_orphan_glm47_marker_index(text, config)
    {
        if let Some((prefix, mut parsed_calls)) =
            recover_bare_glm47_calls(text, marker_idx, config, tools)?
        {
            let recovered_calls = parsed_calls.len();
            warn!(
                why = "bare_body_recovery",
                recovered_calls,
                recovered_bytes = text.len() - prefix.len(),
                kept_prefix_bytes = prefix.len(),
                "GLM-4.7 parser recovered complete bare call body/bodies without <tool_call> start"
            );
            calls.append(&mut parsed_calls);
            return Ok((prefix, calls));
        }
        warn!(
            why = "GLM-4.7 tool-call marker found without <tool_call> start; dropping orphan marker tail so wire tags do not leak into normal_text",
            dropped_block = %truncate_for_log(&text[marker_idx..]),
            "GLM-4.7 parser dropping orphan tool-call marker tail"
        );
        return Ok((orphan_glm47_prefix(text, marker_idx), calls));
    }

    while cursor < text.len() {
        // Find next tool call start
        if let Some(start_pos) = text[cursor..].find(start_token.as_str()) {
            let abs_start = cursor + start_pos;
            let gap = &text[cursor..abs_start];
            // Preserve ALL surrounding natural-language text — the prefix before
            // the first <tool_call>, text BETWEEN calls, and (in the no-more-calls
            // arm below) text AFTER the last call. Only tool-call markup is
            // stripped: recover_bare_glm47_calls_in_span pulls a bare call out of
            // the gap and keeps its `prefix` prose; orphan_glm47_prefix drops a
            // stray wire marker but keeps the text before it. The gap text is kept
            // regardless of whether calls have already been parsed, so inter-call
            // narration is no longer dropped.
            if let Some((prefix, mut parsed_calls)) =
                recover_bare_glm47_calls_in_span(gap, config, tools)?
            {
                normal_parts.push(prefix);
                calls.append(&mut parsed_calls);
            } else if let Some(marker_idx) = first_orphan_glm47_marker_index(gap, config) {
                normal_parts.push(orphan_glm47_prefix(gap, marker_idx));
            } else {
                normal_parts.push(gap.to_string());
            }

            // Find the corresponding end token
            if let Some(end_pos) = text[abs_start..].find(end_token.as_str()) {
                let abs_end = abs_start + end_pos + end_token.len();
                let block = &text[abs_start..abs_end];

                // Parse this tool call block. Unparseable blocks (malformed
                // <tool_call>...</tool_call> markup the parser can't extract)
                // are dropped — emitting the raw markup as normal_text leaks
                // wire tags downstream. vLLM and SGLang both drop on this
                // path; aligning Dynamo to that contract.
                match parse_tool_call_block(block, config, tools) {
                    Ok(parsed_call) => calls.push(parsed_call),
                    Err(e) => {
                        warn!(
                            reason = %e,
                            why = "block has open + close fence but content failed to parse \
                                   as a GLM-4.7 tool call (e.g. empty function name, \
                                   missing <arg_key>, malformed args); dropping to avoid \
                                   leaking wire tags through normal_text",
                            dropped_block = %truncate_for_log(block),
                            "GLM-4.7 parser dropping unparseable tool_call block"
                        );
                    }
                }

                cursor = abs_end;
            } else {
                // Recovery: outer </tool_call> absent (max_tokens / EOS
                // truncation). Gated on `allow_eof_recovery` so streaming
                // early-exit doesn't fire mid-stream. Also requires an
                // `<arg_key>` opener in the trailing slice as the structural
                // signal that a real tool call was emitted.
                let block = &text[abs_start..];
                let arg_key_start = &config.arg_key_start;
                if config.allow_eof_recovery && block.contains(arg_key_start.as_str()) {
                    match parse_tool_call_block(block, config, tools) {
                        Ok(parsed_call) => {
                            calls.push(parsed_call);
                            cursor = text.len();
                            continue;
                        }
                        Err(e) => {
                            warn!(
                                reason = %e,
                                why = "EOF recovery enabled and <arg_key> opener present, \
                                       but parse_tool_call_block failed on the truncated \
                                       tail; dropping to avoid leaking wire tags through \
                                       normal_text",
                                dropped_block = %truncate_for_log(block),
                                "GLM-4.7 parser dropping truncated tool_call block (recovery attempt failed)"
                            );
                        }
                    }
                } else {
                    // Either recovery disabled (production default for GLM-4.7)
                    // or no <arg_key> in the tail (so this is plausibly not a
                    // real tool call at all, just a stray <tool_call> token).
                    let reason = if !config.allow_eof_recovery {
                        "allow_eof_recovery=false (production default for GLM-4.7 to match \
                         vLLM/SGLang on truncated tool calls)"
                    } else {
                        "no <arg_key> in the tail after the <tool_call> start fence, so the \
                         block does not look like a structurally-real GLM-4.7 tool call"
                    };
                    warn!(
                        why = %reason,
                        dropped_block = %truncate_for_log(block),
                        "GLM-4.7 parser dropping truncated tool_call block (no end fence)"
                    );
                }
                // Drop the truncated/unrecoverable tail. Emitting the raw
                // <tool_call>...<arg_key>...<arg_value>... prefix as
                // normal_text would leak wire tags into message.content; vLLM
                // strips the same way on truncation.
                break;
            }
        } else {
            // No more tool calls: this gap is the text AFTER the last </tool_call>
            // (or the whole message when no call was found). Preserve it verbatim,
            // stripping only tool-call markup (bare-call recovery keeps its prose
            // prefix; orphan_glm47_prefix drops a stray wire marker but keeps the
            // text before it).
            let gap = &text[cursor..];
            if let Some((prefix, mut parsed_calls)) =
                recover_bare_glm47_calls_in_span(gap, config, tools)?
            {
                normal_parts.push(prefix);
                calls.append(&mut parsed_calls);
            } else if let Some(marker_idx) = first_orphan_glm47_marker_index(gap, config) {
                normal_parts.push(orphan_glm47_prefix(gap, marker_idx));
            } else {
                normal_parts.push(gap.to_string());
            }
            break;
        }
    }

    let normal_text = normal_parts.join("");
    let normal_text = if calls.is_empty() {
        normal_text.trim().to_string()
    } else {
        normal_text
    };
    Ok((normal_text, calls))
}

fn recover_bare_glm47_calls_in_span(
    span: &str,
    config: &Glm47ParserConfig,
    tools: Option<&[ToolDefinition]>,
) -> anyhow::Result<Option<(String, Vec<ToolCallResponse>)>> {
    let Some(marker_idx) = first_orphan_glm47_marker_index(span, config) else {
        return Ok(None);
    };
    let recovered = recover_bare_glm47_calls(span, marker_idx, config, tools)?;
    if let Some((prefix, parsed_calls)) = recovered {
        let recovered_calls = parsed_calls.len();
        warn!(
            why = "bare_body_gap_recovery",
            recovered_calls,
            recovered_bytes = span.len() - prefix.len(),
            kept_prefix_bytes = prefix.len(),
            "GLM-4.7 parser recovered complete bare call body/bodies before a later <tool_call>"
        );
        return Ok(Some((prefix, parsed_calls)));
    }
    Ok(None)
}

fn first_orphan_glm47_marker_index(text: &str, config: &Glm47ParserConfig) -> Option<usize> {
    [
        config.tool_call_end.as_str(),
        config.arg_key_start.as_str(),
        config.arg_key_end.as_str(),
        config.arg_value_start.as_str(),
        config.arg_value_end.as_str(),
    ]
    .into_iter()
    .filter_map(|marker| text.find(marker))
    .min()
}

fn orphan_glm47_prefix(text: &str, marker_idx: usize) -> String {
    let prefix = text[..marker_idx].trim_end();
    let token_start = prefix
        .char_indices()
        .rev()
        .find(|(_, ch)| ch.is_whitespace())
        .map(|(idx, ch)| idx + ch.len_utf8())
        .unwrap_or(0);
    let tail = &prefix[token_start..];
    if !tail.is_empty()
        && tail
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '.'))
    {
        prefix[..token_start].trim().to_string()
    } else {
        prefix.trim().to_string()
    }
}

fn recover_bare_glm47_calls(
    text: &str,
    marker_idx: usize,
    config: &Glm47ParserConfig,
    tools: Option<&[ToolDefinition]>,
) -> anyhow::Result<Option<(String, Vec<ToolCallResponse>)>> {
    if !text[marker_idx..].contains(config.tool_call_end.as_str()) {
        return Ok(None);
    }

    let before_marker = text[..marker_idx].trim_end();
    let function_name_start = before_marker
        .char_indices()
        .rev()
        .find(|(_, ch)| ch.is_whitespace())
        .map(|(idx, ch)| idx + ch.len_utf8())
        .unwrap_or(0);

    let candidate_name = before_marker[function_name_start..].trim();
    if candidate_name.is_empty()
        || !candidate_name
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '.'))
    {
        return Ok(None);
    }

    let prefix = text[..function_name_start].to_string();
    let mut cursor = function_name_start;
    let mut calls = Vec::new();

    while cursor < text.len() {
        let rest = &text[cursor..];
        let trim_offset = rest.len() - rest.trim_start().len();
        cursor += trim_offset;

        let tail = &text[cursor..];
        let Some(end_pos) = tail.find(config.tool_call_end.as_str()) else {
            break;
        };
        let call_end = cursor + end_pos + config.tool_call_end.len();
        let wrapped = format!("{}{}", config.tool_call_start, &text[cursor..call_end]);
        calls.push(parse_tool_call_block(&wrapped, config, tools)?);
        cursor = call_end;

        if is_glm47_close_marker_spam(&text[cursor..], config) {
            warn!(
                why = "orphan_close_marker_spam",
                dropped_block = %truncate_for_log(&text[cursor..]),
                "GLM-4.7 parser dropping orphan close-marker spam after recovered bare call"
            );
            break;
        }

        if first_orphan_glm47_marker_index(&text[cursor..], config).is_none() {
            break;
        }
    }

    if calls.is_empty() {
        return Ok(None);
    }
    Ok(Some((prefix, calls)))
}

fn is_glm47_close_marker_spam(text: &str, config: &Glm47ParserConfig) -> bool {
    let mut rest = text.trim_start();
    let mut saw_close = false;
    while let Some(after_close) = rest.strip_prefix(config.tool_call_end.as_str()) {
        saw_close = true;
        rest = after_close.trim_start();
    }
    saw_close && rest.is_empty()
}

/// Decode XML character entities in a string.
/// Handles the five predefined XML entities: &lt; &gt; &amp; &quot; &apos;
fn decode_xml_entities(s: &str) -> String {
    s.replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&amp;", "&")
        .replace("&quot;", "\"")
        .replace("&apos;", "'")
}

/// Coerce a raw string value using the tool's parameter schema.
/// Falls back to string if no schema is available or the type is unrecognized.
fn coerce_value(raw: &str, schema_type: Option<&str>) -> ParsedValue {
    let trimmed = raw.trim();

    // If the value already looks like JSON (object, array, or quoted string), parse it directly
    if (trimmed.starts_with('{') || trimmed.starts_with('[') || trimmed.starts_with('"'))
        && let Ok(v) = serde_json::from_str::<Value>(trimmed)
    {
        return v.into();
    }

    // Use schema type hints for coercion when available
    match schema_type {
        Some("integer") | Some("int") => {
            if let Some(value) = coerce_integer_literal(trimmed) {
                return value;
            }
        }
        Some("number") | Some("float") | Some("double") => {
            if let Some(value) = coerce_integer_literal(trimmed) {
                return value;
            }
            if let Ok(n) = trimmed.parse::<f64>()
                && let Some(num) = serde_json::Number::from_f64(n)
            {
                return Value::Number(num).into();
            }
        }
        Some("boolean") | Some("bool") => match trimmed.to_lowercase().as_str() {
            "true" | "1" | "yes" => return Value::Bool(true).into(),
            "false" | "0" | "no" => return Value::Bool(false).into(),
            _ => {}
        },
        Some("array") => {
            // Try JSON parse first, then fall back to comma-separated splitting
            if let Ok(v) = serde_json::from_str::<Value>(trimmed)
                && v.is_array()
            {
                return v.into();
            }
            let items: Vec<Value> = trimmed
                .split(',')
                .map(|s| Value::String(s.trim().to_string()))
                .collect();
            return Value::Array(items).into();
        }
        Some("null") if trimmed == "null" || trimmed == "None" || trimmed.is_empty() => {
            return Value::Null.into();
        }
        _ => {}
    }

    Value::String(raw.to_string()).into()
}

/// Look up the JSON Schema type for a parameter by name from a tool's parameter schema.
fn get_param_schema_type<'a>(
    tools: Option<&'a [ToolDefinition]>,
    function_name: &str,
    param_name: &str,
) -> Option<&'a str> {
    let tool = tools?.iter().find(|t| t.name == function_name)?;
    let schema = tool.parameters.as_ref()?;
    let props = schema.get("properties")?;
    let param = props.get(param_name)?;
    param.get("type")?.as_str()
}

/// Parse a single GLM-4.7 tool call block
/// Format: <tool_call>function_name<arg_key>key1</arg_key><arg_value>value1</arg_value>...</tool_call>
fn parse_tool_call_block(
    block: &str,
    config: &Glm47ParserConfig,
    tools: Option<&[ToolDefinition]>,
) -> anyhow::Result<ToolCallResponse> {
    // Remove the outer <tool_call> tags
    let start_token = &config.tool_call_start;
    let end_token = &config.tool_call_end;

    // Strip the outer start token. The end token is optional so we can
    // recover from max_tokens / EOS truncation that drops `</tool_call>`.
    let after_start = block
        .strip_prefix(start_token.as_str())
        .ok_or_else(|| anyhow::anyhow!("Invalid tool call block format"))?;
    let content = after_start
        .strip_suffix(end_token.as_str())
        .unwrap_or(after_start);

    // Extract function name (everything before first <arg_key> or end)
    let arg_key_start = &config.arg_key_start;
    let function_name = if let Some(pos) = content.find(arg_key_start.as_str()) {
        content[..pos].trim().to_string()
    } else {
        // No arguments, just function name
        content.trim().to_string()
    };

    if function_name.is_empty() {
        anyhow::bail!("Empty function name in tool call");
    }

    // Parse key-value pairs
    let mut arguments = HashMap::new();
    let args_section = &content[function_name.len()..];

    // Build regex patterns
    let arg_key_start_escaped = regex::escape(&config.arg_key_start);
    let arg_key_end_escaped = regex::escape(&config.arg_key_end);
    let arg_value_start_escaped = regex::escape(&config.arg_value_start);
    let arg_value_end_escaped = regex::escape(&config.arg_value_end);

    // Pattern: <arg_key>key</arg_key><arg_value>value(</arg_value> | end-of-block)
    // The `</arg_value>` close is treated as OPTIONAL. A final value whose close
    // tag was dropped (mismatched/missing fences — batch case 4.d
    // `<arg_value>NYC</tool_call>`, whose trailing `</tool_call>` is stripped
    // before this regex, leaving the value at end-of-block) is recovered by
    // terminating at `\z` instead of being dropped to empty args. `</arg_value>`
    // is listed first so a well-formed value (including a multi-line one) still
    // terminates exactly there; `\z` only applies when the close tag is absent.
    // (The `regex` crate has no lookahead, so the terminator is a plain
    // alternation of the close tag and the end-of-text anchor.)
    // (?s) enables dotall mode so (.*?) matches across newlines — required
    // because models often emit multi-line content in arg values.
    let pattern = format!(
        r"(?s){}([^<]+){}{}(.*?)(?:{}|\z)",
        arg_key_start_escaped, arg_key_end_escaped, arg_value_start_escaped, arg_value_end_escaped
    );

    let regex = Regex::new(&pattern)?;

    for cap in regex.captures_iter(args_section) {
        let key = cap.get(1).map(|m| m.as_str().trim()).unwrap_or("");
        let raw_value = cap.get(2).map(|m| m.as_str()).unwrap_or("");

        if !key.is_empty() {
            // Decode XML entities (e.g. &lt; → <, &amp; → &) before parsing
            let decoded = decode_xml_entities(raw_value);

            // Look up the expected type from the tool's parameter schema
            let schema_type = get_param_schema_type(tools, &function_name, key);
            let json_value = coerce_value(&decoded, schema_type);

            arguments.insert(key.to_string(), json_value);
        }
    }

    // A tool call for a function not in the request's tools list is still
    // emitted, not dropped. The parser's job is to extract what the model
    // produced; validation/rejection belongs to the serving layer. The model
    // may legitimately reference a tool the caller adds later, or one baked into
    // its chat template, and dropping it silently hides a real call. This
    // matches vLLM (which passes the call through) and the other Dynamo parsers
    // (kimi_k2 / xml / gemma4 already warn-and-pass-through here). We log for
    // visibility only.
    if let Some(tools_list) = tools
        && !tools_list.iter().any(|t| t.name == function_name)
    {
        warn!(
            function = %function_name,
            why = "tool_not_in_request_tool_list",
            "GLM-4.7 tool call references a function not in the request's tools list; \
             passing it through (serving layer validates)"
        );
    }

    Ok(ToolCallResponse {
        id: Uuid::new_v4().to_string(),
        tp: ToolCallType::Function,
        function: CalledFunction {
            name: function_name,
            arguments: serde_json::to_string(&arguments)?,
        },
    })
}
