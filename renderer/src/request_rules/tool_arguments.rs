// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tool-call argument normalization for templates that read arguments as a map.

/// Parse a JSON object string into a serde_json Map, preserving the exact
/// spelling of numbers that cannot be round-tripped through f64 without loss.
/// Numbers that survive f64 conversion unchanged are kept as `Value::Number`;
/// numbers that would be corrupted (e.g. integers with >15 significant digits)
/// are kept as `Value::String` so the original value is available to templates.
/// Returns `None` when the top-level JSON value is not an object.
fn parse_args_object(
    s: &str,
) -> anyhow::Result<Option<serde_json::Map<String, serde_json::Value>>> {
    // First pass: parse the top-level value to check it is an object.
    let top: serde_json::Value = serde_json::from_str(s)?;
    let top_obj = match top {
        serde_json::Value::Object(m) => m,
        _ => return Ok(None),
    };

    // Second pass: re-parse each entry's value via its RawValue to detect
    // numbers that serde_json would corrupt through f64.
    // We use a raw string-to-rawvalue helper to get the exact bytes.
    let mut out = serde_json::Map::with_capacity(top_obj.len());
    for (k, cooked) in top_obj {
        // For non-numbers, use the already-parsed value directly.
        if !cooked.is_number() {
            out.insert(k, cooked);
            continue;
        }
        // For numbers: find the raw token in the source string so we can
        // verify that serde_json's f64 representation matches the original.
        // We do this by re-serializing the cooked number and comparing.
        let cooked_str = cooked.to_string();
        if let Some(raw_token) = extract_value_token(s, &k) {
            let raw_trimmed = raw_token.trim();
            if raw_trimmed != cooked_str.as_str() {
                // Precision was lost (e.g. 30-digit integer → 1.23e+29).
                // Keep the original spelling as a string.
                out.insert(k, serde_json::Value::String(raw_trimmed.to_string()));
                continue;
            }
        }
        out.insert(k, cooked);
    }
    Ok(Some(out))
}

/// Extract the raw JSON token for `key` from a JSON object string `s`.
/// Returns `None` when the key is not found or parsing fails.
fn extract_value_token<'a>(s: &'a str, key: &str) -> Option<&'a str> {
    // Build the escaped key as it appears in JSON.
    let key_json = serde_json::to_string(key).ok()?;
    // Find `"key":` in the source.
    let needle = format!("{}:", key_json);
    let start = s.find(needle.as_str())?;
    let after_colon = s[start + needle.len()..].trim_start();
    // Measure the extent of the JSON token.
    let end = token_end(after_colon)?;
    let source_offset = s.len() - s[start + needle.len()..].len()
        + (after_colon.as_ptr() as usize - s[start + needle.len()..].as_ptr() as usize);
    Some(&s[source_offset..source_offset + end])
}

/// Return the byte length of the first JSON token in `s`.
fn token_end(s: &str) -> Option<usize> {
    let s = s.trim_start();
    let first = s.chars().next()?;
    match first {
        '"' => {
            let mut i = 1;
            let b = s.as_bytes();
            while i < b.len() {
                if b[i] == b'\\' {
                    i += 2;
                } else if b[i] == b'"' {
                    return Some(i + 1);
                } else {
                    i += 1;
                }
            }
            None
        }
        '{' | '[' => {
            // Skip balanced braces/brackets — good enough for flat objects.
            let (open, close) = if first == '{' {
                (b'{', b'}')
            } else {
                (b'[', b']')
            };
            let mut depth = 0i32;
            let mut in_str = false;
            let mut escape = false;
            for (i, &b) in s.as_bytes().iter().enumerate() {
                if escape {
                    escape = false;
                    continue;
                }
                if in_str {
                    if b == b'\\' {
                        escape = true;
                    } else if b == b'"' {
                        in_str = false;
                    }
                } else {
                    match b {
                        b'"' => in_str = true,
                        b if b == open => depth += 1,
                        b if b == close => {
                            depth -= 1;
                            if depth == 0 {
                                return Some(i + 1);
                            }
                        }
                        _ => {}
                    }
                }
            }
            None
        }
        _ => {
            // Number, bool, null — ends at first delimiter.
            let end = s
                .find(|c: char| c == ',' || c == '}' || c == ']' || c.is_whitespace())
                .unwrap_or(s.len());
            if end == 0 { None } else { Some(end) }
        }
    }
}

/// Parse `tool_calls[*].function.arguments` from JSON string to object before
/// handing messages to MiniJinja. Some chat templates (e.g. GLM-5.2) call
/// `.items()` on arguments and require a dict; the OpenAI wire format carries
/// arguments as a JSON string. This matches TRT-LLM's chat_utils.py behaviour.
///
/// A large integer keeps its exact spelling. With serde_json's
/// `arbitrary_precision` feature it stays a number; without it,
/// `parse_args_object` keeps it as a string rather than round it through f64.
///
/// Returns an error if any tool_call's arguments string contains invalid JSON,
/// so callers can surface a 400-style error instead of fabricating empty args.
pub(crate) fn normalize_tool_call_arguments(
    messages_json: &mut serde_json::Value,
) -> anyhow::Result<()> {
    let Some(messages) = messages_json.as_array_mut() else {
        return Ok(());
    };
    for message in messages {
        let Some(tool_calls) = message
            .get_mut("tool_calls")
            .and_then(serde_json::Value::as_array_mut)
        else {
            continue;
        };
        for tc in tool_calls.iter_mut() {
            let Some(args_str) = tc.pointer("/function/arguments").and_then(|v| v.as_str()) else {
                continue;
            };
            // Empty string is a valid no-argument call; normalise to {}.
            if args_str.is_empty() {
                if let Some(obj) = tc
                    .get_mut("function")
                    .and_then(serde_json::Value::as_object_mut)
                {
                    obj.insert(
                        "arguments".to_string(),
                        serde_json::Value::Object(serde_json::Map::new()),
                    );
                }
                continue;
            }
            let value = match parse_args_object(args_str) {
                Ok(Some(map)) => serde_json::Value::Object(map),
                Ok(None) => {
                    // Scalar or array — GLM's .items() would panic at render time.
                    tracing::warn!(
                        args_len = args_str.len(),
                        "tool_call arguments parsed to a non-object; \
                         substituting {{}} for template safety"
                    );
                    serde_json::Value::Object(serde_json::Map::new())
                }
                Err(e) => {
                    anyhow::bail!(
                        "tool_call arguments are not valid JSON (len={}): {e}",
                        args_str.len()
                    );
                }
            };
            if let Some(obj) = tc
                .get_mut("function")
                .and_then(serde_json::Value::as_object_mut)
            {
                obj.insert("arguments".to_string(), value);
            }
        }
    }
    Ok(())
}

/// True when the deployment's template reads tool-call arguments as a map, so
/// historical `function.arguments` strings must be parsed before rendering.
///
/// `arguments_format_is_json_object` is the deployment's own setting for this
/// (Dynamo's `tool_call_arguments_format: json_object`). The `glm47` tool-call
/// parser always needs it: GLM-5.2 and later templates call `.items()` on the
/// arguments.
pub fn should_normalize_tool_call_arguments(
    tool_call_parser: Option<&str>,
    arguments_format_is_json_object: bool,
) -> bool {
    arguments_format_is_json_object || tool_call_parser == Some("glm47")
}
