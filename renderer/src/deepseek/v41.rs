// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Text prompt encoding for DeepSeek V4.1.

use anyhow::{Context, Result, ensure};
use serde_json::Value;

use super::common::{ThinkingMode, resolve_thinking_mode, to_json};
use super::v4::{Encoding, encode_messages_with_encoding};

pub(super) fn find_last_user_index(messages: &[Value]) -> Option<usize> {
    messages.iter().enumerate().rposition(|(index, message)| {
        let role = message.get("role").and_then(Value::as_str);
        role == Some("user") || (role == Some("system") && index > 0)
    })
}

pub(super) fn drop_thinking_messages(mut messages: Vec<Value>) -> Vec<Value> {
    if let Some(last_user) = find_last_user_index(&messages) {
        for message in &mut messages[..last_user] {
            if message.get("role").and_then(Value::as_str) == Some("assistant") {
                message.as_object_mut().unwrap().remove("reasoning_content");
            }
        }
    }
    messages
}

pub(super) fn encode_arguments(tool_call: &Value) -> Result<String> {
    let original = tool_call
        .get("arguments")
        .context("Missing tool arguments")?;
    let mut arguments = original.clone();
    for _ in 0..2 {
        let Some(text) = arguments.as_str() else {
            break;
        };
        let Ok(decoded) = serde_json::from_str(text) else {
            break;
        };
        arguments = decoded;
    }
    if !arguments.is_object() {
        arguments = serde_json::json!({"arguments": original});
    }
    let parameters = arguments
        .as_object()
        .unwrap()
        .iter()
        .map(|(name, value)| {
            let content = value
                .as_str()
                .map(str::to_owned)
                .unwrap_or_else(|| to_json(value));
            format!(
                "<｜DSML｜ parameter name=\"{name}\" string=\"{}\">{content}</｜DSML｜ parameter>",
                value.is_string()
            )
        })
        .collect::<Vec<_>>();
    Ok(parameters.join("\n"))
}

fn normalize_text(messages: &mut [Value]) -> Result<()> {
    for message in messages {
        for field in ["tools", "tool_calls"] {
            for tool in message
                .get(field)
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
            {
                ensure!(
                    tool.get("namespace").is_none_or(Value::is_null)
                        && tool
                            .get("function")
                            .and_then(|function| function.get("namespace"))
                            .is_none_or(Value::is_null),
                    "DeepSeek V4.1 native formatter does not support explicit tool namespaces; use a qualified function name"
                );
            }
        }
        ensure!(
            message.get("content_blocks").is_none(),
            "DeepSeek V4.1 expects OpenAI text content blocks in content"
        );
        ensure!(
            !message
                .get("reasoning_content")
                .and_then(Value::as_str)
                .is_some_and(|text| text.contains("<｜deepseek_image｜>")),
            "DeepSeek V4.1 native formatter supports text content only"
        );
        if message.get("role").and_then(Value::as_str) == Some("developer") {
            message["role"] = Value::String("system".into());
        }
        if let Some(content) = message.get("content") {
            let text = match content {
                Value::Null => String::new(),
                Value::String(text) => text.clone(),
                Value::Array(blocks) => {
                    let mut texts = Vec::with_capacity(blocks.len());
                    for block in blocks {
                        ensure!(
                            block.get("type").and_then(Value::as_str) == Some("text"),
                            "DeepSeek V4.1 native formatter supports text content only"
                        );
                        texts.push(
                            block
                                .get("text")
                                .and_then(Value::as_str)
                                .context("Text block requires text")?,
                        );
                    }
                    texts.join("\n\n")
                }
                _ => anyhow::bail!("DeepSeek V4.1 message content must be text"),
            };
            ensure!(
                !text.contains("<｜deepseek_image｜>"),
                "DeepSeek V4.1 native formatter supports text content only"
            );
            message["content"] = Value::String(text);
        }
    }
    Ok(())
}

/// Encode text messages with the model's numeric reasoning effort (1–100).
pub fn encode_messages(
    messages: &[Value],
    thinking_mode: ThinkingMode,
    drop_thinking: bool,
    reasoning_effort: u8,
) -> Result<String> {
    ensure!(
        (1..=100).contains(&reasoning_effort),
        "DeepSeek V4.1 reasoning effort must be within 1–100"
    );
    let mut messages = messages.to_vec();
    normalize_text(&mut messages)?;
    encode_messages_with_encoding(
        &messages,
        thinking_mode,
        true,
        drop_thinking,
        Encoding::V41(reasoning_effort),
    )
}

/// Native text formatter with OpenAI reasoning-effort names mapped as in the
/// DeepSeek V4.1 reference encoder.
#[derive(Debug, Default)]
pub struct DeepSeekV41Formatter;

impl crate::OAIPromptFormatter for DeepSeekV41Formatter {
    fn supports_add_generation_prompt(&self) -> bool {
        false
    }

    fn render(&self, req: &dyn crate::OAIChatLikeRequest) -> Result<String> {
        let args = req.chat_template_args();
        let effort = req
            .reasoning_effort()
            .map(|value| serde_json::to_value(value).context("Serialize reasoning effort"))
            .transpose()?
            .or_else(|| args.and_then(|args| args.get("reasoning_effort").cloned()));
        let mut thinking_mode = resolve_thinking_mode(args, ThinkingMode::Thinking);
        let budget = match effort.as_ref() {
            None | Some(Value::Null) => 75,
            Some(value) => match value.as_str() {
                Some("none") => {
                    thinking_mode = ThinkingMode::Chat;
                    75
                }
                Some("low") => 50,
                Some("high") => 75,
                Some("max") => 100,
                _ => value
                    .as_u64()
                    .filter(|v| (1..=100).contains(v))
                    .context("DeepSeek V4.1 reasoning effort must be low, high, max, none, or an integer within 1–100")?
                    as u8,
            },
        };
        let drop_thinking = match args.and_then(|args| args.get("drop_thinking")) {
            None => true,
            Some(value) => value.as_bool().context("drop_thinking must be a boolean")?,
        };
        let mut messages: Vec<Value> =
            serde_json::from_value(serde_json::to_value(req.messages())?)?;
        let tools_enabled =
            req.tool_choice().as_ref().and_then(|value| value.as_str()) != Some("none");
        let tools = req
            .tools()
            .filter(|tools| tools_enabled && tools.len().is_some_and(|length| length > 0))
            .map(serde_json::to_value)
            .transpose()?;
        let response_format = req
            .response_format()
            .map(serde_json::to_value)
            .transpose()?;
        if tools.is_some() || response_format.is_some() {
            if !matches!(
                messages
                    .first()
                    .and_then(|m| m.get("role"))
                    .and_then(Value::as_str),
                Some("system" | "developer")
            ) {
                messages.insert(0, serde_json::json!({"role": "system", "content": ""}));
            }
            if let Some(tools) = tools {
                messages[0]["tools"] = tools;
            }
            if let Some(response_format) = response_format {
                messages[0]["response_format"] = response_format;
            }
        }
        encode_messages(&messages, thinking_mode, drop_thinking, budget)
    }
}
