// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Ordered reasoning, text, and tool-call parsing.
//!
//! Unlike [`crate::ToolParseResult`], this contract preserves source ordering
//! when one input chunk contains multiple assistant channels.

mod kimi_k3;
mod qwen3_coder;
mod scan;

pub use kimi_k3::{KIMI_K3_FAMILY, KimiK3StructuralTagBuilder, KimiK3UnifiedParser};
pub use qwen3_coder::{QWEN3_CODER_FAMILY, QWEN3_REASONING_FAMILY, Qwen3CoderUnifiedParser};

use serde_json::Value;

use crate::{Tool, ToolCallDelta};

/// One event emitted by a unified assistant-output parser.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UnifiedParserEvent {
    /// Normal assistant-visible text.
    Text(String),
    /// Hidden reasoning text.
    Reasoning(String),
    /// One tool-call update.
    ToolCall(ToolCallDelta),
}

/// Ordered output committed while advancing a unified parser.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct UnifiedParserOutput {
    pub events: Vec<UnifiedParserEvent>,
}

impl UnifiedParserOutput {
    pub fn push_text(&mut self, delta: impl AsRef<str> + Into<String>) {
        if delta.as_ref().is_empty() {
            return;
        }
        if let Some(UnifiedParserEvent::Text(text)) = self.events.last_mut() {
            text.push_str(delta.as_ref());
        } else {
            self.events.push(UnifiedParserEvent::Text(delta.into()));
        }
    }

    pub fn push_reasoning(&mut self, delta: impl AsRef<str> + Into<String>) {
        if delta.as_ref().is_empty() {
            return;
        }
        if let Some(UnifiedParserEvent::Reasoning(reasoning)) = self.events.last_mut() {
            reasoning.push_str(delta.as_ref());
        } else {
            self.events
                .push(UnifiedParserEvent::Reasoning(delta.into()));
        }
    }

    pub fn push_call(&mut self, call: ToolCallDelta) {
        self.events.push(UnifiedParserEvent::ToolCall(call));
    }

    pub fn append(&mut self, other: Self) {
        for event in other.events {
            match event {
                UnifiedParserEvent::Text(text) => self.push_text(text),
                UnifiedParserEvent::Reasoning(reasoning) => {
                    self.push_reasoning(reasoning);
                }
                UnifiedParserEvent::ToolCall(call) => self.push_call(call),
            }
        }
    }
}

/// Assistant channel already opened by the rendered generation prompt.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum UnifiedParserPrefill {
    #[default]
    None,
    Reasoning,
    Response,
}

/// Tool-choice policy used to build model-native structural guidance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnifiedToolChoice<'a> {
    None,
    Auto,
    Required,
    Named(&'a str),
}

/// Tool-call wire format the backend will emit for this request.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum UnifiedToolOutputMode<'a> {
    /// Model-native tool-call markup.
    #[default]
    Native,
    /// Guided bare JSON. Named choices emit only the selected tool's arguments;
    /// required choices emit objects containing a tool name and arguments.
    GuidedJson { named_tool: Option<&'a str> },
}

/// Request-scoped structural-guidance inputs.
#[derive(Debug, Clone, Copy)]
pub struct UnifiedToolCallFormatContext<'a> {
    pub tool_choice: UnifiedToolChoice<'a>,
    pub tools: &'a [Tool],
    pub parallel_tool_calls: Option<bool>,
    pub strict_schema: bool,
    pub starts_in_reasoning: bool,
}

/// Model-family structural-guidance builder paired with a unified parser.
pub trait UnifiedStructuralTagBuilder: Send + Sync {
    fn build_tool_call_format(
        &self,
        ctx: &UnifiedToolCallFormatContext<'_>,
    ) -> anyhow::Result<Option<Value>>;

    fn build_tool_call_ban(&self) -> anyhow::Result<Option<Value>> {
        Ok(None)
    }
}

/// Incremental parser that emits ordered assistant events.
pub trait UnifiedParser: Send {
    fn create(tools: &[Tool]) -> anyhow::Result<Box<dyn UnifiedParser>>
    where
        Self: Sized + 'static;

    fn initialize(&mut self, _prefill: UnifiedParserPrefill) -> anyhow::Result<()> {
        Ok(())
    }

    /// Initialize request-scoped parser state, including the backend's actual
    /// tool-call wire format.
    fn initialize_with_output_mode(
        &mut self,
        prefill: UnifiedParserPrefill,
        _tool_output_mode: UnifiedToolOutputMode<'_>,
    ) -> anyhow::Result<()> {
        self.initialize(prefill)
    }

    fn preserve_special_tokens(&self) -> bool {
        false
    }

    fn structural_tag_builder(&self) -> Option<&dyn UnifiedStructuralTagBuilder> {
        None
    }

    fn tool_call_id(&self, _tool_index: usize) -> Option<&str> {
        None
    }

    fn parse_into(&mut self, delta: &str, output: &mut UnifiedParserOutput) -> anyhow::Result<()>;

    fn push(&mut self, delta: &str) -> anyhow::Result<UnifiedParserOutput> {
        let mut output = UnifiedParserOutput::default();
        self.parse_into(delta, &mut output)?;
        Ok(output)
    }

    fn finish(&mut self) -> anyhow::Result<UnifiedParserOutput> {
        Ok(UnifiedParserOutput::default())
    }

    fn reset(&mut self) -> String {
        String::new()
    }

    fn parse_complete(&mut self, output: &str) -> anyhow::Result<UnifiedParserOutput> {
        let mut parsed = self.push(output)?;
        parsed.append(self.finish()?);
        Ok(parsed)
    }
}

/// Construct one registered unified parser by family name.
pub fn create_unified_parser_for_family(
    family: &str,
    tools: &[Tool],
) -> anyhow::Result<Box<dyn UnifiedParser>> {
    match family {
        KIMI_K3_FAMILY => KimiK3UnifiedParser::create(tools),
        QWEN3_CODER_FAMILY => Qwen3CoderUnifiedParser::create(tools),
        other => anyhow::bail!("no Dynamo unified parser for family '{other}'"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_coalesces_only_adjacent_channels() {
        let mut output = UnifiedParserOutput::default();
        output.push_text("hello");
        output.push_text(" world");
        output.push_reasoning("think");
        output.push_reasoning("ing");
        output.push_call(ToolCallDelta {
            tool_index: 0,
            name: Some("lookup".to_string()),
            arguments: "{}".to_string(),
        });
        output.push_text("!");

        assert_eq!(
            output.events,
            vec![
                UnifiedParserEvent::Text("hello world".to_string()),
                UnifiedParserEvent::Reasoning("thinking".to_string()),
                UnifiedParserEvent::ToolCall(ToolCallDelta {
                    tool_index: 0,
                    name: Some("lookup".to_string()),
                    arguments: "{}".to_string(),
                }),
                UnifiedParserEvent::Text("!".to_string()),
            ]
        );
    }
}
