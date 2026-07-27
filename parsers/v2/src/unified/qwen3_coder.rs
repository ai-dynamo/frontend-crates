// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unified Qwen3 reasoning and Qwen3-Coder tool-call parser.
//!
//! Qwen3 reasoning is delimited by `<think>` and `</think>`. Visible spans are
//! forwarded to the existing Qwen3-Coder XML tool parser for model-native
//! output, or buffered as bare JSON when guided decoding forces a required or
//! named tool. One request-scoped state machine owns the complete reasoning,
//! text, and tool-call lifecycle.

#[cfg(test)]
mod tests;

use super::scan::partial_prefix_len;
use super::{UnifiedParser, UnifiedParserOutput, UnifiedParserPrefill, UnifiedToolOutputMode};
use crate::tool_calling::scan::WrappedBlockSink;
use crate::{Qwen3CoderToolStreamParser, Tool, ToolParser};
use serde_json::value::RawValue;

pub const QWEN3_CODER_FAMILY: &str = "qwen3_coder";
pub const QWEN3_REASONING_FAMILY: &str = "qwen3";

const THINK_OPEN: &str = "<think>";
const THINK_CLOSE: &str = "</think>";
const COMPACT_MIN_CONSUMED: usize = 4096;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
enum Qwen3Mode {
    #[default]
    OutsideReasoning,
    Reasoning,
    VisibleOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Boundary {
    Open,
    Close,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
enum ToolOutputMode {
    #[default]
    NativeXml,
    GuidedJson {
        named_tool: Option<String>,
    },
}

#[derive(Debug, serde::Deserialize)]
struct GuidedToolCall {
    name: String,
    #[serde(default)]
    parameters: Option<Box<RawValue>>,
    #[serde(default)]
    arguments: Option<Box<RawValue>>,
}

impl WrappedBlockSink for UnifiedParserOutput {
    fn push_text(&mut self, text: &str) {
        UnifiedParserOutput::push_text(self, text);
    }

    fn push_call(&mut self, call: crate::ToolCallDelta) {
        UnifiedParserOutput::push_call(self, call);
    }
}

/// Ordered parser for the Qwen3 reasoning and Qwen3-Coder tool-call pair.
pub struct Qwen3CoderUnifiedParser {
    buffer: String,
    cursor: usize,
    mode: Qwen3Mode,
    tool_parser: Qwen3CoderToolStreamParser,
    tool_output_mode: ToolOutputMode,
    guided_json: String,
    finished: bool,
}

impl Qwen3CoderUnifiedParser {
    pub fn new(tools: &[Tool]) -> Self {
        Self {
            buffer: String::new(),
            cursor: 0,
            mode: Qwen3Mode::OutsideReasoning,
            tool_parser: Qwen3CoderToolStreamParser::new(tools),
            tool_output_mode: ToolOutputMode::NativeXml,
            guided_json: String::new(),
            finished: false,
        }
    }

    fn pending(&self) -> &str {
        &self.buffer[self.cursor..]
    }

    fn take_pending(&mut self) -> String {
        let pending = self.pending().to_string();
        self.buffer.clear();
        self.cursor = 0;
        pending
    }

    fn advance(&mut self, consumed: usize) {
        self.cursor += consumed;
        if self.cursor >= COMPACT_MIN_CONSUMED && self.cursor >= self.buffer.len() / 2 {
            self.buffer.drain(..self.cursor);
            self.cursor = 0;
        }
    }

    fn emit_reasoning(&mut self, len: usize, output: &mut UnifiedParserOutput) {
        let text = self.pending()[..len].to_string();
        output.push_reasoning(text);
        self.advance(len);
    }

    fn emit_visible(&mut self, len: usize, output: &mut UnifiedParserOutput) -> anyhow::Result<()> {
        let start = self.cursor;
        let end = start + len;
        match &self.tool_output_mode {
            ToolOutputMode::NativeXml => {
                self.tool_parser
                    .push_into(&self.buffer[start..end], output)?;
            }
            ToolOutputMode::GuidedJson { .. } => {
                self.guided_json.push_str(&self.buffer[start..end]);
            }
        }
        self.advance(len);
        Ok(())
    }

    fn finish_guided_json(&mut self, output: &mut UnifiedParserOutput) -> anyhow::Result<()> {
        let payload = self.guided_json.trim();
        if payload.is_empty() {
            return Ok(());
        }

        let calls = match &self.tool_output_mode {
            ToolOutputMode::GuidedJson {
                named_tool: Some(name),
            } => serde_json::from_str::<Box<RawValue>>(payload)
                .ok()
                .map(|arguments| vec![(name.clone(), arguments.get().to_string())])
                .unwrap_or_default(),
            ToolOutputMode::GuidedJson { named_tool: None } => {
                parse_required_guided_calls(payload)?
            }
            ToolOutputMode::NativeXml => unreachable!("guided JSON finish in native XML mode"),
        };

        if calls.is_empty() {
            output.push_text(std::mem::take(&mut self.guided_json));
            return Ok(());
        }

        for (tool_index, (name, arguments)) in calls.into_iter().enumerate() {
            output.push_call(crate::ToolCallDelta {
                tool_index,
                name: Some(name),
                arguments,
            });
        }
        self.guided_json.clear();
        Ok(())
    }

    fn parse_buffer(&mut self, output: &mut UnifiedParserOutput) -> anyhow::Result<()> {
        loop {
            if self.pending().is_empty() {
                return Ok(());
            }

            match self.mode {
                Qwen3Mode::VisibleOnly => {
                    let len = self.pending().len();
                    self.emit_visible(len, output)?;
                }
                Qwen3Mode::OutsideReasoning => {
                    if let Some((offset, boundary)) = next_boundary(self.pending()) {
                        if offset > 0 {
                            self.emit_visible(offset, output)?;
                            continue;
                        }
                        match boundary {
                            Boundary::Open => {
                                self.advance(THINK_OPEN.len());
                                self.mode = Qwen3Mode::Reasoning;
                            }
                            Boundary::Close => {
                                // A close marker outside reasoning is parser-owned
                                // syntax. Drop it instead of leaking it as content.
                                self.advance(THINK_CLOSE.len());
                            }
                        }
                        continue;
                    }

                    let safe = safe_len(self.pending());
                    if safe == 0 {
                        return Ok(());
                    }
                    self.emit_visible(safe, output)?;
                }
                Qwen3Mode::Reasoning => {
                    if let Some((offset, boundary)) = next_boundary(self.pending()) {
                        if offset > 0 {
                            self.emit_reasoning(offset, output);
                            continue;
                        }
                        match boundary {
                            Boundary::Open => {
                                // Prompt-prefilled reasoning may still be followed
                                // by a redundant model-emitted opener.
                                self.advance(THINK_OPEN.len());
                            }
                            Boundary::Close => {
                                self.advance(THINK_CLOSE.len());
                                self.mode = Qwen3Mode::OutsideReasoning;
                            }
                        }
                        continue;
                    }

                    let safe = safe_len(self.pending());
                    if safe == 0 {
                        return Ok(());
                    }
                    self.emit_reasoning(safe, output);
                }
            }
        }
    }
}

impl UnifiedParser for Qwen3CoderUnifiedParser {
    fn create(tools: &[Tool]) -> anyhow::Result<Box<dyn UnifiedParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self::new(tools)))
    }

    fn initialize(&mut self, prefill: UnifiedParserPrefill) -> anyhow::Result<()> {
        self.buffer.clear();
        self.cursor = 0;
        self.guided_json.clear();
        self.mode = match prefill {
            UnifiedParserPrefill::None => Qwen3Mode::OutsideReasoning,
            UnifiedParserPrefill::Reasoning => Qwen3Mode::Reasoning,
            UnifiedParserPrefill::Response => Qwen3Mode::VisibleOnly,
        };
        self.finished = false;
        Ok(())
    }

    fn initialize_with_output_mode(
        &mut self,
        prefill: UnifiedParserPrefill,
        tool_output_mode: UnifiedToolOutputMode<'_>,
    ) -> anyhow::Result<()> {
        self.tool_output_mode = match tool_output_mode {
            UnifiedToolOutputMode::Native => ToolOutputMode::NativeXml,
            UnifiedToolOutputMode::GuidedJson { named_tool } => ToolOutputMode::GuidedJson {
                named_tool: named_tool.map(str::to_string),
            },
        };
        self.initialize(prefill)
    }

    fn preserve_special_tokens(&self) -> bool {
        self.tool_parser.preserve_special_tokens()
    }

    fn parse_into(&mut self, delta: &str, output: &mut UnifiedParserOutput) -> anyhow::Result<()> {
        if self.finished {
            anyhow::bail!("cannot push after Qwen unified parser finish");
        }
        self.buffer.push_str(delta);
        self.parse_buffer(output)
    }

    fn finish(&mut self) -> anyhow::Result<UnifiedParserOutput> {
        if self.finished {
            anyhow::bail!("Qwen unified parser finish called more than once");
        }
        self.finished = true;

        let mut output = UnifiedParserOutput::default();
        let pending = self.take_pending();
        match self.mode {
            Qwen3Mode::Reasoning => output.push_reasoning(pending),
            Qwen3Mode::OutsideReasoning | Qwen3Mode::VisibleOnly => match &self.tool_output_mode {
                ToolOutputMode::NativeXml => {
                    self.tool_parser.push_into(&pending, &mut output)?;
                }
                ToolOutputMode::GuidedJson { .. } => self.guided_json.push_str(&pending),
            },
        }
        match &self.tool_output_mode {
            ToolOutputMode::NativeXml => self.tool_parser.finish_into(&mut output)?,
            ToolOutputMode::GuidedJson { .. } => self.finish_guided_json(&mut output)?,
        }
        Ok(output)
    }

    fn reset(&mut self) -> String {
        self.mode = Qwen3Mode::OutsideReasoning;
        self.finished = false;
        let mut recovered = std::mem::take(&mut self.guided_json);
        recovered.push_str(&self.take_pending());
        recovered
    }
}

fn parse_required_guided_calls(payload: &str) -> anyhow::Result<Vec<(String, String)>> {
    fn convert(call: GuidedToolCall) -> Option<(String, String)> {
        let arguments = call.parameters.or(call.arguments)?;
        Some((call.name, arguments.get().to_string()))
    }

    if let Ok(raw_calls) = serde_json::from_str::<Vec<Box<RawValue>>>(payload) {
        let calls = raw_calls
            .into_iter()
            .filter_map(|raw| {
                serde_json::from_str::<GuidedToolCall>(raw.get())
                    .ok()
                    .and_then(convert)
            })
            .collect::<Vec<_>>();
        return Ok(calls);
    }

    if let Ok(call) = serde_json::from_str::<GuidedToolCall>(payload) {
        return Ok(convert(call).into_iter().collect());
    }

    Ok(Vec::new())
}

fn next_boundary(text: &str) -> Option<(usize, Boundary)> {
    match (text.find(THINK_OPEN), text.find(THINK_CLOSE)) {
        (Some(open), Some(close)) if open <= close => Some((open, Boundary::Open)),
        (Some(_), Some(close)) => Some((close, Boundary::Close)),
        (Some(open), None) => Some((open, Boundary::Open)),
        (None, Some(close)) => Some((close, Boundary::Close)),
        (None, None) => None,
    }
}

fn safe_len(text: &str) -> usize {
    let keep = partial_prefix_len(text, THINK_OPEN).max(partial_prefix_len(text, THINK_CLOSE));
    text.len() - keep
}
