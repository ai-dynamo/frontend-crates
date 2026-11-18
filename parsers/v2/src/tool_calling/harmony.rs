// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tool-only Harmony API projected from the GPT-OSS Unified parser.

use crate::tool_calling::traits::{Result, Tool, ToolParseResult, ToolParser};
use crate::tool_calling::{CalledFunctionStream, ToolCallResponseChunk, ToolCallType};
use crate::unified::{UnifiedParser, UnifiedParserEvent, UnifiedParserExt};
use std::collections::BTreeMap;
use std::panic::{RefUnwindSafe, UnwindSafe};

pub use crate::unified::harmony::{decode_harmony, encode_harmony};

/// Assemble streamed deltas into `(name, arguments-json-string)` per call index.
pub fn assemble_tool_calls(chunks: &[ToolCallResponseChunk]) -> Vec<(String, String)> {
    let mut names = BTreeMap::<u32, String>::new();
    let mut arguments = BTreeMap::<u32, String>::new();
    for chunk in chunks {
        if let Some(function) = &chunk.function {
            if let Some(name) = &function.name {
                names.entry(chunk.index).or_default().push_str(name);
            }
            if let Some(arguments_delta) = &function.arguments {
                arguments
                    .entry(chunk.index)
                    .or_default()
                    .push_str(arguments_delta);
            }
        }
    }
    names
        .into_iter()
        .map(|(index, name)| (name, arguments.get(&index).cloned().unwrap_or_default()))
        .collect()
}

/// Per-chunk result retained for callers of the legacy public API.
#[derive(Default, Debug)]
pub struct ToolStreamResult {
    pub normal_text: String,
    pub tool_call_chunks: Vec<ToolCallResponseChunk>,
}

/// Compatibility projection for the old Harmony tool-only parser contract.
pub struct HarmonyToolStreamParser {
    parser: Box<dyn UnifiedParser + Sync + UnwindSafe + RefUnwindSafe>,
    pending_normal_text: String,
    next_id: u64,
}

impl HarmonyToolStreamParser {
    pub fn new() -> anyhow::Result<Self> {
        let mut parser = crate::unified::harmony::gpt_oss_legacy_tool_adapter();
        parser.initialize(&[])?;
        Ok(Self {
            parser,
            pending_normal_text: String::new(),
            next_id: 0,
        })
    }

    /// Clear parser state and return text buffered by the Unified parser. This
    /// preserves the legacy tool-only parser's reset and reuse contract.
    pub fn reset(&mut self) -> String {
        let mut recovered = std::mem::take(&mut self.pending_normal_text);
        recovered.push_str(&self.parser.reset());
        self.next_id = 0;
        recovered
    }

    fn push_text(&mut self, chunk: &str) -> Result<Vec<UnifiedParserEvent>> {
        self.parser.push(chunk)
    }

    fn push_token_ids(&mut self, ids: &[u32]) -> Result<Vec<UnifiedParserEvent>> {
        self.parser.push_tokens(ids)
    }

    fn finish_events(&mut self) -> Result<Vec<UnifiedParserEvent>> {
        let mut events = Vec::new();
        events.extend(self.parser.finish()?.events);
        Ok(events)
    }

    fn project(&mut self, events: Vec<UnifiedParserEvent>, flush_text: bool) -> ToolParseResult {
        let mut result = ToolParseResult::default();
        for event in events {
            match event {
                UnifiedParserEvent::Text(text) => self.pending_normal_text.push_str(&text),
                // Legacy tool-only projection has no message-phase surface.
                UnifiedParserEvent::Reasoning(_)
                | UnifiedParserEvent::MessageStart(_)
                | UnifiedParserEvent::MessageEnd => {}
                UnifiedParserEvent::ToolCall(call) => result.calls.push(call),
            }
        }
        if flush_text {
            result.normal_text = std::mem::take(&mut self.pending_normal_text);
        }
        result
    }

    fn stream_result(&mut self, result: ToolParseResult) -> ToolStreamResult {
        let mut tool_call_chunks = Vec::new();
        for call in result.calls {
            let id = format!("call_{:08}", self.next_id);
            self.next_id += 1;
            tool_call_chunks.push(ToolCallResponseChunk {
                index: call.tool_index as u32,
                id: Some(id),
                tp: Some(ToolCallType::Function),
                function: Some(CalledFunctionStream {
                    name: call.name,
                    arguments: None,
                }),
            });
            tool_call_chunks.push(ToolCallResponseChunk {
                index: call.tool_index as u32,
                id: None,
                tp: None,
                function: Some(CalledFunctionStream {
                    name: None,
                    arguments: Some(call.arguments),
                }),
            });
        }
        ToolStreamResult {
            normal_text: result.normal_text,
            tool_call_chunks,
        }
    }

    pub fn parse_tool_call_streaming_text(&mut self, delta_text: &str) -> ToolStreamResult {
        match self.push_text(delta_text) {
            Ok(events) => {
                let result = self.project(events, false);
                self.stream_result(result)
            }
            Err(error) => {
                tracing::warn!(why = "harmony_text_projection_failed", error = %error, "legacy Harmony text projection failed");
                ToolStreamResult::default()
            }
        }
    }

    pub fn parse_tool_call_streaming_incremental(&mut self, token_ids: &[u32]) -> ToolStreamResult {
        match self.push_token_ids(token_ids) {
            Ok(events) => {
                let result = self.project(events, false);
                self.stream_result(result)
            }
            Err(error) => {
                tracing::warn!(why = "harmony_token_projection_failed", error = %error, "legacy Harmony token projection failed");
                ToolStreamResult::default()
            }
        }
    }

    pub fn finish_tool_call_stream(&mut self) -> ToolStreamResult {
        match self.finish_events() {
            Ok(events) => {
                let result = self.project(events, true);
                self.stream_result(result)
            }
            Err(error) => {
                tracing::warn!(why = "harmony_finish_projection_failed", error = %error, "legacy Harmony finish projection failed");
                ToolStreamResult::default()
            }
        }
    }
}

impl ToolParser for HarmonyToolStreamParser {
    fn create(_tools: &[Tool]) -> Result<Box<dyn ToolParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self::new()?))
    }

    fn preserve_special_tokens(&self) -> bool {
        true
    }

    fn prefers_tokens(&self) -> bool {
        true
    }

    fn push(&mut self, chunk: &str) -> Result<ToolParseResult> {
        let events = self.push_text(chunk)?;
        Ok(self.project(events, false))
    }

    fn push_tokens(&mut self, ids: &[u32]) -> Result<ToolParseResult> {
        let events = self.push_token_ids(ids)?;
        Ok(self.project(events, false))
    }

    fn finish(&mut self) -> Result<ToolParseResult> {
        let events = self.finish_events()?;
        Ok(self.project(events, true))
    }

    fn tool_call_id(&self, _tool_index: usize) -> Option<&str> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legacy_parser_preserves_auto_traits() {
        fn assert_traits<T: Send + Sync + UnwindSafe + RefUnwindSafe>() {}
        assert_traits::<HarmonyToolStreamParser>();
    }

    #[test]
    fn legacy_adapter_keeps_missing_recipient_body_out_of_visible_text() {
        let input = concat!(
            "<|start|>assistant<|channel|>commentary <|constrain|>json<|message|>",
            r#"{"location":"NYC"}<|call|>"#
        );
        let mut parser = HarmonyToolStreamParser::new().unwrap();
        assert!(parser.push(input).unwrap().calls.is_empty());
        let result = parser.finish().unwrap();
        assert!(result.calls.is_empty());
        assert!(result.normal_text.is_empty());
        assert!(parser.reset().is_empty());
        let _ = parser
            .push("<|start|>assistant<|channel|>final<|message|>answer<|return|>")
            .unwrap();
        assert_eq!(parser.finish().unwrap().normal_text, "answer");
    }

    #[test]
    fn legacy_parser_preserves_arguments_beyond_sorting_depth_limit() {
        for (open, close) in [("[", "]"), ("{\"item\":", "}"), ("[{\"item\":", "}]")] {
            let arguments = format!("{}0{}", open.repeat(3000), close.repeat(3000));
            let input = format!(
                "<|start|>assistant to=functions.lookup<|channel|>commentary<|message|>{arguments}<|call|>"
            );
            let mut parser = HarmonyToolStreamParser::new().unwrap();
            let result = parser.parse_complete(&input).unwrap();
            assert_eq!(result.calls.len(), 1);
            assert_eq!(result.calls[0].name.as_deref(), Some("lookup"));
            assert_eq!(result.calls[0].arguments, arguments);
        }
    }

    #[test]
    fn legacy_parser_reset_allows_reuse() {
        let input = concat!(
            "<|start|>assistant to=functions.lookup<|channel|>commentary<|message|>",
            r#"{"q":"first"}<|call|>"#
        );
        let mut parser = HarmonyToolStreamParser::new().unwrap();
        let _ = parser.parse_tool_call_streaming_text(input);
        let _ = parser.finish_tool_call_stream();
        assert!(parser.reset().is_empty());

        let next = concat!(
            "<|start|>assistant to=functions.lookup<|channel|>commentary<|message|>",
            r#"{"q":"second"}<|call|>"#
        );
        let mut chunks = parser.parse_tool_call_streaming_text(next).tool_call_chunks;
        chunks.extend(parser.finish_tool_call_stream().tool_call_chunks);
        assert_eq!(
            assemble_tool_calls(&chunks),
            vec![("lookup".to_string(), r#"{"q":"second"}"#.to_string())]
        );
    }
}
