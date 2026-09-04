// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Legacy tool-only projection of the Kimi K3 unified event stream.

use crate::tool_calling::traits::{Tool, ToolParseResult, ToolParser};
use crate::unified::{UnifiedParser, UnifiedParserExt, kimi_k3};

/// Kimi K3 tool parser backed by the native unified parser.
pub struct KimiK3ToolStreamParser {
    parser: Box<dyn UnifiedParser>,
}

impl KimiK3ToolStreamParser {
    pub fn new(tools: &[Tool]) -> Self {
        Self {
            parser: kimi_k3::kimi_k3_unified(tools),
        }
    }
}

impl ToolParser for KimiK3ToolStreamParser {
    fn create(tools: &[Tool]) -> anyhow::Result<Box<dyn ToolParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self::new(tools)))
    }

    fn preserve_special_tokens(&self) -> bool {
        self.parser.preserve_special_tokens()
    }

    fn push(&mut self, chunk: &str) -> anyhow::Result<ToolParseResult> {
        Ok(ToolParseResult::from_deltas(self.parser.push(chunk)?))
    }

    fn finish(&mut self) -> anyhow::Result<ToolParseResult> {
        Ok(ToolParseResult::from_deltas(self.parser.finish()?.events))
    }

    fn tool_call_id(&self, tool_index: usize) -> Option<&str> {
        self.parser.tool_call_id(tool_index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::unified::{UnifiedParserExt, assemble};

    #[test]
    fn legacy_surface_is_a_projection_of_the_same_events() {
        let input = concat!(
            "<|open|>think<|sep|>plan<|close|>think<|sep|>",
            "<|open|>response<|sep|>checking<|close|>response<|sep|>",
            "<|open|>tools<|sep|>",
            "<|open|>call tool=\"weather\" index=\"1\"<|sep|>",
            "<|open|>argument key=\"city\" type=\"string\"<|sep|>Paris",
            "<|close|>argument<|sep|><|close|>call<|sep|>",
            "<|close|>tools<|sep|>"
        );
        let mut unified = kimi_k3::kimi_k3_unified(&[]);
        let unified_events = unified.push(input).unwrap();
        let mut legacy = KimiK3ToolStreamParser::new(&[]);
        let projected = legacy.push(input).unwrap();

        assert_eq!(
            projected,
            ToolParseResult::from_deltas(unified_events.clone())
        );
        assert_eq!(projected.normal_text, "planchecking");
        assert_eq!(projected.calls.len(), 1);
        assert_eq!(projected.calls[0].name.as_deref(), Some("weather"));
        assert!(projected.calls[0].complete);
        assert_eq!(assemble(&unified_events).len(), 3);
    }

    #[test]
    fn legacy_parser_reuses_unified_lifecycle() {
        let mut parser = KimiK3ToolStreamParser::new(&[]);
        assert_eq!(parser.push("plain").unwrap().normal_text, "plain");
        parser.finish().unwrap();
        assert!(parser.push("later").is_err());
        assert!(parser.finish().is_err());
    }

    #[test]
    fn legacy_projection_preserves_model_call_id() {
        let input = concat!(
            "<|open|>tools<|sep|>",
            "<|open|>call tool=\"weather\" index=\"2\"<|sep|>",
            "<|open|>argument key=\"city\" type=\"string\"<|sep|>Paris",
            "<|close|>argument<|sep|><|close|>call<|sep|>",
            "<|close|>tools<|sep|>"
        );
        let mut parser = KimiK3ToolStreamParser::new(&[]);
        parser.push(input).unwrap();
        assert_eq!(parser.tool_call_id(0), Some("weather:1"));
    }
}
