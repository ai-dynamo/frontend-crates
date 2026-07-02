// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::{BasicReasoningParser, ParserResult, ReasoningParser};

/// MiniMax Append-Think Reasoning Parser.
///
/// MiniMax starts generating reasoning immediately without emitting a
/// `<think>` opener, then emits `</think>` before normal content or tool calls.
/// Treat the missing opener as an implicit force-reasoning state and consume
/// both markers so callers receive the same reasoning/content split as other
/// force-reasoning model families.
///
/// References:
/// - SGLang MiniMaxAppendThinkDetector:
///   <https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/parser/reasoning_parser.py>
/// - vLLM MiniMaxM2AppendThinkReasoningParser:
///   <https://github.com/vllm-project/vllm/blob/main/vllm/reasoning/minimax_m2_reasoning_parser.py>
#[derive(Debug)]
pub struct MiniMaxAppendThinkParser {
    inner: BasicReasoningParser,
}

impl MiniMaxAppendThinkParser {
    pub fn new() -> Self {
        Self {
            inner: BasicReasoningParser::new(
                "<think>".to_string(),
                "</think>".to_string(),
                true,
                true,
            ),
        }
    }
}

impl Default for MiniMaxAppendThinkParser {
    fn default() -> Self {
        Self::new()
    }
}

impl ReasoningParser for MiniMaxAppendThinkParser {
    fn detect_and_parse_reasoning(&mut self, text: &str, _token_ids: &[u32]) -> ParserResult {
        self.inner.detect_and_parse_reasoning(text, _token_ids)
    }

    fn parse_reasoning_streaming_incremental(
        &mut self,
        text: &str,
        _token_ids: &[u32],
    ) -> ParserResult {
        self.inner
            .parse_reasoning_streaming_incremental(text, _token_ids)
    }

    fn finish_reasoning_stream(&mut self) -> ParserResult {
        self.inner.finish_reasoning_stream()
    }

    fn set_in_reasoning(&mut self, in_reasoning: bool) {
        self.inner.set_in_reasoning(in_reasoning)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_and_parse_truncated_reasoning() {
        let mut parser = MiniMaxAppendThinkParser::new();
        let result = parser.detect_and_parse_reasoning("reasoning content here", &[]);
        assert_eq!(result.normal_text, "");
        assert_eq!(result.reasoning_text, "reasoning content here");
    }

    #[test]
    fn test_detect_and_parse_splits_reasoning_and_content() {
        let mut parser = MiniMaxAppendThinkParser::new();
        let result =
            parser.detect_and_parse_reasoning("reasoning content</think>normal response", &[]);
        assert_eq!(result.normal_text, "normal response");
        assert_eq!(result.reasoning_text, "reasoning content");
    }

    #[test]
    fn test_streaming_splits_implicit_reasoning() {
        let mut parser = MiniMaxAppendThinkParser::new();

        let r1 = parser.parse_reasoning_streaming_incremental("I need to ", &[]);
        assert_eq!(r1.normal_text, "");
        assert_eq!(r1.reasoning_text, "I need to ");

        let r2 = parser.parse_reasoning_streaming_incremental("check the weather", &[]);
        assert_eq!(r2.normal_text, "");
        assert_eq!(r2.reasoning_text, "check the weather");

        let r3 = parser.parse_reasoning_streaming_incremental("</think>The weather is sunny.", &[]);
        assert_eq!(r3.normal_text, "The weather is sunny.");
        assert_eq!(r3.reasoning_text, "");
    }

    #[test]
    fn test_streaming_bare_json_without_boundary_is_reasoning() {
        let mut parser = MiniMaxAppendThinkParser::new();
        let r = parser.parse_reasoning_streaming_incremental(
            r#"[{"name":"get_weather","parameters":{"location":"San Francisco"}}]"#,
            &[],
        );
        assert_eq!(r.normal_text, "");
        assert_eq!(
            r.reasoning_text,
            r#"[{"name":"get_weather","parameters":{"location":"San Francisco"}}]"#
        );
    }

    #[test]
    fn test_streaming_tool_call_after_reasoning_is_normal_text() {
        let mut parser = MiniMaxAppendThinkParser::new();

        let r1 = parser.parse_reasoning_streaming_incremental("let me call a tool", &[]);
        assert_eq!(r1.normal_text, "");
        assert_eq!(r1.reasoning_text, "let me call a tool");

        let r2 = parser.parse_reasoning_streaming_incremental(
            "</think><minimax:tool_call><invoke name=\"get_weather\">",
            &[],
        );
        assert_eq!(
            r2.normal_text,
            "<minimax:tool_call><invoke name=\"get_weather\">"
        );
        assert_eq!(r2.reasoning_text, "");
    }
}
