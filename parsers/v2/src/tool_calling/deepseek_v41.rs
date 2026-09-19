// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tool-only projection of the existing DeepSeek V4.1 unified parser.

use crate::tool_calling::traits::{Tool, ToolParseResult, ToolParser};
use crate::unified::{
    UnifiedParser, UnifiedParserExt, UnifiedParserInit, UnifiedParserStartingState, deepseek_v41,
};

/// Extracts tools while leaving reasoning markers intact for the caller.
pub struct DeepSeekV41ToolStreamParser {
    parser: Box<dyn UnifiedParser>,
}

impl ToolParser for DeepSeekV41ToolStreamParser {
    fn create(tools: &[Tool]) -> anyhow::Result<Box<dyn ToolParser>> {
        let mut parser = deepseek_v41::deepseek_v41_unified(tools);
        // Response mode disables reasoning extraction, preserving its delimiters as text.
        parser.initialize_request(UnifiedParserInit {
            starting_state: UnifiedParserStartingState::Response,
            ..Default::default()
        })?;
        Ok(Box::new(Self { parser }))
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
