// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Muse Spark ATEM tool-call parser.
//!
//! Muse Spark uses Muse Glimmer's channel and ATEM grammar, with one additional
//! reserved recipient: `commentary` is user-visible intermediate text rather than a
//! tool channel. The scanner remains shared so framing, recovery, argument typing and
//! stream-boundary behavior cannot drift between the two families.

use crate::tool_calling::muse_glimmer::{MuseChannelScanner, muse_spark_scanner};
use crate::tool_calling::traits::{Tool, ToolParseResult, ToolParser};

/// Stream parser for Muse Spark ATEM tool calls.
pub struct MuseSparkToolStreamParser {
    scanner: MuseChannelScanner,
}

impl MuseSparkToolStreamParser {
    pub fn new(tools: &[Tool]) -> Self {
        Self {
            scanner: muse_spark_scanner(tools),
        }
    }
}

impl ToolParser for MuseSparkToolStreamParser {
    fn create(tools: &[Tool]) -> anyhow::Result<Box<dyn ToolParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self::new(tools)))
    }

    fn preserve_special_tokens(&self) -> bool {
        true
    }

    fn push(&mut self, chunk: &str) -> anyhow::Result<ToolParseResult> {
        self.scanner.push(chunk)
    }

    fn finish(&mut self) -> anyhow::Result<ToolParseResult> {
        self.scanner.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn commentary_is_visible_normal_text_not_a_tool_channel() {
        let mut parser = MuseSparkToolStreamParser::new(&[]);
        let mut out = parser
            .push("<|start|>assistant to=commentary<|message|>Checking.<|eom|>")
            .expect("push");
        out.append(parser.finish().expect("finish"));
        assert_eq!(out.normal_text, "Checking.");
        assert!(out.calls.is_empty());
    }
}
