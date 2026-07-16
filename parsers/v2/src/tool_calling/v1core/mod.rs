// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Vendored batch tool-call extraction, owned by `parsers/v2`.
//!
//! The v2 streaming parsers own the streaming concern (buffering, chunk-split
//! marker safety, normal_text suppression) but delegate per-call VALUE TYPING to
//! a batch parser. Historically that batch parser lived in the `dynamo_parsers`
//! (v1) crate and v2 linked it. That coupling is deliberately gone: v1 and v2 are
//! independent and v1 is slated for deletion, so v2 vendors its own copy of the
//! batch extraction here and never references `dynamo_parsers`.
//!
//! These files are copied from v1's `tool_calling` extraction core, trimmed to
//! what v2 actually calls (test modules and unused `detect_`/`find_` helpers
//! dropped). When v1 is removed, this module is the sole owner. See the project
//! rule "v1-v2-independent-no-shared-code".

use serde_json::Value;

pub mod config;
pub mod gemma4;
pub mod response;
pub mod xml;

/// A tool the model may call, as the batch extractors expect it.
#[derive(Debug, Clone)]
pub struct ToolDefinition {
    pub name: String,
    pub parameters: Option<Value>,
}

impl From<&crate::tool_calling::traits::Tool> for ToolDefinition {
    fn from(t: &crate::tool_calling::traits::Tool) -> Self {
        Self {
            name: t.name.clone(),
            parameters: Some(t.parameters.clone()),
        }
    }
}

pub use config::{Glm47ParserConfig, KimiK2ParserConfig, MiniMaxM3ParserConfig, XmlParserConfig};
pub use response::{CalledFunctionStream, ToolCallResponseChunk, ToolCallType};
pub use xml::{
    try_tool_call_parse_glm47, try_tool_call_parse_kimi_k2, try_tool_call_parse_minimax_m3,
    try_tool_call_parse_xml,
};
