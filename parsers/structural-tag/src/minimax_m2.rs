// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! MiniMax M2 native XML tool-call grammar.

use crate::Tool;
use crate::format::{
    Format, JsonSchemaFormat, JsonSchemaStyle, TagFormat, TagsWithSeparatorFormat,
};

pub const BLOCK_BEGIN: &str = "<minimax:tool_call>";
pub const BLOCK_END: &str = "</minimax:tool_call>";
pub const INVOKE_PREFIX: &str = "<invoke name=";
pub const INVOKE_BEGIN: &str = "<invoke name=\"";
pub const INVOKE_END: &str = "</invoke>";

/// Build a native call block from selected tools and resolved argument schemas.
pub fn block(tools: &[Tool<'_>], stop_after_first: bool) -> TagFormat {
    TagFormat {
        begin: format!("{BLOCK_BEGIN}\n"),
        content: Box::new(Format::TagsWithSeparator(TagsWithSeparatorFormat {
            tags: tools
                .iter()
                .map(|tool| TagFormat {
                    begin: format!("{INVOKE_BEGIN}{}\">\n", tool.name),
                    content: Box::new(Format::JsonSchema(JsonSchemaFormat {
                        json_schema: tool.parameters.clone(),
                        style: JsonSchemaStyle::MinimaxXml,
                    })),
                    end: format!("\n{INVOKE_END}"),
                })
                .collect(),
            separator: "\n".into(),
            at_least_one: true,
            stop_after_first,
        })),
        end: format!("\n{BLOCK_END}"),
    }
}
