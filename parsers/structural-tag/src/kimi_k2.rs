// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Kimi K2 native structural-tag generation.
//!
//! K2 emits tool calls inside a special-token section rather than as the raw
//! JSON array used by the legacy forced-tool path. The numeric suffix belongs
//! to the model-generated call ID and must remain dynamic for parallel calls.
//!
//! This builder supports K2-Instruct and K2.5/K2.6 reasoning prompts. The
//! K2.5/K2.6 chat template injects `<think>` into the generation prompt, and
//! the model later emits `</think>`. It does not support original K2-Thinking,
//! whose chat template leaves the opening `<think>` for the model to generate.

use super::format::{
    ConstStringFormat, Format, JsonSchemaFormat, JsonSchemaStyle, RegexFormat, SequenceFormat,
    StructuralTag, TagFormat, TagsWithSeparatorFormat, TriggeredTagsFormat,
};
use crate::Tool;

pub const TOOL_CALL_BEGIN_PREFIX: &str = "<|tool_call_begin|>functions.";
pub const TOOL_CALL_ARGUMENT_BEGIN: &str = "<|tool_call_argument_begin|>";
pub const TOOL_CALL_END: &str = "<|tool_call_end|>";
pub const TOOL_CALLS_SECTION_BEGIN: &str = "<|tool_calls_section_begin|>";
pub const TOOL_CALLS_SECTION_END: &str = "<|tool_calls_section_end|>";

pub fn call_tag(tool: &Tool<'_>) -> TagFormat {
    TagFormat {
        begin: format!("{TOOL_CALL_BEGIN_PREFIX}{}:", tool.name),
        content: Box::new(Format::Sequence(SequenceFormat {
            elements: vec![
                Format::Regex(RegexFormat {
                    pattern: r"\d+".to_string(),
                }),
                Format::ConstString(ConstStringFormat {
                    value: TOOL_CALL_ARGUMENT_BEGIN.to_string(),
                }),
                Format::JsonSchema(JsonSchemaFormat {
                    json_schema: tool.parameters.clone(),
                    style: JsonSchemaStyle::Json,
                }),
            ],
        })),
        end: TOOL_CALL_END.to_string(),
    }
}

/// Wrap native calls in the K2 section envelope.
pub fn section(content: Format) -> TagFormat {
    TagFormat {
        begin: TOOL_CALLS_SECTION_BEGIN.to_string(),
        content: Box::new(content),
        end: TOOL_CALLS_SECTION_END.to_string(),
    }
}

/// Build the legacy format shape after request policy has been resolved.
pub fn build(tools: &[Tool<'_>], auto: bool, named: bool, stop_after_first: bool) -> StructuralTag {
    let calls: Vec<_> = tools.iter().map(call_tag).collect();
    let calls_format = if named {
        Format::Tag(calls.into_iter().next().expect("selected named tool"))
    } else {
        Format::TagsWithSeparator(TagsWithSeparatorFormat {
            tags: calls,
            separator: String::new(),
            at_least_one: true,
            stop_after_first,
        })
    };
    let format = if auto {
        Format::TriggeredTags(TriggeredTagsFormat {
            triggers: vec![TOOL_CALLS_SECTION_BEGIN.to_string()],
            tags: vec![section(calls_format)],
            at_least_one: false,
            stop_after_first,
        })
    } else {
        Format::Sequence(SequenceFormat {
            elements: vec![
                Format::ConstString(ConstStringFormat {
                    value: TOOL_CALLS_SECTION_BEGIN.to_string(),
                }),
                calls_format,
                Format::ConstString(ConstStringFormat {
                    value: TOOL_CALLS_SECTION_END.to_string(),
                }),
            ],
        })
    };
    StructuralTag { format }
}
