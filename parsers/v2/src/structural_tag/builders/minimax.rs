// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{StructuralTagBuilder, ToolCallGrammar, triggered_calls_format};
use crate::structural_tag::policy::{ResolvedToolCallingPolicy, resolve_tool_schema};
use crate::structural_tag::wire::{
    Format, JsonSchemaFormat, JsonSchemaStyle, TagFormat, TagsWithSeparatorFormat,
};

pub(crate) static MINIMAX_M2: StructuralTagBuilder = StructuralTagBuilder::new(&M2);

struct MiniMaxFormat {
    block_begin: &'static str,
    block_end: &'static str,
    invoke_begin: &'static str,
    invoke_end: &'static str,
    style: JsonSchemaStyle,
    excludes: &'static [&'static str],
}

static M2: MiniMaxFormat = MiniMaxFormat {
    block_begin: "<minimax:tool_call>",
    block_end: "</minimax:tool_call>",
    invoke_begin: "<invoke name=\"",
    invoke_end: "</invoke>",
    style: JsonSchemaStyle::MinimaxXml,
    excludes: &["<minimax:tool_call>", "<invoke name="],
};

impl MiniMaxFormat {
    fn block(&self, policy: &ResolvedToolCallingPolicy<'_>, any_order: bool) -> TagFormat {
        TagFormat {
            begin: format!("{}\n", self.block_begin),
            content: Box::new(Format::TagsWithSeparator(TagsWithSeparatorFormat {
                tags: policy
                    .tools
                    .iter()
                    .map(|tool| TagFormat {
                        begin: format!("{}{}\">\n", self.invoke_begin, tool.name),
                        content: Box::new(Format::JsonSchema(JsonSchemaFormat {
                            json_schema: resolve_tool_schema(tool, policy.schema_mode),
                            style: self.style,
                            any_order,
                        })),
                        end: format!("\n{}", self.invoke_end),
                    })
                    .collect(),
                separator: "\n".into(),
                at_least_one: true,
                stop_after_first: policy.stop_after_first(),
            })),
            end: format!("\n{}", self.block_end),
        }
    }
}

impl ToolCallGrammar for MiniMaxFormat {
    fn reasoning_begin(&self) -> Option<&'static str> {
        Some("<think>")
    }
    fn reasoning_end(&self) -> Option<&'static str> {
        Some("</think>")
    }
    fn tool_call_excludes(&self) -> &'static [&'static str] {
        self.excludes
    }

    fn build_triggered_calls(
        &self,
        policy: &ResolvedToolCallingPolicy<'_>,
        exclude_special_tokens: bool,
        any_order: bool,
    ) -> anyhow::Result<Format> {
        Ok(triggered_calls_format(
            self.block_begin,
            vec![self.block(policy, any_order)],
            self.reasoning_begin(),
            self.reasoning_end(),
            &[self.invoke_begin, self.invoke_end, self.block_end],
            exclude_special_tokens,
            policy,
        ))
    }
    fn build_tool_calls_only(
        &self,
        policy: &ResolvedToolCallingPolicy<'_>,
        any_order: bool,
    ) -> anyhow::Result<Format> {
        Ok(Format::Tag(self.block(policy, any_order)))
    }
}
