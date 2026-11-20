// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_structural_tag::{Tool, minimax_m2};
use serde_json::Value;

use super::{StructuralTagBuilder, ToolCallGrammar, triggered_calls_format};
use crate::structural_tag::policy::{ResolvedToolCallingPolicy, uses_declared_tool_schema};
use crate::structural_tag::wire::{self, Format, TagFormat};

pub(crate) static MINIMAX_M2: StructuralTagBuilder = StructuralTagBuilder::new(&MiniMaxM2);

struct MiniMaxM2;

fn block(policy: &ResolvedToolCallingPolicy<'_>, any_order: bool) -> TagFormat {
    let relaxed = Value::Bool(true);
    let tools: Vec<_> = policy
        .tools
        .iter()
        .map(|tool| Tool {
            name: &tool.name,
            parameters: if uses_declared_tool_schema(tool, policy.schema_mode) {
                &tool.parameters
            } else {
                &relaxed
            },
        })
        .collect();
    wire::import_tag(
        minimax_m2::block(&tools, policy.stop_after_first()),
        any_order,
    )
}

impl ToolCallGrammar for MiniMaxM2 {
    fn reasoning_begin(&self) -> Option<&'static str> {
        Some("<think>")
    }
    fn reasoning_end(&self) -> Option<&'static str> {
        Some("</think>")
    }
    fn tool_call_excludes(&self) -> &'static [&'static str] {
        &[
            minimax_m2::BLOCK_BEGIN,
            minimax_m2::BLOCK_END,
            minimax_m2::INVOKE_PREFIX,
        ]
    }

    fn build_triggered_calls(
        &self,
        policy: &ResolvedToolCallingPolicy<'_>,
        exclude_special_tokens: bool,
        any_order: bool,
    ) -> anyhow::Result<Format> {
        Ok(triggered_calls_format(
            minimax_m2::BLOCK_BEGIN,
            vec![block(policy, any_order)],
            self.reasoning_begin(),
            self.reasoning_end(),
            &[
                minimax_m2::INVOKE_PREFIX,
                minimax_m2::INVOKE_END,
                minimax_m2::BLOCK_END,
            ],
            exclude_special_tokens,
            policy,
        ))
    }
    fn build_tool_calls_only(
        &self,
        policy: &ResolvedToolCallingPolicy<'_>,
        any_order: bool,
    ) -> anyhow::Result<Format> {
        Ok(Format::Tag(block(policy, any_order)))
    }
}
