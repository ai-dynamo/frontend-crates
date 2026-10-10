// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_structural_tag::{Tool, format as native, kimi_k2, kimi_k3};
use serde_json::Value;

use super::{StructuralTagBuilder, ToolCallGrammar, triggered_calls_format};
use crate::structural_tag::policy::{
    ResolvedToolCallingPolicy, ToolCallingMode, resolve_tool_schema,
};
use crate::structural_tag::wire::{self, Format, OrFormat};

pub(crate) const KIMI_K2: StructuralTagBuilder = StructuralTagBuilder::new(&KimiK2);
pub(crate) const KIMI_K3: StructuralTagBuilder = StructuralTagBuilder::new(&KimiK3);
struct KimiK2;
struct KimiK3;

fn with_tools<T>(
    policy: &ResolvedToolCallingPolicy<'_>,
    build: impl FnOnce(&[Tool<'_>]) -> T,
) -> T {
    let schemas: Vec<_> = policy
        .tools
        .iter()
        .map(|tool| resolve_tool_schema(tool, policy.schema_mode))
        .collect();
    let tools: Vec<_> = policy
        .tools
        .iter()
        .zip(&schemas)
        .map(|(tool, parameters)| Tool {
            name: &tool.name,
            parameters,
        })
        .collect();
    build(&tools)
}

fn k2_section(policy: &ResolvedToolCallingPolicy<'_>) -> native::TagFormat {
    with_tools(policy, |tools| {
        kimi_k2::section(native::Format::TagsWithSeparator(
            native::TagsWithSeparatorFormat {
                tags: tools.iter().map(kimi_k2::call_tag).collect(),
                separator: String::new(),
                at_least_one: true,
                stop_after_first: policy.stop_after_first(),
            },
        ))
    })
}

impl ToolCallGrammar for KimiK2 {
    fn reasoning_begin(&self) -> Option<&'static str> {
        Some("<think>")
    }
    fn reasoning_end(&self) -> Option<&'static str> {
        Some("</think>")
    }
    fn tool_call_excludes(&self) -> &'static [&'static str] {
        &[
            kimi_k2::TOOL_CALLS_SECTION_BEGIN,
            kimi_k2::TOOL_CALL_BEGIN_PREFIX,
        ]
    }
    fn build_triggered_calls(
        &self,
        policy: &ResolvedToolCallingPolicy<'_>,
        exclude: bool,
        any_order: bool,
    ) -> anyhow::Result<Format> {
        Ok(triggered_calls_format(
            kimi_k2::TOOL_CALLS_SECTION_BEGIN,
            vec![wire::import_tag(k2_section(policy), any_order)],
            self.reasoning_begin(),
            self.reasoning_end(),
            &[
                kimi_k2::TOOL_CALLS_SECTION_END,
                kimi_k2::TOOL_CALL_END,
                kimi_k2::TOOL_CALL_BEGIN_PREFIX,
                kimi_k2::TOOL_CALL_ARGUMENT_BEGIN,
            ],
            exclude,
            policy,
        ))
    }
    fn build_tool_calls_only(
        &self,
        policy: &ResolvedToolCallingPolicy<'_>,
        any_order: bool,
    ) -> anyhow::Result<Format> {
        Ok(Format::Tag(wire::import_tag(k2_section(policy), any_order)))
    }
}

impl ToolCallGrammar for KimiK3 {
    fn reasoning_begin(&self) -> Option<&'static str> {
        Some("<|open|>think<|sep|>")
    }
    fn reasoning_end(&self) -> Option<&'static str> {
        Some("<|close|>think<|sep|>")
    }
    fn tool_call_excludes(&self) -> &'static [&'static str] {
        &[kimi_k3::TOOLS_OPEN, "<|open|>call "]
    }

    fn build_triggered_calls(
        &self,
        policy: &ResolvedToolCallingPolicy<'_>,
        exclude: bool,
        any_order: bool,
    ) -> anyhow::Result<Format> {
        let response = if policy.mode == ToolCallingMode::Required {
            native::Format::ConstString(native::ConstStringFormat {
                value: String::new(),
            })
        } else {
            // These boundaries select the constrained tools branch. They must
            // remain active even when optional marker exclusions are disabled.
            let mut excludes: Vec<_> = self
                .tool_call_excludes()
                .iter()
                .map(|s| (*s).to_owned())
                .collect();
            if exclude {
                excludes.extend([
                    self.reasoning_begin().unwrap().to_owned(),
                    self.reasoning_end().unwrap().to_owned(),
                ]);
            }
            native::Format::AnyText(native::AnyTextFormat { excludes })
        };
        with_tools(policy, |tools| {
            Ok(wire::import_native(
                kimi_k3::build(
                    tools,
                    policy.mode == ToolCallingMode::Required,
                    policy.stop_after_first(),
                    response,
                    any_order,
                )?
                .format,
                any_order,
            ))
        })
    }

    fn build_tool_calls_only(
        &self,
        policy: &ResolvedToolCallingPolicy<'_>,
        any_order: bool,
    ) -> anyhow::Result<Format> {
        with_tools(policy, |tools| {
            Ok(wire::import_native(
                kimi_k3::build(
                    tools,
                    true,
                    policy.stop_after_first(),
                    native::Format::ConstString(native::ConstStringFormat {
                        value: String::new(),
                    }),
                    any_order,
                )?
                .format,
                any_order,
            ))
        })
    }

    fn build_auto_with_structured_output(
        &self,
        policy: &ResolvedToolCallingPolicy<'_>,
        schema: &Value,
        any_order: bool,
    ) -> anyhow::Result<Format> {
        // Import the response separately so argument ordering never weakens the final response schema.
        let response = kimi_k3::response(native::Format::JsonSchema(native::JsonSchemaFormat {
            json_schema: schema.clone(),
            style: native::JsonSchemaStyle::Json,
        }));
        Ok(Format::Or(OrFormat {
            elements: vec![
                self.build_tool_calls_only(policy, any_order)?,
                wire::import_native(response, false),
            ],
        }))
    }
}
