// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde_json::{Value, json};

use crate::Tool;

use super::{StructuralTagContext, StructuralTagSchemaMode, StructuralTagToolChoice};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ToolCallingMode {
    Auto,
    Required,
    Named,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ToolCallCardinality {
    ZeroOrMore,
    ZeroOrOne,
    OneOrMore,
    ExactlyOne,
}

pub(crate) struct ResolvedToolCallingPolicy<'a> {
    pub tools: Vec<&'a Tool>,
    /// Preserved because some model formats use different wire ASTs for
    /// required and named calls even when cardinality is identical.
    pub mode: ToolCallingMode,
    pub cardinality: ToolCallCardinality,
    pub schema_mode: StructuralTagSchemaMode,
}

impl ResolvedToolCallingPolicy<'_> {
    pub fn stop_after_first(&self) -> bool {
        matches!(
            self.cardinality,
            ToolCallCardinality::ZeroOrOne | ToolCallCardinality::ExactlyOne
        )
    }
}

pub(crate) enum ResolvedToolCalling<'a> {
    Disabled,
    Enabled(ResolvedToolCallingPolicy<'a>),
}

/// Resolve request-wide tool-calling semantics before model-specific formatting.
pub(crate) fn resolve_tool_calling<'a>(
    context: &StructuralTagContext<'a>,
) -> anyhow::Result<ResolvedToolCalling<'a>> {
    let parallel_disabled = context.parallel_tool_calls == Some(false);
    let optional_cardinality = if parallel_disabled {
        ToolCallCardinality::ZeroOrOne
    } else {
        ToolCallCardinality::ZeroOrMore
    };
    let required_cardinality = if parallel_disabled {
        ToolCallCardinality::ExactlyOne
    } else {
        ToolCallCardinality::OneOrMore
    };

    match context.tool_choice {
        StructuralTagToolChoice::None => Ok(ResolvedToolCalling::Disabled),
        StructuralTagToolChoice::Auto => {
            Ok(ResolvedToolCalling::Enabled(ResolvedToolCallingPolicy {
                tools: context.tools.iter().collect(),
                mode: ToolCallingMode::Auto,
                cardinality: optional_cardinality,
                schema_mode: context.schema_mode,
            }))
        }
        StructuralTagToolChoice::Required => {
            anyhow::ensure!(
                !context.tools.is_empty(),
                "tool_choice is \"required\" but tools is empty"
            );
            Ok(ResolvedToolCalling::Enabled(ResolvedToolCallingPolicy {
                tools: context.tools.iter().collect(),
                mode: ToolCallingMode::Required,
                cardinality: required_cardinality,
                schema_mode: context.schema_mode,
            }))
        }
        StructuralTagToolChoice::Named(name) => {
            let tool = context
                .tools
                .iter()
                .find(|tool| tool.name == name)
                .ok_or_else(|| {
                    anyhow::anyhow!("tool named \"{name}\" in tool_choice is not present in tools")
                })?;
            Ok(ResolvedToolCalling::Enabled(ResolvedToolCallingPolicy {
                tools: vec![tool],
                mode: ToolCallingMode::Named,
                cardinality: ToolCallCardinality::ExactlyOne,
                schema_mode: context.schema_mode,
            }))
        }
    }
}

pub(crate) fn resolve_tool_schema(tool: &Tool, schema_mode: StructuralTagSchemaMode) -> Value {
    let enforce_schema =
        schema_mode == StructuralTagSchemaMode::Strict || tool.strict.unwrap_or(false);

    if enforce_schema && !tool.parameters.is_null() {
        tool.parameters.clone()
    } else {
        // xgrammar uses `true` for syntactically valid, schema-unconstrained JSON.
        json!(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn required_and_named_modes_remain_distinct_at_equal_cardinality() {
        let tools = [Tool {
            name: "search".to_string(),
            description: None,
            parameters: json!(true),
            strict: None,
        }];
        let context = |tool_choice| StructuralTagContext {
            tool_choice,
            tools: &tools,
            parallel_tool_calls: Some(false),
            schema_mode: StructuralTagSchemaMode::Auto,
            structured_output_schema: None,
            starts_in_reasoning: false,
        };

        let ResolvedToolCalling::Enabled(required) =
            resolve_tool_calling(&context(StructuralTagToolChoice::Required)).unwrap()
        else {
            panic!("required request should be enabled");
        };
        let ResolvedToolCalling::Enabled(named) =
            resolve_tool_calling(&context(StructuralTagToolChoice::Named("search"))).unwrap()
        else {
            panic!("named request should be enabled");
        };

        assert_eq!(required.mode, ToolCallingMode::Required);
        assert_eq!(named.mode, ToolCallingMode::Named);
        assert_eq!(required.cardinality, ToolCallCardinality::ExactlyOne);
        assert_eq!(named.cardinality, ToolCallCardinality::ExactlyOne);
        assert!(required.stop_after_first());
        assert!(named.stop_after_first());
    }
}
