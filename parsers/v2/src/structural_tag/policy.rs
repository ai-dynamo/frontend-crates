// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_structural_tag::schema::canonicalize_local_refs;
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

pub(crate) fn uses_declared_tool_schema(tool: &Tool, schema_mode: StructuralTagSchemaMode) -> bool {
    let enforce_schema =
        schema_mode == StructuralTagSchemaMode::Strict || tool.strict.unwrap_or(false);

    enforce_schema && !tool.parameters.is_null()
}

/// One tool's argument schema exactly as declared (or `true` when the policy
/// leaves it unconstrained), for builders that resolve `$ref`s themselves.
///
/// Who needs which spelling:
/// - JSON-forwarding builders hand the schema to the backend verbatim and take
///   [`resolve_tool_schema`]: every `TemplatedToolCallFormat` (Qwen3-Coder,
///   GLM-4.7, Harmony, Gemma 4, …), DeepSeek DSML, MiniMax M3 (it validates the
///   shape with a literal pointer lookup, like the backend), Muse Spark, Kimi K2.
/// - Self-resolving builders decode `$ref`s while typing arguments and must see
///   the schema as written, taking this function and canonicalizing only what
///   they forward afterwards: Kimi K3. (Beam types its XML arguments from
///   `tool.parameters` directly and does not go through either function.)
pub(crate) fn declared_tool_schema(tool: &Tool, schema_mode: StructuralTagSchemaMode) -> Value {
    if uses_declared_tool_schema(tool, schema_mode) {
        tool.parameters.clone()
    } else {
        // xgrammar uses `true` for syntactically valid, schema-unconstrained JSON.
        json!(true)
    }
}

/// A declared schema reaches the grammar backend with its local `$ref` targets
/// in the literal spelling the backend resolves ([`canonicalize_local_refs`]);
/// an escaped target such as `#/$defs/registry~1Model~0v1%20name` otherwise
/// fails to compile and the request is rejected (ai-dynamo/dynamo#17661). See
/// [`declared_tool_schema`] for the builders that must not take this spelling.
pub(crate) fn resolve_tool_schema(tool: &Tool, schema_mode: StructuralTagSchemaMode) -> Value {
    if uses_declared_tool_schema(tool, schema_mode) {
        canonicalize_local_refs(tool.parameters.clone())
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
