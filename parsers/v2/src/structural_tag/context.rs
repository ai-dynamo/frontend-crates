// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::Tool;

/// OpenAI tool-selection semantics normalized by the serving adapter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StructuralTagToolChoice<'a> {
    None,
    Auto,
    Required,
    Named(&'a str),
}

/// Controls which function-argument schemas guided decoding enforces.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StructuralTagSchemaMode {
    /// Enforce a tool's schema only when that tool declares `strict: true`.
    #[default]
    Auto,
    /// Enforce every request-provided tool schema.
    Strict,
}

/// Controls which layer owns a reasoning boundary opened by the prompt.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningBoundary {
    /// Include the reasoning body and closing marker in the structural tag.
    #[default]
    StructuralTag,
    /// Build only the post-reasoning suffix; the caller activates it externally.
    External,
}

/// Optional overrides for model-family structural-tag behavior.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct StructuralTagOptions {
    /// Overrides whether reasoning and tool-call markers are excluded from free-text regions.
    pub exclude_special_tokens: Option<bool>,
    /// Controls which layer owns a reasoning boundary opened by the prompt.
    pub reasoning_boundary: ReasoningBoundary,
    /// Allow tool argument object properties to appear in any order.
    ///
    /// This weakens JSON Schema validation: required properties may be absent
    /// and duplicate properties may be accepted. It does not affect a
    /// structured-output response schema.
    pub tool_arguments_any_order: bool,
}

/// Request-scoped inputs for building a model-family structural tag.
#[derive(Debug, Clone, Copy)]
pub struct StructuralTagContext<'a> {
    pub tool_choice: StructuralTagToolChoice<'a>,
    pub tools: &'a [Tool],
    pub parallel_tool_calls: Option<bool>,
    pub schema_mode: StructuralTagSchemaMode,
    /// Optional final-response schema used as an alternative for `auto` tool choice.
    pub structured_output_schema: Option<&'a Value>,
    /// Whether generation starts inside a reasoning block opened by the prompt.
    pub starts_in_reasoning: bool,
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{ReasoningBoundary, StructuralTagOptions};

    #[test]
    fn structural_tag_options_use_stable_json_names() {
        let options: StructuralTagOptions = serde_json::from_value(json!({
            "exclude_special_tokens": false
        }))
        .expect("valid structural tag options should deserialize");
        assert_eq!(options.exclude_special_tokens, Some(false));

        let options: StructuralTagOptions = serde_json::from_value(json!({
            "reasoning_boundary": "external",
            "tool_arguments_any_order": true
        }))
        .expect("valid structural tag options should deserialize");
        assert_eq!(options.reasoning_boundary, ReasoningBoundary::External);
        assert!(options.tool_arguments_any_order);

        assert_eq!(
            StructuralTagOptions::default().reasoning_boundary,
            ReasoningBoundary::StructuralTag
        );
        assert!(!StructuralTagOptions::default().tool_arguments_any_order);

        assert!(serde_json::from_value::<StructuralTagOptions>(json!({"unknown": true})).is_err());
    }
}
