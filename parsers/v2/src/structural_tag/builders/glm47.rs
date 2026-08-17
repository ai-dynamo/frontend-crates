// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::StructuralTagBuilder;
use super::templated_tool_calls::TemplatedToolCallFormat;
use crate::structural_tag::wire::JsonSchemaStyle;

pub(crate) static GLM47: StructuralTagBuilder = StructuralTagBuilder::new(&GLM47_FORMAT);

const TOOL_CALL_BEGIN: &str = "<tool_call>";
const TOOL_CALL_END: &str = "</tool_call>";
const THINK_BEGIN: &str = "<think>";
const THINK_END: &str = "</think>";
const TOOL_CALL_MARKERS: &[&str] = &[
    TOOL_CALL_BEGIN,
    TOOL_CALL_END,
    "<arg_key>",
    "</arg_key>",
    "<arg_value>",
    "</arg_value>",
];

static GLM47_FORMAT: TemplatedToolCallFormat = TemplatedToolCallFormat {
    tool_call_begin_prefix: TOOL_CALL_BEGIN,
    tool_call_begin_suffix: "",
    tool_call_end: TOOL_CALL_END,
    tool_call_trigger: TOOL_CALL_BEGIN,
    first_tool_call_prefix: "",
    tool_call_separator: "",
    arguments_style: JsonSchemaStyle::GlmXml,
    reasoning_begin: Some(THINK_BEGIN),
    reasoning_end: Some(THINK_END),
    tool_call_excludes: TOOL_CALL_MARKERS,
    exclude_special_tokens: true,
};

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{GLM47, THINK_BEGIN, THINK_END, TOOL_CALL_MARKERS};
    use crate::structural_tag::test_support::{build, tools};
    use crate::structural_tag::{StructuralTagSchemaMode, StructuralTagToolChoice};

    #[test]
    fn auto_builds_glm_triggered_dispatch() {
        let tools = tools();
        let actual = build(
            &GLM47,
            StructuralTagToolChoice::Auto,
            &tools,
            None,
            StructuralTagSchemaMode::Auto,
            false,
        );

        assert_eq!(actual["format"]["type"], "triggered_tags");
        assert_eq!(actual["format"]["triggers"], json!(["<tool_call>"]));
        assert_eq!(actual["format"]["at_least_one"], false);
        assert_eq!(actual["format"]["stop_after_first"], false);
        assert_eq!(
            actual["format"]["excludes"],
            json!([THINK_BEGIN, THINK_END])
        );
        assert_eq!(
            actual["format"]["tags"][0]["begin"],
            "<tool_call>add_numbers"
        );
        assert_eq!(actual["format"]["tags"][0]["end"], "</tool_call>");
        assert_eq!(actual["format"]["tags"][0]["content"]["style"], "glm_xml");
        assert!(
            actual["format"]["tags"][0]["content"]
                .get("any_order")
                .is_none()
        );
        assert_eq!(
            actual["format"]["tags"][1]["content"]["json_schema"],
            tools[1].parameters
        );
    }

    #[test]
    fn required_starts_with_a_triggered_tool_call() {
        let tools = tools();
        let actual = build(
            &GLM47,
            StructuralTagToolChoice::Required,
            &tools,
            Some(false),
            StructuralTagSchemaMode::Auto,
            false,
        );

        assert_eq!(actual["format"]["type"], "triggered_tags");
        assert_eq!(actual["format"]["triggers"], json!(["<tool_call>"]));
        assert_eq!(actual["format"]["at_least_one"], true);
        assert_eq!(actual["format"]["stop_after_first"], true);
    }

    #[test]
    fn named_builds_one_exact_glm_call() {
        let tools = tools();
        let actual = build(
            &GLM47,
            StructuralTagToolChoice::Named("get_weather"),
            &tools,
            Some(true),
            StructuralTagSchemaMode::Auto,
            false,
        );

        assert_eq!(actual["format"]["type"], "tag");
        assert_eq!(actual["format"]["begin"], "<tool_call>get_weather");
        assert_eq!(actual["format"]["end"], "</tool_call>");
        assert_eq!(actual["format"]["content"]["style"], "glm_xml");
        assert!(actual["format"].get("triggers").is_none());
    }

    #[test]
    fn none_excludes_glm_tool_call_markers() {
        let tools = tools();
        let actual = build(
            &GLM47,
            StructuralTagToolChoice::None,
            &tools,
            None,
            StructuralTagSchemaMode::Auto,
            false,
        );

        assert_eq!(actual["format"]["content"]["type"], "any_text");
        assert_eq!(
            actual["format"]["content"]["excludes"],
            json!(TOOL_CALL_MARKERS)
        );
    }
}
