// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! GLM-4.7 / GLM-5.x native structural-tag generation, matching xgrammar's
//! builtin `glm_4_7` tag.

use std::sync::LazyLock;

use serde_json::{Value, json};

use super::builder::{ToolCallFormatBuildContext, resolve_tool_schema, resolve_tools_to_include};
use crate::tool_calling::ToolChoice;

const TOOL_CALL_BEGIN: &str = "<tool_call>";
const TOOL_CALL_END: &str = "</tool_call>";
const THINK_BEGIN: &str = "<think>";
const THINK_END: &str = "</think>";
const ARG_MARKERS: [&str; 4] = ["<arg_key>", "</arg_key>", "<arg_value>", "</arg_value>"];

/// Every tool-call control token, banned for `tool_choice="none"`.
pub(crate) static BAN_TOKENS: LazyLock<Vec<String>> = LazyLock::new(|| {
    [TOOL_CALL_BEGIN, TOOL_CALL_END]
        .into_iter()
        .chain(ARG_MARKERS)
        .map(String::from)
        .collect()
});

/// Keywords that do not constrain a value, so a `$ref` carrying only these can be inlined.
const ANNOTATIONS: [&str; 7] = [
    "title",
    "description",
    "default",
    "examples",
    "deprecated",
    "$comment",
    "$id",
];

/// glm_xml does not reliably see `type: string` through a top-level `$ref`
/// and forces a quoted JSON value, so inline refs whose siblings are annotations.
fn glm_schema(mut schema: Value) -> Value {
    let root = schema.clone();
    if let Some(props) = schema.get_mut("properties").and_then(Value::as_object_mut) {
        for prop in props.values_mut() {
            // Bounded so a ref cycle stays a `$ref`.
            for _ in 0..8 {
                let Some(src) = prop.as_object() else { break };
                if src
                    .keys()
                    .any(|k| k != "$ref" && !ANNOTATIONS.contains(&k.as_str()))
                {
                    break;
                }
                let Some(target) = src
                    .get("$ref")
                    .and_then(Value::as_str)
                    .and_then(|r| root.pointer(r.strip_prefix('#')?))
                else {
                    break;
                };
                let mut inlined = target.clone();
                if let Some(dst) = inlined.as_object_mut() {
                    for (k, v) in src.iter().filter(|(k, _)| *k != "$ref") {
                        dst.insert(k.clone(), v.clone());
                    }
                }
                *prop = inlined;
            }
        }
    }
    schema
}

pub(crate) fn build_glm47(ctx: &ToolCallFormatBuildContext<'_>) -> anyhow::Result<Option<Value>> {
    let (tools, at_least_one) = resolve_tools_to_include(ctx)?;
    let mut tags: Vec<Value> = tools
        .into_iter()
        .map(|tool| {
            let schema = glm_schema(resolve_tool_schema(tool, ctx.strict_schema()));
            json!({
                "type": "tag",
                "begin": format!("{TOOL_CALL_BEGIN}{}", tool.name),
                "content": {"type": "json_schema", "json_schema": schema, "style": "glm_xml"},
                "end": TOOL_CALL_END,
            })
        })
        .collect();
    if tags.is_empty() {
        return Ok(None);
    }

    // A named call is the whole reply, so the grammar ends with it.
    let mut format = if matches!(ctx.tool_choice, ToolChoice::Named(_)) {
        tags.remove(0)
    } else {
        // `<tool_call>` stays allowed in free text as the trigger.
        let excludes: Vec<&str> = [THINK_BEGIN, THINK_END, TOOL_CALL_END]
            .into_iter()
            .chain(ARG_MARKERS)
            .collect();
        json!({
            "type": "triggered_tags",
            "triggers": [TOOL_CALL_BEGIN],
            "tags": tags,
            "excludes": excludes,
            "at_least_one": at_least_one,
            "stop_after_first": ctx.stop_after_first(),
        })
    };
    // The free text bans `</think>`, so a prompt-opened reasoning block is closed first.
    if ctx.starts_in_reasoning {
        let excludes: Vec<&str> = [THINK_BEGIN, THINK_END, TOOL_CALL_BEGIN, TOOL_CALL_END]
            .into_iter()
            .chain(ARG_MARKERS)
            .collect();
        let reasoning = json!({
            "type": "tag",
            "begin": "",
            "content": {"type": "any_text", "excludes": excludes},
            "end": THINK_END,
        });
        format = json!({"type": "sequence", "elements": [reasoning, format]});
    }
    Ok(Some(json!({"type": "structural_tag", "format": format})))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn level_defs() -> Value {
        json!({"Level": {"type": "string", "enum": ["low", "high"]}, "Timeout": {"type": "integer", "maximum": 1000}})
    }

    #[test]
    fn inlines_annotation_only_ref() {
        let schema = json!({"type": "object", "$defs": level_defs(),
            "properties": {"level": {"$ref": "#/$defs/Level", "description": "d"}}});
        assert_eq!(
            glm_schema(schema)["properties"]["level"],
            json!({"type": "string", "enum": ["low", "high"], "description": "d"})
        );
    }

    #[test]
    fn keeps_ref_with_constraining_sibling() {
        let prop = json!({"$ref": "#/$defs/Timeout", "maximum": 600});
        let schema =
            json!({"type": "object", "$defs": level_defs(), "properties": {"seconds": prop}});
        assert_eq!(glm_schema(schema)["properties"]["seconds"], prop);
    }

    #[test]
    fn literal_ref_in_default_keeps_schema() {
        let schema = json!({"type": "object", "required": ["seconds"], "properties": {
            "seconds": {"type": "integer", "maximum": 600},
            "opts": {"type": "object", "default": {"$ref": "literal"}}}});
        assert_eq!(glm_schema(schema.clone()), schema);
    }

    #[test]
    fn dangling_ref_left_for_compiler() {
        let schema =
            json!({"type": "object", "properties": {"x": {"$ref": "#/definitions/Missing"}}});
        assert_eq!(glm_schema(schema.clone()), schema);
    }
}
