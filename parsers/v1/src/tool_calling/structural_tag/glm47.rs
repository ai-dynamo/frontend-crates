// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! GLM-4.7 / GLM-5.x native structural-tag generation, matching xgrammar's
//! builtin `glm_4_7` tag.

use std::{collections::HashSet, sync::LazyLock};

use anyhow::Context;
use serde_json::{Value, json};

use super::builder::{ToolCallFormatBuildContext, resolve_tool_schema, resolve_tools_to_include};
use crate::tool_calling::ToolChoice;
use crate::tool_calling::xml::resolve_local_schema_ref;

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

// XGrammar ignores constraining $ref siblings in glm_xml. A sibling is safe
// only when it exactly repeats the resolved target's constraint. Follow aliases
// too: otherwise an annotation-only alias can hide an ignored constraint.
fn checked_ref_target<'a>(
    schema: &'a Value,
    root: &'a Value,
    depth: usize,
) -> anyhow::Result<Option<&'a Value>> {
    let Some(reference) = schema.get("$ref") else {
        return Ok(Some(schema));
    };
    let target = if depth < 16 {
        reference
            .as_str()
            .and_then(|reference| resolve_local_schema_ref(reference, root))
            .map(|target| checked_ref_target(target, root, depth + 1))
            .transpose()?
            .flatten()
    } else {
        None
    };
    if let Some(siblings) = schema.as_object() {
        for (keyword, value) in siblings {
            if keyword != "$ref" && !ANNOTATIONS.contains(&keyword.as_str()) {
                anyhow::ensure!(
                    target.is_some_and(|target| target.get(keyword) == Some(value)),
                    "GLM structural tag cannot enforce $ref sibling `{keyword}`: \
                     the constraint is not provably redundant with its resolved target"
                );
            }
        }
    }
    Ok(target)
}

// Visit schema positions only. Data under default/examples/const/enum and unused
// definitions must not be interpreted as constraints. Shared/ref-cyclic nodes
// are visited once, and oversized graphs fail before a grammar is emitted.
fn check_schema_ref_siblings(root: &Value) -> anyhow::Result<()> {
    let mut pending = vec![(root, String::from("$"))];
    let mut visited = HashSet::new();
    while let Some((schema, path)) = pending.pop() {
        if !visited.insert(schema as *const Value) {
            continue;
        }
        anyhow::ensure!(
            visited.len() <= 4096,
            "GLM schema reference validation exceeded its node budget"
        );
        checked_ref_target(schema, root, 0).with_context(|| format!("GLM schema `{path}`"))?;
        if let Some(target) = schema
            .get("$ref")
            .and_then(Value::as_str)
            .and_then(|reference| resolve_local_schema_ref(reference, root))
        {
            pending.push((target, format!("{path}.$ref")));
        }
        for keyword in [
            "properties",
            "patternProperties",
            "dependentSchemas",
            "dependencies",
        ] {
            if let Some(properties) = schema.get(keyword).and_then(Value::as_object) {
                for (name, child) in properties {
                    if child.is_object() || child.is_boolean() {
                        pending.push((child, format!("{path}.{keyword}.{name}")));
                    }
                }
            }
        }
        for keyword in ["allOf", "anyOf", "oneOf", "prefixItems", "items"] {
            if let Some(children) = schema.get(keyword).and_then(Value::as_array) {
                for (index, child) in children.iter().enumerate() {
                    pending.push((child, format!("{path}.{keyword}[{index}]")));
                }
            }
        }
        for keyword in [
            "items",
            "additionalItems",
            "additionalProperties",
            "contains",
            "propertyNames",
            "not",
            "if",
            "then",
            "else",
            "unevaluatedProperties",
            "unevaluatedItems",
            "contentSchema",
        ] {
            if let Some(child) = schema.get(keyword)
                && (child.is_object() || child.is_boolean())
            {
                pending.push((child, format!("{path}.{keyword}")));
            }
        }
    }
    Ok(())
}

/// glm_xml does not reliably see `type: string` through a top-level `$ref`
/// and forces a quoted JSON value, so inline refs whose siblings are annotations.
fn glm_schema(mut schema: Value) -> anyhow::Result<Value> {
    let root = schema.clone();
    check_schema_ref_siblings(&root)?;
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
                    .and_then(|r| resolve_local_schema_ref(r, &root))
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
    Ok(schema)
}

pub(crate) fn build_glm47(ctx: &ToolCallFormatBuildContext<'_>) -> anyhow::Result<Option<Value>> {
    let (tools, at_least_one) = resolve_tools_to_include(ctx)?;
    let mut tags: Vec<Value> = tools
        .into_iter()
        .map(|tool| {
            let schema = glm_schema(resolve_tool_schema(tool, ctx.strict_schema()))
                .with_context(|| format!("GLM tool `{}`", tool.name))?;
            Ok(json!({
                "type": "tag",
                "begin": format!("{TOOL_CALL_BEGIN}{}", tool.name),
                "content": {"type": "json_schema", "json_schema": schema, "style": "glm_xml"},
                "end": TOOL_CALL_END,
            }))
        })
        .collect::<anyhow::Result<_>>()?;
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
            glm_schema(schema).unwrap()["properties"]["level"],
            json!({"type": "string", "enum": ["low", "high"], "description": "d"})
        );
    }

    #[test]
    fn inlines_percent_encoded_and_escaped_local_refs() {
        for reference in [
            "#/$defs/Caf%C3%A9",
            "#/$defs/A~1B~0C",
            "#/%24defs/A%7E1B%7E0C",
        ] {
            let schema = json!({"type": "object",
                "$defs": {"Café": {"const": "\"hello\""}, "A/B~C": {"const": "\"hello\""}},
                "properties": {"value": {"$ref": reference, "description": "literal"}}});
            assert_eq!(
                glm_schema(schema).unwrap()["properties"]["value"],
                json!({"const": "\"hello\"", "description": "literal"}),
                "{reference}"
            );
        }
    }

    #[test]
    fn rejects_nonredundant_ref_siblings() {
        for maximum in [600, 2000] {
            let schema = json!({"type": "object", "$defs": level_defs(),
                "properties": {"seconds": {"$ref": "#/$defs/Timeout", "maximum": maximum}}});
            let error = glm_schema(schema).unwrap_err();
            let message = format!("{error:#}");
            assert!(message.contains("seconds") && message.contains("maximum"));
        }
        for (target, property) in [
            (
                json!({"type": "integer"}),
                json!({"$ref": "#/$defs/Target", "maximum": 600}),
            ),
            (
                json!({"anyOf": [{"const": "auto"}, {"type": "integer"}]}),
                json!({"$ref": "#/$defs/Target", "type": "integer"}),
            ),
        ] {
            let schema = json!({"type": "object", "$defs": {"Target": target},
                "properties": {"value": property}});
            assert!(glm_schema(schema).is_err());
        }
    }

    #[test]
    fn accepts_exactly_redundant_ref_siblings() {
        let property = json!({"$ref": "#/$defs/Timeout", "type": "integer", "maximum": 1000});
        let schema =
            json!({"type": "object", "$defs": level_defs(), "properties": {"seconds": property}});
        assert_eq!(
            glm_schema(schema).unwrap()["properties"]["seconds"],
            property
        );
    }

    #[test]
    fn validates_alias_targets_before_annotation_inlining() {
        for (alias, property, allowed) in [
            (
                json!({"$ref": "#/$defs/Timeout", "description": "alias"}),
                json!({"$ref": "#/$defs/Alias", "description": "property"}),
                true,
            ),
            (
                json!({"$ref": "#/$defs/Timeout"}),
                json!({"$ref": "#/$defs/Alias", "maximum": 1000}),
                true,
            ),
            (
                json!({"$ref": "#/$defs/Timeout"}),
                json!({"$ref": "#/$defs/Alias", "maximum": 600}),
                false,
            ),
            (
                json!({"$ref": "#/$defs/Timeout", "maximum": 600}),
                json!({"$ref": "#/$defs/Alias"}),
                false,
            ),
        ] {
            let mut defs = level_defs();
            defs["Alias"] = alias;
            let schema =
                json!({"type": "object", "$defs": defs, "properties": {"seconds": property}});
            let result = glm_schema(schema);
            assert_eq!(result.is_ok(), allowed);
            if allowed && property.get("maximum").is_none() {
                assert_eq!(
                    result.unwrap()["properties"]["seconds"],
                    json!({"type": "integer", "maximum": 1000, "description": "property"})
                );
            }
        }
    }

    #[test]
    fn bounded_unknown_refs_cannot_prove_redundancy() {
        for reference in ["#/$defs/Missing", "#/$defs/Loop"] {
            let schema = json!({"type": "object", "$defs": {"Loop": {"$ref": "#/$defs/Loop"}},
                "properties": {"seconds": {"$ref": reference, "type": "integer"}}});
            assert!(glm_schema(schema).is_err());
        }
        let schema = json!({"type": "object", "$defs": {"Loop": {"$ref": "#/$defs/Loop"}},
            "properties": {"seconds": {"$ref": "#/$defs/Loop"}}});
        assert!(glm_schema(schema).is_ok());
    }

    #[test]
    fn rejects_ref_siblings_in_nested_schema_positions() {
        let conflicting = json!({"$ref": "#/$defs/Timeout", "maximum": 600});
        for property in [
            json!({"anyOf": [conflicting, {"type": "null"}]}),
            json!({"allOf": [conflicting]}),
            json!({"type": "array", "items": conflicting}),
            json!({"type": "object", "properties": {"nested": conflicting}}),
        ] {
            let schema = json!({"type": "object", "$defs": level_defs(),
                "properties": {"seconds": property}});
            let error = format!("{:#}", glm_schema(schema).unwrap_err());
            assert!(
                error.contains("seconds") && error.contains("maximum"),
                "{error}"
            );
        }
    }

    #[test]
    fn schema_validation_leaves_literal_data_and_unused_defs_untouched() {
        let literal = json!({"$ref": "#/$defs/Timeout", "maximum": 600});
        let schema = json!({"type": "object", "$defs": {"Unused": literal},
            "properties": {"opts": {"type": "object", "default": literal,
                "examples": [literal], "const": literal, "enum": [literal]}}});
        assert_eq!(glm_schema(schema.clone()).unwrap(), schema);
    }

    #[test]
    fn schema_validation_bounds_large_graphs_and_terminates_shared_cycles() {
        let repeated = json!({"$ref": "#/$defs/Loop"});
        let schema = json!({"type": "object", "$defs": {"Loop": {"anyOf": [repeated, repeated]}},
            "properties": {"seconds": repeated}});
        assert!(glm_schema(schema).is_ok());
        let schema = json!({"anyOf": vec![json!({"type": "integer"}); 4096]});
        let error = format!("{:#}", glm_schema(schema).unwrap_err());
        assert!(error.contains("node budget"), "{error}");
    }

    #[test]
    fn literal_ref_in_default_keeps_schema() {
        let schema = json!({"type": "object", "required": ["seconds"], "properties": {
            "seconds": {"type": "integer", "maximum": 600},
            "opts": {"type": "object", "default": {"$ref": "literal"}}}});
        assert_eq!(glm_schema(schema.clone()).unwrap(), schema);
    }

    #[test]
    fn dangling_ref_left_for_compiler() {
        let schema =
            json!({"type": "object", "properties": {"x": {"$ref": "#/definitions/Missing"}}});
        assert_eq!(glm_schema(schema.clone()).unwrap(), schema);
    }
}
