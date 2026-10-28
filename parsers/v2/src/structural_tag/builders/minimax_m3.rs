// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{StructuralTagBuilder, ToolCallGrammar, triggered_calls_format};
use crate::structural_tag::policy::{ResolvedToolCallingPolicy, resolve_tool_schema};
use crate::structural_tag::wire::{
    Format, JsonSchemaFormat, JsonSchemaStyle, TagFormat, TagsWithSeparatorFormat,
};
use crate::tool_calling::v1core::xml::minimax_m3_parser::{
    native_types_require_json, native_union_children_ambiguous,
};
use serde_json::Value;

pub(crate) static MINIMAX_M3: StructuralTagBuilder = StructuralTagBuilder::new(&MiniMaxM3);

struct MiniMaxM3;
const BLOCK_BEGIN: &str = "]<]minimax[>[<tool_call>";
const BLOCK_END: &str = "]<]minimax[>[</tool_call>";
const INVOKE_BEGIN: &str = "]<]minimax[>[<invoke name=\"";
const INVOKE_END: &str = "]<]minimax[>[</invoke>";

/// Keywords on which XGrammar's `SchemaParser::Parse` dispatches. A node carrying
/// none of them is parsed as `Any`, which the M3 XML converter rejects after this
/// builder has already installed the native tag. Identical in XGrammar 0.2.7 and
/// 0.2.8. Annotations such as `description`, `title` and `default`, and bare bounds
/// such as `minimum`, do not constrain a node on their own.
const XGRAMMAR_SHAPE_KEYWORDS: &[&str] = &[
    "$ref",
    "const",
    "enum",
    "anyOf",
    "oneOf",
    "allOf",
    "type",
    "properties",
    "additionalProperties",
    "unevaluatedProperties",
    "items",
    "prefixItems",
    "unevaluatedItems",
];

impl MiniMaxM3 {
    fn block(
        &self,
        policy: &ResolvedToolCallingPolicy<'_>,
        any_order: bool,
    ) -> anyhow::Result<TagFormat> {
        // XGrammar's M3 converter needs fixed-name argument schemas. Reject
        // unconstrained input here so the serving adapter can select its guided
        // JSON compatibility path, instead of sending an uncompilable grammar.
        for tool in &policy.tools {
            anyhow::ensure!(
                tool.name.is_ascii(),
                "MiniMax M3 native grammar requires ASCII tool names"
            );
            let schema = resolve_tool_schema(tool, policy.schema_mode);
            validate_native_schema(&schema).map_err(|error| {
                anyhow::anyhow!(
                    "MiniMax M3 tool {:?} requires guided JSON: {error}",
                    tool.name
                )
            })?;
        }
        Ok(TagFormat {
            begin: format!("{}\n", BLOCK_BEGIN),
            content: Box::new(Format::TagsWithSeparator(TagsWithSeparatorFormat {
                tags: policy
                    .tools
                    .iter()
                    .map(|tool| TagFormat {
                        begin: format!("{}{}\">", INVOKE_BEGIN, tool.name),
                        content: Box::new(Format::JsonSchema(JsonSchemaFormat {
                            json_schema: resolve_tool_schema(tool, policy.schema_mode),
                            style: JsonSchemaStyle::MinimaxM3Xml,
                            any_order,
                            // Bound structural whitespace so decoding cannot stay
                            // between fields indefinitely. String contents remain
                            // unconstrained by this formatting limit.
                            max_whitespace_cnt: Some(1),
                        })),
                        end: INVOKE_END.into(),
                    })
                    .collect(),
                separator: "\n".into(),
                at_least_one: true,
                stop_after_first: policy.stop_after_first(),
            })),
            end: format!("\n{}", BLOCK_END),
        })
    }
}

/// Whether XGrammar treats `schema` as a constrained node rather than `Any`.
fn constrains_value(schema: &Value) -> bool {
    match schema {
        Value::Bool(true) => false,
        Value::Object(object) => XGRAMMAR_SHAPE_KEYWORDS
            .iter()
            .any(|keyword| object.contains_key(*keyword)),
        _ => true,
    }
}

/// XGrammar's M3 XML style cannot express these schema features, or renders
/// nested tags whose types the M3 parser cannot recover. Decline native tags
/// while preserving the schema for the serving adapter's existing JSON
/// fallback. This is a conservative eligibility check, not schema rewriting.
fn validate_native_schema(root: &Value) -> anyhow::Result<()> {
    let mut pending = vec![(root, 0)];
    let mut remaining = 1024;
    while let Some((schema, depth)) = pending.pop() {
        anyhow::ensure!(
            remaining > 0 && depth <= 64,
            "schema exceeds native eligibility work limit"
        );
        remaining -= 1;
        anyhow::ensure!(
            constrains_value(schema),
            "native tags require a constrained tool schema"
        );
        let Some(object) = schema.as_object() else {
            continue;
        };
        // The native converter cannot intersect schemas, and ignores negation,
        // conditionals and dependent schemas. A successful compilation alone
        // therefore does not establish that the native grammar implements these
        // constraints. JSON fallback preserves the requested schema; enforcement
        // still depends on the backend.
        for keyword in ["allOf", "not", "if", "dependentSchemas"] {
            anyhow::ensure!(
                !object.contains_key(keyword),
                "native tags do not support {keyword} constraints"
            );
        }
        if let Some(reference) = object.get("$ref") {
            let pointer = reference
                .as_str()
                .and_then(|reference| reference.strip_prefix('#'))
                .ok_or_else(|| anyhow::anyhow!("native tags require local schema references"))?;
            // XGrammar's native resolver does not decode URI or JSON Pointer
            // escapes. Do not advertise native support that it cannot compile.
            anyhow::ensure!(
                !pointer.contains(['%', '~']),
                "escaped schema references require guided JSON"
            );
            let target = root.pointer(pointer).ok_or_else(|| {
                anyhow::anyhow!(
                    "unresolved schema reference {reference}; JSON fallback cannot repair it"
                )
            })?;
            // Retain sibling constraints, and apply the same depth/work limits
            // to references as inline schemas (including recursive references).
            pending.push((target, depth + 1));
        }
        // XGrammar's M3 string rule excludes the namespace token anywhere in
        // a value, although JSON Schema's unrestricted string allows it. Do not
        // constrain away valid argument bytes. Finite string literals are safe
        // once the reserved-marker check below has validated their values.
        let allows_string = object.get("type").is_some_and(|kind| {
            kind == "string"
                || kind
                    .as_array()
                    .is_some_and(|types| types.iter().any(|t| t == "string"))
        });
        anyhow::ensure!(
            !allows_string || object.contains_key("const") || object.contains_key("enum"),
            "unrestricted strings require JSON to preserve literal namespace bytes"
        );
        anyhow::ensure!(
            !native_types_require_json(root, schema),
            "native tags cannot preserve mixed string and non-string value types"
        );
        // The parser recovers a child's type through `properties`, an
        // object-valued `additionalProperties`, an object-valued `items`, or a
        // union with exactly one object or array alternative. XGrammar renders
        // the shapes below as nested tags all the same, so their children would
        // come back as strings. Keep them on the JSON path.
        anyhow::ensure!(
            !object.contains_key("prefixItems")
                && object.get("items").is_none_or(|items| !items.is_array()),
            "native tags cannot type positional array items"
        );
        anyhow::ensure!(
            !native_union_children_ambiguous(root, schema),
            "native tags cannot type unions with several object or array alternatives"
        );
        for keyword in ["additionalProperties", "unevaluatedProperties"] {
            anyhow::ensure!(
                object
                    .get(keyword)
                    .is_none_or(|value| value == &Value::Bool(false)),
                "{keyword} requires fixed object property names"
            );
        }
        anyhow::ensure!(
            !object.contains_key("propertyNames")
                && object
                    .get("patternProperties")
                    .is_none_or(|value| value.as_object().is_some_and(|value| value.is_empty())),
            "native tags require fixed object property names"
        );
        anyhow::ensure!(
            !object.contains_key("pattern")
                && !object.contains_key("format")
                && object
                    .get("minLength")
                    .is_none_or(|value| value.as_u64() == Some(0))
                && !object.contains_key("maxLength"),
            "native strings do not support pattern, format, or length constraints"
        );
        for keyword in ["const", "enum"] {
            if let Some(value) = object.get(keyword) {
                let literals: Vec<&Value> = match (keyword, value) {
                    ("enum", Value::Array(options)) => options.iter().collect(),
                    _ => vec![value],
                };
                anyhow::ensure!(
                    literals
                        .iter()
                        .all(|literal| !literal.is_object() && !literal.is_array()),
                    "native tags cannot type nested {keyword} literals"
                );
                // JSON serialization retains the namespace bytes; no schema
                // nodes are rewritten.
                anyhow::ensure!(
                    !value.to_string().contains("]<]minimax[>["),
                    "native string literals cannot contain the namespace marker"
                );
                // XGrammar 0.2.7's literal rules do not accept the UTF-8 bytes
                // for non-ASCII literals. Keep these values on the JSON path.
                anyhow::ensure!(
                    value.to_string().is_ascii(),
                    "non-ASCII literals require guided JSON"
                );
            }
        }
        // Definitions are declarations, not active constraints. Visit only the
        // definitions reached through references above.
        if let Some(children) = object.get("properties").and_then(Value::as_object) {
            anyhow::ensure!(
                children.keys().all(|key| key.is_ascii()
                    && !key.is_empty()
                    && !key.starts_with('/')
                    && !key.contains('>')
                    && key.bytes().any(|byte| !byte.is_ascii_whitespace())),
                "native tags require representable parameter names"
            );
            pending.extend(children.values().map(|child| (child, depth + 1)));
        }
        for keyword in ["anyOf", "oneOf"] {
            if let Some(children) = object.get(keyword).and_then(Value::as_array) {
                pending.extend(children.iter().map(|child| (child, depth + 1)));
            }
        }
        for keyword in ["items", "additionalItems", "contains"] {
            if let Some(child) = object.get(keyword) {
                pending.push((child, depth + 1));
            }
        }
    }
    Ok(())
}

impl ToolCallGrammar for MiniMaxM3 {
    fn supports_reasoning_gate_override(&self) -> bool {
        true
    }

    fn reasoning_begin(&self) -> Option<&'static str> {
        Some("<mm:think>")
    }
    fn reasoning_end(&self) -> Option<&'static str> {
        Some("</mm:think>")
    }
    fn tool_call_excludes(&self) -> &'static [&'static str] {
        &[BLOCK_BEGIN, "]<]minimax[>[<invoke name="]
    }

    fn build_triggered_calls(
        &self,
        policy: &ResolvedToolCallingPolicy<'_>,
        exclude_special_tokens: bool,
        any_order: bool,
    ) -> anyhow::Result<Format> {
        Ok(triggered_calls_format(
            BLOCK_BEGIN,
            vec![self.block(policy, any_order)?],
            self.reasoning_begin(),
            self.reasoning_end(),
            &[INVOKE_BEGIN, INVOKE_END, BLOCK_END],
            exclude_special_tokens,
            policy,
        ))
    }
    fn build_tool_calls_only(
        &self,
        policy: &ResolvedToolCallingPolicy<'_>,
        any_order: bool,
    ) -> anyhow::Result<Format> {
        Ok(Format::Tag(self.block(policy, any_order)?))
    }
}
