// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Kimi K3 XTML structural-tag generation.
//!
//! K3 does not emit the generic JSON shape used by legacy forced-tool guided
//! decoding. Tool calls live in a native XTML `tools` channel with one or more
//! nested `call` and typed `argument` elements. This builder mirrors that wire
//! format so named and required tool choices can be constrained without
//! changing what the K3 parser expects.

use percent_encoding::percent_decode_str;
use serde_json::{Map, Value};

use super::builder::{ToolCallFormatBuildContext, resolve_tools_to_include};
use super::format::{
    AnyTextFormat, ConstStringFormat, Format, JsonSchemaFormat, JsonSchemaStyle, OptionalFormat,
    OrFormat, RegexFormat, SequenceFormat, StarFormat, StructuralTag, TagFormat,
    TagsWithSeparatorFormat,
};
use crate::tool_calling::ToolDefinition;

const OPEN: &str = "<|open|>";
const CLOSE: &str = "<|close|>";
const SEP: &str = "<|sep|>";
const RESPONSE_OPEN: &str = "<|open|>response<|sep|>";
const RESPONSE_CLOSE: &str = "<|close|>response<|sep|>";
const TOOLS_OPEN: &str = "<|open|>tools<|sep|>";
const TOOLS_CLOSE: &str = "<|close|>tools<|sep|>";
const CALL_CLOSE: &str = "<|close|>call<|sep|>";
const ARGUMENT_CLOSE: &str = "<|close|>argument<|sep|>";
const MESSAGE_CLOSE: &str = "<|close|>message<|sep|>";

const STRING_ATOM: &str = r"(?:[^<]|<[^|])";

fn escape_attr(value: &str) -> String {
    value.replace('&', "&amp;").replace('"', "&quot;")
}

fn optional(content: Format) -> Format {
    Format::Optional(OptionalFormat {
        content: Box::new(content),
    })
}

fn star(content: Format) -> Format {
    Format::Star(StarFormat {
        content: Box::new(content),
    })
}

fn one_of(elements: Vec<Format>) -> Format {
    if elements.len() == 1 {
        elements.into_iter().next().expect("one element")
    } else {
        Format::Or(OrFormat { elements })
    }
}

fn bounded_string_regex(schema: &Map<String, Value>) -> Option<String> {
    let max_len = schema.get("maxLength")?.as_u64()?;
    if max_len > 4096 {
        return None;
    }
    let min_len = schema
        .get("minLength")
        .and_then(Value::as_u64)
        .filter(|min| *min <= max_len)
        .unwrap_or(0);
    Some(format!("{STRING_ATOM}{{{min_len},{max_len}}}"))
}

// Resolve before extracting a property's JSON grammar: local pointers belong
// to the complete tool schema, not to the extracted argument schema.
fn resolve_argument_schema(root: &Value, schema: &Value, depth: usize) -> Option<Value> {
    resolve_argument_schema_inner(root, schema, depth, &mut 4096)
}

fn resolve_argument_schema_inner(
    root: &Value,
    schema: &Value,
    depth: usize,
    remaining: &mut usize,
) -> Option<Value> {
    // Bound expansion as well as cycles: repeated references can form a DAG.
    *remaining = remaining.checked_sub(1)?;
    if depth >= 16 {
        return None;
    }
    let Some(object) = schema.as_object() else {
        return schema.is_boolean().then(|| schema.clone());
    };
    // A nested resource changes reference scope. Leave unsupported scopes and
    // dynamic references on the existing permissive path.
    if ["$id", "$dynamicRef", "$recursiveRef"]
        .iter()
        .any(|key| object.contains_key(*key))
    {
        return None;
    }
    if let Some(reference) = object.get("$ref") {
        if object.keys().any(|key| {
            !matches!(
                key.as_str(),
                "$ref"
                    | "description"
                    | "title"
                    | "$comment"
                    | "default"
                    | "examples"
                    | "deprecated"
                    | "readOnly"
                    | "writeOnly"
            )
        }) {
            return None; // Do not discard sibling validation constraints.
        }
        let pointer = reference.as_str()?.strip_prefix('#')?;
        let pointer = percent_decode_str(pointer).decode_utf8().ok()?;
        return resolve_argument_schema_inner(root, root.pointer(&pointer)?, depth + 1, remaining);
    }
    let mut resolved = object.clone();
    // Active references are inlined below; unused definitions need no new scope.
    resolved.remove("$defs");
    resolved.remove("definitions");
    // Visit schema positions only; a literal enum/const object containing a
    // "$ref" key is data and must remain unchanged.
    for key in ["properties", "patternProperties", "dependentSchemas"] {
        if let Some(children) = object.get(key).and_then(Value::as_object) {
            let children = children
                .iter()
                .map(|(name, child)| {
                    Some((
                        name.clone(),
                        resolve_argument_schema_inner(root, child, depth, remaining)?,
                    ))
                })
                .collect::<Option<Map<_, _>>>()?;
            resolved.insert(key.into(), Value::Object(children));
        }
    }
    if let Some(dependencies) = object.get("dependencies").and_then(Value::as_object) {
        let dependencies = dependencies
            .iter()
            .map(|(name, dependency)| {
                let dependency = if dependency.is_array() {
                    dependency.clone()
                } else {
                    resolve_argument_schema_inner(root, dependency, depth, remaining)?
                };
                Some((name.clone(), dependency))
            })
            .collect::<Option<Map<_, _>>>()?;
        resolved.insert("dependencies".into(), Value::Object(dependencies));
    }
    for key in ["allOf", "anyOf", "oneOf", "prefixItems"] {
        if let Some(children) = object.get(key).and_then(Value::as_array) {
            let children = children
                .iter()
                .map(|child| resolve_argument_schema_inner(root, child, depth, remaining))
                .collect::<Option<Vec<_>>>()?;
            resolved.insert(key.into(), Value::Array(children));
        }
    }
    for key in [
        "items",
        "additionalItems",
        "additionalProperties",
        "contains",
        "propertyNames",
        "not",
        "if",
        "then",
        "else",
        "unevaluatedItems",
        "unevaluatedProperties",
    ] {
        if let Some(child) = object.get(key) {
            let child = if let Some(items) = child.as_array() {
                Value::Array(
                    items
                        .iter()
                        .map(|item| resolve_argument_schema_inner(root, item, depth, remaining))
                        .collect::<Option<Vec<_>>>()?,
                )
            } else {
                resolve_argument_schema_inner(root, child, depth, remaining)?
            };
            resolved.insert(key.into(), child);
        }
    }
    Some(Value::Object(resolved))
}

fn argument_tag(
    key: &str,
    schema: &Value,
    root_defs: Option<&Map<String, Value>>,
) -> Option<TagFormat> {
    let schema_object = schema.as_object()?;
    let (json_type, xtml_type) = match schema_object.get("type").and_then(Value::as_str)? {
        "string" => ("string", "string"),
        "integer" => ("integer", "number"),
        "number" => ("number", "number"),
        "boolean" => ("boolean", "boolean"),
        "null" => ("null", "null"),
        "object" => ("object", "object"),
        "array" => ("array", "array"),
        _ => return None,
    };
    let begin = format!(
        "{OPEN}argument key=\"{}\" type=\"{xtml_type}\"{SEP}",
        escape_attr(key)
    );

    let content = if json_type == "string" {
        let enum_values = schema_object
            .get("enum")
            .and_then(Value::as_array)
            .cloned()
            .or_else(|| {
                schema_object
                    .get("const")
                    .and_then(Value::as_str)
                    .map(|value| vec![Value::String(value.to_string())])
            });
        if let Some(values) = enum_values.filter(|values| {
            !values.is_empty()
                && values.len() <= 256
                && values
                    .iter()
                    .all(|value| value.as_str().is_some_and(|string| !string.contains("<|")))
        }) {
            one_of(
                values
                    .iter()
                    .filter_map(Value::as_str)
                    .map(|value| {
                        Format::ConstString(ConstStringFormat {
                            value: value.to_string(),
                        })
                    })
                    .collect(),
            )
        } else if let Some(pattern) = bounded_string_regex(schema_object) {
            Format::Regex(RegexFormat { pattern })
        } else {
            Format::AnyText(AnyTextFormat {
                excludes: vec![CLOSE.to_string()],
            })
        }
    } else {
        let mut embedded = schema_object.clone();
        if let Some(root_defs) = root_defs {
            for (key, value) in root_defs {
                embedded.entry(key.clone()).or_insert_with(|| value.clone());
            }
        }
        Format::JsonSchema(JsonSchemaFormat {
            json_schema: Value::Object(embedded),
            style: JsonSchemaStyle::Json,
        })
    };

    Some(TagFormat {
        begin,
        content: Box::new(content),
        end: ARGUMENT_CLOSE.to_string(),
    })
}

fn permissive_argument_tag() -> TagFormat {
    TagFormat {
        begin: format!("{OPEN}argument "),
        content: Box::new(Format::Sequence(SequenceFormat {
            elements: vec![
                Format::Regex(RegexFormat {
                    pattern: format!(r"[^<]*{}", SEP.replace('|', r"\|")),
                }),
                Format::AnyText(AnyTextFormat {
                    excludes: vec![CLOSE.to_string()],
                }),
            ],
        })),
        end: ARGUMENT_CLOSE.to_string(),
    }
}

fn arguments_block(parameters: Option<&Value>) -> Format {
    let Some(parameters) = parameters.and_then(Value::as_object) else {
        return star(Format::Tag(permissive_argument_tag()));
    };
    let Some(properties) = parameters.get("properties").and_then(Value::as_object) else {
        return star(Format::Tag(permissive_argument_tag()));
    };
    if properties.is_empty() {
        return star(Format::Tag(permissive_argument_tag()));
    }

    let root_defs: Map<String, Value> = ["$defs", "definitions"]
        .into_iter()
        .filter_map(|key| {
            parameters
                .get(key)
                .and_then(Value::as_object)
                .map(|value| (key.to_string(), Value::Object(value.clone())))
        })
        .collect();
    let root_schema = Value::Object(parameters.clone());
    let tags = properties
        .iter()
        .map(|(key, schema)| {
            Format::Tag(
                resolve_argument_schema(&root_schema, schema, 0)
                    .and_then(|resolved| argument_tag(key, &resolved, None))
                    // Unsupported reference scopes must not weaken an already
                    // supported typed schema.
                    .or_else(|| argument_tag(key, schema, Some(&root_defs)))
                    .unwrap_or_else(permissive_argument_tag),
            )
        })
        .collect();
    star(one_of(tags))
}

fn call_tag(tool: &ToolDefinition, strict_schema: bool) -> TagFormat {
    // Match vLLM's K3 behavior: use the declared schema unless the caller
    // explicitly sets strict=false. Global strict mode overrides that opt-out.
    let parameters = if super::builder::kimi_uses_declared_tool_schema(tool, strict_schema) {
        tool.parameters.as_ref()
    } else {
        None
    };
    let begin = format!("{OPEN}call tool=\"{}\" index=\"", escape_attr(&tool.name));
    TagFormat {
        begin,
        content: Box::new(Format::Sequence(SequenceFormat {
            elements: vec![
                Format::Regex(RegexFormat {
                    pattern: "[0-9]+".to_string(),
                }),
                Format::ConstString(ConstStringFormat {
                    value: format!("\"{SEP}"),
                }),
                arguments_block(parameters),
            ],
        })),
        end: CALL_CLOSE.to_string(),
    }
}

/// Build the format-style xgrammar tag for K3's response + tools channels.
pub(crate) fn build_kimi_k3(
    ctx: &ToolCallFormatBuildContext<'_>,
) -> anyhow::Result<Option<StructuralTag>> {
    let (tools, at_least_one) = resolve_tools_to_include(ctx)?;
    if tools.is_empty() {
        return Ok(None);
    }

    // Moonshot's named-tool contract returns the selected call with no
    // assistant content. Leaving the response body as `any_text` lets the model
    // put a second, generic `<tool_call>...</tool_call>` representation there
    // before emitting the structurally constrained XTML call. Restrict only
    // named choice; auto/required may legitimately include response text.
    let response_content = if matches!(ctx.tool_choice, crate::tool_calling::ToolChoice::Named(_)) {
        Format::ConstString(ConstStringFormat {
            value: String::new(),
        })
    } else {
        // Reserve XTML controls for channel transitions so a direct tools
        // channel cannot be swallowed as response text (which masks EOS).
        Format::AnyText(AnyTextFormat {
            excludes: vec![OPEN.to_string(), CLOSE.to_string()],
        })
    };
    // Native output may skip the response channel or go straight from its
    // body to tools, so both response markers are optional.
    let response = vec![
        optional(Format::ConstString(ConstStringFormat {
            value: RESPONSE_OPEN.to_string(),
        })),
        response_content,
        optional(Format::ConstString(ConstStringFormat {
            value: RESPONSE_CLOSE.to_string(),
        })),
    ];
    let calls = Format::TagsWithSeparator(TagsWithSeparatorFormat {
        tags: tools
            .into_iter()
            .map(|tool| call_tag(tool, ctx.strict_schema()))
            .collect(),
        separator: String::new(),
        at_least_one: true,
        stop_after_first: ctx.stop_after_first(),
    });
    let tools_channel = Format::Tag(TagFormat {
        begin: TOOLS_OPEN.to_string(),
        content: Box::new(calls),
        end: TOOLS_CLOSE.to_string(),
    });

    let tools_part = if at_least_one {
        tools_channel
    } else {
        optional(tools_channel)
    };
    let mut elements = response;
    elements.push(tools_part);
    elements.push(optional(Format::ConstString(ConstStringFormat {
        value: MESSAGE_CLOSE.to_string(),
    })));

    Ok(Some(StructuralTag {
        format: Format::Sequence(SequenceFormat { elements }),
    }))
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::tool_calling::structural_tag::builder::StructuralTagSchemaMode;
    use crate::tool_calling::{ToolChoice, ToolDefinition};

    fn tools() -> Vec<ToolDefinition> {
        vec![
            ToolDefinition {
                name: "get_weather".to_string(),
                parameters: Some(json!({
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "days": {"type": "integer"}
                    },
                    "required": ["city"]
                })),
                strict: None,
            },
            ToolDefinition {
                name: "run_command".to_string(),
                parameters: Some(json!({
                    "type": "object",
                    "properties": {"command": {"type": "string"}}
                })),
                strict: None,
            },
        ]
    }

    fn context<'a>(
        choice: &'a ToolChoice,
        tools: &'a [ToolDefinition],
    ) -> ToolCallFormatBuildContext<'a> {
        ToolCallFormatBuildContext {
            tool_choice: choice,
            tools,
            parallel_tool_calls: None,
            schema_mode: StructuralTagSchemaMode::Auto,
            starts_in_reasoning: false,
        }
    }

    #[test]
    fn local_references_retain_object_and_deep_enum_constraints() {
        let object_schema = json!({
            "$defs": {"Schema": {
                "properties": {"input": {"type": "string"}, "notes": {"type": "string"}},
                "required": ["input", "notes"], "type": "object", "additionalProperties": false
            }},
            "properties": {"data": {"$ref": "#/$defs/Schema"}},
            "required": ["data"], "type": "object", "additionalProperties": false
        });
        let resolved =
            resolve_argument_schema(&object_schema, &object_schema["properties"]["data"], 0)
                .unwrap();
        let tag = serde_json::to_value(argument_tag("data", &resolved, None).unwrap()).unwrap();
        assert_eq!(
            tag["begin"],
            format!("{OPEN}argument key=\"data\" type=\"object\"{SEP}")
        );
        assert_eq!(
            tag["content"]["json_schema"],
            object_schema["$defs"]["Schema"]
        );

        let deep_schema = json!({
            "type": "object", "properties": {
                "operations": {"type": "array", "items": {"anyOf": [
                    {"type": "object", "properties": {"title": {"type": "string"}}, "required": ["title"]},
                    {"type": "object", "properties": {"kind": {"type": "string", "enum": ["MP1580_DEEP_REF_INLINED"]}}, "required": ["kind"]}
                ]}},
                "marker": {"$ref": "#/properties/operations/items/anyOf/1/properties/kind"}
            }, "required": ["operations", "marker"]
        });
        let resolved =
            resolve_argument_schema(&deep_schema, &deep_schema["properties"]["marker"], 0).unwrap();
        let tag = serde_json::to_value(argument_tag("marker", &resolved, None).unwrap()).unwrap();
        assert_eq!(tag["content"]["value"], "MP1580_DEEP_REF_INLINED");
    }

    #[test]
    fn embedded_reference_uses_original_root_and_preserves_literal_data() {
        let schema = json!({
            "$defs": {"Payload": {"type": "object", "properties": {
                "value": {"$ref": "#/properties/count"},
                "literal": {"const": {"$ref": "this is data"}}
            }}},
            "properties": {
                "count": {"type": "integer"},
                "data": {"$ref": "#/$defs/Payload"}
            }
        });
        let resolved = resolve_argument_schema(&schema, &schema["properties"]["data"], 0).unwrap();
        assert_eq!(resolved["properties"]["value"], json!({"type": "integer"}));
        assert_eq!(
            resolved["properties"]["literal"]["const"],
            json!({"$ref": "this is data"})
        );
    }

    #[test]
    fn percent_encoded_reference_retains_argument_type() {
        let root = json!({
            "$defs": {"Foo Bar": {"type": "integer"}},
            "properties": {"value": {"$ref": "#/$defs/Foo%20Bar"}}
        });
        let resolved = resolve_argument_schema(&root, &root["properties"]["value"], 0).unwrap();
        assert_eq!(resolved, json!({"type": "integer"}));
    }

    #[test]
    fn dependency_schemas_resolve_local_references_without_changing_property_dependencies() {
        let root = json!({
            "$defs": {"Value": {"type": "integer"}},
            "properties": {"data": {
                "type": "object",
                "dependencies": {
                    "mode": {"properties": {"value": {"$ref": "#/$defs/Value"}}},
                    "name": ["mode"]
                }
            }}
        });
        let resolved = resolve_argument_schema(&root, &root["properties"]["data"], 0).unwrap();
        assert_eq!(
            resolved["dependencies"]["mode"]["properties"]["value"],
            json!({"type": "integer"})
        );
        assert_eq!(resolved["dependencies"]["name"], json!(["mode"]));
    }

    #[test]
    fn resolver_limits_preserve_preexisting_typed_argument_grammars() {
        let large_properties: Map<String, Value> = (0..4097)
            .map(|index| (format!("field{index}"), json!({"type": "integer"})))
            .collect();
        let field_schema = json!({"type": "object", "properties": large_properties});
        let parameters = json!({"properties": {"value": field_schema.clone()}});
        let value = serde_json::to_value(arguments_block(Some(&parameters))).unwrap();
        assert_eq!(
            value["content"]["begin"],
            format!("{OPEN}argument key=\"value\" type=\"object\"{SEP}")
        );
        assert_eq!(value["content"]["content"]["json_schema"], field_schema);
    }

    #[test]
    fn unsupported_reference_scopes_and_constraints_remain_permissive() {
        let root =
            json!({"$defs": {"Cycle": {"$ref": "#/$defs/Cycle"}, "Text": {"type": "string"}}});
        for schema in [
            json!({"$ref": "#/$defs/Cycle"}),
            json!({"$ref": "#/$defs/Missing"}),
            json!({"$ref": "https://example.test/schema"}),
            json!({"$ref": "#/$defs/Text", "enum": ["fixed"]}),
            json!({"$id": "nested", "type": "object"}),
        ] {
            assert!(resolve_argument_schema(&root, &schema, 0).is_none());
        }
    }

    #[test]
    fn named_choice_requires_only_selected_xtml_call() {
        let tools = tools();
        let choice = ToolChoice::Named("get_weather".to_string());
        let value =
            serde_json::to_value(build_kimi_k3(&context(&choice, &tools)).unwrap().unwrap())
                .unwrap();

        assert_eq!(value["type"], "structural_tag");
        assert_eq!(value["format"]["type"], "sequence");
        let tools_tag = &value["format"]["elements"][3];
        assert_eq!(tools_tag["begin"], TOOLS_OPEN);
        let calls = tools_tag["content"]["tags"].as_array().unwrap();
        assert_eq!(calls.len(), 1);
        assert!(
            calls[0]["begin"]
                .as_str()
                .unwrap()
                .contains("tool=\"get_weather\"")
        );
        assert!(
            !value.to_string().contains("tool=\\\"run_command\\\""),
            "a named choice must exclude every other tool"
        );
    }

    #[test]
    fn named_choice_requires_an_empty_response_body() {
        let tools = tools();
        let choice = ToolChoice::Named("get_weather".to_string());
        let value =
            serde_json::to_value(build_kimi_k3(&context(&choice, &tools)).unwrap().unwrap())
                .unwrap();

        let response_body = &value["format"]["elements"][1];
        assert_eq!(response_body["type"], "const_string");
        assert_eq!(response_body["value"], "");
    }

    #[test]
    fn response_text_cannot_swallow_xtml_channels() {
        let tools = tools();

        for choice in [ToolChoice::Auto, ToolChoice::Required] {
            let value =
                serde_json::to_value(build_kimi_k3(&context(&choice, &tools)).unwrap().unwrap())
                    .unwrap();
            let response_body = &value["format"]["elements"][1];

            assert_eq!(
                response_body["type"], "any_text",
                "{choice:?} must retain response text"
            );
            assert_eq!(
                response_body["excludes"],
                json!([OPEN, CLOSE]),
                "{choice:?} must reserve control markers for channel transitions"
            );
        }
    }

    #[test]
    fn named_choice_is_mandatory_and_auto_is_optional() {
        let tools = tools();
        let named = ToolChoice::Named("get_weather".to_string());
        let named_value =
            serde_json::to_value(build_kimi_k3(&context(&named, &tools)).unwrap().unwrap())
                .unwrap();
        assert_eq!(named_value["format"]["elements"][3]["type"], "tag");

        let auto = ToolChoice::Auto;
        let auto_value =
            serde_json::to_value(build_kimi_k3(&context(&auto, &tools)).unwrap().unwrap()).unwrap();
        assert_eq!(auto_value["format"]["elements"][3]["type"], "optional");
    }

    #[test]
    fn named_choice_keeps_declared_argument_schema() {
        let tools = tools();
        let choice = ToolChoice::Named("get_weather".to_string());
        let value =
            serde_json::to_value(build_kimi_k3(&context(&choice, &tools)).unwrap().unwrap())
                .unwrap();
        let call = &value["format"]["elements"][3]["content"]["tags"][0];
        let argument_alternatives = &call["content"]["elements"][2]["content"]["elements"];
        assert!(
            argument_alternatives
                .to_string()
                .contains("key=\\\"city\\\"")
        );
        assert!(
            argument_alternatives
                .to_string()
                .contains("key=\\\"days\\\"")
        );
    }

    #[test]
    fn explicit_non_strict_tool_uses_vllm_permissive_argument_shape() {
        let tools = vec![ToolDefinition {
            name: "get_weather".to_string(),
            parameters: Some(json!({
                "type": "object",
                "properties": {"city": {"type": "string"}}
            })),
            strict: Some(false),
        }];
        let choice = ToolChoice::Named("get_weather".to_string());
        let value =
            serde_json::to_value(build_kimi_k3(&context(&choice, &tools)).unwrap().unwrap())
                .unwrap();
        let call = &value["format"]["elements"][3]["content"]["tags"][0];
        let arguments = &call["content"]["elements"][2];

        assert_eq!(arguments["type"], "star");
        assert_eq!(arguments["content"]["begin"], "<|open|>argument ");
        assert!(
            !arguments.to_string().contains("key=\\\"city\\\""),
            "vLLM treats strict=false as a permissive argument schema"
        );
    }

    #[test]
    fn parallel_false_stops_after_the_first_k3_call() {
        let tools = tools();
        let choice = ToolChoice::Required;
        let ctx = ToolCallFormatBuildContext {
            tool_choice: &choice,
            tools: &tools,
            parallel_tool_calls: Some(false),
            schema_mode: StructuralTagSchemaMode::Auto,
            starts_in_reasoning: false,
        };
        let value = serde_json::to_value(build_kimi_k3(&ctx).unwrap().unwrap()).unwrap();

        assert_eq!(
            value["format"]["elements"][3]["content"]["stop_after_first"],
            true
        );
    }
}
