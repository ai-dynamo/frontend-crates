// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Add the template-native empty Qwen call without changing parameter framing.

use serde_json::{Value, json};
use std::collections::HashSet;

use crate::format::{ConstStringFormat, Format, OrFormat, SequenceFormat, TagFormat};

struct NoExternalSchemas;

const MAX_SCHEMA_DEPTH: usize = 64;
const MAX_SCHEMA_VISITS: usize = 4096;
const MAX_LOCAL_REFERENCE_LEN: usize = 4096;

/// JSON Schema validators can recurse without consuming instance data when a
/// local reference loops through applicators such as `allOf` or `not`. Reject
/// those graphs before invoking the validator. This is a bounded structural
/// guard, not a second schema evaluator: unknown reference scopes and graphs
/// beyond the budget simply keep the legacy grammar.
struct SchemaGuard<'a> {
    root: &'a Value,
    remaining: usize,
    active: HashSet<usize>,
}

impl<'a> SchemaGuard<'a> {
    fn spend(&mut self) -> bool {
        let Some(remaining) = self.remaining.checked_sub(1) else {
            return false;
        };
        self.remaining = remaining;
        true
    }

    fn visit(&mut self, schema: &'a Value, depth: usize) -> bool {
        if depth > MAX_SCHEMA_DEPTH || !self.spend() {
            return false;
        }
        let id = schema as *const Value as usize;
        if !self.active.insert(id) {
            return false;
        }
        let safe = self.visit_contents(schema, depth);
        self.active.remove(&id);
        safe
    }

    fn visit_contents(&mut self, schema: &'a Value, depth: usize) -> bool {
        let object = match schema {
            Value::Bool(_) => return true,
            Value::Object(object) => object,
            _ => return false,
        };
        // These keywords change how a reference resolves. Only document-local
        // JSON Pointers with an unchanged scope are proved safe here.
        if [
            "$id",
            "id",
            "$anchor",
            "$dynamicAnchor",
            "$dynamicRef",
            "$recursiveAnchor",
            "$recursiveRef",
        ]
        .iter()
        .any(|keyword| object.contains_key(*keyword))
        {
            return false;
        }
        for (keyword, value) in object {
            if !self.spend() {
                return false;
            }
            let safe = match keyword.as_str() {
                "$ref" => {
                    let Some(reference) = value.as_str() else {
                        return false;
                    };
                    let Some(pointer) = reference.strip_prefix('#') else {
                        return false;
                    };
                    if reference.len() > MAX_LOCAL_REFERENCE_LEN
                        || (!pointer.is_empty() && !pointer.starts_with('/'))
                        || pointer.contains('%')
                    {
                        return false;
                    }
                    let mut chars = pointer.chars();
                    while let Some(character) = chars.next() {
                        if character == '~' && !matches!(chars.next(), Some('0' | '1')) {
                            return false;
                        }
                    }
                    let Some(target) = self.root.pointer(pointer) else {
                        return false;
                    };
                    self.visit(target, depth + 1)
                }
                "$defs" | "definitions" | "properties" | "patternProperties"
                | "dependentSchemas" => {
                    let Some(children) = value.as_object() else {
                        return false;
                    };
                    children.values().all(|child| self.visit(child, depth + 1))
                }
                "allOf" | "anyOf" | "oneOf" | "prefixItems" => {
                    let Some(children) = value.as_array() else {
                        return false;
                    };
                    children.iter().all(|child| self.visit(child, depth + 1))
                }
                "items" if value.is_array() => value
                    .as_array()
                    .unwrap()
                    .iter()
                    .all(|child| self.visit(child, depth + 1)),
                "additionalProperties"
                | "unevaluatedProperties"
                | "propertyNames"
                | "items"
                | "additionalItems"
                | "contains"
                | "not"
                | "if"
                | "then"
                | "else"
                | "unevaluatedItems"
                | "contentSchema" => self.visit(value, depth + 1),
                "dependencies" => {
                    let Some(children) = value.as_object() else {
                        return false;
                    };
                    children.values().all(|child| {
                        self.spend() && (child.is_array() || self.visit(child, depth + 1))
                    })
                }
                _ => true,
            };
            if !safe {
                return false;
            }
        }
        true
    }
}

fn safe_to_validate(schema: &Value) -> bool {
    SchemaGuard {
        root: schema,
        remaining: MAX_SCHEMA_VISITS,
        active: HashSet::new(),
    }
    .visit(schema, 0)
}

impl jsonschema::Retrieve for NoExternalSchemas {
    fn retrieve(
        &self,
        _uri: &jsonschema::Uri<String>,
    ) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
        Err("external schemas are not retrieved during grammar construction".into())
    }
}

/// Whether the effective argument schema accepts `{}`. Invalid, cyclic,
/// externally referenced, scope-changing, or over-budget schemas retain the old
/// grammar rather than bypassing validation. Ordinary acyclic document-local
/// references and compositions are validated against the complete schema.
fn allows_empty_arguments(schema: &Value) -> bool {
    match schema {
        Value::Bool(allow) => *allow,
        _ if safe_to_validate(schema) => jsonschema::options()
            .with_retriever(NoExternalSchemas)
            .build(schema)
            .is_ok_and(|validator| validator.is_valid(&json!({}))),
        _ => false,
    }
}

/// Preserve the original body + closing delimiter as one branch, adding only
/// the native empty close when the schema permits it. The common opening tag
/// already ends in a newline; a native empty call needs no second newline.
/// The legacy branch deliberately keeps its required leading closing newline.
///
/// `schema` must be policy-resolved but retain its original JSON Schema reference
/// spelling. Backend-canonicalized references use literal path segments; passing
/// them here would decode JSON Pointer escapes a second time and can select a
/// different target. The tag's argument body still uses the backend spelling.
pub fn with_native_empty_call(mut tag: TagFormat, schema: &Value) -> TagFormat {
    if !tag.begin.ends_with('\n') || !allows_empty_arguments(schema) {
        return tag;
    }
    let Some(native_close) = tag.end.strip_prefix('\n') else {
        return tag;
    };
    tag.content = Box::new(Format::Or(OrFormat {
        elements: vec![
            Format::ConstString(ConstStringFormat {
                value: native_close.to_string(),
            }),
            Format::Sequence(SequenceFormat {
                elements: vec![
                    *tag.content,
                    Format::ConstString(ConstStringFormat {
                        value: std::mem::take(&mut tag.end),
                    }),
                ],
            }),
        ],
    }));
    tag
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_branch_obeys_the_complete_schema() {
        for (schema, expected) in [
            (json!(true), true),
            (json!(false), false),
            (json!({}), true),
            (
                json!({"type":"object", "properties":{"city":{"type":"string"}}}),
                true,
            ),
            (
                json!({"type":"object", "properties":{"city":{"type":"string"}}, "required":["city"]}),
                false,
            ),
            (json!({"type":"object", "minProperties":1}), false),
            (json!({"$ref":"#"}), false),
            (json!({"type":"object", "enum":[{}]}), true),
            (json!({"type":"object", "enum":[{"city":"Paris"}]}), false),
            (
                json!({"$defs":{"args":{"type":"object","required":["city"]}}, "$ref":"#/$defs/args"}),
                false,
            ),
            (
                json!({"$defs":{"args":{"type":"object","properties":{}}}, "$ref":"#/$defs/args"}),
                true,
            ),
            (
                json!({"allOf":[{"type":"object"},{"minProperties":1}]}),
                false,
            ),
            (json!({"anyOf":[{"type":"string"},{"type":"object"}]}), true),
            (
                json!({"oneOf":[{"type":"object"},{"type":"object"}]}),
                false,
            ),
            (json!({"not":{"type":"object"}}), false),
            (
                json!({"if":{"type":"object"},"then":{"required":["city"]}}),
                false,
            ),
            (json!({"$ref":"https://example.invalid/args"}), false),
            (json!({"$ref":"file:///nonexistent/args.json"}), false),
            (json!({"type":"not-a-type"}), false),
        ] {
            assert_eq!(allows_empty_arguments(&schema), expected, "{schema}");
        }
    }

    #[test]
    fn cyclic_references_and_unknown_scopes_keep_the_legacy_grammar() {
        for schema in [
            json!({"$defs":{"args":{"allOf":[{"$ref":"#/$defs/args"}]}},"$ref":"#/$defs/args"}),
            json!({"type":"object","$defs":{"args":{"type":"object","allOf":[{"$ref":"#/$defs/args"}]}},"$ref":"#/$defs/args"}),
            json!({"$defs":{"args":{"not":{"$ref":"#/$defs/args"}}},"$ref":"#/$defs/args"}),
            json!({"$defs":{"a":{"$ref":"#/$defs/b"},"b":{"$ref":"#/$defs/a"}},"$ref":"#/$defs/a"}),
            json!({"$dynamicRef":"#node"}),
            json!({"$recursiveRef":"#"}),
            json!({"$id":"https://example.invalid/schema","type":"object"}),
            json!({"$anchor":"node","type":"object"}),
            json!({"$defs":{"a b":{"type":"object"}},"$ref":"#/$defs/a%20b"}),
        ] {
            assert!(!safe_to_validate(&schema), "{schema}");
            assert!(!allows_empty_arguments(&schema), "{schema}");
            let tag = TagFormat {
                begin: "<tool_call>\n<function=f>\n".to_string(),
                content: Box::new(Format::JsonSchema(crate::format::JsonSchemaFormat {
                    json_schema: schema.clone(),
                    style: crate::format::JsonSchemaStyle::QwenXml,
                })),
                end: "\n</function>\n</tool_call>".to_string(),
            };
            let before = serde_json::to_value(&tag).unwrap();
            assert_eq!(
                serde_json::to_value(with_native_empty_call(tag, &schema)).unwrap(),
                before
            );
        }
    }

    #[test]
    fn schema_guard_bounds_depth_and_aggregate_dag_expansion() {
        let mut deep = json!({"type":"object"});
        for _ in 0..=MAX_SCHEMA_DEPTH {
            deep = json!({"allOf":[deep]});
        }
        assert!(!safe_to_validate(&deep));
        assert!(!allows_empty_arguments(&deep));

        // A tiny graph can expand exponentially when each reference appears
        // twice. Revisits deliberately consume budget rather than being
        // memoized away, bounding the validator's prospective expansion too.
        let mut definitions = serde_json::Map::new();
        definitions.insert("leaf".into(), json!({"type":"object"}));
        let mut previous = "leaf".to_string();
        for index in 0..14 {
            let name = format!("n{index}");
            let reference = format!("#/$defs/{previous}");
            definitions.insert(
                name.clone(),
                json!({"allOf":[{"$ref":reference},{"$ref":reference}]}),
            );
            previous = name;
        }
        let dag = json!({"$defs":definitions,"$ref":format!("#/$defs/{previous}")});
        assert!(!safe_to_validate(&dag));
        assert!(!allows_empty_arguments(&dag));
        assert!(allows_empty_arguments(
            &json!({"$defs":{"a":{"type":"object"}},"allOf":[{"$ref":"#/$defs/a"},{"$ref":"#/$defs/a"}]})
        ));
    }

    #[test]
    fn legacy_parameter_branch_is_preserved_verbatim() {
        let tag = TagFormat {
            begin: "<tool_call>\n<function=f>\n".to_string(),
            content: Box::new(Format::JsonSchema(crate::format::JsonSchemaFormat {
                json_schema: json!(true),
                style: crate::format::JsonSchemaStyle::QwenXml,
            })),
            end: "\n</function>\n</tool_call>".to_string(),
        };
        let original = serde_json::to_value(&tag).unwrap();
        let actual = serde_json::to_value(with_native_empty_call(tag, &json!(true))).unwrap();
        assert_eq!(actual["begin"], original["begin"]);
        assert_eq!(actual["end"], "");
        assert_eq!(
            actual["content"]["elements"][0]["value"],
            "</function>\n</tool_call>"
        );
        assert_eq!(
            actual["content"]["elements"][1]["elements"][0],
            original["content"]
        );
        assert_eq!(
            actual["content"]["elements"][1]["elements"][1]["value"],
            original["end"]
        );
    }
}
