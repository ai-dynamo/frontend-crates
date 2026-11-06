// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Add the template-native empty Qwen call without changing parameter framing.

use serde_json::{Value, json};

use crate::format::{ConstStringFormat, Format, OrFormat, SequenceFormat, TagFormat};

struct NoExternalSchemas;

impl jsonschema::Retrieve for NoExternalSchemas {
    fn retrieve(
        &self,
        _uri: &jsonschema::Uri<String>,
    ) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
        Err("external schemas are not retrieved during grammar construction".into())
    }
}

/// Whether the effective argument schema accepts `{}`. Invalid or externally
/// referenced schemas retain the old grammar rather than bypassing validation.
fn allows_empty_arguments(schema: &Value) -> bool {
    match schema {
        Value::Bool(allow) => *allow,
        _ => jsonschema::options()
            .with_retriever(NoExternalSchemas)
            .build(schema)
            .is_ok_and(|validator| validator.is_valid(&json!({}))),
    }
}

/// Preserve the original body + closing delimiter as one branch, adding only
/// the native empty close when the schema permits it. The common opening tag
/// already ends in a newline; a native empty call needs no second newline.
/// The legacy branch deliberately keeps its required leading closing newline.
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
            (json!({"$ref":"#"}), true),
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
