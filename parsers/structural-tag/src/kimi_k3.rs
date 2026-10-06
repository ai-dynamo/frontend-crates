// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Kimi K3 XTML structural-tag generation.
//!
//! K3 does not emit the generic JSON shape used by legacy forced-tool guided
//! decoding. Tool calls live in a native XTML `tools` channel with one or more
//! nested `call` and typed `argument` elements. This builder mirrors that wire
//! format so named and required tool choices can be constrained without
//! changing what the K3 parser expects.

use serde_json::{Map, Value};

use super::format::{
    AnyTextFormat, ConstStringFormat, Format, JsonSchemaFormat, JsonSchemaStyle, OptionalFormat,
    OrFormat, RegexFormat, SequenceFormat, StarFormat, StructuralTag, TagFormat,
    TagsWithSeparatorFormat,
};
use crate::Tool;

const OPEN: &str = "<|open|>";
const CLOSE: &str = "<|close|>";
const SEP: &str = "<|sep|>";
pub const RESPONSE_OPEN: &str = "<|open|>response<|sep|>";
pub const RESPONSE_CLOSE: &str = "<|close|>response<|sep|>";
pub const TOOLS_OPEN: &str = "<|open|>tools<|sep|>";
pub const TOOLS_CLOSE: &str = "<|close|>tools<|sep|>";
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
    let tags = properties
        .iter()
        .map(|(key, schema)| {
            Format::Tag(
                argument_tag(key, schema, Some(&root_defs)).unwrap_or_else(permissive_argument_tag),
            )
        })
        .collect();
    star(one_of(tags))
}

pub fn call_tag(tool: &Tool<'_>) -> TagFormat {
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
                arguments_block(Some(tool.parameters)),
            ],
        })),
        end: CALL_CLOSE.to_string(),
    }
}

/// Build the format-style xgrammar tag for K3's response + tools channels.
pub fn build(
    tools: &[Tool<'_>],
    at_least_one: bool,
    stop_after_first: bool,
    response_content: Format,
) -> StructuralTag {
    let calls = Format::TagsWithSeparator(TagsWithSeparatorFormat {
        tags: tools.iter().map(call_tag).collect(),
        separator: String::new(),
        at_least_one: true,
        stop_after_first,
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
    let mut elements = response_elements(response_content);
    elements.push(tools_part);
    elements.push(optional(Format::ConstString(ConstStringFormat {
        value: MESSAGE_CLOSE.to_string(),
    })));

    StructuralTag {
        format: Format::Sequence(SequenceFormat { elements }),
    }
}

/// A final response without tool calls, including the native channel envelope.
pub fn response(content: Format) -> Format {
    let mut elements = response_elements(content);
    elements.push(optional(Format::ConstString(ConstStringFormat {
        value: MESSAGE_CLOSE.to_string(),
    })));
    Format::Sequence(SequenceFormat { elements })
}

fn response_elements(response_content: Format) -> Vec<Format> {
    vec![
        optional(Format::ConstString(ConstStringFormat {
            value: RESPONSE_OPEN.to_string(),
        })),
        Format::Tag(TagFormat {
            begin: String::new(),
            content: Box::new(response_content),
            end: RESPONSE_CLOSE.to_string(),
        }),
    ]
}
