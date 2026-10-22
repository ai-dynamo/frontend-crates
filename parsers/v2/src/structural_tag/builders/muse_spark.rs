// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Muse Spark's ATEM arguments are raw scalars and JSON containers inside named
//! XML parameters. XGrammar's other XML styles have different delimiters. Compose
//! ordinary structural-tag nodes instead of depending on an engine-specific style.

use anyhow::{Context, bail, ensure};
use serde_json::{Map, Value};

use super::{StructuralTagBuilder, ToolCallGrammar};
use crate::structural_tag::policy::{
    ResolvedToolCallingPolicy, ToolCallingMode, resolve_tool_schema,
};
use crate::structural_tag::wire::{
    AnyTextFormat, ConstStringFormat, Format, JsonSchemaFormat, JsonSchemaStyle, OrFormat,
    RegexFormat, RepeatFormat, SequenceFormat, TagFormat,
};

pub(crate) static MUSE_SPARK: StructuralTagBuilder = StructuralTagBuilder::new(&MuseSparkGrammar);
struct MuseSparkGrammar;

const HEADER: &str = "<|start|>assistant";
const PARAM_END: &str = "</atem:parameter>";
const MARKERS: &[&str] = &["<atem:function_calls>", "<atem:invoke", "<atem:parameter"];

fn constant(value: impl Into<String>) -> Format {
    Format::ConstString(ConstStringFormat {
        value: value.into(),
    })
}

fn sequence(elements: Vec<Format>) -> Format {
    if elements.is_empty() {
        constant("")
    } else {
        Format::Sequence(SequenceFormat { elements })
    }
}

fn alternatives(elements: Vec<Format>) -> Format {
    Format::Or(OrFormat { elements })
}

fn repeat(content: Format, min: usize, max: i64) -> Format {
    Format::Repeat(RepeatFormat {
        content: Box::new(content),
        min,
        max,
    })
}

fn whitespace() -> Format {
    Format::Regex(RegexFormat {
        pattern: "[ \\t\\r\\n]*".into(),
    })
}

fn attr(value: &str) -> anyhow::Result<String> {
    ensure!(
        !value.contains(['\r', '\n']) && !value.contains("<|"),
        "invalid ATEM header name"
    );
    Ok(value
        .replace('&', "&amp;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
        .replace('<', "&lt;")
        .replace('>', "&gt;"))
}

fn raw_text(excludes: &[&str]) -> Format {
    Format::AnyText(AnyTextFormat {
        excludes: excludes.iter().map(|s| (*s).into()).collect(),
    })
}

/// Local references in the XML envelope must retain their root document scope.
/// JSON-valued arguments keep their complete schema and definitions for XGrammar.
fn dereference<'a>(mut schema: &'a Value, root: &'a Value) -> anyhow::Result<&'a Value> {
    let mut seen = std::collections::HashSet::new();
    while let Some(reference) = schema.get("$ref").and_then(Value::as_str) {
        ensure!(seen.insert(reference), "recursive ATEM envelope reference");
        let pointer = reference
            .strip_prefix('#')
            .context("ATEM envelope requires a local $ref")?;
        ensure!(
            schema
                .as_object()
                .unwrap()
                .keys()
                .all(|key| matches!(key.as_str(), "$ref" | "title" | "description")),
            "ATEM envelope $ref siblings require an explicit schema"
        );
        schema = root
            .pointer(pointer)
            .context("unresolved ATEM envelope $ref")?;
    }
    Ok(schema)
}

fn string_value(schema: &Map<String, Value>) -> anyhow::Result<Format> {
    const SUPPORTED: &[&str] = &[
        "type",
        "enum",
        "const",
        "minLength",
        "maxLength",
        "pattern",
        "title",
        "description",
        "default",
        "examples",
        "deprecated",
        "readOnly",
        "writeOnly",
        // An annotation, also used by the Muse parser to preserve freeform
        // custom-tool input instead of JSON-decoding JSON-looking strings.
        "contentMediaType",
    ];
    ensure!(
        schema.keys().all(|k| SUPPORTED.contains(&k.as_str())),
        "unsupported raw ATEM string constraint"
    );
    let length = |key| -> anyhow::Result<Option<u64>> {
        schema
            .get(key)
            .map(|v| {
                v.as_u64()
                    .context("string length must be a nonnegative integer")
            })
            .transpose()
    };
    let min = length("minLength")?.unwrap_or(0);
    let max = length("maxLength")?;
    ensure!(
        max.is_none_or(|max| max >= min),
        "inconsistent ATEM string lengths"
    );
    let pattern = schema
        .get("pattern")
        .map(|v| v.as_str().context("pattern must be a string"))
        .transpose()?;
    let regex = pattern.map(regex::Regex::new).transpose()?;
    let values = schema
        .get("enum")
        .map(|v| v.as_array().context("enum must be an array"))
        .transpose()?;
    if values.is_some() || schema.contains_key("const") {
        let constants: Vec<&Value> = values
            .map(|v| v.iter().collect())
            .unwrap_or_else(|| vec![&schema["const"]]);
        let mut formats = Vec::new();
        for value in constants {
            let Some(text) = value.as_str() else { continue };
            let len = text.chars().count() as u64;
            if len >= min
                && max.is_none_or(|max| len <= max)
                && regex.as_ref().is_none_or(|re| re.is_match(text))
                && schema.get("const").is_none_or(|c| c == value)
                && !text.contains(PARAM_END)
                && !text.contains("<|")
            {
                let rendered = if schema.get("contentMediaType").and_then(Value::as_str)
                    != Some("text/plain")
                    && matches!(serde_json::from_str::<Value>(text), Ok(Value::String(_)))
                {
                    serde_json::to_string(text)?
                } else {
                    text.to_owned()
                };
                formats.push(constant(rendered));
            }
        }
        ensure!(
            !formats.is_empty(),
            "ATEM string schema has no representable values"
        );
        return Ok(alternatives(formats));
    }
    // Unbounded raw-text regexes could consume a parameter-closing delimiter
    // as value data. XGrammar has no intersection node to combine them with
    // delimiter exclusion. Finite enums above can be checked safely instead.
    ensure!(
        pattern.is_none() && min == 0 && max.is_none(),
        "unsupported raw ATEM string pattern/length constraints without enum or const"
    );
    Ok(raw_text(&[PARAM_END, "<|"]))
}

fn value_format(schema: &Value, root: &Value, any_order: bool) -> anyhow::Result<Format> {
    // Native ATEM strings are unquoted. The Spark parser preserves their
    // declared string type, including JSON-looking values such as 42 and true.
    if schema.get("type").and_then(Value::as_str) == Some("string") {
        return string_value(schema.as_object().unwrap());
    }
    let mut scoped = schema.clone();
    if let Some(object) = scoped.as_object_mut() {
        for key in ["$defs", "definitions"] {
            if let Some(defs) = root.get(key) {
                object.entry(key).or_insert_with(|| defs.clone());
            }
        }
    }
    Ok(Format::JsonSchema(JsonSchemaFormat {
        json_schema: scoped,
        style: JsonSchemaStyle::Json,
        any_order,
    }))
}

fn parameter(name: &str, content: Format) -> anyhow::Result<Format> {
    // The native parser reads attribute names literally rather than XML-decoding
    // them. Reject names it cannot round-trip instead of changing argument keys.
    ensure!(
        !name.is_empty() && attr(name)? == name,
        "ATEM parameter names requiring XML escaping or empty names are unsupported"
    );
    Ok(sequence(vec![
        whitespace(),
        Format::Tag(TagFormat {
            begin: format!("<atem:parameter name=\"{}\">", attr(name)?),
            content: Box::new(content),
            end: PARAM_END.into(),
        }),
    ]))
}

fn arguments(schema: &Value, any_order: bool) -> anyhow::Result<Format> {
    if schema == &Value::Bool(true) {
        return Ok(repeat(
            sequence(vec![
                whitespace(),
                Format::Tag(TagFormat {
                    begin: "<atem:parameter name=\"".into(),
                    content: Box::new(sequence(vec![
                        Format::Regex(RegexFormat {
                            pattern: "[A-Za-z_][A-Za-z0-9_.-]*".into(),
                        }),
                        constant("\">"),
                        raw_text(&[PARAM_END, "<|"]),
                    ])),
                    end: PARAM_END.into(),
                }),
            ]),
            0,
            -1,
        ));
    }
    let root = schema;
    let schema = dereference(schema, root)?;
    ensure!(
        schema.is_object() && (schema.get("type").is_none() || schema["type"] == "object"),
        "ATEM tool arguments must be an object"
    );
    const SUPPORTED: &[&str] = &[
        "type",
        "properties",
        "required",
        "additionalProperties",
        "$defs",
        "definitions",
        "title",
        "description",
        "default",
        "examples",
        "deprecated",
    ];
    ensure!(
        schema
            .as_object()
            .unwrap()
            .keys()
            .all(|k| SUPPORTED.contains(&k.as_str())),
        "unsupported ATEM argument object constraint"
    );
    ensure!(
        schema
            .get("additionalProperties")
            .is_none_or(|v| v == &Value::Bool(false)),
        "ATEM constrained arguments require closed properties"
    );
    let properties = schema
        .get("properties")
        .map(|v| v.as_object().context("properties must be an object"))
        .transpose()?;
    let required = schema
        .get("required")
        .map(|v| v.as_array().context("required must be an array"))
        .transpose()?;
    let required = required
        .into_iter()
        .flatten()
        .map(|v| v.as_str().context("required names must be strings"))
        .collect::<anyhow::Result<Vec<_>>>()?;
    ensure!(
        required
            .iter()
            .all(|name| properties.is_some_and(|p| p.contains_key(*name))),
        "required ATEM property is not declared"
    );
    let mut fields = Vec::new();
    for (name, schema) in properties.into_iter().flatten() {
        let field = parameter(name, value_format(schema, root, any_order)?)?;
        fields.push((field, required.contains(&name.as_str())));
    }
    Ok(sequence(
        fields
            .into_iter()
            .map(|(f, required)| if required { f } else { repeat(f, 0, 1) })
            .collect(),
    ))
}

impl MuseSparkGrammar {
    fn call_tags(
        &self,
        policy: &ResolvedToolCallingPolicy<'_>,
        any_order: bool,
    ) -> anyhow::Result<Vec<TagFormat>> {
        let mut tags = Vec::new();
        for tool in &policy.tools {
            ensure!(
                !tool.name.is_empty()
                    && !tool.name.chars().any(|c| c.is_whitespace()
                        || c.is_control()
                        || matches!(c, '<' | '>' | '"' | '\'' | '&')),
                "invalid ATEM tool recipient"
            );
            attr(&tool.name)?;
            // Reserved recipients denote message phases, even when a tool has
            // the same name. The native generic tool channel avoids that clash.
            let recipient = if matches!(
                tool.name.as_str(),
                "self" | "user" | "commentary" | "developer"
            ) {
                "tool"
            } else {
                &tool.name
            };
            let invoke = format!("<atem:invoke name=\"{}\">", attr(&tool.name)?);
            let schema = if tool.parameters.is_null()
                || tool.parameters.as_object().is_some_and(Map::is_empty)
            {
                serde_json::json!({"type":"object", "properties":{}, "additionalProperties":false})
            } else {
                resolve_tool_schema(tool, policy.schema_mode)
            };
            let args = arguments(&schema, any_order)?;
            for end in ["<|eom|>", ""] {
                let mut content = vec![whitespace(), constant(&invoke)];
                content.extend([
                    args.clone(),
                    whitespace(),
                    constant("</atem:invoke>"),
                    whitespace(),
                ]);
                tags.push(TagFormat {
                    begin: format!(" to={recipient}<|message|><atem:function_calls>"),
                    content: Box::new(sequence(content)),
                    end: format!("</atem:function_calls>{end}"),
                });
            }
        }
        Ok(tags)
    }

    fn calls(
        &self,
        policy: &ResolvedToolCallingPolicy<'_>,
        any_order: bool,
    ) -> anyhow::Result<Format> {
        // The invoke name is fixed, including for generic tool-channel recipients.
        let tags = self.call_tags(policy, any_order)?;
        // The checkpoint treats <|eot|> as EOS. Requiring it as literal grammar
        // text masks the stop token until an impossible extra token is produced.
        // Complete the native payload first, then let the engine accept EOS.
        let (continued, final_calls): (Vec<_>, Vec<_>) = tags
            .into_iter()
            .partition(|tag| tag.end.ends_with("<|eom|>"));
        let finished = sequence(vec![
            alternatives(final_calls.into_iter().map(Format::Tag).collect()),
            repeat(constant("<|eom|>"), 0, 1),
        ]);
        if policy.stop_after_first() {
            return Ok(finished);
        }
        Ok(sequence(vec![
            repeat(
                sequence(vec![
                    alternatives(continued.into_iter().map(Format::Tag).collect()),
                    constant(HEADER),
                ]),
                0,
                -1,
            ),
            finished,
        ]))
    }

    fn final_answer(&self, content: Format) -> Format {
        sequence(vec![constant(" to=user<|message|>"), content])
    }

    fn structured_answer(&self, schema: &Value) -> Format {
        self.final_answer(Format::JsonSchema(JsonSchemaFormat {
            json_schema: schema.clone(),
            style: JsonSchemaStyle::Json,
            any_order: false,
        }))
    }

    fn with_progress(&self, output: Format) -> Format {
        let progress = |recipient| {
            sequence(vec![
                Format::Tag(TagFormat {
                    begin: format!(" to={recipient}<|message|>"),
                    content: Box::new(raw_text(&["<|"])),
                    end: "<|eom|>".into(),
                }),
                constant(HEADER),
            ])
        };
        sequence(vec![
            repeat(progress("self"), 0, 1),
            repeat(progress("commentary"), 0, 1),
            output,
        ])
    }
}

impl ToolCallGrammar for MuseSparkGrammar {
    fn validate_options(
        &self,
        options: &crate::structural_tag::StructuralTagOptions,
    ) -> anyhow::Result<()> {
        ensure!(
            !options.tool_arguments_any_order,
            "Muse Spark native ATEM does not support tool_arguments_any_order; required fields must retain their declared order"
        );
        ensure!(
            options.reasoning_boundary != crate::structural_tag::ReasoningBoundary::External,
            "Muse Spark structural tags require grammar-owned reasoning; an engine gate that waits for a tool channel cannot enforce forced tool choice"
        );
        Ok(())
    }

    fn build_without_tools(&self) -> Option<Format> {
        Some(self.with_progress(self.final_answer(raw_text(&["<|"]))))
    }

    fn build_structured_output(&self, schema: &Value) -> Format {
        // Schema-only answers use the same native envelope as answers with tools.
        self.with_progress(self.structured_answer(schema))
    }

    fn reasoning_begin(&self) -> Option<&'static str> {
        Some(" to=self<|message|>")
    }
    fn reasoning_end(&self) -> Option<&'static str> {
        Some("<|eom|>")
    }
    fn reasoning_suffix(&self) -> &'static str {
        HEADER
    }
    fn tool_call_excludes(&self) -> &'static [&'static str] {
        MARKERS
    }

    fn build_triggered_calls(
        &self,
        policy: &ResolvedToolCallingPolicy<'_>,
        _exclude: bool,
        any_order: bool,
    ) -> anyhow::Result<Format> {
        let calls = self.calls(policy, any_order)?;
        let output = match policy.mode {
            ToolCallingMode::Auto => {
                alternatives(vec![calls, self.final_answer(raw_text(&["<|"]))])
            }
            ToolCallingMode::Required => calls,
            ToolCallingMode::Named => bail!("auto or required policy expected"),
        };
        Ok(self.with_progress(output))
    }

    fn build_tool_calls_only(
        &self,
        policy: &ResolvedToolCallingPolicy<'_>,
        any_order: bool,
    ) -> anyhow::Result<Format> {
        Ok(self.with_progress(self.calls(policy, any_order)?))
    }

    fn build_auto_with_structured_output(
        &self,
        policy: &ResolvedToolCallingPolicy<'_>,
        schema: &Value,
        any_order: bool,
    ) -> anyhow::Result<Format> {
        Ok(self.with_progress(alternatives(vec![
            self.calls(policy, any_order)?,
            self.structured_answer(schema),
        ])))
    }
}
