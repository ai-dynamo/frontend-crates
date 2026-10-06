// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Private serialization model for the xgrammar formats used by v2 builders.

use serde::ser::SerializeMap;
use serde::{Serialize, Serializer};
use serde_json::Value;

#[derive(Debug, Clone)]
pub(crate) struct StructuralTag {
    pub format: Format,
}

impl Serialize for StructuralTag {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(2))?;
        map.serialize_entry("type", "structural_tag")?;
        map.serialize_entry("format", &self.format)?;
        map.end()
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum Format {
    ConstString(ConstStringFormat),
    Regex(dynamo_structural_tag::format::RegexFormat),
    Optional(NestedFormat),
    Star(NestedFormat),
    Tag(TagFormat),
    TriggeredTags(TriggeredTagsFormat),
    TagsWithSeparator(TagsWithSeparatorFormat),
    Sequence(SequenceFormat),
    JsonSchema(JsonSchemaFormat),
    AnyText(AnyTextFormat),
    AnyTokens(dynamo_structural_tag::format::AnyTokensFormat),
    Or(OrFormat),
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct ConstStringFormat {
    pub value: String,
}

#[derive(Debug, Clone)]
pub(crate) struct TagFormat {
    pub begin: String,
    pub content: Box<Format>,
    pub end: String,
}

impl Serialize for TagFormat {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(4))?;
        map.serialize_entry("type", "tag")?;
        map.serialize_entry("begin", &self.begin)?;
        map.serialize_entry("content", &self.content)?;
        map.serialize_entry("end", &self.end)?;
        map.end()
    }
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct TriggeredTagsFormat {
    pub triggers: Vec<String>,
    pub tags: Vec<TagFormat>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub excludes: Vec<String>,
    pub at_least_one: bool,
    pub stop_after_first: bool,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct TagsWithSeparatorFormat {
    pub tags: Vec<TagFormat>,
    pub separator: String,
    pub at_least_one: bool,
    pub stop_after_first: bool,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct SequenceFormat {
    pub elements: Vec<Format>,
}

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum JsonSchemaStyle {
    Json,
    QwenXml,
    DeepseekXml,
    GlmXml,
    MinimaxXml,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct JsonSchemaFormat {
    pub json_schema: Value,
    pub style: JsonSchemaStyle,
    #[serde(skip_serializing_if = "is_false")]
    pub any_order: bool,
}

fn is_false(value: &bool) -> bool {
    !value
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct AnyTextFormat {
    pub excludes: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct OrFormat {
    pub elements: Vec<Format>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct NestedFormat {
    pub content: Box<Format>,
}

/// Import a shared native grammar and apply v2's argument-order option.
/// This is a typed translation, so new shared nodes require an explicit decision.
pub(super) fn import_native(
    format: dynamo_structural_tag::format::Format,
    any_order: bool,
) -> Format {
    use dynamo_structural_tag::format as native;
    match format {
        native::Format::Tag(tag) => Format::Tag(import_tag(tag, any_order)),
        native::Format::TriggeredTags(f) => Format::TriggeredTags(TriggeredTagsFormat {
            triggers: f.triggers,
            tags: f
                .tags
                .into_iter()
                .map(|t| import_tag(t, any_order))
                .collect(),
            excludes: vec![],
            at_least_one: f.at_least_one,
            stop_after_first: f.stop_after_first,
        }),
        native::Format::TagsWithSeparator(f) => {
            Format::TagsWithSeparator(TagsWithSeparatorFormat {
                tags: f
                    .tags
                    .into_iter()
                    .map(|t| import_tag(t, any_order))
                    .collect(),
                separator: f.separator,
                at_least_one: f.at_least_one,
                stop_after_first: f.stop_after_first,
            })
        }
        native::Format::Sequence(f) => Format::Sequence(SequenceFormat {
            elements: f
                .elements
                .into_iter()
                .map(|f| import_native(f, any_order))
                .collect(),
        }),
        native::Format::JsonSchema(f) => Format::JsonSchema(JsonSchemaFormat {
            json_schema: f.json_schema,
            style: match f.style {
                native::JsonSchemaStyle::Json => JsonSchemaStyle::Json,
                native::JsonSchemaStyle::QwenXml => JsonSchemaStyle::QwenXml,
                native::JsonSchemaStyle::MinimaxXml => JsonSchemaStyle::MinimaxXml,
                native::JsonSchemaStyle::DeepseekXml => JsonSchemaStyle::DeepseekXml,
            },
            any_order,
        }),
        native::Format::AnyTokens(f) => Format::AnyTokens(f),
        native::Format::AnyText(f) => Format::AnyText(AnyTextFormat {
            excludes: f.excludes,
        }),
        native::Format::ConstString(f) => Format::ConstString(ConstStringFormat { value: f.value }),
        native::Format::Regex(f) => Format::Regex(f),
        native::Format::Optional(f) => Format::Optional(NestedFormat {
            content: Box::new(import_native(*f.content, any_order)),
        }),
        native::Format::Star(f) => Format::Star(NestedFormat {
            content: Box::new(import_native(*f.content, any_order)),
        }),
        native::Format::Or(f) => Format::Or(OrFormat {
            elements: f
                .elements
                .into_iter()
                .map(|f| import_native(f, any_order))
                .collect(),
        }),
    }
}

pub(super) fn import_tag(
    tag: dynamo_structural_tag::format::TagFormat,
    any_order: bool,
) -> TagFormat {
    TagFormat {
        begin: tag.begin,
        content: Box::new(import_native(*tag.content, any_order)),
        end: tag.end,
    }
}
