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
    Tag(TagFormat),
    TriggeredTags(TriggeredTagsFormat),
    TagsWithSeparator(TagsWithSeparatorFormat),
    Sequence(SequenceFormat),
    JsonSchema(JsonSchemaFormat),
    AnyText(AnyTextFormat),
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
