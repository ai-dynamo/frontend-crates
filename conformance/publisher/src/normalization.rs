// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::{source, yaml::Node};
use anyhow::{Result, anyhow};

#[path = "../../src/normalization.rs"]
mod storage;

impl storage::Value for Node {
    fn object() -> Self {
        Node::map()
    }
    fn text(value: &str) -> Self {
        Node::Str(value.into())
    }
    fn integer(value: u64) -> Self {
        Node::Int(value.into())
    }
    fn array(values: Vec<Self>) -> Self {
        Node::Array(values)
    }
    fn null() -> Self {
        Node::Null
    }
    fn is_null(&self) -> bool {
        matches!(self, Node::Null)
    }
    fn as_bool(&self) -> Option<bool> {
        if let Node::Bool(value) = self {
            Some(*value)
        } else {
            None
        }
    }
    fn as_integer(&self) -> Option<u64> {
        if let Node::Int(value) = self {
            value.to_string().parse().ok()
        } else {
            None
        }
    }
    fn as_text(&self) -> storage::Result<&str> {
        self.string().map_err(|e| e.to_string())
    }
    fn entries(&self) -> storage::Result<Vec<(&str, &Self)>> {
        self.mapping()
            .map_err(|e| e.to_string())?
            .iter()
            .map(|(key, value)| Ok((key.string().map_err(|e| e.to_string())?, value)))
            .collect()
    }
    fn items(&self) -> storage::Result<&[Self]> {
        self.array().map_err(|e| e.to_string())
    }
    fn get(&self, name: &str) -> Option<&Self> {
        Node::get(self, name)
    }
    fn set(&mut self, name: &str, value: Self) -> storage::Result<()> {
        Node::set(self, name, value).map_err(|e| e.to_string())
    }
    fn digest(&self) -> storage::Result<String> {
        source::request_digest(self).map_err(|e| e.to_string())
    }
}

pub fn family(document: &Node) -> Result<Node> {
    storage::family(document).map_err(|e| anyhow!(e))
}
pub fn snapshot(document: &Node, authored: &Node) -> Result<Node> {
    storage::snapshot(document, authored).map_err(|e| anyhow!(e))
}
pub fn is_normalized(document: &Node) -> bool {
    storage::is_normalized(document)
}

pub fn summary_records(document: &Node) -> Result<Node> {
    let mut result = Node::map();
    for (id, stored) in document.field("records")?.mapping()? {
        let mut value = Node::map();
        value.set(
            "document",
            storage::metadata(document, stored).map_err(|e| anyhow!(e))?,
        )?;
        let mut observation = Node::map();
        for (source, target) in [
            ("output", "value"),
            ("error", "error"),
            ("unavailable", "unavailable"),
        ] {
            if let Some(payload) = stored.get(source) {
                observation.set(target, payload.clone())?;
            }
        }
        value.set("observation", observation)?;
        let mut wrapped = Node::map();
        wrapped.set("value", value)?;
        result.set(id.string()?, wrapped)?;
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalized_yaml_preserves_typed_fields_and_literal_schema_refs() -> Result<()> {
        let fixture =
            crate::yaml::load_str(include_str!("../../utils/tests/fixtures/normalized.yaml"))?;
        for variant in fixture.field("variants")?.array()? {
            assert!(family(variant.field("authored")?)?.equivalent(variant.field("expected")?));
        }
        let authored = fixture.field("authored")?;
        let snapshot_source = fixture.field("snapshot")?;
        assert!(family(authored)?.equivalent(fixture.field("expected_family")?));
        assert!(
            snapshot(snapshot_source, authored)?.equivalent(fixture.field("expected_snapshot")?)
        );
        let mut malformed = authored.clone();
        malformed.set("tools", Node::map())?;
        assert!(family(&malformed).is_err());
        let mut malformed = snapshot_source.clone();
        malformed.set("producers", Node::map())?;
        assert!(snapshot(&malformed, authored).is_err());
        let mut malformed = snapshot_source.clone();
        let mut records = malformed.field("records")?.clone();
        let mut record = records.field("one")?.clone();
        record.set("context", Node::Null)?;
        records.set("one", record)?;
        malformed.set("records", records)?;
        assert!(snapshot(&malformed, authored).is_err());
        Ok(())
    }
}
