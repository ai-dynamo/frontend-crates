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
    fn payload_digest(&self) -> storage::Result<String> {
        fn portable(value: &Node) -> bool {
            match value {
                Node::Null | Node::Bool(_) | Node::Int(_) | Node::Str(_) => true,
                Node::Array(items) => items.iter().all(portable),
                Node::Map(entries) => entries
                    .iter()
                    .all(|(key, value)| matches!(key, Node::Str(_)) && portable(value)),
                _ => false,
            }
        }
        if !matches!(self, Node::Map(_)) || !portable(self) {
            return Err("result references require integer-only JSON payloads".into());
        }
        self.digest()
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
    let contexts = storage::contexts(document).map_err(|e| anyhow!(e))?;
    for (id, stored) in document.field("records")?.mapping()? {
        let mut value = Node::map();
        value.set(
            "document",
            storage::metadata(document, stored, &contexts).map_err(|e| anyhow!(e))?,
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

    #[test]
    fn archived_result_references_preserve_complete_payloads() -> Result<()> {
        let fixture = crate::yaml::load_str(include_str!(
            "../../utils/tests/fixtures/normalized-results.yaml"
        ))?;
        assert!(
            snapshot(fixture.field("snapshot")?, fixture.field("authored")?)?
                .equivalent(fixture.field("expected_snapshot")?)
        );
        let expanded = snapshot(fixture.field("snapshot")?, fixture.field("authored")?)?;
        let receipts = |document: &Node| -> Result<Node> {
            Ok(document
                .field("records")?
                .field("one")?
                .field("value")?
                .field("document")?
                .field("record_metadata")?
                .field("verified_recaptures")?
                .clone())
        };
        assert_eq!(
            receipts(&expanded)?,
            receipts(fixture.field("expected_snapshot")?)?
        );
        assert!(
            summary_records(fixture.field("snapshot")?)?
                .field("one")?
                .field("value")?
                .field("document")?
                .equivalent(
                    fixture
                        .field("expected_snapshot")?
                        .field("records")?
                        .field("one")?
                        .field("value")?
                        .field("document")?
                )
        );
        let mut invalid = fixture.field("snapshot")?.clone();
        let mut contexts = invalid.field("contexts")?.clone();
        let mut context = contexts.field("shared")?.clone();
        let mut metadata = context.field("record_metadata")?.clone();
        let mut archives = metadata.field("verified_recaptures")?.array()?.to_vec();
        archives[0].set("result_ref", Node::Str("0".repeat(64)))?;
        metadata.set("verified_recaptures", Node::Array(archives))?;
        context.set("record_metadata", metadata)?;
        contexts.set("shared", context)?;
        invalid.set("contexts", contexts)?;
        assert!(snapshot(&invalid, fixture.field("authored")?).is_err());
        let mut unused = fixture.field("snapshot")?.clone();
        let mut contexts = unused.field("contexts")?.clone();
        contexts.set(
            "unused",
            crate::yaml::load_str("record_metadata:\n  verified_recaptures:\n  - result_ref: bad")?,
        )?;
        unused.set("contexts", contexts)?;
        assert!(snapshot(&unused, fixture.field("authored")?).is_err());
        assert!(summary_records(&unused).is_err());
        for payload in [
            "value: 1.0",
            "value: 1000000000000000.25",
            "value: .inf",
            "1: typed key",
            "value: 2026-01-01",
        ] {
            let mut invalid = fixture.field("snapshot")?.clone();
            let mut contexts = invalid.field("contexts")?.clone();
            let mut context = contexts.field("shared")?.clone();
            let mut metadata = context.field("record_metadata")?.clone();
            let mut active = metadata.field("verified_recapture")?.clone();
            let payload = crate::yaml::load_str(payload)?;
            // A matching ordinary digest must not bypass the portable-type check.
            assert!(storage::Value::payload_digest(&payload).is_err());
            let digest = source::request_digest(&payload).unwrap_or_else(|_| "0".repeat(64));
            active.set("result", payload)?;
            let mut archives = metadata.field("verified_recaptures")?.array()?.to_vec();
            for archive in &mut archives {
                archive.set("result_ref", Node::Str(digest.clone()))?;
            }
            metadata.set("verified_recapture", active)?;
            metadata.set("verified_recaptures", Node::Array(archives))?;
            context.set("record_metadata", metadata)?;
            contexts.set("shared", context)?;
            invalid.set("contexts", contexts)?;
            assert!(snapshot(&invalid, fixture.field("authored")?).is_err());
            assert!(summary_records(&invalid).is_err());
        }
        Ok(())
    }
}
