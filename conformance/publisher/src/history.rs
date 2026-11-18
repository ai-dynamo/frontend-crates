// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Ordered sparse history resolution. Metadata is owned by the observation and
//! checkpoint that introduced it, rather than by the selected display version.

use crate::yaml::Node;
use anyhow::{Context, Result, ensure};
use num_bigint::BigInt;
use regex::Regex;
use std::{borrow::Cow, collections::BTreeSet, sync::LazyLock};

/// Restrict shared YAML references to one named file in its owning directory.
pub fn source_reference<'a>(reference: &'a str, directory: &str) -> Result<&'a str> {
    if crate::source::hex(reference, 64) {
        return Ok(reference);
    }
    let name = reference
        .strip_prefix(directory)
        .and_then(|s| s.strip_prefix('/'));
    ensure!(
        name.is_some_and(|name| name.ends_with(".yaml")
            && name
                .as_bytes()
                .first()
                .is_some_and(u8::is_ascii_alphanumeric)
            && name
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"_.+-".contains(&b))),
        "invalid {directory} filename reference: {reference}"
    );
    Ok(reference)
}

pub fn source_path(reference: &str, directory: &str) -> Result<String> {
    source_reference(reference, directory)?;
    Ok(if crate::source::hex(reference, 64) {
        format!("{directory}/{reference}.yaml")
    } else {
        reference.into()
    })
}

static VERSION: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^[0-9]+(?:\.[0-9]+){2,}(?:\.post[0-9]+|-[0-9A-Za-z.-]+)?$").unwrap()
});

/// Matches the maintained reader's numeric tuple, prerelease, stable, post order.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct Release(Vec<BigInt>, u8, String, BigInt);

impl Release {
    /// Frozen record provenance may retain a source-qualified identity.
    pub fn parse_provenance(version: &str) -> Result<Self> {
        let release = if let Some((release, source)) = version.split_once("+source.") {
            ensure!(
                source.len() == 64
                    && source
                        .bytes()
                        .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
                "invalid provenance source identity"
            );
            release
        } else {
            version
        };
        Self::parse(release)
    }

    /// Parse a capture version without imposing SemVer's three-component limit.
    pub fn parse(version: &str) -> Result<Self> {
        ensure!(
            VERSION.is_match(version),
            "invalid Unified capture version: {version}"
        );
        let (release, prerelease) = version
            .split_once('-')
            .map_or((version, None), |(a, b)| (a, Some(b)));
        let (release, post) = release
            .split_once(".post")
            .map_or((release, None), |(a, b)| (a, Some(b)));
        let numeric = release
            .split('.')
            .map(str::parse)
            .collect::<std::result::Result<Vec<BigInt>, _>>()?;
        Ok(Self(
            numeric,
            if post.is_some() {
                2
            } else if prerelease.is_some() {
                0
            } else {
                1
            },
            prerelease.unwrap_or("").into(),
            post.unwrap_or("0").parse()?,
        ))
    }
}

/// One fully expanded, validated checkpoint.
#[derive(Clone, Debug)]
pub struct Capture {
    pub id: String,
    pub version: String,
    pub member: String,
    pub document: Node,
    /// Complete checkpoint payload when reading the versioned snapshot format.
    pub resolved_records: Option<Node>,
}

/// A single family/implementation history, in release order.
pub struct History {
    pub implementation: String,
    pub captures: Vec<Capture>,
}

pub(crate) fn truthy(node: &Node) -> bool {
    match node {
        Node::Map(v) => !v.is_empty(),
        Node::Array(v) => !v.is_empty(),
        Node::Str(v) => !v.is_empty(),
        Node::Null | Node::Bool(false) => false,
        _ => true,
    }
}

/// Require exact top-level fields with optional additions.
pub fn fields(node: &Node, required: &[&str], optional: &[&str]) -> Result<()> {
    for key in required {
        node.field(key)?;
    }
    for (key, _) in node.mapping()? {
        let key = key.string()?;
        ensure!(
            required.contains(&key) || optional.contains(&key),
            "unknown field {key}"
        );
    }
    Ok(())
}

/// Resolves state in-place, retaining insertion order even when changes update
/// an existing observation. Removing then reinserting a case moves it to the end.
#[derive(Default)]
pub struct Cursor {
    state: Option<Node>,
    current: Option<usize>,
}

impl History {
    /// Validate parent linkage and every transition before exposing snapshots.
    pub fn validate(&mut self) -> Result<()> {
        let mut ordered = self
            .captures
            .drain(..)
            .map(|c| Ok((Release::parse(&c.version)?, c)))
            .collect::<Result<Vec<_>>>()?;
        ordered.sort_by(|a, b| a.0.cmp(&b.0));
        self.captures = ordered.into_iter().map(|(_, c)| c).collect();
        ensure!(!self.captures.is_empty(), "history has no captures");
        let mut previous = None;
        let mut cursor = Cursor::default();
        for (index, capture) in self.captures.iter().enumerate() {
            if let Some(parent) = capture.document.get("parent") {
                ensure!(
                    Some(parent.string()?) == previous,
                    "capture parent is not the preceding checkpoint: {}",
                    capture.member
                );
            }
            previous = Some(capture.id.as_str());
            cursor
                .apply(self, index)
                .with_context(|| capture.member.clone())?;
        }
        Ok(())
    }
}

impl Cursor {
    /// Advance exactly once through an ordered chain.
    pub fn apply(&mut self, history: &History, index: usize) -> Result<()> {
        ensure!(
            self.current.map_or(0, |i| i + 1) == index,
            "history cursor advanced out of order"
        );
        let capture = &history.captures[index];
        if let Some(records) = &capture.resolved_records {
            self.state = Some(records.clone());
            self.current = Some(index);
            return Ok(());
        }
        let state = self.state.get_or_insert_with(Node::map);
        for (case_id, change) in capture.document.field("changes")?.mapping()? {
            let case_id = case_id.string()?;
            if change.mapping()?.len() == 1 && change.get("absent") == Some(&Node::Bool(true)) {
                state.remove(case_id)?;
                continue;
            }
            let previous = state.get(case_id);
            let inherited_metadata = previous.and_then(|v| v.get("_record_metadata")).cloned();
            let inherited_parser = previous
                .and_then(|v| v.get("_document_overrides"))
                .and_then(|v| v.get("parser_path"))
                .cloned();
            let mut record = match previous {
                Some(previous) => {
                    let mut record = previous.clone();
                    record.update(change)?;
                    if change.get("observation").is_some() {
                        record.remove("_document_overrides")?;
                        record.set("_origin_capture_id", Node::Str(capture.id.clone()))?;
                    }
                    record
                }
                None => {
                    fields(change, &["case_key", "stimulus", "observation"], &[])?;
                    let mut record = change.clone();
                    record.set("_origin_capture_id", Node::Str(capture.id.clone()))?;
                    record
                }
            };
            if let Some(metadata) = inherited_metadata {
                record.set("_record_metadata", metadata)?;
            }
            if let Some(parser) = inherited_parser {
                let mut overrides = record
                    .remove("_document_overrides")?
                    .unwrap_or_else(Node::map);
                overrides.set("parser_path", parser)?;
                record.set("_document_overrides", overrides)?;
            }
            state.set(case_id, record)?;
        }
        for (case_id, metadata) in capture.document.field("metadata_changes")?.mapping()? {
            let case_id = case_id.string()?;
            let mut record = state
                .field(case_id)
                .context("metadata change has no observation")?
                .clone();
            if truthy(metadata) {
                record.set("_record_metadata", metadata.clone())?;
                record.set("_record_metadata_capture_id", Node::Str(capture.id.clone()))?;
            } else {
                record.remove("_record_metadata")?;
                record.remove("_record_metadata_capture_id")?;
            }
            state.set(case_id, record)?;
        }
        for (case_id, overrides) in capture.document.field("document_overrides")?.mapping()? {
            let case_id = case_id.string()?;
            let mut record = state
                .field(case_id)
                .context("document override has no observation")?
                .clone();
            let mut existing = record
                .remove("_document_overrides")?
                .unwrap_or_else(Node::map);
            existing.update(overrides)?;
            record.set("_document_overrides", existing)?;
            state.set(case_id, record)?;
        }
        self.current = Some(index);
        Ok(())
    }

    /// Produce the exact ordered record/member envelopes for a selected version.
    pub fn records(&self, history: &History, version: &str) -> Result<Node> {
        let current =
            &history.captures[self.current.context("history cursor has no checkpoint")?];
        if let Some(records) = &current.resolved_records {
            return Ok(selected_snapshot(records, &current.version, version)?.into_owned());
        }
        let mut result = Node::map();
        for (case_id, change) in self
            .state
            .as_ref()
            .context("history cursor has no state")?
            .mapping()?
        {
            let case_id = case_id.string()?;
            let mut record = change.clone();
            let mut metadata = record.remove("_record_metadata")?;
            let metadata_capture = record.remove("_record_metadata_capture_id")?;
            if (metadata_capture.as_ref().map(Node::string).transpose()?
                != Some(current.id.as_str())
                || current.id != format!("{}-{version}", history.implementation))
                && let Some(metadata) = &mut metadata
            {
                metadata.remove("verified_recapture")?;
            }
            let mut overrides = record
                .remove("_document_overrides")?
                .unwrap_or_else(Node::map);
            let suppress_inherited_from = overrides
                .get("capture_provenance")
                .and_then(|provenance| provenance.get("inherited_from"))
                == Some(&Node::Null);
            let origin_id = record
                .remove("_origin_capture_id")?
                .context("observation has no origin")?;
            let origin_id = origin_id.string()?;
            let origin = history
                .captures
                .iter()
                .find(|c| c.id == origin_id)
                .context("unknown observation origin")?;
            let mut document = origin.document.field("document")?.clone();
            let mut captured_with = Node::map();
            captured_with.set(&history.implementation, Node::Str(origin.version.clone()))?;
            document.set("captured_with", captured_with)?;
            let provenance = origin.document.field("provenance")?;
            if provenance.field("status")?.string()? == "captured" {
                document.set("capture_origin", provenance.field("origin")?.clone())?;
            }
            if origin.version != version {
                document.set("inherited_from", Node::Str(origin.version.clone()))?;
            }
            if let Some(provenance) = overrides.remove("capture_provenance")? {
                for (key, value) in provenance.mapping()? {
                    let key = key.string()?;
                    if *value == Node::Null {
                        document.remove(key)?;
                    } else {
                        document.set(key, value.clone())?;
                    }
                }
            }
            if overrides.get("parser_path") == Some(&Node::Null) {
                overrides.remove("parser_path")?;
                document.remove("parser_path")?;
            }
            document.update(&overrides)?;
            if let Some(metadata) = metadata.filter(truthy) {
                document.set("record_metadata", metadata)?;
            }
            record.set("document", document)?;
            let mut wrapped = Node::map();
            wrapped.set("value", record)?;
            wrapped.set("member", Node::Str(origin.member.clone()))?;
            if suppress_inherited_from {
                wrapped.set("suppress_inherited_from", Node::Bool(true))?;
            }
            result.set(case_id, wrapped)?;
        }
        Ok(result)
    }
}

/// Project checkpoint-scoped execution metadata onto a displayed release.
/// Permanent snapshots retain all evidence; a later selection is not a re-execution.
pub fn selected_snapshot<'a>(
    records: &'a Node,
    checkpoint: &str,
    selected: &str,
) -> Result<Cow<'a, Node>> {
    if checkpoint == selected {
        return Ok(Cow::Borrowed(records));
    }
    let mut result = records.clone();
    let Node::Map(rows) = &mut result else {
        anyhow::bail!("expected snapshot records")
    };
    for (_, wrapped) in rows {
        let mut value = wrapped.field("value")?.clone();
        let mut document = value.field("document")?.clone();
        if let Some(metadata) = document.get("record_metadata") {
            let mut metadata = metadata.clone();
            metadata.remove("verified_recapture")?;
            if truthy(&metadata) {
                document.set("record_metadata", metadata)?;
            } else {
                document.remove("record_metadata")?;
            }
        }
        if document.get("inherited_from").is_none()
            && wrapped.get("suppress_inherited_from") != Some(&Node::Bool(true))
        {
            document.set("inherited_from", Node::Str(checkpoint.into()))?;
        }
        value.set("document", document)?;
        wrapped.set("value", value)?;
    }
    Ok(Cow::Owned(result))
}

/// Validate execution membership and expand only the designated receipt slots.
pub fn expand_execution(
    manifest: &Node,
    family: &str,
    implementation: &str,
    version: &str,
    receipts: &std::collections::BTreeMap<String, Node>,
) -> Result<Node> {
    expand_execution_bounded(
        manifest,
        family,
        implementation,
        version,
        receipts,
        u64::MAX,
    )
}

pub fn expand_execution_bounded(
    manifest: &Node,
    family: &str,
    implementation: &str,
    version: &str,
    receipts: &std::collections::BTreeMap<String, Node>,
    maximum_bytes: u64,
) -> Result<Node> {
    fields(
        manifest,
        &[
            "schema_version",
            "family",
            "implementation",
            "version",
            "checkpoint",
            "cases",
            "defaults",
            "exceptions",
        ],
        &["receipts", "proof"],
    )?;
    ensure!(
        manifest.field("schema_version")? == &Node::Int(1.into()),
        "invalid execution schema"
    );
    for (key, expected) in [
        ("family", family),
        ("implementation", implementation),
        ("version", version),
    ] {
        ensure!(
            manifest.field(key)?.string()? == expected,
            "execution manifest producer identity mismatch"
        );
    }
    let checkpoint = manifest.field("checkpoint")?;
    fields(
        checkpoint,
        &["provenance", "document", "document_overrides"],
        &[],
    )?;
    let cases = manifest
        .field("cases")?
        .array()?
        .iter()
        .map(Node::string)
        .collect::<Result<Vec<_>>>()?;
    let unique: BTreeSet<_> = cases.iter().copied().collect();
    ensure!(
        unique.len() == cases.len(),
        "duplicate execution case membership"
    );
    ensure!(
        !cases.is_empty() || manifest.get("proof").is_some(),
        "empty execution manifest"
    );
    let defaults = manifest.field("defaults")?;
    defaults.mapping()?;
    ensure!(
        !cases.is_empty() || defaults.mapping()?.is_empty(),
        "execution defaults require case membership"
    );
    let exceptions = manifest.field("exceptions")?;
    for (case, value) in exceptions.mapping()? {
        ensure!(
            unique.contains(case.string()?),
            "execution exception outside case membership"
        );
        value.mapping()?;
    }
    let mut observed = std::collections::BTreeMap::<String, BTreeSet<String>>::new();
    let mut metadata = Node::map();
    let mut expanded_bytes = 0u64;
    for case in cases {
        let mut value = defaults.merge(exceptions.get(case).unwrap_or(&Node::map()))?;
        expand_slots(&mut value, case, receipts, &mut observed)?;
        expanded_bytes = expanded_bytes
            .checked_add(value.heap_bytes())
            .context("execution expansion size overflow")?;
        ensure!(
            expanded_bytes <= maximum_bytes,
            "execution expansion exceeds its memory reservation"
        );
        metadata.set(case, value)?;
    }
    let mut result = checkpoint.clone();
    result.set("metadata_changes", metadata)?;
    if let Some(proof) = manifest.get("proof") {
        fields(proof, &["receipt", "requests"], &[])?;
        let reference = proof.field("receipt")?.string()?;
        let digest = source_reference(receipt_key(reference), "receipts")?;
        let receipt = receipts
            .get(digest)
            .context("missing execution proof receipt body")?;
        let members = proof
            .field("requests")?
            .mapping()?
            .iter()
            .map(|(k, _)| Ok(k.string()?.to_string()))
            .collect::<Result<BTreeSet<_>>>()?;
        observed.entry(digest.into()).or_default().extend(members);
        let mut proof = proof.clone();
        proof.set("receipt", receipt.clone())?;
        result.set("execution_proof", proof)?;
    }
    let empty = Node::map();
    let memberships = manifest.get("receipts").unwrap_or(&empty);
    for (digest, members) in memberships.mapping()? {
        let digest = digest.string()?;
        source_reference(digest, "receipts")?;
        let members = members
            .array()?
            .iter()
            .map(|n| Ok(n.string()?.to_string()))
            .collect::<Result<Vec<_>>>()?;
        let unique: BTreeSet<_> = members.iter().cloned().collect();
        ensure!(
            !members.is_empty()
                && unique.len() == members.len()
                && observed.get(digest) == Some(&unique),
            "execution receipt case membership mismatch"
        );
        observed.remove(digest);
    }
    ensure!(
        observed.is_empty(),
        "execution receipt membership is missing"
    );
    Ok(result)
}

fn expand_slots(
    metadata: &mut Node,
    case: &str,
    receipts: &std::collections::BTreeMap<String, Node>,
    observed: &mut std::collections::BTreeMap<String, BTreeSet<String>>,
) -> Result<()> {
    let keys = metadata
        .mapping()?
        .iter()
        .filter_map(|(k, _)| {
            if let Node::Str(s) = k {
                Some(s.clone())
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    for key in keys {
        if key == "capture_provenance" || key.ends_with("_capture_provenance") {
            expand_slot(metadata, &key, case, receipts, observed)?;
        }
    }
    if let Some(mut owner) = metadata.get("verified_recapture").cloned() {
        if matches!(owner, Node::Map(_)) && owner.get("provenance").is_some() {
            expand_slot(&mut owner, "provenance", case, receipts, observed)?;
        }
        metadata.set("verified_recapture", owner)?;
    }
    if let Some(Node::Array(mut owners)) = metadata.get("verified_recaptures").cloned() {
        for owner in &mut owners {
            if matches!(owner, Node::Map(_)) && owner.get("provenance").is_some() {
                expand_slot(owner, "provenance", case, receipts, observed)?;
            }
        }
        metadata.set("verified_recaptures", Node::Array(owners))?;
    }
    Ok(())
}

fn receipt_key(reference: &str) -> &str {
    reference
        .strip_prefix("receipt:")
        .filter(|digest| crate::source::hex(digest, 64))
        .unwrap_or(reference)
}

fn expand_slot(
    owner: &mut Node,
    key: &str,
    case: &str,
    receipts: &std::collections::BTreeMap<String, Node>,
    observed: &mut std::collections::BTreeMap<String, BTreeSet<String>>,
) -> Result<()> {
    if let Node::Str(reference) = owner.field(key)?
        && (reference.starts_with("receipts/") || receipt_key(reference) != reference)
    {
        let digest = source_reference(receipt_key(reference), "receipts")?;
        let receipt = receipts
            .get(digest)
            .context("execution receipt body is missing")?;
        observed
            .entry(digest.into())
            .or_default()
            .insert(case.into());
        owner.set(key, receipt.clone())?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::yaml::load_str;

    #[test]
    fn selected_complete_snapshot_preserves_provenance_overrides() -> Result<()> {
        let records = load_str(
            r#"
plain: {value: {document: {record_metadata: {verified_recapture: {version: 1.0.0}, kept: true}}}}
without_metadata: {value: {document: {}}}
empty_metadata: {value: {document: {record_metadata: {}}}}
only_verified: {value: {document: {record_metadata: {verified_recapture: {version: 1.0.0}}}}}
only_sibling: {value: {document: {record_metadata: {kept: true}}}}
suppressed: {value: {document: {}}, suppress_inherited_from: true}
override: {value: {document: {inherited_from: explicit}}}
null_override: {value: {document: {inherited_from: null}}}
"#,
        )?;
        assert!(matches!(
            selected_snapshot(&records, "1.0.0", "1.0.0")?,
            Cow::Borrowed(_)
        ));
        let selected = selected_snapshot(&records, "1.0.0", "1.1.0")?;
        for name in ["plain", "without_metadata"] {
            assert_eq!(
                selected
                    .field(name)?
                    .field("value")?
                    .field("document")?
                    .field("inherited_from")?
                    .string()?,
                "1.0.0"
            );
        }
        assert!(
            selected
                .field("suppressed")?
                .field("value")?
                .field("document")?
                .get("inherited_from")
                .is_none()
        );
        let metadata = selected
            .field("plain")?
            .field("value")?
            .field("document")?
            .field("record_metadata")?;
        assert!(metadata.get("verified_recapture").is_none());
        assert_eq!(metadata.field("kept")?, &Node::Bool(true));
        for name in ["empty_metadata", "only_verified", "without_metadata"] {
            assert!(
                selected
                    .field(name)?
                    .field("value")?
                    .field("document")?
                    .get("record_metadata")
                    .is_none()
            );
        }
        assert_eq!(
            selected
                .field("only_sibling")?
                .field("value")?
                .field("document")?
                .field("record_metadata")?
                .field("kept")?,
            &Node::Bool(true)
        );

        assert_eq!(
            selected
                .field("override")?
                .field("value")?
                .field("document")?
                .field("inherited_from")?
                .string()?,
            "explicit"
        );
        assert_eq!(
            selected
                .field("null_override")?
                .field("value")?
                .field("document")?
                .field("inherited_from")?,
            &Node::Null
        );
        assert!(
            records
                .field("plain")?
                .field("value")?
                .field("document")?
                .get("inherited_from")
                .is_none()
        );
        Ok(())
    }

    fn capture(version: &str, body: &str) -> Result<Capture> {
        let id = format!("dynamo_v2-{version}");
        Ok(Capture {
            member: format!("family/{id}.yaml"),
            id,
            version: version.into(),
            document: load_str(body)?,
            resolved_records: None,
        })
    }

    #[test]
    fn sparse_observation_origin_and_metadata_ownership() -> Result<()> {
        let first = capture(
            "1.0.0",
            r#"
provenance: {status: captured, origin: {crate_version: 1.0.0}}
document: {family: sample}
changes:
  a: {case_key: a, stimulus: {ref: current}, observation: {value: {assembled: []}}}
metadata_changes:
  a: {verified_recapture: {version: 1.0.0}, kept: true}
document_overrides:
  a: {parser_path: unified, capture_provenance: {inherited_from: original}}
"#,
        )?;
        let second = capture(
            "1.1.0",
            r#"
parent: dynamo_v2-1.0.0
provenance: {status: legacy, captured_with: {dynamo_v2: 1.1.0}}
document: {family: sample}
changes:
  a: {observation: {error: {code: changed}}}
metadata_changes: {}
document_overrides: {}
"#,
        )?;
        let mut history = History {
            implementation: "dynamo_v2".into(),
            captures: vec![second, first],
        };
        history.validate()?;
        let mut cursor = Cursor::default();
        cursor.apply(&history, 0)?;
        let original = cursor.records(&history, "1.0.0")?;
        let document = original.field("a")?.field("value")?.field("document")?;
        assert!(
            document
                .field("record_metadata")?
                .get("verified_recapture")
                .is_some()
        );
        assert_eq!(document.field("inherited_from")?.string()?, "original");
        cursor.apply(&history, 1)?;
        let changed = cursor.records(&history, "1.1.0")?;
        let record = changed.field("a")?.field("value")?;
        let document = record.field("document")?;
        assert_eq!(document.field("parser_path")?.string()?, "unified");
        assert!(document.get("inherited_from").is_none());
        assert!(document.get("capture_origin").is_none());
        assert!(
            document
                .field("record_metadata")?
                .get("verified_recapture")
                .is_none()
        );
        assert_eq!(
            changed.field("a")?.field("member")?.string()?,
            "family/dynamo_v2-1.1.0.yaml"
        );
        assert_eq!(
            record
                .mapping()?
                .iter()
                .map(|(k, _)| k.string().unwrap())
                .collect::<Vec<_>>(),
            vec!["case_key", "stimulus", "observation", "document"]
        );
        Ok(())
    }

    #[test]
    fn absent_records_cannot_receive_metadata_and_parents_are_exact() -> Result<()> {
        let first = capture(
            "1.0.0",
            "provenance: {}\ndocument: {}\nchanges: {}\nmetadata_changes: {missing: {x: 1}}\ndocument_overrides: {}\n",
        )?;
        let mut history = History {
            implementation: "dynamo_v2".into(),
            captures: vec![first],
        };
        assert!(history.validate().is_err());
        let first = capture(
            "1.0.0",
            "parent: dynamo_v2-0.9.0\nprovenance: {}\ndocument: {}\nchanges: {}\nmetadata_changes: {}\ndocument_overrides: {}\n",
        )?;
        let mut history = History {
            implementation: "dynamo_v2".into(),
            captures: vec![first],
        };
        assert!(history.validate().is_err());
        Ok(())
    }

    #[test]
    fn execution_receipt_membership_merge_and_order() -> Result<()> {
        let digest = "receipts/sample--dynamo_v2-1.0.0--receipt-001.yaml".to_string();
        let manifest = load_str(&format!(
            r#"
schema_version: 1
family: sample
implementation: dynamo_v2
version: 1.0.0
checkpoint: {{provenance: {{}}, document: {{}}, document_overrides: {{}}}}
cases: [a]
defaults:
  before: 1
  verified_recapture: {{provenance: '{digest}'}}
  after: 2
exceptions: {{a: {{after: 3}}}}
receipts: {{{digest}: [a]}}
"#
        ))?;
        let receipts = [(digest.clone(), load_str("version: 1.0.0\n")?)]
            .into_iter()
            .collect();
        let expanded = expand_execution(&manifest, "sample", "dynamo_v2", "1.0.0", &receipts)?;
        let metadata = expanded.field("metadata_changes")?.field("a")?;
        assert_eq!(
            metadata
                .mapping()?
                .iter()
                .map(|(k, _)| k.string().unwrap())
                .collect::<Vec<_>>(),
            vec!["before", "verified_recapture", "after"]
        );
        assert_eq!(metadata.field("after")?, &Node::Int(3.into()));
        let mut corrupt = manifest;
        corrupt.set("receipts", Node::map())?;
        assert!(expand_execution(&corrupt, "sample", "dynamo_v2", "1.0.0", &receipts).is_err());
        Ok(())
    }

    #[test]
    fn frozen_provenance_keeps_strict_capture_versions() -> Result<()> {
        let qualified = format!("0.7.25+source.{}", "a".repeat(64));
        assert_eq!(
            Release::parse_provenance(&qualified)?,
            Release::parse("0.7.25")?
        );
        assert!(Release::parse(&qualified).is_err());
        for bad in ["0.7.25+source.a", "0.7.25+other.abc", "0.7.25+source./bad"] {
            assert!(Release::parse_provenance(bad).is_err());
        }
        Ok(())
    }

    #[test]
    fn release_order_matches_maintained_tuple() -> Result<()> {
        let versions = [
            "1.0.0-a.10",
            "1.0.0-a.2",
            "1.0.0",
            "1.0.0.post1",
            "1.0.0.post10",
            "1.0.0.0",
        ];
        let parsed = versions
            .iter()
            .map(|v| Release::parse(v))
            .collect::<Result<Vec<_>>>()?;
        assert!(parsed.windows(2).all(|w| w[0] < w[1]));
        Ok(())
    }

    #[test]
    fn sparse_null_override_survives_later_release_projection() -> Result<()> {
        let document = load_str(
            "provenance: {status: legacy}\ndocument: {}\nchanges: {a: {case_key: a, stimulus: {ref: current}, observation: {value: {assembled: []}}}}\nmetadata_changes: {}\ndocument_overrides: {a: {capture_provenance: {inherited_from: null}}}\n",
        )?;
        let mut history = History {
            implementation: "dynamo_v2".into(),
            captures: vec![Capture {
                id: "dynamo_v2-1.0.0".into(),
                version: "1.0.0".into(),
                member: "source.yaml".into(),
                document,
                resolved_records: None,
            }],
        };
        history.validate()?;
        let mut cursor = Cursor::default();
        cursor.apply(&history, 0)?;
        let records = cursor.records(&history, "1.0.0")?;
        assert_eq!(
            records.field("a")?.field("suppress_inherited_from")?,
            &Node::Bool(true)
        );
        let selected = selected_snapshot(&records, "1.0.0", "1.1.0")?;
        assert!(
            selected
                .field("a")?
                .field("value")?
                .field("document")?
                .get("inherited_from")
                .is_none()
        );
        Ok(())
    }

    #[test]
    fn original_hash_receipts_retain_membership_and_null() -> Result<()> {
        let digest = "a".repeat(64);
        let manifest = load_str(&format!(
            "schema_version: 1\nfamily: sample\nimplementation: dynamo_v2\nversion: 1.0.0\ncheckpoint: {{provenance: {{}}, document: {{}}, document_overrides: {{}}}}\ncases: [a]\ndefaults: {{verified_recapture: {{provenance: 'receipt:{digest}'}}}}\nexceptions: {{}}\nreceipts: {{{digest}: [a]}}\n"
        ))?;
        let receipt = load_str("version: 1.0.0\nnull_value: null\n")?;
        let receipts = [(digest.clone(), receipt.clone())].into_iter().collect();
        let expanded = expand_execution(&manifest, "sample", "dynamo_v2", "1.0.0", &receipts)?;
        assert_eq!(
            expanded
                .field("metadata_changes")?
                .field("a")?
                .field("verified_recapture")?
                .field("provenance")?,
            &receipt
        );
        assert_eq!(
            source_path(&digest, "executions")?,
            format!("executions/{digest}.yaml")
        );
        assert!(
            expand_execution(
                &manifest,
                "sample",
                "dynamo_v2",
                "1.0.0",
                &std::collections::BTreeMap::new()
            )
            .is_err()
        );
        Ok(())
    }

    #[test]
    fn malformed_metadata_receipt_is_preserved() -> Result<()> {
        let mut value = load_str("capture_provenance: receipt:x\n")?;
        let original = value.clone();
        let mut observed = std::collections::BTreeMap::new();
        expand_slots(
            &mut value,
            "a",
            &std::collections::BTreeMap::new(),
            &mut observed,
        )?;
        assert_eq!(value, original);
        assert!(observed.is_empty());
        Ok(())
    }
}
