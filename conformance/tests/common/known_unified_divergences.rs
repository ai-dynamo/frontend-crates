// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, BTreeSet};

use serde::Deserialize;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GoldenDivergence {
    pub note: String,
    pub actual: String,
}

#[derive(Debug, Deserialize)]
#[serde(try_from = "ChunkInvarianceRow")]
pub struct ChunkInvarianceDivergence {
    pub note: String,
    pub baseline: String,
    pub divergent: Vec<String>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum ChunkOutputs {
    Single(String),
    Multiple(Vec<String>),
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ChunkInvarianceRow {
    note: String,
    baseline: String,
    divergent: ChunkOutputs,
}

impl TryFrom<ChunkInvarianceRow> for ChunkInvarianceDivergence {
    type Error = String;

    fn try_from(row: ChunkInvarianceRow) -> Result<Self, Self::Error> {
        let divergent = match row.divergent {
            ChunkOutputs::Single(output) => vec![output],
            ChunkOutputs::Multiple(outputs) => outputs,
        };
        if divergent.is_empty() {
            return Err("chunk divergence requires at least one exact output".into());
        }
        let mut unique = BTreeSet::new();
        for output in &divergent {
            if output == &row.baseline {
                return Err("chunk divergent output equals its baseline".into());
            }
            if !unique.insert(output) {
                return Err("duplicate exact chunk divergent output".into());
            }
        }
        Ok(Self {
            note: row.note,
            baseline: row.baseline,
            divergent,
        })
    }
}

impl ChunkInvarianceDivergence {
    pub fn observe(
        &self,
        baseline: &str,
        output: &str,
        observed: &mut BTreeSet<String>,
    ) -> Result<(), String> {
        if baseline != self.baseline {
            return Err(format!(
                "known baseline changed: expected {:?}, actual {baseline:?}",
                self.baseline
            ));
        }
        if output == baseline {
            return Ok(());
        }
        if !self.divergent.iter().any(|allowed| allowed == output) {
            return Err(format!(
                "unknown split output: expected {:?}, actual {output:?}",
                self.divergent
            ));
        }
        observed.insert(output.to_owned());
        Ok(())
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StreamBatchDivergence {
    pub note: String,
    pub batch: String,
    pub stream: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CaseDivergence {
    #[serde(default)]
    pub golden: Option<GoldenDivergence>,
    #[serde(default)]
    pub projection: Option<GoldenDivergence>,
    #[serde(default)]
    pub chunk_invariance: Option<ChunkInvarianceDivergence>,
    #[serde(default)]
    pub stream_batch: Option<StreamBatchDivergence>,
}

pub type KnownDivergences = BTreeMap<String, BTreeMap<String, CaseDivergence>>;
pub type ObservedChunkDivergences = BTreeMap<(String, String), BTreeSet<String>>;

#[derive(Clone, Copy)]
pub enum Check {
    Golden,
    Projection,
    ChunkInvariance,
    StreamBatch,
}

pub enum Expected<'a> {
    Golden(&'a GoldenDivergence),
    Projection(&'a GoldenDivergence),
    ChunkInvariance(&'a ChunkInvarianceDivergence),
    StreamBatch(&'a StreamBatchDivergence),
}

impl CaseDivergence {
    fn has_check(&self, check: Check) -> bool {
        match check {
            Check::Golden => self.golden.is_some(),
            Check::Projection => self.projection.is_some(),
            Check::ChunkInvariance => self.chunk_invariance.is_some(),
            Check::StreamBatch => self.stream_batch.is_some(),
        }
    }
}

pub fn load() -> KnownDivergences {
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("unified-known-divergences.yaml");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let published: KnownDivergences =
        serde_yaml::from_str(&text).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let mut known = KnownDivergences::new();
    for (family, cases) in published {
        let inputs_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures-unified-v2/families")
            .join(&family)
            .join("inputs_and_golden.yaml");
        let inputs: serde_yaml::Value = serde_yaml::from_str(
            &std::fs::read_to_string(&inputs_path)
                .unwrap_or_else(|e| panic!("{}: {e}", inputs_path.display())),
        )
        .unwrap_or_else(|e| panic!("{}: {e}", inputs_path.display()));
        let authored = inputs["cases"].as_mapping().expect("family cases mapping");
        let mut projected = BTreeMap::new();
        for (case_id, divergence) in cases {
            let owner = authored.values().find(|case| {
                case["display_id"].as_str() == Some(case_id.as_str())
                    || case["historical_ids"].as_sequence().is_some_and(|ids| {
                        ids.iter().any(|id| id.as_str() == Some(case_id.as_str()))
                    })
            });
            // Tests still address internal scenario names; the store owns the
            // mapping from published labels, so Rust does not duplicate taxonomy.
            let internal_id = owner.map_or(case_id, |case| {
                format!(
                    "UNIFIED.{}.{}",
                    case["scenario"].as_str().expect("scenario"),
                    family
                )
            });
            assert!(
                projected.insert(internal_id, divergence).is_none(),
                "duplicate divergence owner"
            );
        }
        known.insert(family, projected);
    }
    for (family, cases) in &known {
        for (case_id, divergence) in cases {
            assert!(
                divergence.golden.is_some()
                    || divergence.projection.is_some()
                    || divergence.chunk_invariance.is_some()
                    || divergence.stream_batch.is_some(),
                "{family}:{case_id}: no check declared in unified-known-divergences.yaml"
            );
            for note in [
                divergence.golden.as_ref().map(|d| d.note.as_str()),
                divergence.projection.as_ref().map(|d| d.note.as_str()),
                divergence
                    .chunk_invariance
                    .as_ref()
                    .map(|d| d.note.as_str()),
                divergence.stream_batch.as_ref().map(|d| d.note.as_str()),
            ]
            .into_iter()
            .flatten()
            {
                assert!(
                    !note.trim().is_empty(),
                    "{family}:{case_id}: empty note in unified-known-divergences.yaml"
                );
            }
        }
    }
    known
}

pub fn expected<'a>(
    known: &'a KnownDivergences,
    family: &str,
    case_id: &str,
    check: Check,
) -> Option<Expected<'a>> {
    let divergence = known.get(family)?.get(case_id)?;
    match check {
        Check::Golden => divergence.golden.as_ref().map(Expected::Golden),
        Check::Projection => divergence.projection.as_ref().map(Expected::Projection),
        Check::ChunkInvariance => divergence
            .chunk_invariance
            .as_ref()
            .map(Expected::ChunkInvariance),
        Check::StreamBatch => divergence.stream_batch.as_ref().map(Expected::StreamBatch),
    }
}

pub fn reconcile(
    known: &KnownDivergences,
    check: Check,
    observed: &BTreeSet<(String, String)>,
) -> Vec<String> {
    assert!(
        !matches!(check, Check::ChunkInvariance),
        "chunk invariance must reconcile exact outputs, not case presence"
    );
    let mut failures = Vec::new();
    for (family, cases) in known {
        for (case_id, divergence) in cases {
            if divergence.has_check(check) && !observed.contains(&(family.clone(), case_id.clone()))
            {
                failures.push(format!(
                    "{family}:{case_id}: listed in unified-known-divergences.yaml but no longer diverges"
                ));
            }
        }
    }
    failures
}

pub fn reconcile_chunk_invariance(
    known: &KnownDivergences,
    observed: &ObservedChunkDivergences,
) -> Vec<String> {
    let mut failures = Vec::new();
    for (family, cases) in known {
        for (case_id, divergence) in cases {
            let Some(expected) = &divergence.chunk_invariance else {
                continue;
            };
            let seen = observed.get(&(family.clone(), case_id.clone()));
            for output in seen.into_iter().flatten() {
                if !expected.divergent.contains(output) {
                    failures.push(format!(
                        "{family}:{case_id}: unlisted exact chunk output {output:?} was observed"
                    ));
                }
            }
            for output in &expected.divergent {
                if !seen.is_some_and(|seen| seen.contains(output)) {
                    failures.push(format!(
                        "{family}:{case_id}: exact chunk output {output:?} listed in unified-known-divergences.yaml was not observed"
                    ));
                }
            }
        }
    }
    failures
}

#[cfg(test)]
mod tests {
    use super::*;

    fn known(outputs: serde_json::Value) -> KnownDivergences {
        serde_yaml::from_str(
            &serde_json::json!({"sample":{"case":{"chunk_invariance":{
                "note":"exact measured outputs", "baseline":"whole", "divergent":outputs,
            }}}})
            .to_string(),
        )
        .unwrap()
    }

    #[test]
    fn legacy_single_chunk_output_and_single_item_list_remain_exact() {
        for output in [serde_json::json!("split"), serde_json::json!(["split"])] {
            let known = known(output);
            let expected = known["sample"]["case"].chunk_invariance.as_ref().unwrap();
            let mut observed = ObservedChunkDivergences::new();
            let seen = observed
                .entry(("sample".into(), "case".into()))
                .or_default();
            expected.observe("whole", "whole", seen).unwrap();
            assert!(seen.is_empty());
            expected.observe("whole", "split", seen).unwrap();
            assert!(reconcile_chunk_invariance(&known, &observed).is_empty());
        }
    }

    #[test]
    fn every_exact_chunk_alternative_must_be_observed() {
        let known = known(serde_json::json!(["text(\"é \")", "text(\"é\")"]));
        let expected = known["sample"]["case"].chunk_invariance.as_ref().unwrap();
        let mut observed = ObservedChunkDivergences::new();
        assert_eq!(reconcile_chunk_invariance(&known, &observed).len(), 2);
        let seen = observed
            .entry(("sample".into(), "case".into()))
            .or_default();
        expected.observe("whole", "text(\"é \")", seen).unwrap();
        assert_eq!(reconcile_chunk_invariance(&known, &observed).len(), 1);
        let seen = observed.get_mut(&("sample".into(), "case".into())).unwrap();
        expected.observe("whole", "text(\"é\")", seen).unwrap();
        expected.observe("whole", "text(\"é\")", seen).unwrap();
        assert_eq!(seen.len(), 2);
        assert!(reconcile_chunk_invariance(&known, &observed).is_empty());
    }

    #[test]
    fn unknown_outputs_and_baseline_drift_never_match_a_known_case() {
        let known = known(serde_json::json!(["split-a", "split-b"]));
        let expected = known["sample"]["case"].chunk_invariance.as_ref().unwrap();
        let mut seen = BTreeSet::new();
        assert!(expected.observe("whole", "unknown", &mut seen).is_err());
        assert!(expected.observe("changed", "split-a", &mut seen).is_err());
        assert!(expected.observe("changed", "changed", &mut seen).is_err());
        assert!(seen.is_empty());
        seen.extend(["split-a".into(), "split-b".into(), "unknown".into()]);
        let observed = BTreeMap::from([(("sample".into(), "case".into()), seen)]);
        assert_eq!(reconcile_chunk_invariance(&known, &observed).len(), 1);
    }

    #[test]
    #[should_panic(expected = "must reconcile exact outputs")]
    fn case_presence_cannot_bypass_exact_chunk_output_reconciliation() {
        let known = known(serde_json::json!(["split-a", "split-b"]));
        let cases = BTreeSet::from([("sample".into(), "case".into())]);
        reconcile(&known, Check::ChunkInvariance, &cases);
    }

    #[test]
    fn malformed_chunk_alternative_sets_are_rejected() {
        for outputs in [
            serde_json::json!([]),
            serde_json::json!(["split", "split"]),
            serde_json::json!("whole"),
            serde_json::json!(["split", "whole"]),
            serde_json::json!(["split", 7]),
        ] {
            let row = serde_json::json!({"note":"exact", "baseline":"whole", "divergent":outputs});
            assert!(
                serde_yaml::from_str::<ChunkInvarianceDivergence>(&row.to_string()).is_err(),
                "accepted malformed row {row}"
            );
        }
    }

    #[test]
    fn whitespace_argument_types_and_raw_errors_remain_distinct() {
        for (allowed, mutation) in [
            ("text(\"é \")", "text(\"é\")"),
            ("tool_call(f, {\"n\":1})", "tool_call(f, {\"n\":\"1\"})"),
            ("ERROR: broken\n  detail ", "ERROR: broken detail"),
        ] {
            let known = known(serde_json::json!(allowed));
            let expected = known["sample"]["case"].chunk_invariance.as_ref().unwrap();
            let mut seen = BTreeSet::new();
            assert!(expected.observe("whole", mutation, &mut seen).is_err());
            expected.observe("whole", allowed, &mut seen).unwrap();
            assert_eq!(seen, BTreeSet::from([allowed.into()]));
        }
    }

    #[test]
    fn exact_empty_output_is_an_observation_not_an_empty_alternative_set() {
        let known = known(serde_json::json!(""));
        let expected = known["sample"]["case"].chunk_invariance.as_ref().unwrap();
        let mut observed = ObservedChunkDivergences::new();
        let seen = observed
            .entry(("sample".into(), "case".into()))
            .or_default();
        expected.observe("whole", "", seen).unwrap();
        assert!(reconcile_chunk_invariance(&known, &observed).is_empty());
    }
}
