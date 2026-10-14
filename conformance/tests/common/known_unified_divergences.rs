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

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChunkInvarianceDivergence {
    pub note: String,
    pub baseline: String,
    pub divergent: String,
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
    pub chunk_invariance: Option<ChunkInvarianceDivergence>,
    #[serde(default)]
    pub stream_batch: Option<StreamBatchDivergence>,
}

pub type KnownDivergences = BTreeMap<String, BTreeMap<String, CaseDivergence>>;

#[derive(Clone, Copy)]
pub enum Check {
    Golden,
    ChunkInvariance,
    StreamBatch,
}

pub enum Expected<'a> {
    Golden(&'a GoldenDivergence),
    ChunkInvariance(&'a ChunkInvarianceDivergence),
    StreamBatch(&'a StreamBatchDivergence),
}

impl CaseDivergence {
    fn has_check(&self, check: Check) -> bool {
        match check {
            Check::Golden => self.golden.is_some(),
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
                    || divergence.chunk_invariance.is_some()
                    || divergence.stream_batch.is_some(),
                "{family}:{case_id}: no check declared in unified-known-divergences.yaml"
            );
            for note in [
                divergence.golden.as_ref().map(|d| d.note.as_str()),
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
