// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;

use serde::Deserialize;

#[derive(Deserialize, Debug, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Assembled {
    pub calls: Vec<(String, serde_json::Value)>,
    pub normal_text: String,
}

#[derive(Deserialize, Debug, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RawAssembled {
    pub calls: Vec<(String, String)>,
    pub normal_text: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RawDivergence {
    pub original: RawAssembled,
    pub split: RawAssembled,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Divergence {
    pub note: String,
    pub got: Assembled,
    pub want: Assembled,
    #[serde(default)]
    pub raw: Option<RawDivergence>,
}

pub type KnownDivergences = BTreeMap<String, BTreeMap<String, Divergence>>;

pub fn load() -> KnownDivergences {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("toolcalling/known-chunking-divergences.yaml");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let known: KnownDivergences =
        serde_yaml::from_str(&text).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    for (family, cases) in &known {
        for (case_id, divergence) in cases {
            assert!(
                !divergence.note.trim().is_empty(),
                "{family}:{case_id}: empty note in known-chunking-divergences.yaml"
            );
        }
    }
    known
}
