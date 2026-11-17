// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What the integration tests share: fixture paths, digests in the fixtures'
//! byte order, and the per-item counts of a token layout.

// Each test binary compiles this module and uses a part of it.
#![allow(dead_code)]

use std::path::{Path, PathBuf};

use dynamo_multimodal::processor::{ExpansionPart, Segment, TokenLayout};
use serde_json::Value;
use sha2::{Digest, Sha256};

pub fn mm_parity() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/mm_parity")
}

pub fn image(name: &str) -> Vec<u8> {
    let path = mm_parity().join("images").join(name);
    std::fs::read(&path).unwrap_or_else(|error| panic!("{}: {error}", path.display()))
}

/// A model table, by file name.
pub fn table(name: &str) -> Value {
    read_json(&mm_parity().join(name))
}

pub fn read_json(path: &Path) -> Value {
    let text =
        std::fs::read_to_string(path).unwrap_or_else(|error| panic!("{}: {error}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|error| panic!("{}: {error}", path.display()))
}

pub fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// SHA-256 of little-endian words, as the fixtures hash `f32` features, `i32`
/// ids and `i64` positions.
pub fn sha256_le<const N: usize>(words: impl IntoIterator<Item = [u8; N]>) -> String {
    sha256(&words.into_iter().flatten().collect::<Vec<u8>>())
}

/// Per media item, in prompt order: the `Feature` positions (what
/// `apply_layout` takes) and the `Literal` ids of its expansion.
pub fn layout_counts(layout: &TokenLayout) -> Vec<(usize, usize)> {
    (layout.segments.iter())
        .filter_map(|segment| match segment {
            Segment::Media { expansion, .. } => Some(expansion.iter().fold(
                (0, 0),
                |(features, literals), part| match part {
                    ExpansionPart::Feature { n, .. } => (features + n, literals),
                    ExpansionPart::Literal(ids) => (features, literals + ids.len()),
                    _ => panic!("an expansion part this helper does not count"),
                },
            )),
            Segment::Text(_) => None,
        })
        .collect()
}
