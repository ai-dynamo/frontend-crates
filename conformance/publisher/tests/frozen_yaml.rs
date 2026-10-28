// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::{Context, Result, ensure};
use conformance_publisher::{resources::Policy, yaml};
use rayon::prelude::*;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{env, fs, path::Path};

fn compare_frozen_value(row: &Value, bytes: &[u8], label: &str) -> Result<()> {
    ensure!(
        format!("{:x}", Sha256::digest(bytes)) == row["source_sha256"],
        "frozen source changed: {label}"
    );
    let loaded = std::str::from_utf8(bytes)
        .context("invalid UTF-8")
        .and_then(yaml::load_str);
    if row["rejected"] == true {
        ensure!(
            loaded.is_err(),
            "accepted rejected oracle document: {label}"
        );
        return Ok(());
    }
    let value = loaded.with_context(|| label.to_string())?;
    let typed = serde_json::to_vec(&value.typed())?;
    ensure!(
        format!("{:x}", Sha256::digest(&typed)) == row["typed_sha256"],
        "typed YAML mismatch: {label}"
    );
    Ok(())
}

#[test]
fn frozen_semantic_probes_match() -> Result<()> {
    let rows: Vec<Value> = serde_json::from_str(include_str!("fixtures/yaml_semantics.json"))?;
    ensure!(rows.len() == 67, "frozen semantic probe inventory differs");
    for row in rows {
        compare_frozen_value(
            &row,
            row["source"].as_str().context("missing source")?.as_bytes(),
            row["label"].as_str().context("missing label")?,
        )?;
    }
    Ok(())
}

/// The manifest is produced by the independently frozen Python loader. Each
/// entry names immutable YAML bytes and the hash of its exact typed encoding.
#[test]
#[ignore = "requires CONFORMANCE_FROZEN_YAML pointing to an external oracle manifest"]
fn all_frozen_yaml_values_match() -> Result<()> {
    let manifest = env::var("CONFORMANCE_FROZEN_YAML")?;
    let policy =
        Policy::detect(&env::var("CONFORMANCE_WORKERS").unwrap_or_else(|_| "auto".into()))?;
    eprintln!("{policy:?}");
    let pool = policy.pool()?;
    let rows: Vec<Value> = serde_json::from_slice(&fs::read(manifest)?)?;
    ensure!(!rows.is_empty(), "empty oracle manifest");
    pool.install(|| {
        rows.par_iter().try_for_each(|row| -> Result<()> {
            let path = Path::new(row["path"].as_str().context("missing path")?);
            let bytes = fs::read(path)?;
            compare_frozen_value(row, &bytes, &path.display().to_string())
        })
    })?;
    drop(pool);
    eprintln!("compared {} frozen strict-loader documents", rows.len());
    Ok(())
}
