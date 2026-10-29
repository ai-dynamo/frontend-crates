// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
use anyhow::{Context, Result, ensure};
use conformance_publisher::source::canonical_alias;
use serde_json::Value;
use std::{env, fs};

#[test]
#[ignore = "requires CONFORMANCE_FROZEN_ALIASES from the independent taxonomy oracle"]
fn all_aliases_match_frozen_python() -> Result<()> {
    let manifest: Value =
        serde_json::from_slice(&fs::read(env::var("CONFORMANCE_FROZEN_ALIASES")?)?)?;
    let rows = manifest["rows"]
        .as_array()
        .context("missing alias oracle rows")?;
    ensure!(!rows.is_empty(), "empty alias oracle");
    for row in rows {
        let alias = row["alias"].as_str().context("missing oracle alias")?;
        let family = row["family"].as_str().context("missing oracle family")?;
        let actual = canonical_alias(alias, family, &manifest["metadata"]);
        if row["expected"].is_null() {
            ensure!(actual.is_err(), "accepted ambiguous alias {family}/{alias}");
        } else {
            ensure!(
                actual?
                    == row["expected"]
                        .as_str()
                        .context("invalid expected alias")?
                        .strip_prefix("UNIFIED.")
                        .context("missing oracle prefix")?,
                "alias mismatch {family}/{alias}"
            );
        }
    }
    eprintln!("compared {} frozen taxonomy aliases", rows.len());
    Ok(())
}
