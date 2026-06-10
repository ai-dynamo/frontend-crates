// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared helpers for the conformance parity test binaries (audit B8): fixture
//! discovery + the crate-relative display path used in failure messages, which
//! were copied verbatim across `parity_toolcalling`, `parity_toolcalling_stream`,
//! and `parity_toolcalling_batch_via_stream`. Each test binary declares
//! `mod common;` so this compiles into it; a binary that uses only a subset is
//! fine (hence the allow).
#![allow(dead_code)]

use std::path::{Path, PathBuf};

/// Recursively collect `*.yaml` fixture files under `dir` into `out`.
pub fn collect_yaml(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in rd.flatten() {
        let p = entry.path();
        if p.is_dir() {
            collect_yaml(&p, out);
        } else if p.extension().is_some_and(|x| x == "yaml") {
            out.push(p);
        }
    }
}

/// Crate-relative display path for a fixture (for failure messages).
pub fn fixture_name(path: &Path) -> String {
    path.strip_prefix(env!("CARGO_MANIFEST_DIR"))
        .unwrap_or(path)
        .display()
        .to_string()
}
