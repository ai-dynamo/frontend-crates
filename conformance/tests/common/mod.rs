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

/// Ensures fixture files are available and returns the fixtures root path.
///
/// Priority:
/// 1. `CONFORMANCE_FIXTURES_ROOT` env var — set by `check.sh` after it has
///    already downloaded and verified the cache.
/// 2. HF cache at `~/.cache/dynamo/conformance-fixtures/` (or `$XDG_CACHE_HOME`).
///    If the `toolcalling/` subdir is missing, runs `download_fixtures.py`
///    automatically. A `flock` on `/tmp/dynamo-conformance-download.lock`
///    serializes parallel test binaries so only one download runs at a time.
///
/// If the download fails (e.g. no HF credentials), the test panics with a
/// message that includes the exact command to retry after setting `HF_TOKEN`.
pub fn ensure_fixtures() -> PathBuf {
    if let Ok(r) = std::env::var("CONFORMANCE_FIXTURES_ROOT") {
        return PathBuf::from(r);
    }

    let cache_root = std::env::var("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(std::env::var("HOME").expect("HOME not set")).join(".cache")
        })
        .join("dynamo/conformance-fixtures");

    if cache_root.join("toolcalling").is_dir() {
        return cache_root;
    }

    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("utils/src/download_fixtures.py");
    eprintln!(
        "[conformance] fixtures not cached — downloading (first run only): {}",
        script.display()
    );

    // flock serializes parallel test binaries so only one download runs.
    let status = std::process::Command::new("flock")
        .args([
            "/tmp/dynamo-conformance-download.lock",
            "python3",
            script.to_str().expect("non-UTF-8 script path"),
        ])
        .status()
        .expect("flock/python3 not found — ensure python3 is in PATH");

    if !status.success() {
        panic!(
            "fixture download failed (exit {}). Set HF_TOKEN and retry:\n  \
             export HF_TOKEN=<read-token>\n  python3 {}",
            status.code().unwrap_or(-1),
            script.display()
        );
    }

    cache_root
}

/// Crate-relative display path for a fixture (for failure messages).
pub fn fixture_name(path: &Path) -> String {
    path.strip_prefix(env!("CARGO_MANIFEST_DIR"))
        .unwrap_or(path)
        .display()
        .to_string()
}
