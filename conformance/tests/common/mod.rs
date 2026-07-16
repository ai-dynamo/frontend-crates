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
///    already extracted and verified the cache.
/// 2. Cache at `~/.cache/dynamo/conformance-fixtures/` (or `$XDG_CACHE_HOME`),
///    kept current by running `extract_fixtures.py` every time (extracts the
///    in-repo LFS shard store; no network). The script exits instantly on a
///    cache hit and re-extracts when the committed manifest pin moved — an
///    exists-check here would silently test against a stale snapshot. A
///    `flock` on `/tmp/dynamo-conformance-extract.lock` serializes parallel
///    test binaries so only one extraction runs at a time.
///
/// If extraction fails (e.g. shards are un-pulled git-lfs pointers), the test
/// panics with the exact command to fix the checkout.
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

    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("utils/src/extract_fixtures.py");

    // flock serializes parallel test binaries so only one extraction runs.
    let status = std::process::Command::new("flock")
        .args([
            "/tmp/dynamo-conformance-extract.lock",
            "python3",
            script.to_str().expect("non-UTF-8 script path"),
        ])
        .status()
        .expect("flock/python3 not found — ensure python3 is in PATH");

    if !status.success() {
        panic!(
            "fixture extraction failed (exit {}). If the shards are git-lfs \
             pointers, run:\n  git lfs install && git lfs pull\nthen retry:\n  python3 {}",
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

/// Version-sorted capture dirs for one impl prefix (e.g. `dynamo-` under
/// fixtures-batch-v1, `dynamo_v2-` under fixtures-stream-v2), ASCENDING by
/// numeric version. Multiple dirs per impl are capture HISTORY (never deleted);
/// readers fold them ascending so the latest capture wins per case.
pub fn version_dirs_ascending(root: &Path, prefix: &str) -> Vec<PathBuf> {
    let mut dirs: Vec<(Vec<u64>, PathBuf)> = std::fs::read_dir(root)
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.is_dir()
                && p.file_name().and_then(|n| n.to_str()).is_some_and(|n| {
                    // `<ver>.patchN` dirs are DISPLAY-ONLY overlays: an OLD parser
                    // binary re-run to backfill a newer case onto version `<ver>`
                    // (e.g. dynamo_v2-0.1.11.patch1 = the 0.1.11 binary on streamv2.5.h,
                    // rendered under the 0.1.11 column in HTML). They are NOT the current
                    // parser, so they must never join this "latest capture wins" fold —
                    // otherwise a stale old-binary result can shadow the real latest.
                    n.starts_with(prefix) && !n.contains(".patch")
                })
        })
        .map(|p| {
            let key: Vec<u64> = p
                .file_name()
                .and_then(|n| n.to_str())
                .and_then(|n| n.strip_prefix(prefix))
                .unwrap_or("")
                .split(|c: char| !c.is_ascii_digit())
                .filter(|s| !s.is_empty())
                .map(|s| s.parse().unwrap_or(0))
                .collect();
            (key, p)
        })
        .collect();
    dirs.sort();
    dirs.into_iter().map(|(_, p)| p).collect()
}
