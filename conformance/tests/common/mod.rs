// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared helpers for the conformance parity test binaries (audit B8): fixture
//! discovery + the crate-relative display path used in failure messages, which
//! were copied verbatim across `conformance_toolcalling`, `conformance_toolcalling_stream`,
//! and `conformance_toolcalling_batch_via_stream`. Each test binary declares
//! `mod common;` so this compiles into it; a binary that uses only a subset is
//! fine (hence the allow).
#![allow(dead_code)]

use std::io::Write;
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

    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("utils/src/extract_fixtures.py");

    // flock serializes parallel test binaries so only one extraction runs.
    let output = std::process::Command::new("flock")
        .args([
            "/tmp/dynamo-conformance-extract.lock",
            "python3",
            script.to_str().expect("non-UTF-8 script path"),
        ])
        .output()
        .expect("flock/python3 not found — ensure python3 is in PATH");

    // `.output()` captures stderr instead of inheriting it (needed to also
    // capture stdout below) -- forward it so extraction progress ("Extracting
    // N shard(s)...", "Cache hit: ...") is still visible in the test run,
    // not silently swallowed. A failure writing to this process's own
    // stderr is itself unusual enough to fail fast on rather than ignore.
    std::io::stderr()
        .write_all(&output.stderr)
        .expect("failed to forward extract_fixtures.py stderr to this process's stderr");

    if !output.status.success() {
        panic!(
            "fixture extraction failed (exit {}). If the shards are git-lfs \
             pointers, run:\n  git lfs install && git lfs pull\nthen retry:\n  python3 {}",
            output.status.code().unwrap_or(-1),
            script.display()
        );
    }

    // `extract_fixtures.py` prints its resolved, content-addressed snapshot
    // dir as the last stdout line. Return THAT, not `cache_root` (the
    // directory holding the mutable `toolcalling`/`reasoning`/`unified`
    // symlinks): every caller does `ensure_fixtures().join("<family>/...")`
    // and then reads many files under it over the test's lifetime, and
    // `Path::join` never touches the filesystem — the OS re-resolves any
    // symlink component on EVERY subsequent file access. A concurrent
    // sibling checkout publishing a different manifest's identity and
    // retargeting the symlink mid-test would silently switch which
    // snapshot later reads in the SAME test see, even though extraction
    // itself is now race-free (`fixtures_identity`-keyed, atomically
    // published). Resolving to the immutable identity dir once, up front,
    // matches the same fix `_common.sh` already applies for the identical
    // reason (see its `FIXTURES_SNAP` comment) — one shared pattern, not two.
    let stdout = String::from_utf8(output.stdout).expect("extract_fixtures.py stdout is not UTF-8");
    match resolve_snap_dir(&stdout) {
        Ok(snap_dir) => snap_dir,
        // A missing, malformed, or non-directory printed path is NOT
        // recovered by falling back to `cache_root` — that fallback is
        // exactly the mutable, racy path this function exists to stop
        // returning. Fail loudly with the full captured output instead, so a
        // broken contract is caught here, not silently downgraded back to
        // the old ownership model.
        Err(reason) => panic!(
            "extract_fixtures.py did not print a valid resolved snapshot directory as its \
             last stdout line: {reason}.\nfull stdout: {stdout:?}\nstderr: {:?}",
            String::from_utf8_lossy(&output.stderr)
        ),
    }
}

/// Pure parsing/validation of `ensure_fixtures`'s subprocess contract,
/// split out so the failure shapes (empty output, a non-existent path, a
/// malformed line, extra noisy lines) are directly unit-testable without a
/// real `flock`/`python3` subprocess.
fn resolve_snap_dir(stdout: &str) -> Result<PathBuf, String> {
    let printed = stdout.lines().next_back().unwrap_or("").trim();
    if printed.is_empty() {
        return Err("stdout was empty (or only blank lines)".to_string());
    }
    let snap_dir = PathBuf::from(printed);
    if !snap_dir.is_dir() {
        return Err(format!(
            "printed path {printed:?} is not an existing directory"
        ));
    }
    Ok(snap_dir)
}

#[cfg(test)]
mod resolve_snap_dir_tests {
    use super::resolve_snap_dir;

    #[test]
    fn accepts_a_real_directory_on_the_last_line() {
        let dir = std::env::temp_dir().join(format!(
            "dynamo-resolve-snap-dir-test-{}-{}",
            std::process::id(),
            "ok"
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let stdout = format!("Extracting 3 shard(s) into ...\n{}\n", dir.display());
        assert_eq!(resolve_snap_dir(&stdout), Ok(dir.clone()));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn rejects_empty_stdout() {
        assert!(resolve_snap_dir("").is_err());
        assert!(resolve_snap_dir("\n\n").is_err());
    }

    #[test]
    fn rejects_a_malformed_or_missing_path() {
        let err = resolve_snap_dir("not a real path at all\n").unwrap_err();
        assert!(err.contains("not an existing directory"), "{err}");
    }

    #[test]
    fn rejects_a_path_that_does_not_exist_on_disk() {
        let err = resolve_snap_dir("/definitely/does/not/exist/anywhere\n").unwrap_err();
        assert!(err.contains("not an existing directory"), "{err}");
    }

    #[test]
    fn uses_only_the_last_line_ignoring_noisy_progress_output() {
        let dir = std::env::temp_dir().join(format!(
            "dynamo-resolve-snap-dir-test-{}-{}",
            std::process::id(),
            "noisy"
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let stdout = format!(
            "Extracting 47 shard(s) into {}\n  [extract] a.tar.gz -> ...\n  [extract] b.tar.gz -> ...\n{}\n",
            dir.display(),
            dir.display()
        );
        assert_eq!(resolve_snap_dir(&stdout), Ok(dir.clone()));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn rejects_a_directory_that_is_actually_a_file() {
        let file = std::env::temp_dir().join(format!(
            "dynamo-resolve-snap-dir-test-{}-{}",
            std::process::id(),
            "file"
        ));
        std::fs::write(&file, b"not a directory").unwrap();
        let stdout = format!("{}\n", file.display());
        let err = resolve_snap_dir(&stdout).unwrap_err();
        assert!(err.contains("not an existing directory"), "{err}");
        std::fs::remove_file(&file).unwrap();
    }
}

/// Ensures the authored unified golden spec exists and returns its directory.
///
/// The golden corpus is the AUTHORED oracle — a spec, not a capture — so it is
/// NOT committed (that would leave a stray loose YAML tree next to the versioned
/// `*.tar.gz` shards). Instead `gen_unified_golden.py` renders it from one
/// scenario spec into the gitignored build tree (`conformance/unified/golden_spec/`)
/// on demand, mirroring how [`ensure_fixtures`] shells out to `extract_fixtures.py`.
/// The committed `golden.tar.gz` shard is DERIVED from this via render -> explode
/// -> package. A `flock` serializes the two unified test binaries so they don't
/// race writing the same files. Panics with the fix command if generation fails.
pub fn ensure_unified_golden() -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let script = manifest.join("utils/src/gen_unified_golden.py");
    let status = std::process::Command::new("flock")
        .args([
            "/tmp/dynamo-unified-golden.lock",
            "python3",
            script.to_str().expect("non-UTF-8 script path"),
        ])
        .status()
        .expect("flock/python3 not found — ensure python3 is in PATH");
    if !status.success() {
        panic!(
            "unified golden generation failed (exit {}). Run manually:\n  python3 {}",
            status.code().unwrap_or(-1),
            script.display()
        );
    }
    manifest.join("unified/golden_spec")
}

/// Crate-relative display path for a fixture (for failure messages).
pub fn fixture_name(path: &Path) -> String {
    path.strip_prefix(env!("CARGO_MANIFEST_DIR"))
        .unwrap_or(path)
        .display()
        .to_string()
}

/// Current parser-version capture used by stream parity and interleave tests.
pub const STREAM_DYNAMO_V2_CURRENT_CAPTURE: &str = "dynamo_v2-0.4.0";

pub const UNIFIED_DYNAMO_V2_CURRENT_CAPTURE: &str = "dynamo_v2-0.4.1";

/// Version-sorted capture dirs for one impl prefix (e.g. `dynamo-` under
/// fixtures-batch-v1, `dynamo_v2-` under fixtures-stream-v2), ASCENDING by
/// numeric version. Multiple dirs per impl are capture HISTORY (never deleted);
/// readers fold them ascending so the latest capture wins per case.
pub type VersionCaptureSortKey = (Vec<u64>, bool, String);

pub fn version_dirs_ascending(root: &Path, prefix: &str) -> Vec<PathBuf> {
    let mut dirs: Vec<(VersionCaptureSortKey, PathBuf)> = std::fs::read_dir(root)
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .filter_map(|p| {
            let key = p
                .file_name()
                .and_then(|n| n.to_str())
                .and_then(|n| version_capture_sort_key(n, prefix))?;
            Some((key, p))
        })
        .collect();
    dirs.sort();
    dirs.into_iter().map(|(_, p)| p).collect()
}

/// The normal capture history plus one caller-selected current capture. `+tag`
/// directories remain historical unless a test names the exact directory it needs.
pub fn version_dirs_ascending_with_current(
    root: &Path,
    prefix: &str,
    current_dir: &str,
) -> Vec<PathBuf> {
    let current = root.join(current_dir);
    assert!(
        current.is_dir(),
        "expected current capture directory {}",
        current.display()
    );
    let mut dirs = version_dirs_ascending(root, prefix)
        .into_iter()
        .filter(|path| path != &current && !path.to_string_lossy().contains('+'))
        .collect::<Vec<_>>();
    dirs.push(current);
    dirs
}

/// Sort a PR-qualified capture after its matching release. The PR capture represents
/// the current branch, while a plain release remains a historical comparison point.
pub fn version_capture_sort_key(name: &str, prefix: &str) -> Option<VersionCaptureSortKey> {
    let version = name.strip_prefix(prefix)?;
    // `.patchN` dirs are display-only overlays from an old binary. They fill missing
    // cases in that binary's column and must never participate in latest-capture folds.
    if version.contains(".patch") {
        return None;
    }
    let base = version.split_once('+').map_or(version, |(base, _)| base);
    let numeric = base
        .split(|c: char| !c.is_ascii_digit())
        .filter(|s| !s.is_empty())
        .map(|s| s.parse().unwrap_or(0))
        .collect();
    Some((numeric, version.contains('+'), version.to_string()))
}

/// One row of the `unified:` block in `conformance/utils/src/parser_families.yaml`.
///
/// That block is the ONE place a family is declared for the unified tab. It replaced
/// five lists that had to agree — `FAMILIES` / `FAM_FILE` / `UNIFIED_FAMILIES` in
/// `gen_unified_golden.py`, `parsers_for()` in two separate Rust test binaries, the
/// `family_leak` match, and `MARKER_FAMILY` — where a family added to one and missed in
/// another either panicked at "no parser mapping" or silently lost its leak detection.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct UnifiedFamily {
    /// Key into `families:` / `markers:`; differs from the corpus name for qwen3.
    #[serde(default)]
    pub registry: Option<String>,
    /// Has a native UnifiedParser, versus being driven through the v1/v2 split path.
    #[serde(default)]
    pub native: bool,
    pub reasoning_parser: String,
    pub tool_parser: String,
    pub golden_spec: String,
    /// Markup invisible to the shared leak list, so it has to be named per family.
    #[serde(default)]
    pub leak_markers: Vec<String>,
}

impl UnifiedFamily {
    /// The `families:` / `markers:` key for this corpus family.
    pub fn registry_key<'a>(&'a self, corpus_name: &'a str) -> &'a str {
        self.registry.as_deref().unwrap_or(corpus_name)
    }
}

/// Path to the family manifest. `CONFORMANCE_FAMILIES` overrides it so a harness copied
/// into an OLDER worktree (see `capture_cross_version.rs`) still reads the CURRENT
/// declarations rather than whatever that commit happened to ship.
pub fn family_manifest_path() -> PathBuf {
    if let Ok(p) = std::env::var("CONFORMANCE_FAMILIES") {
        return PathBuf::from(p);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("utils/src/parser_families.yaml")
}

/// Every family declared for the unified tab, keyed by CORPUS name.
pub fn unified_families() -> std::collections::BTreeMap<String, UnifiedFamily> {
    #[derive(serde::Deserialize)]
    struct Doc {
        unified: std::collections::BTreeMap<String, UnifiedFamily>,
    }
    let path = family_manifest_path();
    let text =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let doc: Doc =
        serde_yaml::from_str(&text).unwrap_or_else(|e| panic!("parse {}: {e}", path.display()));
    doc.unified
}

/// One family, or a panic naming the manifest — a family reaching the harness without a
/// declaration is a missing row, and saying so beats a generic "no parser mapping".
pub fn unified_family(corpus_name: &str) -> UnifiedFamily {
    unified_families().remove(corpus_name).unwrap_or_else(|| {
        panic!(
            "family `{corpus_name}` is not declared under `unified:` in {} — add a row there",
            family_manifest_path().display()
        )
    })
}

/// The request-scoped parser configuration a unified case declares.
///
/// Read from the case's `init:` block and passed to the parser verbatim, by BOTH
/// unified harnesses (`unified_render` draws the tab, `unified_parity` gates CI). It
/// lives here so there is exactly one answer to "how is a case's parser configured":
/// each harness previously carried its own copy that INFERRED the config by sniffing
/// the input text, and the copies could disagree with each other and with the `init:`
/// the popup displayed — a case could declare `tool_output_mode=GuidedJson` and be
/// parsed as `Native` because its input did not happen to start with `[`.
#[derive(Debug, Clone, Default, serde::Deserialize, serde::Serialize, PartialEq)]
pub struct Init {
    #[serde(default)]
    pub starting_state: String,
    #[serde(default)]
    pub tool_output_mode: String,
    #[serde(default)]
    pub named_tool: Option<String>,
}

impl Init {
    /// An unknown value is a spec bug, not something to paper over with a default:
    /// silently falling back to `None`/`Native` is exactly the failure this replaced.
    pub fn starting_state(&self) -> dynamo_parsers_v2::UnifiedParserStartingState {
        use dynamo_parsers_v2::UnifiedParserStartingState as P;
        match self.starting_state.as_str() {
            "" | "None" => P::None,
            "Reasoning" => P::Reasoning,
            "Response" => P::Response,
            other => panic!("unknown init.starting_state `{other}` (None|Reasoning|Response)"),
        }
    }

    pub fn output_mode(&self) -> dynamo_parsers_v2::UnifiedToolOutputMode {
        use dynamo_parsers_v2::UnifiedToolOutputMode as O;
        match self.tool_output_mode.as_str() {
            "" | "Native" => O::Native,
            "GuidedJson" => O::GuidedJson {
                named_tool: self.named_tool.clone(),
            },
            other => panic!("unknown init.tool_output_mode `{other}` (Native|GuidedJson)"),
        }
    }

    /// Apply this configuration to a freshly created parser.
    pub fn apply(&self, parser: &mut Box<dyn dynamo_parsers_v2::UnifiedParser>, what: &str) {
        use dynamo_parsers_v2::{InvalidGuidedPayloadPolicy, UnifiedParserInit};
        parser
            .initialize_request(UnifiedParserInit {
                starting_state: self.starting_state(),
                tool_output_mode: self.output_mode(),
                invalid_guided_payload: InvalidGuidedPayloadPolicy::RecoverAsText,
                ..UnifiedParserInit::default()
            })
            .unwrap_or_else(|e| panic!("{what}: initialize_request {self:?}: {e}"));
    }

    /// The config as APPLIED, not as written — an omitted field is reported as the
    /// value the parser actually received, so what the popup shows and what the
    /// parser ran under are the same object by construction.
    pub fn applied(&self) -> serde_json::Value {
        use dynamo_parsers_v2::UnifiedToolOutputMode as O;
        serde_json::json!({
            "starting_state": format!("{:?}", self.starting_state()),
            "tool_output_mode": match self.output_mode() {
                O::Native => "Native",
                O::GuidedJson { .. } => "GuidedJson",
            },
            "named_tool": self.named_tool,
        })
    }
}
