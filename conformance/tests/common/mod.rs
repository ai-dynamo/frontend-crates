// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared helpers for the conformance parity test binaries (audit B8): fixture
//! discovery + the crate-relative display path used in failure messages, which
//! were copied verbatim across `conformance_toolcalling`, `conformance_toolcalling_stream`,
//! and `conformance_toolcalling_batch_via_stream`. Each test binary declares
//! `mod common;` so this compiles into it; a binary that uses only a subset is
//! fine (hence the allow).
#![allow(dead_code)]
// The copied historical harness supplies this cfg without editing old manifests.
#![allow(unexpected_cfgs)]

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use serde_json::Value;

/// Copy the schema file with historical harnesses: tool registration and argument
/// typing are request inputs, not parser-version differences.
pub fn unified_tools() -> Vec<dynamo_parsers_v2::Tool> {
    serde_json::from_value(unified_tool_schemas()).expect("Unified corpus tool schemas")
}

pub fn unified_tool_schemas() -> serde_json::Value {
    serde_json::from_str(include_str!("../../utils/src/unified_tools.json"))
        .expect("Unified corpus tool schemas")
}

pub trait UnifiedEventView: PartialEq {
    fn reasoning_text(&self) -> Option<&str>;
    fn visible_text(&self) -> Option<&str>;
    fn tool_call(&self) -> Option<(&str, &Value)>;
}

pub fn classify_unified_events<E: UnifiedEventView>(
    family: &str,
    golden: &[E],
    got: &[E],
) -> &'static str {
    if golden == got {
        return "MATCH";
    }
    const MARKERS: &[&str] = &[
        "<|",
        "|>",
        "<think>",
        "</think>",
        "◁",
        "<channel",
        "channel|>",
    ];
    let family_leak = unified_family(family).leak_markers;
    let leaks = got.iter().any(|event| {
        event
            .visible_text()
            .or_else(|| event.reasoning_text())
            .is_some_and(|text| {
                MARKERS
                    .iter()
                    .copied()
                    .chain(family_leak.iter().map(String::as_str))
                    .any(|marker| text.contains(marker))
            })
    });
    if leaks {
        return "LEAK";
    }
    if got.iter().filter_map(E::reasoning_text).count()
        < golden.iter().filter_map(E::reasoning_text).count()
    {
        return "MERGE";
    }
    let calls = |events: &[E]| -> Vec<(String, Value)> {
        events
            .iter()
            .filter_map(|event| {
                event
                    .tool_call()
                    .map(|(name, arguments)| (name.to_string(), arguments.clone()))
            })
            .collect()
    };
    let (golden_calls, got_calls) = (calls(golden), calls(got));
    if golden_calls.len() == got_calls.len()
        && golden_calls
            .iter()
            .zip(&got_calls)
            .all(|(golden_call, got_call)| golden_call.0 == got_call.0)
        && golden_calls
            .iter()
            .zip(&got_calls)
            .any(|(golden_call, got_call)| golden_call.1 != got_call.1)
    {
        return "ARG_MISMATCH";
    }
    let concatenate = |events: &[E], reasoning: bool| -> String {
        events
            .iter()
            .filter_map(|event| {
                if reasoning {
                    event.reasoning_text()
                } else {
                    event.visible_text()
                }
            })
            .collect()
    };
    if golden_calls == got_calls
        && concatenate(golden, true) == concatenate(got, true)
        && concatenate(golden, false) == concatenate(got, false)
    {
        return "ORDER";
    }
    "LOSS"
}

pub fn validate_current_dynamo_expectation(
    verdict: Option<&str>,
    expected_class: Option<&str>,
    note: Option<&str>,
    actual_class: &str,
) -> Result<(), String> {
    match (verdict, actual_class) {
        (None, "MATCH") => Ok(()),
        (None, actual_class) => Err(format!(
            "unexplained current-native divergence {actual_class}; add expect.dynamo_current with the exact class and a note"
        )),
        (Some("diverge"), "MATCH") => Err(
            "documented current-native divergence now matches; remove the stale exception".into(),
        ),
        (Some("diverge"), actual_class) => {
            let expected_class = expected_class
                .filter(|class| !class.trim().is_empty())
                .ok_or_else(|| "documented current-native divergence has no class".to_string())?;
            note.filter(|note| !note.trim().is_empty())
                .ok_or_else(|| "documented current-native divergence has no note".to_string())?;
            if actual_class != expected_class {
                return Err(format!(
                    "documented current-native {expected_class} but observed {actual_class}"
                ));
            }
            Ok(())
        }
        (Some(other), _) => Err(format!(
            "expect.dynamo_current must document a divergence, found verdict {other:?}"
        )),
    }
}

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
/// The committed canonical family YAML is DERIVED from this via render -> explode
/// -> package. Each test process copies the generated tree to its own immutable
/// directory while holding the lock, because the generator truncates files before
/// rewriting them and another test binary may start as soon as the lock is released.
/// Panics with the fix command if generation fails.
pub fn ensure_unified_golden() -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let script = manifest.join("utils/src/gen_unified_golden.py");
    let generated = manifest.join("unified/golden_spec");
    static GOLDEN_COPY_ID: AtomicUsize = AtomicUsize::new(0);
    let copy_id = GOLDEN_COPY_ID.fetch_add(1, Ordering::Relaxed);
    let isolated = manifest.join(format!(
        "unified/golden_spec-{}-{copy_id}",
        std::process::id()
    ));
    let status = std::process::Command::new("flock")
        .args([
            "/tmp/dynamo-unified-golden.lock",
            "sh",
            "-c",
            "python3 \"$GOLDEN_SCRIPT\" && rm -rf \"$GOLDEN_DEST\" && cp -a \"$GOLDEN_SOURCE\" \"$GOLDEN_DEST\"",
        ])
        .env("GOLDEN_SCRIPT", &script)
        .env("GOLDEN_SOURCE", &generated)
        .env("GOLDEN_DEST", &isolated)
        .status()
        .expect("flock/python3 not found — ensure python3 is in PATH");
    if !status.success() {
        panic!(
            "unified golden generation failed (exit {}). Run manually:\n  python3 {}",
            status.code().unwrap_or(-1),
            script.display()
        );
    }
    isolated
}

/// Crate-relative display path for a fixture (for failure messages).
pub fn fixture_name(path: &Path) -> String {
    path.strip_prefix(env!("CARGO_MANIFEST_DIR"))
        .unwrap_or(path)
        .display()
        .to_string()
}

/// Historical stream baseline, not the current source identity. The legacy CURRENT
/// name is retained for stream parity callers; Unified resolves its source separately.
pub const STREAM_DYNAMO_V2_CURRENT_CAPTURE: &str = "dynamo_v2-0.5.1";

// Consumers may reuse verified archives in tagless clones; producers still require tags.
pub const UNIFIED_DYNAMO_V2_CURRENT_CAPTURE: &str = "dynamo_v2-current";

fn dynamo_identity_command() -> std::process::Command {
    let repo = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    // Historical worktrees need the current checker, not their version-only helper.
    let script = std::env::var_os("CONFORMANCE_DYNAMO_PROVENANCE_SCRIPT")
        .map(PathBuf::from)
        .unwrap_or_else(|| repo.join("conformance/utils/src/dynamo_version.py"));
    let mut command = std::process::Command::new("python3");
    command.arg(script).arg("--repo-root").arg(repo);
    command
}

pub fn dynamo_capture_provenance(label: Option<&str>) -> serde_json::Value {
    let mut command = dynamo_identity_command();
    if let Some(label) = label {
        command.arg("--label").arg(label);
    }
    let output = command.output().expect("run Dynamo capture identity check");
    assert!(
        output.status.success(),
        "Dynamo capture identity check failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).expect("Dynamo capture provenance JSON")
}

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
    version_dirs_with_identity_command(root, prefix, current_dir, dynamo_identity_command())
}

fn capture_provenance_inventory(
    root: &Path,
    prefix: &str,
) -> serde_json::Map<String, serde_json::Value> {
    let mut captures = serde_json::Map::new();
    for entry in std::fs::read_dir(root).expect("read capture root") {
        let path = entry.expect("read capture entry").path();
        if !path.is_dir() {
            continue;
        }
        let name = path
            .file_name()
            .unwrap()
            .to_str()
            .expect("capture directory name");
        let Some(version) = name.strip_prefix(prefix) else {
            continue;
        };
        if !version.as_bytes().first().is_some_and(u8::is_ascii_digit) {
            continue;
        }
        for family in std::fs::read_dir(&path).expect("read capture families") {
            let family = family.expect("read capture family").path();
            if !family.is_dir() {
                continue;
            }
            let family_name = family.file_name().unwrap().to_str().unwrap();
            for file in std::fs::read_dir(&family).expect("read capture cases") {
                let file = file.expect("read capture case").path();
                if file.extension().is_none_or(|extension| extension != "yaml") {
                    continue;
                }
                let doc: serde_yaml::Value =
                    serde_yaml::from_slice(&std::fs::read(&file).expect("read capture YAML"))
                        .unwrap_or_else(|error| panic!("{}: {error}", file.display()));
                let provenance = serde_json::to_value(&doc["capture_provenance"])
                    .expect("capture provenance JSON");
                let layer = captures.entry(version.to_string()).or_insert_with(|| {
                    serde_json::json!({
                        "complete_snapshot": path.join("capture-snapshot.json").is_file(),
                        "records": {},
                    })
                });
                let records = layer["records"].as_object_mut().unwrap();
                for key in doc["cases"]
                    .as_mapping()
                    .expect("capture cases mapping")
                    .keys()
                {
                    let key = format!("{family_name}/{}", key.as_str().expect("capture case key"));
                    if let Some(previous) = records.insert(key.clone(), provenance.clone()) {
                        assert_eq!(
                            previous, provenance,
                            "conflicting capture provenance: {key}"
                        );
                    }
                }
            }
        }
    }
    captures
}

fn version_dirs_with_identity_command(
    root: &Path,
    prefix: &str,
    current_dir: &str,
    mut command: std::process::Command,
) -> Vec<PathBuf> {
    let resolved;
    let current_dir = if current_dir == UNIFIED_DYNAMO_V2_CURRENT_CAPTURE {
        let captures = capture_provenance_inventory(root, prefix);
        let mut child = command
            .args(["--select-capture", "--format", "label"])
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .expect("run Dynamo capture selector");
        child
            .stdin
            .take()
            .unwrap()
            .write_all(&serde_json::to_vec(&captures).unwrap())
            .expect("write capture provenance to selector");
        let output = child
            .wait_with_output()
            .expect("wait for Dynamo capture selector");
        assert!(
            output.status.success(),
            "Dynamo capture selection failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let label = String::from_utf8(output.stdout).expect("capture label UTF-8");
        resolved = format!("{prefix}{}", label.trim());
        &resolved
    } else {
        current_dir
    };
    let current = root.join(current_dir);
    let current = if current.is_dir() || current_dir.contains("+source.") {
        current
    } else {
        let patch_prefix = format!("{current_dir}.patch");
        std::fs::read_dir(root)
            .into_iter()
            .flatten()
            .flatten()
            .map(|entry| entry.path())
            .filter(|path| path.is_dir())
            .filter_map(|path| {
                let patch = path
                    .file_name()?
                    .to_str()?
                    .strip_prefix(&patch_prefix)?
                    .parse::<u64>()
                    .ok()?;
                Some((patch, path))
            })
            .max_by_key(|(patch, _)| *patch)
            .map_or(current, |(_, path)| path)
    };
    let current = if current
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.contains("+source."))
    {
        let output = capture_stimulus_command()
            .arg("--select-source-snapshot")
            .arg(&current)
            .output()
            .expect("select complete current source snapshot");
        assert!(
            output.status.success(),
            "source snapshot selection failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        PathBuf::from(
            String::from_utf8(output.stdout)
                .expect("snapshot path UTF-8")
                .trim(),
        )
    } else {
        current
    };
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

/// Use the same snapshot and request-binding validator as the Python renderer.
pub fn capture_stimulus_command() -> std::process::Command {
    let mut command = std::process::Command::new("python3");
    command.arg(Path::new(env!("CARGO_MANIFEST_DIR")).join("utils/src/capture_stimulus.py"));
    command
}

#[cfg(test)]
mod capture_selector_tests {
    use super::*;

    fn git(root: &Path, args: &[&str]) -> String {
        let output = std::process::Command::new("git")
            .arg("-C")
            .arg(root)
            .args(args)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        String::from_utf8(output.stdout).unwrap().trim().to_string()
    }

    fn checker(repo: &Path) -> std::process::Command {
        let mut command = dynamo_identity_command();
        command
            .arg("--repo-root")
            .arg(repo)
            .env_remove("CONFORMANCE_DYNAMO_V2_LABEL");
        command
    }

    fn select(root: &Path, repo: &Path, label: Option<&str>) -> Vec<PathBuf> {
        let mut command = checker(repo);
        if let Some(label) = label {
            command.env("CONFORMANCE_DYNAMO_V2_LABEL", label);
        }
        version_dirs_with_identity_command(
            root,
            "dynamo_v2-",
            UNIFIED_DYNAMO_V2_CURRENT_CAPTURE,
            command,
        )
    }

    fn write_capture_with_record(
        dir: &Path,
        provenance: &serde_json::Value,
        record: serde_json::Value,
    ) {
        std::fs::create_dir_all(dir.join("gemma4")).unwrap();
        std::fs::write(
            dir.join("gemma4/probe.yaml"),
            serde_json::to_string(&serde_json::json!({
                "family": "gemma4",
                "capture_provenance": provenance,
                "cases": {"probe": record}
            }))
            .unwrap(),
        )
        .unwrap();
    }

    fn write_capture(dir: &Path, provenance: &serde_json::Value) {
        write_capture_with_record(dir, provenance, serde_json::json!({}));
    }

    #[test]
    fn tagless_rust_consumer_reuses_verified_release() {
        let scratch =
            std::env::temp_dir().join(format!("dynamo-tagless-selector-{}", std::process::id()));
        let repo = scratch.join("release");
        std::fs::create_dir_all(repo.join("parsers/v2/src")).unwrap();
        std::fs::write(
            repo.join("parsers/v2/Cargo.toml"),
            "[package]\nname='dynamo-parsers-v2'\nversion='0.6.0'\n",
        )
        .unwrap();
        std::fs::write(repo.join("parsers/v2/src/lib.rs"), "pub fn parser() {}\n").unwrap();
        git(&repo, &["init", "-q"]);
        git(&repo, &["add", "."]);
        let tree = git(&repo, &["write-tree"]);
        let commit = git(
            &repo,
            &[
                "-c",
                "user.name=Test",
                "-c",
                "user.email=test@example.invalid",
                "commit-tree",
                &tree,
                "-m",
                "fixture",
            ],
        );
        git(&repo, &["update-ref", "HEAD", &commit]);
        git(
            &repo,
            &["update-ref", "refs/tags/dynamo-parsers-v2-v0.6.0", &commit],
        );
        let output = checker(&repo).output().unwrap();
        assert!(output.status.success());
        let recorded: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(recorded["label"], "0.6.0");
        let clone = scratch.join("shallow");
        git(
            &repo,
            &[
                "clone",
                "--depth=1",
                "--no-tags",
                &format!("file://{}", repo.display()),
                clone.to_str().unwrap(),
            ],
        );
        assert_eq!(
            git(&clone, &["rev-parse", "--is-shallow-repository"]),
            "true"
        );
        assert_eq!(git(&clone, &["tag", "--list"]), "");
        let patch_only = scratch.join("patch-only");
        let historical = patch_only.join("dynamo_v2-0.5.3");
        write_capture_with_record(
            &historical,
            &recorded,
            serde_json::json!({
                "assembled": [],
                "capture_input": {
                    "input": "probe",
                    "init": {
                        "starting_state": "None",
                        "tool_output_mode": "Native",
                        "named_tool": null
                    },
                    "finish_reason": "stop",
                    "tools": [],
                    "chunks": [{"delta_text": "probe"}, {"delta_text": "‹finish›"}]
                }
            }),
        );
        assert!(std::panic::catch_unwind(|| select(&patch_only, &repo, None)).is_err());
        assert!(std::panic::catch_unwind(|| select(&patch_only, &clone, None)).is_err());
        let release_patch2 = patch_only.join("dynamo_v2-0.6.0.patch2");
        let release_patch10 = patch_only.join("dynamo_v2-0.6.0.patch10");
        write_capture(&release_patch2, &recorded);
        write_capture(&release_patch10, &recorded);
        assert_eq!(
            select(&patch_only, &repo, None).last(),
            Some(&release_patch10)
        );
        assert_eq!(
            select(&patch_only, &clone, None).last(),
            Some(&release_patch10)
        );
        let captures = scratch.join("unified");
        let release = captures.join("dynamo_v2-0.6.0");
        write_capture(&release, &recorded);
        let selected = select(&captures, &clone, None);
        assert_eq!(selected.last(), Some(&release));
        assert!(
            !checker(&clone)
                .args(["--label", "0.6.0"])
                .output()
                .unwrap()
                .status
                .success()
        );
        for override_label in ["current", "0.6.0"] {
            assert!(
                std::panic::catch_unwind(|| select(&captures, &clone, Some(override_label)))
                    .is_err()
            );
        }
        write_capture(&release, &serde_json::Value::Null);
        assert!(std::panic::catch_unwind(|| select(&captures, &clone, None)).is_err());
        write_capture(&release, &recorded);
        let patch = captures.join("dynamo_v2-0.6.0.patch1");
        let mut wrong = recorded.clone();
        wrong["source_id"] = serde_json::json!("wrong source");
        write_capture(&patch, &wrong);
        assert!(std::panic::catch_unwind(|| select(&captures, &clone, None)).is_err());
        write_capture(&patch, &recorded);
        assert_eq!(select(&captures, &clone, None).last(), Some(&release));

        let output = checker(&clone).output().unwrap();
        let current: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
        let qualified = captures.join(format!("dynamo_v2-{}", current["label"].as_str().unwrap()));
        write_capture(&qualified, &current);
        assert_eq!(select(&captures, &clone, None).last(), Some(&qualified));
        assert_eq!(
            select(&captures, &clone, Some("current")).last(),
            Some(&qualified)
        );
        assert!(select(&captures, &clone, None).contains(&release));

        let source_patch = qualified.with_file_name(format!(
            "{}.patch10",
            qualified.file_name().unwrap().to_str().unwrap()
        ));
        write_capture(&source_patch, &current);
        std::fs::write(
            source_patch.join("capture-snapshot.json"),
            r#"{"schema_version":1,"records":["gemma4/probe.yaml"]}"#,
        )
        .unwrap();
        assert_eq!(select(&captures, &clone, None).last(), Some(&source_patch));

        let stream = captures.join(STREAM_DYNAMO_V2_CURRENT_CAPTURE);
        std::fs::create_dir_all(&stream).unwrap();
        let stream_dirs = version_dirs_with_identity_command(
            &captures,
            "dynamo_v2-",
            STREAM_DYNAMO_V2_CURRENT_CAPTURE,
            std::process::Command::new("this-command-must-not-run"),
        );
        assert_eq!(stream_dirs.last(), Some(&stream));
        std::fs::write(clone.join("parsers/v2/src/lib.rs"), "pub fn changed() {}\n").unwrap();
        assert!(std::panic::catch_unwind(|| select(&captures, &clone, None)).is_err());
    }
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

#[cfg(not(any(conformance_legacy_init, conformance_split_only)))]
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
        self.try_apply(parser)
            .unwrap_or_else(|e| panic!("{what}: initialize_request {self:?}: {e}"));
    }

    pub fn try_apply(
        &self,
        parser: &mut Box<dyn dynamo_parsers_v2::UnifiedParser>,
    ) -> anyhow::Result<()> {
        use dynamo_parsers_v2::{InvalidGuidedPayloadPolicy, UnifiedParserInit};
        parser.initialize_request(UnifiedParserInit {
            starting_state: self.starting_state(),
            tool_output_mode: self.output_mode(),
            invalid_guided_payload: InvalidGuidedPayloadPolicy::RecoverAsText,
            ..UnifiedParserInit::default()
        })
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
