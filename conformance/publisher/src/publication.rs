// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Validated source generation, web-only assembly and transactional installation.
use crate::{
    codec::{canonical_json_bytes as catalog_identity, json_bytes, sha256},
    pipeline,
    resources::{MemoryBudget, Policy},
    source::{self, STORE},
    transaction::{self, Lease},
};
use anyhow::{Context, Result, ensure};
use base64::{Engine, engine::general_purpose::STANDARD};
use rayon::prelude::*;
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Component, Path, PathBuf},
    process::Command,
    time::Instant,
};

#[derive(Clone)]
pub struct Options {
    pub root: PathBuf,
    pub output: PathBuf,
    pub revision: String,
    pub candidate_tree: Option<String>,
    pub data_only: bool,
    pub workers: String,
}

fn absolute(path: &Path) -> Result<PathBuf> {
    let path = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()?.join(path)
    };
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::ParentDir => {
                ensure!(normalized.pop(), "path escapes root");
            }
            Component::CurDir => (),
            other => normalized.push(other.as_os_str()),
        }
    }
    source::no_links(&path)?;
    source::no_links(&normalized)?;
    Ok(normalized)
}

fn disjoint(left: &Path, right: &Path) -> bool {
    !left.starts_with(right) && !right.starts_with(left)
}

fn validate_destination(
    root: &Path,
    output: &Path,
    data_only: bool,
    check_existing: bool,
) -> Result<()> {
    ensure!(
        !root.starts_with(output) || (data_only && output == root),
        "output overlaps source checkout"
    );
    ensure!(
        !data_only || output == root,
        "data-only destination must be source checkout"
    );
    if !data_only {
        for name in [
            ".git",
            "Cargo.toml",
            "Cargo.lock",
            "parsers",
            "conformance/fixtures-manifest.json",
            "conformance/viewer",
            "conformance/utils",
            "conformance/inputs",
            "conformance/fixtures",
            "conformance/fixtures-unified-v2",
            "index.html",
            "conformance/CONFORMANCE_v2.html",
            "conformance/viewer-catalog.json",
            "conformance/viewer-unified-catalog.json",
        ] {
            ensure!(
                disjoint(output, &root.join(name)),
                "output overlaps maintained source"
            );
        }
        if check_existing && output.exists() {
            ensure!(
                output.is_dir(),
                "existing site destination must be a directory"
            );
            ensure!(
                !output.starts_with(root)
                    || output.join("conformance/viewer-publication.json").is_file()
                    || fs::read_to_string(output.join("index.html"))
                        .is_ok_and(|html| html.contains("id=\"conformance-embedded\"")),
                "existing destination overlaps source checkout"
            );
        }
    }
    Ok(())
}

fn capture_inventory(
    root: &Path,
    catalog: &Value,
    budget: &MemoryBudget,
) -> Result<BTreeSet<String>> {
    if let Some(datasets) = catalog["sources"]["history_datasets"].as_array() {
        let mut plain = catalog.clone();
        plain["sources"]
            .as_object_mut()
            .unwrap()
            .remove("history_datasets");
        plain["sources"]["unified"] = json!(
            catalog["sources"]["unified"]
                .as_array()
                .context("missing unified inventory")?
                .iter()
                .filter(|row| row["dataset_root"].is_null())
                .collect::<Vec<_>>()
        );
        let mut inventory = capture_inventory(root, &plain, budget)?
            .into_iter()
            .map(|name| format!("{STORE}/{name}"))
            .collect::<BTreeSet<_>>();
        let mut roots = BTreeSet::new();
        for dataset in datasets {
            let prefix = dataset["root"].as_str().context("missing dataset root")?;
            source::relative(prefix)?;
            ensure!(roots.insert(prefix), "duplicate dataset root");
            let mut resources = dataset["resources"]
                .as_array()
                .context("missing dataset inventory")?
                .clone();
            for resource in &mut resources {
                ensure!(
                    resource["dataset_root"] == prefix && resource["dataset"] == dataset["id"],
                    "dataset source scope mismatch"
                );
                resource
                    .as_object_mut()
                    .context("invalid dataset resource")?
                    .remove("dataset_root");
            }
            plain["sources"]["unified"] = json!(resources);
            inventory.extend(
                capture_inventory(&root.join(prefix), &plain, budget)?
                    .into_iter()
                    .map(|name| format!("{prefix}/{STORE}/{name}")),
            );
        }
        for selected in catalog["sources"]["unified"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|r| !r["dataset_root"].is_null())
        {
            ensure!(
                datasets.iter().any(|dataset| dataset["resources"]
                    .as_array()
                    .is_some_and(|resources| resources.contains(selected))),
                "selected source is outside its dataset inventory"
            );
        }
        return Ok(inventory);
    }
    let inventory = source::inventory(&root.join(STORE))?;
    let expected = catalog["sources"]["unified"]
        .as_array()
        .context("missing unified inventory")?
        .iter()
        .map(|r| {
            let name = r.get("archive").unwrap_or(r)["path"]
                .as_str()
                .context("missing resource path")?;
            Ok(name
                .strip_prefix(&format!("{STORE}/"))
                .context("unified resource outside store")?
                .to_string())
        })
        .collect::<Result<BTreeSet<_>>>()?;
    ensure!(
        catalog["sources"]["unified"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|r| r["path"].as_str())
            .collect::<BTreeSet<_>>()
            .len()
            == catalog["sources"]["unified"].as_array().unwrap().len(),
        "duplicate inventory resource"
    );
    for name in &inventory {
        let path = root.join(STORE).join(name);
        let parts = name.split('/').collect::<Vec<_>>();
        if path.is_dir() {
            ensure!(
                matches!(
                    parts.as_slice(),
                    ["families"] | ["executions"] | ["receipts"] | ["versions"] | ["families", _]
                ),
                "unknown store directory: {name}"
            );
            if parts.len() == 2 {
                ensure!(
                    path.join("inputs_and_golden.yaml").is_file(),
                    "family has no authored input: {name}"
                );
            }
        } else {
            ensure!(
                matches!(
                    parts.as_slice(),
                    ["families", _, _] | ["executions", _] | ["receipts", _]
                ) && name.ends_with(".yaml")
                    || matches!(parts.as_slice(), ["versions", _]) && name.ends_with(".tar.gz"),
                "unknown store file: {name}"
            );
        }
    }
    let files = inventory
        .iter()
        .filter(|name| root.join(STORE).join(name).is_file())
        .cloned()
        .collect::<BTreeSet<_>>();
    ensure!(files == expected, "source inventory differs from catalog");
    let mut checked: BTreeMap<&str, (&Value, BTreeSet<String>, u64)> = BTreeMap::new();
    for resource in catalog["sources"]["unified"].as_array().unwrap() {
        if let Some(archive) = resource.get("archive") {
            let name = archive["path"]
                .as_str()
                .context("missing source archive path")?;
            let member = resource["path"]
                .as_str()
                .context("missing archived resource path")?
                .strip_prefix(&format!("{STORE}/"))
                .context("archived member outside source store")?
                .to_string();
            let size = resource["size"]
                .as_u64()
                .context("missing archived resource size")?;
            let (_, members, max_member_size) = checked
                .entry(name)
                .or_insert_with(|| (resource, BTreeSet::new(), 0));
            ensure!(members.insert(member), "duplicate archived source");
            *max_member_size = (*max_member_size).max(size);
        }
    }
    checked.par_iter().try_for_each(
        |(expected_archive_path, (resource, members, member_size))| {
            let archive = resource.get("archive").context("missing source archive")?;
            let archive_path = archive["path"]
                .as_str()
                .context("missing source archive path")?;
            ensure!(
                archive_path == *expected_archive_path,
                "archive path changed during inventory"
            );
            let archive_bytes = source::source_size(root, expected_archive_path)?;
            let _reservation = budget.reserve(source::archive_reservation_bytes(
                archive_bytes,
                *member_size,
            )?)?;
            let actual = source::archived_inventory(root, resource)?;
            ensure!(
                &actual == members,
                "archived source inventory differs from catalog"
            );
            Ok(())
        },
    )?;
    Ok(inventory)
}

fn manifest_identity(root: &Path, catalog: &Value) -> Result<Vec<u8>> {
    let bytes = source::read(root, "conformance/fixtures-manifest.json")?;
    let manifest: Value = serde_json::from_slice(&bytes)?;
    let shards = manifest["shards"]
        .as_array()
        .context("missing fixture manifest shards")?;
    let mut archives = BTreeMap::new();
    for shard in shards {
        let name = shard["path"].as_str().context("missing shard path")?;
        source::relative(name)?;
        if shard["format"] == "unified-history" {
            ensure!(
                &catalog["sources"]["unified_history"] == shard,
                "history manifest differs from catalog"
            );
        } else {
            archives.insert(format!("conformance/fixtures/{name}"), shard);
        }
    }
    ensure!(
        archives.len()
            == catalog["sources"]["archives"]
                .as_array()
                .context("missing archive sources")?
                .len(),
        "manifest archive membership differs from catalog"
    );
    for resource in catalog["sources"]["archives"].as_array().unwrap() {
        let shard = archives
            .get(resource["path"].as_str().context("missing archive path")?)
            .context("archive missing from manifest")?;
        ensure!(
            shard["sha256"] == resource["sha256"] && shard["size"] == resource["size"],
            "archive manifest identity differs from catalog"
        );
    }
    if let Some(shard) = catalog["sources"]
        .get("unified_history")
        .filter(|v| !v.is_null())
    {
        let mut rows = catalog["sources"]["unified"].as_array().context("missing Unified sources")?.iter().map(|r| Ok(json!({"path":r["path"].as_str().context("missing source path")?.strip_prefix(&format!("{STORE}/")).context("source outside store")?,"sha256":r["sha256"],"size":r["size"]}))).collect::<Result<Vec<_>>>()?;
        rows.sort_by(|a, b| a["path"].as_str().cmp(&b["path"].as_str()));
        ensure!(
            shard["sha256"] == sha256(&catalog_identity(&json!(rows))?)
                && shard["size"].as_u64()
                    == Some(rows.iter().map(|r| r["size"].as_u64().unwrap()).sum()),
            "history inventory digest differs from manifest"
        );
    }
    Ok(bytes)
}

fn metadata_sources(root: &Path, catalog: &Value) -> Result<BTreeMap<String, String>> {
    let fixed = [
        (
            "parser_families",
            "conformance/utils/src/parser_families.yaml",
        ),
        ("legacy_taxonomy", "conformance/case-taxonomy.yaml"),
        (
            "unified_taxonomy",
            "conformance/utils/src/unified_taxonomy.py",
        ),
        ("implementations", "conformance/utils/src/impls.py"),
        ("peer_versions", "conformance/utils/src/pyproject.stub.toml"),
        ("parser_config", "parsers/v1/src/tool_calling/config.rs"),
        (
            "divergence_notes",
            "conformance/toolcalling/known-divergences.yaml",
        ),
        (
            "unified_authoring",
            "conformance/utils/src/gen_unified_golden.py",
        ),
        ("unified_tools", "conformance/utils/src/unified_tools.json"),
        (
            "parser_registrations",
            "parsers/v1/src/tool_calling/parsers.rs",
        ),
    ];
    let hashes = &catalog["metadata"]["source_hashes"];
    let mut sources = BTreeMap::new();
    for (key, path) in fixed {
        if let Some(digest) = hashes.get(key) {
            sources.insert(
                path.to_string(),
                digest
                    .as_str()
                    .context("invalid metadata hash")?
                    .to_string(),
            );
        }
    }
    if let Some(docs) = hashes["documentation"].as_object() {
        for (path, digest) in docs {
            sources.insert(
                path.clone(),
                digest
                    .as_str()
                    .context("invalid document hash")?
                    .to_string(),
            );
        }
    }
    for (path, digest) in &sources {
        ensure!(
            sha256(&source::read(root, path)?) == *digest,
            "metadata source identity mismatch: {path}"
        );
    }
    Ok(sources)
}

fn asset_graph(root: &Path) -> Result<BTreeMap<String, Vec<u8>>> {
    let html = regex::Regex::new(r#"(?:src|href)=["']([^"']+)["']"#)?;
    let js = regex::Regex::new(r#"(?:from\s+|import\s*(?:\(\s*)?|new URL\(\s*)["'](\.[^"']+)"#)?;
    let css = regex::Regex::new(r#"url\(["']?([^"')]+)"#)?;
    let inline =
        regex::Regex::new(r"(?is)(<script\b[^>]*>).*?</script>|<style\b[^>]*>.*?</style>")?;
    let mut pending = vec![
        "index.html".to_string(),
        "conformance/CONFORMANCE_v2.html".to_string(),
    ];
    let mut files = BTreeMap::new();
    while let Some(name) = pending.pop() {
        if files.contains_key(&name) {
            continue;
        }
        ensure!(
            ![".yaml", ".yml", ".tar.gz", ".md", ".txt", ".test.js"]
                .iter()
                .any(|suffix| name.ends_with(suffix)),
            "forbidden web artifact: {name}"
        );
        let mut bytes = source::read(root, &name)?;
        if name == "index.html" {
            bytes = String::from_utf8(bytes)?
                .replace(
                    "conformance/viewer/development.js",
                    "conformance/viewer/app.js",
                )
                .into_bytes();
        }
        if [".html", ".js", ".css"]
            .iter()
            .any(|suffix| name.ends_with(suffix))
        {
            let text = std::str::from_utf8(&bytes)?;
            let pattern = if name.ends_with(".html") {
                &html
            } else if name.ends_with(".js") {
                &js
            } else {
                &css
            };
            let dependencies = if name.ends_with(".html") {
                inline.replace_all(text, "$1").into_owned()
            } else {
                text.to_owned()
            };
            for capture in pattern.captures_iter(&dependencies) {
                let link = capture[1].split(['?', '#']).next().unwrap();
                if link.is_empty() || link.contains(':') || link.starts_with("//") {
                    continue;
                }
                ensure!(
                    !link.starts_with('/') && !link.contains('%'),
                    "nonrelative or encoded runtime asset: {link}"
                );
                let parent = Path::new(&name).parent().context("asset has no parent")?;
                let mut normalized = PathBuf::new();
                for component in parent.join(link).components() {
                    match component {
                        Component::CurDir => (),
                        Component::ParentDir => {
                            ensure!(normalized.pop(), "runtime asset escapes site");
                        }
                        Component::Normal(c) => normalized.push(c),
                        _ => anyhow::bail!("runtime asset escapes site"),
                    }
                }
                let child = normalized.to_str().context("nonUTF8 asset")?.to_string();
                if root.join(&child).is_dir() && name.ends_with(".html") {
                    continue;
                }
                if root.join(&child).is_file() {
                    pending.push(child);
                } else {
                    anyhow::bail!("referenced runtime asset missing: {child}");
                }
            }
        }
        files.insert(name, bytes);
    }
    Ok(files)
}

fn bundle_assets_with_installation(
    root: &Path,
    assets: &BTreeMap<String, Vec<u8>>,
    installation: Option<&Path>,
) -> Result<BTreeMap<String, Vec<u8>>> {
    if !assets.contains_key("conformance/viewer/app.js") {
        return Ok(assets.clone());
    }
    let mut command = Command::new("node");
    command
        .arg(root.join("conformance/viewer/build.mjs"))
        .current_dir(root);
    if let Some(installation) = installation {
        command.env("CONFORMANCE_ESBUILD_ROOT", installation);
    }
    let output = command
        .output()
        .context("launch locked production JavaScript bundler")?;
    ensure!(
        output.status.success(),
        "production bundler failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let result: Value = serde_json::from_slice(&output.stdout)?;
    ensure!(
        result["schema"] == 1,
        "unsupported production bundle schema"
    );
    for input in result["inputs"]
        .as_array()
        .context("missing bundle dependency graph")?
    {
        let path = input.as_str().context("invalid bundle dependency")?;
        ensure!(
            assets.contains_key(path),
            "bundle consumed an undeclared development dependency: {path}"
        );
    }
    let bundles = result["files"]
        .as_object()
        .context("missing production bundles")?;
    ensure!(bundles.len() == 2, "unexpected production bundles");
    let mut published = assets
        .iter()
        .filter(|(path, _)| !path.ends_with(".js"))
        .map(|(path, bytes)| (path.clone(), bytes.clone()))
        .collect::<BTreeMap<_, _>>();
    for path in ["conformance/viewer/app.js", "conformance/viewer/worker.js"] {
        let bytes = STANDARD.decode(
            bundles
                .get(path)
                .and_then(Value::as_str)
                .context("missing production bundle")?,
        )?;
        ensure!(!bytes.is_empty(), "empty production bundle");
        published.insert(path.to_string(), bytes);
    }
    let mut index = String::from_utf8(published["index.html"].clone())?;
    let external_css = regex::Regex::new(r"(?i)@import|url\s*\(|</style")?;
    for (path, bytes) in &published {
        if !path.ends_with(".css") {
            continue;
        }
        let stylesheet = std::str::from_utf8(bytes)?;
        ensure!(
            !external_css.is_match(stylesheet),
            "standalone stylesheet has an external dependency: {path}"
        );
        let link = format!("<link rel=\"stylesheet\" href=\"{path}\">");
        ensure!(
            index.matches(&link).count() == 1,
            "unexpected stylesheet link"
        );
        index = index.replacen(&link, &format!("<style>{stylesheet}</style>"), 1);
    }
    let script = "<script type=\"module\" src=\"conformance/viewer/app.js\"></script>";
    ensure!(index.matches(script).count() == 1, "unexpected app script");
    let application = std::str::from_utf8(&published["conformance/viewer/app.js"])?;
    let escaped = regex::Regex::new(r"(?i)</script")?.replace_all(application, "<\\/script");
    index = index.replacen(
        script,
        &format!("<script type=\"module\">{escaped}</script>"),
        1,
    );
    published.retain(|path, _| path.ends_with(".html"));
    published.insert("index.html".into(), index.into_bytes());
    Ok(published)
}

fn validate_catalog(catalog: &Value) -> Result<()> {
    ensure!(catalog["schema"] == 1, "unsupported catalog schema");
    let mut identity = catalog.clone();
    let revision = identity
        .as_object_mut()
        .context("invalid catalog")?
        .remove("revision")
        .context("missing catalog revision")?;
    ensure!(
        revision == sha256(&catalog_identity(&identity)?),
        "catalog revision mismatch"
    );
    let mut paths = BTreeSet::new();
    for kind in ["unified", "archives"] {
        for resource in catalog["sources"][kind]
            .as_array()
            .context("missing sources")?
        {
            let path = resource["path"].as_str().context("missing source path")?;
            source::relative(path)?;
            ensure!(paths.insert(path), "duplicate source path");
            if kind == "archives" {
                ensure!(
                    path.starts_with("conformance/fixtures/")
                        && path.ends_with(".tar.gz")
                        && resource["members"].is_array(),
                    "invalid archive source descriptor"
                );
            }

            ensure!(
                resource["sha256"]
                    .as_str()
                    .is_some_and(|v| source::hex(v, 64))
                    && resource["size"]
                        .as_u64()
                        .is_some_and(|n| n < source::MAX_ARTIFACT),
                "invalid source descriptor"
            );
        }
    }
    for resource in catalog["sources"]["unified"].as_array().unwrap() {
        let name = resource["path"]
            .as_str()
            .unwrap()
            .strip_prefix(&format!("{STORE}/"))
            .context("Unified source outside store")?;
        let parts = name.split('/').collect::<Vec<_>>();
        match parts.as_slice() {
            ["executions", file] | ["receipts", file] => {
                let kind = if parts[0] == "executions" {
                    "execution"
                } else {
                    "receipt"
                };
                ensure!(
                    resource["kind"] == kind
                        && resource["family"].is_null()
                        && resource["implementation"].is_null()
                        && resource["version"].is_null()
                        && file.ends_with(".yaml"),
                    "shared source identity mismatch"
                );
            }
            ["families", family, file] => {
                ensure!(
                    resource["family"] == *family,
                    "source family identity mismatch"
                );
                if *file == "inputs_and_golden.yaml" {
                    ensure!(
                        resource["kind"] == "inputs-and-golden"
                            && resource["implementation"].is_null()
                            && resource["version"].is_null(),
                        "authored source identity mismatch"
                    );
                } else {
                    let stem = file
                        .strip_suffix(".yaml")
                        .context("unknown family source extension")?;
                    let (implementation, version) = stem
                        .split_once('-')
                        .context("invalid capture source identity")?;
                    crate::history::Release::parse(version)?;
                    ensure!(
                        implementation
                            .bytes()
                            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_')
                            && resource["kind"] == "history"
                            && resource["implementation"] == implementation
                            && resource["version"] == version,
                        "capture source identity mismatch"
                    );
                }
            }
            _ => anyhow::bail!("invalid Unified source topology"),
        }
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct PhaseClock {
    wall: Instant,
    cpu_seconds: f64,
}

fn usage_cpu_seconds(usage: &libc::rusage) -> f64 {
    usage.ru_utime.tv_sec as f64
        + usage.ru_utime.tv_usec as f64 / 1e6
        + usage.ru_stime.tv_sec as f64
        + usage.ru_stime.tv_usec as f64 / 1e6
}

impl PhaseClock {
    fn now() -> Result<Self> {
        let mut usage = std::mem::MaybeUninit::<libc::rusage>::zeroed();
        ensure!(
            unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) } == 0,
            "cannot measure publication CPU: {}",
            std::io::Error::last_os_error()
        );
        Ok(Self {
            wall: Instant::now(),
            cpu_seconds: usage_cpu_seconds(&unsafe { usage.assume_init() }),
        })
    }
}

fn statistics(start: PhaseClock, phase: &str, extra: Value) {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::zeroed();
    let status = unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) };
    let mut result = json!({"phase":phase,"wall_seconds":start.wall.elapsed().as_secs_f64(),"timing_scope":"inclusive","details":extra});
    if status == 0 {
        let usage = unsafe { usage.assume_init() };
        let cpu_seconds = usage_cpu_seconds(&usage);
        result["process_cpu_seconds"] = json!(cpu_seconds);
        result["phase_cpu_seconds"] = json!(cpu_seconds - start.cpu_seconds);
        result["peak_process_memory_bytes"] = json!(usage.ru_maxrss as u64 * 1024);
    } else {
        result["statistics_error"] = json!(std::io::Error::last_os_error().to_string());
    }
    eprintln!("{result}");
}

fn publication_time() -> Result<String> {
    let now = unsafe { libc::time(std::ptr::null_mut()) };
    let mut local = std::mem::MaybeUninit::<libc::tm>::uninit();
    ensure!(
        !unsafe { libc::localtime_r(&now, local.as_mut_ptr()) }.is_null(),
        "cannot format publication time"
    );
    let mut buffer = [0i8; 64];
    let format = c"%Y-%m-%d %H:%M %Z";
    let length = unsafe {
        libc::strftime(
            buffer.as_mut_ptr(),
            buffer.len(),
            format.as_ptr(),
            local.as_ptr(),
        )
    };
    ensure!(length > 0, "cannot format publication time");
    Ok(unsafe { std::ffi::CStr::from_ptr(buffer.as_ptr()) }
        .to_str()?
        .to_string())
}

fn command(options: &Options, root: &Path, output: &Path) -> Result<String> {
    let guard = root.join("conformance/capture-source-guard.json");
    let prepared: Option<Value> = if guard.is_file() {
        Some(serde_json::from_slice(&fs::read(guard)?)?)
    } else {
        None
    };
    let mut args = vec![
        "conformance-publish".to_string(),
        "--root".into(),
        prepared
            .as_ref()
            .and_then(|value| value["root"].as_str())
            .unwrap_or(root.to_str().context("non UTF8 root")?)
            .into(),
        "--output".into(),
        output.to_str().context("non UTF8 output")?.into(),
        "--revision".into(),
        options.revision.clone(),
    ];
    if let Some(history) = prepared
        .as_ref()
        .and_then(|value| value["history_source"].as_str())
    {
        args.extend([
            "--include-history".into(),
            "--history-source".into(),
            history.into(),
        ]);
    }
    if let Some(tree) = &options.candidate_tree {
        args.extend(["--candidate-tree".into(), tree.clone()]);
    }
    if options.data_only {
        args.push("--data-only".into());
    }
    if options.workers != "auto" {
        args.extend(["--workers".into(), options.workers.clone()]);
    }
    Ok(args
        .iter()
        .map(|arg| {
            if arg
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"_@%+=:,./-".contains(&b))
            {
                arg.clone()
            } else {
                format!("'{}'", arg.replace('\'', "'\\''"))
            }
        })
        .collect::<Vec<_>>()
        .join(" "))
}

fn validation_catalog(root: &Path, memory_bytes: u64) -> Result<(Value, u64)> {
    let catalog_size = source::source_size(root, "conformance/viewer-catalog.json")?;
    let reserved = catalog_size
        .checked_mul(32)
        .context("catalog reservation overflow")?;
    let control_budget = MemoryBudget::new(reserved);
    let _control = control_budget.reserve(reserved)?;
    let _job_memory = memory_bytes
        .checked_sub(reserved)
        .context("catalog exceeds validation memory budget")?;
    let catalog = serde_json::from_slice(&source::read_bounded(
        root,
        "conformance/viewer-catalog.json",
        catalog_size,
    )?)?;
    Ok((catalog, reserved))
}

/// Validate permanent evidence without preparing or installing report resources.
pub fn validate(root: &Path, workers: &str) -> Result<Value> {
    let root = absolute(root)?;
    let _source_lease = Lease::read(&root.join("conformance"))?;
    let _session = source::archive_session();
    let policy = Policy::detect(workers)?;
    let pool = policy.pool()?;
    let (catalog, reserved) = validation_catalog(&root, policy.memory_bytes)?;
    let budget = MemoryBudget::new(
        policy
            .memory_bytes
            .checked_sub(reserved)
            .context("catalog exceeds validation memory budget")?,
    );
    validate_catalog(&catalog)?;
    manifest_identity(&root, &catalog)?;
    let inventory = pool.install(|| capture_inventory(&root, &catalog, &budget))?;
    let stage = tempfile::tempdir()?;
    pool.install(|| {
        pipeline::snapshots(
            &root,
            stage.path(),
            &catalog,
            &budget,
            pipeline::SnapshotEncoding::ValidateOnly,
        )
    })?;
    ensure!(
        pool.install(|| capture_inventory(&root, &catalog, &budget))? == inventory,
        "source changed during validation"
    );
    for resource in catalog["sources"]["unified"].as_array().unwrap() {
        let size = resource["size"].as_u64().context("missing resource size")?;
        let _reservation = budget.reserve(size.max(1024 * 1024))?;
        source::verified(&root, resource)?;
    }
    Ok(
        json!({"validated":true,"source_revision":catalog["revision"],"resources":catalog["sources"]["unified"].as_array().unwrap().len()}),
    )
}

pub fn publish(options: Options) -> Result<()> {
    publish_mode(options, false, None)
}

pub fn publish_mode(
    options: Options,
    include_history: bool,
    history_source: Option<&Path>,
) -> Result<()> {
    publish_mode_cached(options, include_history, history_source, None)
}

pub fn publish_mode_cached(
    mut options: Options,
    include_history: bool,
    history_source: Option<&Path>,
    cache: Option<&Path>,
) -> Result<()> {
    options.root = absolute(&options.root)?.canonicalize()?;
    options.output = absolute(&options.output)?;
    // Preserve the caller's source identity before private preparation changes root.
    validate_destination(&options.root, &options.output, options.data_only, true)?;
    let cache = cache
        .map(Path::to_path_buf)
        .unwrap_or_else(|| options.root.join("build/conformance-cache"));
    let cache = absolute(&cache)?;
    validate_destination(&options.root, &cache, false, false)?;
    if !options.data_only {
        ensure!(
            disjoint(&cache, &options.output),
            "cache overlaps publication output"
        );
    }
    if let Some(history) = history_source {
        let history = absolute(history)?.canonicalize()?;
        if history != options.root {
            ensure!(
                disjoint(&history, &options.output),
                "output overlaps history source"
            );
            ensure!(disjoint(&history, &cache), "cache overlaps history source");
        }
    }
    ensure!(
        include_history || history_source.is_none(),
        "--history-source requires --include-history"
    );
    let resolver = options
        .root
        .join("conformance/utils/src/capture_sources.py");
    let has_latest = if resolver.is_file() {
        let result = Command::new(
            std::env::var_os("CONFORMANCE_PYTHON").unwrap_or_else(|| "python3".into()),
        )
        .arg(&resolver)
        .arg("--root")
        .arg(&options.root)
        .arg("--has-latest")
        .output()?;
        ensure!(result.status.success(), "latest capture discovery failed");
        serde_json::from_slice::<bool>(&result.stdout)?
    } else {
        false
    };
    if has_latest {
        let bootstrap = if include_history {
            Some(tempfile::tempdir()?)
        } else {
            None
        };
        let bootstrap_site = bootstrap
            .as_ref()
            .map(|directory| directory.path().join("latest"));
        if let Some(site) = &bootstrap_site {
            let mut latest = options.clone();
            latest.output = site.clone();
            publish_split(latest, false, None, &cache, None, false)?;
        }
        return publish_split(
            options,
            include_history,
            history_source,
            &cache,
            bootstrap_site.as_deref(),
            !include_history,
        );
    }
    ensure!(
        !include_history,
        "history mode requires a split capture checkout"
    );
    publish_prepared(options, None, &cache, None, false)
}

fn publish_split(
    mut options: Options,
    include_history: bool,
    history_source: Option<&Path>,
    cache: &Path,
    bootstrap: Option<&Path>,
    minimal: bool,
) -> Result<()> {
    ensure!(
        !options.data_only,
        "split capture publication requires a separate output directory"
    );
    let temporary = tempfile::tempdir()?;
    let mut command =
        Command::new(std::env::var_os("CONFORMANCE_PYTHON").unwrap_or_else(|| "python3".into()));
    command
        .arg(
            options
                .root
                .join("conformance/utils/src/capture_sources.py"),
        )
        .arg("--root")
        .arg(&options.root)
        .arg("--prepare")
        .arg(temporary.path())
        .arg("--publisher")
        .arg(std::env::current_exe()?)
        .env("CONFORMANCE_CONVERSION_CACHE", cache);
    if include_history {
        command.arg("--include-history");
    }
    if let Some(source) = history_source {
        command.arg("--history-source").arg(source);
    }
    ensure!(
        command.status()?.success(),
        "capture source preparation failed"
    );
    // The prepared view contains source modules, while the locked package
    // installation remains in the caller's checkout.
    let installation = std::env::var_os("CONFORMANCE_ESBUILD_ROOT")
        .map(PathBuf::from)
        .unwrap_or(options.root.join("conformance/viewer"));
    let installation = absolute(&installation)?;
    options.root = temporary.path().to_path_buf();
    publish_prepared(options, Some(&installation), cache, bootstrap, minimal)
}

fn publish_prepared(
    options: Options,
    installation: Option<&Path>,
    cache: &Path,
    bootstrap: Option<&Path>,
    minimal: bool,
) -> Result<()> {
    let _archives = source::archive_session();
    let started = PhaseClock::now()?;
    ensure!(
        source::hex(&options.revision, 40),
        "revision must be a full immutable commit SHA"
    );
    if let Some(tree) = &options.candidate_tree {
        ensure!(
            source::hex(tree, 40),
            "candidate tree must be a full Git tree SHA"
        );
    }
    let root = absolute(&options.root)?.canonicalize()?;
    let output = absolute(&options.output)?;
    validate_destination(&root, &output, options.data_only, true)?;
    fs::create_dir_all(output.parent().context("missing output parent")?)?;
    let transaction_parent = if options.data_only {
        root.clone()
    } else {
        output.parent().unwrap().to_path_buf()
    };
    let source_parent = root.join("conformance");
    let _source_lease = if options.data_only || transaction_parent == source_parent {
        Lease::acquire(&source_parent)?
    } else {
        Lease::read(&source_parent)?
    };
    let _output_lease = if transaction_parent != root.join("conformance") {
        Some(Lease::acquire(&transaction_parent)?)
    } else {
        None
    };
    if output.exists() && !options.data_only {
        transaction::sync_tree(&output)?;
    }
    let policy = Policy::detect(&options.workers)?;
    let pool = policy.pool()?;
    let control_bytes = source::source_size(&root, "conformance/viewer-catalog.json")?
        .checked_mul(32)
        .and_then(|n| n.checked_add(256 * 1024 * 1024))
        .context("catalog memory reservation overflow")?;
    let control_budget = MemoryBudget::new(control_bytes);
    let _control = control_budget.reserve(control_bytes)?;
    let job_memory = policy
        .memory_bytes
        .checked_sub(control_bytes)
        .filter(|n| *n > 0)
        .context("catalog and assets exceed publication memory budget")?;
    let budget = MemoryBudget::new(job_memory);
    let catalog_bytes = source::read(&root, "conformance/viewer-catalog.json")?;
    let catalog: Value = serde_json::from_slice(&catalog_bytes)?;
    validate_catalog(&catalog)?;
    let manifest = manifest_identity(&root, &catalog)?;
    let projection_path = root.join("conformance/viewer-unified-catalog.json");
    let original_projection = if projection_path.exists() {
        let bytes = source::read(&root, "conformance/viewer-unified-catalog.json")?;
        ensure!(
            serde_json::from_slice::<Value>(&bytes)? == pipeline::projection(&catalog)?,
            "source projection mismatch"
        );
        Some(bytes)
    } else {
        None
    };
    let metadata = metadata_sources(&root, &catalog)?;
    let inventory = pool.install(|| capture_inventory(&root, &catalog, &budget))?;
    let assets = asset_graph(&root)?;
    let build_inputs = if !options.data_only && assets.contains_key("conformance/viewer/app.js") {
        [
            "conformance/viewer/build.mjs",
            "conformance/viewer/package.json",
            "conformance/viewer/package-lock.json",
            "conformance/viewer/vendor/fflate/LICENSE",
        ]
        .into_iter()
        .map(|path| Ok((path, source::read(&root, path)?)))
        .collect::<Result<BTreeMap<_, _>>>()?
    } else {
        BTreeMap::new()
    };
    let web_assets = if options.data_only {
        assets.clone()
    } else {
        bundle_assets_with_installation(&root, &assets, installation)?
    };
    for path in metadata.keys().chain(assets.keys()) {
        let path = root.join(path);
        ensure!(
            options.data_only || (!path.starts_with(&output) && !output.starts_with(&path)),
            "output overlaps source input"
        );
    }
    statistics(
        started,
        "admission",
        json!({"detected_cpus":policy.detected_cpus,"workers":policy.workers,"memory_budget":policy.memory_bytes,"control_reserved_bytes":control_bytes,"job_memory_budget":job_memory,"allocator_arenas":policy.allocator_arenas}),
    );
    let stage = transaction::unique(&transaction_parent, ".native-publication-stage-")?;
    let result = (|| -> Result<()> {
        let phase = PhaseClock::now()?;
        let resolved = pool.install(|| {
            pipeline::snapshots_cached(
                &root,
                &stage,
                &catalog,
                &budget,
                if options.data_only {
                    pipeline::SnapshotEncoding::CheckoutGzip
                } else {
                    pipeline::SnapshotEncoding::CanonicalSpool
                },
                Some(cache),
            )
        })?;
        statistics(
            phase,
            "resolve-encode",
            json!({"resolved_resources":resolved["sources"]["unified"].as_array().map(Vec::len)}),
        );
        let phase = PhaseClock::now()?;
        let embedded = if options.data_only {
            let mut projection = pipeline::projection(&resolved)?;
            crate::web_transport::relocate_checkout_generation(&stage, &mut projection)?;
            pipeline::write(
                &stage,
                "conformance/viewer-resolved-catalog.json",
                &json_bytes(&projection)?,
            )?;
            pipeline::write(
                &stage,
                "conformance/viewer-resolved-catalog.json.gz",
                &crate::codec::gzip(&json_bytes(&projection)?)?,
            )?;
            None
        } else {
            let runtime = pool.install(|| {
                crate::web_transport::assemble(&root, &stage, &catalog, &resolved, &budget)
            })?;
            if bootstrap.is_none() {
                Some(crate::web_transport::embedded(&stage, &runtime, &budget)?)
            } else {
                None
            }
        };
        let mut publication = json!({"schema":1,"revision":options.revision,"generated_revision":options.revision,"catalog_revision":if options.data_only {&catalog["revision"]} else {&resolved["revision"]},"generated_on":publication_time()?,"command":command(&options,&root,&output)?,"output":output.to_str(),"resolved_catalog_revision":resolved["revision"]});
        if let Some(tree) = &options.candidate_tree {
            publication["candidate_tree"] = json!(tree);
        }
        if let Some(generation) = catalog.get("capture_generation") {
            publication["generation_mode"] = generation["mode"].clone();
            publication["current_captures"] = generation["current"].clone();
            publication["history_pin"] = generation["history_pin"].clone();
            let guard: Value = serde_json::from_slice(&source::read(
                &root,
                "conformance/capture-source-guard.json",
            )?)?;
            publication["publisher_inputs_sha256"] = json!(sha256(&json_bytes(
                &json!({"inputs":guard["inputs"],"controls":guard["controls"]})
            )?));
        }
        if options.data_only {
            publication["source_checkout"] = json!(true);
        }
        if let Some(site) = bootstrap {
            let latest: Value = serde_json::from_slice(&source::read(
                site,
                "conformance/viewer-publication.json",
            )?)?;
            let guard: Value = serde_json::from_slice(&source::read(
                &root,
                "conformance/capture-source-guard.json",
            )?)?;
            let controls = guard["controls"]
                .as_array()
                .context("missing source controls")?
                .iter()
                .filter(|control| control["path"] != "history-catalog")
                .cloned()
                .collect::<Vec<_>>();
            let latest_inputs = sha256(&json_bytes(
                &json!({"inputs":guard["inputs"],"controls":controls}),
            )?);
            ensure!(
                latest["current_captures"] == publication["current_captures"]
                    && latest["publisher_inputs_sha256"] == latest_inputs
                    && latest["revision"] == publication["revision"],
                "latest sources changed between bootstrap and history publication"
            );
            // Bind optional history to the exact latest catalog embedded in either report.
            publication["catalog_revision"] = latest["catalog_revision"].clone();
        }
        pipeline::write(
            &stage,
            "conformance/viewer-publication.json",
            &json_bytes(&publication)?,
        )?;
        statistics(phase, "resolved-assembly", json!({}));
        if !options.data_only {
            let phase = PhaseClock::now()?;
            for (path, bytes) in &web_assets {
                pipeline::write(&stage, path, bytes)?;
            }
            let index = String::from_utf8(web_assets["index.html"].clone())?;
            ensure!(
                index.contains("</head>") && !index.contains("id=\"conformance-publication\""),
                "unexpected publication index"
            );
            let page = if let Some(site) = bootstrap {
                String::from_utf8(source::read(site, "index.html")?)?
            } else {
                // Invocation paths, timestamps and history policy belong in the sidecar.
                let identity = json!({"schema":1,"revision":publication["revision"],"catalog_revision":publication["catalog_revision"],"resolved_catalog_revision":publication["resolved_catalog_revision"]});
                let inline = serde_json::to_string(&identity)?.replace('<', "\\u003c");
                let embedded = serde_json::to_string(
                    &embedded
                        .as_ref()
                        .context("missing embedded resources")?
                        .value,
                )?
                .replace('<', "\\u003c");
                index.replacen("</head>", &format!("<script type=\"application/json\" id=\"conformance-publication\">{inline}</script><script type=\"application/json\" id=\"conformance-embedded\">{embedded}</script></head>"), 1)
            };
            pipeline::write(&stage, "index.html", page.as_bytes())?;
            // Both entry points use the same page. History assets resolve from
            // the site root when the report opens under conformance/.
            pipeline::write(
                &stage,
                "conformance/CONFORMANCE_v2.html",
                page.replacen("<head>", "<head><base href=\"../\">", 1)
                    .as_bytes(),
            )?;
            statistics(
                phase,
                "originals-assembly",
                json!({"runtime_assets":web_assets.len()}),
            );
        }
        if !options.data_only {
            // Recheck the compiled graph: source membership cannot prove that
            // bundling left no dangling imports or unused published modules.
            let published_graph = asset_graph(&stage)?;
            ensure!(
                published_graph.keys().eq(web_assets.keys()),
                "compiled runtime dependency graph differs from declared assets"
            );
            let inventory = crate::web_transport::inventory(&stage, web_assets.keys().cloned())?;
            if minimal {
                // Validate every embedded dependency before discarding its duplicate on disk.
                fs::remove_dir_all(stage.join("conformance"))?;
                ensure!(
                    source::inventory(&stage)? == BTreeSet::from(["index.html".into()]),
                    "standalone output contains extra files"
                );
            } else {
                let audit_path = output.with_extension("inventory.json");
                fs::write(audit_path, json_bytes(&inventory)?)?;
            }
        }
        // Source verification reuses this budget after the HTML consumers finish.
        drop(embedded);
        let phase = PhaseClock::now()?;
        let paths = source::inventory(&stage)?;
        let mut bytes = 0u64;
        let mut count = 0;
        for path in paths {
            let path = stage.join(path);
            if path.is_file() {
                let size = fs::metadata(&path)?.len();
                ensure!(size < source::MAX_ARTIFACT, "artifact is at least 1 GB");
                bytes += size;
                count += 1;
            }
        }
        statistics(
            phase,
            "artifact-inventory",
            json!({"files":count,"output_bytes":bytes}),
        );
        let (pairs, retain_on_rollback) = if options.data_only {
            let mut pairs = Vec::new();
            let mut retain_on_rollback = Vec::new();
            let generations = stage.join("conformance/viewer-data/generations");
            let generation = fs::read_dir(&generations)?
                .next()
                .context("resolved catalog has no immutable data generation")??
                .file_name()
                .into_string()
                .map_err(|_| anyhow::anyhow!("invalid data generation name"))?;
            let generation_root = format!("conformance/viewer-data/generations/{generation}");
            ensure!(
                generation_root.starts_with("conformance/viewer-data/generations/"),
                "resolved catalog data generation path is invalid"
            );
            for name in [
                generation_root.as_str(),
                "conformance/viewer-resolved-catalog.json",
                "conformance/viewer-resolved-catalog.json.gz",
                "conformance/viewer-publication.json",
            ] {
                if stage.join(name).exists() {
                    let destination = output.join(name);
                    source::no_links(&destination)?;
                    if name == generation_root {
                        fs::create_dir_all(
                            destination
                                .parent()
                                .context("generation destination has no parent")?,
                        )?;
                        retain_on_rollback.push(destination.clone());
                    }
                    pairs.push((stage.join(name), destination));
                }
            }
            (pairs, retain_on_rollback)
        } else {
            (vec![(stage.clone(), output.clone())], Vec::new())
        };
        let install_started = PhaseClock::now()?;
        transaction::publish_checked(&transaction_parent, &pairs, &retain_on_rollback, || {
            if root.join("conformance/capture-source-guard.json").is_file() {
                ensure!(
                    Command::new(
                        std::env::var_os("CONFORMANCE_PYTHON").unwrap_or_else(|| "python3".into())
                    )
                    .arg(root.join("conformance/utils/src/capture_sources.py"))
                    .arg("--root")
                    .arg(&root)
                    .arg("--verify-prepared")
                    .status()?
                    .success(),
                    "capture source changed during publication"
                );
            }
            let phase = PhaseClock::now()?;
            ensure!(
                source::read(&root, "conformance/fixtures-manifest.json")? == manifest,
                "fixture manifest changed during publication"
            );
            ensure!(
                source::read(&root, "conformance/viewer-catalog.json")? == catalog_bytes,
                "catalog changed during publication"
            );
            if let Some(bytes) = &original_projection {
                ensure!(
                    source::read(&root, "conformance/viewer-unified-catalog.json")? == *bytes,
                    "projection changed during publication"
                );
            }
            ensure!(
                pool.install(|| capture_inventory(&root, &catalog, &budget))? == inventory,
                "source layout changed during publication"
            );
            pool.install(|| {
                catalog["sources"]["unified"]
                    .as_array()
                    .unwrap()
                    .par_iter()
                    .chain(
                        catalog["sources"]["archives"]
                            .as_array()
                            .unwrap()
                            .par_iter(),
                    )
                    .try_for_each(|r| -> Result<()> {
                        let _reservation = budget.reserve(
                            r["size"]
                                .as_u64()
                                .context("invalid resource size")?
                                .checked_mul(2)
                                .context("reread reservation overflow")?
                                .max(1024 * 1024),
                        )?;
                        source::verified(&root, r).map(|_| ())
                    })
            })?;
            for (path, digest) in metadata {
                ensure!(
                    sha256(&source::read(&root, &path)?) == digest,
                    "metadata changed during publication"
                );
            }
            ensure!(
                asset_graph(&root)? == assets,
                "runtime assets changed during publication"
            );
            for (path, bytes) in &build_inputs {
                ensure!(
                    source::read(&root, path)? == *bytes,
                    "production build input changed during publication: {path}"
                );
            }
            statistics(phase, "source-recheck", json!({}));
            Ok(())
        })?;
        statistics(install_started, "durable-install", json!({}));
        statistics(
            started,
            "complete",
            json!({"files":count,"output_bytes":bytes,"workers":policy.workers}),
        );
        Ok(())
    })();
    let phase = PhaseClock::now()?;
    // Never remove a recovery candidate after an indeterminate commit.
    if !transaction_parent
        .join(".conformance-transaction.json")
        .exists()
        && stage.exists()
    {
        fs::remove_dir_all(&stage)?;
    }
    statistics(phase, "cleanup", json!({}));
    let phase = PhaseClock::now()?;
    drop(pool);
    statistics(phase, "pool-shutdown", json!({}));
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_output_and_cache_aliases_fail_before_preparation() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let root = directory.path().join("source");
        fs::create_dir_all(root.join("conformance"))?;
        fs::write(root.join("Cargo.toml"), b"preserve source")?;
        let options = Options {
            root: root.clone(),
            output: directory.path().join("site"),
            revision: "a".repeat(40),
            candidate_tree: None,
            data_only: false,
            workers: "1".into(),
        };
        for output in [
            root.clone(),
            directory.path().to_path_buf(),
            root.join("parsers/new"),
            root.join("conformance/fixtures-unified-v2/new"),
            root.join(".git/new"),
        ] {
            let mut rejected = options.clone();
            rejected.output = output;
            let error = publish_mode_cached(rejected, false, None, None).unwrap_err();
            assert!(error.to_string().contains("overlaps"), "{error:#}");
            assert_eq!(fs::read(root.join("Cargo.toml"))?, b"preserve source");
        }
        for cache in [
            options.output.clone(),
            options.output.join("cache"),
            directory.path().to_path_buf(),
        ] {
            let error =
                publish_mode_cached(options.clone(), false, None, Some(&cache)).unwrap_err();
            assert!(error.to_string().contains("overlaps"), "{error:#}");
        }
        let history = directory.path().join("history");
        fs::create_dir(&history)?;
        for output in [history.clone(), history.join("new-site")] {
            let mut rejected = options.clone();
            rejected.output = output;
            let error = publish_mode_cached(rejected, true, Some(&history), None).unwrap_err();
            assert!(
                error.to_string().contains("overlaps history source"),
                "{error:#}"
            );
        }
        validate_destination(&root, &options.output, false, true)?;
        validate_destination(&root, &root.join("build/site"), false, true)?;
        validate_destination(&root, &root, true, true)?;
        fs::create_dir_all(root.join("build/site/conformance"))?;
        fs::write(
            root.join("build/site/conformance/viewer-publication.json"),
            b"{}",
        )?;
        validate_destination(&root, &root.join("build/site"), false, true)?;
        let standalone = root.join("build/standalone");
        fs::create_dir_all(&standalone)?;
        fs::write(standalone.join("index.html"), b"unrelated source")?;
        assert!(validate_destination(&root, &standalone, false, true).is_err());
        fs::write(
            standalone.join("index.html"),
            b"<script id=\"conformance-embedded\">{}</script>",
        )?;
        validate_destination(&root, &standalone, false, true)?;
        Ok(())
    }

    #[test]
    fn asset_graph_distinguishes_inline_content_from_external_dependencies() -> Result<()> {
        let root = tempfile::tempdir()?;
        pipeline::write(root.path(), "index.html", br#"<style>p::after { content: 'href="missing.css"'; }</style><script>const sample = 'src="missing.js"';</script><script src="asset.js"></script>"#)?;
        pipeline::write(
            root.path(),
            "conformance/CONFORMANCE_v2.html",
            b"<a href=\"../\">Report</a>",
        )?;
        pipeline::write(root.path(), "asset.js", b"self.value = 1;")?;
        let graph = asset_graph(root.path())?;
        assert_eq!(graph.len(), 3);
        assert!(graph.contains_key("asset.js"));
        fs::remove_file(root.path().join("asset.js"))?;
        assert!(
            asset_graph(root.path())
                .unwrap_err()
                .to_string()
                .contains("referenced runtime asset missing")
        );
        Ok(())
    }

    #[test]
    fn validation_admits_catalog_before_reading_and_parsing() -> Result<()> {
        let root = tempfile::tempdir()?;
        let catalog = root.path().join("conformance/viewer-catalog.json");
        fs::create_dir_all(catalog.parent().unwrap())?;
        fs::write(&catalog, b"not JSON")?;

        let error = validation_catalog(root.path(), 1).unwrap_err().to_string();

        assert!(error.contains("catalog exceeds validation memory budget"));
        Ok(())
    }

    #[test]
    fn capture_inventory_reserves_before_cold_archive_admission() -> Result<()> {
        let root = tempfile::tempdir()?;
        let archive_name = format!("{STORE}/versions/dynamo-0.0.0.tar.gz");
        let archive = root.path().join(&archive_name);
        fs::create_dir_all(archive.parent().unwrap())?;
        fs::write(&archive, b"not a gzip archive")?;
        let catalog = json!({"sources":{"unified":[{
            "path":format!("{STORE}/families/sample/dynamo_v2-0.0.0.yaml"),
            "size":18,
            "archive":{"path":archive_name}
        }]}});
        let budget = MemoryBudget::new(8 * 1024 * 1024);

        let error = capture_inventory(root.path(), &catalog, &budget)
            .unwrap_err()
            .to_string();

        assert!(error.contains("requires 16777252 bytes"));
        Ok(())
    }
}
