// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::{Context, Result};
use base64::{Engine, engine::general_purpose::STANDARD};
use conformance_publisher::{
    codec::{self, json_bytes},
    pipeline, source, transaction, transport,
};
use serde_json::{Value, json};
use std::{
    collections::BTreeSet,
    fs,
    path::{Path, PathBuf},
    process::{Command, Output},
};

struct Sandbox(PathBuf);
impl Sandbox {
    fn new() -> Result<Self> {
        Ok(Self(transaction::unique(
            &std::env::temp_dir(),
            "native-publication-test-",
        )?))
    }
}
impl Drop for Sandbox {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
fn put(root: &Path, path: &str, bytes: &[u8]) -> Result<()> {
    pipeline::write(root, path, bytes)
}
fn sorted(value: &Value) -> Value {
    match value {
        Value::Object(values) => Value::Object(
            values
                .iter()
                .map(|(k, v)| (k.clone(), sorted(v)))
                .collect::<std::collections::BTreeMap<_, _>>()
                .into_iter()
                .collect(),
        ),
        Value::Array(values) => Value::Array(values.iter().map(sorted).collect()),
        value => value.clone(),
    }
}
fn fixture(root: &Path) -> Result<()> {
    put(root, "index.html", b"<head></head><body></body>")?;
    put(
        root,
        "conformance/CONFORMANCE_v2.html",
        b"<a href=\"../\">viewer</a>",
    )?;
    let family =
        b"input_document: {family: sample}\ngolden_document: {family: sample}\ncases: {}\n";
    let path = "conformance/fixtures-unified-v2/families/sample/inputs_and_golden.yaml";
    put(root, path, family)?;
    put(
        root,
        "conformance/fixtures-manifest.json",
        b"{\"shards\":[]}",
    )?;
    let mut catalog = json!({"schema":1,"snapshot":"fixture","sources":{"archives":[],"unified":[{"path":path,"sha256":codec::sha256(family),"size":family.len(),"kind":"inputs-and-golden","family":"sample","implementation":null,"version":null}],"candidates":[]},"metadata":{"tabs":[],"source_hashes":{}}});
    catalog["revision"] = json!(codec::sha256(&serde_json::to_vec(&sorted(&catalog))?));
    put(
        root,
        "conformance/viewer-catalog.json",
        &json_bytes(&catalog)?,
    )?;
    put(
        root,
        "conformance/viewer-unified-catalog.json",
        &json_bytes(&pipeline::projection(&catalog)?)?,
    )?;
    Ok(())
}
fn run(root: &Path, output: &Path, extra: &[&str]) -> Result<Output> {
    let scratch = tempfile::tempdir()?;
    let result = Command::new(env!("CARGO_BIN_EXE_conformance-publish"))
        .args([
            "--root",
            root.to_str().unwrap(),
            "--output",
            output.to_str().unwrap(),
            "--revision",
            &"a".repeat(40),
        ])
        .args(extra)
        .env("TMPDIR", scratch.path())
        .output()?;
    for entry in fs::read_dir(scratch.path())? {
        assert!(
            !entry?
                .file_name()
                .as_encoded_bytes()
                .starts_with(b"unified-source-"),
            "the completed publisher retained an admitted source archive"
        );
    }
    Ok(result)
}

#[test]
fn archive_members_publish_as_exact_originals_and_extra_members_are_rejected() -> Result<()> {
    let sandbox = Sandbox::new()?;
    let root = sandbox.0.join("source");
    fixture(&root)?;
    let name = "receipts/producer.yaml";
    let bytes = b"version: 0.1.0\r\n";
    let mut archive = tar::Builder::new(Vec::new());
    let mut header = tar::Header::new_ustar();
    header.set_size(bytes.len() as u64);
    header.set_mode(0o644);
    header.set_cksum();
    archive.append_data(&mut header, name, bytes.as_slice())?;
    let encoded = codec::gzip(&archive.into_inner()?)?;
    let archive_path = format!("{}/versions/dynamo-0.1.0.tar.gz", source::STORE);
    put(&root, &archive_path, &encoded)?;
    let path = format!("{}/{name}", source::STORE);
    let mut catalog: Value =
        serde_json::from_slice(&fs::read(root.join("conformance/viewer-catalog.json"))?)?;
    catalog["sources"]["unified"].as_array_mut().unwrap().push(
        json!({"path":path,"size":bytes.len(),"sha256":codec::sha256(bytes),
        "kind":"receipt","family":null,"implementation":null,"version":null,
        "archive":{"path":archive_path,"size":encoded.len(),"sha256":codec::sha256(&encoded)}}),
    );
    catalog.as_object_mut().unwrap().remove("revision");
    catalog["revision"] = json!(codec::sha256(&serde_json::to_vec(&sorted(&catalog))?));
    put(
        &root,
        "conformance/viewer-catalog.json",
        &json_bytes(&catalog)?,
    )?;
    put(
        &root,
        "conformance/viewer-unified-catalog.json",
        &json_bytes(&pipeline::projection(&catalog)?)?,
    )?;
    let output = sandbox.0.join("published");
    let result = run(&root, &output, &["--workers", "1"])?;
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let runtime = compressed_json(&output, "conformance/viewer-resolved-catalog.json.gz")?;
    let mut consumed = BTreeSet::new();
    let bucket = &codec::sha256(path.as_bytes())[..2];
    let raw = read_index(
        &output,
        &runtime["transport"]["raw_buckets"][bucket],
        &mut consumed,
    )?;
    let original = &raw["resources"][&path]["resource"];
    assert!(original.get("archive").is_none());
    assert_eq!(
        consume_recipe(&output, &original["recipe"], &mut consumed)?,
        bytes
    );
    // Removing its descriptor must leave an unreferenced archive, which admission rejects.
    catalog["sources"]["unified"].as_array_mut().unwrap().pop();
    catalog.as_object_mut().unwrap().remove("revision");
    catalog["revision"] = json!(codec::sha256(&serde_json::to_vec(&sorted(&catalog))?));
    put(
        &root,
        "conformance/viewer-catalog.json",
        &json_bytes(&catalog)?,
    )?;
    put(
        &root,
        "conformance/viewer-unified-catalog.json",
        &json_bytes(&pipeline::projection(&catalog)?)?,
    )?;
    assert!(!run(&root, &output, &["--workers", "1"])?.status.success());
    Ok(())
}

fn compressed_json(root: &Path, path: &str) -> Result<Value> {
    let bytes = fs::read(root.join(path))?;
    Ok(serde_json::from_slice(&codec::gunzip(
        &bytes,
        &codec::sha256(&bytes),
        16 * 1024 * 1024,
    )?)?)
}

fn read_index(root: &Path, descriptor: &Value, consumed: &mut BTreeSet<String>) -> Result<Value> {
    let path = descriptor["path"].as_str().unwrap();
    let encoded = fs::read(root.join(path))?;
    assert_eq!(encoded.len() as u64, descriptor["size"].as_u64().unwrap());
    let decoded = codec::gunzip(
        &encoded,
        descriptor["sha256"].as_str().unwrap(),
        descriptor["decoded_size"].as_u64().unwrap() as usize,
    )?;
    assert_eq!(
        decoded.len() as u64,
        descriptor["decoded_size"].as_u64().unwrap()
    );
    assert_eq!(
        codec::sha256(&decoded),
        descriptor["decoded_sha256"].as_str().unwrap()
    );
    consumed.insert(path.to_string());
    Ok(serde_json::from_slice(&decoded)?)
}

fn consume_recipe(root: &Path, recipe: &Value, consumed: &mut BTreeSet<String>) -> Result<Vec<u8>> {
    let bytes = transport::reconstruct(root, recipe)?;
    for chunk in recipe["chunks"].as_array().unwrap() {
        consumed.insert(chunk["pack"]["path"].as_str().unwrap().to_string());
    }
    Ok(bytes)
}

#[test]
fn publish_is_lossless_deterministic_and_failure_preserves_output() -> Result<()> {
    let sandbox = Sandbox::new()?;
    let root = sandbox.0.join("source");
    fixture(&root)?;
    let one = sandbox.0.join("one");
    let two = sandbox.0.join("two");
    for (output, workers) in [(&one, "1"), (&two, "2")] {
        let result = run(&root, output, &["--workers", workers])?;
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        let measurements = String::from_utf8(result.stderr)?
            .lines()
            .filter(|line| line.starts_with('{'))
            .map(serde_json::from_str::<Value>)
            .collect::<std::result::Result<Vec<_>, _>>()?;
        for measurement in &measurements {
            assert_eq!(measurement["timing_scope"], "inclusive");
            assert!(measurement["phase_cpu_seconds"].as_f64().unwrap() >= 0.0);
            assert!(
                measurement["process_cpu_seconds"].as_f64().unwrap()
                    >= measurement["phase_cpu_seconds"].as_f64().unwrap()
            );
        }
        let phase_cpu = |phase| {
            measurements
                .iter()
                .find(|measurement| measurement["phase"] == phase)
                .unwrap()["phase_cpu_seconds"]
                .as_f64()
                .unwrap()
        };
        assert!(phase_cpu("durable-install") >= phase_cpu("source-recheck"));
    }
    let catalog_path = "conformance/viewer-resolved-catalog.json.gz";
    assert_eq!(
        fs::read(one.join(catalog_path))?,
        fs::read(two.join(catalog_path))?
    );
    for relative in [
        "conformance/viewer-catalog.json",
        "conformance/viewer-unified-catalog.json",
        "conformance/viewer-resolved-catalog.json",
        "conformance/viewer-startup.json",
        "conformance/viewer-data/bootstrap.json.gz",
        "conformance/viewer-data/inputs-and-golden.json.gz",
        "conformance/viewer-data/bundles/inputs-and-golden.json.gz",
        ".native-envelopes",
        ".native-parts",
        "conformance/viewer-data/originals",
    ] {
        assert!(
            !one.join(relative).exists(),
            "redundant publication remains: {relative}"
        );
    }
    let catalog = compressed_json(&one, catalog_path)?;
    let html = fs::read_to_string(one.join("index.html"))?;
    let embedded: Value = serde_json::from_str(
        html.split("id=\"conformance-embedded\">")
            .nth(1)
            .context("missing embedded data")?
            .split("</script>")
            .next()
            .unwrap(),
    )?;
    let embedded_catalog = STANDARD.decode(embedded["catalog"]["bytes"].as_str().unwrap())?;
    assert_eq!(embedded_catalog, fs::read(one.join(catalog_path))?);
    for resource in embedded["resources"].as_array().unwrap() {
        assert_eq!(
            STANDARD.decode(resource["bytes"].as_str().unwrap())?,
            source::verified(&one, &resource["descriptor"])?
        );
    }
    assert!(!embedded["resources"].as_array().unwrap().is_empty());
    assert_eq!(catalog["transport"]["schema"], 2);
    let family = &catalog["sources"]["unified"].as_array().unwrap()[0];
    assert_eq!(family["transport"], "chunked-v2");
    assert_eq!(catalog["sources"]["unified"].as_array().unwrap().len(), 1);
    let mut consumed = [
        "index.html",
        "conformance/CONFORMANCE_v2.html",
        "conformance/viewer-publication.json",
        catalog_path,
    ]
    .into_iter()
    .map(str::to_string)
    .collect::<BTreeSet<_>>();
    let inputs = read_index(&one, &family["recipe_index"], &mut consumed)?;
    let family_recipe = &inputs["resources"][family["path"].as_str().unwrap()];
    let family_bytes = consume_recipe(&one, family_recipe, &mut consumed)?;
    assert_eq!(
        codec::sha256(&family_bytes),
        family["sha256"].as_str().unwrap()
    );
    assert_eq!(family_bytes.len() as u64, family["size"].as_u64().unwrap());
    let data_only = run(&root, &root, &["--data-only", "--workers", "1"])?;
    assert!(
        data_only.status.success(),
        "{}",
        String::from_utf8_lossy(&data_only.stderr)
    );
    assert!(!root.join("conformance/viewer-startup.json").exists());
    assert!(!root.join("conformance/viewer-data/bootstrap").exists());
    assert!(
        !root
            .join("conformance/viewer-data/bundles/inputs-and-golden.json.gz")
            .exists()
    );
    let checkout_catalog = compressed_json(&root, "conformance/viewer-resolved-catalog.json.gz")?;
    let checkout_family = checkout_catalog["sources"]["unified"]
        .as_array()
        .unwrap()
        .iter()
        .find(|resource| resource["kind"] == "inputs-and-golden" && resource["family"] == "sample")
        .context("missing checkout family resource")?;
    let baseline_bytes = fs::read(root.join(checkout_family["path"].as_str().unwrap()))?;
    assert_eq!(
        family_bytes,
        codec::gunzip(
            &baseline_bytes,
            &codec::sha256(&baseline_bytes),
            1024 * 1024
        )?
    );
    let source = "conformance/fixtures-unified-v2/families/sample/inputs_and_golden.yaml";
    assert!(!one.join(source).exists());
    let original = fs::read(root.join(source))?;
    let bucket = &codec::sha256(source.as_bytes())[..2];
    let raw = read_index(
        &one,
        &catalog["transport"]["raw_buckets"][bucket],
        &mut consumed,
    )?;
    let raw_resource = &raw["resources"][source]["resource"];
    assert_eq!(raw_resource["sha256"], codec::sha256(&original));
    assert_eq!(
        consume_recipe(&one, &raw_resource["recipe"], &mut consumed)?,
        original
    );
    for descriptor in catalog["transport"]["raw_buckets"]
        .as_object()
        .unwrap()
        .values()
    {
        let index = read_index(&one, descriptor, &mut consumed)?;
        for entry in index["resources"].as_object().unwrap().values() {
            if let Some(resource) = entry.get("resource") {
                consume_recipe(&one, &resource["recipe"], &mut consumed)?;
            }
        }
    }
    let files = source::inventory(&one)?
        .into_iter()
        .filter(|path| one.join(path).is_file())
        .collect::<BTreeSet<_>>();
    assert_eq!(
        files, consumed,
        "every published file must have a runtime consumer"
    );
    for relative in consumed.iter().filter(|path| {
        path.as_str() == catalog_path || path.starts_with("conformance/viewer-data/")
    }) {
        assert_eq!(
            fs::read(one.join(relative))?,
            fs::read(two.join(relative))?,
            "nondeterministic transport file: {relative}"
        );
    }
    let before = fs::read(one.join("index.html"))?;
    put(&root, source, b"corrupted: true\n")?;
    let result = run(&root, &one, &["--workers", "1"])?;
    assert!(!result.status.success());
    assert_eq!(fs::read(one.join("index.html"))?, before);
    Ok(())
}

#[test]
fn paths_topology_and_data_only_are_admitted_before_install() -> Result<()> {
    let sandbox = Sandbox::new()?;
    let root = sandbox.0.join("source");
    fixture(&root)?;
    let result = run(&root, &root.join("conformance/viewer"), &[])?;
    assert!(!result.status.success());
    let outside = sandbox.0.join("outside");
    fs::create_dir(&outside)?;
    std::os::unix::fs::symlink(&outside, sandbox.0.join("linked"))?;
    assert!(
        !run(&root, &sandbox.0.join("linked/site"), &[])?
            .status
            .success()
    );
    fs::create_dir(root.join("conformance/fixtures-unified-v2/unknown"))?;
    assert!(!run(&root, &outside.join("site"), &[])?.status.success());
    fs::remove_dir(root.join("conformance/fixtures-unified-v2/unknown"))?;
    let original = fs::read(root.join("conformance/viewer-catalog.json"))?;
    let result = run(&root, &root, &["--data-only", "--workers", "1"])?;
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(
        fs::read(root.join("conformance/viewer-catalog.json"))?,
        original
    );
    let resolved = compressed_json(&root, "conformance/viewer-resolved-catalog.json.gz")?;
    let family = resolved["sources"]["unified"]
        .as_array()
        .unwrap()
        .iter()
        .find(|resource| resource["kind"] == "inputs-and-golden" && resource["family"] == "sample")
        .context("missing checkout family resource")?;
    assert!(root.join(family["path"].as_str().unwrap()).is_file());
    Ok(())
}

#[test]
fn prepared_python_journal_rolls_back_and_committed_journal_keeps_new_generation() -> Result<()> {
    for committed in [false, true] {
        let sandbox = Sandbox::new()?;
        let destination = sandbox.0.join("site");
        fs::create_dir(&destination)?;
        fs::write(destination.join("value"), b"old")?;
        let backup = sandbox.0.join(".conformance-transaction-backup-a1b2c3d4");
        fs::create_dir(&backup)?;
        let saved = backup.join("00-site");
        fs::rename(&destination, &saved)?;
        fs::create_dir(&destination)?;
        fs::write(destination.join("value"), b"new")?;
        let journal = json!({"schema_version":1,"phase":if committed {"committed"} else {"prepared"},"backup_root":backup.file_name().unwrap().to_str(),"states":[{"candidate":"stage/site","destination":"site","backup":format!("{}/00-site",backup.file_name().unwrap().to_string_lossy()),"had_original":true}]});
        fs::write(
            sandbox.0.join(".conformance-transaction.json"),
            json_bytes(&journal)?,
        )?;
        let _lease = transaction::Lease::acquire(&sandbox.0)?;
        assert_eq!(
            fs::read(destination.join("value"))?,
            if committed { b"new" } else { b"old" }
        );
        assert!(!backup.exists());
        assert!(!sandbox.0.join(".conformance-transaction.json").exists());
    }
    Ok(())
}

#[test]
fn complete_snapshots_scope_metadata_without_mutating_or_expanding_sources() -> Result<()> {
    let sandbox = Sandbox::new()?;
    let root = sandbox.0.join("source");
    fixture(&root)?;
    let authored = b"input_document: {family: sample}\ngolden_document: {family: sample}\ncases:\n  id:\n    lifecycle: active\n    scenario: sample\n    description: sample\n    policy: []\n    display_id: UNIFIED.1\n    historical_ids: []\n    request: {input: sample, init: {}, finish_reason: stop, tools: [], chunks: []}\n    golden: {assembled: []}\n";
    let authored_path = format!("{}/families/sample/inputs_and_golden.yaml", source::STORE);
    put(&root, &authored_path, authored)?;
    let member = format!("{}/families/sample/dynamo_v2-0.1.0.yaml", source::STORE);
    let capture = format!(
        "schema_version: 2\nprovenance: {{status: legacy, captured_with: {{dynamo_v2: '0.1.0'}}}}\ndocument: {{}}\nrecords:\n  id:\n    member: {member}\n    value:\n      case_key: UNIFIED.1\n      stimulus: {{ref: current}}\n      observation: {{value: {{assembled: []}}}}\n      document: {{captured_with: {{dynamo_v2: '0.1.0'}}, record_metadata: {{retained: null, verified_recapture: {{receipt: checkpoint}}}}}}\n"
    );
    put(&root, &member, capture.as_bytes())?;
    let mut catalog: Value =
        serde_json::from_slice(&fs::read(root.join("conformance/viewer-catalog.json"))?)?;
    catalog["sources"]["unified"] = json!([
        {"path":authored_path,"size":authored.len(),"sha256":codec::sha256(authored),"kind":"inputs-and-golden","family":"sample","implementation":null,"version":null},
        {"path":member,"size":capture.len(),"sha256":codec::sha256(capture.as_bytes()),"kind":"history","family":"sample","implementation":"dynamo_v2","version":"0.1.0"}
    ]);
    catalog["sources"]["candidates"] = json!([
        {"implementation":"dynamo_v2","mode":"unified","version":"0.1.0"},
        {"implementation":"dynamo_v2","mode":"unified","version":"0.1.1"}
    ]);
    catalog["metadata"]["unified_capture_policy"] = json!({"dynamo_v2":{"mode":"stream"}});
    catalog.as_object_mut().unwrap().remove("revision");
    catalog["revision"] = json!(codec::sha256(&serde_json::to_vec(&sorted(&catalog))?));
    put(
        &root,
        "conformance/viewer-catalog.json",
        &json_bytes(&catalog)?,
    )?;
    put(
        &root,
        "conformance/viewer-unified-catalog.json",
        &json_bytes(&pipeline::projection(&catalog)?)?,
    )?;
    let before = source::inventory(&root.join(source::STORE))?;
    let receipt = sandbox.0.join("validation.json");
    let validation = run(&root, &receipt, &["--validate-unified", "--workers", "4"])?;
    assert!(
        validation.status.success(),
        "{}",
        String::from_utf8_lossy(&validation.stderr)
    );
    assert_eq!(
        serde_json::from_slice::<Value>(&fs::read(receipt)?)?["validated"],
        true
    );
    let mut outputs = Vec::new();
    for workers in ["1", "4"] {
        let output = root.clone();
        let result = run(&root, &output, &["--workers", workers, "--data-only"])?;
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        let resolved: Value = serde_json::from_slice(&fs::read(
            output.join("conformance/viewer-resolved-catalog.json"),
        )?)?;
        let bundle_path = |version: &str| -> Result<&str> {
            resolved["sources"]["unified"]
                .as_array()
                .context("missing resolved resources")?
                .iter()
                .find(|resource| {
                    resource["kind"] == "resolved-capture"
                        && resource["implementation"] == "dynamo_v2"
                        && resource["version"] == version
                })
                .and_then(|resource| resource["path"].as_str())
                .context("missing resolved capture bundle path")
        };
        let first = compressed_json(&output, bundle_path("0.1.0")?)?;
        let later = compressed_json(&output, bundle_path("0.1.1")?)?;
        let first = serde_json::to_string(&first)?;
        let later = serde_json::to_string(&later)?;
        assert!(first.contains("verified_recapture"));
        assert!(!later.contains("verified_recapture"));
        assert!(later.contains("retained"));
        outputs.push((first, later));
    }
    assert_eq!(outputs[0], outputs[1]);
    // The publication lock is adjacent to conformance, never in the evidence tree.
    assert_eq!(before, source::inventory(&root.join(source::STORE))?);
    assert_eq!(fs::read(root.join(&member))?, capture.as_bytes());
    assert_eq!(fs::read(root.join(&authored_path))?, authored);
    Ok(())
}
