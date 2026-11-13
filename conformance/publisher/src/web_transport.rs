// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Assemble only runtime-consumed indexes and shared payload packs.
use crate::{
    codec::{self, json_bytes, sha256},
    history::Release,
    pipeline,
    resources::{MemoryBudget, Reservation},
    source,
    transport::PackWriter,
};
use anyhow::{Context, Result, ensure};
use base64::{Engine, engine::general_purpose::STANDARD};
use rayon::prelude::*;
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::Path,
};

fn index(stage: &Path, path: &str, value: &Value) -> Result<Value> {
    let decoded = json_bytes(value)?;
    let bytes = codec::gzip(&decoded)?;
    pipeline::write(stage, path, &bytes)?;
    Ok(
        json!({"path":path,"sha256":sha256(&bytes),"size":bytes.len(),"decoded_size":decoded.len(),"decoded_sha256":sha256(&decoded)}),
    )
}

fn recent_captures(catalog: &Value) -> Result<BTreeSet<(String, String)>> {
    if catalog["capture_generation"]["mode"] == "latest" {
        return catalog["sources"]["candidates"]
            .as_array()
            .context("missing candidates")?
            .iter()
            .filter(|c| c["mode"] == "unified")
            .map(|c| {
                Ok((
                    c["implementation"]
                        .as_str()
                        .context("missing implementation")?
                        .to_string(),
                    c["version"]
                        .as_str()
                        .context("missing version")?
                        .to_string(),
                ))
            })
            .collect();
    }
    let mut implementations = BTreeMap::<String, Vec<(Release, String)>>::new();
    for candidate in catalog["sources"]["candidates"]
        .as_array()
        .context("missing candidates")?
        .iter()
        .filter(|c| c["mode"] == "unified" && c["has_captured_results"] != false)
    {
        let implementation = candidate["implementation"]
            .as_str()
            .context("missing implementation")?;
        let version = candidate["version"].as_str().context("missing version")?;
        implementations
            .entry(implementation.into())
            .or_default()
            .push((Release::parse(version)?, version.into()));
    }
    let mut selected = BTreeSet::new();
    for (implementation, mut versions) in implementations {
        versions.sort_by(|a, b| b.0.cmp(&a.0));
        versions.dedup_by(|a, b| a.1 == b.1);
        selected.extend(
            versions
                .into_iter()
                .take(1)
                .map(|(_, version)| (implementation.clone(), version)),
        );
    }
    Ok(selected)
}

fn is_recent(resource: &Value, recent: &BTreeSet<(String, String)>) -> bool {
    resource["kind"] == "inputs-and-golden"
        || recent.contains(&(
            resource["implementation"].as_str().unwrap_or("").into(),
            resource["version"].as_str().unwrap_or("").into(),
        ))
}

/// Embed the same verified transport bytes used by on-demand reads. Keeping
/// recipe indexes and packs avoids a separate decoder or weaker admission path.
pub struct Embedded<'a> {
    pub value: Value,
    _memory: Reservation<'a>,
}

fn embedded_memory(total: u64, catalog: usize, compressed_catalog: usize) -> Result<u64> {
    // Base64, JSON serialization and HTML replacement can coexist as complete
    // copies. Retain this admission until the caller finishes writing HTML.
    total
        .checked_mul(8)
        .and_then(|n| n.checked_add((catalog as u64).checked_mul(32)?))
        .and_then(|n| n.checked_add((compressed_catalog as u64).checked_mul(8)?))
        .and_then(|n| n.checked_add(16 * 1024 * 1024))
        .context("embedded memory reservation overflow")
}

pub fn embedded<'a>(
    stage: &Path,
    catalog: &Value,
    budget: &'a MemoryBudget,
) -> Result<Embedded<'a>> {
    let recent = recent_captures(catalog)?;
    let mut descriptors = BTreeMap::<String, Value>::new();
    let catalog_bytes = json_bytes(catalog)?;
    let compressed = codec::gzip(&catalog_bytes)?;
    ensure!(
        catalog_bytes.len() <= 64 * 1024 * 1024 && compressed.len() <= 64 * 1024 * 1024,
        "embedded catalog exceeds limit"
    );
    for resource in catalog["sources"]["unified"]
        .as_array()
        .context("missing resources")?
        .iter()
        .filter(|r| is_recent(r, &recent))
    {
        let descriptor = &resource["recipe_index"];
        let path = descriptor["path"]
            .as_str()
            .context("missing recipe index")?;
        let size = descriptor["size"]
            .as_u64()
            .context("missing recipe index size")?;
        let decoded_size = descriptor["decoded_size"]
            .as_u64()
            .context("missing recipe index decoded size")?;
        ensure!(
            size <= 64 * 1024 * 1024 && decoded_size <= 64 * 1024 * 1024,
            "embedded recipe index exceeds limit"
        );
        let scratch = decoded_size
            .checked_mul(32)
            .and_then(|n| n.checked_add(size.checked_mul(2)?))
            .and_then(|n| n.checked_add(4 * 1024 * 1024))
            .context("embedded index reservation overflow")?;
        let _scratch = budget.reserve(scratch)?;
        let bytes = source::verified(stage, descriptor)?;
        let decoded = codec::gunzip(
            &bytes,
            descriptor["sha256"].as_str().context("missing hash")?,
            usize::try_from(decoded_size)?,
        )?;
        let index: Value = serde_json::from_slice(&decoded)?;
        let recipe =
            &index["resources"][resource["path"].as_str().context("missing resource path")?];
        for chunk in recipe["chunks"]
            .as_array()
            .context("missing embedded recipe")?
        {
            let pack = &chunk["pack"];
            descriptors.insert(
                pack["path"].as_str().context("missing pack path")?.into(),
                pack.clone(),
            );
        }
        descriptors.insert(path.into(), descriptor.clone());
    }
    ensure!(descriptors.len() <= 4096, "too many embedded resources");
    if catalog["capture_generation"]["mode"] == "latest" {
        for name in source::inventory(&stage.join("conformance/viewer-data"))? {
            let path = format!("conformance/viewer-data/{name}");
            if !stage.join(&path).is_file() {
                continue;
            }
            let bytes = source::read(stage, &path)?;
            descriptors.insert(
                path.clone(),
                json!({"path":path,"size":bytes.len(),"sha256":sha256(&bytes)}),
            );
        }
    }
    let mut total = 0u64;
    for descriptor in descriptors.values() {
        let size = descriptor["size"]
            .as_u64()
            .context("missing embedded size")?;
        ensure!(size <= 64 * 1024 * 1024, "embedded resource exceeds limit");
        total = total.checked_add(size).context("embedded size overflow")?;
    }
    ensure!(
        total <= 256 * 1024 * 1024,
        "embedded resources exceed memory limit"
    );
    let memory = budget.reserve(embedded_memory(
        total,
        catalog_bytes.len(),
        compressed.len(),
    )?)?;
    let resources = descriptors.values().map(|descriptor| {
        Ok(json!({"descriptor":descriptor,"bytes":STANDARD.encode(source::verified(stage, descriptor)?)}))
    }).collect::<Result<Vec<_>>>()?;
    let inventory: Vec<Value> = descriptors
        .values()
        .map(|d| json!([d["path"], d["size"], d["sha256"]]))
        .collect();
    Ok(Embedded {
        value: json!({"schema":1,"catalog":{"bytes":STANDARD.encode(&compressed),"size":compressed.len(),"sha256":sha256(&compressed),"decoded_size":catalog_bytes.len()},"resources":resources,"resources_sha256":sha256(&json_bytes(&json!(inventory))?),"captures":recent}),
        _memory: memory,
    })
}

pub fn assemble(
    root: &Path,
    stage: &Path,
    original: &Value,
    resolved: &Value,
    budget: &MemoryBudget,
) -> Result<Value> {
    let mut catalog = pipeline::projection(resolved)?;
    let recent = recent_captures(&catalog)?;
    let reference = catalog["metadata"]["selection_policy"]["unified"]["reference"]
        .as_str()
        .unwrap_or("");
    let selected = catalog["sources"]["candidates"]
        .as_array()
        .context("missing candidates")?
        .iter()
        .filter(|c| c["implementation"] == reference)
        .map(|c| {
            Ok((
                Release::parse(c["version"].as_str().context("missing version")?)?,
                c,
            ))
        })
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .max_by(|a, b| a.0.cmp(&b.0));
    let default_version = selected
        .as_ref()
        .and_then(|(_, c)| c["version"].as_str())
        .unwrap_or("");
    let default_families: BTreeSet<String> = selected
        .as_ref()
        .map(|(_, c)| {
            let key = if reference == "dynamo_v2" {
                "native_unified_families"
            } else {
                "capture_families"
            };
            c[key]
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default();
    let mut resources = catalog["sources"]["unified"]
        .as_array()
        .context("missing resolved resources")?
        .clone();
    let priority = |r: &Value| {
        if (r["kind"] == "inputs-and-golden"
            && default_families.contains(r["family"].as_str().unwrap_or("")))
            || (r["implementation"] == reference && r["version"] == default_version)
        {
            0
        } else if is_recent(r, &recent) {
            1
        } else {
            2
        }
    };
    resources.sort_by_key(|r| (priority(r), r["path"].as_str().unwrap_or("").to_string()));
    let mut writer = PackWriter::new(stage);
    let mut tokens = Vec::new();
    let mut phase = 0;
    for resource in &mut resources {
        let next = priority(resource);
        if next != phase {
            writer.flush_phase()?;
            phase = next;
        }
        let path = resource["path"]
            .as_str()
            .context("missing bundle path")?
            .to_string();
        ensure!(
            resource["compression"] == "none",
            "production requires canonical envelopes"
        );
        let bytes = source::verified(&stage.join(".native-envelopes"), resource)?;
        tokens.push(writer.add(&bytes)?);
        resource["transport"] = json!("chunked-v2");
        fs::remove_file(stage.join(".native-envelopes").join(path))?;
    }
    writer.flush_phase()?;
    let mut raw = BTreeMap::<String, Value>::new();
    let mut original_tokens = Vec::new();
    let originals = source::original_sources(original)?;
    // Admission releases in each task before returning. Only descriptors are
    // collected, so bounded readers never retain a whole history in memory.
    originals
        .par_iter()
        .try_for_each(|resource| -> Result<()> {
            let path = resource["path"].as_str().context("missing original path")?;
            let size = if path.ends_with(".tar.gz") {
                source::archive_source_size(root, path)?.checked_mul(3)
            } else {
                resource["size"]
                    .as_u64()
                    .context("missing original size")?
                    .checked_mul(64)
            }
            .context("original validation reservation overflow")?;
            let _reservation = budget.reserve(size.max(16 * 1024 * 1024))?;
            let bytes = source::verified(root, resource)?;
            if path.ends_with(".yaml") {
                crate::yaml::load_str(std::str::from_utf8(&bytes)?)?;
            }
            if path.ends_with(".tar.gz") {
                source::archive(&bytes, resource)?;
            }
            Ok(())
        })?;
    for resource in &originals {
        // Re-read and verify the same pinned identity admitted above. Pack
        // insertion order stays independent of validation completion order.
        let size = resource["size"].as_u64().context("missing original size")?;
        let _reservation = budget.reserve(size.max(1024 * 1024))?;
        let bytes = source::verified(root, resource)?;
        let path = resource["path"].as_str().context("missing original path")?;
        let token = writer.add(&bytes)?;
        let mut original_descriptor = resource.clone();
        // Source containers are a checkout concern. Published originals use
        // pack recipes and must not reference unpublished source archives.
        original_descriptor
            .as_object_mut()
            .unwrap()
            .remove("archive");
        original_tokens.push((path.to_string(), token, original_descriptor));
        if let Some(members) = resource["members"].as_array() {
            for member in members.iter().filter(|m| m["kind"] == "file") {
                let member = member["path"].as_str().context("missing archive member")?;
                for alias in [member.to_string(), format!("conformance/{member}")] {
                    ensure!(!raw.contains_key(&alias), "ambiguous raw alias: {alias}");
                    raw.insert(alias, json!({"path":path,"member":member}));
                }
            }
        }
    }
    writer.flush_phase()?;
    let mut inputs = json!({});
    for (resource, token) in resources.iter_mut().zip(tokens) {
        let path = resource["path"].as_str().unwrap().to_string();
        let recipe = writer.recipe(token)?;
        if resource["kind"] == "inputs-and-golden" {
            inputs[&path] = recipe;
        } else {
            let name = format!(
                "conformance/viewer-data/indexes/capture-{}.json.gz",
                resource["sha256"].as_str().unwrap()
            );
            resource["recipe_index"] =
                index(stage, &name, &json!({"schema":2,"resources":{path:recipe}}))?;
        }
    }
    let input_index = index(
        stage,
        "conformance/viewer-data/indexes/inputs.json.gz",
        &json!({"schema":2,"resources":inputs}),
    )?;
    for resource in resources
        .iter_mut()
        .filter(|r| r["kind"] == "inputs-and-golden")
    {
        resource["recipe_index"] = input_index.clone();
    }
    for (path, token, resource) in original_tokens {
        ensure!(!raw.contains_key(&path), "ambiguous original path");
        raw.insert(path.clone(), json!({"resource":{"path":path,"sha256":resource["sha256"],"size":resource["size"],"recipe":writer.recipe(token)?},"member":null}));
    }
    let mut buckets = BTreeMap::<String, BTreeMap<String, Value>>::new();
    for (path, entry) in raw {
        buckets
            .entry(sha256(path.as_bytes())[..2].to_string())
            .or_default()
            .insert(path, entry);
    }
    let mut lookup = json!({});
    for (bucket, entries) in buckets {
        let path = format!("conformance/viewer-data/indexes/raw-{bucket}.json.gz");
        lookup[&bucket] = index(stage, &path, &json!({"schema":2,"resources":entries}))?;
    }
    // The reader only needs these identities to validate resolved record members.
    catalog["sources"]["original_unified"] = json!(originals.iter()
        .chain(original["sources"]["lineage"].as_array().into_iter().flatten())
        .filter(|r| r["kind"] == "history").map(|r| json!({"path":r["path"],"kind":r["kind"],"family":r["family"],"implementation":r["implementation"],"version":r["version"]})).collect::<Vec<_>>());
    for candidate in catalog["sources"]["candidates"].as_array_mut().unwrap() {
        candidate.as_object_mut().unwrap().remove("sources");
        candidate.as_object_mut().unwrap().remove("source");
    }
    catalog["sources"]["unified"] = json!(resources);
    catalog["transport"] = json!({"schema":2,"raw_buckets":lookup});
    // Reject unreferenced chunks and damaged pack offsets before installation.
    let _validation_memory = budget.reserve(
        (crate::transport::VALIDATION_CACHE_BYTES + 4 * crate::transport::MAX_PACK) as u64,
    )?;
    crate::transport::validate_recipes(stage, &writer.finish()?)?;
    // The source revision remains the provenance identity; this digest identifies
    // this exact runtime representation, independently of the old gzip bytes.
    catalog["runtime_sha256"] = json!(sha256(&json_bytes(&catalog)?));
    let spools = stage.join(".native-envelopes");
    if spools.exists() {
        fs::remove_dir_all(spools)?;
    }
    pipeline::write(
        stage,
        "conformance/viewer-resolved-catalog.json.gz",
        &codec::gzip(&json_bytes(&catalog)?)?,
    )?;
    Ok(catalog)
}

fn rewrite_paths(value: &mut Value, from: &str, to: &str) {
    match value {
        Value::String(path) if path.starts_with(from) => {
            *path = format!("{to}{}", &path[from.len()..]);
        }
        Value::Array(values) => values
            .iter_mut()
            .for_each(|value| rewrite_paths(value, from, to)),
        Value::Object(values) => values
            .values_mut()
            .for_each(|value| rewrite_paths(value, from, to)),
        _ => {}
    }
}

pub fn relocate_checkout_generation(stage: &Path, catalog: &mut Value) -> Result<()> {
    let generation = sha256(&json_bytes(catalog)?);
    let generation_root = format!("conformance/viewer-data/generations/{generation}");
    let data = stage.join("conformance/viewer-data");
    let temporary = stage.join("conformance/.viewer-data-generation");
    ensure!(
        !temporary.exists(),
        "stale viewer data generation staging path"
    );
    fs::rename(&data, &temporary)?;
    let destination = stage.join(&generation_root);
    fs::create_dir_all(
        destination
            .parent()
            .context("generation path has no parent")?,
    )?;
    fs::rename(temporary, destination)?;
    rewrite_paths(
        catalog,
        "conformance/viewer-data/",
        &format!("{generation_root}/"),
    );
    Ok(())
}

/// Traverse the actual runtime dependency graph, then reject every extra file.
/// This audit lives beside the site, never inside the browser publication.
pub fn inventory(stage: &Path, assets: impl Iterator<Item = String>) -> Result<Value> {
    let mut pending: Vec<String> = assets.collect();
    pending.extend([
        "conformance/viewer-resolved-catalog.json.gz".into(),
        "conformance/viewer-publication.json".into(),
    ]);
    let mut expected = BTreeMap::<String, Value>::new();
    fn references(
        value: &Value,
        stage: &Path,
        pending: &mut Vec<String>,
        checked: &mut BTreeMap<String, Value>,
    ) -> Result<()> {
        match value {
            Value::Object(fields) => {
                if value.get("sha256").is_some()
                    && value.get("size").is_some()
                    && value.get("path").is_some()
                    && value.get("recipe").is_none()
                    && value["transport"] != "chunked-v2"
                {
                    let path = value["path"].as_str().context("invalid dependency path")?;
                    source::relative(path)?;
                    if let Some(previous) = checked.get(path) {
                        ensure!(previous == value, "conflicting runtime dependency: {path}");
                        return Ok(());
                    }
                    checked.insert(path.to_string(), value.clone());
                    let bytes = fs::read(stage.join(path))
                        .with_context(|| format!("missing runtime dependency: {path}"))?;
                    ensure!(
                        value["sha256"] == sha256(&bytes)
                            && value["size"].as_u64() == Some(bytes.len() as u64),
                        "runtime dependency identity mismatch: {path}"
                    );
                    pending.push(path.to_string());
                }
                for child in fields.values() {
                    references(child, stage, pending, checked)?;
                }
            }
            Value::Array(values) => {
                for child in values {
                    references(child, stage, pending, checked)?;
                }
            }
            _ => (),
        }
        Ok(())
    }
    let mut checked = BTreeMap::new();
    while let Some(path) = pending.pop() {
        if expected.contains_key(&path) {
            continue;
        }
        let bytes = fs::read(stage.join(&path))?;
        expected.insert(
            path.clone(),
            json!({"sha256":sha256(&bytes),"size":bytes.len()}),
        );
        if path.ends_with(".json.gz") {
            let decoded =
                codec::gunzip(&bytes, &sha256(&bytes), source::MAX_ARTIFACT as usize - 1)?;
            references(
                &serde_json::from_slice::<Value>(&decoded)?,
                stage,
                &mut pending,
                &mut checked,
            )?;
        }
    }
    let actual: BTreeSet<String> = source::inventory(stage)?
        .into_iter()
        .filter(|path| stage.join(path).is_file())
        .collect();
    let declared: BTreeSet<String> = expected.keys().cloned().collect();
    ensure!(
        actual == declared,
        "publication dependency inventory mismatch; extra={:?}, missing={:?}",
        actual.difference(&declared).collect::<Vec<_>>(),
        declared.difference(&actual).collect::<Vec<_>>()
    );
    Ok(json!({"schema":2,"files":expected,"file_count":actual.len()}))
}

#[cfg(test)]
mod embedded_tests {
    use super::*;

    #[test]
    fn recent_releases_are_semantic_per_lineage_and_require_captures() -> Result<()> {
        let candidates = [
            ("dynamo_v2", "0.9.9", true),
            ("dynamo_v2", "0.10.1", true),
            ("dynamo_v2", "0.10.2", true),
            ("dynamo_v2", "0.11.0", false),
            ("vllm_python", "1.0.0", true),
            ("vllm_python_batch", "1.0.0", true),
            ("vllm_rust", "0.30.0", true),
        ];
        let catalog = json!({"sources":{"candidates":candidates.map(|(implementation,version,captured)|json!({"mode":"unified","implementation":implementation,"version":version,"has_captured_results":captured}))}});
        let recent = recent_captures(&catalog)?;
        assert_eq!(recent.len(), 4);
        assert!(!recent.contains(&("dynamo_v2".into(), "0.9.9".into())));
        assert!(!recent.contains(&("dynamo_v2".into(), "0.10.1".into())));
        assert!(!recent.contains(&("dynamo_v2".into(), "0.11.0".into())));
        for (implementation, version) in [
            ("dynamo_v2", "0.10.2"),
            ("vllm_python", "1.0.0"),
            ("vllm_python_batch", "1.0.0"),
            ("vllm_rust", "0.30.0"),
        ] {
            assert!(recent.contains(&(implementation.into(), version.into())));
        }
        Ok(())
    }

    #[test]
    fn embedded_admission_precedes_payload_and_survives_html_serialization() -> Result<()> {
        let stage = tempfile::tempdir()?;
        let bytes = codec::gzip(b"payload")?;
        let pack = json!({"path":"packs/sample.gz","size":bytes.len(),"sha256":sha256(&bytes)});
        pipeline::write(stage.path(), "packs/sample.gz", &bytes)?;
        let decoded = json_bytes(&json!({"resources":{"input.json":{"chunks":[{"pack":pack}]}}}))?;
        let bytes = codec::gzip(&decoded)?;
        pipeline::write(stage.path(), "indexes/input.gz", &bytes)?;
        let catalog = json!({"sources":{"candidates":[],"unified":[{"kind":"inputs-and-golden","path":"input.json","recipe_index":{"path":"indexes/input.gz","size":bytes.len(),"sha256":sha256(&bytes),"decoded_size":decoded.len()}}]}});
        let insufficient = MemoryBudget::new(8 * 1024 * 1024);
        let rejected = embedded(stage.path(), &catalog, &insufficient);
        assert!(
            rejected
                .err()
                .context("embedded payload ignored admission")?
                .to_string()
                .contains("publication job requires")
        );

        let catalog_bytes = json_bytes(&catalog)?;
        let required = embedded_memory(
            (pack["size"].as_u64().unwrap()) + bytes.len() as u64,
            catalog_bytes.len(),
            codec::gzip(&catalog_bytes)?.len(),
        )?;
        let budget = MemoryBudget::new(required);
        let payload = embedded(stage.path(), &catalog, &budget)?;
        let serialized = serde_json::to_string(&payload.value)?.replace('<', "\\u003c");
        let html = "<head></head>".replacen(
            "</head>",
            &format!("<script>{serialized}</script></head>"),
            1,
        );
        assert!(html.contains("resources_sha256"));
        std::thread::scope(|scope| -> Result<()> {
            let (started, ready) = std::sync::mpsc::sync_channel(1);
            let (finished, released) = std::sync::mpsc::sync_channel(1);
            let budget = &budget;
            let source = stage.path();
            let resource = &pack;
            scope.spawn(move || {
                started.send(()).unwrap();
                let _reservation = budget.reserve(1024 * 1024).unwrap();
                source::verified(source, resource).unwrap();
                finished.send(()).unwrap();
            });
            ready.recv_timeout(std::time::Duration::from_secs(5))?;
            assert!(
                released
                    .recv_timeout(std::time::Duration::from_millis(50))
                    .is_err(),
                "embedded reservation was released before HTML ownership ended"
            );
            drop(payload);
            released.recv_timeout(std::time::Duration::from_secs(5))?;
            Ok(())
        })?;
        Ok(())
    }
}
