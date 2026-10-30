// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Parallel family resolution followed by bounded bundle assembly in one pool.
use crate::{
    codec::{self, json_bytes, sha256},
    history::Release,
    resources::MemoryBudget,
    source::{self, STORE},
    yaml::{self, Node},
};
use anyhow::{Context, Result, ensure};
use rayon::prelude::*;
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::Path,
    sync::atomic::{AtomicU64, Ordering},
    time::Instant,
};

#[derive(Default)]
struct Timings {
    read: AtomicU64,
    yaml: AtomicU64,
    validation: AtomicU64,
    preparation: AtomicU64,
    encoding: AtomicU64,
    compression: AtomicU64,
    writes: AtomicU64,
}

fn elapsed(counter: &AtomicU64, started: Instant) {
    counter.fetch_add(
        started.elapsed().as_nanos().min(u64::MAX as u128) as u64,
        Ordering::Relaxed,
    );
}

impl Timings {
    fn report(&self) {
        let seconds = |counter: &AtomicU64| counter.load(Ordering::Relaxed) as f64 / 1e9;
        eprintln!(
            "snapshot timings: {}",
            json!({"phase":"snapshot-worker-timings","scope":"summed worker wall seconds",
            "source_reads":seconds(&self.read),"yaml_decoding":seconds(&self.yaml),"validation":seconds(&self.validation),
            "json_preparation":seconds(&self.preparation),"json_encoding":seconds(&self.encoding),"compression":seconds(&self.compression),"disk_writes":seconds(&self.writes)})
        );
    }
}

pub fn write(stage: &Path, relative: &str, bytes: &[u8]) -> Result<()> {
    source::relative(relative)?;
    ensure!(
        (bytes.len() as u64) < source::MAX_ARTIFACT,
        "artifact is at least 1 GB: {relative}"
    );
    let path = stage.join(relative);
    source::no_links(&path)?;
    fs::create_dir_all(path.parent().context("missing output parent")?)?;
    fs::write(path, bytes)?;
    Ok(())
}

fn descriptor(
    path: &str,
    kind: &str,
    family: Option<&str>,
    families: Option<Vec<String>>,
    capture: Option<(&str, &str)>,
    fingerprint: &str,
    catalog: &Value,
) -> Value {
    let (implementation, version) = capture.map_or((None, None), |(i, v)| (Some(i), Some(v)));
    let mut descriptor = json!({"path":path,"format":"typed-json-v1","compression":"gzip","kind":kind,"family":family});
    if let Some(families) = families {
        descriptor["families"] = json!(families);
    }
    let object = descriptor.as_object_mut().unwrap();
    object.insert("implementation".into(), json!(implementation));
    object.insert("version".into(), json!(version));
    object.insert(
        "record_contract".into(),
        implementation
            .map(|i| catalog["metadata"]["unified_capture_policy"][i]["mode"].clone())
            .unwrap_or(Value::Null),
    );
    object.insert(
        "capture_identity".into(),
        implementation
            .map(|i| json!(format!("{i}@unified@{}", version.unwrap())))
            .unwrap_or(Value::Null),
    );
    object.insert("source_fingerprint".into(), json!(fingerprint));
    descriptor
}

/// Logical envelopes have one producer; only their final encoding differs.
#[derive(Clone, Copy)]
pub enum SnapshotEncoding {
    CheckoutGzip,
    CanonicalSpool,
    ValidateOnly,
}

fn bundle(
    stage: &Path,
    descriptor: Value,
    typed: Value,
    metadata: Option<Value>,
    encoding: SnapshotEncoding,
    timings: &Timings,
) -> Result<Value> {
    let mut envelope = json!({"schema":1,"kind":descriptor["kind"],"family":descriptor["family"]});
    if let Some(families) = descriptor.get("families") {
        envelope["families"] = families.clone();
    }
    for key in ["implementation", "version", "source_fingerprint"] {
        envelope[key] = descriptor[key].clone();
    }
    envelope["value"] = typed;
    let clock = Instant::now();
    let canonical = json_bytes(&envelope)?;
    elapsed(&timings.encoding, clock);
    let path = descriptor["path"].as_str().context("bundle path missing")?;
    let (data, output, compression) = match encoding {
        SnapshotEncoding::CheckoutGzip => {
            let clock = Instant::now();
            let compressed = codec::gzip_with_level(&canonical, 6)?;
            elapsed(&timings.compression, clock);
            (compressed, path.to_string(), "gzip")
        }
        SnapshotEncoding::CanonicalSpool => {
            (canonical, format!(".native-envelopes/{path}"), "none")
        }
        SnapshotEncoding::ValidateOnly => anyhow::bail!("validation cannot encode an envelope"),
    };
    let clock = Instant::now();
    write(stage, &output, &data)?;
    elapsed(&timings.writes, clock);
    let mut result = json!({"path":path,"sha256":sha256(&data),"size":data.len()});
    for (key, value) in descriptor.as_object().unwrap() {
        if key != "path" {
            result[key] = value.clone();
        }
    }
    result["compression"] = json!(compression);
    if let Some(metadata) = metadata {
        result["resolved_metadata_sha256"] = json!(sha256(&json_bytes(&metadata)?));
    }
    Ok(result)
}

fn metadata(records: &Node) -> Result<Node> {
    let mut result = Node::map();
    for (id, record) in records.mapping()? {
        let mut entry = Node::map();
        entry.set(
            "document",
            record.field("value")?.field("document")?.clone(),
        )?;
        entry.set("member", record.field("member")?.clone())?;
        result.set(id.string()?, entry)?;
    }
    Ok(result)
}

#[derive(Clone)]
struct Selection {
    implementation: String,
    version: String,
    release: Release,
    scope: Option<BTreeSet<String>>,
}
fn selections(catalog: &Value) -> Result<Vec<Selection>> {
    let mut selections = BTreeMap::new();
    for candidate in catalog["sources"]["candidates"]
        .as_array()
        .context("missing candidates")?
    {
        if candidate["mode"] != "unified" {
            continue;
        }
        let implementation = candidate["implementation"]
            .as_str()
            .context("missing implementation")?
            .to_string();
        let version = candidate["version"]
            .as_str()
            .context("missing version")?
            .to_string();
        let release = Release::parse(&version)?;
        let scope = candidate
            .get("capture_families")
            .map(|scope| {
                scope
                    .as_array()
                    .context("invalid capture scope")?
                    .iter()
                    .map(|n| Ok(n.as_str().context("invalid scoped family")?.to_string()))
                    .collect::<Result<BTreeSet<_>>>()
            })
            .transpose()?;
        source::relative(&format!("{implementation}-{version}"))?;
        ensure!(
            selections
                .insert(
                    (implementation.clone(), release.clone(), version.clone()),
                    Selection {
                        implementation,
                        version,
                        release,
                        scope
                    }
                )
                .is_none(),
            "duplicate candidate release"
        );
    }
    Ok(selections.into_values().collect())
}

fn family_sources<'a>(
    resources: &[&'a Value],
    index: &BTreeMap<String, &'a Value>,
) -> Result<BTreeMap<String, &'a Value>> {
    let mut pending = resources.to_vec();
    let mut sources = BTreeMap::new();
    while let Some(resource) = pending.pop() {
        let path = resource["path"]
            .as_str()
            .context("missing family source path")?;
        if sources.contains_key(path) {
            continue;
        }
        if let Some(reference) = resource["execution"].as_str() {
            let path = format!(
                "{STORE}/{}",
                crate::history::source_path(reference, "executions")?
            );
            pending.push(*index.get(&path).context("execution is not indexed")?);
        }
        for reference in resource["receipts"].as_array().into_iter().flatten() {
            let path = format!(
                "{STORE}/{}",
                crate::history::source_path(
                    reference.as_str().context("invalid receipt reference")?,
                    "receipts"
                )?
            );
            pending.push(*index.get(&path).context("receipt is not indexed")?);
        }
        sources.insert(path.to_string(), resource);
    }
    Ok(sources)
}

pub(crate) fn family_source_bytes(
    resources: &[&Value],
    index: &BTreeMap<String, &Value>,
) -> Result<u64> {
    family_sources(resources, index)?
        .values()
        .try_fold(0u64, |sum, resource| {
            sum.checked_add(resource["size"].as_u64().context("missing resource size")?)
                .context("family dependency size overflow")
        })
}

fn source_fingerprint(resources: &[Value]) -> Result<String> {
    // Container identity belongs to source admission. Moving unchanged YAML
    // into an archive must not invalidate canonical report content identities.
    let logical = resources
        .iter()
        .map(|resource| {
            let mut resource = resource.clone();
            resource.as_object_mut().unwrap().remove("archive");
            resource.as_object_mut().unwrap().remove("dataset_root");
            resource
        })
        .collect::<Vec<_>>();
    Ok(sha256(&json_bytes(&json!({"resources":logical}))?))
}

fn append_history_snapshots(
    snapshots: &mut Vec<Value>,
    history: &crate::history::History,
    directory: &Path,
    family_key: &str,
    family: &str,
    authored: &Node,
) -> Result<()> {
    let mut cursor = crate::history::Cursor::default();
    for (position, capture) in history.captures.iter().enumerate() {
        cursor.apply(history, position)?;
        let key = sha256(&json_bytes(&json!([
            family_key,
            history.implementation,
            capture.version
        ]))?);
        let mut records = cursor.records(history, &capture.version)?;
        source::bind_record_requests(&mut records, authored)?;
        let mut summary = crate::catalog::record_summary(&records)?;
        summary["family"] = json!(family);
        summary["implementation"] = json!(history.implementation);
        summary["version"] = json!(capture.version);
        crate::conversion_cache::write(directory, &key, &records.typed())?;
        let mut metadata = json!({"runtime_version":capture.version,
            "provenance":capture.document.field("provenance")?.typed(),
            "document":capture.document.field("document")?.typed()});
        for field in ["execution_proof", "historical_evidence"] {
            if let Some(value) = capture.document.get(field) {
                metadata[field] = value.typed();
            }
        }
        snapshots.push(json!({"implementation":history.implementation,
            "version":capture.version, "fragment":key, "summary":summary,
            "capture":metadata}));
    }
    Ok(())
}

pub(crate) fn family_cache_key(
    resources: &[&Value],
    index: &BTreeMap<String, &Value>,
    catalog: &Value,
    family: &str,
) -> Result<String> {
    let dependencies = family_sources(resources, index)?
        .into_iter()
        .map(|(path, resource)| {
            let mut logical = resource.clone();
            let object = logical
                .as_object_mut()
                .context("invalid family dependency")?;
            object.remove("archive");
            object.remove("dataset_root");
            Ok((path, logical))
        })
        .collect::<Result<BTreeMap<_, _>>>()?;
    Ok(sha256(&codec::canonical_json_bytes(
        &json!({"schema":1, "family":family, "sources":dependencies,
        "aliases":catalog["metadata"]["unified_aliases"],
        "policy":catalog["metadata"]["unified_capture_policy"],
        "code":[include_str!("pipeline.rs"), include_str!("source.rs"), include_str!("yaml.rs"),
                include_str!("history.rs"), include_str!("codec.rs"), include_str!("catalog.rs"),
                include_str!("conversion_cache.rs")]}),
    )?))
}

fn read_family_cache(directory: &Path, key: &str, family: &str, aliases: &Value) -> Option<Value> {
    let payload = crate::conversion_cache::read(directory, key)?;
    let authored = Node::from_typed(&payload["authored"]).ok()?;
    source::family(&authored, family, aliases).ok()?;
    let mut identities = BTreeSet::new();
    for row in payload["snapshots"].as_array()? {
        let implementation = row["implementation"].as_str()?;
        let version = row["version"].as_str()?;
        if !identities.insert((row["dataset"].as_str(), implementation, version))
            || !crate::catalog::valid_summary(&row["summary"], family, implementation, version)
            || row["capture"]["runtime_version"] != version
        {
            return None;
        }
        let records = crate::conversion_cache::read(directory, row["fragment"].as_str()?)?;
        let mut snapshot = Node::map();
        snapshot.set("schema_version", Node::Int(2.into())).ok()?;
        for field in [
            "provenance",
            "document",
            "execution_proof",
            "historical_evidence",
        ] {
            if let Some(value) = row["capture"].get(field) {
                snapshot.set(field, Node::from_typed(value).ok()?).ok()?;
            }
        }
        snapshot
            .set("records", Node::from_typed(&records).ok()?)
            .ok()?;
        source::resolved_snapshot(&snapshot, &authored, family, implementation, version).ok()?;
    }
    Some(payload)
}

pub(crate) struct FamilyCache<'a> {
    pub directory: &'a Path,
    pub key: &'a str,
    pub reuse: bool,
}

pub(crate) fn converted_family(
    root: &Path,
    resources: &[&Value],
    index: &BTreeMap<String, &Value>,
    catalog: &Value,
    name: &str,
    reserved: u64,
    cache: FamilyCache<'_>,
) -> Result<Value> {
    let FamilyCache {
        directory,
        key,
        reuse,
    } = cache;
    if reuse
        && let Some(payload) = read_family_cache(
            directory,
            key,
            name,
            &catalog["metadata"]["unified_aliases"],
        )
    {
        for resource in family_sources(resources, index)?.values() {
            source::verified(root, resource)?;
        }
        return Ok(payload);
    }
    eprintln!("source conversion: {name} miss");
    let mut snapshots = Vec::new();
    let input = resources
        .iter()
        .find(|r| r["kind"] == "inputs-and-golden")
        .context("dataset has no authored family")?;
    let bytes = source::verified(root, input)?;
    let authored = yaml::load_str(std::str::from_utf8(&bytes)?)?;
    source::family(&authored, name, &catalog["metadata"]["unified_aliases"])?;
    let mut implementations = BTreeMap::<String, Vec<&Value>>::new();
    for resource in resources {
        if let Some(implementation) = resource["implementation"].as_str() {
            implementations
                .entry(implementation.into())
                .or_default()
                .push(resource);
        }
    }
    for (implementation, resources) in implementations {
        let batch =
            catalog["metadata"]["unified_capture_policy"][&implementation]["mode"] == "batch";
        let first = source::verified(root, resources[0])?;
        let complete = yaml::load_str(std::str::from_utf8(&first)?)?.get("schema_version")
            == Some(&Node::Int(2.into()));
        drop(first);
        if complete {
            for resource in resources {
                let history = source::load_history(
                    root,
                    &authored,
                    name,
                    &implementation,
                    &[resource],
                    batch,
                    source::HistoryAdmission {
                        limit: reserved,
                        index,
                    },
                )?;
                append_history_snapshots(
                    &mut snapshots,
                    &history,
                    directory,
                    key,
                    name,
                    &authored,
                )?;
            }
        } else {
            let history = source::load_history(
                root,
                &authored,
                name,
                &implementation,
                &resources,
                batch,
                source::HistoryAdmission {
                    limit: reserved,
                    index,
                },
            )?;
            append_history_snapshots(&mut snapshots, &history, directory, key, name, &authored)?;
        }
    }

    let payload = json!({"authored":authored.typed(), "snapshots":snapshots});
    crate::conversion_cache::write(directory, key, &payload)?;
    Ok(payload)
}

/// Resolve independent families concurrently, never parallelizing a history cursor.
pub fn snapshots(
    root: &Path,
    stage: &Path,
    catalog: &Value,
    budget: &MemoryBudget,
    encoding: SnapshotEncoding,
) -> Result<Value> {
    snapshots_cached(root, stage, catalog, budget, encoding, None)
}

pub fn snapshots_cached(
    root: &Path,
    stage: &Path,
    catalog: &Value,
    budget: &MemoryBudget,
    encoding: SnapshotEncoding,
    cache: Option<&Path>,
) -> Result<Value> {
    let original = catalog["sources"]["unified"]
        .as_array()
        .context("missing Unified sources")?;
    if original.is_empty() {
        return Ok(catalog.clone());
    }
    let fingerprint = source_fingerprint(&source::original_sources(catalog)?)?;
    let selected = selections(catalog)?;
    let mut families = BTreeMap::<String, Vec<&Value>>::new();
    for resource in original {
        if let Some(family) = resource["family"].as_str() {
            families
                .entry(family.to_string())
                .or_default()
                .push(resource);
        }
    }
    let index = catalog["sources"]["lineage"]
        .as_array()
        .into_iter()
        .flatten()
        .chain(original.iter())
        .map(|r| {
            Ok((
                r["path"]
                    .as_str()
                    .context("missing source path")?
                    .to_string(),
                r,
            ))
        })
        .collect::<Result<BTreeMap<_, _>>>()?;
    let parts = stage.join(".native-parts");
    fs::create_dir(&parts)?;
    let timings = Timings::default();
    let fragment_directory = cache
        .map(Path::to_path_buf)
        .unwrap_or_else(|| parts.join("conversion"));
    let per_family = families
        .par_iter()
        .map(|(name, resources)| -> Result<Value> {
            // Fragments are spooled before selection so a family's full release
            // history never needs to coexist as typed JSON in memory.
            let units = if let Some(datasets) = catalog["sources"]["history_datasets"].as_array() {
                let mut units = datasets
                    .iter()
                    .map(|dataset| -> Result<_> {
                        let all = dataset["resources"]
                            .as_array()
                            .context("missing dataset sources")?;
                        let scoped = all
                            .iter()
                            .map(|r| {
                                Ok((
                                    r["path"]
                                        .as_str()
                                        .context("missing dataset path")?
                                        .to_string(),
                                    r,
                                ))
                            })
                            .collect::<Result<BTreeMap<_, _>>>()?;
                        let family = all
                            .iter()
                            .filter(|r| r["family"] == name.as_str())
                            .collect::<Vec<_>>();
                        Ok((
                            dataset["id"]
                                .as_str()
                                .context("missing dataset identity")?
                                .to_string(),
                            family,
                            scoped,
                        ))
                    })
                    .collect::<Result<Vec<_>>>()?;
                units.push((
                    String::new(),
                    resources
                        .iter()
                        .copied()
                        .filter(|r| r["dataset_root"].is_null())
                        .collect(),
                    index.clone(),
                ));
                units
            } else {
                vec![(String::new(), resources.clone(), index.clone())]
            };
            let source_bytes = units.iter().try_fold(0u64, |total, (_, family, scoped)| {
                total
                    .checked_add(family_source_bytes(family, scoped)?)
                    .context("dataset reservation overflow")
            })?;
            let reserved = source_bytes
                .checked_mul(64)
                .context("family reservation overflow")?
                .max(16 * 1024 * 1024);
            let _reservation = budget.reserve(reserved)?;
            let authored_resource = resources
                .iter()
                .find(|r| r["kind"] == "inputs-and-golden")
                .context("missing authored family")?;
            ensure!(
                authored_resource["path"]
                    == format!("{STORE}/families/{name}/inputs_and_golden.yaml"),
                "family path mismatch"
            );
            let unit_keys = units
                .iter()
                .map(|(id, family, scoped)| {
                    Ok(json!([
                        id,
                        family_cache_key(family, scoped, catalog, name)?
                    ]))
                })
                .collect::<Result<Vec<_>>>()?;
            let cache_key = sha256(&json_bytes(&json!([
                authored_resource["sha256"],
                unit_keys
            ]))?);
            let cached = if matches!(encoding, SnapshotEncoding::ValidateOnly) {
                None
            } else {
                cache
                    .and_then(|directory| {
                        read_family_cache(
                            directory,
                            &cache_key,
                            name,
                            &catalog["metadata"]["unified_aliases"],
                        )
                    })
                    .filter(|payload| {
                        payload["snapshots"]
                            .as_array()
                            .is_some_and(|rows| rows.iter().all(|row| row["dataset"].is_string()))
                    })
            };
            let payload = if let Some(payload) = cached {
                for (_, family, scoped) in &units {
                    for resource in family_sources(family, scoped)?.values() {
                        source::verified(root, resource)?;
                    }
                }
                eprintln!("family conversion: {name} hit");
                payload
            } else {
                eprintln!("family conversion: {name} miss");
                let bytes = source::verified(root, authored_resource)?;
                let mut authored = yaml::load_str(std::str::from_utf8(&bytes)?)?;
                source::family(&authored, name, &catalog["metadata"]["unified_aliases"])?;
                let mut retained_cases = authored.field("cases")?.clone();
                let mut snapshots = Vec::new();
                for (dataset, family, scoped) in &units {
                    if family.is_empty() {
                        continue;
                    }
                    let key = family_cache_key(family, scoped, catalog, name)?;
                    let mut unit = converted_family(
                        root,
                        family,
                        scoped,
                        catalog,
                        name,
                        reserved,
                        FamilyCache {
                            directory: &fragment_directory,
                            key: &key,
                            reuse: !matches!(encoding, SnapshotEncoding::ValidateOnly),
                        },
                    )?;
                    let retained = Node::from_typed(&unit["authored"])?;
                    for (id, case) in retained.field("cases")?.mapping()? {
                        if retained_cases.get(id.string()?).is_none() {
                            let mut retired = case.clone();
                            retired.set("lifecycle", Node::Str("retired".into()))?;
                            retired.set("display_id", Node::Null)?;
                            retired.set("historical_ids", Node::Array(Vec::new()))?;
                            retained_cases.set(id.string()?, retired)?;
                        }
                    }
                    for mut row in unit["snapshots"]
                        .take()
                        .as_array()
                        .context("missing dataset fragments")?
                        .iter()
                        .cloned()
                    {
                        row["dataset"] = json!(dataset);
                        snapshots.push(row);
                    }
                }
                authored.set("cases", retained_cases)?;
                source::family(&authored, name, &catalog["metadata"]["unified_aliases"])?;
                let payload = json!({"authored":authored.typed(), "snapshots":snapshots});
                if !matches!(encoding, SnapshotEncoding::ValidateOnly)
                    && let Some(directory) = cache
                {
                    crate::conversion_cache::write(directory, &cache_key, &payload)?;
                }
                payload
            };
            if matches!(encoding, SnapshotEncoding::ValidateOnly) {
                return Ok(Value::Null);
            }
            let input_descriptor = descriptor(
                &format!("conformance/viewer-data/{name}/inputs-and-golden.json.gz"),
                "inputs-and-golden",
                Some(name),
                None,
                None,
                &fingerprint,
                catalog,
            );
            let input = bundle(
                stage,
                input_descriptor,
                payload["authored"].clone(),
                None,
                encoding,
                &timings,
            )?;
            let snapshots = payload["snapshots"]
                .as_array()
                .context("missing cached snapshots")?;
            for selection in &selected {
                if selection
                    .scope
                    .as_ref()
                    .is_some_and(|scope| !scope.contains(name))
                {
                    continue;
                }
                // Select the owning dataset only after each sparse timeline has
                // resolved independently. Interleaving their checkpoints changes inheritance.
                let route = resources
                    .iter()
                    .filter(|row| row["implementation"] == selection.implementation)
                    .map(|row| {
                        Ok((
                            Release::parse(
                                row["version"].as_str().context("missing route version")?,
                            )?,
                            row,
                        ))
                    })
                    .collect::<Result<Vec<_>>>()?
                    .into_iter()
                    .filter(|(release, _)| *release <= selection.release)
                    .max_by(|a, b| a.0.cmp(&b.0));
                let Some((_, route)) = route else {
                    continue;
                };
                let dataset = route["dataset"].as_str().unwrap_or("");
                let checkpoint = snapshots
                    .iter()
                    .find(|row| {
                        row["implementation"] == selection.implementation
                            && row["version"] == route["version"]
                            && row["dataset"] == dataset
                    })
                    .context("selected checkpoint has no resolved dataset fragment")?;
                let fragment = checkpoint["fragment"]
                    .as_str()
                    .context("missing fragment identity")?;
                let records = crate::conversion_cache::read(&fragment_directory, fragment)
                    .context("conversion fragment changed during assembly")?;
                let records = Node::from_typed(&records)?;
                let mut projected = crate::history::selected_snapshot(
                    &records,
                    checkpoint["version"].as_str().unwrap(),
                    &selection.version,
                )?;
                if !dataset.is_empty()
                    && let Node::Map(records) = projected.to_mut()
                {
                    for (_, wrapped) in records {
                        let member =
                            source::dataset_member(dataset, wrapped.field("member")?.string()?)?;
                        wrapped.set("member", Node::Str(member))?;
                    }
                }
                let part =
                    json!({"value":projected.typed(), "metadata":metadata(&projected)?.typed()});
                write(
                    stage,
                    &format!(
                        ".native-parts/{}-{}/{name}.json",
                        selection.implementation, selection.version
                    ),
                    &json_bytes(&part)?,
                )?;
            }
            Ok(input)
        })
        .collect::<Result<Vec<_>>>()?;
    if cache.is_none() && fragment_directory.exists() {
        fs::remove_dir_all(&fragment_directory)?;
    }
    if matches!(encoding, SnapshotEncoding::ValidateOnly) {
        timings.report();
        fs::remove_dir(&parts)?;
        return Ok(catalog.clone());
    }
    let mut resources = per_family;
    let captures = selected
        .par_iter()
        .map(|selection| -> Result<Option<Value>> {
            let path = parts.join(format!(
                "{}-{}",
                selection.implementation, selection.version
            ));
            if !path.exists() {
                return Ok(None);
            }
            let mut paths = fs::read_dir(&path)?
                .map(|p| Ok(p?.path()))
                .collect::<Result<Vec<_>>>()?;
            paths.sort();
            let source_bytes = paths
                .iter()
                .map(|p| fs::metadata(p).map(|m| m.len()))
                .collect::<std::io::Result<Vec<_>>>()?
                .into_iter()
                .sum::<u64>();
            let _reservation = budget.reserve(
                source_bytes
                    .checked_mul(20)
                    .context("bundle reservation overflow")?
                    .max(16 * 1024 * 1024),
            )?;
            let mut values = Vec::new();
            let mut metadata = Vec::new();
            let mut names = Vec::new();
            for path in paths {
                let name = path
                    .file_stem()
                    .and_then(|n| n.to_str())
                    .context("invalid fragment")?;
                let mut part: Value = serde_json::from_slice(&fs::read(&path)?)?;
                // Each private fragment has one consumer; release it before envelopes accumulate.
                fs::remove_file(&path)?;
                values.push(Value::Array(vec![
                    json!(["str", name]),
                    part["value"].take(),
                ]));
                metadata.push(Value::Array(vec![
                    json!(["str", name]),
                    part["metadata"].take(),
                ]));
                names.push(name.to_string());
            }
            fs::remove_dir(&path)?;
            let descriptor = descriptor(
                &format!(
                    "conformance/viewer-data/bundles/{}-{}.json.gz",
                    selection.implementation, selection.version
                ),
                "resolved-capture",
                None,
                Some(names),
                Some((&selection.implementation, &selection.version)),
                &fingerprint,
                catalog,
            );
            Ok(Some(bundle(
                stage,
                descriptor,
                yaml::tagged("object", Value::Array(values)),
                Some(yaml::tagged("object", Value::Array(metadata))),
                encoding,
                &timings,
            )?))
        })
        .collect::<Result<Vec<_>>>()?;
    resources.extend(captures.into_iter().flatten());
    timings.report();
    resources.sort_by_key(|r| {
        (
            if r["kind"] == "inputs-and-golden" {
                0
            } else {
                1
            },
            r["family"]
                .as_str()
                .or(r["implementation"].as_str())
                .unwrap_or("")
                .to_string(),
            r["version"].as_str().unwrap_or("").to_string(),
        )
    });
    let mut result = catalog.clone();
    result["source_revision"] = catalog["revision"].clone();
    result["sources"]["original_unified"] = json!(
        source::original_sources(catalog)?
            .into_iter()
            .filter(|resource| resource["kind"].as_str().is_some())
            .collect::<Vec<_>>()
    );
    result["sources"]["unified"] = json!(resources);
    result["sources"]
        .as_object_mut()
        .unwrap()
        .remove("history_datasets");
    result.as_object_mut().unwrap().remove("revision");
    result["revision"] = json!(sha256(&json_bytes(&result)?));
    fs::remove_dir_all(parts)?;
    Ok(result)
}

pub fn projection(catalog: &Value) -> Result<Value> {
    let keys = [
        "selection_policy",
        "unavailable_categories",
        "implementations",
        "dynamo_v2_producer",
        "unified_display",
        "unified_capture_policy",
        "markers",
        "unified",
        "unified_presentation",
        "folds",
        "unified_taxonomy",
        "unified_aliases",
        "unified_leak_markers",
        "documentation",
        "peer_versions",
    ];
    let mut metadata = json!({});
    for key in keys {
        if let Some(value) = catalog["metadata"].get(key) {
            metadata[key] = value.clone();
        }
    }
    let mut legacy = json!({});
    let slug = regex::Regex::new("[^a-zA-Z0-9]+")?;
    for candidate in catalog["sources"]["candidates"]
        .as_array()
        .context("missing candidates")?
    {
        let mode = candidate["mode"]
            .as_str()
            .context("missing candidate mode")?;
        if !["batch", "stream", "batch-on-stream", "unified"].contains(&mode) {
            continue;
        }
        let implementation = candidate["implementation"]
            .as_str()
            .context("missing candidate implementation")?;
        let version = candidate["version"]
            .as_str()
            .context("missing candidate version")?;
        let mode_key = if mode == "batch" {
            "b-"
        } else if mode == "batch-on-stream" {
            "s-"
        } else {
            ""
        };
        legacy[format!(
            "{implementation}-{mode_key}{}",
            slug.replace_all(version, "-").trim_matches('-')
        )] = json!(format!("{implementation}@{version}"));
    }
    metadata["legacy_candidate_identities"] = legacy;
    metadata["tabs"] = json!(
        catalog["metadata"]["tabs"]
            .as_array()
            .context("missing tabs")?
            .iter()
            .filter(|t| t["suite"] == "unified")
            .collect::<Vec<_>>()
    );
    metadata["selection_policy"] = json!({"unified":metadata["selection_policy"].get("unified").cloned().unwrap_or(json!({}))});
    if metadata.get("folds").is_none() {
        metadata["folds"] = json!({});
    }
    let mut result = json!({});
    if let Some(generation) = catalog.get("capture_generation") {
        result["capture_generation"] = generation.clone();
    }
    if let Some(revision) = catalog.get("source_revision") {
        result["source_revision"] = revision.clone();
    }
    for (key, value) in [
        ("schema", catalog.get("schema").cloned().unwrap_or(json!(1))),
        ("revision", catalog["revision"].clone()),
        ("snapshot", catalog["snapshot"].clone()),
        ("metadata", metadata),
    ] {
        result[key] = value;
    }
    let mut sources = json!({});
    if let Some(original) = catalog["sources"].get("original_unified") {
        sources["original_unified"] = original.clone();
    }
    sources["archives"] = json!([]);
    sources["unified"] = catalog["sources"]["unified"].clone();
    sources["candidates"] = json!(
        catalog["sources"]["candidates"]
            .as_array()
            .context("missing candidates")?
            .iter()
            .filter(|c| c["mode"] == "unified")
            .collect::<Vec<_>>()
    );
    result["sources"] = sources;
    Ok(result)
}

#[cfg(test)]
mod tests {
    #[test]
    fn unified_projection_preserves_producer_display_identity() -> anyhow::Result<()> {
        let producer = serde_json::json!({"crate_version":"12.3.4","kind":"unpublished"});
        let catalog = serde_json::json!({
            "revision":"catalog",
            "metadata":{"dynamo_v2_producer":producer,"tabs":[]},
            "sources":{"candidates":[],"unified":[]}
        });
        let projected = super::projection(&catalog)?;
        assert_eq!(projected["metadata"]["dynamo_v2_producer"], producer);
        Ok(())
    }

    #[test]
    fn family_cache_identity_excludes_selection_renderer_and_container_changes()
    -> anyhow::Result<()> {
        let resource = serde_json::json!({"path":"families/one/dynamo_v2-1.0.0.yaml", "family":"one", "size":1, "sha256":"a"});
        let other = serde_json::json!({"path":"families/two/dynamo_v2-1.0.0.yaml", "family":"two", "size":1, "sha256":"b"});
        let resources = vec![&resource];
        let index = [("one".to_string(), &resource), ("two".to_string(), &other)]
            .into_iter()
            .collect();
        let mut catalog =
            serde_json::json!({"metadata":{"unified_capture_policy":{},"unified_aliases":{}}});
        let first = super::family_cache_key(&resources, &index, &catalog, "one")?;
        catalog["revision"] = serde_json::json!("new-commit-and-time");
        catalog["sources"]["candidates"] =
            serde_json::json!([{"implementation":"dynamo_v2","version":"2.0.0"}]);
        catalog["metadata"]["source_hashes"]["renderer"] = serde_json::json!("changed");
        assert_eq!(
            first,
            super::family_cache_key(&resources, &index, &catalog, "one")?
        );
        let mut moved = resource.clone();
        moved["archive"] = serde_json::json!({"path":"relocated.tar.gz","sha256":"container"});
        let moved_index = [("one".to_string(), &moved)].into_iter().collect();
        assert_eq!(
            first,
            super::family_cache_key(&[&moved], &moved_index, &catalog, "one")?
        );
        moved["sha256"] = serde_json::json!("changed YAML");
        let changed_index = [("one".to_string(), &moved)].into_iter().collect();
        assert_ne!(
            first,
            super::family_cache_key(&[&moved], &changed_index, &catalog, "one")?
        );
        catalog["metadata"]["unified_capture_policy"]["new"] = serde_json::json!("policy");
        assert_ne!(
            first,
            super::family_cache_key(&resources, &index, &catalog, "one")?
        );
        Ok(())
    }

    use super::*;
    #[test]
    fn source_identity_depends_on_yaml_bytes_not_archive_encoding() -> Result<()> {
        let original = json!({"path":"families/sample/dynamo_v2-0.1.0.yaml","size":10,"sha256":"a".repeat(64)});
        let mut packed = original.clone();
        packed["archive"] =
            json!({"path":"versions/dynamo-0.1.0.tar.gz","size":100,"sha256":"b".repeat(64)});
        assert_eq!(
            source_fingerprint(std::slice::from_ref(&original))?,
            source_fingerprint(std::slice::from_ref(&packed))?
        );
        packed["sha256"] = json!("c".repeat(64));
        assert_ne!(
            source_fingerprint(&[original])?,
            source_fingerprint(&[packed])?
        );
        Ok(())
    }
    #[test]
    fn canonical_spool_and_checkout_gzip_share_one_envelope() -> Result<()> {
        let stage = crate::transaction::unique(&std::env::temp_dir(), "envelope-encoding-")?;
        let result = (|| -> Result<()> {
            let descriptor = descriptor(
                "conformance/viewer-data/family/inputs-and-golden.json.gz",
                "inputs-and-golden",
                Some("family"),
                None,
                None,
                "fingerprint",
                &json!({}),
            );
            let typed = json!(["object", []]);
            let metadata = Some(json!(["object", []]));
            let canonical = bundle(
                &stage,
                descriptor.clone(),
                typed.clone(),
                metadata.clone(),
                SnapshotEncoding::CanonicalSpool,
                &Timings::default(),
            )?;
            let checkout = bundle(
                &stage,
                descriptor,
                typed,
                metadata,
                SnapshotEncoding::CheckoutGzip,
                &Timings::default(),
            )?;
            let bytes = source::verified(&stage.join(".native-envelopes"), &canonical)?;
            let compressed = source::verified(&stage, &checkout)?;
            assert_eq!(canonical["compression"], "none");
            assert_eq!(checkout["compression"], "gzip");
            assert_eq!(
                bytes,
                codec::gunzip(
                    &compressed,
                    checkout["sha256"].as_str().unwrap(),
                    bytes.len()
                )?
            );
            assert_eq!(
                canonical["resolved_metadata_sha256"],
                checkout["resolved_metadata_sha256"]
            );
            assert!(!stage.join(".native-parts/inputs").exists());
            assert!(
                !stage
                    .join("conformance/viewer-data/bundles/inputs-and-golden.json.gz")
                    .exists()
            );
            Ok(())
        })();
        fs::remove_dir_all(stage)?;
        result
    }

    #[test]
    fn conversion_identity_ignores_nested_object_insertion_order() -> Result<()> {
        let first: Value = serde_json::from_str(
            r#"{"path":"families/sample/inputs_and_golden.yaml","size":1,"sha256":"source","extra":{"a":1,"b":2}}"#,
        )?;
        let second: Value = serde_json::from_str(
            r#"{"extra":{"b":2,"a":1},"sha256":"source","size":1,"path":"families/sample/inputs_and_golden.yaml"}"#,
        )?;
        let left: Value = serde_json::from_str(
            r#"{"metadata":{"unified_aliases":{"owners":{},"canonical":{}},"unified_capture_policy":{"demo":{"mode":"stream","version":"1.0.0"}}}}"#,
        )?;
        let right: Value = serde_json::from_str(
            r#"{"metadata":{"unified_capture_policy":{"demo":{"version":"1.0.0","mode":"stream"}},"unified_aliases":{"canonical":{},"owners":{}}}}"#,
        )?;
        assert_eq!(left, right);
        assert_eq!(first, second);
        assert_eq!(
            family_cache_key(&[&first], &BTreeMap::new(), &left, "sample")?,
            family_cache_key(&[&second], &BTreeMap::new(), &right, "sample")?
        );
        Ok(())
    }

    #[test]
    fn compact_dependency_bytes_are_part_of_family_admission() -> Result<()> {
        let pointer =
            json!({"path":"family/pointer.yaml","size":1,"execution":"executions/abc.yaml"});
        let execution = json!({"path":format!("{STORE}/executions/abc.yaml"),"size":1_000_000,"receipts":["receipts/def.yaml"]});
        let receipt = json!({"path":format!("{STORE}/receipts/def.yaml"),"size":2_000_000});
        let index = [&pointer, &execution, &receipt]
            .into_iter()
            .map(|r| (r["path"].as_str().unwrap().to_string(), r))
            .collect();
        assert_eq!(family_source_bytes(&[&pointer], &index)?, 3_000_001);
        let manifest = yaml::load_str(
            "schema_version: 1\nfamily: sample\nimplementation: demo\nversion: 1.0.0\ncheckpoint: {provenance: {}, document: {}, document_overrides: {}}\ncases: [one, two]\ndefaults: {metadata: expansion}\nexceptions: {}\n",
        )?;
        assert!(
            crate::history::expand_execution_bounded(
                &manifest,
                "sample",
                "demo",
                "1.0.0",
                &BTreeMap::new(),
                1
            )
            .is_err()
        );
        Ok(())
    }
}
