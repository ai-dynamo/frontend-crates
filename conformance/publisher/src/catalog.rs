// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Compact catalog preparation. Historical YAML never crosses the Python boundary.
use crate::{
    codec::sha256,
    history::Release,
    resources::{MemoryBudget, Policy, Reservation},
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
};

type IndexedDocument = (String, Value, Value, Option<Value>, Option<Value>);

fn admit_version_archive<'a>(
    root: &Path,
    name: &str,
    budget: &'a MemoryBudget,
) -> Result<(Value, std::sync::Arc<tempfile::TempDir>, Reservation<'a>)> {
    let source_name = format!("{STORE}/{name}");
    let size = source::source_size(root, &source_name)?;
    let reservation = budget.reserve(source::archive_reservation_bytes(
        size,
        source::MAX_ARCHIVE_MEMBER_BYTES,
    )?)?;
    let (descriptor, directory) = source::discover_archive(root, &source_name)?;
    Ok((descriptor, directory, reservation))
}

fn index_version_archive(
    root: &Path,
    name: &str,
    budget: &MemoryBudget,
    cache: Option<&Path>,
) -> Result<Vec<IndexedDocument>> {
    let (descriptor, directory, archive_reservation) = admit_version_archive(root, name, budget)?;
    let members = source::inventory(directory.path())?
        .into_iter()
        .filter(|name| directory.path().join(name).is_file())
        .collect::<Vec<_>>();
    let maximum = members
        .iter()
        .map(|name| fs::metadata(directory.path().join(name)).map(|m| m.len()))
        .collect::<std::io::Result<Vec<_>>>()?
        .into_iter()
        .max()
        .unwrap_or(0);
    drop(archive_reservation);
    let _reservation = budget.reserve(
        maximum
            .checked_mul(64)
            .context("catalog reservation overflow")?
            .max(16 * 1024 * 1024),
    )?;
    let mut resources = Vec::new();
    for member in members {
        let bytes = source::read(directory.path(), &member)?;
        let mut archive = descriptor.clone();
        archive["path"] = json!(name);
        let resource =
            json!({"path":member,"sha256":sha256(&bytes),"size":bytes.len(),"archive":archive});
        let (indexed, family, snapshot) =
            document_cached(&member, &bytes, cache).with_context(|| member.clone())?;
        resources.push((member, resource, indexed, family, snapshot));
    }
    Ok(resources)
}

fn observation_code(value: &Node) -> Result<Value> {
    let recapture = value
        .field("document")?
        .get("record_metadata")
        .and_then(|metadata| metadata.get("verified_recapture"))
        .filter(|recapture| **recapture != Node::Null);
    // Availability follows the selected checkpoint result, including its absent
    // typed observation. Archived plural proofs never select an observation.
    let observation = match recapture {
        Some(recapture) => recapture.field("result")?.get("capture_observation"),
        None => Some(value.field("observation")?),
    };
    Ok(observation
        .and_then(|observation| observation.get("unavailable"))
        .and_then(|unavailable| unavailable.get("code"))
        .map(Node::string)
        .transpose()?
        .map_or(Value::Null, |code| json!(code)))
}

pub(crate) fn record_summary(records: &Node) -> Result<Value> {
    let mut codes = serde_json::Map::new();
    let mut has_results = false;
    let mut unified = false;
    let mut split = false;
    let mut route_recorded = false;
    for (case, wrapped) in records.mapping()? {
        let value = wrapped.field("value")?;
        let observation = value.field("observation")?;
        codes.insert(case.string()?.to_string(), observation_code(value)?);
        has_results |= observation.get("value").is_some() || observation.get("error").is_some();
        let document = value.field("document")?;
        route_recorded |= document
            .get("record_metadata")
            .and_then(|n| n.get("parser_path"))
            .is_some()
            || observation
                .get("value")
                .and_then(|n| n.get("parser_path"))
                .is_some()
            || document.get("parser_path").is_some();
        let route = document
            .get("record_metadata")
            .and_then(|n| n.get("parser_path"))
            .filter(|n| crate::history::truthy(n))
            .or_else(|| {
                observation
                    .get("value")
                    .and_then(|n| n.get("parser_path"))
                    .filter(|n| crate::history::truthy(n))
            })
            .or_else(|| document.get("parser_path"));
        unified |= route == Some(&Node::Str("unified".into()));
        split |= route == Some(&Node::Str("split".into()));
    }
    Ok(
        json!({"codes":codes,"has_results":has_results,"native":unified && !split,
        "route_recorded":route_recorded,"historical_evidence":0}),
    )
}

fn document(name: &str, bytes: &[u8]) -> Result<(Value, Option<Value>, Option<Value>)> {
    let node = yaml::load_str(std::str::from_utf8(bytes)?)?;
    node.mapping()?;
    let parts = name.split('/').collect::<Vec<_>>();
    let mut indexed = json!({"schema_version": match node.get("schema_version") { Some(Node::Int(value)) => serde_json::from_str::<Value>(&value.to_string())?, _ => Value::Null }});
    let mut family = None;
    let mut snapshot = None;
    if let ["families", name, "inputs_and_golden.yaml"] = parts.as_slice() {
        let active = node
            .field("cases")?
            .mapping()?
            .iter()
            .filter(|(_, value)| value.get("lifecycle") != Some(&Node::Str("retired".into())))
            .map(|(key, _)| key.string())
            .collect::<Result<Vec<_>>>()?;
        indexed["active_cases"] = json!(active);
        family = Some(json!({"family":name,"active_cases":active}));
    } else if let ["families", family, file] = parts.as_slice() {
        let stem = file.strip_suffix(".yaml").context("non YAML source")?;
        let (implementation, version) = stem.split_once('-').context("invalid capture identity")?;
        Release::parse(version)?;
        if node.get("schema_version") != Some(&Node::Int(2.into())) {
            indexed["execution"] = node
                .get("execution")
                .map(Node::string)
                .transpose()?
                .map_or(Value::Null, |reference| json!(reference));
            return Ok((indexed, None, None));
        }
        let mut summary = record_summary(node.field("records")?)?;
        summary["family"] = json!(family);
        summary["implementation"] = json!(implementation);
        summary["version"] = json!(version);
        summary["historical_evidence"] = json!(
            node.get("historical_evidence")
                .map(Node::array)
                .transpose()?
                .map_or(0, |items| items.len())
        );
        indexed["execution"] = Value::Null;
        snapshot = Some(summary);
    } else if let ["executions", _] = parts.as_slice() {
        indexed["receipts"] = json!(
            node.get("receipts")
                .map(Node::mapping)
                .transpose()?
                .unwrap_or(&[])
                .iter()
                .map(|(key, _)| key.string())
                .collect::<Result<Vec<_>>>()?
        );
    } else if !matches!(parts.as_slice(), ["receipts", _]) {
        anyhow::bail!("unknown metadata in store: {name}");
    }
    Ok((indexed, family, snapshot))
}

pub(crate) fn valid_summary(
    summary: &Value,
    family: &str,
    implementation: &str,
    version: &str,
) -> bool {
    summary.is_object()
        && summary["family"] == family
        && summary["implementation"] == implementation
        && summary["version"] == version
        && summary["codes"].as_object().is_some_and(|codes| {
            codes
                .values()
                .all(|code| code.is_null() || code.is_string())
        })
        && ["has_results", "native", "route_recorded"]
            .iter()
            .all(|key| summary[key].is_boolean())
        && summary["historical_evidence"].is_u64()
}

fn string_list(value: &Value) -> bool {
    value
        .as_array()
        .is_some_and(|items| items.iter().all(Value::is_string))
}

fn valid_document(name: &str, result: &(Value, Option<Value>, Option<Value>)) -> bool {
    let (indexed, family, snapshot) = result;
    if !indexed.is_object()
        || !indexed
            .get("schema_version")
            .is_some_and(|version| version.is_null() || version.is_i64() || version.is_u64())
    {
        return false;
    }
    match name.split('/').collect::<Vec<_>>().as_slice() {
        ["families", name, "inputs_and_golden.yaml"] => {
            snapshot.is_none()
                && family.as_ref().is_some_and(|family| {
                    family.is_object()
                        && family["family"] == *name
                        && string_list(&family["active_cases"])
                        && family["active_cases"] == indexed["active_cases"]
                })
        }
        ["families", name, file] => {
            let Some((implementation, version)) = file
                .strip_suffix(".yaml")
                .and_then(|stem| stem.split_once('-'))
            else {
                return false;
            };
            family.is_none()
                && Release::parse(version).is_ok()
                && indexed
                    .get("execution")
                    .is_some_and(|execution| execution.is_null() || execution.is_string())
                && if indexed["schema_version"] == 2 {
                    snapshot.as_ref().is_some_and(|summary| {
                        valid_summary(summary, name, implementation, version)
                    })
                } else {
                    snapshot.is_none()
                }
        }
        ["executions", _] => {
            family.is_none() && snapshot.is_none() && string_list(&indexed["receipts"])
        }
        ["receipts", _] => family.is_none() && snapshot.is_none(),
        _ => false,
    }
}

fn document_cached(
    name: &str,
    bytes: &[u8],
    cache: Option<&Path>,
) -> Result<(Value, Option<Value>, Option<Value>)> {
    let key = sha256(&crate::codec::json_bytes(&json!([
        "catalog-v1",
        name,
        sha256(bytes),
        include_str!("catalog.rs"),
        include_str!("yaml.rs"),
        include_str!("source.rs"),
        include_str!("history.rs")
    ]))?);
    if let Some(payload) =
        cache.and_then(|directory| crate::conversion_cache::read(directory, &key))
        && let Ok(result) = serde_json::from_value(payload)
        && valid_document(name, &result)
    {
        return Ok(result);
    }
    let result = document(name, bytes)?;
    if let Some(directory) = cache {
        crate::conversion_cache::write(directory, &key, &serde_json::to_value(&result)?)?;
    }
    Ok(result)
}

/// Admit each archive once, decode members in the existing bounded worker pool,
/// and return only inventory and selector metadata in deterministic path order.
pub fn index(root: &Path, workers: &str) -> Result<Value> {
    index_cached(root, workers, None)
}

pub fn index_cached(root: &Path, workers: &str, cache: Option<&Path>) -> Result<Value> {
    let root = root.canonicalize()?;
    let _session = source::archive_session();
    let policy = Policy::detect(workers)?;
    let budget = MemoryBudget::new(policy.memory_bytes);
    let store = root.join(STORE);
    let files = source::inventory(&store)?
        .into_iter()
        .filter(|name| store.join(name).is_file())
        .collect::<Vec<_>>();
    let groups = policy.pool()?.install(|| {
        files
            .par_iter()
            .map(|name| -> Result<Vec<IndexedDocument>> {
                let source_name = format!("{STORE}/{name}");
                let mut resources = Vec::new();
                if name.starts_with("versions/") && name.ends_with(".tar.gz") {
                    resources.extend(index_version_archive(&root, name, &budget, cache)?);
                } else {
                    let _reservation = budget.reserve(
                        source::source_size(&root, &source_name)?
                            .checked_mul(64)
                            .context("catalog reservation overflow")?
                            .max(16 * 1024 * 1024),
                    )?;
                    let bytes = source::read(&root, &source_name)?;
                    let resource = json!({"path":name,"sha256":sha256(&bytes),"size":bytes.len()});
                    let (indexed, family, snapshot) =
                        document_cached(name, &bytes, cache).with_context(|| name.clone())?;
                    resources.push((name.to_string(), resource, indexed, family, snapshot));
                }
                Ok(resources)
            })
            .collect::<Result<Vec<_>>>()
    })?;
    let mut rows = BTreeMap::new();
    for group in groups {
        for (name, resource, indexed, family, snapshot) in group {
            ensure!(
                rows.insert(name, (resource, indexed, family, snapshot))
                    .is_none(),
                "duplicate Unified source"
            );
        }
    }
    let mut inventory = Vec::new();
    let mut documents = serde_json::Map::new();
    let mut families = serde_json::Map::new();
    let mut snapshots = Vec::new();
    for (name, (resource, indexed, family, snapshot)) in rows {
        inventory.push(resource);
        documents.insert(name, indexed);
        if let Some(family) = family {
            families.insert(
                family["family"].as_str().unwrap().to_string(),
                family["active_cases"].clone(),
            );
        }
        if let Some(snapshot) = snapshot {
            snapshots.push(snapshot);
        }
    }
    let context_path = root.join("conformance/catalog-conversion-context.json");
    if snapshots.is_empty() && context_path.is_file() {
        let context: Value = serde_json::from_slice(&source::read(
            &root,
            "conformance/catalog-conversion-context.json",
        )?)?;
        let mut resources = Vec::new();
        for resource in &inventory {
            let name = resource["path"].as_str().context("missing source name")?;
            let parts = name.split('/').collect::<Vec<_>>();
            let mut row = resource.clone();
            row["path"] = json!(format!("{STORE}/{name}"));
            row["family"] = Value::Null;
            row["implementation"] = Value::Null;
            row["version"] = Value::Null;
            row["kind"] = json!(if parts[0] == "executions" {
                "execution"
            } else {
                "receipt"
            });
            if parts[0] == "families" {
                row["family"] = json!(parts[1]);
                if parts[2] == "inputs_and_golden.yaml" {
                    row["kind"] = json!("inputs-and-golden");
                } else {
                    let (implementation, version) = parts[2]
                        .strip_suffix(".yaml")
                        .context("invalid capture extension")?
                        .split_once('-')
                        .context("invalid capture name")?;
                    row["kind"] = json!("history");
                    row["implementation"] = json!(implementation);
                    row["version"] = json!(version);
                }
            }
            for key in ["execution", "receipts"] {
                if !documents[name][key].is_null() {
                    row[key] = documents[name][key].clone();
                }
            }
            if let Some(archive) = row.get_mut("archive") {
                archive["path"] = json!(format!(
                    "{STORE}/{}",
                    archive["path"].as_str().context("invalid archive path")?
                ));
            }
            if let Some(dataset) = context["dataset"].as_str() {
                row["dataset"] = json!(dataset);
            }
            resources.push(row);
        }
        let index = resources
            .iter()
            .map(|r| (r["path"].as_str().unwrap().to_string(), r))
            .collect::<BTreeMap<_, _>>();
        let temporary = tempfile::tempdir()?;
        let directory = cache.unwrap_or(temporary.path());
        let groups = policy.pool()?.install(|| {
            families
                .keys()
                .collect::<Vec<_>>()
                .par_iter()
                .map(|name| -> Result<Vec<Value>> {
                    let family = resources
                        .iter()
                        .filter(|r| r["family"] == name.as_str())
                        .collect::<Vec<_>>();
                    let reserved = crate::pipeline::family_source_bytes(&family, &index)?
                        .checked_mul(64)
                        .context("catalog conversion reservation overflow")?
                        .max(16 * 1024 * 1024);
                    let _reservation = budget.reserve(reserved)?;
                    let key = crate::pipeline::family_cache_key(&family, &index, &context, name)?;
                    let mut converted = crate::pipeline::converted_family(
                        &root,
                        &family,
                        &index,
                        &context,
                        name,
                        reserved,
                        crate::pipeline::FamilyCache {
                            directory,
                            key: &key,
                            reuse: true,
                        },
                    )?;
                    Ok(converted["snapshots"]
                        .take()
                        .as_array()
                        .context("missing converted checkpoints")?
                        .iter()
                        .map(|row| {
                            let mut summary = row["summary"].clone();
                            summary["fragment"] = row["fragment"].clone();
                            summary["capture"] = row["capture"].clone();
                            summary
                        })
                        .collect())
                })
                .collect::<Result<Vec<_>>>()
        })?;
        snapshots.extend(groups.into_iter().flatten());
    }
    let evidence = snapshots
        .iter()
        .map(|s| s["historical_evidence"].as_u64().unwrap())
        .sum::<u64>();
    let records = snapshots
        .iter()
        .map(|s| s["codes"].as_object().unwrap().len())
        .sum::<usize>();
    for snapshot in &mut snapshots {
        compact_codes(snapshot, &families)?;
    }
    Ok(
        json!({"statistics":{"captures":snapshots.len(),"records":records,"historical_evidence":evidence},
        "inventory":inventory,"documents":documents,"families":families,"snapshots":snapshots}),
    )
}

fn compact_codes(snapshot: &mut Value, families: &serde_json::Map<String, Value>) -> Result<()> {
    let active = families
        .get(snapshot["family"].as_str().context("missing family")?)
        .context("snapshot family is not authored")?
        .as_array()
        .context("invalid active cases")?;
    let object = snapshot
        .as_object_mut()
        .context("invalid snapshot summary")?;
    let codes = object.remove("codes").context("missing code map")?;
    let codes = codes.as_object().context("invalid code map")?;
    let values = codes
        .values()
        .map(|code| code.as_str())
        .collect::<BTreeSet<_>>();
    object.insert("record_count".into(), json!(codes.len()));
    object.insert("code_values".into(), json!(values));
    object.insert(
        "active_cases_complete".into(),
        json!(
            active
                .iter()
                .all(|case| codes.contains_key(case.as_str().unwrap()))
        ),
    );
    object.remove("historical_evidence");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{sync::mpsc, time::Duration};

    #[test]
    fn authenticated_malformed_document_metadata_is_a_cache_miss() -> Result<()> {
        for (name, bytes) in [
            (
                "families/sample/inputs_and_golden.yaml",
                "cases: {alpha: {lifecycle: active}}\n",
            ),
            (
                "families/sample/dynamo_v2-0.1.0.yaml",
                "schema_version: 2\nrecords: {}\n",
            ),
            ("executions/manifest.yaml", "receipts: {}\n"),
        ] {
            let cache = tempfile::tempdir()?;
            let expected = document_cached(name, bytes.as_bytes(), Some(cache.path()))?;
            let file = fs::read_dir(cache.path())?
                .next()
                .context("missing cache")??
                .path();
            let key = file.file_stem().unwrap().to_str().unwrap();
            for malformed in [
                json!([{}, true, null]),
                json!([{"schema_version":null,"active_cases":[false]}, {"family":"sample","active_cases":[false]}, null]),
                json!([{"schema_version":null,"active_cases":[]}, {"family":"other","active_cases":[]}, null]),
                json!([{"schema_version":2,"execution":null}, null, {}]),
                json!([{"schema_version":null,"receipts":[false]}, null, null]),
            ] {
                crate::conversion_cache::write(cache.path(), key, &malformed)?;
                assert_eq!(
                    document_cached(name, bytes.as_bytes(), Some(cache.path()))?,
                    expected
                );
            }
        }
        Ok(())
    }

    #[test]
    fn explicit_route_presence_includes_null_and_empty() -> Result<()> {
        for scope in ["metadata", "observation", "document"] {
            for route in [
                Value::Null,
                json!(""),
                json!("unknown"),
                json!("split"),
                json!("unified"),
            ] {
                let mut value =
                    json!({"document":{"record_metadata":{}},"observation":{"value":{}}});
                let target = match scope {
                    "metadata" => &mut value["document"]["record_metadata"],
                    "observation" => &mut value["observation"]["value"],
                    _ => &mut value["document"],
                };
                target["parser_path"] = route.clone();
                let bytes = serde_json::to_vec(
                    &json!({"schema_version":2,"records":{"active":{"value":value}}}),
                )?;
                let (_, _, snapshot) = document("families/probe/dynamo_v2-2.0.0.yaml", &bytes)?;
                let snapshot = snapshot.unwrap();
                assert_eq!(snapshot["route_recorded"], true);
                assert_eq!(snapshot["native"], route == json!("unified"));
            }
        }
        Ok(())
    }

    #[test]
    fn checkpoint_codes_select_only_singular_recaptures() -> Result<()> {
        let rows = [
            (json!({}), json!("unsupported_family")),
            (
                json!({"verified_recapture":null}),
                json!("unsupported_family"),
            ),
            (
                json!({"verified_recaptures":[{"result":{"chunks":[]}}]}),
                json!("unsupported_family"),
            ),
            (
                json!({"verified_recapture":{"result":{"chunks":[]}}}),
                Value::Null,
            ),
            (
                json!({"verified_recapture":{"result":{"error":"failure"}}}),
                Value::Null,
            ),
            (
                json!({"verified_recapture":{"result":{"unavailable":"unsupported request","capture_stimulus":{}}}}),
                Value::Null,
            ),
            (
                json!({"verified_recapture":{"result":{"capture_observation":{"unavailable":{"code":"unsupported_family"}}}}}),
                json!("unsupported_family"),
            ),
            (
                json!({"verified_recapture":{"result":{"capture_observation":{"unavailable":{"code":""}}}}}),
                json!(""),
            ),
        ];
        for implementation in [
            "sglang_python",
            "vllm_python",
            "vllm_rust",
            "dynamo_v2",
            "sglang_python_batch",
        ] {
            for (metadata, expected) in &rows {
                let value = json!({"document":{"parser_path":"unified","record_metadata":metadata},
                    "observation":{"unavailable":{"code":"unsupported_family"}}});
                let bytes = serde_json::to_vec(
                    &json!({"schema_version":2,"records":{"active":{"value":value}}}),
                )?;
                let (_, _, snapshot) = document(
                    &format!("families/probe/{implementation}-0.5.0.yaml"),
                    &bytes,
                )?;
                let snapshot = snapshot.unwrap();
                assert_eq!(&snapshot["codes"]["active"], expected);
                assert_eq!(snapshot["has_results"], false);
                assert_eq!(snapshot["native"], true);
            }
        }
        let malformed = yaml::load_str(
            "document: {record_metadata: {verified_recapture: {}}}\nobservation: {}\n",
        )?;
        assert!(observation_code(&malformed).is_err());
        Ok(())
    }

    #[test]
    fn compact_codes_preserves_coverage_cardinality_and_null() -> Result<()> {
        let families =
            serde_json::Map::from_iter([("sample".into(), json!(["active", "missing"]))]);
        let mut snapshot = json!({"family":"sample", "codes":{"active":"unsupported_family", "retired":null, "other":"", "duplicate":"unsupported_family"}, "historical_evidence":1});
        compact_codes(&mut snapshot, &families)?;
        assert_eq!(snapshot["record_count"], 4);
        assert_eq!(snapshot["active_cases_complete"], false);
        assert_eq!(
            snapshot["code_values"],
            json!([null, "", "unsupported_family"])
        );
        assert!(snapshot.get("codes").is_none());
        Ok(())
    }

    #[test]
    fn archive_index_is_worker_independent_and_never_expands_sources() -> Result<()> {
        let root = tempfile::tempdir()?;
        let store = root.path().join(STORE);
        fs::create_dir_all(store.join("families/sample"))?;
        fs::create_dir_all(store.join("versions"))?;
        fs::write(
            store.join("families/sample/inputs_and_golden.yaml"),
            b"cases: {id: {lifecycle: active}}\n",
        )?;
        let snapshot = b"schema_version: 2\nrecords:\n  id:\n    value:\n      document: {record_metadata: {parser_path: null}}\n      observation: {value: {parser_path: unified}}\n";
        for version in ["0.1.0", "0.2.0"] {
            let name = format!("families/sample/dynamo_v2-{version}.yaml");
            let mut tar = tar::Builder::new(Vec::new());
            let mut header = tar::Header::new_ustar();
            header.set_size(snapshot.len() as u64);
            header.set_mode(0o644);
            header.set_cksum();
            tar.append_data(&mut header, name, snapshot.as_slice())?;
            fs::write(
                store.join(format!("versions/dynamo-{version}.tar.gz")),
                crate::codec::gzip(&tar.into_inner()?)?,
            )?;
        }
        let before = source::inventory(&store)?
            .into_iter()
            .filter(|n| store.join(n).is_file())
            .map(|n| Ok((n.clone(), fs::read(store.join(n))?)))
            .collect::<Result<BTreeMap<_, _>>>()?;
        let one = index(root.path(), "1")?;
        assert_eq!(one, index(root.path(), "4")?);
        assert_eq!(one["inventory"].as_array().unwrap().len(), 3);
        assert!(
            one["snapshots"]
                .as_array()
                .unwrap()
                .iter()
                .all(|s| s["native"] == true)
        );
        let after = source::inventory(&store)?
            .into_iter()
            .filter(|n| store.join(n).is_file())
            .map(|n| Ok((n.clone(), fs::read(store.join(n))?)))
            .collect::<Result<BTreeMap<_, _>>>()?;
        assert_eq!(before, after);
        Ok(())
    }

    #[test]
    fn archive_memory_is_reserved_before_read_or_extraction() -> Result<()> {
        let root = tempfile::tempdir()?;
        let archive = root.path().join(STORE).join("versions/dynamo-0.0.0.tar.gz");
        fs::create_dir_all(archive.parent().unwrap())?;
        fs::write(&archive, b"not a gzip archive")?;
        let budget = MemoryBudget::new(8 * 1024 * 1024);

        let error =
            match admit_version_archive(root.path(), "versions/dynamo-0.0.0.tar.gz", &budget) {
                Err(error) => error.to_string(),
                Ok(_) => panic!("archive admission exceeded its memory budget"),
            };

        assert!(error.contains("requires 50331666 bytes"));
        Ok(())
    }

    #[test]
    fn archive_inventory_releases_admission_before_member_parsing() -> Result<()> {
        let root = tempfile::tempdir()?;
        let archive = root.path().join(STORE).join("versions/dynamo-0.1.0.tar.gz");
        fs::create_dir_all(archive.parent().unwrap())?;
        let name = "families/sample/dynamo_v2-0.1.0.yaml";
        let snapshot = b"schema_version: 2\nrecords: {}\n";
        let mut tar = tar::Builder::new(Vec::new());
        let mut header = tar::Header::new_ustar();
        header.set_size(snapshot.len() as u64);
        header.set_mode(0o644);
        header.set_cksum();
        tar.append_data(&mut header, name, snapshot.as_slice())?;
        fs::write(archive, crate::codec::gzip(&tar.into_inner()?)?)?;

        let root = root.path().to_path_buf();
        let budget = MemoryBudget::new(64 * 1024 * 1024);
        let (sender, receiver) = mpsc::channel();
        std::thread::spawn(move || {
            let _ = sender.send(index_version_archive(
                &root,
                "versions/dynamo-0.1.0.tar.gz",
                &budget,
                None,
            ));
        });
        let documents = receiver
            .recv_timeout(Duration::from_secs(15))
            .context("archive indexing held admission while waiting for a second reservation")??;

        assert_eq!(documents.len(), 1);
        assert_eq!(documents[0].0, name);
        Ok(())
    }
}
