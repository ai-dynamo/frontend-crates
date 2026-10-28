# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from copy import deepcopy
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest
import ci_coverage
import capture_sources
import unified_archives
import unified_history
import migrate_capture_sources
import viewer_catalog
from fixture_test_support import native_snapshots, proof_catalog, snapshot_bytes
from viewer_codec import decode_typed, typed_equal
from verify_publication_transport import read_index
from conformance.utils.tests.test_execution_manifests import compact_store
import hashlib
from fixture_test_support import native_snapshots, runtime_catalog, snapshot_bytes
from test_unified_history import _store


ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "conformance/utils/src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def test_snapshot_oracle_binds_source_requests_without_erasing_differences():
    result = subprocess.run(["node", "--input-type=module", "-e", """
        import assert from 'node:assert/strict';
        import {boundOracleRecord} from './conformance/viewer/resources.js';
        import {sameValue} from './conformance/viewer/values.js';
        const request = {input:'original',init:{starting_state:'Reasoning'},finish_reason:'stop',tools:[{name:'probe'}],chunks:[{delta_text:'original'}]};
        const original = {stimulus:{ref:'current',semantic_sha256:'bound'},observation:{value:{assembled:[]}},document:{opaque:null}};
        const expected = { ...structuredClone(original),stimulus:{inline:request,semantic_sha256:'bound'}};
        assert(sameValue(boundOracleRecord(original,{request}), expected));
        assert.equal(original.stimulus.ref,'current');
        for (const field of ['input','init','finish_reason','tools','chunks']) {
          const changed = structuredClone(expected);
          changed.stimulus.inline[field] = null;
          assert.equal(sameValue(boundOracleRecord(original,{request}),changed),false,field);
        }
        const explicit = { ...original,stimulus:{inline:{input:'historical'}}};
        assert(sameValue(boundOracleRecord(explicit,{request}),explicit));
    """], cwd=ROOT, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("fault", [None, "missing-execution", "missing-receipt", "corrupt-execution", "corrupt-receipt", "corrupt-history", "corrupt-snapshot", "omitted-version", "omitted-family", "extra-version", "duplicate-selection", "missing-input-family", "missing-all-inputs", "extra-input-family"])
def test_compact_snapshot_parity_uses_validated_capture_reader(tmp_path, fault):
    _snapshot_parity_case(tmp_path, fault)


def test_complete_snapshot_parity_reads_source_records(tmp_path):
    _snapshot_parity_case(tmp_path, None, complete=True)


def _snapshot_parity_case(tmp_path, fault, *, complete=False):
    membership_faults = {"omitted-version", "omitted-family", "extra-version", "duplicate-selection"}
    input_faults = {"missing-input-family", "missing-all-inputs", "extra-input-family"}
    original, _first, _second, _before = compact_store(tmp_path / "seed")
    root = tmp_path / "root"
    store = root / "conformance/fixtures-unified-v2"
    shutil.copytree(original, store)
    if fault is None or fault in membership_faults or fault in input_faults:
        loaded = unified_history.load_store(store)
        for path, document in unified_history._store_documents(loaded, compact_executions=False).items():
            target = store / "families/qwen3" / path.name
            target.parent.mkdir(parents=True, exist_ok=True)
            if path.name == unified_history.FAMILY_DOCUMENT_NAME:
                document["input_document"]["family"] = "qwen3"
                document["golden_document"]["family"] = "qwen3"
            target.write_text(unified_history.dump_yaml(document))
        unified_history.compact_execution_store(store)
    sparse_resources = viewer_catalog._unified_records(root)
    sparse = tmp_path / "sparse"
    store.rename(sparse)
    migrate_capture_sources.convert_snapshots(sparse, store, tmp_path)
    site = tmp_path / "site"
    catalog = {
        "revision": "a" * 64,
        "metadata": {"unified_capture_policy": {"dynamo_v2": {"mode": "stream"}}},
        "sources": {
            "unified": viewer_catalog._unified_records(root),
            "candidates": [{"mode": "unified", "implementation": "dynamo_v2", "version": version}
                           for version in ("0.1.0", "0.2.0")],
        },
    }
    original_candidates = [dict(item) for item in catalog["sources"]["candidates"]]
    if fault == "omitted-family":
        catalog["sources"]["candidates"][0]["capture_families"] = ["gemma4"]
    elif fault == "extra-version":
        catalog["sources"]["candidates"].append({"mode": "unified", "implementation": "dynamo_v2", "version": "0.3.0"})
    published = native_snapshots(root, site, catalog)
    if not complete:
        # Sparse decoding belongs to this independent historical oracle, never
        # the production publisher. Compare both readers with the same packs.
        store.rename(tmp_path / "complete-source")
        sparse.rename(store)
        published["proof_original_unified"] = sparse_resources
    published["sources"]["candidates"] = original_candidates
    if fault == "omitted-version":
        published["sources"]["unified"] = [item for item in published["sources"]["unified"]
                                           if item.get("version") != "0.2.0"]
    elif fault == "duplicate-selection":
        published["sources"]["unified"].append(next(item for item in published["sources"]["unified"] if item["kind"] == "resolved-capture"))
    elif fault == "missing-input-family":
        published["sources"]["unified"] = [item for item in published["sources"]["unified"]
                                           if item["kind"] != "inputs-and-golden" or "qwen3" not in item.get("families", [item.get("family")])]
    elif fault == "missing-all-inputs":
        published["sources"]["unified"] = [item for item in published["sources"]["unified"] if item["kind"] != "inputs-and-golden"]
    elif fault == "extra-input-family":
        resource = next(item for item in published["sources"]["unified"] if item["kind"] == "inputs-and-golden" and item.get("family") == "gemma4")
        published["sources"]["unified"].append({**resource, "family": "unexpected_family"})
    elif fault and fault not in membership_faults:
        action, kind = fault.split("-")
        resources = published["sources"]["unified"] if kind == "snapshot" else published["proof_original_unified"]
        resource = next(item for item in resources if item["kind"] == ("resolved-capture" if kind == "snapshot" else kind))
        if action == "missing":
            resources.remove(resource)
        else:
            if kind == "snapshot":
                recipe = read_index(site, resource["recipe_index"])[resource["path"]]
                path = site / recipe["chunks"][0]["pack"]["path"]
            else:
                path = root / resource["path"]
            data = path.read_bytes()
            path.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
    catalog_path = tmp_path / "published-catalog.json"
    catalog_path.write_text(json.dumps(published))
    result = subprocess.run(
        ["node", "conformance/viewer/qualification.js", "snapshots", str(root), str(site), str(catalog_path)],
        cwd=ROOT, text=True, capture_output=True, check=False, timeout=30,
    )
    if fault:
        assert result.returncode != 0
        expected = "published authored family membership" if fault in input_faults else "published capture selection" if fault in membership_faults else "not indexed" if fault.startswith("missing-") else "SHA-256 mismatch"
        assert expected in result.stderr
    else:
        assert result.returncode == 0, result.stderr
        assert "2 authored family comparisons, 4 resolved captures, 4 complete records" in result.stdout
        assert "4 checkpoint visits, 2 capture bundles decoded, 1 retained capture bundle" in result.stdout


def test_full_corpus_json_snapshots_match_python_and_javascript_resolution(tmp_path, transport_published_viewer):
    site = transport_published_viewer
    published = proof_catalog(ROOT, site)
    if published.get("capture_generation", {}).get("mode") == "latest":
        source = tmp_path / "current-source"
        capture_sources.stage_unified(ROOT, source / capture_sources.STORE)
    else:
        with capture_sources.history_view(ROOT, ROOT / capture_sources.HISTORIC) as source:
            _assert_routed_history_parity(source, site, published)
        return
    _assert_full_corpus_parity(tmp_path, source, site, published)


def test_bundled_history_json_snapshots_match_original_yaml(tmp_path, history_source, historical_published_viewer):
    site = historical_published_viewer
    if capture_sources.history_pin(ROOT, history_source)["schema"] == 2:
        source = tmp_path / "publication-source"
        capture_sources.stage_unified(ROOT, source / capture_sources.STORE)
        capture_sources.stage_history(ROOT, source, history_source)
        datasets = json.loads((source / capture_sources.DATASETS).read_bytes())
        _assert_retained_dataset_descriptor(datasets)
        _assert_routed_history_parity(source, site, proof_catalog(ROOT, site))
        return
    with capture_sources.history_view(ROOT, history_source) as source:
        store = source / capture_sources.STORE
        if (source / capture_sources.DATASETS).is_file():
            _assert_routed_history_parity(source, site, proof_catalog(source, site))
            return
        inventory = unified_archives.inventory(store)
        originals = viewer_catalog._unified_records(source, inventory=inventory)
        published = proof_catalog(source, site, source_resources=originals)
        _assert_full_corpus_parity(tmp_path, source, site, published)


def _assert_retained_dataset_descriptor(datasets):
    assert datasets == [{"id": "retained", "root": "conformance/history-datasets/retained"}], "retained dataset namespace differs"


@pytest.mark.parametrize("fault", ["id", "root", "extra"])
def test_retained_parity_rejects_wrong_dataset_namespace(fault):
    datasets = [{"id": "retained", "root": "conformance/history-datasets/retained"}]
    _assert_retained_dataset_descriptor(datasets)
    if fault == "extra":
        datasets.append(dict(datasets[0]))
    else:
        datasets[0][fault] = "different"
    with pytest.raises(AssertionError, match="retained dataset namespace differs"):
        _assert_retained_dataset_descriptor(datasets)


def _source_capture(owners, family, implementation, version):
    captures = [(unified_history._capture_release_sort_key(capture["runtime_version"]), priority, prefix, store, history, identity)
                for priority, prefix, store in owners
                if (history := store.histories.get((family, implementation))) is not None
                for identity, capture in history.captures.items()
                if unified_history._capture_release_sort_key(capture["runtime_version"])
                <= unified_history._capture_release_sort_key(version)]
    assert captures, (family, version)
    return max(captures, key=lambda item: item[:2])[2:]


def test_original_source_precedence_keeps_dataset_order_and_current_override():
    stores = []
    for priority, prefix in [(0, "first"), (1, "second"), (2, "")]:
        history = SimpleNamespace(captures={"dynamo_v2-0.1.0": {"runtime_version": "0.1.0"}})
        stores.append((priority, prefix, SimpleNamespace(histories={("sample", "dynamo_v2"): history})))
    assert _source_capture(stores[:2], "sample", "dynamo_v2", "0.1.0")[0] == "second"
    assert _source_capture(stores, "sample", "dynamo_v2", "0.1.0")[0] == ""
    stores[0][2].histories["sample", "dynamo_v2"].captures["dynamo_v2-0.2.0"] = {"runtime_version": "0.2.0"}
    assert _source_capture(stores, "sample", "dynamo_v2", "0.2.0")[0] == "first"
    assert _source_capture(stores, "sample", "dynamo_v2", "0.1.0")[0] == ""


def _assert_routed_history_parity(source, site, published):
    # Read original YAML directly. Routed load_store consumes publisher-created
    # fragments and would make the expected result circular with the publication.
    datasets = json.loads((source / capture_sources.DATASETS).read_bytes())
    owners = [(len(datasets), "", unified_history._load_store_unlocked(source / capture_sources.STORE))]
    owners.extend((index, dataset["root"], unified_history._load_store_unlocked(source / dataset["root"] / capture_sources.STORE))
                  for index, dataset in enumerate(datasets))
    resources = ci_coverage.select_resources([row for row in published["sources"]["unified"]
                                             if row["kind"] == "resolved-capture"])
    checked = 0
    for resource in resources:
        bundle = decode_typed(json.loads(snapshot_bytes(site, resource))["value"])
        for family in resource.get("families", [resource.get("family")]):
            prefix, store, history, selected = _source_capture(owners, family, resource["implementation"], resource["version"])
            requested = f"{resource['implementation']}-{resource['version']}"
            expected = history.resolve(selected, view_capture_id=requested)
            snapshot = history.complete_snapshot(selected)
            records = bundle[family] if "families" in resource else bundle
            assert set(records) == set(expected), (family, resource["version"])
            members = {case: wrapped["member"] for case, wrapped in snapshot["records"].items()}
            for case, value in expected.items():
                value = deepcopy(value)
                if value["stimulus"].get("ref") == "current":
                    value["stimulus"].pop("ref")
                    value["stimulus"]["inline"] = deepcopy(store.families[family].cases[case]["request"])
                assert typed_equal(records[case]["value"], value), (family, resource["version"], case)
                expected_member = f"{prefix}/{members[case]}" if prefix else members[case]
                assert (source / expected_member).is_file(), expected_member
                assert records[case]["member"] == expected_member
                checked += 1
    assert checked > 0
    print(f"routed history transport parity passed: {len(resources)} capture bundles, {checked} records")


def _assert_full_corpus_parity(tmp_path, source, site, published):
    catalog_path = tmp_path / "published-catalog.json"
    catalog_path.write_text(json.dumps(published))
    selection = tmp_path / "selection.json"
    resources = [row for row in published["sources"]["unified"] if row["kind"] == "resolved-capture"]
    selected = ci_coverage.select_resources(resources)
    selection.write_text(json.dumps([row["path"] for row in selected]))
    result = subprocess.run(
        ["node", "conformance/viewer/qualification.js", "snapshots", str(source), str(site), str(catalog_path), str(selection)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=60 * 60,
    )
    assert result.returncode == 0, result.stderr
    assert "full corpus parity passed:" in result.stdout
    print(f"parity selected {len(selected)}/{len(resources)} published capture bundles")
    print("\n".join(result.stdout.splitlines()[-2:]))


ROOT = Path(__file__).resolve().parents[3]


def test_snapshot_generation_resolves_history_and_explicit_empty_scope(tmp_path):
    root = tmp_path / "source"
    _store(root / "conformance/fixtures-unified-v2")
    catalog = {"sources": {"candidates": [{"mode": "unified", "implementation": "dynamo_v2", "version": version,
                                           **({"capture_families": []} if version == "0.5.3" else {})}
                                          for version in ("0.5.0", "0.5.2", "0.5.3")]},
               "metadata": {"unified_capture_policy": {"dynamo_v2": {"mode": "stream"}}}}
    site = tmp_path / "site"
    published = native_snapshots(root, site, catalog)
    resources = published["sources"]["unified"]
    assert len(resources) == 3
    assert not any(resource["version"] == "0.5.3" for resource in resources)
    for resource in resources:
        data = snapshot_bytes(site, resource)
        assert len(data) == resource["size"]
        assert hashlib.sha256(data).hexdigest() == resource["sha256"]
        decoded = data
        assert decoded.endswith(b"\n") and not decoded.endswith(b"\n\n")
        assert b"sourceBytes" not in decoded
    assert not (site / "conformance/viewer-startup.json").exists()




def test_startup_publication_is_content_addressed_and_default_scoped(published_viewer):
    site = published_viewer
    catalog = runtime_catalog(site)
    assert catalog["transport"]["schema"] == 2
    for path in ("viewer-catalog.json", "viewer-unified-catalog.json", "viewer-resolved-catalog.json", "viewer-startup.json", "viewer-original-resources.json.gz"):
        assert not (site / "conformance" / path).exists()
    reference = catalog["metadata"]["selection_policy"]["unified"]["reference"]
    candidates = [candidate for candidate in catalog["sources"]["candidates"] if candidate["implementation"] == reference]
    selected = max(candidates, key=lambda candidate: tuple(int(part) for part in candidate["version"].split(".")))
    families = selected["native_unified_families" if reference == "dynamo_v2" else "capture_families"]
    defaults = [resource for resource in catalog["sources"]["unified"] if (resource["kind"] == "inputs-and-golden" and resource["family"] in families) or (resource["implementation"] == reference and resource["version"] == selected["version"])]
    default_packs = {}
    for resource in defaults:
        recipe = read_index(site, resource["recipe_index"])[resource["path"]]
        for chunk in recipe["chunks"]:
            pack = chunk["pack"]
            entry = default_packs.setdefault(pack["path"], {"size": pack["decoded_size"], "ranges": set()})
            entry["ranges"].add((chunk["offset"], chunk["size"]))
    assert default_packs
    for pack in default_packs.values():
        end = 0
        for offset, length in sorted(pack["ranges"]):
            assert offset == end, "default pack contains bytes outside default-view dependencies"
            end += length
        assert end == pack["size"]
