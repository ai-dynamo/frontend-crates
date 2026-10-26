# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import execution_manifests
import unified_history
from capture_unified_job import build_job
from import_peer_unified import import_capture_groups, digest
from conformance.utils.tests.test_unified_history import _change, _write_capture, _write_family, _request
from viewer_codec import canonical_typed, encode_typed, typed_equal
from viewer_catalog import _unified_records
from conformance.utils.tests.test_import_peer_unified import _receipted_jobs


@pytest.mark.parametrize("before,after", [(True, 1), (1, 1.0), (0.0, -0.0), ({}, {"null": None}),
    ({1: "same"}, {True: "same"}), ({1: True, "zero": -0.0}, {1: 1, "zero": 0.0})])
def test_history_deltas_preserve_typed_value_changes(before, after):
    assert not typed_equal(before, after)
    assert canonical_typed(before) != canonical_typed(after)
    old = {"case": {"case_key": "case", "stimulus": {"partial": before}, "observation": {"value": before},
                    "document": {"record_metadata": {"typed": before}, "capture_origin": {"typed": before}}}}
    new = copy.deepcopy(old)
    new["case"]["stimulus"]["partial"] = after
    new["case"]["observation"]["value"] = after
    new["case"]["document"]["record_metadata"]["typed"] = after
    new["case"]["document"]["capture_origin"]["typed"] = after
    for preserve in (True, False):
        changes, metadata, overrides = unified_history._snapshot_delta(new, old, preserve_binding_details=preserve)
        assert typed_equal(changes["case"]["stimulus"], new["case"]["stimulus"])
        assert typed_equal(changes["case"]["observation"], new["case"]["observation"])
        assert typed_equal(metadata["case"], new["case"]["document"]["record_metadata"])
        assert typed_equal(overrides["case"]["capture_provenance"]["capture_origin"], {"typed": after})


def test_typed_mapping_comparison_ignores_order_without_stringifying_keys():
    first = {1: {"null": None, "zero": -0.0}, "1": [True, 1]}
    second = {"1": [True, 1], 1: {"zero": -0.0, "null": None}}
    assert canonical_typed(first) == canonical_typed(second)
    assert typed_equal(first, second)


@pytest.mark.parametrize("implementation", ["vllm_python", "vllm_rust", "sglang_python"])
@pytest.mark.parametrize("already_compacted", [False, True])
def test_compact_historical_insertion_preserves_mixed_key_metadata_and_receipts(tmp_path, implementation, already_compacted):
    store = tmp_path / "store"
    _write_family(store)
    for version in ("0.26.0", "0.27.0"):
        path = _write_capture(store, version, {"text_only": _change(version)}, implementation=implementation)
        document = unified_history.load_yaml(path)
        receipt = {"version": version, "job_sha256": "a" * 64, "adapter_sha256": "b" * 64, "source_sha256": "c" * 64}
        document["metadata_changes"] = {"text_only": {implementation + "_capture_provenance": receipt,
            "verified_recaptures": [{"provenance": receipt, "result": {"old": None}}],
            "typed": {1: True, "zero": -0.0}, "missing_vs_null": None}}
        path.write_text(unified_history.dump_yaml(document))
    before = dict(unified_history.load_store(store).histories[("gemma4", implementation)].iter_resolved_snapshots())
    if already_compacted:
        unified_history.compact_execution_store(store)
    job = tmp_path / "job.json"
    job.write_text(json.dumps(build_job(store / "families", "gemma4")))
    entries = []
    for version in ("0.25.1", "0.26.1"):
        capture = tmp_path / (version + ".json")
        version_key = {"vllm_python": "vllm_version", "vllm_rust": "vllm_rust_version", "sglang_python": "sglang_version"}[implementation]
        capture.write_text(json.dumps({version_key: version, "results": {"UNIFIED.text_only.gemma4": {
            "capture_input": _request(), "assembled": [{"kind": "text", "text": version}],
            "chunks": [[{"kind": "text", "text": version}]]}}}))
        receipt = {"implementation": implementation, "version": version, "job_sha256": digest(job),
                   "source_sha256": "d" * 64, "adapter_sha256": "e" * 64}
        entries.append((job, capture, receipt))
    import_capture_groups(store, {implementation: entries}, compact_executions=True)
    after = dict(unified_history.load_store(store).histories[("gemma4", implementation)].iter_resolved_snapshots())
    assert all(typed_equal(records, after[identity]) for identity, records in before.items())
    assert len(after) == 4
    assert (store / "executions").is_dir()
    assert unified_history.compact_execution_store(store) == []


def test_compaction_checks_full_record_parity_before_publication(tmp_path, monkeypatch):
    store = tmp_path / "store"
    _write_family(store)
    path = _write_capture(store, "0.1.0", {"text_only": _change()})
    document = unified_history.load_yaml(path)
    document["metadata_changes"] = {"text_only": {"note": "original"}}
    path.write_text(unified_history.dump_yaml(document))
    before = {path.relative_to(store): path.read_bytes() for path in store.rglob("*.yaml")}
    original_documents = unified_history._store_documents

    def corrupt_valid_manifest(loaded, **options):
        documents = original_documents(loaded, **options)
        manifest_path = next(path for path in documents if path.parent.name == "executions")
        manifest = documents.pop(manifest_path)
        manifest["defaults"]["note"] = "silently changed"
        documents[manifest_path] = manifest
        return documents

    monkeypatch.setattr(unified_history, "_store_documents", corrupt_valid_manifest)
    with pytest.raises(ValueError, match="changed decoded records"):
        unified_history.compact_execution_store(store)
    assert {path.relative_to(store): path.read_bytes() for path in store.rglob("*.yaml")} == before
    assert not (store / "executions").exists()


def test_unchanged_emission_cannot_skip_expected_record_parity(tmp_path):
    root = tmp_path / "store"
    _write_family(root)
    _write_capture(root, "0.1.0", {"text_only": _change()})
    expected = unified_history.load_store(root)
    documents = unified_history._store_documents(expected)
    expected.histories[("gemma4", "dynamo_v2")].captures["dynamo_v2-0.1.0"]["metadata_changes"] = {"text_only": {"receipt": "missing execution"}}
    with pytest.raises(ValueError, match="changed decoded records"):
        unified_history._commit_store_documents(root, documents, expected_store=expected)


def test_factoring_preserves_missing_null_types_and_nested_receipts():
    document = {"provenance": {}, "document": {}, "document_overrides": {}, "metadata_changes": {
        "a": {"receipt": {"source": "a" * 64, "version": "1.0.0"}, "only_a": None, "type": {"v": True}, "empty": {}, "key_type": {1: "value"}},
        "b": {"receipt": {"source": "a" * 64, "version": "1.0.0"}, "type": {"v": 1}, "empty": {}, "key_type": {True: "value"}},
    }}
    identity = {"family": "sample", "implementation": "dynamo_v2", "version": "1.0.0"}
    manifest = execution_manifests.compact(document, **identity)
    assert "source" in manifest["defaults"]["receipt"]
    assert "only_a" not in manifest["defaults"]
    assert "type" not in manifest["defaults"]
    assert "key_type" not in manifest["defaults"]
    restored = execution_manifests.expand(manifest, **identity)
    assert execution_manifests._equal(restored, document)
    assert restored["metadata_changes"]["a"]["type"]["v"] is True
    assert type(restored["metadata_changes"]["b"]["type"]["v"]) is int
    assert type(next(iter(restored["metadata_changes"]["b"]["key_type"]))) is bool


def compact_store(tmp_path):
    store = tmp_path / "store"
    _write_family(store)
    first = _write_capture(store, "0.1.0", {"text_only": _change()})
    document = unified_history.load_yaml(first)
    document["metadata_changes"] = {"text_only": {"receipt": {"source": "a" * 64}, "null": None, "typed": {1: True, "negative_zero": -0.0},
        "dynamo_v2_capture_provenance": {"version": "0.1.0", "job_sha256": "a" * 64, "adapter_sha256": "b" * 64, "crate_checksum": None}}}
    first.write_text(unified_history.dump_yaml(document))
    second = _write_capture(store, "0.2.0", {})
    document = unified_history.load_yaml(second)
    document["metadata_changes"] = {"text_only": {"receipt": {"source": "b" * 64},
        "dynamo_v2_capture_provenance": {"version": "0.2.0", "job_sha256": "a" * 64, "adapter_sha256": "b" * 64, "crate_checksum": None}}}
    second.write_text(unified_history.dump_yaml(document))
    before = {identity: records for identity, records in unified_history.load_store(store).histories[("gemma4", "dynamo_v2")].iter_resolved_snapshots()}
    unified_history.compact_execution_store(store)
    return store, first, second, before


def test_compact_checkpoint_record_parity_and_idempotence(tmp_path):
    store, first, second, before = compact_store(tmp_path)
    assert set(unified_history.load_yaml(second)) == {"parent", "execution"}
    assert unified_history.load_yaml(second)["parent"] == "dynamo_v2-0.1.0"
    after = dict(unified_history.load_store(store).histories[("gemma4", "dynamo_v2")].iter_resolved_snapshots())
    assert after == before
    assert len(list((store / "executions").glob("*.yaml"))) == 2
    assert len(list((store / "receipts").glob("*.yaml"))) == 2
    assert unified_history.compact_execution_store(store) == []
    assert set(unified_history.load_yaml(first)) == {"execution", "changes"}


def assert_javascript_records(store, before, tmp_path):
    expected = tmp_path / "expected.json"
    expected.write_text(json.dumps(encode_typed(before)))
    script = r'''
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { parseYamlBytes } from './conformance/viewer/yaml_data.js';
import { validateUnifiedCapture } from './conformance/viewer/resources.js';
import { decodeTyped } from './conformance/viewer/values.js';
import { resolveUnified } from './conformance/viewer/model.js';
const [root, expectedPath] = process.argv.slice(1);
const expected = decodeTyped(JSON.parse(fs.readFileSync(expectedPath, 'utf8')));
const inputs = parseYamlBytes(fs.readFileSync(`${root}/families/gemma4/inputs_and_golden.yaml`), { unified: true });
const captures = [];
for (const identity of Object.keys(expected)) {
  const implementation = identity.slice(0, identity.lastIndexOf('-'));
  const version = identity.slice(identity.lastIndexOf('-') + 1);
  const path = `${root}/families/gemma4/${identity}.yaml`;
  const bytes = fs.readFileSync(path);
  const value = parseYamlBytes(bytes, { unified: true });
  let manifest = null;
  if (value.execution) {
    const sourceBytes = fs.readFileSync(`${root}/${value.execution}`);
    manifest = { value: parseYamlBytes(sourceBytes, { unified: true }), sourceBytes };
    manifest.receipts = new Map(Object.keys(manifest.value.receipts || {}).map(digest => {
      const sourceBytes = fs.readFileSync(`${root}/${digest}`);
      return [digest, { sourceBytes, value: parseYamlBytes(sourceBytes, { unified: true }) }];
    }));
  }
  const resource = { family: 'gemma4', implementation, version, path };
  const document = await validateUnifiedCapture(value, inputs, resource, manifest);
  captures.push({ ...resource, bytes, document });
  assert.deepEqual(Object.fromEntries([...resolveUnified(captures, version)].map(([id, record]) => [id, record.value])), expected[identity]);
}
'''
    result = subprocess.run(["node", "--input-type=module", "-e", script, str(store), str(expected)],
        cwd=Path(__file__).resolve().parents[3], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


def test_python_and_javascript_read_the_same_compact_yaml(tmp_path):
    store, first, second, before = compact_store(tmp_path)
    assert_javascript_records(store, before, tmp_path)


def test_filename_references_read_yaml_edits_without_renaming(tmp_path):
    store, first, _second, before = compact_store(tmp_path)
    reference = unified_history.load_yaml(first)["execution"]
    assert reference == "executions/gemma4--dynamo_v2-0.1.0.yaml"
    path = store / reference
    manifest = unified_history.load_yaml(path)
    receipt = next(iter(manifest["receipts"]))
    assert receipt == "receipts/gemma4--dynamo_v2-0.1.0--receipt-001.yaml"
    for dependency in (path, store / receipt):
        dependency.write_bytes(dependency.read_bytes() + b"\n# Formatting does not change file identity.\n")
    history = unified_history.load_store(store).histories[("gemma4", "dynamo_v2")]
    assert dict(history.iter_resolved_snapshots()) == before
    assert_javascript_records(store, before, tmp_path)
    manifest["defaults"]["note"] = "edited in place"
    path.write_text(unified_history.dump_yaml(manifest))
    history = unified_history.load_store(store).histories[("gemma4", "dynamo_v2")]
    assert history.resolve("dynamo_v2-0.1.0")["text_only"]["document"]["record_metadata"]["note"] == "edited in place"
    assert unified_history.load_yaml(first)["execution"] == reference
    assert_javascript_records(store, dict(history.iter_resolved_snapshots()), tmp_path)


@pytest.mark.parametrize("reference", ["../outside.yaml", "executions/../outside.yaml", "executions/nested/file.yaml",
                                       "receipts/file.yaml", "executions/file.json", "A" * 64, "a" * 63])
def test_execution_references_require_a_filename_in_the_owning_directory(reference):
    with pytest.raises(ValueError, match="filename reference"):
        execution_manifests.source_reference(reference, "executions")


def proof_store(tmp_path, *, extra_unsupported=False):
    store = tmp_path / "store"
    _write_family(store)
    if extra_unsupported:
        path = store / "families/gemma4/inputs_and_golden.yaml"
        document = unified_history.load_yaml(path)
        case = copy.deepcopy(document["cases"]["text_only"])
        case["scenario"] = "extra"
        case["display_id"] = "extra"
        case["historical_ids"] = []
        document["cases"]["extra"] = case
        path.write_text(unified_history.dump_yaml(document))
    _write_capture(store, "0.0.1", {"text_only": _change()})
    implementation = "vllm_python_batch"
    rows = _receipted_jobs(store, tmp_path / "captures", implementation, "batch", versions=("0.1.0", "0.2.0"))
    for job, path, receipt in rows:
        value = json.loads(path.read_text())
        result = value["results"]["UNIFIED.text_only.gemma4"]
        result["assembled"] = [{"kind": "text", "text": "same"}]
        result["batch_final"]["normal_text"] = "same"
        if extra_unsupported:
            value["results"]["UNIFIED.extra.gemma4"] = {"capture_input": _request(), "unavailable": "unsupported"}
        receipt["results_sha256"] = hashlib.sha256(json.dumps(value["results"], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        value["provenance"] = receipt
        path.write_text(json.dumps(value))
    import_capture_groups(store, {implementation: rows}, compact_executions=True)
    unified_history._mutate_store(store, lambda loaded: loaded.histories[("gemma4", implementation)].captures["vllm_python_batch-0.2.0"].pop("execution_proof", None), compact_executions=True)
    history = unified_history.load_store(store).histories[("gemma4", implementation)]
    before = dict(history.iter_resolved_snapshots())
    checkpoint = store / "families/gemma4/vllm_python_batch-0.2.0.yaml"
    assert "execution" not in unified_history.load_yaml(checkpoint)
    return store, implementation, rows, before, checkpoint


def test_proof_only_unchanged_checkpoint_preserves_full_records_and_shared_readers(tmp_path):
    store, implementation, rows, before, checkpoint = proof_store(tmp_path)
    import_capture_groups(store, {implementation: [rows[1]]}, compact_executions=True, proof_only=True)
    loaded = unified_history.load_store(store)
    after = dict(loaded.histories[("gemma4", implementation)].iter_resolved_snapshots())
    assert typed_equal(before, after)
    raw = unified_history.load_yaml(checkpoint)
    assert raw["parent"] == "vllm_python_batch-0.1.0"
    manifest = unified_history.load_yaml(store / raw["execution"])
    assert manifest["cases"] == []
    assert_javascript_records(store, before, tmp_path)
    assert import_capture_groups(store, {implementation: [rows[1]]}, compact_executions=True, proof_only=True) == []


def test_proof_only_keeps_document_overrides(tmp_path):
    store, implementation, rows, _before, checkpoint = proof_store(tmp_path)
    value = unified_history.load_yaml(checkpoint)
    value["document_overrides"] = {"text_only": {"parser_path": "original-path"}}
    checkpoint.write_text(unified_history.dump_yaml(value))
    before = dict(unified_history.load_store(store).histories[("gemma4", implementation)].iter_resolved_snapshots())
    import_capture_groups(store, {implementation: [rows[1]]}, compact_executions=True, proof_only=True)
    after = dict(unified_history.load_store(store).histories[("gemma4", implementation)].iter_resolved_snapshots())
    assert typed_equal(before, after)
    assert_javascript_records(store, before, tmp_path)


@pytest.mark.parametrize("outcome", ["parser_error", "empty_output", "mixed_supported"])
def test_proof_only_admits_real_errors_empty_outputs_and_mixed_capabilities(tmp_path, outcome):
    store, implementation, rows, before, _checkpoint = proof_store(tmp_path, extra_unsupported=outcome == "mixed_supported")
    job, path, receipt = rows[1]
    value = json.loads(path.read_text())
    if outcome == "parser_error":
        value["results"]["UNIFIED.text_only.gemma4"] = {"capture_input": _request(), "error": "parser failed"}
    elif outcome == "empty_output":
        value["results"]["UNIFIED.text_only.gemma4"] = {"capture_input": _request(), "assembled": [], "batch_final": {"normal_text": "", "reasoning_text": "", "calls": []}}
    receipt["results_sha256"] = hashlib.sha256(json.dumps(value["results"], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    value["provenance"] = receipt
    path.write_text(json.dumps(value))
    import_capture_groups(store, {implementation: [(job, path, receipt)]}, compact_executions=True, proof_only=True)
    loaded = unified_history.load_store(store)
    after = dict(loaded.histories[("gemma4", implementation)].iter_resolved_snapshots())
    assert typed_equal(before, after)
    proof = loaded.histories[("gemma4", implementation)].captures[f"{implementation}-0.2.0"]["execution_proof"]
    assert set(proof["requests"]) == ({"text_only", "extra"} if outcome == "mixed_supported" else {"text_only"})
    assert_javascript_records(store, before, tmp_path)


@pytest.mark.parametrize("corruption,proof_only", [(kind, True) for kind in ("unsupported_only", "no_outcome", "mixed_outcome", "invalid_events", "wrong_version", "altered_request", "receipt_body")] + [(kind, False) for kind in ("no_outcome", "mixed_outcome", "invalid_events", "altered_request")])
def test_proof_only_admission_rejects_unexecuted_or_unbound_capture(tmp_path, corruption, proof_only):
    store, implementation, rows, _before, _checkpoint = proof_store(tmp_path)
    job, path, receipt = rows[1]
    value = json.loads(path.read_text())
    result = value["results"]["UNIFIED.text_only.gemma4"]
    if corruption == "unsupported_only":
        value["results"]["UNIFIED.text_only.gemma4"] = {"capture_input": result["capture_input"], "unavailable": "unsupported"}
    elif corruption == "no_outcome":
        value["results"]["UNIFIED.text_only.gemma4"] = {"capture_input": result["capture_input"]}
    elif corruption == "mixed_outcome":
        result["unavailable"] = "unsupported"
    elif corruption == "invalid_events":
        result["batch_final"] = {"normal_text": 1, "reasoning_text": "", "calls": []}
        result["assembled"] = [{"kind": "text", "text": 1}]
    elif corruption == "wrong_version":
        receipt["version"] = value["version"] = "9.9.9"
    elif corruption == "altered_request":
        result["capture_input"]["input"] = "altered"
    else:
        value["provenance"]["artifact_sha256"] = "0" * 64
    receipt["results_sha256"] = hashlib.sha256(json.dumps(value["results"], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    if corruption != "receipt_body":
        value["provenance"] = receipt
    path.write_text(json.dumps(value))
    before = {p.relative_to(store): p.read_bytes() for p in store.rglob("*.yaml")}
    with pytest.raises(ValueError):
        import_capture_groups(store, {implementation: [(job, path, receipt)]}, compact_executions=True, proof_only=proof_only)
    assert before == {p.relative_to(store): p.read_bytes() for p in store.rglob("*.yaml")}


@pytest.mark.parametrize("corruption", ["version", "requests", "omitted_request", "extra_request", "null_proof", "unowned_defaults", "membership", "receipt_version"])
def test_proof_only_readers_reject_corrupt_receipts_and_membership(tmp_path, corruption):
    store, implementation, rows, before, checkpoint = proof_store(tmp_path)
    import_capture_groups(store, {implementation: [rows[1]]}, compact_executions=True, proof_only=True)
    raw = unified_history.load_yaml(checkpoint)
    manifest_path = store / raw["execution"]
    manifest = unified_history.load_yaml(manifest_path)
    if corruption == "version":
        manifest["version"] = "9.9.9"
    elif corruption == "requests":
        manifest["proof"]["requests"]["text_only"] = "0" * 64
    elif corruption == "membership":
        manifest["receipts"][next(iter(manifest["receipts"]))] = []
    elif corruption == "omitted_request":
        manifest["proof"]["requests"] = {}
    elif corruption == "extra_request":
        manifest["proof"]["requests"]["extra"] = "0" * 64
    elif corruption == "null_proof":
        manifest["proof"] = None
    elif corruption == "unowned_defaults":
        manifest["defaults"] = {"unowned": True}
    else:
        receipt = store / next(iter(manifest["receipts"]))
        value = unified_history.load_yaml(receipt)
        value["version"] = "9.9.9"
        receipt.write_text(unified_history.dump_yaml(value))
    manifest_path.write_text(unified_history.dump_yaml(manifest, shared_metadata=True))
    checkpoint.write_text(unified_history.dump_yaml(raw))
    with pytest.raises(ValueError):
        unified_history.load_store(store)
    with pytest.raises(AssertionError):
        assert_javascript_records(store, before, tmp_path)


def test_receipts_shared_by_subsets_and_two_checkpoints_have_one_global_body(tmp_path):
    store = tmp_path / "store"
    _write_family(store)
    family_path = store / "families/gemma4/inputs_and_golden.yaml"
    family = unified_history.load_yaml(family_path)
    template = family["cases"].pop("text_only")
    family["cases"] = {name: {**copy.deepcopy(template), "display_id": f"UNIFIED.1-{index}", "scenario": name}
                       for index, name in enumerate(("a", "b", "c"), 1)}
    family_path.write_text(unified_history.dump_yaml(family))
    first_receipt = {"version": "0.1.0", "job_sha256": "a" * 64, "adapter_sha256": "b" * 64}
    second_receipt = {**first_receipt, "job_sha256": "c" * 64}
    for version in ("0.1.0", "0.2.0"):
        changes = {case: {**_change(), "case_key": value["display_id"]} for case, value in family["cases"].items()} if version == "0.1.0" else {}
        path = _write_capture(store, version, changes)
        document = unified_history.load_yaml(path)
        document["metadata_changes"] = {case: {"dynamo_v2_capture_provenance": first_receipt if case in {"a", "b"} else second_receipt} for case in family["cases"]}
        path.write_text(unified_history.dump_yaml(document))
    before = dict(unified_history.load_store(store).histories[("gemma4", "dynamo_v2")].iter_resolved_snapshots())
    unified_history.compact_execution_store(store)
    assert len(list((store / "executions").glob("*.yaml"))) == 2
    assert len(list((store / "receipts").glob("*.yaml"))) == 2
    after = dict(unified_history.load_store(store).histories[("gemma4", "dynamo_v2")].iter_resolved_snapshots())
    assert execution_manifests._equal(before, after)
    for path in (store / "executions").glob("*.yaml"):
        assert sorted(unified_history.load_yaml(path)["receipts"].values()) == [["a", "b"], ["c"]]


def test_post_release_capture_identity_order_catalog_and_materialization(tmp_path):
    store = tmp_path / "store"
    _write_family(store)
    versions = ["0.4.4", "0.4.4.post2", "0.4.4.post3", "0.5.4", "0.5.4.post1", "0.5.4.post2", "0.5.5", "0.8.5", "0.8.5.post1", "0.9.0.1", "0.10.1.1"]
    for version in reversed(versions):
        _write_capture(store, version, {"text_only": _change(version)}, implementation="sglang_python")
    loaded = unified_history.load_store(store)
    history = loaded.histories[("gemma4", "sglang_python")]
    assert history.ordered_capture_ids() == [f"sglang_python-{version}" for version in versions]
    sources = _unified_records(unified=store)
    assert {source["version"] for source in sources if source["kind"] == "history"} == set(versions)
    destination = tmp_path / "materialized"
    unified_history.materialize_store(store, destination, store=loaded)
    for version in versions:
        assert list(destination.rglob(f"sglang_python-{version}"))


@pytest.mark.parametrize("invalid", ["0.5.4.post", "0.5.4.post2junk", "0.5.4.patch1", "0.5.4+source.abcd", "1.2"])
def test_unified_post_release_validation_rejects_malformed_versions(invalid):
    assert unified_history.UNIFIED_VERSION_RE.fullmatch(invalid) is None
    with pytest.raises(ValueError, match="invalid Unified capture version"):
        unified_history._capture_release_sort_key(invalid)


@pytest.mark.parametrize("corruption", ["version", "membership", "parent", "missing", "receipt_missing", "receipt_membership", "undeclared_receipt"])
def test_compact_checkpoint_rejects_invalid_manifests(tmp_path, corruption):
    store, first, second, _before = compact_store(tmp_path)
    reference = unified_history.load_yaml(second)
    manifest_path = store / reference["execution"]
    manifest = unified_history.load_yaml(manifest_path)
    if corruption == "parent":
        reference["parent"] = "dynamo_v1-0.1.0"
    elif corruption == "missing":
        manifest_path.unlink()
    elif corruption == "receipt_missing":
        receipt_path = store / next(iter(manifest["receipts"]))
        receipt_path.unlink()
    else:
        if corruption == "version":
            manifest["version"] = "0.9.0"
        elif corruption == "receipt_membership":
            manifest["receipts"][next(iter(manifest["receipts"]))] = []
        elif corruption == "undeclared_receipt":
            manifest["receipts"] = {}
        else:
            manifest["cases"] = []
        manifest_path.write_text(unified_history.dump_yaml(manifest))
    second.write_text(unified_history.dump_yaml(reference))
    with pytest.raises((ValueError, FileNotFoundError)):
        unified_history.load_store(store)


@pytest.mark.parametrize("implementation,mode", [("dynamo_v1", "batch"), ("dynamo_v1_jail", "batch"), ("dynamo_v2", "stream"), ("vllm_python_batch", "batch"), ("sglang_python_batch", "batch"), ("vllm_python", "stream"), ("sglang_python", "stream"), ("vllm_rust", "stream")])
def test_ordinary_verified_import_retains_receipt_without_duplicate_proof(tmp_path, implementation, mode):
    store = tmp_path / "store"
    _write_family(store)
    _write_capture(store, "0.0.1", {"text_only": _change()})
    rows = _receipted_jobs(store, tmp_path / "captures", implementation, "unified" if implementation in {"vllm_python", "sglang_python", "vllm_rust"} else mode, versions=("0.1.0", "0.2.0"))
    for job, path, receipt in rows:
        document = json.loads(path.read_text())
        result = document["results"]["UNIFIED.text_only.gemma4"]
        result["assembled"] = [{"kind": "text", "text": "same"}]
        if mode == "batch":
            result["batch_final"]["normal_text"] = "same"
        else:
            result["chunks"] = [[{"kind": "text", "text": "same"}]]
        receipt["results_sha256"] = hashlib.sha256(json.dumps(document["results"], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        document["provenance"] = receipt
        path.write_text(json.dumps(document))
    import_capture_groups(store, {implementation: rows}, compact_executions=True)
    loaded = unified_history.load_store(store)
    item = loaded.histories[("gemma4", implementation)]
    before = dict(item.iter_resolved_snapshots())
    for _job, _path, receipt in rows:
        identity = f"{implementation}-{receipt['version']}"
        checkpoint = item.captures[identity]
        ordinary = any(typed_equal(owner[key], receipt) for metadata in checkpoint["metadata_changes"].values() for owner, key in execution_manifests._receipt_slots(metadata))
        proof = checkpoint.get("execution_proof")
        assert ordinary != (proof is not None)
        assert ordinary or typed_equal(proof["receipt"], receipt)
        raw = unified_history.load_yaml(store / "families/gemma4" / (identity + ".yaml"))
        assert "execution" in raw
        if receipt["version"] == "0.2.0":
            assert raw["parent"] == f"{implementation}-0.1.0"
    assert_javascript_records(store, before, tmp_path)
    if implementation in {"vllm_python_batch", "sglang_python_batch", "dynamo_v2", "dynamo_v1", "dynamo_v1_jail"}:
        assert import_capture_groups(store, {implementation: rows}, compact_executions=True) == []
        after = dict(unified_history.load_store(store).histories[("gemma4", implementation)].iter_resolved_snapshots())
        assert typed_equal(before, after)


def test_ordinary_verified_receipt_cannot_silently_drop_mismatched_origin(tmp_path):
    store, implementation, rows, _before, _checkpoint = proof_store(tmp_path)
    job, path, receipt = rows[1]
    value = json.loads(path.read_text())
    receipt["git_commit"] = "a" * 40
    value["provenance"] = receipt
    path.write_text(json.dumps(value))
    before = {p.relative_to(store): p.read_bytes() for p in store.rglob("*.yaml")}
    with pytest.raises(ValueError, match="cannot be retained"):
        import_capture_groups(store, {implementation: [(job, path, receipt)]}, compact_executions=True)
    assert before == {p.relative_to(store): p.read_bytes() for p in store.rglob("*.yaml")}


def test_ordinary_unsupported_capture_does_not_claim_execution_proof(tmp_path):
    store = tmp_path / "store"
    _write_family(store)
    _write_capture(store, "0.0.1", {"text_only": _change()})
    implementation = "vllm_python_batch"
    rows = _receipted_jobs(store, tmp_path / "captures", implementation, "batch", versions=("0.1.0",))
    job, path, receipt = rows[0]
    value = json.loads(path.read_text())
    value["results"] = {"UNIFIED.text_only.gemma4": {"capture_input": _request(), "unavailable": "unsupported"}}
    receipt["results_sha256"] = hashlib.sha256(json.dumps(value["results"], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    value["provenance"] = receipt
    path.write_text(json.dumps(value))
    import_capture_groups(store, {implementation: rows}, compact_executions=True)
    checkpoint = unified_history.load_store(store).histories[("gemma4", implementation)].captures[f"{implementation}-0.1.0"]
    assert "execution_proof" not in checkpoint


def test_original_hash_references_expand_without_rewriting_sparse_bytes():
    digest = "a" * 64
    manifest = {"schema_version": 1, "family": "sample", "implementation": "dynamo_v2", "version": "1.0.0",
                "checkpoint": {"provenance": {}, "document": {}, "document_overrides": {}},
                "cases": ["case"], "defaults": {"verified_recapture": {"provenance": "receipt:" + digest}, "capture_provenance": "receipt:x"},
                "exceptions": {}, "receipts": {digest: ["case"]}}
    before = copy.deepcopy(manifest)
    receipt = {"version": "1.0.0", "missing_vs_null": None}
    restored = execution_manifests.expand(manifest, family="sample", implementation="dynamo_v2", version="1.0.0", receipt_documents={digest: receipt})
    assert restored["metadata_changes"]["case"] == {"verified_recapture": {"provenance": receipt}, "capture_provenance": "receipt:x"}
    assert manifest == before
    assert execution_manifests.source_reference(digest, "executions") == "executions/" + digest + ".yaml"
    with pytest.raises(ValueError, match="missing"):
        execution_manifests.expand(manifest, family="sample", implementation="dynamo_v2", version="1.0.0", receipt_documents={})
