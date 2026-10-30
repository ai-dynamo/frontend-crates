# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import hashlib
import io
import json
import shutil
from pathlib import Path
import sys
import tarfile

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import capture_dynamo_unified
import capture_dynamo_unified_batch
from capture_dynamo_versions import coverage_inventory, freeze_adapters, validate_adapter_binding, validate_frozen_adapters, validate_source_identity
import gen_unified_golden as golden
import unified_history
import generation_workers
from viewer_catalog import _index_unified_families
from capture_stimulus import select_peer_observation
from capture_unified_job import build_job
from import_peer_unified import digest, import_capture, import_captures, import_capture_groups
from conformance.utils.tests.test_unified_history import _change, _request, _write_capture, _write_family

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("source_kind,expected", [("absent", False), ("unsupported", True), ("supported", False)])
def test_muse_guided_capability_is_derived_from_the_exact_upstream_source(tmp_path, source_kind, expected):
    if source_kind != "absent":
        path = tmp_path / "src/tool_calling/muse_glimmer.rs"
        path.parent.mkdir(parents=True)
        body = 'if init.tool_output_mode != UnifiedToolOutputMode::Native { anyhow::bail!("muse_glimmer has no reasoning marker pair"); }'
        path.write_text(body if source_kind == "unsupported" else "fn apply_init() { Ok(()) }")
    assert capture_dynamo_unified.muse_guided_json_unsupported(tmp_path) is expected


@pytest.mark.parametrize("corruption", [None, "empty", "wrong", "bundle", "manifest", "missing_manifest", "bool_schema", "binding"])
def test_dynamo_coverage_requires_the_exact_frozen_producer(v2_capture, tmp_path, corruption):
    store, _job, _capture, payload = v2_capture
    work = tmp_path / "coverage"
    work.mkdir()
    run = {"targets": [], "adapter_manifest_sha256": freeze_adapters(work)}
    manifest = validate_frozen_adapters(work, run)
    relative = "captures/dynamo_v2-0.7.24/gemma4.json"
    expected = validate_adapter_binding(manifest, "dynamo_v2", relative)
    for mode in ("stream", "batch"):
        path = work / "jobs" / mode / "gemma4.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(build_job(store / "families", "gemma4", mode=mode)))
    run["jobs"] = {str(path.relative_to(work)): digest(path) for path in (work / "jobs").rglob("*.json")}
    (work / "frozen-jobs.json").write_text(json.dumps(run["jobs"]))
    target = {"implementation": "dynamo_v2", "version": "0.7.24", "checksum": "c" * 64, "git_commit": None}
    (work / "inventory.yaml").write_text(yaml.safe_dump({"targets": [target]}))
    receipt = payload["provenance"]
    receipt.update(adapter_sha256=expected, job_sha256=run["jobs"]["jobs/stream/gemma4.json"],
                   crate_checksum=target["checksum"], git_commit=None, cargo_lock_sha256="d" * 64)
    receipt["source_sha256"] = hashlib.sha256(json.dumps({"target": target, "lock_sha256": receipt["cargo_lock_sha256"]}, sort_keys=True).encode()).hexdigest()
    if corruption in {"empty", "wrong"}:
        receipt["adapter_sha256"] = "" if corruption == "empty" else "f" * 64
    elif corruption == "bundle":
        bundle = manifest["bundles"][expected]
        path = work / bundle["directory"] / bundle["files"][0]["path"]
        path.write_bytes(path.read_bytes() + b"\n")
        bundle["files"][0]["sha256"] = digest(path)
    elif corruption == "bool_schema":
        manifest["schema_version"] = True
    elif corruption == "binding":
        manifest["implementations"].pop("dynamo_v2")
    if corruption in {"bundle", "bool_schema", "binding"}:
        path = work / "adapter-manifest.json"
        path.write_text(json.dumps(manifest))
        run["adapter_manifest_sha256"] = digest(path)
    elif corruption == "manifest":
        (work / "adapter-manifest.json").write_text("{}")
    elif corruption == "missing_manifest":
        (work / "adapter-manifest.json").unlink()
    path = work / relative
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(payload))
    if corruption:
        with pytest.raises(ValueError, match="adapter"):
            coverage_inventory(work, run)
    else:
        assert coverage_inventory(work, run)["counts"] == {"executed": 1}
        manifest["captures"][relative] = expected
        manifest["implementations"].pop("dynamo_v2")
        assert validate_adapter_binding(manifest, "dynamo_v2", relative) == expected


@pytest.mark.parametrize("corruption", [None, "flags", "source_hash", "lock"])
def test_profiled_coverage_binds_compiler_flags_to_the_verified_source(tmp_path, monkeypatch, corruption):
    producer = tmp_path / "producers/dynamo_v1-0.5.0/dynamo_v1/gemma4"
    producer.mkdir(parents=True)
    lock = producer / "Cargo.lock"
    lock.write_text('[[package]]\nname="dynamo-parsers"\nversion="0.5.0"\nsource="registry+https://github.com/rust-lang/crates.io-index"\nchecksum="' + 'a' * 64 + '"\n')
    source = tmp_path / "verified-crate"
    (producer / "source-metadata.log").write_text(json.dumps({"packages": [{"name": "dynamo-parsers", "version": "0.5.0", "manifest_path": str(source / "Cargo.toml")}]}) + '\nwarning: retained stderr\n')
    monkeypatch.setattr(capture_dynamo_unified_batch, "compatibility_flags", lambda path, has_protocol: ["old_api"] if path == source and not has_protocol else [])
    identity = {"crate_checksum": "a" * 64, "cargo_lock_sha256": digest(lock), "compatibility_flags": ["old_api"]}
    receipt = {**identity, "source_sha256": hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()}
    if corruption == "flags":
        receipt["compatibility_flags"] = []
    elif corruption == "source_hash":
        receipt["source_sha256"] = "b" * 64
    elif corruption == "lock":
        lock.write_text(lock.read_text() + "\n")
    frozen = {"version": "0.5.0", "checksum": "a" * 64}
    if corruption:
        with pytest.raises(ValueError, match="coverage"):
            validate_source_identity(tmp_path, receipt, frozen, "dynamo_v1", "gemma4")
    else:
        validate_source_identity(tmp_path, receipt, frozen, "dynamo_v1", "gemma4")
        receipt.pop("compatibility_flags")
        validate_source_identity(tmp_path, receipt, frozen, "dynamo_v1", "gemma4")


def test_frozen_inventory_has_all_release_artifacts_and_separate_retained_anchors():
    inventory = yaml.safe_load((ROOT / "conformance/dynamo-capture-inventory.yaml").read_text())
    targets = inventory["targets"]
    published = [target for target in targets if target["implementation"] == "dynamo_v2" and target["source_kind"] == "crate"]
    assert len(published) == 67
    assert len({target["version"] for target in published}) == 67
    assert all(len(target["checksum"]) == 64 for target in published)
    anchors = {target["version"]: target for target in targets if target["source_kind"] == "git"}
    assert set(anchors) == {"0.7.9", "0.7.16"}
    assert all(len(target["git_commit"]) == 40 and len(target["source_sha256"]) == 64 for target in anchors.values())
    assert {target["version"] for target in targets if target["implementation"] == "dynamo_v1"} == {"10.0.7", "10.0.6"}


def test_all_engine_inventory_retains_stable_releases_yanks_and_source_blockers():
    inventory = yaml.safe_load((ROOT / "conformance/all-capture-inventory.yaml").read_text())
    targets = inventory["targets"]
    assert {name: sum(target["implementation"] == name for target in targets)
            for name in ("dynamo_v1", "dynamo_v2", "vllm", "sglang")} == {
                "dynamo_v1": 81, "dynamo_v2": 69, "vllm": 99, "sglang": 150}
    assert len({(target["implementation"], target["version"]) for target in targets}) == len(targets)
    for target in targets:
        if target["source_kind"] == "crate":
            assert len(target["checksum"]) == 64
            assert isinstance(target["yanked"], bool)
        elif target["source_kind"] == "pypi":
            assert target["git_commit"] is not None or target["source_blocker"]
            assert target["artifacts"] or target["artifact_blocker"]
            assert all(len(artifact["sha256"]) == 64 and isinstance(artifact["yanked"], bool) for artifact in target["artifacts"])


def test_wrong_published_artifact_is_rejected_before_unpacking(monkeypatch, tmp_path):
    monkeypatch.setattr(capture_dynamo_unified.urllib.request, "urlopen", lambda *args, **kwargs: io.BytesIO(b"wrong crate"))
    with pytest.raises(ValueError, match="checksum differs"):
        capture_dynamo_unified.crate_source({"version": "0.7.24", "checksum": "0" * 64}, tmp_path)
    assert not (tmp_path / "source").exists()


@pytest.mark.parametrize("family", ["deepseek_v3", "deepseek_v3_1", "harmony", "hermes", "inkling"])
def test_reported_families_have_independent_tool_text_reasoning_schema_and_recovery_cases(family):
    cases = golden.build_cases(family)
    def case(name):
        return cases[f"UNIFIED.{name}.{family}"]
    assert case("tool_only")["golden"] == [{"kind": "tool_call", "name": "get_weather", "arguments": {"city": "Paris"}}]
    assert case("text_only")["golden"] == [{"kind": "text", "text": "The answer is 42, no tools needed."}]
    assert case("reason_then_tool")["golden"][0] == {"kind": "reasoning", "text": "Check weather."}
    assert case("truncated_tool_eof")["golden"] == []
    assert [event["arguments"]["city"] for event in case("malformed_json_then_two_valid_calls")["golden"]] == ["Tokyo", "Berlin"]
    assert any(key.startswith("UNIFIED.schema_") for key in cases)
    assert family in golden.scenario_families("tool_only")


@pytest.fixture
def v2_capture(tmp_path):
    store = tmp_path / "store"
    _write_family(store)
    _write_capture(store, "0.7.24", {"text_only": _change("original")}, implementation="dynamo_v2")
    job = tmp_path / "job.json"
    job.write_text(json.dumps(build_job(store / "families", "gemma4", mode="stream")))
    provenance = {"implementation": "dynamo_v2", "version": "0.7.24", "job_sha256": digest(job),
                  "source_sha256": "a" * 64, "adapter_sha256": "b" * 64}
    result = {"capture_input": _request(), "assembled": [{"kind": "text", "text": "fresh"}],
              "chunks": [{"expected": [{"kind": "text", "text": "fresh"}]}]}
    results = {"UNIFIED.text_only.gemma4": result}
    provenance["results_sha256"] = hashlib.sha256(json.dumps(results, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    payload = {"version": "0.7.24", "results": results, "provenance": provenance}
    capture = tmp_path / "capture.json"
    capture.write_text(json.dumps(payload))
    return store, job, capture, payload


def test_v2_recapture_preserves_the_original_record_and_receipt(v2_capture):
    store, job, capture, payload = v2_capture
    previous = copy.deepcopy(unified_history.load_store(store).histories[("gemma4", "dynamo_v2")].resolve("dynamo_v2-0.7.24")["text_only"])
    import_capture(store, job, capture, payload["provenance"], implementation="dynamo_v2")
    history = unified_history.load_store(store).histories[("gemma4", "dynamo_v2")]
    current = history.resolve("dynamo_v2-0.7.24")["text_only"]
    assert current["stimulus"] == previous["stimulus"]
    assert current["observation"] == previous["observation"]
    assert current["document"]["record_metadata"]["verified_recapture"]["result"] == payload["results"]["UNIFIED.text_only.gemma4"]
    record, metadata = unified_history.materialized_record(history.family.cases["text_only"], current)
    record.update(metadata["record_metadata"])
    selected = select_peer_observation(record, _request(), implementation="dynamo_v2", version="0.7.24")
    assert selected["assembled"] == [{"kind": "text", "text": "fresh"}]


@pytest.mark.parametrize("selection", [{"text_only"}, set(), {"missing"}])
def test_selected_recapture_preserves_other_cases_and_checks_the_full_job(v2_capture, selection):
    store, job, capture, payload = v2_capture
    path = store / "families/gemma4/inputs_and_golden.yaml"
    family = unified_history.load_yaml(path)
    second = copy.deepcopy(family["cases"]["text_only"])
    second.update(scenario="second", display_id="UNIFIED.1-2", historical_ids=[])
    family["cases"]["second"] = second
    path.write_text(unified_history.dump_yaml(family))
    _write_capture(store, "0.7.24", {"text_only": _change("original"), "second": _change("untouched")})
    job.write_text(json.dumps(build_job(store / "families", "gemma4", mode="stream")))
    payload["results"]["UNIFIED.second.gemma4"] = copy.deepcopy(payload["results"]["UNIFIED.text_only.gemma4"])
    payload["provenance"].update(job_sha256=digest(job), results_sha256=hashlib.sha256(json.dumps(payload["results"], sort_keys=True, ensure_ascii=False).encode()).hexdigest())
    capture.write_text(json.dumps(payload))
    before = unified_history.load_store(store).histories[("gemma4", "dynamo_v2")].resolve("dynamo_v2-0.7.24")
    if not selection or "missing" in selection:
        with pytest.raises(ValueError, match="selected capture cases"):
            import_capture(store, job, capture, payload["provenance"], implementation="dynamo_v2", selected_cases=selection)
        return
    import_capture(store, job, capture, payload["provenance"], implementation="dynamo_v2", selected_cases=selection)
    after = unified_history.load_store(store).histories[("gemma4", "dynamo_v2")].resolve("dynamo_v2-0.7.24")
    assert after["second"] == before["second"]
    assert after["text_only"]["observation"] == before["text_only"]["observation"]
    assert "verified_recapture" in after["text_only"]["document"]["record_metadata"]
    unchanged = {p: (p.read_bytes(), p.stat().st_ino) for p in store.rglob("*.yaml")}
    assert import_capture(store, job, capture, payload["provenance"], implementation="dynamo_v2", selected_cases=selection) == []
    assert unchanged == {p: (p.read_bytes(), p.stat().st_ino) for p in store.rglob("*.yaml")}
    payload["results"]["UNIFIED.second.gemma4"]["capture_input"]["input"] = "unselected but corrupted"
    payload["provenance"]["results_sha256"] = hashlib.sha256(json.dumps(payload["results"], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    capture.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="request differs from job"):
        import_capture(store, job, capture, payload["provenance"], implementation="dynamo_v2", selected_cases=selection)


@pytest.mark.parametrize("corruption", ["version", "request", "results", "receipt"])
def test_wrong_producer_request_or_stale_receipt_cannot_be_imported(v2_capture, corruption):
    store, job, capture, payload = v2_capture
    provenance = copy.deepcopy(payload["provenance"])
    if corruption == "version":
        payload["version"] = "0.7.23"
    elif corruption == "request":
        payload["results"]["UNIFIED.text_only.gemma4"]["capture_input"]["input"] = "altered"
    elif corruption == "results":
        payload["results"]["UNIFIED.text_only.gemma4"]["assembled"] = []
    else:
        payload["provenance"]["job_sha256"] = "f" * 64
    capture.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        import_capture(store, job, capture, provenance, implementation="dynamo_v2")
    result = unified_history.load_store(store).histories[("gemma4", "dynamo_v2")].resolve("dynamo_v2-0.7.24")["text_only"]
    assert "verified_recapture" not in result["document"].get("record_metadata", {})


def test_catalog_uses_fresh_family_evidence_and_keeps_missing_history_visible(v2_capture):
    store, job, capture, payload = v2_capture
    change = _change()
    change["observation"] = {"unavailable": {"code": "unsupported_family", "detail": "original adapter"}}
    _write_capture(store, "0.7.24", {"text_only": change}, implementation="dynamo_v2")
    import_capture(store, job, capture, payload["provenance"], implementation="dynamo_v2")
    loaded = unified_history.load_store(store)
    candidates = [{"mode": "unified", "implementation": implementation, "version": "0.7.24"}
                  for implementation in ("dynamo_v2", "missing_engine")]
    _index_unified_families(loaded, candidates)
    assert all(candidate["capture_families"] == ["gemma4"] for candidate in candidates)
    _write_capture(store, "0.7.25", {}, implementation="dynamo_v2")
    later = {"mode": "unified", "implementation": "dynamo_v2", "version": "0.7.25"}
    _index_unified_families(unified_history.load_store(store), [later])
    assert later["capture_families"] == []


@pytest.mark.parametrize("implementation", ["dynamo_v1", "dynamo_v1_jail"])
@pytest.mark.parametrize("drop_both", [False, True])
def test_dynamo_batch_results_receipt_is_mandatory(v2_capture, implementation, drop_both):
    store, job, capture, payload = v2_capture
    job.write_text(json.dumps(build_job(store / "families", "gemma4", mode="batch")))
    receipt = {**payload["provenance"], "implementation": implementation, "job_sha256": digest(job)}
    result = payload["results"]["UNIFIED.text_only.gemma4"]
    result.pop("chunks")
    result["batch_final"] = {"reasoning_text": "", "normal_text": "fresh", "calls": []}
    payload["provenance"] = receipt
    receipt["results_sha256"] = "0" * 64
    supplied = {key: value for key, value in receipt.items() if key != "results_sha256"}
    if drop_both:
        receipt.pop("results_sha256")
    capture.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="receipt differs from captured results"):
        import_capture(store, job, capture, supplied, implementation=implementation)


@pytest.mark.parametrize("compact", [False, True])
def test_batched_import_matches_sequential_complete_records_and_preserves_later_history(v2_capture, tmp_path, compact):
    store, job, _capture, payload = v2_capture
    _write_capture(store, "0.7.25", {"text_only": _change("later")}, implementation="dynamo_v2")
    if compact:
        path = store / "families/gemma4/dynamo_v2-0.7.24.yaml"
        document = unified_history.load_yaml(path)
        document["metadata_changes"] = {"text_only": {"audit_note": "retained original metadata"}}
        path.write_text(unified_history.dump_yaml(document))
        unified_history.compact_execution_store(store)
        assert (store / "executions").is_dir()
    sequential = tmp_path / "sequential"
    shutil.copytree(store, sequential)
    entries = []
    for version in ("0.7.23", "0.7.24"):
        document = copy.deepcopy(payload)
        document["version"] = version
        document["provenance"]["version"] = version
        capture = tmp_path / (version + ".json")
        capture.write_text(json.dumps(document))
        entries.append((job, capture, document["provenance"]))
        import_capture(sequential, job, capture, document["provenance"], implementation="dynamo_v2")
    import_captures(store, list(reversed(entries)), implementation="dynamo_v2")
    actual = unified_history.load_store(store).histories[("gemma4", "dynamo_v2")]
    expected = unified_history.load_store(sequential).histories[("gemma4", "dynamo_v2")]
    assert actual.ordered_capture_ids() == expected.ordered_capture_ids()
    assert {identity: actual.resolve(identity) for identity in actual.ordered_capture_ids()} == {
        identity: expected.resolve(identity) for identity in expected.ordered_capture_ids()}
    assert import_captures(store, entries, implementation="dynamo_v2") == []


@pytest.mark.parametrize("implementation", ["dynamo_v1", "dynamo_v1_jail"])
@pytest.mark.parametrize("operation", ["sequential", "batched", "compact"])
def test_legacy_stream_record_keeps_its_contract_inside_batch_checkpoint(v2_capture, implementation, operation):
    store, job, capture, payload = v2_capture
    origin = {"crate_version": "0.7.24", **{key: payload["provenance"][key] for key in ("source_sha256", "adapter_sha256", "job_sha256")}}
    _write_capture(store, "0.7.24", {"text_only": _change("legacy stream")}, implementation=implementation,
                   provenance={"status": "captured", "origin": origin},
                   document_overrides={"text_only": {"capture_provenance": {
                       "captured_with": {implementation: "0.7.23"}, "capture_origin": None, "inherited_from": None}}})
    identity = implementation + "-0.7.24"
    before = unified_history.load_store(store).histories[("gemma4", implementation)].resolve(identity)
    if operation == "compact":
        unified_history._mutate_store(store, lambda loaded: unified_history._compact_history(loaded.histories[("gemma4", implementation)]))
    else:
        job.write_text(json.dumps(build_job(store / "families", "gemma4", mode="batch")))
        result = payload["results"]["UNIFIED.text_only.gemma4"]
        result.pop("chunks")
        result["batch_final"] = {"reasoning_text": "", "normal_text": "fresh", "calls": []}
        payload["provenance"].update(implementation=implementation, job_sha256=digest(job),
            results_sha256=hashlib.sha256(json.dumps(payload["results"], sort_keys=True, ensure_ascii=False).encode()).hexdigest())
        capture.write_text(json.dumps(payload))
        if operation == "sequential":
            import_capture(store, job, capture, payload["provenance"], implementation=implementation)
        else:
            import_captures(store, [(job, capture, payload["provenance"])], implementation=implementation)
    history = unified_history.load_store(store).histories[("gemma4", implementation)]
    current = history.resolve(identity)
    preserved = copy.deepcopy(current)
    preserved["text_only"]["document"].pop("record_metadata", None)
    assert preserved == before
    assert "capture_origin" not in current["text_only"]["document"]
    history.captures[identity]["document_overrides"].pop("text_only")
    with pytest.raises(ValueError, match="batch observation does not match its final fields"):
        history.resolve(identity)


def test_grouped_batch_and_jail_import_is_atomic_and_matches_separate_histories(v2_capture, tmp_path):
    store, job, _capture, payload = v2_capture
    separate = tmp_path / "separate"
    shutil.copytree(store, separate)
    job.write_text(json.dumps(build_job(store / "families", "gemma4", mode="batch")))
    groups = {}
    for implementation in ("dynamo_v1", "dynamo_v1_jail"):
        document = copy.deepcopy(payload)
        result = document["results"]["UNIFIED.text_only.gemma4"]
        result.pop("chunks")
        result["assembled"] = [{"kind": "text", "text": implementation}]
        result["batch_final"] = {"reasoning_text": "", "normal_text": implementation, "calls": []}
        document["provenance"].update(implementation=implementation, job_sha256=digest(job),
            results_sha256=hashlib.sha256(json.dumps(document["results"], sort_keys=True, ensure_ascii=False).encode()).hexdigest())
        path = tmp_path / (implementation + ".json")
        path.write_text(json.dumps(document))
        entries = [(job, path, document["provenance"])]
        groups[implementation] = entries
        import_captures(separate, entries, implementation=implementation)
    before = {str(path.relative_to(store)): path.read_bytes() for path in store.rglob("*.yaml")}
    bad_groups = copy.deepcopy(groups)
    bad_groups["dynamo_v1_jail"][0][2]["results_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="receipt differs from captured results"):
        import_capture_groups(store, bad_groups)
    assert {str(path.relative_to(store)): path.read_bytes() for path in store.rglob("*.yaml")} == before
    import_capture_groups(store, groups)
    actual, expected = unified_history.load_store(store), unified_history.load_store(separate)
    assert set(actual.histories) == set(expected.histories)
    for key, history in actual.histories.items():
        assert dict(history.iter_resolved_snapshots()) == dict(expected.histories[key].iter_resolved_snapshots())
    assert import_capture_groups(store, groups) == []


def test_published_artifact_commit_must_match_frozen_inventory(monkeypatch, tmp_path):
    content = json.dumps({"git": {"sha1": "a" * 40}}).encode()
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        member = tarfile.TarInfo("dynamo-parsers-v2-0.7.24/.cargo_vcs_info.json")
        member.size = len(content)
        archive.addfile(member, io.BytesIO(content))
    artifact = buffer.getvalue()
    monkeypatch.setattr(capture_dynamo_unified.urllib.request, "urlopen", lambda *args, **kwargs: io.BytesIO(artifact))
    with pytest.raises(ValueError, match="commit differs"):
        capture_dynamo_unified.crate_source({"version": "0.7.24", "checksum": hashlib.sha256(artifact).hexdigest(),
                                             "git_commit": "b" * 40}, tmp_path)


@pytest.mark.parametrize("implementation", ["dynamo_v1", "dynamo_v1_jail"])
def test_new_batch_capture_retains_complete_execution_receipt(v2_capture, implementation):
    store, job, capture, payload = v2_capture
    job.write_text(json.dumps(build_job(store / "families", "gemma4", mode="batch")))
    result = payload["results"]["UNIFIED.text_only.gemma4"]
    result.pop("chunks")
    result["batch_final"] = {"reasoning_text": "", "normal_text": "fresh", "calls": []}
    receipt = {**payload["provenance"], "implementation": implementation, "job_sha256": digest(job),
               "crate_checksum": "c" * 64, "cargo_lock_sha256": "d" * 64,
               "results_sha256": hashlib.sha256(json.dumps(payload["results"], sort_keys=True, ensure_ascii=False).encode()).hexdigest()}
    payload["provenance"] = receipt
    capture.write_text(json.dumps(payload))
    import_capture(store, job, capture, receipt, implementation=implementation)
    loaded = unified_history.load_store(store)
    record = loaded.histories[("gemma4", implementation)].resolve(f"{implementation}-0.7.24")["text_only"]
    assert record["document"]["record_metadata"][f"{implementation}_capture_provenance"] == receipt


@pytest.mark.parametrize("implementation,slots", [("dynamo_v2", 1), ("dynamo_v1", 2)])
def test_coverage_keeps_every_unreported_target_as_missing_evidence(v2_capture, tmp_path, implementation, slots):
    store, job, _capture, _payload = v2_capture
    work = tmp_path / "coverage"
    work.mkdir()
    (work / "inventory.yaml").write_text(yaml.safe_dump({"targets": [{"implementation": implementation, "version": "0.7.24"}]}))
    for mode in ("stream", "batch"):
        directory = work / "jobs" / mode
        directory.mkdir(parents=True)
        (directory / "gemma4.json").write_text(json.dumps(build_job(store / "families", "gemma4", mode=mode)))
    frozen = {str(path.relative_to(work)): digest(path) for path in (work / "jobs").rglob("*.json")}
    (work / "frozen-jobs.json").write_text(json.dumps(frozen))
    coverage = coverage_inventory(work, {"targets": [], "jobs": frozen})
    assert coverage["counts"] == {"missing_evidence": slots}
    assert len(coverage["combinations"]) == slots
    assert all(row["scenario"] == "text_only" for row in coverage["combinations"])


@pytest.mark.parametrize("corruption", ["missing", "extra", "bytes", "manifest", "result_manifest"])
def test_coverage_cannot_silently_shrink_or_change_frozen_jobs(v2_capture, tmp_path, corruption):
    store, _job, _capture, _payload = v2_capture
    work = tmp_path / "coverage"
    for mode in ("stream", "batch"):
        path = work / "jobs" / mode / "gemma4.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(build_job(store / "families", "gemma4", mode=mode)))
    frozen = {str(path.relative_to(work)): digest(path) for path in (work / "jobs").rglob("*.json")}
    (work / "frozen-jobs.json").write_text(json.dumps(frozen))
    (work / "inventory.yaml").write_text(yaml.safe_dump({"targets": [{"implementation": "dynamo_v2", "version": "0.7.24"}]}))
    result = {"targets": [], "jobs": frozen.copy()}
    assert coverage_inventory(work, result)["counts"] == {"missing_evidence": 1}
    path = work / "jobs/stream/gemma4.json"
    if corruption == "missing":
        path.unlink()
    elif corruption == "extra":
        shutil.copyfile(path, path.with_name("unexpected.json"))
    elif corruption == "bytes":
        path.write_text(path.read_text() + "\n")
    elif corruption == "manifest":
        (work / "frozen-jobs.json").write_text("{}")
    else:
        result["jobs"].pop("jobs/stream/gemma4.json")
    with pytest.raises(ValueError, match="frozen"):
        coverage_inventory(work, result)


def test_parallel_store_emission_matches_serial_bytes(v2_capture, tmp_path):
    store, job, capture, payload = v2_capture
    serial = tmp_path / "serial"
    shutil.copytree(store, serial)
    with generation_workers.worker_policy("1"):
        import_capture(serial, job, capture, payload["provenance"], implementation="dynamo_v2")
    with generation_workers.worker_policy("2"):
        import_capture(store, job, capture, payload["provenance"], implementation="dynamo_v2")
    assert {str(path.relative_to(store)): path.read_bytes() for path in store.rglob("*.yaml")} == {
        str(path.relative_to(serial)): path.read_bytes() for path in serial.rglob("*.yaml")}


@pytest.mark.parametrize("family", [family for family in golden.FAMILIES
    if golden._MANIFEST[family].get("scenario_profile") == "native"
    and golden._MANIFEST[family].get("reasoning_grammar") != "none"])
def test_structural_tool_recovery_agrees_with_shared_corpus_policy(family):
    cases = golden.build_cases(family)
    call = {"kind": "tool_call", "name": "get_weather", "arguments": {"city": "Paris"}}
    assert cases[f"UNIFIED.tool_in_reason.{family}"]["golden"] == [call]
    assert cases[f"UNIFIED.tool_in_reason_with_text.{family}"]["golden"] == [
        {"kind": "text", "text": "before"}, call, {"kind": "text", "text": "after"}]
