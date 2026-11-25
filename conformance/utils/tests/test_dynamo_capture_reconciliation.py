# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
from contextlib import contextmanager
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import capture_coverage
import capture_dynamo_versions
import capture_sources
import import_peer_unified
import reconcile_dynamo_captures as reconcile
import unified_history
from capture_stimulus import unavailable_result
from capture_unified_job import build_job
from conformance.utils.tests.test_dynamo_release_capture import v2_capture
from conformance.utils.tests.test_import_peer_unified import _receipted_jobs
from conformance.utils.tests.test_unified_history import _change, _write_capture, _write_family


def _release(version, *, yanked=False, checksum="a" * 64):
    return {"num": version, "yanked": yanked, "checksum": checksum, "created_at": "2026-11-16T00:00:00Z"}


def test_discovery_retains_evidence_and_includes_every_future_stable_major(tmp_path, monkeypatch):
    old = [{"implementation": "dynamo_v1", "version": "10.0.0", "source_kind": "crate", "checksum": "a" * 64},
           {"implementation": "dynamo_v1", "version": "11.0.1", "source_kind": "crate", "checksum": "a" * 64, "yanked": True},
           {"implementation": "dynamo_v2", "version": "0.1.0", "source_kind": "git", "git_commit": "b" * 40}]
    monkeypatch.setattr(capture_coverage, "fetch", lambda url: {"versions": (
        [_release("10.0.0"), _release("10.1.0"), _release("11.0.0"), _release("11.0.1", yanked=True),
         _release("11.0.2", yanked=True), _release("12.0.0"), _release("12.1.0-rc.1")]
        if url.endswith("/dynamo-parsers") else [_release("1.0.0"), _release("1.0.1", yanked=True)]
    )})
    monkeypatch.setattr(capture_coverage, "published_v2_identity", lambda target: {"git_commit": "c" * 40, "reasoning_version": "12.0.0"})
    actual = capture_coverage.discover_dynamo_releases(old, tmp_path)
    assert {(target["implementation"], target["version"]) for target in actual} == {
        ("dynamo_v1", "10.0.0"), ("dynamo_v1", "11.0.0"), ("dynamo_v1", "11.0.1"), ("dynamo_v1", "12.0.0"),
        ("dynamo_v2", "0.1.0"), ("dynamo_v2", "1.0.0")}
    for retained in old:
        result = next(item for item in actual if (item["implementation"], item["version"]) == (retained["implementation"], retained["version"]))
        assert {key: value for key, value in result.items() if key != "yanked"} == {key: value for key, value in retained.items() if key != "yanked"}
    assert len(list(tmp_path.glob("*.json"))) == 2
    assert capture_coverage.discover_dynamo_releases(actual, tmp_path) == actual


def test_discovery_rejects_changed_retained_artifact(tmp_path, monkeypatch):
    old = [{"implementation": "dynamo_v1", "version": "11.0.0", "source_kind": "crate", "checksum": "b" * 64}]
    monkeypatch.setattr(capture_coverage, "fetch", lambda url: {"versions": [_release("11.0.0")]})
    with pytest.raises(ValueError, match="conflicts with retained"):
        capture_coverage.discover_dynamo_releases(old, tmp_path)


def test_previously_yanked_release_reenters_automatic_discovery_when_unyanked(tmp_path, monkeypatch):
    old = [{"implementation": "dynamo_v1", "version": "11.0.0", "source_kind": "crate", "checksum": "a" * 64, "yanked": True}]
    monkeypatch.setattr(capture_coverage, "fetch", lambda url: {"versions": [_release("11.0.0")] if url.endswith("/dynamo-parsers") else []})
    assert capture_coverage.discover_dynamo_releases(old, tmp_path) == [{**old[0], "yanked": False}]
    assert old[0]["yanked"] is True


def test_version_filter_freezes_only_the_executed_inventory(tmp_path, monkeypatch):
    store = tmp_path / "store"
    _write_family(store)
    inventory = tmp_path / "inventory.yaml"
    targets = [{"implementation": "dynamo_v2", "version": version} for version in ("0.1.0", "0.1.1")]
    inventory.write_text(yaml.safe_dump({"targets": targets}))
    seen = []
    def capture(target, jobs, output, producer, target_dir):
        seen.append(target["version"])
        return []
    monkeypatch.setattr(capture_dynamo_versions.capture_dynamo_unified, "capture", capture)
    result = capture_dynamo_versions.run_inventory(inventory, tmp_path / "run", 1, ["0.1.1"], store_root=store / "families")
    assert seen == ["0.1.1"]
    assert [target["version"] for target in result["targets"]] == seen
    assert yaml.safe_load((tmp_path / "run/inventory.yaml").read_text())["targets"] == [targets[1]]


@pytest.mark.parametrize("corruption", [None, "checksum", "duplicate", "symlink", "missing_lock", "ambiguous_reasoning", "workspace_reasoning", "invalid_workspace_reasoning"])
def test_new_v2_identity_comes_from_the_verified_artifact(tmp_path, monkeypatch, corruption):
    payload = io.BytesIO()
    lock = '[[package]]\nname="dynamo-parsers"\nversion="12.0.0"\n'
    if corruption == "ambiguous_reasoning":
        lock += '[[package]]\nname="dynamo-parsers"\nversion="11.0.0"\n'
    elif corruption in {"workspace_reasoning", "invalid_workspace_reasoning"}:
        lock = '[[package]]\nname="dynamo-parsers-v2"\nversion="1.0.0"\n'
    sources = {".cargo_vcs_info.json": json.dumps({"git": {"sha1": "c" * 40}}).encode(), "Cargo.lock": lock.encode()}
    if corruption == "missing_lock":
        sources.pop("Cargo.lock")
    with tarfile.open(fileobj=payload, mode="w:gz") as archive:
        for name, value in sources.items():
            member = tarfile.TarInfo("dynamo-parsers-v2-1.0.0/" + name)
            member.size = len(value)
            if corruption == "symlink" and name == "Cargo.lock":
                member.type = tarfile.SYMTYPE
                member.linkname = "elsewhere"
            archive.addfile(member, io.BytesIO(value))
            if corruption == "duplicate" and name == "Cargo.lock":
                archive.addfile(member, io.BytesIO(value))
    artifact = payload.getvalue()
    manifest = b'[workspace.dependencies]\ndynamo-parsers={version="12.0.0",path="parsers/v1"}\n'
    if corruption == "invalid_workspace_reasoning":
        manifest = manifest.replace(b'12.0.0', b'^12.0.0')
    monkeypatch.setattr(capture_coverage.urllib.request, "urlopen", lambda request, **kwargs:
                        io.BytesIO(manifest if "raw.githubusercontent.com" in request.full_url else artifact))
    target = {"version": "1.0.0", "checksum": "0" * 64 if corruption == "checksum" else hashlib.sha256(artifact).hexdigest()}
    if corruption == "workspace_reasoning":
        assert capture_coverage.published_v2_identity(target) == {
            "git_commit": "c" * 40, "reasoning_version": "12.0.0", "reasoning_source_sha256": hashlib.sha256(manifest).hexdigest()}
    elif corruption:
        with pytest.raises(ValueError):
            capture_coverage.published_v2_identity(target)
    else:
        assert capture_coverage.published_v2_identity(target) == {"git_commit": "c" * 40, "reasoning_version": "12.0.0"}


@pytest.mark.parametrize("change", [None, "adapter", "request", "checksum", "source", "mode", "case", "receipt", "reasoning", "dependency_pins"])
def test_skip_requires_each_frozen_request_mode_and_producer_receipt(v2_capture, change):
    store, job, capture, payload = v2_capture
    payload["provenance"].update(crate_checksum="c" * 64, cargo_lock_sha256="d" * 64, git_commit="e" * 40)
    target = {"version": "0.7.24", "source_kind": "crate", "checksum": "c" * 64, "git_commit": "e" * 40}
    payload["provenance"]["source_sha256"] = hashlib.sha256(json.dumps({"target": target, "lock_sha256": "d" * 64}, sort_keys=True).encode()).hexdigest()
    capture.write_text(json.dumps(payload))
    import_peer_unified.import_capture(store, job, capture, payload["provenance"], implementation="dynamo_v2")
    history = unified_history.load_store(store).histories["gemma4", "dynamo_v2"]
    adapter = payload["provenance"]["adapter_sha256"]
    data = job.read_bytes()
    implementation = "dynamo_v2"
    if change == "adapter":
        adapter = "f" * 64
    elif change == "request":
        changed = json.loads(data)
        changed["cases"][0]["request"]["input"] = "different"
        data = json.dumps(changed).encode()
    elif change == "checksum":
        target["checksum"] = "f" * 64
    elif change == "source":
        target["git_commit"] = "f" * 40
    elif change == "mode":
        implementation = "dynamo_v1_jail"
    elif change == "reasoning":
        target["reasoning_version"] = "11.0.1"
    elif change == "dependency_pins":
        target["dependency_pins"] = {"serde": "1.0.219"}
    elif change == "case":
        changed = json.loads(data)
        changed["cases"].append({**changed["cases"][0], "case_id": "new"})
        data = json.dumps(changed).encode()
    elif change == "receipt":
        identity = "dynamo_v2-0.7.24"
        history.captures[identity].pop("execution_proof", None)
        history.captures[identity]["metadata_changes"]["text_only"].clear()
    assert reconcile.qualified_capture(history, target, implementation, data, adapter) is (change is None)


def test_pending_target_requires_both_direct_and_jail_for_every_family(monkeypatch):
    calls = []
    missing = ("gemma4", "dynamo_v1_jail")
    class Store:
        histories = {("gemma4", mode): mode for mode in ("dynamo_v1", "dynamo_v1_jail")}
    def qualified(history, target, implementation, data, adapter):
        calls.append((json.loads(data)["family"], implementation))
        return calls[-1] != missing
    monkeypatch.setattr(reconcile, "qualified_capture", qualified)
    target = {"implementation": "dynamo_v1", "version": "12.0.0"}
    jobs = {("batch", "gemma4"): b'{"family":"gemma4"}'}
    assert reconcile.pending_targets(Store(), [target], jobs, {mode: mode for mode in Store.histories.values()}) == [target]
    assert calls == [("gemma4", "dynamo_v1"), ("gemma4", "dynamo_v1_jail")]


def test_atomic_publication_preserves_latest_and_unrelated_artifacts(tmp_path):
    root = tmp_path / "checkout"
    live = root / capture_sources.STORE
    _write_family(live)
    _write_capture(live, "0.1.0", {"text_only": _change("before")})
    (live / "families/gemma4/dynamo_v2.yaml").write_bytes(b"latest bytes remain independent\n")
    (live / "operator.capture").write_bytes(b"operator-owned")
    inventory = root / "conformance/dynamo-capture-inventory.yaml"
    inventory.write_bytes(b"before inventory\n")
    baseline = unified_history.store_inventory(live)
    staged = tmp_path / "candidate"
    _write_family(staged)
    _write_capture(staged, "0.1.0", {"text_only": _change("after")})
    reconcile.publish_history_store(root, staged, baseline, inventory_update=(inventory, b"before inventory\n", b"after inventory\n"))
    assert (live / "families/gemma4/dynamo_v2-0.1.0.yaml").read_bytes() == (staged / "families/gemma4/dynamo_v2-0.1.0.yaml").read_bytes()
    assert (live / "families/gemma4/dynamo_v2.yaml").read_bytes() == b"latest bytes remain independent\n"
    assert (live / "operator.capture").read_bytes() == b"operator-owned"
    assert inventory.read_bytes() == b"after inventory\n"
    with pytest.raises(ValueError, match="store changed"):
        reconcile.publish_history_store(root, staged, baseline)


def test_inventory_conflict_cannot_publish_any_capture_change(tmp_path):
    root = tmp_path / "checkout"
    live = root / capture_sources.STORE
    _write_family(live)
    baseline = unified_history.store_inventory(live)
    inventory = root / "conformance/dynamo-capture-inventory.yaml"
    inventory.write_bytes(b"changed inventory")
    staged = tmp_path / "candidate"
    shutil.copytree(live, staged)
    _write_capture(staged, "0.1.0", {"text_only": _change("after")})
    with pytest.raises(ValueError, match="inventory changed"):
        reconcile.publish_history_store(root, staged, baseline, inventory_update=(inventory, b"before inventory", b"after inventory"))
    assert unified_history.store_inventory(live) == baseline
    assert inventory.read_bytes() == b"changed inventory"


@pytest.mark.parametrize("failure", [None, "source_build_blocker", "missing_evidence"])
def test_reconciliation_imports_only_complete_runs_and_skips_the_identical_rerun(v2_capture, tmp_path, monkeypatch, failure):
    source, _job, _capture, payload = v2_capture
    root = tmp_path / "checkout"
    store = root / capture_sources.STORE
    shutil.copytree(source, store)
    _write_capture(store, "11.0.13", {"text_only": _change("eligible")}, implementation="dynamo_v1")
    _write_capture(store, "11.0.14", {"text_only": _change("yanked")}, implementation="dynamo_v1")
    retained_v1 = {path.name: path.read_bytes() for path in (store / "families/gemma4").glob("dynamo_v1-*.yaml")}
    inventory = root / "conformance/dynamo-capture-inventory.yaml"
    target = {"implementation": "dynamo_v2", "version": "0.7.24", "source_kind": "crate", "checksum": "c" * 64, "git_commit": "e" * 40}
    inventory.write_text(yaml.safe_dump({"targets": [target]}))
    original_inventory = inventory.read_bytes()
    original_store = unified_history.store_inventory(store)
    calls = []

    @contextmanager
    def history_view(checkout, history):
        with tempfile.TemporaryDirectory(dir=tmp_path) as directory:
            staged = Path(directory)
            shutil.copytree(checkout / capture_sources.STORE, staged / capture_sources.STORE)
            yield staged

    def capture_inventory(inventory, work, workers, *, store_root):
        calls.append(work)
        jobs = capture_dynamo_versions.frozen_family_jobs(store_root)
        for (mode, family), data in jobs.items():
            path = work / "jobs" / mode / (family + ".json")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        capture_dynamo_versions.freeze_adapters(work)
        adapters = json.loads((work / "adapter-manifest.json").read_text())["implementations"]
        document = copy.deepcopy(payload)
        document["provenance"].update(crate_checksum=target["checksum"], cargo_lock_sha256="d" * 64,
                                      git_commit=target["git_commit"], adapter_sha256=adapters["dynamo_v2"],
                                      job_sha256=hashlib.sha256(jobs["stream", "gemma4"]).hexdigest())
        document["provenance"]["source_sha256"] = hashlib.sha256(json.dumps({"target": target, "lock_sha256": "d" * 64}, sort_keys=True).encode()).hexdigest()
        capture = work / "captures/dynamo_v2-0.7.24/gemma4.json"
        capture.parent.mkdir(parents=True)
        capture.write_text(json.dumps(document))
        return {"targets": [target]}

    monkeypatch.setattr(capture_sources, "history_view", history_view)
    monkeypatch.setattr(capture_dynamo_versions, "run_inventory", capture_inventory)
    monkeypatch.setattr(capture_dynamo_versions, "coverage_inventory", lambda *args: {
        "combinations": [{"implementation": "dynamo_v2", "version": "0.7.24", "family": "gemma4", "status": failure or "executed"}],
    })
    if failure:
        with pytest.raises(RuntimeError, match="no evidence published"):
            reconcile.reconcile(root, tmp_path / "run-1", discover=False)
        assert unified_history.store_inventory(store) == original_store
        assert inventory.read_bytes() == original_inventory
    else:
        assert reconcile.reconcile(root, tmp_path / "run-1", discover=False, versions=[target["version"]]) == {"targets": 1, "pending": 1, "qualified": 0}
        assert retained_v1 == {path.name: path.read_bytes() for path in (store / "families/gemma4").glob("dynamo_v1-*.yaml")}
        after = unified_history.store_inventory(store)
        assert reconcile.reconcile(root, tmp_path / "run-2", discover=False) == {"targets": 1, "pending": 0, "qualified": 1}
        assert unified_history.store_inventory(store) == after
    assert len(calls) == 1


def test_capture_workflow_is_separate_from_release_and_reuses_signed_reviewed_prs():
    root = Path(__file__).resolve().parents[3]
    release = yaml.load((root / ".github/workflows/release.yml").read_text(), Loader=yaml.BaseLoader)
    job = release["jobs"]["reconcile-captures"]
    assert job["needs"] == "release"
    assert job["uses"] == "./.github/workflows/reconcile-captures.yml"
    workflow = yaml.load((root / ".github/workflows/reconcile-captures.yml").read_text(), Loader=yaml.BaseLoader)
    assert set(workflow["on"]) == {"workflow_dispatch", "workflow_call"}
    assert workflow["concurrency"]["cancel-in-progress"] == "false"
    steps = workflow["jobs"]["reconcile"]["steps"]
    publication = next(step for step in steps if step.get("uses", "").startswith("peter-evans/create-pull-request@"))
    assert publication["with"]["branch"] == "automation/dynamo-captures"
    assert publication["with"]["sign-commits"] == "true"
    assert publication["with"]["signoff"] == "true"
    assert publication["with"]["token"] == "${{ steps.capture-pr-token.outputs.token }}"
    capture_index = next(index for index, step in enumerate(steps) if step["name"] == "Reconcile published releases")
    fresh_index = next(index for index, step in enumerate(steps) if step.get("id") == "capture-pr-token")
    publication_index = steps.index(publication)
    assert capture_index < fresh_index < publication_index
    assert "--workers 4" in next(step["run"] for step in steps if step["name"] == "Reconcile published releases")


@pytest.mark.parametrize("implementation", ["dynamo_v1", "dynamo_v1_jail", "dynamo_v2"])
@pytest.mark.parametrize("outcome", ["value", "error", "unsupported"])
def test_new_release_reuses_unchanged_payloads_and_retains_its_own_attempt(tmp_path, implementation, outcome):
    store = tmp_path / "store"
    _write_family(store)
    authored = store / "families/gemma4/inputs_and_golden.yaml"
    document = unified_history.load_yaml(authored)
    document["cases"]["second"] = {**copy.deepcopy(document["cases"]["text_only"]), "scenario": "reason_only", "display_id": "UNIFIED.10-reason_only"}
    authored.write_text(unified_history.dump_yaml(document))
    with pytest.raises(ValueError, match="no history files"):
        unified_history.load_store(store)
    mode = "stream" if implementation == "dynamo_v2" else "batch"
    job = tmp_path / "job.json"
    job.write_text(json.dumps(build_job(store / "families", "gemma4", mode=mode)))
    cases = json.loads(job.read_bytes())["cases"]

    def capture(version, *, changed=False):
        results = {}
        for case in cases:
            text = "changed" if changed and case["scenario"] == "reason_only" else "same"
            if outcome == "unsupported":
                result = unavailable_result("unsupported_request_mode", text, capture=case["request"], observation=True)
            elif outcome == "error":
                result = {"capture_input": case["request"], "error": text}
            else:
                events = [{"kind": "text", "text": text}]
                result = {"capture_input": case["request"], "assembled": events}
                if mode == "stream":
                    result["chunks"] = [{"expected": events}]
                else:
                    result["batch_final"] = {"reasoning_text": "", "normal_text": text, "calls": []}
            results[case["id"]] = result
        provenance = {"implementation": implementation, "version": version,
                      "source_sha256": "a" * 64, "adapter_sha256": "b" * 64,
                      "job_sha256": hashlib.sha256(job.read_bytes()).hexdigest(),
                      "results_sha256": hashlib.sha256(json.dumps(results, sort_keys=True, ensure_ascii=False).encode()).hexdigest()}
        path = tmp_path / f"capture-{version}.json"
        path.write_text(json.dumps({"version": version, "provenance": provenance, "results": results}))
        return job, path, provenance

    first, second = capture("11.0.0"), capture("11.0.1")
    import_peer_unified.import_captures(store, [first, second], implementation=implementation)
    history = unified_history.load_store(store).histories["gemma4", implementation]
    checkpoint = history.captures[implementation + "-11.0.1"]
    assert checkpoint["changes"] == {}
    if outcome == "unsupported":
        assert "execution_proof" not in checkpoint
        assert all(import_peer_unified._measurement_metadata(metadata) == {} for metadata in checkpoint["metadata_changes"].values())
    else:
        assert checkpoint["metadata_changes"] == {}
        assert checkpoint["execution_proof"]["receipt"] == second[2]
    before = copy.deepcopy(dict(history.iter_resolved_snapshots(include_members=True)))
    members_before = copy.deepcopy(history.snapshot_members)
    third = capture("11.0.2", changed=True)
    import_peer_unified.import_captures(store, [third], implementation=implementation)
    history = unified_history.load_store(store).histories["gemma4", implementation]
    checkpoint = history.captures[implementation + "-11.0.2"]
    assert set(checkpoint["changes"]) == {"second"}
    if outcome == "unsupported":
        assert "execution_proof" not in checkpoint
    else:
        assert checkpoint["execution_proof"]["receipt"] == third[2]
    for identity, records in history.iter_resolved_snapshots(include_members=True):
        if identity in before:
            assert records == before[identity]
            assert history.snapshot_members[identity] == members_before[identity]
    assert import_peer_unified.import_captures(store, [first, second, third], implementation=implementation) == []


def test_inserted_release_preserves_later_original_source_member(tmp_path):
    store = tmp_path / "store"
    _write_family(store)
    _write_capture(store, "11.0.0", {"text_only": _change("same")}, implementation="dynamo_v1")
    _write_capture(store, "11.0.2", {}, implementation="dynamo_v1")
    history = unified_history.load_store(store).histories["gemma4", "dynamo_v1"]
    before = dict(history.iter_resolved_snapshots(include_members=True))
    members = copy.deepcopy(history.snapshot_members)
    job = tmp_path / "job.json"
    document = build_job(store / "families", "gemma4", mode="batch")
    job.write_text(json.dumps(document))
    result = {"capture_input": document["cases"][0]["request"], "assembled": [{"kind": "text", "text": "inserted"}],
              "batch_final": {"reasoning_text": "", "normal_text": "inserted", "calls": []}}
    results = {document["cases"][0]["id"]: result}
    provenance = {"implementation": "dynamo_v1", "version": "11.0.1", "source_sha256": "a" * 64, "adapter_sha256": "b" * 64,
                  "job_sha256": hashlib.sha256(job.read_bytes()).hexdigest(),
                  "results_sha256": hashlib.sha256(json.dumps(results, sort_keys=True, ensure_ascii=False).encode()).hexdigest()}
    capture = tmp_path / "capture.json"
    capture.write_text(json.dumps({"version": "11.0.1", "provenance": provenance, "results": results}))
    import_peer_unified.import_captures(store, [(job, capture, provenance)], implementation="dynamo_v1")
    history = unified_history.load_store(store).histories["gemma4", "dynamo_v1"]
    after = dict(history.iter_resolved_snapshots(include_members=True))
    assert after["dynamo_v1-11.0.2"] == before["dynamo_v1-11.0.2"]
    assert history.snapshot_members["dynamo_v1-11.0.2"] == members["dynamo_v1-11.0.2"]


@pytest.mark.parametrize("state", ["supported", "missing_api", "wrong_source", "foreign_error", "missing_result", "false_success"])
def test_recorded_context_probe_binds_capability_flags_to_owned_compiler_result(tmp_path, state):
    batch = capture_dynamo_versions.capture_dynamo_unified_batch
    example = tmp_path / "examples" / (batch.CONTEXT_PROBE_NAME + ".rs")
    example.parent.mkdir()
    example.write_text("wrong source" if state == "wrong_source" else batch.CONTEXT_PROBE)
    target = {"name": batch.CONTEXT_PROBE_NAME, "src_path": str(example)}
    rows = []
    supported = state not in {"missing_api", "foreign_error"}
    if not supported:
        rows.append({"reason": "compiler-message", "target": {**target, "src_path": str(tmp_path / "foreign.rs")} if state == "foreign_error" else target,
                     "message": {"level": "error", "code": {"code": "E0432"}, "spans": [{}]}})
    elif state != "false_success":
        rows.append({"reason": "compiler-artifact", "target": target})
    if state != "missing_result":
        rows.append({"reason": "build-finished", "success": supported})
    (tmp_path / "reasoning-context-probe.log").write_text("\n".join(json.dumps(row) for row in rows) + "\nretained cargo stderr\n")
    if state in {"supported", "missing_api"}:
        assert capture_dynamo_versions.recorded_context_flags(tmp_path) == (["conformance_reasoning_context"] if supported else [])
    else:
        with pytest.raises(ValueError, match="coverage context capability"):
            capture_dynamo_versions.recorded_context_flags(tmp_path)


@pytest.mark.parametrize("implementation", ["dynamo_v1", "dynamo_v1_jail", "dynamo_v2"])
@pytest.mark.parametrize("initial_proof", [False, True])
@pytest.mark.parametrize("outcome", ["value", "mixed", "unsupported"])
def test_expanded_recapture_retains_original_origin_proof_and_new_case_receipt(tmp_path, implementation, initial_proof, outcome):
    store = tmp_path / "store"
    _write_family(store)
    mode = "stream" if implementation == "dynamo_v2" else "batch"

    def captures(directory, versions, *, fresh=False):
        rows = _receipted_jobs(store, directory, implementation, mode, versions=versions)
        for job, capture, receipt in rows:
            document = json.loads(capture.read_bytes())
            template = next(iter(document["results"].values()))
            results = {}
            for case in json.loads(job.read_bytes())["cases"]:
                unsupported = fresh and (outcome == "unsupported" or outcome == "mixed" and case["scenario"] == "reason_only")
                if unsupported:
                    result = unavailable_result("unsupported_request_mode", "unavailable", capture=case["request"], observation=True)
                else:
                    result = {**copy.deepcopy(template), "capture_input": case["request"], "assembled": [{"kind": "text", "text": "same"}]}
                    if mode == "batch":
                        result["batch_final"]["normal_text"] = "same"
                    else:
                        result["chunks"] = [[{"kind": "text", "text": "same"}]]
                results[case["id"]] = result
            document["results"] = results
            if fresh:
                receipt["adapter_sha256"] = "c" * 64
            receipt["results_sha256"] = hashlib.sha256(json.dumps(results, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            document["provenance"] = receipt
            capture.write_text(json.dumps(document))
        return rows

    original_rows = captures(tmp_path / "old", ("11.0.0", "11.0.1") if initial_proof else ("11.0.1",))
    import_peer_unified.import_captures(store, original_rows, implementation=implementation)
    history = unified_history.load_store(store).histories["gemma4", implementation]
    original_checkpoint = copy.deepcopy(history.captures[implementation + "-11.0.1"])
    assert ("execution_proof" in original_checkpoint) is initial_proof
    authored = store / "families/gemma4/inputs_and_golden.yaml"
    document = unified_history.load_yaml(authored)
    document["cases"]["second"] = {**copy.deepcopy(document["cases"]["text_only"]), "scenario": "reason_only", "display_id": "UNIFIED.10-reason_only"}
    authored.write_text(unified_history.dump_yaml(document))
    rows = captures(tmp_path / "new", ("11.0.0", "11.0.1"), fresh=True)
    import_peer_unified.import_captures(store, rows, implementation=implementation)
    history = unified_history.load_store(store).histories["gemma4", implementation]
    checkpoint = history.captures[implementation + "-11.0.1"]
    assert checkpoint["provenance"] == original_checkpoint["provenance"]
    if initial_proof:
        preserved = [checkpoint.get("execution_proof"), *[evidence.get("execution_proof") for evidence in checkpoint.get("historical_evidence", [])]]
        assert original_checkpoint["execution_proof"] in preserved
    record = history.resolve(implementation + "-11.0.1")["second"]
    assert any(receipt["adapter_sha256"] == "c" * 64 and receipt["version"] == "11.0.1"
               for owner, key in import_peer_unified.execution_manifests._receipt_slots(record["document"]["record_metadata"])
               for receipt in [owner[key]])
    assert import_peer_unified.import_captures(store, rows, implementation=implementation) == []


@pytest.mark.parametrize("corruption", [None, "unknown", "missing", "digest", "root", "receipt_membership", "record_missing", "receipt_hash"])
def test_archived_proof_qualification_requires_original_receipt_and_retained_request_binding(v2_capture, corruption):
    store, job, _capture, payload = v2_capture
    history = unified_history.load_store(store).histories["gemma4", "dynamo_v2"]
    receipt = copy.deepcopy(payload["provenance"])
    target = {"implementation": "dynamo_v2", "version": "0.7.24", "source_kind": "crate", "checksum": "c" * 64, "git_commit": "d" * 40}
    receipt.update(crate_checksum=target["checksum"], git_commit=target["git_commit"], cargo_lock_sha256="e" * 64)
    receipt["source_sha256"] = hashlib.sha256(json.dumps({"target": target, "lock_sha256": receipt["cargo_lock_sha256"]}, sort_keys=True).encode()).hexdigest()
    checkpoint = {"provenance": {"status": "captured", "origin": {"crate_version": "0.7.24", **{
        key: receipt[key] for key in ("source_sha256", "adapter_sha256", "job_sha256")}}},
        "document": {"mode": "unified"}, "document_overrides": {}, "metadata_changes": {},
        "execution_proof": {"receipt": receipt, "requests": {"text_only": unified_history._request_digest(history.family.cases["text_only"]["request"])}}}
    receipts = {}
    manifest = import_peer_unified.execution_manifests.compact(checkpoint, family="gemma4", implementation="dynamo_v2", version="0.7.24", receipt_documents=receipts, receipt_names={})
    evidence = import_peer_unified.execution_manifests.historical_evidence(manifest, family="gemma4", implementation="dynamo_v2", version="0.7.24", receipt_documents=receipts)
    proof = evidence["execution_proof"]
    if corruption == "unknown":
        proof["requests"]["unknown"] = proof["requests"]["text_only"]
        evidence["receipt_membership"][0]["members"].append("unknown")
    elif corruption == "missing":
        proof["requests"].clear()
    elif corruption == "digest":
        proof["requests"]["text_only"] = "0" * 64
    elif corruption == "root":
        evidence["checkpoint"]["provenance"]["origin"]["source_sha256"] = "0" * 64
    elif corruption == "receipt_membership":
        evidence["receipt_membership"][0]["members"].clear()
    elif corruption == "receipt_hash":
        proof["receipt"]["results_sha256"] = "not-a-hash"
    identity = "dynamo_v2-0.7.24"
    history.captures[identity]["historical_evidence"] = [evidence]
    if corruption == "record_missing":
        history.captures[identity]["changes"].clear()
    if corruption:
        with pytest.raises(ValueError):
            reconcile.qualified_capture(history, target, "dynamo_v2", job.read_bytes(), receipt["adapter_sha256"])
    else:
        assert reconcile.qualified_capture(history, target, "dynamo_v2", job.read_bytes(), receipt["adapter_sha256"])
