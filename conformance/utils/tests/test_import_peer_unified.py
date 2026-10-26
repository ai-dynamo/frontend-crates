# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Backfills preserve historical outputs and each peer's final-message contract."""

from collections import Counter
import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import unified_history
import import_peer_unified as importer
from capture_stimulus import select_peer_observation
from capture_unified_job import build_job
from import_peer_unified import digest, import_capture, import_capture_groups, main
from capture_stimulus import assemble_stream
from conformance.utils.tests.test_unified_history import _add_backfill_case, _change, _request, _write_capture, _write_family


def _receipted_jobs(store, root, implementation, mode, versions=("0.25.1", "0.26.0"), family="gemma4"):
    root.mkdir(parents=True)
    rows = []
    for version in versions:
        job = root / f"job-{version}.json"
        job.write_text(json.dumps(build_job(store / "families", family, mode=mode)))
        result = {"capture_input": _request(), "assembled": [{"kind": "text", "text": version}]}
        if mode == "batch":
            result["batch_final"] = {"reasoning_text": "", "normal_text": version, "calls": []}
        else:
            result["chunks"] = [[{"kind": "text", "text": version}]]
        results = {f"UNIFIED.text_only.{family}": result}
        provenance = {"implementation": implementation, "version": version, "source_sha256": "a" * 64,
            "adapter_sha256": "b" * 64, "job_sha256": digest(job),
            "results_sha256": hashlib.sha256(json.dumps(results, sort_keys=True, ensure_ascii=False).encode()).hexdigest()}
        field = {"sglang_python": "sglang_version", "vllm_python": "vllm_version", "vllm_rust": "vllm_rust_version"}.get(implementation, "version")
        capture = root / f"capture-{version}.json"
        capture.write_text(json.dumps({field: version, "results": results, "provenance": provenance}))
        rows.append((job, capture, provenance))
    return rows


def test_grouped_expected_jobs_are_owned_by_each_history_and_transaction(tmp_path, monkeypatch):
    store = tmp_path / "store"
    _write_family(store)
    _write_capture(store, "0.24.0", {"text_only": _change("baseline")})
    family = unified_history.load_yaml(store / "families/gemma4/inputs_and_golden.yaml")
    family["input_document"]["family"] = "qwen3"
    family["golden_document"]["family"] = "qwen3"
    path = store / "families/qwen3/inputs_and_golden.yaml"
    path.parent.mkdir()
    path.write_text(unified_history.dump_yaml(family))
    families = ("gemma4", "qwen3")
    modes = {"dynamo_v1": "batch", "dynamo_v1_jail": "batch", "vllm_python_batch": "batch", "sglang_python_batch": "batch",
             "dynamo_v2": "stream", "vllm_python": "unified", "sglang_python": "unified", "vllm_rust": "unified"}
    groups = {implementation: [row for family in families
                              for row in _receipted_jobs(store, tmp_path / implementation / family, implementation, mode, family=family)]
              for implementation, mode in modes.items()}
    calls = []
    original = importer.build_job
    def counted(root, family, *, mode):
        calls.append((family, mode))
        return original(root, family, mode=mode)
    monkeypatch.setattr(importer, "build_job", counted)
    for iteration in (1, 2):
        import_capture_groups(store, groups)
        assert Counter(calls) == Counter({(family, mode): count * iteration for family in families
                                          for mode, count in (("batch", 4), ("stream", 1), ("unified", 3))})
        loaded = unified_history.load_store(store)
        for implementation in modes:
            for family in families:
                history = loaded.histories[(family, implementation)]
                for version in ("0.25.1", "0.26.0"):
                    current = history.resolve(f"{implementation}-{version}")["text_only"]
                    record, metadata = unified_history.materialized_record(history.family.cases["text_only"], current)
                    record.update(metadata.get("record_metadata", {}))
                    selected = select_peer_observation(record, _request(), implementation=implementation, version=version)
                    assert selected["assembled"] == [{"kind": "text", "text": version}]


@pytest.mark.parametrize("altered_index", [0, 1])
def test_grouped_job_cache_rejects_resealed_altered_requests(tmp_path, monkeypatch, altered_index):
    store = tmp_path / "store"
    _write_family(store)
    _write_capture(store, "0.24.0", {"text_only": _change("baseline")})
    rows = _receipted_jobs(store, tmp_path / "captures", "dynamo_v2", "stream")
    job, capture, provenance = rows[altered_index]
    request = copy.deepcopy(_request())
    request["input"] = "changed"
    request["chunks"] = [{"delta_text": "changed"}]
    job_document = json.loads(job.read_text())
    job_document["cases"][0].update(input="changed", chunks=["changed"], request=request)
    job.write_text(json.dumps(job_document))
    captured = json.loads(capture.read_text())
    captured["results"]["UNIFIED.text_only.gemma4"]["capture_input"] = request
    provenance.update(job_sha256=digest(job), results_sha256=hashlib.sha256(json.dumps(captured["results"], sort_keys=True, ensure_ascii=False).encode()).hexdigest())
    captured["provenance"] = provenance
    capture.write_text(json.dumps(captured))
    before = {str(path.relative_to(store)): path.read_bytes() for path in store.rglob("*") if path.is_file()}
    calls = []
    original = importer.build_job
    def counted(root, family, *, mode):
        calls.append((family, mode))
        return original(root, family, mode=mode)
    monkeypatch.setattr(importer, "build_job", counted)
    with pytest.raises(ValueError, match="capture job differs from the final candidate request corpus"):
        import_capture_groups(store, {"dynamo_v2": rows})
    assert calls == [("gemma4", "stream")]
    assert {str(path.relative_to(store)): path.read_bytes() for path in store.rglob("*") if path.is_file()} == before


def test_single_import_keeps_fresh_exact_job_validation(tmp_path, monkeypatch):
    store = tmp_path / "store"
    _write_family(store)
    _write_capture(store, "0.24.0", {"text_only": _change("baseline")})
    rows = _receipted_jobs(store, tmp_path / "captures", "dynamo_v2", "stream")
    calls = []
    original = importer.build_job
    def counted(root, family, *, mode):
        calls.append((family, mode))
        return original(root, family, mode=mode)
    monkeypatch.setattr(importer, "build_job", counted)
    for job, capture, provenance in rows:
        import_capture(store, job, capture, provenance, implementation="dynamo_v2")
    assert calls == [("gemma4", "stream"), ("gemma4", "stream")]


def test_peer_stream_assembly_coalesces_text_and_joins_tool_arguments_in_order():
    chunks = [
        [{"kind": "reasoning", "text": "think"}],
        [
            {"kind": "reasoning", "text": " more"},
            {"kind": "tool_call", "name": "lookup", "arguments": '{"q":', "complete": False},
        ],
        [
            {"kind": "tool_call", "arguments": '"x"}', "complete": True},
            {"kind": "text", "text": "done"},
        ],
    ]
    assert assemble_stream(chunks) == [
        {"kind": "reasoning", "text": "think more"},
        {"kind": "tool_call", "name": "lookup", "arguments": {"q": "x"}},
        {"kind": "text", "text": "done"},
    ]


@pytest.mark.parametrize("implementation", ["vllm_python", "vllm_rust", "sglang_python"])
@pytest.mark.parametrize("later_changes", [False, True])
@pytest.mark.parametrize("override_provenance", [False, True])
def test_older_import_preserves_later_snapshot_defaults_and_overrides(
    tmp_path, implementation, later_changes, override_provenance,
):
    # Provenance reset/inheritance is storage behavior, outside the parser corpus schema.
    store = tmp_path / "store"
    _write_family(store)
    _write_capture(store, "0.26.0", {"text_only": _change("one")}, implementation=implementation)
    overrides = {"text_only": {"capture_provenance": {
        "captured_with": {implementation: "0.26.0"},
        "capture_origin": None, "inherited_from": None,
    }}} if override_provenance else {}
    _write_capture(
        store, "0.27.0", {"text_only": _change("two")} if later_changes else {},
        implementation=implementation, document_overrides=overrides,
    )
    history = unified_history.load_store(store).histories[("gemma4", implementation)]
    snapshots = {key: history.resolve(key) for key in history.ordered_capture_ids()}
    job = tmp_path / "job.json"
    job.write_text(json.dumps(build_job(store / "families", "gemma4")))
    capture = tmp_path / "capture.yaml"
    version_key = "sglang_version" if implementation == "sglang_python" else (
        "vllm_version" if implementation == "vllm_python" else "vllm_rust_version"
    )
    capture.write_text(unified_history.dump_yaml({version_key: "0.25.1", "results": {
        "UNIFIED.text_only.gemma4": {
            "capture_input": _request(), "assembled": [{"kind": "text", "text": "zero"}],
            "chunks": [[{"kind": "text", "text": "zero"}]],
        },
    }}))
    provenance = {"implementation": implementation, "version": "0.25.1",
                  "job_sha256": digest(job), "source_sha256": "a" * 64,
                  "adapter_sha256": "b" * 64}
    import_capture(store, job, capture, provenance, implementation=implementation)
    history = unified_history.load_store(store).histories[("gemma4", implementation)]
    assert {key: history.resolve(key) for key in snapshots} == snapshots
    assert history.resolve(f"{implementation}-0.25.1")["text_only"]["document"]["captured_with"] == {
        implementation: "0.25.1",
    }
    assert import_capture(
        store, job, capture, provenance, implementation=implementation, backfill_only=True,
    ) == []


@pytest.fixture(params=["vllm_python", "vllm_rust"])
def peer_capture(request, tmp_path):
    implementation = request.param
    store = tmp_path / "store"
    _write_family(store)
    _write_capture(store, "0.25.1", {}, implementation=implementation)
    job = tmp_path / "job.json"
    job.write_text(json.dumps(build_job(store / "families", "gemma4")))
    capture = tmp_path / "capture.yaml"
    version_key = "vllm_version" if implementation == "vllm_python" else "vllm_rust_version"
    capture.write_text(unified_history.dump_yaml({version_key: "0.25.1", "results": {
        "UNIFIED.text_only.gemma4": {
            "capture_input": _request(),
            "assembled": [{"kind": "text", "text": "batch"}],
            "chunks": [[{"kind": "text", "text": "stream"}]],
        },
    }}))
    provenance = {"implementation": implementation, "version": "0.25.1",
                  "job_sha256": digest(job), "source_sha256": "a" * 64,
                  "adapter_sha256": "b" * 64}
    return store, job, capture, provenance, implementation


def test_missing_result_backfill_keeps_python_batch_and_rust_stream_assembly(peer_capture):
    store, job, capture, provenance, implementation = peer_capture
    _write_capture(store, "0.26.0", {}, implementation=implementation)
    import_capture(store, job, capture, provenance, implementation=implementation, backfill_only=True)
    history = unified_history.load_store(store).histories[("gemma4", implementation)]
    result = history.resolve(f"{implementation}-0.25.1")["text_only"]
    record, _document = unified_history.materialized_record(history.family.cases["text_only"], result)
    expected = "batch" if implementation == "vllm_python" else "stream"
    assert record["assembled"] == [{"kind": "text", "text": expected}]
    assert record["capture_input"] == _request()
    assert history.resolve(f"{implementation}-0.26.0") == {}


def test_backfill_retains_existing_bound_result_and_later_version(peer_capture):
    store, job, capture, provenance, implementation = peer_capture
    _write_capture(store, "0.25.1", {"text_only": _change("historic")}, implementation=implementation)
    _write_capture(store, "0.26.0", {}, implementation=implementation)
    history = unified_history.load_store(store).histories[("gemma4", implementation)]
    snapshots = {key: history.resolve(key) for key in history.ordered_capture_ids()}
    assert import_capture(store, job, capture, provenance, implementation=implementation, backfill_only=True) == []
    history = unified_history.load_store(store).histories[("gemma4", implementation)]
    assert {key: history.resolve(key) for key in history.ordered_capture_ids()} == snapshots


@pytest.mark.parametrize("stale", [False, True])
def test_grouped_import_binds_the_preserved_store_generation(peer_capture, stale):
    store, job, capture, provenance, implementation = peer_capture
    inventory = unified_history.store_inventory(store)
    family = store / "families/gemma4/inputs_and_golden.yaml"
    if stale:
        family.write_bytes(family.read_bytes() + b"\n")
    before = {str(path.relative_to(store)): path.read_bytes() for path in store.rglob("*") if path.is_file()}
    groups = {implementation: [(job, capture, provenance)]}
    if stale:
        with pytest.raises(ValueError, match="store generation changed"):
            import_capture_groups(store, groups, expected_inventory=inventory)
        assert {str(path.relative_to(store)): path.read_bytes() for path in store.rglob("*") if path.is_file()} == before
    else:
        assert import_capture_groups(store, groups, expected_inventory=inventory)


@pytest.mark.parametrize("implementation", ["sglang_python", "vllm_python", "vllm_rust", "dynamo_v2"])
def test_receipted_stream_recapture_archives_prior_execution_without_rewriting_history(tmp_path, implementation):
    store = tmp_path / "store"
    _write_family(store)
    _write_capture(store, "0.25.1", {"text_only": _change("historic")}, implementation=implementation)
    job = tmp_path / "job.json"
    job.write_text(json.dumps(build_job(store / "families", "gemma4", mode="stream" if implementation == "dynamo_v2" else "unified")))
    capture = tmp_path / "capture.json"
    key = {"sglang_python": "sglang_version", "vllm_python": "vllm_version", "vllm_rust": "vllm_rust_version"}.get(implementation, "version")
    history = unified_history.load_store(store).histories[("gemma4", implementation)]
    original = history.resolve(f"{implementation}-0.25.1")["text_only"]["observation"]
    for text in ("first", "second"):
        results = {"UNIFIED.text_only.gemma4": {"capture_input": _request(),
            "chunks": [[{"kind": "text", "text": text}]], "assembled": [{"kind": "text", "text": text}]}}
        provenance = {"implementation": implementation, "version": "0.25.1", "job_sha256": digest(job),
            "source_sha256": "a" * 64, "adapter_sha256": "b" * 64,
            "results_sha256": hashlib.sha256(json.dumps(results, sort_keys=True, ensure_ascii=False).encode()).hexdigest()}
        capture.write_text(json.dumps({key: "0.25.1", "results": results, "provenance": provenance}))
        import_capture(store, job, capture, provenance, implementation=implementation)
        history = unified_history.load_store(store).histories[("gemma4", implementation)]
        current = history.resolve(f"{implementation}-0.25.1")["text_only"]
        assert current["observation"] == original
        metadata = current["document"]["record_metadata"]
        assert metadata["verified_recapture"]["resolution"] == "fresh_execution"
        record, document = unified_history.materialized_record(history.family.cases["text_only"], current)
        record.update(document["record_metadata"])
        assert select_peer_observation(record, _request(), implementation=implementation, version="0.25.1")["assembled"][0]["text"] == text
    assert metadata["verified_recaptures"][0]["result"]["assembled"][0]["text"] == "first"
    altered_provenance = {**provenance, "source_sha256": "c" * 64}
    with pytest.raises(ValueError, match="receipt differs from supplied provenance"):
        import_capture(store, job, capture, altered_provenance, implementation=implementation)
    results = {"UNIFIED.text_only.gemma4": {"capture_input": _request(), "unavailable": "No published parser API",
        "capture_observation": {"unavailable": {"code": "unsupported_request_mode", "detail": "No published parser API"}}}}
    provenance = {**provenance, "results_sha256": hashlib.sha256(json.dumps(results, sort_keys=True, ensure_ascii=False).encode()).hexdigest()}
    capture.write_text(json.dumps({key: "0.25.1", "results": results, "provenance": provenance}))
    import_capture(store, job, capture, provenance, implementation=implementation)
    history = unified_history.load_store(store).histories[("gemma4", implementation)]
    current = history.resolve(f"{implementation}-0.25.1")["text_only"]
    assert current["observation"] == original
    metadata = current["document"]["record_metadata"]
    assert [archived["result"]["assembled"][0]["text"] for archived in metadata["verified_recaptures"]] == ["first", "second"]
    assert metadata["verified_recapture"]["provenance"] == provenance
    record, document = unified_history.materialized_record(history.family.cases["text_only"], current)
    record.update(document["record_metadata"])
    selected = select_peer_observation(record, _request(), implementation=implementation, version="0.25.1")
    assert "remains unavailable" in selected["historical_capture_note"]


def test_existing_older_backfill_preserves_later_provenance(peer_capture):
    store, job, capture, provenance, implementation = peer_capture
    _add_backfill_case(store)
    overrides = {"text_only": {"capture_provenance": {
        "captured_with": {implementation: "0.24.0"},
        "capture_origin": None, "inherited_from": None,
    }}}
    _write_capture(store, "0.25.1", {"text_only": _change("old")},
                   implementation=implementation, document_overrides=overrides)
    _write_capture(store, "0.26.0", {"text_only": _change("later")},
                   implementation=implementation, document_overrides=overrides)
    history = unified_history.load_store(store).histories[("gemma4", implementation)]
    later = history.resolve(f"{implementation}-0.26.0")
    job.write_text(json.dumps(build_job(store / "families", "gemma4")))
    payload = unified_history.load_yaml(capture)
    payload["results"]["UNIFIED.new_case.gemma4"] = {
        "capture_input": _request("new"), "assembled": [{"kind": "text", "text": "new"}],
        "chunks": [[{"kind": "text", "text": "new"}]],
    }
    capture.write_text(unified_history.dump_yaml(payload))
    provenance["job_sha256"] = digest(job)
    import_capture(store, job, capture, provenance, implementation=implementation, backfill_only=True)
    history = unified_history.load_store(store).histories[("gemma4", implementation)]
    assert "new_case" in history.resolve(f"{implementation}-0.25.1")
    assert history.resolve(f"{implementation}-0.26.0") == later


@pytest.mark.parametrize("fresh_unavailable", [False, True])
def test_new_execution_preserves_unavailable_adapter_evidence(peer_capture, fresh_unavailable):
    store, job, capture, provenance, implementation = peer_capture
    change = _change("unused")
    change["observation"] = {"unavailable": {"code": "adapter_parser_not_registered", "detail": "old adapter lacked this parser"}}
    _write_capture(store, "0.25.1", {"text_only": change}, implementation=implementation)
    _write_capture(store, "0.26.0", {}, implementation=implementation)
    history = unified_history.load_store(store).histories[("gemma4", implementation)]
    before = history.resolve(f"{implementation}-0.25.1")["text_only"]
    later = history.resolve(f"{implementation}-0.26.0")
    if fresh_unavailable:
        version_key = "vllm_version" if implementation == "vllm_python" else "vllm_rust_version"
        capture.write_text(unified_history.dump_yaml({version_key: "0.25.1", "results": {
            "UNIFIED.text_only.gemma4": {
                "capture_input": _request(), "unavailable": "GuidedJson request mode is unsupported",
            },
        }}))
    import_capture(store, job, capture, provenance, implementation=implementation, backfill_only=True)
    history = unified_history.load_store(store).histories[("gemma4", implementation)]
    after = history.resolve(f"{implementation}-0.25.1")["text_only"]
    assert after["stimulus"] == before["stimulus"]
    assert after["observation"] == before["observation"]
    assert history.resolve(f"{implementation}-0.26.0") == later
    record, document = unified_history.materialized_record(history.family.cases["text_only"], after)
    record.update(document["record_metadata"])
    selected = select_peer_observation(record, _request(), implementation=implementation, version="0.25.1")
    if fresh_unavailable:
        assert selected["unavailable"] == "GuidedJson request mode is unsupported"
        assert "was not executed" in selected["historical_capture_note"]
    else:
        assert "unavailable" not in selected
    assert "historical_capture_note" in selected
    with pytest.raises(ValueError, match="implementation"):
        select_peer_observation(record, _request(), implementation="vllm_python" if implementation == "vllm_rust" else "vllm_rust", version="0.25.1")


def test_parser_error_retains_ordered_partial_chunks(peer_capture):
    store, job, capture, provenance, implementation = peer_capture
    payload = {
        "vllm_version" if implementation == "vllm_python" else "vllm_rust_version": "0.25.1",
        "results": {"UNIFIED.text_only.gemma4": {
            "capture_input": _request(), "error": "ParserError: malformed call",
            "chunks": [[{"kind": "text", "text": "before error"}]],
        }},
    }
    capture.write_text(unified_history.dump_yaml(payload))
    import_capture(store, job, capture, provenance, implementation=implementation)
    history = unified_history.load_store(store).histories[("gemma4", implementation)]
    record, document = unified_history.materialized_record(history.family.cases["text_only"], history.resolve(f"{implementation}-0.25.1")["text_only"])
    record.update(document["record_metadata"])
    assert record["error"] == "ParserError: malformed call"
    assert record["partial_chunks"] == [{"expected": [{"kind": "text", "text": "before error"}]}]


def test_backfill_keeps_each_releases_unavailable_explanation(peer_capture):
    store, job, capture, provenance, implementation = peer_capture
    code = "vllm_rust_parser_not_registered"
    later = _change("unused")
    later["observation"] = {"unavailable": {"code": code, "detail": "No parser at 0.26.0."}}
    _write_capture(store, "0.26.0", {"text_only": later}, implementation=implementation)
    version_key = "vllm_version" if implementation == "vllm_python" else "vllm_rust_version"
    capture.write_text(unified_history.dump_yaml({version_key: "0.25.1", "results": {
        "UNIFIED.text_only.gemma4": {
            "capture_input": _request(), "unavailable": "No parser at 0.25.1.",
            "capture_observation": {"unavailable": {"code": code, "detail": "No parser at 0.25.1."}},
        },
    }}))
    import_capture(store, job, capture, provenance, implementation=implementation, backfill_only=True)
    history = unified_history.load_store(store).histories[("gemma4", implementation)]
    assert history.resolve(f"{implementation}-0.25.1")["text_only"]["observation"]["unavailable"]["detail"] == "No parser at 0.25.1."
    assert history.resolve(f"{implementation}-0.26.0")["text_only"]["observation"]["unavailable"]["detail"] == "No parser at 0.26.0."


@pytest.mark.parametrize("nested_source", [False, True])
@pytest.mark.parametrize("structured_job", [False, True])
@pytest.mark.parametrize("corrupt", [None, "adapter", "job"])
def test_cli_verifies_each_frozen_bundle_format(peer_capture, tmp_path, monkeypatch, nested_source, structured_job, corrupt):
    store, job, capture, provenance, implementation = peer_capture
    bundle = tmp_path / "bundle"
    source = bundle / "utils/src" if nested_source else bundle
    source.mkdir(parents=True)
    adapter = Path(__file__).resolve().parents[1] / "src/capture_stimulus.py"
    (source / adapter.name).write_bytes(adapter.read_bytes())
    expected_job = digest(job)
    if corrupt == "job":
        expected_job = "c" * 64
    manifest = bundle / "manifest.json"
    manifest.write_text(json.dumps({"files": {adapter.name: digest(adapter)},
        "jobs": {"gemma4": {"sha256": expected_job} if structured_job else expected_job}}))
    provenance["adapter_sha256"] = digest(manifest)
    provenance_path = tmp_path / "provenance.json"
    provenance_path.write_text(json.dumps(provenance))
    if corrupt == "adapter":
        (source / adapter.name).write_bytes(b"changed adapter")
    before = {path.relative_to(store): path.read_bytes() for path in store.rglob("*.yaml")}
    monkeypatch.setattr(sys, "argv", ["import_peer_unified.py", "--store", str(store),
        "--job", str(job), "--capture", str(capture), "--provenance", str(provenance_path),
        "--bundle", str(bundle), "--implementation", implementation, "--backfill-only"])
    if corrupt:
        with pytest.raises(ValueError, match="adapter differs" if corrupt == "adapter" else "job differs"):
            main()
        assert {path.relative_to(store): path.read_bytes() for path in store.rglob("*.yaml")} == before
    else:
        main()
        history = unified_history.load_store(store).histories[("gemma4", implementation)]
        assert "text_only" in history.resolve(f"{implementation}-0.25.1")


def test_cli_rejects_implementation_bundle_without_consumed_registry(peer_capture, tmp_path, monkeypatch):
    store, job, capture, provenance, implementation = peer_capture
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    source = Path(__file__).resolve().parents[1] / "src/impls.py"
    (bundle / source.name).write_bytes(source.read_bytes())
    (bundle / "manifest.json").write_text(json.dumps({"files": {source.name: digest(source)}, "jobs": {}}))
    monkeypatch.setattr(sys, "argv", ["import_peer_unified.py", "--store", str(store),
        "--job", str(job), "--capture", str(capture), "--provenance", str(tmp_path / "unused.json"),
        "--bundle", str(bundle), "--implementation", implementation])
    with pytest.raises(ValueError, match="omits parser_families.yaml"):
        main()
