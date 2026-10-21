# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Import request-bound peer captures without replacing historical observations."""

import argparse
import copy
import hashlib
import json
import re
from pathlib import Path

import yaml
import yaml_fast  # noqa: F401

from capture_unified_job import build_job
from capture_stimulus import batch_final_events, capture_input, select_peer_observation
from explode_unified_fixtures import _peer_cell
from generate_conformance_table import _assemble_stream
from impls import BATCH_SOURCE_BLOCKERS, UNIFIED_BATCH_IMPLS
import unified_history


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def import_capture(store_root: Path, job_path: Path, capture_path: Path, provenance: dict, *, implementation="sglang_python", backfill_only=False, transaction_store=None) -> list[Path]:
    source_unavailable = provenance.get("status") == "source_unavailable"
    if source_unavailable and (implementation, provenance.get("version")) not in BATCH_SOURCE_BLOCKERS:
        raise ValueError("unrecognized batch producer source blocker")
    for key in (("adapter_sha256", "job_sha256") if source_unavailable else ("source_sha256", "adapter_sha256", "job_sha256")):
        if not isinstance(provenance.get(key), str) or re.fullmatch(r"[0-9a-f]{64}", provenance[key]) is None:
            raise ValueError(f"invalid capture provenance: {key}")
    batch = implementation in UNIFIED_BATCH_IMPLS
    if batch and provenance.get("implementation") != implementation:
        raise ValueError("batch producer implementation differs from history")
    if source_unavailable and "source_sha256" in provenance:
        raise ValueError("source-unavailable producer cannot claim a source hash")
    job = json.loads(job_path.read_text())
    family = job["family"]
    if job != build_job(store_root / "families", family, mode="batch" if batch else "unified"):
        raise ValueError("capture job differs from the final candidate request corpus")
    capture = json.loads(capture_path.read_text()) if capture_path.suffix == ".json" else yaml.safe_load(capture_path.read_text())
    version_field = "version" if batch else {"sglang_python": "sglang_version", "vllm_python": "vllm_version", "vllm_rust": "vllm_rust_version"}[implementation]
    version = capture[version_field]
    if provenance["version"] != version or provenance["job_sha256"] != digest(job_path):
        raise ValueError("capture provenance differs from job or installed version")
    origin = None
    if batch and not source_unavailable:
        origin = {"crate_version": version, **{key: provenance[key] for key in ("source_sha256", "adapter_sha256", "job_sha256")}}
        if "git_commit" in provenance:
            origin["git_commit"] = provenance["git_commit"]
    if batch:
        receipt = capture.get("provenance")
        receipt_keys = ("implementation", "version", "adapter_sha256", "job_sha256") if source_unavailable else ("implementation", "version", "source_sha256", "adapter_sha256", "job_sha256")
        if "git_commit" in provenance:
            receipt_keys += ("git_commit",)
        if not isinstance(receipt, dict) or any(receipt.get(key) != provenance.get(key) for key in receipt_keys):
            raise ValueError("batch producer receipt differs from supplied provenance")
    by_id = {case["id"]: case for case in job["cases"]}
    if capture["results"].keys() != by_id.keys():
        raise ValueError("capture results do not cover the exact family job")
    records = {}
    for key, result in capture["results"].items():
        case = by_id[key]
        request = case["request"] if batch else capture_input({**case, "chunks": [{"delta_text": text} for text in case["chunks"]]})
        if result.get("capture_input") != request:
            raise ValueError(f"capture request differs from job: {key}")
        if batch:
            if source_unavailable and result.get("capture_observation") != {"unavailable": {"code": "source_unavailable", "detail": BATCH_SOURCE_BLOCKERS[(implementation, version)]}}:
                raise ValueError("source-unavailable producer cannot claim an execution")
            if "chunks" in result:
                raise ValueError("batch captures cannot contain chunk emissions")
            if "assembled" in result:
                if result["assembled"] != batch_final_events(result.get("batch_final", {})):
                    raise ValueError("batch assembly differs from its final message fields")
            elif "batch_final" in result:
                raise ValueError("failed batch capture cannot carry final fields")
            records[case["scenario"]] = result
            continue
        if "error" in result and "chunks" in result and "partial_chunks" not in result:
            result = {**result, "partial_chunks": result["chunks"]}
        if implementation != "vllm_python" and "error" not in result and "unavailable" not in result:
            result = {**result, "assembled": _assemble_stream(result["chunks"])}
        records[case["scenario"]] = _peer_cell(result, len(case["chunks"]))

    def mutate(store):
        fam = store.families[family]
        capture_id = f"{implementation}-{version}"
        history = store.histories.get((family, implementation))
        existing = history.resolve(capture_id) if history and capture_id in history.captures else {}
        if batch and existing and origin is not None:
            previous_origin = history.captures[capture_id]["provenance"].get("origin", {})
            if any(previous_origin.get(key) != origin.get(key) for key in ("crate_version", "source_sha256", "job_sha256", "git_commit")):
                raise ValueError("batch producer provenance is immutable")
        target = copy.deepcopy(existing)
        previous_ids = [key for key in history.ordered_capture_ids() if unified_history._capture_release_sort_key(history.captures[key]["runtime_version"]) < unified_history._capture_release_sort_key(version)] if history else []
        previous = history.resolve(previous_ids[-1]) if batch and previous_ids else {}
        for case_id, case in fam.cases.items():
            if case.get("lifecycle", "active") == "retired":
                continue
            result = records[case["scenario"]]
            if result["capture_input"] != case["request"]:
                raise ValueError(f"capture request changed during import: {family}/{case_id}")
            before = existing.get(case_id)
            document = {"mode": "unified", "captured_with": {implementation: version}}
            unbound = before is not None and before["stimulus"] == {
                "unavailable": {"code": "original_request_not_retained"}
            }
            adapter_unavailable = (
                before is not None and implementation != "sglang_python"
                and "unavailable" in before["observation"]
                and (not batch or before["observation"]["unavailable"].get("code") == "unsupported_request_mode")
                and ("unavailable" not in result
                     or result["unavailable"] != before["observation"]["unavailable"].get("detail"))
            )
            if unbound or adapter_unavailable:
                # Retain the earlier observation and store this recapture separately;
                # an adapter limitation or a lost request is not a parser result.
                supplement = {"result": result, "provenance": provenance,
                              "resolution": ("historical_request_not_retained" if unbound else "historical_adapter_unsupported")}
                original, metadata = unified_history.materialized_record(case, before)
                original.update(metadata.get("record_metadata", {}))
                original["verified_recapture"] = supplement
                select_peer_observation(original, case["request"], implementation=implementation, version=version)
                old_supplement = before["document"].get("record_metadata", {}).get("verified_recapture")
                if old_supplement is not None and old_supplement != supplement:
                    raise ValueError(f"verified recapture is immutable: {family}/{case_id}")
                target[case_id]["document"].setdefault("record_metadata", {})["verified_recapture"] = supplement
                continue
            if before is not None and backfill_only:
                continue
            stimulus, observation, metadata = unified_history._legacy_state(
                result, b"", case_id, {}, case["request"],
            )
            original_metadata = copy.deepcopy(before["document"].get("record_metadata", {})) if before is not None else {}
            supplement = original_metadata.pop("verified_recapture", None)
            if before is not None and supplement is not None and before["stimulus"] == stimulus and before["observation"] == observation:
                continue
            if batch and before is not None and before["stimulus"] == stimulus and before["observation"] == observation and original_metadata == metadata:
                continue
            document["record_metadata"] = metadata if batch else {**metadata, f"{implementation.removesuffix("_python")}_capture_provenance": provenance}
            if origin is not None:
                document["capture_origin"] = origin
            after = {"case_key": case_id, "stimulus": stimulus, "observation": observation, "document": document}
            if before is not None and (
                unified_history._capture_semantic(before, case_id) != unified_history._capture_semantic(after, case_id)
                or before["observation"] != after["observation"]
            ):
                raise ValueError(f"bound {"SGLang" if implementation == "sglang_python" else implementation} observation is immutable: {family}/{case_id}")
            prior = previous.get(case_id)
            if batch and prior is not None and prior["stimulus"] == after["stimulus"] and prior["observation"] == after["observation"] and prior["document"].get("record_metadata") == after["document"].get("record_metadata"):
                after = copy.deepcopy(prior)
                after["document"]["inherited_from"] = prior["document"]["captured_with"][implementation]
            target[case_id] = after
        if target == existing:
            return
        if history is None:
            history = store.history_for_capture(family, implementation)
        if capture_id not in history.captures:
            history.captures[capture_id] = {
                "provenance": {"status": "legacy", "captured_with": {implementation: version}},
                "document": {"mode": "unified"}, "changes": {},
                "metadata_changes": {}, "document_overrides": {}, "runtime_version": version,
            }
        if origin is not None and not existing:
            history.captures[capture_id]["provenance"] = {"status": "captured", "origin": origin}
        history.capture_paths[capture_id] = fam.path.parent / f"{capture_id}.yaml"
        unified_history._replace_capture_snapshot(history, capture_id, target)

    if transaction_store is not None:
        mutate(transaction_store)
        return []
    return unified_history._mutate_store(store_root, mutate)


def import_captures(store_root: Path, captures: list[tuple[Path, Path, dict]], *, implementation="sglang_python", backfill_only=False) -> list[Path]:
    """Publish a multi-version capture batch as one validated store generation."""
    def mutate(store):
        for job, capture, provenance in captures:
            import_capture(store_root, job, capture, provenance, implementation=implementation, backfill_only=backfill_only, transaction_store=store)
    return unified_history._mutate_store(store_root, mutate)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--implementation", choices=("sglang_python", "vllm_python", "vllm_rust", *UNIFIED_BATCH_IMPLS), default="sglang_python")
    parser.add_argument("--backfill-only", action="store_true", help="Preserve existing bound observations and append missing results")
    args = parser.parse_args()
    manifest_path = args.bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if "impls.py" in manifest["files"] and "parser_families.yaml" not in manifest["files"]:
        raise ValueError("frozen adapter bundle omits parser_families.yaml required by impls.py")
    source = Path(__file__).resolve().parent
    bundle_source = args.bundle / "utils/src" if (args.bundle / "utils/src").is_dir() else args.bundle
    for name, expected in manifest["files"].items():
        if digest(bundle_source / name) != expected or digest(source / name) != expected:
            raise ValueError(f"capture adapter differs from final candidate: {name}")
    job = json.loads(args.job.read_text())
    expected_job = manifest["jobs"][job["family"]]
    if isinstance(expected_job, dict):
        expected_job = expected_job["sha256"]
    if digest(args.job) != expected_job:
        raise ValueError("job differs from frozen bundle")
    provenance = json.loads(args.provenance.read_text())
    if provenance["adapter_sha256"] != digest(manifest_path):
        raise ValueError("provenance differs from frozen adapter bundle")
    for path in import_capture(args.store, args.job, args.capture, provenance, implementation=args.implementation, backfill_only=args.backfill_only):
        print(path)


if __name__ == "__main__":
    main()
