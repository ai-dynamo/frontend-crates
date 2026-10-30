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
from capture_stimulus import execution_outcome
from capture_stimulus import capture_input, select_peer_observation
from explode_unified_fixtures import _peer_cell
from impls import BATCH_SOURCE_BLOCKERS, UNIFIED_BATCH_IMPLS
from capture_stimulus import assemble_stream
import unified_history
import execution_manifests
from viewer_codec import typed_equal


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _execution_proof(family, provenance):
    return {"receipt": copy.deepcopy(provenance),
            "requests": {key: unified_history._request_digest(case["request"])
                         for key, case in family.cases.items() if case.get("lifecycle", "active") != "retired"}}


def import_capture(store_root: Path, job_path: Path, capture_path: Path, provenance: dict, *, implementation="sglang_python", backfill_only=False, proof_only=False, selected_cases: set[str] | None = None, transaction_store=None, transaction_snapshots=None) -> list[Path]:
    source_unavailable = provenance.get("status") == "source_unavailable"
    if source_unavailable and (implementation, provenance.get("version")) not in BATCH_SOURCE_BLOCKERS:
        raise ValueError("unrecognized batch producer source blocker")
    for key in (("adapter_sha256", "job_sha256") if source_unavailable else ("source_sha256", "adapter_sha256", "job_sha256")):
        if not isinstance(provenance.get(key), str) or re.fullmatch(r"[0-9a-f]{64}", provenance[key]) is None:
            raise ValueError(f"invalid capture provenance: {key}")
    batch = implementation in UNIFIED_BATCH_IMPLS
    execution = batch or implementation == "dynamo_v2"
    if execution and provenance.get("implementation") != implementation:
        raise ValueError("batch producer implementation differs from history")
    if source_unavailable and "source_sha256" in provenance:
        raise ValueError("source-unavailable producer cannot claim a source hash")
    job = json.loads(job_path.read_text())
    family = job["family"]
    mode = "batch" if batch else "stream" if implementation == "dynamo_v2" else "unified"
    if transaction_snapshots is None:
        expected_job = build_job(store_root / "families", family, mode=mode)
    else:
        # The enclosing mutation lease fixes authored inputs for this transaction.
        pending = transaction_snapshots[(family, implementation)]
        if "expected_job" not in pending:
            pending["expected_job"] = build_job(store_root / "families", family, mode=mode)
        expected_job = pending["expected_job"]
    if not typed_equal(job, expected_job):
        raise ValueError("capture job differs from the final candidate request corpus")
    capture = json.loads(capture_path.read_text()) if capture_path.suffix == ".json" else yaml.safe_load(capture_path.read_text())
    version_field = "version" if execution else {"sglang_python": "sglang_version", "vllm_python": "vllm_version", "vllm_rust": "vllm_rust_version"}[implementation]
    version = capture[version_field]
    if provenance["version"] != version or provenance["job_sha256"] != digest(job_path):
        raise ValueError("capture provenance differs from job or installed version")
    origin = None
    if execution and not source_unavailable:
        origin = {"crate_version": version, **{key: provenance[key] for key in ("source_sha256", "adapter_sha256", "job_sha256")}}
        if "git_commit" in provenance:
            origin["git_commit"] = provenance["git_commit"]
    if execution:
        receipt = capture.get("provenance")
        receipt_keys = ("implementation", "version", "adapter_sha256", "job_sha256") if source_unavailable else ("implementation", "version", "source_sha256", "adapter_sha256", "job_sha256")
        if "git_commit" in provenance:
            receipt_keys += ("git_commit",)
        if not isinstance(receipt, dict) or any(receipt.get(key) != provenance.get(key) for key in receipt_keys):
            raise ValueError("batch producer receipt differs from supplied provenance")
    if (implementation in {"dynamo_v2", "dynamo_v1", "dynamo_v1_jail"} and not source_unavailable) or "results_sha256" in provenance or "results_sha256" in capture.get("provenance", {}):
        expected_results = hashlib.sha256(json.dumps(capture["results"], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        if provenance.get("results_sha256") != expected_results or capture.get("provenance", {}).get("results_sha256") != expected_results:
            raise ValueError("producer receipt differs from captured results")
    stream_receipt = not execution and "results_sha256" in provenance
    if stream_receipt and not typed_equal(capture.get("provenance"), provenance):
        raise ValueError("producer receipt differs from supplied provenance")
    verified_execution = execution or stream_receipt
    if verified_execution and provenance.get("implementation") != implementation:
        raise ValueError("execution receipt implementation differs from history")
    by_id = {case["id"]: case for case in job["cases"]}
    if capture["results"].keys() != by_id.keys():
        raise ValueError("capture results do not cover the exact family job")
    outcomes = [execution_outcome(result, "batch" if batch else "stream")[0] for result in capture["results"].values()] if verified_execution else []
    records = {}
    for key, result in capture["results"].items():
        case = by_id[key]
        request = case["request"] if execution else capture_input({**case, "chunks": [{"delta_text": text} for text in case["chunks"]]})
        if not typed_equal(result.get("capture_input"), request):
            raise ValueError(f"capture request differs from job: {key}")
        if implementation == "dynamo_v2":
            records[case["scenario"]] = result
            continue
        if batch:
            if source_unavailable and result.get("capture_observation") != {"unavailable": {"code": "source_unavailable", "detail": BATCH_SOURCE_BLOCKERS[(implementation, version)]}}:
                raise ValueError("source-unavailable producer cannot claim an execution")
            records[case["scenario"]] = result
            continue
        if "error" in result and "chunks" in result and "partial_chunks" not in result:
            result = {**result, "partial_chunks": result["chunks"]}
        if implementation != "vllm_python" and "error" not in result and "unavailable" not in result:
            result = {**result, "assembled": assemble_stream(result["chunks"])}
        records[case["scenario"]] = _peer_cell(result, len(case["chunks"]))

    def mutate(store):
        fam = store.families[family]
        active_cases = {key for key, case in fam.cases.items() if case.get("lifecycle", "active") != "retired"}
        if selected_cases is not None and (not selected_cases or not selected_cases <= active_cases):
            raise ValueError("selected capture cases must name existing active cases")
        capture_id = f"{implementation}-{version}"
        history = store.histories.get((family, implementation))
        if proof_only:
            if not verified_execution or source_unavailable or history is None or capture_id not in history.captures:
                raise ValueError("execution proof requires an existing executed checkpoint")
            checkpoint = history.captures[capture_id]
            if checkpoint["provenance"] != {"status": "captured", "origin": origin}:
                raise ValueError("execution proof differs from checkpoint origin")
            if not typed_equal(capture.get("provenance"), provenance):
                raise ValueError("execution proof receipt differs from sealed capture")
            if not any(outcome in {"executed", "parser_error"} for outcome in outcomes) or "results_sha256" not in provenance:
                raise ValueError("execution proof requires sealed parser execution")
            checkpoint["execution_proof"] = _execution_proof(fam, provenance)
            return
        snapshots = transaction_snapshots[(family, implementation)] if transaction_snapshots is not None else None
        def resolve(identity):
            if snapshots is None:
                return history.resolve(identity)
            return snapshots["targets"].get(identity, snapshots["original"].get(identity, {}))
        existing = resolve(capture_id) if history and capture_id in history.captures else {}
        target = copy.deepcopy(existing)
        previous_ids = [key for key in history.ordered_capture_ids() if unified_history._capture_release_sort_key(history.captures[key]["runtime_version"]) < unified_history._capture_release_sort_key(version)] if history else []
        previous = resolve(previous_ids[-1]) if batch and previous_ids else {}
        for case_id, case in fam.cases.items():
            if case.get("lifecycle", "active") == "retired":
                continue
            if selected_cases is not None and case_id not in selected_cases:
                continue
            result = records[case["scenario"]]
            if not typed_equal(result["capture_input"], case["request"]):
                raise ValueError(f"capture request changed during import: {family}/{case_id}")
            before = existing.get(case_id)
            document = {"mode": "unified", "captured_with": {implementation: version}}
            if execution and before is not None:
                root_origin = history.captures[capture_id]["provenance"].get("origin", {})
                known = all(root_origin.get(key) == provenance.get(key) for key in ("source_sha256", "adapter_sha256", "job_sha256"))
                stimulus, observation, _metadata = unified_history._legacy_state(result, b"", case_id, {}, case["request"])
                if known and typed_equal(before["stimulus"], stimulus) and typed_equal(before["observation"], observation):
                    continue
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
            if unbound or adapter_unavailable or (verified_execution and before is not None and not backfill_only):
                # Retain the earlier observation and store this recapture separately;
                # an adapter limitation or a lost request is not a parser result.
                supplement = {"result": result, "provenance": provenance,
                              "resolution": ("historical_request_not_retained" if unbound else "fresh_execution" if verified_execution else "historical_adapter_unsupported")}
                original, metadata = unified_history.materialized_record(case, before)
                original.update(metadata.get("record_metadata", {}))
                original["verified_recapture"] = supplement
                select_peer_observation(original, case["request"], implementation=implementation, version=version)
                old_supplement = before["document"].get("record_metadata", {}).get("verified_recapture")
                metadata = target[case_id]["document"].setdefault("record_metadata", {})
                if old_supplement is not None and not typed_equal(old_supplement, supplement):
                    if not verified_execution:
                        raise ValueError(f"verified recapture is immutable: {family}/{case_id}")
                    archive = metadata.setdefault("verified_recaptures", [])
                    if not any(typed_equal(old_supplement, archived) for archived in archive):
                        archive.append(old_supplement)
                metadata["verified_recapture"] = supplement
                continue
            if before is not None and backfill_only:
                continue
            stimulus, observation, metadata = unified_history._legacy_state(
                result, b"", case_id, {}, case["request"],
            )
            original_metadata = copy.deepcopy(before["document"].get("record_metadata", {})) if before is not None else {}
            supplement = original_metadata.pop("verified_recapture", None)
            if before is not None and supplement is not None and typed_equal(before["stimulus"], stimulus) and typed_equal(before["observation"], observation):
                continue
            if batch and before is not None and typed_equal(before["stimulus"], stimulus) and typed_equal(before["observation"], observation) and typed_equal(original_metadata, metadata):
                continue
            document["record_metadata"] = metadata if batch and implementation not in {"dynamo_v1", "dynamo_v1_jail"} else {**metadata, f"{implementation.removesuffix("_python")}_capture_provenance": provenance}
            if origin is not None:
                document["capture_origin"] = origin
            after = {"case_key": case_id, "stimulus": stimulus, "observation": observation, "document": document}
            if before is not None and (
                not typed_equal(unified_history._capture_semantic(before, case_id), unified_history._capture_semantic(after, case_id))
            ):
                raise ValueError(f"bound {"SGLang" if implementation == "sglang_python" else implementation} observation is immutable: {family}/{case_id}")
            prior = previous.get(case_id)
            if execution and prior is not None and typed_equal(prior["stimulus"], after["stimulus"]) and typed_equal(prior["observation"], after["observation"]) and typed_equal(prior["document"].get("record_metadata"), after["document"].get("record_metadata")):
                after = copy.deepcopy(prior)
                after["document"]["inherited_from"] = prior["document"]["captured_with"][implementation]
            target[case_id] = after
        def retain_dropped_receipt(checkpoint):
            if ("results_sha256" not in provenance
                    or not any(outcome in {"executed", "parser_error"} for outcome in outcomes)):
                return
            proof = checkpoint.get("execution_proof")
            if proof is not None and typed_equal(proof["receipt"], provenance):
                return
            if any(typed_equal(owner[key], provenance) for record in target.values()
                   for owner, key in execution_manifests._receipt_slots(record["document"].get("record_metadata", {}))):
                return
            if proof is not None or checkpoint["provenance"] != {"status": "captured", "origin": origin}:
                raise ValueError("sealed execution receipt cannot be retained without changing checkpoint provenance")
            checkpoint["execution_proof"] = _execution_proof(fam, provenance)

        if typed_equal(target, existing):
            if history is not None and capture_id in history.captures:
                retain_dropped_receipt(history.captures[capture_id])
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
        retain_dropped_receipt(history.captures[capture_id])
        if snapshots is None:
            unified_history._replace_capture_snapshot(history, capture_id, target)
        else:
            snapshots["targets"][capture_id] = target

    if transaction_store is not None:
        mutate(transaction_store)
        return []
    return unified_history._mutate_store(store_root, mutate)


def import_captures(store_root: Path, captures: list[tuple[Path, Path, dict]], *, implementation="sglang_python", backfill_only=False) -> list[Path]:
    """Publish a multi-version capture batch as one validated store generation."""
    return import_capture_groups(store_root, {implementation: captures}, backfill_only=backfill_only)


def import_capture_groups(store_root: Path, groups: dict[str, list[tuple[Path, Path, dict]]], *, backfill_only=False, proof_only=False, compact_executions=None, expected_inventory=None) -> list[Path]:
    """Publish separate implementation histories in one validated generation."""
    def mutate(store):
        snapshots = {}
        for implementation, captures in groups.items():
            for job, _capture, _provenance in captures:
                family = json.loads(job.read_text())["family"]
                key = (family, implementation)
                if key not in snapshots:
                    history = store.histories.get(key)
                    snapshots[key] = {"original": dict(history.iter_resolved_snapshots()) if history else {},
                                      "targets": {}}
            for job, capture, provenance in sorted(captures, key=lambda entry: (unified_history._capture_release_sort_key(entry[2]["version"]), str(entry[0]))):
                import_capture(store_root, job, capture, provenance, implementation=implementation,
                               backfill_only=backfill_only, proof_only=proof_only, transaction_store=store, transaction_snapshots=snapshots)
        for key, pending in snapshots.items():
            if not pending["targets"]:
                continue
            history = store.histories[key]
            previous = {}
            cursor = unified_history._HistoryCursor(history)
            for identity in history.ordered_capture_ids():
                target = pending["targets"].get(identity, pending["original"].get(identity, {}))
                unified_history._set_capture_snapshot(history, identity, target, previous, cursor=cursor)
                previous = target

    return unified_history._mutate_store(store_root, mutate, compact_executions=compact_executions,
                                        expected_inventory=expected_inventory)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--implementation", choices=("sglang_python", "vllm_python", "vllm_rust", "dynamo_v2", *UNIFIED_BATCH_IMPLS), default="sglang_python")
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
