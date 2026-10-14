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
from capture_stimulus import capture_input, select_peer_observation
from explode_unified_fixtures import _peer_cell
from generate_conformance_table import _assemble_stream
import unified_history


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def import_capture(store_root: Path, job_path: Path, capture_path: Path, provenance: dict, *, implementation="sglang_python", backfill_only=False, transaction_store=None) -> list[Path]:
    for key in ("source_sha256", "adapter_sha256", "job_sha256"):
        if not isinstance(provenance.get(key), str) or re.fullmatch(r"[0-9a-f]{64}", provenance[key]) is None:
            raise ValueError(f"invalid capture provenance: {key}")
    job = json.loads(job_path.read_text())
    family = job["family"]
    if job != build_job(store_root / "families", family):
        raise ValueError("capture job differs from the final candidate request corpus")
    capture = yaml.safe_load(capture_path.read_text())
    version_field = {"sglang_python": "sglang_version", "vllm_python": "vllm_version", "vllm_rust": "vllm_rust_version"}[implementation]
    version = capture[version_field]
    if provenance["version"] != version or provenance["job_sha256"] != digest(job_path):
        raise ValueError("capture provenance differs from job or installed version")
    by_id = {case["id"]: case for case in job["cases"]}
    if capture["results"].keys() != by_id.keys():
        raise ValueError("capture results do not cover the exact family job")
    records = {}
    for key, result in capture["results"].items():
        case = by_id[key]
        request = capture_input({**case, "chunks": [{"delta_text": text} for text in case["chunks"]]})
        if result.get("capture_input") != request:
            raise ValueError(f"capture request differs from job: {key}")
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
        target = copy.deepcopy(existing)
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
            document["record_metadata"] = {**metadata, f"{implementation.removesuffix("_python")}_capture_provenance": provenance}
            after = {"case_key": case_id, "stimulus": stimulus, "observation": observation, "document": document}
            if before is not None and (
                unified_history._capture_semantic(before, case_id) != unified_history._capture_semantic(after, case_id)
                or before["observation"] != after["observation"]
            ):
                raise ValueError(f"bound {"SGLang" if implementation == "sglang_python" else implementation} observation is immutable: {family}/{case_id}")
            target[case_id] = after
        if target == existing:
            return
        if history is None:
            history = unified_history.History(fam, implementation, {}, fam.path.parent, {})
            store.histories[(family, implementation)] = history
        if capture_id not in history.captures:
            history.captures[capture_id] = {
                "provenance": {"status": "legacy", "captured_with": {implementation: version}},
                "document": {"mode": "unified"}, "changes": {},
                "metadata_changes": {}, "document_overrides": {}, "runtime_version": version,
            }
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
    parser.add_argument("--implementation", choices=("sglang_python", "vllm_python", "vllm_rust"), default="sglang_python")
    parser.add_argument("--backfill-only", action="store_true", help="Preserve existing bound observations and append missing results")
    args = parser.parse_args()
    manifest_path = args.bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
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
