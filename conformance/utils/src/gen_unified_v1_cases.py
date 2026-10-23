#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate family-native Unified cases and Dynamo v1 captures from archived fixtures."""

from __future__ import annotations

import argparse
import copy
import json
import re
import shutil
import tempfile
import tarfile
from pathlib import Path
from typing import Any

import yaml

import gen_unified_golden
import unified_history
import unified_archives
from viewer_codec import typed_equal

ROOT = Path(__file__).resolve().parents[3]
INPUT_ARCHIVE = Path("conformance/fixtures/toolcalling/fixtures-batch-v1/inputs.tar.gz")
MANIFEST = Path("conformance/fixtures-manifest.json")
CASE_KEY = "v1_split_parallel_tool_calls"
DISPLAY_ID = "UNIFIED.v1_split-parallel_tool_calls"
PREFERRED_LEGACY_CASE_ID = "TOOLCALLING.batch.2.a"
FAMILY_ALIASES = {"gpt_oss": "harmony", "harmony_text": "harmony"}


def _capture_archives(root: Path) -> list[tuple[tuple[int, ...], str, Path]]:
    manifest = json.loads((root / MANIFEST).read_text())
    archives = []
    for shard in manifest["shards"]:
        relative = shard["path"]
        match = re.search(r"/dynamo_v1-([0-9]+(?:\.[0-9]+)+)\.tar\.gz$", relative)
        if match and "/fixtures-batch-v1/" in relative:
            version = match.group(1)
            archives.append((tuple(int(part) for part in version.split(".")), version, root / "conformance/fixtures" / relative))
    return sorted(archives)


def _archive_document(archive_path: Path, member: str) -> dict[str, Any] | None:
    with tarfile.open(archive_path, "r:gz") as archive:
        try:
            stream = archive.extractfile(member)
        except KeyError:
            return None
        if stream is None:
            return None
        document = yaml.safe_load(stream)
    return document if isinstance(document, dict) else None


def _input_cases(root: Path, source_family: str) -> dict[str, dict[str, Any]]:
    prefix = f"toolcalling/fixtures-batch-v1/inputs/{source_family}/"
    cases = {}
    with tarfile.open(root / INPUT_ARCHIVE, "r:gz") as archive:
        for member in archive.getnames():
            if not member.startswith(prefix) or not member.endswith(".yaml"):
                continue
            stream = archive.extractfile(member)
            if stream is None:
                continue
            document = yaml.safe_load(stream)
            cases.update((case_id, case) for case_id, case in (document or {}).get("cases", {}).items())
    return cases


def _captured_call_map(
    archives: list[tuple[tuple[int, ...], str, Path]], families: set[str]
 ) -> dict[tuple[str, str, str], dict]:
    captures = {}
    for _, version, archive_path in archives:
        member_prefix = f"toolcalling/fixtures-batch-v1/dynamo_v1-{version}/"
        with tarfile.open(archive_path, "r:gz") as archive:
            for member in archive.getnames():
                if not member.startswith(member_prefix) or not member.endswith(".yaml"):
                    continue
                family = member[len(member_prefix):].split("/", 1)[0]
                if family not in families:
                    continue
                stream = archive.extractfile(member)
                if stream is None:
                    continue
                document = yaml.safe_load(stream)
                for case_id, case in (document or {}).get("cases", {}).items():
                    result = case.get("expected", {}).get("dynamo_v1")
                    if not isinstance(result, dict) or len(result.get("calls", [])) != 2 or result.get("normal_text", ""):
                        continue
                    events = [
                        {"kind": "tool_call", "name": call["name"], "arguments": call["arguments"]}
                        for call in result["calls"]
                    ]
                    captures[(family, version, case_id)] = {"assembled": events, "chunks": []}
    return captures


def _merge_generated_case(root: Path, path: Path, content: str) -> str:
    """Add this case without replacing existing family inputs or histories."""
    generated = yaml.safe_load(content)
    existing_path = root / path
    if not existing_path.is_file():
        return content
    original = existing_path.read_bytes().decode("utf-8")
    existing = unified_history.load_yaml(existing_path)
    if existing.get("schema_version") == 2:
        generated_capture = generated
        family = existing_path.parent.name
        identity = unified_history.CAPTURE_DIRECTORY_RE.fullmatch(existing_path.stem)
        if identity is None:
            raise ValueError(f"invalid complete snapshot identity: {existing_path}")
        value = next(iter(generated_capture.get("changes", {}).values()), None)
        if value is None:
            raise ValueError(f"generated Unified snapshot has no case: {existing_path}")
        case_id = CASE_KEY
        provenance = generated_capture.get("document_overrides", {}).get(case_id, {}).get("capture_provenance", {})
        before = copy.deepcopy(existing)
        retained = existing.get("records", {}).get(case_id, {}).get("value", {})
        record_document = copy.deepcopy(retained.get("document", existing.get("document", {})))
        for key in ("captured_with", "capture_origin", "inherited_from"):
            if key in provenance:
                if provenance[key] is None:
                    record_document.pop(key, None)
                else:
                    record_document[key] = copy.deepcopy(provenance[key])
        record = {key: copy.deepcopy(value[key]) for key in ("case_key", "stimulus", "observation")}
        record["document"] = record_document
        existing.setdefault("records", {})[case_id] = {
            "value": record,
            "member": (Path("conformance/fixtures-unified-v2") / "families" / family
                       / f"{existing_path.name}").as_posix(),
        }
        return original if typed_equal(before, existing) else unified_history.dump_yaml(existing, shared_metadata=True)
    compact = "execution" in existing
    if compact:
        family_path = existing_path.parent / unified_history.FAMILY_DOCUMENT_NAME
        family = unified_history._validate_family(family_path, unified_history.load_yaml(family_path))
        _, _, _, capture = unified_history._validate_capture_file(existing_path, existing, {family.name: family})
        existing = {key: value for key, value in capture.items() if key != "runtime_version"}
    before = copy.deepcopy(existing)
    if path.name == "inputs_and_golden.yaml":
        existing.setdefault("cases", {}).update(generated["cases"])
    else:
        for field in ("changes", "metadata_changes", "document_overrides"):
            existing.setdefault(field, {}).update(generated.get(field, {}))
    if compact and typed_equal(existing, before):
        return original
    # A changed compact checkpoint must be emitted wholly expanded; mixing its
    # execution reference with inline metadata violates the checkpoint schema.
    return unified_history.dump_yaml(existing)


def generated_documents(root: Path = ROOT) -> dict[Path, str]:
    with unified_archives.expanded(root / "conformance/fixtures-unified-v2") as source:
        return _generated_documents(root, source)


def _generated_documents(root: Path, source: Path) -> dict[Path, str]:
    archives = _capture_archives(root)
    source_families = {FAMILY_ALIASES.get(family, family) for family in gen_unified_golden.V1_SPLIT_FAMILIES}
    captures = _captured_call_map(archives, source_families)
    documents = {}
    for family in sorted(gen_unified_golden.V1_SPLIT_FAMILIES):
        source_family = FAMILY_ALIASES.get(family, family)
        input_cases = _input_cases(root, source_family)
        captured = None
        for _, version, _ in reversed(archives):
            eligible = sorted(
                case_id for (captured_family, captured_version, case_id) in captures
                if captured_family == source_family and captured_version == version and case_id in input_cases
            )
            if not eligible:
                continue
            case_id = PREFERRED_LEGACY_CASE_ID if PREFERRED_LEGACY_CASE_ID in eligible else eligible[0]
            captured = (version, case_id, captures[(source_family, version, case_id)])
            break
        if captured is None:
            raise ValueError(f"no matching Dynamo v1 two-call fixture for {source_family}")
        version, legacy_case_id, value = captured
        input_case = input_cases[legacy_case_id]
        model_text = input_case.get("model_text")
        tools = input_case.get("tools")
        if not isinstance(model_text, str) or not isinstance(tools, list):
            raise ValueError(f"{source_family}/{legacy_case_id} has an incomplete archived input")
        request = {
            "input": model_text,
            "init": {"starting_state": "None", "tool_output_mode": "Native", "named_tool": None},
            "finish_reason": "stop",
            "tools": tools,
            "chunks": [{"delta_text": model_text}],
        }
        description = (
            f"Two parallel tool calls in the native {source_family} grammar, using the archived "
            f"{legacy_case_id} input and Dynamo v1 capture."
        )
        input_document = {
            "input_document": {"family": family, "mode": "unified", "model_label": family},
            "golden_document": {"family": family, "mode": "unified", "captured_with": {"golden": "v1"}},
            "cases": {
                CASE_KEY: {
                    "lifecycle": "active",
                    "scenario": CASE_KEY,
                    "description": description,
                    "policy": [f"input: {source_family}/{legacy_case_id}"],
                    "display_id": DISPLAY_ID,
                    "historical_ids": [],
                    "request": request,
                    "golden": value,
                }
            },
        }
        capture = {
            "provenance": {"status": "legacy", "captured_with": {"dynamo_v1": version}},
            "document": {"mode": "unified"},
            "changes": {
                CASE_KEY: {
                    "case_key": DISPLAY_ID,
                    "stimulus": {
                        "inline": unified_history.capture_stimulus.capture_input(request),
                        "semantic_sha256": unified_history._request_digest(
                            unified_history.capture_stimulus.capture_input(request)
                        ),
                    },
                    "observation": {"value": value},
                }
            },
            "metadata_changes": {},
            "document_overrides": {
                CASE_KEY: {
                    "capture_provenance": {
                        "captured_with": {"dynamo_v1": version},
                        "capture_origin": None,
                        "inherited_from": None,
                    }
                }
            },
        }
        family_root = Path("conformance/fixtures-unified-v2/families") / family
        input_path = family_root / "inputs_and_golden.yaml"
        capture_path = family_root / f"dynamo_v1-{version}.yaml"
        documents[root / input_path] = _merge_generated_case(
            source, input_path.relative_to("conformance/fixtures-unified-v2"), unified_history.dump_yaml(input_document)
        )
        documents[root / capture_path] = _merge_generated_case(
            source, capture_path.relative_to("conformance/fixtures-unified-v2"), unified_history.dump_yaml(capture)
        )
        # Admit the complete proposed pair before refresh writes either file:
        # retained execution proofs must still bind the merged authored inputs.
        merged_family = unified_history._validate_family(
            source / input_path.relative_to("conformance/fixtures-unified-v2"), yaml.safe_load(documents[root / input_path])
        )
        proposed_capture = yaml.safe_load(documents[root / capture_path])
        if proposed_capture.get("schema_version") == 2:
            identity = unified_history.CAPTURE_DIRECTORY_RE.fullmatch(capture_path.stem)
            unified_history._validate_complete_snapshot(
                source / capture_path.relative_to("conformance/fixtures-unified-v2"),
                proposed_capture, merged_family, identity["implementation"], identity["runtime_version"],
            )
        else:
            unified_history._validate_capture_file(
                source / capture_path.relative_to("conformance/fixtures-unified-v2"), proposed_capture,
                {family: merged_family},
            )
    return documents


def refresh(root: Path = ROOT, *, check: bool = False) -> bool:
    store_root = root / "conformance/fixtures-unified-v2"
    with unified_history._store_mutation_lock(store_root):
        with unified_archives.expanded(store_root) as source:
            documents = _generated_documents(root, source)
            current = all((source / path.relative_to(store_root)).is_file()
                          and (source / path.relative_to(store_root)).read_text() == content
                          for path, content in documents.items())
            if check or current:
                return current
            with tempfile.TemporaryDirectory(prefix="unified-v1-", dir=store_root.parent) as temporary:
                staged = Path(temporary) / store_root.name
                shutil.copytree(source, staged)
                for path, content in documents.items():
                    destination = staged / path.relative_to(store_root)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_text(content, encoding="utf-8")
                if (store_root / unified_archives.DIRECTORY).exists():
                    unified_archives.pack(staged, unified_history.CAPTURE_DIRECTORY_RE,
                                          unified_history._capture_release_sort_key)
                unified_history._load_store_unlocked(staged)
                unified_history.publish_paths_transactionally([(staged, store_root)], backup_parent=store_root.parent)
            return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="checkout containing fixture archives")
    parser.add_argument("--check", action="store_true", help="fail if generated v1 Unified fixture files are stale")
    args = parser.parse_args()
    current = refresh(args.root.resolve(), check=args.check)
    if not current:
        print("v1 Unified fixtures are stale")
        return 1
    print("v1 Unified fixtures current" if args.check else "wrote v1 Unified family fixtures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
