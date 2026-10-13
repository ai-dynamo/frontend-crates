#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build the browser viewer's resource index from canonical fixture owners."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import sys
import subprocess
import tarfile
import tempfile
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import yaml
import unified_history
import unified_taxonomy
import gen_unified_golden
from unified_tools import unified_tools
from impls import IMPL_SPECS, VLLM_RUST_UNAVAILABLE
from fixture_disposition import TOOLCALLING_CASE_ALIASES
from null_cases import NULL_DESCRIPTIONS, MIXED_CASE_FAMILIES
from schema_cases import schema_fold
from reasoning_metadata import CASE_GROUPS, PARSER_TO_REASONING_FAMILY, _REASONING_LABEL_OVERRIDES, _REASONING_TOP_N_APPEND, _REASONING_HIDDEN_TOOL_FAMILIES, _FAMILY_METADATA, _REASONING_MODE_METADATA
from reasoning_metadata import _DEFAULT_REASONING_MARKUP_RE, _REASONING_MARKUP_BY_FAMILY
from tables.common import TOP_N_TOOL_CALLING_FAMILIES, _FAMILY_TO_VLLM_REASONING, _FAMILY_TO_SGLANG_REASONING
from parser_metadata import family_inheritance, family_references, _SHARED_BACKEND_SHORT, _V2_STREAM_PARSER_CELLS
from viewer_metadata import UNIFIED_FAMILY_LEAK_MARKERS, UNIFIED_LEAK_MARKERS, VIEWER_TAB_SPECS, VIEWER_SELECTION_POLICY, SUB_CASE_GROUPS_BY_MODE, SPLIT_PARENT_SUBCASES, STREAM_CAPTURE_EXPLANATION
from viewer_metadata import _TOOL_CALLING_LABEL_OVERRIDES, _HIDDEN_TOOL_CALLING_FAMILIES, _V2_TOP_N_TOOL_CALLING_FAMILIES, read_subcase_descriptions

ROOT = Path(__file__).resolve().parents[3]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe_member_name(name: str) -> None:
    path = PurePosixPath(name)
    if not name or name.startswith("/") or "\\" in name or any(part in ("", ".", "..") for part in name.split("/")):
        raise ValueError(f"unsafe archive member path: {name!r}")
    if path.is_absolute():
        raise ValueError(f"absolute archive member path: {name!r}")


def _archive_inventory(path: Path, root: Path = ROOT) -> list[dict[str, Any]]:
    seen: set[str] = set()
    members: list[dict[str, Any]] = []
    with tarfile.open(fileobj=io.BytesIO(path.read_bytes()), mode="r:gz") as archive:
        for member in archive:
            _safe_member_name(member.name.rstrip("/") if member.isdir() else member.name)
            if member.name in seen:
                raise ValueError(f"duplicate member {member.name!r} in {path.relative_to(root)}")
            seen.add(member.name)
            if member.isfile():
                kind = "file"
            elif member.isdir():
                kind = "directory"
            else:
                raise ValueError(f"unsupported tar member type {member.type!r} in {path.relative_to(root)}")
            members.append({"path": member.name, "kind": kind, "size": member.size})
    return members


def _availability_document(document: dict[str, Any], reason_controls: list | None = None) -> dict[str, Any]:
    """Index capture structure without retaining parser output or input text.

    The JavaScript history resolver replays these controls to keep parser labels
    independent of the candidates currently downloaded by the browser.
    """
    cases = {}
    for case_id, record in (document.get("cases") or {}).items():
        control = {}
        expected = record.get("expected")
        if isinstance(expected, dict):
            control["expected"] = {
                implementation: ("unavailable" if "unavailable" in block else "present")
                if isinstance(block, dict) else "invalid"
                for implementation, block in expected.items()
            }
            if reason_controls is not None:
                for implementation, block in expected.items():
                    if isinstance(block, dict) and "unavailable" in block:
                        reason_controls.append((control["expected"], implementation, block["unavailable"], "expected"))
        for field in ("unavailable", "exception"):
            if field in record:
                value = record[field]
                control[field] = {implementation: True for implementation in value} if isinstance(value, dict) else True
                if field == "unavailable" and reason_controls is not None:
                    if isinstance(value, dict):
                        reason_controls.extend((control[field], implementation, reason, "state") for implementation, reason in value.items())
                    else:
                        reason_controls.append((control, field, value, "state"))
        if isinstance(record.get("chunks"), list):
            chunks = []
            for chunk in record["chunks"]:
                if not isinstance(chunk, dict):
                    chunks.append({})
                    continue
                projected = {}
                if "expected" in chunk:
                    value = chunk["expected"]
                    projected["expected"] = list(value) if isinstance(value, dict) else True
                if "normal_text" in chunk:
                    value = chunk["normal_text"]
                    projected["normal_text"] = {
                        implementation: bool(text)
                        for implementation, text in value.items()
                    } if isinstance(value, dict) else bool(value)
                chunks.append(projected)
            runs = []
            for chunk in chunks:
                if runs and runs[-1][1] == chunk:
                    runs[-1][0] += 1
                else:
                    runs.append([1, chunk])
            control["chunk_runs"] = runs
        cases[case_id] = control
    return {"family": document.get("family"), "mode": document.get("mode"), "cases": cases}


def _apply_unavailable_controls(reason_controls: list, root: Path = ROOT) -> None:
    if not reason_controls:
        return
    environment = {key: os.environ[key] for key in ("PATH", "LANG", "LC_ALL") if key in os.environ}
    result = subprocess.run(
        ["node", str(root / "conformance/viewer/unavailable_controls.js")],
        input=json.dumps([reason if isinstance(reason, str) else None for _, _, reason, _ in reason_controls]),
        text=True, capture_output=True, check=True, env=environment,
    )
    flags = json.loads(result.stdout)
    if len(flags) != len(reason_controls) or any(type(flag) is not bool for flag in flags):
        raise ValueError("invalid unavailable-control response from shared JavaScript owner")
    for (control, key, _, kind), failed in zip(reason_controls, flags):
        if failed:
            control[key] = "unavailable-error" if kind == "expected" else "parser-error"


def _archive_records(manifest: dict[str, Any], root: Path = ROOT, fixtures: Path | None = None) -> list[dict[str, Any]]:
    fixtures = fixtures or root / "conformance/fixtures"
    records: list[dict[str, Any]] = []
    reason_controls = []
    for entry in sorted(manifest.get("shards", []), key=lambda row: row["path"]):
        rel = entry["path"]
        if entry.get("format") == "unified-history":
            continue
        path = (fixtures / rel).resolve()
        if not path.is_relative_to(fixtures.resolve()) or not path.is_file():
            raise FileNotFoundError(f"missing or escaped active fixture archive: {rel}")
        data = path.read_bytes()
        digest = _sha256(data)
        if digest != entry["sha256"] or len(data) != entry["size"]:
            raise ValueError(f"fixture archive does not match manifest: {rel}")
        records.append({
            "path": (Path("conformance/fixtures") / rel).as_posix(),
            "sha256": digest,
            "size": len(data),
            "members": _archive_inventory(path, fixtures),
        })
        records[-1]["families"] = sorted({PurePosixPath(member["path"]).parent.name for member in records[-1]["members"] if member["kind"] == "file" and member["path"].endswith(".yaml")})
        if "/fixtures-batch-v1/" in rel or "/fixtures-stream-v1/" in rel or "/reasoning/fixtures-v1/" in f"/{rel}":
            controls = {}
            with tarfile.open(path, "r:gz") as archive:
                for member in archive:
                    if member.isfile() and member.name.endswith(".yaml"):
                        document = yaml.safe_load(archive.extractfile(member).read())
                        controls[member.name] = _availability_document(document, reason_controls)
            records[-1]["availability_documents"] = controls
        if path.name == "inputs.tar.gz" and "fixtures-batch-v1" in rel:
            labels = {}
            with tarfile.open(path, "r:gz") as archive:
                for member in archive:
                    if not member.isfile() or PurePosixPath(member.name).name != "TOOLCALLING.batch.yaml":
                        continue
                    document = yaml.safe_load(archive.extractfile(member).read())
                    labels[document["family"]] = document.get("model_label", document["family"])
            records[-1]["model_labels"] = labels
    _apply_unavailable_controls(reason_controls, root)
    return records


def _unified_records(root: Path = ROOT, unified: Path | None = None) -> list[dict[str, Any]]:
    unified = unified or root / "conformance/fixtures-unified-v2"
    records: list[dict[str, Any]] = []
    for path in sorted(unified.rglob("*.yaml")):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Unified fixture must be a regular file: {path.relative_to(unified)}")
        relative = path.relative_to(unified).as_posix()
        parts = PurePosixPath(relative).parts
        family = parts[1] if len(parts) >= 3 and parts[0] == "families" else None
        stem = path.stem
        match = re.fullmatch(r"(.+)-([0-9]+(?:\.[0-9]+)+(?:[-+][A-Za-z0-9.-]+)?)", stem)
        implementation, version = (match.group(1), match.group(2)) if match else (None, None)
        records.append({
            "path": (Path("conformance/fixtures-unified-v2") / relative).as_posix(),
            "sha256": _sha256(path.read_bytes()),
            "size": path.stat().st_size,
            "family": family,
            "implementation": implementation,
            "version": version,
            "kind": "history" if implementation else "inputs-and-golden",
        })
    return records


def _source_metadata(root: Path = ROOT) -> dict[str, Any]:
    parser_path = root / "conformance" / "utils" / "src" / "parser_families.yaml"
    parser_declarations = yaml.safe_load(parser_path.read_text()) or {}
    taxonomy_path = root / "conformance" / "case-taxonomy.yaml"
    case_taxonomy = yaml.safe_load(taxonomy_path.read_text()) or {}
    unified_taxonomy_path = root / "conformance" / "utils" / "src" / "unified_taxonomy.py"
    unified_labels = unified_taxonomy.display_case_labels(unified_taxonomy.UNIFIED_TAX)
    unified_order = {scenario: index for index, scenario in enumerate(sorted(unified_taxonomy.UNIFIED_TAX, key=unified_taxonomy.display_sort_key))}
    parser_sources = root / "parsers/v1/src/tool_calling"
    parser_refs = family_references(parser_sources)
    documentation_paths = [
        "conformance/README.md",
        "conformance/utils/lib/parsers/TOOLCALLING_CASES.md",
        "conformance/utils/lib/parsers/TOOLCALLING_STREAMING_V1_CASES.md",
        "conformance/utils/lib/parsers/REASONING_CASES.md",
        "conformance/utils/lib/parsers/UNIFIED_CASES.md",
    ]
    documentation = {
        "toolcalling": "conformance/utils/lib/parsers/TOOLCALLING_CASES.md",
        "stream": "conformance/utils/lib/parsers/TOOLCALLING_STREAMING_V1_CASES.md",
        "reasoning": "conformance/utils/lib/parsers/REASONING_CASES.md",
        "unified": "conformance/utils/lib/parsers/UNIFIED_CASES.md",
    }
    implementations = [
        {"key": spec.key, "legacy_key": spec.legacy_key, "engine": spec.engine,
         "language": spec.language, "label": spec.display, "modes": list(spec.modes)}
        for spec in IMPL_SPECS
    ]
    pins_text = (root / "conformance/utils/src/pyproject.stub.toml").read_text()
    peer_versions = {}
    for name, implementation in (("vllm", "vllm_python"), ("sglang", "sglang_python")):
        match = re.search(rf'"{name}(?:\[[^\]]*\])?==([0-9][^"]*)"', pins_text)
        if match:
            peer_versions[implementation] = match.group(1)
    return {
        "tabs": [dict(item) for item in VIEWER_TAB_SPECS],
        "selection_policy": VIEWER_SELECTION_POLICY,
        "reasoning_display": {
            "groups": CASE_GROUPS,
            "parser_families": PARSER_TO_REASONING_FAMILY,
            "labels": _REASONING_LABEL_OVERRIDES,
            "top_append": _REASONING_TOP_N_APPEND,
            "hidden": sorted(_REASONING_HIDDEN_TOOL_FAMILIES),
            "families": _FAMILY_METADATA,
            "modes": _REASONING_MODE_METADATA,
            "top": TOP_N_TOOL_CALLING_FAMILIES,
            "descriptions": read_subcase_descriptions((root / "conformance/utils/lib/parsers/REASONING_CASES.md").read_text(), None, "REASONING"),
            "markup_patterns": {family: pattern.pattern for family, pattern in _REASONING_MARKUP_BY_FAMILY.items()},
            "default_markup_pattern": _DEFAULT_REASONING_MARKUP_RE.pattern,
            "peer_families": {"vllm_python": _FAMILY_TO_VLLM_REASONING, "sglang_python": _FAMILY_TO_SGLANG_REASONING},
        },
        "documentation": documentation,
        "unified_leak_markers": {
            "common": list(UNIFIED_LEAK_MARKERS),
            "families": {family: list(markers) for family, markers in UNIFIED_FAMILY_LEAK_MARKERS.items()},
        },
        "implementations": implementations,
        "peer_versions": peer_versions,
        "families": parser_declarations.get("families", {}),
        "parser_sources": {"root": parser_sources.relative_to(root).as_posix(), "references": parser_refs, "inheritance": family_inheritance(parser_refs, parser_sources),
                           "shared_backend_labels": [{"variant": key[0], "sub_variant": key[1], "label": value} for key, value in _SHARED_BACKEND_SHORT.items()],
                           "stream_cells": _V2_STREAM_PARSER_CELLS},
        "markers": parser_declarations.get("markers", {}),
        "unified": parser_declarations.get("unified", {}),
        "unified_presentation": {
            "shared_tools": unified_tools(),
            "authored_scenarios": [item[0] for item in (*gen_unified_golden.CLEAN, *gen_unified_golden.EDGE)],
            "scenario_families": {item[0]: sorted(gen_unified_golden.scenario_families(item[0])) for item in (*gen_unified_golden.CLEAN, *gen_unified_golden.EDGE)},
            "marker_families": {family: unified_taxonomy.marker_family(family) for family in parser_declarations.get("unified", {})},
        },
        "legacy_taxonomy": case_taxonomy,
        "legacy_case_groups": SUB_CASE_GROUPS_BY_MODE,
        "legacy_split_parents": SPLIT_PARENT_SUBCASES,
        "legacy_aliases": TOOLCALLING_CASE_ALIASES,
        "legacy_null_groups": list(NULL_DESCRIPTIONS),
        "folds": {
            "null_descriptions": NULL_DESCRIPTIONS,
            "mixed_families": {key: sorted(value) for key, value in MIXED_CASE_FAMILIES.items()},
            "unified": {scenario: {
                "legacy_label": unified_taxonomy.legacy_case_label(scenario),
                "canonical_label": unified_taxonomy.case_label(scenario),
                "taxonomy_order": unified_taxonomy.taxonomy_sort_key(scenario),
                "schema": schema_fold(scenario),
            } for scenario in unified_taxonomy.UNIFIED_TAX},
        },
        "legacy_display": {
            "top": _V2_TOP_N_TOOL_CALLING_FAMILIES,
            "hidden": sorted(_HIDDEN_TOOL_CALLING_FAMILIES),
            "labels": _TOOL_CALLING_LABEL_OVERRIDES,
            "descriptions": {
                "batch": read_subcase_descriptions((root / documentation["toolcalling"]).read_text(), "batch"),
                "streamv1": read_subcase_descriptions((root / documentation["stream"]).read_text(), "streamv1"),
            },
        },
        "recording": {"rustUnavailable": VLLM_RUST_UNAVAILABLE, "streamExplanation": STREAM_CAPTURE_EXPLANATION},
        "divergence_notes": {family: {case_id: {key: value for key, value in note.items() if key in {"v2", "jail", "stream_vs_batch", "baseline_defect"}} for case_id, note in cases.items()} for family, cases in (yaml.safe_load((root / "conformance/toolcalling/known-divergences.yaml").read_text()) or {}).items()},
        "unified_taxonomy": {
            scenario: {"group": identity[0], "suffix": identity[1], **unified_labels[scenario],
                       "display_group": unified_taxonomy.display_tax(scenario)[0], "display_order": unified_order[scenario],
                       "section": unified_taxonomy.display_section(scenario)[0], "section_label": unified_taxonomy.display_section(scenario)[1]}
            for scenario, identity in unified_taxonomy.UNIFIED_TAX.items()
        },
        "unified_aliases": {
            "owners": {label: sorted(owners) for label, owners in unified_taxonomy._LABEL_OWNERS.items()},
            "canonical": {scenario: unified_taxonomy.case_label(scenario) for scenario in unified_taxonomy.UNIFIED_TAX},
            "families": unified_taxonomy.HISTORICAL_FAMILY_LABEL_ALIASES,
        },
        "source_hashes": {
            "parser_families": _sha256(parser_path.read_bytes()),
            "legacy_taxonomy": _sha256(taxonomy_path.read_bytes()),
            "documentation": {relative: _sha256((root / relative).read_bytes()) for relative in documentation_paths},
            "unified_taxonomy": _sha256(unified_taxonomy_path.read_bytes()),
            "viewer_metadata": _sha256((root / "conformance/utils/src/viewer_metadata.py").read_bytes()),
            "reasoning_metadata": _sha256((root / "conformance/utils/src/reasoning_metadata.py").read_bytes()),
            "implementations": _sha256((root / "conformance/utils/src/impls.py").read_bytes()),
            "peer_versions": _sha256((root / "conformance/utils/src/pyproject.stub.toml").read_bytes()),
            "parser_config": _sha256((parser_sources / "config.rs").read_bytes()),
            "parser_metadata": _sha256((root / "conformance/utils/src/parser_metadata.py").read_bytes()),
            "divergence_notes": _sha256((root / "conformance/toolcalling/known-divergences.yaml").read_bytes()),
            "unified_authoring": _sha256((root / "conformance/utils/src/gen_unified_golden.py").read_bytes()),
            "unified_tools": _sha256((root / "conformance/utils/src/unified_tools.json").read_bytes()),
            "parser_registrations": _sha256((parser_sources / "parsers.rs").read_bytes()),
            "availability_controls": {
                "projection": _sha256(Path(__file__).read_bytes()),
                "predicate": _sha256((root / "conformance/viewer/comparison.js").read_bytes()),
                "indexer": _sha256((root / "conformance/viewer/unavailable_controls.js").read_bytes()),
            },
        },
    }


def build_catalog(root: Path = ROOT, *, fixtures: Path | None = None, unified: Path | None = None, manifest: Path | None = None) -> dict[str, Any]:
    manifest = json.loads((manifest or root / "conformance/fixtures-manifest.json").read_text())
    unified_shard = next((entry for entry in manifest.get("shards", [])
                          if entry.get("format") == "unified-history"), None)
    archives = _archive_records(manifest, root, fixtures)
    unified = _unified_records(root, unified)
    source_metadata = _source_metadata(root)
    implementations = source_metadata["implementations"]
    candidates = []
    for archive in archives:
        path = archive["path"]
        match = re.search(r"/fixtures-(batch-v1|stream-v1)/([^/]+)\.tar\.gz$", path)
        reasoning = re.search(r"/reasoning/fixtures-v1/([^/]+)\.tar\.gz$", path)
        if match:
            mode = "batch" if match.group(1) == "batch-v1" else "stream"
            filename = match.group(2)
        elif reasoning:
            mode = "reasoning"
            filename = reasoning.group(1)
        else:
            continue
        impl = next((item["key"] for item in implementations
                     if filename.startswith(item["key"] + "-")), None)
        if impl:
            candidates.append({"mode": mode, "implementation": impl,
                               "version": filename[len(impl) + 1:], "archive": path})
    if not any(item["implementation"] == "dynamo_v1" and item["mode"] == "reasoning" for item in candidates):
        dynamo_version = manifest.get("crates", {}).get("dynamo-parsers")
        if dynamo_version:
            candidates.append({"mode": "reasoning", "implementation": "dynamo_v1",
                               "version": dynamo_version,
                               "archive": "conformance/fixtures/reasoning/fixtures-v1/inputs.tar.gz"})
    for item in unified:
        if item["implementation"]:
            candidates.append({"mode": "unified", "implementation": item["implementation"],
                               "version": item["version"], "source": item["path"]})
    overlay = next((item for item in archives if item["path"] == "conformance/fixtures/toolcalling/fixtures-batch-on-stream-v1.tar.gz"), None)
    if overlay:
        versions = {}
        # Only control provenance is indexed, never observations or comparisons.
        with tarfile.open((fixtures or root / "conformance/fixtures") / overlay["path"].removeprefix("conformance/fixtures/"), "r:gz") as archive:
            for member in sorted(archive.getmembers(), key=lambda item: item.name):
                if not member.isfile() or not PurePosixPath(member.name).name.startswith("TOOLCALLING.batch") or not member.name.endswith(".yaml"):
                    continue
                document = yaml.safe_load(archive.extractfile(member).read()) or {}
                for implementation, version in (document.get("captured_with") or {}).items():
                    if implementation == "dynamo_v2" or implementation not in {spec.key for spec in IMPL_SPECS if "stream" in spec.modes}:
                        continue
                    token = str(version).split()[0].lstrip("v")
                    if re.match(r"\d", token):
                        versions.setdefault(implementation, token)
        # Dynamo uses the active crate release, independently of overlay provenance.
        versions["dynamo_v2"] = manifest["crates"]["dynamo-parsers-v2"]
        for spec in IMPL_SPECS:
            if "stream" in spec.modes:
                candidates.append({"mode": "batch-on-stream", "implementation": spec.key,
                                   "version": versions.get(spec.key), "archive": overlay["path"]})
    unique_candidates = {}
    for candidate in candidates:
        identity = (candidate["implementation"], candidate["mode"], candidate["version"])
        if identity not in unique_candidates:
            unique_candidates[identity] = {**candidate, "id": "@".join(identity), "sources": []}
        unique_candidates[identity]["sources"].append(candidate.get("archive", candidate.get("source")))
    catalog = {
        "schema": 1,
        "snapshot": manifest.get("snapshot"),
        "sources": {
            "archives": archives,
            "unified": unified,
            "unified_history": unified_shard,
            "candidates": list(unique_candidates.values()),
        },
        "metadata": source_metadata,
    }
    catalog["revision"] = _sha256(json.dumps(catalog, sort_keys=True, separators=(",", ":")).encode())
    return catalog


@dataclass
class CatalogGeneration:
    root: Path
    owner: int
    active: bool = True


@contextmanager
def catalog_generation(root: Path = ROOT):
    root = Path(root).resolve()
    with unified_history._store_mutation_lock(root / "conformance/fixtures-unified-v2"):
        generation = CatalogGeneration(root, threading.get_ident())
        try:
            yield generation
        finally:
            generation.active = False


def refresh_catalog(root: Path = ROOT, *, destination: Path | None = None, check: bool = False,
                    fixtures: Path | None = None, unified: Path | None = None,
                    manifest: Path | None = None, generation: CatalogGeneration | None = None) -> bool:
    """Read and publish one catalog generation while holding the fixture writer lease."""
    root = Path(root).resolve()
    if generation is None:
        with catalog_generation(root) as generation:
            return refresh_catalog(root, destination=destination, check=check, fixtures=fixtures,
                                   unified=unified, manifest=manifest, generation=generation)
    if not generation.active or generation.root != root or generation.owner != threading.get_ident():
        raise ValueError("catalog generation lease is inactive or belongs to another root/thread")
    document = json.dumps(build_catalog(root, fixtures=fixtures, unified=unified, manifest=manifest), ensure_ascii=False, indent=2) + "\n"
    destination = destination or root / "conformance/viewer-catalog.json"
    if check:
        return destination.is_file() and destination.read_text() == document
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=destination.parent, suffix=".json", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(document)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o644)
        os.replace(temporary, destination)
        unified_history._fsync_directory(destination.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="checkout containing metadata and fixture resources")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="fail if the checked-in catalog is stale")
    mode.add_argument("--output", type=Path, help="write catalog to this path (default: checked-in catalog)")
    args = parser.parse_args()
    root = args.root.resolve()
    destination = args.output or root / "conformance/viewer-catalog.json"
    current = refresh_catalog(root, destination=destination, check=args.check)
    label = destination.relative_to(root).as_posix() if destination.is_relative_to(root) else str(destination)
    if not current:
        print(f"viewer catalog is stale: {label}", file=sys.stderr)
        return 1
    print(f"viewer catalog current: {label}" if args.check else f"wrote {label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
