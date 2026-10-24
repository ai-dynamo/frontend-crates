#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build the browser viewer's resource index from canonical fixture owners."""
from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import os
import re
import shlex
import sys
import subprocess
import tarfile
import tempfile
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

import yaml
import yaml_fast  # noqa: F401
import capture_sources
import unified_history
import unified_archives
import generation_workers
import unified_taxonomy
from capture_stimulus import _UNAVAILABLE_CATEGORIES, select_peer_observation
import gen_unified_golden
from unified_tools import unified_tools
from viewer_provenance import generation_metadata
from impls import IMPL_SPECS, VLLM_RUST_UNAVAILABLE, UNIFIED_BATCH_IMPLS, BATCH_SOURCE_IMPL, BATCH_SOURCE_BLOCKERS, UNIFIED_DISPLAY
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


def _unified_records(root: Path = ROOT, unified: Path | None = None, *, store=None, inventory=None, document_index=None) -> list[dict[str, Any]]:
    unified = unified or root / "conformance/fixtures-unified-v2"
    records: list[dict[str, Any]] = []
    inventory = inventory if inventory is not None else unified_archives.inventory(unified)
    locations = {row["path"]: row.get("archive") for row in inventory}
    for resource in inventory:
        path = unified / resource["path"]
        archive = locations[resource["path"]]
        if not archive and (path.is_symlink() or not path.is_file()):
            raise ValueError(f"Unified fixture must be a regular file: {path.relative_to(unified)}")
        relative = path.relative_to(unified).as_posix()
        identity = unified_history.source_identity(relative)
        implementation = identity["implementation"]
        execution = None
        receipts = None
        indexed = (store.resources.get(relative) if store is not None else None)
        indexed = indexed or (document_index or {}).get(relative)
        if implementation or identity["kind"] == "execution":
            if indexed is None:
                descriptor = {**resource, **({"archive": archive} if archive else {})}
                indexed = yaml.safe_load(unified_archives.read_source(unified, descriptor))
            if implementation:
                execution = indexed.get("execution")
            else:
                receipts = sorted(indexed.get("receipts", {}))
        records.append({
            "path": (Path("conformance/fixtures-unified-v2") / relative).as_posix(),
            "sha256": resource["sha256"],
            "size": resource["size"],
            **identity,
            **({"archive": {**archive, "path": (Path("conformance/fixtures-unified-v2") / archive["path"]).as_posix()}} if archive else {}),
            **({"execution": execution} if execution else {}),
            **({"receipts": receipts} if receipts is not None else {}),
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
    unified_parser_generations = {
        family: {
            "generation": "v1+v2 split" if declaration.get("dynamo_v2") else "v1",
            "label": "v1/v2 split" if declaration.get("dynamo_v2") else "v1",
            "native_unified_v2": family in gen_unified_golden.UNIFIED_FAMILIES,
        }
        for family, declaration in parser_declarations.get("families", {}).items()
    }
    for family, declaration in parser_declarations.get("unified", {}).items():
        generation = "v1+v2 split" if family in gen_unified_golden.V1_SPLIT_FAMILIES else "v2"
        label = "v1/v2 split" if generation == "v1+v2 split" else "v2"
        registry_family = declaration.get("registry", family)
        has_native_v2 = family in gen_unified_golden.UNIFIED_FAMILIES or registry_family in gen_unified_golden.UNIFIED_FAMILIES
        native = {"generation": generation, "label": label, "native_unified_v2": has_native_v2}
        unified_parser_generations[family] = native
        if registry_family in gen_unified_golden.V1_SPLIT_FAMILIES:
            unified_parser_generations[registry_family] = {
                "generation": "v1+v2 split", "label": "v1/v2 split",
                "native_unified_v2": has_native_v2,
            }
        else:
            unified_parser_generations[registry_family] = native
    unified_parser_generations["deepseek_v41"] = {
        "generation": "v2", "label": "v2", "native_unified_v2": True,
    }
    retained_scenarios = {case["scenario"] for _name, data in capture_sources.authored_sources(root)
                          for case in yaml.safe_load(data)["cases"].values()}
    # Optional legacy imports are not a prerequisite for reading the retained
    # authored corpus. Ordinary native scenarios retain their completeness gate.
    split_scenarios = [scenario for scenario in gen_unified_golden.V1_SPLIT_SCENARIOS
                       if scenario in retained_scenarios]
    unified_scenarios = [scenario for scenario in unified_taxonomy.UNIFIED_TAX
                         if gen_unified_golden.scenario_families(scenario)
                         and (scenario not in gen_unified_golden.V1_SPLIT_SCENARIOS or scenario in split_scenarios)]
    unified_marker_families = {
        family: unified_taxonomy.marker_family(family)
        for family in sorted({*parser_declarations.get("families", {}), *parser_declarations.get("unified", {})})
    }
    unified_marker_families["gpt_oss"] = "harmony"
    unified_marker_families["harmony_text"] = "harmony"
    pins_text = (root / "conformance/utils/src/pyproject.stub.toml").read_text()
    peer_versions = {}
    for name, implementation in (("vllm", "vllm_python"), ("sglang", "sglang_python")):
        match = re.search(rf'"{name}(?:\[[^\]]*\])?==([0-9][^"]*)"', pins_text)
        if match:
            peer_versions[implementation] = match.group(1)
    return {
        "tabs": [dict(item) for item in VIEWER_TAB_SPECS],
        "selection_policy": VIEWER_SELECTION_POLICY,
        "unified_display": {
            **{key: value for key, value in UNIFIED_DISPLAY.items() if key != "group_labels"},
            "group_labels": {str(key): value for key, value in UNIFIED_DISPLAY.get("group_labels", {}).items()},
        },
        "unavailable_categories": dict(_UNAVAILABLE_CATEGORIES),
        "unified_capture_policy": {
            implementation: {
                "producer": BATCH_SOURCE_IMPL.get(implementation, implementation),
                "engine": "dynamo" if implementation.startswith("dynamo") else "vllm" if implementation.startswith("vllm") else "sglang",
                "language": "rust" if implementation.startswith("dynamo") or implementation == "vllm_rust" else "python",
                "mode": "batch" if implementation in UNIFIED_BATCH_IMPLS else "stream",
                "source_blockers": {version: reason for (key, version), reason in BATCH_SOURCE_BLOCKERS.items() if key == implementation},
            }
            for implementation in sorted({*UNIFIED_BATCH_IMPLS, *(spec.key for spec in IMPL_SPECS)})
        },
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
            "parser_generations": unified_parser_generations,
            "authored_scenarios": unified_scenarios,
            "v1_split_scenarios": split_scenarios,
            "scenario_families": {scenario: sorted(gen_unified_golden.scenario_families(scenario)) for scenario in unified_scenarios},
            "marker_families": unified_marker_families,
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
                "case": unified_taxonomy.case_fold(scenario),
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
            "family_scenarios": {family: {label: scenario for (owner, label), scenario in unified_taxonomy.FAMILY_LEGACY_CASE_LABELS.items() if owner == family}
                                 for family in sorted({family for family, _label in unified_taxonomy.FAMILY_LEGACY_CASE_LABELS})},
            "owner_families": {scenario: family for scenario in unified_taxonomy.UNIFIED_TAX
                               if (family := next((name for name in sorted({owner for owner, _label in unified_taxonomy.FAMILY_LEGACY_CASE_LABELS}, key=len, reverse=True)
                                                   if scenario.startswith(name + "_")), None)) is not None},
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
            "unified_v1_authoring": _sha256((root / "conformance/utils/src/gen_unified_v1_cases.py").read_bytes()),
            "unified_tools": _sha256((root / "conformance/utils/src/unified_tools.json").read_bytes()),
            "parser_registrations": _sha256((parser_sources / "parsers.rs").read_bytes()),
            "availability_controls": {
                "projection": _sha256(Path(__file__).read_bytes()),
                "predicate": _sha256((root / "conformance/viewer/comparison.js").read_bytes()),
                "indexer": _sha256((root / "conformance/viewer/unavailable_controls.js").read_bytes()),
            },
        },
    }


def _index_unified_families(store: unified_history.Store, candidates: list[dict]) -> None:
    """Index parser presence, retaining failures and missing expected observations."""
    unsupported = {"unsupported_family", "peer_detector_unavailable",
                   "vllm_parser_not_registered", "vllm_rust_parser_not_registered"}
    presence = {}
    captured_results = {}
    native_presence = {}
    for identity, history in sorted(store.histories.items()):
        states = {}
        captured = set()
        parser_paths = {}
        native_checkpoints = []
        checkpoints = []
        result_checkpoints = []
        needed = {capture_id for capture_id, capture in history.captures.items()
                  if any("verified_recapture" in metadata for metadata in capture["metadata_changes"].values())}
        resolved = iter(history.iter_resolved_snapshots(needed)) if needed else iter(())
        for capture_id in history.ordered_capture_ids():
            capture = history.captures[capture_id]
            for case_id, change in capture["changes"].items():
                if change == {"absent": True}:
                    states.pop(case_id, None)
                    captured.discard(case_id)
                    parser_paths.pop(case_id, None)
                elif "observation" in change:
                    observation = change["observation"]
                    states[case_id] = observation.get("unavailable", {}).get("code")
                    if "value" in observation or "error" in observation:
                        captured.add(case_id)
                    else:
                        captured.discard(case_id)
                    path = parser_paths.setdefault(case_id, {})
                    path["observation"] = observation.get("value", {}).get("parser_path")
                    path["document"] = capture.get("document", {}).get("parser_path")
            # Execution metadata is a replacement and persists through sparse observations.
            # It records the selected release's actual route, unlike today's registry.
            for case_id, metadata in capture["metadata_changes"].items():
                parser_paths.setdefault(case_id, {})["metadata"] = metadata.get("parser_path")
            for case_id, override in capture.get("document_overrides", {}).items():
                if "parser_path" in override:
                    parser_paths.setdefault(case_id, {})["override"] = override["parser_path"]
            routes = {
                path.get("metadata") or path.get("observation")
                or (path["override"] if "override" in path else path.get("document"))
                for path in parser_paths.values()
            }
            native_checkpoints.append((unified_history._capture_release_sort_key(capture["runtime_version"]),
                                       "unified" in routes and "split" not in routes))
            selected_states = dict(states)
            if capture_id in needed:
                resolved_id, snapshot = next(resolved)
                if resolved_id != capture_id:
                    raise ValueError("catalog checkpoint resolution order differs")
                for case_id, change in snapshot.items():
                    record, metadata = unified_history.materialized_record(history.family.cases[case_id], change)
                    record.update(metadata.get("record_metadata", {}))
                    selected = select_peer_observation(record, history.family.cases[case_id]["request"],
                                                       implementation=history.implementation, version=capture["runtime_version"])
                    observation = selected.get("capture_observation", {})
                    selected_states[case_id] = observation.get("unavailable", {}).get("code")
            # Missing expected results stay visible; registration failures do not.
            checkpoints.append((unified_history._capture_release_sort_key(capture["runtime_version"]),
                                not selected_states or any(code is None for code in selected_states.values())
                                or not any(code in unsupported for code in selected_states.values())))
            result_checkpoints.append((unified_history._capture_release_sort_key(capture["runtime_version"]),
                                       bool(captured)))
        presence[identity] = checkpoints
        captured_results[identity] = result_checkpoints
        native_presence[identity] = native_checkpoints
    for candidate in candidates:
        if candidate["mode"] != "unified":
            continue
        version = unified_history._capture_release_sort_key(candidate["version"])
        has_captured_results = False
        for (_family, implementation), checkpoints in captured_results.items():
            if implementation != candidate["implementation"]:
                continue
            eligible = [captured for checkpoint, captured in checkpoints if checkpoint <= version]
            if eligible and eligible[-1]:
                has_captured_results = True
                break
        candidate["has_captured_results"] = has_captured_results
        families = []
        for family in sorted(store.families):
            checkpoints = presence.get((family, candidate["implementation"]), [])
            eligible = [available for checkpoint, available in checkpoints if checkpoint <= version]
            if not eligible or eligible[-1]:
                families.append(family)
        candidate["capture_families"] = families
        if candidate["implementation"] == "dynamo_v2":
            candidate["native_unified_families"] = [
                family for family in families
                if (eligible := [native for checkpoint, native in native_presence.get((family, "dynamo_v2"), [])
                                 if checkpoint <= version]) and eligible[-1]
            ]


def _native_catalog_document_index(unified_root: Path, *, generation=None, workers=None, conversion_context=None):
    root = unified_root.resolve().parents[1]
    if conversion_context is not None:
        (root / "conformance/catalog-conversion-context.json").write_bytes(capture_sources.encoded(conversion_context))
    identities = {
        path.relative_to(unified_root).as_posix(): unified_archives._stat_identity(path.stat())
        for path in unified_root.rglob("*") if path.is_file()
    }
    workers = generation_workers.requested_workers() if workers is None else workers
    binary = capture_sources.publisher_binary(ROOT)
    with tempfile.TemporaryDirectory(prefix="unified-index-") as temporary:
        output = Path(temporary) / "index.json"
        command = [str(binary), "--root", str(root), "--output", str(output),
                   "--index-unified", "--workers", str(workers)]
        cache = os.environ.get("CONFORMANCE_CONVERSION_CACHE")
        if cache:
            command.extend(["--cache-dir", cache])
        subprocess.run(command, check=True)
        result = json.loads(output.read_bytes())
    current = {
        path.relative_to(unified_root).as_posix(): unified_archives._stat_identity(path.stat())
        for path in unified_root.rglob("*") if path.is_file()
    }
    if current != identities:
        raise ValueError("Unified sources changed during native catalog generation")
    if generation is not None:
        generation.track_sources(unified_root, identities)
    snapshots = {}
    for row in result["snapshots"]:
        row = dict(row)
        identity = (row.pop("family"), row.pop("implementation"))
        row["runtime_version"] = row["version"]
        row["version"] = unified_history._capture_release_sort_key(row["version"])
        snapshots.setdefault(identity, []).append(row)
    for rows in snapshots.values():
        rows.sort(key=lambda row: row["version"])
    return result["inventory"], result["documents"], {"families": result["families"], "snapshots": snapshots}


def _catalog_document_index(unified_root: Path, *, generation=None) -> tuple[list[dict], dict[str, dict], dict]:
    """Use native preparation for the permanent store; keep an independent fixture oracle."""
    if unified_root.resolve() == (ROOT / "conformance/fixtures-unified-v2").resolve():
        return _native_catalog_document_index(unified_root, generation=generation)
    documents = {}
    families = {}
    snapshots = {}

    def collect(name: str, data: bytes, _archive: dict | None) -> None:
        document = yaml.safe_load(data)
        if not isinstance(document, dict):
            raise ValueError(f"Unified source must contain a mapping: {name}")
        parts = PurePosixPath(name).parts
        indexed = {"schema_version": document.get("schema_version")}
        if parts[0] == "families" and len(parts) == 3 and parts[2] == "inputs_and_golden.yaml":
            indexed["active_cases"] = [case for case, item in (document.get("cases") or {}).items()
                                        if item.get("lifecycle", "active") != "retired"]
            families[parts[1]] = indexed["active_cases"]
        elif parts[0] == "families" and len(parts) == 3:
            match = unified_history.CAPTURE_DIRECTORY_RE.fullmatch(Path(parts[2]).stem)
            if match:
                implementation = match["implementation"]
                version = match["runtime_version"]
                if document.get("schema_version") == 2:
                    records = document.get("records") or {}
                    snapshots.setdefault((parts[1], implementation), []).append({
                        "version": unified_history._capture_release_sort_key(version),
                        **_record_snapshot_metadata({case: record["value"] for case, record in records.items()}),
                    })
                indexed["execution"] = document.get("execution")
                if "receipts" in document:
                    indexed["receipts"] = sorted(document.get("receipts", {}))
        elif parts[0] == "executions":
            indexed["receipts"] = sorted(document.get("receipts", {}))
        documents[name] = indexed

    source_identities = {}
    inventory = unified_archives.inventory(
        unified_root,
        member_callback=collect,
        source_callback=lambda path, identity: source_identities.__setitem__(path, identity),
    )
    if generation is not None:
        generation.track_sources(unified_root, source_identities)
    for (family, _implementation), rows in snapshots.items():
        for row in rows:
            row.update(_snapshot_code_summary(row.pop("codes"), families[family]))
        rows.sort(key=lambda row: row["version"])
    return inventory, documents, {"families": families, "snapshots": snapshots}


def _record_snapshot_metadata(records: dict) -> dict:
    routes = {record.get("document", {}).get("record_metadata", {}).get("parser_path")
              or record.get("observation", {}).get("value", {}).get("parser_path")
              or record.get("document", {}).get("parser_path") for record in records.values()}
    return {"codes": {case: _snapshot_observation_code(record) for case, record in records.items()},
            "has_results": any(any(state in record["observation"] for state in ("value", "error"))
                               for record in records.values()),
            "native": "unified" in routes and "split" not in routes,
            "route_recorded": any("parser_path" in scope for record in records.values()
                                  for scope in (record.get("document", {}).get("record_metadata", {}),
                                                record.get("observation", {}).get("value", {}),
                                                record.get("document", {})))}


def _history_dataset_sources(root: Path, unified: list[dict], metadata: dict, source_metadata: dict, *, generation=None) -> list[dict]:
    declaration = root / "conformance/history-datasets.json"
    if not declaration.is_file():
        return []
    datasets = json.loads(declaration.read_bytes())
    selected = {row["path"]: row for row in unified}
    updates = {row["path"]: row for row in unified if row["implementation"]}
    update_snapshots = copy.deepcopy(metadata["snapshots"])
    snapshots = {}
    seen = set()
    result = []
    for dataset in datasets:
        identity, prefix = dataset["id"], capture_sources.relative(dataset["root"])
        if identity in seen:
            raise ValueError("duplicate historical dataset identity")
        seen.add(identity)
        dataset_root = root / prefix
        store_root = dataset_root / capture_sources.STORE
        inventory, documents, indexed = _native_catalog_document_index(store_root, generation=generation,
            conversion_context={"dataset": identity, "metadata": source_metadata})
        resources = _unified_records(dataset_root, store_root, inventory=inventory, document_index=documents)
        for resource in resources:
            resource.update(dataset=identity, dataset_root=prefix)
            if resource["kind"] != "inputs-and-golden":
                selected[resource["path"]] = resource
        # Native indexing uses the same pre-envelope conversion cache as final
        # assembly, resolving each dataset against its own authored evidence.
        for (family, implementation), rows in indexed["snapshots"].items():
            for row in rows:
                snapshots[family, implementation, row["version"]] = row
        result.append({**dataset, "resources": resources, "checkpoints": [
            {"family": family, "implementation": implementation,
             "version": row["runtime_version"], "fragment": row["fragment"], "capture": row["capture"]}
            for (family, implementation), rows in indexed["snapshots"].items() for row in rows]})
    metadata["snapshots"] = {}
    for (family, implementation, _version), row in snapshots.items():
        metadata["snapshots"].setdefault((family, implementation), []).append(row)
    for rows in metadata["snapshots"].values():
        rows.sort(key=lambda row: row["version"])
    for identity, rows in update_snapshots.items():
        combined = {row["version"]: row for row in metadata["snapshots"].get(identity, [])}
        combined.update({row["version"]: row for row in rows})
        metadata["snapshots"][identity] = [combined[version] for version in sorted(combined)]
    selected.update(updates)
    unified[:] = [selected[name] for name in sorted(selected)]
    return result


def _snapshot_observation_code(value: dict) -> str | None:
    recapture = value.get("document", {}).get("record_metadata", {}).get("verified_recapture")
    observation = value["observation"] if recapture is None else recapture["result"].get("capture_observation", {})
    return observation.get("unavailable", {}).get("code")


def _snapshot_code_summary(codes: dict, active_cases: list[str]) -> dict:
    return {
        "record_count": len(codes),
        "code_values": sorted(set(codes.values()), key=lambda code: (code is not None, code or "")),
        "active_cases_complete": all(case in codes for case in active_cases),
    }


def _index_snapshot_families(candidates: list[dict], metadata: dict, *,
                             active_version: str | None = None, declared_native_families: frozenset[str] | set[str] = frozenset(),
                             unrouted_native_captures: dict[str, str] | None = None) -> None:
    """Index complete snapshots from the compact metadata collected at admission."""
    unsupported = {"unsupported_family", "peer_detector_unavailable",
                   "vllm_parser_not_registered", "vllm_rust_parser_not_registered"}
    active_release = unified_history._capture_release_sort_key(active_version) if active_version else None
    qualified = {family: unified_history._capture_release_sort_key(version)
                 for family, version in (unrouted_native_captures or {}).items()}
    snapshots = metadata["snapshots"]
    family_inputs = metadata["families"]
    for candidate in candidates:
        if candidate["mode"] != "unified":
            continue
        version = unified_history._capture_release_sort_key(candidate["version"])
        implementation = candidate["implementation"]
        has_captured_results = False
        families = []
        native_families = []
        for family in sorted(family_inputs):
            rows = [row for row in snapshots.get((family, implementation), []) if row["version"] <= version]
            if not rows:
                families.append(family)
                continue
            selected = rows[-1]
            codes = selected["code_values"]
            if selected["has_results"]:
                has_captured_results = True
            if (not selected["record_count"] or not selected["active_cases_complete"]
                    or None in codes or not any(code in unsupported for code in codes)):
                families.append(family)
            # Exact retained qualifications survive a newer family release; they
            # never qualify an older snapshot or override a recorded parser route.
            qualified_version = (selected["version"] == qualified.get(family)
                                 or version == active_release and selected["version"] == active_release)
            implicit_native = (implementation == "dynamo_v2" and qualified_version
                               and family in declared_native_families and selected["has_results"]
                               and selected["record_count"] and not selected["route_recorded"])
            if selected["native"] or implicit_native:
                native_families.append(family)
        candidate["has_captured_results"] = has_captured_results
        candidate["capture_families"] = families
        if implementation == "dynamo_v2":
            candidate["native_unified_families"] = native_families


def historical_catalog_fingerprint(catalog: dict) -> str:
    """Pin complete retained descriptors independently of the catalog self-revision."""
    historical = {
        "archives": catalog["sources"]["archives"],
        "candidates": [candidate for candidate in catalog["sources"]["candidates"]
                       if candidate["mode"] != "unified"],
    }
    return _sha256(json.dumps(historical, sort_keys=True, separators=(",", ":")).encode())


def retained_archive_catalog(root: Path, manifest: dict) -> dict:
    """Bind historical descriptors to retained pins without opening archive bytes."""
    catalog = json.loads((root / "conformance/viewer-catalog.json").read_text())
    content = {key: value for key, value in catalog.items() if key != "revision"}
    if catalog.get("revision") != _sha256(json.dumps(content, sort_keys=True, separators=(",", ":")).encode()):
        raise ValueError("retained archive catalog revision is invalid")
    if manifest.get("historical_catalog_sha256") != historical_catalog_fingerprint(catalog):
        raise ValueError("retained historical catalog differs from manifest fingerprint; refresh historical catalog explicitly")
    pins = [entry for entry in manifest.get("shards", []) if entry.get("format") != "unified-history"]
    expected = {"conformance/fixtures/" + entry["path"]: (entry["sha256"], entry["size"]) for entry in pins}
    if len(expected) != len(pins):
        raise ValueError("retained archive manifest has duplicate pins")
    archives = catalog["sources"]["archives"]
    actual = {entry["path"]: (entry["sha256"], entry["size"]) for entry in archives}
    if len(actual) != len(archives) or actual != expected:
        raise ValueError("retained archive descriptors differ from manifest pins; refresh historical catalog explicitly")
    for candidate in catalog["sources"]["candidates"]:
        if candidate["mode"] != "unified" and any(path not in actual for path in candidate["sources"]):
            raise ValueError("retained candidate refers to an unpinned archive")
    return catalog


def _retained_candidates(catalog: dict, *, overlay: bool) -> list[dict]:
    candidates = []
    for candidate in catalog["sources"]["candidates"]:
        if candidate["mode"] == "unified" or (candidate["mode"] == "batch-on-stream") != overlay:
            continue
        for source in candidate["sources"]:
            candidates.append({**{key: copy.deepcopy(value) for key, value in candidate.items()
                                  if key not in {"id", "sources", "archive"}}, "archive": source})
    return candidates


def build_catalog(root: Path = ROOT, *, fixtures: Path | None = None, unified: Path | None = None, manifest: Path | None = None, generation: CatalogGeneration | None = None, unified_only: bool = False,
                  include_history: bool = False, history_source: Path | None = None) -> dict[str, Any]:
    root = Path(root).resolve()
    if capture_sources.is_current(root) and fixtures is None and unified is None and manifest is None:
        with tempfile.TemporaryDirectory(prefix="current-catalog-") as temporary:
            view = Path(temporary)
            capture_sources.prepare(root, view, publisher=os.environ.get("CONFORMANCE_PUBLISHER"),
                                    include_history=include_history, history_source=history_source)
            return json.loads((view / "conformance/viewer-catalog.json").read_bytes())
    if generation is not None and generation.root != root:
        raise ValueError("catalog generation lease belongs to another root")
    manifest = json.loads((manifest or root / "conformance/fixtures-manifest.json").read_text())
    unified_shard = next((entry for entry in manifest.get("shards", [])
                          if entry.get("format") == "unified-history"), None)
    retained = retained_archive_catalog(root, manifest) if unified_only else None
    archives = copy.deepcopy(retained["sources"]["archives"]) if retained is not None else _archive_records(manifest, root, fixtures)
    unified_root = unified or root / "conformance/fixtures-unified-v2"
    inventory, document_index, snapshot_metadata = _catalog_document_index(unified_root, generation=generation)
    first_history = next((resource for resource in inventory
                          if resource["path"].startswith("families/")
                          and unified_history.CAPTURE_DIRECTORY_RE.fullmatch(Path(resource["path"]).stem)), None)
    snapshot_format = bool(first_history and document_index[first_history["path"]].get("schema_version") == 2)
    store = None
    if inventory and not snapshot_format and not (root / capture_sources.DATASETS).is_file():
        store = generation.validated_store(unified_root) if generation is not None else unified_history.load_store(unified_root)
    unified = _unified_records(root, unified_root, inventory=inventory, document_index=document_index)
    source_metadata = _source_metadata(root)
    datasets = _history_dataset_sources(root, unified, snapshot_metadata, source_metadata, generation=generation)
    if datasets:
        snapshot_format = True
    implementations = source_metadata["implementations"]
    candidates = []
    for archive in (() if retained is not None else archives):
        path = archive["path"]
        match = re.search(r"/fixtures-(batch-v1|stream-v1|batch-on-stream-v1)/([^/]+)\.tar\.gz$", path)
        reasoning = re.search(r"/reasoning/fixtures-v1/([^/]+)\.tar\.gz$", path)
        if match:
            mode = {"batch-v1": "batch", "stream-v1": "stream", "batch-on-stream-v1": "batch-on-stream"}[match.group(1)]
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
    if retained is not None:
        candidates.extend(_retained_candidates(retained, overlay=False))
    if retained is None and not any(item["implementation"] == "dynamo_v1" and item["mode"] == "reasoning" for item in candidates):
        dynamo_version = manifest.get("crates", {}).get("dynamo-parsers")
        if dynamo_version:
            candidates.append({"mode": "reasoning", "implementation": "dynamo_v1",
                               "version": dynamo_version,
                               "archive": "conformance/fixtures/reasoning/fixtures-v1/inputs.tar.gz"})
    for item in unified:
        if item["implementation"]:
            policy = source_metadata["unified_capture_policy"].get(item["implementation"], {})
            candidates.append({"mode": "unified", "implementation": item["implementation"],
                               "version": item["version"], "source": item["path"],
                               "engine": policy.get("engine"), "language": policy.get("language"),
                               "capture_mode": policy.get("mode"), "producer": policy.get("producer"),
                               "source_blocker": policy.get("source_blockers", {}).get(item["version"])})
    overlay = next((item for item in archives if item["path"] == "conformance/fixtures/toolcalling/fixtures-batch-on-stream-v1.tar.gz"), None)
    if retained is not None:
        candidates.extend(_retained_candidates(retained, overlay=True))
    elif overlay:
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
    if unified:
        if snapshot_format:
            _index_snapshot_families(
                list(unique_candidates.values()), snapshot_metadata,
                active_version=manifest["crates"]["dynamo-parsers-v2"],
                declared_native_families={family for family, declaration in source_metadata["unified_presentation"]["parser_generations"].items()
                                 if declaration["native_unified_v2"]},
                unrouted_native_captures={family: declaration["unrouted_native_capture"]
                                         for family, declaration in gen_unified_golden._MANIFEST.items()
                                         if declaration.get("native") and "unrouted_native_capture" in declaration})
        else:
            _index_unified_families(store, list(unique_candidates.values()))
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
    if datasets:
        catalog["sources"]["history_datasets"] = datasets
    generation_path = root / "conformance/capture-generation.json"
    if generation_path.is_file():
        generation = json.loads(generation_path.read_bytes())
        catalog["capture_generation"] = generation
        catalog["sources"]["lineage"] = generation["lineage"]
    catalog["revision"] = _sha256(json.dumps(catalog, sort_keys=True, separators=(",", ":")).encode())
    return catalog


def unified_projection(catalog: dict) -> dict:
    """Keep browser initialization independent of legacy archives and metadata."""
    keys = ["selection_policy", "unavailable_categories", "implementations", "unified_display", "unified_capture_policy", "markers", "unified", "unified_presentation", "folds", "unified_taxonomy", "unified_aliases", "unified_leak_markers", "documentation", "peer_versions"]
    metadata = {key: catalog["metadata"][key] for key in keys if key in catalog["metadata"]}
    legacy_candidate_identities = {}
    for candidate in catalog["sources"].get("candidates", []):
        mode = candidate["mode"]
        if mode not in {"batch", "stream", "batch-on-stream", "unified"}:
            continue
        implementation = candidate["implementation"]
        version = candidate["version"]
        version_slug = re.sub(r"[^a-zA-Z0-9]+", "-", version).strip("-")
        mode_key = "b-" if mode == "batch" else "s-" if mode == "batch-on-stream" else ""
        legacy_candidate_identities[f"{implementation}-{mode_key}{version_slug}"] = f"{implementation}@{version}"
    metadata["legacy_candidate_identities"] = legacy_candidate_identities
    metadata["tabs"] = [tab for tab in catalog["metadata"].get("tabs", []) if tab["suite"] == "unified"]
    metadata["selection_policy"] = {"unified": metadata.get("selection_policy", {}).get("unified", {})}
    metadata["folds"] = metadata.get("folds", {})
    return {**({"capture_generation": catalog["capture_generation"]} if "capture_generation" in catalog else {}),
            **({"source_revision": catalog["source_revision"]} if "source_revision" in catalog else {}),
            "schema": catalog.get("schema", 1), "revision": catalog["revision"], "snapshot": catalog.get("snapshot"),
            "metadata": metadata, "sources": {**({"original_unified": catalog["sources"]["original_unified"]} if "original_unified" in catalog["sources"] else {}),
            "archives": [], "unified": catalog["sources"]["unified"],
            "candidates": [item for item in catalog["sources"].get("candidates", []) if item["mode"] == "unified"]}}


@dataclass
class CatalogGeneration:
    root: Path
    owner: int
    active: bool = True
    read_only: bool = False
    stores: dict = field(default_factory=dict)
    sources: dict = field(default_factory=dict)

    def assert_active(self):
        if not self.active or self.owner != threading.get_ident():
            raise ValueError("catalog generation lease is inactive or belongs to another thread")

    def layout(self, root):
        return [(path.relative_to(root).as_posix(), path.lstat().st_mode,
                 str(path.resolve()) if path.is_symlink() else None)
                for path in sorted(root.rglob("*"))]

    def validated_store(self, root):
        self.assert_active()
        root = Path(root).resolve()
        if not root.is_relative_to(self.root):
            raise ValueError("validated store belongs to another root")
        inventory = unified_history._store_inventory_unlocked(root)
        layout = self.layout(root)
        if root in self.stores:
            store, expected, expected_layout = self.stores[root]
            if inventory != expected or layout != expected_layout:
                raise ValueError("validated store inventory changed during generation")
            return store
        store = unified_history._load_store_unlocked(root)
        if unified_history._store_inventory_unlocked(root) != inventory or self.layout(root) != layout:
            raise ValueError("store changed during validation")
        self.stores[root] = (store, inventory, layout)
        return store

    def track_sources(self, root, identities):
        self.assert_active()
        root = Path(root).resolve()
        if not root.is_relative_to(self.root):
            raise ValueError("catalog sources belong to another root")
        files = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
        if files != set(identities):
            raise ValueError("Unified source inventory changed during catalog generation")
        for relative, identity in identities.items():
            path = root / relative
            if path.is_symlink() or unified_archives._stat_identity(path.stat()) != identity:
                raise ValueError("Unified source changed during catalog generation")
        self.sources[root] = (dict(identities), self.layout(root))

    def validate_inventories(self):
        self.assert_active()
        for root in self.stores:
            self.validated_store(root)
        for root, (identities, expected_layout) in self.sources.items():
            if self.layout(root) != expected_layout:
                raise ValueError("Unified source layout changed during generation")
            for relative, identity in identities.items():
                path = root / relative
                if path.is_symlink() or not path.is_file() or unified_archives._stat_identity(path.stat()) != identity:
                    raise ValueError("Unified source changed during catalog generation")


@contextmanager
def catalog_generation(root: Path = ROOT, *, read_only: bool = False):
    root = Path(root).resolve()
    lease = unified_history._store_read_lock if read_only else unified_history._store_mutation_lock
    with lease(root / "conformance/fixtures-unified-v2"):
        generation = CatalogGeneration(root, threading.get_ident(), read_only=read_only)
        try:
            yield generation
        finally:
            generation.active = False
            generation.stores.clear()
            generation.sources.clear()


def refresh_catalog(root: Path = ROOT, *, destination: Path | None = None, check: bool = False,
                    fixtures: Path | None = None, unified: Path | None = None,
                    manifest: Path | None = None, generation: CatalogGeneration | None = None,
                    unified_only: bool = False, include_history: bool = False,
                    history_source: Path | None = None) -> bool:
    """Check under a reader lease; install updates under exclusive ownership."""
    root = Path(root).resolve()
    if generation is None:
        with catalog_generation(root, read_only=check) as generation:
            return refresh_catalog(root, destination=destination, check=check, fixtures=fixtures,
                                   unified=unified, manifest=manifest, generation=generation, unified_only=unified_only,
                                   include_history=include_history, history_source=history_source)
    if not generation.active or generation.root != root or generation.owner != threading.get_ident():
        raise ValueError("catalog generation lease is inactive or belongs to another root/thread")
    if generation.read_only and not check:
        raise ValueError("catalog updates require a writer lease")
    catalog = build_catalog(root, fixtures=fixtures, unified=unified, manifest=manifest, generation=generation, unified_only=unified_only,
                            include_history=include_history, history_source=history_source)
    generation.validate_inventories()
    document = json.dumps(catalog, ensure_ascii=False, indent=2) + "\n"
    destination = destination or root / "conformance/viewer-catalog.json"
    projection = destination.parent / "viewer-unified-catalog.json"
    projection_document = json.dumps(unified_projection(catalog), ensure_ascii=False, separators=(",", ":")) + "\n"
    destination = destination or root / "conformance/viewer-catalog.json"
    if check:
        return destination.is_file() and destination.read_text() == document and projection.is_file() and projection.read_text() == projection_document
    destination.parent.mkdir(parents=True, exist_ok=True)
    projection_temporary = projection.with_suffix(".json.new")
    try:
        projection_temporary.write_text(projection_document)
        os.chmod(projection_temporary, 0o644)
        unified_history._fsync_candidate_payload(projection_temporary)
        os.replace(projection_temporary, projection)
    finally:
        projection_temporary.unlink(missing_ok=True)
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
    parser.add_argument("--unified-only", action="store_true", help="reuse unchanged manifest-bound historical archive descriptors without opening archives")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="fail if the generated catalog is stale")
    mode.add_argument("--output", type=Path, help="write catalog to this path (default: generated catalog)")
    generation_workers.add_worker_argument(parser)
    parser.add_argument("--include-history", action="store_true")
    parser.add_argument("--history-source", type=Path)
    args = parser.parse_args()
    if args.history_source is not None and not args.include_history:
        parser.error("--history-source requires --include-history")
    root = args.root.resolve()
    destination = args.output or root / "conformance/viewer-catalog.json"
    with generation_workers.worker_policy(args.workers):
        current = refresh_catalog(root, destination=destination, check=args.check, unified_only=args.unified_only,
                                  include_history=args.include_history, history_source=args.history_source)
    label = destination.relative_to(root).as_posix() if destination.is_relative_to(root) else str(destination)
    if not current:
        print(f"viewer catalog is stale: {label}", file=sys.stderr)
        return 1
    if not args.check:
        guard_path = root / "conformance/capture-source-guard.json"
        revision_root = Path(json.loads(guard_path.read_bytes())["root"]) if guard_path.is_file() else root
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=revision_root, check=True, capture_output=True, text=True).stdout.strip()
        catalog_revision = json.loads(destination.read_text())["revision"]
        command = shlex.join(["python3", "conformance/utils/src/viewer_catalog.py", *sys.argv[1:]])
        provenance = generation_metadata(revision, catalog_revision, command, str(destination))
        provenance["source_checkout"] = True
        (destination.parent / "viewer-publication.json").write_text(json.dumps(provenance, sort_keys=True) + "\n")
    print(f"viewer catalog current: {label}" if args.check else f"wrote {label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
