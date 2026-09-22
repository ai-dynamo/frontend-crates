# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Plain per-family stream checkpoints; extracted readers retain their existing layout."""

import re
from pathlib import Path

import unified_history

TREE = Path("toolcalling/fixtures-stream-v2")


def capture_identity(label: str) -> tuple[str, str]:
    match = unified_history.CAPTURE_DIRECTORY_RE.fullmatch(label)
    if match is None:
        raise ValueError(f"stream capture must use a plain semantic version: {label}")
    return match["implementation"], match["runtime_version"]


def checkpoint(documents: dict[str, dict], label: str, family: str) -> dict:
    implementation, version = capture_identity(label)
    fixtures = {}
    for name, document in sorted(documents.items()):
        if not re.fullmatch(r"TOOLCALLING\.streamv2\.[A-Za-z0-9_.-]+\.yaml", name):
            raise ValueError(f"invalid stream fixture filename: {name}")
        if document.get("family") != family or document.get("mode") != "streamv2":
            raise ValueError(f"stream fixture family/mode mismatch: {name}")
        if document.get("captured_with") != {implementation: version}:
            raise ValueError(f"stream fixture capture version mismatch: {name}")
        fixtures[name] = {
            key: value for key, value in document.items()
            if key not in ("family", "mode", "captured_with")
        }
        if not isinstance(fixtures[name].get("cases"), dict):
            raise ValueError(f"stream fixture cases must be a mapping: {name}")
    if not fixtures:
        raise ValueError(f"empty stream checkpoint: {family}/{label}")
    return {"fixtures": fixtures}


def documents(value: dict, label: str, family: str) -> dict[str, dict]:
    implementation, version = capture_identity(label)
    if set(value) != {"fixtures"} or not isinstance(value["fixtures"], dict):
        raise ValueError("stream checkpoint must contain a fixtures mapping")
    if any(not isinstance(fixture, dict) for fixture in value["fixtures"].values()):
        raise ValueError("stream fixtures must be mappings")
    result = {
        name: {
            "family": family,
            "mode": "streamv2",
            "captured_with": {implementation: version},
            **fixture,
        }
        for name, fixture in value["fixtures"].items()
    }
    # Validate names and metadata before any path is written.
    if checkpoint(result, label, family) != value:
        raise ValueError("stream checkpoint repeats derived metadata")
    return result


def materialize(source: Path, relative: str, destination: Path) -> None:
    path = Path(relative)
    if (
        path.parts[:3] != (*TREE.parts, "families")
        or len(path.parts) != 5
        or not re.fullmatch(r"[a-z0-9_]+", path.parent.name)
        or path.suffix != ".yaml"
    ):
        raise ValueError(f"invalid stream checkpoint path: {relative}")
    family, label = path.parent.name, path.stem
    records = documents(unified_history.load_yaml(source), label, family)
    for name, document in records.items():
        output = destination / TREE / label / family / name
        if output.exists():
            raise ValueError(f"duplicate stream capture: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(unified_history.dump_yaml(document), encoding="utf-8")


def package(capture: Path, output_root: Path, *, prior_root: Path | None = None) -> list[Path]:
    implementation, version = capture_identity(capture.name)
    prior_dirs = []
    if prior_root is not None:
        prior_dirs = sorted(
            (
                path for path in prior_root.glob(f"{implementation}-*")
                if path.is_dir()
                and unified_history.CAPTURE_DIRECTORY_RE.fullmatch(path.name)
                and unified_history._capture_release_sort_key(path.name.partition("-")[2])
                <= unified_history._capture_release_sort_key(version)
            ),
            key=lambda path: unified_history._capture_release_sort_key(path.name.partition("-")[2]),
        )
    outputs = []
    for family in sorted(path for path in capture.iterdir() if path.is_dir()):
        if not re.fullmatch(r"[a-z0-9_]+", family.name):
            raise ValueError(f"invalid stream family: {family.name}")
        records = {
            path.name: unified_history.load_yaml(path)
            for path in sorted(family.glob("*.yaml"))
        }
        if not records:
            continue
        checkpoint(records, capture.name, family.name)
        # Compare whole case observations, not source stamps. Absent cases carry forward.
        changed = {}
        for name, record in records.items():
            prior_cases = {}
            for directory in prior_dirs:
                previous = directory / family.name / name
                if previous.is_file():
                    prior_cases.update(unified_history.load_yaml(previous)["cases"])
            cases = {
                key: case for key, case in record["cases"].items()
                if key not in prior_cases or prior_cases[key] != case
            }
            if cases:
                changed[name] = {**record, "cases": cases}
        if not changed:
            continue
        value = checkpoint(changed, capture.name, family.name)
        output = output_root / TREE / "families" / family.name / f"{capture.name}.yaml"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(unified_history.dump_yaml(value), encoding="utf-8")
        outputs.append(output)
    return outputs
