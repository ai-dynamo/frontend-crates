#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build a deterministic, family-scoped job for a Unified peer capture.

Every family reuses display IDs such as ``UNIFIED.30-1``. The capture
entrypoints therefore consume stable IDs of the form
``UNIFIED.<scenario>.<family>`` and reject jobs that mix families or repeat an
ID. The request fields are copied from ``inputs_and_golden.yaml`` without
changing the authored initialization, tool schema, finish reason, or chunk
schedule.
"""

import argparse
import json
from pathlib import Path

import yaml

from capture_stimulus import validate_family_scoped_cases


def _family_path(input_root: Path, family: str) -> Path:
    path = input_root / family / "inputs_and_golden.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Unified input corpus not found: {path}")
    return path


def _case_job(family: str, case: dict) -> dict:
    scenario = case.get("scenario")
    if not isinstance(scenario, str) or not scenario:
        raise ValueError(f"{family}: case has no non-empty scenario")
    request = case.get("request")
    if not isinstance(request, dict):
        raise ValueError(f"{family}/{scenario}: case has no request")
    required = ("input", "init", "finish_reason", "tools", "chunks")
    missing = [key for key in required if key not in request]
    if missing:
        raise ValueError(f"{family}/{scenario}: request is missing {missing}")
    chunks = request["chunks"]
    if not isinstance(chunks, list):
        raise ValueError(f"{family}/{scenario}: request chunks must be a list")
    texts = []
    for index, chunk in enumerate(chunks):
        if not isinstance(chunk, dict) or not isinstance(chunk.get("delta_text"), str):
            raise ValueError(
                f"{family}/{scenario}: chunk {index} must contain string delta_text"
            )
        texts.append(chunk["delta_text"])
    return {
        "id": f"UNIFIED.{scenario}.{family}",
        "family": family,
        "scenario": scenario,
        "display_id": case.get("display_id"),
        "input": request["input"],
        "init": request["init"],
        "finish_reason": request["finish_reason"],
        "tools": request["tools"],
        "chunks": texts,
    }


def build_job(input_root: Path, family: str) -> dict:
    path = _family_path(input_root, family)
    document = yaml.safe_load(path.read_text())
    if not isinstance(document, dict):
        raise ValueError(f"{path}: expected a YAML mapping")
    input_document = document.get("input_document") or {}
    if input_document.get("family") != family:
        raise ValueError(
            f"{path}: input_document.family={input_document.get('family')!r} does not match {family!r}"
        )
    if input_document.get("mode") != "unified":
        raise ValueError(f"{path}: input_document.mode must be 'unified'")
    cases = document.get("cases")
    if not isinstance(cases, dict):
        raise ValueError(f"{path}: cases must be a mapping")
    jobs = [
        _case_job(family, case)
        for case in cases.values()
        if case.get("lifecycle", "active") != "retired"
    ]
    validate_family_scoped_cases(jobs)
    return {"family": family, "cases": jobs}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--family", required=True, help="one family directory under --input-root"
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "fixtures-unified-v2" / "families",
    )
    parser.add_argument("--output", type=Path, help="write JSON here instead of stdout")
    args = parser.parse_args()
    payload = json.dumps(build_job(args.input_root, args.family), indent=2, ensure_ascii=False)
    if args.output:
        args.output.write_text(payload + "\n")
    else:
        print(payload)


if __name__ == "__main__":
    main()
