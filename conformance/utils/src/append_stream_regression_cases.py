# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import argparse
import difflib
from pathlib import Path

import yaml

from stream_regression_cases import (
    ENTITY_CASE,
    ENTITY_CASES,
    NESTED_UNION_CASE,
    NESTED_UNION_CASES,
    OBJECT_REFERENCE_CASE,
    OBJECT_REFERENCE_CASES,
    REASONING_CASE,
    REASONING_CASES,
    REFERENCE_TYPE_CASE,
    REFERENCE_TYPE_CASES,
    SCALAR_CASE,
    SCALAR_CASES,
    SELECTOR_CASE,
    SELECTOR_CASES,
    STRING_CASE,
    STRING_CASES,
)


def _fixture_path(inputs_root: Path, family: str, case_id: str) -> Path:
    if family in {"kimi_k3", "muse_glimmer"}:
        return inputs_root / family / "TOOLCALLING.streamv1.yaml"
    suffix = case_id.removeprefix("TOOLCALLING.streamv1.")
    group = suffix.split(".", 1)[0].split("-", 1)[0]
    return inputs_root / family / f"TOOLCALLING.streamv1.{group}.yaml"


def _append(
    inputs_root: Path,
    cases: dict[str, dict],
    case_id: str,
    *,
    replace_existing: bool = False,
    pending_updates: dict[Path, dict] | None = None,
) -> None:
    updates = {}
    for family, case in cases.items():
        path = _fixture_path(inputs_root, family, case_id)
        if pending_updates is not None and path in pending_updates:
            document = pending_updates[path]
        elif path.exists():
            document = yaml.safe_load(path.read_text())
        else:
            templates = [path.with_name("TOOLCALLING.streamv1.50.yaml")]
            templates.extend(sorted(inputs_root.joinpath(family).glob("TOOLCALLING.streamv1*.yaml")))
            template = next((candidate for candidate in templates if candidate.is_file()), None)
            if template is None:
                raise FileNotFoundError(f"no family input template exists for {family}")
            template_document = yaml.safe_load(template.read_text())
            document = {key: value for key, value in template_document.items() if key != "cases"}
            document["cases"] = {}
        if case_id in document["cases"]:
            existing = document["cases"][case_id]
            if existing == case:
                continue
            if not replace_existing:
                old_text = yaml.safe_dump(existing, sort_keys=False, allow_unicode=True, width=4096).splitlines()
                new_text = yaml.safe_dump(case, sort_keys=False, allow_unicode=True, width=4096).splitlines()
                diff = "\n".join(
                    difflib.unified_diff(
                        old_text,
                        new_text,
                        fromfile=f"{path}:{case_id}:existing",
                        tofile=f"{path}:{case_id}:requested",
                        lineterm="",
                    )
                )
                raise ValueError(f"conflicting authored case {case_id} for {family}:\n{diff}")
        document["cases"][case_id] = case
        updates[path] = document

    if pending_updates is not None:
        pending_updates.update(updates)
    else:
        for path, document in updates.items():
            path.write_text(yaml.safe_dump(document, sort_keys=False, allow_unicode=True, width=4096))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs-root", type=Path, required=True)
    parser.add_argument(
        "--replace-case-id",
        action="append",
        default=[],
        metavar="CASE_ID",
        help="Allow rewriting this exact authored case ID",
    )
    args = parser.parse_args()
    cases = {
        SCALAR_CASE: SCALAR_CASES,
        STRING_CASE: STRING_CASES,
        REASONING_CASE: REASONING_CASES,
        ENTITY_CASE: ENTITY_CASES,
        NESTED_UNION_CASE: NESTED_UNION_CASES,
        REFERENCE_TYPE_CASE: REFERENCE_TYPE_CASES,
        OBJECT_REFERENCE_CASE: OBJECT_REFERENCE_CASES,
        SELECTOR_CASE: SELECTOR_CASES,
    }
    unknown = sorted(set(args.replace_case_id) - cases.keys())
    if unknown:
        parser.error(f"unknown case IDs in --replace-case-id: {unknown}")
    pending_updates = {}
    for case_id, family_cases in cases.items():
        _append(
            args.inputs_root,
            family_cases,
            case_id,
            replace_existing=case_id in args.replace_case_id,
            pending_updates=pending_updates,
        )
    for path, document in pending_updates.items():
        path.write_text(yaml.safe_dump(document, sort_keys=False, allow_unicode=True, width=4096))


if __name__ == "__main__":
    main()
