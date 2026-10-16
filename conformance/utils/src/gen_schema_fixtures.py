#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Append shared schema probes without altering any existing legacy stimulus."""

from pathlib import Path
from typing import Any

import yaml

import gen_unified_golden as golden
from refresh_dynamo_captures import ensure_tree
from schema_cases import (
    CONFORMANCE_CASES, legacy_schema_wire, schema_case_label,
    schema_arguments, schema_description, schema_tools,
)

ROOT = Path(__file__).resolve().parents[3]
REGISTRY = yaml.safe_load(Path(__file__).with_name("parser_families.yaml").read_text())["families"]
HEADER = "# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n# SPDX-License-Identifier: Apache-2.0\n"


def authored_legacy_cases(family: str, stream: bool) -> dict[str, dict[str, Any]]:
    cases = {}
    for probe in CONFORMANCE_CASES:
        label = schema_case_label(probe)
        prefix = "TOOLCALLING.streamv1" if stream else "TOOLCALLING.batch"
        wire = legacy_schema_wire(family, probe, golden.r_tool_arguments)
        case = {
            "description": schema_description(probe), "tools": schema_tools(probe),
            # This is authored independently; recorded implementation output never owns it.
            "golden": {"calls": [{"name": "schema_probe", "arguments": schema_arguments(family, probe)}], "normal_text": ""},
        }
        if stream:
            pieces = [wire[offset:offset + 3] for offset in range(0, len(wire), 3)]
            case["chunks"] = [{"delta_text": piece} for piece in pieces]
            case["chunks"][-1]["finish_reason"] = "stop"
        else:
            case["model_text"] = wire
        cases[f"{prefix}.{label}"] = case
    return cases


def main() -> None:
    batch = ensure_tree("fixtures-batch-v1") / "inputs"
    stream = ensure_tree("fixtures-stream-v1") / "inputs"
    for family, registration in sorted(REGISTRY.items()):
        # Harmony text and token registrations share one v1 batch grammar. Token
        # fixtures require real token IDs; author their text-path counterpart.
        if family not in ("harmony_text", "muse_glimmer"):
            doc = {"family": family, "mode": "batch", "cases": authored_legacy_cases(family, False)}
            output = batch / family / "TOOLCALLING.batch.7.schema.yaml"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(HEADER + yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=100000))
        if family == "harmony":
            continue
        doc = {"family": family, "mode": "streamv1", "cases": authored_legacy_cases(family, True)}
        if not registration.get("dynamo_v2"):
            for case in doc["cases"].values():
                case["unavailable"] = {"dynamo_v2": "No registered Dynamo v2 parser for this family."}
        output = stream / family / "TOOLCALLING.streamv1.7.schema.yaml"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(HEADER + yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=100000))
    print(f"Authored {len(CONFORMANCE_CASES)} schema probes for {len(REGISTRY)} registered families.")


if __name__ == "__main__":
    main()
