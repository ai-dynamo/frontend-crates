#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Add framing regressions to staged legacy inputs without rewriting existing cases.

Run after extract_fixtures.py and staging the toolcalling trees. Captures remain
measurements; this author only constructs native grammar inputs, never outputs.
"""
import copy
from pathlib import Path
from typing import Any

import yaml

import gen_unified_golden as unified
from schema_cases import legacy_schema_wire, schema_tools

ROOT = Path(__file__).resolve().parents[3]
TREE = ROOT / "conformance" / "toolcalling"
TOOL = [{"name": "get_weather", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}}}]


def write_new(tree: str, family: str, filename: str, mode: str, cases: dict[str, dict[str, Any]]) -> None:
    path = TREE / tree / "inputs" / family / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = yaml.safe_load(path.read_text()) if path.exists() else {"family": family, "mode": mode, "cases": {}}
    for cid, case in cases.items():
        if cid in doc["cases"] and doc["cases"][cid] != case:
            raise ValueError(f"refusing to rewrite authored input {cid} in {path}")
        doc["cases"][cid] = case
    path.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=4096))


def stream_case(case: dict[str, Any], chunks: list[str] | None = None) -> dict[str, Any]:
    return {"description": case["description"], "ref": "DSML framing and argument fidelity contract",
            "tools": copy.deepcopy(case.get("tools", [])),
            **({"unavailable": copy.deepcopy(case["unavailable"])} if "unavailable" in case else {}),
            "chunks": [{"delta_text": part} for part in (chunks or [case["model_text"]])]
                      + [{"delta_text": "", "finish_reason": "stop"}]}


def main() -> None:
    for family in ("deepseek_v3_2", "deepseek_v4"):
        for letter, (_, description, content, separator, bare) in zip("efghijk", unified._DSML_FRAMING):
            probe = {"value": "NYC", "raw": "NYC"}
            wire = legacy_schema_wire(family, probe, unified.r_tool_arguments).replace('schema_probe', 'get_weather').replace('name="value"', 'name="location"')
            if bare:
                wire = wire[wire.index(">") + 1:wire.rindex("</｜DSML｜")]
            case = {"description": description, "model_text": content + separator + wire, "tools": TOOL}
            write_new("fixtures-batch-v1", family, "TOOLCALLING.batch.dsml.yaml", "batch", {f"TOOLCALLING.batch.8.{letter}": case})
            write_new("fixtures-stream-v1", family, "TOOLCALLING.streamv1.dsml.yaml", "streamv1", {f"TOOLCALLING.streamv1.8.{letter}": stream_case(case)})
        wire = legacy_schema_wire(family, {"value": "NYC", "raw": "NYC"}, unified.r_tool_arguments).replace('schema_probe', 'get_weather').replace('name="value"', 'name="location"')
        for letter, description, prefix in [
            ("b", "Separator LFs and the calls opener arrive in separate chunks", ["I will check the weather.", "\n", "\n"]),
            ("c", "Punctuation and separator share a token-shaped chunk before a partial opener", ["I will check the weather", ".\n\n"]),
        ]:
            opener_end = wire.index(">") + 1
            chunks = prefix + [wire[:opener_end-3], wire[opener_end-3:]]
            case = {"description": description, "model_text": "".join(chunks), "tools": TOOL}
            write_new("fixtures-stream-v1", family, "TOOLCALLING.streamv1.dsml.yaml", "streamv1", {f"TOOLCALLING.streamv1.50.{letter}": stream_case(case, chunks)})

    registry = yaml.safe_load((Path(__file__).parent / "parser_families.yaml").read_text())["families"]
    for family in registry:
        # Keep the existing tool-only no-call wire contract. Native Unified
        # visible channels have their own independently authored 3-2 control.
        plain = {"description": "Plain content with trailing newlines and no tool opener", "model_text": "Plain content.\n\n", "tools": TOOL}
        value = "before\n\n" + native_opener(family) + "\n\nafter"
        probe = {"value": value, "raw": value, "schema": {"type": "string"}}
        wire = legacy_schema_wire(family, probe, unified.r_tool_arguments)
        arg = {"description": "String argument preserves two LFs and its family's opener literally", "model_text": wire, "tools": schema_tools(probe)}
        for tree, mode, prefix in [("fixtures-batch-v1", "batch", "TOOLCALLING.batch"), ("fixtures-stream-v1", "streamv1", "TOOLCALLING.streamv1")]:
            if not (TREE / tree / "inputs" / family).is_dir():
                continue
            cases = {f"{prefix}.3.a": copy.deepcopy(plain), f"{prefix}.7.x": copy.deepcopy(arg)}
            if mode == "streamv1":
                cases = {cid: stream_case(case) for cid, case in cases.items()}
                if family == "harmony":
                    for case in cases.values():
                        case["unavailable"] = {"dynamo_v2": "Text-only authored control has no tokenizer-derived IDs; use the Harmony text-path family."}
            write_new(tree, family, f"{prefix}.framing-controls.yaml", mode, cases)
        # The refresher updates existing batch-on-stream IDs, so seed only these
        # new authored IDs. Actual observations are supplied by its recorder.
        if registry[family].get("dynamo_v2") and (TREE / "fixtures-batch-v1" / "inputs" / family).is_dir():
            seed = TREE / "fixtures-batch-on-stream-v1" / family / "TOOLCALLING.batch.framing-controls.yaml"
            seed.parent.mkdir(parents=True, exist_ok=True)
            seed_capture(seed, family, ["TOOLCALLING.batch.3.a", "TOOLCALLING.batch.7.x"])
    # DSML's new batch IDs also need seeds on the v2 batch-on-stream path.
    seed = TREE / "fixtures-batch-on-stream-v1/deepseek_v4/TOOLCALLING.batch.dsml.yaml"
    seed_capture(seed, "deepseek_v4", [f"TOOLCALLING.batch.8.{letter}" for letter in "efghijk"])


def seed_capture(path: Path, family: str, case_ids: list[str]) -> None:
    doc = yaml.safe_load(path.read_text()) if path.exists() else {"family": family, "mode": "batch-on-stream", "cases": {}}
    if all(case_id in doc["cases"] for case_id in case_ids):
        return
    for case_id in case_ids:
        doc["cases"].setdefault(case_id, {})
    path.write_text(yaml.safe_dump(doc, sort_keys=False))


def native_opener(family: str) -> str:
    probe = {"value": "x", "raw": "x"}
    wire = legacy_schema_wire(family, probe, unified.r_tool_arguments)
    # Native grammar opener, including the non-angle-bracket grammars.
    if wire.startswith("["):
        return "["
    if wire.startswith("functools["):
        return "functools["
    return wire.split(">", 1)[0] + ">"


if __name__ == "__main__":
    main()
