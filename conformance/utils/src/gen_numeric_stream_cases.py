# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate native numeric stream inputs and independent raw JSON oracles."""

import argparse
import subprocess
from pathlib import Path

import yaml
import yaml_fast

import gen_unified_golden as unified
from gen_null_stream_cases import native_call
from numeric_cases import NUMERIC_VARIANTS, NUMERIC_DESCRIPTIONS, NumericLiteral, applicable, arguments_json
from refresh_dynamo_captures import V2_FAMILIES


def build_cases(family):
    cases = {}
    for _, label, schema, raw, expected in NUMERIC_VARIANTS:
        if not applicable(family, label):
            continue
        if family in {"minimax_m2", "minimax_m3", "qwen3_coder"}:
            text = native_call(family, "get_weather", {"value": NumericLiteral(raw)})
        elif family in {"harmony", "harmony_text"}:
            text = ("<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>"
                    + arguments_json(raw) + "<|call|>")
        else:
            text = unified.r_tool(family, "get_weather", "value", NumericLiteral(raw), 0)
        cases[f"TOOLCALLING.streamv1.{label}"] = {
            "description": NUMERIC_DESCRIPTIONS[label.split(".")[0]] + f" Input {raw}; expected {expected}.",
            "ref": "https://github.com/ai-dynamo/frontend-crates/pull/339",
            "tools": [{"name": "get_weather", "parameters": {"type": "object", "properties": {"value": schema}}}],
            "golden": {"calls": [{"name": "get_weather", "arguments": arguments_json(expected)}], "normal_text": ""},
            "chunks": [{"delta_text": char} for char in text] + [{"delta_text": "", "finish_reason": "stop"}],
        }
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--token-stamper", type=Path, required=True, help="Built stamp_stream_token_ids binary")
    args = parser.parse_args()
    for family in V2_FAMILIES:
        path = args.output / family / "TOOLCALLING.streamv1.7-numeric.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump({"family": family, "mode": "streamv1", "cases": dict(sorted(build_cases(family).items()))},
                                       sort_keys=False, allow_unicode=True, width=4096))
        if family in {"harmony", "harmony_text"}:
            subprocess.run([str(args.token_stamper.resolve()), "--input", str(path)], check=True)


if __name__ == "__main__":
    main()
