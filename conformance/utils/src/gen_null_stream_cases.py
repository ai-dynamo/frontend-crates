# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Author null-schema stream fixtures; observations are recorded separately."""

import argparse
import copy
from pathlib import Path

import yaml

import gen_unified_golden as unified
from null_cases import (
    MIXED_CASE_FAMILIES, MIXED_GREP_ARGS, MIXED_GREP_SCHEMA, MIXED_LABELS_ARGS, MIXED_LABELS_SCHEMA,
    NULL_VARIANTS, null_description,
)


def native_call(family, name, arguments):
    if family == "minimax_m2":
        parameters = "".join(f'<parameter name="{key}">null</parameter>' for key in arguments)
        return f'<minimax:tool_call><invoke name="{name}">{parameters}</invoke></minimax:tool_call>'
    if family == "minimax_m3":
        marker = "]<]minimax[>["
        parameters = "".join(f"{marker}<{key}>null{marker}</{key}>" for key in arguments)
        return f'{marker}<tool_call>{marker}<invoke name="{name}">{parameters}{marker}</invoke>{marker}</tool_call>'
    if family == "glm47":
        parameters = "".join(f"<arg_key>{key}</arg_key><arg_value>null</arg_value>" for key in arguments)
        return f"<tool_call>{name}{parameters}</tool_call>"
    if family == "qwen3_coder":
        parameters = "".join(f"<parameter={key}>null</parameter>" for key in arguments)
        return f"<tool_call><function={name}>{parameters}</function></tool_call>"
    key, value = next(iter(arguments.items()))
    assert len(arguments) == 1
    return unified.r_tool(family, name, key, value, 0)


def make_case(family, name, schema, arguments, description, strict=None):
    text = native_call(family, name, arguments)
    tool = {"name": name, "parameters": copy.deepcopy(schema)}
    if strict is not None:
        tool["strict"] = strict
    return {
        "description": description, "ref": "https://github.com/ai-dynamo/frontend-crates/pull/287",
        "tools": [tool],
        "golden": {"calls": [{"name": name, "arguments": arguments}], "normal_text": ""},
        "chunks": [{"delta_text": char} for char in text] + [{"delta_text": "", "finish_reason": "stop"}],
    }


def build_cases(family):
    cases = {
        f"TOOLCALLING.streamv1.{label}": make_case(
            family, "get_weather", {"type": "object", "properties": {"city": schema}},
            {"city": value}, null_description(label, detail),
        ) for _, label, schema, value, detail in NULL_VARIANTS
    }
    if family in MIXED_CASE_FAMILIES["7-4.mixed_labels"]:
        cases["TOOLCALLING.streamv1.7-4.mixed_labels"] = make_case(
            family, "set_labels", MIXED_LABELS_SCHEMA, MIXED_LABELS_ARGS,
            'PR #268: the request schema uses anyOf for nullable label, a type array for nullable note, and non-nullable string for literal. Expect {"label": null, "note": null, "literal": "null"}.',
        )
    if family in MIXED_CASE_FAMILIES["7-4.mixed_grep"]:
        cases["TOOLCALLING.streamv1.7-4.mixed_grep"] = make_case(
            family, "grep", MIXED_GREP_SCHEMA, MIXED_GREP_ARGS,
            'PR #269: the request schema requires string pattern and nullable path through anyOf, with strict: true. Expect {"pattern": "null", "path": null}. This constructed native input is a regression probe.',
            strict=True,
        )
    return cases


FAMILIES = ("deepseek_v4", "gemma4", "glm47", "kimi_k2", "kimi_k3", "muse_glimmer",
            "qwen3_coder", "minimax_m2", "minimax_m3")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for family in FAMILIES:
        path = args.output / family / "TOOLCALLING.streamv1.7-null.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump({"family": family, "mode": "streamv1", "cases": build_cases(family)},
                                       sort_keys=False, allow_unicode=True, width=4096))


if __name__ == "__main__":
    main()
