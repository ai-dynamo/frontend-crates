#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Author TOOLCALLING 8.f-8.l and streamv1 50.b-50.c: DSML content-to-call separator framing.

The DeepSeek V3.2 and V4 encoders render an assistant tool turn as
`content + "\\n\\n" + calls`. Inputs reuse the Unified scenarios and the encoder's
exact call layout, so the legacy and Unified corpora measure the same stimuli.
"""

import copy
from pathlib import Path
from typing import Any

import yaml

import gen_unified_golden as unified
from refresh_dynamo_captures import ensure_tree

REGISTRY = yaml.safe_load(Path(__file__).with_name("parser_families.yaml").read_text())["families"]
HEADER = "# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n# SPDX-License-Identifier: Apache-2.0\n"
FAMILIES = ("deepseek_v3_2", "deepseek_v4")
TOOLS = [{
    "name": "get_weather",
    "parameters": {"type": "object", "properties": {"location": {"type": "string"}}},
}]
CALLS = [{"name": "get_weather", "arguments": {"location": "NYC"}}]
# Chunk-boundary variants exist only here: Unified treats chunking as the same behavior.
SPLITS = [
    ("The separator's LFs and the calls opener arrive in separate chunks.", ["I will check the weather.", "\n", "\n"]),
    ("Punctuation and the separator share one token-shaped chunk before a partial opener.", ["I will check the weather", ".\n\n"]),
]


def authored(family: str, description: str, normal_text: str, chunks: list[str] | None, text: str) -> dict[str, Any]:
    case: dict[str, Any] = {
        "description": description,
        "tools": copy.deepcopy(TOOLS),
        # Authored from the encoder's framing rule; recorded output never owns it.
        "golden": {"calls": copy.deepcopy(CALLS), "normal_text": normal_text},
    }
    if chunks is None:
        case["model_text"] = text
        return case
    case["chunks"] = [{"delta_text": part} for part in chunks]
    case["chunks"][-1]["finish_reason"] = "stop"
    if not REGISTRY[family].get("dynamo_v2"):
        case["unavailable"] = {"dynamo_v2": "No registered Dynamo v2 parser for this family."}
    return case


def write(path: Path, family: str, mode: str, cases: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {"family": family, "mode": mode, "cases": cases}
    path.write_text(HEADER + yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=100000))


def main() -> None:
    batch = ensure_tree("fixtures-batch-v1") / "inputs"
    stream = ensure_tree("fixtures-stream-v1") / "inputs"
    for family in FAMILIES:
        batch_cases, stream_cases = {}, {}
        for letter, (_, description, content, separator, bare) in zip("fghijkl", unified.DSML_FRAMING, strict=True):
            text = content + separator + unified.r_dsml_template_call(family, bare)
            batch_cases[f"TOOLCALLING.batch.8.{letter}"] = authored(family, description, content, None, text)
            stream_cases[f"TOOLCALLING.streamv1.8.{letter}"] = authored(family, description, content, [text], text)
        write(batch / family / "TOOLCALLING.batch.8.dsml.yaml", family, "batch", batch_cases)
        write(stream / family / "TOOLCALLING.streamv1.8.dsml.yaml", family, "streamv1", stream_cases)

        split_cases = {}
        call = unified.r_dsml_template_call(family)
        opener_end = call.index(">") + 1
        for letter, (description, prefix) in zip("bc", SPLITS, strict=True):
            chunks = [*prefix, call[:opener_end - 3], call[opener_end - 3:]]
            split_cases[f"TOOLCALLING.streamv1.50.{letter}"] = authored(
                family, description, "I will check the weather.", chunks, "".join(chunks)
            )
        write(stream / family / "TOOLCALLING.streamv1.50.dsml.yaml", family, "streamv1", split_cases)
    print(f"Authored DSML framing cases for {', '.join(FAMILIES)}.")


if __name__ == "__main__":
    main()
