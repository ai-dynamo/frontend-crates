# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Static viewer identities and marker metadata, independent of report models."""

import re
from tables.common import CASE_DESCRIPTION_SUFFIX
from tables.common import TOP_N_TOOL_CALLING_FAMILIES as TOP_N_FAMILIES


def read_subcase_descriptions(text: str, mode: str | None, suite: str = "TOOLCALLING") -> dict[str, str]:
    mode_pattern = re.escape(mode) if mode is not None else "batch|stream"
    pattern = re.compile(rf"\*\*`{re.escape(suite)}\.(?P<mode>{mode_pattern})\.(?P<sub>{CASE_DESCRIPTION_SUFFIX})`\*\*\s+(?P<desc>.+)")
    descriptions = {}
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        match = pattern.search(lines[index])
        if not match:
            index += 1
            continue
        sub = match.group('sub') if mode is not None else f"{match.group('mode')}.{match.group('sub')}"
        parts = [match.group('desc').strip()]
        following = index + 1
        while following < len(lines):
            line = lines[following]
            if not line.strip() or not line.startswith(" ") or pattern.search(line):
                break
            parts.append(line.strip())
            following += 1
        descriptions.setdefault(sub, " ".join(parts).rstrip("."))
        index = following
    return descriptions

STREAM_CAPTURE_EXPLANATION = "Captured from the {label} streaming parser. Streaming output differs from Dynamo parser v2 token-incremental behavior by design (text vs token streaming); see per-chunk `expected` in the fixture."

# Selection policy is presentation metadata; capture histories remain unrestricted.
VIEWER_SELECTION_POLICY = {
    "batch": {"reference": "dynamo_v1", "golden_bucket": "B", "inventory_availability": "inputs"},
    "stream": {"reference": "dynamo_v2", "fold_patch_suffix": True, "golden_bucket": "B", "inventory_defaults": {"dynamo_v1": "oldest", "vllm_rust": "oldest"}},
    "reasoning": {"reference": "dynamo_v1", "implementation_order": ["dynamo_v1", "vllm_python", "sglang_python"]},
    "unified": {"reference": "dynamo_v2", "implementation_order": ["dynamo_v2", "vllm_python", "vllm_rust"], "golden_bucket": "C", "historical_control_aliases": {"vllm_python@0.25.1": "vllm", "vllm_rust@0.25.1": "vllm_rust"}},
}

VIEWER_TAB_SPECS = (
    {"id": "tab-toolcalling-batch", "label": "Tool Calling v1 (batch data)",
     "mode": "batch", "suite": "toolcalling", "root": "toolcalling/fixtures-batch-v1"},
    {"id": "tab-toolcalling-streamv1", "label": "Tool Calling legacy stream (v1 corpus)",
     "mode": "stream", "suite": "toolcalling", "root": "toolcalling/fixtures-stream-v1"},
    {"id": "tab-reasoning-batch", "label": "Reasoning v1 (batch data)",
     "mode": "batch", "suite": "reasoning", "root": "reasoning/fixtures-v1"},
    {"id": "tab-reasoning-stream", "label": "Reasoning v1 (stream data)",
     "mode": "stream", "suite": "reasoning", "root": "reasoning/fixtures-v1"},
    {"id": "tab-unified", "label": "Unified v2 (reasoning + tools)",
     "mode": "unified", "suite": "unified", "root": "conformance/fixtures-unified-v2"},
)

UNIFIED_LEAK_MARKERS = ("<|", "|>", "<think>", "</think>", "◁", "<channel", "channel|>")

UNIFIED_FAMILY_LEAK_MARKERS = {
    "gemma4": ("thought\n",),
    "qwen3": ("<tool_call>", "</tool_call>", "<function=", "</function>",
              "<parameter=", "</parameter>"),
}


BATCH_SUB_CASE_GROUPS = [
    ("Single-call", ("1.a", "1.b", "1.c", "1.d")),
    ("Core", ("1", "3", "9", "9.a", "9.b")),
    ("Multi-call", ("2.a", "2.b", "2.c", "2.d", "2.e", "10")),
    (
        "Malformed / recovery",
        (
            "4.a",
            "4.b",
            "4.c",
            "4.d",
            "4.e",
            "4.f",
            "5.a",
            "5.b",
            "5.c",
            "5.d",
            "5.e",
            "5.f",
            "5.g",
            # streamv1.5.h (split orphan close) is streaming-only but shares the 5.*
            # Malformed band. TODO: group by parent number (5) instead of listing every
            # 5.<letter>, and dedup this list against tests/parity/toolcalling/table.py
            # (two copies of BATCH_SUB_CASE_GROUPS drift — a new sub-case must be added
            # to BOTH today).
            "5.h",
        ),
    ),
    (
        "Args",
        (
            "6.a",
            "6.b",
            "6.c",
            "7.a",
            "7.b",
            "7.c",
            "7.d",
            "7.e",
            "7.f",
            "7-4",
            "7-5",
            "7-6",
            "7-7",
            "7-8",
            "7-9",
            "7-10",
            "7-11",
            "7-12",
            "7-13",
            "7.g",
            "7.h",
            "7.i",
            "7.j",
            "7.k",
            "7.l",
            "7.m",
            "7.n",
            "7.o", "7.p", "7.q", "7.r", "7.s", "7.t", "7.u", "7.v", "7.w",
        ),
    ),
    ("Text interleaving", ("8.a", "8.b", "8.c", "8.d")),
    ("Unknown tools", ("13", "13.a", "13.c")),
    (
        "String contents",
        ("30", "30.a", "30.b", "30.c", "31", "31.a", "31.b"),
    ),
]

SPLIT_PARENT_SUBCASES = {
    # Once a taxonomy bucket has leaf cases, the matrix should render only the
    # leaves. Existing parent fixtures still carry useful expectations/reasons
    # for parser families that have not been rewritten to leaf IDs yet.
    "1": ("1.a",),
    "9": ("9.a",),
    "30": ("30.a", "30.b", "30.c"),
    "31": ("31.a", "31.b"),
    "13": ("13.a",),
}

STREAM_SUB_CASE_GROUPS = BATCH_SUB_CASE_GROUPS + [
    ("Partial-token", ("50",)),
    ("Reasoning projection", ("51.a", "51.b")),
]

SUB_CASE_GROUPS_BY_MODE = {
    "batch": BATCH_SUB_CASE_GROUPS,
    "streamv1": STREAM_SUB_CASE_GROUPS,
}


_TOOL_CALLING_LABEL_OVERRIDES = {
    "qwen3_coder": "Qwen 3 Coder / Nemotron V3‡",
}

_HIDDEN_TOOL_CALLING_FAMILIES = {"nemotron_deci", "nemotron_nano"}


_V2_TOP_N_TOOL_CALLING_FAMILIES = []
for family in TOP_N_FAMILIES:
    _V2_TOP_N_TOOL_CALLING_FAMILIES.append(family)
    if family == "harmony":
        _V2_TOP_N_TOOL_CALLING_FAMILIES.append("harmony_text")
