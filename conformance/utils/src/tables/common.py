# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared contract types + canonical-JSON diff for parity (parser) impls.

Every impl wrapper (parser-mode and e2e-mode) returns ParseResult so the
harness can diff results uniformly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any



# Case descriptions use legacy dotted IDs and numbered hyphenated IDs.
CASE_DESCRIPTION_SUFFIX = r"[0-9]+(?:\.[a-z]|-[0-9]+)?"


# Curated featured-model order shared by the parser and reasoning parity tables.
# The row labels still come from fixture metadata; this list only decides which
# tool calling families lead the table and in what order.
TOP_N_TOOL_CALLING_FAMILIES = [
    "deepseek_v4",
    "gemma4",
    "glm47",
    "harmony",
    "kimi_k2",
    "minimax_m2",
    "minimax_m3",
    "qwen3_coder",
]

# Dynamo reasoning-parser family -> peer parser name. Kept here (a dependency-free
# module) so metadata readers can use them without importing the sglang
# or vllm wrapper modules, which pull in the heavy engine packages at import time.
_FAMILY_TO_SGLANG_REASONING = {
    "deepseek_r1": "deepseek-r1",
    "deepseek_v3": "deepseek-v3",
    "deepseek_v4": "deepseek-v4",
    "gemma4": "gemma4",
    "gpt_oss": "gpt-oss",
    "kimi": "kimi",
    "kimi_k25": "kimi_k2",
    "minimax_append_think": "minimax-append-think",
    "mistral": "mistral",
    "nemotron_deci": "glm45",
    "qwen3": "qwen3",
}

_FAMILY_TO_VLLM_REASONING = {
    "deepseek_r1": "deepseek_r1",
    "deepseek_v3": "deepseek_v3",
    "deepseek_v4": "deepseek_v4",
    "gemma4": "gemma4",
    "gpt_oss": "openai_gptoss",
    "granite": "granite",
    "kimi_k25": "kimi_k2",
    "mistral": "mistral",
    "minimax_append_think": "minimax_m2_append_think",
    "minimax_m3": "minimax_m3",
    "nemotron_deci": "glm45",
    "qwen3": "qwen3",
}


@dataclass
class ParseResult:
    """Uniform shape returned by every impl wrapper.

    `calls` is a list of {"name": str, "arguments": dict}.  Argument values
    are dicts (parsed from JSON) so canonical comparison ignores whitespace
    differences in the wire encoding.
    """

    calls: list[dict[str, Any]] = field(default_factory=list)
    normal_text: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "normal_text": self.normal_text,
            "error": self.error,
        }


@dataclass
class ReasoningResult:
    """Uniform shape returned by every reasoning impl wrapper."""

    reasoning_text: str | None = None
    normal_text: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "reasoning_text": self.reasoning_text,
            "normal_text": self.normal_text,
            "error": self.error,
        }


def _normalize_normal_text(v: Any) -> Any:
    """Treat any whitespace-only or None value as equivalent for comparison.

    Engines disagree on the carrier for "no narration":
      * Dynamo emits ``""`` (or ``"\\n"`` carried through between back-to-back
        tool-call envelopes — see DSML TOOLCALLING.batch.2.b).
      * vLLM emits ``None``.
      * SGLang emits ``""``.

    All three express the same semantic and should compare equal."""
    if v is None:
        return None
    if isinstance(v, str) and v.strip() == "":
        return None
    return v


def canonical(d: dict[str, Any]) -> str:
    """Canonical JSON for diffing: sorted keys, no whitespace, with empty-string ↔ None
    normalization applied to parser text fields."""
    if "normal_text" in d:
        d = {**d, "normal_text": _normalize_normal_text(d["normal_text"])}
    if "reasoning_text" in d:
        d = {**d, "reasoning_text": _normalize_normal_text(d["reasoning_text"])}
    return json.dumps(d, sort_keys=True, separators=(",", ":"))


def decode_arguments(args: Any) -> Any:
    """Decode a tool-call `arguments` field to a comparable Python value.

    Parsers surface `arguments` slightly differently: JSON-encoded
    string (Dynamo, vLLM), already-parsed dict (some SGLang paths),
    or the truncated raw string verbatim on malformed input. Decode
    JSON if it parses; otherwise return as-is so the canonical-diff
    surfaces the mismatch rather than swallowing it.
    """
    if not args:
        return {}
    if not isinstance(args, str):
        return args
    try:
        return json.loads(args)
    except (json.JSONDecodeError, TypeError):
        return args


def decode_stream_calls(
    stream_calls: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    calls = []
    for _, call in sorted(stream_calls.items()):
        if not call.get("name") and not call.get("arguments"):
            continue
        calls.append(
            {
                "name": call.get("name") or "",
                "arguments": decode_arguments(call.get("arguments") or ""),
            }
        )
    return calls
