#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate the REASONING.* parity table from YAML fixtures."""

from __future__ import annotations

import argparse
import datetime
import functools
import html as html_lib
import json
import os
import re
import subprocess
import zoneinfo
from pathlib import Path
from typing import Any

import yaml
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from tests.parity import common
from tests.parity.common import _FAMILY_TO_SGLANG_REASONING, _FAMILY_TO_VLLM_REASONING
from tests.parity.common import TOP_N_TOOL_CALLING_FAMILIES as TOP_N_FAMILIES
from tests.parity.common import (
    build_parity_tooltip_html,
    linkify_text_html,
    parity_cell_class,
)
from tests.parity.markup import colorize_markup

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURES = REPO_ROOT / "tests/parity/reasoning/fixtures"
PARSER_FIXTURES = REPO_ROOT / "tests/parity/toolcalling/fixtures"
REASONING_CASES_MD = REPO_ROOT / "lib/parsers/REASONING_CASES.md"
SCRIPT_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = REPO_ROOT / "tests/parity"

DisplayRow = dict[str, str | None]
_IMPL_DISPLAY = {"dynamo": "Dynamo", "vllm": "vLLM", "sglang": "SGLang"}

CASE_GROUPS = [
    (
        "Core",
        ("batch.1.a", "batch.1.b"),
    ),
    (
        "Reasoning extraction",
        (
            "batch.2.a",
            "batch.2.b",
            "batch.2.c",
            "batch.2.d",
            "batch.2.e",
            "batch.2.f",
        ),
    ),
    (
        "Tool call boundary",
        (
            "batch.3.a",
            "batch.3.b",
            "batch.3.c",
            "batch.3.d",
            "batch.3.e",
            "batch.3.f",
        ),
    ),
    ("Malformed / recovery", ("batch.4", "batch.5")),
    ("Multi-span", ("batch.6.a", "batch.6.b")),
    (
        "Core",
        ("stream.1.a", "stream.1.b"),
    ),
    ("Reasoning extraction", ("stream.2.a",)),
    ("Multi-span", ("stream.2.b", "stream.2.c")),
    ("Chunk boundaries", ("stream.3.a", "stream.3.b", "stream.3.c")),
]
_CASE_GROUP_BY_CASE = {
    case_id: label for label, case_ids in CASE_GROUPS for case_id in case_ids
}
_CASE_GROUP_INDEX_BY_CASE = {
    case_id: group_idx
    for group_idx, (_, case_ids) in enumerate(CASE_GROUPS)
    for case_id in case_ids
}
_CASE_DISPLAY_ORDER = {
    case_id: (group_idx, case_idx)
    for group_idx, (_, case_ids) in enumerate(CASE_GROUPS)
    for case_idx, case_id in enumerate(case_ids)
}

# Public tool-calling parser rows can share a reasoning implementation. Keep this
# map in user-facing parser-family names, not Rust enum names: `deepseek_v3`
# currently routes to `ReasoningParserType::DeepseekR1`, but the parity table
# should still display `deepseek_v3` because that is the public reasoning parser
# family and fixture directory.
PARSER_TO_REASONING_FAMILY = {
    "deepseek_v3": "deepseek_v3",
    "deepseek_v3_1": "deepseek_v3",
    "deepseek_v3_2": "deepseek_v3",
    "deepseek_v4": "deepseek_v4",
    "gemma4": "gemma4",
    "glm47": "nemotron_deci",
    "harmony": "gpt_oss",
    "kimi_k2": "kimi_k25",
    "minimax_m2": "minimax_append_think",
    "minimax_m3": "minimax_m3",
    "mistral": "mistral",
    "nemotron_deci": "nemotron_deci",
    "nemotron_nano": "deepseek_r1",
    "qwen3_coder": "qwen3",
}

# Row label / placement overrides keyed by tool calling family; ‡ is explained
# by the legend note (see `_legend_html`).
_REASONING_LABEL_OVERRIDES = {"nemotron_nano": "Nemotron V3‡"}
_REASONING_TOP_N_APPEND = ["nemotron_nano"]
# nemotron_deci: for older nemotron v2 models, hide to avoid confusion with nemotron v3 models
_REASONING_HIDDEN_TOOL_FAMILIES = {"nemotron_deci"}


def _model_label_html(model: str) -> str:
    """Escape a model label, styling any ‡ marker like the †/§ suffixes."""
    return html_lib.escape(model).replace("‡", '<span class="parser-suffix">‡</span>')


_FAMILY_METADATA = {
    "basic": {
        "models": ["Generic CoT models"],
        "rust_enum": "ReasoningParserType::Basic",
        "implementation": "BasicReasoningParser `<think>` / `</think>`",
        "shared_with": ["qwen3", "deepseek_v4", "nemotron_deci", "glm45"],
    },
    "qwen3": {
        "models": ["Qwen3.5", "QwQ-32B", "Qwen3-Think", "Qwen3-Coder"],
        "rust_enum": "ReasoningParserType::Qwen",
        "implementation": "BasicReasoningParser `<think>` / `</think>`",
        "shared_with": ["basic", "deepseek_v4", "nemotron_deci", "glm45"],
    },
    "deepseek_v4": {
        "models": ["DeepSeek V4 Pro", "DeepSeek V4 Flash"],
        "rust_enum": "ReasoningParserType::DeepSeekV4",
        "implementation": "BasicReasoningParser `<think>` / `</think>`",
        "shared_with": ["basic", "qwen3", "nemotron_deci", "glm45"],
        "aliases": ["deepseek-v4", "deepseekv4"],
    },
    "nemotron_deci": {
        "models": [
            "Nemotron-Super-v1 / Nemotron-Ultra-v1 / Nemotron-Deci-v1",
            "Llama-Nemotron",
            "GLM-4.5 / GLM-4.6 via glm45 alias",
        ],
        "rust_enum": "ReasoningParserType::NemotronDeci",
        "implementation": "BasicReasoningParser `<think>` / `</think>`",
        "shared_with": ["basic", "qwen3", "deepseek_v4", "glm45"],
        "aliases": ["glm45"],
    },
    "deepseek_r1": {
        "models": [
            "DeepSeek R1",
            "DeepSeek V3.x aliases",
            "Nemotron force-reasoning aliases",
        ],
        "rust_enum": "ReasoningParserType::DeepseekR1",
        "implementation": (
            "BasicReasoningParser `<think>` / `</think>`, force_reasoning=true"
        ),
        "shared_with": [
            "deepseek_v3",
            "step3",
            "nemotron_nano",
            "nemotron3",
            "nemotron_v3",
        ],
    },
    "deepseek_v3": {
        "models": ["DeepSeek V3", "DeepSeek V3.1", "DeepSeek V3.2"],
        "rust_enum": "ReasoningParserType::DeepseekR1",
        "implementation": (
            "BasicReasoningParser `<think>` / `</think>`, force_reasoning=true"
        ),
        "shared_with": [
            "deepseek_r1",
            "step3",
            "nemotron_nano",
            "nemotron3",
            "nemotron_v3",
        ],
        "aliases": ["deepseek_v3_1", "deepseek_v3_2"],
    },
    "kimi": {
        "models": ["Kimi K2 Instruct / Thinking using Unicode think delimiters"],
        "rust_enum": "ReasoningParserType::Kimi",
        "implementation": "BasicReasoningParser `◁think▷` / `◁/think▷`",
    },
    "kimi_k25": {
        "models": ["Kimi K2.5 / Kimi K2.6 style `<think>` force-reasoning models"],
        "rust_enum": "ReasoningParserType::KimiK25",
        "implementation": (
            "BasicReasoningParser `<think>` / `</think>`, "
            "force_reasoning=true, Kimi tool-section exit"
        ),
    },
    "mistral": {
        "models": ["Magistral"],
        "rust_enum": "ReasoningParserType::Mistral",
        "implementation": (
            "BasicReasoningParser `[THINK]` / `[/THINK]`, force_reasoning=true"
        ),
    },
    "granite": {
        "models": ["IBM Granite 3.x", "IBM Granite 3.2 language models"],
        "rust_enum": "ReasoningParserType::Granite",
        "implementation": "GraniteReasoningParser",
    },
    "gpt_oss": {
        "models": ["gpt-oss-20b", "gpt-oss-120b"],
        "rust_enum": "ReasoningParserType::GptOss",
        "implementation": "GptOssReasoningParser / Harmony StreamableParser",
    },
    "minimax_append_think": {
        "models": ["MiniMax M2", "MiniMax M2.1"],
        "rust_enum": "ReasoningParserType::MiniMaxAppendThink",
        "implementation": "MiniMaxAppendThinkParser",
    },
    "minimax_m3": {
        "models": ["MiniMax M3"],
        "rust_enum": "ReasoningParserType::MiniMaxM3",
        "implementation": (
            "BasicReasoningParser `<mm:think>` / `</mm:think>`, "
            "dangling-end recovery"
        ),
        "aliases": ["minimax-m3"],
    },
    "gemma4": {
        "models": ["Google Gemma 4 thinking models"],
        "rust_enum": "ReasoningParserType::Gemma4",
        "implementation": "Gemma4ReasoningParser",
        "aliases": ["gemma-4"],
    },
}

_REASONING_MODE_METADATA = {
    "basic": {
        "label": "explicit markers",
        "control": "mostly static",
        "summary": (
            "Reasoning starts only after an opening `<think>` marker or a "
            "prompt-injected start state."
        ),
        "static": [
            "BasicReasoningParser `<think>` / `</think>`",
            "force_reasoning=false",
            "stream_reasoning=true",
        ],
    },
    "qwen3": {
        "label": "explicit markers",
        "control": "frontend-tunable",
        "summary": (
            "Qwen-style reasoning uses explicit `<think>` markers; templates "
            "may inject the opening marker."
        ),
        "static": [
            "BasicReasoningParser `<think>` / `</think>`",
            "force_reasoning=false",
            "stream_reasoning=true",
        ],
    },
    "deepseek_v4": {
        "label": "explicit markers",
        "control": "frontend-tunable",
        "summary": (
            "DeepSeek V4 uses `<think>` markers, but the prompt formatter "
            "controls whether thinking is enabled."
        ),
        "static": [
            "BasicReasoningParser `<think>` / `</think>`",
            "force_reasoning=false",
            "stream_reasoning=true",
        ],
    },
    "nemotron_deci": {
        "label": "explicit markers",
        "control": "frontend-tunable",
        "summary": "GLM/Nemotron-Deci style parsing uses explicit `<think>` markers.",
        "static": [
            "BasicReasoningParser `<think>` / `</think>`",
            "force_reasoning=false",
            "stream_reasoning=true",
        ],
    },
    "deepseek_r1": {
        "label": "force reasoning",
        "control": "frontend-tunable",
        "summary": (
            "Generation may begin already inside reasoning; marker-free text "
            "can be reasoning until an end marker."
        ),
        "static": [
            "BasicReasoningParser `<think>` / `</think>`",
            "force_reasoning=true",
            "stream_reasoning=true",
        ],
    },
    "deepseek_v3": {
        "label": "force reasoning",
        "control": "frontend-tunable",
        "summary": "DeepSeek V3.x routes to the R1 force-reasoning implementation in Dynamo.",
        "static": [
            "Rust enum: ReasoningParserType::DeepseekR1",
            "BasicReasoningParser `<think>` / `</think>`",
            "force_reasoning=true",
            "stream_reasoning=true",
        ],
    },
    "kimi": {
        "label": "explicit unicode",
        "control": "mostly static",
        "summary": "Kimi legacy reasoning uses Unicode delimiters instead of `<think>`.",
        "static": [
            "BasicReasoningParser `◁think▷` / `◁/think▷`",
            "force_reasoning=false",
            "stream_reasoning=true",
        ],
    },
    "kimi_k25": {
        "label": "force reasoning",
        "control": "frontend-tunable",
        "summary": (
            "Kimi K2.5 can start in reasoning and uses the tool call section "
            "marker as a reasoning escape boundary."
        ),
        "static": [
            "BasicReasoningParser `<think>` / `</think>`",
            "force_reasoning=true",
            "stream_reasoning=true",
            "tool_start_token=`<|tool_calls_section_begin|>`",
        ],
    },
    "mistral": {
        "label": "force reasoning",
        "control": "mostly static",
        "summary": (
            "Mistral/Magistral reasoning uses bracket markers and starts in "
            "force-reasoning mode."
        ),
        "static": [
            "BasicReasoningParser `[THINK]` / `[/THINK]`",
            "force_reasoning=true",
            "stream_reasoning=true",
        ],
    },
    "granite": {
        "label": "phrase markers",
        "control": "mostly static",
        "summary": "Granite reasoning is split by natural-language phrase markers.",
        "static": [
            "GraniteReasoningParser",
            "`Here is my thought process:` / `Here is my response:`",
        ],
    },
    "gpt_oss": {
        "label": "Harmony channels",
        "control": "mostly static",
        "summary": (
            "GPT-OSS reasoning is the Harmony `analysis` channel, parsed by "
            "the Harmony stream parser."
        ),
        "static": [
            "GptOssReasoningParser / Harmony StreamableParser",
            "requires special tokens to remain visible in decoded text",
        ],
    },
    "minimax_append_think": {
        "label": "append-think",
        "control": "mostly static",
        "summary": (
            "MiniMax append-think has its own contract instead of normal "
            "reasoning extraction."
        ),
        "static": [
            "MiniMaxAppendThinkParser",
            "parser-specific content wrapper behavior",
        ],
    },
    "minimax_m3": {
        "label": "explicit M3 markers",
        "control": "frontend-tunable",
        "summary": (
            "MiniMax M3 reasoning uses explicit `<mm:think>` markers, with "
            "dangling-end recovery for prompt-prefilled reasoning starts."
        ),
        "static": [
            "BasicReasoningParser `<mm:think>` / `</mm:think>`",
            "force_reasoning=false",
            "stream_reasoning=true",
            "recover_dangling_end=true",
        ],
    },
    "gemma4": {
        "label": "Gemma channels",
        "control": "frontend-tunable",
        "summary": "Gemma 4 reasoning is wrapped by channel markers with a `thought` role label.",
        "static": [
            "Gemma4ReasoningParser",
            "requires special tokens to remain visible in decoded text",
        ],
    },
}


def _make_jinja_env() -> Environment:
    env = Environment(
        loader=FileSystemLoader(TEMPLATE_DIR),
        trim_blocks=False,
        lstrip_blocks=True,
        undefined=StrictUndefined,
    )
    # Same shared CSS/JS the v2 table and toolcalling parity page inline.
    assets = TEMPLATE_DIR / "assets"
    env.globals["conformance_css"] = (assets / "conformance.css").read_text(encoding="utf-8")
    env.globals["conformance_js"] = (assets / "conformance.js").read_text(encoding="utf-8")
    return env


def _commit_sha() -> str | None:
    try:
        return (
            subprocess.check_output(
                ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def _case_sort_key(case_id: str) -> tuple[int, int, int, str]:
    doc_id = case_id.replace("REASONING.", "", 1)
    order = _CASE_DISPLAY_ORDER.get(doc_id)
    if order is not None:
        group_idx, case_idx = order
        return (0, group_idx, case_idx, "")
    parts = doc_id.split(".")
    mode = 0 if parts[0] == "batch" else 1
    top = int(parts[1])
    sub = parts[2] if len(parts) > 2 else ""
    return (1, mode, top, sub)


def _normalize_text(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, str) and v.strip() == "":
        return None
    return v


def _explanation(block: object) -> str | None:
    """The intentional-divergence note on an expected block / case. `explanation` is the
    current key; `reason` is the legacy spelling still present in older fixtures. Read
    both (explanation wins); new fixtures/captures write `explanation`."""
    if not isinstance(block, dict):
        return None
    v = block.get("explanation")
    return v if v is not None else block.get("reason")


def _canonical(d: dict[str, Any]) -> str:
    d = {
        **d,
        "normal_text": _normalize_text(d.get("normal_text")),
        "reasoning_text": _normalize_text(d.get("reasoning_text")),
    }
    d.pop("reason", None)
    d.pop("explanation", None)
    return json.dumps(d, sort_keys=True, separators=(",", ":"))


def _reasoning_cmp_sig(block: Any) -> str:
    """Canonical signature of a reasoning candidate's output; equal signatures =
    same output. Mirrors generate_conformance_table._candidate_sig but over the
    reasoning {reasoning_text, normal_text} block shape."""
    if block is None or (isinstance(block, dict) and "unavailable" in block):
        return "na"
    if isinstance(block, dict) and "error" in block:
        return f"err:{block.get('error')}"
    return _canonical(block)


def _reasoning_cmp_json(case: dict[str, Any] | None, family: str | None) -> str:
    """Per-cell `data-cmp` payload for the reasoning panels, mirroring
    generate_conformance_table._candidate_cmp_json but keyed on the legacy
    reasoning impl keys (dynamo/vllm/sglang): {impl: {"sig": int, "leak": 0|1,
    "na": 0|1}}. `sig` is a per-cell group id (impls sharing an id have identical
    output). Only impls that appear in this cell's `expected` are included."""
    if not isinstance(case, dict) or "expected" not in case:
        return ""
    expected = case.get("expected", {})
    if not isinstance(expected, dict):
        return ""
    raw: dict[str, dict[str, Any]] = {}
    for impl in ("dynamo", "vllm", "sglang"):
        if impl not in expected:
            continue
        block = expected.get(impl)
        sig = _reasoning_cmp_sig(block)
        raw[impl] = {
            "s": sig,
            "leak": 1 if (isinstance(block, dict) and _block_leak_reason(block, family)) else 0,
            # na = None/unavailable; still shown, but excluded from the diff count.
            "na": 1 if sig == "na" else 0,
        }
    if not raw:
        return ""
    ids: dict[str, int] = {}
    out = {
        impl: {"sig": ids.setdefault(e["s"], len(ids)), "leak": e["leak"], "na": e["na"]}
        for impl, e in raw.items()
    }
    return html_lib.escape(json.dumps(out, separators=(",", ":")), quote=True)


_DEFAULT_REASONING_MARKUP_RE = re.compile(r"</?think>")
_NO_AUTO_REASONING_MARKUP_RE = re.compile(r"(?!)")
_REASONING_MARKUP_BY_FAMILY: dict[str, re.Pattern[str]] = {
    "deepseek_r1": re.compile(r"</?think>"),
    "deepseek_v3": re.compile(r"</?think>"),
    "deepseek_v4": re.compile(r"</?think>"),
    "gemma4": re.compile(r"<\|channel>|<channel\|>"),
    "gpt_oss": re.compile(r"<\|(?:start|end|return|channel|message)\|>"),
    "granite": re.compile(
        r"Here is my thought process:|Here is my response:|Here's my response:"
    ),
    "kimi": re.compile(r"◁/?think▷"),
    "kimi_k25": re.compile(r"</?think>"),
    "minimax_append_think": _NO_AUTO_REASONING_MARKUP_RE,
    "mistral": re.compile(r"\[/?THINK\]"),
    "nemotron_deci": re.compile(r"</?think>"),
    "qwen3": re.compile(r"</?think>"),
}


def _reasoning_markup_re(family: str | None) -> re.Pattern[str]:
    return _REASONING_MARKUP_BY_FAMILY.get(family or "", _DEFAULT_REASONING_MARKUP_RE)


def _is_gpt_oss_tool_handoff(family: str | None, field: str, value: str) -> bool:
    # Both `commentary to=functions.X` and `analysis to=functions.X` are
    # tool-call handoffs after PR #10366: the recipient is the signal, not
    # the channel label. Either lands in normal_text as the jail's input.
    return (
        family == "gpt_oss"
        and field == "normal_text"
        and (
            "<|channel|>commentary to=functions." in value
            or "<|channel|>analysis to=functions." in value
        )
        and "<|call|>" in value
    )


def _block_leak_reason(block: dict[str, Any], family: str | None) -> str | None:
    if not isinstance(block, dict):
        return None
    marker_re = _reasoning_markup_re(family)
    for field in ("reasoning_text", "normal_text"):
        value = block.get(field)
        if (
            isinstance(value, str)
            and marker_re.search(value)
            and not _is_gpt_oss_tool_handoff(family, field, value)
        ):
            return str(
                _explanation(block)
                or "Dynamo leaks reasoning markup or final-answer text."
            )
    return None


def _dynamo_leak_reason(expected: dict[str, Any], family: str | None) -> str | None:
    dynamo = expected.get("dynamo", {})
    if not isinstance(dynamo, dict):
        return None
    return _block_leak_reason(dynamo, family)


def _has_dynamo_leak(case: dict[str, Any], family: str | None) -> bool:
    if case.get("dynamo_leak"):
        return True
    expected = case.get("expected")
    return (
        isinstance(expected, dict) and _dynamo_leak_reason(expected, family) is not None
    )


def _overview_status(case: dict[str, Any] | None, family: str | None, impl: str) -> str:
    if case is None:
        return "na"
    if "expected" not in case:
        return "problem" if impl in _python_exception_impls(case, family) else "na"
    block = case.get("expected", {}).get(impl)
    if not isinstance(block, dict) or "unavailable" in block:
        return "na"
    if "error" in block or _block_leak_reason(block, family):
        return "problem"
    return "ok"


def _overview_status_attrs(case: dict[str, Any] | None, family: str | None) -> str:
    return " ".join(
        f'data-status-{impl}="{_overview_status(case, family, impl)}"'
        for impl in ("dynamo", "vllm", "sglang")
    )


def _canonical_reasoning_output(block: object) -> str | None:
    if not isinstance(block, dict) or "unavailable" in block or "error" in block:
        return None
    return _canonical(block)


def _selected_parity_marker(
    case: dict[str, Any] | None,
    family: str | None,
    impl: str,
) -> str | None:
    if case is None or "expected" not in case:
        return None
    expected = case.get("expected", {})
    outputs = {
        impl: _canonical_reasoning_output(expected.get(impl))
        for impl in ("dynamo", "vllm", "sglang")
    }
    if any(value is None for value in outputs.values()):
        return None
    if outputs["dynamo"] == outputs["vllm"] == outputs["sglang"]:
        return "="
    selected = outputs[impl]
    peers = (
        ("dynamo", "D"),
        ("vllm", "V"),
        ("sglang", "S"),
    )
    marker = "".join(
        letter for peer, letter in peers if peer != impl and outputs[peer] != selected
    )
    return marker or "="


def _selected_parity_suffix(
    case: dict[str, Any] | None,
    family: str | None,
    impl: str,
) -> str:
    if case is None or "expected" not in case:
        return ""
    block = case.get("expected", {}).get(impl)
    if isinstance(block, dict) and _block_leak_reason(block, family):
        return "↯"
    return ""


def _parity_marker(
    case: dict[str, Any] | None,
    family: str | None,
    impl: str,
) -> str:
    marker = _selected_parity_marker(case, family, impl)
    if marker is None:
        return _parser_marker(case, family, impl)
    return _selected_parity_suffix(case, family, impl) + marker


def _parser_marker(case: dict[str, Any] | None, family: str | None, impl: str) -> str:
    if case is None:
        return "—"
    if "expected" not in case:
        if impl in _python_exception_impls(case, family):
            return "✗"
        return "n/a"
    expected = case.get("expected", {})
    block = expected.get(impl)
    if not isinstance(block, dict) or "unavailable" in block:
        return "n/a"
    if "error" in block:
        return "✗"
    if _block_leak_reason(block, family):
        return "↯"
    if impl == "dynamo":
        peers = (expected.get("vllm"), expected.get("sglang"))
        if all(isinstance(peer, dict) and "unavailable" in peer for peer in peers):
            return "·"
    return ""


def _parser_marker_attrs(case: dict[str, Any] | None, family: str | None) -> str:
    attrs = [
        f'data-marker-{impl}="{html_lib.escape(_parser_marker(case, family, impl))}"'
        for impl in ("dynamo", "vllm", "sglang")
    ]
    attrs.extend(
        f'data-marker-parity-{impl}="{html_lib.escape(_parity_marker(case, family, impl))}"'
        for impl in ("dynamo", "vllm", "sglang")
    )
    return " ".join(attrs)


def _load() -> tuple[dict[str, dict[str, Any]], list[str], dict[tuple[str, str], Path]]:
    rows: dict[str, dict[str, Any]] = {}
    columns = set()
    refs = {}
    for fp in sorted(FIXTURES.glob("*/REASONING.*.yaml")):
        doc = yaml.safe_load(fp.read_text())
        family = doc["family"]
        row = rows.setdefault(
            family,
            {
                "family": family,
                "model_label": doc.get("model_label", family),
                "cases": {},
                "captured_with": {},
            },
        )
        # Merge the fixture's captured-peer versions into the family row. Only
        # files whose peer output was reproduced by the container carry this
        # block (stamped by src/capture_reasoning.py), so it is a real provenance
        # claim -- the batch and stream files for one family may differ.
        captured = doc.get("captured_with")
        if isinstance(captured, dict):
            row["captured_with"].update(captured)
        for case_id, case in doc["cases"].items():
            columns.add(case_id)
            row["cases"][case_id] = case
            refs[(family, case_id)] = fp
    return rows, sorted(columns, key=_case_sort_key), refs


def _load_parser_labels() -> dict[str, str]:
    labels = {}
    for fp in sorted(PARSER_FIXTURES.glob("*/TOOLCALLING.batch.yaml")):
        doc = yaml.safe_load(fp.read_text())
        labels[doc["family"]] = doc.get("model_label", doc["family"])
    return labels


def _build_display_groups(
    rows: dict[str, dict[str, Any]],
) -> tuple[list[DisplayRow], list[DisplayRow], list[DisplayRow]]:
    parser_labels = _load_parser_labels()
    parser_families = set(parser_labels)

    def make_row(tool_family: str) -> dict[str, str | None]:
        return {
            "model_label": _REASONING_LABEL_OVERRIDES.get(
                tool_family, parser_labels.get(tool_family, tool_family)
            ),
            "tool_family": tool_family,
            "reasoning_family": PARSER_TO_REASONING_FAMILY.get(tool_family),
        }

    def has_reasoning_fixture(tool_family: str) -> bool:
        reasoning_family = PARSER_TO_REASONING_FAMILY.get(tool_family)
        return reasoning_family in rows

    def displayable(tool_family: str) -> bool:
        return (
            tool_family in parser_families
            and has_reasoning_fixture(tool_family)
            and tool_family not in _REASONING_HIDDEN_TOOL_FAMILIES
        )

    top_n_families = [
        tool_family for tool_family in TOP_N_FAMILIES if displayable(tool_family)
    ]
    top_n_families += [
        tool_family
        for tool_family in _REASONING_TOP_N_APPEND
        if displayable(tool_family) and tool_family not in top_n_families
    ]
    top_n = [make_row(tool_family) for tool_family in top_n_families]

    excluded = (
        set(TOP_N_FAMILIES)
        | set(_REASONING_TOP_N_APPEND)
        | _REASONING_HIDDEN_TOOL_FAMILIES
    )
    other_tool_families = sorted(
        (
            tool_family
            for tool_family in parser_families - excluded
            if has_reasoning_fixture(tool_family)
        ),
        key=lambda family: parser_labels.get(family, family).lower(),
    )
    others = [make_row(tool_family) for tool_family in other_tool_families]

    mapped_reasoning = {
        family for family in PARSER_TO_REASONING_FAMILY.values() if family in rows
    }
    reasoning_only = [
        {
            "model_label": rows[family]["model_label"],
            "tool_family": None,
            "reasoning_family": family,
        }
        for family in sorted(set(rows) - mapped_reasoning)
    ]
    return top_n, others, reasoning_only


def _derive_no_peer_sets(rows: dict[str, dict[str, Any]]) -> tuple[set[str], set[str]]:
    """Reasoning families where every expected case marks the peer unavailable."""

    def all_unavailable(cases: dict[str, dict[str, Any]], impl: str) -> bool:
        expected_cases = [
            case for case in cases.values() if isinstance(case.get("expected"), dict)
        ]
        if not expected_cases:
            return False
        for case in expected_cases:
            block = case.get("expected", {}).get(impl)
            if not isinstance(block, dict) or "unavailable" not in block:
                return False
        return True

    no_vllm = {
        family
        for family, row in rows.items()
        if family not in _FAMILY_TO_VLLM_REASONING
        and all_unavailable(row["cases"], "vllm")
    }
    no_sglang = {
        family
        for family, row in rows.items()
        if family not in _FAMILY_TO_SGLANG_REASONING
        and all_unavailable(row["cases"], "sglang")
    }
    return no_vllm, no_sglang


def family_suffix(
    reasoning_family: str | None,
    no_vllm: set[str],
    no_sglang: set[str],
) -> str:
    if reasoning_family is None:
        return ""
    suffix = ""
    if reasoning_family in no_vllm:
        suffix += "†"
    if reasoning_family in no_sglang:
        suffix += "§"
    return suffix


def _columns_for_mode(columns: list[str], mode: str) -> list[str]:
    return [case_id for case_id in columns if case_id.startswith(f"REASONING.{mode}.")]


def _is_na_stub(case: dict[str, Any]) -> bool:
    return (
        set(case) <= {"description", "reason", "explanation", "ref", "spec_ref"}
        and _explanation(case) is not None
    )


def _case_has_parser_input(case: dict[str, Any]) -> bool:
    return "model_text" in case or "chunks" in case


def _python_peer_has_parser(family: str | None, impl: str) -> bool:
    if family is None:
        return False
    if impl == "vllm":
        return family in _FAMILY_TO_VLLM_REASONING
    if impl == "sglang":
        return family in _FAMILY_TO_SGLANG_REASONING
    return False


def _python_exception_impls(
    case: dict[str, Any] | None,
    family: str | None,
) -> tuple[str, ...]:
    """Impls whose Python parser would raise on this input-less n/a stub.

    A no-`expected` n/a stub carries no `model_text`/`chunks`, so feeding it to
    the vLLM/SGLang Python parser raises ``KeyError: 'model_text'``. Surface that
    as a parser exception for any family that has a Python peer parser.
    """
    if not case or "expected" in case or not _is_na_stub(case):
        return ()
    if _case_has_parser_input(case):
        return ()
    return tuple(
        impl for impl in ("vllm", "sglang") if _python_peer_has_parser(family, impl)
    )


def _python_exception_marker(
    case: dict[str, Any] | None,
    family: str | None,
) -> str:
    letters = {"vllm": "V", "sglang": "S"}
    return "".join(
        f"{letters[impl]}✗" for impl in _python_exception_impls(case, family)
    )


def _python_exception_tooltip_lines(
    case: dict[str, Any] | None,
    family: str | None,
) -> list[str]:
    names = {"vllm": "vLLM Python", "sglang": "SGLang Python"}
    return [
        f"{names[impl]}: parser exception — KeyError: 'model_text'"
        for impl in _python_exception_impls(case, family)
    ]


def _cell(case: dict[str, Any] | None, family: str | None = None) -> tuple[str, str]:
    if case is None:
        return "—", "missing fixture coverage"
    if "expected" not in case:
        if _is_na_stub(case):
            # No Dynamo `expected` block, so from the Dynamo-as-reference compare view
            # this cell is simply n/a — don't surface peer parser exceptions (V✗/S✗)
            # in the grid. The exception detail still shows in the tooltip.
            marker = _python_exception_marker(case, family)
            if marker:
                parts = [_explanation(case), *_python_exception_tooltip_lines(case, family)]
                return "n/a", "\n".join(parts)
            return "n/a", _explanation(case)
        return "?", "fixture has no expected block"

    expected = case["expected"]
    dynamo = expected["dynamo"]
    dynamo_leak = _has_dynamo_leak(case, family)
    dynamo_leak_reason = _dynamo_leak_reason(expected, family) if dynamo_leak else None
    markers = []
    unavailable = 0
    tooltip_parts = [case.get("description", "")]

    for impl, letter in (("vllm", "V"), ("sglang", "S")):
        spec = expected[impl]
        if "unavailable" in spec:
            unavailable += 1
            tooltip_parts.append(f"{impl}: unavailable — {spec['unavailable']}")
            continue
        if "error" in spec:
            markers.append(f"{letter}✗")
            tooltip_parts.append(f"{impl}: parser exception — {spec['error']}")
            continue
        if _canonical(spec) == _canonical(dynamo):
            tooltip_parts.append(f"{impl}: matches Dynamo")
            continue
        suffix = (
            "?"
            if (dynamo_leak and not dynamo_leak_reason)
            or (not dynamo_leak and not _explanation(spec))
            else ""
        )
        markers.append(f"{letter}{suffix}")
        reason = (
            dynamo_leak_reason if dynamo_leak else (_explanation(spec) or "research-needed")
        )
        tooltip_parts.append(f"{impl}: diverges — {reason}")

    if unavailable == 2:
        if dynamo_leak:
            return "↯·", "\n".join(p for p in tooltip_parts if p)
        return "·", "\n".join(p for p in tooltip_parts if p)
    if dynamo_leak:
        return "↯" + ("".join(markers) or "?"), "\n".join(p for p in tooltip_parts if p)
    if markers:
        return "".join(markers), "\n".join(p for p in tooltip_parts if p)
    if unavailable:
        return "=", "\n".join(p for p in tooltip_parts if p)
    return "=", "\n".join(p for p in tooltip_parts if p)


def _render_markdown_row(
    row: dict[str, str | None],
    rows: dict[str, dict[str, Any]],
    columns: list[str],
    no_vllm: set[str],
    no_sglang: set[str],
) -> str:
    reasoning_family = row["reasoning_family"]
    tool_family = row["tool_family"]
    parser_label = _parser_label_text(tool_family, reasoning_family, no_vllm, no_sglang)
    cells = [str(row["model_label"]), parser_label]
    for case_id in columns:
        if reasoning_family is None:
            marker = "n/a"
        else:
            marker, _ = _cell(
                rows[reasoning_family]["cases"].get(case_id),
                reasoning_family,
            )
        cells.append(marker)
    return "| " + " | ".join(cells) + " |"


def _markdown(rows: dict[str, dict[str, Any]], columns: list[str]) -> str:
    out = []
    top_n, others, reasoning_only = _build_display_groups(rows)
    no_vllm, no_sglang = _derive_no_peer_sets(rows)
    header = [
        "model",
        "Reasoning family",
        *[_display_case_id(c) for c in columns],
    ]
    out.append("| " + " | ".join(header) + " |")
    out.append("|---|---|" + "|".join([":-:" for _ in columns]) + "|")
    out.append("| **Top-N models** |   |" + "   |" * len(columns))
    for row in top_n:
        out.append(_render_markdown_row(row, rows, columns, no_vllm, no_sglang))
    out.append("| **Others** |   |" + "   |" * len(columns))
    for row in others:
        out.append(_render_markdown_row(row, rows, columns, no_vllm, no_sglang))
    if reasoning_only:
        out.append("| **Reasoning-only** |   |" + "   |" * len(columns))
        for row in reasoning_only:
            out.append(_render_markdown_row(row, rows, columns, no_vllm, no_sglang))
    return "\n".join(out)


def _display_case_id(case_id: str) -> str:
    parts = case_id.split(".")
    return ".".join(parts[2:])


def _case_doc_id(case_id: str) -> str:
    return case_id.replace("REASONING.", "", 1)


def _case_mode(case_id: str) -> str:
    return _case_doc_id(case_id).split(".", 1)[0]


def _case_group_label(case_id: str) -> str:
    return _CASE_GROUP_BY_CASE.get(_case_doc_id(case_id), "Other")


def _case_group_key(case_id: str) -> str:
    label_key = re.sub(r"[^a-z0-9]+", "_", _case_group_label(case_id).lower())
    return f"{_case_mode(case_id)}_{label_key.strip('_')}"


def _case_band_class(case_id: str) -> str:
    group_idx = _CASE_GROUP_INDEX_BY_CASE.get(_case_doc_id(case_id), len(CASE_GROUPS))
    return f"case-band-{group_idx % 2}"


def _case_runs(columns: list[str]) -> list[list[str]]:
    runs = []
    start = 0
    while start < len(columns):
        group_key = _case_group_key(columns[start])
        end = start + 1
        while end < len(columns) and _case_group_key(columns[end]) == group_key:
            end += 1
        runs.append(columns[start:end])
        start = end
    return runs


def _column_placeholder_html(key: str, tag: str = "td") -> str:
    key_attr = html_lib.escape(key)
    return (
        f'<{tag} class="col-placeholder col-hidden" '
        f'data-col-placeholder-group="{key_attr}"></{tag}>'
    )


def _column_control_header_html(
    key: str,
    label: str,
    *,
    default_visible: bool,
    css_class: str = "",
    colspan: int | None = None,
) -> str:
    key_attr = html_lib.escape(key)
    visible = "true" if default_visible else "false"
    classes = " ".join(part for part in ("column-control", css_class) if part)
    span_size = colspan if colspan is not None else 1
    if colspan is not None:
        span_attr = f'colspan="{colspan}" data-expanded-colspan="{colspan}"'
    else:
        span_attr = 'rowspan="2"'
    action = "Collapse" if default_visible else "Expand"
    return (
        f'<th class="{html_lib.escape(classes)}" data-col-control-group="{key_attr}" '
        f"{span_attr}>"
        f'<button type="button" class="col-toggle" data-col-toggle="{key_attr}" '
        f'data-col-label="{html_lib.escape(label)}" data-col-span="{span_size}" '
        f'data-default-visible="{visible}" aria-pressed="{visible}" '
        f'aria-label="{action} {html_lib.escape(label)} column">'
        '<span class="col-toggle-symbol" aria-hidden="true"></span>'
        f'<span class="col-toggle-label">{html_lib.escape(label)}</span>'
        "</button></th>"
    )


def _case_group_headers_html(columns: list[str]) -> str:
    headers = [
        _column_control_header_html("model", "Model", default_visible=True),
        _column_control_header_html("parser", "Reasoning family", default_visible=True),
    ]
    for run in _case_runs(columns):
        label = _case_group_label(run[0])
        group_key = _case_group_key(run[0])
        headers.append(
            _column_control_header_html(
                group_key,
                label,
                default_visible=True,
                css_class=f"case-group {_case_band_class(run[0])}",
                colspan=len(run),
            )
        )
    return "".join(headers)


def _parse_case_descriptions() -> dict[str, str]:
    if not REASONING_CASES_MD.exists():
        return {}
    pat = re.compile(
        r"\*\*`REASONING\.(batch|stream)\.([0-9]+(?:\.[a-z])?)`\*\*\s+(.+)"
    )
    out = {}
    lines = REASONING_CASES_MD.read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(lines):
        match = pat.search(lines[i])
        if not match:
            i += 1
            continue
        mode, sub, desc = match.groups()
        body_parts = [desc.strip()]
        j = i + 1
        while j < len(lines):
            nxt = lines[j]
            if not nxt.strip() or not nxt.startswith(" "):
                break
            if pat.search(nxt):
                break
            body_parts.append(nxt.strip())
            j += 1
        out.setdefault(f"{mode}.{sub}", " ".join(body_parts).rstrip("."))
        i = j
    return out


def _case_header_html(case_id: str, descriptions: dict[str, str]) -> str:
    display = _display_case_id(case_id)
    desc = descriptions.get(_case_doc_id(case_id)) or ""
    href = common.LINKS["reasoning_cases"]
    return (
        f'<th class="case-sub {_case_band_class(case_id)}" '
        f'data-col-hide-group="{html_lib.escape(_case_group_key(case_id))}">'
        f'<a href="{href}" title="{html_lib.escape(desc)}">'
        f"{html_lib.escape(display)}</a></th>"
    )


def _case_headers_html(columns: list[str], descriptions: dict[str, str]) -> str:
    headers = []
    for run in _case_runs(columns):
        headers.extend(_case_header_html(case_id, descriptions) for case_id in run)
        headers.append(_column_placeholder_html(_case_group_key(run[0]), tag="th"))
    return "".join(headers)


def _glossary_groups(
    descriptions: dict[str, str], columns: list[str]
) -> list[dict[str, object]]:
    if not descriptions:
        return []
    return [
        {
            "label": _case_group_label(run[0]),
            "rows": [
                (
                    _case_doc_id(case_id),
                    descriptions.get(_case_doc_id(case_id), ""),
                )
                for case_id in run
            ],
        }
        for run in _case_runs(columns)
    ]


def _markup_family(family: str | None) -> str | None:
    if family == "gpt_oss":
        return "harmony"
    return family


def _colorize_kimi(text: str) -> str:
    pieces = []
    for part in re.split(r"(◁/?think▷)", text):
        if part == "◁think▷" or part == "◁/think▷":
            pieces.append(f'<span class="tt-c0">{html_lib.escape(part)}</span>')
        else:
            pieces.append(html_lib.escape(part))
    return "".join(pieces)


def _colorize_granite(text: str) -> str:
    replacements = {
        "Here is my thought process:": "tt-c0",
        "Here is my response:": "tt-c1",
        "Here's my thought process:": "tt-c0",
        "Here's my response:": "tt-c1",
    }
    pattern = re.compile("|".join(re.escape(k) for k in replacements))
    pieces = []
    last = 0
    for match in pattern.finditer(text):
        pieces.append(html_lib.escape(text[last : match.start()]))
        marker = match.group(0)
        cls = replacements[marker]
        pieces.append(f'<span class="{cls}">{html_lib.escape(marker)}</span>')
        last = match.end()
    pieces.append(html_lib.escape(text[last:]))
    return "".join(pieces)


_STREAM_TAG_RE = re.compile(r"<[^<>]+>|\[/?[A-Z][A-Z0-9_]*\]")
_HARMONY_TOKEN_RE = re.compile(r"<\|([A-Za-z_]+)\|>")
_HARMONY_TURN_CLOSE = frozenset({"end", "return", "call"})
_HARMONY_SEGMENT_CLASS = {
    "start": "tt-h-start",
    "channel": "tt-h-channel",
    "constrain": "tt-h-constrain",
    "message": "tt-h-message",
    "end": "tt-h-stop",
    "return": "tt-h-stop",
    "call": "tt-h-call",
}
_GEMMA4_CHANNEL_RE = re.compile(r"<\|channel>[A-Za-z_]+(?:\n)?|<channel\|>")
_PIPES = ("|", "｜")
_BEGIN_SUFFIXES = ("_begin", "▁begin")
_END_SUFFIXES = ("_end", "▁end")


def _split_name(value: str) -> str:
    return re.split(r"[\s/>=]", value, 1)[0].rstrip("|")


def _strip_marker_suffix(value: str, suffixes: tuple[str, ...]) -> str | None:
    for suffix in suffixes:
        if value.endswith(suffix) and len(value) > len(suffix):
            return value[: -len(suffix)]
    return None


def _stream_tag_kind(token: str) -> tuple[str | None, str]:
    if token.startswith("[/"):
        return "close", token[2:-1]
    if token.startswith("["):
        return "open", token[1:-1]

    inner = token[1:-1]
    if inner.startswith("/"):
        return "close", _split_name(inner[1:])
    starts_pipe = inner[:1] in _PIPES
    ends_pipe = inner[-1:] in _PIPES
    if starts_pipe and ends_pipe and len(inner) >= 2:
        middle = inner[1:-1]
        stripped = _strip_marker_suffix(middle, _BEGIN_SUFFIXES)
        if stripped is not None:
            return "open", stripped
        stripped = _strip_marker_suffix(middle, _END_SUFFIXES)
        if stripped is not None:
            return "close", stripped
        return "singleton", middle
    if starts_pipe and inner[:1] == "|":
        return "open", _split_name(inner[1:])
    if ends_pipe and inner[-1:] == "|":
        return "close", _split_name(inner[:-1])
    return "open", _split_name(inner)


def _tag_intervals(text: str) -> list[dict[str, Any]]:
    intervals: list[dict[str, Any]] = []
    stack: list[tuple[str, int]] = []
    color_seq = 0

    def next_class() -> str:
        nonlocal color_seq
        cls = f"tt-c{color_seq % 8}"
        color_seq += 1
        return cls

    for match in _STREAM_TAG_RE.finditer(text):
        idx = len(intervals)
        token = match.group(0)
        kind, name = _stream_tag_kind(token)
        intervals.append({"start": match.start(), "end": match.end(), "class": None})
        if kind == "singleton":
            intervals[idx]["class"] = next_class()
        elif kind == "close":
            match_at = -1
            for i in range(len(stack) - 1, -1, -1):
                if stack[i][0] == name:
                    match_at = i
                    break
            if match_at >= 0:
                for _, unmatched_idx in stack[match_at + 1 :]:
                    intervals[unmatched_idx]["class"] = "tt-orphan"
                cls = next_class()
                intervals[stack[match_at][1]]["class"] = cls
                intervals[idx]["class"] = cls
                del stack[match_at:]
            else:
                intervals[idx]["class"] = "tt-orphan"
        elif kind == "open":
            stack.append((name, idx))
        else:
            intervals[idx]["class"] = "tt-orphan"
    for _, idx in stack:
        intervals[idx]["class"] = "tt-orphan"
    return intervals


def _simple_marker_intervals(
    text: str, markers: dict[str, str]
) -> list[dict[str, Any]]:
    pattern = re.compile("|".join(re.escape(marker) for marker in markers))
    return [
        {"start": m.start(), "end": m.end(), "class": markers[m.group(0)]}
        for m in pattern.finditer(text)
    ]


def _harmony_intervals(text: str) -> list[dict[str, Any]]:
    intervals = []
    for match in _HARMONY_TOKEN_RE.finditer(text):
        token_name = match.group(1)
        if token_name in _HARMONY_TURN_CLOSE:
            end = match.end()
        else:
            next_match = _HARMONY_TOKEN_RE.search(text, match.end())
            end = next_match.start() if next_match else len(text)
        intervals.append(
            {
                "start": match.start(),
                "end": end,
                "class": _HARMONY_SEGMENT_CLASS.get(token_name, "tt-h-other"),
                "token_start": match.start(),
                "token_end": match.end(),
                "harmony": True,
            }
        )
    return intervals


def _gemma4_channel_intervals(text: str) -> list[dict[str, Any]]:
    intervals: list[dict[str, Any]] = []
    stack: list[int] = []
    color_seq = 0

    def next_class() -> str:
        nonlocal color_seq
        cls = f"tt-c{color_seq % 8}"
        color_seq += 1
        return cls

    for match in _GEMMA4_CHANNEL_RE.finditer(text):
        token = match.group(0)
        idx = len(intervals)
        intervals.append({"start": match.start(), "end": match.end(), "class": None})
        if token.startswith("<|channel>"):
            stack.append(idx)
        elif stack:
            open_idx = stack.pop()
            cls = next_class()
            intervals[open_idx]["class"] = cls
            intervals[idx]["class"] = cls
        else:
            intervals[idx]["class"] = "tt-orphan"

    for idx in stack:
        intervals[idx]["class"] = "tt-orphan"
    return intervals


def _stream_intervals(text: str, family: str | None) -> list[dict[str, Any]]:
    if family == "gpt_oss":
        return _harmony_intervals(text)
    if family == "gemma4":
        return _gemma4_channel_intervals(text)
    if family == "kimi":
        return _simple_marker_intervals(
            text,
            {"◁think▷": "tt-c0", "◁/think▷": "tt-c0"},
        )
    if family == "granite":
        return _simple_marker_intervals(
            text,
            {
                "Here is my thought process:": "tt-c0",
                "Here is my response:": "tt-c1",
                "Here's my thought process:": "tt-c0",
                "Here's my response:": "tt-c1",
            },
        )
    return _tag_intervals(text)


def _render_harmony_interval_slice(
    text: str, interval: dict[str, Any], start: int, end: int
) -> str:
    token_start = interval["token_start"]
    token_end = interval["token_end"]
    parts = []
    cursor = start
    token_slice_start = max(start, token_start)
    token_slice_end = min(end, token_end)
    if cursor < token_slice_start:
        parts.append(html_lib.escape(text[cursor:token_slice_start]))
        cursor = token_slice_start
    if token_slice_start < token_slice_end:
        parts.append(
            '<span class="tt-h-token">'
            f"{html_lib.escape(text[token_slice_start:token_slice_end])}</span>"
        )
        cursor = token_slice_end
    if cursor < end:
        parts.append(html_lib.escape(text[cursor:end]))
    return f'<span class="tt-h {interval["class"]}">{"".join(parts)}</span>'


def _render_interval_slice(
    text: str, interval: dict[str, Any], start: int, end: int
) -> str:
    if interval.get("harmony"):
        return _render_harmony_interval_slice(text, interval, start, end)
    return (
        f'<span class="{interval["class"]}">{html_lib.escape(text[start:end])}</span>'
    )


def _colorize_chunk_slice(
    text: str, intervals: list[dict[str, Any]], start: int, end: int
) -> str:
    pieces = []
    cursor = start
    for interval in intervals:
        istart = interval["start"]
        iend = interval["end"]
        if iend <= start:
            continue
        if istart >= end:
            break
        overlap_start = max(start, istart)
        overlap_end = min(end, iend)
        if cursor < overlap_start:
            pieces.append(html_lib.escape(text[cursor:overlap_start]))
        pieces.append(
            _render_interval_slice(text, interval, overlap_start, overlap_end)
        )
        cursor = overlap_end
    if cursor < end:
        pieces.append(html_lib.escape(text[cursor:end]))
    return "".join(pieces)


def _colorize_stream_chunks(chunks: list[Any], family: str) -> list[str]:
    raw_chunks = [str(chunk) for chunk in chunks]
    text = "".join(raw_chunks)
    intervals = _stream_intervals(text, family)
    rendered = []
    cursor = 0
    for chunk in raw_chunks:
        end = cursor + len(chunk)
        rendered.append(_colorize_chunk_slice(text, intervals, cursor, end))
        cursor = end
    return rendered


def _colorize_gemma4(text: str) -> str:
    return _colorize_chunk_slice(text, _gemma4_channel_intervals(text), 0, len(text))


def _colorize_reasoning_markup(value: str, family: str | None) -> str:
    if family == "gemma4":
        return _colorize_gemma4(value)
    if family == "kimi":
        return _colorize_kimi(value)
    if family == "granite":
        return _colorize_granite(value)
    return colorize_markup(value, _markup_family(family))


def _format_value_html(name: str, value: Any, family: str | None = None) -> str:
    if isinstance(value, str):
        return f'{html_lib.escape(name)}="{_colorize_reasoning_markup(value, family)}"'
    rendered = html_lib.escape(json.dumps(value, ensure_ascii=False))
    return f"{html_lib.escape(name)}={rendered}"


def _display_family_mapping_note(family: str, display_family: str | None) -> str:
    if display_family and display_family != family:
        return f"{display_family!r} (fixture: {family!r})"
    return repr(family)


def _format_unavailable_html(
    message: Any,
    family: str,
    display_family: str | None,
) -> str:
    rendered = str(message)
    if display_family and display_family != family:
        replacement = f"{display_family!r} (fixture: {family!r})"
        replaced = rendered.replace(repr(family), replacement)
        if replaced == rendered:
            replaced = f"{rendered} (row family: {replacement})"
        rendered = replaced
    return html_lib.escape(f"unavailable: {rendered}")


def _format_output_block_html(
    block: dict[str, Any] | None,
    family: str,
    *,
    display_family: str | None = None,
) -> str:
    if block is None:
        return html_lib.escape("(no expectation)")
    if "unavailable" in block:
        return _format_unavailable_html(block["unavailable"], family, display_family)
    if "error" in block:
        return html_lib.escape(f"error matching {block['error']!r}")

    lines = [
        _format_value_html("reasoning_text", block.get("reasoning_text"), family),
        _format_value_html("normal_text", block.get("normal_text"), family),
    ]
    return "\n".join(lines)


def _input_text_html(case: dict[str, Any], family: str) -> str:
    if "chunks" in case:
        chunk_lines = ["chunks=["]
        for i, chunk_html in enumerate(_colorize_stream_chunks(case["chunks"], family)):
            chunk_lines.append(f'  {i}: "{chunk_html}"')
        chunk_lines.append("]")
        return "\n".join(chunk_lines)
    return _format_value_html("input_text", case.get("model_text", ""), family)


def _vllm_parser_name(family: str, case: dict[str, Any]) -> str | None:
    return case.get("vllm_parser") or _FAMILY_TO_VLLM_REASONING.get(family)


def _vllm_chat_template_kwargs(
    parser_name: str | None, case: dict[str, Any]
) -> dict[str, Any]:
    if not parser_name:
        return {}
    kwargs: dict[str, Any] = {}
    if parser_name == "deepseek_v4":
        kwargs["enable_thinking"] = True
    if parser_name == "deepseek_v3":
        kwargs["thinking"] = True
    kwargs.update(case.get("chat_template_kwargs", {}))
    return kwargs


def _sglang_parser_name(family: str, case: dict[str, Any]) -> str | None:
    return case.get("sglang_parser") or _FAMILY_TO_SGLANG_REASONING.get(family)


def _flag_value(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _presence(value: Any) -> str:
    return "provided" if value else "none"


def _harness_flags_html(
    case_id: str,
    family: str,
    case: dict[str, Any],
    *,
    display_family: str | None = None,
) -> str:
    vllm_parser = _vllm_parser_name(family, case)
    vllm_kwargs = _vllm_chat_template_kwargs(vllm_parser, case)
    sglang_parser = _sglang_parser_name(family, case)

    mode = _case_mode(case_id)
    if mode == "stream":
        dynamo_token_flag = f"token_chunks={_presence(case.get('token_chunks'))}"
        if case.get("token_chunks"):
            vllm_token_flag = "delta_token_ids=fixture token_chunks"
        elif vllm_parser == "openai_gptoss":
            vllm_token_flag = "delta_token_ids=derived from Harmony tokenizer"
        else:
            vllm_token_flag = "delta_token_ids=derived from stub tokenizer"
    else:
        dynamo_token_flag = f"token_ids={_presence(case.get('token_ids'))}"
        vllm_token_flag = "request.skip_special_tokens=false"

    family_flag = (
        f"parser_family={display_family}; fixture_family={family}"
        if display_family and display_family != family
        else f"parser_family={family}"
    )
    lines = [
        "Parity harness flags used for this result:",
        (
            "dynamo: "
            f"{family_flag}; "
            f"in_reasoning={_flag_value(case.get('in_reasoning', False))}; "
            f"{dynamo_token_flag}"
        ),
    ]

    if vllm_parser:
        lines.append(
            "vllm: "
            f"reasoning_parser={vllm_parser}; "
            f"chat_template_kwargs={_flag_value(vllm_kwargs)}; "
            f"{vllm_token_flag}"
        )
    else:
        lines.append(
            "vllm: unavailable for Dynamo parser family "
            f"{_display_family_mapping_note(family, display_family)}"
        )

    if sglang_parser:
        lines.append(
            "sglang: "
            f"model_type={sglang_parser}; "
            "stream_reasoning=true; "
            f"force_reasoning={_flag_value(case.get('force_reasoning', False))}"
        )
    else:
        lines.append(
            "sglang: unavailable for Dynamo parser family "
            f"{_display_family_mapping_note(family, display_family)}"
        )

    lines.extend(
        [
            "",
            "Not set by this parser-level parity harness:",
            "- frontend `tool_choice`",
            "- SGLang `separate_reasoning`",
            "- request `enable_thinking` / `thinking` / `thinking_mode` "
            "unless shown in vLLM chat_template_kwargs above",
        ]
    )
    return html_lib.escape("\n".join(lines))


def _tooltip_for(
    case: dict[str, Any],
    dyn: dict[str, Any],
    family: str | None,
) -> str:
    """Build the peer-divergence explanation used by hover popups."""

    parts: list[str] = []
    dynamo_leak = _has_dynamo_leak(case, family)
    expected = case.get("expected", {})
    dynamo_leak_reason = _dynamo_leak_reason(expected, family) if dynamo_leak else None
    for impl in ("vllm", "sglang"):
        block = expected.get(impl)
        if not isinstance(block, dict) or block is dyn:
            continue
        if "unavailable" in block:
            continue
        name = _IMPL_DISPLAY.get(impl, impl)
        if "error" in block:
            parts.append(f"{name}: parser exception matching {block['error']!r}")
            continue
        if _canonical(block) == _canonical(dyn):
            continue
        if dynamo_leak_reason:
            continue
        if dynamo_leak:
            parts.append(f"{name}: (research-needed — no `explanation:` field yet)")
        elif _explanation(block) and not dynamo_leak:
            parts.append(f"{name}: {_explanation(block)}")
        elif "reasoning_text" in block or "normal_text" in block:
            parts.append(f"{name}: (research-needed — no `explanation:` field yet)")
    return "\n".join(parts)


def _explanations_for(
    case: dict[str, Any],
    dyn: dict[str, Any],
    family: str | None,
) -> str:
    parts = []
    peer_reasons = _tooltip_for(case, dyn, family)
    if peer_reasons:
        parts.append(peer_reasons)
    if isinstance(_explanation(dyn), str) and not _has_dynamo_leak(case, family):
        parts.append(f"Dynamo: {_explanation(dyn)}")
    return "\n".join(parts)


def _reasoning_candidate_chart_html(
    case_id: str,
    family: str,
    case: dict[str, Any],
    *,
    display_family: str | None = None,
) -> tuple[str, str] | None:
    """Left-to-right candidate chart (same layout/JS contract as the toolcalling
    charts): one column per compare candidate (`data-cand` = the compare-bar keys
    `dynamo`/`vllm`/`sglang`), REF column ordered first + `★ … ← REF`-marked by the
    shared JS. Batch input is a single text, so there is one `output` row; stream
    cases additionally list their input chunks (one row each) — the reasoning corpus
    records final output only, so per-chunk candidate cells stay `—`. A leaking
    candidate keeps its red `↯` on the column header (the list sections this chart
    replaces carried it)."""
    expected = case.get("expected")
    if not isinstance(expected, dict):
        return None
    mode = "stream" if ".stream." in case_id else "batch"
    has_chunks = isinstance(case.get("chunks"), list) and bool(case.get("chunks"))
    header = ""
    cells = ""
    for impl in ("dynamo", "vllm", "sglang"):
        blk = expected.get(impl)
        label = html_lib.escape(_reasoning_cand_label(impl, mode))
        leak = (
            ' <span class="ttip-leak">↯</span>'
            if isinstance(blk, dict) and _block_leak_reason(blk, family) is not None
            else ""
        )
        note = (
            common.timing_note("final output only; per-chunk timing not recorded")
            if has_chunks
            else ""
        )
        header += common.cand_th(impl, f"{label}{leak}{note}")
        body = _format_output_block_html(blk, family, display_family=display_family)
        note = _explanation(blk)
        if note:
            body += "\nexplanation: " + html_lib.escape(str(note))
        cells += common.cand_td(impl, body.replace(chr(10), "<br>"))
    rows = []
    chunks = case.get("chunks")
    if isinstance(chunks, list) and chunks:
        chunk_html = _colorize_stream_chunks(chunks, family)
        empty = "".join(
            common.cand_td(impl, "—") for impl in ("dynamo", "vllm", "sglang")
        )
        for i in range(len(chunks)):
            rows.append(f'<tr><td class="cin">{chunk_html[i]}</td>{empty}</tr>')
        rows.append(f'<tr class="ttip-final"><td class="cin">output</td>{cells}</tr>')
    else:
        rows.append(
            f'<tr class="ttip-final"><td class="cin">{_input_text_html(case, family)}</td>{cells}</tr>'
        )
    table = common.candidate_chart_table(header, rows)
    return ("Output (recorded from parser = expected)", table)


def _tooltip_html(
    case_id: str,
    family: str,
    case: dict[str, Any],
    *,
    display_family: str | None = None,
) -> str:
    head_family = display_family or family
    head = f"{case_id} — {head_family}"
    description = case.get("description")
    extra_sections: list[tuple[str, str]] = []
    if display_family and display_family != family:
        extra_sections.append(
            (
                "Family mapping",
                html_lib.escape(
                    "This row uses the public GLM/Nemotron reasoning parser "
                    f"family {display_family!r}; fixtures are shared with "
                    f"{family!r} because GLM and Nemotron use the same "
                    "parser implementation here."
                ),
            )
        )
    if "expected" not in case:
        reason = _explanation(case) or "fixture has no expected block"
        extra_sections = [("Why not applicable", linkify_text_html(str(reason)))]
        parser_exceptions = _python_exception_tooltip_lines(case, family)
        if parser_exceptions:
            extra_sections.append(
                (
                    "Python parser exceptions",
                    html_lib.escape("\n".join(parser_exceptions)),
                )
            )
        return build_parity_tooltip_html(
            head=head,
            description=str(description) if description else None,
            extra_sections=extra_sections,
            refs=[("Ref", case.get("ref")), ("Spec ref", case.get("spec_ref"))],
        )

    expected = case["expected"]
    dyn = expected.get("dynamo")
    # Per-candidate reasons live in each candidate's own toggleable section below (so
    # they show only when that candidate is selected), not in a global cross-engine
    # "Explanations" blob that would name engines outside the Base/Compare selection.
    extra_sections.append(
        (
            "Harness flags",
            _harness_flags_html(
                case_id,
                family,
                case,
                display_family=display_family,
            ),
        )
    )
    # The candidate chart (one column per compare candidate, left-to-right) replaces
    # the old per-impl list sections — its output row carries each candidate's block
    # + explanation, and its headers keep the per-candidate `↯`. The input lives in
    # the chart's first column, so the separate Input section is dropped too.
    chart = _reasoning_candidate_chart_html(
        case_id, family, case, display_family=display_family
    )

    dynamo_leak = (
        _dynamo_leak_reason(expected, family)
        if _has_dynamo_leak(case, family)
        else None
    )
    return build_parity_tooltip_html(
        head=head,
        description=str(description) if description else None,
        input_label=None if chart else ("Input chunks" if "chunks" in case else "Input"),
        input_html=None if chart else _input_text_html(case, family),
        output_sections=None,
        divergent_reasons=None,
        leak_label="↯ Dynamo leaks",
        leak_text=dynamo_leak
        or ("unresolved" if _has_dynamo_leak(case, family) else None),
        extra_sections=extra_sections,
        chart=chart,
        refs=[("Ref", case.get("ref")), ("Spec ref", case.get("spec_ref"))],
    )


def _missing_tooltip_html(
    case_id: str,
    family: str,
    *,
    display_family: str | None = None,
) -> str:
    head_family = display_family or family
    return build_parity_tooltip_html(
        head=f"{case_id} — {head_family}",
        extra_sections=[
            ("Missing fixture", html_lib.escape("missing fixture coverage"))
        ],
    )


def _no_reasoning_tooltip_html(case_id: str, tool_family: str) -> str:
    return build_parity_tooltip_html(
        head=f"{case_id} — {tool_family}",
        extra_sections=[
            (
                "Why not applicable",
                html_lib.escape(
                    "This tool calling parser row has no mapped Dynamo reasoning "
                    "parser fixture."
                ),
            )
        ],
    )


def _render_cell_html(
    case: dict[str, Any] | None,
    family: str,
    case_id: str,
    refs: dict[tuple[str, str], Path],
    *,
    display_family: str | None = None,
) -> str:
    if case is None:
        marker = "—"
        href = ""
        tooltip = _missing_tooltip_html(
            case_id,
            family,
            display_family=display_family,
        )
    else:
        marker, _ = _cell(case, family)
        href = common.fixture_href(
            "reasoning/fixtures/"
            + Path(os.path.relpath(refs[(family, case_id)], FIXTURES)).as_posix()
        )
        tooltip = _tooltip_html(
            case_id,
            family,
            case,
            display_family=display_family,
        )

    classes = f"cell {parity_cell_class(marker)} {_case_band_class(case_id)}"
    status_attrs = _overview_status_attrs(case, family)
    marker_attrs = _parser_marker_attrs(case, family)
    group_key = html_lib.escape(_case_group_key(case_id))
    label = html_lib.escape(marker)
    if href:
        body = f'<a href="{html_lib.escape(href)}">{label}</a>'
    else:
        body = label
    # Compare-candidates model (shared with the toolcalling batch tab): embed the
    # per-impl signature payload + a JS-filled marker span. JS colors the cell and
    # fills the count from the Base/Compare selection; inactive by default (falls
    # back to the parser-radio view before JS runs).
    cmp_json = _reasoning_cmp_json(case, family)
    cmp_attr = f' data-cmp="{cmp_json}"' if cmp_json else ""
    cmp_span = (
        '<span class="cmp-marker"><span class="marker-text"></span></span>'
        if cmp_json
        else ""
    )
    return (
        f'<td class="{classes}" data-col-hide-group="{group_key}"{cmp_attr} '
        f"{status_attrs} {marker_attrs}>"
        f"{cmp_span}{body}{tooltip}</td>"
    )


def _render_no_reasoning_cell_html(tool_family: str, case_id: str) -> str:
    classes = f"cell na {_case_band_class(case_id)}"
    group_key = html_lib.escape(_case_group_key(case_id))
    tooltip = _no_reasoning_tooltip_html(case_id, tool_family)
    return (
        f'<td class="{classes}" data-col-hide-group="{group_key}" '
        'data-status-dynamo="na" data-status-vllm="na" data-status-sglang="na" '
        'data-marker-dynamo="n/a" data-marker-vllm="n/a" data-marker-sglang="n/a" '
        'data-marker-parity-dynamo="n/a" data-marker-parity-vllm="n/a" '
        'data-marker-parity-sglang="n/a">'
        f"n/a{tooltip}</td>"
    )


def _implementation_label(reasoning_family: str | None) -> str:
    if reasoning_family is None:
        return "n/a"
    meta = _FAMILY_METADATA.get(reasoning_family, {})
    implementation = str(meta.get("implementation") or reasoning_family)
    class_name = implementation.split(" ", 1)[0]
    marker = re.search(r"`([^`]+)`", implementation)
    if class_name == "BasicReasoningParser" and marker:
        return f"{class_name} {marker.group(1)}"
    return class_name


def _display_reasoning_family(
    tool_family: str | None,
    reasoning_family: str | None,
) -> str | None:
    if tool_family == "glm47" and reasoning_family == "nemotron_deci":
        return "glm45"
    return reasoning_family


def _parser_label_text(
    tool_family: str | None,
    reasoning_family: str | None,
    no_vllm: set[str] | None = None,
    no_sglang: set[str] | None = None,
) -> str:
    if reasoning_family is None:
        return "n/a"
    suffix = family_suffix(reasoning_family, no_vllm or set(), no_sglang or set())
    return f"`{_implementation_label(reasoning_family)}`{suffix}"


def _reasoning_mode_meta(
    tool_family: str | None,
    reasoning_family: str | None,
) -> dict[str, Any]:
    if reasoning_family is None:
        return {
            "label": "n/a",
            "control": "not configured",
            "summary": (
                "This tool calling parser row has no mapped Dynamo reasoning "
                "parser fixture."
            ),
            "static": [],
        }

    meta = dict(_REASONING_MODE_METADATA.get(reasoning_family, {}))
    if not meta:
        meta = {
            "label": "custom",
            "control": "mostly static",
            "summary": "Custom reasoning parser family.",
            "static": [],
        }

    row_notes = []
    if tool_family and tool_family != reasoning_family:
        row_notes.append(
            f"Tool calling parser row `{tool_family}` maps to reasoning "
            f"family `{reasoning_family}`."
        )
    if tool_family in {"deepseek_v3_1", "deepseek_v3_2"}:
        row_notes.append(
            "DeepSeek V3.1/V3.2 aliases share the DeepSeek V3.x reasoning fixtures."
        )
    if tool_family == "glm47":
        row_notes.append(
            "GLM tool rows use the public `glm45` reasoning alias; that alias "
            "maps to the `nemotron_deci` implementation and fixtures."
        )
    if tool_family == "harmony":
        row_notes.append(
            "Tool calling parser `harmony` pairs with `gpt_oss` reasoning."
        )
    if tool_family == "kimi_k2":
        row_notes.append(
            "Tool calling parser `kimi_k2` pairs with `kimi_k25` reasoning."
        )

    if row_notes:
        meta["row_notes"] = row_notes
    return meta


def _reasoning_parser_tree_lines(
    tool_family: str | None,
    reasoning_family: str | None,
    meta: dict[str, Any],
) -> list[str]:
    if reasoning_family is None:
        return []

    implementation = _implementation_label(reasoning_family)
    shared = [name for name in meta.get("shared_with", []) if name != reasoning_family]
    aliases = [
        name
        for name in meta.get("aliases", [])
        if name != reasoning_family and name not in shared
    ]
    known_children = {*shared, *aliases}
    row_only = (
        [tool_family]
        if tool_family
        and tool_family != reasoning_family
        and tool_family not in known_children
        else []
    )
    children = [
        (reasoning_family, ""),
        *[(name, "") for name in sorted(row_only)],
        *[(name, "") for name in sorted(shared)],
        *[(name, " (alias)") for name in sorted(aliases)],
    ]

    lines = [
        "",
        "Shared implementation tree:",
        html_lib.escape(implementation),
    ]
    for i, (name, suffix) in enumerate(children):
        branch = "└── " if i == len(children) - 1 else "├── "
        name_label = html_lib.escape(name)
        if tool_family == name:
            name_label = f"<strong>{name_label}</strong>"
        elif tool_family is None and name == reasoning_family:
            name_label = f"<strong>{name_label}</strong>"
        lines.append(f"{branch}{name_label}{html_lib.escape(suffix)}")
    return lines


def _parser_cell_html(
    tool_family: str | None,
    reasoning_family: str | None,
    no_vllm: set[str],
    no_sglang: set[str],
) -> str:
    meta = _FAMILY_METADATA.get(reasoning_family or "", {})
    display_family = _display_reasoning_family(tool_family, reasoning_family)
    implementation_label = _implementation_label(reasoning_family)
    mode_meta = _reasoning_mode_meta(tool_family, reasoning_family)
    activation_name = display_family or "n/a"
    tooltip_lines = [
        html_lib.escape(
            f'This is activated via "--dyn-reasoning-parser {activation_name}".'
        ),
    ]
    if display_family:
        tooltip_lines.append(f"Parser family: {html_lib.escape(display_family)}")
    if tool_family and tool_family != display_family:
        tooltip_lines.append(f"Tool calling row: {html_lib.escape(tool_family)}")
    if display_family:
        if display_family != reasoning_family:
            tooltip_lines.append(
                f"Fixture family: {html_lib.escape(reasoning_family or '')}"
            )
    else:
        tooltip_lines.append(html_lib.escape("Reasoning parser family: n/a"))
    if meta.get("models"):
        tooltip_lines.append(
            "Models: "
            + html_lib.escape(", ".join(str(model) for model in meta["models"]))
        )
    if meta.get("rust_enum"):
        tooltip_lines.append(f"Rust enum: {html_lib.escape(meta['rust_enum'])}")
    if meta.get("implementation"):
        tooltip_lines.append(
            f"Implementation: {html_lib.escape(meta['implementation'])}"
        )
    peer_notes = []
    if reasoning_family in no_vllm:
        peer_notes.append("no vLLM peer reasoning parser")
    if reasoning_family in no_sglang:
        peer_notes.append("no SGLang peer reasoning parser")
    if peer_notes:
        tooltip_lines.append(
            "Peer availability: " + html_lib.escape(", ".join(peer_notes))
        )
    tooltip_lines.extend(
        [
            "",
            "Mode:",
            "- " + html_lib.escape(f"{mode_meta['label']} / {mode_meta['control']}"),
        ]
    )
    static_config = mode_meta.get("static", [])
    config_lines = [
        str(line)
        for line in static_config
        if not str(line).startswith(
            (
                "BasicReasoningParser",
                "Gemma4ReasoningParser",
                "GptOssReasoningParser",
                "GraniteReasoningParser",
                "MiniMaxAppendThinkParser",
            )
        )
    ]
    if config_lines:
        tooltip_lines.extend(
            ["", "Config:"] + [f"- {html_lib.escape(line)}" for line in config_lines]
        )
    tooltip_lines.extend(
        _reasoning_parser_tree_lines(tool_family, reasoning_family, meta)
    )
    parser_label = _parser_label_text(tool_family, reasoning_family, no_vllm, no_sglang)
    tooltip = (
        f'<div class="ttip-head">{html_lib.escape(parser_label)}</div>'
        f'<pre class="ttip-pre">{chr(10).join(tooltip_lines)}</pre>'
    )
    suffix = family_suffix(reasoning_family, no_vllm, no_sglang)
    suffix_html = (
        f'<span class="parser-suffix">{html_lib.escape(suffix)}</span>'
        if suffix
        else ""
    )
    if tool_family is None:
        label = f"<code>{html_lib.escape(implementation_label)}</code>"
    elif reasoning_family is None:
        label = "<code>n/a</code>"
    elif tool_family == reasoning_family:
        label = f"<code>{html_lib.escape(implementation_label)}</code>"
    else:
        label = f"<code>{html_lib.escape(implementation_label)}</code>"
    return (
        '<td class="parser" data-col-hide-group="parser">'
        f'{label}{suffix_html}<div class="ttip">{tooltip}</div></td>'
    )


def _render_row_html(
    row: dict[str, str | None],
    rows: dict[str, dict[str, Any]],
    columns: list[str],
    refs: dict[tuple[str, str], Path],
    no_vllm: set[str],
    no_sglang: set[str],
) -> str:
    tool_family = row["tool_family"]
    reasoning_family = row["reasoning_family"]
    cells = [
        f'<tr><td class="model" data-col-hide-group="model">'
        f"{_model_label_html(str(row['model_label']))}</td>",
        _column_placeholder_html("model"),
        _parser_cell_html(tool_family, reasoning_family, no_vllm, no_sglang),
        _column_placeholder_html("parser"),
    ]
    for run in _case_runs(columns):
        if reasoning_family is None:
            cells.extend(
                _render_no_reasoning_cell_html(str(tool_family), case_id)
                for case_id in run
            )
        else:
            cases = rows[reasoning_family]["cases"]
            display_family = _display_reasoning_family(tool_family, reasoning_family)
            cells.extend(
                _render_cell_html(
                    cases.get(case_id),
                    reasoning_family,
                    case_id,
                    refs,
                    display_family=display_family,
                )
                for case_id in run
            )
        cells.append(_column_placeholder_html(_case_group_key(run[0])))
    cells.append("</tr>")
    return "".join(cells)


def _compute_stats(
    rows: dict[str, dict[str, Any]],
    columns: list[str],
    display_rows: list[dict[str, str | None]],
) -> dict[str, int]:
    stats = {
        "families": len(display_rows),
        "sub_cases": len(columns),
        "slots": len(display_rows) * len(columns),
        "real": 0,
        "parity": 0,
        "dynamo_only": 0,
        "documented": 0,
        "research": 0,
        "errors": 0,
        "na": 0,
        "missing": 0,
    }
    for row in display_rows:
        family = row["reasoning_family"]
        for case_id in columns:
            if family is None:
                marker = "n/a"
            else:
                marker, _ = _cell(rows[family]["cases"].get(case_id), family)
            if marker == "—":
                stats["missing"] += 1
            elif marker == "n/a":
                stats["na"] += 1
            else:
                stats["real"] += 1
                if marker == "=":
                    stats["parity"] += 1
                elif marker in {"D", "·"}:
                    stats["dynamo_only"] += 1
                elif "!" in marker or "✗" in marker:
                    stats["errors"] += 1
                elif "?" in marker:
                    stats["research"] += 1
                else:
                    stats["documented"] += 1
    return stats


def _omitted_tool_calling_rows_html(rows: dict[str, dict[str, Any]]) -> str:
    parser_labels = _load_parser_labels()
    omitted = []
    for tool_family, label in parser_labels.items():
        reasoning_family = PARSER_TO_REASONING_FAMILY.get(tool_family)
        if reasoning_family not in rows:
            omitted.append((label, tool_family))
    if not omitted:
        return ""

    omitted.sort(key=lambda item: item[0].lower())
    rendered = ", ".join(
        f"{html_lib.escape(label)} (<code>{html_lib.escape(family)}</code>)"
        for label, family in omitted
    )
    return (
        "<br><br>"
        "<strong>Omitted tool calling parser rows:</strong> "
        "no corresponding Dynamo reasoning parser fixture for "
        f"{rendered}."
    )


def _has_missing_cells(rows: dict[str, dict[str, Any]], columns: list[str]) -> bool:
    top_n, others, reasoning_only = _build_display_groups(rows)
    for row in [*top_n, *others, *reasoning_only]:
        reasoning_family = row["reasoning_family"]
        if reasoning_family is None:
            continue
        cases = rows[reasoning_family]["cases"]
        if any(case_id not in cases for case_id in columns):
            return True
    return False


def _legend_html(rows: dict[str, dict[str, Any]], columns: list[str]) -> str:
    missing_text = ""
    if _has_missing_cells(rows, columns):
        missing_text = (
            " · " '<span style="color:#8a6d3b">—</span> missing fixture coverage'
        )
    legend = (
        "<strong>Legend:</strong> "
        '<span style="color:#0a7d2c">=</span> all captured peers match Dynamo · '
        '<span style="color:#8b949e">·</span> Dynamo-only fixture '
        "(both peers unavailable) · "
        '<span style="color:#555">V/S</span> divergence '
        "(V = vLLM, S = SGLang; intentional, has <code>explanation:</code>) · "
        '<span style="color:#b00">?</span> more research needed '
        "(e.g. V?, S? — diverges with no <code>explanation:</code> yet) · "
        '<span style="color:#b00">↯</span> Dynamo leaks reasoning markup '
        "or final-answer text · "
        '<span style="color:#b00">✗</span> parser exception '
        "(e.g. V✗, S✗ — Python parser raised) · "
        '<span style="color:#aaa">n/a</span> not applicable'
        f"{missing_text}."
        "<br>"
        '<span class="parser-suffix">†</span> = no vLLM peer reasoning parser '
        "for this family · "
        '<span class="parser-suffix">§</span> = no SGLang peer reasoning parser '
        "for this family."
        "<br>"
        '<span class="parser-suffix">‡</span> Nemotron V3 (Ultra) reuses the '
        "qwen3_coder tool calling parser; Nemotron V1 / V2 (DeciLM) is removed "
        "from the chart for being an older generation, but the nemotron_deci "
        "parser is still supported."
        "<br><br>"
        "<strong>Tooltip fields:</strong> "
        "<code>input_text</code>=raw model output or stream chunks fed into "
        "the reasoning parser · "
        "<code>reasoning_text</code>=hidden reasoning content extracted by "
        "the parser · "
        "<code>normal_text</code>=residual text passed onward to response "
        "assembly or downstream parsers · "
        "<code>harness flags</code>=the exact parser-level arguments used for "
        "the parity result."
    )
    return legend + _omitted_tool_calling_rows_html(rows)


def _mode_label(mode: str) -> str:
    if mode == "batch":
        return "REASONING.batch.*"
    if mode == "stream":
        return "REASONING.stream.*"
    return mode


# Standardized reasoning candidate label base: "<Engine> <Runtime>". Dynamo's
# reasoning parser is a Rust crate (dynamo-parsers 3.0.0); vLLM/SGLang reasoning
# parsers are Python. Full label: "<base> <version> (<mode>)".
_REASONING_ENGINE_RUNTIME = {
    "dynamo": "Dynamo Rust",
    "vllm": "vLLM Python",
    "sglang": "SGLang Python",
}


@functools.lru_cache(maxsize=1)
def _reasoning_version_by_impl() -> dict[str, str | None]:
    """{impl: display version} for reasoning candidates: Dynamo from the v1 crate
    Cargo.toml, vLLM/SGLang from the fixtures' captured_with. Cached so the tooltip
    section labels can carry the same version as the compare chips without reloading
    fixtures per cell."""
    rows, _, _ = _load()
    return {"dynamo": _dynamo_v1_version(), **_peer_captured_versions(rows)}


def _reasoning_cand_label(impl: str, mode: str) -> str:
    """Full compare-candidate label "<Engine> <Runtime> <version> (<mode>)", shared
    by the chips and the tooltip sections so the pop-up keys match the buckets.
    Dynamo's reasoning parser is the v1 crate (dynamo-parsers 3.x), so it reads
    "Dynamo Rust v1 3.0.0 (batch)" / "(stream)"; peers have no crate split."""
    base = _REASONING_ENGINE_RUNTIME.get(impl, _IMPL_DISPLAY[impl])
    if impl == "dynamo":
        eng, _, rt = base.partition(" ")  # "Dynamo" / "Rust" -> "Dynamo v1 Rust"
        base = f"{eng} v1 {rt}".strip()
    ver = _reasoning_version_by_impl().get(impl)
    return f"{base} {ver} ({mode})" if ver else f"{base} ({mode})"


def _panel_candidates(
    rows: dict[str, dict[str, Any]],
    columns: list[str],
    display_rows: list[DisplayRow],
    mode: str,
) -> list[dict[str, str]]:
    """Ordered compare-candidates for this panel: Dynamo, vLLM, SGLang — but only
    impls that appear in at least one displayed cell's `expected`. The first
    included candidate is the reference bucket "A"; all others default to "B".
    Labels are "<Engine> <Runtime> <version> (<mode>)" to match the other tabs.
    Mirrors generate_conformance_table._candidate_items (default_bucket layout)."""
    present: set[str] = set()
    for row in display_rows:
        reasoning_family = row["reasoning_family"]
        if reasoning_family is None or reasoning_family not in rows:
            continue
        cases = rows[reasoning_family]["cases"]
        for case_id in columns:
            case = cases.get(case_id)
            expected = case.get("expected") if isinstance(case, dict) else None
            if not isinstance(expected, dict):
                continue
            for impl in ("dynamo", "vllm", "sglang"):
                if impl in expected:
                    present.add(impl)
    # Version sourcing (Dynamo from the v1 crate Cargo.toml, peers from the fixtures'
    # captured_with) and the full label are shared with the tooltip sections via
    # _reasoning_cand_label so chips and pop-up keys read identically.
    candidates: list[dict[str, str]] = []
    for impl in ("dynamo", "vllm", "sglang"):
        if impl not in present:
            continue
        candidates.append(
            {
                "key": impl,
                "label": _reasoning_cand_label(impl, mode),
                "default_bucket": "A" if not candidates else "B",
            }
        )
    return candidates


def _peer_captured_versions(rows: dict[str, dict[str, Any]]) -> dict[str, str]:
    """Captured peer reasoning-parser versions, keyed by impl (vllm/sglang).

    Reads the `captured_with` blocks merged onto the family rows in `_load`. Only
    one version per engine is expected across the fixtures (all captured against
    the same container: vLLM 0.24.0 / SGLang 0.5.14). A single container per
    engine is available, so there is no older reasoning image to support a
    two-version compare -- each candidate just carries its real captured version.
    The last non-empty value wins if fixtures ever disagree."""
    key_by_impl = {"vllm": "vllm_python", "sglang": "sglang_python"}
    out: dict[str, str] = {}
    for row in rows.values():
        captured = row.get("captured_with") or {}
        for impl, key in key_by_impl.items():
            version = captured.get(key)
            if version:
                out[impl] = str(version)
    return out


def _dynamo_v1_version() -> str | None:
    """Version label for the Dynamo v1 reasoning parser, taken from the PUBLISHED fixture
    provenance — the `dynamo-<ver>` dir in the tool-calling batch corpus (the same v1
    crate powers reasoning and tool-calling) — NOT the live `parsers/v1/Cargo.toml`.

    Sourcing the label from the fixtures keeps it consistent with the tool-calling tabs
    (both read "3.0.0") and, crucially, matching the version the data was actually
    captured against. Reading the live Cargo.toml instead makes the label drift ahead of
    the fixtures the moment the crate is bumped but before a re-capture/republish — the
    label would claim 4.1.0 while every fixture still holds 3.0.0-era output."""
    cache = os.environ.get("CONFORMANCE_FIXTURES_ROOT")
    if not cache:
        return None
    root = Path(cache) / "toolcalling" / "fixtures-batch-v1"
    if not root.is_dir():
        return None
    versions = [
        d.name.split("-", 1)[1]
        for d in root.iterdir()
        if d.is_dir() and d.name.startswith("dynamo-")
    ]
    if not versions:
        return None
    # Highest recorded capture (normally exactly one v1 dynamo dir exists).
    return max(versions, key=lambda v: tuple(int(x) for x in re.findall(r"\d+", v)))


def _html_panel(
    rows: dict[str, dict[str, Any]],
    columns: list[str],
    refs: dict[tuple[str, str], Path],
    no_vllm: set[str],
    no_sglang: set[str],
    *,
    mode: str,
    active: bool,
) -> dict[str, object]:
    descriptions = _parse_case_descriptions()
    top_n, others, reasoning_only = _build_display_groups(rows)
    display_rows = [*top_n, *others, *reasoning_only]
    body_rows = []
    sections = [
        ("Top-N models", top_n),
        ("Others", others),
        ("Reasoning-only", reasoning_only),
    ]
    for label, section_rows in sections:
        if not section_rows:
            continue
        body_rows.append(
            f'<tr class="section"><td data-section-span colspan="{2 + len(columns)}">'
            f"{html_lib.escape(label)}</td></tr>"
        )
        for row in section_rows:
            body_rows.append(
                _render_row_html(row, rows, columns, refs, no_vllm, no_sglang)
            )

    return {
        "id": f"tab-{mode}",
        "mode": mode,
        "label": _mode_label(mode),
        "active": active,
        "group_headers": _case_group_headers_html(columns),
        "sub_headers": _case_headers_html(columns, descriptions),
        "body_rows": body_rows,
        "stats": _compute_stats(rows, columns, display_rows),
        "glossary_groups": _glossary_groups(descriptions, columns),
        "candidates": _panel_candidates(rows, columns, display_rows, mode),
    }


def _html(
    rows: dict[str, dict[str, Any]],
    columns: list[str],
    refs: dict[tuple[str, str], Path],
    modes: list[str],
) -> str:
    generated = datetime.datetime.now(
        zoneinfo.ZoneInfo("America/Los_Angeles")
    ).strftime("%Y-%m-%d %H:%M %Z")
    sha = _commit_sha()
    no_vllm, no_sglang = _derive_no_peer_sets(rows)
    panels = [
        _html_panel(
            rows,
            _columns_for_mode(columns, mode),
            refs,
            no_vllm,
            no_sglang,
            mode=mode,
            active=i == 0,
        )
        for i, mode in enumerate(modes)
    ]
    command = "python3 tests/parity/generate_parity_table_v1.py reasoning --html"
    output = "tests/parity/reasoning/PARITY.html"
    if len(modes) == 1:
        command += f" --mode {modes[0]}"
        output = f"tests/parity/reasoning/PARITY.{modes[0]}.html"
    tabs = []
    for i, mode in enumerate(modes):
        panel_id = f"tab-{mode}"
        active = " active" if i == 0 else ""
        selected = "true" if i == 0 else "false"
        tabs.append(
            f'<button class="tab-button{active}" id="{panel_id}-button" '
            f'type="button" role="tab" aria-selected="{selected}" '
            f'data-tab-target="{panel_id}">'
            f"{html_lib.escape(_mode_label(mode))}</button>"
        )

    return (
        _make_jinja_env()
        .get_template("parity_table_v1.html.j2")
        .render(
            title="Dynamo Reasoning Parser - Parity Table",
            stamp=generated,
            sha=sha,
            short_sha=sha[:12] if sha else "",
            command=command,
            output=output,
            tabs=tabs,
            panels=panels,
            peer_versions=[],
            intro_html="",
            legend_html=_legend_html(rows, columns),
            case_docs_href=common.LINKS["reasoning_cases"],
            case_docs_label="lib/parsers/REASONING_CASES.md",
            case_prefix="REASONING.",
        )
    )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--html", action="store_true", help="emit HTML instead of Markdown")
    ap.add_argument(
        "--mode",
        choices=("all", "batch", "stream"),
        help=(
            "Fixture mode to render. For HTML, default is all reasoning modes as "
            "tabs. For Markdown, default is batch."
        ),
    )
    args = ap.parse_args(argv)

    rows, columns, refs = _load()
    if args.html:
        modes = ["batch", "stream"] if args.mode in (None, "all") else [args.mode]
        print(_html(rows, columns, refs, modes))
    else:
        if args.mode == "all":
            ap.error("--mode all is only supported with --html")
        mode = args.mode or "batch"
        print(_markdown(rows, _columns_for_mode(columns, mode)))


if __name__ == "__main__":
    main()
