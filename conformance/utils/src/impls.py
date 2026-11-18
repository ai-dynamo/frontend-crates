# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Single source of truth for conformance implementation identity (audit B1).

One `ImplSpec` per parser implementation. Every other module derives its keys,
legacy aliases, display names, and matrix markers from `IMPL_SPECS` instead of
re-declaring parallel tables that drift — the vLLM Rust stream parser vs a
(nonexistent) vLLM Rust batch parser, `vllm` vs `vllm_python`, missing-parser-option
class of bug.

Capture, validation, metadata indexing, and implementation-identity tests all
import this module directly.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


def parser_families_path() -> Path:
    """The registry lives at src/parser_families.yaml in the repo; the render stage
    copies this module flat into tests/parity/ and the registry to <stage>/src/.

    Public because every consumer of the registry has to resolve BOTH layouts, and a
    second copy of this walk is a second thing to get wrong -- `unified_taxonomy` did
    exactly that and broke the render, which only runs from the stage."""
    here = Path(__file__).resolve()
    for cand in (
        here.parent / "parser_families.yaml",
        here.parents[2] / "src" / "parser_families.yaml",
    ):
        if cand.is_file():
            return cand
    raise FileNotFoundError(f"parser_families.yaml not found relative to {here}")


_REGISTRY = yaml.safe_load(parser_families_path().read_text())


@dataclass(frozen=True)
class ImplSpec:
    key: str                 # canonical fixture/render key
    legacy_key: str | None   # pre-canonical spelling still accepted on read
    engine: str              # dynamo | vllm | sglang
    language: str            # rust | python
    display: str             # human label
    marker_letter: str       # matrix marker letter (D/R/V/S)
    marker_lang: str         # matrix language subscript (r/p)
    modes: tuple[str, ...]   # tabs this impl participates in: batch and/or stream


# Dynamo v1 and v2 are COMPLETELY SEPARATE implementations (no shared code), so
# they are separate impls — exactly like vllm_python vs vllm_rust. v1
# (dynamo-parsers: batch + jail) is interim and will be removed at v2 parity;
# v2 (dynamo-parsers-v2: streaming) is the ultimate implementation (WIP). The
# baseline the peers are compared against is therefore MODE-dependent: v1 on
# the batch tabs, v2 on the stream tabs.
BASELINE_BATCH_IMPL = "dynamo_v1"
BASELINE_STREAM_IMPL = "dynamo_v2"
BASELINE_IMPLS: tuple[str, ...] = (BASELINE_BATCH_IMPL, BASELINE_STREAM_IMPL)

# Order is the canonical column order across the matrix. vLLM Rust is stream-only
# (no batch parser exists), so it is absent from the batch tabs. SGLang's language
# subscript is `r` by existing convention (not `p`) — preserved deliberately.
IMPL_SPECS: tuple[ImplSpec, ...] = (
    ImplSpec("dynamo_v1", "dynamo", "dynamo", "rust", "Dynamo v1 Rust", "D", "r", ("batch",)),
    ImplSpec("dynamo_v2", "dynamo_rust", "dynamo", "rust", "Dynamo v2 Rust", "D", "r", ("stream",)),
    ImplSpec("vllm_rust", None, "vllm", "rust", "vLLM Rust", "R", "r", ("stream",)),
    ImplSpec("vllm_python", "vllm", "vllm", "python", "vLLM Python", "V", "p", ("batch", "stream")),
    ImplSpec("sglang_python", "sglang", "sglang", "python", "SGLang Python", "S", "r", ("batch", "stream")),
)


def baseline_impl(impl_keys: tuple[str, ...]) -> str:
    """The Dynamo baseline for a tab's impl-key tuple (v1 for batch, v2 for stream)."""
    return next(k for k in impl_keys if k in BASELINE_IMPLS)

IMPL_KEYS: tuple[str, ...] = tuple(s.key for s in IMPL_SPECS)
STREAM_IMPL_KEYS: tuple[str, ...] = tuple(s.key for s in IMPL_SPECS if "stream" in s.modes)
BATCH_IMPL_KEYS: tuple[str, ...] = tuple(s.key for s in IMPL_SPECS if "batch" in s.modes)
PEER_IMPL_KEYS: tuple[str, ...] = tuple(k for k in IMPL_KEYS if k not in BASELINE_IMPLS)
LEGACY_IMPL_ALIASES: dict[str, str] = {s.legacy_key: s.key for s in IMPL_SPECS if s.legacy_key}
IMPL_DISPLAY: dict[str, str] = {s.key: s.display for s in IMPL_SPECS}
ENGINE_LETTER: dict[str, str] = {s.key: s.marker_letter for s in IMPL_SPECS}
IMPL_LANG_MARKER: dict[str, str] = {s.key: s.marker_lang for s in IMPL_SPECS}
# Aliases validate.py accepts on a `--impl` flag: peers only (Dynamo is not validated
# via validate.py; it goes through cargo test).
FIXTURE_IMPL_ALIASES: dict[str, str] = {
    s.legacy_key: s.key for s in IMPL_SPECS if s.legacy_key and s.key not in BASELINE_IMPLS
}


# B11: the capture wrapper stamps this marker into an `unavailable` reason when a
# peer parser was invoked and THREW (as opposed to not running). The renderer
# matches this SAME constant — a shared contract, not a private guessed regex — to
# classify the `✗` error marker. Going forward, captures can also record a
# structured `error: {kind, message}` block, which the renderer renders as `✗`
# directly; a plain-string `error` stays a declared expected-error (`!`).
PARSER_NOT_CAPTURED = "parser not captured"


# Adapter-backed batch rows require final fields; preserved legacy rows retain their recorded contract.
_BATCH_POLICY = _REGISTRY["unified_batch"]
UNIFIED_PEER_SOURCES = _REGISTRY["unified_peer_sources"]
UNIFIED_BATCH_IMPLS = tuple(_BATCH_POLICY)
BATCH_SOURCE_IMPL = {key: value["source"] for key, value in _BATCH_POLICY.items()}
BATCH_SOURCE_BLOCKERS = {
    (key, version): detail
    for key, value in _BATCH_POLICY.items()
    for version, detail in value.get("source_blockers", {}).items()
}
UNIFIED_DISPLAY = _REGISTRY["unified_display"]
UNIFIED_COMPARISON_DEFAULTS = tuple(UNIFIED_DISPLAY["comparison_defaults"])
UNIFIED_COMPARISON_SLOT_ENGINES = tuple(UNIFIED_DISPLAY["comparison_slot_engines"])
UNIFIED_MARKER_FAMILIES = {
    family: row["registry"] for family, row in _REGISTRY["unified"].items()
    if row.get("registry") and row["registry"] != family
}


def unified_grammar_family(family: str) -> str:
    return _REGISTRY["unified"].get(family, {}).get("grammar", family)


def _unified_pipelines(implementation: str) -> dict[str, tuple[str, str]]:
    return {family: tuple(row["capture_pipelines"][implementation])
            for family, row in _REGISTRY["unified"].items()
            if implementation in row.get("capture_pipelines", {})}


VLLM_UNIFIED_PIPELINES = _unified_pipelines("vllm_python")
SGLANG_UNIFIED_PIPELINES = _unified_pipelines("sglang_python")
DYNAMO_UNIFIED_PIPELINES = _unified_pipelines("dynamo_v1")
DYNAMO_JAIL_UNIFIED_PIPELINES = {**DYNAMO_UNIFIED_PIPELINES, **_unified_pipelines("dynamo_v1_jail")}
VLLM_TOKENIZER_MARKERS = {family: row["vllm_tokenizer_markers"]
                          for family, row in _REGISTRY["unified"].items()
                          if "vllm_tokenizer_markers" in row}

def batch_capture_label(implementation: str, version: str) -> str:
    source = next(spec for spec in IMPL_SPECS if spec.key == BATCH_SOURCE_IMPL[implementation])
    mode = "batch + jail-buffer" if implementation == "dynamo_v1_jail" else "batch"
    label = (f"{source.display} {version} ({mode})" if source.engine == "dynamo"
             else f"{source.display} {version} · {mode}")
    return label + (" (source unavailable)" if (implementation, version) in BATCH_SOURCE_BLOCKERS else "")
VLLM_RUST_UNAVAILABLE = "vLLM Rust source not available; set VLLM_RUST_SOURCE and run the Rust capture probe."
