# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Single source of truth for conformance implementation identity (audit B1).

One `ImplSpec` per parser implementation. Every other module derives its keys,
legacy aliases, display names, and matrix markers from `IMPL_SPECS` instead of
re-declaring parallel tables that drift — the `V_rs` vs `V_rb`, `vllm` vs
`vllm_python`, missing-parser-option class of bug.

Consumers (`generate_conformance_table.py`, `build_stream_fixtures.py`,
`validate.py`, `tests/test_stream_on_batch.py`) all `import impls`. The generator
runs from the staged `tests/parity/` layout, so `_common.sh build_stage_conformance`
copies this file next to it; the other tools run from `conformance/utils/` directly.
"""
from __future__ import annotations

from dataclasses import dataclass


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
