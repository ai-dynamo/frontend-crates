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


BASELINE_IMPL = "dynamo_rust"

# Order is the canonical column order across the matrix. vLLM Rust is stream-only
# (no batch parser exists), so it is absent from the batch tabs. SGLang's language
# subscript is `r` by existing convention (not `p`) — preserved deliberately.
IMPL_SPECS: tuple[ImplSpec, ...] = (
    ImplSpec("dynamo_rust", "dynamo", "dynamo", "rust", "Dynamo Rust", "D", "r", ("batch", "stream")),
    ImplSpec("vllm_rust", None, "vllm", "rust", "vLLM Rust", "R", "r", ("stream",)),
    ImplSpec("vllm_python", "vllm", "vllm", "python", "vLLM Python", "V", "p", ("batch", "stream")),
    ImplSpec("sglang_python", "sglang", "sglang", "python", "SGLang Python", "S", "r", ("batch", "stream")),
)

IMPL_KEYS: tuple[str, ...] = tuple(s.key for s in IMPL_SPECS)
STREAM_IMPL_KEYS: tuple[str, ...] = tuple(s.key for s in IMPL_SPECS if "stream" in s.modes)
BATCH_IMPL_KEYS: tuple[str, ...] = tuple(s.key for s in IMPL_SPECS if "batch" in s.modes)
PEER_IMPL_KEYS: tuple[str, ...] = tuple(k for k in IMPL_KEYS if k != BASELINE_IMPL)
LEGACY_IMPL_ALIASES: dict[str, str] = {s.legacy_key: s.key for s in IMPL_SPECS if s.legacy_key}
IMPL_DISPLAY: dict[str, str] = {s.key: s.display for s in IMPL_SPECS}
ENGINE_LETTER: dict[str, str] = {s.key: s.marker_letter for s in IMPL_SPECS}
IMPL_LANG_MARKER: dict[str, str] = {s.key: s.marker_lang for s in IMPL_SPECS}
# Aliases validate.py accepts on a `--impl` flag: peers only (Dynamo is not validated
# via validate.py; it goes through cargo test).
FIXTURE_IMPL_ALIASES: dict[str, str] = {
    s.legacy_key: s.key for s in IMPL_SPECS if s.legacy_key and s.key != BASELINE_IMPL
}


def canonical_impl_key(impl: str) -> str:
    return LEGACY_IMPL_ALIASES.get(impl, impl)


# B11: the capture wrapper stamps this marker into an `unavailable` reason when a
# peer parser was invoked and THREW (as opposed to not running). The renderer
# matches this SAME constant — a shared contract, not a private guessed regex — to
# classify the `✗` error marker. Going forward, captures can also record a
# structured `error: {kind, message}` block, which the renderer renders as `✗`
# directly; a plain-string `error` stays a declared expected-error (`!`).
PARSER_NOT_CAPTURED = "parser not captured"
