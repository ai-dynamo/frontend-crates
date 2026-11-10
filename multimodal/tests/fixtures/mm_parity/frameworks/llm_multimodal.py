# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""llm-multimodal, the Rust preprocessor behind vLLM's Rust frontend and smg,
through the harness in `llm-mm-harness/`, which README.md describes."""

import hashlib
from pathlib import Path

from frameworks import worker

HARNESS = Path(__file__).resolve().parent.parent / "llm-mm-harness"


def run(request, args):
    response = worker([str(args.llm_mm_harness)], request)
    files = ("Cargo.toml", "Cargo.lock", "src/main.rs")
    source = hashlib.sha256(b"".join((HARNESS / name).read_bytes() for name in files)).hexdigest()
    built = response["versions"]["harness_sha256"]
    assert built == source, f"{args.llm_mm_harness}: build it again from {HARNESS}"
    return response
