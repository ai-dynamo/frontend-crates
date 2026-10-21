#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check a Unified request with the pinned CPU XGrammar matcher."""

import importlib.util
import json
import sys
from importlib.metadata import distribution, version
from pathlib import Path

from tvm_ffi.libinfo import load_lib_module


ENGINE_VERSION = next(
    line.split("==", 1)[1]
    for line in (Path(__file__).parents[1] / "structural-tag-requirements.in").read_text().splitlines()
    if line.startswith("xgrammar==")
)
if version("xgrammar") != ENGINE_VERSION:
    raise RuntimeError("install the pinned XGrammar verifier")
PACKAGE = Path(distribution("xgrammar").locate_file("xgrammar"))
# Load the wheel's native API directly: generation checking needs neither the
# tensor-mask helpers nor their torch/transformers convenience dependencies.
LIBRARY = load_lib_module("xgrammar", "xgrammar_bindings", extra_lib_paths=[PACKAGE])
SPEC = importlib.util.spec_from_file_location(
    "conformance_xgrammar_native", PACKAGE / "tvm_ffi_binding" / "_ffi_api.py"
)
NATIVE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(NATIVE)


def verify(job: dict) -> dict:
    grammar = NATIVE.Grammar.from_structural_tag(json.dumps(job["structural_tag"]))
    compiler = NATIVE.GrammarCompiler(NATIVE.TokenizerInfo([], 0, None, None, False), 1, False, -1)
    compiled = compiler.compile_grammar_ebnf(grammar)

    def accepts(chunks: list[str]) -> bool:
        matcher = NATIVE.GrammarMatcher(compiled, None, False, 0, None)
        return all(matcher.accept_string(chunk, False) for chunk in chunks) and matcher.is_completed()

    chunks = job["chunks"]
    rejected = job["rejected"]
    if not chunks or not all(isinstance(chunk, str) for chunk in chunks):
        raise ValueError("empty or invalid accepted stream")
    if not rejected or not all(isinstance(text, str) for text in rejected):
        raise ValueError("missing rejection probes")
    if not accepts(chunks):
        raise ValueError("grammar rejected the authored stream")
    text = "".join(chunks)
    for split in range(len(text) + 1):
        if not accepts([text[:split], text[split:]]):
            raise ValueError(f"grammar rejected accepted split {split}")
    for index, invalid in enumerate(rejected):
        if accepts([invalid]):
            raise ValueError(f"grammar accepted rejection probe {index}")
    return {"engine": "xgrammar", "version": ENGINE_VERSION, "checks": len(text) + 2 + len(rejected)}


if __name__ == "__main__":
    print(json.dumps(verify(json.load(sys.stdin))))
