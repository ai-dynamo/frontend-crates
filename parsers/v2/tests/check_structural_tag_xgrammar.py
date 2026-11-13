#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Execute builder output with the shared lean CPU XGrammar verifier.

From the repository root, install the same environment used by required CI:
  python3 -m venv /tmp/structural-tag-venv
  /tmp/structural-tag-venv/bin/pip install --no-deps -r conformance/utils/structural-tag-requirements.in
  CONFORMANCE_GRAMMAR_PYTHON=/tmp/structural-tag-venv/bin/python cargo test --locked -p dynamo-parsers-v2 --test structural_tag_xgrammar -- --include-ignored --nocapture

The required rust job runs the ignored tests explicitly. The optional 0.2.7
compatibility job replaces only XGrammar and selects CONFORMANCE_GRAMMAR_VERSION;
the default version comes from the shared requirements file. Native TVM-FFI
bindings avoid importing torch, transformers, pydantic, or tensor-mask helpers.

Positive XML/DSML cases also exercise native Unified extraction across all
chunk boundaries. Grammar-negative cases do not require parser rejection:
schema validation and native recovery are separate contracts. Harmony checks
qualify grammar strings separately because its terminal <|call|> is engine-owned.
The local byte vocabulary qualifies strings, not production tokenizer masks.
"""

import importlib.util
import json
import os
import sys
from importlib.metadata import distribution, version
from pathlib import Path

from tvm_ffi.libinfo import load_lib_module


REQUIREMENTS = Path(__file__).resolve().parents[3] / "conformance/utils/structural-tag-requirements.in"
PINNED_VERSION = next(
    line.split("==", 1)[1]
    for line in REQUIREMENTS.read_text().splitlines()
    if line.startswith("xgrammar==")
)
EXPECTED_VERSION = os.environ.get("CONFORMANCE_GRAMMAR_VERSION", PINNED_VERSION)
if version("xgrammar") != EXPECTED_VERSION:
    raise RuntimeError(f"install XGrammar {EXPECTED_VERSION} for the selected verifier")
PACKAGE = Path(distribution("xgrammar").locate_file("xgrammar"))
LIBRARY = load_lib_module("xgrammar", "xgrammar_bindings", extra_lib_paths=[PACKAGE])
SPEC = importlib.util.spec_from_file_location(
    "structural_tag_xgrammar_native", PACKAGE / "tvm_ffi_binding" / "_ffi_api.py"
)
NATIVE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(NATIVE)


def main():
    requests = json.load(sys.stdin)
    if not requests:
        raise ValueError("no grammar cases received")
    tokenizer = NATIVE.TokenizerInfo([bytes([i]) for i in range(256)], 0, None, [], False)
    compiler = NATIVE.GrammarCompiler(tokenizer, 1, False, -1)
    results = []
    for request in requests:
        try:
            grammar = compiler.compile_structural_tag(json.dumps(request["grammar"]))
        except Exception as error:
            raise RuntimeError(f"compile {request['id']}: {error}") from error
        matcher = NATIVE.GrammarMatcher(grammar, None, True, 0, None)
        accepted = matcher.accept_string(request["input"], False)
        rejected_at = None
        if not accepted:
            diagnostic = NATIVE.GrammarMatcher(grammar, None, True, 0, None)
            for offset, character in enumerate(request["input"]):
                if not diagnostic.accept_string(character, False):
                    rejected_at = request["input"][: offset + 1]
                    break
        results.append({
            "id": request["id"],
            "accepted": accepted,
            "complete": accepted and matcher.is_completed(),
            "rejected_prefix": rejected_at,
        })
    json.dump(results, sys.stdout)


if __name__ == "__main__":
    main()
