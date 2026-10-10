# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compile actual Rust builder output and check complete string acceptance."""
import importlib.metadata
import json
import sys

import xgrammar as xgr


def main():
    assert importlib.metadata.version("xgrammar") == "0.2.8"
    requests = json.load(sys.stdin)
    assert requests, "no grammar cases received"
    # String-language qualification only. No model, tokenizer download, or GPU.
    tokenizer = xgr.TokenizerInfo([bytes([i]) for i in range(256)], stop_token_ids=[])
    compiler = xgr.GrammarCompiler(tokenizer, max_threads=1)
    results = []
    for request in requests:
        try:
            grammar = compiler.compile_structural_tag(json.dumps(request["grammar"]))
        except Exception as error:
            raise RuntimeError(f"compile {request['id']}: {error}") from error
        matcher = xgr.GrammarMatcher(grammar, terminate_without_stop_token=True)
        accepted = matcher.accept_string(request["input"])
        rejected_at = None
        if not accepted:
            diagnostic = xgr.GrammarMatcher(grammar, terminate_without_stop_token=True)
            for offset, character in enumerate(request["input"]):
                if not diagnostic.accept_string(character):
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
