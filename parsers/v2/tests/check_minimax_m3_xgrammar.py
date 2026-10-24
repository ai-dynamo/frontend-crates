# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Execute the native M3 grammars exported by minimax_m3_structural_tag.rs.

Install the serving image's XGrammar version, then pass the
MINIMAX_M3_XGRAMMAR_CASES JSON path. Tested with 0.2.7 and 0.2.8.
The byte vocabulary makes this a CPU-only grammar acceptance test.
An optional second path supplies serialized TokenizerInfo from the pinned M3
tokenizer for exact token-mask checks; no model weights are needed.
"""
import importlib.metadata
import json
import sys

import xgrammar as xgr

version = importlib.metadata.version("xgrammar")
compiler = xgr.GrammarCompiler(
    xgr.TokenizerInfo([bytes([i]) for i in range(256)], stop_token_ids=[])
)
with open(sys.argv[1]) as source:
    cases = json.load(source)
assert cases, "no exported grammar cases"
checks = 0
token_checks = 0
tokenizer_info = None
if len(sys.argv) > 2:
    with open(sys.argv[2]) as source:
        tokenizer_info = xgr.TokenizerInfo.deserialize_json(source.read())
    token_compiler = xgr.GrammarCompiler(tokenizer_info)
    marker_ids = {
        marker: tokenizer_info.decoded_vocab.index(marker.encode())
        for marker in ["<mm:think>", "</mm:think>"]
    }
for case in cases:
    if tokenizer_info is not None and case.get("token_masks"):
        token_grammar = token_compiler.compile_structural_tag(json.dumps(case["tag"]))
        for check in case["token_masks"]:
            matcher = xgr.GrammarMatcher(token_grammar)
            assert matcher.accept_string(check["prefix"]), (case["case"], check["prefix"])
            mask = xgr.allocate_token_bitmask(1, tokenizer_info.vocab_size)
            matcher.fill_next_token_bitmask(mask)
            for marker, expected in check["tokens"].items():
                token = marker_ids[marker]
                actual = bool(int(mask[0, token // 32].item()) & (1 << (token % 32)))
                assert actual == expected, (case["case"], check["prefix"], marker, token, actual)
                token_checks += 1
    compiled = compiler.compile_grammar(xgr.Grammar.from_structural_tag(json.dumps(case["tag"])))
    for expected, key in [(True, "accepted"), (False, "rejected")]:
        for sample in case[key]:
            matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
            actual = matcher.accept_string(sample) and matcher.is_terminated()
            assert actual == expected, (case["case"], key, sample)
            checks += 1
print(f"XGrammar {version} checked {checks} accepted/rejected samples and {token_checks} token masks across {len(cases)} M3 grammars")
