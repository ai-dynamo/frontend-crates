# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate the corpus exported by kimi_structural_tag.rs with real XGrammar.

KIMI_GRAMMAR_CASES=/tmp/kimi.json cargo test -p dynamo-parsers-v2 --test kimi_structural_tag
uv run --with xgrammar==0.2.8 python parsers/v2/tests/check_kimi_xgrammar.py /tmp/kimi.json
"""
import json
import sys

import xgrammar as xgr

compiler = xgr.GrammarCompiler(xgr.TokenizerInfo([bytes([i]) for i in range(256)], stop_token_ids=[]))
checks = 0
for case in json.load(open(sys.argv[1])):
    grammar = xgr.Grammar.from_structural_tag(json.dumps(case["tag"]))
    compiled = compiler.compile_grammar(grammar)
    for expected, key in [(True, "accepted"), (False, "rejected")]:
        for sample in case[key]:
            matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
            accepted = matcher.accept_string(sample) and matcher.is_terminated()
            assert accepted == expected, (case["family"], key, sample, case["tag"])
            checks += 1
print(f"XGrammar accepted/rejected {checks} samples as expected")
