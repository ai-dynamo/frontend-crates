# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Execute actual MiniMax builder output exported by minimax_structural_tag.rs."""
import importlib.metadata
import json
import sys

import xgrammar as xgr

assert importlib.metadata.version("xgrammar") == "0.2.8"
compiler = xgr.GrammarCompiler(xgr.TokenizerInfo([bytes([i]) for i in range(256)], stop_token_ids=[]))
checks = 0
with open(sys.argv[1]) as source:
    cases = json.load(source)
assert cases, "no exported grammar cases"
for case in cases:
    compiled = compiler.compile_grammar(xgr.Grammar.from_structural_tag(json.dumps(case["tag"])))
    for expected, key in [(True, "accepted"), (False, "rejected")]:
        for sample in case[key]:
            matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
            actual = matcher.accept_string(sample) and matcher.is_terminated()
            assert actual == expected, (case["family"], key, sample)
            checks += 1
print(f"XGrammar accepted/rejected {checks} samples as expected")
