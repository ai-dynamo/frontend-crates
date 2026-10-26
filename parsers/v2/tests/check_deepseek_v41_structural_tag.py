# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.metadata
import json
import sys

import xgrammar as xg


def accepts(compiled, text):
    matcher = xg.GrammarMatcher(compiled)
    return matcher.accept_string(text) and matcher.is_completed()


def check(probe, compiler):
    spaced = probe["family"] == "deepseek_v41"
    prefix = "｜DSML｜ " if spaced else "｜DSML｜"
    block = "calls" if spaced else "tool_calls"
    opening = f"<{prefix}{block}>\n"
    closing = f"</{prefix}{block}>"
    invoke = (
        f'<{prefix}invoke name="Bash">\n'
        f'<{prefix}parameter name="command" string="true">pwd</{prefix}parameter>\n'
        f'<{prefix}parameter name="timeout" string="false">1000</{prefix}parameter>\n'
        f"</{prefix}invoke>\n"
    )
    call = "\n\n" + opening + invoke + closing
    compiled = compiler.compile_structural_tag(json.dumps(probe["tag"]))
    choice = probe["choice"]
    if choice == "none":
        good, bad = ["ordinary answer"], [call]
    else:
        good = [call]
        if choice == "auto":
            good += ["ordinary answer", "explanation" + call]
        malformed = [
            invoke.replace('invoke name="Bash"', 'parameter name="Bash"', 1),
            invoke.replace('name="Bash"', 'name="Unknown"', 1),
            invoke.replace('name="command"', 'name="undeclared"', 1),
            invoke.replace('string="true">pwd', 'string="false">pwd', 1),
            invoke.replace('string="false">1000', 'string="true">1000', 1),
            invoke.replace('string="false">1000', 'string="false">false', 1),
            invoke.replace(f"</{prefix}invoke>", f"</{prefix}parameter>", 1),
            invoke.replace(f"</{prefix}parameter>\n", f"</{prefix}parameter>\nEOF\n", 1),
        ]
        bad = ["\n\n" + opening + body + closing for body in malformed]
        bad.append(call.removesuffix(closing))
        if choice != "auto":
            bad.append("ordinary answer")
        two_calls = "\n\n" + opening + invoke + invoke + closing
        if choice == "named" or not probe["parallel"]:
            bad.append(two_calls)
        else:
            good.append(two_calls)

        # A real token mask must reject an unknown tool name before it is sampled.
        matcher = xg.GrammarMatcher(compiled)
        assert matcher.accept_string("\n\n" + opening + f'<{prefix}invoke name="')
        mask = xg.allocate_token_bitmask(1, 256)
        matcher.fill_next_token_bitmask(mask)
        row = mask[0].tolist()

        def allowed(token):
            return bool((row[token // 32] >> (token % 32)) & 1)

        assert allowed(ord("B")) and not allowed(ord("X")), probe

    for text in good:
        assert accepts(compiled, text), (probe["family"], choice, "valid rejected", text)
    for text in bad:
        assert not accepts(compiled, text), (probe["family"], choice, "invalid accepted", text)
    return len(good), len(bad), int(choice != "none")


compiler = xg.GrammarCompiler(xg.TokenizerInfo([bytes([i]) for i in range(256)]))
results = [check(probe, compiler) for probe in json.load(sys.stdin)]
print(json.dumps({
    "xgrammar": importlib.metadata.version("xgrammar"),
    "compiled_grammars": len(results),
    "valid_accepted": sum(good for good, _, _ in results),
    "invalid_rejected": sum(bad for _, bad, _ in results),
    "token_masks_checked": sum(masks for _, _, masks in results),
}))
