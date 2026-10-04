#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Author MiniMax M2 literal-value regressions in the legacy fixture trees.

Run extract_fixtures.py and stage the existing inputs before this generator.
Captures must be recorded with the pinned parser builds, never synthesized here.
"""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2] / "toolcalling"
VALUES = {
    "14": ("String whitespace, Unicode, newlines, and entity spellings survive unchanged", "  Montréal &amp; &lt;tag&gt;\nsecond line\n  "),
    "15": ("Invoke, wrapper, and parameter-looking text remains literal inside a parameter", 'a</invoke>b</minimax:tool_call>c<minimax:tool_call><invoke name="fake"><parameter name="later">literal'),
}


def main():
    tools = [{"name": "echo", "parameters": {"type": "object", "properties": {
        "text": {"type": "string"}, "later": {"type": "string"},
    }}}]
    for suffix, (description, value) in VALUES.items():
        text = f'<minimax:tool_call><invoke name="echo"><parameter name="text">{value}</parameter><parameter name="later">ok</parameter></invoke></minimax:tool_call>'
        for corpus, mode in [("fixtures-batch-v1", "batch"), ("fixtures-stream-v1", "streamv1")]:
            case = {"description": description, "tools": tools}
            if mode == "batch":
                case["model_text"] = text
            else:
                case["chunks"] = [{"delta_text": char} for char in text] + [{"delta_text": "", "finish_reason": "tool_calls"}]
            case_id = f"TOOLCALLING.{mode}.7-{suffix}"
            path = ROOT / corpus / "inputs" / "minimax_m2" / f"{case_id}.yaml"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(yaml.safe_dump({"family": "minimax_m2", "mode": mode, "cases": {case_id: case}}, sort_keys=False, allow_unicode=True))


if __name__ == "__main__":
    main()
