# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Live-capture vLLM 0.25.x parser output for the Unified conformance tab.

Runs INSIDE a vLLM container (needs `import vllm`). Reads a JSON job on stdin:

    {"cases": [{"id": "...", "family": "gemma4", "input": "...",
                "chunks": ["<chunk1>", "<chunk2>", ...]}]}

and writes on stdout, per case:

    {"results": {"<id>": {
        "assembled": [ {kind: reasoning|text|tool_call, ...} ],   # batch parse()
        "chunks":    [ [ {kind,...}, ... ], ... ]                  # per-chunk parse_delta
    }}}

Batch `parse()` gives the real FINAL-MESSAGE fields (reasoning, content, tools)
projected to an ordered event list; streaming `parse_delta` gives the real
per-chunk deltas. No GPU / model needed — the parser only lexes text, so a stub
tokenizer (empty vocab -> markers matched as text) is enough.
"""
import json
import sys
import yaml

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.parser.parser_manager import ParserManager

# family -> (reasoning_parser_name, tool_parser_name) shared by the released
# 0.25.1 and 0.26.0 Python captures.
FAMILY_PARSERS = {
    "gemma4": ("gemma4", "gemma4"),
    "qwen3": ("qwen3", "qwen3_coder"),
    "kimi_k2": ("kimi_k2", "kimi_k2"),
}

# Tool schemas the seed cases reference (string params), so arg typing matches.
TOOLS = [
    {"type": "function", "function": {"name": n, "parameters": {
        "type": "object", "properties": {k: {"type": "string"}}}}}
    # Must match `tools()` in conformance/tests/unified_parity.rs, or the `log`
    # cases (UNIFIED.12.a / 12.c) capture a harness-induced dropped call.
    for n, k in (("get_weather", "city"), ("f", "x"), ("g", "y"), ("run", "cmd"),
                 ("log", "note"))
]


class StubTokenizer:
    """Text-only tokenizer: empty vocab so every marker is matched as text."""
    all_special_tokens = []
    is_fast = True

    def get_vocab(self):
        return {}

    def convert_tokens_to_ids(self, t):
        return None

    def get_added_vocab(self):
        return {}

    def encode(self, t, add_special_tokens=False):
        return []

    def decode(self, ids, **k):
        return ""

    @property
    def vocab_size(self):
        return 0


def _fn(tc):
    return getattr(tc, "function", tc)


def _tool_args_json(args):
    if not args:
        return {}
    try:
        return json.loads(args)
    except (ValueError, TypeError):
        return args


def _assembled_events(reasoning, content, tool_calls):
    """Project vLLM's (reasoning, content, tool_calls) final message to events."""
    events = []
    if reasoning:
        events.append({"kind": "reasoning", "text": reasoning})
    if content:
        events.append({"kind": "text", "text": content})
    for tc in tool_calls or []:
        fn = _fn(tc)
        events.append({"kind": "tool_call",
                       "name": getattr(fn, "name", None) or "",
                       "arguments": _tool_args_json(getattr(fn, "arguments", None))})
    return events


def _delta_events(dm):
    """One parse_delta DeltaMessage -> raw per-chunk unified deltas."""
    out = []
    if dm is None:
        return out
    if getattr(dm, "reasoning_content", None):
        out.append({"kind": "reasoning", "text": dm.reasoning_content})
    if getattr(dm, "content", None):
        out.append({"kind": "text", "text": dm.content})
    for tc in getattr(dm, "tool_calls", None) or []:
        fn = _fn(tc)
        out.append({"kind": "tool_call",
                    "name": getattr(fn, "name", None),
                    "arguments": getattr(fn, "arguments", None)})
    return out


def main():
    job = json.load(sys.stdin)
    mgr = ParserManager()
    req = ChatCompletionRequest(messages=[{"role": "user", "content": "x"}],
                                tools=TOOLS, tool_choice="auto")
    results = {}
    for case in job.get("cases", []):
        fam = case["family"]
        if fam not in FAMILY_PARSERS:
            continue
        rn, tn = FAMILY_PARSERS[fam]
        cls = mgr.get_parser(tool_parser_name=tn, reasoning_parser_name=rn,
                             enable_auto_tools=True, model_name=fam)
        if cls is None:
            continue

        # Batch: the real final-message fields, projected to an ordered list.
        reasoning, content, tool_calls = cls(StubTokenizer()).parse(
            case["input"], req, True, None)
        assembled = _assembled_events(reasoning, content, tool_calls)

        # Streaming: real per-chunk deltas.
        p = cls(StubTokenizer())
        if hasattr(p, "initialize_streaming"):
            p.initialize_streaming()
        chunks = case.get("chunks", [])
        per_chunk = []
        for i, ch in enumerate(chunks):
            dm = p.parse_delta(ch, [], req, [], finished=(i == len(chunks) - 1))
            per_chunk.append(_delta_events(dm))

        results[case["id"]] = {"assembled": assembled, "chunks": per_chunk}

    # YAML to match the conformance fixture corpus. Container stdout is log-polluted,
    # so a recapture writes this to a file (or strips lines before the first top-level
    # key) rather than grepping a single JSON line.
    yaml.dump({"results": results, "vllm_version": _vllm_version()}, sys.stdout,
              default_flow_style=False, sort_keys=False, allow_unicode=True, width=4096)


def _vllm_version():
    try:
        import vllm
        return vllm.__version__
    except Exception:
        return "unknown"


if __name__ == "__main__":
    main()
