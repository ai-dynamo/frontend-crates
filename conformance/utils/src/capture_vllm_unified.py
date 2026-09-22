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

from capture_stimulus import capture_input, unavailable_result
from unified_tools import unified_tools

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.parser.parser_manager import ParserManager

# family -> (reasoning_parser_name, tool_parser_name) for the released Python
# parser registrations used by the Unified corpus. DeepSeek V4.1 shares the
# `deepseek_v4` registration in vLLM 0.27.1.
FAMILY_PARSERS = {
    "deepseek_v4": ("deepseek_v4", "deepseek_v4"),
    "deepseek_v41": ("deepseek_v4", "deepseek_v4"),
    "gemma4": ("gemma4", "gemma4"),
    "kimi_k3": ("kimi_k3", "kimi_k3"),
    "qwen3": ("qwen3", "qwen3_coder"),
    "glm47": ("glm47", "glm47"),
    "kimi_k2": ("kimi_k2", "kimi_k2"),
}

TOOLS = [
    {"type": "function", "function": tool} for tool in unified_tools()
]
TOOL_SCHEMAS = [tool["function"] for tool in TOOLS]
FINISH_SENTINEL = "‹finish›"


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


def _request(case):
    """Build the request that the authored Unified row describes.

    vLLM's parser reads tool choice and chat-template kwargs from the request,
    so using one hard-coded ``auto`` request would silently turn the GuidedJson,
    named-tool, and response-prefilled rows into unrelated captures.
    """
    init = case.get("init") or {}
    mode = init.get("tool_output_mode", "Native")
    named_tool = init.get("named_tool")
    if mode == "GuidedJson":
        if named_tool:
            tool_choice = {"type": "function", "function": {"name": named_tool}}
        else:
            tool_choice = "required"
    else:
        tool_choice = "auto"
    chat_template_kwargs = {}
    if init.get("starting_state") == "Response":
        # The prompt already opened the visible response channel.  vLLM uses
        # this request flag for the same no-thinking state.
        chat_template_kwargs["enable_thinking"] = False
    return ChatCompletionRequest(
        messages=[{"role": "user", "content": "x"}],
        tools=TOOLS,
        tool_choice=tool_choice,
        chat_template_kwargs=chat_template_kwargs or None,
    )


def _unsupported_parser_request(parser, case):
    """Return a typed reason when the direct vLLM parser API cannot run a row."""
    init = case.get("init") or {}
    starting_state = init.get("starting_state", "None")
    if starting_state != "None":
        return (
            "vllm_starting_state_unsupported",
            f"vLLM Python parser API has no request-prefilled {starting_state} starting-state input.",
        )
    if case.get("finish_reason", "stop") != "stop":
        return (
            "vllm_finish_reason_unsupported",
            "vLLM Python parser API capture supports only stop termination.",
        )
    if init.get("named_tool"):
        return (
            "vllm_named_tool_unsupported",
            "vLLM request capture requires named_tool to be paired with GuidedJson.",
        )
    if init.get("tool_output_mode", "Native") == "GuidedJson":
        tool_parser = getattr(parser, "tool_parser", None)
        if tool_parser is None or not tool_parser.supports_required_and_named:
            return (
                "vllm_guided_json_unsupported",
                "this vLLM release does not expose required/named GuidedJson parsing for the selected parser",
            )
    return None


def _assembled_from_chunks(chunks):
    """Coalesce vLLM stream deltas into the final event projection."""
    events = []
    for delta in chunks:
        for event in delta:
            if event["kind"] != "tool_call":
                if events and events[-1]["kind"] == event["kind"]:
                    events[-1]["text"] += event["text"]
                else:
                    events.append(dict(event))
                continue
            if (
                events
                and events[-1]["kind"] == "tool_call"
                and event.get("name") is None
            ):
                prior = events[-1]
                fragment = event.get("arguments")
                if fragment is not None:
                    if prior.get("arguments") is None:
                        prior["arguments"] = fragment
                    elif isinstance(prior["arguments"], str):
                        prior["arguments"] += fragment
                continue
            events.append({
                "kind": "tool_call",
                "name": event.get("name") or "",
                "arguments": event.get("arguments"),
            })
    for event in events:
        if event["kind"] == "tool_call":
            event["arguments"] = _tool_args_json(event.get("arguments"))
    return events


def _capture_cases(cases):
    mgr = ParserManager()
    results = {}
    for case in cases:
        fam = case["family"]
        if fam not in FAMILY_PARSERS:
            continue
        rn, tn = FAMILY_PARSERS[fam]
        try:
            cls = mgr.get_parser(tool_parser_name=tn, reasoning_parser_name=rn,
                                 enable_auto_tools=True, model_name=fam)
        except (KeyError, TypeError):
            results[case["id"]] = unavailable_result(
                "vllm_parser_not_registered",
                f"vLLM {rn} parser is not registered for {fam}",
            )
            continue
        if cls is None:
            results[case["id"]] = unavailable_result(
                "vllm_parser_not_registered",
                f"vLLM parser manager has no parser for {fam}",
            )
            continue
        if "tools" in case and case["tools"] != TOOL_SCHEMAS:
            results[case["id"]] = unavailable_result(
                "vllm_tool_schema_unsupported",
                "authored tools schema differs from the executable Unified schema",
            )
            continue
        parser = cls(StubTokenizer(), tools=TOOLS)
        reason = _unsupported_parser_request(parser, case)
        if reason is not None:
            results[case["id"]] = unavailable_result(*reason)
            continue
        req = _request(case)
        parser_kwargs = {"tools": TOOLS}
        # Batch: the real final-message fields, projected to an ordered list.
        reasoning, content, tool_calls = parser.parse(case["input"], req, True, None)
        assembled = _assembled_events(reasoning, content, tool_calls)

        # Streaming: real per-chunk deltas.
        p = cls(StubTokenizer(), **parser_kwargs)
        if hasattr(p, "initialize_streaming"):
            p.initialize_streaming()
        chunks = case.get("chunks", [])
        per_chunk = []
        for i, ch in enumerate(chunks):
            dm = p.parse_delta(ch, [], req, [], finished=(not case["terminal_step"] and i == len(chunks) - 1))
            per_chunk.append(_delta_events(dm))
        if case["terminal_step"]:
            per_chunk.append(_delta_events(p.parse_delta("", [], req, [], finished=True)))
        results[case["id"]] = {"assembled": assembled, "chunks": per_chunk}

    return results


def main():
    job = json.load(sys.stdin)
    ready, results = [], {}
    for case in job.get("cases", []):
        if case["family"] not in FAMILY_PARSERS:
            continue
        chunks = case.get("chunks") or []
        terminal_step = bool(
            chunks
            and chunks[-1] == FINISH_SENTINEL
            and "".join(chunks[:-1]) == case.get("input", "")
        )
        literal_input = "".join(chunks) == case.get("input", "")
        if "tools" in case and case["tools"] != TOOL_SCHEMAS:
            results[case["id"]] = unavailable_result(
                "vllm_tool_schema_unsupported",
                "authored tools schema differs from the executable Unified schema",
            )
            continue
        if not terminal_step and not literal_input:
            results[case["id"]] = unavailable_result(
                "vllm_chunk_schedule_unsupported",
                "authored chunk schedule is not a literal input followed by an explicit finish step",
            )
            continue
        ready.append({
            **case,
            "chunks": chunks[:-1] if terminal_step else chunks,
            "terminal_step": terminal_step,
        })
    if ready:
        captured = _capture_cases(ready)
        for case in ready:
            result = captured.get(
                case["id"],
                unavailable_result("vllm_capture_missing_result", "vLLM capture returned no result"),
            )
            if "capture_stimulus" not in result:
                captured_chunks = list(case["chunks"])
                if case["terminal_step"]:
                    captured_chunks.append(FINISH_SENTINEL)
                result["capture_input"] = capture_input(
                    {**case, "chunks": [{"delta_text": c} for c in captured_chunks]}
                )
            results[case["id"]] = result

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
