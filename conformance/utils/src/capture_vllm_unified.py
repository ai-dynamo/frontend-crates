# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Live-capture vLLM 0.25.x parser output for the Unified conformance tab.

Runs INSIDE a vLLM container (needs `import vllm`). Reads a JSON job for one family on stdin:

    {"cases": [{"id": "...", "family": "gemma4", "input": "...",
                "chunks": ["<chunk1>", "<chunk2>", ...]}]}

The caller must submit one family per job because normalized Unified IDs repeat
across families. Mixed-family and duplicate-ID jobs are rejected. It writes on
stdout, per case:

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

from capture_stimulus import capture_input, unavailable_result, validate_family_scoped_cases
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


_PARSER_MARKERS = {
    "deepseek_v4": (
        "<think>",
        "</think>",
        "<｜DSML｜tool_calls>",
        "</｜DSML｜tool_calls>",
        "<｜DSML｜invoke>",
        "</｜DSML｜invoke>",
        "</｜DSML｜parameter>",
    ),
    "deepseek_v41": (
        "<think>",
        "</think>",
        "<｜DSML｜tool_calls>",
        "</｜DSML｜tool_calls>",
        "<｜DSML｜invoke>",
        "</｜DSML｜invoke>",
        "</｜DSML｜parameter>",
    ),
    "gemma4": (
        "<|channel>",
        "<channel|>",
        "<|tool_call>",
        "<tool_call|>",
        "<|turn>",
        "<|tool_response>",
    ),
    "glm47": (
        "<think>",
        "</think>",
        "<tool_call>",
        "</tool_call>",
        "<arg_key>",
        "</arg_key>",
        "<arg_value>",
        "</arg_value>",
    ),
    "kimi_k2": (
        "<think>",
        "</think>",
        "<|tool_calls_section_begin|>",
        "<|tool_calls_section_end|>",
        "<|tool_call_begin|>",
        "<|tool_call_end|>",
        "<|tool_call_argument_begin|>",
    ),
    "kimi_k3": (
        "<|open|>think<|sep|>",
        "<|close|>think<|sep|>",
        "<|open|>response<|sep|>",
        "<|close|>response<|sep|>",
        "<|close|>message<|sep|>",
    ),
    "qwen3": (
        "<think>",
        "</think>",
        "<tool_call>",
        "</tool_call>",
        "<function=",
        "</function>",
        "<parameter=",
        "</parameter>",
    ),
}


class StubTokenizer:
    """Deterministic marker tokenizer for parser-only capture.

    The parser APIs inspect token IDs when deciding whether a prompt ended inside
    reasoning. An empty vocabulary makes that state impossible to exercise, so
    this tokenizer exposes the exact marker strings used by each parser family
    with stable synthetic IDs. It still returns no IDs for ordinary model text.
    """
    all_special_tokens = []
    is_fast = True

    def __init__(self, family):
        markers = _PARSER_MARKERS[family]
        self._vocab = {marker: 256 + index for index, marker in enumerate(markers)}
        self._encoded = {
            marker: [token_id] for marker, token_id in self._vocab.items()
        }
        if family == "kimi_k3":
            self._encoded["<|open|>think<|sep|>"] = [256, 257, 258]
            self._encoded["<|close|>think<|sep|>"] = [259, 260, 261]
            self._encoded["<|open|>response<|sep|>"] = [262, 263, 264]
            self._encoded["<|close|>response<|sep|>"] = [265, 266, 267]
            self._encoded["<|close|>message<|sep|>"] = [268, 269, 270]

    def get_vocab(self):
        return self._vocab

    def convert_tokens_to_ids(self, t):
        return self._vocab.get(t)

    def get_added_vocab(self):
        return self._vocab

    def encode(self, t, add_special_tokens=False):
        return list(self._encoded.get(t, []))

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
    # vLLM 0.27 renamed DeltaMessage.reasoning_content to reasoning.  Keep the
    # compatibility fallback because the capture matrix includes older releases,
    # but prefer the current field so prefilled reasoning is not silently dropped.
    reasoning = getattr(dm, "reasoning", None)
    if reasoning is None:
        reasoning = getattr(dm, "reasoning_content", None)
    if reasoning:
        out.append({"kind": "reasoning", "text": reasoning})
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
    thinking_enabled = init.get("starting_state") != "Response"
    # The prompt already opened the visible response channel for Response rows.
    # Pass both spellings because Kimi K3 reads ``thinking`` while the other
    # engine adapters read ``enable_thinking``.
    chat_template_kwargs = {
        "thinking": thinking_enabled,
        "enable_thinking": thinking_enabled,
    }
    return ChatCompletionRequest(
        messages=[{"role": "user", "content": "x"}],
        tools=TOOLS,
        tool_choice=tool_choice,
        chat_template_kwargs=chat_template_kwargs or None,
    )


def _prompt_token_ids(case, family):
    """Return synthetic prompt IDs for a prefilled reasoning request."""
    init = case.get("init") or {}
    if init.get("starting_state") != "Reasoning":
        return None
    markers = _PARSER_MARKERS[family]
    tokenizer = StubTokenizer(family)
    if family == "kimi_k3":
        marker = "<|open|>think<|sep|>"
    elif family == "gemma4":
        marker = "<|channel>"
    else:
        marker = "<think>"
    if marker not in markers:
        raise ValueError(f"missing reasoning marker profile for {family}")
    return tokenizer.encode(marker, add_special_tokens=False)


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
                observation=True,
            )
            continue
        if cls is None:
            results[case["id"]] = unavailable_result(
                "vllm_parser_not_registered",
                f"vLLM parser manager has no parser for {fam}",
                observation=True,
            )
            continue
        if "tools" in case and case["tools"] != TOOL_SCHEMAS:
            results[case["id"]] = unavailable_result(
                "vllm_tool_schema_unsupported",
                "authored tools schema differs from the executable Unified schema",
            )
            continue
        req = _request(case)
        parser_kwargs = {"tools": TOOLS, "chat_template_kwargs": req.chat_template_kwargs}
        try:
            parser = cls(StubTokenizer(fam), **parser_kwargs)

            # Streaming is the only vLLM API that accepts prompt token IDs. A
            # prefilled reasoning row therefore gets its assembled projection
            # from the initialized stream rather than from parse(), which has no
            # prompt-state argument.
            p = cls(StubTokenizer(fam), **parser_kwargs)
            if hasattr(p, "initialize_streaming"):
                p.initialize_streaming()
            chunks = case.get("chunks", [])
            per_chunk = []
            prompt_ids = _prompt_token_ids(case, fam)
            for i, ch in enumerate(chunks):
                dm = p.parse_delta(
                    ch,
                    [],
                    req,
                    prompt_ids if i == 0 else None,
                    finished=(not case["terminal_step"] and i == len(chunks) - 1),
                )
                per_chunk.append(_delta_events(dm))
            if case["terminal_step"]:
                per_chunk.append(_delta_events(p.parse_delta("", [], req, None, finished=True)))

            if (case.get("init") or {}).get("starting_state") == "Reasoning":
                assembled = _assembled_from_chunks(per_chunk)
            else:
                # Batch: the real final-message fields, projected to an ordered
                # list. Response-prefilled rows are represented by request kwargs.
                reasoning, content, tool_calls = parser.parse(case["input"], req, True, None)
                assembled = _assembled_events(reasoning, content, tool_calls)
            results[case["id"]] = {"assembled": assembled, "chunks": per_chunk}
        except Exception as exc:  # per-case parser boundary; preserve the error
            results[case["id"]] = {
                "error": f"{type(exc).__name__}: {exc}",
                "capture_observation": {
                    "error": {
                        "code": "vllm_parser_error",
                        "detail": f"{type(exc).__name__}: {exc}",
                    }
                },
            }

    return results


def main():
    job = json.load(sys.stdin)
    validate_family_scoped_cases(job.get("cases", []))
    ready, results = [], {}

    def request_binding(case):
        chunks = case.get("chunks") or []
        return capture_input({**case, "chunks": [{"delta_text": c} for c in chunks]})

    def unavailable_for(case, code, detail):
        return unavailable_result(code, detail, capture=request_binding(case))

    for case in job.get("cases", []):
        if case["family"] not in FAMILY_PARSERS:
            results[case["id"]] = unavailable_result(
                "vllm_parser_not_registered",
                f"No released vLLM Python parser is registered for {case['family']}.",
                capture=request_binding(case),
                observation=True,
            )
            continue
        chunks = case.get("chunks") or []
        terminal_step = bool(
            chunks
            and chunks[-1] == FINISH_SENTINEL
            and "".join(chunks[:-1]) == case.get("input", "")
        )
        literal_input = "".join(chunks) == case.get("input", "")
        if "tools" in case and case["tools"] != TOOL_SCHEMAS:
            results[case["id"]] = unavailable_for(
                case,
                "vllm_tool_schema_unsupported",
                "authored tools schema differs from the executable Unified schema",
            )
            continue
        if not terminal_step and not literal_input:
            results[case["id"]] = unavailable_for(
                case,
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
                unavailable_result(
                    "vllm_capture_missing_result",
                    "vLLM capture returned no result",
                    capture=request_binding(case),
                    observation=True,
                ),
            )
            original_case = next(
                authored for authored in job.get("cases", []) if authored["id"] == case["id"]
            )
            result.setdefault("capture_input", request_binding(original_case))
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
