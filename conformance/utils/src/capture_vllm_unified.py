# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Live-capture versioned vLLM parser output for the Unified conformance tab.

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
tokenizer with deterministic marker IDs is enough.
"""
import json
import re
import sys
import traceback

import vllm
import yaml

from capture_stimulus import batch_final_events, capture_input, unavailable_result, validate_family_scoped_cases
from unified_tools import openai_tools

from peer_modules import AdapterContractError, resolve_peer_module, vllm_parser_factory, vllm_stream_is_legacy, parser_error_detail
from peer_modules import capture_parser

ChatCompletionRequest = resolve_peer_module("vllm", "protocol").ChatCompletionRequest
_PARSER_FACTORY = vllm_parser_factory()

# family -> (reasoning_parser_name, tool_parser_name) for the released Python
# parser registrations used by the Unified corpus. DeepSeek V4.1 shares the
# `deepseek_v4` registration in vLLM 0.27.1.
from impls import VLLM_TOKENIZER_MARKERS, VLLM_UNIFIED_PIPELINES as FAMILY_PARSERS

TOOLS = openai_tools()
FINISH_SENTINEL = "‹finish›"


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
        self.source_text = None
        self.marker_at_start = False
        markers = VLLM_TOKENIZER_MARKERS[family]
        self._vocab = {marker: 256 + index for index, marker in enumerate(markers)}
        self._encoded = {
            marker: [token_id] for marker, token_id in self._vocab.items()
        }
        if family == "kimi_k3":
            self._vocab = {token: 256 + index for index, token in enumerate(
                ("<|open|>", "<|close|>", "<|sep|>", "think", "response", "message"))}
            self._encoded.update({token: [token_id] for token, token_id in self._vocab.items()})
            self._encoded["<|open|>think<|sep|>"] = [256, 259, 258]
            self._encoded["<|close|>think<|sep|>"] = [257, 259, 258]
            self._encoded["<|open|>response<|sep|>"] = [256, 260, 258]
            self._encoded["<|close|>response<|sep|>"] = [257, 260, 258]
            self._encoded["<|close|>message<|sep|>"] = [257, 261, 258]
        self._patterns = []
        for marker, ids in self._encoded.items():
            pattern = re.escape(marker)
            if family == "kimi_k3":
                # vLLM's XTML decoder may insert spaces between marker tokens.
                pattern = pattern.replace("think", r"\s*think\s*")
                pattern = pattern.replace("response", r"\s*response\s*")
                pattern = pattern.replace("message", r"\s*message\s*")
            self._patterns.append((re.compile(pattern), ids))

    def get_vocab(self):
        return self._vocab

    def convert_tokens_to_ids(self, t):
        return self._vocab.get(t)

    def get_added_vocab(self):
        return self._vocab

    def encode(self, t, add_special_tokens=False):
        return [token for _start, _end, tokens in self.token_spans(t) for token in tokens]

    def token_spans(self, text):
        matches = sorted((match.start(), -len(match.group()), match.end(), ids)
                         for pattern, ids in self._patterns for match in pattern.finditer(text))
        spans, cursor = [], 0
        for start, _length, end, ids in matches:
            if start < cursor:
                continue
            spans.extend((index, index + 1, [1000 + ord(text[index])]) for index in range(cursor, start))
            spans.append((start, end, ids))
            cursor = end
        spans.extend((index, index + 1, [1000 + ord(text[index])]) for index in range(cursor, len(text)))
        return spans

    def bind_input(self, text, *, marker_at_start):
        self.source_text, self.marker_at_start = text, marker_at_start

    def generated_ids(self, previous_text, current_text):
        """Expose completed control markers at their authored chunk positions.

        DelegatingParser uses IDs to leave reasoning even when its detector
        extracts text correctly. A split marker must not transition early.
        Ordinary text gets an opaque ID: its vocabulary is irrelevant here.
        """
        if self.marker_at_start:
            if self.source_text is None or not self.source_text.startswith(current_text):
                raise AdapterContractError("Token engine input differs from the bound authored text")
            # A token can precede its fully decoded spelling; the published
            # scanner explicitly defers it until the corresponding text arrives.
            return [token for start, _end, ids in self.token_spans(self.source_text)
                    if len(previous_text) <= start < len(current_text) for token in ids]
        return [token for _start, end, ids in self.token_spans(current_text)
                if end > len(previous_text) for token in ids]

    def decode(self, ids, **k):
        markers = {token_id: marker for marker, token_id in self._vocab.items()}
        return "".join(chr(token_id - 1000) if token_id >= 1000 else markers[token_id] for token_id in ids)

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
    calls = []
    for tc in tool_calls or []:
        fn = _fn(tc)
        calls.append({"name": fn.name or "", "arguments": _tool_args_json(fn.arguments)})
    return batch_final_events({"reasoning_text": reasoning or "", "normal_text": content or "", "calls": calls})


def _delta_events(dm):
    """One parse_delta DeltaMessage -> raw per-chunk unified deltas."""
    out = []
    if dm is None:
        return out
    if not any(hasattr(dm, field) for field in ("content", "reasoning", "reasoning_content", "tool_calls")):
        raise AdapterContractError("Published delta API returned an unexpected shape")
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
    tools = openai_tools(case.get("tools"))
    return ChatCompletionRequest(
        messages=[{"role": "user", "content": "x"}],
        tools=tools,
        tool_choice=tool_choice,
        chat_template_kwargs=chat_template_kwargs or None,
    )


def _prompt_token_ids(case, family):
    init = case.get("init") or {}
    if init.get("starting_state") != "Reasoning":
        return None
    markers = VLLM_TOKENIZER_MARKERS[family]
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
    results = {}
    for case in cases:
        fam = case["family"]
        if fam not in FAMILY_PARSERS:
            continue
        rn, tn = FAMILY_PARSERS[fam]
        try:
            cls = _PARSER_FACTORY(tool_parser_name=tn, reasoning_parser_name=rn,
                                 enable_auto_tools=True, model_name=fam)
        except (KeyError, TypeError) as error:
            # Published managers can wrap a lazy dependency failure as a
            # registration error; it still blocks execution of the release.
            parser_error_detail(error)
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
        if not hasattr(cls, "parse") and (case.get("init") or {}).get("tool_output_mode", "Native") == "GuidedJson":
            results[case["id"]] = unavailable_result("unsupported_request_mode",
                "Published manager exposes only native extraction; structured serving uses a separate controller path",
                observation=True)
            continue
        req = _request(case)
        parser_kwargs = {
            "tools": openai_tools(case.get("tools")),
            "chat_template_kwargs": req.chat_template_kwargs,
        }
        per_chunk = []
        try:
            full_tokenizer = StubTokenizer(fam)
            parser = capture_parser(cls, full_tokenizer, **parser_kwargs)
            full_tokenizer.bind_input(case["input"], marker_at_start=parser.token_engine)

            # Streaming is the only vLLM API that accepts prompt token IDs. A
            # prefilled reasoning row therefore gets its assembled projection
            # from the initialized stream rather than from parse(), which has no
            # prompt-state argument.
            tokenizer = StubTokenizer(fam)
            p = capture_parser(cls, tokenizer, **parser_kwargs)
            tokenizer.bind_input(case["input"], marker_at_start=p.token_engine)
            if hasattr(p, "initialize_streaming"):
                p.initialize_streaming()
            chunks = case.get("chunks", [])
            per_chunk = []
            prompt_ids = _prompt_token_ids(case, fam)
            previous_text = ""
            for i, ch in enumerate(chunks):
                current_text = previous_text + ch
                dm = p.parse_delta(
                    ch,
                    tokenizer.generated_ids(previous_text, current_text),
                    req,
                    prompt_ids if i == 0 else None,
                    finished=(not case["terminal_step"] and i == len(chunks) - 1),
                )
                per_chunk.append(_delta_events(dm))
                previous_text = current_text
            if case["terminal_step"]:
                per_chunk.append(_delta_events(p.parse_delta("", [], req, None, finished=True)))

            if (case.get("init") or {}).get("starting_state") == "Reasoning":
                assembled = _assembled_from_chunks(per_chunk)
            else:
                # Batch: the real final-message fields, projected to an ordered
                # list. Response-prefilled rows are represented by request kwargs.
                reasoning, content, tool_calls = parser.parse(case["input"], req, True)
                assembled = _assembled_events(reasoning, content, tool_calls)
            results[case["id"]] = {"assembled": assembled, "chunks": per_chunk}
        except Exception as exc:  # per-case parser supervisor; retain concrete failure evidence
            detail = parser_error_detail(exc)
            results[case["id"]] = {
                "error": detail,
                "error_traceback": traceback.format_exc(),
                "partial_chunks": per_chunk,
                "capture_observation": {
                    "error": {
                        "code": "vllm_parser_error",
                        "detail": detail,
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
        if case.get("finish_reason", "stop") not in {"stop", "length"}:
            results[case["id"]] = unavailable_for(case, "peer_termination_unsupported",
                "Published parser capture supports serving stop/length termination only")
            continue
        if case["family"] not in FAMILY_PARSERS:
            results[case["id"]] = unavailable_result(
                "vllm_parser_not_registered",
                f"No released vLLM Python parser is registered for {case['family']}.",
                capture=request_binding(case),
                observation=True,
            )
            continue
        if vllm_stream_is_legacy() and (case.get("init") != {"starting_state": "None", "tool_output_mode": "Native", "named_tool": None}
                                      or case.get("finish_reason", "stop") not in {"stop", "length"}):
            results[case["id"]] = unavailable_for(case, "unsupported_request_mode", "Published legacy serving composition supports native/default initialization and stop/length termination")
            continue
        chunks = case.get("chunks") or []
        terminal_step = bool(
            chunks
            and chunks[-1] == FINISH_SENTINEL
            and "".join(chunks[:-1]) == case.get("input", "")
        )
        literal_input = "".join(chunks) == case.get("input", "")
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
    return vllm.__version__


if __name__ == "__main__":
    main()
