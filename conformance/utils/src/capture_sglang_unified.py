# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Capture SGLang's serving-order reasoning and function-call streaming detectors.

Jobs are family-scoped. Unsupported detector registrations and requests retain
explicit provenance instead of being silently skipped or mapped to another family.
"""
import inspect
import json
import sys
import traceback
from importlib.metadata import version

import yaml

from capture_stimulus import capture_peer_results, validate_family_scoped_cases
from unified_tools import unified_tools

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.parser.reasoning_parser import ReasoningParser

# family -> (reasoning model_type, tool_call_parser). Kimi K2.5 uses <think>; SGLang's
# `kimi_k2` reasoning detector matches the golden corpus.
from impls import SGLANG_UNIFIED_PIPELINES as FAMILY_PARSERS

TOOL_SCHEMAS = unified_tools()


# Here the reasoning parser's normal text always feeds a tool parser, which is exactly
# what `tool_call_parser_active` declares; SGLang's own serving_chat passes it the same
# way. Channel-framed formats (muse) need it: without it the reasoning detector unwraps
# the `to=user` channel, and the tool detector then reads that leading prose as "no
# header is coming", never resyncs on the later tool channel, and drops the call.
#
# The kwarg arrived WITH muse (PR #34262), so released SGLang — including the 0.5.14 that
# captured the committed shard — has no such parameter and raises `TypeError` on it.
# Probe the signature the same way SGLang forwards the flag to its own detectors, or a
# recapture on a muse-less SGLang would error out every case of every family.
_ACTIVE_KWARG = (
    {"tool_call_parser_active": True}
    if "tool_call_parser_active" in inspect.signature(ReasoningParser.__init__).parameters
    else {}
)


def _tool_stream_delta(item):
    """Streaming tool delta: keep the RAW (possibly partial) argument fragment so the
    consumer can join fragments across chunks — don't json.loads a partial blob."""
    d = item.model_dump()
    return {"kind": "tool_call", "name": d.get("name"),
            "arguments": d.get("parameters") if d.get("parameters") is not None else d.get("arguments")}


def _registration_unavailable(case):
    if case.get("finish_reason", "stop") not in {"stop", "length"}:
        return "peer_termination_unsupported", "SGLang harness supports serving stop/length termination only"
    rname, tname = FAMILY_PARSERS[case["family"]]
    missing = []
    if rname not in ReasoningParser.DetectorMap:
        missing.append(f"ReasoningParser {rname}")
    if tname not in FunctionCallParser.ToolCallParserEnum:
        missing.append(f"FunctionCallParser {tname}")
    if missing:
        return "peer_detector_unavailable", "SGLang " + version("sglang") + " does not register " + ", ".join(missing)
    return None


def _remaining_tool_args(fp):
    """Mirror serving_chat's final argument suffix emission, including its guard."""
    detector = fp.detector
    if not hasattr(detector, "prev_tool_call_arr") or not detector.prev_tool_call_arr:
        return []
    if not hasattr(detector, "streamed_args_for_tool") or not detector.streamed_args_for_tool:
        return []
    index = len(detector.prev_tool_call_arr) - 1
    if index >= len(detector.streamed_args_for_tool):
        return []
    expected = detector.prev_tool_call_arr[index].get("arguments", {})
    actual = detector.streamed_args_for_tool[index]
    # 0.5.14 serving used JSON encoding and substring removal; later releases
    # preserve raw argument strings and emit only a verified prefix suffix.
    if version("sglang") == "0.5.14":
        expected = json.dumps(expected, ensure_ascii=False)
        remaining = expected.replace(actual, "", 1) if actual in expected else ""
    else:
        expected = expected if isinstance(expected, str) else json.dumps(expected, ensure_ascii=False)
        remaining = expected[len(actual):] if expected.startswith(actual) else ""
    return [{"kind": "tool_call", "name": None, "arguments": remaining}] if remaining else []


def _stream_chunks(family, chunks, *, terminal_step=False, tools=TOOL_SCHEMAS, rows=None):
    rname, tname = FAMILY_PARSERS[family]
    # A native/default request has no template forcing reasoning. Serving passes
    # False explicitly rather than using family detector constructor defaults.
    rp = ReasoningParser(model_type=rname, stream_reasoning=True, force_reasoning=False, **_ACTIVE_KWARG)
    fp = FunctionCallParser(
        tools=[Tool(type="function", function=Function(**tool)) for tool in tools],
        tool_call_parser=tname,
    )
    rows = [] if rows is None else rows
    schedule = list(chunks) + ([""] if terminal_step else [])
    for index, ch in enumerate(schedule):
        final = index == len(schedule) - 1
        # The authored terminal row is a serving boundary, not model text.
        rd, nd = ("", "") if terminal_step and final else rp.parse_stream_chunk(ch)
        if final and hasattr(rp, "parse_stream_end"):
            end_rd, end_nd = rp.parse_stream_end()
            rd = (rd or "") + (end_rd or "")
            nd = (nd or "") + (end_nd or "")
        deltas = [{"kind": "reasoning", "text": rd}] if rd else []
        tn, calls = fp.parse_stream_chunk(nd) if nd else ("", [])
        if final and hasattr(fp, "parse_stream_end"):
            end_tn, end_calls = fp.parse_stream_end()
            tn = (tn or "") + (end_tn or "")
            calls = list(calls or []) + list(end_calls or [])
        if tn:
            deltas.append({"kind": "text", "text": tn})
        deltas.extend(_tool_stream_delta(c) for c in (calls or []))
        if final and hasattr(fp, "detector"):
            deltas.extend(_remaining_tool_args(fp))
        rows.append(deltas)
    return rows


def _capture_cases(cases):
    results = {}
    for case in cases:
        fam = case["family"]
        if fam not in FAMILY_PARSERS:
            continue
        entry = {"parser": "SGLang Python " + version("sglang") + " (stream, reasoning + tool detectors)"}
        partial_rows = []
        try:
            entry["chunks"] = _stream_chunks(fam, case.get("chunks") or [], terminal_step=case.get("terminal_step", False), tools=case.get("tools", TOOL_SCHEMAS), rows=partial_rows)
        except Exception as exc:  # noqa: BLE001 — record per-case, keep the batch going
            entry["partial_chunks"] = partial_rows
            entry["error"] = f"{type(exc).__name__}: {exc}"
            entry["error_traceback"] = traceback.format_exc()
        results[case["id"]] = entry
    return results


def main():
    job = json.load(sys.stdin)
    cases = job.get("cases", [])
    validate_family_scoped_cases(cases)
    results = {}
    for case in cases:
        results.update(capture_peer_results(
            [case], FAMILY_PARSERS, _capture_cases, tools=case["tools"],
            unsupported_reason=_registration_unavailable,
            supports_finish=True,
        ))
    # YAML to match the conformance fixture corpus. Container stdout is log-polluted,
    # so a recapture writes this to a file (or strips lines before the first top-level
    # key) rather than grepping a single JSON line.
    yaml.dump({"sglang_version": version("sglang"),
               "results": results}, sys.stdout,
              default_flow_style=False, sort_keys=False, allow_unicode=True, width=4096)


if __name__ == "__main__":
    main()
