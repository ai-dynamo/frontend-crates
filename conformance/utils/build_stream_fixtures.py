#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Assemble a new-format streaming fixture from source inputs + captured per-chunk data.

New format (per-chunk, per-impl):
  Each chunk carries its input (delta_text + optional delta_token_ids) and, under
  `expected`, the tool-call deltas each parser emits at that chunk. `normal_text`
  (per-impl, only when non-empty) records non-tool text emitted — this is where
  leaked tool markup shows up. Cross-impl comparison is at the assembled level.

Inputs (all per-case, keyed by case id):
  --source       the source fixture (chunks/tools/metadata; expected.* ignored)
  --dynamo JSON  {case: [[delta,...], ...]}            (from record_dynamo_stream)
  --sglang JSON  {case: [{deltas, normal_text}, ...]}  (from container probe)
  --vllm JSON    {case: [{deltas, normal_text}, ...]}  (from container probe)
  --unavailable impl=reason   case-level unavailable (repeatable)
  --na impl                   mark impl not-applicable per-chunk (omit from expected)

A delta: {index, id?, name?, arguments?}.  id:true = an id was emitted.
"""
import argparse
import json
import sys

import yaml


def _norm_dynamo(raw):
    # recorder: {case: [[delta,...], ...]} -> {case: [{deltas, normal_text}, ...]}
    out = {}
    for cid, chunks in raw.items():
        out[cid] = [{"deltas": deltas, "normal_text": ""} for deltas in chunks]
    return out


def _load(path):
    if not path:
        return {}
    return json.load(open(path))


def _q(s) -> str:
    """YAML single-quote a scalar (safe for any newline-free string)."""
    return "'" + str(s).replace("'", "''") + "'"


def _delta_flow(d: dict) -> str:
    parts = [f"index: {d['index']}"]
    if d.get("id"):
        parts.append("id: true")
    if d.get("name") is not None:
        parts.append(f"name: {_q(d['name'])}")
    if d.get("arguments") is not None:
        parts.append(f"arguments: {_q(d['arguments'])}")
    return "{" + ", ".join(parts) + "}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--dynamo")
    ap.add_argument("--sglang")
    ap.add_argument("--vllm")
    ap.add_argument("--unavailable", action="append", default=[])
    ap.add_argument("--na", action="append", default=[])
    ap.add_argument("--captured", action="append", default=[],
                    help="impl=version — engine version the per-chunk data was captured against")
    ap.add_argument("--family", help="override the family id (e.g. harmony_text)")
    ap.add_argument("--label", help="override the model_label")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    src = yaml.safe_load(open(args.source))
    family = args.family or src["family"]
    model_label = args.label or src.get("model_label", family)

    caps = {
        "dynamo": _norm_dynamo(_load(args.dynamo)),
        "sglang": _load(args.sglang),
        "vllm": _load(args.vllm),
    }
    unavail = dict(u.split("=", 1) for u in args.unavailable)
    na_impls = set(args.na)
    captured_versions = dict(c.split("=", 1) for c in args.captured)

    L = []
    L.append("# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.")
    L.append("# SPDX-License-Identifier: Apache-2.0")
    L.append("")
    L.append("# Streaming fixture (per-chunk, per-impl). Each chunk: input (delta_text +")
    L.append("# optional delta_token_ids) and `expected.<impl>` = tool-call deltas emitted at")
    L.append("# that chunk. `normal_text.<impl>` (when non-empty) = non-tool text (leaks show")
    L.append("# here). Assembled result is derived by concatenating per index. A delta:")
    L.append("# {index, id?, name?, arguments?}; id:true = an id was emitted.")
    L.append("")
    L.append(f"family: {family}")
    L.append(f"model_label: {_q(model_label)}")
    L.append("mode: stream")
    # Engine versions the vLLM/SGLang per-chunk data was captured against. A
    # divergence is only meaningful relative to these — re-capture when bumping.
    if captured_versions:
        L.append("captured_with:")
        for impl in ("vllm", "sglang"):
            if impl in captured_versions:
                L.append(f"  {impl}: {_q(captured_versions[impl])}")
    L.append("cases:")

    for cid, case in src["cases"].items():
        L.append(f"  {cid}:")
        if "description" in case:
            L.append(f"    description: {_q(case['description'])}")
        if "ref" in case:
            L.append(f"    ref: {_q(case['ref'])}")
        # tools (block style)
        if case.get("tools"):
            L.append("    tools:")
            L.append(_indent(yaml.safe_dump(case["tools"], default_flow_style=False,
                                            allow_unicode=True, sort_keys=False).rstrip(), 4))
        # unavailable block
        case_unavail = dict(unavail)
        if case_unavail:
            L.append("    unavailable:")
            for impl, reason in case_unavail.items():
                L.append(f"      {impl}: {_q(reason)}")
        # chunks
        L.append("    chunks:")
        chunks = case.get("chunks", [])
        for ci, chunk in enumerate(chunks):
            L.append(f"    - delta_text: {_q(chunk.get('delta_text', ''))}")
            if "delta_token_ids" in chunk:
                ids = ", ".join(str(t) for t in chunk["delta_token_ids"])
                L.append(f"      delta_token_ids: [{ids}]")
            if chunk.get("finish_reason"):
                L.append(f"      finish_reason: {chunk['finish_reason']}")
            # expected deltas per impl
            exp_lines = []
            nt_lines = []
            for impl in ("dynamo", "vllm", "sglang"):
                if impl in case_unavail or impl in na_impls:
                    continue
                cap = caps.get(impl, {}).get(cid)
                if cap is None:
                    continue
                chunk_cap = cap[ci] if ci < len(cap) else {"deltas": [], "normal_text": ""}
                deltas = chunk_cap.get("deltas", [])
                nt = chunk_cap.get("normal_text", "") or ""
                if deltas:
                    exp_lines.append(f"        {impl}:")
                    for d in deltas:
                        exp_lines.append(f"        - {_delta_flow(d)}")
                else:
                    exp_lines.append(f"        {impl}: []")
                if nt:
                    nt_lines.append(f"        {impl}: {_q(nt)}")
            if exp_lines:
                L.append("      expected:")
                L.extend(exp_lines)
            if nt_lines:
                L.append("      normal_text:")
                L.extend(nt_lines)

    text = "\n".join(L) + "\n"
    # validate
    yaml.safe_load(text)
    open(args.out, "w").write(text)
    print(f"wrote {args.out}", file=sys.stderr)


def _indent(block: str, n: int) -> str:
    pad = " " * n
    return "\n".join(pad + line if line else line for line in block.splitlines())


if __name__ == "__main__":
    main()
