#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Author and capture MiniMax M3's required legacy streaming coverage.

Run with --write to stage inputs and live captures before package_fixtures.py.
Without --write, verify the packaged snapshot against the authored scenarios and
live recorders. Build record_dynamo_stream, record_dynamo_jail_stream and
record_dynamo_reasoning first; --bin-dir selects their Cargo output directory.
Captures use the pinned crate versions. Existing versioned archives are immutable.
"""
import argparse
import json
import subprocess
import shutil
import tempfile
from pathlib import Path

import yaml

from dynamo_version import crate_version
from fixture_snapshot import fixture_snapshot_root

ROOT = Path(__file__).resolve().parents[3]
NS = "]<]minimax[>["
OPEN, CLOSE = NS + "<tool_call>", NS + "</tool_call>"
TOOLS = [{"name": name, "parameters": {"type": "object", "properties": {"location": {"type": "string"}}}}
         for name in ("get_weather", "get_time", "ping")]


def invoke(name="get_weather", location="NYC", body=None):
    if body is None:
        body = f"{NS}<location>{location}{NS}</location>"
    return f'{NS}<invoke name="{name}">{body}{NS}</invoke>'


def stream_cases():
    first, second = invoke(), invoke("get_time", "LA")
    one = [("get_weather", {"location": "NYC"})]
    two = one + [("get_time", {"location": "LA"})]
    truncated = f'{NS}<invoke name="get_time">{NS}<location>L'
    # Expected calls and prose are authored independently of recorder output.
    return {
        "2.a": ("Two calls inside one native wrapper", [OPEN, first, second, CLOSE], two, ""),
        "2.c": ("Parallel calls with surrounding narration", ["Before. " + OPEN, first, second, CLOSE + " After."], two, "Before.  After."),
        "2.d": ("Same tool name twice with distinct call indexes", [OPEN, first, invoke(location="LA"), CLOSE], one + [("get_weather", {"location": "LA"})], ""),
        "4.a": ("Native wrapper with garbage and no invoke", [OPEN, "not a call", CLOSE], [], ""),
        "5.b": ("Orphan native close marker without opener", [CLOSE], [], ""),
        "5.c": ("Truncation inside a parameter value drops incomplete call", [OPEN, truncated], [], ""),
        "5.d": ("Two complete calls with final wrapper close missing", [OPEN, first, second], two, ""),
        "5.e": ("Complete first call survives truncated second parameter", [OPEN, first, truncated], one, ""),
        "6.a": ("Canonical empty native invoke body", [OPEN, invoke("ping", body=""), CLOSE], [("ping", {})], ""),
        "6.b": ("Whitespace-only empty native invoke body", [OPEN, invoke("ping", body=" \n\t "), CLOSE], [("ping", {})], ""),
        "8.a": ("Narration before native tool call", ["Before. ", OPEN, first, CLOSE], one, "Before. "),
        "8.b": ("Narration after native tool call", [OPEN, first, CLOSE, " After."], one, " After."),
        "8.c": ("Narration before and after native tool call", ["Before. " + OPEN, first, CLOSE + " After."], one, "Before.  After."),
        "8.d": ("Narration between two native tool-call wrappers", [OPEN, first, CLOSE, "Then check LA. ", OPEN, second, CLOSE], two, "Then check LA. "),
        "50": ("Namespace and invoke close split across chunk boundaries", [OPEN[:7], OPEN[7:], first[:-9], first[-9:-3], first[-3:], CLOSE[:11], CLOSE[11:]], one, ""),
    }


def reasoning_cases():
    call = OPEN + invoke() + CLOSE
    calls = call + OPEN + invoke("get_time", "LA") + CLOSE
    return {
        "4.b": ("Native downstream boundary split across chunks exits open reasoning", ["<mm:think>thinking" + NS[:7], NS[7:] + "<tool_call>" + invoke() + CLOSE], "thinking", call),
        "4.c": ("Two native downstream tool calls remain intact", ["<mm:think>thinking", call, OPEN + invoke("get_time", "LA") + CLOSE], "thinking", calls),
        "4.d": ("Partial downstream namespace fakeout recovers as visible text", ["<mm:think>thinking<", "/mm:think>plain ]<]mini", "mal answer"], "thinking", "plain ]<]minimal answer"),
    }


def run(bin_dir, binary, payload, suffix=".json"):
    with tempfile.NamedTemporaryFile("w", suffix=suffix) as file:
        if suffix == ".yaml":
            yaml.safe_dump(payload, file, sort_keys=False)
        else:
            json.dump(payload, file)
        file.flush()
        return json.loads(subprocess.check_output([str(bin_dir / binary), file.name], text=True))


def assert_assembled(record, expected_calls, expected_text, case_id):
    calls = {}
    for chunk in record:
        for delta in chunk.get("deltas", []):
            index = delta["index"]
            call = calls.setdefault(index, {"name": "", "arguments": ""})
            call["name"] += delta.get("name") or ""
            call["arguments"] += delta.get("arguments") or ""
    actual = [(c["name"], json.loads(c["arguments"])) for _, c in sorted(calls.items())]
    assert list(calls) == list(range(len(expected_calls))), (case_id, calls)
    assert actual == expected_calls, (case_id, actual, expected_calls)
    actual_text = "".join(chunk.get("normal_text", "") for chunk in record)
    assert actual_text == expected_text, (case_id, actual_text, expected_text)


def capture_doc(record, implementation, version):
    cases = {}
    for cid, chunks in record.items():
        cases[cid] = {"chunks": [dict(expected=c.get("deltas", []), **({"normal_text": c["normal_text"]} if c.get("normal_text") else {})) for c in chunks]}
    return {"family": "minimax_m3", "mode": "streamv1", "captured_with": {implementation: version}, "cases": cases}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--split-check", action="store_true", help="Also check whole input and every character boundary")
    parser.add_argument("--bin-dir", type=Path, default=ROOT / "target/debug")
    args = parser.parse_args()
    snapshot = fixture_snapshot_root()
    v1, v2 = crate_version(ROOT / "parsers/v1/Cargo.toml"), crate_version(ROOT / "parsers/v2/Cargo.toml")
    pending = {}
    for suffix, (description, chunks, calls, text) in stream_cases().items():
        cid = "TOOLCALLING.streamv1." + suffix
        doc = {"family": "minimax_m3", "mode": "streamv1", "model_label": "MiniMax M3", "cases": {cid: {
            "description": description, "tools": TOOLS,
            "chunks": [{"delta_text": c} for c in chunks] + [{"delta_text": "", "finish_reason": "tool_calls"}],
        }}}
        filename = cid + ".yaml"
        base = Path("toolcalling/fixtures-stream-v1")
        pending[base / "inputs/minimax_m3" / filename] = doc
        record = run(args.bin_dir, "record_dynamo_stream", doc, ".yaml")
        assert_assembled(record[cid], calls, text, cid)
        if args.split_check:
            full = "".join(chunks)
            schedules = [[full]] + [[full[:i], full[i:]] for i in range(1, len(full))]
            probes = {f"{cid}@{i}": {"tools": TOOLS, "chunks": [{"delta_text": c} for c in parts]}
                      for i, parts in enumerate(schedules)}
            result = run(args.bin_dir, "record_dynamo_stream", {"family": "minimax_m3", "cases": probes}, ".yaml")
            for probe, output in result.items():
                assert_assembled(output, calls, text, probe)
        pending[base / f"dynamo_v2-{v2}/minimax_m3" / filename] = capture_doc(record, "dynamo_v2", v2)
        jail = run(args.bin_dir, "record_dynamo_jail_stream", {"family": "minimax_m3", "cases": {cid: {"chunks": chunks + [""], "tools": TOOLS}}})
        pending[base / f"dynamo_v1-{v1}/minimax_m3" / filename] = capture_doc(jail, "dynamo_v1", v1)

    cases = {}
    for suffix, (description, chunks, reasoning, normal) in reasoning_cases().items():
        cases["REASONING.stream." + suffix] = {"description": description, "chunks": chunks, "expected": {"dynamo_v1": {"reasoning_text": reasoning, "normal_text": normal}}}
    for case in cases.values():
        for peer in ("vllm_python", "sglang_python"):
            case["expected"][peer] = {"unavailable": "No peer capture has been collected for this fixture."}
    doc = {"family": "minimax_m3", "model_label": "MiniMax M3", "mode": "stream", "captured_with": {"dynamo_v1": v1}, "cases": cases}
    record = run(args.bin_dir, "record_dynamo_reasoning", doc)
    for cid, case in cases.items():
        assert record[cid] == case["expected"]["dynamo_v1"], (cid, record[cid], case["expected"])
    if args.split_check:
        for cid, case in cases.items():
            full = "".join(case["chunks"])
            schedules = [[full]] + [[full[:i], full[i:]] for i in range(1, len(full))]
            probes = {f"{cid}@{i}": {"chunks": chunks} for i, chunks in enumerate(schedules)}
            result = run(args.bin_dir, "record_dynamo_reasoning", {"family": "minimax_m3", "mode": "stream", "cases": probes})
            for probe, output in result.items():
                assert output == case["expected"]["dynamo_v1"], (probe, output)
    pending[Path("reasoning/fixtures-v1/inputs/minimax_m3/REASONING.stream.4-boundaries.yaml")] = doc

    if args.write:
        # Existing shared shards retain every unrelated family and prior case.
        for relative in {Path(*path.parts[:3]) for path in pending}:
            source, destination = snapshot / relative, ROOT / "conformance" / relative
            if source.is_dir():
                shutil.copytree(source, destination, dirs_exist_ok=True)

    for relative, doc in pending.items():
        if args.write:
            path = ROOT / "conformance" / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=120))
            path.chmod(0o644)
        else:
            actual = yaml.safe_load((snapshot / relative).read_text())
            assert actual == doc, relative
    print(f"{'Staged' if args.write else 'Verified packaged and live'} 15 tool-stream and 3 reasoning-stream cases (v1 {v1}, v2 {v2}).")


if __name__ == "__main__":
    main()
