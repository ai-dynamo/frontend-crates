#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build the v1-jail golden/after fixtures for the "TC v1 (stream data on jail)" tab.

Input is the same per-chunk corpus as the v2 stream tab (fixtures-stream-v2): same
taxonomy, same chunk inputs. For each case we record:
  - chunks: the input delta_text stream (copied from fixtures-stream-v2)
  - golden: the v1 jail's assembled output BEFORE #10570 (from a main capture)
  - now:    the v1 jail's assembled output AFTER  #10570 (from a pr10570 capture)

The conformance tab colors a cell green when now == golden and red otherwise, so a
single render shows exactly which cases #10570 changed. Captures are produced by
capture_dynamo_jail.py (record_jail_streamv2) over the streamv2 chunk inputs;
they are passed here as JSON {family: {case: [{deltas, normal_text} per chunk]}}.

Output: conformance/toolcalling/fixtures/<family>/TOOLCALLING.jail.<N>.yaml
"""
import argparse
import glob
import json
import os

import yaml

STREAMV2_ROOT = "conformance/toolcalling/fixtures-stream-v2"
OUT_ROOT = "conformance/toolcalling/fixtures"

# Conformance-only families with no real v1 jail parser. The v1 jail registers only
# "harmony"; "harmony_text" is the text-input gpt-oss variant, so running the jail
# with that name is a passthrough (golden == now, misleadingly all-green). Exclude
# it so the jail tab renders n/a for the family instead.
NON_JAIL_FAMILIES = {"harmony_text"}


def _assemble(rec: list) -> dict:
    """Assemble {calls, normal_text} from a capture's per-chunk list (deltas +
    normal_text fragments), reconstructing each tool call per index."""
    names: dict[int, str] = {}
    args: dict[int, str] = {}
    order: list[int] = []
    normal = ""
    for chunk in rec or []:
        for d in chunk.get("deltas", []) or []:
            idx = d["index"]
            if idx not in order:
                order.append(idx)
            if d.get("name") is not None:
                names[idx] = names.get(idx, "") + d["name"]
            if d.get("arguments") is not None:
                args[idx] = args.get(idx, "") + d["arguments"]
        normal += chunk.get("normal_text", "") or ""
    calls = []
    for idx in order:
        raw = args.get(idx, "")
        try:
            parsed = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            parsed = raw
        calls.append({"name": names.get(idx, ""), "arguments": parsed})
    return {"calls": calls, "normal_text": normal}


def _input_chunks(case: dict) -> list:
    """Keep only the stream INPUT fields per chunk (drop peer expected/normal_text)."""
    out = []
    for chunk in case.get("chunks", []) or []:
        if not isinstance(chunk, dict):
            continue
        keep = {"delta_text": chunk.get("delta_text", "")}
        if "delta_token_ids" in chunk:
            keep["delta_token_ids"] = chunk["delta_token_ids"]
        if chunk.get("finish_reason"):
            keep["finish_reason"] = chunk["finish_reason"]
        out.append(keep)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", required=True, help="capture JSON before #10570 (main)")
    ap.add_argument("--after", required=True, help="capture JSON after #10570 (pr10570)")
    args = ap.parse_args()
    golden = json.load(open(args.golden))
    after = json.load(open(args.after))

    n_files = n_cases = 0
    for src in sorted(glob.glob(f"{STREAMV2_ROOT}/*/TOOLCALLING.streamv2.*.yaml")):
        fam = os.path.basename(os.path.dirname(src))
        if fam in NON_JAIL_FAMILIES:
            continue
        fam_golden = golden.get(fam)
        fam_after = after.get(fam)
        if not fam_golden or not fam_after:
            continue  # family with no v1 jail capture (e.g. minimax_m3)
        doc = yaml.safe_load(open(src))
        out_cases = {}
        for cid, case in (doc.get("cases") or {}).items():
            g_rec, a_rec = fam_golden.get(cid), fam_after.get(cid)
            if g_rec is None or a_rec is None:
                continue
            jail_cid = cid.replace("TOOLCALLING.streamv2.", "TOOLCALLING.jail.")
            entry = {}
            if "description" in case:
                entry["description"] = case["description"]
            if "ref" in case:
                entry["ref"] = case["ref"]
            entry["chunks"] = _input_chunks(case)
            entry["golden"] = _assemble(g_rec)
            entry["now"] = _assemble(a_rec)
            out_cases[jail_cid] = entry
        if not out_cases:
            continue
        out_doc = {
            "family": fam,
            "model_label": doc.get("model_label", fam),
            "mode": "jail",
            "cases": out_cases,
        }
        n = os.path.basename(src).replace("TOOLCALLING.streamv2.", "").replace(".yaml", "")
        out_dir = os.path.join(OUT_ROOT, fam)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"TOOLCALLING.jail.{n}.yaml")
        with open(out_path, "w") as f:
            f.write("# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n")
            f.write("# SPDX-License-Identifier: Apache-2.0\n\n")
            f.write("# v1 jail golden/after fixture (generated by build_jail_fixtures.py).\n")
            f.write("# chunks = stream input; golden = jail output before #10570; now = after #10570.\n")
            f.write("# The tab is green when now == golden, red otherwise. Do not edit by hand.\n\n")
            yaml.safe_dump(out_doc, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        n_files += 1
        n_cases += len(out_cases)
    print(f"wrote {n_cases} jail cases across {n_files} files under {OUT_ROOT}/<family>/")


if __name__ == "__main__":
    main()
