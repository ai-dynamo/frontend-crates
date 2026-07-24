#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Re-capture the vLLM Rust stream parser against a NEWER vLLM source checkout and
write the changed-only stream overlay + refresh the batch-on-stream vllm_rust
blocks, the vLLM-Rust analogue of capture_batch_versions.py / recapture_batch_on_stream.py.

vLLM Rust is stream-only (no Python package), so it is captured through the
`capture_vllm_rust.py` cargo probe (`capture_driver._vllm_rust_capture`) rather
than an engine container. This tool drives that probe twice:

  stream           feed each `fixtures-stream-v2/inputs/<family>` case's chunks
                   through the 0.25.x parser, diff each case against the
                   `vllm_rust-0.23.0` anchor full tree, and write a changed-only
                   overlay `fixtures-stream-v2/vllm_rust-<clean_version>/<family>/`
                   (mode: streamv2, per-changed-case full chunk list — the same
                   shape the `vllm_python-0.24.0` peer-version overlays use, so
                   resolve_stream_fixtures folds it ascending).

  batch-on-stream  feed each `fixtures-batch-v1/inputs/<family>` case's model_text
                   through the streaming parser and rewrite ONLY the `vllm_rust`
                   block (+ `captured_with.vllm_rust`) of each existing
                   `fixtures-batch-on-stream-v2/<family>` fixture, leaving the
                   vllm_python / sglang_python / dynamo_v2 blocks untouched.

Only families with a `vllm_rust` parser in parser_families.yaml are captured.
gemma4 is listed there but lost its `tool::ToolParser` in vLLM 0.25.0 (it became a
native unified parser); the probe reports that and it is recorded as unavailable.
Append-only: the 0.23.0 anchor tree is never touched.
"""
import argparse
import glob
import os
import sys
from pathlib import Path

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import capture_driver as cd  # noqa: E402  (VLLM_RUST map + _vllm_rust_capture plumbing)

# gemma4 keeps a `vllm_rust` parser name in parser_families.yaml, but vLLM 0.25.0
# turned it into a native unified parser (vllm_parser::unified::Gemma4UnifiedParser)
# with a tokenizer-backed API that is not reachable through the tool::ToolParser
# probe, so it is recorded unavailable with this explicit note rather than dropped.
GEMMA4_UNAVAILABLE = (
    "gemma4 moved to the native unified parser in vLLM 0.25.0; "
    "not exposed via the tool::ToolParser probe."
)


def _clean_version(raw):
    """'v0.25.1 <sha>' -> '0.25.1' (dir name + stream captured_with stamp)."""
    return raw.split()[0].lstrip("v") if raw else raw


def _norm_deltas(deltas):
    """Probe/anchor delta list -> canonical [{index, name?, arguments?}] for diff
    and for serialization (drop absent name/arguments, keep first-seen order)."""
    out = []
    for d in deltas or []:
        if not isinstance(d, dict):
            continue
        e = {"index": d["index"]}
        if d.get("name") is not None:
            e["name"] = d["name"]
        if d.get("arguments") is not None:
            e["arguments"] = d["arguments"]
        out.append(e)
    return out


def _anchor_case_form(case):
    """Comparable form of an anchor (vllm_rust-0.23.0) case. Both engines marking a
    case unavailable counts as 'no divergence' regardless of message wording."""
    if "unavailable" in case:
        return ("unavail",)
    chunks = [
        (_norm_deltas(ch.get("expected")), ch.get("normal_text") or "")
        for ch in (case.get("chunks") or [])
        if isinstance(ch, dict)
    ]
    return ("chunks", chunks)


def _captured_case_form(cap):
    """Comparable form of one 0.25.x probe case result (chunk list or {error})."""
    if isinstance(cap, dict):  # {"error": ...}
        return ("unavail",)
    chunks = [
        (_norm_deltas(ch.get("deltas")), ch.get("normal_text") or "")
        for ch in cap
    ]
    return ("chunks", chunks)


def _captured_case_doc(cap):
    """0.23.0-shaped case dict for the overlay from one probe case result."""
    if isinstance(cap, dict):  # {"error": ...}
        return {"unavailable": f"vllm_rust parser not captured: {cap['error']}"}
    chunks = []
    for ch in cap:
        entry = {"expected": _norm_deltas(ch.get("deltas"))}
        nt = ch.get("normal_text") or ""
        if nt:
            entry["normal_text"] = nt
        chunks.append(entry)
    return {"chunks": chunks}


def _run_stream(args, source, work):
    sv2 = os.path.join(args.root, "conformance/toolcalling/fixtures-stream-v2")
    inputs_root = os.path.join(sv2, "inputs")
    anchor_root = os.path.join(sv2, "vllm_rust-0.23.0")

    families = sorted(cd.VLLM_RUST)
    if args.family:
        families = [f for f in families if f == args.family]

    # One probe build, all fixtures batched.
    jobs, job_meta = [], {}
    for family in families:
        parser = cd.VLLM_RUST[family]
        for fp in sorted(glob.glob(os.path.join(inputs_root, family, "TOOLCALLING.streamv2.*.yaml"))):
            jobs.append({"src": fp, "parser": parser})
            job_meta[fp] = (family, os.path.basename(fp))
    if not jobs:
        raise SystemExit("no vllm_rust stream inputs found")

    print(f"[stream] capturing {len(jobs)} fixtures through the vLLM Rust probe...", file=sys.stderr)
    raw_version, caps = cd._vllm_rust_capture(source, "stream", jobs, work)
    version = _clean_version(raw_version)
    print(f"[stream] vLLM Rust source {raw_version} -> version {version}", file=sys.stderr)

    per_family = {}
    n_files = n_cases = n_missing_anchor = 0
    for fp, (family, base) in job_meta.items():
        entry = caps.get(fp, {})
        if "cases" not in entry:
            print(f"  [stream] {family}/{base}: whole-fixture capture error "
                  f"({str(entry.get('error'))[:120]})", file=sys.stderr)
            continue
        captured_cases = entry["cases"]
        anchor_fp = os.path.join(anchor_root, family, base)
        if not os.path.exists(anchor_fp):
            n_missing_anchor += 1
            print(f"  [stream] {family}/{base}: no 0.23.0 anchor; skipping", file=sys.stderr)
            continue
        anchor_doc = yaml.safe_load(open(anchor_fp))
        changed = {}
        for cid, anchor_case in (anchor_doc.get("cases") or {}).items():
            cap = captured_cases.get(cid)
            if cap is None:
                continue
            if _captured_case_form(cap) != _anchor_case_form(anchor_case):
                if family == "gemma4":
                    changed[cid] = {"unavailable": GEMMA4_UNAVAILABLE}
                else:
                    changed[cid] = _captured_case_doc(cap)
        if not changed:
            continue
        outdir = os.path.join(sv2, f"vllm_rust-{version}", family)
        os.makedirs(outdir, exist_ok=True)
        doc = {
            "family": family,
            "mode": "streamv2",
            "captured_with": {"vllm_rust": version},
            "cases": changed,
        }
        with open(os.path.join(outdir, base), "w") as f:
            yaml.safe_dump(doc, f, allow_unicode=True, sort_keys=False, width=4096)
        n_files += 1
        n_cases += len(changed)
        per_family[family] = per_family.get(family, 0) + len(changed)

    print("\n[stream] changed cases per family (vs vllm_rust-0.23.0):", file=sys.stderr)
    for family in families:
        print(f"  {family}: {per_family.get(family, 0)}", file=sys.stderr)
    print(f"[stream] wrote vllm_rust-{version}: {n_cases} changed case(s) across {n_files} "
          f"file(s); {n_missing_anchor} file(s) without a 0.23.0 anchor", file=sys.stderr)


def _run_batch_on_stream(args, source, work):
    root = args.root
    inputs_root = os.path.join(root, "conformance/toolcalling/fixtures-batch-v1/inputs")
    out_root = os.path.join(root, "conformance/toolcalling/fixtures-batch-on-stream-v2")

    families = sorted(cd.VLLM_RUST)
    if args.family:
        families = [f for f in families if f == args.family]

    jobs, job_family = [], {}
    for family in families:
        parser = cd.VLLM_RUST[family]
        for src in sorted(glob.glob(os.path.join(inputs_root, family, "TOOLCALLING.batch*.yaml"))):
            jobs.append({"src": src, "parser": parser})
            job_family[src] = family
    if not jobs:
        raise SystemExit("no vllm_rust batch inputs found")

    print(f"[bos] capturing {len(jobs)} batch fixtures through the vLLM Rust probe...", file=sys.stderr)
    raw_version, caps = cd._vllm_rust_capture(source, "batch-on-stream", jobs, work)
    print(f"[bos] vLLM Rust source {raw_version}", file=sys.stderr)

    n_updated = 0
    per_family = {}
    for src in job_family:
        family = job_family[src]
        outfp = os.path.join(out_root, family, os.path.basename(src))
        if not os.path.exists(outfp):
            continue
        doc = yaml.safe_load(open(outfp)) or {}
        doc.setdefault("captured_with", {})["vllm_rust"] = raw_version
        entry = caps.get(src, {})
        vllm_rust_cases = entry.get("cases", {}) if "cases" in entry else {}
        src_doc = yaml.safe_load(open(src))
        src_cases = src_doc.get("cases") or {}
        changed_here = 0
        for cid, row in (doc.get("cases") or {}).items():
            case = src_cases.get(cid, {})
            if cid in vllm_rust_cases:
                cap = vllm_rust_cases[cid]
                if isinstance(cap, dict) and "error" in cap:
                    # batch-on-stream records an expected parser exception under the
                    # `error` key verbatim (README: expected.<impl>.error); gemma4's
                    # "parser moved" is an unavailability, not a parse error.
                    if family == "gemma4":
                        block = {"unavailable": GEMMA4_UNAVAILABLE}
                    else:
                        block = {"error": cap["error"]}
                else:
                    block = cap
            elif "model_text" not in case:
                block = {"unavailable": "No batch model_text for this case."}
            else:
                block = {"unavailable": "Capture did not return this case."}
            if row.get("vllm_rust") != block:
                changed_here += 1
            row["vllm_rust"] = block
        with open(outfp, "w") as f:
            yaml.safe_dump(doc, f, allow_unicode=True, sort_keys=False)
        n_updated += 1
        per_family[family] = per_family.get(family, 0) + changed_here

    # Keep the vLLM Rust column version uniform: families without a vllm_rust parser
    # (jamba/phi4/pythonic/minimax_m3/harmony) carry a "No vLLM Rust parser" block but
    # still stamp `captured_with.vllm_rust` — bump that stamp to the current source so
    # the whole column reads one version. Block contents for those families do not change.
    n_stamped = 0
    if not args.family:
        for fp in sorted(glob.glob(os.path.join(out_root, "*", "*.yaml"))):
            doc = yaml.safe_load(open(fp)) or {}
            cw = doc.get("captured_with") or {}
            if "vllm_rust" in cw and cw["vllm_rust"] != raw_version:
                cw["vllm_rust"] = raw_version
                with open(fp, "w") as f:
                    yaml.safe_dump(doc, f, allow_unicode=True, sort_keys=False)
                n_stamped += 1

    print("\n[bos] changed vllm_rust blocks per family:", file=sys.stderr)
    for family in families:
        print(f"  {family}: {per_family.get(family, 0)}", file=sys.stderr)
    print(f"[bos] rewrote vllm_rust block in {n_updated} batch-on-stream fixture(s); "
          f"stamp-only bump in {n_stamped} no-parser fixture(s)", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=("stream", "batch-on-stream"))
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
    ap.add_argument("--vllm-rust-source", help="vLLM source checkout root; defaults to VLLM_RUST_SOURCE")
    ap.add_argument("--family", help="restrict to one family (debugging)")
    ap.add_argument("--work", default="/tmp/vllm_rust_versions_work")
    args = ap.parse_args()

    source = args.vllm_rust_source or os.environ.get("VLLM_RUST_SOURCE")
    if not source:
        ap.error("--vllm-rust-source or VLLM_RUST_SOURCE is required")
    os.makedirs(args.work, exist_ok=True)

    if args.mode == "stream":
        _run_stream(args, source, args.work)
    else:
        _run_batch_on_stream(args, source, args.work)


if __name__ == "__main__":
    main()
