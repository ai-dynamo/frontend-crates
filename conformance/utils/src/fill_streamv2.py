#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate the TC stream-v2 fixtures for the non-harmony families.

For each family + each batch case that has `model_text`, chunk the text into ~1-3
"token" pieces, run the vLLM and SGLang streaming parsers over the chunks (inside
the engine containers, one engine import each), and assemble the per-chunk fixture locally, then publish to HuggingFace
(`ai-dynamo/conformance-fixtures`) via `package_and_publish.py`.
Dynamo is marked unavailable/TODO (no parser v2 stream parser for these families
yet); the synthetic partial-token case `50` has no batch source and is left
untouched.

This mirrors each family's streamv2 tab to its batch taxonomy (the same sub-cases
the batch tab shows). Harmony / harmony_text are NOT handled here — they use the
token-id capture flow (capture_driver.py).

Usage:
  python3 conformance/utils/fill_streamv2.py                 # all non-harmony families
  python3 conformance/utils/fill_streamv2.py qwen25 mistral  # specific families
"""
import argparse
import glob
import os
import re
import subprocess
import sys
import tempfile

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import capture_driver as cd  # noqa: E402  (parser maps + container-capture helpers)

DYNAMO_TODO = (
    "Dynamo parser v2 TC streaming not yet implemented for this family; "
    "vLLM Python/SGLang per-chunk output is the target to match."
)

# Deterministic ~1-3 "token" chunker. A real model tokenizer emits special tool
# markers (`<|...|>`, `<tool_call>`, `[TOOL_CALLS]`) as ATOMIC tokens — never split
# mid-marker — so chunk boundaries land at marker closers (`>`/`]`) and whitespace,
# not inside a marker. Char-shredding a marker is unfaithful: it makes special-token
# parsers leak in ways the real engine never would.
_PAT = [2, 1, 3, 2, 1, 2, 3, 1, 2, 1]


def _tokenize(text):
    """Approximate model tokens: whitespace runs atomic; within a non-space run,
    end a token at each `>` or `]` so a complete `<...>` / `[...]` marker is one
    token (markers stay intact across chunk boundaries)."""
    toks = []
    for run in re.findall(r"\S+|\s+", text):
        if not run.strip():
            toks.append(run)
        else:
            toks.extend(re.findall(r"[^>\]]*[>\]]|[^>\]]+", run))
    return toks


def chunk_text(text):
    toks = _tokenize(text)
    chunks, cur, n, k = [], "", 0, 0
    target = _PAT[0]
    for t in toks:
        cur += t
        if t.strip():  # count only non-space tokens toward the group size
            n += 1
        if n >= target:
            chunks.append(cur)
            cur, n, k = "", 0, k + 1
            target = _PAT[k % len(_PAT)]
    if cur:
        chunks.append(cur)
    return chunks or [text]


def _expects_calls(case):
    exp = case.get("expected") or {}
    for impl in ("dynamo_rust", "vllm_rust", "vllm_python", "sglang"):
        b = exp.get(impl)
        if isinstance(b, dict) and b.get("calls"):
            return True
    return False


def build_sources(family, fixtures_root, srcdir):
    """Write one source fixture per batch case-number (every batch sub-case that
    has model_text) so the family's streamv2 tab mirrors the batch taxonomy.
    Returns {num: source_path}."""
    by_num = {}
    label = family
    for f in sorted(glob.glob(os.path.join(fixtures_root, family, "TOOLCALLING.batch*.yaml"))):
        doc = yaml.safe_load(open(f))
        label = doc.get("model_label", family)
        for cid, c in (doc.get("cases") or {}).items():
            m = re.match(r"TOOLCALLING\.batch\.(\d+)(.*)$", cid)
            if not m:
                continue
            num, suffix = m.group(1), m.group(2)
            if not isinstance(c.get("model_text"), str):  # skip na-stubs (no text)
                continue
            fr = "tool_calls" if _expects_calls(c) else "stop"
            chunks = [{"delta_text": ch} for ch in chunk_text(c["model_text"])]
            chunks.append({"delta_text": "", "finish_reason": fr})
            scid = f"TOOLCALLING.streamv2.{num}{suffix}"
            by_num.setdefault(num, {})[scid] = {
                "description": c.get("description", ""),
                "ref": f"derived from {cid}",
                "tools": c.get("tools") or [],
                "chunks": chunks,
            }
    out = {}
    for num, cases in by_num.items():
        src = {"family": family, "model_label": label, "mode": "streamv2", "cases": cases}
        p = os.path.join(srcdir, f"{family}__streamv2.{num}.yaml")
        yaml.safe_dump(src, open(p, "w"), allow_unicode=True, sort_keys=False)
        out[num] = p
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("families", nargs="*", help="families to fill (default: all non-harmony)")
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(HERE)),
                    help="repo root (default: two levels above this script)")
    ap.add_argument("--vllm-container", default="vllm-localdev")
    ap.add_argument("--sglang-container", default="sglang-localdev")
    ap.add_argument("--vllm-rust-source", help="vLLM source checkout root; defaults to VLLM_RUST_SOURCE")
    ap.add_argument("--work", help="work dir (default: a fresh temp dir)")
    args = ap.parse_args()

    fixtures_root = os.path.join(args.root, "conformance/toolcalling/fixtures-batch-v1")
    out_root = os.path.join(args.root, "conformance/toolcalling/fixtures-stream-v2")
    work = args.work or tempfile.mkdtemp(prefix="streamv2_fill_")
    srcdir = os.path.join(work, "src")
    os.makedirs(srcdir, exist_ok=True)
    families = args.families or sorted(cd.VLLM.keys())

    # 1. build source fixtures, collect capture jobs
    family_sources, vllm_jobs, vllm_rust_jobs, sglang_jobs = {}, [], [], []
    for family in families:
        srcs = build_sources(family, fixtures_root, srcdir)
        family_sources[family] = srcs
        for fp in srcs.values():
            if cd.VLLM.get(family):
                vllm_jobs.append({"src": fp, "container_path": cd._cpath(fp, "stream"),
                                  "parser": cd.VLLM[family]})
            if cd.VLLM_RUST.get(family):
                vllm_rust_jobs.append({"src": fp, "parser": cd.VLLM_RUST[family]})
            if cd.SGLANG.get(family):
                sglang_jobs.append({"src": fp, "container_path": cd._cpath(fp, "stream"),
                                    "parser": cd.SGLANG[family]})
    total = sum(len(s) for s in family_sources.values())
    print(f"built {total} source fixtures across {len(families)} families", file=sys.stderr)
    if total == 0:
        return

    # 2. capture (one engine import per container)
    cd._copy_worker((args.vllm_container, args.sglang_container))
    print(f"capturing vllm ({len(vllm_jobs)} fixtures)...", file=sys.stderr)
    vllm_ver, vllm_caps = cd._container_capture(args.vllm_container, "vllm", "stream", vllm_jobs, work)
    vllm_rust_source = cd._vllm_rust_source_arg(args)
    print(f"capturing vllm rust ({len(vllm_rust_jobs)} fixtures)...", file=sys.stderr)
    vllm_rust_ver, vllm_rust_caps = cd._vllm_rust_capture(
        vllm_rust_source, "stream", vllm_rust_jobs, work)
    print(f"capturing sglang ({len(sglang_jobs)} fixtures)...", file=sys.stderr)
    sglang_ver, sglang_caps = cd._container_capture(args.sglang_container, "sglang", "stream", sglang_jobs, work)
    vllm_rust_source_version = vllm_rust_ver or cd._vllm_rust_source_version(vllm_rust_source)

    # 3. assemble each fixture
    for family, srcs in family_sources.items():
        for num, fp in srcs.items():
            base = f"TOOLCALLING.streamv2.{num}.yaml"
            outdir = os.path.join(out_root, family)
            os.makedirs(outdir, exist_ok=True)
            outfp = os.path.join(outdir, base)
            cmd = ["python3", os.path.join(HERE, "build_stream_fixtures.py"),
                   "--source", fp, "--out", outfp,
                   "--unavailable", f"dynamo_rust={DYNAMO_TODO}"]
            if vllm_rust_source:
                cmd += cd._impl_args("vllm_rust", family, cd.VLLM_RUST.get(family),
                                     vllm_rust_caps.get(fp, {}),
                                     vllm_rust_source_version, work, f"{family}_{num}", fp)
            else:
                cmd += ["--unavailable", f"vllm_rust={cd._vllm_rust_unavailable(vllm_rust_source_version)}"]
            cmd += cd._impl_args("vllm_python", family, cd.VLLM.get(family),
                                 vllm_caps.get(fp, {}), vllm_ver, work, f"{family}_{num}", fp)
            cmd += cd._impl_args("sglang_python", family, cd.SGLANG.get(family),
                                 sglang_caps.get(fp, {}), sglang_ver, work, f"{family}_{num}", fp)
            subprocess.run(cmd, check=True)
            # build_stream_fixtures.py hardcodes `mode: stream`; this is the v2 tab.
            txt = open(outfp).read().replace("\nmode: stream\n", "\nmode: streamv2\n", 1)
            open(outfp, "w").write(txt)
            print(f"  built {family}/{base}", file=sys.stderr)


if __name__ == "__main__":
    main()
