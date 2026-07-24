#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Re-capture a peer engine's parser output against a NEWER version and write
changed-only version overlays, so every conformance tab can compare peer versions
the way the batch tab does. ONE engine x corpus core replaces the four per-corpus
capturers (capture_batch_versions.py, capture_streamv2_versions.py,
capture_reasoning_versions.py, capture_vllm_rust_versions.py).

Corpora (`--corpus`):
  batch      fixtures-batch-v1: vLLM/SGLang non-streaming `extract_tool_calls`
             over `inputs/<family>/TOOLCALLING.batch*.yaml`, diffed against the
             LOWEST `<impl>-<version>/` anchor's `expected.<impl>` {calls,
             normal_text} block. Writes changed-only top-level
             `fixtures-batch-v1/<impl>-<version>/<family>/TOOLCALLING.batch*.yaml`.
  stream     fixtures-stream-v2: per-chunk streaming deltas. Container engines
             (vllm_python/sglang_python) run capture.py in the engine container,
             diff each chunk against the lowest-version resolved anchor's
             per-chunk `expected.<impl>`/`normal_text.<impl>`, and write a
             changed-only per-chunk overlay under
             `fixtures-stream-v2/overlays/<impl>-<version>/` (promote to the top
             level before packaging). vllm_rust runs the cargo probe over the
             shared `inputs/` cases, diffs whole cases against the lowest
             `vllm_rust-<version>/` anchor, and writes a full-chunk changed-case
             dir `fixtures-stream-v2/vllm_rust-<version>/`.
  reasoning  reasoning/fixtures-v1: vLLM/SGLang reasoning parser over
             `inputs/<family>/REASONING.{batch,stream}.yaml`, diffed against the
             inputs' `expected.<impl>` {reasoning_text, normal_text} anchor.
             Writes changed-only `<impl>-<version>/<family>/REASONING.{mode}.yaml`.

Engines (`--impl`):
  vllm_python   vLLM Python container (default vllm-localdev); tool-calling map
                capture_driver.VLLM, reasoning map _FAMILY_TO_VLLM_REASONING.
  sglang_python SGLang Python container (default sglang-localdev); tool-calling
                map capture_driver.SGLANG, reasoning map _FAMILY_TO_SGLANG_REASONING.
  vllm_rust     source-only cargo probe (stream corpus only; needs a vLLM source
                checkout via --vllm-rust-source / VLLM_RUST_SOURCE); map
                capture_driver.VLLM_RUST.

All engine versions are read LIVE at capture time (container `__version__` /
source tag), never hardcoded. Only cases that DIVERGE from the anchor are written;
cases that error in-container / have no parser are logged and carried forward,
never fabricated. Existing version dirs are never touched (append-only).

Usage:
  python3 capture_peer_versions.py --corpus batch                       # all impls
  python3 capture_peer_versions.py --corpus batch --impl sglang_python  # one impl
  python3 capture_peer_versions.py --corpus stream --impl vllm_rust \
      --vllm-rust-source ~/dev/vllm-0.25.1
  python3 capture_peer_versions.py --corpus reasoning --family qwen3
"""
import argparse
import glob
import os
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Optional

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
# This module lives in conformance/utils/src/, so the repo root is 3 up and the
# parity test package (tests.parity.*) lives under conformance/utils/.
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
for _p in (HERE, os.path.join(ROOT, "conformance", "utils")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import capture_driver as cd  # noqa: E402  (parser maps + container/probe capture plumbing)
import capture_reasoning as cr  # noqa: E402  (reasoning worker + _container_run + _blocks_match)
import resolve_fixtures  # noqa: E402  (batch anchor resolution)
import resolve_stream_fixtures  # noqa: E402  (stream anchor staging)
import validate  # noqa: E402  (run_container ships the tool-calling parity adapter)
from tests.parity.common import (  # noqa: E402
    _FAMILY_TO_SGLANG_REASONING,
    _FAMILY_TO_VLLM_REASONING,
    canonical,
)


# --------------------------------------------------------------------------- #
# Engine specs. The engine axis is what SGLang batch/reasoning support adds to:
# a corpus loop asks the engine for its parser-family map and how to run one case.
# --------------------------------------------------------------------------- #
@dataclass
class EngineSpec:
    name: str  # canonical impl key + version-dir prefix (vllm_python / sglang_python / vllm_rust)
    short: str  # capture.py / adapter short name (vllm / sglang); "" for source-only
    default_container: Optional[str]
    tc_map: dict  # family -> tool-calling parser name for this engine
    reasoning_map: Optional[dict]  # family -> reasoning parser name (None: no reasoning path)
    corpora: frozenset  # which corpora this engine can run
    source_based: bool = False  # vllm_rust: cargo probe over a source checkout, no container

    def container(self, args) -> Optional[str]:
        if self.source_based:
            return None
        return getattr(args, f"{self.short}_container", None) or self.default_container


ENGINES = {
    "vllm_python": EngineSpec(
        name="vllm_python", short="vllm", default_container="vllm-localdev",
        tc_map=cd.VLLM, reasoning_map=_FAMILY_TO_VLLM_REASONING,
        corpora=frozenset({"batch", "stream", "reasoning"}),
    ),
    "sglang_python": EngineSpec(
        name="sglang_python", short="sglang", default_container="sglang-localdev",
        tc_map=cd.SGLANG, reasoning_map=_FAMILY_TO_SGLANG_REASONING,
        corpora=frozenset({"batch", "stream", "reasoning"}),
    ),
    "vllm_rust": EngineSpec(
        name="vllm_rust", short="", default_container=None,
        tc_map=cd.VLLM_RUST, reasoning_map=None,
        corpora=frozenset({"stream"}), source_based=True,
    ),
}


class _QuotedStr(str):
    """Version string forced to single-quoted YAML, matching the batch overlays'
    `captured_with: {<impl>: '<version>'}` shape."""


def _represent_quoted(dumper, data):
    return dumper.represent_scalar("tag:yaml.org,2002:str", str(data), style="'")


yaml.SafeDumper.add_representer(_QuotedStr, _represent_quoted)

_SPDX = (
    "# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n"
    "# SPDX-License-Identifier: Apache-2.0\n"
)


def _report(impl, version, n_cases, n_files, families_touched, n_errored):
    print(f"[{impl}] version {version}: {n_cases} changed case(s) across {n_files} "
          f"file(s) in {len(families_touched)} family(ies); {n_errored} errored/carried-forward",
          file=sys.stderr)


def _lowest_impl_dir(fixtures_root, impl):
    """The lowest `<impl>-<version>` dir = that impl's full anchor (every case)."""
    vdirs = [
        d for d in glob.glob(os.path.join(fixtures_root, f"{impl}-*")) if os.path.isdir(d)
    ]
    if not vdirs:
        raise SystemExit(f"no {impl}-* anchor under {fixtures_root}")
    return min(
        vdirs,
        key=lambda d: resolve_fixtures.version_key(os.path.basename(d).split("-", 1)[1]),
    )


# =========================================================================== #
# Block corpora (batch, reasoning): per-case expected block, changed-only
# top-level version dir. One shared loop, two specs.
# =========================================================================== #
@dataclass
class BlockCorpus:
    name: str
    fixtures_root_rel: str
    require_anchor: bool  # reasoning skips cases with no valid anchor; batch writes them

    def fixtures_root(self, root):
        return os.path.join(root, self.fixtures_root_rel)

    # --- filled in per corpus below ---
    build_anchor: object = None       # (engine, fixtures_root, work) -> ({(fam,base,cid): block}, desc)
    run_all: object = None            # (engine, container, fixtures_root, families, work) -> (result, version)
    is_error: object = None           # (cap) -> bool
    captured_matches: object = None   # (cap, anchor_block) -> bool
    changed_block: object = None      # (engine, cap) -> dict
    write: object = None              # (fixtures_root, engine, version, family, base, mode, changed)


def _run_block(corpus: BlockCorpus, engine: EngineSpec, args):
    fixtures_root = corpus.fixtures_root(args.root)
    work = args.work or tempfile.mkdtemp(prefix=f"{corpus.name}_ver_")
    os.makedirs(work, exist_ok=True)
    container = engine.container(args)

    anchor, desc = corpus.build_anchor(engine, fixtures_root, work)
    print(f"[{engine.name}] anchor = {desc} ({len(anchor)} baseline cases)", file=sys.stderr)

    result, version = corpus.run_all(engine, container, fixtures_root, args.family, work)
    if not result:
        print(f"[{engine.name}] no input cases; nothing to do", file=sys.stderr)
        return
    print(f"[{engine.name}] engine version {version}", file=sys.stderr)

    n_files = n_cases = n_errored = 0
    families_touched = set()
    for (family, base, mode), caps in sorted(result.items()):
        changed = {}
        for cid, cap in caps.items():
            anchor_block = anchor.get((family, base, cid))
            if corpus.require_anchor and anchor_block is None:
                continue
            if corpus.is_error(cap):
                n_errored += 1
                print(f"  [{engine.name}] {family}/{base} {cid}: parser error, carried forward",
                      file=sys.stderr)
                continue
            if anchor_block is not None and corpus.captured_matches(cap, anchor_block):
                continue
            changed[cid] = corpus.changed_block(engine, cap)
        if not changed:
            continue
        corpus.write(fixtures_root, engine, version, family, base, mode, changed)
        n_files += 1
        n_cases += len(changed)
        families_touched.add(family)
        print(f"  [{engine.name}] wrote {family}/{base} ({len(changed)} changed case(s))",
              file=sys.stderr)

    _report(engine.name, version, n_cases, n_files, families_touched, n_errored)


# --------------------------------------------------------------------------- #
# batch corpus hooks
# --------------------------------------------------------------------------- #
def _batch_build_anchor(engine, fixtures_root, work):
    """Fold the lowest `<impl>-<version>` dir onto inputs and return
    {(family, base, cid): {calls, normal_text}} baseline blocks."""
    lowest = os.path.basename(_lowest_impl_dir(fixtures_root, engine.name)).split("-", 1)[1]
    staged = os.path.join(work, "baseline")
    resolve_fixtures.resolve(fixtures_root, staged, [f"{engine.name}-{lowest}"])
    baseline = {}
    for fp in sorted(glob.glob(os.path.join(staged, "*", "*.yaml"))):
        family = os.path.basename(os.path.dirname(fp))
        base = os.path.basename(fp)
        doc = yaml.safe_load(open(fp))
        for cid, case in (doc.get("cases") or {}).items():
            if not isinstance(case, dict):
                continue
            exp = (case.get("expected") or {}).get(engine.name)
            if isinstance(exp, dict) and "unavailable" not in exp:
                baseline[(family, base, cid)] = {
                    "calls": exp.get("calls", []),
                    "normal_text": exp.get("normal_text"),
                }
    return baseline, f"{engine.name}-{lowest}"


def _batch_run_all(engine, container, fixtures_root, families, work):
    """Every input batch case with a model_text through the engine's batch parser,
    grouped as {(family, base, 'batch'): {cid: cap}}."""
    version = validate.engine_version(engine.short, container)
    if not version:
        raise SystemExit(f"could not read {engine.short} version from {container}")
    inputs = os.path.join(fixtures_root, "inputs")
    fam_dirs = families or sorted(
        d for d in os.listdir(inputs) if os.path.isdir(os.path.join(inputs, d))
    )
    cases, by_file = [], {}
    for family in fam_dirs:
        for fp in sorted(glob.glob(os.path.join(inputs, family, "TOOLCALLING.batch*.yaml"))):
            base = os.path.basename(fp)
            doc = yaml.safe_load(open(fp))
            fam = doc["family"]
            for cid, case in (doc.get("cases") or {}).items():
                if not isinstance(case, dict) or "model_text" not in case:
                    continue
                key = f"{fam}/{base}/{cid}"
                cases.append({
                    "key": key, "family": fam, "mode": "batch",
                    "tools": case.get("tools"), "model_text": case.get("model_text"),
                })
                by_file.setdefault((fam, base, "batch"), []).append((cid, key))
    if not cases:
        return {}, version
    print(f"[{engine.name}] capturing {len(cases)} batch cases in {container} (1 import)...",
          file=sys.stderr)
    got = validate.run_container(engine.short, container, cases)
    result = {}
    for key3, cids in by_file.items():
        d = {}
        for cid, jobkey in cids:
            cap = got.get(jobkey)
            if cap is not None:
                d[cid] = cap
        result[key3] = d
    return result, version


def _batch_matches(cap, anchor_block):
    captured = {"calls": cap.get("calls", []), "normal_text": cap.get("normal_text")}
    return canonical(captured) == canonical(anchor_block)


def _batch_changed_block(engine, cap):
    # vLLM emits None for "no narration"; the corpus renders that as '' — normalize
    # so the stored shape matches (the diff already treats ''/None equal via canonical).
    return {"expected": {engine.name: {
        "calls": cap.get("calls", []),
        "normal_text": cap.get("normal_text") or "",
    }}}


def _batch_write(fixtures_root, engine, version, family, base, mode, changed):
    outdir = os.path.join(fixtures_root, f"{engine.name}-{version}", family)
    os.makedirs(outdir, exist_ok=True)
    out = {
        "family": family, "mode": "batch",
        "captured_with": {engine.name: _QuotedStr(version)},
        "cases": changed,
    }
    with open(os.path.join(outdir, base), "w") as f:
        f.write(_SPDX)
        f.write("# Version overlay (changed-only): cases where this impl@version diverges from baseline.\n")
        yaml.safe_dump(out, f, allow_unicode=True, sort_keys=False, default_flow_style=False)


BATCH = BlockCorpus(
    name="batch",
    fixtures_root_rel="conformance/toolcalling/fixtures-batch-v1",
    require_anchor=False,
    build_anchor=_batch_build_anchor,
    run_all=_batch_run_all,
    is_error=lambda cap: bool(cap.get("error")),
    captured_matches=_batch_matches,
    changed_block=_batch_changed_block,
    write=_batch_write,
)


# --------------------------------------------------------------------------- #
# reasoning corpus hooks
# --------------------------------------------------------------------------- #
def _reasoning_mode(base):
    """'REASONING.batch.yaml' -> 'batch'; 'REASONING.stream.yaml' -> 'stream'."""
    return base.split(".")[1]


def _reasoning_build_anchor(engine, fixtures_root, work):
    """The inputs' `expected.<impl>` {reasoning_text, normal_text} blocks are the
    anchor; only cases with a real (non-unavailable) block AND a parser input count."""
    inputs = os.path.join(fixtures_root, "inputs")
    baseline = {}
    for fp in sorted(glob.glob(os.path.join(inputs, "*", "REASONING.*.yaml"))):
        family = os.path.basename(os.path.dirname(fp))
        base = os.path.basename(fp)
        doc = yaml.safe_load(open(fp))
        for cid, case in (doc.get("cases") or {}).items():
            if not isinstance(case, dict) or "expected" not in case:
                continue
            b = (case.get("expected") or {}).get(engine.name)
            if not isinstance(b, dict) or "unavailable" in b:
                continue
            if "model_text" not in case and "chunks" not in case:
                continue
            baseline[(family, base, cid)] = b
    return baseline, "inputs/"


def _reasoning_run_all(engine, container, fixtures_root, families, work):
    """Drive the engine's reasoning parser over each inputs REASONING.{mode}.yaml,
    grouped as {(family, base, mode): {cid: cap}}. Families without a parser and
    whole-fixture failures are skipped (carried forward)."""
    inputs = os.path.join(fixtures_root, "inputs")
    fam_dirs = families or sorted(
        d for d in os.listdir(inputs) if os.path.isdir(os.path.join(inputs, d))
    )
    result, version = {}, None
    no_parser = []
    for family in fam_dirs:
        parser = engine.reasoning_map.get(family)
        if parser is None:
            no_parser.append(family)
            continue
        for mode in ("batch", "stream"):
            fixture = os.path.join(inputs, family, f"REASONING.{mode}.yaml")
            if not os.path.exists(fixture):
                continue
            try:
                captured = cr._container_run(container, engine.short, fixture, parser)
            except Exception as e:  # noqa: BLE001 - whole-fixture failure, carry forward
                print(f"  [{engine.name}] {family}/REASONING.{mode}: capture error, carried "
                      f"forward ({str(e)[:120]})", file=sys.stderr)
                continue
            version = captured["version"]
            base = f"REASONING.{mode}.yaml"
            result[(family, base, mode)] = dict(captured["cases"])
    if no_parser:
        print(f"[{engine.name}] no reasoning parser (skipped): {', '.join(no_parser)}",
              file=sys.stderr)
    return result, version


def _reasoning_changed_block(engine, cap):
    # Render "absent text" as '' (None -> '') to match the anchor overlay shape;
    # _blocks_match already treats ''/None equal for the diff decision.
    return {"expected": {engine.name: {
        "reasoning_text": cap.get("reasoning_text") or "",
        "normal_text": cap.get("normal_text") or "",
    }}}


def _reasoning_write(fixtures_root, engine, version, family, base, mode, changed):
    outdir = os.path.join(fixtures_root, f"{engine.name}-{version}", family)
    os.makedirs(outdir, exist_ok=True)
    out = {"family": family, "mode": mode, "cases": changed}
    with open(os.path.join(outdir, base), "w") as f:
        f.write(_SPDX)
        f.write(f"# Changed-only {engine.short} {version} reasoning overlay (vs the inputs/ anchor).\n")
        yaml.safe_dump(out, f, allow_unicode=True, sort_keys=False, default_flow_style=False)


REASONING = BlockCorpus(
    name="reasoning",
    fixtures_root_rel="conformance/reasoning/fixtures-v1",
    require_anchor=True,
    build_anchor=_reasoning_build_anchor,
    run_all=_reasoning_run_all,
    is_error=lambda cap: "error" in cap,
    captured_matches=lambda cap, anchor_block: cr._blocks_match(cap, anchor_block),
    changed_block=_reasoning_changed_block,
    write=_reasoning_write,
)


# =========================================================================== #
# stream corpus: per-chunk deltas. Container engines write a per-chunk overlay;
# vllm_rust writes a full-chunk changed-case dir.
# =========================================================================== #
STREAM_ROOT_REL = "conformance/toolcalling/fixtures-stream-v2"

# gemma4 keeps a `vllm_rust` parser name in parser_families.yaml, but vLLM 0.25.0
# turned it into a native unified parser not reachable through the tool::ToolParser
# probe, so it is recorded unavailable with this explicit note rather than dropped.
GEMMA4_UNAVAILABLE = (
    "gemma4 moved to the native unified parser in vLLM 0.25.0; "
    "not exposed via the tool::ToolParser probe."
)


# --- container engines (vllm_python / sglang_python): streamv2-overlay format --- #
def _ov_norm_deltas(deltas):
    """Canonical per-chunk delta list for the streamv2-overlay format (raw dicts,
    keeping id): YAML and JSON load id/name/arguments the same way, so plain
    dict/list equality works once both sides are lists of dicts."""
    return [dict(d) for d in (deltas or []) if isinstance(d, dict)]


def _anchor_chunk_impl(chunk, impl):
    """(deltas, normal_text) the resolved anchor recorded for `impl` at this chunk."""
    exp = chunk.get("expected") or {}
    nt = chunk.get("normal_text") or {}
    return _ov_norm_deltas(exp.get(impl)), (nt.get(impl) or "")


def _build_overlay(anchor_doc, captured_cases, impl):
    """{cid: {chunk_idx: {expected, normal_text}}} for cases whose newly captured
    per-chunk output differs from the anchor, plus a list of errored cids. Only
    differing chunks are recorded."""
    overlay_cases, errored = {}, []
    for cid, case in (anchor_doc.get("cases") or {}).items():
        if impl in (case.get("unavailable") or {}):
            continue
        cap = captured_cases.get(cid)
        if cap is None:
            continue
        anchor_chunks = case.get("chunks") or []
        if any(isinstance(c, dict) and c.get("error") for c in cap):
            errored.append(cid)
            continue
        changed = {}
        for idx, anchor_chunk in enumerate(anchor_chunks):
            if not isinstance(anchor_chunk, dict) or idx >= len(cap):
                continue
            a_deltas, a_nt = _anchor_chunk_impl(anchor_chunk, impl)
            c_deltas = _ov_norm_deltas(cap[idx].get("deltas"))
            c_nt = cap[idx].get("normal_text") or ""
            if c_deltas != a_deltas or c_nt != a_nt:
                changed[idx] = {"expected": c_deltas, "normal_text": c_nt}
        if changed:
            overlay_cases[cid] = changed
    return overlay_cases, errored


def _run_stream_container(engine, args):
    """vLLM Python / SGLang Python stream capture -> changed-only per-chunk overlay
    under fixtures-stream-v2/overlays/<impl>-<version>/."""
    sv2_root = os.path.join(args.root, STREAM_ROOT_REL)
    work = args.work or tempfile.mkdtemp(prefix="streamv2_ver_")
    os.makedirs(work, exist_ok=True)
    container = engine.container(args)

    # The anchor is the lowest-version-per-impl resolved tree: each chunk carries
    # per-impl `expected`/`normal_text` PLUS the shared delta_text the parser reads.
    staged = os.path.join(work, "anchor")
    resolve_stream_fixtures.resolve(sv2_root, staged, [])

    families = [args.family] if args.family else sorted(engine.tc_map.keys())
    anchor_files = {}
    for family in families:
        fs = sorted(glob.glob(os.path.join(staged, family, "TOOLCALLING.streamv2.*.yaml")))
        if fs:
            anchor_files[family] = fs

    jobs, job_family = [], {}
    for family, files in anchor_files.items():
        parser = engine.tc_map.get(family)
        if not parser:
            continue
        for fp in files:
            jobs.append({"src": fp, "container_path": cd._cpath(fp, "stream"), "parser": parser})
            job_family[fp] = family
    if not jobs:
        print(f"[{engine.name}] no fixtures with a parser; skipping", file=sys.stderr)
        return

    cd._copy_worker((container,))
    print(f"[{engine.name}] capturing {len(jobs)} fixtures in {container} (1 import)...",
          file=sys.stderr)
    version, caps = cd._container_capture(container, engine.short, "stream", jobs, work)
    print(f"[{engine.name}] engine version {version}", file=sys.stderr)

    n_files = n_cases = n_errored = 0
    families_touched = set()
    for fp, family in job_family.items():
        entry = caps.get(fp, {})
        if "cases" not in entry:
            print(f"  [{engine.name}] {family}/{os.path.basename(fp)}: capture error, carried "
                  f"forward ({str(entry.get('error', '?'))[:120]})", file=sys.stderr)
            n_errored += 1
            continue
        anchor_doc = yaml.safe_load(open(fp))
        overlay_cases, errored = _build_overlay(anchor_doc, entry["cases"], engine.name)
        n_errored += len(errored)
        if not overlay_cases:
            continue
        outdir = os.path.join(sv2_root, "overlays", f"{engine.name}-{version}", family)
        os.makedirs(outdir, exist_ok=True)
        out = {
            "family": family, "mode": "streamv2-overlay",
            "overlay_impl": engine.name, "overlay_version": version,
            "cases": overlay_cases,
        }
        with open(os.path.join(outdir, os.path.basename(fp)), "w") as f:
            yaml.safe_dump(out, f, allow_unicode=True, sort_keys=False, width=4096)
        n_files += 1
        n_cases += len(overlay_cases)
        families_touched.add(family)
        print(f"  [{engine.name}] wrote overlay {family}/{os.path.basename(fp)} "
              f"({len(overlay_cases)} changed case(s))", file=sys.stderr)

    _report(engine.name, version, n_cases, n_files, families_touched, n_errored)


# --- vllm_rust: full-chunk changed-case streamv2 dir --- #
def _clean_version(raw):
    """'v0.25.1 <sha>' -> '0.25.1' (dir name + captured_with stamp)."""
    return raw.split()[0].lstrip("v") if raw else raw


def _rust_norm_deltas(deltas):
    """Probe/anchor delta list -> canonical [{index, name?, arguments?}] for diff and
    serialization (drop absent name/arguments and the id flag, keep first-seen order)."""
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


def _rust_anchor_case_form(case):
    """Comparable form of a vllm_rust anchor case. Both engines marking a case
    unavailable counts as 'no divergence' regardless of message wording."""
    if "unavailable" in case:
        return ("unavail",)
    chunks = [
        (_rust_norm_deltas(ch.get("expected")), ch.get("normal_text") or "")
        for ch in (case.get("chunks") or [])
        if isinstance(ch, dict)
    ]
    return ("chunks", chunks)


def _rust_captured_case_form(cap):
    """Comparable form of one probe case result (chunk list or {error})."""
    if isinstance(cap, dict):  # {"error": ...}
        return ("unavail",)
    return ("chunks", [(_rust_norm_deltas(ch.get("deltas")), ch.get("normal_text") or "") for ch in cap])


def _rust_captured_case_doc(cap):
    """0.23.0-shaped case dict for the overlay from one probe case result."""
    if isinstance(cap, dict):  # {"error": ...}
        return {"unavailable": f"vllm_rust parser not captured: {cap['error']}"}
    chunks = []
    for ch in cap:
        entry = {"expected": _rust_norm_deltas(ch.get("deltas"))}
        nt = ch.get("normal_text") or ""
        if nt:
            entry["normal_text"] = nt
        chunks.append(entry)
    return {"chunks": chunks}


def _run_stream_rust(engine, args):
    """vLLM Rust cargo-probe stream capture -> full-chunk changed-case dir
    fixtures-stream-v2/vllm_rust-<version>/ (diffed against the lowest vllm_rust anchor)."""
    source = args.vllm_rust_source or os.environ.get("VLLM_RUST_SOURCE")
    if not source:
        raise SystemExit("--vllm-rust-source or VLLM_RUST_SOURCE is required for vllm_rust")
    sv2 = os.path.join(args.root, STREAM_ROOT_REL)
    inputs_root = os.path.join(sv2, "inputs")
    anchor_root = _lowest_impl_dir(sv2, "vllm_rust")
    work = args.work or tempfile.mkdtemp(prefix="vllm_rust_ver_")
    os.makedirs(work, exist_ok=True)

    families = sorted(engine.tc_map)
    if args.family:
        families = [f for f in families if f == args.family]

    jobs, job_meta = [], {}
    for family in families:
        parser = engine.tc_map[family]
        for fp in sorted(glob.glob(os.path.join(inputs_root, family, "TOOLCALLING.streamv2.*.yaml"))):
            jobs.append({"src": fp, "parser": parser})
            job_meta[fp] = (family, os.path.basename(fp))
    if not jobs:
        raise SystemExit("no vllm_rust stream inputs found")

    print(f"[vllm_rust] capturing {len(jobs)} fixtures through the vLLM Rust probe...",
          file=sys.stderr)
    raw_version, caps = cd._vllm_rust_capture(source, "stream", jobs, work)
    version = _clean_version(raw_version)
    print(f"[vllm_rust] vLLM Rust source {raw_version} -> version {version}", file=sys.stderr)

    n_files = n_cases = n_errored = n_missing_anchor = 0
    families_touched = set()
    for fp, (family, base) in job_meta.items():
        entry = caps.get(fp, {})
        if "cases" not in entry:
            print(f"  [vllm_rust] {family}/{base}: whole-fixture capture error "
                  f"({str(entry.get('error'))[:120]})", file=sys.stderr)
            n_errored += 1
            continue
        anchor_fp = os.path.join(anchor_root, family, base)
        if not os.path.exists(anchor_fp):
            n_missing_anchor += 1
            print(f"  [vllm_rust] {family}/{base}: no anchor; skipping", file=sys.stderr)
            continue
        anchor_doc = yaml.safe_load(open(anchor_fp))
        changed = {}
        for cid, anchor_case in (anchor_doc.get("cases") or {}).items():
            cap = entry["cases"].get(cid)
            if cap is None:
                continue
            if _rust_captured_case_form(cap) != _rust_anchor_case_form(anchor_case):
                changed[cid] = ({"unavailable": GEMMA4_UNAVAILABLE} if family == "gemma4"
                                else _rust_captured_case_doc(cap))
        if not changed:
            continue
        outdir = os.path.join(sv2, f"vllm_rust-{version}", family)
        os.makedirs(outdir, exist_ok=True)
        doc = {
            "family": family, "mode": "streamv2",
            "captured_with": {"vllm_rust": version}, "cases": changed,
        }
        with open(os.path.join(outdir, base), "w") as f:
            yaml.safe_dump(doc, f, allow_unicode=True, sort_keys=False, width=4096)
        n_files += 1
        n_cases += len(changed)
        families_touched.add(family)

    print(f"[vllm_rust] wrote vllm_rust-{version}: {n_cases} changed case(s) across {n_files} "
          f"file(s) in {len(families_touched)} family(ies); {n_missing_anchor} without an anchor",
          file=sys.stderr)


def _run_stream(engine: EngineSpec, args):
    if engine.source_based:
        _run_stream_rust(engine, args)
    else:
        _run_stream_container(engine, args)


CORPORA = {"batch": BATCH, "reasoning": REASONING, "stream": "stream"}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--corpus", required=True, choices=("batch", "stream", "reasoning"))
    ap.add_argument("--impl", choices=tuple(ENGINES),
                    help="capture only this engine (default: all engines valid for the corpus)")
    ap.add_argument("--root", default=ROOT, help="repo root (default: src/../../..)")
    ap.add_argument("--family", help="capture only this family (default: all)")
    ap.add_argument("--vllm-container", default="vllm-localdev")
    ap.add_argument("--sglang-container", default="sglang-localdev")
    ap.add_argument("--vllm-rust-source", help="vLLM source checkout root; defaults to VLLM_RUST_SOURCE")
    ap.add_argument("--work", help="work dir (default: a fresh temp dir)")
    args = ap.parse_args()

    corpus = args.corpus
    if args.impl:
        impls = [args.impl]
        if corpus not in ENGINES[args.impl].corpora:
            raise SystemExit(f"--impl {args.impl} does not support --corpus {corpus} "
                             f"(supported: {', '.join(sorted(ENGINES[args.impl].corpora))})")
    else:
        impls = [name for name, spec in ENGINES.items() if corpus in spec.corpora]

    for name in impls:
        engine = ENGINES[name]
        if corpus == "stream":
            _run_stream(engine, args)
        else:
            _run_block(CORPORA[corpus], engine, args)


if __name__ == "__main__":
    main()
