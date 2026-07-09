#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Capture peer (vLLM / SGLang) REASONING parser output for the parity fixtures.

There is no reasoning equivalent of capture_cli / capture.py (those only cover
tool calling), so this file provides one. It has two modes:

  worker   Runs INSIDE an engine container (vllm-localdev / sglang-localdev) and
           drives that engine's reasoning parser over one fixture, emitting
           {version, cases: {cid: {reasoning_text, normal_text} | {error}}}.
           All engine imports are lazy so the file imports cleanly in either
           container -- the vLLM-only and SGLang-only paths never load the other.

               capture_reasoning.py worker --impl {vllm,sglang} \
                   --fixture /tmp/f.yaml --parser <name> [--mode batch|stream]

  drive    Runs on the HOST. For each reasoning fixture it docker-cp's this file
           + the fixture into the matching container, runs the worker, compares
           the captured output against the hand-authored expected.<impl> block,
           and (with --write) stamps `captured_with:` into the fixture for every
           impl whose captured output matches the fixture across all its cases.
           It never fabricates: an impl is only stamped when the container
           reproduces the fixture exactly, so `captured_with` is a real
           provenance claim rather than a guess.

               capture_reasoning.py drive [--family F] [--write]

The family -> peer parser-name maps live in tests/parity/common.py
(_FAMILY_TO_VLLM_REASONING / _FAMILY_TO_SGLANG_REASONING); the driver imports
them from there so this file never forks that mapping.
"""
import argparse
import json
import os
import subprocess
import sys

import yaml

# --------------------------------------------------------------------------- #
# Shared: synthetic special-token ids. The vLLM reasoning parsers that subclass
# BaseThinkingReasoningParser decide start/end of reasoning from *token ids*, not
# text, during streaming. Give each reasoning delimiter a stable synthetic id so
# the streaming path can run without a real model tokenizer -- the same trick the
# tool-calling capture.py uses for its markers.
# --------------------------------------------------------------------------- #
_REASONING_MARKER_IDS = {
    "<think>": 201,
    "</think>": 202,
    "<mm:think>": 203,
    "</mm:think>": 204,
    "[THINK]": 205,
    "[/THINK]": 206,
    "◁think▷": 207,  # Kimi legacy unicode delimiters
    "◁/think▷": 208,
    # Gemma 4 channel markers (used by the ParserEngine adapter's boundary check).
    "<|channel>": 209,
    "<channel|>": 210,
}
_MARKERS_BY_LENGTH = sorted(_REASONING_MARKER_IDS, key=len, reverse=True)


def _synthetic_token_ids(text):
    ids = []
    pos = 0
    while pos < len(text):
        matches = [
            (text.find(token, pos), token)
            for token in _MARKERS_BY_LENGTH
            if text.find(token, pos) != -1
        ]
        if not matches:
            break
        start, token = min(matches)
        ids.append(_REASONING_MARKER_IDS[token])
        pos = start + len(token)
    return ids


def engine_version(impl):
    if impl == "vllm":
        import vllm

        return vllm.__version__
    sys.path.insert(0, "/sgl-workspace/sglang/python")
    import sglang

    return sglang.__version__


# --------------------------------------------------------------------------- #
# vLLM: a mock tokenizer exposing just enough of the tokenizer API for the
# text/marker reasoning parsers to construct and run.
# --------------------------------------------------------------------------- #
def _vllm_mock_tokenizer():
    vocab = dict(_REASONING_MARKER_IDS)

    class MockTok:
        all_special_tokens = list(vocab)
        vocab_size = 1000

        def get_vocab(self):
            return dict(vocab)

        def convert_ids_to_tokens(self, ids, **kw):
            by_id = {v: k for k, v in vocab.items()}
            return [by_id.get(i, "") for i in ids]

        def convert_tokens_to_ids(self, tokens):
            if isinstance(tokens, str):
                return vocab.get(tokens)
            return [vocab.get(token) for token in tokens]

        def decode(self, ids, **kw):
            return ""

        def encode(self, text, **kw):
            return _synthetic_token_ids(text)

    return MockTok()


def _vllm_chat_template_kwargs(parser_name, case):
    """Mirror tests/parity/reasoning/table.py::_vllm_chat_template_kwargs so the
    capture matches the documented parity-harness flags."""
    kwargs = {}
    if parser_name == "deepseek_v4":
        kwargs["enable_thinking"] = True
    if parser_name == "deepseek_v3":
        kwargs["thinking"] = True
    kwargs.update(case.get("chat_template_kwargs") or {})
    return kwargs


def _vllm_reasoning_parser(parser_name, case):
    from vllm.reasoning import ReasoningParserManager

    cls = ReasoningParserManager.get_reasoning_parser(parser_name)
    kwargs = _vllm_chat_template_kwargs(parser_name, case)
    tok = _vllm_mock_tokenizer()
    if kwargs:
        return cls(tok, chat_template_kwargs=kwargs)
    return cls(tok)


def _vllm_batch(parser_name, cases):
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest

    if parser_name == "openai_gptoss":
        return _vllm_gptoss_batch(cases)

    out = {}
    for cid, case in cases.items():
        if "model_text" not in case:
            continue
        try:
            parser = _vllm_reasoning_parser(parser_name, case)
            req = ChatCompletionRequest(model="x", messages=[])
            reasoning, normal = parser.extract_reasoning(case["model_text"], req)
            out[cid] = {"reasoning_text": reasoning, "normal_text": normal}
        except Exception as e:  # noqa: BLE001 - surface the parser error into the cell
            out[cid] = {"error": f"{type(e).__name__}: {e}"}
    return out


def _vllm_stream(parser_name, cases):
    if parser_name == "openai_gptoss":
        return _vllm_gptoss_stream(cases)

    out = {}
    for cid, case in cases.items():
        if "chunks" not in case:
            continue
        try:
            parser = _vllm_reasoning_parser(parser_name, case)
            prev, prev_ids = "", []
            reasoning, normal = "", ""
            for chunk in case["chunks"]:
                delta = str(chunk)
                cur = prev + delta
                delta_ids = _synthetic_token_ids(delta)
                cur_ids = prev_ids + delta_ids
                dm = parser.extract_reasoning_streaming(
                    prev, cur, delta, prev_ids, cur_ids, delta_ids
                )
                if dm is not None:
                    if getattr(dm, "reasoning", None):
                        reasoning += dm.reasoning
                    if getattr(dm, "content", None):
                        normal += dm.content
                prev, prev_ids = cur, cur_ids
            out[cid] = {"reasoning_text": reasoning, "normal_text": normal}
        except Exception as e:  # noqa: BLE001
            out[cid] = {"error": f"{type(e).__name__}: {e}"}
    return out


# --------------------------------------------------------------------------- #
# vLLM gpt-oss: reasoning is the Harmony `analysis` channel, parsed via the
# Harmony StreamableParser (the openai_gptoss ReasoningParser only detects the
# reasoning-end boundary and cannot extract on its own).
# --------------------------------------------------------------------------- #
_HARMONY_START = 200006
_HARMONY_PREAMBLE = [200006, 173781]  # <|start|>assistant


def _vllm_gptoss_harness():
    from openai_harmony import HarmonyEncodingName, load_harmony_encoding
    from vllm.parser.harmony import HarmonyParser
    from vllm.reasoning.gptoss_reasoning_parser import GptOssReasoningParser

    enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    end_id = enc.encode("<|end|>", allowed_special="all")[0]

    class HTok:
        all_special_tokens = []
        vocab_size = 0

        def get_vocab(self):
            return {"<|end|>": end_id}

        def convert_ids_to_tokens(self, ids, **kw):
            return []

        def convert_tokens_to_ids(self, tokens):
            return None

        def decode(self, ids, **kw):
            return ""

        def encode(self, text, **kw):
            return enc.encode(text, allowed_special="all")

    # Wire the gpt-oss reasoning parser into HarmonyParser so the parse() path
    # classifies analysis-channel text as reasoning instead of dropping it.
    HarmonyParser.reasoning_parser_cls = GptOssReasoningParser
    return enc, HTok, HarmonyParser


def _harmony_ids(enc, text):
    ids = enc.encode(text, allowed_special="all")
    if ids and ids[0] != _HARMONY_START:
        ids = _HARMONY_PREAMBLE + ids
    return ids


def _vllm_gptoss_batch(cases):
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest

    enc, HTok, HarmonyParser = _vllm_gptoss_harness()
    req = ChatCompletionRequest(model="x", messages=[])
    out = {}
    for cid, case in cases.items():
        if "model_text" not in case:
            continue
        try:
            hp = HarmonyParser(HTok())
            ids = _harmony_ids(enc, case["model_text"])
            reasoning, content, _calls = hp.parse("", req, model_output_token_ids=ids)
            out[cid] = {"reasoning_text": reasoning, "normal_text": content}
        except Exception as e:  # noqa: BLE001
            out[cid] = {"error": f"{type(e).__name__}: {e}"}
    return out


def _vllm_gptoss_stream(cases):
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest

    enc, HTok, HarmonyParser = _vllm_gptoss_harness()
    req = ChatCompletionRequest(model="x", messages=[])
    out = {}
    for cid, case in cases.items():
        if "chunks" not in case:
            continue
        try:
            hp = HarmonyParser(HTok())
            text = "".join(str(c) for c in case["chunks"])
            ids = _harmony_ids(enc, text)
            reasoning, content, _calls = hp.parse("", req, model_output_token_ids=ids)
            out[cid] = {"reasoning_text": reasoning, "normal_text": content}
        except Exception as e:  # noqa: BLE001
            out[cid] = {"error": f"{type(e).__name__}: {e}"}
    return out


# --------------------------------------------------------------------------- #
# SGLang: self-contained detectors, no tokenizer needed.
# --------------------------------------------------------------------------- #
def _sglang_parser(model_type, case):
    sys.path.insert(0, "/sgl-workspace/sglang/python")
    from sglang.srt.parser.reasoning_parser import ReasoningParser

    return ReasoningParser(
        model_type=model_type,
        stream_reasoning=True,
        force_reasoning=bool(case.get("force_reasoning", False)),
    )


def _sglang_batch(model_type, cases):
    out = {}
    for cid, case in cases.items():
        if "model_text" not in case:
            continue
        try:
            parser = _sglang_parser(model_type, case)
            reasoning, normal = parser.parse_non_stream(case["model_text"])
            out[cid] = {"reasoning_text": reasoning, "normal_text": normal}
        except Exception as e:  # noqa: BLE001
            out[cid] = {"error": f"{type(e).__name__}: {e}"}
    return out


def _sglang_stream(model_type, cases):
    out = {}
    for cid, case in cases.items():
        if "chunks" not in case:
            continue
        try:
            parser = _sglang_parser(model_type, case)
            reasoning, normal = "", ""
            for chunk in case["chunks"]:
                r, n = parser.parse_stream_chunk(str(chunk))
                if r:
                    reasoning += r
                if n:
                    normal += n
            out[cid] = {"reasoning_text": reasoning, "normal_text": normal}
        except Exception as e:  # noqa: BLE001
            out[cid] = {"error": f"{type(e).__name__}: {e}"}
    return out


def _run_worker(args):
    doc = yaml.safe_load(open(args.fixture))
    cases = doc.get("cases", {})
    mode = args.mode or doc.get("mode") or "batch"
    if args.impl == "vllm":
        fn = _vllm_batch if mode == "batch" else _vllm_stream
    else:
        fn = _sglang_batch if mode == "batch" else _sglang_stream
    result = fn(args.parser, cases)
    print(json.dumps({"version": engine_version(args.impl), "cases": result},
                     ensure_ascii=False))


# --------------------------------------------------------------------------- #
# drive: host-side orchestrator. Copies this file + each fixture into the right
# container, runs the worker, compares to the fixture's expected blocks, and
# (with --write) stamps captured_with for every impl that matches everywhere.
# --------------------------------------------------------------------------- #
_HERE = os.path.dirname(os.path.abspath(__file__))
# capture_reasoning.py lives in conformance/utils/src/, so the repo root is 3 up.
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
_FIXTURES = os.path.join(_ROOT, "conformance", "reasoning", "fixtures-v1", "inputs")
_CAPTURED_KEY = {"vllm": "vllm_python", "sglang": "sglang_python"}


def _load_family_maps():
    sys.path.insert(0, os.path.join(_ROOT, "conformance", "utils", "tests"))
    from parity.common import _FAMILY_TO_SGLANG_REASONING, _FAMILY_TO_VLLM_REASONING

    return _FAMILY_TO_VLLM_REASONING, _FAMILY_TO_SGLANG_REASONING


def _norm(v):
    """Empty string and None are the same 'absent text' for comparison."""
    if v is None:
        return None
    if isinstance(v, str) and v.strip() == "":
        return None
    return v


def _blocks_match(captured, expected):
    if not isinstance(captured, dict) or "error" in captured:
        return False
    return _norm(captured.get("reasoning_text")) == _norm(
        expected.get("reasoning_text")
    ) and _norm(captured.get("normal_text")) == _norm(expected.get("normal_text"))


def _container_run(container, impl, fixture, parser):
    cpath = f"/tmp/reason_{impl}_{os.path.basename(os.path.dirname(fixture))}_{os.path.basename(fixture)}"
    subprocess.run(
        ["docker", "cp", os.path.join(_HERE, "capture_reasoning.py"),
         f"{container}:/tmp/capture_reasoning.py"], check=True)
    subprocess.run(["docker", "cp", fixture, f"{container}:{cpath}"], check=True)
    proc = subprocess.run(
        ["docker", "exec", container, "bash", "-lc",
         f"python3 /tmp/capture_reasoning.py worker --impl {impl} "
         f"--fixture {cpath} --parser {parser}"],
        capture_output=True, text=True)
    out = "\n".join(l for l in proc.stdout.splitlines() if l.strip().startswith("{"))
    if not out:
        raise RuntimeError(f"{container} capture failed: {proc.stderr[-800:]}")
    return json.loads(out)


def _compare_fixture(fixture, impl, parser, container):
    """Run capture, return (version, all_match, mismatches) for one fixture+impl.

    Only cases that have a real (non-unavailable) expected[impl] block AND parser
    input are considered. `all_match` is None when there is nothing to compare."""
    doc = yaml.safe_load(open(fixture))
    cases = doc.get("cases", {})
    captured = _container_run(container, impl, fixture, parser)
    version = captured["version"]
    comparable = 0
    mismatches = []
    for cid, case in cases.items():
        if not isinstance(case, dict) or "expected" not in case:
            continue
        expected = case["expected"].get(impl)
        if not isinstance(expected, dict) or "unavailable" in expected:
            continue
        if "model_text" not in case and "chunks" not in case:
            continue
        comparable += 1
        got = captured["cases"].get(cid)
        if not _blocks_match(got, expected):
            mismatches.append((cid, expected, got))
    all_match = None if comparable == 0 else (not mismatches)
    return version, all_match, mismatches, comparable


def _stamp_captured_with(fixture, versions):
    """Insert/replace `captured_with:` after the `mode:` line, preserving comments
    and anchors (a YAML round-trip would drop both), matching the toolcalling
    overlay format."""
    with open(fixture) as f:
        lines = f.readlines()
    block = ["captured_with:\n"]
    for impl in ("vllm", "sglang"):
        if impl in versions:
            block.append(f"  {_CAPTURED_KEY[impl]}: '{versions[impl]}'\n")
    if len(block) == 1:
        return
    # Drop any existing captured_with block (from a prior run) first.
    out, i = [], 0
    while i < len(lines):
        if lines[i].startswith("captured_with:"):
            i += 1
            while i < len(lines) and lines[i].startswith(("  ", "\t")):
                i += 1
            continue
        out.append(lines[i])
        i += 1
    # Re-insert right after the `mode:` line.
    final = []
    inserted = False
    for line in out:
        final.append(line)
        if not inserted and line.startswith("mode:"):
            final.extend(block)
            inserted = True
    with open(fixture, "w") as f:
        f.writelines(final)


def _run_drive(args):
    vllm_map, sglang_map = _load_family_maps()
    fixtures = sorted(
        os.path.join(_FIXTURES, d, f)
        for d in os.listdir(_FIXTURES)
        for f in ("REASONING.batch.yaml", "REASONING.stream.yaml")
        if os.path.exists(os.path.join(_FIXTURES, d, f))
    )
    # Group batch+stream results per family so captured_with is stamped once per
    # file but the match decision is per (family, impl) across both modes.
    per_file = {}
    for fixture in fixtures:
        family = os.path.basename(os.path.dirname(fixture))
        if args.family and family != args.family:
            continue
        per_file.setdefault(fixture, {})
        for impl, fam_map, container in (
            ("vllm", vllm_map, args.vllm_container),
            ("sglang", sglang_map, args.sglang_container),
        ):
            parser = fam_map.get(family)
            if parser is None:
                print(f"  {family:22s} {os.path.basename(fixture):22s} "
                      f"{impl:6s}: no peer parser (kept unavailable)")
                continue
            try:
                version, all_match, mismatches, n = _compare_fixture(
                    fixture, impl, parser, container)
            except Exception as e:  # noqa: BLE001
                print(f"  {family:22s} {os.path.basename(fixture):22s} "
                      f"{impl:6s}: CAPTURE ERROR {e}")
                continue
            if all_match is None:
                print(f"  {family:22s} {os.path.basename(fixture):22s} "
                      f"{impl:6s}: no comparable cases")
                continue
            status = "MATCH" if all_match else f"MISMATCH ({len(mismatches)}/{n})"
            print(f"  {family:22s} {os.path.basename(fixture):22s} "
                  f"{impl:6s} v{version}: {status}")
            for cid, exp, got in mismatches:
                print(f"      - {cid}: expected {exp} | captured {got}")
            if all_match:
                per_file[fixture].setdefault("versions", {})[impl] = version

    if not args.write:
        print("\n(dry run -- pass --write to stamp captured_with)")
        return
    for fixture, info in per_file.items():
        versions = info.get("versions")
        if versions:
            _stamp_captured_with(fixture, versions)
            print(f"stamped {os.path.relpath(fixture, _ROOT)}: {versions}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    w = sub.add_parser("worker", help="run inside a container over one fixture")
    w.add_argument("--impl", required=True, choices=("vllm", "sglang"))
    w.add_argument("--fixture", required=True)
    w.add_argument("--parser", required=True)
    w.add_argument("--mode", choices=("batch", "stream"))

    d = sub.add_parser("drive", help="host-side: capture, compare, stamp")
    d.add_argument("--vllm-container", default="vllm-localdev")
    d.add_argument("--sglang-container", default="sglang-localdev")
    d.add_argument("--family", help="restrict to one reasoning fixture family")
    d.add_argument("--write", action="store_true", help="stamp captured_with")

    args = ap.parse_args(argv)
    if args.cmd == "worker":
        _run_worker(args)
    else:
        _run_drive(args)


if __name__ == "__main__":
    main()
