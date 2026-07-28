#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Repair stream-v2 `inputs/` whose newlines were folded into spaces by YAML quoting.

THE BUG
-------
`build_stream_fixtures._q` emitted every scalar single-quoted, and a single-quoted YAML
scalar FOLDS an embedded newline into a space when read back. `delta_text` contains
newlines, so the writer silently corrupted its own output; a later re-emission from the
already-folded values baked the loss in permanently.

What it costs: vLLM's DeepSeek V3 parser delimits its payload with
`V3_JSON_START = "\\n```json\\n"` (rust/src/tool-parser/src/deepseek_json/mod.rs). Folded
to ``get_weather ```json {...}`` that delimiter stops matching, the parser never enters
the JSON state, and `finish()` reports "incomplete DeepSeek V3 tool call". Qwen3 Coder
loses the newlines around a parameter value, so `<parameter=location>\\nNYC\\n</parameter>`
becomes `<parameter=location> NYC </parameter>` and the captured argument turns into
`{"location":" NYC "}` — which reads as a real parser quirk rather than a fixture bug.

THE REPAIR
----------
`\\n` -> ` ` is a 1-for-1 character substitution, so the joined length is unchanged and
the chunk boundaries still line up. The batch-v1 `model_text` each stream case is derived
from is intact, so the original text can be recovered exactly by re-splitting the batch
text at the SAME offsets. A case is only touched when it matches
`joined == model_text.replace("\\n", " ")`; anything else is left alone.

Gates (all assert, none are advisory):
  * the raw `- delta_text:` lines must line up 1:1 with the parsed chunks,
  * the re-split must consume the batch text exactly,
  * the rewritten file must reload to exactly the intended values.

AFTER RUNNING THIS
------------------
Repairing the inputs INVALIDATES every impl's recorded `expected` for the repaired
cases, because those were captured against the folded text. Re-capture each impl AT ITS
PUBLISHED VERSION before publishing, or the corpus will describe behavior on text that
no longer exists. `build_stream_fixtures._q` is already fixed, so the corruption does not
come back.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import yaml


def _batch_model_text(batch_inputs: pathlib.Path) -> dict[tuple[str, str], str]:
    """(family, batch case id) -> intact `model_text` from the batch-v1 corpus."""
    out: dict[tuple[str, str], str] = {}
    for f in batch_inputs.glob("*/TOOLCALLING.batch*.yaml"):
        doc = yaml.safe_load(f.read_text()) or {}
        for cid, case in (doc.get("cases") or {}).items():
            if isinstance(case, dict) and case.get("model_text") is not None:
                out[(f.parent.name, cid)] = case["model_text"]
    return out


def _batch_ref(case: dict) -> str:
    return (case.get("ref") or "").replace("derived from ", "").strip("` ")


def repair_file(path: pathlib.Path, batch: dict[tuple[str, str], str], apply: bool):
    """Return (cases_repaired, lines_rewritten); write the file when `apply`."""
    family = path.parent.name
    doc = yaml.safe_load(path.read_text()) or {}
    raw = path.read_text().split("\n")

    ordered: list[str] = []      # every chunk's delta_text, in document order
    repairs: dict[int, str] = {}  # index into `ordered` -> repaired value
    cases_repaired = 0

    for _cid, case in (doc.get("cases") or {}).items():
        deltas = [(ch.get("delta_text") or "") for ch in (case.get("chunks") or [])]
        base = len(ordered)
        ordered.extend(deltas)
        model_text = batch.get((family, _batch_ref(case)))
        if model_text is None:
            continue
        joined = "".join(deltas)
        if joined == model_text or "\n" not in model_text:
            continue
        if joined != model_text.replace("\n", " "):
            continue  # a genuine difference, not folding damage — leave it alone
        pos = 0
        for i, delta in enumerate(deltas):
            fixed = model_text[pos:pos + len(delta)]
            pos += len(delta)
            if fixed != delta:
                repairs[base + i] = fixed
        assert pos == len(model_text), f"{path}: re-split did not consume the batch text"
        cases_repaired += 1

    if not repairs:
        return 0, 0

    idxs = [i for i, line in enumerate(raw) if line.lstrip().startswith("- delta_text:")]
    assert len(idxs) == len(ordered), (
        f"{path}: {len(idxs)} `- delta_text:` lines vs {len(ordered)} parsed chunks; "
        "a multi-line scalar would make the line rewrite below unsafe"
    )

    for slot, fixed in repairs.items():
        line = raw[idxs[slot]]
        indent = line[: len(line) - len(line.lstrip())]
        # json.dumps is a valid YAML double-quoted scalar and round-trips losslessly.
        raw[idxs[slot]] = f"{indent}- delta_text: {json.dumps(fixed, ensure_ascii=False)}"

    out = "\n".join(raw)
    want = list(ordered)
    for slot, fixed in repairs.items():
        want[slot] = fixed
    got: list[str] = []
    for _cid, case in ((yaml.safe_load(out) or {}).get("cases") or {}).items():
        got.extend((ch.get("delta_text") or "") for ch in (case.get("chunks") or []))
    assert got == want, f"{path}: round-trip mismatch after repair"

    if apply:
        path.write_text(out)
    return cases_repaired, len(repairs)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stream-inputs", required=True,
                    help="extracted toolcalling/fixtures-stream-v2/inputs")
    ap.add_argument("--batch-inputs", required=True,
                    help="extracted toolcalling/fixtures-batch-v1/inputs (the intact source)")
    ap.add_argument("--apply", action="store_true",
                    help="write the repaired files (default: report only)")
    args = ap.parse_args()

    batch = _batch_model_text(pathlib.Path(args.batch_inputs))
    files = cases = lines = 0
    for f in sorted(pathlib.Path(args.stream_inputs).glob("*/TOOLCALLING.streamv2*.yaml")):
        c, n = repair_file(f, batch, args.apply)
        if n:
            files += 1
            cases += c
            lines += n
    verb = "repaired" if args.apply else "would repair"
    print(f"{verb}: {files} files, {cases} cases, {lines} delta_text lines")
    if not args.apply:
        print("re-run with --apply to write")
    else:
        print("NOW RE-CAPTURE every impl at its published version — the recorded "
              "`expected` values describe the OLD folded text.")


if __name__ == "__main__":
    main()
