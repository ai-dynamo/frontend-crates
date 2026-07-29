# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""`repair_stream_inputs` must restore folded newlines EXACTLY and touch nothing else.

The repair only works because `\\n` -> ` ` is a 1-for-1 substitution: the joined length is
unchanged, so the intact batch-v1 `model_text` can be re-split at the same chunk offsets.
These tests pin that property, and pin that a case which genuinely differs from its batch
source is left alone rather than "repaired" into something invented.
"""

import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import repair_stream_inputs as rsi  # noqa: E402

# The real DeepSeek V3 shape: the ```json fence the parser matches on is newline-delimited.
MODEL_TEXT = 'function<|sep|>get_weather\n```json\n{"location": "NYC"}\n```<|end|>'
FOLDED = MODEL_TEXT.replace("\n", " ")


def _write_corpus(tmp_path, chunks, model_text=MODEL_TEXT, ref="TOOLCALLING.batch.1"):
    stream = tmp_path / "stream" / "fam"
    batch = tmp_path / "batch" / "fam"
    stream.mkdir(parents=True)
    batch.mkdir(parents=True)
    (batch / "TOOLCALLING.batch.1.yaml").write_text(yaml.safe_dump(
        {"cases": {ref: {"model_text": model_text}}}, sort_keys=False, allow_unicode=True))
    lines = ["family: fam", "mode: streamv2", "cases:", "  TOOLCALLING.streamv2.1:",
             f"    ref: derived from {ref}", "    chunks:"]
    for c in chunks:
        # json.dumps is an unambiguous YAML double-quoted scalar — the corruption under
        # test is in the VALUE (spaces where newlines belong), not in how it is written.
        lines.append("    - delta_text: " + json.dumps(c, ensure_ascii=False))
    (stream / "TOOLCALLING.streamv2.1.yaml").write_text("\n".join(lines) + "\n")
    return stream, batch


def _joined(stream_dir):
    doc = yaml.safe_load((stream_dir / "TOOLCALLING.streamv2.1.yaml").read_text())
    case = doc["cases"]["TOOLCALLING.streamv2.1"]
    return "".join(ch.get("delta_text") or "" for ch in case["chunks"])


def test_restores_folded_newlines_across_chunk_boundaries(tmp_path):
    # Split mid-fence so the repair has to work per chunk, not just on the joined text.
    chunks = [FOLDED[:20], FOLDED[20:41], FOLDED[41:]]
    stream, batch = _write_corpus(tmp_path, chunks)
    assert _joined(stream) == FOLDED
    rsi.repair_file(stream / "TOOLCALLING.streamv2.1.yaml",
                    rsi._batch_model_text(batch.parent), apply=True)
    assert _joined(stream) == MODEL_TEXT
    assert "\n```json\n" in _joined(stream)


def test_chunk_lengths_are_preserved(tmp_path):
    chunks = [FOLDED[:20], FOLDED[20:41], FOLDED[41:]]
    stream, batch = _write_corpus(tmp_path, chunks)
    rsi.repair_file(stream / "TOOLCALLING.streamv2.1.yaml",
                    rsi._batch_model_text(batch.parent), apply=True)
    doc = yaml.safe_load((stream / "TOOLCALLING.streamv2.1.yaml").read_text())
    got = [len(ch["delta_text"]) for ch in doc["cases"]["TOOLCALLING.streamv2.1"]["chunks"]]
    assert got == [len(c) for c in chunks]


def test_leaves_an_uncorrupted_case_untouched(tmp_path):
    chunks = [MODEL_TEXT]  # already correct
    stream, batch = _write_corpus(tmp_path, chunks)
    before = (stream / "TOOLCALLING.streamv2.1.yaml").read_text()
    cases, lines = rsi.repair_file(stream / "TOOLCALLING.streamv2.1.yaml",
                                   rsi._batch_model_text(batch.parent), apply=True)
    assert (cases, lines) == (0, 0)
    assert (stream / "TOOLCALLING.streamv2.1.yaml").read_text() == before


def test_leaves_a_genuinely_different_case_untouched(tmp_path):
    """Not every mismatch is folding damage; only the exact \\n->space shape is repaired."""
    chunks = ["something else entirely"]
    stream, batch = _write_corpus(tmp_path, chunks)
    cases, lines = rsi.repair_file(stream / "TOOLCALLING.streamv2.1.yaml",
                                   rsi._batch_model_text(batch.parent), apply=True)
    assert (cases, lines) == (0, 0)
    assert _joined(stream) == "something else entirely"
