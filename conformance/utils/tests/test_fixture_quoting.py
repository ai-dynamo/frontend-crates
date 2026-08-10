# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""`build_stream_fixtures._q` must round-trip a scalar through YAML unchanged.

A single-quoted YAML scalar FOLDS an embedded newline into a space when read back, so
the previous `_q` silently corrupted every `delta_text` containing one. The concrete
damage: DeepSeek V3's tool-call payload is delimited by `V3_JSON_START = "\\n```json\\n"`
(vLLM `rust/src/tool-parser/src/deepseek_json/mod.rs`). Once folded to
`get_weather ```json {...}` the delimiter no longer matches, the parser never enters the
JSON state, and `finish()` reports "incomplete DeepSeek V3 tool call" -- turning 19
previously-passing conformance cases into errors that look like parser regressions.
"""

import sys
from pathlib import Path

import os

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import build_stream_fixtures  # noqa: E402
import capture_peer_versions  # noqa: E402  (import also applies yaml_fast's safe_dump patch)


class _Engine:
    name = "vllm_python"

# The real DeepSeek V3 shape is first: it is the case that actually broke.
ROUND_TRIP_CASES = [
    'get_weather\n```json\n{"location": "NYC"}\n```',
    "plain text, no quoting hazard",
    "it's got an apostrophe",
    'it has "double quotes"',
    "back\\slash",
    "tab\tseparated",
    "trailing newline\n",
    "\nleading newline",
    "multi\nline\nvalue",
    "<｜tool▁calls▁begin｜>unicode markers<｜tool▁call▁end｜>",
    "",
]


def _round_trip(value: str):
    """Emit `value` the way the fixture writer does, then read it back with YAML."""
    doc = yaml.safe_load("key: " + build_stream_fixtures._q(value) + "\n")
    return doc["key"]


def test_q_round_trips_every_scalar():
    for value in ROUND_TRIP_CASES:
        assert _round_trip(value) == value, f"_q corrupted {value!r}"


def test_q_preserves_the_deepseek_v3_json_fence():
    """The specific delimiter vLLM's DeepSeek V3 parser matches on must survive."""
    text = 'function<｜tool▁sep｜>get_weather\n```json\n{"location": "NYC"}\n```'
    assert "\n```json\n" in _round_trip(text)


def test_single_quoted_folding_is_why_this_test_exists():
    """Guard the assumption: a single-quoted scalar really does fold newlines."""
    folded = yaml.safe_load("key: 'a\nb'\n")["key"]
    assert folded == "a b"


def test_batch_write_survives_the_libyaml_safe_dump_patch(tmp_path):
    """`_QuotedStr` must be registered on the dumper `yaml.safe_dump` ACTUALLY uses.

    yaml_fast repoints `yaml.safe_dump` at `CSafeDumper` when libyaml is available, and
    importing any resolver applies that patch process-wide. Registering the representer
    only on `SafeDumper` made `_batch_write` raise
    `RepresenterError('cannot represent an object')` before writing a single byte -- the
    consolidated capturer's headline batch path never crossed its first serialization
    boundary. Importing capture_peer_versions here pulls the same patch in, so this test
    fails if the registration regresses to one dumper.
    """

    capture_peer_versions._STAGED_VERSION_ROOTS.clear()
    capture_peer_versions._batch_write(
        str(tmp_path), _Engine, "0.26.0", "hermes", "TOOLCALLING.batch.1.yaml", "batch",
        {"TOOLCALLING.batch.1.a": {"expected": {"vllm_python": {"calls": [], "normal_text": ""}}}},
    )
    capture_peer_versions._publish_staged()  # writes land in staging until the run publishes
    written = tmp_path / "vllm_python-0.26.0" / "hermes" / "TOOLCALLING.batch.1.yaml"
    assert written.exists(), "batch writer produced no file"
    # the version stays single-quoted, which is the whole point of _QuotedStr
    assert "vllm_python: '0.26.0'" in written.read_text()
    assert yaml.safe_load(written.read_text())["captured_with"] == {"vllm_python": "0.26.0"}


def test_version_dirs_are_append_only(tmp_path):
    """Publication is atomic: a run stages its tree and renames it into place at the end.

    Every writer used to do `makedirs(exist_ok=True)` + truncating `open(w)` directly into
    the final dir, so a re-run overwrote the files that still differed and left stale files
    from the earlier run that no longer did -- one version label, half old capture and half
    new. An `exists()` check alone did not fix it either: two processes both passed the
    check and both wrote. The rename is the real guard, so this test covers all three
    properties -- same-run reuse, second-run refusal, and no final dir after a mid-run
    failure.
    """
    case = {"TOOLCALLING.batch.1.a": {"expected": {"vllm_python": {"calls": [], "normal_text": ""}}}}
    root = tmp_path / "corpus"
    root.mkdir()

    capture_peer_versions._STAGED_VERSION_ROOTS.clear()
    capture_peer_versions._batch_write(str(root), _Engine, "0.26.0", "hermes", "b.yaml", "batch", case)
    capture_peer_versions._batch_write(str(root), _Engine, "0.26.0", "qwen25", "b.yaml", "batch", case)
    # nothing is visible at the final path until the run publishes
    assert not (root / "vllm_python-0.26.0").exists(), "published before the run finished"
    capture_peer_versions._publish_staged()
    assert (root / "vllm_python-0.26.0" / "qwen25" / "b.yaml").exists()
    assert not list(tmp_path.glob("corpus.staging-*")), "staging dir left behind"

    # a mid-run failure must leave NO final dir (staging is discarded, not promoted)
    capture_peer_versions._STAGED_VERSION_ROOTS.clear()
    capture_peer_versions._batch_write(str(root), _Engine, "0.27.0", "hermes", "b.yaml", "batch", case)
    capture_peer_versions._STAGED_VERSION_ROOTS.clear()  # simulate: run died before publish
    assert not (root / "vllm_python-0.27.0").exists()

    # a NEW run against the now-existing 0.26.0 dir refuses
    capture_peer_versions._STAGED_VERSION_ROOTS.clear()
    with pytest.raises(SystemExit, match="append-only"):
        capture_peer_versions._batch_write(str(root), _Engine, "0.26.0", "hermes", "b.yaml", "batch", case)


def test_two_concurrent_runs_cannot_both_publish(tmp_path):
    """The failure the `exists()` check could not catch: two runs racing the same version.

    Both stage happily -- staging paths are pid-scoped -- and then both try to rename onto
    the same final path. `os.rename` onto a non-empty directory raises ENOTEMPTY, so exactly
    one wins and the other fails loudly with its tree preserved.
    """
    root = tmp_path / "corpus"
    root.mkdir()
    final = root / "vllm_python-0.26.0"
    winner = tmp_path / "corpus.staging-A" / "vllm_python-0.26.0"
    loser = tmp_path / "corpus.staging-B" / "vllm_python-0.26.0"
    for d in (winner, loser):
        (d / "hermes").mkdir(parents=True)
        (d / "hermes" / "b.yaml").write_text("cases: {}\n")

    os.rename(winner, final)                       # first run publishes
    with pytest.raises(OSError):                   # second cannot clobber it
        os.rename(loser, final)
    assert (final / "hermes" / "b.yaml").exists()
    assert (loser / "hermes" / "b.yaml").exists(), "loser's capture must be preserved"
