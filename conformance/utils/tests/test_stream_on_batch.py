# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the conformance-table marker semantics.

Contract pinned here:

  * DEFAULT view (Conformance toggle OFF), every tab: per-engine status only —
    empty when clean, `↯` when the engine leaks tool-call markup, plus
    `…`/`n/a`/`·`/`!`/`—`. NEVER `=` or a divergence letter.
  * CONFORMANCE view (toggle ON):
      - batch tab: cross-engine `=`/`D_rb`/`V_pb`/`S_rb`; no `V_rb`.
      - stream tabs (batch-on-stream + TC stream v2): two dimensions per cell —
        COLOR (`data-status`) is each engine's STREAM parse vs its OWN BATCH parse
        (green consistent, red "problem" on divergence); MARKER
        (`data-marker-parity`) concatenates an own stream-vs-batch token (`X_rs`/`X_ps` when the
        engine's stream differs from its own batch) and cross-engine output
        tokens (`D_rs`/`V_ps`/`V_rs`/`S_rs`), `=` when all agree.

The stream tabs flow through the same render_cell_html/render_row_html/
render_html_panel pipeline as the others; only the `comparison` strategy differs.
"""

from __future__ import annotations

import itertools
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml

UTILS = Path(__file__).resolve().parents[1]
SRC = UTILS / "src"  # internal modules moved under conformance/utils/src/
REPO = UTILS.parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _fixtures_cache_root() -> Path:
    """HuggingFace fixture download cache root. Fixture YAMLs live on HF (not the
    repo) since DIS-2310, so path-existence checks resolve against the cache.
    `_common.sh` exports CONFORMANCE_FIXTURES_ROOT pointing here."""
    env = os.environ.get("CONFORMANCE_FIXTURES_ROOT")
    if env:
        return Path(env)
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg) if xdg else Path.home() / ".cache"
    return base / "dynamo/conformance-fixtures"


def _resolve_conformance_ref(ref: str) -> Path:
    """Map a `conformance/...` doc path to disk: fixture trees resolve against the HF
    cache (they're not in the repo), everything else against the repo."""
    rel = ref[len("conformance/"):]
    if rel.startswith(("toolcalling/fixtures", "reasoning/fixtures")):
        return _fixtures_cache_root() / rel
    return REPO / ref

import build_stream_fixtures as b  # noqa: E402
import capture_vllm_rust as r  # noqa: E402
import generate_conformance_table as g  # noqa: E402
import impls  # noqa: E402
import validate_fixtures as vf  # noqa: E402
from tests.parity.reasoning import table as reasoning_table  # noqa: E402

_TODO_MSG = "Dynamo parser v2 stream parser not yet implemented for this family"
# Identity comes from impls.py via the generator's re-export (audit B1).
D, R, V, S = g.IMPL_KEYS
IMPLS = g.IMPL_KEYS


def test_reasoning_python_exceptions_render_as_na() -> None:
    # Dynamo-as-reference: an na-stub (no Dynamo `expected`) shows n/a in the grid,
    # never peer parser exception markers (V✗/S✗). The exceptions still surface in the
    # tooltip. The dormant per-engine parser markers/status are unchanged.
    case = {
        "description": "No parser input",
        "reason": "not applicable",
    }

    marker, tooltip = reasoning_table._cell(case, "gpt_oss")
    assert marker == "n/a"
    assert "vLLM Python: parser exception" in tooltip
    assert "SGLang Python: parser exception" in tooltip

    assert reasoning_table._parser_marker(case, "gpt_oss", "dynamo") == "n/a"
    assert reasoning_table._parser_marker(case, "gpt_oss", "vllm") == "✗"
    assert reasoning_table._parser_marker(case, "gpt_oss", "sglang") == "✗"
    assert reasoning_table._overview_status(case, "gpt_oss", "vllm") == "problem"
    assert reasoning_table._overview_status(case, "gpt_oss", "sglang") == "problem"


def test_reasoning_python_exception_rendering_respects_missing_peer_parser() -> None:
    case = {
        "description": "No parser input",
        "reason": "not applicable",
    }

    # Grid marker is n/a regardless of which peers raised; the exception detail is in
    # the tooltip / dormant per-engine markers only.
    assert reasoning_table._cell(case, "kimi")[0] == "n/a"
    assert reasoning_table._parser_marker(case, "kimi", "vllm") == "n/a"
    assert reasoning_table._parser_marker(case, "kimi", "sglang") == "✗"


def test_reasoning_python_exception_cell_uses_na_class_and_tooltip() -> None:
    case = {
        "description": "No parser input",
        "reason": "not applicable",
    }
    refs = {("gpt_oss", "REASONING.batch.2.d"): reasoning_table.SCRIPT_DIR / "fake.yaml"}

    html = reasoning_table._render_cell_html(
        case,
        "gpt_oss",
        "REASONING.batch.2.d",
        refs,
    )

    assert 'class="cell na ' in html
    assert ">n/a<" in html
    assert ">V✗S✗<" not in html
    # Dormant per-engine data attributes still record the peer exceptions.
    assert 'data-marker-vllm="✗"' in html
    assert 'data-status-vllm="problem"' in html
    assert "Python parser exceptions" in html
    assert "KeyError: &#x27;model_text&#x27;" in html


def test_readme_documents_vllm_rust_capture_flow() -> None:
    text = (UTILS / "README.md").read_text(encoding="utf-8")
    for required in (
        "V_ps",
        "V_pb",
        "V_rs",
        "does not exist as a separate captured implementation",
        "VLLM_RUST_SOURCE",
        "capture_vllm_rust.py",
        "captured_with.vllm_rust",
        "expected.vllm_rust",
        "conformance/utils/render_table_v2.sh",
        "The verification-only path reads HF-downloaded fixtures and reports mismatches",
        "it does not run vLLM Rust",
        "conformance/utils/check.sh dynamo stream",
        "vLLM Python vs Rust is an HF fixture comparison",
        "capture.sh",
        "Parser Implementations",
        "Dynamo v1",
        "batch only",
        "Dynamo v2 Rust",
        "upcoming Dynamo-owned Rust stream parser",
        "vLLM Python",
        "batch and stream",
        "SGLang Python",
        "`capture.sh` is not the v1 batch rewrite tool",
        "Example: capture all",
        "Example: capture one",
        "Harmony fixture paths below are examples only",
        "Harmony is not the intended scope limit",
    ):
        assert required in text


def _markers(html: str) -> dict[str, str]:
    return dict(re.findall(r'(data-(?:status|marker(?:-parity)?)-\w+)="([^"]*)"', html))


def _xcase(expected: dict) -> dict:
    """Cross-engine case (batch/stream tab shape)."""
    return {
        "__family": "fam",
        "__case_id": "TOOLCALLING.batch.1",
        "__fixture_path": "toolcalling/fixtures/fam/TOOLCALLING.batch.1.yaml",
        "description": "demo",
        "model_text": "hello",
        "expected": expected,
    }


def _sobcase(stream: dict, batch: dict) -> dict:
    """Batch-on-stream case: stream output (`expected`) + batch reference."""
    return {
        "__family": "harmony",
        "__case_id": "TOOLCALLING.batch.5.d",
        "__fixture_path": "toolcalling/fixtures/harmony/TOOLCALLING.batch.5.yaml",
        "description": "demo",
        "model_text": "hello",
        "expected": stream,
        "batch_expected": batch,
    }


def _calls(*names):
    return {"calls": [{"name": n, "arguments": {}} for n in names], "normal_text": ""}


# --------------------------------------------------------------------------- #
# build_stream_fixtures: source checkout metadata
# --------------------------------------------------------------------------- #
def test_vllm_rust_batch_capture_groups_by_parser(monkeypatch, tmp_path) -> None:
    fixture_one = tmp_path / "one.yaml"
    fixture_two = tmp_path / "two.yaml"
    fixture_one.write_text(
        yaml.safe_dump({"cases": {"case.a": {"model_text": "a", "tools": []}}}),
        encoding="utf-8",
    )
    fixture_two.write_text(
        yaml.safe_dump({"cases": {"case.b": {"model_text": "b", "tools": []}}}),
        encoding="utf-8",
    )
    calls = []

    def fake_run_probe(_source: str, payload: dict, _work: str | None) -> dict:
        calls.append(payload)
        return {case_id: {"parser": payload["parser"]} for case_id in payload["cases"]}

    monkeypatch.setattr(r, "_run_probe", fake_run_probe)
    out = r._run_batch(
        "/tmp/vllm",
        "batch-on-stream",
        [
            {"fixture": str(fixture_one), "parser": "hermes"},
            {"fixture": str(fixture_two), "parser": "hermes"},
            {"fixture": str(fixture_two), "parser": "mistral"},
        ],
        None,
    )

    assert [payload["parser"] for payload in calls] == ["hermes", "mistral"]
    assert len(calls[0]["cases"]) == 2
    assert out[str(fixture_one)]["cases"]["case.a"] == {"parser": "hermes"}
    assert out[str(fixture_two)]["cases"]["case.b"] == {"parser": "mistral"}


def test_build_stream_fixture_records_vllm_rust_source(monkeypatch, tmp_path) -> None:
    source_root = tmp_path / "vllm-src"
    (source_root / "rust/src/tool-parser").mkdir(parents=True)
    (source_root / "rust/src/tool-parser/Cargo.toml").write_text(
        "[package]\nname = 'vllm-tool-parser'\n",
        encoding="utf-8",
    )
    source = tmp_path / "source.yaml"
    source.write_text(
        yaml.safe_dump(
            {
                "family": "hermes",
                "model_label": "Hermes",
                "mode": "stream",
                "cases": {
                    "TOOLCALLING.stream.1": {
                        "description": "demo",
                        "chunks": [{"delta_text": "hello"}],
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    out = tmp_path / "out.yaml"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_stream_fixtures.py",
            "--source",
            str(source),
            "--out",
            str(out),
            "--vllm-rust-source",
            str(source_root),
        ],
    )

    b.main()

    doc = yaml.safe_load(out.read_text(encoding="utf-8"))
    captured = doc["captured_with"]["vllm_rust"]
    assert "untagged unknown" in captured
    assert str(source_root.resolve()) not in captured
    unavailable = doc["cases"]["TOOLCALLING.stream.1"]["unavailable"]["vllm_rust"]
    assert "source checkout is available for the Rust probe" in unavailable
    assert str(source_root.resolve()) not in unavailable


def test_build_stream_fixture_uses_vllm_rust_capture(monkeypatch, tmp_path) -> None:
    source_root = tmp_path / "vllm-src"
    (source_root / "rust/src/tool-parser").mkdir(parents=True)
    (source_root / "rust/src/tool-parser/Cargo.toml").write_text(
        "[package]\nname = 'vllm-tool-parser'\n",
        encoding="utf-8",
    )
    source = tmp_path / "source.yaml"
    source.write_text(
        yaml.safe_dump(
            {
                "family": "hermes",
                "model_label": "Hermes",
                "mode": "stream",
                "cases": {
                    "TOOLCALLING.stream.1": {
                        "description": "demo",
                        "chunks": [{"delta_text": "hello"}],
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    capture = tmp_path / "vllm_rust.json"
    capture.write_text(
        json.dumps(
            {
                "TOOLCALLING.stream.1": [
                    {
                        "deltas": [
                            {"index": 0, "name": "get_weather"},
                            {"index": 0, "arguments": '{"location":"NYC"}'},
                        ],
                        "normal_text": "",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "out.yaml"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_stream_fixtures.py",
            "--source",
            str(source),
            "--out",
            str(out),
            "--vllm-rust",
            str(capture),
            "--vllm-rust-source",
            str(source_root),
        ],
    )

    b.main()

    doc = yaml.safe_load(out.read_text(encoding="utf-8"))
    case = doc["cases"]["TOOLCALLING.stream.1"]
    assert "unavailable" not in case
    assert case["chunks"][0]["expected"]["vllm_rust"] == [
        {"index": 0, "name": "get_weather"},
        {"index": 0, "arguments": '{"location":"NYC"}'},
    ]


# --------------------------------------------------------------------------- #
# _stream_on_batch_expected: overlay -> standard expected block
# --------------------------------------------------------------------------- #
def test_expected_dynamo_absent_renders_as_todo() -> None:
    exp = g._stream_on_batch_expected({"vllm": {"calls": []}, "sglang": {"calls": []}})
    assert g._is_todo_unavailable(exp[D])
    assert "unavailable" in exp[R]
    # New captures write the `explanation` key (not the legacy `reason`).
    assert "explanation" in exp[V] and "explanation" in exp[S]
    assert "reason" not in exp[V] and "reason" not in exp[S]
    assert "SGLang Python streaming parser" in exp[S]["explanation"]


def test_expected_dynamo_absent_without_batch_text_is_structural_na() -> None:
    exp = g._stream_on_batch_expected(
        {"vllm": {"unavailable": "No batch model_text for this case."}},
        has_batch_text=False,
    )
    assert exp[D] == {"unavailable": "No batch model_text for this case."}
    assert not g._is_todo_unavailable(exp[D])


def test_expected_dynamo_present_and_peer_unavailable() -> None:
    exp = g._stream_on_batch_expected(
        {
            D: {"calls": [{"name": "f", "arguments": {}}], "normal_text": ""},
            V: {"calls": []},
            S: {"unavailable": "SGLang has no detector for family"},
        }
    )
    assert exp[D]["calls"] == [{"name": "f", "arguments": {}}]
    assert "explanation" not in exp[D] and "reason" not in exp[D]
    assert exp[S] == {"unavailable": "SGLang has no detector for family"}


def test_explanation_and_legacy_reason_both_recognized() -> None:
    # Backward-compat: the divergence note reads from `explanation` (current) or the
    # legacy `reason` (older fixtures / Dynamo-synced code); explanation wins.
    assert g._explanation({"explanation": "new"}) == "new"
    assert g._explanation({"reason": "old"}) == "old"
    assert g._explanation({"explanation": "new", "reason": "old"}) == "new"
    assert g._explanation({}) is None
    # A divergent peer carrying EITHER key is treated as intentional (marker without
    # the research-needed `?`), not as an un-triaged gap.
    dyn = {"calls": [{"name": "f", "arguments": {}}], "normal_text": ""}
    for key in ("reason", "explanation"):
        case = _xcase({D: dyn, V: {"calls": [], "normal_text": "", key: "intentional"}})
        kind, unknown = g.peer_status(case, dyn, V)
        assert kind == "div" and unknown is False, key


def test_dsv4_v2_parser_cell_links_dsml_parser() -> None:
    html = g._parser_cell_html(
        "deepseek_v4",
        {},
        set(),
        set(),
        {},
        stream_context="streamv2",
    )
    assert "DeepSeekV4ToolStreamParser text path" in html
    assert "parsers/v2/src/tool_calling/dsml.rs" in html
    assert "not implemented" not in html


# --------------------------------------------------------------------------- #
# DEFAULT marker is leak-only on every tab (the reported bug)
# --------------------------------------------------------------------------- #
def test_parser_marker_never_emits_equals_or_divergence_letters() -> None:
    allowed = {"", "↯", "!", "·", "…", "n/a"}
    blocks = [
        {"calls": [], "normal_text": ""},
        {"calls": [{"name": "f", "arguments": {"a": 1}}], "normal_text": ""},
        {"unavailable": "structural n/a"},
        {"unavailable": _TODO_MSG},
        {"error": "boom"},
        {"calls": [], "normal_text": "<tool_call>leak"},
    ]
    for combo in itertools.product(blocks, repeat=len(IMPLS)):
        case = _xcase(dict(zip(IMPLS, combo)))
        for impl in IMPLS:
            assert g._parser_marker(case, impl) in allowed


# --------------------------------------------------------------------------- #
# stream tabs: 2-D cell — COLOR = stream-vs-own-batch, MARKER = cross-engine outputs
# --------------------------------------------------------------------------- #
def test_sob_cell_two_dimensions_color_and_cross_engine_marker() -> None:
    # Stream outputs: dynamo got nothing, vllm/sglang each got [f].
    # Batch refs:     dynamo recovered [f], vllm [f], sglang got nothing.
    # COLOR (stream-vs-own-batch): dynamo stream([]) != batch([f]) -> problem;
    #   vllm stream([f]) == batch([f]) -> ok; sglang stream([f]) != batch([]) -> problem.
    # MARKER (cross-engine batch-on-stream): dynamo([]) differs from vllm & sglang -> V_psS_rs;
    #   vllm([f]) differs only from dynamo -> D_rs; sglang([f]) differs only from dynamo -> D_rs.
    case = _sobcase(
        stream={
            D: _calls(),  # stream got nothing
            V: _calls("f"),
            S: _calls("f"),
        },
        batch={
            D: _calls("f"),  # batch recovered it
            V: _calls("f"),
            S: _calls(),
        },
    )
    html = g.render_cell_html(
        case, "batch", "harmony", "5.d", "stream", "stream_vs_batch", "batch_on_stream"
    )
    m = _markers(html)
    # DEFAULT: leak-only (all clean) -> empty, never a letter
    assert m[f"data-marker-{D}"] == ""
    assert m[f"data-marker-{V}"] == ""
    assert m[f"data-marker-{S}"] == ""
    # COLOR: stream-vs-own-batch divergence is red (problem); agreement is green (ok)
    assert m[f"data-status-{D}"] == "problem"
    assert m[f"data-status-{V}"] == "ok"
    assert m[f"data-status-{S}"] == "problem"
    # MARKER (Conformance): own stream-vs-batch token (X_rs/X_ps when stream != own batch) +
    # cross-engine stream outputs. dynamo: diverges from its batch (D_rs) and from both
    # peer stream outputs (V_psS_rs). vllm: matches its batch, differs only from dynamo
    # (D_rs). sglang: diverges from its batch (S_rs), differs only from dynamo stream output (D_rs).
    assert m[f"data-marker-parity-{D}"] == "D_rsV_psS_rs"
    assert m[f"data-marker-parity-{V}"] == "D_rs"
    assert m[f"data-marker-parity-{S}"] == "S_rsD_rs"
    assert "D<sub>RS</sub>V<sub>PS</sub>S<sub>RS</sub>" in html
    assert "V<sub>RB</sub>" not in html
    assert "\u1d66" not in html
    # Compare model: coloring is leak-only, so the global cross-impl
    # "X output diverges from X batch" blob is gone (it named engines regardless of
    # the Base/Compare selection); per-candidate reasons live in their own sections.
    assert "output diverges from" not in html

    stream_html = g.render_cell_html(
        case, "streamv2", "harmony", "5.d", "stream", "stream_vs_batch", "streamv2"
    )
    stream_markers = _markers(stream_html)
    assert stream_markers[f"data-marker-parity-{D}"] == "D_rsV_psS_rs"
    assert "V<sub>PS</sub>S<sub>RS</sub>" in stream_html


def test_sob_marker_all_consistent_is_equals_and_green() -> None:
    case = _sobcase(
        stream={i: _calls("f") for i in IMPLS},
        batch={i: _calls("f") for i in IMPLS},
    )
    m = _markers(
        g.render_cell_html(
            case, "batch", "harmony", "1", "stream", "stream_vs_batch", "batch_on_stream"
        )
    )
    for impl in IMPLS:
        assert m[f"data-marker-{impl}"] == ""  # default: nothing
        assert m[f"data-marker-parity-{impl}"] == "="  # conformance: consistent
        assert m[f"data-status-{impl}"] == "ok"  # green


def test_sob_dynamo_na_when_no_v2_parser() -> None:
    # A family the Dynamo v2 stream parser doesn't implement is a plain neutral n/a
    # (the v1 table has no "TODO" concept), not a distinct orange "todo"/"…" state.
    case = _sobcase(
        stream={
            D: {"unavailable": _TODO_MSG},
            V: _calls("f"),
            S: _calls("f"),
        },
        batch={i: _calls("f") for i in (D, V, S)},
    )
    assert g._sob_status(case, D) == "na"
    # dynamo unavailable -> marker is a clean n/a, no distinct "…" TODO marker
    assert g._stream_xeng_marker(case, D, "batch_on_stream") == "n/a"
    # vllm/sglang streams both present and agree -> cross-engine '='
    assert g._stream_xeng_marker(case, V, "batch_on_stream") == "="


def test_sob_leak_marks_red_and_lightning() -> None:
    case = _sobcase(
        stream={
            D: {"calls": [], "normal_text": "<tool_call>leaked"},
            V: _calls("f"),
            S: _calls("f"),
        },
        batch={i: _calls("f") for i in (D, V, S)},
    )
    m = _markers(
        g.render_cell_html(
            case, "batch", "harmony", "1", "stream", "stream_vs_batch", "batch_on_stream"
        )
    )
    assert m[f"data-marker-{D}"] == "↯"  # leak shows in default
    assert m[f"data-status-{D}"] == "problem"  # red


def test_vllm_python_leak_marks_red_and_lightning() -> None:
    case = _xcase(
        {
            D: {"calls": [], "normal_text": ""},
            V: {"calls": [], "normal_text": "<tool_call>leaked", "reason": "leak"},
            S: {"calls": [], "normal_text": ""},
        }
    )
    m = _markers(g.render_cell_html(case, "batch", "fam", "1", "batch", "cross_engine"))
    assert m[f"data-status-{V}"] == "problem"
    assert m[f"data-marker-{V}"] == "↯"
    assert m[f"data-marker-parity-{V}"] == "↯D_rbS_rb"


def test_sob_tooltip_labels_stream_and_batch() -> None:
    case = _sobcase(
        stream={i: _calls("f") for i in IMPLS},
        batch={i: _calls("f") for i in IMPLS},
    )
    ttip = g._build_sob_tooltip(case)
    # Sections now use the standardized "<Engine> <Runtime> <version> (<mode>)" candidate
    # label (same key as the Base/Compare buckets), wrapped as cand-<impl> so the
    # selection can toggle them. The version comes from fixture provenance and may be
    # absent for a synthetic case with no captured_with, so match it optionally.
    for lbl in ("Dynamo Rust", "vLLM Rust", "vLLM Python", "SGLang Python"):
        assert re.search(rf"{re.escape(lbl)}(?: \S+)? \(stream\):", ttip), f"missing {lbl} (stream)"
    for lbl in ("Dynamo Rust", "vLLM Python", "SGLang Python"):
        assert re.search(rf"{re.escape(lbl)}(?: \S+)? \(batch\):", ttip), f"missing {lbl} (batch)"
    assert not re.search(r"vLLM Rust(?: \S+)? \(batch\):", ttip)
    assert "V<sub>RB</sub>" not in ttip
    assert "cand cand-vllm_rust" in ttip


def test_stream_v2_x_marker_shows_vllm_rust_error_message() -> None:
    error = "vLLM Rust parser not captured: tool parser parsing failed: invalid Hermes"
    case = {
        "__family": "hermes",
        "__case_id": "TOOLCALLING.streamv2.4.a",
        "__fixture_path": "toolcalling/fixtures-stream-v2/hermes/TOOLCALLING.streamv2.4.yaml",
        "description": "demo",
        "unavailable": {R: error},
        "expected": {
            D: {"calls": [], "normal_text": ""},
            R: {"unavailable": error},
            V: {"calls": [], "normal_text": ""},
            S: {"calls": [], "normal_text": ""},
        },
        "chunks": [
            {
                "delta_text": "<tool_call>not json</tool_call>",
                "expected": {D: [], V: [], S: []},
            }
        ],
        "batch_expected": {
            D: {"calls": [], "normal_text": ""},
        },
    }

    html = g.render_cell_html(
        case,
        "streamv2",
        "hermes",
        "4.a",
        "stream",
        "stream_vs_batch",
        "streamv2",
    )

    assert f'data-marker-{R}="✗"' in html
    assert f'data-status-{R}="problem"' in html
    assert '<div class="ttip-unavail">' in html
    assert error in html


# --------------------------------------------------------------------------- #
# cross-engine conformance (batch / stream tabs)
# --------------------------------------------------------------------------- #
def test_cross_engine_all_three_agree() -> None:
    case = _xcase({i: {"calls": [], "normal_text": ""} for i in (D, V, S)})
    assert g._selected_parity_marker(case, D) == "="


def test_cross_engine_requires_all_three_present() -> None:
    case = _xcase(
        {
            D: {"unavailable": "x"},
            V: {"calls": [{"name": "f", "arguments": {}}], "normal_text": ""},
            S: {"calls": [], "normal_text": ""},
        }
    )
    # The selected parser can still compare against available peers when vLLM Rust
    # is unavailable; this keeps the Conformance toggle useful before Rust capture lands.
    assert g._selected_parity_marker(case, V) == "S_rb"


def test_cross_engine_divergence_letters() -> None:
    case = _xcase(
        {
            D: {"calls": [{"name": "f", "arguments": {}}], "normal_text": ""},
            V: {"calls": [], "normal_text": ""},
            S: {"calls": [], "normal_text": ""},
        }
    )
    assert g._selected_parity_marker(case, D) == "V_pbS_rb"
    assert g._selected_parity_marker(case, V) == "D_rb"


# --------------------------------------------------------------------------- #
# _build_stream_on_batch_cases: overlay + batch reference
# --------------------------------------------------------------------------- #
def test_build_cases_carries_stream_and_batch(monkeypatch) -> None:
    batch_cases = {
        ("fam", "1"): {
            "__case_id": "TOOLCALLING.batch.1",
            "__fixture_path": "toolcalling/fixtures/fam/TOOLCALLING.batch.1.yaml",
            "description": "d",
            "model_text": "x",
            "expected": {D: _calls("f"), V: _calls("f"), S: _calls("f")},
        },
        ("fam", "3"): {
            "__case_id": "TOOLCALLING.batch.3",
            "__fixture_path": "toolcalling/fixtures/fam/TOOLCALLING.batch.3.yaml",
        },
    }
    overlay = {("fam", "TOOLCALLING.batch.1"): {V: {"calls": []}, S: {"calls": []}}}
    monkeypatch.setattr(g, "_load_stream_on_batch_overlay", lambda: overlay)

    cases = g._build_stream_on_batch_cases(batch_cases)
    assert ("fam", "1") in cases and ("fam", "3") not in cases
    built = cases[("fam", "1")]
    assert built["model_text"] == "x"
    assert g._is_todo_unavailable(built["expected"][D])  # dynamo absent -> todo
    assert built["batch_expected"][V] == _calls("f")  # batch reference carried
    # COLOR: vllm stream calls=[] vs batch calls=[f] -> diverge -> problem (red)
    assert g._sob_status(built, V) == "problem"
    # MARKER: vllm diverges from its own batch (V_ps); dynamo todo, sglang stream also
    # empty so no cross-engine token.
    assert g._stream_xeng_marker(built, V, "batch_on_stream") == "V_ps"


def test_template_has_compare_picker_and_reasoning_candidates() -> None:
    # The compare bar is a SHARED Jinja partial (used by both the v2 conformance
    # table and the v1 parity page): one column per engine, each parser row a
    # Reference radio + Compare-with checkbox. The drag/drop buckets are gone.
    template = (SRC / "conformance_table.html.j2").read_text()
    assert '{% include "_compare_bar.html.j2" %}' in template
    assert "data-bucket" not in template  # old drag/drop buckets are gone
    partial = (UTILS / "tests" / "parity" / "_compare_bar.html.j2").read_text()
    assert 'data-engine="Dynamo"' in partial or "'Dynamo'" in partial
    assert 'class="cmp-ref"' in partial  # Reference radio
    assert 'class="cmp-on"' in partial  # Compare-with checkbox
    assert ">compare with<" in partial.lower()
    assert "data-cmp-base" not in partial  # old radio-based control is gone
    # The compare JS drives cells from each cell's data-cmp payload.
    js = (SRC / "assets" / "conformance.js").read_text()
    assert "function applyCtl" in js
    assert "cmp-leak" in js and "cmp-eq" in js
    assert "function initCompareInputs" in js  # radio/checkbox wiring replaced DnD
    hrefs = {
        "reasoning_fixtures": "#",
        "reasoning_cases": "#",
        "reasoning_src": "#",
        "toolcalling_src": "#",
        "streaming_harmony_src": "#",
        "streaming_src": "#",
        "toolcalling_streaming_cases": "#",
        "toolcalling_cases": "#",
        "pyproject_stub": "#",
    }
    panels = g._combined_reasoning_panels(hrefs)
    assert {panel["id"] for panel in panels} == {"tab-reasoning-batch", "tab-reasoning-stream"}
    # No vLLM Rust for reasoning (parser_options already excludes it).
    assert all(panel["parser_options"] == ("dynamo_rust", "vllm_python", "sglang_python") for panel in panels)


def test_template_overview_cells_do_not_expand_from_hidden_marker_text() -> None:
    # Static styles now live in the CSS asset, inlined at render (audit B7).
    css = (SRC / "assets" / "conformance.css").read_text()
    assert "td.cell { text-align: center; width: 44px; min-width: 44px; max-width: 44px;" in css
    assert ".view-overview td.cell { font-size: 0; line-height: 0; }" in css
    assert ".view-overview td.cell .cell-marker { display: none; }" in css
    assert ".view-overview td.cell .ttip { font-size: 12px; line-height: 1.4; }" in css
    assert ".view-details td.cell > a { height: 16px; overflow: hidden; white-space: nowrap; }" in css


def test_template_detail_cells_fit_conformance_markers() -> None:
    css = (SRC / "assets" / "conformance.css").read_text()
    assert ".view-details td.cell { color: transparent; }" in css
    assert ".view-details td.cell .cell-marker .marker-text { display: inline-block; color: inherit; white-space: nowrap; }" in css


def test_conformance_marker_hide_rule_ordered_after_status_show() -> None:
    """In Conformance mode the per-engine status markers and the cross-engine parity
    markers are absolutely positioned in the same cell box. The rule that hides the
    status marker (`.view-details.parity-mode td.cell .cell-marker { display: none }`)
    has equal specificity to the status-marker show rules, so it MUST appear after
    them and before the parity-marker show rules, or both render and visibly overlap.
    Regression guard for the B7 asset-extraction reorder. The rule lives in the
    generated CSS (one source of truth), not the static asset."""
    css = g._impl_status_css()
    lines = css.splitlines()
    hide = [i for i, l in enumerate(lines)
            if l.strip() == ".view-details.parity-mode td.cell .cell-marker { display: none; }"]
    status_show = [i for i, l in enumerate(lines)
                   if ".marker-" in l and "parity" not in l and "display: flex" in l]
    parity_show = [i for i, l in enumerate(lines)
                   if "marker-parity-" in l and "display: flex" in l]
    assert hide, "hide-in-parity rule must be emitted by _impl_status_css()"
    assert max(status_show) < hide[0] < min(parity_show), (
        "hide-in-parity rule must sit after the status-show rules and before the parity-show rules"
    )
    # One source of truth: the static asset must NOT also carry this rule (it would
    # inline before the generated block and lose the specificity tie).
    asset = (SRC / "assets" / "conformance.css").read_text()
    assert ".parity-mode td.cell .cell-marker { display: none; }" not in asset


def test_tab_labels_put_version_after_family() -> None:
    plain, html = g._tab_label("TC", "stream", "stream", True)
    assert plain == "TC v2 (stream data on stream-parser)"
    assert html.startswith('TC v2 <span class="tab-sub">')
    assert "(v2)" not in plain

    # Reasoning has a single parser, so its tab drops the "on <parser>-parser" clause
    # and shows the data word only: "(batch data)" / "(stream data)".
    plain, html = g._tab_label("Reasoning", "batch", None, False, on_parser=False)
    assert plain == "Reasoning v1 (batch data)"
    assert html.startswith('Reasoning v1 <span class="tab-sub">')
    assert "on parser" not in plain and "on parser" not in html
    stream_plain, _ = g._tab_label("Reasoning", "stream", None, False, on_parser=False)
    assert stream_plain == "Reasoning v1 (stream data)"


def test_common_legend_defines_v1_v2() -> None:
    legend = g._common_legend_html()
    assert "<strong>v1</strong> = the stable batch parser crate" in legend
    assert "<strong>v2</strong> = the WIP streaming parser crate" in legend
    assert "<code>parsers/v1/src/...</code>" in legend
    assert "<code>parsers/v2/src/...</code>" in legend


def test_common_legend_defines_green_by_reference_cleanliness() -> None:
    # Green is defined by the Reference parser being leak-free (compare-model), not by
    # the old "all peers match Dynamo Rust". The stale per-impl marker keys (D_rb,
    # V_ps, …) and the "match Dynamo Rust" / donly lines were removed.
    legend = g._common_legend_html()
    assert "Reference</strong> parser output is clean" in legend
    assert "whether or not any Compare parser is selected" in legend
    assert "all captured peers match Dynamo Rust" not in legend
    assert "Dynamo Rust batch parser" not in legend
    assert "Dynamo Rust-only fixture" not in legend


def test_template_cells_do_not_clip_hover_tooltips() -> None:
    # CSS and JS now live in static assets, inlined at render (audit B7).
    css = (SRC / "assets" / "conformance.css").read_text()
    js = (SRC / "assets" / "conformance.js").read_text()
    cell_rule = re.search(r"td\.cell \{([^}]*)\}", css)
    assert cell_rule is not None
    assert "overflow: hidden" not in cell_rule.group(1)
    assert ".ttip-visible" in css
    assert "cell.addEventListener('pointerenter', scheduleShow);" in js


def test_toolcalling_parser_options_are_mode_specific() -> None:
    assert g._impl_keys_for_output_kind("batch") == (D, V, S)
    assert g._impl_keys_for_output_kind("stream") == (D, R, V, S)


_DOC_FILES = (
    UTILS / "README.md",
    REPO / "parsers" / "v2" / "README.md",
    REPO / "conformance" / "README.md",
    REPO / "conformance" / "toolcalling" / "fixtures-stream-v2" / "README.md",
)
_STALE_COMMAND_NAMES = (
    "capture_v2.sh", "check_v2.sh",
    "test_stream_on_batch_v2.py", "record_v2.sh", "capture_all_families.sh",
)


def test_readme_fixture_paths_exist() -> None:
    """D1: every concrete conformance/*.yaml path in the doc set resolves (A2 regression).

    Fixture YAMLs live on HuggingFace (not the repo) since DIS-2310, so fixture paths
    resolve against the download cache; skip if the cache isn't populated yet."""
    if not (_fixtures_cache_root() / "toolcalling").is_dir():
        import pytest

        pytest.skip("fixtures not downloaded (run download_fixtures.py)")
    for doc in _DOC_FILES:
        if not doc.exists():
            continue
        for ref in re.findall(r"conformance/[\w./*<>-]+\.yaml", doc.read_text()):
            if "<" in ref or "*" in ref:  # placeholder/glob, not a concrete path
                continue
            assert _resolve_conformance_ref(ref).exists(), (
                f"{doc.name}: missing fixture path {ref}"
            )


def test_repo_docs_have_no_stale_command_names() -> None:
    """D1/A5: removed *_v2 / old command names must not reappear in the doc set."""
    for doc in _DOC_FILES:
        if not doc.exists():
            continue
        text = doc.read_text()
        for name in _STALE_COMMAND_NAMES:
            assert name not in text, f"{doc.name}: stale command name {name}"


def test_check_sh_dry_run_all_covers_every_parser() -> None:
    """D2: `check.sh all --dry-run` exercises all three Dynamo targets and the
    peer engines, and the peer-failure opt-out flag exists."""
    out = subprocess.run(
        ["bash", str(UTILS / "check.sh"), "all", "--dry-run"],
        capture_output=True, text=True,
    )
    txt = out.stdout + out.stderr
    for target in ("parity_toolcalling", "parity_toolcalling_stream", "parity_toolcalling_batch_via_stream"):
        assert target in txt, f"check.sh all dry-run missing {target}"
    assert "vllm" in txt and "sglang" in txt
    assert "--allow-peer-failures" in (UTILS / "check.sh").read_text()


def test_check_sh_all_does_not_suppress_failures() -> None:
    """D2: the `all` block fails on parser failures (no `|| true`) and exits with rc
    unless --allow-peer-failures is set."""
    all_block = (UTILS / "check.sh").read_text().split("\n  all)\n", 1)[1].split("\n  *)", 1)[0]
    assert "|| true" not in all_block
    assert "run_dynamo all || rc=1" in all_block
    assert "|| peer_rc=1" in all_block
    assert 'exit "$rc"' in all_block


def test_structured_error_block_marks_x_string_error_marks_bang() -> None:
    """B11: a structured (dict) `error` = a peer parser ran and threw -> `✗`; a
    plain-string `error` is a declared expected-error -> `!`. The failure marker the
    capture stamps is the shared PARSER_NOT_CAPTURED contract, not a private regex."""
    failed = {"expected": {"vllm_rust": {"error": {"kind": "parse_error", "message": "boom"}}}}
    assert g._parser_marker(failed, "vllm_rust") == "✗"
    expected_err = {"expected": {"vllm_rust": {"error": "KeyError: name"}}}
    assert g._parser_marker(expected_err, "vllm_rust") == "!"
    # the renderer detects the shared contract the capture stamps (not a private guess)
    assert g._PARSER_ERROR_RE.search(f"vLLM Rust '{impls.PARSER_NOT_CAPTURED}': boom")


def test_v2_overlays_are_canonical_only() -> None:
    """D3: the v2 overlays carry no legacy impl keys and stamp captured_with — locks
    the canonical-key migration (the renderer's legacy aliases exist only for the
    Dynamo-synced v1 corpus, so legacy keys here are silent drift)."""
    assert vf.validate_overlays() == []


def test_every_stream_family_has_registry_row_and_fixtures() -> None:
    """D6: each fixtures-stream-v2/inputs/<family> has a parser_families.yaml row, and
    each Dynamo-v2 family in the registry has at least one stream input fixture.

    The stream corpus is versioned like the batch corpus (no unversioned anchor):
    families live under `inputs/`; the sibling `<impl>-<version>/` dirs are per-impl
    expected, not families (resolve_stream_fixtures.py folds them into the inputs)."""
    registry = yaml.safe_load((SRC / "parser_families.yaml").read_text())["families"]
    # Fixtures live on HuggingFace since DIS-2310; resolve against the download cache.
    inputs_root = _fixtures_cache_root() / "toolcalling" / "fixtures-stream-v2" / "inputs"
    if not inputs_root.is_dir():
        import pytest

        pytest.skip("fixtures not downloaded (run download_fixtures.py)")
    for fam_dir in sorted(p for p in inputs_root.iterdir() if p.is_dir()):
        assert fam_dir.name in registry, f"family {fam_dir.name} has no parser_families.yaml row"
    for fam, spec in registry.items():
        if spec.get("dynamo_v2"):
            assert list((inputs_root / fam).glob("*.yaml")), f"dynamo_v2 family {fam} has no stream fixtures"


def test_impl_spec_is_single_identity_source() -> None:
    """D5: ImplSpec is the one identity table; the generator's derived dicts match
    it, every spec is complete, markers/displays are unique, and vLLM Rust has no
    batch (`V_rb`) parser option. Catches the marker/legend/alias drift class."""
    assert tuple(s.key for s in impls.IMPL_SPECS) == g.IMPL_KEYS
    for s in impls.IMPL_SPECS:
        assert s.display and s.marker_letter and s.marker_lang and s.engine and s.language
        assert g.IMPL_DISPLAY[s.key] == s.display
        assert g.ENGINE_LETTER[s.key] == s.marker_letter
        assert g.IMPL_LANG_MARKER[s.key] == s.marker_lang
        assert s.legacy_key is None or impls.LEGACY_IMPL_ALIASES[s.legacy_key] == s.key
    assert len({s.marker_letter for s in impls.IMPL_SPECS}) == len(impls.IMPL_SPECS)
    assert len({s.display for s in impls.IMPL_SPECS}) == len(impls.IMPL_SPECS)
    # vLLM Rust is stream-only: no `V_rb` batch parser option exists anywhere.
    assert "vllm_rust" not in g.BATCH_IMPL_KEYS
    assert "vllm_rust" in g.STREAM_IMPL_KEYS


def test_candidate_label_html_colors_mode_word() -> None:
    """Compare candidate labels color the trailing mode word: (batch) maroon,
    (stream) NVIDIA green — via cand-batch / cand-stream spans, still HTML-escaped."""
    assert (
        g._candidate_label_html("Dynamo v1 Rust 3.0.0 (batch)")
        == 'Dynamo v1 Rust 3.0.0 (<span class="cand-batch">batch</span>)'
    )
    assert (
        g._candidate_label_html("vLLM Rust 0.23.0 (stream)")
        == 'vLLM Rust 0.23.0 (<span class="cand-stream">stream</span>)'
    )
    # Only the trailing mode parenthetical is recolored; escaping still applies.
    assert g._candidate_label_html("A & B (batch)").startswith("A &amp; B (")
    # No mode parenthetical -> unchanged (but escaped).
    assert g._candidate_label_html("plain label") == "plain label"


def test_compare_legend_documents_delta_and_drops_stale_parity_explainer() -> None:
    """The single compare-model legend documents the Δ divergence count and no longer
    describes the removed per-parser "names output that differs" markers; and no panel
    carries the stale parity_explainer_html field."""
    legend = g._common_legend_html()
    assert "Δ" in legend, "legend should document the Δ divergence count"
    assert "names output that differs" not in legend, "stale per-parser marker text gone"
    hrefs = {
        k: "#"
        for k in (
            "reasoning_fixtures", "reasoning_cases", "reasoning_src", "toolcalling_src",
            "streaming_harmony_src", "streaming_src", "toolcalling_streaming_cases",
            "toolcalling_cases", "pyproject_stub",
        )
    }
    # _apply_common_legend gives every panel the same rich legend.
    panels = [{"id": "tab-a"}, {"id": "tab-b"}, {"id": "tab-toolcalling-batch"}]
    g._apply_common_legend(panels, hrefs)
    assert len({p["legend_html"] for p in panels}) == 1, "all tabs share one legend"
    # Reasoning panels no longer carry the stale per-parser parity-explainer.
    assert not any(
        "parity_explainer_html" in p for p in g._combined_reasoning_panels(hrefs)
    )


def test_compare_bar_renders_when_candidate_lacks_label_html() -> None:
    """The compare bar is shared with the v1 parity page, whose candidates carry no
    label_html. Under StrictUndefined the template must guard the optional field and
    fall back to the plain label instead of raising (regression: PR #105)."""
    bar = (UTILS / "tests" / "parity" / "_compare_bar.html.j2").read_text(encoding="utf-8")
    env = g.Environment(undefined=g.StrictUndefined, autoescape=True)
    html = env.from_string(bar).render(
        panel={
            "id": "p",
            "candidates": [{"key": "dynamo_x", "label": "X (batch)", "default_bucket": "A"}],
        }
    )
    assert "cmprow-label" in html and "X (batch)" in html


def test_transpose_feature_is_wired() -> None:
    """DIS-2280 Transpose: the toggle, the JS mirror builder, and the CSS all ship
    in the assets, and the JS integrates with #98's compare engine (applyCtl) rather
    than the removed per-parser status model."""
    tmpl = (SRC / "conformance_table.html.j2").read_text(encoding="utf-8")
    js = (SRC / "assets" / "conformance.js").read_text(encoding="utf-8")
    css = (SRC / "assets" / "conformance.css").read_text(encoding="utf-8")
    # toolbar checkbox + case-axis data attrs the mirror's corner label reads
    assert "data-transpose-toggle" in tmpl
    assert "data-case-prefix" in tmpl and "data-mode" in tmpl
    # JS builder + integration with the compare engine
    assert "buildTransposed" in js and "data-transpose-table" in js
    assert "if (panelCtl(panel)) { applyCtl(panel); }" in js  # recolor the mirror
    assert "!cell.closest('[data-transpose-table]')" in js     # don't double-count
    # CSS shows the mirror in transpose mode and hides the original
    assert "body.transpose-mode" in css and ".transpose-table" in css
    assert "sideways-lr" in css  # rotated bottom-up model headers
