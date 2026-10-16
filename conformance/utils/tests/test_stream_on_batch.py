# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the conformance-table marker semantics.

Contract pinned here:

  * DEFAULT view (Conformance toggle OFF), every tab: per-engine status only —
    empty when clean, `↯` when the engine leaks tool-call markup, plus
    `…`/`n/a`/`·`/`!`/`—`. NEVER `=` or a divergence letter.
  * CONFORMANCE view (toggle ON):
      - batch tab: cross-engine agreement, expressed as structured
        `comparison_facts()` (`agrees` vs the Dynamo v1 batch baseline); vLLM Rust
        has no batch parser.
      - stream tabs (batch-on-stream + TC stream v2): two dimensions per cell —
        COLOR (`data-status`) is each engine's STREAM parse vs its OWN BATCH parse
        (green consistent, red "problem" on divergence); AGREEMENT concatenates an
        own stream-vs-batch token (when the engine's stream differs from its own
        batch) and one token per diverging peer, `=` when all agree.

The stream-tab tests below still assert on `_stream_xeng_marker`'s exact compact
strings (e.g. `D_rsV_psS_rs`). Those strings are the INTERNAL representation the
stats aggregator buckets on — never shown to users — so they are pinned here as
implementation detail, not as user-facing terminology.

These are marker/status SEMANTICS tests: they assert on the comparison functions in
`markers.py` (via the generator's re-exports) and on the structured model the JS view
consumes — not on any server-rendered HTML (the Python HTML emitters were retired in
DIS-2434; the page is one JSON model + a JS view). Model-shape guards live in
`test_model.py`.
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
import pytest

UTILS = Path(__file__).resolve().parents[1]
SRC = UTILS / "src"  # internal modules moved under conformance/utils/src/
REPO = UTILS.parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _fixtures_cache_root() -> Path:
    """Manifest-pinned fixture snapshot dir (honors CONFORMANCE_FIXTURES_ROOT,
    which `_common.sh` exports). Resolved via check_family_coverage's shared
    resolver — never the shared `<cache>/toolcalling` symlink, which sibling
    checkouts race to repoint mid-run (conformance/README "Invariants")."""
    return cfc.resolve_fixtures_root()


def _resolve_conformance_ref(ref: str) -> Path:
    """Map a `conformance/...` doc path to disk: fixture trees resolve against the HF
    cache (they're not in the repo), everything else against the repo."""
    rel = ref[len("conformance/"):]
    if rel.startswith(("toolcalling/fixtures", "reasoning/fixtures")):
        return _fixtures_cache_root() / rel
    return REPO / ref

import build_stream_fixtures as b  # noqa: E402
import capture_vllm_rust as r  # noqa: E402
import check_family_coverage as cfc  # noqa: E402
import impls  # noqa: E402
import validate_fixtures as vf  # noqa: E402

_TODO_MSG = "Dynamo parser v2 stream parser not yet implemented for this family"
# Identity comes from impls.py via the generator's re-export (audit B1).


def test_readme_documents_vllm_rust_capture_flow() -> None:
    text = (UTILS / "README.md").read_text(encoding="utf-8")
    for required in (
        "vLLM Python stream parser",
        "vLLM Python batch parser",
        "vLLM Rust stream parser",
        "vLLM Rust has no separate batch parser",
        "VLLM_RUST_SOURCE",
        "capture_vllm_rust.py",
        "captured_with.vllm_rust",
        "expected.vllm_rust",
        "viewer_catalog.py",
        "The verification-only path reads the extracted fixtures and reports mismatches",
        "it does not run vLLM Rust",
        "conformance/utils/check.sh dynamo stream",
        "vLLM Python vs Rust is a fixture comparison",
        "capture.sh",
        "Parser Implementations",
        "Dynamo v1",
        "batch only",
        "Dynamo v2 Rust",
        "directory or case ID identifies the convention",
        "DYN_ENABLE_EXPERIMENTAL_PARSERS_V2",
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
    (source_root / "rust/src/parser").mkdir(parents=True)
    (source_root / "rust/src/parser/Cargo.toml").write_text(
        "[package]\nname = 'vllm-parser'\n",
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
    (source_root / "rust/src/parser").mkdir(parents=True)
    (source_root / "rust/src/parser/Cargo.toml").write_text(
        "[package]\nname = 'vllm-parser'\n",
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


# test_dsv4_v2_parser_cell_links_dsml_parser moved to test_model.py as a model/view
# assertion (test_v2_deepseek_v4_streamv1_parser_links_dsml) — it now checks the parser
# cell in the built model rather than calling the _parser_cell_html emitter directly.


# --------------------------------------------------------------------------- #
# DEFAULT marker is leak-only on every tab (the reported bug)
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# stream tabs: 2-D cell — COLOR = stream-vs-own-batch, MARKER = cross-engine outputs
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# cross-engine conformance (batch tab): the structured comparison_facts / cmp_model
# replacement for the retired parity-marker shorthand.
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# _build_stream_on_batch_cases: overlay + batch reference
# --------------------------------------------------------------------------- #


_DOC_FILES = (
    UTILS / "README.md",
    REPO / "parsers" / "v2" / "README.md",
    REPO / "conformance" / "README.md",
    REPO / "conformance" / "toolcalling" / "fixtures-stream-v1" / "README.md",
)
_STALE_COMMAND_NAMES = (
    "capture_v2.sh", "check_v2.sh",
    "test_stream_on_batch_v2.py", "record_v2.sh", "capture_all_families.sh",
)


def test_readme_fixture_paths_exist() -> None:
    """D1: every concrete conformance/*.yaml path in the doc set resolves (A2 regression).

    Fixture YAMLs are extracted from the in-repo LFS shard store, so fixture paths
    resolve against the extraction cache; skip if the cache isn't populated yet."""
    if not (_fixtures_cache_root() / "toolcalling").is_dir():
        pytest.skip("fixtures not extracted (run extract_fixtures.py)")
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
    for target in ("conformance_toolcalling", "conformance_toolcalling_stream", "conformance_toolcalling_batch_via_stream"):
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


def test_v2_overlays_are_canonical_only() -> None:
    """D3: the v2 overlays carry no legacy impl keys and stamp captured_with — locks
    the canonical-key migration (the renderer's legacy aliases exist only for the
    v1 corpus, so legacy keys here are silent drift)."""
    assert vf.validate_overlays() == []


def test_every_stream_family_has_registry_row_and_fixtures() -> None:
    """D6: each fixtures-stream-v1/inputs/<family> has a parser_families.yaml row, and
    each Dynamo-v2 family in the registry has at least one stream input fixture.

    The stream corpus is versioned like the batch corpus (no unversioned anchor):
    families live under `inputs/`; the sibling `<impl>-<version>/` dirs are per-impl
    expected, not families (resolve_stream_fixtures.py folds them into the inputs)."""
    registry = yaml.safe_load((SRC / "parser_families.yaml").read_text())["families"]
    # Fixtures are extracted from the in-repo LFS store; resolve against the cache.
    inputs_root = _fixtures_cache_root() / "toolcalling" / "fixtures-stream-v1" / "inputs"
    if not inputs_root.is_dir():
        pytest.skip("fixtures not extracted (run extract_fixtures.py)")
    for fam_dir in sorted(p for p in inputs_root.iterdir() if p.is_dir()):
        assert fam_dir.name in registry, f"family {fam_dir.name} has no parser_families.yaml row"
    for fam, spec in registry.items():
        if spec.get("dynamo_v2"):
            assert list((inputs_root / fam).glob("*.yaml")), f"dynamo_v2 family {fam} has no stream fixtures"


def test_impl_spec_is_single_identity_source() -> None:
    """D5: ImplSpec is the one identity table; the generator's derived dicts match
    it, every spec is complete, markers/displays are unique, and vLLM Rust has no
    batch parser option. Catches the marker/legend/alias drift class."""
    assert tuple(s.key for s in impls.IMPL_SPECS) == impls.IMPL_KEYS
    for s in impls.IMPL_SPECS:
        assert s.display and s.marker_letter and s.marker_lang and s.engine and s.language
        assert impls.IMPL_DISPLAY[s.key] == s.display
        assert impls.ENGINE_LETTER[s.key] == s.marker_letter
        assert impls.IMPL_LANG_MARKER[s.key] == s.marker_lang
        assert s.legacy_key is None or impls.LEGACY_IMPL_ALIASES[s.legacy_key] == s.key
    # Marker letters are unique per MODE: dynamo_v1 (batch) and dynamo_v2
    # (stream) intentionally share "D" but never appear in the same tab.
    for mode in ("batch", "stream"):
        letters = [s.marker_letter for s in impls.IMPL_SPECS if mode in s.modes]
        assert len(set(letters)) == len(letters), mode
    assert len({s.display for s in impls.IMPL_SPECS}) == len(impls.IMPL_SPECS)
    # vLLM Rust is stream-only: no batch parser option exists anywhere.
    assert "vllm_rust" not in impls.BATCH_IMPL_KEYS
    assert "vllm_rust" in impls.STREAM_IMPL_KEYS
