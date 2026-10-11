# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import copy
import sys
from pathlib import Path

import pytest
import yaml

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import fixtures
import generate_conformance_table as table
from fixture_corpus import load_corpus
from impls import IMPL_KEYS
from resolve_stream_fixtures import resolve_docs


@pytest.fixture(autouse=True)
def reader_paths(monkeypatch):
    monkeypatch.setattr(fixtures, "REPO_ROOT", SRC.parents[2] / "conformance")
    monkeypatch.setattr(fixtures, "FIXTURES", SRC / "toolcalling/fixtures")
    monkeypatch.setattr(fixtures, "_RESOLVE_SRC_DIR", SRC)


def _project_selection(docs, impl):
    """Project caller-owned documents without collapsing empty selected entries."""
    def project(owner, field):
        if field not in owner or not isinstance(owner[field], dict):
            return
        mapping = fixtures._normalize_impl_mapping(owner[field])
        if impl not in mapping:
            del owner[field]
            return
        value = (fixtures._impl_get(owner[field], impl)
                 if field == "normal_text" else mapping[impl])
        owner[field] = {impl: value}

    for doc in docs.values():
        project(doc, "captured_with")
        for case in (doc.get("cases") or {}).values():
            for field in ("unavailable", "exception"):
                project(case, field)
            for chunk in case.get("chunks") or []:
                if isinstance(chunk, dict):
                    for field in ("expected", "normal_text"):
                        project(chunk, field)
    return docs


def _assert_selection(root, corpus, impl, version, *, full_assembly=True):
    selected = [f"{impl}-{version}"]
    full_docs, _ = resolve_docs(root, selected, corpus=corpus)
    filtered_docs, _ = resolve_docs(root, selected, corpus=corpus, implementations={impl})
    if not full_assembly:
        assert list(full_docs) == list(filtered_docs)
        for key in full_docs:
            assert list(full_docs[key].get("cases", {})) == list(filtered_docs[key].get("cases", {}))
            for cid, case in full_docs[key].get("cases", {}).items():
                peer = filtered_docs[key]["cases"][cid]
                for left, right in zip(case.get("chunks") or [], peer.get("chunks") or [], strict=True):
                    # Recording uses first-key normalization, while assembly uses
                    # canonical-key priority. Prove both observations independently.
                    if isinstance(left, dict) and isinstance(right, dict):
                        left_text = fixtures._normalize_impl_mapping(left.get("normal_text") or {})
                        right_text = fixtures._normalize_impl_mapping(right.get("normal_text") or {})
                        assert {k: v for k, v in left_text.items() if k == impl} == {
                            k: v for k, v in right_text.items() if k == impl
                        }
        assert _project_selection(full_docs, impl) == _project_selection(filtered_docs, impl)
        # Identical complete reader inputs need only one assembly. Synthetic cases
        # below still compare the unfiltered and filtered assembly paths directly.
        result, _ = fixtures.load_all_cases("streamv1", docs=filtered_docs, implementations={impl})
        assert result
        return result
    full, labels = fixtures.load_all_cases(
        "streamv1", docs=full_docs, implementations=None if full_assembly else {impl}
    )
    filtered, filtered_labels = fixtures.load_all_cases(
        "streamv1", docs=filtered_docs, implementations={impl}
    )
    assert list(filtered) == list(full)
    assert filtered_labels == labels
    for key, case in full.items():
        got = filtered[key]
        assert got["expected"] == {impl: case["expected"][impl]}
        assert table._overview_status(got, impl) == table._overview_status(case, impl)
        for field in ("description", "__case_id", "__synthetic_from_case_id", "__fixture_path"):
            assert got.get(field) == case.get(field)
        for actual, expected in zip(got.get("chunks", []), case.get("chunks", []), strict=True):
            for field in ("expected", "normal_text"):
                assert actual.get(field, {}).get(impl) == expected.get(field, {}).get(impl)
    return filtered


def test_every_retained_stream_selection_matches_full_resolution(monkeypatch):
    """Compare selected results after full peer overlay resolution at every version."""
    monkeypatch.setattr(fixtures, "_CAPTURED_WITH_BY_MODE", {})
    root = table._STREAM_SRC
    corpus = load_corpus(root)
    before = copy.deepcopy(dict(corpus))
    for impl, versions in corpus.version_dirs.items():
        for _key, version, _directory in versions:
            _assert_selection(root, corpus, impl, version, full_assembly=False)
    assert dict(corpus) == before


def test_sparse_patches_states_and_short_replacements(tmp_path, monkeypatch):
    monkeypatch.setattr(fixtures, "_CAPTURED_WITH_BY_MODE", {})
    cases = {
        f"TOOLCALLING.streamv1.{sub}": {"chunks": [{"delta_text": "a"}, {"delta_text": "b"}]}
        for sub in ("1", "2", "3", "30", "30.a")
    }
    def write(top, records):
        path = tmp_path / top / "qwen3" / "TOOLCALLING.streamv1.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump({"family": "qwen3", "mode": "streamv1", "cases": records}))
    write("inputs", cases)
    write("dynamo_v2-1.0.0", {
        key: {"chunks": [{"expected": [{"index": 0, "name": "echo", "arguments": '{"x":'}]},
                         {"expected": [{"index": 0, "arguments": '1}'}]}]}
        for key in cases if key != "TOOLCALLING.streamv1.3"
    })
    write("dynamo_v2-1.0.0.patch1", {"TOOLCALLING.streamv1.2": {"exception": "broken"}})
    write("dynamo_v2-2.0.0", {
        "TOOLCALLING.streamv1.1": {"chunks": [{"expected": []}]},
        "TOOLCALLING.streamv1.2": {"unavailable": "unsupported"},
    })
    write("dynamo_v2-3.0.0", {
        "TOOLCALLING.streamv1.2": {"chunks": [{"expected": [], "normal_text": "recovered"}]},
    })
    write("vllm_python-1.0.0", {"TOOLCALLING.streamv1.1": {"chunks": [{"expected": []}]}})
    corpus = load_corpus(tmp_path)
    before = copy.deepcopy(dict(corpus))
    for version in ("1.0.0", "1.0.0.patch1", "2.0.0", "3.0.0"):
        result = _assert_selection(tmp_path, corpus, "dynamo_v2", version)
        assert ("qwen3", "30") not in result
        assert any("__synthetic_from_case_id" in case for case in result.values())
    empty = result[("qwen3", "1")]["expected"]["dynamo_v2"]
    missing = result[("qwen3", "3")]["expected"]["dynamo_v2"]
    assert empty == {"calls": [], "normal_text": ""}
    assert "unavailable" in missing
    assert result[("qwen3", "2")]["expected"]["dynamo_v2"]["normal_text"] == "recovered"
    result[("qwen3", "1")]["chunks"][0]["expected"]["dynamo_v2"].append({"index": 9})
    assert dict(corpus) == before


def test_assembly_aliases_fragments_and_canonical_text_priority():
    case = {"chunks": [
        {"expected": {"vllm": [{"index": 1, "name": "ec", "arguments": '{"x":'}]},
         "normal_text": {"vllm": "old", "vllm_python": "new"}},
        {"expected": {"vllm_python": [{"index": 1, "name": "ho", "arguments": '2}'}]},
         "normal_text": {"vllm": "!"}},
    ]}
    before = copy.deepcopy(case)
    expected = fixtures._derive_stream_expected(case)
    assert expected["vllm_python"]["calls"] == [{"name": "echo", "arguments": {"x": 2}}]
    assert expected["vllm_python"]["normal_text"] == "new!"
    for impl in IMPL_KEYS:
        assert fixtures._derive_stream_expected(case, implementations={impl}) == {impl: expected[impl]}
    assert case == before


@pytest.mark.parametrize("present", [False, True])
@pytest.mark.parametrize("mode", ["batch", "streamv1"])
def test_history_restores_captured_globals_even_when_absent(monkeypatch, present, mode):
    saved = {"old": "value"} if present else None
    state = {mode: saved} if present else {}
    monkeypatch.setattr(table, "_CAPTURED_WITH_BY_MODE", state)
    monkeypatch.setattr(fixtures, "_CAPTURED_WITH_BY_MODE", state)
    monkeypatch.setattr(table, "_stream_impl_versions", lambda: {"dynamo_v2": ["0.1.11"]})
    monkeypatch.setattr(table, "_batch_impl_versions", lambda: {"dynamo_v1": ["3.0.0"]})
    if mode == "streamv1":
        assert table._stream_version_status_map()
    else:
        assert table._batch_version_status_map()
    assert state == ({mode: saved} if present else {})
