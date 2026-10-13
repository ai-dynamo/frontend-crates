# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Storage selection retains exact records without mutating the shared corpus."""
import copy
import sys
from pathlib import Path

import yaml

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from fixture_corpus import load_corpus
from fixture_snapshot import fixture_snapshot_root
from markers import _impl_get, _normalize_impl_mapping
from resolve_stream_fixtures import resolve_docs


def _project_selection(docs, impl):
    def project(owner, field):
        if field not in owner or not isinstance(owner[field], dict):
            return
        mapping = _normalize_impl_mapping(owner[field])
        if impl not in mapping:
            del owner[field]
            return
        value = _impl_get(owner[field], impl) if field == "normal_text" else mapping[impl]
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


def _assert_selection(root, corpus, impl, version):
    selected = [f"{impl}-{version}"]
    full, _ = resolve_docs(root, selected, corpus=corpus)
    filtered, _ = resolve_docs(root, selected, corpus=corpus, implementations={impl})
    assert list(full) == list(filtered)
    for key in full:
        assert list(full[key].get("cases", {})) == list(filtered[key].get("cases", {}))
    assert _project_selection(full, impl) == _project_selection(filtered, impl)
    return filtered


def test_every_retained_stream_selection_matches_full_resolution():
    root = fixture_snapshot_root() / "toolcalling/fixtures-stream-v1"
    corpus = load_corpus(root)
    before = copy.deepcopy(dict(corpus))
    for impl, versions in corpus.version_dirs.items():
        for _key, version, _directory in versions:
            _assert_selection(root, corpus, impl, version)
    assert dict(corpus) == before


def test_sparse_patches_states_and_short_replacements(tmp_path):
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
    document = next(iter(result.values()))
    records = document["cases"]
    assert "TOOLCALLING.streamv1.30" in records
    assert "TOOLCALLING.streamv1.30.a" in records
    assert records["TOOLCALLING.streamv1.1"]["chunks"][0]["expected"]["dynamo_v2"] == []
    assert "dynamo_v2" not in records["TOOLCALLING.streamv1.1"]["chunks"][1].get("expected", {})
    assert "expected" not in records["TOOLCALLING.streamv1.3"]["chunks"][0]
    assert records["TOOLCALLING.streamv1.2"]["chunks"][0]["normal_text"]["dynamo_v2"] == "recovered"
    records["TOOLCALLING.streamv1.1"]["chunks"][0]["expected"]["dynamo_v2"].append({"index": 9})
    assert dict(corpus) == before
