# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from migration_compare import differences  # noqa: E402
from migration_identity import cache_inventory, file_hash, verify_seal, verify_tree  # noqa: E402
from migration_report import keyed, report_snapshot, source_links, unique_object, verified_link  # noqa: E402
from validate_conformance_status import build_status  # noqa: E402
from migration_snapshot import raw_history  # noqa: E402
from fixture_snapshot import fixture_snapshot_root  # noqa: E402
import generate_conformance_table as table  # noqa: E402
import yaml  # noqa: E402


def test_link_relocation_never_normalizes_parser_text_or_arguments(tmp_path):
    html = tmp_path / "report.html"
    relative = 'href="asset.html"'
    before = {"cell": {"tooltip": {
        "input": {"text": relative},
        "candidates": [{"block": {"calls": [{"arguments": {"content": relative, "html": relative}}]}}],
    }}}
    after = json.loads(json.dumps(before).replace('asset.html', str(tmp_path / 'asset.html')))
    assert len(list(differences(before, after))) == 3
    assert list(differences(source_links(before, html, tmp_path), source_links(after, html, tmp_path))) == list(differences(before, after))
    for record in ({"page": {"legend_html": relative}}, {"tab": {"toolbar_desc_html": relative}},
                   {"row": {"parser": {"html": relative}}},
                   {"cell": {"tooltip": {"dynamo_notes": [["note", relative]]}}}):
        relocated = json.loads(json.dumps(record).replace('asset.html', str(tmp_path / 'asset.html')))
        assert source_links(record, html, tmp_path) == source_links(relocated, html, tmp_path)


@pytest.mark.parametrize("suffix", ["?case=1#section", "#section"])
def test_presentation_link_suffix_is_semantic(tmp_path, suffix):
    value = {"page": {"legend_html": f'href="asset.html{suffix}"'}}
    assert source_links(value, tmp_path / "report.html", tmp_path) == value


@pytest.mark.parametrize("encoded,delimiter", [("%3F", "?"), ("%23", "#")])
def test_encoded_filename_delimiter_is_not_a_url_suffix(tmp_path, encoded, delimiter):
    before = {"page": {"legend_html": f'href="asset{encoded}case=1"'}}
    after = {"page": {"legend_html": f'href="asset{delimiter}case=1"'}}
    assert source_links(before, tmp_path / "report.html", tmp_path) != source_links(after, tmp_path / "report.html", tmp_path)


@pytest.mark.parametrize("prefix", ["file://hostA", "file://hostB", "custom:", "//hostA"])
def test_nonlocal_presentation_links_are_not_relocated(tmp_path, prefix):
    value = {"page": {"legend_html": f'href="{prefix}{tmp_path}/asset.html"'}}
    assert source_links(value, tmp_path / "report.html", tmp_path) == value


@pytest.mark.parametrize("fixture", [False, True])
def test_verified_report_links_preserve_all_url_components(tmp_path, fixture):
    (tmp_path / "asset.yaml").write_text("case: []\n")
    html = tmp_path / "report.html"
    plain = verified_link("asset.yaml", html, tmp_path, fixture=fixture)
    for suffix, field in [("?case=1", "query"), ("#case=1", "fragment")]:
        linked = verified_link(f"asset.yaml{suffix}", html, tmp_path, fixture=fixture)
        assert linked[field] == "case=1"
        assert linked != plain
        assert linked == verified_link(f"file://{tmp_path}/asset.yaml{suffix}", html, tmp_path, fixture=fixture)
    for prefix in ("file://hostA", "file://hostB", "custom:", "//hostA"):
        with pytest.raises(ValueError, match="expected local report link"):
            verified_link(f"{prefix}{tmp_path}/asset.yaml", html, tmp_path, fixture=fixture)
    for encoded, delimiter in [("%3F", "?"), ("%23", "#")]:
        (tmp_path / f"asset.yaml{delimiter}case=1").write_text("case: []\n")
        assert verified_link(f"asset.yaml{encoded}case=1", html, tmp_path, fixture=fixture) != verified_link(
            f"asset.yaml{delimiter}case=1", html, tmp_path, fixture=fixture)
    with pytest.raises(ValueError, match="broken report link"):
        verified_link("asset.yaml;case=1", html, tmp_path, fixture=fixture)
    (tmp_path / "asset.yaml;case=1").write_text("case: [1]\n")
    semicolon = verified_link("asset.yaml;case=1", html, tmp_path, fixture=fixture)
    assert semicolon["path"] == "asset.yaml;case=1"
    assert semicolon["sha256"] != plain["sha256"]
    assert semicolon == verified_link("asset.yaml%3Bcase=1", html, tmp_path, fixture=fixture)
    assert semicolon == verified_link(f"file://{tmp_path}/asset.yaml;case=1", html, tmp_path, fixture=fixture)
    literal = {"page": {"legend_html": 'href="asset.yaml;case=1"'}}
    encoded = {"page": {"legend_html": 'href="asset.yaml%3Bcase=1"'}}
    assert source_links(literal, html, tmp_path) == source_links(encoded, html, tmp_path)


@pytest.mark.parametrize("corpus,impl,older,newer", [
    (corpus, *table.capture_policy.split_sel(source), table.capture_policy.split_sel(rule["equivalent_to"])[1])
    for corpus, rules in table.capture_policy.load_policy()["corpora"].items()
    for source, rule in rules.get("selectors", {}).items() if rule.get("equivalent_to")
])
def test_hidden_selector_preserves_identical_complete_capture(corpus, impl, older, newer):
    root = fixture_snapshot_root() / f"toolcalling/fixtures-{corpus}-v1"
    names = sorted(path.name for path in root.iterdir() if path.is_dir() and path.name != "inputs")
    records = {}
    raw_history(root, corpus, names, records)
    snapshots = []
    for version in (older, newer):
        prefix = f"{corpus}/raw-history/{impl}-{version}/"
        snapshots.append({key.removeprefix(prefix): value for key, value in records.items()
                          if key.startswith(prefix)})
    assert snapshots[0], "a missing capture cannot establish equivalence"
    assert snapshots[0] == snapshots[1]


def test_equivalent_selector_requires_present_replacement_and_preserves_data(monkeypatch):
    policy = {"corpora": {"stream": {"selectors": {
        "vllm_python-1.0.0": {"visible": False, "equivalent_to": "vllm_python-1.1.0"}}}}}
    monkeypatch.setattr(table, "_capture_policy", lambda: policy)
    old = {"key": "vllm_python-1-0-0", "label": "vLLM Python 1.0.0 (stream)"}
    new = {"key": "vllm_python-1-1-0", "label": "vLLM Python 1.1.0 (stream)"}
    with pytest.raises(ValueError, match="missing equivalent report candidate"):
        table._candidate_model([old], corpus="stream")
    candidates = table._candidate_model([old, new], corpus="stream")
    assert len(candidates) == 2
    assert candidates[0]["version"] == "1.0.0"
    assert candidates[0]["equivalent_to"] == new["key"]
    assert candidates[0]["visible"] is False


@pytest.mark.parametrize("before,after", [
    ({"case": None}, {}), ({}, {"case": None}),
    ({"events": [1, 2]}, {"events": [2, 1]}),
    ({"label": "0.6.1"}, {"label": "0.7.0"}),
    ({"explanation": "captured"}, {"explanation": "inherited"}),
    ({"expected": {"old": []}}, {"expected": {"new": []}}),
])
def test_semantic_mutations_fail_command(tmp_path, before, after):
    paths = [tmp_path / name for name in ("baseline.json", "candidate.json", "result.json")]
    for path, value in zip(paths, [before, after]):
        path.write_text(json.dumps({"inventory": [], "records": value}))
    result = subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / "migration_compare.py"),
                             "--baseline", str(paths[0]), "--candidate", str(paths[1]), "--output", str(paths[2])],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 1
    assert json.loads(paths[2].read_text())["unexpected"]


def test_duplicate_identities_are_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        json.loads('{"case": null, "case": 1}', object_pairs_hook=unique_object)
    with pytest.raises(ValueError, match="duplicate"):
        keyed([{"key": "dynamo"}, {"key": "dynamo"}], "key")


def test_html_json_disagreement(tmp_path):
    page = {"schema": 1, "meta": {}, "tabs": []}
    html, status = tmp_path / "report.html", tmp_path / "report.json"
    html.write_text('<script type="application/json" id="conformance-model">' + json.dumps(page) + '</script>')
    value = build_status(copy.deepcopy(page), [], [], html)
    value["reports"].append({"invented": "cell"})
    status.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="HTML/JSON disagreement"):
        report_snapshot(html, status, tmp_path, tmp_path)


def test_stale_report_fails_seal(tmp_path, monkeypatch):
    report = tmp_path / "report.html"
    report.write_text("candidate")
    seal = {"source": str(tmp_path), "tree": "tree", "outputs": {str(report): file_hash(report)}}
    monkeypatch.setattr("migration_identity.verify_tree", lambda _source, _tree: 1)
    report.write_text("stale baseline")
    with pytest.raises(ValueError, match="candidate/output identity mismatch"):
        verify_seal(seal)


def test_union_comparison_has_no_changes_for_equal_values():
    assert not list(differences({"empty": [], "null": None}, {"null": None, "empty": []}))


def test_capture_only_cases_and_annotations_cannot_disappear(tmp_path):
    root = tmp_path / "stream"
    capture = root / "dynamo_v2-1.0.0/family/one.yaml"
    capture.parent.mkdir(parents=True)
    document = {"family": "family", "cases": {"A": {"chunks": [], "note": None}, "ORPHAN": {"chunks": []}}}
    capture.write_text(yaml.safe_dump(document))
    before = {}
    raw_history(root, "stream", ["dynamo_v2-1.0.0"], before)
    del document["cases"]["ORPHAN"]
    del document["cases"]["A"]["note"]
    capture.write_text(yaml.safe_dump(document))
    after = {}
    raw_history(root, "stream", ["dynamo_v2-1.0.0"], after)
    assert len(list(differences(before, after))) == 2


def test_source_inventory_and_modes_are_part_of_tree_identity(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    source = tmp_path / "reader.py"
    source.write_text("pass\n")
    subprocess.run(["git", "add", "reader.py"], cwd=tmp_path, check=True)
    tree = subprocess.check_output(["git", "write-tree"], cwd=tmp_path, text=True).strip()
    assert verify_tree(tmp_path, tree) == 1
    source.chmod(0o755)
    with pytest.raises(ValueError, match="mode mismatch"):
        verify_tree(tmp_path, tree)
    source.chmod(0o644)
    (tmp_path / "untracked.py").write_text("pass\n")
    with pytest.raises(ValueError, match="untracked source"):
        verify_tree(tmp_path, tree)


def test_tree_seal_does_not_consume_an_existing_fixture_cache(tmp_path):
    source = tmp_path / "source"
    renderer = source / "conformance/utils/render_table_v2.sh"
    renderer.parent.mkdir(parents=True)
    renderer.write_text('''#!/bin/bash
set -eu
python3 - "$2" <<'PY'
import os
from pathlib import Path
import shutil
import sys
cache = Path(os.environ["XDG_CACHE_HOME"])
fixture = cache / "fixture.yaml"
if not fixture.exists():
    shutil.copyfile("fixture.yaml", fixture)
output = Path(sys.argv[1])
output.write_text(fixture.read_text())
output.with_suffix(".json").write_text("{}")
PY
''')
    (source / "fixture.yaml").write_text("input: committed\n")
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    subprocess.run(["git", "add", "fixture.yaml", "conformance/utils/render_table_v2.sh"], cwd=source, check=True)
    tree = subprocess.check_output(["git", "write-tree"], cwd=source, text=True).strip()
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "fixture.yaml").write_text("input: uncommitted\n")
    html = tmp_path / "report.html"
    seal_path = tmp_path / "seal.json"
    command = [sys.executable, str(Path(__file__).resolve().parents[1] / "migration_identity.py"),
               "--source", str(source), "--tree", tree, "--cache", str(cache), "--html", str(html),
               "--log", str(tmp_path / "render.log"), "--output", str(seal_path)]
    environment = {"PATH": os.environ["PATH"], "PYTHONDONTWRITEBYTECODE": "1"}
    subprocess.run(command, check=True, capture_output=True, text=True, env=environment)
    assert html.read_text() == "input: committed\n"
    seal = json.loads(seal_path.read_text())
    private_cache = Path(seal["cache"])
    assert private_cache.parent == cache
    assert private_cache.stat().st_mode & 0o777 == 0o755
    assert (private_cache / "fixture.yaml").read_text() == "input: committed\n"
    assert (cache / "fixture.yaml").read_text() == "input: uncommitted\n"
    verify_seal(seal)
    (private_cache / "fixture.yaml").write_text("input: changed after rendering\n")
    with pytest.raises(ValueError, match="fixture input identity mismatch"):
        verify_seal(seal)


@pytest.mark.parametrize("mutation", ["change", "remove", "add", "retarget"])
def test_seal_rejects_fixture_cache_mutations(tmp_path, monkeypatch, mutation):
    cache = tmp_path / "cache"
    cache.mkdir()
    fixture = cache / "fixture.yaml"
    fixture.write_text("input: committed\n")
    link = cache / "current"
    link.symlink_to("fixture.yaml")
    report = tmp_path / "report.html"
    report.write_text("candidate")
    seal = {"source": str(tmp_path), "tree": "tree", "cache": str(cache),
            "outputs": {str(report): file_hash(report)}, "fixture_inputs": cache_inventory(cache)}
    monkeypatch.setattr("migration_identity.verify_tree", lambda _source, _tree: 1)
    verify_seal(seal)
    if mutation == "change":
        fixture.write_text("input: uncommitted\n")
    elif mutation == "remove":
        fixture.unlink()
    elif mutation == "add":
        (cache / "extra.yaml").write_text("input: extra\n")
    else:
        link.unlink()
        link.symlink_to("missing.yaml")
    with pytest.raises(ValueError, match="fixture input identity mismatch"):
        verify_seal(seal)


def test_old_seal_without_fixture_inputs_requires_regeneration(tmp_path, monkeypatch):
    monkeypatch.setattr("migration_identity.verify_tree", lambda _source, _tree: 1)
    with pytest.raises(ValueError, match="no fixture input identity"):
        verify_seal({"source": str(tmp_path), "tree": "tree", "outputs": {}})
