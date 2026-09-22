# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import json
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import extract_fixtures  # noqa: E402
import package_fixtures  # noqa: E402
import stream_history  # noqa: E402
import unified_history  # noqa: E402

REPO = SRC.parents[2]
RELATIVE = "toolcalling/fixtures-stream-v2/families/glm47/dynamo_v2-0.6.1.yaml"
CAPTURE = REPO / "conformance/fixtures" / RELATIVE


def test_checked_in_checkpoint_roundtrips_through_publisher_and_reader(tmp_path, monkeypatch):
    shard = next(
        shard for shard in json.loads((REPO / "conformance/fixtures-manifest.json").read_text())["shards"]
        if shard["path"] == RELATIVE
    )
    source = extract_fixtures.shard_file(shard)
    stage = tmp_path / "stage"
    extract_fixtures.materialize_shard(shard, source, stage)
    family = stage / stream_history.TREE / "dynamo_v2-0.6.1/glm47"
    records = {path.name: unified_history.load_yaml(path) for path in family.glob("*.yaml")}
    assert stream_history.checkpoint(records, "dynamo_v2-0.6.1", "glm47") == unified_history.load_yaml(CAPTURE)

    monkeypatch.setattr(package_fixtures, "PER_SUBDIR_TREES", [str(stream_history.TREE)])
    monkeypatch.setattr(package_fixtures, "WHOLE_TREE_SHARDS", [])
    monkeypatch.setattr(package_fixtures, "_extracted_snapshot_dir", lambda: None)
    blobs = tmp_path / "blobs"
    built = package_fixtures.build_shards(stage, blobs)
    assert built == [shard]
    assert (blobs / RELATIVE).read_bytes() == CAPTURE.read_bytes()
    assert not list(blobs.rglob("*.tar.gz"))
    second = tmp_path / "second"
    extract_fixtures.materialize_shard(built[0], blobs / RELATIVE, second)
    assert {
        str(path.relative_to(stage)): unified_history.load_yaml(path)
        for path in stage.rglob("*.yaml")
    } == {
        str(path.relative_to(second)): unified_history.load_yaml(path)
        for path in second.rglob("*.yaml")
    }


def test_unchanged_cases_carry_forward_without_another_checkpoint(tmp_path):
    prior = tmp_path / "prior"
    stream_history.materialize(CAPTURE, RELATIVE, prior)
    records = stream_history.documents(unified_history.load_yaml(CAPTURE), "dynamo_v2-0.6.2", "glm47")
    current = tmp_path / "loose/dynamo_v2-0.6.2/glm47"
    current.mkdir(parents=True)
    for name, document in records.items():
        (current / name).write_text(unified_history.dump_yaml(document))
    assert stream_history.package(current.parent, tmp_path / "blobs", prior_root=prior / stream_history.TREE) == []

    name = "TOOLCALLING.streamv2.1.yaml"
    case_id = next(iter(records[name]["cases"]))
    changed_case = copy.deepcopy(records[name]["cases"][case_id])
    changed_case["chunks"][0]["normal_text"] = "changed observation"
    records[name]["cases"][case_id] = changed_case
    (current / name).write_text(unified_history.dump_yaml(records[name]))
    paths = stream_history.package(current.parent, tmp_path / "blobs", prior_root=prior / stream_history.TREE)
    assert len(paths) == 1
    assert unified_history.load_yaml(paths[0]) == {"fixtures": {name: {"cases": {case_id: changed_case}}}}


@pytest.mark.parametrize("label", ["dynamo_v2-0.6.1+source.abc", "dynamo_v2-0.6.1.patch2", "dynamo_v2-current"])
def test_rejects_legacy_capture_names(label):
    with pytest.raises(ValueError, match="plain semantic version"):
        stream_history.capture_identity(label)


@pytest.mark.parametrize("relative", [
    "toolcalling/fixtures-stream-v2/families/../dynamo_v2-0.6.1.yaml",
    "/toolcalling/fixtures-stream-v2/families/glm47/dynamo_v2-0.6.1.yaml",
    "toolcalling/fixtures-stream-v2/families/glm47/dynamo_v2-0.6.1+source.abc.yaml",
])
def test_rejects_invalid_paths_before_writing(tmp_path, relative):
    with pytest.raises(ValueError):
        stream_history.materialize(CAPTURE, relative, tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("mutation", ["traversal", "metadata", "empty", "duplicate"])
def test_rejects_invalid_checkpoint_before_writing(tmp_path, mutation):
    value = unified_history.load_yaml(CAPTURE)
    name = next(iter(value["fixtures"]))
    if mutation == "traversal":
        value["fixtures"]["../../escape.yaml"] = value["fixtures"].pop(name)
    elif mutation == "metadata":
        value["fixtures"][name]["family"] = "glm47"
    elif mutation == "empty":
        value["fixtures"] = {}
    source = tmp_path / "capture.yaml"
    body = unified_history.dump_yaml(value)
    if mutation == "duplicate":
        body += "fixtures: {}\n"
    source.write_text(body)
    with pytest.raises(ValueError):
        stream_history.materialize(source, RELATIVE, tmp_path / "out")
    assert not (tmp_path / "out").exists()
