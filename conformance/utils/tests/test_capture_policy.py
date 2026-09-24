# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Policy decisions depend on discovered captures and verified observation equality."""
import copy
from pathlib import Path
import sys

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import capture_policy as policy
import resolve_reasoning_fixtures
import resolve_fixtures
import resolve_stream_fixtures


def rules():
    return {"schema_version": 1, "corpora": {name: {} for name in policy.CORPORA}}


def test_historical_selection_discovers_versions_and_filters_qualified():
    value = rules()
    rule = {"selection": "latest", "include_qualified": False}
    value["corpora"]["stream"]["historical_readers"] = {"dynamo_v2": rule}
    available = ["dynamo_v2-1.0.2", "dynamo_v2-1.0.11", "dynamo_v2-1.0.12+current", "vllm_rust-9.0.0"]
    assert policy.reader_metadata("stream", "dynamo_v2", available, policy=value)["selected"] == "dynamo_v2-1.0.11"
    rule["include_qualified"] = True
    assert policy.reader_metadata("stream", "dynamo_v2", available, policy=value)["selected"] == "dynamo_v2-1.0.12+current"
    missing = policy.reader_metadata("stream", "dynamo_v2", [], policy=value)
    assert missing["selected"] is None and missing["eligible"] == [] and missing["unavailable"]
    rule["selection"] = "dynamo_v2-2.0.0"
    with pytest.raises(ValueError, match="missing or ineligible"):
        policy.reader_metadata("stream", "dynamo_v2", available, policy=value)


def test_alias_requires_complete_equal_observations_and_present_visible_target():
    value = rules()
    value["corpora"]["stream"]["selectors"] = {"dynamo_v2-1.0.0": {"visible": False, "equivalent_to": "dynamo_v2-1.1.0"}}
    available = {"stream": ["dynamo_v2-1.0.0", "dynamo_v2-1.1.0"]}
    states = {"stream": {name: {"case": {"chunks": [1, 2], "reason": "measured"}} for name in available["stream"]}}
    policy.validate_policy(value, available, states)
    for broken in ({}, {"stream": {"dynamo_v2-1.1.0": {"case": 1}}}):
        with pytest.raises(ValueError, match="complete observations"):
            policy.validate_policy(value, available, broken)
    changed = copy.deepcopy(states)
    changed["stream"]["dynamo_v2-1.1.0"]["case"]["reason"] = "different annotation"
    with pytest.raises(ValueError, match="complete observations"):
        policy.validate_policy(value, available, changed)
    with pytest.raises(ValueError, match="missing equivalent target"):
        policy.validate_policy(value, {"stream": ["dynamo_v2-1.0.0"]}, states)


def _capture(root, identity, case):
    path = root / policy.CORPORA["stream"] / identity / "family" / "CASE.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(yaml.safe_dump({"cases": {"A": case}}))


def test_removal_certificate_survives_future_validation_and_rejects_changed_target(tmp_path):
    value = rules()
    value["corpora"]["stream"]["references"] = {"dynamo_v2": {"selection": "dynamo_v2-1.0.0"}}
    before, after = tmp_path / "before", tmp_path / "after"
    for root, versions in [(before, ["1.0.0", "1.1.0"]), (after, ["1.1.0"])]:
        for version in versions:
            _capture(root, "dynamo_v2-" + version, {"chunks": [1, 2]})
    with pytest.raises(ValueError, match="requires replacement"):
        policy.retire_capture(value, "stream", "dynamo_v2-1.0.0")
    retired = policy.retire_capture(value, "stream", "dynamo_v2-1.0.0", replacement="dynamo_v2-1.1.0", before_root=before)
    policy.validate_materialized_policy(retired, after)
    path = after / policy.CORPORA["stream"] / "dynamo_v2-1.1.0/family/CASE.yaml"
    path.write_text(yaml.safe_dump({"cases": {"A": {"chunks": [2, 1]}}}))
    with pytest.raises(ValueError, match="complete observations"):
        policy.validate_materialized_policy(retired, after)


def test_removed_saved_link_requires_explicit_disposition():
    value = rules()
    value["corpora"]["unified"]["saved_links"] = {"old-key": "dynamo_v2-1.0.0"}
    with pytest.raises(ValueError, match="requires replacement"):
        policy.retire_capture(value, "unified", "dynamo_v2-1.0.0")
    retired = policy.retire_capture(value, "unified", "dynamo_v2-1.0.0", unavailable="Measurement retired")
    policy.validate_policy(retired, {})
    assert retired["corpora"]["unified"]["selectors"]["dynamo_v2-1.0.0"]["unavailable"] == "Measurement retired"


def test_explicit_unavailable_reader_excludes_existing_history():
    value = rules()
    value["corpora"]["unified"]["historical_readers"] = {
        "vllm_rust": {"selection": "unavailable", "reason": "No retained reference", "include_qualified": False}}
    result = policy.reader_metadata("unified", "vllm_rust", ["vllm_rust-1.0.0"], policy=value)
    assert result["selected"] is None
    assert result["unavailable"] == "No retained reference"


def test_consecutive_equivalent_removals_preserve_old_saved_links(tmp_path):
    value = rules()
    value["corpora"]["stream"]["saved_links"] = {"old": "dynamo_v2-1.0.0"}
    first, second, third = (tmp_path / name for name in ("first", "second", "third"))
    for root, versions in [(first, ["1.0.0", "1.1.0", "1.2.0"]), (second, ["1.1.0", "1.2.0"]), (third, ["1.2.0"])]:
        for version in versions:
            _capture(root, "dynamo_v2-" + version, {"chunks": [1]})
    retired = policy.retire_capture(value, "stream", "dynamo_v2-1.0.0", replacement="dynamo_v2-1.1.0", before_root=first)
    policy.validate_materialized_policy(retired, second)
    retired = policy.retire_capture(retired, "stream", "dynamo_v2-1.1.0", replacement="dynamo_v2-1.2.0", before_root=second)
    policy.validate_materialized_policy(retired, third)
    assert retired["corpora"]["stream"]["selectors"]["dynamo_v2-1.0.0"]["equivalent_to"] == "dynamo_v2-1.2.0"


def test_reasoning_policy_selects_exact_snapshot_or_unavailable_without_anchor_leak(tmp_path):
    value = rules()
    value["corpora"]["reasoning"]["references"] = {"dynamo_v1": {"selection": "dynamo_v1-1.0.0"}}
    root = tmp_path / "fixtures"
    for name, text in [("inputs", "old-anchor"), ("dynamo_v1-1.0.0", "measured"), ("dynamo_v1-2.0.0", "newer")]:
        file = root / name / "family/CASE.yaml"
        file.parent.mkdir(parents=True)
        file.write_text(yaml.safe_dump({"cases": {"A": {"expected": {"dynamo_v1": {"normal_text": text}}}}}))
    out = tmp_path / "out"
    resolve_reasoning_fixtures.resolve(root, out, [], policy=value)
    assert yaml.safe_load((out / "family/CASE.yaml").read_text())["cases"]["A"]["expected"]["dynamo_v1"]["normal_text"] == "measured"
    value["corpora"]["reasoning"]["references"]["dynamo_v1"] = {"selection": "unavailable", "reason": "retired"}
    resolve_reasoning_fixtures.resolve(root, out, [], policy=value)
    assert yaml.safe_load((out / "family/CASE.yaml").read_text())["cases"]["A"]["expected"]["dynamo_v1"] == {"unavailable": "retired"}


@pytest.mark.parametrize("empty_directory", [False, True])
def test_complete_family_snapshot_does_not_resurrect_absent_cases(tmp_path, empty_directory):
    value = rules()
    value["corpora"]["stream"]["selectors"] = {"dynamo_v2-1.0.0": {"visible": False, "equivalent_to": "dynamo_v2-1.1.0"}}
    for view in (tmp_path, tmp_path / ".reader-views/rust"):
        _capture(view, "dynamo_v2-1.0.0", {"chunks": ["measured"]})
        path = view / policy.CORPORA["stream"] / "dynamo_v2-1.1.0/family/CASE.yaml"
        path.parent.mkdir(parents=True)
        if not empty_directory:
            path.write_text("cases: {}\n")
        sibling = view / policy.CORPORA["stream"] / "dynamo_v2-1.0.0/sibling/CASE.yaml"
        sibling.parent.mkdir()
        sibling.write_text("cases: {B: {chunks: [retained]}}\n")
    marker = tmp_path / ".reader-views/legacy-checkpoints.json"
    marker.write_text('{"schema_version":1,"complete_family_snapshots":true}')
    inventory, evidence = policy.materialized_evidence(tmp_path)
    assert ("family", "CASE.yaml", "A") not in evidence["stream"]["dynamo_v2-1.1.0"]
    assert evidence["stream"]["dynamo_v2-1.1.0"][("sibling", "CASE.yaml", "B")] == {"chunks": ["retained"]}
    with pytest.raises(ValueError, match="complete observations"):
        policy.validate_policy(value, inventory, evidence)


def test_stream_equivalence_also_requires_equal_rust_view(tmp_path):
    value = rules()
    value["corpora"]["stream"]["selectors"] = {"dynamo_v2-1.0.0": {"visible": False, "equivalent_to": "dynamo_v2-1.1.0"}}
    for version in ("1.0.0", "1.1.0"):
        _capture(tmp_path, "dynamo_v2-" + version, {"chunks": ["same"]})
        _capture(tmp_path / ".reader-views/rust", "dynamo_v2-" + version, {"chunks": [version]})
    (tmp_path / ".reader-views/legacy-checkpoints.json").write_text('{"schema_version":1,"complete_family_snapshots":true}')
    with pytest.raises(ValueError, match="complete observations"):
        policy.validate_materialized_policy(value, tmp_path)


def test_snapshot_policy_wins_over_repository_environment(tmp_path, monkeypatch):
    packaged = rules()
    packaged["corpora"]["stream"]["references"] = {"dynamo_v2": {"selection": "unavailable", "reason": "packaged"}}
    (tmp_path / "capture-policy.yaml").write_text(yaml.safe_dump(packaged))
    captures = tmp_path / "toolcalling/fixtures-stream-v1"
    captures.mkdir(parents=True)
    monkeypatch.setenv("CONFORMANCE_CAPTURE_POLICY", str(tmp_path / "nonexistent.yaml"))
    assert policy.policy_for_snapshot(captures) == packaged


@pytest.mark.parametrize("field", ["notes", "mode"])
def test_legacy_document_annotations_are_part_of_equivalence(tmp_path, field):
    value = rules()
    value["corpora"]["batch"]["selectors"] = {"dynamo_v1-1.0.0": {"visible": False, "equivalent_to": "dynamo_v1-1.1.0"}}
    for version in ("1.0.0", "1.1.0"):
        path = tmp_path / policy.CORPORA["batch"] / f"dynamo_v1-{version}/family/CASE.yaml"
        path.parent.mkdir(parents=True)
        path.write_text(yaml.safe_dump({"captured_with": {"dynamo_v1": version}, field: version, "cases": {"A": {"expected": []}}}))
    with pytest.raises(ValueError, match="complete observations"):
        policy.validate_materialized_policy(value, tmp_path)
    for path in (tmp_path / policy.CORPORA["batch"]).glob("*/*/*.yaml"):
        doc = yaml.safe_load(path.read_text())
        doc[field] = "same"
        path.write_text(yaml.safe_dump(doc))
    policy.validate_materialized_policy(value, tmp_path)


def test_multiple_default_report_references_are_rejected():
    value = rules()
    value["corpora"]["unified"]["references"] = {"dynamo_v2": {"selection": "latest"}, "vllm_rust": {"selection": "latest"}}
    with pytest.raises(ValueError, match="only one default reference"):
        policy.validate_policy(value, {})


@pytest.mark.parametrize("corpus", ["batch", "reasoning", "stream"])
@pytest.mark.parametrize("empty_directory", [False, True])
def test_all_legacy_readers_preserve_complete_checkpoint_absence(tmp_path, corpus, empty_directory):
    root = tmp_path / policy.CORPORA[corpus]
    impl = "dynamo_v2" if corpus == "stream" else "dynamo_v1"
    initial = ({"chunks": [{"delta_text": "x", "expected": {"other": []}}]}
               if corpus == "stream" else {"model_text": "x", "expected": {"other": {"normal_text": "keep"}}})
    measured = ({"chunks": [{"expected": [{"index": 0, "name": "old"}]}]}
                if corpus == "stream" else {"expected": {impl: {"normal_text": "old"}}})
    for family in ("family", "sibling"):
        for capture, case in [("inputs", initial), (impl + "-1.0.0", measured)]:
            path = root / capture / family / "CASE.yaml"
            path.parent.mkdir(parents=True)
            path.write_text(yaml.safe_dump({"cases": {"A": case}}))
    absent = root / (impl + "-1.1.0") / "family/CASE.yaml"
    absent.parent.mkdir(parents=True)
    if not empty_directory:
        absent.write_text("cases: {}\n")
    marker = tmp_path / ".reader-views/legacy-checkpoints.json"
    marker.parent.mkdir()
    marker.write_text('{"schema_version":1,"complete_family_snapshots":true}')
    if corpus == "reasoning":
        out = tmp_path / "out"
        resolve_reasoning_fixtures.resolve(root, out, [impl + "-1.1.0"])
        docs = {(p.parent.name, p.name): yaml.safe_load(p.read_text()) for p in out.glob("*/*.yaml")}
    else:
        resolver = resolve_stream_fixtures if corpus == "stream" else resolve_fixtures
        docs = resolver.resolve_docs(root, [impl + "-1.1.0"])[0]
    current = docs[("family", "CASE.yaml")]["cases"]["A"]
    sibling = docs[("sibling", "CASE.yaml")]["cases"]["A"]
    if corpus == "stream":
        assert "No observation" in current["unavailable"][impl]
        assert impl not in current["chunks"][0]["expected"]
        assert current["chunks"][0]["expected"]["other"] == []
        assert sibling["chunks"][0]["expected"][impl][0]["name"] == "old"
    else:
        assert "No observation" in current["expected"][impl]["unavailable"]
        assert current["expected"]["other"]["normal_text"] == "keep"
        assert sibling["expected"][impl]["normal_text"] == "old"


def test_declared_unversioned_reasoning_anchor_retains_measurements(tmp_path):
    value = rules()
    rule = {"selection": "inputs", "reason": "Original producer version was not retained"}
    value["corpora"]["reasoning"] = {"references": {"dynamo_v1": rule}, "historical_readers": {"dynamo_v1": rule}}
    root, out = tmp_path / "reasoning", tmp_path / "out"
    path = root / "inputs/family/CASE.yaml"
    path.parent.mkdir(parents=True)
    path.write_text("cases: {A: {expected: {dynamo_v1: {reasoning_text: original}}}}\n")
    resolve_reasoning_fixtures.resolve(root, out, [], policy=value)
    assert yaml.safe_load((out / "family/CASE.yaml").read_text())["cases"]["A"]["expected"]["dynamo_v1"] == {"reasoning_text": "original"}
    metadata = policy.reader_metadata("reasoning", "dynamo_v1", [], policy=value)
    assert metadata["selected"] is None and metadata["anchor"] == "inputs" and metadata["unavailable"] is None
    policy.validate_policy(value, {})


@pytest.mark.parametrize("corpus", ["batch", "stream"])
def test_complete_snapshot_retains_shared_capability_unavailability(tmp_path, corpus):
    root = tmp_path / policy.CORPORA[corpus]
    impl = "dynamo_v2"
    shared = {"unavailable": {impl: "This family has no implementation"}}
    shared.update({"chunks": [{"delta_text": "x"}]} if corpus == "stream" else {"model_text": "x"})
    path = root / "inputs/family/CASE.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(yaml.safe_dump({"cases": {"A": shared}}))
    (root / (impl + "-1.0.0") / "family").mkdir(parents=True)
    marker = tmp_path / ".reader-views/legacy-checkpoints.json"
    marker.parent.mkdir()
    marker.write_text('{"schema_version":1,"complete_family_snapshots":true}')
    resolver = resolve_stream_fixtures if corpus == "stream" else resolve_fixtures
    current = resolver.resolve_docs(root, [impl + "-1.0.0"])[0][("family", "CASE.yaml")]["cases"]["A"]
    assert current["unavailable"][impl] == "This family has no implementation"


def test_removing_resolved_latest_requires_disposition_and_updates_selection(tmp_path):
    value = rules()
    value["corpora"]["stream"]["references"] = {"dynamo_v2": {"selection": "latest"}}
    for version in ("1.0.0", "1.1.0"):
        _capture(tmp_path, "dynamo_v2-" + version, {"chunks": ["same"]})
    with pytest.raises(ValueError, match="requires replacement"):
        policy.retire_capture(value, "stream", "dynamo_v2-1.1.0", before_root=tmp_path)
    changed = policy.retire_capture(value, "stream", "dynamo_v2-1.1.0", replacement="dynamo_v2-1.0.0", before_root=tmp_path)
    assert changed["corpora"]["stream"]["references"]["dynamo_v2"]["selection"] == "dynamo_v2-1.0.0"


def test_partial_selected_family_requires_candidate_reference_policy(tmp_path):
    value = rules()
    value["corpora"]["stream"]["references"] = {"dynamo_v2": {"selection": "latest"}}
    _capture(tmp_path, "dynamo_v2-1.0.0", {"chunks": ["same"]})
    newer = tmp_path / policy.CORPORA["stream"] / "dynamo_v2-2.0.0/sibling"
    newer.mkdir(parents=True)
    with pytest.raises(ValueError, match="candidate policy"):
        policy.validate_partial_removal_policy(value, "stream", "dynamo_v2-1.0.0", "family", tmp_path)
    value["corpora"]["stream"]["references"]["dynamo_v2"] = {"selection": "unavailable", "reason": "Reference retired"}
    policy.validate_partial_removal_policy(value, "stream", "dynamo_v2-1.0.0", "family", tmp_path)



def test_complete_batch_reader_keeps_chunked_batch_results_and_authored_na(tmp_path):
    root = tmp_path / policy.CORPORA["batch"]
    shared = {"family": "family", "mode": "stream", "cases": {
        "A": {"chunks": [{"delta_text": "hello"}]},
        "NA": {"description": "Not applicable", "explanation": "No request for this family"}}}
    for name, document in [("inputs", shared), ("dynamo_v1-1.0.0", {"cases": {"A": {"expected": {"dynamo_v1": {"normal_text": "hello"}}}}})]:
        path = root / name / "family/CASE.yaml"
        path.parent.mkdir(parents=True)
        path.write_text(yaml.safe_dump(document))
    marker = tmp_path / ".reader-views/legacy-checkpoints.json"
    marker.parent.mkdir()
    marker.write_text('{"schema_version":1,"complete_family_snapshots":true}')
    cases = resolve_fixtures.resolve_docs(root, ["dynamo_v1-1.0.0"])[0][("family", "CASE.yaml")]["cases"]
    assert cases["A"] == {"chunks": [{"delta_text": "hello"}], "expected": {"dynamo_v1": {"normal_text": "hello"}}}
    assert cases["NA"] == shared["cases"]["NA"]


def test_empty_stream_checkpoint_does_not_claim_producer_header(tmp_path):
    root = tmp_path / policy.CORPORA["stream"]
    for name, document in [("inputs", {"cases": {"A": {"chunks": [{"delta_text": "hello"}]}}}),
                           ("dynamo_v2-1.0.0", {"cases": {}})]:
        path = root / name / "family/CASE.yaml"
        path.parent.mkdir(parents=True)
        path.write_text(yaml.safe_dump(document))
    marker = tmp_path / ".reader-views/legacy-checkpoints.json"
    marker.parent.mkdir()
    marker.write_text('{"schema_version":1,"complete_family_snapshots":true}')
    document = resolve_stream_fixtures.resolve_docs(root, ["dynamo_v2-1.0.0"])[0][("family", "CASE.yaml")]
    assert "captured_with" not in document
    assert "No observation" in document["cases"]["A"]["unavailable"]["dynamo_v2"]
    assert document["cases"]["A"]["chunks"] == [{"delta_text": "hello"}]


@pytest.mark.parametrize("replacement", [{"chunks": [{"expected": []}]}, {"exception": "recorded failure"}])
def test_later_stream_checkpoint_removes_synthetic_empty_wrapper(tmp_path, replacement):
    root = tmp_path / policy.CORPORA["stream"]
    inputs = {"cases": {"A": {"chunks": [{"delta_text": "x"}]}}}
    captures = [("inputs", inputs),
                ("dynamo_v2-1.0.0", {"cases": {}}),
                ("dynamo_v2-1.1.0", {"cases": {}}),
                ("dynamo_v2-1.2.0", {"cases": {"A": replacement}})]
    for name, document in captures:
        path = root / name / "family/CASE.yaml"
        path.parent.mkdir(parents=True)
        path.write_text(yaml.safe_dump(document))
    marker = tmp_path / ".reader-views/legacy-checkpoints.json"
    marker.parent.mkdir()
    marker.write_text('{"schema_version":1,"complete_family_snapshots":true}')
    case = resolve_stream_fixtures.resolve_docs(root, ["dynamo_v2-1.2.0"])[0][("family", "CASE.yaml")]["cases"]["A"]
    assert "unavailable" not in case
    if "exception" in replacement:
        assert case["exception"] == {"dynamo_v2": "recorded failure"}
    else:
        assert case["chunks"][0]["expected"] == {"dynamo_v2": []}


def test_stream_replacement_preserves_preexisting_empty_wrappers(tmp_path):
    root = tmp_path / policy.CORPORA["stream"]
    inputs = {"cases": {"A": {"chunks": [{"delta_text": "x"}]}}}
    captures = [("inputs", inputs),
                ("dynamo_v2-1.0.0", {"cases": {"A": {"unavailable": "previously unavailable"}}}),
                ("dynamo_v2-1.1.0", {"cases": {"A": {"chunks": [{"expected": [], "normal_text": "x"}]}}}),
                ("dynamo_v2-1.2.0", {"cases": {"A": {"chunks": [{"expected": []}]}}})]
    for name, document in captures:
        path = root / name / "family/CASE.yaml"
        path.parent.mkdir(parents=True)
        path.write_text(yaml.safe_dump(document))
    marker = tmp_path / ".reader-views/legacy-checkpoints.json"
    marker.parent.mkdir()
    marker.write_text('{"schema_version":1,"complete_family_snapshots":true}')
    case = resolve_stream_fixtures.resolve_docs(root, ["dynamo_v2-1.2.0"])[0][("family", "CASE.yaml")]["cases"]["A"]
    assert case["unavailable"] == {}
    assert case["chunks"][0]["normal_text"] == {}
