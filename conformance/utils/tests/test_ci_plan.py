# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from dataclasses import replace
import json
import os
import subprocess
import sys

import pytest

import ci_coverage
import ci_plan


@pytest.mark.parametrize("module", sorted(path.name for path in (ci_plan.ROOT / ci_plan.TESTS).glob("test_*browser*.py")))
def test_browser_modules_have_a_browser_lane(module):
    selector = f"{ci_plan.TESTS}/{module}"
    for lane in ("fixtures", "core"):
        assert f"--ignore={selector}" in ci_plan.pytest_arguments(lane)
    owners = [lane.name for lane in ci_plan.LANES if selector in ci_plan.pytest_arguments(lane.name)]
    assert owners
    assert set(owners) <= {"browser", "viewer", "native-browser", "first-matrix"}


def test_matrix_changes_group_count_without_workflow_changes(monkeypatch):
    before = ci_plan.plan()
    original = next(lane for lane in ci_plan.LANES if lane.name == "viewer")
    # Deliberately change the workload budget to prove no three-group protocol.
    count = original.count + 2
    monkeypatch.setattr(ci_plan, "LANES", tuple(replace(lane, count=count) if lane is original else lane
                                             for lane in ci_plan.LANES))
    after = ci_plan.plan()
    workers = [item for item in after if item["lane"] == original.name]
    assert [item["index"] for item in workers] == list(range(count))
    assert all(item["count"] == count for item in workers)
    assert all(ci_plan.test_group(item) == f"{item['index']}/{count}" for item in workers)
    assert [item for item in before if item["lane"] != original.name] == [
        item for item in after if item["lane"] != original.name]
    assert all(set(item) == {"id", "lane", "index", "count", "bundler"} for item in after)


@pytest.mark.parametrize("fault", ["empty", "duplicate", "zero", "negative", "processes", "partition", "name"])
def test_plan_rejects_invalid_execution_policy(fault):
    lane = ci_plan.LANES[0]
    lanes = [] if fault == "empty" else [lane, lane] if fault == "duplicate" else [
        replace(lane, **({"count": 0} if fault == "zero" else {"count": -1} if fault == "negative"
                        else {"processes": -1} if fault == "processes" else {"name": "bad'quoted-name"}
                        if fault == "name" else {"partition": "unknown"}))]
    with pytest.raises(ValueError, match="CI plan"):
        ci_plan.plan(lanes)


def test_plan_identity_includes_commands_and_worker_policy(monkeypatch):
    before = ci_plan.plan_digest()
    matrix = ci_plan.plan()
    # A command change cannot keep admitting receipts from the previous policy.
    monkeypatch.setattr(ci_plan, "QUALIFICATION_NODES", (*ci_plan.QUALIFICATION_NODES, "synthetic.py::test_extra"))
    assert ci_plan.plan() == matrix
    assert ci_plan.plan_digest() != before
    after = ci_plan.plan_digest()
    monkeypatch.setattr(ci_plan, "LANES", tuple(replace(lane, processes=lane.processes + 1)
                                             for lane in ci_plan.LANES))
    assert ci_plan.plan_digest() != after


@pytest.mark.parametrize("coordinate", ["negative", "outside", "wrong-count", "unknown"])
def test_entry_rejects_unknown_or_inconsistent_worker(coordinate):
    first = ci_plan.plan()[0]
    name = "unknown" if coordinate == "unknown" else first["lane"]
    index = -1 if coordinate == "negative" else first["count"] if coordinate == "outside" else 0
    count = first["count"] + 1 if coordinate == "wrong-count" else first["count"]
    with pytest.raises(ValueError, match="invalid coordinates"):
        ci_plan.entry(name, index, count)


def test_resource_coordinates_are_not_total_matrix_size():
    lane = next(lane for lane in ci_plan.LANES if lane.partition == "resources")
    item = ci_plan.entry(lane.name, lane.count - 1, lane.count)
    assert lane.count != len(ci_plan.plan())
    stale = str(len(ci_plan.plan()) + 1)
    environment = ci_plan.lane_environment(item, {"CONFORMANCE_PARITY_SHARD": stale, "CONFORMANCE_PARITY_SHARDS": stale})
    assert environment["CONFORMANCE_PARITY_SHARD"] == str(lane.count - 1)
    assert environment["CONFORMANCE_PARITY_SHARDS"] == str(lane.count)
    assert ci_plan.resource_shard(item) == {"index": lane.count - 1, "count": lane.count}
    for other in ci_plan.plan():
        if other["lane"] != lane.name:
            assert ci_plan.resource_shard(other) == {"index": 0, "count": 1}


@pytest.mark.parametrize("history", [False, True])
def test_core_runner_preserves_local_history_and_user_pytest_options(monkeypatch, history):
    calls = []
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.setenv("CONFORMANCE_INCLUDE_HISTORY", str(history).lower())
    monkeypatch.setenv("PYTEST_ADDOPTS", "--maxfail=2")
    monkeypatch.setattr(ci_plan.subprocess, "run", lambda args, **kwargs: calls.append((args, kwargs)))
    ci_plan.run_lane("core")
    assert calls[0][0][1:] == [str(ci_plan.ROOT / "conformance/utils/src/viewer_catalog.py"), "--check",
                                   *(["--include-history"] if history else [])]
    command, options = calls[1]
    assert command[1:3] == ["-m", "pytest"]
    assert options["env"]["PYTEST_ADDOPTS"] == "--maxfail=2"
    assert options["env"]["CONFORMANCE_INCLUDE_HISTORY"] == str(history).lower()


def test_ci_runner_forces_history_and_clears_stale_resource_coordinates(monkeypatch):
    calls = []
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("CONFORMANCE_INCLUDE_HISTORY", "false")
    monkeypatch.setenv("CONFORMANCE_COVERAGE", "latest-two")
    stale = str(len(ci_plan.plan()) + 1)
    monkeypatch.setenv("CONFORMANCE_PARITY_SHARD", stale)
    monkeypatch.setenv("CONFORMANCE_PARITY_SHARDS", stale)
    monkeypatch.setattr(ci_plan.subprocess, "run", lambda args, **kwargs: calls.append((args, kwargs)))
    lane = next(lane for lane in ci_plan.LANES if lane.partition == "tests")
    item = ci_plan.entry(lane.name, lane.count - 1, lane.count)
    ci_plan.run_lane(item["lane"], item["index"], item["count"])
    command, options = calls[0]
    environment = options["env"]
    assert environment["CONFORMANCE_INCLUDE_HISTORY"] == "true"
    assert environment["CONFORMANCE_COVERAGE"] == "exhaustive"
    assert environment["CONFORMANCE_PARITY_SHARD"] == "0"
    assert environment["CONFORMANCE_PARITY_SHARDS"] == "1"
    assert environment["CONFORMANCE_PLAN_ID"] == ci_plan.plan_digest()
    assert environment["CONFORMANCE_PLAN_ENTRY"] == item["id"]
    assert f"--conformance-group={item['index']}/{item['count']}" in command


def test_fixture_baseline_is_unfiltered_and_has_no_worker_identity(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("RUNNER_TEMP", str(tmp_path))
    monkeypatch.delenv("CONFORMANCE_BASELINE_DIR", raising=False)
    monkeypatch.setenv("PYTEST_ADDOPTS", "-k imaginary_subset")
    monkeypatch.setattr(ci_plan.subprocess, "run", lambda args, **kwargs: calls.append((args, kwargs)))
    ci_plan.run_lane("fixtures")
    baseline_command, baseline_options = calls[0]
    assert baseline_command[1:] == ["-m", "pytest", "conformance/utils/tests", "--collect-only", "-q"]
    baseline_environment = baseline_options["env"]
    assert "PYTEST_ADDOPTS" not in baseline_environment
    assert "CONFORMANCE_CI_LANE" not in baseline_environment
    assert not any(name.startswith("CONFORMANCE_PLAN_") for name in baseline_environment)
    assert baseline_environment["CONFORMANCE_SELECTION_DIR"] == str(tmp_path / "baseline-selection")
    assert baseline_environment["CONFORMANCE_INCLUDE_HISTORY"] == "true"
    assert calls[1][1]["env"]["CONFORMANCE_PLAN_ENTRY"] == ci_plan.entry("fixtures", 0, 1)["id"]


def test_matrix_cli_uses_same_plan():
    result = subprocess.run([sys.executable, ci_plan.__file__, "--matrix"], capture_output=True, text=True, check=True)
    assert json.loads(result.stdout) == ci_plan.plan()


def test_all_keeps_local_full_suite_without_ci_partition(monkeypatch):
    calls = []
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.setenv("CONFORMANCE_INCLUDE_HISTORY", "false")
    stale = str(len(ci_plan.plan()) + 1)
    monkeypatch.setenv("CONFORMANCE_PARITY_SHARD", stale)
    monkeypatch.setenv("CONFORMANCE_PARITY_SHARDS", stale)
    monkeypatch.setenv("CONFORMANCE_PLAN_ENTRY", "stale")
    monkeypatch.setattr(ci_plan.subprocess, "run", lambda args, **kwargs: calls.append((args, kwargs)))
    ci_plan.run_lane("all")
    assert calls[1][0][1:] == ["-m", "pytest", "conformance/utils/tests", "-q"]
    environment = calls[1][1]["env"]
    assert environment["CONFORMANCE_PARITY_SHARD"] == "0"
    assert environment["CONFORMANCE_PARITY_SHARDS"] == "1"
    assert "CONFORMANCE_PLAN_ENTRY" not in environment
    assert environment["CONFORMANCE_INCLUDE_HISTORY"] == "false"


def test_local_qualification_covers_full_corpus_until_sharding_is_explicit(monkeypatch):
    calls = []
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.delenv("CONFORMANCE_PARITY_SHARD", raising=False)
    monkeypatch.delenv("CONFORMANCE_PARITY_SHARDS", raising=False)
    monkeypatch.setenv("CONFORMANCE_PLAN_ENTRY", "stale")
    monkeypatch.setattr(ci_plan.subprocess, "run", lambda args, **kwargs: calls.append(kwargs["env"]))
    lane = next(lane for lane in ci_plan.LANES if lane.partition == "resources")
    resources = [{"path": f"resource-{index}"} for index in range(2 * lane.count + 1)]

    def selected(environment):
        return ci_coverage.select_resources(resources, profile="exhaustive",
            shard=int(environment["CONFORMANCE_PARITY_SHARD"]),
            shards=int(environment["CONFORMANCE_PARITY_SHARDS"]))

    ci_plan.run_lane(lane.name)
    assert selected(calls[-1]) == sorted(resources, key=lambda row: row["path"])
    assert "CONFORMANCE_CI_LANE" not in calls[-1]
    assert not any(name.startswith("CONFORMANCE_PLAN_") for name in calls[-1])
    for ci in (False, True):
        monkeypatch.setenv("GITHUB_ACTIONS", str(ci).lower())
        ci_plan.run_lane(lane.name, lane.count - 1, lane.count)
        environment = calls[-1]
        assert environment["CONFORMANCE_PLAN_ENTRY"] == ci_plan.entry(lane.name, lane.count - 1, lane.count)["id"]
        assert environment["CONFORMANCE_PARITY_SHARD"] == str(lane.count - 1)
        assert environment["CONFORMANCE_PARITY_SHARDS"] == str(lane.count)
        assert 0 < len(selected(environment)) < len(resources)


@pytest.mark.parametrize("lane", [lane for lane in ci_plan.LANES if lane.partition != "none"], ids=lambda lane: lane.name)
def test_local_partitioned_lanes_run_fully_unless_coordinates_are_explicit(monkeypatch, lane):
    calls = []
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.delenv("CONFORMANCE_PARITY_SHARD", raising=False)
    monkeypatch.delenv("CONFORMANCE_PARITY_SHARDS", raising=False)
    # Keep an operator-supplied subset option without adding a competing group.
    options = f"--conformance-group=0/{lane.count}" if lane.partition == "tests" else "--maxfail=2"
    monkeypatch.setenv("PYTEST_ADDOPTS", options)
    monkeypatch.setattr(ci_plan.subprocess, "run", lambda args, **kwargs: calls.append((args, kwargs["env"])))
    ci_plan.run_lane(lane.name)
    command, environment = calls[-1]
    assert not any(argument.startswith("--conformance-group") for argument in command)
    assert environment["PYTEST_ADDOPTS"] == options
    assert environment["CONFORMANCE_PARITY_SHARD"] == "0"
    assert environment["CONFORMANCE_PARITY_SHARDS"] == "1"
    assert not any(name.startswith("CONFORMANCE_PLAN_") for name in environment)
    assert "CONFORMANCE_CI_LANE" not in environment
    ci_plan.run_lane(lane.name, lane.count - 1, lane.count)
    command, environment = calls[-1]
    assert environment["CONFORMANCE_PLAN_ENTRY"] == ci_plan.entry(lane.name, lane.count - 1, lane.count)["id"]
    if lane.partition == "tests":
        assert f"--conformance-group={lane.count - 1}/{lane.count}" in command
    else:
        assert environment["CONFORMANCE_PARITY_SHARD"] == str(lane.count - 1)
        assert environment["CONFORMANCE_PARITY_SHARDS"] == str(lane.count)


@pytest.mark.parametrize("lane", [lane for lane in ci_plan.LANES if lane.count > 1], ids=lambda lane: lane.name)
def test_multiworker_ci_requires_explicit_coordinates(monkeypatch, lane):
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.delenv("CONFORMANCE_PARITY_SHARD", raising=False)
    monkeypatch.delenv("CONFORMANCE_PARITY_SHARDS", raising=False)
    with pytest.raises(ValueError, match="require explicit coordinates"):
        ci_plan.run_lane(lane.name)


def test_local_resource_environment_preserves_unplanned_sharding(monkeypatch):
    calls = []
    lane = next(lane for lane in ci_plan.LANES if lane.partition == "resources")
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.setenv("CONFORMANCE_PARITY_SHARD", str(lane.count - 1))
    monkeypatch.setenv("CONFORMANCE_PARITY_SHARDS", str(lane.count))
    monkeypatch.setattr(ci_plan.subprocess, "run", lambda args, **kwargs: calls.append(kwargs["env"]))
    ci_plan.run_lane(lane.name)
    assert "CONFORMANCE_PLAN_ENTRY" not in calls[-1]
    assert calls[-1]["CONFORMANCE_PARITY_SHARD"] == str(lane.count - 1)
    assert calls[-1]["CONFORMANCE_PARITY_SHARDS"] == str(lane.count)
    monkeypatch.setenv("CONFORMANCE_PARITY_SHARDS", str(lane.count + 1))
    ci_plan.run_lane(lane.name)
    assert calls[-1]["CONFORMANCE_PARITY_SHARDS"] == str(lane.count + 1)


def test_setup_exposes_virtualenv_to_same_process_children(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setenv("RUNNER_TEMP", str(tmp_path))
    monkeypatch.setenv("GITHUB_PATH", str(tmp_path / "github-path"))
    monkeypatch.setenv("GITHUB_ENV", str(tmp_path / "github-env"))
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.setenv("PATH", os.environ["PATH"])
    monkeypatch.setenv("CONFORMANCE_CI_PYTHON", "previous-python")
    monkeypatch.setattr(ci_plan.subprocess, "run", lambda args, **kwargs: calls.append((args, kwargs)))
    monkeypatch.setattr(sys, "argv", [ci_plan.__file__, "--setup", "--run", "core"])
    ci_plan.main()
    binary = tmp_path / "conformance-ci-venv/bin"
    python = str(binary / "python3")
    for command, options in calls[-2:]:
        assert command[0] == python
        assert options["env"]["PATH"].split(os.pathsep)[0] == str(binary)
        assert options["env"]["CONFORMANCE_CI_PYTHON"] == python
    assert (tmp_path / "github-path").read_text().splitlines()[-1] == str(binary)
    assert (tmp_path / "github-env").read_text().splitlines()[-1] == f"CONFORMANCE_CI_PYTHON={python}"


@pytest.mark.parametrize("case", ["shard-only", "count-only", "custom", "negative-index", "outside", "zero-count", "negative-count"])
def test_local_resource_coordinate_defaults_and_validation(monkeypatch, case):
    calls = []
    lane = next(lane for lane in ci_plan.LANES if lane.partition == "resources")
    custom = lane.count + 1
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.delenv("CONFORMANCE_PARITY_SHARD", raising=False)
    monkeypatch.delenv("CONFORMANCE_PARITY_SHARDS", raising=False)
    monkeypatch.setattr(ci_plan.subprocess, "run", lambda args, **kwargs: calls.append(kwargs["env"]))
    index = 0 if case in {"shard-only", "count-only", "zero-count", "negative-count"} else custom - 1
    count = 1 if case == "shard-only" else custom
    if case == "negative-index":
        index = -1
    elif case == "outside":
        index = count
    elif case == "zero-count":
        count = 0
    elif case == "negative-count":
        count = -1
    if case != "count-only":
        monkeypatch.setenv("CONFORMANCE_PARITY_SHARD", str(index))
    if case != "shard-only":
        monkeypatch.setenv("CONFORMANCE_PARITY_SHARDS", str(count))
    if not 0 <= index < count:
        with pytest.raises(ValueError, match="invalid local resource"):
            ci_plan.run_lane(lane.name)
        assert not calls
        return
    ci_plan.run_lane(lane.name)
    assert calls[-1]["CONFORMANCE_PARITY_SHARD"] == str(index)
    assert calls[-1]["CONFORMANCE_PARITY_SHARDS"] == str(count)
    assert not any(name.startswith("CONFORMANCE_PLAN_") for name in calls[-1])


def test_custom_local_resource_partitions_cover_corpus_once(monkeypatch):
    calls = []
    lane = next(lane for lane in ci_plan.LANES if lane.partition == "resources")
    count = lane.count + 1
    resources = [{"path": f"resource-{index}"} for index in range(2 * count + 1)]
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.setenv("CONFORMANCE_PARITY_SHARDS", str(count))
    monkeypatch.setattr(ci_plan.subprocess, "run", lambda args, **kwargs: calls.append(kwargs["env"]))
    observed = []
    for index in range(count):
        monkeypatch.setenv("CONFORMANCE_PARITY_SHARD", str(index))
        ci_plan.run_lane(lane.name)
        environment = calls[-1]
        assert not any(name.startswith("CONFORMANCE_PLAN_") for name in environment)
        observed.extend(row["path"] for row in ci_coverage.select_resources(resources, profile="exhaustive",
            shard=int(environment["CONFORMANCE_PARITY_SHARD"]), shards=int(environment["CONFORMANCE_PARITY_SHARDS"])))
    assert sorted(observed) == sorted(row["path"] for row in resources)


@pytest.mark.parametrize("fault", ["unknown-field", "bad-count", "duplicate", "missing-fixtures", "partitioned-fixtures"])
def test_lane_file_uses_plan_validation(tmp_path, fault):
    rows = json.loads((ci_plan.ROOT / "conformance/utils/ci-lanes.json").read_bytes())
    if fault == "unknown-field":
        rows[0]["unknown"] = 1
    elif fault == "bad-count":
        rows[0]["count"] = True
    elif fault == "duplicate":
        rows.append(rows[0])
    elif fault == "missing-fixtures":
        rows.pop()
    else:
        rows[-1]["partition"] = "tests"
    path = tmp_path / "lanes.json"
    path.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="CI plan"):
        ci_plan.load_lanes(path)


def test_lane_file_preserves_all_worker_defaults():
    lanes, fixtures = ci_plan.load_lanes()
    assert lanes == ci_plan.LANES
    assert fixtures == ci_plan.FIXTURES
    assert len(ci_plan.plan()) == 16
