# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Own conformance CI assignments and commands independently of GitHub orchestration."""
import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]
TESTS = "conformance/utils/tests"

# These are the full test identities whose resource corpus is partitioned across
# qualification runners. Other tests execute once, rather than once per shard.
QUALIFICATION_NODES = (
    f"{TESTS}/test_viewer_snapshot_parity.py::test_bundled_history_json_snapshots_match_original_yaml",
    f"{TESTS}/test_viewer_snapshot_parity.py::test_full_corpus_json_snapshots_match_python_and_javascript_resolution",
    f"{TESTS}/test_unified_taxonomy_covers_corpus.py::test_reference_backfills_require_request_bound_version_receipts",
    f"{TESTS}/test_model.py::test_shared_javascript_matrix_contracts",
    f"{TESTS}/test_structural_response_case.py::test_structural_request_is_bound_in_every_retained_dynamo_column",
)
BROWSER_MODULES = (
    "test_browser_smoke.py", "test_browser_viewer.py", "test_native_publication_browser.py",
    "test_capture_split_browser.py", "test_browser_download_scheduler.py",
)
VIEWER_SMOKE_TESTS = (
    "test_schema_popup_opens_and_fits_across_themes_viewports_and_transpose",
    "test_popup_headings_include_compact_references",
)
FIRST_MATRIX_TEST = "test_first_visible_matrix_matches_transpose_url"


@dataclass(frozen=True)
class Lane:
    name: str
    count: int = 1
    processes: int = 0
    bundler: bool = False
    partition: str = "none"


# Counts and process limits preserve the existing runner workloads, not model or
# release counts. Browser lanes remain separate machines from CPU-heavy checks.
LANES = (
    Lane("core", processes=4, bundler=True),
    Lane("qualification", count=8, processes=3, partition="resources"),
    Lane("native-browser", bundler=True),
    Lane("browser", count=2, partition="tests"),
    Lane("viewer", count=3, bundler=True, partition="tests"),
    Lane("first-matrix"),
)
FIXTURES = Lane("fixtures", processes=4)


def qualification_nodes():
    return QUALIFICATION_NODES


def plan(lanes=None):
    lanes = LANES if lanes is None else tuple(lanes)
    if not lanes or len({lane.name for lane in lanes}) != len(lanes):
        raise ValueError("CI plan requires unique, nonempty lane names")
    if any(not isinstance(lane.name, str) or re.fullmatch(r"[a-z][a-z0-9-]*", lane.name) is None
           or type(lane.count) is not int or lane.count < 1
           or type(lane.processes) is not int or lane.processes < 0
           or type(lane.bundler) is not bool
           or lane.partition not in {"none", "tests", "resources"} for lane in lanes):
        raise ValueError("CI plan requires positive counts and nonnegative process limits")
    result = [{"id": f"{lane.name}-{index}", "lane": lane.name, "index": index,
               "count": lane.count, "bundler": lane.bundler}
              for lane in lanes for index in range(lane.count)]
    if len({item["id"] for item in result}) != len(result):
        raise ValueError("CI plan contains duplicate worker identities")
    return result


def entry(lane, index=0, count=None):
    if type(index) is not int or count is not None and type(count) is not int:
        raise ValueError(f"unknown CI lane or invalid coordinates: {lane} {index}/{count}")
    candidates = plan((*LANES, FIXTURES))
    matches = [item for item in candidates if item["lane"] == lane and item["index"] == index
               and (count is None or item["count"] == count)]
    if len(matches) != 1:
        raise ValueError(f"unknown CI lane or invalid coordinates: {lane} {index}/{count}")
    return matches[0]


def lane_config(name):
    return next(lane for lane in (*LANES, FIXTURES) if lane.name == name)


def test_group(item):
    return f"{item['index']}/{item['count']}" if lane_config(item["lane"]).partition == "tests" else None


def resource_shard(item):
    return {"index": item["index"], "count": item["count"]} if lane_config(item["lane"]).partition == "resources" else {"index": 0, "count": 1}


def pytest_arguments(lane):
    if lane in {"fixtures", "core"}:
        return [TESTS, "-q", "--conformance-scope", "fixtures" if lane == "fixtures" else "publication",
                *[f"--ignore={TESTS}/{name}" for name in BROWSER_MODULES],
                *[f"--deselect={node}" for node in QUALIFICATION_NODES]]
    if lane == "qualification":
        return [*QUALIFICATION_NODES, "-vv", "-rP", "--durations=25"]
    if lane == "native-browser":
        return [*[f"{TESTS}/{name}" for name in BROWSER_MODULES[2:]], "-vv", "--durations=25"]
    if lane == "browser":
        return [f"{TESTS}/test_browser_smoke.py", "-k",
                " and ".join(f"not {name}" for name in (FIRST_MATRIX_TEST, *VIEWER_SMOKE_TESTS)),
                "-vv", "--durations=25"]
    if lane == "first-matrix":
        return [f"{TESTS}/test_browser_smoke.py", "-k", FIRST_MATRIX_TEST, "-vv", "--durations=25"]
    if lane == "viewer":
        return [f"{TESTS}/test_browser_viewer.py",
                *[f"{TESTS}/test_browser_smoke.py::{name}" for name in VIEWER_SMOKE_TESTS], "-vv", "--durations=25"]
    if lane == "all":
        return [TESTS, "-q"]
    raise ValueError(f"unknown CI lane: {lane}")


def plan_digest():
    lanes = (*LANES, FIXTURES)
    policy = {"lanes": [asdict(lane) for lane in lanes],
              "pytest": {lane.name: pytest_arguments(lane.name) for lane in lanes}}
    return hashlib.sha256(json.dumps(policy, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def lane_environment(item, environment):
    environment = dict(environment)
    environment.update({"CONFORMANCE_PLAN_ID": plan_digest(), "CONFORMANCE_PLAN_ENTRY": item["id"],
                        "CONFORMANCE_CI_LANE": item["lane"], "CONFORMANCE_PLAN_INDEX": str(item["index"]),
                        "CONFORMANCE_PLAN_COUNT": str(item["count"]), "CONFORMANCE_PARITY_SHARD": "0",
                        "CONFORMANCE_PARITY_SHARDS": "1"})
    resource = resource_shard(item)
    environment.update({"CONFORMANCE_PARITY_SHARD": str(resource["index"]),
                        "CONFORMANCE_PARITY_SHARDS": str(resource["count"])})
    if environment.get("GITHUB_ACTIONS") == "true":
        environment.update({"CONFORMANCE_INCLUDE_HISTORY": "true", "CONFORMANCE_COVERAGE": "exhaustive"})
    return environment


def setup():
    # RUNNER_TEMP is private to the GitHub job; use one environment across its
    # steps so publication, gate, and lane bootstraps cannot drift.
    runtime = Path(os.environ["RUNNER_TEMP"])
    binary = runtime / "conformance-ci-venv" / "bin"
    subprocess.run([sys.executable, "-m", "venv", str(binary.parent)], check=True)
    python = str(binary / "python3")
    subprocess.run([python, "-m", "pip", "install", "--quiet", "-r",
                    str(ROOT / "conformance/utils/ci-requirements.in")], check=True)
    # GitHub command files affect later steps; same-process shell and Node
    # children must resolve the same Python environment immediately.
    os.environ["CONFORMANCE_CI_PYTHON"] = python
    if os.environ.get("PATH", "").split(os.pathsep, 1)[0] != str(binary):
        os.environ["PATH"] = str(binary) + os.pathsep + os.environ.get("PATH", "")
    for variable, value in (("GITHUB_PATH", str(binary)),
                            ("GITHUB_ENV", f"CONFORMANCE_CI_PYTHON={python}")):
        if os.environ.get(variable):
            with Path(os.environ[variable]).open("a") as output:
                output.write(value + "\n")
    return python


def baseline(python, environment):
    destination = environment.get("CONFORMANCE_BASELINE_DIR")
    if not destination and environment.get("GITHUB_ACTIONS") == "true":
        destination = str(Path(environment["RUNNER_TEMP"]) / "baseline-selection")
    if not destination:
        return
    baseline_environment = {key: value for key, value in environment.items()
                            if not key.startswith("CONFORMANCE_PLAN_")
                            and key not in {"CONFORMANCE_CI_LANE", "PYTEST_ADDOPTS", "CONFORMANCE_CI_SHARD"}}
    baseline_environment["CONFORMANCE_SELECTION_DIR"] = destination
    path = Path(destination)
    path.mkdir(parents=True, exist_ok=True)
    # Collection is independent of lane membership; omitted selectors must fail
    # accounting rather than shrink both the plan and its expected inventory.
    with (path.parent / "test-inventory.log").open("w") as output:
        subprocess.run([python, "-m", "pytest", TESTS, "--collect-only", "-q"], cwd=ROOT,
                       env=baseline_environment, stdout=output, stderr=subprocess.STDOUT, check=True)


def run_lane(lane, index=None, count=None, *, python=None):
    environment = dict(os.environ)
    python = python or environment.get("CONFORMANCE_CI_PYTHON", sys.executable)
    config = None if lane == "all" else lane_config(lane)
    # Local commands historically run the whole lane, with optional user-supplied
    # pytest filtering. Only explicit coordinates opt into the distributed plan.
    local_unplanned = environment.get("GITHUB_ACTIONS") != "true" and index is None and count is None
    arguments = pytest_arguments(lane)
    if lane == "all" or local_unplanned:
        environment = {key: value for key, value in environment.items()
                       if not key.startswith("CONFORMANCE_PLAN_") and key != "CONFORMANCE_CI_LANE"}
        resource_index, resource_count = 0, 1
        if lane != "all" and config.partition == "resources":
            # Local resource partitioning predates the hosted worker plan and
            # permits arbitrary budgets, with independent defaults of 0 and 1.
            resource_index = int(environment.get("CONFORMANCE_PARITY_SHARD", "0"))
            resource_count = int(environment.get("CONFORMANCE_PARITY_SHARDS", "1"))
            if not 0 <= resource_index < resource_count:
                raise ValueError("invalid local resource coordinates")
        environment.update({"CONFORMANCE_PARITY_SHARD": str(resource_index),
                            "CONFORMANCE_PARITY_SHARDS": str(resource_count)})
    else:
        if index is None and config.partition == "resources" and "CONFORMANCE_PARITY_SHARD" in environment:
            index = int(environment["CONFORMANCE_PARITY_SHARD"])
        if count is None and config.partition == "resources" and "CONFORMANCE_PARITY_SHARDS" in environment:
            count = int(environment["CONFORMANCE_PARITY_SHARDS"])
        if environment.get("GITHUB_ACTIONS") == "true" and config.count > 1 and (index is None or count is None):
            raise ValueError("multi-worker CI lanes require explicit coordinates")
        index = 0 if index is None else index
        item = entry(lane, index, count)
        environment = lane_environment(item, environment)
        group = test_group(item)
        if group is not None:
            arguments.append(f"--conformance-group={group}")
    if config is not None and config.processes:
        arguments.extend([f"--numprocesses={config.processes}", "--dist=loadfile"])
        if config.partition != "resources":
            arguments.extend(["--no-loadscope-reorder", "--durations=25", "-vv"])
    if lane == "fixtures":
        baseline(python, environment)
    if environment.get("GITHUB_ACTIONS") == "true":
        environment.update({"CONFORMANCE_INCLUDE_HISTORY": "true", "CONFORMANCE_COVERAGE": "exhaustive"})
    if lane in {"all", "core"}:
        history = ["--include-history"] if environment.get("CONFORMANCE_INCLUDE_HISTORY") == "true" else []
        subprocess.run([python, str(ROOT / "conformance/utils/src/viewer_catalog.py"), "--check", *history],
                       cwd=ROOT, env=environment, check=True)
    subprocess.run([python, "-m", "pytest", *arguments], cwd=ROOT, env=environment, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", action="store_true")
    parser.add_argument("--setup", action="store_true")
    parser.add_argument("--run", choices=[*(lane.name for lane in LANES), "fixtures", "all"])
    parser.add_argument("--index", type=int)
    parser.add_argument("--count", type=int)
    args = parser.parse_args()
    if args.matrix:
        if args.setup or args.run or args.index is not None or args.count is not None:
            parser.error("--matrix cannot be combined with execution options")
        print(json.dumps(plan()))
        return
    if not args.setup and not args.run or not args.run and (args.index is not None or args.count is not None):
        parser.error("--setup or --run is required; coordinates require --run")
    python = setup() if args.setup else None
    if args.run:
        run_lane(args.run, args.index, args.count, python=python)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        sys.exit(error.returncode)
