# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import yaml

import report_capture_verification


@pytest.mark.parametrize("fault", [None, "golden", "recorded", "missing", "empty", "wrong_severity", "stale"])
def test_executed_receipts_warn_for_golden_and_fail_for_unverified_yaml(tmp_path, fault, monkeypatch):
    monkeypatch.delenv("CONFORMANCE_CANDIDATE_TREE", raising=False)
    monkeypatch.delenv("GITHUB_SHA", raising=False)
    if fault == "stale":
        monkeypatch.setenv("GITHUB_SHA", "new-head")
    for suite, blocking in report_capture_verification.SUITES.items():
        if fault == "missing" and blocking:
            continue
        differences = ["case: expected differs from actual"] if fault == ("recorded" if blocking else "golden") else []
        document = {"schema": 1, "suite": suite, "checked": 0 if fault == "empty" else 1,
                    "blocking": False if fault == "wrong_severity" else blocking, "differences": differences}
        (tmp_path / f"{suite}.json").write_text(json.dumps(document))
    script = Path(report_capture_verification.__file__)
    result = subprocess.run([sys.executable, str(script), "--directory", str(tmp_path), "--summary", str(tmp_path / "summary")],
                            capture_output=True, text=True)
    assert (result.returncode == 0) == (fault in {None, "golden"}), result.stderr
    if fault == "golden":
        assert "::warning::" in result.stdout
        assert "::error::" not in result.stdout
    if fault in {"recorded", "missing"}:
        assert "::error::" in result.stdout
    if fault in {None, "golden", "recorded", "missing"}:
        assert "recorded-capture" in (tmp_path / "summary").read_text()


def test_required_rust_job_publishes_executed_comparisons_even_after_failure():
    root = Path(__file__).resolve().parents[3]
    workflow = yaml.safe_load((root / ".github/workflows/ci.yml").read_text())
    steps = workflow["jobs"]["rust"]["steps"]
    execute = next(step for step in steps if step.get("id") == "rust")
    assert execute["run"] == "ci/rust.sh"
    upload = next(step for step in steps if step.get("name") == "Retain executed capture comparisons")
    assert upload["if"] == "${{ !cancelled() && steps.rust.outputs.capture_report_directory != '' }}"
    assert upload["with"]["path"] == "${{ steps.rust.outputs.capture_report_directory }}/*.json"
    assert upload["with"]["if-no-files-found"] == "error"
    detect = next(step for step in workflow["jobs"]["changes"]["steps"] if step.get("id") == "filter")
    filters = yaml.safe_load(detect["with"]["filters"])
    assert "ci/**" in filters["conformance"]
    assert "ci/**" in filters["structural-tag"]
    assert not (root / ".github/workflows/capture-approval.yml").exists()


@pytest.mark.parametrize("fault,code,started", [
    ("success", 0, True), ("before-tests", 23, False), ("oracle-setup", 23, False), ("during-tests", 23, True),
    ("after-tests", 23, True), ("during-tests-missing", 23, True),
    ("missing", 1, True), ("golden", 0, True), ("recorded", 1, True),
    ("managed-grammar", 0, True), ("grammar-setup", 23, False), ("grammar-deps", 23, False),
])
def test_rust_ci_retains_real_receipts_only_after_tests_start(tmp_path, fault, code, started):
    root = Path(__file__).resolve().parents[3]
    commands = tmp_path / "commands"
    commands.mkdir()
    stub = f"#!{sys.executable}\n" + textwrap.dedent(f"""\
        import json
        import os
        import sys
        from pathlib import Path

        tool = Path(sys.argv[0]).name
        args = sys.argv[1:]
        fault = os.environ["RUST_CI_FAULT"]
        with open(os.environ["RUST_CI_TRACE"], "a") as trace:
            trace.write(json.dumps({{"tool": tool, "args": args, "optimized": os.environ.get("PYTHONOPTIMIZE"),
                                    "golden_root": os.environ.get("CONFORMANCE_GOLDEN_ROOT")}}) + "\\n")
        if tool == "python3":
            if args[:2] == ["-m", "venv"]:
                if fault == "grammar-setup":
                    sys.exit(23)
                scripts = Path(args[2]) / "bin"
                scripts.mkdir(parents=True, exist_ok=True)
                pip = scripts / "pip"
                pip.write_text(Path(__file__).read_text())
                pip.chmod(0o755)
                sys.exit(0)
            if args[0].endswith("gen_unified_golden.py"):
                if fault == "oracle-setup":
                    sys.exit(23)
                output = Path(args[args.index("--output") + 1])
                output.mkdir(parents=True, exist_ok=True)
                (output / "prepared").touch()
                sys.exit(0)
            os.execv(sys.executable, [sys.executable, *args])
        if tool == "pip" and fault == "grammar-deps":
            sys.exit(23)
        if tool == "cargo":
            if args[:2] == ["nextest", "run"]:
                assert Path(os.environ["CONFORMANCE_GOLDEN_ROOT"], "prepared").is_file()
                directory = Path(os.environ["CONFORMANCE_CAPTURE_REPORT_DIR"])
                for suite, blocking in {report_capture_verification.SUITES!r}.items():
                    if fault in {{"missing", "during-tests-missing"}} and blocking:
                        continue
                    differences = ["difference"] if fault == ("recorded" if blocking else "golden") else []
                    (directory / f"{{suite}}.json").write_text(json.dumps({{
                        "schema": 1, "suite": suite, "checked": 1, "blocking": blocking, "differences": differences,
                    }}))
            if ((fault == "before-tests" and args[0] == "fmt")
                    or (fault in {{"during-tests", "during-tests-missing"}} and args[0] == "nextest")
                    or (fault == "after-tests" and args[0] == "publish")):
                sys.exit(23)
        """)
    for name in ("cargo", "sudo", "python3"):
        path = commands / name
        path.write_text(stub)
        path.chmod(0o755)
    environment = {
        "PATH": f"{commands}{os.pathsep}{os.defpath}", "RUNNER_TEMP": str(tmp_path),
        "GITHUB_OUTPUT": str(tmp_path / "outputs"), "GITHUB_STEP_SUMMARY": str(tmp_path / "summary"),
        "RUST_CI_FAULT": fault, "RUST_CI_TRACE": str(tmp_path / "trace"),
    }
    if fault not in {"managed-grammar", "grammar-setup", "grammar-deps"}:
        environment["CONFORMANCE_GRAMMAR_PYTHON"] = sys.executable
    for _ in range(2 if fault == "managed-grammar" else 1):
        result = subprocess.run([str(root / "ci/rust.sh")], capture_output=True, text=True, env=environment)
        assert result.returncode == code, result.stdout + result.stderr
    assert ("Capture verification" in (tmp_path / "summary").read_text()) == started
    outputs = tmp_path / "outputs"
    assert outputs.exists() == started
    if fault == "managed-grammar":
        trace = list(map(json.loads, (tmp_path / "trace").read_text().splitlines()))
        virtualenvs = [Path(item["args"][2]) for item in trace if item["args"][:2] == ["-m", "venv"]]
        assert len(virtualenvs) == len(set(virtualenvs)) == 2
        assert all(path.parent.parent == tmp_path for path in virtualenvs)
    if started:
        directory = Path(outputs.read_text().splitlines()[-1].removeprefix("capture_report_directory="))
        assert directory.is_dir()
        assert directory.parent.parent == tmp_path
    else:
        assert "Missing executed comparison" not in result.stdout
    if fault == "golden":
        assert "::warning::" in result.stdout
        assert "::error::" not in result.stdout
    if fault in {"missing", "during-tests-missing", "recorded"}:
        assert "::error::" in result.stdout
    if fault == "success":
        cargo = [item for item in map(json.loads, (tmp_path / "trace").read_text().splitlines()) if item["tool"] == "cargo"]
        assert [item["args"] for item in cargo] == [
            ["machete"], ["deny", "--all-features", "check", "bans", "licenses"], ["fmt", "--all", "--", "--check"],
            ["test", "--locked", "-p", "dynamo-parsers-v2", "--test", "structural_tag_xgrammar", "--", "--ignored", "--nocapture"],
            ["nextest", "run", "--workspace", "--all-targets", "--locked"],
            ["test", "--workspace", "--locked", "--test", "unified_parity", "structural_response_case_checks_the_schema_before_parsing"],
            ["test", "-p", "dynamo-multimodal", "--all-targets", "--all-features", "--locked"],
            ["test", "--locked", "-p", "dynamo-parsers-v2", "--features", "test-utils", "--test", "gemma4_native_recovery"],
            ["test", "--workspace", "--doc", "--locked"], ["test", "-p", "dynamo-protocols", "--features", "protocol-schema", "--locked"],
            ["clippy", "--workspace", "--all-targets", "--all-features", "--locked", "--", "-D", "warnings"],
            ["publish", "--workspace", "--dry-run", "--locked"],
        ]
        assert cargo[5]["optimized"] == "1"
        assert len({item["golden_root"] for item in cargo[3:]}) == 1
