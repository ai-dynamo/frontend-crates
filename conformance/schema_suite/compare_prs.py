#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the frozen schema suite on PR merge bases and heads; compare observed results."""

import argparse
import collections
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
COMMAND = [
    "cargo",
    "test",
    "--locked",
    "-p",
    "dynamo-conformance-fixtures-v2",
    "--test",
    "schema_coercion_contract",
    "--",
    "--test-threads=4",
]
BAD = {"fail", "error", "not_run"}


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def git(*args, cwd=ROOT):
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    assert len(rows) == len(
        {(r["id"], r["surface"]) for r in rows}
    ), "Duplicate observations"
    return rows


def summarize(rows):
    counts = dict(collections.Counter(r["status"] for r in rows))
    groups = {}
    for row in rows:
        groups.setdefault(row["group"], []).append(row["status"])
    return {
        "counts": counts,
        "checks": sum(r.get("checks", 0) for r in rows),
        "failed_checks": sum(r.get("failed_checks", 0) for r in rows),
        "groups": {
            g: "fail" if any(s in BAD for s in statuses) else "pass"
            for g, statuses in groups.items()
        },
    }


def run(args, manifest):
    baseline = json.loads((args.baseline / "metadata.json").read_text())
    sources = {
        path: subprocess.check_output(
            ["git", "show", manifest["suite_commit"] + ":" + path], cwd=ROOT
        )
        for path in baseline["suite_files"]
    }
    assert all(
        hashlib.sha256(content).hexdigest() == baseline["suite_files"][path]
        for path, content in sources.items()
    )
    version = subprocess.check_output(
        [
            args.xgrammar_python,
            "-c",
            'import importlib.metadata; print(importlib.metadata.version("xgrammar"))',
        ],
        text=True,
    ).strip()
    assert version == "0.2.7", version
    if not args.worktree.exists():
        subprocess.run(
            [
                "git",
                "worktree",
                "add",
                "--detach",
                str(args.worktree),
                manifest["main_sha"],
            ],
            cwd=ROOT,
            check=True,
            stdout=subprocess.DEVNULL,
        )
    assert not git(
        "status", "--porcelain", "--untracked-files=no", cwd=args.worktree
    ), "Tracked edits in audit worktree"
    for path, content in sources.items():
        dest = args.worktree / path
        assert (
            subprocess.run(
                ["git", "ls-files", "--error-unmatch", path],
                cwd=args.worktree,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ).returncode
            != 0
        )
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)
    template_rows = read_rows(args.baseline / "results.jsonl")
    wanted = {(r["id"], r["surface"]): r for r in template_rows}
    revisions = {}
    # A shared merge base is executed once and reused by SHA, never by a similar result.
    for pr in manifest["prs"]:
        revisions.setdefault(pr["merge_base"], []).append(f'PR #{pr["number"]} before')
        revisions.setdefault(pr["headRefOid"], []).append(f'PR #{pr["number"]} after')
    campaign = {
        "started_utc": utc(),
        "suite_commit": manifest["suite_commit"],
        "suite_sha256": baseline["suite_sha256"],
        "suite_files": baseline["suite_files"],
        "main_sha": manifest["main_sha"],
        "xgrammar_version": version,
        "python": args.python,
        "xgrammar_python": args.xgrammar_python,
        "rustc": subprocess.check_output(["rustc", "--version"], text=True).strip(),
        "worktree": str(args.worktree),
        "method": "exact merge-base -> exact PR head; identical test overlay; shared bases deduplicated by SHA",
        "revision_count": len(revisions),
    }
    (args.output / "campaign.json").write_text(json.dumps(campaign, indent=2) + "\n")
    for index, (sha, labels) in enumerate(revisions.items(), 1):
        folder = args.output / "runs" / sha
        folder.mkdir(parents=True, exist_ok=True)
        saved = folder / "metadata.json"
        if saved.exists():
            old = json.loads(saved.read_text())
            if (
                old.get("complete")
                and old.get("suite_sha256") == baseline["suite_sha256"]
                and old.get("results_sha256") == digest(folder / "results.jsonl")
            ):
                print(
                    f"[{index}/{len(revisions)}] reuse {sha[:10]} {'; '.join(labels)}",
                    flush=True,
                )
                continue
        print(
            f"[{index}/{len(revisions)}] start {sha[:10]} {'; '.join(labels)} {utc()}",
            flush=True,
        )
        subprocess.run(
            ["git", "checkout", "--detach", sha],
            cwd=args.worktree,
            check=True,
            stdout=subprocess.DEVNULL,
        )
        assert git("rev-parse", "HEAD", cwd=args.worktree) == sha
        assert not git("diff", "HEAD", cwd=args.worktree)
        assert all(
            digest(args.worktree / p) == h for p, h in baseline["suite_files"].items()
        )
        result_path = folder / "results.jsonl"
        result_path.write_text("")
        env = {
            **os.environ,
            "CARGO_TARGET_DIR": str(args.target_dir),
            "SCHEMA_SUITE_PYTHON": args.python,
            "SCHEMA_SUITE_XGRAMMAR_PYTHON": args.xgrammar_python,
            "SCHEMA_SUITE_RESULTS": str(result_path),
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        env.pop("SCHEMA_SUITE_FAMILY", None)
        meta = {
            "sha": sha,
            "labels": labels,
            "commit_time": git("show", "-s", "--format=%cI", sha),
            "started_utc": utc(),
            "suite_sha256": baseline["suite_sha256"],
            "cargo_lock_sha256": digest(args.worktree / "Cargo.lock"),
            "command": COMMAND,
            "complete": False,
        }
        saved.write_text(json.dumps(meta, indent=2) + "\n")
        start = time.monotonic()
        with (folder / "contracts.log").open("w") as log:
            process = subprocess.Popen(
                COMMAND,
                cwd=args.worktree,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                status = process.wait(timeout=600)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                status = 124
        meta.update(
            ended_utc=utc(),
            seconds=round(time.monotonic() - start, 3),
            exit_code=status,
        )
        observed = read_rows(result_path)
        keys = {(r["id"], r["surface"]) for r in observed}
        meta["missing_observations"] = len(set(wanted) - keys)
        meta["unexpected_observations"] = sorted(keys - set(wanted))
        with result_path.open("a") as f:
            for key in sorted(set(wanted) - keys):
                row = {k: wanted[key][k] for k in ["id", "group", "family", "surface"]}
                row.update(
                    status="not_run", reason="No observation; inspect build/test log"
                )
                f.write(json.dumps(row) + "\n")
        rows = read_rows(result_path)
        meta.update(summarize(rows))
        log_text = (folder / "contracts.log").read_text()
        meta["test_summary"] = re.findall(r"test result:.*", log_text)
        meta["group_tests"] = dict(
            re.findall(r"^test ([a-h]\d\d) \.\.\. (ok|FAILED)$", log_text, re.M)
        )
        meta["run_status"] = (
            "complete"
            if len(meta["group_tests"]) == 110 and not meta["missing_observations"]
            else "timeout"
            if status == 124
            else "incomplete"
        )
        meta["production_diff_empty"] = not git("diff", "HEAD", cwd=args.worktree)
        assert meta["production_diff_empty"] and all(
            digest(args.worktree / p) == h for p, h in baseline["suite_files"].items()
        )
        meta["results_sha256"] = digest(result_path)
        meta["log_sha256"] = digest(folder / "contracts.log")
        meta["complete"] = True
        saved.write_text(json.dumps(meta, indent=2) + "\n")
        print(
            f"[{index}/{len(revisions)}] {meta['run_status']} {meta['seconds']}s {meta['counts']} {meta['test_summary']}",
            flush=True,
        )
    campaign["finished_utc"] = utc()
    (args.output / "campaign.json").write_text(json.dumps(campaign, indent=2) + "\n")


def relevant(row):
    """Exclude generated call IDs and timing from outcome comparisons."""
    if row is None:
        return None
    fields = [
        "status",
        "reason",
        "checks",
        "failed_checks",
        "failure_kind",
        "parser_roundtrip",
    ]
    value = {k: row[k] for k in fields if k in row}

    def clean(item):
        if isinstance(item, dict):
            return {k: clean(v) for k, v in item.items() if k != "ids"}
        if isinstance(item, list):
            return [clean(v) for v in item]
        if isinstance(item, str):
            return re.sub(r"^(RuntimeError: )\[\d{2}:\d{2}:\d{2}\]", r"\1[time]", item)
        return item

    # Passing-output differences are outside this pass/fail audit. For failures,
    # retain changed witnesses even if the number of failed partitions is constant.
    if row["status"] in BAD:
        for field in ["actual", "failures"]:
            if field in row:
                value[field] = clean(row[field])
    return clean(value)


def compare(args, manifest):
    prs = []
    for pr in manifest["prs"]:
        before_dir, after_dir = [
            args.output / "runs" / sha for sha in [pr["merge_base"], pr["headRefOid"]]
        ]
        before_meta, after_meta = [
            json.loads((d / "metadata.json").read_text())
            for d in [before_dir, after_dir]
        ]
        before, after = [
            {(r["id"], r["surface"]): r for r in read_rows(d / "results.jsonl")}
            for d in [before_dir, after_dir]
        ]
        changes = []
        comparable = before_meta["run_status"] == after_meta["run_status"] == "complete"
        for key in sorted(set(before) | set(after)):
            b, a = before.get(key), after.get(key)
            if relevant(b) == relevant(a):
                continue
            bs, ats = (
                (b or {}).get("status", "missing"),
                (a or {}).get("status", "missing"),
            )
            if bs in BAD and ats == "pass":
                kind = "fixed"
            elif bs == "pass" and ats in BAD:
                kind = "regressed"
            elif bs != ats:
                kind = (
                    "availability_changed"
                    if {bs, ats} & {"unavailable", "not_applicable", "missing"}
                    else "execution_changed"
                )
            elif bs in BAD and (b or {}).get("failed_checks", 0) > (a or {}).get(
                "failed_checks", 0
            ):
                kind = "partially_improved"
            elif bs in BAD and (b or {}).get("failed_checks", 0) < (a or {}).get(
                "failed_checks", 0
            ):
                kind = "worsened"
            elif bs in BAD:
                kind = "changed_failure"
            else:
                kind = "check_count_changed"
            changes.append(
                {
                    "id": key[0],
                    "surface": key[1],
                    "group": (a or b)["group"],
                    "family": (a or b)["family"],
                    "kind": kind,
                    "before": b,
                    "after": a,
                }
            )
        group_changes = []
        for group in sorted(
            set(before_meta["group_tests"]) | set(after_meta["group_tests"])
        ):
            b, a = (
                before_meta["group_tests"].get(group, "not_run"),
                after_meta["group_tests"].get(group, "not_run"),
            )
            if b != a:
                group_changes.append({"group": group.upper(), "before": b, "after": a})
        delta = {
            status: after_meta["counts"].get(status, 0)
            - before_meta["counts"].get(status, 0)
            for status in [
                "pass",
                "fail",
                "error",
                "not_run",
                "unavailable",
                "not_applicable",
            ]
        }
        limitations = []
        for key in sorted(set(before) & set(after)):
            b, a = before[key], after[key]
            if b["status"] == a["status"] == "error" and b.get("reason") == a.get(
                "reason"
            ):
                limitations.append(
                    f"{key[0]}: {b.get('reason')}; unchanged harness error, not introduced by this PR."
                )
            if (
                b["group"] == "H05"
                and b.get("expected_supported")
                and not b.get("actual_supported")
                and a.get("actual_supported") == b.get("actual_supported")
            ):
                limitations.append(
                    f"{b['family']} / {b['checked_surface']} is absent on both revisions although the latest-main suite inventory expects it."
                )
        prs.append(
            {
                **pr,
                "baseline_limitations": limitations,
                "comparable": comparable,
                "before": before_meta,
                "after": after_meta,
                "delta": delta,
                "change_counts": dict(collections.Counter(c["kind"] for c in changes)),
                "group_changes": group_changes,
                "changes": changes,
            }
        )
    result = {
        "manifest": manifest,
        "campaign": json.loads((args.output / "campaign.json").read_text()),
        "prs": prs,
    }
    (args.output / "comparisons.json").write_text(
        json.dumps(result, ensure_ascii=False) + "\n"
    )
    summary = [
        {
            "pr": p["number"],
            "author": p["author"]["login"],
            "comparable": p["comparable"],
            "delta": p["delta"],
            "changes": p["change_counts"],
            "group_changes": p["group_changes"],
        }
        for p in prs
    ]
    (args.output / "comparison-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["run", "compare"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--python", default="/usr/bin/python3")
    parser.add_argument("--xgrammar-python", required=True)
    args = parser.parse_args()
    manifest = json.loads((args.output / "manifest.json").read_text())
    if args.mode == "run":
        run(args, manifest)
    compare(args, manifest)


if __name__ == "__main__":
    main()
