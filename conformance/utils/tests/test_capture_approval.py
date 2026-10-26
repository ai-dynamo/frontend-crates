# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import os
from pathlib import Path
import subprocess

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def approval_workflow():
    return yaml.load((ROOT / ".github/workflows/capture-approval.yml").read_text(), Loader=yaml.BaseLoader)


@pytest.mark.parametrize("path,permission,state,revision,allowed", [
    ("conformance/captures/unified/sample/dynamo_v2.yaml", "admin", "APPROVED", "a", True),
    ("conformance/captures/unified/sample/dynamo_v2.yaml", "maintain", "APPROVED", "a", False),
    ("conformance/historic/original.yaml", "admin", "APPROVED", "old", False),
    ("conformance/historic/original.yaml", "admin", "DISMISSED", "a", False),
    ("conformance/historic/original.yaml", "admin", "CHANGES_REQUESTED", "a", False),
    ("conformance/fixtures-unified-v2/families/sample/inputs_and_golden.yaml", "read", "COMMENTED", "old", True),
    ("conformance/inputs/stream/sample/case.yaml", "read", "COMMENTED", "old", True),
])
@pytest.mark.parametrize("truncated", [False, True])
def test_capture_approval_requires_current_head_administrator(path, permission, state, revision, allowed, truncated, approval_workflow, tmp_path):
    source = next(step["with"]["script"] for step in approval_workflow["jobs"]["capture-admin-approval"]["steps"] if step["uses"].startswith("actions/github-script@"))
    script = r"""
const [path, permission, state, revision, truncated, source] = JSON.parse(process.argv[1]);
const rest = {pulls: {get: async () => ({data:{changed_files:truncated ? 50000 : 1,head:{sha:'a'}}}), listFiles:'files', listReviews:'reviews'},
  repos:{getCollaboratorPermissionLevel:async () => ({data:{permission}})}};
const github = {rest, paginate:async endpoint => endpoint === 'files' ? [{filename:path}] : [{user:{login:'reviewer'},state,commit_id:revision}]};
const AsyncFunction = Object.getPrototypeOf(async function(){}).constructor;
new AsyncFunction('github', 'context', 'core', 'require', source)(github,
  {repo:{owner:'test',repo:'test'},payload:{pull_request:{number:1}}},{info:()=>{}},require)
  .catch(error => {console.error(error.message);process.exitCode=1;});
"""
    result = subprocess.run(["node", "-e", script, json.dumps([path, permission, state, revision, truncated, source])], cwd=tmp_path, env={"PATH": os.environ["PATH"]}, capture_output=True, text=True)
    authored = path.startswith("conformance/inputs/") or path.endswith("/inputs_and_golden.yaml")
    assert (result.returncode == 0) == (allowed and not (authored and truncated)), result.stderr
    if result.returncode:
        expected = "Incomplete PR file inventory" if authored and truncated else "Recorded captures/history require a repository administrator approval of current head a."
        assert expected in result.stderr


def test_capture_approval_uses_read_only_events_without_checkout(approval_workflow):
    assert set(approval_workflow["on"]) == {"pull_request", "pull_request_review"}
    assert "synchronize" in approval_workflow["on"]["pull_request"]["types"]
    assert approval_workflow["permissions"] == {"contents": "read", "pull-requests": "read"}
    steps = approval_workflow["jobs"]["capture-admin-approval"]["steps"]
    assert all(not step["uses"].startswith("actions/checkout@") for step in steps)
