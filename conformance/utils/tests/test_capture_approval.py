# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[3]


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
def test_capture_approval_requires_current_head_administrator(path, permission, state, revision, allowed, truncated):
    script = r"""
const {check} = require('./.github/scripts/capture-approval.cjs');
const [path, permission, state, revision, truncated] = JSON.parse(process.argv[1]);
const rest = {pulls: {get: async () => ({data:{changed_files:truncated ? 50000 : 1,head:{sha:'a'}}}), listFiles:'files', listReviews:'reviews'},
  repos:{getCollaboratorPermissionLevel:async () => ({data:{permission}})}};
const github = {rest, paginate:async endpoint => endpoint === 'files' ? [{filename:path}] : [{user:{login:'reviewer'},state,commit_id:revision}]};
check({github,context:{repo:{owner:'test',repo:'test'},payload:{pull_request:{number:1}}},core:{info:()=>{}}})
  .catch(error => {console.error(error.message);process.exitCode=1;});
"""
    result = subprocess.run(["node", "-e", script, json.dumps([path, permission, state, revision, truncated])], cwd=ROOT, capture_output=True, text=True)
    authored = path.startswith("conformance/inputs/") or path.endswith("/inputs_and_golden.yaml")
    assert (result.returncode == 0) == (allowed and not (authored and truncated)), result.stderr
