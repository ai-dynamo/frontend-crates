# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The ONE label a Dynamo capture is filed under.

A capture is stored as `dynamo_v2-<label>/` and stamped `captured_with:
{dynamo_v2: <label>}`. Every writer in the pipeline must agree on that string:
`refresh_dynamo_captures` creates the dir, `explode_unified_fixtures` re-derives
it to place exploded cases, and a mismatch produces a capture dir the table
cannot find. They used to compute it separately — regex vs tomllib, with
different failure modes — so this is the shared parent.

The label defaults to the live `parsers/v2` crate version, which gives RELEASE
granularity. Releases are explicit `chore: release` commits, so several merges
can share one version and a release can contain no parser change at all. To
compare a single change instead, override the label:

    CONFORMANCE_DYNAMO_V2_LABEL=0.1.24+pr163 ...
    refresh_dynamo_captures.py --label 0.1.24+pr163

That writes an ADDITIONAL `dynamo_v2-0.1.24+pr163/` dir beside the released one
rather than replacing it. Version dirs are capture HISTORY (see
`conformance/tests/common/mod.rs`) — never rewrite one; only add.
"""

import os
import re
from pathlib import Path

ENV_OVERRIDE = "CONFORMANCE_DYNAMO_V2_LABEL"

# `0.1.24`, and the PR/SHA-qualified forms that make a capture attributable to
# one change: `0.1.24+pr163`, `0.1.24+g06bc1f2`. Kept strict so a typo becomes a
# missing-column error at capture time, not a mystery dir nobody reads.
_LABEL_RE = re.compile(r"^\d+\.\d+\.\d+(?:[.\w-]*)?(?:\+[0-9A-Za-z._-]+)?$")


def crate_version(cargo_toml: Path) -> str:
    """The `version = "..."` from a Cargo.toml. Raises if absent — a capture filed
    under a guessed version is worse than a failed capture."""
    m = re.search(r'^version\s*=\s*"([^"]+)"', cargo_toml.read_text(), re.MULTILINE)
    if not m:
        raise ValueError(f"no [package] version in {cargo_toml}")
    return m.group(1)


def dynamo_v2_label(repo_root: Path, override: str | None = None) -> str:
    """Label to file this Dynamo v2 capture under.

    Precedence: explicit `override` (a `--label` flag) > `$CONFORMANCE_DYNAMO_V2_LABEL`
    > the live parsers/v2 crate version.
    """
    # An explicitly-supplied-but-blank override is an error, not a request for the
    # default: `--label ""` / `CONFORMANCE_DYNAMO_V2_LABEL=` means the caller meant
    # to name this capture and the name got lost. Falling back would silently file
    # it under the released version and overwrite that comparison point.
    for supplied in (override, os.environ.get(ENV_OVERRIDE)):
        if supplied is None:
            continue
        if not supplied.strip():
            raise ValueError("empty Dynamo v2 capture label; omit the override to use the crate version")
        label = supplied.strip()
        break
    else:
        label = crate_version(repo_root / "parsers" / "v2" / "Cargo.toml").strip()
    if not _LABEL_RE.match(label):
        raise ValueError(
            f"bad Dynamo v2 capture label {label!r}: expected <version>[+<tag>], "
            "e.g. 0.1.24 or 0.1.24+pr163"
        )
    return label
