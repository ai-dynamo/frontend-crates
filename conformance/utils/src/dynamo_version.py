# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The ONE label a Dynamo capture is filed under.

A capture is stored as `dynamo_v2-<version>/` and stamped `captured_with:
{dynamo_v2: <version>}`. Every writer in the pipeline must agree on that string:
`refresh_dynamo_captures` creates the dir, `explode_unified_fixtures` re-derives
it to place exploded cases, and a mismatch produces a capture dir the table
cannot find. They used to compute it separately — regex vs tomllib, with
different failure modes — so this is the shared parent.

The label defaults to the live `parsers/v2` crate version, which gives RELEASE
granularity. Releases are explicit `chore: release` commits, so several merges
can share one version and a release can contain no parser change at all. Do not
override it with a branch, pull-request, or commit-qualified label. Unreleased
captures stay in the working tree until the parser is published. Version dirs
are capture HISTORY (see `conformance/tests/common/mod.rs`) — never rewrite one;
only add a capture for the published version that produced it.
"""

import os
import re
from pathlib import Path

ENV_OVERRIDE = "CONFORMANCE_DYNAMO_V2_LABEL"

# Only published SemVer-like crate versions are valid capture labels. A
# change-qualified `+tag` label would make an unreleased parser look like a
# release column in the Unified table.
_LABEL_RE = re.compile(r"^\d+\.\d+\.\d+(?:[.\w-]*)?$")


def crate_version(cargo_toml: Path) -> str:
    """The `version = "..."` from a Cargo.toml. Raises if absent — a capture filed
    under a guessed version is worse than a failed capture."""
    m = re.search(r'^version\s*=\s*"([^"]+)"', cargo_toml.read_text(), re.MULTILINE)
    if not m:
        raise ValueError(f"no [package] version in {cargo_toml}")
    return m.group(1)


def dynamo_v2_label(repo_root: Path, override: str | None = None) -> str:
    """Label to file this Dynamo v2 capture under.

    The explicit and environment overrides are accepted only when they match the live
    parsers/v2 crate version, so a capture cannot misattribute one build as another.
    """
    # An explicitly-supplied-but-blank override is an error, not a request for the
    # default: `--label ""` / `CONFORMANCE_DYNAMO_V2_LABEL=` means the caller meant
    # to name this capture and the name got lost. Falling back would silently file
    # it under the released version and overwrite that comparison point.
    version = crate_version(repo_root / "parsers" / "v2" / "Cargo.toml").strip()
    for supplied in (override, os.environ.get(ENV_OVERRIDE)):
        if supplied is None:
            continue
        if not supplied.strip():
            raise ValueError("empty Dynamo v2 capture label; omit the override to use the crate version")
        label = supplied.strip()
        if label != version:
            raise ValueError(f"capture label {label!r} does not match crate version {version!r}")
        break
    else:
        label = version
    if not _LABEL_RE.match(label):
        raise ValueError(
            f"bad Dynamo v2 capture label {label!r}: expected a published crate version, "
            "e.g. 0.1.24 or 0.5.4"
        )
    return label
