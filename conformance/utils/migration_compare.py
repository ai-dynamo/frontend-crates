#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare complete JSON records, distinguishing absent fields from null values."""

import argparse
import json
from pathlib import Path


def differences(before, after, path=()):
    if type(before) is not type(after):
        yield {"path": list(path), "old": before, "new": after}
    elif isinstance(before, dict):
        for key in sorted(before.keys() | after.keys()):
            if key not in before:
                yield {"path": [*path, key], "new": after[key], "missing": "baseline"}
            elif key not in after:
                yield {"path": [*path, key], "old": before[key], "missing": "candidate"}
            else:
                yield from differences(before[key], after[key], (*path, key))
    elif isinstance(before, list):
        if len(before) != len(after):
            yield {"path": list(path), "old": before, "new": after}
        else:
            for index, (old, new) in enumerate(zip(before, after, strict=True)):
                yield from differences(old, new, (*path, index))
    elif before != after:
        yield {"path": list(path), "old": before, "new": after}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected", type=Path, help="Exact, separately justified field changes; no wildcards")
    args = parser.parse_args()
    before = json.loads(args.baseline.read_text())
    after = json.loads(args.candidate.read_text())
    changes = list(differences({k: before[k] for k in ("inventory", "records")},
                               {k: after[k] for k in ("inventory", "records")}))
    expected = json.loads(args.expected.read_text())["changes"] if args.expected else []
    unmatched = [change for change in changes if change not in expected]
    unused = [change for change in expected if change not in changes]
    result = {"changes": changes, "unexpected": unmatched, "unused_exceptions": unused,
              "baseline_records": len(before["records"]), "candidate_records": len(after["records"])}
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: len(v) if isinstance(v, list) else v for k, v in result.items()}))
    return bool(unmatched or unused)


if __name__ == "__main__":
    raise SystemExit(main())
