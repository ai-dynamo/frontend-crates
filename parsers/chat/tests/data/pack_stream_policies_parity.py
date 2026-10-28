# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pack the three recorder runs into one fixture.

Usage: pack_stream_policies_parity.py OUT.jsonl.gz AUTO.jsonl V1.jsonl V2.jsonl

The fixture holds every record from the run with DYN_PARSER_VERSION unset,
then each record from the DYN_PARSER_VERSION=1 and =2 runs that differs from
the record of the same kind and name in the first run.
"""

import gzip
import json
import sys


def load(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def without_version(record):
    return {key: value for key, value in record.items() if key != "version"}


out_path, auto_path, *other_paths = sys.argv[1:]
auto = load(auto_path)
by_key = {(r["kind"], r["name"]): without_version(r) for r in auto}
records = list(auto)
for path in other_paths:
    for record in load(path):
        if by_key.get((record["kind"], record["name"])) != without_version(record):
            records.append(record)
text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
with gzip.GzipFile(out_path, "wb", compresslevel=9, mtime=0) as f:
    f.write(text.encode())
