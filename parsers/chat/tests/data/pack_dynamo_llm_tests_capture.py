# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pack a capture of dynamo-llm's test runs into one fixture.

Usage: pack_dynamo_llm_tests_capture.py OUT.jsonl.gz CAPTURE.jsonl

Keeps each distinct call once. A call's name is the test that made it, with
`#n` for its order among that test's distinct calls. Records are sorted by name.
"""

import collections
import gzip
import json
import sys

out_path, capture_path = sys.argv[1:3]
seen = set()
per_test = collections.Counter()
records = []
with open(capture_path) as f:
    for line in f:
        record = json.loads(line)
        key = json.dumps({k: v for k, v in record.items() if k != "name"}, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        per_test[record["name"]] += 1
        record["name"] = f"{record['name']}#{per_test[record['name']]}"
        records.append(record)
records.sort(key=lambda record: record["name"])
text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
with gzip.GzipFile(out_path, "wb", compresslevel=9, mtime=0) as f:
    f.write(text.encode())
