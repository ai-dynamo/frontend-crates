# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import datetime
import json
import subprocess
import sys

import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "conformance/utils/src"))
import viewer_codec


def test_python_codec_decodes_to_lossless_javascript_values():
    value = {
        "values": [None, True, 1, 1.0, -0.0, 9007199254740993, float("nan"), float("inf"), float("-inf")],
        "mapping": {1: "integer", "1": "string", None: "null"},
        "date": datetime.date(2026, 10, 15),
        "time": datetime.datetime(2026, 10, 15, 1, 2, 3, 123456, datetime.timezone(datetime.timedelta(hours=1))),
        "binary": b"bytes",
    }
    script = """
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { decodeTyped } from './conformance/viewer/typed_json.js';
import { temporalText } from './conformance/viewer/values.js';
const value = decodeTyped(JSON.parse(readFileSync(0, 'utf8')));
assert.deepEqual(value.values, [null, true, 1n, 1, -0, 9007199254740993n, NaN, Infinity, -Infinity]);
assert.deepEqual(value.mapping, new Map([[1n, 'integer'], ['1', 'string'], [null, 'null']]));
assert.equal(temporalText(value.date), '2026-10-15');
assert.equal(temporalText(value.time), '2026-10-15T00:02:03.123456+00:00');
assert.deepEqual(value.binary, new TextEncoder().encode('bytes'));
assert.equal(Object.hasOwn(value, 'missing'), false);
"""
    subprocess.run(["node", "--input-type=module", "-e", script], input=json.dumps(viewer_codec.encode_typed(value)), text=True, cwd=ROOT, capture_output=True, check=True)


def test_typed_json_negative_controls():
    subprocess.run(["node", "conformance/viewer/typed_json.test.js"], cwd=ROOT, capture_output=True, check=True)


def test_selected_json_resources_and_integrity_retry():
    subprocess.run(["node", "conformance/viewer/json_resources.test.js"], cwd=ROOT, capture_output=True, check=True)


@pytest.mark.parametrize("value", [None, True, False, 12345678901234567890, -0.0, 1.5, float("inf"),
    "scalar", b"\x00\xff", datetime.date(2026, 10, 23),
    datetime.datetime(2026, 10, 23, tzinfo=datetime.timezone.utc),
    {"missing_vs_null": None, 3: [True, "3", {}]}])
def test_native_fragment_codec_roundtrip(value):
    assert viewer_codec.typed_equal(viewer_codec.decode_typed(viewer_codec.encode_typed(value)), value)


@pytest.mark.parametrize("value", [[], ["null", 0], ["bool", 1], ["str", None], ["int", "bad"],
    ["binary", "%%%"], ["unknown", 1], ["object", [[1]]],
    ["object", [[["str", "same"], ["null"]], [["str", "same"], ["null"]]]]])
def test_native_fragment_codec_rejects_invalid_values(value):
    with pytest.raises(ValueError):
        viewer_codec.decode_typed(value)
