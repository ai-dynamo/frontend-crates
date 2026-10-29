# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
import subprocess
from pathlib import Path
import pytest
from conformance.utils.tests.fixture_test_support import run_node_contract
import datetime
import json
import sys
import viewer_codec


ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("filename", (
    "values.test.js", "data.test.js", "resources.test.js", "model.test.js",
    "presentation.test.js", "startup.test.js",
))
def test_lossless_values_and_unified_contracts(source_catalog, filename):
    run_node_contract(filename)


def test_browser_data_primitives_and_tar_guards():
    script = r'''
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { gzipSync } from './conformance/viewer/vendor/fflate/esm/browser.js';
import { readTarGzip, sha256 } from './conformance/viewer/data.js';
import { parseYaml } from './conformance/viewer/yaml_data.js';

const enc = new TextEncoder();
function octal(block, offset, length, value) {
  const text = value.toString(8).padStart(length - 1, '0') + '\0';
  block.set(enc.encode(text), offset);
}
function header(name, data, type = '0') {
  const block = new Uint8Array(512);
  block.set(enc.encode(name), 0);
  octal(block, 100, 8, 0o644);
  octal(block, 108, 8, 0);
  octal(block, 116, 8, 0);
  octal(block, 124, 12, data.length);
  octal(block, 136, 12, 0);
  block.fill(32, 148, 156);
  block[156] = type.charCodeAt(0);
  block.set(enc.encode('ustar\0'), 257);
  block.set(enc.encode('00'), 263);
  let sum = 0;
  for (const byte of block) sum += byte;
  const checksum = sum.toString(8).padStart(6, '0') + '\0 ';
  block.set(enc.encode(checksum), 148);
  return block;
}
function entry(name, data, type = '0') {
  const bytes = data instanceof Uint8Array ? data : enc.encode(data);
  const body = new Uint8Array(Math.ceil(bytes.length / 512) * 512);
  body.set(bytes);
  return [header(name, bytes, type), body];
}
function tar(...parts) {
  const all = new Uint8Array(parts.flat().reduce((n, item) => n + item.length, 1024));
  let offset = 0;
  for (const item of parts.flat()) { all.set(item, offset); offset += item.length; }
  return gzipSync(all);
}
function paxPath(name) {
  let length = 0;
  do { length = `${length} path=${name}\n`.length; } while (length !== `${length} path=${name}\n`.length);
  const text = `${length} path=${name}\n`;
  return enc.encode(text);
}

const yaml = parseYaml('values: [null, "null", 1, "1", false]\n');
assert.deepEqual(yaml.values, [null, 'null', 1n, '1', false]);
assert.throws(() => parseYaml('value: 1\nvalue: 2\n', { unified: true }), /duplicate|unique/i);
assert.equal(await sha256(enc.encode('abc')), 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad');
const savedCrypto = globalThis.crypto;
Object.defineProperty(globalThis, 'crypto', { configurable: true, value: undefined });
assert.equal(await sha256(enc.encode('abc')), 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad');
Object.defineProperty(globalThis, 'crypto', { configurable: true, value: savedCrypto });

const longPath = `family/${'long-name-'.repeat(20)}.yaml`;
const pax = entry('PaxHeaders/item', paxPath(longPath), 'x');
const paxArchive = readTarGzip(tar(pax, entry('placeholder', 'value: 3\n')), { selectedPaths: [longPath] });
assert.equal(parseYaml(new TextDecoder().decode(paxArchive.get(longPath))).value, 3n);
const gnuName = `${longPath}.gnu`;
const gnuArchive = readTarGzip(tar(entry('././@LongLink', enc.encode(`${gnuName}\0`), 'L'), entry('short', 'ok\n')));
assert.equal(gnuArchive.has(gnuName), true);
assert.throws(() => readTarGzip(tar(entry('../escape.yaml', 'x'))), /unsafe/i);
assert.throws(() => readTarGzip(tar(entry('link.yaml', 'target', '2'))), /unsupported/i);
const corrupt = tar(entry('bad.yaml', 'x'));
const raw = new Uint8Array((await import('./conformance/viewer/vendor/fflate/esm/browser.js')).gunzipSync(corrupt));
raw[0] ^= 1;
assert.throws(() => readTarGzip(gzipSync(raw)), /checksum/i);
const pointer = enc.encode('version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 1\n');
assert.throws(() => readTarGzip(pointer), /LFS/i);
const member = 'toolcalling/fixtures-stream-v1/inputs/deepseek_v3/TOOLCALLING.streamv1.1.yaml';
const retained = readTarGzip(await readFile('conformance/fixtures/toolcalling/fixtures-stream-v1/inputs.tar.gz'), { selectedPaths: [member] });
const authored = retained.get(member);
assert.ok(authored?.length, 'retained archive must contain the original authored input');
const archiveBytes = tar(entry(member, authored), entry('unselected.yaml', 'value: 1\n'));
assert.equal(readTarGzip(archiveBytes, { selectedPaths: [member] }).size, 1);
console.log('viewer data primitives passed');
'''
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "viewer data primitives passed"


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "conformance/utils/src"))


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
import { decodeTyped } from './conformance/viewer/values.js';
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
