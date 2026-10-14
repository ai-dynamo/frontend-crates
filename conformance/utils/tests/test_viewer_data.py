from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def test_lossless_values_and_unified_contracts():
    for filename in ("values.test.js", "unified.test.js", "comparison.test.js", "resources.test.js", "legacy.test.js", "folds.test.js", "toolcalling.test.js", "unified_panel.test.js", "reasoning_panel.test.js", "selection.test.js", "engine_selection.test.js"):
        result = subprocess.run(
            ["node", str(ROOT / "conformance" / "viewer" / filename)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr


def test_browser_data_primitives_and_tar_guards():
    script = r'''
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { gzipSync } from './conformance/viewer/vendor/fflate/esm/browser.js';
import { parseYaml, readTarGzip, sha256 } from './conformance/viewer/data.js';

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
const archiveBytes = new Uint8Array(await readFile('conformance/fixtures/toolcalling/fixtures-stream-v1/inputs.tar.gz'));
const member = 'toolcalling/fixtures-stream-v1/inputs/deepseek_v3/TOOLCALLING.streamv1.1.yaml';
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
