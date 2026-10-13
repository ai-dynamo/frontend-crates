from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def test_shared_model_preserves_versions_history_and_classifications():
    script = r'''
import assert from 'node:assert/strict';
import { classifyUnified, compareVersions, resolveUnified, sameValue, unifiedEvents } from './conformance/viewer/model.js';

assert.equal(compareVersions('0.7.9', '0.7.10'), -1);
assert.equal(compareVersions('0.7.10.patch1', '0.7.10'), 1);
assert.equal(sameValue(null, 'null'), false);
assert.equal(sameValue(1, '1'), false);
const captures = [
  { version: '0.1.0', implementation: 'dynamo_v2', path: 'first.yaml', bytes: new Uint8Array([1]), document: {
    provenance: { status: 'captured', origin: { crate_version: '0.1.0', source_sha256: 'a'.repeat(64) } },
    document: { mode: 'unified' },
    changes: { case_a: { observation: { value: { assembled: [{ kind: 'text', text: 'old' }] } } }, case_removed: { observation: { value: { assembled: [] } } } },
    metadata_changes: { case_a: { source: 'old' } },
    document_overrides: { case_a: { parser_path: 'old/parser' } },
  } },
  { version: '0.2.0', implementation: 'dynamo_v2', path: 'second.yaml', bytes: new Uint8Array([2]), document: {
    provenance: { status: 'captured', origin: { crate_version: '0.2.0', source_sha256: 'b'.repeat(64) } },
    document: { mode: 'unified' },
    changes: { case_a: { observation: { value: { chunks: [[{ kind: 'text', text: 'new' }]] } } }, case_removed: { absent: true } },
    metadata_changes: { case_a: { source: 'new' } },
    document_overrides: { case_a: { parser_path: 'new/parser', capture_provenance: { captured_with: { dynamo_v2: '0.2.0' }, capture_origin: null } } },
  } },
  { version: '0.3.0', implementation: 'dynamo_v2', path: 'third.yaml', bytes: new Uint8Array([3]), document: {
    provenance: { status: 'captured', origin: { crate_version: '0.3.0', source_sha256: 'c'.repeat(64) } },
    document: { mode: 'unified' }, changes: {}, metadata_changes: {}, document_overrides: {},
  } },
];
const resolved = resolveUnified(captures, '0.3.0');
assert.equal(resolved.has('case_removed'), false);
const inherited = resolved.get('case_a');
assert.deepEqual(unifiedEvents(inherited.value.observation.value), [{ kind: 'text', text: 'new' }]);
assert.equal(inherited.value.document.inherited_from, '0.2.0');
assert.equal(inherited.value.document.parser_path, 'new/parser');
assert.equal(inherited.value.document.record_metadata.source, 'new');
assert.equal(Object.hasOwn(inherited.value.document, 'capture_origin'), false);

const markers = { common: ['<control>'], families: { sample: ['<tool>'] } };
assert.equal(classifyUnified('sample', [{ kind: 'text', text: 'x' }], [{ kind: 'text', text: 'x' }], markers), 'MATCH');
assert.equal(classifyUnified('sample', [{ kind: 'reasoning', text: 'x' }], [{ kind: 'text', text: 'x' }], markers), 'MERGE');
assert.equal(classifyUnified('sample', [{ kind: 'text', text: '' }], [{ kind: 'text', text: '<tool>' }], markers), 'LEAK');
assert.equal(classifyUnified('sample', [{ kind: 'tool_call', name: 'f', arguments: { x: null } }], [{ kind: 'tool_call', name: 'f', arguments: { x: 'null' } }], markers), 'ARG_MISMATCH');
assert.equal(classifyUnified('sample', [{ kind: 'reasoning', text: 'x' }, { kind: 'text', text: 'y' }], [{ kind: 'text', text: 'y' }, { kind: 'reasoning', text: 'x' }], markers), 'ORDER');
console.log('viewer model semantics passed');
'''
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "viewer model semantics passed"
