# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

import migrate_unified_snapshots
from conformance.utils.tests.test_unified_history import _write_family, _write_capture, _change
import unified_archives
import unified_history


def test_archive_bytes_are_deterministic_and_members_are_readable(tmp_path: Path) -> None:
    member = "families/sample/dynamo_v2-1.2.3.yaml"
    payload = b"schema_version: 2\nrecords: {}\n"
    archives = []
    for name in ("first", "second"):
        root = tmp_path / name
        family = root / "families/sample"
        family.mkdir(parents=True)
        (family / "inputs_and_golden.yaml").write_text("family: sample\ncases: {}\n")
        (root / member).write_bytes(payload)
        unified_archives.pack(root, unified_history.CAPTURE_DIRECTORY_RE,
                              unified_history._capture_release_sort_key)
        archives.append(next((root / "versions").iterdir()))

    assert archives[0].read_bytes() == archives[1].read_bytes()
    assert list(unified_archives.members(archives[0])) == [(member, payload)]


@pytest.mark.parametrize('override', [None, '0.9.0', 'absent'])
def test_conversion_preserves_derived_inheritance_overrides(tmp_path, override):
    source = tmp_path / 'source'
    _write_family(source)
    _write_capture(source, '1.0.0', {'text_only': _change()})
    path = _write_capture(source, '1.1.0', {})
    document = unified_history.load_yaml(path)
    if override != 'absent':
        document['document_overrides'] = {'text_only': {'capture_provenance': {'inherited_from': override, 'captured_with': {'dynamo_v2': '1.1.0'}, 'capture_origin': None}}}
        path.write_text(unified_history.dump_yaml(document))
    sparse = unified_history.load_store(source).histories[('gemma4', 'dynamo_v2')]
    expected = sparse.resolve('dynamo_v2-1.1.0', view_capture_id='dynamo_v2-1.2.0')
    complete = tmp_path / 'complete'
    migrate_unified_snapshots.convert(source, complete, tmp_path)
    history = unified_history.load_store(complete).histories[('gemma4', 'dynamo_v2')]
    actual = history.resolve('dynamo_v2-1.1.0', view_capture_id='dynamo_v2-1.2.0')
    assert unified_history.typed_equal(actual, expected)
    assert ('text_only' in history.snapshot_suppressed_inheritance['dynamo_v2-1.1.0']) == (override is None)
    rewritten = unified_history._store_documents(unified_history.load_store(complete))
    converted = rewritten[history.capture_path('dynamo_v2-1.1.0')]
    if override is None:
        assert converted['records']['text_only']['suppress_inherited_from'] is True
        converted['records']['text_only']['suppress_inherited_from'] = False
        with pytest.raises(ValueError, match='invalid snapshot inheritance projection'):
            unified_history._validate_complete_snapshot(path, converted, history.family, 'dynamo_v2', '1.1.0')


def test_complete_snapshot_updates_keep_and_clear_projection_markers(tmp_path):
    source = tmp_path / 'source'
    _write_family(source)
    path = _write_capture(source, '1.0.0', {'text_only': _change()})
    document = unified_history.load_yaml(path)
    document['document_overrides'] = {'text_only': {'capture_provenance': {
        'captured_with': {'dynamo_v2': '1.0.0'}, 'capture_origin': None, 'inherited_from': None,
    }}}
    path.write_text(unified_history.dump_yaml(document))
    complete = tmp_path / 'complete'
    migrate_unified_snapshots.convert(source, complete, tmp_path)
    history = unified_history.load_store(complete).histories[('gemma4', 'dynamo_v2')]
    first = 'dynamo_v2-1.0.0'
    later = 'dynamo_v2-1.1.0'
    history.captures[later] = {**history.captures[first], 'runtime_version': '1.1.0'}
    history.capture_paths[later] = history.family.path.parent / (later + '.yaml')
    original = history.resolve(first)
    unified_history._set_capture_snapshot(history, later, original, original)
    assert history.snapshot_suppressed_inheritance[later] == {'text_only'}
    assert 'inherited_from' not in history.resolve(later, view_capture_id='dynamo_v2-1.2.0')['text_only']['document']
    changed = history.resolve(first)
    changed['text_only']['observation'] = {'value': {'assembled': [{'kind': 'text', 'text': 'new'}]}}
    unified_history._set_capture_snapshot(history, first, changed, {})
    assert history.snapshot_suppressed_inheritance[first] == set()
    unified_history._set_capture_snapshot(history, later, {}, original)
    assert history.snapshot_suppressed_inheritance[later] == set()
