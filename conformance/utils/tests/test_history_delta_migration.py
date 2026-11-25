# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Historical delta storage retains independent executions and source lineage."""
import copy
from contextlib import contextmanager
import hashlib
import tarfile
from pathlib import Path

import pytest
import yaml

import capture_sources
import migrate_history_deltas
import normalized_yaml
import unified_history as history
from test_unified_history import _write_family, _write_capture, _change, _request


@pytest.mark.parametrize('later', [False, True])
@pytest.mark.parametrize('implementation', ['dynamo_v1', 'dynamo_v2'])
def test_preserve_current_without_versioned_checkpoint(tmp_path, later, implementation):
    root = tmp_path / capture_sources.STORE
    _write_family(root)
    old_path = _write_capture(root, '10.2.2', {'text_only': _change()}, implementation=implementation)
    original_store = history.load_store(root)
    original_owner = original_store.histories['gemma4', implementation]
    snapshot = original_owner.complete_snapshot(f'{implementation}-10.2.2')
    snapshot['records']['text_only']['value']['document']['record_metadata'] = {}
    row = {'surface': 'unified', 'mode': 'unified', 'family': 'gemma4',
           'implementation': implementation, 'version': '10.2.2',
           'source': f'{capture_sources.STORE}/families/gemma4/{implementation}-10.2.2.yaml'}
    current = root / f'families/gemma4/{implementation}.yaml'
    old_path.unlink()
    if later:
        _write_capture(root, '11.0.0', {'text_only': _change('new release')}, implementation=implementation)
    store = history._load_store_unlocked(root, allow_empty_history=True)
    current.write_bytes(capture_sources.selected_bytes(history.dump_yaml(snapshot).encode(), row, [],
        authored=original_store.families['gemma4'].document))
    snapshot = history.load_yaml(current)
    before = migrate_history_deltas.evidence(store)
    assert history.preserve_current_snapshots(tmp_path, store, implementations=(implementation,))
    owner = store.histories['gemma4', implementation]
    assert all(migrate_history_deltas.evidence(store)[key] == value for key, value in before.items())
    old = owner.captures[f'{implementation}-10.2.2']
    archived = next(item for item in old['historical_evidence'] if 'source_snapshot' in item)
    assert history.resolve_historical_evidence(owner, f'{implementation}-10.2.2', archived)['records'] == snapshot['records']
    assert not history.preserve_current_snapshots(tmp_path, store, implementations=(implementation,))

    if implementation == 'dynamo_v1':
        migrated = migrate_history_deltas.migrate(tmp_path, tmp_path / 'migration')
        assert len(migrated['current_evidence']) == 1
        assert migrated['checkpoints'] == 1 + int(later)


@pytest.mark.parametrize('difference', ['same', 'observation', 'request', 'metadata', 'source', 'empty_metadata', 'null_parser_path'])
@pytest.mark.parametrize('normalized', [False, True])
def test_refresh_archives_displaced_current_evidence_without_changing_history(tmp_path, difference, normalized):
    root = tmp_path / capture_sources.STORE
    _write_family(root)
    for version in ('11.0.0', '11.0.1'):
        _write_capture(root, version, {'text_only': _change()}, provenance={
            'status': 'captured', 'origin': {'crate_version': version, 'source_sha256': '1' * 64}})
    store = history.load_store(root)
    measured = store.histories['gemma4', 'dynamo_v2']
    before = dict(measured.iter_resolved_snapshots())
    snapshot = measured.complete_snapshot('dynamo_v2-11.0.0')
    value = snapshot['records']['text_only']['value']
    if difference == 'observation':
        value['observation'] = _change('old current result')['observation']
    elif difference == 'request':
        request = _request('old current request')
        value['stimulus'] = {'inline': request, 'semantic_sha256': history._request_digest(request)}
    elif difference == 'metadata':
        value['document']['record_metadata'] = {'old_current_note': {'empty': [], 'flag': False}}
    elif difference == 'source':
        value['document']['capture_origin']['source_sha256'] = '2' * 64
    elif difference == 'empty_metadata':
        value['document']['record_metadata'] = {}
    elif difference == 'null_parser_path':
        value['document']['parser_path'] = None
    logical = f'{capture_sources.STORE}/families/gemma4/dynamo_v2-11.0.0.yaml'
    row = {'surface': 'unified', 'mode': 'unified', 'family': 'gemma4',
           'implementation': 'dynamo_v2', 'version': '11.0.0', 'source': logical}
    current = root / 'families/gemma4/dynamo_v2.yaml'
    current.write_bytes(capture_sources.selected_bytes(history.dump_yaml(snapshot).encode(), row, [],
        authored=store.families['gemma4'].document))
    original_bytes = current.read_bytes()
    original = history.load_yaml(current)
    if normalized:
        family = store.families['gemma4']
        family.path.write_text(history.dump_yaml(normalized_yaml.encode_family(family.document)))
    history.refresh_current_snapshots(tmp_path, store, implementations=('dynamo_v2',))
    assert history.load_yaml(current)['records']
    if normalized:
        assert yaml.safe_load(store.families['gemma4'].path.read_text())['schema_version'] == 3
    assert dict(measured.iter_resolved_snapshots()) == before
    reloaded = history._load_selected_original_store(root, {('gemma4', 'dynamo_v2'): {'11.0.0'}},
        source_data={path.relative_to(root).as_posix(): path.read_bytes() for path in root.rglob('*.yaml')})
    retained = reloaded.histories['gemma4', 'dynamo_v2']
    archived = [item for item in retained.captures['dynamo_v2-11.0.0'].get('historical_evidence', [])
                if 'source_snapshot' in item]
    assert len(archived) == (0 if difference == 'same' else 1)
    if archived:
        assert archived[0]['source_snapshot']['sha256'] == hashlib.sha256(original_bytes).hexdigest()
        restored = history.resolve_historical_evidence(retained, 'dynamo_v2-11.0.0', archived[0])
        assert restored['records'] == original['records']
        assert bool(archived[0]['changes'].get('text_only', {}).get('observation')) == (difference == 'observation')
        malformed = copy.deepcopy(archived[0])
        malformed['checkpoint']['ignored_unrecognized_field'] = True
        with pytest.raises(ValueError, match='checkpoint'):
            history._validate_historical_evidence({'historical_evidence': [malformed]}, retained.family,
                'dynamo_v2', '11.0.0', 'mutation', records=retained.resolve('dynamo_v2-11.0.0'))
    assert history.refresh_current_snapshots(tmp_path, store, implementations=('dynamo_v2',)) == []


@pytest.mark.parametrize('difference', ['none', 'observation', 'request', 'removed', 'error', 'unsupported', 'member'])
def test_complete_history_uses_delta_writer_without_changing_evidence(tmp_path, difference):
    root = tmp_path / 'store'
    _write_family(root)
    _write_capture(root, '11.0.0', {'text_only': _change()})
    _write_capture(root, '11.0.1', {})
    store = history.load_store(root)
    original = store.histories['gemma4', 'dynamo_v2']
    checkpoints = list(original.iter_resolved_snapshots(include_members=True))
    for identity, records in checkpoints:
        if identity.endswith('11.0.1'):
            if difference == 'observation':
                records['text_only']['observation']['value']['assembled'][0]['text'] = 'changed'
            elif difference == 'request':
                records['text_only']['stimulus'] = {'inline': _request('changed'), 'semantic_sha256': history._request_digest(_request('changed'))}
            elif difference == 'removed':
                records.clear()
            elif difference in {'error', 'unsupported'}:
                records['text_only']['observation'] = {'error': {'code': 'parser_error', 'detail': 'bad input'}} if difference == 'error' else {'unavailable': {'code': 'unsupported_request_mode', 'detail': 'unsupported initialization'}}
        snapshot = original.complete_snapshot(identity, records=records)
        if difference == 'member' and identity.endswith('11.0.1'):
            snapshot['records']['text_only']['member'] = original._member(identity)
        original.capture_path(identity).write_text(history.dump_yaml(snapshot))
    before = history.load_store(root)
    expected = {key: list(value.iter_resolved_snapshots(include_members=True)) for key, value in before.histories.items()}
    expected_members = copy.deepcopy(before.histories['gemma4', 'dynamo_v2'].snapshot_members)
    documents = history._store_documents(before)
    later_path = root / 'families/gemma4/dynamo_v2-11.0.1.yaml'
    later = documents[later_path]
    assert 'records' not in later
    if difference in {'none', 'member'}:
        assert later['changes'] == {}
    if difference == 'request':
        assert set(later['changes']['text_only']) == {'stimulus'}
    if difference == 'removed':
        assert later['changes'] == {'text_only': {'absent': True}}
    for path, document in documents.items():
        path.write_text(history.dump_yaml(document))
    after = history.load_store(root)
    assert {key: list(value.iter_resolved_snapshots(include_members=True)) for key, value in after.histories.items()} == expected
    assert after.histories['gemma4', 'dynamo_v2'].snapshot_members == expected_members
    encoded = {path: path.read_bytes() for path in root.rglob('*.yaml')}
    history.rewrite_store(root)
    assert {path: path.read_bytes() for path in root.rglob('*.yaml')} == encoded


@pytest.mark.parametrize('member', [
    'conformance/fixtures-unified-v2/families/other/dynamo_v2-11.0.0.yaml',
    'conformance/fixtures-unified-v2/families/gemma4/dynamo_v1-11.0.0.yaml',
    'conformance/fixtures-unified-v2/families/gemma4/dynamo_v2-11.0.2.yaml',
    'conformance/fixtures-unified-v2/families/gemma4/dynamo_v2-10.0.0.yaml',
])
def test_member_delta_rejects_foreign_future_and_missing_sources(tmp_path, member):
    _write_family(tmp_path)
    path = _write_capture(tmp_path, '11.0.0', {'text_only': _change()})
    document = yaml.safe_load(path.read_text())
    document['member_changes'] = {'text_only': member}
    path.write_text(history.dump_yaml(document))
    with pytest.raises(ValueError, match='source member'):
        history.load_store(tmp_path)


def test_refresh_current_preserves_history_and_needs_no_historical_reads(tmp_path, monkeypatch):
    root = tmp_path / capture_sources.STORE
    _write_family(root)
    _write_capture(root, '11.0.0', {'text_only': _change()}, provenance={
        'status': 'captured', 'origin': {'crate_version': '11.0.0', 'source_sha256': '1' * 64}})
    store = history.load_store(root)
    before = {path: path.read_bytes() for path in root.rglob('*.yaml')}
    changed = history.refresh_current_snapshots(tmp_path, store, implementations=('dynamo_v2',))
    assert changed == [root / 'families/gemma4/dynamo_v2.yaml']
    assert all(path.read_bytes() == data for path, data in before.items())
    assert history.refresh_current_snapshots(tmp_path, store, implementations=('dynamo_v2',)) == []
    original_read = Path.read_bytes

    def deny_historical(path):
        if path.name.startswith('dynamo_v2-'):
            raise AssertionError('latest reader opened historical data')
        return original_read(path)

    monkeypatch.setattr(Path, 'read_bytes', deny_historical)
    selected = history.load_store(root)
    assert len(selected.histories) == 1
    assert selected.histories['gemma4', 'dynamo_v2'].resolve('dynamo_v2-11.0.0')['text_only']['observation'] == _change()['observation']


@pytest.mark.parametrize('suppressed', [False, True])
def test_complete_projection_flags_survive_delta_conversion(tmp_path, suppressed):
    _write_family(tmp_path)
    _write_capture(tmp_path, '11.0.0', {'text_only': _change()})
    _write_capture(tmp_path, '11.0.1', {})
    original = history.load_store(tmp_path).histories['gemma4', 'dynamo_v2']
    checkpoints = list(original.iter_resolved_snapshots(include_members=True))
    for identity, records in checkpoints:
        records['text_only']['document'].pop('inherited_from', None)
        snapshot = original.complete_snapshot(identity, records=records)
        snapshot['records']['text_only']['member'] = original._member(identity)
        if suppressed:
            snapshot['records']['text_only']['suppress_inherited_from'] = True
        original.capture_path(identity).write_text(history.dump_yaml(snapshot))
    store = history.load_store(tmp_path)
    selected = 'dynamo_v2-11.0.2'
    expected = store.histories['gemma4', 'dynamo_v2'].resolve('dynamo_v2-11.0.1', view_capture_id=selected)
    documents = history._store_documents(store)
    for path, document in documents.items():
        path.write_text(history.dump_yaml(document))
    actual = history.load_store(tmp_path).histories['gemma4', 'dynamo_v2'].resolve('dynamo_v2-11.0.1', view_capture_id=selected)
    assert actual == expected


def test_migration_keeps_archived_releases_and_latest_only_families(tmp_path):
    import_root = tmp_path / 'repo'
    root = import_root / capture_sources.STORE
    _write_family(root)
    _write_capture(root, '11.0.0', {'text_only': _change()})
    store = history.load_store(root)
    owner = store.histories['gemma4', 'dynamo_v2']
    identity = owner.ordered_capture_ids()[0]
    owner.capture_path(identity).write_text(history.dump_yaml(owner.complete_snapshot(identity)))
    archived_family = root / 'families/gemma4'
    latest_family = root / 'families/latest_only'
    latest_family.mkdir()
    authored = yaml.safe_load((archived_family / history.FAMILY_DOCUMENT_NAME).read_text())
    authored['input_document']['family'] = 'latest_only'
    authored['golden_document']['family'] = 'latest_only'
    (latest_family / history.FAMILY_DOCUMENT_NAME).write_text(history.dump_yaml(authored))
    history.archive_versions(root)
    result = migrate_history_deltas.migrate(import_root, tmp_path / 'migration')
    assert result['checkpoints'] == 1
    migrated = tmp_path / 'migration/candidate' / capture_sources.STORE
    assert (migrated / 'families/latest_only' / history.FAMILY_DOCUMENT_NAME).read_bytes() == (latest_family / history.FAMILY_DOCUMENT_NAME).read_bytes()
    assert len(history.load_store(migrated).histories['gemma4', 'dynamo_v2'].captures) == 1


def test_migration_rejects_live_drift_after_frozen_backup(tmp_path, monkeypatch):
    import_root = tmp_path / 'repo'
    root = import_root / capture_sources.STORE
    _write_family(root)
    capture = _write_capture(root, '11.0.0', {'text_only': _change()})
    original_open = tarfile.open

    @contextmanager
    def alter_after_backup(path, mode='r', **kwargs):
        with original_open(path, mode, **kwargs) as archive:
            yield archive
        if mode == 'w':
            document = yaml.safe_load(capture.read_text())
            document['changes']['text_only']['observation']['value']['assembled'][0]['text'] = 'concurrent'
            capture.write_text(history.dump_yaml(document))

    monkeypatch.setattr(migrate_history_deltas.tarfile, 'open', alter_after_backup)
    with pytest.raises(ValueError, match='changed during migration'):
        migrate_history_deltas.migrate(import_root, tmp_path / 'migration')


def test_inserting_older_observation_preserves_later_source_member(tmp_path):
    _write_family(tmp_path)
    _write_capture(tmp_path, '11.0.0', {'text_only': _change()})
    _write_capture(tmp_path, '11.0.2', {})
    owner = history.load_store(tmp_path).histories['gemma4', 'dynamo_v2']
    original = dict(owner.iter_resolved_snapshots(include_members=True))
    member = owner.snapshot_members['dynamo_v2-11.0.2']['text_only']
    target = copy.deepcopy(original['dynamo_v2-11.0.0'])
    target['text_only']['observation']['value']['assembled'][0]['text'] = 'different middle execution'
    target['text_only']['document']['captured_with']['dynamo_v2'] = '11.0.1'
    identity = 'dynamo_v2-11.0.1'
    owner.captures[identity] = copy.deepcopy(owner.captures['dynamo_v2-11.0.0'])
    owner.captures[identity].update(runtime_version='11.0.1', changes={})
    owner.captures[identity]['provenance']['captured_with']['dynamo_v2'] = '11.0.1'
    owner.capture_paths[identity] = tmp_path / 'families/gemma4' / (identity + '.yaml')
    history._replace_capture_snapshot(owner, identity, target)
    assert owner.resolve('dynamo_v2-11.0.2') == original['dynamo_v2-11.0.2']
    list(owner.iter_resolved_snapshots(include_members=True))
    assert owner.snapshot_members['dynamo_v2-11.0.2']['text_only'] == member


def test_new_observation_replaces_an_explicit_older_source_member(tmp_path):
    _write_family(tmp_path)
    _write_capture(tmp_path, '11.0.0', {'text_only': _change()})
    path = _write_capture(tmp_path, '11.0.1', {})
    document = yaml.safe_load(path.read_text())
    older = 'conformance/fixtures-unified-v2/families/gemma4/dynamo_v2-11.0.0.yaml'
    document['member_changes'] = {'text_only': older}
    path.write_text(history.dump_yaml(document))
    owner = history.load_store(tmp_path).histories['gemma4', 'dynamo_v2']
    first = owner.resolve('dynamo_v2-11.0.0')
    target = owner.resolve('dynamo_v2-11.0.1')
    target['text_only']['observation']['value']['assembled'][0]['text'] = 'new'
    target['text_only']['document'] = {'mode': 'unified', 'captured_with': {'dynamo_v2': '11.0.1'}}
    history._set_capture_snapshot(owner, 'dynamo_v2-11.0.1', target, first)
    assert owner.complete_snapshot('dynamo_v2-11.0.1')['records']['text_only']['member'] == owner._member('dynamo_v2-11.0.1')
