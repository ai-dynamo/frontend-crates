#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Migrate retained checkpoints through the shared delta writer with evidence parity."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

import capture_sources
import reconcile_dynamo_captures
import unified_archives
import unified_history
from viewer_codec import canonical_typed


def evidence(store):
    rows = {}
    for (family, implementation), history in store.histories.items():
        for identity, records in history.iter_resolved_snapshots(include_members=True):
            checkpoint = history.captures[identity]
            payload = {'records': records, 'members': history.snapshot_members[identity],
                       'suppressed': sorted(history.snapshot_suppressed_inheritance.get(identity, set())),
                       'checkpoint': {key: checkpoint[key] for key in ('provenance', 'document', 'execution_proof', 'historical_evidence') if key in checkpoint}}
            rows[f'{family}/{implementation}/{identity}'] = hashlib.sha256(canonical_typed(payload).encode()).hexdigest()
    return rows


def current_evidence(checkout, store, *, implementations=('dynamo_v1', 'dynamo_v1_jail')):
    rows = {}
    for name, family in store.families.items():
        for implementation in implementations:
            path = checkout / capture_sources.STORE / 'families' / name / f'{implementation}.yaml'
            if not path.exists():
                continue
            snapshot = unified_history.load_yaml(path)
            version = snapshot['capture_selection']['version']
            identity = f'{implementation}-{version}'
            history = store.histories[name, implementation]
            expected = unified_history._current_snapshot_digest(snapshot)
            retained = capture_sources.independent_snapshot(history.complete_snapshot(identity), family.document,
                source_member=f'{capture_sources.STORE}/families/{name}/{identity}.yaml')
            archives = history.captures[identity].get('historical_evidence', [])
            source_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
            if unified_history._current_snapshot_digest(retained) != expected:
                alternatives = [item for item in archives
                                if item.get('source_snapshot', {}).get('sha256') == source_sha256]
                if len(alternatives) != 1:
                    raise ValueError(f'current snapshot evidence was not retained: {path}')
                restored = unified_history.resolve_historical_evidence(history, identity, alternatives[0])
                if unified_history._current_snapshot_digest(restored) != expected:
                    raise ValueError(f'current snapshot evidence changed: {path}')
            for item in snapshot.get('historical_evidence', []):
                if not any(unified_history.typed_equal(item, other) for other in archives):
                    raise ValueError(f'current archived execution evidence was not retained: {path}')
            rows[f'{name}/{identity}'] = {'source_sha256': source_sha256, 'semantic_sha256': expected}
    return rows


def migrate(root: Path, work: Path, *, publish=False):
    root = root.resolve()
    live = root / capture_sources.STORE
    work.mkdir(parents=True, exist_ok=False)
    inventory = unified_history.store_inventory(live)
    manifest = work / 'original-manifest.json'
    manifest.write_text(json.dumps(inventory, indent=2) + '\n')
    archive = work / 'original-store.tar'
    with tarfile.open(archive, 'w') as output:
        output.add(live, arcname=capture_sources.STORE)
    manifest.chmod(0o400)
    archive.chmod(0o400)
    frozen = work / 'original'
    frozen.mkdir()
    with tarfile.open(archive, 'r:') as original:
        for member in original.getmembers():
            parts = Path(member.name).parts
            if Path(member.name).is_absolute() or any(part in {'.', '..'} for part in parts):
                raise ValueError('unsafe original archive member')
            target = frozen / member.name
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                with original.extractfile(member) as source, target.open('xb') as output:
                    shutil.copyfileobj(source, output)
            else:
                raise ValueError('original store contains a nonregular member')
    frozen_store = frozen / capture_sources.STORE
    if unified_history.store_inventory(frozen_store) != inventory:
        raise ValueError('original store changed while preserving migration evidence')
    candidate = work / 'candidate' / capture_sources.STORE
    before_evidence = {}
    after_evidence = {}
    current_snapshots = {}
    with unified_archives.expanded(frozen_store, temporary_parent=work) as expanded:
        family_names = sorted(path.parent.name for path in (expanded / 'families').glob('*/inputs_and_golden.yaml'))
        for name in family_names:
            original = work / 'families' / name / capture_sources.STORE
            directory = original / 'families' / name
            directory.mkdir(parents=True)
            for path in sorted((expanded / 'families' / name).glob('*.yaml')):
                if path.name == unified_history.FAMILY_DOCUMENT_NAME or '-' in path.stem:
                    shutil.copyfile(path, directory / path.name)
            # Receipts are admitted by the existing reader; retain their exact bytes.
            for resource in ('executions', 'receipts'):
                source = expanded / resource
                if source.exists():
                    shutil.copytree(source, original / resource)
            store = unified_history._load_store_unlocked(original, allow_empty_history=True)
            unified_history.preserve_current_snapshots(frozen, store)
            if not store.histories:
                target = candidate / 'families' / name / unified_history.FAMILY_DOCUMENT_NAME
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(directory / unified_history.FAMILY_DOCUMENT_NAME, target)
                continue
            expected = evidence(store)
            expected_current = current_evidence(frozen, store)
            before_evidence.update(expected)
            documents = unified_history._store_documents(store, compact_executions=True)
            unified_history._commit_store_documents(original, documents, expected_store=store)
            reloaded = unified_history._load_store_unlocked(original, allow_empty_history=True)
            actual = evidence(reloaded)
            if current_evidence(frozen, reloaded) != expected_current:
                raise ValueError(f'history migration changed current evidence: {name}')
            current_snapshots.update(expected_current)
            if actual != expected:
                changed = sorted(key for key in expected.keys() | actual.keys() if expected.get(key) != actual.get(key))
                raise ValueError(f'history migration changed resolved evidence: {changed}')
            after_evidence.update(actual)
            for path in original.rglob('*.yaml'):
                target = candidate / path.relative_to(original)
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists() and target.read_bytes() != path.read_bytes():
                    raise ValueError(f'conflicting migrated resource: {target}')
                shutil.copyfile(path, target)
            print(f'{name}: {len(actual)} checkpoints preserved', flush=True)
    before_size = sum(row['size'] for row in inventory if '-' in Path(row['path']).stem or row['path'].startswith(('executions/', 'receipts/')))
    after_size = sum(path.stat().st_size for path in candidate.rglob('*.yaml') if path.name != unified_history.FAMILY_DOCUMENT_NAME)
    if unified_history.store_inventory(live) != inventory:
        raise ValueError('capture store changed during migration')
    result = {'checkpoints': len(before_evidence), 'before_history_bytes': before_size,
              'after_history_bytes': after_size, 'evidence': after_evidence,
              'current_evidence': current_snapshots, 'published': publish}
    (work / 'migration.json').write_text(json.dumps(result, indent=2) + '\n')
    if publish:
        reconcile_dynamo_captures.publish_history_store(root, candidate, inventory)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    result = migrate(args.root, args.work, publish=args.publish)
    print(json.dumps({key: value for key, value in result.items() if key != 'evidence'}, sort_keys=True))


if __name__ == '__main__':
    main()
