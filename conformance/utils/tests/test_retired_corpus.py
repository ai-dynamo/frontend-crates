# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Retiring the old corpus must keep Unified comparisons usable without its store."""
import json
import subprocess
import sys
from pathlib import Path

import pytest
import extract_fixtures


ROOT = Path(__file__).resolve().parents[3]


def assert_unified_manifest(manifest):
    assert manifest['shards'], 'Unified history pin is missing'
    assert all(row.get('format') == 'unified-history' for row in manifest['shards'])
    assert 'historical_catalog_sha256' not in manifest
    assert not manifest.get('inactive_shards')


def test_manifest_and_catalog_have_no_retired_corpus(source_catalog):
    assert_unified_manifest(json.loads((ROOT / 'conformance/fixtures-manifest.json').read_bytes()))
    assert not list((ROOT / 'conformance/fixtures').rglob('*.tar.gz'))
    assert source_catalog['sources']['archives'] == []
    assert all(row['mode'] == 'unified' for row in source_catalog['sources']['candidates'])
    implementations = {row['implementation'] for row in source_catalog['sources']['candidates']}
    assert {'dynamo_v1', 'dynamo_v1_jail', 'dynamo_v2'} <= implementations


@pytest.mark.parametrize(('script', 'arguments'), [
    ('extract_fixtures.py', ['--legacy-only']),
    ('package_fixtures.py', ['--stream-only']),
    ('publish_current_capture.py', ['--root', '.', '--mode', 'stream']),
])
def test_retired_corpus_commands_are_rejected(script, arguments):
    result = subprocess.run([sys.executable, str(ROOT / 'conformance/utils/src' / script), *arguments],
                            cwd=ROOT, capture_output=True, text=True, check=False)
    assert result.returncode == 2
    assert 'unrecognized arguments' in result.stderr or 'invalid choice' in result.stderr


def test_migration_cli_only_exposes_unified_snapshots():
    result = subprocess.run([sys.executable, str(ROOT / 'conformance/utils/src/migrate_capture_sources.py'), '--help'],
                            cwd=ROOT, capture_output=True, text=True, check=False)
    assert result.returncode == 0
    assert all(flag in result.stdout for flag in ('--source', '--destination', '--spool'))
    assert all(flag not in result.stdout for flag in ('--root', '--history-source', '--all-versions'))


def test_retired_archive_shards_are_rejected_before_filesystem_access():
    with pytest.raises(ValueError, match="only Unified history"):
        extract_fixtures.shard_file({'format': 'tar.gz', 'path': 'retired.tar.gz'})
