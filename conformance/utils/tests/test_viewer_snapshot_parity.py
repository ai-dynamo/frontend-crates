# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "conformance/utils/src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
import unified_history
import viewer_snapshots


def test_full_corpus_json_snapshots_match_python_and_javascript_resolution(tmp_path):
    corpus = ROOT / "conformance/fixtures-unified-v2"
    catalog = json.loads((ROOT / "conformance/viewer-catalog.json").read_text())
    store = unified_history.load_store(corpus)
    site = tmp_path / "site"
    published = viewer_snapshots.publish_snapshots(ROOT, site, catalog, store=store)
    catalog_path = tmp_path / "published-catalog.json"
    catalog_path.write_text(json.dumps(published))
    result = subprocess.run(
        ["node", "conformance/viewer/snapshot_parity.js", str(ROOT), str(site), str(catalog_path)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
        timeout=600,
    )
    assert "full corpus parity passed:" in result.stdout
