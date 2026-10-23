# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Isolated checkout identity for synthetic fixture-extraction tests."""
from pathlib import Path

import extract_fixtures


REPO = Path(__file__).resolve().parents[3]


def configure_extractor_root(root, monkeypatch, *, manifest_path=None):
    # ROOT selects current versus historical storage and supplies release metadata.
    # Partial path patches can accidentally materialize the real checkout instead.
    package = root / "parsers/v2/Cargo.toml"
    package.parent.mkdir(parents=True, exist_ok=True)
    package.write_bytes((REPO / "parsers/v2/Cargo.toml").read_bytes())
    conformance = root / "conformance"
    fixtures = conformance / "fixtures"
    history = conformance / "fixtures-unified-v2"
    fixtures.mkdir(parents=True, exist_ok=True)
    history.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(extract_fixtures, "ROOT", root)
    monkeypatch.setattr(extract_fixtures, "MANIFEST_PATH", manifest_path or conformance / "fixtures-manifest.json")
    monkeypatch.setattr(extract_fixtures, "FIXTURES_DIR", fixtures)
    monkeypatch.setattr(extract_fixtures, "HISTORY_DIR", history)
