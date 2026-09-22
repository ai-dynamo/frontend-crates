from __future__ import annotations

import io
import json
import hashlib
import tarfile
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import validate_legacy_convention as validator


def test_archive_member_with_old_name_is_rejected(tmp_path: Path) -> None:
    archive = tmp_path / "sample.tar.gz"
    old_root = "fixtures-stream-" + "v2"
    data = b"mode: streamv1\ncases: {}\n"
    with tarfile.open(archive, "w:gz") as tar:
        info = tarfile.TarInfo(f"toolcalling/{old_root}/case.yaml")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))

    errors = validator._archive_errors(archive, "sample.tar.gz")
    assert errors
    assert any("stale legacy-v2" in error for error in errors)


def test_validate_rejects_stale_member_from_manifest_archive(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    archive = repo / "conformance/fixtures/toolcalling/fixtures-stream-v1"
    archive.mkdir(parents=True)
    archive_path = archive / "inputs.tar.gz"
    old_root = "fixtures-stream-" + "v2"
    data = b"mode: streamv1\ncases: {}\n"
    with tarfile.open(archive_path, "w:gz") as tar:
        info = tarfile.TarInfo(f"toolcalling/{old_root}/case.yaml")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))

    manifest = {
        "shards": [
            {
                "path": "toolcalling/fixtures-stream-v1/inputs.tar.gz",
                "sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
                "size": archive_path.stat().st_size,
            }
        ]
    }
    manifest_path = repo / "conformance/fixtures-manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest))

    errors = validator.validate(repo)
    assert any("stale legacy-v2 name" in error for error in errors)
