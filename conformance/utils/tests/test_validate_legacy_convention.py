from __future__ import annotations

import io
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
