from __future__ import annotations

import tarfile
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import validate_legacy_convention as validator


def test_active_tree_has_no_legacy_v2_names() -> None:
    repo = Path(__file__).resolve().parents[3]
    assert validator.validate(repo) == []


def test_archive_member_with_old_name_is_rejected(tmp_path: Path) -> None:
    archive = tmp_path / "sample.tar.gz"
    old_root = "fixtures-stream-" + "v2"
    old_mode = "stream" + "v2"
    source = tmp_path / old_root
    source.mkdir()
    (source / "case.yaml").write_text(f"mode: {old_mode}\ncases: {{}}\n")
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(source, arcname="toolcalling/" + old_root)

    errors = validator._archive_errors(archive, "sample.tar.gz")
    assert errors
    assert any("stale legacy-v2" in error for error in errors)


def test_legitimate_parser_v2_names_are_not_stale() -> None:
    assert all(token not in validator.LEGITIMATE_V2 for token in validator.OLD_TOKENS)
