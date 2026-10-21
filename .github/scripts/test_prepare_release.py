# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise release routing in real Git repositories, with recording CLI stubs."""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("prepare-release.sh").resolve()


class PrepareReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.repo = root / "repo"
        self.repo.mkdir()
        binaries = root / "bin"
        binaries.mkdir()
        self.log = root / "calls"
        self.log.touch()
        self.env = {
            **os.environ,
            "PATH": f"{binaries}:{os.environ['PATH']}",
            "CALL_LOG": str(self.log),
            "SEMVER_EXIT_CODE": "0",
            "GIT_AUTHOR_NAME": "Release test",
            "GIT_AUTHOR_EMAIL": "release-test@example.com",
            "GIT_COMMITTER_NAME": "Release test",
            "GIT_COMMITTER_EMAIL": "release-test@example.com",
        }
        stub = """#!/usr/bin/env python3
import os
from pathlib import Path
import sys
tool = Path(sys.argv[0]).name
with open(os.environ['CALL_LOG'], 'a') as log:
    log.write(' '.join([tool, *sys.argv[1:]]) + '\\n')
if tool == 'cargo':
    sys.exit(int(os.environ['SEMVER_EXIT_CODE']))
manifest = Path(sys.argv[-1]) / 'Cargo.toml'
manifest.write_text(manifest.read_text().replace('0.1.0', '0.1.1'))
"""
        for name in ("cargo", "release-plz"):
            executable = binaries / name
            executable.write_text(stub)
            executable.chmod(0o755)
        self.git("init", "-q")
        self.write("Cargo.toml", '[workspace]\nmembers = [\n"sample",\n"sample-v2",\n"private",\n]\n')
        for name in ("sample", "sample-v2", "private"):
            self.write(f"{name}/Cargo.toml", f'[package]\nname = "{name}"\nversion = "0.1.0"\n')
        with (self.repo / "private/Cargo.toml").open("a") as manifest:
            manifest.write("publish = false\n")
        self.commit()

    def git(self, *args):
        return subprocess.run(
            ["git", "-c", "commit.gpgsign=false", "-c", "tag.gpgsign=false",
             "-c", "core.hooksPath=/dev/null", *args],
            cwd=self.repo, env=self.env, check=True, capture_output=True, text=True,
        )

    def write(self, name, contents):
        path = self.repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents)

    def commit(self):
        self.git("add", ".")
        self.git("commit", "-sqm", "fix: test change")

    def prepare(self, expected_code=0):
        result = subprocess.run(
            ["bash", str(SCRIPT)], cwd=self.repo, env=self.env,
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, expected_code, result.stdout + result.stderr)
        return self.log.read_text().splitlines()

    def test_manual_pin_checks_latest_exact_package_tag(self):
        for version in ("0.0.9", "0.0.10"):
            self.git("tag", f"sample-v{version}")
        self.git("tag", "sample-v2-v0.0.99")
        self.assertEqual(self.prepare(), [
            "cargo semver-checks check-release --package sample --baseline-rev sample-v0.0.10",
            "cargo semver-checks check-release --package sample-v2 --baseline-rev sample-v2-v0.0.99",
        ])

    def test_semver_failure_stops_before_later_package_updates(self):
        self.git("tag", "sample-v0.0.9")
        self.git("tag", "sample-v2-v0.1.0")
        self.write("sample-v2/src/lib.rs", "pub fn new_api() {}\n")
        self.commit()
        self.env["SEMVER_EXIT_CODE"] = "42"
        self.assertEqual(self.prepare(expected_code=42), [
            "cargo semver-checks check-release --package sample --baseline-rev sample-v0.0.9",
        ])

    def test_no_release_and_unchanged_packages_are_skipped(self):
        # The other crate's prefix must not become sample's release baseline.
        self.git("tag", "sample-v2-v0.1.0")
        self.git("tag", "private-v0.0.9")
        self.assertEqual(self.prepare(), [])

    def test_only_changed_package_is_automatically_sized(self):
        for name in ("sample", "sample-v2"):
            self.git("tag", f"{name}-v0.1.0")
        self.write("sample/src/lib.rs", "pub fn new_api() {}\n")
        self.commit()
        self.assertEqual(self.prepare(), ["release-plz update --allow-dirty -p sample"])

    def test_root_manifest_changes_update_all_published_packages(self):
        for name in ("sample", "sample-v2", "private"):
            self.git("tag", f"{name}-v0.1.0")
        with (self.repo / "Cargo.toml").open("a") as manifest:
            manifest.write("# changed workspace dependency\n")
        self.commit()
        self.assertEqual(self.prepare(), [
            "release-plz update --allow-dirty -p sample",
            "release-plz update --allow-dirty -p sample-v2",
        ])
        self.assertIn('version = "0.1.1"', (self.repo / "sample-v2/Cargo.toml").read_text())

    def test_preexisting_dirty_changes_are_rejected(self):
        self.write("unexpected", "must not be packaged\n")
        self.assertEqual(self.prepare(expected_code=1), [])


if __name__ == "__main__":
    unittest.main()
