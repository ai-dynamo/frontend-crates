# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for the content-addressed fixture-cache publish contract.

`extract_fixtures.py` used to name its extraction directory by `manifest["snapshot"]`
(a human-readable pin) alone. Shard content is not uniquely determined by that pin --
a shard set can be re-pinned in place under an unchanged pin (it happened once, to
fixtures-batch-on-stream-v2). When that happened, the stale-check failed and the code
ran `shutil.rmtree()` directly on the live, previously-published directory, then
re-extracted in place -- while a concurrent, unlocked reader (a test in another git
worktree, or `_common.sh`, which never acquires the flock at all) could still be
reading files through the `toolcalling`/`reasoning`/`unified` symlinks. This was
reproduced for real: a concurrent reader hit `FileNotFoundError` after 103/3345
fixture files were successfully opened.

The fix: key the cache directory by `fixtures_identity(shards)` (content, not pin),
build every extraction into a private `.tmp.*` directory, and publish with ONE atomic
rename. These tests exercise that contract directly, with mocked/tiny extraction and
no real concurrency or timing -- the correctness property is about ORDERING (build
under tmp, one rename at the end), so a mocked mid-build failure exercises the exact
code path a real SIGKILL would leave behind.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

UTILS = Path(__file__).resolve().parents[1]
SRC = UTILS / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import extract_fixtures  # noqa: E402


def _shard(path, sha256, size=1):
    return {"path": path, "sha256": sha256, "size": size}


def _fake_extract_tarball(_tarball_path, dest_dir, verbose=False):
    """Stand-in for the real tar extraction: writes one marker file so the
    directory is non-empty and distinguishable, without needing a real
    tarball on disk."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    (dest_dir / "marker.txt").write_text(str(dest_dir))


@pytest.fixture
def cache_root(tmp_path, monkeypatch):
    root = tmp_path / "cache"
    monkeypatch.setattr(extract_fixtures, "get_cache_root", lambda: root)
    monkeypatch.setattr(extract_fixtures, "shard_file", lambda s: Path(f"/fake/{s['path']}"))
    monkeypatch.setattr(extract_fixtures, "extract_tarball", _fake_extract_tarball)
    return root


def _run_main(tmp_path, monkeypatch, manifest, argv=()):
    # A real file on disk, not a global Path.exists/read_text monkeypatch:
    # patching a method on the Path CLASS itself is process-wide, and calling
    # through to "the real" Path.exists/read_text from inside the patched
    # version just calls the patched version again -- infinite recursion.
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr(extract_fixtures, "MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(sys, "argv", ["extract_fixtures.py", *argv])
    extract_fixtures.main()


def test_two_shard_sets_under_the_same_pin_do_not_collide(cache_root, tmp_path, monkeypatch, capsys):
    """(A) A re-pin in place -- same `snapshot` stamp, different shard hashes --
    must produce two distinct, coexisting directories, not one rmtree-ing the
    other. The first directory's own content must be untouched afterward."""
    manifest_v1 = {"snapshot": "20260101_000000", "shards": [_shard("toolcalling/a.tar.gz", "hash1")]}
    _run_main(tmp_path, monkeypatch, manifest_v1)
    out_v1 = capsys.readouterr().out.strip()
    dir_v1 = Path(out_v1)
    assert dir_v1.is_dir()
    marker_v1 = (dir_v1 / "marker.txt").read_text()

    manifest_v2 = {"snapshot": "20260101_000000", "shards": [_shard("toolcalling/a.tar.gz", "hash2")]}
    _run_main(tmp_path, monkeypatch, manifest_v2)
    out_v2 = capsys.readouterr().out.strip()
    dir_v2 = Path(out_v2)

    assert dir_v1 != dir_v2, "different shard content under the same pin must get different directories"
    assert dir_v1.is_dir(), "the first identity's directory must survive the second extraction untouched"
    assert (dir_v1 / "marker.txt").read_text() == marker_v1


def test_interrupted_build_never_appears_at_the_published_name(cache_root, tmp_path, monkeypatch):
    """(B) A failure partway through extraction must leave NO directory at the
    content-addressed published name -- only a `.tmp.*` remnant, proving the
    SIGKILL-safety property: nothing is published except via the final
    single-rename step."""

    def _boom(_tarball_path, dest_dir, verbose=False):
        # Mirror the real extract_tarball's mkdir-then-populate ordering (it
        # creates dest_dir before writing any content) so the crash leaves a
        # realistic partial `.tmp.*` directory on disk to assert against,
        # not an early raise that never touches the filesystem at all.
        dest_dir.mkdir(parents=True, exist_ok=True)
        raise RuntimeError("simulated crash mid-extraction")

    monkeypatch.setattr(extract_fixtures, "extract_tarball", _boom)
    manifest = {"snapshot": "20260101_000000", "shards": [_shard("toolcalling/a.tar.gz", "hash1")]}
    fid = extract_fixtures.fixtures_identity(manifest["shards"])
    published = cache_root / f"20260101_000000-{fid}"

    with pytest.raises(RuntimeError, match="simulated crash"):
        _run_main(tmp_path, monkeypatch, manifest)

    assert not published.exists(), "a failed build must never appear at the published identity name"
    leftovers = list(cache_root.glob(".tmp.*")) if cache_root.exists() else []
    assert leftovers, "the partial build should be visible only under a .tmp.* scratch name"


def test_identical_identity_is_a_cache_hit_not_a_rebuild(cache_root, tmp_path, monkeypatch, capsys):
    """(C) A second call with the identical shard content is a cache hit --
    `extract_tarball` must not run again."""
    manifest = {"snapshot": "20260101_000000", "shards": [_shard("toolcalling/a.tar.gz", "hash1")]}
    _run_main(tmp_path, monkeypatch, manifest)
    capsys.readouterr()

    def _fail_if_called(*a, **k):
        raise AssertionError("extract_tarball must not be called on a cache hit")

    monkeypatch.setattr(extract_fixtures, "extract_tarball", _fail_if_called)
    _run_main(tmp_path, monkeypatch, manifest)
    out = capsys.readouterr()
    assert "Cache hit" in out.err


def test_full_refresh_builds_a_new_generation_without_touching_the_old_one(cache_root, tmp_path, monkeypatch, capsys):
    """Reviewer-caught regression, then a reviewer-caught regression IN the
    first fix: `--full-refresh` against an already-published identity used to
    (1) silently discard the rebuild (defeating the flag), then a first fix
    attempt (2) renamed the existing published directory out of the way to
    make room for the rebuild -- which reintroduces the EXACT mutation-under
    -a-live-reader hazard this whole module exists to close. A reader that
    resolved the OLD generation's path before `--full-refresh` ran must keep
    working, unaffected, for as long as it keeps reading -- there is no safe
    moment to rename or delete a path a reader might still be using.

    The correct contract: `--full-refresh` against an already-published
    identity NEVER touches the existing directory. It publishes a NEW,
    disambiguated generation (`{identity}.refresh1`, `.refresh2`, ...) and
    only the symlinks (the one mutable "current" locator) retarget to it.
    """
    call_count = {"n": 0}

    def _counting_extract_tarball(_tarball_path, dest_dir, verbose=False):
        call_count["n"] += 1
        dest_dir.mkdir(parents=True, exist_ok=True)
        (dest_dir / "marker.txt").write_text(f"build #{call_count['n']}")

    monkeypatch.setattr(extract_fixtures, "extract_tarball", _counting_extract_tarball)
    manifest = {"snapshot": "20260101_000000", "shards": [_shard("toolcalling/a.tar.gz", "hash1")]}
    _run_main(tmp_path, monkeypatch, manifest)
    out_v1 = capsys.readouterr().out.strip()
    old_dir = Path(out_v1)
    old_marker = old_dir / "marker.txt"
    assert old_marker.read_text() == "build #1"

    _run_main(tmp_path, monkeypatch, manifest, argv=["--full-refresh"])
    out_v2 = capsys.readouterr()
    new_dir = Path(out_v2.out.strip())

    assert call_count["n"] == 2, "extract_tarball must actually run again under --full-refresh"
    assert "built a new generation" in out_v2.err
    assert new_dir != old_dir, "the refresh must publish to a NEW path, never reuse the old one"
    assert new_dir.name.endswith(".refresh1"), new_dir.name
    assert (new_dir / "marker.txt").read_text() == "build #2"
    # The old generation's directory and content are byte-for-byte untouched --
    # this is the property a concurrent reader's continued access depends on.
    assert old_dir.is_dir(), "the OLD generation must still exist -- a reader may still be reading it"
    assert old_marker.read_text() == "build #1", "the OLD generation's content must be completely unchanged"


def test_plain_run_after_full_refresh_resolves_to_the_new_generation(cache_root, tmp_path, monkeypatch, capsys):
    """MUST finding (blind audit of the `--full-refresh` redesign): a plain
    (no-flag) run must resolve to whatever generation `--full-refresh` most
    recently published, never silently fall back to the abandoned original.
    `_common.sh` runs this script with no flags before every conformance
    test/render, so an invisible refresh means the fix `--full-refresh` was
    run to apply never actually takes effect for anything downstream --
    reproduced against the unmodified module before this fix: the very next
    plain run flipped the symlink straight back to the corrupted original.
    """
    manifest = {"snapshot": "20260101_000000", "shards": [_shard("toolcalling/a.tar.gz", "hash1")]}
    _run_main(tmp_path, monkeypatch, manifest)
    capsys.readouterr()
    _run_main(tmp_path, monkeypatch, manifest, argv=["--full-refresh"])
    refreshed_dir = Path(capsys.readouterr().out.strip())
    assert refreshed_dir.name.endswith(".refresh1")

    _run_main(tmp_path, monkeypatch, manifest)  # plain run, no flag
    out = capsys.readouterr()
    plain_dir = Path(out.out.strip())

    assert plain_dir == refreshed_dir, (
        "a plain run after --full-refresh must resolve to the refreshed "
        f"generation, not the abandoned original; got {plain_dir}"
    )
    assert "Cache hit" in out.err


def test_extracted_snapshot_dir_resolves_to_the_new_generation_after_full_refresh(
    cache_root, tmp_path, monkeypatch, capsys
):
    """Same MUST finding, the other consumer the audit specifically asked to
    be hunted for: `package_fixtures.py`'s `_extracted_snapshot_dir()` used
    to reconstruct the bare `{pin}-{fid}` path directly instead of following
    `resolve_current_generation`, so it also kept resolving to the abandoned
    (possibly corrupted) generation after a refresh."""
    package_fixtures_src = SRC
    if str(package_fixtures_src) not in sys.path:
        sys.path.insert(0, str(package_fixtures_src))
    import package_fixtures  # noqa: E402  (sibling module, same sys.path entry as extract_fixtures)

    manifest = {"snapshot": "20260101_000000", "shards": [_shard("toolcalling/a.tar.gz", "hash1")]}
    _run_main(tmp_path, monkeypatch, manifest)
    capsys.readouterr()
    _run_main(tmp_path, monkeypatch, manifest, argv=["--full-refresh"])
    refreshed_dir = Path(capsys.readouterr().out.strip())
    assert refreshed_dir.name.endswith(".refresh1")

    manifest_path = tmp_path / "package_fixtures_manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr(package_fixtures, "ROOT", tmp_path)
    monkeypatch.setattr(package_fixtures, "MANIFEST_REL", manifest_path.relative_to(tmp_path))

    resolved = package_fixtures._extracted_snapshot_dir()
    assert resolved == refreshed_dir, (
        "_extracted_snapshot_dir() must resolve to the refreshed generation, "
        f"not the abandoned original; got {resolved}"
    )


def test_refresh_publish_retries_past_a_colliding_generation_name(cache_root, tmp_path, monkeypatch, capsys):
    """MUST finding (blind audit): the `.refreshN` candidate picked by
    `resolve_current_generation` is check-then-act -- a second, genuinely
    concurrent `--full-refresh` run can occupy that exact name between the
    check and this process's own rename. The loser must retry the next
    generation number via the same narrowed-errno handling the generation-0
    publish path already uses, not raise an unhandled `OSError`.

    A first version of this test injected the competitor on the FIRST call
    to `resolve_current_generation` -- but `main()`'s `--full-refresh` path
    calls it TWICE: once for the cache-hit check near the top (which still
    runs even when `--full-refresh` is set, before the flag is even
    consulted), and once inside this retry loop, right before computing
    `refresh_n`. Injecting on the first call means the SECOND call already
    observes the competitor and correctly picks `.refresh2` on its very
    first attempt -- the retry-on-collision branch this test exists to
    cover never actually runs, even though the assertions still pass (a
    coverage gap a later audit caught: the test's own docstring claimed to
    reproduce the race but did not). This version counts calls and injects
    strictly AFTER the SECOND call returns its own (pre-injection) result,
    so `refresh_n` is computed as if no competitor exists yet, and the
    injected competitor is only on disk by the time the loop's own
    `tmp_dir.rename(candidate)` actually runs -- a genuine collision.
    """
    manifest = {"snapshot": "20260101_000000", "shards": [_shard("toolcalling/a.tar.gz", "hash1")]}
    fid = extract_fixtures.fixtures_identity(manifest["shards"])
    _run_main(tmp_path, monkeypatch, manifest)  # publish generation 0
    capsys.readouterr()

    real_resolve = extract_fixtures.resolve_current_generation
    real_write_state = extract_fixtures.write_state
    call_count = {"n": 0}

    def _resolve_then_inject_competitor_after_the_second_call(cache_root_, pin_, fid_, pinned_shards_):
        call_count["n"] += 1
        result = real_resolve(cache_root_, pin_, fid_, pinned_shards_)
        if call_count["n"] == 2:
            # Call #1 was the cache-hit check near the top of main() (runs
            # even under --full-refresh); call #2 is the one inside the
            # retry loop, immediately before `refresh_n` is computed from
            # its result. Injecting AFTER this call returns means `result`
            # (and therefore `refresh_n`) reflects the pre-competitor state,
            # but the competitor is on disk before the loop's own rename
            # attempt -- the exact check-to-rename race window.
            competitor = cache_root_ / f"{pin_}-{fid_}.refresh1"
            competitor.mkdir(parents=True)
            (competitor / "marker.txt").write_text("sentinel-from-a-competing-refresh")
            real_write_state(competitor, pin_, manifest["shards"])
        return result

    monkeypatch.setattr(
        extract_fixtures, "resolve_current_generation", _resolve_then_inject_competitor_after_the_second_call
    )

    _run_main(tmp_path, monkeypatch, manifest, argv=["--full-refresh"])
    out = capsys.readouterr()
    new_dir = Path(out.out.strip())

    assert call_count["n"] == 2, "sanity: --full-refresh must call resolve_current_generation exactly twice"
    assert new_dir.name.endswith(".refresh2"), (
        f"the retry must land on the next free generation after the collision, got {new_dir.name}"
    )
    competitor = cache_root / f"20260101_000000-{fid}.refresh1"
    assert (
        competitor / "marker.txt"
    ).read_text() == "sentinel-from-a-competing-refresh", "the competitor's generation must survive completely untouched"


def test_refresh_publish_collision_retry_negative_control(cache_root, tmp_path, monkeypatch, capsys):
    """Negative control for the test above: prove it can actually fail. The
    retry loop's collision handling is disabled -- but ONLY from the moment
    the SAME real collision the positive test exercises would occur, not
    from the start of the run. Disabling `RENAME_DEST_EXISTS_ERRNOS` before
    `_run_main` even starts (a first draft of this control did that) also
    breaks the unrelated, pre-existing generation-0 collision check earlier
    in `main()`, so the run fails there instead -- a real failure, but not
    evidence the RETRY loop's own handling matters. Instead the errno tuple
    is flipped from inside the `resolve_current_generation` wrapper, exactly
    when the positive test's competitor is injected: the first (pre-existing)
    collision check has already run and succeeded by that point, so only the
    retry loop's own narrowed catch is disabled. With it disabled, the
    identical race must surface as an unhandled `OSError` instead of a
    successful `.refresh2` publish -- proving the positive test's green
    result actually depends on this code, not on some other path."""
    manifest = {"snapshot": "20260101_000000", "shards": [_shard("toolcalling/a.tar.gz", "hash1")]}
    _run_main(tmp_path, monkeypatch, manifest)  # publish generation 0
    capsys.readouterr()

    real_resolve = extract_fixtures.resolve_current_generation
    real_write_state = extract_fixtures.write_state
    call_count = {"n": 0}

    def _resolve_then_inject_competitor_and_disable_retry_handling(cache_root_, pin_, fid_, pinned_shards_):
        call_count["n"] += 1
        result = real_resolve(cache_root_, pin_, fid_, pinned_shards_)
        if call_count["n"] == 2:
            competitor = cache_root_ / f"{pin_}-{fid_}.refresh1"
            competitor.mkdir(parents=True)
            (competitor / "marker.txt").write_text("sentinel-from-a-competing-refresh")
            real_write_state(competitor, pin_, manifest["shards"])
            monkeypatch.setattr(extract_fixtures, "RENAME_DEST_EXISTS_ERRNOS", ())
        return result

    monkeypatch.setattr(
        extract_fixtures, "resolve_current_generation", _resolve_then_inject_competitor_and_disable_retry_handling
    )

    with pytest.raises(OSError):
        _run_main(tmp_path, monkeypatch, manifest, argv=["--full-refresh"])
    assert call_count["n"] == 2, "sanity: the race must still reach the same collision point as the positive test"


def test_concurrent_double_publish_of_one_identity_is_a_safe_noop(cache_root, tmp_path, monkeypatch, capsys):
    """(D) Simulates a genuine concurrent double-publish: a SECOND process
    finishes and publishes the identical identity while THIS process is still
    mid-build, so this process's own final rename genuinely collides with a
    valid, freshly-published directory. Pre-creating the "competitor" before
    `main()` even starts would just take the ordinary cache-hit path (no
    rename ever attempted) -- the race only exists at the rename step, so the
    competitor has to appear there, injected via the same `write_state` hook
    `main()` calls right before its own rename."""
    manifest = {"snapshot": "20260101_000000", "shards": [_shard("toolcalling/a.tar.gz", "hash1")]}
    shards = manifest["shards"]
    fid = extract_fixtures.fixtures_identity(shards)
    published = cache_root / f"20260101_000000-{fid}"

    real_write_state = extract_fixtures.write_state

    def _write_state_then_let_a_competitor_publish_first(snap_dir, snapshot, shards_):
        real_write_state(snap_dir, snapshot, shards_)
        # `snap_dir` here is THIS process's own tmp_dir, about to be renamed.
        # Publish the "other process's" identical-identity result at the real
        # published name right now, simulating it winning the race.
        published.mkdir(parents=True)
        (published / "toolcalling").mkdir()
        (published / "toolcalling" / "marker.txt").write_text("sentinel-from-the-other-publisher")
        real_write_state(published, snapshot, shards_)

    monkeypatch.setattr(extract_fixtures, "write_state", _write_state_then_let_a_competitor_publish_first)

    _run_main(tmp_path, monkeypatch, manifest)
    out = capsys.readouterr()

    assert "already published" in out.err
    assert (published / "toolcalling" / "marker.txt").read_text() == "sentinel-from-the-other-publisher"
    assert not list(cache_root.glob(".tmp.*")), "the redundant temp build must be discarded, not left behind"


def test_rejects_an_untrusted_directory_occupying_the_identity_name(cache_root, tmp_path, monkeypatch):
    """A directory at the identity name with NO valid state file (or a
    mismatched one) must not be silently trusted as an authoritative prior
    publish -- the rename failure must propagate instead of discarding a
    verified fresh build in favor of unknown content."""
    manifest = {"snapshot": "20260101_000000", "shards": [_shard("toolcalling/a.tar.gz", "hash1")]}
    fid = extract_fixtures.fixtures_identity(manifest["shards"])
    stray = cache_root / f"20260101_000000-{fid}"
    stray.mkdir(parents=True)
    (stray / "not_a_real_state_file.txt").write_text("garbage")

    with pytest.raises(OSError):
        _run_main(tmp_path, monkeypatch, manifest)


def test_extraction_never_touches_an_older_published_directory(cache_root, tmp_path, monkeypatch):
    """A published, content-addressed directory (never named `.tmp.*`) is
    never touched by ANY later extraction, regardless of its age. There is
    deliberately no orphan-cleanup sweep of any kind (removed after review:
    an mtime-based age gate could delete a directory a different,
    genuinely-still-running extraction is actively building into, since
    `extract_tarball`'s `mkdir(exist_ok=True)` means the victim wouldn't
    even notice and would go on to publish a corrupted "complete" state).
    Leftover `.tmp.*` dirs are an accepted disk-hygiene cost, not a
    correctness concern -- never referenced by any symlink, never mistaken
    for a valid cache."""
    manifest = {"snapshot": "20260101_000000", "shards": [_shard("toolcalling/a.tar.gz", "hash1")]}
    _run_main(tmp_path, monkeypatch, manifest)
    published = [d for d in cache_root.iterdir() if not d.name.startswith(".")]
    assert len(published) == 1
    marker = published[0] / "marker.txt"  # _fake_extract_tarball writes it at the extraction root
    old_time = 0  # 1970 -- as old as an mtime can be
    os.utime(published[0], (old_time, old_time))

    # A second extraction (different identity) must not disturb the first.
    manifest2 = {"snapshot": "20260101_000000", "shards": [_shard("toolcalling/a.tar.gz", "hash2")]}
    _run_main(tmp_path, monkeypatch, manifest2)

    assert published[0].is_dir(), "an old PUBLISHED directory must survive a later, unrelated extraction"
    assert marker.is_file()


def test_extraction_never_removes_any_tmp_dir_of_any_age(cache_root, tmp_path, monkeypatch):
    """No sweep exists at all: a `.tmp.*` dir -- fresh (may belong to a
    genuinely still-running concurrent extraction; `_common.sh` has no lock
    at all) or old (a leftover from a past crash) -- is left alone by a
    later extraction either way. This is the intentional post-review design,
    not a gap: age-based cleanup was removed rather than made more precise,
    since it has no correctness value and the corruption risk it introduced
    outweighed the disk-hygiene benefit."""
    fresh_tmp = cache_root / ".tmp.somebody-else.999.deadbeef"
    fresh_tmp.mkdir(parents=True)
    (fresh_tmp / "in_progress.txt").write_text("still building")

    old_tmp = cache_root / ".tmp.orphaned.111.cafebabe"
    old_tmp.mkdir(parents=True)
    os.utime(old_tmp, (0, 0))

    manifest = {"snapshot": "20260101_000000", "shards": [_shard("toolcalling/a.tar.gz", "hash1")]}
    _run_main(tmp_path, monkeypatch, manifest)

    assert fresh_tmp.is_dir(), "a recent .tmp.* dir must be left alone -- it may be a live concurrent build"
    assert old_tmp.is_dir(), "an old .tmp.* dir must also be left alone -- no sweep exists to remove it"


def test_show_info_marks_the_current_pin_under_the_new_naming(cache_root, tmp_path, monkeypatch, capsys):
    """`show_info`'s `d.name == pin` check always misses once directories are
    named `{pin}-{fid}`; must use a prefix match instead."""
    manifest = {"snapshot": "20260101_000000", "shards": [_shard("toolcalling/a.tar.gz", "hash1")]}
    _run_main(tmp_path, monkeypatch, manifest)
    capsys.readouterr()

    extract_fixtures.show_info(manifest, cache_root)
    out = capsys.readouterr().out
    assert "<- current pin" in out
