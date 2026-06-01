#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Check (or apply) a one-way sync from a local ai-dynamo/dynamo checkout into this repo.
#
# Sources:  $DYNAMO_SRC/lib/{protocols,tokenizers,parsers,renderer}/
# Targets:  ./{protocols,tokenizers,parsers,renderer}/
#
# Usage:
#   scripts/sync-from-dynamo.sh                 # check, against $DYNAMO_SRC (default /ephemeral/dynamo)
#   scripts/sync-from-dynamo.sh /path/to/dynamo # check, against a specific dynamo checkout
#   scripts/sync-from-dynamo.sh --apply         # apply changes to src/ and tests/ only
#   scripts/sync-from-dynamo.sh --apply /path   # apply against a specific checkout
#
# What gets synced:
#   src/        — code (always synced in --apply mode)
#   tests/      — tests (always synced in --apply mode)
#   llm/tests/data/sample-models/ — tokenizer fixtures the synced tests load
#                 via "$CARGO_MANIFEST_DIR/../llm/tests/data/..." paths. Mirrored
#                 verbatim so synced tests run offline without path edits.
#
# What does NOT get synced (flagged for manual review in check mode):
#   Cargo.toml  — local copy is inlined for standalone publishing.
#   CLAUDE.md, AGENTS.md, README.md, *_CASES.md — local copies are reframed
#                 for the standalone repo. Compare manually if dynamo's diverge.

set -euo pipefail

APPLY=0
DYNAMO_SRC=""
for arg in "$@"; do
  case "$arg" in
    --apply) APPLY=1 ;;
    -h|--help)
      grep -E '^#' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *) DYNAMO_SRC="$arg" ;;
  esac
done
DYNAMO_SRC="${DYNAMO_SRC:-${DYNAMO_SRC_ENV:-/ephemeral/dynamo}}"

if [ ! -d "$DYNAMO_SRC/lib" ]; then
  echo "error: $DYNAMO_SRC does not look like a dynamo checkout (no lib/ dir)" >&2
  exit 1
fi

HERE="$(cd "$(dirname "$0")/.." && pwd)"
CRATES=(protocols tokenizers parsers renderer)

# Files we own locally that we don't want overwritten by a sync.
MANUAL_REVIEW=(Cargo.toml README.md CLAUDE.md AGENTS.md)

if [ "$APPLY" = "1" ]; then
  echo "=== applying sync from $DYNAMO_SRC ==="
else
  echo "=== checking sync from $DYNAMO_SRC (dry run) ==="
  echo "    +++ create  --- delete  >f. update"
  echo
fi

CODE_CHANGED=0
for c in "${CRATES[@]}"; do
  src="$DYNAMO_SRC/lib/$c"
  dst="$HERE/$c"
  echo "--- $c ---"

  # TODO(renderer): drop this guard once PR ai-dynamo/dynamo#9946 lands
  # lib/renderer on dynamo main. Until then `renderer` exists in this repo
  # but not upstream, so the hourly sync-check (which runs against dynamo
  # main) has nothing to diff against. Skip it explicitly — rather than
  # silently comparing nothing — so the temporary state is visible in the
  # output and intentional.
  if [ ! -d "$src" ]; then
    echo "  SKIP: lib/$c not present in $DYNAMO_SRC (not yet on dynamo main); skipping until upstream lands."
    continue
  fi

  # Sync src/ and tests/ subdirs only. These are pure code.
  # --checksum compares by content hash, not size+mtime — important for CI,
  # where a freshly-cloned dynamo gives every file a "now" timestamp.
  for sub in src tests; do
    [ -d "$src/$sub" ] || continue
    if [ "$APPLY" = "1" ]; then
      rsync -a --delete --checksum "$src/$sub/" "$dst/$sub/"
    else
      # Only flag real content changes (lines starting with > < c *), not
      # metadata-only attribute drift (lines starting with .).
      out=$(rsync -a --delete --checksum --dry-run --itemize-changes "$src/$sub/" "$dst/$sub/" | grep -E '^[<>c*]' || true)
      if [ -n "$out" ]; then
        echo "  $sub/ would change:"
        echo "$out" | sed 's/^/    /'
        CODE_CHANGED=1
      fi
    fi
  done

  # Files we own — flag drift, never auto-apply.
  for f in "${MANUAL_REVIEW[@]}"; do
    [ -f "$src/$f" ] || continue
    [ -f "$dst/$f" ] || continue
    if ! diff -q "$src/$f" "$dst/$f" >/dev/null 2>&1; then
      echo "  NOTE: $f differs from dynamo's copy (local copy intentionally diverged)."
      echo "        Review with: diff $src/$f $dst/$f"
    fi
  done
done

# Test fixtures live outside the three crates (dynamo's lib/llm/tests/data) but
# are loaded by the synced tests via relative paths. Mirror them so the
# standalone repo is self-contained and fixture drift is caught like code drift.
FIXTURE_REL="llm/tests/data/sample-models"
fsrc="$DYNAMO_SRC/lib/$FIXTURE_REL"
fdst="$HERE/$FIXTURE_REL"
if [ -d "$fsrc" ]; then
  echo "--- test fixtures ($FIXTURE_REL) ---"
  if [ "$APPLY" = "1" ]; then
    mkdir -p "$fdst"
    rsync -a --delete --checksum "$fsrc/" "$fdst/"
  else
    out=$(rsync -a --delete --checksum --dry-run --itemize-changes "$fsrc/" "$fdst/" | grep -E '^[<>c*]' || true)
    if [ -n "$out" ]; then
      echo "  would change:"
      echo "$out" | sed 's/^/    /'
      CODE_CHANGED=1
    fi
  fi
fi

if [ "$APPLY" = "1" ]; then
  echo
  echo "done. review with: git -C $HERE status"
  echo "then commit: git -C $HERE add -A && git -C $HERE commit -m 'sync from dynamo @ <sha>'"
elif [ "$CODE_CHANGED" = "1" ]; then
  echo
  echo "code changes detected. apply with: $0 --apply $DYNAMO_SRC"
  exit 1
else
  echo
  echo "code is up to date with dynamo. (review any NOTE lines above for non-code drift.)"
fi
