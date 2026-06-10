#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# TEMPORARY V1 SYNC ONLY — do not call from CI or automation.
#
# This bridge script keeps the old Dynamo parser source and v1 parity renderer
# available while the frontend-crates parser crate is being prepared for release.
# It is deliberately manual because v2 work lives outside this mirror in
# parsers_v2* and conformance/utils/generate_conformance_table.py.
#
# After Dynamo consumes the released frontend-crates parser crate directly, stop
# using this script for parsers and merge the v1/v2 renderers in this repo.
#
# Usage:
#   scripts/manual-sync-parsers.sh /path/to/dynamo         # dry-run: shows what would change
#   scripts/manual-sync-parsers.sh --apply /path/to/dynamo # apply changes
#
# See PARSERS-V2-MIGRATION-PLAN.md for background, excluded v2 files, and the migration plan.

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

if [ -z "$DYNAMO_SRC" ]; then
  echo "usage: $0 [--apply] /path/to/dynamo" >&2
  exit 1
fi

if [ ! -d "$DYNAMO_SRC/lib/parsers" ]; then
  echo "error: $DYNAMO_SRC does not look like a dynamo checkout (no lib/parsers dir)" >&2
  exit 1
fi

HERE="$(cd "$(dirname "$0")/.." && pwd)"

if [ "$APPLY" = "1" ]; then
  echo "=== applying parser sync from $DYNAMO_SRC ==="
else
  echo "=== dry run: parser sync from $DYNAMO_SRC ==="
  echo "    re-run with --apply to apply"
  echo
fi

CHANGED=0

# --- parsers/src/ ---
echo "--- parsers/src/ ---"
if [ "$APPLY" = "1" ]; then
  rsync -a --delete --checksum "$DYNAMO_SRC/lib/parsers/src/" "$HERE/parsers/src/"
else
  out=$(rsync -a --delete --checksum --dry-run --itemize-changes \
    "$DYNAMO_SRC/lib/parsers/src/" "$HERE/parsers/src/" | grep -E '^[<>c*]' || true)
  if [ -n "$out" ]; then
    echo "$out" | sed 's/^/  /'
    CHANGED=1
  else
    echo "  up to date"
  fi
fi

# --- parsers/tests/ (if present in dynamo) ---
if [ -d "$DYNAMO_SRC/lib/parsers/tests" ]; then
  echo "--- parsers/tests/ ---"
  if [ "$APPLY" = "1" ]; then
    rsync -a --delete --checksum "$DYNAMO_SRC/lib/parsers/tests/" "$HERE/parsers/tests/"
  else
    out=$(rsync -a --delete --checksum --dry-run --itemize-changes \
      "$DYNAMO_SRC/lib/parsers/tests/" "$HERE/parsers/tests/" | grep -E '^[<>c*]' || true)
    if [ -n "$out" ]; then
      echo "$out" | sed 's/^/  /'
      CHANGED=1
    else
      echo "  up to date"
    fi
  fi
fi

# --- conformance/utils/tests/parity/ (whitelisted files only) ---
PH_FILTER=(
  --include='/__init__.py' --include='/common.py' --include='/markup.py'
  --include='/generate_parity_table.py' --include='/parity_table.html.j2'
  --include='/toolcalling/' --include='/reasoning/'
  --include='/toolcalling/__init__.py' --include='/toolcalling/table.py'
  --include='/toolcalling/vllm.py' --include='/toolcalling/sglang.py' --include='/toolcalling/dynamo.py'
  --include='/reasoning/__init__.py' --include='/reasoning/table.py'
  --include='/reasoning/vllm.py' --include='/reasoning/sglang.py' --include='/reasoning/dynamo.py'
  --exclude='*'
)
echo "--- conformance/utils/tests/parity/ ---"
if [ "$APPLY" = "1" ]; then
  mkdir -p "$HERE/conformance/utils/tests/parity"
  rsync -a --delete --checksum "${PH_FILTER[@]}" \
    "$DYNAMO_SRC/tests/parity/" "$HERE/conformance/utils/tests/parity/"
else
  out=$(rsync -a --delete --checksum --dry-run --itemize-changes "${PH_FILTER[@]}" \
    "$DYNAMO_SRC/tests/parity/" "$HERE/conformance/utils/tests/parity/" | grep -E '^[<>c*]' || true)
  if [ -n "$out" ]; then
    echo "$out" | sed 's/^/  /'
    CHANGED=1
  else
    echo "  up to date"
  fi
fi

# --- conformance/utils/lib/parsers/*_CASES.md ---
echo "--- conformance/utils/lib/parsers/ ---"
for cm in TOOLCALLING_CASES.md REASONING_CASES.md; do
  src="$DYNAMO_SRC/lib/parsers/$cm"
  dst="$HERE/conformance/utils/lib/parsers/$cm"
  [ -f "$src" ] || continue
  if [ "$APPLY" = "1" ]; then
    mkdir -p "$(dirname "$dst")"
    rsync -a --checksum "$src" "$dst"
  elif ! diff -q "$src" "$dst" >/dev/null 2>&1; then
    echo "  $cm would change"
    CHANGED=1
  else
    echo "  $cm up to date"
  fi
done

echo

if [ "$APPLY" = "1" ]; then
  echo "done. review with: git -C $HERE diff --stat"
  echo "then verify:       conformance/utils/render_table_v1.sh"
  echo "                   conformance/utils/render_table_v2.sh"
elif [ "$CHANGED" = "1" ]; then
  echo "changes detected. re-run with --apply to apply."
  exit 1
else
  echo "everything up to date."
fi
