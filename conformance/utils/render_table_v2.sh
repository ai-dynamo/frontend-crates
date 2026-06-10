#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# render_table_v2.sh [--dry-run] [--output PATH]
#   Render the conformance matrix to an HTML file
#   (all four tabs: TC batch / TC stream / TC batch-on-stream / Reasoning).
#   No engines needed.

usage() {
  cat <<'EOF'
usage: conformance/utils/render_table_v2.sh [--dry-run] [--output PATH]

Render the v2 conformance matrix to an HTML file.

Options:
  --output PATH   Write to PATH. Relative paths resolve from the repo root.
                  Default: conformance/CONFORMANCE.html
  --dry-run       Print what would run.
  --help          Show this help.
EOF
}

DRY=0
OUT_ARG=""
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run|--dryrun)
      DRY=1
      shift
      ;;
    --output)
      if [ $# -lt 2 ]; then
        echo "error: --output requires a path" >&2
        exit 2
      fi
      OUT_ARG="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done
source "$(dirname "$0")/src/_common.sh"

OUT="$ROOT/conformance/CONFORMANCE.html"
if [ -n "$OUT_ARG" ]; then
  case "$OUT_ARG" in
    /*) OUT="$OUT_ARG" ;;
    *) OUT="$ROOT/$OUT_ARG" ;;
  esac
fi
if [ "$DRY" = 1 ]; then
  echo "[dry-run] build .stage, then render the conformance table > $OUT"
  exit 0
fi
build_stage_conformance
mkdir -p "$(dirname "$OUT")"
( cd "$STAGE" && PYTHONPATH="$STAGE" python3 tests/parity/generate_conformance_table.py all --html --output-path "$OUT" --artifact-root "$ROOT" ) > "$OUT"
echo "wrote $OUT"
