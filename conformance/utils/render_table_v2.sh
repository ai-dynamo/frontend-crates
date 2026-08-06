#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# render_table_v2.sh [--dry-run] [--output PATH]
#   Render the conformance matrix to an HTML file
#   (all five tabs: TC batch / TC stream / Reasoning batch / Reasoning stream /
#   Unified). SMG's Rust parsers are run live; no inference engines are needed.

usage() {
  cat <<'EOF'
usage: conformance/utils/render_table_v2.sh [--dry-run] [--output PATH]

Render the v2 conformance matrix to an HTML file.

Options:
  --output PATH   Write to PATH. Relative paths resolve from the repo root.
                  Default: conformance/CONFORMANCE_v2.html
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

OUT="$ROOT/conformance/CONFORMANCE_v2.html"
if [ -n "$OUT_ARG" ]; then
  case "$OUT_ARG" in
    /*) OUT="$OUT_ARG" ;;
    *) OUT="$ROOT/$OUT_ARG" ;;
  esac
fi
if [ "$DRY" = 1 ]; then
  echo "[dry-run] build .stage, capture SMG parser outputs, then render the conformance table > $OUT"
  exit 0
fi
build_stage_conformance
SMG_CAPTURE="$STAGE/smg_capture.json"
SMG_TOOLCALLING_FIXTURES="$STAGE/tests/parity/toolcalling/fixtures" \
SMG_TOOLCALLING_STREAM_FIXTURES="$STAGE/tests/parity/toolcalling/fixtures" \
SMG_REASONING_FIXTURES="$STAGE/tests/parity/reasoning/fixtures" \
SMG_UNIFIED_FIXTURES="$FIXTURES_ROOT/unified/inputs" \
SMG_CAPTURE_OUTPUT="$SMG_CAPTURE" \
  "$CARGO" test --locked -p dynamo-conformance-fixtures-v2 \
    --test smg_capture smg_capture_covers_all_conformance_inputs -- --nocapture
export SMG_CAPTURE_PATH="$SMG_CAPTURE"
mkdir -p "$(dirname "$OUT")"
# Render to a working file, then atomically move it into place. The `>` redirect
# truncates its target for the WHOLE render (~2 min), so anything reading $OUT during
# that window (CI, a live viewer, a verify script) sees a 0-byte / partial file. Writing
# to CONFORMANCE_v2.working.html and mv-ing on success means readers only ever see the
# previous complete file or the new complete one. --output-path stays $OUT so link
# resolution targets the final location (the working file is in the same dir, so hrefs
# are identical); on failure the real file is left untouched.
case "$OUT" in
  *.html) WORK="${OUT%.html}.working.html" ;;
  *)      WORK="$OUT.working" ;;
esac
if ( cd "$STAGE" && PYTHONPATH="$STAGE" python3 tests/parity/generate_conformance_table.py all --html --output-path "$OUT" --artifact-root "$ROOT" ) > "$WORK"; then
  mv -f "$WORK" "$OUT"
  echo "wrote $OUT"
else
  rc=$?
  rm -f "$WORK"
  echo "render failed (exit $rc); left $OUT untouched" >&2
  exit "$rc"
fi
