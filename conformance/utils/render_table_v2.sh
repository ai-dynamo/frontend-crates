#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# render_table_v2.sh [--dry-run] [--output PATH] [web-link options]
#   Render the conformance matrix to an HTML file
#   (all four tabs: TC batch / TC stream / TC batch-on-stream / Reasoning).
#   No engines needed.

usage() {
  cat <<'EOF'
usage: conformance/utils/render_table_v2.sh [--dry-run] [--output PATH] [web-link options]

  Render the v2 conformance matrix to an HTML file and write a sibling status JSON.

Options:
  --output PATH   Write to PATH. Relative paths resolve from the repo root.
                  Default: conformance/CONFORMANCE_v2.html
  --github-repository OWNER/REPO
                  Repository used for immutable GitHub source links.
  --github-revision SHA
                  Full 40-character commit SHA used for source links.
  --fixture-base-url URL
                  HTTPS base URL for published fixture YAMLs.
                  All three web-link options must be supplied together.
  --dry-run       Print what would run.
  --help          Show this help.
EOF
}

DRY=0
OUT_ARG=""
GITHUB_REPOSITORY=""
GITHUB_REVISION=""
FIXTURE_BASE_URL=""
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
    --github-repository)
      [ $# -ge 2 ] || { echo "error: --github-repository requires a value" >&2; exit 2; }
      GITHUB_REPOSITORY="$2"
      shift 2
      ;;
    --github-revision)
      [ $# -ge 2 ] || { echo "error: --github-revision requires a value" >&2; exit 2; }
      GITHUB_REVISION="$2"
      shift 2
      ;;
    --fixture-base-url)
      [ $# -ge 2 ] || { echo "error: --fixture-base-url requires a value" >&2; exit 2; }
      FIXTURE_BASE_URL="$2"
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

if [ -n "$GITHUB_REPOSITORY$GITHUB_REVISION$FIXTURE_BASE_URL" ] &&
   { [ -z "$GITHUB_REPOSITORY" ] || [ -z "$GITHUB_REVISION" ] || [ -z "$FIXTURE_BASE_URL" ]; }; then
  echo "error: --github-repository, --github-revision, and --fixture-base-url must be supplied together" >&2
  exit 2
fi
if [ -n "$GITHUB_REPOSITORY" ]; then
  if [[ ! "$GITHUB_REPOSITORY" =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$ ]]; then
    echo "error: invalid --github-repository: $GITHUB_REPOSITORY" >&2
    exit 2
  fi
  if [[ ! "$GITHUB_REVISION" =~ ^[0-9a-fA-F]{40}$ ]]; then
    echo "error: --github-revision must be a full 40-character commit SHA" >&2
    exit 2
  fi
  if [[ ! "$FIXTURE_BASE_URL" =~ ^https://[^?#]+$ ]]; then
    echo "error: --fixture-base-url must be an HTTPS URL without query or fragment" >&2
    exit 2
  fi
fi
source "$(dirname "$0")/src/_common.sh"

OUT="$ROOT/conformance/CONFORMANCE_v2.html"
if [ -n "$OUT_ARG" ]; then
  case "$OUT_ARG" in
    /*) OUT="$OUT_ARG" ;;
    *) OUT="$ROOT/$OUT_ARG" ;;
  esac
fi
if [ "$DRY" = 1 ]; then
  echo "[dry-run] build .stage, then render the conformance table > $OUT"
  if [ -n "$GITHUB_REPOSITORY" ]; then
    echo "[dry-run] link source to $GITHUB_REPOSITORY@$GITHUB_REVISION and fixtures under $FIXTURE_BASE_URL"
  fi
  exit 0
fi
build_stage_conformance
mkdir -p "$(dirname "$OUT")"
# Keep both published files untouched until generation and validation succeed.
# Each invocation owns its temporary files; the subshell preserves the stage EXIT trap.
(
WORK=""
STATUS_WORK=""
trap 'rm -f -- "$WORK" "$STATUS_WORK"' EXIT
WORK=$(mktemp "$OUT.XXXXXX")
case "$OUT" in
  *.html) STATUS="${OUT%.html}.json" ;;
  *)      STATUS="$OUT.status.json" ;;
esac
STATUS_WORK=$(mktemp "$STATUS.XXXXXX")
RENDER_ARGS=(
  all --html
  --output-path "$OUT"
  --artifact-root "$ROOT"
)
if [ -n "$GITHUB_REPOSITORY" ]; then
  RENDER_ARGS+=(
    --github-repository "$GITHUB_REPOSITORY"
    --github-revision "$GITHUB_REVISION"
    --fixture-base-url "$FIXTURE_BASE_URL"
  )
fi
if ( cd "$STAGE" && PYTHONPATH="$STAGE" python3 tests/parity/generate_conformance_table.py "${RENDER_ARGS[@]}" ) > "$WORK"; then
  :
else
  rc=$?
  echo "ERROR: render failed (exit $rc); left $OUT and $STATUS untouched" >&2
  exit "$rc"
fi
if python3 "$TOOLS/validate_conformance_status.py" \
    --html "$WORK" --report-path "$OUT" --status-path "$STATUS_WORK" --summary-only \
    --unified-fixtures "$FIXTURES_SNAP/unified"; then
  chmod a+r "$WORK" "$STATUS_WORK"
  \mv -f "$WORK" "$OUT"
  \mv -f "$STATUS_WORK" "$STATUS"
  echo "wrote $OUT"
  echo "wrote $STATUS"
else
  rc=$?
  echo "ERROR: validation failed (exit $rc); left $OUT and $STATUS untouched" >&2
  exit "$rc"
fi
)
