#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Unified parser tests and publication qualification share the source checkout.
usage() { echo "usage: conformance/utils/check.sh <dynamo|coverage|status|ci|all> [--dry-run]" >&2; exit "${1:-2}"; }
if [ "${1:-}" = --help ] || [ "${1:-}" = -h ]; then usage 0; fi

DRY=0; args=()
while [ $# -gt 0 ]; do case "$1" in --dry-run|--dryrun) DRY=1; shift;; *) args+=("$1"); shift;; esac; done
set -- ${args+"${args[@]}"}
status_root="$(cd "$(dirname "$0")/../.." && pwd)"
ROOT="$status_root"
UTILS="$ROOT/conformance/utils"
TOOLS="$UTILS/src"
CARGO="${CARGO:-cargo}"
if [ "${1:-}" = status ]; then
  shift
  if [ "$DRY" = 1 ]; then
    echo "[dry-run] node $status_root/conformance/viewer/qualification.js status --require-green $*"
    exit 0
  fi
  exec node "$status_root/conformance/viewer/qualification.js" status --root "$status_root" --require-green "$@"
fi
run_dynamo() {
  if [ "$DRY" = 1 ]; then
    echo "[dry-run] $CARGO test --locked -p dynamo-conformance-fixtures-v2 --test unified_parity --test unified_render --test unified_schema_roundtrip"
    return
  fi
  (cd "$ROOT" && "$CARGO" test --locked -p dynamo-conformance-fixtures-v2 --test unified_parity --test unified_render --test unified_schema_roundtrip)
}

run_coverage() {
  if [ "$DRY" = 1 ]; then
    echo "[dry-run] python3 $TOOLS/check_family_coverage.py ${*:---all}"
    return
  fi
  if [ $# -gt 0 ]; then python3 "$TOOLS/check_family_coverage.py" "$@"
  else python3 "$TOOLS/check_family_coverage.py" --all; fi
}

run_ci() {  # the conformance-data CI gate; fail-fast, also runnable locally
  if [ "$DRY" = 1 ]; then
    echo "[dry-run] validate Unified viewer catalog and selected invariant pytest shard"
    return
  fi
  local shard="${CONFORMANCE_CI_SHARD:-all}"
  case "$shard" in all|fixtures|core|qualification|native-browser|browser|viewer|first-matrix) ;; *) echo "invalid CONFORMANCE_CI_SHARD: $shard" >&2; return 2 ;; esac
  set -e
  if [ "$shard" = all ] || [ "$shard" = core ]; then
    python3 "$TOOLS/viewer_catalog.py" --check
  fi
  # Keep every test file covered. Selenium modules run on separate CI runners so
  # Chrome does not contend with CPU-heavy shared-model Node checks under xdist.
  # Qualification owns historical parity and receipt proofs; keep negative probes in core.
  case "$shard" in
    fixtures|core)
      local scope=publication
      if [ "$shard" = fixtures ]; then scope=fixtures; fi
      python3 -m pytest "$UTILS/tests" -q --conformance-scope "$scope" --ignore="$UTILS/tests/test_browser_smoke.py" --ignore="$UTILS/tests/test_browser_viewer.py" --ignore="$UTILS/tests/test_native_publication_browser.py" --ignore="$UTILS/tests/test_capture_split_browser.py" --ignore="$UTILS/tests/test_browser_download_scheduler.py" --deselect="conformance/utils/tests/test_viewer_snapshot_parity.py::test_bundled_history_json_snapshots_match_original_yaml" --deselect="conformance/utils/tests/test_viewer_snapshot_parity.py::test_full_corpus_json_snapshots_match_python_and_javascript_resolution" --deselect="conformance/utils/tests/test_unified_taxonomy_covers_corpus.py::test_reference_backfills_require_request_bound_version_receipts" --deselect="conformance/utils/tests/test_model.py::test_shared_javascript_matrix_contracts" --deselect="conformance/utils/tests/test_structural_response_case.py::test_structural_request_is_bound_in_every_retained_dynamo_column"
      ;;
    qualification)
      python3 -m pytest "$UTILS/tests/test_viewer_snapshot_parity.py::test_bundled_history_json_snapshots_match_original_yaml" "$UTILS/tests/test_viewer_snapshot_parity.py::test_full_corpus_json_snapshots_match_python_and_javascript_resolution" "$UTILS/tests/test_unified_taxonomy_covers_corpus.py::test_reference_backfills_require_request_bound_version_receipts" "$UTILS/tests/test_model.py::test_shared_javascript_matrix_contracts" "$UTILS/tests/test_structural_response_case.py::test_structural_request_is_bound_in_every_retained_dynamo_column" -vv -rP --durations=25
      ;;
    native-browser) python3 -m pytest "$UTILS/tests/test_native_publication_browser.py" "$UTILS/tests/test_capture_split_browser.py" "$UTILS/tests/test_browser_download_scheduler.py" -vv --durations=25 ;;
    browser) python3 -m pytest "$UTILS/tests/test_browser_smoke.py" -k 'not first_visible_matrix_matches_transpose_url and not schema_popup_opens_and_fits_across_themes_viewports_and_transpose and not popup_headings_include_compact_references' -vv --durations=25 ;;
    first-matrix) python3 -m pytest "$UTILS/tests/test_browser_smoke.py" -k first_visible_matrix_matches_transpose_url -vv --durations=25 ;;
    viewer)
      python3 -m pytest "$UTILS/tests/test_browser_viewer.py" "$UTILS/tests/test_browser_smoke.py::test_schema_popup_opens_and_fits_across_themes_viewports_and_transpose" "$UTILS/tests/test_browser_smoke.py::test_popup_headings_include_compact_references" -vv --durations=25
      ;;
    all) python3 -m pytest "$UTILS/tests" -q ;;
  esac
}

engine="${1:-}"; shift || true
[ -n "$engine" ] || usage
case "$engine" in
  dynamo) run_dynamo "${1:-all}" ;;
  vllm) echo "Legacy conformance retired; use capture_vllm_unified.py" >&2; exit 2 ;;
  sglang) echo "Legacy conformance retired; use capture_sglang_unified.py" >&2; exit 2 ;;
  coverage) run_coverage "$@" ;;
  ci)     run_ci "$@" ;;
  all) run_dynamo ;;
  *) usage ;;
esac
