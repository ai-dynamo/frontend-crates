#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# check.sh <dynamo|vllm|sglang|coverage|status|ci|all> [batch|stream|all] [--container N|--pip] [--dry-run]
#   Run a parser against the committed fixtures and report pass/fail (read-only).
#     dynamo [batch|stream|all]  Dynamo Rust parser vs expected.dynamo_v2 / expected.dynamo_v1 (cargo test)
#     vllm   [--container N|--pip]   vLLM Python parser vs expected.vllm_python / expected.vllm
#     sglang [--container N|--pip]   SGLang Python parser vs expected.sglang_python
#     coverage [--family F ...]      fixture-coverage + marker-registration lint vs
#                                    case-taxonomy.yaml (defaults to --all families)
#     status [--model F ...] [--tab T ...]
#                                    read source resources with the shared JS model, then fail on empty/red selected cells
#     ci     what the conformance-data CI job runs: catalog, coverage, legacy-store,
#            and invariant checks
#            Core gates live in run_ci; CI workflows own native publication and browser qualification.
#     all    [--container-vllm N --container-sglang M] [--allow-peer-failures]
#            dynamo(all) + vllm + sglang + coverage. Fails the run on any parser
#            failure; pass --allow-peer-failures (alias --best-effort-peers) to keep
#            the old behavior of reporting peer failures without failing the command.

DRY=0; args=()
while [ $# -gt 0 ]; do case "$1" in --dry-run|--dryrun) DRY=1; shift;; *) args+=("$1"); shift;; esac; done
set -- ${args+"${args[@]}"}
# Direct readers and dry runs must not eagerly materialize historical captures.
if [ "${1:-}" = status ] || [ "${1:-}" = ci ] || [ "$DRY" = 1 ]; then
  status_root="$(cd "$(dirname "$0")/../.." && pwd)"
fi
if [ "${1:-}" = status ]; then
  shift
  if [ "$DRY" = 1 ]; then
    echo "[dry-run] node $status_root/conformance/viewer/status.js --require-green $*"
    exit 0
  fi
  exec node "$status_root/conformance/viewer/status.js" --root "$status_root" --require-green "$@"
fi
if [ "${1:-}" = ci ] || [ "$DRY" = 1 ]; then
  ROOT="$status_root"
  UTILS="$ROOT/conformance/utils"
  TOOLS="$UTILS/src"
  CARGO="${CARGO:-cargo}"
else
  source "$(dirname "$0")/src/_common.sh"
fi

usage() { echo "usage: conformance/utils/check.sh <dynamo|vllm|sglang|coverage|status|ci|all> [batch|stream|all] [--container N|--pip]" >&2; exit 2; }

run_dynamo() {  # $1 = batch|stream|all ; returns non-zero if any target fails
  local targets=() rc=0
  case "${1:-all}" in
    batch)  targets=(conformance_toolcalling) ;;
    stream) targets=(conformance_toolcalling_stream conformance_toolcalling_batch_via_stream) ;;
    all)    targets=(conformance_toolcalling conformance_toolcalling_stream conformance_toolcalling_batch_via_stream) ;;
    *) usage ;;
  esac
  for t in "${targets[@]}"; do
    if [ "$DRY" = 1 ]; then echo "[dry-run] (cd $ROOT && $CARGO test -p dynamo-conformance-fixtures-v2 --test $t -- --nocapture)"
    else ( cd "$ROOT" && $CARGO test -p dynamo-conformance-fixtures-v2 --test "$t" -- --nocapture ) || rc=1; fi
  done
  return "$rc"
}

run_engine() {  # $1=vllm|sglang  $2..=passthrough (--container N|--pip)
  local impl="$1"; shift
  if [ "$DRY" = 1 ]; then echo "[dry-run] build .stage, then validate $impl against staged toolcalling fixtures $*"; return; fi
  build_stage_conformance
  PYTHONPATH="$STAGE" python3 "$TOOLS/validate.py" --impl "$impl" \
    --fixtures "$STAGE/tests/parity/toolcalling/fixtures" "$@"
}

run_coverage() {  # $@ = passthrough (--family F ...); defaults to --all
  if [ "$DRY" = 1 ]; then echo "[dry-run] python3 $TOOLS/check_family_coverage.py ${*:---all}"; return; fi
  if [ $# -gt 0 ]; then python3 "$TOOLS/check_family_coverage.py" "$@"
  else python3 "$TOOLS/check_family_coverage.py" --all; fi
}

run_ci() {  # the conformance-data CI gate; fail-fast, also runnable locally
  if [ "$DRY" = 1 ]; then
    echo "[dry-run] validate viewer catalog, coverage, legacy store, and selected invariant pytest shard"
    return
  fi
  local shard="${CONFORMANCE_CI_SHARD:-all}"
  case "$shard" in all|fixtures|core|qualification|native-browser|browser|viewer|first-matrix) ;; *) echo "invalid CONFORMANCE_CI_SHARD: $shard" >&2; return 2 ;; esac
  set -e
  if [ "$shard" = fixtures ]; then
    CONFORMANCE_FIXTURES_ROOT="$(python3 "$TOOLS/extract_fixtures.py" --legacy-only)"
    export CONFORMANCE_FIXTURES_ROOT
  fi
  if [ "$shard" = all ] || [ "$shard" = core ]; then
    # Unified readers use the maintained source or qualified publication directly.
    CONFORMANCE_FIXTURES_ROOT="$(python3 "$TOOLS/extract_fixtures.py" --legacy-only)"
    export CONFORMANCE_FIXTURES_ROOT
    python3 "$TOOLS/viewer_catalog.py" --check
    run_coverage
    python3 "$TOOLS/validate_legacy_convention.py"
  fi
  # Keep every test file covered. Selenium modules run on separate CI runners so
  # Chrome does not contend with CPU-heavy shared-model Node checks under xdist.
  # Qualification owns historical parity and receipt proofs; keep negative probes in core.
  case "$shard" in
    fixtures|core)
      local scope=publication
      if [ "$shard" = fixtures ]; then scope=fixtures; fi
      python3 -m pytest "$UTILS/tests" -q --conformance-scope "$scope" --ignore="$UTILS/tests/test_browser_smoke.py" --ignore="$UTILS/tests/test_browser_viewer.py" --ignore="$UTILS/tests/test_native_publication_browser.py" --deselect="conformance/utils/tests/test_viewer_snapshot_parity.py::test_full_corpus_json_snapshots_match_python_and_javascript_resolution" --deselect="conformance/utils/tests/test_unified_taxonomy_covers_corpus.py::test_reference_backfills_require_request_bound_version_receipts" --deselect="conformance/utils/tests/test_model.py::test_shared_javascript_matrix_contracts" --deselect="conformance/utils/tests/test_structural_response_case.py::test_structural_request_is_bound_in_every_retained_dynamo_column"
      ;;
    qualification)
      python3 -m pytest "$UTILS/tests/test_viewer_snapshot_parity.py::test_full_corpus_json_snapshots_match_python_and_javascript_resolution" "$UTILS/tests/test_unified_taxonomy_covers_corpus.py::test_reference_backfills_require_request_bound_version_receipts" "$UTILS/tests/test_model.py::test_shared_javascript_matrix_contracts" "$UTILS/tests/test_structural_response_case.py::test_structural_request_is_bound_in_every_retained_dynamo_column" -vv -rP --durations=25
      ;;
    native-browser) python3 -m pytest "$UTILS/tests/test_native_publication_browser.py" -vv --durations=25 ;;
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
  vllm)   run_engine vllm "$@" ;;
  sglang) run_engine sglang "$@" ;;
  coverage) run_coverage "$@" ;;
  ci)     run_ci "$@" ;;
  all)
    cv=""; cs=""; allow_peer=0; rc=0
    while [ $# -gt 0 ]; do case "$1" in
        --container-vllm)   cv="$2"; shift 2 ;;
        --container-sglang) cs="$2"; shift 2 ;;
        --allow-peer-failures|--best-effort-peers) allow_peer=1; shift ;;
        *) shift ;;
      esac; done
    run_dynamo all || rc=1
    run_coverage || rc=1
    peer_rc=0
    if [ -n "$cv" ]; then run_engine vllm --container "$cv" || peer_rc=1
    else echo "(skipped vllm: pass --container-vllm NAME or 'check.sh vllm --pip')"; fi
    if [ -n "$cs" ]; then run_engine sglang --container "$cs" || peer_rc=1
    else echo "(skipped sglang: pass --container-sglang NAME or 'check.sh sglang --pip')"; fi
    if [ "$peer_rc" = 1 ]; then
      if [ "$allow_peer" = 1 ]; then echo "(peer parser check failed; --allow-peer-failures set, not failing the run)"
      else rc=1; fi
    fi
    exit "$rc"
    ;;
  *) usage ;;
esac
