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
#            Add/change conformance gates in run_ci below — never in .github/workflows.
#     all    [--container-vllm N --container-sglang M] [--allow-peer-failures]
#            dynamo(all) + vllm + sglang + coverage. Fails the run on any parser
#            failure; pass --allow-peer-failures (alias --best-effort-peers) to keep
#            the old behavior of reporting peer failures without failing the command.

DRY=0; args=()
while [ $# -gt 0 ]; do case "$1" in --dry-run|--dryrun) DRY=1; shift;; *) args+=("$1"); shift;; esac; done
set -- ${args+"${args[@]}"}
# Status is a direct resource reader; sourcing the legacy helpers would extract
# fixtures before this command even dispatches.
if [ "${1:-}" = status ]; then
  status_root="$(cd "$(dirname "$0")/../.." && pwd)"
  shift
  if [ "$DRY" = 1 ]; then
    echo "[dry-run] node $status_root/conformance/viewer/status.js --require-green $*"
    exit 0
  fi
  exec node "$status_root/conformance/viewer/status.js" --root "$status_root" --require-green "$@"
fi
source "$(dirname "$0")/src/_common.sh"

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
  case "$shard" in all|core|browser|viewer|first-matrix) ;; *) echo "invalid CONFORMANCE_CI_SHARD: $shard" >&2; return 2 ;; esac
  set -e
  if [ "$shard" = all ] || [ "$shard" = core ]; then
    python3 "$TOOLS/viewer_catalog.py" --check
    run_coverage
    python3 "$TOOLS/validate_legacy_convention.py"
  fi
  # Keep every test file covered. Selenium modules run on separate CI runners so
  # Chrome does not contend with CPU-heavy shared-model Node checks under xdist.
  case "$shard" in
    core) python3 -m pytest "$UTILS/tests" -q --ignore="$UTILS/tests/test_browser_smoke.py" --ignore="$UTILS/tests/test_browser_viewer.py" ;;
    browser) python3 -m pytest "$UTILS/tests/test_browser_smoke.py" -k 'not first_visible_matrix_matches_transpose_url and not schema_popup_opens_and_fits_across_themes_viewports_and_transpose and not popup_headings_include_compact_references' -vv --durations=25 ;;
    first-matrix) python3 -m pytest "$UTILS/tests/test_browser_smoke.py" -k first_visible_matrix_matches_transpose_url -vv --durations=25 ;;
    viewer)
      python3 -m pytest "$UTILS/tests/test_browser_viewer.py" -vv --durations=25
      python3 -m pytest "$UTILS/tests/test_browser_smoke.py" -k 'schema_popup_opens_and_fits_across_themes_viewports_and_transpose or popup_headings_include_compact_references' -vv --durations=25
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
