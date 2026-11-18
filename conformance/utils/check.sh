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
    echo "[dry-run] node $status_root/conformance/viewer/qualification.js status $*"
    exit 0
  fi
  exec node "$status_root/conformance/viewer/qualification.js" status --root "$status_root" "$@"
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
  python3 "$TOOLS/ci_plan.py" --run "${CONFORMANCE_CI_SHARD:-all}"
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
