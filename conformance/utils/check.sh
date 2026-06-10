#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# check.sh <dynamo|vllm|sglang|all> [batch|stream|all] [--container N|--pip] [--dry-run]
#   Run a parser against the committed fixtures and report pass/fail (read-only).
#     dynamo [batch|stream|all]  Dynamo Rust parser vs expected.dynamo_rust / expected.dynamo (cargo test)
#     vllm   [--container N|--pip]   vLLM Python parser vs expected.vllm_python / expected.vllm
#     sglang [--container N|--pip]   SGLang Python parser vs expected.sglang_python
#     all    [--container-vllm N --container-sglang M] [--allow-peer-failures]
#            dynamo(all) + vllm + sglang. Fails the run on any parser failure;
#            pass --allow-peer-failures (alias --best-effort-peers) to keep the
#            old behavior of reporting peer failures without failing the command.

DRY=0; args=()
while [ $# -gt 0 ]; do case "$1" in --dry-run|--dryrun) DRY=1; shift;; *) args+=("$1"); shift;; esac; done
set -- ${args+"${args[@]}"}
source "$(dirname "$0")/src/_common.sh"

usage() { echo "usage: conformance/utils/check.sh <dynamo|vllm|sglang|all> [batch|stream|all] [--container N|--pip]" >&2; exit 2; }

run_dynamo() {  # $1 = batch|stream|all ; returns non-zero if any target fails
  local targets=() rc=0
  case "${1:-all}" in
    batch)  targets=(parity_toolcalling) ;;
    stream) targets=(parity_toolcalling_stream parity_toolcalling_batch_via_stream) ;;
    all)    targets=(parity_toolcalling parity_toolcalling_stream parity_toolcalling_batch_via_stream) ;;
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

engine="${1:-}"; shift || true
[ -n "$engine" ] || usage
case "$engine" in
  dynamo) run_dynamo "${1:-all}" ;;
  vllm)   run_engine vllm "$@" ;;
  sglang) run_engine sglang "$@" ;;
  all)
    cv=""; cs=""; allow_peer=0; rc=0
    while [ $# -gt 0 ]; do case "$1" in
        --container-vllm)   cv="$2"; shift 2 ;;
        --container-sglang) cs="$2"; shift 2 ;;
        --allow-peer-failures|--best-effort-peers) allow_peer=1; shift ;;
        *) shift ;;
      esac; done
    run_dynamo all || rc=1
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
