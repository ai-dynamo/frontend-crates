#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# capture.sh <stream|batch-on-stream|dynamo-stream|dynamo-batch-on-stream|token-ids> [options]
#   Consistent entry point for capturing parser behavior used to refresh v2 fixtures.

DRY=0; args=()
for arg in "$@"; do
  case "$arg" in
    --dry-run|--dryrun) DRY=1 ;;
    *) args+=("$arg") ;;
  esac
done
set -- "${args[@]}"
source "$(dirname "$0")/_common.sh"

usage() {
  local status="${1:-2}"
  cat >&2 <<'USAGE'
usage:
  conformance/utils/capture.sh stream [--vllm-container NAME] [--sglang-container NAME] [--vllm-rust-source PATH] [--work PATH]
  conformance/utils/capture.sh batch-on-stream [--vllm-container NAME] [--sglang-container NAME] [--vllm-rust-source PATH] [--work PATH] [--dynamo-rust-json PATH | --capture-dynamo-rust-json PATH]
  conformance/utils/capture.sh dynamo-stream --fixture PATH [--text] [--output PATH]
  conformance/utils/capture.sh dynamo-batch-on-stream --output PATH
  conformance/utils/capture.sh token-ids
USAGE
  exit "$status"
}

run_capture_driver() {
  if [ "$DRY" = 1 ]; then
    printf '[dry-run]'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

run_cargo_bin() {
  local output=""
  if [ "$1" = "--output" ]; then
    output="$2"
    shift 2
  fi
  if [ "$DRY" = 1 ]; then
    printf '[dry-run] (cd %q && %q run -p dynamo-parsers-v2 --bin' "$ROOT" "$CARGO"
    printf ' %q' "$@"
    if [ -n "$output" ]; then
      printf ' > %q' "$output"
    fi
    printf ')\n'
  elif [ -n "$output" ]; then
    ( cd "$ROOT" && $CARGO run -p dynamo-parsers-v2 --bin "$@" ) > "$output"
  else
    ( cd "$ROOT" && $CARGO run -p dynamo-parsers-v2 --bin "$@" )
  fi
}

target="${1:-}"
[ -n "$target" ] || usage
shift || true

case "$target" in
  stream)
    vllm_container="vllm-localdev"
    sglang_container="sglang-localdev"
    vllm_rust_source=""
    work="${TMPDIR:-/tmp}/capture_stream_$$"
    dynamo_todo="Dynamo parser v2 TC streaming not yet implemented for this family; vLLM/SGLang per-chunk output is the target to match."
    while [ $# -gt 0 ]; do
      case "$1" in
        --vllm-container) vllm_container="$2"; shift 2 ;;
        --sglang-container) sglang_container="$2"; shift 2 ;;
        --vllm-rust-source) vllm_rust_source="$2"; shift 2 ;;
        --work) work="$2"; shift 2 ;;
        --dynamo-todo) dynamo_todo="$2"; shift 2 ;;
        *) usage ;;
      esac
    done
    mkdir -p "$work"
    cmd=(python3 "$TOOLS/capture_driver.py" --mode stream --root "$ROOT" --work "$work" --vllm-container "$vllm_container" --sglang-container "$sglang_container" --dynamo-todo "$dynamo_todo")
    [ -n "$vllm_rust_source" ] && cmd+=(--vllm-rust-source "$vllm_rust_source")
    run_capture_driver "${cmd[@]}"
    ;;
  batch-on-stream)
    vllm_container="vllm-localdev"
    sglang_container="sglang-localdev"
    vllm_rust_source=""
    work="${TMPDIR:-/tmp}/capture_batch_on_stream_$$"
    dynamo_rust_json=""
    capture_dynamo_rust_json=""
    while [ $# -gt 0 ]; do
      case "$1" in
        --vllm-container) vllm_container="$2"; shift 2 ;;
        --sglang-container) sglang_container="$2"; shift 2 ;;
        --vllm-rust-source) vllm_rust_source="$2"; shift 2 ;;
        --work) work="$2"; shift 2 ;;
        --dynamo-rust-json) dynamo_rust_json="$2"; shift 2 ;;
        --capture-dynamo-rust-json) capture_dynamo_rust_json="$2"; shift 2 ;;
        *) usage ;;
      esac
    done
    if [ -n "$dynamo_rust_json" ] && [ -n "$capture_dynamo_rust_json" ]; then
      echo "choose either --dynamo-rust-json or --capture-dynamo-rust-json, not both" >&2
      exit 2
    fi
    mkdir -p "$work"
    if [ -n "$capture_dynamo_rust_json" ]; then
      run_cargo_bin --output "$capture_dynamo_rust_json" record_batch_via_stream
      dynamo_rust_json="$capture_dynamo_rust_json"
    fi
    cmd=(python3 "$TOOLS/capture_driver.py" --mode batch-on-stream --root "$ROOT" --work "$work" --vllm-container "$vllm_container" --sglang-container "$sglang_container")
    [ -n "$vllm_rust_source" ] && cmd+=(--vllm-rust-source "$vllm_rust_source")
    [ -n "$dynamo_rust_json" ] && cmd+=(--dynamo-rust-json "$dynamo_rust_json")
    run_capture_driver "${cmd[@]}"
    ;;
  dynamo-stream)
    fixture=""
    output=""
    text=0
    while [ $# -gt 0 ]; do
      case "$1" in
        --fixture) fixture="$2"; shift 2 ;;
        --output) output="$2"; shift 2 ;;
        --text) text=1; shift ;;
        *) [ -z "$fixture" ] && fixture="$1" && shift || usage ;;
      esac
    done
    [ -n "$fixture" ] || usage
    args=(record_dynamo_stream -- "$fixture")
    [ "$text" = 1 ] && args+=(--text)
    if [ -n "$output" ]; then
      run_cargo_bin --output "$output" "${args[@]}"
    else
      run_cargo_bin "${args[@]}"
    fi
    ;;
  dynamo-batch-on-stream)
    output=""
    while [ $# -gt 0 ]; do
      case "$1" in
        --output) output="$2"; shift 2 ;;
        *) usage ;;
      esac
    done
    [ -n "$output" ] || usage
    run_cargo_bin --output "$output" record_batch_via_stream
    ;;
  token-ids)
    [ $# -eq 0 ] || usage
    run_cargo_bin stamp_stream_token_ids
    ;;
  --help|-h|help)
    usage 0
    ;;
  *)
    usage
    ;;
esac
