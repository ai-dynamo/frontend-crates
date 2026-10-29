#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
if [[ "${1:-}" == --help ]]; then
  echo 'usage: ci/rust.sh'
  exit 0
fi
[[ $# == 0 ]] || exit 2
cd "$(dirname "$0")/.."

runtime=$(mktemp -d "${RUNNER_TEMP:-${TMPDIR:-/tmp}}/rust-ci.XXXXXX")
export CONFORMANCE_CAPTURE_REPORT_DIR="$runtime/capture-verification"
mkdir "$CONFORMANCE_CAPTURE_REPORT_DIR"
tests_started=false

run() {
  local label="$1" started=$SECONDS result=0
  shift
  printf '::group::%s\n' "$label"
  "$@" || result=$?
  printf '::endgroup::\n'
  if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
    printf '| %s | %ss | %s |\n' "$label" "$((SECONDS - started))" "$result" >> "$GITHUB_STEP_SUMMARY"
  fi
  return "$result"
}

finish() {
  local result=$?
  trap - EXIT
  if [[ "$tests_started" == true ]]; then
    local args=(--directory "$CONFORMANCE_CAPTURE_REPORT_DIR")
    if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then args+=(--summary "$GITHUB_STEP_SUMMARY"); fi
    python3 conformance/utils/src/report_capture_verification.py "${args[@]}" || {
      if [[ "$result" == 0 ]]; then result=1; fi
    }
  fi
  exit "$result"
}
trap finish EXIT

if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
  printf '\nRust CI\n\n| Task | Duration | Exit code |\n|---|---:|---:|\n' >> "$GITHUB_STEP_SUMMARY"
fi
run 'Unused dependencies' cargo machete
run 'Dependency policy' cargo deny --all-features check bans licenses
run 'Formatting' cargo fmt --all -- --check
export CONFORMANCE_GOLDEN_ROOT="$runtime/unified-golden"
run 'Authored oracle preparation' python3 conformance/utils/src/gen_unified_golden.py --output "$CONFORMANCE_GOLDEN_ROOT"
if [[ -z "${CONFORMANCE_GRAMMAR_PYTHON:-}" ]]; then
  run 'Grammar environment' python3 -m venv "$runtime/structural-tag-venv"
  run 'Grammar dependencies' "$runtime/structural-tag-venv/bin/pip" install --no-deps -r conformance/utils/structural-tag-requirements.in
  export CONFORMANCE_GRAMMAR_PYTHON="$runtime/structural-tag-venv/bin/python"
fi
run 'Structural-tag contracts' cargo test --locked -p dynamo-parsers-v2 --test structural_tag_xgrammar -- --ignored --nocapture

# Publish receipts after test failures, but never when an earlier check skipped the tests.
tests_started=true
if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
  printf 'capture_report_directory=%s\n' "$CONFORMANCE_CAPTURE_REPORT_DIR" >> "$GITHUB_OUTPUT"
fi
run 'Workspace tests' cargo nextest run --workspace --all-targets --locked
run 'Optimized Python grammar' env PYTHONOPTIMIZE=1 cargo test --workspace --locked --test unified_parity structural_response_case_checks_the_schema_before_parsing
# Video tests use libvpx for CPU decoding on runners without an NVIDIA GPU.
run 'Video dependencies' sudo apt-get update
run 'Video dependencies' sudo apt-get install -y --no-install-recommends libvpx9
run 'Multimodal features' cargo test -p dynamo-multimodal --all-targets --all-features --locked
run 'Parser recovery work bounds' cargo test --locked -p dynamo-parsers-v2 --features test-utils --test gemma4_native_recovery
# nextest excludes doctests, so they need their own invocation.
run 'Doctests' cargo test --workspace --doc --locked
run 'Protocol schema' cargo test -p dynamo-protocols --features protocol-schema --locked
run 'Clippy' cargo clippy --workspace --all-targets --all-features --locked -- -D warnings
run 'Publishability' cargo publish --workspace --dry-run --locked
