#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Parity harness orchestrator. Runs the parity-matrix generator and the
# cross-impl conformance lanes against the vendored fixtures.
#
#   run.sh table                      render PARITY.html from the fixtures (no engines)
#   run.sh dynamo                     Rust parser vs expected.dynamo (cargo test)
#   run.sh vllm    [--container N|--pip]   vLLM parser vs expected.vllm
#   run.sh sglang  [--container N|--pip]   SGLang parser vs expected.sglang
#   run.sh all     [--container-vllm N --container-sglang M]   dynamo + vllm + sglang + table
#
# The Python generator/adapters are vendored verbatim from dynamo (under
# parity-harness/tests/parity) and hard-code dynamo's repo layout. We build an
# ephemeral .stage/ that presents that layout — the package is COPIED (so
# Path(__file__).resolve() stays inside the stage) and the data (fixtures, rust
# parser source, case docs, pyproject) is symlinked to this repo's real paths.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"   # frontend-crates repo root
PH="$ROOT/parity-harness"
STAGE="$PH/.stage"

build_stage() {
  rm -rf "$STAGE"
  mkdir -p "$STAGE/tests" "$STAGE/lib/parsers/src"
  # COPY the vendored python package so resolved __file__ → REPO_ROOT == $STAGE.
  cp -R "$PH/tests/parity" "$STAGE/tests/parity"
  cp "$PH/tests/__init__.py" "$STAGE/tests/__init__.py"
  # SYMLINK the data the tools read by REPO_ROOT-relative path.
  ln -s "$ROOT/conformance/toolcalling/fixtures" "$STAGE/tests/parity/toolcalling/fixtures"
  ln -s "$ROOT/conformance/reasoning/fixtures"   "$STAGE/tests/parity/reasoning/fixtures"
  ln -s "$ROOT/parsers/src/tool_calling"         "$STAGE/lib/parsers/src/tool_calling"
  ln -s "$PH/lib/parsers/TOOLCALLING_CASES.md"   "$STAGE/lib/parsers/TOOLCALLING_CASES.md"
  ln -s "$PH/lib/parsers/REASONING_CASES.md"     "$STAGE/lib/parsers/REASONING_CASES.md"
  ln -s "$PH/pyproject.stub.toml"                "$STAGE/pyproject.toml"
  # so the generator's git-SHA header resolves to this repo
  [ -e "$ROOT/.git" ] && ln -s "$ROOT/.git" "$STAGE/.git" || true
}

lane_table() {
  build_stage
  ( cd "$STAGE" && PYTHONPATH="$STAGE" python3 tests/parity/generate_parity_table.py all --html ) > "$PH/PARITY.html"
  echo "wrote $PH/PARITY.html"
}

lane_dynamo() {
  ( cd "$ROOT" && cargo test -p dynamo-conformance --test parity_toolcalling -- --nocapture )
}

lane_engine() {  # $1=impl  $2..=passthrough mode flags (--container N | --pip)
  local impl="$1"; shift
  build_stage
  PYTHONPATH="$STAGE" python3 "$PH/validate.py" --impl "$impl" \
    --fixtures "$STAGE/tests/parity/toolcalling/fixtures" "$@"
}

cmd="${1:-}"; shift || true
case "$cmd" in
  table)  lane_table ;;
  dynamo) lane_dynamo ;;
  vllm)   lane_engine vllm "$@" ;;
  sglang) lane_engine sglang "$@" ;;
  all)
    # parse optional --container-vllm / --container-sglang
    cv=""; cs=""
    while [ $# -gt 0 ]; do
      case "$1" in
        --container-vllm)   cv="$2"; shift 2 ;;
        --container-sglang) cs="$2"; shift 2 ;;
        *) shift ;;
      esac
    done
    lane_dynamo
    if [ -n "$cv" ]; then lane_engine vllm --container "$cv" || true
    else echo "(skipped vllm: pass --container-vllm NAME or run 'vllm --pip')"; fi
    if [ -n "$cs" ]; then lane_engine sglang --container "$cs" || true
    else echo "(skipped sglang: pass --container-sglang NAME or run 'sglang --pip')"; fi
    lane_table
    ;;
  *)
    grep -E '^#' "$0" | sed 's/^# \{0,1\}//'
    exit 2 ;;
esac
