# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Shared helpers for the conformance/utils scripts. Sourced, not executed.
# Each caller strips --dry-run (sets DRY=1)
# before sourcing this file.
#
# The vendored Python generator/adapters hard-code dynamo's repo layout, so the
# build_stage_* helpers build an ephemeral stage tree that presents it: the package
# is copied so Path(__file__).resolve() stays inside the stage, fixture views
# are copied, and Rust parser source, case docs, and pyproject metadata are
# symlinked.

set -euo pipefail

# conformance/utils/src/ (internal modules) is three levels below the repo root.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export FRONTEND_CRATES_ROOT="$ROOT"
# tests/ and lib/ stay at conformance/utils/ (Dynamo-sync targets); the rest is in src/.
UTILS="$ROOT/conformance/utils"
TOOLS="$ROOT/conformance/utils/src"
# Ephemeral build tree stays at conformance/utils/.stage (UTILS), not inside src/,
# so CI and .gitignore find it where they always have.
STAGE="${STAGE:-$UTILS/.stage}"
# Override when the default cargo can't build the workspace (edition 2024 /
# resolver "3" needs >= 1.85): CARGO='cargo +1.93.1' conformance/utils/check.sh ...
CARGO="${CARGO:-cargo}"
: "${DRY:=0}"

_build_stage_base() {
  rm -rf "$STAGE"
  mkdir -p "$STAGE/tests" "$STAGE/lib/parsers/src"
  # COPY the vendored python package so resolved __file__ -> REPO_ROOT == $STAGE.
  \cp -Rf "$UTILS/tests/parity" "$STAGE/tests/parity"
  \cp -f "$UTILS/tests/__init__.py" "$STAGE/tests/__init__.py"
  ln -s "$ROOT/conformance/reasoning/fixtures"   "$STAGE/tests/parity/reasoning/fixtures"
  # Recorded Dynamo parser v2 stream-on-batch fixture overlay.
  if [ -d "$ROOT/conformance/toolcalling/fixtures-batch-on-stream-v2" ]; then
    mkdir -p "$STAGE/tests/parity/toolcalling"
    \cp -Rf "$ROOT/conformance/toolcalling/fixtures-batch-on-stream-v2" \
      "$STAGE/tests/parity/toolcalling/fixtures-batch-on-stream-v2"
  fi
  ln -s "$ROOT/parsers/src/tool_calling"         "$STAGE/lib/parsers/src/tool_calling"
  ln -s "$UTILS/lib/parsers/TOOLCALLING_CASES.md"   "$STAGE/lib/parsers/TOOLCALLING_CASES.md"
  ln -s "$UTILS/lib/parsers/TOOLCALLING_STREAMING_V2_CASES.md" "$STAGE/lib/parsers/TOOLCALLING_STREAMING_V2_CASES.md"
  ln -s "$UTILS/lib/parsers/REASONING_CASES.md"     "$STAGE/lib/parsers/REASONING_CASES.md"
  ln -s "$TOOLS/pyproject.stub.toml"                "$STAGE/pyproject.toml"
  [ -e "$ROOT/.git" ] && ln -s "$ROOT/.git" "$STAGE/.git" || true
}

_copy_toolcalling_v1_fixtures() {
  mkdir -p "$STAGE/tests/parity/toolcalling/fixtures"
  \cp -Rf "$ROOT/conformance/toolcalling/fixtures/." "$STAGE/tests/parity/toolcalling/fixtures/"
}

_copy_toolcalling_v2_fixtures() {
  # v2 reads v1 batch fixtures, then replaces TC stream with v2 per-chunk fixtures.
  mkdir -p "$STAGE/tests/parity/toolcalling/fixtures"
  for family_dir in "$ROOT/conformance/toolcalling/fixtures"/*/; do
    family="$(basename "$family_dir")"
    mkdir -p "$STAGE/tests/parity/toolcalling/fixtures/$family"
    for f in "$family_dir"TOOLCALLING.batch*.yaml; do
      [ -f "$f" ] && \cp -f "$f" "$STAGE/tests/parity/toolcalling/fixtures/$family/"
    done
  done
  if [ -d "$ROOT/conformance/toolcalling/fixtures-stream-v2" ]; then
    for family_dir in "$ROOT/conformance/toolcalling/fixtures-stream-v2"/*/; do
      [ -d "$family_dir" ] || continue
      family="$(basename "$family_dir")"
      mkdir -p "$STAGE/tests/parity/toolcalling/fixtures/$family"
      for f in "$family_dir"TOOLCALLING.stream*.yaml; do
        [ -f "$f" ] && \cp -f "$f" "$STAGE/tests/parity/toolcalling/fixtures/$family/"
      done
    done
  fi
}

build_stage_v1() {
  _build_stage_base
  _copy_toolcalling_v1_fixtures
}

build_stage_conformance() {
  _build_stage_base
  # Keep the current conformance harness owned by conformance/utils while presenting
  # it in Dynamo's staged tests/parity layout for imports and template lookup.
  \cp -f "$TOOLS/generate_conformance_table.py" "$STAGE/tests/parity/generate_conformance_table.py"
  \cp -f "$TOOLS/impls.py" "$STAGE/tests/parity/impls.py"
  \cp -f "$TOOLS/markers.py" "$STAGE/tests/parity/markers.py"
  \cp -f "$TOOLS/fixtures.py" "$STAGE/tests/parity/fixtures.py"
  \cp -f "$TOOLS/conformance_table.html.j2" "$STAGE/tests/parity/conformance_table.html.j2"
  # Static CSS/JS (audit B7) inlined into the page at render time.
  mkdir -p "$STAGE/tests/parity/assets"
  \cp -f "$TOOLS/assets/conformance.css" "$STAGE/tests/parity/assets/conformance.css"
  \cp -f "$TOOLS/assets/conformance.js" "$STAGE/tests/parity/assets/conformance.js"
  _copy_toolcalling_v2_fixtures
}

build_stage() {
  echo "build_stage is ambiguous; use build_stage_v1 or build_stage_conformance" >&2
  return 2
}
