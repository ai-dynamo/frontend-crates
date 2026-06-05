# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Shared helpers for the conformance/utils scripts (check_v2.sh / render_table_v2.sh /
# record_v2.sh). Sourced, not executed. Each caller strips --dry-run (sets DRY=1)
# before sourcing this file.
#
# The vendored Python generator/adapters hard-code dynamo's repo layout, so the
# build_stage_* helpers build an ephemeral stage tree that presents it: the package
# is copied so Path(__file__).resolve() stays inside the stage, fixture views
# are copied, and Rust parser source, case docs, and pyproject metadata are
# symlinked.

set -euo pipefail

# conformance/utils/ is two levels below the repo root.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TOOLS="$ROOT/conformance/utils"
STAGE="${STAGE:-$TOOLS/.stage-v2}"
# Override when the default cargo can't build the workspace (edition 2024 /
# resolver "3" needs >= 1.85): CARGO='cargo +1.93.1' conformance/utils/check_v2.sh ...
CARGO="${CARGO:-cargo}"
: "${DRY:=0}"

_build_stage_base() {
  rm -rf "$STAGE"
  mkdir -p "$STAGE/tests" "$STAGE/lib/parsers/src"
  # COPY the vendored python package so resolved __file__ -> REPO_ROOT == $STAGE.
  \cp -Rf "$TOOLS/tests/parity" "$STAGE/tests/parity"
  \cp -f "$TOOLS/tests/__init__.py" "$STAGE/tests/__init__.py"
  ln -s "$ROOT/conformance/reasoning/fixtures"   "$STAGE/tests/parity/reasoning/fixtures"
  # Recorded Dynamo stream-on-batch results for the batch-on-stream tab.
  [ -f "$TOOLS/harmony_batch_stream.json" ] && \cp -f "$TOOLS/harmony_batch_stream.json" "$STAGE/harmony_batch_stream.json"
  ln -s "$ROOT/parsers/src/tool_calling"         "$STAGE/lib/parsers/src/tool_calling"
  ln -s "$TOOLS/lib/parsers/TOOLCALLING_CASES.md"   "$STAGE/lib/parsers/TOOLCALLING_CASES.md"
  ln -s "$TOOLS/lib/parsers/REASONING_CASES.md"     "$STAGE/lib/parsers/REASONING_CASES.md"
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

build_stage_v2() {
  _build_stage_base
  # Keep v2 owned by conformance/utils while presenting it in Dynamo's staged
  # tests/parity layout for imports and template lookup.
  \cp -f "$TOOLS/generate_conformance_table_v2.py" "$STAGE/tests/parity/generate_conformance_table_v2.py"
  \cp -f "$TOOLS/conformance_table_v2.html.j2" "$STAGE/tests/parity/conformance_table_v2.html.j2"
  _copy_toolcalling_v2_fixtures
}

build_stage() {
  echo "build_stage is ambiguous; use build_stage_v1 or build_stage_v2" >&2
  return 2
}
