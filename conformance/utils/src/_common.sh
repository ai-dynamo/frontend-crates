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
# Fixture trees are cached in ~/.cache/dynamo/conformance-fixtures/ (downloaded
# from HuggingFace on first use). Run download_fixtures.py to populate.
FIXTURES_ROOT="${XDG_CACHE_HOME:-$HOME/.cache}/dynamo/conformance-fixtures"
if [ ! -d "$FIXTURES_ROOT/toolcalling" ]; then
  echo "error: fixtures not found at $FIXTURES_ROOT" >&2
  echo "  Run: export HF_TOKEN=<read-token>" >&2
  echo "       python3 $TOOLS/download_fixtures.py" >&2
  exit 1
fi
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
  if [ -d "$FIXTURES_ROOT/toolcalling/fixtures-batch-on-stream-v2" ]; then
    mkdir -p "$STAGE/tests/parity/toolcalling"
    \cp -Rf "$FIXTURES_ROOT/toolcalling/fixtures-batch-on-stream-v2" \
      "$STAGE/tests/parity/toolcalling/fixtures-batch-on-stream-v2"
  fi
  ln -s "$ROOT/parsers/v1/src/tool_calling"      "$STAGE/lib/parsers/src/tool_calling"
  ln -s "$UTILS/lib/parsers/TOOLCALLING_CASES.md"   "$STAGE/lib/parsers/TOOLCALLING_CASES.md"
  ln -s "$UTILS/lib/parsers/TOOLCALLING_STREAMING_V2_CASES.md" "$STAGE/lib/parsers/TOOLCALLING_STREAMING_V2_CASES.md"
  ln -s "$UTILS/lib/parsers/REASONING_CASES.md"     "$STAGE/lib/parsers/REASONING_CASES.md"
  ln -s "$TOOLS/pyproject.stub.toml"                "$STAGE/pyproject.toml"
  [ -e "$ROOT/.git" ] && ln -s "$ROOT/.git" "$STAGE/.git" || true
}

# Fixtures are versioned per impl: fixtures/inputs/ (shared inputs) + fixtures/<impl>-<version>/
# (lowest version = full anchor, higher = changed-only overlays). Resolve the pinned
# version of each impl into the flat tree the readers/renderers expect. Peer versions
# come from pyproject.stub.toml; the Dynamo version from parsers/Cargo.toml.
_resolve_toolcalling_fixtures() {
  local out="$1"; mkdir -p "$out"
  local vllm_v sglang_v dynamo_v
  vllm_v=$(grep -oE 'vllm\[[^]]*\]==[^"]+' "$TOOLS/pyproject.stub.toml" | sed -E 's/.*==//')
  sglang_v=$(grep -oE 'sglang\[[^]]*\]==[^"]+' "$TOOLS/pyproject.stub.toml" | sed -E 's/.*==//')
  dynamo_v=$(grep -m1 -E '^version = ' "$ROOT/parsers/v1/Cargo.toml" | sed -E 's/.*"([^"]+)".*/\1/')
  python3 "$TOOLS/resolve_fixtures.py" \
    --fixtures-root "$FIXTURES_ROOT/toolcalling/fixtures-batch-v1" \
    --out "$out" --select "dynamo-${dynamo_v}" "vllm-${vllm_v}" "sglang-${sglang_v}"
}

_copy_toolcalling_v1_fixtures() {
  _resolve_toolcalling_fixtures "$STAGE/tests/parity/toolcalling/fixtures"
}

_copy_toolcalling_v2_fixtures() {
  # v2 reads v1 batch fixtures, then replaces TC stream with v2 per-chunk fixtures.
  # Resolve the versioned v1 corpus, then drop its stream fixtures so only batch remains.
  local dst="$STAGE/tests/parity/toolcalling/fixtures"
  _resolve_toolcalling_fixtures "$dst"
  find "$dst" -name 'TOOLCALLING.stream*.yaml' -delete
  # The stream-v2 corpus is versioned like the batch corpus (no unversioned anchor):
  # inputs/ (shared per-chunk delta_text) + <impl>-<version>/ (per-impl expected;
  # lowest version = full anchor, higher = changed-only). Resolve the PINNED (latest)
  # peer versions into the flat tree the renderer expects; single-version impls
  # (dynamo_rust, vllm_rust) default to their lowest. The generator re-resolves each
  # version for the compare model.
  local sv2="$FIXTURES_ROOT/toolcalling/fixtures-stream-v2"
  if [ -d "$sv2" ]; then
    local vllm_v sglang_v tmp family f
    vllm_v=$(grep -oE 'vllm\[[^]]*\]==[^"]+' "$TOOLS/pyproject.stub.toml" | sed -E 's/.*==//')
    sglang_v=$(grep -oE 'sglang\[[^]]*\]==[^"]+' "$TOOLS/pyproject.stub.toml" | sed -E 's/.*==//')
    tmp="$(mktemp -d)"
    python3 "$TOOLS/resolve_stream_fixtures.py" \
      --fixtures-root "$sv2" --out "$tmp" \
      --select "vllm_python-${vllm_v}" "sglang_python-${sglang_v}"
    for f in "$tmp"/*/TOOLCALLING.stream*.yaml; do
      [ -f "$f" ] || continue
      family="$(basename "$(dirname "$f")")"
      mkdir -p "$dst/$family"
      \cp -f "$f" "$dst/$family/"
    done
    rm -rf "$tmp"
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
