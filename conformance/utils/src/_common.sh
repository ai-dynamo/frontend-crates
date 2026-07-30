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
# Fixture trees are cached in ~/.cache/dynamo/conformance-fixtures/ (extracted
# from the in-repo LFS shard store via extract_fixtures.py). Run it every time,
# not only when the cache is empty: it exits instantly on a cache hit, and it
# re-extracts when the committed manifest pin moved — otherwise a render after
# pulling new shards would silently use a stale snapshot.
FIXTURES_ROOT="${XDG_CACHE_HOME:-$HOME/.cache}/dynamo/conformance-fixtures"
# extract_fixtures prints THIS manifest's snapshot dir on stdout. Point readers at
# that exact dir, NOT the shared `<cache>/toolcalling` symlink: sibling checkouts
# pinning a different snapshot race to repoint that symlink, so a render could read
# an older snapshot mid-flight (e.g. one missing dynamo_v2-0.1.22). Reading the
# pinned snapshot dir directly is immune to that race.
FIXTURES_SNAP=$(python3 "$TOOLS/extract_fixtures.py" 2>/dev/null | tail -1)
if [ -z "$FIXTURES_SNAP" ] || [ ! -d "$FIXTURES_SNAP/toolcalling" ]; then
  # Fall back to a plain extract (and the symlink) if the snapshot path is unusable.
  python3 "$TOOLS/extract_fixtures.py" >/dev/null || {
    echo "[conformance] fixture extraction failed. If shards are LFS pointers, run:" >&2
    echo "  git lfs install && git lfs pull" >&2
    exit 1
  }
else
  FIXTURES_ROOT="$FIXTURES_SNAP"
fi
# Export so cargo test subprocesses read the same pinned snapshot.
export CONFORMANCE_FIXTURES_ROOT="$FIXTURES_ROOT"
# Ephemeral build tree stays at conformance/utils/.stage (UTILS), not inside src/,
# so CI and .gitignore find it where they always have.
STAGE="${STAGE:-$UTILS/.stage}"
# Override when the default cargo can't build the workspace (edition 2024 /
# resolver "3" needs >= 1.85): CARGO='cargo +1.96.1' conformance/utils/check.sh ...
CARGO="${CARGO:-cargo}"
: "${DRY:=0}"

_build_stage_base() {
  rm -rf "$STAGE"
  mkdir -p "$STAGE/tests/parity" "$STAGE/lib/parsers/src"
  # COPY the table package so resolved __file__ -> REPO_ROOT == $STAGE. It is staged
  # as <stage>/tables (not under tests/parity) so `import tables.…` resolves the same
  # way here as it does from conformance/utils/src in the repo — one import name, both
  # contexts. The tests/parity/ subtree below stays the fixture + template layout the
  # generator's path constants and relative-link resolution are written against.
  \cp -Rf "$TOOLS/tables" "$STAGE/tables"
  # Family marker declarations (DIS-2442): the staged tests/parity/markup.py resolves
  # its registry at <stage-root>/src/parser_families.yaml, mirroring the repo layout.
  mkdir -p "$STAGE/src"
  \cp -f "$TOOLS/parser_families.yaml" "$STAGE/src/parser_families.yaml"
  # Static CSS/JS inlined into the conformance page at render time (the renderer
  # reads tests/parity/assets/*). The compare-bar/coloring logic lives in one place.
  mkdir -p "$STAGE/tests/parity/assets"
  \cp -f "$TOOLS/assets/conformance.css" "$STAGE/tests/parity/assets/conformance.css"
  \cp -f "$TOOLS/assets/conformance.js" "$STAGE/tests/parity/assets/conformance.js"
  # Markup colorizer (port of markup.py); inlined before conformance_view.js.
  \cp -f "$TOOLS/assets/colorize.js" "$STAGE/tests/parity/assets/colorize.js"
  # JSON data model + JS view: model.py builds it (imported via reasoning_table),
  # conformance_view.js renders it.
  \cp -f "$TOOLS/model.py" "$STAGE/tests/parity/model.py"
  # markers.py (comparison/facts semantics) + impls.py (identity) are the single
  # source the tables import instead of forking their own comparison copies.
  \cp -f "$TOOLS/impls.py" "$STAGE/tests/parity/impls.py"
  \cp -f "$TOOLS/markers.py" "$STAGE/tests/parity/markers.py"
  \cp -f "$TOOLS/unified_taxonomy.py" "$STAGE/tests/parity/unified_taxonomy.py"
  [ -f "$TOOLS/assets/conformance_view.js" ] && \
    \cp -f "$TOOLS/assets/conformance_view.js" "$STAGE/tests/parity/assets/conformance_view.js" || true
  # Reasoning fixtures are resolved (at the pinned peer versions) by
  # build_stage_conformance — not here in the shared base.
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
    --out "$out" --select "dynamo_v1-${dynamo_v}" "vllm_python-${vllm_v}" "sglang_python-${sglang_v}"
}

# Reasoning fixtures are versioned like toolcalling: inputs/ = the OLD (v1-era) anchor,
# <impl>-<version>/ = changed-only overlays for a newer engine. The page picks which
# version to render, so this takes the versions as args ($2=vllm, $3=sglang).
_resolve_reasoning_fixtures() {
  local out="$1" vllm_v="$2" sglang_v="$3"; mkdir -p "$out"
  python3 "$TOOLS/resolve_reasoning_fixtures.py" \
    --fixtures-root "$FIXTURES_ROOT/reasoning/fixtures-v1" \
    --out "$out" --select "vllm_python-${vllm_v}" "sglang_python-${sglang_v}"
}

# The pinned reasoning peer versions = the engines pinned in pyproject.stub.toml.
_reasoning_pinned_ver() {  # $1 = vllm | sglang
  grep -oE "$1\[[^]]*\]==[^\"]+" "$TOOLS/pyproject.stub.toml" | sed -E 's/.*==//'
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
  # peer versions into the flat tree the renderer expects; a genuinely single-version
  # impl (vllm_rust) defaults to its lowest. The generator re-resolves each version for
  # the compare model.
  local sv2="$FIXTURES_ROOT/toolcalling/fixtures-stream-v2"
  if [ -d "$sv2" ]; then
    local vllm_v sglang_v tmp family f
    vllm_v=$(grep -oE 'vllm\[[^]]*\]==[^"]+' "$TOOLS/pyproject.stub.toml" | sed -E 's/.*==//')
    sglang_v=$(grep -oE 'sglang\[[^]]*\]==[^"]+' "$TOOLS/pyproject.stub.toml" | sed -E 's/.*==//')
    # dynamo_v2 is NOT single-version any more: 0.1.11 (anchor, 4 families) +
    # 0.1.11.patch1 + 0.1.22 (9 families). Defaulting it to the lowest staged only the
    # anchor, so gemma4/glm47/kimi_k2/minimax_m2/minimax_m3 had no per-chunk `expected`
    # and rendered "—" for every chunk — while the column was LABELLED 0.1.22, because
    # _dynamo_v2_version() reads the published dir list. Select the latest so the data
    # matches its own label.
    local dynamo_v2_v
    dynamo_v2_v=$(ls -d "$sv2"/dynamo_v2-* 2>/dev/null | sed 's|.*/dynamo_v2-||' | sort -V | tail -1)
    tmp="$(mktemp -d)"
    python3 "$TOOLS/resolve_stream_fixtures.py" \
      --fixtures-root "$sv2" --out "$tmp" \
      --select "vllm_python-${vllm_v}" "sglang_python-${sglang_v}" \
              ${dynamo_v2_v:+"dynamo_v2-${dynamo_v2_v}"}
    for f in "$tmp"/*/TOOLCALLING.stream*.yaml; do
      [ -f "$f" ] || continue
      family="$(basename "$(dirname "$f")")"
      mkdir -p "$dst/$family"
      \cp -f "$f" "$dst/$family/"
    done
    rm -rf "$tmp"
  fi
}

build_stage_conformance() {
  _build_stage_base
  # Keep the current conformance harness owned by conformance/utils while presenting
  # it in Dynamo's staged tests/parity layout for imports and template lookup.
  \cp -f "$TOOLS/generate_conformance_table.py" "$STAGE/tests/parity/generate_conformance_table.py"
  # impls.py + markers.py are staged in _build_stage_base.
  \cp -f "$TOOLS/fixtures.py" "$STAGE/tests/parity/fixtures.py"
  \cp -f "$TOOLS/conformance_table.html.j2" "$STAGE/tests/parity/conformance_table.html.j2"
  # Shared CSS/JS assets are staged in _build_stage_base.
  _copy_toolcalling_v2_fixtures
  # Current page: reasoning shows the pinned NEW peer versions, in sync with the v2
  # toolcalling tab (both compare against the current engines).
  _resolve_reasoning_fixtures "$STAGE/tests/parity/reasoning/fixtures" \
    "$(_reasoning_pinned_ver vllm)" "$(_reasoning_pinned_ver sglang)"
}

build_stage() {
  echo "build_stage is deprecated; use build_stage_conformance" >&2
  return 2
}
