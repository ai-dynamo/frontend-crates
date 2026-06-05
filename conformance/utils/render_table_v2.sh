#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# render_table_v2.sh [--dry-run]
#   Render the conformance matrix to conformance/CONFORMANCE_v2.html
#   (all four tabs: TC batch / TC stream / TC batch-on-stream / Reasoning).
#   No engines needed.

DRY=0; args=()
while [ $# -gt 0 ]; do case "$1" in --dry-run|--dryrun) DRY=1; shift;; *) args+=("$1"); shift;; esac; done
set -- ${args+"${args[@]}"}
source "$(dirname "$0")/_common.sh"

OUT="$ROOT/conformance/CONFORMANCE_v2.html"
if [ "$DRY" = 1 ]; then
  echo "[dry-run] build v2 .stage-v2, then render the v2 conformance table > $OUT"
  exit 0
fi
build_stage_v2
( cd "$STAGE" && PYTHONPATH="$STAGE" python3 tests/parity/generate_conformance_table_v2.py all --html --output-path "$OUT" --artifact-root "$ROOT" ) > "$OUT"
echo "wrote $OUT"
