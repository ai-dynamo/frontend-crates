#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# render_table_v1.sh [--dry-run]
#   Render old Dynamo parity HTML to conformance/utils/.stage/tests/parity/PARITY_v1.html.
#   No engines needed.

DRY=0; args=()
while [ $# -gt 0 ]; do case "$1" in --dry-run|--dryrun) DRY=1; shift;; *) args+=("$1"); shift;; esac; done
set -- ${args+"${args[@]}"}
source "$(dirname "$0")/src/_common.sh"

STAGE="$UTILS/.stage"
OUT="$STAGE/tests/parity/PARITY_v1.html"
if [ "$DRY" = 1 ]; then
  echo "[dry-run] build v1 .stage, then render the old Dynamo parity table > $OUT"
  exit 0
fi
build_stage_v1
( cd "$STAGE" && PYTHONPATH="$STAGE" python3 tests/parity/generate_parity_table_v1.py all --html > tests/parity/PARITY_v1.html )
echo "wrote $OUT"
