#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# record_v2.sh <stream <fixture.yaml> [--text] | batch | tokens> [--dry-run]
#   Regenerate fixture data with the frontend-crate v2 Rust parser; you then commit the YAML.
#     stream <fixture.yaml> [--text]   print per-chunk expected.dynamo JSON for one
#                                      stream fixture (paste into chunks[].expected.dynamo)
#     batch                            print the frontend-crate v2 stream-on-batch result JSON
#                                      (merge via merge_batch_stream.py -> harmony_batch_stream.json)
#     tokens                           stamp delta_token_ids into the harmony stream fixtures
#
#   Needs a cargo that can build the workspace (edition 2024). If the default cargo
#   is too old: CARGO='cargo +1.93.1' conformance/utils/record_v2.sh ...

DRY=0; args=()
while [ $# -gt 0 ]; do case "$1" in --dry-run|--dryrun) DRY=1; shift;; *) args+=("$1"); shift;; esac; done
set -- ${args+"${args[@]}"}
source "$(dirname "$0")/_common.sh"

runbin() {  # $@ = <bin> [-- args...]
  if [ "$DRY" = 1 ]; then echo "[dry-run] (cd $ROOT && $CARGO run -p dynamo-parsers-v2 --bin $*)"
  else ( cd "$ROOT" && $CARGO run -p dynamo-parsers-v2 --bin "$@" ); fi
}

sub="${1:-}"; shift || true
case "$sub" in
  stream)
    [ $# -ge 1 ] || { echo "usage: conformance/utils/record_v2.sh stream <fixture.yaml> [--text]" >&2; exit 2; }
    runbin record_dynamo_stream -- "$@"
    ;;
  batch)  runbin record_batch_via_stream ;;
  tokens) runbin stamp_stream_token_ids ;;
  *) echo "usage: conformance/utils/record_v2.sh <stream <fixture.yaml> [--text] | batch | tokens>" >&2; exit 2 ;;
esac
