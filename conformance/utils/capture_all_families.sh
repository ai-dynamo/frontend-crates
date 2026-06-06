#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Capture per-chunk vLLM + SGLang streaming deltas for every non-harmony family
# and build the new-format conformance/toolcalling/fixtures-stream-v2/<family>/TOOLCALLING.stream.*.yaml.
# Dynamo parser v2 is marked unavailable (TODO) for all of these — no Dynamo parser v2 Rust streaming
# parser exists yet; vLLM/SGLang per-chunk data is the target to match.
#
# Runs the engine parsers INSIDE the containers (docker exec), version-matched.
#
# Usage:  conformance/utils/capture_all_families.sh [VLLM_CONTAINER] [SGLANG_CONTAINER]

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PH="$ROOT/conformance/utils"
VLLM_C="${1:-vllm-localdev}"
SGLANG_C="${2:-sglang-localdev}"
CONF="$ROOT/conformance/toolcalling/fixtures"
WORK="${TMPDIR:-/tmp}/streamcap_families"
DYNAMO_TODO="Dynamo parser v2 TC streaming not yet implemented for this family; vLLM/SGLang per-chunk output is the target to match."

rm -rf "$WORK"; mkdir -p "$WORK"

python3 "$PH/capture_all_families_driver.py" \
  --root "$ROOT" --work "$WORK" \
  --vllm-container "$VLLM_C" --sglang-container "$SGLANG_C" \
  --dynamo-todo "$DYNAMO_TODO"
