#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Steps of the bazel-rust CI job. The workflow keeps only checkout and
# actions/cache, so tool versions, cache keys, and Bazel flags change here.
#
#   ci/bazel/ci.sh plan   install Bazelisk; emit cache key, restore prefix, and paths
#   ci/bazel/ci.sh test   run every Bazel test and summarize action reuse
set -euo pipefail

BAZELISK_VERSION=1.28.1
BAZELISK_SHA256=22e7d3a188699982f661cf4687137ee52d1f24fec1ec893d91a6c4d791a75de8
# Bazelisk checks the Bazel binary named in .bazelversion against this digest.
export BAZELISK_VERIFY_SHA256=2e14c55c298752d71e8a0bbd762665ffce5966c9ad00ec5d1a8b82a58e7bfa06
# Bump to drop every saved entry.
CACHE_VERSION=v2
# The workflow timeout is only a ceiling; this budget ends Bazel cleanly first.
TEST_BUDGET=40m
CACHE_ROOT="$HOME/.cache/frontend-crates-bazel"

plan() {
    local bin="$RUNNER_TEMP/bin"
    mkdir -p "$bin"
    curl -fsSL -o "$bin/bazel" \
        "https://github.com/bazelbuild/bazelisk/releases/download/v${BAZELISK_VERSION}/bazelisk-linux-amd64"
    echo "${BAZELISK_SHA256}  $bin/bazel" | sha256sum -c -
    chmod +x "$bin/bazel"
    echo "$bin" >> "$GITHUB_PATH"

    # main refreshes its entry on every merge. A PR saves once on its first run
    # and later pushes reuse it, so each PR holds one bounded entry.
    local inputs prefix scope
    inputs=$(cat .bazelversion MODULE.bazel.lock | sha256sum | cut -c1-16)
    prefix="bazel-${CACHE_VERSION}-${RUNNER_OS}-${inputs}-"
    scope=$GITHUB_REF_NAME
    [[ $GITHUB_REF_NAME == main ]] && scope=$GITHUB_SHA
    # Unpacked repositories are about 2.3 GB and recreate in under a minute, so
    # only build results and downloaded archives are cached.
    {
        echo "save=true"
        echo "key=${prefix}${scope}"
        echo "restore-prefix=${prefix}"
        echo "paths<<PATHS"
        echo "$CACHE_ROOT/actions"
        echo "$CACHE_ROOT/repository/content_addressable"
        echo "PATHS"
    } >> "$GITHUB_OUTPUT"
}

run_tests() {
    local log="$RUNNER_TEMP/bazel.log" status=0
    timeout --signal=INT --kill-after=2m "$TEST_BUDGET" \
        bazel --nohome_rc --nosystem_rc test //... \
            --disk_cache="$CACHE_ROOT/actions" \
            --repository_cache="$CACHE_ROOT/repository" \
            --keep_going 2>&1 | tee "$log" || status=$?
    {
        echo "### Bazel action reuse"
        echo '```'
        grep -E '^INFO: ([0-9,]+ processes|Elapsed time|Build completed)|Executed [0-9]+ out of' "$log" || true
        echo '```'
    } >> "$GITHUB_STEP_SUMMARY"
    return "$status"
}

case "${1:-}" in
    plan) plan ;;
    test) run_tests ;;
    *) echo "usage: $0 plan|test" >&2; exit 2 ;;
esac
