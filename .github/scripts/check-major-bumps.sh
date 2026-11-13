#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Fail if bump-versions.sh changed any manifest's major version relative to the
# checked-out commit. A major version pegged in the commit itself is allowed.
set -euo pipefail

fail=0
while IFS= read -r toml; do
  old=$(git show "HEAD:$toml" | sed -nE 's/^version *= *"([^"]+)".*/\1/p' | head -1)
  new=$(sed -nE 's/^version *= *"([^"]+)".*/\1/p' "$toml" | head -1)
  [ -n "$old" ] && [ -n "$new" ] || continue
  if [ "${old%%.*}" != "${new%%.*}" ]; then
    echo "::error file=$toml::This PR is marked as a breaking change, so release-plz would bump $old -> $new. Remove the breaking marker from the PR title, or set the major version in this PR."
    fail=1
  fi
done < <(git ls-files | grep -E '(^|/)Cargo\.toml$')
exit "$fail"
