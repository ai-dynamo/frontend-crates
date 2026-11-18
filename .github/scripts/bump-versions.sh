#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Path-scoped auto-bump shared by the release and preview jobs. Runs
# `release-plz update -p <crate>` only for crates whose OWN directory changed
# since their last release tag, and never touches a version a human bumped:
#   - version != last release tag  -> manually PEGGED: leave it alone
#   - version == last tag, no changes under the crate dir -> nothing
#   - version == last tag, changes under the crate dir -> auto-bump this crate
#     only (release-plz sizes the bump from conventional commits)
set -euo pipefail

# `--allow-dirty` below is intentional: each package update leaves its
# version/changelog edits for the final combined release commit. Fail here so
# it cannot mask changes that predate this loop.
if [ -n "$(git status --porcelain)" ]; then
  git status --short
  echo "Release checkout must start clean."
  exit 1
fi

members=$(sed -n '/^members = \[/,/^]/p' Cargo.toml | grep -oE '"[^"]+"' | tr -d '"')
for dir in $members; do
  grep -q '^publish *= *false' "$dir/Cargo.toml" && continue
  name=$(grep -m1 '^name *=' "$dir/Cargo.toml" | sed -E 's/.*"([^"]+)".*/\1/')
  ver=$(grep -m1 '^version *=' "$dir/Cargo.toml" | sed -E 's/.*"([^"]+)".*/\1/')
  # Exact-match this crate's tags: the glob "dynamo-parsers-v*" would also
  # match dynamo-parsers-v2-v0.1.21, so require <name>-v<semver>.
  last=$(git tag -l "$name-v*" | grep -E "^$name-v[0-9]+\.[0-9]+\.[0-9]+$" | sort -V | tail -1 || true)
  if [ -z "$last" ]; then
    echo "$name: no release tag yet -> first release is manual, skipping"
  elif [ "$ver" != "${last#"$name"-v}" ]; then
    echo "$name: $ver != last released ${last#"$name"-v} -> manually pegged, publish as-is"
  # The root Cargo.toml is part of every crate's change scope:
  # workspace-inherited dependency versions and [workspace.package] metadata
  # land in the published manifest of each member crate.
  elif git diff --quiet "$last"..HEAD -- "$dir" Cargo.toml; then
    echo "$name: no changes under $dir/ or the root Cargo.toml since $last"
  else
    echo "$name: changes under $dir/ (or root Cargo.toml) since $last -> auto-bump"
    release-plz update --allow-dirty -p "$name"
  fi
done
