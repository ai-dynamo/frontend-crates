#!/usr/bin/env bash
set -euo pipefail

root=$(git rev-parse --show-toplevel)
output=$(mktemp -d "${TMPDIR:-/tmp}/conformance-report.XXXXXX")
trap 'rm -rf "$output"' EXIT
cd "$root"
cargo build --locked --release -p conformance-publisher
npm ci --prefix conformance/viewer
target/release/conformance-publish --root "$root" --output "$output" --revision "$(git rev-parse HEAD)"
cp "$output/index.html" conformance/CONFORMANCE_v2.html
printf '%s\n' "$root/conformance/CONFORMANCE_v2.html"
