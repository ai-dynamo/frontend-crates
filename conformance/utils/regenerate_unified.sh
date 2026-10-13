#!/usr/bin/env bash
set -euo pipefail

ROOT=$(git rev-parse --show-toplevel)
cd "$ROOT"

run() {
  printf '\n==> %s\n' "$*"
  "$@"
}

run python3 conformance/utils/src/gen_unified_golden.py

if ! run cargo test --locked -p dynamo-conformance-fixtures-v2 --test unified_render -- --exact render_unified_conformance_html --nocapture; then
  exit 1
fi
if ! run python3 conformance/utils/src/explode_unified_fixtures.py; then
  exit 1
fi
if ! run python3 conformance/utils/src/package_fixtures.py; then
  exit 1
fi
materialized_history=$(mktemp -d /tmp/dynamo-unified-history.XXXXXX)
trap '\rm -rf "$materialized_history"' EXIT
if ! run python3 conformance/utils/src/unified_history.py --materialize --store conformance/fixtures-unified-v2 --output "$materialized_history"; then
  exit 1
fi
if ! run python3 conformance/utils/src/extract_fixtures.py --full-refresh; then
  exit 1
fi
run python3 conformance/utils/src/viewer_catalog.py
run python3 conformance/utils/src/viewer_catalog.py --check

status=0
run cargo test --locked -p dynamo-parsers-v2 --lib -- --nocapture || status=1
run cargo test --locked -p dynamo-conformance-fixtures-v2 --test unified_schema_roundtrip -- --nocapture || status=1
run cargo test --locked -p dynamo-conformance-fixtures-v2 --test unified_parity -- --nocapture || status=1
run cargo test --locked -p dynamo-conformance-fixtures-v2 --test unified_render -- --nocapture || status=1
run python3 -m pytest -q \
  conformance/utils/tests/test_viewer_model.py \
  conformance/utils/tests/test_viewer_status.py \
  conformance/utils/tests/test_unified_taxonomy_covers_corpus.py \
  conformance/utils/tests/test_unified_fixture_overlays.py || status=1

run python3 - <<'PY' || status=1
from pathlib import Path

import sys

sys.path.insert(0, "conformance/utils/src")
import gen_unified_golden as golden
from dynamo_version import dynamo_v2_label
from unified_history import load_store, _capture_release_sort_key
from unified_taxonomy import numbered_id

root = Path("conformance/fixtures-unified-v2")
expected = {
    family: {
        numbered_id(key[len("UNIFIED."):].rsplit(".", 1)[0])
        for key in golden.build_cases(family)
    }
    for family in golden.FAMILIES
}
current_version = dynamo_v2_label(Path.cwd())
store = load_store(root)
for family, case_ids in expected.items():
    canonical = {
        case["display_id"]
        for case in store.families[family].cases.values()
        if case["lifecycle"] == "active"
    }
    history = store.histories[(family, "dynamo_v2")]
    current_captures = [
        (capture_id, capture)
        for capture_id, capture in history.captures.items()
        if _capture_release_sort_key(capture["runtime_version"]) <= _capture_release_sort_key(current_version)
    ]
    if not current_captures:
        raise SystemExit(f"missing generated Unified capture at or before {current_version}/{family}")
    current_label, _capture = max(
        current_captures,
        key=lambda item: _capture_release_sort_key(item[1]["runtime_version"]),
    )
    captured = {
        history.family.cases[case_id]["display_id"]
        for case_id in history.resolve(current_label)
        if history.family.cases[case_id]["lifecycle"] == "active"
    }
    for kind, actual in (("inputs/golden", canonical), ("capture", captured)):
        missing = sorted(case_ids - actual)
        extra = sorted(actual - case_ids)
        if missing or extra:
            raise SystemExit(
                f"{kind} {current_label}/{family} differs from generator; "
                f"missing={missing} extra={extra}"
            )

print("Unified source gate passed: generated YAML history and captures are current.")
PY

run node conformance/viewer/status.js --tab unified \
  --known-divergences conformance/unified-known-divergences.yaml --summary-only || status=1

exit "$status"
