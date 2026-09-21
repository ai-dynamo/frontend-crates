#!/usr/bin/env bash
set -euo pipefail

ROOT=$(git rev-parse --show-toplevel)
cd "$ROOT"

# Freeze the label before generation; later consumers reject any source drift.
CONFORMANCE_DYNAMO_V2_LABEL=$(python3 conformance/utils/src/dynamo_version.py --format label)
export CONFORMANCE_DYNAMO_V2_LABEL

run() {
  printf '\n==> %s\n' "$*"
  "$@"
}

render_report() {
  printf '\n==> bash conformance/utils/render_table_v2.sh --output conformance/CONFORMANCE_v2.html\n'
  bash conformance/utils/render_table_v2.sh --output conformance/CONFORMANCE_v2.html
}

run python3 conformance/utils/src/gen_unified_golden.py

if ! run cargo test --locked -p dynamo-conformance-fixtures-v2 --test unified_render -- --exact render_unified_conformance_html --nocapture; then
  render_report || true
  exit 1
fi
if ! run python3 conformance/utils/src/explode_unified_fixtures.py; then
  render_report || true
  exit 1
fi
if ! run python3 conformance/utils/src/package_fixtures.py; then
  render_report || true
  exit 1
fi
materialized_history=$(mktemp -d /tmp/dynamo-unified-history.XXXXXX)
trap '\rm -rf "$materialized_history"' EXIT
if ! run python3 conformance/utils/src/unified_history.py --materialize --store conformance/fixtures-unified-v2 --output "$materialized_history"; then
  render_report || true
  exit 1
fi
if ! run python3 conformance/utils/src/extract_fixtures.py --full-refresh; then
  render_report || true
  exit 1
fi
render_report

status=0
run cargo test --locked -p dynamo-parsers-v2 --lib -- --nocapture || status=1
run cargo test --locked -p dynamo-conformance-fixtures-v2 --test unified_schema_roundtrip -- --nocapture || status=1
run cargo test --locked -p dynamo-conformance-fixtures-v2 --test unified_parity -- --nocapture || status=1
run cargo test --locked -p dynamo-conformance-fixtures-v2 --test unified_render -- --nocapture || status=1
run python3 -m pytest -q \
  conformance/utils/tests/test_model.py \
  conformance/utils/tests/test_unified_taxonomy_covers_corpus.py \
  conformance/utils/tests/test_unified_fixture_overlays.py || status=1

run python3 - <<'PY' || status=1
import json
from pathlib import Path

import sys

sys.path.insert(0, "conformance/utils/src")
import gen_unified_golden as golden
from dynamo_version import dynamo_v2_label
from unified_history import load_store
from unified_taxonomy import numbered_id

root = Path("conformance/fixtures-unified-v2")
current = json.loads(Path("conformance/CONFORMANCE_v2.json").read_text())
expected = {
    family: {
        numbered_id(key[len("UNIFIED."):].rsplit(".", 1)[0])
        for key in golden.build_cases(family)
    }
    for family in golden.FAMILIES
}

current_label = f"dynamo_v2-{dynamo_v2_label(Path.cwd())}"
store = load_store(root)
for family, case_ids in expected.items():
    canonical = {
        case["display_id"]
        for case in store.families[family].cases.values()
        if case["lifecycle"] == "active"
    }
    history = store.histories[(family, "dynamo_v2")]
    if current_label not in history.captures:
        raise SystemExit(f"missing generated Unified capture: {current_label}/{family}")
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

for report in current["reports"]:
    if report.get("tab") == "tab-unified" and (report["empty"] or report["red"]):
        raise SystemExit(
            f"Unified display is not clean for {report['model']}: "
            f"empty={report['empty']} red={report['red']}"
        )

print("Unified regeneration gate passed: generated YAML history and rendered JSON are current.")
PY

exit "$status"
