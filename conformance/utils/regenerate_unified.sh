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
from dynamo_version import dynamo_v2_provenance
from fixtures import _version_sort_key
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
expected_red = {
    family: {
        key[len("UNIFIED."):].rsplit(".", 1)[0]
        for key, case in golden.build_cases(family).items()
        if case.get("expect", {}).get("dynamo_current", {}).get("verdict") == "diverge"
    }
    for family in golden.FAMILIES
}

current_version = dynamo_v2_provenance(Path.cwd())["crate_version"]
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
        if _version_sort_key(capture["runtime_version"]) <= _version_sort_key(current_version)
    ]
    if not current_captures:
        raise SystemExit(f"missing generated Unified capture at or before {current_version}/{family}")
    current_label, _capture = max(
        current_captures,
        key=lambda item: _version_sort_key(item[1]["runtime_version"]),
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

seen_reports = set()
for report in current["reports"]:
    if report.get("tab") != "tab-unified":
        continue
    family = report["model"]
    seen_reports.add(family)
    actual_red = {
        issue["scenario"]
        for issue in report.get("issues", [])
        if issue.get("state") == "red"
    }
    if report["empty"] or actual_red != expected_red[family]:
        raise SystemExit(
            f"Unified display differs from documented current-Dynamo expectations for {family}: "
            f"empty={report['empty']} expected_red={sorted(expected_red[family])} "
            f"actual_red={sorted(actual_red)}"
        )
if seen_reports != set(golden.FAMILIES):
    raise SystemExit(
        f"Unified display families differ from the generator: "
        f"missing={sorted(set(golden.FAMILIES) - seen_reports)} "
        f"extra={sorted(seen_reports - set(golden.FAMILIES))}"
    )

print("Unified regeneration gate passed: generated YAML history and rendered JSON are current.")
PY

exit "$status"
