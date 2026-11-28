#!/usr/bin/env bash
set -euo pipefail

ROOT=$(git rev-parse --show-toplevel)
cd "$ROOT"


run() {
  printf '\n==> %s\n' "$*"
  "$@"
}

run python3 conformance/utils/src/gen_unified_golden.py
run python3 conformance/utils/src/explode_unified_fixtures.py --authored-inputs

materialized_history=$(mktemp -d "${TMPDIR:-/tmp}/dynamo-unified-history.XXXXXX")
trap '\rm -rf "$materialized_history"' EXIT
export CONFORMANCE_FIXTURES_ROOT="$materialized_history"
run python3 conformance/utils/src/unified_history.py --materialize --store conformance/fixtures-unified-v2 --output "$materialized_history/unified" --capture latest:dynamo_v2 --capture latest:vllm_rust

if ! run cargo test --locked -p dynamo-conformance-fixtures-v2 --test unified_render -- --exact render_unified_conformance_html --nocapture; then
  exit 1
fi
if ! run python3 conformance/utils/src/explode_unified_fixtures.py; then
  exit 1
fi
if ! run python3 conformance/utils/src/package_fixtures.py --unified-only; then
  exit 1
fi
run python3 conformance/utils/src/viewer_catalog.py --unified-only
if ! run python3 conformance/utils/src/unified_history.py --materialize --store conformance/fixtures-unified-v2 --output "$materialized_history/unified" --capture latest:dynamo_v2 --capture latest:vllm_rust; then
  exit 1
fi
run python3 conformance/utils/src/unified_history.py --validate --store conformance/fixtures-unified-v2
run python3 conformance/utils/src/viewer_catalog.py --unified-only --check

status=0
run cargo test --locked -p dynamo-parsers-v2 --lib -- --nocapture || status=1
run cargo test --locked -p dynamo-conformance-fixtures-v2 --test unified_schema_roundtrip -- --nocapture || status=1
run cargo test --locked -p dynamo-conformance-fixtures-v2 --test unified_parity -- --nocapture || status=1
run cargo test --locked -p dynamo-conformance-fixtures-v2 --test unified_render -- --nocapture || status=1
run python3 -m pytest -q \
  conformance/utils/tests/test_viewer_data.py \
  conformance/utils/tests/test_unified_taxonomy_covers_corpus.py || status=1

run python3 - <<'PY' || status=1
from pathlib import Path

import sys

sys.path.insert(0, "conformance/utils/src")
import gen_unified_golden as golden
from dynamo_version import dynamo_v2_label
from unified_history import load_store, _capture_release_sort_key
from unified_taxonomy import numbered_id
from explode_unified_fixtures import _split_only_cases
from fixture_disposition import canonical_unified_case_key
from impls import parser_families_path
import yaml

root = Path("conformance/fixtures-unified-v2")
store = load_store(root)
expected = {
    family: {canonical_unified_case_key(family, key, row["scenario"])
             for key, row in _split_only_cases(family, {"family": family, "cases": golden.build_cases(family)}).items()}
    for family in store.families
}
for family in golden.scenario_families("v1_split_parallel_tool_calls"):
    if family in store.families and any(
        case["lifecycle"] == "active" and case["scenario"] == "v1_split_parallel_tool_calls"
        for case in store.families[family].cases.values()
    ):
        expected[family].add(numbered_id("v1_split_parallel_tool_calls"))
current_version = dynamo_v2_label(Path.cwd())
registry = yaml.safe_load(parser_families_path().read_text())["unified"]
for family, case_ids in expected.items():
    canonical = {
        canonical_unified_case_key(family, case["display_id"], case["scenario"])
        for case in store.families[family].cases.values()
        if case["lifecycle"] == "active"
    }
    if case_ids != canonical:
        raise SystemExit(f"inputs/golden {family} differs from generator; "
                         f"missing={sorted(case_ids - canonical)} extra={sorted(canonical - case_ids)}")
    history = store.histories.get((family, "dynamo_v2"))
    if history is None:
        if registry[family]["native"]:
            raise SystemExit(f"missing native Dynamo v2 capture for {family}")
        continue
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
        canonical_unified_case_key(family, history.family.cases[case_id]["display_id"],
                                   history.family.cases[case_id]["scenario"])
        for case_id in history.resolve(current_label)
        if history.family.cases[case_id]["lifecycle"] == "active"
    }
    if case_ids != captured:
        raise SystemExit(f"capture {current_label}/{family} differs from generator; "
                         f"missing={sorted(case_ids - captured)} extra={sorted(captured - case_ids)}")

print("Unified source gate passed: generated YAML history and captures are current.")
PY

run node conformance/viewer/qualification.js status --tab unified \
  --known-divergences conformance/unified-known-divergences.yaml --summary-only || status=1

exit "$status"
