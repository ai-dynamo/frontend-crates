#!/usr/bin/env bash
set -euo pipefail

ROOT=$(git rev-parse --show-toplevel)
cd "$ROOT"

run() {
  printf '\n==> %s\n' "$*"
  "$@"
}

render_report() {
  printf '\n==> bash conformance/utils/render_table_v2.sh --output conformance/CONFORMANCE_v2.html\n'
  bash conformance/utils/render_table_v2.sh --output conformance/CONFORMANCE_v2.html
}

run python3 conformance/utils/src/gen_unified_golden.py

if ! run cargo test --locked -p dynamo-conformance-fixtures-v2 --test unified_render -- --nocapture; then
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
import tarfile
from pathlib import Path

import sys

sys.path.insert(0, "conformance/utils/src")
import gen_unified_golden as golden
from unified_taxonomy import numbered_id

root = Path("conformance/fixtures/unified")
current = json.loads(Path("conformance/CONFORMANCE_v2.json").read_text())
expected = {
    family: {
        numbered_id(key[len("UNIFIED."):].rsplit(".", 1)[0])
        for key in golden.build_cases(family)
    }
    for family in golden.FAMILIES
}

current_label = next(
    line.split('"')[1]
    for line in Path("conformance/tests/common/mod.rs").read_text().splitlines()
    if "UNIFIED_DYNAMO_V2_CURRENT_CAPTURE" in line
)
archives = [root / "inputs.tar.gz", root / "golden.tar.gz", root / f"{current_label}.tar.gz"]
for archive in archives:
    if not archive.exists():
        raise SystemExit(f"missing generated Unified archive: {archive}")
    with tarfile.open(archive) as tar:
        actual = {family: set() for family in golden.FAMILIES}
        for member in tar.getnames():
            parts = member.split("/")
            if len(parts) == 4 and parts[2] in actual and member.endswith(".yaml"):
                actual[parts[2]].add(parts[3][:-5])
    for family, case_ids in expected.items():
        missing = sorted(case_ids - actual[family])
        extra = sorted(actual[family] - case_ids)
        if missing or extra:
            raise SystemExit(
                f"{archive}: {family} differs from generator; missing={missing} extra={extra}"
            )

for report in current["reports"]:
    if report.get("tab") == "tab-unified" and (report["empty"] or report["red"]):
        raise SystemExit(
            f"Unified display is not clean for {report['model']}: "
            f"empty={report['empty']} red={report['red']}"
        )

print("Unified regeneration gate passed: generated archives and rendered JSON are current.")
PY

exit "$status"
