# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fixture-coverage and marker-registration lint (DIS-2442).

Diffs each parser family's fixture case IDs against the machine-readable case
taxonomy (conformance/case-taxonomy.yaml) and fails on unexplained gaps, so
"complete coverage" is a tool answer instead of a review-time reverse-engineering
exercise (PR #120: batch groups 6/8/30 and the whole stream-v2 corpus were
missing and nothing said so before human review).

Checks, per family:
  * family present in parser_families.yaml (toolcalling registry);
  * a fixtures dir exists for every applicable suite (catches "ALL stream test
    cases missing" — a registry family with no fixtures-stream-v2/inputs dir);
  * every `required:` taxonomy group/case is present, either as real input
    (model_text / chunks) or as an explicit placeholder with an `explanation:`;
  * placeholders without an `explanation:` fail; "not yet authored" placeholders
    warn (acknowledged TODO);
  * case IDs unknown to the taxonomy fail (forces taxonomy updates in the same
    PR that introduces a new group/sub-case);
  * every registry family declares its grammar tokens in the `markers:` section
    of parser_families.yaml, each declared token is matched by the leak detector
    (markers._TOOL_CALL_MARKUP_RE) and renders non-orphan in the popup colorizer
    (tables.markup) — the two registries nobody knew existed in PR #120.

Reasoning suites are checked when the family has a reasoning fixtures dir, or
always with --expect-reasoning (use when the PR adds a reasoning parser).

Exit code 1 on any FAIL. Warnings (TODO placeholders, grandfathered known_gaps)
never fail the run; they are the backfill work list.
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
# HERE (conformance/utils/src) is on sys.path above, which is where tables/ lives.

import markers as markers_mod  # noqa: E402
from tables import markup  # noqa: E402

TAXONOMY_PATH = HERE.parents[1] / "case-taxonomy.yaml"
REGISTRY_PATH = HERE / "parser_families.yaml"

SUITE_INPUTS = {
    "toolcalling.batch": ("toolcalling/fixtures-batch-v1", "TOOLCALLING.batch"),
    "toolcalling.stream": ("toolcalling/fixtures-stream-v2", "TOOLCALLING.streamv2"),
    "reasoning.batch": ("reasoning/fixtures-v1", "REASONING.batch"),
    "reasoning.stream": ("reasoning/fixtures-v1", "REASONING.stream"),
}
TOOLCALLING_SUITES = ("toolcalling.batch", "toolcalling.stream")
REASONING_SUITES = ("reasoning.batch", "reasoning.stream")

# Placeholder-case classification (see case-taxonomy.yaml "Semantics").
_TODO_MARKER = "not yet authored"


def resolve_fixtures_root() -> Path:
    """The manifest-pinned snapshot dir. Runs extract_fixtures.py (instant on cache
    hit) and reads the snapshot path it prints — NEVER the shared
    `<cache>/toolcalling` symlink, which sibling checkouts pinning other snapshots
    race to repoint mid-run (see conformance/README "Invariants"). An exported
    CONFORMANCE_FIXTURES_ROOT (set by _common.sh) wins."""
    env = os.environ.get("CONFORMANCE_FIXTURES_ROOT")
    if env:
        root = Path(env)
        if not (root / "toolcalling").is_dir():
            sys.exit(f"fixtures root {root} has no toolcalling/ dir")
        return root
    proc = subprocess.run(
        [sys.executable, str(HERE / "extract_fixtures.py")],
        check=True,
        capture_output=True,
        text=True,
    )
    snap = Path(proc.stdout.strip().splitlines()[-1])
    if not (snap / "toolcalling").is_dir():
        sys.exit(f"extract_fixtures.py returned an unusable snapshot dir: {snap}")
    return snap


class Report:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.warnings: list[str] = []

    def fail(self, msg: str) -> None:
        self.failures.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)


def load_family_cases(root: Path, suite: str, family: str) -> dict[str, dict] | None:
    """All of `family`'s cases for `suite`, keyed by ID suffix ('' for the bare
    group case). None when the family has no fixtures dir for the suite."""
    subdir, prefix = SUITE_INPUTS[suite]
    fam_dir = root / subdir / "inputs" / family
    if not fam_dir.is_dir():
        return None
    cases: dict[str, dict] = {}
    for path in sorted(fam_dir.glob(f"{prefix}*.yaml")):
        data = yaml.safe_load(path.read_text()) or {}
        for cid, case in (data.get("cases") or {}).items():
            if not cid.startswith(prefix + "."):
                continue
            cases[cid[len(prefix) + 1 :]] = case if isinstance(case, dict) else {}
    return cases if cases else None


def case_status(case: dict) -> str:
    """'real' | 'todo' | 'na' | 'bad' (placeholder without explanation)."""
    if "model_text" in case or "chunks" in case:
        return "real"
    if not (case.get("explanation") or case.get("reason")):
        return "bad"
    if _TODO_MARKER in str(case.get("description", "")):
        return "todo"
    return "na"


def group_of(suffix: str) -> str:
    return re.match(r"\d+", suffix).group(0)


def sub_of(suffix: str) -> str:
    group = group_of(suffix)
    return suffix[len(group) + 1 :] if len(suffix) > len(group) else ""


def check_suite(
    report: Report,
    taxonomy: dict,
    root: Path,
    suite: str,
    family: str,
    batch_real_groups: set[str],
) -> None:
    spec = taxonomy["suites"][suite]
    exemption = (taxonomy.get("suite_exemptions") or {}).get(family, {}).get(suite)
    if exemption:
        return
    gaps = set((taxonomy.get("known_gaps") or {}).get(suite, {}).get(family, []))
    subdir, prefix = SUITE_INPUTS[suite]
    cases = load_family_cases(root, suite, family)
    if cases is None:
        report.fail(
            f"{family} {suite}: NO fixtures under {subdir}/inputs/{family} — the entire"
            f" {suite} corpus is missing (author {prefix}.* input YAMLs)"
        )
        return

    by_group: dict[str, dict[str, str]] = {}
    for suffix, case in cases.items():
        by_group.setdefault(group_of(suffix), {})[sub_of(suffix)] = case_status(case)

    for suffix, case in sorted(cases.items()):
        status = case_status(case)
        if status == "bad":
            report.fail(
                f"{family} {suite}: {prefix}.{suffix} is a placeholder without an"
                " explanation: — annotate why it is n/a or author the input"
            )
        elif status == "todo":
            report.warn(f"{family} {suite}: {prefix}.{suffix} not yet authored (TODO placeholder)")
        if group_of(suffix) not in spec["groups"]:
            report.fail(
                f"{family} {suite}: {prefix}.{suffix} belongs to no taxonomy group — add"
                f" group {group_of(suffix)} to case-taxonomy.yaml in this PR"
            )
        elif sub_of(suffix) not in spec["groups"][group_of(suffix)]["cases"]:
            report.fail(
                f"{family} {suite}: {prefix}.{suffix} is not in the taxonomy — add the"
                " sub-case to case-taxonomy.yaml in this PR"
            )

    for gid, group in spec["groups"].items():
        applies_ref = group.get("applies_if_real_in")
        if applies_ref is not None:
            if gid not in batch_real_groups:
                continue
            group_required = True
        else:
            group_required = bool(group.get("required"))
        present = by_group.get(gid, {})
        if not present:
            if not group_required:
                continue
            required_ids = [
                f"{prefix}.{gid}" + (f".{sub}" if sub else "")
                for sub, case in group["cases"].items()
                if case.get("required")
            ] or [f"{prefix}.{gid}.*"]
            report_line = (
                f"{family} {suite}: group {gid} ({group['title']}) has NO cases — author"
                f" {', '.join(required_ids)} or an explicit n/a placeholder"
            )
            if _all_grandfathered(gid, group, gaps):
                report.warn(f"{report_line} [grandfathered known_gap]")
            else:
                report.fail(report_line)
            continue
        for sub, case in group["cases"].items():
            if not case.get("required") or sub in present:
                continue
            case_id = f"{prefix}.{gid}" + (f".{sub}" if sub else "")
            if (f"{gid}.{sub}" if sub else gid) in gaps:
                report.warn(f"{family} {suite}: {case_id} missing [grandfathered known_gap — backfill]")
            else:
                report.fail(
                    f"{family} {suite}: required case {case_id} is missing —"
                    f" \"{case['desc']}\" (author it or add an n/a placeholder with explanation)"
                )


def _all_grandfathered(gid: str, group: dict, gaps: set[str]) -> bool:
    required = [
        (f"{gid}.{sub}" if sub else gid)
        for sub, case in group["cases"].items()
        if case.get("required")
    ]
    return bool(required) and all(rid in gaps for rid in required)


def real_groups(root: Path, suite: str, family: str) -> set[str]:
    cases = load_family_cases(root, suite, family) or {}
    return {
        group_of(suffix)
        for suffix, case in cases.items()
        if case_status(case) == "real"
    }


def check_markers(report: Report, registry: dict, family: str) -> None:
    """Marker-registration lint: the family's grammar tokens must be declared in
    parser_families.yaml `markers:`, leak-detectable, and popup-colorizable."""
    declared = (registry.get("markers") or {}).get(family)
    if declared is None:
        report.fail(
            f"{family} markers: no `markers:` entry in parser_families.yaml — declare the"
            " family's grammar tokens (pairs/singletons/leak), or an explicit empty {} for"
            " markup-less grammars; without it the leak detector and popup colorizer are blind"
        )
        return
    pairs = declared.get("pairs") or []
    singletons = declared.get("singletons") or []
    leak = declared.get("leak") or []
    for token in [t for pair in pairs for t in pair] + list(singletons) + list(leak):
        if not markers_mod._TOOL_CALL_MARKUP_RE.search(token):
            report.fail(
                f"{family} markers: declared token {token!r} is NOT matched by the leak"
                " detector (markers._TOOL_CALL_MARKUP_RE) — a leak of this token would"
                " render as a clean green cell"
            )
    for open_tok, close_tok in pairs:
        html = markup.colorize_markup(f"{open_tok}x{close_tok}", family=family)
        if "tt-orphan" in html:
            report.fail(
                f"{family} markers: pair {open_tok!r}/{close_tok!r} renders as tt-orphan in"
                " the popup colorizer — register/classify it in src/tables/markup.py"
            )
    for token in singletons:
        html = markup.colorize_markup(str(token), family=family)
        if "tt-orphan" in html:
            report.fail(
                f"{family} markers: singleton {token!r} renders as tt-orphan in the popup"
                " colorizer — register/classify it in src/tables/markup.py"
            )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--family", action="append", default=[], help="family to check (repeatable)")
    ap.add_argument("--all", action="store_true", help="check every registry + fixtures family")
    ap.add_argument(
        "--expect-reasoning",
        action="store_true",
        help="fail when a checked family has no reasoning fixtures (use when the PR adds a reasoning parser)",
    )
    ap.add_argument("--fixtures-root", type=Path, default=None, help="override the extracted-fixtures root")
    ap.add_argument("--taxonomy", type=Path, default=TAXONOMY_PATH)
    ap.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    ap.add_argument("--skip-markers", action="store_true", help="skip the marker-registration lint")
    ap.add_argument("-q", "--quiet", action="store_true", help="suppress warnings, print failures only")
    args = ap.parse_args()
    if not args.family and not args.all:
        ap.error("pass --family <f> (repeatable) or --all")

    if args.fixtures_root:
        root = args.fixtures_root
        if not (root / "toolcalling").is_dir():
            sys.exit(f"fixtures root {root} has no toolcalling/ dir")
    else:
        root = resolve_fixtures_root()
    taxonomy = yaml.safe_load(args.taxonomy.read_text())
    registry = yaml.safe_load(args.registry.read_text())
    registry_families = set(registry["families"])

    tc_dir_families = {
        p.name
        for suite in TOOLCALLING_SUITES
        for p in (root / SUITE_INPUTS[suite][0] / "inputs").glob("*")
        if p.is_dir()
    }
    reasoning_dir_families = {
        p.name for p in (root / SUITE_INPUTS["reasoning.batch"][0] / "inputs").glob("*") if p.is_dir()
    }

    report = Report()
    if args.all:
        toolcalling_families = sorted(registry_families | tc_dir_families)
        reasoning_families = sorted(reasoning_dir_families)
        for fam in sorted(tc_dir_families - registry_families):
            report.fail(f"{fam}: fixtures dir exists but family has no parser_families.yaml row")
    else:
        requested = sorted(set(args.family))
        # A family known ONLY to the reasoning corpus (e.g. gpt_oss, granite) skips
        # the toolcalling suites. Unknown families still default to toolcalling so a
        # brand-new family fails loudly before registration.
        reasoning_only = {
            f
            for f in requested
            if f in reasoning_dir_families
            and f not in registry_families
            and f not in tc_dir_families
        }
        toolcalling_families = sorted(set(requested) - reasoning_only)
        reasoning_families = sorted(
            f for f in requested if f in reasoning_dir_families or args.expect_reasoning
        )

    for fam in toolcalling_families:
        if fam not in registry_families:
            report.fail(f"{fam}: not in parser_families.yaml — register the family row first")
        batch_real = real_groups(root, "toolcalling.batch", fam)
        for suite in TOOLCALLING_SUITES:
            check_suite(report, taxonomy, root, suite, fam, batch_real)
        if not args.skip_markers:
            check_markers(report, registry, fam)
    for fam in reasoning_families:
        for suite in REASONING_SUITES:
            check_suite(report, taxonomy, root, suite, fam, set())

    if not args.quiet:
        for msg in report.warnings:
            print(f"WARN  {msg}")
    for msg in report.failures:
        print(f"FAIL  {msg}")
    checked = ", ".join(toolcalling_families) or "(none)"
    print(
        f"coverage lint: {len(report.failures)} failure(s), {len(report.warnings)} warning(s)"
        f" — toolcalling: {checked}; reasoning: {', '.join(reasoning_families) or '(none)'}"
    )
    return 1 if report.failures else 0


if __name__ == "__main__":
    sys.exit(main())
