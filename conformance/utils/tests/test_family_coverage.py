# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Guards for the fixture-coverage lint and the single-source family markers
(DIS-2442, distilled from the PR #120 review):

  * the lint is green on the committed corpus (CI gate stays meaningful);
  * an inkling-shaped family (batch groups 6/8/30 missing, no stream corpus, no
    marker registration) FAILS the lint — the exact gap class three reviewers had
    to find by hand on PR #120;
  * every parser_families.yaml row has a markers declaration and vice versa;
  * the YAML-derived leak detector is a strict superset of the retired hardcoded
    regex on the committed corpus (no silently-lost `↯` detections);
  * declared marker tokens render non-orphan in the popup colorizer, including
    tokens the heuristics cannot classify (the red-orphan popup class).
"""
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml

UTILS = Path(__file__).resolve().parents[1]
SRC = UTILS / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(UTILS) not in sys.path:
    sys.path.insert(0, str(UTILS))

import check_family_coverage as cfc  # noqa: E402
import markers as markers_mod  # noqa: E402
from tables import markup  # noqa: E402


def _ensure_fixtures() -> Path:
    # Manifest-pinned snapshot, immune to sibling checkouts repointing the shared
    # cache symlink mid-run (the flake class conformance/README documents).
    return cfc.resolve_fixtures_root()


def _registry() -> dict:
    return yaml.safe_load((SRC / "parser_families.yaml").read_text())


def _taxonomy() -> dict:
    return yaml.safe_load(cfc.TAXONOMY_PATH.read_text())


def test_lint_green_on_committed_corpus() -> None:
    """The committed corpus passes with zero failures (warnings — TODO placeholders
    and grandfathered known_gaps — are the backfill list, never CI-red)."""
    root = _ensure_fixtures()
    taxonomy = _taxonomy()
    registry = _registry()
    report = cfc.Report()
    for fam in sorted(registry["families"]):
        batch_real = cfc.real_groups(root, "toolcalling.batch", fam)
        for suite in cfc.TOOLCALLING_SUITES:
            cfc.check_suite(report, taxonomy, root, suite, fam, batch_real)
        cfc.check_markers(report, registry, fam)
    assert report.failures == []


def test_lint_fails_on_inkling_shaped_family(tmp_path) -> None:
    """A family with the PR #120 gap profile fails the lint: batch groups 6/8/30
    absent, the entire stream-v2 corpus missing, and no `markers:` registration."""
    root = _ensure_fixtures()
    fake_root = tmp_path / "fixtures"
    src_fam = root / "toolcalling/fixtures-batch-v1/inputs/hermes"
    dst_fam = fake_root / "toolcalling/fixtures-batch-v1/inputs/newfam"
    dst_fam.mkdir(parents=True)
    (fake_root / "toolcalling/fixtures-stream-v2/inputs").mkdir(parents=True)
    (fake_root / "reasoning/fixtures-v1/inputs").mkdir(parents=True)
    for path in src_fam.glob("TOOLCALLING.batch*.yaml"):
        data = yaml.safe_load(path.read_text())
        data["family"] = "newfam"
        cases = {
            cid: case
            for cid, case in data["cases"].items()
            if cfc.group_of(cid[len("TOOLCALLING.batch") + 1 :]) not in ("6", "8", "30")
        }
        if not cases:
            continue
        data["cases"] = cases
        (dst_fam / path.name).write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False))

    registry = _registry()
    registry["families"]["newfam"] = {"vllm_python": None}
    taxonomy = _taxonomy()
    report = cfc.Report()
    batch_real = cfc.real_groups(fake_root, "toolcalling.batch", "newfam")
    for suite in cfc.TOOLCALLING_SUITES:
        cfc.check_suite(report, taxonomy, fake_root, suite, "newfam", batch_real)
    cfc.check_markers(report, registry, "newfam")

    text = "\n".join(report.failures)
    assert "group 6 (Empty arguments) has NO cases" in text
    assert "group 8 (Narration around calls) has NO cases" in text
    assert "group 30 (Own-grammar delimiters inside a string argument (single call)) has NO cases" in text
    assert "the entire toolcalling.stream corpus is missing" in text
    assert "no `markers:` entry in parser_families.yaml" in text


def test_taxonomy_case_entries_well_formed() -> None:
    """Every taxonomy case entry is exactly {desc, required[, note]} with a non-empty
    desc — catches YAML flow-mapping typos (an unquoted comma inside a desc silently
    splits the entry into junk keys)."""
    for suite, spec in _taxonomy()["suites"].items():
        for gid, group in spec["groups"].items():
            assert group.get("title"), (suite, gid)
            for sub, case in group["cases"].items():
                assert set(case) <= {"desc", "required", "note"}, (suite, gid, sub, case)
                assert isinstance(case.get("desc"), str) and case["desc"], (suite, gid, sub)
                assert isinstance(case.get("required"), bool), (suite, gid, sub)


def test_markers_declared_for_every_family() -> None:
    """markers: and families: stay key-aligned — a new family cannot register a
    parser row without declaring (possibly empty) grammar tokens."""
    registry = _registry()
    assert set(registry["markers"]) == set(registry["families"])


def test_leak_regex_superset_of_retired_hardcoded_regex() -> None:
    """The YAML-derived leak detector matches everywhere the retired hardcoded
    regex did, on every normal_text in the committed corpus — token registration
    can only ADD detections, never silently lose them."""
    old = re.compile(
        r"</?tool_call|</?tool_calls|<\|tool_call|<\|tool_calls|"
        r"<\|(?:channel|message|call|python_tag)\|>|"
        r"</?TOOLCALL|TOOL_CALLS|<｜(?:DSML｜)?(?:tool|tool▁call|tool▁calls)|"
        r"<｜DSML｜|</?minimax:tool_call|</?invoke|</?arg_key|</?arg_value"
    )
    root = _ensure_fixtures()
    texts: set[str] = set()

    def walk(node) -> None:
        if isinstance(node, dict):
            nt = node.get("normal_text")
            if isinstance(nt, str):
                texts.add(nt)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    for path in root.glob("toolcalling/**/*.yaml"):
        walk(yaml.safe_load(path.read_text()))
    assert texts, "no corpus normal_texts found — fixtures not extracted?"
    lost = [t for t in texts if old.search(t) and not markers_mod._TOOL_CALL_MARKUP_RE.search(t)]
    assert lost == []


def test_declared_tokens_render_non_orphan() -> None:
    """Every declared pair/singleton renders without tt-orphan, including tokens the
    heuristic classifier cannot type (deepseek `<｜tool▁sep｜>` was a red orphan
    before the declared-marker lookup; Inkling's `<|message_model|>` class)."""
    registry = _registry()
    for fam, decl in registry["markers"].items():
        for open_tok, close_tok in decl.get("pairs") or []:
            assert "tt-orphan" not in markup.colorize_markup(f"{open_tok}x{close_tok}", family=fam), (fam, open_tok)
        for tok in decl.get("singletons") or []:
            assert "tt-orphan" not in markup.colorize_markup(str(tok), family=fam), (fam, tok)


def test_reasoning_only_family_skips_toolcalling_suites() -> None:
    """`--family gpt_oss` (reasoning-only corpus family) must check the reasoning
    suites ONLY — not fail on missing toolcalling registry/fixtures."""
    proc = subprocess.run(
        [sys.executable, str(SRC / "check_family_coverage.py"), "--family", "gpt_oss", "--expect-reasoning"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "toolcalling" not in "".join(
        line for line in proc.stdout.splitlines() if line.startswith("FAIL")
    )
    assert "reasoning: gpt_oss" in proc.stdout


def test_undeclared_pipe_token_still_orphans() -> None:
    """The declared-marker lookup is per-family and exact: an Inkling-style token
    that is NOT declared keeps rendering as tt-orphan (the lint, not the colorizer,
    is what forces registration)."""
    assert "tt-orphan" in markup.colorize_markup("<|message_model|>hello<|end_message|>", family="hermes")
