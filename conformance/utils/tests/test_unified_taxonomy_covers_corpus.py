# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Every unified corpus scenario carries a taxonomy number.

`tax()` answers an unmapped scenario with `(9, <slug>)` instead of raising, so a case
added to `gen_unified_golden.py` without a `UNIFIED_TAX` entry still renders — it just
silently lands in group 9 under its raw slug rather than the group it belongs to. That
is a wrong answer delivered confidently: the page looks complete, the case is numbered,
and nothing says the number is a fallback.

It has already happened: `guided_json_escaped_string_args` and `guided_json_array_argument`
were added to the corpus and rendered as `UNIFIED.9.*` for a full render cycle before
anyone noticed they were missing from the map. These two tests make that a failure at
the point the case is added, and name the file to edit.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

UTILS = Path(__file__).resolve().parents[1]
SRC = UTILS / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gen_unified_golden import CLEAN, EDGE, FAMILIES, control_tokens  # noqa: E402
from unified_taxonomy import UNIFIED_GROUP_LABEL, UNIFIED_TAX, numbered_id, tax  # noqa: E402

TAXONOMY_FILE = "conformance/utils/src/unified_taxonomy.py"


def corpus_scenarios() -> list[str]:
    """Scenario slugs the generator actually emits — the first element of each case."""
    return [spec[0] for spec in (*CLEAN, *EDGE)]


def test_every_corpus_scenario_has_a_taxonomy_entry() -> None:
    unmapped = sorted(s for s in corpus_scenarios() if s not in UNIFIED_TAX)
    assert not unmapped, (
        f"{len(unmapped)} corpus scenario(s) have no UNIFIED_TAX entry and would render "
        f"as UNIFIED.9.<slug> instead of their real group: {unmapped}. "
        f"Add them to UNIFIED_TAX in {TAXONOMY_FILE}."
    )


def test_taxonomy_has_no_entry_without_a_corpus_case() -> None:
    """The other direction: a stale entry means a case was renamed or deleted and the map
    still claims a number for it, so the number is reserved against nothing."""
    scenarios = set(corpus_scenarios())
    stale = sorted(s for s in UNIFIED_TAX if s not in scenarios)
    assert not stale, (
        f"{len(stale)} UNIFIED_TAX entr(ies) name a scenario the corpus does not emit: "
        f"{stale}. Remove them from {TAXONOMY_FILE} or restore the case in "
        f"conformance/utils/src/gen_unified_golden.py."
    )


def test_no_corpus_scenario_falls_back_to_group_9() -> None:
    """Belt and braces on the fallback itself: assert through `tax()`, the function the
    renderer calls, so this still fails if the fallback moves or changes shape."""
    fell_back = sorted(s for s in corpus_scenarios() if tax(s)[0] == 9)
    assert not fell_back, f"scenario(s) resolved to the group-9 fallback: {fell_back}"


def test_every_used_group_has_a_label() -> None:
    """An unlabelled group renders a numbered heading with no name."""
    used = {tax(s)[0] for s in corpus_scenarios()}
    missing = sorted(g for g in used if g not in UNIFIED_GROUP_LABEL)
    assert not missing, (
        f"group(s) {missing} are used by the corpus but absent from UNIFIED_GROUP_LABEL "
        f"in {TAXONOMY_FILE}."
    )


# --- End-to-end test cases: the SAME mapping is written in two places -----------
# Case descriptions in gen_unified_golden.py carry `End-to-end: <case> (e2e case-NNNN)` tags, and
# UNIFIED_CASES.md repeats them in its artifact-index table. Two copies of one fact drift
# — that is the defect this whole surface keeps hitting — so pin them to each other.

CASES_MD = UTILS / "lib" / "parsers" / "UNIFIED_CASES.md"
_E2E_TAG = re.compile(r"\be2e case-(\d{4})-")
_MD_ROW = re.compile(r"^\|\s*`(\d+\.[a-z])`\s*\|\s*`([^`]+)`\s*\|\s*`end-to-end case-(\d{4})-", re.M)


def _e2e_ids_from_descriptions() -> dict[str, set[str]]:
    """numbered case id -> {'0047', ...} as declared in the generator's descriptions."""
    out: dict[str, set[str]] = {}
    for spec in (*CLEAN, *EDGE):
        scenario, desc = spec[0], spec[1]
        ids = set(_E2E_TAG.findall(desc))
        if ids:
            out.setdefault(f"UNIFIED.{tax(scenario)[0]}.{tax(scenario)[1]}", set()).update(ids)
    return out


def _e2e_ids_from_markdown() -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for num, _live_case, artifact_id in _MD_ROW.findall(CASES_MD.read_text(encoding="utf-8")):
        out.setdefault(f"UNIFIED.{num}", set()).add(artifact_id)
    return out


def test_e2e_tags_agree_between_descriptions_and_markdown() -> None:
    from_desc, from_md = _e2e_ids_from_descriptions(), _e2e_ids_from_markdown()
    assert from_desc, "no `e2e case-NNNN-` citations found in generator descriptions — did the format change?"
    assert from_md, f"no artifact-index rows parsed out of {CASES_MD.name} — did the table change?"
    # A description may cite ONE representative for a bulk group (10.b stands for 32 e2e
    # cases), so the relation is subset, not equality: every filename a description names
    # must be a real row in the index. Equality would force 32 filenames into one popup.
    unindexed = {k: sorted(v - from_md.get(k, set())) for k, v in from_desc.items() if v - from_md.get(k, set())}
    assert not unindexed, (
        f"description(s) cite e2e artifacts absent from the index table in {CASES_MD.name}: {unindexed}. "
        "Add the row, or fix the filename."
    )
    untagged = sorted(set(from_md) - set(from_desc))
    assert not untagged, (
        f"index table has rows for {untagged} but no description cites them — the tag was dropped "
        "from gen_unified_golden.py."
    )


# --- Cross-suite case references resolve ---------------------------------------
# Descriptions and the docs cite sibling suites' cases ("streaming form of X"). Those
# citations were BARE and some were wrong: `REASONING.2.a` named nothing, because the real
# id carries a stage segment (`REASONING.batch.2.a`). A reader following it finds nothing
# and nothing complained. Require the full name AND require it to exist.

_SIBLING_DOCS = {
    "TOOLCALLING.streamv2": UTILS / "lib" / "parsers" / "TOOLCALLING_STREAMING_V2_CASES.md",
    "TOOLCALLING.batch": UTILS / "lib" / "parsers" / "TOOLCALLING_CASES.md",
    "REASONING.batch": UTILS / "lib" / "parsers" / "REASONING_CASES.md",
}
_QUALIFIED = re.compile(r"\b(?:TOOLCALLING|REASONING)\.(?:batch|streamv2)\.\d+(?:\.[a-z])?")
# a stage segment with no axis in front of it — the shape that named nothing
_BARE = re.compile(r"(?<![.\w])(?:batch|streamv2)\.\d+(?:\.[a-z])?")
_CITING = [UTILS / "lib" / "parsers" / "UNIFIED_CASES.md", SRC / "gen_unified_golden.py"]


def test_sibling_case_references_are_fully_qualified() -> None:
    offenders = {f.name: sorted(set(_BARE.findall(f.read_text(encoding="utf-8")))) for f in _CITING}
    offenders = {k: v for k, v in offenders.items() if v}
    assert not offenders, (
        f"unqualified case references (missing the axis prefix): {offenders}. "
        "Cite the full name, e.g. `TOOLCALLING.streamv2.2.a`, not `streamv2.2.a`."
    )


def test_sibling_case_references_exist() -> None:
    bodies = {k: p.read_text(encoding="utf-8") for k, p in _SIBLING_DOCS.items()}
    dangling: dict[str, list[str]] = {}
    for f in _CITING:
        bad = [
            ref
            for ref in sorted(set(_QUALIFIED.findall(f.read_text(encoding="utf-8"))))
            # group-level ids (`...streamv2.2`) have no entry of their own; a sub-case does
            if not any(ref.startswith(k) and ref in body for k, body in bodies.items())
        ]
        if bad:
            dangling[f.name] = bad
    assert not dangling, (
        f"case references that resolve to nothing: {dangling}. "
        f"Defined ids live in {', '.join(p.name for p in _SIBLING_DOCS.values())}."
    )


# --- e2e completeness: every end-to-end case has a home in the taxonomy ---------
# The report and its JSON artifacts live outside this repo, so `e2e_cases.json` is the
# committed snapshot CI can check. Completeness runs BOTH ways: no e2e case may be left
# unclassified, and no mapping may point at a UNIFIED case that does not exist.

E2E_MANIFEST = SRC / "e2e_cases.json"


def _e2e() -> dict:
    return json.loads(E2E_MANIFEST.read_text(encoding="utf-8"))


def test_every_e2e_case_is_classified() -> None:
    cases = _e2e()["cases"]
    unclassified = sorted(k for k, v in cases.items() if not v.get("unified"))
    assert not unclassified, (
        f"{len(unclassified)} end-to-end case(s) map to no UNIFIED case: {unclassified}. "
        "Give each one the UNIFIED case whose output SHAPE covers it (UNIFIED may be a "
        f"superset), or record why none can, in {E2E_MANIFEST.name}."
    )


def test_e2e_mappings_name_real_unified_cases() -> None:
    numbered = {numbered_id(s).removeprefix("UNIFIED.") for s in corpus_scenarios()}
    bad = sorted({u for v in _e2e()["cases"].values() for u in v.get("unified", []) if u not in numbered})
    assert not bad, (
        f"e2e mapping(s) name UNIFIED cases that do not exist: {bad}. "
        f"Valid ids come from UNIFIED_TAX in {TAXONOMY_FILE}."
    )


def test_e2e_manifest_totals_are_self_consistent() -> None:
    m = _e2e()
    assert m["distinct_cases"] == len(m["cases"]), "distinct_cases disagrees with the cases map"
    artifacts = sum(len(v["artifacts"]) for v in m["cases"].values())
    assert artifacts == m["logical_cases"], (
        f"{artifacts} artifacts across cases but logical_cases says {m['logical_cases']} — "
        "the snapshot is stale; regenerate it from the report."
    )


def test_every_e2e_artifact_appears_in_the_index_table() -> None:
    """The Artifact index in the docs must list every artifact the manifest knows about."""
    listed = set(re.findall(r"`end-to-end (case-[\w.-]+\.json)`", CASES_MD.read_text(encoding="utf-8")))
    known = {a for v in _e2e()["cases"].values() for a in v["artifacts"]}
    missing = sorted(known - listed)
    assert not missing, (
        f"{len(missing)} e2e artifact(s) are in {E2E_MANIFEST.name} but absent from the "
        f"Artifact index in {CASES_MD.name}: {missing[:5]}{' …' if len(missing) > 5 else ''}"
    )

def test_marker_inside_argument_golden_matches_the_input_marker() -> None:
    """The I7 fidelity case must assert the FAMILY'S OWN marker, not a placeholder.

    Authoring a stand-in like "MARKER" in the golden while feeding the real closer
    in the input validates nothing: the case would pass whatever the parser did to
    the argument. The golden argument and the input must carry the same bytes.
    """
    case = next(
        c for c in list(CLEAN) + list(EDGE) if c[0] == "guided_json_marker_inside_argument"
    )
    per_family = case[-1]
    for fam in FAMILIES:
        entry = per_family[fam]
        raw_input, fill = entry[0], entry[-1]
        expected = control_tokens(fam)[1]
        assert fill == expected, f"{fam}: golden fill {fill!r} is not the family marker"
        assert expected in raw_input, f"{fam}: input {raw_input!r} lacks {expected!r}"


def test_every_rendered_config_key_exists_in_the_emitted_init() -> None:
    """A producer/renderer rename must not silently render every case as "unset".

    `conformance_view.js` reads `init[spec.key]` for each `CONFIG_KEYS` entry. When
    `prefill` was renamed to `starting_state` the producers moved and the renderer
    did not, so every case's popup claimed the request setting was never chosen —
    and nothing failed, because a missing key just reads as the default. This pins
    the two sides together.
    """
    js = (UTILS / "src/assets/conformance_view.js").read_text()
    keys = set(re.findall(r"\{\s*key:\s*'([a-z_]+)'", js))
    assert keys, "CONFIG_KEYS not found in conformance_view.js"

    emitted = set()
    for case in list(CLEAN) + list(EDGE):
        init = next((f for f in case if isinstance(f, dict) and "tool_output_mode" in f), None)
        if init:
            emitted |= set(init)
    missing = sorted(keys - emitted)
    assert not missing, (
        f"conformance_view.js renders {missing}, which no case emits in `init` — "
        "every case would show that setting as unset"
    )
