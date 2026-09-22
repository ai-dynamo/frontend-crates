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

from collections import defaultdict
import json
import re
import sys
from pathlib import Path

UTILS = Path(__file__).resolve().parents[1]
SRC = UTILS / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pytest  # noqa: E402
import yaml  # noqa: E402

import gen_unified_golden as G  # noqa: E402
import unified_history  # noqa: E402
from fixture_disposition import historical_unified_case_key  # noqa: E402
from gen_unified_golden import (  # noqa: E402
    CLEAN,
    OnlyFamilies,
    EDGE,
    FAMILIES,
    build_cases,
    control_tokens,
    invoke_header_prefix,
)
from unified_taxonomy import (  # noqa: E402
    UNIFIED_GROUP_LABEL,
    UNIFIED_TAX,
    case_label,
    numbered_id,
    tax,
    taxonomy_sort_key,
)

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


def test_invoke_header_prefix_is_inner_and_unterminated() -> None:
    for family in FAMILIES:
        prefix = invoke_header_prefix(family)
        assert prefix
        assert prefix == prefix.lstrip()
        if family != "deepseek_v41":
            assert not prefix.startswith(control_tokens(family)[2])
        assert prefix.rsplit(">", 1)[-1]


def test_request_scoped_cases_never_claim_vllm_match() -> None:
    for family in FAMILIES:
        for case_id, case in build_cases(family).items():
            init = case["init"]
            request_scoped = init.get("starting_state") != "None" or init.get("tool_output_mode") != "Native"
            if request_scoped:
                assert case["expect"]["vllm"]["verdict"] == "diverge", case_id


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


def test_case_labels_keep_gemma_specific_cases_out_of_the_generic_guided_series() -> None:
    """Gemma-only call-prefix cases use their own named numeric group."""
    assert tax("guided_json_quoted_bare_header_in_answer") == (35, "1")
    assert tax("guided_json_quoted_bare_tool_header_in_answer") == ("muse", "1")
    assert tax("guided_json_quoted_bare_header_after_payload") == (35, "2")
    assert tax("guided_json_bare_tool_header_recovers_inside_a_thought") == (34, "7")
    assert tax("gemma4_guided_json_visible_call_prose_before_reasoning") == ("gemma", "1")
    assert tax("gemma4_guided_json_malformed_call_prefix_before_reasoning") == ("gemma", "2")
    assert case_label("guided_json_quoted_bare_header_in_answer") == "35-1"
    assert case_label("guided_json_quoted_bare_tool_header_in_answer") == "muse-1"
    assert case_label("guided_json_quoted_bare_header_after_payload") == "35-2"
    assert case_label("guided_json_bare_tool_header_recovers_inside_a_thought") == "34-7"
    assert case_label("gemma4_guided_json_visible_call_prose_before_reasoning") == "gemma-1"
    assert case_label("gemma4_guided_json_malformed_call_prefix_before_reasoning") == "gemma-2"
    assert numbered_id("guided_json_quoted_bare_header_in_answer") == "UNIFIED.35-1"
    assert numbered_id("gemma4_guided_json_visible_call_prose_before_reasoning") == "UNIFIED.gemma-1"
    assert case_label("guided_json_invalid_call") == "31-1"
    assert numbered_id("guided_json_invalid_call") == "UNIFIED.31-1"
    guided = [sub for group, sub in UNIFIED_TAX.values() if isinstance(group, int) and 30 <= group <= 35]
    assert all(sub.isdecimal() for sub in guided)

    ordered = sorted(
        (
            "guided_json_quoted_bare_header_in_answer",
            "guided_json_quoted_bare_tool_header_in_answer",
            "guided_json_quoted_bare_header_after_payload",
            "guided_json_bare_tool_header_recovers_inside_a_thought",
        ),
        key=taxonomy_sort_key,
    )
    assert [case_label(scenario) for scenario in ordered] == ["34-7", "35-1", "35-2", "muse-1"]


def test_kimi_k3_cases_use_model_specific_numeric_suffixes() -> None:
    k3 = [
        "kimi_k3_typed_argument_values",
        "kimi_k3_raw_json_arguments",
        "kimi_k3_spaced_xtml_markers",
        "kimi_k3_message_end_after_response",
        "kimi_k3_elided_think_close_to_response",
        "kimi_k3_malformed_call_then_valid",
        "kimi_k3_raw_json_eof",
        "kimi_k3_guided_native_wrapper",
    ]
    assert [case_label(scenario) for scenario in k3] == [f"kimi-{i}" for i in range(1, 9)]
    assert [numbered_id(scenario) for scenario in k3] == [
        f"UNIFIED.kimi-{i}" for i in range(1, 9)
    ]


# --- End-to-end test cases: the SAME mapping is written in two places -----------
# Case descriptions in gen_unified_golden.py carry `End-to-end: <case> (e2e case-NNNN)` tags, and
# UNIFIED_CASES.md repeats them in its artifact-index table. Two copies of one fact drift
# — that is the defect this whole surface keeps hitting — so pin them to each other.

CASES_MD = UTILS / "lib" / "parsers" / "UNIFIED_CASES.md"
_E2E_TAG = re.compile(r"\be2e case-(\d{4})-")
_MD_ROW = re.compile(r"^\|\s*`([\w]+(?:\.[a-z]|-\d+))`\s*\|\s*`([^`]+)`\s*\|\s*`end-to-end case-(\d{4})-", re.M)


def _e2e_ids_from_descriptions() -> dict[str, set[str]]:
    """numbered case id -> {'0047', ...} as declared in the generator's descriptions."""
    out: dict[str, set[str]] = {}
    for spec in (*CLEAN, *EDGE):
        scenario, desc = spec[0], spec[1]
        ids = set(_E2E_TAG.findall(desc))
        if ids:
            out.setdefault(numbered_id(scenario), set()).update(ids)
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

def test_no_two_scenarios_have_identical_behaviour() -> None:
    """Two names for one behaviour is worse than a gap.

    A generated crossing collided with a hand-authored scenario three times
    (`guided_json_valid_*` vs `guided_json_tool_*`), giving 3 x 3 families = 9 cases
    with byte-identical `(input, init, golden)`. They inflated the case count while
    testing nothing new, and the pair would drift apart on the next edit.

    Reads the SPEC (`CLEAN`/`EDGE` in the generator), not `conformance/unified/`:
    that tree is a gitignored build artifact, so a test that reads it passes locally
    and fails in CI — which is exactly what the first version of this did.
    """
    for fam in FAMILIES:
        seen = defaultdict(list)
        for name, case in build_cases(fam).items():
            seen[
                json.dumps(
                    {
                        "input": case["input"],
                        "init": case["init"],
                        "golden": case["golden"],
                    },
                    sort_keys=True,
                )
            ].append(name)
        dupes = {k: v for k, v in seen.items() if len(v) > 1}
        assert not dupes, f"{fam}: scenarios with identical behaviour: {list(dupes.values())}"


def test_deepseek_v41_follows_declared_scope_including_prefilled_cases() -> None:
    declared = {
        spec[0]
        for spec in (*CLEAN, *EDGE)
        if not isinstance(spec[-1], OnlyFamilies) or "deepseek_v41" in spec[-1]
    }
    actual = {case_id.split(".", 2)[1] for case_id in build_cases("deepseek_v41")}
    assert actual == declared


def test_deepseek_v41_guided_narration_uses_an_unfinished_dsml_invoke() -> None:
    case = build_cases("deepseek_v41")[
        "UNIFIED.guided_json_narrated_invoke_in_reasoning.deepseek_v41"
    ]
    assert '<think>I\'ll use <｜DSML｜ invoke name=" next</think>' in case["input"]
    assert "<think>I'll use <｜DSML｜ calls>" not in case["input"]


@pytest.mark.parametrize("family,prefix", [
    ("deepseek_v4", '<｜DSML｜invoke name="'),
    ("deepseek_v41", '<｜DSML｜ invoke name="'),
    ("gemma4", "call:"),
    ("kimi_k2", "<|tool_call_begin|>functions."),
    ("kimi_k3", '<|open|>call tool="'),
    ("muse_glimmer", '<atem:invoke name="'),
    ("qwen3", "<function="),
])
def test_historical_bare_header_stimulus_keeps_30m(family, prefix):
    scenario = "guided_json_gt_in_argument_bare_opener"
    case = build_cases(family)[f"UNIFIED.{scenario}.{family}"]
    assert numbered_id(scenario) == "UNIFIED.30-13"
    assert case["input"] == prefix + '[{"name": "get_weather", "arguments": {"city": "a > b"}}]'
    assert case["init"] == {
        "starting_state": "None", "tool_output_mode": "GuidedJson", "named_tool": None,
    }
    assert case["golden"] == [
        {"kind": "tool_call", "name": "get_weather", "arguments": {"city": "a > b"}},
    ]


def test_deepseek_v41_empty_calls_envelope_keeps_4b():
    scenario = "tool_markup_only_emits_nothing"
    case = build_cases("deepseek_v41")[f"UNIFIED.{scenario}.deepseek_v41"]
    assert numbered_id(scenario) == "UNIFIED.4-2"
    assert case["input"] == "<｜DSML｜ calls></｜DSML｜ calls>"
    assert case["golden"] == []
    assert case["init"] == {
        "starting_state": "None", "tool_output_mode": "Native", "named_tool": None,
    }


def test_response_state_uses_only_control_marker_contracts() -> None:
    response_scenarios = {
        "prefilled_response_reasoning_markers_literal",
        "guided_json_quoted_bare_header_in_answer",
        "guided_json_quoted_bare_header_after_payload",
    }
    for family in FAMILIES:
        cases = build_cases(family)
        response_cases = {
            case_id.split(".", 2)[1]: case
            for case_id, case in cases.items()
            if case["init"]["starting_state"] == "Response"
        }
        extra = {"guided_json_quoted_bare_tool_header_in_answer"} if family == "muse_glimmer" else set()
        assert set(response_cases) == response_scenarios | extra
        marker = "<|message|>" if family == "muse_glimmer" else control_tokens(family)[0]
        for case in response_cases.values():
            assert marker in case["input"]


# --- scenario scope must be DECLARED, never inferred from a gap -----------------

def test_an_undeclared_missing_family_fails_generation():
    """An accidentally omitted family breaks generation instead of reading as n/a.

    This is the invariant a blanket `if fam not in per_fam: continue` destroyed: a
    missing input silently became an accepted skip, so real missing coverage rendered
    as the same "not applicable" cell the corpus uses for a genuine structural gap. A
    reviewer cannot tell those apart, and the whole point of the table is telling them
    apart.
    """
    scenario = (
        "deliberately_incomplete_scenario",
        "authored for one family only, WITHOUT declaring that scope",
        [],
        [{"kind": "text", "text": "x"}],
        None,
        {"muse_glimmer": ("x", G.M, G.M)},
    )
    original = list(G.EDGE)
    G.EDGE.append(scenario)
    try:
        missing = sorted(set(G.FAMILIES) - {"muse_glimmer"})
        assert missing, "fixture assumes more than one family exists"
        with pytest.raises(KeyError, match="deliberately_incomplete_scenario"):
            for fam in missing:
                G.build_cases(fam)
    finally:
        G.EDGE[:] = original


def test_request_state_boundary_scenarios_generate_for_every_family():
    """The shared request-state boundaries have a family-specific input.

    Muse exercises dynamic recipient headers, while marker-pair families exercise their
    own reasoning/tool boundaries under the same request initialization. A missing
    family here would be a coverage gap, not an unsupported grammar.
    """
    scoped = {
        "guided_json_quoted_bare_header_in_answer",
        "guided_json_quoted_bare_header_after_payload",
        "guided_json_bare_tool_header_recovers_inside_a_thought",
    }
    for fam in FAMILIES:
        names = {k.split(".", 2)[1] for k in build_cases(fam)}
        present = scoped & names
        assert present == scoped, f"{fam} is missing {scoped - present}"


def test_scenario_families_matches_declared_scope():
    scoped = {
        "guided_json_quoted_bare_header_in_answer": set(FAMILIES),
        "guided_json_quoted_bare_tool_header_in_answer": {"muse_glimmer"},
        "gemma4_guided_json_visible_call_prose_before_reasoning": {"gemma4"},
        "gemma4_guided_json_malformed_call_prefix_before_reasoning": {"gemma4"},
    }
    for scenario, families in scoped.items():
        assert G.scenario_families(scenario) == families
    assert G.scenario_families("tool_only") == set(FAMILIES)


def test_only_families_rejects_an_empty_or_unknown_scope():
    """A scope that names nothing, or names a family that does not exist, is a typo."""
    with pytest.raises(ValueError, match="declares nothing"):
        OnlyFamilies({})
    with pytest.raises(ValueError, match="do not exist"):
        OnlyFamilies({"no_such_family": ("x",)})


# --- generated YAML must round-trip every authored byte -------------------------

def _emitted_spec(fam: str):
    """The generated golden spec, produced and reloaded IN MEMORY.

    Never read from `conformance/unified/golden_spec/`: that whole tree is gitignored
    (`.gitignore:32`) and does not exist in a clean checkout, so a gate that reads it
    passes locally on leftover state and errors out in CI — which is to say it guards
    nothing where it matters. `emit_yaml` is tracked source, so emitting and reloading
    here tests the same emitter with no workspace dependency.
    """
    import yaml

    return yaml.safe_load(G.emit_yaml(fam))["cases"]


def test_every_authored_case_survives_emission_and_reload():
    """What the generator CONSTRUCTS is what the corpus measures.

    `input: |-` lets YAML infer a block's indentation from its first non-empty line,
    so an input that legitimately BEGINS with a space loses that byte on reload — the
    reader cannot tell content-space from indent-space. `34-7` is authored with a
    leading space (the bare-header form muse accepts when the prompt consumed the
    turn's framing) and was emitted at 110 bytes and reloaded at 109. The corpus was
    scoring the parser against an input nobody wrote.

    Asserted over EVERY case and all three authored fields, not just the one that
    bit us: any future field that grows a leading-whitespace value fails here rather
    than silently measuring something else.
    """
    for fam in FAMILIES:
        loaded_cases = _emitted_spec(fam)
        for cid, case in build_cases(fam).items():
            loaded = loaded_cases[cid]
            assert loaded["input"] == case["input"], (
                f"{cid}: input changed across emission/reload — "
                f"{len(case['input'])} bytes out, {len(loaded['input'])} back"
            )
            assert loaded["golden"] == case["golden"], f"{cid}: golden changed"
            assert loaded["init"] == case["init"], f"{cid}: init changed"


# --- counts live where they can be checked, not in registry prose ---------------

def test_unified_case_counts_match_the_generator():
    """Name exactly what is counted, and count it from the generator.

    Counted: cases the generator EMITS per family — the shared scenarios every family
    gets, plus any the scenario itself scoped with `OnlyFamilies`. Not counted: names
    reserved in the taxonomy map that the generator does not emit, which is how a stale
    denominator survived in the prose before.
    """
    per_family = {fam: len(build_cases(fam)) for fam in FAMILIES}
    for fam in FAMILIES:
        family_specific = {
            "deepseek_v4": 80,
            "deepseek_v41": 80,
            "gemma4": 82,
            "kimi_k2": 80,
            "kimi_k3": 88,
            "muse_glimmer": 81,
            "qwen3": 80,
        }[fam]
        assert per_family[fam] == family_specific, f"{fam} diverged from the expected case count"
    assert sum(per_family.values()) == 571


def test_deferred_case_ids_are_not_in_the_active_taxonomy():
    deferred = {"1-2", "7-3", "30-14", "50-1", "50-2"} | {
        f"31-{number}" for number in range(31, 41)
    }
    assert len(UNIFIED_TAX) == 91
    assert not {f"UNIFIED.{case_id}" for case_id in deferred} & {
        numbered_id(scenario) for scenario in UNIFIED_TAX
    }


def _assert_documented_deepseek_counts(text):
    counts = re.search(r"current corpus emits (\d+) of the (\d+) taxonomy cases", text)
    exclusions = re.search(r"The (\d+) omitted cases", text)
    assert counts and exclusions, "DeepSeek V4.1 applicability counts are missing"
    applicable = len(build_cases("deepseek_v41"))
    assert tuple(map(int, counts.groups())) == (applicable, len(UNIFIED_TAX))
    assert int(exclusions.group(1)) == len(UNIFIED_TAX) - applicable


def test_documented_deepseek_counts_match_generated_applicability():
    text = (UTILS / "lib/parsers/UNIFIED_CASES.md").read_text()
    _assert_documented_deepseek_counts(text)


@pytest.mark.parametrize("pattern", [r"current corpus emits \d+", r"The \d+ omitted cases"])
def test_documented_deepseek_counts_reject_stale_prose(pattern):
    text = (UTILS / "lib/parsers/UNIFIED_CASES.md").read_text()
    _assert_documented_deepseek_counts(text)
    match = re.search(pattern, text)
    assert match
    changed = re.sub(r"\d+", "0", match.group())
    with pytest.raises(AssertionError):
        _assert_documented_deepseek_counts(text.replace(match.group(), changed, 1))


def test_kimi_k3_corpus_uses_xtml_not_kimi_k2_wire_syntax():
    forbidden = ("<think>", "</think>", "<|tool_calls_section_begin|>",
                 "<|tool_call_begin|>", "<|tool_call_argument_begin|>")
    cases = build_cases("kimi_k3")
    assert cases
    for case_id, case in cases.items():
        leaked = [marker for marker in forbidden if marker in case["input"]]
        assert not leaked, f"{case_id} copied Kimi K2 wire syntax: {leaked}"


def test_kimi_k3_manifest_is_canonical_and_alias_free():
    manifest = yaml.safe_load((SRC / "parser_families.yaml").read_text())
    assert "kimi_k3" in manifest["families"]
    assert "kimi_k3" in manifest["markers"]
    assert "kimi_k3" in manifest["unified"]
    assert "kimi-k3" not in manifest["families"]
    assert "kimi-k3" not in manifest["markers"]
    assert "kimi-k3" not in manifest["unified"]


def test_kimi_k3_response_framing_covers_visible_text_after_tools_and_between_calls():
    cases = build_cases("kimi_k3")
    after = cases["UNIFIED.trailing_text_after_tool.kimi_k3"]
    between = cases["UNIFIED.text_between_calls.kimi_k3"]
    assert "<|close|>tools<|sep|><|open|>response<|sep|>" in after["input"]
    assert [event["kind"] for event in after["golden"]] == ["tool_call", "text"]
    assert "<|close|>tools<|sep|><|open|>response<|sep|>" in between["input"]
    assert [event["kind"] for event in between["golden"]] == [
        "tool_call", "text", "tool_call"
    ]


def test_registry_prose_states_no_unified_case_count():
    """`parser_families.yaml` must not carry a case count.

    A number there is unverifiable against anything and drifts the moment a scenario is
    added — it claimed "all 81" while the generator emitted 86 for muse and 332 overall.
    Counts belong in the test above, where they are computed.
    """
    import re

    registry = (SRC / "parser_families.yaml").read_text()
    stale = re.findall(r"\b\d+\s+unified scenarios\b", registry)
    assert not stale, f"parser_families.yaml states a unified case count: {stale}"


# --- the corpus must be identical at EVERY layer, not just the first two --------

def _scenario_of(cid: str, fam: str) -> str:
    """`UNIFIED.<scenario>.<family>` -> `<scenario>`; anything else is already one."""
    if cid.startswith("UNIFIED.") and cid.endswith(f".{fam}"):
        return cid[len("UNIFIED.") : -(len(fam) + 1)]
    return cid


def _packed_layer(root: Path, directory: str, fam: str, field_map):
    """Case records from a PACKAGED shard, keyed by scenario.

    The shards are tracked LFS artifacts, so this layer exists in a clean checkout. They
    are the committed form of the exploded loose tree, which means comparing against them
    checks the bytes that actually ship.
    """
    import yaml

    out = {}
    for path in sorted((root / directory / fam).glob("*.yaml")):
        doc = yaml.safe_load(path.read_bytes()) or {}
        for cid, case in (doc.get("cases") or {}).items():
            scenario = case.get("scenario") or _taxonomy_scenarios(fam).get(cid)
            if scenario is None:
                continue
            out.setdefault(scenario, {}).update(
                {want: case[have] for want, have in field_map.items() if have in case}
            )
    return out


def _materialized_unified_root(tmp_path: Path) -> Path:
    root = tmp_path / "unified"
    unified_history.materialize_store(UTILS.parent / "fixtures-unified-v2", root)
    return root


def _taxonomy_scenarios(fam: str):
    """Taxonomy id (`UNIFIED.34-7`) -> scenario slug, from packaged input layers.

    The golden shard keys by taxonomy id and carries no scenario field, so the mapping
    comes from the one shard holding both. A key JOIN, not a value normalization.
    """
    cached = _taxonomy_scenarios._cache.get(fam)
    if cached is not None:
        return cached
    out = {}
    store = unified_history.load_store(UTILS.parent / "fixtures-unified-v2")
    for case in store.families[fam].cases.values():
        if case["scenario"] and case["display_id"]:
            out[case["display_id"]] = case["scenario"]
    _taxonomy_scenarios._cache[fam] = out
    return out


_taxonomy_scenarios._cache = {}


def test_every_case_triple_is_identical_at_every_layer(tmp_path):
    """One corpus, every tracked representation, zero drift, across all three fields.

    Compares `(input, init, golden)` for every case of every family across the generator's
    constructed cases, the emitted-and-reloaded golden spec, and the packaged
    canonical family YAML. Only key NAMES are normalized — `assembled`
    -> `golden`, taxonomy id -> scenario slug. No byte, missing field, value, ordering or
    type is normalized away; a missing field fails by name rather than being skipped.

    Every layer read here is tracked, so this runs identically in a clean checkout. The
    previous version read `conformance/unified/`, which `.gitignore:32` excludes entirely:
    it passed locally on leftover generated state and died with `FileNotFoundError` in
    0.16s on a fresh checkout, so the corpus it was written to protect shipped unguarded.

    Why this matters at all: the emitter once ate the leading space of `34-7`, the spec
    and feed carried the fix, and the loose and packaged inputs kept the pre-fix bytes
    because they had been exploded first. Every gate was green and the shipped corpus was
    wrong.
    """
    checked = 0
    materialized = _materialized_unified_root(tmp_path)
    for fam in FAMILIES:
        spec = _emitted_spec(fam)
        packed_in = _packed_layer(materialized, "inputs", fam, {"input": "input", "init": "init"})
        packed_gold = _packed_layer(materialized, "golden", fam, {"golden": "assembled"})

        for cid, case in build_cases(fam).items():
            scenario = _scenario_of(cid, fam)
            want = {k: case[k] for k in ("input", "init", "golden")}
            layers = {
                "emitted golden spec": {k: spec[cid][k] for k in want if k in spec[cid]},
                "packaged shard": {
                    **packed_in.get(scenario, {}),
                    **packed_gold.get(scenario, {}),
                },
            }
            for layer, got in layers.items():
                for field, expected in want.items():
                    assert field in got, f"{cid}: {field} missing from the {layer} layer"
                    assert got[field] == expected, (
                        f"{cid}: {field} differs at the {layer} layer\n"
                        f"  authored: {expected!r}\n"
                        f"  {layer}: {got[field]!r}"
                    )
            checked += 1
    assert checked == sum(len(build_cases(f)) for f in FAMILIES), (
        "the gate must cover every generated case"
    )


def _assert_retained_capture_coverage(captured, expected):
    assert captured, "no retained Dynamo Unified captures in history"
    assert expected <= captured, f"missing retained captures {sorted(expected - captured)}"


def test_retained_unified_captures_cover_the_current_corpus():
    store = unified_history.load_store(UTILS.parent / "fixtures-unified-v2")
    captured = set()
    for (family, implementation), history in store.histories.items():
        if implementation != "dynamo_v2":
            continue
        capture_id = history.ordered_capture_ids()[-1]
        for case_id in history.resolve(capture_id):
            case = history.family.cases[case_id]
            case_key = case["display_id"] or case["historical_ids"][0]
            captured.add((family, historical_unified_case_key(family, case_key)))
    expected = {
        (family, numbered_id(_scenario_of(case_id, family)))
        for family in FAMILIES for case_id in build_cases(family)
    }
    _assert_retained_capture_coverage(captured, expected)


def test_retained_capture_coverage_rejects_one_missing_case():
    expected = {("qwen3", "UNIFIED.1-1"), ("qwen3", "UNIFIED.32-5")}
    with pytest.raises(AssertionError, match="missing retained captures"):
        _assert_retained_capture_coverage(expected - {("qwen3", "UNIFIED.32-5")}, expected)


def _family_value(scenario, family):
    reason_open, reason_close, _, _ = control_tokens(family)
    if scenario == "arg_marker_in_string":
        close = {
            "deepseek_v4": "</｜DSML｜invoke>",
            "deepseek_v41": "</｜DSML｜ invoke>",
            "gemma4": "}<tool_call|>",
            "kimi_k2": "<|tool_call_end|>",
            "kimi_k3": "<|close|>call<|sep|>",
            "muse_glimmer": "</atem:function_calls>",
            "qwen3": "</tool_call>",
        }[family]
        return f"git log {close} --oneline"
    if scenario in {"reason_markup_in_arg", "reason_markup_in_arg_with_text"}:
        return ("to=self<|message|>reconsider" if family == "muse_glimmer"
                else f"{reason_open}reconsider{reason_close}")
    if scenario == "guided_json_marker_inside_argument":
        return reason_close
    quoted = {
        "guided_json_quoted_bare_header_in_answer": "self",
        "guided_json_quoted_bare_tool_header_in_answer": "get_weather",
        "guided_json_quoted_bare_header_after_payload": "self",
    }
    if scenario in quoted:
        recipient = quoted[scenario]
        return (f"I mean to={recipient}literal" if family == "muse_glimmer"
                else f"I mean {reason_open}{recipient} literal{reason_close}")
    if scenario == "prefilled_response_reasoning_markers_literal":
        return ("to=selfliteral" if family == "muse_glimmer"
                else f"{reason_open}literal{reason_close}") + " then a call"
    return None


def _logical_events(scenario, family, events):
    """Replace only the authored family-marker value, never event kinds or keys."""
    family_value = _family_value(scenario, family)
    out = json.loads(json.dumps(events))
    if family_value is not None:
        matches = 0
        for event in out:
            if event.get("text") == family_value:
                event["text"] = "FAMILY_MARKER_VALUE"
                matches += 1
            for key, value in event.get("arguments", {}).items():
                if value == family_value:
                    event["arguments"][key] = "FAMILY_MARKER_VALUE"
                    matches += 1
        assert matches == 1, (scenario, family, "wrong marker role or payload", events)
    return out


def _json_values(raw):
    decoder = json.JSONDecoder()
    values = []
    for at, char in enumerate(raw):
        if char not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(raw[at:])
        except json.JSONDecodeError:
            continue
        values.append(value)
    return values


def _native_input_calls(family, raw):
    """Read authored complete argument fields, not runtime recovery decisions.

    This fixture-only projection ignores invoke EOF policy: a syntactically present
    body is not evidence that the parser must dispatch it. The caller keeps DSML's
    empty-output EOF contract separate.
    """
    headers = {
        "deepseek_v4": r'<｜DSML｜invoke name="([^"]+)">',
        "deepseek_v41": r'<｜DSML｜ invoke name="([^"]+)">',
        "qwen3": r'<function=([^>]+)>',
        "muse_glimmer": r'<atem:invoke name="([^"]+)">',
        "gemma4": r'call:([\w.-]+)\{',
        "kimi_k2": r'<\|tool_call_begin\|>(?:functions\.)?([\w.-]+):\d+<\|tool_call_argument_begin\|>',
        "kimi_k3": r'<\|open\|>\s*call tool="([^"]+)" index="\d+"\s*<\|sep\|>',
    }
    found = list(re.finditer(headers[family], raw))
    calls = []
    for index, match in enumerate(found):
        body = raw[match.end():found[index + 1].start() if index + 1 < len(found) else len(raw)]
        arguments = {}
        if family in {"deepseek_v4", "deepseek_v41"}:
            gap = " " if family == "deepseek_v41" else ""
            pattern = rf'<｜DSML｜{gap}parameter name="([^"]+)" string="(true|false)">(.*?)</｜DSML｜{gap}parameter>'
            for key, is_string, value in re.findall(pattern, body, re.S):
                arguments[key] = value if is_string == "true" else json.loads(value)
        elif family == "qwen3":
            arguments = {key: value.strip() for key, value in re.findall(r'<parameter=([^>]+)>(.*?)</parameter>', body, re.S)}
        elif family == "muse_glimmer":
            for key, value in re.findall(r'<atem:parameter name="([^"]+)">(.*?)</atem:parameter>', body, re.S):
                try:
                    arguments[key] = json.loads(value)
                except json.JSONDecodeError:
                    arguments[key] = value
        elif family == "gemma4":
            arguments = {key: value for key, value in re.findall(r'(\w+):<\|"\|>(.*?)<\|"\|>', body, re.S)}
        elif family == "kimi_k2":
            arguments, _ = json.JSONDecoder().raw_decode(body)
        else:
            pattern = r'<\|open\|>\s*argument key="([^"]+)" type="([^"]+)"\s*<\|sep\|>(.*?)<\|close\|>\s*argument\s*<\|sep\|>'
            for key, kind, value in re.findall(pattern, body, re.S):
                arguments[key] = value if kind == "string" else json.loads(value)
            if not arguments and re.match(r'<\|open\|>\s*json ', body):
                values = _json_values(body)
                if values:
                    arguments = values[0]
        calls.append({"kind": "tool_call", "name": match[1], "arguments": arguments})
    return calls


@pytest.mark.parametrize("header,name", [
    ("functions.:17", "functions."), ("functions.functions.:17", "functions."),
    ("f:0", "f"), ("functions.f:0", "f"), ("functions..:0", "."),
])
def test_kimi_k2_fixture_projection_preserves_optional_prefix_names(header, name):
    raw = f"<|tool_call_begin|>{header}<|tool_call_argument_begin|>{{}}<|tool_call_end|>"
    assert _native_input_calls("kimi_k2", raw) == [{"kind": "tool_call", "name": name, "arguments": {}}]


def test_kimi_k2_fixture_projection_rejects_an_empty_name():
    raw = "<|tool_call_begin|>:17<|tool_call_argument_begin|>{}<|tool_call_end|>"
    assert _native_input_calls("kimi_k2", raw) == []


def _assert_input_carries_events(family, scenario, case):
    raw = case["input"]
    tools = [event for event in case["golden"] if event["kind"] == "tool_call"]
    if tools:
        if case["init"]["tool_output_mode"] == "Native":
            candidates = _native_input_calls(family, raw)
        else:
            candidates = []
            for value in _json_values(raw):
                if case["init"]["named_tool"] is not None:
                    candidates.append({"kind": "tool_call", "name": case["init"]["named_tool"], "arguments": value})
                else:
                    for call in value if isinstance(value, list) else [value]:
                        if isinstance(call, dict) and "name" in call and "arguments" in call:
                            candidates.append({"kind": "tool_call", **call})
        cursor = 0
        for event in tools:
            assert event in candidates[cursor:], (family, scenario, "input call differs from golden", event, candidates)
            cursor = candidates.index(event, cursor) + 1
    for event in case["golden"]:
        if event["kind"] in {"reasoning", "text"}:
            text = event["text"]
            if text in raw:
                continue
            controls_removed = raw
            for marker in control_tokens(family):
                controls_removed = controls_removed.replace(marker, "")
            if text in controls_removed:
                continue
            # Coalescing can remove intervening control/native spans, but it may
            # not invent or reorder the ordinary words on either side of them.
            plain = re.sub(r"<[^<>]*>", "", raw)
            cursor = 0
            for word in re.findall(r"\S+", text):
                at = plain.find(word, cursor)
                assert at >= 0, (family, scenario, "input lost output prose", word)
                cursor = at + len(word)
            continue
        assert event["kind"] == "tool_call"
        name, arguments = event["name"], event["arguments"]
        assert name in raw or case["init"]["named_tool"] == name, (family, scenario, "input tool name", name)
        assert isinstance(arguments, dict), (family, scenario, "argument object")
        for key, value in arguments.items():
            assert key in raw, (family, scenario, "input argument key", key)
            if isinstance(value, str):
                spellings = (value, json.dumps(value, ensure_ascii=False)[1:-1], json.dumps(value)[1:-1])
                assert any(spelling in raw for spelling in spellings), (family, scenario, "input argument value", key, value)
    reasons = [event for event in case["golden"] if event["kind"] == "reasoning"]
    if reasons and case["init"]["starting_state"] == "None":
        assert control_tokens(family)[0] in raw, (family, scenario, "missing reasoning opener")
    if reasons and any(event["kind"] != "reasoning" for event in case["golden"]):
        unclosed = {
            "truncated_tool_eof", "prefilled_reasoning_truncated",
            "guided_json_unterminated_reasoning_then_wrapped_payload",
            "guided_json_bare_tool_header_recovers_inside_a_thought",
            "deepseek_v4_guided_reasoning_opener_inside_native_body",
            "kimi_k3_elided_think_close_to_response",
        }
        if scenario not in unclosed:
            assert control_tokens(family)[1] in raw, (family, scenario, "missing reasoning closer")


def _assert_cross_family_contract(corpus):
    by_scenario = defaultdict(dict)
    clean = {row[0]: row[3] for row in CLEAN}
    prefills = {name for name, segments in clean.items() if segments[0][0] == "reason"} | {"reason_unterminated"}
    for family in FAMILIES:
        for cid, case in corpus[family].items():
            scenario = _scenario_of(cid, family)
            _assert_input_carries_events(family, scenario, case)
            if scenario == "guided_json_quoted_bare_tool_header_in_answer":
                # A tool name inside a thought is prose, not a recipient header.
                assert family == "muse_glimmer" and "to=get_weather<|message|>" in case["input"], (
                    family, scenario, "missing tool-recipient boundary",
                )
            init = dict(case["init"])
            if family == "deepseek_v41" and scenario in prefills:
                assert init == {"starting_state": "Reasoning", "tool_output_mode": "Native", "named_tool": None}
                assert not case["input"].startswith("<think>"), (family, scenario, "prefilled opener must be absent")
                init["starting_state"] = "None"
            events = _logical_events(scenario, family, case["golden"])
            if scenario == "tool_no_close":
                # DSML's public EOF contract drops an invoke without its closer.
                # Assert that exception independently; do not normalize [] to a call.
                if family in {"deepseek_v4", "deepseek_v41"}:
                    assert events == [], (family, scenario, "DSML requires invoke close")
                    assert _native_input_calls(family, case["input"]) == [
                        {"kind": "tool_call", "name": "get_weather", "arguments": {"city": "Paris"}},
                    ], (family, scenario, "EOF body differs")
                    gap = " " if family == "deepseek_v41" else ""
                    assert f'</｜DSML｜{gap}parameter>' in case["input"]
                    assert f'</｜DSML｜{gap}invoke>' not in case["input"]
                else:
                    assert events == [{"kind": "tool_call", "name": "get_weather", "arguments": {"city": "Paris"}}]
                events = None
            suppressed = []
            if init["tool_output_mode"] == "GuidedJson":
                # Complete native argument bodies are input even when guided mode
                # suppresses them. Goldens alone cannot detect a changed stress value.
                suppressed = [call for call in _native_input_calls(family, case["input"]) if call["arguments"]]
            by_scenario[scenario][family] = (events, init, case["finish_reason"], case["policy"], suppressed)
    assert set(by_scenario) == set(UNIFIED_TAX)
    for scenario, families in by_scenario.items():
        assert set(families) == set(G.scenario_families(scenario)), (scenario, "applicability", set(families))
        contracts = list(families.items())
        first_family, reference = contracts[0]
        for family, contract in contracts[1:]:
            assert contract == reference, (scenario, first_family, family, "cross-family contract", reference, contract)


def test_all_scenarios_preserve_cross_family_input_and_output_contracts():
    _assert_cross_family_contract({family: build_cases(family) for family in FAMILIES})


@pytest.mark.parametrize("family", [family for family in FAMILIES if family != "muse_glimmer"])
def test_cross_family_contract_rejects_prose_only_recipient_case(family):
    corpus = {family: build_cases(family) for family in FAMILIES}
    _assert_cross_family_contract(corpus)
    source = corpus[family][f"UNIFIED.guided_json_quoted_bare_header_in_answer.{family}"]
    case = json.loads(json.dumps(source).replace("self literal", "get_weather literal"))
    corpus[family][f"UNIFIED.guided_json_quoted_bare_tool_header_in_answer.{family}"] = case
    with pytest.raises(AssertionError, match="missing tool-recipient boundary"):
        _assert_cross_family_contract(corpus)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("mutation", [
    "tool_name", "argument_key", "reasoning_prose", "input_tool_name",
    "input_argument_key", "input_reasoning_prose", "marker_role", "missing_event",
    "named_tool", "starting_state", "output_mode", "input_value",
    "guided_array_value", "argument_marker_role", "input_call_order",
])
def test_cross_family_contract_rejects_single_family_mutations(family, mutation):
    corpus = {family: build_cases(family) for family in FAMILIES}
    _assert_cross_family_contract(corpus)
    case = corpus[family][f"UNIFIED.reason_then_tool.{family}"]
    if mutation in {"tool_name", "input_tool_name"}:
        case["input"] = case["input"].replace("get_weather", "run")
        if mutation == "tool_name":
            case["golden"][1]["name"] = "run"
    elif mutation in {"argument_key", "input_argument_key"}:
        case["input"] = case["input"].replace("city", "cmd")
        if mutation == "argument_key":
            case["golden"][1]["arguments"] = {"cmd": "Paris"}
    elif mutation in {"reasoning_prose", "input_reasoning_prose"}:
        case["input"] = case["input"].replace("Check weather.", "Other prose.")
        if mutation == "reasoning_prose":
            case["golden"][0]["text"] = "Other prose."
    elif mutation == "marker_role":
        reason_open, _, tool_open, _ = control_tokens(family)
        if family == "deepseek_v41":
            case["input"] = case["input"].replace("</think>", "</｜DSML｜ calls>")
        else:
            case["input"] = case["input"].replace(reason_open, tool_open, 1)
    elif mutation == "missing_event":
        case["golden"].pop(0)
    elif mutation == "named_tool":
        case["init"]["named_tool"] = "get_weather"
    elif mutation == "starting_state":
        case["init"]["starting_state"] = "Response"
    elif mutation == "output_mode":
        case["init"]["tool_output_mode"] = "GuidedJson"
    elif mutation == "input_value":
        case["input"] = case["input"].replace("Paris", "Rome")
    elif mutation == "guided_array_value":
        case = corpus[family][f"UNIFIED.guided_json_array_argument.{family}"]
        calls = json.loads(case["input"])
        calls[0]["arguments"]["values"][-1] += 1
        case["input"] = json.dumps(calls)
    elif mutation == "argument_marker_role":
        case = corpus[family][f"UNIFIED.arg_marker_in_string.{family}"]
        value = case["golden"][0]["arguments"]["cmd"]
        wrong = f"git log {control_tokens(family)[1]} --oneline"
        case["input"] = case["input"].replace(value, wrong)
        case["golden"][0]["arguments"]["cmd"] = wrong
    else:
        case = corpus[family][f"UNIFIED.two_calls.{family}"]
        segments = next(row[3] for row in CLEAN if row[0] == "two_calls")
        case["input"] = G.render_input(family, list(reversed(segments)))
        assert case["input"] != build_cases(family)[f"UNIFIED.two_calls.{family}"]["input"]
    with pytest.raises(AssertionError):
        _assert_cross_family_contract(corpus)


@pytest.mark.parametrize("family", ["deepseek_v4", "deepseek_v41"])
@pytest.mark.parametrize("mutation", ["dispatch", "tool", "key", "value", "closer"])
def test_dsml_eof_contract_rejects_invented_success_or_changed_body(family, mutation):
    corpus = {family: build_cases(family) for family in FAMILIES}
    _assert_cross_family_contract(corpus)
    case = corpus[family][f"UNIFIED.tool_no_close.{family}"]
    if mutation == "dispatch":
        case["golden"] = [{"kind": "tool_call", "name": "get_weather", "arguments": {"city": "Paris"}}]
    elif mutation == "closer":
        gap = " " if family == "deepseek_v41" else ""
        case["input"] += f'</｜DSML｜{gap}invoke>'
    else:
        old, new = {"tool": ("get_weather", "run"), "key": ("city", "cmd"), "value": ("Paris", "Rome")}[mutation]
        case["input"] = case["input"].replace(old, new)
    with pytest.raises(AssertionError):
        _assert_cross_family_contract(corpus)
