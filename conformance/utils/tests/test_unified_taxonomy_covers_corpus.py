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
    assert tax("guided_json_quoted_bare_header_in_answer") == (31, "25")
    assert tax("guided_json_quoted_bare_tool_header_in_answer") == (31, "26")
    assert tax("guided_json_quoted_bare_header_after_payload") == (31, "27")
    assert tax("guided_json_bare_tool_header_recovers_inside_a_thought") == (31, "28")
    assert tax("gemma4_guided_json_visible_call_prose_before_reasoning") == ("g", "1")
    assert tax("gemma4_guided_json_malformed_call_prefix_before_reasoning") == ("g", "2")
    assert case_label("guided_json_quoted_bare_header_in_answer") == "31-25"
    assert case_label("guided_json_quoted_bare_tool_header_in_answer") == "31-26"
    assert case_label("guided_json_quoted_bare_header_after_payload") == "31-27"
    assert case_label("guided_json_bare_tool_header_recovers_inside_a_thought") == "31-28"
    assert case_label("gemma4_guided_json_visible_call_prose_before_reasoning") == "g-1"
    assert case_label("gemma4_guided_json_malformed_call_prefix_before_reasoning") == "g-2"
    assert numbered_id("guided_json_quoted_bare_header_in_answer") == "UNIFIED.31-25"
    assert numbered_id("gemma4_guided_json_visible_call_prose_before_reasoning") == "UNIFIED.g-1"
    assert case_label("guided_json_invalid_call") == "31-1"
    assert numbered_id("guided_json_invalid_call") == "UNIFIED.31-1"
    guided = [sub for group, sub in UNIFIED_TAX.values() if group == 31]
    assert all(sub.isdecimal() for sub in guided)
    assert sorted(map(int, guided)) == list(range(1, 29))

    ordered = sorted(
        (
            "guided_json_quoted_bare_header_in_answer",
            "guided_json_quoted_bare_tool_header_in_answer",
            "guided_json_quoted_bare_header_after_payload",
            "guided_json_bare_tool_header_recovers_inside_a_thought",
        ),
        key=taxonomy_sort_key,
    )
    assert [tax(scenario)[1] for scenario in ordered] == ["25", "26", "27", "28"]


def test_kimi_k3_cases_use_numeric_suffixes() -> None:
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
    assert [case_label(scenario) for scenario in k3] == [f"k3-{i}" for i in range(1, 9)]
    assert [numbered_id(scenario) for scenario in k3] == [
        f"UNIFIED.k3-{i}" for i in range(1, 9)
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
    """All four request-state boundaries have a family-specific input.

    Muse exercises dynamic recipient headers, while marker-pair families exercise their
    own reasoning/tool boundaries under the same request initialization. A missing
    family here would be a coverage gap, not an unsupported grammar.
    """
    scoped = {
        "guided_json_quoted_bare_header_in_answer",
        "guided_json_quoted_bare_tool_header_in_answer",
        "guided_json_quoted_bare_header_after_payload",
        "guided_json_bare_tool_header_recovers_inside_a_thought",
    }
    for fam in FAMILIES:
        names = {k.split(".", 2)[1] for k in build_cases(fam)}
        present = scoped & names
        assert present == scoped, f"{fam} is missing {scoped - present}"


def test_scenario_families_matches_declared_scope():
    scoped = {
        "guided_json_quoted_bare_header_in_answer": {"muse_glimmer"},
        "gemma4_guided_json_visible_call_prose_before_reasoning": {"gemma4"},
        "gemma4_guided_json_malformed_call_prefix_before_reasoning": {"gemma4"},
    }
    for scenario, families in scoped.items():
        expected = families if scenario.startswith("gemma4_") else set(FAMILIES)
        assert G.scenario_families(scenario) == expected
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
    reader cannot tell content-space from indent-space. `31-28` is authored with a
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
    shared = min(per_family.values())
    for fam in FAMILIES:
        family_specific = 2 if fam == "gemma4" else 8 if fam == "kimi_k3" else 0
        expected = shared + family_specific
        assert per_family[fam] == expected, f"{fam} diverged from the expected case count"
    assert sum(per_family.values()) == sum(len(build_cases(f)) for f in FAMILIES)


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


def _packed_layer(shard: str, fam: str, field_map):
    """Case records from a PACKAGED shard, keyed by scenario.

    The shards are tracked LFS artifacts, so this layer exists in a clean checkout. They
    are the committed form of the exploded loose tree, which means comparing against them
    checks the bytes that actually ship.
    """
    import tarfile

    import yaml

    out = {}
    path = UTILS.parent / "fixtures" / "unified" / shard
    with tarfile.open(path) as tar:
        for member in tar.getmembers():
            if not member.name.endswith(".yaml") or f"/{fam}/" not in member.name:
                continue
            doc = yaml.safe_load(tar.extractfile(member).read()) or {}
            for cid, case in (doc.get("cases") or {}).items():
                scenario = case.get("scenario") or _taxonomy_scenarios(fam).get(cid)
                if scenario is None:
                    continue
                out.setdefault(scenario, {}).update(
                    {want: case[have] for want, have in field_map.items() if have in case}
                )
    return out


def _taxonomy_scenarios(fam: str):
    """Taxonomy id (`UNIFIED.31-28`) -> scenario slug, from packaged input layers.

    The golden shard keys by taxonomy id and carries no scenario field, so the mapping
    comes from the one shard holding both. A key JOIN, not a value normalization.
    """
    import tarfile

    import yaml

    cached = _taxonomy_scenarios._cache.get(fam)
    if cached is not None:
        return cached
    out = {}
    fixtures = UTILS.parent / "fixtures" / "unified"
    for path in sorted(fixtures.glob("inputs*.tar.gz")):
        with tarfile.open(path) as tar:
            for member in tar.getmembers():
                if not member.name.endswith(".yaml") or f"/{fam}/" not in member.name:
                    continue
                doc = yaml.safe_load(tar.extractfile(member).read()) or {}
                for cid, case in (doc.get("cases") or {}).items():
                    if case.get("scenario"):
                        out[cid] = case["scenario"]
    _taxonomy_scenarios._cache[fam] = out
    return out


_taxonomy_scenarios._cache = {}


def test_every_case_triple_is_identical_at_every_layer():
    """One corpus, every tracked representation, zero drift, across all three fields.

    Compares `(input, init, golden)` for every case of every family across the generator's
    constructed cases, the emitted-and-reloaded golden spec, and the packaged
    `inputs.tar.gz` / `golden.tar.gz` shards. Only key NAMES are normalized — `assembled`
    -> `golden`, taxonomy id -> scenario slug. No byte, missing field, value, ordering or
    type is normalized away; a missing field fails by name rather than being skipped.

    Every layer read here is tracked, so this runs identically in a clean checkout. The
    previous version read `conformance/unified/`, which `.gitignore:32` excludes entirely:
    it passed locally on leftover generated state and died with `FileNotFoundError` in
    0.16s on a fresh checkout, so the corpus it was written to protect shipped unguarded.

    Why this matters at all: the emitter once ate the leading space of `31-28`, the spec
    and feed carried the fix, and the loose and packaged inputs kept the pre-fix bytes
    because they had been exploded first. Every gate was green and the shipped corpus was
    wrong.
    """
    checked = 0
    for fam in FAMILIES:
        spec = _emitted_spec(fam)
        packed_in = {}
        packed_gold = {}
        for path in sorted((UTILS.parent / "fixtures" / "unified").glob("inputs*.tar.gz")):
            packed_in.update(_packed_layer(path.name, fam, {"input": "input", "init": "init"}))
        for path in sorted((UTILS.parent / "fixtures" / "unified").glob("golden*.tar.gz")):
            packed_gold.update(_packed_layer(path.name, fam, {"golden": "assembled"}))

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
