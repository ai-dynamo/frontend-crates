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

import ast
from collections import defaultdict
from dataclasses import replace
from decimal import Decimal
from copy import deepcopy
import json
import hashlib
import re
import sys
from pathlib import Path

UTILS = Path(__file__).resolve().parents[1]
SRC = UTILS / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from conformance.utils.tests import GPT_OSS_STANDALONE_HEADER_SCENARIOS
from conformance.utils.tests.schema_oracle import matches_schema
from conformance.utils.tests.fixture_test_support import assert_unique_behaviour, semantic_key

import pytest  # noqa: E402
import yaml  # noqa: E402

import gen_unified_golden as G  # noqa: E402
from schema_cases import CONFORMANCE_CASES, XML_COMPATIBILITY_CASES, schema_arguments  # noqa: E402
import unified_history  # noqa: E402
import unified_archives
import ci_coverage
import capture_stimulus  # noqa: E402
import unified_taxonomy as taxonomy  # noqa: E402
from capture_stimulus import _without_synthetic_finish_step  # noqa: E402
from dynamo_version import dynamo_v2_label  # noqa: E402
from fixture_disposition import historical_unified_case_key  # noqa: E402
from gen_unified_golden import (  # noqa: E402
    PARALLEL_BOUNDARY_CASES,
    UNIFIED_PARALLEL_BOUNDARY_FAMILIES,
    CLEAN,
    OnlyFamilies,
    EDGE,
    FAMILIES,
    build_cases,
    control_tokens,
    invoke_header_prefix,
)
from numeric_cases import NUMERIC_VARIANTS, applicable as numeric_applicable, canonical_events, numeric_expected, unified_numeric_expected
from unified_taxonomy import (  # noqa: E402
    historical_case_label,
    UNIFIED_GROUP_LABEL,
    UNIFIED_TAX,
    case_label,
    numbered_id,
    tax,
    taxonomy_sort_key,
)

TAXONOMY_FILE = "conformance/utils/src/unified_taxonomy.py"
NATIVE_FAMILIES = tuple(family for family in FAMILIES if family in G.UNIFIED_FAMILIES and G._MANIFEST[family].get("scenario_profile") != "native")
V1_SPLIT_SCENARIOS = frozenset(G.V1_SPLIT_SCENARIOS)
NATIVE_UNIFIED_TAX = {scenario: entry for scenario, entry in UNIFIED_TAX.items() if scenario not in V1_SPLIT_SCENARIOS and scenario not in taxonomy.RETIRED_SCENARIOS}


def test_descriptive_ids_and_all_published_aliases():
    labels = [case_label(s) for s in UNIFIED_TAX]
    assert len(set(labels)) == len(labels)
    for scenario in UNIFIED_TAX:
        label = case_label(scenario)
        assert re.fullmatch(r"[\w]+-[\w]+", label), label
        legacy = taxonomy.legacy_case_label(scenario)
        if len(taxonomy._LABEL_OWNERS[legacy]) > 1:
            with pytest.raises(ValueError, match="conflicting Unified alias ownership"):
                historical_case_label(legacy)
        else:
            assert historical_case_label(legacy) == label
        assert historical_case_label(legacy, scenario=scenario) == label
        assert historical_case_label(label) == label
        assert historical_case_label(label, scenario=scenario) == label
        for prefix in taxonomy.GROUP_DESCRIPTION_PREFIXES.get(tax(scenario)[0], ()):
            assert not label.split("-", 1)[1].startswith(prefix)
    for alias, scenario in taxonomy.LEGACY_CASE_LABELS.items():
        assert historical_case_label(alias) == case_label(scenario)
    for probe in CONFORMANCE_CASES:
        assert case_label("schema_" + probe["name"]) == "7-" + probe["name"]
    assert historical_case_label("31-29", family="gemma4") == case_label(
        "gemma4_guided_json_visible_call_prose_before_reasoning")
    assert historical_case_label("31.a") == case_label("guided_json_invalid_call")


def test_alias_conflicts_and_wrong_scenario_ownership_are_rejected(monkeypatch):
    with pytest.raises(ValueError, match="conflicting Unified scenario ownership"):
        historical_case_label("7-1", scenario="arg_marker_in_string")
    owners = taxonomy._label_owners(
        {"arg_marker_in_string": ("7-arg_unicode",), "arg_unicode": ("7-arg_unicode",)},
        {"7-1": "arg_unicode"},
    )
    assert owners["7-arg_unicode"] == {"arg_marker_in_string", "arg_unicode"}
    assert owners["7-1"] == {"arg_unicode"}
    monkeypatch.setattr(taxonomy, "_LABEL_OWNERS", owners)
    with pytest.raises(ValueError, match="conflicting Unified alias ownership"):
        historical_case_label("7-arg_unicode")
    assert taxonomy.scenario_for_label("7-arg_unicode", scenario="arg_marker_in_string") == "arg_marker_in_string"


def corpus_scenarios() -> list[str]:
    """Include dynamically authored request variants and retained v1 scenarios."""
    variants = {scenario + "_pr424" for scenario, active in taxonomy.GPT_OSS_PR424_REQUEST_VARIANTS.items() if active}
    return sorted(({spec[0] for spec in (*CLEAN, *EDGE)} | variants | V1_SPLIT_SCENARIOS) - taxonomy.RETIRED_SCENARIOS)


def test_every_corpus_scenario_has_a_taxonomy_entry() -> None:
    unmapped = sorted(s for s in corpus_scenarios() if s not in UNIFIED_TAX)
    assert not unmapped, (
        f"{len(unmapped)} corpus scenario(s) have no UNIFIED_TAX entry and would render "
        f"as UNIFIED.9.<slug> instead of their real group: {unmapped}. "
        f"Add them to UNIFIED_TAX in {TAXONOMY_FILE}."
    )


def test_every_active_taxonomy_scenario_has_a_generator_owner():
    for scenario in UNIFIED_TAX:
        owners = G.scenario_families(scenario)
        assert bool(owners) == (scenario not in taxonomy.RETIRED_SCENARIOS), scenario
    with pytest.raises(KeyError, match="unknown unified scenario"):
        G.scenario_families("not_an_authored_scenario")
    for scenario in V1_SPLIT_SCENARIOS:
        assert not (G.scenario_families(scenario) & G.MUSE_FAMILIES)


def test_taxonomy_has_no_entry_without_a_corpus_case() -> None:
    """The other direction: a stale entry means a case was renamed or deleted and the map
    still claims a number for it, so the number is reserved against nothing."""
    scenarios = set(corpus_scenarios()) | taxonomy.RETIRED_SCENARIOS
    stale = sorted(s for s in UNIFIED_TAX if s not in scenarios)
    assert not stale, (
        f"{len(stale)} UNIFIED_TAX entr(ies) name a scenario the corpus does not emit: "
        f"{stale}. Remove them from {TAXONOMY_FILE} or restore the case in "
        f"conformance/utils/src/gen_unified_golden.py."
    )


def test_invoke_header_prefix_is_inner_and_unterminated() -> None:
    for family in NATIVE_FAMILIES:
        prefix = invoke_header_prefix(family)
        if family in {"glm47", "gpt_oss"}:
            # GLM uses its outer marker as the invoke opener. Harmony routes a
            # native call in the message header, so neither has a body prefix.
            assert prefix == ""
            continue
        assert prefix
        assert prefix == prefix.lstrip()
        if family != "deepseek_v41":
            assert not prefix.startswith(control_tokens(family)[2])
        assert prefix.rsplit(">", 1)[-1]
    assert invoke_header_prefix("gpt_oss") == ""


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
    assert case_label("guided_json_quoted_bare_header_in_answer") == "35-guided_json_quoted_bare_header_in_answer"
    assert case_label("guided_json_quoted_bare_tool_header_in_answer") == "muse-guided_json_quoted_bare_tool_header_in_answer"
    assert case_label("guided_json_quoted_bare_header_after_payload") == "35-guided_json_quoted_bare_header_after_payload"
    assert case_label("guided_json_bare_tool_header_recovers_inside_a_thought") == "34-guided_json_bare_tool_header_recovers_inside_a_thought"
    assert case_label("gemma4_guided_json_visible_call_prose_before_reasoning") == "gemma-guided_json_visible_call_prose_before_reasoning"
    assert case_label("gemma4_guided_json_malformed_call_prefix_before_reasoning") == "gemma-guided_json_malformed_call_prefix_before_reasoning"
    assert numbered_id("guided_json_quoted_bare_header_in_answer") == "UNIFIED.35-guided_json_quoted_bare_header_in_answer"
    assert numbered_id("gemma4_guided_json_visible_call_prose_before_reasoning") == "UNIFIED.gemma-guided_json_visible_call_prose_before_reasoning"
    assert case_label("guided_json_invalid_call") == "31-guided_json_invalid_call"
    assert numbered_id("guided_json_invalid_call") == "UNIFIED.31-guided_json_invalid_call"
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
    assert [case_label(scenario) for scenario in ordered] == ["34-guided_json_bare_tool_header_recovers_inside_a_thought", "35-guided_json_quoted_bare_header_in_answer", "35-guided_json_quoted_bare_header_after_payload", "muse-guided_json_quoted_bare_tool_header_in_answer"]


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
    assert [case_label(scenario) for scenario in k3] == ["kimi-" + scenario.removeprefix("kimi_k3_") for scenario in k3]
    assert [numbered_id(scenario) for scenario in k3] == [
        "UNIFIED.kimi-" + scenario.removeprefix("kimi_k3_") for scenario in k3
    ]


# --- End-to-end test cases: the SAME mapping is written in two places -----------
# Case descriptions in gen_unified_golden.py carry `End-to-end: <case> (e2e case-NNNN)` tags, and
# UNIFIED_CASES.md repeats them in its artifact-index table. Two copies of one fact drift
# — that is the defect this whole surface keeps hitting — so pin them to each other.

CASES_MD = UTILS / "lib" / "parsers" / "UNIFIED_CASES.md"
_E2E_TAG = re.compile(r"\be2e case-(\d{4})-")
_MD_ROW = re.compile(r"^\|\s*`([\w]+(?:\.[a-z]|-[\w]+))`\s*\|\s*`([^`]+)`\s*\|\s*`end-to-end case-(\d{4})-", re.M)


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
    "TOOLCALLING.streamv1": UTILS / "lib" / "parsers" / "TOOLCALLING_STREAMING_V1_CASES.md",
    "TOOLCALLING.batch": UTILS / "lib" / "parsers" / "TOOLCALLING_CASES.md",
    "REASONING.batch": UTILS / "lib" / "parsers" / "REASONING_CASES.md",
}
_QUALIFIED = re.compile(r"\b(?:TOOLCALLING|REASONING)\.(?:batch|streamv1)\.\d+(?:\.[a-z])?")
# a stage segment with no axis in front of it — the shape that named nothing
_BARE = re.compile(r"(?<![.\w])(?:batch|streamv1)\.\d+(?:\.[a-z])?")
_CITING = [UTILS / "lib" / "parsers" / "UNIFIED_CASES.md", SRC / "gen_unified_golden.py"]


def test_sibling_case_references_are_fully_qualified() -> None:
    offenders = {f.name: sorted(set(_BARE.findall(f.read_text(encoding="utf-8")))) for f in _CITING}
    offenders = {k: v for k, v in offenders.items() if v}
    assert not offenders, (
        f"unqualified case references (missing the axis prefix): {offenders}. "
        "Cite the full name, e.g. `TOOLCALLING.streamv1.2.a`, not `streamv1.2.a`."
    )


def test_sibling_case_references_exist() -> None:
    bodies = {k: p.read_text(encoding="utf-8") for k, p in _SIBLING_DOCS.items()}
    dangling: dict[str, list[str]] = {}
    for f in _CITING:
        bad = [
            ref
            for ref in sorted(set(_QUALIFIED.findall(f.read_text(encoding="utf-8"))))
            # group-level ids (`...streamv1.2`) have no entry of their own; a sub-case does
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
    for fam in G.GUIDED_FAMILIES:
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
    for family in FAMILIES:
        for case in build_cases(family).values():
            emitted |= set(case["init"])
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
        assert_unique_behaviour(fam, build_cases(fam))


def test_semantic_inventory_rejects_duplicates_and_reused_ids():
    case = {"input": "<call>x</call>", "init": {"starting_state": "None", "tool_output_mode": "Native"},
            "tools": [{"name": "run", "parameters": {"type": "object", "properties": {"x": {"type": "integer"}}}}],
            "golden": [{"kind": "tool_call", "name": "run", "arguments": {"x": 1}}]}
    with pytest.raises(AssertionError, match="identical behaviour"):
        assert_unique_behaviour("sample", {"first": case, "duplicate": deepcopy(case)})
    changed = deepcopy(case)
    changed["golden"][0]["arguments"]["x"] = "1"
    with pytest.raises(AssertionError, match="reused authored case ID"):
        assert_unique_behaviour("sample", [("same", case), ("same", changed)])
    assert semantic_key(case) != semantic_key(changed)
    changed = deepcopy(case)
    changed["tools"][0]["parameters"]["properties"]["x"]["type"] = "string"
    assert semantic_key(case) != semantic_key(changed)
    changed = deepcopy(case)
    changed["input"] = "<other>x</other>"
    assert semantic_key(case) != semantic_key(changed)
    binding = {"purpose": "Retain a source-bound regression", "input_source": "inputs.tar.gz:family/case", "capture_source": "capture.tar.gz:family/case", "version": "1.0.0"}
    inventory = assert_unique_behaviour("sample", {"first": case, "retained": deepcopy(case)}, retained_archives={"retained": binding})
    assert inventory["exact_duplicates"] == [["first", "retained"]]
    with pytest.raises(AssertionError, match="identical behaviour"):
        assert_unique_behaviour("sample", {"first": case, "duplicate": deepcopy(case), "retained": deepcopy(case)}, retained_archives={"retained": binding})
    retired = {"lifecycle": "retired", "request": None, "golden": None}
    assert assert_unique_behaviour("sample", {"old": retired})["retired_cases"] == ["old"]
    with pytest.raises(AssertionError, match="no active authored case"):
        assert_unique_behaviour("sample", {"old": retired}, retained_archives={"old": binding})
    with pytest.raises(AssertionError, match="requires source and version binding"):
        assert_unique_behaviour("sample", {"retained": case}, retained_archives={"retained": {"purpose": "overlap"}})


def test_deepseek_v41_follows_declared_scope_including_prefilled_cases() -> None:
    declared = {
        spec[0]
        for spec in (*CLEAN, *EDGE)
        if "deepseek_v41" in G.scenario_families(spec[0])
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
    assert numbered_id(scenario) == "UNIFIED.30-guided_json_gt_in_argument_bare_opener"
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
    assert numbered_id(scenario) == "UNIFIED.4-tool_markup_only_emits_nothing"
    assert case["input"] == "<｜DSML｜ calls></｜DSML｜ calls>"
    assert case["golden"] == []
    assert case["init"] == {
        "starting_state": "None", "tool_output_mode": "Native", "named_tool": None,
    }


def test_gpt_oss_response_applicability_is_not_derived_from_the_denylist() -> None:
    scenarios = {
        "native_unmatched_quote_then_call",
        "guided_json_orphan_tool_close_before_payload",
        "guided_response_quoted_control_braces_required",
        "guided_response_quoted_control_braces_named",
        "guided_json_native_parameter_object_before_payload",
        "guided_json_native_parameter_array_before_payload",
        "guided_json_native_parameter_string_before_payload",
        "guided_json_noncall_before_native_array_before_payload",
    }
    cases = build_cases("gpt_oss")
    for scenario in scenarios:
        case = cases[f"UNIFIED.{scenario}.gpt_oss"]
        assert any(event["kind"] == "tool_call" for event in case["golden"]), scenario
        assert scenario not in G.HARMONY_INAPPLICABLE_EDGE_CASES


def test_response_state_uses_only_control_marker_contracts() -> None:
    response_scenarios = {
        "prefilled_response_reasoning_markers_literal",
        "guided_json_quoted_bare_header_in_answer",
        "guided_json_quoted_bare_header_after_payload",
        "native_quoted_control_in_response",
        "native_single_quoted_word_control",
        "native_single_quote_contraction_response",
        "native_quoted_incomplete_header",
        "native_quoted_control_then_call",
        "native_unmatched_quote_then_call",
        "guided_response_quoted_control_braces_named",
        "guided_response_quoted_control_braces_required",
        "guided_json_native_parameter_object_before_payload",
        "guided_json_native_parameter_array_before_payload",
        "guided_json_native_parameter_string_before_payload",
        "guided_json_noncall_before_native_array_before_payload",
        "guided_response_native_wrapper_whitespace_before_prose",
    }
    for family in NATIVE_FAMILIES:
        cases = build_cases(family)
        response_cases = {
            case_id.split(".", 2)[1]: case
            for case_id, case in cases.items()
            if case["init"]["starting_state"] == "Response"
        }
        extra = set()
        if family in {"muse_glimmer", "muse_spark"}:
            extra.update({
                "guided_json_quoted_bare_tool_header_in_answer",
                "muse_quoted_reserved_eom_argument",
                "muse_quoted_reserved_eot_argument",
                "muse_quoted_reserved_start_argument",
                "prefilled_response_guided_closer_inside_invoke_quote",
                "guided_json_native_parameter_object_before_payload",
                "guided_json_native_parameter_array_before_payload",
            })
        if family == "deepseek_v4":
            extra.add("guided_response_rejected_header_quote_ownership")
        if family == "deepseek_v41":
            extra.add("prefilled_response_guided_pending_invoke_header")
        if family == "gemma4":
            extra.add("gemma4_guided_response_extra_body_opener")
        if G.unified_grammar_family(family) == "qwen3":
            extra.update({
                "guided_json_native_parameter_object_before_payload",
                "guided_json_native_parameter_array_before_payload",
            })
        expected = response_scenarios | extra
        if family == "gpt_oss":
            expected -= G.HARMONY_INAPPLICABLE_EDGE_CASES
            expected |= {
                scenario + "_pr424"
                for scenario, active in taxonomy.GPT_OSS_PR424_REQUEST_VARIANTS.items()
                if active and scenario in expected
            }
        assert set(response_cases) == expected
        native_scenarios = {
            "native_quoted_control_in_response",
            "native_single_quoted_word_control",
            "native_single_quote_contraction_response",
            "native_quoted_incomplete_header",
            "native_quoted_control_then_call",
            "native_unmatched_quote_then_call",
            "guided_response_quoted_control_braces_named",
            "guided_response_quoted_control_braces_required",
        }
        for scenario, case in response_cases.items():
            if scenario.startswith("muse_quoted_reserved_"):
                marker = {
                    "muse_quoted_reserved_eom_argument": "<|eom|>",
                    "muse_quoted_reserved_eot_argument": "<|eot|>",
                    "muse_quoted_reserved_start_argument": "<|start|>",
                }[scenario]
            elif family == "gpt_oss" and scenario == "prefilled_response_reasoning_markers_literal":
                marker = "<think>"
            elif scenario in native_scenarios:
                marker = G._NATIVE_QUOTED_CONTROL.get(G.unified_grammar_family(family), control_tokens(family)[2])
            elif scenario == "guided_response_rejected_header_quote_ownership":
                marker = '<｜DSML｜invoke'
            elif scenario == "gemma4_guided_response_extra_body_opener":
                marker = "call:get_weather{"
            elif scenario == "guided_response_native_wrapper_whitespace_before_prose":
                if family in {"muse_glimmer", "muse_spark"}:
                    marker = "<atem:invoke"
                elif family == "beam":
                    # Beam's tool header carries its recipient; the template's NAMEX
                    # placeholder never appears verbatim, so assert the routed lead.
                    marker = "<|start|>assistant to=f<|channel|>analysis <|constrain|>xml<|message|>"
                else:
                    marker = control_tokens(family)[2]
            elif family in {"muse_glimmer", "muse_spark"}:
                marker = "<|message|>"
            else:
                marker = control_tokens(family)[0]
                if family == "gpt_oss":
                    marker = marker.removeprefix("<|start|>assistant")
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
        missing = sorted(set(G.FAMILIES) - {"muse_glimmer", "muse_spark"})
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
    for fam in G.GUIDED_FAMILIES:
        names = {k.split(".", 2)[1] for k in build_cases(fam)}
        present = scoped & names
        assert present == scoped, f"{fam} is missing {scoped - present}"


def test_scenario_families_matches_declared_scope():
    scoped = {
        "guided_json_quoted_bare_header_in_answer": set(G.GUIDED_FAMILIES),
        "guided_json_quoted_bare_tool_header_in_answer": {"muse_glimmer", "muse_spark"},
        "gemma4_guided_json_visible_call_prose_before_reasoning": {"gemma4"},
        "gemma4_guided_json_malformed_call_prefix_before_reasoning": {"gemma4"},
        "arg_json_null_ref": set(G._TEMPLATE_FAMILIES),
        "arg_string_null_ref": set(G._TEMPLATE_FAMILIES),
        "arg_null_mixed_labels": set(G._TEMPLATE_FAMILIES),
        "glm_ref_object": set(FAMILIES),
        "glm_ref_encoded_targets": set(FAMILIES),
        "glm_ref_json_looking_strings": set(FAMILIES),
        "glm_ref_scalar_types": set(FAMILIES),
    }
    for scenario, families in scoped.items():
        assert G.scenario_families(scenario) == families
    assert G.scenario_families("arg_string_null") == set(FAMILIES)
    assert G.scenario_families("arg_json_null") == set(FAMILIES)
    assert G.scenario_families("deepseek_v41_mixed_control_text_in_string") == set(NATIVE_FAMILIES)
    assert G.scenario_families("tool_only") == set(FAMILIES)


def test_only_families_rejects_an_empty_or_unknown_scope():
    """A scope that names nothing, or names a family that does not exist, is a typo."""
    with pytest.raises(ValueError, match="declares nothing"):
        OnlyFamilies({})
    with pytest.raises(ValueError, match="do not exist"):
        OnlyFamilies({"no_such_family": ("x",)})


def test_schema_null_cases_keep_their_native_inputs_and_history():
    qwen = build_cases("qwen3")
    string = qwen["UNIFIED.arg_string_null.qwen3"]
    nullable = qwen["UNIFIED.arg_json_null.qwen3"]
    assert string["input"] == nullable["input"]
    assert string["tools"][0]["parameters"]["properties"]["city"] == {"type": "string"}
    assert nullable["tools"][0]["parameters"]["properties"]["city"] == {"type": ["string", "null"]}
    glm = build_cases("glm47")["UNIFIED.arg_string_null.glm47"]
    assert "<arg_value>null</arg_value>" in glm["input"]
    assert glm["tools"] == string["tools"]
    assert glm["golden"] == string["golden"]
    assert numbered_id("arg_string_null") == "UNIFIED.7-arg_string_null"
    assert numbered_id("arg_json_null") == "UNIFIED.7-arg_json_null"
    assert historical_unified_case_key("qwen3", "UNIFIED.qwen-1") == "UNIFIED.7-arg_json_null"


@pytest.mark.parametrize("family", NATIVE_FAMILIES)
@pytest.mark.parametrize("scenario,value", [("arg_json_null", None), ("arg_string_null", "null")])
def test_null_variants_preserve_the_declared_json_type(family, scenario, value):
    case = build_cases(family)[f"UNIFIED.{scenario}.{family}"]
    expected_type = "string" if value is not None else ["string", "null"]
    assert case["tools"][0]["parameters"]["properties"]["city"] == {"type": expected_type}
    assert case["golden"] == [{"kind": "tool_call", "name": "get_weather", "arguments": {"city": value}}]
    _assert_input_carries_events(family, scenario, case)
    assert "schema" in case["description"]


@pytest.mark.parametrize("scenario,value", [("arg_json_null_ref", None), ("arg_string_null_ref", "null")])
@pytest.mark.parametrize("family", NATIVE_FAMILIES)
def test_reference_goldens_use_the_tool_parameters_root(family, scenario: str, value: object) -> None:
    case = build_cases(family)[f"UNIFIED.{scenario}.{family}"]
    case["tools"] = json.loads(json.dumps(case["tools"]))
    parameters = case["tools"][0]["parameters"]
    assert parameters["properties"]["city"] == {"$ref": "#/$defs/City"}
    assert case["golden"][0]["arguments"] == {"city": value}
    assert matches_schema({"city": value}, parameters)
    _assert_input_carries_events(family, scenario, case)
    parameters["$defs"]["City"] = {"type": "integer"}
    assert not matches_schema({"city": value}, parameters)


@pytest.mark.parametrize("family", NATIVE_FAMILIES)
def test_mixed_null_fields_keep_independent_schema_types(family):
    case = build_cases(family)[f"UNIFIED.arg_null_mixed_labels.{family}"]
    arguments = case["golden"][0]["arguments"]
    assert arguments == {"label": None, "note": None, "literal": "null"}
    parameters = case["tools"][0]["parameters"]
    assert matches_schema(arguments, parameters)
    assert not matches_schema({**arguments, "literal": None}, parameters)
    _assert_input_carries_events(family, "arg_null_mixed_labels", case)


def test_qwen_reference_string_null_uses_root_schema_and_native_input() -> None:
    case = build_cases("qwen3")["UNIFIED.arg_string_null_ref.qwen3"]
    parameters = case["tools"][0]["parameters"]
    assert parameters["properties"]["city"] == {"$ref": "#/$defs/City"}
    assert parameters["$defs"]["City"] == {"type": "string"}
    assert case["golden"][0]["arguments"] == {"city": "null"}
    assert matches_schema({"city": "null"}, parameters)
    _assert_input_carries_events("qwen3", "arg_string_null_ref", case)


@pytest.mark.parametrize("scenario,label,arguments,references", [
    ("glm_ref_object", "7-glm_ref_object", {"payload": {"x": 1}},
     {"payload": "#/$defs/Payload"}),
    ("glm_ref_encoded_targets", "7-glm_ref_encoded_targets", {"space": 42, "utf8_plus": 42, "pointer": 42},
     {"space": "#/$defs/postal%20code", "utf8_plus": "#/$defs/caf%c3%a9+",
      "pointer": "#/$defs/a%7E1b%7E0c"}),
    ("glm_ref_json_looking_strings", "7-glm_ref_json_looking_strings",
     {"object_text": '{"x":1}', "array_text": '[1,2]',
      "quoted_text": '"hello"', "inline_text": '{"x":1}'},
     {key: "#/$defs/Text" for key in ("object_text", "array_text", "quoted_text")}),
    ("glm_ref_scalar_types", "7-glm_ref_scalar_types", {"count": 42, "ratio": 3.5, "flag": True, "narrowed": 42},
     {"count": "#/$defs/Integer", "ratio": "#/$defs/Number",
      "flag": "#/$defs/Boolean", "narrowed": "#/$defs/Scalar"}),
])
def test_glm_type_reference_goldens_keep_raw_refs_and_schema_valid_arguments(
    scenario: str, label: str, arguments: dict, references: dict[str, str],
) -> None:
    case = build_cases("glm47")[f"UNIFIED.{scenario}.glm47"]
    parameters = case["tools"][0]["parameters"]
    assert numbered_id(scenario) == "UNIFIED." + historical_case_label(label)
    for key, reference in references.items():
        assert parameters["properties"][key] == (
            {"$ref": reference, "type": "integer"} if key == "narrowed" else {"$ref": reference}
        )
    assert case["golden"] == [{"kind": "tool_call", "name": "capture_payload", "arguments": arguments}]
    assert matches_schema(arguments, parameters)
    _assert_input_carries_events("glm47", scenario, case)


@pytest.mark.parametrize("family", NATIVE_FAMILIES)
@pytest.mark.parametrize("scenario,key,value", [
    ("glm_ref_object", "payload", '{"x":1}'),
    ("glm_ref_encoded_targets", "pointer", "42"),
    ("glm_ref_json_looking_strings", "quoted_text", "hello"),
    ("glm_ref_scalar_types", "narrowed", "42"),
    ("unused_reference_graph_parameter_types", "count", "42"),
    ("nullable_reference_alias_literals", "const_text", None),
    ("local_schema_id_preserves_type", "count", "42"),
])
def test_reference_input_oracle_rejects_changed_golden_values(
    family: str, scenario: str, key: str, value: object,
) -> None:
    case = json.loads(json.dumps(build_cases(family)[f"UNIFIED.{scenario}.{family}"]))
    _assert_input_carries_events(family, scenario, case)
    case["golden"][0]["arguments"][key] = value
    with pytest.raises(AssertionError, match="input call differs from golden"):
        _assert_input_carries_events(family, scenario, case)


def test_reference_oracle_preserves_chained_targets_and_sibling_constraints() -> None:
    parameters = {"$defs": {"City/Type": {"type": ["string", "null"]},
                            "Alias": {"$ref": "#/$defs/City~1Type"}},
                  "properties": {"city": {"$ref": "#/$defs/Alias", "type": "string"}}}
    assert matches_schema({"city": "null"}, parameters)
    assert not matches_schema({"city": None}, parameters)


@pytest.mark.parametrize("reference", ["https://example.com/schema", "#/$defs/missing", "#/$defs/Loop"])
def test_reference_oracle_rejects_unresolved_or_cyclic_targets(reference: str) -> None:
    parameters = {"$defs": {"Loop": {"$ref": "#/$defs/Loop"}},
                  "properties": {"city": {"$ref": reference}}}
    with pytest.raises(AssertionError):
        matches_schema({"city": None}, parameters)


@pytest.mark.parametrize("family", NATIVE_FAMILIES)
@pytest.mark.parametrize("scenario", ["arg_json_null", "arg_string_null"])
def test_null_variant_contract_rejects_a_changed_schema(family, scenario):
    corpus = {name: build_cases(name) for name in NATIVE_FAMILIES}
    case = corpus[family][f"UNIFIED.{scenario}.{family}"]
    if not case["tools"]:
        pytest.skip("the preserved GPT-OSS request has no offered tool schema")
    case["tools"] = json.loads(json.dumps(case["tools"]))
    case["tools"][0]["parameters"]["properties"]["city"]["type"] = "number"
    with pytest.raises(AssertionError):
        _assert_cross_family_contract(corpus)


def test_deepseek_mixed_control_string_preserves_the_historical_id():
    family = "deepseek_v41"
    case = build_cases(family)["UNIFIED.deepseek_v41_mixed_control_text_in_string." + family]
    value = case["golden"][0]["arguments"]["x"]
    assert value == G._DS41_MIXED_STRING
    assert "<think>quoted</think>" in value
    assert '&amp; "x"' + "\\" + "\n" in value
    assert numbered_id("deepseek_v41_mixed_control_text_in_string") == "UNIFIED.7-deepseek_v41_mixed_control_text_in_string"


@pytest.mark.parametrize("family", NATIVE_FAMILIES)
def test_mixed_control_string_uses_native_markers_and_exact_string(family):
    scenarios = ["deepseek_v41_mixed_control_text_in_string"]
    if family == "gpt_oss":
        scenarios.append("deepseek_v41_mixed_control_text_in_string_pr424")
    cases = build_cases(family)
    for scenario in scenarios:
        case = cases[f"UNIFIED.{scenario}.{family}"]
        value = case["golden"][0]["arguments"]["x"]
        expected = (G._MIXED_CONTROL_STRINGS[G.unified_grammar_family(family)] if scenario.endswith("_pr424")
                    else _family_value(scenario, family))
        assert value == expected
        assert value.startswith(" ") and value.endswith("\n ")
        reason = G.r_reason(family, "quoted")
        if family == "gpt_oss" and not scenario.endswith("_pr424"):
            reason = reason.removeprefix("<|start|>assistant")
        assert reason in value
        assert '&amp; "x"' + "\\" in value
        assert case["input"] == G.default_response_prefix(family) + G.r_tool(family, "f", "x", value, 0)


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
    reader cannot tell content-space from indent-space. `34-guided_json_bare_tool_header_recovers_inside_a_thought` is authored with a
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
            assert loaded.get("tools") == case.get("tools"), f"{cid}: tools changed"
            assert loaded.get("input_chunks") == case.get("input_chunks"), f"{cid}: chunk schedule changed"


@pytest.mark.parametrize("family", [family for family in FAMILIES if G._MANIFEST[family].get("grammar")])
def test_inherited_family_preserves_published_chunk_schedules(family):
    path = UTILS.parent / "fixtures-unified-v2" / "families" / family / "inputs_and_golden.yaml"
    historical = yaml.safe_load(path.read_text())["cases"]
    cases = build_cases(family)
    for record in historical.values():
        if record["lifecycle"] != "active":
            continue
        request = _without_synthetic_finish_step(record["request"])
        case = cases[f"UNIFIED.{record['scenario']}.{family}"]
        assert case.get("input_chunks") == [chunk["delta_text"] for chunk in request["chunks"]], record["scenario"]


# --- counts live where they can be checked, not in registry prose ---------------

def test_unified_case_counts_match_the_generator(authored_unified_store):
    """Every active authored request has one generator owner; v1 has its own generator."""
    assert set(authored_unified_store.families) == set(NATIVE_FAMILIES)
    for family in NATIVE_FAMILIES:
        expected = {case["scenario"] for case in authored_unified_store.families[family].cases.values()
                    if case["lifecycle"] == "active" and case["scenario"] not in V1_SPLIT_SCENARIOS}
        actual = {_scenario_of(case_id, family) for case_id in build_cases(family)}
        assert actual == expected, family


def test_deferred_case_ids_are_not_in_the_active_taxonomy():
    deferred = {"1-2", "5-5", "6-2", "32-6", "50-1", "50-2"} | {
        f"31-{number}" for number in range(31, 41)
    }
    assert len(UNIFIED_TAX) == 239 + len(CONFORMANCE_CASES) + len(PARALLEL_BOUNDARY_CASES) + len(V1_SPLIT_SCENARIOS) + len(G.DOTTED_NAME_SCENARIOS) + len(NUMERIC_VARIANTS) + 1
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


def _packed_layer(root: Path, directory: str, fam: str, field_map, scenarios):
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
            scenario = case.get("scenario") or scenarios.get(cid)
            if scenario is None:
                continue
            out.setdefault(scenario, {}).update(
                {want: case[have] for want, have in field_map.items() if have in case}
            )
    return out


@pytest.fixture(scope="module")
def authored_unified_store():
    return unified_history.load_authored_store(UTILS.parent / "fixtures-unified-v2")


def _store_resource(row):
    prefix = "conformance/fixtures-unified-v2/"
    return {**row, "path": row["path"].removeprefix(prefix),
            **({"archive": {**row["archive"], "path": row["archive"]["path"].removeprefix(prefix)}}
               if "archive" in row else {})}


def retained_unified_snapshots(store, *, families=None, source_root=None, resources=None):
    """Yield complete views of selected sparse or complete retained captures."""
    root = source_root or store.root
    if resources is None:
        catalog = json.loads((UTILS.parent / "viewer-catalog.json").read_bytes())
        resources = catalog["sources"]["unified"]
    resources = ci_coverage.select_resources(
        [row for row in resources
         if row["kind"] == "history" and row["implementation"] == "dynamo_v2"
         and (families is None or row["family"] in families)], profile="exhaustive")
    sparse = any("schema_version" not in unified_history.load_yaml(root / _store_resource(row)["path"])
                 for row in resources if "archive" not in row)
    if sparse or (root.parent.parent / "conformance/history-datasets.json").is_file():
        resolved = unified_history.load_store(root)
        selected_histories = {}
        for resource in resources:
            capture_id = f"{resource['implementation']}-{resource['version']}"
            selected_histories.setdefault((resource["family"], resource["implementation"]), {})[capture_id] = resource
        for key, selected in selected_histories.items():
            history = resolved.histories[key]
            for capture_id, records in history.iter_resolved_snapshots(set(selected), include_members=True):
                yield selected[capture_id], history.complete_snapshot(capture_id, records=records)
        return
    selected = [_store_resource(row) for row in resources]
    with unified_history._store_read_lock(root):
        for resource, data in unified_archives.read_sources(root, selected):
            document = yaml.load(data, Loader=unified_history.StrictLoader)
            assert document["schema_version"] == 2
            yield resource, document


def _materialized_unified_root(tmp_path: Path, store) -> Path:
    root = tmp_path / "unified"
    # This comparison consumes only authored inputs and GOLDEN, never capture views.
    inputs = unified_history.Store(store.root, store.families, {})
    unified_history.materialize_store(UTILS.parent / "fixtures-unified-v2", root, store=inputs)
    return root


def _taxonomy_scenarios(fam: str, store):
    """Taxonomy id (`UNIFIED.34-guided_json_bare_tool_header_recovers_inside_a_thought`) -> scenario slug, from packaged input layers.

    The golden shard keys by taxonomy id and carries no scenario field, so the mapping
    comes from the one shard holding both. A key JOIN, not a value normalization.
    """
    out = {}
    for case in store.families[fam].cases.values():
        if case["scenario"] and case["display_id"]:
            out[case["display_id"]] = case["scenario"]
    return out


def test_every_case_triple_is_identical_at_every_layer(tmp_path, authored_unified_store):
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

    Why this matters at all: the emitter once ate the leading space of `34-guided_json_bare_tool_header_recovers_inside_a_thought`, the spec
    and feed carried the fix, and the loose and packaged inputs kept the pre-fix bytes
    because they had been exploded first. Every gate was green and the shipped corpus was
    wrong.
    """
    checked = 0
    store = authored_unified_store
    materialized = _materialized_unified_root(tmp_path, store)
    for fam in NATIVE_FAMILIES:
        spec = _emitted_spec(fam)
        scenarios = _taxonomy_scenarios(fam, store)
        packed_in = _packed_layer(materialized, "inputs", fam, {"input": "input", "init": "init"}, scenarios)
        packed_gold = _packed_layer(materialized, "golden", fam, {"golden": "assembled"}, scenarios)

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
    assert checked == sum(len(build_cases(f)) for f in NATIVE_FAMILIES), (
        "the gate must cover every generated case with a retained authored family"
    )


def _assert_retained_capture_coverage(captured, expected):
    assert captured, "no retained Dynamo Unified captures in history"
    assert expected <= captured, f"missing retained captures {sorted(expected - captured)}"


def _assert_reference_backfill_record(record, document, request, version):
    bound = {**record, **document.get("record_metadata", {})}
    selected = capture_stimulus.select_peer_observation(
        bound, request, implementation="dynamo_v2", version=version,
    )
    receipt = selected.get("producer_origin") or bound.get("dynamo_v2_capture_provenance")
    assert isinstance(receipt, dict), "reference backfill has no execution receipt"
    assert receipt.get("implementation") == "dynamo_v2", "reference backfill producer differs"
    assert receipt.get("version") == version, "reference backfill version differs"
    for key in ("source_sha256", "adapter_sha256", "job_sha256", "results_sha256", "cargo_lock_sha256"):
        assert re.fullmatch(r"[0-9a-f]{64}", receipt.get(key, "")), f"reference backfill has invalid {key}"
    assert selected.get("capture_input") == request, "reference backfill request differs"


def test_reference_backfills_require_request_bound_version_receipts(authored_unified_store, historical_unified_sources):
    root, resources = historical_unified_sources
    snapshots = 0
    backfills = 0
    retained_absences = []
    for resource, document in retained_unified_snapshots(authored_unified_store, source_root=root, resources=resources):
        snapshots += 1
        family = authored_unified_store.families[resource["family"]]
        version = resource["version"]
        introduced = {"UNIFIED.7-unused_reference_graph_parameter_types", "UNIFIED.7-nullable_reference_alias_literals", "UNIFIED.7-local_schema_id_preserves_type"}
        if family.name != "glm47":
            introduced |= {f"UNIFIED.7-glm_ref_{suffix}" for suffix in ("object", "encoded_targets", "json_looking_strings", "scalar_types")}
        if family.name == "qwen3":
            introduced.add("UNIFIED.7-arg_string_null_ref")
        introduced_at = (0, 7, 29) if family.name == "beam" else (0, 7, 18)
        if tuple(int(part) for part in version.split(".")) >= introduced_at:
            measured = {family.cases[key]["display_id"] for key in document["records"]}
            missing = introduced - measured
            if missing:
                source = unified_archives.read_source(root, _store_resource(resource))
                assert hashlib.sha256(source).hexdigest() == resource["sha256"]
                retained_absences.append({"family": family.name, "version": version, "missing": sorted(missing), "source_sha256": resource["sha256"]})
            continue
        for case_id, record in document["records"].items():
            case = family.cases[case_id]
            if case["display_id"] in introduced:
                value, metadata = unified_history.materialized_record(case, record["value"])
                _assert_reference_backfill_record(value, metadata, case["request"], version)
                backfills += 1
    assert snapshots > 0
    print(f"reference history qualification passed: {snapshots} Dynamo snapshots, {backfills} request-bound backfill records")
    print("retained reference absences (backfill deferred): " + json.dumps(retained_absences, sort_keys=True))

@pytest.mark.parametrize("corruption", [None, "receipt", "producer", "version", "request", "results"])
def test_reference_backfill_binding_rejects_unproven_older_results(corruption):
    request = capture_stimulus.capture_input(next(iter(build_cases("qwen3").values())))
    record = {"capture_input": deepcopy(request), "unavailable": "unsupported request mode"}
    receipt = {"implementation": "dynamo_v2", "version": "0.1.0",
               **{key: "a" * 64 for key in ("source_sha256", "adapter_sha256", "job_sha256", "results_sha256", "cargo_lock_sha256")}}
    document = {"record_metadata": {"dynamo_v2_capture_provenance": receipt}}
    if corruption == "receipt":
        document.clear()
    elif corruption == "producer":
        receipt["implementation"] = "dynamo_v1"
    elif corruption == "version":
        receipt["version"] = "0.1.1"
    elif corruption == "request":
        record["capture_input"]["input"] += "altered request"
    elif corruption == "results":
        receipt.pop("results_sha256")
    if corruption is None:
        _assert_reference_backfill_record(record, document, request, "0.1.0")
    else:
        with pytest.raises(AssertionError, match="reference backfill"):
            _assert_reference_backfill_record(record, document, request, "0.1.0")


def test_retained_unified_captures_cover_the_current_corpus(authored_unified_store):
    store = authored_unified_store
    captured = set()
    # Coverage is a latest-capture assertion, independent of the parity shard.
    catalog = json.loads((UTILS.parent / "viewer-catalog.json").read_bytes())
    resources = [row for row in catalog["sources"]["unified"]
                 if row["kind"] == "history" and row["implementation"] == "dynamo_v2"]
    latest = {}
    for row in resources:
        if row["family"] not in latest or unified_history._capture_release_sort_key(row["version"]) > unified_history._capture_release_sort_key(latest[row["family"]]["version"]):
            latest[row["family"]] = row
    root = UTILS.parent / "fixtures-unified-v2"
    selected = [_store_resource(row) for row in latest.values()]
    with unified_history._store_read_lock(root):
        for resource, data in unified_archives.read_sources(root, selected):
            document = yaml.load(data, Loader=unified_history.StrictLoader)
            family = resource["family"]
            for case_id in document["records"]:
                case = store.families[family].cases[case_id]
                case_key = case["display_id"] or case["historical_ids"][0]
                captured.add((family, historical_unified_case_key(family, case_key)))
    expected = {
        (family, numbered_id(_scenario_of(case_id, family)))
        for family in NATIVE_FAMILIES for case_id in build_cases(family)
    }
    _assert_retained_capture_coverage(captured, expected)


def test_retained_capture_coverage_rejects_one_missing_case():
    expected = {("qwen3", "UNIFIED.1-tool_only"), ("qwen3", "UNIFIED.32-guided_json_native_markup_only")}
    with pytest.raises(AssertionError, match="missing retained captures"):
        _assert_retained_capture_coverage(expected - {("qwen3", "UNIFIED.32-guided_json_native_markup_only")}, expected)


def _family_value(scenario, family):
    family = G.unified_grammar_family(family)
    if scenario == "guided_json_noncall_before_native_array_before_payload":
        return "{}" + _family_value("guided_json_native_parameter_array_before_payload", family)
    reason_open, reason_close, _, _ = control_tokens(family)
    marker = G._NATIVE_QUOTED_CONTROL.get(family, control_tokens(family)[2])
    if family == "beam":
        # Beam's markers are special tokens, so a quotation never keeps one literal
        # (UNIFIED_CASES.md P6): the prose before the quote stays and the quoted header
        # is framing. The authored beam goldens therefore end at the opening quote.
        beam_quoted = {
            "native_quoted_control_in_response": 'The literal "',
            "native_quoted_control_in_reasoning": 'The literal "',
            "native_quoted_control_then_call": 'The literal "',
            "native_single_quoted_word_control": "The literal 'example ",
            "native_single_quote_contraction_response": "The literal 'doesn't ",
            "native_single_quote_contraction_reasoning": "The literal 'doesn't ",
            "native_quoted_incomplete_header": "The literal `",
        }
        if scenario in beam_quoted:
            return beam_quoted[scenario]
    if scenario in {
        "native_quoted_control_in_response",
        "native_quoted_control_in_reasoning",
        "native_quoted_control_then_call",
    }:
        return f'The literal "{marker}" marker is part of the explanation.'
    if scenario == "native_single_quoted_word_control":
        return f"The literal 'example {marker} marker' is part of the explanation."
    if scenario in {
        "native_single_quote_contraction_response",
        "native_single_quote_contraction_reasoning",
    }:
        return f"The literal 'doesn't {marker} marker' stays quoted."
    if scenario in {
        "guided_quoted_reasoning_closer_named",
        "guided_quoted_reasoning_closer_required",
    }:
        return f"The literal 'doesn't {reason_close} marker' stays quoted."
    if scenario in {
        "guided_response_quoted_control_braces_named",
        "guided_response_quoted_control_braces_required",
    }:
        return f'The literal "{marker} {{ example }}" stays visible. '
    if scenario == "native_quoted_incomplete_header":
        header = G.r_tool(family, "get_weather", "city", "Paris", 0).split("Paris", 1)[0]
        return f"The literal `{header}` header is part of the explanation."
    if scenario == "deepseek_v41_mixed_control_text_in_string":
        if family == "gpt_oss":
            implicit = reason_open.removeprefix("<|start|>assistant")
            return f' {implicit}quoted{reason_close} <|start|> <|message|> <|call|> <|return|> &amp; "x"\\\n '
        return G._MIXED_CONTROL_STRINGS[G.unified_grammar_family(family)]
    if scenario == "arg_marker_in_string":
        close = {
            "beam": "<|call|>",
            "deepseek_v4": "</｜DSML｜invoke>",
            "deepseek_v41": "</｜DSML｜ invoke>",
            "gemma4": "}<tool_call|>",
            "glm47": "</tool_call>",
            "kimi_k2": "<|tool_call_end|>",
            "kimi_k3": "<|close|>call<|sep|>",
            "muse_glimmer": "</atem:function_calls>",
            "muse_spark": "</atem:function_calls>",
            "gpt_oss": "<|call|>",
            "qwen3": "</tool_call>",
        }[family]
        return f"git log {close} --oneline"
    if scenario in {"reason_markup_in_arg", "reason_markup_in_arg_with_text"}:
        if family == "gpt_oss":
            return "<|start|>assistant<|channel|>analysis<|message|>secret<|end|>"
        return ("to=self<|message|>reconsider" if family in {"muse_glimmer", "muse_spark"}
                else f"{reason_open}reconsider{reason_close}")
    if scenario == "guided_json_marker_inside_argument":
        return reason_close
    if scenario == "gpt_oss_control_marker_inside_argument_string":
        return {
            "deepseek_v4": '<think></think><｜DSML｜tool_calls><｜DSML｜invoke name="lookup"><｜DSML｜parameter name="q" string="true"></｜DSML｜parameter></｜DSML｜invoke></｜DSML｜tool_calls><｜DSML｜</｜DSML｜',
            "deepseek_v41": '<think></think><｜DSML｜ calls><｜DSML｜ invoke name="lookup"><｜DSML｜ parameter name="q" string="true"></｜DSML｜ parameter></｜DSML｜ invoke></｜DSML｜ calls><｜DSML｜</｜DSML｜',
            "gpt_oss": "<|start|><|channel|><|constrain|><|message|><|end|><|return|><|call|>",
            "kimi_k2": "<think></think><|tool_calls_section_begin|><|tool_call_begin|><|tool_call_argument_begin|><|tool_call_end|><|tool_calls_section_end|>",
            "kimi_k3": "<|open|><|sep|><|close|><|end_of_msg|>",
        }[family]
    if scenario == "gpt_oss_quoted_message_header_literal":
        return {
            "gpt_oss": "<|call|><|start|>assistant<|channel|>final<|message|>literal<|return|>",
            "muse_glimmer": "<|eom|><|start|>assistant to=user<|message|>literal<|eom|>",
            "muse_spark": "<|eom|><|start|>assistant to=user<|message|>literal<|eom|>",
            "kimi_k3": "<|close|>call<|sep|><|open|>response<|sep|>literal<|close|>response<|sep|>",
        }[family]
    quoted = {
        "guided_json_quoted_bare_header_in_answer": "self",
        "guided_json_quoted_bare_tool_header_in_answer": "get_weather",
        "guided_json_quoted_bare_header_after_payload": "self",
    }
    if scenario in quoted:
        recipient = quoted[scenario]
        if family == "gpt_oss":
            reason_open = reason_open.removeprefix("<|start|>assistant")
        return (f"I mean to={recipient}literal" if family in {"muse_glimmer", "muse_spark"}
                else f"I mean {reason_open}{recipient} literal{reason_close}")
    if scenario == "guided_json_narrated_invoke_in_reasoning":
        # Beam: the narrated tool header is a framed `<|start|>` that ends the open
        # thought, so the words after it are not reasoning (class F divergence).
        return "I'll use " if family == "beam" else "I'll use  next"
    if scenario == "orphan_close_after_prose":
        return "I will check that. "
    if scenario in {
        "prefilled_response_reasoning_markers_literal",
        "guided_json_native_parameter_object_before_payload",
        "guided_json_native_parameter_array_before_payload",
        "guided_json_native_parameter_string_before_payload",
    }:
        if scenario.startswith("guided_json_native_parameter_") and family not in {"qwen3", "muse_glimmer", "muse_spark"}:
            return f"I mean {reason_open}self literal{reason_close}"
        suffix = " then a call" if scenario == "prefilled_response_reasoning_markers_literal" else ""
        if family == "gpt_oss" and scenario == "prefilled_response_reasoning_markers_literal":
            return "<think>literal</think> then a call"
        if family == "beam" and scenario == "prefilled_response_reasoning_markers_literal":
            # Beam's framing is special tokens: under Response the framed analysis
            # message is consumed as control and its body is visible text.
            return "literal then a call"
        return ("to=selfliteral" if family in {"muse_glimmer", "muse_spark"}
                else f"{reason_open}literal{reason_close}") + suffix
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


def _json_values(raw, *, exact_numbers=False):
    decoder = json.JSONDecoder(parse_float=Decimal if exact_numbers else float)
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


def _parse_gemma_value(raw, index=0, *, exact_numbers=False):
    while index < len(raw) and raw[index].isspace():
        index += 1
    if raw.startswith('<|"|>', index):
        start = index + len('<|"|>')
        end = raw.find('<|"|>', start)
        assert end >= 0, ("unterminated Gemma string", raw)
        return raw[start:end], end + len('<|"|>')
    if index < len(raw) and raw[index] in "{[":
        opener = raw[index]
        closer = "}" if opener == "{" else "]"
        index += 1
        values = {} if opener == "{" else []
        while True:
            while index < len(raw) and (raw[index].isspace() or raw[index] == ","):
                index += 1
            if index < len(raw) and raw[index] == closer:
                return values, index + 1
            if opener == "{":
                key = re.match(r"[A-Za-z0-9_.-]+", raw[index:])
                assert key is not None, ("invalid Gemma object key", raw[index:])
                index += key.end()
                assert index < len(raw) and raw[index] == ":", ("missing Gemma key separator", raw)
                value, index = _parse_gemma_value(raw, index + 1, exact_numbers=exact_numbers)
                values[key[0]] = value
            else:
                value, index = _parse_gemma_value(raw, index, exact_numbers=exact_numbers)
                values.append(value)
    token = re.match(r"(?:true|false|null|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", raw[index:])
    assert token is not None, ("unsupported Gemma value", raw[index:])
    return json.loads(token[0], parse_float=Decimal if exact_numbers else float), index + token.end()


def _atem_parameter_end(raw, value_start):
    # Independently decode a JSON value before looking for its XML closer;
    # otherwise this fixture validator repeats the parser's quoted-closer bug.
    minimum = value_start
    at = value_start + len(raw[value_start:]) - len(raw[value_start:].lstrip())
    if raw[at:at + 1] in {"{", "[", '"'}:
        try:
            _, minimum = json.JSONDecoder().raw_decode(raw, at)
        except json.JSONDecodeError:
            pass
    return raw.find("</atem:parameter>", minimum)


def _native_input_calls(family, raw, *, exact_numbers=False):
    """Read authored complete argument fields, not runtime recovery decisions.

    This fixture-only projection ignores invoke EOF policy: a syntactically present
    body is not evidence that the parser must dispatch it. The caller keeps DSML's
    empty-output EOF contract separate.
    """
    if family == "glm47":
        calls = []
        cursor = 0
        while True:
            header = re.search(
                r'(?:<tool_call>|\A)([A-Za-z0-9_.-]+)(?=<arg_key>|</tool_call>)',
                raw[cursor:],
            )
            if header is None:
                return calls
            name = header[1]
            at = cursor + header.end()
            arguments = {}
            while raw.startswith("<arg_key>", at):
                key_end = raw.find("</arg_key>", at)
                assert key_end >= 0, (family, "unterminated argument key", raw)
                key = raw[at + len("<arg_key>"):key_end]
                value_start = key_end + len("</arg_key>")
                assert raw.startswith("<arg_value>", value_start), (
                    family, "missing argument value", raw,
                )
                value_start += len("<arg_value>")
                value_end = raw.find("</arg_value>", value_start)
                if value_end < 0:
                    break
                arguments[key] = raw[value_start:value_end]
                at = value_end + len("</arg_value>")
            calls.append({"kind": "tool_call", "name": name, "arguments": arguments})
            outer_end = raw.find("</tool_call>", at)
            if outer_end < 0:
                return calls
            cursor = outer_end + len("</tool_call>")

    if family == "beam":
        # Independent authoring-contract decoder for the Beam wire (not the parser):
        # only a `to=NAME` header on the `analysis` channel is a tool call; its body
        # runs to the next framed header. Values keep their literal bytes, with the
        # one LF the renderer frames each edge with removed; no typing.
        header = re.compile(
            r'<\|start\|>assistant to=([A-Za-z_][A-Za-z0-9_.\-]*)<\|channel\|>analysis'
            r'(?: <\|constrain\|>[A-Za-z0-9]+)?<\|message\|>'
        )
        # The reference tag rule: a value runs to the FIRST `</rfl:parameter>`, so an
        # embedded header is value data. Parameters are read in sequence after the
        # header; anything else (a terminator, a new header, EOF) ends the body.
        parameter = re.compile(r'\s*<rfl:parameter name="([^"]+)">(.*?)</rfl:parameter>', re.S)
        calls = []
        for match in header.finditer(raw):
            at = match.end()
            arguments = {}
            while True:
                item = parameter.match(raw, at)
                if item is None:
                    break
                arguments[item[1]] = item[2].removeprefix("\n").removesuffix("\n")
                at = item.end()
            calls.append({"kind": "tool_call", "name": match[1], "arguments": arguments})
        return calls

    family = G.unified_grammar_family(family)
    if family == "gpt_oss":
        calls = []
        decoder = json.JSONDecoder(parse_float=Decimal if exact_numbers else float)
        endings = ("<|call|>", "<|end|>", "<|return|>")
        owned_until = 0
        headers = re.finditer(
            r"(?:<\|start\|>assistant|<\|channel\|>|\A(?=\s*to=functions\.))(?P<header>.*?)<\|message\|>",
            raw,
            re.S,
        )
        for header in headers:
            if header.start() < owned_until:
                continue
            recipient = re.search(r"\bto=functions\.([\w.-]+)", header["header"])
            if recipient is None:
                continue
            name = recipient[1]
            body_start = header.end()
            at = body_start + len(raw[body_start:]) - len(raw[body_start:].lstrip())
            if raw.startswith(endings, at):
                calls.append({"kind": "tool_call", "name": name, "arguments": {}})
                continue
            try:
                arguments, end = decoder.raw_decode(raw, at)
            except json.JSONDecodeError:
                continue
            owned_until = end
            if not isinstance(arguments, dict) or not raw[end:].lstrip().startswith(endings):
                continue
            calls.append({"kind": "tool_call", "name": name, "arguments": arguments})
        return calls

    headers = {
        "deepseek_v4": r'<｜DSML｜invoke name="([^"]+)">',
        "deepseek_v41": r'<｜DSML｜ invoke name="([^"]+)">',
        "qwen3": r'<function=([^>]+)>',
        "muse_glimmer": r'<atem:invoke name="([^"]+)">',
        "muse_spark": r'<atem:invoke name="([^"]+)">',
        "gemma4": r'call:([\w.-]+)\{',
        "kimi_k2": r'<\|tool_call_begin\|>(?:functions\.)?([\w.-]+):\d+<\|tool_call_argument_begin\|>',
        "kimi_k3": r'<\|open\|>\s*call tool="([^"]+)" index="\d+"\s*<\|sep\|>',
    }
    if family in {"deepseek_v4", "deepseek_v41"}:
        gap = " " if family == "deepseek_v41" else ""
        invocation_end = f"</｜DSML｜{gap}invoke>"
        parameter_pattern = re.compile(
            rf'<｜DSML｜{gap}parameter name="([^"]+)" string="(true|false)">(.*?)</｜DSML｜{gap}parameter>',
            re.S,
        )
        decoder = json.JSONDecoder(parse_float=Decimal if exact_numbers else float)
        calls = []
        cursor = 0
        while match := re.search(headers[family], raw[cursor:]):
            name = match[1]
            body_start = cursor + match.end()
            start = body_start + len(raw[body_start:]) - len(raw[body_start:].lstrip())
            try:
                value, end = decoder.raw_decode(raw, start)
            except json.JSONDecodeError:
                value = None
            else:
                suffix = raw[end:]
                closer_start = end + len(suffix) - len(suffix.lstrip())
                if isinstance(value, dict) and raw.startswith(invocation_end, closer_start):
                    cursor = closer_start + len(invocation_end)
                    calls.append({"kind": "tool_call", "name": name, "arguments": value})
                    continue

            arguments = {}
            body_cursor = body_start
            while True:
                body_end = raw.find(invocation_end, body_cursor)
                parameter = parameter_pattern.search(raw, body_cursor)
                if parameter and (body_end < 0 or parameter.start() < body_end):
                    key, is_string, value = parameter.groups()
                    arguments[key] = value if is_string == "true" else json.loads(
                        value, parse_float=Decimal if exact_numbers else float
                    )
                    body_cursor = parameter.end()
                    continue
                cursor = len(raw) if body_end < 0 else body_end + len(invocation_end)
                break
            calls.append({"kind": "tool_call", "name": name, "arguments": arguments})
        return calls

    found = list(re.finditer(headers[family], raw))
    calls = []
    for index, match in enumerate(found):
        body = raw[match.end():found[index + 1].start() if index + 1 < len(found) else len(raw)]
        # Muse parameters own nested invocation-shaped text. The independent
        # fixture projection must not clip an outer argument at an inner header.
        if family in {"muse_glimmer", "muse_spark"}:
            tail = raw[match.end():]
            cursor = 0
            while True:
                close = tail.find("</atem:invoke>", cursor)
                parameter = re.search(r'<atem:parameter name="[^"]+">', tail[cursor:])
                if parameter is None or (close >= 0 and close < cursor + parameter.end()):
                    body = tail[:close] if close >= 0 else tail
                    break
                value_start = cursor + parameter.end()
                end = _atem_parameter_end(tail, value_start)
                if end < 0:
                    body = tail
                    break
                cursor = end + len("</atem:parameter>")
        arguments = {}
        if G.unified_grammar_family(family) == "qwen3":
            # The generator frames values with one newline; payload whitespace is data.
            arguments = {key: value.removeprefix("\n").removesuffix("\n")
                         for key, value in re.findall(r'<parameter=([^>]+)>(.*?)</parameter>', body, re.S)}
        elif family in {"muse_glimmer", "muse_spark"}:
            cursor = 0
            while parameter := re.search(r'<atem:parameter name="([^"]+)">', body[cursor:]):
                key = parameter[1]
                value_start = cursor + parameter.end()
                end = _atem_parameter_end(body, value_start)
                if end < 0:
                    break
                value = body[value_start:end]
                cursor = end + len("</atem:parameter>")
                try:
                    arguments[key] = json.loads(
                        value, parse_float=Decimal if exact_numbers else float
                    )
                except json.JSONDecodeError:
                    arguments[key] = value
        elif family == "gemma4":
            arguments, _ = _parse_gemma_value("{" + body, exact_numbers=exact_numbers)
        elif family == "kimi_k2":
            start = len(body) - len(body.lstrip(" \t\r\n"))
            decoder = json.JSONDecoder(parse_float=Decimal if exact_numbers else float)
            try:
                arguments, _ = decoder.raw_decode(body, start)
            except json.JSONDecodeError:
                # Kimi K2 preserves a named malformed JSON call as empty arguments.
                arguments = {}
        else:
            pattern = r'<\|open\|>\s*argument key="([^"]+)" type="([^"]+)"\s*<\|sep\|>(.*?)<\|close\|>\s*argument\s*<\|sep\|>'
            for key, kind, value in re.findall(pattern, body, re.S):
                arguments[key] = value if kind == "string" else json.loads(
                    value, parse_float=Decimal if exact_numbers else float
                )
            if not arguments and re.match(r'<\|open\|>\s*json ', body):
                values = _json_values(body, exact_numbers=exact_numbers)
                if values:
                    arguments = values[0]
        calls.append({"kind": "tool_call", "name": match[1], "arguments": arguments})
    return calls


@pytest.mark.parametrize("header", [
    "<|start|>assistant to=functions.lookup<|channel|>commentary",
    "<|channel|>commentary to=functions.lookup",
    " to=functions.lookup<|channel|>commentary <|constrain|>json",
])
@pytest.mark.parametrize("end", ["<|call|>", "<|end|>", "<|return|>"])
def test_harmony_fixture_projection_keeps_header_and_boundary_variants(header, end):
    raw = header + '<|message|>{"q":"x"}' + end
    assert _native_input_calls("gpt_oss", raw) == [
        {"kind": "tool_call", "name": "lookup", "arguments": {"q": "x"}}
    ]


def test_harmony_fixture_projection_keeps_plaintext_and_quoted_headers_as_data():
    assert _native_input_calls("gpt_oss", "to=functions.lookup ordinary prose") == []
    quoted = '<|start|>assistant to=functions.fake<|message|>{}<|call|>'
    arguments = {"q": quoted}
    raw = '<|start|>assistant to=functions.lookup<|message|>' + json.dumps(arguments) + '<|call|>'
    assert _native_input_calls("gpt_oss", raw) == [
        {"kind": "tool_call", "name": "lookup", "arguments": arguments}
    ]


@pytest.mark.parametrize(("family", "invocation_start", "fake_invocation", "invocation_end"), [
    ("deepseek_v4", '<｜DSML｜invoke name="inspect">', '<｜DSML｜invoke name="fake">', "</｜DSML｜invoke>"),
    ("deepseek_v41", '<｜DSML｜ invoke name="inspect">', '<｜DSML｜ invoke name="fake">', "</｜DSML｜ invoke>"),
])
def test_deepseek_json_body_projection_keeps_marker_text_inside_json_data(
    family, invocation_start, fake_invocation, invocation_end
):
    marker_text = f"literal {fake_invocation} and {invocation_end}"
    valid = invocation_start + json.dumps({"value": marker_text}) + invocation_end
    malformed = invocation_start + '{"ok":true} trailing text' + invocation_end

    assert _native_input_calls(family, valid) == [{
        "kind": "tool_call",
        "name": "inspect",
        "arguments": {"value": marker_text},
    }]
    assert _native_input_calls(family, malformed) == [{
        "kind": "tool_call",
        "name": "inspect",
        "arguments": {},
    }]


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


def test_kimi_k2_fixture_projection_skips_framing_whitespace_without_trimming_values():
    value = " \n\tinside \r\n"
    body = json.dumps({"q": value})
    raw = "<|tool_call_begin|>functions.lookup:0<|tool_call_argument_begin|> \n\t" + body + "<|tool_call_end|>"
    assert _native_input_calls("kimi_k2", raw) == [
        {"kind": "tool_call", "name": "lookup", "arguments": {"q": value}},
    ]


def _assert_malformed_recovery(family, case):
    family = G.unified_grammar_family(family)
    raw = case["input"]
    assert case["input_chunks"] == [raw]
    assert case["init"] == {"starting_state": "None", "tool_output_mode": "Native", "named_tool": None}
    openers = {
        "kimi_k2": "<|tool_call_begin|>functions.echo:",
        "kimi_k3": '<|open|>call tool="echo"',
        "qwen3": "<tool_call>\n<function=echo>",
        "glm47": "<tool_call>echo",
        "deepseek_v4": '<｜DSML｜invoke name="echo">',
        "deepseek_v41": '<｜DSML｜ invoke name="echo">',
        "gemma4": "<|tool_call>call:echo{",
        "muse_glimmer": "<|start|>assistant to=echo<|message|>",
        "muse_spark": "<|start|>assistant to=echo<|message|>",
        "gpt_oss": "<|start|>assistant<|channel|>commentary to=functions.echo",
        "beam": "<|start|>assistant to=echo<|channel|>",
    }
    at = raw.index(openers[family])
    prefix, suffix = raw[:at], raw[at:]
    broken = (
        'x:<|"|>unfinished' if family == "gemma4"
        else '{\\"x\\":\\"unfinished' if family == "gpt_oss"
        else '{"x":"unfinished'
    )
    if family == "gpt_oss":
        assert broken in prefix
        calls = _native_input_calls(family, prefix)
        assert len(calls) == 1
        assert calls[0]["arguments"] == {
            "value": '{"x":"unfinished',
        }
    else:
        assert broken in prefix and broken + '"' not in prefix
    assert "bad" in prefix
    closers = {
        "kimi_k2": "<|tool_call_end|>", "kimi_k3": "<|close|>json<|sep|><|close|>call<|sep|>",
        "qwen3": "</parameter>\n</function>\n</tool_call>", "glm47": "</arg_value></tool_call>",
        "deepseek_v4": "</｜DSML｜parameter></｜DSML｜invoke>",
        "deepseek_v41": "</｜DSML｜ parameter></｜DSML｜ invoke>",
        "gemma4": "}}<tool_call|>", "muse_glimmer": "</atem:invoke>", "muse_spark": "</atem:invoke>",
        "gpt_oss": "<|call|>",
        "beam": "</rfl:parameter><|call|>",
    }
    assert closers[family] in prefix
    echoes = [{"kind": "tool_call", "name": "echo", "arguments": {"value": value}} for value in ("é", "Café")]
    assert _native_input_calls(family, suffix) == echoes
    bad = []
    if family == "kimi_k2":
        bad = [{"kind": "tool_call", "name": "bad", "arguments": {}}]
    elif family in {"qwen3", "glm47", "deepseek_v4", "muse_glimmer", "gpt_oss", "muse_spark", "beam"}:
        bad = [{"kind": "tool_call", "name": "bad", "arguments": {"value": '{"x":"unfinished'}}]
    assert case["golden"] == bad + echoes
    if family != "gpt_oss":
        assert case["tools"] == [
            {"name": "bad", "parameters": {"type": "object", "properties": {"value": {"type": "object"}}}},
            {"name": "echo", "parameters": {"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"]}},
        ]
    return echoes


def _assert_input_carries_events(family: str, scenario: str, case: dict) -> None:
    default_reasoning = bool(G.default_response_prefix(family))
    family = G.unified_grammar_family(family)
    if scenario == "malformed_json_then_two_valid_calls":
        _assert_malformed_recovery(family, case)
        return
    raw = case["input"]
    tools = [event for event in case["golden"] if event["kind"] == "tool_call"]
    if tools:
        if case["init"]["tool_output_mode"] == "Native":
            candidates = _native_input_calls(family, raw, exact_numbers=True)
            for candidate in candidates:
                tool_schema = next(
                    (tool for tool in case.get("tools", []) if tool["name"] == candidate["name"]),
                    None,
                )
                if tool_schema is None:
                    continue
                parameters = tool_schema.get("parameters", {})
                properties = parameters.get("properties", {})
                for key, value in candidate["arguments"].items():
                    schema = properties.get(key, {})
                    # Families whose wire carries untyped text values: the authoring
                    # contract types them against the declared schema (a literal
                    # `null` for a nullable property, JSON for a non-string one).
                    if G.unified_grammar_family(family) not in {"qwen3", "glm47", "beam"} or not isinstance(value, str):
                        continue
                    if scenario in {probe["name"] for probe in XML_COMPATIBILITY_CASES} and schema == {"type": "custom"}:
                        assert family == "qwen3"
                        candidate["arguments"][key] = ast.literal_eval(value.strip())
                        continue
                    if value == "null" and matches_schema(None, schema, parameters):
                        candidate["arguments"][key] = None
                    elif (not matches_schema(value, schema, parameters)
                          or (G.unified_grammar_family(family) in ("qwen3", "beam") and scenario in {
                              "schema_integer_const_ambiguous", "schema_object_const_union",
                              "schema_type_array_string_integer",
                          }) or family == "qwen3"):
                        try:
                            decoded = json.loads(value, parse_float=Decimal)
                        except json.JSONDecodeError:
                            continue
                        if matches_schema(decoded, schema, parameters) and (
                            family != "qwen3" or not isinstance(decoded, str)
                        ):
                            candidate["arguments"][key] = decoded
        else:
            candidates = []
            for value in _json_values(raw, exact_numbers=True):
                if case["init"]["named_tool"] is not None:
                    candidates.append({"kind": "tool_call", "name": case["init"]["named_tool"], "arguments": value})
                else:
                    for call in value if isinstance(value, list) else [value]:
                        if isinstance(call, dict) and "name" in call and "arguments" in call:
                            candidates.append({"kind": "tool_call", **call})
        cursor = 0
        for event in tools:
            expected = event
            if isinstance(event.get("arguments"), str):
                expected = {
                    **event,
                    "arguments": json.loads(event["arguments"], parse_float=Decimal),
                }
            assert expected in candidates[cursor:], (
                family, scenario, "input call differs from golden", event, candidates,
            )
            cursor = candidates.index(expected, cursor) + 1
    for event in case["golden"]:
        if event["kind"] in {"message_start", "message_end"}:
            assert family == "muse_spark" and case["init"].get("preserve_message_phases")
            if event["kind"] == "message_start":
                recipient = {"commentary":"commentary", "final_answer":"user"}[event["phase"]]
                assert f"assistant to={recipient}<|message|>" in raw
            else:
                assert "<|eom|>" in raw or "<|eot|>" in raw
            continue
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
        if isinstance(arguments, str):
            arguments = json.loads(arguments, parse_float=Decimal)
        assert name in raw or case["init"]["named_tool"] == name, (family, scenario, "input tool name", name)
        assert isinstance(arguments, dict), (family, scenario, "argument object")
        for key, value in arguments.items():
            assert key in raw, (family, scenario, "input argument key", key)
            if isinstance(value, str):
                spellings = (value, json.dumps(value, ensure_ascii=False)[1:-1], json.dumps(value)[1:-1])
                assert any(spelling in raw for spelling in spellings), (family, scenario, "input argument value", key, value)
            elif value is None:
                assert "null" in raw, (family, scenario, "input null argument value", key)
    reasons = [event for event in case["golden"] if event["kind"] == "reasoning"]
    if reasons and case["init"]["starting_state"] == "None" and not default_reasoning:
        if family == "gpt_oss":
            assert re.search(r"<\|channel\|>(?:analysis|redacted)<\|message\|>", raw), (
                family, scenario, "missing Harmony reasoning channel",
            )
        else:
            assert control_tokens(family)[0] in raw, (family, scenario, "missing reasoning opener")
    if reasons and any(event["kind"] != "reasoning" for event in case["golden"]):
        unclosed = {
            "truncated_tool_eof", "prefilled_reasoning_truncated", "default_reasoning_tool_without_closer",
            "guided_json_unterminated_reasoning_then_wrapped_payload",
            "guided_json_bare_tool_header_recovers_inside_a_thought",
            "deepseek_v4_guided_reasoning_opener_inside_native_body",
            "kimi_k3_elided_think_close_to_response",
            "gpt_oss_interrupted_reasoning_then_response",
        }
        if scenario.removesuffix("_pr424") not in unclosed:
            assert control_tokens(family)[1] in raw, (family, scenario, "missing reasoning closer")


BEAM_GRAMMAR_EXCEPTIONS = {
    "guided_json_native_parameter_body_inside_reasoning",
    "guided_json_reasoning_markers_inside_native_parameter",
}
BEAM_NUMERIC_STRING_SCENARIOS = {
    scenario for scenario, label, _schema, raw, expected in NUMERIC_VARIANTS
    if unified_numeric_expected("beam", label, raw, expected)
    != numeric_expected("beam", label, raw, expected)
}


def _assert_generic_cases_span_families(scenarios):
    coverage = defaultdict(set)
    for scenario, families in scenarios.items():
        parent = taxonomy.coverage_case(scenario)
        coverage[parent].update(families)
    singletons = {
        scenario: sorted(families)
        for scenario, families in coverage.items()
        if isinstance(taxonomy.display_tax(scenario)[0], int) and len(families) < 2
    }
    assert not singletons, (
        "single-family generic cases: expand family-native coverage or move to a "
        "named family-specific section", singletons,
    )


def _assert_cross_family_contract(corpus):
    by_scenario = defaultdict(dict)
    clean = {row[0]: row[3] for row in CLEAN}
    prefills = {name for name, segments in clean.items() if segments[0][0] == "reason"} | {"reason_unterminated"}
    for family in NATIVE_FAMILIES:
        for cid, case in corpus[family].items():
            scenario = _scenario_of(cid, family)
            _assert_input_carries_events(family, scenario, case)
            if scenario == "guided_json_quoted_bare_tool_header_in_answer":
                # A tool name inside a thought is prose, not a recipient header.
                assert family in {"muse_glimmer", "muse_spark"} and "to=get_weather<|message|>" in case["input"], (
                    family, scenario, "missing tool-recipient boundary",
                )
            init = dict(case["init"])
            if family == "deepseek_v41" and scenario in prefills:
                assert init == {"starting_state": "Reasoning", "tool_output_mode": "Native", "named_tool": None}
                assert not case["input"].startswith("<think>"), (family, scenario, "prefilled opener must be absent")
                init["starting_state"] = "None"
            events = (_assert_malformed_recovery(family, case) if scenario == "malformed_json_then_two_valid_calls"
                      else _logical_events(scenario, family, case["golden"]))
            if scenario == "gpt_oss_multiple_unknown_and_empty_tools":
                raw_value = "2" if G.unified_grammar_family(family) in {"beam", "qwen3", "glm47"} else 2
                assert case["tools"] == []
                assert events == [{"kind": "tool_call", "name": "run", "arguments": {}},
                                  {"kind": "tool_call", "name": "not_offered", "arguments": {"x": raw_value}}]
                events[1]["arguments"]["x"] = 2
            if scenario in {"gpt_oss_invalid_tool_body_then_empty_call", "gpt_oss_invalid_tool_body_then_valid_call"}:
                empty = scenario.endswith("empty_call")
                malformed = "not-json" if empty else '{"city":'
                assert malformed in case["input"]
                assert case["input"].index(malformed) < case["input"].rindex("get_weather")
                following = [{"kind": "tool_call", "name": "get_weather", "arguments": {} if empty else {"city": "Paris"}}]
                assert events == ([{"kind": "tool_call", "name": "bad", "arguments": {}}] if family in {"kimi_k2", "deepseek_v4"} else []) + following
                events = following
            if scenario == "gpt_oss_truncated_tool_resynchronizes_at_next_message":
                following = [{"kind": "text", "text": "recovered"}]
                assert events == ([{"kind": "tool_call", "name": "lookup", "arguments": {"q": "unfinished"}}]
                                  if family in {"muse_glimmer", "muse_spark"} else []) + following
                events = following
            if scenario == "gpt_oss_quoted_truncated_tool_then_final":
                assert events == [{"kind": "text", "text": "Answer"}], (family, scenario, "missing recovered answer")
            if scenario == "structured_response_after_reasoning":
                # Compare the schema and invalid payloads after checking each
                # native envelope; Harmony generates headers, others prefill.
                assert case["tools"] == []
                assert init["starting_state"] == ("None" if family == "gpt_oss" else "Reasoning")
                response = events[1]["text"]
                if G.unified_grammar_family(family) == "qwen3":
                    assert response.startswith("\n\n")
                    response = response[2:]
                schema = init["structural_tag"]["schema"]
                assert matches_schema(json.loads(response), schema)
                ending = "<|return|>" if family == "gpt_oss" else ""
                assert case["input"].endswith(response + ending)
                prefix = case["input"][:-len(response + ending)]
                probes = init["structural_tag"]["rejected"]
                assert len(probes) == (6 if family == "gpt_oss" else 5)
                if family == "gpt_oss":
                    assert probes[-1] == prefix.replace("final<|message|>", "final<|constrain|>json<|message|>") + "invalid JSON" + ending
                payloads = []
                for probe in probes[:5]:
                    assert probe.startswith(prefix) and probe.endswith(ending)
                    payload = probe[len(prefix):len(probe) - len(ending) if ending else None]
                    try:
                        value = json.loads(payload)
                    except json.JSONDecodeError:
                        pass
                    else:
                        assert not matches_schema(value, schema)
                    payloads.append(payload)
                init["starting_state"] = "None"
                init["structural_tag"] = {"schema": schema, "rejected": payloads}
                events = [events[0], {"kind": "text", "text": response}]
            if family == "gpt_oss" and scenario in G.GPT_OSS_LEGACY_NON_HARMONY_SCENARIOS:
                assert events == [{"kind": "text", "text": "still thinking about it"}]
                # Other grammars opened a tool block; this retained Harmony request did not.
                events = []
            if scenario in {"schema_integer_const_ambiguous", "schema_object_const_union", "schema_type_array_string_integer"}:
                probe = next(probe for probe in CONFORMANCE_CASES if scenario == f"schema_{probe['name']}")
                expected_events = [{"kind": "tool_call", "name": "schema_probe", "arguments": schema_arguments(family, probe)}]
                assert json.dumps(events, sort_keys=True) == json.dumps(expected_events, sort_keys=True)
                assert matches_schema(events[0]["arguments"], case["tools"][0]["parameters"])
                # These native spellings are ambiguous. Validate each family's
                # established interpretation, then compare the shared scenario.
                events = [{"kind": "tool_call", "name": "schema_probe", "arguments": {"value": probe["value"]}}]

            if scenario.startswith("arg_numeric_"):
                events = canonical_events(events)
            if scenario in {"arg_json_null", "arg_string_null"}:
                value = None if scenario == "arg_json_null" else "null"
                expected_type = "string" if value is not None else ["string", "null"]
                assert case["tools"][0]["parameters"]["properties"]["city"]["type"] == expected_type
                assert events == [{"kind": "tool_call", "name": "get_weather", "arguments": {"city": value}}]
            if scenario == "tool_no_close":
                # DSML invokes and Harmony tool messages require their own terminator.
                # Assert that exception independently; do not normalize [] to a call.
                if family in {"deepseek_v4", "deepseek_v41", "gpt_oss"}:
                    assert events == [], (family, scenario, "native message requires termination")
                    expected_eof_call = [
                        {"kind": "tool_call", "name": "get_weather", "arguments": {"city": "Paris"}},
                    ]
                    if family == "gpt_oss":
                        assert _native_input_calls(family, case["input"]) == []
                    else:
                        assert _native_input_calls(family, case["input"]) == expected_eof_call, (
                            family, scenario, "EOF body differs",
                        )
                        gap = " " if family == "deepseek_v41" else ""
                        assert f'</｜DSML｜{gap}parameter>' in case["input"]
                        assert f'</｜DSML｜{gap}invoke>' not in case["input"]
                else:
                    assert events == [{"kind": "tool_call", "name": "get_weather", "arguments": {"city": "Paris"}}]
                    if family == "beam":
                        # Beam's serving grammar lets EOS stand for `<|call|>`: the
                        # delimited body is the call, with no terminator in the input.
                        assert "</rfl:parameter>" in case["input"]
                        assert "<|call|>" not in case["input"]
                events = None
            suppressed = []
            if init["tool_output_mode"] == "GuidedJson":
                # Complete native argument bodies are input even when guided mode
                # suppresses them. Goldens alone cannot detect a changed stress value.
                if scenario == "gemma4_guided_response_extra_body_opener":
                    assert family == "gemma4"
                    assert init["starting_state"] == "Response"
                    assert case["input"] == "é call:get_weather{" + G.GUIDED_ONE_CALL
                    assert events == [
                        {"kind": "text", "text": "é "},
                        {"kind": "tool_call", "name": "get_weather", "arguments": {"city": "Paris"}},
                    ]
                else:
                    suppressed = [call for call in _native_input_calls(family, case["input"]) if call["arguments"]]
                if scenario in {"guided_json_reasoning_markers_inside_native_parameter", "guided_json_reasoning_markers_inside_native_parameter_pr424"}:
                    if family == "beam":
                        # Beam has no bare native invocation: without its routed header the
                        # parameter tag is text, so nothing is suppressed (authored golden).
                        assert suppressed == [], (family, scenario, "Beam suppresses no bare invocation")
                    else:
                        opener, closer, _, _ = control_tokens(family)
                        if family == "gpt_oss" and not scenario.endswith("_pr424"):
                            opener = opener.removeprefix("<|start|>assistant")
                        assert suppressed == [{"kind": "tool_call", "name": "f", "arguments": {"x": opener + "quoted" + closer}}], (
                            family, scenario, "wrong suppressed marker role or payload",
                        )
                        suppressed[0]["arguments"]["x"] = "FAMILY_REASONING_MARKERS"
                if scenario == "guided_json_native_parameter_string_before_payload":
                    for call in suppressed:
                        value = call["arguments"]["x"]
                        assert isinstance(value, str)
                        if value.startswith('"'):
                            call["arguments"]["x"] = json.loads(value)
                if scenario in {
                    "guided_json_native_parameter_object_before_payload",
                    "guided_json_native_parameter_array_before_payload",
                    "guided_json_noncall_before_native_array_before_payload",
                    "guided_response_native_wrapper_whitespace_before_prose",
                }:
                    for call in suppressed:
                        if isinstance(call["arguments"]["x"], str):
                            call["arguments"]["x"] = json.loads(call["arguments"]["x"])
            by_scenario[scenario][family] = (events, init, case["finish_reason"], case["policy"], suppressed)
    assert set(by_scenario) == set(NATIVE_UNIFIED_TAX)
    _assert_generic_cases_span_families(by_scenario)
    for scenario, families in by_scenario.items():
        assert set(families) == (set(G.scenario_families(scenario)) & set(NATIVE_FAMILIES)), (scenario, "applicability", set(families))
        # Beam authors its own golden where its grammar changes the outcome (its parameter
        # tags are plain text outside a routed header; its reasoning markers are special
        # tokens), so it stands outside the shared contract for those scenarios only.
        contracts = [
            (family, contract) for family, contract in families.items()
            if not (family == "beam" and scenario in BEAM_GRAMMAR_EXCEPTIONS | BEAM_NUMERIC_STRING_SCENARIOS)
        ]
        first_family, reference = contracts[0]
        for family, contract in contracts[1:]:
            assert contract == reference, (scenario, first_family, family, "cross-family contract", reference, contract)


def test_all_scenarios_preserve_cross_family_input_and_output_contracts():
    _assert_cross_family_contract({family: build_cases(family) for family in NATIVE_FAMILIES})


def test_gpt_oss_numeric_input_oracle_preserves_exact_decimal_values():
    scenario = "arg_numeric_14_large_decimal"
    case = build_cases("gpt_oss")[f"UNIFIED.{scenario}.gpt_oss"]
    calls = _native_input_calls("gpt_oss", case["input"], exact_numbers=True)
    assert calls[0]["arguments"]["value"] == Decimal("9007199254740993.0")

@pytest.mark.parametrize("scenario", ["reason_only", "reason_markup_in_arg"])
def test_cross_family_contract_rejects_nemotron_only_generic_reasoning(scenario, monkeypatch):
    corpus = {family: build_cases(family) for family in NATIVE_FAMILIES}
    for family, cases in corpus.items():
        if family != "nemotron_v3":
            del cases[f"UNIFIED.{scenario}.{family}"]
    scenario_families = G.scenario_families
    monkeypatch.setattr(
        G, "scenario_families",
        lambda name: frozenset({"nemotron_v3"}) if name == scenario else scenario_families(name),
    )
    with pytest.raises(AssertionError, match="single-family generic cases"):
        _assert_cross_family_contract(corpus)

@pytest.mark.parametrize("family", NATIVE_FAMILIES)
@pytest.mark.parametrize("scenario", ["reason_only", "arg_unicode", "guided_json_named_tool", "future_generic_case"])
def test_generic_case_scope_rejects_any_single_family(family, scenario):
    with pytest.raises(AssertionError, match="single-family generic cases"):
        _assert_generic_cases_span_families({scenario: {family}})


def test_generic_case_scope_uses_display_placement_and_variant_coverage(monkeypatch):
    parent = "arg_marker_in_string"
    variant = "arg_marker_in_string_pr424"
    with pytest.raises(AssertionError, match="single-family generic cases"):
        _assert_generic_cases_span_families({parent: {"gpt_oss"}, variant: {"gpt_oss"}})
    _assert_generic_cases_span_families({parent: {"qwen3"}, variant: {"gpt_oss"}})
    monkeypatch.setitem(taxonomy._DISPLAY_GROUP, parent, "gpt_oss")
    _assert_generic_cases_span_families({parent: {"gpt_oss"}, variant: {"gpt_oss"}})


def test_generic_case_scope_rejects_single_family_semantic_variants(monkeypatch):
    monkeypatch.setitem(taxonomy.SHARED_CASE_VARIANTS, "future_semantic_variant", taxonomy.CaseVariant("two_calls", "Additional behavior"))
    for scenario in ("gpt_oss_multiple_unknown_and_empty_tools", "future_semantic_variant"):
        with pytest.raises(AssertionError, match="single-family generic cases"):
            _assert_generic_cases_span_families({"two_calls": set(NATIVE_FAMILIES), scenario: {"gpt_oss"}})


def test_default_reasoning_cases_use_a_named_family_section():
    for scenario in ("default_reasoning_without_opener", "default_reasoning_tool_without_closer"):
        assert G.scenario_families(scenario) == {"nemotron_v3"}
        assert taxonomy.display_tax(scenario)[0] == "nemotron_v3"
        assert taxonomy.tax(scenario)[0] in {10, 12}


@pytest.mark.parametrize("family", [family for family in NATIVE_FAMILIES if family not in {"muse_glimmer", "muse_spark"}])
def test_cross_family_contract_rejects_prose_only_recipient_case(family):
    corpus = {family: build_cases(family) for family in NATIVE_FAMILIES}
    _assert_cross_family_contract(corpus)
    source = corpus[family][f"UNIFIED.guided_json_quoted_bare_header_in_answer.{family}"]
    case = json.loads(json.dumps(source).replace("self literal", "get_weather literal"))
    corpus[family][f"UNIFIED.guided_json_quoted_bare_tool_header_in_answer.{family}"] = case
    with pytest.raises(AssertionError, match="missing tool-recipient boundary"):
        _assert_cross_family_contract(corpus)


@pytest.mark.parametrize("family", NATIVE_FAMILIES)
@pytest.mark.parametrize("mutation", [
    "tool_name", "argument_key", "reasoning_prose", "input_tool_name",
    "input_argument_key", "input_reasoning_prose", "marker_role", "missing_event",
    "named_tool", "starting_state", "output_mode", "input_value",
    "guided_array_value", "argument_marker_role", "input_call_order",
    "quoted_response_header", "quoted_response_payload", "missing_recovered_answer",
])
def test_cross_family_contract_rejects_single_family_mutations(family, mutation):
    if mutation in {"quoted_response_header", "quoted_response_payload", "missing_recovered_answer"} and family not in {"gpt_oss", "muse_glimmer", "kimi_k3"}:
        pytest.skip("this family has no applicable framed-response specimen")
    corpus = {family: build_cases(family) for family in NATIVE_FAMILIES}
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
        reason_open, reason_close, tool_open, _ = control_tokens(family)
        if family == "deepseek_v41":
            case["input"] = case["input"].replace("</think>", "</｜DSML｜ calls>")
        elif G.default_response_prefix(family):
            case["input"] = case["input"].replace(reason_close, tool_open)
        elif family == "gpt_oss":
            case["input"] = case["input"].replace(
                "<|channel|>analysis<|message|>",
                "<|channel|>final<|message|>",
                1,
            )
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
        prefix = G.default_response_prefix(family)
        calls = json.loads(case["input"].removeprefix(prefix))
        calls[0]["arguments"]["values"][-1] += 1
        case["input"] = prefix + json.dumps(calls)
    elif mutation == "argument_marker_role":
        case = corpus[family][f"UNIFIED.arg_marker_in_string.{family}"]
        value = case["golden"][0]["arguments"]["cmd"]
        wrong = f"git log {control_tokens(family)[1]} --oneline"
        case["input"] = case["input"].replace(value, wrong)
        case["golden"][0]["arguments"]["cmd"] = wrong
    elif mutation in {"quoted_response_header", "quoted_response_payload"}:
        scenario = "gpt_oss_quoted_message_header_literal"
        case = corpus[family][f"UNIFIED.{scenario}.{family}"]
        value = case["golden"][0]["arguments"]["q"]
        header = {"gpt_oss": "final", "muse_glimmer": "user", "kimi_k3": "response"}[family]
        wrong = value.replace(header, "wrong") if mutation == "quoted_response_header" else value.replace("literal", "changed")
        case["input"] = case["input"].replace(value, wrong)
        case["golden"][0]["arguments"]["q"] = wrong
        with pytest.raises(AssertionError, match="wrong marker role or payload"):
            _logical_events(scenario, family, case["golden"])
    elif mutation == "missing_recovered_answer":
        case = corpus[family][f"UNIFIED.gpt_oss_quoted_truncated_tool_then_final.{family}"]
        case["golden"] = []
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
    corpus = {family: build_cases(family) for family in NATIVE_FAMILIES}
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


@pytest.mark.parametrize("family", NATIVE_FAMILIES)
@pytest.mark.parametrize("mutation", ["echo_order", "closed_string", "fallback", "chunks"])
def test_malformed_recovery_contract_rejects_changed_stimulus(family, mutation):
    case = build_cases(family)[f"UNIFIED.malformed_json_then_two_valid_calls.{family}"]
    _assert_malformed_recovery(family, case)
    if mutation == "echo_order":
        case["input"] = case["input"].replace("Café", "OTHER").replace("é", "Café").replace("OTHER", "é")
        case["input_chunks"] = [case["input"]]
    elif mutation == "closed_string":
        case["input"] = case["input"].replace("unfinished", 'unfinished"')
        case["input_chunks"] = [case["input"]]
    elif mutation == "fallback":
        case["golden"][0]["arguments"] = {"value": "wrong"}
    else:
        case["input_chunks"] = [case["input"][:1], case["input"][1:]]
    with pytest.raises(AssertionError):
        _assert_malformed_recovery(family, case)


def test_schema_id_oracle_preserves_scalar_constraints_and_rejects_reference_scope():
    schema = {"$id": "https://example.com/count", "type": "integer"}
    assert matches_schema(42, schema)
    assert not matches_schema("42", schema)
    with pytest.raises(AssertionError):
        matches_schema(42, {**schema, "$ref": "#/$defs/Count"})


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("scenario", [
    "unused_reference_graph_parameter_types",
    "nullable_reference_alias_literals",
    "local_schema_id_preserves_type",
])
def test_second_pass_reference_goldens_satisfy_the_authored_schema(family, scenario):
    case = build_cases(family)[f"UNIFIED.{scenario}.{family}"]
    assert matches_schema(case["golden"][0]["arguments"], case["tools"][0]["parameters"])


def test_display_ranks_follow_descriptions_without_changing_capture_ids():
    scenarios = list(UNIFIED_TAX)
    identities = {scenario: numbered_id(scenario) for scenario in scenarios}
    labels = taxonomy.display_case_labels(reversed(scenarios))
    assert labels == taxonomy.display_case_labels(scenarios + scenarios)
    groups = defaultdict(list)
    for scenario in sorted(scenarios, key=taxonomy.display_sort_key):
        groups[taxonomy.display_tax(scenario)[0]].append(scenario)
    for members in groups.values():
        sections = defaultdict(list)
        for scenario in members:
            sections[taxonomy.display_section(scenario)[0]].append(scenario)
        for section in sections.values():
            assert [taxonomy.case_description(s) for s in section] == sorted(taxonomy.case_description(s) for s in section)
    compact_prefixes = defaultdict(list)
    for scenario in sorted(scenarios, key=taxonomy.display_sort_key):
        compact_prefixes[labels[scenario]["compact_prefix"]].append(scenario)
    for members in compact_prefixes.values():
        assert [labels[s]["display_number"] for s in members] == list(range(1, len(members) + 1))
    v4 = labels["guided_response_rejected_header_quote_ownership"]
    assert v4["label"] == "deepseek_v4-guided_response_rejected_header_quote_ownership"
    assert v4["compact_prefix"] == "ds4"
    v41 = labels["deepseek_v41_json_invocation_body"]
    assert v41["label"] == "deepseek_v41-json_invocation_body"
    assert v41["compact_prefix"] == "ds41"
    assert identities == {scenario: numbered_id(scenario) for scenario in scenarios}


@pytest.mark.parametrize("scenario", [
    scenario for scenario in UNIFIED_TAX
    if G.scenario_families(scenario) == frozenset({"gpt_oss"})
])
def test_gpt_oss_variants_use_shared_parents_or_explicit_header_grammar(scenario):
    fold = taxonomy.case_fold(scenario)
    if fold:
        parent, _ = fold
        assert parent != scenario
        assert len(G.scenario_families(parent)) > 1
        assert taxonomy.display_section(scenario) == taxonomy.display_section(parent)
    else:
        assert scenario in GPT_OSS_STANDALONE_HEADER_SCENARIOS
        assert taxonomy.display_tax(scenario)[0] == "gpt_oss"
    assert taxonomy.scenario_for_label(numbered_id(scenario), scenario=scenario) == scenario


@pytest.mark.parametrize(("scenario", "parent", "families"), [
    ("gpt_oss_tool_body_whitespace", "tool_only", {"gpt_oss", "kimi_k2", "kimi_k3"}),
    ("gpt_oss_quoted_truncated_tool_then_final", "tool_no_close", {"gpt_oss", "muse_glimmer", "muse_spark", "kimi_k3"}),
    ("gpt_oss_quoted_message_header_literal", "arg_marker_in_string", {"gpt_oss", "muse_glimmer", "muse_spark", "kimi_k3"}),
])
def test_retained_harmony_review_ids_cover_native_shared_variants(scenario, parent, families):
    assert G.scenario_families(scenario) == families
    assert taxonomy.case_fold(scenario)[0] == parent
    assert taxonomy.display_tax(scenario)[0] == taxonomy.display_tax(parent)[0]
    assert scenario not in GPT_OSS_STANDALONE_HEADER_SCENARIOS
    for family in families - {"gpt_oss"}:
        case = build_cases(family)[f"UNIFIED.{scenario}.{family}"]
        assert "Harmony" not in case["description"]
        assert case["init"] == {"starting_state": "None", "tool_output_mode": "Native", "named_tool": None}
        if parent == "tool_no_close":
            assert case["golden"] == [{"kind": "text", "text": "Answer"}]
            if family in {"muse_glimmer", "muse_spark"}:
                assert '<atem:parameter name="q">abc<|eom|>' in case["input"]
                assert "</atem:parameter>" not in case["input"]
            else:
                assert '{"q":"abc<|close|>json<|sep|>' in case["input"]
                assert '<|open|>response<|sep|>Answer<|close|>response<|sep|>' in case["input"]
                assert case["expect"]["dynamo"]["class"] == "LOSS"
        else:
            event = case["golden"][0]
            assert event["name"] == "lookup"
            assert matches_schema(event["arguments"], case["tools"][0]["parameters"])
            if parent == "tool_only":
                assert ' \n\t{"q":' in case["input"]
                assert event["arguments"] == {"q": "x"}
            else:
                payload = event["arguments"]["q"]
                assert payload in case["input"]
                assert "literal" in payload
                assert ("assistant to=user<|message|>" if family in {"muse_glimmer", "muse_spark"}
                        else '<|open|>response<|sep|>') in payload


def test_display_policy_has_no_dangling_groups_or_scenarios():
    assert set(taxonomy._DISPLAY_GROUP) <= set(UNIFIED_TAX)
    assert set(taxonomy._DISPLAY_GROUP.values()) <= set(UNIFIED_GROUP_LABEL)
    assert set(taxonomy.DISPLAY_PREFIXES) <= set(UNIFIED_GROUP_LABEL)
    for prefixes in taxonomy.DISPLAY_PREFIXES.values():
        assert len(prefixes) == 2
        assert all(re.fullmatch(r"[a-z0-9_]+", prefix) for prefix in prefixes)


def test_preserved_requests_and_goldens_are_independent_across_builds():
    case_id = "UNIFIED.tool_only.gpt_oss"
    expected = json.loads(json.dumps(build_cases("gpt_oss")[case_id]))
    changed = build_cases("gpt_oss")[case_id]
    changed["tools"].append({"name": "unrelated", "parameters": {}})
    changed["golden"][0]["arguments"]["city"] = "changed"
    changed["init"]["starting_state"] = "Reasoning"
    assert build_cases("gpt_oss")[case_id] == expected


def test_retired_harmony_eof_case_keeps_its_history_and_shared_contract():
    path = SRC.parents[2] / "conformance/fixtures-unified-v2/families/gpt_oss/inputs_and_golden.yaml"
    cases = yaml.safe_load(path.read_text())["cases"]
    retired = cases["gpt_oss_truncated_tool_at_eof_is_dropped"]
    parent = cases["tool_no_close"]
    assert retired["lifecycle"] == "retired"
    assert retired["display_id"] is None
    assert numbered_id("gpt_oss_truncated_tool_at_eof_is_dropped") in retired["historical_ids"]
    assert retired["golden"]["assembled"] == parent["golden"]["assembled"] == []
    for field in ("init", "tools", "finish_reason"):
        assert retired["request"][field] == parent["request"][field]
    for case in (retired, parent):
        raw = case["request"]["input"]
        body = raw.split("<|message|>", 1)[1]
        assert isinstance(json.loads(body), dict)
        assert not raw.endswith(control_tokens("gpt_oss")[3])
    assert not G.scenario_families("gpt_oss_truncated_tool_at_eof_is_dropped")
    assert "UNIFIED.gpt_oss_truncated_tool_at_eof_is_dropped.gpt_oss" not in G.build_cases("gpt_oss")
