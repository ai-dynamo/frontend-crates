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
from conformance.utils.tests.schema_oracle import _local_reference, matches_schema
from conformance.utils.tests.fixture_test_support import (
    assert_unique_behaviour, semantic_key, GPT58_NATIVE_REQUEST_DIGESTS, assert_gpt58_native_contract,
    assert_native_response_counterparts,
)

import pytest  # noqa: E402
import yaml  # noqa: E402

import gen_unified_golden as G  # noqa: E402
from schema_cases import CONFORMANCE_CASES, XML_COMPATIBILITY_CASES, schema_arguments  # noqa: E402
import unified_history  # noqa: E402
import unified_archives
from fixture_test_support import retained_source_bytes
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
from numeric_cases import NUMERIC_VARIANTS, OVERFLOW_SIBLING_VARIANTS, applicable as numeric_applicable, canonical_events, numeric_expected, unified_numeric_expected
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
    changed = deepcopy(case)
    changed["input_chunks"] = ["<call>", "x", "</call>"]
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
        "prefilled_response_guided_pending_invoke_header",
        "gemma4_guided_response_extra_body_opener",
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
        if family == "kimi_k3":
            extra.update({"kimi_k3_direct_typed_response_recovery", "kimi_k3_direct_json_response_recovery"})
        if family in {"beam", "gpt_oss", "muse_glimmer", "muse_spark"}:
            extra.add("kimi_k3_direct_typed_response_recovery")
        if family in {"beam", "gpt_oss", "kimi_k3"}:
            extra.update({"muse_quoted_reserved_eom_argument", "muse_quoted_reserved_eot_argument", "muse_quoted_reserved_start_argument"})
        if family in {"beam", "gpt_oss"}:
            extra.add("guided_json_quoted_bare_tool_header_in_answer")
        if family in {"deepseek_v4", "deepseek_v41", "kimi_k3", "minimax_m3"}:
            extra.add("prefilled_response_guided_closer_inside_invoke_quote")
        if G.invoke_header_prefix(family).endswith('="'):
            extra.add("guided_response_rejected_header_quote_ownership")
        if family in {"muse_glimmer", "muse_spark"}:
            extra.update({"guided_json_quoted_bare_header_after_payload_pr424", "guided_json_quoted_bare_header_in_answer_pr424"})
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
            if (scenario, family) in _GUIDED_HEADER_CONTRACTS:
                _assert_guided_header_case(family, scenario, case)
                continue
            if scenario.startswith("muse_quoted_reserved_"):
                marker = _family_value(scenario, family)[2:-2]
            elif family == "gpt_oss" and scenario == "prefilled_response_reasoning_markers_literal":
                marker = "<think>"
            elif scenario in native_scenarios:
                marker = G._NATIVE_QUOTED_CONTROL.get(G.unified_grammar_family(family), control_tokens(family)[2])
            elif scenario in {"kimi_k3_direct_typed_response_recovery", "kimi_k3_direct_json_response_recovery"}:
                marker = {
                    "kimi_k3": '<|open|>call tool="lookup"',
                    "beam": "<|start|>assistant to=lookup",
                    "gpt_oss": "<|start|>assistant<|channel|>commentary to=functions.lookup",
                    "muse_glimmer": '<atem:invoke name="lookup">',
                    "muse_spark": '<atem:invoke name="lookup">',
                }[family]
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
        "guided_json_quoted_bare_tool_header_in_answer": {"beam", "gpt_oss", "muse_glimmer", "muse_spark"},
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
    assert len(UNIFIED_TAX) == 260 + len(G.K3_RESPONSE_CASES) + len(CONFORMANCE_CASES) + len(PARALLEL_BOUNDARY_CASES) + len(V1_SPLIT_SCENARIOS) + len(G.DOTTED_NAME_SCENARIOS) + len(NUMERIC_VARIANTS) + len(OVERFLOW_SIBLING_VARIANTS) + 5
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


def _assert_reference_chain_coverage(corpus):
    scenario = "glm47_reference_type_intersection"
    reference = corpus["glm47"][f"UNIFIED.{scenario}.glm47"]
    expected = [{"kind": "tool_call", "name": "capture_payload",
                 "arguments": {"payload": '{"x":1}', "count": 42}}]
    assert taxonomy.display_section(scenario)[0] == "unified_g7_reference"
    for family in NATIVE_FAMILIES:
        case_id = f"UNIFIED.{scenario}.{family}"
        assert case_id in corpus[family], (family, "missing applicable reference-chain request")
        case = corpus[family][case_id]
        assert case["tools"] == reference["tools"]
        assert case["golden"] == expected
        _assert_input_carries_events(family, scenario, case)


def test_reference_chain_coverage_uses_each_native_grammar():
    _assert_reference_chain_coverage({family: build_cases(family) for family in NATIVE_FAMILIES})


def test_reference_chain_coverage_rejects_an_omitted_supported_family():
    corpus = {family: build_cases(family) for family in NATIVE_FAMILIES}
    del corpus["beam"]["UNIFIED.glm47_reference_type_intersection.beam"]
    with pytest.raises(AssertionError, match="missing applicable reference-chain"):
        _assert_reference_chain_coverage(corpus)


def _assert_embedded_parameterless_call_coverage(corpus):
    scenario = "glm47_parameterless_call_shape_inside_argument"
    reference = corpus["glm47"][f"UNIFIED.{scenario}.glm47"]
    assert reference["tools"] == G.unified_tools(), "original offered schema changed"
    assert taxonomy.display_section(scenario)[0] == "unified_g12"
    for family in NATIVE_FAMILIES:
        key = f"UNIFIED.{scenario}.{family}"
        assert key in corpus[family], (family, "missing applicable embedded-call request")
        case = corpus[family][key]
        assert case["init"] == reference["init"]
        assert case["tools"] == reference["tools"]
        assert len(case["golden"]) == 1
        event = case["golden"][0]
        assert event["kind"] == "tool_call" and event["name"] == "run"
        assert set(event["arguments"]) == {"cmd"}
        value = event["arguments"]["cmd"]
        assert value == _family_value(scenario, family), (family, "incidental payload changed")
        assert isinstance(value, str) and value.startswith("before ") and value.endswith(" after")
        assert "get_weather" in value
        _assert_input_carries_events(family, scenario, case)
        assert _native_input_calls(family, case["input"], case["tools"]) == case["golden"], (family, "nested argument became an extra call")


def test_embedded_parameterless_call_coverage_preserves_the_offered_schema():
    _assert_embedded_parameterless_call_coverage({family: build_cases(family) for family in NATIVE_FAMILIES})


def test_embedded_parameterless_call_coverage_rejects_a_missing_family():
    corpus = {family: build_cases(family) for family in NATIVE_FAMILIES}
    del corpus["beam"]["UNIFIED.glm47_parameterless_call_shape_inside_argument.beam"]
    with pytest.raises(AssertionError, match="missing applicable embedded-call"):
        _assert_embedded_parameterless_call_coverage(corpus)


def test_embedded_parameterless_call_coverage_rejects_incidental_payload_drift():
    corpus = {family: build_cases(family) for family in NATIVE_FAMILIES}
    case = corpus["beam"]["UNIFIED.glm47_parameterless_call_shape_inside_argument.beam"]
    case["input"] = case["input"].replace("before ", "before changed ")
    case["golden"][0]["arguments"]["cmd"] = case["golden"][0]["arguments"]["cmd"].replace("before ", "before changed ")
    with pytest.raises(AssertionError):
        _assert_embedded_parameterless_call_coverage(corpus)


def test_embedded_parameterless_call_coverage_rejects_coordinated_schema_loss():
    corpus = {family: build_cases(family) for family in NATIVE_FAMILIES}
    for family in NATIVE_FAMILIES:
        corpus[family][f"UNIFIED.glm47_parameterless_call_shape_inside_argument.{family}"]["tools"] = []
    with pytest.raises(AssertionError, match="original offered schema changed"):
        _assert_embedded_parameterless_call_coverage(corpus)


@pytest.mark.parametrize("family", ["beam", "gpt_oss", "kimi_k3"])
@pytest.mark.parametrize("boundary", ["eom", "eot", "start"])
def test_quoted_message_boundaries_remain_native_argument_data(family, boundary):
    scenario = f"muse_quoted_reserved_{boundary}_argument"
    case = build_cases(family)[f"UNIFIED.{scenario}.{family}"]
    original = build_cases("muse_glimmer")[f"UNIFIED.{scenario}.muse_glimmer"]
    assert case["init"] == original["init"] == {
        "starting_state": "Response", "tool_output_mode": "Native", "named_tool": None,
    }
    assert case["tools"] == original["tools"]
    assert case["golden"] == [{"kind": "tool_call", "name": "get_weather", "arguments": {"city": _family_value(scenario, family)}}]
    assert taxonomy.display_section(scenario)[0] == "unified_g12"
    _assert_input_carries_events(family, scenario, case)


@pytest.mark.parametrize("parent", ["reason_markup_in_arg", "reason_markup_in_arg_with_text"])
def test_consistent_marker_prose_preserves_the_historical_request(parent):
    path = UTILS.parent / "fixtures-unified-v2/families/gpt_oss/inputs_and_golden.yaml"
    records = yaml.safe_load(path.read_text())["cases"]
    old = records[parent]
    active = records[parent + "_consistent"]
    assert old["lifecycle"] == "retired" and old["historical_ids"]
    assert active["lifecycle"] == "active"
    assert old["request"]["input"].count("secret") == 1
    assert active["request"]["input"].count("reconsider") == 1
    assert active["request"]["input"].replace("reconsider", "secret") == old["request"]["input"]
    original_event = next(event for event in old["golden"]["assembled"] if event["kind"] == "tool_call")
    active_event = next(event for event in active["golden"]["assembled"] if event["kind"] == "tool_call")
    assert active_event["arguments"]["note"].replace("reconsider", "secret") == original_event["arguments"]["note"]
    assert taxonomy.coverage_case(parent + "_consistent") == parent
    assert f"UNIFIED.{parent}.gpt_oss" not in build_cases("gpt_oss")


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
    sparse = any("schema_version" not in yaml.load(retained_source_bytes(root, row), Loader=unified_history.StrictLoader)
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
                source = retained_source_bytes(root, resource)
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
    if scenario == "kimi_k3_json_marker_then_long_field":
        if family == "kimi_k3":
            return "<|open|>response<|sep|>literal<|close|>response<|sep|>"
        if family in {"muse_glimmer", "muse_spark"}:
            return "<|start|>assistant to=user<|message|>literal<|eom|>"
        assert family in {"beam", "gpt_oss"}, (scenario, family)
        return "<|start|>assistant<|channel|>final<|message|>literal<|return|>"
    if scenario == "kimi_k3_typed_json_marker_payload":
        return G.NATIVE_ARGUMENT_BOUNDARIES[family]
    if scenario == "glm47_parameterless_call_shape_inside_argument":
        return "before " + control_tokens(family)[3] + G.r_tool_arguments(family, "get_weather", {}, 0) + " after"
    if scenario.startswith("muse_quoted_reserved_"):
        boundary = scenario.removeprefix("muse_quoted_reserved_").removesuffix("_argument")
        message_markers = {"eom": "<|eom|>", "eot": "<|eot|>", "start": "<|start|>"}
        if family in {"beam", "gpt_oss"}:
            message_markers.update(eom="<|end|>", eot="<|return|>")
        elif family == "kimi_k3":
            message_markers.update(
                eom="<|close|>message<|sep|>", eot="<|end_of_msg|>",
                start='<|open|>message role="assistant"<|sep|>',
            )
        return f'a"{message_markers[boundary]}"b'
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
            "minimax_m3": "]<]minimax[>[</tool_call>",
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
            'beam': '<|start|><|channel|><|message|><|end|><|return|><|constrain|><rfl:parameter name="q"></rfl:parameter><|call|><rfl:parameter</rfl:parameter',
            'muse_glimmer': '<|start|><|message|><|eom|><atem:function_calls><atem:invoke name="lookup"><atem:parameter name="q"></atem:parameter></atem:invoke></atem:function_calls><atem:invoke<atem:parameter',
            'muse_spark': '<|start|><|message|><|eom|><atem:function_calls><atem:invoke name="lookup"><atem:parameter name="q"></atem:parameter></atem:invoke></atem:function_calls><atem:invoke<atem:parameter',
            "deepseek_v4": '<think></think><｜DSML｜tool_calls><｜DSML｜invoke name="lookup"><｜DSML｜parameter name="q" string="true"></｜DSML｜parameter></｜DSML｜invoke></｜DSML｜tool_calls><｜DSML｜</｜DSML｜',
            "deepseek_v41": '<think></think><｜DSML｜ calls><｜DSML｜ invoke name="lookup"><｜DSML｜ parameter name="q" string="true"></｜DSML｜ parameter></｜DSML｜ invoke></｜DSML｜ calls><｜DSML｜</｜DSML｜',
            "gpt_oss": "<|start|><|channel|><|constrain|><|message|><|end|><|return|><|call|>",
            "kimi_k2": "<think></think><|tool_calls_section_begin|><|tool_call_begin|><|tool_call_argument_begin|><|tool_call_end|><|tool_calls_section_end|>",
            "kimi_k3": "<|open|><|sep|><|close|><|end_of_msg|>",
        }[family]
    if scenario == "gpt_oss_quoted_message_header_literal":
        return {
            "beam": '<|call|><|start|>assistant<|channel|>final<|message|>literal<|return|>',
            "gpt_oss": "<|call|><|start|>assistant<|channel|>final<|message|>literal<|return|>",
            "muse_glimmer": "<|eom|><|start|>assistant to=user<|message|>literal<|eom|>",
            "muse_spark": "<|eom|><|start|>assistant to=user<|message|>literal<|eom|>",
            "kimi_k3": "<|close|>call<|sep|><|open|>response<|sep|>literal<|close|>response<|sep|>",
        }[family]
    quoted = {
        "prefilled_response_guided_closer_inside_invoke_quote": "self",
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
                elif isinstance(value, dict) and value.get("inner") == family_value:
                    event["arguments"][key]["inner"] = "FAMILY_MARKER_VALUE"
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


def _fixture_property_schema(parameters, key, root=None, references=()):
    root = parameters if root is None else root
    if key in parameters.get("properties", {}):
        return parameters["properties"][key]
    if "$ref" in parameters:
        reference = parameters["$ref"]
        assert reference not in references, ("cyclic fixture root reference", reference)
        return _fixture_property_schema(
            _local_reference(root, reference), key, root, references + (reference,)
        )
    for keyword in ("allOf", "anyOf", "oneOf"):
        found = [_fixture_property_schema(branch, key, root, references) for branch in parameters.get(keyword, [])]
        found = [schema for schema in found if schema]
        if found:
            return found[0] if len(found) == 1 else {keyword: found}
    return {}


def _m3_input_calls(raw, tools, *, exact_numbers=False):
    """Read complete authored tags independently of the Rust recovery scanner.

    This intentionally handles fixture grammar only. A malformed or unfinished
    node is not treated as evidence for a complete expected call.
    """
    ns = "]<]minimax[>["
    header = re.compile(re.escape(ns) + r'<invoke name="([^"]+)">')
    opener = re.compile(re.escape(ns) + r'<([^<>/\s]+)>')

    def node(at, schema, root):
        match = opener.match(raw, at)
        if match is None:
            return None
        key = match[1]
        at = match.end()
        close = ns + f"</{key}>"
        declared = schema.get("type")
        string = declared == "string" or isinstance(declared, list) and "string" in declared
        if not string and opener.match(raw, at):
            values = {}
            items = []
            while not raw.startswith(close, at):
                child_match = opener.match(raw, at)
                if child_match is None:
                    return None
                child_schema = schema.get("items", {}) if declared == "array" else _fixture_property_schema(schema, child_match[1])
                child = node(at, child_schema, root)
                if child is None:
                    return None
                child_key, value, at = child
                values[child_key] = value
                items.append(value)
            value = items if declared == "array" else values
        else:
            end = raw.find(close, at)
            if end < 0:
                return None
            value = raw[at:end]
            at = end
            if declared == "array" and not value.startswith("["):
                schema = schema.get("items", {})
            if value == "null" and matches_schema(None, schema, root):
                value = None
            elif not matches_schema(value, schema, root):
                try:
                    decoded = json.loads(value, parse_float=Decimal if exact_numbers else float)
                except json.JSONDecodeError:
                    pass
                else:
                    if matches_schema(decoded, schema, root):
                        value = decoded
        return key, value, at + len(close)

    calls = []
    cursor = 0
    while match := header.search(raw, cursor):
        name = match[1]
        root = next((tool.get("parameters", {}) for tool in tools if tool["name"] == name), {})
        arguments = {}
        cursor = match.end()
        while parameter := opener.match(raw, cursor):
            parsed = node(cursor, _fixture_property_schema(root, parameter[1]), root)
            if parsed is None:
                break
            key, value, cursor = parsed
            if key in arguments:
                previous = arguments[key]
                arguments[key] = (previous if isinstance(previous, list) else [previous]) + [value]
            else:
                arguments[key] = value
        calls.append({"kind": "tool_call", "name": name, "arguments": arguments})
    return calls


def _argument_value_end(raw, value_start, closer, *, json_owned=True, parameters=None, key=None):
    # Independently decode a JSON value before looking for its XML closer;
    # otherwise this fixture validator repeats the parser's quoted-closer bug.
    minimum = value_start
    at = value_start + len(raw[value_start:]) - len(raw[value_start:].lstrip())
    if json_owned and raw[at:at + 1] in {"{", "[", '"'}:
        try:
            value, minimum = json.JSONDecoder().raw_decode(raw, at)
        except json.JSONDecodeError:
            pass
        else:
            if parameters is not None:
                schema = _fixture_property_schema(parameters, key)
                if (schema.get("type") == "custom"
                        or not matches_schema(value, schema, parameters)
                        or matches_schema(raw[at:minimum], schema, parameters)):
                    minimum = value_start
    return raw.find(closer, minimum)


def _k3_owned_call(raw, start, call_header):
    """Find the outer close and its literal nested call spans independently."""
    markers = re.compile(
        rf'(?P<call>{call_header})|(?P<close><\|close\|>\s*call\s*<\|sep\|>)'
        r'|(?P<argument><\|open\|>\s*argument key="[^"]+" type="(?P<kind>[^"]+)"\s*<\|sep\|>)'
        r'|(?P<argument_close><\|close\|>\s*argument\s*<\|sep\|>)'
        r'|(?P<json><\|open\|>\s*json\b[^<]*<\|sep\|>)',
    )
    decoder = json.JSONDecoder()
    nested_arguments = []
    literal_calls = []
    cursor = start
    while marker := markers.search(raw, cursor):
        cursor = marker.end()
        if marker["argument"] is not None and nested_arguments:
            nested_arguments[-1][1] += 1
        if marker["json"] is not None or (marker["argument"] is not None and marker["kind"] != "string"):
            value_start = cursor + len(raw[cursor:]) - len(raw[cursor:].lstrip())
            try:
                _, cursor = decoder.raw_decode(raw, value_start)
            except json.JSONDecodeError:
                pass
        elif marker["call"] is not None:
            nested_arguments.append([marker.start(), 0])
        elif marker["argument_close"] is not None and nested_arguments:
            if nested_arguments[-1][1]:
                nested_arguments[-1][1] -= 1
            else:
                # An unclosed literal call header cannot borrow the outer
                # argument's real close and turn its call close into recovery.
                literal_calls.extend((at, marker.start()) for at, _ in nested_arguments)
                nested_arguments.clear()
        elif marker["close"] is not None:
            if nested_arguments:
                at, _ = nested_arguments.pop()
                literal_calls.append((at, cursor))
            else:
                return cursor, literal_calls
    return None, []


def _native_input_calls(family, raw, tools=(), *, exact_numbers=False):
    """Read authored complete argument fields, not runtime recovery decisions.

    This fixture-only projection ignores invoke EOF policy: a syntactically present
    body is not evidence that the parser must dispatch it. The caller keeps DSML's
    empty-output EOF contract separate.
    """
    if family == "minimax_m3":
        return _m3_input_calls(raw, tools, exact_numbers=exact_numbers)
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
            parameters = next((tool.get("parameters", {}) for tool in tools if tool["name"] == name), {})
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
                value_end = _argument_value_end(raw, value_start, "</arg_value>", parameters=parameters, key=key)
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
        parameter = re.compile(r'\s*<rfl:parameter name="([^"]+)">', re.S)
        calls = []
        owned_until = 0
        for match in header.finditer(raw):
            if match.start() < owned_until:
                continue
            at = match.end()
            arguments = {}
            while True:
                item = parameter.match(raw, at)
                if item is None:
                    break
                start = item.end()
                # Native raw slots use newline framing when present; compact
                # legacy slots retain their first-closer rule even for JSON-looking text.
                framed_close = raw.find("\n</rfl:parameter>", start + 1) if raw.startswith("\n", start) else -1
                follow = raw.find('\n<rfl:parameter name="', start + 1)
                if follow >= 0 and (framed_close < 0 or follow < framed_close):
                    arguments[item[1]] = raw[start:follow + 1].removeprefix("\n").removesuffix("\n")
                    at = follow + 1
                    continue
                end = framed_close + 1 if framed_close >= 0 else raw.find("</rfl:parameter>", start)
                if end < 0:
                    break
                arguments[item[1]] = raw[item.end():end].removeprefix("\n").removesuffix("\n")
                at = end + len("</rfl:parameter>")
            owned_until = at
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

    if family == "gemma4":
        calls = []
        owned_until = 0
        for header in re.finditer(r'call:([\w.-]+)\{', raw):
            if header.start() < owned_until:
                continue
            # Parse the complete value before looking for another call header;
            # a quoted argument may contain both a closer and a complete call.
            arguments, owned_until = _parse_gemma_value(raw, header.end() - 1, exact_numbers=exact_numbers)
            calls.append({"kind": "tool_call", "name": header[1], "arguments": arguments})
        return calls

    headers = {
        "deepseek_v4": r'<｜DSML｜invoke name="([^"]+)">',
        "deepseek_v41": r'<｜DSML｜ invoke name="([^"]+)">',
        "qwen3": r'<function=([^>]+)>',
        "muse_glimmer": r'<atem:invoke name="([^"]+)">',
        "muse_spark": r'<atem:invoke name="([^"]+)">',
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
                    end = _argument_value_end(raw, parameter.start(3), f"</｜DSML｜{gap}parameter>", json_owned=is_string == "false")
                    assert end >= 0
                    value = raw[parameter.start(3):end]
                    arguments[key] = value if is_string == "true" else json.loads(
                        value, parse_float=Decimal if exact_numbers else float
                    )
                    body_cursor = end + len(f"</｜DSML｜{gap}parameter>")
                    continue
                cursor = len(raw) if body_end < 0 else body_end + len(invocation_end)
                break
            calls.append({"kind": "tool_call", "name": name, "arguments": arguments})
        return calls

    k3_parameter = re.compile(
        r'<\|open\|>\s*argument key="([^"]+)" type="([^"]+)"\s*<\|sep\|>(.*?)'
        r'<\|close\|>\s*argument\s*<\|sep\|>'
        r'(?=\s*(?:<\|open\|>\s*argument key="[^"]+" type="[^"]+"\s*<\|sep\|>|<\|close\|>\s*(?:call|tools)\s*<\|sep\|>|$))', re.S,
    )
    k3_response_parameter = re.compile(
        r'<\|open\|>\s*argument key="([^"]+)" type="([^"]+)"\s*<\|sep\|>(.*?)'
        r'<\|close\|>\s*argument\s*<\|sep\|>(?=\s*<\|open\|>\s*response\s*<\|sep\|>)',
        re.S,
    )
    qwen_parameter = re.compile(r'<parameter=([^>]+)>')
    found = list(re.finditer(headers[family], raw))
    calls = []
    owned_until = 0
    for index, match in enumerate(found):
        if match.start() < owned_until:
            continue
        body = raw[match.end():found[index + 1].start() if index + 1 < len(found) else len(raw)]
        parameters = next((tool.get("parameters", {}) for tool in tools if tool["name"] == match[1]), {})
        k3_arguments = []
        if family in {"qwen3", "kimi_k3"}:
            parameter_pattern = qwen_parameter if family == "qwen3" else k3_parameter
            closer = r'</function>' if family == "qwen3" else r'<\|close\|>\s*call\s*<\|sep\|>'
            tail = raw[match.end():]
            cursor = 0
            owned_call_end, literal_calls = _k3_owned_call(raw, match.end(), headers[family]) if family == "kimi_k3" else (None, [])
            parameter_source = tail
            if literal_calls:
                # Ignore nested syntax only while locating the owning field's
                # close. Argument values below always slice the original input.
                locator = list(tail)
                for start, end in literal_calls:
                    locator[start - match.end():end - match.end()] = [" "] * (end - start)
                parameter_source = "".join(locator)
            json_header = re.match(r'\s*<\|open\|>\s*json\b[^<]*<\|sep\|>\s*', tail) if family == "kimi_k3" else None
            if json_header is not None:
                decoder = json.JSONDecoder(parse_float=Decimal if exact_numbers else float)
                try:
                    _, end = decoder.raw_decode(tail, json_header.end())
                except json.JSONDecodeError:
                    pass
                else:
                    body = tail[:end]
                    owned_until = match.end() + end
                    cursor = end
            while True:
                if json_header is not None and owned_until > match.end():
                    break
                close = re.search(closer, tail[cursor:])
                parameter = parameter_pattern.search(parameter_source, cursor)
                next_header = re.search(headers[family], tail[cursor:])
                if family == "kimi_k3" and owned_call_end is None:
                    recovered = k3_response_parameter.search(tail, cursor)
                    if parameter is None:
                        parameter = recovered
                    elif next_header is not None:
                        next_start = cursor + next_header.start()
                        # A later call cannot lend its field close to a completed
                        # argument followed by response recovery. Complete outer
                        # envelopes keep response and nested headers as payload.
                        if (parameter.start() < next_start < parameter.end()
                                and recovered is not None
                                and recovered.start() == parameter.start()
                                and recovered.end() <= next_start):
                            parameter = recovered
                if (next_header is not None
                        and (parameter is None or cursor + next_header.start() < parameter.start())
                        and (close is None or next_header.start() < close.start())):
                    end = cursor + next_header.start()
                    body = tail[:end]
                    owned_until = match.end() + end
                    break
                if parameter is None or (close is not None and cursor + close.start() < parameter.start()):
                    end = cursor + close.start() if close is not None else len(tail)
                    body = tail[:end]
                    owned_until = match.end() + (cursor + close.end() if close is not None else len(tail))
                    break
                if family == "qwen3":
                    end = _argument_value_end(tail, parameter.end(), "</parameter>", parameters=parameters, key=parameter[1])
                    if end < 0:
                        body = tail
                        owned_until = len(raw)
                        break
                    cursor = end + len("</parameter>")
                else:
                    k3_arguments.append((parameter[1], parameter[2], tail[parameter.start(3):parameter.end(3)]))
                    cursor = parameter.end()
        # Muse parameters own nested invocation-shaped text. The independent
        # fixture projection must not clip an outer argument at an inner header.
        if family in {"muse_glimmer", "muse_spark"}:
            tail = raw[match.end():]
            cursor = 0
            while True:
                close = tail.find("</atem:invoke>", cursor)
                parameter = re.search(r'<atem:parameter name="[^"]+">', tail[cursor:])
                next_header = re.search(headers[family], tail[cursor:])
                if (next_header is not None
                        and (parameter is None or next_header.start() < parameter.start())
                        and (close < 0 or cursor + next_header.start() < close)):
                    end = cursor + next_header.start()
                    body = tail[:end]
                    owned_until = match.end() + end
                    break
                if parameter is None or (close >= 0 and close < cursor + parameter.end()):
                    body = tail[:close] if close >= 0 else tail
                    owned_until = match.end() + (close + len("</atem:invoke>") if close >= 0 else len(tail))
                    break
                value_start = cursor + parameter.end()
                end = _argument_value_end(tail, value_start, "</atem:parameter>")
                if end < 0:
                    body = tail
                    break
                cursor = end + len("</atem:parameter>")
        arguments = {}
        if G.unified_grammar_family(family) == "qwen3":
            # The generator frames values with one newline; payload whitespace is data.
            cursor = 0
            while parameter := qwen_parameter.search(body, cursor):
                end = _argument_value_end(body, parameter.end(), "</parameter>", parameters=parameters, key=parameter[1])
                if end < 0:
                    break
                arguments[parameter[1]] = body[parameter.end():end].removeprefix("\n").removesuffix("\n")
                cursor = end + len("</parameter>")
        elif family in {"muse_glimmer", "muse_spark"}:
            cursor = 0
            while parameter := re.search(r'<atem:parameter name="([^"]+)">', body[cursor:]):
                key = parameter[1]
                value_start = cursor + parameter.end()
                end = _argument_value_end(body, value_start, "</atem:parameter>")
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
        elif family == "kimi_k2":
            start = match.end() + len(raw[match.end():]) - len(raw[match.end():].lstrip(" \t\r\n"))
            decoder = json.JSONDecoder(parse_float=Decimal if exact_numbers else float)
            try:
                arguments, owned_until = decoder.raw_decode(raw, start)
            except json.JSONDecodeError:
                # Kimi K2 preserves a named malformed JSON call as empty arguments.
                arguments = {}
        else:
            for key, kind, value in k3_arguments:
                arguments[key] = value if kind == "string" else json.loads(
                    value, parse_float=Decimal if exact_numbers else float
                )
            if not arguments and re.match(r'<\|open\|>\s*json ', body):
                values = _json_values(body, exact_numbers=exact_numbers)
                if values:
                    arguments = values[0]
        calls.append({"kind": "tool_call", "name": match[1], "arguments": arguments})
    return calls


@pytest.mark.parametrize("scenario", [
    "kimi_k3_typed_response_marker_payload",
    "kimi_k3_json_missing_close_literal_response",
    "kimi_k3_json_marker_then_long_field",
    "kimi_k3_typed_response_then_call",
    "kimi_k3_typed_response_argument_literal",
    "kimi_k3_direct_typed_response_recovery",
    "kimi_k3_direct_json_response_recovery",
    "kimi_k3_typed_json_marker_payload",
    "glm47_parameterless_call_shape_inside_argument",
    "arg_marker_in_string",
])
def test_kimi_k3_fixture_projection_preserves_response_argument_ownership(scenario):
    case = build_cases("kimi_k3")[f"UNIFIED.{scenario}.kimi_k3"]
    expected = [event for event in case["golden"] if event["kind"] == "tool_call"]
    assert _native_input_calls("kimi_k3", case["input"], case["tools"]) == expected


@pytest.mark.parametrize("family", ["muse_glimmer", "muse_spark"])
@pytest.mark.parametrize("scenario", [
    "kimi_k3_typed_response_then_call", "glm47_parameterless_call_shape_inside_argument",
])
def test_muse_fixture_projection_preserves_field_and_recovery_call_ownership(family, scenario):
    case = build_cases(family)[f"UNIFIED.{scenario}.{family}"]
    if scenario == "kimi_k3_typed_response_then_call":
        # Closed fields remain input evidence even when Muse drops an invoke
        # without its required close. Projection does not decide dispatch.
        expected = [
            {"kind": "tool_call", "name": "lookup", "arguments": {"q": "first"}},
            {"kind": "tool_call", "name": "next", "arguments": {"n": 7}},
        ]
    else:
        expected = case["golden"]
    assert _native_input_calls(family, case["input"], case["tools"]) == expected


@pytest.mark.parametrize("nested_close", ["", "<|close|>call<|sep|>"])
@pytest.mark.parametrize("nested_body", ["", '<|open|>argument key="n" type="integer"<|sep|>7<|close|>argument<|sep|>'])
def test_kimi_k3_fixture_projection_keeps_response_and_nested_call_inside_owned_string(nested_close, nested_body):
    payload = (
        'before<|close|>argument<|sep|><|open|>response<|sep|>literal<|close|>response<|sep|>'
        '<|open|>call tool="fake" index="9"<|sep|>' + nested_body + nested_close + 'after'
    )
    raw = (
        '<|open|>call tool="lookup" index="1"<|sep|>'
        '<|open|>argument key="q" type="string"<|sep|>' + payload
        + '<|close|>argument<|sep|><|close|>call<|sep|>'
    )
    assert _native_input_calls("kimi_k3", raw) == [
        {"kind": "tool_call", "name": "lookup", "arguments": {"q": payload}},
    ]


def test_kimi_k3_fixture_projection_preserves_exact_numbers_before_response_recovery():
    value = "0.12345678901234567890123456789"
    raw = (
        '<|open|>call tool="lookup" index="1"<|sep|>'
        '<|open|>argument key="q" type="string"<|sep|>first<|close|>argument<|sep|>'
        '<|open|>argument key="n" type="number"<|sep|>' + value
        + '<|close|>argument<|sep|><|open|>response<|sep|>recovered<|close|>response<|sep|>'
        '<|open|>tools<|sep|><|open|>call tool="next" index="2"<|sep|>'
        '<|open|>argument key="n" type="integer"<|sep|>7<|close|>argument<|sep|>'
        '<|close|>call<|sep|><|close|>tools<|sep|>'
    )
    assert _native_input_calls("kimi_k3", raw, exact_numbers=True) == [
        {"kind": "tool_call", "name": "lookup", "arguments": {"q": "first", "n": Decimal(value)}},
        {"kind": "tool_call", "name": "next", "arguments": {"n": 7}},
    ]


@pytest.mark.parametrize("value", ["first", "7"])
def test_kimi_k3_fixture_projection_detects_payload_changes_around_response_recovery(value):
    scenario = "kimi_k3_typed_response_then_call"
    case = deepcopy(build_cases("kimi_k3")[f"UNIFIED.{scenario}.kimi_k3"])
    case["input"] = case["input"].replace(value, "changed" if value == "first" else "8", 1)
    with pytest.raises(AssertionError, match="input call differs from golden"):
        _assert_input_carries_events("kimi_k3", scenario, case)


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


@pytest.mark.parametrize("framed", [False, True])
def test_beam_fixture_projection_keeps_raw_slot_framing(framed):
    value = '"a</rfl:parameter>b"'
    body = f"\n{value}\n" if framed else value
    raw = ('<|start|>assistant to=get_weather<|channel|>analysis <|constrain|>xml<|message|>'
           + '<rfl:parameter name="city">' + body + '</rfl:parameter><|call|>')
    assert _native_input_calls("beam", raw) == [{
        "kind": "tool_call", "name": "get_weather", "arguments": {"city": value if framed else '"a'},
    }]


@pytest.mark.parametrize("family,gap", [("deepseek_v4", ""), ("deepseek_v41", " ")])
def test_dsml_raw_string_field_does_not_acquire_json_ownership(family, gap):
    closer = f"</｜DSML｜{gap}parameter>"
    raw = (f'<｜DSML｜{gap}invoke name="get_weather"><｜DSML｜{gap}parameter name="city" string="true">'
           + '"a' + closer + 'b"' + closer + f'</｜DSML｜{gap}invoke>')
    assert _native_input_calls(family, raw) == [{
        "kind": "tool_call", "name": "get_weather", "arguments": {"city": '"a'},
    }]


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
        "minimax_m3": ']<]minimax[>[<invoke name="echo">',
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
        "minimax_m3": "]<]minimax[>[</value>]<]minimax[>[</invoke>",
    }
    assert closers[family] in prefix
    echoes = [{"kind": "tool_call", "name": "echo", "arguments": {"value": value}} for value in ("é", "Café")]
    assert _native_input_calls(family, suffix) == echoes
    bad = []
    if family == "kimi_k2":
        bad = [{"kind": "tool_call", "name": "bad", "arguments": {}}]
    elif family in {"qwen3", "glm47", "deepseek_v4", "muse_glimmer", "gpt_oss", "muse_spark", "beam", "minimax_m3"}:
        bad = [{"kind": "tool_call", "name": "bad", "arguments": {"value": '{"x":"unfinished'}}]
    assert case["golden"] == bad + echoes
    if family != "gpt_oss":
        assert case["tools"] == [
            {"name": "bad", "parameters": {"type": "object", "properties": {"value": {"type": "object"}}}},
            {"name": "echo", "parameters": {"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"]}},
        ]
    return echoes


_GUIDED_NATIVE_RECOVERY_SCENARIOS = (
    "guided_json_native_markup_only",
    "guided_native_after_prefilled_reasoning_required",
    "guided_native_after_prefilled_reasoning_named",
    "kimi_k3_guided_native_wrapper",
)


def _assert_guided_native_recovery(family, scenario, case):
    """Pin the native stimulus and the deliberate fallback policy difference."""
    assert scenario in _GUIDED_NATIVE_RECOVERY_SCENARIOS
    if scenario == "kimi_k3_guided_native_wrapper":
        assert case["init"] == {"starting_state": "None", "tool_output_mode": "GuidedJson", "named_tool": None}
        assert case["tools"] == G.unified_tools()
        prefix = G.default_response_prefix(family)
        assert case["input"] == prefix + G.r_tool(family, "ignored", "quoted", "literal", 0) + G.GUIDED_ONE_CALL, (family, scenario, "native-recovery wrapper")
        assert _native_input_calls(family, case["input"], case["tools"]) == [
            {"kind": "tool_call", "name": "ignored", "arguments": {"quoted": "literal"}},
        ]
        expected = ([{"kind": "tool_call", "name": "ignored", "arguments": {"quoted": "literal"}},
                     {"kind": "text", "text": G.GUIDED_ONE_CALL}]
                    if family == "minimax_m3" else
                    [{"kind": "tool_call", "name": "get_weather", "arguments": {"city": "Paris"}}])
        assert case["golden"] == expected, (family, scenario, "native-recovery output contract")
        assert case["finish_reason"] == "stop"
        return []
    prefilled = scenario != "guided_json_native_markup_only"
    assert family == "minimax_m3" or prefilled
    named = scenario == "guided_native_after_prefilled_reasoning_named"
    call = {"kind": "tool_call", "name": "get_weather", "arguments": {"city": "Paris"}}
    reasoning = [{"kind": "reasoning", "text": "Check."}] if prefilled else []
    assert case["init"] == {
        "starting_state": "Reasoning" if prefilled else "None",
        "tool_output_mode": "GuidedJson",
        "named_tool": "get_weather" if named else None,
    }, (family, scenario, "native-recovery request contract")
    prefix = "Check." + control_tokens(family)[1] if prefilled else G.default_response_prefix(family)
    assert case["input"] == prefix + G.r_tool(family, "get_weather", "city", "Paris", 0), (
        family, scenario, "native-recovery input framing",
    )
    # Check the semantic call independently of the generator's golden output.
    assert _native_input_calls(family, case["input"]) == [call], (
        family, scenario, "native-recovery input contract",
    )
    expected = reasoning + ([call] if family == "minimax_m3" else [])
    assert case["golden"] == expected, (family, scenario, "native-recovery output contract")
    assert case["finish_reason"] == "stop", (family, scenario, "native-recovery backend finish")
    return reasoning


@pytest.mark.parametrize("scenario", _GUIDED_NATIVE_RECOVERY_SCENARIOS[1:])
def test_guided_prefilled_native_controls_cover_all_native_families(scenario):
    assert set(G.scenario_families(scenario)) & set(NATIVE_FAMILIES) == set(NATIVE_FAMILIES)
    for family in NATIVE_FAMILIES:
        _assert_guided_native_recovery(family, scenario, build_cases(family)[f"UNIFIED.{scenario}.{family}"])
    changed = build_cases("qwen3")[f"UNIFIED.{scenario}.qwen3"]
    changed["golden"].append({"kind": "tool_call", "name": "get_weather", "arguments": {"city": "Paris"}})
    with pytest.raises(AssertionError, match="native-recovery output contract"):
        _assert_guided_native_recovery("qwen3", scenario, changed)


@pytest.mark.parametrize("scenario", _GUIDED_NATIVE_RECOVERY_SCENARIOS[1:])
def test_guided_prefilled_native_controls_reject_unjustified_m3_only_scope(scenario, monkeypatch):
    corpus = {family: build_cases(family) for family in NATIVE_FAMILIES}
    for family, cases in corpus.items():
        if family != "minimax_m3":
            del cases[f"UNIFIED.{scenario}.{family}"]
    original = G.scenario_families
    monkeypatch.setattr(G, "scenario_families", lambda name: frozenset({"minimax_m3"}) if name == scenario else original(name))
    with pytest.raises(AssertionError, match="single-family generic cases"):
        _assert_cross_family_contract(corpus)


def test_prompt_consumed_assistant_header_has_native_sibling_coverage():
    scenario = "gpt_oss_implicit_recipient_first_header"
    assert G.scenario_families(scenario) == {"beam", "gpt_oss", "muse_glimmer", "muse_spark"}
    assert taxonomy.display_tax(scenario)[0] == 1
    for family in G.scenario_families(scenario):
        case = build_cases(family)[f"UNIFIED.{scenario}.{family}"]
        assert case["input"].startswith(" to=")
        assert case["tools"] == []
        assert case["golden"] == [{"kind": "tool_call", "name": "lookup", "arguments": {"q": "x"}}]


def _assert_input_carries_events(family: str, scenario: str, case: dict) -> None:
    source_family = family
    default_reasoning = bool(G.default_response_prefix(family))
    family = G.unified_grammar_family(family)
    if scenario == "malformed_json_then_two_valid_calls":
        _assert_malformed_recovery(family, case)
        return
    if scenario.startswith("kimi_native_"):
        _assert_kimi_native_golden(source_family, scenario, case)
        return
    if (scenario, source_family) in _MUSE_NATIVE_CONTRACTS:
        _assert_muse_native_case(source_family, scenario, case)
        return
    raw = case["input"]
    if scenario == "gpt_oss_implicit_recipient_first_header":
        assert family in {"beam", "gpt_oss", "muse_glimmer", "muse_spark"}
        assert raw.startswith(" to=")
        # The prompt owns the assistant prefix; restore that known context only
        # for the independent wire decoder, never for the executed request.
        raw = "<|start|>assistant" + raw
    tools = [event for event in case["golden"] if event["kind"] == "tool_call"]
    if tools:
        if (case["init"]["tool_output_mode"] == "Native"
                or (family == "minimax_m3" and scenario in _GUIDED_NATIVE_RECOVERY_SCENARIOS)):
            candidates = _native_input_calls(family, raw, case.get("tools", []), exact_numbers=True)
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
                    schema = _fixture_property_schema(parameters, key)
                    # Families whose wire carries untyped text values: the authoring
                    # contract types them against the declared schema (a literal
                    # `null` for a nullable property, JSON for a non-string one).
                    if G.unified_grammar_family(family) not in {"qwen3", "glm47", "beam", "minimax_m3"} or not isinstance(value, str):
                        continue
                    if scenario in {probe["name"] for probe in XML_COMPATIBILITY_CASES} and schema == {"type": "custom"}:
                        assert family == "qwen3" or (
                            family == "glm47" and scenario == "xml_custom_type_double_quoted_object")
                        candidate["arguments"][key] = (
                            ast.literal_eval(value.strip()) if family == "qwen3"
                            else json.loads(value, parse_float=Decimal))
                        continue
                    if value == "null" and matches_schema(None, schema, parameters):
                        candidate["arguments"][key] = None
                    elif (not matches_schema(value, schema, parameters)
                          or (G.unified_grammar_family(family) in ("qwen3", "beam") and scenario in {
                              "schema_integer_const_ambiguous", "schema_object_const_union",
                              "schema_type_array_string_integer",
                          }) or (family == "minimax_m3" and scenario in {
                              "schema_integer_const_ambiguous", "schema_object_const_union",
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
        elif family == "beam" and scenario == "structured_response_after_reasoning":
            assert raw.startswith(control_tokens(family)[0].removeprefix("<|start|>assistant")), (
                family, scenario, "missing generated Beam reasoning channel",
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
            "interrupted_reasoning_then_response",
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



# These exact contracts were authored before generator expansion and checked
# against the frozen public API. They include source controls, not measured
# outputs as oracles. Hashing the typed object pins input bytes, chunks, schemas,
# initialization, finish and GOLDEN without a second native grammar renderer.
# Source ownership: DS4.1 paired thoughts close only at </think>; Beam analysis
# closes at <|end|>; Qwen and Glimmer native invoke closes are not thought closes.
# Spark filters unmatched ATEM closers. GLM/M3/K2/Gemma native body suppression
# and K3 malformed-body recovery retain their distinct output contracts.
_GUIDED_HEADER_CONTRACTS = {
    ('deepseek_v41_json_invocation_body', 'deepseek_v4'): 'e70a32b9e8a688a3767cd1bf6588aead65e6e106b13dd4094963a6392c659302',
    ('deepseek_v41_json_invocation_body', 'deepseek_v41'): 'f2bcb6900c7e63dcb86b71f30ac77523e785518ab82dff36b61ff65a7838f36e',
    ('deepseek_v41_json_invocation_body', 'gpt_oss'): 'ef1462dedc0cb07027ee2e7a20c9e770611f5700ddad0e6ae153e387f760ea74',
    ('deepseek_v41_json_invocation_body', 'kimi_k2'): '02b3c073c54cb94f19c8805eb4873923e9dc810424b163ef12298891ad726c7e',
    ('deepseek_v41_json_invocation_body', 'kimi_k3'): 'ea2e6ee973c2d0382b817896d5756b074841eb1c85c22a5bb98e8c1b783aef1e',
    ('kimi_k3_malformed_call_then_valid', 'beam'): '339407e1cb4dd3e3ff08858edcb8df6a8df5139bf6d9c96bd4804ce24962aa2d',
    ('kimi_k3_malformed_call_then_valid', 'deepseek_v4'): 'e6fb82fb7098aadeea52d0005b9e3232b8e289db2a558a1382204501f68288a8',
    ('kimi_k3_malformed_call_then_valid', 'deepseek_v41'): 'c341e455714683c42f65ae57979aa688cf614bf5396d81c49e34072cc4513ab6',
    ('kimi_k3_malformed_call_then_valid', 'gemma4'): '05d7352d34ebd6f17d6fa5ad7ec331803b826b5f5ab41091bd269e4d645b998f',
    ('kimi_k3_malformed_call_then_valid', 'gpt_oss'): 'd2b523b148915825b7f5bf27b7503c3dd57dcdb8fe4c98466de03ef199e54dab',
    ('kimi_k3_malformed_call_then_valid', 'kimi_k2'): '1116fa10eba6e4c2d5980a78067fff0c4d87097d7d4ded0b4a928220290fbaea',
    ('kimi_k3_malformed_call_then_valid', 'kimi_k3'): '37fd4fd46fc844b091d4b613714aa065d118f32056d9e96b78ac61006f7a2bce',
    ('kimi_k3_malformed_call_then_valid', 'minimax_m3'): '9b90f4bfb970cf7aed40a42087e4ea614cda04b36afdb2a4b859e4a17f2deec9',
    ('kimi_k3_malformed_call_then_valid', 'muse_glimmer'): '445c5f2df28559eea96a7500c91a8935a6083247c06e97f0ec127ab9c43abbcf',
    ('kimi_k3_malformed_call_then_valid', 'muse_spark'): '445c5f2df28559eea96a7500c91a8935a6083247c06e97f0ec127ab9c43abbcf',
    ('kimi_k3_malformed_call_then_valid', 'nemotron_v3'): '87875d58b7077bb87ffb8fc57ae597f3bd06f5b69d41bfc04f1734415e382c79',
    ('kimi_k3_malformed_call_then_valid', 'qwen3'): '55284d49f86242bd832eac172e98b8d3c509b9c5d44217e966748e2c641219ed',
    ('gpt_oss_channel_header_variants', 'beam'): '032f4f8695998f54b39d223742df3f983a5edaa24860b3576d95a33e62725928',
    ('gpt_oss_channel_header_variants', 'gpt_oss'): '267b1f4c0627db20319ce84572654b1cbb39617d9da107ed20d07b81c53dc43c',
    ('guided_json_quoted_bare_tool_header_in_answer', 'beam'): 'bc2cd592fd4ebc5432462809a90d8761628d058a6ecc1a7a2766b0170fa7d69c',
    ('guided_json_quoted_bare_tool_header_in_answer', 'gpt_oss'): '3facc58a85310e9d07aa9c0950b8cbc8bd058d7da6269d49f66d09cc11ec9b39',
    ('guided_json_quoted_bare_tool_header_in_answer', 'muse_glimmer'): 'e40560e521651d02411f27f90f607f6272c842fc633f6ed28f49d284f12620ff',
    ('guided_json_quoted_bare_tool_header_in_answer', 'muse_spark'): 'e40560e521651d02411f27f90f607f6272c842fc633f6ed28f49d284f12620ff',
    ('qwen3_guided_non_ascii_header_in_closed_reasoning', 'beam'): '3f2f6c5051de7c32b358374ffddece841b237b0ea8d80f3fb911efb033fe7864',
    ('qwen3_guided_non_ascii_header_in_closed_reasoning', 'deepseek_v4'): 'ef11a943597a6c5ada1ff1a8386fb3bbabaa5218ec896bd9a9a2f0b3ad67e0d7',
    ('qwen3_guided_non_ascii_header_in_closed_reasoning', 'deepseek_v41'): '16508337f2a531edbee2fbddade49c44ebaa2be6293f2ab545b75591b8ea0ae8',
    ('qwen3_guided_non_ascii_header_in_closed_reasoning', 'gemma4'): 'faa51868cb5e03cdc10075b8ffb84ff74bcf9ceaf3881ff96dfd969ad28badee',
    ('qwen3_guided_non_ascii_header_in_closed_reasoning', 'glm47'): 'ad921cade87dc33f1d026e1f38ed9957a6af472cb279522dec1058ed9b3a0a04',
    ('qwen3_guided_non_ascii_header_in_closed_reasoning', 'gpt_oss'): '105c0d668bc8eed483fc4aeccca1e4ea73cecd32bd2fb010cbd158fdbe79e624',
    ('qwen3_guided_non_ascii_header_in_closed_reasoning', 'kimi_k2'): '0ee281e87bb7ed55ab7e58045dd7086f1cbc2d97667cd00147ca387c0840c37b',
    ('qwen3_guided_non_ascii_header_in_closed_reasoning', 'kimi_k3'): '6af14aee61f9a2bc7d8584de5faa9e84e5e9ea91041ca02bcbd88a022705d0c9',
    ('qwen3_guided_non_ascii_header_in_closed_reasoning', 'minimax_m3'): '010a33adcae9797d44e9b1d9aeebfc6694b534c5281faf438434487afce3ac61',
    ('qwen3_guided_non_ascii_header_in_closed_reasoning', 'muse_glimmer'): '8435f2fb179186d286fa8b9a434675629ad43a352b3c633230f0382d46549f46',
    ('qwen3_guided_non_ascii_header_in_closed_reasoning', 'muse_spark'): '77191de682ced7c81f000b137326dce911c7fcfd7e1afc532078db5275dfa1e3',
    ('qwen3_guided_non_ascii_header_in_closed_reasoning', 'nemotron_v3'): '9b0f1ebca5d41e98a1515b5c4f533565c467afdecf57f6de791ee877e607c04c',
    ('qwen3_guided_non_ascii_header_in_closed_reasoning', 'qwen3'): 'f0643a0be36345d9fa4d4104a331a9af9b3dc0166e5438b4a32113e3ee58f636',
    ('qwen3_guided_non_ascii_header_in_truncated_reasoning', 'beam'): '08f423d284fd39592675efe8c9271313949d6cdb6c664efb0ef440412bc5a789',
    ('qwen3_guided_non_ascii_header_in_truncated_reasoning', 'deepseek_v4'): '5e416e1eef41793961a8ec5dc800482270a66764b3aee99af8ce33cc43863daa',
    ('qwen3_guided_non_ascii_header_in_truncated_reasoning', 'deepseek_v41'): '378d5f6bb5d87579ffda106ac906c97f566c1585b1ef50fae3e02fc19f6c6e2a',
    ('qwen3_guided_non_ascii_header_in_truncated_reasoning', 'gemma4'): 'ad509ecdfe21a434f669e5a6d74cb19713a5e531d2bd9ed762e2b327f654c404',
    ('qwen3_guided_non_ascii_header_in_truncated_reasoning', 'glm47'): '2b191ea1f4334dedf3fcca96be956d3435c6530fe747b8e52d8663b465b9b682',
    ('qwen3_guided_non_ascii_header_in_truncated_reasoning', 'gpt_oss'): 'e998137bb77ea4f4352dce6947869eb67641a85951bb7664f7430c6f35cc2a10',
    ('qwen3_guided_non_ascii_header_in_truncated_reasoning', 'kimi_k2'): 'b16d7fe7d9babec7d70e970334eb01900ff14420cef9037beb65963bbceccefc',
    ('qwen3_guided_non_ascii_header_in_truncated_reasoning', 'kimi_k3'): '99cee06b1f920b52792c3960b5d8265d536db802a6b929639b327423bcc8a3a1',
    ('qwen3_guided_non_ascii_header_in_truncated_reasoning', 'minimax_m3'): 'f6c3b0d4006f991a567a45e165d62871fe663ff909d5b3b34c7942b71ee658b0',
    ('qwen3_guided_non_ascii_header_in_truncated_reasoning', 'muse_glimmer'): 'bb025c58a22f4f539846846b3c5b17d046c1f39657c805ebb1935d4a014056de',
    ('qwen3_guided_non_ascii_header_in_truncated_reasoning', 'muse_spark'): 'bb025c58a22f4f539846846b3c5b17d046c1f39657c805ebb1935d4a014056de',
    ('qwen3_guided_non_ascii_header_in_truncated_reasoning', 'nemotron_v3'): '2807139322bc39f4fd3546196dda5d26003c7751871d7a0cbc4d2fae93ee6f32',
    ('qwen3_guided_non_ascii_header_in_truncated_reasoning', 'qwen3'): 'bb0617470b571dae9ada0b20f331b405c907aad0b7d53e6a5d202693256049b9',
    ('qwen3_guided_reasoning_opener_inside_native_header', 'beam'): '8406c855adeb9c921f8632d1fba100c71e425ee5aa4a926493dd5630a8dc664c',
    ('qwen3_guided_reasoning_opener_inside_native_header', 'deepseek_v4'): '0364e5803ee0b04dda97ed81e7ba037f528daa938112be0a938f86daa9c4947c',
    ('qwen3_guided_reasoning_opener_inside_native_header', 'deepseek_v41'): '6cedfcd697cdd2435a4bf67378718834f5ce9d8e5bdbc7196819c2e09d6bb819',
    ('qwen3_guided_reasoning_opener_inside_native_header', 'gemma4'): 'da284342b735af89a152c06fd5d6ba117b389dd926f6ac77074d8c33848dcf19',
    ('qwen3_guided_reasoning_opener_inside_native_header', 'glm47'): 'ae3e25e6f6e466e44b9d37d6e683f97dd60b22c33d1e1deca4e3ce4f87ae3e8c',
    ('qwen3_guided_reasoning_opener_inside_native_header', 'gpt_oss'): '53a41d09d3e91400686f15bd21d9d07d09eae3c9efa368febb96818c8a754581',
    ('qwen3_guided_reasoning_opener_inside_native_header', 'kimi_k2'): 'd71859336314c6ce07736b10aa9e9b9fffb9a0d0920a339907c44e892add2ed8',
    ('qwen3_guided_reasoning_opener_inside_native_header', 'kimi_k3'): '91df8b22eb40bb2520ae2903a5aad8988dc6cb9b06f9e3d59f04a8eb329f5635',
    ('qwen3_guided_reasoning_opener_inside_native_header', 'minimax_m3'): '8fa1010e7ee80a1d87ccca87c063647b26c1d0f802d86eb27740dbc897f41c02',
    ('qwen3_guided_reasoning_opener_inside_native_header', 'muse_glimmer'): '6d7e8b03cd27c04f4cc3205e541cee1da6f1e48db008b200c8ad3fbb9b0146fa',
    ('qwen3_guided_reasoning_opener_inside_native_header', 'muse_spark'): '5af08ffb68765e96de92e4750a232afb785a4b0b6e2abeaccecfe39731f61a47',
    ('qwen3_guided_reasoning_opener_inside_native_header', 'nemotron_v3'): '783d52f460abe41207e22c54a99ef8a771d4c71933523a6e876cb949256a7d34',
    ('qwen3_guided_reasoning_opener_inside_native_header', 'qwen3'): '2e22c07ce4e4a6c0784cca7767e293451bcd2bd79dc73a78a0a98421ab40677b',
    ('deepseek_v4_guided_reasoning_opener_inside_native_body', 'beam'): '665ab61ff5149c17bc7c9933391639c631f34c452b19cb3e1271364b722fd905',
    ('deepseek_v4_guided_reasoning_opener_inside_native_body', 'deepseek_v4'): '4f419464e02cb63c945bccc5874e806d6aac4bb7b03ee97141ef965cd5e4d9d7',
    ('deepseek_v4_guided_reasoning_opener_inside_native_body', 'deepseek_v41'): 'a6814bd8b6177493aab3376e0a45cbb3d2cd5832670fe5ee083b5a1772f2c216',
    ('deepseek_v4_guided_reasoning_opener_inside_native_body', 'gemma4'): 'c4004f361fd729a51bc11683714df5bd63671794c10ba00e236ab66c2b332332',
    ('deepseek_v4_guided_reasoning_opener_inside_native_body', 'glm47'): '637b84aec76d836183f7e94ad65efae22451b1999c05dbb5b7b8db9abe8f35e4',
    ('deepseek_v4_guided_reasoning_opener_inside_native_body', 'gpt_oss'): 'd0806b8c0609dc979792a6144959804e161381c9e8332fbc9aafc3900646def0',
    ('deepseek_v4_guided_reasoning_opener_inside_native_body', 'kimi_k2'): 'c9e242478cd9c9313f5d2759ca94ded4b3ad160aa82b8e7ef1f32f0c3ceb3d69',
    ('deepseek_v4_guided_reasoning_opener_inside_native_body', 'kimi_k3'): '9ce7d26143f9841293c7f177f3671af04d5e20189244b9fdb0ebb30cb6900d6f',
    ('deepseek_v4_guided_reasoning_opener_inside_native_body', 'minimax_m3'): '97a4f0c3a7ebbf43dadc395bd6cd5079e464073cc27cc2ede781dda1b058bfce',
    ('deepseek_v4_guided_reasoning_opener_inside_native_body', 'muse_glimmer'): 'ae42698e2eb30a96b9f23cc7089d5a88f14caf1b513a574bd2760fddc166c935',
    ('deepseek_v4_guided_reasoning_opener_inside_native_body', 'muse_spark'): '737a4a66204e4e353378b62d0fc4d696b5f59b174692d00c0d7f65f164efa6c6',
    ('deepseek_v4_guided_reasoning_opener_inside_native_body', 'nemotron_v3'): 'a4e2be8ec4fa3fcbea2da3c354c80726fcaab1ef655587adcc82466ec7a464a7',
    ('deepseek_v4_guided_reasoning_opener_inside_native_body', 'qwen3'): 'a4e2be8ec4fa3fcbea2da3c354c80726fcaab1ef655587adcc82466ec7a464a7',
    ('gemma4_guided_reasoning_opener_after_call_prefix', 'beam'): '305276eed19a8522986af0be0d41eee1ceb732b3e54fc9732642fe2568512f70',
    ('gemma4_guided_reasoning_opener_after_call_prefix', 'deepseek_v4'): '4654cedc69859562abb631c60e49014cb05253788c42ea026f9a393f91022661',
    ('gemma4_guided_reasoning_opener_after_call_prefix', 'deepseek_v41'): 'afdaf3f5d1c558982c9a6ae48ba39ded8c26d9d934dda3d6bbb2ffc6328b2a6f',
    ('gemma4_guided_reasoning_opener_after_call_prefix', 'gemma4'): 'eb3f8c653f5031d7f45d405f29dcca19a5bf20e49db97322f88e1bcf8e802125',
    ('gemma4_guided_reasoning_opener_after_call_prefix', 'glm47'): 'ae14a85e41345e88823a5b9e54218828cb06b07fb66b36a219b5dba58a9af227',
    ('gemma4_guided_reasoning_opener_after_call_prefix', 'gpt_oss'): '6b1dbeaaadf0e888b59589221e9354733ea5748c252f8c2d99530e0085357cf5',
    ('gemma4_guided_reasoning_opener_after_call_prefix', 'kimi_k2'): 'fc9bd7f704a8090d10a5b743da82a7b180f82655f6476ed25b8d7f746f58047e',
    ('gemma4_guided_reasoning_opener_after_call_prefix', 'kimi_k3'): '14fdf0591e9d01e042939b132f1400d73ec8d269ea8aff4a1904e18139b8617e',
    ('gemma4_guided_reasoning_opener_after_call_prefix', 'minimax_m3'): '37c2c798d4569b28d8501ce349b0e822ef3a2eaa4996dbc0ce1a0eb507147e4e',
    ('gemma4_guided_reasoning_opener_after_call_prefix', 'muse_glimmer'): '9213b7c5becd333c01c394404bb22d34c7c088e9b0e0cc99ac80aad3389b6cf8',
    ('gemma4_guided_reasoning_opener_after_call_prefix', 'muse_spark'): '9213b7c5becd333c01c394404bb22d34c7c088e9b0e0cc99ac80aad3389b6cf8',
    ('gemma4_guided_reasoning_opener_after_call_prefix', 'nemotron_v3'): '2e1fffdec10f775fdcabdfa86a7fffd771db02b08bc5e0726735a98ee40a09c0',
    ('gemma4_guided_reasoning_opener_after_call_prefix', 'qwen3'): '2e1fffdec10f775fdcabdfa86a7fffd771db02b08bc5e0726735a98ee40a09c0',
    ('gemma4_guided_response_extra_body_opener', 'beam'): '7630bd3f534b3ca90c2a27bdc2137abd8163d5e0c4e36ec2e5c032c4c952c779',
    ('gemma4_guided_response_extra_body_opener', 'deepseek_v4'): '797fd6322473c36dc0e66a0011dbd05c6e1ab5204fe4a4f72c68dd898dcc1e0d',
    ('gemma4_guided_response_extra_body_opener', 'deepseek_v41'): '5d6e106381e888a85c03f5b4ec9daefeb5ab08153131b55572cab70e5bec6dc1',
    ('gemma4_guided_response_extra_body_opener', 'gemma4'): 'c059da4695e650138d5920354e6c7757856906a223c2bc112f734271dc353934',
    ('gemma4_guided_response_extra_body_opener', 'glm47'): 'd07d8e07b6a9d0668e41d6abc370ea7804f1e0e176946e62085a38d68843b55d',
    ('gemma4_guided_response_extra_body_opener', 'gpt_oss'): 'f1ecf984b79f9f00e1adda7e7aca72c371375e213145b994cec2540c928d81ee',
    ('gemma4_guided_response_extra_body_opener', 'kimi_k2'): '4162c432c3f4c1cdefd149985d7a51ed4eb60d9e0a6f17e186b00eb27bed0163',
    ('gemma4_guided_response_extra_body_opener', 'kimi_k3'): '5459c223674e693bce57da05298049b01b86f452b62ee36fd7f1f04412555c54',
    ('gemma4_guided_response_extra_body_opener', 'minimax_m3'): '4523c0d2c5cc91501de08a5a11f2553ef2a2d46228bc35fa24d895473752060d',
    ('gemma4_guided_response_extra_body_opener', 'muse_glimmer'): '96d512ae30ae104722da4fe1367d1cb641e5153b52052fc102f9d16f8c802d8d',
    ('gemma4_guided_response_extra_body_opener', 'muse_spark'): '96d512ae30ae104722da4fe1367d1cb641e5153b52052fc102f9d16f8c802d8d',
    ('gemma4_guided_response_extra_body_opener', 'nemotron_v3'): '29e01910d9bdaddd3c10d5f51fbe8336734ac25b400cbb452accec798d3c9010',
    ('gemma4_guided_response_extra_body_opener', 'qwen3'): '29e01910d9bdaddd3c10d5f51fbe8336734ac25b400cbb452accec798d3c9010',
    ('guided_json_quoted_bare_header_after_payload_pr424', 'gpt_oss'): 'b1eeafa3681a897de3a8a07fe7b853919edd773a33534b42fd6cef8ed73157eb',
    ('guided_json_quoted_bare_header_after_payload_pr424', 'muse_glimmer'): '0132c15e4ec64aa21abb7e87e842c9d57e57d3045a943a5e936ae09d6e209a9f',
    ('guided_json_quoted_bare_header_after_payload_pr424', 'muse_spark'): '0132c15e4ec64aa21abb7e87e842c9d57e57d3045a943a5e936ae09d6e209a9f',
    ('guided_json_quoted_bare_header_in_answer_pr424', 'gpt_oss'): '8e132b80d42b5d9a43efada5065a35703571eb1b4b295f3b2bf40d3fb59fa83a',
    ('guided_json_quoted_bare_header_in_answer_pr424', 'muse_glimmer'): 'd178e6166c15f90267b98be1e7fab781dc32fedd1d3d711b2da778d3f2f77076',
    ('guided_json_quoted_bare_header_in_answer_pr424', 'muse_spark'): 'd178e6166c15f90267b98be1e7fab781dc32fedd1d3d711b2da778d3f2f77076',
    ('guided_response_rejected_header_quote_ownership', 'deepseek_v4'): 'aad8755c282d3020618d908af2738dda7947c1e3fc121bbd17335d348a49bfdb',
    ('guided_response_rejected_header_quote_ownership', 'deepseek_v41'): '73759729650e439bce4b6e233cd71d1a5434e4d1fe3fbddedc9bf6b190146a49',
    ('guided_response_rejected_header_quote_ownership', 'kimi_k3'): '315a83be03ba450d0000b0ea2057b88a1c3955919f860677ebb66ca7be89e24a',
    ('guided_response_rejected_header_quote_ownership', 'minimax_m3'): '1d7e5a118dbc3086334a6cd5c02f83cc54de333d274527bbf8f356c3577b16f9',
    ('guided_response_rejected_header_quote_ownership', 'muse_glimmer'): 'afbcf0fc1841ddc8e317d53fd977addb47d7b7b789d18e1af4901a0c4df09606',
    ('guided_response_rejected_header_quote_ownership', 'muse_spark'): 'afbcf0fc1841ddc8e317d53fd977addb47d7b7b789d18e1af4901a0c4df09606',
    ('prefilled_response_guided_pending_invoke_header', 'beam'): 'fead872eabfb55339a1669a09a34f5fea4dc78d6bb3e34242adae8a67b9d7b05',
    ('prefilled_response_guided_pending_invoke_header', 'deepseek_v4'): '8830a850102fc5a8a3cae4d00ec119cf227da74cf7cec85914157336f1374991',
    ('prefilled_response_guided_pending_invoke_header', 'deepseek_v41'): '565207975e0d4dc983108ec6a678ea3f0c4bee8fd8e4ab5e5342b12477fcff8b',
    ('prefilled_response_guided_pending_invoke_header', 'gemma4'): '117383ce66c963c0cec4e375dca0ad8e27c17aa2339aa7e67daaba7a20be018f',
    ('prefilled_response_guided_pending_invoke_header', 'glm47'): '16015c08d3cd6fdbc300d4c61ca438fe8b7b1c1d19c80c49b71a0f985f1ae29b',
    ('prefilled_response_guided_pending_invoke_header', 'gpt_oss'): '5c0aeed9feb2cc010f9db542b41446bdc5f07dae3cbf0c0aee74d37c221afe59',
    ('prefilled_response_guided_pending_invoke_header', 'kimi_k2'): '524e8e4ee42a8df26bde2d83023cc9035ac5e6fe763732af2e985163b9942320',
    ('prefilled_response_guided_pending_invoke_header', 'kimi_k3'): '5b406e2b0946fb888e5dfcffa617a2bdca587481ceb62a8fcf97361a4ccf8a26',
    ('prefilled_response_guided_pending_invoke_header', 'minimax_m3'): 'e0e45522d68bef562dfdc0c98a62b0ec09d12f9816cd8910fcfb6e5851249aea',
    ('prefilled_response_guided_pending_invoke_header', 'muse_glimmer'): '7106260daf172c4a3612096b68395c092507bf1f821eb2ca1961c8c95b2d64d4',
    ('prefilled_response_guided_pending_invoke_header', 'muse_spark'): '7106260daf172c4a3612096b68395c092507bf1f821eb2ca1961c8c95b2d64d4',
    ('prefilled_response_guided_pending_invoke_header', 'nemotron_v3'): '6d54e87dbce89158033fb9e7e6fb4a1ebcc9260dcfb3e5d2e37b516cd522ddf2',
    ('prefilled_response_guided_pending_invoke_header', 'qwen3'): '6d54e87dbce89158033fb9e7e6fb4a1ebcc9260dcfb3e5d2e37b516cd522ddf2',
    ('muse_glimmer_guided_message_end_inside_native_header', 'beam'): 'd4adeec9f2407da6e6e1b14d90523ac298741a3dc62f4c57caf2057d82b5792b',
    ('muse_glimmer_guided_message_end_inside_native_header', 'gpt_oss'): '2ecbb95c15734192ca9cb409f032f073850d30a1525841ca41000404790ef6b1',
    ('muse_glimmer_guided_message_end_inside_native_header', 'kimi_k3'): 'c85257f184f28ca1fd49909a92474aec641eb27ccec4d64faf1cceea0ef9365c',
    ('muse_glimmer_guided_message_end_inside_native_header', 'muse_glimmer'): '1757c0006b0daab4d0bad93f240b84ea9ebda239399b409097095b6c25ddcd46',
    ('muse_glimmer_guided_message_end_inside_native_header', 'muse_spark'): '1757c0006b0daab4d0bad93f240b84ea9ebda239399b409097095b6c25ddcd46',
}


def _guided_header_scopes():
    all_families = set(NATIVE_FAMILIES)
    # This is a syntactic quote owned by a tool-name attribute. Unquoted tool
    # names and quoted parameter names cannot substitute for that owner.
    quoted_names = {family for family in all_families
                    if G.invoke_header_prefix(family).endswith('="')}
    assert quoted_names == {"deepseek_v4", "deepseek_v41", "minimax_m3", "kimi_k3", "muse_glimmer", "muse_spark"}
    # The retained child adds a full native role/message frame. Other families'
    # complete native paired/channel forms already exercise its semantic parent.
    framed = {family for family in all_families
              if _family_value("guided_json_quoted_bare_header_in_answer", family)
              != "I mean " + G.r_reason(family, "self literal")}
    assert framed == {"gpt_oss", "muse_glimmer", "muse_spark"}
    # Recipient-first messages exist in Beam/Harmony and ATEM; an invocation
    # name/argument in a paired-thought grammar cannot stand in for that owner.
    recipients = {family for family in all_families
                  if "to=" in G.r_tool_arguments(family, "sentinel", {}, 0).split("<|message|>", 1)[0]}
    assert recipients == {"beam", "gpt_oss", "muse_glimmer", "muse_spark"}
    channels = {family for family in all_families
                if "<|channel|>" in G.r_reason(family, "check")
                and "<|channel|>final" in G.r_text(family, "found")}
    assert channels == {"beam", "gpt_oss"}
    # These public native bodies encode JSON strings directly; XML parameter
    # projection and Gemma value syntax do not test that same body owner.
    raw_json_bodies = {"deepseek_v4", "deepseek_v41", "gpt_oss", "kimi_k2", "kimi_k3"}
    # GLM's unquoted name has no header/body delimiter before arg_key. Plain
    # not-an-argument therefore extends its name instead of entering this body.
    body_headers = {family for family in all_families
                    if G._keyless_native_body(family, "bad")[0].endswith((">", "{"))}
    assert body_headers == all_families - {"glm47"}
    return {
        "deepseek_v41_json_invocation_body": raw_json_bodies,
        "kimi_k3_malformed_call_then_valid": body_headers,
        "qwen3_guided_non_ascii_header_in_truncated_reasoning": all_families,
        "qwen3_guided_non_ascii_header_in_closed_reasoning": all_families,
        "qwen3_guided_reasoning_opener_inside_native_header": all_families,
        "guided_json_quoted_bare_tool_header_in_answer": recipients,
        "gpt_oss_channel_header_variants": channels,
        "muse_glimmer_guided_message_end_inside_native_header": {
            "beam", "gpt_oss", "kimi_k3", "muse_glimmer", "muse_spark"},
        "prefilled_response_guided_pending_invoke_header": all_families,
        "gemma4_guided_reasoning_opener_after_call_prefix": all_families,
        "gemma4_guided_response_extra_body_opener": all_families,
        "deepseek_v4_guided_reasoning_opener_inside_native_body": all_families,
        "guided_response_rejected_header_quote_ownership": quoted_names,
        "guided_json_quoted_bare_header_after_payload_pr424": framed,
        "guided_json_quoted_bare_header_in_answer_pr424": framed,
    }


def _assert_guided_header_case(family, scenario, case):
    request = {
        "input": case["input"], "init": case["init"],
        "finish_reason": case["finish_reason"],
        "tools": case.get("tools", G.unified_tools()),
        "chunks": [{"delta_text": chunk} for chunk in case.get("input_chunks", [case["input"]])],
        "golden": case["golden"],
    }
    digest = hashlib.sha256(json.dumps(request, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    assert digest == _GUIDED_HEADER_CONTRACTS[(scenario, family)], (family, scenario, "authored native header contract")


def _assert_guided_header_contracts(corpus):
    for scenario, expected in _guided_header_scopes().items():
        assert set(G.scenario_families(scenario)) == expected, (scenario, "native applicability")
        present = {family for family, cases in corpus.items()
                   if f"UNIFIED.{scenario}.{family}" in cases}
        assert present == expected, (scenario, "missing native counterpart", expected - present)
        for family in expected:
            _assert_guided_header_case(family, scenario, corpus[family][f"UNIFIED.{scenario}.{family}"])
        if scenario.endswith("_pr424"):
            parent = scenario.removesuffix("_pr424")
            assert taxonomy.case_fold(scenario)[0] == parent
            for family in expected:
                assert f"UNIFIED.{parent}.{family}" in corpus[family], (family, scenario, "missing folded parent")


def test_guided_header_native_scopes_and_exact_source_contracts():
    _assert_guided_header_contracts({family: build_cases(family) for family in NATIVE_FAMILIES})


@pytest.mark.parametrize("message_header,mutation", [
    *[(False, mutation) for mutation in ("omitted", "exclusion", "input_payload", "golden_payload", "folded_child")],
    *[(True, mutation) for mutation in ("omitted", "exclusion", "input_payload", "golden_payload")],
])
def test_guided_header_contract_rejects_omissions_exclusions_and_payload_drift(monkeypatch, mutation, message_header):
    corpus = {family: build_cases(family) for family in NATIVE_FAMILIES}
    scenario = ("guided_json_quoted_bare_header_after_payload_pr424" if mutation == "folded_child"
                else "prefilled_response_guided_pending_invoke_header")
    family = "muse_glimmer" if mutation == "folded_child" else "qwen3"
    if message_header:
        scenario = "muse_glimmer_guided_message_end_inside_native_header"
        family = "beam"
    key = f"UNIFIED.{scenario}.{family}"
    if mutation == "input_payload":
        case = corpus[family][key]
        case["input"] = (case["input"].replace('"name":"f"', '"name":"g"') if message_header
                         else case["input"].replace("Paris", "Rome"))
        case["input_chunks"] = list(case["input"])
    elif mutation == "golden_payload":
        if message_header:
            corpus[family][key]["golden"][-1]["name"] = "g"
        else:
            corpus[family][key]["golden"][-1]["arguments"]["city"] = "Rome"
    else:
        corpus[family].pop(key)
        if mutation == "folded_child":
            assert f"UNIFIED.{scenario.removesuffix('_pr424')}.{family}" in corpus[family]
        if mutation == "exclusion":
            original = G.scenario_families
            monkeypatch.setattr(G, "scenario_families", lambda value: (
                original(value) - {family} if value == scenario else original(value)))
    with pytest.raises(AssertionError):
        _assert_guided_header_contracts(corpus)



@pytest.mark.parametrize("scenario", [
    "qwen3_guided_non_ascii_header_in_truncated_reasoning",
    "qwen3_guided_non_ascii_header_in_closed_reasoning",
    "qwen3_guided_reasoning_opener_inside_native_header",
    "guided_json_quoted_bare_tool_header_in_answer",
    "gpt_oss_channel_header_variants",
])
@pytest.mark.parametrize("mutation", ["omitted", "exclusion", "native_bytes", "golden"])
def test_native_unicode_header_contract_rejects_missing_scope_and_literal_changes(monkeypatch, scenario, mutation):
    corpus = {family: build_cases(family) for family in NATIVE_FAMILIES}
    family = "glm47" if scenario == "qwen3_guided_reasoning_opener_inside_native_header" else "beam"
    key = f"UNIFIED.{scenario}.{family}"
    if mutation == "native_bytes":
        case = corpus[family][key]
        before, after = (("é", "e") if "non_ascii" in scenario else
                         ("get_weather", "get_time") if scenario == "guided_json_quoted_bare_tool_header_in_answer" else
                         ("weather", "forecast") if scenario == "gpt_oss_channel_header_variants" else
                         ("<tool_call>", "<tool>") )
        assert before in case["input"]
        case["input"] = case["input"].replace(before, after)
        case["input_chunks"] = list(case["input"])
    elif mutation == "golden":
        corpus[family][key]["golden"] = []
    else:
        corpus[family].pop(key)
        if mutation == "exclusion":
            original = G.scenario_families
            monkeypatch.setattr(G, "scenario_families", lambda value: (
                original(value) - {family} if value == scenario else original(value)))
    with pytest.raises(AssertionError):
        _assert_guided_header_contracts(corpus)


@pytest.mark.parametrize("producer_failure", ["character_marker_leak", "whole_reasoning_drop"])
def test_glm_unfinished_header_oracle_rejects_adopting_measured_parser_failures(producer_failure):
    scenario = "qwen3_guided_reasoning_opener_inside_native_header"
    case = build_cases("glm47")[f"UNIFIED.{scenario}.glm47"]
    expected = [{"kind": "reasoning", "text": "x"}, {"kind": "tool_call", "name": "f", "arguments": {}}]
    assert case["golden"] == expected
    if producer_failure == "character_marker_leak":
        case["golden"].insert(0, {"kind": "text", "text": "<tool_call>"})
    else:
        case["golden"].pop(0)
    with pytest.raises(AssertionError):
        _assert_guided_header_case("glm47", scenario, case)



@pytest.mark.parametrize("scenario", ["deepseek_v41_json_invocation_body", "kimi_k3_malformed_call_then_valid"])
@pytest.mark.parametrize("mutation", ["omitted", "exclusion", "input", "golden", "schema"])
def test_native_json_body_and_restart_contract_rejects_coordinated_drift(monkeypatch, scenario, mutation):
    corpus = {family: build_cases(family) for family in NATIVE_FAMILIES}
    family = "kimi_k2"
    key = f"UNIFIED.{scenario}.{family}"
    if mutation == "input":
        case = corpus[family][key]
        old, new = (("café", "cafe") if scenario == "deepseek_v41_json_invocation_body" else ("not-an-argument", "not-argument"))
        assert old in case["input"]
        case["input"] = case["input"].replace(old, new)
        case["input_chunks"] = list(case["input"])
    elif mutation == "golden":
        corpus[family][key]["golden"] = []
    elif mutation == "schema":
        corpus[family][key]["tools"] = []
    else:
        corpus[family].pop(key)
        if mutation == "exclusion":
            original = G.scenario_families
            monkeypatch.setattr(G, "scenario_families", lambda value: (
                original(value) - {family} if value == scenario else original(value)))
    with pytest.raises(AssertionError):
        _assert_guided_header_contracts(corpus)


@pytest.mark.parametrize("family", ["deepseek_v4", "deepseek_v41", "minimax_m3", "qwen3", "nemotron_v3", "muse_glimmer", "muse_spark"])
def test_native_unclosed_body_recovery_preserves_its_source_policy(family):
    scenario = "kimi_k3_malformed_call_then_valid"
    case = build_cases(family)[f"UNIFIED.{scenario}.{family}"]
    _assert_guided_header_case(family, scenario, case)
    if family == "deepseek_v41":
        assert case["golden"] == []
        assert case["expect"]["dynamo"]["class"] == "ERROR"
    case["golden"] = [{"kind": "tool_call", "name": "g", "arguments": {"y": "2"}}]
    with pytest.raises(AssertionError):
        _assert_guided_header_case(family, scenario, case)


def test_kimi_restart_oracle_does_not_adopt_the_measured_lost_call():
    scenario = "kimi_k3_malformed_call_then_valid"
    case = build_cases("kimi_k2")[f"UNIFIED.{scenario}.kimi_k2"]
    assert case["golden"] == [{"kind": "tool_call", "name": "g", "arguments": {"y": "2"}}]
    assert case["expect"]["dynamo"]["class"] == "LOSS"
    case["golden"] = []
    with pytest.raises(AssertionError):
        _assert_guided_header_case("kimi_k2", scenario, case)


def test_shared_guided_header_cases_use_generic_display_groups_without_renumbering():
    groups = {
        "deepseek_v41_json_invocation_body": 7,
        "kimi_k3_malformed_call_then_valid": 4,
        "qwen3_guided_non_ascii_header_in_truncated_reasoning": 34,
        "qwen3_guided_non_ascii_header_in_closed_reasoning": 34,
        "qwen3_guided_reasoning_opener_inside_native_header": 34,
        "guided_json_quoted_bare_tool_header_in_answer": 35,
        "gpt_oss_channel_header_variants": 11,
        "muse_glimmer_guided_message_end_inside_native_header": 32,
        "prefilled_response_guided_pending_invoke_header": 35,
        "gemma4_guided_reasoning_opener_after_call_prefix": 34,
        "gemma4_guided_response_extra_body_opener": 35,
        "deepseek_v4_guided_reasoning_opener_inside_native_body": 34,
        "guided_response_rejected_header_quote_ownership": 35,
    }
    for scenario, group in groups.items():
        assert taxonomy.display_tax(scenario)[0] == group, (scenario, taxonomy.display_tax(scenario), group)
    assert taxonomy.numbered_id("muse_glimmer_guided_message_end_inside_native_header") == "UNIFIED.muse-glimmer_guided_message_end_inside_native_header"
    assert taxonomy.numbered_id("prefilled_response_guided_pending_invoke_header") == "UNIFIED.deepseek_v41-prefilled_response_guided_pending_invoke_header"
    assert taxonomy.numbered_id("gemma4_guided_reasoning_opener_after_call_prefix") == "UNIFIED.gemma-guided_reasoning_opener_after_call_prefix"
    assert taxonomy.numbered_id("gemma4_guided_response_extra_body_opener") == "UNIFIED.gemma-guided_response_extra_body_opener"
    assert taxonomy.numbered_id("deepseek_v4_guided_reasoning_opener_inside_native_body") == "UNIFIED.deepseek_v4-deepseek_v4_guided_reasoning_opener_inside_native_body"


def _assert_quoted_tool_recipient_boundary(family, case):
    # Validate the semantic boundary before scope/digest checks so replacing a
    # recipient with ordinary reasoning prose still fails at its owning invariant.
    headers = {
        "beam": "to=get_weather" + control_tokens("beam")[2].split("NAMEX", 1)[1],
        "gpt_oss": "to=functions.get_weather<|channel|>commentary <|constrain|>json<|message|>",
        "muse_glimmer": "to=get_weather<|message|>",
        "muse_spark": "to=get_weather<|message|>",
    }
    assert family in headers and headers[family] in case["input"], (family, "guided_json_quoted_bare_tool_header_in_answer", "missing tool-recipient boundary")


def _assert_cross_family_contract(corpus):
    assert_native_response_counterparts(corpus, G, taxonomy)
    _assert_m3_nested_marker_contract(corpus)
    _assert_muse_native_scopes(corpus)
    _assert_kimi_native_scopes(corpus)
    for family, cases in corpus.items():
        key = f"UNIFIED.guided_json_quoted_bare_tool_header_in_answer.{family}"
        if key in cases:
            _assert_quoted_tool_recipient_boundary(family, cases[key])
    _assert_guided_header_contracts(corpus)
    _assert_gpt58_native_contracts(corpus)
    by_scenario = defaultdict(dict)
    clean = {row[0]: row[3] for row in CLEAN}
    prefills = {name for name, segments in clean.items() if segments[0][0] == "reason"} | {"reason_unterminated"}
    for family in NATIVE_FAMILIES:
        for cid, case in corpus[family].items():
            scenario = _scenario_of(cid, family)
            if (scenario, family) in _GUIDED_HEADER_CONTRACTS:
                # Exact source-authored inputs and source-policy outputs were
                # validated above before normalizing these grammar differences.
                by_scenario[scenario][family] = (None, None, None, None, [])
                continue
            if scenario in _GUIDED_NATIVE_RECOVERY_SCENARIOS and (family == "minimax_m3" or scenario != "guided_json_native_markup_only"):
                _assert_guided_native_recovery(family, scenario, case)
            _assert_input_carries_events(family, scenario, case)
            if scenario == "guided_json_quoted_bare_tool_header_in_answer":
                # A tool name inside a thought is prose, not a recipient header.
                _assert_quoted_tool_recipient_boundary(family, case)
            init = dict(case["init"])
            if family == "deepseek_v41" and scenario in prefills:
                assert init == {"starting_state": "Reasoning", "tool_output_mode": "Native", "named_tool": None}
                assert not case["input"].startswith("<think>"), (family, scenario, "prefilled opener must be absent")
                init["starting_state"] = "None"
            events = (_assert_malformed_recovery(family, case) if scenario == "malformed_json_then_two_valid_calls"
                      else _logical_events(scenario, family, case["golden"]))
            if (scenario, family) in _MUSE_NATIVE_CONTRACTS:
                # Source-qualified native EOF policies and literal marker values
                # were checked before comparing the shared ownership semantics.
                events = []
            if scenario.startswith("kimi_native_"):
                # Exact independent requests and family-specific recovery/marker
                # oracles were checked before comparing the shared semantics.
                events = []
            if scenario in {"kimi_k3_typed_response_then_call", "kimi_k3_direct_typed_response_recovery"}:
                # K3 recovers a complete value without its outer wrapper close;
                # these four native grammars require their call/invoke terminator.
                following = [{"kind": "text", "text": "recovered"}]
                if scenario == "kimi_k3_typed_response_then_call":
                    following.append({"kind": "tool_call", "name": "next", "arguments": {"n": 7}})
                assert events == ([{"kind": "tool_call", "name": "lookup", "arguments": {"q": "first"}}]
                                  if family == "kimi_k3" else []) + following, (family, scenario, "native recovery policy")
                events = following
            if scenario == "gpt_oss_multiple_unknown_and_empty_tools":
                raw_value = "2" if G.unified_grammar_family(family) in {"beam", "qwen3", "glm47", "minimax_m3"} else 2
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
                                  if family in {"muse_glimmer", "muse_spark", "kimi_k3"} else []) + following
                events = following
            if scenario == "gpt_oss_quoted_truncated_tool_then_final":
                assert events == [{"kind": "text", "text": "Answer"}], (family, scenario, "missing recovered answer")
            if scenario == "structured_response_after_reasoning":
                # Compare the schema and invalid payloads after checking each
                # native envelope; Harmony generates headers, others prefill.
                assert case["tools"] == []
                assert init["starting_state"] == ("None" if family in {"beam", "gpt_oss"} else "Reasoning")
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
            if family == "minimax_m3" and scenario in _GUIDED_NATIVE_RECOVERY_SCENARIOS:
                # Exact input/output assertions above precede normalization of
                # this documented exception; unrelated guided cases stay strict.
                events = ([{"kind": "tool_call", "name": "get_weather", "arguments": {"city": "Paris"}}]
                          if scenario == "kimi_k3_guided_native_wrapper" else
                          [event for event in events if event["kind"] != "tool_call"])
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
    _assert_generic_cases_span_families(by_scenario)
    assert set(by_scenario) == set(NATIVE_UNIFIED_TAX)
    assert all(family in by_scenario.get(scenario, {}) for scenario, family in G.INHERITED_GOLDEN), "INHERITED_GOLDEN names a case that does not exist"
    for scenario, families in by_scenario.items():
        assert set(families) == (set(G.scenario_families(scenario)) & set(NATIVE_FAMILIES)), (scenario, "applicability", set(families))
        # Beam authors its own golden where its grammar changes the outcome (its parameter
        # tags are plain text outside a routed header; its reasoning markers are special
        # tokens), so it stands outside the shared contract for those scenarios only.
        # A family-scoped parser rule declares its own golden in `INHERITED_GOLDEN`
        # (Nemotron V3's tool-only-turn whitespace, issue #406); that cell is checked
        # against the declaration instead of the shared contract.
        for family in families:
            if (scenario, family) in G.INHERITED_GOLDEN:
                assert families[family][0] == G.INHERITED_GOLDEN[(scenario, family)], (scenario, family, "inherited golden")
        contracts = [
            (family, contract) for family, contract in families.items()
            if not (family == "beam" and scenario in BEAM_GRAMMAR_EXCEPTIONS | BEAM_NUMERIC_STRING_SCENARIOS)
            and (scenario, family) not in G.INHERITED_GOLDEN
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
            for case_id in list(cases):
                if taxonomy.coverage_case(_scenario_of(case_id, family)) == scenario:
                    del cases[case_id]
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


@pytest.mark.parametrize("scenario", _GUIDED_NATIVE_RECOVERY_SCENARIOS)
@pytest.mark.parametrize("mutation", ["dispatch", "native_argument"])
def test_guided_native_recovery_contract_rejects_changed_calls(scenario, mutation):
    family = "minimax_m3"
    case = build_cases(family)[f"UNIFIED.{scenario}.{family}"]
    _assert_guided_native_recovery(family, scenario, case)
    if mutation == "dispatch":
        case["golden"] = [event for event in case["golden"] if event["kind"] != "tool_call"]
    else:
        case["input"] = case["input"].replace("Paris", "Rome")
    with pytest.raises(AssertionError, match="native-recovery"):
        _assert_guided_native_recovery(family, scenario, case)


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


def test_k3_response_recovery_oracle_requires_complete_json_and_preserved_text():
    cases = build_cases("kimi_k3")
    recovered = cases["UNIFIED.gpt_oss_truncated_tool_resynchronizes_at_next_message.kimi_k3"]
    raw = recovered["input"].split('<|open|>json type="object"<|sep|>', 1)[1].split('<|close|>json<|sep|>', 1)[0]
    assert json.loads(raw) == {"q": "unfinished"}
    assert recovered["golden"] == [
        {"kind": "tool_call", "name": "lookup", "arguments": {"q": "unfinished"}},
        {"kind": "text", "text": "recovered"},
    ]
    malformed = cases["UNIFIED.gpt_oss_quoted_truncated_tool_then_final.kimi_k3"]
    raw = malformed["input"].split('<|open|>json type="object"<|sep|>', 1)[1].split('<|close|>json<|sep|>', 1)[0]
    with pytest.raises(json.JSONDecodeError):
        json.loads(raw)
    assert malformed["golden"] == [{"kind": "text", "text": "Answer"}]


def test_kimi_display_sections_match_their_applicable_family():
    labels = taxonomy.display_case_labels(UNIFIED_TAX)
    for scenario in UNIFIED_TAX:
        group = taxonomy.display_tax(scenario)[0]
        if group not in {"kimi", "kimi_k2"}:
            continue
        family = "kimi_k3" if group == "kimi" else "kimi_k2"
        assert G.scenario_families(scenario) == frozenset({family}), scenario
        assert labels[scenario]["label"].startswith(f"{family}-"), scenario
        assert labels[scenario]["compact_prefix"] == family, scenario
    for scenario in UNIFIED_TAX:
        if scenario.startswith("kimi_native_"):
            expected = ("kimi" if scenario.startswith(("kimi_native_k3_typed_", "kimi_native_k3_json_"))
                        else 7 if "_count_content_chunk" in scenario or scenario.endswith("interfield_whitespace")
                        else 12 if scenario.endswith("rejected_next_header_progress") else 5)
            assert taxonomy.display_tax(scenario)[0] == expected, scenario
            assert taxonomy.numbered_id(scenario) == "UNIFIED.kimi-" + scenario


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
    assert v4["label"] == "35-guided_response_rejected_header_quote_ownership"
    assert v4["compact_prefix"] == "35"
    v41 = labels["deepseek_v41_json_invocation_body"]
    assert v41["label"] == "7-json_invocation_body"
    assert v41["compact_prefix"] == "7"
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


def test_native_channel_header_variants_share_the_interleaving_section():
    scenario = "gpt_oss_channel_header_variants"
    assert G.scenario_families(scenario) == {"beam", "gpt_oss"}
    assert taxonomy.case_fold(scenario) is None
    assert taxonomy.display_tax(scenario) == (11, "1")
    assert scenario not in GPT_OSS_STANDALONE_HEADER_SCENARIOS


@pytest.mark.parametrize(("scenario", "parent", "families"), [
    ("tool_body_whitespace", "tool_only", {"gpt_oss", "kimi_k2", "kimi_k3", "deepseek_v4", "deepseek_v41"}),
    ("gpt_oss_quoted_truncated_tool_then_final", "tool_no_close", {"beam", "gpt_oss", "muse_glimmer", "muse_spark", "kimi_k3"}),
    ("gpt_oss_quoted_message_header_literal", "arg_marker_in_string", {"beam", "gpt_oss", "muse_glimmer", "muse_spark", "kimi_k3"}),
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
            if family == "beam":
                assert '<rfl:parameter name="q">\nabc<|call|>' in case["input"]
                assert "</rfl:parameter>" not in case["input"]
                assert "final<|message|>Answer<|return|>" in case["input"]
            elif family in {"muse_glimmer", "muse_spark"}:
                assert '<atem:parameter name="q">abc<|eom|>' in case["input"]
                assert "</atem:parameter>" not in case["input"]
            else:
                assert '{"q":"abc<|close|>json<|sep|>' in case["input"]
                assert '<|open|>response<|sep|>Answer<|close|>response<|sep|>' in case["input"]
                assert case["expect"]["dynamo"]["verdict"] == "match"
        else:
            event = case["golden"][0]
            assert event["name"] == ("get_weather" if parent == "tool_only" else "lookup")
            if family == "beam":
                assert_gpt58_native_contract(family, scenario, case)
            else:
                assert matches_schema(event["arguments"], case["tools"][0]["parameters"])
            if parent == "tool_only":
                assert ' \n\t{"city":' in case["input"]
                assert event["arguments"] == {"city": "Paris"}
            else:
                payload = event["arguments"]["q"]
                assert payload in case["input"]
                assert "literal" in payload
                header = ("assistant to=user<|message|>" if family in {"muse_glimmer", "muse_spark"}
                          else "final<|message|>" if family == "beam" else '<|open|>response<|sep|>')
                assert header in payload


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


def test_single_call_variants_keep_the_shared_weather_example():
    assert historical_case_label("1-3") == "1-dotted_name_minimal"
    expected = [{"kind": "tool_call", "name": "get_weather", "arguments": {"city": "Paris"}}]
    source = next(case for case in G._GPT_OSS_REVIEW_CASES if case[0] == "tool_body_whitespace")
    assert source[3] == expected
    assert ' \n\t{"city":"Paris"}' in source[2]
    for family, (raw, golden, _) in G._native_review_variants("tool_body_whitespace").items():
        assert golden == expected, family
        assert ' \n\t{"city":' in raw, family
    for family in G.UNIFIED_FAMILIES:
        cases = build_cases(family)
        assert cases[f"UNIFIED.tool_only.{family}"]["golden"] == expected, family
        variant = (cases[f"UNIFIED.tool_body_whitespace.{family}"]
                   if family in {"gpt_oss", "kimi_k2", "kimi_k3"} else None)
        if variant is None:
            continue
        assert variant["golden"] == expected, family
        assert ' \n\t{"city":' in variant["input"], family
        assert taxonomy.case_fold("tool_body_whitespace")[0] == "tool_only"


def _assert_interrupted_response_examples(corpus):
    scenario = "interrupted_reasoning_then_response"
    # A response opener supplies the missing boundary; paired think tags with
    # bare visible text cannot express this request without adding a closer.
    applicable = {family for family in NATIVE_FAMILIES
                  if G.r_text(family, "sentinel") != "sentinel"}
    assert applicable == {"beam", "gpt_oss", "kimi_k3", "muse_glimmer", "muse_spark"}
    assert G.scenario_families(scenario) == applicable
    present = {family for family, cases in corpus.items()
               if f"UNIFIED.{scenario}.{family}" in cases}
    assert present == applicable
    for family in applicable:
        case = corpus[family][f"UNIFIED.{scenario}.{family}"]
        parent = corpus[family][f"UNIFIED.reason_then_content.{family}"]
        assert case["golden"] == parent["golden"]
        assert case["input"] == (
            G.r_reason(family, "let me think").removesuffix(control_tokens(family)[1])
            + G.r_text(family, "The answer is 42.")
        )


def test_interrupted_response_examples_have_native_applicability_and_parent_payloads():
    _assert_interrupted_response_examples({family: build_cases(family) for family in NATIVE_FAMILIES})


@pytest.mark.parametrize("mutation", ["omitted", "exclusion", "payload"])
def test_interrupted_response_contract_rejects_coordinated_omission_and_payload_drift(monkeypatch, mutation):
    corpus = {family: build_cases(family) for family in NATIVE_FAMILIES}
    scenario = "interrupted_reasoning_then_response"
    case_id = f"UNIFIED.{scenario}.kimi_k3"
    if mutation == "payload":
        corpus["kimi_k3"][case_id]["golden"][0]["text"] = "checking"
    else:
        corpus["kimi_k3"].pop(case_id)
        if mutation == "exclusion":
            original = G.scenario_families
            monkeypatch.setattr(G, "scenario_families", lambda value: (
                original(value) - {"kimi_k3"} if value == scenario else original(value)))
    with pytest.raises(AssertionError):
        _assert_interrupted_response_examples(corpus)


def test_dotted_names_include_the_registered_minimax_native_grammar():
    for scenario in G.DOTTED_NAME_SCENARIOS:
        case = build_cases("minimax_m3")[f"UNIFIED.{scenario}.minimax_m3"]
        assert case["golden"] == build_cases("qwen3")[f"UNIFIED.{scenario}.qwen3"]["golden"]
        for event in case["golden"]:
            assert event["name"] in case["input"]
            assert event["name"] in {tool["name"] for tool in case["tools"]}


@pytest.mark.parametrize("family", ["gpt_oss", "kimi_k2", "kimi_k3"])
def test_retired_whitespace_request_preserves_lookup_history(family):
    path = SRC.parents[2] / f"conformance/fixtures-unified-v2/families/{family}/inputs_and_golden.yaml"
    cases = yaml.safe_load(path.read_text())["cases"]
    retained = cases["gpt_oss_tool_body_whitespace"]
    assert retained["lifecycle"] == "retired"
    assert retained["display_id"] is None
    assert "UNIFIED.gpt_oss-tool_body_whitespace" in retained["historical_ids"]
    assert retained["golden"]["assembled"] == [
        {"kind": "tool_call", "name": "lookup", "arguments": {"q": "x"}},
    ]
    assert "lookup" in retained["request"]["input"]
    assert not G.scenario_families("gpt_oss_tool_body_whitespace")
    assert "tool_body_whitespace" in cases


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


_KIMI_NATIVE_REQUEST_DIGESTS = {
    ('kimi_native_incompatible_string_recovery', 'deepseek_v4'): 'd1ef0e7a9a6ea568f8016ed2f35153f82916f958ef5dded172487698a09d0a62',
    ('kimi_native_incompatible_string_recovery', 'deepseek_v41'): 'b80ac979024c3c8dacd0e3c9b382ec1179ffd36ee0a9309ae5ebc103a78791a0',
    ('kimi_native_incompatible_string_recovery', 'gemma4'): 'b5aa7543cfa5a0e48c906fff9bdebbc8b74196c12168fc06d2f8ce56324e347e',
    ('kimi_native_incompatible_string_recovery', 'gpt_oss'): 'e0f9e0531e45783242a54008bc9f1b6fe760fa787fb367912bee9e6a01c01465',
    ('kimi_native_incompatible_string_recovery', 'kimi_k2'): 'fb048e83aace703f910134c3db55ce3e35596e5c3c38e73ddb8d22133a91f6f4',
    ('kimi_native_incompatible_string_recovery', 'kimi_k3'): '7444b6e31b39c6b81749672ad50b0bc71c3353faa2075157e041feb4dbc0a4c9',
    ('kimi_native_incremental_interfield_whitespace', 'beam'): '16e1997983919314d3aaf43b1883b2babb2dc74025abe3349950bca70cdc34b4',
    ('kimi_native_incremental_interfield_whitespace', 'deepseek_v4'): 'bce33da43d1f781fb41d7b615c9c762029ff47ea5106ac7ce7e914975397eb42',
    ('kimi_native_incremental_interfield_whitespace', 'deepseek_v41'): '03412b8d6d9f53684515861f8e412e752e4e018e7110d6b380f1e1d470040d43',
    ('kimi_native_incremental_interfield_whitespace', 'gemma4'): '528458fc5961b3dbe7d6bdcd369912a4f955b055682f72f73c3207d6c8f0b881',
    ('kimi_native_incremental_interfield_whitespace', 'glm47'): '6530678bf6bd4652c3939e9db94c127eddc3730d594431b1aabce67294817a50',
    ('kimi_native_incremental_interfield_whitespace', 'gpt_oss'): 'c11847f9c9259818871b05099819b44575219c6eaeb48ef7203e61d02320a586',
    ('kimi_native_incremental_interfield_whitespace', 'kimi_k2'): '23ae2018e8f3d886b601f749c8bfc5a24f4be892a4b04a3d6d462089a7f19e55',
    ('kimi_native_incremental_interfield_whitespace', 'kimi_k3'): '99e02e143faee7b227bed46b6353385271c0d95836ea5cd8d997be73dfe03462',
    ('kimi_native_incremental_interfield_whitespace', 'minimax_m3'): '531ed08167bcca4e725c47c158a81c35213d744d1482a1797fd9ef064ecdbe6f',
    ('kimi_native_incremental_interfield_whitespace', 'muse_glimmer'): '4526857ed7db5b59e4f65c734d457995249fcdb072e730f10b8d9f0fa9920ea6',
    ('kimi_native_incremental_interfield_whitespace', 'muse_spark'): '4526857ed7db5b59e4f65c734d457995249fcdb072e730f10b8d9f0fa9920ea6',
    ('kimi_native_incremental_interfield_whitespace', 'nemotron_v3'): '5fa580bc6af1164bf88767f00f358a0d38db278781b016838f721a4ed5b32fa6',
    ('kimi_native_incremental_interfield_whitespace', 'qwen3'): '32f938ea493484aa4a7e4ef0f4fd484ccf513a74f33bd34770c340405ee5d8f0',
    ('kimi_native_k2_count_content_chunk1024', 'beam'): '3c7d66e2cf7760d7d42acc21cddd6b39b9082b26565289b58af5ffa377cbd3d1',
    ('kimi_native_k2_count_content_chunk1024', 'deepseek_v4'): '9ba76a50b2afc16f3907d54fb1dfef12f84a60e3756b7724ea43d8c272312297',
    ('kimi_native_k2_count_content_chunk1024', 'deepseek_v41'): 'ce7d6b086365f06708942b320ea3dd27dd3223108ce09c7b08b5111138141861',
    ('kimi_native_k2_count_content_chunk1024', 'gemma4'): '2b99650589d9aa61f209fbb5487957d920246d1e670b72b929dd523c0e68f0ef',
    ('kimi_native_k2_count_content_chunk1024', 'glm47'): 'c59839bef14ea1b29b6cb77d15944cacb3c0ee0157e619f48ba2d6afbd5e7f2e',
    ('kimi_native_k2_count_content_chunk1024', 'gpt_oss'): 'b0f6a64afea18ca9ed3769c945637cac5d2c017706003f0db5f281e29e97d3f6',
    ('kimi_native_k2_count_content_chunk1024', 'kimi_k2'): 'f629f6ecb8ed8ddef512e079acacfffd57ee7bf86180f13ee1fe09d46c2dec87',
    ('kimi_native_k2_count_content_chunk1024', 'kimi_k3'): 'c242650467aacd2c4f094547204fca86021b24f15c6c95ea99192f16c274da2b',
    ('kimi_native_k2_count_content_chunk1024', 'minimax_m3'): 'e7a212aedeac5d1352d0169b723925aa7ecfe2142eb68dfa795bbbfe6e0534bc',
    ('kimi_native_k2_count_content_chunk1024', 'muse_glimmer'): 'a6caff65710f7b3614424b8948b00eaf534fe68798b9a1d4b2582edf8945dad5',
    ('kimi_native_k2_count_content_chunk1024', 'muse_spark'): 'a6caff65710f7b3614424b8948b00eaf534fe68798b9a1d4b2582edf8945dad5',
    ('kimi_native_k2_count_content_chunk1024', 'nemotron_v3'): '77d7452902311f94f1fd3535157f54d73ad7a29fb73e39b3f1edd8281228877d',
    ('kimi_native_k2_count_content_chunk1024', 'qwen3'): 'bf596ea1cc863811d03c31bc260ab82afe515bb8c70871d5861bb6dafd5e94a3',
    ('kimi_native_k2_count_content_chunk7', 'beam'): 'c83e9fa6488f624e0b069aa76cfb1631d20dc7b59668e8e382a9259a037a3c15',
    ('kimi_native_k2_count_content_chunk7', 'deepseek_v4'): 'a1a1eea2478a573becd17979489aff6c371bd76714cb859abb14d284cb366a78',
    ('kimi_native_k2_count_content_chunk7', 'deepseek_v41'): '76ee84720b69dbb075008f5c7bfad6d56782ce2a0cabf6d838902a4d9896a0b1',
    ('kimi_native_k2_count_content_chunk7', 'gemma4'): '56bb9e569638da2310a2d54c22afc140dd4cfd2bcc5a8206593562d68644cf14',
    ('kimi_native_k2_count_content_chunk7', 'glm47'): '7112ec689a5c0967a13f32de44a28c677a1ac167b7199da0a404ce7eab9398f2',
    ('kimi_native_k2_count_content_chunk7', 'gpt_oss'): 'bdfd7f661e804f6d280a853ff07795afbb357eb294ea61db8dd7bf733d479d10',
    ('kimi_native_k2_count_content_chunk7', 'kimi_k2'): '453b8f61e572f4861549176b3644474615adc19a64ca4325c255def06958ffda',
    ('kimi_native_k2_count_content_chunk7', 'kimi_k3'): 'bd377305b85e0c1388359555d1222fbd80415776117f98831ae2b5399677823e',
    ('kimi_native_k2_count_content_chunk7', 'minimax_m3'): '50daed6bb0c51372b021ec05459b776034ee0761d382bd4b346a1b25b9ef5af1',
    ('kimi_native_k2_count_content_chunk7', 'muse_glimmer'): '7d277c1c0a6acb08945430f43a9b6c07ef07c77217773ca82d0c2c16cc6b7d37',
    ('kimi_native_k2_count_content_chunk7', 'muse_spark'): '7d277c1c0a6acb08945430f43a9b6c07ef07c77217773ca82d0c2c16cc6b7d37',
    ('kimi_native_k2_count_content_chunk7', 'nemotron_v3'): '67b0e9e1256a4b6a413f5f05d8acd207cbf167b218845641418f2cd01625be58',
    ('kimi_native_k2_count_content_chunk7', 'qwen3'): '0eea1ab6e09c9b615ef35d8c179f02176cd1bdc61cf852900b24a26a6ffd7a7c',
    ('kimi_native_k3_json_count_content_chunk1024', 'kimi_k3'): 'e7c859585b979ed51d95220b450fca346f6b409cb66e0ad1e684fd749428af04',
    ('kimi_native_k3_json_count_content_chunk7', 'kimi_k3'): '5bfd5c03c6c6b2c5f70ea64cf52680589d557dec63ed0e12dae7617c997347c4',
    ('kimi_native_k3_typed_count_content_chunk1024', 'kimi_k3'): '9f5ea2549536714006dc2d407675bb691c689fec016291a39abb48c3c21f2722',
    ('kimi_native_k3_typed_count_content_chunk7', 'kimi_k3'): 'c97da62bbe306731e6218d8095f0471b205fbb664cf75841ee093f21d636d687',
    ('kimi_native_rejected_next_header_progress', 'beam'): '3aeb8e3dfaec75097c515dd64e54aca1639103e1dbd615fbfe8f9cf784d11130',
    ('kimi_native_rejected_next_header_progress', 'kimi_k3'): 'edba52e4319f2ec80c17d9a4f6823cb0e583d47654a932fe7dc622c8a0a53494',
    ('kimi_native_trailing_syntax_whitespace', 'deepseek_v4'): '06e0e8688c6d398c26e7036df1b55c972b2e8b507a25fc2ab2163cb9843108f1',
    ('kimi_native_trailing_syntax_whitespace', 'deepseek_v41'): '0fd0fe2cd454b3aaf939f00b1a6b85ce244f3102d519b741a66bdd3fe1d648f7',
    ('kimi_native_trailing_syntax_whitespace', 'gemma4'): '43f743be26dee6c06c45058ed0a4f22fa9fb69fe211a2395378a8b65f9f82dd8',
    ('kimi_native_trailing_syntax_whitespace', 'gpt_oss'): '59aba168cc91d5d92b7b56639e02e008ce2143b475ec3f174a8fbc771482de2b',
    ('kimi_native_trailing_syntax_whitespace', 'kimi_k2'): '1002833b55e0a8330d11409f118b35f9b34c9ecb4250130383ce9296d5716d1c',
    ('kimi_native_trailing_syntax_whitespace', 'kimi_k3'): 'cd96ba55e815d9fefb2ff30c924a53b0086ea2c69c455a2b96acf5248d698612',
    ('kimi_native_valid_object_trailing_garbage', 'deepseek_v4'): '8d3bce6fe53366a7c43ee2980fcef7d2a3a2e02b771aa749568c35b86a251af7',
    ('kimi_native_valid_object_trailing_garbage', 'deepseek_v41'): 'ffe7ca494f09604c56d75cf0ec0c103549e078a4c51038c241c9c2ec7b9639b0',
    ('kimi_native_valid_object_trailing_garbage', 'gemma4'): 'b63d5ff27dee21d7a757342f3da105f6902d3dd6eb058b368129ca226aadc290',
    ('kimi_native_valid_object_trailing_garbage', 'gpt_oss'): '10137e1dfd0a66e4ff222d3d211f596b77de93401eb8329fdc577b5de54cd622',
    ('kimi_native_valid_object_trailing_garbage', 'kimi_k2'): '816b4ca4842a7b3c3fa2a0f0ac031c0f886113580d9a74f8dd5e49b18dbfaebd',
    ('kimi_native_valid_object_trailing_garbage', 'kimi_k3'): 'dba37f2bf499646e3d456c0caaa7e434c24c1a58c0c9c8d9595fe499131aa106',
}


def _kimi_native_scope(scenario):
    if scenario.startswith(("kimi_native_k3_typed_", "kimi_native_k3_json_")):
        return {"kimi_k3"}
    if "_count_content_chunk" in scenario or scenario.endswith("interfield_whitespace"):
        return set(NATIVE_FAMILIES)
    if scenario.endswith("rejected_next_header_progress"):
        return {"beam", "kimi_k3"}
    assert scenario in {
        "kimi_native_trailing_syntax_whitespace", "kimi_native_valid_object_trailing_garbage",
        "kimi_native_incompatible_string_recovery",
    }
    return {"kimi_k2", "kimi_k3", "gpt_oss", "deepseek_v4", "deepseek_v41", "gemma4"}


def _assert_kimi_native_scopes(corpus):
    for scenario in UNIFIED_TAX:
        if not scenario.startswith("kimi_native_"):
            continue
        expected = _kimi_native_scope(scenario)
        assert set(G.scenario_families(scenario)) == expected, (scenario, "independent native applicability")
        present = {family for family, cases in corpus.items() if f"UNIFIED.{scenario}.{family}" in cases}
        assert present == expected, (scenario, "missing applicable native child", expected - present)
        assert taxonomy.case_fold(scenario) is None, (scenario, "native delivery identities must remain explicit")
        for family in expected:
            _assert_kimi_native_golden(family, scenario, corpus[family][f"UNIFIED.{scenario}.{family}"])


def _assert_kimi_native_golden(family: str, scenario: str, case: dict) -> None:
    request = {"input": case["input"], "init": case["init"], "tools": case["tools"],
               "finish_reason": case["finish_reason"],
               "chunks": [{"delta_text": chunk} for chunk in case["input_chunks"]]}
    digest = hashlib.sha256(json.dumps(request, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    assert digest == _KIMI_NATIVE_REQUEST_DIGESTS[(scenario, family)], (family, scenario, "independent native request contract")
    grammar = G.unified_grammar_family(family)
    if "_count_content_chunk" in scenario:
        arguments = {"count": 7, "content": "q" * 4096}
    elif scenario.endswith("interfield_whitespace"):
        arguments = {"count": 7, "content": "q"}
    elif scenario.endswith("rejected_next_header_progress"):
        marker = ("</rfl:parameter> <rfl:parameter gibberish>" if grammar == "beam" else
                  "<|close|>argument<|sep|> <|open|>argument gibberish<|sep|>")
        arguments = {"count": 7, "content": "head" + marker + "q" * 3072}
    elif scenario.endswith("valid_object_trailing_garbage") and grammar == "kimi_k2":
        arguments = {"count": 7, "content": "q"}
    elif grammar == "kimi_k2" or (grammar == "deepseek_v4" and not scenario.endswith("incompatible_string_recovery")):
        arguments = {}
    else:
        assert grammar in {"gpt_oss", "kimi_k3", "deepseek_v4", "deepseek_v41", "gemma4"}
        assert case["golden"] == [], (family, scenario, "strict native malformed-call policy")
        return
    assert case["golden"] == [{"kind": "tool_call", "name": "write_file", "arguments": arguments}], (
        family, scenario, "native Kimi golden differs from independent contract")


def test_kimi_native_scopes_have_independent_positive_evidence():
    _assert_kimi_native_scopes({family: build_cases(family) for family in NATIVE_FAMILIES})


@pytest.mark.parametrize("mutation", ["omitted", "exclusion", "input_payload", "golden_payload", "folded_child"])
def test_kimi_native_contract_rejects_omissions_exclusions_and_payload_drift(monkeypatch, mutation):
    corpus = {family: build_cases(family) for family in NATIVE_FAMILIES}
    scenario = "kimi_native_k2_count_content_chunk7"
    family = "qwen3"
    key = f"UNIFIED.{scenario}.{family}"
    if mutation == "input_payload":
        corpus[family][key]["input"] = corpus[family][key]["input"].replace("q" * 4096, "x" * 4096)
    elif mutation == "golden_payload":
        corpus[family][key]["golden"][0]["arguments"]["count"] = 8
    else:
        del corpus[family][key]
        if mutation == "exclusion":
            original = G.scenario_families
            monkeypatch.setattr(G, "scenario_families", lambda name: original(name) - {family} if name == scenario else original(name))
        elif mutation == "folded_child":
            parent = "kimi_native_k2_count_content_chunk1024"
            assert f"UNIFIED.{parent}.{family}" in corpus[family]
            original_fold = taxonomy.case_fold
            monkeypatch.setattr(taxonomy, "case_fold", lambda name: (parent, "delivery") if name == scenario else original_fold(name))
    with pytest.raises(AssertionError):
        _assert_kimi_native_scopes(corpus)

@pytest.mark.parametrize("family,scenario", [
    (family, name) for name in UNIFIED_TAX if name.startswith("kimi_native_")
    for family in G.scenario_families(name)
])
@pytest.mark.parametrize("mutation", ["empty", "name", "arguments"])
def test_kimi_native_oracle_rejects_changed_goldens(family, scenario, mutation):
    case = build_cases(family)[f"UNIFIED.{scenario}.{family}"]
    _assert_input_carries_events(family, scenario, case)
    if case["golden"]:
        if mutation == "empty":
            case["golden"] = []
        elif mutation == "name":
            case["golden"][0]["name"] = "invented"
        else:
            case["golden"][0]["arguments"] = {"count": 8, "content": "wrong"}
    else:
        case["golden"] = [{"kind": "tool_call", "name": "write_file", "arguments": {}}]
    with pytest.raises(AssertionError):
        _assert_input_carries_events(family, scenario, case)


@pytest.mark.parametrize("form", ["typed", "json"])
@pytest.mark.parametrize("cadence", [7, 1024])
def test_k3_native_representation_axes_stay_distinct_only_in_k3(form, cadence):
    scenario = f"kimi_native_k3_{form}_count_content_chunk{cadence}"
    assert G.scenario_families(scenario) == {"kimi_k3"}
    shared = f"kimi_native_k2_count_content_chunk{cadence}"
    assert set(G.scenario_families(shared)) == set(NATIVE_FAMILIES)
    for family in NATIVE_FAMILIES:
        case = build_cases(family)[f"UNIFIED.{shared}.{family}"]
        _assert_kimi_native_golden(family, shared, case)
        if family != "kimi_k3":
            assert f"UNIFIED.{scenario}.{family}" not in build_cases(family)



def _assert_gpt58_native_contracts(corpus):
    scopes = {
        "gpt_oss_control_marker_inside_argument_string": {"beam", "deepseek_v4", "deepseek_v41", "gpt_oss", "kimi_k2", "kimi_k3", "muse_glimmer", "muse_spark"},
        "gpt_oss_quoted_message_header_literal": {"beam", "gpt_oss", "kimi_k3", "muse_glimmer", "muse_spark"},
        "gpt_oss_quoted_truncated_tool_then_final": {"beam", "gpt_oss", "kimi_k3", "muse_glimmer", "muse_spark"},
    }
    for scenario, expected in scopes.items():
        assert set(G.scenario_families(scenario)) == expected, (scenario, "independently justified native applicability")
        present = {family for family, cases in corpus.items() if f"UNIFIED.{scenario}.{family}" in cases}
        assert present == expected, (scenario, "missing applicable folded child", expected - present)
    for scenario, family in GPT58_NATIVE_REQUEST_DIGESTS:
        case = corpus[family][f"UNIFIED.{scenario}.{family}"]
        assert_gpt58_native_contract(family, scenario, case)
        parent = "tool_no_close" if scenario == "gpt_oss_quoted_truncated_tool_then_final" else "arg_marker_in_string"
        assert taxonomy.case_fold(scenario)[0] == parent
        assert f"UNIFIED.{parent}.{family}" in corpus[family], (family, scenario, "missing folded parent")


def test_gpt58_native_marker_and_header_omissions_have_independent_contracts():
    _assert_gpt58_native_contracts({family: build_cases(family) for family in NATIVE_FAMILIES})


@pytest.mark.parametrize("scenario,family", sorted(GPT58_NATIVE_REQUEST_DIGESTS))
@pytest.mark.parametrize("mutation", ["omitted", "exclusion", "input_payload", "golden_payload", "folded_child"])
def test_gpt58_native_contracts_reject_omissions_exclusions_and_payload_drift(monkeypatch, scenario, family, mutation):
    corpus = {f: build_cases(f) for f in NATIVE_FAMILIES}
    key = f"UNIFIED.{scenario}.{family}"
    case = corpus[family][key]
    if mutation == "input_payload":
        case["input"] += "incidental prose"
    elif mutation == "golden_payload":
        case["golden"].append({"kind": "text", "text": "incidental prose"})
    else:
        del corpus[family][key]
        if mutation == "exclusion":
            original = G.scenario_families
            monkeypatch.setattr(G, "scenario_families", lambda name: original(name) - {family} if name == scenario else original(name))
        elif mutation == "folded_child":
            parent = taxonomy.case_fold(scenario)[0]
            assert f"UNIFIED.{parent}.{family}" in corpus[family]
    with pytest.raises(AssertionError):
        _assert_gpt58_native_contracts(corpus)


@pytest.mark.parametrize("family", ["muse_glimmer", "muse_spark"])
def test_gpt58_muse_native_json_scalar_keeps_parameter_closers_literal(family):
    scenario = "gpt_oss_control_marker_inside_argument_string"
    case = build_cases(family)[f"UNIFIED.{scenario}.{family}"]
    assert_gpt58_native_contract(family, scenario, case)
    assert _native_input_calls(family, case["input"], []) == case["golden"]
    value = _family_value(scenario, family)
    assert "</atem:parameter>" in value
    assert json.dumps(value) in case["input"]


_MUSE_NATIVE_CONTRACTS = {
    ('muse_opaque_declared_invoke', 'beam'): '96851d54a62d9ed66be652e2dd58571ad873e70ff88eb08bd173f6cba961269e',
    ('muse_opaque_declared_invoke', 'deepseek_v4'): 'f3654a5848e2a791bc9ecb006bc30d496907c67fa66199e14077f393472b510f',
    ('muse_opaque_declared_invoke', 'deepseek_v41'): '0633ddc299de20efbc80a86269b1a4582a65cf28e9092a7d5e276b631d38e8ef',
    ('muse_opaque_declared_invoke', 'glm47'): '17c0833963ff0f58fef9bfda5aec731ade8817af908348fd7f7faf4e95e04c2d',
    ('muse_opaque_declared_invoke', 'gpt_oss'): 'd715e727867de0277d5e49585808852a288af81e5ad2d8124be02bdc7627aa6a',
    ('muse_opaque_declared_invoke', 'kimi_k2'): 'ac52c60f1a9ef8ca5a7c6bc34aa6d9da038b79351b5c5f3b844c8b9873102854',
    ('muse_opaque_declared_invoke', 'kimi_k3'): '7407a8654816a585f6f85839e2f107e39a5070e4250b86d156391b1a4e0fe405',
    ('muse_opaque_declared_invoke', 'minimax_m3'): '20f6d71b536c25c46974e1d1a978384d463ab6adfa1e27516cad38d4c5e7967c',
    ('muse_opaque_declared_invoke', 'muse_glimmer'): '2c127aa844bdff11a0e8f3acee47584a9a39638f5dee71ea8a455a1c64665808',
    ('muse_opaque_declared_invoke', 'muse_spark'): '2c127aa844bdff11a0e8f3acee47584a9a39638f5dee71ea8a455a1c64665808',
    ('muse_opaque_declared_invoke', 'nemotron_v3'): 'd4d553c7346854470314475e3506100435cbf185f82fa3fcc496ef1c6559cc1f',
    ('muse_opaque_declared_invoke', 'qwen3'): '60e14d81d8db60b0541c99fbe571756bcda764349bb5286ebf5ae57d6cf827b1',
    ('muse_opaque_undeclared_invoke', 'beam'): '2bc05791b1333beebc637734beda160e9149c2cb82701ac28b1f7f8dc727100e',
    ('muse_opaque_undeclared_invoke', 'deepseek_v4'): '5ba1fdfce10541e66f8c951d2c581a363582236bed098d3bdd170aa7793045df',
    ('muse_opaque_undeclared_invoke', 'deepseek_v41'): 'b7d3f093b83166c3b89bc91e3b8694f4c8424f9c33ada4cb87446d3205e19efb',
    ('muse_opaque_undeclared_invoke', 'glm47'): '29d70036e3872f39307e6fcf52c2dbcd0df9cf38cf4e8b4a1d3ad01f0b74c1f0',
    ('muse_opaque_undeclared_invoke', 'gpt_oss'): '429643a388cdde2372fe6a853e982d4ccd3301fd6a57f55be0cc0511760f00d2',
    ('muse_opaque_undeclared_invoke', 'kimi_k2'): 'e5759450ff9d4d2f9d45abc0df81bf7c6bb427021120b7dfbd5757bf626b2a96',
    ('muse_opaque_undeclared_invoke', 'kimi_k3'): '801d4e1154614ba9a5374a58948e8e1b4c4c4aac71e451b89b9e40a573a1c47a',
    ('muse_opaque_undeclared_invoke', 'minimax_m3'): '7f5f30d845de9df342e70c1f34233353558fec4becb45e5b48cb322d54836824',
    ('muse_opaque_undeclared_invoke', 'muse_glimmer'): '4be03037eecf0c48e572a3d485a81eca5754c15b937bc039eecfc7300f07518d',
    ('muse_opaque_undeclared_invoke', 'muse_spark'): '4be03037eecf0c48e572a3d485a81eca5754c15b937bc039eecfc7300f07518d',
    ('muse_opaque_undeclared_invoke', 'nemotron_v3'): '81916f693fa23cfe1cf4a3dc8afb96ec079fa2b7a629760f175490bdf824f3b5',
    ('muse_opaque_undeclared_invoke', 'qwen3'): '016095abef86c16301ccf634fe07f0c41039f13df8e1b82efb8282ea361634eb',
    ('muse_recovery_second_parameter', 'beam'): 'd6ff8c6990fcbe6be8ecee9d39c3bfc2fb52f8a0008f375089e766a0e77df2bc',
    ('muse_recovery_second_parameter', 'deepseek_v4'): '57e85959ed8347538f3f32aefec5066db5ee82357b2b1309532fe51b68e0e2ef',
    ('muse_recovery_second_parameter', 'deepseek_v41'): '72bf833e7b8bd00e678e80a017f5a1f9aef991c47fa749e8f3e951b367a60e2c',
    ('muse_recovery_second_parameter', 'glm47'): '346b9503e0add7ffad3530e3158069185329f50666cd3a313a67b9106fb28c94',
    ('muse_recovery_second_parameter', 'gpt_oss'): 'c5ffe3faa0e17db8cda4f322f87854f80a57d037d57da2f448247c07606e95a9',
    ('muse_recovery_second_parameter', 'kimi_k2'): '9a730b4bbd85b71154f80929d53c31dae04c0613f68dfeb3a042c2078e96b1e1',
    ('muse_recovery_second_parameter', 'kimi_k3'): '2f0492cfec8d757ad845b21f14e2dfce9247bd19d4d9d31e347618ae407e3685',
    ('muse_recovery_second_parameter', 'muse_glimmer'): '6c19ad62908ef40301b2a95a4d38aec46b26e528d32724ebe3ba0f2108e9938a',
    ('muse_recovery_second_parameter', 'muse_spark'): '6c19ad62908ef40301b2a95a4d38aec46b26e528d32724ebe3ba0f2108e9938a',
    ('muse_recovery_second_parameter', 'nemotron_v3'): '0de45f2659c72de72da4c7ef675d7120bca00af880a513e14e140518d037534f',
    ('muse_recovery_second_parameter', 'qwen3'): '77572f0b3f8cb2c8e461cfabef6b31e95579d8b5877b53c75f771f669774fd4c',
    ('muse_unterminated_parameter_then_call', 'muse_glimmer'): 'dc3d4055362b37907ae3e7b27b76220fa5c7fd44ab38753792c4cf523b4f6274',
    ('muse_unterminated_parameter_then_call', 'muse_spark'): 'dc3d4055362b37907ae3e7b27b76220fa5c7fd44ab38753792c4cf523b4f6274',
    ('muse_unterminated_parameter_then_empty_call', 'beam'): '3c0738ef369a4a8aad46e9cee3498bd1993ebbaa56c2042dda86bbcf6cefeb9a',
    ('muse_unterminated_parameter_then_empty_call', 'deepseek_v4'): '83421799b98b4a420cc61fade96d909de6581bfa9a80930172461e53935692d0',
    ('muse_unterminated_parameter_then_empty_call', 'deepseek_v41'): '46bbeeb5560caf336419b8563b1843759dab493710abf922e11e351d3a45706d',
    ('muse_unterminated_parameter_then_empty_call', 'gemma4'): '565a48283e953196d1d7d18827e181a908a415f52066aa9ad930ac5e4f333e92',
    ('muse_unterminated_parameter_then_empty_call', 'glm47'): 'd208189e20447671b7cfbedb6c30fd77ab5b20c4f54e7132c65d12948f1a3a86',
    ('muse_unterminated_parameter_then_empty_call', 'gpt_oss'): '9f189fbb98dc61c6cab75b694de3d40fbd8dc53d3e4b8301a95e994fcc894489',
    ('muse_unterminated_parameter_then_empty_call', 'kimi_k2'): '60c7b1735c6c672ff5db933cbe138ae5e20eac68a807883aa6df197c514731e3',
    ('muse_unterminated_parameter_then_empty_call', 'kimi_k3'): '53fb539318c55838615a8bd77b03adb4f909e4d5cc7bcd747ad35fab6536b069',
    ('muse_unterminated_parameter_then_empty_call', 'minimax_m3'): '24af3d0d6abdcf88097ffc49d79f77f50f44c963c8fe0007c64abbc10a04aab2',
    ('muse_unterminated_parameter_then_empty_call', 'muse_glimmer'): '78a76812c2a5487511c17281d7d2718760407dd04504bef5a09e675869c1e90d',
    ('muse_unterminated_parameter_then_empty_call', 'muse_spark'): '78a76812c2a5487511c17281d7d2718760407dd04504bef5a09e675869c1e90d',
    ('muse_unterminated_parameter_then_empty_call', 'nemotron_v3'): '9b2f0eada5e20c54231bdb227f27b7a678b469459782be51a28b3e265abb10d3',
    ('muse_unterminated_parameter_then_empty_call', 'qwen3'): 'd5ef059a8abdb7c00051c1b59d2abccb2d0c3b16bb17f54418a5e9ba943d3555',
    ('muse_unterminated_reasoning_guided', 'beam'): 'bf0c6ee36f6847aa798afb31e123e222d53dd44f8768c82fc2803adcd631e224',
    ('muse_unterminated_reasoning_guided', 'deepseek_v4'): '67347c3c7fce4a2034d7bd816fc24d2f22f8af02636cbfab3f220db149f06f79',
    ('muse_unterminated_reasoning_guided', 'deepseek_v41'): '093d5515ccddb8362ab82e02cad92f8e774fc91ee38643e879c6cf65c800d4c3',
    ('muse_unterminated_reasoning_guided', 'gemma4'): 'f251a7e891c9921d8aa709893210cd1857e68db6012009f2ad0c7326529fa660',
    ('muse_unterminated_reasoning_guided', 'glm47'): '4a8ada03a9e9bf8e82643db6c7d0b3b720780e8655cab35490b429165b823f3d',
    ('muse_unterminated_reasoning_guided', 'gpt_oss'): '7839327129cd7ca4fbfc536d4e18aceacc7d5654f03e6468a2f7cb68ad21771a',
    ('muse_unterminated_reasoning_guided', 'kimi_k2'): 'a967829398fccfac47e98c19d6dc2a01bcb58b0416b619e735cf65cbc3b1477f',
    ('muse_unterminated_reasoning_guided', 'kimi_k3'): 'c50aca7e4c54cb63d7f524e7fd81c28c3831c569696d8c02a74c62def838eefa',
    ('muse_unterminated_reasoning_guided', 'minimax_m3'): 'eb1dcc372cb85293a0a7c7ae87dd16b8f4a65cdb38013da7bd1e9399bba403bc',
    ('muse_unterminated_reasoning_guided', 'muse_glimmer'): '8ecccd2be10b1bff3de9cb6686c3f13204793135dcfac34774d93437d49316c1',
    ('muse_unterminated_reasoning_guided', 'muse_spark'): '8ecccd2be10b1bff3de9cb6686c3f13204793135dcfac34774d93437d49316c1',
    ('muse_unterminated_reasoning_guided', 'nemotron_v3'): '37f03498d86c33fa5036c450bbc887a81f3cbfe7912635f86993105b9bb0875f',
    ('muse_unterminated_reasoning_guided', 'qwen3'): 'a786cd9249410f23ae88e92238e91dd9d6bf90f548cb722f5d3ceba3da7652ca',
    ('muse_unterminated_reasoning_native', 'gemma4'): '9383cd55d5f58f4b1a952eb9b28c7171bd06e52f17b06ad279313f23ec50f445',
    ('muse_unterminated_reasoning_native', 'muse_glimmer'): 'ee2d156ab503f92c3cacd39e023e3d2af6831e021a9f6b075c3fe4cd24105d79',
    ('muse_unterminated_reasoning_native', 'muse_spark'): 'ee2d156ab503f92c3cacd39e023e3d2af6831e021a9f6b075c3fe4cd24105d79',
}


def _muse_native_scope(scenario):
    if scenario in {"muse_opaque_declared_invoke", "muse_opaque_undeclared_invoke"}:
        return set(NATIVE_FAMILIES) - {"gemma4"}
    if scenario == "muse_recovery_second_parameter":
        return set(NATIVE_FAMILIES) - {"gemma4", "minimax_m3"}
    if scenario == "muse_unterminated_parameter_then_call":
        return {"muse_glimmer", "muse_spark"}
    if scenario == "muse_unterminated_parameter_then_empty_call":
        return set(NATIVE_FAMILIES)
    if scenario == "muse_unterminated_reasoning_native":
        return {"muse_glimmer", "muse_spark", "gemma4"}
    assert scenario == "muse_unterminated_reasoning_guided"
    return set(NATIVE_FAMILIES)


def _assert_muse_native_case(family, scenario, case):
    request = {"input": case["input"], "init": case["init"], "tools": case["tools"],
               "finish_reason": case["finish_reason"],
               "chunks": [{"delta_text": chunk} for chunk in case.get("input_chunks", [case["input"]])]}
    contract = {**_without_synthetic_finish_step(request), "golden": case["golden"]}
    digest = hashlib.sha256(json.dumps(contract, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    assert digest == _MUSE_NATIVE_CONTRACTS[(scenario, family)], (family, scenario, "independent native Muse contract")
    assert case["tools"] == G.unified_tools(), (family, scenario, "original seven-tool request")


def _assert_muse_native_scopes(corpus):
    for scenario in {scenario for scenario, _family in _MUSE_NATIVE_CONTRACTS}:
        expected = _muse_native_scope(scenario)
        assert set(G.scenario_families(scenario)) == expected, (scenario, "independent native Muse scope")
        present = {family for family, cases in corpus.items() if f"UNIFIED.{scenario}.{family}" in cases}
        assert present == expected, (scenario, "missing applicable native Muse child", expected - present)
        assert taxonomy.case_fold(scenario) is None, (scenario, "published native identities remain explicit")
        for family in expected:
            _assert_muse_native_case(family, scenario, corpus[family][f"UNIFIED.{scenario}.{family}"])


def test_muse_native_scopes_and_independent_contracts():
    _assert_muse_native_scopes({family: build_cases(family) for family in NATIVE_FAMILIES})


@pytest.mark.parametrize("mutation", ["omitted", "exclusion", "input_payload", "golden_payload", "folded_child", "unsupported", "alias_duplicate"])
def test_muse_native_contract_rejects_scope_and_payload_mutations(monkeypatch, mutation):
    corpus = {family: build_cases(family) for family in NATIVE_FAMILIES}
    scenario = "muse_opaque_declared_invoke"
    family = "qwen3"
    key = f"UNIFIED.{scenario}.{family}"
    original_scope = G.scenario_families
    if mutation == "input_payload":
        corpus[family][key]["input"] = corpus[family][key]["input"].replace("doc: ", "incidental: ")
        corpus[family][key]["input_chunks"] = list(corpus[family][key]["input"])
    elif mutation == "golden_payload":
        corpus[family][key]["golden"][0]["arguments"]["x"] = "changed"
    elif mutation in {"unsupported", "alias_duplicate"}:
        family = "gemma4" if mutation == "unsupported" else "qwen3"
        target = scenario if mutation == "unsupported" else "muse_unterminated_parameter_then_call"
        corpus[family][f"UNIFIED.{target}.{family}"] = deepcopy(corpus["qwen3"][key])
        monkeypatch.setattr(G, "scenario_families", lambda name: original_scope(name) | {family} if name == target else original_scope(name))
    else:
        del corpus[family][key]
        if mutation == "exclusion":
            monkeypatch.setattr(G, "scenario_families", lambda name: original_scope(name) - {family} if name == scenario else original_scope(name))
        elif mutation == "folded_child":
            assert f"UNIFIED.muse_opaque_undeclared_invoke.{family}" in corpus[family]
            original_fold = taxonomy.case_fold
            monkeypatch.setattr(taxonomy, "case_fold", lambda name: ("muse_opaque_undeclared_invoke", "declared") if name == scenario else original_fold(name))
    with pytest.raises(AssertionError):
        _assert_muse_native_scopes(corpus)


def test_shared_muse_cases_have_generic_display_groups_without_changing_ids():
    groups = {"muse_opaque_declared_invoke": 12, "muse_opaque_undeclared_invoke": 12,
              "muse_recovery_second_parameter": 12, "muse_unterminated_parameter_then_empty_call": 5,
              "muse_unterminated_reasoning_native": 12, "muse_unterminated_reasoning_guided": 34}
    for scenario, group in groups.items():
        assert taxonomy.display_tax(scenario)[0] == group
        assert numbered_id(scenario) == "UNIFIED.muse-" + scenario.removeprefix("muse_")



def _assert_m3_nested_marker_contract(corpus):
    scenario = "kimi_k3_typed_json_marker_payload"
    # The original inspect schema declares x.inner as string. M3's native
    # nested tags therefore preserve sibling closing/opening tags as data.
    assert set(G.scenario_families(scenario)) == set(NATIVE_FAMILIES)
    for family in NATIVE_FAMILIES:
        assert f"UNIFIED.{scenario}.{family}" in corpus[family], (family, scenario, "missing applicable nested marker child")
    case = corpus["minimax_m3"][f"UNIFIED.{scenario}.minimax_m3"]
    request = {"input": case["input"], "init": case["init"], "tools": case["tools"],
               "finish_reason": case["finish_reason"],
               "chunks": [{"delta_text": c} for c in case.get("input_chunks", [case["input"]])]}
    contract = {**_without_synthetic_finish_step(request), "golden": case["golden"]}
    digest = hashlib.sha256(json.dumps(contract, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    assert digest == '91c638e2a236bd4f852772de9e333d977d3e575c15eeffcef541bc47c9082013', "independent M3 nested marker contract"


def test_m3_nested_marker_contract_and_scope():
    _assert_m3_nested_marker_contract({family: build_cases(family) for family in NATIVE_FAMILIES})


@pytest.mark.parametrize("mutation", ["omitted", "exclusion", "input_payload", "golden_payload"])
def test_m3_nested_marker_contract_rejects_mutations(monkeypatch, mutation):
    corpus = {family: build_cases(family) for family in NATIVE_FAMILIES}
    scenario = "kimi_k3_typed_json_marker_payload"
    key = f"UNIFIED.{scenario}.minimax_m3"
    if mutation in {"omitted", "exclusion"}:
        del corpus["minimax_m3"][key]
        if mutation == "exclusion":
            original = G.scenario_families
            monkeypatch.setattr(G, "scenario_families", lambda name: original(name) - {"minimax_m3"} if name == scenario else original(name))
    elif mutation == "input_payload":
        corpus["minimax_m3"][key]["input"] += "incidental"
        corpus["minimax_m3"][key]["input_chunks"] = list(corpus["minimax_m3"][key]["input"])
    else:
        corpus["minimax_m3"][key]["golden"][0]["arguments"]["x"]["inner"] = "changed"
    with pytest.raises(AssertionError):
        _assert_m3_nested_marker_contract(corpus)
