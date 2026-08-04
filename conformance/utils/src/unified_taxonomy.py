# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Single source of the UNIFIED case taxonomy: scenario slug -> numbered id
(UNIFIED.<group>.<sub>) and the per-group axis labels. Shared by the fixture
exploder (names case files by number) and the conformance generator (renders the
group labels), so the numbering can't drift between them.
Groups 1-9 mirror the tool-calling STREAM taxonomy (TOOLCALLING.streamv2.N) as
tool-only unified cases (UNIFIED subsumes STREAM). Group 10 is the reasoning axis
(REASONING.*). Group 11 is unique to unified: reasoning<->tool interleaving that
neither STREAM (no reasoning) nor REASONING (no ordered tool events) can express.
Group 12 is adversarial nesting (a marker of one channel inside another).
Request-scoped modes use PAIRED TENS: X0 is that mode's happy base case, X1 its
weird/malformed counterpart — 30/31 guided decoding, 40/41 prefilled reasoning,
50/51 prefilled response. A new mode takes the next ten.
There is no separate "input stream mode" axis: which channel the prompt pre-opened IS
`init.starting_state`, so a case that varied only that duplicated groups 31-33, and one that
varied nothing but a finish_reason label duplicated groups 1-12 (the parser cannot see
that value — `finish()` takes no argument).
"""

import yaml

import markers

UNIFIED_TAX = {
    # Group 1 — Single call
    "tool_only": (1, "a"),
    # Group 2 — Multiple calls (streamv2.2)
    "two_calls": (2, "a"), "two_calls_same_name": (2, "b"),
    # Group 3 — No call (streamv2.3)
    "text_only": (3, "a"),
    # Group 4 — Malformed envelope. Labelled but EMPTY until now.
    "tool_block_never_closed_then_text": (4, "a"),
    "tool_markup_only_emits_nothing": (4, "b"),

    # Group 5 — Truncation / recovery (streamv2.5)
    "truncated_tool_eof": (5, "a"), "tool_no_close": (5, "b"),
    "orphan_close_after_prose": (5, "c"),
    # Group 6 — Empty body (streamv2.6)
    "empty_args": (6, "a"),
    # Group 7 — Argument fidelity (streamv2.7)
    "arg_unicode": (7, "a"), "arg_marker_in_string": (7, "b"),
    # Group 8 — Content / narration position (streamv2.8)
    "text_before_tool": (8, "a"), "trailing_text_after_tool": (8, "b"),
    "text_sandwich": (8, "c"), "text_between_calls": (8, "d"),
    "narrated_calls": (8, "e"),
    # Group 10 — Reasoning span (REASONING.*), reasoning-only
    "reason_only": (10, "a"), "reason_then_content": (10, "b"),
    "two_reason_spans": (10, "c"), "reason_unterminated": (10, "d"),
    # Group 11 — Reasoning <-> tool interleaving (UNIQUE to unified)
    "reason_then_tool": (11, "a"), "reason_after_tool": (11, "b"),
    "reason_interleaved": (11, "c"), "reason_tool_text_reason_tool": (11, "d"),
    "interstitial_text": (11, "e"), "content_then_reason_then_tool": (11, "f"),
    "content_then_reason": (11, "g"), "reason_tool_reason_tool_reason": (11, "h"),
    "reason_between_calls": (11, "i"), "text_reason_tool_text_reason_tool": (11, "j"),
    # Group 12 — Adversarial nesting (a marker of one channel inside another)
    "reason_markup_in_arg": (12, "a"), "tool_in_reason": (12, "b"),
    "reason_markup_in_arg_with_text": (12, "c"), "tool_in_reason_with_text": (12, "d"),
    # --- Request-scoped modes. Each axis is a PAIR of tens: X0 is the happy base
    # case for that mode, X1 is the weird / malformed one. A new axis takes the
    # next ten, so "which mode" and "well-formed or not" stay independent and a
    # malformed case can never be mistaken for the mode's baseline.
    # Group 30 — Guided decoding, happy
    "guided_json_named_tool": (30, "a"), "guided_json_required_tool": (30, "b"),
    "guided_json_two_calls": (30, "c"),
    "guided_json_escaped_string_args": (30, "d"), "guided_json_array_argument": (30, "e"),
    "guided_json_after_reasoning": (30, "f"), "guided_json_marker_inside_argument": (30, "g"),

    # Group 31 — Guided decoding, weird / malformed
    "guided_json_invalid_call": (31, "a"), "guided_json_malformed_json": (31, "b"),
    "guided_json_partial_calls": (31, "c"),
    "guided_json_list_with_broken_element": (31, "d"),
    # 31.e-31.k — the SURROUNDINGS of a guided payload, not the payload itself.
    "guided_json_tool_open_before_payload": (31, "e"),
    "guided_json_tool_close_after_payload": (31, "f"),
    "guided_json_wrapped_in_tool_markup": (31, "g"),
    "guided_json_narrated_invoke_in_reasoning": (31, "h"),
    "guided_json_prose_before_reasoning": (31, "i"),
    "guided_json_orphan_reason_close_before_payload": (31, "j"),
    "guided_json_orphan_tool_close_before_payload": (31, "k"),
    # Generated crossings (`_guided_product` in gen_unified_golden.py): payload
    # shape x surrounding grammar. The 31.* rows are the quadrant that had ZERO
    # cases — markup present AND no call recoverable — where both the P2 recovery
    # leak and the unbounded invoke-header scan lived.
    "guided_json_valid_trailing_close": (30, "h"),
    "guided_json_valid_wrapped": (30, "i"),
    "guided_json_valid_bare_opener": (30, "j"),
    "guided_json_syntax_error_trailing_close": (31, "l"),
    "guided_json_syntax_error_wrapped": (31, "m"),
    "guided_json_syntax_error_bare_opener": (31, "n"),
    "guided_json_schema_error_not_a_call_trailing_close": (31, "o"),
    "guided_json_schema_error_not_a_call_wrapped": (31, "p"),
    "guided_json_schema_error_not_a_call_bare_opener": (31, "q"),
    "guided_json_schema_error_nameless_element_trailing_close": (31, "r"),
    "guided_json_schema_error_nameless_element_wrapped": (31, "s"),
    "guided_json_schema_error_nameless_element_bare_opener": (31, "t"),

    # Devin-found crossings, added as AXIS entries so the next payload/surrounding
    # combination is generated rather than noticed later.
    "guided_json_gt_in_argument_trailing_close": (30, "k"),
    "guided_json_gt_in_argument_wrapped": (30, "l"),
    "guided_json_gt_in_argument_bare_opener": (30, "m"),

    # Group 40 — Prefilled reasoning, happy
    "prefilled_reasoning_with_tool": (40, "a"), "prefilled_reasoning_with_guided_json": (40, "b"),
    "prefilled_reasoning_then_text_then_tool": (40, "c"), "prefilled_reasoning_then_text": (40, "d"),
    # Group 41 — Prefilled reasoning, weird / malformed
    "prefilled_reasoning_redundant_opener": (41, "a"), "prefilled_reasoning_truncated": (41, "b"),
    # Group 50 — Prefilled response, happy
    "prefilled_response_with_tool": (50, "a"), "prefilled_response_with_guided_json": (50, "b"),
    "prefilled_response_guided_json_two_calls": (50, "c"),
    "prefilled_response_reasoning_markers_literal": (50, "d"),
    # Group 51 — Prefilled response, weird / malformed
    "prefilled_response_truncated": (51, "a"),
    "prefilled_response_guided_json_partial_calls": (51, "b"),
}

# Axis prefix makes each group's channel explicit: "TC" = tool-calling only (groups
# 1-9 mirror the tool STREAM suite), "Reasoning" = reasoning only, groups 11-12 mix both.
UNIFIED_GROUP_LABEL = {
    1: "TC Single call", 2: "TC Multiple calls", 3: "TC No call",
    4: "TC Malformed envelope", 5: "TC Truncation / recovery", 6: "TC Empty body",
    7: "TC Argument fidelity", 8: "TC Content position",
    10: "Reasoning span",
    11: "Reasoning ↔ tool interleaving", 12: "Adversarial nesting (reasoning + tool)",
    30: "Guided Decoding", 31: "Guided Decoding — payload REJECTED (syntax or schema) / recovery",
    40: "Prefilled Reasoning", 41: "Prefilled Reasoning — malformed",
    50: "Prefilled Response", 51: "Prefilled Response — malformed",
}


def tax(scenario):
    """(group_num, sub_letter) for a scenario slug; group 9 for anything unmapped.

    NUMBERING ONLY. A case's parser configuration (`init`) and its stream
    properties (`finish_reason`) are declared per case in `gen_unified_golden.py` and flow
    through the fixtures to the page. Keeping a second copy here would be a
    divergent copy of the same fact, free to drift from what the harness applies.
    """
    return UNIFIED_TAX.get(scenario, (9, scenario))


def numbered_id(scenario):
    """Scenario slug -> intrinsic numbered case id, e.g. 'arg_marker_in_string' ->
    'UNIFIED.7.b' (mirrors TOOLCALLING.streamv2.N naming)."""
    g, sub = tax(scenario)
    return f"UNIFIED.{g}.{sub}"


# The unified corpus names a family by its MODEL family (`qwen3`); the grammar-token
# registry in parser_families.yaml names the SAME grammar by its parser family
# (`qwen3_coder`). The popup colorizer is driven by the registry, so a corpus family
# has to be translated before it is used to color markup — otherwise the family has no
# declared markers, the colorizer falls back to heuristics, and its opaque
# argument-value regions (`opaque:`) are not applied.
# Derived from the ONE declaration in parser_families.yaml (`unified:` -> `registry`),
# so a family whose corpus name differs from its registry name says so in one place.
MARKER_FAMILY = {
    f: r["registry"]
    for f, r in yaml.safe_load(
        markers.parser_families_path().read_text()
    )["unified"].items()
    if r.get("registry") and r["registry"] != f
}


def marker_family(family):
    """Corpus family -> the parser_families.yaml `markers:` family that types it."""
    return MARKER_FAMILY.get(family, family)
