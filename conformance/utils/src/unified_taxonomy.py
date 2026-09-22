# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Single source of the UNIFIED case taxonomy: scenario slug -> numbered id
(UNIFIED.<group>-<sub>) and the per-group axis labels. Shared by the fixture
exploder (names case files by number) and the conformance generator (renders the
group labels), so the numbering can't drift between them.
Groups 1-9 mirror the tool-calling STREAM taxonomy (TOOLCALLING.streamv2.N) as
tool-only unified cases (UNIFIED subsumes STREAM). Group 10 is the reasoning axis
(REASONING.*). Group 11 is unique to unified: reasoning<->tool interleaving that
neither STREAM (no reasoning) nor REASONING (no ordered tool events) can express.
Group 12 is adversarial nesting (a marker of one channel inside another).
Groups 30-39 are Guided Decoding, split by the behavior under test. Groups 40/41
cover prefilled reasoning and 50/51 prefilled response. Model-specific cases use
named groups and sort after every numeric group.
There is no separate "input stream mode" axis: which channel the prompt pre-opened IS
`init.starting_state`, so a case that varied only that duplicated groups 31-33, and one that
varied nothing but a finish_reason label duplicated groups 1-12 (the parser cannot see
that value — `finish()` takes no argument).
"""

import re

import yaml

import markers

UNIFIED_TAX = {
    # Group 1 — Single call
    "tool_only": (1, "1"),
    # TODO: restore 1-2 in the follow-up to PR #232 (deferred-cases).
    # Group 2 — Multiple calls (streamv2.2)
    "two_calls": (2, "1"), "two_calls_same_name": (2, "2"),
    # Group 3 — No call (streamv2.3)
    "text_only": (3, "1"),
    # Group 4 — Malformed envelope. Labelled but EMPTY until now.
    "tool_block_never_closed_then_text": (4, "1"),
    "tool_markup_only_emits_nothing": (4, "2"),

    # Group 5 — Truncation / recovery (streamv2.5)
    "truncated_tool_eof": (5, "1"), "tool_no_close": (5, "2"),
    "orphan_close_after_prose": (5, "3"),
    "wrapped_saved_closer_partial_marker": (5, "4"),
    "bare_saved_closer": (5, "5"),
    # Group 6 — Empty body (streamv2.6)
    "empty_args": (6, "1"), "bare_parameterless_call": (6, "2"),
    # Group 7 — Argument fidelity (streamv2.7)
    "arg_unicode": (7, "1"), "arg_marker_in_string": (7, "2"),
    # TODO: restore 7-3 in the follow-up to PR #232 (deferred-cases).
    # Group 8 — Content / narration position (streamv2.8)
    "text_before_tool": (8, "1"), "trailing_text_after_tool": (8, "2"),
    "text_sandwich": (8, "3"), "text_between_calls": (8, "4"),
    "narrated_calls": (8, "5"),
    # Group 10 — Reasoning span (REASONING.*), reasoning-only
    "reason_only": (10, "1"), "reason_then_content": (10, "2"),
    "two_reason_spans": (10, "3"), "reason_unterminated": (10, "4"),
    "two_adjacent_reason_spans": (10, "5"),
    # Group 11 — Reasoning <-> tool interleaving (UNIQUE to unified)
    "reason_then_tool": (11, "1"), "reason_after_tool": (11, "2"),
    "reason_interleaved": (11, "3"), "reason_tool_text_reason_tool": (11, "4"),
    "interstitial_text": (11, "5"), "content_then_reason_then_tool": (11, "6"),
    "content_then_reason": (11, "7"), "reason_tool_reason_tool_reason": (11, "8"),
    "reason_between_calls": (11, "9"), "text_reason_tool_text_reason_tool": (11, "10"),
    # Group 12 — Adversarial nesting (a marker of one channel inside another)
    "reason_markup_in_arg": (12, "1"), "tool_in_reason": (12, "2"),
    "reason_markup_in_arg_with_text": (12, "3"), "tool_in_reason_with_text": (12, "4"),
    "kimi_k3_typed_argument_values": ("kimi", "1"),
    "kimi_k3_raw_json_arguments": ("kimi", "2"),
    "kimi_k3_spaced_xtml_markers": ("kimi", "3"),
    "kimi_k3_message_end_after_response": ("kimi", "4"),
    "kimi_k3_elided_think_close_to_response": ("kimi", "5"),
    "kimi_k3_malformed_call_then_valid": ("kimi", "6"),
    "kimi_k3_raw_json_eof": ("kimi", "7"),
    "kimi_k3_guided_native_wrapper": ("kimi", "8"),
    # Group 30 — Guided decoding: baseline and argument fidelity.
    "guided_json_named_tool": (30, "1"), "guided_json_required_tool": (30, "2"),
    "guided_json_two_calls": (30, "3"),
    "guided_json_escaped_string_args": (30, "4"), "guided_json_array_argument": (30, "5"),
    "guided_json_after_reasoning": (30, "6"), "guided_json_marker_inside_argument": (30, "7"),

    # Group 31 — Guided decoding: invalid JSON or call structure.
    "guided_json_invalid_call": (31, "1"), "guided_json_malformed_json": (31, "2"),
    "guided_json_partial_calls": (31, "3"),
    "guided_json_list_with_broken_element": (31, "4"),
    # Group 32 — Guided decoding: tool markup around the payload.
    "guided_json_tool_open_before_payload": (32, "1"),
    "guided_json_tool_close_after_payload": (32, "2"),
    "guided_json_wrapped_in_tool_markup": (32, "3"),
    "guided_json_orphan_tool_close_before_payload": (32, "4"),
    "guided_json_native_markup_only": (32, "5"),
    "guided_json_native_envelope_after_prose": (32, "6"),
    # Generated crossings (`_guided_product` in gen_unified_golden.py): payload
    # shape x surrounding grammar. The 31-12 through 31-20 rows are the quadrant that had ZERO
    # cases — markup present AND no call recoverable — where both the P2 recovery
    # leak and the unbounded invoke-header scan lived.
    "guided_json_syntax_error_trailing_close": (33, "1"),
    "guided_json_syntax_error_wrapped": (33, "2"),
    "guided_json_syntax_error_bare_opener": (33, "3"),
    "guided_json_schema_error_not_a_call_trailing_close": (33, "4"),
    "guided_json_schema_error_not_a_call_wrapped": (33, "5"),
    "guided_json_schema_error_not_a_call_bare_opener": (33, "6"),
    "guided_json_schema_error_nameless_element_trailing_close": (33, "7"),
    "guided_json_schema_error_nameless_element_wrapped": (33, "8"),
    "guided_json_schema_error_nameless_element_bare_opener": (33, "9"),

    # Devin-found crossings, added as AXIS entries so the next payload/surrounding
    # combination is generated rather than noticed later.
    "guided_json_gt_in_argument_trailing_close": (30, "11"),
    "guided_json_gt_in_argument_wrapped": (30, "12"),
    "guided_json_gt_in_argument_bare_opener": (30, "13"),
    # TODO: restore 30-14 in the follow-up to PR #232 (deferred-cases).

    # Marker OWNERSHIP: which control marker owns a `>` when two compete. The
    # corpus had no such case, and the gap leaked private reasoning as text.
    # Group 34 — Guided decoding: reasoning boundaries.
    "guided_json_narrated_invoke_in_reasoning": (34, "1"),
    "guided_json_prose_before_reasoning": (34, "2"),
    "guided_json_orphan_reason_close_before_payload": (34, "3"),
    "guided_json_stray_prefix_before_reasoning": (34, "4"),
    "guided_json_narrated_prefix_inside_reasoning": (34, "5"),
    "guided_json_unterminated_reasoning_then_wrapped_payload": (34, "6"),
    "guided_json_bare_tool_header_recovers_inside_a_thought": (34, "7"),
    # Group 35 — Guided decoding: markers in visible answers.
    "guided_json_quoted_bare_header_in_answer": (35, "1"),
    "guided_json_quoted_bare_header_after_payload": (35, "2"),
    "guided_json_quoted_bare_tool_header_in_answer": ("muse", "1"),
    # 31-29 and 31-30 remain reserved for the historical Gemma-only cases.
    # TODO: restore 31-31 through 31-40 in the follow-up to PR #232 (deferred-cases).
    "gemma4_guided_json_visible_call_prose_before_reasoning": ("gemma", "1"),
    "gemma4_guided_json_malformed_call_prefix_before_reasoning": ("gemma", "2"),
    "glm47_parameterless_call_shape_inside_argument": ("glm5", "1"),

    # Group 40 — Prefilled reasoning, happy
    "prefilled_reasoning_with_tool": (40, "1"), "prefilled_reasoning_with_guided_json": (40, "2"),
    "prefilled_reasoning_then_text_then_tool": (40, "3"), "prefilled_reasoning_then_text": (40, "4"),
    # Group 41 — Prefilled reasoning, weird / malformed
    "prefilled_reasoning_redundant_opener": (41, "1"), "prefilled_reasoning_truncated": (41, "2"),
    "prefilled_response_reasoning_markers_literal": (50, "4"),
    # TODO: restore 50-1 and 50-2 in the follow-up to PR #232 (deferred-cases).
}

# Axis prefix makes each group's channel explicit: "TC" = tool-calling only (groups
# 1-9 mirror the tool STREAM suite), "Reasoning" = reasoning only, groups 11-12 mix both.
UNIFIED_GROUP_LABEL = {
    1: "TC Single call", 2: "TC Multiple calls", 3: "TC No call",
    4: "TC Malformed envelope", 5: "TC Truncation / recovery", 6: "TC Empty body",
    7: "TC Argument fidelity", 8: "TC Content position",
    10: "Reasoning span",
    11: "Reasoning ↔ tool interleaving", 12: "Adversarial nesting (reasoning + tool)",
    30: "Guided Decoding — baseline and argument fidelity",
    31: "Guided Decoding — invalid JSON or call structure",
    32: "Guided Decoding — tool markup around the payload",
    33: "Guided Decoding — invalid payload plus tool markup",
    34: "Guided Decoding — reasoning boundaries",
    35: "Guided Decoding — markers in visible answers",
    40: "Prefilled Reasoning", 41: "Prefilled Reasoning — malformed",
    50: "Prefilled Response", 51: "Prefilled Response — malformed",
    "gemma": "Gemma 4 guided call-prefix boundaries",
    "glm5": "GLM 5 argument-marker boundaries",
    "kimi": "Kimi K3 XTML",
    "muse": "Muse-specific",
}


def tax(scenario):
    """(group_label, subcase_label) for a scenario slug; group 9 for anything unmapped.

    NUMBERING ONLY. A case's parser configuration (`init`) and its stream
    properties (`finish_reason`) are declared per case in `gen_unified_golden.py` and flow
    through the fixtures to the page. Keeping a second copy here would be a
    divergent copy of the same fact, free to drift from what the harness applies.
    """
    return UNIFIED_TAX.get(scenario, (9, scenario))


def taxonomy_sort_key(scenario):
    """Sort numeric groups first and their numeric suffixes numerically."""
    group, sub = tax(scenario)
    group_key = (0, group) if isinstance(group, int) else (1, str(group))
    if sub.isdecimal():
        return group_key, int(sub), ""
    return group_key, 10_000, sub


def case_label(scenario):
    """Scenario slug -> short case label with a numeric suffix."""
    group, sub = tax(scenario)
    return f"{group}-{sub}"


def numbered_id(scenario):
    """Scenario slug -> intrinsic case id, such as `UNIFIED.7-1` or `UNIFIED.kimi-1`."""
    return f"UNIFIED.{case_label(scenario)}"


# Historical capture archives remain byte-for-byte evidence. This map is the one
# read-side bridge from their former labels to the scenario-owned current label.
# Current inputs and goldens use ``numbered_id`` and never write these aliases.
LEGACY_CASE_LABELS = {
    "1.a": "tool_only",
    "2.a": "two_calls", "2.b": "two_calls_same_name",
    "3.a": "text_only",
    "4.a": "tool_block_never_closed_then_text", "4.b": "tool_markup_only_emits_nothing",
    "5.a": "truncated_tool_eof", "5.b": "tool_no_close", "5.c": "orphan_close_after_prose",
    "6.a": "empty_args",
    "7.a": "arg_unicode", "7.b": "arg_marker_in_string",
    "8.a": "text_before_tool", "8.b": "trailing_text_after_tool", "8.c": "text_sandwich",
    "8.d": "text_between_calls", "8.e": "narrated_calls",
    "10.a": "reason_only", "10.b": "reason_then_content", "10.c": "two_reason_spans",
    "10.d": "reason_unterminated", "10.e": "two_adjacent_reason_spans",
    "11.a": "reason_then_tool", "11.b": "reason_after_tool", "11.c": "reason_interleaved",
    "11.d": "reason_tool_text_reason_tool", "11.e": "interstitial_text",
    "11.f": "content_then_reason_then_tool", "11.g": "content_then_reason",
    "11.h": "reason_tool_reason_tool_reason", "11.i": "reason_between_calls",
    "11.j": "text_reason_tool_text_reason_tool",
    "12.a": "reason_markup_in_arg", "12.b": "tool_in_reason",
    "12.c": "reason_markup_in_arg_with_text", "12.d": "tool_in_reason_with_text",
    "k3-1": "kimi_k3_typed_argument_values", "k3-2": "kimi_k3_raw_json_arguments",
    "k3-3": "kimi_k3_spaced_xtml_markers", "k3-4": "kimi_k3_message_end_after_response",
    "k3-5": "kimi_k3_elided_think_close_to_response", "k3-6": "kimi_k3_malformed_call_then_valid",
    "k3-7": "kimi_k3_raw_json_eof", "k3-8": "kimi_k3_guided_native_wrapper",
    "30.a": "guided_json_named_tool", "30.b": "guided_json_required_tool",
    "30.c": "guided_json_two_calls", "30.d": "guided_json_escaped_string_args",
    "30.e": "guided_json_array_argument", "30.f": "guided_json_after_reasoning",
    "30.g": "guided_json_marker_inside_argument", "30.k": "guided_json_gt_in_argument_trailing_close",
    "30.l": "guided_json_gt_in_argument_wrapped", "30.m": "guided_json_gt_in_argument_bare_opener",
    "31-1": "guided_json_invalid_call", "31-2": "guided_json_malformed_json",
    "31-3": "guided_json_partial_calls", "31-4": "guided_json_list_with_broken_element",
    "31-5": "guided_json_tool_open_before_payload", "31-6": "guided_json_tool_close_after_payload",
    "31-7": "guided_json_wrapped_in_tool_markup", "31-8": "guided_json_narrated_invoke_in_reasoning",
    "31-9": "guided_json_prose_before_reasoning", "31-10": "guided_json_orphan_reason_close_before_payload",
    "31-11": "guided_json_orphan_tool_close_before_payload",
    "31-12": "guided_json_syntax_error_trailing_close", "31-13": "guided_json_syntax_error_wrapped",
    "31-14": "guided_json_syntax_error_bare_opener",
    "31-15": "guided_json_schema_error_not_a_call_trailing_close",
    "31-16": "guided_json_schema_error_not_a_call_wrapped",
    "31-17": "guided_json_schema_error_not_a_call_bare_opener",
    "31-18": "guided_json_schema_error_nameless_element_trailing_close",
    "31-19": "guided_json_schema_error_nameless_element_wrapped",
    "31-20": "guided_json_schema_error_nameless_element_bare_opener",
    "31-21": "guided_json_stray_prefix_before_reasoning",
    "31-22": "guided_json_narrated_prefix_inside_reasoning",
    "31-23": "guided_json_native_markup_only",
    "31-24": "guided_json_unterminated_reasoning_then_wrapped_payload",
    "31-25": "guided_json_quoted_bare_header_in_answer",
    "31-26": "guided_json_quoted_bare_tool_header_in_answer",
    "31-27": "guided_json_quoted_bare_header_after_payload",
    "31-28": "guided_json_bare_tool_header_recovers_inside_a_thought",
    "g4-1": "gemma4_guided_json_visible_call_prose_before_reasoning",
    "g4-2": "gemma4_guided_json_malformed_call_prefix_before_reasoning",
    "40.a": "prefilled_reasoning_with_tool", "40.b": "prefilled_reasoning_with_guided_json",
    "40.c": "prefilled_reasoning_then_text_then_tool", "40.d": "prefilled_reasoning_then_text",
    "41.a": "prefilled_reasoning_redundant_opener", "41.b": "prefilled_reasoning_truncated",
    "50.d": "prefilled_response_reasoning_markers_literal",
}


def historical_case_label(label):
    """Translate a pre-reorganization case label while leaving current labels unchanged."""
    dotted_group_31 = re.fullmatch(r"31\.([a-x])", label)
    if dotted_group_31:
        label = f"31-{ord(dotted_group_31[1]) - ord('a') + 1}"
    scenario = LEGACY_CASE_LABELS.get(label)
    return case_label(scenario) if scenario is not None else label


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
