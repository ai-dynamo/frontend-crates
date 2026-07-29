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
"""

UNIFIED_TAX = {
    # Group 1 — Single call
    "tool_only": (1, "a"),
    # Group 2 — Multiple calls (streamv2.2)
    "two_calls": (2, "a"), "two_calls_same_name": (2, "b"),
    # Group 3 — No call (streamv2.3)
    "text_only": (3, "a"),
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
}

# Axis prefix makes each group's channel explicit: "TC" = tool-calling only (groups
# 1-9 mirror the tool STREAM suite), "Reasoning" = reasoning only, groups 11-12 mix both.
UNIFIED_GROUP_LABEL = {
    1: "TC Single call", 2: "TC Multiple calls", 3: "TC No call",
    4: "TC Malformed envelope", 5: "TC Truncation / recovery", 6: "TC Empty body",
    7: "TC Argument fidelity", 8: "TC Content position",
    10: "Reasoning span",
    11: "Reasoning ↔ tool interleaving", 12: "Adversarial nesting (reasoning + tool)",
}


def tax(scenario):
    """(group_num, sub_letter) for a scenario slug; group 9 for anything unmapped."""
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
MARKER_FAMILY = {"qwen3": "qwen3_coder"}


def marker_family(family):
    """Corpus family -> the parser_families.yaml `markers:` family that types it."""
    return MARKER_FAMILY.get(family, family)
