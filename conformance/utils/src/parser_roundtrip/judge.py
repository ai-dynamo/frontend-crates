# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Judge each case: x = decode(output minus terminal) goes through the Unified
parser, its ordered events fold into one assistant message, and the template
renders that message after the request history. The rendered turn, without its
end of turn, must equal the generation prompt's tail plus x. A parse that loses
or alters text cannot render back to x."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .bundle import Bundle, template_kwargs
from .driver import PARSER_COMMAND, run_parser
from .folds import PROBE_CONTENT, fold, shape_for
from .message import assistant_turn, message_from_events, representable
from .prompt import generation_prompt
from .serving import (
    FAMILY_ALIASES,
    function_tools,
    reasoning_disabled,
    starting_state,
    unserved_reason,
)
from .trace import Case

WORD = re.compile(r"\w+")


@dataclass(frozen=True)
class Prepared:
    case: Case
    prompt: str  # rendered generation prompt
    text: str  # x
    reasoning_disabled: bool
    job: dict[str, Any]


def skip(case: Case, reason: str, detail: str | None = None) -> dict[str, Any]:
    result = {"id": case.id, "verdict": f"skip:{reason}"}
    return {**result, "detail": detail} if detail else result


def prepare(case: Case, bundle: Bundle, family: str) -> Prepared | dict[str, Any]:
    """The parser job for a case, or its skip result."""
    family = FAMILY_ALIASES.get(family, family)
    if case.finish_reason == "length":
        return skip(case, "incomplete")
    request = case.request
    kwargs = template_kwargs(request)
    reason = unserved_reason(request, kwargs)
    if reason:
        return skip(case, reason)
    try:
        prompt = generation_prompt(bundle, case)
    except Exception as exc:  # the reference template cannot render the request
        return skip(case, "template-error", type(exc).__name__)
    output = case.output_token_ids
    terminal = case.terminal(bundle.eos_token_ids)
    text = bundle.decode(output[: len(output) - len(terminal)])
    disabled = reasoning_disabled(family, kwargs)
    job = {
        "id": case.id,
        "text": text,
        "family": family,
        "starting_state": starting_state(family, prompt, disabled),
        "tools": function_tools(request),
    }
    return Prepared(case, prompt, text, disabled, job)


def common_prefix(left: str, right: str) -> int:
    size = 0
    for a, b in zip(left, right):
        if a != b:
            break
        size += 1
    return size


def signature(raw: str, rendered: str) -> str:
    """Both sides of the first difference with word runs folded to `w`, so
    failures of one kind share a key."""
    return json.dumps(
        [WORD.sub("w", raw[:16]), WORD.sub("w", rendered[:16])], ensure_ascii=False
    )


def judge(prepared: Prepared, bundle: Bundle, parsed: dict[str, Any]) -> dict[str, Any]:
    case, job = prepared.case, prepared.job
    result: dict[str, Any] = {
        "id": case.id,
        "starting_state": job["starting_state"],
        "reasoning_disabled": prepared.reasoning_disabled,
    }
    if case.request.get("tool_choice") == "none":
        # Serving drops calls under tool_choice none; this judges the parsed calls.
        result["tool_choice"] = "none"
    if "error" in parsed:
        return {
            **result,
            "verdict": "fail",
            "signature": "parser_error",
            "detail": str(parsed["error"])[:200],
        }
    message, order = message_from_events(parsed["events"], prepared.reasoning_disabled)
    result["order"] = order
    result["incomplete_calls"] = parsed.get("incomplete_calls", 0)
    if not representable(order):
        return {**result, **skip(case, "not-representable")}
    turn = assistant_turn(message)
    if turn is None:
        return {**result, "verdict": "fail", "signature": "invalid_arguments"}
    shape = shape_for(bundle, case.request)
    if "tool_calls" in turn and not shape.renders_calls:
        return {**result, **skip(case, "template-renders-no-tool-calls")}
    if "reasoning_content" in turn and not shape.renders_reasoning:
        return {**result, **skip(case, "template-renders-no-reasoning")}
    request, prompt = case.request, prepared.prompt
    probe_turn = {"role": "assistant", "content": PROBE_CONTENT}
    try:
        history = bundle.render_request(request, False)
        full = bundle.render_request(request, False, turn)
        appends = full.startswith(history) or bundle.render_request(
            request, False, probe_turn
        ).startswith(history)
    except Exception as exc:  # the reference template cannot render the turn
        return {**result, **skip(case, "template-error", type(exc).__name__)}
    # Without a turn rendered after the history there is no boundary to compare at.
    if not prompt.startswith(history) or not appends:
        return {**result, **skip(case, "template-does-not-append-turns")}
    start = len(history) if full.startswith(history) else common_prefix(prompt, full)
    rendered = full[start:]
    rendered = next(
        (rendered[: -len(end)] for end in shape.end_of_turn if rendered.endswith(end)),
        rendered,
    )
    generation = prompt[start:]
    raw, rendered, note = fold(turn, shape, generation, prepared.text, rendered)
    if note:
        result["note"] = note
    if raw == rendered:
        return {**result, "verdict": "pass"}
    at = common_prefix(raw, rendered)
    # The first difference in x; negative inside the generation prompt's tail.
    in_text = at - (len(raw) - len(prepared.text))
    # Snippets come from the unfolded raw, which is generation plus x.
    in_raw = len(generation) + in_text
    unfolded = generation + prepared.text
    return {
        **result,
        "verdict": "fail",
        "signature": signature(raw[at:], rendered[at:]),
        "at": in_text,
        "raw": unfolded[max(0, in_raw - 40) : in_raw + 80],
        "rendered": rendered[max(0, at - 40) : at + 80],
    }


def run(
    cases: list[Case],
    bundle: Bundle,
    family: str,
    stderr_path: Path,
    command: Sequence[str] = PARSER_COMMAND,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    outcomes = [prepare(case, bundle, family) for case in cases]
    jobs = [item.job for item in outcomes if isinstance(item, Prepared)]
    header, answers = run_parser(jobs, command, stderr_path)
    results = [
        judge(item, bundle, answers[item.case.id])
        if isinstance(item, Prepared)
        else item
        for item in outcomes
    ]
    return header, results
