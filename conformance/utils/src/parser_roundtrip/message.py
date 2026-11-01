# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The parser's ordered events folded into the one assistant message serving
returns: `content`, `reasoning_content` and `tool_calls`."""

from __future__ import annotations

import json
from typing import Any

# The event kinds the parser_roundtrip example writes.
KINDS = ("reasoning", "text", "tool_call")


def message_from_events(
    events: list[dict[str, Any]], disabled: bool
) -> tuple[dict[str, Any], str]:
    """The message serving assembles from ordered events, and the events' kinds
    in order with runs collapsed. Content is None when empty; with thinking off,
    reasoning is content where it occurred, without its markers."""
    parts: dict[str, list[str]] = {"reasoning": [], "text": []}
    calls: list[dict[str, Any]] = []
    kinds: list[str] = []
    for event in events:
        kind = event["kind"]
        if kind not in KINDS:
            raise ValueError(f"unknown event kind {kind!r}")
        if kind == "reasoning" and disabled:
            kind = "text"
        if kinds[-1:] != [kind]:
            kinds.append(kind)
        if kind == "tool_call":
            calls.append({key: event.get(key) for key in ("name", "arguments", "id")})
        else:
            parts[kind].append(event["text"])
    message = {
        "content": "".join(parts["text"]) or None,
        "reasoning_content": "".join(parts["reasoning"]),
        "tool_calls": calls,
    }
    return message, ",".join(kinds)


def representable(order: str) -> bool:
    """Whether one message can hold events in this order. Templates render
    reasoning first, so reasoning after text or a call cannot round-trip; text
    after a call folds into content, as serving does, and is judged."""
    kinds = [kind for kind in order.split(",") if kind]
    return "reasoning" not in kinds[1:]


def assistant_turn(message: dict[str, Any]) -> dict[str, Any] | None:
    """The parsed message as a history turn; None when arguments are not a JSON object."""
    # Empty content goes back as "", as vLLM normalizes a null before rendering;
    # a template that prints it unguarded (GLM-4.7) would otherwise write "None".
    turn: dict[str, Any] = {"role": "assistant", "content": message["content"] or ""}
    if message["reasoning_content"]:
        turn["reasoning_content"] = message["reasoning_content"]
    calls = []
    for call in message["tool_calls"]:
        try:
            arguments = json.loads(call["arguments"])
        except json.JSONDecodeError:
            return None
        if not isinstance(arguments, dict):
            return None
        function = {"name": call["name"], "arguments": arguments}
        identifier = {"id": call["id"]} if call.get("id") else {}
        calls.append({**identifier, "type": "function", "function": function})
    if calls:
        turn["tool_calls"] = calls
    return turn
