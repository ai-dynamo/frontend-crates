# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Where a chat template cannot write back what the model wrote. A probe renders
marker turns to learn how the template frames a final assistant turn, and a
fold forgives exactly the differences that framing forces, with a note."""

from __future__ import annotations

import functools
import json
import re
from dataclasses import dataclass
from typing import Any

from .bundle import Bundle, template_kwargs

PROBE_CONTENT = "__roundtrip_content__"
PROBE_REASONING = "__roundtrip_reasoning__"
PROBE_TOOL = "__roundtrip_tool__"
TAG = re.compile(r"<[^<>\s]+>")


@dataclass(frozen=True)
class Shape:
    """How the template renders a final assistant turn under one set of kwargs."""

    reasoning_opener: str | None  # markup right before rendered reasoning
    reasoning_closer: str | None  # text between rendered reasoning and content
    renders_reasoning: bool
    renders_calls: bool
    plain_prefix: str | None  # turn text before content, without reasoning
    empty_prefix: str | None  # turn text before content, with an empty reasoning block
    end_of_turn: tuple[str, ...]  # longest first


def last_markup(text: str, bundle: Bundle) -> int:
    """Where the last added token or <tag> in `text` starts; -1 without one."""
    vocab = bundle.tokenizer.get_added_vocab()
    starts = [text.rfind(token) for token in vocab if token]
    starts += [match.start() for match in TAG.finditer(text)]
    return max(starts, default=-1)


@functools.cache
def probe(bundle: Bundle, kwargs_json: str) -> Shape:
    kwargs = json.loads(kwargs_json)
    tool = {
        "type": "function",
        "function": {
            "name": PROBE_TOOL,
            "parameters": {"type": "object", "properties": {}},
        },
    }
    call = {"type": "function", "function": {"name": PROBE_TOOL, "arguments": {}}}

    def render(
        extra: dict[str, Any] | None, tools: list[dict[str, Any]] | None = None
    ) -> str | None:
        """A user turn, then an assistant turn unless `extra` is None; None when
        the template rejects it."""
        messages = [{"role": "user", "content": "probe"}]
        if extra is not None:
            messages.append({"role": "assistant", "content": PROBE_CONTENT, **extra})
        try:
            return bundle.render(messages, tools, False, kwargs)
        except Exception:  # a template rejecting a probe does not render that field
            return None

    base, plain = render(None), render({})
    reasoned = render({"reasoning_content": PROBE_REASONING}) or ""

    def turn(text: str | None) -> str:
        """The assistant turn after the user-only render; empty when unknown."""
        if base is None or text is None or not text.startswith(base):
            return ""
        return text[len(base) :]

    plain_turn, reasoned_turn = turn(plain), turn(reasoned)
    plain_prefix = None
    if PROBE_CONTENT in plain_turn:
        plain_prefix = plain_turn[: plain_turn.index(PROBE_CONTENT)]
    opener = closer = empty_prefix = None
    if PROBE_REASONING in reasoned:
        at = reasoned.index(PROBE_REASONING)
        before = reasoned[:at].rstrip()
        start = last_markup(before, bundle)
        if start >= 0 and not any(char.isspace() for char in before[start:]):
            opener = before[start:]
        after = at + len(PROBE_REASONING)
        if PROBE_CONTENT in reasoned[after:]:
            closer = reasoned[after : reasoned.index(PROBE_CONTENT, after)]
    if PROBE_REASONING in reasoned_turn and closer is not None:
        empty_prefix = reasoned_turn[: reasoned_turn.index(PROBE_REASONING)] + closer
    eos = [bundle.decode([token]) for token in bundle.eos_token_ids]
    ends = {*eos, *(text + "\n" for text in eos)}
    if plain is not None and PROBE_CONTENT in plain:
        ends.add(plain[plain.rindex(PROBE_CONTENT) + len(PROBE_CONTENT) :])
    with_call, without = render({"tool_calls": [call]}, [tool]), render({}, [tool])
    return Shape(
        reasoning_opener=opener,
        reasoning_closer=closer,
        renders_reasoning=PROBE_REASONING in reasoned,
        renders_calls=with_call is not None
        and without is not None
        and with_call != without,
        plain_prefix=plain_prefix,
        empty_prefix=empty_prefix,
        end_of_turn=tuple(sorted(ends - {""}, key=len, reverse=True)),
    )


def shape_for(bundle: Bundle, request: dict[str, Any]) -> Shape:
    return probe(bundle, json.dumps(template_kwargs(request), sort_keys=True))


def fold(
    turn: dict[str, Any], shape: Shape, generation: str, text: str, rendered: str
) -> tuple[str, str, str | None]:
    """The compared (raw, rendered) and a note when the template cannot render
    what the model wrote; raw keeps x's tail, so positions map back to x."""
    raw, plain = generation + text, shape.plain_prefix
    opener, closer = shape.reasoning_opener, shape.reasoning_closer
    # A turn without reasoning always renders after `plain`, so neither a missing
    # nor an empty reasoning block renders back.
    if "reasoning_content" not in turn and plain and not raw.startswith(plain):
        if shape.empty_prefix and raw.startswith(shape.empty_prefix):
            return plain + raw[len(shape.empty_prefix) :], rendered, "empty_reasoning"
        if (
            len(plain) > len(generation)
            and plain.startswith(generation)
            and not (opener and text.lstrip().startswith(opener))
        ):
            return plain + text, rendered, "template_inserted_prefix"
    # The template closes reasoning that the model ended with its stop token,
    # unless the reasoning already holds a closer the parser did not honor.
    if (
        "reasoning_content" in turn
        and not turn["content"]
        and "tool_calls" not in turn
        and closer
        and closer.strip() not in turn["reasoning_content"]
        and rendered.endswith(closer)
        and not raw.endswith(closer)
    ):
        return raw, rendered[: -len(closer)], "unclosed_reasoning"
    return raw, rendered, None
