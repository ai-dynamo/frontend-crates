# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The `dynamo-replay-trace/v1` subset the round trip reads: one JSONL line per
case with `schema`, `id`, `request`, `generation.output_token_ids` (including
the stop token) and `generation.finish_reason`."""

from __future__ import annotations

import json
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA = "dynamo-replay-trace/v1"
FINISH_REASONS = ("stop", "length")


class TraceError(ValueError):
    """A diagnostic safe to print: it never embeds request or output text."""


@dataclass(frozen=True)
class Case:
    id: str
    request: dict[str, Any]
    output_token_ids: list[int]
    finish_reason: str
    # Declared terminal suffix; empty means it is inferred.
    terminal_token_ids: list[int]

    def terminal(self, eos_token_ids: Collection[int]) -> list[int]:
        """The declared terminal suffix, else the last ID when it is an EOS ID
        or one of the request's `stop_token_ids`."""
        if self.terminal_token_ids:
            return self.terminal_token_ids
        stops = set(self.request.get("stop_token_ids") or [])
        last = self.output_token_ids[-1:]
        return last if last and (last[0] in eos_token_ids or last[0] in stops) else []


def token_ids(value: Any, name: str) -> list[int]:
    if not isinstance(value, list) or any(
        type(item) is not int or item < 0 for item in value
    ):
        raise TraceError(f"{name} must be a list of token IDs")
    return value


def parse_case(obj: Any) -> Case:
    if not isinstance(obj, dict) or obj.get("schema") != SCHEMA:
        raise TraceError(f"schema is not {SCHEMA}")
    identifier = obj.get("id")
    if not isinstance(identifier, str) or not identifier:
        raise TraceError("id is missing")
    request = obj.get("request")
    if not isinstance(request, dict) or not isinstance(request.get("messages"), list):
        raise TraceError("request has no messages")
    if request.get("stop_token_ids") is not None:
        token_ids(request["stop_token_ids"], "stop_token_ids")
    generation = obj.get("generation")
    if not isinstance(generation, dict):
        raise TraceError("generation is missing")
    output = token_ids(generation.get("output_token_ids"), "output_token_ids")
    finish = generation.get("finish_reason")
    if finish not in FINISH_REASONS:
        raise TraceError("finish_reason must be stop or length")
    terminal = generation.get("terminal_token_ids")
    terminal = [] if terminal is None else token_ids(terminal, "terminal_token_ids")
    if terminal and output[-len(terminal) :] != terminal:
        raise TraceError("terminal_token_ids are not the output suffix")
    return Case(identifier, request, output, finish, terminal)


def load(path: Path) -> list[Case]:
    cases: list[Case] = []
    seen: set[str] = set()
    # Split on LF only: JSON keeps U+2028 literal, which splitlines() would break.
    for number, line in enumerate(path.read_text(encoding="utf-8").split("\n"), 1):
        if not line.strip():
            continue
        try:
            case = parse_case(json.loads(line))
        except json.JSONDecodeError:
            raise TraceError(f"trace line {number}: invalid JSON") from None
        except TraceError as exc:
            raise TraceError(f"trace line {number}: {exc}") from None
        if case.id in seen:
            raise TraceError(f"trace line {number}: duplicate id")
        seen.add(case.id)
        cases.append(case)
    return cases
