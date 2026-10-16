# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Normalize peer streaming deltas into the final ordered event list."""

import json


def assemble_stream(chunk_deltas: list, *, recorded_assembly: list | None = None) -> list:
    """Build stream output without replacing its order with a batch assembly."""
    events: list = []
    cur_tool = None
    for deltas in chunk_deltas:
        for delta in deltas or []:
            kind = delta.get("kind")
            if kind in ("reasoning", "text"):
                cur_tool = None
                if events and events[-1]["kind"] == kind:
                    events[-1]["text"] += delta.get("text") or ""
                else:
                    events.append({"kind": kind, "text": delta.get("text") or ""})
            elif kind == "tool_call":
                name, arguments = delta.get("name"), delta.get("arguments")
                if name:
                    events.append({
                        "kind": "tool_call",
                        "name": name,
                        "_raw": arguments or "",
                        "_complete": delta.get("complete", True),
                    })
                    cur_tool = len(events) - 1
                elif cur_tool is not None and arguments:
                    previous = events[cur_tool].get("_raw", "")
                    if isinstance(previous, str) and isinstance(arguments, str):
                        events[cur_tool]["_raw"] = previous + arguments
                    else:
                        # A complete structured fragment supersedes earlier raw text.
                        events[cur_tool]["_raw"] = arguments
                if cur_tool is not None and delta.get("complete", True):
                    events[cur_tool]["_complete"] = True
    events = [event for event in events if event.get("kind") != "tool_call" or event.get("_complete", True)]
    malformed = set()
    for position, event in enumerate(events):
        if event["kind"] == "tool_call":
            raw = event.pop("_raw", "")
            event.pop("_complete", None)
            if isinstance(raw, (dict, list)):
                event["arguments"] = raw
            elif not (raw or "").strip():
                event["arguments"] = {}
            else:
                try:
                    event["arguments"] = json.loads(raw)
                except (ValueError, TypeError):
                    malformed.add(position)
                    event["arguments"] = raw
    if malformed and recorded_assembly is not None:
        calls = [(position, event) for position, event in enumerate(events) if event["kind"] == "tool_call"]
        recorded = [event for event in recorded_assembly if event["kind"] == "tool_call"]
        # Batch fallback is safe only when the complete call sequence aligns.
        if len(calls) == len(recorded) and all(
            event["name"] == observed["name"] and (
                event["arguments"] == observed["arguments"]
                or position in malformed and observed["arguments"] == {}
            ) for (position, event), observed in zip(calls, recorded)
        ):
            for (position, event), observed in zip(calls, recorded):
                if position in malformed:
                    event["arguments"] = observed["arguments"]
    return events
