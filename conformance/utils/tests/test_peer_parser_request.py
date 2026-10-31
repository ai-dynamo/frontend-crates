# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pydantic root construction must retain the nested types consumed by parsers."""

from copy import deepcopy

import pytest

import peer_modules

pydantic = pytest.importorskip("pydantic", minversion="2")


class Function(pydantic.BaseModel):
    name: str
    parameters: dict = {}


class Tool(pydantic.BaseModel):
    type: str
    function: Function


class NamedChoice(pydantic.BaseModel):
    type: str
    function: Function


class Request(pydantic.BaseModel):
    tools: list[Tool] | None = None
    tool_choice: str | NamedChoice = "auto"

    @pydantic.model_validator(mode="after")
    def reject_empty_serving_offers(self):
        if self.tools == []:
            raise ValueError("serving requires nonempty offered tools")
        return self


@pytest.mark.parametrize("tools", [[], [{"name": "lookup", "parameters": {"type": "object"}}]])
def test_parser_context_keeps_nested_protocol_types_and_original_fields(tools):
    fields = {"tools": tools, "tool_choice": {"type": "function", "function": {"name": "lookup"}}}
    original = deepcopy(fields)
    if not tools:
        with pytest.raises(ValueError, match="nonempty offered tools"):
            Request(**fields)
    context = peer_modules.parser_only_request(Request, tool_type=Tool, named_choice_type=NamedChoice, **fields)
    assert context.tool_choice.function.name == "lookup"
    assert isinstance(context.tool_choice, NamedChoice)
    if tools:
        assert isinstance(context.tools[0], Tool)
        assert context.tools[0].function.name == "lookup"
    else:
        assert context.tools == []
    assert fields == original


def test_parser_context_still_rejects_invalid_nested_tool_fields():
    with pytest.raises(pydantic.ValidationError):
        peer_modules.parser_only_request(Request, tool_type=Tool, tools=[{"name": 123}])
    with pytest.raises(pydantic.ValidationError):
        peer_modules.parser_only_request(Request, tool_type=Tool, named_choice_type=NamedChoice,
                                         tools=[], tool_choice={"type": "function", "function": {"name": 123}})
