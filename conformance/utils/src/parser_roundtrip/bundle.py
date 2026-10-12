# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The model's tokenizer and chat template through HF transformers, which
shares no code with dynamo-renderer or the parsers under test."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from transformers import AutoTokenizer, PreTrainedTokenizerBase


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.is_file() else {}


def chat_template(model_dir: Path) -> str:
    """`chat_template.jinja`, else the default template in tokenizer_config.json."""
    path = model_dir / "chat_template.jinja"
    if path.is_file():
        return path.read_text()
    template = read_json(model_dir / "tokenizer_config.json").get("chat_template")
    if isinstance(template, list):
        template = {item.get("name"): item.get("template") for item in template}
        template = template.get("default")
    if not isinstance(template, str) or not template:
        raise ValueError("model directory has no chat template")
    return template


def template_kwargs(request: dict[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if request.get("reasoning_effort") is not None:
        kwargs["reasoning_effort"] = request["reasoning_effort"]
    kwargs.update(request.get("chat_template_kwargs") or {})
    return kwargs


def template_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """History tool-call argument strings parsed to objects, as the frontend does."""
    messages = copy.deepcopy(messages)
    for message in messages:
        for call in message.get("tool_calls") or []:
            function = call.get("function") if isinstance(call, dict) else None
            if isinstance(function, dict) and isinstance(
                function.get("arguments"), str
            ):
                try:
                    function["arguments"] = json.loads(function["arguments"])
                except json.JSONDecodeError:
                    pass
    return messages


# Identity equality, so probes can be cached per bundle.
@dataclass(frozen=True, eq=False)
class Bundle:
    tokenizer: PreTrainedTokenizerBase = field(repr=False)
    chat_template: str = field(repr=False)
    eos_token_ids: frozenset[int]

    @classmethod
    def load(cls, model_dir: Path) -> Bundle:
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
        eos = {tokenizer.eos_token_id}
        config = read_json(model_dir / "config.json")
        for source in (
            config,
            config.get("text_config") or {},
            read_json(model_dir / "generation_config.json"),
        ):
            value = source.get("eos_token_id")
            eos.update(value if isinstance(value, list) else [value])
        return cls(
            tokenizer=tokenizer,
            chat_template=chat_template(model_dir),
            eos_token_ids=frozenset(item for item in eos if isinstance(item, int)),
        )

    def render(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        add_generation_prompt: bool,
        kwargs: dict[str, Any],
    ) -> str:
        text = self.tokenizer.apply_chat_template(
            template_messages(messages),
            tools=[*tools] if tools else None,
            chat_template=self.chat_template,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            **kwargs,
        )
        return cast(str, text)

    def render_request(
        self,
        request: dict[str, Any],
        add_generation_prompt: bool,
        turn: dict[str, Any] | None = None,
    ) -> str:
        """The request's conversation, with `turn` appended when given."""
        messages = [*request["messages"], *([turn] if turn else [])]
        return self.render(
            messages,
            request.get("tools"),
            add_generation_prompt,
            template_kwargs(request),
        )

    def decode(self, ids: list[int]) -> str:
        text = self.tokenizer.decode(
            ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        return cast(str, text)
