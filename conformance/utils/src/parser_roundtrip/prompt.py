# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The generation prompt the model continued: the request rendered by the
model's chat template with the generation prompt added. Its tail decides where
the output starts, so the parser's starting state and the model turn both
depend on it.

The prompt is rendered, not recorded, so a request that lacks something the
server applied, such as a default `chat_template_kwargs` entry, renders a prompt
the model never saw. When the trace records the engine's prompt token IDs, the
rendered prompt must tokenize to exactly those IDs."""

from __future__ import annotations

from .bundle import Bundle
from .trace import Case

TAIL_TOKENS = 8


class PromptMismatch(Exception):
    """The rendered prompt is not the one the engine tokenized; the message
    quotes a few prompt tokens, so it belongs in private results only."""


def generation_prompt(bundle: Bundle, case: Case) -> str:
    """The request's conversation rendered up to the open assistant turn,
    checked against the recorded prompt when there is one."""
    rendered = bundle.render_request(case.request, True)
    recorded = case.prompt_token_ids
    if recorded is None:
        return rendered
    ids = bundle.tokenizer.encode(rendered, add_special_tokens=False)
    if ids != recorded:
        pairs = enumerate(zip(ids, recorded))
        at = next((i for i, (a, b) in pairs if a != b), min(len(ids), len(recorded)))
        recorded_tail = bundle.decode(recorded[at : at + TAIL_TOKENS])
        rendered_tail = bundle.decode(ids[at : at + TAIL_TOKENS])
        raise PromptMismatch(
            f"token {at} of {len(recorded)}: recorded {recorded_tail!r}, "
            f"rendered {rendered_tail!r}"
        )
    return rendered
