# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What dynamo.frontend decides before the Unified parser sees a token: the
family, whether thinking is off, the channel the generation prompt leaves open,
the tools, and the requests it serves some other way. Each rule names the
Dynamo function it mirrors (lib/llm/src/preprocessor.rs unless stated).

MiniMax M3 follows the Unified route of ai-dynamo/dynamo#17236. Dynamo main
still serves M3 through its reasoning parser and tool-call jail, which this
round trip does not judge."""

from __future__ import annotations

from typing import Any

# Crate aliases (`unified_registry!` in parsers/v2/src/unified/mod.rs).
FAMILY_ALIASES = {
    "qwen3_coder": "qwen3",
    "kimi-k3": "kimi_k3",
    "minimax-m3": "minimax_m3",
    "nemotron3": "nemotron_v3",
    "muse_spark_open": "muse_spark",
    "harmony": "gpt_oss",
}
# Dynamo hands this parser a prompt prefix that a whole-text job cannot carry
# (`prompt_reasoning_prefill_at_boundary` for GPT-OSS).
UNMODELED_FAMILIES = ("gpt_oss",)
# Beam keeps the model's stop-token text (`<|call|>`, `<|return|>`) in the parser's
# input, because a call completes only at its marker (`retain_model_stop_token_text`).
RETAIN_STOP_TOKEN_TEXT = ("beam",)
# How the parser frames native argument values, as Dynamo chooses it per request
# (`argument_framing` in unified_parser.rs). Beam's default turn grammar types every
# held tool's arguments, so non-string values own their strings.
ARGUMENT_FRAMING = {"beam": "schema_constrained"}
BEAM_HEADER = "<|start|>assistant"
# Dynamo's reasoning openers by family (`prompt_injected_reasoning_start`);
# every other family uses `<think>`.
REASONING_OPENERS = {
    "gemma4": ("<|channel>thought", "<|channel>"),
    "kimi_k3": ("<|open|>think<|sep|>",),
    "minimax_m3": ("<mm:think>",),
    "muse_glimmer": ("assistant to=self<|message|>",),
}
MINIMAX_M3_CLOSER = "</mm:think>"
# MiniMax M3's template reads only `thinking_mode`, and only these values.
MINIMAX_M3_MODES = ("enabled", "disabled", "adaptive")
# Families whose parser starts in the response channel when thinking is off
# (`UnifiedRequestPolicy::starting_state` in unified_parser.rs).
RESPONSE_WHEN_DISABLED = ("gemma4", "minimax_m3", "nemotron_v3")
# Reasoning never turns off (`request_disables_reasoning_with_family`).
REASONING_ALWAYS_ON = ("gpt_oss", "muse_spark", "beam")
THINKING_KEYS = ("thinking", "enable_thinking", "thinking_mode")
BOOL_WORDS = ("1", "true", "on", "yes", "0", "false", "off", "no")


def unmodeled_thinking_flag(family: str, kwargs: dict[str, Any]) -> bool:
    """Whether Dynamo rewrites the request's thinking flag before it renders
    (`normalize_thinking_aliases`). The first of `thinking`, `enable_thinking`
    and `thinking_mode` that holds a bool, string or number decides. Dynamo turns
    a string or number in the first two, or a bool, number or on/off word in
    `thinking_mode`, into bool `thinking` and `enable_thinking`, so the recorded
    prompt is not the one Dynamo serves.

    For MiniMax M3 (`normalize_minimax_m3_thinking_mode`) Dynamo writes that
    bool back as `thinking_mode` "enabled" or "disabled", which the template
    reads, so only a native `thinking_mode` value renders as recorded."""
    for key in THINKING_KEYS:
        value = kwargs.get(key)
        if family == "minimax_m3" and isinstance(value, (bool, int, float, str)):
            return key != "thinking_mode" or value not in MINIMAX_M3_MODES
        if isinstance(value, bool):
            return key == "thinking_mode"
        if isinstance(value, (int, float)):
            return True
        if isinstance(value, str):
            if key != "thinking_mode":
                return True
            switch = value.lower() in ("enabled", "disabled")
            return switch or value.strip().lower() in BOOL_WORDS
    return False


def minimax_m3_forces_thinking_off(
    request: dict[str, Any], kwargs: dict[str, Any]
) -> bool:
    """Whether Dynamo sets M3's `thinking_mode` to "disabled" because the output
    is constrained and the client set no thinking control
    (`has_minimax_m3_constrained_generation`)."""
    if request.get("thinking") is not None or any(key in kwargs for key in THINKING_KEYS):
        return False
    response_format = request.get("response_format") or {}
    structured = response_format.get("type") in ("json_schema", "json_object")
    return structured or any(key.startswith("guided_") for key in request)


def unserved_reason(
    family: str, request: dict[str, Any], kwargs: dict[str, Any]
) -> str | None:
    """The skip reason for a request Dynamo would not serve as this round trip
    models it, or None."""
    if family in UNMODELED_FAMILIES:
        return "unmodeled-family"
    if request.get("stop"):
        # The frontend trims a matched stop string, which x does not model.
        return "stop-strings"
    choice = request.get("tool_choice")
    if choice == "required" or isinstance(choice, dict):
        # Dynamo serves these with guided JSON, not the native markup parsed here,
        # and turns thinking off for a named choice under Kimi K3.
        return "guided-tool-choice"
    if unmodeled_thinking_flag(family, kwargs):
        return "unmodeled-thinking-flag"
    if family == "minimax_m3" and minimax_m3_forces_thinking_off(request, kwargs):
        return "unmodeled-thinking-flag"
    return None


def reasoning_disabled(family: str, kwargs: dict[str, Any], prompt: str) -> bool:
    """Whether Dynamo's Unified path serves reasoning as content for this request
    (`request_disables_reasoning_with_family`), given a flag Dynamo does not
    rewrite (`unmodeled_thinking_flag`): `thinking`, else `enable_thinking`,
    decides when it is a bool. A deployment's default thinking mode is not
    modeled. MiniMax M3 turns reasoning off only when its rendered prompt also
    closed the thought with `</mm:think>`."""
    flags = (kwargs.get(key) for key in ("thinking", "enable_thinking"))
    thinking = next((flag for flag in flags if isinstance(flag, bool)), None)
    if family in REASONING_ALWAYS_ON:
        return False
    if family == "minimax_m3":
        closed = prompt.rstrip().endswith(MINIMAX_M3_CLOSER)
        return kwargs.get("thinking_mode") == "disabled" and closed
    if family == "gemma4":
        return thinking is not True
    if family in ("deepseek_v4", "deepseek_v41"):
        # Otherwise `thinking_mode` decides, and thinking defaults on.
        return thinking is False or (
            thinking is None and kwargs.get("thinking_mode") == "chat"
        )
    return thinking is False


def opens_reasoning(family: str, prompt: str) -> bool:
    """Whether the generation prompt ends with the family's reasoning opener;
    Kimi K3 ignores whitespace inside its opener."""
    prompt = prompt.rstrip()
    if family == "kimi_k3":
        prompt = "".join(prompt.split())
    return prompt.endswith(REASONING_OPENERS.get(family, ("<think>",)))


def beam_open_channel(prompt: str) -> str:
    """The channel Beam's prompt leaves open after its last `<|start|>assistant`
    (`beam_prompt_open_channel`): none for the bare header slot, response for an
    open final message, reasoning for an open analysis message."""
    at = prompt.rfind(BEAM_HEADER)
    if at < 0:
        return "none"
    tail = prompt[at + len(BEAM_HEADER) :]

    def is_open(opener: str, closers: tuple[str, ...]) -> bool:
        start = tail.rfind(opener)
        rest = tail[start + len(opener) :]
        return start >= 0 and not any(closer in rest for closer in closers)

    if is_open("<|channel|>final<|message|>", ("<|return|>", "<|end|>")):
        return "response"
    if is_open("<|channel|>analysis<|message|>", ("<|end|>",)):
        return "reasoning"
    return "none"


def starting_state(family: str, prompt: str, disabled: bool) -> str:
    """The channel the generation prompt leaves open, as Dynamo infers it
    (`prompt_reasoning_prefill_for_parsers`, then `stream_prefill` and
    `UnifiedRequestPolicy::starting_state` in unified_parser.rs): reasoning when
    the prompt ends with the family's opener, even with thinking off. Gemma 4,
    MiniMax M3 and Nemotron 3 with thinking off start in the response channel,
    and Nemotron 3 otherwise always starts in reasoning. Gemma 4's partial
    thought header is not modeled. Beam reads its open channel instead."""
    if family == "beam":
        return beam_open_channel(prompt)
    if disabled and family in RESPONSE_WHEN_DISABLED:
        return "response"
    if opens_reasoning(family, prompt) or family == "nemotron_v3":
        return "reasoning"
    return "none"


def function_tools(request: dict[str, Any]) -> list[dict[str, Any]]:
    """The request's function tools, the only kind the parser is given. Tools a
    Kimi-style system message declares, which Dynamo also passes, are not."""
    tools = request.get("tools") or []
    return [
        tool
        for tool in tools
        if isinstance(tool, dict) and isinstance(tool.get("function"), dict)
    ]
