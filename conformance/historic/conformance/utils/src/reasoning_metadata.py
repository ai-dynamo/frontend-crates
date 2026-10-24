# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reasoning labels, row routing, and parser descriptions shared by readers."""

import re

CASE_GROUPS = [
    (
        "Core",
        ("batch.1.a", "batch.1.b"),
    ),
    (
        "Reasoning extraction",
        (
            "batch.2.a",
            "batch.2.b",
            "batch.2.c",
            "batch.2.d",
            "batch.2.e",
            "batch.2.f",
        ),
    ),
    (
        "Tool call boundary",
        (
            "batch.3.a",
            "batch.3.b",
            "batch.3.c",
            "batch.3.d",
            "batch.3.e",
            "batch.3.f",
        ),
    ),
    ("Malformed / recovery", ("batch.4", "batch.5")),
    ("Multi-span", ("batch.6.a", "batch.6.b")),
    (
        "Core",
        ("stream.1.a", "stream.1.b"),
    ),
    ("Reasoning extraction", ("stream.2.a",)),
    ("Multi-span", ("stream.2.b", "stream.2.c")),
    ("Chunk boundaries", ("stream.3.a", "stream.3.b", "stream.3.c")),
]

PARSER_TO_REASONING_FAMILY = {
    "deepseek_v3": "deepseek_v3",
    "deepseek_v3_1": "deepseek_v3",
    "deepseek_v3_2": "deepseek_v3",
    "deepseek_v4": "deepseek_v4",
    "gemma4": "gemma4",
    "glm47": "nemotron_deci",
    "harmony": "gpt_oss",
    "kimi_k2": "kimi_k25",
    "minimax_m2": "minimax_append_think",
    "minimax_m3": "minimax_m3",
    "mistral": "mistral",
    "nemotron_deci": "nemotron_deci",
    "nemotron_nano": "deepseek_r1",
    "qwen3_coder": "qwen3",
}

_REASONING_LABEL_OVERRIDES = {"nemotron_nano": "Nemotron V3‡"}

_REASONING_TOP_N_APPEND = ["nemotron_nano"]

_REASONING_HIDDEN_TOOL_FAMILIES = {"nemotron_deci"}

_FAMILY_METADATA = {
    "basic": {
        "models": ["Generic CoT models"],
        "rust_enum": "ReasoningParserType::Basic",
        "implementation": "BasicReasoningParser `<think>` / `</think>`",
        "shared_with": ["qwen3", "deepseek_v4", "nemotron_deci", "glm45"],
    },
    "qwen3": {
        "models": ["Qwen3.5", "QwQ-32B", "Qwen3-Think", "Qwen3-Coder"],
        "rust_enum": "ReasoningParserType::Qwen",
        "implementation": "BasicReasoningParser `<think>` / `</think>`",
        "shared_with": ["basic", "deepseek_v4", "nemotron_deci", "glm45"],
    },
    "deepseek_v4": {
        "models": ["DeepSeek V4 Pro", "DeepSeek V4 Flash"],
        "rust_enum": "ReasoningParserType::DeepSeekV4",
        "implementation": "BasicReasoningParser `<think>` / `</think>`",
        "shared_with": ["basic", "qwen3", "nemotron_deci", "glm45"],
        "aliases": ["deepseek-v4", "deepseekv4"],
    },
    "nemotron_deci": {
        "models": [
            "Nemotron-Super-v1 / Nemotron-Ultra-v1 / Nemotron-Deci-v1",
            "Llama-Nemotron",
            "GLM-4.5 / GLM-4.6 via glm45 alias",
        ],
        "rust_enum": "ReasoningParserType::NemotronDeci",
        "implementation": "BasicReasoningParser `<think>` / `</think>`",
        "shared_with": ["basic", "qwen3", "deepseek_v4", "glm45"],
        "aliases": ["glm45"],
    },
    "deepseek_r1": {
        "models": [
            "DeepSeek R1",
            "DeepSeek V3.x aliases",
            "Nemotron force-reasoning aliases",
        ],
        "rust_enum": "ReasoningParserType::DeepseekR1",
        "implementation": (
            "BasicReasoningParser `<think>` / `</think>`, force_reasoning=true"
        ),
        "shared_with": [
            "deepseek_v3",
            "step3",
            "nemotron_nano",
            "nemotron3",
            "nemotron_v3",
        ],
    },
    "deepseek_v3": {
        "models": ["DeepSeek V3", "DeepSeek V3.1", "DeepSeek V3.2"],
        "rust_enum": "ReasoningParserType::DeepseekR1",
        "implementation": (
            "BasicReasoningParser `<think>` / `</think>`, force_reasoning=true"
        ),
        "shared_with": [
            "deepseek_r1",
            "step3",
            "nemotron_nano",
            "nemotron3",
            "nemotron_v3",
        ],
        "aliases": ["deepseek_v3_1", "deepseek_v3_2"],
    },
    "kimi": {
        "models": ["Kimi K2 Instruct / Thinking using Unicode think delimiters"],
        "rust_enum": "ReasoningParserType::Kimi",
        "implementation": "BasicReasoningParser `◁think▷` / `◁/think▷`",
    },
    "kimi_k25": {
        "models": ["Kimi K2.5 / Kimi K2.6 style `<think>` force-reasoning models"],
        "rust_enum": "ReasoningParserType::KimiK25",
        "implementation": (
            "BasicReasoningParser `<think>` / `</think>`, "
            "force_reasoning=true, Kimi tool-section exit"
        ),
    },
    "mistral": {
        "models": ["Magistral"],
        "rust_enum": "ReasoningParserType::Mistral",
        "implementation": (
            "BasicReasoningParser `[THINK]` / `[/THINK]`, force_reasoning=true"
        ),
    },
    "granite": {
        "models": ["IBM Granite 3.x", "IBM Granite 3.2 language models"],
        "rust_enum": "ReasoningParserType::Granite",
        "implementation": "GraniteReasoningParser",
    },
    "gpt_oss": {
        "models": ["gpt-oss-20b", "gpt-oss-120b"],
        "rust_enum": "ReasoningParserType::GptOss",
        "implementation": "GptOssReasoningParser / Harmony StreamableParser",
    },
    "minimax_append_think": {
        "models": ["MiniMax M2", "MiniMax M2.1"],
        "rust_enum": "ReasoningParserType::MiniMaxAppendThink",
        "implementation": "MiniMaxAppendThinkParser",
    },
    "minimax_m3": {
        "models": ["MiniMax M3"],
        "rust_enum": "ReasoningParserType::MiniMaxM3",
        "implementation": (
            "BasicReasoningParser `<mm:think>` / `</mm:think>`, "
            "dangling-end recovery"
        ),
        "aliases": ["minimax-m3"],
    },
    "gemma4": {
        "models": ["Google Gemma 4 thinking models"],
        "rust_enum": "ReasoningParserType::Gemma4",
        "implementation": "Gemma4ReasoningParser",
        "aliases": ["gemma-4"],
    },
}

_REASONING_MODE_METADATA = {
    "basic": {
        "label": "explicit markers",
        "control": "mostly static",
        "summary": (
            "Reasoning starts only after an opening `<think>` marker or a "
            "prompt-injected start state."
        ),
        "static": [
            "BasicReasoningParser `<think>` / `</think>`",
            "force_reasoning=false",
            "stream_reasoning=true",
        ],
    },
    "qwen3": {
        "label": "explicit markers",
        "control": "frontend-tunable",
        "summary": (
            "Qwen-style reasoning uses explicit `<think>` markers; templates "
            "may inject the opening marker."
        ),
        "static": [
            "BasicReasoningParser `<think>` / `</think>`",
            "force_reasoning=false",
            "stream_reasoning=true",
        ],
    },
    "deepseek_v4": {
        "label": "explicit markers",
        "control": "frontend-tunable",
        "summary": (
            "DeepSeek V4 uses `<think>` markers, but the prompt formatter "
            "controls whether thinking is enabled."
        ),
        "static": [
            "BasicReasoningParser `<think>` / `</think>`",
            "force_reasoning=false",
            "stream_reasoning=true",
        ],
    },
    "nemotron_deci": {
        "label": "explicit markers",
        "control": "frontend-tunable",
        "summary": "GLM/Nemotron-Deci style parsing uses explicit `<think>` markers.",
        "static": [
            "BasicReasoningParser `<think>` / `</think>`",
            "force_reasoning=false",
            "stream_reasoning=true",
        ],
    },
    "deepseek_r1": {
        "label": "force reasoning",
        "control": "frontend-tunable",
        "summary": (
            "Generation may begin already inside reasoning; marker-free text "
            "can be reasoning until an end marker."
        ),
        "static": [
            "BasicReasoningParser `<think>` / `</think>`",
            "force_reasoning=true",
            "stream_reasoning=true",
        ],
    },
    "deepseek_v3": {
        "label": "force reasoning",
        "control": "frontend-tunable",
        "summary": "DeepSeek V3.x routes to the R1 force-reasoning implementation in Dynamo.",
        "static": [
            "Rust enum: ReasoningParserType::DeepseekR1",
            "BasicReasoningParser `<think>` / `</think>`",
            "force_reasoning=true",
            "stream_reasoning=true",
        ],
    },
    "kimi": {
        "label": "explicit unicode",
        "control": "mostly static",
        "summary": "Kimi legacy reasoning uses Unicode delimiters instead of `<think>`.",
        "static": [
            "BasicReasoningParser `◁think▷` / `◁/think▷`",
            "force_reasoning=false",
            "stream_reasoning=true",
        ],
    },
    "kimi_k25": {
        "label": "force reasoning",
        "control": "frontend-tunable",
        "summary": (
            "Kimi K2.5 can start in reasoning and uses the tool call section "
            "marker as a reasoning escape boundary."
        ),
        "static": [
            "BasicReasoningParser `<think>` / `</think>`",
            "force_reasoning=true",
            "stream_reasoning=true",
            "tool_start_token=`<|tool_calls_section_begin|>`",
        ],
    },
    "mistral": {
        "label": "force reasoning",
        "control": "mostly static",
        "summary": (
            "Mistral/Magistral reasoning uses bracket markers and starts in "
            "force-reasoning mode."
        ),
        "static": [
            "BasicReasoningParser `[THINK]` / `[/THINK]`",
            "force_reasoning=true",
            "stream_reasoning=true",
        ],
    },
    "granite": {
        "label": "phrase markers",
        "control": "mostly static",
        "summary": "Granite reasoning is split by natural-language phrase markers.",
        "static": [
            "GraniteReasoningParser",
            "`Here is my thought process:` / `Here is my response:`",
        ],
    },
    "gpt_oss": {
        "label": "Harmony channels",
        "control": "mostly static",
        "summary": (
            "GPT-OSS reasoning is the Harmony `analysis` channel, parsed by "
            "the Harmony stream parser."
        ),
        "static": [
            "GptOssReasoningParser / Harmony StreamableParser",
            "requires special tokens to remain visible in decoded text",
        ],
    },
    "minimax_append_think": {
        "label": "append-think",
        "control": "mostly static",
        "summary": (
            "MiniMax append-think has its own contract instead of normal "
            "reasoning extraction."
        ),
        "static": [
            "MiniMaxAppendThinkParser",
            "parser-specific content wrapper behavior",
        ],
    },
    "minimax_m3": {
        "label": "explicit M3 markers",
        "control": "frontend-tunable",
        "summary": (
            "MiniMax M3 reasoning uses explicit `<mm:think>` markers, with "
            "dangling-end recovery for prompt-prefilled reasoning starts."
        ),
        "static": [
            "BasicReasoningParser `<mm:think>` / `</mm:think>`",
            "force_reasoning=false",
            "stream_reasoning=true",
            "recover_dangling_end=true",
        ],
    },
    "gemma4": {
        "label": "Gemma channels",
        "control": "frontend-tunable",
        "summary": "Gemma 4 reasoning is wrapped by channel markers with a `thought` role label.",
        "static": [
            "Gemma4ReasoningParser",
            "requires special tokens to remain visible in decoded text",
        ],
    },
}


_DEFAULT_REASONING_MARKUP_RE = re.compile(r"</?think>")

_NO_AUTO_REASONING_MARKUP_RE = re.compile(r"(?!)")

_REASONING_MARKUP_BY_FAMILY: dict[str, re.Pattern[str]] = {
    "deepseek_r1": re.compile(r"</?think>"),
    "deepseek_v3": re.compile(r"</?think>"),
    "deepseek_v4": re.compile(r"</?think>"),
    "gemma4": re.compile(r"<\|channel>|<channel\|>"),
    "gpt_oss": re.compile(r"<\|(?:start|end|return|channel|message)\|>"),
    "granite": re.compile(
        r"Here is my thought process:|Here is my response:|Here's my response:"
    ),
    "kimi": re.compile(r"◁/?think▷"),
    "kimi_k25": re.compile(r"</?think>"),
    "minimax_append_think": _NO_AUTO_REASONING_MARKUP_RE,
    "mistral": re.compile(r"\[/?THINK\]"),
    "nemotron_deci": re.compile(r"</?think>"),
    "qwen3": re.compile(r"</?think>"),
}
