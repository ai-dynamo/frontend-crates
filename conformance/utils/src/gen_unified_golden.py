# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render the unified golden corpus for ALL families from ONE scenario spec.

The GOLDEN event list is the authored oracle (best-effort error recovery, see
UNIFIED_CASES.md). Scenario meaning is shared; family-specific marker values and
the DSML missing-invoke-close contract are explicit exceptions. Native inputs use
each family's grammar, while request modes and payloads keep the same contract.

The family registry supplies the seven golden-spec paths under the gitignored
conformance/unified/golden_spec/ tree. Explicit scenario scopes and the redundant
DeepSeek V4.1 prefill rows determine applicability. This spec is the harness INPUT (unified_render.rs reads it to
compute the live Dynamo column; unified_schema_roundtrip.rs validates it); it is
NOT committed. The committed canonical family YAML is DERIVED from it via
render -> explode -> package, exactly like every other conformance fixture shard.

Run:  python3 conformance/utils/src/gen_unified_golden.py
"""
import json
from copy import deepcopy
from functools import lru_cache
import os
import re
from pathlib import Path

import yaml

import markers
from impls import unified_grammar_family
from schema_cases import (
    CONFORMANCE_CASES, XML_COMPATIBILITY_CASES, schema_description, schema_tools,
    unified_schema_cases,
)
from unified_taxonomy import GPT_OSS_PR424_REQUEST_VARIANTS, RETIRED_SCENARIOS, numbered_id
from numeric_cases import (
    NUMERIC_DESCRIPTIONS,
    NUMERIC_VARIANTS,
    SCHEMA_DIRECTED_FAMILIES,
    NumericLiteral,
    applicable,
    arguments_json,
    numeric_description,
    unified_numeric_expected,
    numeric_schema,
)
from null_cases import MIXED_CASE_FAMILIES, NULL_VARIANTS, MIXED_LABELS_SCHEMA, MIXED_LABELS_ARGS, null_description

# Families and their golden-spec filenames come from the ONE declaration in
# parser_families.yaml (`unified:`), so adding a family to this generator is adding a
# row there rather than editing three lists that had to agree.
_MANIFEST = yaml.safe_load(markers.parser_families_path().read_text())["unified"]
FAMILIES = sorted(_MANIFEST)
TEMPLATE_FAMILIES = [f for f in FAMILIES if not _MANIFEST[f].get("grammar")]
SHARED_FAMILIES = [f for f in FAMILIES if f != "deepseek_v41"]
FAM_FILE = {f: r["golden_spec"] for f, r in _MANIFEST.items()}
UNIFIED_FAMILIES = {f for f, r in _MANIFEST.items() if r.get("native")}
DOTTED_NAME_SCENARIOS = frozenset({
    "dotted_name_minimal",
    "dotted_name_mixed_punctuation",
    "dotted_name_multiple_segments",
    "dotted_name_shared_prefix",
})
DOTTED_NAME_FAMILIES = frozenset({
    "beam",
    "deepseek_v4",
    "deepseek_v41",
    "gemma4",
    "glm47",
    "gpt_oss",
    "kimi_k2",
    "kimi_k3",
    "muse_glimmer",
    "muse_spark",
    "nemotron_v3",
    "qwen3",
})
# Families whose unified parser accepts GUIDED tool output, declared in the manifest.
# Every family with a native unified parser does today. The gate is kept rather than
# deleted because "can this family honour a guided request mode" is a real per-family
# fact a new family can answer no to; what it must NOT be read as is "does this family
# have a reasoning marker pair", which is how muse_glimmer sat opted out while having a
# perfectly good reasoning channel routed by recipient. Prefilled channels are a
# separate axis and are supported everywhere, so they are not gated.
GUIDED_FAMILIES = {f for f, r in _MANIFEST.items() if r.get("guided_tool_output", True)}
MUSE_FAMILIES = {"muse_glimmer", "muse_spark"}

GRAMMAR_NOTE = {
    "beam": "Beam channel grammar (NOT Harmony): reasoning `<|start|>assistant<|channel|>analysis<|message|>...<|end|>`, visible `<|start|>assistant<|channel|>final<|message|>...<|return|>`, tool `<|start|>assistant to=NAME<|channel|>analysis <|constrain|>xml<|message|><rfl:parameter name=\"KEY\">\\nVALUE\\n</rfl:parameter><|call|>` (a call dispatches only once `<|call|>`/`<|end|>` arrives; analysis closes only on `<|end|>`; one LF is stripped at each value edge).",
    "deepseek_v41": "prompt-prefilled reasoning ends at `</think>`; `<｜DSML｜ calls>` contains V4.1 invoke and parameter tags and ends the turn.",
    "deepseek_v4": "reasoning `<think>...</think>`, tool `<｜DSML｜tool_calls><｜DSML｜invoke name=\"NAME\"><｜DSML｜parameter name=\"KEY\" string=\"true\">VALUE</｜DSML｜parameter></｜DSML｜invoke></｜DSML｜tool_calls>`.",
    "gemma4": "reasoning `<|channel>thought\\n...<channel|>`, tool `<|tool_call>call:NAME{key:<|\"|>value<|\"|>}<tool_call|>` (string values wrapped in `<|\"|>`; an embedded `<tool_call|>` inside a `<|\"|>` string is data, not the end marker).",
    "glm47": "reasoning `<think>...</think>`, tool `<tool_call>NAME<arg_key>KEY</arg_key><arg_value>VALUE</arg_value></tool_call>`.",
    "qwen3": "reasoning `<think>...</think>`, tool `<tool_call><function=NAME><parameter=KEY>VALUE</parameter></function></tool_call>`.",
    "kimi_k2": "reasoning `<think>...</think>`, tool section `<|tool_calls_section_begin|><|tool_call_begin|>functions.NAME:IDX<|tool_call_argument_begin|>{...}<|tool_call_end|><|tool_calls_section_end|>`.",
    "kimi_k3": "reasoning `<|open|>think<|sep|>...<|close|>think<|sep|>`, tool `<|open|>tools<|sep|><|open|>call tool=\"NAME\" index=\"IDX\"<|sep|><|open|>argument key=\"KEY\" type=\"string\"<|sep|>VALUE<|close|>argument<|sep|><|close|>call<|sep|><|close|>tools<|sep|>`.",
    "muse_glimmer": "recipient-routed messages `<|start|>assistant to=RCPT<|message|>...<|eom|>`: `self` is reasoning, `user` is visible content, any other recipient opens a tool channel whose body is ATEM XML `<atem:function_calls><atem:invoke name=\"NAME\"><atem:parameter name=\"KEY\">VALUE</atem:parameter></atem:invoke></atem:function_calls>`. `<|eom|>` closes a message with more to follow, `<|eot|>` ends the turn. Spec: https://huggingface.co/meta-models/Muse-Glimmer-30B.",
    "gpt_oss": "GPT-OSS 20B and 120B templates allow `analysis`, `commentary`, and `final` channels. Reasoning uses `analysis`; visible output uses `final`; tool calls use `assistant to=functions.NAME<|channel|>commentary json<|message|>...<|call|>`. Specs: https://huggingface.co/openai/gpt-oss-20b/blob/main/chat_template.jinja and https://huggingface.co/openai/gpt-oss-120b/blob/main/chat_template.jinja.",
    "muse_spark": "Muse Glimmer's recipient-routed ATEM grammar plus the reserved `commentary` recipient, which is user-visible intermediate text rather than a tool channel. `self` is reasoning and `user` is final visible content.",
}


# --- grammar renderers: one semantic segment -> that family's raw text --------

def r_reason(fam, text):
    if fam == "gpt_oss":
        return f"<|start|>assistant<|channel|>analysis<|message|>{text}<|end|>"
    if fam == "beam":
        return f"<|start|>assistant<|channel|>analysis<|message|>{text}<|end|>"
    if fam == "kimi_k3":
        return k3_channel("think", text)
    if fam == "gemma4":
        return f"<|channel>thought\n{text}<channel|>"
    if fam in MUSE_FAMILIES:
        return f"<|start|>assistant to=self<|message|>{text}<|eom|>"
    return f"<think>{text}</think>"


def r_text(fam, text):
    """Visible content.

    Muse has no unframed content channel — every message is recipient-routed —
    so visible text renders as a closed `to=user` message. `<|eom|>` and not
    `<|eot|>`: `<|eot|>` ends the TURN, which would make any following segment
    unreachable. The marker-pair grammars leave visible text bare.
    """
    if fam == "gpt_oss":
        return f"<|start|>assistant<|channel|>final<|message|>{text}<|return|>"
    if fam in MUSE_FAMILIES:
        return f"<|start|>assistant to=user<|message|>{text}<|eom|>"
    if fam == "beam":
        return f"<|start|>assistant<|channel|>final<|message|>{text}<|return|>"
    if fam == "kimi_k3":
        return k3_channel("response", text)
    return text


def _atem_value(val):
    """One ATEM parameter value.

    The Muse decode spec types parameter values with `value_parser: json` and
    `allow_non_json: true`, so a bare `1` types as the NUMBER 1 while every other
    family's grammar types it as the string "1". The shared golden says string, so
    emit the JSON spelling exactly when the bare form would not produce one — the
    common case (`Paris`) does not parse as JSON and stays bare and byte-preserving.

    A value that is ITSELF a JSON string (`"hi"`) parses, and it parses to `hi`, not
    to `"hi"`: bare, the parser reads the quotes as syntax and drops them. So it needs
    the quoted spelling like every other value that parses. Keeping it bare authored a
    golden no correct parser can emit.
    """
    if isinstance(val, NumericLiteral):
        return str(val)
    try:
        json.loads(val)
    except ValueError:
        return val
    return json.dumps(val)


def k3_open(tag, attrs=(), spaced=False):
    gap = " " if spaced else ""
    rendered_attrs = "".join(f' {key}="{value}"' for key, value in attrs)
    return f"<|open|>{gap}{tag}{rendered_attrs}{gap}<|sep|>"


def k3_close(tag, spaced=False):
    gap = " " if spaced else ""
    return f"<|close|>{gap}{tag}{gap}<|sep|>"


def k3_channel(tag, text, spaced=False):
    return f"{k3_open(tag, spaced=spaced)}{text}{k3_close(tag, spaced=spaced)}"


def k3_argument(key, arg_type, value, spaced=False):
    return (
        f"{k3_open('argument', [('key', key), ('type', arg_type)], spaced)}"
        f"{value}{k3_close('argument', spaced)}"
    )


def k3_json(raw, spaced=False):
    return (
        f"{k3_open('json', [('type', 'object')], spaced)}"
        f"{raw}{k3_close('json', spaced)}"
    )


def k3_call(name, index, body, *, close=True, spaced=False):
    rendered = f"{k3_open('call', [('tool', name), ('index', str(index))], spaced)}{body}"
    return rendered + (k3_close("call", spaced) if close else "")


def k3_tools(body, *, close=True, spaced=False):
    rendered = f"{k3_open('tools', spaced=spaced)}{body}"
    return rendered + (k3_close("tools", spaced) if close else "")


def k3_raw_tool(name, raw, index=1, *, close=True, spaced=False):
    return k3_tools(
        k3_call(name, index, k3_json(raw, spaced), close=close, spaced=spaced),
        close=close,
        spaced=spaced,
    )


def _gemma_value(value):
    if isinstance(value, NumericLiteral):
        return str(value)
    if isinstance(value, str):
        return f'<|"|>{value}<|"|>'
    if isinstance(value, dict):
        return "{" + ",".join(f"{key}:{_gemma_value(item)}" for key, item in value.items()) + "}"
    if isinstance(value, list):
        return "[" + ",".join(_gemma_value(item) for item in value) + "]"
    return json.dumps(value, ensure_ascii=False)


def r_tool(fam, name, key, val, idx):
    return r_tool_arguments(fam, name, {key: val}, idx)


def _json_with_numeric_literals(value):
    if isinstance(value, NumericLiteral):
        return str(value)
    if isinstance(value, dict):
        return "{" + ",".join(
            json.dumps(key, ensure_ascii=False) + ":" + _json_with_numeric_literals(item)
            for key, item in value.items()
        ) + "}"
    if isinstance(value, list):
        return "[" + ",".join(_json_with_numeric_literals(item) for item in value) + "]"
    return json.dumps(value, ensure_ascii=False)


def _contains_numeric_literal(value):
    if isinstance(value, NumericLiteral):
        return True
    if isinstance(value, dict):
        return any(_contains_numeric_literal(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_numeric_literal(item) for item in value)
    return False


def r_tool_arguments(fam, name, arguments, idx, raw_arguments=None):
    fam = unified_grammar_family(fam)
    # Raw spellings preserve published stimuli independently of the typed oracle.
    raw = raw_arguments if raw_arguments is not None else {
        key: str(value) if isinstance(value, NumericLiteral)
        else value if isinstance(value, str) else _json_with_numeric_literals(value)
        for key, value in arguments.items()
    }
    if fam == "gpt_oss":
        args = (_json_with_numeric_literals(arguments)
                if _contains_numeric_literal(arguments)
                else json.dumps(arguments, ensure_ascii=False))
        return (f"<|start|>assistant<|channel|>commentary to=functions.{name} "
                f"<|constrain|>json<|message|>{args}<|call|>")
    if fam in ("deepseek_v4", "deepseek_v41"):
        gap = " " if fam == "deepseek_v41" else ""
        envelope = "calls" if gap else "tool_calls"
        params = "".join(
            f'<｜DSML｜{gap}parameter name="{key}" string="{str(isinstance(value, str) and not isinstance(value, NumericLiteral)).lower()}">{raw[key]}</｜DSML｜{gap}parameter>'
            for key, value in arguments.items()
        )
        return (f'<｜DSML｜{gap}{envelope}><｜DSML｜{gap}invoke name="{name}">'
                f'{params}</｜DSML｜{gap}invoke></｜DSML｜{gap}{envelope}>')
    if fam == "beam":
        params = "".join(
            f'<rfl:parameter name="{key}">\n{raw[key]}\n</rfl:parameter>' for key in arguments
        )
        return (f"<|start|>assistant to={name}<|channel|>analysis <|constrain|>xml<|message|>"
                f"{params}<|call|>")
    if fam == "gemma4":
        native_values = {key: value for key, value in arguments.items()}
        return f"<|tool_call>call:{name}{_gemma_value(native_values)}<tool_call|>"
    if fam == "qwen3":
        params = "\n".join(
            f"<parameter={key}>\n{value}\n</parameter>" for key, value in raw.items()
        )
        return f"<tool_call>\n<function={name}>\n{params}\n</function>\n</tool_call>"
    if fam == "glm47":
        params = "".join(
            f"<arg_key>{key}</arg_key><arg_value>{value}</arg_value>"
            for key, value in raw.items()
        )
        return f"<tool_call>{name}{params}</tool_call>"
    if fam in MUSE_FAMILIES:
        params = "".join(
            f'<atem:parameter name="{key}">{_atem_value(raw[key]) if isinstance(value, str) and not isinstance(value, NumericLiteral) else raw[key]}</atem:parameter>\n'
            for key, value in arguments.items()
        )
        return (f"<|start|>assistant to={name}<|message|><atem:function_calls>\n"
                f'<atem:invoke name="{name}">\n{params}'
                f"</atem:invoke>\n</atem:function_calls><|eom|>")
    if fam == "kimi_k3":
        types = {str: "string", type(None): "null", bool: "boolean", int: "integer",
                 float: "number", dict: "object", list: "array"}
        params = "".join(k3_argument(
            key, "number" if isinstance(value, NumericLiteral) else types[type(value)], raw[key]
        ) for key, value in arguments.items())
        return k3_tools(k3_call(name, idx + 1, params))
    assert fam == "kimi_k2", fam
    args = (_json_with_numeric_literals(arguments) if _contains_numeric_literal(arguments)
            else json.dumps(arguments, ensure_ascii=False))
    return (f"<|tool_calls_section_begin|><|tool_call_begin|>functions.{name}:{idx}"
            f"<|tool_call_argument_begin|>{args}<|tool_call_end|><|tool_calls_section_end|>")

def qwen3_input_as_glm47(input_text):
    """Translate a Qwen-shaped edge fixture into GLM XML."""
    function = re.compile(r"<function=([^>]+)>(.*?)</function>", re.DOTALL)
    parameter = re.compile(r"<parameter=([^>]+)>(.*?)</parameter>", re.DOTALL)

    def convert_function(match):
        name, body = match.groups()
        rendered = parameter.sub(
            lambda parameter_match: (
                f"<arg_key>{parameter_match.group(1).strip()}</arg_key>"
                f"<arg_value>{parameter_match.group(2).strip()}</arg_value>"
            ),
            body,
        )
        return f"{name}{rendered.strip()}"

    converted = function.sub(convert_function, input_text)
    converted = converted.replace("<tool_call>\n", "<tool_call>")
    converted = re.sub(r"<function=([^>]+)>\n?", r"\1", converted)
    converted = re.sub(
        r"<parameter=([^>]+)>\n?", r"<arg_key>\1</arg_key><arg_value>", converted
    )
    converted = converted.replace("\n</parameter>", "</arg_value>")
    converted = converted.replace("\n</function>", "")
    return converted.replace("\n</tool_call>", "</tool_call>")


def kimi_input_as_dsml(input_text):
    """Translate a Kimi-shaped edge fixture into DSML without changing its meaning."""
    text = input_text.replace("<|tool_calls_section_begin|>", "<｜DSML｜tool_calls>")
    text = text.replace("<|tool_calls_section_end|>", "</｜DSML｜tool_calls>")
    opener = "<|tool_call_begin|>functions."
    args_marker = "<|tool_call_argument_begin|>"
    end = "<|tool_call_end|>"
    start = text.find(opener)
    if start < 0:
        return text.replace(end, "</｜DSML｜invoke>")
    args = text.find(args_marker, start)
    if args < 0:
        return text.replace(end, "</｜DSML｜invoke>")
    name_and_index = text[start + len(opener):args]
    name = name_and_index.rsplit(":", 1)[0]
    close = text.rfind(end)
    if close < args:
        # The incomplete source still needs DSML's incomplete invoke spelling.
        return (text[:start] + f'<｜DSML｜invoke name="{name}">' +
                text[args + len(args_marker):].replace(end, "</｜DSML｜invoke>"))
    raw = text[args + len(args_marker):close]
    try:
        arguments = json.loads(raw)
    except json.JSONDecodeError:
        return (text[:start] + f'<｜DSML｜invoke name="{name}">{raw}</｜DSML｜invoke>' +
                text[close + len(end):].replace(end, "</｜DSML｜invoke>"))
    if not isinstance(arguments, dict):
        return (text[:start] + f'<｜DSML｜invoke name="{name}">{raw}</｜DSML｜invoke>' +
                text[close + len(end):].replace(end, "</｜DSML｜invoke>"))
    params = []
    for key, value in arguments.items():
        if isinstance(value, str):
            value_text = value
            if value_text == end:
                value_text = "</｜DSML｜invoke>"
            attr = ' string="true"'
        else:
            value_text, attr = json.dumps(value, ensure_ascii=False), ' string="false"'
        params.append(f'<｜DSML｜parameter name="{key}"{attr}>{value_text}</｜DSML｜parameter>')
    return (text[:start] + f'<｜DSML｜invoke name="{name}">' + ''.join(params) +
            '</｜DSML｜invoke>' + text[close + len(end):].replace(end, "</｜DSML｜invoke>"))


def tool_segment(segment):
    if len(segment) == 3:
        return segment[1], segment[2]
    _, name, key, value = segment
    return name, {key: value}


def render_input(fam, segs):
    """Concatenate rendered segments (grammars are self-delimiting)."""
    out = []
    tool_idx = 0
    index = 0
    while index < len(segs):
        s = segs[index]
        if s[0] == "reason":
            out.append(r_reason(fam, s[1]))
        elif s[0] == "text":
            out.append(r_text(fam, s[1]))
        elif s[0] == "tool":
            if fam == "kimi_k3":
                calls = []
                while index < len(segs) and segs[index][0] == "tool":
                    name, arguments = tool_segment(segs[index])
                    calls.append(r_tool_arguments(fam, name, arguments, tool_idx)
                                 .removeprefix(k3_open("tools")).removesuffix(k3_close("tools")))
                    tool_idx += 1
                    index += 1
                out.append(k3_tools("".join(calls)))
                continue
            name, arguments = tool_segment(s)
            out.append(r_tool_arguments(fam, name, arguments, tool_idx))
            tool_idx += 1
        index += 1
    return "".join(out)


def golden_of(segs):
    ev = []
    for s in segs:
        if s[0] == "reason":
            ev.append({"kind": "reasoning", "text": s[1]})
        elif s[0] == "text":
            ev.append({"kind": "text", "text": s[1]})
        elif s[0] == "tool":
            name, arguments = tool_segment(s)
            ev.append({"kind": "tool_call", "name": name, "arguments": arguments})
    return ev


# --- verdict shorthands -------------------------------------------------------

M = {"verdict": "match"}


def D(cls, note):
    return {"verdict": "diverge", "class": cls, "note": note}


# --- per-family input helpers for EDGE scenarios ------------------------------

class OnlyFamilies(dict):
    """A per-family map that DECLARES its scenario applies to a subset of families.

    Absence from a PLAIN dict must stay a hard failure: an accidentally omitted family
    is missing coverage, and letting it read as "not applicable" hides exactly what this
    corpus exists to measure. So the narrow scope is a statement the scenario makes
    about itself, carried by its own type, rather than something inferred from a gap.

    Use for grammar-specific scenarios or explicitly scoped regression reproductions,
    and state the scope at the authoring site and in UNIFIED_CASES.md. An undeclared
    omission is a coverage gap, not a scope.
    Disable grammar inheritance when a request needs a separately registered capability.
    """

    def __init__(self, mapping, *, inherit_grammar=True):
        super().__init__(mapping)
        self.inherit_grammar = inherit_grammar
        if not self:
            raise ValueError("OnlyFamilies() with no families declares nothing")
        unknown = sorted(set(self) - set(FAMILIES))
        if unknown:
            raise ValueError(f"OnlyFamilies() names families that do not exist: {unknown}")


def every_family(input_text, vllm, dynamo, *rest):
    """One input for EVERY family.

    Guided decoding is a BACKEND feature: it constrains the model to bare JSON,
    so the family's own grammar never appears in the payload and there is nothing
    to render per family. Writing these per family is how gemma4 and kimi_k2 ended
    up carrying NATIVE markup under an `init.tool_output_mode=GuidedJson` label —
    a case that renders green while testing nothing, because the parser was handed
    the one input shape the mode it declares never produces.
    """
    # The `dynamo` verdict is applied ONLY to families that actually have a native
    # unified parser. A family still on the v1-reasoning + v2-tool split ignores
    # `init` entirely, so it cannot honour a guided request mode — it emits the
    # payload as text. Recording `match` for it would be a false claim in the spec:
    # nothing asserts this field (the Dynamo column is computed live), so it would
    # never fail, it would just quietly mislead anyone reading the corpus.
    split = D(
        "UNSUPPORTED",
        "no native unified parser in this build, so the split path ignores `init` "
        "and cannot honour a guided request mode",
    )
    return {
        fam: (input_text, vllm, dynamo if fam in UNIFIED_FAMILIES else split, *rest)
        for fam in TEMPLATE_FAMILIES
    }


def by_family(render, vllm, dynamo, *rest):
    """`render(fam) -> input` for the scenarios where only the reasoning envelope
    around an otherwise identical payload is grammar-specific."""
    return {fam: (render(fam), vllm, dynamo, *rest) for fam in TEMPLATE_FAMILIES}


# A family whose tool block opener spans more than its first control token, mapped to
# the marker the opener runs THROUGH. Absent means the first token is the whole opener.
_TOOL_OPEN_THROUGH = {fam: "<atem:function_calls>" for fam in MUSE_FAMILIES} | {"beam": "<|message|>"}

# A family whose OUTER message terminator is shared across channels, mapped to the
# token that closes its tool STRUCTURE specifically. Absent means the two are the
# same and `control_tokens`' closer already distinguishes them.
#
# Muse ends every message with `<|eom|>`, whatever channel it was routed to, so the
# last token of a rendered CALL is the same token that ends a THOUGHT. That made
# `guided_json_orphan_tool_close_before_payload` render bytes identical to
# `guided_json_orphan_reason_close_before_payload` — two scenario names for one
# input, which the corpus rejects and which would drift apart on the next edit.
#
# Used ONLY by that scenario, not by `control_tokens`. Outside a tool channel this
# family reads ATEM as ordinary text (its safety rule against prose that quotes a
# call), so a bare `</atem:function_calls>` is NOT "tool markup that emits nothing"
# natively, and `tool_markup_only_emits_nothing` must keep rendering `<|eom|>`.
# Under GUIDED decoding the payload is bare JSON by construction, so native markup
# around it is stray no matter which marker it is — which is what this scenario asks.
_STRAY_TOOL_CLOSE = {fam: "</atem:function_calls>" for fam in MUSE_FAMILIES}


def stray_tool_close(fam):
    """The token a guided case means by 'an orphan TOOL closer'."""
    return _STRAY_TOOL_CLOSE.get(fam, control_tokens(fam)[3])


def control_tokens(fam):
    """Bare control tokens for `fam`, DERIVED from the renderers the corpus already
    uses (`r_reason` / `r_tool`) rather than a second grammar table — a parallel
    marker map is the kind of divergent copy that goes stale the first time a
    family's grammar moves.

    The tool pair is the OUTER wrapper: the first and last control tokens of a
    rendered call. Splitting on the tool NAME instead returns the inner fragment
    (`call:` for gemma4, `<function=` for qwen3, `functions.` plus the call-begin
    marker for kimi_k2), which is not the envelope these cases mean to place around
    a payload.

    Returns `(reason_open, reason_close, tool_open, tool_close)`.
    """
    if fam == "gpt_oss":
        return ("<|start|>assistant<|channel|>analysis<|message|>",
                "<|end|>", "<|call|>", "<|call|>")
    if fam == "kimi_k3":
        return k3_open("think"), k3_close("think"), k3_open("tools"), k3_close("tools")
    reason_open, reason_close = r_reason(fam, "\x00").split("\x00")
    rendered = r_tool(fam, "NAMEX", "KEYX", "VALX", 0)
    tokens = re.findall(r"<[^<>]*>", rendered)
    # The opener is the FIRST token only for a family whose block starts with one
    # marker. Muse opens a tool block with a routed header AND the block marker
    # (`<|start|>assistant to=NAME<|message|><atem:function_calls>`), so its first
    # token alone is `<|start|>`, which opens nothing. A case built from that token
    # tests prose after stray framing rather than an unterminated envelope, which is
    # a different scenario wearing this one's name.
    through = _TOOL_OPEN_THROUGH.get(fam)
    tool_open = rendered[: rendered.index(through) + len(through)] if through else tokens[0]
    return reason_open, reason_close, tool_open, tokens[-1]


def invoke_header_prefix(fam):
    """Inner invoke header through the tool name, without its terminator."""
    if fam == "gpt_oss":
        return ""
    if fam == "kimi_k3":
        return '<|open|>call tool="'
    if fam == "glm47":
        return ""
    if fam == "beam":
        # The recipient sits in the routed header, before the channel marker.
        return "<|start|>assistant to="
    rendered = r_tool(fam, "NAMEX", "KEYX", "VALX", 0)
    outer = control_tokens(fam)[2]
    # Search for the name AFTER the opener. A family whose opener already carries the
    # recipient name (muse routes on it) has an earlier `NAMEX` inside the opener
    # itself, and anchoring at zero returns an empty prefix instead of the invoke
    # header. Every other family's first `NAMEX` already follows the opener, so the
    # anchor changes nothing for them.
    return rendered[len(outer):rendered.index("NAMEX", len(outer))].lstrip()


def guided_invoke_prefix(fam):
    if fam == "deepseek_v41":
        return '<｜DSML｜ invoke name="'
    return invoke_header_prefix(fam)


def guided_named_invoke_prefix(fam):
    fam = unified_grammar_family(fam)
    if fam == "gpt_oss":
        return ""
    return {
        # Beam routes on the recipient in the message header, before the channel marker.
        "beam": "<|start|>assistant to=get_weather",
        "deepseek_v4": '<｜DSML｜invoke name="get_weather"',
        "deepseek_v41": '<｜DSML｜ invoke name="get_weather"',
        "gemma4": "call:get_weather",
        "glm47": "<tool_call>get_weather",
        "kimi_k2": "<|tool_call_begin|>functions.get_weather:0",
        "kimi_k3": '<|open|>call tool="get_weather"',
        "muse_glimmer": '<atem:invoke name="get_weather"',
        "muse_spark": '<atem:invoke name="get_weather"',
        "qwen3": "<function=get_weather",
    }[fam]


def r_bare_tool(fam, name, key, value, index):
    rendered = r_tool(fam, name, key, value, index)
    if fam == "gpt_oss":
        return rendered
    _, _, opener, closer = control_tokens(fam)
    rendered = rendered.removeprefix(opener.replace("NAMEX", name))
    if fam != "gemma4":
        rendered = rendered.removesuffix(closer)
    if fam in MUSE_FAMILIES:
        rendered = rendered.removesuffix("</atem:function_calls>")
    return rendered.strip()


def native_body_in_guided_reasoning(fam, value):
    return r_reason(fam, "before " + r_bare_tool(fam, "f", "x", value, 0) + " after") + GUIDED_ONE_CALL


def guided_surroundings(render, dynamo_note, fill=None):
    """A guided case whose SURROUNDINGS carry native grammar, so the input has to be
    per family — `every_family` is only right when the bytes are grammar-independent.

    `render(fam) -> input`. vLLM stays `GUIDED_UNSUPPORTED`: the request contract is
    `tool_output_mode=GuidedJson`, and a peer that never emits guided JSON is not an
    equivalent comparison just because the malformed surroundings happen to contain
    markup it could parse natively. Families with no native unified parser record the
    split-path divergence, same rule as `SPLIT`.
    """
    split = D(
        "UNSUPPORTED",
        "no native unified parser in this build, so the split path ignores `init` "
        "and cannot honour a guided request mode",
    )
    return {
        fam: (
            render(fam),
            GUIDED_UNSUPPORTED,
            {"verdict": "match", "note": dynamo_note} if fam in UNIFIED_FAMILIES else split,
            *( (fill(fam),) if fill else () ),
        )
        for fam in TEMPLATE_FAMILIES
    }


# Guided payloads, written once. `named` is what a NAMED choice emits (that
# tool's arguments alone); the arrays are what a REQUIRED choice emits.
GUIDED_NAMED_ARGS = '{"city": "Paris"}'
GUIDED_ONE_CALL = '[{"name": "get_weather", "arguments": {"city": "Paris"}}]'
GUIDED_TWO_CALLS = ('[{"name": "get_weather", "arguments": {"city": "Paris"}}, '
                     '{"name": "run", "arguments": {"cmd": "git log"}}]')
GUIDED_PARTIAL_CALLS = ('[{"name": "get_weather", "arguments": {"city": "Paris"}}, '
                        '{"arguments": {"city": "Tokyo"}}]')
GUIDED_UNSUPPORTED = D("UNSUPPORTED",
                       "vLLM base case doesn't emit guided JSON; conformance captures native XML only")
# vLLM's Muse Glimmer parsers exist only in unmerged PR #51655, so no released
# engine can be captured for this family and the cell has no measured value. The
# annotation records the published decode spec's intent and is UNVERIFIED until a
# release carries the parser.
V_MUSE = {
    "verdict": "match",
    "note": "vLLM muse_glimmer is unmerged (PR #51655); no released engine can be captured — unverified annotation",
}
V_MUSE_SPARK = {
    "verdict": "match",
    "note": "vLLM Muse Spark capture coverage is pending — unverified annotation",
}

# Families `capture_vllm_unified.py` has no entry for. The Unified tab falls back to
# the AUTHORED `expect.vllm` whenever a capture is missing, so for these families it
# falls back on EVERY case and draws the same plain `expected: MATCH` a captured
# family earns. Carrying the caveat only on the cases that happened to need a
# per-family verdict published the other 22 as if an engine had produced them.
VLLM_UNCAPTURABLE = {
    "muse_glimmer": V_MUSE,
    "gpt_oss": "No GPT-OSS vLLM Unified capture is configured; Dynamo Unified is captured separately.",
    "muse_spark": V_MUSE_SPARK,
    # The model builder's reference is the grammar's source of truth, not a peer engine,
    # and no released vLLM or SGLang parser reads this channel grammar.
    "beam": D("UNSUPPORTED", "no released vLLM or SGLang Beam parser; no peer capture exists, so no peer verdict is claimed"),
}


# --- CLEAN scenarios: same segments for every family, input is templated ------
# Each: (name, description, policy, segments, vllm, dynamo)
# vllm/dynamo are either a single entry (all families) or {family: entry}.

CLEAN = [
    ("tool_only",
     "Single tool call, no reasoning. Must stay green everywhere (the existing tool suite's world).",
     [], [("tool", "get_weather", "city", "Paris")], M, M),

    ("reason_then_tool",
     "Reasoning fully precedes one tool call (baseline).",
     [], [("reason", "Check weather."), ("tool", "get_weather", "city", "Paris")], M, M),

    ("reason_then_content",
     "Reasoning then visible content, no tool call (baseline). This is also covered in: e2e case-0001-chinese_arithmetic__non-stream-budget_capped.json (+ 42 more: every `reasoning/core`, `reasoning/complex` and `reasoning/history` case, `tool_none_arithmetic__*`, and the SECOND step of both `lifecycle_*` — each with its `-budget_unlimited` pair).",
     [], [("reason", "let me think"), ("text", "The answer is 42.")], M, M),

    ("interstitial_text",
     "Reasoning, then visible text, THEN a tool call. Text between reasoning-end and the call must survive as its own event, in order.",
     [], [("reason", "a"), ("text", "Here you go: "), ("tool", "get_weather", "city", "Paris")], M, M),

    ("reason_after_tool",
     "Reasoning AFTER a tool call, then final text (Example A). The split cannot represent reasoning between the call and the answer.",
     [], [("reason", "Look it up."), ("tool", "get_weather", "city", "Paris"),
          ("reason", "Now answer."), ("text", "It's 18C.")],
     M, D("MERGE", "v1 reasoning runs over the whole stream first -> both think spans merge into one event ahead of the tool_call")),

    ("content_then_reason",
     "Visible content, then reasoning, then more content. The split hoists reasoning to the front and merges the two content spans.",
     [], [("text", "Hello there. "), ("reason", "let me recall"), ("text", "The capital is Paris.")],
     M, D("ORDER", "reasoning hoisted ahead of leading content; the two text spans merge")),

    ("content_then_reason_then_tool",
     "Visible content BEFORE reasoning, then a tool call. The split hoists all reasoning to the front, so content-before-reasoning loses order.",
     [], [("text", "Sure, one sec. "), ("reason", "checking the forecast"),
          ("tool", "get_weather", "city", "Paris")],
     M, D("ORDER", "reasoning hoisted ahead of the leading content")),

    ("reason_interleaved",
     "reason -> tool -> reason -> tool. Two calls, each preceded by its own thought.",
     [], [("reason", "A"), ("tool", "f", "x", "1"), ("reason", "B"), ("tool", "g", "y", "2")],
     M, D("MERGE", "both think spans merge up front, ahead of both calls")),

    ("reason_tool_text_reason_tool",
     "reason -> tool -> text -> reason -> tool. Two reasoning spans separated by a call and text.",
     [], [("reason", "A"), ("tool", "f", "x", "1"), ("text", "working on it"),
          ("reason", "B"), ("tool", "g", "y", "2")],
     M, D("MERGE", "reasoning A and B merge up front; the second reasoning span loses its position")),

    ("trailing_text_after_tool",
     "Arbitrary visible prose AFTER the tool call (the point is it could be ANY content, so it must survive). Policy P1 (best-effort recovery) — trailing model text is preserved, not suppressed.",
     ["P1"], [("tool", "get_weather", "city", "Paris"),
              ("text", "The forecast shows clear skies for the rest of the week.")],
     {"gemma4": M, "qwen3": M, "muse_glimmer": V_MUSE, "gpt_oss": M,
    "muse_spark": V_MUSE_SPARK,
      "beam": VLLM_UNCAPTURABLE["beam"],
      "kimi_k3": M,
      "kimi_k2": D("LOSS", "kimi config stays in a tool state and SUPPRESSES trailing text -> arbitrary content dropped; violates best-effort recovery (preserve visible prose, conformance/README.md:142)")},
     {"gemma4": M, "qwen3": M,
      "beam": {"verdict": "match", "note": "the call ends at its own `<|call|>`; the following `final` message is ordinary content"},
      "muse_glimmer": {"verdict": "match", "note": "the tool channel closes at its own `<|eom|>`, so the following `to=user` message is ordinary content"},
      "kimi_k3": {"verdict": "match", "note": "K3 response framing preserves visible prose after the tools channel"},
      "kimi_k2": {"verdict": "match", "note": "P1 resolved by the v2 recovery contract: preserve trailing prose. Verify v2 kimi_k2 at capture time"},
      "gpt_oss": M}),

    # --- Group 2: multiple tool calls (TOOLCALLING.streamv1.2) — tool-only, green everywhere ---
    ("two_calls",
     "Two tool calls back-to-back, no reasoning. Both must surface as ordered events. This is also covered in: TOOLCALLING.streamv1.2.a.",
     [], [("tool", "f", "x", "1"), ("tool", "g", "y", "2")], M, M),
    ("two_calls_same_name",
     "The same tool called twice with different args. Both calls are distinct events. This is also covered in: TOOLCALLING.streamv1.2.d.",
     [], [("tool", "get_weather", "city", "Paris"), ("tool", "get_weather", "city", "Tokyo")], M, M),

    # --- Group 3: no tool call ---
    ("text_only",
     "Plain answer, no reasoning and no tool call. Pure content passthrough. This is also covered in: TOOLCALLING.streamv1.3. No e2e case has this shape: Qwen3.6 always emits a reasoning span, so the plain-content case is corpus-only.",
     [], [("text", "The answer is 42, no tools needed.")], M, M),

    # --- Group 7: argument fidelity (TOOLCALLING.streamv1.7) ---
    ("arg_unicode",
     "Unicode + spaces in a string argument value. Preserved exactly (I7). This is also covered in: TOOLCALLING.streamv1.7.b.",
     [], [("tool", "get_weather", "city", "São Paulo 東京")], M, M),

    # --- Group 8: content / narration position (TOOLCALLING.streamv1.8) ---
    ("text_before_tool",
     "Visible text before a single tool call, no reasoning. This is also covered in: TOOLCALLING.streamv1.8.a.",
     [], [("text", "On it: "), ("tool", "get_weather", "city", "Paris")], M, M),
    ("text_sandwich",
     "Visible text both before and after a tool call. This is also covered in: TOOLCALLING.streamv1.8.c.",
     [], [("text", "Before. "), ("tool", "get_weather", "city", "Paris"), ("text", " After.")], M, M),
    ("text_between_calls",
     "Visible text between two tool calls. This is also covered in: TOOLCALLING.streamv1.8.d.",
     [], [("tool", "f", "x", "1"), ("text", " then "), ("tool", "g", "y", "2")], M, M),
    ("narrated_calls",
     "Multiple tool calls with visible narration between each — tool_call -> text -> tool_call -> text -> tool_call. The agentic pattern: call, narrate, call again. Every call and every inter-call text span must surface as its own ordered event.",
     [], [("tool", "get_weather", "city", "Paris"), ("text", " then I'll run "),
          ("tool", "f", "x", "1"), ("text", " and "), ("tool", "g", "y", "2")], M, M),

    # --- Group 10: reasoning span (reasoning-only; REASONING.batch.2 / REASONING.batch.6) ---
    ("reason_only",
     "A reasoning span with no visible answer and no tool call. This is also covered in: REASONING.batch.2.a.",
     [], [("reason", "just thinking, no answer")], M, M),
    ("two_reason_spans",
     "Two reasoning spans separated by visible text, no tool call. Streaming keeps both spans in order; batch merges them. This is also covered in: REASONING.batch.6.a.",
     [], [("reason", "first thought"), ("text", "interlude "),
          ("reason", "second thought"), ("text", "done")],
     M, D("MERGE", "batch v1 reasoning merges both spans into one leading event")),

    # --- Group 11: reasoning <-> tool interleaving (UNIQUE to unified) ---
    ("reason_tool_reason_tool_reason",
     "reason -> tool -> reason -> tool -> reason. Three reasoning spans around two calls, including reasoning AFTER the last call — the split cannot place any of them.",
     [], [("reason", "A"), ("tool", "f", "x", "1"), ("reason", "B"),
          ("tool", "g", "y", "2"), ("reason", "C")],
     M, D("MERGE", "batch v1 reasoning merges A+B+C into one event ahead of both calls")),
    ("reason_between_calls",
     "Reasoning BETWEEN two tool calls with no surrounding text — the tightest interleave.",
     [], [("tool", "f", "x", "1"), ("reason", "mid"), ("tool", "g", "y", "2")],
     M, D("MERGE", "batch v1 hoists the mid-call reasoning ahead of both calls")),
    ("text_reason_tool_text_reason_tool",
     "Deep well-formed interleave — visible text, reasoning, and tool calls alternating (text -> reason -> tool -> text -> reason -> tool). Every segment must survive in emitted order; the point is that user text, reasoning, and calls all mix in one stream.",
     [], [("text", "Sure. "), ("reason", "check A"), ("tool", "f", "x", "1"),
          ("text", " and "), ("reason", "check B"), ("tool", "g", "y", "2")],
     M, D("MERGE", "batch v1 reasoning hoists both think spans ahead of everything; the interleaved text/call order collapses")),
]


# --- EDGE scenarios: grammar-specific raw input per family --------------------
# Each: (name, description, policy, golden, {family: (input, vllm, dynamo)})

_DS41_MIXED_STRING = ' <think>quoted</think> <｜DSML｜ calls> </｜DSML｜ calls> </｜DSML｜ invoke> &amp; "x"' + "\\" + "\n "

# Keep the original DS4.1 payload stable so its existing capture remains comparable.
_MIXED_CONTROL_STRINGS = {
    "deepseek_v41": _DS41_MIXED_STRING,
    **{
        family: " " + r_reason(family, "quoted") + " " + markers + ' &amp; "x"' + "\\" + "\n "
        for family, markers in {
            "deepseek_v4": "<｜DSML｜tool_calls> </｜DSML｜tool_calls> </｜DSML｜invoke>",
            "gemma4": "<|tool_call> <tool_call|>",
        "gpt_oss": "<|start|> <|channel|> <|constrain|> <|message|> <|end|> <|return|> <|call|>",
            "glm47": "<tool_call> </tool_call>",
            "qwen3": "<tool_call> </tool_call> </function>",
            "kimi_k2": "<|tool_calls_section_begin|> <|tool_calls_section_end|> <|tool_call_end|>",
            "kimi_k3": k3_open("tools") + " " + k3_close("tools") + " " + k3_close("call"),
            "muse_glimmer": "<atem:function_calls> </atem:function_calls> </atem:invoke>",
            # Every Beam channel terminator, plus a header opener; only the parameter
            # closer is structural inside a value, so all of them stay data.
            "beam": "<|call|> <|end|> <|return|> <|start|>",
        }.items()
    },
}

_MIXED_CONTROL_STRINGS["muse_spark"] = _MIXED_CONTROL_STRINGS["muse_glimmer"]

EDGE = [
    ("commentary_then_tool",
     "Muse Spark only: the reserved `commentary` recipient is visible intermediate text, then a native ATEM tool channel follows. It must not be interpreted as a tool named commentary or dropped as an unknown recipient.",
     [],
     [{"kind": "text", "text": "Checking the weather."},
      {"kind": "tool_call", "name": "get_weather", "arguments": {"city": "Paris"}}],
     {"starting_state": "None", "tool_output_mode": "Native", "named_tool": None},
     OnlyFamilies({
         "muse_spark": (
             "<|start|>assistant to=commentary<|message|>Checking the weather.<|eom|>"
             + r_tool("muse_spark", "get_weather", "city", "Paris", 0),
             V_MUSE_SPARK,
             {"verdict": "match", "note": "commentary remains visible and the following ATEM call is parsed"},
         ),
     })),

    # Phase-preserving output is opt-in and Spark-specific: other current
    # families do not reserve a distinct commentary recipient.
    ("muse_unterminated_reasoning_native",
     "Native Muse reasoning preserves an unterminated invocation header as literal prose. Guided JSON treats the same native prefix as stray control syntax (see the paired guided case).",
     ["I5"],
     [{"kind":"reasoning","text":'Maybe <atem:invoke name="x and more thought'}, {"kind":"text","text":"Done."}],
     {"starting_state":"None","tool_output_mode":"Native","named_tool":None},
     OnlyFamilies({family: ('<|start|>assistant to=self<|message|>Maybe <atem:invoke name="x and more thought<|eom|><|start|>assistant to=user<|message|>Done.<|eot|>', V_MUSE, M) for family in MUSE_FAMILIES})),
    ("muse_unterminated_reasoning_guided",
     "JSON-guided Muse reasoning strips an unterminated native invoke-start marker while preserving its following prose; this intentional guided/native distinction is chunk invariant.",
     ["I5"],
     [{"kind":"reasoning","text":"Maybe x and more thought"}, {"kind":"tool_call","name":"get_weather","arguments":{"city":"Paris"}}],
     {"starting_state":"None","tool_output_mode":"GuidedJson","named_tool":"get_weather"},
     OnlyFamilies({family: ('<|start|>assistant to=self<|message|>Maybe <atem:invoke name="x and more thought<|eom|>{"city":"Paris"}', V_MUSE, M) for family in MUSE_FAMILIES})),
    ("muse_opaque_undeclared_invoke",
     "Muse arguments own invocation-shaped text naming an undeclared tool. A complete outer string must not become an undeclared call during recovery.",
     ["I5", "I7"],
     [{"kind":"tool_call","name":"f","arguments":{"x":'doc: </atem:invoke><atem:invoke name="rm_rf"><atem:parameter name="path">/'}}],
     {"starting_state":"None","tool_output_mode":"Native","named_tool":None},
     OnlyFamilies({family: ('<|start|>assistant to=f<|message|><atem:invoke name="f"><atem:parameter name="x">doc: </atem:invoke><atem:invoke name="rm_rf"><atem:parameter name="path">/</atem:parameter></atem:invoke><|eot|>', V_MUSE, M) for family in MUSE_FAMILIES})),
    ("muse_opaque_declared_invoke",
     "Muse arguments own invocation-shaped text even when the embedded tool is declared. The complete outer argument remains data instead of dispatching the embedded call.",
     ["I5", "I7"],
     [{"kind":"tool_call","name":"f","arguments":{"x":'doc: </atem:invoke><atem:invoke name="get_weather"><atem:parameter name="city">Paris'}}],
     {"starting_state":"None","tool_output_mode":"Native","named_tool":None},
     OnlyFamilies({family: ('<|start|>assistant to=f<|message|><atem:invoke name="f"><atem:parameter name="x">doc: </atem:invoke><atem:invoke name="get_weather"><atem:parameter name="city">Paris</atem:parameter></atem:invoke><|eot|>', V_MUSE, M) for family in MUSE_FAMILIES})),
    ("muse_recovery_second_parameter",
     "A later Muse parameter owns its bytes through its closer. Declared invocation-shaped content inside it must not be promoted to an executable call.",
     ["I5", "I7"],
     [{"kind":"tool_call","name":"f","arguments":{"x":"done","y":'broken</atem:invoke><atem:invoke name="get_weather"><atem:parameter name="city">Paris'}}],
     {"starting_state":"None","tool_output_mode":"Native","named_tool":None},
     OnlyFamilies({family: ('<|start|>assistant to=f<|message|><atem:invoke name="f"><atem:parameter name="x">done</atem:parameter><atem:parameter name="y">broken</atem:invoke><atem:invoke name="get_weather"><atem:parameter name="city">Paris</atem:parameter></atem:invoke><|eot|>', V_MUSE, M) for family in MUSE_FAMILIES})),

    # ATEM parameters own nested invocation syntax. Without a closer, discard
    # the ambiguous invocation; a later closer completes the outer value rather
    # than proving a new executable call (PR #450 review).
    ("muse_unterminated_parameter_then_empty_call",
     "Muse: an unclosed parameter contains parameterless invocation-shaped text. At EOF discard the ambiguous outer invocation instead of guessing an executable nested call. This turn emits nothing.",
     ["P2"],
     [],
     {"starting_state": "None", "tool_output_mode": "Native", "named_tool": None},
     OnlyFamilies({"muse_glimmer": (
         '<|start|>assistant to=tools<|message|><atem:function_calls>'
         '<atem:invoke name="get_weather"><atem:parameter name="city">bad</atem:invoke>'
         '<atem:invoke name="f"></atem:invoke></atem:function_calls><|eot|>',
         V_MUSE, M,
     )})),

    ("muse_unterminated_parameter_then_call",
     "Muse: a parameter owns invocation-shaped text until its closer. Preserve the complete outer argument instead of treating the embedded declared tool as a new call.",
     ["P2", "I7"],
     [{"kind": "tool_call", "name": "f", "arguments": {"x": 'bad</atem:invoke><atem:invoke name="get_weather"><atem:parameter name="city">Paris'}}],
     {"starting_state": "None", "tool_output_mode": "Native", "named_tool": None},
     OnlyFamilies({"muse_glimmer": (
         '<|start|>assistant to=tools<|message|><atem:function_calls>'
         '<atem:invoke name="f"><atem:parameter name="x">bad</atem:invoke>'
         '<atem:invoke name="get_weather"><atem:parameter name="city">Paris</atem:parameter>'
         '</atem:invoke></atem:function_calls><|eot|>',
         V_MUSE, M,
     )})),
    ("gpt_oss_channel_header_variants",
     "A `redacted` channel stays withheld in reasoning, while a commentary tool recipient and final channel keep their event order.",
     ["I1", "I2", "I6"],
     [
         {"kind": "reasoning", "text": "check"},
         {"kind": "tool_call", "name": "lookup", "arguments": {"q": "weather"}},
         {"kind": "text", "text": "found"},
     ],
     {"starting_state": "None", "tool_output_mode": "Native", "named_tool": None},
     OnlyFamilies({
         "gpt_oss": (
             "<|start|>assistant<|channel|>analysis<|message|>check<|end|>"
             "<|start|>assistant to=functions.lookup<|channel|>commentary "
             "<|constrain|>json<|message|>{\"q\":\"weather\"}<|call|>"
             "<|start|>assistant<|channel|>final<|message|>found<|return|>",
             D("UNSUPPORTED", "No GPT-OSS vLLM Unified capture is configured."),
             M,
         ),
     })),

    ("gpt_oss_chat_template_tool_header",
     "GPT-OSS 20B and 120B template order: analysis reasoning, recipient-first commentary JSON tool call, and final text.",
     ["I1", "I2", "I6"],
     [
         {"kind": "reasoning", "text": "check"},
         {"kind": "tool_call", "name": "lookup", "arguments": {"q": "weather"}},
         {"kind": "text", "text": "found"},
     ],
     {"starting_state": "None", "tool_output_mode": "Native", "named_tool": None},
     OnlyFamilies({
         "gpt_oss": (
             "<|start|>assistant<|channel|>analysis<|message|>check<|end|>"
             "<|start|>assistant to=functions.lookup<|channel|>commentary json<|message|>"
             "{\"q\": \"weather\"}<|call|>"
             "<|start|>assistant<|channel|>final<|message|>found<|return|>",
             D("UNSUPPORTED", "No GPT-OSS vLLM Unified capture is configured."),
             M,
         ),
     })),

    ("gpt_oss_channel_first_tool_header",
     "A channel-first commentary header routes its recipient to a tool call with the template JSON format label.",
     ["I1", "I2", "I6"],
     [{"kind": "tool_call", "name": "lookup", "arguments": {"q": "weather"}}],
     {"starting_state": "None", "tool_output_mode": "Native", "named_tool": None},
     OnlyFamilies({
         "gpt_oss": (
             "<|start|>assistant<|channel|>commentary to=functions.lookup json"
             "<|message|>{\"q\": \"weather\"}<|call|>",
             D("UNSUPPORTED", "No GPT-OSS vLLM Unified capture is configured."),
             M,
         ),
     })),

    ("gpt_oss_multiple_unknown_and_empty_tools",
     "Multiple tool messages preserve order, empty arguments become an empty object, and the parser emits a tool name even when it is not in the offered schema.",
     ["I6", "I7"],
     [
         {"kind": "tool_call", "name": "run", "arguments": {}},
         {"kind": "tool_call", "name": "not_offered", "arguments": {"x": 2}},
     ],
     {"starting_state": "None", "tool_output_mode": "Native", "named_tool": None},
     OnlyFamilies({
         "gpt_oss": (
             "<|start|>assistant to=functions.run<|channel|>commentary "
             "<|message|><|call|>"
             "<|start|>assistant to=functions.not_offered<|channel|>commentary "
             "<|constrain|>json<|message|>{\"x\":2}<|call|>",
             D("UNSUPPORTED", "No GPT-OSS vLLM Unified capture is configured."),
             M,
         ),
     })),

    ("gpt_oss_control_marker_inside_argument_string",
     "Native control marker spellings inside a JSON string are payload and must not end, split, or resynchronize the tool call.",
     ["I5", "I7"],
     [{"kind": "tool_call", "name": "lookup", "arguments": {
         "markers": "<|start|><|channel|><|constrain|><|message|><|end|><|return|><|call|>"
     }}],
     {"starting_state": "None", "tool_output_mode": "Native", "named_tool": None},
     OnlyFamilies({
         "gpt_oss": (
             "<|start|>assistant to=functions.lookup<|channel|>commentary "
             "<|message|>{\"markers\":\"<|start|><|channel|><|constrain|><|message|>"
             "<|end|><|return|><|call|>\"}<|call|>",
             D("UNSUPPORTED", "No GPT-OSS vLLM Unified capture is configured."),
             M,
         ),
     })),

    ("gpt_oss_truncated_tool_resynchronizes_at_next_message",
     "A tool message omits its outer terminator before a visible-response header. Preserve the later content and each grammar's complete-call policy.",
     ["I3", "I4", "I5"],
     [{"kind": "text", "text": "recovered"}],
     {"starting_state": "None", "tool_output_mode": "Native", "named_tool": None},
     OnlyFamilies({
         "gpt_oss": (
             "<|start|>assistant to=functions.lookup<|channel|>commentary "
             "<|message|>{\"q\":\"unfinished\"}"
             "<|start|>assistant<|channel|>final<|message|>recovered<|return|>",
             D("UNSUPPORTED", "No GPT-OSS vLLM Unified capture is configured."),
             M,
         ),
     })),

    ("glm47_parameterless_call_shape_inside_argument",
     "GLM 5 only: an offered parameterless-call shape appears inside an open argument value. The embedded close/open markers remain argument data and must not dispatch a second call.",
     ["I7"],
     [{"kind": "tool_call", "name": "run", "arguments": {
         "cmd": "before </tool_call><tool_call>get_weather</tool_call> after",
     }}],
     {"starting_state": "None", "tool_output_mode": "Native", "named_tool": None},
     OnlyFamilies({
         "glm47": (
             "<tool_call>run<arg_key>cmd</arg_key><arg_value>before </tool_call><tool_call>get_weather</tool_call> after</arg_value></tool_call>",
             D("UNSUPPORTED", "no released vLLM UnifiedParser capture for GLM 5"),
             M,
         ),
     })),

    ("truncated_tool_eof",
     "Stream ends mid tool call (no close marker). Policy P2 — drop the incomplete call, keep valid preceding output, no error, no leaked markup.",
     ["P2"],
     [{"kind": "reasoning", "text": "ok"}],
     {"starting_state": "None", "tool_output_mode": "Native", "named_tool": None},
     {
        "deepseek_v41": ('<think>ok</think><｜DSML｜ calls><｜DSML｜ invoke name="get_weather"><｜DSML｜ parameter name="city" string="true">Par', M, M),
        "deepseek_v4": ('<think>ok</think><｜DSML｜tool_calls><｜DSML｜invoke name="get_weather"><｜DSML｜parameter name="city" string="true">Par', M, M),
        "gemma4": ("<|channel>thought\nok<channel|><|tool_call>call:get_weather{city:<|\"|>Par",
                   D("ERROR", "native Gemma4UnifiedParser finish() returns a hard Err -> erroring is the opposite of best-effort recovery"),
                   {"verdict": "match", "note": "P2: drop the partial trailing call, keep the preceding reasoning, never error/leak (TOOLCALLING.batch.5.e)"}),
        "qwen3": ("<think>ok</think><tool_call>\n<function=get_weather>\n<parameter=city>\nPar",
                  {"verdict": "match", "note": "P2: drop the unterminated call and keep the preceding reasoning"},
                  {"verdict": "match", "note": "P2: v2 drops the partial trailing call, keeps reasoning"}),
        "kimi_k2": ("<think>ok</think><|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{\"city\": \"Par",
                    {"verdict": "match", "note": "P2: drop the unterminated call and keep the preceding reasoning"},
                    {"verdict": "match", "note": "P2: v2 drops the partial trailing call, keeps reasoning"}),
        "kimi_k3": (r_reason("kimi_k3", "ok") + k3_tools(
                        k3_call("get_weather", 1, k3_open(
                            "argument", [("key", "city"), ("type", "string")]
                        ) + "Par", close=False),
                        close=False),
                    M,
                    {"verdict": "match", "note": "P2: drop the partial XTML argument and keep preceding reasoning"}),
        "muse_glimmer": ("<|start|>assistant to=self<|message|>ok<|eom|><|start|>assistant to=get_weather<|message|><atem:function_calls>\n<atem:invoke name=\"get_weather\">\n<atem:parameter name=\"city\">Par",
                         V_MUSE,
                         {"verdict": "match", "note": "P2: the invoke never reached its `</atem:invoke>` fence, so the call is dropped and its markup never leaks; the reasoning channel is kept"}),
     }),

    ("reason_unterminated",
     "Stream ends while still inside reasoning (no close marker). Open reasoning is promoted at finish, not dropped and not leaked as text.",
     [],
     [{"kind": "reasoning", "text": "thinking but stream ends"}],
     {"starting_state": "None", "tool_output_mode": "Native", "named_tool": None},
     {
        "gemma4": ("<|channel>thought\nthinking but stream ends",
                   M, {"verdict": "match", "note": "verify against v1 gemma4 reasoning finish() at capture time"}),
        "qwen3": ("<think>thinking but stream ends",
                  M, {"verdict": "match", "note": "verify against v1 qwen3 reasoning finish() at capture time"}),
        "deepseek_v41": ("<think>thinking but stream ends", M, M),
        "kimi_k2": ("<think>thinking but stream ends",
                    M, {"verdict": "match", "note": "verify against v1 kimi reasoning finish() at capture time"}),
        "kimi_k3": (k3_open("think") + "thinking but stream ends",
                    M,
                    {"verdict": "match", "note": "open K3 think channel promoted at finish"}),
        "muse_glimmer": ("<|start|>assistant to=self<|message|>thinking but stream ends",
                         V_MUSE,
                         {"verdict": "match", "note": "the open `to=self` body is promoted as reasoning at finish, not dropped and not leaked as text"}),
     }),

    ("arg_marker_in_string",
     "A close-marker-looking sequence INSIDE a string arg value. Invariant I7 — the value is data, preserved exactly, not truncated at the marker-looking substring.",
     [],
     [{"kind": "tool_call", "name": "run", "arguments": {"cmd": None}}],  # cmd filled per family below
     {"starting_state": "None", "tool_output_mode": "Native", "named_tool": None},
     {
        "deepseek_v41": (r_tool("deepseek_v41", "run", "cmd", "git log </｜DSML｜ invoke> --oneline", 0),
                         M, M, "git log </｜DSML｜ invoke> --oneline"),
        "gemma4": ("<|tool_call>call:run{cmd:<|\"|>git log }<tool_call|> --oneline<|\"|>}<tool_call|>",
                   D("ARG_MISMATCH", "char-by-char streamed-arg coercion truncates args at the marker-looking boundary (regression class #48702/#47977)"),
                   {"verdict": "match", "note": "emit-on-close typing sees the whole balanced value; find_tool_call_end_position_gemma4 ignores <tool_call|> inside <|\"|> strings"},
                   "git log }<tool_call|> --oneline"),
        "qwen3": ("<tool_call>\n<function=run>\n<parameter=cmd>\ngit log </tool_call> --oneline\n</parameter>\n</function>\n</tool_call>",
                  {"verdict": "match", "note": "the parameter boundary owns the value; embedded `</tool_call>` is data"},
                  {"verdict": "match", "note": "v2 reads the parameter value up to `</parameter>`; embedded `</tool_call>` preserved"},
                  "git log </tool_call> --oneline"),
        "kimi_k2": ("<|tool_calls_section_begin|><|tool_call_begin|>functions.run:0<|tool_call_argument_begin|>{\"cmd\": \"git log <|tool_call_end|> --oneline\"}<|tool_call_end|><|tool_calls_section_end|>",
                     {"verdict": "match", "note": "the JSON string owns embedded `<|tool_call_end|>` bytes as data"},
                     {"verdict": "match", "note": "v2 parses the JSON arg blob; the marker inside the string is data"},
                     "git log <|tool_call_end|> --oneline"),
        "deepseek_v4": ("<｜DSML｜tool_calls><｜DSML｜invoke name=\"run\"><｜DSML｜parameter name=\"cmd\" string=\"true\">git log </｜DSML｜invoke> --oneline</｜DSML｜parameter></｜DSML｜invoke></｜DSML｜tool_calls>",
                        {"verdict": "match", "note": "the parameter boundary owns the embedded invoke close as data"},
                        {"verdict": "match", "note": "v2 reads the parameter value up to its own DSML close"},
                        "git log </｜DSML｜invoke> --oneline"),
        "kimi_k3": (k3_tools(k3_call(
                        "run", 1,
                        k3_argument("cmd", "string", "git log <|close|>call<|sep|> --oneline"))),
                    M,
                    {"verdict": "match", "note": "typed argument owns the embedded K3 call close as data"},
                    "git log <|close|>call<|sep|> --oneline"),
        "muse_glimmer": ("<|start|>assistant to=run<|message|><atem:function_calls>\n<atem:invoke name=\"run\">\n<atem:parameter name=\"cmd\">git log </atem:function_calls> --oneline</atem:parameter>\n</atem:invoke>\n</atem:function_calls><|eom|>",
                         V_MUSE,
                         {"verdict": "match", "note": "the parameter value runs to its own `</atem:parameter>`, so the enclosing `</atem:function_calls>` inside it is data"},
                         "git log </atem:function_calls> --oneline"),
     }),

    ("orphan_close_after_prose",
     "Prose followed by an orphan close marker with no matching open. Best-effort recovery — the prose stays as content, the orphan marker is stripped, nothing leaks.",
     [],
     [{"kind": "text", "text": "I will check that. "}],
     {"starting_state": "None", "tool_output_mode": "Native", "named_tool": None},
     {
        "gemma4": ("I will check that. <tool_call|>",
                   D("LEAK", "vLLM/SGLang leak the orphan close marker into content as the whole tail (TOOLCALLING_CASES.md 5.g)"),
                   D("LEAK", "LIVE finding: v2 gemma4 leaks a lone <tool_call|> end marker into content — with no matching <|tool_call> open the scanner treats it as text. Best-effort-recovery gap (should strip per TOOLCALLING 5.g).")),
        "qwen3": ("I will check that. </tool_call>",
                  {"verdict": "match", "note": "the orphan close is stripped and the preceding prose remains visible"},
                  {"verdict": "match", "note": "the orphan close is stripped and the preceding prose remains visible"}),
        "kimi_k2": ("I will check that. <|tool_call_end|>",
                    D("LEAK", "the orphan `<|tool_call_end|>` remains in the assembled reasoning output"),
                    D("LEAK", "the split path retains the orphan `<|tool_call_end|>` in assembled reasoning")),
        "kimi_k3": ("I will check that. " + k3_close("call"),
                    M,
                    {"verdict": "match", "note": "orphan K3 call closer stripped after prose"}),
        # `<|eot|>` already ended the turn, so the trailing `<|eom|>` closes nothing.
         "muse_glimmer": ("<|start|>assistant to=user<|message|>I will check that. <|eot|><|eom|>",
                          V_MUSE,
                          {"verdict": "match", "note": "an orphan terminator outside any routed message is stripped, never emitted as content"}),
         "deepseek_v41": ("I will check that. </｜DSML｜ calls>", M, M),
      }),

    ("empty_args",
     "A tool call with an empty argument object {}. Policy P3 — empty args serialize to {}. This is also covered in: TOOLCALLING.streamv1.6.a.",
     ["P3"],
     [{"kind": "tool_call", "name": "get_weather", "arguments": {}}],
     {"starting_state": "None", "tool_output_mode": "Native", "named_tool": None},
     {
        "deepseek_v41": ('<｜DSML｜ calls><｜DSML｜ invoke name="get_weather"></｜DSML｜ invoke></｜DSML｜ calls>', M, M),
        "gemma4": ("<|tool_call>call:get_weather{}<tool_call|>", M, M),
        "qwen3": ("<tool_call>\n<function=get_weather>\n</function>\n</tool_call>", M, M),
        "kimi_k2": ("<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{}<|tool_call_end|><|tool_calls_section_end|>", M, M),
        "kimi_k3": (k3_tools(k3_call("get_weather", 1, "")),
                    M, M),
        "muse_glimmer": ("<|start|>assistant to=get_weather<|message|><atem:function_calls>\n<atem:invoke name=\"get_weather\">\n</atem:invoke>\n</atem:function_calls><|eom|>",
                         V_MUSE, M),
     }),

    # Beam: the serving grammar makes the whole parameter body optional for a tool whose
    # schema requires nothing, and lets EOS stand for the call closer, so a tool header the
    # stream ends after is the complete call `{}`. Scope is Beam-specific: the grammar
    # itself admits this byte sequence (review of #438); other grammars are not asserted.
    ("empty_args_no_close",
     "Beam: a header for a strict tool with no properties, then EOS — no parameter tags, no newline, no `<|call|>`. The grammar admits it (optional body, optional closer), so the parser emits `noop({})`.",
     ["P3"],
     [{"kind": "tool_call", "name": "noop", "arguments": {}}],
     {"starting_state": "None", "tool_output_mode": "Native", "named_tool": None},
     {"finish_reason": "stop"},
     OnlyFamilies({"beam": ("<|start|>assistant to=noop<|channel|>analysis <|constrain|>xml<|message|>", VLLM_UNCAPTURABLE["beam"], M)}),
     [{"name": "noop", "parameters": {"type": "object", "properties": {}, "additionalProperties": False}, "strict": True}]),

    ("optional_args_no_close",
     "Beam: a header for a tool whose only property is optional, then EOS — no parameter tags, no `<|call|>`. Nothing is required, so the grammar admits the bare header and the parser emits `noop({})`; a required argument would keep this a truncation.",
     ["P3"],
     [{"kind": "tool_call", "name": "noop", "arguments": {}}],
     {"starting_state": "None", "tool_output_mode": "Native", "named_tool": None},
     {"finish_reason": "stop"},
     OnlyFamilies({"beam": ("<|start|>assistant to=noop<|channel|>analysis <|constrain|>xml<|message|>", VLLM_UNCAPTURABLE["beam"], M)}),
     [{"name": "noop", "parameters": {"type": "object", "properties": {"a": {"type": "string"}}, "additionalProperties": False}, "strict": True}]),

    ("tool_no_close",
     "A single tool call whose body is complete but the close marker never arrives before EOF. Most grammars recover the complete call at finish; DSML requires the invoke close, so its malformed turn emits nothing. This is also covered in: TOOLCALLING.streamv1.5.a.",
     [],
     [{"kind": "tool_call", "name": "get_weather", "arguments": {"city": "Paris"}}],
     {"starting_state": "None", "tool_output_mode": "Native", "named_tool": None},
     {
        "deepseek_v41": ('<｜DSML｜ calls><｜DSML｜ invoke name="get_weather"><｜DSML｜ parameter name="city" string="true">Paris</｜DSML｜ parameter>', M, M, []),
        "deepseek_v4": ('<｜DSML｜tool_calls><｜DSML｜invoke name="get_weather"><｜DSML｜parameter name="city" string="true">Paris</｜DSML｜parameter>',
                        {"verdict": "match", "note": "the missing invoke close makes this DSML call malformed, so it is dropped"},
                        {"verdict": "match", "note": "the missing invoke close makes this DSML call malformed, so it is dropped"},
                        []),
        "gemma4": ("<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>}",
                   {"verdict": "match", "note": "body complete; recover the call at finish"},
                   {"verdict": "match", "note": "body complete; recover the call at finish"}),
        "qwen3": ("<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>",
                  {"verdict": "match", "note": "body complete; recover at finish"},
                  D("DROP", "the complete call body produces no events when the outer close is absent")),
        "kimi_k2": ("<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{\"city\": \"Paris\"}",
                    {"verdict": "match", "note": "body complete; recover at finish"},
                    D("DROP", "the complete call body produces no events when the outer close is absent")),
        "kimi_k3": (k3_tools(
                        k3_call("get_weather", 1, k3_argument("city", "string", "Paris"), close=False),
                        close=False),
                    M,
                    {"verdict": "match", "note": "complete typed arguments recover at EOF without a call close"}),
        "muse_glimmer": ("<|start|>assistant to=get_weather<|message|><atem:function_calls>\n<atem:invoke name=\"get_weather\">\n<atem:parameter name=\"city\">Paris</atem:parameter>\n</atem:invoke>\n</atem:function_calls>",
                         V_MUSE,
                         {"verdict": "match", "note": "the invoke closed its own `</atem:invoke>` fence, so the call is complete even though the message never emitted `<|eom|>`"}),
     }),

    # --- Group 12: adversarial nesting (a marker of one channel inside another) ---
    ("reason_markup_in_arg",
     "'Tool call contains reasoning' — a reasoning-channel marker sits inside a QUOTED tool-arg value. This is NOT a leak: a leak is control markup surfacing in visible content or reasoning, but here the markup is a tool ARGUMENT VALUE (data bound for the function, inside the grammar's string delimiters), so by I7 the parser preserves it byte-exact. The gemma4 native UnifiedParser confirms this golden exactly. Failure mode: a reasoning-first pipeline extracts the `<think>`/`<|channel>` from inside the arg BEFORE tool parsing, hoisting it into a spurious reasoning event and corrupting the arg to empty.",
     [],
     [{"kind": "tool_call", "name": "log", "arguments": {"note": None}}],  # note filled per family
     {"starting_state": "None", "tool_output_mode": "Native", "named_tool": None},
     {
        "deepseek_v41": (r_tool("deepseek_v41", "log", "note", "<think>reconsider</think>", 0), M, M, "<think>reconsider</think>"),
        "gemma4": ("<|tool_call>call:log{note:<|\"|><|channel>thought\nreconsider<channel|><|\"|>}<tool_call|>",
                   D("ARG_MISMATCH", "the reasoning extractor lifts the `<|channel>...<channel|>` out of the arg value before tool parsing, so the logged note no longer matches golden"),
                   D("MERGE", "v1 reasoning runs first over the whole stream and pulls the arg's embedded `<|channel>...<channel|>` into a leading reasoning event, corrupting the tool arg"),
                   "<|channel>thought\nreconsider<channel|>"),
        "qwen3": ("<tool_call>\n<function=log>\n<parameter=note>\n<think>reconsider</think>\n</parameter>\n</function>\n</tool_call>",
                  D("ARG_MISMATCH", "captured: tool_call(log) — the `<think>...</think>` inside the parameter value is extracted as reasoning first, corrupting the arg"),
                  D("MERGE", "captured: tool_call(log) — v1 reasoning lifts the embedded `<think>` out of the arg"),
                  "<think>reconsider</think>"),
        "kimi_k2": ("<|tool_calls_section_begin|><|tool_call_begin|>functions.log:0<|tool_call_argument_begin|>{\"note\": \"<think>reconsider</think>\"}<|tool_call_end|><|tool_calls_section_end|>",
                    D("ARG_MISMATCH", "captured: reasoning(reconsider) | text(Logging now: ) | tool_call(log) | text( done.) — the `<think>` inside the JSON string arg is extracted as reasoning first, corrupting the arg"),
                    D("MERGE", "captured: reasoning(reconsider) | tool_call(log) — v1 reasoning lifts the embedded `<think>` out of the JSON arg"),
                    "<think>reconsider</think>"),
        "kimi_k3": (k3_tools(k3_call(
                        "log", 1, k3_argument("note", "string", k3_channel("think", "reconsider")))),
                    M,
                    {"verdict": "match", "note": "K3 think markers inside a typed string remain argument data"},
                    k3_channel("think", "reconsider")),
        # Muse's reasoning opener is a header, not a marker pair, so the quoted
        # reasoning markup inside the value is a bare `to=self<|message|>` run.
        "muse_glimmer": ("<|start|>assistant to=log<|message|><atem:function_calls>\n<atem:invoke name=\"log\">\n<atem:parameter name=\"note\">to=self<|message|>reconsider</atem:parameter>\n</atem:invoke>\n</atem:function_calls><|eom|>",
                         V_MUSE,
                         {"verdict": "match", "note": "the header is resolved once, at the message boundary; inside an open tool body a quoted `to=self<|message|>` is argument data"},
                         "to=self<|message|>reconsider"),
     }),

    ("tool_in_reason",
     "'Reasoning contains tool call' — a well-formed tool-call envelope nested INSIDE a reasoning span. This is the OPPOSITE of reason_markup_in_arg: a reasoning span is opaque TEXT, not a quoted data region, so a real tool-call marker inside it IS structural. Best-effort recovery breaks out of reasoning, emits the call, and resumes reasoning after its close (golden: reason -> call -> reason). Leaking the raw `<|tool_call>...<tool_call|>` into reasoning_content, or dropping the call, is the regression — which is what every reasoning-first engine does here.",
     [],
     [{"kind": "reasoning", "text": "I should check. "},
      {"kind": "tool_call", "name": "get_weather", "arguments": {"city": "Paris"}},
      {"kind": "reasoning", "text": " now answer"}],
     {"starting_state": "None", "tool_output_mode": "Native", "named_tool": None},
     {
        "deepseek_v41": ("<think>I should check. " + r_tool("deepseek_v41", "get_weather", "city", "Paris", 0) + " now answer</think>", M, M),
        "gemma4": ("<|channel>thought\nI should check. <|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>}<tool_call|> now answer<channel|>",
                   D("LEAK", "the reasoning extractor consumes to `<channel|>`, so the nested `<|tool_call>...<tool_call|>` leaks into reasoning_content and the call is dropped; break-out recovery not implemented"),
                   D("LEAK", "v1 reasoning runs to `<channel|>`, swallowing the nested tool markup into one reasoning event; the call is lost")),
        "qwen3": ("<think>I should check. <tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call> now answer</think>",
                  D("LEAK", "captured: reasoning(I should check. ) | tool_call(get_weather) | reasoning( now answer) — the `</think>` closes only after the nested call, so the tool markup leaks into reasoning and the call is dropped"),
                  D("LEAK", "captured: text(Sure. ) | reasoning(I should check. ) | tool_call(get_weather) | reasoning( now answer) | text( Here you go.) — v1 reasoning consumes to `</think>`, leaking the nested tool markup")),
        "kimi_k2": ("<think>I should check. <|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{\"city\": \"Paris\"}<|tool_call_end|><|tool_calls_section_end|> now answer</think>",
                    D("LEAK", "captured: reasoning(I should check. ) | tool_call(get_weather) | text( now answer</think>) — the tool section nested in `<think>...</think>` leaks into reasoning and the call is dropped"),
                    D("LEAK", "captured: reasoning(I should check. ) | text(Sure. ) | tool_call(get_weather) | text( now answer</think> Here you g) — v1 reasoning consumes to `</think>`, leaking the nested section")),
        "kimi_k3": (k3_open("think") + "I should check. "
                    + r_tool("kimi_k3", "get_weather", "city", "Paris", 0)
                    + " now answer" + k3_close("think"),
                    M,
                    {"verdict": "match", "note": "K3 tools inside a thought break out and the thought resumes afterward"}),
        # Muse's channels never nest: the model abandons the analysis channel by
        # writing the tool header directly, without `<|eom|>`. Recovering that
        # boundary is what puts the call between the two thoughts.
        "muse_glimmer": ("<|start|>assistant to=self<|message|>I should check. to=get_weather<|message|><atem:function_calls>\n<atem:invoke name=\"get_weather\">\n<atem:parameter name=\"city\">Paris</atem:parameter>\n</atem:invoke>\n</atem:function_calls><|eom|><|start|>assistant to=self<|message|> now answer<|eom|>",
                         V_MUSE,
                         {"verdict": "match", "note": "the reasoning body ends at the bare tool header (missing-`<|eom|>` recovery), so the call surfaces between the two thoughts instead of being swallowed"}),
     }),

    ("reason_markup_in_arg_with_text",
     "reason_markup_in_arg (tool arg value contains reasoning markup, I7 data) WITH visible narration before and after the call. All three channels at once: leading text -> tool call whose arg holds reasoning markup -> trailing text. Golden keeps the visible text as text, the call as a call, and the markup byte-exact in the arg. A reasoning-first pipeline both corrupts the arg (extracting the embedded reasoning) and can reorder/misroute the surrounding text.",
     [],
     [{"kind": "text", "text": "Logging now: "},
      {"kind": "tool_call", "name": "log", "arguments": {"note": None}},  # filled per family
      {"kind": "text", "text": " done."}],
     {"starting_state": "None", "tool_output_mode": "Native", "named_tool": None},
     {
        "deepseek_v41": ("Logging now: " + r_tool("deepseek_v41", "log", "note", "<think>reconsider</think>", 0) + " done.", M, M, "<think>reconsider</think>"),
        "gemma4": ("Logging now: <|tool_call>call:log{note:<|\"|><|channel>thought\nreconsider<channel|><|\"|>}<tool_call|> done.",
                   D("ARG_MISMATCH", "the reasoning extractor lifts the `<|channel>...<channel|>` out of the arg before tool parsing; the note no longer matches and the surrounding text can shift"),
                   D("MERGE", "v1 reasoning hoists the arg's embedded `<|channel>...<channel|>` ahead of the visible text and corrupts the tool arg"),
                   "<|channel>thought\nreconsider<channel|>"),
        "qwen3": ("Logging now: <tool_call>\n<function=log>\n<parameter=note>\n<think>reconsider</think>\n</parameter>\n</function>\n</tool_call> done.",
                  D("ARG_MISMATCH", "captured: text(Logging now: ) | tool_call(log) | text( done.) — the `<think>` inside the parameter value is extracted as reasoning first, corrupting the arg"),
                  D("MERGE", "captured: text(Logging now: ) | tool_call(log) | text( done.) — v1 reasoning lifts the embedded `<think>` out of the arg and ahead of the text"),
                  "<think>reconsider</think>"),
        "kimi_k2": ("Logging now: <|tool_calls_section_begin|><|tool_call_begin|>functions.log:0<|tool_call_argument_begin|>{\"note\": \"<think>reconsider</think>\"}<|tool_call_end|><|tool_calls_section_end|> done.",
                    D("ARG_MISMATCH", "captured: reasoning(reconsider) | text(Logging now: ) | tool_call(log) | text( done.) — the `<think>` inside the JSON string arg is extracted as reasoning first, corrupting the arg"),
                    D("MERGE", "captured: reasoning(reconsider) | text(Logging now: ) | tool_call(log) | text( done.) — v1 reasoning lifts the embedded `<think>` out of the JSON arg and ahead of the text"),
                    "<think>reconsider</think>"),
        "kimi_k3": (r_text("kimi_k3", "Logging now: ")
                    + k3_tools(k3_call(
                        "log", 1, k3_argument("note", "string", k3_channel("think", "reconsider"))))
                    + r_text("kimi_k3", " done."),
                    M,
                    {"verdict": "match", "note": "response channels surround a call whose typed string owns embedded think markers"},
                    k3_channel("think", "reconsider")),
        "muse_glimmer": ("<|start|>assistant to=user<|message|>Logging now: <|eom|><|start|>assistant to=log<|message|><atem:function_calls>\n<atem:invoke name=\"log\">\n<atem:parameter name=\"note\">to=self<|message|>reconsider</atem:parameter>\n</atem:invoke>\n</atem:function_calls><|eom|><|start|>assistant to=user<|message|> done.<|eom|>",
                         V_MUSE,
                         {"verdict": "match", "note": "both `to=user` messages keep their position and the quoted header stays argument data"},
                         "to=self<|message|>reconsider"),
     }),

    ("tool_in_reason_with_text",
     "tool_in_reason (a tool call nested inside a reasoning span, break-out recovery) WITH visible narration before and after the reasoning span. All three channels at once: leading text -> reasoning that wraps a real call -> trailing text. Golden: text -> reason -> call -> reason -> text. Engines that treat reasoning as opaque-until-close leak the nested tool markup into reasoning_content and drop the call.",
     [],
     [{"kind": "text", "text": "Sure. "},
      {"kind": "reasoning", "text": "I should check. "},
      {"kind": "tool_call", "name": "get_weather", "arguments": {"city": "Paris"}},
      {"kind": "reasoning", "text": " now answer"},
      {"kind": "text", "text": " Here you go."}],
     {"starting_state": "None", "tool_output_mode": "Native", "named_tool": None},
     {
        "deepseek_v41": ("Sure. <think>I should check. " + r_tool("deepseek_v41", "get_weather", "city", "Paris", 0) + " now answer</think> Here you go.", M, M),
        "gemma4": ("Sure. <|channel>thought\nI should check. <|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>}<tool_call|> now answer<channel|> Here you go.",
                   D("LEAK", "the reasoning extractor consumes to `<channel|>`, leaking the nested tool markup into reasoning_content and dropping the call; the visible text survives on both sides"),
                   D("LEAK", "v1 reasoning runs to `<channel|>`, swallowing the nested tool markup; the call is lost")),
        "qwen3": ("Sure. <think>I should check. <tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call> now answer</think> Here you go.",
                  D("LEAK", "captured: text(Sure. ) | reasoning(I should check. ) | tool_call(get_weather) | reasoning( now answer) | text( Here you go.) — `</think>` closes only after the nested call, so the tool markup leaks into reasoning and the call is dropped"),
                  D("LEAK", "captured: text(Sure. ) | reasoning(I should check. ) | tool_call(get_weather) | reasoning( now answer) | text( Here you go.) — v1 reasoning consumes to `</think>`, leaking the nested tool markup")),
        "kimi_k2": ("Sure. <think>I should check. <|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{\"city\": \"Paris\"}<|tool_call_end|><|tool_calls_section_end|> now answer</think> Here you go.",
                    D("LEAK", "captured: reasoning(I should check. ) | text(Sure. ) | tool_call(get_weather) | text( now answer</think> Here you g) — the nested tool section leaks into reasoning and the call is dropped"),
                    D("LEAK", "captured: reasoning(I should check. ) | text(Sure. ) | tool_call(get_weather) | text( now answer</think> Here you g) — v1 reasoning consumes to `</think>`, leaking the nested section")),
        "kimi_k3": (r_text("kimi_k3", "Sure. ") + k3_open("think")
                    + "I should check. " + r_tool("kimi_k3", "get_weather", "city", "Paris", 0)
                    + " now answer" + k3_close("think") + r_text("kimi_k3", " Here you go."),
                    M,
                    {"verdict": "match", "note": "framed K3 responses remain visible around the nested thought and call"}),
        "muse_glimmer": ("<|start|>assistant to=user<|message|>Sure. <|eom|><|start|>assistant to=self<|message|>I should check. to=get_weather<|message|><atem:function_calls>\n<atem:invoke name=\"get_weather\">\n<atem:parameter name=\"city\">Paris</atem:parameter>\n</atem:invoke>\n</atem:function_calls><|eom|><|start|>assistant to=self<|message|> now answer<|eom|><|start|>assistant to=user<|message|> Here you go.<|eom|>",
                         V_MUSE,
                         {"verdict": "match", "note": "the bare-header recovery is latched to a reasoning body, so it fires here and stays off inside the surrounding `to=user` messages"}),
     }),

    ("two_adjacent_reason_spans",
     "Two reasoning spans with nothing between them, then the answer. The single `reasoning_text` field every batch parser exposes can only concatenate them, so the separator is part of the contract: adjacent spans join with a newline. The counterpart is already covered by `reason_after_tool` / `reason_interleaved`, where two spans separated by a call must NOT join — a parser that always joins invents a newline the model never emitted, and one that never joins loses the batch parity every engine has.",
     [],
     [{"kind": "reasoning", "text": "first\nsecond"}, {"kind": "text", "text": "done"}],
     {"starting_state": "None", "tool_output_mode": "Native", "named_tool": None},
     {
        "deepseek_v41": ("<think>first</think><think>\nsecond</think>done", M, M),
        "gemma4": ("<|channel>thought\nfirst<channel|><|channel>thought\n\nsecond<channel|>done", M,
                   {"verdict": "match", "note": "the split path merges both spans into one reasoning event, which is what this scenario expects"}),
        "qwen3": ("<think>first</think><think>\nsecond</think>done", M,
                  {"verdict": "match", "note": "adjacent reasoning runs coalesce into one event (I8)"}),
        "kimi_k2": ("<think>first</think><think>\nsecond</think>done", M,
                    {"verdict": "match", "note": "adjacent reasoning runs coalesce into one event (I8)"}),
        "kimi_k3": (k3_channel("think", "first") + k3_channel("think", "\nsecond")
                    + r_text("kimi_k3", "done"),
                    M,
                    {"verdict": "match", "note": "adjacent K3 think channels coalesce with the authored separator"}),
        "muse_glimmer": ("<|start|>assistant to=self<|message|>first<|eom|><|start|>assistant to=self<|message|>second<|eom|><|start|>assistant to=user<|message|>done<|eom|>",
                         V_MUSE,
                         {"verdict": "match", "note": "the newline is emitted between two ADJACENT `to=self` messages only, matching v1 and both engines' batch parsers"}),
     }),

    # --- Group 13: request-scoped modes (guided decoding, prefilled channels) ---
    ("guided_json_named_tool",
     "Guided decoding with a named tool (tool_choice=specific_tool). The model emits bare JSON object, not XML markup, which the parser receives with tool_output_mode=GuidedJson{named_tool=get_weather}. This is also covered in: e2e case-0047-tool_add_named__non-stream-budget_capped.json, e2e case-0048-tool_add_named__stream-budget_capped.json, e2e case-0054-tool_translate_named__stream-budget_capped.json (each with its `-budget_unlimited` pair).",
     [],
     [{"kind": "tool_call", "name": "get_weather", "arguments": {"city": "Paris"}}],
     {"starting_state": "None", "tool_output_mode": "GuidedJson", "named_tool": "get_weather"},
     every_family(GUIDED_NAMED_ARGS, GUIDED_UNSUPPORTED,
                  {"verdict": "match", "note": "Dynamo v2 unified parser with tool_output_mode=GuidedJson{named_tool=get_weather}"})),

    ("guided_json_required_tool",
     "Guided decoding with required tool (tool_choice=required or auto after tool narrowing). The model emits a JSON array of call objects, parsed with tool_output_mode=GuidedJson{named_tool=None}. This is also covered in: e2e case-0129-lifecycle_single_result__stream-budget_capped.json, e2e case-0145-lifecycle_chained_calculation__stream-budget_capped.json (FIRST step of each; both with their `-budget_unlimited` pair).",
     [],
     [{"kind": "tool_call", "name": "get_weather", "arguments": {"city": "Paris"}}],
     {"starting_state": "None", "tool_output_mode": "GuidedJson", "named_tool": None},
     every_family(GUIDED_ONE_CALL, GUIDED_UNSUPPORTED,
                  {"verdict": "match", "note": "Dynamo v2 unified parser with tool_output_mode=GuidedJson{named_tool=None}"})),

    # Argument VALUE shapes on the guided path. `arg_unicode` already covers a non-ASCII
    # value, but only in native mode, where the value sits as raw text between markers and
    # no escaping is involved. Guided decoding carries the same value as a JSON string, so
    # the escaping is the parser's problem only here — covering it natively proves nothing
    # about this path.
    ("guided_json_escaped_string_args",
     "A named choice whose argument value carries non-ASCII, escaped quotes and Windows backslashes. A named choice constrains output to the argument object alone and the parser passes that object through verbatim, so every escape has to survive untouched: re-escaping or unescaping here hands the tool a different string than the model wrote, and the tool still runs. This is also covered in: e2e case-0105-schema_escaped_unicode_string__non-stream-budget_capped.json (and its `-budget_unlimited` pair).",
     [],
     [{"kind": "tool_call", "name": "run", "arguments": {"cmd": 'echo "雪" > C:\\tmp\\a.txt'}}],
     {"starting_state": "None", "tool_output_mode": "GuidedJson", "named_tool": "run"},
     every_family(r'{"cmd": "echo \"雪\" > C:\\tmp\\a.txt"}', GUIDED_UNSUPPORTED,
                  {"verdict": "match", "note": "escapes and non-ASCII survive GuidedJson{named_tool=run} verbatim"})),

    ("guided_json_array_argument",
     "A required choice whose argument VALUE is an array. Every other guided case passes scalar arguments, and the array-shaped payloads in this group are arrays OF CALLS — one level up. A list-valued argument has to reach the tool as a list; arriving as its string rendering is a silently wrong call, not a failed one. This is also covered in: e2e case-0108-schema_array__stream-budget_capped.json (and its `-budget_unlimited` pair).",
     [],
     [{"kind": "tool_call", "name": "sum_values", "arguments": {"values": [2, 3, 5]}}],
     {"starting_state": "None", "tool_output_mode": "GuidedJson", "named_tool": None},
     every_family('[{"name": "sum_values", "arguments": {"values": [2, 3, 5]}}]', GUIDED_UNSUPPORTED,
                  {"verdict": "match", "note": "list-valued argument stays a list through GuidedJson{named_tool=None}"})),

    ("guided_json_two_calls",
     "A required choice returns an array containing two calls. With no channel prefilled, both must surface as separate ordered events with distinct indices.",
     [],
     [{"kind": "tool_call", "name": "get_weather", "arguments": {"city": "Paris"}},
      {"kind": "tool_call", "name": "run", "arguments": {"cmd": "git log"}}],
     {"starting_state": "None", "tool_output_mode": "GuidedJson", "named_tool": None},
     every_family(GUIDED_TWO_CALLS, GUIDED_UNSUPPORTED,
                  {"verdict": "match", "note": "two DIFFERENT tools in one array, ordered"})),

    ("guided_json_partial_calls",
     "A guided array where one element is not a call (no `name`), with nothing pre-filled. The whole payload surfaces as text and no call is dispatched, because extracting a call from a document that failed validation would fail OPEN on a side-effecting action.",
     ["P2"],
     [{"kind": "text", "text": '[{"name": "get_weather", "arguments": {"city": "Paris"}}, {"arguments": {"city": "Tokyo"}}]'}],
     {"starting_state": "None", "tool_output_mode": "GuidedJson", "named_tool": None},
     every_family('[{"name": "get_weather", "arguments": {"city": "Paris"}}, {"arguments": {"city": "Tokyo"}}]',
                  GUIDED_UNSUPPORTED,
                  {"verdict": "match", "note": "one invalid element voids the whole array; payload surfaces as text"})),

    ("guided_json_list_with_broken_element",
     "A guided array whose SECOND element is not valid JSON — the payload is `[<valid call>, <broken>]`, which is what a constrained decode produces when it is cut off partway through a later call. Output is the whole payload as text and no call, same as 31-3 but reached differently: there the array parsed and one element failed to convert, here the array does not parse at all, so per-element recovery never gets a chance. Both land on all-or-nothing, which is the point — a half-validated array must not dispatch the half that looked fine.",
     ["P2"],
     [{"kind": "text", "text": '[{"name": "get_weather", "arguments": {"city": "Paris"}}, {"name": "run", "arguments": {"cmd": ]'}],
     {"starting_state": "None", "tool_output_mode": "GuidedJson", "named_tool": None},
     every_family('[{"name": "get_weather", "arguments": {"city": "Paris"}}, {"name": "run", "arguments": {"cmd": ]',
                  GUIDED_UNSUPPORTED,
                  {"verdict": "match", "note": "the array itself fails to parse; nothing is dispatched"})),

    # --- Guided decoding: the SURROUNDINGS, not just the payload -----------------
    # Every guided case above varies the PAYLOAD and delivers it bare. Nothing
    # varied what sits AROUND it, and that is precisely where every guided defect
    # in this surface has been found: prose before a thought surfaced the model's
    # private reasoning as the answer, a narrated invoke swallowed the payload, an
    # orphan closer leaked, and markup bracketing the payload lost the call. Those
    # are pinned by unit tests; without these cases the corpus reads green through
    # all of them.
    ("guided_json_after_reasoning",
     "A normal thought precedes the guided payload. The reasoning closes before the constrained call begins; both must survive in their original order.",
     [],
     [{"kind": "reasoning", "text": "checking"},
      {"kind": "tool_call", "name": "get_weather", "arguments": {"city": "Paris"}}],
     {"starting_state": "None", "tool_output_mode": "GuidedJson", "named_tool": None},
     guided_surroundings(
         lambda fam: f"{control_tokens(fam)[0]}checking{control_tokens(fam)[1]}{GUIDED_ONE_CALL}",
         "reasoning closes, then the guided payload dispatches")),

    ("guided_json_marker_inside_argument",
     "A control marker of the family's OWN grammar inside a guided argument VALUE. Once the payload has opened, a marker is argument DATA and must survive byte-exact (`I7`) — re-reading it as a channel token corrupts the call the tool receives while looking like a successful dispatch.",
     ["I7"],
     [{"kind": "tool_call", "name": "log", "arguments": {"note": None}}],  # filled per family
     {"starting_state": "None", "tool_output_mode": "GuidedJson", "named_tool": None},
     guided_surroundings(
         lambda fam: json.dumps(
             [{"name": "log", "arguments": {"note": control_tokens(fam)[1]}}], ensure_ascii=False),
         "a marker inside a started payload stays argument data",
         lambda fam: control_tokens(fam)[1])),

    ("guided_json_tool_open_before_payload",
     "A native tool OPENER precedes the constrained payload. Guided decoding delivers the call as JSON, so leading markup is stray: it must be stripped, not carried into the payload buffer where it breaks the parse and costs the call.",
     ["P2"],
     [{"kind": "tool_call", "name": "get_weather", "arguments": {"city": "Paris"}}],
     {"starting_state": "None", "tool_output_mode": "GuidedJson", "named_tool": None},
     guided_surroundings(
         lambda fam: f"{control_tokens(fam)[2]}{GUIDED_ONE_CALL}",
         "leading tool markup stripped; the call still dispatches")),

    ("guided_json_tool_close_after_payload",
     "A native tool CLOSER follows the payload. The leading side was handled long before this one: once the payload's opening brace latches visible-only, every later byte is appended verbatim, so a trailing marker rides into the buffer and the call is lost. Markers can BRACKET a payload, not only precede it.",
     ["P2"],
     [{"kind": "tool_call", "name": "get_weather", "arguments": {"city": "Paris"}}],
     {"starting_state": "None", "tool_output_mode": "GuidedJson", "named_tool": None},
     guided_surroundings(
         lambda fam: f"{GUIDED_ONE_CALL}{control_tokens(fam)[3]}",
         "trailing tool markup stripped; the call still dispatches")),

    ("guided_json_wrapped_in_tool_markup",
     "The payload wrapped in a full native envelope, opener AND closer. This is the shape a template emits when guided decoding is applied INSIDE a tool block; handling only one end still loses the call.",
     ["P2"],
     [{"kind": "tool_call", "name": "get_weather", "arguments": {"city": "Paris"}}],
     {"starting_state": "None", "tool_output_mode": "GuidedJson", "named_tool": None},
     guided_surroundings(
         lambda fam: f"{control_tokens(fam)[2]}{GUIDED_ONE_CALL}{control_tokens(fam)[3]}",
         "envelope stripped at both ends; the call still dispatches")),

    ("guided_json_narrated_invoke_in_reasoning",
     "The model NARRATES a tool opener while thinking, then the real call arrives as JSON. Guided decoding leaves the reasoning channel unconstrained, so that markup is prose the model wrote — treating it as structure ends the turn and discards the payload.",
     ["P2"],
     [{"kind": "reasoning", "text": "I'll use  next"},
      {"kind": "tool_call", "name": "get_weather", "arguments": {"city": "Paris"}}],
     {"starting_state": "None", "tool_output_mode": "GuidedJson", "named_tool": None},
     guided_surroundings(
         lambda fam: f"{control_tokens(fam)[0]}I'll use {guided_invoke_prefix(fam) if fam == 'deepseek_v41' else control_tokens(fam)[2]} next{control_tokens(fam)[1]}{GUIDED_ONE_CALL}",
         "narrated markup stripped, thought preserved, payload survives",
         # Beam: its markers are special tokens, so the narrated header IS a header. It
         # addresses an undeclared tool with a junk body, so the call is dropped and the
         # thought keeps only the words before it; the payload still survives.
         fill=lambda fam: ([{"kind": "reasoning", "text": "I'll use "},
                            {"kind": "tool_call", "name": "get_weather", "arguments": {"city": "Paris"}}]
                           if fam == "beam" else None))),

    ("gemma4_guided_json_visible_call_prose_before_reasoning",
     "Gemma 4 only: ordinary visible prose contains `call:` immediately before a thought and the guided JSON payload. `call:` is a lexical prefix in Gemma's native invoke grammar, but without a valid native call body here it is prose and must remain visible. This cannot be shared with another family because only Gemma's grammar assigns any possible structural meaning to this exact prefix.",
     [],
     [{"kind": "text", "text": "I will call: you tomorrow"},
      {"kind": "reasoning", "text": "checking"},
      {"kind": "tool_call", "name": "get_weather", "arguments": {"city": "Paris"}}],
     {"starting_state": "None", "tool_output_mode": "GuidedJson", "named_tool": None},
     OnlyFamilies({
         "gemma4": (
             "I will call: you tomorrow<|channel>thought\nchecking<channel|>" + GUIDED_ONE_CALL,
             D("UNSUPPORTED", "vLLM base case does not use GuidedJson"),
             {"verdict": "match", "note": "ordinary visible `call:` prose remains text before reasoning and the guided payload"},
         ),
     })),

    ("gemma4_guided_json_malformed_call_prefix_before_reasoning",
     "Gemma 4 only: an incomplete native-looking `call:get_weather` prefix sits immediately before a thought and guided JSON. It lacks the `{` that makes a Gemma invoke body, so it remains visible text rather than being silently suppressed; the following thought and guided payload still route normally. The contrast with `gemma-1` fixes the boundary between ordinary `call:` prose and a malformed-but-still-visible Gemma candidate.",
     ["P2"],
     [{"kind": "text", "text": "call:get_weather"},
      {"kind": "reasoning", "text": "secret"},
      {"kind": "tool_call", "name": "get_weather", "arguments": {"city": "Paris"}}],
     {"starting_state": "None", "tool_output_mode": "GuidedJson", "named_tool": None},
     OnlyFamilies({
         "gemma4": (
             "call:get_weather<|channel>thought\nsecret<channel|>" + GUIDED_ONE_CALL,
             D("UNSUPPORTED", "vLLM base case does not use GuidedJson"),
             {"verdict": "match", "note": "incomplete Gemma `call:get_weather` remains visible; reasoning and guided payload survive"},
         ),
     })),

    ("guided_json_prose_before_reasoning",
     "Visible prose precedes a thought and the guided payload. The leading prose must not latch the payload buffer and cause later private reasoning to appear in the visible answer.",
     ["P2"],
     [{"kind": "text", "text": "Sure. "},
      {"kind": "reasoning", "text": "checking"},
      {"kind": "tool_call", "name": "get_weather", "arguments": {"city": "Paris"}}],
     {"starting_state": "None", "tool_output_mode": "GuidedJson", "named_tool": None},
     guided_surroundings(
         lambda fam: f"Sure. {control_tokens(fam)[0]}checking{control_tokens(fam)[1]}{GUIDED_ONE_CALL}",
         "prose stays visible text, the thought stays reasoning, the call dispatches")),

    ("guided_json_orphan_reason_close_before_payload",
     "An orphan reasoning CLOSER with nothing open, ahead of the payload. The native scanner strips a stray closer wherever it appears before an opener; the guided path must agree or the same bytes read differently by request mode (`I3`).",
     ["P2"],
     [{"kind": "tool_call", "name": "get_weather", "arguments": {"city": "Paris"}}],
     {"starting_state": "None", "tool_output_mode": "GuidedJson", "named_tool": None},
     guided_surroundings(
         lambda fam: f"{control_tokens(fam)[1]}{GUIDED_ONE_CALL}",
         "orphan reasoning closer stripped; the call still dispatches")),

    ("guided_json_orphan_tool_close_before_payload",
     "An orphan tool CLOSER before the payload. Paired with the opener case above: for a while the closer was stripped and the opener beside it was not, so which marker leaked depended on which one the model happened to emit.",
     ["P2"],
     [{"kind": "tool_call", "name": "get_weather", "arguments": {"city": "Paris"}}],
     {"starting_state": "None", "tool_output_mode": "GuidedJson", "named_tool": None},
     guided_surroundings(
         lambda fam: f"{stray_tool_close(fam)}{GUIDED_ONE_CALL}",
         "orphan tool closer stripped; the call still dispatches")),

    ("guided_json_invalid_call",
     "Guided decoding emits JSON that is well-formed but is NOT a tool call — no `name`, so there is nothing to dispatch. Policy P2: surface the payload as visible content rather than dropping it or erroring. Dropping it would lose the model's entire output; erroring would fail a request the user can still read.",
     ["P2"],
     [{"kind": "text", "text": '{"unexpected": "shape"}'}],
     {"starting_state": "None", "tool_output_mode": "GuidedJson", "named_tool": None},
     every_family('{"unexpected": "shape"}', GUIDED_UNSUPPORTED,
                  {"verdict": "match", "note": "P2: unparseable-as-a-call guided payload is surfaced as text"})),

    ("guided_json_malformed_json",
     "Guided decoding emits JSON that does not PARSE — a truncated object, which is what a constrained decode looks like when the token budget runs out mid-payload. Distinct from the wrong-shape case: there the JSON was valid and merely not a call. Policy P2: surface the bytes as visible content. Dropping them loses the output silently, and erroring fails a request whose text is still readable.",
     ["P2"],
     [{"kind": "text", "text": '{"name": "get_weather", "arguments": {"city": "Par'}],
     {"starting_state": "None", "tool_output_mode": "GuidedJson", "named_tool": None},
     every_family('{"name": "get_weather", "arguments": {"city": "Par', GUIDED_UNSUPPORTED,
                  {"verdict": "match", "note": "P2: unparseable guided payload is surfaced as text, not dropped"})),

    ("prefilled_reasoning_with_tool",
     "Reasoning channel is pre-filled by the generation prompt (policy P5), so the stream begins inside <think> with no opener. The model emits: reasoning tail -> closer -> tool call.",
     ["P5"],
     [{"kind": "reasoning", "text": "checking weather"},
      {"kind": "tool_call", "name": "get_weather", "arguments": {"city": "Paris"}}],
     {"starting_state": "Reasoning", "tool_output_mode": "Native", "named_tool": None},
     {"finish_reason": "stop"},
     {
        "deepseek_v41": ("checking weather</think>" + r_tool("deepseek_v41", "get_weather", "city", "Paris", 0), M, M),
        "qwen3": ("checking weather</think><tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>",
                  D("UNSUPPORTED", "vLLM base case doesn't set a starting channel state; conformance captures default generation only"),
                  {"verdict": "match", "note": "Dynamo v2 unified parser with starting_state=Reasoning"}),
        # The prompt consumed `<|start|>assistant to=self<|message|>`, so the stream opens
        # INSIDE the thought and its first `<|eom|>` closes it.
        "muse_glimmer": ("checking weather<|eom|><|start|>assistant to=get_weather<|message|><atem:function_calls>\n<atem:invoke name=\"get_weather\">\n<atem:parameter name=\"city\">Paris</atem:parameter>\n</atem:invoke>\n</atem:function_calls><|eom|>",
                         V_MUSE,
                         {"verdict": "match", "note": "starting_state=Reasoning opens the scanner in the to=self channel"}),
        "gemma4": ("checking weather<channel|><|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>}<tool_call|>", M, M),
        "kimi_k2": ("checking weather</think><|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{\"city\": \"Paris\"}<|tool_call_end|><|tool_calls_section_end|>", M, M),
        "kimi_k3": ("checking weather" + k3_close("think")
                    + r_tool("kimi_k3", "get_weather", "city", "Paris", 0),
                    M, M),
     }),

    ("prefilled_reasoning_then_text_then_tool",
     "Reasoning is pre-filled, the model closes it, writes VISIBLE prose, and only then calls a tool. All three channels in one prefilled stream. The prose must surface as text, not be swept into the reasoning span it follows nor into the call it precedes — the boundary on each side is a different marker, and a prefilled stream has no opener to anchor the first one.",
     ["P5"],
     [{"kind": "reasoning", "text": "weighing options"},
      {"kind": "text", "text": "Here's what I found: "},
      {"kind": "tool_call", "name": "get_weather", "arguments": {"city": "Paris"}}],
     {"starting_state": "Reasoning", "tool_output_mode": "Native", "named_tool": None},
     {
        "deepseek_v41": ("weighing options</think>Here's what I found: " + r_tool("deepseek_v41", "get_weather", "city", "Paris", 0), M, M),
        "qwen3": ("weighing options</think>Here's what I found: <tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>",
                  D("UNSUPPORTED", "vLLM base case doesn't set a starting channel state; conformance captures default generation only"),
                  {"verdict": "match", "note": "reasoning -> text -> call, all three ordered in one prefilled stream"}),
        "muse_glimmer": ("weighing options<|eom|><|start|>assistant to=user<|message|>Here's what I found: <|eom|><|start|>assistant to=get_weather<|message|><atem:function_calls>\n<atem:invoke name=\"get_weather\">\n<atem:parameter name=\"city\">Paris</atem:parameter>\n</atem:invoke>\n</atem:function_calls><|eom|>",
                         V_MUSE,
                         {"verdict": "match", "note": "all three channels ordered out of one prefilled stream"}),
        "gemma4": ("weighing options<channel|>Here's what I found: <|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>}<tool_call|>", M, M),
        "kimi_k2": ("weighing options</think>Here's what I found: <|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{\"city\": \"Paris\"}<|tool_call_end|><|tool_calls_section_end|>", M, M),
        "kimi_k3": ("weighing options" + k3_close("think")
                    + r_text("kimi_k3", "Here's what I found: ")
                    + r_tool("kimi_k3", "get_weather", "city", "Paris", 0),
                    M, M),
     }),

    ("prefilled_reasoning_then_text",
     "Reasoning is pre-filled, the model closes it and answers in prose with NO tool call — the ordinary shape of a prefilled request that needs no tool. Pins that closing a prefilled thought returns the stream to visible content rather than leaving it in reasoning, which would swallow the whole answer.",
     ["P5"],
     [{"kind": "reasoning", "text": "no tool needed"},
      {"kind": "text", "text": "The answer is 42."}],
     {"starting_state": "Reasoning", "tool_output_mode": "Native", "named_tool": None},
     {
        "deepseek_v41": ("no tool needed</think>The answer is 42.", M, M),
        "qwen3": ("no tool needed</think>The answer is 42.",
                  D("UNSUPPORTED", "vLLM base case doesn't set a starting channel state; conformance captures default generation only"),
                  {"verdict": "match", "note": "closing a prefilled thought returns to visible content"}),
        "muse_glimmer": ("no tool needed<|eom|><|start|>assistant to=user<|message|>The answer is 42.<|eot|>",
                         V_MUSE,
                         {"verdict": "match", "note": "closing a prefilled thought returns the stream to visible content"}),
        "gemma4": ("no tool needed<channel|>The answer is 42.", M, M),
        "kimi_k2": ("no tool needed</think>The answer is 42.", M, M),
        "kimi_k3": ("no tool needed" + k3_close("think")
                    + r_text("kimi_k3", "The answer is 42."),
                    M, M),
     }),

    ("prefilled_reasoning_with_guided_json",
     "Reasoning channel is pre-filled (policy P5), stream begins inside <think> with no opener, and the model emits tool calls as guided JSON.",
     ["P5"],
     [{"kind": "reasoning", "text": "checking weather"},
      {"kind": "tool_call", "name": "get_weather", "arguments": {"city": "Paris"}}],
     {"starting_state": "Reasoning", "tool_output_mode": "GuidedJson", "named_tool": None},
     {"finish_reason": "stop"},
     guided_surroundings(
         lambda fam: f"checking weather{control_tokens(fam)[1]}{GUIDED_ONE_CALL}",
         "Dynamo v2 unified parser with starting_state=Reasoning and tool_output_mode=GuidedJson{named_tool=None}")),

    ("prefilled_reasoning_redundant_opener",
     "Reasoning is pre-filled, and the backend ALSO re-emits the `<think>` opener the prompt already wrote. Exactly one such echo is consumed rather than leaked into reasoning_content; a second would be stray markup and stripped (I3). This is the only case where a prefilled stream legitimately carries an opener.",
     [],
     [{"kind": "reasoning", "text": "checking weather"}, {"kind": "tool_call", "name": "get_weather", "arguments": {"city": "London"}}],
     {"starting_state": "Reasoning", "tool_output_mode": "Native", "named_tool": None},
     {"finish_reason": "stop"},
     {
        "deepseek_v41": ("<think>checking weather</think>" + r_tool("deepseek_v41", "get_weather", "city", "London", 0), M, M),
        "gemma4": ("<|channel>thought\nchecking weather<channel|><|tool_call>call:get_weather{city:<|\"|>London<|\"|>}<tool_call|>", M, M),
        # Muse's opener is the routed header itself. Re-emitting it cuts a ZERO-length
        # body, which must neither emit an event nor arm the adjacency newline.
        "muse_glimmer": ("<|start|>assistant to=self<|message|>checking weather<|eom|><|start|>assistant to=get_weather<|message|><atem:function_calls>\n<atem:invoke name=\"get_weather\">\n<atem:parameter name=\"city\">London</atem:parameter>\n</atem:invoke>\n</atem:function_calls><|eom|>",
                         V_MUSE,
                         {"verdict": "match", "note": "the echoed header is consumed, not leaked, and adds no separator"}),
        "qwen3": ("<think>checking weather</think><tool_call>\n<function=get_weather>\n<parameter=city>\nLondon\n</parameter>\n</function>\n</tool_call>", M, M),
        "kimi_k2": ("<think>checking weather</think><|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{\"city\": \"London\"}<|tool_call_end|><|tool_calls_section_end|>", M, M),
        "kimi_k3": (r_reason("kimi_k3", "checking weather")
                    + r_tool("kimi_k3", "get_weather", "city", "London", 0),
                    M, M),
     }),


    ("prefilled_reasoning_truncated",
     "Reasoning is pre-filled and the token budget runs out mid tool call — the input is truncated, which is what finish_reason=length MEANS on the wire. Policy P2: keep the completed reasoning, drop the incomplete call, no error and no leaked markup.",
     ["P2"],
     [{"kind": "reasoning", "text": "analyzing data"}],
     {"starting_state": "Reasoning", "tool_output_mode": "Native", "named_tool": None},
     {"finish_reason": "length"},
     {
        "deepseek_v41": ('analyzing data</think><｜DSML｜ calls><｜DSML｜ invoke name="get_weather"><｜DSML｜ parameter name="city" string="true">Par', M, M),
        "deepseek_v4": ('analyzing data</think><｜DSML｜tool_calls><｜DSML｜invoke name="get_weather"><｜DSML｜parameter name="city" string="true">Par', M, M),
        "gemma4": ("analyzing data<channel|><|tool_call>call:get_weather{city:<|\"|>Par",
                   D("ERROR", "native Gemma4UnifiedParser finish() returns a hard Err on a partial call rather than recovering"),
                   {"verdict": "match", "note": "P2: drop the partial trailing call, keep the prefilled reasoning"}),
        "muse_glimmer": ("analyzing data<|eom|><|start|>assistant to=get_weather<|message|><atem:function_calls>\n<atem:invoke name=\"get_weather\">\n<atem:parameter name=\"city\">Par",
                         V_MUSE,
                         {"verdict": "match", "note": "P2: the unterminated invoke is dropped, the prefilled thought survives"}),
        "qwen3": ("analyzing data</think><tool_call>\n<function=get_weather>\n<parameter=city>\nPar",
                  {"verdict": "match", "note": "P2: drop the unterminated call and keep prefilled output"},
                  {"verdict": "match", "note": "P2: v2 drops the partial trailing call, keeps the prefilled reasoning"}),
        "kimi_k2": ("analyzing data</think><|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{\"city\": \"Par",
                    {"verdict": "match", "note": "P2: drop the unterminated call and keep prefilled output"},
                    {"verdict": "match", "note": "P2: v2 drops the partial trailing call, keeps the prefilled reasoning"}),
        "kimi_k3": ("analyzing data" + k3_close("think") + k3_tools(
                        k3_call("get_weather", 1, k3_open(
                            "argument", [("key", "city"), ("type", "string")]
                        ) + "Par", close=False),
                        close=False),
                    M,
                    {"verdict": "match", "note": "P2: drop the partial call and keep prefilled K3 reasoning"}),
     }),


    ("prefilled_response_reasoning_markers_literal",
     "A prefilled-Response case where starting_state=Response is observable. Response says the prompt already opened visible content, so this stream has no reasoning channel at all and `<think>`/`</think>` are ordinary characters the model happened to write — they must reach the user as text, markers and all. Marker-free prefilled-Response variants parse identically under starting_state=None and are deliberately omitted. This one does not.",
     ["P5"],
     # The literal text is the family's OWN reasoning markers, so the golden is
     # filled per family (below) rather than hardcoding one grammar's.
     [{"kind": "text", "text": None},
      {"kind": "tool_call", "name": "get_weather", "arguments": {"city": "Paris"}}],
     {"starting_state": "Response", "tool_output_mode": "Native", "named_tool": None},
     {"finish_reason": "stop"},
     {
        "deepseek_v41": ("<think>literal</think> then a call" + r_tool("deepseek_v41", "get_weather", "city", "Paris", 0),
                         M, M, "<think>literal</think> then a call"),
        # Muse answers this scenario DIFFERENTLY from the marker-pair families, and the
        # difference is the point. Response turns the turn-start latch off, so the bare
        # `to=self<|message|>` is prose rather than a live header — the routing is
        # correctly not honoured. But muse's markers ARE special tokens, so `I3` strips
        # them from the text on the way out: they never reach the client, markers and all.
        # The recipient word survives because it is ordinary characters, not a marker.
        "muse_glimmer": ("to=self<|message|>literal<|eom|> then a call<|eom|><|start|>assistant to=get_weather<|message|><atem:function_calls>\n<atem:invoke name=\"get_weather\">\n<atem:parameter name=\"city\">Paris</atem:parameter>\n</atem:invoke>\n</atem:function_calls><|eom|>",
                         V_MUSE,
                         {"verdict": "match", "note": "the header is not honoured as routing (Response clears the latch), and I3 strips the markers themselves from the text"},
                         "to=selfliteral then a call"),
        "qwen3": ("<think>literal</think> then a call<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>",
                  D("UNSUPPORTED", "vLLM base case doesn't set a starting channel state; it reads the markers as a reasoning span"),
                  {"verdict": "match", "note": "reasoning disabled, so the markers stay literal text"},
                  "<think>literal</think> then a call"),
        "gemma4": ("<|channel>thought\nliteral<channel|> then a call<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>}<tool_call|>",
                   D("UNSUPPORTED", "vLLM base case doesn't set a starting channel state; it reads the markers as a reasoning span"),
                   {"verdict": "match", "note": "reasoning disabled, so `<|channel>thought\\n…<channel|>` stays literal text — role label included"},
                   "<|channel>thought\nliteral<channel|> then a call"),
        "kimi_k2": ("<think>literal</think> then a call<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{\"city\": \"Paris\"}<|tool_call_end|><|tool_calls_section_end|>",
                    D("UNSUPPORTED", "vLLM base case doesn't set a starting channel state; it reads the markers as a reasoning span"),
                    M,
                    "<think>literal</think> then a call"),
        "kimi_k3": (k3_channel("think", "literal") + " then a call"
                    + r_tool("kimi_k3", "get_weather", "city", "Paris", 0),
                    M, M,
                    k3_channel("think", "literal") + " then a call"),
     }),

    (
        "deepseek_v41_mixed_control_text_in_string",
        "A native string argument contains its family's reasoning and tool delimiters, entity text, quotes, a backslash, a newline, and surrounding spaces. Preserve the decoded string exactly; this combines marker classes and string preservation beyond 7-2's single closer.",
        ["I7"],
        [{"kind": "tool_call", "name": "f", "arguments": {"x": None}}],
        {"starting_state": "None", "tool_output_mode": "Native", "named_tool": None},
        {
            family: (
                r_tool(family, "f", "x", value, 0),
                VLLM_UNCAPTURABLE.get(family, M),
                M,
                value,
            )
            for family, value in _MIXED_CONTROL_STRINGS.items()
        },
    ),
]

EDGE += [
    ("kimi_k3_typed_argument_values",
     "Kimi K3 native XTML carries each argument in its own typed channel. String, number, boolean, object, array, and null values must preserve their JSON types instead of being coerced to strings.",
     ["I7"],
     [{"kind": "tool_call", "name": "run", "arguments": {
         "cmd": "echo ok", "count": 2, "force": True,
         "options": {"cwd": "/tmp"}, "tags": ["a", "b"], "note": None,
     }}],
     {"starting_state": "None", "tool_output_mode": "Native", "named_tool": None},
     OnlyFamilies({
         "kimi_k3": (
             k3_tools(k3_call("run", 1, "".join([
                 k3_argument("cmd", "string", "echo ok"),
                 k3_argument("count", "number", "2"),
                 k3_argument("force", "boolean", "true"),
                 k3_argument("options", "object", '{"cwd": "/tmp"}'),
                 k3_argument("tags", "array", '["a", "b"]'),
                 k3_argument("note", "null", "null"),
             ]))),
             M, M,
         ),
     })),

    ("kimi_k3_raw_json_arguments",
     "Kimi K3 can carry the complete argument object in a raw JSON channel. Nested values and marker-looking strings remain JSON data until that channel's own closer.",
     ["I7"],
     [{"kind": "tool_call", "name": "run", "arguments": {
         "cmd": "literal <|close|>call<|sep|>",
         "options": {"retries": 2},
     }}],
     {"starting_state": "None", "tool_output_mode": "Native", "named_tool": None},
     OnlyFamilies({
         "kimi_k3": (
             k3_raw_tool("run", '{"cmd":"literal <|close|>call<|sep|>","options":{"retries":2}}'),
             M, M,
         ),
     })),

    ("kimi_k3_spaced_xtml_markers",
     "Kimi K3 accepts the checkpoint's spaced XTML marker spelling for response, tools, call, argument, message, and end-of-message framing.",
     [],
     [{"kind": "text", "text": "Checking. "},
      {"kind": "tool_call", "name": "get_weather", "arguments": {"city": "Paris"}}],
     {"starting_state": "None", "tool_output_mode": "Native", "named_tool": None},
     OnlyFamilies({
         "kimi_k3": (
             k3_channel("response", "Checking. ", spaced=True)
             + k3_tools(k3_call(
                 "get_weather", 1, k3_argument("city", "string", "Paris", spaced=True),
                 spaced=True), spaced=True)
             + k3_close("message", spaced=True) + "<|end_of_msg|>",
             M, M,
         ),
     })),

    ("kimi_k3_message_end_after_response",
     "Canonical Kimi K3 message-close and end-of-message markers terminate a completed response without suppressing its visible text or leaking framing.",
     [],
     [{"kind": "text", "text": "done"}],
     {"starting_state": "None", "tool_output_mode": "Native", "named_tool": None},
     OnlyFamilies({
         "kimi_k3": (
             k3_channel("response", "done") + k3_close("message") + "<|end_of_msg|>",
             M, M,
         ),
     })),

    ("kimi_k3_elided_think_close_to_response",
     "Kimi K3 may omit the think closer when it opens the response channel. The channel transition still ends private reasoning and emits the following response as visible text.",
     ["P2"],
     [{"kind": "reasoning", "text": "checking"},
      {"kind": "text", "text": "The answer is 42."}],
     {"starting_state": "None", "tool_output_mode": "Native", "named_tool": None},
     OnlyFamilies({
         "kimi_k3": (
             k3_open("think") + "checking" + k3_channel("response", "The answer is 42."),
             M, M,
         ),
     })),

    ("kimi_k3_malformed_call_then_valid",
     "A malformed Kimi K3 call body followed by a complete call resynchronizes at the later call. The malformed prefix neither leaks as text nor costs the valid call.",
     ["P2"],
     [{"kind": "tool_call", "name": "g", "arguments": {"y": "2"}}],
     {"starting_state": "None", "tool_output_mode": "Native", "named_tool": None},
     OnlyFamilies({
         "kimi_k3": (
             k3_open("tools")
             + k3_open("call", [("tool", "bad"), ("index", "1")]) + "not-an-argument"
             + k3_call("g", 2, k3_argument("y", "string", "2"))
             + k3_close("tools"),
             M, M,
         ),
     })),

    ("kimi_k3_raw_json_eof",
     "EOF inside Kimi K3's raw JSON argument channel drops the incomplete call and emits nothing, without leaking the partial JSON as visible text.",
     ["P2"],
     [],
     {"starting_state": "None", "tool_output_mode": "Native", "named_tool": None},
     OnlyFamilies({
         "kimi_k3": (
             k3_open("tools") + k3_open("call", [("tool", "run"), ("index", "1")])
             + k3_open("json", [("type", "object")]) + '{"cmd":"unfinished',
             M, M,
         ),
     })),

    ("kimi_k3_guided_native_wrapper",
     "Guided Kimi K3 output may arrive inside a native XTML tools/call/argument wrapper. The wrapper is stripped and the guided JSON payload still dispatches exactly once.",
     ["I3"],
     [{"kind": "tool_call", "name": "get_weather", "arguments": {"city": "Paris"}}],
     {"starting_state": "None", "tool_output_mode": "GuidedJson", "named_tool": None},
     OnlyFamilies({
         "kimi_k3": (
             k3_tools(k3_call("ignored", 1, k3_argument("quoted", "string", "literal")))
             + GUIDED_ONE_CALL,
             GUIDED_UNSUPPORTED, M,
         ),
     })),
]


# ---------------------------------------------------------------------------
# GENERATED semantic cross-product.
#
# Four hand-authored rows would have closed exactly the hole Devin found and left
# the NEXT crossing open — the defects live in axis crossings, not in the example
# that happened to expose them. So the guided edge region is a PRODUCT of two
# authored bases: what the payload is, and what surrounds it.
#
# Measured before this existed (qwen3, guided cases, surrounding-markup x
# golden-dispatches-a-call): 12 / 5 / 5 / ZERO. The empty quadrant — markup
# present AND no call recoverable — is where the P2 recovery leak and the
# unbounded invoke-header scan both lived, and no authored case could reach it.
#
# Products that say nothing are dropped by a predicate rather than never written,
# so the reason a crossing is absent stays visible here instead of being implicit
# in someone's case list.
# ---------------------------------------------------------------------------

# name -> (payload text, does a well-formed parse dispatch a call?)
GUIDED_PAYLOADS = {
    "valid": (GUIDED_ONE_CALL, {"city": "Paris"}),
    # WHICH LAYER rejects the payload, not a vague "malformed". Only the first of
    # these fails to parse; the other two are well-formed JSON that is not a call
    # list. Collapsing them under one word made the corpus read as covering a
    # syntax failure when it was really testing schema rejection.
    "syntax_error": ('[{"name": "get_weather", "arguments": {"city": ', None),
    "schema_error_not_a_call": ('{"unexpected": "shape"}', None),
    "schema_error_nameless_element":
        ('[{"name": "get_weather", "arguments": {"city": "Paris"}}, {"arguments": {}}]', None),
    # An argument containing the character a prefix-form invoke header terminates
    # on. `control_marker_at` and `guided_holdback_len` once disagreed about whether
    # such a `>` completed the marker, so `<function=` was neither consumed nor held
    # back and the call was lost as text. Payload shape, so it crosses every
    # surrounding automatically.
    "gt_in_argument": ('[{"name": "get_weather", "arguments": {"city": "a > b"}}]', {"city": "a > b"}),
}

# name -> (wrap(payload, fam) -> input, one-line description of the surrounding)
# The third element is whether the surrounding puts a marker AFTER the payload.
# It decides the recovery bytes and is the `I7`/`I3` boundary: a stripped TAIL
# marker also trims the whitespace it was attached to, while a payload with no
# trailing marker is handed back byte-identical — trailing space and all. The
# corpus records that difference instead of each case guessing at it.
GUIDED_SURROUNDS = {
    "clean": (lambda pay, fam: pay, "no surrounding grammar", False),
    "trailing_close": (lambda pay, fam: f"{pay}{control_tokens(fam)[3]}",
                       "a stray tool CLOSE after the payload", True),
    "wrapped": (lambda pay, fam: f"{control_tokens(fam)[2]}{pay}{control_tokens(fam)[3]}",
                "the payload wrapped in native tool markup", True),
    "bare_opener": (lambda pay, fam: f"{guided_invoke_prefix(fam)}{pay}",
                    "a bare invoke HEADER before the payload, never terminated", False),
    "named_bare_opener": (lambda pay, fam: f"{guided_named_invoke_prefix(fam)}{pay}",
                          "an invoke HEADER containing a tool name but no terminator before the payload", False),
}


def _guided_product():
    """Every (payload x surrounding) crossing that says something distinct.

    `clean` x `valid` is `30-1`/`30-2` and `clean` x the malformed payloads is
    `31-1` through `31-4`; those already exist, so the predicate drops them rather than
    emitting a duplicate under a second name.
    """
    out = []
    for pay_name, (payload, want_args) in GUIDED_PAYLOADS.items():
        dispatches = want_args is not None
        for sur_name, (wrap, sur_desc, strips_tail) in GUIDED_SURROUNDS.items():
            # `clean` is already authored as 30-1/30-2 and 31-1 through 31-4. The
            # `valid` payload crossings are also already authored by hand
            # (guided_json_tool_open_before_payload / _tool_close_after_payload /
            # _wrapped_in_tool_markup) — generating them produced 3 scenarios x 3
            # families = 9 cases with byte-identical (input, init, golden). A
            # duplicate is worse than a gap: it inflates the case count while
            # testing nothing new, and two names for one behaviour drift apart.
            if sur_name == "clean" or pay_name == "valid":
                continue
            if sur_name == "named_bare_opener" and pay_name != "gt_in_argument":
                continue
            scenario = f"guided_json_{pay_name}_{sur_name}"
            golden = ([{"kind": "tool_call", "name": "get_weather",
                        "arguments": want_args}] if dispatches else
                      [{"kind": "text", "text": None}])
            note = (f"guided payload ({pay_name}) with {sur_desc}: "
                    + ("the call still dispatches and no marker reaches the user"
                       if dispatches else
                       "no call is recoverable, and the recovery TEXT carries none "
                       "of the markup the parse stripped"))
            family_inputs = guided_surroundings(
                lambda fam, w=wrap, pl=payload: w(pl, fam),
                note,
                fill=(None if dispatches else
                      (lambda fam, pl=payload, st=strips_tail: pl.rstrip() if st else pl)),
            )
            if scenario == "guided_json_schema_error_not_a_call_bare_opener":
                # GLM's outer tool marker is itself the complete invoke opener.
                # It has no separate bare inner header, so this crossing is the
                # same bare JSON behavior covered by `guided_json_invalid_call`.
                family_inputs = OnlyFamilies({
                    family: spec
                    for family, spec in family_inputs.items()
                    if family != "glm47"
                })
            if scenario == "guided_json_gt_in_argument_named_bare_opener":
                # Beam's recipient sits in the routed header and ends at `<|channel|>`,
                # a special token: there is no name-terminated bare opener for `>` to
                # cut, so the crossing does not exist on its wire either.
                family_inputs = OnlyFamilies({
                    family: spec
                    for family, spec in family_inputs.items()
                    if family not in ("glm47", "beam")
                })
            out.append((
                scenario,
                f"Guided JSON, payload is {pay_name}, surrounded by {sur_desc}. "
                + ("Markers around a recoverable payload must not cost the call (`I3`)."
                   if dispatches else
                   "Nothing parses as a call, so the payload surfaces as text — and the "
                   "text must not contain the control markup that was stripped to "
                   "attempt the parse (`I3`). This crossing had NO case before: every "
                   "authored markup case carried a well-formed payload, and every "
                   "malformed payload was authored bare."),
                ["P2"] if not dispatches else ["I3"],
                golden,
                {"starting_state": "None", "tool_output_mode": "GuidedJson", "named_tool": None},
                {"finish_reason": "stop"},
                family_inputs,
            ))
    return out


EDGE += _guided_product()

EDGE += [
    ("qwen3_guided_non_ascii_header_in_truncated_reasoning",
     "Guided Qwen reasoning ends inside an unfinished function header with a non-ASCII name. "
     "The control prefix is stripped at EOF, but the remaining reasoning bytes survive without a call or visible text.",
     ["I3", "P2"],
     [{"kind": "reasoning", "text": "éaaaaaaaaax"}],
     {"starting_state": "None", "tool_output_mode": "GuidedJson", "named_tool": None},
     OnlyFamilies({"qwen3": ("<think><function=éaaaaaaaaax", GUIDED_UNSUPPORTED, M)})),
    ("qwen3_guided_non_ascii_header_in_closed_reasoning",
     "Guided Qwen reasoning contains a malformed non-ASCII function header closed by its function terminator. "
     "That control span is stripped, the thought closes, and the following guided call dispatches.",
     ["I3"],
     [{"kind": "tool_call", "name": "get_weather", "arguments": {"city": "Paris"}}],
     {"starting_state": "None", "tool_output_mode": "GuidedJson", "named_tool": None},
     OnlyFamilies({"qwen3": (
         "<think><function=éaaaaaaaaax</function></think>" + GUIDED_ONE_CALL,
         GUIDED_UNSUPPORTED, M,
     )})),
    ("guided_json_native_parameter_body_inside_reasoning",
     "Guided reasoning contains a complete native invocation with arguments, followed by the actual guided call. "
     "The native invocation is suppressed as one control span; its argument body must not leak when the header and body arrive separately. "
     "Each family uses its own native argument framing, and the native call differs from the guided call so it cannot substitute for the payload.",
     ["I3", "I5"],
     [{"kind": "reasoning", "text": "before  after"},
      {"kind": "tool_call", "name": "get_weather", "arguments": {"city": "Paris"}}],
     {"starting_state": "None", "tool_output_mode": "GuidedJson", "named_tool": None},
     OnlyFamilies({
         family: spec
         for family, spec in guided_surroundings(
             lambda fam: native_body_in_guided_reasoning(fam, "é🙂<par{value}"),
             "native body is suppressed",
         ).items()
         if family != "glm47"
     })),
    ("guided_json_reasoning_markers_inside_native_parameter",
     "Guided reasoning contains a complete native f(x) invocation whose parameter value quotes the family's reasoning markers. "
     "The parameter owns those markers: they cannot close the surrounding thought or leak parameter tags into visible text. "
     "The native invocation is suppressed; only the following guided Paris call dispatches.",
     ["I3", "I7"],
     [{"kind": "reasoning", "text": "before  after"},
      {"kind": "tool_call", "name": "get_weather", "arguments": {"city": "Paris"}}],
     {"starting_state": "None", "tool_output_mode": "GuidedJson", "named_tool": None},
     OnlyFamilies({
         family: spec
         for family, spec in guided_surroundings(
             lambda fam: native_body_in_guided_reasoning(fam, r_reason(fam, "quoted")),
             "parameter markers remain data",
         ).items()
         if family != "glm47"
     })),
]

for _phase_name, _phase_description, _phase_input, _phase_golden in [
    ("spark_phase_tool_final", "Commentary, native tool, and final keep separate ordered phase boundaries.",
     "<|start|>assistant to=commentary<|message|>Checking.<|eom|>" + r_tool("muse_spark", "get_weather", "city", "Paris", 0) + "<|start|>assistant to=user<|message|>Done.<|eot|>",
     [{"kind":"message_start","phase":"commentary"},{"kind":"text","text":"Checking."},{"kind":"message_end"},
      {"kind":"tool_call","name":"get_weather","arguments":{"city":"Paris"}},
      {"kind":"message_start","phase":"final_answer"},{"kind":"text","text":"Done."},{"kind":"message_end"}]),
    ("spark_phase_repeated", "Repeated commentary messages retain independent start/end boundaries.",
     "<|start|>assistant to=commentary<|message|>One.<|eom|><|start|>assistant to=commentary<|message|>Two.<|eom|>",
     [{"kind":"message_start","phase":"commentary"},{"kind":"text","text":"One."},{"kind":"message_end"},{"kind":"message_start","phase":"commentary"},{"kind":"text","text":"Two."},{"kind":"message_end"}]),
    ("spark_phase_empty", "An explicitly closed empty commentary still has a balanced lifecycle.",
     "<|start|>assistant to=commentary<|message|><|eom|>",
     [{"kind":"message_start","phase":"commentary"},{"kind":"message_end"}]),
    ("spark_phase_eof", "EOF does not invent a completed commentary boundary; the server owns incomplete status.",
     "<|start|>assistant to=commentary<|message|>Working.",
     [{"kind":"message_start","phase":"commentary"},{"kind":"text","text":"Working."}]),
]:
    EDGE.append((_phase_name, _phase_description, ["I2", "I5"], _phase_golden,
                 {"starting_state":"None","tool_output_mode":"Native","named_tool":None,"preserve_message_phases":True},
                 OnlyFamilies({"muse_spark":(_phase_input, V_MUSE_SPARK, M)})))
def _guided_response_markup(fam, recipient, after_payload=False):
    """A response-state control marker followed by (or following) guided JSON.

    `to=…<|message|>` is the control marker that can be quoted for muse. Fixed
    marker-pair grammars cannot quote an opener without a response-state contract, so
    their family-equivalent input is their own reasoning envelope while Response is
    prefilled. Return the expected visible text separately because muse strips its
    special-token framing while the marker-pair families preserve their literal bytes.
    """
    if fam in MUSE_FAMILIES:
        text = f"I mean to={recipient}literal"
        markup = f"I mean to={recipient}<|message|>literal<|eom|>"
    else:
        reason_open, reason_close, _tool_open, _tool_close = control_tokens(fam)
        text = f"I mean {reason_open}{recipient} literal{reason_close}"
        markup = text
    return ((f"{GUIDED_ONE_CALL}{markup}" if after_payload else f"{markup}{GUIDED_ONE_CALL}"), text)



EDGE.append((
    "guided_response_native_wrapper_whitespace_before_prose",
    "Response-prefilled guided output contains a native invocation wrapper, then whitespace and visible prose before the real guided call. "
    "Muse uses its inner ATEM invocation, without a new recipient-routed message envelope. "
    "Whitespace after the suppressed native wrapper is structural across every chunk boundary; the following prose remains visible. "
    "The native Rome argument tests lexical ownership rather than validation against the offered schema; only the guided Paris call dispatches.",
    ["I3", "I7"],
    [{"kind": "text", "text": "ok then "},
     {"kind": "tool_call", "name": "get_weather", "arguments": {"city": "Paris"}}],
    {"starting_state": "Response", "tool_output_mode": "GuidedJson", "named_tool": None},
    guided_surroundings(
        lambda fam: (
            r_bare_tool(fam, "f", "x", {"city": "Rome"}, 0) if fam in MUSE_FAMILIES
            else r_tool_arguments(fam, "f", {"x": {"city": "Rome"}}, 0)
        ) + " \n\t ok then " + GUIDED_ONE_CALL,
        "native wrapper and following structural whitespace are suppressed; prose and guided call are preserved",
    ),
))


for _shape, _body in (
    ("object", '{"city":"Rome"}'),
    ("array", '[{"name":"get_weather","arguments":{"city":"Rome"}}]'),
    ("string", json.dumps('[{"name":"get_weather","arguments":{"city":"Rome"}}]')),
):
    _native_json_inputs = {
        family: (stimulus + opener + _body + closer + GUIDED_ONE_CALL, GUIDED_UNSUPPORTED, M, visible)
        for family, stimulus, visible, opener, closer in (
            ("qwen3", "<think>literal</think>", "<think>literal</think>", "<function=f><parameter=x>", "</parameter></function>"),
            ("muse_glimmer", "to=self<|message|>literal<|eom|>", "to=selfliteral", '<atem:invoke name="f"><atem:parameter name="x">', "</atem:parameter></atem:invoke>"),
        )
    }
    # Both Muse families use inner ATEM invokes in an already-prefilled response.
    _native_json_inputs["muse_spark"] = _native_json_inputs["muse_glimmer"]
    for _family in FAMILIES:
        if _family in _native_json_inputs:
            continue
        _prefix, _visible = _guided_response_markup(_family, "self")
        _native_json_inputs[_family] = (
            _prefix.removesuffix(GUIDED_ONE_CALL)
            + r_tool_arguments(_family, "f", {"x": json.loads(_body)}, 0)
            + GUIDED_ONE_CALL,
            GUIDED_UNSUPPORTED, M, _visible,
        )
    EDGE.append((
        f"guided_json_native_parameter_{_shape}_before_payload",
        f"Response-prefilled guided output contains literal reasoning markup, then a native invocation with a JSON {_shape} argument and the real guided call. "
        "The native body remains owned by its invocation across every chunk boundary. "
        "This tests lexical ownership; the native argument need not satisfy the offered schema. "
        "Only the following Paris call dispatches; the native Rome value cannot become a guided payload or visible text.",
        ["I3", "I7"],
        [{"kind": "text", "text": None}, {"kind": "tool_call", "name": "get_weather", "arguments": {"city": "Paris"}}],
        {"starting_state": "Response", "tool_output_mode": "GuidedJson", "named_tool": None},
        _native_json_inputs,
    ))
    if _shape == "array":
        EDGE.append((
            "guided_json_noncall_before_native_array_before_payload",
            "Response-prefilled required guided output begins with a complete non-call JSON object, "
            "then literal prose, a native invocation containing a call-shaped array, and the real guided call. "
            "The non-call value remains visible; it cannot disable native ownership of the later array. "
            "Only the final Paris call dispatches across every chunk boundary.",
            ["I3", "I7"],
            [{"kind": "text", "text": None}, {"kind": "tool_call", "name": "get_weather", "arguments": {"city": "Paris"}}],
            {"starting_state": "Response", "tool_output_mode": "GuidedJson", "named_tool": None},
            {family: ("{}" + stimulus, verdict, peer, "{}" + visible)
             for family, (stimulus, verdict, peer, visible) in _native_json_inputs.items()},
        ))

EDGE += [
    ("qwen3_guided_reasoning_opener_inside_native_header",
     "Qwen guided output with no prefilled channel has a reasoning opener inside an unfinished native function header. "
     "The reasoning opener wins: without a reasoning closer, the function terminator and following JSON remain reasoning, not a dispatched call.",
     ["I3", "I5"],
     [{"kind": "reasoning", "text": 'x</function>[{"name":"f","arguments":{}}]'}],
     {"starting_state": "None", "tool_output_mode": "GuidedJson", "named_tool": None},
     OnlyFamilies({"qwen3": ('<function=<think>x</function>[{"name":"f","arguments":{}}]', GUIDED_UNSUPPORTED, M)})),
    ("muse_glimmer_guided_message_end_inside_native_header",
     "Muse guided output with no prefilled channel has a message-end marker inside an unfinished native invoke header. "
     "The native prefix is stripped, its remaining x is visible text, and the following guided call dispatches without leaking the header.",
     ["I3", "I5"],
     [{"kind": "text", "text": "x"}, {"kind": "tool_call", "name": "f", "arguments": {}}],
     {"starting_state": "None", "tool_output_mode": "GuidedJson", "named_tool": None},
     OnlyFamilies({"muse_glimmer": ('<atem:invoke name="x<|eom|>[{"name":"f","arguments":{}}]', GUIDED_UNSUPPORTED, M)})),
    ("deepseek_v4_guided_reasoning_opener_inside_native_body",
     "DeepSeek V4 Reasoning-prefilled guided output has a repeated reasoning opener inside a native invoke without parameters. "
     "The competing opener prevents whole-invocation suppression: the native prefix is stripped, f\">x remains reasoning, and the following guided call dispatches.",
     ["I3", "I5"],
     [{"kind": "reasoning", "text": 'f">x'}, {"kind": "tool_call", "name": "f", "arguments": {}}],
     {"starting_state": "Reasoning", "tool_output_mode": "GuidedJson", "named_tool": None},
     OnlyFamilies({"deepseek_v4": ('<｜DSML｜invoke name="f"><think>x</｜DSML｜invoke>[{"name":"f","arguments":{}}]', GUIDED_UNSUPPORTED, M)})),
    ("gemma4_guided_reasoning_opener_after_call_prefix",
     "Gemma Reasoning-prefilled guided output has a repeated thought opener immediately after bare call: with no function name or argument opener. "
     "The call: prefix remains reasoning text across the repeated thought marker; the thought closer then permits the following guided call.",
     ["I3", "I5"],
     [{"kind": "reasoning", "text": "call:x"}, {"kind": "tool_call", "name": "f", "arguments": {}}],
     {"starting_state": "Reasoning", "tool_output_mode": "GuidedJson", "named_tool": None},
     OnlyFamilies({"gemma4": ('call:<|channel>thought\nx<channel|>[{"name":"f","arguments":{}}]', GUIDED_UNSUPPORTED, M)})),
    ("gemma4_guided_response_extra_body_opener",
     "Gemma Response-prefilled guided output has an extra opening brace after a bare call header, without a native parameter key. The malformed wrapper is stripped and the following guided array still dispatches; the preceding UTF-8 prose remains visible.",
     ["I3", "I5"],
     [{"kind": "text", "text": "é "}, {"kind": "tool_call", "name": "get_weather", "arguments": {"city": "Paris"}}],
     {"starting_state": "Response", "tool_output_mode": "GuidedJson", "named_tool": None},
     OnlyFamilies({"gemma4": ('é call:get_weather{' + GUIDED_ONE_CALL, GUIDED_UNSUPPORTED, M)})),
    ("kimi_k2_optional_prefix_name_overlap",
     "Kimi K2's optional functions. prefix is also a legal complete tool name. In functions.:17 the prefix must remain the name, including when a chunk ends immediately after the colon. The call uses the declared functions. tool with empty object arguments.",
     ["I5", "I7"],
     [{"kind": "tool_call", "name": "functions.", "arguments": {}}],
     {"starting_state": "None", "tool_output_mode": "Native", "named_tool": None},
     OnlyFamilies({"kimi_k2": ("<|tool_calls_section_begin|><|tool_call_begin|>functions.:17<|tool_call_argument_begin|>{}<|tool_call_end|><|tool_calls_section_end|>", M, M)})),
]

# Group 4 (TC Malformed envelope) was a LABELLED group with zero cases, and the
# degenerate shape below had none either: no row anywhere pinned that control
# markup ALONE emits nothing. Both are native, so the input is per family.
EDGE += [
    ("tool_markup_only_emits_nothing",
     "The whole generated output is control markup and nothing else — an empty DSML calls envelope for DeepSeek V4.1, or a stray close for the other families. "
     "Everything is stripped, so the parser emits NO events at all. Until "
     "this case there was no row with an empty golden: every case asserted something was "
     "produced, so 'markup alone leaks nothing' (`I3`) was never actually pinned.",
     ["P2"],
     [],
     {"starting_state": "None", "tool_output_mode": "Native", "named_tool": None},
     {"finish_reason": "stop"},
     by_family(lambda fam: (control_tokens(fam)[2] if fam == "deepseek_v41" else "") + control_tokens(fam)[3],
               D("UNSUPPORTED", "vLLM base case does not capture a markup-only turn"),
               {"verdict": "match", "note": "control markup emits nothing"})),

    ("tool_block_never_closed_then_text",
     "A tool block opens and the model never closes it, then keeps writing prose. Nothing is "
     "emitted: the prose is BLOCK CONTENT, not the user's answer, so it drops with the "
     "unrecoverable call — the same contract `truncated_tool_eof` (5.a) pins, here with the "
     "block opening at position 0 so no reasoning survives to mask it. Worth pinning "
     "precisely because the bytes look like an answer; the envelope is what decides.",
     ["P2"],
     [],
     {"starting_state": "None", "tool_output_mode": "Native", "named_tool": None},
     {"finish_reason": "stop"},
     by_family(lambda fam: (
                   "<|start|>assistant to=functions.get_weather<|channel|>commentary "
                   "<|message|>still thinking about it"
                   if fam == "gpt_oss"
                   else f"{control_tokens(fam)[2]}still thinking about it"
               ),
               D("UNSUPPORTED", "vLLM base case does not capture an unterminated envelope"),
               {"verdict": "match", "note": "P2: unterminated envelope drops its content"})),
]

HARMONY_EDGE_INPUTS = {
    "deepseek_v41_mixed_control_text_in_string": (
        "<|start|>assistant to=functions.f<|channel|>commentary <|message|>"
        + json.dumps(
            {"x": _MIXED_CONTROL_STRINGS["gpt_oss"]}, ensure_ascii=False
        )
        + "<|call|>"
    ),
    "truncated_tool_eof": (
        "<|start|>assistant<|channel|>analysis<|message|>ok<|end|>"
        "<|start|>assistant to=functions.get_weather<|channel|>commentary "
        "<|message|>{\"city\":\"Par"
    ),
    "reason_unterminated": (
        "<|start|>assistant<|channel|>analysis<|message|>thinking but stream ends"
    ),
    "arg_marker_in_string": (
        "<|start|>assistant to=functions.run<|channel|>commentary "
        "<|message|>{\"cmd\":\"git log <|call|> --oneline\"}<|call|>"
    ),
    "orphan_close_after_prose": "I will check that. <|end|>",
    "empty_args": (
        "<|start|>assistant to=functions.get_weather<|channel|>commentary "
        "<|message|>{}<|call|>"
    ),
    "tool_no_close": (
        "<|start|>assistant to=functions.get_weather<|channel|>commentary "
        "<|message|>{\"city\":\"Paris\"}"
    ),
    "reason_markup_in_arg": (
        "<|start|>assistant to=functions.log<|channel|>commentary <|message|>"
        "{\"note\":\"<|start|>assistant<|channel|>analysis<|message|>secret<|end|>\"}"
        "<|call|>"
    ),
    "reason_markup_in_arg_with_text": (
        "<|start|>assistant<|channel|>final<|message|>Logging now: "
        "<|start|>assistant to=functions.log<|channel|>commentary <|message|>"
        "{\"note\":\"<|start|>assistant<|channel|>analysis<|message|>secret<|end|>\"}"
        "<|call|><|start|>assistant<|channel|>final<|message|> done.<|return|>"
    ),
    "two_adjacent_reason_spans": (
        "<|start|>assistant<|channel|>analysis<|message|>first<|end|>"
        "<|start|>assistant<|channel|>analysis<|message|>\nsecond<|end|>"
        "<|start|>assistant<|channel|>final<|message|>done<|return|>"
    ),
    "prefilled_reasoning_with_tool": (
        "checking weather<|end|><|start|>assistant to=functions.get_weather"
        "<|channel|>commentary <|message|>{\"city\":\"Paris\"}<|call|>"
    ),
    "prefilled_reasoning_then_text_then_tool": (
        "weighing options<|end|><|start|>assistant<|channel|>final<|message|>"
        "Here's what I found: <|return|>"
        "<|start|>assistant to=functions.get_weather<|channel|>commentary "
        "<|message|>{\"city\":\"Paris\"}<|call|>"
    ),
    "prefilled_reasoning_then_text": (
        "no tool needed<|end|><|start|>assistant<|channel|>final<|message|>"
        "The answer is 42.<|return|>"
    ),
    "prefilled_reasoning_redundant_opener": (
        "<|start|>assistant<|channel|>analysis<|message|>checking weather<|end|>"
        "<|start|>assistant to=functions.get_weather<|channel|>commentary "
        "<|message|>{\"city\":\"London\"}<|call|>"
    ),
    "prefilled_reasoning_truncated": "analyzing data",
    "prefilled_response_reasoning_markers_literal": (
        "<think>literal</think> then a call"
        "<|start|>assistant to=functions.get_weather<|channel|>commentary "
        "<|message|>{\"city\":\"Paris\"}<|call|>"
    ),
}

HARMONY_EDGE_GOLDEN_VALUES = {
    "arg_marker_in_string": "git log <|call|> --oneline",
    "deepseek_v41_mixed_control_text_in_string": _MIXED_CONTROL_STRINGS["gpt_oss"],
    "reason_markup_in_arg": "<|start|>assistant<|channel|>analysis<|message|>secret<|end|>",
    "reason_markup_in_arg_with_text": "<|start|>assistant<|channel|>analysis<|message|>secret<|end|>",
    "prefilled_response_reasoning_markers_literal": "<think>literal</think> then a call",
}

HARMONY_EDGE_GOLDENS = {
    "tool_no_close": [],
}

# Harmony messages cannot contain a second live message start nested inside an
# unfinished reasoning body. Those two cross-channel nesting probes are outside
# the valid GPT-OSS message grammar; recovery is covered by the resynchronization
# case above.
HARMONY_INAPPLICABLE_EDGE_CASES = {
    # Harmony control spellings in message prose are actual control tokens. The
    # equivalent payload-ownership case is covered inside a JSON argument string.
    "native_quoted_control_in_response",
    "native_quoted_control_in_reasoning",
    "native_single_quoted_word_control",
    "native_single_quote_contraction_response",
    "native_single_quote_contraction_reasoning",
    "native_quoted_incomplete_header",
    "native_quoted_control_then_call",
    # Harmony has one call terminator, not distinct tool-open and tool-close tokens.
    "guided_json_tool_open_before_payload",
    # These literal specimens rely on separate invoke or parameter brackets.
    # Interrupted Harmony reasoning is represented by its native variant beneath
    # the same shared heading, rather than by inventing another nested request.
    "guided_json_schema_error_not_a_call_bare_opener",
    "guided_json_gt_in_argument_bare_opener",
    "guided_json_gt_in_argument_named_bare_opener",
    "tool_in_reason",
    "tool_in_reason_with_text",
}

GPT_OSS_LEGACY_NON_HARMONY_SCENARIOS = {
    "tool_block_never_closed_then_text",
}


for _with_text in (False, True):
    _before = "I should check. "
    _after = " now answer"
    _events = [
        {"kind": "reasoning", "text": _before},
        {"kind": "tool_call", "name": "get_weather", "arguments": {"city": "Paris"}},
        {"kind": "reasoning", "text": _after},
    ]
    _input = (
        r_reason("gpt_oss", _before).removesuffix(control_tokens("gpt_oss")[1])
        + r_tool("gpt_oss", "get_weather", "city", "Paris", 0)
        + r_reason("gpt_oss", _after)
    )
    if _with_text:
        _input = r_text("gpt_oss", "Sure. ") + _input + r_text("gpt_oss", " Here you go.")
        _events = [{"kind": "text", "text": "Sure. "}, *_events,
                   {"kind": "text", "text": " Here you go."}]
    EDGE.append((
        "gpt_oss_interrupted_reasoning_then_tool" + ("_with_text" if _with_text else ""),
        "A Harmony tool message interrupts analysis without its message-end token. "
        "The tool call survives, and a later explicit analysis header resumes reasoning. "
        + ("Visible messages before and after retain their order." if _with_text else "No implicit reasoning stack is assumed."),
        ["I3", "I6"], _events,
        {"starting_state": "None", "tool_output_mode": "Native", "named_tool": None},
        OnlyFamilies({"gpt_oss": (_input, D("UNSUPPORTED", VLLM_UNCAPTURABLE["gpt_oss"]), M)}),
    ))

for _empty, _bad_body in ((True, "not-json"), (False, '{"city":')):
    _arguments = {} if _empty else {"city": "Paris"}
    _header = r_tool_arguments("gpt_oss", "bad", {}, 0).split("{", 1)[0]
    EDGE.append((
        "gpt_oss_invalid_tool_body_then_" + ("empty_call" if _empty else "valid_call"),
        "A completed native JSON tool message contains invalid JSON. Preserve the following call and the family's malformed-body fallback, including "
        + ("empty arguments." if _empty else "arguments."),
        ["P2", "I7"],
        [{"kind": "tool_call", "name": "get_weather", "arguments": _arguments}],
        {"starting_state": "None", "tool_output_mode": "Native", "named_tool": None},
        OnlyFamilies({"gpt_oss": (
            _header + _bad_body + control_tokens("gpt_oss")[3]
            + r_tool_arguments("gpt_oss", "get_weather", _arguments, 0),
            D("UNSUPPORTED", VLLM_UNCAPTURABLE["gpt_oss"]), M,
        )}),
    ))

EDGE.append((
    "gpt_oss_interrupted_reasoning_then_response",
    "An explicit native response header interrupts reasoning without its prior channel terminator. Preserve the reasoning and the following visible answer.",
    ["I1", "I3"],
    [{"kind": "reasoning", "text": "checking"}, {"kind": "text", "text": "done"}],
    {"starting_state": "None", "tool_output_mode": "Native", "named_tool": None},
    OnlyFamilies({"gpt_oss": (
        r_reason("gpt_oss", "checking").removesuffix(control_tokens("gpt_oss")[1])
        + r_text("gpt_oss", "done"), D("UNSUPPORTED", VLLM_UNCAPTURABLE["gpt_oss"]), M,
    )}),
))


EDGE += [
    ("guided_json_native_markup_only",
     "Guided decoding receives one complete native tool call instead of bare JSON. The whole turn is control markup, so it emits no events. Every stream split must match the whole-input result; consuming the invoke header before its terminator leaks the parameter body as user-visible text (`I6`).",
     ["I3", "I6"],
     [],
     {"starting_state": "None", "tool_output_mode": "GuidedJson", "named_tool": None},
     {"finish_reason": "stop"},
     guided_surroundings(
         lambda fam: r_tool(fam, "get_weather", "city", "Paris", 0),
         "native markup is stripped as one control-only turn, independent of chunking")),

    # Missing reasoning terminator CROSSED with a guided wrapper. `31-7`
    # (`guided_json_wrapped_in_tool_markup`) already pins a wrapper around the payload
    # OUTSIDE reasoning, and `41.*` pins an unterminated thought on its own; neither
    # asks what happens when a thought the model never closed runs straight into the
    # wrapper. That crossing is where both shipped families emitted the payload as
    # REASONING and dispatched nothing — the worst outcome available, because the
    # client sees a plausible answer and never learns a call was lost.
    ("guided_json_unterminated_reasoning_then_wrapped_payload",
     "A thought whose closer never arrives, running straight into native tool markup wrapping the guided payload. The model routed away from the reasoning channel and simply omitted the terminator, so the thought ends at that markup and the payload is a call. Contrast with `31-8`, where the same markup has PROSE behind it and is narration the model wrote while thinking — there the span stays open and the markup is stripped. What separates the two is whether the guided payload follows, not which marker appeared.",
     ["P2", "I6"],
     [{"kind": "reasoning", "text": "thinking"},
      {"kind": "tool_call", "name": "get_weather", "arguments": {"city": "Paris"}}],
     {"starting_state": "None", "tool_output_mode": "GuidedJson", "named_tool": None},
     {"finish_reason": "tool_calls"},
     guided_surroundings(
         lambda fam: (f"{control_tokens(fam)[0]}thinking"
                      f"{control_tokens(fam)[2]}{GUIDED_ONE_CALL}{control_tokens(fam)[3]}"),
         "unterminated thought ends at the wrapper; the wrapped payload still dispatches")),
]


# A control marker in a response that the prompt already opened is text, not structure.
# Muse exercises this through its unframed recipient headers; marker-pair families use
# their reasoning envelope. The bytes differ, but each variant tests the same request
# initialization boundary: the parser must not reopen a private channel after Response
# was selected, then must still dispatch the following guided payload.
QUOTED_BARE_HEADER = [
    ("guided_json_quoted_bare_header_in_answer", "self",
     "A `to=self` header QUOTED inside the visible answer, after the turn has already been routed to the user. The words are the model's prose and only the marker is structural, so the answer stays one run. Promoting the quote opened a real THOUGHT and split the answer in two, which reaches the client as an answer plus chain-of-thought the model never meant to expose."),
    ("guided_json_quoted_bare_tool_header_in_answer", "get_weather",
     "Muse's quoted TOOL-recipient header remains visible answer text. Promoting it deletes the `to=…` words instead of splitting the answer. Only Muse has this recipient boundary; changing a word inside another family's reasoning envelope duplicates `35-1`."),
]

# The scope siblings: turn position and open channel are independent axes, and the
# quoted-header pair above only exercises one of them (routed by a HEADER). These two
# cross the other axis — routed by a PAYLOAD, and inside an open thought — which is
# where a single boolean silently gave the wrong answer in both directions.
QUOTED_BARE_HEADER += [
    ("guided_json_quoted_bare_header_after_payload", "self",
     "A `to=self` header quoted AFTER the guided payload has already dispatched. No header routed this turn — the payload did — so a reader that tracks only 'has a header been seen' stays permissive and promotes the quote into a thought. Same corruption as the header-routed case, reached down the other axis."),
]

_RECOVERY_INSIDE_THOUGHT = (
    "guided_json_bare_tool_header_recovers_inside_a_thought",
    "A bare `to=NAME` header inside an OPEN thought, leading into the guided payload — the missing-terminator recovery boundary, with no framing on the header because the prompt consumed the turn's opening framing. The contrast with the quoted cases is the point: the same bare shape is structural here and prose there, decided by scope, not by whether a header has been seen before. A reader that closes its latch on the turn's first header demotes this one and leaks `to=NAME` into the reasoning.",
)



def _guided_response_markup_cases(recipient, after_payload=False):
    families = ("muse_glimmer",) if recipient != "self" else TEMPLATE_FAMILIES
    cases = {
        fam: (
            _guided_response_markup(fam, recipient, after_payload)[0],
            GUIDED_UNSUPPORTED,
            {"verdict": "match", "note": "Response keeps the quoted control marker out of the reasoning channel and the guided payload dispatches"},
            _guided_response_markup(fam, recipient, after_payload)[1],
        )
        for fam in families
    }
    return OnlyFamilies(cases) if recipient != "self" else cases


for _family, _scenario, _header in (
    ("deepseek_v41", "prefilled_response_guided_pending_invoke_header", '<｜DSML｜ invoke name="'),
    ("muse_glimmer", "prefilled_response_guided_closer_inside_invoke_quote", '<atem:invoke name="</atem:invoke>'),
):
    _markup, _visible = _guided_response_markup(_family, "self")
    EDGE.append((
        _scenario,
        "Response-prefilled guided output contains a quoted reasoning control marker, then an unfinished native invoke header before the JSON call. "
        "Response keeps the first marker as visible text; the pending tool header must remain owned by the scanner across chunks and never leak into that text.",
        ["I3", "I5"],
        [{"kind": "text", "text": _visible},
         {"kind": "tool_call", "name": "get_weather", "arguments": {"city": "Paris"}}],
        {"starting_state": "Response", "tool_output_mode": "GuidedJson", "named_tool": None},
        OnlyFamilies({_family: (_markup.removesuffix(GUIDED_ONE_CALL) + _header + GUIDED_ONE_CALL, GUIDED_UNSUPPORTED, M)}),
    ))


for _name, _rcpt, _desc in QUOTED_BARE_HEADER:
    EDGE.append((
        _name,
        _desc,
        ["I3"],
         ([{"kind": "tool_call", "name": "get_weather", "arguments": {"city": "Paris"}},
           {"kind": "text", "text": None}]
          if _name.endswith("after_payload") else
          [{"kind": "text", "text": None},
           {"kind": "tool_call", "name": "get_weather", "arguments": {"city": "Paris"}}]),
        {"starting_state": "Response", "tool_output_mode": "GuidedJson", "named_tool": None},
         {"finish_reason": "tool_calls"},
         _guided_response_markup_cases(_rcpt, _name.endswith("after_payload")),
    ))

EDGE.append((
    _RECOVERY_INSIDE_THOUGHT[0],
    _RECOVERY_INSIDE_THOUGHT[1],
    ["P2"],
    # `"thinking "` keeps the separator space, byte-for-byte what the native scan emits:
    # it cuts the body at the `to=`, so that space is the thought's last byte. A bare
    # header ABSORBS that space when it opens a channel, which is right there and wrong
    # here, and the one-byte difference is still a parity failure.
    [{"kind": "reasoning", "text": "thinking "},
     {"kind": "tool_call", "name": "get_weather", "arguments": {"city": "Paris"}}],
    {"starting_state": "Reasoning", "tool_output_mode": "GuidedJson", "named_tool": None},
    {"finish_reason": "tool_calls"},
    {
        fam: (
            f"thinking {control_tokens(fam)[2]}{GUIDED_ONE_CALL}{control_tokens(fam)[3]}"
            if fam not in MUSE_FAMILIES
            else f"thinking to=get_weather<|message|>{GUIDED_ONE_CALL}",
            GUIDED_UNSUPPORTED,
            {"verdict": "match",
             "note": "a native tool boundary inside prefilled reasoning ends the thought and dispatches the guided payload"},
        )
        for fam in TEMPLATE_FAMILIES
    },
))


# The corpus had no case where one control marker's terminator sits INSIDE a later
# marker, so nothing exercised "which marker owns this `>`". That gap let a stray
# prefix header borrow the `>` from a following thought opener and emit the model's
# PRIVATE reasoning as visible text. Added as a scenario, not just a unit test,
# because the property is grammar-shaped and every family has the same question.
EDGE += [
    ("guided_json_stray_prefix_before_reasoning",
     "A bare invoke HEADER with no terminator of its own sits before a reasoning span, so the only "
     "`>` in reach belongs to the thought opener. The header must NOT claim it: doing so consumed "
     "the opener, and the model's private reasoning was emitted as user-visible text (`I3`, and a "
     "privacy failure, not just a cosmetic leak). The header is incomplete markup and is stripped; "
     "the thought stays a thought.",
     ["I3"],
     [{"kind": "reasoning", "text": "secret"},
      {"kind": "tool_call", "name": "get_weather", "arguments": {"city": "Paris"}}],
     {"starting_state": "None", "tool_output_mode": "GuidedJson", "named_tool": None},
     {"finish_reason": "stop"},
     guided_surroundings(
          lambda fam: f"{guided_invoke_prefix(fam)}{r_reason(fam, 'secret')}{GUIDED_ONE_CALL}",
         "a bare invoke header before a thought must not borrow the thought's terminator")),

    ("guided_json_narrated_prefix_inside_reasoning",
     "The model NARRATES an invoke header inside its thought and never terminates it, so the only "
     "`>` in reach belongs to the thought's own closer. The header is literal text the model wrote, "
     "so it is stripped and the surrounding thought survives intact — it must not swallow the closer "
     "and it must not survive into the reasoning the user sees.",
     ["I3"],
     [{"kind": "reasoning", "text": None},
      {"kind": "tool_call", "name": "get_weather", "arguments": {"city": "Paris"}}],
     {"starting_state": "None", "tool_output_mode": "GuidedJson", "named_tool": None},
     {"finish_reason": "stop"},
     guided_surroundings(
          lambda fam: f"{r_reason(fam, 'I' + chr(39) + 'll call ' + guided_invoke_prefix(fam) + 'get_weather')}{GUIDED_ONE_CALL}",
         "a narrated invoke header inside a thought is stripped, closer and thought intact",
         fill=lambda fam: "I'll call get_weather")),
]


def _entry(spec, fam):
    """Resolve a vllm/dynamo verdict spec (single or per-family) for `fam`."""
    if isinstance(spec, dict) and set(spec) <= set(FAMILIES) and "verdict" not in spec:
        if fam == "muse_spark" and fam not in spec and "muse_glimmer" in spec:
            return spec["muse_glimmer"]
        if fam in {"deepseek_v4", "deepseek_v41", "glm47"} and fam not in spec:
            return spec["qwen3"]
        return spec[fam]
    return spec


def _init_is_request_scoped(init):
    """True when a case declares a request mode a pre-unified build cannot see."""
    init = init or {}
    return (init.get("tool_output_mode", "Native") != "Native"
            or init.get("starting_state", "None") != "None"
            or init.get("structural_tag") is not None)
def _vllm_entry(spec, fam):
    """`_entry` for the vLLM column, annotating a family no released vLLM can parse.

    An authored verdict that already carries a note said something specific about
    this case; only the shared, noteless `M` is replaced.
    """
    entry = _entry(spec, fam)
    caveat = VLLM_UNCAPTURABLE.get(fam)
    if caveat is not None and (not entry.get("note") or (fam == "muse_spark" and entry is V_MUSE)):
        return caveat
    return entry


def _vllm_entry_for_init(spec, fam, init):
    if fam == "gpt_oss":
        return D("UNSUPPORTED", "No GPT-OSS vLLM Unified capture is configured.")
    if _init_is_request_scoped(init):
        if init.get("tool_output_mode") != "Native":
            return GUIDED_UNSUPPORTED
        return D(
            "UNSUPPORTED",
            "vLLM base case does not set a starting channel state; conformance captures default generation only",
        )
    return _vllm_entry(spec, fam)


def _edge_case_family_policy(edge_case):
    return edge_case[-2] if len(edge_case) == 8 else edge_case[-1]


RECOVERY_RAW = '{"x":"unfinished'
RECOVERY_TOOLS = [
    {"name": "bad", "parameters": {"type": "object", "properties": {"value": {"type": "object"}}}},
    {"name": "echo", "parameters": {"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"]}},
]


def malformed_json_recovery(fam):
    following = [{"kind": "tool_call", "name": "echo", "arguments": {"value": value}} for value in ("é", "Café")]
    if fam == "kimi_k3":
        inp = k3_tools(k3_call("bad", 1, k3_json(RECOVERY_RAW)) + "".join(
            k3_call("echo", index, k3_argument("value", "string", value))
            for index, value in ((2, "é"), (3, "Café"))))
        golden = following
    elif fam == "kimi_k2":
        inp = ("<|tool_calls_section_begin|><|tool_call_begin|>functions.bad:0"
               f"<|tool_call_argument_begin|>{RECOVERY_RAW}<|tool_call_end|><|tool_calls_section_end|>")
        golden = [{"kind": "tool_call", "name": "bad", "arguments": {}}] + following
    elif fam == "gemma4":
        inp = '<|tool_call>call:bad{value:{x:<|"|>unfinished}}<tool_call|>'
        golden = following
    else:
        inp = r_tool(fam, "bad", "value", RECOVERY_RAW, 0)
        if fam in ("deepseek_v4", "deepseek_v41"):
            inp = inp.replace('string="true"', 'string="false"')
        golden = ([{"kind": "tool_call", "name": "bad", "arguments": {"value": RECOVERY_RAW}}]
                  if fam != "deepseek_v41" else []) + following
    if fam != "kimi_k3":
        inp += "".join(r_tool(fam, "echo", "value", value, index)
                       for index, value in ((1, "é"), (2, "Café")))
    return inp, golden


EDGE.append((
    "malformed_json_then_two_valid_calls",
    "An unfinished JSON string with explicit call closers precedes echo(value=é) and echo(value=Café). Deliver the entire native input in one push, then finish: marker-aligned delivery masks the Kimi K3 0.7.13 lost-call defect. Preserve family malformed-value fallbacks and both later calls.",
    ["I2", "I7", "P2"], [],
    {"starting_state": "None", "tool_output_mode": "Native", "named_tool": None},
    {"finish_reason": "stop", "single_chunk": True},
    {fam: (malformed_json_recovery(fam)[0], D("UNSUPPORTED", "No peer capture recorded."),
           D("LOSS" if fam == "gemma4" else "ERROR", "Known malformed-call recovery defect; later calls must survive.")
           if fam in ("gemma4", "deepseek_v41") else M, malformed_json_recovery(fam)[1])
     for fam in TEMPLATE_FAMILIES},
    RECOVERY_TOOLS,
))


DEEPSEEK_V41_REDUNDANT_SCENARIOS = {
    "prefilled_reasoning_with_tool": "reason_then_tool",
    "prefilled_reasoning_then_text_then_tool": "interstitial_text",
    "prefilled_reasoning_then_text": "reason_then_content",
}

DEEPSEEK_V41_SCENARIOS = {
    spec[0]
    for spec in (*CLEAN, *EDGE)
    if not isinstance(_edge_case_family_policy(spec), OnlyFamilies)
    or "deepseek_v41" in _edge_case_family_policy(spec)
} - set(DEEPSEEK_V41_REDUNDANT_SCENARIOS)


def _deepseek_v41_input(segments):
    # V4.1 groups adjacent invokes inside one calls envelope and starts initial
    # reasoning in the prompt; neither changes the scenario's events or tools.
    text = ""
    reasoning_open = False
    starting_state = "None"
    emitted_output = False
    tool_block_open = False
    for segment in segments:
        if segment[0] == "reason":
            if tool_block_open:
                text += "</｜DSML｜ calls>"
                tool_block_open = False
            if not reasoning_open:
                if not emitted_output:
                    starting_state = "Reasoning"
                else:
                    text += "<think>"
                reasoning_open = True
            text += segment[1]
            emitted_output = True
        elif segment[0] == "text":
            if tool_block_open:
                text += "</｜DSML｜ calls>"
                tool_block_open = False
            if reasoning_open:
                text += "</think>"
                reasoning_open = False
            text += segment[1]
            emitted_output = True
        else:
            tool, arguments = tool_segment(segment)
            if reasoning_open:
                text += "</think>"
                reasoning_open = False
            if not tool_block_open:
                text += "<｜DSML｜ calls>"
                tool_block_open = True
            text += r_tool_arguments("deepseek_v41", tool, arguments, 0).removeprefix("<｜DSML｜ calls>").removesuffix("</｜DSML｜ calls>")
            emitted_output = True
    if reasoning_open:
        text += "</think>"
    if tool_block_open:
        text += "</｜DSML｜ calls>"
    return text, starting_state


PARALLEL_BOUNDARY_CASES = json.loads(
    Path(__file__).with_name("unified_parallel_boundary_cases.json").read_text()
)
UNIFIED_PARALLEL_BOUNDARY_FAMILIES = frozenset(
    family for cases in PARALLEL_BOUNDARY_CASES.values() for family in cases
)


def _parallel_boundary_specs():
    specs = []
    descriptions = {
        "adjacent_calls_partial_header": "Adjacent calls: a delta closes one call and ends inside the next call header",
        "three_calls_partial_header": "Three adjacent calls: retain the middle call after a partial next header",
    }
    for scenario, families in PARALLEL_BOUNDARY_CASES.items():
        per_family, schedules, schemas = {}, {}, {}
        for family, source in families.items():
            chunks = source["input_chunks"]
            per_family[family] = ("".join(chunks), VLLM_UNCAPTURABLE.get(family, M), M, source["golden"])
            schedules[family] = chunks
            schemas[family] = source["tools"]
        specs.append((scenario, descriptions[scenario], ["I2", "I3", "I5", "I7"], [],
                      {"starting_state": "None", "tool_output_mode": "Native", "named_tool": None},
                      {"finish_reason": "tool_calls", "input_chunks": schedules},
                      OnlyFamilies(per_family), schemas))
    return specs


EDGE += _parallel_boundary_specs()


# Native grammars encode the oracle type; Qwen and GLM consult the request schema.
_NULL_TEXT_INPUTS = {
    "qwen3": "<tool_call><function=get_weather><parameter=city>null</parameter></function></tool_call>",
    "glm47": "<tool_call>get_weather<arg_key>city</arg_key><arg_value>null</arg_value></tool_call>",
}

EDGE += [
    (
        scenario,
        null_description(label, detail),
        ["I7"],
        [{"kind": "tool_call", "name": "get_weather", "arguments": {"city": value}}],
        {"starting_state": "None", "tool_output_mode": "Native", "named_tool": None},
        {"finish_reason": "stop"},
        OnlyFamilies({
            family: (
                _NULL_TEXT_INPUTS[family] if family in _NULL_TEXT_INPUTS
                else r_tool(family, "get_weather", "city", value, 0),
                VLLM_UNCAPTURABLE.get(family, M), M,
            )
            for family in TEMPLATE_FAMILIES
        }),
        {family: [{"name": "get_weather", "parameters": {
            "type": "object", "properties": {"city": json.loads(json.dumps(schema))},
        }}] for family in TEMPLATE_FAMILIES},
    )
    for scenario, label, schema, value, detail in NULL_VARIANTS
]

# Keep references unresolved so schema-driven grammars must consult the tool
# parameters root; explicitly typed grammars preserve the same authored values.
EDGE += [
    (
        scenario,
        null_description(label, detail),
        ["I7"],
        [{"kind": "tool_call", "name": "get_weather", "arguments": {"city": value}}],
        {"starting_state": "None", "tool_output_mode": "Native", "named_tool": None},
        {"finish_reason": "stop"},
        OnlyFamilies({
            family: (
                _NULL_TEXT_INPUTS[family] if family in _NULL_TEXT_INPUTS
                else r_tool(family, "get_weather", "city", value, 0),
                D("UNSUPPORTED", f"No peer capture is recorded for this {family} reference-schema probe."), M,
            )
            for family in TEMPLATE_FAMILIES
        }),
        {family: [{"name": "get_weather", "parameters": {
            "type": "object", "$defs": {"City": schema},
            "properties": {"city": {"$ref": "#/$defs/City"}},
        }}] for family in TEMPLATE_FAMILIES},
    )
    for scenario, label, schema, value, detail in (
        ("arg_json_null_ref", "7-4.ref", {"type": ["string", "null"]}, None,
         '`city` uses a local $ref to the tool parameters root; resolving its nullable definition turns bare null text into JSON null.'),
        ("arg_string_null_ref", "7-5.ref", {"type": "string"}, "null",
         'A local $ref resolves to a string-only definition; bare null text remains the string "null".'),
    )
]

EDGE.append((
    "arg_null_mixed_labels",
    'set_labels has nullable label (anyOf), nullable note (type array), and non-nullable literal (string). Schema-directed values use bare null text; typed grammars use native null and string values. Expect {"label": null, "note": null, "literal": "null"}. This single capture is referenced by both 7-4 and 7-5.',
    ["I7"],
    [{"kind": "tool_call", "name": "set_labels", "arguments": MIXED_LABELS_ARGS}],
    {"starting_state": "None", "tool_output_mode": "Native", "named_tool": None},
    {"finish_reason": "stop"},
    OnlyFamilies({family: (
        ("<tool_call>set_labels"
        "<arg_key>label</arg_key><arg_value>null</arg_value>"
        "<arg_key>note</arg_key><arg_value>null</arg_value>"
        "<arg_key>literal</arg_key><arg_value>null</arg_value></tool_call>") if family == "glm47"
        else r_tool_arguments(family, "set_labels", MIXED_LABELS_ARGS, 0), M, M,
    ) for family in FAMILIES if family in MIXED_CASE_FAMILIES["7-4.mixed_labels"]}),
    {family: [{"name": "set_labels", "parameters": MIXED_LABELS_SCHEMA}]
     for family in FAMILIES if family in MIXED_CASE_FAMILIES["7-4.mixed_labels"]},
))

# Keep historical scenario IDs and raw spellings; applicability is shared.
# GLM/Qwen consult schemas; explicitly typed grammars test value preservation.
for scenario, description, parameters, raw_arguments, arguments in (
    (
        "unused_reference_graph_parameter_types",
        'An unused compact reference graph precedes ordinary integer, boolean, and string parameters; its traversal must not consume their coercion budget.',
        {"type": "object", "$defs": {
            f"N{i}": ({"type": "string"} if i == 6 else {"allOf": [
                {"$ref": f"#/$defs/N{i + 1}"} for _ in range(4)
            ]}) for i in range(7)
        }, "properties": {
            "a": {"$ref": "#/$defs/N0"}, "count": {"type": "integer"},
            "flag": {"type": "boolean"}, "text": {"type": "string"},
        }},
        {"count": "42", "flag": "false", "text": "null"},
        {"count": 42, "flag": False, "text": "null"},
    ),
    (
        "nullable_reference_alias_literals",
        'Nullable reference aliases retain string const and enum restrictions; bare null is the schema-valid string "null".',
        {"type": "object", "$defs": {
            "ConstAlias": {"$ref": "#/$defs/ConstText"},
            "EnumAlias": {"$ref": "#/$defs/EnumText"},
            "ConstText": {"type": "string", "const": "null"},
            "EnumText": {"type": "string", "enum": ["null"]},
        }, "properties": {
            "const_text": {"$ref": "#/$defs/ConstAlias", "nullable": True},
            "enum_text": {"$ref": "#/$defs/EnumAlias", "nullable": True},
        }},
        {"const_text": "null", "enum_text": "null"},
        {"const_text": "null", "enum_text": "null"},
    ),
    (
        "local_schema_id_preserves_type",
        'A directly declared integer remains typed when a local $id disables reference resolution.',
        {"type": "object", "properties": {
            "count": {"type": "integer", "$id": "https://example.com/count"},
        }},
        {"count": "42"},
        {"count": 42},
    ),
    (
        "glm_ref_object",
        'PR #271 compatibility control: a local object reference keeps JSON object text as an object.',
        {"type": "object", "$defs": {"Payload": {
            "type": "object", "properties": {"x": {"type": "integer"}},
        }}, "properties": {"payload": {"$ref": "#/$defs/Payload"}}},
        {"payload": '{"x":1}'},
        {"payload": {"x": 1}},
    ),
    (
        "glm_ref_encoded_targets",
        'PR #271: URI percent decoding precedes JSON Pointer unescaping. Spaces, UTF-8, literal plus, slash, and tilde in definition names resolve to integer types.',
        {"type": "object", "$defs": {
            "postal code": {"type": "integer"}, "café+": {"type": "integer"},
            "a/b~c": {"type": "integer"},
        }, "properties": {
            "space": {"$ref": "#/$defs/postal%20code"},
            "utf8_plus": {"$ref": "#/$defs/caf%c3%a9+"},
            "pointer": {"$ref": "#/$defs/a%7E1b%7E0c"},
        }},
        {"space": "42", "utf8_plus": "42", "pointer": "42"},
        {"space": 42, "utf8_plus": 42, "pointer": 42},
    ),
    (
        "glm_ref_json_looking_strings",
        'PR #271: referenced strings preserve JSON-looking object, array, and quoted text verbatim, including literal quotes; an inline string is the control.',
        {"type": "object", "$defs": {"Text": {"type": "string"}}, "properties": {
            "object_text": {"$ref": "#/$defs/Text"},
            "array_text": {"$ref": "#/$defs/Text"},
            "quoted_text": {"$ref": "#/$defs/Text"},
            "inline_text": {"type": "string"},
        }},
        {"object_text": '{"x":1}', "array_text": '[1,2]',
         "quoted_text": '"hello"', "inline_text": '{"x":1}'},
        {"object_text": '{"x":1}', "array_text": '[1,2]',
         "quoted_text": '"hello"', "inline_text": '{"x":1}'},
    ),
    (
        "glm_ref_scalar_types",
        'PR #271: referenced integer, number, and boolean values keep their JSON types. A sibling integer type narrows a referenced string-or-integer union.',
        {"type": "object", "$defs": {
            "Integer": {"type": "integer"}, "Number": {"type": "number"},
            "Boolean": {"type": "boolean"}, "Scalar": {"type": ["string", "integer"]},
        }, "properties": {
            "count": {"$ref": "#/$defs/Integer"}, "ratio": {"$ref": "#/$defs/Number"},
            "flag": {"$ref": "#/$defs/Boolean"},
            "narrowed": {"$ref": "#/$defs/Scalar", "type": "integer"},
        }},
        {"count": "42", "ratio": "3.5", "flag": "true", "narrowed": "42"},
        {"count": 42, "ratio": 3.5, "flag": True, "narrowed": 42},
    ),
):
    EDGE.append((
        scenario, description + " GLM and Qwen exercise schema-driven typing; explicitly typed grammars preserve native values with unresolved refs in the request, without proving reference resolution.", ["I7"],
        [{"kind": "tool_call", "name": "capture_payload", "arguments": arguments}],
        {"starting_state": "None", "tool_output_mode": "Native", "named_tool": None},
        {"finish_reason": "stop"},
        OnlyFamilies({family: (
            r_tool_arguments(family, "capture_payload", arguments, 0, raw_arguments),
            D("UNSUPPORTED", "No peer capture is recorded for this reference-schema probe."), M,
        ) for family in TEMPLATE_FAMILIES}),
        {family: [{"name": "capture_payload", "parameters": parameters}] for family in TEMPLATE_FAMILIES},
    ))
for scenario, value, value_schema in (
    ("arg_object_quoted_parameter_close", {"s": "</atem:parameter>"},
     {"type": "object", "properties": {"s": {"type": "string"}},
      "required": ["s"], "additionalProperties": False}),
    ("arg_array_quoted_parameter_close", ['é \\" </atem:parameter>'],
     {"type": "array", "items": {"type": "string"}}),
):
    arguments = {"value": value}
    EDGE.append((
        scenario,
        "JSON container strings preserve a literal ATEM parameter closer. All native families carry the same data using their own argument encoding; Muse must not interpret the quoted closer as framing.",
        ["I7"],
        [{"kind": "tool_call", "name": "record", "arguments": arguments}],
        {"starting_state": "None", "tool_output_mode": "Native", "named_tool": None},
        {"finish_reason": "stop"},
        OnlyFamilies({family: (
            r_tool_arguments(family, "record", arguments, 0),
            D("UNSUPPORTED", "No peer capture recorded for this JSON delimiter probe."), M,
        ) for family in TEMPLATE_FAMILIES}),
        {family: [{"name": "record", "parameters": {
            "type": "object", "properties": {"value": value_schema},
            "required": ["value"], "additionalProperties": False,
        }}] for family in TEMPLATE_FAMILIES},
    ))

_DS41_JSON_BODY_ARGS = {
    "value": ' café 🐈 </｜DSML｜ invoke> </｜DSML｜ calls> &amp; "x"\n',
    "nested": {"values": [True, 42, -1250]},
}
_DS41_JSON_BODY = json.dumps(_DS41_JSON_BODY_ARGS, ensure_ascii=False)
EDGE.append((
    "deepseek_v41_json_invocation_body",
    "PR #250: DeepSeek V4.1 accepts a JSON object as the invocation body and preserves marker-looking text inside its string values.",
    ["I7"],
    [
        {"kind": "tool_call", "name": "inspect", "arguments": _DS41_JSON_BODY_ARGS},
        {"kind": "tool_call", "name": "done", "arguments": {}},
    ],
    {"starting_state": "None", "tool_output_mode": "Native", "named_tool": None},
    {"finish_reason": "stop"},
    OnlyFamilies({"deepseek_v41": (
        '<｜DSML｜ calls>\n'
        '<｜DSML｜ invoke name="inspect">\n'
        + _DS41_JSON_BODY
        + '\n</｜DSML｜ invoke>\n'
        '<｜DSML｜ invoke name="done">{}</｜DSML｜ invoke>\n'
        '</｜DSML｜ calls>', M, M,
    )}),
    {"deepseek_v41": [
        {"name": "inspect", "parameters": {
            "type": "object",
            "properties": {
                "value": {"type": "string"},
                "nested": {"type": "object"},
            },
            "required": ["value", "nested"],
        }},
        {"name": "done", "parameters": {"type": "object"}},
    ]},
))

_GLM_REFERENCE_TOOLS = [{"name": "capture_payload", "parameters": {
    "type": "object",
    "$defs": {
        "Text": {"type": "string"},
        "TextAlias": {"$ref": "#/$defs/Text"},
        "Count": {"type": "integer"},
    },
    "properties": {
        "payload": {"$ref": "#/$defs/TextAlias", "allOf": [{"type": "string"}]},
        "count": {"$ref": "#/$defs/Count", "minimum": 1},
    },
    "required": ["payload", "count"],
}}]
EDGE.append((
    "glm47_reference_type_intersection",
    "PR #271: GLM resolves local reference chains before coercing string and integer arguments. This case does not distinguish sibling type intersections.",
    ["I7"],
    [{"kind": "tool_call", "name": "capture_payload", "arguments": {
        "payload": '{"x":1}', "count": 42,
    }}],
    {"starting_state": "None", "tool_output_mode": "Native", "named_tool": None},
    {"finish_reason": "stop"},
    OnlyFamilies({"glm47": (
        '<tool_call>capture_payload'
        '<arg_key>payload</arg_key><arg_value>{"x":1}</arg_value>'
        '<arg_key>count</arg_key><arg_value>42</arg_value></tool_call>', M,
        M,
    )}),
    {"glm47": _GLM_REFERENCE_TOOLS},
))


EDGE += [
    (scenario, numeric_description(label) + f" Input {raw}; expected {expected}.",
     ["I7"], [{"kind": "tool_call", "name": "get_weather", "arguments": arguments_json(expected)}],
     {"starting_state": "None", "tool_output_mode": "Native", "named_tool": None},
     {"finish_reason": "stop"},
     OnlyFamilies({family: (r_tool(family, "get_weather", "value", NumericLiteral(raw), 0),
                             VLLM_UNCAPTURABLE.get(family, M), M,
                             [{"kind": "tool_call", "name": "get_weather",
                                "arguments": arguments_json(unified_numeric_expected(family, label, raw, expected))}])
                   for family in FAMILIES if applicable(family, label)}),
     {family: [{"name": "get_weather", "parameters": {
         "type": "object", "properties": {"value": numeric_schema(family, label, schema)}}}]
      for family in FAMILIES if applicable(family, label)})
    for scenario, label, schema, raw, expected in NUMERIC_VARIANTS
]


_NESTED_NUMERIC_ARGUMENTS = {
    "nested": {
        "fraction": NumericLiteral("9007199254740992.5"),
        "tiny": NumericLiteral("1e-400"),
    },
    "values": [NumericLiteral("9007199254740993"), NumericLiteral("-9007199254740993")],
}
_NESTED_NUMERIC_GOLDEN = (
    '{"nested":{"fraction":9007199254740992.5,"tiny":1e-400},'
    '"values":[9007199254740993,-9007199254740993]}'
)
_NESTED_NUMERIC_SCHEMA = [{"name": "inspect", "parameters": {
    "type": "object", "properties": {
        "nested": {"type": "object", "properties": {
            "fraction": {"type": "number"}, "tiny": {"type": "number"},
        }},
        "values": {"type": "array", "items": {"type": "integer"}},
    },
}}]
EDGE.append((
    "arg_numeric_nested_exact",
    "Nested objects and arrays preserve exact large integers, long decimals, and underflow-sized numbers.",
    ["I7"],
    [{"kind": "tool_call", "name": "inspect", "arguments": _NESTED_NUMERIC_GOLDEN}],
    {"starting_state": "None", "tool_output_mode": "Native", "named_tool": None},
    {"finish_reason": "stop"},
    OnlyFamilies({family: (
        r_tool_arguments(family, "inspect", _NESTED_NUMERIC_ARGUMENTS, 0),
        VLLM_UNCAPTURABLE.get(family, M), M,
        [{"kind": "tool_call", "name": "inspect", "arguments": _NESTED_NUMERIC_GOLDEN}],
    ) for family in FAMILIES}),
    {family: _NESTED_NUMERIC_SCHEMA for family in FAMILIES},
))


_NATIVE_QUOTED_CONTROL = {
    "gpt_oss": r_tool("gpt_oss", "get_weather", "city", "Paris", 0).split("{", 1)[0],
    "deepseek_v4": "<｜DSML｜tool_calls>",
    "deepseek_v41": "<｜DSML｜ calls>",
    "gemma4": "<|tool_call>",
    "glm47": "<tool_call>",
    "kimi_k2": "<|tool_calls_section_begin|>",
    "kimi_k3": "<|open|>call",
    "muse_glimmer": "<|start|>assistant to=get_weather<|message|>",
    "beam": "<|start|>assistant to=get_weather<|channel|>",
    "qwen3": "<tool_call>",
}

_NATIVE_QUOTED_CONTROL["muse_spark"] = _NATIVE_QUOTED_CONTROL["muse_glimmer"]


def _native_quoted_control_cases():
    for scenario, state, prefix_kind, suffix_call in (
        ("native_quoted_control_in_response", "Response", "text", False),
        ("native_quoted_control_in_reasoning", "Reasoning", "reasoning", False),
        ("native_single_quoted_word_control", "Response", "text", False),
        ("native_single_quote_contraction_response", "Response", "text", False),
        ("native_single_quote_contraction_reasoning", "Reasoning", "reasoning", False),
        ("native_quoted_incomplete_header", "Response", "text", False),
        ("native_quoted_control_then_call", "Response", "text", True),
        ("native_unmatched_quote_then_call", "Response", "text", True),
    ):
        families = {}
        for family, marker in _NATIVE_QUOTED_CONTROL.items():
            if scenario == "native_single_quoted_word_control":
                prose = f"The literal 'example {marker} marker' is part of the explanation."
            elif scenario.startswith("native_single_quote_contraction_"):
                prose = f"The literal 'doesn't {marker} marker' stays quoted."
            elif scenario == "native_quoted_incomplete_header":
                header = r_tool(family, "get_weather", "city", "Paris", 0).split("Paris", 1)[0]
                prose = f"The literal `{header}` header is part of the explanation."
            elif scenario == "native_unmatched_quote_then_call":
                prose = 'He said "maybe'
            else:
                prose = f'The literal "{marker}" marker is part of the explanation.'
            golden = [{"kind": prefix_kind, "text": prose}]
            raw = prose
            if suffix_call:
                raw += r_tool(family, "get_weather", "city", "Paris", 0)
                golden.append({"kind": "tool_call", "name": "get_weather", "arguments": {"city": "Paris"}})
            if family == "beam" and scenario != "native_unmatched_quote_then_call":
                # Beam's markers are special tokens the engine emits as such, so a
                # quotation is no protection: the quoted header is a header. The
                # prose before the quote is visible; the quoted (incomplete or
                # unterminated) header and whatever follows it on the same line are
                # framing debris, and a real call after it still dispatches. The
                # builder's reference reads it the same way.
                golden = [{"kind": prefix_kind, "text": prose.split(marker, 1)[0] if marker in prose else prose.split("`", 1)[0] + "`"}]
                if suffix_call:
                    golden.append({"kind": "tool_call", "name": "get_weather", "arguments": {"city": "Paris"}})
            families[family] = (raw, M, M, golden)
        yield (
            scenario,
            "Balanced quoted native controls remain prose; an unmatched quotation must not hide an actual call.",
            ["I7", "P2"], [],
            {"starting_state": state, "tool_output_mode": "Native", "named_tool": None},
            OnlyFamilies(families),
        )


EDGE += list(_native_quoted_control_cases())


EDGE.append((
    "guided_response_rejected_header_quote_ownership",
    "DeepSeek V4 prefilled guided Response rejects an incomplete invoke header, then strips its parameter markup; the rejected attribute quote cannot become a prose quote and protect that markup.",
    ["P2"],
    [{"kind": "text", "text": 'f"x'}, {"kind": "tool_call", "name": "f", "arguments": {"x": "ok"}}],
    {"starting_state": "Response", "tool_output_mode": "GuidedJson", "named_tool": None},
    {"finish_reason": "stop"},
    OnlyFamilies({"deepseek_v4": (
        '<｜DSML｜invoke name="f"<｜DSML｜parameter name="x" string="true">x[{"name":"f","arguments":{"x":"ok"}}]', M, M,
    )}),
    {"deepseek_v4": [{"name": "f", "parameters": {"type": "object", "properties": {"x": {"type": "string"}}}}]},
))


for _named in (False, True):
    _families = {}
    for _family in FAMILIES:
        _closer = control_tokens(_family)[1]
        _reasoning = f"The literal 'doesn't {_closer} marker' stays quoted."
        _payload = json.dumps({"city": "Paris"} if _named else {"name": "get_weather", "arguments": {"city": "Paris"}})
        _families[_family] = (
            _reasoning + _closer + _payload, M, M,
            [{"kind": "reasoning", "text": _reasoning},
             {"kind": "tool_call", "name": "get_weather", "arguments": {"city": "Paris"}}],
        )
    EDGE.append((
        "guided_quoted_reasoning_closer_" + ("named" if _named else "required"),
        "Prefilled guided reasoning quotes its closer inside a contraction-bearing single quotation; only the later unquoted closer ends reasoning and dispatches the payload.",
        ["I7"], [],
        {"starting_state": "Reasoning", "tool_output_mode": "GuidedJson", "named_tool": "get_weather" if _named else None},
        OnlyFamilies(_families),
    ))

for _named in (False, True):
    _families = {}
    for _family, _marker in _NATIVE_QUOTED_CONTROL.items():
        _prose = f'The literal "{_marker} {{ example }}" stays visible. '
        _payload = json.dumps({"city": "Paris"} if _named else {"name": "get_weather", "arguments": {"city": "Paris"}})
        _families[_family] = (
            _prose + _payload, M, M,
            [{"kind": "text", "text": _prose},
             {"kind": "tool_call", "name": "get_weather", "arguments": {"city": "Paris"}}],
        )
    EDGE.append((
        "guided_response_quoted_control_braces_" + ("named" if _named else "required"),
        "Prefilled guided Response quotes a native opener and braces as visible prose before the real payload; a brace inside the balanced quotation cannot start payload ownership.",
        ["I7"], [],
        {"starting_state": "Response", "tool_output_mode": "GuidedJson", "named_tool": "get_weather" if _named else None},
        OnlyFamilies(_families),
    ))


for _suffix, _marker in (("eom", "<|eom|>"), ("eot", "<|eot|>"), ("start", "<|start|>")):
    _value = f'a"{_marker}"b'
    EDGE.append((
        f"muse_quoted_reserved_{_suffix}_argument",
        "A balanced quotation in a Muse parameter value names a reserved channel token; preserve that literal value and the real call.",
        ["I7"],
        [{"kind": "tool_call", "name": "get_weather", "arguments": {"city": _value}}],
        {"starting_state": "Response", "tool_output_mode": "Native", "named_tool": None},
        OnlyFamilies({"muse_glimmer": (r_tool("muse_glimmer", "get_weather", "city", _value, 0), V_MUSE, M)}),
    ))



def _beam_header(name="get_weather"):
    return f"<|start|>assistant to={name}<|channel|>analysis <|constrain|>xml<|message|>"


def _beam_param(key, value):
    return f'<rfl:parameter name="{key}">\n{value}\n</rfl:parameter>'


def _beam_edge_entries():
    """Beam-native wire and independent golden for each shared scenario.

    Beam frames every message (`<|start|>assistant<|channel|>NAME<|message|>…`), so the
    shared scenarios keep their meaning but not their bytes: reasoning closes only on
    `<|end|>`, visible text closes on `<|return|>`/`<|end|>`, a call dispatches only once
    its `<|call|>`/`<|end|>` arrives, and a tool header cuts an open analysis body
    without consuming it. Where that grammar changes the expected events, the entry
    carries its own golden (list) instead of the shared one, and says why.
    """
    unavailable = VLLM_UNCAPTURABLE["beam"]
    paris = _beam_header() + _beam_param("city", "Paris") + "<|call|>"
    spans = r_reason("beam", "first") + r_reason("beam", "\nsecond")

    def note(text):
        return {"verdict": "match", "note": text}

    return {
        "truncated_tool_eof": (
            r_reason("beam", "ok") + _beam_header() + '<rfl:parameter name="city">\nPar',
            unavailable,
            note("P2: no `<|call|>`/`<|end|>` arrived, so the call never dispatches; the closed reasoning survives and nothing leaks"),
        ),
        "reason_unterminated": (
            "<|start|>assistant<|channel|>analysis<|message|>thinking but stream ends",
            unavailable,
            note("the open analysis body is promoted as reasoning at finish"),
        ),
        "arg_marker_in_string": (
            r_tool("beam", "run", "cmd", "git log <|call|> --oneline", 0),
            unavailable,
            note("only `</rfl:parameter>` ends a value, so an inner `<|call|>` is argument data"),
            "git log <|call|> --oneline",
        ),
        "orphan_close_after_prose": (
            r_text("beam", "I will check that. <|call|>"),
            unavailable,
            note("`<|call|>` is one of the final answer's closers in the serving grammar, so it ends the message; the trailing `<|return|>` is debris between messages"),
            [{"kind": "text", "text": "I will check that. "}],
        ),
        "empty_args": (
            _beam_header() + "<|call|>",
            unavailable,
            M,
        ),
        "tool_no_close": (
            _beam_header() + _beam_param("city", "Paris"),
            unavailable,
            note("the serving grammar lets EOS stand for the call closer: a body of complete parameters at the end of the stream is the call (a header alone, or an open parameter, still is not)"),
            [{"kind": "tool_call", "name": "get_weather", "arguments": {"city": "Paris"}}],
        ),
        "reason_markup_in_arg": (
            r_tool("beam", "log", "note", r_reason("beam", "reconsider"), 0),
            unavailable,
            note("an analysis channel written inside a parameter value is argument data until `</rfl:parameter>`"),
            r_reason("beam", "reconsider"),
        ),
        "tool_in_reason": (
            "<|start|>assistant<|channel|>analysis<|message|>I should check. " + paris
            + "<|start|>assistant<|channel|>analysis<|message|> now answer<|end|>",
            unavailable,
            note("the tool header cuts the open analysis body, the call dispatches, and a new analysis message resumes the thought"),
        ),
        "reason_markup_in_arg_with_text": (
            r_text("beam", "Logging now: ")
            + r_tool("beam", "log", "note", r_reason("beam", "reconsider"), 0)
            + r_text("beam", " done."),
            unavailable,
            note("framed `final` messages surround a call whose value owns the embedded analysis markup"),
            r_reason("beam", "reconsider"),
        ),
        "tool_in_reason_with_text": (
            r_text("beam", "Sure. ")
            + "<|start|>assistant<|channel|>analysis<|message|>I should check. " + paris
            + "<|start|>assistant<|channel|>analysis<|message|> now answer<|end|>"
            + r_text("beam", " Here you go."),
            unavailable,
            note("visible `final` messages stay visible on both sides of the nested thought and call"),
        ),
        "two_adjacent_reason_spans": (
            spans + r_text("beam", "done"),
            unavailable,
            note("adjacent analysis messages coalesce with the authored separator"),
        ),
        "prefilled_reasoning_with_tool": (
            "checking weather<|end|>" + paris,
            unavailable,
            note("starting_state=Reasoning opens the scanner inside the analysis body, closed by `<|end|>`"),
        ),
        "prefilled_reasoning_then_text_then_tool": (
            "weighing options<|end|>" + r_text("beam", "Here's what I found: ") + paris,
            unavailable,
            note("reasoning -> final message -> call, all ordered in one prefilled stream"),
        ),
        "prefilled_reasoning_then_text": (
            "no tool needed<|end|>" + r_text("beam", "The answer is 42."),
            unavailable,
            note("closing a prefilled analysis body returns the stream to visible content"),
        ),
        "prefilled_reasoning_redundant_opener": (
            r_reason("beam", "checking weather")
            + _beam_header() + _beam_param("city", "London") + "<|call|>",
            unavailable,
            note("the echoed analysis header opens a zero-length first body, so nothing leaks and nothing is added"),
        ),
        "prefilled_reasoning_truncated": (
            "analyzing data<|end|>" + _beam_header() + '<rfl:parameter name="city">\nPar',
            unavailable,
            note("P2: the unterminated call is dropped, the prefilled thought survives"),
        ),
        "guided_json_native_parameter_body_inside_reasoning": (
            native_body_in_guided_reasoning("beam", "é🙂<par{value}"),
            unavailable,
            note("Beam has no bare native invocation: without its routed header a "
                 "`<rfl:parameter>` tag inside the analysis is ordinary text and stays "
                 "in the reasoning; the guided call still dispatches"),
            [{"kind": "reasoning",
              "text": "before <rfl:parameter name=\"x\">\né🙂<par{value}\n</rfl:parameter> after"},
             {"kind": "tool_call", "name": "get_weather", "arguments": {"city": "Paris"}}],
        ),
        "guided_json_reasoning_markers_inside_native_parameter": (
            native_body_in_guided_reasoning("beam", r_reason("beam", "quoted")),
            unavailable,
            note("Beam's reasoning markers are special tokens: the quoted `<|end|>` closes "
                 "the analysis wherever it appears (class C), the rest of the parameter "
                 "tag is visible text, and the guided call still dispatches"),
            [{"kind": "reasoning", "text": "before <rfl:parameter name=\"x\">\nquoted"},
             {"kind": "text", "text": "\n</rfl:parameter> after"},
             {"kind": "tool_call", "name": "get_weather", "arguments": {"city": "Paris"}}],
        ),
        "prefilled_response_reasoning_markers_literal": (
            r_reason("beam", "literal") + " then a call"
            + _beam_header() + _beam_param("city", "Paris") + "<|call|>",
            unavailable,
            note("Response means the stream has no reasoning channel: the framed analysis "
                 "message is consumed as control framing and its body is visible text, "
                 "joined with the prose that follows; the call still dispatches"),
            [{"kind": "text", "text": "literal then a call"},
             {"kind": "tool_call", "name": "get_weather", "arguments": {"city": "Paris"}}],
        ),
    }


def _wire_beam_into_edge_cases():
    entries = _beam_edge_entries()
    # Scenarios whose shared per-family rendering (`guided_surroundings`) already names
    # Beam, but whose grammar-level outcome Beam authors itself.
    replaces = {
        "guided_json_native_parameter_body_inside_reasoning",
        "guided_json_reasoning_markers_inside_native_parameter",
    }
    seen = set()
    for edge_case in EDGE:
        name = edge_case[0]
        if name not in entries:
            continue
        per_fam = _edge_case_family_policy(edge_case)
        assert name in replaces or "beam" not in per_fam, f"{name}: Beam is already authored"
        per_fam["beam"] = entries[name]
        seen.add(name)
    missing = sorted(set(entries) - seen)
    assert not missing, f"Beam entries for scenarios that do not exist: {missing}"
    # Beam receives every shared null schema UNCHANGED. Its typing resolves sibling
    # `type` and anyOf/oneOf constraints conjunctively (a deliberate correction of the
    # frozen reference's union), so no per-family "effective schema" substitution exists.


_wire_beam_into_edge_cases()

# Shared schema cases are part of the same scenario inventory as every other case.
def _dotted_native_input(family, calls):
    segments = [("tool", name, arguments) for name, arguments in calls]
    text = _deepseek_v41_input(segments)[0] if family == "deepseek_v41" else render_input(family, segments)
    # Harmony generation continues an already-open assistant turn. A second
    # initial assistant header prevents the reasoning parser's tool handoff.
    return text.removeprefix("<|start|>assistant") if family == "gpt_oss" else text


for _scenario, _description, _calls in (
    ("dotted_name_minimal", "A native call to a.b preserves the complete name and empty arguments.",
     [("a.b", {})]),
    ("dotted_name_mixed_punctuation", "A native call to my-tool.v2 preserves both punctuation characters and its Beijing argument (#416, #423).",
     [("my-tool.v2", {"location": "Beijing"})]),
    ("dotted_name_multiple_segments", "A native call preserves every segment of server.tools.weather.",
     [("server.tools.weather", {"location": "Beijing"})]),
    ("dotted_name_shared_prefix", "Two native calls preserve server.weather then server.time, with distinct arguments and no protocol text.",
     [("server.weather", {"location": "Beijing"}), ("server.time", {"timezone": "Asia/Shanghai"})]),
):
    EDGE.append((
        _scenario, _description, [],
        [{"kind": "tool_call", "name": name, "arguments": arguments} for name, arguments in _calls],
        {"starting_state": "None", "tool_output_mode": "Native", "named_tool": None},
        {"finish_reason": "stop", "single_chunk_families": ["gpt_oss"], "finish_step": True},
        OnlyFamilies(by_family(lambda family: _dotted_native_input(family, _calls), M, M)),
        [{"name": name, "parameters": {
            "type": "object", "properties": {key: {"type": "string"} for key in arguments},
            "required": list(arguments), "additionalProperties": False,
        }} for name, arguments in _calls],
    ))


_SCHEMA_CASES_BY_FAMILY = {
    family: unified_schema_cases(family, r_tool_arguments) for family in TEMPLATE_FAMILIES
}
for _schema_probe in CONFORMANCE_CASES:
    _scenario = f"schema_{_schema_probe['name']}"
    _prototype = _SCHEMA_CASES_BY_FAMILY[FAMILIES[0]][f"UNIFIED.{_scenario}.{FAMILIES[0]}"]
    EDGE.append((
        _scenario, _prototype["description"], _prototype["policy"], _prototype["golden"],
        _prototype["init"], {"finish_reason": "stop"},
        OnlyFamilies({family: (
            _SCHEMA_CASES_BY_FAMILY[family][f"UNIFIED.{_scenario}.{family}"]["input"],
            _prototype["expect"]["vllm"], M,
            _SCHEMA_CASES_BY_FAMILY[family][f"UNIFIED.{_scenario}.{family}"]["golden"],
        ) for family in TEMPLATE_FAMILIES}),
        _prototype["tools"],
    ))


for _probe in XML_COMPATIBILITY_CASES:
    EDGE.append((
        _probe["name"], schema_description(_probe), ["I7"],
        [{"kind": "tool_call", "name": "schema_probe", "arguments": {"value": _probe["value"]}}],
        {"starting_state": "None", "tool_output_mode": "Native", "named_tool": None},
        {"finish_reason": "stop"},
        OnlyFamilies({_family: (
            r_tool_arguments(_family, "schema_probe", {"value": _probe["value"]}, 0,
                             {"value": _probe["raw"]}),
            D("UNSUPPORTED", "No peer capture recorded for this quote compatibility probe."), M,
        ) for _family in (TEMPLATE_FAMILIES if _probe["schema"]["type"] == "string" else ["qwen3"])}),
        schema_tools(_probe),
    ))


_DSML_TEMPLATE_DIALECTS = {
    "deepseek_v3_2": ("", "function_calls"),
    "deepseek_v4": ("", "tool_calls"),
    "deepseek_v41": (" ", " calls"),
}


def r_dsml_template_call(fam, bare=False):
    """One get_weather call laid out exactly as the DeepSeek encoder renders it."""
    gap, block = _DSML_TEMPLATE_DIALECTS[fam]
    invoke = (
        f'<｜DSML｜{gap}invoke name="get_weather">\n'
        f'<｜DSML｜{gap}parameter name="location" string="true">NYC</｜DSML｜{gap}parameter>\n'
        f'</｜DSML｜{gap}invoke>'
    )
    return invoke if bare else f"<｜DSML｜{block}>\n{invoke}\n</｜DSML｜{block}>"


# The DeepSeek encoders render an assistant tool turn as `content + "\n\n" + calls`,
# even when content is empty. Exactly that one separator is framing; any other
# whitespace is content, so parsing and re-rendering reproduces the generated text.
DSML_FRAMING = [
    ("dsml_separator_two_lf", "The encoder's two-LF separator before a calls block is framing, not content.",
     "I will check the weather.", "\n\n", False),
    ("dsml_separator_four_lf", "Four LFs before a calls block keep the two that belong to the content.",
     "I will check the weather.\n\n", "\n\n", False),
    ("dsml_no_separator", "A calls block directly after content removes nothing from the content.",
     "I will check the weather.", "", False),
    ("dsml_empty_content_separator", "A tool-only turn renders just the separator before the calls block, so no content is emitted.",
     "", "\n\n", False),
    ("dsml_bare_invoke_two_lf", "A bare invocation consumes the two-LF separator, like a wrapped calls block.",
     "I will check the weather.", "\n\n", True),
    ("dsml_bare_invoke_four_lf", "A bare invocation keeps the two LFs that belong to the content.",
     "I will check the weather.\n\n", "\n\n", True),
    ("dsml_whitespace_content_separator", "Whitespace-only content keeps its two LFs; only the separator is framing.",
     "\n\n", "\n\n", False),
]
_DSML_FAMILIES = ("deepseek_v4", "deepseek_v41")
_DSML_INIT = {"starting_state": "None", "tool_output_mode": "Native", "named_tool": None}
_DSML_CALL = {"kind": "tool_call", "name": "get_weather", "arguments": {"location": "NYC"}}
for _scenario, _description, _content, _separator, _bare in DSML_FRAMING:
    EDGE.append((
        _scenario, _description, ["I7"],
        ([{"kind": "text", "text": _content}] if _content else []) + [_DSML_CALL],
        _DSML_INIT,
        OnlyFamilies({f: (_content + _separator + r_dsml_template_call(f, _bare), M, M) for f in _DSML_FAMILIES}),
    ))
EDGE.append((
    "dsml_reasoning_then_separator",
    "Reasoning, the separator, then a calls block: the common tool-only turn emits no content.",
    ["I7"],
    [{"kind": "reasoning", "text": "Check weather."}, _DSML_CALL],
    _DSML_INIT,
    OnlyFamilies({f: (r_reason(f, "Check weather.") + "\n\n" + r_dsml_template_call(f), M, M) for f in _DSML_FAMILIES}),
))


DEFAULT_REASONING_CASES = [
    ("default_reasoning_without_opener", "The family starts in reasoning without an opener; the closer switches to visible Unicode content.",
     ["P5", "I5"], [{"kind": "reasoning", "text": "Check Montréal."}, {"kind": "text", "text": "雪"}],
     {"starting_state": "None", "tool_output_mode": "Native", "named_tool": None},
     OnlyFamilies({f: ("Check Montréal.</think>雪", M, M) for f in FAMILIES if _MANIFEST[f].get("default_reasoning")})),
    ("default_reasoning_tool_without_closer", "Default reasoning omits both think markers; native calls interrupt and resume reasoning, preserving literal markers in arguments.",
     ["P5", "I2", "I7"], [{"kind": "reasoning", "text": "Check."}, {"kind": "tool_call", "name": "log", "arguments": {"note": "<think>雪</think>"}}, {"kind": "reasoning", "text": "Then."}],
     {"starting_state": "None", "tool_output_mode": "Native", "named_tool": None},
     OnlyFamilies({f: ("Check." + r_tool(f, "log", "note", "<think>雪</think>", 0) + "Then.", M, M) for f in FAMILIES if _MANIFEST[f].get("default_reasoning")})),
]


# Inherited grammars expose their authored rows to applicability and schema consumers too.
for _spec in EDGE:
    _policy = _edge_case_family_policy(_spec)
    if isinstance(_policy, OnlyFamilies) and not _policy.inherit_grammar:
        continue
    for _family, _row in _MANIFEST.items():
        _grammar = _row.get("grammar")
        if _grammar in _policy:
            _policy[_family] = _policy[_grammar]
        if len(_spec) == 8 and isinstance(_spec[-1], dict) and _grammar in _spec[-1]:
            _spec[-1][_family] = _spec[-1][_grammar]

EDGE.extend(DEFAULT_REASONING_CASES)

def default_response_prefix(fam):
    return control_tokens(fam)[1] if _MANIFEST[fam].get("default_reasoning") else ""
# Harmony partial headers are a distinct grammar boundary. The previous GPT-OSS
# projections used no invocation prefix (or an orphan handoff); retain their
# captured history and replace them with explicitly authored Harmony scenarios.
GPT_OSS_HEADER_INPUTS = {
    'harmony_guided_json_gt_in_argument_bare_opener': '<|start|>assistant<|channel|>commentary to=functions.[{"name": "get_weather", "arguments": {"city": "a > b"}}]',
    'harmony_guided_json_gt_in_argument_named_bare_opener': '<|start|>assistant<|channel|>commentary to=functions.get_weather[{"name": "get_weather", "arguments": {"city": "a > b"}}]',
    'harmony_guided_json_narrated_prefix_inside_reasoning': '<|channel|>analysis<|message|>I\'ll call <|start|>assistant<|channel|>commentary to=functions.get_weather<|end|>[{"name": "get_weather", "arguments": {"city": "Paris"}}]',
    'harmony_guided_json_schema_error_nameless_element_bare_opener': '<|start|>assistant<|channel|>commentary to=functions.[{"name": "get_weather", "arguments": {"city": "Paris"}}, {"arguments": {}}]',
    'harmony_guided_json_schema_error_not_a_call_bare_opener': '<|start|>assistant<|channel|>commentary to=functions.{"unexpected": "shape"}',
    'harmony_guided_json_stray_prefix_before_reasoning': '<|start|>assistant<|channel|>commentary to=functions.<|channel|>analysis<|message|>secret<|end|>[{"name": "get_weather", "arguments": {"city": "Paris"}}]',
    'harmony_guided_json_syntax_error_bare_opener': '<|start|>assistant<|channel|>commentary to=functions.[{"name": "get_weather", "arguments": {"city": ',
    'harmony_guided_json_tool_open_before_payload': '<|start|>assistant<|channel|>commentary to=functions.[{"name": "get_weather", "arguments": {"city": "Paris"}}]',
}
GPT_OSS_RETIRED_HEADER_SCENARIOS = frozenset(
    scenario.removeprefix("harmony_") for scenario in GPT_OSS_HEADER_INPUTS
)
for _edge in tuple(EDGE):
    if _edge[0] in GPT_OSS_RETIRED_HEADER_SCENARIOS:
        _name, _desc, _policy, _golden, _init, *_ = _edge
        _harmony_name = "harmony_" + _name
        EDGE.append((_harmony_name, "Harmony partial header: " + _desc,
                     _policy, _golden, _init,
                     OnlyFamilies({"gpt_oss": (GPT_OSS_HEADER_INPUTS[_harmony_name],
                                               VLLM_UNCAPTURABLE["gpt_oss"], M,
                                               *_edge_case_family_policy(_edge)["gpt_oss"][3:])})))


def _structured_response_case():
    schema = {
        "type": "object",
        "properties": {
            "location": {"type": "string", "enum": ["London", "New York"]},
            "temperature": {"type": "number"},
            "conditions": {"type": "string"},
        },
        "required": ["location", "temperature", "conditions"],
        "additionalProperties": False,
    }
    body = json.dumps({"location": "London", "temperature": 12, "conditions": "cloudy"}, separators=(",", ":"))
    reason = "Check London météo🙂 before answering; this is prose, not JSON."
    inputs, initializations = {}, {}
    unsupported = D("UNSUPPORTED", "Parser-only peer capture does not execute structural-tag generation constraints.")
    for family in ("gpt_oss", "qwen3", "glm47", "deepseek_v4", "muse_spark"):
        if family == "muse_spark":
            prefix = reason + "<|eom|><|start|>assistant to=user<|message|>"
            suffix, state, visible_prefix = "", "Reasoning", ""
        elif family == "gpt_oss":
            prefix = r_reason(family, reason).removeprefix("<|start|>assistant") + r_text(family, "").removesuffix("<|return|>")
            suffix, state, visible_prefix = "<|return|>", "None", ""
        else:
            visible_prefix = "\n\n" if family == "qwen3" else ""
            prefix = reason + "</think>" + visible_prefix
            suffix, state = "", "Reasoning"
        rejected = [prefix + invalid + suffix for invalid in (
            "[]", '{"location":',
            '{"location":"Tokyo","temperature":12,"conditions":"cloudy"}',
            '{"location":"London"}',
            '{"location":"London","temperature":12,"conditions":"cloudy","analysis":"leaked"}',
        )]
        if family == "gpt_oss":
            rejected.append(prefix.replace("final<|message|>", "final<|constrain|>json<|message|>") + "invalid JSON" + suffix)
        initializations[family] = {
            "starting_state": state, "tool_output_mode": "Native", "named_tool": None,
            "structural_tag": {"schema": schema, "rejected": rejected},
        }
        inputs[family] = (prefix + body + suffix, unsupported, M, [
            {"kind": "reasoning", "text": reason},
            {"kind": "text", "text": visible_prefix + body},
        ])
    return (
        "structured_response_after_reasoning",
        "No-tools response_format schema with free reasoning before the constrained JSON response. GPT-OSS emits its analysis and final headers; other supported builders close prompt-prefilled reasoning. The real structural tag must accept the authored Unicode stream and every split, reject wrong types, missing/extra keys and invalid values, then the native parser must preserve reasoning and response separately. Dynamo issue #16787.",
        ["P5", "I5"], [], OnlyFamilies(initializations), {"finish_reason": "stop"},
        OnlyFamilies(inputs, inherit_grammar=False), [],
    )


EDGE.append(_structured_response_case())


_GPT_OSS_REVIEW_CASES = (
    ("gpt_oss_tool_body_whitespace", "A native JSON argument body begins with space, newline, and tab. Structural whitespace must not change cursor offsets or argument values.",
     '<|start|>assistant<|channel|>commentary to=functions.lookup <|constrain|>json<|message|> \n\t{"q":"x"}<|call|>',
     [{"kind":"tool_call", "name":"lookup", "arguments":{"q":"x"}}]),
    ("gpt_oss_constrained_final_header", "Harmony final header has a content type and no recipient. The JSON body stays visible response content; it is not a routed tool message.",
     '<|start|>assistant<|channel|>final <|constrain|>json<|message|>{"a":1}<|return|>',
     [{"kind":"text", "text":'{"a":1}'}]),
    ("gpt_oss_implicit_recipient_first_header", "Harmony generated suffix begins with the recipient before its channel, without prompt token IDs. Hold the header until its message marker and dispatch the native call without leaking the recipient.",
     ' to=functions.lookup<|channel|>commentary <|constrain|>json<|message|>{"q":"x"}<|call|>',
     [{"kind":"tool_call", "name":"lookup", "arguments":{"q":"x"}}]),
    ("gpt_oss_quoted_truncated_tool_then_final", "A malformed native argument body precedes a complete visible-response message. At EOF discard the malformed call and recover the response without leaking its argument body.",
     '<|start|>assistant to=functions.lookup <|constrain|>json<|message|>{"q":"abc<|call|><|start|>assistant<|channel|>final<|message|>Answer<|return|>',
     [{"kind":"text", "text":"Answer"}]),
    ("gpt_oss_quoted_message_header_literal", "A native argument value contains a call or message closer followed by a complete visible-response header. Argument ownership keeps that adjacent header literal instead of routing its body as visible text.",
     '<|start|>assistant to=functions.lookup<|message|>{"q":"<|call|><|start|>assistant<|channel|>final<|message|>literal<|return|>"}<|call|>',
     [{"kind":"tool_call", "name":"lookup", "arguments":{"q":"<|call|><|start|>assistant<|channel|>final<|message|>literal<|return|>"}}]),
    ("gpt_oss_complete_alternate_message_ends", "Harmony native function messages contain complete JSON and end with message-end or turn-end instead of handoff. Unified recovers both delimiter-complete calls; legacy missing-handoff behavior remains separate.",
     '<|start|>assistant to=functions.lookup<|message|>{"q":"end"}<|end|><|start|>assistant to=functions.lookup<|message|>{"q":"return"}<|return|>',
     [{"kind":"tool_call", "name":"lookup", "arguments":{"q":"end"}}, {"kind":"tool_call", "name":"lookup", "arguments":{"q":"return"}}]),
)


def _dsml_json_tool(family, name, body):
    return r_tool_arguments(family, name, {}, 0).replace(
        "></｜DSML｜", ">" + body + "</｜DSML｜", 1,
    )


def _native_review_variants(scenario):
    arguments = {"q": "x"}
    if scenario == "gpt_oss_multiple_unknown_and_empty_tools":
        variants = {}
        for family in TEMPLATE_FAMILIES:
            if family == "gpt_oss":
                continue
            value = "2" if family in {"beam", "qwen3", "glm47"} else 2
            golden = [{"kind": "tool_call", "name": "run", "arguments": {}},
                      {"kind": "tool_call", "name": "not_offered", "arguments": {"x": value}}]
            variants[family] = (r_tool_arguments(family, "run", {}, 0)
                                + r_tool_arguments(family, "not_offered", {"x": 2}, 1), golden, M)
        return variants
    if scenario == "gpt_oss_control_marker_inside_argument_string":
        variants = {}
        for family in ("deepseek_v4", "deepseek_v41", "kimi_k2", "kimi_k3"):
            frames = r_reason(family, "") + r_text(family, "") + r_tool(family, "lookup", "q", "x", 0)
            markers = re.findall(r"<[^<>]*>", frames) + _MANIFEST[family].get("leak_markers", [])
            payload = "".join(dict.fromkeys(markers))
            arguments = {"markers": payload}
            if family.startswith("deepseek_"):
                native_input = _dsml_json_tool(family, "lookup", json.dumps(arguments))
            elif family == "kimi_k3":
                native_input = k3_raw_tool("lookup", json.dumps(arguments))
            else:
                native_input = r_tool_arguments(family, "lookup", arguments, 0)
            variants[family] = (native_input, [{"kind": "tool_call", "name": "lookup", "arguments": arguments}], M)
        return variants
    if scenario in {"gpt_oss_invalid_tool_body_then_empty_call", "gpt_oss_invalid_tool_body_then_valid_call"}:
        empty = scenario.endswith("empty_call")
        body = "not-json" if empty else '{"city":'
        arguments = {} if empty else {"city": "Paris"}
        following = [{"kind": "tool_call", "name": "get_weather", "arguments": arguments}]
        variants = {
            "kimi_k2": (r_tool_arguments("kimi_k2", "bad", {}, 0).replace("{}", body, 1)
                        + r_tool_arguments("kimi_k2", "get_weather", arguments, 1),
                        [{"kind": "tool_call", "name": "bad", "arguments": {}}] + following, M),
            "kimi_k3": (k3_raw_tool("bad", body) + r_tool_arguments("kimi_k3", "get_weather", arguments, 1), following, M),
        }
        for family in ("deepseek_v4", "deepseek_v41"):
            golden = ([{"kind": "tool_call", "name": "bad", "arguments": {}}]
                      if family == "deepseek_v4" else []) + following
            dynamo = (D("ERROR", "An invalid invocation aborts parsing before the following empty call.")
                      if family == "deepseek_v41" and empty else M)
            variants[family] = (_dsml_json_tool(family, "bad", body)
                                + r_tool_arguments(family, "get_weather", arguments, 1), golden, dynamo)
        return variants
    if scenario == "gpt_oss_truncated_tool_resynchronizes_at_next_message":
        variants = {}
        for family in ("beam", "muse_glimmer", "kimi_k3"):
            if family == "kimi_k3":
                native_input = k3_tools(k3_call("lookup", 1, k3_json('{"q":"unfinished"}'), close=False), close=False)
            else:
                native_input = r_tool(family, "lookup", "q", "unfinished", 0).removesuffix(control_tokens(family)[3])
            golden = ([{"kind": "tool_call", "name": "lookup", "arguments": {"q": "unfinished"}}]
                      if family == "muse_glimmer" else []) + [{"kind": "text", "text": "recovered"}]
            dynamo = (D("LOSS", "An unclosed tool call swallows the later response; retain the visible-answer oracle.")
                      if family == "kimi_k3" else M)
            variants[family] = (native_input + r_text(family, "recovered"), golden, dynamo)
        return variants
    if scenario == "gpt_oss_interrupted_reasoning_then_response":
        golden = [{"kind": "reasoning", "text": "checking"}, {"kind": "text", "text": "done"}]
        return {family: (r_reason(family, "checking").removesuffix(control_tokens(family)[1])
                         + r_text(family, "done"), golden, M) for family in ("beam", "muse_glimmer")}
    if scenario == "gpt_oss_tool_body_whitespace":
        golden = [{"kind": "tool_call", "name": "lookup", "arguments": arguments}]
        return {
            "kimi_k2": (
                r_tool_arguments("kimi_k2", "lookup", arguments, 0).replace(
                    "<|tool_call_argument_begin|>", "<|tool_call_argument_begin|> \n\t", 1
                ), golden, M,
            ),
            "kimi_k3": (k3_raw_tool("lookup", ' \n\t{"q":"x"}'), golden, M),
        }
    if scenario == "gpt_oss_quoted_truncated_tool_then_final":
        golden = [{"kind": "text", "text": "Answer"}]
        return {
            "muse_glimmer": (
                '<|start|>assistant to=lookup<|message|><atem:invoke name="lookup">'
                '<atem:parameter name="q">abc<|eom|>' + r_text("muse_glimmer", "Answer"),
                golden, M,
            ),
            "kimi_k3": (
                k3_raw_tool("lookup", '{"q":"abc') + r_text("kimi_k3", "Answer"),
                golden, D("LOSS", "An unterminated raw JSON string swallows the later response; retain the visible-answer oracle."),
            ),
        }
    if scenario == "gpt_oss_quoted_message_header_literal":
        variants = {}
        for family in ("muse_glimmer", "kimi_k3"):
            closer = "<|eom|>" if family == "muse_glimmer" else k3_close("call")
            payload = closer + r_text(family, "literal")
            arguments = {"q": payload}
            native_input = (
                r_tool_arguments(family, "lookup", arguments, 0)
                if family == "muse_glimmer"
                else k3_raw_tool("lookup", json.dumps(arguments))
            )
            variants[family] = (native_input, [
                {"kind": "tool_call", "name": "lookup", "arguments": arguments},
            ], M)
        return variants
    return {}


for _name, _description, _input, _golden in _GPT_OSS_REVIEW_CASES:
    _variants = _native_review_variants(_name)
    _inputs = OnlyFamilies({"gpt_oss": (_input, D("UNSUPPORTED", VLLM_UNCAPTURABLE["gpt_oss"]), M)})
    _inputs.update({family: (native_input, D("UNSUPPORTED", "No peer capture recorded."), dynamo, golden)
                    for family, (native_input, golden, dynamo) in _variants.items()})
    _tools = {"gpt_oss": [], **{family: [{"name": "lookup", "parameters": {
        "type": "object", "properties": {"q": {"type": "string"}},
    }}] for family in _variants}}
    EDGE.append((
        _name, _description, ["I3", "I5", "I7"], _golden,
        {"starting_state":"None", "tool_output_mode":"Native", "named_tool":None},
        {"finish_reason": "stop"}, _inputs, _tools,
    ))


def _apply_numeric_tool_schemas(cases, family):
    for scenario, label, schema, _raw, _expected in NUMERIC_VARIANTS:
        key = f"UNIFIED.{scenario}.{family}"
        if key in cases:
            cases[key]["tools"] = [{
                "name": "get_weather",
                "parameters": {"type": "object", "properties": {"value": numeric_schema(family, label, schema)}},
            }]
    return cases


def _apply_unified_numeric_goldens(cases, family):
    for scenario, label, _schema, raw, expected in NUMERIC_VARIANTS:
        case = cases.get(f"UNIFIED.{scenario}.{family}")
        if case is not None:
            case["golden"][0]["arguments"] = arguments_json(
                unified_numeric_expected(family, label, raw, expected)
            )
    return cases

for _index, _edge in enumerate(EDGE):
    _variants = _native_review_variants(_edge[0])
    _inputs = _edge_case_family_policy(_edge)
    _new = {family: variant for family, variant in _variants.items() if family not in _inputs}
    if not _new:
        continue
    _inputs.update({family: (native_input, D("UNSUPPORTED", "No peer capture recorded."), dynamo, golden)
                    for family, (native_input, golden, dynamo) in _new.items()})
    _parts = list(_edge)
    if len(_parts) == 6:
        _parts.insert(5, {"finish_reason": "stop"})
    if len(_parts) == 7:
        _parts.append({"gpt_oss": []})
    _parts[-1].update({family: [] for family in _new})
    EDGE[_index] = tuple(_parts)


def build_cases(fam):
    grammar = _MANIFEST[fam].get("grammar")
    if grammar is None:
        cases = _build_cases(fam)
    else:
        cases = {}
        for key, original in _build_cases(grammar).items():
            scenario = key[len("UNIFIED."):].rsplit(".", 1)[0]
            if fam not in scenario_families(scenario):
                continue
            case = json.loads(json.dumps(original))
            # Shared Qwen scenarios begin outside reasoning; Ultra's empty thought
            # closes that default channel without disabling later reasoning markers.
            if (case.get("init") or {}).get("starting_state", "None") == "None":
                case["input"] = default_response_prefix(fam) + case["input"]
                if "input_chunks" in case:
                    case["input_chunks"][0] = default_response_prefix(fam) + case["input_chunks"][0]
            cases[key.rsplit(".", 1)[0] + "." + fam] = case
        cases.update(_build_edge_cases(fam, DEFAULT_REASONING_CASES))
        _apply_numeric_tool_schemas(cases, fam)
        _preserve_historical_requests(cases, fam)
        cases = _apply_unified_numeric_goldens(cases, fam)
    return {
        key: case
        for key, case in cases.items()
        if key.rsplit(".", 1)[0].removeprefix("UNIFIED.") not in DOTTED_NAME_SCENARIOS
        or fam in DOTTED_NAME_FAMILIES
    }


def _build_cases(fam):
    """Every CLEAN + EDGE scenario for one family, keyed by case id."""
    cases = {}
    for name, desc, policy, segs, vllm, dynamo in CLEAN:
        inp, state = (_deepseek_v41_input(segs) if fam == "deepseek_v41"
                      else (render_input(fam, segs), "None"))
        cid = f"UNIFIED.{name}.{fam}"
        init = {"starting_state": state, "tool_output_mode": "Native", "named_tool": None}
        cases[cid] = {
            "description": desc,
            "policy": policy,
            "input": inp,
            "golden": golden_of(segs),
            "expect": {"vllm": _vllm_entry_for_init(vllm, fam, init), "dynamo": _entry(dynamo, fam)},
            "init": init,
            "finish_reason": "stop",
        }
    cases.update(_build_edge_cases(fam, EDGE))
    if fam in ("gemma4", "deepseek_v41"):
        case = cases[f"UNIFIED.malformed_json_then_two_valid_calls.{fam}"]
        case["expect"]["dynamo_current"] = case["expect"]["dynamo"]
    if fam == "deepseek_v41":
        case = cases[f"UNIFIED.reason_unterminated.{fam}"]
        case["input"] = case["input"].removeprefix("<think>")
        case["init"] = {**case["init"], "starting_state": "Reasoning"}
        case["expect"]["vllm"] = _vllm_entry_for_init(
            case["expect"]["vllm"], fam, case["init"]
        )
    if fam == "gpt_oss":
        for case in cases.values():
            case.setdefault("tools", [])
    if fam == "gpt_oss":
        for scenario, active in GPT_OSS_PR424_REQUEST_VARIANTS.items():
            if not active:
                continue
            original = f"UNIFIED.{scenario}.{fam}"
            cases[f"UNIFIED.{scenario}_pr424.{fam}"] = dict(cases[original])
    _preserve_historical_requests(cases, fam)
    if fam == "gpt_oss":
        _preserve_gpt_oss_historical_oracle(cases)
    return _apply_unified_numeric_goldens(cases, fam)


@lru_cache(maxsize=16)
def _historical_cases(document_text):
    return yaml.safe_load(document_text)["cases"]


def _preserve_historical_requests(cases, family):
    """Keep published request bytes and chunk schedules stable by display ID."""
    # The renderer copies this module into a temporary layout. Its existing
    # repository context keeps history tied to the checkout, not that copy.
    source_root = Path(os.environ.get("FRONTEND_CRATES_ROOT", Path(__file__).resolve().parents[3]))
    historical_path = source_root / f"conformance/fixtures-unified-v2/families/{family}/inputs_and_golden.yaml"
    historical = _historical_cases(historical_path.read_text())
    cases_by_scenario = {
        case_id.removeprefix("UNIFIED.").rsplit(".", 1)[0]: case
        for case_id, case in cases.items()
    }
    for old_case in historical.values():
        scenario = old_case["scenario"]
        case = cases_by_scenario.get(scenario)
        if case is None:
            if old_case.get("lifecycle") != "active":
                continue
            raise KeyError(f"historical {family} case disappeared: {scenario}")
        # Validation callers mutate returned cases; isolate the cached canonical record.
        case["golden"] = deepcopy(old_case["golden"]["assembled"])
        if scenario.endswith("_pr424"):
            case["description"] = old_case["description"]
        request = deepcopy(old_case["request"])
        case["input"] = request["input"]
        case["init"] = request["init"]
        case["finish_reason"] = request["finish_reason"]
        if "tools" in request:
            case["tools"] = request["tools"]
        chunks = request.get("chunks", [])
        if any(set(chunk) != {"delta_text"} for chunk in chunks):
            raise ValueError(
                f"historical {family} request has unsupported chunk schedule: {scenario}"
            )
        input_chunks = [chunk["delta_text"] for chunk in chunks]
        if (
            input_chunks
            and input_chunks[-1] == "‹finish›"
            and "".join(input_chunks[:-1]) == request["input"]
        ):
            input_chunks.pop()
        if "".join(input_chunks) != request["input"]:
            raise ValueError(
                f"historical {family} chunk schedule does not reproduce {scenario}"
            )
        case["input_chunks"] = input_chunks


def _preserve_gpt_oss_historical_oracle(cases):
    """Keep the corrected text-only oracle for a retained non-Harmony request."""
    legacy_plain_case = cases["UNIFIED.tool_block_never_closed_then_text.gpt_oss"]
    if "to=functions." in legacy_plain_case["input"] or not legacy_plain_case["input"].endswith(
        "still thinking about it"
    ):
        raise ValueError("the preserved GPT-OSS historical tool-block request changed")
    # This retained request contains plain prose, not a Harmony tool header, so its
    # current-parser oracle is visible text. Valid Harmony truncation has separate cases.
    legacy_plain_case["golden"] = [{"kind": "text", "text": "still thinking about it"}]
    legacy_plain_case["description"] = (
        "An orphan Harmony handoff precedes prose without opening a tool message. "
        "Strip the orphan marker and retain the prose. Genuine truncated tool bodies are covered separately."
    )


def _build_edge_cases(fam, specs):
    cases = {}
    for edge_case in specs:
        if fam == "gpt_oss" and edge_case[0] in HARMONY_INAPPLICABLE_EDGE_CASES:
            continue
        if fam == "deepseek_v41" and edge_case[0] in DEEPSEEK_V41_REDUNDANT_SCENARIOS:
            continue
        # Support legacy tuples, stream config, and per-case request tool schemas.
        case_tools = None
        if len(edge_case) == 6:
            name, desc, policy, golden, init, per_fam = edge_case
            stream_config = {"finish_reason": "stop"}
        elif len(edge_case) == 8:
            name, desc, policy, golden, init, stream_config, per_fam, case_tools = edge_case
        else:
            name, desc, policy, golden, init, stream_config, per_fam = edge_case

        if fam == "muse_spark" and "muse_glimmer" in per_fam and fam not in per_fam:
            per_fam = type(per_fam)(dict(per_fam, muse_spark=per_fam["muse_glimmer"]))
        if isinstance(per_fam, OnlyFamilies) and fam not in per_fam:
            continue
        if isinstance(init, OnlyFamilies):
            init = init[fam]

        # A family that rejects the mode cannot produce a cell for it. The harness
        # applies `init` before parsing and panics on the rejection, so the case is
        # skipped rather than recorded as a divergence. Only GUIDED output is gated;
        # a prefilled starting state is honoured by every family.
        if (init or {}).get("tool_output_mode", "Native") != "Native" and fam not in GUIDED_FAMILIES:
            continue

        # A scenario may DECLARE a grammar- or regression-specific scope with
        # OnlyFamilies. Absence from a plain map is still a hard failure — an
        # accidentally omitted family must break generation rather than quietly read as
        # "not applicable", which would hide missing coverage behind the same cell the
        # corpus uses for a real structural gap.
        if isinstance(per_fam, OnlyFamilies) and fam not in per_fam:
            continue

        cid = f"UNIFIED.{name}.{fam}"
        if fam not in per_fam:
            if fam == "gpt_oss" and name in HARMONY_EDGE_INPUTS:
                per_fam = dict(per_fam)
                extra = (
                    [HARMONY_EDGE_GOLDEN_VALUES[name]]
                    if name in HARMONY_EDGE_GOLDEN_VALUES
                    else [HARMONY_EDGE_GOLDENS[name]]
                    if name in HARMONY_EDGE_GOLDENS
                    else []
                )
                per_fam[fam] = (
                    HARMONY_EDGE_INPUTS[name],
                    D("UNSUPPORTED", "No GPT-OSS vLLM Unified capture is configured."),
                    M,
                    *extra,
                )
            elif fam == "gpt_oss" and name in HARMONY_INAPPLICABLE_EDGE_CASES:
                continue
            elif fam == "gpt_oss":
                raise KeyError(
                    f"{name}: GPT-OSS has no authored Harmony input; add a case or "
                    "declare a grammar-specific exclusion"
                )
            elif fam == "deepseek_v4" and "kimi_k2" in per_fam:
                kimi_input, *rest = per_fam["kimi_k2"]
                per_fam = dict(per_fam)
                per_fam[fam] = (kimi_input_as_dsml(kimi_input), *rest)
            elif fam == "glm47" and "qwen3" in per_fam:
                qwen_input, *rest = per_fam["qwen3"]
                per_fam = dict(per_fam)
                per_fam[fam] = (qwen3_input_as_glm47(qwen_input), *rest)
            else:
                raise KeyError(
                    f"{name}: no input authored for family {fam!r}. Add one, or wrap the map "
                    f"in OnlyFamilies({{...}}) for an explicit grammar- or regression-specific "
                    f"scope (and say why at the authoring site and in UNIFIED_CASES.md)."
                )
        inp, vllm, dynamo, *rest = per_fam[fam]
        g = json.loads(json.dumps(golden))  # deep copy
        if rest and isinstance(rest[0], list):
            g = json.loads(json.dumps(rest[0]))
        elif rest:
            # Fill the ONE `None` placeholder in the golden with this family's
            # value. It may be an argument value (a marker-looking string that has
            # to survive byte-exact, 12-1) or a whole text payload (the family's
            # own markers reaching the user as literal text, 50.d) — either way
            # the scenario is shared and only the grammar-specific bytes differ.
            for ev in g:
                args = ev.get("arguments")
                if args and any(v is None for v in args.values()):
                    fk = next(k for k, v in args.items() if v is None)
                    args[fk] = rest[0]
                    break
                if ev.get("kind") in ("text", "reasoning") and ev.get("text") is None:
                    ev["text"] = rest[0]
                    break
        # ENFORCED HERE, not at each authoring site. `every_family()` and
        # `guided_surroundings()` already substitute UNSUPPORTED for a family with
        # no native unified parser, but a scenario hand-written as an explicit
        # per-family dict bypasses them and can hand gemma4/kimi_k2 a bare `match`
        # under a guided or prefilled `init` — a family on the v1-reasoning +
        # v2-tool split ignores `init` entirely, so it cannot honour that mode.
        # Nothing asserts this field (the Dynamo column is computed live), so a
        # false `match` never fails; it just tells a reader two engines handle
        # request modes they cannot see. One gate every case passes through is the
        # only way an authoring shortcut cannot route around it.
        if fam not in UNIFIED_FAMILIES and _init_is_request_scoped(init):
            dynamo = D(
                "UNSUPPORTED",
                "no native unified parser in this build, so the split path ignores "
                "`init` and cannot honour this request mode",
            )
        case = {
            "description": desc,
            "policy": policy,
            "input": inp,
            "golden": g,
            "expect": {"vllm": _vllm_entry_for_init(vllm, fam, init), "dynamo": dynamo},
            "init": init,
            "finish_reason": stream_config.get("finish_reason", "stop"),
        }
        if stream_config.get("single_chunk") or fam in stream_config.get("single_chunk_families", []):
            case["input_chunks"] = [inp]
        elif "input_chunks" in stream_config:
            schedules = stream_config["input_chunks"]
            # Spark inherits the shared ATEM input and its authored chunk boundaries.
            source = "muse_glimmer" if fam == "muse_spark" and fam not in schedules else fam
            chunks = schedules[source]
            if not all(isinstance(chunk, str) for chunk in chunks) or "".join(chunks) != inp:
                raise ValueError(f"{cid}: authored input_chunks must concatenate to input")
            case["input_chunks"] = list(chunks)
        if stream_config.get("finish_step"):
            case["finish_step"] = True
        if case_tools is not None:
            if isinstance(case_tools, dict) and fam == "muse_spark" and fam not in case_tools:
                case["tools"] = case_tools["muse_glimmer"]
            else:
                case["tools"] = case_tools[fam] if isinstance(case_tools, dict) else case_tools
        cases[cid] = case
    return cases


def scenario_families(scenario):
    """Return the families for which an authored scenario is applicable.

    A plain per-family map means the scenario is part of the full matrix. An
    ``OnlyFamilies`` map declares a grammar-specific or regression-specific scope;
    absence does not assert that the omitted families lack the capability. The table builder
    uses this same declaration to render explicit n/a cells instead of
    silently dropping them.
    """
    if scenario in RETIRED_SCENARIOS:
        return frozenset()
    original = scenario.removesuffix("_pr424")
    if original != scenario and original in GPT_OSS_PR424_REQUEST_VARIANTS:
        return frozenset({"gpt_oss"}) if GPT_OSS_PR424_REQUEST_VARIANTS[original] else frozenset()
    if scenario in DOTTED_NAME_SCENARIOS:
        return DOTTED_NAME_FAMILIES
    for name, *_rest in CLEAN:
        if name == scenario:
            return frozenset(FAMILIES if scenario in DEEPSEEK_V41_SCENARIOS else SHARED_FAMILIES)
    for edge_case in EDGE:
        name = edge_case[0]
        if name != scenario:
            continue
        per_fam = _edge_case_family_policy(edge_case)
        families = (set(per_fam) if isinstance(per_fam, OnlyFamilies)
                    else set(FAMILIES if scenario in DEEPSEEK_V41_SCENARIOS else SHARED_FAMILIES))
        if isinstance(per_fam, OnlyFamilies) and "muse_glimmer" in per_fam:
            families.add("muse_spark")
        if name in HARMONY_INAPPLICABLE_EDGE_CASES:
            families.discard("gpt_oss")
        elif not isinstance(per_fam, OnlyFamilies) and "gpt_oss" not in per_fam and name not in HARMONY_EDGE_INPUTS:
            raise KeyError(f"{name}: GPT-OSS applicability has no Harmony input or explicit exclusion")
        init = edge_case[4] if len(edge_case) >= 6 else {}
        if (init or {}).get("tool_output_mode", "Native") != "Native":
            families.intersection_update(GUIDED_FAMILIES)
        if not isinstance(per_fam, OnlyFamilies) or per_fam.inherit_grammar:
            families.update(f for f in FAMILIES if _MANIFEST[f].get("grammar") in families)
        return frozenset(families)
    raise KeyError(f"unknown unified scenario {scenario!r}")


# --- YAML emitter: `input` as a block literal, everything else as inline JSON
# (valid YAML, and json.dumps escapes the marker-heavy strings safely). --------

def emit_yaml(fam):
    cases = build_cases(fam)
    lines = [
        f"# Golden (spec-derived) unified event cases for the {fam} grammar.",
        "#",
        "# GENERATED by conformance/utils/src/gen_unified_golden.py from ONE scenario",
        "# spec -- do not edit by hand; edit the spec so every family stays in lockstep.",
        "# GOLDEN is the AUTHORED correctness oracle (what a correct UnifiedParser MUST",
        "# emit), reasoned from UNIFIED_CASES.md -- NOT captured from any implementation.",
        f"# {fam} grammar: {GRAMMAR_NOTE[unified_grammar_family(fam)]}",
        "version: 1",
        f"family: {fam}",
        "cases:",
    ]
    for cid in sorted(cases):
        c = cases[cid]
        lines.append(f"  {cid}:")
        lines.append(f"    description: {json.dumps(c['description'], ensure_ascii=False)}")
        lines.append(f"    policy: {json.dumps(c['policy'])}")
        lines.append(f"    init: {json.dumps(c['init'], ensure_ascii=False)}")
        lines.append(f"    finish_reason: {json.dumps(c['finish_reason'])}")
        if c.get("finish_step"):
            lines.append("    finish_step: true")
        # EXPLICIT indentation indicator. A bare `|-` lets YAML infer the block's
        # indentation from its first non-empty line, so an input that legitimately
        # BEGINS with a space loses that byte on reload — the reader cannot tell
        # content-space from indent-space. `34-7` is authored with a leading space
        # (the bare-header form) and was emitted 110 bytes, reloaded 109: the corpus
        # was measuring a different input than the one authored. `2` is the content
        # indentation relative to this mapping node, and the trailing `-` keeps the
        # existing strip-final-newline behaviour.
        lines.append("    input: |2-")
        for ln in c["input"].split("\n"):
            lines.append(f"      {ln}")
        lines.append(f"    golden: {json.dumps(c['golden'], ensure_ascii=False)}")
        lines.append(f"    expect: {json.dumps(c['expect'], ensure_ascii=False)}")
        if "input_chunks" in c:
            chunks = c["input_chunks"]
            if not isinstance(chunks, list) or any(not isinstance(chunk, str) for chunk in chunks) or "".join(chunks) != c["input"]:
                raise ValueError(f"{cid}: input_chunks must concatenate to input")
            lines.append(f"    input_chunks: {json.dumps(chunks, ensure_ascii=False)}")
        if c.get("tools") is not None:
            lines.append(f"    tools: {json.dumps(c['tools'], ensure_ascii=False)}")
    return "\n".join(lines) + "\n"


def main():
    root = os.path.join(os.path.dirname(__file__), "..", "..", "unified", "golden_spec")
    root = os.path.abspath(root)
    os.makedirs(root, exist_ok=True)
    for fam in FAMILIES:
        out = os.path.join(root, FAM_FILE[fam])
        with open(out, "w") as fh:
            fh.write(emit_yaml(fam))
        print(f"wrote {out} ({len(build_cases(fam))} cases)")


if __name__ == "__main__":
    main()
