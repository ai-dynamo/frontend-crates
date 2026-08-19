# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render the unified golden corpus for ALL families from ONE scenario spec.

The GOLDEN event list is the authored, spec-derived oracle (best-effort error
recovery, see UNIFIED_CASES.md). It is grammar-INDEPENDENT: a scenario means the
same thing for every family, so its golden events are written once here. Only the
raw model `input` is grammar-specific, rendered from each family's markers. This
is the single source of truth so a scenario can't drift between families
(CLAUDE.md: reuse the shared parent, don't copy-paste divergent cases).

Full matrix: every scenario is emitted for every family (gemma4, qwen3, kimi_k2)
-> conformance/unified/golden_spec/{gemma4,qwen3,kimi}.yaml, the gitignored build
tree. This authored spec is the harness INPUT (unified_render.rs reads it to
compute the live Dynamo column; unified_schema_roundtrip.rs validates it); it is
NOT committed. The committed, versioned golden.tar.gz shard is DERIVED from it via
render -> explode -> package, exactly like every other conformance fixture shard.

Run:  python3 conformance/utils/src/gen_unified_golden.py
"""
import json
import os
import re

import yaml

import markers

# Families and their golden-spec filenames come from the ONE declaration in
# parser_families.yaml (`unified:`), so adding a family to this generator is adding a
# row there rather than editing three lists that had to agree.
_MANIFEST = yaml.safe_load(markers.parser_families_path().read_text())["unified"]
FAMILIES = sorted(_MANIFEST)
FAM_FILE = {f: r["golden_spec"] for f, r in _MANIFEST.items()}
UNIFIED_FAMILIES = {f for f, r in _MANIFEST.items() if r.get("native")}

GRAMMAR_NOTE = {
    "gemma4": "reasoning `<|channel>thought\\n...<channel|>`, tool `<|tool_call>call:NAME{key:<|\"|>value<|\"|>}<tool_call|>` (string values wrapped in `<|\"|>`; an embedded `<tool_call|>` inside a `<|\"|>` string is data, not the end marker).",
    "qwen3": "reasoning `<think>...</think>`, tool `<tool_call><function=NAME><parameter=KEY>VALUE</parameter></function></tool_call>`.",
    "kimi_k2": "reasoning `<think>...</think>`, tool section `<|tool_calls_section_begin|><|tool_call_begin|>functions.NAME:IDX<|tool_call_argument_begin|>{...}<|tool_call_end|><|tool_calls_section_end|>`.",
    "muse_glimmer": "recipient-routed messages `<|start|>assistant to=RCPT<|message|>...<|eom|>`: `self` is reasoning, `user` is visible content, any other recipient opens a tool channel whose body is ATEM XML `<atem:function_calls><atem:invoke name=\"NAME\"><atem:parameter name=\"KEY\">VALUE</atem:parameter></atem:invoke></atem:function_calls>`. `<|eom|>` closes a message with more to follow, `<|eot|>` ends the turn. Spec: https://huggingface.co/meta-models/Muse-Glimmer-30B.",
}


# --- grammar renderers: one semantic segment -> that family's raw text --------

def r_reason(fam, text):
    if fam == "gemma4":
        return f"<|channel>thought\n{text}<channel|>"
    if fam == "muse_glimmer":
        return f"<|start|>assistant to=self<|message|>{text}<|eom|>"
    return f"<think>{text}</think>"


def r_text(fam, text):
    """Visible content.

    Muse has no unframed content channel — every message is recipient-routed —
    so visible text renders as a closed `to=user` message. `<|eom|>` and not
    `<|eot|>`: `<|eot|>` ends the TURN, which would make any following segment
    unreachable. The marker-pair grammars leave visible text bare.
    """
    if fam == "muse_glimmer":
        return f"<|start|>assistant to=user<|message|>{text}<|eom|>"
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
    try:
        json.loads(val)
    except ValueError:
        return val
    return json.dumps(val)


def r_tool(fam, name, key, val, idx):
    if fam == "gemma4":
        return f"<|tool_call>call:{name}{{{key}:<|\"|>{val}<|\"|>}}<tool_call|>"
    if fam == "qwen3":
        return (f"<tool_call>\n<function={name}>\n<parameter={key}>\n"
                f"{val}\n</parameter>\n</function>\n</tool_call>")
    if fam == "muse_glimmer":
        return (f"<|start|>assistant to={name}<|message|><atem:function_calls>\n"
                f"<atem:invoke name=\"{name}\">\n"
                f"<atem:parameter name=\"{key}\">{_atem_value(val)}</atem:parameter>\n"
                f"</atem:invoke>\n</atem:function_calls><|eom|>")
    args = json.dumps({key: val}, ensure_ascii=False)
    return (f"<|tool_calls_section_begin|><|tool_call_begin|>functions.{name}:{idx}"
            f"<|tool_call_argument_begin|>{args}<|tool_call_end|><|tool_calls_section_end|>")


def render_input(fam, segs):
    """Concatenate rendered segments (grammars are self-delimiting)."""
    out = []
    tool_idx = 0
    for s in segs:
        if s[0] == "reason":
            out.append(r_reason(fam, s[1]))
        elif s[0] == "text":
            out.append(r_text(fam, s[1]))
        elif s[0] == "tool":
            _, name, key, val = s
            out.append(r_tool(fam, name, key, val, tool_idx))
            tool_idx += 1
    return "".join(out)


def golden_of(segs):
    ev = []
    for s in segs:
        if s[0] == "reason":
            ev.append({"kind": "reasoning", "text": s[1]})
        elif s[0] == "text":
            ev.append({"kind": "text", "text": s[1]})
        elif s[0] == "tool":
            _, name, key, val = s
            ev.append({"kind": "tool_call", "name": name, "arguments": {key: val}})
    return ev


# --- verdict shorthands -------------------------------------------------------

M = {"verdict": "match"}


def D(cls, note):
    return {"verdict": "diverge", "class": cls, "note": note}


# --- per-family input helpers for EDGE scenarios ------------------------------

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
        for fam in FAMILIES
    }



def by_family(render, vllm, dynamo, *rest):
    """`render(fam) -> input` for the scenarios where only the reasoning envelope
    around an otherwise identical payload is grammar-specific."""
    return {fam: (render(fam), vllm, dynamo, *rest) for fam in FAMILIES}


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
    reason_open, reason_close = r_reason(fam, "\x00").split("\x00")
    tokens = re.findall(r"<[^<>]*>", r_tool(fam, "NAMEX", "KEYX", "VALX", 0))
    return reason_open, reason_close, tokens[0], tokens[-1]


def invoke_header_prefix(fam):
    """Inner invoke header through the tool name, without its terminator."""
    rendered = r_tool(fam, "NAMEX", "KEYX", "VALX", 0)
    outer = control_tokens(fam)[2]
    return rendered[len(outer):rendered.index("NAMEX")].lstrip()


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
        for fam in FAMILIES
    }


# Guided payloads, written once. `named` is what a NAMED choice emits (that
# tool's arguments alone); the arrays are what a REQUIRED choice emits.
GUIDED_NAMED_ARGS = '{"city": "Paris"}'
GUIDED_ONE_CALL = '[{"name": "get_weather", "arguments": {"city": "Paris"}}]'
GUIDED_TWO_CALLS = ('[{"name": "get_weather", "arguments": {"city": "Paris"}}, '
                    '{"name": "run", "arguments": {"cmd": "git log"}}]')
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

# Families `capture_vllm_unified.py` has no entry for. The Unified tab falls back to
# the AUTHORED `expect.vllm` whenever a capture is missing, so for these families it
# falls back on EVERY case and draws the same plain `expected: MATCH` a captured
# family earns. Carrying the caveat only on the cases that happened to need a
# per-family verdict published the other 22 as if an engine had produced them.
VLLM_UNCAPTURABLE = {"muse_glimmer": V_MUSE}


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
     {"gemma4": M, "qwen3": M, "muse_glimmer": V_MUSE,
      "kimi_k2": D("LOSS", "kimi config stays in a tool state and SUPPRESSES trailing text -> arbitrary content dropped; violates best-effort recovery (preserve visible prose, conformance/README.md:142)")},
     {"gemma4": M, "qwen3": M,
      "muse_glimmer": {"verdict": "match", "note": "the tool channel closes at its own `<|eom|>`, so the following `to=user` message is ordinary content"},
      "kimi_k2": {"verdict": "match", "note": "P1 resolved by the v2 recovery contract: preserve trailing prose. Verify v2 kimi_k2 at capture time"}}),

    # --- Group 2: multiple tool calls (TOOLCALLING.streamv2.2) — tool-only, green everywhere ---
    ("two_calls",
     "Two tool calls back-to-back, no reasoning. Both must surface as ordered events. This is also covered in: TOOLCALLING.streamv2.2.a.",
     [], [("tool", "f", "x", "1"), ("tool", "g", "y", "2")], M, M),
    ("two_calls_same_name",
     "The same tool called twice with different args. Both calls are distinct events. This is also covered in: TOOLCALLING.streamv2.2.d.",
     [], [("tool", "get_weather", "city", "Paris"), ("tool", "get_weather", "city", "Tokyo")], M, M),

    # --- Group 3: no tool call ---
    ("text_only",
     "Plain answer, no reasoning and no tool call. Pure content passthrough. This is also covered in: TOOLCALLING.streamv2.3. No e2e case has this shape: Qwen3.6 always emits a reasoning span, so the plain-content case is corpus-only.",
     [], [("text", "The answer is 42, no tools needed.")], M, M),

    # --- Group 7: argument fidelity (TOOLCALLING.streamv2.7) ---
    ("arg_unicode",
     "Unicode + spaces in a string argument value. Preserved exactly (I7). This is also covered in: TOOLCALLING.streamv2.7.b.",
     [], [("tool", "get_weather", "city", "São Paulo 東京")], M, M),

    # --- Group 8: content / narration position (TOOLCALLING.streamv2.8) ---
    ("text_before_tool",
     "Visible text before a single tool call, no reasoning. This is also covered in: TOOLCALLING.streamv2.8.a.",
     [], [("text", "On it: "), ("tool", "get_weather", "city", "Paris")], M, M),
    ("text_sandwich",
     "Visible text both before and after a tool call. This is also covered in: TOOLCALLING.streamv2.8.c.",
     [], [("text", "Before. "), ("tool", "get_weather", "city", "Paris"), ("text", " After.")], M, M),
    ("text_between_calls",
     "Visible text between two tool calls. This is also covered in: TOOLCALLING.streamv2.8.d.",
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

EDGE = [
    ("truncated_tool_eof",
     "Stream ends mid tool call (no close marker). Policy P2 — drop the incomplete call, keep valid preceding output, no error, no leaked markup.",
     ["P2"],
     [{"kind": "reasoning", "text": "ok"}],
     {"starting_state": "None", "tool_output_mode": "Native", "named_tool": None},
     {
        "gemma4": ("<|channel>thought\nok<channel|><|tool_call>call:get_weather{city:<|\"|>Par",
                   D("ERROR", "native Gemma4UnifiedParser finish() returns a hard Err -> erroring is the opposite of best-effort recovery"),
                   {"verdict": "match", "note": "P2: drop the partial trailing call, keep the preceding reasoning, never error/leak (TOOLCALLING.batch.5.e)"}),
        "qwen3": ("<think>ok</think><tool_call>\n<function=get_weather>\n<parameter=city>\nPar",
                  {"verdict": "match", "note": "P2: drop the unterminated call and keep the preceding reasoning"},
                  {"verdict": "match", "note": "P2: v2 drops the partial trailing call, keeps reasoning"}),
        "kimi_k2": ("<think>ok</think><|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{\"city\": \"Par",
                    {"verdict": "match", "note": "P2: drop the unterminated call and keep the preceding reasoning"},
                    {"verdict": "match", "note": "P2: v2 drops the partial trailing call, keeps reasoning"}),
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
        "kimi_k2": ("<think>thinking but stream ends",
                    M, {"verdict": "match", "note": "verify against v1 kimi reasoning finish() at capture time"}),
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
        # `<|eot|>` already ended the turn, so the trailing `<|eom|>` closes nothing.
        "muse_glimmer": ("<|start|>assistant to=user<|message|>I will check that. <|eot|><|eom|>",
                         V_MUSE,
                         {"verdict": "match", "note": "an orphan terminator outside any routed message is stripped, never emitted as content"}),
     }),

    ("empty_args",
     "A tool call with an empty argument object {}. Policy P3 — empty args serialize to {}. This is also covered in: TOOLCALLING.streamv2.6.a.",
     ["P3"],
     [{"kind": "tool_call", "name": "get_weather", "arguments": {}}],
     {"starting_state": "None", "tool_output_mode": "Native", "named_tool": None},
     {
        "gemma4": ("<|tool_call>call:get_weather{}<tool_call|>", M, M),
        "qwen3": ("<tool_call>\n<function=get_weather>\n</function>\n</tool_call>", M, M),
        "kimi_k2": ("<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{}<|tool_call_end|><|tool_calls_section_end|>", M, M),
        "muse_glimmer": ("<|start|>assistant to=get_weather<|message|><atem:function_calls>\n<atem:invoke name=\"get_weather\">\n</atem:invoke>\n</atem:function_calls><|eom|>",
                         V_MUSE, M),
     }),

    ("tool_no_close",
     "A single tool call whose body is complete but the close marker never arrives before EOF. Best-effort recovery emits the complete call at finish. This is also covered in: TOOLCALLING.streamv2.5.a.",
     [],
     [{"kind": "tool_call", "name": "get_weather", "arguments": {"city": "Paris"}}],
     {"starting_state": "None", "tool_output_mode": "Native", "named_tool": None},
     {
        "gemma4": ("<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>}",
                   {"verdict": "match", "note": "body complete; recover the call at finish"},
                   {"verdict": "match", "note": "body complete; recover the call at finish"}),
        "qwen3": ("<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>",
                  {"verdict": "match", "note": "body complete; recover at finish"},
                  D("DROP", "the complete call body produces no events when the outer close is absent")),
        "kimi_k2": ("<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{\"city\": \"Paris\"}",
                    {"verdict": "match", "note": "body complete; recover at finish"},
                    D("DROP", "the complete call body produces no events when the outer close is absent")),
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
        "gemma4": ("<|channel>thought\nI should check. <|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>}<tool_call|> now answer<channel|>",
                   D("LEAK", "the reasoning extractor consumes to `<channel|>`, so the nested `<|tool_call>...<tool_call|>` leaks into reasoning_content and the call is dropped; break-out recovery not implemented"),
                   D("LEAK", "v1 reasoning runs to `<channel|>`, swallowing the nested tool markup into one reasoning event; the call is lost")),
        "qwen3": ("<think>I should check. <tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call> now answer</think>",
                  D("LEAK", "captured: reasoning(I should check. ) | tool_call(get_weather) | reasoning( now answer) — the `</think>` closes only after the nested call, so the tool markup leaks into reasoning and the call is dropped"),
                  D("LEAK", "captured: text(Sure. ) | reasoning(I should check. ) | tool_call(get_weather) | reasoning( now answer) | text( Here you go.) — v1 reasoning consumes to `</think>`, leaking the nested tool markup")),
        "kimi_k2": ("<think>I should check. <|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{\"city\": \"Paris\"}<|tool_call_end|><|tool_calls_section_end|> now answer</think>",
                    D("LEAK", "captured: reasoning(I should check. ) | tool_call(get_weather) | text( now answer</think>) — the tool section nested in `<think>...</think>` leaks into reasoning and the call is dropped"),
                    D("LEAK", "captured: reasoning(I should check. ) | text(Sure. ) | tool_call(get_weather) | text( now answer</think> Here you g) — v1 reasoning consumes to `</think>`, leaking the nested section")),
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
        "gemma4": ("Sure. <|channel>thought\nI should check. <|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>}<tool_call|> now answer<channel|> Here you go.",
                   D("LEAK", "the reasoning extractor consumes to `<channel|>`, leaking the nested tool markup into reasoning_content and dropping the call; the visible text survives on both sides"),
                   D("LEAK", "v1 reasoning runs to `<channel|>`, swallowing the nested tool markup; the call is lost")),
        "qwen3": ("Sure. <think>I should check. <tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call> now answer</think> Here you go.",
                  D("LEAK", "captured: text(Sure. ) | reasoning(I should check. ) | tool_call(get_weather) | reasoning( now answer) | text( Here you go.) — `</think>` closes only after the nested call, so the tool markup leaks into reasoning and the call is dropped"),
                  D("LEAK", "captured: text(Sure. ) | reasoning(I should check. ) | tool_call(get_weather) | reasoning( now answer) | text( Here you go.) — v1 reasoning consumes to `</think>`, leaking the nested tool markup")),
        "kimi_k2": ("Sure. <think>I should check. <|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{\"city\": \"Paris\"}<|tool_call_end|><|tool_calls_section_end|> now answer</think> Here you go.",
                    D("LEAK", "captured: reasoning(I should check. ) | text(Sure. ) | tool_call(get_weather) | text( now answer</think> Here you g) — the nested tool section leaks into reasoning and the call is dropped"),
                    D("LEAK", "captured: reasoning(I should check. ) | text(Sure. ) | tool_call(get_weather) | text( now answer</think> Here you g) — v1 reasoning consumes to `</think>`, leaking the nested section")),
        "muse_glimmer": ("<|start|>assistant to=user<|message|>Sure. <|eom|><|start|>assistant to=self<|message|>I should check. to=get_weather<|message|><atem:function_calls>\n<atem:invoke name=\"get_weather\">\n<atem:parameter name=\"city\">Paris</atem:parameter>\n</atem:invoke>\n</atem:function_calls><|eom|><|start|>assistant to=self<|message|> now answer<|eom|><|start|>assistant to=user<|message|> Here you go.<|eom|>",
                         V_MUSE,
                         {"verdict": "match", "note": "the bare-header recovery is latched to a reasoning body, so it fires here and stays off inside the surrounding `to=user` messages"}),
     }),

    ("two_adjacent_reason_spans",
     "Two reasoning spans with nothing between them, then the answer. The single `reasoning_text` field every batch parser exposes can only concatenate them, so the separator is part of the contract: adjacent spans join with a newline. The counterpart is already covered by `reason_after_tool` / `reason_interleaved`, where two spans separated by a call must NOT join — a parser that always joins invents a newline the model never emitted, and one that never joins loses the batch parity every engine has.",
     [],
     [{"kind": "reasoning", "text": "first\nsecond"}, {"kind": "text", "text": "done"}],
     {
        "gemma4": ("<|channel>thought\nfirst<channel|><|channel>thought\n\nsecond<channel|>done", M,
                   {"verdict": "match", "note": "the split path merges both spans into one reasoning event, which is what this scenario expects"}),
        "qwen3": ("<think>first</think><think>\nsecond</think>done", M,
                  {"verdict": "match", "note": "adjacent reasoning runs coalesce into one event (I8)"}),
        "kimi_k2": ("<think>first</think><think>\nsecond</think>done", M,
                    {"verdict": "match", "note": "adjacent reasoning runs coalesce into one event (I8)"}),
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
     "A required choice returns an ARRAY, so multiple calls are that mode's ordinary shape. Both must surface as separate ordered events with distinct indices. Same array as 50.c but with NOTHING pre-filled, so guided mode starts outside reasoning rather than in visible-only — a different entry into the same payload.",
     [],
     [{"kind": "tool_call", "name": "get_weather", "arguments": {"city": "Paris"}},
      {"kind": "tool_call", "name": "run", "arguments": {"cmd": "git log"}}],
     {"starting_state": "None", "tool_output_mode": "GuidedJson", "named_tool": None},
     every_family(GUIDED_TWO_CALLS, GUIDED_UNSUPPORTED,
                  {"verdict": "match", "note": "two DIFFERENT tools in one array, ordered"})),

    ("guided_json_partial_calls",
     "A guided array where one element is not a call (no `name`), with nothing pre-filled. All-or-nothing, as in 51.b: the whole payload surfaces as text and no call is dispatched, because extracting a call from a document that failed validation would fail OPEN on a side-effecting action.",
     ["P2"],
     [{"kind": "text", "text": '[{"name": "get_weather", "arguments": {"city": "Paris"}}, {"arguments": {"city": "Tokyo"}}]'}],
     {"starting_state": "None", "tool_output_mode": "GuidedJson", "named_tool": None},
     {
        "qwen3": ('[{"name": "get_weather", "arguments": {"city": "Paris"}}, {"arguments": {"city": "Tokyo"}}]',
                  D("UNSUPPORTED", "vLLM base case doesn't emit guided JSON; conformance captures native XML only"),
                  {"verdict": "match", "note": "one invalid element voids the whole array; payload surfaces as text"}),
        "gemma4": ('[{"name": "get_weather", "arguments": {"city": "Paris"}}, {"arguments": {"city": "Tokyo"}}]', M, M),
        "kimi_k2": ('[{"name": "get_weather", "arguments": {"city": "Paris"}}, {"arguments": {"city": "Tokyo"}}]', M, M),
     }),

    ("guided_json_list_with_broken_element",
     "A guided array whose SECOND element is not valid JSON — the payload is `[<valid call>, <broken>]`, which is what a constrained decode produces when it is cut off partway through a later call. Output is the whole payload as text and no call, same as 31.c but reached differently: there the array parsed and one element failed to convert, here the array does not parse at all, so per-element recovery never gets a chance. Both land on all-or-nothing, which is the point — a half-validated array must not dispatch the half that looked fine.",
     ["P2"],
     [{"kind": "text", "text": '[{"name": "get_weather", "arguments": {"city": "Paris"}}, {"name": "run", "arguments": {"cmd": ]'}],
     {"starting_state": "None", "tool_output_mode": "GuidedJson", "named_tool": None},
     {
        "qwen3": ('[{"name": "get_weather", "arguments": {"city": "Paris"}}, {"name": "run", "arguments": {"cmd": ]',
                  D("UNSUPPORTED", "vLLM base case doesn't emit guided JSON; conformance captures native XML only"),
                  {"verdict": "match", "note": "the array itself fails to parse; nothing is dispatched"}),
        "gemma4": ('[{"name": "get_weather", "arguments": {"city": "Paris"}}, {"name": "run", "arguments": {"cmd": ]', M, M),
        "kimi_k2": ('[{"name": "get_weather", "arguments": {"city": "Paris"}}, {"name": "run", "arguments": {"cmd": ]', M, M),
     }),

    # --- Guided decoding: the SURROUNDINGS, not just the payload -----------------
    # Every guided case above varies the PAYLOAD and delivers it bare. Nothing
    # varied what sits AROUND it, and that is precisely where every guided defect
    # in this surface has been found: prose before a thought surfaced the model's
    # private reasoning as the answer, a narrated invoke swallowed the payload, an
    # orphan closer leaked, and markup bracketing the payload lost the call. Those
    # are pinned by unit tests; without these cases the corpus reads green through
    # all of them.
    ("guided_json_after_reasoning",
     "The guided BASELINE that was missing: a normal thought, then the constrained payload. Every other guided case starts at the payload, so nothing pinned the ordinary shape where the model reasons first and the backend constrains only the call. This is the case the surroundings group contrasts with.",
     [],
     [{"kind": "reasoning", "text": "checking"},
      {"kind": "tool_call", "name": "get_weather", "arguments": {"city": "Paris"}}],
     {"starting_state": "None", "tool_output_mode": "GuidedJson", "named_tool": None},
     guided_surroundings(
         lambda fam: f"{control_tokens(fam)[0]}checking{control_tokens(fam)[1]}{GUIDED_ONE_CALL}",
         "reasoning closes, then the guided payload dispatches")),

    ("guided_json_marker_inside_argument",
     "A control marker of the family's OWN grammar inside a guided argument VALUE. Once the payload has opened, a marker is argument DATA and must survive byte-exact (`I7`) — re-reading it as a channel token corrupts the call the tool receives while looking like a successful dispatch.",
     ["P3"],
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
         lambda fam: f"{control_tokens(fam)[0]}I'll use {control_tokens(fam)[2]} next{control_tokens(fam)[1]}{GUIDED_ONE_CALL}",
         "narrated markup stripped, thought preserved, payload survives")),

    ("guided_json_prose_before_reasoning",
     "Visible prose, THEN a thought, then the payload. Every other guided case opens its thought at byte 0; when prose came first the run latched the payload buffer and the model's private thinking was surfaced to the user as the answer.",
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
         lambda fam: f"{control_tokens(fam)[3]}{GUIDED_ONE_CALL}",
         "orphan tool closer stripped; the call still dispatches")),

    ("guided_json_invalid_call",
     "Guided decoding emits JSON that is well-formed but is NOT a tool call — no `name`, so there is nothing to dispatch. Policy P2: surface the payload as visible content rather than dropping it or erroring. Dropping it would lose the model's entire output; erroring would fail a request the user can still read.",
     ["P2"],
     [{"kind": "text", "text": '{"unexpected": "shape"}'}],
     {"starting_state": "None", "tool_output_mode": "GuidedJson", "named_tool": None},
     {
        "qwen3": ('{"unexpected": "shape"}',
                  D("UNSUPPORTED", "vLLM base case doesn't emit guided JSON; conformance captures native XML only"),
                  {"verdict": "match", "note": "P2: unparseable-as-a-call guided payload is surfaced as text"}),
        "gemma4": ('{"unexpected": "shape"}', M, M),
        "kimi_k2": ('{"unexpected": "shape"}', M, M),
     }),

    ("guided_json_malformed_json",
     "Guided decoding emits JSON that does not PARSE — a truncated object, which is what a constrained decode looks like when the token budget runs out mid-payload. Distinct from the wrong-shape case: there the JSON was valid and merely not a call. Policy P2: surface the bytes as visible content. Dropping them loses the output silently, and erroring fails a request whose text is still readable.",
     ["P2"],
     [{"kind": "text", "text": '{"name": "get_weather", "arguments": {"city": "Par'}],
     {"starting_state": "None", "tool_output_mode": "GuidedJson", "named_tool": None},
     {
        "qwen3": ('{"name": "get_weather", "arguments": {"city": "Par',
                  D("UNSUPPORTED", "vLLM base case doesn't emit guided JSON; conformance captures native XML only"),
                  {"verdict": "match", "note": "P2: unparseable guided payload is surfaced as text, not dropped"}),
        "gemma4": ('{"name": "get_weather", "arguments": {"city": "Par', M, M),
        "kimi_k2": ('{"name": "get_weather", "arguments": {"city": "Par', M, M),
     }),

    ("prefilled_reasoning_with_tool",
     "Reasoning channel is pre-filled by the generation prompt (policy P5), so the stream begins inside <think> with no opener. The model emits: reasoning tail -> closer -> tool call.",
     ["P5"],
     [{"kind": "reasoning", "text": "checking weather"},
      {"kind": "tool_call", "name": "get_weather", "arguments": {"city": "Paris"}}],
     {"starting_state": "Reasoning", "tool_output_mode": "Native", "named_tool": None},
     {"finish_reason": "stop"},
     {
        "qwen3": ("checking weather</think><tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>",
                  D("UNSUPPORTED", "vLLM base case doesn't set a starting channel state; conformance captures default generation only"),
                  {"verdict": "match", "note": "Dynamo v2 unified parser with starting_state=Reasoning"}),
        "gemma4": ("checking weather<channel|><|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>}<tool_call|>", M, M),
        "kimi_k2": ("checking weather</think><|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{\"city\": \"Paris\"}<|tool_call_end|><|tool_calls_section_end|>", M, M),
     }),

    ("prefilled_reasoning_then_text_then_tool",
     "Reasoning is pre-filled, the model closes it, writes VISIBLE prose, and only then calls a tool. All three channels in one prefilled stream. The prose must surface as text, not be swept into the reasoning span it follows nor into the call it precedes — the boundary on each side is a different marker, and a prefilled stream has no opener to anchor the first one.",
     ["P5"],
     [{"kind": "reasoning", "text": "weighing options"},
      {"kind": "text", "text": "Here's what I found: "},
      {"kind": "tool_call", "name": "get_weather", "arguments": {"city": "Paris"}}],
     {"starting_state": "Reasoning", "tool_output_mode": "Native", "named_tool": None},
     {
        "qwen3": ("weighing options</think>Here's what I found: <tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>",
                  D("UNSUPPORTED", "vLLM base case doesn't set a starting channel state; conformance captures default generation only"),
                  {"verdict": "match", "note": "reasoning -> text -> call, all three ordered in one prefilled stream"}),
        "gemma4": ("weighing options<channel|>Here's what I found: <|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>}<tool_call|>", M, M),
        "kimi_k2": ("weighing options</think>Here's what I found: <|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{\"city\": \"Paris\"}<|tool_call_end|><|tool_calls_section_end|>", M, M),
     }),

    ("prefilled_reasoning_then_text",
     "Reasoning is pre-filled, the model closes it and answers in prose with NO tool call — the ordinary shape of a prefilled request that needs no tool. Pins that closing a prefilled thought returns the stream to visible content rather than leaving it in reasoning, which would swallow the whole answer.",
     ["P5"],
     [{"kind": "reasoning", "text": "no tool needed"},
      {"kind": "text", "text": "The answer is 42."}],
     {"starting_state": "Reasoning", "tool_output_mode": "Native", "named_tool": None},
     {
        "qwen3": ("no tool needed</think>The answer is 42.",
                  D("UNSUPPORTED", "vLLM base case doesn't set a starting channel state; conformance captures default generation only"),
                  {"verdict": "match", "note": "closing a prefilled thought returns to visible content"}),
        "gemma4": ("no tool needed<channel|>The answer is 42.", M, M),
        "kimi_k2": ("no tool needed</think>The answer is 42.", M, M),
     }),

    ("prefilled_response_with_guided_json",
     "Response channel is pre-filled (the prompt opened visible content), so the stream skips reasoning entirely and emits only tool calls as guided JSON. Same payload as 30.b under a different starting state; identical output, since Response only changes how reasoning markers are read and there are none.",
     ["P5"],
     [{"kind": "tool_call", "name": "get_weather", "arguments": {"city": "Paris"}}],
     {"starting_state": "Response", "tool_output_mode": "GuidedJson", "named_tool": None},
     {"finish_reason": "stop"},
     every_family(GUIDED_ONE_CALL,
                  D("UNSUPPORTED", "vLLM base case doesn't set a starting channel state or use guided JSON"),
                  {"verdict": "match", "note": "Dynamo v2 unified parser with starting_state=Response and tool_output_mode=GuidedJson{named_tool=None}"})),

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

    ("prefilled_response_with_tool",
     "Response channel is pre-filled (the prompt opened visible content), so the stream skips reasoning entirely: the leading `output` is visible CONTENT with no opening marker, then a native-XML tool call. The leading text is generated output and must surface as a text event — routing it to reasoning is the regression, and it is what a reasoning-first split does when nothing told it the response channel was already open. Parses identically under starting_state=None (compare 8.a `text_before_tool`) — no reasoning markers here, so Response has nothing to suppress; 50.d is the case that isolates it.",
     ["P5"],
     [{"kind": "text", "text": "output"},
      {"kind": "tool_call", "name": "get_weather", "arguments": {"city": "Paris"}}],
     {"starting_state": "Response", "tool_output_mode": "Native", "named_tool": None},
     {"finish_reason": "stop"},
     {
        "qwen3": ("output<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>",
                  D("UNSUPPORTED", "vLLM base case doesn't set a starting channel state; conformance captures default generation only"),
                  {"verdict": "match", "note": "Dynamo v2 unified parser with starting_state=Response and tool_output_mode=Native"}),
        "gemma4": ("output<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>}<tool_call|>", M, M),
        "kimi_k2": ("output<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{\"city\": \"Paris\"}<|tool_call_end|><|tool_calls_section_end|>",
                    M,
                    D("MERGE", "the split path has no starting-state signal, so the leading visible `output` is swept into reasoning_content instead of surfacing as text")),
     }),




    ("prefilled_reasoning_redundant_opener",
     "Reasoning is pre-filled, and the backend ALSO re-emits the `<think>` opener the prompt already wrote. Exactly one such echo is consumed rather than leaked into reasoning_content; a second would be stray markup and stripped (I3). This is the only case where a prefilled stream legitimately carries an opener.",
     [],
     [{"kind": "reasoning", "text": "checking weather"}, {"kind": "tool_call", "name": "get_weather", "arguments": {"city": "London"}}],
     {"starting_state": "Reasoning", "tool_output_mode": "Native", "named_tool": None},
     {"finish_reason": "stop"},
     {
        "gemma4": ("<|channel>thought\nchecking weather<channel|><|tool_call>call:get_weather{city:<|\"|>London<|\"|>}<tool_call|>", M, M),
        "qwen3": ("<think>checking weather</think><tool_call>\n<function=get_weather>\n<parameter=city>\nLondon\n</parameter>\n</function>\n</tool_call>", M, M),
        "kimi_k2": ("<think>checking weather</think><|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{\"city\": \"London\"}<|tool_call_end|><|tool_calls_section_end|>", M, M),
     }),



    ("prefilled_reasoning_truncated",
     "Reasoning is pre-filled and the token budget runs out mid tool call — the input is truncated, which is what finish_reason=length MEANS on the wire. Policy P2: keep the completed reasoning, drop the incomplete call, no error and no leaked markup.",
     ["P2"],
     [{"kind": "reasoning", "text": "analyzing data"}],
     {"starting_state": "Reasoning", "tool_output_mode": "Native", "named_tool": None},
     {"finish_reason": "length"},
     {
        "gemma4": ("analyzing data<channel|><|tool_call>call:get_weather{city:<|\"|>Par",
                   D("ERROR", "native Gemma4UnifiedParser finish() returns a hard Err on a partial call rather than recovering"),
                   {"verdict": "match", "note": "P2: drop the partial trailing call, keep the prefilled reasoning"}),
        "qwen3": ("analyzing data</think><tool_call>\n<function=get_weather>\n<parameter=city>\nPar",
                  {"verdict": "match", "note": "P2: drop the unterminated call and keep prefilled output"},
                  {"verdict": "match", "note": "P2: v2 drops the partial trailing call, keeps the prefilled reasoning"}),
        "kimi_k2": ("analyzing data</think><|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{\"city\": \"Par",
                    {"verdict": "match", "note": "P2: drop the unterminated call and keep prefilled output"},
                    {"verdict": "match", "note": "P2: v2 drops the partial trailing call, keeps the prefilled reasoning"}),
     }),



    ("prefilled_response_guided_json_two_calls",
     "Guided decoding with a required choice returns an ARRAY, so the multi-call shape is the array's normal case, not an edge one. Both calls must surface as separate ordered events with distinct indices — collapsing them, or emitting only the first, silently drops work the model asked for. Same array as 30.c under a different starting state; see 50.d for the case where Response actually changes the parse.",
     ["P5"],
     [{"kind": "tool_call", "name": "get_weather", "arguments": {"city": "Paris"}},
      {"kind": "tool_call", "name": "run", "arguments": {"cmd": "git log"}}],
     {"starting_state": "Response", "tool_output_mode": "GuidedJson", "named_tool": None},
     {"finish_reason": "stop"},
     every_family(GUIDED_TWO_CALLS,
                  D("UNSUPPORTED", "vLLM base case doesn't set a starting channel state or use guided JSON"),
                  {"verdict": "match", "note": "two DIFFERENT tools in one array, ordered"})),

    ("prefilled_response_guided_json_partial_calls",
     "A guided array where ONE element is not a call (no `name`). The whole payload surfaces as text and NO call is dispatched — deliberately all-or-nothing, not best-effort per element. A tool call is a side effect, so extracting one from a document that failed validation is failing OPEN: the client would execute a call the parser could not fully verify. Text loses nothing, since the raw payload stays visible.",
     ["P2"],
     [{"kind": "text", "text": '[{"name": "get_weather", "arguments": {"city": "Paris"}}, {"arguments": {"city": "Tokyo"}}]'}],
     {"starting_state": "Response", "tool_output_mode": "GuidedJson", "named_tool": None},
     {"finish_reason": "stop"},
     {
        "qwen3": ('[{"name": "get_weather", "arguments": {"city": "Paris"}}, {"arguments": {"city": "Tokyo"}}]',
                  D("UNSUPPORTED", "vLLM base case doesn't set a starting channel state or use guided JSON"),
                  {"verdict": "match", "note": "one invalid element voids the whole array; payload surfaces as text"}),
        "gemma4": ('[{"name": "get_weather", "arguments": {"city": "Paris"}}, {"arguments": {"city": "Tokyo"}}]', M, M),
        "kimi_k2": ('[{"name": "get_weather", "arguments": {"city": "Paris"}}, {"arguments": {"city": "Tokyo"}}]', M, M),
     }),

    ("prefilled_response_reasoning_markers_literal",
     "The ONLY case where starting_state=Response is observable. Response says the prompt already opened VISIBLE content, so this stream has no reasoning channel at all and `<think>`/`</think>` are ordinary characters the model happened to write — they must reach the user as text, markers and all. Every other 50.*/51.* case has no reasoning markers in its input, which is why they parse identically under starting_state=None (50.a matches 8.a, 50.b matches 30.b, 50.c matches 30.c, 51.b matches 31.c); this one does not.",
     ["P5"],
     # The literal text is the family's OWN reasoning markers, so the golden is
     # filled per family (below) rather than hardcoding one grammar's.
     [{"kind": "text", "text": None},
      {"kind": "tool_call", "name": "get_weather", "arguments": {"city": "Paris"}}],
     {"starting_state": "Response", "tool_output_mode": "Native", "named_tool": None},
     {"finish_reason": "stop"},
     {
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
     }),

    ("prefilled_response_truncated",
     "The response channel is pre-filled and the token budget runs out mid tool call. Policy P2: the visible prose already emitted survives, the incomplete call is dropped, nothing leaks as text.",
     ["P2"],
     [{"kind": "text", "text": "Working on it... "}],
     {"starting_state": "Response", "tool_output_mode": "Native", "named_tool": None},
     {"finish_reason": "length"},
     {
        "gemma4": ("Working on it... <|tool_call>call:get_weather{city:<|\"|>Par",
                   D("ERROR", "native Gemma4UnifiedParser finish() returns a hard Err on a partial call rather than recovering"),
                   {"verdict": "match", "note": "P2: keep the leading visible prose, drop the partial call"}),
        "qwen3": ("Working on it... <tool_call>\n<function=get_weather>\n<parameter=city>\nPar",
                  {"verdict": "match", "note": "P2: keep leading prose and drop the unterminated call"},
                  {"verdict": "match", "note": "P2: v2 keeps the leading prose and drops the partial call"}),
        "kimi_k2": ("Working on it... <|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{\"city\": \"Par",
                    {"verdict": "match", "note": "P2: keep leading prose and drop the unterminated call"},
                    {"verdict": "match", "note": "P2: v2 keeps the leading prose and drops the partial call"}),
     }),
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
    "bare_opener": (lambda pay, fam: f"{invoke_header_prefix(fam)}{pay}",
                    "a bare invoke HEADER before the payload, never terminated", False),
}


def _guided_product():
    """Every (payload x surrounding) crossing that says something distinct.

    `clean` x `valid` is `30.a`/`30.b` and `clean` x the malformed payloads is
    `31.a`-`31.d`; those already exist, so the predicate drops them rather than
    emitting a duplicate under a second name.
    """
    out = []
    for pay_name, (payload, want_args) in GUIDED_PAYLOADS.items():
        dispatches = want_args is not None
        for sur_name, (wrap, sur_desc, strips_tail) in GUIDED_SURROUNDS.items():
            # `clean` is already authored as 30.a/30.b and 31.a-31.d. The
            # `valid` payload crossings are also already authored by hand
            # (guided_json_tool_open_before_payload / _tool_close_after_payload /
            # _wrapped_in_tool_markup) — generating them produced 3 scenarios x 3
            # families = 9 cases with byte-identical (input, init, golden). A
            # duplicate is worse than a gap: it inflates the case count while
            # testing nothing new, and two names for one behaviour drift apart.
            if sur_name == "clean" or pay_name == "valid":
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
                guided_surroundings(
                    lambda fam, w=wrap, pl=payload: w(pl, fam),
                    note,
                    fill=(None if dispatches else
                          (lambda fam, pl=payload, st=strips_tail: pl.rstrip() if st else pl)),
                ),
            ))
    return out


EDGE += _guided_product()


# Group 4 (TC Malformed envelope) was a LABELLED group with zero cases, and the
# degenerate shape below had none either: no row anywhere pinned that control
# markup ALONE emits nothing. Both are native, so the input is per family.
EDGE += [
    ("tool_markup_only_emits_nothing",
     "The whole generated output is control markup and nothing else — a stray close with no "
     "block ever opened. Everything is stripped, so the parser emits NO events at all. Until "
     "this case there was no row with an empty golden: every case asserted something was "
     "produced, so 'markup alone leaks nothing' (`I3`) was never actually pinned.",
     ["P2"],
     [],
     {"starting_state": "None", "tool_output_mode": "Native", "named_tool": None},
     {"finish_reason": "stop"},
     by_family(lambda fam: control_tokens(fam)[3],
               D("UNSUPPORTED", "vLLM base case does not capture a markup-only turn"),
               {"verdict": "match", "note": "orphan close stripped; nothing to emit"})),

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
     by_family(lambda fam: f"{control_tokens(fam)[2]}still thinking about it",
               D("UNSUPPORTED", "vLLM base case does not capture an unterminated envelope"),
               {"verdict": "match", "note": "P2: unterminated envelope drops its content"})),
]


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
]


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
         lambda fam: f"{invoke_header_prefix(fam)}{r_reason(fam, 'secret')}{GUIDED_ONE_CALL}",
         "a bare invoke header before a thought must not borrow the thought's terminator")),

    ("guided_json_narrated_prefix_inside_reasoning",
     "The model NARRATES an invoke header inside its thought and never terminates it, so the only "
     "`>` in reach belongs to the thought's own closer. The header is literal text the model wrote, "
     "so it is stripped and the surrounding thought survives intact — it must not swallow the closer "
     "and it must not survive into the reasoning the user sees.",
     ["I3"],
     [{"kind": "reasoning", "text": "I'll call get_weather"},
      {"kind": "tool_call", "name": "get_weather", "arguments": {"city": "Paris"}}],
     {"starting_state": "None", "tool_output_mode": "GuidedJson", "named_tool": None},
     {"finish_reason": "stop"},
     guided_surroundings(
         lambda fam: f"{r_reason(fam, 'I' + chr(39) + 'll call ' + invoke_header_prefix(fam) + 'get_weather')}{GUIDED_ONE_CALL}",
         "a narrated invoke header inside a thought is stripped, closer and thought intact")),
]


def _entry(spec, fam):
    """Resolve a vllm/dynamo verdict spec (single or per-family) for `fam`."""
    if isinstance(spec, dict) and set(spec) <= set(FAMILIES):
        return spec[fam]
    return spec


def _init_is_request_scoped(init):
    """True when a case declares a request mode a pre-unified build cannot see."""
    init = init or {}
    return (init.get("tool_output_mode", "Native") != "Native"
            or init.get("starting_state", "None") != "None")
def _vllm_entry(spec, fam):
    """`_entry` for the vLLM column, annotating a family no released vLLM can parse.

    An authored verdict that already carries a note said something specific about
    this case; only the shared, noteless `M` is replaced.
    """
    entry = _entry(spec, fam)
    caveat = VLLM_UNCAPTURABLE.get(fam)
    return caveat if caveat is not None and not entry.get("note") else entry


def build_cases(fam):
    """Every CLEAN + EDGE scenario for one family, keyed by case id."""
    cases = {}
    for name, desc, policy, segs, vllm, dynamo in CLEAN:
        cid = f"UNIFIED.{name}.{fam}"
        cases[cid] = {
            "description": desc,
            "policy": policy,
            "input": render_input(fam, segs),
            "golden": golden_of(segs),
            "expect": {"vllm": _vllm_entry(vllm, fam), "dynamo": _entry(dynamo, fam)},
            "init": {"starting_state": "None", "tool_output_mode": "Native", "named_tool": None},
            "finish_reason": "stop",
        }
    for edge_case in EDGE:
        # Support both 6-tuple (legacy) and 7-tuple (stream_config) formats
        if len(edge_case) == 6:
            name, desc, policy, golden, init, per_fam = edge_case
            stream_config = {"finish_reason": "stop"}
        else:
            name, desc, policy, golden, init, stream_config, per_fam = edge_case

        cid = f"UNIFIED.{name}.{fam}"
        inp, vllm, dynamo, *rest = per_fam[fam]
        g = json.loads(json.dumps(golden))  # deep copy
        if rest:
            # Fill the ONE `None` placeholder in the golden with this family's
            # value. It may be an argument value (a marker-looking string that has
            # to survive byte-exact, 12.a) or a whole text payload (the family's
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
        if _init_is_request_scoped(init):
            vllm = GUIDED_UNSUPPORTED if init.get("tool_output_mode") != "Native" else D(
                "UNSUPPORTED",
                "vLLM base case does not set a starting channel state; conformance "
                "captures default generation only",
            )
        cases[cid] = {
            "description": desc,
            "policy": policy,
            "input": inp,
            "golden": g,
            "expect": {"vllm": _vllm_entry(vllm, fam), "dynamo": dynamo},
            "init": init,
            "finish_reason": stream_config.get("finish_reason", "stop"),
        }
    return cases


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
        f"# {fam} grammar: {GRAMMAR_NOTE[fam]}",
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
        lines.append("    input: |-")
        for ln in c["input"].split("\n"):
            lines.append(f"      {ln}")
        lines.append(f"    golden: {json.dumps(c['golden'], ensure_ascii=False)}")
        lines.append(f"    expect: {json.dumps(c['expect'], ensure_ascii=False)}")
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
