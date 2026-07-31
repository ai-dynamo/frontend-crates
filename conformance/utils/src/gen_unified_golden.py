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
}


# --- grammar renderers: one semantic segment -> that family's raw text --------

def r_reason(fam, text):
    if fam == "gemma4":
        return f"<|channel>thought\n{text}<channel|>"
    return f"<think>{text}</think>"


def r_tool(fam, name, key, val, idx):
    if fam == "gemma4":
        return f"<|tool_call>call:{name}{{{key}:<|\"|>{val}<|\"|>}}<tool_call|>"
    if fam == "qwen3":
        return (f"<tool_call>\n<function={name}>\n<parameter={key}>\n"
                f"{val}\n</parameter>\n</function>\n</tool_call>")
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
            out.append(s[1])
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
     "Reasoning then visible content, no tool call (baseline).",
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
     {"gemma4": M, "qwen3": M,
      "kimi_k2": D("LOSS", "kimi config stays in a tool state and SUPPRESSES trailing text -> arbitrary content dropped; violates best-effort recovery (preserve visible prose, conformance/README.md:142)")},
     {"gemma4": M, "qwen3": M,
      "kimi_k2": {"verdict": "match", "note": "P1 resolved by the v2 recovery contract: preserve trailing prose. Verify v2 kimi_k2 at capture time"}}),

    # --- Group 2: multiple tool calls (streamv2.2) — tool-only, green everywhere ---
    ("two_calls",
     "Two tool calls back-to-back, no reasoning (streamv2.2.a). Both must surface as ordered events.",
     [], [("tool", "f", "x", "1"), ("tool", "g", "y", "2")], M, M),
    ("two_calls_same_name",
     "The same tool called twice with different args (streamv2.2.d). Both calls are distinct events.",
     [], [("tool", "get_weather", "city", "Paris"), ("tool", "get_weather", "city", "Tokyo")], M, M),

    # --- Group 3: no tool call ---
    ("text_only",
     "Plain answer, no reasoning and no tool call (streamv2.3). Pure content passthrough.",
     [], [("text", "The answer is 42, no tools needed.")], M, M),

    # --- Group 7: argument fidelity (streamv2.7) ---
    ("arg_unicode",
     "Unicode + spaces in a string argument value (streamv2.7.b). Preserved exactly (I7).",
     [], [("tool", "get_weather", "city", "São Paulo 東京")], M, M),

    # --- Group 8: content / narration position (streamv2.8) ---
    ("text_before_tool",
     "Visible text before a single tool call, no reasoning (streamv2.8.a).",
     [], [("text", "On it: "), ("tool", "get_weather", "city", "Paris")], M, M),
    ("text_sandwich",
     "Visible text both before and after a tool call (streamv2.8.c).",
     [], [("text", "Before. "), ("tool", "get_weather", "city", "Paris"), ("text", " After.")], M, M),
    ("text_between_calls",
     "Visible text between two tool calls (streamv2.8.d).",
     [], [("tool", "f", "x", "1"), ("text", " then "), ("tool", "g", "y", "2")], M, M),
    ("narrated_calls",
     "Multiple tool calls with visible narration between each — tool_call -> text -> tool_call -> text -> tool_call. The agentic pattern: call, narrate, call again. Every call and every inter-call text span must surface as its own ordered event.",
     [], [("tool", "get_weather", "city", "Paris"), ("text", " then I'll run "),
          ("tool", "f", "x", "1"), ("text", " and "), ("tool", "g", "y", "2")], M, M),

    # --- Group 10: reasoning span (reasoning-only; REASONING.2/6) ---
    ("reason_only",
     "A reasoning span with no visible answer and no tool call (REASONING.2.a).",
     [], [("reason", "just thinking, no answer")], M, M),
    ("two_reason_spans",
     "Two reasoning spans separated by visible text, no tool call (REASONING.6.a). Streaming keeps both spans in order; batch merges them.",
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
     {
        "gemma4": ("<|channel>thought\nok<channel|><|tool_call>call:get_weather{city:<|\"|>Par",
                   D("ERROR", "native Gemma4UnifiedParser finish() returns a hard Err -> erroring is the opposite of best-effort recovery"),
                   {"verdict": "match", "note": "P2: drop the partial trailing call, keep the preceding reasoning, never error/leak (TOOLCALLING batch.5.e)"}),
        "qwen3": ("<think>ok</think><tool_call>\n<function=get_weather>\n<parameter=city>\nPar",
                  {"verdict": "match", "note": "hypothesis: qwen3 tool parser drops the unterminated call; verify at capture time"},
                  {"verdict": "match", "note": "P2: v2 drops the partial trailing call, keeps reasoning"}),
        "kimi_k2": ("<think>ok</think><|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{\"city\": \"Par",
                    {"verdict": "match", "note": "hypothesis: kimi tool parser drops the unterminated call; verify at capture time"},
                    {"verdict": "match", "note": "P2: v2 drops the partial trailing call, keeps reasoning"}),
     }),

    ("reason_unterminated",
     "Stream ends while still inside reasoning (no close marker). Open reasoning is promoted at finish, not dropped and not leaked as text.",
     [],
     [{"kind": "reasoning", "text": "thinking but stream ends"}],
     {
        "gemma4": ("<|channel>thought\nthinking but stream ends",
                   M, {"verdict": "match", "note": "verify against v1 gemma4 reasoning finish() at capture time"}),
        "qwen3": ("<think>thinking but stream ends",
                  M, {"verdict": "match", "note": "verify against v1 qwen3 reasoning finish() at capture time"}),
        "kimi_k2": ("<think>thinking but stream ends",
                    M, {"verdict": "match", "note": "verify against v1 kimi reasoning finish() at capture time"}),
     }),

    ("arg_marker_in_string",
     "A close-marker-looking sequence INSIDE a string arg value. Invariant I7 — the value is data, preserved exactly, not truncated at the marker-looking substring.",
     [],
     [{"kind": "tool_call", "name": "run", "arguments": {"cmd": None}}],  # cmd filled per family below
     {
        "gemma4": ("<|tool_call>call:run{cmd:<|\"|>git log }<tool_call|> --oneline<|\"|>}<tool_call|>",
                   D("ARG_MISMATCH", "char-by-char streamed-arg coercion truncates args at the marker-looking boundary (regression class #48702/#47977)"),
                   {"verdict": "match", "note": "emit-on-close typing sees the whole balanced value; find_tool_call_end_position_gemma4 ignores <tool_call|> inside <|\"|> strings"},
                   "git log }<tool_call|> --oneline"),
        "qwen3": ("<tool_call>\n<function=run>\n<parameter=cmd>\ngit log </tool_call> --oneline\n</parameter>\n</function>\n</tool_call>",
                  {"verdict": "match", "note": "hypothesis: value ends at the real `\\n</parameter>`; an embedded `</tool_call>` is data. Verify at capture time"},
                  {"verdict": "match", "note": "v2 reads the parameter value up to `</parameter>`; embedded `</tool_call>` preserved"},
                  "git log </tool_call> --oneline"),
        "kimi_k2": ("<|tool_calls_section_begin|><|tool_call_begin|>functions.run:0<|tool_call_argument_begin|>{\"cmd\": \"git log <|tool_call_end|> --oneline\"}<|tool_call_end|><|tool_calls_section_end|>",
                    {"verdict": "match", "note": "hypothesis: JSON string value survives an embedded `<|tool_call_end|>`. Verify at capture time"},
                    {"verdict": "match", "note": "v2 parses the JSON arg blob; the marker inside the string is data"},
                    "git log <|tool_call_end|> --oneline"),
     }),

    ("orphan_close_after_prose",
     "Prose followed by an orphan close marker with no matching open. Best-effort recovery — the prose stays as content, the orphan marker is stripped, nothing leaks.",
     [],
     [{"kind": "text", "text": "I will check that. "}],
     {
        "gemma4": ("I will check that. <tool_call|>",
                   D("LEAK", "vLLM/SGLang leak the orphan close marker into content as the whole tail (TOOLCALLING_CASES.md 5.g)"),
                   D("LEAK", "LIVE finding: v2 gemma4 leaks a lone <tool_call|> end marker into content — with no matching <|tool_call> open the scanner treats it as text. Best-effort-recovery gap (should strip per TOOLCALLING 5.g).")),
        "qwen3": ("I will check that. </tool_call>",
                  {"verdict": "match", "note": "hypothesis: an orphan `</tool_call>` with no open is stripped. Verify at capture time"},
                  {"verdict": "match", "note": "hypothesis: v2 qwen3 strips the orphan close. Verify at capture time"}),
        "kimi_k2": ("I will check that. <|tool_call_end|>",
                    {"verdict": "match", "note": "hypothesis: an orphan `<|tool_call_end|>` with no section is stripped. Verify at capture time"},
                    {"verdict": "match", "note": "hypothesis: v2 kimi strips the orphan close. Verify at capture time"}),
     }),

    ("empty_args",
     "A tool call with an empty argument object {} (streamv2.6.a). Policy P3 — empty args serialize to {}.",
     ["P3"],
     [{"kind": "tool_call", "name": "get_weather", "arguments": {}}],
     {
        "gemma4": ("<|tool_call>call:get_weather{}<tool_call|>", M, M),
        "qwen3": ("<tool_call>\n<function=get_weather>\n</function>\n</tool_call>", M, M),
        "kimi_k2": ("<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{}<|tool_call_end|><|tool_calls_section_end|>", M, M),
     }),

    ("tool_no_close",
     "A single tool call whose body is complete but the close marker never arrives before EOF (streamv2.5.a). Best-effort recovery emits the complete call at finish.",
     [],
     [{"kind": "tool_call", "name": "get_weather", "arguments": {"city": "Paris"}}],
     {
        "gemma4": ("<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>}",
                   {"verdict": "match", "note": "body complete; recover the call at finish"},
                   {"verdict": "match", "note": "hypothesis: v2 emits the complete call at finish; verify at capture"}),
        "qwen3": ("<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>",
                  {"verdict": "match", "note": "body complete; recover at finish"},
                  {"verdict": "match", "note": "hypothesis: v2 recovers; verify at capture"}),
        "kimi_k2": ("<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{\"city\": \"Paris\"}",
                    {"verdict": "match", "note": "body complete; recover at finish"},
                    {"verdict": "match", "note": "hypothesis: v2 recovers; verify at capture"}),
     }),

    # --- Group 12: adversarial nesting (a marker of one channel inside another) ---
    ("reason_markup_in_arg",
     "'Tool call contains reasoning' — a reasoning-channel marker sits inside a QUOTED tool-arg value. This is NOT a leak: a leak is control markup surfacing in visible content or reasoning, but here the markup is a tool ARGUMENT VALUE (data bound for the function, inside the grammar's string delimiters), so by I7 the parser preserves it byte-exact. The gemma4 native UnifiedParser confirms this golden exactly. Failure mode: a reasoning-first pipeline extracts the `<think>`/`<|channel>` from inside the arg BEFORE tool parsing, hoisting it into a spurious reasoning event and corrupting the arg to empty.",
     [],
     [{"kind": "tool_call", "name": "log", "arguments": {"note": None}}],  # note filled per family
     {
        "gemma4": ("<|tool_call>call:log{note:<|\"|><|channel>thought\nreconsider<channel|><|\"|>}<tool_call|>",
                   D("ARG_MISMATCH", "the reasoning extractor lifts the `<|channel>...<channel|>` out of the arg value before tool parsing, so the logged note no longer matches golden"),
                   D("MERGE", "v1 reasoning runs first over the whole stream and pulls the arg's embedded `<|channel>...<channel|>` into a leading reasoning event, corrupting the tool arg"),
                   "<|channel>thought\nreconsider<channel|>"),
        "qwen3": ("<tool_call>\n<function=log>\n<parameter=note>\n<think>reconsider</think>\n</parameter>\n</function>\n</tool_call>",
                  D("ARG_MISMATCH", "hypothesis: the `<think>...</think>` inside the parameter value is extracted as reasoning first, corrupting the arg. Verify at capture time"),
                  D("MERGE", "hypothesis: v1 reasoning lifts the embedded `<think>` out of the arg. Verify at capture time"),
                  "<think>reconsider</think>"),
        "kimi_k2": ("<|tool_calls_section_begin|><|tool_call_begin|>functions.log:0<|tool_call_argument_begin|>{\"note\": \"<think>reconsider</think>\"}<|tool_call_end|><|tool_calls_section_end|>",
                    D("ARG_MISMATCH", "hypothesis: the `<think>` inside the JSON string arg is extracted as reasoning first, corrupting the arg. Verify at capture time"),
                    D("MERGE", "hypothesis: v1 reasoning lifts the embedded `<think>` out of the JSON arg. Verify at capture time"),
                    "<think>reconsider</think>"),
     }),

    ("tool_in_reason",
     "'Reasoning contains tool call' — a well-formed tool-call envelope nested INSIDE a reasoning span. This is the OPPOSITE of reason_markup_in_arg: a reasoning span is opaque TEXT, not a quoted data region, so a real tool-call marker inside it IS structural. Best-effort recovery breaks out of reasoning, emits the call, and resumes reasoning after its close (golden: reason -> call -> reason). Leaking the raw `<|tool_call>...<tool_call|>` into reasoning_content, or dropping the call, is the regression — which is what every reasoning-first engine does here.",
     [],
     [{"kind": "reasoning", "text": "I should check. "},
      {"kind": "tool_call", "name": "get_weather", "arguments": {"city": "Paris"}},
      {"kind": "reasoning", "text": " now answer"}],
     {
        "gemma4": ("<|channel>thought\nI should check. <|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>}<tool_call|> now answer<channel|>",
                   D("LEAK", "the reasoning extractor consumes to `<channel|>`, so the nested `<|tool_call>...<tool_call|>` leaks into reasoning_content and the call is dropped; break-out recovery not implemented"),
                   D("LEAK", "v1 reasoning runs to `<channel|>`, swallowing the nested tool markup into one reasoning event; the call is lost")),
        "qwen3": ("<think>I should check. <tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call> now answer</think>",
                  D("LEAK", "hypothesis: the `</think>` closes only after the nested call, so the tool markup leaks into reasoning and the call is dropped. Verify at capture time"),
                  D("LEAK", "hypothesis: v1 reasoning consumes to `</think>`, leaking the nested tool markup. Verify at capture time")),
        "kimi_k2": ("<think>I should check. <|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{\"city\": \"Paris\"}<|tool_call_end|><|tool_calls_section_end|> now answer</think>",
                    D("LEAK", "hypothesis: the tool section nested in `<think>...</think>` leaks into reasoning and the call is dropped. Verify at capture time"),
                    D("LEAK", "hypothesis: v1 reasoning consumes to `</think>`, leaking the nested section. Verify at capture time")),
     }),

    ("reason_markup_in_arg_with_text",
     "reason_markup_in_arg (tool arg value contains reasoning markup, I7 data) WITH visible narration before and after the call. All three channels at once: leading text -> tool call whose arg holds reasoning markup -> trailing text. Golden keeps the visible text as text, the call as a call, and the markup byte-exact in the arg. A reasoning-first pipeline both corrupts the arg (extracting the embedded reasoning) and can reorder/misroute the surrounding text.",
     [],
     [{"kind": "text", "text": "Logging now: "},
      {"kind": "tool_call", "name": "log", "arguments": {"note": None}},  # filled per family
      {"kind": "text", "text": " done."}],
     {
        "gemma4": ("Logging now: <|tool_call>call:log{note:<|\"|><|channel>thought\nreconsider<channel|><|\"|>}<tool_call|> done.",
                   D("ARG_MISMATCH", "the reasoning extractor lifts the `<|channel>...<channel|>` out of the arg before tool parsing; the note no longer matches and the surrounding text can shift"),
                   D("MERGE", "v1 reasoning hoists the arg's embedded `<|channel>...<channel|>` ahead of the visible text and corrupts the tool arg"),
                   "<|channel>thought\nreconsider<channel|>"),
        "qwen3": ("Logging now: <tool_call>\n<function=log>\n<parameter=note>\n<think>reconsider</think>\n</parameter>\n</function>\n</tool_call> done.",
                  D("ARG_MISMATCH", "hypothesis: the `<think>` inside the parameter value is extracted as reasoning first, corrupting the arg. Verify at capture time"),
                  D("MERGE", "hypothesis: v1 reasoning lifts the embedded `<think>` out of the arg and ahead of the text. Verify at capture time"),
                  "<think>reconsider</think>"),
        "kimi_k2": ("Logging now: <|tool_calls_section_begin|><|tool_call_begin|>functions.log:0<|tool_call_argument_begin|>{\"note\": \"<think>reconsider</think>\"}<|tool_call_end|><|tool_calls_section_end|> done.",
                    D("ARG_MISMATCH", "hypothesis: the `<think>` inside the JSON string arg is extracted as reasoning first, corrupting the arg. Verify at capture time"),
                    D("MERGE", "hypothesis: v1 reasoning lifts the embedded `<think>` out of the JSON arg and ahead of the text. Verify at capture time"),
                    "<think>reconsider</think>"),
     }),

    ("tool_in_reason_with_text",
     "tool_in_reason (a tool call nested inside a reasoning span, break-out recovery) WITH visible narration before and after the reasoning span. All three channels at once: leading text -> reasoning that wraps a real call -> trailing text. Golden: text -> reason -> call -> reason -> text. Engines that treat reasoning as opaque-until-close leak the nested tool markup into reasoning_content and drop the call.",
     [],
     [{"kind": "text", "text": "Sure. "},
      {"kind": "reasoning", "text": "I should check. "},
      {"kind": "tool_call", "name": "get_weather", "arguments": {"city": "Paris"}},
      {"kind": "reasoning", "text": " now answer"},
      {"kind": "text", "text": " Here you go."}],
     {
        "gemma4": ("Sure. <|channel>thought\nI should check. <|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>}<tool_call|> now answer<channel|> Here you go.",
                   D("LEAK", "the reasoning extractor consumes to `<channel|>`, leaking the nested tool markup into reasoning_content and dropping the call; the visible text survives on both sides"),
                   D("LEAK", "v1 reasoning runs to `<channel|>`, swallowing the nested tool markup; the call is lost")),
        "qwen3": ("Sure. <think>I should check. <tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call> now answer</think> Here you go.",
                  D("LEAK", "hypothesis: `</think>` closes only after the nested call, so the tool markup leaks into reasoning and the call is dropped. Verify at capture time"),
                  D("LEAK", "hypothesis: v1 reasoning consumes to `</think>`, leaking the nested tool markup. Verify at capture time")),
        "kimi_k2": ("Sure. <think>I should check. <|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{\"city\": \"Paris\"}<|tool_call_end|><|tool_calls_section_end|> now answer</think> Here you go.",
                    D("LEAK", "hypothesis: the nested tool section leaks into reasoning and the call is dropped. Verify at capture time"),
                    D("LEAK", "hypothesis: v1 reasoning consumes to `</think>`, leaking the nested section. Verify at capture time")),
     }),
]


def _entry(spec, fam):
    """Resolve a vllm/dynamo verdict spec (single or per-family) for `fam`."""
    if isinstance(spec, dict) and set(spec) <= set(FAMILIES):
        return spec[fam]
    return spec


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
            "expect": {"vllm": _entry(vllm, fam), "dynamo": _entry(dynamo, fam)},
        }
    for name, desc, policy, golden, per_fam in EDGE:
        cid = f"UNIFIED.{name}.{fam}"
        inp, vllm, dynamo, *rest = per_fam[fam]
        g = json.loads(json.dumps(golden))  # deep copy
        if rest:  # fill the per-family arg value into the tool_call event's None-valued key
            for ev in g:
                args = ev.get("arguments")
                if args and any(v is None for v in args.values()):
                    fk = next(k for k, v in args.items() if v is None)
                    args[fk] = rest[0]
                    break
        cases[cid] = {
            "description": desc,
            "policy": policy,
            "input": inp,
            "golden": g,
            "expect": {"vllm": vllm, "dynamo": dynamo},
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
