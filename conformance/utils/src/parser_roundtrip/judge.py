# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Judge each case: x = decode(output minus terminal) goes through the Unified
parser, its ordered events fold into one assistant message, and the template
renders that message after the request history. The rendered turn, without its
end of turn, must equal the generation prompt's tail plus x. A parse that loses
or alters text cannot render back to x."""

from __future__ import annotations

import functools
import json
import os
import re
import subprocess
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from .bundle import Bundle, template_kwargs
from .trace import Case

REPO = Path(__file__).resolve().parents[4]
PARSER_COMMAND = (
    "cargo",
    "run",
    "--quiet",
    "-p",
    "dynamo-conformance-fixtures-v2",
    "--example",
    "parser_roundtrip",
)
PROBE_CONTENT = "__roundtrip_content__"
PROBE_REASONING = "__roundtrip_reasoning__"
PROBE_TOOL = "__roundtrip_tool__"
TAG = re.compile(r"<[^<>\s]+>")
WORD = re.compile(r"\w+")
MAX_LISTED_IDS = 20
# The event kinds the parser_roundtrip example writes.
KINDS = ("reasoning", "text", "tool_call")
# Crate aliases (`unified_registry!` in parsers/v2/src/unified/mod.rs).
FAMILY_ALIASES = {"qwen3_coder": "qwen3", "kimi-k3": "kimi_k3"}
# Dynamo's reasoning openers by family (`prompt_injected_reasoning_start` in
# lib/llm/src/preprocessor.rs); every other family uses `<think>`.
REASONING_OPENERS = {
    "gemma4": ("<|channel>thought", "<|channel>"),
    "kimi_k3": ("<|open|>think<|sep|>",),
    "muse_glimmer": ("assistant to=self<|message|>",),
}
THINKING_KEYS = ("thinking", "enable_thinking", "thinking_mode")
BOOL_WORDS = ("1", "true", "on", "yes", "0", "false", "off", "no")


@dataclass(frozen=True)
class Shape:
    """How the template renders a final assistant turn under one set of kwargs."""

    reasoning_opener: str | None  # markup right before rendered reasoning
    reasoning_closer: str | None  # text between rendered reasoning and content
    renders_reasoning: bool
    renders_calls: bool
    plain_prefix: str | None  # turn text before content, without reasoning
    empty_prefix: str | None  # turn text before content, with an empty reasoning block
    end_of_turn: tuple[str, ...]  # longest first


@dataclass(frozen=True)
class Prepared:
    case: Case
    prompt: str  # rendered generation prompt
    text: str  # x
    reasoning_disabled: bool
    job: dict[str, Any]


def open_private(path: Path) -> TextIO:
    """Open `path` for writing with mode 0600, never through a symlink."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW
    handle = os.fdopen(os.open(path, flags, 0o600), "w", encoding="utf-8")
    os.fchmod(handle.fileno(), 0o600)
    return handle


def last_markup(text: str, bundle: Bundle) -> int:
    """Where the last added token or <tag> in `text` starts; -1 without one."""
    vocab = bundle.tokenizer.get_added_vocab()
    starts = [text.rfind(token) for token in vocab if token]
    starts += [match.start() for match in TAG.finditer(text)]
    return max(starts, default=-1)


@functools.cache
def probe(bundle: Bundle, kwargs_json: str) -> Shape:
    kwargs = json.loads(kwargs_json)
    tool = {
        "type": "function",
        "function": {
            "name": PROBE_TOOL,
            "parameters": {"type": "object", "properties": {}},
        },
    }
    call = {"type": "function", "function": {"name": PROBE_TOOL, "arguments": {}}}

    def render(
        extra: dict[str, Any] | None, tools: list[dict[str, Any]] | None = None
    ) -> str | None:
        """A user turn, then an assistant turn unless `extra` is None; None when
        the template rejects it."""
        messages = [{"role": "user", "content": "probe"}]
        if extra is not None:
            messages.append({"role": "assistant", "content": PROBE_CONTENT, **extra})
        try:
            return bundle.render(messages, tools, False, kwargs)
        except Exception:  # a template rejecting a probe does not render that field
            return None

    base, plain = render(None), render({})
    reasoned = render({"reasoning_content": PROBE_REASONING}) or ""

    def turn(text: str | None) -> str:
        """The assistant turn after the user-only render; empty when unknown."""
        if base is None or text is None or not text.startswith(base):
            return ""
        return text[len(base) :]

    plain_turn, reasoned_turn = turn(plain), turn(reasoned)
    plain_prefix = None
    if PROBE_CONTENT in plain_turn:
        plain_prefix = plain_turn[: plain_turn.index(PROBE_CONTENT)]
    opener = closer = empty_prefix = None
    if PROBE_REASONING in reasoned:
        at = reasoned.index(PROBE_REASONING)
        before = reasoned[:at].rstrip()
        start = last_markup(before, bundle)
        if start >= 0 and not any(char.isspace() for char in before[start:]):
            opener = before[start:]
        after = at + len(PROBE_REASONING)
        if PROBE_CONTENT in reasoned[after:]:
            closer = reasoned[after : reasoned.index(PROBE_CONTENT, after)]
    if PROBE_REASONING in reasoned_turn and closer is not None:
        empty_prefix = reasoned_turn[: reasoned_turn.index(PROBE_REASONING)] + closer
    eos = [bundle.decode([token]) for token in bundle.eos_token_ids]
    ends = {*eos, *(text + "\n" for text in eos)}
    if plain is not None and PROBE_CONTENT in plain:
        ends.add(plain[plain.rindex(PROBE_CONTENT) + len(PROBE_CONTENT) :])
    with_call, without = render({"tool_calls": [call]}, [tool]), render({}, [tool])
    return Shape(
        reasoning_opener=opener,
        reasoning_closer=closer,
        renders_reasoning=PROBE_REASONING in reasoned,
        renders_calls=with_call is not None
        and without is not None
        and with_call != without,
        plain_prefix=plain_prefix,
        empty_prefix=empty_prefix,
        end_of_turn=tuple(sorted(ends - {""}, key=len, reverse=True)),
    )


def shape_for(bundle: Bundle, request: dict[str, Any]) -> Shape:
    return probe(bundle, json.dumps(template_kwargs(request), sort_keys=True))


def unmodeled_thinking_flag(kwargs: dict[str, Any]) -> bool:
    """Whether Dynamo rewrites the request's thinking flag before it renders
    (`normalize_thinking_aliases` in lib/llm/src/preprocessor.rs). The first of
    `thinking`, `enable_thinking` and `thinking_mode` that holds a bool, string or
    number decides. Dynamo turns a string or number in the first two, or a bool,
    number or on/off word in `thinking_mode`, into bool `thinking` and
    `enable_thinking`, so the recorded prompt is not the one Dynamo serves."""
    for key in THINKING_KEYS:
        value = kwargs.get(key)
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


def reasoning_disabled(family: str, kwargs: dict[str, Any]) -> bool:
    """Whether Dynamo's Unified path serves reasoning as content for this request
    (`request_disables_reasoning_with_family` in lib/llm/src/preprocessor.rs),
    given a flag Dynamo does not rewrite (`unmodeled_thinking_flag`): `thinking`,
    else `enable_thinking`, decides when it is a bool. A deployment's default
    thinking mode is not modeled."""
    flags = (kwargs.get(key) for key in ("thinking", "enable_thinking"))
    thinking = next((flag for flag in flags if isinstance(flag, bool)), None)
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


def starting_state(family: str, prompt: str, disabled: bool) -> str:
    """The channel the generation prompt leaves open, as Dynamo infers it
    (`prompt_reasoning_prefill_for_parsers` in lib/llm/src/preprocessor.rs):
    reasoning when the prompt ends with the family's opener, even with thinking
    off. Gemma 4 with thinking off starts in its response channel. Gemma 4's
    partial thought header is not modeled."""
    if family == "gemma4" and disabled:
        return "response"
    return "reasoning" if opens_reasoning(family, prompt) else "none"


def skip(case: Case, reason: str, detail: str | None = None) -> dict[str, Any]:
    result = {"id": case.id, "verdict": f"skip:{reason}"}
    return {**result, "detail": detail} if detail else result


def function_tools(request: dict[str, Any]) -> list[dict[str, Any]]:
    """The request's function tools, the only kind the parser is given. Tools a
    Kimi-style system message declares, which Dynamo also passes, are not."""
    tools = request.get("tools") or []
    return [
        tool
        for tool in tools
        if isinstance(tool, dict) and isinstance(tool.get("function"), dict)
    ]


def prepare(case: Case, bundle: Bundle, family: str) -> Prepared | dict[str, Any]:
    """The parser job for a case, or its skip result."""
    family = FAMILY_ALIASES.get(family, family)
    if case.finish_reason == "length":
        return skip(case, "incomplete")
    request = case.request
    if request.get("stop"):
        # The frontend trims a matched stop string, which x does not model.
        return skip(case, "stop-strings")
    choice = request.get("tool_choice")
    if choice == "required" or isinstance(choice, dict):
        # Dynamo serves these with guided JSON, not the native markup parsed here,
        # and turns thinking off for a named choice under Kimi K3.
        return skip(case, "guided-tool-choice")
    kwargs = template_kwargs(request)
    if unmodeled_thinking_flag(kwargs):
        return skip(case, "unmodeled-thinking-flag")
    try:
        prompt = bundle.render_request(request, True)
    except Exception as exc:  # the reference template cannot render the request
        return skip(case, "template-error", type(exc).__name__)
    output = case.output_token_ids
    terminal = case.terminal(bundle.eos_token_ids)
    text = bundle.decode(output[: len(output) - len(terminal)])
    disabled = reasoning_disabled(family, kwargs)
    job = {
        "id": case.id,
        "text": text,
        "family": family,
        "starting_state": starting_state(family, prompt, disabled),
        "tools": function_tools(request),
    }
    return Prepared(case, prompt, text, disabled, job)


def message_from_events(
    events: list[dict[str, Any]], disabled: bool
) -> tuple[dict[str, Any], str]:
    """The message serving assembles from ordered events, and the events' kinds
    in order with runs collapsed. Content is None when empty; with thinking off,
    reasoning is content where it occurred, without its markers."""
    parts: dict[str, list[str]] = {"reasoning": [], "text": []}
    calls: list[dict[str, Any]] = []
    kinds: list[str] = []
    for event in events:
        kind = event["kind"]
        if kind not in KINDS:
            raise ValueError(f"unknown event kind {kind!r}")
        if kind == "reasoning" and disabled:
            kind = "text"
        if kinds[-1:] != [kind]:
            kinds.append(kind)
        if kind == "tool_call":
            calls.append({key: event.get(key) for key in ("name", "arguments", "id")})
        else:
            parts[kind].append(event["text"])
    message = {
        "content": "".join(parts["text"]) or None,
        "reasoning_content": "".join(parts["reasoning"]),
        "tool_calls": calls,
    }
    return message, ",".join(kinds)


def representable(order: str) -> bool:
    """Whether one message can hold events in this order. Templates render
    reasoning first, so reasoning after text or a call cannot round-trip; text
    after a call folds into content, as serving does, and is judged."""
    kinds = [kind for kind in order.split(",") if kind]
    return "reasoning" not in kinds[1:]


def assistant_turn(message: dict[str, Any]) -> dict[str, Any] | None:
    """The parsed message as a history turn; None when arguments are not a JSON object."""
    # Empty content goes back as "", as vLLM normalizes a null before rendering;
    # a template that prints it unguarded (GLM-4.7) would otherwise write "None".
    turn: dict[str, Any] = {"role": "assistant", "content": message["content"] or ""}
    if message["reasoning_content"]:
        turn["reasoning_content"] = message["reasoning_content"]
    calls = []
    for call in message["tool_calls"]:
        try:
            arguments = json.loads(call["arguments"])
        except json.JSONDecodeError:
            return None
        if not isinstance(arguments, dict):
            return None
        function = {"name": call["name"], "arguments": arguments}
        identifier = {"id": call["id"]} if call.get("id") else {}
        calls.append({**identifier, "type": "function", "function": function})
    if calls:
        turn["tool_calls"] = calls
    return turn


def common_prefix(left: str, right: str) -> int:
    size = 0
    for a, b in zip(left, right):
        if a != b:
            break
        size += 1
    return size


def signature(raw: str, rendered: str) -> str:
    """Both sides of the first difference with word runs folded to `w`, so
    failures of one kind share a key."""
    return json.dumps(
        [WORD.sub("w", raw[:16]), WORD.sub("w", rendered[:16])], ensure_ascii=False
    )


def fold(
    turn: dict[str, Any], shape: Shape, generation: str, text: str, rendered: str
) -> tuple[str, str, str | None]:
    """The compared (raw, rendered) and a note when the template cannot render
    what the model wrote; raw keeps x's tail, so positions map back to x."""
    raw, plain = generation + text, shape.plain_prefix
    opener, closer = shape.reasoning_opener, shape.reasoning_closer
    # A turn without reasoning always renders after `plain`, so neither a missing
    # nor an empty reasoning block renders back.
    if "reasoning_content" not in turn and plain and not raw.startswith(plain):
        if shape.empty_prefix and raw.startswith(shape.empty_prefix):
            return plain + raw[len(shape.empty_prefix) :], rendered, "empty_reasoning"
        if (
            len(plain) > len(generation)
            and plain.startswith(generation)
            and not (opener and text.lstrip().startswith(opener))
        ):
            return plain + text, rendered, "template_inserted_prefix"
    # The template closes reasoning that the model ended with its stop token,
    # unless the reasoning already holds a closer the parser did not honor.
    if (
        "reasoning_content" in turn
        and not turn["content"]
        and "tool_calls" not in turn
        and closer
        and closer.strip() not in turn["reasoning_content"]
        and rendered.endswith(closer)
        and not raw.endswith(closer)
    ):
        return raw, rendered[: -len(closer)], "unclosed_reasoning"
    return raw, rendered, None


def judge(prepared: Prepared, bundle: Bundle, parsed: dict[str, Any]) -> dict[str, Any]:
    case, job = prepared.case, prepared.job
    result: dict[str, Any] = {
        "id": case.id,
        "starting_state": job["starting_state"],
        "reasoning_disabled": prepared.reasoning_disabled,
    }
    if case.request.get("tool_choice") == "none":
        # Serving drops calls under tool_choice none; this judges the parsed calls.
        result["tool_choice"] = "none"
    if "error" in parsed:
        return {
            **result,
            "verdict": "fail",
            "signature": "parser_error",
            "detail": str(parsed["error"])[:200],
        }
    message, order = message_from_events(parsed["events"], prepared.reasoning_disabled)
    result["order"] = order
    result["incomplete_calls"] = parsed.get("incomplete_calls", 0)
    if not representable(order):
        return {**result, **skip(case, "not-representable")}
    turn = assistant_turn(message)
    if turn is None:
        return {**result, "verdict": "fail", "signature": "invalid_arguments"}
    shape = shape_for(bundle, case.request)
    if "tool_calls" in turn and not shape.renders_calls:
        return {**result, **skip(case, "template-renders-no-tool-calls")}
    if "reasoning_content" in turn and not shape.renders_reasoning:
        return {**result, **skip(case, "template-renders-no-reasoning")}
    request, prompt = case.request, prepared.prompt
    probe_turn = {"role": "assistant", "content": PROBE_CONTENT}
    try:
        history = bundle.render_request(request, False)
        full = bundle.render_request(request, False, turn)
        appends = full.startswith(history) or bundle.render_request(
            request, False, probe_turn
        ).startswith(history)
    except Exception as exc:  # the reference template cannot render the turn
        return {**result, **skip(case, "template-error", type(exc).__name__)}
    # Without a turn rendered after the history there is no boundary to compare at.
    if not prompt.startswith(history) or not appends:
        return {**result, **skip(case, "template-does-not-append-turns")}
    start = len(history) if full.startswith(history) else common_prefix(prompt, full)
    rendered = full[start:]
    rendered = next(
        (rendered[: -len(end)] for end in shape.end_of_turn if rendered.endswith(end)),
        rendered,
    )
    generation = prompt[start:]
    raw, rendered, note = fold(turn, shape, generation, prepared.text, rendered)
    if note:
        result["note"] = note
    if raw == rendered:
        return {**result, "verdict": "pass"}
    at = common_prefix(raw, rendered)
    # The first difference in x; negative inside the generation prompt's tail.
    in_text = at - (len(raw) - len(prepared.text))
    # Snippets come from the unfolded raw, which is generation plus x.
    in_raw = len(generation) + in_text
    unfolded = generation + prepared.text
    return {
        **result,
        "verdict": "fail",
        "signature": signature(raw[at:], rendered[at:]),
        "at": in_text,
        "raw": unfolded[max(0, in_raw - 40) : in_raw + 80],
        "rendered": rendered[max(0, at - 40) : at + 80],
    }


def run_parser(
    jobs: list[dict[str, Any]], command: Sequence[str], stderr_path: Path
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """The parser's header line and its answers by case ID. Its stderr goes to
    `stderr_path` (0600): a panic or error message can quote model text."""
    stdin = "".join(json.dumps(job, ensure_ascii=False) + "\n" for job in jobs)
    with open_private(stderr_path) as stderr:
        completed = subprocess.run(
            list(command),
            input=stdin,
            stdout=subprocess.PIPE,
            stderr=stderr,
            encoding="utf-8",
            cwd=REPO,
        )
    if completed.returncode:
        raise RuntimeError(
            f"the parser exited {completed.returncode}; stderr is in {stderr_path}"
        )
    # Split on LF only: JSON keeps U+2028 literal, which splitlines() would break.
    lines = [json.loads(line) for line in completed.stdout.split("\n") if line]
    if not lines:
        raise RuntimeError("the parser wrote no header")
    answers = {answer["id"]: answer for answer in lines[1:]}
    if len(answers) != len(jobs) or any(job["id"] not in answers for job in jobs):
        raise RuntimeError(f"the parser answered {len(answers)} of {len(jobs)} jobs")
    return lines[0], answers


def run(
    cases: list[Case],
    bundle: Bundle,
    family: str,
    stderr_path: Path,
    command: Sequence[str] = PARSER_COMMAND,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    outcomes = [prepare(case, bundle, family) for case in cases]
    jobs = [item.job for item in outcomes if isinstance(item, Prepared)]
    header, answers = run_parser(jobs, command, stderr_path)
    results = [
        judge(item, bundle, answers[item.case.id])
        if isinstance(item, Prepared)
        else item
        for item in outcomes
    ]
    return header, results


def summarize(
    header: dict[str, Any], results: list[dict[str, Any]], meta: dict[str, Any]
) -> dict[str, Any]:
    groups: defaultdict[str, list[str]] = defaultdict(list)
    for result in results:
        if result["verdict"] == "fail":
            groups[result["signature"]].append(result["id"])
    notes = Counter(result["note"] for result in results if result.get("note"))
    orders = Counter(result["order"] for result in results if "order" in result)
    return {
        "schema": "parser-roundtrip-summary/v1",
        **meta,
        "parser": header,
        "cases": len(results),
        "counts": dict(sorted(Counter(r["verdict"] for r in results).items())),
        "notes": dict(sorted(notes.items())),
        "orders": dict(orders.most_common()),
        "incomplete_calls": sum(1 for r in results if r.get("incomplete_calls")),
        "failures": [
            {"signature": key, "count": len(ids), "ids": ids}
            for key, ids in sorted(groups.items(), key=lambda item: -len(item[1]))
        ],
    }


def markdown(summary: dict[str, Any]) -> str:
    def cell(text: str) -> str:
        return text.replace("|", "\\|").replace("`", "'")

    def listed(ids: list[str]) -> str:
        more = " …" if len(ids) > MAX_LISTED_IDS else ""
        return ", ".join(ids[:MAX_LISTED_IDS]) + more

    parser = summary["parser"]
    lines = [
        "# Parser round trip",
        "",
        f"{parser.get('parser')} {parser.get('version')}, family "
        f"`{summary.get('family')}`: {summary['cases']} cases, "
        f"{summary['incomplete_calls']} with incomplete calls.",
        "",
        "| verdict | cases |",
        "|---|---|",
        *(f"| {verdict} | {count} |" for verdict, count in summary["counts"].items()),
    ]
    if summary["orders"]:
        lines += [
            "",
            "Parsed event order, runs collapsed:",
            "",
            "| order | cases |",
            "|---|---|",
            *(
                f"| {order or '(none)'} | {n} |"
                for order, n in summary["orders"].items()
            ),
        ]
    if summary["notes"]:
        lines += [
            "",
            "Template-ambiguous turns judged after a fold:",
            "",
            "| note | cases |",
            "|---|---|",
            *(f"| {note} | {count} |" for note, count in summary["notes"].items()),
        ]
    if summary["failures"]:
        lines += [
            "",
            "## Failures by first difference",
            "",
            "| signature | cases | IDs |",
            "|---|---|---|",
            *(
                f"| `{cell(group['signature'])}` | {group['count']} | "
                f"{listed(group['ids'])} |"
                for group in summary["failures"]
            ),
        ]
    return "\n".join(lines) + "\n"
