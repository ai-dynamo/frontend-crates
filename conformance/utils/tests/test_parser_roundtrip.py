# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Round-trip judging on a toy byte-level bundle. Parser answers are given
directly; the parser_roundtrip Rust example carries its own tests. Skips when
transformers or tokenizers is missing, as in conformance CI."""

from __future__ import annotations

import dataclasses
import json
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("transformers")
pytest.importorskip("tokenizers")

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from tokenizers import Tokenizer, decoders, models, pre_tokenizers  # noqa: E402
from transformers import PreTrainedTokenizerFast  # noqa: E402

from parser_roundtrip.__main__ import private_dir  # noqa: E402
from parser_roundtrip.bundle import Bundle  # noqa: E402
from parser_roundtrip.judge import (  # noqa: E402
    Prepared,
    function_tools,
    judge,
    message_from_events,
    prepare,
    reasoning_disabled,
    run_parser,
    starting_state,
    summarize,
    unmodeled_thinking_flag,
)
from parser_roundtrip.trace import SCHEMA, Case, TraceError, parse_case  # noqa: E402

EOS = 256  # the toy tokenizer's only added token, after the 256 byte tokens
# Reasoning in <think> tags or a lone </think> without it, and a generation
# prompt that opens, closes or (when undefined) omits the thought by enable_thinking.
TEMPLATE = (
    "{%- for m in messages -%}<|{{ m.role }}|>"
    "{%- if m.role == 'assistant' -%}"
    "{%- if m.reasoning_content -%}<think>{{ m.reasoning_content }}</think>"
    "{%- else -%}</think>{%- endif -%}"
    "{{ m.content or '' }}"
    "{%- for call in m.tool_calls or [] -%}"
    "<call>{{ call.function.name }} {{ call.function.arguments | tojson }}</call>"
    "{%- endfor -%}"
    "{%- else -%}{{ m.content }}{%- endif -%}"
    "{{ '<eos>\\n' }}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}<|assistant|>"
    "{%- if enable_thinking is defined -%}"
    "{{ '<think>' if enable_thinking else '</think>' }}"
    "{%- endif -%}"
    "{%- endif -%}"
)
# Answers every job with its text as one text event, like a parser that finds nothing.
ECHO_PARSER = """#!{python}
import json, sys
header = {{"schema": "parser-roundtrip-events/v1", "parser": "echo", "version": "0"}}
print(json.dumps(header))
for line in sys.stdin:
    job = json.loads(line)
    events = [{{"kind": "text", "text": job["text"]}}] if job["text"] else []
    print(json.dumps({{"id": job["id"], "events": events, "incomplete_calls": 0}}))
"""


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("model")
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    vocab = {char: index for index, char in enumerate(alphabet)}
    tokenizer = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False, use_regex=False
    )
    tokenizer.decoder = decoders.ByteLevel()
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, eos_token="<eos>")
    fast.save_pretrained(path)
    (path / "chat_template.jinja").write_text(TEMPLATE)
    return path


@pytest.fixture(scope="module")
def bundle(model_dir: Path) -> Bundle:
    return Bundle.load(model_dir)


def trace(
    bundle: Bundle,
    text: str,
    finish: str = "stop",
    enable_thinking: bool | None = True,
) -> dict[str, Any]:
    request: dict[str, Any] = {"messages": [{"role": "user", "content": "hi"}]}
    if enable_thinking is not None:
        request["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
    return {
        "schema": SCHEMA,
        "id": f"thinking-{enable_thinking}-{finish}",
        "request": request,
        "generation": {
            "output_token_ids": bundle.tokenizer.encode(text, add_special_tokens=False)
            + [EOS],
            "finish_reason": finish,
        },
    }


def prepared(bundle: Bundle, case: Case, family: str = "qwen3") -> Prepared:
    outcome = prepare(case, bundle, family)
    assert isinstance(outcome, Prepared)
    return outcome


def reasoning(text: str) -> dict[str, Any]:
    return {"kind": "reasoning", "text": text}


def text(text: str) -> dict[str, Any]:
    return {"kind": "text", "text": text}


def call(name: str, arguments: str, identifier: str | None = None) -> dict[str, Any]:
    event = {"kind": "tool_call", "index": 0, "name": name, "arguments": arguments}
    return {**event, "id": identifier}


def answer(*events: dict[str, Any], incomplete_calls: int = 0) -> dict[str, Any]:
    return {"id": "c", "events": list(events), "incomplete_calls": incomplete_calls}


def test_faithful_parse_passes(bundle: Bundle) -> None:
    raw = 'Plan.</think>Hi.<call>run {"n": 1}</call>'
    job = prepared(bundle, parse_case(trace(bundle, raw)))
    assert job.text == raw
    assert job.job["starting_state"] == "reasoning" and not job.reasoning_disabled
    parsed = answer(reasoning("Plan."), text("Hi."), call("run", '{"n": 1}'))
    result = judge(job, bundle, parsed)
    assert result["verdict"] == "pass"
    assert (result["order"], result["incomplete_calls"]) == (
        "reasoning,text,tool_call",
        0,
    )


def test_lossy_parse_fails_at_the_first_difference(bundle: Bundle) -> None:
    case = parse_case(trace(bundle, "Plan.</think>a &amp; b"))
    parsed = answer(reasoning("Plan."), text("a & b"))
    result = judge(prepared(bundle, case), bundle, parsed)
    assert result["verdict"] == "fail"
    assert result["at"] == len("Plan.</think>a &")
    assert result["raw"].endswith("a &amp; b")
    assert result["rendered"].endswith("a & b")
    summary = summarize({}, [result, {"id": "other", "verdict": "pass"}], {})
    assert summary["counts"] == {"fail": 1, "pass": 1}
    assert summary["failures"] == [
        {"signature": result["signature"], "count": 1, "ids": [case.id]}
    ]


def test_a_dropped_call_fails_and_is_counted(bundle: Bundle) -> None:
    case = parse_case(trace(bundle, 'Plan.</think>Hi.<call>run {"n"'))
    parsed = answer(reasoning("Plan."), text("Hi."), incomplete_calls=1)
    result = judge(prepared(bundle, case), bundle, parsed)
    assert (result["verdict"], result["incomplete_calls"]) == ("fail", 1)
    assert result["at"] == len("Plan.</think>Hi.")
    assert summarize({}, [result], {})["incomplete_calls"] == 1


@pytest.mark.parametrize(
    ("finish", "request_fields", "verdict"),
    [
        ("length", {}, "skip:incomplete"),
        ("stop", {"stop": ["."]}, "skip:stop-strings"),
        ("stop", {"tool_choice": "required"}, "skip:guided-tool-choice"),
        (
            "stop",
            {"tool_choice": {"type": "function", "function": {"name": "run"}}},
            "skip:guided-tool-choice",
        ),
        (
            "stop",
            {"chat_template_kwargs": {"enable_thinking": "false"}},
            "skip:unmodeled-thinking-flag",
        ),
    ],
    ids=["length", "stop-strings", "required", "named", "string-thinking-flag"],
)
def test_unmodeled_outputs_are_skipped(
    bundle: Bundle, finish: str, request_fields: dict[str, Any], verdict: str
) -> None:
    obj = trace(bundle, "Pla", finish=finish)
    obj["request"].update(request_fields)
    case = parse_case(obj)
    assert prepare(case, bundle, "qwen3") == {"id": case.id, "verdict": verdict}


@pytest.mark.parametrize(
    ("family", "enable_thinking", "state"),
    [
        ("qwen3", True, "reasoning"),
        ("qwen3", False, "none"),
        ("qwen3", None, "none"),
        ("qwen3_coder", True, "reasoning"),
        ("gemma4", True, "none"),
        ("gemma4", None, "response"),
    ],
    ids=[
        "opened",
        "closed",
        "omitted",
        "alias",
        "gemma4-other-opener",
        "gemma4-not-thinking",
    ],
)
def test_starting_state_follows_the_generation_prompt(
    bundle: Bundle, family: str, enable_thinking: bool | None, state: str
) -> None:
    case = parse_case(trace(bundle, "Hi.", enable_thinking=enable_thinking))
    assert prepared(bundle, case, family).job["starting_state"] == state


@pytest.mark.parametrize(
    ("family", "prompt", "disabled", "state"),
    [
        ("glm47", "<|assistant|><think>\n", True, "reasoning"),
        ("gemma4", "<|turn>model\n<|channel>thought\n", False, "reasoning"),
        ("gemma4", "<|turn>model\n<|channel>", False, "reasoning"),
        ("kimi_k3", "<|open|> think <|sep|>\n", False, "reasoning"),
        ("muse_glimmer", "<|start|>assistant to=self<|message|>", False, "reasoning"),
        ("muse_glimmer", "<|start|>assistant<think>", False, "none"),
    ],
    ids=[
        "disabled-keeps-an-opened-thought",
        "gemma4-thought",
        "gemma4-channel",
        "kimi-k3-spaced",
        "muse",
        "muse-not-think",
    ],
)
def test_each_family_has_its_reasoning_opener(
    family: str, prompt: str, disabled: bool, state: str
) -> None:
    assert starting_state(family, prompt, disabled) == state


@pytest.mark.parametrize(
    ("enable_thinking", "raw", "parsed", "note"),
    [
        (None, "Hi.", answer(text("Hi.")), "template_inserted_prefix"),
        (True, "</think>Hi.", answer(text("Hi.")), "empty_reasoning"),
        (True, "Plan.", answer(reasoning("Plan.")), "unclosed_reasoning"),
    ],
    ids=["no-thought-opened", "empty-reasoning", "unclosed-reasoning"],
)
def test_template_ambiguous_turns_pass_with_a_note(
    bundle: Bundle,
    enable_thinking: bool | None,
    raw: str,
    parsed: dict[str, Any],
    note: str,
) -> None:
    case = parse_case(trace(bundle, raw, enable_thinking=enable_thinking))
    result = judge(prepared(bundle, case), bundle, parsed)
    assert (result["verdict"], result["note"]) == ("pass", note)
    assert summarize({}, [result], {})["notes"] == {note: 1}


def test_a_swallowed_closer_is_not_folded_away(bundle: Bundle) -> None:
    # Reasoning that kept the closer and the answer is a parse error, not a model
    # that stopped inside its thought.
    case = parse_case(trace(bundle, "Plan.</think>Hi."))
    result = judge(prepared(bundle, case), bundle, answer(reasoning("Plan.</think>Hi.")))
    assert result["verdict"] == "fail" and "note" not in result


def test_empty_content_renders_as_an_empty_string(bundle: Bundle) -> None:
    # A template that prints content unguarded must not see a null.
    template = TEMPLATE.replace("{{ m.content or '' }}", "{{ m.content }}")
    other = dataclasses.replace(bundle, chat_template=template)
    raw = 'Plan.</think><call>run {"n": 1}</call>'
    job = prepared(other, parse_case(trace(other, raw)))
    parsed = answer(reasoning("Plan."), call("run", '{"n": 1}'))
    assert judge(job, other, parsed)["verdict"] == "pass"


def test_a_fold_keeps_the_difference_position_in_x(bundle: Bundle) -> None:
    case = parse_case(trace(bundle, "a &amp; b", enable_thinking=None))
    result = judge(prepared(bundle, case), bundle, answer(text("a & b")))
    assert (result["verdict"], result["note"]) == ("fail", "template_inserted_prefix")
    assert result["at"] == len("a &")
    assert result["raw"] == "<|assistant|>a &amp; b"


def test_disabled_thinking_serves_reasoning_as_content(bundle: Bundle) -> None:
    # A parser that still reports reasoning: serving folds it into content in place.
    case = parse_case(trace(bundle, "Hi.", enable_thinking=False))
    job = prepared(bundle, case)
    assert job.reasoning_disabled
    result = judge(job, bundle, answer(reasoning("Hi.")))
    assert (result["verdict"], result["order"]) == ("pass", "text")
    kept = dataclasses.replace(job, reasoning_disabled=False)
    assert judge(kept, bundle, answer(reasoning("Hi.")))["verdict"] == "fail"


@pytest.mark.parametrize(
    ("disabled", "content", "reasoning_content", "order"),
    [
        (True, "A.B.C.", "", "text,tool_call"),
        (False, "B.", "A.C.", "reasoning,text,reasoning,tool_call"),
    ],
    ids=["disabled", "enabled"],
)
def test_events_fold_into_one_message_in_order(
    disabled: bool, content: str, reasoning_content: str, order: str
) -> None:
    events = [reasoning("A."), text("B."), reasoning("C."), call("run", "{}")]
    message, folded = message_from_events(events, disabled)
    assert (message["content"], message["reasoning_content"]) == (
        content,
        reasoning_content,
    )
    assert message["tool_calls"] == [{"name": "run", "arguments": "{}", "id": None}]
    assert folded == order
    assert message_from_events([call("run", "{}")], False)[0]["content"] is None


@pytest.mark.parametrize(
    "events",
    [(text("Hi."), reasoning("Plan.")), (call("run", "{}"), reasoning("Plan."))],
    ids=["text-first", "call-first"],
)
def test_reasoning_after_other_events_is_skipped(
    bundle: Bundle, events: tuple[dict[str, Any], ...]
) -> None:
    case = parse_case(trace(bundle, "Hi.<think>Plan.</think>", enable_thinking=None))
    result = judge(prepared(bundle, case), bundle, answer(*events))
    assert result["verdict"] == "skip:not-representable"
    assert "detail" not in result
    order = ",".join(event["kind"] for event in events)
    assert summarize({}, [result], {})["orders"] == {order: 1}


def test_text_after_a_call_is_judged(bundle: Bundle) -> None:
    # As Qwen3 writes parallel calls: a separator between them, folded into content.
    raw = "Plan.</think>Hi.<call>run {}</call>\n<call>run {}</call>"
    parsed = answer(
        reasoning("Plan."),
        text("Hi."),
        call("run", "{}"),
        text("\n"),
        call("run", "{}"),
    )
    result = judge(prepared(bundle, parse_case(trace(bundle, raw))), bundle, parsed)
    assert result["order"] == "reasoning,text,tool_call,text,tool_call"
    assert (result["verdict"], result["at"]) == ("fail", len("Plan.</think>Hi."))


def test_call_ids_reach_the_rendered_turn(bundle: Bundle) -> None:
    template = TEMPLATE.replace("<call>{{ call", "<call>{{ call.id }} {{ call")
    other = dataclasses.replace(bundle, chat_template=template)
    raw = 'Plan.</think>Hi.<call>call_1 run {"n": 1}</call>'
    job = prepared(other, parse_case(trace(other, raw)))
    events = (reasoning("Plan."), text("Hi."))
    parsed = answer(*events, call("run", '{"n": 1}', "call_1"))
    assert judge(job, other, parsed)["verdict"] == "pass"
    unnamed = answer(*events, call("run", '{"n": 1}'))
    assert judge(job, other, unnamed)["verdict"] == "fail"


def test_tool_choice_none_is_recorded(bundle: Bundle) -> None:
    obj = trace(bundle, 'Plan.</think><call>run {"n": 1}</call>')
    obj["request"]["tool_choice"] = "none"
    job = prepared(bundle, parse_case(obj))
    result = judge(job, bundle, answer(reasoning("Plan."), call("run", '{"n": 1}')))
    assert (result["verdict"], result["tool_choice"]) == ("pass", "none")


def test_only_function_tools_reach_the_parser() -> None:
    tool = {"type": "function", "function": {"name": "run"}}
    others = [{"type": "web_search"}, "run", {"type": "function", "function": "run"}]
    assert function_tools({"tools": [tool, *others]}) == [tool]


@pytest.mark.parametrize(
    ("parsed", "signature"),
    [
        ({"id": "c", "error": "parser panicked"}, "parser_error"),
        (answer(call("run", "[1]")), "invalid_arguments"),
    ],
    ids=["parser-error", "invalid-arguments"],
)
def test_unusable_answers_fail(
    bundle: Bundle, parsed: dict[str, Any], signature: str
) -> None:
    job = prepared(bundle, parse_case(trace(bundle, "Hi.")))
    result = judge(job, bundle, parsed)
    assert (result["verdict"], result["signature"]) == ("fail", signature)


@pytest.mark.parametrize(
    ("template", "verdict"),
    [
        # A template that raises on calls does not render them.
        (
            TEMPLATE.replace(
                "{%- for call",
                "{%- if m.tool_calls -%}{{ raise_exception('no calls') }}{%- endif -%}"
                "{%- for call",
            ),
            "skip:template-renders-no-tool-calls",
        ),
        ("{{ messages | length }}" + TEMPLATE, "skip:template-does-not-append-turns"),
    ],
    ids=["no-tool-calls", "no-append"],
)
def test_unrenderable_turns_are_skipped(
    bundle: Bundle, template: str, verdict: str
) -> None:
    other = dataclasses.replace(bundle, chat_template=template)
    job = prepared(other, parse_case(trace(other, "Hi.")))
    result = judge(job, other, answer(text("Hi."), call("run", "{}")))
    assert result["verdict"] == verdict
    assert result["starting_state"] == "reasoning"


@pytest.mark.parametrize(
    ("code", "error"), [(0, "answered 0 of 1"), (101, "exited 101")]
)
def test_run_parser_requires_an_answer_per_job(
    tmp_path: Path, code: int, error: str
) -> None:
    script = (
        "import sys; sys.stderr.write('case text'); " f"print('{{}}'); sys.exit({code})"
    )
    stderr = tmp_path / "parser.stderr"
    with pytest.raises(RuntimeError, match=error):
        run_parser([{"id": "c"}], [sys.executable, "-c", script], stderr)
    assert stderr.read_text() == "case text"
    assert stat.S_IMODE(stderr.stat().st_mode) == 0o600


@pytest.mark.parametrize(
    ("output", "declared", "stop_token_ids", "terminal"),
    [
        ([1, 2, EOS], [], None, [EOS]),
        ([1, 2, 3], [], None, []),
        ([1, 2, 3], [], [3], [3]),
        ([1, 2, 3], [2, 3], None, [2, 3]),
    ],
    ids=["eos", "no-stop-token", "request-stop-token", "declared"],
)
def test_terminal_token_inference(
    output: list[int],
    declared: list[int],
    stop_token_ids: list[int] | None,
    terminal: list[int],
) -> None:
    request: dict[str, Any] = {"messages": []}
    if stop_token_ids:
        request["stop_token_ids"] = stop_token_ids
    generation = {
        "output_token_ids": output,
        "finish_reason": "stop",
        "terminal_token_ids": declared,
    }
    case = parse_case(
        {"schema": SCHEMA, "id": "c", "request": request, "generation": generation}
    )
    assert case.terminal({EOS}) == terminal


@pytest.mark.parametrize(
    ("request_fields", "generation_fields"),
    [({}, {"terminal_token_ids": 0}), ({"stop_token_ids": "1"}, {})],
    ids=["terminal-token-ids", "stop-token-ids"],
)
def test_token_id_fields_are_validated(
    request_fields: dict[str, Any], generation_fields: dict[str, Any]
) -> None:
    generation = {"output_token_ids": [1], "finish_reason": "stop"}
    obj = {
        "schema": SCHEMA,
        "id": "c",
        "request": {"messages": [], **request_fields},
        "generation": {**generation, **generation_fields},
    }
    with pytest.raises(TraceError):
        parse_case(obj)


@pytest.mark.parametrize(
    ("family", "kwargs", "disabled"),
    [
        ("deepseek_v4", {}, False),
        ("deepseek_v4", {"thinking": False, "thinking_mode": "thinking"}, True),
        ("deepseek_v41", {"thinking_mode": "chat"}, True),
        ("gemma4", {}, True),
        ("gemma4", {"enable_thinking": True}, False),
        ("qwen3", {}, False),
        ("deepseek_v4", {"thinking": True, "thinking_mode": "chat"}, False),
        ("qwen3", {"enable_thinking": False}, True),
        ("qwen3", {"thinking": True, "enable_thinking": False}, False),
        ("glm47", {"thinking_mode": "chat"}, False),
    ],
    ids=[
        "deepseek-default",
        "deepseek-bool-wins",
        "deepseek-chat",
        "gemma4-default",
        "gemma4-thinking",
        "qwen3-default",
        "deepseek-bool-wins-over-chat",
        "qwen3-disabled",
        "thinking-wins",
        "glm47-thinking-mode-ignored",
    ],
)
def test_reasoning_disabled_mirrors_dynamo(
    family: str, kwargs: dict[str, Any], disabled: bool
) -> None:
    assert reasoning_disabled(family, kwargs) is disabled


@pytest.mark.parametrize(
    ("kwargs", "unmodeled"),
    [
        ({"thinking": True, "enable_thinking": "no"}, False),
        ({"thinking": None, "enable_thinking": False}, False),
        ({"thinking_mode": "chat"}, False),
        ({"thinking_mode": "adaptive", "thinking": "off"}, True),
        ({"thinking": "false"}, True),
        ({"enable_thinking": 0}, True),
        ({"thinking_mode": "disabled"}, True),
        ({"thinking_mode": " off "}, True),
        ({"thinking_mode": False}, True),
    ],
    ids=[
        "bool-first",
        "null-skipped",
        "deepseek-mode",
        "thinking-key-first",
        "string",
        "number",
        "mode-word",
        "mode-bool-word",
        "mode-bool",
    ],
)
def test_flags_dynamo_rewrites_are_unmodeled(
    kwargs: dict[str, Any], unmodeled: bool
) -> None:
    assert unmodeled_thinking_flag(kwargs) is unmodeled


def test_cli_writes_private_results(
    model_dir: Path, bundle: Bundle, tmp_path: Path
) -> None:
    parser = tmp_path / "parser"
    parser.write_text(ECHO_PARSER.format(python=sys.executable))
    parser.chmod(0o700)
    traces = [
        trace(bundle, "Hi.", enable_thinking=False),
        trace(bundle, "Pla", finish="length"),
    ]
    path = tmp_path / "traces.jsonl"
    path.write_text("".join(json.dumps(item) + "\n" for item in traces))
    output = tmp_path / "out"
    argv = ["--trace", str(path), "--model-dir", str(model_dir), "--family", "qwen3"]
    argv += ["--output-dir", str(output), "--parser-bin", str(parser)]
    # As documented: the package directory run as a script.
    completed = subprocess.run(
        [sys.executable, str(SRC / "parser_roundtrip"), *argv],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["counts"] == {"pass": 1, "skip:incomplete": 1}
    results = (output / "results.jsonl").read_text().splitlines()
    verdicts = [json.loads(line)["verdict"] for line in results]
    assert verdicts == ["pass", "skip:incomplete"]
    summary = json.loads((output / "summary.json").read_text())
    assert (summary["family"], summary["parser"]["parser"]) == ("qwen3", "echo")
    for name in ("results.jsonl", "summary.json", "summary.md", "parser.stderr"):
        assert stat.S_IMODE((output / name).stat().st_mode) == 0o600
    assert stat.S_IMODE(output.stat().st_mode) == 0o700


def test_output_dir_must_be_outside_git_or_ignored_logs(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / ".gitignore").write_text("logs/\n")
    with pytest.raises(SystemExit):
        private_dir(tmp_path / "out")
    assert not (tmp_path / "out").exists()
    (tmp_path / "logs" / "run").mkdir(parents=True, mode=0o755)
    run = private_dir(tmp_path / "logs" / "run")
    assert stat.S_IMODE(run.stat().st_mode) == 0o700
