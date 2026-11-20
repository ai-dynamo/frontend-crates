<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Parser round trip

The round trip checks that the v2 Unified parser loses nothing a model wrote: it parses the model's recorded output, renders the parsed message back with the model's own chat template, and requires the result to equal what the model produced. It needs no expected answer, so any captured output is a test case.

## How it works

Each value below is either **recorded**, read from the trace or the engine, or **computed** by the judge. The example is a real Qwen3.6 case with thinking off, where the user asked "Say hello briefly.".

| Value | Source | Example |
|---|---|---|
| request | recorded: trace `request` | `messages` plus `chat_template_kwargs: {"enable_thinking": false}` |
| prompt token IDs | recorded: trace `prompt.token_ids` | 312 IDs, ending in `<\|im_start\|>assistant\n<think>\n\n</think>\n\n` |
| output token IDs | recorded: trace `generation.output_token_ids` | `Hello`, `!`, `<\|im_end\|>` |
| generation prompt | computed: the template renders the request with the generation prompt added (`prompt.py`); it must tokenize to the recorded prompt IDs | the same 312 tokens |
| tail | computed: the generation prompt minus the rendered history | `<\|im_start\|>assistant\n<think>\n\n</think>\n\n` |
| x | computed: the output IDs without the stop token, decoded | `Hello!` |
| starting state | computed from the tail by dynamo.frontend's rules (`serving.py`) | `none`: the thought is already closed |
| parsed message | computed: the parser reads x from the starting state (`driver.py`), and its events fold into one message (`message.py`) | `{"content": "Hello!"}` |
| MODEL TURN | computed: tail + x | `<\|im_start\|>assistant\n<think>\n\n</think>\n\nHello!` |
| PARSER TURN | computed: the template renders history + parsed message, minus the history and the end-of-turn marker (`judge.py`) | `<\|im_start\|>assistant\n<think>\n\n</think>\n\nHello!` |
| verdict | MODEL TURN == PARSER TURN | `pass` |

The template is the parser's inverse: the parser turns text into a message, and the template turns the message back into text. A parse that drops, moves or rewrites text renders differently. Because the parser turn is what the next request's prompt holds, a mismatch also costs prefix-cache reuse from the first differing token.

Every computed value depends on the generation prompt. If the request lacks something the server applied, such as a default `enable_thinking: false`, the template renders a prompt that ends with an open `<think>\n`. The parser would then be judged from inside a thought the model never had, and a parser that wrongly files the answer as reasoning would still pass. The recorded prompt token IDs catch this: the case becomes `skip:prompt-mismatch` instead.

What a pass does not cover:

- Two messages that render to the same text, such as an unclosed thought labeled as reasoning or as content, cannot be told apart.
- Each output is parsed whole, so chunk boundaries are not exercised.
- Guided tool choice and dynamo.frontend's serving adapter are not exercised.

## Trace format

JSONL, one case per line, schema `dynamo-replay-trace/v1`. This line is shortened; real prompts have hundreds of token IDs:

```json
{"schema": "dynamo-replay-trace/v1", "id": "hello-1", "request": {"messages": [{"role": "user", "content": "Say hello briefly."}], "chat_template_kwargs": {"enable_thinking": false}, "max_tokens": 4096}, "prompt": {"token_ids": [248045, 846, 198]}, "generation": {"output_token_ids": [9419, 0, 248046], "finish_reason": "stop"}}
```

| Field | Required | Meaning |
|---|---|---|
| `schema` | yes | `dynamo-replay-trace/v1` |
| `id` | yes | Unique within the file |
| `request` | yes | The Chat Completions body as the engine applied it: `messages`, `tools`, `tool_choice`, `chat_template_kwargs`, `reasoning_effort` and the rest. Remove capture-only fields such as `return_token_ids` and `stream`. Write server defaults that change rendering into it. |
| `prompt.token_ids` | recommended | The prompt token IDs the engine returned. Without them the judge has to trust that `request` reproduces the prompt. |
| `generation.output_token_ids` | yes | The engine's raw output IDs, including the final stop token. Never re-tokenize the text: special tokens would come back as ordinary ones. |
| `generation.finish_reason` | yes | `stop` or `length`, decided from the IDs: output without a stop token is `length`. Engines can report `tool_calls` for a call cut off by `max_tokens`. |
| `generation.terminal_token_ids` | no | The stop-token suffix, when it is more than one ID or not an EOS ID |
| `request.stop_token_ids` | no | Extra stop IDs the request declared |

Other fields are ignored, so a trace can carry more.

## Capturing traces

Capture from the engine's own server, not through Dynamo. Send each request with `"return_token_ids": true`, `"n": 1`, an explicit `max_tokens`, and `stream` off.

| Server | Prompt token IDs | Output token IDs |
|---|---|---|
| Stock vLLM server (`vllm serve`, 0.10.2 or later) | top-level `prompt_token_ids` | `choices[0].token_ids` |
| Stock SGLang server (`sglang.launch_server`, 0.5.18 or later) | `choices[0].prompt_token_ids` | `choices[0].response_token_ids` |

With a reasoning or tool parser configured, stock vLLM 0.26 and later return no output token IDs when the request sets `include_reasoning: false`, so leave it unset.

## Model directory

`--model-dir` holds the files of the exact checkpoint revision the server ran: `tokenizer.json` and `tokenizer_config.json`, `chat_template.jinja` (or a template inside `tokenizer_config.json`), and `config.json` or `generation_config.json` for the EOS token IDs. Rendering uses HF transformers, which shares no code with dynamo-renderer or the parsers under test.

## Running

The judge needs a Python with `transformers` and `tokenizers`.

```bash
cargo build -p dynamo-conformance-fixtures-v2 --example parser_roundtrip
```

```bash
python3 conformance/utils/src/parser_roundtrip \
  --trace traces.jsonl \
  --model-dir model \
  --family qwen3 \
  --parser-bin target/debug/examples/parser_roundtrip \
  --output-dir /tmp/roundtrip
```

`--family` is a Unified parser family or alias, such as `qwen3`, `deepseek_v4` or `minimax_m3`. The output directory must be outside any Git work tree or an ignored path under its `logs/`, because results quote model text; every file is written with mode 0600. The command prints the counts and exits 1 when any case fails.

## Reading results

`summary.md` lists the counts and groups failures by signature. `results.jsonl` holds one line per case with its `verdict`, `starting_state`, parsed event `order`, and any `note`, `signature`, `detail`, or `raw` and `rendered` snippets.

- **`pass`**: MODEL TURN equals PARSER TURN.
- **`pass` with a `note`**: the template cannot write back what the model wrote, and a fold (`folds.py`) forgave exactly that difference: `empty_reasoning`, `template_inserted_prefix` or `unclosed_reasoning`. Read these too.
- **`fail`**: `at` is the first differing character in x, and `signature` shows both sides there with words folded to `w`. `parser_error` and `invalid_arguments` mean the parser failed or wrote arguments that are not a JSON object.
- **`skip:<reason>`**: the case was not judged.

| Skip reason | Why |
|---|---|
| `incomplete` | `finish_reason` is `length`. |
| `prompt-mismatch` | The rendered prompt does not tokenize to the recorded prompt token IDs; `detail` shows the first differing token. |
| `stop-strings` | dynamo.frontend trims a matched stop string, which x does not model. |
| `guided-tool-choice` | `tool_choice` is `required` or names a tool, which Dynamo serves with guided JSON. |
| `unmodeled-thinking-flag` | dynamo.frontend rewrites this thinking flag before rendering, so the recorded prompt is not the one it would serve. |
| `unmodeled-family` | GPT-OSS and Beam get a prompt prefix or channel state from Dynamo that a whole-text job cannot carry. |
| `commentary-message` | The parser addressed a visible message as commentary, which the template does not render as content. |
| `not-representable` | Reasoning came after text or a call, and one message cannot hold that order. |
| `template-error`, `template-renders-no-tool-calls`, `template-renders-no-reasoning`, `template-does-not-append-turns` | The template cannot render this turn, so there is nothing to compare. |

## Modules

| Module | Stage |
|---|---|
| `trace.py` | Read and validate trace lines |
| `bundle.py` | Load the model directory; render and decode with HF transformers |
| `prompt.py` | Render the generation prompt and check it against the recorded prompt token IDs |
| `serving.py` | dynamo.frontend's decisions before parsing: family aliases, thinking off, starting state, tools and requests it serves otherwise. MiniMax M3 follows the Unified route of ai-dynamo/dynamo#17236. |
| `driver.py` | Run the `parser_roundtrip` example, `conformance/examples/parser_roundtrip.rs` |
| `message.py` | Fold the parser's ordered events into one assistant message |
| `folds.py` | Probe how the template frames a turn, and forgive the differences that framing forces |
| `judge.py` | Prepare each case, compare MODEL TURN with PARSER TURN, and run a trace |
| `report.py` | Write `summary.json`, `summary.md` and private files |
