# Renderer

## Introduction
`dynamo-renderer` turns OpenAI-style chat requests into model-ready prompt strings. It is the *encode* side of inference serving: messages + tools + generation settings in, a fully-rendered prompt out. It is standalone, so an external OpenAI frontend can reuse Dynamo's prompt formatting without pulling in the Dynamo runtime.

It renders HuggingFace `chat_template` jinja2 (via `minijinja` + `minijinja-contrib` pycompat) and also ships native Rust formatters for model families whose prompt protocol cannot be represented faithfully by the published template. The crate is a *bridge* between OpenAI request types (`dynamo-protocols`) and the template engine; Kimi K3 additionally returns segment-aware prompts so structural XTML tokens and ordinary user text remain distinct through tokenization.

## Features
- **HF chat templates**: faithful `apply_chat_template` rendering, including tool-use and generation-prompt handling.
- **Native DeepSeek formatters**: Rust formatters for V4.1 / V4 / V3.2 families (under `deepseek`).
- **Native Inkling formatter**: exact text, image, audio, reasoning, and tool-use framing for `inkling_mm_model` (under `inkling`). Media blocks contain only the marker token; the backend multimodal processor expands per-patch or per-frame placeholders later.
- **Native Kimi K3 formatter**: XTML rendering with explicit trusted-control and ordinary-text segment boundaries (under `kimi_k3`).
- **Bring-your-own request type**: implement `OAIChatLikeRequest` for any request type, or use the ready-made impl for `dynamo-protocols`' OpenAI chat request.
- **Self-contained**: no async runtime or networking; segment-aware prompts interoperate with `dynamo-tokenizers`.

## Quick Start

```rust
use dynamo_renderer::{ChatTemplate, ContextMixins, PromptFormatter};
use dynamo_protocols::types::CreateChatCompletionRequest;

// `config` is parsed from a model's `tokenizer_config.json`.
let config: ChatTemplate = serde_json::from_str(tokenizer_config_json)?;
let PromptFormatter::OAI(formatter) =
    PromptFormatter::from_parts(config, ContextMixins::default(), /* exclude_tools_when_tool_choice_none */ false)?
else {
    unreachable!("from_parts always builds an OAI formatter")
};

// Any type implementing `OAIChatLikeRequest` can be rendered; the standard
// OpenAI chat request works out of the box.
let request: CreateChatCompletionRequest = serde_json::from_str(request_json)?;
let prompt: String = formatter.render(&request)?;
```

## DeepSeek V4.1

The V4.1 formatter supports text messages, tool history, mid-conversation system messages, and numeric reasoning effort. It rejects media content and explicit tool namespace fields; qualified function names are preserved. Generation headers follow the reference encoder and cannot be disabled with `add_generation_prompt`. OpenAI effort names match the reference encoder: `low` is 50, `high` is 75, and `max` is 100; the default is 75. The `xhigh` alias is not supported. Template arguments accept the same names or an integer from 1 to 100. Top-level effort takes precedence over template effort. Set `reasoning_effort` to `none` or the template argument `thinking` to `false` to disable thinking.

Reasoning-effort names and the default follow the model's [Python reference encoder at revision `dba1be0a`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/dba1be0a40aa45a94ad051997016db3960a90277/encoding/encoding.py#L444). Prompt fixtures pass explicit numeric effort to the low-level encoder.

## Relationship to other crates
- `dynamo-protocols` — OpenAI/wire request types this crate renders from.
- `dynamo-tokenizers` — tokenization (the *next* step after rendering); re-exported here for convenience.
- `dynamo-parsers` — the *decode* side (parsing model output back into reasoning / tool calls).

## Ownership

This crate is developed and published from frontend-crates. Dynamo consumes the published crate rather than carrying a source mirror.
