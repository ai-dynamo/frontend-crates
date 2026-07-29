# Renderer

## Introduction
`dynamo-renderer` turns OpenAI-style chat requests into model-ready prompt strings. It is the *encode* side of inference serving: messages + tools + generation settings in, a fully-rendered prompt out. It is standalone, so an external OpenAI frontend can reuse Dynamo's prompt formatting without pulling in the Dynamo runtime.

It renders HuggingFace `chat_template` jinja2 (via `minijinja` + `minijinja-contrib` pycompat) and also ships native Rust formatters for model families whose prompt protocol cannot be represented faithfully by the published template. The crate is a *bridge* between OpenAI request types (`dynamo-protocols`) and the template engine; Kimi K3 additionally returns segment-aware prompts so structural XTML tokens and ordinary user text remain distinct through tokenization.

## Features
- **HF chat templates**: faithful `apply_chat_template` rendering, including tool-use and generation-prompt handling.
- **Native DeepSeek formatters**: Rust formatters for V4 / V3.2 families (under `deepseek`).
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

## Relationship to other crates
- `dynamo-protocols` — OpenAI/wire request types this crate renders from.
- `dynamo-tokenizers` — tokenization (the *next* step after rendering); re-exported here for convenience.
- `dynamo-parsers` — the *decode* side (parsing model output back into reasoning / tool calls).

## Provenance

This crate is a one-way mirror of `lib/renderer` from
[ai-dynamo/dynamo](https://github.com/ai-dynamo/dynamo). `src/` is synced
verbatim; `Cargo.toml` and this README are inlined for standalone publishing.
See the repo root `scripts/sync-from-dynamo.sh`.
