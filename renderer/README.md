# Renderer

## Introduction
`dynamo-renderer` turns OpenAI-style chat requests into model-ready prompt strings. It is the *encode* side of inference serving: messages + tools + generation settings in, a fully-rendered prompt out. It is standalone, so an external OpenAI frontend can reuse Dynamo's prompt formatting without pulling in the Dynamo runtime.

It renders HuggingFace `chat_template` jinja2 (via `minijinja` + `minijinja-contrib` pycompat) and also ships native Rust formatters for model families whose prompt protocol cannot be represented faithfully by the published template. The crate is a *bridge* between OpenAI request types (`dynamo-protocols`) and the template engine; the native DeepSeek, Inkling, and Kimi K3 formatters additionally return segment-aware prompts so their control tokens and ordinary request text remain distinct through tokenization.

## Features
- **HF chat templates**: faithful `apply_chat_template` rendering, including tool-use and generation-prompt handling.
- **vLLM content formats**: `PromptFormatter::from_parts_with_content_format` shapes message content like vLLM's `--chat-template-content-format` (`string` or `openai`) instead of probing the template.
- **Control-token spellings as text**: a control-token spelling in a message, tool definition or `response_format` (for example `<|im_end|>`) comes back from `render_prompt` as an ordinary-text segment, so only the template's own markers encode as control tokens. On by default; `ChatTemplate::with_request_control_tokens_as_text(false)` turns it off, except for Harmony templates, which always render this way. A model that lists its markers only in `tokenizer.json` also needs `ChatTemplate::with_tokenizer_json`.
- **Native DeepSeek formatters**: Rust formatters for V4.1 / V4 / V3.2 families (under `deepseek`). Like the Inkling formatter, they return their control tokens and the request text as separate segments; `native_formatter_for_with(.., false)` turns that off.
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

## Generation-state metadata

`render_prompt()` returns a `RenderedPrompt` with `generation_state()`. Native
DeepSeek V3.2, V4, V4.1, Kimi K3, and Inkling formatters report the continuation
point where their encoder establishes it. Prompt text and tokenization segments
are unchanged. `render()` and `into_text()` discard the metadata.

| State | Meaning |
| --- | --- |
| `Unknown` | The renderer makes no claim about the continuation point. |
| `Unopened` | An assistant prefix is open, but no content channel is selected. |
| `Reasoning` | Generation continues inside reasoning. |
| `Response` | Generation continues inside the response channel. |

Existing `RenderedPrompt` constructors, arbitrary Jinja templates, and the no-op
formatter return `Unknown`. Native endings without a supported continuation
contract also return `Unknown`, including closed assistant turns and prompts
without an assistant generation prefix. DeepSeek V3.2 and V4 still emit their
native transitions when `should_add_generation_prompt()` is false. V4.1 honors
the flag and reports `Unknown` when its final generation prefix is suppressed.

Kimi partial assistant continuation reports `Response`, including when thinking
is enabled. A named tool choice also selects `Response`. Inkling's bare assistant
role prefix reports `Unopened`. These facts come from the encoder branches that
emit the structure, not from scanning user text for marker spellings.

The serving adapter owns the translation into parser and constraint settings.
Resolve known metadata once after all prompt transformations and carry the result
through both streaming and batch processing. `Unknown` requires the consumer's
compatibility fallback. It must not be treated as a known response channel.
`Unopened` retains channel-marker recognition; it does not disable reasoning.
Reasoning-boundary ownership is a separate constraint setting.

For Unified parsers, the corresponding starting states are `None` for `Unopened`,
`Reasoning` for `Reasoning`, and `Response` for `Response`. The adapter must still
check that the selected parser supports the rendered format. In particular,
Unified `Response` disables reasoning-marker interpretation. Derive
`starts_in_reasoning` from the same resolved state when building constraints.
This renderer API alone does not update Dynamo's serving adapter.

Any transformation that changes the continuation point must recompute the state
or invalidate it with `with_generation_state(GenerationState::Unknown)`. Custom
native formatters can attach a state with `with_generation_state`. Equality and
debug output include this metadata. The enum is non-exhaustive so consumers must
also handle future variants without assuming a response channel.

## Default thinking effort

On a renderer that supports `emit_default_thinking_effort`, set the option to `false` to omit the default effort instruction when no effort is supplied. The renderer still applies its usual rules to an explicit effort. Set the option in `chat_template_kwargs` for one request, or start the Dynamo frontend with `--default-chat-template-kwargs '{"emit_default_thinking_effort":false}'` for a deployment-wide default.

Thinking mode and the assistant generation prompt follow the model's usual rules. Dynamo clients can supply an explicit `reasoning_effort` on the request. With the option set to `false`, Muse Glimmer uses that effort when `reasoning_strength` is absent.

| Renderer or stock chat template | Default effort | Template argument for an explicit effort | Implementation |
| --- | --- | --- | --- |
| Kimi K3 and K3.1 | `max` | `thinking_effort` or `reasoning_effort` | [Native formatter](src/kimi_k3.rs) |
| DeepSeek V4 | `high` | `reasoning_effort` | [Native formatter](src/deepseek/v4.rs) |
| GPT-OSS | `medium` | `reasoning_effort` | [Chat template](https://huggingface.co/openai/gpt-oss-120b/blob/b5c939de8f754692c1647ca79fbf85e8c1e70f8a/chat_template.jinja#L203) |
| GLM-5.2 and GLM-5.3 | `max` | `reasoning_effort` | [5.2 template](https://huggingface.co/zai-org/GLM-5.2/blob/cf457fa734ab149ffef225f80893eb38c6ff5cdc/chat_template.jinja#L2), [5.3 template](https://huggingface.co/zai-org/GLM-5.3/blob/ba32f8c19c8738e2c57c71798c601d477a7bf844/chat_template.jinja#L2) |
| Beam | `512` | `reasoning_effort` | [Chat template](https://huggingface.co/reflection/Beam-1.0-501B-A23B/blob/877a5d6973c6ae6a55335483002c06717662d23c/chat_template.jinja#L282) |
| Muse Glimmer | `high` | `reasoning_strength` | [Chat template](https://huggingface.co/meta-models/Muse-Glimmer-30B/blob/1297fd7dfeac4598948a5d15886968435c81c2b7/chat_template.jinja#L83) |

For example, with the option set to `false`, a Kimi request that omits effort opens the thinking turn without an effort message. A request that explicitly sets `reasoning_effort=max` opens the same turn and includes the `thinking_effort=max` message. GPT-OSS, GLM, Beam, and Muse Glimmer support the option through their stock template blocks. A custom chat template can read the option directly.

## DeepSeek V4.1

The V4.1 formatter supports text and image messages, tool history, mid-conversation system messages, and numeric reasoning effort. It rejects audio/video content and explicit tool namespace fields; qualified function names are preserved. Generation headers follow the reference encoder. Setting `add_generation_prompt` to `false` suppresses only the final generation prefix and preserves historical transitions. OpenAI effort names match the official chat template: `low` is 50, `medium` is 62, `high` is 75, and `max` is 100; the default is 75. The `xhigh` alias is not supported. Template arguments accept the same names or an integer from 1 to 100. Top-level effort takes precedence over template effort. Set `reasoning_effort` to `none` or the template argument `thinking` to `false` to disable thinking.

Reasoning-effort names and the default follow the model's [official chat template at revision `2cba9e42`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/2cba9e42aa026125f3ed06c6d98c1db82f7ca027/chat_template.jinja#L19). Prompt fixtures pass explicit numeric effort to the low-level encoder. Image cases use the same fixture schema and include URL/data-URL inputs, mixed content, multiple turns and tool results. Their expected prompts were generated with the [Python reference encoder at revision `dba1be0a`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/dba1be0a40aa45a94ad051997016db3960a90277/encoding/encoding.py#L444) (SHA-256 `502bdaec8a3fd88ebc24c4721a7038fbe42f2063c664638127056107920035c1`). UUID-only image reuse is a Dynamo processor-cache extension tested separately.

Prompt-format tests do not replace live image-fetch, transfer, cache and backend validation. The runtime must collect tool-result images in the same call order as their rendered placeholders.

## Relationship to other crates
- `dynamo-protocols` — OpenAI/wire request types this crate renders from.
- `dynamo-tokenizers` — tokenization (the *next* step after rendering); re-exported here for convenience.
- `dynamo-parsers` — the *decode* side (parsing model output back into reasoning / tool calls).

## Ownership

This crate is developed and published from frontend-crates. Dynamo consumes the published crate rather than carrying a source mirror.
