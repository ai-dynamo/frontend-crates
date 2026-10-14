# dynamo-protocols

Request/response types for OpenAI- and Anthropic-compatible inference servers. Built on [`async-openai`](https://crates.io/crates/async-openai) v0.42, with selective overrides where inference engines need behaviors upstream doesn't support.

## What's included

- **Chat Completions** — multimodal content, reasoning content (DeepSeek-R1 / QwQ), continuous usage stats, tool-calling.
- **Batch API** — re-exported from upstream.
- **Files API** — re-exported from upstream.
- **Responses API** — input chain owned (relaxed for Codex / Agents SDK); output chain re-exported from upstream.
- **Completions** — locally owned request; response types re-exported from upstream.
- **Anthropic Messages** — fully owned (no upstream equivalent).
- **Embeddings**, **Images** — re-exported.

## Optional protocol schemas

The `protocol-schema` feature is disabled by default. When enabled, it supplies utoipa 5 `ToSchema`
implementations for locally owned chat/completion requests, chat responses,
chat stream chunks, and their locally owned nested types. It uses the released
`async-openai` dependency without requiring an upstream schema feature or Git
override. It does not enable an HTTP client or change Serde behavior.
Wholly re-exported types (including `CreateCompletionResponse`) and other
endpoint families are outside this initial schema surface.

```rust,ignore
#[derive(utoipa::OpenApi)]
#[openapi(components(schemas(
    dynamo_protocols::types::CreateChatCompletionRequest,
    dynamo_protocols::types::CreateCompletionRequest,
    dynamo_protocols::types::CreateChatCompletionResponse,
    dynamo_protocols::types::CreateChatCompletionStreamResponse
)))]
struct Contract;
```

The export describes types defined in this crate. When a field uses a type
from `async-openai` that has no schema support, the export keeps the field but
uses a placeholder for its type. For example, a completion request describes
`model` as a string, but leaves the detailed definition of `prompt` as a
placeholder for async-openai's `Prompt` type.

Before comparing that field with another server, a separate tool must fill in
the missing definition using an upstream specification checked against the
async-openai version in use. This crate does not do that step. Until the
definition is filled in, the field's compatibility is unknown, not confirmed.

In the OpenAPI document, definitions from this crate have names beginning with
`dynamo_protocols.`; upstream placeholders use `async_openai.`. These names do
not change the JSON request or response format. Each placeholder has an
`x-dynamo-schema-import` marker identifying the upstream crate and Rust type,
so a separate tool can find the definition to supply.

The placeholder itself allows any non-null JSON value rather than guessing the
missing type's rules. For an `Option<T>` field, the generated schema also allows
null. This is only a limitation of the exported schema: the server still parses
the field using its real Rust type and can reject values the placeholder allows.

Installing a future upstream release with schema support will not automatically
replace these explicit `value_type` overrides. Native upstream support requires
compatible trait versions, Cargo feature wiring, and switching the annotations
away from import slots. This follow-up is tracked in
[#351](https://github.com/ai-dynamo/frontend-crates/issues/351).

### Current coverage limits

Enabling the feature does not provide a complete schema for every protocol in
this crate. The following are not currently covered:

- **Detailed upstream type definitions.** Imported types such as `Prompt`,
  `FunctionObject`, `ResponseFormat`, and `CompletionUsage` remain marked import
  slots, not detailed schemas. Their nested fields, enum alternatives, and
  constraints require external resolution as described above. A resolved local
  `$ref` can still point to an incomplete import slot.
- **Wholly re-exported completion responses.** `CreateCompletionResponse` does
  not gain `ToSchema` from this feature. Completion request coverage must not be
  interpreted as completion response or streaming-response coverage.
- **Other protocol families and errors.** Responses, Anthropic Messages, Batch,
  Files, Embeddings, Images, and Realtime do not have schema exports through
  this feature. It also does not define server error-response contracts.
- **The full accepted-input contract.** The schemas describe canonical
  typed/serialized forms, not every custom deserializer path. Input aliases
  (`reasoning`), object-valued function arguments, media shorthand, tools-only
  system messages, and null stream-option booleans can normalize before
  serialization. Role-specific validation can reject other combinations.
  These differences need separate coverage; the generated schema is not a
  drop-in replacement for request deserialization or validation.
- **Contents of arbitrary JSON fields.** Fields such as `mm_processor_kwargs`
  expose their declared container shape, not backend/model-specific keys or
  semantics. Their presence in the schema does not establish backend support.
- **HTTP and runtime behavior.** Component schemas do not define endpoint
  registration, HTTP status/header behavior, SSE framing, chunk ordering or
  termination, conditional field emission, or backend handling. Even where a
  chunk or response type has a schema, actual serving behavior needs separate
  conformance tests. Schema agreement alone is not proof of server parity.

Run `cargo test -p dynamo-protocols --features protocol-schema` to exercise
schema generation and existing protocol tests.

## Locally-defined extensions

A few fields extend the upstream `async-openai` schema:

- `reasoning_content` on assistant messages
- `mm_processor_kwargs` on chat-completion requests (vLLM multimodal)
- `continuous_usage_stats` on chat stream options
- `FunctionCall.arguments` accepts both string and object forms

## Request compatibility and serving policy

`CreateResponse` accepts `"text":{"verbosity":"low"}` without a `format`.
The upstream `ResponseTextParam` defaults the format to `text`. An explicit
`text`, `json_object`, or `json_schema` format is preserved independently of
verbosity. A serving adapter that cannot honor verbosity should ignore that
hint while retaining the output-format constraint. Map supported fields to
the engine request instead of forwarding the Responses `text` object.

`CreateChatCompletionRequest` and `CreateCompletionRequest` accept and preserve
`stream_options` when `stream` is absent, null, false, or true. These types do
not enforce cross-field streaming policy. A serving adapter that chooses
permissive compatibility should clear `stream_options` when
`request.stream != Some(true)` before backend validation or forwarding. Keep
the options when streaming, and keep normal non-streaming response usage.
This normalization belongs in the serving adapter because protocol types
also serve callers that need to preserve the original request.

Run the wire-level regressions for
[#299](https://github.com/ai-dynamo/frontend-crates/issues/299) with
`cargo test -p dynamo-protocols --test request_compatibility`.
