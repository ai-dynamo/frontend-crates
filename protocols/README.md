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

Use `protocol-schema` to generate OpenAPI schemas for chat/completion requests,
chat responses and stream chunks, and `CreateCompletionResponse`. The `utoipa`
feature is an alias for `protocol-schema`.

Enable this feature on your `dynamo-protocols` dependency. The feature enables
the crate's schema dependencies automatically; you do not need to declare them
yourself just to enable it.

The export example below directly uses `utoipa` and `serde_json`, so add
`utoipa` 5 and `serde_json` 1 to your dependencies to run it. Register the types
you need, then serialize the generated document:

```rust
use utoipa::OpenApi;

#[derive(utoipa::OpenApi)]
#[openapi(components(schemas(
    dynamo_protocols::types::CreateChatCompletionRequest,
    dynamo_protocols::types::CreateCompletionRequest,
    dynamo_protocols::types::CreateChatCompletionResponse,
    dynamo_protocols::types::CreateChatCompletionStreamResponse,
    dynamo_protocols::types::CreateCompletionResponse
)))]
struct Contract;

fn main() -> Result<(), serde_json::Error> {
    println!("{}", serde_json::to_string_pretty(&Contract::openapi())?);
    Ok(())
}
```

The document includes schemas for the registered types and the nested types they
reference, such as tool definitions and usage statistics. Dynamo-owned schemas
use `dynamo_protocols.*` names; upstream schemas use `async_openai.*` names.
This feature requires async-openai 0.42.2 or a newer compatible version. It does
not enable the async-openai HTTP client or change JSON parsing and serialization.

### Contract and coverage limits

- **Schema limits do not enforce server policy.** Adding numeric limits does not
  change JSON parsing. For example, `n=0` still parses, but the serving layer
  rejects it. The `n: default=1` annotation also does not fill in a value: omitted
  or null input remains `None` for the server to resolve. Other sampling defaults
  depend on the backend and model.
- **Intentional differences remain.** Dynamo's `top_p` excludes zero and its
  logit biases may be fractional. Local reasoning, multimodal, token-stop,
  streaming-usage, and permissive-input extensions retain their own schemas.
- **Schemas do not describe every accepted input form.** Custom parsers convert
  object-valued function arguments to strings, expand media shorthand, and
  normalize reasoning aliases and null stream-option flags. Tests check these
  values after parsing and serialization, not just the original input.
- **Other API families remain uncovered.** This feature does not provide complete
  schemas for Responses, embeddings, images, Realtime, Batch, Files, or Anthropic.
  Some upstream response schemas also allow fields to be omitted even though
  serialized responses always include them.
- **Server behavior needs separate tests.** These component schemas do not define
  HTTP errors, server-sent event (SSE) framing and ordering, configured output
  keys, or engine support. Resolved schema references do not prove server
  compatibility.

### Verify schema generation

From the repository root, run the protocol suite with and without schemas:

```sh
cargo test -p dynamo-protocols --features protocol-schema --locked
cargo test -p dynamo-protocols --no-default-features --locked
```

The protocol tests check schema shapes and references as well as JSON parsing
and serialization.

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
