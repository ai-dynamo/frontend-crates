# dynamo-parsers (v1, legacy)

Rust crate for parsing **tool calls** and **reasoning content** out of raw LLM output. This is the v1 **batch** path: it jails (buffers) the whole model output, then parses.

> **This is the crate to depend on today.** `dynamo-parsers` (v1) is the stable, published parser API. The pure-streaming **v2** crate (`dynamo-parsers-v2`, under `../v2/`) is **still under active development (WIP)** — its API changes freely on the `0.x` line and it is not yet a drop-in replacement. v2 will eventually replace v1 (v1 is not being merged into v2 — when v2 is done this crate is removed outright), but until then, external consumers should use v1.

## Using the crate

Add it from crates.io (published as `dynamo-parsers`):

```toml
[dependencies]
dynamo-parsers = "3"
```

```rust
use dynamo_parsers::tool_calling::detect_and_parse_tool_call;

// registry dispatch by parser name; returns (tool calls, leftover normal_text)
let (calls, normal_text) = detect_and_parse_tool_call(input, "qwen3_coder", None)?;
```

The crate import path is `dynamo_parsers` (crate name, unaffected by the directory move under `parsers/v1/`). The WIP streaming crate is `dynamo_parsers_v2` — **do not** reach for it in production yet.

## v1-specific API

### Tool execution boundary

The GLM47 parser preserves syntactically valid calls even when their names are absent from the supplied tools. Parsing does not authorize execution: every executor must validate each returned name against the tools offered in that request before dispatching it, including when using a broader global tool registry. Unknown calls have no matching request schema, so their arguments use the parser's schema-free coercion rules.

### Source recording

A caller that maps output text back to input tokens can ask where each returned byte came from. Call `record_sources()` on a `ReasoningParser` before the first chunk. After each streaming call, `take_sources(&mut out)` moves out one list of `SourceRun`s in input order, each tagged `Normal` or `Reasoning`. The input ranges of a channel's runs, read in order, equal that channel's text. `record_sources()` returns `false` when a parser cannot record. `BasicReasoningParser` records; the other parsers keep the default and return `false`. Batch parsing records nothing.

With recording off, the output does not change. `sources::ChoiceSources` and `sources::SourceSlot` let a stream chunk type carry ranges per choice. The tool-call jail keeps ranges internally for `hermes`, `kimi_k2`, and `minimax_m3`. Its stream boundary does not expose them yet.

### Entry points

The v1 batch entry points (in `src/tool_calling/parsers.rs`) that v2 does not mirror:

- `detect_and_parse_tool_call(input, parser_name, schema) -> (calls, normal_text)` — registry dispatch.
- `try_tool_call_parse(input, config) -> (calls, normal_text)` — lower-level, bypasses the registry.
- `detect_tool_call_start(chunk, parser_name)` — streaming: "is this chunk starting a tool-call block?"
- `find_tool_call_end_position(chunk, parser_name)` — streaming: "where does the block end in this chunk?"

The registry (`tool_calling/parsers.rs`) maps a parser name to a `ParserConfig` variant (`Dsml` / `Json` / `Xml` / `KimiK2` / `Pythonic` / `Harmony`); per-parser presets are in `tool_calling/config.rs`. Reasoning parsers register by name in `reasoning/mod.rs`; see [`src/reasoning/README.md`](src/reasoning/README.md).

### Stream stages

dynamo's frontend runs these stages on a chat chunk stream, on the v1 route (reasoning stream, then the jail):

- `reasoning::stream::parse_reasoning_content_from_stream` splits `content` into `reasoning_content` and `content`, per choice.
- `reasoning::stream::strip_leading_reasoning_start_from_stream` strips a leading reasoning opener.
- `tool_calling::jail::apply_tool_calling_jail` parses tool calls.
- `tool_calling::jail::apply_jail_terminal_error_contract` runs a jail stage so that a backend error ends the stream with that error, and nothing the jail builds after it.

The two reasoning stages and the contract are generic over `chat_stream::ChatStreamItem`. The jail's `Annotated` implements it. A caller with its own item type, as dynamo has, implements it to keep its own per-chunk fields.

For the family-to-grammar-to-file mapping (the cheat-sheet), the parser goals, and the step-by-step add flow, see [`../v2/README.md`](../v2/README.md).
