# dynamo-parsers (v1, legacy)

Rust crate for parsing **tool calls** and **reasoning content** out of raw LLM output. This is the v1 **batch** path: it jails (buffers) the whole model output, then parses.

> `dynamo-parsers` is the published legacy batch/jail API. For new ordered streaming integrations, evaluate [Unified parsing in `dynamo-parsers-v2`](../v2/README.md#using-unified-parsing). V2 is used by Dynamo for compatible configurations, has a separate `0.x` API, and is not a drop-in replacement for this crate. Pin and test the version you integrate.

## Using the crate

Add it from crates.io (published as `dynamo-parsers`):

```toml
[dependencies]
anyhow = "1"
dynamo-parsers = "=10.2.2"
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
```

```rust
use dynamo_parsers::tool_calling::detect_and_parse_tool_call;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let input = "<tool_call><function=ping></function></tool_call>";
    let (calls, normal_text) =
        detect_and_parse_tool_call(input, Some("qwen3_coder"), None).await?;
    assert_eq!(calls[0].function.name, "ping");
    assert!(normal_text.as_deref().unwrap_or_default().is_empty());
    Ok(())
}
```

Returns `(Vec<ToolCallResponse>, Option<String>)`: calls contain an ID, `function.name`, and JSON-text `function.arguments`; the optional string is remaining text. Pass an optional registry name and optional request tool definitions for schema-directed conversion. The call is async.

## v1-specific API

### Tool execution boundary

The GLM47 parser preserves syntactically valid calls even when their names are absent from the supplied tools. Parsing does not authorize execution: every executor must validate each returned name against the tools offered in that request before dispatching it, including when using a broader global tool registry. Unknown calls have no matching request schema, so their arguments use the parser's schema-free coercion rules.

### Entry points

The v1 batch entry points (in `src/tool_calling/parsers.rs`) that v2 does not mirror:

- `detect_and_parse_tool_call(input, Option<&str>, Option<&[ToolDefinition]>).await -> Result<(Vec<ToolCallResponse>, Option<String>)>` — registry dispatch.
- `try_tool_call_parse(input, config, tools).await -> Result<(Vec<ToolCallResponse>, Option<String>)>` — lower-level, bypasses the registry.
- `detect_tool_call_start(chunk, parser_name)` — streaming: "is this chunk starting a tool-call block?"
- `find_tool_call_end_position(chunk, parser_name)` — streaming: "where does the block end in this chunk?"

The registry (`tool_calling/parsers.rs`) maps a parser name to a `ToolCallConfig` preset; per-parser presets are in `tool_calling/config.rs`. Reasoning parsers register by name in `reasoning/mod.rs`; see [`src/reasoning/README.md`](src/reasoning/README.md).

For the family-to-grammar-to-file mapping (the cheat-sheet), the parser goals, and the step-by-step add flow, see [`../v2/README.md`](../v2/README.md).
