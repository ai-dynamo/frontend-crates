# dynamo-parsers (v1, legacy)

Rust crate for parsing **tool calls** and **reasoning content** out of raw LLM output. This is the v1 **batch** path: it jails (buffers) the whole model output, then parses.

> **Still in use — to be removed once v2 is done.** This v1 crate is the jail-and-buffer (batch) parser. The pure-streaming v2 path under `../parsers_v2/` is under development and will fully replace it; v1 is not being merged into v2 — when v2 is done this entire crate and its docs will be removed outright. The canonical parser documentation now lives in [`../parsers_v2/README.md`](../parsers_v2/README.md): parser goals, the family cheat-sheet, how to add a parser, fixtures, and conformance all live there. Do new parser work in v2; only touch v1 to fix a batch bug that v2's `batch_via_stream` parity depends on.

## v1-specific API

The v1 batch entry points (in `src/tool_calling/parsers.rs`) that v2 does not mirror:

- `detect_and_parse_tool_call(input, parser_name, schema) -> (calls, normal_text)` — registry dispatch.
- `try_tool_call_parse(input, config) -> (calls, normal_text)` — lower-level, bypasses the registry.
- `detect_tool_call_start(chunk, parser_name)` — streaming: "is this chunk starting a tool-call block?"
- `find_tool_call_end_position(chunk, parser_name)` — streaming: "where does the block end in this chunk?"

The registry (`tool_calling/parsers.rs`) maps a parser name to a `ParserConfig` variant (`Dsml` / `Json` / `Xml` / `KimiK2` / `Pythonic` / `Harmony`); per-parser presets are in `tool_calling/config.rs`. Reasoning parsers register by name in `reasoning/mod.rs`; see [`src/reasoning/README.md`](src/reasoning/README.md).

For the family-to-grammar-to-file mapping (the cheat-sheet), the parser goals, and the step-by-step add flow, see [`../parsers_v2/README.md`](../parsers_v2/README.md).
