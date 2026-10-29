<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# dynamo-chat-parsing

The rules an OpenAI chat frontend uses to pick and drive Dynamo's parsers. The parser crates parse model output, but they do not choose which parser serves a request. The Dynamo frontend makes that choice from the deployment's tool-call and reasoning parser names. This crate holds those rules, so another frontend gives a client the same reply as Dynamo.

| Module | Rule |
| -- | -- |
| `version` | `ParserVersion`: `auto`, `1`, or `2`. The caller passes it in. The crate does not read `DYN_PARSER_VERSION`. |
| `families` | The unified v2 family that a parser pair selects, with Dynamo's aliases (for example `glm45` selects `glm47`). |
| `route` | The path a request takes: the unified v2 parser, the v1 tool-call jail, or no tool parsing. Also the start-up version check. |
| `prefill` | Whether the rendered prompt already opened a reasoning block. |
| `special_tokens` | The `skip_special_tokens` default for each parser. |
| `stop_tokens` | A tool-call end token that is a hidden EOS becomes a visible stop. |
| `tools` | The effective tools (top-level and system-message tools) and the tool-parsing gate. |
| `tool_choice` | The grammar a forced `tool_choice` installs (a JSON schema, or an exact regex for a tool that takes only `{}`), and the `GuidedToolConstraint` a request implies. The caller passes in whether the request's own `guided_*` fields install a grammar. |
| `unified_stream` | The v2 stream adapter: one unified parser per choice turns raw model text into reasoning, content, and tool-call chunks, then fixes each choice's finish reason. |
| `stream_item` | The stream item that `unified_stream` reads. Dynamo implements it for its own item, so its per-chunk fields (`nvext`, metrics, completion evidence) stay in place. A chunk with a `SourceSlot` (from `dynamo-parsers`) also gets, for each choice, the input byte ranges its text came from. |
| `guided` | `GuidedToolConstraint`: the tool-output grammar a request installed. |

```rust
use dynamo_chat_parsing::{ParserVersion, ToolProcessingRoute, tool_processing_route};

let version = ParserVersion::try_from(std::env::var("DYN_PARSER_VERSION").ok().as_deref())?;
let route = tool_processing_route(Some("glm45"), Some("glm45"), &request, version)?;
assert_eq!(route, ToolProcessingRoute::Unified("glm47"));
```

## Tests

```bash
cargo test -p dynamo-chat-parsing
```

`tests/dynamo_llm_parity.rs` replays the output of dynamo-llm's own code, recorded by `tests/data/recorder.rs`, and checks that this crate gives the same answer on the same input. `tests/moved_from_dynamo_llm.rs` holds dynamo-llm's tests of these rules.

`tests/unified_stream_parity.rs` does the same for the v2 stream adapter, with output recorded by `tests/data/unified_stream_recorder.rs`. dynamo-llm's unit tests for the adapter are in `src/unified_stream_tests.rs`.

`tests/tool_choice_parity.rs` checks `tool_choice` against a copy of dynamo-llm's code, over generated tool lists, tool choices, and requests. dynamo-llm's unit tests for it are in `src/tool_choice_tests.rs`.
