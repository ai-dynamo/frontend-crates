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
| `unified_stream` | The v2 stream adapter: one unified parser per choice turns raw model text into reasoning, content, and tool-call chunks, then fixes each choice's finish reason. Also the batch parse of a whole reply. |
| `aggregator` | The non-streaming finalize: the chunks of a `stream: false` reply become one response. It re-parses the text when no tool call came through, drops cut-off calls, and can move reasoning to content. |
| `stream_item` | The stream item that `unified_stream` reads. Dynamo implements it for its own item, so its per-chunk fields (`nvext`, metrics, completion evidence) stay in place. |
| `guided` | `GuidedToolConstraint`: the tool-output grammar a request installed. |
| `chat_stream` | `parse_chat_stream`: the parse stage of a chat stream. It picks the stages for a request (the v2 adapter, or dynamo-parsers' v1 reasoning stream and the caller's v1 jail) and chains the stream policies after them. |
| `reasoning_off` | Whether a request turned reasoning off, and the fold of reasoning into content when it did. |
| `nonempty_content` | The `force_nonempty_content` contract on a stream, and whether a stream can hold back all output (the keep-alive predicate). |
| `tool_call_policy` | The request's tool-call policy on parsed output, and the assistant role on each choice's first delta. |
| `reasoning_usage` | The `reasoning_tokens` estimate in the usage chunk. |

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

`tests/unified_stream_parity.rs` does the same for the v2 stream adapter, with output recorded by `tests/data/unified_stream_recorder.rs`. dynamo-llm's unit tests for the adapter and its batch parse are in `src/unified_stream_tests.rs`.

`tests/aggregator_parity.rs` does the same for the aggregator, with output recorded by `tests/data/aggregator_recorder.rs`. dynamo-llm's unit tests for it are in `src/aggregator_tests.rs`.

`AggregatorOptions::for_request` builds the aggregator's options as dynamo-llm's HTTP handlers do; `tests/aggregator_options_parity.rs` checks it against a copy of their code.

`tests/stream_policies_parity.rs` does the same for `parse_chat_stream` and the stream policies, with two fixtures: a matrix of parser pairs, requests, and streams recorded by `tests/data/stream_policies_recorder.rs`, and every call that dynamo-llm's own tests make, captured by `tests/data/stream_policies_capture.rs`. dynamo-llm's unit tests for the policies are in `src/stream_policies_tests.rs`.
