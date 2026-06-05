<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# conformance

Parser conformance fixtures, fixture-based Rust tests, and HTML renderers for frontend-crates.

## Ownership

Parser v1/v2 terminology, migration steps, fixture ownership, and temporary sync rules are documented in [`../PARSERS-V2-MIGRATION-PLAN.md`](../PARSERS-V2-MIGRATION-PLAN.md). Read that before moving parser code, updating sync scripts, or changing fixture ownership. This README only covers conformance layout, render outputs, and test commands.

## Layout

```
conformance/
├── toolcalling/fixtures/<family>/*.yaml          # Dynamo-synced v1 tool-calling batch cases
├── reasoning/fixtures/<family>/*.yaml             # Dynamo-synced reasoning v1 cases
├── toolcalling/fixtures-stream-v2/<family>/*.yaml # frontend-crate-owned v2 stream cases
├── tests/*.rs                                     # Rust fixture tests
└── utils/                                         # render, check, and record helpers
```

## Render Outputs

| Output | Command | Parser version | Fixture version |
|---|---|---|---|
| v1 parity HTML | `conformance/utils/render_parity_v1.sh` | v1 Dynamo-synced parser code through old Dynamo `generate_parity_table.py` | v1 Dynamo-synced tool-calling and reasoning fixtures; output stays under `conformance/utils/.stage/tests/parity/PARITY_v1.html` so old relative links resolve. |
| v2 conformance HTML | `conformance/utils/render_table_v2.sh` | Mixed bridge table: `TC batch (v1)` and reasoning tabs use v1 Dynamo-synced parser code; `TC batch-on-stream (v2)` and `TC stream (v2)` use frontend-crate v2 parser code. | `TC batch (v1)` and `TC batch-on-stream (v2)` use v1 batch fixtures; `TC stream (v2)` uses v2 stream fixtures; reasoning tabs use v1 reasoning fixtures. Output is `conformance/CONFORMANCE_v2.html`. |

## Running the tests

Use the repo's pinned toolchain (Rust 1.93.1 via rustup; a system `cargo` may be too old for the workspace):

```bash
# tool-calling batch parity, all families:
cargo test --locked -p dynamo-conformance-fixtures-v2 --test parity_toolcalling

# same, but print fixture names and the per-run case count:
cargo test --locked -p dynamo-conformance-fixtures-v2 --test parity_toolcalling -- --nocapture

# as part of the whole workspace (what CI runs):
cargo test --workspace
```

The test package is named `dynamo-conformance-fixtures-v2` for historical compatibility, but the code ownership still follows the v1/v2 split.

| Test | Code under test | Fixtures | Notes |
|---|---|---|---|
| `parity_toolcalling` | v1 Dynamo-synced batch parser in `parsers/src/tool_calling/` | v1 batch fixtures in `conformance/toolcalling/fixtures/` | Each `batch` case's `model_text` is fed through `detect_and_parse_tool_call_with_recovery(text, Some(family), tools)` and compared to `expected.dynamo`. |
| `parity_toolcalling_batch_via_stream` | frontend-crate v2 stream parser in `parsers_v2/src/tool_calling/*`; current Harmony implementation in `parsers_v2/src/tool_calling/harmony.rs` | v1 Harmony batch fixtures in `conformance/toolcalling/fixtures/harmony/` | Feeds complete batch text into the v2 stream parser and compares assembled calls to the v1 batch expected output. |
| `parity_toolcalling_stream` | frontend-crate v2 stream parser in `parsers_v2/src/tool_calling/*`; current Harmony implementation in `parsers_v2/src/tool_calling/harmony.rs` | v2 stream fixtures in `conformance/toolcalling/fixtures-stream-v2/` | Checks token-id and text streaming paths per chunk, then checks assembled calls. |

The fixture `family` field is the parser name, the same value Dynamo's `parse_tool_calls_batch` binding takes for v1. The `expected.dynamo` fixture key remains the local-parser output key even when the local parser is frontend-crate v2 code.

Reasoning fixtures are Dynamo-synced and rendered in the v2 HTML table; a Rust fixture harness for reasoning is still a follow-up.

## Refreshing Dynamo-Synced Fixtures (v1)

Use the sync commands in [`../PARSERS-V2-MIGRATION-PLAN.md`](../PARSERS-V2-MIGRATION-PLAN.md#temporary-sync-commands). The v1 fixture refresh does not update frontend-crate-owned v2 stream fixtures.
