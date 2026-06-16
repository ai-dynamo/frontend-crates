<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# conformance

Parser conformance fixtures, fixture-based Rust tests, and HTML renderers for frontend-crates.

## Ownership

Parser v1/v2 terminology, migration steps, fixture ownership, and temporary sync rules are documented in [`../PARSERS-V2-MIGRATION-PLAN.md`](../PARSERS-V2-MIGRATION-PLAN.md). New streaming parser authors should also read [`../parsers_v2/README.md`](../parsers_v2/README.md); it explains the vLLM-shaped Rust parser contract, the v2 fixture schema, and the exact `conformance/toolcalling/*` files to add. This README covers conformance layout, render outputs, and test commands.

## Layout

```
conformance/
├── toolcalling/fixtures/<family>/*.yaml          # legacy v1 tool-calling batch cases
├── reasoning/fixtures/<family>/*.yaml             # legacy reasoning v1 cases
├── toolcalling/fixtures-stream-v2/<family>/*.yaml # frontend-crate-owned v2 stream cases
├── toolcalling/fixtures-batch-on-stream-v2/<family>/*.yaml # frontend-crate-owned complete-text-through-stream cases
├── tests/*.rs                                     # Rust fixture tests
└── utils/                                         # render, check, and record helpers
```

## Render Outputs

| Output | Command | Parser version | Fixture version |
|---|---|---|---|
| v1 parity HTML | `conformance/utils/render_table_v1.sh` | v1 Dynamo-synced parser code through old Dynamo `generate_parity_table.py` | v1 Dynamo-synced tool-calling and reasoning fixtures; output stays under `conformance/utils/.stage/tests/parity/PARITY_v1.html` so old relative links resolve. |
| v2 conformance HTML | `conformance/utils/render_table_v2.sh` | Mixed bridge table: `TC batch (v1)` and reasoning tabs use v1 Dynamo-synced parser code; `TC batch-on-stream (v2)` and `TC stream (v2)` use Dynamo parser v2 code. | `TC batch (v1)` uses v1 batch fixtures; `TC batch-on-stream (v2)` uses v1 batch fixtures plus v2 batch-on-stream overlays; `TC stream (v2)` uses v2 stream fixtures; reasoning tabs use v1 reasoning fixtures. The default example output is `conformance/CONFORMANCE.html`, and the render script also accepts a custom output path. |

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
| `parity_toolcalling_batch_via_stream` | Dynamo parser v2 in `parsers_v2/src/tool_calling/*` | v1 batch fixtures in `conformance/toolcalling/fixtures/` plus v2 overlays in `conformance/toolcalling/fixtures-batch-on-stream-v2/` | Feeds complete batch text into the v2 stream parser and compares assembled calls to the committed batch-on-stream expectations. |
| `parity_toolcalling_stream` | Dynamo parser v2 in `parsers_v2/src/tool_calling/*` | v2 stream fixtures in `conformance/toolcalling/fixtures-stream-v2/` | Checks token-id or text streaming paths per chunk, then checks assembled calls. |

The fixture `family` field is the parser name, the same value Dynamo's `parse_tool_calls_batch` binding takes for v1. Legacy v1 fixtures use `expected.dynamo`, `expected.vllm`, and `expected.sglang`; v2 fixtures should use explicit implementation keys such as `expected.dynamo_rust`, `expected.vllm_rust`, `expected.vllm_python`, and `expected.sglang_python`.

Reasoning fixtures are rendered in the v2 HTML table; a Rust fixture harness for reasoning is still a follow-up.

## Refreshing Legacy Fixtures (v1)

Parser fixture sync from Dynamo is retired. Update v1 fixtures through normal frontend-crates PRs and verify the renderers listed in [`../PARSERS-V2-MIGRATION-PLAN.md`](../PARSERS-V2-MIGRATION-PLAN.md#temporary-sync-commands).

## Adding Streaming Parser V2 Fixtures

Use [`../parsers_v2/README.md`](../parsers_v2/README.md#fixture-files-to-add) for the parser-side checklist. In conformance, a new streaming family normally needs `conformance/toolcalling/fixtures-stream-v2/<family>/TOOLCALLING.streamv2.*.yaml` and `conformance/toolcalling/fixtures-batch-on-stream-v2/<family>/TOOLCALLING.batch*.yaml`; add `conformance/toolcalling/fixtures/<family>/TOOLCALLING.batch*.yaml` only when the v1 batch corpus does not already contain that family or taxonomy case.

The v2 stream fixture schema is documented in [`toolcalling/fixtures-stream-v2/README.md`](toolcalling/fixtures-stream-v2/README.md). Capture and render commands are documented in [`utils/README.md`](utils/README.md).
