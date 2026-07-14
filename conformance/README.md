<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# conformance

Parser conformance fixtures, fixture-based Rust tests, and HTML renderers for frontend-crates.

## Ownership

Parser v1/v2 terminology, migration steps, fixture ownership, and temporary sync rules are documented in [`../docs/PARSERS-V2-MIGRATION-PLAN.md`](../docs/PARSERS-V2-MIGRATION-PLAN.md). New streaming parser authors should also read [`../parsers/v2/README.md`](../parsers/v2/README.md); it explains the vLLM-shaped Rust parser contract, the v2 fixture schema, and the exact `conformance/toolcalling/*` files to add. This README covers conformance layout, render outputs, and test commands.

## Parser paths and modes (universal convention)

- **Dynamo v1** = the batch parser, used two ways: **batch** (the complete text parsed in one call) and **jail+batch** (streaming input is buffered — "jailed" — until a call completes, then batch-parsed and emitted all at once). The jail never streams a call's name/arguments incrementally; only text outside the jail passes through as it arrives.
- **Dynamo v2** = the **streaming** parser (primary mode): emits text and tool-call deltas per chunk, as input arrives. It can also take batch input (the whole text fed as one chunk) — the `batch-on-stream` rows.
- **Per-chunk cells show WHEN output reaches the consumer.** Streaming parsers emit whenever something is parseable, so their cells carry deltas at real chunk positions. The v1 jail bursts at end-of-call — its captures record emission order, not per-chunk timing — so its per-chunk cells stay `—` with a "(bursts at end of call; per-chunk timing not recorded)" header note, and its output appears only in the `assembled` row.
- In the rendered tables, the TC (stream) tab's **default Reference is Dynamo v1 (jail+batch)** — the one stream path with coverage on every family — so every row shows data by default. Star **Dynamo v2** as the Reference to see v2's streaming coverage; families v2 doesn't implement yet gray out as "not implemented".

## Layout

```
conformance/
├── fixtures-manifest.json                         # pins the active HuggingFace fixture snapshot
├── tests/*.rs                                     # Rust fixture tests (fixtures downloaded from HF on first run)
└── utils/                                         # render, check, and record helpers
```

Fixture YAMLs are NOT in the repo. They live on HuggingFace (`ai-dynamo/conformance-fixtures`, private — set a read-capable `HF_TOKEN`) and are downloaded automatically on first use. HF snapshot layout:

```
toolcalling/fixtures-batch-v1/<family>/           # v1 tool-calling batch cases
toolcalling/fixtures-stream-v2/<family>/          # v2 stream cases
toolcalling/fixtures-batch-on-stream-v2/<family>/ # v2 complete-text-through-stream cases
reasoning/fixtures-v1/inputs/<family>/            # v1 reasoning cases
```

## Render Outputs

| Output | Command | Parser version | Fixture version |
|---|---|---|---|
| v1 parity HTML | `conformance/utils/render_table_v1.sh` | v1 Dynamo-synced parser code through old Dynamo `generate_parity_table.py` | v1 Dynamo-synced tool-calling and reasoning fixtures; output stays under `conformance/utils/.stage/tests/parity/PARITY_v1.html` so old relative links resolve. |
| v2 conformance HTML | `conformance/utils/render_table_v2.sh` | Mixed bridge table: `TC batch (v1)` and reasoning tabs use v1 Dynamo-synced parser code; `TC batch-on-stream (v2)` and `TC stream (v2)` use Dynamo parser v2 code. | `TC batch (v1)` uses v1 batch fixtures; `TC batch-on-stream (v2)` uses v1 batch fixtures plus v2 batch-on-stream overlays; `TC stream (v2)` uses v2 stream fixtures; reasoning tabs use v1 reasoning fixtures. The default example output is `conformance/CONFORMANCE_v2.html`, and the render script also accepts a custom output path. |

## Running the tests

Use the repo's pinned toolchain (Rust 1.96.1 via rustup; a system `cargo` may be too old for the workspace):

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
| `parity_toolcalling` | v1 Dynamo-synced batch parser in `parsers/src/tool_calling/` | v1 batch fixtures from HF (`toolcalling/fixtures-batch-v1/`) | Each `batch` case's `model_text` is fed through `detect_and_parse_tool_call_with_recovery(text, Some(family), tools)` and compared to `expected.dynamo`. |
| `parity_toolcalling_batch_via_stream` | Dynamo parser v2 in `parsers_v2/src/tool_calling/*` | v1 batch fixtures from HF (`toolcalling/fixtures-batch-v1/`) plus v2 overlays (`toolcalling/fixtures-batch-on-stream-v2/`) | Feeds complete batch text into the v2 stream parser and compares assembled calls to the HF-hosted batch-on-stream expectations. |
| `parity_toolcalling_stream` | Dynamo parser v2 in `parsers_v2/src/tool_calling/*` | v2 stream fixtures from HF (`toolcalling/fixtures-stream-v2/`) | Checks token-id or text streaming paths per chunk, then checks assembled calls. |

The fixture `family` field is the parser name, the same value Dynamo's `parse_tool_calls_batch` binding takes for v1. Legacy v1 fixtures use `expected.dynamo`, `expected.vllm`, and `expected.sglang`; v2 fixtures should use explicit implementation keys such as `expected.dynamo_rust`, `expected.vllm_rust`, `expected.vllm_python`, and `expected.sglang_python`.

Reasoning fixtures are rendered in the v2 HTML table; a Rust fixture harness for reasoning is still a follow-up.

## Refreshing Legacy Fixtures (v1)

Parser fixture sync from Dynamo is retired. Update v1 fixtures through normal frontend-crates PRs and verify the renderers listed in [`../docs/PARSERS-V2-MIGRATION-PLAN.md`](../docs/PARSERS-V2-MIGRATION-PLAN.md#temporary-sync-commands).

## Adding Streaming Parser V2 Fixtures

Use [`../parsers_v2/README.md`](../parsers_v2/README.md#fixture-files-to-add) for the parser-side checklist. In conformance, a new streaming family normally needs YAML files under `toolcalling/fixtures-stream-v2/<family>/` and `toolcalling/fixtures-batch-on-stream-v2/<family>/`; add `toolcalling/fixtures-batch-v1/<family>/` entries only when the v1 batch corpus does not already contain that family or taxonomy case. Capture locally with `capture.sh`, then publish to HuggingFace with `package_and_publish.py` — do not commit fixture YAMLs to the repo.

The v2 stream fixture schema is documented in [`toolcalling/fixtures-stream-v2/README.md`](toolcalling/fixtures-stream-v2/README.md). Capture and render commands are documented in [`utils/README.md`](utils/README.md).
