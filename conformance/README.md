<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# conformance

Parser conformance fixtures, fixture-based Rust tests, and HTML renderers for frontend-crates.

## Terminology

`v1` means Dynamo-synced code and fixtures. In this bridge period, `parsers/src/`, `conformance/toolcalling/fixtures/`, `conformance/reasoning/fixtures/`, and the old parity renderer are the Dynamo view.

`v2` means frontend-crate-owned code and fixtures. Do not call v2 parser code "Dynamo code"; the v2 parser lives in frontend-crates under `parsers_v2/`, and its stream fixtures live under `conformance/toolcalling/fixtures-stream-v2/`.

Some fixture fields and commands still use `dynamo` as the local-parser key, for example `expected.dynamo` and `check_v2.sh dynamo`. That is a compatibility label in the conformance schema, not ownership. In docs and HTML copy, say "Dynamo code" only for v1 and "frontend-crate code" for v2.

## Migration Plan

Why: the current v1/v2 split exists because v1 parser source, fixtures, and the old parity renderer still mirror Dynamo during the bridge period, while new streaming parser work is owned in frontend-crates under `parsers_v2/` and `conformance/toolcalling/fixtures-stream-v2/`.

How today works: v1 remains the Dynamo-synced view. v2 uses frontend-crates-owned parser code and stream fixtures. The v2 renderer stages the synced batch/reasoning data, overlays the v2 stream fixtures, and writes `conformance/CONFORMANCE_v2.html`. The v2 fixture update flow does not update v1 fixtures or v1 HTML.

Steps:

1. Keep the bridge split in this PR: the v1 mirror stays under `parsers/src/`, `conformance/toolcalling/fixtures/`, `conformance/reasoning/fixtures/`, and `conformance/utils/tests/parity/`; v2 parser and stream work stays under `parsers_v2*`, `conformance/toolcalling/fixtures-stream-v2/`, and the v2 renderer files in `conformance/utils/`.
2. Release the frontend-crates parser crate. The release must include the v1 parser API Dynamo already uses plus the v2 streaming parser API needed for this conformance work.
3. Update Dynamo so it consumes that released crate directly instead of carrying synced parser source. After that Dynamo PR lands, stop syncing parser source from Dynamo into frontend-crates.
4. Remove the parser-source portion of the manual sync runbook. Keep fixture sync only if fixtures still need a Dynamo source of truth during the transition.
5. Merge v1 and v2 inside frontend-crates: fold the v2 streaming parser into the normal parser crate layout, merge the old parity renderer and the v2 conformance renderer into one owned renderer, and retire temporary `_v2` names once the merged table is the only table.
6. Delete bridge-only artifacts after the merge, including the v1 sync whitelist entries that only existed to preserve old Dynamo output.

Do not do step 5 before step 3 lands in Dynamo. Until Dynamo consumes the released crate directly, the v1 mirror is useful because it shows exactly what old Dynamo would have generated.

## Fixture Ownership

The Dynamo-synced v1 fixtures are mirrored from `ai-dynamo/dynamo`'s `tests/parity/<stage>/fixtures/` and kept current by `scripts/sync-from-dynamo.sh`. Do not hand-edit them; edits would drift from Dynamo and `sync-check` CI would flag it.

The v2 stream fixtures live under `conformance/toolcalling/fixtures-stream-v2/` and are owned in frontend-crates. Update them with the v2 fixture flow in `conformance/utils/README.md`.

The manual sync boundary for parser source, vendored renderer files, and parser case docs is documented in [`../PARSERS-SYNC.md`](../PARSERS-SYNC.md).

## Layout

```
conformance/
├── toolcalling/fixtures/<family>/*.yaml          # Dynamo-synced v1 tool-calling batch cases
├── reasoning/fixtures/<family>/*.yaml             # Dynamo-synced reasoning v1 cases
├── toolcalling/fixtures-stream-v2/<family>/*.yaml # frontend-crates-owned v2 stream cases
├── tests/*.rs                                     # Rust fixture tests
└── utils/                                         # render, check, and record helpers
```

## Render Outputs

| Output | Command | Source view |
|---|---|---|
| v1 parity HTML | `conformance/utils/render_parity_v1.sh` | Dynamo-synced v1 fixtures and old Dynamo `generate_parity_table.py`; output stays under `conformance/utils/.stage/tests/parity/PARITY_v1.html` so old relative links resolve. |
| v2 conformance HTML | `conformance/utils/render_table_v2.sh` | Mixed bridge table: `TC batch (v1)` is Dynamo code on v1 batch fixtures; `TC batch-on-stream (v2)` is frontend-crate v2 code on v1 batch fixtures; `TC stream (v2)` is frontend-crate v2 code on v2 stream fixtures; reasoning tabs are v1 Dynamo code on v1 reasoning fixtures. Output is `conformance/CONFORMANCE_v2.html`. |

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
| `parity_toolcalling_batch_via_stream` | frontend-crate v2 stream parser in `parsers_v2/src/lib.rs` | v1 Harmony batch fixtures in `conformance/toolcalling/fixtures/harmony/` | Feeds complete batch text into the v2 stream parser and compares assembled calls to the v1 batch expected output. |
| `parity_toolcalling_stream` | frontend-crate v2 stream parser in `parsers_v2/src/lib.rs` | v2 stream fixtures in `conformance/toolcalling/fixtures-stream-v2/` | Checks token-id and text streaming paths per chunk, then checks assembled calls. |

The fixture `family` field is the parser name, the same value Dynamo's `parse_tool_calls_batch` binding takes for v1. The `expected.dynamo` fixture key remains the local-parser output key even when the local parser is frontend-crate v2 code.

Reasoning fixtures are Dynamo-synced and rendered in the v2 HTML table; a Rust fixture harness for reasoning is still a follow-up.

## Refreshing Dynamo-Synced Fixtures (v1)

```bash
git clone --depth 1 --branch main https://github.com/ai-dynamo/dynamo.git /tmp/dynamo
scripts/sync-from-dynamo.sh /tmp/dynamo            # check for drift (dry-run)
scripts/sync-from-dynamo.sh --apply /tmp/dynamo    # apply the update
```

`sync-check` CI runs the dry-run against `dynamo@main` on every PR, so stale fixtures surface as a failed check rather than silently rotting. This refreshes the Dynamo-synced v1 fixtures; it does not update the frontend-crates-owned v2 stream fixtures.
