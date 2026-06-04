<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# conformance

Vendored Dynamo parser parity fixtures plus a Rust harness that runs them against `dynamo-parsers`.

The fixtures are mirrored from `ai-dynamo/dynamo`'s `tests/parity/<stage>/fixtures/` and kept current by `scripts/sync-from-dynamo.sh`. Do not hand-edit them — edits would drift from dynamo and `sync-check` CI would flag it.

## Layout

```
conformance/
├── toolcalling/fixtures/<family>/*.yaml   # tool-calling cases (vendored, 209)
├── reasoning/fixtures/<family>/*.yaml      # reasoning cases (vendored, 24)
└── tests/parity_toolcalling.rs             # the harness
```

## Running the tests

Use the repo's pinned toolchain (Rust 1.93.1 via rustup; a system `cargo` may be too old for the workspace):

```bash
# tool-calling batch parity, all families:
cargo test -p dynamo-conformance --test parity_toolcalling

# same, but print the per-run case count (e.g. "606/606 cases passed"):
cargo test -p dynamo-conformance --test parity_toolcalling -- --nocapture

# as part of the whole workspace (what CI runs):
cargo test --workspace
```

Each `batch` case's `model_text` is fed through `detect_and_parse_tool_call_with_recovery(text, Some(family), tools)` and the result (`calls` + `normal_text`) is compared to `expected.dynamo`. The fixture `family` field is the parser name — the same value dynamo's `parse_tool_calls_batch` binding takes.

## Scope

- **Tool-calling, batch mode** is wired today (all families pass).
- **Streaming** is out of scope here: the streaming path lives in dynamo's `lib/llm` jail, not in this crate.
- **Reasoning** fixtures are vendored but not yet exercised — a `parity_reasoning.rs` harness (using the reasoning API, token-aware for gpt-oss) is a follow-up.

## Refreshing the fixtures

```bash
git clone --depth 1 --branch main https://github.com/ai-dynamo/dynamo.git /tmp/dynamo
scripts/sync-from-dynamo.sh /tmp/dynamo            # check for drift (dry-run)
scripts/sync-from-dynamo.sh --apply /tmp/dynamo    # apply the update
```

`sync-check` CI runs the dry-run against `dynamo@main` on every PR, so stale fixtures surface as a failed check rather than silently rotting.
