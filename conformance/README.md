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
├── fixtures-manifest.json                         # pins the active fixture snapshot (sha256 per shard)
├── fixtures/                                      # LFS-tracked shard tarballs (the fixture store)
├── tests/*.rs                                     # Rust fixture tests (fixtures extracted from the store on first run)
└── utils/                                         # render, check, and record helpers
```

Fixture YAMLs are not loose in the repo. They live in `conformance/fixtures/` as git-lfs tarball shards (run `git lfs pull` on a fresh clone) and are extracted into `~/.cache/dynamo/conformance-fixtures/` automatically on first use. Snapshot layout:

```
toolcalling/fixtures-batch-v1/<family>/           # v1 tool-calling batch cases
toolcalling/fixtures-stream-v2/<family>/          # v2 stream cases
toolcalling/fixtures-batch-on-stream-v2/<family>/ # v2 complete-text-through-stream cases
reasoning/fixtures-v1/inputs/<family>/            # v1 reasoning cases
```

**Capture version dirs are append-only — NEVER delete or overwrite an existing `<impl>-<version>/` dir when re-recording.** Every version dir (`dynamo_v1-3.0.0` AND `dynamo_v2-0.1.11` AND `dynamo_v2-0.1.22`, `vllm_python-0.23.0` AND `vllm_python-0.24.0`, …) is capture history: the chart renders each one as a comparison candidate, and readers fold them ascending WITHIN an impl so the latest capture wins per case (`dynamo_v1` and `dynamo_v2` are separate impls and never fold together). Re-recording after a parser change writes the CURRENT crate version's dir alongside the old ones (`refresh_dynamo_captures.py` / `capture_dynamo_jail_stream.py` do this); re-recording at the same version replaces that one dir only. Deleting an old version dir silently destroys the chart's version-comparison columns — it happened once (version dirs were wiped by a refresh and had to be restored) and the tooling has since been made additive. If a dir looks obsolete, it still is not yours to delete: the git-lfs store keeps it, and the manifest-pinned snapshot is what the chart shows.

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

**Lifecycle:** v1 (`dynamo-parsers`, batch + jail) is **interim** — once v2 reaches parity, v1 is removed outright (not merged). v2 (`dynamo-parsers-v2`, streaming-first) is the **ultimate implementation, currently WIP**. New parser work goes to v2. The v1 fixture trees and parity tests exist to hold the line until then; expect them to be deleted together with v1.

| Test | Code under test | Fixtures | Notes |
|---|---|---|---|
| `parity_toolcalling` | v1 Dynamo-synced batch parser in `parsers/src/tool_calling/` | v1 batch fixtures (`toolcalling/fixtures-batch-v1/`) | Each `batch` case's `model_text` is fed through `detect_and_parse_tool_call_with_recovery(text, Some(family), tools)` and compared to `expected.dynamo_v1`. |
| `parity_toolcalling_batch_via_stream` | Dynamo parser v2 in `parsers_v2/src/tool_calling/*` | v1 batch fixtures (`toolcalling/fixtures-batch-v1/`) plus v2 overlays (`toolcalling/fixtures-batch-on-stream-v2/`) | Feeds complete batch text into the v2 stream parser and compares assembled calls to the batch-on-stream expectations. |
| `parity_toolcalling_stream` | Dynamo parser v2 in `parsers_v2/src/tool_calling/*` | v2 stream fixtures (`toolcalling/fixtures-stream-v2/`) | Checks token-id or text streaming paths per chunk, then checks assembled calls. |

The fixture `family` field is the parser name, the same value Dynamo's `parse_tool_calls_batch` binding takes for v1. Every fixture uses an explicit implementation key: `expected.dynamo_v1`, `expected.dynamo_v2`, `expected.vllm_rust`, `expected.vllm_python`, `expected.sglang_python`. Dynamo v1 and v2 are separate impls with separate version lineages (`dynamo_v1-3.0.0/`, `dynamo_v2-0.1.11/`) — exactly like the vLLM/SGLang runtime variants. Legacy spellings (`dynamo`, `dynamo_rust`, `vllm`, `sglang`) are still accepted on read via the alias table in `utils/src/impls.py`.

Reasoning fixtures are rendered in the v2 HTML table; a Rust fixture harness for reasoning is still a follow-up.

## Refreshing Legacy Fixtures (v1)

Parser fixture sync from Dynamo is retired. Update v1 fixtures through normal frontend-crates PRs and verify the renderers listed in [`../docs/PARSERS-V2-MIGRATION-PLAN.md`](../docs/PARSERS-V2-MIGRATION-PLAN.md#temporary-sync-commands).

## Adding Streaming Parser V2 Fixtures

Use [`../parsers_v2/README.md`](../parsers_v2/README.md#fixture-files-to-add) for the parser-side checklist. In conformance, a new streaming family normally needs YAML files under `toolcalling/fixtures-stream-v2/<family>/` and `toolcalling/fixtures-batch-on-stream-v2/<family>/`; add `toolcalling/fixtures-batch-v1/<family>/` entries only when the v1 batch corpus does not already contain that family or taxonomy case. Capture locally with `capture.sh`, then run `package_fixtures.py` to rebuild the LFS shard store + manifest, and commit both — do not commit loose fixture YAMLs to the repo.

The v2 stream fixture schema is documented in [`toolcalling/fixtures-stream-v2/README.md`](toolcalling/fixtures-stream-v2/README.md). Capture and render commands are documented in [`utils/README.md`](utils/README.md).

## Fixture Workflows

The four routine loops. All of them end the same way: `package_fixtures.py` rebuilds the LFS shard store + manifest, and you commit `conformance/fixtures/` + `conformance/fixtures-manifest.json` together (see [`utils/README.md`](utils/README.md#fixture-store-git-lfs)).

### 1. Capture a new vLLM/SGLang engine version (new peer shards)

1. Pin the new engine version in `utils/src/pyproject.stub.toml` — peer versions are read from there, never hardcoded.
2. Re-capture against containers running the new engines: `capture.sh stream` / `capture.sh batch-on-stream` (peer-only refresh of the batch-on-stream tree: `recapture_batch_on_stream.py`, which preserves the `vllm_rust`/`dynamo_v2` blocks).
3. Captures land as a NEW `<impl>-<newver>/` dir next to the existing ones — never edit or delete old version dirs. Lowest dir = full anchor, higher dirs = changed-only overlays; `resolve_*.py` folds them at read time.
4. `package_fixtures.py` → a new `<impl>-<newver>.tar.gz` shard appears in the store; existing shards are untouched (byte-identical rebuilds thanks to deterministic tars). Commit store + manifest.
5. Nothing else to wire up: the next `render_table_v2.sh` / `cargo test` runs `extract_fixtures.py` automatically and follows the manifest pin (re-extracts on a pin move, retargets the cache symlinks, instant on a hit), and the generator discovers the new `<impl>-<newver>` dir as a new candidate column with its version label taken from fixture provenance.

### 2. Fix a Dynamo parser and refresh its expected outputs

1. Fix the code under `parsers/v1/` or `parsers/v2/`.
2. `cargo test --workspace` — if the fix changes output, the parity tests FAIL. That is the regression gate working: decide whether the diff is a bug in your fix or an intended behavior change.
3. For an intended change: bump the crate version first (workflow 3), re-capture the Dynamo fixtures (`capture.sh dynamo-stream` / `dynamo-batch-on-stream`; v1 batch `expected.dynamo_v1` blocks are updated through the same capture flow), then `package_fixtures.py`.
4. Commit the parser fix + fixture shards + manifest + `Cargo.toml` bump in the SAME PR. CI is green only when the code and the pinned expectations agree again.

### 3. Version rule: fixture dirs carry the crate version that ships them

Capture stamps versions from the crates themselves — version dirs (`dynamo_v1-<ver>/`, `dynamo_v2-<ver>/`) and `captured_with.*` fields are read from `Cargo.toml` at capture time. So when a parser fix changes captured output, bump `parsers/v1/Cargo.toml` or `parsers/v2/Cargo.toml` to the NEXT release version BEFORE capturing. The new fixture dirs then carry exactly the version crates.io publishes when the PR merges (the manual-peg flow in [`../RELEASING.md`](../RELEASING.md#manual-version-peg-fixture-synced-releases)): outputs and release stay on one number by construction. Never rename or delete an old version dir — a re-record ADDS a dir.

### 4. What CI actually checks (the regression gate)

The `rust` CI job checks out the LFS store (`lfs: true`), extracts the manifest-pinned snapshot, and runs the parity tests — current parser code vs pinned expected YAML:

- `parity_toolcalling`: v1 code vs `expected.dynamo_v1` in the `dynamo_v1-<ver>/` dir of `fixtures-batch-v1`.
- `parity_toolcalling_stream`: v2 code vs `expected.dynamo_v2` folded from the LOWEST `dynamo_v2-<ver>/` dir — the v2 anchor. (The v1-jail reference lives in its own `dynamo_v1-3.0.0/` namespace and never enters the v2 fold. Overlay folding up to the pinned crate version is a follow-up; until then an intended v2 output change must be reflected in the anchor's expected blocks at re-capture.)
- `parity_toolcalling_batch_via_stream`: v2 code vs the `fixtures-batch-on-stream-v2` expectations.

A parser change that alters output fails CI until the fixtures are re-captured and committed (workflow 2) — CI compares Dynamo against the pinned shard YAMLs, nothing else. The `conformance-table` CI job additionally re-renders both HTML pages from the same pinned store and runs the chart-invariant guards (`utils/tests/test_chart_invariants.py`).
