# Parsers V2 Migration Plan

This is the single source of truth for the parser v1/v2 bridge, the temporary Dynamo sync boundary, and the final migration out of `parsers_v2*`.

## Terminology

`v1` means Dynamo-synced parser code, fixtures, and old parity renderer behavior. During the bridge period, v1 lives in `parsers/src/`, `conformance/toolcalling/fixtures/`, `conformance/reasoning/fixtures/`, `conformance/utils/tests/parity/`, and `conformance/utils/lib/parsers/*_CASES.md`.

`v2` means frontend-crate-owned parser code, Python binding code, stream fixtures, and conformance renderer behavior. During the bridge period, v2 lives in `parsers_v2/`, `parsers_v2-py/`, `conformance/toolcalling/fixtures-stream-v2/`, `conformance/utils/generate_conformance_table_v2.py`, and `conformance/utils/conformance_table_v2.html.j2`.

Use `Dynamo parser v2` as the parser label. The fixture key `expected.dynamo` and helper subcommand `check_v2.sh dynamo` are compatibility labels for local parser output; ownership during the bridge is still described separately as frontend-crates-owned.

## Why The Bridge Exists

The current split exists because frontend-crates still needs to show and validate the old Dynamo view while building Dynamo parser v2. v1 stays resettable to Dynamo so `PARITY_v1.html` can show what old Dynamo would have generated. v2 stays outside the sync target so new streaming parser work cannot be clobbered by a Dynamo rsync.

Do not move v2 parser code into `parsers/src/` until Dynamo consumes the released frontend-crates parser crate directly and parser-source rsync stops. Until then, `parsers/src/` is a v1 mirror.

## Current Bridge Layout

| Area | Owner during bridge | Rule |
|---|---|---|
| `parsers/src/` | v1 Dynamo-synced | Resettable to Dynamo. Do not put v2 work here yet. |
| `parsers/tests/` | v1 Dynamo-synced when present upstream | Resettable to Dynamo. |
| `conformance/toolcalling/fixtures/` | v1 Dynamo-synced | Batch tool-calling fixtures mirrored from Dynamo. Do not hand-edit for v2 behavior. |
| `conformance/reasoning/fixtures/` | v1 Dynamo-synced | Reasoning fixtures mirrored from Dynamo. |
| `conformance/utils/tests/parity/` | v1 Dynamo-synced | Old Dynamo parity generator package. Keep it close to Dynamo so `PARITY_v1.html` stays old-output compatible. |
| `conformance/utils/lib/parsers/TOOLCALLING_CASES.md` and `REASONING_CASES.md` | v1 Dynamo-synced | Case docs used by the old v1 renderer. |
| `parsers_v2/src/tool_calling/*` | v2 frontend-crate-owned | Temporary Rust home for new streaming tool-calling parsers. Current Harmony implementation is `parsers_v2/src/tool_calling/harmony.rs`. |
| `parsers_v2-py/` | v2 frontend-crate-owned | Temporary PyO3 package exposing the v2 parser to Python as `dynamo_parsers_v2`. |
| `conformance/toolcalling/fixtures-stream-v2/` | v2 frontend-crate-owned | Stream fixtures for v2 parser behavior. |
| `conformance/utils/generate_conformance_table_v2.py` and `conformance/utils/conformance_table_v2.html.j2` | v2 frontend-crate-owned | New conformance table renderer. |

## Migration Steps

1. Keep the bridge split in this PR: the v1 mirror stays under the v1 paths above; v2 parser, binding, stream fixture, and renderer work stays under the v2 paths above.
2. Release the frontend-crates parser crate. The release must include the v1 parser API Dynamo already uses plus the v2 streaming parser API needed for this conformance work.
3. Update Dynamo so it consumes that released crate directly instead of carrying synced parser source. After that Dynamo PR lands, stop syncing parser source from Dynamo into frontend-crates.
4. Remove the parser-source portion of the manual sync runbook. Keep fixture sync only if fixtures still need Dynamo as a source of truth during the transition.
5. Merge v1 and v2 inside frontend-crates: move frontend-crate-owned parser code from `parsers_v2/src/tool_calling/*` into the normal parser crate layout under `parsers/src/tool_calling/*`; move the Python binding surface from `parsers_v2-py` into the final parser Python binding package; remove the temporary `parsers_v2*` crate/package boundary; merge the old parity renderer and v2 conformance renderer into one owned renderer; retire temporary `_v2` names once the merged table is the only table.
6. Delete bridge-only artifacts after the merge, including the v1 sync whitelist entries that only existed to preserve old Dynamo output.

Do not do step 5 before step 3 lands in Dynamo. Until Dynamo consumes the released crate directly, the v1 mirror is useful because it shows exactly what old Dynamo would have generated.

## Final Target Shape

Rust parser code should end in the normal parser crate, not in a permanent `tool_calling_v2` module:

```text
parsers_v2/src/tool_calling/*  ->  parsers/src/tool_calling/*
```

The exact Harmony file layout can still be chosen during the merge. The expected direction is a family-owned module under `parsers/src/tool_calling/`, for example `parsers/src/tool_calling/harmony/streaming.rs` or `parsers/src/tool_calling/streaming/harmony.rs`, with public exports through the normal `dynamo-parsers` API.

Python bindings should also lose the v2 package name after the bridge:

```text
parsers_v2-py / dynamo_parsers_v2  ->  final parser Python binding package
```

The final package name should not carry `v2`. Use the parser crate's release packaging decision for the exact module name.

## Temporary Sync Commands

Use the general sync script for the ordinary Dynamo mirrors and v1 fixtures:

```bash
scripts/sync-from-dynamo.sh /path/to/dynamo          # dry-run
scripts/sync-from-dynamo.sh --apply /path/to/dynamo  # apply
```

Use the parser-only sync script only when deliberately refreshing the temporary v1 parser mirror and old parity renderer:

```bash
scripts/manual-sync-parsers.sh /path/to/dynamo          # dry-run
scripts/manual-sync-parsers.sh --apply /path/to/dynamo  # apply
```

After applying a parser sync, verify both renderers:

```bash
conformance/utils/render_parity_v1.sh
conformance/utils/render_table_v2.sh
```

## Manual Version Pins

`sync-from-dynamo.sh` syncs `src/`, `tests/`, and fixtures but never dependency versions. It lists `Cargo.toml` as manual-review and never auto-applies it. Check this table on every sync. `last-synced` is the value verified against Dynamo `main` on 2026-06-04; re-verify against current `main`, not a stale local checkout.

| Pin | frontend-crates file | Dynamo file | last-synced value | Notes |
|---|---|---|---|---|
| `openai-harmony` (Rust crate) | root `Cargo.toml` `[workspace.dependencies]` | `lib/parsers/Cargo.toml` | `0.0.3` (both) | Build matches. The real risk is the runtime gap below. |
| `openai_harmony` (Python, in the engine containers) | recorded as `captured_with` in `conformance/toolcalling/fixtures-stream-v2/harmony*/` | n/a (engine container) | vLLM container `0.0.8`, SGLang container `0.0.4` | The gpt-oss/Harmony parser behavior is defined by the Harmony grammar; a Rust-`0.0.3`-vs-Python-`0.0.8` gap is the most likely source of a Harmony conformance mismatch. Re-check the in-container version after any vLLM/SGLang bump. Consider bumping the Rust crate to match. |
| `fastokens` (Rust) | root `Cargo.toml` | root `Cargo.toml` | frontend-crates `0.1.0` vs Dynamo `0.2.0` (skew) | Tokenizer backend; low parser conformance impact but the one hard Rust skew. Bump to `0.2.0` to stay honest. |
| `vllm` / `sglang` (Python engine pins) | `conformance/utils/pyproject.stub.toml` | `pyproject.toml` | `vllm==0.22.0`, `sglang==0.5.12.post1` | Matches current `main`. After bumping, re-capture peer streaming data and update `captured_with`. |
| Shared crate versions + parser deps | `parsers/`, `tokenizers/`, `protocols/`, `renderer/` `Cargo.toml` + root | `lib/*/Cargo.toml` + root | all `1.3.0`; async-openai `0.34`, tokenizers `0.21.4`, tiktoken-rs `0.9`, rustpython-parser `0.4.0`, minijinja `2.20.0`; Rust `1.93.1` | Should always match the Dynamo workspace; verify on sync. |

## Frontend-Crate-Only Files

These files have no upstream Dynamo counterpart. Never overwrite them during a sync.

| File | Purpose |
|---|---|
| `parsers_v2/` | Temporary Rust parser crate for v2 streaming work. |
| `parsers_v2-py/` | Temporary PyO3 binding crate/package for v2 streaming work. |
| `conformance/toolcalling/fixtures-stream-v2/` | v2 stream fixtures. |
| `conformance/utils/_common.sh` | Shared stage builder for conformance scripts. |
| `conformance/utils/check_v2.sh` | Runs local-parser, vLLM, and SGLang checks against staged fixtures; v2 local-parser checks run Dynamo parser v2 code. |
| `conformance/utils/render_table_v2.sh` | Renders `conformance/CONFORMANCE_v2.html` with the v2 conformance generator. |
| `conformance/utils/render_parity_v1.sh` | Renders `.stage/tests/parity/PARITY_v1.html` with old Dynamo `generate_parity_table.py`. |
| `conformance/utils/record_v2.sh` | Records Dynamo parser v2 stream fixture data. |
| `conformance/utils/validate.py` | Cross-implementation validation via `docker exec` or pip. |
| `conformance/utils/build_stream_fixtures.py` | Builds v2 per-chunk stream fixtures from source cases and captured engine output. |
| `conformance/utils/capture_stream.py` | Captures vLLM/SGLang per-chunk stream output inside engine containers. |
| `conformance/utils/capture_all_families.sh` and `capture_all_families_driver.py` | Batch-captures vLLM/SGLang stream output for non-Harmony families. |
| `conformance/utils/capture_harmony_batch_stream.py` | Captures engine streaming parsers on Harmony batch samples for the batch-on-stream tab. |
| `conformance/utils/capture_vllm_harmony.py` | Captures vLLM token-native Harmony stream path for one stream fixture. |
| `conformance/utils/merge_batch_stream.py` | Merges per-engine batch-on-stream captures into `harmony_batch_stream.json`. |
| `conformance/utils/harmony_batch_stream.json` | Recorded batch-on-stream comparison data consumed by the v2 table. |
| `conformance/utils/generate_conformance_table_v2.py` | frontend-crate-owned conformance renderer; staged into `tests/parity/` at render time. |
| `conformance/utils/conformance_table_v2.html.j2` | frontend-crate-owned conformance HTML template; staged into `tests/parity/` at render time. |
| `conformance/utils/README.md` | Usage docs for validate, render, and record helpers. |
| `conformance/utils/.gitignore` | Excludes `.stage*/`, local `CONFORMANCE*.html` outputs, and Python bytecode. |
| `conformance/utils/tests/__init__.py` | Empty package root for `.stage/` imports. |
| `parsers/Cargo.toml` | Inlined for standalone publishing. |
