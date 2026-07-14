# Parsers V2 Migration Plan

This is the single source of truth for the parser v1/v2 split inside frontend-crates and the final migration to a single parser crate.

## Directory Layout (current)

All three parser crates live under `parsers/`, grouped but still separately packaged. This grouping is packaging-neutral: the crate names, versions, and the `dynamo_parsers_v2` Python module name are unchanged, so Dynamo (which pins `dynamo-parsers` and `dynamo-parsers-v2` from crates.io) is unaffected.

| Directory | Crate | Published? |
|---|---|---|
| `parsers/v1/` | `dynamo-parsers` (stable batch parser, the crate to depend on) | crates.io |
| `parsers/v2/` | `dynamo-parsers-v2` (WIP streaming parser, `0.x`) | crates.io |
| `parsers/v2-py/` | `dynamo-parsers-v2-py` (PyO3 binding, module `dynamo_parsers_v2`) | **no** — test-only, `publish = false` |

`parsers/v2-py` is the conformance harness's Python binding. It is a cdylib (a Python extension module, useless as a crates.io dependency), has no consumer outside this repo, and no PyPI publish job exists — so it is not published.

## Terminology

`v1` means the stable batch parser crate (`dynamo-parsers`, under `parsers/v1/`), its legacy fixtures (`conformance/toolcalling/fixtures-v1/`, `conformance/reasoning/fixtures/`), the old parity renderer (`conformance/utils/tests/parity/`), and `conformance/utils/lib/parsers/*_CASES.md`.

`v2` means the WIP streaming parser crate (`dynamo-parsers-v2`, under `parsers/v2/`), its Python binding (`parsers/v2-py/`), stream fixtures, batch-on-stream fixtures, and the conformance renderer (`conformance/utils/src/generate_conformance_table.py`, `conformance/utils/src/conformance_table.html.j2`).

Use `Dynamo parser v2` as the parser label. The fixture key `expected.dynamo` and helper subcommand `check.sh dynamo` are compatibility labels for local parser output.

## Why The Split Exists

Parser source and parser fixtures are **frontend-crates-owned** — `scripts/sync-from-dynamo.sh` no longer syncs `parsers/`, parser fixtures, or `conformance/utils/`. frontend-crates publishes `dynamo-parsers` to crates.io and Dynamo consumes it from there.

The v1/v2 split is kept because **v2 is still under active development**: it lives on a `0.x` line where breaking changes are free and expected, while `dynamo-parsers` (v1) is a stable, semver-checked `3.x` crate. Merging v2 into v1 now would either force major bumps on the stable crate or block v2's fast iteration under `cargo-semver-checks`. Keep them separate until the v2 streaming API stabilizes.

## Layout Details

| Area | Owner | Rule |
|---|---|---|
| `parsers/v1/src/` | v1 frontend-crates-owned | Stable batch parser (`dynamo-parsers`). Bug-fix only; do not put v2 streaming work here. |
| `parsers/v1/tests/` | v1 frontend-crates-owned | v1 crate tests. |
| `conformance/toolcalling/fixtures-v1/` | frontend-crates legacy v1 | Batch tool-calling fixtures retained for v1 behavior. Do not hand-edit for v2 behavior. |
| `conformance/reasoning/fixtures/` | frontend-crates legacy v1 | Reasoning fixtures rendered in the conformance table. |
| `conformance/utils/tests/parity/` | frontend-crates-owned | Parity generator package for `PARITY_v1.html`. |
| `conformance/utils/lib/parsers/TOOLCALLING_CASES.md` and `REASONING_CASES.md` | frontend-crates-owned | Case docs used by the v1 renderer. |
| `parsers/v2/src/tool_calling/*` | v2 frontend-crate-owned | Rust home for streaming tool-calling parsers. Current Harmony implementation is `parsers/v2/src/tool_calling/harmony.rs`. |
| `parsers/v2-py/` | v2 frontend-crate-owned | Test-only PyO3 package exposing the v2 parser to Python as `dynamo_parsers_v2`. Not published. |
| `conformance/toolcalling/fixtures-stream-v2/` | v2 frontend-crate-owned | Stream fixtures for v2 parser behavior. |
| `conformance/toolcalling/fixtures-batch-on-stream-v2/` | v2 frontend-crate-owned | Complete batch text captured through streaming parsers for stream-vs-batch comparison. |
| `conformance/utils/src/generate_conformance_table.py` and `conformance/utils/src/conformance_table.html.j2` | v2 frontend-crate-owned | Conformance table renderer. |

## Migration Steps

Already done:

- Parser source, fixtures, and conformance utilities are frontend-crates-owned; `sync-from-dynamo.sh` no longer touches `parsers/`.
- `dynamo-parsers` (v1) is published to crates.io and consumed by Dynamo from there.
- All three parser crates are grouped under `parsers/{v1,v2,v2-py}` (packaging-neutral; names unchanged).
- `dynamo-parsers-v2-py` is marked `publish = false` (test-only).
- The tokenizer test fixtures moved out of the top-level `llm/tests/data` into `tokenizers/tests/data` (removing the lone root `llm/` dir), and the `tokenizers/` crate was **detached from the Dynamo sync** as a result. The synced tokenizer tests hard-coded `../llm/tests/data`, so the fixtures could not move without either detaching or fragile post-sync path patching; `tokenizers/` is now frontend-crates-owned like `parsers/`. Trade-off: `tokenizers/src` no longer receives Dynamo updates automatically — port upstream tokenizer changes by hand. Only `protocols/` and `renderer/` are still Dynamo-synced.

Remaining, gated on **v2's streaming API stabilizing** (do not start while v2 is still churning on `0.x`):

1. Stabilize the v2 streaming parser API and cut a `1.0` for `dynamo-parsers-v2`.
2. Merge v2 into `dynamo-parsers`: move the streaming parsers into the crate under a behavior-named module (not a permanent `v2` name — see below), fold the binding into the crate's final Python package, and retire the separate `dynamo-parsers-v2` / `dynamo-parsers-v2-py` crate boundary in one coordinated release.
3. Coordinate the Dynamo cutover: land a Dynamo PR that drops the `dynamo-parsers-v2` dependency and rewrites `use dynamo_parsers_v2::…` to the merged path. This is a breaking change — sequence it (publish merged crate → Dynamo PR → remove old crates), do not do it unilaterally.
4. Merge the old parity renderer and the v2 conformance renderer into one owned renderer; retire the `_v2` fixture/table names once the merged table is the only one.

## Final Target Shape

One `dynamo-parsers` crate, with the streaming parsers exposed under a **behavior-named** module, not a permanent `v2` name (`v2` is a migration label — shipping `dynamo_parsers::v2` just means renaming it again later, another breaking change):

```text
parsers/v2/src/tool_calling/*  ->  parsers/v1/src/tool_calling/*   (e.g. dynamo_parsers::tool_calling::streaming)
```

The `v1`/`v2` directory names are transitional; when the merge happens the surviving crate keeps the `dynamo-parsers` name and the directory split collapses. The Python binding should also lose the `v2` name (there is no v1 Python binding, so this is just a `dynamo_parsers_v2` → final-name module rename).

## Sync Commands

Use the general sync script for the ordinary non-parser Dynamo mirrors:

```bash
scripts/sync-from-dynamo.sh /path/to/dynamo          # dry-run
scripts/sync-from-dynamo.sh --apply /path/to/dynamo  # apply
```

Parser source, parser fixtures, and conformance utilities are frontend-crates-owned after the parser crate migration; do not sync them from Dynamo. After changing parser fixtures or conformance code, verify both renderers:

```bash
conformance/utils/render_table_v1.sh
conformance/utils/render_table_v2.sh
```

## Manual Version Pins

`sync-from-dynamo.sh` syncs non-parser `src/`, `tests/`, and tokenizer fixtures but never dependency versions. It lists `Cargo.toml` as manual-review and never auto-applies it. Check this table on every sync. `last-synced` is the value verified against Dynamo `main` on 2026-06-04; re-verify against current `main`, not a stale local checkout.

| Pin | frontend-crates file | Dynamo file | last-synced value | Notes |
|---|---|---|---|---|
| `openai-harmony` (Rust crate) | root `Cargo.toml` `[workspace.dependencies]` | `lib/parsers/Cargo.toml` | `0.0.3` (both) | Build matches. The real risk is the runtime gap below. |
| `openai_harmony` (Python, in the engine containers) | recorded as `captured_with` in `conformance/toolcalling/fixtures-stream-v2/harmony*/` | n/a (engine container) | vLLM container `0.0.8`, SGLang container `0.0.4` | The gpt-oss/Harmony parser behavior is defined by the Harmony grammar; a Rust-`0.0.3`-vs-Python-`0.0.8` gap is the most likely source of a Harmony conformance mismatch. Re-check the in-container version after any vLLM/SGLang bump. Consider bumping the Rust crate to match. |
| `fastokens` (Rust) | root `Cargo.toml` | root `Cargo.toml` | frontend-crates `0.1.0` vs Dynamo `0.2.0` (skew) | Tokenizer backend; low parser conformance impact but the one hard Rust skew. Bump to `0.2.0` to stay honest. |
| `vllm` / `sglang` (Python engine pins) | `conformance/utils/src/pyproject.stub.toml` | `pyproject.toml` | `vllm==0.22.0`, `sglang==0.5.12.post1` | Matches current `main`. After bumping, re-capture peer streaming data and update `captured_with`. |
| Shared crate versions + parser deps | `parsers/v1/`, `tokenizers/`, `protocols/`, `renderer/` `Cargo.toml` + root | `lib/*/Cargo.toml` + root | all `1.3.0`; async-openai `0.34`, tokenizers `0.21.4`, tiktoken-rs `0.9`, rustpython-parser `0.4.0`, minijinja `2.20.0`; Rust `1.96.1` | Should always match the Dynamo workspace; verify on sync. |

## Frontend-Crate-Only Files

These files have no upstream Dynamo counterpart. Never overwrite them during a sync.

| File | Purpose |
|---|---|
| `parsers/v2/` | Temporary Rust parser crate for v2 streaming work. |
| `parsers/v2-py/` | Temporary PyO3 binding crate/package for v2 streaming work. |
| `conformance/toolcalling/fixtures-stream-v2/` | v2 stream fixtures. |
| `conformance/toolcalling/fixtures-batch-on-stream-v2/` | v2 batch-on-stream fixture overlays. |
| `conformance/utils/src/_common.sh` | Shared stage builder for conformance scripts. |
| `conformance/utils/check.sh` | Runs local-parser, vLLM, and SGLang checks against staged fixtures; v2 local-parser checks run Dynamo parser v2 code. |
| `conformance/utils/render_table_v2.sh` | Renders `conformance/CONFORMANCE.html` with the v2 conformance generator. |
| `conformance/utils/render_table_v1.sh` | Renders `.stage/tests/parity/PARITY_v1.html` with old Dynamo `generate_parity_table.py`. |
| `conformance/utils/src/validate.py` | Cross-implementation validation via `docker exec` or pip. |
| `conformance/utils/src/build_stream_fixtures.py` | Builds v2 per-chunk stream fixtures from source cases and captured engine output. |
| `conformance/utils/src/capture.py` | In-container worker for an engine's tool-call parser: `--mode stream` (per-chunk), `--mode batch-on-stream` (batch text through the streaming parser), `--mode harmony-batch` (Harmony batch samples), `--mode harmony-chunk` (vLLM token-native Harmony). |
| `conformance/utils/src/capture_driver.py` | Host orchestrator: `--mode stream` batch-captures non-Harmony stream fixtures, `--mode batch-on-stream` rewrites the overlays, `--mode merge` builds `harmony_batch_stream.json`. |
| `conformance/utils/harmony_batch_stream.json` | Recorded batch-on-stream comparison data consumed by the v2 table. |
| `conformance/utils/src/generate_conformance_table.py` | frontend-crate-owned conformance renderer; staged into `tests/parity/` at render time. |
| `conformance/utils/src/conformance_table.html.j2` | frontend-crate-owned conformance HTML template; staged into `tests/parity/` at render time. |
| `conformance/utils/README.md` | Usage docs for validate, render, and record helpers. |
| `conformance/utils/.gitignore` | Excludes `.stage*/`, local `CONFORMANCE*.html` outputs, and Python bytecode. |
| `conformance/utils/tests/__init__.py` | Empty package root for `.stage/` imports. |
| `parsers/v1/Cargo.toml` | Inlined for standalone publishing. |
