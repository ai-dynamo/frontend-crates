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

There are two separate version axes. Do not infer one from the other:

- **Implementation generation:** `dynamo-parsers` / `dynamo_v1` is the stable batch parser, also used behind the streaming jail. `dynamo-parsers-v2` / `dynamo_v2` is the newer crate containing the incremental tool-only parsers and Unified parsers. These implementation names stay unchanged.
- **Conformance convention:** `fixtures-stream-v1`, `fixtures-batch-on-stream-v1`, and `TOOLCALLING.streamv1.*` identify the legacy, non-Unified streaming convention. `fixtures-unified-v2` identifies Unified, where one parser owns the ordered reasoning, visible-text, and tool-call stream.

The non-Unified incremental parser was originally called `stream-v2` because it was intended to replace the original batch+jail implementation. Unified arrived later and also used the v2 name. That left two different “v2” conventions. The corpus rename resolves the collision: **streaming v1 means legacy/non-Unified streaming; v2 means Unified**. A `dynamo_v2-*` capture can therefore appear inside `fixtures-stream-v1`; the directory names the convention, while the capture key names the implementation.

In the current Dynamo runtime, parser names come from model deployment configuration. `DYN_ENABLE_EXPERIMENTAL_PARSERS_V2` enables both experimental routes; it does not choose one by itself. Routing checks Unified first. The exact `tool_call_parser=qwen3_coder` plus `reasoning_parser=qwen3` pair uses Qwen3 Unified when the flag is on. If no Unified pair matches, the flag allows eligible `qwen3_coder` or `deepseek_v4` requests to use the older non-Unified incremental parser. All remaining tool-parsing requests use batch+jail. DeepSeek V4.1 and Muse have separate Unified routes that do not depend on this flag. This repository rename does not change that routing.

## Why The Split Exists

Parser source, parser fixtures, and conformance utilities are **frontend-crates-owned**. frontend-crates publishes `dynamo-parsers` to crates.io and Dynamo consumes it from there.

The v1/v2 split is kept because **v2 is still under active development**: it lives on a `0.x` line where breaking changes are free and expected, while `dynamo-parsers` (v1) is a stable, semver-checked `3.x` crate. Merging v2 into v1 now would either force major bumps on the stable crate or block v2's fast iteration under `cargo-semver-checks`. Keep them separate until the v2 streaming API stabilizes.

## Layout Details

| Area | Owner | Rule |
|---|---|---|
| `parsers/v1/src/` | v1 frontend-crates-owned | Stable batch parser (`dynamo-parsers`). Bug-fix only; do not put v2 streaming work here. |
| `parsers/v1/tests/` | v1 frontend-crates-owned | v1 crate tests. |
| `conformance/toolcalling/fixtures-v1/` | frontend-crates legacy v1 | Batch tool-calling fixtures retained for v1 behavior. Do not hand-edit for v2 behavior. |
| `conformance/reasoning/fixtures/` | frontend-crates legacy v1 | Reasoning fixtures rendered in the conformance table. |
| `conformance/utils/src/tables/` | frontend-crates-owned | Shared table/markup modules the conformance generator imports. |
| `conformance/utils/lib/parsers/TOOLCALLING_CASES.md` and `REASONING_CASES.md` | frontend-crates-owned | Case docs used by the conformance renderer. |
| `parsers/v2/src/tool_calling/*` | v2 frontend-crate-owned | Rust home for streaming tool-calling parsers. Current Harmony implementation is `parsers/v2/src/tool_calling/harmony.rs`. |
| `parsers/v2-py/` | v2 frontend-crate-owned | Test-only PyO3 package exposing the v2 parser to Python as `dynamo_parsers_v2`. Not published. |
| `conformance/toolcalling/fixtures-stream-v1/` | legacy streaming convention | Per-chunk fixtures for non-Unified stream parsers, including `dynamo_v2` implementation captures. |
| `conformance/toolcalling/fixtures-batch-on-stream-v1/` | legacy streaming convention | Complete batch text captured through non-Unified stream parsers for stream-vs-batch comparison. |
| `conformance/utils/src/generate_conformance_table.py` and `conformance/utils/src/conformance_table.html.j2` | v2 frontend-crate-owned | Conformance table renderer. |

## Migration Steps

Already done:

- Parser source, fixtures, and conformance utilities are frontend-crates-owned.
- `dynamo-parsers` (v1) is published to crates.io and consumed by Dynamo from there.
- All three parser crates are grouped under `parsers/{v1,v2,v2-py}` (packaging-neutral; names unchanged).
- `dynamo-parsers-v2-py` is marked `publish = false` (test-only).
- The tokenizer test fixtures moved out of the top-level `llm/tests/data` into `tokenizers/tests/data` (removing the lone root `llm/` dir). All frontend crates are now owned here and consumed by Dynamo as published dependencies.

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

## Ownership

All source, tests, fixtures, and conformance utilities in this repository are frontend-crates-owned. After changing parser fixtures or conformance code, verify the renderer:

```bash
conformance/utils/render_table_v2.sh
```

## Version Pins

Frontend crate and parser dependency versions are owned by the workspace `Cargo.toml` files. Peer engine versions used by conformance fixtures are owned by `conformance/utils/src/pyproject.stub.toml`; Dynamo manages its published-crate pins independently.

## Migration-Only Files

These files exist only for the parser v1/v2 migration and conformance workflow.

| File | Purpose |
|---|---|
| `parsers/v2/` | Temporary Rust parser crate for v2 streaming work. |
| `parsers/v2-py/` | Temporary PyO3 binding crate/package for v2 streaming work. |
| `conformance/toolcalling/fixtures-stream-v1/` | Legacy, non-Unified stream fixtures using the v1 corpus convention. |
| `conformance/toolcalling/fixtures-batch-on-stream-v1/` | Legacy, non-Unified batch-on-stream fixture overlays. |
| `conformance/utils/src/_common.sh` | Shared stage builder for conformance scripts. |
| `conformance/utils/check.sh` | Runs local-parser, vLLM, and SGLang checks against staged fixtures; v2 local-parser checks run Dynamo parser v2 code. |
| `conformance/utils/render_table_v2.sh` | Renders `conformance/CONFORMANCE.html` with the v2 conformance generator. |
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
