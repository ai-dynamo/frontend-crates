# Parsers V2 Migration Plan

This is the single source of truth for the parser v1/v2 split inside frontend-crates and the final migration to a single parser crate.

## Directory Layout (current)

The parser implementations and test binding live under `parsers/`, grouped but still separately packaged. This grouping is packaging-neutral: the crate names, versions, and the `dynamo_parsers_v2` Python module name are unchanged, so Dynamo (which pins `dynamo-parsers` and `dynamo-parsers-v2` from crates.io) is unaffected.

| Directory | Crate | Published? |
|---|---|---|
| `parsers/v1/` | `dynamo-parsers` (legacy batch/jail API) | crates.io |
| `parsers/v2/` | `dynamo-parsers-v2` (incremental + Unified, `0.x`) | crates.io |
| `parsers/structural-tag/` | `dynamo-structural-tag` (shared grammar/schema helpers) | crates.io |
| `parsers/v2-py/` | `dynamo-parsers-v2-py` (PyO3 binding, module `dynamo_parsers_v2`) | **no** — test-only, `publish = false` |

`parsers/v2-py` is the conformance harness's Python extension. It has no external consumer or PyPI publish job.

## Terminology

There are two separate version axes. Do not infer one from the other:

- **Implementation generation:** `dynamo-parsers` / `dynamo_v1` is the stable batch parser, also used behind the streaming jail. `dynamo-parsers-v2` / `dynamo_v2` is the newer crate containing the incremental tool-only parsers and Unified parsers. These implementation names stay unchanged.
- **Conformance convention:** `fixtures-stream-v1`, `fixtures-batch-on-stream-v1`, and `TOOLCALLING.streamv1.*` identify the legacy, non-Unified streaming convention. `fixtures-unified-v2` identifies Unified, where one parser owns the ordered reasoning, visible-text, and tool-call stream.

The non-Unified incremental parser was originally called `stream-v2` because it was intended to replace the original batch+jail implementation. Unified arrived later and also used the v2 name. That left two different “v2” conventions. The corpus rename resolves the collision: **streaming v1 means legacy/non-Unified streaming; v2 means Unified**. A `dynamo_v2-*` capture can therefore appear inside `fixtures-stream-v1`; the directory names the convention, while the capture key names the implementation.

Dynamo selects parsers through `DYN_PARSER_VERSION=auto|1|2` and its pinned registry/adapters; conformance names do not determine runtime behavior. See [Dynamo routing](https://github.com/ai-dynamo/dynamo/blob/main/docs/fern/pages/use-cases/tool-calling-and-reasoning/troubleshooting-tool-calls.md#parser-generation-routing) for defaults and fallback.

## Why The Split Exists

Parser source, parser fixtures, and conformance utilities are **frontend-crates-owned**. frontend-crates publishes `dynamo-parsers` to crates.io and Dynamo consumes it from there.

The v1/v2 split is kept because **v2 is still under active development**: it lives on a `0.x` line where breaking changes are free and expected, and is already adopted by compatible Dynamo routes. `dynamo-parsers` (v1) retains a separate legacy API; callers must test the exact published version they integrate. Merging v2 into v1 now would either force major bumps on the stable crate or slow v2's fast iteration. Keep them separate until the v2 streaming API stabilizes.

## Layout Details

| Area | Owner | Rule |
|---|---|---|
| `parsers/v1/src/` | v1 frontend-crates-owned | Stable batch parser (`dynamo-parsers`). Bug-fix only; do not put v2 streaming work here. |
| `parsers/v1/tests/` | v1 frontend-crates-owned | v1 crate tests. |
| `conformance/fixtures/toolcalling/fixtures-batch-v1/` | frontend-crates legacy v1 | Batch tool-calling fixtures retained for v1 behavior. Do not hand-edit for v2 behavior. |
| `conformance/fixtures/reasoning/fixtures-v1/` | frontend-crates legacy v1 | Reasoning fixtures rendered in the conformance table. |
| `conformance/utils/src/tables/` | frontend-crates-owned | Capture/storage value helpers and source metadata. |
| `conformance/utils/lib/parsers/TOOLCALLING_CASES.md` and `REASONING_CASES.md` | frontend-crates-owned | Case descriptions indexed by the catalog and read by browser/Node models. |
| `parsers/v2/src/tool_calling/*` | v2 frontend-crate-owned | Rust home for streaming tool-calling parsers. Current Harmony implementation is `parsers/v2/src/tool_calling/harmony.rs`. |
| `parsers/v2-py/` | v2 frontend-crate-owned | Test-only PyO3 package exposing the v2 parser to Python as `dynamo_parsers_v2`. Not published. |
| `conformance/fixtures/toolcalling/fixtures-stream-v1/` | legacy streaming convention | Per-chunk fixtures for non-Unified stream parsers, including `dynamo_v2` implementation captures. |
| `conformance/fixtures/toolcalling/fixtures-batch-on-stream-v1.tar.gz` | legacy streaming convention | Complete batch text captured through non-Unified stream parsers for stream-vs-batch comparison. |
| `conformance/viewer/` and `conformance/utils/src/assets/` | frontend-crate-owned | Shared JavaScript data/model engine and maintained matrix renderer. |

## Migration Steps

Already done:

- Parser source, fixtures, and conformance utilities are frontend-crates-owned.
- `dynamo-parsers` (v1) is published to crates.io and consumed by Dynamo from there.
- The parser implementations and test binding are grouped under `parsers/{v1,v2,v2-py}` (packaging-neutral; names unchanged).
- `dynamo-parsers-v2-py` is marked `publish = false` (test-only).
- The tokenizer test fixtures moved out of the top-level `llm/tests/data` into `tokenizers/tests/data` (removing the lone root `llm/` dir). All frontend crates are now owned here and consumed by Dynamo as published dependencies.

Remaining, gated on **v2's streaming API stabilizing** (do not start while v2 is still churning on `0.x`):

1. Stabilize the v2 streaming parser API and cut a `1.0` for `dynamo-parsers-v2`.
2. Merge v2 into `dynamo-parsers`: move the streaming parsers into the crate under a behavior-named module (not a permanent `v2` name — see below), fold the binding into the crate's final Python package, and retire the separate `dynamo-parsers-v2` / `dynamo-parsers-v2-py` crate boundary in one coordinated release.
3. Coordinate the Dynamo cutover: land a Dynamo PR that drops the `dynamo-parsers-v2` dependency and rewrites `use dynamo_parsers_v2::…` to the merged path. This is a breaking change — sequence it (publish merged crate → Dynamo PR → remove old crates), do not do it unilaterally.
4. Retire migration-only fixture/table names after the crate transition. The Rust publisher owns latest/history report generation; preserve published URLs when changing the crate transition.

## Final Target Shape

One `dynamo-parsers` crate, with the streaming parsers exposed under a **behavior-named** module, not a permanent `v2` name (`v2` is a migration label — shipping `dynamo_parsers::v2` just means renaming it again later, another breaking change):

```text
parsers/v2/src/tool_calling/*  ->  parsers/v1/src/tool_calling/*   (e.g. dynamo_parsers::tool_calling::streaming)
```

The `v1`/`v2` directory names are transitional; when the merge happens the surviving crate keeps the `dynamo-parsers` name and the directory split collapses. The Python binding should also lose the `v2` name (there is no v1 Python binding, so this is just a `dynamo_parsers_v2` → final-name module rename).

## Ownership

Contributors edit parsers and authored inputs/GOLDEN, run [current qualification](../conformance/README.md#current-parser-tests-and-qualification), and [generate the latest report](../conformance/README.md#generate-the-latest-or-history-report). Refresh `conformance/CONFORMANCE_v2.html` and inspect it over HTTP. Leave recorded YAML and history unchanged; administrators own those updates under the [capture plan](CONFORMANCE-LATEST-PLAN.md).

## Version Pins

Frontend crate and parser dependency versions are owned by the workspace `Cargo.toml` files. Peer engine versions used by conformance fixtures are owned by `conformance/utils/src/pyproject.stub.toml`; Dynamo manages its published-crate pins independently.

## Migration-Only Files

These files exist only for the parser v1/v2 migration and conformance workflow.

| File | Purpose |
|---|---|
| `parsers/v2/` | Temporary Rust parser crate for v2 streaming work. |
| `parsers/v2-py/` | Temporary PyO3 binding crate/package for v2 streaming work. |
| `conformance/fixtures/toolcalling/fixtures-stream-v1/` | Legacy, non-Unified stream fixtures using the v1 corpus convention. |
| `conformance/fixtures/toolcalling/fixtures-batch-on-stream-v1.tar.gz` | Legacy, non-Unified batch-on-stream fixture overlays. |
| `conformance/utils/check.sh` | Runs Unified v2 conformance, source status, and CI qualification without legacy staging. |
| `conformance/utils/src/viewer_catalog.py` | Indexes active source resources and checks their hashes without rendering a report. |
| `conformance/utils/src/validate.py` | Cross-implementation validation via `docker exec` or pip. |
| `conformance/viewer/presentation.js` | Shared browser/Node matrix-model construction. |
| `conformance/publisher/` | Rust latest/history publisher; default output is a self-contained `index.html`. |
| `conformance/CONFORMANCE_v2.html` | Unstaged canonical local copy of the generated latest report. |
| `conformance/utils/README.md` | Usage docs for validation, catalog refresh, status, and capture helpers. |
| `conformance/utils/.gitignore` | Excludes `.stage*/`, local `CONFORMANCE*.html` outputs, and Python bytecode. |
| `conformance/utils/tests/__init__.py` | Empty package root for `.stage/` imports. |
| `parsers/v1/Cargo.toml` | Inlined for standalone publishing. |
