<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
# Schema and coercion correctness suite

110 named Rust test groups exercise authored native inputs against v1 batch/finalize,
v1 `JailedStream`, v2 `ToolParser`, and native `UnifiedParser`. The corpus covers
Kimi K2/K3, DeepSeek V3/V3.1/V3.2/V4/V4.1, GLM, MiniMax M2/M3, Qwen2.5/Qwen3-Coder,
Muse, Gemma4, and GPT-OSS Harmony. Unimplemented surfaces are recorded as unavailable,
never counted as passing. Alias smoke uses canonical inputs and the real alias registry.

`cases.py` owns independent inputs and expected typed values; `catalogue.json` maps
them to the original A–H test-plan groups. The Rust test owns actual parser calls and
delta assertions. `support.py` exercises real structural-tag builders through pinned
XGrammar and the existing Python conformance producer/capture validator. No parser,
compiler, or capture validator is mocked. H01–H03 substitute only the author's input
cases to probe the real YAML emitter; later pipeline stages run only if that emitter
preserves the required data.

## Run

Use the repository's Rust toolchain and Python with PyYAML. Grammar groups also need
**XGrammar 0.2.7**. No model server, GPU, CUDA operation, or inference is used.
Use an existing compatible environment or provision one with `uv`:

```bash
uv run --with 'xgrammar==0.2.7' --with pyyaml python -c 'import sys; print(sys.executable)'
export SCHEMA_SUITE_XGRAMMAR_PYTHON=/absolute/path/printed/above
export SCHEMA_SUITE_PYTHON=python3
export SCHEMA_SUITE_RESULTS=/absolute/path/results.jsonl
: > "$SCHEMA_SUITE_RESULTS"
cargo test --locked -p dynamo-conformance-fixtures-v2 \
  --test schema_coercion_contract -- --test-threads=4
```

A failing contract fails its named Rust test, after recording every observation in
that group. There are no ignored tests, expected-failure masks, or output-derived
expectations. For a focused investigation, append a group name such as `b07` before
`--`, or set `SCHEMA_SUITE_FAMILY=glm47` (parser groups only).

For a complete timestamped baseline and standalone HTML report, overlay these test
files on a checkout whose HEAD equals the latest main, then run:

```bash
python3 conformance/schema_suite/report.py \
  --xgrammar-python "$SCHEMA_SUITE_XGRAMMAR_PYTHON" \
  --output /absolute/path/baseline
```

The driver fetches `origin/main`, requires HEAD to match, verifies production files
are unchanged, records commit/check/run UTC timestamps, tool versions, lockfile and
suite source hashes, and preserves the exact suite sources. `report.html` embeds all
observations and works offline. Sibling JSON/log files supply machine-readable evidence.
The report driver succeeds after producing a report even if cargo fails; the report
records cargo's exit code and all red groups. Re-render existing evidence without
running tests using `--render-only --output /absolute/path/baseline`.

## Assertions and limits

- Rows are case/surface combinations, not distinct bugs. `checks` counts partitions
  (or grammar candidates); the report does not count unavailable rows as passes.
- Whole text, all valid UTF-8 two-part splits, character chunks, deterministic multipart
  chunks, and empty chunks cover streaming. Exhaustive two-part splits target boundary
  witnesses; they are not multiplied across every schema case.
- Raw v2/Unified deltas are checked before assembly: names, indices, completion,
  valid completed JSON, and no updates after completion. v1 jail exposes the OpenAI
  projection, which lacks a per-tool completion bit; it is tested at that public boundary.
- JSON comparisons distinguish strings, numbers, booleans, null, arrays, and objects.
  Exact numeric lexemes and key order have separate assertions. Integral floats normalize
  only within the exact integer range; huge-number tests never rely on float equality.
- Grammar tests use real complete-language acceptance, not snapshot matching or
  prefix acceptance. G14 additionally parses accepted Kimi payloads on all four surfaces.
- `regression` labels describe intended invariants, including fixes proposed by open
  PRs; they do not establish the commit that introduced a failure. `target` is an
  explicitly stronger desired contract; `compatibility` pins a family-specific policy.
  Neither coercion nor parser acceptance claims general JSON Schema validation.
- Work-budget probes use finite chains and cyclic schemas. The runner has a 600-second
  subprocess timeout; internal traversal-counter tests remain in the existing parser
  unit suites. This finite suite is not a proof of termination for all schema graphs.
- Source-order and exact huge-number targets can be stricter than legacy behavior.
  Red rows retain the observed output rather than silently weakening the expectation.
- Canonical historical captures are not regenerated. Render their separate report with
  `conformance/utils/render_table_v2.sh`; it is a different artifact from this live baseline.

## PR provenance

The suite implements the schema/coercion audit across these frontend-crates PRs:

| Groups | PRs and behavior |
|---|---|
| A, B | [215](https://github.com/ai-dynamo/frontend-crates/pull/215), [248](https://github.com/ai-dynamo/frontend-crates/pull/248), [251](https://github.com/ai-dynamo/frontend-crates/pull/251), [268](https://github.com/ai-dynamo/frontend-crates/pull/268), [269](https://github.com/ai-dynamo/frontend-crates/pull/269), [280](https://github.com/ai-dynamo/frontend-crates/pull/280): strings, nullable values, unions, intersections, literal constraints |
| C | [271](https://github.com/ai-dynamo/frontend-crates/pull/271), [273](https://github.com/ai-dynamo/frontend-crates/pull/273): references, scope, cycles |
| D | [220](https://github.com/ai-dynamo/frontend-crates/pull/220), [223](https://github.com/ai-dynamo/frontend-crates/pull/223), [270](https://github.com/ai-dynamo/frontend-crates/pull/270): order, duplicate replacement, nested MiniMax object selection |
| E, F | [201](https://github.com/ai-dynamo/frontend-crates/pull/201), [247](https://github.com/ai-dynamo/frontend-crates/pull/247), [249](https://github.com/ai-dynamo/frontend-crates/pull/249), [250](https://github.com/ai-dynamo/frontend-crates/pull/250), [253](https://github.com/ai-dynamo/frontend-crates/pull/253), [255](https://github.com/ai-dynamo/frontend-crates/pull/255): streaming, native types/entities, dialect routing |
| G | [196](https://github.com/ai-dynamo/frontend-crates/pull/196), [274](https://github.com/ai-dynamo/frontend-crates/pull/274), [277](https://github.com/ai-dynamo/frontend-crates/pull/277): grammar enforcement, nullable alternatives, refs; generic omitted-strict policy is pinned to main, not assumed to include #196 |
| H | [267](https://github.com/ai-dynamo/frontend-crates/pull/267), [275](https://github.com/ai-dynamo/frontend-crates/pull/275): per-case schema propagation and capture integrity |

PR status is not inferred from test results. The report's immutable main SHA is the
implementation under test; the test overlay is identified by its own SHA-256 digest.

## Compare PR revisions

`compare_prs.py` runs only this new suite on the exact merge base and head of each
PR in a saved `manifest.json`. Shared merge bases run once per SHA. Every revision
uses the six frozen test files from the manifest's `suite_commit`, independent of
later report-renderer edits. It preserves logs, UTC timestamps, dependency-lock
hashes, source hashes, and every case/surface result under `runs/<full-sha>/`.

```bash
python3 conformance/schema_suite/compare_prs.py run \
  --output /absolute/path/pr-comparison \
  --baseline /absolute/path/baseline \
  --worktree /absolute/path/isolated-audit-worktree \
  --target-dir /absolute/path/cargo-target \
  --xgrammar-python "$SCHEMA_SUITE_XGRAMMAR_PYTHON"
python3 conformance/schema_suite/report.py --render-only \
  --output /absolute/path/baseline \
  --comparisons /absolute/path/pr-comparison/comparisons.json
```

The manifest pins `main_sha`, `suite_commit`, discovery time, and a `prs` array
containing each PR's number, URL, author, title, `merge_base`, `headRefOid`, and
changed files. The referenced commits must already be fetched locally. Resuming
reuses only completed runs with matching suite/result hashes. `compare` in place
of `run` recomputes deltas from saved evidence without executing tests.

Fixed means a failing case/surface becomes passing; regressed is the reverse.
Still-failing cases with fewer/more failed partitions are partial improvements or
worsenings. Changed failure witnesses are retained separately; generated call IDs
and XGrammar diagnostic clock timestamps are excluded from change detection. Group-level Rust
test changes are reported independently: fixing one case may leave a group red.
Unchanged harness errors on older revisions are identified as pre-existing.
An incomplete build/run is explicit and is not presented as a comparable clean
result. This compares each PR with its own base, not a cumulative PR stack or a
simulated merge onto current main; overlapping improvements cannot be summed as
unique fixes.

## Combined integration audit (2026-09-24)

Branch `rmccormick/09-24-schema-combined` merges the exact audited heads of
#251, #263, #268, #269, #270, #271, #273, #274, #275, #276, #277, and #280,
in that order, onto main `17d5f76bafc0b06b7236afb282a00da3ec2b6aa6`.
The tested production merge is `ac378f7d2b0d97b8fdc19e009550b201d117f96f`;
subsequent commits add the frozen suite and report tooling only.

Both revisions ran the same frozen suite, with all 5,944 observations recorded:

| Measurement | Fresh main | Combined |
|---|---:|---:|
| Named Rust tests passed / failed | 49 / 61 | 67 / 43 |
| Case/surface rows passed / failed | 4,108 / 541 | 4,277 / 372 |
| Unavailable / not applicable | 1,230 / 65 | 1,230 / 65 |
| Test runtime, excluding compilation | 44.33 s | 44.72 s |

There are **179 fixed and 10 regressed rows** (+169 passing). The 18 newly
passing named tests are B02, B03, B04, B10, B20, C01, C04, C11, D01, D02, D12,
G01, G03, G08, G09, H01, H02, and H03; none becomes newly failing.
Combined run: 2026-09-24T20:09:13Z–20:10:03Z (50.406 s including compilation).

The remaining regressions are all GLM: B07 enum unions, B08 const unions, and
B09 unconstrained alternatives on v1 batch/jail (six rows), plus B17 enum
exclusion of null on all four surfaces (four rows). B09 is an explicit
compatibility policy; the other three are regression contracts. B12/B13's
four v1 regressions from #268 disappear when combined. C03 nullable references
on v1 batch/jail and H03 schema emission/loading/Unified parsing pass only in
the combination. Every individually observed fix passes on the combined branch.
No regression appears solely in the combination.

Conflict resolutions preserve both sides' tests. GLM nullable preference uses
the reference-aware matcher; scalar-union selection follows direct/referenced
type selection. Kimi resolves references before building nullable argument
alternatives and retains the typed fallback. MiniMax conflicts only joined
adjacent added tests. No expectations were relaxed and no new parser fixes
were added beyond composing these PR changes.

The evidence bundle has sibling `baseline/`, `pr-comparison/`, and `combined/`
directories. `combined/manifest.json` pins every integrated head and the
resolution notes; `comparisons.json` contains every changed observation;
`interactions.json` lists differences from the individual PR audit. Render both
comparison sections while preserving the original main baseline:

```bash
python3 conformance/schema_suite/report.py --render-only \
  --output /absolute/path/baseline \
  --comparisons /absolute/path/pr-comparison/comparisons.json \
  --combined /absolute/path/combined/comparisons.json
```

To repeat the combined comparison, use `compare_prs.py run` with `--output`
pointing to a copy of the combined manifest directory and an isolated audit
worktree. Its manifest uses `kind: "combined"` and one comparison entry whose
`merge_base` is current main and `headRefOid` is the production merge commit.

The report also ranks the combined branch's remaining failures by estimated fix
effort. `remaining_failures.json` owns the reviewed category definitions and
source locations; the renderer derives counts from the hashed combined run.
Every failing row belongs to exactly one category. The model/category matrix
includes unavailable counts, and category/model expanders retain complete raw
failure evidence. Estimates are not implementation timings or promised fix
counts; no parser changes or additional suite runs were made for this triage.
