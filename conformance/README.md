<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# conformance

Parser conformance fixtures, fixture-based Rust tests, and the browser-loaded Unified viewer for frontend-crates. The checkout root `index.html` reads the compact Unified catalog and selected versioned YAML over HTTP. Publication generates gzip-compressed typed JSON inputs and resolved captures in the site output; YAML remains authoritative, and raw-source links retain original YAML bytes. The full catalog remains available to Node model checks and legacy raw archive links.

## Active corpus and capture policy

`TOOLCALLING.*` is deprecated. New cases, capture jobs, and normal reports use `UNIFIED.*`. Retain the archived evidence; deprecated v1 conformance executors and tests are retired; do not migrate or relabel their observations, read them during normal Unified regeneration, or derive Unified GOLDEN expectations from recorded parser output.

Contributors edit parsers and authored inputs/GOLDEN, run latest parser tests, and generate the latest report. Recorded observations and historical backfills are maintained by repository administrators; they are not a contributor prerequisite. Generated publication JSON stays outside Git.

## Latest captures and retained history

The implementation scope and acceptance checks are in [`../docs/CONFORMANCE-LATEST-PLAN.md`](../docs/CONFORMANCE-LATEST-PLAN.md). Add 77 complete versionless latest files beside the existing captures at `fixtures-unified-v2/families/<family>/<implementation>.yaml`, across 12 families, including Muse Spark. Batch and jail variants remain separate implementations. Authored requests and GOLDEN stay in each family's `inputs_and_golden.yaml`.

No backfill is needed. Keep all 689 existing versioned v2 capture files, all 12 authored input/GOLDEN files, and existing legacy archives unchanged at their original paths. Versioned v2 files will be removed in later PRs. Historical moves, archive expansion, and new compression are deferred.

Each latest file contains complete records and sealed request bindings, so it needs no historical parent. `capture_selection` identifies the latest selection separately from the producer provenance in each record. Local parser outputs are re-captured from current code. Peer outputs retain their recorded producer versions; selecting an older observation does not mean it was measured again.

### Generate the latest or history report

Build the Rust publisher and install the viewer's locked npm build dependency. Generate each report into its own output directory:

```bash
cargo build --locked --release -p conformance-publisher
npm ci --prefix conformance/viewer
target/release/conformance-publish --root . --output ../reports/latest --revision "$(git rev-parse HEAD)"
target/release/conformance-publish --root . --output ../reports/history --revision "$(git rev-parse HEAD)" --include-history
```

Default generation reads authored inputs and versionless latest YAML only. `--include-history` adds historical selections from the unchanged versioned YAML and legacy archives at their current paths. `--history-source <directory>` selects an alternate dataset and requires `--include-history`. Latest and history use the same HTML template and JavaScript; their generated catalogs determine which releases and comparisons are available. Historical captures with recorded results remain selectable even when they used the Combined parser path; the current native family list bounds the displayed rows. Muse Spark message start/end events and commentary/final phases remain visible in both assembled output and per-chunk details. Default publication contains exactly one self-contained `index.html`, with no required JSON, JavaScript, CSS, fonts, or other files. Full history adds optional JSON sidecars beside the same HTML. Copy the default `index.html` to `conformance/CONFORMANCE_v2.html` to refresh the local reader-facing report; it can be moved or opened offline without companion files.

Reuse a valid complete-publication cache before source staging. On a miss, restore compatible converted fragments before assembly. Historical YAML conversion runs only for explicit history generation when matching JSON is missing, incompatible, or corrupt. Cache admission checks source and payload hashes. Conversion identities are independent of commit metadata, renderer changes, and physical relocation; final envelopes and presentation may be rebuilt. Generated JSON, HTML, assets, and caches stay outside Git.

### Current parser tests and qualification

Current native tests discover versionless latest files through the shared resolver and materialize only those complete snapshots plus authored inputs. They do not sort or open historical captures and require no network access. Use `extract_fixtures.py --include-history` for retained historical fixtures, or supply `--history-source` for an alternate dataset.

Ordinary CI generates latest only. The explicit `include-history` workflow input enables historical publication and qualification. Qualification covers cold and warm cache equivalence, independent family conversion, corrupt-entry recovery, and Chromium release selection, comparison, reload, and downloads. Latest generation must leave capture YAML unchanged and work when historical access and network sockets are denied.

## Case applicability

A case or rendered column that is applicable to only one parser family must NEVER appear in a generic section. Put it in a named family-specific section, including after variant aggregation. Before classifying a case as family-specific, inspect every registered family and add native inputs and independent expected outputs wherever the behavior applies. Missing captures do not make an applicable family inapplicable; retain them as missing evidence. Preserve published case IDs when changing display placement.

Before adding a conformance case, inspect every registered family. Always evaluate every new or existing single-family case against every registered family when adding, changing, or reviewing it. Determine whether the behavior can be expressed in another family's native grammar and request mode. If it can, author inputs and expected outputs for every applicable family and place the case in a shared section. If it cannot, document the grammar or mode limitation and move the case into a named family-specific test section. Discovering a bug in one family does not justify limiting coverage to that family. Use numeric Unified groups for shared behaviors; preserve published IDs when changing display placement. List every newly added conformance ID, its meaning, applicable families, and intentional exclusions in the PR description.

## Ownership

New conformance cases and captures are added only to Unified for the v2 parser. Do not add new v1 cases or v1 batch/jail captures. The default report displays only Unified; legacy conformance suites and renderers are retired. Existing v1 captures remain available as historical comparisons in Unified.

Parser v1/v2 terminology, migration steps, and fixture ownership are documented in [`../docs/PARSERS-V2-MIGRATION-PLAN.md`](../docs/PARSERS-V2-MIGRATION-PLAN.md). New streaming parser authors should also read [`../parsers/v2/README.md`](../parsers/v2/README.md); it explains the vLLM-shaped Rust parser contract, the v2 fixture schema, and the exact `conformance/toolcalling/*` files to add. This README covers conformance layout, browser resources, and test commands.

## Parser paths and modes (universal convention)

- **Implementation generation:** `dynamo_v1` means the `dynamo-parsers` batch implementation, used for plain batch and jail+batch. `dynamo_v2` means the `dynamo-parsers-v2` implementation crate, which contains both non-Unified incremental parsers and Unified parsers.
- **Conformance convention:** streaming v1 means the legacy, non-Unified stream and batch-on-stream corpora. Unified v2 means one parser owns the ordered reasoning, visible-text, and tool-call stream. A `dynamo_v2-*` capture inside `fixtures-stream-v1` names a v2 implementation measured against the v1 convention.
- **Why the names changed:** the incremental parser was originally called `stream-v2` because it was intended to replace batch+jail. Unified later became a second v2 convention. Renaming the older corpus to streaming v1 removes that collision; it does not rename parser crates or implementation keys.
- **Runtime behavior today:** parser identities are configured on the deployed model. `DYN_ENABLE_EXPERIMENTAL_PARSERS_V2` enables both experimental routes; it does not choose one by itself. Routing checks Unified first: the exact `tool_call_parser=qwen3_coder` plus `reasoning_parser=qwen3` pair uses Qwen3 Unified when the flag is on. If no Unified pair matches, eligible `qwen3_coder` or `deepseek_v4` requests use the older non-Unified incremental parser. All remaining tool-parsing requests use batch+jail. DeepSeek V4.1 and Muse have separate Unified routes that do not depend on this flag. There is no general fallback chain after a selected parser fails.
- **Per-chunk cells show WHEN output reaches the consumer.** Streaming parsers emit whenever something is parseable, so their cells carry deltas at real chunk positions. The v1 jail bursts at end-of-call — its captures record emission order, not per-chunk timing — so its per-chunk cells stay `—` with a "(bursts at end of call; per-chunk timing not recorded)" header note, and its output appears only in the `assembled` row.
- The public report shows only Unified. Its three selected captures are compared against authored GOLDEN. Legacy stream, batch-on-stream, batch, and reasoning corpora remain available to capture tools and regression tests.

## Layout

```
conformance/
├── fixtures-manifest.json                         # pins the active fixture snapshot (sha256 per shard)
├── fixtures-unified-v2/families/<family>/<implementation>.yaml # complete latest observations
├── fixtures-unified-v2/families/                    # authored inputs and GOLDEN
├── fixtures-unified-v2/families/<family>/<implementation>-<version>.yaml # unchanged retained history
├── tests/*.rs                                     # Rust fixture tests (fixtures extracted from the store on first run)
└── utils/                                         # catalog, check, and capture helpers
```

`extract_fixtures.py` materializes latest YAML and authored inputs into `~/.cache/dynamo/conformance-fixtures/` for existing consumers. `--include-history` resolves the retained versioned YAML and legacy archives explicitly. Materialized compatibility layout:

```
toolcalling/fixtures-batch-v1/<family>/           # v1 tool-calling batch cases
toolcalling/fixtures-stream-v1/<family>/          # legacy, non-Unified stream cases
toolcalling/fixtures-batch-on-stream-v1/<family>/ # legacy, non-Unified batch-on-stream cases
reasoning/fixtures-v1/inputs/<family>/            # v1 reasoning cases
```

**Latest captures are complete current snapshots.** Each family/implementation/mode YAML contains its current records independently of historical parents. Latest records use `member` to download their supplied complete snapshot. Their record metadata retains the original producer, and the selection header identifies the selected release. Historical records retain their original sparse source members; a newer selection label does not imply a new execution. Authored inputs and GOLDEN are contributor files. Administrators maintain recorded observations and historical request bindings through the capture workflow; no historical backfill is required in this PR. `dynamo_v1` and `dynamo_v2` remain separate implementations.

Latest execution metadata and receipt bodies are inline in complete snapshots. Historical conversion follows the unchanged sparse records and caches resolved JSON fragments. Original YAML and evidence are never rewritten by generation or tests. Source catalog and published transport hashes continue to validate the bytes used by a publication.

## End-to-end test cases (a separate surface, kept elsewhere)

Everything under `conformance/` is HERMETIC: a fixed byte string goes into a parser and an exact event list comes out, with no model and no worker. There is a second, complementary surface — the **end-to-end test cases** — which sends real requests to a real Dynamo worker and checks the returned response. It has its own suite/category taxonomy (`reasoning` and `tool_calling` suites; `core` / `complex` / `history` / `tool_boundary` / `arguments_schema` / `parallel_lifecycle` categories), each case run in `stream` and `non-stream` mode, under two thinking-budget variants, at worker `--stream-interval` 20 and 1.

**That suite and its artifacts do not live in this repo.** Its cases arrive as a self-contained HTML report (e.g. `qwen36_pr163_test_cases.html`) whose `const REPORT` embeds every request, expectation and response; the per-case artifacts it names are files of the form `end-to-end case-<num>-<case>-<variant>.json` on whichever machine ran the harness.

The two surfaces answer different questions and neither replaces the other, so where a hermetic case has an e2e counterpart it carries an `End-to-end:` tag naming it. **Those tags and the full artifact index are maintained in ONE place — `utils/lib/parsers/UNIFIED_CASES.md` ("End-to-end test cases" and "Artifact index").** Do not copy the mapping into another doc; a second copy drifts, and `utils/tests/test_unified_taxonomy_covers_corpus.py` only pins the two that already exist.

Per-case tagging currently covers the UNIFIED surface only. The reasoning and tool-calling case docs below have no e2e tags yet — that mapping has not been worked out, and an untagged case means "not yet mapped", not "no e2e coverage".

## Browser Viewer

The independent YAML reader is vendored as one ESM bundle from the exact `yaml@2.8.1` dependency pinned by the viewer lockfile. Run `npm ci --prefix conformance/viewer`, then `npm run vendor:yaml --prefix conformance/viewer` to reproduce it; retain its adjacent license. This reader supports development and independent qualification and is excluded from production HTML.


Capture changes update the selectors and URL immediately. The displayed table, legend, and open popup keep showing the previous captures until the replacement is ready. Loading and failure messages leave those results usable; Retry repeats the requested selection. A successful replacement preserves filters, Vertical/Detailed state, scrolling, and an open popup when its case still exists.

The report contains one Unified table in HTML and JSON; the Unified tab button is hidden, and links to removed tabs open this table. Three comparison slots remain above the original report. Clicking a slot first shows Dynamo, SGLang, and vLLM buttons (plus None for slots 2 and 3), then the selected engine's normal version selector. The version selector contains captures only; None belongs to the engine picker. Slot 1 always requires an engine and capture, defaulting to the latest Dynamo v2 Rust Unified parser. Slots 2 and 3 default to None and may select another distinct capture. Different versions or modes of the same engine can occupy separate slots. GOLDEN remains the fixed oracle. The original filters, transpose control, detailed table, and case popups stay in place.

Clicking a D, S, or V engine badge opens its engine and capture picker. Changes apply immediately; there is no Done button. The picker stays beside its trigger, closes on Escape or outside click, and restores focus to the trigger. The selected captures feed the original Unified report table and its case-output popups. Rounded result boxes contain one equal-width segment per active capture in slot order; two active captures make two halves. None contributes no segment. Segment labels and popup outputs distinguish matches, disagreements, markup leaks, parser errors, unsupported requests or families, missing captures, and blocked producer sources. Scenario exclusions use a short family-specific not-applicable message; uncaptured outputs say “Output not yet captured”. The legend uses the same status labels and colors as the rounded cells in Overview, Detailed, and transposed views.

Cell clicks open a centered output popup; case-heading clicks pin the original input/output table across families. Small popups stay content-sized; larger popups grow to 97% of the available viewport width and height and scroll internally when needed. The URL records the clicked model, family, and case, so opening a shared link restores the same popup and comparison selections. Closing the popup removes that selection from the URL. Hover and keyboard focus do not open these popups. Enter or Space activates a focused case cell or heading. Click outside a popup to close it. The original lazy renderer retains folded variants, schemas, selected outputs, fixture links, and streaming chunks. Text-input popups retain their input chart and add a GOLDEN disclosure when an oracle is available. The detailed reference table remains behind its disclosure and keeps its horizontal orientation.

Category selections use stable group keys per tab. Shared URLs preserve ordered slots, reference, categories, layout, focus, search, peer filtering, and Vertical. Selecting all categories omits the category parameter; selecting none preserves an explicit empty value. Vertical writes `vertical=1`; `transpose=1` and `transpose=true` remain supported aliases. Disabling Vertical removes both orientation overrides. Existing Unified version/reference links remain supported. Legacy selections restore their reference followed by their first two other captures; oversized slot selections retain their first three positions. Both display a three-slot-limit notice. Reset restores defaults on the current tab.

The detailed reference table retains compact Unified `<group>-<number>` headings. Numbers start at 1 within each group in report order: behavior sections first, then alphabetical descriptions. DeepSeek V4 and V4.1 use `ds4` and `ds41` as compact prefixes. Popups retain full descriptive labels, compact references, canonical IDs, and grouped child-variant IDs. Display labels do not change stored case IDs or historical references.

Case-group headings align left with an 8px inset in both horizontal and transposed tables. Header and result popups show the full descriptive label followed by the compact reference in parentheses, with its number small and gray. Canonical IDs remain available on the popup heading hover; grouped child variants retain their own IDs.

Generate the browser matrix with the [latest report command](#generate-the-latest-or-history-report), then serve that output directory and open `/index.html?tab=tab-unified`. Refreshing a source catalog alone does not regenerate the published report.

Both publication modes emit byte-identical HTML for the same latest inputs and source-link revision. Default output retains only `index.html`; the publisher validates the embedded dependency graph, then discards duplicate transport files and the second HTML entry point before installation. History output retains its catalog, transport files, and canonical entry point. Cache restamping updates the standalone HTML directly without creating a metadata file. The HTML embeds CSS, JavaScript, worker, latest runtime catalog, current comparison data, and original latest YAML downloads. Opening the file locally needs no external reads. Over HTTP, the same HTML probes an optional publication sidecar after the initial matrix; a matching history publication exposes older selections and original historical downloads through on-demand JSON. Missing or mismatched sidecars leave latest results usable. Catalog upgrades cancel old selection generations and rebuild the current selection without replacing a newer user choice. Embedded and fetched data share hash, size, recipe, and worker validation; corrupt embedded data fails without a network fallback. Invocation paths, generation timestamps, original generation revision, and generation command remain in the sidecar so cache restamping preserves common HTML. `startup=external`, `startup=bootstrap` and `startup=inline` remain aliases for the same loader. `--data-only` is reserved for unsplit development fixtures and is rejected for the split checkout.

### Historical batch pipelines on Unified inputs

Batch captures run reasoning extraction followed by batch tool extraction against the unchanged family-scoped Unified request. Dynamo v1 direct batch, Dynamo v1 batch + jail-buffer, vLLM Python batch, and SGLang Python batch use separate `dynamo_v1`, `dynamo_v1_jail`, `vllm_python_batch`, and `sglang_python_batch` histories. Where the published crate exposes `JailedStream`, the jail-buffer producer feeds each original chunk through the historical Dynamo v1 reasoning parser and that jail, then records the final fields; it does not rewrite or replace direct-batch history. Each complete snapshot retains source, adapter, and job identities; each observation is bound to the original input, initialization, tools, finish reason, and chunk schedule. Unchanged results are copied with the original source member and provenance. The GPT-OSS split-only family adds 159 active input/GOLDEN pairs for each Dynamo v1 batch identity; these are request-bound historical pipelines, not native Unified parser captures. Historical Unified observations and legacy archives remain unchanged.

The frozen inventory in [`all-capture-inventory.yaml`](all-capture-inventory.yaml) accounts for these releases and modes. Each mode keeps its own execution receipts and history.

| Engine and parser mode | Frozen version coverage |
|---|---|
| Dynamo v1 direct batch | All 81 published stable releases. |
| Dynamo v1 batch + jail-buffer | All 81 published stable releases; absent jail APIs are explicit unsupported capabilities. |
| Dynamo v2 stream | All 67 published stable releases and the retained source labels `0.7.9` and `0.7.16`. |
| vLLM Python batch and stream | All 99 published stable releases in each mode. |
| SGLang Python batch and stream | All 150 published stable releases in each mode. |
| vLLM Rust stream | All 99 release identities accounted for; absent Rust APIs are unsupported capabilities, and `0.0.1` has a source blocker. |

The accounting covers 828 version/mode selections and 5,212 active family/case pairs across 27 families: 4,315,536 combinations in total. Successful execution records can still disagree with authored GOLDEN.

| Outcome | Combinations | Meaning |
|---|---:|---|
| Executed | 786,345 | The pipeline returned a recorded output. |
| Parser error | 550 | A parser invocation returned an error or panicked. |
| Unsupported capability | 3,523,429 | The family, request mode, or parser API is unavailable. These rows do not count as executions. |
| Source/build blocker | 5,212 | vLLM Rust `0.0.1` has no matching upstream release tag in the frozen refs; execution evidence is blocked. |
| Missing evidence | 0 | No combination lacks a recorded outcome or explicit blocker. |

Both direct batch and batch + jail-buffer observations project final reasoning text, visible content, and ordered tool calls into final events. Those events are compared to the original ordered GOLDEN. Lost reasoning/content/tool interleaving is a mismatch even when every final field retains its text. The matrix, case popups, and Detailed table use the same selected captures against GOLDEN. Popups show the actual final fields and label batch rows “Order/chunk not applicable for batch”; batch observations contain no synthetic chunk emissions. JSON includes per-capture accounting and mismatch counts. Considered counts active family/case pairs checked against a version; attempted counts pipeline invocations, including parser exceptions. Captured counts successful final observations; unsupported modes/families, source blockers, and missing captures are separate.

The batch adapters execute native/default initialization with stop or length termination. Dynamo and SGLang also apply native Reasoning initialization through their public historical APIs. vLLM requests requiring an explicit initial reasoning state remain unsupported by this adapter. Verified recaptures retain the original observations and bind supplemental results to their producer receipts. Missing registrations and unsupported request modes remain explicit unavailable observations; parser exceptions retain their own observation code. Published Dynamo v1 `3.0.0` is included in both batch modes. Its 5,212 jail-buffer results report `unsupported_capture_mode` because that crate has no public `JailedStream`; no jail parser execution is claimed. The nonpublished Dynamo v1 `9.2.13` direct-batch and jail-buffer observations remain retained historical source blockers outside the frozen denominator.

`capture_unified_job.build_job(..., mode="batch")` preserves the complete request and GOLDEN. Dynamo v1 direct batch uses `legacy_batch_probe.rs`; batch + jail-buffer uses `legacy_jail_probe.rs` against the version-matched historical crate and records the producer identity in each checkpoint. Its CLI accepts `--mode batch --family <family> --output <job.json>` to write the same bound job. `capture_unified_batch.py --engine vllm|sglang --jobs <family-job-directory> --output <owned-output-directory>` runs inside the matching immutable engine environment. The Rust `legacy_batch_probe.rs` is compiled against a checksum-verified historical `dynamo-parsers` crate and receives the shared registry's reasoning/tool parser names. `import_peer_unified.py` validates the embedded producer receipt and imports all exact-job results through the existing transactional history owner. Package with `package_fixtures.py --unified-only`, then refresh the viewer catalog with `viewer_catalog.py --unified-only` and inspect the maintained browser page.

## Running the tests

Use the pinned Rust toolchain and the Unified targets:

```bash
cargo test --locked -p dynamo-parsers-v2 --lib
conformance/utils/check.sh dynamo
cargo test --locked -p dynamo-conformance-fixtures-v2
node --test conformance/viewer/*.test.js
```

`unified_parity`, `unified_render`, and `unified_schema_roundtrip` exercise Unified v2. Beam, DSML, and guided-streaming tests retain their v2 contracts. The old batch, stream, batch-on-stream, Reasoning, interleave, and split-sweep targets are retired. Existing archives remain available as evidence; their generators and capture writers are retired. Dynamo v1 results measured against Unified requests remain comparison data.

## Fixture Workflows

The commands below cover Unified v2. Preserve archived evidence and historical readers; do not regenerate the retired corpus.

**Title fixture/table-only PRs `chore(conformance):`, never `feat:`.** The repo is squash-merge only and GitHub is set to `squash_merge_commit_title: PR_TITLE` with a BLANK body, so the PR TITLE becomes the entire commit message on `main` — it is the only Conventional Commit that release-plz ever reads. The branch's own commit types are discarded, so retitling the PR is both necessary and sufficient. `feat:` proposes a MINOR version bump on every crate whose packaged contents changed ([`../RELEASING.md`](../RELEASING.md#bump-policy) has the full bump table); re-capturing fixtures or re-rendering the table is not a library feature and must not move a published version. Use `feat:` only when parser CODE under `parsers/` changed behaviour. Fixture-only work is outside every crate's packaged contents, so it proposes no bump.

### Edit parsers and authored inputs/GOLDEN

Reproduce the behavior in a parser test, edit the shared parser owner, and run the latest tests below. Author expected GOLDEN independently of observed parser output. An output mismatch is a parser regression or a deliberate authored expectation change; historical recapture is not required to submit the parser fix.

```bash
cargo test --locked -p dynamo-parsers-v2
cargo test --locked -p dynamo-conformance-fixtures-v2
```

Use versionless YAML in `fixtures-unified-v2/families/<family>/` for recorded latest expectations. Contributors do not maintain historical versions. Request an administrator capture update when the latest recorded observation needs to advance. Generate and inspect the latest report after the final edit using [Generate the latest or history report](#generate-the-latest-or-history-report).

### Structural-tag request checks

Unified cases may carry `init.structural_tag` with a response `schema` and complete `rejected` streams. The shared Rust request handler builds the real family structural tag and invokes `utils/src/check_structural_tag.py` before feeding the accepted stream into the parser. The CPU matcher checks the authored chunk schedule, every character split, and each rejection probe. A schema-only fixture with parser checks alone does not cover generation constraints. Parser-only peer adapters report unsupported request mode.

Install the pinned CPU verifier with `python3 -m pip install --no-deps -r conformance/utils/structural-tag-requirements.in`. Set `CONFORMANCE_GRAMMAR_PYTHON` to that interpreter when it differs from `python3`. The verifier loads XGrammar's native TVM-FFI binding and does not import torch or transformers. Rust CI installs this environment before running conformance. `UNIFIED.36-structured_response_after_reasoning` covers the available GPT-OSS, Qwen3, GLM and DeepSeek V4 builders.

### 4. What CI actually checks (the regression gate)

The `rust` CI job materializes the local current YAML and authored inputs, then runs current parser code against those expectations. It does not retrieve history or build the report publisher first:

- `unified_parity`: ordered events, request initialization, stream/batch parity, reuse, and family applicability.
- `unified_render`: native rendering and current capture agreement.
- `unified_schema_roundtrip`: authored schema and request round trips.

Parser mismatches require a parser fix or a deliberate authored expectation change. Contributors request an administrator update when recorded latest observations need to advance. Rust and fixture checks start alongside latest publication. `pages-build` uploads the candidate site before qualification, and independent qualification jobs compare latest YAML with resolved JSON and exercise the current matrix. Historical receipt bindings and older-release comparisons run only when the workflow input `include-history` is enabled. Authored-only checks load family inputs without expanding history; publication and tests preserve recorded YAML.

The `conformance-core` job checks the Unified catalog and publication-dependent utility tests. Smoke tests run in two independent browser groups; interactive viewer tests and popup regressions run in three groups. First-visible-matrix and strict native publication/browser acceptance run separately. Each browser group owns its fixtures and Chrome processes. Every conformance job has a 20-minute limit. The stable `conformance-table` status requires Rust, publication, fixture checks, all qualification shards, and all browser groups before deployment.

Qualification validates publication membership before partitioning comparison work and prints its selected resource paths. Ordinary CI qualifies latest data. For explicit history runs, `ci_coverage.py` selects `latest-two` only when every changed path belongs to its presentation allowlist; evidence, reader, schema, provenance, conversion, deleted, and unclassified changes use exhaustive historical coverage. Historical receipt and introduced-case membership checks remain exhaustive whenever history is requested. Test groups partition sorted collected node IDs through the shared collection owner.

Native pack compression uses the publication worker pool with deterministic pack identities, dependency boundaries, and gzip level 9. The publication cache hashes publication inputs: permanent source bytes, the reader and publisher, authored inputs, publication configuration and policy, release tags, and toolchain identity. Unrelated source, test, and documentation changes can reuse a matching publication. Cache reuse updates the current source-link revision in JSON and embedded HTML; the original generation revision, command, time, catalog identities, and payload bytes remain unchanged. Cached archive and binary hashes are checked before reuse; missing or corrupt cache entries fall back to cold generation. Cache hits still run qualification and browser acceptance. Permanent capture YAML and authored YAML remain unchanged. Running `check.sh ci` locally without `CONFORMANCE_CI_SHARD` runs the full suite; the workflow owns CI grouping.

### 5. Add or change an authored test case

Author family-native inputs and independent GOLDEN in `utils/src/gen_unified_golden.py`, reusing `schema_cases.native_call_wire` and stable IDs from `unified_taxonomy.py`. Preserve existing identities, describe grammar exclusions in `utils/lib/parsers/UNIFIED_CASES.md`, and update the coverage taxonomy. Generate authored YAML, run latest parser tests, then generate and inspect the latest report. Historical backfills are administrator work and are not required for contributor changes. A changed authored request does not rewrite the frozen request bound to an older observation.

### Classify a Unified parser difference

Authored GOLDEN remains the correctness oracle. Capture differences retain exact observations and provenance; they do not authorize changing GOLDEN to match the parser.

A separately triaged, pre-existing v2 defect may be retained while a scoped change is reviewed, following the exact-output policy in `unified-known-divergences.yaml`. Use a distinct `baseline_defect` note starting with `FIXME (v2 defect):` and `baseline_defect_actual` containing the exact observed result as JSON. The gate checks both that observation and the pinned capture, reports the defect separately, and rejects changed or stale entries. The report keeps the differing outputs and FIXME diagnostic visible without the intended-difference badge. This records a deferred parser fix; it does not change a golden, bless the defective result, or permit a new regression.

### Unified applicability and preservation

`utils/src/unified_taxonomy.py` owns published case IDs and display placement. The generator owns family-native scenarios; `test_unified_taxonomy_covers_corpus.py` checks ownership, applicability, emission, and reload. Unknown scenario names fail explicitly. Family markers stay in `parser_families.yaml` and feed the shared comparison model and colorizer.

Native tests consume immutable latest snapshots. Historical readers preserve source identities independently of publication. Full historical packaging pins archive descriptors; Unified-only packaging preserves those pins. No historical rewrite is required for a contributor parser change.

## Safe pruning of Unified history

**Deleting a capture that appears in another snapshot's `member` breaks source identity. Keep the original capture available for every retained member reference.**

Prune only from an immutable copy of the original store and its manifest. The original remains the comparison oracle throughout the operation. Choose retained versions separately for each implementation and mode: keep the newest Dynamo v1 release in each major for both direct-batch and jail-buffer histories, and keep the newest Dynamo v2 release in each major.minor line. For example, if v2 has 0.5.1, 0.5.6, and 0.5.99, keep 0.5.99 for the 0.5 minor line. Complete snapshots keep their resolved results without parent loads, but each retained record's `member` must still name an available original capture.

1. Preserve an immutable original store and manifest.
2. Choose retained versions separately for each implementation.
3. Enumerate every retained snapshot's `records.*.member` value and keep each named source capture.
4. Preserve observations, source provenance, request bindings, and inline receipt bodies.
5. Remove only captures outside the retention policy and outside every retained `member` reference.
6. Reload the pruned store and compare every retained record, provenance field, and member reference with the original.
7. Regenerate packaging and catalogs; verify published data and raw-source links.

Do not rewrite complete snapshots into parent deltas as part of pruning.

Latest publication reuses validated family JSON fragments before assembling final envelopes and transport packs. The local cache is `build/conformance-cache`; `--cache-dir <directory>` selects another location. Source and output hashes detect missing or corrupt entries, which are rebuilt. Viewer-only changes rebuild presentation and reuse converted data. CI restores family caches within each branch namespace and ordinary CI publishes latest only; the explicit `include-history` workflow input requests history.
