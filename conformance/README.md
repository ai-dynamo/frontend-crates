<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# conformance

Parser conformance fixtures, fixture-based Rust tests, and the browser-loaded Unified viewer for frontend-crates. The checkout root `index.html` reads the compact Unified catalog and selected versioned YAML over HTTP. Publication generates gzip-compressed typed JSON inputs and resolved captures in the site output; YAML remains authoritative, and raw-source links retain original YAML bytes. The full catalog remains available to Node model checks and legacy raw archive links.

## Active corpus and capture policy

`TOOLCALLING.*` is deprecated. New cases, capture jobs, and normal reports use `UNIFIED.*`. Retain the archived evidence and historical tests; do not migrate or relabel their observations, read them during normal Unified regeneration, or derive Unified GOLDEN expectations from recorded parser output.

Capture every published stable Dynamo v1, Dynamo v2, vLLM, and SGLang release in `all-capture-inventory.yaml`. The frozen inventory contains 81 Dynamo v1 releases, 67 published Dynamo v2 releases plus two retained source labels, 99 vLLM releases, and 150 SGLang releases. It retains exact published artifacts, checksums, yanked flags, and upstream source identities. Keep Dynamo v1 batch and jail-buffer, Dynamo v2 stream, peer Python batch and stream, and vLLM Rust stream separate. Reuse execution only after validating its exact version, artifact, frozen requests, and receipts; equal output does not prove execution. Missing parser APIs remain unsupported capabilities, parser exceptions remain parser errors, and source or build blockers remain missing execution evidence. Generated publication JSON stays outside Git; YAML is authoritative.

## Local current captures and pinned external history

Authored Unified requests and GOLDEN remain in `fixtures-unified-v2/families/<family>/inputs_and_golden.yaml`. Authored legacy inputs remain in `inputs/<mode>/<family>/`. Current observations live in `captures/<surface>/<family>/<implementation>.yaml`; filenames contain no release version. `capture-current.json` pins those files, their actual release identities, and the producer identities referenced by inherited observations. Select the newest recorded complete snapshot independently for every family, implementation, and mode. A family with no capture at an engine's newest release keeps its own newest recorded version.

Unified current YAML is copied byte for byte from the original complete snapshot. Its records retain request bindings, original source members, producer versions, execution proofs, and historical execution metadata. Legacy current YAML contains the complete documents resolved by the existing batch, stream, and reasoning readers plus the identities of their original source records. Missing fields, nulls, parser errors, and unsupported observations remain distinct. A selected release does not establish that an inherited observation ran again.

`capture-history-pin.json` pins an external plain directory by its checksummed inventory and original source revision. That directory preserves Unified and legacy archives, original inputs, receipts, execution evidence, manifests, and interpretation metadata. Archive bytes and member paths remain unchanged. `capture_sources.py` owns resolution and inventory verification; the publisher and fixture tools use that owner. Machine paths are supplied explicitly. Hosted retrieval is pending creation of the capture repository.

### Generate the latest or history report

Build the existing Rust publisher and install the viewer's locked npm build dependency. Generate each report into its own output directory:

```bash
cargo build --locked --release -p conformance-publisher
target/release/conformance-publish --root . --output ../reports/latest --revision "$(git rev-parse HEAD)"
target/release/conformance-publish --root . --output ../reports/history --revision "$(git rev-parse HEAD)" --include-history --history-source /path/to/frontend-parser-captures
```

The default command reads local authored inputs and current YAML only. Its `index.html` embeds the data, styles, application, and worker needed to compare current results and download current YAML over `file://` or HTTP. `--history-source` requires `--include-history`. History publication verifies the external pin before preparing its private source view, then uses the same catalog, Rust result validation, transport, and Unified viewer. Historical payloads remain outside HTML and load on demand. The generated directory contains its own original-download packs and remains usable after the source directory becomes unavailable. Legacy historical tables are not restored.

Publication stages output, verifies its original inputs again, and installs only after validation. Incorrect pins, missing evidence, corrupt archives, and invalid records fail before replacing an existing report. Generation mode, current YAML identities, the applicable history pin, and publisher inputs belong to publication/cache identity. Generated HTML, catalogs, JSON, transport packs, and logs stay outside Git.

### Current parser tests and frozen qualification

Current native tests materialize local authored inputs and selected current YAML through `ensure_unified_fixtures()` or `extract_fixtures.py`. They need neither a publisher build nor a history directory. Explicit frozen-version qualification supplies `--history-source` to `unified_history.py` together with its selected `--capture`; the source is checked against the pin before materialization. Legacy historical maintenance uses the retained format readers against the explicitly verified external dataset.

### Archive capture evidence before advancing current

Publish a complete, validated Unified snapshot with its real producer identity:

```bash
python3 conformance/utils/src/publish_current_capture.py --root . --history-source /path/to/frontend-parser-captures --capture /path/to/complete-snapshot.yaml --family qwen3 --implementation dynamo_v2 --version 0.7.30
```

The writer validates the complete snapshot against local authored requests, archives the evidence first, and transactionally updates the selected current YAML, current index, and history pin. A durable pending record allows the same command to recover an interrupted history or local-pointer install. To recover without submitting another snapshot, run `publish_current_capture.py --root . --history-source /path/to/frontend-parser-captures --recover`. It rejects changed bytes for an existing frozen identity. An older backfill is archived without replacing newer current YAML. `package_fixtures.py` refuses to recreate local historical archives in a split checkout. The archive maintenance commands later in this document apply to historical maintenance checkouts. The existing `package_fixtures.py --unified-only` and `regenerate_unified.sh` capture workflow uses `CONFORMANCE_HISTORY_SOURCE` to archive imported snapshots before advancing current files; capture regeneration requires this explicit source before making changes.

### Pin the release version before capture

When adding or converting a parser, or changing its captured behavior, manually pin the intended release version in the same PR before recording output. The invariant is `capture version == crate version == version published after merge`. An automatic bump after capture would break that equality.

1. Fetch the current release tags and check published versions. Choose the next appropriate unpublished semantic version; a feature branch still carrying an old version does not make its new behavior part of that old release.
2. Set the parser crate's `Cargo.toml` version explicitly. Update its dependent manifests, `Cargo.lock`, and the conformance manifest's crate version together. For Dynamo v2, this includes `parsers/v2/Cargo.toml`, `parsers/v2-py/Cargo.toml`, `conformance/Cargo.toml`, and `conformance/fixtures-manifest.json`.
3. Build and capture from that pinned version. Keep the Unified YAML provenance and current index version aligned with it, then run capture validation, refresh the catalog, and reload the browser viewer. Renaming an older capture without validating it against the pinned build is insufficient.
4. Before merge, recheck that the pinned version is still unpublished. If another PR has released it, choose a new version and repeat the capture validation. After merge, verify the release tag and published crate use the captured version.

The release workflow honors this [manual version override](../RELEASING.md#manual-version-peg-fixture-synced-releases): it checks compatibility and publishes the explicitly pinned version without another automatic bump. For #234, GLM Unified and its pending release are both `0.7.0`; GLM Unified has no `0.6.x` captures. Unchanged families continue to inherit their existing captures.

## Case applicability

Before adding a conformance case, inspect every registered family. Always evaluate every new or existing single-family case against every registered family when adding, changing, or reviewing it. Determine whether the behavior can be expressed in another family's native grammar and request mode. If it can, author inputs and expected outputs for every applicable family and place the case in a shared section. If it cannot, document the grammar or mode limitation and move the case into a named family-specific test section. Discovering a bug in one family does not justify limiting coverage to that family. Use numeric Unified groups for shared behaviors; preserve published IDs when changing display placement. List every newly added conformance ID, its meaning, applicable families, and intentional exclusions in the PR description.

## Ownership

New conformance cases and captures are added only to Unified for the v2 parser. Do not add new v1 cases or v1 batch/jail captures. The default report displays only Unified; legacy suites are retained for regression checks. Existing v1 captures remain available as historical comparisons in Unified.

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
├── fixtures/                                      # LFS-tracked shard tarballs for tool-calling/reasoning fixtures
├── fixtures-unified-v2/                            # reviewable append-only YAML history for Unified captures
├── tests/*.rs                                     # Rust fixture tests (fixtures extracted from the store on first run)
└── utils/                                         # catalog, check, and capture helpers
```

Tool-calling and reasoning archives remain under `conformance/fixtures/`, extracted into `~/.cache/dynamo/conformance-fixtures/`. Unified history uses reviewable YAML under `conformance/fixtures-unified-v2/`. `extract_fixtures.py` materializes both stores for existing consumers. Snapshot layout:

```
toolcalling/fixtures-batch-v1/<family>/           # v1 tool-calling batch cases
toolcalling/fixtures-stream-v1/<family>/          # legacy, non-Unified stream cases
toolcalling/fixtures-batch-on-stream-v1/<family>/ # legacy, non-Unified batch-on-stream cases
reasoning/fixtures-v1/inputs/<family>/            # v1 reasoning cases
```

**Unified captures are complete version snapshots.** Each engine/version archive contains every capture recorded for that engine release, across families and implementation modes. A snapshot stores its checkpoint provenance and document fields plus every resolved case record. Each record names the original source capture in `member`, preserving inherited provenance without claiming the result ran again. Source SHA remains origin metadata and does not create another capture identity. Re-running an unchanged capture does not change its source YAML. Authored inputs and GOLDEN remain immutable after capture: an approved change requires recapturing every affected version. `dynamo_v1` and `dynamo_v2` have separate version histories and never fold together. The pinned manifest and archive identities determine what the chart shows.

Execution metadata and receipt bodies are repeated inline in the resolved records, so reading a version does not follow execution or receipt files. Inline execution proof retains its request bindings; historical unreferenced execution manifests and receipt bodies remain under `historical_evidence` with their original metadata membership. Source catalog and published transport hashes continue to validate the bytes used by a publication.

## End-to-end test cases (a separate surface, kept elsewhere)

Everything under `conformance/` is HERMETIC: a fixed byte string goes into a parser and an exact event list comes out, with no model and no worker. There is a second, complementary surface — the **end-to-end test cases** — which sends real requests to a real Dynamo worker and checks the returned response. It has its own suite/category taxonomy (`reasoning` and `tool_calling` suites; `core` / `complex` / `history` / `tool_boundary` / `arguments_schema` / `parallel_lifecycle` categories), each case run in `stream` and `non-stream` mode, under two thinking-budget variants, at worker `--stream-interval` 20 and 1.

**That suite and its artifacts do not live in this repo.** Its cases arrive as a self-contained HTML report (e.g. `qwen36_pr163_test_cases.html`) whose `const REPORT` embeds every request, expectation and response; the per-case artifacts it names are files of the form `end-to-end case-<num>-<case>-<variant>.json` on whichever machine ran the harness.

The two surfaces answer different questions and neither replaces the other, so where a hermetic case has an e2e counterpart it carries an `End-to-end:` tag naming it. **Those tags and the full artifact index are maintained in ONE place — `utils/lib/parsers/UNIFIED_CASES.md` ("End-to-end test cases" and "Artifact index").** Do not copy the mapping into another doc; a second copy drifts, and `utils/tests/test_unified_taxonomy_covers_corpus.py` only pins the two that already exist.

Per-case tagging currently covers the UNIFIED surface only. The reasoning and tool-calling case docs below have no e2e tags yet — that mapping has not been worked out, and an untagged case means "not yet mapped", not "no e2e coverage".

## Browser Viewer

Capture changes update the selectors and URL immediately. The displayed table, legend, and open popup keep showing the previous captures until the replacement is ready. Loading and failure messages leave those results usable; Retry repeats the requested selection. A successful replacement preserves filters, Vertical/Detailed state, scrolling, and an open popup when its case still exists.

The report contains one Unified table in HTML and JSON; the Unified tab button is hidden, and links to removed tabs open this table. Three comparison slots remain above the original report. Clicking a slot first shows Dynamo, SGLang, and vLLM buttons (plus None for slots 2 and 3), then the selected engine's normal version selector. The version selector contains captures only; None belongs to the engine picker. Slot 1 always requires an engine and capture, defaulting to the latest Dynamo v2 Rust Unified parser. Slots 2 and 3 default to None and may select another distinct capture. Different versions or modes of the same engine can occupy separate slots. GOLDEN remains the fixed oracle. The original filters, transpose control, detailed table, and case popups stay in place.

Clicking a D, S, or V engine badge opens its engine and capture picker. Changes apply immediately; there is no Done button. The picker stays beside its trigger, closes on Escape or outside click, and restores focus to the trigger. The selected captures feed the original Unified report table and its case-output popups. Rounded result boxes contain one equal-width segment per active capture in slot order; two active captures make two halves. None contributes no segment. Segment labels and popup outputs distinguish matches, disagreements, markup leaks, parser errors, unsupported requests or families, missing captures, and blocked producer sources. Scenario exclusions use a short family-specific not-applicable message; uncaptured outputs say “Output not yet captured”. The legend uses the same status labels and colors as the rounded cells in Overview, Detailed, and transposed views.

Cell clicks open a centered output popup; case-heading clicks pin the original input/output table across families. Small popups stay content-sized; larger popups grow to 97% of the available viewport width and height and scroll internally when needed. The URL records the clicked model, family, and case, so opening a shared link restores the same popup and comparison selections. Closing the popup removes that selection from the URL. Hover and keyboard focus do not open these popups. Enter or Space activates a focused case cell or heading. Click outside a popup to close it. The original lazy renderer retains folded variants, schemas, selected outputs, fixture links, and streaming chunks. Text-input popups retain their input chart and add a GOLDEN disclosure when an oracle is available. The detailed reference table remains behind its disclosure and keeps its horizontal orientation.

Category selections use stable group keys per tab. Shared URLs preserve ordered slots, reference, categories, layout, focus, search, peer filtering, and Vertical. Selecting all categories omits the category parameter; selecting none preserves an explicit empty value. Vertical writes `vertical=1`; `transpose=1` and `transpose=true` remain supported aliases. Disabling Vertical removes both orientation overrides. Existing Unified version/reference links remain supported. Legacy selections restore their reference followed by their first two other captures; oversized slot selections retain their first three positions. Both display a three-slot-limit notice. Reset restores defaults on the current tab.

The detailed reference table retains compact Unified `<group>-<number>` headings. Numbers start at 1 within each group in report order: behavior sections first, then alphabetical descriptions. DeepSeek V4 and V4.1 use `ds4` and `ds41` as compact prefixes. Popups retain full descriptive labels, compact references, canonical IDs, and grouped child-variant IDs. Display labels do not change stored case IDs or historical references.

Case-group headings align left with an 8px inset in both horizontal and transposed tables. Header and result popups show the full descriptive label followed by the compact reference in parentheses, with its number small and gray. Canonical IDs remain available on the popup heading hover; grouped child variants retain their own IDs.

| Output | Command | Parser version | Fixture version |
|---|---|---|---|
| Browser matrix | Serve the checkout over HTTP and open `/index.html?tab=tab-unified` | The maintained page shows Dynamo, SGLang, and vLLM Unified captures with GOLDEN comparisons. | Production loads the compressed catalog and shared packs. The checkout development entry point can load `conformance/viewer-unified-catalog.json`; legacy archive raw links fetch the full source catalog on demand. Generate both unversioned source catalogs with `python3 conformance/utils/src/viewer_catalog.py` before first use. After Unified-only changes, refresh with `python3 conformance/utils/src/viewer_catalog.py --unified-only` and check with `python3 conformance/utils/src/viewer_catalog.py --unified-only --check`. CI generates these build inputs once and reuses them across test jobs. |

Production HTML is a single file containing its CSS, JavaScript, parser worker, runtime catalog, shared inputs, and compressed JSON transport for the newest captured release per implementation lineage: Dynamo v1 batch and jail, Dynamo v2, SGLang batch and stream, and vLLM Python batch/stream and Rust. It initially renders only the newest Dynamo v2. On `file://`, it offers only these embedded releases and makes no external requests. On HTTP/HTTPS, it renders the same embedded results immediately, then checks the neighboring `conformance/viewer-publication.json` sidecar against the embedded catalog identity. A matching sidecar enables older selections and original YAML Raw/download views through the existing on-demand transport. An absent, invalid or unreachable sidecar leaves the latest-only standalone view usable. A delayed discovery preserves any selection the user already made. Dynamo pickers retain both recent releases plus the newest release in each historical v1 major or v2 minor line when history is available. Embedded and fetched data share the same hash, size, recipe and worker validation; corrupt embedded data fails without a network fallback. `startup=external`, `startup=bootstrap` and `startup=inline` remain aliases for the same loader. For this split checkout, generate a separate latest or history report directory with `conformance-publish` and open its `index.html`. `--data-only` is reserved for unsplit development fixtures and is rejected for the split checkout.

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

Use the repo's pinned toolchain (Rust 1.96.1 via rustup; a system `cargo` may be too old for the workspace):

```bash
# tool-calling batch parity, all families:
cargo test --locked -p dynamo-conformance-fixtures-v2 --test conformance_toolcalling

# same, but print fixture names and the per-run case count:
cargo test --locked -p dynamo-conformance-fixtures-v2 --test conformance_toolcalling -- --nocapture

# as part of the whole workspace (what CI runs):
cargo test --workspace
```

The test package is named `dynamo-conformance-fixtures-v2` for historical compatibility, but the code ownership still follows the v1/v2 split.

**Lifecycle:** the `dynamo-parsers` batch+jail implementation is interim; the `dynamo-parsers-v2` crate is the successor implementation and currently owns Unified. This implementation migration is independent of the corpus convention names above.

| Test | Code under test | Fixtures | Notes |
|---|---|---|---|
| `conformance_toolcalling` | v1 batch parser in `parsers/src/tool_calling/` | v1 batch fixtures (`toolcalling/fixtures-batch-v1/`) | Each `batch` case's `model_text` is fed through `detect_and_parse_tool_call_with_recovery(text, Some(family), tools)` and compared to `expected.dynamo_v1`. |
| `conformance_toolcalling_batch_via_stream` | Dynamo parser v2 in `parsers_v2/src/tool_calling/*` | v1 batch fixtures (`toolcalling/fixtures-batch-v1/`) plus legacy streaming-v1 overlays (`toolcalling/fixtures-batch-on-stream-v1/`) | Feeds complete batch text into the non-Unified incremental parser and compares assembled calls to the batch-on-stream expectations. |
| `conformance_toolcalling_stream` | Dynamo parser v2 in `parsers_v2/src/tool_calling/*` | Legacy stream fixtures (`toolcalling/fixtures-stream-v1/`) | Checks token-id or text streaming paths per chunk, then checks assembled calls. |

The fixture `family` field is the parser name, the same value Dynamo's `parse_tool_calls_batch` binding takes for v1. Every fixture uses an explicit implementation key: `expected.dynamo_v1`, `expected.dynamo_v2`, `expected.vllm_rust`, `expected.vllm_python`, `expected.sglang_python`. Dynamo v1 and v2 are separate impls with separate version lineages (`dynamo_v1-3.0.0/`, `dynamo_v2-0.1.11/`) — exactly like the vLLM/SGLang runtime variants. Legacy spellings (`dynamo`, `dynamo_rust`, `vllm`, `sglang`) are still accepted on read via the alias table in `utils/src/impls.py`.

Reasoning captures appear in the browser matrix; a Rust fixture harness for reasoning is still a follow-up.

## Historical archive maintenance: legacy fixtures (v1)

Parser fixture sync from Dynamo is retired. Update v1 fixtures through normal frontend-crates PRs and run the catalog and viewer checks documented in [`../docs/PARSERS-V2-MIGRATION-PLAN.md`](../docs/PARSERS-V2-MIGRATION-PLAN.md#ownership).

## Historical streaming fixture format (deprecated corpus)

Use [`../parsers/v2/README.md`](../parsers/v2/README.md#fixture-files-to-add) for the parser-side checklist. Keep the existing stream archive format and include only the affected family's changed captures. The Unified YAML migration does not change this workflow.

The legacy streaming-v1 fixture schema is documented in [`toolcalling/fixtures-stream-v1/README.md`](toolcalling/fixtures-stream-v1/README.md). Capture and viewer commands are documented in [`utils/README.md`](utils/README.md).

## Fixture Workflows

The commands below cover both the older archive-backed tests and Unified. Apply the recent-YAML and version-archive contract above to Unified only; do not migrate the other stores as part of a Unified change.

**Title fixture/table-only PRs `chore(conformance):`, never `feat:`.** The repo is squash-merge only and GitHub is set to `squash_merge_commit_title: PR_TITLE` with a BLANK body, so the PR TITLE becomes the entire commit message on `main` — it is the only Conventional Commit that release-plz ever reads. The branch's own commit types are discarded, so retitling the PR is both necessary and sufficient. `feat:` proposes a MINOR version bump on every crate whose packaged contents changed ([`../RELEASING.md`](../RELEASING.md#bump-policy) has the full bump table); re-capturing fixtures or re-rendering the table is not a library feature and must not move a published version. Use `feat:` only when parser CODE under `parsers/` changed behaviour. Fixture-only work is outside every crate's packaged contents, so it proposes no bump.

### 1. Capture a new vLLM/SGLang engine version against Unified

New peer capture jobs use `UNIFIED.*`. Preserve existing archive-backed batch, stream, batch-on-stream, and reasoning observations as historical evidence.

1. Pin the exact peer package or source version and use its matching immutable environment. Keep the capture adapter bundle with the run.
2. Build one job per applicable family with `capture_unified_job.py --family <family> --mode <mode> --output <job.json>`. The job copies the authored input, initialization, schemas, finish reason, and chunk schedule. Use `batch` for a batch pipeline; retain separate implementation identities for batch and streaming results.
3. Run the version's own registered parser path. `capture_unified_batch.py --engine vllm|sglang --jobs <job-directory> --output <capture-directory>` captures batch jobs inside the matching environment. Use the shared Unified streaming adapter for streaming jobs. Record parser errors and unsupported capabilities explicitly.
4. Import each result with `import_peer_unified.py --store <store> --job <job.json> --capture <capture.json> --provenance <provenance.json> --bundle <adapter-bundle> --implementation <implementation>`. Preserve original observations and execution receipts.
5. Set `CONFORMANCE_HISTORY_SOURCE` to the verified external directory and package with `package_fixtures.py --unified-only`, refresh and check with `viewer_catalog.py --unified-only` and `viewer_catalog.py --unified-only --check`, then publish through `conformance-publish`. Inspect the maintained Unified viewer's version selectors, selected-family union, unavailable explanations, and original source links.

The legacy capture commands in `utils/README.md` are retained for explicit historical maintenance. They are not part of a new Unified capture run.

### Unified peer comparisons and recapture provenance

The three engine cards retain the original report layout and select among existing Dynamo Unified, vLLM Python, vLLM Rust, SGLang Python, and separate historical batch captures. Only slot 1 has a default: the latest Dynamo v2 Rust Unified parser. Slots 2 and 3 display None until selected. Batch observations retain final reasoning, visible content, and ordered calls; their rows say “Order/chunk not applicable for batch.” Family labels show `v1`, `v1/v2 split`, or `v2` coverage: the matrix includes `UNIFIED.v1_split-parallel_tool_calls` for every family with an archived Dynamo v1 path, including GPT-OSS, while Muse Glimmer remains v2-only.

Each family stores its authored request and GOLDEN in `inputs_and_golden.yaml`. Every version snapshot stores the complete set of resolved records that exists at that checkpoint, including unavailable explanations and missing cases. Records inherited from an earlier capture retain their original `member` and producer provenance. Inserting or correcting one version leaves later snapshots unchanged unless propagation is requested.

Complete snapshots repeat resolved metadata and receipt bodies in their records, with no parent, execution-manifest, or receipt-file dependency. Inline execution proof retains the sealed receipt identity and request hash bindings. Historical unreferenced manifests are preserved as `historical_evidence`; their execution case membership, metadata membership, receipt bodies, and receipt membership remain distinct. Unsupported cases remain represented as unavailable observations, and membership does not claim that every case executed. Import admission validates sealed raw results and rejects an entirely unsupported capture; coverage keeps per-case statuses separate.

Peer jobs bind the family, input, initialization, tools, chunk schedule, and termination to the capture. `capture_unified_job.py` builds the family jobs; the capture caller freezes those jobs and the adapter bundle before running against the selected installed package or tagged Rust source. `import_peer_unified.py` validates the job and provenance before publishing a transaction. Its compatibility entry point is `import_sglang_unified.py`. A historical result whose request was not retained remains in YAML; a request-bound execution is stored separately as `verified_recapture`. The report uses that supplement and explains why its difference from the original does not prove a parser change. Adapter-unavailable vLLM results can also retain a separate recapture without replacing the original evidence.

An unsupported request is an explicit unavailable result carrying the attempted request and a reason; it is not a successful empty parse. Parser errors retain the error and partial streaming evidence separately. Package the plain-version YAML with `package_fixtures.py --unified-only`, refresh the catalog with `viewer_catalog.py --unified-only`, and inspect the maintained browser viewer. Inspect the sibling JSON and the Unified comparison controls, unavailable explanations, and divergent outputs. The generated HTML, JSON, and capture matrices are review artifacts, not source files to commit.

### 2. Fix a Dynamo parser and refresh its expected outputs

For each conformance fix, add a regression test column that shows red for the previous (before) version and green for the fixed (after) version in the rendered report. Capture both versions using the same input, tools, initialization, and chunk schedule, and compare both against the same authored GOLDEN. Preserve the delivery pattern that triggers the bug, including single-chunk input when chunking affects the failure.

1. Fix the code under `parsers/v1/` or `parsers/v2/`.
2. `cargo test --workspace` — if the fix changes output, the parity tests FAIL. That is the regression gate working: decide whether the diff is a bug in your fix or an intended behavior change.
3. For an intended Unified change, use the plain-version YAML contract above. Keep the first capture's source origin in metadata; do not create another identity when its SHA changes. For the older tests, retain their existing capture format.
4. Commit only the parser fix, affected family captures, and required manifest changes. CI must compare code against the recorded outputs; release publication is a separate workflow.

### Unified parser hard gate

Unified work is complete only when the selected current Dynamo column has **zero empty cells and zero red cells** for the affected family. This is a hard gate. `reason:`, `unavailable:`, a historical column, stale HTML, or changing GOLDEN to match broken output does not satisfy it.

For a model-family conversion, collection is mandatory work, not a follow-up. Generate the family's Unified authored inputs and golden events, run the live Unified parser capture with its verified source identity, explode it, update the YAML history and manifest pin, refresh the catalog, and reload the browser viewer. The current capture node must exist in each affected family history before the row can be considered collected. The maintained root `index.html` is the viewer; `conformance/CONFORMANCE_v2.html` redirects old links and preserves their query and fragment.

Use this loop:

1. **Write:** make the parser or capture change.
2. **Read:** serve the checkout over HTTP, open `/?tab=tab-unified`, and inspect every empty or red cell's popup. Compare its exact input, request initialization, chunks, GOLDEN events, and current Dynamo events.
3. **Fix:** repair the owner of the discrepancy. An empty current cell is missing capture data. A red current cell is a parser mismatch unless the authored GOLDEN is demonstrably wrong.
4. **Regenerate:** publish the current plain-version YAML and required manifest changes, then refresh the catalog from the same worktree and reload the viewer.
5. **Re-read:** inspect the selected Unified column again. Repeat until both counts are zero.

The scoped gate is `conformance/utils/check.sh status --model qwen3 --tab unified`. Node reads the checked-in source resources, prints every empty or red case, and exits 1 until the row is clear. Invalid arguments and resource/model failures exit 2. `node conformance/viewer/status.js --status-path <output.json>` writes the same structured status report without rendering HTML. Run the native `unified_render` capture test and `unified_parity` test as well; the viewer and status command do not replace parser validation.

The complete current-family collection sequence is:

```bash
bash conformance/utils/regenerate_unified.sh
```

Use `bash conformance/utils/regenerate_unified.sh` to run this sequence as one gate. It compiles and captures the live Unified parser, updates the YAML history and materializes the selected Unified releases into a private fixture tree, refreshes the catalog, and runs the consuming tests and Node status checks. It fails when current capture coverage or Unified results disagree with the authored cases and known-divergence registry. Reload the browser viewer to inspect any red or empty cells.

Unified regeneration requires an explicit `CONFORMANCE_HISTORY_SOURCE`. It verifies the pin, preserves new execution evidence in the external dataset, then updates current YAML and the pin together. Current parser tests and latest report generation use only local authored inputs and current YAML; they do not open history.

The browser viewer reads the compact Unified resource catalog on normal visits. Legacy archive raw links load the full catalog on demand and require the same revision. Refresh both catalogs after packaging or source metadata changes; they contain resource identities and control metadata, with no captured outputs or computed cells.

### Structural-tag request checks

Unified cases may carry `init.structural_tag` with a response `schema` and complete `rejected` streams. The shared Rust request handler builds the real family structural tag and invokes `utils/src/check_structural_tag.py` before feeding the accepted stream into the parser. The CPU matcher checks the authored chunk schedule, every character split, and each rejection probe. A schema-only fixture with parser checks alone does not cover generation constraints. Parser-only peer adapters report unsupported request mode.

Install the pinned CPU verifier with `python3 -m pip install --no-deps -r conformance/utils/structural-tag-requirements.in`. Set `CONFORMANCE_GRAMMAR_PYTHON` to that interpreter when it differs from `python3`. The verifier loads XGrammar's native TVM-FFI binding and does not import torch or transformers. Rust CI installs this environment before running conformance. `UNIFIED.36-structured_response_after_reasoning` covers the available GPT-OSS, Qwen3, GLM and DeepSeek V4 builders.

### 3. Version rule: one capture name per crate version

External Unified captures retain release identities such as `dynamo_v2-0.6.1.yaml`. Local current filenames are generic, such as `captures/unified/qwen3/dynamo_v2.yaml`; their YAML and current-index metadata retain the real release and original producer identity. The first capture records its source SHA and Git commit as origin metadata. Changed bytes cannot replace an existing frozen capture identity. Historical backfills do not advance newer current snapshots. Crate publication follows [`../RELEASING.md`](../RELEASING.md#manual-version-peg-fixture-synced-releases).

The Rust capture readers, cross-version capture harness, and live Unified guard now select plain semantic-version captures directly. They no longer use the JSON `--select-capture` inventory, `--select-source-snapshot`, or patch/source-snapshot overlays. Python report readers use `--format label`, and newly packaged Unified YAML uses semantic versions. The Unified peer controls and preservation rules above describe the backfilled vLLM and SGLang histories; older corpus formats keep their existing readers.

### 4. What CI actually checks (the regression gate)

The `rust` CI job materializes the local current YAML and authored inputs, then runs current parser code against those expectations. It does not retrieve history or build the report publisher first:

- `conformance_toolcalling`: v1 code vs `expected.dynamo_v1` in the `dynamo_v1-<ver>/` dir of `fixtures-batch-v1`.
- `conformance_toolcalling_stream`: v2 code against the selected current per-family stream observations. The v1 jail-buffer reference remains a separate implementation and mode.
- `conformance_toolcalling_batch_via_stream`: v2 code vs the `fixtures-batch-on-stream-v1` expectations.

A parser change that alters output fails CI until the fixtures are re-captured and committed (workflow 2). Rust and fixture-only checks start alongside native publication. `pages-build` uploads the candidate site before qualification; eight independent qualification jobs compare source YAML against resolved JSON, partition the complete matrix candidate inventory, and check historical receipt bindings. Within each job, these read-only checks run in three independent workers. Matrix partitions include candidate controls without retained measurements; every partition also retains the default Reference and targeted earliest-history probes. Catalog checks share the store reader lease; catalog installation and recovery retain exclusive ownership. Authored-only checks load family inputs without expanding capture history, and historical receipt checks stream one snapshot at a time.

The `conformance-core` job checks the catalog, coverage, legacy storage, and publication-dependent utility tests. Smoke tests run in two independent browser groups; interactive viewer tests and popup regressions run in three groups. First-visible-matrix and strict native publication/browser acceptance run separately. Each browser group owns its fixtures and Chrome processes. Every conformance job has a 20-minute limit. The stable `conformance-table` status requires Rust, publication, fixture checks, all qualification shards, and all browser groups before deployment.

`ci_coverage.py` selects `latest-two` only when every changed path belongs to its explicit presentation allowlist. This profile compares the latest two JSON captures per implementation and mode, plus targeted historical model regressions. Evidence, reader, schema, provenance, conversion, classifier, deleted, empty, and unclassified changes use exhaustive coverage. Historical receipt and introduced-case membership checks remain exhaustive under both profiles. Qualification validates complete publication membership before partitioning comparison work and prints its selected resource paths. Test groups partition sorted collected node IDs; unknown tests remain in the fixture or publication lane through the shared collection owner.

Native pack compression uses the publication worker pool with deterministic pack identities, dependency boundaries, and gzip level 9. The publication cache hashes publication inputs: permanent source bytes, the reader and publisher, authored inputs, publication configuration and policy, release tags, and toolchain identity. Unrelated source, test, and documentation changes can reuse a matching publication. Cache reuse updates the current source-link revision in JSON and embedded HTML; the original generation revision, command, time, catalog identities, and payload bytes remain unchanged. Cached archive and binary hashes are checked before reuse; missing or corrupt cache entries fall back to cold generation. Cache hits still run qualification and browser acceptance. Permanent capture YAML and authored YAML remain unchanged. Running `check.sh ci` locally without `CONFORMANCE_CI_SHARD` runs the full suite; the workflow owns CI grouping.

### 5. Add a new Unified test case

Author family-native inputs and GOLDEN expectations in `utils/src/gen_unified_golden.py`, independently of parser output. Reuse `schema_cases.native_call_wire` for registered grammars and `unified_taxonomy.py` for stable descriptive IDs. Preserve existing inputs, identities, and expectations. Scope a grammar-specific case explicitly; a runtime that lacks a request mode must record an unsupported capability for the applicable case.

1. Add the shared scenario and every applicable family renderer, tool schema, initialization, finish reason, and chunk schedule.
2. Document its contract and grammar exclusions in `utils/lib/parsers/UNIFIED_CASES.md`.
3. Generate the authored corpus with `python3 utils/src/gen_unified_golden.py` and append new authoritative inputs with `python3 utils/src/explode_unified_fixtures.py --authored-inputs`, from `conformance/`.
4. Execute the frozen inventory through the shared family job builder and version adapters. Import request-bound results with `import_peer_unified.py`; retain prior observations and origins when adding verified recaptures. Account for every version, family, and applicable case as executed, error, unsupported, missing, or blocked.
5. Refresh and check `viewer_catalog.py --unified-only`, publish typed JSON through `conformance-publish`, and verify the maintained viewer's three selectors, selected-family union, Vertical layout, and case popups. Missing captures must remain missing evidence.

### 6. Changing an existing authored test input

Ask the user before modifying an authored test input. Changing one affects every published checkpoint that uses it. After approval, recapture every affected version from its matching parser source, package with `package_fixtures.py --unified-only`, materialize the Unified YAML for native tests, refresh and check with `viewer_catalog.py --unified-only`, and reload the viewer.

#### Historical stream archive maintenance

The archive-specific commands below apply only to explicitly requested maintenance of retained historical evidence. Every newly published Dynamo v2 stream archive, appended case, or correction requires a receipt generated by `refresh_dynamo_captures.py stream --receipt <path>` from the matching parser source; packaging verifies the source fingerprint and exact input/result hashes. An existing archive may add a case only when the shared input archive adds that case ID. Corrections require `package_fixtures.py --stream-only --replace-stream-case <full-case-id>` for each approved case ID and the same receipt passed with `--stream-capture-receipt`. The validator checks that all other case results, capture metadata, archive members, and non-YAML members remain unchanged. Never label a newer parser capture as an older version. Do not add a duplicate capture, source-qualified name, or patch overlay for one version; the history must remain one semantic checkpoint per meaningful output change.

### 7. Classify a v1-batch vs v2-stream difference (`known-divergences.yaml`)

`conformance_toolcalling_batch_via_stream` compares the v2 stream parser on batch text against v1's `expected.dynamo_v1`. v1 and v2 differ **by design** (v2 preserves surrounding/inter-call prose that v1 batch trims; v2 recovers bare calls v1 drops). When a case legitimately diverges, add it to `toolcalling/known-divergences.yaml` under `<family> → TOOLCALLING.batch.<case> → stream_vs_batch: <note>` (reuse the `*svb-surrounding-text` / `*svb-recovery` anchors). An entry with a note is an allowed, documented difference; a MISSING entry fails the test — so the file is also the audit trail of "v2 improved on v1 here." Do NOT add an entry to paper over an actual regression (v2 dropping text, leaking markup, corrupting args) — fix the parser instead.

The gate also asserts exact calls and normal text against every executed case's pinned `fixtures-batch-on-stream-v1` v2 capture. A parity note cannot permit a different live result. Missing or wrong-version captures, changed output, empty notes, and stale or unvisited entries fail. Independent authored GOLDEN values remain the correctness oracle; pinned parser captures describe observed behavior.

A separately triaged, pre-existing v2 defect may be retained while a scoped change is reviewed, following the exact-output policy in `unified-known-divergences.yaml`. Use a distinct `baseline_defect` note starting with `FIXME (v2 defect):` and `baseline_defect_actual` containing the exact observed result as JSON. The gate checks both that observation and the pinned capture, reports the defect separately, and rejects changed or stale entries. The report keeps the differing outputs and FIXME diagnostic visible without the intended-difference badge. This records a deferred parser fix; it does not change a golden, bless the defective result, or permit a new regression.

### 8. Coverage taxonomy: what "complete fixtures for a family" means (DIS-2442)

`conformance/case-taxonomy.yaml` is the machine-readable definition of complete coverage — every batch/stream/reasoning case group and sub-case, with per-case requiredness and applicability rules. It replaces the old implicit standard (the union of `description:` fields across ~20 families that reviewers had to reverse-engineer per PR).

For Unified corpus changes, regeneration is part of the edit, not a final cleanup step. After every change to a generator, taxonomy, golden specification, fixture manifest, capture label, or coverage documentation, run the generator, write complete snapshots into the affected engine/version archives, refresh the manifest pin and viewer catalog, and run the consuming Rust/Python tests. Reload the maintained browser viewer from those resources after the final edit. Before reporting or pushing, assert that every snapshot and its authored `inputs_and_golden.yaml` cover the expected case IDs and that source member references preserve the original capture identity. Source tests against stale YAML or a stale catalog do not validate the change.

```bash
# The authoring loop for a new family: the FAIL list is the fixture TODO list.
conformance/utils/check.sh coverage --family <family>

# Whole-corpus gate (what CI runs in the conformance-table job):
conformance/utils/check.sh coverage
```

Rules the lint enforces: a required case must exist as real input (`model_text`/`chunks`) or as a placeholder carrying an `explanation:` (silence fails; "not yet authored" placeholders warn — they are the acknowledged backfill list). A family registered in `parser_families.yaml` with no fixtures dir for a suite fails (the "ALL stream cases missing" class). Case IDs unknown to the taxonomy fail, so a PR that invents a new group/sub-case must extend `case-taxonomy.yaml` in the same PR. Pre-taxonomy gaps are grandfathered under `known_gaps:` — remove the ID when the fixture lands.

The same command runs the marker-registration lint: each family declares its grammar tokens once in the `markers:` section of `parser_families.yaml` (`pairs` / `singletons` / `leak`), from which the comparison leak detector (`viewer/comparison.js`) and popup token coloring (`utils/src/assets/colorize.js`) are derived.

### Invariants the tooling now enforces (so you don't have to remember)

- **Native tests read the manifest-pinned snapshot directory directly**, rather than the shared `<cache>/toolcalling` symlink. Capture staging uses the same pinned directory through `utils/src/_common.sh`; the browser and Node models validate the active resources indexed by the catalog.
- **Full historical packaging recomputes archive hashes and pins complete historical catalog descriptors.** Unified-only packaging preserves those manifest-bound historical descriptors and pins without reading archives; it recomputes and validates the Unified store digest. A changed historical descriptor requires an explicit full refresh.
- **Fixture coverage is diffed against `case-taxonomy.yaml` in CI** (`check_family_coverage.py`, section 8) — a new family missing required case groups, a whole stream corpus, or an unexplained placeholder fails the conformance-table job before human review (the PR #120 gap classes).
- **Family grammar tokens are declared once in `parser_families.yaml` `markers:`** — the leak detector and popup colorizer derive from it, and a family without a declaration fails the coverage lint (undeclared tokens used to render leaks as clean cells and every token as a red orphan).

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
