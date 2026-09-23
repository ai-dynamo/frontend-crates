<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# conformance

Parser conformance fixtures, fixture-based Rust tests, and HTML renderers for frontend-crates.

## Unified storage contract: plain versioned YAML only

The intent of #257 is to remove code, duplicate captures, merge conflicts, and file-change noise. Unified conformance uses readable YAML named by implementation and semantic version, such as `families/glm47/dynamo_v2-0.7.0.yaml`. The batch, streaming, batch-on-stream, and reasoning corpora also use reviewable YAML under `fixtures-v1/`, with their original capture identities and resolution rules.

- No new Unified `*.patch*.yaml`, `.tar.gz` capture archives, `+source.<hash>`, `+sha<hash>`, or other hash/commit-qualified capture names. Do not add these formats to satisfy an old Unified reader or test.
- One version identifies one checkpoint per family and implementation. A different source SHA alone does not justify a recapture, duplicate file, or manifest change. The first capture may keep its original source SHA inside the YAML as origin metadata; later SHAs do not change its identity.
- New parser behavior requires a manually pinned, unpublished crate version before capture. Follow [Pin the release version before capture](#pin-the-release-version-before-capture).
- Readers and HTML/JSON generators discover recorded checkpoints by filename and semantic-version order. The default Unified reference selects each family's newest actual capture and displays that version in its cell details. A concrete version with no checkpoint for that family remains unavailable. Within a recorded checkpoint, omitted cases inherit prior observations; an actual capture with unchanged output needs only its provenance and an empty `changes` mapping.
- Keep inputs and GOLDEN separate from recorded outputs. Changing an existing input is discouraged; if necessary, rerun and update every prior affected version. Never change GOLDEN to hide a parser failure.
- Family-specific PRs add only that family's captures and required code. Do not bundle other families' YAML in an archive. Shared cases and non-GLM fixes from #234 belong to #241.

The `v2` directory names describe the older streaming parser tests as well as Unified; they do not define the storage format. Unified's plain semantic-version rule does not rename legacy `.patch1` or `+current` checkpoints. The legacy store materializes the original loose layout for the existing readers.

### Pin the release version before capture

When adding or converting a parser, or changing its captured behavior, manually pin the intended release version in the same PR before recording output. The invariant is `capture version == crate version == version published after merge`. An automatic bump after capture would break that equality.

1. Fetch the current release tags and check published versions. Choose the next appropriate unpublished semantic version; a feature branch still carrying an old version does not make its new behavior part of that old release.
2. Set the parser crate's `Cargo.toml` version explicitly. Update its dependent manifests, `Cargo.lock`, and the conformance manifest's crate version together. For Dynamo v2, this includes `parsers/v2/Cargo.toml`, `parsers/v2-py/Cargo.toml`, `conformance/Cargo.toml`, and `conformance/fixtures-manifest.json`.
3. Build and capture from that pinned version. Keep the Unified YAML filename and origin version aligned with it, then run the capture validation and regenerate the report. Renaming an older capture without validating it against the pinned build is insufficient.
4. Before merge, recheck that the pinned version is still unpublished. If another PR has released it, choose a new version and repeat the capture validation. After merge, verify the release tag and published crate use the captured version.

The release workflow honors this [manual version override](../RELEASING.md#manual-version-peg-fixture-synced-releases): it checks compatibility and publishes the explicitly pinned version without another automatic bump. For #234, GLM Unified and its pending release are both `0.7.0`; GLM Unified has no `0.6.x` captures. Unchanged families continue to inherit their existing captures.

## Ownership

Parser v1/v2 terminology, migration steps, and fixture ownership are documented in [`../docs/PARSERS-V2-MIGRATION-PLAN.md`](../docs/PARSERS-V2-MIGRATION-PLAN.md). New streaming parser authors should also read [`../parsers/v2/README.md`](../parsers/v2/README.md); it explains the vLLM-shaped Rust parser contract, the v2 fixture schema, and the exact `conformance/toolcalling/*` files to add. This README covers conformance layout, render outputs, and test commands.

## Parser paths and modes (universal convention)

- **Implementation generation:** `dynamo_v1` means the `dynamo-parsers` batch implementation, used for plain batch and jail+batch. `dynamo_v2` means the `dynamo-parsers-v2` implementation crate, which contains both non-Unified incremental parsers and Unified parsers.
- **Conformance convention:** streaming v1 means the legacy, non-Unified stream and batch-on-stream corpora. Unified v2 means one parser owns the ordered reasoning, visible-text, and tool-call stream. A `dynamo_v2-*` capture inside `fixtures-stream-v1` names a v2 implementation measured against the v1 convention.
- **Why the names changed:** the incremental parser was originally called `stream-v2` because it was intended to replace batch+jail. Unified later became a second v2 convention. Renaming the older corpus to streaming v1 removes that collision; it does not rename parser crates or implementation keys.
- **Runtime behavior today:** parser identities are configured on the deployed model. `DYN_ENABLE_EXPERIMENTAL_PARSERS_V2` enables both experimental routes; it does not choose one by itself. Routing checks Unified first: the exact `tool_call_parser=qwen3_coder` plus `reasoning_parser=qwen3` pair uses Qwen3 Unified when the flag is on. If no Unified pair matches, eligible `qwen3_coder` or `deepseek_v4` requests use the older non-Unified incremental parser. All remaining tool-parsing requests use batch+jail. DeepSeek V4.1 and Muse have separate Unified routes that do not depend on this flag. There is no general fallback chain after a selected parser fails.
- **Per-chunk cells show WHEN output reaches the consumer.** Streaming parsers emit whenever something is parseable, so their cells carry deltas at real chunk positions. The v1 jail bursts at end-of-call — its captures record emission order, not per-chunk timing — so its per-chunk cells stay `—` with a "(bursts at end of call; per-chunk timing not recorded)" header note, and its output appears only in the `assembled` row.
- In the rendered tables, the TC (stream) tab's **default Reference is Dynamo v1 (jail+batch)** — the one stream path with coverage on every family — so every row shows data by default. Star **Dynamo v2** as the Reference to see v2's streaming coverage; families v2 doesn't implement yet gray out as "not implemented".

## Layout

```
conformance/
├── fixtures-manifest.json                         # pins the active fixture snapshot (sha256 per shard)
├── fixtures-v1/                                   # reviewable batch, stream, batch-on-stream, and reasoning history
├── fixtures-unified-v2/                           # reviewable append-only YAML history for Unified captures
├── tests/*.rs                                     # Rust fixture tests (fixtures extracted from the store on first run)
└── utils/                                         # render, check, and record helpers
```

Tool-calling and reasoning captures live under `conformance/fixtures-v1/`. Unified history lives under `conformance/fixtures-unified-v2/`. `extract_fixtures.py` materializes both stores in `~/.cache/dynamo/conformance-fixtures/` for existing consumers. Snapshot layout:

```
toolcalling/fixtures-batch-v1/<family>/           # v1 tool-calling batch cases
toolcalling/fixtures-stream-v1/<family>/          # legacy, non-Unified stream cases
toolcalling/fixtures-batch-on-stream-v1/<family>/ # legacy, non-Unified batch-on-stream cases
reasoning/fixtures-v1/inputs/<family>/            # v1 reasoning cases
```

**Unified capture history is append-only.** Each capture is one `<implementation>-<semantic-version>.yaml` checkpoint. Its first capture retains a compact source origin in YAML metadata; source SHA does not create another capture identity. Record only changed case observations at subsequent checkpoints. A later measured version with identical output keeps its capture provenance and inherits the observations. Publishing a crate version alone does not create a capture checkpoint or relabel earlier output. Re-running an unchanged checkpoint does not create a file or manifest change. Test inputs are immutable after capture: changing one requires recapturing and updating every prior semantic version. `dynamo_v1` and `dynamo_v2` have separate version histories and never fold together. The manifest-pinned snapshot determines what the chart shows.

## Maintaining cases and capture history

A report column is a test case shared by the applicable families. A capture file records one implementation/version for one family. Adding a column and adding a parser version therefore change different parts of the store.

| Change | Inputs and files that must change | Capture and history requirement |
|---|---|---|
| Add a test column | Update the authored case, its applicability/description, and the affected families' `inputs_and_golden.yaml`. Unified authoring uses `utils/src/gen_unified_golden.py` and `utils/src/unified_taxonomy.py`; legacy authoring uses the loose input files and `case-taxonomy.yaml`. | Run every applicable implementation on the new input. Historical columns need captures from their actual historical producers; an unsupported API must remain explicitly unavailable. The incremental importer rejects additions to an already recorded checkpoint, so historical backfill needs a separately reviewed reconstruction of those checkpoints. It is not automatic. |
| Modify an existing test input | A change to text, chunks, tool schema, initialization, or finish behavior changes the request that produced the output. Description-only edits do not. | Normal packaging rejects changing a captured request. Prefer a new stable case ID. Reusing the old ID requires recapturing every affected recorded version with its original producer and rebuilding the affected history together; there is no automated rewrite command for that operation. |
| Add a parser version | Record the actual producer version and add `<implementation>-<version>.yaml` for each captured family. A Dynamo behavior change also requires the crate/version updates described above. | Capture the active cases before compaction. An unchanged measured Unified version keeps its provenance with `changes: {}`; copying an older capture into a new version directory is not a measurement. A family/version that was never captured stays unavailable. |
| Hide an equivalent version in the report | Update the selector equivalence mapping in `utils/src/generate_conformance_table.py` and its equality regression. | Compare complete captured observations, including event order and annotations. Keep the stored version and provenance; saved links resolve to its visible equivalent. No history needs to move. |
| Physically remove a stored version | This changes the history, not just the selector. Do not delete a checkpoint file by hand. | Resolve every retained version first, move any required observations and metadata into surviving checkpoints, and rebuild their deltas. Compare every retained result and its provenance before publishing. There is no general removal/rebase command; legacy `--prune` rejects a deletion that changes retained resolved state and does not transfer the missing data for you. Unified has no version-prune operation. |

For example, if version 1 records `A`, version 2 records `B`, and version 3 inherits `B`, deleting version 2 makes version 3 inherit `A`. The required `B` observation must be stored in version 3 before version 2 can be removed. Its original producer must remain identifiable; copying `B` and assigning version 3 as its producer would falsify the history. Removing the oldest version may require a new full anchor. Removing the newest version changes the default selection and discards that recorded measurement. Neither action is equivalent to hiding a selector.

Every published store change also updates `fixtures-manifest.json`, then requires extraction, reader checks, and regeneration of `CONFORMANCE_v2.html` and `CONFORMANCE_v2.json`. Captures from another parser version cannot substitute for missing historical measurements. The generated reports and extracted compatibility files are verification outputs, not additional authored sources.

### What compaction guarantees

Unified stores compare resolved requests and observations with the preceding capture. Equivalent inline and referenced requests share that comparison. Changed output is stored once, omitted cases inherit, and per-case absence and metadata changes have explicit representations. The small version/provenance checkpoint remains even when there is no new output. Re-importing the same checkpoint leaves its file unchanged.

Legacy batch, stream, and reasoning histories also omit unchanged observations where their readers can reconstruct them. Compaction preserves case annotations, including their removal, and streaming keeps enough output for the Rust reader that excludes qualified captures. These requirements can retain an identical payload at a later version. Batch-on-stream currently stores complete blocks per implementation/version because its selected snapshots are reconstructed independently; it does not yet share the sparse compactor. This PR therefore does not promise zero repeated YAML across every corpus.

The immutable-capture checks and legacy prune comparison protect the normal packaging path. They do not make arbitrary manual edits or deletions safe, and a manifest hash only identifies the resulting bytes. In particular, older Unified observations can reference the current request without storing its digest: directly editing that request can silently change the apparent historical input. Loading the store alone does not detect this; use the guarded packaging path and compare against the original store. A future rewrite/removal tool must prove equivalence against the original store, preserve capture provenance, update selection/unavailability metadata, and publish the stores and manifest together. Until that tool exists, keep historical files and use selector hiding for display cleanup.

## End-to-end test cases (a separate surface, kept elsewhere)

Everything under `conformance/` is HERMETIC: a fixed byte string goes into a parser and an exact event list comes out, with no model and no worker. There is a second, complementary surface — the **end-to-end test cases** — which sends real requests to a real Dynamo worker and checks the returned response. It has its own suite/category taxonomy (`reasoning` and `tool_calling` suites; `core` / `complex` / `history` / `tool_boundary` / `arguments_schema` / `parallel_lifecycle` categories), each case run in `stream` and `non-stream` mode, under two thinking-budget variants, at worker `--stream-interval` 20 and 1.

**That suite and its artifacts do not live in this repo.** Its cases arrive as a self-contained HTML report (e.g. `qwen36_pr163_test_cases.html`) whose `const REPORT` embeds every request, expectation and response; the per-case artifacts it names are files of the form `end-to-end case-<num>-<case>-<variant>.json` on whichever machine ran the harness.

The two surfaces answer different questions and neither replaces the other, so where a hermetic case has an e2e counterpart it carries an `End-to-end:` tag naming it. **Those tags and the full artifact index are maintained in ONE place — `utils/lib/parsers/UNIFIED_CASES.md` ("End-to-end test cases" and "Artifact index").** Do not copy the mapping into another doc; a second copy drifts, and `utils/tests/test_unified_taxonomy_covers_corpus.py` only pins the two that already exist.

Per-case tagging currently covers the UNIFIED surface only. The reasoning and tool-calling case docs below have no e2e tags yet — that mapping has not been worked out, and an untagged case means "not yet mapped", not "no e2e coverage".

## Render Outputs

| Output | Command | Parser version | Fixture version |
|---|---|---|---|
| v2 conformance HTML | `conformance/utils/render_table_v2.sh` | Mixed bridge table: `TC batch (v1)` and reasoning tabs use the v1 parser; legacy stream and batch-on-stream tabs use Dynamo parser v2 code. | `TC batch (v1)` uses v1 batch fixtures; legacy tabs use the legacy non-Unified v1 corpus convention; reasoning tabs use v1 reasoning fixtures. The default example output is `conformance/CONFORMANCE_v2.html`, and the render script also accepts a custom output path. |

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

Reasoning fixtures are rendered in the v2 HTML table; a Rust fixture harness for reasoning is still a follow-up.

## Refreshing Legacy Fixtures (v1)

Parser fixture sync from Dynamo is retired. Update v1 fixtures through normal frontend-crates PRs and run the renderer documented in [`../docs/PARSERS-V2-MIGRATION-PLAN.md`](../docs/PARSERS-V2-MIGRATION-PLAN.md#ownership).

## Adding Streaming Parser V2 Fixtures

Use [`../parsers/v2/README.md`](../parsers/v2/README.md#fixture-files-to-add) for the parser-side checklist. Package the affected family's changed captures into the YAML history store.

The legacy streaming-v1 fixture schema is documented in [`toolcalling/fixtures-stream-v1/README.md`](toolcalling/fixtures-stream-v1/README.md). Capture and render commands are documented in [`utils/README.md`](utils/README.md).

## Fixture Workflows

The commands below cover both YAML stores. Unified uses plain semantic-version checkpoints; legacy captures retain their historical names and sparse overlays.

**Title fixture/table-only PRs `chore(conformance):`, never `feat:`.** The repo is squash-merge only and GitHub is set to `squash_merge_commit_title: PR_TITLE` with a BLANK body, so the PR TITLE becomes the entire commit message on `main` — it is the only Conventional Commit that release-plz ever reads. The branch's own commit types are discarded, so retitling the PR is both necessary and sufficient. `feat:` proposes a MINOR version bump on every crate whose packaged contents changed ([`../RELEASING.md`](../RELEASING.md#bump-policy) has the full bump table); re-capturing fixtures or re-rendering the table is not a library feature and must not move a published version. Use `feat:` only when parser CODE under `parsers/` changed behaviour. Fixture-only work is outside every crate's packaged contents, so it proposes no bump.

### 1. Capture a new vLLM/SGLang engine version (new peer shards)

**The rule is ALL corpora, ALL families.** A new vLLM/SGLang version must land on EVERY tab — `TC batch`, legacy stream, legacy batch-on-stream, Unified, and reasoning — so the table stays consistent across tabs. Refreshing one tab and leaving the others behind is a bug, not a shortcut: every tab is multi-version and renders each captured version as its own comparison candidate.

0. **Rebase onto `main` FIRST.** The conformance renderer + capture tooling change often (the whole page was rewritten in DIS-2434; the marker layer in #127). Capturing on a stale base means redoing the render/verify against a since-rewritten generator. Rebase, then capture.
1. **Pin the version** in `utils/src/pyproject.stub.toml` (peer versions are read from there, never hardcoded). This repository owns the peer versions independently of Dynamo's runtime pins.
2. **Bring up the engine containers at the new versions and extract the corpus** so the capture seeds are on disk. Parser capture needs NO GPU — it feeds stored model-text through the parser, not the model:
   ```bash
   docker run -d --name vllm-localdev   --entrypoint sleep <vllm-image>   infinity
   docker run -d --name sglang-localdev --entrypoint sleep <sglang-image> infinity
   python3 conformance/utils/src/extract_fixtures.py   # materialize inputs the capturers read
   ```
3. **Capture EVERY corpus** — one command per corpus × peer. Each writes a NEW `<impl>-<newver>/` version dir (append-only; see the rule above). Capturers read the version LIVE from the container's `vllm.__version__` / `sglang.__version__`, write only cases that DIVERGE from the lowest-version anchor, never touch existing dirs, and skip (never fabricate) cases a parser can't run in-container.

   All peer captures run through one shared tool, `capture_peer_versions.py --corpus {batch,stream,reasoning} --impl {vllm_python,sglang_python,vllm_rust}` (omit `--impl` for all engines valid on that corpus; `vllm_rust` is stream-only):

   | Tab | vLLM Python | vLLM Rust | SGLang |
   |---|---|---|---|
   | `TC legacy stream (v1 convention)` | `capture_peer_versions.py --corpus stream --impl vllm_python` | `capture_peer_versions.py --corpus stream --impl vllm_rust` ‡ | `capture_peer_versions.py --corpus stream --impl sglang_python` |
   | `TC batch (v1)` | `capture_peer_versions.py --corpus batch --impl vllm_python` | — (no Rust batch parser) | `capture_peer_versions.py --corpus batch --impl sglang_python` |
   | `TC legacy batch-on-stream (v1 convention)` | `recapture_batch_on_stream.py` (in-place, single-snapshot) | `recapture_batch_on_stream.py` | `recapture_batch_on_stream.py` |
   | `Reasoning` | `capture_peer_versions.py --corpus reasoning --impl vllm_python` | — | `capture_peer_versions.py --corpus reasoning --impl sglang_python` |

   ‡ vLLM Rust is source-only: set `VLLM_RUST_SOURCE=<vllm checkout at the tag>` (or pass `--vllm-rust-source`) first. In vLLM ≥ 0.25 the crate is `vllm-parser` at `rust/src/parser` (was `vllm-tool-parser` at `rust/src/tool-parser`), and `ToolParserOutput` is an ordered events list. A parser that moved to the native `unified::` interface between releases is marked unavailable via the `tool::` probe — expected, not a failure.
4. **Publish:** run `package_fixtures.py` to update the affected family YAML and manifest pin, then extract and render the pinned snapshot.
5. **Verify the new version shows on ALL tabs.** The generator discovers each `<impl>-<newver>/` dir as its own candidate. Run `render_table_v2.sh` and confirm the new version is a Reference/Compare candidate on all four tabs (grep the rendered HTML), then `python3 -m pytest conformance/utils/tests/` and `check.sh dynamo all`. `test_model.py::test_v2_reasoning_uses_current_peers` specifically guards that the Reasoning tab surfaces every peer version dir — if it fails after you add a version, the tab lost multi-version rendering.

### 2. Fix a Dynamo parser and refresh its expected outputs

1. Fix the code under `parsers/v1/` or `parsers/v2/`.
2. `cargo test --workspace` — if the fix changes output, the parity tests FAIL. That is the regression gate working: decide whether the diff is a bug in your fix or an intended behavior change.
3. For an intended Unified change, use the plain-version YAML contract above. Keep the first capture's source origin in metadata; do not create another identity when its SHA changes. For the older tests, retain their existing capture format.
4. Commit only the parser fix, affected family captures, and required manifest changes. CI must compare code against the recorded outputs; release publication is a separate workflow.

### Unified parser hard gate

Unified work is complete only when the selected current Dynamo column has **zero empty cells and zero red cells** for the affected family. This is a hard gate. `reason:`, `unavailable:`, a historical column, stale HTML, or changing GOLDEN to match broken output does not satisfy it.

For a model-family conversion, collection is mandatory work, not a follow-up. Generate the family's Unified authored inputs and golden events, run the live Unified parser capture with its verified source identity, explode it, update the YAML history and manifest pin, extract the pinned snapshot, and render the canonical v2 report. The current capture node must exist in each affected family history before the row can be considered collected. Never generate `CONFORMANCE_unified.html`; `CONFORMANCE_v2.html` is the only HTML report for this workflow.

Use this loop:

1. **Write:** make the parser or capture change.
2. **Read:** render `conformance/CONFORMANCE_v2.html`, open the Unified tab, and inspect every empty or red cell's popup. Compare its exact input, request initialization, chunks, GOLDEN events, and current Dynamo events.
3. **Fix:** repair the owner of the discrepancy. An empty current cell is missing capture data. A red current cell is a parser mismatch unless the authored GOLDEN is demonstrably wrong.
4. **Regenerate:** publish the current plain-version YAML and required manifest changes, then render the HTML again from the same worktree.
5. **Re-read:** inspect the rendered Unified column again. Repeat until both counts are zero.

`render_table_v2.sh` always writes `conformance/CONFORMANCE_v2.json` beside the HTML and prints the total empty/red count. The standard scoped gate is `conformance/utils/check.sh status --model qwen3 --tab unified`; it renders first, prints every empty or red case, and exits nonzero until the row is clear. Use the lower-level `validate_conformance_status.py` only to inspect an already-rendered file. Then run `cargo test --locked -p dynamo-conformance-fixtures-v2 --test unified_render -- --nocapture` and `cargo test --locked -p dynamo-conformance-fixtures-v2 --test unified_parity -- --nocapture`. Do not report Unified work complete while either rendered count is nonzero.

The complete current-family collection sequence is:

```bash
python3 conformance/utils/src/gen_unified_golden.py
cargo test --locked -p dynamo-conformance-fixtures-v2 --test unified_render -- --nocapture
python3 conformance/utils/src/explode_unified_fixtures.py
python3 conformance/utils/src/package_fixtures.py
python3 conformance/utils/src/extract_fixtures.py --full-refresh
conformance/utils/render_table_v2.sh --output conformance/CONFORMANCE_v2.html
conformance/utils/check.sh status --model <family> --tab unified
cargo test --locked -p dynamo-conformance-fixtures-v2 --test unified_parity -- --nocapture
python3 -m pytest conformance/utils/tests/test_model.py
```

Use `bash conformance/utils/regenerate_unified.sh` to run this sequence as one gate. It compiles and captures the live Unified parser, updates and re-extracts the YAML history, renders the JSON/HTML report, runs the consuming tests, and fails when the generator, current history, or rendered Unified cells disagree. The script renders JSON and HTML even when a later validation stage fails, so humans can inspect the current red or empty cells; a failed exit code means the report is diagnostic, not ready to publish.

Do not substitute a loose harness feed for the package step. The v2 table reads the extracted packaged snapshot, so an un-packaged family cannot appear in its Unified tab.

### 3. Version rule: one capture name per crate version

The Unified storage contract at the top of this document owns naming, carry-forward, source-origin metadata, and input-change rules for Unified. Crate publication follows [`../RELEASING.md`](../RELEASING.md#manual-version-peg-fixture-synced-releases); it does not introduce another capture naming scheme.

TODO (follow-up PR; Rust cleanup is deferred from #257): remove the deprecated `dynamo_version.py` JSON producer and `--select-capture` inventory protocol, `capture_stimulus.py --select-source-snapshot`, and their patch/source compatibility helpers after migrating `conformance/tests/common/mod.rs`, `capture_cross_version.rs`, and `unified_render.rs`. These Rust harnesses still require the old protocol and tests. Python report readers use only `--format label`; newly packaged Unified YAML uses only semantic versions. Shared-family tests remain deferred to #241, and recovery of the pre-existing missing vLLM captures remains in #256.

### 4. What CI actually checks (the regression gate)

The `rust` CI job checks out the YAML stores, materializes the manifest-pinned snapshot, and runs the parity tests against the pinned expected output:

- `conformance_toolcalling`: v1 code vs `expected.dynamo_v1` in the `dynamo_v1-<ver>/` dir of `fixtures-batch-v1`.
- `conformance_toolcalling_stream`: v2 code vs `expected.dynamo_v2` folded from the LOWEST `dynamo_v2-<ver>/` dir — the v2 anchor. (The v1-jail reference lives in its own `dynamo_v1-3.0.0/` namespace and never enters the v2 fold. Overlay folding up to the pinned crate version is a follow-up; until then an intended v2 output change must be reflected in the anchor's expected blocks at re-capture.)
- `conformance_toolcalling_batch_via_stream`: v2 code vs the `fixtures-batch-on-stream-v1` expectations.

A parser change that alters output fails CI until the fixtures are re-captured and committed (workflow 2) — CI compares Dynamo against the pinned shard YAMLs, nothing else. The `conformance-table` CI job runs exactly one command, `conformance/utils/check.sh ci`, which re-renders both HTML pages from the same pinned store, runs the coverage/marker lint (section 8), and the chart-invariant guards. To add or change a conformance gate, edit `run_ci()` in `check.sh`; the workflow file stays untouched.

### 5. Add a new test case (e.g. a new `TOOLCALLING.streamv1.5.h`)

A "case" is one numeric-suffix sub-case shared across families. New case IDs use `<num>-<num>` or `<letters/num>-<num>`; do not allocate new letter suffixes because a long-running taxonomy can exhaust `a` through `z`. Adding one is FOUR edits, in order:

1. **Input.** Add the case to `toolcalling/fixtures-stream-v1/inputs/<family>/TOOLCALLING.streamv1.<N>.yaml` for each family it applies to — the shared per-chunk `delta_text` (schema in [`toolcalling/fixtures-stream-v1/README.md`](toolcalling/fixtures-stream-v1/README.md#fixture-schema)). Batch cases go under `toolcalling/fixtures-batch-v1/inputs/<family>/` instead.
2. **Description.** Add a bullet to `utils/lib/parsers/TOOLCALLING_STREAMING_V1_CASES.md` (or the batch/reasoning CASES.md) — the HTML "Case descriptions" section renders it, and the tooltip links to it.
3. **Grouping (easy to miss).** Add the case id to its band in **`utils/src/fixtures.py`** `BATCH_SUB_CASE_GROUPS` (the streamv1 tab reuses the batch taxonomy). If you skip this, the column still renders but sorts to the FAR RIGHT as an "unknown" case instead of beside its `<num>.*` siblings. That list now lives in exactly one place, so a case is one edit. A new `<num>.<letter>` should ideally key on its parent `<num>`, not enumerate every letter.
4. **Capture + package.** `refresh_dynamo_captures.py stream` (records the Dynamo v2 output for the new case), then `package_fixtures.py`, then commit all three published fixture paths. Peer engines (vLLM/SGLang) only cover the new case once re-captured against containers (workflow 1); until then the peer cells read `(no expectation)`.

### 6. Changing an existing Unified test input

The normal importer rejects this operation because existing output would no longer describe the input shown beside it. Use a new stable case ID, or perform the separately reviewed historical reconstruction described in "Maintaining cases and capture history" above. Running the ordinary packager after changing the input does not recapture old versions or authorize rewriting their results.

### 7. Classify a v1-batch vs v2-stream difference (`known-divergences.yaml`)

`conformance_toolcalling_batch_via_stream` compares the v2 stream parser on batch text against v1's `expected.dynamo_v1`. v1 and v2 differ **by design** (v2 preserves surrounding/inter-call prose that v1 batch trims; v2 recovers bare calls v1 drops). When a case legitimately diverges, add it to `toolcalling/known-divergences.yaml` under `<family> → TOOLCALLING.batch.<case> → stream_vs_batch: <note>` (reuse the `*svb-surrounding-text` / `*svb-recovery` anchors). An entry with a note is an allowed, documented difference; a MISSING entry fails the test — so the file is also the audit trail of "v2 improved on v1 here." Do NOT add an entry for an actual regression (v2 dropping text, leaking markup, corrupting args) — fix the parser instead.

### 8. Coverage taxonomy: what "complete fixtures for a family" means (DIS-2442)

`conformance/case-taxonomy.yaml` is the machine-readable definition of complete coverage — every batch/stream/reasoning case group and sub-case, with per-case requiredness and applicability rules. It replaces the old implicit standard (the union of `description:` fields across ~20 families that reviewers had to reverse-engineer per PR).

For Unified corpus changes, regeneration is part of the edit, not a final cleanup step. After every change to a generator, taxonomy, golden specification, fixture manifest, capture label, or coverage documentation, immediately run the generator, explode the loose captures, update `conformance/fixtures-unified-v2/`, refresh the manifest pin, render `CONFORMANCE_v2.json` and `CONFORMANCE_v2.html`, and run the consuming Rust/Python tests. Repeat that complete chain after the final edit. Before reporting or pushing, assert that the seven `inputs_and_golden.yaml` family documents and each Dynamo capture chain after resolving sparse checkpoints cover exactly the generator's case-ID set. Source tests against stale YAML or stale HTML do not validate the change.

```bash
# The authoring loop for a new family: the FAIL list is the fixture TODO list.
conformance/utils/check.sh coverage --family <family>

# Whole-corpus gate (what CI runs in the conformance-table job):
conformance/utils/check.sh coverage
```

Rules the lint enforces: a required case must exist as real input (`model_text`/`chunks`) or as a placeholder carrying an `explanation:` (silence fails; "not yet authored" placeholders warn — they are the acknowledged backfill list). A family registered in `parser_families.yaml` with no fixtures dir for a suite fails (the "ALL stream cases missing" class). Case IDs unknown to the taxonomy fail, so a PR that invents a new group/sub-case must extend `case-taxonomy.yaml` in the same PR. Pre-taxonomy gaps are grandfathered under `known_gaps:` — remove the ID when the fixture lands.

The same command runs the marker-registration lint: each family declares its grammar tokens once in the `markers:` section of `parser_families.yaml` (`pairs` / `singletons` / `leak`), from which the `↯` leak regex (`markers.py`) and the popup token coloring (`utils/src/tables/markup.py`) are derived.

### Invariants the tooling now enforces (so you don't have to remember)

- **Renders/tests read the manifest-pinned snapshot dir directly**, not the shared `<cache>/toolcalling` symlink — sibling `frontend-crates*` checkouts pinning other snapshots used to race to repoint it, so a render could read a snapshot missing the newest version. Enforced in `utils/src/_common.sh`.
- **`package_fixtures.py` recomputes the sha256 of every kept shard from disk**, never copying the prior manifest's value — a restored/re-pinned store file can no longer produce a manifest that lies about its content (which broke `extract_fixtures`' sha-verify).
- **Fixture coverage is diffed against `case-taxonomy.yaml` in CI** (`check_family_coverage.py`, section 8) — a new family missing required case groups, a whole stream corpus, or an unexplained placeholder fails the conformance-table job before human review (the PR #120 gap classes).
- **Family grammar tokens are declared once in `parser_families.yaml` `markers:`** — the leak detector and popup colorizer derive from it, and a family without a declaration fails the coverage lint (undeclared tokens used to render leaks as clean cells and every token as a red orphan).
