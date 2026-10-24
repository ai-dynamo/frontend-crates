# Conformance/Utils

Latest YAML lives beside retained versioned captures at `conformance/fixtures-unified-v2/families/<family>/<implementation>.yaml`. Each latest file seals its complete records and request bindings, with selection metadata separate from producer provenance. Latest generation and tests do not open historical captures or legacy archives. No backfill is needed; versioned v2 removal, historical moves, and compression are deferred to later PRs. Follow [the scope and acceptance plan](../../docs/CONFORMANCE-LATEST-PLAN.md) and [the report guide](../README.md#generate-the-latest-or-history-report).

This directory contains parser checks, fixture tooling, and the catalog builder. Both publication modes use the existing Rust publisher and Unified viewer. Generated latest HTML is standalone; history reports carry their own on-demand data and original downloads. Historical format readers remain available for explicit qualification. They resolve independent sparse datasets against their retained authored requests, then read validated checkpoint fragments lazily from the conversion cache. `--cache-dir` selects the publisher cache; the local default is `build/conformance-cache/`. Global report envelopes and transport packs are rebuilt from those fragments when presentation or global metadata changes.

Most work has three steps:

1. Verify parser code, table code, and fixture YAML.
2. Update parser code and/or fixture YAML.
3. Refresh the catalog and inspect the browser matrix.

## What Is A Fixture?

A fixture is a YAML test case for parser behavior. It contains model output text or stream chunks, plus the structured parser output each engine is expected to produce.

For tool-calling, the important fields are:

| Field | Meaning |
|---|---|
| `model_text` | Complete model output for batch-style parsing. |
| `chunks[].delta_text` / `chunks[].delta_token_ids` | Incremental model output for stream-style parsing. |
| `expected.dynamo_v1` / `expected.dynamo_v2` | Dynamo v1 (batch+jail) / v2 (stream) parser output — two SEPARATE impls, like `vllm_python` vs `vllm_rust`. |
| `expected.vllm_rust` | vLLM Rust parser output captured from a vLLM source checkout. |
| `expected.vllm_python` | vLLM Python parser output captured from the pinned Python package. |
| `expected.sglang_python` | SGLang Python parser output captured from the pinned Python package. |
| `captured_with.*` | Parser version used when output was captured. |
| `unavailable.*` | The parser does NOT exist for this family (no detector / not exposed), or is an intentional TODO. Benign — the cell renders grey `n/a`. |
| `exception.*` | The parser EXISTS and was run but RAISED on this input. `<msg>` is the verbatim exception (for vLLM Rust the named crate variant, e.g. `ToolParserError::ParsingFailed (near " not")`; for the Python peers `<ExceptionType>: <message>`). Distinct from `unavailable`: it renders the `✗` error marker, and when it is the selected Reference and a compared parser produced real output the cell reddens (a genuine disagreement), rather than grey `n/a`. |

Fixture locations:

Current observations for tool-calling, reasoning, and Unified live in local capture YAML. `extract_fixtures.py` materializes local current records into the compatibility tree for native Rust tests. A checkout viewer reads selected Unified YAML; a published viewer reads selected typed JSON. Legacy archive source views remain available on demand. `conformance/utils/src/parser_families.yaml` is parser configuration, not a fixture.

| Store path (inside snapshot) | Used By |
|---|---|
| `toolcalling/fixtures-batch-v1/` | Legacy batch regression corpus. Complete model output through batch parsers. |
| `toolcalling/fixtures-batch-on-stream-v1/` | Legacy streaming v1 convention. Complete batch text through non-Unified streaming parsers. |
| `toolcalling/fixtures-stream-v1/` | Legacy streaming v1 convention. Incremental chunks through non-Unified streaming parsers. |
| `reasoning/fixtures-v1/inputs/` | Legacy reasoning regression corpora. |

## Parser Implementations

Keep the implementation and mode separate when reading or updating fixtures.

The original incremental parser was called `stream-v2` because it was intended to replace batch+jail. Unified later created a second v2 convention. Corpus names now use **streaming v1** for legacy/non-Unified data and **Unified v2** for the ordered reasoning+text+tool protocol. Implementation labels remain `dynamo_v1` and `dynamo_v2`, so `expected.dynamo_v2` inside `fixtures-stream-v1` is correct.

At runtime, `DYN_ENABLE_EXPERIMENTAL_PARSERS_V2` enables both experimental routes; it does not force the non-Unified parser. Dynamo checks Unified first. The exact Qwen3 tool+reasoning parser pair uses Unified when the flag is on. If no Unified pair matches, eligible Qwen3-Coder or DeepSeek V4 requests use the non-Unified incremental parser. Other tool-parsing requests use batch+jail; DeepSeek V4.1 and Muse have separate Unified routes that do not depend on the flag.

- Dynamo v1 is batch only. It writes `expected.dynamo_v1` in `conformance/toolcalling/fixtures-batch-v1/`. This is the current batch baseline, not the upcoming v2 stream parser. **v1 is interim** — it is removed outright once v2 reaches parity.
- Dynamo v2 Rust writes `expected.dynamo_v2` for both the legacy non-Unified stream convention and Unified. The directory or case ID identifies the convention; the expected-output key identifies the implementation.
- vLLM Python is batch and stream. Legacy batch fixtures use `expected.vllm`; v2 fixtures use `expected.vllm_python`. Batch output is vLLM's complete-text parser. Stream output is vLLM's streaming parser.
- vLLM Rust is stream only. It writes `expected.vllm_rust`. vLLM Rust does not expose a separate batch parser here. Complete text is tested by feeding the full text through the Rust streaming parser.
- SGLang Python is batch and stream where SGLang has a detector for that family. Legacy batch fixtures use `expected.sglang`; v2 fixtures use `expected.sglang_python`. Missing detectors are recorded under `unavailable.sglang_python`.

## 1. Verify

Run this when you want to check the local Dynamo parser code, the shared viewer model, and the local authored and current capture YAML.

The verification-only path reads the extracted fixtures and reports mismatches. It does not rewrite fixture YAML. The vLLM and SGLang verification commands run live peer parsers against the extracted expected output, but they do not recapture or update fixtures.

```bash
# Runs regression tests for the shared viewer model, marker semantics, vLLM Rust capture plumbing, and path scrubbing.
python3 -m pytest conformance/utils/tests/test_stream_on_batch.py

# Runs Rust smoke tests (quick check) for the v2 parser implementation.
cargo test --locked -p dynamo-parsers-v2 -- --nocapture

# Runs fixture-based tests against the extracted YAML fixtures for Dynamo Rust parser behavior.
cargo test --locked -p dynamo-conformance-fixtures-v2 -- --nocapture

# Example: check Dynamo v1 batch behavior against extracted `expected.dynamo_v1` blocks in `toolcalling/fixtures-batch-v1/`.
conformance/utils/check.sh dynamo batch

# Example: check the legacy stream corpus with Dynamo v2 stream and batch-on-stream parsers.
conformance/utils/check.sh dynamo stream

# Example: check vLLM Python batch and stream behavior against extracted legacy `expected.vllm` and v2 `expected.vllm_python` blocks.
conformance/utils/check.sh vllm --container vllm-localdev

# Example: check SGLang Python batch and stream behavior against extracted legacy `expected.sglang` and v2 `expected.sglang_python` blocks.
conformance/utils/check.sh sglang --container sglang-localdev

# Fixture-coverage + marker-registration lint: diffs each family's case IDs against
# conformance/case-taxonomy.yaml and checks its `markers:` declaration in
# parser_families.yaml. Default is all families; --family narrows it.
conformance/utils/check.sh coverage
conformance/utils/check.sh coverage --family gemma4

# Formats Rust changes.
cargo fmt

# Checks for whitespace errors and conflict markers.
git diff --check
```

For viewer changes, run the shared JavaScript model tests and the maintained-viewer Selenium tests. Refresh and check the catalog after source metadata changes.

## 2. Update Code Or Fixtures

Edit the parser and independently authored inputs/GOLDEN, then run the latest parser and fixture tests above. Latest YAML is discovered from versionless files under `fixtures-unified-v2/families/<family>/`; normal tests do not sort historical releases or read historical captures. Generate the latest report after the final edit. Contributors do not backfill historical versions or rewrite recorded observations.

Capture maintenance and historical backfills belong to repository administrators. Use the existing transactional publisher described in [the scoped capture plan](../../docs/CONFORMANCE-LATEST-PLAN.md).

## 3. Inspect the Matrix

Generate a separate report directory after fixture or source metadata changes, then serve that output over HTTP:

```bash
python3 conformance/utils/src/viewer_catalog.py --unified-only
python3 conformance/utils/src/viewer_catalog.py --unified-only --check
cargo build --locked --release -p conformance-publisher
target/release/conformance-publish --root . --output ../reports/latest --revision "$(git rev-parse HEAD)"
python3 -m http.server 8000 --directory ../reports/latest
```

Open `http://localhost:8000/?tab=tab-unified`. The browser matrix uses three ordered capture slots and compares each selected capture with GOLDEN. Slots two and three can be None, and duplicate captures are rejected. The default selects the latest available Dynamo v2 Rust Unified capture, followed by None and None. Rows show the union of parser families available in the selected capture versions. Selecting Dynamo v2 Rust shows only families whose selected-release capture records a native `parser_path: unified`. Add `--include-history --history-source /path/to/frontend-parser-captures` to publish historical selections. Mixed selections show the union of the selected families. A family appears once; a selected capture without that family has an unavailable segment. None adds no rows. Capture errors and missing expected results remain visible. Slot selections, themes, Detailed view, column visibility, Vertical, and popup links persist in the URL. Pinned popups can be resized and remain selected while captures reload. Raw YAML links return the original archive-member or document bytes. Generated reports use typed JSON snapshots; original YAML remains available through Raw/download links. YAML remains authoritative, and JSON shards are generated in publication output and are not checked in. Loading errors stay separate from parser results, and Retry revalidates failed resources.

The scoped status gate uses the same JavaScript model:

```bash
conformance/utils/check.sh status --model qwen3 --tab unified
```

Repeat `--model` or `--tab` to select more than one. Status prints each empty or red model/case pair and exits 1 when requested results are rejected. Invalid arguments and resource/model failures exit 2. Node can write the structured report with `node conformance/viewer/status.js --status-path <output.json>`; this does not render HTML or extract fixtures.

The 24 `v1_split_parallel_tool_calls` cases retain source-bound archived regressions. They intentionally overlap the broad purpose of the shared two-call scenario while preserving the archived native syntax, tool schemas, argument values, and recorded Dynamo v1 producer. Replacing them with a generic request would break the binding to the historical observation. The [shared semantic inventory](tests/corpus_semantics.py) compares input, initialization, effective tool schemas, and assembled outcomes without erasing syntax, schemas, or argument keys, types, and values. Both the [canonical uniqueness check](tests/test_unified_taxonomy_covers_corpus.py) and the [retained-profile check](tests/test_gen_unified_v1_cases.py) use that owner. The latter inventories the exact 24-family exception set, records each selected archive source and capture version, and verifies the complete retained records before admitting the exception. These are documented overlaps in scenario purpose, not a claim that every displayed row tests a different behavior.

Packaging, catalog generation, and native publication accept `--workers auto|N`. Use `--workers 1` for the serial reference. Python capture/packaging tools cap automatic selection at eight processes; explicit `N` is bounded by available CPUs, memory, and job count. Native publication uses every process-available CPU, subject to memory admission without changing the configured pool size. Parallel work validates each family in chronological order and writes bundles only in staging. The coordinator publishes the completed generation after every worker succeeds. Progress includes completed jobs, phase time, worker CPU time, and aggregate peak resident memory. The default is `auto`; independent library callers remain serial unless they select a worker policy.

Generate the source catalogs with `python3 conformance/utils/src/viewer_catalog.py` before first publication, viewer tests, or source-status checks. `viewer-catalog.json` and `viewer-unified-catalog.json` are local build inputs and are not versioned. After this initial generation, `--unified-only` can refresh Unified data while reusing manifest-bound historical archive descriptors. CI generates the catalogs once and passes the same generation to each conformance test shard; the source catalogs are not deployed with the public site.

Build the publisher with `cargo build --locked --release -p conformance-publisher`, then run `target/release/conformance-publish --root . --output <site> --revision <full-commit-SHA>`. `--root` defaults to the current directory and `--candidate-tree` records a preview tree. Use a separate output directory for split-checkout reports; `--data-only` is reserved for unsplit development fixtures. The shared envelope producer writes canonical JSON to bounded disk spools for production packing; production never compresses individual snapshots just to decompress and repack them. Both modes use family input documents without an aggregate input or bootstrap payload. Production assets require Node 18 or newer and the locked bundler: run `npm ci --prefix conformance/viewer --ignore-scripts` before publishing. To install build dependencies outside the checkout, copy `conformance/viewer/package.json` and `package-lock.json` into that directory, run `npm ci --prefix <directory> --ignore-scripts`, and set `CONFORMANCE_ESBUILD_ROOT=<directory>`. Production publishes two JavaScript bundles, one compressed runtime catalog, lazy recipe indexes, and shared gzip packs. FastCDC chunks canonical JSON and exact original bytes with 8 KiB minimum, 32 KiB target, and 128 KiB maximum sizes; each decoded pack is at most 1 MiB. The default dependencies are packed before historical captures and originals. Raw lookups use path-hash buckets fetched only for Raw views and downloads, including retained archive-member aliases. The reader checks pack, chunk, reconstructed-content, and resolved-metadata identities before exposing data. Rust validation and browser browsing each cap their decoded-pack cache at 64 MiB; evicted packs are revalidated on reload, and Rust reserves its validation buffers through shared memory admission. Original downloads retain their exact byte hashes and sizes. Published sites contain no standalone YAML, tarballs, Markdown, text sources, development modules, or tests. Missing or corrupt published data fails with Retry. Direct source reads require the explicit checkout development entry point or a source-parity tool configured with `allowSourceFiles`; production disables them. The publisher rejects undeclared files and writes its dependency inventory beside the site. The prior site stays visible during atomic replacement, and a durable journal recovers interrupted transactions. Those JSON shards are disposable publication output and are not checked in. Publication metadata supplies the full revision for immutable GitHub source links. The report header shows the recorded generation time and source revision. Detailed puts the build, publication, and HTTP serving commands in the final Generate this HTML and JSON section. The publication command wraps as text within that section. The legend stays directly below the chart in both orientations. Table selection changes show Loading... before rebuilding the matrix. Case descriptions include the full case ID followed by the same short reference used in the table, for example UNIFIED.1-tool_only (1-1). Popup headings omit the UNIFIED. prefix and show the case ID and short reference before the description, for example 1-tool_only (1-1). A live checkout with missing or stale generation metadata explains that the browser constructs the view from the current catalog; no publication time is inferred from a capture snapshot. Published pages use embedded generation metadata even if its JSON sidecar cannot be fetched. Hovering compact case labels (for example, `1-1` and `2-1`) shows the full case name away from the pointer in both orientations. The legend reads "One segment is an output capture from the parser when compared with golden output" and labels green segments "Matches golden". Mirror branches validate the viewer without deploying it. Configure the repository's Pages source as GitHub Actions and restrict the `github-pages` deployment environment to `main` before enabling deployment.

Use the matrix to inspect captured vLLM Python and Rust behavior. `check.sh vllm` runs the live vLLM Python parser against extracted YAML; it does not run vLLM Rust. vLLM Python vs Rust is a fixture comparison in the legacy stream and batch-on-stream tabs. Refresh the Rust capture through `capture.sh` when its source version changes.

## Matrix Legend

Each matrix cell compares the selected captures with authored ordered GOLDEN for one family and case. The comparison layout shows matches, mismatches, parser errors, unsupported families, unsupported requests, source-unavailable producers, and missing captures as distinct statuses. Popups show the selected capture's output and provenance; the Detailed table uses the same three slots and GOLDEN.

Batch candidates run reasoning extraction followed by tool extraction and retain final fields. Their final reasoning/content/ordered-call projection is compared with original ordered GOLDEN, so lost interleaving or call order remains a mismatch. Existing Unified candidates compare ordered emitted events. Batch popups label rows “Order/chunk not applicable for batch.”

Legacy render builders and their regression fixtures retain the older Dynamo-reference comparison facts and glyphs. They do not add navigation buttons or panels to the generated Unified report.

The retained legacy capture tools use these parser implementations:

| Implementation | Batch parser | Stream parser |
|---|---|---|
| Dynamo | Dynamo v1 batch parser (`dynamo-parsers`); Dynamo v1 jail+batch for the stream-on-batch overlay. | Dynamo v2 stream parser (`dynamo-parsers-v2`). |
| vLLM Python | vLLM Python batch parser. | vLLM Python stream parser. |
| vLLM Rust | vLLM Rust has no separate batch parser; complete parsing delegates through streaming `parse_into(full_output, ...)` and `finish()` in vLLM Rust 0.23.0. | vLLM Rust stream parser. |
| SGLang | SGLang batch parser. | SGLang stream parser. |

## Scripts

Run these from `conformance/utils/`:

| Command | Purpose |
|---|---|
| `python3 src/viewer_catalog.py --unified-only` | Refreshes the resource and metadata catalog after source changes; `--check` rejects a stale catalog. |
| `check.sh status --model <family> --tab <tab>` | Reads source resources through the shared Node model and fails with the exact empty/red model-case pairs. |
| `check.sh` | Runs Dynamo, vLLM Python, and SGLang checks against staged fixtures. |
| `python3 src/capture_dynamo_versions.py --help` | Executes the frozen Dynamo release inventory against Unified family jobs. |
| `capture.sh` | Retained entry point for explicit historical archive maintenance. |

Python tooling under `src/` owns capture, packaging, storage validation, source metadata, and catalog indexing. The capture chain is `capture_cli.py` / `capture_driver.py` / `capture.py` / `capture_vllm_rust.py`; fixture builders include `build_stream_fixtures.py` / `fill_streamv1.py` / `gen_harmony_text_fixtures.py`, and validators include `validate.py` / `validate_fixtures.py`. Use `capture.sh` and `check.sh` for operator runs; invoke these modules directly when developing the harness. Shared ES modules under `conformance/viewer/` own lossless parsing, history resolution, comparison, tab models, and Node status reports. The maintained view/controller and colorizer live in `src/assets/`.

**Viewer architecture.** Both modes use the Rust publisher, shared comparison model, validated transport, and Unified viewer. Latest HTML embeds its current data, styles, scripts, and worker and works alone over `file://` or HTTP. History reports carry their own on-demand packs and original downloads. Late responses cannot repaint a newer selection, and popups load lazily. Source-reader development tests stage a private logical YAML view; the split checkout itself is not a versioned history tree.

## Local current fixtures and bundled history

Local authored inputs and versionless latest capture YAML serve current parser tests. Selection headers replace the shared current index. Existing versioned v2 files and legacy archives remain unchanged at their current paths; their removal or relocation is deferred. Use `--include-history` for retained historical fixture extraction, or add `--history-source /path/to/verified-history` for an alternate dataset. In-place historical archival is read-only in this change; publication and backfill support for that retained layout is deferred to a later PR.

### Run conformance tests (fixtures extract transparently)

Extraction is cached locally; subsequent runs are instant. No token, no network.

```bash
python3 conformance/utils/src/extract_fixtures.py
conformance/utils/check.sh dynamo all        # batch + stream + batch-via-stream
conformance/utils/check.sh vllm --container <name>
conformance/utils/check.sh sglang --container <name>
```

Fixtures materialize into immutable directories under `~/.cache/dynamo/conformance-fixtures/` (or `$XDG_CACHE_HOME/dynamo/conformance-fixtures/`). Current cache identity includes capture bytes, both Unified and legacy authored inputs, and materialization reader sources. `--full-refresh` creates a new generation and later normal runs select it. Historical extraction retains the original manifest-based generation and compatibility-link behavior.

To see what snapshot is pinned and whether it is already cached:

```bash
python3 conformance/utils/src/extract_fixtures.py --info
```

### Add or update authored Unified fixtures

Register the family, author applicable scenarios and independent GOLDEN, regenerate authored YAML, and run latest tests. Generate the latest report with `conformance-publish --root . --output <site> --revision <full-commit-SHA>`. Add `--include-history` only for historical selection/comparison; `--history-source` selects alternate historical evidence. The local conversion cache defaults to `build/conformance-cache`, or use `--cache-dir <directory>`. No historical backfill is required for this change; see the scoped capture plan.

## Notes

Peer parser versions for vLLM Python and SGLang are pinned in `src/pyproject.stub.toml`, the source of truth for peer capture and validation versions. Captures record their concrete versions under `captured_with`; vLLM Rust is captured from a local source checkout. The catalog indexes the recorded versions for viewer selection.

Capture adapters use an ephemeral stage tree created by `src/_common.sh` to provide the historical Dynamo repository layout.
