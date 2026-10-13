# Conformance/Utils

This directory contains parser checks, fixture tooling, and the browser viewer's catalog builder. Serve the checkout over HTTP and open the root `index.html` to inspect active fixtures. `viewer_catalog.py` refreshes or checks the resource index; it does not extract captures or precompute comparison cells. The frozen #385 Python oracle lives outside the production tree for migration comparisons.

For Unified capture publication, follow [the plain-version YAML contract](../README.md#unified-storage-contract-plain-versioned-yaml-only). This migration applies only to Unified. Keep the existing archive storage for the older tool-calling stream, batch-on-stream, batch, and reasoning tests.

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

Tool-calling and reasoning fixture YAMLs live in git-lfs tarball shards under `conformance/fixtures/`. Unified fixtures live as reviewable YAML under `conformance/fixtures-unified-v2/`. `extract_fixtures.py` materializes both stores into one compatibility tree for capture tooling and native Rust tests. The browser viewer reads the active archives and Unified YAML directly. `conformance/utils/src/parser_families.yaml` is parser configuration, not a fixture.

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

Run this when you want to check the local Dynamo parser code, the shared viewer model, and the fixture YAMLs from the in-repo store.

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

Change parser code under `parsers/v2/` when Dynamo behavior is wrong. When fixture output changes or a new case is added, capture locally then rebuild + commit a new snapshot via `package_fixtures.py`.

### Capture Parser Behavior Into Fixtures

Use the same pattern for capture commands: `conformance/utils/capture.sh <target> ...`. The `stream` and `batch-on-stream` targets capture v2 fixture YAMLs locally. The `dynamo-stream`, `dynamo-batch-on-stream`, and `token-ids` targets capture local Dynamo Rust behavior or token IDs. After capturing, rebuild + commit the shard store with `package_fixtures.py`.

`capture.sh` is not the v1 batch rewrite tool for the Dynamo `expected.dynamo_v1` blocks — those are verified in Section 1 against extracted fixtures; update the YAMLs locally and re-package when the expected batch output changes.

Peer columns, by contrast, are re-captured as changed-only version overlays against a new engine version. One shared tool, `capture_peer_versions.py --corpus {batch,stream,reasoning} --impl {vllm_python,sglang_python,vllm_rust}`, captures every corpus x engine (it replaces the per-corpus `capture_streamv1_versions.py`; `recapture_peer.py` still exists separately). Omit `--impl` to capture ALL engines valid for the corpus. A full refresh runs one command per corpus so every tab gets the new version (see [`../README.md`](../README.md#fixture-workflows) workflow 1 for the ordered runbook):
- `--corpus stream` — vLLM Python + SGLang Python stream parsers over the resolved streamv1 anchors → changed-only full-case dir `fixtures-stream-v1/<impl>-<ver>/` (written straight to the top level; `package_fixtures.py` shards it directly); `--impl vllm_rust` drives the cargo probe over the `inputs/` cases → full-chunk changed-case dir `fixtures-stream-v1/vllm_rust-<ver>/` (needs `VLLM_RUST_SOURCE` / `--vllm-rust-source`, see below).
- `--corpus batch` — vLLM Python / SGLang Python non-streaming `extract_tool_calls` over the `fixtures-batch-v1/inputs/` seeds → `fixtures-batch-v1/<impl>-<ver>/` (legacy batch corpus).
- `--corpus reasoning` — vLLM Python / SGLang Python reasoning parser over the `reasoning/fixtures-v1/inputs/` seeds → `reasoning/fixtures-v1/<impl>-<ver>/` (Reasoning tab; this tab is multi-version — both old and new versions must render).
- `recapture_batch_on_stream.py` — refreshes the single-snapshot batch-on-stream tree in place (preserves the `vllm_rust`/`dynamo_v2` blocks); the vLLM Rust batch-on-stream block is refreshed by the full `capture.sh batch-on-stream` flow.

All read the version live from the container's `vllm.__version__` / `sglang.__version__`, write only cases that diverge from the lowest-version anchor, never touch existing version dirs (append-only), and skip (never fabricate) cases the parser can't run in-container. Run them against `vllm-localdev` / `sglang-localdev`, then `package_fixtures.py`.

Harmony fixture paths below are examples only. Harmony is not the intended scope limit. As DS4 and the other v2 stream parsers land, use the same commands with those fixture paths and families.

```bash
# Example: capture one Dynamo v2 Rust stream fixture into JSON for manual fixture editing.
conformance/utils/capture.sh dynamo-stream \
  --fixture conformance/toolcalling/fixtures-stream-v1/inputs/harmony/TOOLCALLING.streamv1.1.yaml \
  --output /tmp/dynamo_stream.json

# Example: capture all vLLM Python, vLLM Rust, and SGLang Python stream behavior, then refresh `fixtures-stream-v1/`.
conformance/utils/capture.sh stream \
  --vllm-container vllm-localdev \
  --sglang-container sglang-localdev \
  --vllm-rust-source ~/dev/vllm-0.23.0

# Example: capture all Dynamo v2 Rust, vLLM Python, vLLM Rust, and SGLang Python batch-on-stream behavior, then refresh `fixtures-batch-on-stream-v1/`.
conformance/utils/capture.sh batch-on-stream \
  --vllm-container vllm-localdev \
  --sglang-container sglang-localdev \
  --vllm-rust-source ~/dev/vllm-0.23.0 \
  --capture-dynamo-rust-json /tmp/dynamo_batch_on_stream.json

# Example: capture one Dynamo v2 Rust batch-on-stream JSON file without refreshing fixture YAML.
conformance/utils/capture.sh dynamo-batch-on-stream \
  --output /tmp/dynamo_batch_on_stream.json

# Example: capture token IDs after `delta_text` changes in supported token-based stream fixtures.
conformance/utils/capture.sh token-ids
```

The `stream` and `batch-on-stream` targets update YAML. The `dynamo-stream`, `dynamo-batch-on-stream`, and `token-ids` targets capture local Dynamo Rust behavior or token IDs used by the fixture update.

### Notes on setting up vLLM Rust source before capturing

Use this before capture commands that refresh `expected.vllm_rust`.

```bash
# Download the vLLM source tree and check out the target tag (use ~/dev, not ~/dynamo).
git clone https://github.com/vllm-project/vllm.git ~/dev/vllm-<ver>
git -C ~/dev/vllm-<ver> checkout v<ver>          # e.g. v0.25.1

# Confirm the Rust parser crate exists. The crate + path moved in vLLM 0.25:
#   >= 0.25:  crate `vllm-parser`      at rust/src/parser/Cargo.toml
#   <  0.25:  crate `vllm-tool-parser` at rust/src/tool-parser/Cargo.toml
test -f ~/dev/vllm-<ver>/rust/src/parser/Cargo.toml   # 0.25.x layout

# Makes capture scripts pick up this checkout.
export VLLM_RUST_SOURCE=~/dev/vllm-<ver>

# Shows the source version that will be stamped into YAML.
git -C "$VLLM_RUST_SOURCE" describe --tags --exact-match
git -C "$VLLM_RUST_SOURCE" rev-parse HEAD
```

The crate build may need a sibling crate materialized if the checkout is sparse (0.25.x's `vllm-parser` depends on `vllm-tokenizer`: `git -C "$VLLM_RUST_SOURCE" sparse-checkout add rust/src/tokenizer`). Parsers that moved to the native `unified::` interface in a release (e.g. gemma4 in 0.25) have no `tool::ToolParser` and are recorded unavailable — expected, not a capture failure.

`capture.sh` routes vLLM Rust capture jobs through `src/capture_cli.py` and `src/capture_driver.py` to `src/capture_vllm_rust.py`. The host probe builds a temporary Rust binary against the selected source checkout, then feeds the fixture cases through its streaming parser. Supply the checkout with `--vllm-rust-source` or `VLLM_RUST_SOURCE`; the `stream` and `batch-on-stream` capture targets refresh `expected.vllm_rust`.

The local checkout path is not written to fixture YAML or displayed in the matrix. Fixtures record only the vLLM tag and commit under `captured_with.vllm_rust`.

## 3. Inspect the Matrix

Refresh the catalog after fixture or source metadata changes, then serve the checkout over HTTP:

```bash
python3 conformance/utils/src/viewer_catalog.py
python3 conformance/utils/src/viewer_catalog.py --check
python3 -m http.server 8000
```

Open `http://localhost:8000/?tab=tab-unified`. The maintained matrix loads active inputs and selected parser histories directly. Reference, multiple Compare selections, Detailed, column visibility, Transpose, and per-tab URL state use the existing matrix controls. An explicitly empty Reference survives reload. Themes apply to the matrix and popups. Touch-pinned popups can be resized; selected data updates preserve their owner and size. Raw YAML links show and download original archive-member or document bytes. Loading and resource errors are separate from recorded parser results; Retry revalidates failed resources.

The scoped status gate uses the same JavaScript model:

```bash
conformance/utils/check.sh status --model qwen3 --tab unified
```

Repeat `--model` or `--tab` to select more than one. Status prints each empty or red model/case pair and exits 1 when requested results are rejected. Invalid arguments and resource/model failures exit 2. Node can write the structured report with `node conformance/viewer/status.js --status-path <output.json>`; this does not render HTML or extract fixtures.

The Pages workflow copies the maintained page, assets, catalog, active archives, Unified YAML, and referenced docs. Publication metadata supplies the full revision for immutable GitHub source links. Mirror branches validate the viewer without deploying it. Configure the repository's Pages source as GitHub Actions and restrict the `github-pages` deployment environment to `main` before enabling deployment.

Use the matrix to inspect captured vLLM Python and Rust behavior. `check.sh vllm` runs the live vLLM Python parser against extracted YAML; it does not run vLLM Rust. vLLM Python vs Rust is a fixture comparison in the legacy stream and batch-on-stream tabs. Refresh the Rust capture through `capture.sh` when its source version changes.

### UnifiedParser conversion collection

When a model family is converted to `UnifiedParser`, collecting its current fixture capture is required in the same change. Set the shipping `parsers/v2` version first, then run:

```bash
python3 conformance/utils/src/gen_unified_golden.py
cargo test --locked -p dynamo-conformance-fixtures-v2 --test unified_render -- --nocapture
python3 conformance/utils/src/explode_unified_fixtures.py
python3 conformance/utils/src/package_fixtures.py
python3 conformance/utils/src/extract_fixtures.py --full-refresh
python3 conformance/utils/src/viewer_catalog.py
python3 conformance/utils/src/viewer_catalog.py --check
conformance/utils/check.sh status --model <family> --tab unified
```

This updates the affected family input/golden document and sparse implementation-version captures under `conformance/fixtures-unified-v2/families/<family>/`, refreshes the manifest pin, and refreshes the viewer catalog. Reload the maintained viewer to inspect the current capture. A family conversion is unfinished until the scoped status command reports zero red and zero empty current Dynamo cells.

Run `bash conformance/utils/regenerate_unified.sh` when changing Unified parser code or corpus inputs. It performs the live capture, packaging, native fixture preparation, catalog refresh, case-ID equality checks, and consuming tests. Inspect recorded red or empty cells in the maintained viewer when the gate fails.

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
| `python3 src/viewer_catalog.py` | Refreshes the resource and metadata catalog after source changes; `--check` rejects a stale catalog. |
| `check.sh status --model <family> --tab <tab>` | Reads source resources through the shared Node model and fails with the exact empty/red model-case pairs. |
| `check.sh` | Runs Dynamo, vLLM Python, and SGLang checks against staged fixtures. |
| `capture.sh` | Consistent entry point for capturing parser behavior and refreshing v2 fixtures. |

Python tooling under `src/` owns capture, packaging, storage validation, source metadata, and catalog indexing. The capture chain is `capture_cli.py` / `capture_driver.py` / `capture.py` / `capture_vllm_rust.py`; fixture builders include `build_stream_fixtures.py` / `fill_streamv1.py` / `gen_harmony_text_fixtures.py`, and validators include `validate.py` / `validate_fixtures.py`. Use `capture.sh` and `check.sh` for operator runs; invoke these modules directly when developing the harness. Shared ES modules under `conformance/viewer/` own lossless parsing, history resolution, comparison, tab models, and Node status reports. The maintained view/controller and colorizer live in `src/assets/`.

**Viewer architecture.** The root `index.html` loads the catalog, then fetches inputs and selected Reference/Compare histories for the active tab. Validated bytes are cached by content hash, and YAML members are parsed on demand in a worker. Browser matrices and Node status checks use the same records and comparison model. Late responses cannot repaint a newer selection. Popups load lazily. Serve the checkout over HTTP; `file://` cannot load its resources. Model assertions run against the JavaScript engine, and Selenium tests exercise the maintained viewer.

## Fixture Store (git-lfs)

Fixture shard tarballs live in the repo at `conformance/fixtures/`, tracked via git-lfs (`.gitattributes`). The manifest (`conformance/fixtures-manifest.json`) pins the current snapshot and the sha256 of every shard. On a fresh clone, make sure the LFS objects are present: `git lfs install && git lfs pull`.

### Run conformance tests (fixtures extract transparently)

Extraction is cached locally; subsequent runs are instant. No token, no network.

```bash
python3 conformance/utils/src/extract_fixtures.py
conformance/utils/check.sh dynamo all        # batch + stream + batch-via-stream
conformance/utils/check.sh vllm --container <name>
conformance/utils/check.sh sglang --container <name>
```

Fixtures always extract to `~/.cache/dynamo/conformance-fixtures/` (or `$XDG_CACHE_HOME/dynamo/conformance-fixtures/` if set). Stable symlinks `toolcalling/` and `reasoning/` point at the current snapshot subdir. Cache hit = the script compares the manifest pin against the local state file and exits immediately if they match.

To see what snapshot is pinned and whether it is already cached:

```bash
python3 conformance/utils/src/extract_fixtures.py --info
```

### Update existing fixtures (re-capture after a parser version bump)

For Unified corpus edits, regenerate after every source change: run the generator, explode the loose capture, update the family input/golden and capture YAML files, refresh the manifest pin, refresh and check the viewer catalog, and run the consuming tests before making the next review claim. Repeat the chain after the final edit and reload the browser viewer. Verify that every resolved current capture contains the same case-ID set as the current generator.

After re-capturing YAML locally with `capture.sh`, rebuild the store and commit it together with the manifest:

```bash
# 1. Re-capture (see capture.sh commands above)

# 2. Rebuild the shard store + manifest
python3 conformance/utils/src/package_fixtures.py

# 3. Commit both stores + manifest pin
git add conformance/fixtures conformance/fixtures-unified-v2 conformance/fixtures-manifest.json
git commit -s -m "fixtures: snapshot <stamp printed by the script>"
```

The script builds deterministic per-version tarball shards for tool-calling and reasoning fixtures, updates the reviewable Unified YAML store, and writes the manifest that pins both stores.

### Add new fixtures (new SGLang / vLLM / Dynamo family)

Add YAML files locally under the appropriate fixture tree, then publish a new snapshot exactly as above. The new family appears as a new subdirectory in the tarball; warm-path downloads on other machines will fetch only the shard(s) that changed.

## Notes

Peer parser versions for vLLM Python and SGLang are pinned in `src/pyproject.stub.toml`, the source of truth for peer capture and validation versions. Captures record their concrete versions under `captured_with`; vLLM Rust is captured from a local source checkout. The catalog indexes the recorded versions for viewer selection.

Capture adapters use an ephemeral stage tree created by `src/_common.sh` to provide the historical Dynamo repository layout.
