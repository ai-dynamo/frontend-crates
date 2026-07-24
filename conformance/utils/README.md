# Conformance/Utils

This directory is for checking parser behavior and generating an HTML conformance matrix. By default `render_table_v2.sh` writes `conformance/CONFORMANCE_v2.html`, but you can write another file such as `index.html`.

Most work has three steps:

1. Verify parser code, table code, and fixture YAML.
2. Update parser code and/or fixture YAML.
3. Generate the HTML matrix.

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
| `unavailable.*` | A parser was not available or is intentionally TODO for this case. |

Fixture locations:

All fixture YAMLs live in the repo as git-lfs tarball shards under `conformance/fixtures/` and are extracted automatically on first use via `extract_fixtures.py`. Do NOT commit loose fixture YAMLs — rebuild the shard store via `package_fixtures.py` instead (see "Fixture Store" below). `conformance/utils/src/parser_families.yaml` is a parser config file, not a fixture — it stays loose in the repo.

| Store path (inside snapshot) | Used By |
|---|---|
| `toolcalling/fixtures-batch-v1/` | `TC batch (v1)` tab. Complete model output through batch parsers. |
| `toolcalling/fixtures-batch-on-stream-v2/` | `TC batch-on-stream (v2)` tab. Complete batch text through streaming parsers. |
| `toolcalling/fixtures-stream-v2/` | `TC stream (v2)` tab. Incremental chunks through streaming parsers. |
| `reasoning/fixtures-v1/inputs/` | Reasoning parser tabs. |

## Parser Implementations

Keep the implementation and mode separate when reading or updating fixtures.

- Dynamo v1 is batch only. It writes `expected.dynamo_v1` in `conformance/toolcalling/fixtures-batch-v1/`. This is the current batch baseline, not the upcoming v2 stream parser. **v1 is interim** — it is removed outright once v2 reaches parity.
- Dynamo v2 Rust is stream and batch-on-stream. It writes `expected.dynamo_v2` in new v2 fixture shapes. This is the upcoming Dynamo-owned Rust stream parser — **the ultimate implementation (WIP)** that fully replaces v1. Harmony is only the example wired today; DS4 and the other v2 stream parsers should use the same flow as they land.
- vLLM Python is batch and stream. Legacy batch fixtures use `expected.vllm`; v2 fixtures use `expected.vllm_python`. Batch output is vLLM's complete-text parser. Stream output is vLLM's streaming parser.
- vLLM Rust is stream only. It writes `expected.vllm_rust`. vLLM Rust does not expose a separate batch parser here. Complete text is tested by feeding the full text through the Rust streaming parser.
- SGLang Python is batch and stream where SGLang has a detector for that family. Legacy batch fixtures use `expected.sglang`; v2 fixtures use `expected.sglang_python`. Missing detectors are recorded under `unavailable.sglang_python`.

## 1. Verify

Run this when you want to check the local Dynamo parser code, the conformance table generator (`conformance/utils/`), and the fixture YAMLs from the in-repo store.

The verification-only path reads the extracted fixtures and reports mismatches. It does not rewrite fixture YAML. The vLLM and SGLang verification commands run live peer parsers against the extracted expected output, but they do not recapture or update fixtures.

```bash
# Runs Python regression tests for table generation, marker semantics, vLLM Rust capture plumbing, and path scrubbing.
python3 -m pytest conformance/utils/tests/test_stream_on_batch.py

# Runs Rust smoke tests (quick check) for the v2 parser implementation.
cargo test --locked -p dynamo-parsers-v2 -- --nocapture

# Runs fixture-based tests against the extracted YAML fixtures for Dynamo Rust parser behavior.
cargo test --locked -p dynamo-conformance-fixtures-v2 -- --nocapture

# Example: check Dynamo v1 batch behavior against extracted `expected.dynamo_v1` blocks in `toolcalling/fixtures-batch-v1/`.
conformance/utils/check.sh dynamo batch

# Example: check Dynamo v2 stream fixtures and Dynamo v2 batch-on-stream behavior.
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

If you only changed docs or the HTML generator, the Python regression test and `conformance/utils/render_table_v2.sh` are usually enough.

## 2. Update Code Or Fixtures

Change parser code under `parsers/v2/` when Dynamo behavior is wrong. When fixture output changes or a new case is added, capture locally then rebuild + commit a new snapshot via `package_fixtures.py`.

### Capture Parser Behavior Into Fixtures

Use the same pattern for capture commands: `conformance/utils/capture.sh <target> ...`. The `stream` and `batch-on-stream` targets capture v2 fixture YAMLs locally. The `dynamo-stream`, `dynamo-batch-on-stream`, and `token-ids` targets capture local Dynamo Rust behavior or token IDs. After capturing, rebuild + commit the shard store with `package_fixtures.py`.

`capture.sh` is not the v1 batch rewrite tool for the Dynamo `expected.dynamo_v1` blocks — those are verified in Section 1 against extracted fixtures; update the YAMLs locally and re-package when the expected batch output changes.

Peer columns, by contrast, are re-captured as changed-only version overlays against a new engine version. One shared tool, `capture_peer_versions.py --corpus {batch,stream,reasoning} --impl {vllm_python,sglang_python,vllm_rust}`, captures every corpus x engine (the four old per-corpus capturers — `capture_batch_versions.py`, `capture_streamv2_versions.py`, `capture_reasoning_versions.py`, `capture_vllm_rust_versions.py` — are consolidated into it). Omit `--impl` to capture ALL engines valid for the corpus. A full refresh runs one command per corpus so every tab gets the new version (see [`../README.md`](../README.md#fixture-workflows) workflow 1 for the ordered runbook):
- `--corpus stream` — vLLM Python + SGLang Python stream parsers over the resolved streamv2 anchors → changed-only per-chunk overlay `fixtures-stream-v2/overlays/<impl>-<ver>/` (promote to top-level `<impl>-<ver>/` before packaging); `--impl vllm_rust` drives the cargo probe over the `inputs/` cases → full-chunk changed-case dir `fixtures-stream-v2/vllm_rust-<ver>/` (needs `VLLM_RUST_SOURCE` / `--vllm-rust-source`, see below).
- `--corpus batch` — vLLM Python / SGLang Python non-streaming `extract_tool_calls` over the `fixtures-batch-v1/inputs/` seeds → `fixtures-batch-v1/<impl>-<ver>/` (TC batch tab).
- `--corpus reasoning` — vLLM Python / SGLang Python reasoning parser over the `reasoning/fixtures-v1/inputs/` seeds → `reasoning/fixtures-v1/<impl>-<ver>/` (Reasoning tab; this tab is multi-version — both old and new versions must render).
- `recapture_batch_on_stream.py` — refreshes the single-snapshot batch-on-stream tree in place (preserves the `vllm_rust`/`dynamo_v2` blocks); the vLLM Rust batch-on-stream block is refreshed by the full `capture.sh batch-on-stream` flow.

All read the version live from the container's `vllm.__version__` / `sglang.__version__`, write only cases that diverge from the lowest-version anchor, never touch existing version dirs (append-only), and skip (never fabricate) cases the parser can't run in-container. Run them against `vllm-localdev` / `sglang-localdev`, then `package_fixtures.py`.

Harmony fixture paths below are examples only. Harmony is not the intended scope limit. As DS4 and the other v2 stream parsers land, use the same commands with those fixture paths and families.

```bash
# Example: capture one Dynamo v2 Rust stream fixture into JSON for manual fixture editing.
conformance/utils/capture.sh dynamo-stream \
  --fixture conformance/toolcalling/fixtures-stream-v2/inputs/harmony/TOOLCALLING.streamv2.1.yaml \
  --output /tmp/dynamo_stream.json

# Example: capture all vLLM Python, vLLM Rust, and SGLang Python stream behavior, then refresh `fixtures-stream-v2/`.
conformance/utils/capture.sh stream \
  --vllm-container vllm-localdev \
  --sglang-container sglang-localdev \
  --vllm-rust-source ~/dynamo/vllm-0.23.0

# Example: capture all Dynamo v2 Rust, vLLM Python, vLLM Rust, and SGLang Python batch-on-stream behavior, then refresh `fixtures-batch-on-stream-v2/`.
conformance/utils/capture.sh batch-on-stream \
  --vllm-container vllm-localdev \
  --sglang-container sglang-localdev \
  --vllm-rust-source ~/dynamo/vllm-0.23.0 \
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

The local checkout path is not written to YAML or HTML. Fixtures record only the vLLM tag and commit under `captured_with.vllm_rust`.

## 3. Generate Matrix

Run this after updating code or committing new fixture snapshots.

```bash
# Generates an HTML matrix at the default example path: `conformance/CONFORMANCE_v2.html`.
conformance/utils/render_table_v2.sh

# Generates the same matrix at a custom path, for example `index.html`.
conformance/utils/render_table_v2.sh --output index.html

# Prints the render command without writing the table.
conformance/utils/render_table_v2.sh --dry-run
```

Open the generated HTML file in a browser. The table is generated from extracted fixture directories staged by `render_table_v2.sh`.

Use the generated matrix to inspect vLLM Python vs vLLM Rust behavior. `check.sh vllm` runs the live vLLM Python parser against extracted YAML; it does not run vLLM Rust. vLLM Python vs Rust is a fixture comparison in the `TC stream (v2)` and `TC batch-on-stream (v2)` tabs.

## Matrix Legend

Each cell compares the parser implementations for one model family and case. The page carries **structured comparison facts** (`markers.comparison_facts()` in `src/markers.py`) — a per-implementation record of `status` (ok / problem / na), whether it `agrees` with the Dynamo baseline, whether a divergence is `intentional` (documented), whether it `leak`s tool-call markup, and its `error_kind`. The JS view (`assets/conformance_view.js`) turns those facts into the glyphs and popups, and labels every parser with its full descriptive name. There is no shorthand code to memorize.

The implementations shown per cell:

| Implementation | Batch parser | Stream parser |
|---|---|---|
| Dynamo | Dynamo v1 batch parser (`dynamo-parsers`); Dynamo v1 jail+batch for the stream-on-batch overlay. | Dynamo v2 stream parser (`dynamo-parsers-v2`). |
| vLLM Python | vLLM Python batch parser. | vLLM Python stream parser. |
| vLLM Rust | none — vLLM Rust has no separate batch parser; batch-style complete parsing delegates through streaming `parse_into(full_output, ...)` and `finish()` in vLLM Rust 0.23.0. | vLLM Rust stream parser. |
| SGLang | SGLang batch parser. | SGLang stream parser. |

A cell's status glyphs: `=` (every compared parser matches the Dynamo baseline), `↯` (the parser leaks tool-call markup into the visible `normal_text`), `!` (a declared expected-error), `✗` (the parser ran but failed to parse), `·` (Dynamo-only fixture; peers unavailable), `n/a` (case doesn't apply), `—` (no fixture yet). In the Detailed view a cell shows how many selected Compare parsers diverge from the Reference.

## Scripts

Run these from `conformance/utils/`:

| Command | Purpose |
|---|---|
| `render_table_v2.sh` | Builds `.stage/` and writes the v2 conformance HTML matrix. The default example path is `conformance/CONFORMANCE_v2.html`. |
| `render_table_v1.sh` | Renders the legacy v1 Dynamo parity table into `.stage/`. |
| `check.sh` | Runs Dynamo, vLLM Python, and SGLang checks against staged fixtures. |
| `capture.sh` | Consistent entry point for capturing parser behavior and refreshing v2 fixtures. |

The implementation lives under `src/` — don't run these directly unless you're developing the harness: `_common.sh`, the renderer (`generate_conformance_table.py` + `impls.py` / `markers.py` / `fixtures.py` / `model.py`, `conformance_table.html.j2` + `assets/`), the capture chain (`capture_cli.py` / `capture_driver.py` / `capture.py` / `capture_vllm_rust.py`), the fixture builders (`build_stream_fixtures.py` / `fill_streamv2.py` / `gen_harmony_text_fixtures.py`), the validators (`validate.py` / `validate_fixtures.py`), and the data files (`parser_families.yaml`, `pyproject.stub.toml`). `tests/` and `lib/` stay at the top level because they are Dynamo-sync targets.

**Render architecture (DIS-2434).** Python computes ONE documented JSON data model — the whole page (both the v2 conformance table and the v1 parity page) — and the templates inline it as a single `<script type="application/json" id="conformance-model">` blob. A JS view (`assets/conformance_view.js`) parses that blob and builds the tabs, table, cells, compare bar, legend, and popups (popups lazily on hover); `assets/conformance.js` then wires the compare engine, tab switching, column toggles, URL state, and transpose. The comparison/parity SEMANTICS stay in Python (`markers.py`/`impls.py` are the single source of truth) — `model.py` orchestrates them into the schema documented at the top of `model.py`; the view only decides display. The former parser-marker shorthand mini-language (encoded `data-marker-parity-*` attribute strings) is gone: cells carry structured comparison facts and the view renders the glyphs with full descriptive parser labels. `file://` keeps working (the model is inlined, no fetch). Greppability of the rendered `.html` is an explicit non-goal. Structural guards on the model live in `tests/test_model.py`; a few selenium DOM smokes live in `tests/test_browser_*.py`.

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

After re-capturing YAML locally with `capture.sh`, rebuild the store and commit it together with the manifest:

```bash
# 1. Re-capture (see capture.sh commands above)

# 2. Rebuild the shard store + manifest
python3 conformance/utils/src/package_fixtures.py

# 3. Commit the store + manifest pin (shards are LFS-tracked)
git add conformance/fixtures conformance/fixtures-manifest.json
git commit -s -m "fixtures: snapshot <stamp printed by the script>"
```

The script builds deterministic per-version shard tarballs (mtime/uid normalized, so unchanged trees produce byte-identical shards and no git churn), removes stale shards no longer in the set, and writes the new manifest.

### Add new fixtures (new SGLang / vLLM / Dynamo family)

Add YAML files locally under the appropriate fixture tree, then publish a new snapshot exactly as above. The new family appears as a new subdirectory in the tarball; warm-path downloads on other machines will fetch only the shard(s) that changed.

## Notes

Peer parser versions for vLLM Python and SGLang are pinned in `src/pyproject.stub.toml` — currently vLLM Python `0.23.0` and SGLang `0.5.12.post1`. This stub is the single source of truth for both the v2 `captured_with` stamps and the v1 batch fixtures (`expected.vllm` / `expected.sglang`), which carry no per-file version stamp; `check.sh vllm|sglang --container` validates the live engine against the committed v1 fixtures and reports it as `pinned (fixtures captured against)`. As of the 0.23.0 bump the live vLLM `0.23.0` and SGLang `0.5.12.post1` batch parsers match all committed v1 cases (519 vLLM, 448 SGLang). vLLM Rust is captured from a local source checkout and recorded in YAML under `captured_with.vllm_rust`.

The scripts build an ephemeral `.stage/` tree because the vendored Dynamo Python table code assumes Dynamo's repo layout. `.stage*/` is gitignored.
