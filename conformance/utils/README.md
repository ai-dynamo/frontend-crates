# Conformance utilities

Unified v2 is the active conformance workflow. Deprecated toolcalling batch, stream, batch-on-stream, and Reasoning runners, generators, renderers, and tests are retired. Existing archives and capture YAML remain unchanged. Dynamo v1 observations captured against Unified requests remain selectable comparisons; they do not reactivate the deprecated test corpus.

Latest YAML lives at `conformance/fixtures-unified-v2/families/<family>/<implementation>.yaml`. Each file contains complete records and sealed request bindings, with producer provenance separate from selection metadata. Authored inputs/GOLDEN live beside the captures in `inputs_and_golden.yaml`. Latest generation and parser tests read those inputs and versionless snapshots without historical reads or network access. Backfill, versioned-v2 removal, historical moves, and compression remain deferred; see [the scope and acceptance plan](../../docs/CONFORMANCE-LATEST-PLAN.md).

## Check changes

Run commands from the repository root:

```bash
cargo test --locked -p dynamo-parsers-v2 --lib
conformance/utils/check.sh dynamo
node --test conformance/viewer/*.test.js
python3 -m pytest conformance/utils/tests
```

The six Node suites cover values, data/transport, resources/request bindings, model/comparison, presentation, and startup/selection. Python owns capture, storage, publication, and real Chromium integration. Its shared Node bridge runs each suite once. Chromium checks selection, comparison, reload, phase display, downloads, and offline behavior; pure model assertions stay in Node.

Structural-tag cases require the pinned CPU verifier from `structural-tag-requirements.in`. Set `CONFORMANCE_GRAMMAR_PYTHON` to its interpreter when it differs from `python3`. The CI shard commands remain in `check.sh ci`; collection receipts reject missing or duplicate assignments.

## Generate and inspect the report

Follow [the report guide](../README.md#generate-the-latest-or-history-report). Install the locked viewer build dependency and build the publisher:

```bash
npm ci --prefix conformance/viewer --ignore-scripts
cargo build --locked --release -p conformance-publisher
target/release/conformance-publish --root . --output ../reports/latest --revision "$(git rev-parse HEAD)"
target/release/conformance-publish --root . --output ../reports/history --revision "$(git rev-parse HEAD)" --include-history
python3 -m http.server 8000 --directory ../reports/latest
```

Default output contains exactly one self-contained `index.html`. It embeds latest selections, styles, scripts, worker code, and original latest downloads. Full history is a user option to add JSON sidecars beside byte-identical HTML. The browser automatically discovers a matching history sidecar over HTTP; a missing sidecar leaves latest usable. Local-file viewing makes zero external reads.

The three capture slots compare selected results with independently authored GOLDEN. None adds no rows; missing measurements remain explicit. Delayed reads cannot repaint a newer selection. Raw views and downloads preserve original bytes, and Retry revalidates failed resources. Selected captures, theme, detailed view, orientation, and popup links persist in the URL.

Refresh the raw-checkout viewer catalog after source changes:

```bash
python3 conformance/utils/src/viewer_catalog.py
python3 conformance/utils/src/viewer_catalog.py --check
conformance/utils/check.sh status --model qwen3 --tab unified
```

Source catalogs and publication JSON are generated inputs and stay outside Git. The checkout viewer uses the development YAML reader; production excludes that reader and its dependency. Status reads the same model without rendering HTML. Empty/red selected results exit 1; invalid arguments or resource failures exit 2. `node conformance/viewer/qualification.js --help` lists status, snapshot, model, and marker qualification commands.

## Capture and retained evidence

Contributors edit parser code and independently authored inputs/GOLDEN, run latest tests, then generate and inspect the report. Recorded observations and historical backfills require administrator approval. A changed authored request never rewrites an older observation's request binding.

Unified capture entrypoints are `src/capture_dynamo_versions.py`, `src/capture_peer_unified_versions.py`, and the engine-specific Unified adapters. Run their `--help` before use. Batch, stream, and jail-buffer identities remain distinct. Peer exceptions, unsupported APIs, missing source, and missing evidence remain explicit outcomes. The shared capture-stimulus owner validates request membership and normalized results; the peer adapter owns published-package compatibility.

Historical interpretation remains independent of publication. The migration owner retains distinct source-layout and snapshot conversions. Read-only archive resolution preserves sparse history, provenance, and exact source downloads. The retained Unified split scenarios keep their original IDs and frozen observations; the old archive-import writer is retired.

## Source ownership and caching

Seven runtime owners cover application startup, exact values, data transport/cache, model/comparison, resources/request binding, presentation, and the worker. The DOM controller, detail renderer, and syntax colorizer stay separate in `src/assets/`. Three development-only modules provide checkout startup, YAML parsing, and its worker. `build.mjs` and `qualification.js` own build and Node qualification.

The publisher reuses a complete publication before staging. Otherwise it reuses validated family fragments and rebuilds envelopes and packs. `--cache-dir` selects the cache; the local default is `build/conformance-cache`. Warm conversion reuse, independent family invalidation, historical-only isolation, and corrupt-cache recovery are qualified separately. Renderer changes may rebuild presentation without reconverting captures. Generated sites and cache contents remain outside Git.

The stdlib-only version helper, sparse-checkout CI verifier, typed-value codec, capture transaction owner, and independent publication verifier retain separate boundaries. Production validates hashes, sizes, chunks, and reconstructed resources before displaying or downloading them. Historical source files and parser execution environments are never folded into the browser runtime.
