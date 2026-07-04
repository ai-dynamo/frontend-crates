# Conformance/Utils

This directory is for checking parser behavior and generating an HTML conformance matrix. By default `render_table_v2.sh` writes `conformance/CONFORMANCE.html`, but you can write another file such as `index.html`.

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
| `expected.dynamo_rust` | Dynamo Rust parser output. |
| `expected.vllm_rust` | vLLM Rust parser output captured from a vLLM source checkout. |
| `expected.vllm_python` | vLLM Python parser output captured from the pinned Python package. |
| `expected.sglang_python` | SGLang Python parser output captured from the pinned Python package. |
| `captured_with.*` | Parser version used when output was captured. |
| `unavailable.*` | A parser was not available or is intentionally TODO for this case. |

Fixture locations:

| Path | Used By |
|---|---|
| `conformance/toolcalling/fixtures/` | `TC batch (v1)` tab. Complete model output through batch parsers. |
| `conformance/toolcalling/fixtures-batch-on-stream-v2/` | `TC batch-on-stream (v2)` tab. Complete batch text through streaming parsers. |
| `conformance/toolcalling/fixtures-stream-v2/` | `TC stream (v2)` tab. Incremental chunks through streaming parsers. |
| `conformance/reasoning/fixtures/` | Reasoning parser tabs. |

## Parser Implementations

Keep the implementation and mode separate when reading or updating fixtures.

- Dynamo v1 is batch only. It writes `expected.dynamo` in `conformance/toolcalling/fixtures/`. This is the current batch baseline, not the upcoming v2 stream parser.
- Dynamo v2 Rust is stream and batch-on-stream. It writes `expected.dynamo_rust` in new v2 fixture shapes, though some older example fixtures still use `expected.dynamo`. This is the upcoming Dynamo-owned Rust stream parser. Harmony is only the example wired today; DS4 and the other v2 stream parsers should use the same flow as they land.
- vLLM Python is batch and stream. Legacy batch fixtures use `expected.vllm`; v2 fixtures use `expected.vllm_python`. Batch output is vLLM's complete-text parser. Stream output is vLLM's streaming parser.
- vLLM Rust is stream only. It writes `expected.vllm_rust`. vLLM Rust does not expose a separate batch parser here. Complete text is tested by feeding the full text through the Rust streaming parser.
- SGLang Python is batch and stream where SGLang has a detector for that family. Legacy batch fixtures use `expected.sglang`; v2 fixtures use `expected.sglang_python`. Missing detectors are recorded under `unavailable.sglang_python`.

## 1. Verify

Run this when you want to check the local Dynamo parser code, the conformance table generator (`conformance/utils/`), and the committed YAML fixtures.

The verification-only path reads committed YAML and reports mismatches. It does not rewrite fixture YAML. The vLLM and SGLang verification commands run live peer parsers against committed expected output, but they do not recapture or update fixtures.

```bash
# Runs Python regression tests for table generation, marker semantics, vLLM Rust capture plumbing, and path scrubbing.
python3 -m pytest conformance/utils/tests/test_stream_on_batch.py

# Runs Rust smoke tests (quick check) for the v2 parser implementation.
cargo test --locked -p dynamo-parsers-v2 -- --nocapture

# Runs fixture-based tests against committed YAML fixtures for Dynamo Rust parser behavior.
cargo test --locked -p dynamo-conformance-fixtures-v2 -- --nocapture

# Example: check Dynamo v1 batch behavior against committed `expected.dynamo` blocks in `conformance/toolcalling/fixtures/`.
conformance/utils/check.sh dynamo batch

# Example: check Dynamo v2 stream fixtures and Dynamo v2 batch-on-stream behavior.
conformance/utils/check.sh dynamo stream

# Example: check vLLM Python batch and stream behavior against committed legacy `expected.vllm` and v2 `expected.vllm_python` blocks.
conformance/utils/check.sh vllm --container vllm-localdev

# Example: check SGLang Python batch and stream behavior against committed legacy `expected.sglang` and v2 `expected.sglang_python` blocks.
conformance/utils/check.sh sglang --container sglang-localdev

# Formats Rust changes.
cargo fmt

# Checks for whitespace errors and conflict markers.
git diff --check
```

If you only changed docs or the HTML generator, the Python regression test and `conformance/utils/render_table_v2.sh` are usually enough.

## 2. Update Code Or Fixtures

Change parser code under `parsers/v2/` when Dynamo behavior is wrong. Change fixture YAML when expected output changed, new peer output was captured, or a new case was added.

### Capture Parser Behavior Into Fixtures

Use the same pattern for capture commands: `conformance/utils/capture.sh <target> ...`. The `stream` and `batch-on-stream` targets update v2 fixture YAML. The `dynamo-stream`, `dynamo-batch-on-stream`, and `token-ids` targets capture local Dynamo Rust behavior or token IDs used by fixture updates.

`capture.sh` is not the v1 batch rewrite tool. Dynamo v1 batch, vLLM Python batch, and SGLang Python batch are verified in Section 1 against committed fixtures. Change those YAML files directly when the committed batch expectation is wrong.

Harmony fixture paths below are examples only. Harmony is not the intended scope limit. As DS4 and the other v2 stream parsers land, use the same commands with those fixture paths and families.

```bash
# Example: capture one Dynamo v2 Rust stream fixture into JSON for manual fixture editing.
conformance/utils/capture.sh dynamo-stream \
  --fixture conformance/toolcalling/fixtures-stream-v2/harmony/TOOLCALLING.streamv2.1.yaml \
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
# Downloads the vLLM source tree used for current Rust captures.
git clone https://github.com/vllm-project/vllm.git ~/dynamo/vllm-0.23.0

# Checks out the pinned vLLM version.
git -C ~/dynamo/vllm-0.23.0 checkout v0.23.0

# Confirms the Rust tool-parser crate exists.
test -f ~/dynamo/vllm-0.23.0/rust/src/tool-parser/Cargo.toml

# Makes capture scripts pick up this checkout.
export VLLM_RUST_SOURCE=~/dynamo/vllm-0.23.0

# Shows the source version that will be stamped into YAML.
git -C "$VLLM_RUST_SOURCE" describe --tags --exact-match
git -C "$VLLM_RUST_SOURCE" rev-parse HEAD
```

The local checkout path is not written to YAML or HTML. Fixtures record only the vLLM tag and commit under `captured_with.vllm_rust`.

## 3. Generate Matrix

Run this after updating code or fixture YAML.

```bash
# Generates an HTML matrix at the default example path: `conformance/CONFORMANCE.html`.
conformance/utils/render_table_v2.sh

# Generates the same matrix at a custom path, for example `index.html`.
conformance/utils/render_table_v2.sh --output index.html

# Prints the render command without writing the table.
conformance/utils/render_table_v2.sh --dry-run
```

Open the generated HTML file in a browser. The table is generated from the committed fixture directories plus the staged v2 fixture view built by `render_table_v2.sh`.

Use the generated matrix to inspect vLLM Python vs vLLM Rust behavior. `check.sh vllm` runs the live vLLM Python parser against committed YAML; it does not run vLLM Rust. vLLM Python vs Rust is a committed fixture comparison in the `TC stream (v2)` and `TC batch-on-stream (v2)` tabs.

## Matrix Legend

The matrix has four parser identities:

| Selector | Marker form |
|---|---|
| Dynamo Rust | `D_rs` (Dynamo Rust stream parser), `D_rb` (Dynamo Rust batch parser). |
| vLLM Rust | `V_rs` (vLLM Rust stream parser). There is no `V_rb`; vLLM Rust batch-style complete parsing delegates through streaming `parse_into(full_output, ...)` and `finish()` in vLLM Rust 0.23.0. |
| vLLM Python | `V_ps` (vLLM Python stream parser), `V_pb` (vLLM Python batch parser). |
| SGLang | `S_rs` (SGLang stream parser), `S_rb` (SGLang batch parser). |

HTML markers use real subscripts, for example `D<sub>RS</sub>`, `D<sub>RB</sub>`, `V<sub>PS</sub>`, `V<sub>PB</sub>`, `V<sub>RS</sub>`, `S<sub>RS</sub>`, and `S<sub>RB</sub>`. Non-HTML output uses `D_rs`, `D_rb`, `V_ps`, `V_pb`, `V_rs`, `S_rs`, and `S_rb`.

vLLM shorthand:

| Name | Meaning |
|---|---|
| `V_ps` | vLLM Python stream parser. |
| `V_pb` | vLLM Python batch parser. |
| `V_rs` | vLLM Rust stream parser. |
| `V_rb` | vLLM Rust batch parser does not exist as a separate captured implementation. |

## Scripts

Run these from `conformance/utils/`:

| Command | Purpose |
|---|---|
| `render_table_v2.sh` | Builds `.stage/` and writes the v2 conformance HTML matrix. The default example path is `conformance/CONFORMANCE.html`. |
| `render_table_v1.sh` | Renders the legacy v1 Dynamo parity table into `.stage/`. |
| `check.sh` | Runs Dynamo, vLLM Python, and SGLang checks against staged fixtures. |
| `capture.sh` | Consistent entry point for capturing parser behavior and refreshing v2 fixtures. |

The implementation lives under `src/` — don't run these directly unless you're developing the harness: `_common.sh`, the renderer (`generate_conformance_table.py` + `impls.py` / `markers.py` / `fixtures.py`, `conformance_table.html.j2` + `assets/`), the capture chain (`capture_cli.py` / `capture_driver.py` / `capture.py` / `capture_vllm_rust.py`), the fixture builders (`build_stream_fixtures.py` / `fill_streamv2.py` / `gen_harmony_text_fixtures.py`), the validators (`validate.py` / `validate_fixtures.py`), and the data files (`parser_families.yaml`, `pyproject.stub.toml`). `tests/` and `lib/` stay at the top level because they are Dynamo-sync targets.

## Notes

Peer parser versions for vLLM Python and SGLang are pinned in `src/pyproject.stub.toml` — currently vLLM Python `0.23.0` and SGLang `0.5.12.post1`. This stub is the single source of truth for both the v2 `captured_with` stamps and the v1 batch fixtures (`expected.vllm` / `expected.sglang`), which carry no per-file version stamp; `check.sh vllm|sglang --container` validates the live engine against the committed v1 fixtures and reports it as `pinned (fixtures captured against)`. As of the 0.23.0 bump the live vLLM `0.23.0` and SGLang `0.5.12.post1` batch parsers match all committed v1 cases (519 vLLM, 448 SGLang). vLLM Rust is captured from a local source checkout and recorded in YAML under `captured_with.vllm_rust`.

The scripts build an ephemeral `.stage/` tree because the vendored Dynamo Python table code assumes Dynamo's repo layout. `.stage*/` is gitignored.
