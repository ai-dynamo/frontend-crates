# conformance/toolcalling/fixtures-stream-v2

Per-chunk streaming fixtures for the `TC stream (v2)` conformance tab. These are frontend-crate-owned v2 overlays; `render_table_v2.sh` stages them together with the Dynamo-synced `conformance/toolcalling/fixtures-batch-v1/` batch corpus when building the HTML matrix.

## Why A Separate Overlay Exists

The Dynamo-synced `conformance/toolcalling/fixtures-batch-v1/` corpus is batch-first v1 data. Streaming is different: vLLM Python, vLLM Rust, SGLang Python, and Dynamo Rust stream parsers emit per-chunk deltas, and those deltas can differ even when the final assembled call is the same. Streaming evidence lives here, not in the synced v1 corpus.

Complete batch text fed through streaming parsers lives in `conformance/toolcalling/fixtures-batch-on-stream-v2/`. Use both directories when adding a v2 streaming parser: stream fixtures check chunk behavior, and batch-on-stream fixtures check whether the streaming parser reconstructs the batch result.

## Layout (versioned, no unversioned baseline)

This corpus is versioned exactly like the batch corpus (`conformance/toolcalling/fixtures-batch-v1/`): there is **no** unversioned "baseline"/anchor dir. The baseline is whichever version is lowest, per impl, found dynamically at resolve time.

```
fixtures-stream-v2/
  inputs/<family>/TOOLCALLING.streamv2.*.yaml        # shared per-chunk delta_text
                                                     # (+ delta_token_ids/finish_reason,
                                                     #  tools, description) — NO expected
  <impl>-<version>/<family>/TOOLCALLING.streamv2.*.yaml
                                                     # that impl's per-chunk expected
                                                     # (+ normal_text); lowest version =
                                                     # full anchor, higher = changed-only
```

Current version dirs: `dynamo_v2-0.1.11` (Dynamo v2 stream parser), `dynamo_v1-3.0.0` (Dynamo v1 batch parser run against stream data via the streaming jail — captured by `capture_dynamo_jail_stream.py` / the `record_dynamo_jail_stream` bin), `vllm_rust-0.23.0`, `vllm_python-0.23.0` + `vllm_python-0.24.0`, `sglang_python-0.5.12.post1` + `sglang_python-0.5.14`. So the stream tab compares the v1 batch parser (jailed), the v2 stream parser, and the peer stream parsers on the same chunk inputs.

`resolve_stream_fixtures.py` copies `inputs/` and folds each impl's version dirs (ascending, up to the selected version) back into the shared chunks — the stream analogue of `resolve_fixtures.py`. Every impl is included at its lowest version by default; `--select <impl>-<version>` bumps a specific impl higher. Readers see the assembled flat tree.

### Fixture Schema

**Shared input** (`inputs/<family>/…`) — no expected, no captured_with:

```yaml
family: harmony
mode: streamv2
cases:
  TOOLCALLING.streamv2.1.a:
    tools: [...]
    chunks:
    - {delta_text: '<|message|>', delta_token_ids: [200008]}
    - {delta_text: '{"location":"NYC"}<|call|>'}
```

**Per-impl expected** (`<impl>-<version>/<family>/…`) — one impl, chunk-aligned to the shared inputs, plus `captured_with`:

```yaml
family: harmony
mode: streamv2
captured_with: {vllm_python: '0.23.0'}
cases:
  TOOLCALLING.streamv2.1.a:
    chunks:
    - {expected: []}
    - expected:
      - {index: 0, name: get_weather, arguments: '{"location":"NYC"}'}
```

A delta is `{index, name?, arguments?}`. The v2 Rust core delta does not include parser-minted IDs; serving adapters add IDs outside the parser. The assembled call is derived by concatenating each index's name and argument fragments. Cross-implementation comparison happens at the assembled level.

## Parser Keys

- `expected.dynamo_v2` is Dynamo parser v2 output; `expected.dynamo_v1` (inside `dynamo_v1-<ver>/`) is the v1 jail reference.
- `expected.vllm_rust` is vLLM Rust stream parser output captured from the checked-out `vllm-tool-parser` crate. There is no separate `V_rb`.
- `expected.vllm_python` is vLLM Python stream parser output from the pinned Python package.
- `expected.sglang_python` is SGLang Python stream parser output from the pinned Python package.
- `unavailable.<impl>` or `expected.<impl>.unavailable` records a parser that does not exist, cannot run, or failed before output was available.
- `expected.<impl>.error` records a parser exception that is expected for the case.

Every parser-failure marker in the generated HTML must have the exact error text in YAML. Do not rely on renderer inference for v2 parser failures.

## Families

- `harmony/` uses the gpt-oss Harmony parser through the token-id path. Label: `gpt-oss (harmony, token-id)`.
- `harmony_text/` uses the same parser through the text path. Label: `gpt-oss (harmony, text)`. The text path re-tokenizes a held suffix so character-split Harmony markers can settle before token commit.
- Other family directories hold peer stream captures and Dynamo TODOs until their Dynamo parser v2 implementation lands.

Harmony is only the first example. DS4 and later streaming families should use the same schema, capture flow, and validation flow.

## Capture Flow

Use `conformance/utils/capture.sh` for new captures:

```bash
# Example: capture one Dynamo v2 Rust stream fixture into JSON for inspection.
conformance/utils/capture.sh dynamo-stream \
  --fixture conformance/toolcalling/fixtures-stream-v2/inputs/harmony/TOOLCALLING.streamv2.1.yaml \
  --output /tmp/dynamo_stream.json

# Example: capture all stream behavior and refresh `fixtures-stream-v2/`.
conformance/utils/capture.sh stream \
  --vllm-container vllm-localdev \
  --sglang-container sglang-localdev \
  --vllm-rust-source ~/dynamo/vllm-0.23.0
```

Use `capture.sh stream` for all-family captures; do not add a second wrapper for the same workflow.

## Conformance Tests

`conformance/tests/parity_toolcalling_stream.rs` drives Dynamo parser v2 over stream fixtures, checks per-chunk output, and checks the assembled result. `conformance/tests/parity_toolcalling_batch_via_stream.rs` checks complete batch text through the same streaming parsers using `fixtures-batch-on-stream-v2/`.

```bash
cargo test --locked -p dynamo-conformance-fixtures-v2 -- --nocapture
python3 -m pytest conformance/utils/tests/test_stream_on_batch.py
conformance/utils/render_table_v2.sh
```
