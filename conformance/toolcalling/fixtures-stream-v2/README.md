# conformance/toolcalling/fixtures-stream-v2

Per-chunk streaming fixtures for the `TC stream (v2)` conformance tab. These are frontend-crate-owned v2 overlays; `render_table.sh` stages them together with the Dynamo-synced `conformance/toolcalling/fixtures/` batch corpus when building the HTML matrix.

## Why A Separate Overlay Exists

The Dynamo-synced `conformance/toolcalling/fixtures/` corpus is batch-first v1 data. Streaming is different: vLLM Python, vLLM Rust, SGLang Python, and Dynamo Rust stream parsers emit per-chunk deltas, and those deltas can differ even when the final assembled call is the same. Streaming evidence lives here, not in the synced v1 corpus.

Complete batch text fed through streaming parsers lives in `conformance/toolcalling/fixtures-batch-on-stream-v2/`. Use both directories when adding a v2 streaming parser: stream fixtures check chunk behavior, and batch-on-stream fixtures check whether the streaming parser reconstructs the batch result.

## Fixture Schema

Each stream fixture records chunks under `cases.<case>.chunks`. Each chunk has input text and optional token IDs, then an `expected` block keyed by implementation:

```yaml
captured_with:
  dynamo_rust: Dynamo parser v2
  vllm_rust: v0.22.0 0b3ba88f165976e77ca5e6a7a3f5bba4562b80af
  vllm_python: 0.22.0
  sglang_python: 0.5.12.post1
cases:
  TOOLCALLING.streamv2.1.a:
    tools: [...]
    chunks:
    - delta_text: '<|message|>'
      delta_token_ids: [200008]
      expected:
        dynamo_rust: []
        vllm_rust: []
        vllm_python: []
        sglang_python: []
    - delta_text: '{"location":"NYC"}<|call|>'
      expected:
        dynamo_rust:
        - {index: 0, name: get_weather, arguments: '{"location":"NYC"}'}
        vllm_rust:
        - {index: 0, name: get_weather, arguments: '{"location":"NYC"}'}
        vllm_python:
        - {index: 0, name: get_weather, arguments: '{"location":"NYC"}'}
        sglang_python:
        - {index: 0, name: get_weather, arguments: '{"location":"NYC"}'}
```

A delta is `{index, name?, arguments?}`. The v2 Rust core delta does not include parser-minted IDs; serving adapters add IDs outside the parser. The assembled call is derived by concatenating each index's name and argument fragments. Cross-implementation comparison happens at the assembled level.

## Parser Keys

- `expected.dynamo_rust` is Dynamo parser v2 output.
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
  --fixture conformance/toolcalling/fixtures-stream-v2/harmony/TOOLCALLING.streamv2.1.yaml \
  --output /tmp/dynamo_stream.json

# Example: capture all stream behavior and refresh `fixtures-stream-v2/`.
conformance/utils/capture.sh stream \
  --vllm-container vllm-localdev \
  --sglang-container sglang-localdev \
  --vllm-rust-source ~/dynamo/vllm-0.22.0
```

Use `capture.sh stream` for all-family captures; do not add a second wrapper for the same workflow.

## Conformance Tests

`conformance/tests/parity_toolcalling_stream.rs` drives Dynamo parser v2 over stream fixtures, checks per-chunk output, and checks the assembled result. `conformance/tests/parity_toolcalling_batch_via_stream.rs` checks complete batch text through the same streaming parsers using `fixtures-batch-on-stream-v2/`.

```bash
cargo test --locked -p dynamo-conformance-fixtures-v2 -- --nocapture
python3 -m pytest conformance/utils/tests/test_stream_on_batch.py
conformance/utils/render_table.sh
```
