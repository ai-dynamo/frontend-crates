# conformance/toolcalling/fixtures-stream-v2

Per-chunk streaming fixtures for the TC stream conformance tab. These are the frontend-crate-owned overlay; `render_table_v2.sh` overlays them on top of the Dynamo-synced `conformance/toolcalling/fixtures/` batch corpus when building `.stage-v2/`.

## Why a separate overlay

The Dynamo-synced `conformance/` corpus stays unchanged from Dynamo. Streaming is different: vLLM/SGLang stream on text, the Dynamo parser v2 Harmony parser has token-id and text-entry paths, and each implementation emits different per-chunk deltas by design. Recording that lives here, not in the synced corpus.

## Per-chunk format

Each chunk carries its input and, under `expected`, the tool-call deltas each implementation emits at that chunk boundary:

```yaml
captured_with:            # engine versions the vLLM/SGLang data was captured against
  sglang: 0.5.12.post1
cases:
  TOOLCALLING.stream.1.b:
    tools: [...]
    unavailable:          # impls that can't run this case at all
      vllm: <reason>
    chunks:
    - delta_text: '<|message|>'
      delta_token_ids: [200008]      # present when token-aligned (for the token path)
      expected:
        dynamo:
        - {index: 0, id: true, name: get_weather}   # name first
        sglang: []
      normal_text:        # per-impl non-tool text (leaks show here), only when non-empty
        sglang: '...'
```

A delta is `{index, id?, name?, arguments?}`; `id: true` means an id was emitted. The assembled call is derived by concatenating each index's name and argument fragments. Cross-implementation comparison happens at that assembled level. The per-chunk deltas differ by design and are the evidence, not the comparison key.

## Families

- `harmony/` uses the gpt-oss harmony parser through the token-id path (`parse_tool_call_streaming_incremental`). Label: "gpt-oss (harmony, token-id)".
- `harmony_text/` uses the same parser through the text path (`parse_tool_call_streaming_text`). Label: "gpt-oss (harmony, text)". The text path re-tokenizes a held suffix so character-split Harmony markers can settle before token commit. It is incremental, but it can lag by a small suffix on tiny chunks.
- All other families have no Dynamo parser v2 TC streaming implementation yet, so `expected.dynamo` is `unavailable` (TODO). The fixture key remains `dynamo` because it is the local-parser key in the shared schema. These rows render as `...` in the table. Their vLLM/SGLang behavior is the target to match when streaming is implemented.

## Tooling

- `build_stream_fixtures.py` assembles a fixture from source chunks and captured per-implementation per-chunk data (`--dynamo/--vllm/--sglang` JSON, `--unavailable`, `--captured`, `--family/--label`). The `--dynamo` flag is the local-parser fixture key.
- `record_dynamo_stream` records Dynamo parser v2 per-chunk emit through the token path, or through the text path with `--text`. The binary name is legacy.
- `stamp_stream_token_ids` stamps token-aligned `delta_token_ids` into the harmony overlay fixtures. It updates the overlay only, never the Dynamo-synced corpus.
- `capture_stream.py` / `capture_all_families.sh` captures vLLM/SGLang per-chunk output inside the engine containers (`docker exec`) and records the engine version. Use this when a family's streaming gets implemented and you need peer comparison data.

## Conformance test

`conformance/tests/parity_toolcalling_stream.rs` drives the Dynamo parser v2 Harmony parser over both `harmony/` (token) and `harmony_text/` (text), asserts the per-chunk emit matches, and checks the assembled result. It should run in less than one second; if it hangs, that is a bug, not slowness.
