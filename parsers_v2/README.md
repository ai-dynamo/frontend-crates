# dynamo-parsers-v2

Rust crate for Dynamo-owned token-incremental tool-call parsers. This is the v2 path for streaming parser behavior, and its public Rust contract intentionally mimics vLLM Rust's parser contract so vLLM can move toward using the frontend-crate parser instead of carrying a separate Rust parser surface.

## Why It Mimics vLLM Rust

The important DIS-2218 comparison is vLLM Rust vs Dynamo Rust. vLLM Python is still useful coverage and behavioral evidence, but it is not the API target.

vLLM Rust 0.22.0 source was checked at tag `v0.22.0`, commit `0b3ba88f165976e77ca5e6a7a3f5bba4562b80af`. Its parser crate is `rust/src/tool-parser/Cargo.toml`, crate name `vllm-tool-parser`. Local checkout paths must not be written into fixtures, docs, or generated HTML; fixtures record only source versions under `captured_with.*`.

The vLLM Rust parser API is streaming-first. `push()` consumes decoded text deltas, `finish()` flushes buffered state, and `parse_complete()` is a helper over the same streaming path. There is no separate `V_rb` implementation in the conformance matrix; batch-shaped text through vLLM Rust is still `V_rs`.

Dynamo duplicates the small Rust data model instead of depending on vLLM crates directly. The names and fields are aligned so an adapter stays trivial now, and so vLLM can later import Dynamo-owned parser types if it switches to frontend-crates.

## What's In The Crate

```text
src/
├── tool_calling/
│   ├── traits.rs      # Dynamo-owned mirror of the vLLM Rust parser contract
│   ├── mod.rs         # family registry
│   ├── harmony.rs     # gpt-oss / Harmony streaming parser, text or token IDs
│   └── dsml.rs        # DeepSeek V4 DSML streaming parser
└── bin/
    ├── record_dynamo_stream.rs       # capture Dynamo v2 per-chunk stream output
    ├── record_batch_via_stream.rs    # capture complete batch text through stream parser
    └── stamp_stream_token_ids.rs     # stamp Harmony token IDs into stream fixtures
```

## Parser Contract

`tool_calling/traits.rs` defines Dynamo-owned versions of the vLLM Rust parser contract:

- `Tool`
- `ToolCallDelta`
- `ToolParseResult`
- `ToolParser`

Keep these names and field meanings aligned with vLLM Rust unless Dynamo explicitly needs a small extension. Current Dynamo extension: a parser may accept decoded text chunks or token-id chunks through `ToolParserInput`, `push_tokens`, and `push_input`.

```rust
// Mirrors vLLM Rust `Tool` verbatim.
pub struct Tool {
    pub name: String,
    pub description: Option<String>,
    pub parameters: serde_json::Value,
    pub strict: Option<bool>,
}

// Mirrors vLLM Rust `ToolCallDelta` verbatim.
pub struct ToolCallDelta {
    pub tool_index: usize,
    pub name: Option<String>,
    pub arguments: String,
}

// Mirrors vLLM Rust `ToolParseResult` verbatim.
pub struct ToolParseResult {
    pub normal_text: String,
    pub calls: Vec<ToolCallDelta>,
}

// Dynamo extension: vLLM Rust is text-only; Dynamo can also route token chunks.
pub enum ToolParserInput<'a> {
    Text(&'a str),
    Tokens(&'a [u32]),
}

// Mirrors vLLM Rust `ToolParser` except for the explicitly marked token-input extensions.
pub trait ToolParser: Send {
    fn create(tools: &[Tool]) -> anyhow::Result<Box<dyn ToolParser>> where Self: Sized + 'static;
    fn preserve_special_tokens(&self) -> bool { false }
    fn push(&mut self, chunk: &str) -> anyhow::Result<ToolParseResult>;
    // Dynamo extension: token-native parser input.
    fn push_tokens(&mut self, ids: &[u32]) -> anyhow::Result<ToolParseResult> { Ok(ToolParseResult::default()) }
    // Dynamo extension: caller-selected text or token input.
    fn push_input(&mut self, input: ToolParserInput<'_>) -> anyhow::Result<ToolParseResult> { ... }
    fn finish(&mut self) -> anyhow::Result<ToolParseResult> { Ok(ToolParseResult::default()) }
    fn parse_complete(&mut self, output: &str) -> anyhow::Result<ToolParseResult> { ... }
}
```

Rules:

- `ToolCallDelta` has no parser-minted `id`; the serving layer owns IDs.
- `arguments` is a `String`, not `Option<String>`. Use `""` for a name-only delta.
- `normal_text` is first-class and must contain only content that should be returned to the user.
- Keep parser recovery from leaking tool markers into `normal_text` when the grammar can recover or safely suppress malformed tool syntax.
- Text and token input should not be mixed for one parser run. Use all text chunks or all token chunks for a fixture capture.

**Do not drift from vLLM Rust here.** These four types intentionally mirror the vLLM Rust `ToolParser` contract, not vLLM Python wire deltas — vLLM Rust may later depend on this frontend crate, so Dynamo keeps a small duplicated contract that stays shaped like vLLM Rust. The one allowed Dynamo-only extension is token input (`push_tokens` / `push_input` / `prefers_tokens`), which token-native parsers like Harmony need; everything else should match vLLM Rust field-for-field.

## Fixture Files To Add

For a new streaming parser family, add or update these files:

- `parsers_v2/src/tool_calling/<family>.rs` for the parser implementation.
- `parsers_v2/src/tool_calling/mod.rs` for the family registry entry.
- `conformance/toolcalling/fixtures-stream-v2/<family>/TOOLCALLING.streamv2.*.yaml` for per-chunk stream captures.
- `conformance/toolcalling/fixtures-batch-on-stream-v2/<family>/TOOLCALLING.batch*.yaml` for complete batch text fed through streaming parsers.
- `conformance/toolcalling/fixtures/<family>/TOOLCALLING.batch*.yaml` only when the family or taxonomy cases do not already exist in the v1 batch corpus.
- `conformance/utils/lib/parsers/TOOLCALLING_STREAMING_V2_CASES.md` when adding a new stream-only case or changing stream case descriptions.
- `conformance/toolcalling/fixtures-stream-v2/README.md` only if the fixture schema or capture convention changes.

Fix legacy v1 parser bugs in `parsers/src/` and the matching v1 fixtures in `conformance/toolcalling/fixtures/`. During the bridge, keep v2-only parser behavior in `parsers_v2/`, `fixtures-stream-v2/`, and `fixtures-batch-on-stream-v2/` until the v1/v2 merge lands.

## Fixture Format

v2 fixtures should use explicit implementation names. Do not rely on renderer inference for parser failures.

```yaml
captured_with:
  dynamo_rust: Dynamo parser v2
  vllm_rust: v0.22.0 0b3ba88f165976e77ca5e6a7a3f5bba4562b80af
  vllm_python: 0.22.0
  sglang_python: 0.5.12.post1
cases:
  TOOLCALLING.streamv2.4.a:
    expected:
      dynamo_rust:
        calls: []
        normal_text: ''
      vllm_rust:
        unavailable: 'vLLM Rust parser not captured: tool parser parsing failed: invalid Hermes'
      vllm_python:
        calls: []
        normal_text: ''
      sglang_python:
        calls: []
        normal_text: ''
```

Rules:

- Use `expected.dynamo_rust`, `expected.vllm_rust`, `expected.vllm_python`, and `expected.sglang_python` for parser output in v2 fixtures.
- Use `unavailable.<impl>` or `expected.<impl>.unavailable` when a parser does not exist, cannot run, or capture failed before output was available.
- Use `expected.<impl>.error` only when the parser ran and the expected behavior is a thrown parser exception.
- Every `X` or `✗` parser-failure marker in generated HTML must have the exact error text in YAML and in the pop-out.
- Put source versions under `captured_with.*`; do not write local checkout paths into YAML.

## Adding A Day-0 Tool-Calling Parser

In order:

1. Read the model's tool-call output spec and its tokenizer / special-token behavior — token boundaries decide whether the parser is text- or token-native.
2. Inspect the vLLM **Rust** parser first: `Tool`, `ToolCallDelta`, `ToolParseResult`, and `ToolParser` are the API target (Rust vs. Rust). Do not shape the parser like vLLM Python wire deltas.
3. Inspect vLLM **Python** and **SGLang** for behavior and coverage — they are the peer references the matrix compares against.
4. Decide the parser family id and peer parser names; add a row to `conformance/utils/src/parser_families.yaml` (`vllm_python` / `vllm_rust` / `sglang_python` / `dynamo_v2` / `preferred_input`).
5. Implement `parsers_v2/src/tool_calling/<family>.rs`, returning `ToolParseResult` from every chunk; start from `harmony.rs` (token/channel grammar) or `dsml.rs` (text incremental state machine).
6. Register the family in `create_tool_parser_for_family` in `parsers_v2/src/tool_calling/mod.rs`; override `prefers_tokens()` if the parser is token-native.
7. Add Rust unit tests for: one call, multiple calls, partial chunks, malformed recovery, `normal_text`, and EOF.
8. Add or update fixture files (see "Which Fixture Do I Edit?").
9. Capture one case (`conformance/utils/capture.sh dynamo-stream --fixture … --output …`), inspect, fix the parser, then capture all peer behavior (`capture.sh stream` / `capture.sh batch-on-stream`, optionally `--family <family>`).
10. Verify (`conformance/utils/check.sh`), render the HTML matrix (`conformance/utils/render_table_v2.sh`), and record any intentional divergence (see "How To Record Divergences").

Harmony is only the first example; DS4 and the other streaming families follow the same file layout, fixture schema, capture flow, and validation flow.

## Which Fixture Do I Edit?

- `conformance/toolcalling/fixtures/<family>/TOOLCALLING.batch*.yaml` — legacy v1 batch input and the current batch baseline. Do not hand-edit for v2 work; it is also the seed for stream capture.
- `conformance/toolcalling/fixtures-stream-v2/<family>/TOOLCALLING.streamv2.*.yaml` — per-chunk streaming behavior (the TC stream tab). Edit/capture here for streaming parser work.
- `conformance/toolcalling/fixtures-batch-on-stream-v2/<family>/TOOLCALLING.batch*.yaml` — each batch sample's full text run through the stream parser (the batch-on-stream tab).

Decision rule for a new model: add the stream cases under `fixtures-stream-v2/`, capture peers, and let the batch-on-stream overlay derive from the v1 batch corpus.

## How To Record Divergences

Every cell in the matrix must be backed by exact YAML — the renderer infers nothing:

- `reason:` — an **intentional** output difference (the parser deliberately differs; the cell shows the divergence marker without `?`).
- `expected.<impl>.error` — the parser **ran and threw**. A structured `{kind, message}` renders `✗`; a plain string is a declared expected-error and renders `!`.
- `unavailable.<impl>` — the parser **did not run** or cannot exist (no model_text, no parser for the family, source not set up). Renders neutral `n/a`.
- `captured_with:` — the engine version each peer block was captured against; required whenever a peer has captured output.

A divergent peer block with no `reason:` renders `?` (research needed) — never leave one unexplained.

## Done Means

For a new parser family, done means:

- Rust parser unit tests pass and the Dynamo fixture tests pass.
- vLLM Python / SGLang live checks pass, or each failure is explicitly recorded (`error`/`unavailable` with exact text).
- vLLM Rust captures include the source tag/commit in `captured_with` when available.
- The HTML matrix is regenerated locally and has no unexplained `?` and no accidental tool-call markup leaks (`↯`).
- Case descriptions exist for every new sub-case.

## Commands

Run this quick Rust check for the v2 parser crate:

```bash
cargo test --locked -p dynamo-parsers-v2 -- --nocapture
```

Run this fixture-based check for committed YAML:

```bash
cargo test --locked -p dynamo-conformance-fixtures-v2 -- --nocapture
```

Capture one Dynamo v2 stream fixture into JSON:

```bash
conformance/utils/capture.sh dynamo-stream \
  --fixture conformance/toolcalling/fixtures-stream-v2/harmony/TOOLCALLING.streamv2.1.yaml \
  --output /tmp/dynamo_stream.json
```

Capture all stream behavior and refresh v2 stream fixtures:

```bash
conformance/utils/capture.sh stream \
  --vllm-container vllm-localdev \
  --sglang-container sglang-localdev \
  --vllm-rust-source ~/dynamo/vllm-0.22.0
```

Capture all batch-on-stream behavior and refresh v2 batch-on-stream fixtures:

```bash
conformance/utils/capture.sh batch-on-stream \
  --vllm-container vllm-localdev \
  --sglang-container sglang-localdev \
  --vllm-rust-source ~/dynamo/vllm-0.22.0 \
  --capture-dynamo-rust-json /tmp/dynamo_batch_on_stream.json
```

Generate the HTML matrix after code or fixture changes:

```bash
conformance/utils/render_table_v2.sh
```

Run the table and marker regression tests:

```bash
python3 -m pytest conformance/utils/tests/test_stream_on_batch.py
```

## Reasoning Migration TODO

Reasoning fixtures are still v1 today. They use `expected.dynamo`, `expected.vllm`, and `expected.sglang`, and the current HTML renderer infers some Python parser exceptions from v1 n/a stubs with no `model_text`.

TODO for reasoning v2 migration: move reasoning fixtures to the v2 explicit implementation format before treating the table as the source of truth. The migrated YAML must record parser failures directly, for example:

```yaml
expected:
  dynamo_rust:
    unavailable: No reasoning parser v2 for this family yet.
  vllm_python:
    error: "KeyError: 'model_text'"
  sglang_python:
    error: "KeyError: 'model_text'"
```

Do not keep inferred Python exception markers after reasoning moves to v2. The YAML should say exactly which parser failed and with what message.
