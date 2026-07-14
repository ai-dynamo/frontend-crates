# dynamo-parsers-v2

Rust crate for Dynamo-owned token-incremental tool-call parsers. This is the v2 path for streaming parser behavior, and its public Rust contract intentionally mimics vLLM Rust's parser contract so vLLM can move toward using the frontend-crate parser instead of carrying a separate Rust parser surface.

This README is the canonical parser documentation for the workspace. The goals, family taxonomy, and how-to-add-a-parser guidance below cover both the v1 batch crate (`../parsers/`) and this v2 streaming crate, and `../parsers/README.md` defers here.

**Two parser paths exist today (universal convention).** v1 (`../parsers/`) is the **batch** parser, used two ways: plain **batch** (complete text parsed in one call) and **jail+batch** (streaming input is buffered — "jailed" — until a call completes, then batch-parsed and emitted all at once; a call is never streamed incrementally). v2 (this crate) is the **streaming** parser — its primary mode emits deltas per chunk as input arrives, and it can also take batch input (the whole text as one chunk, the batch-on-stream path). v2 is still under development. The two are kept in agreement (goal 8 below). v1 is not being merged into v2; when v2 is done it will fully replace v1, and all v1 code and docs will be removed outright. Put new parser work and documentation here, not in v1.

## Parser goals (read first)

These goals govern every tool-calling and reasoning parser, v1 and v2, and are the tie-breakers when the engines (vLLM, SGLang, Dynamo) disagree.

1. **Follow the model's own spec.** The model's chat template / tool-calling guide defines the emission grammar — parse to that, not to another engine's parser. Record the spec source in the fixture YAML so a reviewer can check the grammar against ground truth: add a `spec:` key (the HuggingFace chat-template or tool-calling-guide URL) on the family/fixture alongside the existing `captured_with:` engine versions. Spec URLs that currently live only in `parsers/v1/src/tool_calling/config.rs` comments should migrate into the fixtures over time.
2. **Error recovery is under-specified, so engine divergence is expected.** A spec says how a model EMITS a well-formed call; it rarely says what to do with malformed, truncated, or surrounding output. vLLM, SGLang, and Dynamo each interpret recovery differently, so a divergence on a recovery / edge case is normal and is documented with a `reason:` — it is not a bug to "fix" by matching a peer.
3. **Never leak tool-call markup into user-visible `normal_text`.** The `↯` conformance marker exists to catch this; a parser must strip recognized markup, never surface it as content.
4. **Never leak reasoning markup into user-visible content.** The reasoning parser moves `<think>...</think>` (and equivalents) into the reasoning channel; neither the reasoning nor the tool parser may leave its markup in `content`.
5. **Make a reasonable, bounded attempt to recover — recover only what is provably complete.** A captured value is complete when a delimiter follows it (the next marker such as `<`, or a closing `"` / `}` / `]` / `)`); recover that value — and the call — even when an outer end or wrapper marker is missing. A value that runs straight into end-of-stream with nothing after it is ambiguous: it may be truncated mid-token (is `NY` the whole value, or a cut-off `NYC`?), so drop it — never guess. Keying recovery on this single delimiter-terminated test is what keeps the decision consistent across families: same input shape, same drop/recover outcome. Recovery never invents content the model did not emit, never leaks markup, and `tracing::warn!`s with a stable `why=` field plus small byte counts (never raw model text or arguments). **A published spec overrides this rule (see goal 1):** if the model publishes a parser/regex that requires a fence, honor it — drop when that fence is missing, even though the value is delimiter-terminated — and document the divergence with the spec URL and a verbatim quote (e.g. MiniMax-M2.1 publishes `invoke_regex = re.compile(r"<invoke name=(.*?)</invoke>", re.DOTALL)`, which requires `</invoke>`, so a missing `</invoke>` yields no call). In the v1 batch parsers this rule is carried by the `allow_eof_recovery` flag (finalize / non-streaming only — streaming jails keep it `false`): `true` recovers a call whose outer end/wrapper marker is missing as long as the body still parses (and still drops a body truncated mid-value), `false` drops it. State the consequence in the fixture `reason:` — e.g. `allow_eof_recovery=true, so the complete-but-unterminated call is recovered` or `allow_eof_recovery=false (spec requires the fence), so the call is dropped`. Recovering an inner value whose own close tag is missing — the value is terminated by the next marker — is a separate value-capture concern, not this flag.
6. **Preserve as much of the original output as possible.** When the spec is silent, `normal_text` is the model output with only the recognized tool-call / reasoning markup spans removed — text before, between, and after calls is kept verbatim. Do not drop surrounding narration.
7. **Parsing is separate from validation.** Emit a tool call even when its function name is not in the request's tools list; the serving layer validates and rejects unknown tools. This matches vLLM.
8. **v1 (batch) and v2 (streaming) must always agree.** Fed the same complete output, both paths produce identical results — same calls, same `normal_text`. The only intended difference is mechanism, not output: v1 is the simple, inefficient reference that jails (buffers) the entire response before parsing, while v2 parses token-incrementally and jails only the minimal ambiguous suffix, so it can emit unambiguous text and calls as soon as they arrive (lower latency). v2 also delegates value-typing to the v1 batch parser, so the two stay in agreement by construction. This equivalence is v2's correctness contract, enforced by the `batch_via_stream` conformance test (`conformance/tests/parity_toolcalling_batch_via_stream.rs`); any intentional stream-vs-batch difference is recorded in its `known_divergences` allowlist.
9. **Mirror vLLM Rust's contract shape, not necessarily its output.** The types and method names mirror vLLM Rust so vLLM can adopt this crate, but Dynamo may intentionally diverge on output content (e.g. preserve surrounding text where vLLM drops it) — every such divergence carries a `reason:`.

The end-to-end workflow for adding a parser — fetch the chat template, identify the format, prefer an existing parser + config over new code, register, test, render — lives in the `tool-parser-generator` skill under `.agents/skills/tool-parser-generator/`. Reasoning parsers follow the same goals; see [`../parsers/v1/src/reasoning/README.md`](../parsers/v1/src/reasoning/README.md).

## Parser families

This is the single source of truth for the parser-family taxonomy — which grammar family a model belongs to and where the grammar is implemented. Families are shared across both paths: the **batch (v1)** implementation under `parsers/v1/src/tool_calling/` owns the grammar and value-typing, and the **streaming (v2)** implementation under `parsers/v2/src/tool_calling/` reuses it. Pick the family first when adding a model; only write a new module when the grammar is genuinely new.

A request flows reasoning-parser first (`<think>` stripping), then the tool-call parser on the non-reasoning tail, producing `Vec<ToolCallResponse>` + `normal_text` (v1) or a stream of `ToolParseResult` deltas (v2).

Tool-call families:

| Family | Grammar | Batch impl (`parsers/v1/src/tool_calling/`) | Examples |
| -- | -- | -- | -- |
| **DSML** | `<｜DSML｜tool_calls>...` with typed `string="true\|false"` parameters | `dsml/parser.rs` | DeepSeek V3.2, V4 |
| **XML** | `<tool_call>...</tool_call>` with nested `<parameter>` / `<function>` (or special-token variants) | `xml/parser.rs` (generic) or own file per variant | hermes, qwen3_coder, minimax_m2, glm47 (own file), kimi_k2 (own file, special-token XML) |
| **JSON** | Start sentinel + JSON `{name, arguments}` (single object or array) | `json/base_json_parser.rs` (+ variant files) | deepseek_v3, deepseek_v3_1, nemotron_deci, nemotron_nano, jamba, mistral, phi4, llama3_json, qwen25 |
| **Harmony** | OpenAI Harmony token stream with `<\|channel\|>`, `<\|message\|>`, `<\|call\|>` | `harmony/harmony_parser.rs` (wraps external `openai_harmony` crate) | gpt-oss-20B / 120B |
| **Pythonic** | `[func_name(arg=value, ...)]` Python function-call syntax | `pythonic/pythonic_parser.rs` | some Llama variants |
| **Gemma 4** | Custom: `<\|tool_call>call:name{key:<\|"\|>val<\|"\|>}<tool_call\|>`, bare keys, custom string delimiter | `gemma4/parser.rs` (recursive-descent into `serde_json::Value`) | Google Gemma 4 thinking models |

Reasoning families:

| Family | Grammar | Batch impl (`parsers/v1/src/reasoning/`) | Examples |
| -- | -- | -- | -- |
| **Basic (think-tag)** | `<think>...</think>` | `base_parser.rs` (`BasicReasoningParser`) | Qwen3, Nemotron, Kimi K2.5, DeepSeek R1 / V4, GLM-4.5+ |
| **Append-think** | `<think>...</think>` left inline as text, with `<think>` prefix on first chunk | `minimax_append_think_parser.rs` | MiniMax M2 |
| **Harmony channel** | Hidden `analysis` channel | `gpt_oss_parser.rs` (wraps external `openai_harmony`) | gpt-oss-20B / 120B |
| **Granite** | Custom start/end tokens | `granite_parser.rs` | IBM Granite |
| **Gemma 4 channel** | `<\|channel>thought\n...<channel\|>` with role-label prefix stripped | `gemma4_parser.rs` | Google Gemma 4 thinking models |

Streaming (v2) implementations exist today for `harmony`, `deepseek_v4` (DSML), and `qwen3_coder`; the remaining families run on the v1 batch parser until their streaming port lands.

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

- `parsers/v2/src/tool_calling/<family>.rs` for the parser implementation.
- `parsers/v2/src/tool_calling/mod.rs` for the family registry entry.
- `conformance/toolcalling/fixtures-stream-v2/<family>/TOOLCALLING.streamv2.*.yaml` for per-chunk stream captures.
- `conformance/toolcalling/fixtures-batch-on-stream-v2/<family>/TOOLCALLING.batch*.yaml` for complete batch text fed through streaming parsers.
- `conformance/toolcalling/fixtures-batch-v1/<family>/TOOLCALLING.batch*.yaml` only when the family or taxonomy cases do not already exist in the v1 batch corpus.
- `conformance/utils/lib/parsers/TOOLCALLING_STREAMING_V2_CASES.md` when adding a new stream-only case or changing stream case descriptions.
- `conformance/toolcalling/fixtures-stream-v2/README.md` only if the fixture schema or capture convention changes.

Fix legacy v1 parser bugs in `parsers/src/` and the matching v1 fixtures in `conformance/toolcalling/fixtures-batch-v1/`. While both paths coexist, keep v2-only parser behavior in `parsers_v2/`, `fixtures-stream-v2/`, and `fixtures-batch-on-stream-v2/` until v2 replaces v1.

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
5. Implement `parsers/v2/src/tool_calling/<family>.rs`, returning `ToolParseResult` from every chunk; start from `harmony.rs` (token/channel grammar) or `dsml.rs` (text incremental state machine).
6. Register the family in `create_tool_parser_for_family` in `parsers/v2/src/tool_calling/mod.rs`; override `prefers_tokens()` if the parser is token-native.
7. Add Rust unit tests for: one call, multiple calls, partial chunks, malformed recovery, `normal_text`, and EOF.
8. Add or update fixture files (see "Which Fixture Do I Edit?").
9. Capture one case (`conformance/utils/capture.sh dynamo-stream --fixture … --output …`), inspect, fix the parser, then capture all peer behavior (`capture.sh stream` / `capture.sh batch-on-stream`, optionally `--family <family>`).
10. Verify (`conformance/utils/check.sh`), render the HTML matrix (`conformance/utils/render_table_v2.sh`), and record any intentional divergence (see "How To Record Divergences").

Harmony is only the first example; DS4 and the other streaming families follow the same file layout, fixture schema, capture flow, and validation flow.

## Which Fixture Do I Edit?

- `conformance/toolcalling/fixtures-batch-v1/<family>/TOOLCALLING.batch*.yaml` — legacy v1 batch input and the current batch baseline. Do not hand-edit for v2 work; it is also the seed for stream capture.
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
  --fixture conformance/toolcalling/fixtures-stream-v2/inputs/harmony/TOOLCALLING.streamv2.1.yaml \
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
