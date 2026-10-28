# dynamo-parsers-v2

Rust crate for Dynamo-owned token-incremental tool-call parsers. This is the v2 path for streaming parser behavior, and its public Rust contract intentionally mimics vLLM Rust's parser contract so vLLM can move toward using the frontend-crate parser instead of carrying a separate Rust parser surface.

This README is the canonical parser documentation for the workspace. The goals, family taxonomy, and how-to-add-a-parser guidance below cover both the v1 batch crate (`../parsers/`) and this v2 streaming crate, and `../parsers/README.md` defers here.

**Implementation names and conformance convention names are different axes.** `dynamo-parsers` / `dynamo_v1` is the batch implementation, used for plain batch parsing and for jail+batch, where streaming input is buffered until a call completes and then emitted all at once. `dynamo-parsers-v2` / `dynamo_v2` is this newer crate; it contains both the first incremental tool-only parsers and the later Unified parsers. The first incremental path was originally called `stream-v2` because it was intended to replace batch+jail. Unified later introduced a second meaning for “v2.” Conformance now calls the older non-Unified convention **streaming v1** (`fixtures-stream-v1`, `TOOLCALLING.streamv1.*`) and reserves **v2** for Unified (`fixtures-unified-v2`). A `dynamo_v2-*` implementation capture inside `fixtures-stream-v1` is therefore expected, not contradictory.

**Runtime selection is separate from this naming cleanup.** Dynamo reads parser names from the model deployment. `DYN_ENABLE_EXPERIMENTAL_PARSERS_V2` enables both experimental routes; it does not choose one by itself. Routing checks Unified first: the exact `tool_call_parser=qwen3_coder` plus `reasoning_parser=qwen3` pair uses Qwen3 Unified when the flag is on. If no Unified pair matches, eligible `qwen3_coder` or `deepseek_v4` requests use the older non-Unified incremental parser. All remaining tool-parsing requests use batch+jail. DeepSeek V4.1 and Muse have separate Unified routes that do not depend on this flag. A parser failure does not trigger a general Unified-to-incremental-to-jail retry chain.

## Parser goals (read first)

These goals govern every tool-calling and reasoning parser, v1 and v2, and are the tie-breakers when the engines (vLLM, SGLang, Dynamo) disagree.

1. **Follow the model's own spec.** The model's chat template / tool-calling guide defines the emission grammar — parse to that, not to another engine's parser. Record the spec source in the fixture YAML so a reviewer can check the grammar against ground truth: add a `spec:` key (the HuggingFace chat-template or tool-calling-guide URL) on the family/fixture alongside the existing `captured_with:` engine versions. Spec URLs that currently live only in `parsers/v1/src/tool_calling/config.rs` comments should migrate into the fixtures over time.
2. **Error recovery is under-specified, so engine divergence is expected.** A spec says how a model EMITS a well-formed call; it rarely says what to do with malformed, truncated, or surrounding output. vLLM, SGLang, and Dynamo each interpret recovery differently, so a divergence on a recovery / edge case is normal and is documented with a `reason:` — it is not a bug to "fix" by matching a peer.
3. **Never leak tool-call markup into user-visible `normal_text`.** The `↯` conformance marker exists to catch this; a parser must strip recognized markup, never surface it as content.
4. **Never leak reasoning markup into user-visible content.** The reasoning parser moves `<think>...</think>` (and equivalents) into the reasoning channel; neither the reasoning nor the tool parser may leave its markup in `content`.
5. **Make a reasonable, bounded attempt to recover — recover only what is provably complete.** A captured value is complete when a delimiter follows it (the next marker such as `<`, or a closing `"` / `}` / `]` / `)`); recover that value — and the call — even when an outer end or wrapper marker is missing. A value that runs straight into end-of-stream with nothing after it is ambiguous: it may be truncated mid-token (is `NY` the whole value, or a cut-off `NYC`?), so drop it — never guess. Keying recovery on this single delimiter-terminated test is what keeps the decision consistent across families: same input shape, same drop/recover outcome. Recovery never invents content the model did not emit, never leaks markup, and `tracing::warn!`s with a stable `why=` field plus small byte counts (never raw model text or arguments). **A published spec overrides this rule (see goal 1):** if the model publishes a parser/regex that requires a fence, honor it — drop when that fence is missing, even though the value is delimiter-terminated — and document the divergence with the spec URL and a verbatim quote (e.g. MiniMax-M2.1 publishes `invoke_regex = re.compile(r"<invoke name=(.*?)</invoke>", re.DOTALL)`, which requires `</invoke>`, so a missing `</invoke>` yields no call). In the v1 batch parsers this rule is carried by the `allow_eof_recovery` flag (finalize / non-streaming only — streaming jails keep it `false`): `true` recovers a call whose outer end/wrapper marker is missing as long as the body still parses (and still drops a body truncated mid-value), `false` drops it. State the consequence in the fixture `reason:` — e.g. `allow_eof_recovery=true, so the complete-but-unterminated call is recovered` or `allow_eof_recovery=false (spec requires the fence), so the call is dropped`. Recovering an inner value whose own close tag is missing — the value is terminated by the next marker — is a separate value-capture concern, not this flag.
6. **Preserve as much of the original output as possible.** When the spec is silent, `normal_text` is the model output with only the recognized tool-call / reasoning markup spans removed — text before, between, and after calls is kept verbatim. Do not drop surrounding narration.
7. **Parsing is separate from validation.** Emit a tool call even when its function name is not in the request's tools list; the serving layer validates and rejects unknown tools. This matches vLLM.
8. **v1 and v2 can produce different argument values.** v1 uses the batch parser, while v2 owns its streaming conversion path and preserves exact argument text. `UnifiedEvent.arguments` remains a `serde_json::Value` projection and can lose numeric precision; use `tool_arguments_raw()` when the exact assembled JSON text matters. The `batch_via_stream` conformance test (`conformance/tests/conformance_toolcalling_batch_via_stream.rs`) records intentional differences in its `known_divergences` allowlist.
9. **Mirror vLLM Rust's contract shape, not necessarily its output.** The types and method names mirror vLLM Rust so vLLM can adopt this crate, but Dynamo may intentionally diverge on output content (e.g. preserve surrounding text where vLLM drops it) — every such divergence carries a `reason:`.

The end-to-end workflow for adding a parser — fetch the chat template, identify the format, prefer an existing parser + config over new code, register, test, render — lives in the `tool-parser-generator` skill under `.agents/skills/tool-parser-generator/`. Adding a family to the **unified** path (one ordered reasoning + content + tool stream, rather than the reasoning-then-tool split) is a different workflow with its own tiers and traps — see [`UNIFIED_PORTING.md`](UNIFIED_PORTING.md). Reasoning parsers follow the same goals; see [`../parsers/v1/src/reasoning/README.md`](../parsers/v1/src/reasoning/README.md).

**Writing a parser in YOUR OWN crate — including replacing a family this crate already ships — does not require forking or a PR here.** Implement `UnifiedParser`, call `register_unified_parser("family", factory)` at startup, and your parser is used for that family; registering an existing name shadows the built-in, and unregistering restores it. See [`CUSTOM_PARSERS.md`](CUSTOM_PARSERS.md) for the trait, the contract a parser must honour, and how to run the conformance corpus against your implementation.

## Parser families

This is the single source of truth for the parser-family taxonomy — which grammar family a model belongs to and where the grammar is implemented. Families are shared across both paths: the **batch (v1)** implementation under `parsers/v1/src/tool_calling/` owns the grammar and value-typing, and the **streaming (v2)** implementation under `parsers/v2/src/tool_calling/` reuses it. Pick the family first when adding a model; only write a new module when the grammar is genuinely new.

A request flows reasoning-parser first (`<think>` stripping), then the tool-call parser on the non-reasoning tail, producing `Vec<ToolCallResponse>` + `normal_text` (v1) or a stream of `ToolParseResult` deltas (v2).

Tool-call families:

| Family | Grammar | Batch impl (`parsers/v1/src/tool_calling/`) | Examples |
| -- | -- | -- | -- |
| **DSML** | `<｜DSML｜tool_calls>...` with typed `string="true\|false"` parameters | `dsml/parser.rs` | DeepSeek V3.2, V4 |
| **XML** | `<tool_call>...</tool_call>` with nested `<parameter>` / `<function>` (or special-token variants) | `xml/parser.rs` (generic) or own file per variant | hermes, qwen3_coder, minimax_m2, GLM-4.7/5.x via `glm47` (shared implementation), kimi_k2 (own file, special-token XML) |
| **JSON** | Start sentinel + JSON `{name, arguments}` (single object or array) | `json/base_json_parser.rs` (+ variant files) | deepseek_v3, deepseek_v3_1, nemotron_deci, nemotron_nano, jamba, mistral, phi4, llama3_json, qwen25 |
| **Harmony** | OpenAI Harmony token stream with `<\|channel\|>`, `<\|message\|>`, `<\|call\|>` | `harmony/harmony_parser.rs` (wraps external `openai_harmony` crate) | gpt-oss-20B / 120B |
| **Pythonic** | `[func_name(arg=value, ...)]` Python function-call syntax | `pythonic/pythonic_parser.rs` | some Llama variants |
| **Gemma 4** | Custom: `<\|tool_call>call:name{key:<\|"\|>val<\|"\|>}<tool_call\|>`, bare keys, custom string delimiter | `gemma4/parser.rs` (recursive-descent into `serde_json::Value`) | Google Gemma 4 thinking models |
| **ATEM** | Recipient-routed channels `<\|start\|>assistant to=<rcpt><\|message\|>...<\|eom\|>` whose tool bodies hold `<atem:invoke name="NAME"><atem:parameter name="KEY">VALUE</atem:parameter></atem:invoke>` | `tool_calling/muse_glimmer.rs`, `tool_calling/muse_spark.rs` | Muse Glimmer, Muse Spark |

Reasoning families:

| Family | Grammar | Batch impl (`parsers/v1/src/reasoning/`) | Examples |
| -- | -- | -- | -- |
| **Basic (think-tag)** | `<think>...</think>` | `base_parser.rs` (`BasicReasoningParser`) | Qwen3, Nemotron, Kimi K2.5, DeepSeek R1 / V4, GLM-4.5+ |
| **Append-think** | `<think>...</think>` left inline as text, with `<think>` prefix on first chunk | `minimax_append_think_parser.rs` | MiniMax M2 |
| **Harmony channel** | Hidden `analysis` channel | `gpt_oss_parser.rs` (wraps external `openai_harmony`) | gpt-oss-20B / 120B |
| **Granite** | Custom start/end tokens | `granite_parser.rs` | IBM Granite |
| **Gemma 4 channel** | `<\|channel>thought\n...<channel\|>` with role-label prefix stripped | `gemma4_parser.rs` | Google Gemma 4 thinking models |
| **ATEM channel** | `<\|start\|>assistant to=self<\|message\|>...<\|eom\|>` — the recipient picks the channel, so there is no delimiter pair; Muse Spark additionally reserves `commentary` as visible text | `unified/muse_glimmer.rs`, `unified/muse_spark.rs` | Muse Glimmer, Muse Spark |

Streaming (v2) implementations exist today for `harmony`, `deepseek_v4` (DSML), `qwen3_coder`, `gemma4`, `muse_glimmer`, and `muse_spark` (ATEM); the remaining families run on the v1 batch parser until their streaming port lands.

The `deepseek_v4` tool-only selector also accepts V4.1's spaced DSML tags (`<｜DSML｜ calls>`, `<｜DSML｜ invoke ...>`). Both dialects use one scanner with their existing boundary parsers and decoders; there is no separate tool-only selector. Reasoning behavior and V4.1 JSON-body support remain separate changes (#253 and #250).

## Nemotron 3 Ultra

Nemotron 3 Ultra 550B-A55B uses Unified family `nemotron_v3` (alias `nemotron3`), combining `nemotron_v3` reasoning with `qwen3_coder` XML tool calls. It shares the Qwen scanner, emitter, guided-prefix handling, and guided router. With no explicit starting state, output begins in reasoning; `Reasoning` preserves that channel and `Response` disables reasoning-marker interpretation. Reset retains the request configuration. Complete native calls are extracted even when `</think>` is omitted, and reasoning resumes after the call. Reasoning-looking markers inside argument values remain literal data. Older Deci/Ultra 253B formats use their existing separate families. This adds parser-library support only; serving integration is separate.

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

Keep these names and field meanings aligned with vLLM Rust unless Dynamo explicitly needs a small extension. Dynamo extensions are token-id input through `ToolParserInput`, `push_tokens`, and `push_input`, plus `ToolCallDelta::complete` so adapters can suppress provisional calls that never become valid.

```rust
// Mirrors vLLM Rust `Tool` verbatim.
pub struct Tool {
    pub name: String,
    pub description: Option<String>,
    pub parameters: serde_json::Value,
    pub strict: Option<bool>,
}

// Dynamo extension: `complete` records whether this update commits the call.
pub struct ToolCallDelta {
    pub tool_index: usize,
    pub name: Option<String>,
    pub arguments: String,
    pub complete: bool,
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

**Do not drift from vLLM Rust here without an explicit contract reason.** These types follow the vLLM Rust `ToolParser` shape, not vLLM Python wire deltas. Dynamo currently adds token input (`push_tokens` / `push_input` / `prefers_tokens`) and `ToolCallDelta::complete`; keep other fields aligned.

## Fixture Files To Add

New cases, capture jobs, and normal reports use `UNIFIED.*`. Register the family in `conformance/utils/src/parser_families.yaml`, author applicable family-native scenarios and independent GOLDEN expectations in `conformance/utils/src/gen_unified_golden.py`, and document their contracts in `conformance/utils/lib/parsers/UNIFIED_CASES.md`. Preserve published IDs and expectations. New IDs use `UNIFIED.<group>-<description>` through `conformance/utils/src/unified_taxonomy.py`; existing numeric IDs retain their aliases.

Keep authored requests/GOLDEN in `conformance/fixtures-unified-v2/families/<family>/inputs_and_golden.yaml`. Latest parser tests read versionless recorded observations from `conformance/fixtures-unified-v2/families/<family>/<implementation>.yaml`. Contributors run latest tests and generate the latest report; administrators update recorded captures through the [latest capture plan](../../docs/CONFORMANCE-LATEST-PLAN.md).

Retain archive-backed `TOOLCALLING.*` fixtures and their original observations as historical evidence. Their format below is for explicitly requested historical maintenance; normal Unified generation does not read those archives or derive GOLDEN from their results.

## Historical Fixture Format

Retained non-Unified v2 fixtures use explicit implementation names. Do not rely on renderer inference for parser failures.

```yaml
captured_with:
  dynamo_v2: Dynamo parser v2
  vllm_rust: v0.22.0 0b3ba88f165976e77ca5e6a7a3f5bba4562b80af
  vllm_python: 0.22.0
  sglang_python: 0.5.12.post1
cases:
  TOOLCALLING.streamv1.4.a:
    expected:
      dynamo_v2:
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
4. Decide the parser family id and peer parser names; add a row to `conformance/utils/src/parser_families.yaml` (`vllm_python` / `vllm_rust` / `sglang_python` / `dynamo_v2` / `preferred_input`), AND declare the family's grammar tokens in the `markers:` section of the same file (`pairs` / `singletons` / `leak`; explicit `{}` for markup-less grammars). The `↯` leak detector and the popup token coloring are derived from that declaration — skipping it makes leaks render as clean cells and every token render as a red orphan, and `check.sh coverage` fails on it.
5. Implement `parsers/v2/src/tool_calling/<family>.rs`, returning `ToolParseResult` from every chunk; start from `harmony.rs` (token/channel grammar) or `dsml.rs` (text incremental state machine).
6. Register the family in `create_tool_parser_for_family` in `parsers/v2/src/tool_calling/registry.rs`; override `prefers_tokens()` if the parser is token-native.
7. Add Rust unit tests for: one call, multiple calls, partial chunks, malformed recovery, `normal_text`, and EOF.
8. Add or update fixture files (see "Which Fixture Do I Edit?"). What "complete" means is machine-readable: `conformance/case-taxonomy.yaml` lists every case group and sub-case with requiredness. Run `conformance/utils/check.sh coverage --family <family>` — its FAIL list IS your fixture TODO list; a case your grammar cannot express gets a placeholder entry with an `explanation:` instead of silence.
9. Run the latest parser tests against the authored requests and independent GOLDEN, and fix parser mismatches.
10. Generate the [latest report](../../conformance/README.md#generate-the-latest-or-history-report), inspect the affected family over HTTP, and explain any deliberate authored contract change. Repository administrators maintain recorded captures and historical backfills.

### Unified parser validation

Contributors edit the shared parser owner and authored inputs/GOLDEN, then run the latest tests:

```bash
cargo test --locked -p dynamo-conformance-fixtures-v2 --test unified_render -- --nocapture
cargo test --locked -p dynamo-conformance-fixtures-v2 --test unified_parity -- --nocapture
```

Generate the [latest report](../../conformance/README.md#generate-the-latest-or-history-report) and inspect affected inputs, initialization, chunks, GOLDEN, and recorded observations over HTTP. Missing recorded output requires an administrator capture update; it does not require contributors to capture historical versions. A live parser mismatch must be fixed or explained by a deliberate authored contract change. Do not hide it with `reason:`, `unavailable:`, stale HTML, or an unsupported GOLDEN edit.

The legacy Harmony example documents retained historical fixtures. Recorded capture updates use the [latest capture plan](../../docs/CONFORMANCE-LATEST-PLAN.md).

## Which Fixture Do I Edit?

New family and scenario coverage belongs in the Unified generator, shared registry, and `UNIFIED_CASES.md`; preserve existing authored requests and independent GOLDEN values. Run latest parser tests and generate the latest report. Historical backfills are administrator work.

### Retained historical fixture paths

- `conformance/toolcalling/fixtures-batch-v1/<family>/TOOLCALLING.batch*.yaml` — legacy v1 batch input and the current batch baseline. Do not hand-edit for v2 work; it is also the seed for stream capture.
- `conformance/toolcalling/fixtures-stream-v1/<family>/TOOLCALLING.streamv1.*.yaml` — per-chunk streaming behavior (the TC stream tab). Retain for historical stream regression tests.
- `conformance/toolcalling/fixtures-batch-on-stream-v1/<family>/TOOLCALLING.batch*.yaml` — each batch sample's full text run through the stream parser (the batch-on-stream tab).

New models use the Unified workflow above. Do not seed new Unified expectations or observations from these historical archives.

## How To Record Historical Divergences

Every cell in the matrix must be backed by exact YAML — the renderer infers nothing:

- `reason:` — an **intentional** output difference (the parser deliberately differs; the cell shows the divergence marker without `?`).
- `expected.<impl>.error` — the parser **ran and threw**. A structured `{kind, message}` renders `✗`; a plain string is a declared expected-error and renders `!`.
- `unavailable.<impl>` — the parser **did not run** or cannot exist (no model_text, no parser for the family, source not set up). Renders neutral `n/a`.
- `captured_with:` — the engine version each peer block was captured against; required whenever a peer has captured output.

A divergent peer block with no `reason:` renders `?` (research needed) — never leave one unexplained.

## Done Means

For a new parser family, done means:

- Rust parser unit tests and latest Dynamo fixture tests pass.
- Applicable Unified scenarios have family-native authored inputs and independent GOLDEN. Grammar exclusions have explicit explanations, and the family's `markers:` are declared in `parser_families.yaml`.
- The latest report is regenerated locally and affected cases are inspected for tool-call markup leaks, undeclared tokens, and mismatches against GOLDEN. Missing recorded observations are identified for an administrator capture update.
- Case descriptions exist for every new scenario, and its stable ID is declared in the Unified taxonomy.

## Commands

Run this quick Rust check for the v2 parser crate:

```bash
cargo test --locked -p dynamo-parsers-v2 -- --nocapture
```

Run this fixture-based check for committed YAML:

```bash
cargo test --locked -p dynamo-conformance-fixtures-v2 -- --nocapture
```

Generate and serve the [latest report](../../conformance/README.md#generate-the-latest-or-history-report) after parser, authored input, or metadata changes. Capture updates and historical backfills use the [latest capture plan](../../docs/CONFORMANCE-LATEST-PLAN.md).

Run the table and marker regression tests:

```bash
python3 -m pytest conformance/utils/tests/test_stream_on_batch.py
```

## Reasoning Migration TODO

Reasoning fixtures are still v1 today. They use `expected.dynamo`, `expected.vllm`, and `expected.sglang`, and the shared JavaScript model preserves inference of some Python parser exceptions from v1 n/a stubs with no `model_text`.

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

### Muse Spark Open structural tags

The `muse_spark_open` / `muse_spark` structural-tag builder constrains native
ATEM calls, including qualified tool names, raw scalar parameters, JSON-valued
containers, required/named choice, and single-call policy. It preserves the
model's reasoning header and allows one pre-tool commentary message. Auto choice can also constrain the
final answer inside its native `to=user` envelope.

The grammar must run from the first generated token. An external reasoning gate
that activates only after a tool channel opens cannot enforce forced choice and
is rejected. The grammar completes before the checkpoint's end-of-turn stop
token, allowing the engine to terminate rather than masking EOS. Dynamo configures this ownership for its SGLang and vLLM workers.

Constrained arguments support closed objects with declared properties and
optional/required fields in schema order; `tool_arguments_any_order` is rejected
rather than allowing duplicate or missing fields. Strings use native unquoted ATEM text; the Spark
parser retains their declared string type even for values such as `42` and
`true`. Enum and constant strings are constrained directly. Other scalar and
container parameters use JSON Schema validation inside the ATEM envelope.
Freeform custom input retains its raw, verbatim representation. Non-strict
tools retain unconstrained parameter values.

Unsupported XML-envelope constraints (for example open top-level argument
properties, escaped or empty parameter names, and unrestricted text constraints
on raw string input) return build errors rather than weaker grammars.
Literal ATEM parameter-closing delimiters inside argument strings are outside
this native format's qualification. Reserved tool names such as `commentary`
use the generic `to=tool` channel while retaining their exact function name.
