# dynamo-parsers-v2

Rust parsers for ordered reasoning, text, and tool-call output. Includes Unified parsing and a tool-only API shaped like vLLM Rust's parser contract.

This guide covers parser goals, grammar families, and contributor workflows for both [v1](../v1/) and v2.

**Crate versions and corpus names differ.** `dynamo-parsers` / `dynamo_v1` is the batch/jail implementation. `dynamo-parsers-v2` / `dynamo_v2` includes incremental tool-only and Unified parsers. In conformance, **streaming v1** means non-Unified (`fixtures-stream-v1`, `TOOLCALLING.streamv1.*`); **v2** means Unified (`fixtures-unified-v2`). A `dynamo_v2-*` capture in `fixtures-stream-v1` is therefore valid. See [migration terminology](../../docs/PARSERS-V2-MIGRATION-PLAN.md#terminology) for the naming history.

**Pin and test the version you integrate.** Dynamo uses this published crate for compatible configurations, but its `0.x` API can change between minor releases. The [Dynamo routing guide](https://github.com/ai-dynamo/dynamo/blob/main/docs/fern/pages/use-cases/tool-calling-and-reasoning/troubleshooting-tool-calls.md#parser-generation-routing) owns serving selection. Library families and vendor registrations do not automatically add serving adapters or qualification.

## Using Unified Parsing

This example targets `dynamo-parsers-v2` `0.8.4` and `anyhow` `1`:

```toml
[dependencies]
anyhow = "1"
dynamo-parsers-v2 = "=0.8.4"
```

```rust
use dynamo_parsers_v2::{
    create_unified_parser_for_family, UnifiedParserInit, UnifiedParserOutput,
};

fn main() -> anyhow::Result<()> {
    let mut parser = create_unified_parser_for_family("qwen3", &[])?;
    parser.initialize_request(UnifiedParserInit::default())?;
    let mut output = UnifiedParserOutput::default();
    for delta in ["<think>check", " again", "</think>Hello", " there"] {
        // A retained output coalesces its last event; forward a fresh buffer.
        let mut committed = UnifiedParserOutput::default();
        parser.parse_into(delta, &mut committed)?;
        for event in &committed.events {
            println!("{event:?}");
        }
        output.append(&mut committed);
    }
    let mut tail = parser.finish()?;
    for event in &tail.events {
        println!("{event:?}");
    }
    output.append(&mut tail);
    assert_eq!(output.assembled().len(), 2);

    // Reuse only after the previous stream has finished or been abandoned.
    let _uncommitted_tail = parser.reset();
    parser.initialize_request(UnifiedParserInit::default())?;
    let mut next = UnifiedParserOutput::default();
    parser.parse_into("Next answer", &mut next)?;
    next.append(&mut parser.finish()?);
    assert!(!next.events.is_empty());
    Ok(())
}
```

Create one parser per request choice. Preserve special tokens when `preserve_special_tokens()` is true, and check `input_capabilities()` before feeding token IDs. Before parsing, resolve prompt prefill, native/guided output, and invalid-payload policy into `UnifiedParserInit`; the example uses native defaults.

Finalize once. Use `finish_at_eos()` only for a backend-confirmed grammar-ending EOS; it can complete native arguments with an implicit terminator, as in Beam. Use `finish()` for truncation, cancellation, or transport EOF, as above.

Events appended before a `parse_into` error remain committed. `reset()` returns uncommitted bytes and clears stream state; initialize the next request because built-in reset may retain request configuration.

Forward a fresh output buffer after each advance and after finalization, then append it to retained output. Adjacent text and reasoning events coalesce, so `output.events.len()` cannot track newly committed output.

Tool updates carry a stable `tool_index`, optional name, raw `arguments` fragments, and `complete` flag. Concatenate fragments per index, or use retained output's `tool_arguments_raw()`, when exact JSON text, key order, or numeric representation matters. `assembled()` returns ordered `UnifiedEvent` values with typed `serde_json::Value` arguments and may change their lexical representation. Do not serialize that projection back to wire arguments when fidelity matters.

Provisional updates do not authorize execution. The serving layer must validate tool availability, authorization, schema, and call policy.

## Numeric values and Serde consumers

The default build keeps `serde_json` numbers compatible with independent Serde serializers such as Pythonizer, MessagePack, and YAML. Parser argument deltas and `tool_arguments_raw()` preserve exact numeric JSON tokens, including large integers, fractions, and exponents, through the shared raw-argument representation. `UnifiedEvent.arguments` is a lossy `serde_json::Value` projection; use the raw accessor for serving exact argument text.

Enable the `arbitrary-precision` Cargo feature only when a consumer needs arbitrary-precision `serde_json::Value` numbers and supports their Serde representation. This enables `serde_json/arbitrary_precision`, which serializes numbers through a private struct rather than native numeric scalar methods. Independent serializers may expose that struct as a map or reject it. Cargo unifies dependency features, so enabling this feature also affects other `serde_json` users in the same dependency graph. `raw_value` remains enabled in both builds.

## Parser goals (read first)

These goals govern every tool-calling and reasoning parser, v1 and v2, and are the tie-breakers when the engines (vLLM, SGLang, Dynamo) disagree.

1. **Follow the model's own spec.** The model's chat template / tool-calling guide defines the emission grammar — parse to that, not to another engine's parser. Record the spec source in the fixture YAML so a reviewer can check the grammar against ground truth: add a `spec:` key (the HuggingFace chat-template or tool-calling-guide URL) on the family/fixture alongside the existing `captured_with:` engine versions. Spec URLs that currently live only in `parsers/v1/src/tool_calling/config.rs` comments should migrate into the fixtures over time.
2. **Error recovery is under-specified, so engine divergence is expected.** A spec says how a model EMITS a well-formed call; it rarely says what to do with malformed, truncated, or surrounding output. vLLM, SGLang, and Dynamo each interpret recovery differently, so a divergence on a recovery / edge case is normal and is documented with a `reason:` — it is not a bug to "fix" by matching a peer.
3. **Never leak tool-call markup into user-visible `normal_text`.** The `↯` conformance marker exists to catch this; a parser must strip recognized markup, never surface it as content.
4. **Never leak reasoning markup into user-visible content.** The reasoning parser moves `<think>...</think>` (and equivalents) into the reasoning channel; neither the reasoning nor the tool parser may leave its markup in `content`.
5. **Make a reasonable, bounded attempt to recover — recover only what is provably complete.** A captured value is complete when a delimiter follows it (the next marker such as `<`, or a closing `"` / `}` / `]` / `)`); recover that value — and the call — even when an outer end or wrapper marker is missing. A value that runs straight into end-of-stream with nothing after it is ambiguous: it may be truncated mid-token (is `NY` the whole value, or a cut-off `NYC`?), so drop it — never guess. Keying recovery on this single delimiter-terminated test is what keeps the decision consistent across families: same input shape, same drop/recover outcome. Recovery never invents content the model did not emit, never leaks markup, and `tracing::warn!`s with a stable `why=` field plus small byte counts (never raw model text or arguments). **A published spec overrides this rule (see goal 1):** if the model publishes a parser/regex that requires a fence, honor it — drop when that fence is missing, even though the value is delimiter-terminated — and document the divergence with the spec URL and a verbatim quote (e.g. MiniMax-M2.1 publishes `invoke_regex = re.compile(r"<invoke name=(.*?)</invoke>", re.DOTALL)`, which requires `</invoke>`, so a missing `</invoke>` yields no call). In the v1 batch parsers this rule is carried by the `allow_eof_recovery` flag (finalize / non-streaming only — streaming jails keep it `false`): `true` recovers a call whose outer end/wrapper marker is missing as long as the body still parses (and still drops a body truncated mid-value), `false` drops it. State the consequence in the fixture `reason:` — e.g. `allow_eof_recovery=true, so the complete-but-unterminated call is recovered` or `allow_eof_recovery=false (spec requires the fence), so the call is dropped`. Recovering an inner value whose own close tag is missing — the value is terminated by the next marker — is a separate value-capture concern, not this flag.
6. **Preserve as much of the original output as possible.** When the spec is silent, `normal_text` is the model output with only the recognized tool-call / reasoning markup spans removed — text before, between, and after calls is kept verbatim. Do not drop surrounding narration.
7. **Parsing is separate from validation.** Emit a tool call even when its function name is not in the request's tools list; the serving layer validates and rejects unknown tools. This matches vLLM.
8. **v1 and v2 can produce different argument values.** v1 uses the batch parser, while v2 owns its streaming conversion path and preserves exact argument text. `UnifiedEvent.arguments` remains a `serde_json::Value` projection and can change the lexical representation; use `tool_arguments_raw()` when the exact assembled JSON text matters. Legacy captures retain historical differences; current qualification uses the [latest parser tests](../../conformance/README.md#current-parser-tests-and-qualification).
9. **Mirror vLLM Rust's contract shape, not necessarily its output.** The types and method names mirror vLLM Rust so vLLM can adopt this crate, but Dynamo may intentionally diverge on output content (e.g. preserve surrounding text where vLLM drops it) — every such divergence carries a `reason:`.

The end-to-end workflow for adding a parser — fetch the chat template, identify the format, prefer an existing parser + config over new code, register, test, render — lives in the `tool-parser-generator` skill under `.agents/skills/tool-parser-generator/`. Adding a family to the **unified** path (one ordered reasoning + content + tool stream, rather than the reasoning-then-tool split) is a different workflow with its own tiers and traps — see [`UNIFIED_PORTING.md`](UNIFIED_PORTING.md). Reasoning parsers follow the same goals; see [`../v1/src/reasoning/README.md`](../v1/src/reasoning/README.md).

To supply or replace a family without forking, implement `UnifiedParser` and call `register_unified_parser("family", factory)` at startup. Unregistering restores the built-in. See [CUSTOM_PARSERS.md](CUSTOM_PARSERS.md) for the contract and conformance enrollment.

## Parser families

This is the single source of truth for the parser-family taxonomy — which grammar family a model belongs to and where the grammar is implemented. V1 owns its legacy batch grammar under `parsers/v1/src/tool_calling/`. V2 owns shared scanners and emitters under `parsers/v2/src/tool_calling/` and conversion helpers under `tool_calling/v1core`; it does not import the v1 crate. Pick the family first when adding a model; only write a new module when the grammar is genuinely new.

The legacy split pipeline runs reasoning extraction before tool extraction. Unified instead emits reasoning, visible text, and calls in source order from one parser. `ToolParser` remains a separate compatibility API returning `ToolParseResult`; inspect its registry rather than assuming that every Unified family has a tool-only entry.

Tool-call families:

| Family | Grammar | Batch impl (`parsers/v1/src/tool_calling/`) | Examples |
| -- | -- | -- | -- |
| **DSML** | `<｜DSML｜tool_calls>...` with typed `string="true\|false"` parameters | `dsml/parser.rs` | DeepSeek V3.2, V4 |
| **XML** | `<tool_call>...</tool_call>` with nested `<parameter>` / `<function>` (or special-token variants) | `xml/parser.rs` (generic) or own file per variant | qwen3_coder, nemotron_nano, minimax_m2, GLM-4.7/5.x via `glm47` (shared implementation), kimi_k2 (own file, special-token XML) |
| **JSON** | Start sentinel + JSON `{name, arguments}` (single object or array) | `json/base_json_parser.rs` (+ variant files) | deepseek_v3, deepseek_v3_1, hermes, nemotron_deci, jamba, mistral, phi4, llama3_json, qwen25 |
| **Harmony** | OpenAI Harmony token stream with `<\|channel\|>`, `<\|message\|>`, `<\|call\|>` | `harmony/harmony_parser.rs` (wraps external `openai_harmony` crate) | gpt-oss-20B / 120B |
| **Pythonic** | `[func_name(arg=value, ...)]` Python function-call syntax | `pythonic/pythonic_parser.rs` | some Llama variants |
| **Gemma 4** | Custom: `<\|tool_call>call:name{key:<\|"\|>val<\|"\|>}<tool_call\|>`, bare keys, custom string delimiter | `gemma4/parser.rs` (recursive-descent into `serde_json::Value`) | Google Gemma 4 thinking models |
| **ATEM** | Recipient-routed channels `<\|start\|>assistant to=<rcpt><\|message\|>...<\|eom\|>` whose tool bodies hold `<atem:invoke name="NAME"><atem:parameter name="KEY">VALUE</atem:parameter></atem:invoke>` | No v1 implementation; v2 `tool_calling/muse_glimmer.rs`, `tool_calling/muse_spark.rs` | Muse Glimmer, Muse Spark |

Reasoning families:

| Family | Grammar | Batch impl (`parsers/v1/src/reasoning/`) | Examples |
| -- | -- | -- | -- |
| **Basic (think-tag)** | `<think>...</think>` | `base_parser.rs` (`BasicReasoningParser`) | Qwen3, Nemotron, Kimi K2.5, DeepSeek R1 / V4, GLM-4.5+ |
| **Append-think** | `<think>...</think>` left inline as text, with `<think>` prefix on first chunk | `minimax_append_think_parser.rs` | MiniMax M2 |
| **Harmony channel** | Hidden `analysis` channel | `gpt_oss_parser.rs` (wraps external `openai_harmony`) | gpt-oss-20B / 120B |
| **Granite** | Custom start/end tokens | `granite_parser.rs` | IBM Granite |
| **Gemma 4 channel** | `<\|channel>thought\n...<channel\|>` with role-label prefix stripped | `gemma4_parser.rs` | Google Gemma 4 thinking models |
| **ATEM channel** | `<\|start\|>assistant to=self<\|message\|>...<\|eom\|>` — the recipient picks the channel, so there is no delimiter pair; Muse Spark additionally reserves `commentary` as visible text | No v1 implementation; v2 `unified/muse_glimmer.rs`, `unified/muse_spark.rs` | Muse Glimmer, Muse Spark |

The built-in Unified registry in `src/unified/mod.rs` contains `deepseek_v4`, `deepseek_v41`, `gemma4`, `glm47`, `qwen3` / `qwen3_coder`, `nemotron_v3` / `nemotron3`, `muse_glimmer`, `muse_spark` / `muse_spark_open`, `beam`, `kimi_k2`, `kimi_k3` / `kimi-k3`, and `gpt_oss` / `harmony`. `builtin_unified_families()` includes aliases; `canonical_unified_family()` resolves them. The separate tool-only registry is `src/tool_calling/registry.rs` and also includes MiniMax M2/M3; their presence there does not imply Unified support.

DeepSeek V4.1 uses Unified family `deepseek_v41` for its distinct reasoning and JSON-body behavior. The legacy tool-only `deepseek_v4` selector also accepts spaced V4.1 DSML tags; that compatibility is not a replacement for the Unified V4.1 factory.

## MiniMax M3

MiniMax M3 uses Unified family `minimax_m3` for reasoning and tool calls. Native structural tags require schemas the backend can represent without excluding valid argument values. Schemas allowing unrestricted string arguments require fallback: Dynamo uses guided JSON for required or named tool choices, while unforced auto requests remain unconstrained. Guided JSON preserves the requested schema; argument validation remains the caller’s responsibility.

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
├── unified/           # ordered event API, registry, shared routing, family factories
├── tool_calling/
│   ├── registry.rs    # compatibility tool-only registry
│   ├── scan.rs        # shared wrapped-block scanner
│   ├── v1core/        # v2-owned batch conversion helpers
│   └── *.rs           # family grammar builders and emitters
└── structural_tag/    # v2 structural-tag builders
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

Keep authored requests/GOLDEN in `conformance/fixtures-unified-v2/families/<family>/inputs_and_golden.yaml`. Latest parser tests read versionless recorded observations from `conformance/fixtures-unified-v2/families/<family>/<implementation>.yaml`. Contributors run latest tests, recapture affected latest snapshots through the [latest capture plan](../../docs/CONFORMANCE-LATEST-PLAN.md), and generate the latest report.

Retain `TOOLCALLING.*` archives as historical evidence. Latest-only generation does not read them or derive GOLDEN from recorded results.

## Historical Fixture Format

Historical archives use several schemas. Preserve their implementation keys, request bindings, and producer provenance; never relabel `dynamo_v2` as `dynamo_rust` or rewrite old observations into a current schema. See [fixture storage](../../conformance/fixtures/README.md) and the [capture plan](../../docs/CONFORMANCE-LATEST-PLAN.md).

## Adding A Day-0 Tool-Calling Parser

In order:

1. Read the model's tool-call output spec and its tokenizer / special-token behavior — token boundaries decide whether the parser is text- or token-native.
2. Inspect the vLLM **Rust** parser first: `Tool`, `ToolCallDelta`, `ToolParseResult`, and `ToolParser` are the API target (Rust vs. Rust). Do not shape the parser like vLLM Python wire deltas.
3. Inspect vLLM **Python** and **SGLang** for behavior and coverage — they are the peer references the matrix compares against.
4. Decide the parser family id and peer parser names; add a row to `conformance/utils/src/parser_families.yaml` (`vllm_python` / `vllm_rust` / `sglang_python` / `dynamo_v2` / `preferred_input`), AND declare the family's grammar tokens in the `markers:` section of the same file (`pairs` / `singletons` / `leak`; explicit `{}` for markup-less grammars). The `↯` leak detector and the popup token coloring are derived from that declaration — skipping it makes leaks render as clean cells and every token render as a red orphan, so review the declared markers alongside the latest qualification results.
5. Implement or extend one shared grammar owner and Unified factory using [UNIFIED_PORTING.md](UNIFIED_PORTING.md). Keep required tool-only compatibility as an event projection; do not introduce a second parser state machine.
6. Register the Unified factory through `unified_registry!` in `src/unified/mod.rs`. If existing callers require tool-only compatibility, register its projection through `family_registry!` in `src/tool_calling/registry.rs`. Declare token-input capabilities on the API that actually accepts them.
7. Add Rust unit tests for: one call, multiple calls, partial chunks, malformed recovery, `normal_text`, and EOF.
8. Author applicable Unified inputs and independent GOLDEN (see "Which Fixture Do I Edit?"). Inspect every registered family for shared scenarios and explain grammar or request-mode exclusions. Preserve published IDs.
9. Run the latest parser tests against the authored requests and independent GOLDEN, and fix parser mismatches.
10. Generate the [latest report](../../conformance/README.md#generate-the-latest-or-history-report), inspect the affected family over HTTP, and explain any deliberate authored contract change. Recapture affected latest snapshots; historical backfills are separate reviewed changes.

### Unified parser validation

Contributors edit the shared parser owner and authored inputs/GOLDEN, then run the latest tests:

```bash
cargo test --locked -p dynamo-conformance-fixtures-v2 --test unified_render -- --nocapture
cargo test --locked -p dynamo-conformance-fixtures-v2 --test unified_parity -- --nocapture
```

Generate the [latest report](../../conformance/README.md#generate-the-latest-or-history-report) and inspect affected inputs, initialization, chunks, GOLDEN, and observations over HTTP. Missing observations require recapturing the affected latest snapshots. Fix live mismatches or explain deliberate contract changes; never hide them with `reason:`, `unavailable:`, stale HTML, or unsupported GOLDEN edits.

## Which Fixture Do I Edit?

Use the [fixture files](#fixture-files-to-add) above for new families and scenarios. Preserve authored requests and independent GOLDEN; historical backfills are separate reviewed changes.

### Retained historical fixture paths

Historical batch, stream, batch-on-stream, and reasoning archives are retained under `conformance/fixtures/`; versioned Unified captures remain under `conformance/fixtures-unified-v2/`. See the [capture layout](../../conformance/README.md) for current ownership. These are retained observations, not seeds for new Unified GOLDEN or contributor capture requirements.

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
- The latest report is regenerated locally and affected cases are inspected for tool-call markup leaks, undeclared tokens, and mismatches against GOLDEN. Missing recorded observations are recaptured in the same PR.
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

Use the [current qualification instructions](../../conformance/README.md#current-parser-tests-and-qualification) for publisher/model/browser checks rather than retired stream-on-batch Python tests.

## Reasoning Migration Status

Unified already owns ordered reasoning, text, and tool events for its registered families. Retained v1 reasoning archives preserve their original implementation keys and observations; do not relabel them as new measurements. New scenarios use authored Unified inputs/GOLDEN and explicit recorded implementation results. Historical backfills are separate reviewed changes.

### Muse Spark Open structural tags

The `muse_spark_open` / `muse_spark` structural-tag builder constrains native
ATEM calls, including qualified tool names, raw scalar parameters, JSON-valued
containers, required/named choice, and single-call policy. It preserves the
model's reasoning header and allows one pre-tool commentary message. Auto choice can also constrain the
final answer inside its native `to=user` envelope.

The grammar must run from the first generated token. An external reasoning gate
that activates only after a tool channel opens cannot enforce forced choice and
is rejected. The grammar completes before the checkpoint's end-of-turn stop
token, allowing the engine to terminate rather than masking EOS. A serving adapter must configure this ownership separately; the builder's presence in this library does not establish that a pinned Dynamo release or backend supports it.

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
