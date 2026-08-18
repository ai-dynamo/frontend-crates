# Porting a model family to the UnifiedParser

The unified parser is ONE state machine per stream that owns reasoning, visible content, and tool calls, and emits ONE ordered event list. The split path Dynamo still serves for most families runs the v1 reasoning parser over the whole stream first, then a v2 tool parser on the leftover — which cannot represent WHERE reasoning happened, so every thought is hoisted to the front and merged. See [`../../conformance/utils/lib/parsers/UNIFIED_CASES.md`](../../conformance/utils/lib/parsers/UNIFIED_CASES.md) for what that costs, case by case.

This doc is for adding a family to the unified path. `qwen3` is on it today; `gemma4` follows in the stacked PR, and the machinery it needed is already shared here.

## What you get for free

`ScannerUnified` (`src/unified/mod.rs`) is the shared body. It is generic over the emitter and holds a `WrappedBlockScanner`, and the whole `UnifiedParser` impl is family-blind: `initialize`, `initialize_request`, `parse_into`, `finish`, `reset`.

Everything below is already generic in `src/tool_calling/scan.rs` — do NOT reimplement any of it per family:

- block open/close, bare-invoke recovery, orphan-close stripping
- chunk-boundary marker holdback (`marker_prefix_suffix_len`)
- reasoning open/close, and EOF promotion of an unterminated thought
- tool-open break-out from INSIDE a thought, and resume after the call closes (invariant `I3`; case `12.b`)
- guided-JSON decoding, including the reasoning envelope around the payload
- a grammar-aware invoke scan for families whose markers cannot delimit a call on their own (`InvokeScan`)
- an optional structural role label after the reasoning opener (`ReasoningSpec::start_label`)

The last two are the gemma4 port's contribution. Reach for them before writing a
bespoke drain — the whole point of Tier C was to find out whether the shared loop
could be *extended* to the hardest family rather than forked for it. It could.

Guided decoding is a BACKEND feature, not a model feature — any family can be served with it — which is why `GuidedState` lives on the shared type and is parameterized on the family's `ReasoningSpec` markers rather than on any one grammar. A family that joins the unified path inherits it.

## Adding a family: two edits

**1. Write `parsers/v2/src/unified/<family>.rs`** — a factory over the shared `ScannerUnified`. `unified/qwen3.rs` is the reference; its entire non-test body is two consts and an 11-line factory.

```rust
pub(crate) fn <family>_unified(tools: &[Tool]) -> Box<dyn UnifiedParser> {
    Box::new(ScannerUnified::new(<family>_scanner(tools).with_reasoning(
        ReasoningSpec { start: "<think>", end: "</think>", forced_start: false },
    )))
}
```

It needs two things from the tool-only side, both `pub(crate)`: the emitter type, and a `<family>_scanner(tools)` builder. **Extract that builder first if it does not exist.** Most families construct their scanner inline inside `new()`, so the unified factory cannot reuse it without copying the spec — and a copied spec is a spec that drifts. Have BOTH callers use the one builder.

**2. Add one line to `unified_registry!`** in `unified/mod.rs`:

```rust
unified_registry! {
    "qwen3" | "qwen3_coder" => qwen3::qwen3_unified,
    "<family>"              => <family>::<family>_unified,
}
```

Aliases after the `|` — use them when the corpus name differs from the tool registry name. The macro generates both the constructor match and `REGISTERED_UNIFIED_FAMILIES`, so they cannot disagree.

That is the whole parser change. There is no separate const to update, and no registry list anywhere else in this crate.

## Adding a family to the conformance tab: one row

`conformance/utils/src/parser_families.yaml`, under `unified:`:

```yaml
  <family>:
    registry: <family>          # key into families:/markers: — omit if identical
    native: true                # false while it still runs on the SPLIT path
    reasoning_parser: <v1 name> # used by the split path
    tool_parser: <v2 name>      # used by the split path
    golden_spec: <family>.yaml
    leak_markers: []            # markup the shared leak list cannot see
```

That row feeds the golden generator, both capture harnesses, the leak check and the colorizer. `manifest_and_parser_registry_agree_on_native_families` fails if this row and `unified_registry!` disagree in either direction, so the two declarations stay honest.

Then add the family's inputs to `gen_unified_golden.py` and run the pipeline (see Checklist).

## Ranking: how much work is a given family

**Tier A — factory plus boilerplate.** Already on `WrappedBlockScanner`: `kimi_k2`, `minimax_m2`, `minimax_m3`. Widen the emitter, extract the scanner builder, add the factory and registry lines. Most of the effort is conformance, not code — the golden corpus for `kimi_k2` already exists.

**Tier B — generalize the scanner, no new state machine.** `deepseek_v4` (dsml) and `glm47` have bespoke drain loops. dsml's grammar maps onto `WrappedBlockSpec` almost directly; its only special part is an early-name state. glm47 needs `WrappedBlockSpec.invoke_start` to become optional, because its block IS the invoke (`<tool_call>NAME<arg_key>…</tool_call>`) with no inner marker. Tier B touches shared code, so every Tier A family regression-tests with it.

**Tier C — real new machinery.** `gemma4` is DONE, and what it needed is now shared. Its end marker can legitimately appear inside a `<|"|>`-delimited string value, so a plain `find` cuts the value (`I7`, case `7.b`); the same string rule makes `call:` ambiguous with the English word and lets a body be complete before its closer streams. All three are answered by one `InvokeScan` hook rather than a bespoke drain — see `tool_calling/gemma4.rs`. `harmony` is still Tier C and is not a marker scan at all: it routes by channel and depends on token IDs, and `UnifiedParser` has no `push_tokens`, so porting it means extending the trait, not adding a factory.

**Not portable yet.** `granite`, `inkling`, `mistral`, `step3`, `kimi_k3`, `minimax_append_think` have v1 reasoning parsers but no v2 tool parser. `ScannerUnified` has no reasoning-only shape, so there is nothing to hang them on until a tool parser exists.

## The reasoning half is the easy half

This is the opposite of what it looks like. `ReasoningSpec` is just `{start, end, forced_start}` — a marker pair plus a flag — and most families fit it directly: `<think>`/`</think>` for qwen3, kimi, minimax_m2, deepseek, glm45; `<mm:think>` for minimax_m3; `[THINK]` for mistral.

You do NOT need a new field for v1's `with_tool_start_token`: `drain_reasoning` already derives tool break-out from `spec.block_starts` and `spec.invoke_start`. The unified semantics are strictly better — v1 exits reasoning permanently at a tool marker, unified suspends and RESUMES after the call closes, which is what case `12.b` requires.

Families that genuinely do not fit a marker pair:

- **harmony / gpt_oss** — no pair exists; channel routing plus token-ID matching.
- **granite** — two alternative starts and two alternative ends; needs a list on both sides.
- **inkling** — a six-state machine over eight markers.

**gemma4 used to be on that list** and no longer is. Its opener is `<|channel>` plus a
`thought\n` role label, and the label is OPTIONAL. Folding it into `start` parses the
whole corpus — which is why it looks acceptable — but then a bare `<|channel>` matches
nothing and its markup surfaces as visible text, a leak (`I3`) in exactly the case the
narrowing was supposed to tolerate. `ReasoningSpec::start_label` keeps the label
structural and optional, so policy `P4` is IMPLEMENTED rather than deferred. Take the
same route for `granite`: extend the spec, do not narrow the grammar to the corpus.

## Traps

**Audit the emitter for `I7` before claiming the goldens.** qwen3's emitter was changed to call `parse_tool_call_block` directly because re-wrapping the body in `<tool_call>` made the batch parser truncate a value at an embedded `</tool_call>` — an argument value is DATA and must survive byte-exact. gemma4 hit the same wall from the other side: handing its block back to the whole-message extractor re-derived bounds the scanner had already resolved, and that extractor also refuses a call missing BOTH its opener (consumed as block markup) and its closer (never streamed), which is exactly case `5.b`. Its emitter now calls `parse_one_tool_call_gemma4` on the delimited invoke. **Rule: once the scanner has delimited an invoke, the emitter TYPES it — it never re-finds it.** The `minimax_m2`, `minimax_m3` and `kimi_k2` emitters still re-wrap. That is not confirmed broken, it is unaudited; case `7.b` (`arg_marker_in_string`) is the one that catches it — and it is not theoretical: vLLM 0.25.1's Python gemma4 parser truncates that value to `git log ` in the live capture.

**Dangling-end recovery differs.** A `</think>` with nothing open is stripped by the shared scanner and the preceding bytes become TEXT; v1's `with_dangling_end_recovery` classifies them as REASONING. Affects `minimax_m3` and `kimi_k3`. Passing `UnifiedParserStartingState::Reasoning` resolves it, which means the caller has to actually pass it — see below.

**Guided JSON needs a reasoning-aware scanner.** A family registered without a `ReasoningSpec` cannot support `GuidedJson`; `initialize_request` rejects that `UnifiedParserInit` before mutating parser state.

**A guided-decoding case whose input is native markup tests nothing.** Guided
decoding constrains the model to bare JSON, so the payload is grammar-independent and
the family's own markup can never appear in it. gemma4 and kimi_k2 carried NATIVE
markup under `init.tool_output_mode=GuidedJson` for six scenarios — green, because the
families had no unified parser to run them, and unfixable-looking the moment one did.
Render guided inputs once for every family (`every_family` in `gen_unified_golden.py`),
not per family.

**Nothing outside `parsers/v2` and `conformance/` consumes `UnifiedParser` yet.** The conformance harness builds `UnifiedParserInit` with `RecoverAsText`; a serving integration should build the same one owned object after prompt rendering and backend request resolution, leaving the default invalid-guided-payload policy at `Reject`. This keeps prompt tokens, starting state, output mode, and failure policy under one initialization owner. Test that object at the serving boundary, not only inside the parser.

## Checklist

1. Extract `<family>_scanner(tools)`; point the existing tool-only parser at it too. If that changes tool-only behavior, `parity_toolcalling_stream` says so — a clean run there is what proves the port was a refactor and not a rewrite.
2. Add `unified/<family>.rs` with the factory and its `ReasoningSpec`.
3. Add one line to `unified_registry!` in `unified/mod.rs`.
4. Add the `unified:` row in `conformance/utils/src/parser_families.yaml` — `native: true` once the factory exists, or `a_unified_family_is_never_annotated_as_diverging` fails with annotations still claiming the split path's behavior.
5. Add the family's inputs to `conformance/utils/src/gen_unified_golden.py` so every scenario gets one in that grammar.
6. Run the harness, then `explode` → `package` → render. The capture drift guard fails if the committed shard disagrees with the live parser.
7. Check `7.b` and `12.a`/`12.b` specifically — argument fidelity and adversarial nesting are where a re-wrapping emitter and a reasoning-first assumption break.
8. Re-capture the PEERS if any input changed. Peer shards are keyed by case id, so a changed input leaves their cells showing output captured from text that no longer exists. `vllm_python` + `sglang_python` run in their containers; `vllm_rust` needs a vLLM source tree on the host.
9. Verify what the page SHOWS, not just what the model contains. A cell can hold the right data and render nothing.

Steps 1-4 are the port. Step 5 is the bulk of the work today — one input per scenario per family, ~23 strings. That does not scale to many families and is tracked separately: the fix is a per-family grammar spec that RENDERS the canonical scenarios instead of authoring them by hand.
