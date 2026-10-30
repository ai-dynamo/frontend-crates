# Porting a model family to the UnifiedParser

UnifiedParser uses one state machine to emit reasoning, visible text, and tools in source order. The legacy split pipeline extracts reasoning first and merges it at the front. See [UNIFIED_CASES.md](../../conformance/utils/lib/parsers/UNIFIED_CASES.md) for the behavioral differences.

For current families and aliases, see the [registry inventory](README.md#parser-families) and `unified_registry!` in `src/unified/mod.rs`. Extend the matching grammar owner: Beam uses `ChannelEnvelope`; Muse and Harmony use channel-specific owners. Serving support depends on the integrated crate version and adapter.

## What you get for free

`ScannerUnified` (`src/unified/mod.rs`) implements `NativeUnified` over a `WrappedBlockScanner` and family emitter. `GuidedRouted` wraps that native owner, stores request/guided state, and implements the shared `UnifiedParser` lifecycle: initialization, incremental parsing, ordinary or confirmed-EOS finalization, and reset/reuse. Extend the matching shared owner rather than adding a family-specific lifecycle.

Reuse these features from `src/tool_calling/scan.rs` and the guided router in `src/unified/mod.rs`:

- block open/close, bare-invoke recovery, orphan-close stripping
- chunk-boundary marker holdback (`marker_prefix_suffix_len`)
- reasoning open/close, and EOF promotion of an unterminated thought
- tool-open break-out from INSIDE a thought, and resume after the call closes (invariant `I3`; case `12-2`)
- guided-JSON decoding, including the reasoning envelope around the payload
- a grammar-aware invoke scan for families whose markers cannot delimit a call on their own (`InvokeScan`)
- an optional structural role label after the reasoning opener (`ReasoningSpec::start_label`)

The Gemma4 port added invoke scanning and optional reasoning labels. Extend these shared hooks before adding a family-specific drain loop.

Guided decoding is backend-controlled. The shared `GuidedRouted` wrapper owns `GuidedState` and uses the family's `GuidedReasoning` hooks (marker pair or channel); do not duplicate it per family.

## Conversion contract

A conversion replaces the family's v2 state machine. UnifiedParser and its shared scanner must own buffers, marker holdback, invokes, recovery, tool indices, reasoning state, initialization, finish, and reset.

- Delete the old family-specific drain loop and its private scanner state.
- Extract or extend one shared scanner builder, then have the Unified factory use it.
- If the public `ToolParser` registry still needs the family, keep a compatibility adapter only. It may translate `UnifiedParserEvent` events into `ToolParseResult`, but it must not parse DSML/model text itself or retain a second buffer, marker set, recovery policy, or argument decoder.
- Test the compatibility adapter against the UnifiedParser on the same complete input and every valid streaming split. The two API surfaces may regroup deltas only where the legacy wire contract requires it; their final text and tool calls must agree.

A factory or registry entry alone is not a conversion.

## Adding a family: parser edits

**1. Write `parsers/v2/src/unified/<family>.rs`.** Use `src/unified/qwen3.rs` as the wrapped-block reference for `ScannerUnified`, `GuidedRouted`, and guided-prefix resolution.

Copy the current `ReasoningSpec` constructor from source; it needs more than start/end markers. Channel families should extend their existing shared owner.

Expose the emitter and `<family>_scanner(tools)` builder as `pub(crate)`. Extract an inline builder first, then share it between the Unified factory and any remaining caller; do not copy the grammar spec.

**2. Add one line to `unified_registry!`** in `unified/mod.rs`:

```rust
unified_registry! {
    "qwen3" | "qwen3_coder" => qwen3::qwen3_unified,
    "<family>"              => <family>::<family>_unified,
}
```

Add aliases after `|`. The macro generates the constructor match and `REGISTERED_UNIFIED_FAMILIES` together.

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

The row feeds GOLDEN generation, capture harnesses, leak checks, and token coloring. `manifest_and_parser_registry_agree_on_native_families` checks it against the Rust registry.

Then add the family's inputs to `gen_unified_golden.py` and run the pipeline (see Checklist).

## Ranking: how much work is a given family

For wrapped-block families, first reuse `WrappedBlockScanner`, its invoke boundary/emitter hooks, and the guided router. Qwen, DeepSeek V4/V4.1, GLM, Gemma, and Kimi already have Unified factories; their original porting work is not a pending task. MiniMax M2/M3 have tool-only parsers and need a full conversion under the contract above. Channel grammars need their existing channel owner rather than a delimiter-pair approximation; Harmony now has a Unified implementation accepting text or token IDs, so inspect `input_capabilities()` before feeding input.

Families without a Unified registry entry, including Granite, Inkling, Mistral, Step, and legacy append-think, require new grammar/initialization work. A reasoning parser name in v1 alone does not make that family available through Unified.

## The reasoning half is the easy half

`ReasoningSpec` covers marker pairs, optional role labels, forced starts, and special-token preservation. Examples include `<think>` / `</think>`, MiniMax M3's `<mm:think>` / `</mm:think>`, and Mistral's `[THINK]` markers.

`drain_reasoning` derives tool break-out from `spec.block_starts` and `spec.invoke_start`; no separate v1-style `with_tool_start_token` field is needed. Unified suspends reasoning during a call and resumes afterward (case `12-2`); V1 exits reasoning at the tool marker.

Families that genuinely do not fit a marker pair:

- **harmony / gpt_oss** — the existing Unified Harmony owner handles channels and token IDs.
- **granite** — two alternative starts and two alternative ends; needs a list on both sides.
- **inkling** — a six-state machine over eight markers.
- **muse_glimmer** — no pair exists; the recipient in a dynamic header picks the channel.
- **muse_spark** — inherits the Muse Glimmer channel grammar and additionally routes the reserved `commentary` recipient to visible text.

Gemma4 uses `ReasoningSpec::start_label` for the optional `thought\n` after `<|channel>`. Folding the label into the opener would miss a bare `<|channel>` and leak markup. Extend shared specs for grammar variants instead of narrowing them to the corpus; Granite needs the same approach for alternative markers.

## Traps

**Delimit once.** After the scanner bounds an invoke, the emitter converts its arguments; it must not extract the call again. Re-wrapping Qwen bodies let embedded `</tool_call>` truncate argument values. Gemma4's whole-message extractor also rejected invokes missing wrapper markers. Use body-level conversion (`parse_tool_call_block`, `parse_one_tool_call_gemma4`) and check every emitter and compatibility projection against `7-2` (`arg_marker_in_string`).

**Dangling-end recovery differs.** Without an opener, the shared scanner strips the reasoning closer and returns preceding bytes as text; V1's `with_dangling_end_recovery` treats them as reasoning. This affects MiniMax M3 and Kimi K3. Pass `UnifiedParserStartingState::Reasoning` when the prompt already opened reasoning.

**Guided JSON needs reasoning hooks.** Without marker-pair or channel hooks from `guided_reasoning()`, `initialize_request` rejects `GuidedJson` before mutating parser state.

**Use guided payloads in guided tests.** `GuidedJson` constrains the tool payload to bare JSON; reasoning keeps its native grammar. Render shared guided inputs through `every_family` in `gen_unified_golden.py`; do not maintain native-markup substitutes per family.

**Test serving initialization too.** After prompt rendering and backend resolution, build one `UnifiedParserInit` containing prompt tokens, starting state, tool-output mode, and invalid-payload policy. Keep `Reject` unless the serving integration supports recovery; the conformance harness's `RecoverAsText` is not a serving default.

## Checklist

1. Extract `<family>_scanner(tools)` and make it the only grammar/state owner. Delete the existing tool-only drain state; if the `ToolParser` registry needs compatibility, replace it with a thin UnifiedParser event projection. Run direct compatibility-projection regressions as well as the latest Unified tests; preserve existing tool-only output unless the authored contract deliberately changes.
2. Add `unified/<family>.rs` with the factory and matching reasoning/channel hooks.
3. Add one line to `unified_registry!` in `unified/mod.rs`.
4. Add the `unified:` row in `conformance/utils/src/parser_families.yaml` — use `native: true` when the factory is implemented and remove annotations that still claim split-path behavior.
5. Add the family's inputs to `conformance/utils/src/gen_unified_golden.py` so every scenario gets one in that grammar.
6. Run the [current parser tests and qualification](../../conformance/README.md#current-parser-tests-and-qualification), including the latest `unified_render` and `unified_parity` targets described in [README.md](README.md#unified-parser-validation).
7. Check argument fidelity (`7-2`), adversarial nesting (`12-1`/`12-2`), every valid UTF-8 split, prompt-prefilled channels, guided modes, `finish`, reset/reuse, and compatibility-projection parity.
8. Generate the [latest report](../../conformance/README.md#generate-the-latest-or-history-report) with the Rust publisher, refresh `conformance/CONFORMANCE_v2.html`, and inspect affected inputs, GOLDEN, recorded observations, and per-chunk details over HTTP. Missing output is missing evidence; report it.
9. Recapture affected latest snapshots through the [latest capture plan](../../docs/CONFORMANCE-LATEST-PLAN.md), preserving request bindings and producer provenance. Historical backfills, corrections, and peer recaptures use normal PR review; historical capture/package commands are not a routine parser-change prerequisite. Do not bump dependencies.
