# Unified Parser Cases (reasoning + content + tool calls, one ordered stream)

Reference taxonomy for the **unified** conformance surface: one parser owns the whole assistant-output grammar and emits ONE ordered event stream. Sibling stage docs: `REASONING_CASES.md` (reasoning only), `TOOLCALLING_CASES.md` / `TOOLCALLING_STREAMING_V2_CASES.md` (tool calls only). This surface is what those two cannot express — the ORDER between reasoning and tool calls, and reasoning that occurs *between* or *after* tool calls.

The golden corpus is authored by `conformance/utils/src/gen_unified_golden.py` (one scenario spec -> `conformance/unified/golden_spec/<family>.yaml` in the gitignored build tree); the committed, versioned `conformance/fixtures/unified/golden.tar.gz` shard is derived from it. The surface overview is `conformance/unified/README.md`.

## The oracle: GOLDEN is authored, not captured

The truth column (`golden:`) is what a **correct** UnifiedParser MUST emit, reasoned from the invariants and policies below — NOT captured from vLLM, Dynamo, or any implementation. Both engines are measured against it and both can diverge (vLLM has documented spec violations: truncated-tool hard-error, streamed-arg truncation, trailing-text suppression). Never regenerate `golden:` from an engine; it is versioned like code.

## Event schema

One ordered list per case:

```yaml
golden:
  - {kind: reasoning, text: "..."}     # private chain-of-thought
  - {kind: tool_call, name: "...", arguments: {...}}   # final typed args object
  - {kind: text, text: "..."}          # user-visible content
```

Comparison is ORDER-SENSITIVE on the ASSEMBLED list. Streaming delta granularity may differ across engines; the assembled event list is the invariant (same principle as the tool-call chunk sweep).

## Invariants (every correct implementation must satisfy)

- **I1 Faithful segmentation** — every model byte is exactly one of reasoning / visible text / tool-call structure / control marker. Markers are consumed; nothing else dropped or duplicated.
- **I2 Order preservation** — events in the order the model emitted the underlying content (reason -> call -> reason -> text stays in that order).
- **I3 No marker leakage** — control markers never appear inside a text or reasoning payload.
- **I4 Per-stream isolation** — for n>1, each choice's events depend only on that choice's bytes: `demux(parse(interleave(s0,s1)))[i] == parse(s_i)`.
- **I5 Chunk-invariance** — assembled list identical for any chunk splitting.
- **I6 Stream/batch parity** — whole-output parse assembles to the same list as streamed.
- **I7 Argument fidelity** — arguments are the model's actual args, typed per schema; no fabricated/dropped/reordered keys; a marker-looking substring INSIDE a JSON string value is data, preserved exactly.
- **I8 Coalescing** — adjacent same-kind events merge.

## Governing principle: best-effort error recovery

The parser recovers everything it can and NEVER drops valid text, leaks markup, or hard-errors on malformed/truncated input. Documented contract: `conformance/README.md` (v2 "preserves surrounding/inter-call prose... recovers bare calls v1 drops"; "dropping text, leaking markup, corrupting args" is a regression to FIX, not paper over) and `TOOLCALLING_CASES.md` batch.5.e / 5.g (drop only the unrecoverable partial call while earlier output stays recoverable; strip orphan close markers; do not leak). This principle resolves the policy calls below.

## Policy decisions

- **P1 Trailing text after the last tool call** -> emit as `text`. RESOLVED by best-effort recovery: trailing prose is arbitrary visible content and must be preserved (dropping it is a regression). vLLM's kimi config suppresses it -> vLLM red (LOSS).
- **P2 Truncated tool call at EOF** -> DROP the unrecoverable partial call, emit preceding reasoning/text cleanly, no error, no leaked markup. RESOLVED by best-effort recovery (TOOLCALLING batch.5.e). Dynamo drops -> correct; vLLM native gemma4 hard-errors -> red (ERROR).
- **P3 Empty arguments** -> `{}`.
- **P4 Structural whitespace** -> strip only tokenizer-structural whitespace bound to the marker grammar (e.g. gemma4 `thought\n`), preserve model-authored whitespace. Still a JUDGMENT call (needs an owner); best-effort recovery does not fully decide it.
- **P5 Implicit reasoning start** -> prompt-conditioned per family (forced-reasoning models start in reasoning with no `<think>`).
- **P6 Marker quoted in prose** -> counts only as a real control token; text-only input is best-effort (known limitation, not pass/fail).
- **P7 Nested channel markers (a marker of one channel inside another)** -> marker recognition is CHANNEL-SCOPED, and both directions follow the same best-effort-recovery rule (recover real structure, never leak markup, never drop a valid call):
  - Inside a **quoted tool-argument string value**, marker-looking bytes are DATA (I7). A reasoning marker there does NOT open a reasoning channel — it is the literal arg string (`reason_markup_in_arg`). A reasoning-first pipeline that extracts `<think>`/`<|channel>` before tool parsing corrupts the arg -> red (ARG_MISMATCH / MERGE).
  - A **well-formed tool-call envelope inside a reasoning span** is STRUCTURAL: break out of reasoning, emit the call, resume reasoning after its close (`tool_in_reason`). Leaking the raw `<|tool_call>...<tool_call|>` into `reasoning_content`, or dropping the call, is the regression -> red (LEAK). The asymmetry is deliberate: quote delimiters explicitly mark a data region, whereas a reasoning span is opaque text that can still contain recoverable structure.

> P1/P2 are RESOLVED by the documented best-effort-recovery contract above (not open product questions). P4 remains a product judgment; cases depending on it carry a `policy:` tag. P7 is RESOLVED by the same contract (no markup leak, no dropped call); current reasoning-first engines diverge, which is the gap it documents.

## Divergence classes (how a non-matching cell is colored)

`MATCH` (green) · `ORDER` / `MERGE` / `LOSS` (the unification gap) · `LEAK` (markup in text, `↯`) · `ARG_MISMATCH` / `WHITESPACE` (version drift) · `ERROR` (engine hard-errored where the spec expects graceful output).

## Quick reference — numbered taxonomy (`UNIFIED.<group>.<sub>`)

Case IDs use short `group.sub` labels (`1.a`, `2.b`, …) like the other suites; the scenario slug (the golden filename key) is shown in parentheses. **Groups 1–9 mirror the tool-calling STREAM taxonomy** (`TOOLCALLING.streamv2.N`) as reasoning-free unified cases — this surface subsumes STREAM. **Group 10** is the reasoning axis (`REASONING.*`). **Group 11 is UNIQUE to unified**: reasoning↔tool ORDER that neither STREAM (no reasoning) nor REASONING (no ordered tool events) can express. **Group 12** is adversarial nesting — a marker of one channel inside another (P7). 33 scenarios × 3 families (gemma4 / qwen3 / kimi_k2) = 99 cases.

### Group 1 — TC Single call
- **`1.a`** (`tool_only`) One tool call, no reasoning, no surrounding text. The tool suite's baseline.

### Group 2 — TC Multiple calls (streamv2.2)
- **`2.a`** (`two_calls`) Two distinct calls back-to-back, order preserved.
- **`2.b`** (`two_calls_same_name`) Two calls to the SAME function, different args — must not dedup or merge.

### Group 3 — TC No call (streamv2.3)
- **`3.a`** (`text_only`) Plain content, zero tool structure. No spurious call.

### Group 5 — TC Truncation / recovery (streamv2.5)
- **`5.a`** (`truncated_tool_eof`) EOF mid-call. Golden drops the partial, keeps preceding output (P2); vLLM Rust hard-errors (`ParsingFailed`). Class ERROR.
- **`5.b`** (`tool_no_close`) Complete call body but the close marker never arrives. Golden recovers the call at finish; vLLM Rust hard-errors. Class ERROR.
- **`5.c`** (`orphan_close_after_prose`) Orphan close marker after prose. Golden strips it; engines may leak. Class LEAK.

### Group 6 — TC Empty body (streamv2.6)
- **`6.a`** (`empty_args`) Call with `{}` arguments. Must emit the call with an empty object, not drop it.

### Group 7 — TC Argument fidelity (streamv2.7)
- **`7.a`** (`arg_unicode`) Non-ASCII argument value round-trips byte-exact (I7).
- **`7.b`** (`arg_marker_in_string`) A close-marker substring INSIDE a string arg is data, preserved exactly (I7). vLLM Rust truncates. Class ARG_MISMATCH.

### Group 8 — TC Content / narration position (streamv2.8)
- **`8.a`** (`text_before_tool`) Visible narration precedes the call.
- **`8.b`** (`trailing_text_after_tool`) Arbitrary prose AFTER the tool section (P1). vLLM suppresses it. Class LOSS.
- **`8.c`** (`text_sandwich`) text → call → text; both text spans survive in order.
- **`8.d`** (`text_between_calls`) call → text → call; the inter-call prose survives (v2 recovers what v1 drops).
- **`8.e`** (`narrated_calls`) Multiple calls with narration between each — `tool_call → text → tool_call → text → tool_call`. The agentic call/narrate/call pattern; every call and inter-call text span is its own ordered event.

### Group 10 — Reasoning span (`REASONING.*`)
- **`10.a`** (`reason_only`) Reasoning span, nothing else.
- **`10.b`** (`reason_then_content`) Reasoning then visible content, no call.
- **`10.c`** (`two_reason_spans`) Two reasoning spans separated by content. Batch reasoning merges them → Class MERGE.
- **`10.d`** (`reason_unterminated`) Stream ends inside reasoning; open reasoning promoted at finish.

### Group 11 — Reasoning ↔ tool interleaving (UNIQUE to unified; the unification gap)
- **`11.a`** (`reason_then_tool`) Reasoning fully precedes one call. Baseline ordering.
- **`11.b`** (`reason_after_tool`) Reasoning AFTER a call, then text (Example A). Class ORDER.
- **`11.c`** (`reason_interleaved`) reason → tool → reason → tool. Class MERGE.
- **`11.d`** (`reason_tool_text_reason_tool`) reason → tool → text → reason → tool. Class MERGE.
- **`11.e`** (`interstitial_text`) reasoning → visible text → call; the middle text survives in order.
- **`11.f`** (`content_then_reason_then_tool`) Content BEFORE reasoning, then a call. Class ORDER (Dynamo hoists reasoning).
- **`11.g`** (`content_then_reason`) content → reasoning → content. Class ORDER.
- **`11.h`** (`reason_tool_reason_tool_reason`) Each call wrapped by its own thought, trailing thought too. Class MERGE.
- **`11.i`** (`reason_between_calls`) call → reasoning → call; reasoning survives BETWEEN two calls. Class MERGE.
- **`11.j`** (`text_reason_tool_text_reason_tool`) Deep well-formed interleave — text → reason → tool → text → reason → tool; user text, reasoning, and calls all mix in one stream, every segment in order. Class MERGE (batch hoists both thoughts).

### Group 12 — Adversarial nesting (a marker of one channel inside another; P7)
- **`12.a`** (`reason_markup_in_arg`) "Tool call contains reasoning" — a reasoning-channel marker sits inside a quoted tool-arg VALUE. NOT a leak: an arg value is data bound for the function, not a rendered channel, so by I7 the parser preserves it byte-exact (the gemma4 native UnifiedParser confirms the golden exactly). A reasoning-first extractor lifts it out and corrupts the arg. Class ARG_MISMATCH / MERGE.
- **`12.b`** (`tool_in_reason`) "Reasoning contains tool call" — a well-formed tool-call envelope nested inside a reasoning span. OPPOSITE of 12.a: a reasoning span is opaque text (not a quoted data region), so a real tool-call marker inside it IS structural. Golden breaks out (reason → call → reason). Engines leak the tool markup into `reasoning_content` and drop the call. Class LEAK.
- **`12.c`** (`reason_markup_in_arg_with_text`) 12.a WITH visible narration before and after — all three channels at once (text / tool-call-with-markup-arg / text). Golden keeps text as text, the call clean, the markup byte-exact in the arg. Class ARG_MISMATCH / MERGE.
- **`12.d`** (`tool_in_reason_with_text`) 12.b WITH visible narration before and after — text → reason → call → reason → text. Golden breaks out and keeps the surrounding text; engines leak the nested markup. Class LEAK.

## Deferred (not in the U0 seed set)

- **n>1 interleave** (`UNIFIED.interleave_n2.*`, the Example-B n>1 LOSS case) needs a multi-choice interleaved driver (extends PR #135's tool-only lanes to carry reasoning state). Its golden is per-choice, a different shape than the single-stream cases here. Author with the n>1 lane.
