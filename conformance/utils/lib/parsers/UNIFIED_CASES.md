# Unified Parser Cases (reasoning + content + tool calls, one ordered stream)

Reference taxonomy for the **unified** conformance surface: one parser owns the whole assistant-output grammar and emits ONE ordered event stream. Sibling stage docs: `REASONING_CASES.md` (reasoning only), `TOOLCALLING_CASES.md` / `TOOLCALLING_STREAMING_V2_CASES.md` (tool calls only). This surface is what those two cannot express — the ORDER between reasoning and tool calls, and reasoning that occurs *between* or *after* tool calls.

The golden corpus is authored by `conformance/utils/src/gen_unified_golden.py` (one scenario spec -> `conformance/unified/golden_spec/<family>.yaml` in the gitignored build tree); the committed, versioned `conformance/fixtures/unified/golden.tar.gz` shard is derived from it.

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

The parser recovers everything it can and NEVER drops valid text, leaks markup, or hard-errors on malformed/truncated input. Documented contract: `conformance/README.md` (v2 "preserves surrounding/inter-call prose... recovers bare calls v1 drops"; "dropping text, leaking markup, corrupting args" is a regression to FIX, not paper over) and `TOOLCALLING_CASES.md` TOOLCALLING.batch.5.e / TOOLCALLING.batch.5.g (drop only the unrecoverable partial call while earlier output stays recoverable; strip orphan close markers; do not leak). This principle resolves the policy calls below.

## Policy decisions

- **P1 Trailing text after the last tool call** -> emit as `text`. RESOLVED by best-effort recovery: trailing prose is arbitrary visible content and must be preserved (dropping it is a regression). vLLM's kimi config suppresses it -> vLLM red (LOSS).
- **P2 Truncated tool call at EOF** -> DROP the unrecoverable partial call, emit preceding reasoning/text cleanly, no error, no leaked markup. RESOLVED by best-effort recovery (TOOLCALLING.batch.5.e). Dynamo drops -> correct. The two vLLM implementations fail differently, both confirmed by live 0.25.1 capture: the native Rust `Gemma4UnifiedParser` returns `ParsingFailed { "incomplete Gemma4 tool call" }` -> red (ERROR), while the Python parser DISPATCHES the partial call with its truncated arguments (`{city: "Par"}`) — worse for a side-effecting action, since the client executes a call the parser never finished reading.
- **P3 Empty arguments** -> `{}`.
- **P4 Structural whitespace** -> strip only tokenizer-structural whitespace bound to the marker grammar (e.g. gemma4 `thought\n`), preserve model-authored whitespace. RESOLVED for gemma4 by `ReasoningSpec::start_label`: the role label is consumed when present and TOLERATED when absent, so `<|channel>thoughtful musing<channel|>` keeps its first word and a bare `<|channel>` still opens a thought instead of leaking as text. Folding the label into the opener would have passed this corpus and broken both.
- **P5 Implicit reasoning start** -> prompt-conditioned per family (forced-reasoning models start in reasoning with no `<think>`).
- **P6 Marker quoted in prose** -> counts only as a real control token; text-only input is best-effort (known limitation, not pass/fail).
- **P7 Nested channel markers (a marker of one channel inside another)** -> marker recognition is CHANNEL-SCOPED, and both directions follow the same best-effort-recovery rule (recover real structure, never leak markup, never drop a valid call):
  - Inside a **quoted tool-argument string value**, marker-looking bytes are DATA (I7). A reasoning marker there does NOT open a reasoning channel — it is the literal arg string (`reason_markup_in_arg`). A reasoning-first pipeline that extracts `<think>`/`<|channel>` before tool parsing corrupts the arg -> red (ARG_MISMATCH / MERGE).
  - A **well-formed tool-call envelope inside a reasoning span** is STRUCTURAL: break out of reasoning, emit the call, resume reasoning after its close (`tool_in_reason`). Leaking the raw `<|tool_call>...<tool_call|>` into `reasoning_content`, or dropping the call, is the regression -> red (LEAK). The asymmetry is deliberate: quote delimiters explicitly mark a data region, whereas a reasoning span is opaque text that can still contain recoverable structure.

> P1/P2 are RESOLVED by the documented best-effort-recovery contract above (not open product questions). P4 is now resolved in code for gemma4 (optional role label) and stays a judgment call for any family that adds structural whitespace of a different shape; cases depending on it carry a `policy:` tag. P7 is RESOLVED by the same contract (no markup leak, no dropped call); the families still on the split path diverge, which is the gap it documents.

## Divergence classes (how a non-matching cell is colored)

`MATCH` (green) · `ORDER` / `MERGE` / `LOSS` (the unification gap) · `LEAK` (markup in text, `↯`) · `ARG_MISMATCH` / `WHITESPACE` (version drift) · `ERROR` (engine hard-errored where the spec expects graceful output).

The Dynamo column is a per-family MIXTURE: only `qwen3` runs the native `UnifiedParser` (the sole entry in `unified_registry!`); `gemma4` and `kimi_k2` still run the v1-reasoning + v2-tool split and carry the gap. Every remaining red Dynamo cell in this tab is a split-path cell.

## Quick reference — numbered taxonomy (`UNIFIED.<group>.<sub>`)

Case IDs use short `group.sub` labels (`1.a`, `2.b`, …) like the other suites; the scenario slug (the golden filename key) is shown in parentheses. **Groups 1–9 mirror the tool-calling STREAM taxonomy** (`TOOLCALLING.streamv2.N`) as reasoning-free unified cases — this surface subsumes STREAM. **Group 10** is the reasoning axis (`REASONING.*`). **Group 11 is UNIQUE to unified**: reasoning↔tool ORDER that neither STREAM (no reasoning) nor REASONING (no ordered tool events) can express. **Group 12** is adversarial nesting — a marker of one channel inside another (P7). **Groups 30+ are REQUEST-SCOPED modes** — what the serving layer told the parser about this request, rather than what the model emitted; they use PAIRED TENS, `X0` for a mode's happy path and `X1` for its malformed counterpart, so a recovery case never sorts next to the baseline it contrasts with. 80 scenarios × 3 families (gemma4 / qwen3 / kimi_k2) = 240 cases. These counts come from the generator's `CLEAN + EDGE` lists, not from the taxonomy map — the map reserves names the generator does not emit, which is how a stale denominator survived here before.

### Group 1 — TC Single call
- **`1.a`** (`tool_only`) One tool call, no reasoning, no surrounding text. The tool suite's baseline.

### Group 2 — TC Multiple calls (TOOLCALLING.streamv2.2)
- **`2.a`** (`two_calls`) Two distinct calls back-to-back, order preserved. This is also covered in: TOOLCALLING.streamv2.2.a.
- **`2.b`** (`two_calls_same_name`) Two calls to the SAME function, different args — must not dedup or merge. This is also covered in: TOOLCALLING.streamv2.2.d.

### Group 3 — TC No call (TOOLCALLING.streamv2.3)
- **`3.a`** (`text_only`) Plain content, zero tool structure. No spurious call. This is also covered in: TOOLCALLING.streamv2.3. No e2e case has this shape: Qwen3.6 always emits a reasoning span, so the plain-content case is corpus-only.

### Group 5 — TC Truncation / recovery (TOOLCALLING.streamv2.5)
- **`5.a`** (`truncated_tool_eof`) EOF mid-call. Golden drops the partial, keeps preceding output (P2); vLLM Rust hard-errors (`ParsingFailed`). Class ERROR.
- **`5.b`** (`tool_no_close`) Complete call body but the close marker never arrives. Golden recovers the call at finish; vLLM Rust hard-errors. Class ERROR. This is also covered in: TOOLCALLING.streamv2.5.a.
- **`5.c`** (`orphan_close_after_prose`) Orphan close marker after prose. Golden strips it; engines may leak. Class LEAK.

### Group 6 — TC Empty body (TOOLCALLING.streamv2.6)
- **`6.a`** (`empty_args`) Call with `{}` arguments. Must emit the call with an empty object, not drop it. This is also covered in: TOOLCALLING.streamv2.6.a.

### Group 7 — TC Argument fidelity (TOOLCALLING.streamv2.7)
- **`7.a`** (`arg_unicode`) Non-ASCII argument value round-trips byte-exact (I7). This is also covered in: TOOLCALLING.streamv2.7.b.
- **`7.b`** (`arg_marker_in_string`) A close-marker substring INSIDE a string arg is data, preserved exactly (I7). vLLM Rust truncates. Class ARG_MISMATCH.

### Group 8 — TC Content / narration position (TOOLCALLING.streamv2.8)
- **`8.a`** (`text_before_tool`) Visible narration precedes the call. This is also covered in: TOOLCALLING.streamv2.8.a.
- **`8.b`** (`trailing_text_after_tool`) Arbitrary prose AFTER the tool section (P1). vLLM suppresses it. Class LOSS.
- **`8.c`** (`text_sandwich`) text → call → text; both text spans survive in order. This is also covered in: TOOLCALLING.streamv2.8.c.
- **`8.d`** (`text_between_calls`) call → text → call; the inter-call prose survives (v2 recovers what v1 drops). This is also covered in: TOOLCALLING.streamv2.8.d.
- **`8.e`** (`narrated_calls`) Multiple calls with narration between each — `tool_call → text → tool_call → text → tool_call`. The agentic call/narrate/call pattern; every call and inter-call text span is its own ordered event.

### Group 10 — Reasoning span (`REASONING.*`)
- **`10.a`** (`reason_only`) Reasoning span, nothing else. This is also covered in: REASONING.batch.2.a.
- **`10.b`** (`reason_then_content`) Reasoning then visible content, no call. This is also covered in: e2e case-0001-chinese_arithmetic__non-stream-budget_capped.json (+ 42 more: every `reasoning/core`, `reasoning/complex` and `reasoning/history` case, `tool_none_arithmetic__*`, and the SECOND step of both `lifecycle_*` — each with its `-budget_unlimited` pair).
- **`10.c`** (`two_reason_spans`) Two reasoning spans separated by content. Batch reasoning merges them → Class MERGE. This is also covered in: REASONING.batch.6.a.
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

## End-to-end test cases (`End-to-end:` tags)

Cases tagged `End-to-end:` name the corresponding end-to-end test case(s) in the Qwen3.6 run captured for PR #163 (`qwen36_pr163_test_cases.html` — 49 distinct cases, each run under two thinking-budget variants, at worker stream intervals 20 and 1). The two surfaces answer different questions and neither replaces the other:

- **This corpus** is authored and hermetic. It feeds an exact byte string to the parser and asserts the exact event list, so a regression names the grammar construct that broke. It cannot tell you whether a real model ever emits that string.
- **The e2e cases** are captured. They send real requests to a real worker and only check the final response, so they prove the path works end to end — but a failure there implicates the whole stack, and it only covers shapes the model happened to produce.

So an `End-to-end:` tag means "a real model exercised this construct", and its ABSENCE is the interesting signal: it marks a construct this corpus pins that no e2e case reaches. Groups 31, 41 and 51 are entirely untagged by design — malformed guided output, redundant openers and truncated prefills are what a backend produces when something goes wrong, and a healthy worker will not produce them on demand.

Known gaps in the other direction — end-to-end test cases with no corpus analogue:

- `lifecycle_*` are 2-step: the tool result is fed back and the model called again. Multi-turn is a frontend/templating concern; a single-stream parser corpus cannot express it.
- `history_*` cover conversation-history handling, likewise above this layer.
- The `reasoning/core` and `reasoning/complex` cases (32 of the 49) vary the PROMPT, not the output grammar. They exercise the reasoning parser end to end but map onto the same handful of reasoning-span constructs, so tagging each one would add noise, not coverage.

### Artifact index

An `End-to-end:` tag names the end-to-end test case and its artifact index. Each index is TWO JSON artifacts — the same case run under both thinking-budget variants — so `e2e case-0047` means both `end-to-end case-0047-*` files below. The report embeds every case inline in `const REPORT`, so the JSON files are provenance labels, not inputs; they live with whoever ran the harness, not in this repo.

| Case | e2e case | Artifact JSON file |
|---|---|---|
| `10.b` | `chinese_arithmetic__non-stream` | `end-to-end case-0001-chinese_arithmetic__non-stream-budget_capped.json` |
| `10.b` | `chinese_arithmetic__non-stream` | `end-to-end case-0001-chinese_arithmetic__non-stream-budget_unlimited.json` |
| `10.b` | `chinese_arithmetic__stream` | `end-to-end case-0002-chinese_arithmetic__stream-budget_capped.json` |
| `10.b` | `chinese_arithmetic__stream` | `end-to-end case-0002-chinese_arithmetic__stream-budget_unlimited.json` |
| `10.b` | `compare_fractions__non-stream` | `end-to-end case-0003-compare_fractions__non-stream-budget_capped.json` |
| `10.b` | `compare_fractions__non-stream` | `end-to-end case-0003-compare_fractions__non-stream-budget_unlimited.json` |
| `10.b` | `compare_fractions__stream` | `end-to-end case-0004-compare_fractions__stream-budget_capped.json` |
| `10.b` | `compare_fractions__stream` | `end-to-end case-0004-compare_fractions__stream-budget_unlimited.json` |
| `10.b` | `history_not_preserved__non-stream` | `end-to-end case-0013-history_not_preserved__non-stream-budget_capped.json` |
| `10.b` | `history_not_preserved__non-stream` | `end-to-end case-0013-history_not_preserved__non-stream-budget_unlimited.json` |
| `10.b` | `history_not_preserved__stream` | `end-to-end case-0014-history_not_preserved__stream-budget_capped.json` |
| `10.b` | `history_not_preserved__stream` | `end-to-end case-0014-history_not_preserved__stream-budget_unlimited.json` |
| `10.b` | `history_preserved_addition__non-stream` | `end-to-end case-0015-history_preserved_addition__non-stream-budget_capped.json` |
| `10.b` | `history_preserved_addition__non-stream` | `end-to-end case-0015-history_preserved_addition__non-stream-budget_unlimited.json` |
| `10.b` | `history_preserved_addition__stream` | `end-to-end case-0016-history_preserved_addition__stream-budget_capped.json` |
| `10.b` | `history_preserved_addition__stream` | `end-to-end case-0016-history_preserved_addition__stream-budget_unlimited.json` |
| `10.b` | `history_preserved_codeword__non-stream` | `end-to-end case-0017-history_preserved_codeword__non-stream-budget_capped.json` |
| `10.b` | `history_preserved_codeword__non-stream` | `end-to-end case-0017-history_preserved_codeword__non-stream-budget_unlimited.json` |
| `10.b` | `history_preserved_codeword__stream` | `end-to-end case-0018-history_preserved_codeword__stream-budget_capped.json` |
| `10.b` | `history_preserved_codeword__stream` | `end-to-end case-0018-history_preserved_codeword__stream-budget_unlimited.json` |
| `10.b` | `history_unicode__non-stream` | `end-to-end case-0019-history_unicode__non-stream-budget_capped.json` |
| `10.b` | `history_unicode__non-stream` | `end-to-end case-0019-history_unicode__non-stream-budget_unlimited.json` |
| `10.b` | `history_unicode__stream` | `end-to-end case-0020-history_unicode__stream-budget_capped.json` |
| `10.b` | `history_unicode__stream` | `end-to-end case-0020-history_unicode__stream-budget_unlimited.json` |
| `10.b` | `logic_syllogism__non-stream` | `end-to-end case-0021-logic_syllogism__non-stream-budget_capped.json` |
| `10.b` | `logic_syllogism__non-stream` | `end-to-end case-0021-logic_syllogism__non-stream-budget_unlimited.json` |
| `10.b` | `logic_syllogism__stream` | `end-to-end case-0022-logic_syllogism__stream-budget_capped.json` |
| `10.b` | `logic_syllogism__stream` | `end-to-end case-0022-logic_syllogism__stream-budget_unlimited.json` |
| `10.b` | `long_context_retrieval__non-stream` | `end-to-end case-0023-long_context_retrieval__non-stream-budget_capped.json` |
| `10.b` | `long_context_retrieval__non-stream` | `end-to-end case-0023-long_context_retrieval__non-stream-budget_unlimited.json` |
| `10.b` | `long_context_retrieval__stream` | `end-to-end case-0024-long_context_retrieval__stream-budget_capped.json` |
| `10.b` | `long_context_retrieval__stream` | `end-to-end case-0024-long_context_retrieval__stream-budget_unlimited.json` |
| `10.b` | `minutes_to_seconds__non-stream` | `end-to-end case-0025-minutes_to_seconds__non-stream-budget_capped.json` |
| `10.b` | `minutes_to_seconds__non-stream` | `end-to-end case-0025-minutes_to_seconds__non-stream-budget_unlimited.json` |
| `10.b` | `minutes_to_seconds__stream` | `end-to-end case-0026-minutes_to_seconds__stream-budget_capped.json` |
| `10.b` | `minutes_to_seconds__stream` | `end-to-end case-0026-minutes_to_seconds__stream-budget_unlimited.json` |
| `10.b` | `multiline_checksum__non-stream` | `end-to-end case-0027-multiline_checksum__non-stream-budget_capped.json` |
| `10.b` | `multiline_checksum__non-stream` | `end-to-end case-0027-multiline_checksum__non-stream-budget_unlimited.json` |
| `10.b` | `multiline_checksum__stream` | `end-to-end case-0028-multiline_checksum__stream-budget_capped.json` |
| `10.b` | `multiline_checksum__stream` | `end-to-end case-0028-multiline_checksum__stream-budget_unlimited.json` |
| `10.b` | `multiply_17_19__non-stream` | `end-to-end case-0029-multiply_17_19__non-stream-budget_capped.json` |
| `10.b` | `multiply_17_19__non-stream` | `end-to-end case-0029-multiply_17_19__non-stream-budget_unlimited.json` |
| `10.b` | `multiply_17_19__stream` | `end-to-end case-0030-multiply_17_19__stream-budget_capped.json` |
| `10.b` | `multiply_17_19__stream` | `end-to-end case-0030-multiply_17_19__stream-budget_unlimited.json` |
| `10.b` | `parity_expression__non-stream` | `end-to-end case-0031-parity_expression__non-stream-budget_capped.json` |
| `10.b` | `parity_expression__non-stream` | `end-to-end case-0031-parity_expression__non-stream-budget_unlimited.json` |
| `10.b` | `parity_expression__stream` | `end-to-end case-0032-parity_expression__stream-budget_capped.json` |
| `10.b` | `parity_expression__stream` | `end-to-end case-0032-parity_expression__stream-budget_unlimited.json` |
| `10.b` | `python_loop_trace__non-stream` | `end-to-end case-0033-python_loop_trace__non-stream-budget_capped.json` |
| `10.b` | `python_loop_trace__non-stream` | `end-to-end case-0033-python_loop_trace__non-stream-budget_unlimited.json` |
| `10.b` | `python_loop_trace__stream` | `end-to-end case-0034-python_loop_trace__stream-budget_capped.json` |
| `10.b` | `python_loop_trace__stream` | `end-to-end case-0034-python_loop_trace__stream-budget_unlimited.json` |
| `10.b` | `sequence_next__non-stream` | `end-to-end case-0035-sequence_next__non-stream-budget_capped.json` |
| `10.b` | `sequence_next__non-stream` | `end-to-end case-0035-sequence_next__non-stream-budget_unlimited.json` |
| `10.b` | `sequence_next__stream` | `end-to-end case-0036-sequence_next__stream-budget_capped.json` |
| `10.b` | `sequence_next__stream` | `end-to-end case-0036-sequence_next__stream-budget_unlimited.json` |
| `10.b` | `set_intersection__non-stream` | `end-to-end case-0037-set_intersection__non-stream-budget_capped.json` |
| `10.b` | `set_intersection__non-stream` | `end-to-end case-0037-set_intersection__non-stream-budget_unlimited.json` |
| `10.b` | `set_intersection__stream` | `end-to-end case-0038-set_intersection__stream-budget_capped.json` |
| `10.b` | `set_intersection__stream` | `end-to-end case-0038-set_intersection__stream-budget_unlimited.json` |
| `10.b` | `sort_integers__non-stream` | `end-to-end case-0039-sort_integers__non-stream-budget_capped.json` |
| `10.b` | `sort_integers__non-stream` | `end-to-end case-0039-sort_integers__non-stream-budget_unlimited.json` |
| `10.b` | `sort_integers__stream` | `end-to-end case-0040-sort_integers__stream-budget_capped.json` |
| `10.b` | `sort_integers__stream` | `end-to-end case-0040-sort_integers__stream-budget_unlimited.json` |
| `10.b` | `spanish_logic__non-stream` | `end-to-end case-0041-spanish_logic__non-stream-budget_capped.json` |
| `10.b` | `spanish_logic__non-stream` | `end-to-end case-0041-spanish_logic__non-stream-budget_unlimited.json` |
| `10.b` | `spanish_logic__stream` | `end-to-end case-0042-spanish_logic__stream-budget_capped.json` |
| `10.b` | `spanish_logic__stream` | `end-to-end case-0042-spanish_logic__stream-budget_unlimited.json` |
| `10.b` | `structured_json__non-stream` | `end-to-end case-0043-structured_json__non-stream-budget_capped.json` |
| `10.b` | `structured_json__non-stream` | `end-to-end case-0043-structured_json__non-stream-budget_unlimited.json` |
| `10.b` | `structured_json__stream` | `end-to-end case-0044-structured_json__stream-budget_capped.json` |
| `10.b` | `structured_json__stream` | `end-to-end case-0044-structured_json__stream-budget_unlimited.json` |
| `10.b` | `system_instruction__non-stream` | `end-to-end case-0045-system_instruction__non-stream-budget_capped.json` |
| `10.b` | `system_instruction__non-stream` | `end-to-end case-0045-system_instruction__non-stream-budget_unlimited.json` |
| `10.b` | `system_instruction__stream` | `end-to-end case-0046-system_instruction__stream-budget_capped.json` |
| `10.b` | `system_instruction__stream` | `end-to-end case-0046-system_instruction__stream-budget_unlimited.json` |
| `10.b` | `tool_none_arithmetic__non-stream` | `end-to-end case-0051-tool_none_arithmetic__non-stream-budget_capped.json` |
| `10.b` | `tool_none_arithmetic__non-stream` | `end-to-end case-0051-tool_none_arithmetic__non-stream-budget_unlimited.json` |
| `10.b` | `tool_none_arithmetic__stream` | `end-to-end case-0052-tool_none_arithmetic__stream-budget_capped.json` |
| `10.b` | `tool_none_arithmetic__stream` | `end-to-end case-0052-tool_none_arithmetic__stream-budget_unlimited.json` |
| `10.b` | `unicode_symbol_math__non-stream` | `end-to-end case-0067-unicode_symbol_math__non-stream-budget_capped.json` |
| `10.b` | `unicode_symbol_math__non-stream` | `end-to-end case-0067-unicode_symbol_math__non-stream-budget_unlimited.json` |
| `10.b` | `unicode_symbol_math__stream` | `end-to-end case-0068-unicode_symbol_math__stream-budget_capped.json` |
| `10.b` | `unicode_symbol_math__stream` | `end-to-end case-0068-unicode_symbol_math__stream-budget_unlimited.json` |
| `10.b` | `lifecycle_single_result__stream` | `end-to-end case-0129-lifecycle_single_result__stream-budget_capped.json` |
| `10.b` | `lifecycle_single_result__stream` | `end-to-end case-0129-lifecycle_single_result__stream-budget_unlimited.json` |
| `30.a` | `tool_add_named__non-stream` | `end-to-end case-0047-tool_add_named__non-stream-budget_capped.json` |
| `30.a` | `tool_add_named__non-stream` | `end-to-end case-0047-tool_add_named__non-stream-budget_unlimited.json` |
| `30.a` | `tool_add_named__stream` | `end-to-end case-0048-tool_add_named__stream-budget_capped.json` |
| `30.a` | `tool_add_named__stream` | `end-to-end case-0048-tool_add_named__stream-budget_unlimited.json` |
| `30.a` | `tool_translate_named__stream` | `end-to-end case-0054-tool_translate_named__stream-budget_capped.json` |
| `30.a` | `tool_translate_named__stream` | `end-to-end case-0054-tool_translate_named__stream-budget_unlimited.json` |
| `30.b` | `lifecycle_single_result__stream` | `end-to-end case-0129-lifecycle_single_result__stream-budget_capped.json` |
| `30.b` | `lifecycle_single_result__stream` | `end-to-end case-0129-lifecycle_single_result__stream-budget_unlimited.json` |
| `30.b` | `lifecycle_chained_calculation__stream` | `end-to-end case-0145-lifecycle_chained_calculation__stream-budget_capped.json` |
| `30.b` | `lifecycle_chained_calculation__stream` | `end-to-end case-0145-lifecycle_chained_calculation__stream-budget_unlimited.json` |
| `30.d` | `schema_escaped_unicode_string__non-stream` | `end-to-end case-0105-schema_escaped_unicode_string__non-stream-budget_capped.json` |
| `30.d` | `schema_escaped_unicode_string__non-stream` | `end-to-end case-0105-schema_escaped_unicode_string__non-stream-budget_unlimited.json` |
| `30.e` | `schema_array__stream` | `end-to-end case-0108-schema_array__stream-budget_capped.json` |
| `30.e` | `schema_array__stream` | `end-to-end case-0108-schema_array__stream-budget_unlimited.json` |

## Request-scoped modes (groups 30+)

Groups 1–12 vary the model OUTPUT. Groups 30+ vary the request: the resolved `UnifiedParserInit` the serving layer passed to `UnifiedParser::initialize_request` before any output arrived. The pairing is `X0` happy / `X1` malformed, and a new mode takes the next ten.

`starting_state` says which channel the rendered prompt already opened, so the model never emits that opener: `None` (it opens its own), `Reasoning` (the stream begins INSIDE a thought), `Response` (visible content is already open, so there is no reasoning channel at all and reasoning markers are ordinary text). `tool_output_mode` says whether the backend constrained decoding: `Native` (model markup) or `GuidedJson` (bare JSON — a NAMED choice sends that tool's arguments alone, a REQUIRED choice sends one call object or an array of them).

### Group 30 — Guided decoding, happy
- **`30.a`** (`guided_json_named_tool`) `tool_choice` names a tool; the payload is that tool's arguments and the name comes from the request. This is also covered in: e2e case-0047-tool_add_named__non-stream-budget_capped.json, e2e case-0048-tool_add_named__stream-budget_capped.json, e2e case-0054-tool_translate_named__stream-budget_capped.json (each with its `-budget_unlimited` pair).
- **`30.b`** (`guided_json_required_tool`) Required choice; the payload is an array of call objects. This is also covered in: e2e case-0129-lifecycle_single_result__stream-budget_capped.json, e2e case-0145-lifecycle_chained_calculation__stream-budget_capped.json (FIRST step of each; both with their `-budget_unlimited` pair).
- **`30.c`** (`guided_json_two_calls`) Two DIFFERENT tools in one array. Multi-call is the array's ordinary shape, not an edge case.
- **`30.d`** (`guided_json_escaped_string_args`) An argument value carrying non-ASCII, escaped quotes and Windows backslashes. Native mode covers the same value in `7.*`, but there the value is raw text between markers and no escaping is involved — the escaping is only the parser's problem on this path. This is also covered in: e2e case-0105-schema_escaped_unicode_string__non-stream-budget_capped.json (and its `-budget_unlimited` pair).
- **`30.e`** (`guided_json_array_argument`) An argument VALUE that is an array, not a scalar. Distinct from `30.b`/`30.c`, where the array is the list OF CALLS one level up. A list arriving as its string rendering is a silently wrong call, not a failed one. This is also covered in: e2e case-0108-schema_array__stream-budget_capped.json (and its `-budget_unlimited` pair).
- **`30.f`** (`guided_json_after_reasoning`) A normal thought, THEN the constrained payload. Every other guided case starts at the payload, so nothing pinned the ordinary shape where the model reasons first and the backend constrains only the call. This is the baseline group 31's surroundings cases contrast with.
- **`30.g`** (`guided_json_marker_inside_argument`) A control marker of the family's OWN grammar inside a guided argument VALUE. Once the payload has opened, a marker is argument DATA and must survive byte-exact (`I7`); re-reading it as a channel token corrupts the call the tool receives while still looking like a successful dispatch. The golden argument is the family's own marker, not a placeholder — a stand-in would pass whatever the parser did.

### Group 31 — Guided decoding, malformed / recovery
- **`31.a`** (`guided_json_invalid_call`) Valid JSON that is not a call (no `name`). Surfaces as text under the guided malformed-payload policy; no call dispatched.
- **`31.b`** (`guided_json_malformed_json`) JSON that does not parse — a truncated object, what a constrained decode looks like when the budget runs out.
- **`31.c`** (`guided_json_partial_calls`) The array parses but one element is not a call.
- **`31.d`** (`guided_json_list_with_broken_element`) `[<valid call>, <broken JSON>]` — the array itself does not parse, so per-element recovery never runs.
- **`31.e`** (`guided_json_tool_open_before_payload`) A native tool OPENER precedes the payload. Guided decoding delivers the call as JSON, so leading markup is stray: strip it, or it enters the payload buffer, breaks the parse and costs the call.
- **`31.f`** (`guided_json_tool_close_after_payload`) A native tool CLOSER follows the payload. Markers can BRACKET a payload, not only precede it — once the opening brace latches visible-only, every later byte is appended verbatim.
- **`31.g`** (`guided_json_wrapped_in_tool_markup`) Opener AND closer, the shape a template emits when guided decoding is applied INSIDE a tool block. Handling one end only still loses the call.
- **`31.h`** (`guided_json_narrated_invoke_in_reasoning`) The model NARRATES a tool opener while thinking, then the real call arrives as JSON. The reasoning channel is unconstrained under guided decoding, so that markup is prose; treating it as structure ends the turn and discards the payload.
- **`31.i`** (`guided_json_prose_before_reasoning`) Visible prose, then a thought, then the payload. Every other guided case opens its thought at byte 0; with prose first the run can latch the payload buffer and surface the model's private thinking to the user as the answer.
- **`31.j`** (`guided_json_orphan_reason_close_before_payload`) An orphan reasoning CLOSER with nothing open. The native scanner strips a stray closer wherever it appears before an opener; guided must agree or the same bytes read differently by request mode (`I3`).
- **`31.k`** (`guided_json_orphan_tool_close_before_payload`) An orphan tool CLOSER. Paired with `31.e`: while the closer was stripped and the opener beside it was not, which marker leaked depended on which one the model happened to emit.
- **`31.w`** (`guided_json_native_markup_only`) Guided mode receives one complete native tool call instead of bare JSON. The turn is control markup and emits no events; every stream split must match the whole-input result instead of leaking the parameter body as visible text.

`31.c` and `31.d` pin **all-or-nothing**: one bad element voids the whole array and the payload goes out as text, taking the valid call with it. That is deliberate. A tool call is a side effect, so dispatching one extracted from a document that failed validation fails OPEN. Text loses nothing — the raw payload stays visible. `31.a`–`31.d` each also emit `tracing::warn!(why = "unified_guided_json_not_a_tool_call")`: the events alone are indistinguishable from a model that chose to answer in prose, so the log is the only signal the backend's guided decoding failed.

`31.a`–`31.d` are malformed PAYLOADS. `31.e`–`31.k` are well-formed payloads in malformed SURROUNDINGS: they recover the markers, the payload then parses, and the call dispatches — so neither the all-or-nothing rule nor that warning applies to them.

**Peer-engine value here is intentionally limited.** vLLM does not emit guided JSON in the base capture, and the families with no native unified parser cannot honour `init` at all, so those columns are structurally `UNSUPPORTED` rather than a comparison. What these rows do pin: the authored golden contract, the current native parsers' recovery, and the split-family result where 0.1.25 captured it. The `dynamo_v2-0.1.22` column was back-captured through the prior 237-case corpus; cases added later are explicitly missing from that historical capture rather than inferred.

### Group 40 — Prefilled reasoning, happy
- **`40.a`** (`prefilled_reasoning_with_tool`) Stream begins inside a thought, closes it, calls a tool.
- **`40.b`** (`prefilled_reasoning_with_guided_json`) Same, with the call as guided JSON.
- **`40.c`** (`prefilled_reasoning_then_text_then_tool`) reasoning → visible prose → call. All three channels in one prefilled stream.
- **`40.d`** (`prefilled_reasoning_then_text`) reasoning → prose, no call. Pins that closing a prefilled thought returns the stream to VISIBLE content rather than leaving it in reasoning, which would swallow the whole answer.

### Group 41 — Prefilled reasoning, malformed
- **`41.a`** (`prefilled_reasoning_redundant_opener`) The backend re-emits the `<think>` the prompt already wrote. Exactly one echo is consumed, not leaked; a second would be stray markup and stripped (I3). The only case where a prefilled stream legitimately carries an opener.
- **`41.b`** (`prefilled_reasoning_truncated`) Budget runs out mid-call. Keep the completed reasoning, drop the partial call (P2).

### Group 50 — Prefilled response, happy
- **`50.a`** (`prefilled_response_with_tool`) Leading visible content, then a native call.
- **`50.b`** (`prefilled_response_with_guided_json`) Guided payload with the response channel already open.
- **`50.c`** (`prefilled_response_guided_json_two_calls`) Two different tools; enters guided mode visible-only rather than outside-reasoning.
- **`50.d`** (`prefilled_response_reasoning_markers_literal`) **The only case where `starting_state=Response` is observable.** `<think>literal</think>` must reach the user as TEXT, markers and all, because this stream has no reasoning channel. Every other 50/51 case has no reasoning markers in its input and therefore parses identically under `starting_state=None` — 50.a matches 8.a, 50.b matches 30.b, 50.c matches 30.c, 51.b matches 31.c.

### Group 51 — Prefilled response, malformed
- **`51.a`** (`prefilled_response_truncated`) Budget runs out mid-call; the prose already emitted survives.
- **`51.b`** (`prefilled_response_guided_json_partial_calls`) All-or-nothing, as `31.c`, with the response channel prefilled.

## Authoring a case: what to check BEFORE adding one

Every rule here exists because a case was added that could not fail for the reason it claimed. Fake coverage is worse than no coverage — it renders green.

1. **Is it distinguishable from an existing case?** Compare `init` AND input against the corpus. If some existing case has the same configuration and the same input shape, the new case tests nothing. Three groups were deleted for exactly this: a whole axis whose 51/52/53 cases had the same config and inputs as `1.a`, differing only in a label the parser cannot read.
2. **Can it fail for the stated reason?** Write down what would have to break for the case to go red, then confirm the parser can even SEE that input. `finish_reason` cannot: `finish()` takes no argument, in Dynamo and in vLLM alike, so a case that varies only the finish reason varies nothing. If the axis is invisible to the parser, express it as an input shape instead — `length` becomes a TRUNCATED input, which is observable.
3. **Does the field already exist under another name?** A per-case `input_mode` was added that was a 1:1 alias of `init.starting_state` across every row, and could not diverge, because "where the stream starts" IS what the starting state encodes. Grep the case dict before adding a key.
4. **Measure the behavior, do not predict it.** Author the case, run the harness, read what the parser actually emitted, and THEN write the golden and the description around it. The all-or-nothing array semantics were found this way; predicting them would have produced a wrong golden that looked authoritative.
5. **A near-duplicate that survives must say what it duplicates.** If a case is kept because it exercises a different code path despite the same shape, name the sibling in its description (`50.b` says it matches `30.b`), so the next reader does not re-derive the question.
6. **The input must be a shape the declared `init` can actually produce.** Six guided-decoding scenarios rendered NATIVE model markup for gemma4 and kimi_k2 while declaring `tool_output_mode=GuidedJson` — a mode that constrains the model to bare JSON, so that markup is the one input it can never emit. They rendered green for a year because neither family had a unified parser to run them; the moment gemma4 got one, all six failed. Guided payloads are grammar-independent and are now written ONCE for every family (`every_family` in `gen_unified_golden.py`); only the reasoning envelope around them is per family.
7. **A per-family golden needs a per-family fill, not one family's bytes.** `50.d` asserts that the model's own reasoning markers reach the user as literal TEXT, and its golden hardcoded qwen3's `<think>literal</think>` for all three families. Use the `None`-placeholder fill (as `12.a` does for an argument value) so the scenario stays shared and only the grammar-specific bytes differ.

## Verifying a change to the table

The model blob and the rendered page are different things. A cell can carry correct JSON and render nothing — the column-header popup shipped with `init` in every column and an empty config list, because the model that feeds `buildGrammarHtml` is assembled separately and dropped the field.

- Check the DOM, not just `conformance-model` JSON.
- Headless Chrome reports `(hover: hover) = false`, so hover listeners never attach and a naive hover test "fails" on the baseline too. Emulate with `--blink-settings=primaryHoverType=2,availableHoverTypes=2`.
- A synthetic `pointerenter` does NOT set CSS `:hover`. Use it to test JS behavior, a real pointer move to test CSS.
- Never run `render_table_v2.sh` and the pytest suite at the same time: both stage into `conformance/utils/.stage/`, and the collision shows up as ~13 unrelated browser-test errors.
- A `transform` on a cell makes it the containing block for its own popup AND scales it. Use shadow and filter for cell affordances; a transform silently breaks popup placement.

## Deferred (not in the U0 seed set)

- **n>1 interleave** (`UNIFIED.interleave_n2.*`, the Example-B n>1 LOSS case) needs a multi-choice interleaved driver (extends PR #135's tool-only lanes to carry reasoning state). Its golden is per-choice, a different shape than the single-stream cases here. Author with the n>1 lane.

## A defect the corpus missed owes the corpus a case

Every bug found by a reviewer, another agent, or a probe — that the existing cases did NOT catch — is evidence of a missing case, and the fix is not complete until that case exists here. Prefer a taxonomy scenario over a unit test: a scenario runs for every family and every delivery schedule the harness drives, a unit test runs once for one family. Fall back to a unit test only when the schema cannot express the property, and say why in its doc comment.

Name the missing DIMENSION, not the example. `guided_json_stray_prefix_before_reasoning` and `guided_json_narrated_prefix_inside_reasoning` were added after a stray `<function=` header borrowed its `>` from a following thought opener and emitted the model's private reasoning as visible text. The example was one input; the untested axis was **which control marker owns a terminator when two compete** — and nothing in the corpus had ever asked that question.

The check is the count: if a review round produced N defects the corpus missed and the scenario count did not move, the holes are still open.

**A duplicate is worse than a gap.** Before adding, normalize `(input, init, golden)` across the corpus and drop any crossing that already exists. A generated product once recreated three hand-authored scenarios — 9 cases across families — inflating the count while testing nothing new, and leaving two names for one behaviour to drift apart. `test_no_two_scenarios_have_identical_behaviour` now enforces this.
