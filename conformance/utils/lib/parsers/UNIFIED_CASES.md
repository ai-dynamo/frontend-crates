# Unified Parser Cases (reasoning + content + tool calls, one ordered stream)

Reference taxonomy for the **unified** conformance surface: one parser owns the whole assistant-output grammar and emits ONE ordered event stream. Sibling stage docs: `REASONING_CASES.md` (reasoning only), `TOOLCALLING_CASES.md` / `TOOLCALLING_STREAMING_V1_CASES.md` (tool calls only). This surface is what those two cannot express — the ORDER between reasoning and tool calls, and reasoning that occurs *between* or *after* tool calls.

The golden corpus is authored by `conformance/utils/src/gen_unified_golden.py` (one scenario spec -> `conformance/unified/golden_spec/<family>.yaml` in the gitignored build tree); the committed canonical files under `conformance/fixtures-unified-v2/families/` are derived from it.

## Single-family case review

Always evaluate single-family tests against every registered family when adding, changing, or reviewing them, following the [case applicability policy](../../../README.md#case-applicability). If another family can express the behavior, add family-native inputs and authored expected outputs for every applicable family and use a shared section. Otherwise, explain the grammar or request-mode limitation and place the case in a named family-specific test section. Preserve published IDs, capture history, and older unmeasured results when changing section placement.

The corpus guard checks every generic semantic case, independent of which family exposed the regression. A generic case must have native inputs in more than one registered family. Visual folding does not confer parent coverage: semantic variants must independently span families. Only explicit `CaseVariant.coverage_parent` aliases combine equivalent native requests or retained delivery variants; each retains its individual input, expectation, and captures. A standalone case with one family must gain applicable native inputs or move to a named family section. The rendered-model guard checks the resulting columns as well as the authored corpus.

## The oracle: GOLDEN is authored, not captured

## Capture version policy

Follow [the Unified storage contract](../../../README.md#unified-storage-contract-plain-versioned-yaml-only): plain semantic-version YAML only, with unchanged family output carried forward. Source SHA is optional first-capture origin metadata, never a capture name. This migration does not change the older stream or batch-on-stream storage. Changing an existing Unified input requires rerunning and updating every prior affected version.

The shared-family coverage for `7-glm_ref_object`, `7-glm_ref_encoded_targets`, `7-glm_ref_json_looking_strings`, and `7-glm_ref_scalar_types` is first measured in peer families at Dynamo v2 `0.7.18`. Earlier peer-family release cells remain unmeasured; they were not backfilled when coverage expanded. GLM was captured on the released Dynamo v2 `0.7.8` source under PR #271, and its existing history is preserved.

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

Raw streaming tool deltas with `complete: false` are provisional. A malformed or truncated call can leave those bytes on the wire without ever completing; consumers must wait for `complete: true` before treating a call as executable. Dropping an incomplete call during assembly does not retract its earlier fragments.

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
  - **beam** is the exception by design: its markers are tokenizer special tokens the engine emits as such, so a quotation never keeps one literal. The quoted-control scenarios (`native_quoted_control_*`, `native_single_quote*`, `native_quoted_incomplete_header`, `guided_json_narrated_invoke_in_reasoning`) carry beam-specific goldens: the prose before the quote stays, the quoted header is framing (a header it cannot complete is skipped with what follows it), and a real call after it still dispatches. The builder's reference parser reads these the same way.
- **P7 Nested channel markers (a marker of one channel inside another)** -> marker recognition is CHANNEL-SCOPED, and both directions follow the same best-effort-recovery rule (recover real structure, never leak markup, never drop a valid call):
  - Inside a **quoted tool-argument string value**, marker-looking bytes are DATA (I7). A reasoning marker there does NOT open a reasoning channel — it is the literal arg string (`reason_markup_in_arg`). A reasoning-first pipeline that extracts `<think>`/`<|channel>` before tool parsing corrupts the arg -> red (ARG_MISMATCH / MERGE).
  - A **well-formed tool-call envelope inside a reasoning span** is STRUCTURAL: break out of reasoning, emit the call, resume reasoning after its close (`tool_in_reason`). Leaking the raw `<|tool_call>...<tool_call|>` into `reasoning_content`, or dropping the call, is the regression -> red (LEAK). The asymmetry is deliberate: quote delimiters explicitly mark a data region, whereas a reasoning span is opaque text that can still contain recoverable structure.

> P1/P2 are RESOLVED by the documented best-effort-recovery contract above (not open product questions). P4 is now resolved in code for gemma4 (optional role label) and stays a judgment call for any family that adds structural whitespace of a different shape; cases depending on it carry a `policy:` tag. P7 is RESOLVED by the same contract (no markup leak, no dropped call); the families still on the split path diverge, which is the gap it documents.

## Divergence classes (how a non-matching cell is colored)

`MATCH` (green) · `ORDER` / `MERGE` / `LOSS` (the unification gap) · `LEAK` (markup in text, `↯`) · `ARG_MISMATCH` / `WHITESPACE` (version drift) · `ERROR` (engine hard-errored where the spec expects graceful output).

The eleven native `UnifiedParser` families are `deepseek_v4`, `deepseek_v41`, `qwen3`, `gemma4`, `glm47`, `kimi_k2`, `kimi_k3`, `muse_glimmer`, `nemotron_v3`, `muse_spark`, and `gpt_oss`. GPT-OSS retains its historical Dynamo v1 reasoning + batch observations and now records ordered native v2 events.

The dotted-name cases cover minimal, mixed-punctuation, multi-segment, and shared-prefix tool names. They apply to Beam, DeepSeek V4/V4.1, Gemma 4, GLM 4.7, GPT-OSS, Kimi K2/K3, Muse Glimmer, Muse Spark Open, Nemotron 3 Ultra, and Qwen3. New observations are recorded only for v2. GPT-OSS captures before v2 0.7.28 use v1 reasoning followed by the v2 Harmony tool parser and are labeled split; v2 0.7.28 and later use native UnifiedParser. The split Harmony path remains covered by whole-input and every-boundary comparisons against the same GOLDEN.

## Nemotron 3 Ultra applicability

The `nemotron_v3` corpus inherits every applicable Qwen XML scenario through the registry's `grammar: qwen3` declaration. Default-state shared inputs first close Ultra's implicit empty thought with `</think>`; explicitly initialized Reasoning and Response inputs keep their original request state and bytes. This preserves the shared scenario's channel semantics without disabling later reasoning markers. The Qwen XML header cases also apply to Nemotron. The structured-response scenario requires a registered structural-tag builder, which Nemotron does not have; it remains explicitly excluded while shared parser scenarios retain Qwen grammar inheritance. DSML, Harmony, Gemma, GLM, Kimi, and ATEM-specific scenarios remain excluded because they require other native grammars.

These two cases display under the Nemotron v3 default-reasoning section. Their published numeric IDs and capture history remain unchanged. Only Nemotron currently enables implicit reasoning under default initialization; explicitly initializing another family to Reasoning exercises a different request contract.

- `10-default_reasoning_without_opener`: with default initialization, `Check Montréal.</think>雪` emits reasoning followed by visible Unicode content. The same input under Response stays literal text.
- `12-default_reasoning_tool_without_closer`: default reasoning contains a native XML call with literal think markers in its argument, then resumes reasoning at EOF. Both think boundaries are omitted. Other registered families require explicit Reasoning initialization for this contract; their shared prefilled and nesting scenarios cover that state.

## Quick reference — descriptive taxonomy (`UNIFIED.<group>-<description>`)

Every active Unified ID uses `UNIFIED.<group>-<description>`, for example `UNIFIED.7-arg_unicode` or `UNIFIED.kimi-typed_argument_values`. Derive descriptions from scenario names, omit redundant family prefixes in family-specific groups, and use schema probe names directly, such as `UNIFIED.7-declared_array`. Numeric, dotted, and schema-subgroup forms remain historical aliases; do not create new ones. `unified_taxonomy.py` owns canonical IDs, descriptions, ordering, display placement, and aliases. Internal scenario names stay unchanged. The scenario slug is shown in parentheses. **Groups 1–9 mirror the tool-calling STREAM taxonomy** (`TOOLCALLING.streamv1.N`) as reasoning-free unified cases — this surface subsumes STREAM. **Group 10** is the reasoning axis (`REASONING.*`). **Group 11 is UNIQUE to unified**: reasoning↔tool ORDER that neither STREAM (no reasoning) nor REASONING (no ordered tool events) can express. **Group 12** is adversarial nesting — a marker of one channel inside another (P7). **Groups 30–39 are Guided Decoding**, divided by payload validity, surrounding markup, reasoning boundaries, and visible-answer markers. **Groups 40+ are prefilled request states.** Model-specific groups (`gemma`, `glm5`, `kimi`, `muse`) sort after every numeric group. The live per-family and total case counts come from `test_unified_case_counts_match_the_generator`; do not maintain a numeric total in prose.

### Group 1 — TC Single call
- **`1-tool_only`** (`tool_only`) One tool call, no reasoning, no surrounding text. The tool suite's baseline.

### Group 2 — TC Multiple calls (TOOLCALLING.streamv1.2)
- **`2-two_calls`** (`two_calls`) Two distinct calls back-to-back, order preserved. This is also covered in: TOOLCALLING.streamv1.2.a.
- **`2-two_calls_same_name`** (`two_calls_same_name`) Two calls to the SAME function, different args — must not dedup or merge. This is also covered in: TOOLCALLING.streamv1.2.d.
- **`2-adjacent_calls_partial_header`** (`adjacent_calls_partial_header`) Native/default state: one delta closes a complete call and carries the next opener plus a partial call header. Preserve both calls and emit no visible markup. Uses authored Unified inputs, tools, chunk schedules, and independent expected calls; DeepSeek V4.1 splits inside its second invoke in one calls container.
- **`2-three_calls_partial_header`** (`three_calls_partial_header`) Native/default state: three adjacent calls delivered with a boundary inside the second call header. Preserve f, g, h in order and emit no visible markup. Both scenarios apply to nine families: deepseek_v4, deepseek_v41, gemma4, glm47, kimi_k2, kimi_k3, muse_glimmer, nemotron_v3, and qwen3 (18 family-specific cases). Beam and GPT-OSS are excluded from this Unified case set. The authored schedules exercise delivery boundaries that marker-aligned and fixed-width chunking can miss.

### Group 3 — TC No call (TOOLCALLING.streamv1.3)
- **`3-text_only`** (`text_only`) Plain content, zero tool structure. No spurious call. This is also covered in: TOOLCALLING.streamv1.3. No e2e case has this shape: Qwen3.6 always emits a reasoning span, so the plain-content case is corpus-only.

### Group 4 — TC Malformed envelope
- **`4-tool_block_never_closed_then_text`** (`tool_block_never_closed_then_text`) The calls opener arrives without its closing marker and prose follows. The prose remains inside the unterminated tool envelope and is discarded at EOF; it is not visible answer text. This is applicable to DSv4.1 because its native grammar has an explicit calls envelope. The retained GPT-OSS request instead starts with an orphan handoff and never opens a tool message: its following prose remains visible. Genuine Harmony truncation is covered by shared `tool_no_close` and its next-message recovery variant.
- **`4-tool_markup_only_emits_nothing`** (`tool_markup_only_emits_nothing`) A calls envelope contains no invocation. Both markers are control syntax and the parser emits no event.

### Group 5 — TC Truncation / recovery (TOOLCALLING.streamv1.5)
- **`5-truncated_tool_eof`** (`truncated_tool_eof`) EOF mid-call. Golden drops the partial, keeps preceding output (P2); vLLM Rust hard-errors (`ParsingFailed`). Class ERROR.
- **`5-tool_no_close`** (`tool_no_close`) Complete call body but the close marker never arrives. Most grammars recover the complete call at finish; DeepSeek V4 and V4.1 require the invoke closer and drop this malformed call. This is also covered in: TOOLCALLING.streamv1.5.a.
- **`5-orphan_close_after_prose`** (`orphan_close_after_prose`) Orphan close marker after prose. Golden strips it; engines may leak. Class LEAK.
- **`5-malformed_json_then_two_valid_calls`** (`malformed_json_then_two_valid_calls`) Across all native Unified families, an unfinished native JSON/string body with explicit call closers precedes `echo(value="é")` and `echo(value="Café")`. The authored `input_chunks: [input]` delivers one `push()`, then `finish()`. Marker-aligned delivery masks the Kimi K3 defect: published `dynamo-parsers-v2` versions `0.7.13` and `0.7.14` lose the first echo; `0.7.15`, now published, preserves both. During capture qualification, the candidate moved to `0.7.15` after `0.7.14` was published; its earlier unpublished checkpoint was retired rather than relabeled as released output. K2 retains the malformed call as `{}`; Qwen3, GLM47, DeepSeek V4, and Muse retain its malformed object value as a literal string. Those five families preserve both echoes in both versions. Gemma4 returns no calls and DeepSeek V4.1 raises a JSON parse error in both versions; their goldens still require both echoes, and both remain red. Parser fixes for these two failures are outside this change.

### Group 6 — TC Empty body (TOOLCALLING.streamv1.6)
- **`6-empty_args`** (`empty_args`) Call with `{}` arguments. Must emit the call with an empty object, not drop it. This is also covered in: TOOLCALLING.streamv1.6.a.

### Group 7 — TC Argument fidelity (TOOLCALLING.streamv1.7)
- **`7-arg_unicode`** (`arg_unicode`) Non-ASCII argument value round-trips byte-exact (I7). This is also covered in: TOOLCALLING.streamv1.7.b.
- **`7-arg_marker_in_string`** (`arg_marker_in_string`) A close-marker substring INSIDE a string arg is data, preserved exactly (I7). vLLM Rust truncates. Class ARG_MISMATCH.
- **`7-arg_object_quoted_parameter_close`** and **`7-arg_array_quoted_parameter_close`** Nested JSON strings retain a literal `</atem:parameter>`, including escaped quotes and Unicode. All twelve native families use their own argument encoding; both Muse families additionally exercise their parameter framing (I7).
- **`7-deepseek_v41_mixed_control_text_in_string`** (`deepseek_v41_mixed_control_text_in_string`) All native Unified families use their native string encoding to carry mixed reasoning and tool delimiters, entity text, quotes, a backslash, a newline, and surrounding spaces. The decoded string must survive exactly. This extends `7-arg_marker_in_string` beyond a single closer; the internal scenario name and original DeepSeek V4.1 input remain unchanged for capture history.
- **`7-arg_json_null`** (`arg_json_null`) The request tool schema permits JSON null. Bare parameter text `null` must produce JSON `null`; grammars with explicit types use native null syntax. Variants cover nullable type arrays, anyOf, oneOf, nullable, and const. Mixed-field probes also assert that non-nullable fields remain strings. Refs #251, #268, #269. Each schema variant has a distinct fixture ID; the popup lists every applicable result. The two categories reference the same mixed-field captures.
- **`7-arg_string_null`** (`arg_string_null`) The request tool schema requires a string for the tested value. Bare parameter text `null` must remain JSON string `"null"`; grammars with explicit types use native string syntax. Variants cover non-nullable unions and intersecting sibling constraints. Mixed-field probes also assert that nullable fields become null. Refs #251, #268, #269. Each schema variant has a distinct fixture ID; the popup lists every applicable result. The two categories reference the same mixed-field captures.
- **`7-arg_json_null_ref`** (`arg_json_null_ref`) GLM-only regression for PR #268: an unresolved local `$ref` points to a nullable string definition on the tool parameters root. Split-chunk XML input with bare `null` must produce JSON null.
- **`7-arg_string_null_ref`** (`arg_string_null_ref`) GLM and Qwen regression: an unresolved local `$ref` points to a string-only definition on the tool parameters root. Split-chunk XML input with bare `null` must preserve string `"null"`.
- **`7-glm_ref_object`** (`glm_ref_object`) Shared-family compatibility control for PR #271: a local object reference converts JSON object text into an object. The ref remains unresolved in the request.
- **`7-glm_ref_encoded_targets`** (`glm_ref_encoded_targets`) Shared-family regression for PR #271: URI percent decoding and JSON Pointer unescaping resolve definition names containing a space, UTF-8 with literal `+`, and `/` plus `~`. Bare `42` becomes an integer for each target.
- **`7-glm_ref_json_looking_strings`** (`glm_ref_json_looking_strings`) Shared-family regression for PR #271: referenced string fields preserve object-looking, array-looking, and quoted JSON text exactly, including literal quote characters. An inline string field with matching object-looking bytes is a control.
- **`7-glm_ref_scalar_types`** (`glm_ref_scalar_types`) Shared-family regression for PR #271: integer, number, and boolean refs produce typed scalars. A sibling integer constraint narrows a string-or-integer ref; bare `42` must become an integer. All four probes keep request refs unresolved.
- **`7-unused_reference_graph_parameter_types`** (`unused_reference_graph_parameter_types`) An unused compact reference graph precedes directly typed integer, boolean, and string fields. It must not change their types or discard the complete call.
- **`7-nullable_reference_alias_literals`** (`nullable_reference_alias_literals`) Nullable aliases to string `const: "null"` and `enum: ["null"]` retain the schema-valid string null values.
- **`7-local_schema_id_preserves_type`** (`local_schema_id_preserves_type`) A local `$id` disables unsupported reference traversal without discarding a directly declared integer type.
- **`7-arg_numeric_nested_exact`** (`arg_numeric_nested_exact`) Nested objects and arrays preserve a large fractional number, an underflow-sized exponent, and positive and negative integers beyond the exact binary64 range.

### Group 8 — TC Content / narration position (TOOLCALLING.streamv1.8)
- **`8-text_before_tool`** (`text_before_tool`) Visible narration precedes the call. This is also covered in: TOOLCALLING.streamv1.8.a.
- **`8-trailing_text_after_tool`** (`trailing_text_after_tool`) Arbitrary prose AFTER the tool section (P1). vLLM suppresses it. Class LOSS.
- **`8-text_sandwich`** (`text_sandwich`) text → call → text; both text spans survive in order. This is also covered in: TOOLCALLING.streamv1.8.c.
- **`8-text_between_calls`** (`text_between_calls`) call → text → call; the inter-call prose survives (v2 recovers what v1 drops). This is also covered in: TOOLCALLING.streamv1.8.d.
- **`8-narrated_calls`** (`narrated_calls`) Multiple calls with narration between each — `tool_call → text → tool_call → text → tool_call`. The agentic call/narrate/call pattern; every call and inter-call text span is its own ordered event.
- **`8-native_quoted_control_in_response`** (`native_quoted_control_in_response`) A native control marker inside double-quoted visible response text stays literal prose.
- **`8-native_single_quoted_word_control`** (`native_single_quoted_word_control`) A control marker inside a single-quoted word example stays literal prose.
- **`8-native_quoted_control_then_call`** (`native_quoted_control_then_call`) Quoted control text stays visible, and the following real call still dispatches.
- **`8-native_unmatched_quote_then_call`** (`native_unmatched_quote_then_call`) An unmatched quote before a real control marker cannot hide the following call.
- **`8-native_quoted_incomplete_header`** (`native_quoted_incomplete_header`) An incomplete native tool header inside quoted response text stays literal.
- **`8-native_single_quote_contraction_response`** (`native_single_quote_contraction_response`) A control marker inside a contraction-bearing single-quoted sentence stays literal prose.

### Group 10 — Reasoning span (`REASONING.*`)
- **`10-reason_only`** (`reason_only`) Reasoning span, nothing else. This is also covered in: REASONING.batch.2.a.
- **`10-reason_then_content`** (`reason_then_content`) Reasoning then visible content, no call. This is also covered in: e2e case-0001-chinese_arithmetic__non-stream-budget_capped.json (+ 42 more: every `reasoning/core`, `reasoning/complex` and `reasoning/history` case, `tool_none_arithmetic__*`, and the SECOND step of both `lifecycle_*` — each with its `-budget_unlimited` pair).
- **`10-two_reason_spans`** (`two_reason_spans`) Two reasoning spans separated by content. Batch reasoning merges them → Class MERGE. This is also covered in: REASONING.batch.6.a.
- **`10-reason_unterminated`** (`reason_unterminated`) Stream ends inside reasoning; open reasoning promoted at finish.
- **`10-two_adjacent_reason_spans`** (`two_adjacent_reason_spans`) Two reasoning spans with nothing between them, then the answer. The single `reasoning_text` field every batch parser exposes can only concatenate them, so adjacent spans JOIN with a newline. The counterpart — two spans separated by a call must NOT join — is pinned by `11-reason_after_tool` / `11-reason_interleaved`: a parser that always joins invents a newline the model never emitted.

### Group 11 — Reasoning ↔ tool interleaving (UNIQUE to unified; the unification gap)
- **`11-reason_then_tool`** (`reason_then_tool`) Reasoning fully precedes one call. Baseline ordering.
- **`11-reason_after_tool`** (`reason_after_tool`) Reasoning AFTER a call, then text (Example A). Class ORDER.
- **`11-reason_interleaved`** (`reason_interleaved`) reason → tool → reason → tool. Class MERGE.
- **`11-reason_tool_text_reason_tool`** (`reason_tool_text_reason_tool`) reason → tool → text → reason → tool. Class MERGE.
- **`11-interstitial_text`** (`interstitial_text`) reasoning → visible text → call; the middle text survives in order.
- **`11-content_then_reason_then_tool`** (`content_then_reason_then_tool`) Content BEFORE reasoning, then a call. Class ORDER (Dynamo hoists reasoning).
- **`11-content_then_reason`** (`content_then_reason`) content → reasoning → content. Class ORDER.
- **`11-reason_tool_reason_tool_reason`** (`reason_tool_reason_tool_reason`) Each call wrapped by its own thought, trailing thought too. Class MERGE.
- **`11-reason_between_calls`** (`reason_between_calls`) call → reasoning → call; reasoning survives BETWEEN two calls. Class MERGE.
- **`11-text_reason_tool_text_reason_tool`** (`text_reason_tool_text_reason_tool`) Deep well-formed interleave — text → reason → tool → text → reason → tool; user text, reasoning, and calls all mix in one stream, every segment in order. Class MERGE (batch hoists both thoughts).

### Group 12 — Adversarial nesting (a marker of one channel inside another; P7)
- **`12-reason_markup_in_arg`** (`reason_markup_in_arg`) "Tool call contains reasoning" — a reasoning-channel marker sits inside a quoted tool-arg VALUE. NOT a leak: an arg value is data bound for the function, not a rendered channel, so by I7 the parser preserves it byte-exact (the gemma4 native UnifiedParser confirms the golden exactly). A reasoning-first extractor lifts it out and corrupts the arg. Class ARG_MISMATCH / MERGE.
- **`12-tool_in_reason`** (`tool_in_reason`) "Reasoning contains tool call" — a well-formed tool-call envelope nested inside a reasoning span. OPPOSITE of 12-1: a reasoning span is opaque text (not a quoted data region), so a real tool-call marker inside it IS structural. Golden breaks out (reason → call → reason). Engines leak the tool markup into `reasoning_content` and drop the call. Class LEAK.
- **`12-reason_markup_in_arg_with_text`** (`reason_markup_in_arg_with_text`) 12-1 WITH visible narration before and after — all three channels at once (text / tool-call-with-markup-arg / text). Golden keeps text as text, the call clean, the markup byte-exact in the arg. Class ARG_MISMATCH / MERGE.
- **`12-tool_in_reason_with_text`** (`tool_in_reason_with_text`) 12-2 WITH visible narration before and after — text → reason → call → reason → text. Golden breaks out and keeps the surrounding text; engines leak the nested markup. Class LEAK.

### DeepSeek V4.1 applicability
- DeepSeek V4.1 uses the ordered Unified contract for native DSML calls, reasoning interleaving, guided JSON, prefilled states, and dotted tool names. The current corpus emits 276 of the 379 taxonomy cases for this family.
- Generated cases include `30-guided_json_gt_in_argument_bare_opener`, the single-family `deepseek-json_invocation_body` case, applicable Guided Decoding cases in groups 31 through 35 (exclusions listed below), the marker-discriminating Response row `50-prefilled_response_reasoning_markers_literal`, and the 40-series cases `40-prefilled_reasoning_with_guided_json`, `40-native_quoted_control_in_reasoning`, and `40-native_single_quote_contraction_reasoning`, plus `41-prefilled_reasoning_redundant_opener` through `41-prefilled_reasoning_truncated`.
- The 103 omitted cases require another family's framing or schema, or repeat a transition already covered for DeepSeek V4.1. Family applicability comes from `scenario_families()` and the serialized request. Family-specific grammars and schemas require explicit exclusions; Response-prefilled native object, array, and string ownership now cover each applicable native family. Prefilled rows that duplicate another state transition remain excluded for DeepSeek V4.1.

- `30-guided_json_gt_in_argument_bare_opener` retains the historical bare header with no tool name. `34-guided_json_narrated_invoke_in_reasoning` uses an unfinished DSML invoke header inside reasoning rather than a completed calls-block opener.

<!-- TODO: Verify the historical aliases for the 14 cases deferred from PR #232 after the taxonomy merge: 1-2, 30-14, 31-31 through 31-40, and 50-1/2. Preserve their historical IDs. -->

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
| `10-reason_then_content` | `chinese_arithmetic__non-stream` | `end-to-end case-0001-chinese_arithmetic__non-stream-budget_capped.json` |
| `10-reason_then_content` | `chinese_arithmetic__non-stream` | `end-to-end case-0001-chinese_arithmetic__non-stream-budget_unlimited.json` |
| `10-reason_then_content` | `chinese_arithmetic__stream` | `end-to-end case-0002-chinese_arithmetic__stream-budget_capped.json` |
| `10-reason_then_content` | `chinese_arithmetic__stream` | `end-to-end case-0002-chinese_arithmetic__stream-budget_unlimited.json` |
| `10-reason_then_content` | `compare_fractions__non-stream` | `end-to-end case-0003-compare_fractions__non-stream-budget_capped.json` |
| `10-reason_then_content` | `compare_fractions__non-stream` | `end-to-end case-0003-compare_fractions__non-stream-budget_unlimited.json` |
| `10-reason_then_content` | `compare_fractions__stream` | `end-to-end case-0004-compare_fractions__stream-budget_capped.json` |
| `10-reason_then_content` | `compare_fractions__stream` | `end-to-end case-0004-compare_fractions__stream-budget_unlimited.json` |
| `10-reason_then_content` | `history_not_preserved__non-stream` | `end-to-end case-0013-history_not_preserved__non-stream-budget_capped.json` |
| `10-reason_then_content` | `history_not_preserved__non-stream` | `end-to-end case-0013-history_not_preserved__non-stream-budget_unlimited.json` |
| `10-reason_then_content` | `history_not_preserved__stream` | `end-to-end case-0014-history_not_preserved__stream-budget_capped.json` |
| `10-reason_then_content` | `history_not_preserved__stream` | `end-to-end case-0014-history_not_preserved__stream-budget_unlimited.json` |
| `10-reason_then_content` | `history_preserved_addition__non-stream` | `end-to-end case-0015-history_preserved_addition__non-stream-budget_capped.json` |
| `10-reason_then_content` | `history_preserved_addition__non-stream` | `end-to-end case-0015-history_preserved_addition__non-stream-budget_unlimited.json` |
| `10-reason_then_content` | `history_preserved_addition__stream` | `end-to-end case-0016-history_preserved_addition__stream-budget_capped.json` |
| `10-reason_then_content` | `history_preserved_addition__stream` | `end-to-end case-0016-history_preserved_addition__stream-budget_unlimited.json` |
| `10-reason_then_content` | `history_preserved_codeword__non-stream` | `end-to-end case-0017-history_preserved_codeword__non-stream-budget_capped.json` |
| `10-reason_then_content` | `history_preserved_codeword__non-stream` | `end-to-end case-0017-history_preserved_codeword__non-stream-budget_unlimited.json` |
| `10-reason_then_content` | `history_preserved_codeword__stream` | `end-to-end case-0018-history_preserved_codeword__stream-budget_capped.json` |
| `10-reason_then_content` | `history_preserved_codeword__stream` | `end-to-end case-0018-history_preserved_codeword__stream-budget_unlimited.json` |
| `10-reason_then_content` | `history_unicode__non-stream` | `end-to-end case-0019-history_unicode__non-stream-budget_capped.json` |
| `10-reason_then_content` | `history_unicode__non-stream` | `end-to-end case-0019-history_unicode__non-stream-budget_unlimited.json` |
| `10-reason_then_content` | `history_unicode__stream` | `end-to-end case-0020-history_unicode__stream-budget_capped.json` |
| `10-reason_then_content` | `history_unicode__stream` | `end-to-end case-0020-history_unicode__stream-budget_unlimited.json` |
| `10-reason_then_content` | `logic_syllogism__non-stream` | `end-to-end case-0021-logic_syllogism__non-stream-budget_capped.json` |
| `10-reason_then_content` | `logic_syllogism__non-stream` | `end-to-end case-0021-logic_syllogism__non-stream-budget_unlimited.json` |
| `10-reason_then_content` | `logic_syllogism__stream` | `end-to-end case-0022-logic_syllogism__stream-budget_capped.json` |
| `10-reason_then_content` | `logic_syllogism__stream` | `end-to-end case-0022-logic_syllogism__stream-budget_unlimited.json` |
| `10-reason_then_content` | `long_context_retrieval__non-stream` | `end-to-end case-0023-long_context_retrieval__non-stream-budget_capped.json` |
| `10-reason_then_content` | `long_context_retrieval__non-stream` | `end-to-end case-0023-long_context_retrieval__non-stream-budget_unlimited.json` |
| `10-reason_then_content` | `long_context_retrieval__stream` | `end-to-end case-0024-long_context_retrieval__stream-budget_capped.json` |
| `10-reason_then_content` | `long_context_retrieval__stream` | `end-to-end case-0024-long_context_retrieval__stream-budget_unlimited.json` |
| `10-reason_then_content` | `minutes_to_seconds__non-stream` | `end-to-end case-0025-minutes_to_seconds__non-stream-budget_capped.json` |
| `10-reason_then_content` | `minutes_to_seconds__non-stream` | `end-to-end case-0025-minutes_to_seconds__non-stream-budget_unlimited.json` |
| `10-reason_then_content` | `minutes_to_seconds__stream` | `end-to-end case-0026-minutes_to_seconds__stream-budget_capped.json` |
| `10-reason_then_content` | `minutes_to_seconds__stream` | `end-to-end case-0026-minutes_to_seconds__stream-budget_unlimited.json` |
| `10-reason_then_content` | `multiline_checksum__non-stream` | `end-to-end case-0027-multiline_checksum__non-stream-budget_capped.json` |
| `10-reason_then_content` | `multiline_checksum__non-stream` | `end-to-end case-0027-multiline_checksum__non-stream-budget_unlimited.json` |
| `10-reason_then_content` | `multiline_checksum__stream` | `end-to-end case-0028-multiline_checksum__stream-budget_capped.json` |
| `10-reason_then_content` | `multiline_checksum__stream` | `end-to-end case-0028-multiline_checksum__stream-budget_unlimited.json` |
| `10-reason_then_content` | `multiply_17_19__non-stream` | `end-to-end case-0029-multiply_17_19__non-stream-budget_capped.json` |
| `10-reason_then_content` | `multiply_17_19__non-stream` | `end-to-end case-0029-multiply_17_19__non-stream-budget_unlimited.json` |
| `10-reason_then_content` | `multiply_17_19__stream` | `end-to-end case-0030-multiply_17_19__stream-budget_capped.json` |
| `10-reason_then_content` | `multiply_17_19__stream` | `end-to-end case-0030-multiply_17_19__stream-budget_unlimited.json` |
| `10-reason_then_content` | `parity_expression__non-stream` | `end-to-end case-0031-parity_expression__non-stream-budget_capped.json` |
| `10-reason_then_content` | `parity_expression__non-stream` | `end-to-end case-0031-parity_expression__non-stream-budget_unlimited.json` |
| `10-reason_then_content` | `parity_expression__stream` | `end-to-end case-0032-parity_expression__stream-budget_capped.json` |
| `10-reason_then_content` | `parity_expression__stream` | `end-to-end case-0032-parity_expression__stream-budget_unlimited.json` |
| `10-reason_then_content` | `python_loop_trace__non-stream` | `end-to-end case-0033-python_loop_trace__non-stream-budget_capped.json` |
| `10-reason_then_content` | `python_loop_trace__non-stream` | `end-to-end case-0033-python_loop_trace__non-stream-budget_unlimited.json` |
| `10-reason_then_content` | `python_loop_trace__stream` | `end-to-end case-0034-python_loop_trace__stream-budget_capped.json` |
| `10-reason_then_content` | `python_loop_trace__stream` | `end-to-end case-0034-python_loop_trace__stream-budget_unlimited.json` |
| `10-reason_then_content` | `sequence_next__non-stream` | `end-to-end case-0035-sequence_next__non-stream-budget_capped.json` |
| `10-reason_then_content` | `sequence_next__non-stream` | `end-to-end case-0035-sequence_next__non-stream-budget_unlimited.json` |
| `10-reason_then_content` | `sequence_next__stream` | `end-to-end case-0036-sequence_next__stream-budget_capped.json` |
| `10-reason_then_content` | `sequence_next__stream` | `end-to-end case-0036-sequence_next__stream-budget_unlimited.json` |
| `10-reason_then_content` | `set_intersection__non-stream` | `end-to-end case-0037-set_intersection__non-stream-budget_capped.json` |
| `10-reason_then_content` | `set_intersection__non-stream` | `end-to-end case-0037-set_intersection__non-stream-budget_unlimited.json` |
| `10-reason_then_content` | `set_intersection__stream` | `end-to-end case-0038-set_intersection__stream-budget_capped.json` |
| `10-reason_then_content` | `set_intersection__stream` | `end-to-end case-0038-set_intersection__stream-budget_unlimited.json` |
| `10-reason_then_content` | `sort_integers__non-stream` | `end-to-end case-0039-sort_integers__non-stream-budget_capped.json` |
| `10-reason_then_content` | `sort_integers__non-stream` | `end-to-end case-0039-sort_integers__non-stream-budget_unlimited.json` |
| `10-reason_then_content` | `sort_integers__stream` | `end-to-end case-0040-sort_integers__stream-budget_capped.json` |
| `10-reason_then_content` | `sort_integers__stream` | `end-to-end case-0040-sort_integers__stream-budget_unlimited.json` |
| `10-reason_then_content` | `spanish_logic__non-stream` | `end-to-end case-0041-spanish_logic__non-stream-budget_capped.json` |
| `10-reason_then_content` | `spanish_logic__non-stream` | `end-to-end case-0041-spanish_logic__non-stream-budget_unlimited.json` |
| `10-reason_then_content` | `spanish_logic__stream` | `end-to-end case-0042-spanish_logic__stream-budget_capped.json` |
| `10-reason_then_content` | `spanish_logic__stream` | `end-to-end case-0042-spanish_logic__stream-budget_unlimited.json` |
| `10-reason_then_content` | `structured_json__non-stream` | `end-to-end case-0043-structured_json__non-stream-budget_capped.json` |
| `10-reason_then_content` | `structured_json__non-stream` | `end-to-end case-0043-structured_json__non-stream-budget_unlimited.json` |
| `10-reason_then_content` | `structured_json__stream` | `end-to-end case-0044-structured_json__stream-budget_capped.json` |
| `10-reason_then_content` | `structured_json__stream` | `end-to-end case-0044-structured_json__stream-budget_unlimited.json` |
| `10-reason_then_content` | `system_instruction__non-stream` | `end-to-end case-0045-system_instruction__non-stream-budget_capped.json` |
| `10-reason_then_content` | `system_instruction__non-stream` | `end-to-end case-0045-system_instruction__non-stream-budget_unlimited.json` |
| `10-reason_then_content` | `system_instruction__stream` | `end-to-end case-0046-system_instruction__stream-budget_capped.json` |
| `10-reason_then_content` | `system_instruction__stream` | `end-to-end case-0046-system_instruction__stream-budget_unlimited.json` |
| `10-reason_then_content` | `tool_none_arithmetic__non-stream` | `end-to-end case-0051-tool_none_arithmetic__non-stream-budget_capped.json` |
| `10-reason_then_content` | `tool_none_arithmetic__non-stream` | `end-to-end case-0051-tool_none_arithmetic__non-stream-budget_unlimited.json` |
| `10-reason_then_content` | `tool_none_arithmetic__stream` | `end-to-end case-0052-tool_none_arithmetic__stream-budget_capped.json` |
| `10-reason_then_content` | `tool_none_arithmetic__stream` | `end-to-end case-0052-tool_none_arithmetic__stream-budget_unlimited.json` |
| `10-reason_then_content` | `unicode_symbol_math__non-stream` | `end-to-end case-0067-unicode_symbol_math__non-stream-budget_capped.json` |
| `10-reason_then_content` | `unicode_symbol_math__non-stream` | `end-to-end case-0067-unicode_symbol_math__non-stream-budget_unlimited.json` |
| `10-reason_then_content` | `unicode_symbol_math__stream` | `end-to-end case-0068-unicode_symbol_math__stream-budget_capped.json` |
| `10-reason_then_content` | `unicode_symbol_math__stream` | `end-to-end case-0068-unicode_symbol_math__stream-budget_unlimited.json` |
| `10-reason_then_content` | `lifecycle_single_result__stream` | `end-to-end case-0129-lifecycle_single_result__stream-budget_capped.json` |
| `10-reason_then_content` | `lifecycle_single_result__stream` | `end-to-end case-0129-lifecycle_single_result__stream-budget_unlimited.json` |
| `30-guided_json_named_tool` | `tool_add_named__non-stream` | `end-to-end case-0047-tool_add_named__non-stream-budget_capped.json` |
| `30-guided_json_named_tool` | `tool_add_named__non-stream` | `end-to-end case-0047-tool_add_named__non-stream-budget_unlimited.json` |
| `30-guided_json_named_tool` | `tool_add_named__stream` | `end-to-end case-0048-tool_add_named__stream-budget_capped.json` |
| `30-guided_json_named_tool` | `tool_add_named__stream` | `end-to-end case-0048-tool_add_named__stream-budget_unlimited.json` |
| `30-guided_json_named_tool` | `tool_translate_named__stream` | `end-to-end case-0054-tool_translate_named__stream-budget_capped.json` |
| `30-guided_json_named_tool` | `tool_translate_named__stream` | `end-to-end case-0054-tool_translate_named__stream-budget_unlimited.json` |
| `30-guided_json_required_tool` | `lifecycle_single_result__stream` | `end-to-end case-0129-lifecycle_single_result__stream-budget_capped.json` |
| `30-guided_json_required_tool` | `lifecycle_single_result__stream` | `end-to-end case-0129-lifecycle_single_result__stream-budget_unlimited.json` |
| `30-guided_json_required_tool` | `lifecycle_chained_calculation__stream` | `end-to-end case-0145-lifecycle_chained_calculation__stream-budget_capped.json` |
| `30-guided_json_required_tool` | `lifecycle_chained_calculation__stream` | `end-to-end case-0145-lifecycle_chained_calculation__stream-budget_unlimited.json` |
| `30-guided_json_escaped_string_args` | `schema_escaped_unicode_string__non-stream` | `end-to-end case-0105-schema_escaped_unicode_string__non-stream-budget_capped.json` |
| `30-guided_json_escaped_string_args` | `schema_escaped_unicode_string__non-stream` | `end-to-end case-0105-schema_escaped_unicode_string__non-stream-budget_unlimited.json` |
| `30-guided_json_array_argument` | `schema_array__stream` | `end-to-end case-0108-schema_array__stream-budget_capped.json` |
| `30-guided_json_array_argument` | `schema_array__stream` | `end-to-end case-0108-schema_array__stream-budget_unlimited.json` |

## Request-scoped modes

Groups 1–12 vary the model OUTPUT. Groups 30–39 vary Guided Decoding request initialization: the resolved `UnifiedParserInit` the serving layer passed to `UnifiedParser::initialize_request` before any output arrived. Groups 40+ cover prefilled request state.

`starting_state` says which channel the rendered prompt already opened, so the model never emits that opener: `None` (it opens its own), `Reasoning` (the stream begins INSIDE a thought), `Response` (visible content is already open, so there is no reasoning channel at all and reasoning markers are ordinary text). `tool_output_mode` says whether the backend constrained decoding: `Native` (model markup) or `GuidedJson` (bare JSON — a NAMED choice sends that tool's arguments alone, a REQUIRED choice sends one call object or an array of them).

### Group 30 — Guided Decoding: baseline and argument fidelity
- **`30-guided_json_named_tool`** (`guided_json_named_tool`) `tool_choice` names a tool; the payload is that tool's arguments and the name comes from the request. This is also covered in: e2e case-0047-tool_add_named__non-stream-budget_capped.json, e2e case-0048-tool_add_named__stream-budget_capped.json, e2e case-0054-tool_translate_named__stream-budget_capped.json (each with its `-budget_unlimited` pair).
- **`30-guided_json_required_tool`** (`guided_json_required_tool`) Required choice; the payload is an array of call objects. This is also covered in: e2e case-0129-lifecycle_single_result__stream-budget_capped.json, e2e case-0145-lifecycle_chained_calculation__stream-budget_capped.json (FIRST step of each; both with their `-budget_unlimited` pair).
- **`30-guided_json_two_calls`** (`guided_json_two_calls`) Two DIFFERENT tools in one array. Multi-call is the array's ordinary shape, not an edge case.
- **`30-guided_json_escaped_string_args`** (`guided_json_escaped_string_args`) An argument value carrying non-ASCII, escaped quotes and Windows backslashes. Native mode covers the same value in `7.*`, but there the value is raw text between markers and no escaping is involved — the escaping is only the parser's problem on this path. This is also covered in: e2e case-0105-schema_escaped_unicode_string__non-stream-budget_capped.json (and its `-budget_unlimited` pair).
- **`30-guided_json_array_argument`** (`guided_json_array_argument`) An argument VALUE that is an array, not a scalar. Distinct from `30-guided_json_required_tool`/`30-guided_json_two_calls`, where the array is the list OF CALLS one level up. A list arriving as its string rendering is a silently wrong call, not a failed one. This is also covered in: e2e case-0108-schema_array__stream-budget_capped.json (and its `-budget_unlimited` pair).
- **`30-guided_json_after_reasoning`** (`guided_json_after_reasoning`) A normal thought, THEN the constrained payload. Every other guided case starts at the payload, so nothing pinned the ordinary shape where the model reasons first and the backend constrains only the call. This is the baseline group 31's surroundings cases contrast with.
- **`30-guided_json_marker_inside_argument`** (`guided_json_marker_inside_argument`) A control marker of the family's OWN grammar inside a guided argument VALUE. Once the payload has opened, a marker is argument DATA and must survive byte-exact (`I7`); re-reading it as a channel token corrupts the call the tool receives while still looking like a successful dispatch. The golden argument is the family's own marker, not a placeholder — a stand-in would pass whatever the parser did.

- **`30-guided_json_gt_in_argument_trailing_close`** (`guided_json_gt_in_argument_trailing_close`), **`30-guided_json_gt_in_argument_wrapped`** (`guided_json_gt_in_argument_wrapped`), and **`30-guided_json_gt_in_argument_bare_opener`** (`guided_json_gt_in_argument_bare_opener`) keep a literal `>` inside a guided argument while crossing each tool-markup surrounding. They pin that a header scan cannot borrow the argument character as its terminator.

### Group 31 — Guided Decoding: invalid JSON or call structure
- **`31-guided_json_invalid_call`** (`guided_json_invalid_call`) Valid JSON that is not a call (no `name`). Surfaces as text under the guided malformed-payload policy; no call dispatched.
- **`31-guided_json_malformed_json`** (`guided_json_malformed_json`) JSON that does not parse — a truncated object, what a constrained decode looks like when the budget runs out.
- **`31-guided_json_partial_calls`** (`guided_json_partial_calls`) The array parses but one element is not a call.
- **`31-guided_json_list_with_broken_element`** (`guided_json_list_with_broken_element`) `[<valid call>, <broken JSON>]` — the array itself does not parse, so per-element recovery never runs.

### Group 32 — Guided Decoding: tool markup around the payload
- **`32-guided_json_tool_open_before_payload`** (`guided_json_tool_open_before_payload`) A native tool OPENER precedes the payload. Guided decoding delivers the call as JSON, so leading markup is stray: strip it, or it enters the payload buffer, breaks the parse and costs the call.
- **`32-guided_json_tool_close_after_payload`** (`guided_json_tool_close_after_payload`) A native tool CLOSER follows the payload. Markers can BRACKET a payload, not only precede it — once the opening brace latches visible-only, every later byte is appended verbatim.
- **`32-guided_json_wrapped_in_tool_markup`** (`guided_json_wrapped_in_tool_markup`) Opener AND closer, the shape a template emits when guided decoding is applied INSIDE a tool block. Handling one end only still loses the call.
- **`32-guided_json_orphan_tool_close_before_payload`** (`guided_json_orphan_tool_close_before_payload`) An orphan tool CLOSER. Paired with `32-guided_json_tool_open_before_payload`: while the closer was stripped and the opener beside it was not, which marker leaked depended on which one the model happened to emit.
- **`32-guided_json_native_markup_only`** (`guided_json_native_markup_only`) Guided mode receives one complete native tool call instead of bare JSON. The turn is control markup and emits no events; every stream split must match the whole-input result instead of leaking the parameter body as visible text.

### Group 33 — Guided Decoding: invalid payload plus tool markup

- **`33-guided_json_syntax_error_trailing_close`** through **`33-guided_json_syntax_error_bare_opener`** cross a JSON syntax error with a trailing closer, a full wrapper, and a bare opener. **`33-guided_json_schema_error_not_a_call_trailing_close`** through **`33-guided_json_schema_error_not_a_call_bare_opener`** make the payload a schema-invalid non-call under the same three surroundings. **`33-guided_json_schema_error_nameless_element_trailing_close`** through **`33-guided_json_schema_error_nameless_element_bare_opener`** do the same for a nameless array element. Each remains text: invalid payload structure must not dispatch a partial call.

### Group 34 — Guided Decoding: reasoning boundaries
- **`34-guided_json_narrated_invoke_in_reasoning`** (`guided_json_narrated_invoke_in_reasoning`) The model NARRATES a tool opener while thinking, then the real call arrives as JSON. The reasoning channel is unconstrained under guided decoding, so that markup is prose; treating it as structure ends the turn and discards the payload.
- **`34-guided_json_prose_before_reasoning`** (`guided_json_prose_before_reasoning`) Visible prose, then a thought, then the payload. Every other guided case opens its thought at byte 0; with prose first the run can latch the payload buffer and surface the model's private thinking to the user as the answer.
- **`34-guided_json_orphan_reason_close_before_payload`** (`guided_json_orphan_reason_close_before_payload`) An orphan reasoning CLOSER with nothing open. The native scanner strips a stray closer wherever it appears before an opener; guided must agree or the same bytes read differently by request mode (`I3`).
- **`34-guided_json_unterminated_reasoning_then_wrapped_payload`** (`guided_json_unterminated_reasoning_then_wrapped_payload`) A thought whose closer never arrives, running straight into native tool markup wrapping the guided payload. `32-guided_json_wrapped_in_tool_markup` pins a wrapper around the payload OUTSIDE reasoning and `41.*` pins an unterminated thought on its own; neither asks what happens when the two meet, and that crossing is where both native families emitted the payload as REASONING and dispatched nothing. The client sees a plausible answer and never learns a call was lost. Contrast with `34-guided_json_narrated_invoke_in_reasoning`, where the same markup has PROSE behind it and is narration — what separates them is whether the guided payload follows, not which marker appeared.
- **`34-guided_json_bare_tool_header_recovers_inside_a_thought`** (`guided_json_bare_tool_header_recovers_inside_a_thought`) starts in Reasoning and routes through a native tool boundary into guided JSON. Both cases are generated for every supported family, with its own marker grammar; neither absence nor an `UNSUPPORTED` cell can hide a missing family input.
- **`34-guided_quoted_reasoning_closer_named`** (`guided_quoted_reasoning_closer_named`) A quoted reasoning closer does not end prefilled reasoning before the named-tool payload.
- **`34-guided_quoted_reasoning_closer_required`** (`guided_quoted_reasoning_closer_required`) A quoted reasoning closer does not end prefilled reasoning before the required-tool payload.

### Group 35 — Guided Decoding: markers in visible answers
- **`35-guided_json_quoted_bare_header_in_answer`** (`guided_json_quoted_bare_header_in_answer`) A response that already has a visible channel open contains its family's reasoning marker before the guided payload. The marker must not reopen a private channel, and the following JSON must still dispatch.
- **`35-guided_json_quoted_bare_header_after_payload`** (`guided_json_quoted_bare_header_after_payload`) crosses the same Response boundary after the payload has already dispatched: call, then visible control-markup text.
- **`35-guided_response_quoted_control_braces_named`** (`guided_response_quoted_control_braces_named`) A quoted native opener with braces stays visible before a named-tool payload; the quoted brace cannot claim payload ownership.
- **`35-guided_response_quoted_control_braces_required`** (`guided_response_quoted_control_braces_required`) The same quoted-control boundary holds for a required-tool payload.

`31-guided_json_partial_calls` and `31-guided_json_list_with_broken_element` pin **all-or-nothing**: one bad element voids the whole array and the payload goes out as text, taking the valid call with it. That is deliberate. A tool call is a side effect, so dispatching one extracted from a document that failed validation fails OPEN. Text loses nothing — the raw payload stays visible. `31-guided_json_invalid_call` through `31-guided_json_list_with_broken_element` each also emit `tracing::warn!(why = "unified_guided_json_not_a_tool_call")`: the events alone are indistinguishable from a model that chose to answer in prose, so the log is the only signal the backend's guided decoding failed.

`31-guided_json_invalid_call` through `31-guided_json_list_with_broken_element` are malformed PAYLOADS. `32-guided_json_tool_open_before_payload` through `32-guided_json_orphan_tool_close_before_payload` are well-formed payloads in malformed SURROUNDINGS: they recover the markers, the payload then parses, and the call dispatches — so neither the all-or-nothing rule nor that warning applies to them.

**Peer-engine value here is intentionally limited.** The base vLLM captures do not exercise the serialized guided request configuration. Historical backfills use tagged release sources when no producer fingerprint was retained, or sources that match a retained producer fingerprint, including verified source reconstructions. They apply `init` when the API supports it; missing sources and unsupported initialization remain explicitly unavailable. Backfills append observations without rewriting recorded results.

The shared `35-guided_json_native_parameter_object_before_payload`, `35-guided_json_native_parameter_array_before_payload`, `35-guided_json_native_parameter_string_before_payload`, and `35-guided_json_noncall_before_native_array_before_payload` cases place native argument data before the real guided call. Only the final Paris call dispatches. These cases test lexical ownership, including arguments that do not satisfy the offered native schema. A preceding non-call JSON value remains visible without releasing later native arguments to the guided scanner.

- **`35-guided_response_native_wrapper_whitespace_before_prose`** tests all nine families in Response-prefilled guided mode: suppress a native invocation wrapper and its structural whitespace, preserve the following prose, then dispatch only the guided call. Muse uses the inner ATEM invocation; full routed-message recovery is a separate contract. The native argument tests lexical ownership rather than offered-schema validation.

### Group 36 — Guided Decoding: response schemas after reasoning

- **`36-structured_response_after_reasoning`** (`structured_response_after_reasoning`) No-tools response schema with non-JSON Unicode reasoning followed by a constrained weather object. The request carries `init.structural_tag.schema` and complete rejected streams. Capture, parity, and rendering build the actual family structural tag, check the accepted stream and every character split with XGrammar, reject malformed JSON, arrays, missing/extra keys and invalid enum values, then parse the accepted stream into separate reasoning and response events. GPT-OSS generates its analysis/final headers after the consumed assistant role; Qwen3, GLM and DeepSeek V4 begin in prompt-prefilled reasoning. DeepSeek V4.1, Gemma, Kimi and Muse are excluded because they have no registered structural-tag builder. Parser-only peer adapters report this generation contract as unsupported. Refs [Dynamo #16787](https://github.com/ai-dynamo/dynamo/issues/16787).

### Group 40 — Prefilled reasoning, happy
- **`40-prefilled_reasoning_with_tool`** (`prefilled_reasoning_with_tool`) Stream begins inside a thought, closes it, calls a tool.
- **`40-prefilled_reasoning_with_guided_json`** (`prefilled_reasoning_with_guided_json`) Same, with the call as guided JSON.
- **`40-prefilled_reasoning_then_text_then_tool`** (`prefilled_reasoning_then_text_then_tool`) reasoning → visible prose → call. All three channels in one prefilled stream.
- **`40-prefilled_reasoning_then_text`** (`prefilled_reasoning_then_text`) reasoning → prose, no call. Pins that closing a prefilled thought returns the stream to VISIBLE content rather than leaving it in reasoning, which would swallow the whole answer.
- **`40-native_quoted_control_in_reasoning`** (`native_quoted_control_in_reasoning`) A quoted native control marker inside a prefilled reasoning span remains literal reasoning text.
- **`40-native_single_quote_contraction_reasoning`** (`native_single_quote_contraction_reasoning`) A control marker inside a contraction-bearing quote remains literal in prefilled reasoning.

### Group 41 — Prefilled reasoning, malformed
- **`41-prefilled_reasoning_redundant_opener`** (`prefilled_reasoning_redundant_opener`) The backend re-emits the `<think>` the prompt already wrote. Exactly one echo is consumed, not leaked; a second would be stray markup and stripped (I3). The only case where a prefilled stream legitimately carries an opener.
- **`41-prefilled_reasoning_truncated`** (`prefilled_reasoning_truncated`) Budget runs out mid-call. Keep the completed reasoning, drop the partial call (P2).

### Group 50 — Prefilled response
- **`50-prefilled_response_reasoning_markers_literal`** (`prefilled_response_reasoning_markers_literal`) `<think>literal</think>` must reach the user as TEXT, markers and all, because this stream has no reasoning channel. It is the direct visible-marker regression. Beam's reasoning markers are special tokens, so its authored golden differs: the input carries the full framed opener (`<|start|>assistant<|channel|>analysis<|message|>literal<|end|>`), and under Response the stream has no reasoning channel, so that framing is consumed as control (never leaked) while its body `literal` is visible TEXT that joins the prose after it (`literal then a call`); the following call still dispatches.

The marker-free prefilled-Response variants were removed because they emitted the same observable result as their default-state peers. Group 50 retains reasoning-marker stimuli that distinguish Response from default initialization; the ordinary native, guided, multi-call, and malformed payload contracts remain covered by groups 8, 30, and 31.

### GPT-OSS-specific

These cases exercise Harmony message framing, which the other eight grammars do not use. Shared post-wrapper whitespace coverage stays in group 35.

- **`gpt_oss-tool_body_whitespace`** preserves the native argument object when whitespace follows the Harmony message marker.
- **`gpt_oss-constrained_final_header`** retains visible JSON under a final header whose only metadata is its constrained content type.
- **`gpt_oss-implicit_recipient_first_header`** holds the generated recipient-first header until its message body opens, without prompt token IDs or leaked recipient text.
- **`gpt_oss-quoted_truncated_tool_then_final`** recovers the later final answer after an unterminated native argument string; the ambiguous suffix stays buffered until EOF establishes that the original JSON cannot complete.
- **`gpt_oss-quoted_message_header_literal`** is the valid counterfactual: a complete header inside a valid JSON string remains argument data.
- **`gpt_oss-complete_alternate_message_ends`** recovers complete JSON calls closed by Harmony message-end or turn-end. The legacy tool adapter continues to require handoff, and malformed provisional calls never receive completion.

### DeepSeek V4-specific

- **`35-guided_response_rejected_header_quote_ownership`** (`guided_response_rejected_header_quote_ownership`) DeepSeek V4 rejects an incomplete invoke header in Response; its attribute quote cannot protect the following parameter markup as prose.

This case uses `UNIFIED.35-guided_response_rejected_header_quote_ownership` and retains its published `UNIFIED.35-5` ID as a historical alias. DeepSeek V4 independently recognizes parameter markers after rejecting an incomplete invoke header. DeepSeek V4.1 and Muse do not register parameter elements as independent guided control markers; GLM and Qwen do not use quoted invoke-name attributes, and Gemma and Kimi use different call envelopes. The exact rejected-header quote and subsequent parameter-stripping contract is therefore DeepSeek V4-specific. The report groups it under DeepSeek V4-specific tests.

### DSML framing (DeepSeek V4 and V4.1)

The DeepSeek encoders render an assistant tool turn as `content + "\n\n" + calls`, also when the content is empty. Exactly that one separator is framing: a parser that returns it as content makes the client re-render `\n\n\n\n` before the calls block, so the history stops matching the generated tokens. Any other whitespace is content. Inputs use the encoder's own call layout, newlines included, and the goldens follow from that rule rather than from peer parity.

- **`dsml-separator_two_lf`**, **`dsml-separator_four_lf`**, and **`dsml-no_separator`**: one separator is consumed, extra newlines stay content, and an adjacent block removes nothing.
- **`dsml-empty_content_separator`** and **`dsml-reasoning_then_separator`**: a tool-only turn, with and without reasoning, emits no content. The reasoning variant is the common production shape `</think>\n\n<｜DSML｜tool_calls>`.
- **`dsml-bare_invoke_two_lf`** and **`dsml-bare_invoke_four_lf`**: a recovered bare invocation follows the same rule as a wrapped block.
- **`dsml-whitespace_content_separator`**: whitespace-only content keeps its own two newlines.

Chunk-boundary variants live in the legacy stream corpus (`TOOLCALLING.streamv1.50.b` and `.50.c`) because Unified identifies a case by its input, not its chunk schedule.

Scope: the separator is DeepSeek template framing, so other families have no equivalent behavior to test. vLLM Python keeps the separator as content (fix pending in vllm-project/vllm#58295); SGLang and vLLM's Rust frontend consume it. Before this fix, Dynamo v2 0.7.23 returned the separator as content in every case except `dsml-no_separator`.

### Gemma-specific

- **`gemma-guided_json_visible_call_prose_before_reasoning`** and **`gemma-guided_json_malformed_call_prefix_before_reasoning`** cover Gemma 4 guided call-prefix boundaries.

### GLM 5-specific

- **`glm5-parameterless_call_shape_inside_argument`** (`glm47_parameterless_call_shape_inside_argument`) places an offered parameterless-call shape inside an open GLM argument value. The embedded close/open markers remain argument data and must not dispatch a second call.

### Kimi-specific

- **`kimi-typed_argument_values`** through **`kimi-guided_native_wrapper`** cover Kimi K3 XTML typed arguments, raw JSON blocks, spacing variants, message termination, reasoning closure, recovery, and guided wrappers.

### Muse-specific

- **`muse-guided_json_quoted_bare_tool_header_in_answer`** is the Muse tool-recipient header inside visible answer text. Its old `31-26` capture key remains a historical alias; this column is intentionally absent for non-Muse families because it would duplicate `35-guided_json_quoted_bare_header_in_answer`.

- **`muse-unterminated_parameter_then_empty_call`** (`muse_unterminated_parameter_then_empty_call`) places parameterless invocation-shaped text inside a parameter that never closes. At EOF, discard the ambiguous outer invocation without dispatching the embedded call. This covers the missing conformance case raised in [PR #316](https://github.com/ai-dynamo/frontend-crates/pull/316#discussion_r4171144832).
- **`muse-unterminated_parameter_then_call`** (`muse_unterminated_parameter_then_call`) places a declared invocation-shaped sequence inside the outer parameter. Preserve the outer call and its complete argument through the parameter closer; do not guess that the embedded text is a new executable call. Both cases are Muse-only because they exercise the ATEM scanner's treatment of open parameters.
### Beam applicability

Beam (`<|start|>assistant[ to=NAME]<|channel|>analysis|final[ <|constrain|>xml]<|message|>… <|end|>|<|return|>|<|call|>`, NOT Harmony) joins every shared scenario with a Beam-native wire and an independently authored golden; the numeric IDs are unchanged. The grammar changes the bytes, not the meaning, except in three authored exceptions: `5-2` (a call never dispatches without its terminator), `5-3` (`<|call|>` inside `final` is text) and `50-4` (under Response the framed analysis message's body is text and its framing is consumed). Beam receives every shared null schema unchanged (`7-5.*`): its typing intersects a sibling `type` with anyOf/oneOf branches, a deliberate correction of the frozen reference's union, so no per-family "effective schema" is substituted. Quote, contraction, header and unmatched-quote scenarios (`native_quoted_*`, `native_single_quote*`, `native_unmatched_quote_then_call`) use `<|start|>assistant to=get_weather<|channel|>` as the quoted native control. The model builder's reference is the grammar's source of truth, not a peer engine, so the vLLM cell is `UNSUPPORTED` for every Beam case; no peer verdict is claimed.

- **`beam-empty_args_no_close`** (`empty_args_no_close`) A header for a strict tool with no properties, then EOS (no parameter tags, no `<|call|>`). Beam's serving grammar makes the parameter body optional for such a tool and lets EOS stand for the closer, so the parser emits `noop({})`. Beam-specific: the other grammars need an argument body or an inner closing marker.
- **`beam-optional_args_no_close`** (`optional_args_no_close`) The same bare header for a tool whose only property is optional. Nothing is required, so it is the call `noop({})`; a required argument keeps a bare header a truncation.

## Authoring a case: what to check BEFORE adding one

Numeric groups describe shared behaviors: inspect every registered family and author its native input and expected output wherever applicable. Named family sections hold grammar-specific cases; explain genuine exclusions. A bug discovered in one family does not justify restricting shared coverage to it. Preserve previously published IDs.

Every rule here exists because a case was added that could not fail for the reason it claimed. Fake coverage is worse than no coverage — it renders green.

1. **Is it distinguishable from an existing case?** Compare `init` AND input against the corpus. If some existing case has the same configuration and the same input shape, the new case tests nothing. Three groups were deleted for exactly this: a whole axis whose 51/52/53 cases had the same config and inputs as `1-tool_only`, differing only in a label the parser cannot read.
2. **Can it fail for the stated reason?** Write down what would have to break for the case to go red, then confirm the parser can even SEE that input. `finish_reason` cannot: `finish()` takes no argument, in Dynamo and in vLLM alike, so a case that varies only the finish reason varies nothing. If the axis is invisible to the parser, express it as an input shape instead — `length` becomes a TRUNCATED input, which is observable.
3. **Does the field already exist under another name?** A per-case `input_mode` was added that was a 1:1 alias of `init.starting_state` across every row, and could not diverge, because "where the stream starts" IS what the starting state encodes. Grep the case dict before adding a key.
4. **Measure the behavior, do not predict it.** Author the case, run the harness, read what the parser actually emitted, and THEN write the golden and the description around it. The all-or-nothing array semantics were found this way; predicting them would have produced a wrong golden that looked authoritative.
5. **A near-duplicate that survives must name its distinguishing stimulus.** If a case keeps a different request state or mode, say which input bytes make that configuration change the parser's decision, and name the default-state sibling it contrasts with. A serialized `init` value that the input cannot exercise is not a retained contract.
6. **The input must be a shape the declared `init` can actually produce.** Six guided-decoding scenarios rendered NATIVE model markup for gemma4 and kimi_k2 while declaring `tool_output_mode=GuidedJson` — a mode that constrains the model to bare JSON, so that markup is the one input it can never emit. They rendered green for a year because neither family had a unified parser to run them; the moment gemma4 got one, all six failed. Guided payloads are grammar-independent and are now written ONCE for every family (`every_family` in `gen_unified_golden.py`); only the reasoning envelope around them is per family.
7. **A per-family golden needs a per-family fill, not one family's bytes.** `50-prefilled_response_reasoning_markers_literal` asserts that the model's own reasoning markers reach the user as literal TEXT, and its golden hardcoded qwen3's `<think>literal</think>` for all three families. Use the `None`-placeholder fill (as `12-reason_markup_in_arg` does for an argument value) so the scenario stays shared and only the grammar-specific bytes differ.

## Verifying a change to the table

The affected family's selected current Dynamo Unified column must finish with **zero empty cells and zero red cells**.

1. **Write:** change the parser or capture path.
2. **Read:** render the table and inspect each affected popup's input, initialization, chunks, GOLDEN events, and current Dynamo events.
3. **Fix:** treat an empty current cell as missing capture data. Treat a red current cell as a parser defect unless the authored GOLDEN is demonstrably wrong.
4. **Regenerate:** publish the current plain-version YAML and required manifest changes, then rerender from the same worktree.
5. **Re-read:** inspect the rendered column again and repeat until both counts are zero.

Do not use `reason:`, `unavailable:`, a historical column, stale HTML, or an unsupported GOLDEN edit to make a current empty or red cell appear acceptable.

The model blob and the rendered page are different things. A cell can carry correct JSON and render nothing — the column-header popup shipped with `init` in every column and an empty config list, because the model that feeds `buildGrammarHtml` is assembled separately and dropped the field.

- Check the rendered DOM, not only a cell's aggregate `status` or the `conformance-model` JSON. A cell can report `status: ok` while `red_on_diff` and the comparison signatures still make it render red.
- Headless Chrome reports `(hover: hover) = false`, so hover listeners never attach and a naive hover test "fails" on the baseline too. Emulate with `--blink-settings=primaryHoverType=2,availableHoverTypes=2`.
- A synthetic `pointerenter` does NOT set CSS `:hover`. Use it to test JS behavior, a real pointer move to test CSS.
- The renderer and model tests use process-owned temporary stage directories, so concurrent runs do not delete each other's staged files.
- A `transform` on a cell makes it the containing block for its own popup AND scales it. Use shadow and filter for cell affordances; a transform silently breaks popup placement.

## Deferred (not in the U0 seed set)

TODO(#241): Add the shared `wrapped_saved_closer_partial_marker`, `bare_saved_closer`, `bare_parameterless_call`, and `guided_json_native_envelope_after_prose` cases, their non-GLM fixes, and the shared tool-schema checks. These are excluded from #234; its new coverage is GLM-only.

- **n>1 interleave** (`UNIFIED.interleave_n2.*`, the Example-B n>1 LOSS case) needs a multi-choice interleaved driver (extends PR #135's tool-only lanes to carry reasoning state). Its golden is per-choice, a different shape than the single-stream cases here. Author with the n>1 lane.

## A defect the corpus missed owes the corpus a case

Every bug found by a reviewer, another agent, or a probe — that the existing cases did NOT catch — is evidence of a missing case, and the fix is not complete until that case exists here. Prefer a taxonomy scenario over a unit test: a scenario runs for every family and every delivery schedule the harness drives, a unit test runs once for one family. Fall back to a unit test only when the schema cannot express the property, and say why in its doc comment.

Name the missing DIMENSION, not the example. `guided_json_stray_prefix_before_reasoning` and `guided_json_narrated_prefix_inside_reasoning` were added after a stray `<function=` header borrowed its `>` from a following thought opener and emitted the model's private reasoning as visible text. The example was one input; the untested axis was **which control marker owns a terminator when two compete** — and nothing in the corpus had ever asked that question.

The check is the count: if a review round produced N defects the corpus missed and the scenario count did not move, the holes are still open.

**A duplicate is worse than a gap.** Before adding, normalize `(input, init, golden)` across the corpus and drop any crossing that already exists. A generated product once recreated three hand-authored scenarios — 9 cases across families — inflating the count while testing nothing new, and leaving two names for one behaviour to drift apart. `test_no_two_scenarios_have_identical_behaviour` now enforces this.

## Shared schema conformance probes

Expected values are authored in `conformance/utils/src/glm47_schema_cases.json`, independently of parser captures. `schema_cases.py` owns behavior classification; `unified_taxonomy.py` uses it for report sections and ordering. Group 7 has 66 visible entries covering 165 authored scenarios. Display numbers are ranks; fixture IDs and historical aliases stay unchanged.

| Behavior section | Visible entries |
|---|---:|
| Text preservation | 3 |
| Declared types and enums | 12 |
| Type arrays and null interpretation | 6 |
| String constants | 10 |
| Unions, intersections, and ambiguous values | 17 |
| References and schema IDs | 14 |

String constants fold by literal into ten entries: Direct, Explicit string type, allOf, and Reference. anyOf and oneOf stay separate, each with six entries by alternative type; their variants cross String / Typed value with Forward / Reversed branch order. Existing null groups retain their variants. Each popup lists the child's input, initialization, applicable family, capture, and comparison. A failing child keeps its parent failing; missing captures remain distinguishable from inapplicable cases.

Detailed matrix labels show a small `{}` when the entry or any child has an explicit schema. Overview hides it. Hover or focus the existing label to open the popup (tap on touch devices). Its `{ } Schema` disclosure expands inline, using more width when space allows and scrolling within the available viewport height. Identical schema JSON appears once with every associated variant and applicable family. Shared fallback schemas do not get an indicator.

Historical schema-subgroup aliases below retain their original meaning. Active IDs use the probe name directly, for example `UNIFIED.7-declared_array` and `UNIFIED.7-const_array_allof`; subgroup headings and ordering remain separate metadata.

- **`UNIFIED.7-17.*`** Declared JSON argument types.
- **`UNIFIED.7-18.*`** String constants retain their literal spelling.
- **`UNIFIED.7-19.*`** anyOf constant and typed alternatives in both orders.
- **`UNIFIED.7-20.*`** oneOf constant and typed alternatives in both orders.
- **`UNIFIED.7-21.*`** Local references and sibling type intersections.
- **`UNIFIED.7-22.*`** Composition intersects union type hints.
- **`UNIFIED.7-23.*`** Enums preserve declared argument types.
- **`UNIFIED.7-24.*`** Type arrays and nullable argument interpretation.
- **`UNIFIED.7-25.*`** Ambiguous native strings retain established preference.

All registered families are inspected and use native grammar. v1 batch excludes Muse (no v1 parser) and Harmony text (the same batch grammar is covered by Harmony); legacy stream covers the Harmony text path rather than inventing token IDs. Families without a Dynamo v2 parser carry explicit unavailability. Unified covers every registered Unified family. Unresolved/cyclic defensive schemas and the deliberately schema-invalid typed-enum control stay in Rust tests rather than successful conformance goldens. Identical schema/null-string controls are represented once in conformance.

Ambiguous constant unions retain each family's established schema-valid interpretation: Qwen and MiniMax can choose the typed JSON alternative, while GLM prefers a matching string constant. MiniMax M3 uses JSON object/array bodies in these probes; nested native tags do not encode an unconstrained numeric child type. Captures from unrelated families that violate these independent goldens remain visible rather than rewriting expectations.


### Muse Spark phase-preserving cases

`commentary_then_tool` is in the named Muse Spark section: commentary is a reserved visible recipient only in Spark. `spark_phase_tool_final`, `spark_phase_repeated`, `spark_phase_empty`, and `spark_phase_eof` opt in to `init.preserve_message_phases` and assert ordered `message_start`/`message_end` events, including repeated/empty messages and an unclosed message at EOF. Other registered families do not currently expose this reserved recipient through the phase API. Ordinary shared cases keep phase preservation disabled.

Muse reasoning treats `to=self` in prose as literal text. Complete framed headers remain structural. Guided reasoning suppresses complete native tool invocations as a control span, consistently with the shared group-34 GOLDEN, including any framing tokens inside a parameter. Incomplete native headers remain ordinary thought text after the generic prefix handling; they must not consume a thought terminator.

`muse_opaque_undeclared_invoke`, `muse_opaque_declared_invoke`, and `muse_recovery_second_parameter` cover Glimmer and Spark. ATEM argument text remains opaque regardless of whether an embedded tool name is declared or which parameter contains it. Recovery cannot distinguish a malformed call from valid argument text and must not invent executable calls. Other registered families use different argument grammars; shared group-12 scenarios cover their argument opacity. The existing `structured_response_after_reasoning` scenario also covers Spark’s native final-answer envelope and checks it with XGrammar.

`muse_unterminated_reasoning_native` and `muse_unterminated_reasoning_guided` pin an intentional distinction for both Muse families: native reasoning keeps an incomplete invocation header as literal prose, while JSON-guided reasoning strips the native invoke-start control and retains its following prose. Complete native invocations in guided reasoning remain stray controls owned by the shared guided reader. Other registered families do not use ATEM invoke headers.

The local nullable-reference, string-reference, and mixed null/string field probes apply to every registered Unified family. The reference probes retain unresolved references; all three probes use family-native value encoding. Existing published requests remain byte-identical; newly applicable family/case pairs have separate authored requests and recorded results.

### XML custom-type compatibility

The shared group 7 cases `xml_string_type_single_quoted_object` and `xml_string_type_double_quoted_object` preserve the exact string spelling in all 11 Unified families using their native string encoding. Both quote spellings remain literal text, including in JSON-body grammars.

The Qwen3/Nemotron-specific XML section contains `xml_custom_type_single_quoted_object` and `xml_custom_type_double_quoted_object`. Both spellings become an object under the direct, unconstrained `type: custom` compatibility fallback owned by the Qwen XML grammar. `custom` is not a JSON Schema type, so these probes remain separate from the schema-valid suite. GLM uses strict JSON parsing for JSON-looking bodies, MiniMax M3 preserves unknown scalar types as text, and JSON-body or explicitly typed grammars carry the argument type in their native encoding; they do not share this Python-literal conversion rule. MiniMax M2 exercises that fallback in the legacy streaming corpus and has no registered Unified parser.

Only display placement changes: the custom probes retain their stable `UNIFIED.7-xml_custom_type_*` IDs and capture history. They no longer occupy shared group 7 columns. Existing published IDs and older unmeasured observations remain unchanged.


### GPT-OSS Harmony headers and retained request variants

The six standalone GPT-OSS entries test recipient/channel ordering, the template JSON format label, a constrained final header without a tool recipient, an implicit recipient-first header, and complete tool messages ending with a message/turn terminator instead of handoff. Incomplete Harmony headers, interrupted reasoning, quoted controls, malformed bodies, and historical `_pr424` requests are variants beneath their shared semantic parents. `unified_taxonomy.case_fold()` owns these relationships, and `case_variants.group_unified_variants()` folds them through the same path as schema variants. Each popup retains the original case ID, request, initialization, chunk schedule, expected events, and independent observations; a failing or missing applicable child cannot become a passing parent.

The retained `gpt_oss_tool_body_whitespace`, `gpt_oss_quoted_truncated_tool_then_final`, and `gpt_oss_quoted_message_header_literal` names preserve published GPT-OSS identities while covering shared behavior. They fold under `tool_only`, `tool_no_close`, and `arg_marker_in_string`, respectively. JSON-body whitespace adds Kimi K2 and Kimi K3 requests; DeepSeek V4.1 already exercises leading whitespace in `deepseek_v41_json_invocation_body`, and Qwen/Muse already format native call bodies with structural newlines. GLM uses named `arg_key`/`arg_value` pairs rather than a whole-object JSON invocation body; whitespace inside a typed value belongs to argument typing. The malformed-body-to-response variant adds Muse's unterminated ATEM parameter followed by a user message and Kimi K3's unterminated raw JSON string followed by a framed response. The literal-header variant adds native Muse and Kimi K3 values containing a closer immediately followed by a complete user/response header. Families without separate response headers retain the existing delimiter-in-argument coverage instead of receiving artificial message syntax. Kimi K3's malformed raw JSON followed by a response currently loses that response. Its expected visible answer remains the authored oracle, and the missing answer stays a measured divergence rather than an applicability exclusion.

The retained unknown-name/empty-argument probe now runs in all 11 families with an empty offered-tool list. Untyped Beam, Qwen XML, and GLM values retain their scalar string spelling; JSON and typed grammars preserve the number. Native controls inside JSON strings and malformed whole-object JSON recovery also run in DeepSeek V4, DeepSeek V4.1, Kimi K2, and Kimi K3. Both DSML grammars accept raw JSON invocation bodies as well as parameter tags. DeepSeek V4 and Kimi K2 retain an invalid named call with empty arguments; the strict JSON decoders should drop that invocation and preserve the following call. DeepSeek V4.1 currently aborts on the non-JSON body before the empty call, recorded as an error against that oracle. Gemma, GLM, Qwen XML, and ATEM parameter fences do not carry a whole-object JSON invocation body, so those probes do not transfer to their typed argument encodings. Beam uses its existing structured argument representation rather than a raw JSON tool body.

Next-message recovery adds Beam, Muse, and Kimi K3 inputs with a complete argument and an omitted outer terminator. Muse dispatches its completed invocation before the message boundary; Beam and Kimi K3 should preserve the following response. Kimi K3 currently loses that response, which remains a measured divergence. Elided reasoning-close recovery adds Beam and Muse; the existing Kimi K3 elided-close request supplies equivalent coverage through an explicit coverage alias. Families without a separate response header retain their ordinary reasoning-close contract. Existing interrupted-tool-in-reasoning requests already share the `tool_in_reason` contract.

The 30 active `_pr424` variants preserve #424's published requests. Eighteen differ from the existing parent only in delivery: a single push versus marker-aligned pushes. The other twelve preserve distinct header or payload bytes. They share headings rather than creating separate GPT-only behavior categories. A shared heading may be covered by a family-native child even when the original parent specimen does not apply to that family.

The redundant `gpt_oss_truncated_tool_at_eof_is_dropped` execution is retired because shared `tool_no_close` already tests a complete Harmony JSON body without its handoff. Its original request, expected events, aliases, and capture checkpoints remain in history. Next-message recovery remains a separate input variant under the same shared heading. Empty raw bodies, malformed JSON, quoted controls, and elided reasoning closers retain their distinct stimuli; grouping does not assert identical native recovery policies across families.

- `harmony_guided_json_gt_in_argument_bare_opener`: a JSON argument containing `>` settles a partial header.
- `harmony_guided_json_gt_in_argument_named_bare_opener`: the same boundary after a recipient name.
- `harmony_guided_json_narrated_prefix_inside_reasoning`: strip a narrated incomplete header without losing reasoning.
- `harmony_guided_json_schema_error_nameless_element_bare_opener`: preserve recovery text for a nameless call element.
- `harmony_guided_json_schema_error_not_a_call_bare_opener`: preserve recovery text for a non-call JSON object.
- `harmony_guided_json_stray_prefix_before_reasoning`: a partial header cannot claim the following reasoning body.
- `harmony_guided_json_syntax_error_bare_opener`: preserve incomplete JSON without leaking the header.
- `harmony_guided_json_tool_open_before_payload`: recover a valid call after a partial directed header.

GPT-OSS also authors the existing shared `arg_marker_in_string` and
`deepseek_v41_mixed_control_text_in_string` cases with Harmony control spellings.
Literal spellings supplied as ordinary tokens, rather than actual control-token
IDs, have independent native Rust streaming coverage.

GPT-OSS cases also cover channel-first and template tool headers, unknown and empty tools, control spellings inside argument strings, interrupted reasoning, invalid tool bodies followed by another call, and truncated messages at EOF or before resynchronization. Native arguments still stream before handoff; the legacy tool projection and the Unified parser share the runtime.
