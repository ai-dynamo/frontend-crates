# Unified Parser Cases (reasoning + content + tool calls, one ordered stream)

`TOOLCALLING.*` is deprecated. New cases, capture jobs, and normal reports use `UNIFIED.*`. Retain the archived evidence and historical tests; do not migrate or relabel their observations, read them during normal Unified regeneration, or derive Unified GOLDEN expectations from recorded parser output.

Reference taxonomy for the **unified** conformance surface: one parser owns the whole assistant-output grammar and emits ONE ordered event stream. Sibling stage docs: `REASONING_CASES.md` (reasoning only), `TOOLCALLING_CASES.md` / `TOOLCALLING_STREAMING_V1_CASES.md` (tool calls only). This surface is what those two cannot express — the ORDER between reasoning and tool calls, and reasoning that occurs *between* or *after* tool calls.

The golden corpus is authored by `conformance/utils/src/gen_unified_golden.py` (one scenario spec -> `conformance/unified/golden_spec/<family>.yaml` in the gitignored build tree); the committed canonical files under `conformance/fixtures-unified-v2/families/` are derived from it.

## Single-family case review

MiniMax M3 joins the existing shared native and guided corpus.
Its argument-preservation and schema-coercion regressions are also exercised
in `parsers/v2/tests/minimax_m3_unified.rs`, including every two-chunk split
and one-character delivery. Those tests assert streaming progress before a
call closes; assembled event parity alone cannot prove that timing contract.

Always evaluate single-family tests against every registered family when adding, changing, or reviewing them, following the [case applicability policy](../../../README.md#case-applicability). If another family can express the behavior, add family-native inputs and authored expected outputs for every applicable family and use a shared section. Otherwise, explain the grammar or request-mode limitation and place the case in a named family-specific test section. Preserve published IDs, capture history, and older unmeasured results when changing section placement.

The corpus guard checks every generic semantic case, independent of which family exposed the regression. A generic case must have native inputs in more than one registered family. Visual folding does not confer parent coverage: semantic variants must independently span families. Only explicit `CaseVariant.coverage_parent` aliases combine equivalent native requests or retained delivery variants; each retains its individual input, expectation, and captures. A standalone case with one family must gain applicable native inputs or move to a named family section. The rendered-model guard checks the resulting columns as well as the authored corpus.

## The oracle: GOLDEN is authored, not captured

## Capture version policy

Follow [the Unified storage contract](../../../README.md#unified-storage-contract-plain-versioned-yaml-only): plain semantic-version YAML only, with unchanged family output carried forward. Source SHA is optional first-capture origin metadata, never a capture name. This migration does not change the older stream or batch-on-stream storage. Historical back-captures are optional; follow the [historical back-capture policy](../../../README.md#historical-back-captures) and preserve each recorded request/result identity.

The original shared-family observations for `7-glm_ref_object`, `7-glm_ref_encoded_targets`, `7-glm_ref_json_looking_strings`, and `7-glm_ref_scalar_types` began in peer families at Dynamo v2 `0.7.18`. This run adds exact-version, request-bound executions for earlier releases and retains their execution receipts separately from original evidence. Original observations and origins remain preserved. GLM was captured on the released Dynamo v2 `0.7.8` source under PR #271, and its existing history is preserved.

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
- **P8 Whitespace-only text in a Nemotron V3 tool-only turn** -> call framing, not content. When a Nemotron V3 turn completes a tool call and shows no visible (non-whitespace) text, its whitespace-only text is dropped. A turn with any visible text keeps every whitespace byte, and whitespace around a call that never completes (truncated, or invalidated) is visible content, as is whitespace with no call. `reset` returns held whitespace with the parser's unconsumed text. Held whitespace is released in front of the visible text that decides it, so the text channel stays byte-identical. Every other family keeps its output unchanged. RESOLVED for issue #406. See [Tool-only turn whitespace](#tool-only-turn-whitespace-issue-406).
- **P7 Nested channel markers (a marker of one channel inside another)** -> marker recognition is CHANNEL-SCOPED, and both directions follow the same best-effort-recovery rule (recover real structure, never leak markup, never drop a valid call):
  - Inside a **quoted tool-argument string value**, marker-looking bytes are DATA (I7). A reasoning marker there does NOT open a reasoning channel — it is the literal arg string (`reason_markup_in_arg`). A reasoning-first pipeline that extracts `<think>`/`<|channel>` before tool parsing corrupts the arg -> red (ARG_MISMATCH / MERGE).
  - A **well-formed tool-call envelope inside a reasoning span** is STRUCTURAL: break out of reasoning, emit the call, resume reasoning after its close (`tool_in_reason`). Leaking the raw `<|tool_call>...<tool_call|>` into `reasoning_content`, or dropping the call, is the regression -> red (LEAK). The asymmetry is deliberate: quote delimiters explicitly mark a data region, whereas a reasoning span is opaque text that can still contain recoverable structure.

> P1/P2 are RESOLVED by the documented best-effort-recovery contract above (not open product questions). P4 is now resolved in code for gemma4 (optional role label) and stays a judgment call for any family that adds structural whitespace of a different shape; cases depending on it carry a `policy:` tag. P7 is RESOLVED by the same contract (no markup leak, no dropped call); the families still on the split path diverge, which is the gap it documents.

## Divergence classes (how a non-matching cell is colored)

`MATCH` (green) · `ORDER` / `MERGE` / `LOSS` (the unification gap) · `LEAK` (markup in text, `↯`) · `ARG_MISMATCH` / `WHITESPACE` (version drift) · `ERROR` (engine hard-errored where the spec expects graceful output).

The native `UnifiedParser` families are `beam`, `deepseek_v4`, `deepseek_v41`, `qwen3`, `gemma4`, `glm47`, `kimi_k2`, `kimi_k3`, `muse_glimmer`, `nemotron_v3`, `muse_spark`, `minimax_m3`, `mistral_4`, `gpt_oss`, and `hermes`. GPT-OSS retains its historical Dynamo v1 reasoning + batch observations and now records ordered native v2 events.

The dotted-name cases cover minimal, mixed-punctuation, multi-segment, and shared-prefix tool names. They apply to Beam, DeepSeek V4/V4.1, Gemma 4, GLM 4.7, GPT-OSS, Hermes, Kimi K2/K3, Mistral v11+, Muse Glimmer, Muse Spark Open, Nemotron 3 Ultra, and Qwen3. New observations are recorded only for v2. GPT-OSS captures before v2 0.7.28 use v1 reasoning followed by the v2 Harmony tool parser and are labeled split; v2 0.7.28 and later use native UnifiedParser. The split Harmony path remains covered by whole-input and every-boundary comparisons against the same GOLDEN.

## Hermes applicability

Hermes follows the [Hermes 3 tool-use template](https://huggingface.co/NousResearch/Hermes-3-Llama-3.1-8B/blob/main/tokenizer_config.json): one JSON `{name, arguments}` object per `<tool_call>` envelope. Its registry opts into shared scenario renderers with `shared_scenarios: true`; explicit family mappings, schemas, numeric probes, quoted controls, whitespace, dotted names, guided surroundings, and prefilled request states retain their independent GOLDEN. A native JSON envelope inside reasoning interrupts the thought and reasoning resumes after the call. Control-looking bytes inside JSON strings remain data. Native unwrapped arrays and objects remain visible answers.

Hermes has no separate name-bearing native invoke header, so the named bare-header guided crossing duplicates its unnamed wrapper crossing and is excluded. XML parameter typing, DSML separators, Harmony/channel routing, Kimi typed channels, ATEM, and Gemma delimiters do not belong to this grammar. Structural-tag generation is not registered for Hermes; JSON-guided parsing is supported independently of a structural-tag builder. Historical and peer captures are retained as measured evidence; only the current candidate is recaptured for this addition.

## Nemotron 3 Ultra applicability

The `nemotron_v3` corpus inherits every applicable Qwen XML scenario through the registry's `grammar: qwen3` declaration. Default-state shared inputs first close Ultra's implicit empty thought with `</think>`; explicitly initialized Reasoning and Response inputs keep their original request state and bytes. This preserves the shared scenario's channel semantics without disabling later reasoning markers. The Qwen XML header cases also apply to Nemotron. The structured-response scenario requires a registered structural-tag builder, which Nemotron does not have; it remains explicitly excluded while shared parser scenarios retain Qwen grammar inheritance. DSML, Harmony, Gemma, GLM, Kimi, and ATEM-specific scenarios remain excluded because they require other native grammars.

These two cases display under the Nemotron v3 default-reasoning section. Their published numeric IDs and capture history remain unchanged. Only Nemotron currently enables implicit reasoning under default initialization; explicitly initializing another family to Reasoning exercises a different request contract.

- `10-default_reasoning_without_opener`: with default initialization, `Check Montréal.</think>雪` emits reasoning followed by visible Unicode content. The same input under Response stays literal text.
- `12-default_reasoning_tool_without_closer`: default reasoning contains a native XML call with literal think markers in its argument, then resumes reasoning at EOF. Both think boundaries are omitted. Other registered families require explicit Reasoning initialization for this contract; their shared prefilled and nesting scenarios cover that state.

## Quick reference — descriptive taxonomy (`UNIFIED.<group>-<description>`)

New Unified IDs use `UNIFIED.<group>-<description>`, for example `UNIFIED.7-tool-argument-text-preservation-arg_unicode` or `UNIFIED.kimi-typed_argument_values`. Derive descriptions from scenario names, omit redundant family prefixes in family-specific groups, and use schema probe names directly, such as `UNIFIED.7-tool-argument-declared-types-declared_array`. Numeric, dotted, and schema-subgroup forms remain historical aliases; do not create new ones. `unified_taxonomy.py` owns canonical IDs, descriptions, ordering, display placement, and aliases. Internal scenario names stay unchanged. The scenario slug is shown in parentheses. **Groups 1–9 mirror the tool-calling STREAM taxonomy** (`TOOLCALLING.streamv1.N`) as reasoning-free unified cases — this surface subsumes STREAM. **Group 10** is the reasoning axis (`REASONING.*`). **Group 11 is UNIQUE to unified**: reasoning↔tool ORDER that neither STREAM (no reasoning) nor REASONING (no ordered tool events) can express. **Group 12** is adversarial nesting — a marker of one channel inside another (P7). **Groups 30–39 are Guided Decoding**, divided by payload validity, surrounding markup, reasoning boundaries, and visible-answer markers. **Groups 40+ are prefilled request states.** Model-specific groups (`gemma`, `glm5`, `kimi`, `mistral`, `muse`) sort after every numeric group. The live per-family and total case counts come from `test_unified_case_counts_match_the_generator`; do not maintain a numeric total in prose. Existing published numeric IDs remain preserved and resolve through the taxonomy aliases.

### Group 1 — TC Single call
- **`1-tool_only`** (`tool_only`) One tool call, no reasoning, no surrounding text. The tool suite's baseline. Every family uses `get_weather({"city":"Paris"})`. The separate leading-JSON-whitespace case uses the same call in GPT-OSS and Kimi K2/K3, retaining its space/newline/tab prefix and native framing. Historical captures retain the original request and output, including older `lookup({"q":"x"})` examples; they are not relabeled as executions of the current weather request. The active case is `1-tool_only_leading_json_body_whitespace`; the retired `gpt_oss-tool_body_whitespace` identity keeps its original lookup request and GOLDEN so historical request references remain valid. This replaces an existing JSON-body scenario in the same three families; XML parameter-body and Gemma typed-argument whitespace have separate grammar contracts.

### Group 2 — TC Multiple calls (TOOLCALLING.streamv1.2)
- **`2-tool_call_ids_unique`** (`tool_call_ids_unique`) Two identical `get_weather(city="Paris")` invocations must remain separate calls with distinct parser-supplied IDs within one response. K2 uses two distinct native `functions.get_weather:0` / `:1` protocol IDs and preserves them; K3 deliberately repeats XTML `index="1"` and must generate different API IDs. Repeated deltas for the same call keep the same ID. Every registered native Unified family and grammar-sharing alias is covered. Families returning no ID delegate API identity generation to the serving layer; the table does not claim to validate that layer. The companion lifecycle test checks all families with character-split delivery over three turns, including reset/reuse and a fresh parser: K3 IDs must be unique across turns, K2 native IDs are preserved even when reused in another response, and other families must continue delegating IDs. A negative control supplies duplicate K2 IDs within one response and must fail the shared validator. No historical or peer recapture is implied.
The ID policies follow each model protocol, rather than imposing a new wire format on every family. K2's `functions.NAME:INDEX` is model-facing data; its identity remains associated with the assistant message. K3's XTML index is positional metadata, so its API ID is opaque. vLLM makes the same K2/K3 distinction in [issue #50295](https://github.com/vllm-project/vllm/issues/50295) and [the merged K3 fix #50420](https://github.com/vllm-project/vllm/pull/50420). When a parser returns `None`, the consumer must generate an ID once for each call index and emit it only on the first API chunk; that consumer contract belongs in Dynamo integration tests.

- **`2-two_calls`** (`two_calls`) Two distinct calls back-to-back, order preserved. This is also covered in: TOOLCALLING.streamv1.2.a.
- **`2-two_calls_same_name`** (`two_calls_same_name`) Two calls to the SAME function, different args — must not dedup or merge. This is also covered in: TOOLCALLING.streamv1.2.d.
- **`2-adjacent_calls_partial_header`** (`adjacent_calls_partial_header`) Native/default state: one delta closes a complete call and carries the next opener plus a partial call header. Preserve both calls and emit no visible markup. Uses authored Unified inputs, tools, chunk schedules, and independent expected calls; DeepSeek V4.1 splits inside its second invoke in one calls container.
- **`2-three_calls_partial_header`** (`three_calls_partial_header`) Native/default state: three adjacent calls delivered with a boundary inside the second call header. Preserve f, g, h in order and emit no visible markup. Both scenarios apply to eleven families: deepseek_v4, deepseek_v41, gemma4, glm47, kimi_k2, kimi_k3, mistral_4, muse_glimmer, muse_spark, nemotron_v3, and qwen3 (22 family-specific cases). Beam and GPT-OSS are excluded from this Unified case set. The authored schedules exercise delivery boundaries that marker-aligned and fixed-width chunking can miss.

### Group 3 — TC No call (TOOLCALLING.streamv1.3)
- **`3-blank_output_without_call`** (`blank_output_without_call`) Whitespace-only output with no call is visible content and is kept unchanged by every family (P8).
- **`3-text_only`** (`text_only`) Plain content, zero tool structure. No spurious call. This is also covered in: TOOLCALLING.streamv1.3. No e2e case has this shape: Qwen3.6 always emits a reasoning span, so the plain-content case is corpus-only.

### Group 4 — TC Malformed envelope
- **`4-tool_block_never_closed_then_text`** (`tool_block_never_closed_then_text`) The calls opener arrives without its closing marker and prose follows. The prose remains inside the unterminated tool envelope and is discarded at EOF; it is not visible answer text. This is applicable to DSv4.1 because its native grammar has an explicit calls envelope. The retained GPT-OSS request instead starts with an orphan handoff and never opens a tool message: its following prose remains visible. Genuine Harmony truncation is covered by shared `tool_no_close` and its next-message recovery variant.
- **`4-tool_markup_only_emits_nothing`** (`tool_markup_only_emits_nothing`) A calls envelope contains no invocation. Both markers are control syntax and the parser emits no event.

### Group 5 — TC Truncation / recovery (TOOLCALLING.streamv1.5)
- **`5-truncated_tool_eof`** (`truncated_tool_eof`) EOF mid-call. Golden drops the partial, keeps preceding output (P2); vLLM Rust hard-errors (`ParsingFailed`). Class ERROR.
- **`5-blank_framing_incomplete_call`** (`blank_framing_incomplete_call`) A framing newline, then a call cut off mid-parameter: no call completes, so the newline stays text for Nemotron V3 and Qwen3 (P8, P2).
- **`5-tool_no_close`** (`tool_no_close`) Complete call body but the close marker never arrives. Most grammars recover the complete call at finish; DeepSeek V4 and V4.1 require the invoke closer and drop this malformed call. This is also covered in: TOOLCALLING.streamv1.5.a.
- **`5-orphan_close_after_prose`** (`orphan_close_after_prose`) Orphan close marker after prose. Golden strips it; engines may leak. Class LEAK.
- **`5-malformed_json_then_two_valid_calls`** (`malformed_json_then_two_valid_calls`) Across all native Unified families, an unfinished native JSON/string body with explicit call closers precedes `echo(value="é")` and `echo(value="Café")`. The authored `input_chunks: [input]` delivers one `push()`, then `finish()`. Marker-aligned delivery masks the Kimi K3 defect: published `dynamo-parsers-v2` versions `0.7.13` and `0.7.14` lose the first echo; `0.7.15`, now published, preserves both. During capture qualification, the candidate moved to `0.7.15` after `0.7.14` was published; its earlier unpublished checkpoint was retired rather than relabeled as released output. K2 retains the malformed call as `{}`; Qwen3, GLM47, DeepSeek V4, and Muse retain its malformed object value as a literal string. Those five families preserve both echoes in both versions. Gemma4 returns no calls and DeepSeek V4.1 raises a JSON parse error in both versions; their goldens still require both echoes, and both remain red. Parser fixes for these two failures are outside this change.
- **`5-missing_inner_closers_then_valid_call`** (`missing_inner_closers_then_valid_call`) Native mode with default initialization: an outer-close-looking marker and a call-shaped suffix occur before the original argument has its own closer. Beam, DeepSeek V4/V4.1, Kimi K3, MiniMax M3, Muse Glimmer/Spark, Qwen3, and Nemotron V3 preserve the embedded markers and header as one literal location argument through Paris. GLM47 recovers the first final argument at its outer call closer and emits two calls, unfinished then Paris. Gemma4 quote delimiters leave the first candidate syntactically invalid, so it recovers only Paris. Each family result popup explains its native boundary. The published scenario name and IDs remain stable; this is not a universal discard-and-restart test. Kimi K2 and GPT-OSS use bare JSON arguments, already covered by `5-malformed_json_then_two_valid_calls`.

- **`5-dsml_unterminated_parameter_then_empty_call`** DeepSeek V4 and V4.1: drop an invoke with an unterminated parameter and preserve the following complete `f({})` call.
- **`5-dsml_unterminated_parameter_then_call`** DeepSeek V4 and V4.1: drop an invoke with an unterminated parameter and preserve the following complete `get_weather({"city":"Paris"})` call. These inputs require DSML invoke/parameter framing; Muse uses its separate ATEM recovery cases, and other families do not accept DSML envelopes.

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
- **`7-tool-argument-declared-types-declared_integer_boolean_number_order`** (`schema_declared_integer_boolean_number_order`) The value `1` under the ordered type union `["integer", "boolean", "number"]` stays the integer `1` in every native family. Beam must try its declared integer alternative before boolean; inferring that number also admits integers must not remove that preference.
- **`7-tool-argument-declared-types-enum_contradicting_declared_type`** (`enum_contradicting_declared_type`) Shared-family regression for ai-dynamo/frontend-crates#526 and #527: an `enum` written beside `type` instead of under `items` (as in BFCL live functions) admits no value of the declared type, so the declared type decides. The array text `["temperature","humidity"]` is a list and `5` is an integer for `{"type": "integer", "enum": ["auto"]}`. Schema-driven grammars (Qwen3, Nemotron v3, GLM, MiniMax M3, Beam) must ignore the contradicting literals; explicitly typed grammars carry the types on the wire. The strict schema oracle exempts this case because the schema admits no value literally.
- **`7-arg_numeric_nested_exact`** (`arg_numeric_nested_exact`) Nested objects and arrays preserve a large fractional number, an underflow-sized exponent, and positive and negative integers beyond the exact binary64 range.
- **`7-tool-argument-unions-and-intersections-not_keyword_type_complement`** (`not_keyword_type_complement`) Shared-family regression for [ai-dynamo/dynamo#17662](https://github.com/ai-dynamo/dynamo/issues/17662): a `not` keyword constrains the argument to the complement of the negated type. `not string` makes bare `-2500` an integer; `not integer` keeps bare `42` the string `"42"`. Schema-driven grammars (Qwen3, Nemotron v3, GLM, MiniMax) must infer the type; explicitly typed grammars carry it on the wire.
- **`7-tool-argument-unions-and-intersections-conditional_schema_number_types`** (`conditional_schema_number_types`) Shared-family regression for ai-dynamo/dynamo#17662: `if`/`then`/`else` resolve to the union of the branch types. A number in both branches is a number (`21.5`); an integer `if` with a numeric `else` keeps `37` an integer.
- **`7-tool-argument-schema-references-reference_object_recursive_and_composed_types`** (`reference_object_recursive_and_composed_types`) Shared-family regression for ai-dynamo/dynamo#17662: a `$ref` to an object definition, a recursive `$ref` tree, a `type` array with null, and a boolean declared only through `allOf` keep their JSON types instead of degenerating to strings.
- **`7-arg_numeric_overflow_siblings_{finite,positive,negative}`** A finite `1.5` control and signed overflow tokens `1e400` / `-1E+400` accompany `city: "Paris"` in every applicable native grammar. Raw deltas preserve the numeric token; the default typed batch projection may stringify overflow but must retain the city and value fields. Capture reconstruction restores exact numeric text only when the assembled field structure and scalar values survive, so an older assembler returning `{}` remains visible as a failure.

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
- **`8-blank_framing_tool_only`** (`blank_framing_tool_only`) A tool-only turn frames its call with a newline on each side; Nemotron V3 emits no text, other families keep the newlines (P8).
- **`8-blank_framing_between_calls_single_chunk`** (`blank_framing_between_calls_single_chunk`) Newlines before, between, and after two calls, delivered in one push; Nemotron V3 emits no text, other families keep the newlines (P8).
- **`8-blank_framing_with_prose`** (`blank_framing_with_prose`) Prose before the call makes the turn visible, so every whitespace byte is kept (P8).
- **`8-blank_framing_prose_after_call`** (`blank_framing_prose_after_call`) A framing newline, a call, then prose: every whitespace byte is kept; Nemotron V3 releases the held newline in front of the prose, after the call (P8).
- **`8-native_single_quote_contraction_response`** (`native_single_quote_contraction_response`) A control marker inside a contraction-bearing single-quoted sentence stays literal prose.

### Group 10 — Reasoning span (`REASONING.*`)
- **`10-reason_only`** (`reason_only`) Reasoning span, nothing else. This is also covered in: REASONING.batch.2.a.
- **`10-reason_then_content`** (`reason_then_content`) Reasoning then visible content, no call. This is also covered in: e2e case-0001-chinese_arithmetic__non-stream-budget_capped.json (+ 42 more: every `reasoning/core`, `reasoning/complex` and `reasoning/history` case, `tool_none_arithmetic__*`, and the SECOND step of both `lifecycle_*` — each with its `-budget_unlimited` pair).
- **`10-two_reason_spans`** (`two_reason_spans`) Two reasoning spans separated by content. Batch reasoning merges them → Class MERGE. This is also covered in: REASONING.batch.6.a.
- **`10-reason_unterminated`** (`reason_unterminated`) Stream ends inside reasoning; open reasoning promoted at finish.
- **`10-two_adjacent_reason_spans`** (`two_adjacent_reason_spans`) Two reasoning spans with nothing between them, then the answer. Adjacent Unified events concatenate their body bytes without inserting a separator (I8). This request explicitly starts its second body with a newline, so that authored newline remains. The counterpart — two spans separated by a call must NOT join — is pinned by `11-reason_after_tool` / `11-reason_interleaved`: a parser that always joins invents a newline the model never emitted.

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
- **`11-blank_framing_after_reasoning`** (`blank_framing_after_reasoning`) Reasoning, then a newline-framed call and nothing visible: reasoning and the call survive; for Nemotron V3 the framing is not text, other families keep it (P8).
- **`11-text_reason_tool_text_reason_tool`** (`text_reason_tool_text_reason_tool`) Deep well-formed interleave — text → reason → tool → text → reason → tool; user text, reasoning, and calls all mix in one stream, every segment in order. Class MERGE (batch hoists both thoughts).

### Group 12 — Adversarial nesting (a marker of one channel inside another; P7)
- **`12-reason_markup_in_arg`** (`reason_markup_in_arg`) "Tool call contains reasoning" — a reasoning-channel marker sits inside a quoted tool-arg VALUE. NOT a leak: an arg value is data bound for the function, not a rendered channel, so by I7 the parser preserves it byte-exact (the gemma4 native UnifiedParser confirms the golden exactly). A reasoning-first extractor lifts it out and corrupts the arg. Class ARG_MISMATCH / MERGE.
- **`12-tool_in_reason`** (`tool_in_reason`) "Reasoning contains tool call" — a well-formed tool-call envelope nested inside a reasoning span. OPPOSITE of 12-1: a reasoning span is opaque text (not a quoted data region), so a real tool-call marker inside it IS structural. Golden breaks out (reason → call → reason). Engines leak the tool markup into `reasoning_content` and drop the call. Class LEAK.
- **`12-reason_markup_in_arg_with_text`** (`reason_markup_in_arg_with_text`) 12-1 WITH visible narration before and after — all three channels at once (text / tool-call-with-markup-arg / text). Golden keeps text as text, the call clean, the markup byte-exact in the arg. Class ARG_MISMATCH / MERGE.
- **`12-tool_in_reason_with_text`** (`tool_in_reason_with_text`) 12-2 WITH visible narration before and after — text → reason → call → reason → text. Golden breaks out and keeps the surrounding text; engines leak the nested markup. Class LEAK.

### DeepSeek V4.1 applicability
- DeepSeek V4.1 uses the ordered Unified contract for native DSML calls, reasoning interleaving, guided JSON, prefilled states, and dotted tool names. The current corpus emits 302 of the 530 taxonomy cases for this family.
- Generated cases include `30-guided_json_gt_in_argument_bare_opener`, the single-family `deepseek-json_invocation_body` case, applicable Guided Decoding cases in groups 31 through 35 (exclusions listed below), the marker-discriminating Response row `50-prefilled_response_reasoning_markers_literal`, and the 40-series cases `40-prefilled_reasoning_with_guided_json`, `40-native_quoted_control_in_reasoning`, and `40-native_single_quote_contraction_reasoning`, plus `41-prefilled_reasoning_redundant_opener` through `41-prefilled_reasoning_truncated`.
- The 228 omitted cases require another family's framing or schema, or repeat a transition already covered for DeepSeek V4.1. Family applicability comes from `scenario_families()` and the serialized request. Family-specific grammars and schemas require explicit exclusions; Response-prefilled native object, array, and string ownership now cover each applicable native family. Prefilled rows that duplicate another state transition remain excluded for DeepSeek V4.1.

- The two `minimax_m3-guided_native_after_prefilled_reasoning_*` rows cover M3 recovery and are scoped to its conversion.
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

### Possessives after quoted names

The `quoted_name_*` cases cover an ASCII possessive apostrophe immediately after a closed backtick span or double-quoted name. The authored schedule separates the closing span, apostrophe, and `s`, so the parser must retain bounded context across chunks. Each case has its own input, request, GOLDEN, and column.

- **`40-quoted_name_backtick_possessive_reasoning`** and **`40-quoted_name_double_quote_possessive_reasoning`** preserve the genuine reasoning closer after `` `load_config`'s `` or `"load_config"'s`, followed by code containing additional quotes. The final answer remains visible text.
- **`40-quoted_name_backtick_possessive_then_call`** and **`40-quoted_name_double_quote_possessive_then_call`** preserve the same possessive reasoning, extract the following real native call, and retain the trailing answer.
- **`34-quoted_name_backtick_possessive_guided`** and **`34-quoted_name_double_quote_possessive_guided`** end prefilled reasoning at its real closer and extract a required guided JSON call whose string argument contains backticks.
- **`40-quoted_name_backtick_nonpossessive_some`**, **`40-quoted_name_backtick_nonpossessive_s_name`**, **`40-quoted_name_double_quote_nonpossessive_some`**, and **`40-quoted_name_double_quote_nonpossessive_s_name`** retain a genuine single quotation after the quoted name: `some` and `s_name` are not the suffix `s` followed by a word boundary, so their embedded reasoning closer stays literal.

All ten scenarios apply to DeepSeek V4/V4.1, Gemma4, GLM47, Kimi K2/K3, MiniMax M3, Muse Glimmer/Spark, Qwen3, and the grammar-sharing Nemotron v3. Each family uses its own reasoning delimiters, text framing, and native call encoding. Beam and GPT-OSS/Harmony are excluded because their control-token semantics do not use this prose quotation heuristic. The v1 split scanner does not implement this apostrophe state; no legacy fixture is added or relabeled. Peer observations remain unmeasured until separately captured.

### Group 41 — Prefilled reasoning, malformed
- **`41-prefilled_reasoning_redundant_opener`** (`prefilled_reasoning_redundant_opener`) The backend re-emits the `<think>` the prompt already wrote. Exactly one echo is consumed, not leaked; a second would be stray markup and stripped (I3). The only case where a prefilled stream legitimately carries an opener.
- **`41-prefilled_reasoning_truncated`** (`prefilled_reasoning_truncated`) Budget runs out mid-call. Keep the completed reasoning, drop the partial call (P2).

### Group 50 — Prefilled response
- **`50-prefilled_response_reasoning_markers_literal`** (`prefilled_response_reasoning_markers_literal`) `<think>literal</think>` must reach the user as TEXT, markers and all, because this stream has no reasoning channel. It is the direct visible-marker regression. Beam's reasoning markers are special tokens, so its authored golden differs: the input carries the full framed opener (`<|start|>assistant<|channel|>analysis<|message|>literal<|end|>`), and under Response the stream has no reasoning channel, so that framing is consumed as control (never leaked) while its body `literal` is visible TEXT that joins the prose after it (`literal then a call`); the following call still dispatches.

The marker-free prefilled-Response variants were removed because they emitted the same observable result as their default-state peers. Group 50 retains reasoning-marker stimuli that distinguish Response from default initialization; the ordinary native, guided, multi-call, and malformed payload contracts remain covered by groups 8, 30, and 31.

### Qwen3-specific

Qwen's chat template wraps every real call in `<tool_call>…</tool_call>`, so a BARE `<function=…>` header inside an open thought is the model narrating a call, not making one. These two cases pin that reading for `qwen3` and the grammar-sharing `nemotron_v3` only. The other wrapper families (DSML, Kimi, Gemma, MiniMax) deliberately keep the structural reading for a bare invoke inside a thought, and GLM's block is its invoke, so the behavior is not expressible as a shared scenario; `12-tool_in_reason` remains the shared case for a WRAPPED envelope inside a thought. Only the invoke markers leave the thought; the narrated body (`label_count=48`, the model describing its arguments) stays reasoning text.

- **`qwen3-narrated_bare_invoke_in_unterminated_thought`** (`narrated_bare_invoke_in_unterminated_thought`) A thought that never closes (`finish_reason: length`) narrates the same bare invoke three times, the shape of a model looping until `max_tokens` ([ai-dynamo/dynamo#17663](https://github.com/ai-dynamo/dynamo/issues/17663)). The `<function=…>` and `</function>` markers are stripped (I3) and nothing dispatches; the regression dispatched 19 duplicate `tool_calls` from one such thought.
- **`qwen3-narrated_bare_invoke_then_wrapped_call`** (`narrated_bare_invoke_then_wrapped_call`) A bare invoke narrated inside a closed thought, then the real wrapped call after `</think>`. The narration's markers are stripped and exactly one call dispatches, from the wrapped envelope.

### GPT-OSS-specific

These cases exercise Harmony message framing, which the other eight grammars do not use. Shared post-wrapper whitespace coverage stays in group 35.

- **`1-tool_only_leading_json_body_whitespace`** preserves the shared weather argument object when whitespace follows the native JSON body opener. The retired `gpt_oss-tool_body_whitespace` request retains its original lookup call for historical comparisons.
- **`gpt_oss-constrained_final_header`** retains visible JSON under a final header whose only metadata is its constrained content type.
- **`gpt_oss-implicit_recipient_first_header`** holds the generated recipient-first header until its message body opens, without prompt token IDs or leaked recipient text.
- **`5-tool_no_close_malformed_argument_then_visible_response`** recovers the later final answer after an unterminated native argument string; the ambiguous suffix stays buffered until EOF establishes that the original JSON cannot complete.
- **`7-arg_marker_in_string_quoted_closer_and_response_header`** is the valid counterfactual: a complete header inside a valid JSON string remains argument data.
- **`gpt_oss-complete_alternate_message_ends`** recovers complete JSON calls closed by Harmony message-end or turn-end. The legacy v1 tool adapter accepts `call` and `return` action endings; a message-only `end` still does not complete a call, and malformed provisional calls never receive completion.
- **`gpt_oss-single_return_ending_tool_call`** checks the exact channel-first commentary request with constrained JSON arguments and a single turn-end marker; the expected result is one `get_current_weather` call.
- **`gpt_oss-reasoning_then_return_ending_tool_call`** pairs with `11-reason_then_tool`: the request differs only in the tool ending (`return` versus `call`). Both preserve the visible reasoning event and the weather call; this catches a reasoning handoff that retains the thought but drops the tool. Harmony channel and action markers make the new case GPT-OSS-specific.
- **`gpt_oss-quoted_tool_header_in_final_json`** keeps a directed header and action stop inside a later Harmony final-channel JSON string as visible text after recipientless commentary. Reasoning removes both envelopes; the tool parser must retain ownership across their visible-message boundary rather than turn the quoted header into a call. This uses Harmony channel framing; other families retain their existing native literal-marker and response coverage.

### DeepSeek V4-specific

- **`35-guided_response_rejected_header_quote_ownership`** (`guided_response_rejected_header_quote_ownership`) DeepSeek V4 rejects an incomplete invoke header in Response; its attribute quote cannot protect the following parameter markup as prose.

This case uses `UNIFIED.35-guided_response_rejected_header_quote_ownership` and retains its published `UNIFIED.35-5` ID as a historical alias. DeepSeek V4 independently recognizes parameter markers after rejecting an incomplete invoke header. DeepSeek V4.1 and Muse do not register parameter elements as independent guided control markers; GLM and Qwen do not use quoted invoke-name attributes, and Gemma and Kimi use different call envelopes. The exact rejected-header quote and subsequent parameter-stripping contract is therefore DeepSeek V4-specific. The report groups it under DeepSeek V4-specific tests.

### DSML framing (DeepSeek V4 and V4.1)

The DeepSeek encoders render an assistant tool turn as `content + "\n\n" + calls`, also when the content is empty. Exactly that one separator is framing: a parser that returns it as content makes the client re-render `\n\n\n\n` before the calls block, so the history stops matching the generated tokens. Any other whitespace is content. Inputs use the encoder's own call layout, newlines included, and the goldens follow from that rule rather than from peer parity.

- **`dsml-separator_two_lf`**, **`dsml-separator_four_lf`**, and **`dsml-no_separator`**: one separator is consumed, extra newlines stay content, and an adjacent block removes nothing.
- **`dsml-empty_content_separator`** and **`dsml-reasoning_then_separator`**: a tool-only turn, with and without reasoning, emits no content. The reasoning variant is the common production shape `</think>\n\n<｜DSML｜tool_calls>`.
- **`dsml-bare_invoke_two_lf`** and **`dsml-bare_invoke_four_lf`**: a recovered bare invocation follows the same rule as a wrapped block.
- **`dsml-whitespace_content_separator`**: whitespace-only content keeps its own two newlines.

Historical chunk-boundary observations remain in the legacy stream corpus (`TOOLCALLING.streamv1.50.b` and `.50.c`). Author new chunk-boundary regressions in Unified with explicit chunk schedules and preserve those schedules in every capture job.

Scope: the separator is DeepSeek template framing, so other families have no equivalent behavior to test. vLLM Python keeps the separator as content (fix pending in vllm-project/vllm#58295); SGLang and vLLM's Rust frontend consume it. Before this fix, Dynamo v2 0.7.23 returned the separator as content in every case except `dsml-no_separator`.

### Tool-only turn whitespace (issue #406)

Nemotron 3 frames each Qwen3-Coder call with a newline on both sides, so a tool-only turn returned `"\n\n"` as content. Policy P8 treats that whitespace as call framing for Nemotron V3 only. Nemotron's chat template renders assistant content as `content | trim` and treats whitespace-only content as empty, so dropping it re-renders the same prompt tokens: measured on Nemotron 3.5 Lightning, Super, and Ultra, a follow-up request with `"\n\n"`, `"\n"`, `""`, or `null` history content reports the same `prompt_tokens`. A template that renders content verbatim, as Qwen3's does, would re-render a dropped turn differently and lose prefix-cache reuse from that turn on, so every other family keeps its output. Each case's GOLDEN is that unchanged output; Nemotron V3's row carries its own GOLDEN.

- **`8-blank_framing_tool_only`**: the recorded shape, delivered on the tag-aligned default schedule so each newline arrives in its own chunk.
- **`8-blank_framing_between_calls_single_chunk`**: framing before, between, and after two calls, in one push.
- **`11-blank_framing_after_reasoning`**: the common reasoning-then-call turn.
- **`8-blank_framing_prose_after_call`**: whitespace held before a call is released, unchanged, in front of later visible text.
- **`8-blank_framing_with_prose`** and **`3-blank_output_without_call`**: guards. Visible text keeps every whitespace byte, and whitespace with no call stays content.
- **`5-blank_framing_incomplete_call`**: guard. A call that never completes does not make the turn a tool turn, so the framing newline stays text. Scoped to the Qwen3 grammar: Qwen3 is the negative control and Nemotron V3 the family whose rule must not fire.

Inputs render each whitespace run as the family's own visible-text segment: bare text for the marker-pair grammars, a `to=user` message for Muse, a `response` channel for Kimi K3, and a `final` message for GPT-OSS and Beam. The five shared `blank_framing` cases cover Beam, Gemma 4, GLM, GPT-OSS, Kimi K2, Kimi K3, MiniMax M3, Muse Glimmer, Muse Spark, Qwen3, and Nemotron 3; `5-blank_framing_incomplete_call` covers Qwen3 and Nemotron 3. Nemotron V3 inherits Qwen3's inputs with its default `</think>` prefix; its own GOLDEN comes from `INHERITED_GOLDEN` in the generator. DeepSeek V4 and V4.1 are excluded because their parsers own the call framing: the encoder's `"\n\n"` separator is consumed and any other whitespace stays content (see the DSML framing section). `3-blank_output_without_call` applies to every family. Current observations are recorded in versionless latest Dynamo snapshots. Published versioned captures, including 0.8.1, and their historical request bindings remain unchanged; this update does not back-capture older releases.

### Gemma 4 guided call-prefix boundaries

- **`gemma-guided_json_visible_call_prose_before_reasoning`** and **`gemma-guided_json_malformed_call_prefix_before_reasoning`** cover Gemma 4 guided call-prefix boundaries.

### GLM argument handling

- **`glm5-parameterless_call_shape_inside_argument`** (`glm47_parameterless_call_shape_inside_argument`) places an offered parameterless-call shape inside an open GLM argument value. The embedded close/open markers remain argument data and must not dispatch a second call.

### Kimi native progress and recovery
Native Kimi progress cases `kimi-9` through `kimi-14` use `write_file(count: integer, content: string)` with count before a 4096-character content value. Each K2 JSON, K3 typed, and K3 JSON form has 7-byte and 1024-byte delivery schedules, with separate pauses after 3072 content characters, the remaining 1024, value closure, call closure, and section closure. Per-chunk deltas retain native identity and index and stay incomplete until the native call boundary resolves; K3 may require the following section fence or EOF to disambiguate a call closer. These specific envelopes and native ID rules apply to Kimi only. Other parser families retain their existing shared progress cases. Cases `kimi-15` and `kimi-16` preserve malformed trailing syntax whitespace and valid-object trailing-garbage recovery under one-character delivery. Case `kimi-17` withholds ambiguous native markers inside an unterminated string so EOF recovery preserves append-only raw arguments; Unified assembly projects the completed malformed arguments to an empty object. These exercise K2 native recovery; K3 uses a different typed/native boundary contract. Case `kimi-18` covers long inter-field whitespace in typed K3 with seven-character delivery; the work regression doubles that whitespace under one-character delivery and counts lookahead work. Case `kimi-19` releases literal string bytes after a complete invalid next-argument header, before the real value closer.


### Kimi K3 XTML

- **`kimi-typed_argument_values`** through **`kimi-guided_native_wrapper`** cover Kimi K3 XTML typed arguments, raw JSON blocks, spacing variants, message termination, reasoning closure, recovery, and guided wrappers.

### Mistral-specific

Mistral v11+ calls are `[TOOL_CALLS]NAME[ARGS]{json}`, repeated per parallel call. No marker ends a call or the block, so only another call can follow a call: text or reasoning after a call's JSON is that call's leftover argument text and is dropped, as stock vLLM drops it. Shared scenarios that place text or reasoning after a call (`8-trailing_text_after_tool`, `8-text_sandwich`, `8-text_between_calls`, `8-narrated_calls`, `11-reason_after_tool`, `11-reason_interleaved`, `11-reason_tool_text_reason_tool`, `11-reason_tool_reason_tool_reason`, `11-reason_between_calls`, `11-text_reason_tool_text_reason_tool`, `12-reason_markup_in_arg_with_text`) cannot be expressed in this grammar, and neither can a thought resuming after a call (`12-tool_in_reason`, `12-tool_in_reason_with_text`) or a complete body without its closer (`5-tool_no_close`, which would duplicate `1-tool_only`). Mistral markers are special tokens, so the quoted-marker cases in groups 8 and 40 do not apply either, except `8-native_unmatched_quote_then_call`, whose contract is that an odd quote cannot hide a real call. Guided tool output assumes a call terminator, so the parser refuses that request mode and groups 30-35 do not apply. `exclusion_note` in `gen_unified_golden.py` records each reason.

- **`mistral-text_after_call_dropped`** (`mistral_text_after_call_dropped`): text after a call's JSON is dropped (vllm-project/vllm#48975).
- **`mistral-reasoning_after_call_dropped`** (`mistral_reasoning_after_call_dropped`): `[THINK]...[/THINK]` and the answer after a call are leftover argument text too.
- **`mistral-call_inside_reasoning_ends_thought`** (`mistral_call_inside_reasoning_ends_thought`): `[TOOL_CALLS]` inside a thought ends it for good; the thought does not resume.
- **`mistral-brace_fallback_calls`** (`mistral_brace_fallback_calls`): parallel calls in the `[TOOL_CALLS]NAME{json}` form, whose `{` starts the arguments.
- **`mistral-call_id_stripped`** (`mistral_call_id_stripped`): the v11 `[CALL_ID]ID` between the name and `[ARGS]` is not part of the name.
- **`mistral-name_before_marker`** (`mistral_name_before_marker`): `bash[TOOL_CALLS]{...}` keeps `bash` as content and emits a call with an empty name.
- **`mistral-name_without_arguments`** (`mistral_name_without_arguments`): a name with no body is a call with `{}` (P3).
- **`mistral-reasoning_trailing_whitespace_stripped`** (`mistral_reasoning_trailing_whitespace_stripped`): a thought loses its trailing whitespace when it ends.
- **`mistral-duplicate_think_opener_stripped`** (`mistral_duplicate_think_opener_stripped`): a second `[THINK]` inside an open thought is stripped.
- **`mistral-quoted_marker_is_structural`** (`mistral_quoted_marker_is_structural`): an odd quote in a thought cannot make `[/THINK]` literal or hold it until a later quote.
- **`mistral-malformed_body_ends_at_next_call`** (`mistral_malformed_body_ends_at_next_call`): a body that is not yet JSON ends at the next `[TOOL_CALLS]` outside a string; the later call survives.
- **`mistral-text_between_calls_dropped`** (`mistral_text_between_calls_dropped`): text between two calls, braces and a stray `[ARGS]` included, is the first call's leftover; it cannot start a call.
- **`mistral-reasoning_between_calls_dropped`** (`mistral_reasoning_between_calls_dropped`): a thought between two calls is dropped, and the second call survives.
- **`mistral-quoted_tool_marker_in_answer`** (`mistral_quoted_tool_marker_in_answer`): a quoted `[TOOL_CALLS]` opens a call whose prose name is rejected, so the rest of the answer is dropped. Stock vLLM's text path emits a call named by the prose.
- **`mistral-quoted_tool_marker_then_call`** (`mistral_quoted_tool_marker_then_call`): the rejected name does not hide a real call after it.
- **`mistral-call_id_variants`** (`mistral_call_id_variants`): `[CALL_ID]` before a `{` body, before the next call, and with an invalid id; the id never joins the name or the arguments.
- **`mistral-stray_separator_after_args`** (`mistral_stray_separator_after_args`): a repeated `[ARGS]`, or `[CALL_ID]` after `[ARGS]`, does not open a JSON array. This is more lenient than mistral-common's tool-call grammar, which keeps the separator as argument text.
- **`mistral-json_escapes_and_lookalikes`** (`mistral_json_escapes_and_lookalikes`): escaped quotes and backslashes, a surrogate-pair escape, and marker text stay argument data (`I7`).
- **`mistral-marker_lookalikes_stay_text`** (`mistral_marker_lookalikes_stay_text`): `[TOOL_CALL]`, `[ARG]`, `[THINKING]`, and `[/THINK ]` are visible text.
- **`mistral-whitespace_between_markers`** (`mistral_whitespace_between_markers`): a thought keeps leading and loses trailing whitespace, the answer keeps its newline, and the name and body tolerate surrounding whitespace.
- **`mistral-markers_split_across_chunks`** (`mistral_markers_split_across_chunks`): every marker and a name split across deltas (`I2`).
- **`mistral-many_mixed_form_calls`** (`mistral_many_mixed_form_calls`): six calls mixing `[ARGS]`, the `{` form, `[CALL_ID]`, a bare name, and closing brackets inside strings.
- **`mistral-truncated_second_call_dropped`** (`mistral_truncated_second_call_dropped`): EOF inside the second call's string drops that call (P2); the first survives.
- **`mistral-truncated_inside_args_marker`** (`mistral_truncated_inside_args_marker`): EOF inside the second call's `[ARGS]` drops that call (P2).
- **`mistral-non_object_arguments`** (`mistral_non_object_arguments`): non-object JSON arguments pass through and text after the first JSON value is dropped, as stock vLLM's `_extract_args_json` keeps them. A forced tool grammar keeps arguments an object.
- **`mistral-pre_v11_array_not_a_call`** (`mistral_pre_v11_array_not_a_call`): the pre-v11 `[TOOL_CALLS] [{...}]` array emits nothing. Stock SGLang parses it as calls.
- **`mistral-empty_thoughts_emit_nothing`** (`mistral_empty_thoughts_emit_nothing`): empty and whitespace-only thoughts emit no reasoning.
- **`mistral-json_reply_after_reasoning`** (`mistral_json_reply_after_reasoning`): a JSON reply after a thought, with no `[TOOL_CALLS]`, is visible content.
- **`mistral-whitespace_only_arguments`** (`mistral_whitespace_only_arguments`): whitespace-only and `{ }` bodies are `{}`; a body on its own line keeps its arguments.
- **`mistral-closed_string_marker_before_truncation`** (`mistral_closed_string_marker_before_truncation`): a call marker in a closed string stays data when a later string is truncated; the incomplete outer call is dropped and the turn emits nothing (`I7`, P2).
- **`mistral-unclosed_string_marker_at_eof`** (`mistral_unclosed_string_marker_at_eof`): a literal call with quote-free arguments inside a truncated file-content string is not an EOF recovery boundary; the turn emits nothing (`I7`, P2).
- **`mistral-escaped_call_in_unclosed_string`** (`mistral_escaped_call_in_unclosed_string`): JSON-escaped call arguments inside the same truncated string stay data, and the incomplete outer prefix is dropped; the turn emits nothing (`I7`, P2).
- **`mistral-closed_string_marker_before_recovery`** (`mistral_closed_string_marker_before_recovery`): skip a marker in a closed string, then recover a genuine call after a later missing quote only when a syntactically invalid string close confirms malformed JSON. The malformed outer call retains the existing P3 empty-arguments behavior.

These four EOF marker-ownership cases apply only to `mistral_4`, whose calls have no terminator. The other native Unified families use invoke/argument closers (DeepSeek V4/V4.1, Qwen3/Nemotron V3, GLM, Gemma4, Kimi K2/K3 and MiniMax M3) or message/channel terminators (Beam, GPT-OSS and Muse Glimmer/Spark); their truncated-call contracts are covered by the shared group 5 cases and do not infer an EOF boundary from a quoted call marker. The v1 `mistral` parser accepts the pre-v11 JSON-array format, so these compact-call cases do not apply; the retired v1/batch/jail corpus is unchanged. At EOF, an open string alone is insufficient evidence of a missing quote. Recovery requires an invalid non-whitespace character after a candidate close; a genuine missing-quote case with only quote-free later bodies is ambiguous and is dropped under P2.

Mistral 4 also runs the shared dotted-name and partial-header cases, `1-tool_only_leading_json_body_whitespace`, and these cases from the GPT-OSS section, which other grammars run too: `gpt_oss-control_marker_inside_argument_string`, `gpt_oss-invalid_tool_body_then_empty_call`, `gpt_oss-invalid_tool_body_then_valid_call`, and `gpt_oss-quoted_message_header_literal`. `36-structured_response_after_reasoning` does not apply, because Mistral constrains a reply with its lark grammar rather than a structural tag. The GPT-OSS response-header and truncated-tool cases do not apply either, because Mistral has no response header and no call closer. The blank-framing cases `8-blank_framing_tool_only`, `8-blank_framing_between_calls_single_chunk`, `8-blank_framing_with_prose`, `8-blank_framing_prose_after_call` and `11-blank_framing_after_reasoning` do not apply, because whitespace after a Mistral call is that call's leftover argument text (`mistral-text_after_call_dropped`). The `40-quoted_name_*` reasoning cases do not apply, because Mistral markers are special tokens that quotation cannot make literal (`mistral-quoted_marker_is_structural`), and a thought loses its trailing whitespace (`mistral-reasoning_trailing_whitespace_stripped`).

### Muse-specific

- **`muse-guided_json_quoted_bare_tool_header_in_answer`** is the Muse tool-recipient header inside visible answer text. Its old `31-26` capture key remains a historical alias; this column is intentionally absent for non-Muse families because it would duplicate `35-guided_json_quoted_bare_header_in_answer`.

- **`muse-unterminated_parameter_then_empty_call`** (`muse_unterminated_parameter_then_empty_call`) places parameterless invocation-shaped text inside a parameter that never closes. At EOF, discard the ambiguous outer invocation without dispatching the embedded call. This covers the missing conformance case raised in [PR #316](https://github.com/ai-dynamo/frontend-crates/pull/316#discussion_r4171144832).
- **`muse-unterminated_parameter_then_call`** (`muse_unterminated_parameter_then_call`) places a declared invocation-shaped sequence inside the outer parameter. Preserve the outer call and its complete argument through the parameter closer; do not guess that the embedded text is a new executable call. Both cases are Muse-only because they exercise the ATEM scanner's treatment of open parameters.
### Beam applicability

Beam (`<|start|>assistant[ to=NAME]<|channel|>analysis|final[ <|constrain|>xml]<|message|>… <|end|>|<|return|>|<|call|>`, NOT Harmony) joins every shared scenario with a Beam-native wire and an independently authored golden; the numeric IDs are unchanged. The grammar changes the bytes, not the meaning, except in three authored exceptions: `5-2` (a call never dispatches without its terminator), `5-3` (`<|call|>` inside `final` is text) and `50-4` (under Response the framed analysis message's body is text and its framing is consumed). Beam receives every shared null schema unchanged (`7-5.*`): its typing intersects a sibling `type` with anyOf/oneOf branches, a deliberate correction of the frozen reference's union, so no per-family "effective schema" is substituted. Quote, contraction, header and unmatched-quote scenarios (`native_quoted_*`, `native_single_quote*`, `native_unmatched_quote_then_call`) use `<|start|>assistant to=get_weather<|channel|>` as the quoted native control. The model builder's reference is the grammar's source of truth, not a peer engine, so the vLLM cell is `UNSUPPORTED` for every Beam case; no peer verdict is claimed.

- **`beam-empty_args_no_close`** (`empty_args_no_close`) A header for a strict tool with no properties, then EOS (no parameter tags, no `<|call|>`). Beam's serving grammar makes the parameter body optional for such a tool and lets EOS stand for the closer, so the parser emits `noop({})`. Beam-specific: the other grammars need an argument body or an inner closing marker.
- **`beam-optional_args_no_close`** (`optional_args_no_close`) The same bare header for a tool whose only property is optional. Nothing is required, so it is the call `noop({})`; a required argument keeps a bare header a truncation.

## Authoring a case: what to check BEFORE adding one

The registry covers every registered family, including historical split-parser families. Shared native scenarios cover tools, text preservation, schemas, truncated calls, and reasoning where a reasoning grammar exists. Malformed-JSON recovery is scoped to JSON argument grammars; XML and Pythonic grammars do not use that input shape. Existing grammar-specific scenarios retain their authored scope. Runtime support is measured separately by exact-version capture adapters. Freeze each run in `conformance/dynamo-capture-inventory.yaml`: every stable v2 release, retained source-only anchors, and the latest two stable v1 releases in batch and jail-buffer modes.

Numeric groups describe shared behaviors: inspect every registered family and author its native input and expected output wherever applicable. Named family sections hold grammar-specific cases; explain genuine exclusions. A bug discovered in one family does not justify restricting shared coverage to it. Preserve previously published IDs.

Every rule here exists because a case was added that could not fail for the reason it claimed. Fake coverage is worse than no coverage — it renders green.

1. **Is it distinguishable from an existing case?** Compare `init` AND input against the corpus. If some existing case has the same configuration and the same input shape, the new case tests nothing. Three groups were deleted for exactly this: a whole axis whose 51/52/53 cases had the same config and inputs as `1-tool_only`, differing only in a label the parser cannot read.
2. **Can it fail for the stated reason?** Write down what would have to break for the case to go red, then confirm the parser can even SEE that input. `finish_reason` cannot: `finish()` takes no argument, in Dynamo and in vLLM alike, so a case that varies only the finish reason varies nothing. If the axis is invisible to the parser, express it as an input shape instead — `length` becomes a TRUNCATED input, which is observable.
3. **Does the field already exist under another name?** A per-case `input_mode` was added that was a 1:1 alias of `init.starting_state` across every row, and could not diverge, because "where the stream starts" IS what the starting state encodes. Grep the case dict before adding a key.
4. **Keep the oracle independent.** Derive GOLDEN from the grammar contract before capturing parser output. Run the harness to measure disagreements. A disagreement requires a parser fix or a separately demonstrated fixture defect; recorded output must never author its own expectation.
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

## Shared schema conformance probes: 7-tool-argument-interpretation

Expected values are authored in `conformance/utils/src/glm47_schema_cases.json`, independently of parser captures. `schema_cases.py` owns behavior classification; `unified_taxonomy.py` uses it for report sections and ordering. Each authored schema probe has its own displayed column. Display numbers are ranks; historical IDs remain aliases.

| Behavior section | Separate case columns |
|---|---:|
| `7-tool-argument-text-preservation` | 7 |
| `7-tool-argument-numeric-precision` | 23 |
| `7-tool-argument-declared-types` | 14 |
| `7-tool-argument-nullability` | 19 |
| `7-tool-argument-constants` | 41 |
| `7-tool-argument-unions-and-intersections` | 55 |
| `7-tool-argument-schema-references` | 15 |

String constants expose Direct, Explicit string type, allOf, and Reference as separate cases with shared name prefixes. anyOf and oneOf expose each alternative type, String / Typed value choice, and Forward / Reversed branch order separately. Null and numeric variants likewise have one case per column. A case popup shows only that case across applicable families; parser failures and missing captures remain attached to their exact request.

Detailed matrix labels show a small `{}` when the case has an explicit schema. Overview hides it. Hover or focus the existing label to open the popup (tap on touch devices). Its `{ } Schema` disclosure expands inline, using more width when space allows and scrolling within the available viewport height. Identical schema JSON appears once with every applicable family for that case. Shared fallback schemas do not get an indicator.

Historical schema-subgroup aliases below retain their original meaning. Active IDs use the behavior prefix and probe name, for example `UNIFIED.7-tool-argument-declared-types-declared_array` and `UNIFIED.7-tool-argument-constants-const_array_allof`; subgroup headings and ordering remain separate metadata.

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

GPT-OSS v1 does not implement the reasoning parser's optional prefill override. Its batch and jail capture adapters report a requested `Reasoning` start as an unsupported request mode instead of calling the inherited no-op and claiming the mode was applied. Earlier observations remain retained; native v2 prefill coverage is unchanged.

Four recovery cases extend this section: `gpt_oss-orphan_terminator_preserves_messages` preserves completed messages across an extra ending; `gpt_oss-unfinished_directed_analysis` suppresses unfinished directed arguments as reasoning; `gpt_oss-malformed_then_recipient_first_call` recovers the next recipient without interpreting a quoted partial header as a call; and `gpt_oss-unknown_channel_before_final` preserves recipientless unknown-channel text before the later final response. These inputs require Harmony channel and recipient framing. The conformance profiles `harmony` and `harmony_text` select the registered `gpt_oss` reasoning parser and `harmony` tool parser; they have no separate active Unified family store or new report columns. These profile names do not add public parser registry aliases. Other registered families retain their native malformed-call, quoted-marker, and next-message coverage; these exact Harmony header inputs are not added to their grammars. Unit and public API tests cover the additional text/token schedules, both action endings, stop/length completion, repeated finish, directed unknown channels, and completed calls before malformed input.

The retained handoff audit cases cover: `gpt_oss-final_then_unframed_recipient_first_call` retains a call after a completed final; `gpt_oss-unfinished_header_then_valid_call` rejects an unfinished header before a valid call; `gpt_oss-unfinished_directed_header_then_final` lets the later channel own its final body; and `gpt_oss-nonassistant_analysis_then_final` excludes a user-authored analysis message from assistant reasoning. The paired `gpt_oss-recipientless_json_call_then_final` and `gpt_oss-recipientless_json_return_then_final` requests distinguish a call-ended recipientless JSON payload, which has no tool recipient, from the same return-ended visible commentary. In current v2 both bodies remain visible text; the ending cannot retroactively remove emitted bytes. These cases require Harmony author, channel, recipient, and action-ending metadata; other families retain their native next-message and payload-ownership controls. The `gpt_oss-final_then_unframed_visible_tail` request preserves a completed final message before never-emitted trailing text recovered at EOF. The `gpt_oss-recipientless_json_interrupted_by_final` request gives a new framed final header ownership after complete commentary JSON without a message ending, preserving both visible bodies without leaking protocol markers. The `gpt_oss-recipientless_constrained_whitespace_call` request preserves its two visible spaces before the final response: a content type does not establish a tool recipient. The paired `gpt_oss-nonassistant_unfinished_final_header` and `gpt_oss-nonassistant_unfinished_analysis_header` requests keep user-owned header text out of assistant content and reasoning at EOF. The `gpt_oss-named_tool_unfinished_final_header` request exercises the separate native author branch that assigns a tool role to `python`, preventing a fixed role-word allowlist from misclassifying that header. The `gpt_oss-named_author_less_than_lookalike` and `gpt_oss-named_author_nbsp_lookalike` requests retain the complete native tool author when an ordinary less-than character or a non-breaking space follows the apparent assistant prefix. The `gpt_oss-malformed_then_nonassistant_call` request prevents malformed-prefix recovery from executing a completed user-authored call. The `gpt_oss-malformed_then_native_opaque_name` request preserves the entire `f/x` recipient accepted by baseline native extraction when the call follows a malformed predecessor. The `gpt_oss-native_opaque_name_handoff` request keeps an ordinary `@f` recipient intact across reasoning-to-tool projection, including its non-identifier first character. The paired `gpt_oss-recovered_tool_header_return` and `gpt_oss-recovered_final_header_return` requests retain the later tool or final body when a recovered header contains a premature return action before its message delimiter. The `gpt_oss-initial_json_prefix_then_tool` request retains initial visible JSON containing a quoted start marker before a complete call. Its `gpt_oss-initial_json_prefix_then_recovered_final` sibling adds malformed-call recovery and a premature return in the later final header; the bare visible prefix concatenates with the final body without inventing a message separator. The `gpt_oss-tab_separated_assistant_recipient` request uses an ASCII tab between the unframed assistant author and recipient, preserving native header whitespace and a Unicode argument without leaking routing metadata. The retired `gpt_oss-implicit_recipient_prelude` and `gpt_oss-assistant_implicit_recipient_analysis` positive expectations omitted both `to=` and prompt context. They no longer assert tool inference; their original request, oracle, and captures remain retained. The `gpt_oss-assistant_role_only_final` request consumes the assistant author as header metadata before a final answer rather than exposing it as bare prelude text. The `gpt_oss-analysis_text_interrupted_by_final`, `gpt_oss-commentary_text_interrupted_by_final`, and `gpt_oss-commentary_text_interrupted_by_user` requests recover a framed message after ordinary non-JSON body text without an earlier message ending. They retain reasoning and visible attribution, concatenate visible body bytes without inserting a newline, and suppress the later user body. The `gpt_oss-commentary_text_interrupted_by_invalid_author` request adds a complete `python` author frame without native routing metadata between the interrupted commentary and later valid assistant final. Recovery must retain both assistant messages while rejecting the invalid frame and hiding its body, matching the properly ended control. The retired `gpt_oss-unconfirmed_header_return_at_eof` expectation required echoing incomplete header markers as body text. Harmony does not define that malformed recovery policy; retain its history without requiring marker leakage. The `gpt_oss-unknown_channel_header_only_eof` request emits nothing when an unknown channel has no message delimiter or body, including when a known channel word is its prefix. The `gpt_oss-analysis_interrupted_by_final_then_tool` request preserves an intermediate final answer before a later call, preventing batch handoff from borrowing a later action as the earlier analysis ending. The paired `gpt_oss-commentary_text_interrupted_by_bare_final` and `gpt_oss-analysis_text_interrupted_by_bare_final` expectations are retired: a bare channel marker inside an established body does not establish a new message. Existing literal-header controls retain that body-ownership contract. The `gpt_oss-initial_json_prefix_then_bare_final` request retains literal quoted markers and the complete initial JSON before that bare final; its prelude concatenates with the final body without inventing a message separator. The `gpt_oss-literal_bare_analysis_in_visible_body`, `gpt_oss-literal_bare_unknown_channel_in_visible_body`, and `gpt_oss-literal_bare_tool_header_in_visible_body` requests retain native-looking header bytes as literal visible body text; the tool-shaped example must remain visible even when the named tool is registered. The `gpt_oss-visible_json_prelude_indentation` request preserves four leading spaces before visible JSON, retaining Markdown code-block indentation through direct extraction and composed handoff. The `gpt_oss-visible_quoted_json_prelude_whitespace` request crosses these controls: a leading space and newline precede JSON containing a quoted start marker before a bare final. The quoted marker remains data, and all prelude whitespace must survive boundary selection. The `gpt_oss-assistant_default_channel_visible_response` request covers a complete assistant message with neither a channel nor a recipient. Its message delimiter starts visible content, so the return-ended Answer must survive without requiring an explicit final channel. Existing orphan-ending and unknown-channel cases also exercise visible text after the reasoning parser removes its envelopes, so batch tool extraction must preserve that text.

The original GPT-OSS header probes test recipient/channel ordering, the template JSON format label, a constrained final header without a tool recipient, an implicit recipient-first header, and complete tool messages ending with a message/turn terminator instead of handoff. Incomplete Harmony headers, interrupted reasoning, quoted controls, malformed bodies, and historical `_pr424` requests are separate cases related by shared semantic name prefixes. `unified_taxonomy.case_fold()` retains the related-case relationships for naming, ordering, and historical aliases; the viewer displays each active case independently. Each popup retains the original case ID, request, initialization, chunk schedule, expected events, and independent observations; each failure or missing observation stays on its own case column.

The active `tool_body_whitespace`, `gpt_oss_quoted_truncated_tool_then_final`, and `gpt_oss_quoted_message_header_literal` scenarios are related to `tool_only`, `tool_no_close`, and `arg_marker_in_string`, respectively, with separate columns. The original `gpt_oss_tool_body_whitespace` identity is retired with its lookup request intact; the active whitespace variant uses the same weather call as its shared parent. JSON-body whitespace adds Kimi K2 and Kimi K3 requests; DeepSeek V4.1 already exercises leading whitespace in `deepseek_v41_json_invocation_body`, and Qwen/Muse already format native call bodies with structural newlines. GLM uses named `arg_key`/`arg_value` pairs rather than a whole-object JSON invocation body; whitespace inside a typed value belongs to argument typing. The malformed-body-to-response variant adds Muse's unterminated ATEM parameter followed by a user message and Kimi K3's unterminated raw JSON string followed by a framed response. The literal-header variant adds native Muse and Kimi K3 values containing a closer immediately followed by a complete user/response header. Families without separate response headers retain the existing delimiter-in-argument coverage instead of receiving artificial message syntax. Kimi K3 discards the malformed call and preserves the framed answer once EOF establishes that the original JSON string cannot complete. Quoted response markers in valid arguments remain payload data; ambiguous marker suffixes stay provisional until their ownership resolves.

The retained unknown-name/empty-argument probe now runs in all 11 families with an empty offered-tool list. Untyped Beam, Qwen XML, and GLM values retain their scalar string spelling; JSON and typed grammars preserve the number. Native controls inside JSON strings and malformed whole-object JSON recovery also run in DeepSeek V4, DeepSeek V4.1, Kimi K2, and Kimi K3. Both DSML grammars accept raw JSON invocation bodies as well as parameter tags. DeepSeek V4 and Kimi K2 retain an invalid named call with empty arguments; the strict JSON decoders should drop that invocation and preserve the following call. DeepSeek V4.1 currently aborts on the non-JSON body before the empty call, recorded as an error against that oracle. Gemma, GLM, Qwen XML, and ATEM parameter fences do not carry a whole-object JSON invocation body, so those probes do not transfer to their typed argument encodings. Beam uses its existing structured argument representation rather than a raw JSON tool body.

Next-message recovery adds Beam, Muse, and Kimi K3 inputs with a complete argument and an omitted outer terminator. Muse and Kimi K3 preserve the delimiter-complete invocation before the response; Beam retains its existing text-only expectation for this request. The Kimi K3 request contains a valid JSON object and a JSON close marker, so its GOLDEN includes `lookup({"q":"unfinished"})` before `recovered` even though call/tools closes are absent. The published request bytes and case ID remain unchanged; historical failed observations remain evidence of the older parser.

The named Kimi section adds `kimi-typed_response_marker_payload`, `kimi-json_missing_close_literal_response`, `kimi-json_marker_then_long_field`, and `kimi-typed_response_then_call` (published indices 21-24). These exercise typed XTML value ownership at a response header, a closed JSON string whose JSON-channel close is missing, progress after a literal response marker, and recovery before a response followed by another call. K3 alone nests these typed or raw-JSON channels inside its call/tools/response grammar. Kimi K2 and DSML use whole-object invocation bodies without a separate JSON channel; Qwen, GLM, MiniMax, Nemotron, and Muse use their own argument fences; Gemma, Beam, and Harmony use different call/message boundaries. Their shared marker-literal and next-message cases remain applicable where expressible. Scanner-work bounds and provisional completion/identity assertions remain in Rust tests because assembled GOLDEN cannot express those invariants. Elided reasoning-close recovery adds Beam and Muse; the existing Kimi K3 elided-close request supplies equivalent coverage through an explicit coverage alias. Families without a separate response header retain their ordinary reasoning-close contract. Existing interrupted-tool-in-reasoning requests already share the `tool_in_reason` contract.

The twelve active `_pr424` variants retain distinct header or payload bytes from #424's published requests. The eighteen delivery-only variants remain retained as retired records and are covered by the shared GPT-OSS delivery regression. Each active variant has its own column, uses its parent name as a prefix, and appears in a named family section when only one family can express that request. Applicability remains attached to each individual case.

The redundant `gpt_oss_truncated_tool_at_eof_is_dropped` execution is retired because shared `tool_no_close` already tests a complete Harmony JSON body without its handoff. Its original request, expected events, aliases, and capture checkpoints remain in history. Next-message recovery remains a separate case column with the shared `tool_no_close` name prefix. Empty raw bodies, malformed JSON, quoted controls, and elided reasoning closers retain their distinct stimuli; grouping does not assert identical native recovery policies across families.

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

The live M3 native-output/JSON-guidance mismatch adds
`minimax_m3-guided_native_after_prefilled_reasoning_required` and
`minimax_m3-guided_native_after_prefilled_reasoning_named`. These cases verify
M3's native-call recovery after prompt-prefilled reasoning under required and
named tool choice. Coverage is scoped to M3 for this conversion; existing
families retain their authored cases and captures. The existing
`32-guided_json_native_markup_only` input is unchanged; its M3 oracle requires
the recovered native call. Separate M3 split tests cover early arguments,
parallel indices, named-tool rejection, reset, and literal native headers
inside JSON strings.

The Kimi K3 response recovery follow-ups add published indices 25-27: `kimi-typed_response_argument_literal` preserves the original typed string when later XTML argument headers are literal data; `kimi-direct_typed_response_recovery` and `kimi-direct_json_response_recovery` recover direct calls entered from Response without exposing the response header. Structural whitespace before the recovered header is consumed with the call. These require K3 nested argument/JSON channels and its Response-prefilled direct-call routing; the other registered families retain their grammar-native marker and next-message cases.

Known limitation: an unterminated typed string without an argument close, followed by a framed response, remains ambiguous and can yield no events at EOF. The unfinished-string recovery described above applies to the raw-JSON channel only. The argument-header literal cases above retain their complete string payload. A string combining an apparent argument close, response header, and nested call header can still be split into spurious calls; this pre-existing nested-call ambiguity is outside those argument-header cases. A downstream Dynamo dependency update is separate from this unpublished parser fix.

### Current GPT-OSS v2 routing, recovery, and coverage

Harmony recipient metadata uses `to={name}` in the role or channel header ([protocol format](https://github.com/openai/harmony/blob/main/docs/format.md)). Native Unified is a syntax-only JSON API: it preserves opaque recipient names and strips the compatibility `functions.` prefix. Projecting a JSON body addressed to `python` as a generic tool call is a Dynamo policy choice, distinct from vLLM builtin filtering and OpenAI builtin API types. The downstream Dynamo adapter currently maps generic tool calls to Function wire names; adopting this parser does not add builtin execution or authorize undeclared tools. Dynamo also normalizes `to=all` like an omitted recipient, following [openai-harmony 0.0.8 renderer behavior](https://github.com/openai/harmony/blob/ec7606df9e87e3d0a1fec9f50928c1e407f0c438/src/encoding.rs#L842); format.md does not define that normalization. A JSON content type alone supplies no recipient. Adjacent visible payloads concatenate without an invented newline. Eight of the ten oracle corrections already match the original 0.10.1 output; two also require parser fixes, alongside nine additional parser-only fixes. The five retired unsupported-policy expectations remain separate from passing results. Retained v1 and peer observations remain unchanged.

Malformed-header recovery is a local best-effort policy, not an added Harmony protocol requirement. A later framed message owns its role and body; rejected or non-assistant headers cannot release their bodies as assistant text. A repeated channel marker in an unfinished header replaces unfinished routing metadata while retaining a decodable author. An undecodable author is discarded with an explicit warning, never inferred from its valid byte prefix. Embedded premature body terminators remain unsafe metadata at EOF; after a message delimiter confirms the header, recovery discards those premature terminators with a warning. An explicit recipient after a complete JSON value can recover the next unframed call when its channel arrives, while the unfinished earlier call remains provisional. Bare channel or recipient-looking strings inside an established body cannot execute narrated headers. Five unsupported or speculative positive expectations are retired with their original requests, GOLDENs, IDs, and captures intact.

Object, array, and scalar-string JSON answers, including an unframed visible prelude, retain marker-looking strings as data once the JSON completes. Ambiguous quoted suffixes use the same incremental JSON boundary and tail recovery as tool arguments; malformed visible JSON replays through its existing visible-body owner so ordinary suffix text and later complete framed calls survive. `gpt_oss-unfinished_visible_json_then_tool` exercises that distinction, `gpt_oss-broadcast_recipient_analysis` pins broadcast attribution, and `gpt_oss-native_builtin_recipient_json` pins a JSON call without the functions namespace. These are Harmony header cases; other registered families retain their own native header and quoted-payload tests. This pass measures only current Dynamo v2. `gpt_oss-visible_scalar_string_quoted_controls` preserves a top-level JSON string containing Unicode, escaped quotes, and a complete quoted tool header. Ordinary quoted prose and text after a completed JSON root cannot claim later message controls. `gpt_oss-native_builtin_python_raw` remains an active Disagreement: its framed assistant commentary body `print(1)` targets `python` and ends in handoff; GOLDEN requires a `python` call with the JSON string argument value `"print(1)"`. TODO in a later PR: support raw Python bodies through the unchanged JSON tool-call API. Other families do not use this Harmony recipient/channel/handoff framing. The stream viewer removes incomplete tool calls before coalescing adjacent text or reasoning, matching native assembly; completed calls and message boundaries remain barriers.

The GPT-OSS native constructor does not consume offered tool schemas. The 109 retired schema-only repetitions have identical input, initialization, delivery, termination, and GOLDEN to retained representatives named in `parser_families.yaml`; other families keep their schema scenarios. Eighteen retired delivery repetitions move to `gpt_oss_delivery_repetitions_share_split_and_finish_coverage`, which checks the independent GOLDEN, every valid text and token split, character text delivery, empty pushes, pre-finish completion, and reuse. `gpt_oss_long_arguments_make_progress_before_handoff` checks argument bytes before a quoted control and before handoff, completion at handoff, finish, and reuse with the long argument in each of four positions under text and token delivery. Retired records keep their original meanings and history. Duplicate controls distinguish schema-only repetition from a different delivery schedule or starting state; rows with those meaningful differences remain separate.

Quoted commentary recovery is covered by `gpt_oss-unclosed_commentary_then_quoted_final`, its empty-quoted-final delivery counterpart, the non-assistant role counterexample, and the invalid-JSON-prefix counterexample. A quote inside a later framed message cannot settle the earlier candidate before following bytes establish its boundary. Complete JSON objects and arrays, and scalar strings quoting standalone headers, retain literal controls even before ordinary prose. Only a scalar quote spanning a message terminator and new header, with at most ASCII whitespace between them, remains provisional: valid JSON takes precedence at a control boundary or EOF, while subsequent non-whitespace invalidates whole-body JSON and restores structural framing. A complete quoted literal followed by prose can be byte-identical to malformed commentary plus a real header, so this boundary policy does not infer intent from English words. Ambiguous unterminated tails can remain buffered until EOF; unambiguous text and JSON retain their existing progress. `gpt_oss-quoted_call_header_then_visible_prose` prevents quoted tool syntax followed by prose from fabricating a call. PR #596 owns the quoted-marker progress and literal-argument additions; these GPT-OSS recovery cases do not duplicate them.
