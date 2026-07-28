# Fix Kimi K3 reasoning and XTML handoff across stream boundaries

## Summary

Fix Kimi K3 response and tool-call parsing when vLLM coalesces tokens into
larger deltas, including with `--stream-interval 20`.

The parser previously depended on favorable chunk boundaries. A split or
coalesced K3 delimiter could be released as visible content, absorbed into
reasoning, or cause reasoning and the final response/tool call to appear in
the same OpenAI delta.

This change:

- gives the exact K3 reasoning closer priority over overlapping structural
  prefixes such as `<|close|>`;
- buffers incomplete delimiters until the next chunk resolves them;
- recognizes canonical and engine-spaced K3 response, tools, call, argument,
  JSON, message, and end-of-message boundaries;
- keeps the jail active through the outermost visible message terminator;
- recovers a reserved K3 protocol suffix accidentally placed in
  `reasoning_content`;
- emits retained reasoning separately from response content or tool calls;
- strips only the exact orphan `<|close|>think<|sep|>` marker as
  defense-in-depth.

The recovery is scoped to `kimi_k3`/`kimi-k3`; it is not a broad
`<|...|>` sanitizer, so non-reserved literal text remains unchanged.

## Big picture

K3 output passes through two small filters:

```text
vLLM backend chunks
        |
        v
Reasoning parser
  |-- reasoning prose ----------------------> reasoning_content
  |
  `-- response/tools --> K3 jail --> K3 XTML parser --> content/tool_calls
                              ^
                              |
             misplaced known K3 suffix from reasoning_content
```

The reasoning parser separates thought from output. The jail buffers incomplete
markers and hides K3 framing, while the XTML parser extracts response text and
tool calls. A K3-only recovery moves known framing accidentally classified as
reasoning back into the jail.

## Why it failed before

Each parser worked for favorable chunks, but the handoff between them was not
safe for every possible chunk boundary.

For example, the configured reasoning closer is:

```text
<|close|>think<|sep|>
```

It shares a prefix with the generic structural marker:

```text
<|close|>
```

If vLLM sent these chunks:

```text
chunk 1: <|close|>think
chunk 2: <|sep|>
chunk 3: 17
```

the generic `<|close|>` matched too early. The incomplete reasoning closer was
moved into normal content, producing:

```text
<|close|>think<|sep|>17
```

Other failures came from the same boundary assumption:

- a marker split after a single `<` could be emitted before the rest arrived;
- a large chunk could contain response/tools data plus several closing markers,
  but the jail stopped at an inner marker;
- response or tool framing already placed in `reasoning_content` never reached
  the jail.

This PR fixes those handoff points: it waits when a marker is incomplete,
prefers the exact reasoning closer, consumes complete K3 framing through the
outer message boundary, and moves a known K3 suffix back from reasoning when
needed.

## What changed and where

- `reasoning/base_parser.rs` and `reasoning/mod.rs`: wait for incomplete K3
  markers and give the full reasoning closer priority over `<|close|>`.
- `tool_calling/config.rs` and `xtml/kimi_k3_parser.rs`: recognize canonical
  and spaced K3 markers, consume the outer message ending, and remove only
  known K3 framing.
- `tool_calling/jail/mod.rs`: recover K3 response/tool text misplaced in
  `reasoning_content`, then emit reasoning before content or tool calls.
- `tool_calling/xtml/mod.rs`: make the new K3 helpers available to the other
  parser code inside this crate.

## Why is a text filter this involved?

Parsing a complete string is simple. Streaming adds three requirements:

1. A marker may be split after any byte, so the parser must keep an incomplete
   suffix instead of exposing it.
2. One chunk may contain reasoning, response or tools, and message termination,
   so several parser state changes can happen in one call.
3. Reasoning, visible content, and tool calls are separate OpenAI fields and
   must be emitted in the correct order without leaking K3 protocol text.

The added logic handles those boundaries; it does not make K3 content parsing
more general. It uses fixed K3 markers, bounded buffering, and a K3-only
recovery path. Much of the patch is regression coverage that tries the same
logical completion at every marker split.

## Example failure fixed

```text
Expected content: 17
Previous content: <|close|>think<|sep|>17
Fixed content:    17
```

Tool selection and arguments were also correct in the failing tool cases, but
the reasoning delimiter leaked as extra content. The fix treats this as one
parser-boundary defect rather than separate model failures.

## Performance

- K3 reasoning checks three structural prefixes instead of scanning every full
  marker.
- The normal canonical K3 path avoids normalization allocations.
- Choice cloning and content rebuilding occur only when an exact reserved K3
  reasoning-handoff boundary is recovered.
- The expanded jail marker checks are fixed-string scans and run only for K3.
- Non-K3 parsers do not execute the K3 recovery path.

## Benchmark results

| Benchmark | Result | Reference | Coverage |
| --- | --- | --- | --- |
| Custom Tool Calling | **148/148 passed** | — | Simple, named, and forced tool calls, plus reasoning requests |
| [OCRBench](https://github.com/Yuliang-Liu/MultimodalOCR) | **0.88** | 0.89 | Text recognition, scene-text VQA, document VQA, key-information extraction, and handwritten mathematical-expression recognition |
| KVV Tool Call | **In progress** | — | Results pending |

## Validation

- `cargo test -p dynamo-parsers kimi_k3`: **48 passed**
- Parser unit tests: **714 passed, 4 ignored**
- Jail integration tests: **75 passed, 1 ignored**
- Incremental jail tests: **13 passed**
- `cargo fmt --check`: **passed**
- `cargo clippy -D warnings`: **passed**
- ARM64 Dynamo-vLLM container built using the staged local crates
