# Tool-Call Streaming Parser Cases

Streaming corner cases (`TOOLCALLING.streamv2.*`), mirroring the batch taxonomy
in `TOOLCALLING_CASES.md`. Each case feeds the batch sample's `model_text` to the
engine streaming parser **1-3 tokens at a time** and records the per-chunk deltas
each engine emits. The `streamv2` prefix keeps these distinct from the
dynamo-synced `TOOLCALLING.stream.*` cases, and a streaming case carries the
**same number as its batch counterpart** — `TOOLCALLING.streamv2.1` is the
streaming form of `TOOLCALLING.batch.1`, and so on. Streaming-only cases with no
batch analog live in a separate band (e.g. partial-token chunking is
`streamv2.50`).

## Quick reference

- **`TOOLCALLING.streamv2.1`** Single tool call (basic complete envelope). Streaming form of `TOOLCALLING.batch.1`.
- **`TOOLCALLING.streamv2.1.a`** Single complete tool-call payload delivered in one content chunk — one-chunk streaming happy path.
- **`TOOLCALLING.streamv2.1.b`** Single complete tool call split across parser-significant boundaries — buffering streaming happy path.
- **`TOOLCALLING.streamv2.2.a`** Two back-to-back commentary envelopes. Streaming form of `TOOLCALLING.batch.2.a`.
- **`TOOLCALLING.streamv2.2.c`** With surrounding narration. Streaming form of `TOOLCALLING.batch.2.c`.
- **`TOOLCALLING.streamv2.2.d`** Same-name twice. Streaming form of `TOOLCALLING.batch.2.d`.
- **`TOOLCALLING.streamv2.3`** No tool call (bare text without final channel). Streaming form of `TOOLCALLING.batch.3`.
- **`TOOLCALLING.streamv2.4.a`** Channel envelope with garbage body. Streaming form of `TOOLCALLING.batch.4.a`.
- **`TOOLCALLING.streamv2.4.b`** Unterminated JSON in message body. Streaming form of `TOOLCALLING.batch.4.b`.
- **`TOOLCALLING.streamv2.4.c`** No to=functions.X recipient. Streaming form of `TOOLCALLING.batch.4.c`.
- **`TOOLCALLING.streamv2.5.a`** Missing <|call|> end marker (bare envelope). Streaming form of `TOOLCALLING.batch.5.a`.
- **`TOOLCALLING.streamv2.5.b`** Complete commentary tool call without <|start|>assistant prefix. Streaming form of `TOOLCALLING.batch.5.b`.
- **`TOOLCALLING.streamv2.5.c`** Truncation mid-message JSON. Streaming form of `TOOLCALLING.batch.5.c`.
- **`TOOLCALLING.streamv2.5.d`** Multi-call, last call missing only end marker (body complete). Streaming form of `TOOLCALLING.batch.5.d`.
- **`TOOLCALLING.streamv2.5.e`** Multi-call, last call truncated mid-arg-value. Streaming form of `TOOLCALLING.batch.5.e`.
- **`TOOLCALLING.streamv2.6.a`** Canonical empty {} message body. Streaming form of `TOOLCALLING.batch.6.a`.
- **`TOOLCALLING.streamv2.6.b`** Whitespace inside empty {}. Streaming form of `TOOLCALLING.batch.6.b`.
- **`TOOLCALLING.streamv2.6.c`** No <|message|> body. Streaming form of `TOOLCALLING.batch.6.c`.
- **`TOOLCALLING.streamv2.7.a`** Standard scalar types. Streaming form of `TOOLCALLING.batch.7.a`.
- **`TOOLCALLING.streamv2.7.b`** Unicode + escaped chars. Streaming form of `TOOLCALLING.batch.7.b`.
- **`TOOLCALLING.streamv2.7.c`** Schema mismatch — string value where schema declares integer. Streaming form of `TOOLCALLING.batch.7.c`.
- **`TOOLCALLING.streamv2.7.d`** Nested object + array. Streaming form of `TOOLCALLING.batch.7.d`.
- **`TOOLCALLING.streamv2.7.e`** Large / deep JSON-edge argument payload. Streaming form of `TOOLCALLING.batch.7.e`.
- **`TOOLCALLING.streamv2.7.f`** Numeric precision edge preserves integer-like number literal. Streaming form of `TOOLCALLING.batch.7.f`.
- **`TOOLCALLING.streamv2.8.a`** Narration before tool call only. Streaming form of `TOOLCALLING.batch.8.a`.
- **`TOOLCALLING.streamv2.8.b`** Narration after tool call only. Streaming form of `TOOLCALLING.batch.8.b`.
- **`TOOLCALLING.streamv2.8.c`** Narration both before and after (sandwich). Streaming form of `TOOLCALLING.batch.8.c`.
- **`TOOLCALLING.streamv2.8.d`** Narration between multiple tool calls. Streaming form of `TOOLCALLING.batch.8.d`.
- **`TOOLCALLING.streamv2.9.a`** Empty model text. Streaming form of `TOOLCALLING.batch.9.a`.
- **`TOOLCALLING.streamv2.9.b`** Blank / whitespace-only model text. Streaming form of `TOOLCALLING.batch.9.b`.
- **`TOOLCALLING.streamv2.10`** Duplicate calls (same name twice). Streaming form of `TOOLCALLING.batch.10`.
- **`TOOLCALLING.streamv2.13`** Unknown tool name absent from supplied tools. Streaming form of `TOOLCALLING.batch.13`.
- **`TOOLCALLING.streamv2.50`** Partial-token chunking (chunk boundary splits a grammar token mid-string). Partial-token matching must return `keep buffering`, not flush as plain text. Streaming-only — no batch analog.

Stream fixtures may include `delta_token_ids` on each chunk. Text-only chunks are enough for most parser families, but token-ID-dependent streaming parsers (currently vLLM's Harmony / `openai` parser) must record `delta_token_ids`; capture should mark those cases unavailable rather than inventing IDs.

## `TOOLCALLING.streamv2.50` — Partial-token chunking

Streaming-only (no batch analog). Chunk boundary splits a grammar token
mid-string (start fence, end fence, or parameter name / value straddles a chunk
boundary). Partial matches must return "keep buffering" rather than flushing as
plain text and completing on a later chunk.

- Applies to every tool-call parser.
