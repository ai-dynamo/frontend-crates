# Writing your own unified parser

You do not need to fork this crate to change how a model's output is parsed. You can implement the `UnifiedParser` trait in your own crate, register it by family name, and it will be used for every request routed to that name — including for a family this crate already ships, which your registration shadows.

Everything below uses only the public API. There is nothing private you need.

## 1. Implement the trait

One method is required. `finish` is required too, because a parser that forgets to flush would silently drop the tail of every stream.

```rust
use dynamo_parsers_v2::{
    Tool, UnifiedParser, UnifiedParserEvent, UnifiedParserOutput,
};
use anyhow::Result;

#[derive(Default)]
struct AcmeParser {
    buffered: String,
}

impl UnifiedParser for AcmeParser {
    /// Called once per decoded delta. Emit only what is now COMMITTED, and keep
    /// back only what is still ambiguous — here, a trailing `<` that might be the
    /// start of a marker. Emitting `delta` AND retaining it would emit it twice.
    fn parse_into(&mut self, delta: &str, output: &mut UnifiedParserOutput) -> Result<()> {
        self.buffered.push_str(delta);
        if let Some(cut) = self.buffered.rfind('<') {
            let emit: String = self.buffered[..cut].to_string();
            self.buffered = self.buffered[cut..].to_string();
            output.push_text(emit);
        } else {
            let all = std::mem::take(&mut self.buffered);
            output.push_text(all);
        }
        Ok(())
    }

    /// Called once at end of stream. Whatever is still held back is ordinary text
    /// once the stream has ended.
    fn finish(&mut self) -> Result<UnifiedParserOutput> {
        let mut out = UnifiedParserOutput::default();
        out.push_text(std::mem::take(&mut self.buffered));
        Ok(out)
    }

    fn reset(&mut self) -> String {
        std::mem::take(&mut self.buffered)
    }
}
```

This example is compiled and run as an integration test — [`tests/vendor_parser_example.rs`](tests/vendor_parser_example.rs). It uses only this crate's public API, exactly as your crate would.

The block above and that file are compared **by a test**, not by hand: `doc_trait_method_bodies_match_compiled_example` parses the first Rust block out of this file and asserts that the two implement the same trait METHODS with the same bodies. That check exists because the claim it replaces was false — the two drifted, the documented example double-emitted its buffer, and nothing caught it.

Know its limit: the test does not COMPILE this Markdown block, and it compares only the `impl UnifiedParser` methods. Renaming a struct field or changing an import here alone would break the documented example while the test stays green. The compiled file is the authority; treat this block as a copy that is guarded at the method level.

`UnifiedParserOutput` gives you `push_text`, `push_reasoning` and `push_call`. The first two coalesce: appending text onto a trailing text event extends it rather than adding a second one.

## 2. Register it

```rust
fn acme_factory(_tools: &[Tool]) -> Result<Box<dyn UnifiedParser>> {
    Ok(Box::new(AcmeParser::default()))
}

fn main() {
    dynamo_parsers_v2::register_unified_parser("acme_v1", acme_factory);
    // ... start serving. Requests for family "acme_v1" now use your parser.
}
```

Register during startup, before serving. Registration is process-wide and takes effect for parsers created after it returns; a parser already mid-stream keeps the implementation it was built with, so a request cannot straddle the change.

## 3. Replacing a family this crate already ships

Register under the existing name. Your factory is consulted first, so it wins:

```rust
// You disagree with how this crate parses qwen3. Use yours instead.
dynamo_parsers_v2::register_unified_parser("qwen3", my_qwen3_factory);
```

`unregister_unified_parser("qwen3")` removes yours and the built-in becomes reachable again — the built-in is shadowed, never replaced. `builtin_unified_families()` tells you what ships here; `vendor_unified_families()` tells you what has been registered on top.

This is deliberately supported. Disagreeing with one of our families should cost you a registration call, not a fork.

## 4. What the contract requires

These are the responsibilities a unified parser carries, and the reasons they exist. A parser that violates one will look correct in a demo and fail in production. The built-in conformance corpus exercises several of them for the families it ships; it cannot check factory/global-state isolation or your parser's own `reset` and error behaviour, and it does not run your parser at all until you enrol it (see below).

| | Requirement |
|---|---|
| **One parser per stream** | A factory builds one parser for one choice of one request. Keep per-stream state in the parser, never in the factory or a global. |
| **Order is the output** | Emit events in the order the model produced them. Do not hoist all reasoning to the front; that is precisely the defect the ordered event stream exists to remove. |
| **Split-invariance** | The same bytes must produce the same ASSEMBLED output regardless of where the transport split them. Test every split point around your markers, not just a few. Raw event boundaries may legitimately differ between splits — one chunking can commit `Text("alpha ")` + `Text("rest")` where another commits `Text("alpha rest")` — which is why the check is `assembled()`, not the event vector. |
| **Never leak your own markup** | Bytes you consumed as structure must not reappear in visible text. |
| **Arguments keep their meaning and order** | Do not drop, duplicate, or corrupt argument values, and preserve the model's key order where the family contract depends on it. If you emit incremental fragments, the fragments you chose must concatenate into the intended arguments JSON. Verbatim BYTES are only required where the model already emits API-shaped JSON — the shipped XML families deliberately schema-type and re-serialize, then restore source key order. |
| **Recover, do not panic** | Malformed or truncated output is normal. Emit what you can and drop what you cannot; returning an error should be reserved for genuinely unusable input. When driven through `parse_into`, whatever you already appended to the caller's output stays committed on `Err`; `push` owns its buffer and returns `Result<Vec<_>>`, which has nowhere to carry partial output, so it cannot honour that guarantee. |
| **Flush on `finish`** | Do not silently drop the tail of a stream. What to DO with an unterminated channel is your family's policy — the shipped Qwen parser promotes open reasoning rather than leaking it as text, but a grammar with no reasoning channel, or one that treats an unterminated opener as visible recovery text, is making a different and equally valid call. State yours. |
| **Override `reset` if you buffer** | The default returns an empty string and clears nothing. A parser that holds bytes back MUST override it, or a caller following the documented recovery path after a `parse_into` error resumes on your stale buffer and mis-numbers tool indices. Reset every field that carries stream position, tool index included — the text you hand back is a NEW stream. |

## 5. Optional capabilities

All have defaults, so implement only what your grammar needs.

| Method | Implement it when |
|---|---|
| `initialize(&[u32])` | your prompt can end mid-channel and you want to detect that from prompt tokens |
| `preserve_special_tokens()` | your markers ARE tokenizer special tokens, so text that dropped them is unparseable |
| `tool_call_id(idx)` | your grammar names the call itself and the id should come from the model |

## 6. Check it against the corpus

The conformance corpus is the useful part of this repo — the cases you have not thought of: malformed envelopes, markers split mid-token, marker-looking text inside a JSON string, reasoning interleaved with calls.

Running the workspace command below does NOT exercise a vendor parser. There is no flag that points the suite at a registered family: the suite iterates `builtin_unified_families()`, and a vendor registration deliberately does not enrol itself there. Adding your family means adding cases to the corpus and wiring it into the harness. Until you do, a green run says nothing about your parser.

```bash
cargo test --workspace --all-targets --locked
```

A family with no cases would otherwise report as covered while nothing measured it, which is why enrolment is deliberate rather than automatic.

## 7. Alignment with peer traits

This trait is aligned with the peer streaming-parser traits other serving engines expose: `parse_into` is the required method, `finish` returns an output buffer, the event type carries the same variants in the same order, and `initialize` takes prompt token IDs. A parser written against a peer trait ports here mostly by renaming.

Two caveats worth knowing before you plan on a literal drop-in:

- Rust is nominally typed, so an identically-shaped type in another crate is still a different type. Porting is a mechanical translation, not a recompile.
- This crate adds surface the peer traits do not have — `parse_complete` and the assembled `UnifiedEvent` view. Both are additive, so a peer-shaped caller never sees them, and a peer-shaped parser that does not provide them falls back to their defaults.
