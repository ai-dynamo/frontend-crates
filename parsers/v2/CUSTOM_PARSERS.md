# Writing your own unified parser

You do not need to fork this crate to change how a model's output is parsed. You can implement the `UnifiedParser` trait in your own crate, register it by family name, and it will be used for every request routed to that name — including for a family this crate already ships, which your registration shadows.

Everything below uses only the public API. There is nothing private you need.

## 1. Implement the trait

`parse_into` and `finish` are required, because a parser that forgets to flush would silently drop the tail of every stream.

```rust
use dynamo_parsers_v2::{
    Tool, UnifiedParser, UnifiedParserEvent, UnifiedParserExt, UnifiedParserOutput,
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

`doc_trait_method_bodies_match_compiled_example` checks that this block's `impl UnifiedParser` methods match the compiled example, guarding against drift such as double-emitting buffered text.

The test compares method bodies; it does not compile this Markdown or check imports and struct fields. Keep those aligned with the compiled example too.

`UnifiedParserOutput` gives you `push_text`, `push_reasoning` and `push_call`. The first two coalesce: appending text onto a trailing text event extends it rather than adding a second one.

The required lifecycle is `initialize_request` → `parse_into` → `finish`, with `reset` for recovery. The peer-shaped `initialize(&[u32])` is a native-mode adapter that builds `UnifiedParserInit`; request-aware callers resolve prompt state, tool-output mode, and invalid-guided-payload policy into that one owned value. `push` and `parse_complete` come from the blanket `UnifiedParserExt` trait: import it when you want those allocation conveniences. They cannot be overridden through `impl UnifiedParser`, so `parse_complete` always runs the same `parse_into` + `finish` path as streaming.

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

Register before accepting requests. Registration is process-wide; existing parsers and constructors that already looked up a factory keep the old implementation.

## 3. Replacing a family this crate already ships

Register under the existing name. Your factory is consulted first, so it wins:

```rust
// You disagree with how this crate parses qwen3. Use yours instead.
dynamo_parsers_v2::register_unified_parser("qwen3", my_qwen3_factory);
```

`unregister_unified_parser("qwen3")` removes yours and the built-in becomes reachable again — the built-in is shadowed, never replaced. `builtin_unified_families()` tells you what ships here; `vendor_unified_families()` tells you what has been registered on top.

Built-in aliases canonicalize together: registering `qwen3` also shadows `qwen3_coder`, and unregistering either restores both. Arbitrary vendor names remain exact names. Registration changes factory lookup only; it does not add a tool-only entry, a structural-tag builder, a serving route, or conformance enrollment. Integrate and test each of those separately when needed.

## 4. What the contract requires

These are the responsibilities a unified parser carries, and the reasons they exist. A parser that violates one will look correct in a demo and fail in production. The built-in conformance corpus exercises several of them for the families it ships; it cannot check factory/global-state isolation or your parser's own `reset` and error behaviour, and it does not run your parser at all until you enrol it (see below).

| | Requirement |
|---|---|
| **One parser per stream** | A factory builds one parser for one choice of one request. Keep per-stream state in the parser, never in the factory or a global. |
| **Order is the output** | Emit events in the order the model produced them. Do not hoist all reasoning to the front; that is precisely the defect the ordered event stream exists to remove. |
| **Split-invariance** | The same bytes must produce the same ASSEMBLED output regardless of where the transport split them. Test every split point around your markers, not just a few. Raw event boundaries may legitimately differ between splits — one chunking can commit `Text("alpha ")` + `Text("rest")` where another commits `Text("alpha rest")` — which is why the check is `assembled()`, not the event vector. |
| **Never leak your own markup** | Bytes you consumed as structure must not reappear in visible text. |
| **Arguments keep their meaning and order** | Do not drop, duplicate, or corrupt argument values, and preserve the model's key order where the family contract depends on it. If you emit incremental fragments, the fragments you chose must concatenate into the intended arguments JSON. Verbatim BYTES are only required where the model already emits API-shaped JSON — the shipped XML families deliberately schema-type and re-serialize, then restore source key order. |
| **Recover, do not panic** | Malformed or truncated output is normal. Emit what you can and drop what you cannot; returning an error should be reserved for genuinely unusable input. When driven through `parse_into`, whatever you already appended to the caller's output stays committed on `Err`; the `UnifiedParserExt::push` helper owns its buffer and returns `Result<Vec<_>>`, which has nowhere to carry partial output, so it cannot honour that guarantee. |
| **Flush on `finish`** | Do not silently drop the tail of a stream. What to DO with an unterminated channel is your family's policy — the shipped Qwen parser promotes open reasoning rather than leaking it as text, but a grammar with no reasoning channel, or one that treats an unterminated opener as visible recovery text, is making a different and equally valid call. State yours. |
| **Override `reset` if you buffer** | The default returns an empty string and clears nothing. A parser that holds bytes back MUST override it, or a caller following the documented recovery path after a `parse_into` error resumes on your stale buffer and mis-numbers tool indices. Reset every field that carries stream position, tool index included — the text you hand back is a NEW stream. |

## 5. Optional capabilities

All have defaults, so implement only what your grammar needs.

| Method | Implement it when |
|---|---|
| `initialize(&[u32])` | a peer-shaped caller needs neutral native-mode initialization |
| `initialize_request(UnifiedParserInit)` | the caller has resolved prompt tokens, starting channel, tool-output mode, and invalid-guided-payload policy for this request |
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

The required `UnifiedParser` lifecycle is aligned with the peer streaming-parser traits other serving engines expose: `parse_into` is the required advance method, `finish` returns an output buffer, the event type carries the same variants in the same order, and `initialize` takes prompt token IDs. A parser written against a peer trait ports here mostly by renaming.

Two caveats worth knowing before you plan on a literal drop-in:

- Rust is nominally typed, so an identically-shaped type in another crate is still a different type. Porting is a mechanical translation, not a recompile.
- This crate adds surface the peer traits do not have — `initialize_request(UnifiedParserInit)`, the blanket `UnifiedParserExt` helpers (`push` and `parse_complete`), and the assembled `UnifiedEvent` view. The resolved request initializer is an additive default on `UnifiedParser`; the helpers are additive conveniences that vendors cannot override.
