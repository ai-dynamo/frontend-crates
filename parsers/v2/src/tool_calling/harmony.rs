// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Harmony (gpt-oss) tool-call streaming parser.
//!
//! This parser accepts either decoded text or Harmony token IDs and emits
//! `ToolCallResponseChunk` deltas (id + name first, then `arguments`) from the
//! `commentary to=functions.NAME` channel. It reparses the accumulated Harmony
//! text after each chunk and emits only newly completed calls, so incomplete
//! trailing envelopes are suppressed until EOF. At EOF, a directed function call
//! with complete JSON arguments can recover even if only `<|call|>` is missing.
//!
//! Why harmony first: it's the one family where token IDs matter for correctness.
//! The reasoning gpt_oss parser handles the `analysis`/`final` channels; this is
//! the tool-call half over the `commentary` channel.
//!
//! Scope: tool calls only. Reasoning/normal text over the same stream stays with
//! the reasoning parser. Assembly into the OpenAI wire response (finish_reason,
//! n>1, logprobs) is the serving layer's job, not the parser's.

use std::sync::OnceLock;

use crate::tool_calling::v1core::{CalledFunctionStream, ToolCallResponseChunk, ToolCallType};

use crate::tool_calling::traits::{Tool, ToolCallDelta, ToolParseResult, ToolParser};
use openai_harmony::{HarmonyEncoding, HarmonyEncodingName, load_harmony_encoding};

use super::harmony_grammar::{HarmonySnapshot, extract_calls_via_regex};
use super::harmony_recovery::{
    normal_text_after_parse_failure, strip_harmony_protocol_from_normal_text,
};

static GLOBAL_HARMONY_ENCODING: OnceLock<Result<HarmonyEncoding, anyhow::Error>> = OnceLock::new();

/// Load (once) the gpt-oss harmony encoding.
///
/// Mirrors the reasoning parser's OS-thread trick: `load_harmony_encoding` builds
/// and drops a Tokio runtime internally, which panics if dropped inside an async
/// context, so run it on a fresh thread. Init runs at most once per process.
fn get_harmony_encoding() -> &'static Result<HarmonyEncoding, anyhow::Error> {
    GLOBAL_HARMONY_ENCODING.get_or_init(|| {
        std::thread::spawn(|| load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss))
            .join()
            .unwrap_or_else(|_| Err(anyhow::anyhow!("harmony encoding loader thread panicked")))
    })
}

/// Encode text to gpt-oss token ids. Used to build token fixtures from canonical
/// harmony text — the same encode the reasoning parser uses as its WAR — surfaced
/// here so the token path can be exercised without a live model.
pub fn encode_harmony(text: &str) -> anyhow::Result<Vec<u32>> {
    let enc = get_harmony_encoding()
        .as_ref()
        .map_err(|e| anyhow::anyhow!("harmony encoding unavailable: {e}"))?;
    Ok(enc.tokenizer().encode_with_special_tokens(text))
}

/// Decode token ids back to text (for human-readable `delta_text` in fixtures).
pub fn decode_harmony(token_ids: &[u32]) -> anyhow::Result<String> {
    Ok(decode_harmony_strict(token_ids).unwrap_or_default())
}

fn decode_harmony_strict(token_ids: &[u32]) -> anyhow::Result<String> {
    let enc = get_harmony_encoding()
        .as_ref()
        .map_err(|e| anyhow::anyhow!("harmony encoding unavailable: {e}"))?;
    enc.tokenizer()
        .decode_utf8(token_ids)
        .map_err(|e| anyhow::anyhow!("harmony decode failed: {e}"))
}

/// Per-chunk streaming result: the append-only deltas produced from this chunk.
///
/// `normal_text` carries non-tool-call output interleaved with calls. Harmony
/// emits it only at stream finish after protocol cleanup, so split markers do
/// not leak into user-visible content.
#[derive(Default, Debug)]
pub struct ToolStreamResult {
    pub normal_text: String,
    pub tool_call_chunks: Vec<ToolCallResponseChunk>,
}

/// Harmony tool-call streaming parser.
pub struct HarmonyToolStreamParser {
    scan_buffer: String,
    pending_token_ids: Vec<u32>,
    emitted_calls: usize,
    normal_text_emitted: bool,
    next_id: u64,
}

impl HarmonyToolStreamParser {
    pub fn new() -> anyhow::Result<Self> {
        get_harmony_encoding()
            .as_ref()
            .map_err(|e| anyhow::anyhow!("harmony encoding unavailable: {e}"))?;
        Ok(Self {
            scan_buffer: String::new(),
            pending_token_ids: Vec::new(),
            emitted_calls: 0,
            normal_text_emitted: false,
            next_id: 0,
        })
    }

    fn gen_id(&mut self) -> String {
        let id = format!("call_{:08}", self.next_id);
        self.next_id += 1;
        id
    }

    /// Feed one chunk of text.
    pub fn parse_tool_call_streaming_text(&mut self, delta_text: &str) -> ToolStreamResult {
        self.scan_buffer.push_str(delta_text);
        self.emit_snapshot_delta(false)
    }

    /// Feed one chunk of token ids.
    pub fn parse_tool_call_streaming_incremental(
        &mut self,
        delta_token_ids: &[u32],
    ) -> ToolStreamResult {
        self.pending_token_ids.extend_from_slice(delta_token_ids);
        match decode_harmony_strict(&self.pending_token_ids) {
            Ok(text) => {
                self.pending_token_ids.clear();
                self.parse_tool_call_streaming_text(&text)
            }
            Err(e) => {
                tracing::warn!("harmony decode pending token stream failed: {e}");
                ToolStreamResult::default()
            }
        }
    }

    fn emit_snapshot_delta(&mut self, include_normal_text: bool) -> ToolStreamResult {
        let snapshot = parse_harmony_snapshot(&self.scan_buffer, false);
        self.emit_snapshot_delta_from_snapshot(snapshot, include_normal_text)
    }

    /// Stream EOF. Flushes pending token text, emits newly completed calls, and
    /// emits cleaned normal text once.
    pub fn finish_tool_call_stream(&mut self) -> ToolStreamResult {
        if !self.pending_token_ids.is_empty() {
            match decode_harmony_strict(&self.pending_token_ids) {
                Ok(text) => {
                    self.scan_buffer.push_str(&text);
                    self.pending_token_ids.clear();
                }
                Err(e) => {
                    tracing::warn!("harmony decode failed while finishing token stream: {e}");
                    self.pending_token_ids.clear();
                }
            }
        }
        // Match the v1 batch parser: do NOT recover commentary calls missing the
        // closing <|call|> token at EOF. dynamo #10366 moved analysis-channel
        // tool-call recovery to the reasoning parser, so the batch tool-call
        // parser drops these — the stream parser stays consistent with it
        // (batch.5.a -> no call, batch.5.d -> only the fenced call).
        let snapshot = parse_harmony_snapshot(&self.scan_buffer, false);
        self.emit_snapshot_delta_from_snapshot(snapshot, true)
    }

    fn emit_snapshot_delta_from_snapshot(
        &mut self,
        snapshot: HarmonySnapshot,
        include_normal_text: bool,
    ) -> ToolStreamResult {
        let mut chunks = Vec::new();

        for (index, call) in snapshot.calls.iter().enumerate().skip(self.emitted_calls) {
            let id = self.gen_id();
            chunks.push(ToolCallResponseChunk {
                index: index as u32,
                id: Some(id),
                tp: Some(ToolCallType::Function),
                function: Some(CalledFunctionStream {
                    name: Some(call.name.clone()),
                    arguments: None,
                }),
            });
            chunks.push(ToolCallResponseChunk {
                index: index as u32,
                id: None,
                tp: None,
                function: Some(CalledFunctionStream {
                    name: None,
                    arguments: Some(call.arguments.clone()),
                }),
            });
        }
        self.emitted_calls = snapshot.calls.len();

        let normal_text = if include_normal_text && !self.normal_text_emitted {
            self.normal_text_emitted = true;
            snapshot.normal_text
        } else {
            String::new()
        };

        ToolStreamResult {
            normal_text,
            tool_call_chunks: chunks,
        }
    }
}

fn parse_harmony_snapshot(text: &str, allow_eof_recovery: bool) -> HarmonySnapshot {
    let (calls, residual) = extract_calls_via_regex(text, allow_eof_recovery);
    let normal_text = if calls.is_empty() {
        normal_text_after_parse_failure(text, "parse_failed_no_recovered_calls")
    } else {
        strip_harmony_protocol_from_normal_text(&residual, "regex_recovery_residual")
    };
    HarmonySnapshot { calls, normal_text }
}

/// Convert a `ToolStreamResult` to the trait's `ToolParseResult`.
///
/// Maps `ToolCallResponseChunk` → `ToolCallDelta`:
/// - drops parser-minted `id` (vLLM serving layer mints its own)
/// - coerces `arguments: None → ""` (vLLM Rust contract: always-present string)
/// - `normal_text` passes through after Harmony protocol cleanup
fn stream_result_to_parser_output(r: ToolStreamResult) -> ToolParseResult {
    ToolParseResult {
        normal_text: r.normal_text,
        calls: r
            .tool_call_chunks
            .into_iter()
            .map(|c| {
                let name = c.function.as_ref().and_then(|f| f.name.clone());
                let arguments = c
                    .function
                    .as_ref()
                    .and_then(|f| f.arguments.clone())
                    .unwrap_or_default();
                ToolCallDelta {
                    tool_index: c.index as usize,
                    complete: !arguments.is_empty(),
                    name,
                    arguments,
                }
            })
            .collect(),
    }
}

impl HarmonyToolStreamParser {
    /// Clear parser state and return currently uncommitted buffered text.
    pub fn reset(&mut self) -> String {
        let buffered = std::mem::take(&mut self.scan_buffer);
        if let Ok(reset) = Self::new() {
            *self = reset;
        } else {
            self.pending_token_ids.clear();
            self.emitted_calls = 0;
            self.normal_text_emitted = false;
            self.next_id = 0;
        }
        buffered
    }
}

/// `ToolParser` implementation for `HarmonyToolStreamParser`.
///
/// Bridges Dynamo's streaming parser to the vLLM-shaped Rust trait contract:
/// - `push(&str)` → text path
/// - `push_tokens(&[u32])` → token-native path (higher fidelity for Harmony)
/// - `finish()` → EOS flush
/// - `preserve_special_tokens` → `false` (Harmony uses formatting tokens, not Unicode specials)
impl ToolParser for HarmonyToolStreamParser {
    fn create(_tools: &[Tool]) -> anyhow::Result<Box<dyn ToolParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self::new()?))
    }

    fn prefers_tokens(&self) -> bool {
        true
    }

    fn push(&mut self, chunk: &str) -> anyhow::Result<ToolParseResult> {
        Ok(stream_result_to_parser_output(
            self.parse_tool_call_streaming_text(chunk),
        ))
    }

    fn push_tokens(&mut self, ids: &[u32]) -> anyhow::Result<ToolParseResult> {
        Ok(stream_result_to_parser_output(
            self.parse_tool_call_streaming_incremental(ids),
        ))
    }

    fn finish(&mut self) -> anyhow::Result<ToolParseResult> {
        Ok(stream_result_to_parser_output(
            self.finish_tool_call_stream(),
        ))
    }
}

/// Assemble streamed deltas back into `(name, arguments-json-string)` per index —
/// the *consumer's* job (accumulate by index, concat argument fragments), surfaced
/// here for parity tests.
pub fn assemble_tool_calls(chunks: &[ToolCallResponseChunk]) -> Vec<(String, String)> {
    use std::collections::BTreeMap;
    let mut names: BTreeMap<u32, String> = BTreeMap::new();
    let mut args: BTreeMap<u32, String> = BTreeMap::new();
    for c in chunks {
        if let Some(f) = &c.function {
            if let Some(n) = &f.name {
                names.entry(c.index).or_default().push_str(n);
            }
            if let Some(a) = &f.arguments {
                args.entry(c.index).or_default().push_str(a);
            }
        }
    }
    names
        .into_iter()
        .map(|(idx, name)| (name, args.get(&idx).cloned().unwrap_or_default()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool_calling::traits::{ToolParseResult, ToolParserInput};

    // Canonical single tool call (TOOLCALLING.batch.1, harmony family).
    const CANON: &str = "<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{\"location\":\"NYC\"}<|call|>";

    fn parse_complete_for_test(parser: &mut impl ToolParser, text: &str) -> ToolParseResult {
        parser.parse_complete(text).expect("parse_complete")
    }

    #[test]
    fn single_tool_call_from_tokens() {
        let tokens = encode_harmony(CANON).expect("encode");
        let mut parser = HarmonyToolStreamParser::new().expect("new");

        // Feed in 3-token chunks to prove genuine incremental token streaming.
        let mut all = Vec::new();
        for chunk in tokens.chunks(3) {
            all.extend(
                parser
                    .parse_tool_call_streaming_incremental(chunk)
                    .tool_call_chunks,
            );
        }
        all.extend(parser.finish_tool_call_stream().tool_call_chunks);

        let calls = assemble_tool_calls(&all);
        assert_eq!(calls.len(), 1, "expected one tool call, got {calls:?}");
        assert_eq!(calls[0].0, "get_weather");
        let args: serde_json::Value = serde_json::from_str(&calls[0].1).expect("args json");
        assert_eq!(args, serde_json::json!({"location": "NYC"}));

        // vLLM wire shape: the name must stream before any arguments fragment.
        let first_named = all
            .iter()
            .position(|c| c.function.as_ref().and_then(|f| f.name.as_ref()).is_some());
        let first_args = all.iter().position(|c| {
            c.function
                .as_ref()
                .and_then(|f| f.arguments.as_ref())
                .is_some()
        });
        assert!(
            first_named.is_some() && first_named <= first_args,
            "name must stream before arguments (got name={first_named:?}, args={first_args:?})"
        );
    }

    #[test]
    fn adjacent_channel_first_tool_calls_in_one_token_chunk() {
        let combined = concat!(
            "<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{\"location\":\"NYC\"}<|call|>",
            "<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{\"location\":\"LA\"}<|call|>"
        );
        let tokens = encode_harmony(combined).expect("encode");
        let mut parser = HarmonyToolStreamParser::new().expect("new");

        let mut all = parser
            .parse_tool_call_streaming_incremental(&tokens)
            .tool_call_chunks;
        all.extend(parser.finish_tool_call_stream().tool_call_chunks);

        let calls = assemble_tool_calls(&all);
        assert_eq!(calls.len(), 2, "expected two tool calls, got {calls:?}");
        assert_eq!(calls[0].0, "get_weather");
        assert_eq!(calls[1].0, "get_weather");
        let first_args: serde_json::Value =
            serde_json::from_str(&calls[0].1).expect("first args json");
        let second_args: serde_json::Value =
            serde_json::from_str(&calls[1].1).expect("second args json");
        assert_eq!(first_args, serde_json::json!({"location": "NYC"}));
        assert_eq!(second_args, serde_json::json!({"location": "LA"}));
    }

    #[test]
    fn single_tool_call_from_text_streams_before_finish() {
        let mut parser = HarmonyToolStreamParser::new().expect("new");

        let first = parser.parse_tool_call_streaming_text(CANON);
        assert!(
            !first.tool_call_chunks.is_empty(),
            "text path should emit before finish for a complete content chunk"
        );

        let mut all = first.tool_call_chunks;
        all.extend(parser.finish_tool_call_stream().tool_call_chunks);

        let calls = assemble_tool_calls(&all);
        assert_eq!(calls.len(), 1, "expected one tool call, got {calls:?}");
        assert_eq!(calls[0].0, "get_weather");
        let args: serde_json::Value = serde_json::from_str(&calls[0].1).expect("args json");
        assert_eq!(args, serde_json::json!({"location": "NYC"}));
    }

    #[test]
    fn text_path_tolerates_split_harmony_markers() {
        let chunks = [
            "<|",
            "cha",
            "nnel|",
            ">commentary",
            " to=functions.get_w",
            "ea",
            "the",
            "r <|c",
            "onstrain|>j",
            "son<|message|>{\"loc",
            "at",
            "ion",
            "\":\"NY",
            "C\"}<|call|>",
        ];
        let mut parser = HarmonyToolStreamParser::new().expect("new");
        let mut all = Vec::new();
        let mut emitted_before_finish = false;

        for chunk in chunks {
            let result = parser.parse_tool_call_streaming_text(chunk);
            emitted_before_finish |= !result.tool_call_chunks.is_empty();
            all.extend(result.tool_call_chunks);
        }
        assert!(
            emitted_before_finish,
            "text path should not hold every delta until finish"
        );
        all.extend(parser.finish_tool_call_stream().tool_call_chunks);

        let calls = assemble_tool_calls(&all);
        assert_eq!(calls.len(), 1, "expected one tool call, got {calls:?}");
        assert_eq!(calls[0].0, "get_weather");
        let args: serde_json::Value = serde_json::from_str(&calls[0].1).expect("args json");
        assert_eq!(args, serde_json::json!({"location": "NYC"}));
    }

    // ── ToolParser trait tests ────────────────────────────────────────────────

    #[test]
    fn trait_parse_tokens_matches_incremental() {
        let tokens = encode_harmony(CANON).expect("encode");
        let mut parser = HarmonyToolStreamParser::new().expect("new");

        let mut output = ToolParseResult::default();
        for chunk in tokens.chunks(3) {
            output.append(parser.push_tokens(chunk).expect("push_tokens"));
        }
        let r = parser.finish().expect("finish");
        output.append(r);
        let all = output.calls;

        // First delta: name present, arguments == "" (vLLM always-present contract).
        let name_delta = &all[0];
        assert_eq!(name_delta.name.as_deref(), Some("get_weather"));
        assert_eq!(name_delta.arguments, "");
        assert!(!name_delta.complete);
        // All subsequent deltas: name absent, arguments non-empty fragments.
        for delta in &all[1..] {
            assert!(delta.name.is_none());
            assert!(delta.complete);
        }
        // Concatenating all arguments gives the full JSON.
        let full_args_str: String = all.iter().map(|d| d.arguments.as_str()).collect();
        let full_args: serde_json::Value =
            serde_json::from_str(&full_args_str).expect("concatenated args json");
        assert_eq!(full_args, serde_json::json!({"location": "NYC"}));
    }

    #[test]
    fn trait_parse_complete_helper_equals_push_plus_finish() {
        // vLLM keeps parse_complete as a test helper over push + finish + coalesce.
        // Verify it produces one coalesced call matching the streaming path.
        let mut parser = HarmonyToolStreamParser::new().expect("new");
        let result = parse_complete_for_test(&mut parser, CANON);

        assert_eq!(result.normal_text, ""); // Harmony never produces normal_text
        assert_eq!(result.calls.len(), 1, "coalesce should merge to 1 call");
        assert_eq!(result.calls[0].name.as_deref(), Some("get_weather"));
        let args: serde_json::Value =
            serde_json::from_str(&result.calls[0].arguments).expect("args json");
        assert_eq!(args, serde_json::json!({"location": "NYC"}));
    }

    #[test]
    fn trait_arguments_always_string_never_none() {
        // vLLM Rust contract: arguments is always "" on name delta, never absent.
        let tokens = encode_harmony(CANON).expect("encode");
        let mut parser = HarmonyToolStreamParser::new().expect("new");

        let mut output = ToolParseResult::default();
        for chunk in tokens.chunks(1) {
            output.append(parser.push_tokens(chunk).expect("push_tokens"));
        }
        output.append(parser.finish().expect("finish"));
        let all = output.calls;

        // Every delta must have arguments as a String (never panic on empty check).
        for delta in &all {
            // This is the contract: arguments is always a String, even if "".
            let _ = delta.arguments.is_empty(); // would panic if Option
        }
        // The name-only first delta must have arguments == "".
        let name_delta = all.iter().find(|d| d.name.is_some()).expect("name delta");
        assert_eq!(name_delta.arguments, "");
    }

    #[test]
    fn trait_preserve_special_tokens_false() {
        let parser = HarmonyToolStreamParser::new().expect("new");
        assert!(!parser.preserve_special_tokens());
    }

    #[test]
    fn trait_normal_text_empty_for_tool_only_harmony() {
        let mut parser = HarmonyToolStreamParser::new().expect("new");
        let r = parse_complete_for_test(&mut parser, CANON);
        assert_eq!(r.normal_text, "");
    }

    #[test]
    fn surrounding_narration_matches_batch_case_2_c() {
        let text = concat!(
            "I need both. ",
            "<|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{\"location\":\"NYC\"}<|call|>",
            "<|start|>assistant<|channel|>commentary to=functions.get_time <|constrain|>json<|message|>{\"timezone\":\"EST\"}<|call|>",
            " Done."
        );
        let mut parser = HarmonyToolStreamParser::new().expect("new");
        let result = parse_complete_for_test(&mut parser, text);

        assert_eq!(result.normal_text, "I need both.  Done.");
        assert_eq!(result.calls.len(), 2);
        assert_eq!(result.calls[0].name.as_deref(), Some("get_weather"));
        assert_eq!(result.calls[1].name.as_deref(), Some("get_time"));
        let first: serde_json::Value =
            serde_json::from_str(&result.calls[0].arguments).expect("first args");
        let second: serde_json::Value =
            serde_json::from_str(&result.calls[1].arguments).expect("second args");
        assert_eq!(first, serde_json::json!({"location": "NYC"}));
        assert_eq!(second, serde_json::json!({"timezone": "EST"}));
    }

    #[test]
    fn missing_call_marker_drops_batch_case_5_a() {
        // A complete commentary call missing the closing <|call|> at EOF is NOT
        // recovered — stream matches the v1 batch parser (analysis-channel
        // recovery lives in the reasoning parser now; see dynamo #10366).
        let text = "<|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{\"location\":\"NYC\"}";
        let mut parser = HarmonyToolStreamParser::new().expect("new");
        let result = parse_complete_for_test(&mut parser, text);
        assert_eq!(result.normal_text, "");
        assert!(
            result.calls.is_empty(),
            "missing <|call|> must not be recovered (batch parity): {result:?}"
        );
    }

    #[test]
    fn truncated_single_envelope_drops_batch_case_5_c() {
        let text = "<|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{\"loc";
        let mut parser = HarmonyToolStreamParser::new().expect("new");
        let result = parse_complete_for_test(&mut parser, text);
        assert_eq!(result.normal_text, "");
        assert!(
            result.calls.is_empty(),
            "truncated envelope must not synthesize a call: {result:?}"
        );
    }

    #[test]
    fn unfenced_trailing_call_drops_batch_case_5_d() {
        // The trailing call is missing its <|call|>; only the fenced Boston call
        // is recovered — stream matches the v1 batch parser.
        let text = concat!(
            "I'll start by fetching the weather for both Boston and New York at the same time!",
            "<|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{\"location\":\"Boston\"}<|call|>",
            "<|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{\"location\":\"New York\"}"
        );
        let mut parser = HarmonyToolStreamParser::new().expect("new");
        let result = parse_complete_for_test(&mut parser, text);
        assert_eq!(
            result.normal_text,
            "I'll start by fetching the weather for both Boston and New York at the same time!"
        );
        assert_eq!(result.calls.len(), 1);
        assert_eq!(result.calls[0].name.as_deref(), Some("get_weather"));
        let args: serde_json::Value =
            serde_json::from_str(&result.calls[0].arguments).expect("args");
        assert_eq!(args, serde_json::json!({"location": "Boston"}));
    }

    #[test]
    fn truncated_trailing_call_drops_batch_case_5_e() {
        let text = concat!(
            "I'll start by fetching the weather for both Boston and New York at the same time!",
            "<|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{\"location\":\"Boston\"}<|call|>",
            "<|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{\"location\":\"New York"
        );
        let mut parser = HarmonyToolStreamParser::new().expect("new");
        let result = parse_complete_for_test(&mut parser, text);
        assert_eq!(
            result.normal_text,
            "I'll start by fetching the weather for both Boston and New York at the same time!"
        );
        assert_eq!(result.calls.len(), 1);
        assert_eq!(result.calls[0].name.as_deref(), Some("get_weather"));
        let args: serde_json::Value =
            serde_json::from_str(&result.calls[0].arguments).expect("args");
        assert_eq!(args, serde_json::json!({"location": "Boston"}));
    }

    #[test]
    fn narration_between_multiple_calls_matches_batch_case_8_d() {
        let text = concat!(
            "I will check the weather. ",
            "<|channel|>analysis<|message|>Need to use function get_weather.<|end|>",
            "<|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{\"location\":\"NYC\"}<|call|>",
            " Then check LA weather. ",
            "<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{\"location\":\"LA\"}<|call|>"
        );
        let mut parser = HarmonyToolStreamParser::new().expect("new");
        let result = parse_complete_for_test(&mut parser, text);

        // Surrounding text is kept VERBATIM (incl. the trailing space before the
        // second call's envelope) — matching the v1 jail passthrough. The v1
        // BATCH parser trims that boundary space; the divergence is documented
        // in the batch-via-stream allowlist.
        assert_eq!(
            result.normal_text,
            "I will check the weather.  Then check LA weather. "
        );
        assert_eq!(result.calls.len(), 2);
        let first: serde_json::Value =
            serde_json::from_str(&result.calls[0].arguments).expect("first args");
        let second: serde_json::Value =
            serde_json::from_str(&result.calls[1].arguments).expect("second args");
        assert_eq!(first, serde_json::json!({"location": "NYC"}));
        assert_eq!(second, serde_json::json!({"location": "LA"}));
    }

    #[test]
    fn trait_parse_input_accepts_text_or_tokens() {
        let tokens = encode_harmony(CANON).expect("encode");

        let mut text_parser = HarmonyToolStreamParser::new().expect("new");
        let mut text_output = ToolParseResult::default();
        text_output.append(
            text_parser
                .push_input(ToolParserInput::Text(CANON))
                .expect("text input"),
        );
        text_output.append(text_parser.finish().expect("finish"));
        let text_output = text_output.coalesce_calls();

        let mut token_parser = HarmonyToolStreamParser::new().expect("new");
        let mut token_output = ToolParseResult::default();
        for chunk in tokens.chunks(4) {
            token_output.append(
                token_parser
                    .push_input(ToolParserInput::Tokens(chunk))
                    .expect("token input"),
            );
        }
        token_output.append(token_parser.finish().expect("finish"));
        let token_output = token_output.coalesce_calls();

        assert_eq!(text_output, token_output);
        assert_eq!(token_output.calls.len(), 1);
        assert_eq!(token_output.calls[0].name.as_deref(), Some("get_weather"));
    }

    #[test]
    fn trait_create_ignores_tools_for_harmony() {
        let tools = [Tool {
            name: "get_weather".to_string(),
            description: Some("weather lookup".to_string()),
            parameters: serde_json::json!({"type": "object"}),
            strict: None,
        }];
        let mut parser = HarmonyToolStreamParser::create(&tools).expect("create");
        let mut output = ToolParseResult::default();
        output.append(parser.push(CANON).expect("push"));
        output.append(parser.finish().expect("finish"));
        let output = output.coalesce_calls();
        assert_eq!(output.calls.len(), 1);
        assert_eq!(output.calls[0].name.as_deref(), Some("get_weather"));
    }

    #[test]
    fn trait_reset_returns_uncommitted_text_buffer() {
        let mut parser = HarmonyToolStreamParser::new().expect("new");
        let mut output = ToolParseResult::default();
        output.append(parser.push("<|chan").expect("push"));
        assert!(output.calls.is_empty());

        let recovered = parser.reset();
        assert_eq!(recovered, "<|chan");

        output.append(parser.push(CANON).expect("push"));
        output.append(parser.finish().expect("finish"));
        let output = output.coalesce_calls();
        assert_eq!(output.calls.len(), 1);
        assert_eq!(output.calls[0].name.as_deref(), Some("get_weather"));
    }
}
