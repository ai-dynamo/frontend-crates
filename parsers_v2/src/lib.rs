// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! POC: a token-incremental Harmony (gpt-oss) tool-call streaming parser.
//!
//! This is the *token path*. It consumes `delta_token_ids` — not text — and wraps
//! `openai-harmony`'s `StreamableParser`, emitting per-chunk `ToolCallResponseChunk`
//! deltas (id + name first, then `arguments` fragments) off the
//! `commentary to=functions.NAME` channel. That's the vLLM wire shape, produced
//! incrementally as tokens arrive, with no jail and no buffer-then-release.
//!
//! Why harmony for the pilot: it's the one family where tokens genuinely drive
//! the parse (channel/recipient/content come straight out of the token stream),
//! and `StreamableParser` is a real incremental token parser — so this proves the
//! crate can stream on tokens the way vLLM's parsers do. The reasoning gpt_oss
//! parser already wraps the same `StreamableParser` for the `analysis`/`final`
//! channels; this is the tool-call half over the `commentary` channel.
//!
//! Scope: tool calls only. Reasoning/normal text over the same stream stays with
//! the reasoning parser. Assembly into the OpenAI wire response (finish_reason,
//! n>1, logprobs) is the serving layer's job, not the parser's.

use std::sync::OnceLock;

use dynamo_parsers::tool_calling::{CalledFunctionStream, ToolCallResponseChunk, ToolCallType};
use openai_harmony::{
    HarmonyEncoding, HarmonyEncodingName, StreamableParser, load_harmony_encoding,
};

static GLOBAL_HARMONY_ENCODING: OnceLock<Result<HarmonyEncoding, anyhow::Error>> = OnceLock::new();
// Longer than Harmony formatting markers, so text chunks split through
// `<|channel|>` / `<|message|>` settle before token commit.
const TEXT_STREAM_HOLDBACK_BYTES: usize = 16;

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

/// Per-chunk streaming result: the append-only tool-call deltas produced from the
/// tokens in this chunk (mirrors vLLM's `DeltaMessage.tool_calls`).
#[derive(Default, Debug)]
pub struct ToolStreamResult {
    pub tool_call_chunks: Vec<ToolCallResponseChunk>,
}

/// Token-incremental Harmony tool-call streaming parser.
pub struct HarmonyToolStreamParser {
    parser: StreamableParser,
    /// The <|start|> token id — used to detect whether a chunk already carries
    /// the full Harmony preamble (<|start|>assistant...) or starts directly with
    /// <|channel|>. Cached at construction time so we don't re-encode each call.
    start_token: u32,
    /// Preamble tokens (<|start|>assistant) to prepend when the input starts at
    /// <|channel|> without the role announcement.
    preamble_tokens: Vec<u32>,
    /// True when the inner parser is in ExpectStart state (between messages),
    /// i.e. we should prepend the preamble if the next chunk doesn't start with
    /// <|start|>. Starts true; set to false once the first token is processed;
    /// reset to true when a message terminates (<|call|> stop token).
    at_turn_start: bool,
    /// Index of the tool call currently being emitted.
    current_index: u32,
    /// Whether we're inside a `commentary to=functions.*` message right now.
    in_tool_call: bool,
    /// Whether id+name for `current_index` has already been emitted.
    header_emitted: bool,
    next_id: u64,
    /// Uncommitted text-path suffix. Text-only streams are re-tokenized with a
    /// small holdback so token boundaries can settle before they are fed to the
    /// inner Harmony token parser.
    text_buffer: String,
}

impl HarmonyToolStreamParser {
    pub fn new() -> anyhow::Result<Self> {
        let enc = get_harmony_encoding()
            .as_ref()
            .map_err(|e| anyhow::anyhow!("harmony encoding unavailable: {e}"))?;
        // Use None (ExpectStart state) so the parser accepts both:
        //   (a) full preamble:    <|start|>assistant<|channel|>...
        //   (b) channel-first:   <|channel|>...  (we prepend the preamble ourselves)
        let parser = StreamableParser::new(enc.clone(), None)
            .map_err(|e| anyhow::anyhow!("StreamableParser init failed: {e}"))?;
        let start_token = encode_harmony("<|start|>")
            .map_err(|e| anyhow::anyhow!("encode <|start|>: {e}"))?
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("<|start|> encoded to zero tokens"))?;
        let preamble_tokens = encode_harmony("<|start|>assistant")
            .map_err(|e| anyhow::anyhow!("encode preamble: {e}"))?;
        Ok(Self {
            parser,
            start_token,
            preamble_tokens,
            at_turn_start: true,
            current_index: 0,
            in_tool_call: false,
            header_emitted: false,
            next_id: 0,
            text_buffer: String::new(),
        })
    }

    fn gen_id(&mut self) -> String {
        let id = format!("call_{:08}", self.next_id);
        self.next_id += 1;
        id
    }

    /// Feed one chunk of text. Tolerates text split at arbitrary boundaries —
    /// including mid-token (`<|chan` + `nel|>`) — by re-tokenizing the pending
    /// suffix and committing only token prefixes that leave a small text holdback.
    /// That holdback lets BPE and Harmony formatting-token boundaries settle
    /// before the token parser sees them, while still emitting before stream end
    /// for normal text chunks.
    pub fn parse_tool_call_streaming_text(&mut self, delta_text: &str) -> ToolStreamResult {
        self.text_buffer.push_str(delta_text);
        self.flush_text_buffer(false)
    }

    fn flush_text_buffer(&mut self, flush_all: bool) -> ToolStreamResult {
        if self.text_buffer.is_empty() {
            return ToolStreamResult::default();
        }

        let (tokens, committed_bytes) = if flush_all {
            match encode_harmony(&self.text_buffer) {
                Ok(tokens) => (tokens, self.text_buffer.len()),
                Err(e) => {
                    tracing::warn!("harmony encode failed while flushing text stream: {e}");
                    return ToolStreamResult::default();
                }
            }
        } else {
            match committable_text_tokens(&self.text_buffer) {
                Ok(commit) => commit,
                Err(e) => {
                    tracing::warn!("harmony encode failed while streaming text: {e}");
                    return ToolStreamResult::default();
                }
            }
        };

        if tokens.is_empty() {
            return ToolStreamResult::default();
        }

        if committed_bytes == self.text_buffer.len() {
            self.text_buffer.clear();
        } else {
            self.text_buffer = self.text_buffer[committed_bytes..].to_string();
        }
        self.parse_tool_call_streaming_incremental(&tokens)
    }

    /// Feed one chunk of token ids; emit any new tool-call deltas.
    ///
    /// Accepts both formats:
    /// - Full preamble:   `<|start|>assistant<|channel|>commentary to=functions.NAME ...`
    /// - Channel-first:  `<|channel|>commentary to=functions.NAME ...`
    ///
    /// Channel-first inputs (e.g. from the vLLM wire shape, which strips the role
    /// announcement) are automatically prefixed with `<|start|>assistant` so the
    /// inner `StreamableParser` (in ExpectStart mode) can process them correctly.
    pub fn parse_tool_call_streaming_incremental(
        &mut self,
        delta_token_ids: &[u32],
    ) -> ToolStreamResult {
        // Prepend <|start|>assistant preamble when:
        //   (a) we're at the start of a new turn (between messages), AND
        //   (b) the chunk doesn't already carry the preamble (first token != <|start|>)
        // For all subsequent chunks within the same message, at_turn_start is false
        // so no preamble is added.
        let owned;
        let delta_token_ids = if !delta_token_ids.is_empty()
            && self.at_turn_start
            && delta_token_ids.first() != Some(&self.start_token)
        {
            self.at_turn_start = false;
            owned = self
                .preamble_tokens
                .iter()
                .chain(delta_token_ids.iter())
                .copied()
                .collect::<Vec<_>>();
            &owned
        } else {
            if !delta_token_ids.is_empty() {
                self.at_turn_start = false;
            }
            delta_token_ids
        };

        let mut chunks = Vec::new();
        for token in delta_token_ids {
            let prev = current_function_recipient(&self.parser);
            if let Err(e) = self.parser.process(*token) {
                tracing::warn!("harmony parse error for token {token}: {e}");
                break;
            }
            if *token == self.start_token {
                self.at_turn_start = false;
            }
            let recipient = current_function_recipient(&self.parser);

            match (&prev, &recipient) {
                // entered a new commentary->functions message
                (None, Some(_)) => {
                    self.in_tool_call = true;
                    self.header_emitted = false;
                }
                // left the tool-call message: advance index for the next call.
                // The stop token (<|call|>) also returns the inner parser to
                // ExpectStart, so the next message needs the preamble again.
                (Some(_), None) => {
                    if self.in_tool_call {
                        self.current_index += 1;
                    }
                    self.in_tool_call = false;
                    self.at_turn_start = true;
                }
                _ => {}
            }

            if let Some(name) = recipient {
                if !self.header_emitted {
                    let id = self.gen_id();
                    chunks.push(ToolCallResponseChunk {
                        index: self.current_index,
                        id: Some(id),
                        tp: Some(ToolCallType::Function),
                        function: Some(CalledFunctionStream {
                            name: Some(name),
                            arguments: None,
                        }),
                    });
                    self.header_emitted = true;
                }
                // The header (channel/recipient/constrain) is metadata, not content,
                // so `last_content_delta` only starts returning fragments once the
                // `<|message|>` body begins — i.e. the JSON arguments.
                if let Some(delta) = self.parser.last_content_delta().unwrap_or_default()
                    && !delta.is_empty()
                {
                    chunks.push(ToolCallResponseChunk {
                        index: self.current_index,
                        id: None,
                        tp: None,
                        function: Some(CalledFunctionStream {
                            name: None,
                            arguments: Some(delta),
                        }),
                    });
                }
            }
        }
        ToolStreamResult {
            tool_call_chunks: chunks,
        }
    }

    /// Stream EOF. For the text path, flush the held-back suffix (no more text is
    /// coming, so the final re-tokenization is authoritative), emitting its
    /// deltas. Then drive the parser to its terminal state. For the token path
    /// there's no text buffer, so this is just the terminal step.
    pub fn finish_tool_call_stream(&mut self) -> ToolStreamResult {
        let mut chunks = Vec::new();
        if !self.text_buffer.is_empty() {
            chunks.extend(self.flush_text_buffer(true).tool_call_chunks);
        }
        let _ = self.parser.process_eos();
        ToolStreamResult {
            tool_call_chunks: chunks,
        }
    }
}

/// `Some(name)` when the parser is currently inside a `commentary to=functions.NAME`
/// message, else `None`. (Analysis-directed calls are reasoning, not tool calls —
/// matching the batch harmony parser, which only extracts the commentary channel.)
fn current_function_recipient(parser: &StreamableParser) -> Option<String> {
    if parser.current_channel().as_deref() != Some("commentary") {
        return None;
    }
    parser
        .current_recipient()
        .and_then(|r| r.strip_prefix("functions.").map(|n| n.to_string()))
}

fn committable_text_tokens(text: &str) -> anyhow::Result<(Vec<u32>, usize)> {
    if text.len() <= TEXT_STREAM_HOLDBACK_BYTES {
        return Ok((Vec::new(), 0));
    }

    let max_commit_bytes = text.len() - TEXT_STREAM_HOLDBACK_BYTES;
    let tokens = encode_harmony(text)?;
    for token_count in (1..=tokens.len()).rev() {
        let token_prefix = &tokens[..token_count];
        let Ok(decoded_prefix) = decode_harmony_strict(token_prefix) else {
            continue;
        };
        if decoded_prefix.len() <= max_commit_bytes && text.starts_with(&decoded_prefix) {
            return Ok((token_prefix.to_vec(), decoded_prefix.len()));
        }
    }
    Ok((Vec::new(), 0))
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

// ----- token-based fixture schema (shared by the generator bin and the test) -----

/// One streaming chunk, carrying the gpt-oss `delta_token_ids` that drive the
/// parser plus the decoded `delta_text` for human readability.
#[derive(serde::Serialize, serde::Deserialize, Debug)]
pub struct TokenChunk {
    #[serde(default)]
    pub delta_text: String,
    pub delta_token_ids: Vec<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

#[derive(serde::Serialize, serde::Deserialize, Debug)]
pub struct ExpectedCall {
    pub name: String,
    /// Arguments as a JSON object (compared after canonicalizing both sides).
    pub arguments: serde_json::Value,
}

#[derive(serde::Serialize, serde::Deserialize, Debug)]
pub struct TokenFixture {
    pub family: String,
    pub model_label: String,
    pub description: String,
    pub chunks: Vec<TokenChunk>,
    pub expected_calls: Vec<ExpectedCall>,
}

#[cfg(test)]
mod tests {
    use super::*;

    // Canonical single tool call (TOOLCALLING.batch.1, harmony family).
    const CANON: &str = "<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{\"location\":\"NYC\"}<|call|>";

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
}
