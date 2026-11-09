// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::fmt::Debug;

use crate::ReasoningParser;
use crate::tool_calling::harmony::harmony_parser::{
    HARMONY_KNOWN_CHANNELS, HarmonyContentCursor, harmony_body_ending,
    harmony_commentary_is_payload, harmony_content_header_can_start, harmony_current_content_type,
    harmony_current_recipient, harmony_header_eof_is_metadata, harmony_header_is_directed,
    harmony_header_metadata, harmony_header_role, harmony_header_spans, harmony_initial_role,
    harmony_is_function_call, harmony_json_prelude, harmony_markers, harmony_message_is_visible,
    harmony_tool_headers, json_value_prefix_end, parse_harmony_header, process_harmony_token,
};
use crate::tool_calling::harmony::payload::{HarmonyPayload, HarmonyVisible};
use crate::{NormalTextSegment, ParserResult, ReasoningOutput};

use openai_harmony::StreamableParser;
use openai_harmony::chat::{Content, Message, TextContent};
use openai_harmony::{HarmonyEncoding, HarmonyEncodingName, chat::Role, load_harmony_encoding};
use regex::Regex;

use std::sync::OnceLock;

static GLOBAL_HARMONY_GPTOSS_ENCODING: OnceLock<Result<HarmonyEncoding, anyhow::Error>> =
    OnceLock::new();
static HARMONY_ACTION_TOKEN_IDS: OnceLock<Vec<u32>> = OnceLock::new();
static HARMONY_CALL_TOKEN_IDS: OnceLock<Vec<u32>> = OnceLock::new();
static HARMONY_START_TOKEN_ID: OnceLock<Option<u32>> = OnceLock::new();
static HARMONY_ANALYSIS_BLOCK_REGEX: OnceLock<Regex> = OnceLock::new();
const HARMONY_SPECIAL_TOKENS: &[&str] = &[
    "<|start|>",
    "<|end|>",
    "<|return|>",
    "<|channel|>",
    "<|constrain|>",
    "<|message|>",
    "<|call|>",
];

fn get_harmony_encoding() -> &'static Result<HarmonyEncoding, anyhow::Error> {
    GLOBAL_HARMONY_GPTOSS_ENCODING.get_or_init(|| {
        // load_harmony_encoding internally constructs a reqwest::blocking::Client,
        // which builds and drops a Tokio Runtime. Dropping a Runtime from inside
        // an async context (e.g. when this is called for the first time during
        // an HTTP request handler) panics with "Cannot drop a runtime in a
        // context where blocking is not allowed". Run the load on a fresh OS
        // thread so the inner Runtime is dropped outside any async context.
        // The init runs at most once per process via OnceLock.
        std::thread::spawn(|| load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss))
            .join()
            .unwrap_or_else(|_| Err(anyhow::anyhow!("harmony encoding loader thread panicked")))
    })
}

pub struct GptOssReasoningParser {
    parser: StreamableParser,
    start_token_id: u32,
    // Recovery persists across chunks until `<|start|>`.
    resynchronizing: bool,
    content_cursor: HarmonyContentCursor,
    pending_replacement: Option<PendingReplacement>,
    pending_text: String,
    emitted_reasoning_text: bool,
    insert_reasoning_separator: bool,
    emitted_normal_text: bool,
    insert_normal_separator: bool,
    /// Retained across chunks because action stops reset the native recipient
    /// before the downstream tool parser receives its complete envelope.
    last_directed_channel: Option<String>,
    last_directed_recipient: Option<String>,
    /// Set by the first `finish_reasoning_stream` call. `process_eos()` is not
    /// repeatable (it mutates harmony state) and the recovery below is not
    /// naturally idempotent, so a second finalize must return nothing rather
    /// than re-emit text the caller already has.
    parser_finished: bool,
    payload: HarmonyPayload,
    visible_payload: HarmonyVisible,
    pending_commentary: String,
    commentary_payload: Option<bool>,
    // Per-call ranges come only from native directed-envelope producers, never
    // from matching the returned text. Raw two-string APIs remain unchanged.
    protocol_ranges: Vec<std::ops::Range<usize>>,
}

struct PendingReplacement {
    text: String,
    body_prefix_len: usize,
}

/// Implement Debug for GptOssReasoningParser separately because StreamableParser does not implement Debug
impl Debug for GptOssReasoningParser {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GptOssReasoningParser")
            .field("parser", &self.parser.state_json())
            .finish()
    }
}

impl GptOssReasoningParser {
    fn contextual_output(&mut self, result: ParserResult) -> ReasoningOutput {
        let mut normal_segments = Vec::new();
        let mut cursor = 0;
        for range in std::mem::take(&mut self.protocol_ranges) {
            if result.normal_text.is_empty() {
                break;
            }
            if cursor < range.start {
                normal_segments.push(NormalTextSegment::Literal(
                    result.normal_text[cursor..range.start].to_string(),
                ));
            }
            normal_segments.push(NormalTextSegment::Protocol(
                result.normal_text[range.clone()].to_string(),
            ));
            cursor = range.end;
        }
        if cursor < result.normal_text.len() {
            normal_segments.push(NormalTextSegment::Literal(
                result.normal_text[cursor..].to_string(),
            ));
        }
        ReasoningOutput {
            reasoning_text: result.reasoning_text,
            normal_segments,
        }
    }

    pub fn new() -> anyhow::Result<Self> {
        let parser = match get_harmony_encoding().as_ref() {
            Ok(enc) => match StreamableParser::new(enc.clone(), Some(Role::Assistant)) {
                Ok(p) => p,
                Err(e) => {
                    tracing::warn!("Harmony StreamableParser init failed for GPT OSS: {e}");
                    return Err(anyhow::anyhow!(
                        "Failed to load Harmony StreamableParser: {e}"
                    ));
                }
            },
            Err(e) => {
                tracing::warn!("Failed to load Harmony encoding for GPT OSS: {e}");
                return Err(anyhow::anyhow!("Failed to load Harmony encoding: {e}"));
            }
        };
        let start_token_id = harmony_start_token_id()?;
        Ok(Self {
            parser,
            start_token_id,
            resynchronizing: false,
            content_cursor: HarmonyContentCursor::default(),
            pending_replacement: None,
            pending_text: String::new(),
            emitted_reasoning_text: false,
            insert_reasoning_separator: false,
            emitted_normal_text: false,
            insert_normal_separator: false,
            last_directed_channel: None,
            last_directed_recipient: None,
            parser_finished: false,
            payload: HarmonyPayload::default(),
            visible_payload: HarmonyVisible::default(),
            pending_commentary: String::new(),
            commentary_payload: None,
            protocol_ranges: Vec::new(),
        })
    }
}

fn encode_text_to_tokens(text: &str) -> anyhow::Result<Vec<u32>> {
    let enc = get_harmony_encoding()
        .as_ref()
        .map_err(|e| anyhow::anyhow!("Failed to get harmony encoding: {e}"))?;
    Ok(enc.tokenizer().encode_with_special_tokens(text))
}

fn harmony_start_token_id() -> anyhow::Result<u32> {
    HARMONY_START_TOKEN_ID
        .get_or_init(|| {
            let tokens = encode_text_to_tokens("<|start|>").ok()?;
            match tokens.as_slice() {
                [id] => Some(*id),
                _ => None,
            }
        })
        .ok_or_else(|| anyhow::anyhow!("Harmony start marker must encode to one token"))
}

fn harmony_action_token_ids() -> Option<&'static [u32]> {
    let ids = HARMONY_ACTION_TOKEN_IDS.get_or_init(|| {
        get_harmony_encoding()
            .as_ref()
            .ok()
            .and_then(|enc| enc.stop_tokens_for_assistant_actions().ok())
            .map(|ids| {
                let mut ids: Vec<_> = ids.into_iter().collect();
                ids.sort_unstable();
                ids
            })
            .unwrap_or_default()
    });
    if ids.is_empty() {
        None
    } else {
        Some(ids.as_slice())
    }
}

/// Token IDs for the legacy Harmony `<|call|>` terminator.
///
/// `<|call|>` also acts as a gpt-oss EOS token. Pipelines that hide EOS tokens from the
/// decoded output (e.g. the preprocessor's stop-condition handling) must keep
/// these ids visible when the harmony tool-call parser is active — otherwise the
/// terminator is stripped before the parser sees it and tool calls are silently
/// dropped. Empty if the harmony encoding is unavailable.
pub fn harmony_terminator_token_ids() -> Vec<u32> {
    HARMONY_CALL_TOKEN_IDS
        .get_or_init(|| encode_text_to_tokens("<|call|>").unwrap_or_default())
        .clone()
}

fn prepare_initial_harmony_message(
    parser: &mut StreamableParser,
    token_ids: &[u32],
    start_token_id: u32,
) -> anyhow::Result<()> {
    // Continuations inherit an assistant role; a fully framed first message supplies its own.
    if parser.tokens().is_empty() && harmony_initial_role(token_ids, start_token_id).is_none() {
        let encoding = get_harmony_encoding()
            .as_ref()
            .map_err(|error| anyhow::anyhow!("Harmony encoding unavailable: {error}"))?;
        *parser = StreamableParser::new(encoding.clone(), None)?;
    }
    Ok(())
}

fn harmony_analysis_block_regex() -> &'static Regex {
    HARMONY_ANALYSIS_BLOCK_REGEX.get_or_init(|| {
        Regex::new(r"(?:<\|start\|>assistant)?<\|channel\|>analysis<\|message\|>")
            .expect("harmony analysis block regex")
    })
}

fn guard_payload_token(
    payload: &mut HarmonyPayload,
    visible_payload: &mut HarmonyVisible,
    token_id: u32,
    is_visible: bool,
) -> anyhow::Result<Vec<u32>> {
    let enc = get_harmony_encoding()
        .as_ref()
        .map_err(|error| anyhow::anyhow!("Harmony encoding unavailable: {error}"))?;
    let bytes = enc.tokenizer().decode_bytes([token_id])?;
    if enc.tokenizer().is_special_token(token_id) {
        if if is_visible {
            visible_payload.in_string()
        } else {
            payload.in_string()
        } {
            // Both native parsing and envelope reconstruction must see literal data.
            if is_visible {
                visible_payload.push_bytes(&bytes);
            } else {
                payload.push_bytes(&bytes);
            }
            return Ok(enc
                .tokenizer()
                .encode_ordinary(std::str::from_utf8(&bytes)?));
        }
        if bytes == b"<|message|>" {
            payload.begin_body();
        } else {
            *payload = HarmonyPayload::default();
        }
        *visible_payload = HarmonyVisible::default();
    } else if is_visible {
        visible_payload.push_bytes(&bytes);
    } else {
        payload.push_bytes(&bytes);
    }
    Ok(vec![token_id])
}

fn raw_input_text_for_tool_parser(
    text: &str,
    token_ids: &[u32],
    caller_supplied_token_ids: bool,
) -> String {
    if caller_supplied_token_ids && let Ok(enc) = get_harmony_encoding() {
        return enc.tokenizer().decode_utf8(token_ids).unwrap_or_default();
    }

    text.to_string()
}

fn split_incomplete_harmony_suffix(text: &str) -> (&str, &str) {
    let hold_len = HARMONY_SPECIAL_TOKENS
        .iter()
        .flat_map(|token| (1..token.len()).map(|idx| &token[..idx]))
        .filter(|prefix| text.ends_with(*prefix))
        .map(str::len)
        .max()
        .unwrap_or(0);

    text.split_at(text.len() - hold_len)
}

fn contains_directed_harmony_function_call(text: &str) -> bool {
    get_harmony_encoding()
        .as_ref()
        .is_ok_and(|encoding| harmony_tool_headers(text, encoding).next().is_some())
        && harmony_markers(text).any(|(marker, in_json)| {
            !in_json && matches!(marker.as_str(), "<|call|>" | "<|return|>")
        })
}

fn strip_analysis_blocks_for_tool_handoff(text: &str) -> String {
    let mut analysis_headers = harmony_analysis_block_regex().find_iter(text).peekable();
    if analysis_headers.peek().is_none() {
        return text.to_string();
    }
    let Ok(encoding) = get_harmony_encoding() else {
        return text.to_string();
    };
    let tools: Vec<_> = harmony_tool_headers(text, encoding).collect();
    let frames: Vec<_> = harmony_header_spans(text, encoding, &tools)
        .into_iter()
        .filter(|span| text[span.clone()].starts_with("<|start|>"))
        .collect();
    let mut directed_headers = tools.iter().peekable();
    let mut markers = harmony_markers(text)
        .filter_map(|(marker, in_json)| (!in_json).then_some(marker))
        .peekable();
    let mut output = String::new();
    let mut cursor = 0;
    for header in analysis_headers {
        while directed_headers
            .peek()
            .is_some_and(|directed| directed.end <= header.start())
        {
            directed_headers.next();
        }
        // A recipient before the channel still makes this a directed analysis call.
        if directed_headers
            .peek()
            .is_some_and(|directed| directed.start <= header.start())
        {
            continue;
        }
        while markers
            .peek()
            .is_some_and(|marker| marker.start() < header.start())
        {
            markers.next();
        }
        if header.start() < cursor
            || markers
                .peek()
                .is_none_or(|marker| marker.start() != header.start())
        {
            continue;
        }
        output.push_str(&text[cursor..header.start()]);
        let next = frames
            .get(frames.partition_point(|frame| frame.start < header.end()))
            .map_or(text.len(), |frame| frame.start);
        cursor = next;
        while markers.peek().is_some_and(|marker| marker.start() < next) {
            let marker = markers.next().expect("peeked a Harmony marker");
            if marker.start() >= header.end()
                && matches!(marker.as_str(), "<|call|>" | "<|return|>" | "<|end|>")
            {
                cursor = marker.end();
                break;
            }
        }
    }
    output.push_str(&text[cursor..]);
    output
}

fn append_separated(target: &mut String, text: &str) {
    if text.is_empty() {
        return;
    }
    if !target.is_empty() {
        target.push('\n');
    }
    target.push_str(text);
}

fn append_text_content(target: &mut String, content: &[Content]) {
    for item in content {
        if let Content::Text(TextContent { text }) = item {
            append_separated(target, text);
        }
    }
}

fn append_message_by_channel(
    reasoning_text: &mut String,
    normal_text: &mut String,
    msg: &Message,
    has_call_stop: bool,
) {
    if msg.author.role != Role::Assistant {
        return;
    }
    let (recipient, content_type) =
        harmony_header_metadata(msg.recipient.as_deref(), msg.content_type.as_deref());
    if has_call_stop
        && msg.channel.as_deref() == Some("commentary")
        && recipient.is_none()
        && msg.content.iter().any(|content| match content {
            Content::Text(text) => harmony_commentary_is_payload(content_type, &text.text),
            _ => false,
        })
    {
        return;
    }
    match msg.channel.as_deref() {
        // Analysis WITHOUT a recipient is chain-of-thought reasoning.
        // Analysis WITH a functions recipient is a (malformed) tool call whose
        // payload is handed off via strip_analysis_blocks_for_tool_handoff —
        // do not also surface it as reasoning_content.
        Some("analysis") if recipient.is_none() => {
            append_text_content(reasoning_text, &msg.content)
        }
        Some("final") => append_text_content(normal_text, &msg.content),
        Some("commentary") if recipient.is_none() => append_text_content(normal_text, &msg.content),
        // A directed analysis/commentary payload is handed to the tool parser
        // elsewhere and must not also become content.
        Some("analysis") | Some("commentary") | None => {}
        // Unknown channel name: a malformed header harmony still parsed. Treat
        // the content as the answer rather than dropping it, matching what the
        // streaming path emits in place. A directed message
        // (`to=...`) is a tool envelope, not prose: drop it, as the
        // stranded-header path refuses `to=` headers.
        Some(_) if recipient.is_none() => append_text_content(normal_text, &msg.content),
        Some(_) => {}
    }
}

fn append_current_by_channel(
    reasoning_text: &mut String,
    normal_text: &mut String,
    role: Option<Role>,
    channel: Option<String>,
    recipient: Option<&str>,
    current: String,
) {
    if role != Some(Role::Assistant) {
        return;
    }
    match channel.as_deref() {
        Some("analysis") if recipient.is_none() => append_separated(reasoning_text, &current),
        Some("final") => append_separated(normal_text, &current),
        Some("commentary") if recipient.is_none() => append_separated(normal_text, &current),
        Some("analysis") | Some("commentary") | None => {}
        // See `append_message_by_channel`: an unrecognized channel is a
        // malformed header, not a reason to discard the model's output.
        Some(_) if recipient.is_none() => append_separated(normal_text, &current),
        Some(_) => {}
    }
}

fn is_visible_normal_channel(channel: Option<&str>, recipient: Option<&str>) -> bool {
    matches!(channel, Some("final"))
        || (channel.is_some_and(|channel| channel != "analysis") && recipient.is_none())
}

fn project_harmony_header(
    header: &StreamableParser,
    channel_header: &StreamableParser,
    envelope: String,
) -> Option<String> {
    if header.current_role() != Some(Role::Assistant) {
        return None;
    }
    let channel = header.current_channel()?;
    let recipient = harmony_current_recipient(header);
    let directed = harmony_is_function_call(Some(&channel), recipient.as_deref());
    if !directed && !is_visible_normal_channel(Some(&channel), recipient.as_deref()) {
        return None;
    }
    if harmony_current_recipient(channel_header) == recipient {
        return Some(envelope);
    }
    let channel_prefix = format!("<|channel|>{channel}");
    let (source_header, body) = envelope.split_once("<|message|>")?;
    let suffix = source_header.strip_prefix(&channel_prefix)?.trim();
    let mut projected = channel_prefix;
    if let Some(recipient) = recipient {
        projected.push_str(&format!(" to={recipient}"));
    }
    if !suffix.is_empty() {
        projected.push(' ');
        projected.push_str(suffix);
    } else if let Some(content_type) = harmony_current_content_type(header) {
        projected.push_str(&format!(" {content_type}"));
    }
    projected.push_str("<|message|>");
    projected.push_str(body);
    Some(projected)
}

/// Reconstruct a directed native envelope using its cached owning Message.
/// Channel-less calls retain their native header; channel-bearing calls reuse
/// the established projection. Neither path borrows a prior message's body.
fn reconstruct_directed_envelope(
    parser: &StreamableParser,
    channel: Option<&str>,
    recipient: Option<&str>,
    body_start: Option<usize>,
) -> Option<String> {
    if !harmony_is_function_call(channel, recipient) {
        return None;
    }
    let Ok(enc) = get_harmony_encoding() else {
        return None;
    };
    let tokens = parser.tokens();
    let channel_token_id = enc
        .tokenizer()
        .encode_with_special_tokens("<|channel|>")
        .into_iter()
        .next()?;
    let message_token = *enc
        .tokenizer()
        .encode_with_special_tokens("<|message|>")
        .first()?;
    let message_index = body_start?.checked_sub(1)?;
    if tokens.get(message_index) != Some(&message_token) {
        return None;
    }
    let start_token = harmony_start_token_id().ok()?;
    if channel.is_none() {
        let preceding = tokens[..message_index].iter().rposition(|token| {
            enc.tokenizer().is_special_token(*token)
                && enc
                    .tokenizer()
                    .decode_bytes([*token])
                    .is_ok_and(|bytes| bytes != b"<|constrain|>")
        });
        let begin = preceding.map_or(0, |index| {
            if tokens[index] == start_token {
                index
            } else {
                index + 1
            }
        });
        let header = parse_harmony_header(enc, &tokens[begin..=message_index], start_token)?;
        if header.current_role() != Some(Role::Assistant)
            || !harmony_is_function_call(
                header.current_channel().as_deref(),
                harmony_current_recipient(&header).as_deref(),
            )
        {
            return None;
        }
        return enc.tokenizer().decode_utf8(&tokens[begin..]).ok();
    }
    let last_channel_idx = tokens[..message_index]
        .iter()
        .rposition(|t| *t == channel_token_id)?;
    let previous_marker = tokens[..last_channel_idx].iter().rposition(|token| {
        enc.tokenizer().is_special_token(*token)
            && enc
                .tokenizer()
                .decode_utf8([*token])
                .is_ok_and(|marker| marker != "<|constrain|>")
    });
    let begin = previous_marker.unwrap_or(0);
    let envelope = enc
        .tokenizer()
        .decode_utf8(&tokens[last_channel_idx..])
        .ok()?;
    let channel_header =
        parse_harmony_header(enc, &tokens[last_channel_idx..=message_index], start_token)?;
    let preceding_marker =
        previous_marker.and_then(|index| enc.tokenizer().decode_utf8([tokens[index]]).ok());
    if preceding_marker.as_deref() != Some("<|message|>") {
        let header_start = previous_marker.map_or(0, |index| {
            if tokens[index] == start_token {
                index
            } else {
                index + 1
            }
        });
        if let Some(header) =
            parse_harmony_header(enc, &tokens[header_start..=message_index], start_token)
        {
            return project_harmony_header(&header, &channel_header, envelope);
        }
    } else {
        // Channel-first routing is complete here even when a prior malformed body
        // precedes it. A recipient before the channel still needs boundary recovery.
        if harmony_current_recipient(&channel_header)
            .as_deref()
            .is_some_and(|recipient| recipient.starts_with("functions."))
        {
            return project_harmony_header(&channel_header, &channel_header, envelope);
        }
    }
    let decoded = enc.tokenizer().decode_utf8(&tokens[begin..]).ok()?;
    let header = match harmony_tool_headers(&decoded, enc).last() {
        Some(header) => header,
        None => {
            // An interrupted directed message can leave native recipient metadata
            // behind a real final header; the jail still needs that final envelope.
            let latest = enc
                .tokenizer()
                .decode_utf8(&tokens[last_channel_idx..])
                .ok()?;
            let (latest_header, _) = latest.split_once("<|message|>")?;
            let prefix = enc
                .tokenizer()
                .decode_utf8(&tokens[begin..last_channel_idx])
                .ok()?;
            let prefix = if let Some(body) = prefix.strip_prefix("<|message|>") {
                &body[json_value_prefix_end(body).unwrap_or_default()..]
            } else {
                &prefix
            };
            let channel = latest_header
                .strip_prefix("<|channel|>")?
                .split(|ch: char| ch.is_whitespace() || ch == '<')
                .next()?;
            if channel == "final"
                && !harmony_header_is_directed(prefix)
                && !harmony_header_is_directed(latest_header)
            {
                tracing::warn!(
                    why = "harmony_interrupted_call_final_handoff",
                    recovered_bytes = latest.len(),
                    "Recovering a final message after an interrupted tool envelope"
                );
                return Some(latest);
            }
            return None;
        }
    };
    let start = header.start;
    let channel_start = start + decoded[start..].find("<|channel|>")?;
    let mut envelope = decoded[channel_start..].to_string();
    if header.recipient_first {
        let channel_end = "<|channel|>".len() + header.channel.as_ref()?.len();
        envelope.insert_str(channel_end, &format!(" to=functions.{}", header.name));
    }
    Some(envelope)
}

/// Role names harmony may leave at the head of an unterminated header.
const HARMONY_ROLE_NAMES: &[&str] = &["assistant", "system", "developer", "user", "tool"];

/// Strip harmony control markup so recovered text cannot leak raw protocol
/// tokens into assistant `content`.
fn strip_harmony_special_tokens(text: &str) -> String {
    let mut cleaned = String::new();
    let mut cursor = 0;
    for (marker, quoted) in harmony_markers(text) {
        if quoted {
            continue;
        }
        cleaned.push_str(&text[cursor..marker.start()]);
        cursor = marker.end();
    }
    cleaned.push_str(&text[cursor..]);
    cleaned
}

/// Whether `<|start|>` opened the stranded header: the token right before the
/// trailing `header_len` header tokens in the cumulative buffer. `<|start|>`
/// is not kept in `header_tokens`, so this is the only place it is visible.
fn header_opened_by_start(parser: &StreamableParser, header_len: usize) -> bool {
    let Ok(start_token_id) = harmony_start_token_id() else {
        return false;
    };
    let tokens = parser.tokens();
    tokens
        .len()
        .checked_sub(header_len + 1)
        .and_then(|idx| tokens.get(idx))
        .is_some_and(|t| *t == start_token_id)
}

fn stranded_header_is_bare(parser: &StreamableParser) -> bool {
    let Ok(encoding) = get_harmony_encoding() else {
        return false;
    };
    !parser.tokens().iter().any(|token| {
        encoding.tokenizer().is_special_token(*token)
            && encoding
                .tokenizer()
                .decode_bytes([*token])
                .is_ok_and(|bytes| matches!(bytes.as_slice(), b"<|start|>" | b"<|channel|>"))
    })
}

/// Recover the message a malformed harmony header stranded in `StreamState::Header`.
///
/// When a gpt-oss model emits `<|channel|>final` followed by anything other than
/// `<|message|>` — the observed shape is a stray token where `<|message|>` was
/// expected — `StreamableParser` never leaves `Header`. Every later token is
/// pushed onto `header_tokens` instead of becoming a content delta, and
/// `process_eos()` then fails with "Unexpected EOS while waiting for message
/// header to complete". The model's entire final message is in that buffer and
/// is otherwise dropped, which is what surfaces to the client as
/// `content: null` with `finish_reason: "stop"`.
///
/// Decoding is deliberately scoped to the header tokens alone, read back from
/// `state_json()`, rather than to `parser.tokens()`: the cumulative buffer also
/// holds every message already streamed to the caller, so recovering from it
/// would re-emit them. Header tokens have by definition never been emitted, so
/// this cannot double-emit.
///
/// The recovered text keeps the channel the model did manage to name: a
/// stranded `analysis` header is reasoning and must not become the answer.
/// Everything else is attributed to `final`, as vLLM's `harmony.py::flush()`
/// does, so the same generation is reported identically whichever frontend
/// served it.
///
/// Returns `None` when the parser is not stranded, when the state cannot be
/// read, when nothing but markup remains, or when the stranded header is a
/// directed tool call (`to=...`): its payload is a tool envelope, not prose,
/// and surfacing it as `content` would leak the recipient and raw arguments.
pub(crate) fn recover_stranded_header_text(parser: &StreamableParser) -> Option<ParserResult> {
    if parser
        .current_role()
        .is_some_and(|role| role != Role::Assistant)
    {
        return None;
    }
    #[derive(serde::Deserialize)]
    #[serde(tag = "state")]
    enum HarmonyStreamState {
        Header {
            header_tokens: Vec<u32>,
        },
        #[serde(other)]
        Other,
    }

    let state_json = parser.state_json().ok()?;
    let HarmonyStreamState::Header { header_tokens } =
        serde_json::from_str::<HarmonyStreamState>(&state_json).ok()?
    else {
        return None;
    };
    if header_tokens.is_empty() {
        return None;
    }

    let enc = get_harmony_encoding().as_ref().ok()?;
    let decoded = enc.tokenizer().decode_utf8(&header_tokens).ok()?;
    if !header_opened_by_start(parser, header_tokens.len())
        && harmony_json_prelude(&decoded)
        && harmony_markers(&decoded).all(|(_, quoted)| quoted)
    {
        return Some(ParserResult {
            normal_text: decoded,
            reasoning_text: String::new(),
        });
    }
    let source_header = if header_opened_by_start(parser, header_tokens.len()) {
        format!("<|start|>{decoded}")
    } else {
        decoded.clone()
    };
    if harmony_header_eof_is_metadata(enc, &source_header) {
        return None;
    }
    if header_opened_by_start(parser, header_tokens.len())
        && harmony_header_role(enc, &format!("<|start|>{decoded}")) != Some(Role::Assistant)
    {
        return None;
    }

    // `<|start|>assistant<|channel|>final*)__Hello!` arrives here as
    // `assistant<|channel|>final*)__Hello!`, because `<|start|>` is what moved
    // harmony into `Header` and is not itself retained.
    let (role_part, channel, remainder) = match decoded.rfind("<|channel|>") {
        Some(idx) => {
            let after = &decoded[idx + "<|channel|>".len()..];
            // Drop the channel name the model did manage to emit. Without a
            // `<|message|>` delimiter there is no separator to split on, so only
            // an exact known-channel prefix is removed; anything else is left in
            // place rather than guessed at and truncated.
            let (channel, remainder) = HARMONY_KNOWN_CHANNELS
                .iter()
                .find_map(|channel| Some((Some(*channel), after.strip_prefix(channel)?)))
                .unwrap_or((None, after));
            (&decoded[..idx], channel, remainder)
        }
        // No channel marker at all: the header never got past the role name.
        // A role word is header markup only when `<|start|>` opened this
        // header. A parser constructed with a role starts in `Header` without
        // one, so plain output such as `tools are ...` lands here as well and
        // must not lose its leading word.
        None => {
            let remainder = if header_opened_by_start(parser, header_tokens.len()) {
                HARMONY_ROLE_NAMES
                    .iter()
                    .find_map(|role| decoded.strip_prefix(role))
                    .unwrap_or(decoded.as_str())
            } else {
                decoded.as_str()
            };
            ("", None, remainder)
        }
    };

    // Harmony allows the recipient either after the role or after the channel.
    if role_part.contains(" to=") || remainder.trim_start().starts_with("to=") {
        return None;
    }

    let recovered = strip_harmony_special_tokens(remainder);
    if recovered.trim().is_empty() {
        return None;
    }
    Some(if channel == Some("analysis") {
        ParserResult {
            normal_text: String::new(),
            reasoning_text: recovered,
        }
    } else {
        ParserResult {
            normal_text: recovered,
            reasoning_text: String::new(),
        }
    })
}

fn recover_replaced_header_text(parser: &StreamableParser) -> Option<ParserResult> {
    recover_stranded_header_text(parser).filter(|recovered| {
        recovered.normal_text.trim() != "assistant" || !recovered.reasoning_text.is_empty()
    })
}

impl ReasoningParser for GptOssReasoningParser {
    fn detect_and_parse_reasoning_with_context(
        &mut self,
        text: &str,
        token_ids: &[u32],
    ) -> ReasoningOutput {
        *self =
            Self::new().expect("this parser's cached Harmony encoding initialized successfully");
        let mut output = self.parse_reasoning_streaming_with_context(text, token_ids);
        let tail = self.finish_reasoning_stream_with_context();
        output.normal_segments.extend(tail.normal_segments);
        output.reasoning_text.push_str(&tail.reasoning_text);
        output
    }

    fn parse_reasoning_streaming_with_context(
        &mut self,
        text: &str,
        token_ids: &[u32],
    ) -> ReasoningOutput {
        let result = self.parse_reasoning_streaming_incremental(text, token_ids);
        self.contextual_output(result)
    }

    fn finish_reasoning_stream_with_context(&mut self) -> ReasoningOutput {
        let result = self.finish_reasoning_stream();
        self.contextual_output(result)
    }

    /// Finalize the stream, recovering anything harmony is still holding.
    ///
    /// Idempotent: a second call returns an empty result. The first call
    /// consumes `process_eos()`, which cannot be repeated.
    fn finish_reasoning_stream(&mut self) -> ParserResult {
        self.protocol_ranges.clear();
        let incomplete_why = if self.payload.in_string() {
            "harmony_unclosed_json_string_at_eof"
        } else {
            "harmony_unconfirmed_header_at_eof"
        };
        let directed_tool_pending = self.last_directed_recipient.is_some();
        let incomplete_tool = self
            .last_directed_recipient
            .is_some()
            .then(|| {
                reconstruct_directed_envelope(
                    &self.parser,
                    self.last_directed_channel.as_deref(),
                    self.last_directed_recipient.as_deref(),
                    self.content_cursor.body_start(),
                )
            })
            .flatten()
            .filter(|envelope| contains_directed_harmony_function_call(envelope));
        self.last_directed_channel = None;
        self.last_directed_recipient = None;
        if self.parser_finished {
            return ParserResult::default();
        }
        self.parser_finished = true;
        self.content_cursor = HarmonyContentCursor::default();
        self.payload = HarmonyPayload::default();
        self.visible_payload = HarmonyVisible::default();

        // Without its message delimiter, a candidate header still belongs to
        // the old body; EOF must not assign its bytes to the proposed channel.
        let marker_tail = std::mem::take(&mut self.pending_text);
        let mut pending = self
            .pending_replacement
            .take()
            .map(|pending| pending.text)
            .unwrap_or_default();
        pending.push_str(&marker_tail);
        if !directed_tool_pending && !pending.is_empty() {
            let encoding = get_harmony_encoding()
                .as_ref()
                .expect("parser encoding was initialized");
            let ending = harmony_body_ending(&pending, encoding);
            let has_call = ending
                .as_ref()
                .is_some_and(|ending| ending.as_str() == "<|call|>");
            let keep = ending.map_or(pending.len(), |ending| ending.start());
            pending.truncate(keep);
            if self.parser.current_channel().as_deref() == Some("commentary")
                && !harmony_message_is_visible(
                    Some("commentary"),
                    harmony_current_recipient(&self.parser).as_deref(),
                    has_call,
                    harmony_commentary_is_payload(
                        harmony_current_content_type(&self.parser).as_deref(),
                        &self
                            .parser
                            .current_content()
                            .expect("native content is decoded UTF-8"),
                    ),
                )
            {
                pending.clear();
                self.pending_commentary.clear();
            }
        }
        let incomplete_tool = incomplete_tool.map(|mut envelope| {
            envelope.push_str(&marker_tail);
            envelope
        });
        let mut result = ParserResult::default();
        if !self.pending_commentary.is_empty() {
            self.pending_commentary.push_str(&pending);
            if self.insert_normal_separator {
                result.normal_text.push('\n');
                self.insert_normal_separator = false;
            }
            result
                .normal_text
                .push_str(&std::mem::take(&mut self.pending_commentary));
            self.emitted_normal_text = true;
        } else if !pending.is_empty() && !directed_tool_pending {
            append_current_by_channel(
                &mut result.reasoning_text,
                &mut result.normal_text,
                self.parser.current_role(),
                self.parser.current_channel(),
                harmony_current_recipient(&self.parser).as_deref(),
                pending,
            );
        }

        // Finalize harmony itself. On a well-formed stream `process_eos()`
        // merely closes the message whose content was already streamed token by
        // token, so there is deliberately nothing to collect on the `Ok` path —
        // re-reading `messages()` here would duplicate text the caller already
        // received. The error path is the one that matters: it means harmony
        // ended in `Header`, holding a message it never handed over.
        if self.parser.process_eos().is_err()
            && let Some(recovered) = recover_stranded_header_text(&self.parser)
        {
            tracing::warn!(
                "Harmony parser ended in a non-terminal state; returning the recovered raw output."
            );
            if !recovered.normal_text.is_empty() {
                if self.emitted_normal_text || !result.normal_text.is_empty() {
                    result.normal_text.push('\n');
                }
                result.normal_text.push_str(&recovered.normal_text);
                self.emitted_normal_text = true;
            }
            if !recovered.reasoning_text.is_empty() {
                if self.emitted_reasoning_text || !result.reasoning_text.is_empty() {
                    result.reasoning_text.push('\n');
                }
                result.reasoning_text.push_str(&recovered.reasoning_text);
                self.emitted_reasoning_text = true;
            }
        }
        if let Some(envelope) = incomplete_tool {
            tracing::warn!(
                why = incomplete_why,
                recovered_bytes = envelope.len(),
                "Deferring unresolved tool framing to the tool parser's EOF recovery"
            );
            let start = result.normal_text.len();
            result.normal_text.push_str(&envelope);
            self.protocol_ranges.push(start..result.normal_text.len());
        }

        result
    }

    fn detect_and_parse_reasoning(&mut self, text: &str, token_ids: &[u32]) -> ParserResult {
        self.protocol_ranges.clear();
        let caller_supplied_token_ids = !token_ids.is_empty();
        let encoded_tokens;
        let token_ids = if token_ids.is_empty() {
            // WAR: Since we are moving to just text based reasoning parsing, converting to token_ids now using harmony encoding
            encoded_tokens = match encode_text_to_tokens(text) {
                Ok(tokens) => tokens,
                Err(err) => {
                    tracing::warn!("Failed to encode Harmony tokens: {err}");
                    return ParserResult::default();
                }
            };
            encoded_tokens.as_slice()
        } else {
            token_ids
        };

        let parser = &mut self.parser;
        if let Err(error) = prepare_initial_harmony_message(parser, token_ids, self.start_token_id)
        {
            tracing::warn!(why = "harmony_initial_message_unavailable", %error, "Could not initialize the Harmony message");
            return ParserResult::default();
        }

        let mut retained = Vec::new();
        let mut leading_normal_text = String::new();
        let mut call_ended_messages = Vec::new();
        let encoding = get_harmony_encoding()
            .as_ref()
            .expect("parser encoding was initialized");
        for (i, token_id) in token_ids.iter().enumerate() {
            if self.resynchronizing && *token_id != self.start_token_id {
                continue;
            }
            tracing::debug!(
                "Processing token {} of {}: {}",
                i + 1,
                token_ids.len(),
                token_id
            );
            let guarded = match guard_payload_token(
                &mut self.payload,
                &mut self.visible_payload,
                *token_id,
                harmony_current_recipient(parser).is_none(),
            ) {
                Ok(guarded) => guarded,
                Err(error) => {
                    tracing::warn!(why = "harmony_payload_token_error", %error, "Could not decode Harmony payload token");
                    return ParserResult::default();
                }
            };
            for token_id in guarded {
                match process_harmony_token(
                    parser,
                    encoding,
                    &mut self.resynchronizing,
                    &mut self.content_cursor,
                    token_id,
                    self.start_token_id,
                    |parser, intentional| {
                        retained.extend_from_slice(parser.messages());
                        if intentional && let Some(recovered) = recover_replaced_header_text(parser)
                        {
                            let bare = stranded_header_is_bare(parser);
                            for (channel, text) in [
                                ("final", recovered.normal_text),
                                ("analysis", recovered.reasoning_text),
                            ] {
                                if !text.is_empty() {
                                    if bare && channel == "final" {
                                        leading_normal_text.push_str(&text);
                                        continue;
                                    }
                                    let mut message =
                                        Message::from_role_and_content(Role::Assistant, text);
                                    message.channel = Some(channel.to_string());
                                    retained.push(message);
                                }
                            }
                            call_ended_messages.resize(retained.len(), false);
                        }
                        Ok(())
                    },
                ) {
                    Ok(true) => {}
                    result => {
                        self.payload = HarmonyPayload::default();
                        self.visible_payload = HarmonyVisible::default();
                        if let Err(error) = result {
                            tracing::warn!(why = "harmony_resync_unavailable", %error, "Could not recover Harmony parser");
                        }
                    }
                }
                if retained.len() + parser.messages().len() > call_ended_messages.len() {
                    call_ended_messages.push(
                        encoding
                            .tokenizer()
                            .decode_bytes([token_id])
                            .is_ok_and(|bytes| bytes == b"<|call|>"),
                    );
                }
            }
        }

        let output_msgs = parser.messages();
        tracing::debug!("Parser has {} output messages", output_msgs.len());

        let mut reasoning_text = String::new();
        let mut normal_text = String::new();
        for (index, msg) in retained.iter().chain(output_msgs).enumerate() {
            append_message_by_channel(
                &mut reasoning_text,
                &mut normal_text,
                msg,
                call_ended_messages[index],
            );
        }

        let current = parser.current_content().unwrap_or_default();
        let ending = self
            .content_cursor
            .is_pending()
            .then(|| harmony_body_ending(&current, encoding))
            .flatten();
        let has_call = ending
            .as_ref()
            .is_some_and(|ending| ending.as_str() == "<|call|>");
        let current = current[..ending.map_or(current.len(), |ending| ending.start())].to_string();
        let current_channel = parser.current_channel();
        let current_recipient = harmony_current_recipient(parser);
        if current_channel.as_deref() != Some("commentary")
            || harmony_message_is_visible(
                Some("commentary"),
                current_recipient.as_deref(),
                has_call,
                harmony_commentary_is_payload(
                    harmony_current_content_type(parser).as_deref(),
                    &current,
                ),
            )
        {
            append_current_by_channel(
                &mut reasoning_text,
                &mut normal_text,
                parser.current_role(),
                current_channel,
                current_recipient.as_deref(),
                current,
            );
        }

        let raw_input_text =
            raw_input_text_for_tool_parser(text, token_ids, caller_supplied_token_ids);
        normal_text.insert_str(0, &leading_normal_text);
        if contains_directed_harmony_function_call(&raw_input_text) {
            // The serving pipeline feeds reasoning normal_text into the Harmony
            // tool parser. Keep directed commentary envelopes intact so tool
            // calls are not silently dropped.
            normal_text = strip_analysis_blocks_for_tool_handoff(&raw_input_text);
        } else if let Some(recovered) = recover_stranded_header_text(parser) {
            // Batch parity with the streaming path: a header the model never
            // closed with `<|message|>` leaves harmony in `Header` holding the
            // whole final message, which neither `messages()` nor
            // `current_content()` exposes. Skipped on the raw handoff above,
            // which already carries the stranded header verbatim.
            tracing::warn!(
                "Harmony parser ended in a non-terminal state; returning the recovered raw output."
            );
            append_separated(&mut normal_text, &recovered.normal_text);
            append_separated(&mut reasoning_text, &recovered.reasoning_text);
        }

        tracing::debug!(
            "Final result - normal_text: {} chars, reasoning_text: {} chars",
            normal_text.len(),
            reasoning_text.len()
        );

        ParserResult {
            normal_text,
            reasoning_text,
        }
    }

    fn parse_reasoning_streaming_incremental(
        &mut self,
        text: &str,
        token_ids: &[u32],
    ) -> ParserResult {
        self.protocol_ranges.clear();
        let caller_supplied_token_ids = !token_ids.is_empty();
        let text_to_process;
        let text = if caller_supplied_token_ids {
            text
        } else {
            self.pending_text.push_str(text);
            let combined = std::mem::take(&mut self.pending_text);
            let (ready, pending) = split_incomplete_harmony_suffix(&combined);
            self.pending_text.push_str(pending);
            text_to_process = ready.to_string();
            text_to_process.as_str()
        };

        if text.is_empty() && token_ids.is_empty() {
            return ParserResult::default();
        }

        let encoded_tokens;
        let token_ids = if token_ids.is_empty() {
            // WAR: Since we are moving to just text based reasoning parsing, converting to token_ids now using harmony encoding
            encoded_tokens = match encode_text_to_tokens(text) {
                Ok(tokens) => tokens,
                Err(err) => {
                    tracing::warn!("Failed to encode Harmony tokens: {err}");
                    return ParserResult::default();
                }
            };
            encoded_tokens.as_slice()
        } else {
            token_ids
        };

        let parser: &mut StreamableParser = &mut self.parser;
        if let Err(error) = prepare_initial_harmony_message(parser, token_ids, self.start_token_id)
        {
            tracing::warn!(why = "harmony_initial_message_unavailable", %error, "Could not initialize the Harmony message");
            return ParserResult::default();
        }
        let mut normal_delta = String::new();
        let mut reasoning_delta = String::new();
        for (i, original_token) in token_ids.iter().enumerate() {
            // Wait for a message boundary before trusting channel or recipient.
            if self.resynchronizing && *original_token != self.start_token_id {
                continue;
            }
            let guarded = match guard_payload_token(
                &mut self.payload,
                &mut self.visible_payload,
                *original_token,
                harmony_current_recipient(parser).is_none(),
            ) {
                Ok(guarded) => guarded,
                Err(error) => {
                    tracing::warn!(why = "harmony_payload_token_error", %error, "Could not decode Harmony payload token");
                    return ParserResult::default();
                }
            };
            for token_id in &guarded {
                tracing::debug!(
                    "Processing streaming token {} of {}: {}",
                    i + 1,
                    token_ids.len(),
                    token_id
                );
                let previous_role = parser.current_role();
                let previous_channel = parser.current_channel();
                let previous_recipient = harmony_current_recipient(parser);
                let previous_body_start = self.content_cursor.body_start();
                let encoding = get_harmony_encoding()
                    .as_ref()
                    .expect("parser encoding was initialized");
                let starts_header = *token_id == self.start_token_id
                    || (encoding.tokenizer().is_special_token(*token_id)
                        && encoding
                            .tokenizer()
                            .decode_bytes([*token_id])
                            .is_ok_and(|bytes| bytes == b"<|channel|>"));
                let pending_prefix_len = if self.pending_replacement.is_none()
                    && self.content_cursor.in_body()
                    && starts_header
                {
                    match harmony_content_header_can_start(parser, *token_id == self.start_token_id)
                        .and_then(|eligible| {
                            if eligible {
                                parser.current_content().map(|text| Some(text.len()))
                            } else {
                                Ok(None)
                            }
                        }) {
                        Ok(prefix) => prefix,
                        Err(error) => {
                            tracing::warn!(why = "harmony_pending_header_error", %error, "Could not inspect a pending Harmony header");
                            return ParserResult::default();
                        }
                    }
                } else {
                    None
                };
                let mut intentional_reset = false;
                let mut recovered_header = None;
                let mut bare_header = false;
                let mut retained_prefix_len = None;
                match process_harmony_token(
                    parser,
                    get_harmony_encoding()
                        .as_ref()
                        .expect("parser encoding was initialized"),
                    &mut self.resynchronizing,
                    &mut self.content_cursor,
                    *token_id,
                    self.start_token_id,
                    |old, intentional| {
                        intentional_reset = intentional;
                        if intentional {
                            recovered_header = recover_replaced_header_text(old);
                            bare_header = stranded_header_is_bare(old);
                            retained_prefix_len = old
                                .messages()
                                .last()
                                .and_then(|message| message.content.last())
                                .and_then(|content| match content {
                                    Content::Text(text) => Some(text.text.len()),
                                    _ => None,
                                });
                        }
                        Ok(())
                    },
                ) {
                    Ok(true) => {}
                    result => {
                        if let Err(error) = result {
                            tracing::warn!(why = "harmony_resync_unavailable", %error, "Could not recover Harmony parser");
                        }
                        self.last_directed_channel = None;
                        self.last_directed_recipient = None;
                        self.payload = HarmonyPayload::default();
                        self.visible_payload = HarmonyVisible::default();
                        self.pending_commentary.clear();
                        self.commentary_payload = None;
                        self.pending_replacement = None;
                        continue;
                    }
                }
                if intentional_reset {
                    self.last_directed_channel = None;
                    self.last_directed_recipient = None;
                }
                if self.pending_replacement.is_none()
                    && self.content_cursor.is_pending()
                    && let Some(body_prefix_len) = pending_prefix_len
                {
                    self.pending_replacement = Some(PendingReplacement {
                        text: String::new(),
                        body_prefix_len,
                    });
                }
                if self.pending_replacement.is_some()
                    && (intentional_reset || !self.content_cursor.is_pending())
                {
                    let held = self
                        .pending_replacement
                        .take()
                        .expect("checked a pending header");
                    let retained = if intentional_reset {
                        retained_prefix_len
                            .unwrap_or(held.body_prefix_len)
                            .saturating_sub(held.body_prefix_len)
                    } else {
                        held.text.len()
                    };
                    let text = held
                        .text
                        .get(..retained)
                        .expect("native prefix consists of complete UTF-8 content deltas")
                        .to_string();
                    if previous_channel.as_deref() == Some("commentary")
                        && previous_recipient.is_none()
                        && self.commentary_payload != Some(false)
                    {
                        self.pending_commentary.push_str(&text);
                    } else {
                        let mut released = ParserResult::default();
                        append_current_by_channel(
                            &mut released.reasoning_text,
                            &mut released.normal_text,
                            previous_role.clone(),
                            previous_channel.clone(),
                            previous_recipient.as_deref(),
                            text,
                        );
                        self.emitted_normal_text |= !released.normal_text.is_empty();
                        self.emitted_reasoning_text |= !released.reasoning_text.is_empty();
                        normal_delta.push_str(&released.normal_text);
                        reasoning_delta.push_str(&released.reasoning_text);
                    }
                }
                if let Some(recovered) = recovered_header {
                    if !recovered.normal_text.is_empty() {
                        if self.emitted_normal_text {
                            normal_delta.push('\n');
                        }
                        normal_delta.push_str(&recovered.normal_text);
                        self.emitted_normal_text |= !bare_header;
                        self.insert_normal_separator = false;
                    }
                    if !recovered.reasoning_text.is_empty() {
                        if self.emitted_reasoning_text {
                            reasoning_delta.push('\n');
                        }
                        reasoning_delta.push_str(&recovered.reasoning_text);
                        self.emitted_reasoning_text = true;
                        self.insert_reasoning_separator = false;
                    }
                }
                let current_channel = parser.current_channel();
                let current_recipient = harmony_current_recipient(parser);
                let content_delta = match parser.last_content_delta() {
                    Ok(delta) => delta,
                    Err(error) => {
                        tracing::warn!(why = "harmony_content_delta_error", %error, "Could not read the decoded Harmony delta");
                        return ParserResult::default();
                    }
                };
                if self.content_cursor.is_pending()
                    && let Some(held) = &mut self.pending_replacement
                {
                    if encoding.tokenizer().is_special_token(*token_id) {
                        held.text.push_str(
                            &encoding
                                .tokenizer()
                                .decode_utf8([*token_id])
                                .expect("Harmony control tokens have UTF-8 literals"),
                        );
                    } else if let Some(delta) = content_delta {
                        held.text.push_str(&delta);
                    }
                    continue;
                }
                if previous_role == Some(Role::Assistant)
                    && previous_channel.as_deref() == Some("commentary")
                    && previous_recipient.is_none()
                    && (current_channel.is_none() || intentional_reset)
                {
                    let call_stop = harmony_terminator_token_ids().contains(token_id);
                    if !(self.pending_commentary.is_empty()
                        || call_stop && self.commentary_payload == Some(true))
                    {
                        if self.insert_normal_separator {
                            normal_delta.push('\n');
                            self.insert_normal_separator = false;
                        }
                        normal_delta.push_str(&self.pending_commentary);
                        self.emitted_normal_text = true;
                    }
                    self.pending_commentary.clear();
                    self.commentary_payload = None;
                }

                if parser.current_role() != Some(Role::Assistant)
                    && previous_role != Some(Role::Assistant)
                {
                    self.last_directed_channel = None;
                    self.last_directed_recipient = None;
                    continue;
                }

                if harmony_is_function_call(
                    current_channel.as_deref(),
                    current_recipient.as_deref(),
                ) {
                    self.last_directed_channel = current_channel.clone();
                    self.last_directed_recipient = current_recipient.clone();
                }

                // Reconstruct at the action token: a later message in this same
                // chunk would replace the last channel in the cumulative buffer.
                if self.last_directed_recipient.is_some()
                    && previous_role == Some(Role::Assistant)
                    && parser.current_role().is_none()
                {
                    if harmony_action_token_ids().is_some_and(|stops| stops.contains(token_id))
                        && let Some(env) = reconstruct_directed_envelope(
                            parser,
                            self.last_directed_channel.as_deref(),
                            self.last_directed_recipient.as_deref(),
                            previous_body_start,
                        )
                    {
                        let start = normal_delta.len();
                        normal_delta.push_str(&env);
                        self.protocol_ranges.push(start..normal_delta.len());
                    }
                    self.last_directed_channel = None;
                    self.last_directed_recipient = None;
                }

                if parser.current_role() != Some(Role::Assistant) {
                    continue;
                }

                if intentional_reset {
                    if self.emitted_normal_text
                        && is_visible_normal_channel(
                            current_channel.as_deref(),
                            current_recipient.as_deref(),
                        )
                    {
                        self.insert_normal_separator = true;
                    }
                    if self.emitted_reasoning_text
                        && current_channel.as_deref() == Some("analysis")
                        && current_recipient.is_none()
                    {
                        self.insert_reasoning_separator = true;
                    }
                }

                if previous_channel.as_deref() != Some("analysis")
                    && current_channel.as_deref() == Some("analysis")
                    && self.emitted_reasoning_text
                {
                    self.insert_reasoning_separator = true;
                }

                if is_visible_normal_channel(
                    current_channel.as_deref(),
                    current_recipient.as_deref(),
                ) && !is_visible_normal_channel(
                    previous_channel.as_deref(),
                    previous_recipient.as_deref(),
                ) && self.emitted_normal_text
                {
                    self.insert_normal_separator = true;
                }

                if let (Some(delta), Some(channel)) = (content_delta, current_channel) {
                    match channel.as_str() {
                        "final" => {
                            if self.insert_normal_separator {
                                normal_delta.push('\n');
                                self.insert_normal_separator = false;
                            }
                            normal_delta.push_str(&delta);
                            self.emitted_normal_text = true;
                        }
                        // Analysis WITHOUT a recipient is reasoning content.
                        // Analysis WITH a functions recipient is a (malformed)
                        // tool call; its complete envelope is handed off at the action stop.
                        "analysis" if current_recipient.is_none() => {
                            if self.insert_reasoning_separator {
                                reasoning_delta.push('\n');
                                self.insert_reasoning_separator = false;
                            }
                            reasoning_delta.push_str(&delta);
                            self.emitted_reasoning_text = true;
                        }
                        "commentary" if current_recipient.is_none() => {
                            self.pending_commentary.push_str(&delta);
                            if self.commentary_payload.is_none() {
                                let content_type = harmony_current_content_type(parser);
                                if harmony_commentary_is_payload(content_type.as_deref(), &delta) {
                                    self.commentary_payload = Some(true);
                                } else if !delta.trim().is_empty() {
                                    self.commentary_payload = Some(false);
                                }
                            }
                            if self.commentary_payload != Some(false) {
                                continue;
                            }
                            if self.insert_normal_separator {
                                normal_delta.push('\n');
                                self.insert_normal_separator = false;
                            }
                            normal_delta.push_str(&std::mem::take(&mut self.pending_commentary));
                            self.emitted_normal_text = true;
                        }
                        // `analysis`/`commentary` WITH a recipient is a directed
                        // tool call; its payload is reconstructed as a whole
                        // envelope on the action-stop chunk, so dropping the delta
                        // here is correct.
                        "analysis" | "commentary" => {}
                        _ if current_recipient.is_some() => {}
                        _ => {
                            if self.insert_normal_separator {
                                normal_delta.push('\n');
                                self.insert_normal_separator = false;
                            }
                            normal_delta.push_str(&delta);
                            self.emitted_normal_text = true;
                        }
                    }
                }
            }
        }

        if !normal_delta.is_empty() || !reasoning_delta.is_empty() {
            tracing::debug!(
                "Returning aggregated deltas: normal: {} chars, reasoning: {} chars",
                normal_delta.len(),
                reasoning_delta.len()
            );
            return ParserResult {
                normal_text: normal_delta,
                reasoning_text: reasoning_delta,
            };
        }

        tracing::debug!("No deltas to return, returning empty result");
        ParserResult::default()
    }
}

#[cfg(test)]
mod tests {
    fn collect_context_chunks(
        chunks: impl IntoIterator<Item = (String, Vec<u32>)>,
    ) -> ReasoningOutput {
        let mut parser = GptOssReasoningParser::new().unwrap();
        let mut output = ReasoningOutput::default();
        for (text, tokens) in chunks {
            let part = parser.parse_reasoning_streaming_with_context(&text, &tokens);
            assert!(
                part.normal_segments.is_empty(),
                "an unfinished quoted payload cannot be published before EOF"
            );
            output.reasoning_text.push_str(&part.reasoning_text);
            output.normal_segments.extend(part.normal_segments);
        }
        let tail = parser.finish_reasoning_stream_with_context();
        output.reasoning_text.push_str(&tail.reasoning_text);
        output.normal_segments.extend(tail.normal_segments);
        assert!(
            parser
                .finish_reasoning_stream_with_context()
                .normal_segments
                .is_empty()
        );
        output
    }

    #[tokio::test]
    async fn optional_channel_truncated_quoted_arguments_recover_only_at_eof() {
        use crate::tool_calling::parsers::{
            detect_and_parse_tool_call_with_context, detect_and_parse_tool_call_with_recovery,
        };
        let arguments = r#"{"q":"abc"#;
        for channel in ["", "<|channel|>commentary", "<|channel|>analysis"] {
            for ending in ["<|call|>", "<|return|>", ""] {
                let input = format!(
                    "<|start|>assistant to=functions.lookup {channel}<|constrain|>json<|message|>{arguments}{ending}<|start|>assistant<|channel|>final<|message|>Answer<|return|>"
                );
                let expected_calls = usize::from(!ending.is_empty());
                let direct = crate::tool_calling::harmony::harmony_parser::parse_tool_calls_harmony_complete(&input, &Default::default(), None).await.unwrap();
                assert_eq!(direct.0.len(), expected_calls, "{input}");
                assert_eq!(direct.1.as_deref(), Some("Answer"), "{input}");
                let raw_batch = GptOssReasoningParser::new()
                    .unwrap()
                    .detect_and_parse_reasoning(&input, &[]);
                let raw_composed = detect_and_parse_tool_call_with_recovery(
                    &raw_batch.normal_text,
                    Some("harmony"),
                    None,
                )
                .await
                .unwrap();
                assert_eq!(raw_composed.1.as_deref(), Some("Answer"), "{input}");
                let mut outputs = vec![
                    GptOssReasoningParser::new()
                        .unwrap()
                        .detect_and_parse_reasoning_with_context(&input, &[]),
                ];
                for split in 0..=input.len() {
                    outputs.push(collect_context_chunks([
                        (input[..split].to_string(), Vec::new()),
                        (input[split..].to_string(), Vec::new()),
                    ]));
                }
                let tokens = encode_text_to_tokens(&input).unwrap();
                for split in 0..=tokens.len() {
                    outputs.push(collect_context_chunks([
                        (String::new(), tokens[..split].to_vec()),
                        (String::new(), tokens[split..].to_vec()),
                    ]));
                }
                for output in outputs {
                    assert!(output.reasoning_text.is_empty(), "{input}");
                    let parsed =
                        detect_and_parse_tool_call_with_context(&output, Some("harmony"), None)
                            .await
                            .unwrap();
                    assert_eq!(parsed.1.as_deref(), Some("Answer"), "{input}");
                    assert_eq!(parsed.0.len(), expected_calls, "{input}");
                    if expected_calls > 0 {
                        assert_eq!(parsed.0[0].function.arguments, arguments);
                    }
                }
            }
        }
    }

    #[tokio::test]
    async fn optional_channel_function_calls_keep_native_completion_and_quotes() {
        use crate::tool_calling::parsers::detect_and_parse_tool_call_with_context;
        for (prefix, visible) in [
            ("<|start|>assistant to=functions.lookup", ""),
            ("to=functions.lookup", ""),
            ("functions.lookup", ""),
            (
                "<|channel|>final<|message|>Before<|return|>to=functions.lookup",
                "Before",
            ),
        ] {
            for arguments in [
                "",
                r#"{"x":1}"#,
                r#"{"q":"<|call|><|start|>assistant<|channel|>final<|message|>literal<|return|>"}"#,
            ] {
                for ending in ["<|call|>", "<|return|>", "<|end|>", ""] {
                    let input = format!("{prefix}<|message|>{arguments}{ending}");
                    let completed = matches!(ending, "<|call|>" | "<|return|>");
                    let direct = crate::tool_calling::harmony::harmony_parser::parse_tool_calls_harmony_complete(&input, &Default::default(), None).await.unwrap();
                    assert_eq!(direct.0.len(), usize::from(completed), "{input}");
                    let mut outputs = vec![
                        GptOssReasoningParser::new()
                            .unwrap()
                            .detect_and_parse_reasoning_with_context(&input, &[]),
                    ];
                    for split in (0..=input.len()).filter(|&split| input.is_char_boundary(split)) {
                        let mut parser = GptOssReasoningParser::new().unwrap();
                        let mut output = ReasoningOutput::default();
                        for chunk in [&input[..split], &input[split..]] {
                            let part = parser.parse_reasoning_streaming_with_context(chunk, &[]);
                            output.reasoning_text.push_str(&part.reasoning_text);
                            output.normal_segments.extend(part.normal_segments);
                        }
                        let tail = parser.finish_reasoning_stream_with_context();
                        output.reasoning_text.push_str(&tail.reasoning_text);
                        output.normal_segments.extend(tail.normal_segments);
                        assert!(
                            parser
                                .finish_reasoning_stream_with_context()
                                .normal_segments
                                .is_empty()
                        );
                        outputs.push(output);
                    }
                    let tokens = encode_text_to_tokens(&input).unwrap();
                    for split in 0..=tokens.len() {
                        let mut parser = GptOssReasoningParser::new().unwrap();
                        let mut output = ReasoningOutput::default();
                        for chunk in [&tokens[..split], &tokens[split..]] {
                            let part = parser.parse_reasoning_streaming_with_context("", chunk);
                            output.reasoning_text.push_str(&part.reasoning_text);
                            output.normal_segments.extend(part.normal_segments);
                        }
                        let tail = parser.finish_reasoning_stream_with_context();
                        output.reasoning_text.push_str(&tail.reasoning_text);
                        output.normal_segments.extend(tail.normal_segments);
                        outputs.push(output);
                    }
                    for output in outputs {
                        assert!(output.reasoning_text.is_empty(), "{input}");
                        let parsed =
                            detect_and_parse_tool_call_with_context(&output, Some("harmony"), None)
                                .await
                                .unwrap();
                        assert_eq!(parsed.0.len(), direct.0.len(), "{input}");
                        assert_eq!(
                            parsed.1.as_deref(),
                            (!visible.is_empty()).then_some(visible),
                            "{input}"
                        );
                        if completed {
                            assert_eq!(
                                parsed.0[0].function.arguments, direct.0[0].function.arguments,
                                "{input}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[tokio::test]
    async fn context_preserves_literal_protocol_collisions_and_real_calls() {
        use crate::tool_calling::parsers::detect_and_parse_tool_call_with_context;
        let mut cases = Vec::new();
        for channel in ["commentary", "final"] {
            for next in ["analysis", "finalx"] {
                let literal = format!("Before<|channel|>{next}<|message|>Done");
                cases.push((
                    format!("<|channel|>{channel}<|message|>{literal}<|return|>"),
                    literal,
                    "".to_string(),
                    0,
                ));
            }
        }
        let literal = "Use <|channel|>commentary to=functions.lookup<|message|>{}";
        let example = format!("<|channel|>final<|message|>{literal}<|call|>");
        cases.push((example.clone(), literal.to_string(), String::new(), 0));
        cases.push((format!("{example}<|start|>assistant<|channel|>commentary to=functions.real<|message|>{{\"x\":1}}<|return|><|start|>assistant<|channel|>final<|message|>Tail<|return|>"), format!("{literal}\nTail"), String::new(), 1));
        cases.push((
            "<|channel|>final<|message|>Before<|end|><|channel|>analysis<|message|>Done<|return|>"
                .to_string(),
            "Before".to_string(),
            "Done".to_string(),
            0,
        ));
        cases.push(("<|channel|>commentary to=functions.bad<|message|>{\"x\":<|start|>assistant<|channel|>final<|message|>Before<|channel|>analysis<|message|>Done<|return|>".to_string(), "Before<|channel|>analysis<|message|>Done".to_string(), String::new(), 0));
        for (input, normal, reasoning, call_count) in cases {
            let mut outputs = vec![
                GptOssReasoningParser::new()
                    .unwrap()
                    .detect_and_parse_reasoning_with_context(&input, &[]),
            ];
            let mut wrapper = crate::ReasoningParserType::GptOss.get_reasoning_parser();
            outputs.push(wrapper.detect_and_parse_reasoning_with_context(&input, &[]));
            for split in (0..=input.len()).filter(|&split| input.is_char_boundary(split)) {
                let mut parser = GptOssReasoningParser::new().unwrap();
                let mut output = ReasoningOutput::default();
                for chunk in [&input[..split], &input[split..]] {
                    let part = parser.parse_reasoning_streaming_with_context(chunk, &[]);
                    output.normal_segments.extend(part.normal_segments);
                    output.reasoning_text.push_str(&part.reasoning_text);
                }
                let tail = parser.finish_reasoning_stream_with_context();
                output.normal_segments.extend(tail.normal_segments);
                output.reasoning_text.push_str(&tail.reasoning_text);
                assert!(
                    parser
                        .finish_reasoning_stream_with_context()
                        .normal_segments
                        .is_empty()
                );
                outputs.push(output);
            }
            let tokens = encode_text_to_tokens(&input).unwrap();
            for split in 0..=tokens.len() {
                let mut parser = GptOssReasoningParser::new().unwrap();
                let mut output = ReasoningOutput::default();
                for chunk in [&tokens[..split], &tokens[split..]] {
                    let part = parser.parse_reasoning_streaming_with_context("", chunk);
                    output.normal_segments.extend(part.normal_segments);
                    output.reasoning_text.push_str(&part.reasoning_text);
                }
                let tail = parser.finish_reasoning_stream_with_context();
                output.normal_segments.extend(tail.normal_segments);
                output.reasoning_text.push_str(&tail.reasoning_text);
                outputs.push(output);
            }
            for output in outputs {
                assert_eq!(output.reasoning_text, reasoning, "{input}");
                let (calls, text) =
                    detect_and_parse_tool_call_with_context(&output, Some("harmony"), None)
                        .await
                        .unwrap();
                assert_eq!(text.as_deref(), Some(normal.as_str()), "{input}");
                assert_eq!(calls.len(), call_count, "{input}");
                if let Some(call) = calls.first() {
                    assert_eq!(call.function.name, "real");
                }
            }
        }
        // The exact colliding bytes still have their native raw interpretation.
        let (calls, text) =
            crate::tool_calling::harmony::harmony_parser::parse_tool_calls_harmony_complete(
                "Before<|channel|>analysis<|message|>Done",
                &Default::default(),
                None,
            )
            .await
            .unwrap();
        assert!(calls.is_empty());
        assert_eq!(text.as_deref(), Some(""));
        let mut mixed = GptOssReasoningParser::new().unwrap();
        let protocol = "<|channel|>commentary to=functions.real<|message|>{}<|return|>";
        assert!(
            !mixed
                .parse_reasoning_streaming_with_context(protocol, &[])
                .normal_segments
                .is_empty()
        );
        assert!(
            mixed
                .parse_reasoning_streaming_incremental("", &[])
                .normal_text
                .is_empty()
        );
        assert!(
            mixed
                .finish_reasoning_stream_with_context()
                .normal_segments
                .is_empty()
        );
        assert!(
            mixed
                .finish_reasoning_stream_with_context()
                .normal_segments
                .is_empty()
        );
        let bad = ReasoningOutput {
            reasoning_text: String::new(),
            normal_segments: vec![NormalTextSegment::Protocol(protocol.to_string())],
        };
        assert!(
            detect_and_parse_tool_call_with_context(&bad, Some("unregistered-context-test"), None)
                .await
                .is_err()
        );
    }
    use super::*;

    #[test]
    fn header_only_eof_never_invents_channel_suffix_text() {
        for input in [
            "<|channel|>finalx",
            "<|start|>assistant<|channel|>finalx",
            "<|channel|>final",
            "<|start|>assistant<|channel|>final",
            "<|start|>assistant",
        ] {
            assert_recovery_schedules(input, "", "");
        }
    }

    #[tokio::test]
    async fn unfinished_analysis_handoff_stops_at_the_next_framed_message() {
        use crate::tool_calling::parsers::detect_and_parse_tool_call_with_recovery;
        for visible in [
            "",
            "<|start|>assistant<|channel|>final<|message|>Visible<|return|>",
        ] {
            let next = format!(
                "{visible}<|start|>assistant<|channel|>commentary to=functions.lookup<|message|>{{}}<|return|>"
            );
            let input = format!("<|channel|>analysis<|message|>Before{next}");
            let expected_text = if visible.is_empty() { "" } else { "Visible" };
            let batch = GptOssReasoningParser::new()
                .unwrap()
                .detect_and_parse_reasoning(&input, &[]);
            assert_eq!(batch.reasoning_text, "Before");
            let (calls, text) =
                detect_and_parse_tool_call_with_recovery(&batch.normal_text, Some("harmony"), None)
                    .await
                    .unwrap();
            assert_eq!(calls.len(), 1);
            assert_eq!(calls[0].function.name, "lookup");
            assert_eq!(calls[0].function.arguments, "{}");
            assert_eq!(text.as_deref(), Some(expected_text));
            for split in 0..=input.len() {
                let output = collect_stream([&input[..split], &input[split..]]);
                assert_eq!(output.reasoning_text, "Before");
                let (calls, text) = detect_and_parse_tool_call_with_recovery(
                    &output.normal_text,
                    Some("harmony"),
                    None,
                )
                .await
                .unwrap();
                assert_eq!(calls.len(), 1, "split {split}");
                assert_eq!(calls[0].function.name, "lookup");
                assert_eq!(calls[0].function.arguments, "{}");
                assert_eq!(text.as_deref(), Some(expected_text), "split {split}");
            }
        }
    }

    #[test]
    fn complete_framed_headers_replace_every_content_channel() {
        let quoted = "{\"example\":\"<|start|>user<|channel|>final<|message|>literal<|return|>\"}";
        for channel in ["analysis", "commentary", "final", "finalx"] {
            let mut cases = Vec::new();
            for body in ["Before", ""] {
                for role in ["assistant", "user", "python", "python to=browser"] {
                    cases.push((body, role, "<|return|>"));
                }
            }
            cases.extend([
                (" \t", "assistant", "<|return|>"),
                (quoted, "assistant", "<|return|>"),
            ]);
            for ending in ["<|call|>", "<|end|>"] {
                cases.push(("Before", "assistant", ending));
            }
            for (body, role, ending) in cases {
                let prefix = format!("<|channel|>{channel}<|message|>{body}");
                let next = format!("<|start|>{role}<|channel|>final<|message|>Done{ending}");
                let reasoning = if channel == "analysis" {
                    body.to_string()
                } else {
                    String::new()
                };
                let mut normal = if channel == "analysis" {
                    String::new()
                } else {
                    body.to_string()
                };
                if role == "assistant" {
                    if !normal.is_empty() {
                        normal.push('\n');
                    }
                    normal.push_str("Done");
                }
                let control = format!("{prefix}<|end|>{next}");
                assert_recovery_schedules(&control, &normal, &reasoning);
                assert_recovery_schedules(&format!("{prefix}{next}"), &normal, &reasoning);
            }
        }
    }

    #[test]
    fn unconfirmed_framed_headers_remain_in_the_original_content_channel() {
        for channel in ["analysis", "commentary", "final", "finalx"] {
            for tail in [
                "<",
                "<|sta",
                "<|start|>assistant",
                "<|start|>assistant<|channel|>final",
            ] {
                let input = format!("<|channel|>{channel}<|message|>Before{tail}");
                let expected = format!("Before{tail}");
                let (normal, reasoning) = if channel == "analysis" {
                    ("", expected.as_str())
                } else {
                    (expected.as_str(), "")
                };
                assert_recovery_schedules(&input, normal, reasoning);
            }
        }
    }

    #[test]
    fn deferred_body_endings_keep_the_original_channel_and_payload_policy() {
        let tail = "<|start|>assistant<|channel|>final";
        for channel in ["analysis", "commentary", "final", "finalx"] {
            for body in ["Before", "", " \t", "{\"literal\":\"<|return|>\"}"] {
                for ending in ["<|call|>", "<|return|>", "<|end|>"] {
                    let input = format!("<|channel|>{channel}<|message|>{body}{tail}{ending}");
                    let expected =
                        if channel == "commentary" && body.starts_with('{') && ending == "<|call|>"
                        {
                            String::new()
                        } else {
                            format!("{body}{tail}")
                        };
                    let (normal, reasoning) = if channel == "analysis" {
                        ("", expected.as_str())
                    } else {
                        (expected.as_str(), "")
                    };
                    assert_recovery_schedules(&input, normal, reasoning);
                }
            }
        }
    }

    #[test]
    fn complete_replacement_keeps_body_progress_before_finish() {
        for (channel, frame) in ["analysis", "commentary", "final", "finalx"]
            .into_iter()
            .flat_map(|channel| ["", "<|start|>assistant"].map(|frame| (channel, frame)))
        {
            let mut parser = GptOssReasoningParser::new().unwrap();
            let before = parser.parse_reasoning_streaming_incremental(
                &format!("<|channel|>{channel}<|message|>Before"),
                &[],
            );
            assert_eq!(
                before.reasoning_text,
                if channel == "analysis" { "Before" } else { "" }
            );
            assert_eq!(
                before.normal_text,
                if channel == "analysis" { "" } else { "Before" }
            );
            let header = parser
                .parse_reasoning_streaming_incremental(&format!("{frame}<|channel|>final"), &[]);
            assert_eq!(header.normal_text, "");
            assert_eq!(header.reasoning_text, "");
            let transfer = parser.parse_reasoning_streaming_incremental("<|message|>", &[]);
            assert_eq!(transfer.normal_text, "");
            assert_eq!(transfer.reasoning_text, "");
            let done = parser.parse_reasoning_streaming_incremental("Done<|return|>", &[]);
            assert_eq!(
                done.normal_text,
                if channel == "analysis" {
                    "Done"
                } else {
                    "\nDone"
                }
            );
            assert_eq!(done.reasoning_text, "");
            assert!(parser.finish_reasoning_stream().normal_text.is_empty());
            assert!(parser.finish_reasoning_stream().reasoning_text.is_empty());
        }
    }

    #[tokio::test]
    async fn header_owned_action_stop_survives_streaming_handoff() {
        use crate::tool_calling::parsers::detect_and_parse_tool_call_with_recovery;

        for header in [
            "<|channel|>commentary to=functions.lookup ",
            "<|channel|>analysis to=functions.lookup ",
            " to=functions.lookup<|channel|>commentary ",
            " to=functions.lookup<|channel|>analysis ",
            "<|channel|>final ",
        ] {
            for header_stop in ["<|call|>", "<|return|>", "<|end|>"] {
                let tool = header.contains("functions.");
                let body = if tool {
                    "{\"city\":\"Paris\"}"
                } else {
                    "Done."
                };
                let tail = if tool {
                    "<|call|><|start|>assistant<|channel|>final<|message|>Done.<|return|>"
                } else {
                    "<|return|>"
                };
                let input = format!(
                    "<|channel|>commentary to=functions.unfinished<|message|>{{}}<|start|>assistant{header}{header_stop}<|message|>{body}{tail}"
                );
                for split in 0..=input.len() {
                    let mut parser = GptOssReasoningParser::new().unwrap();
                    let mut normal = String::new();
                    for chunk in [&input[..split], &input[split..]] {
                        let delta = parser.parse_reasoning_streaming_incremental(chunk, &[]);
                        assert_eq!(delta.reasoning_text, "");
                        normal.push_str(&delta.normal_text);
                    }
                    let (calls, visible) =
                        detect_and_parse_tool_call_with_recovery(&normal, Some("harmony"), None)
                            .await
                            .unwrap();
                    assert_eq!(calls.len(), usize::from(tool), "split {split}: {normal}");
                    if tool {
                        assert_eq!(calls[0].function.name, "lookup");
                        assert_eq!(calls[0].function.arguments, "{\"city\":\"Paris\"}");
                    }
                    assert_eq!(visible.as_deref(), Some("Done."), "split {split}");
                    assert!(parser.finish_reasoning_stream().normal_text.is_empty());
                }
                let whole = collect_stream([input.as_str()]);
                let tokens = encode_text_to_tokens(&input).unwrap();
                for split in 0..=tokens.len() {
                    let mut parser = GptOssReasoningParser::new().unwrap();
                    let mut output = ParserResult::default();
                    for chunk in [&tokens[..split], &tokens[split..]] {
                        let delta = parser.parse_reasoning_streaming_incremental("", chunk);
                        output.normal_text.push_str(&delta.normal_text);
                        output.reasoning_text.push_str(&delta.reasoning_text);
                    }
                    let tail = parser.finish_reasoning_stream();
                    output.normal_text.push_str(&tail.normal_text);
                    output.reasoning_text.push_str(&tail.reasoning_text);
                    assert_eq!(
                        output.normal_text, whole.normal_text,
                        "token {split}: {header_stop}"
                    );
                    assert_eq!(output.reasoning_text, whole.reasoning_text);
                    assert!(parser.finish_reasoning_stream().normal_text.is_empty());
                }
            }
        }
    }

    #[test]
    fn unframed_replacement_cannot_change_nonassistant_role() {
        let input = "<|start|>user<|channel|>commentary to=functions.old<|message|>{}<|channel|>commentary to=functions.fake <|return|><|message|>{}<|call|><|start|>assistant<|channel|>final<|message|>Done.<|return|>";
        assert_recovery_schedules(input, "Done.", "");
    }

    #[tokio::test]
    async fn unconfirmed_header_stop_keeps_raw_eof_call_policy() {
        use crate::tool_calling::parsers::detect_and_parse_tool_call_with_recovery;

        for ending in ["<|call|>", "<|return|>"] {
            let input = format!(
                "<|channel|>commentary to=functions.lookup<|message|>{{\"q\":1}}<|start|>assistant<|channel|>commentary to=functions.next {ending}"
            );
            let direct = detect_and_parse_tool_call_with_recovery(&input, Some("harmony"), None)
                .await
                .unwrap();
            assert_eq!(direct.0.len(), 1);
            assert_eq!(direct.0[0].function.name, "lookup");
            let mut parser = GptOssReasoningParser::new().unwrap();
            let delta = parser.parse_reasoning_streaming_incremental(&input, &[]);
            assert!(delta.normal_text.is_empty());
            let tail = parser.finish_reasoning_stream();
            let recovered =
                detect_and_parse_tool_call_with_recovery(&tail.normal_text, Some("harmony"), None)
                    .await
                    .unwrap();
            assert_eq!(recovered.0.len(), 1);
            assert_eq!(recovered.0[0].function.name, direct.0[0].function.name);
            assert_eq!(
                recovered.0[0].function.arguments,
                direct.0[0].function.arguments
            );
            assert_eq!(recovered.1, direct.1);
            assert!(parser.finish_reasoning_stream().normal_text.is_empty());
        }
    }

    #[tokio::test]
    async fn initial_header_text_keeps_json_literals_before_replacement() {
        use crate::tool_calling::parsers::detect_and_parse_tool_call_with_recovery;

        let prefix = "{\"visible\":\"<|start|>\"}";
        for following in [
            "<|start|>assistant<|channel|>commentary to=functions.lookup<|message|>{}<|return|>",
            "<|start|>assistant<|channel|>final<|message|>Done.<|return|>",
        ] {
            let input = format!("{prefix}{following}");
            let tool = following.contains("functions.");
            let expected_text = if tool {
                prefix.to_string()
            } else {
                format!("{prefix}Done.")
            };
            for split in 0..=input.len() {
                let output = collect_stream([&input[..split], &input[split..]]);
                assert_eq!(output.reasoning_text, "");
                let (calls, text) = detect_and_parse_tool_call_with_recovery(
                    &output.normal_text,
                    Some("harmony"),
                    None,
                )
                .await
                .unwrap();
                assert_eq!(calls.len(), usize::from(tool), "split {split}");
                assert_eq!(
                    text.as_deref(),
                    Some(expected_text.as_str()),
                    "split {split}"
                );
                if !tool {
                    let batch = GptOssReasoningParser::new()
                        .unwrap()
                        .detect_and_parse_reasoning(&input, &[]);
                    assert_eq!(batch.normal_text, output.normal_text);
                    assert_eq!(batch.reasoning_text, output.reasoning_text);
                }
            }
        }
    }

    #[test]
    fn unfinished_framed_headers_keep_native_role_ownership() {
        for role in ["user", "system", "developer", "tool", "python", "browser"] {
            for channel in ["analysis", "final"] {
                let input = format!("<|start|>{role}<|channel|>{channel} SHOULD_HIDE");
                assert_recovery_schedules(&input, "", "");
            }
        }
    }

    #[tokio::test]
    async fn native_header_domain_survives_recovery_and_handoff() {
        use crate::tool_calling::parsers::detect_and_parse_tool_call_with_recovery;

        for role in [
            "assistant",
            "user",
            "system",
            "developer",
            "tool",
            "python",
            "browser",
        ] {
            for name in ["f/x", "f:x", "f+x", "@f", "雪/☃", "f<g", "f\u{a0}x"] {
                if role != "assistant" && name != "f<g" {
                    continue;
                }
                for channel in ["analysis", "commentary"] {
                    for recipient_first in [false, true] {
                        for prefix in [
                            "",
                            "<|channel|>commentary to=functions.bad<|message|>{\"q\":",
                        ] {
                            for ending in ["<|call|>", "<|return|>"] {
                                let header = if recipient_first {
                                    format!(
                                        "<|start|>{role} to=functions.{name}<|channel|>{channel}<|message|>"
                                    )
                                } else {
                                    format!(
                                        "<|start|>{role}<|channel|>{channel} to=functions.{name}<|message|>"
                                    )
                                };
                                let input = format!("{prefix}{header}{{\"x\":1}}{ending}");
                                let expected = usize::from(role == "assistant");
                                let (calls, normal) = detect_and_parse_tool_call_with_recovery(
                                    &input,
                                    Some("harmony"),
                                    Some(&[]),
                                )
                                .await
                                .unwrap();
                                assert_eq!(calls.len(), expected, "direct {input}: {calls:?}");
                                if expected == 1 {
                                    assert_eq!(calls[0].function.name, name, "direct {input}");
                                    assert_eq!(calls[0].function.arguments, "{\"x\":1}");
                                }
                                assert_eq!(normal.as_deref(), Some(""), "direct {input}");
                                let batch = GptOssReasoningParser::new()
                                    .unwrap()
                                    .detect_and_parse_reasoning(&input, &[]);
                                assert_eq!(batch.reasoning_text, "", "batch {input}");
                                let (calls, normal) = detect_and_parse_tool_call_with_recovery(
                                    &batch.normal_text,
                                    Some("harmony"),
                                    Some(&[]),
                                )
                                .await
                                .unwrap();
                                assert_eq!(calls.len(), expected, "batch {input}: {batch:?}");
                                if expected == 1 {
                                    assert_eq!(calls[0].function.name, name, "batch {input}");
                                }
                                assert_eq!(normal.as_deref(), Some(""), "batch {input}");
                                let whole = collect_stream([input.as_str()]);
                                assert_eq!(whole.reasoning_text, "", "stream {input}");
                                let (calls, normal) = detect_and_parse_tool_call_with_recovery(
                                    &whole.normal_text,
                                    Some("harmony"),
                                    Some(&[]),
                                )
                                .await
                                .unwrap();
                                assert_eq!(calls.len(), expected, "stream {input}: {whole:?}");
                                if expected == 1 {
                                    assert_eq!(calls[0].function.name, name, "stream {input}");
                                    assert_eq!(calls[0].function.arguments, "{\"x\":1}");
                                }
                                assert_eq!(normal.as_deref(), Some(""), "stream {input}");
                                for split in input
                                    .char_indices()
                                    .map(|(index, _)| index)
                                    .chain([input.len()])
                                {
                                    let split_output =
                                        collect_stream([&input[..split], &input[split..]]);
                                    assert_eq!(
                                        split_output.normal_text, whole.normal_text,
                                        "text {split}: {input}"
                                    );
                                    assert_eq!(
                                        split_output.reasoning_text, whole.reasoning_text,
                                        "text {split}: {input}"
                                    );
                                }
                                let tokens = encode_text_to_tokens(&input).unwrap();
                                for split in 0..=tokens.len() {
                                    let mut parser = GptOssReasoningParser::new().unwrap();
                                    let mut output = ParserResult::default();
                                    for chunk in [&tokens[..split], &tokens[split..]] {
                                        let delta =
                                            parser.parse_reasoning_streaming_incremental("", chunk);
                                        output.normal_text.push_str(&delta.normal_text);
                                        output.reasoning_text.push_str(&delta.reasoning_text);
                                    }
                                    let tail = parser.finish_reasoning_stream();
                                    output.normal_text.push_str(&tail.normal_text);
                                    output.reasoning_text.push_str(&tail.reasoning_text);
                                    assert_eq!(
                                        output.normal_text, whole.normal_text,
                                        "token {split}: {input}"
                                    );
                                    assert_eq!(
                                        output.reasoning_text, whole.reasoning_text,
                                        "token {split}: {input}"
                                    );
                                    let repeated = parser.finish_reasoning_stream();
                                    assert_eq!(repeated.normal_text, "");
                                    assert_eq!(repeated.reasoning_text, "");
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn reasoning_only_projects_assistant_messages() {
        let encoding = get_harmony_encoding().as_ref().unwrap();
        for role in ["system", "developer", "user", "tool"] {
            for channel in ["analysis", "final", "commentary", "unknown"] {
                let input = format!(
                    "<|start|>{role}<|channel|>{channel}<|message|>UNTRUSTED<|end|><|start|>assistant<|channel|>final<|message|>ANSWER<|return|>"
                );
                let expected = ParserResult {
                    normal_text: "ANSWER".to_string(),
                    reasoning_text: String::new(),
                };
                let actual = GptOssReasoningParser::new()
                    .unwrap()
                    .detect_and_parse_reasoning(&input, &[]);
                assert_eq!(actual.normal_text, expected.normal_text, "{role}/{channel}");
                assert_eq!(
                    actual.reasoning_text, expected.reasoning_text,
                    "{role}/{channel}"
                );
                for split in input.char_indices().map(|(i, _)| i).chain([input.len()]) {
                    let actual = collect_stream([&input[..split], &input[split..]]);
                    assert_eq!(
                        actual.normal_text, expected.normal_text,
                        "{role}/{channel}/{split}"
                    );
                    assert_eq!(
                        actual.reasoning_text, expected.reasoning_text,
                        "{role}/{channel}/{split}"
                    );
                }
                let mut parser = GptOssReasoningParser::new().unwrap();
                let mut actual = ParserResult::default();
                for token in encoding.tokenizer().encode_with_special_tokens(&input) {
                    let delta = parser.parse_reasoning_streaming_incremental("", &[token]);
                    actual.normal_text.push_str(&delta.normal_text);
                    actual.reasoning_text.push_str(&delta.reasoning_text);
                }
                let tail = parser.finish_reasoning_stream();
                actual.normal_text.push_str(&tail.normal_text);
                actual.reasoning_text.push_str(&tail.reasoning_text);
                assert_eq!(
                    actual.normal_text, expected.normal_text,
                    "token {role}/{channel}"
                );
                assert_eq!(
                    actual.reasoning_text, expected.reasoning_text,
                    "token {role}/{channel}"
                );
                assert!(parser.finish_reasoning_stream().normal_text.is_empty());
            }
        }
    }

    #[tokio::test]
    async fn recovery_stream_retains_unframed_and_replacement_headers() {
        use crate::tool_calling::parsers::detect_and_parse_tool_call_with_recovery;

        for (prefix, visible) in [
            ("<|channel|>final<|message|>before<|return|>", "before"),
            ("<|channel|>commentary to=functions.bad", ""),
            (
                "<|start|>assistant<|channel|>commentary to=functions.bad",
                "",
            ),
        ] {
            for frame in ["", "<|start|>assistant "] {
                if visible.is_empty() && frame.is_empty() {
                    continue;
                }
                for ending in ["<|call|>", "<|return|>"] {
                    let input = format!(
                        "{prefix}{frame}to=functions.lookup<|channel|>commentary<|message|>{{\"path\":\"雪\",\"content\":\"<|return|>\"}}{ending}"
                    );
                    let mut outputs: Vec<_> = input
                        .char_indices()
                        .map(|(i, _)| i)
                        .chain([input.len()])
                        .map(|split| {
                            (
                                format!("text {split}"),
                                collect_stream([&input[..split], &input[split..]]),
                            )
                        })
                        .collect();
                    let tokens = encode_text_to_tokens(&input).unwrap();
                    for split in 0..=tokens.len() {
                        let mut parser = GptOssReasoningParser::new().unwrap();
                        let mut output = ParserResult::default();
                        for chunk in [&tokens[..split], &tokens[split..]] {
                            let delta = parser.parse_reasoning_streaming_incremental("", chunk);
                            output.normal_text.push_str(&delta.normal_text);
                            output.reasoning_text.push_str(&delta.reasoning_text);
                        }
                        let tail = parser.finish_reasoning_stream();
                        output.normal_text.push_str(&tail.normal_text);
                        output.reasoning_text.push_str(&tail.reasoning_text);
                        assert!(parser.finish_reasoning_stream().normal_text.is_empty());
                        outputs.push((format!("token {split}"), output));
                    }
                    for (split, output) in outputs {
                        let (calls, normal) = detect_and_parse_tool_call_with_recovery(
                            &output.normal_text,
                            Some("harmony"),
                            Some(&[]),
                        )
                        .await
                        .unwrap();
                        assert_eq!(output.reasoning_text, "", "{input} split={split}");
                        assert_eq!(calls.len(), 1, "{input} split={split}: {output:?}");
                        assert_eq!(calls[0].function.name, "lookup", "{input} split={split}");
                        assert_eq!(
                            calls[0].function.arguments,
                            "{\"path\":\"雪\",\"content\":\"<|return|>\"}"
                        );
                        assert_eq!(normal.as_deref(), Some(visible), "{input} split={split}");
                    }
                }
            }
        }
    }

    #[tokio::test]
    async fn recovery_replacement_final_header_keeps_visible_body() {
        use crate::tool_calling::parsers::detect_and_parse_tool_call_with_recovery;

        for (prefix, expected) in [
            ("", "ANSWER"),
            ("<|start|>assistant", "ANSWER"),
            ("<|start|>user", ""),
            ("<|start|>system", ""),
            ("<|start|>developer", ""),
            ("<|start|>tool", ""),
        ] {
            let input = format!(
                "{prefix}<|channel|>commentary to=functions.bad<|channel|>final<|message|>ANSWER<|return|>"
            );
            assert_recovery_schedules(&input, expected, "");
            let (calls, normal) =
                detect_and_parse_tool_call_with_recovery(&input, Some("harmony"), Some(&[]))
                    .await
                    .unwrap();
            assert!(calls.is_empty(), "{input}: {calls:?}");
            assert_eq!(normal.as_deref(), Some(expected), "{input}");
        }
    }

    #[test]
    fn recovery_unframed_tail_keeps_message_separation() {
        assert_recovery_schedules(
            "<|channel|>final<|message|>before<|return|>after",
            "before\nafter",
            "",
        );
        assert_recovery_schedules(
            "<|channel|>final<|message|>before<|return|><|channel|>final<|message|>after<|return|>",
            "before\nafter",
            "",
        );
    }

    #[tokio::test]
    async fn unfinished_visible_or_analysis_body_transfers_to_bare_final() {
        use crate::tool_calling::harmony::harmony_parser::parse_tool_calls_harmony_complete;
        for channel in ["commentary", "analysis", "final", "unknown"] {
            for body in ["Before", "", " ", r#"{"x":"<|start|>"}"#] {
                for ending in ["<|return|>", "<|call|>", "<|end|>", ""] {
                    let input = format!(
                        "<|channel|>{channel}<|message|>{body}<|channel|>final<|message|>Done{ending}"
                    );
                    let visible = if channel == "analysis" || body.is_empty() {
                        "Done".to_string()
                    } else {
                        format!("{body}\nDone")
                    };
                    assert_recovery_schedules(
                        &input,
                        &visible,
                        if channel == "analysis" { body } else { "" },
                    );
                    let (calls, normal) =
                        parse_tool_calls_harmony_complete(&input, &Default::default(), None)
                            .await
                            .unwrap();
                    assert!(calls.is_empty(), "{input}");
                    assert_eq!(
                        normal.as_deref(),
                        Some(
                            if channel == "analysis" {
                                "Done".to_string()
                            } else {
                                format!("{body}Done")
                            }
                            .as_str()
                        ),
                        "{input}"
                    );
                }
            }
            for next in ["analysis", "finalx"] {
                let literal = format!("Before<|channel|>{next}<|message|>Done");
                let input = format!("<|channel|>{channel}<|message|>{literal}<|return|>");
                assert_recovery_schedules(
                    &input,
                    if channel == "analysis" { "" } else { &literal },
                    if channel == "analysis" { &literal } else { "" },
                );
            }
        }
    }

    #[tokio::test]
    async fn json_prelude_keeps_quoted_markers_before_bare_final() {
        use crate::tool_calling::harmony::harmony_parser::parse_tool_calls_harmony_complete;
        for prefix in [
            r#"{"x":"<|start|>"}"#,
            r#"["<|channel|>",1]"#,
            r#""<|start|>\"<|return|>""#,
            r#""plain""#,
            "{\"x\":1",
            "[1,2",
            "{",
            "[",
        ] {
            for frame in ["", "<|start|>assistant"] {
                let input = format!("{prefix}{frame}<|channel|>final<|message|>Done<|return|>");
                assert_recovery_schedules(&input, &format!("{prefix}Done"), "");
                let (calls, normal) =
                    parse_tool_calls_harmony_complete(&input, &Default::default(), None)
                        .await
                        .unwrap();
                assert!(calls.is_empty(), "{input}");
                assert_eq!(
                    normal.as_deref(),
                    Some(format!("{prefix}Done").as_str()),
                    "{input}"
                );
            }
        }
        for prefix in [
            "true",
            "false",
            "null",
            "42",
            "-1.5",
            "python",
            "functions.lookup",
            "to=functions.lookup",
        ] {
            let input = format!("{prefix}<|channel|>final<|message|>Done<|return|>");
            assert_recovery_schedules(&input, "Done", "");
            let (calls, normal) =
                parse_tool_calls_harmony_complete(&input, &Default::default(), None)
                    .await
                    .unwrap();
            assert!(calls.is_empty(), "{input}");
            assert_eq!(normal.as_deref(), Some("Done"), "{input}");
        }
    }

    #[test]
    fn recipientless_commentary_json_obeys_its_ending() {
        for (constraint, content, payload) in [
            ("", "{\"x\":1}", true),
            ("", "  [1,2]", true),
            ("", " \n{\"x\":\"<|call|>雪\"}", true),
            ("", "  ", false),
            ("", "plain { prose", false),
            (" <|constrain|>json", "42", true),
            (" <|constrain|>json", "  ", true),
        ] {
            for ending in ["<|call|>", "<|return|>", "<|end|>", ""] {
                let input =
                    format!("<|channel|>commentary{constraint}<|message|>{content}{ending}");
                let expected = if ending == "<|call|>" && payload {
                    ""
                } else {
                    content
                };
                assert_recovery_schedules(&input, expected, "");
            }
        }
    }

    #[tokio::test]
    async fn explicit_constraint_recipient_stays_directed() {
        use crate::tool_calling::parsers::detect_and_parse_tool_call_with_recovery;

        for ending in ["<|call|>", "<|return|>", "<|end|>", ""] {
            let input = format!(
                "<|channel|>commentary to=<|constrain|>json<|message|>{{\"secret\":\"directed\"}}{ending}"
            );
            assert_recovery_schedules(&input, "", "");
            let (calls, normal) =
                detect_and_parse_tool_call_with_recovery(&input, Some("harmony"), Some(&[]))
                    .await
                    .unwrap();
            assert!(calls.is_empty(), "{input}");
            assert_eq!(normal.as_deref(), Some(""), "{input}");
        }
    }

    #[tokio::test]
    async fn recovery_interrupted_recipientless_json_keeps_message_ownership() {
        use crate::tool_calling::parsers::detect_and_parse_tool_call_with_recovery;

        let content = "{\"x\":\"<|start|>雪\"}";
        for role in ["assistant", "user", "tool"] {
            for constraint in ["", " <|constrain|>json"] {
                for frame in ["", "<|start|>assistant"] {
                    let input = format!(
                        "<|start|>{role}<|channel|>commentary{constraint}<|message|>{content}{frame}<|channel|>final<|message|>Done<|return|>"
                    );
                    let expected = if role == "assistant" {
                        format!("{content}\nDone")
                    } else if frame.is_empty() {
                        String::new()
                    } else {
                        "Done".to_string()
                    };
                    assert_recovery_schedules(&input, &expected, "");
                    let (calls, normal) = detect_and_parse_tool_call_with_recovery(
                        &input,
                        Some("harmony"),
                        Some(&[]),
                    )
                    .await
                    .unwrap();
                    assert!(calls.is_empty(), "{input}: {calls:?}");
                    assert_eq!(
                        normal.as_deref(),
                        Some(expected.replace('\n', "").as_str()),
                        "{input}"
                    );
                }
            }
        }
    }

    fn assert_recovery_schedules(input: &str, normal: &str, reasoning: &str) {
        let batch = GptOssReasoningParser::new()
            .unwrap()
            .detect_and_parse_reasoning(input, &[]);
        assert_eq!(batch.normal_text, normal);
        assert_eq!(batch.reasoning_text, reasoning);
        for split in (0..=input.len()).filter(|&split| input.is_char_boundary(split)) {
            let output = collect_stream([&input[..split], &input[split..]]);
            assert_eq!(output.normal_text, normal, "text split {split}");
            assert_eq!(output.reasoning_text, reasoning, "text split {split}");
        }
        let tokens = encode_text_to_tokens(input).unwrap();
        for split in 0..=tokens.len() {
            let mut parser = GptOssReasoningParser::new().unwrap();
            let mut output = ParserResult::default();
            for chunk in [&tokens[..split], &tokens[split..]] {
                let delta = parser.parse_reasoning_streaming_incremental("", chunk);
                output.normal_text.push_str(&delta.normal_text);
                output.reasoning_text.push_str(&delta.reasoning_text);
            }
            let tail = parser.finish_reasoning_stream();
            output.normal_text.push_str(&tail.normal_text);
            output.reasoning_text.push_str(&tail.reasoning_text);
            assert_eq!(output.normal_text, normal, "token split {split}");
            assert_eq!(output.reasoning_text, reasoning, "token split {split}");
            let repeated = parser.finish_reasoning_stream();
            assert_eq!(repeated.normal_text, "");
            assert_eq!(repeated.reasoning_text, "");
        }
    }

    #[test]
    fn recovery_batch_keeps_completed_messages_after_orphan_stop() {
        for framing in ["", "<|start|>assistant"] {
            for stop in ["<|return|>", "<|call|>"] {
                for discarded in ["", "unframed", r#"{"x":""#] {
                    let input = format!(
                        "{framing}<|channel|>analysis<|message|>thought<|end|><|start|>assistant<|channel|>commentary<|message|>雪<|end|>{stop}{discarded}<|start|>assistant<|channel|>final<|message|>done<|return|>"
                    );
                    assert_recovery_schedules(&input, "雪\ndone", "thought");
                }
            }
        }
    }

    #[test]
    fn recovery_unfinished_directed_analysis_is_not_reasoning() {
        for framing in ["", "<|start|>assistant"] {
            for channel in ["analysis", "commentary"] {
                for recipient in ["functions.lookup", "browser"] {
                    for suffix in ["", "<", "<|"] {
                        let input = format!(
                            r#"{framing}<|channel|>{channel} to={recipient}<|message|>{{"path":"雪","content":"unfinished{suffix}"#
                        );
                        assert_recovery_schedules(&input, "", "");
                    }
                }
            }
        }
    }

    #[test]
    fn recovery_unknown_channel_keeps_stream_position() {
        for framing in ["", "<|start|>assistant"] {
            for channel in ["finalx", "custom"] {
                for suffix in ["", "<", "<|"] {
                    let input = format!(
                        "{framing}<|channel|>{channel}<|message|>first<|end|><|start|>assistant<|channel|>final<|message|>雪{suffix}"
                    );
                    assert_recovery_schedules(&input, &format!("first\n雪{suffix}"), "");
                }
                let input = format!(
                    "{framing}<|channel|>{channel} to=functions.lookup<|message|>{{}}<|return|><|start|>assistant<|channel|>final<|message|>visible<|return|>"
                );
                assert_recovery_schedules(&input, "visible", "");
            }
        }
    }

    #[test]
    fn test_harmony_action_stops_and_legacy_terminator_export() {
        let encoding = get_harmony_encoding().as_ref().unwrap();
        let native = encoding.stop_tokens_for_assistant_actions().unwrap();
        let action_stops: std::collections::HashSet<_> = harmony_action_token_ids()
            .unwrap()
            .iter()
            .copied()
            .collect();
        assert_eq!(action_stops, native);
        assert_eq!(
            harmony_terminator_token_ids(),
            encoding.tokenizer().encode_with_special_tokens("<|call|>")
        );
        for marker in ["<|call|>", "<|return|>"] {
            let ids = encoding.tokenizer().encode_with_special_tokens(marker);
            assert_eq!(ids.len(), 1);
            assert!(action_stops.contains(&ids[0]));
        }
    }

    fn collect_stream<'a>(chunks: impl IntoIterator<Item = &'a str>) -> ParserResult {
        let mut parser = GptOssReasoningParser::new().expect("Failed to create parser");
        let mut output = ParserResult::default();
        for chunk in chunks {
            let delta = parser.parse_reasoning_streaming_incremental(chunk, &[]);
            output.normal_text.push_str(&delta.normal_text);
            output.reasoning_text.push_str(&delta.reasoning_text);
        }
        let delta = parser.finish_reasoning_stream();
        output.normal_text.push_str(&delta.normal_text);
        output.reasoning_text.push_str(&delta.reasoning_text);
        output
    }

    #[test]
    fn test_gpt_oss_streaming_error_preserves_already_parsed_deltas() {
        let input = "<|channel|>analysis<|message|>thought<|end|><|start|>assistant<|channel|>final<|message|>answer<|return|><|return|>unframed tail";
        let output = collect_stream([input]);
        assert_eq!(output.reasoning_text, "thought");
        assert_eq!(output.normal_text, "answer");
    }

    #[test]
    fn test_gpt_oss_streaming_error_resynchronizes_at_every_text_split() {
        let input = "<|channel|>analysis<|message|>thought<|end|><|return|>unframed<|call|><|start|>assistant<|channel|>final<|message|>answer<|return|>assistant<|start|>assistant<|channel|>final<|message|>again<|return|>";
        for split in 0..=input.len() {
            let output = collect_stream([&input[..split], &input[split..]]);
            assert_eq!(output.reasoning_text, "thought", "split {split}");
            assert_eq!(output.normal_text, "answer\nagain", "split {split}");
        }
    }

    #[test]
    fn test_gpt_oss_streaming_error_resynchronizes_at_every_token_chunk_size() {
        let input = "<|channel|>analysis<|message|>thought<|end|><|return|>unframed<|call|><|start|>assistant<|channel|>final<|message|>答え🌍<|return|><|return|><|start|>assistant<|channel|>final<|message|>again<|return|>";
        let tokens = encode_text_to_tokens(input).expect("Failed to encode input");
        for size in 1..=tokens.len() {
            let mut parser = GptOssReasoningParser::new().expect("Failed to create parser");
            let mut output = ParserResult::default();
            for chunk in tokens.chunks(size) {
                let delta = parser.parse_reasoning_streaming_incremental("", chunk);
                output.normal_text.push_str(&delta.normal_text);
                output.reasoning_text.push_str(&delta.reasoning_text);
            }
            assert_eq!(output.reasoning_text, "thought", "chunk size {size}");
            assert_eq!(output.normal_text, "答え🌍\nagain", "chunk size {size}");
        }
    }

    #[test]
    fn test_gpt_oss_streaming_error_before_first_text_recovers_without_separator() {
        let input = "<|channel|>final<|message|><|return|><|return|>unframed<|start|>assistant<|channel|>final<|message|>answer<|return|>";
        let output = collect_stream([input]);
        assert_eq!(output.normal_text, "answer");
        assert_eq!(output.reasoning_text, "");
    }

    #[test]
    fn test_gpt_oss_streaming_error_preserves_completed_call_and_drops_unframed_call() {
        let call = "<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{\"city\":\"SF\"}<|call|>";
        let input = format!(
            "{call}<|return|><|channel|>commentary to=functions.bad<|message|>{{}}<|call|><|start|>assistant<|channel|>final<|message|>done<|return|>"
        );
        for split in 0..=input.len() {
            let output = collect_stream([&input[..split], &input[split..]]);
            assert_eq!(output.normal_text, format!("{call}done"), "split {split}");
            assert_eq!(output.reasoning_text, "", "split {split}");
        }
    }

    #[test]
    fn test_gpt_oss_streaming_error_recovers_directed_call_then_reasoning() {
        let call = "<|channel|>analysis to=functions.search code<|message|>{\"q\":\"x\"}<|call|>";
        let input = format!(
            "<|channel|>final<|message|><|return|><|return|><|start|>assistant{call}<|start|>assistant<|channel|>analysis<|message|>thought<|end|><|start|>assistant<|channel|>final<|message|>done<|return|>"
        );
        for split in 0..=input.len() {
            let output = collect_stream([&input[..split], &input[split..]]);
            assert_eq!(output.normal_text, format!("{call}done"), "split {split}");
            assert_eq!(output.reasoning_text, "thought", "split {split}");
        }
    }

    #[test]
    fn test_gpt_oss_streaming_error_does_not_recover_unframed_tokens_at_eof() {
        let mut parser = GptOssReasoningParser::new().expect("Failed to create parser");
        let output = parser.parse_reasoning_streaming_incremental(
            "<|channel|>analysis<|message|>thought<|end|><|return|>unframed<|sta",
            &[],
        );
        assert_eq!(output.reasoning_text, "thought");
        assert_eq!(output.normal_text, "");
        for _ in 0..2 {
            let output = parser.finish_reasoning_stream();
            assert_eq!(output.normal_text, "");
            assert_eq!(output.reasoning_text, "");
        }
    }

    #[test] // REASONING.batch.2.c, TOOLCALLING.harmony.1
    fn test_gpt_oss_reasoning_parser() {
        let mut parser = GptOssReasoningParser::new().expect("Failed to create parser");
        let text = "<|channel|>analysis<|message|>The user asks a simple factual question: capital of Brazil. The answer is Brasília. No additional explanation needed.<|end|><|start|>assistant<|channel|>final<|message|>The capital of Brazil is Brasília.";
        let result = parser.detect_and_parse_reasoning(text, &[]);
        assert!(result.normal_text == "The capital of Brazil is Brasília.");
        assert!(
            result.reasoning_text
                == "The user asks a simple factual question: capital of Brazil. The answer is Brasília. No additional explanation needed."
        );
    }

    #[test] // REASONING.batch.4
    fn test_gpt_oss_final_channel_without_analysis_is_normal_text() {
        let mut parser = GptOssReasoningParser::new().expect("Failed to create parser");
        let result = parser.detect_and_parse_reasoning("<|channel|>final<|message|>answer", &[]);
        assert_eq!(result.reasoning_text, "");
        assert_eq!(result.normal_text, "answer");
    }

    #[test] // REASONING.batch.6.a
    fn test_gpt_oss_multiple_analysis_spans_join_before_final() {
        let mut parser = GptOssReasoningParser::new().expect("Failed to create parser");
        let text = "<|channel|>analysis<|message|>first<|end|><|start|>assistant<|channel|>analysis<|message|>second<|end|><|start|>assistant<|channel|>final<|message|>done";
        let result = parser.detect_and_parse_reasoning(text, &[]);
        assert_eq!(result.reasoning_text, "first\nsecond");
        assert_eq!(result.normal_text, "done");
    }

    #[test]
    fn test_gpt_oss_directed_commentary_is_tool_parser_handoff() {
        let mut parser = GptOssReasoningParser::new().expect("Failed to create parser");
        let text = "<|channel|>analysis<|message|>think<|end|><|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{\"city\":\"SF\"}<|call|><|start|>assistant<|channel|>final<|message|>It is sunny.";
        let result = parser.detect_and_parse_reasoning(text, &[]);
        assert_eq!(result.reasoning_text, "think");
        assert_eq!(
            result.normal_text,
            "<|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{\"city\":\"SF\"}<|call|><|start|>assistant<|channel|>final<|message|>It is sunny."
        );
    }

    async fn assert_weather_handoff(output: ParserResult, location: &str, context: &str) {
        use crate::tool_calling::parsers::detect_and_parse_tool_call_with_recovery;

        assert_eq!(output.reasoning_text, "");
        let (calls, normal) = detect_and_parse_tool_call_with_recovery(
            &output.normal_text,
            Some("harmony"),
            Some(&[]),
        )
        .await
        .unwrap();
        assert_eq!(
            calls.len(),
            1,
            "{context}: handoff {:?}",
            output.normal_text
        );
        assert_eq!(calls[0].function.name, "get_current_weather");
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&calls[0].function.arguments).unwrap(),
            serde_json::json!({"format": "celsius", "location": location})
        );
        assert!(normal.as_deref().is_none_or(str::is_empty));
    }

    #[tokio::test]
    async fn test_gpt_oss_action_endings_survive_batch_and_stream_handoff() {
        for channel in ["commentary", "analysis"] {
            for prefix in ["", "<|start|>assistant"] {
                for ending in ["<|call|>", "<|return|>"] {
                    for location in ["San Francisco", "San Francisco\u{1f30d}"] {
                        let arguments =
                            serde_json::json!({"format": "celsius", "location": location});
                        let text = format!(
                            "{prefix}<|channel|>{channel} to=functions.get_current_weather <|constrain|>json<|message|>{arguments}{ending}"
                        );
                        let token_ids = encode_text_to_tokens(&text).unwrap();
                        for supplied_ids in [&[][..], token_ids.as_slice()] {
                            let mut parser = GptOssReasoningParser::new().unwrap();
                            let output = parser.detect_and_parse_reasoning(&text, supplied_ids);
                            assert_weather_handoff(
                                output,
                                location,
                                &format!("batch tokens={} {text}", !supplied_ids.is_empty()),
                            )
                            .await;
                        }
                        for split in text.char_indices().map(|(i, _)| i).chain([text.len()]) {
                            assert_weather_handoff(
                                collect_stream([&text[..split], &text[split..]]),
                                location,
                                &format!("text split={split} {text}"),
                            )
                            .await;
                        }
                        for split in 0..=token_ids.len() {
                            let mut parser = GptOssReasoningParser::new().unwrap();
                            let mut output = ParserResult::default();
                            for ids in [&token_ids[..split], &token_ids[split..]] {
                                let delta = parser.parse_reasoning_streaming_incremental("", ids);
                                output.normal_text.push_str(&delta.normal_text);
                                output.reasoning_text.push_str(&delta.reasoning_text);
                            }
                            let finished = parser.finish_reasoning_stream();
                            output.normal_text.push_str(&finished.normal_text);
                            output.reasoning_text.push_str(&finished.reasoning_text);
                            let repeated = parser.finish_reasoning_stream();
                            assert!(repeated.normal_text.is_empty());
                            assert!(repeated.reasoning_text.is_empty());
                            assert_weather_handoff(
                                output,
                                location,
                                &format!("token split={split} {text}"),
                            )
                            .await;
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_gpt_oss_action_endings_preserve_recipientless_text() {
        for (prefix, channel) in [
            ("", "commentary"),
            ("", "final"),
            ("<|start|>assistant", "commentary"),
            ("<|start|>assistant", "final"),
        ] {
            let text = format!("{prefix}<|channel|>{channel}<|message|>Done.<|return|>");
            let mut parser = GptOssReasoningParser::new().unwrap();
            let output = parser.detect_and_parse_reasoning(&text, &[]);
            assert_eq!(output.normal_text, "Done.");
            assert_eq!(output.reasoning_text, "");
            let streamed = collect_stream([text.as_str()]);
            assert_eq!(streamed.normal_text, "Done.");
            assert_eq!(streamed.reasoning_text, "");
        }
        for ending in ["", "<|end|>"] {
            let text = format!(
                "<|channel|>commentary to=functions.get_current_weather<|message|>{{}}{ending}"
            );
            let mut parser = GptOssReasoningParser::new().unwrap();
            assert!(
                parser
                    .detect_and_parse_reasoning(&text, &[])
                    .normal_text
                    .is_empty()
            );
        }
    }

    #[tokio::test]
    async fn test_gpt_oss_final_marker_examples_are_not_tool_calls() {
        use crate::tool_calling::parsers::detect_and_parse_tool_call_with_recovery;

        for prefix in ["", "<|start|>assistant"] {
            let text = format!(
                "{prefix}<|channel|>final<|message|>Example: <|channel|>commentary to=functions.get_current_weather<|message|>{{\"format\":\"celsius\",\"location\":\"San Francisco\"}}<|return|>"
            );
            let mut parser = GptOssReasoningParser::new().unwrap();
            let output = parser.detect_and_parse_reasoning(&text, &[]);
            let (calls, normal) = detect_and_parse_tool_call_with_recovery(
                &output.normal_text,
                Some("harmony"),
                Some(&[]),
            )
            .await
            .unwrap();
            assert!(calls.is_empty(), "{text}");
            assert!(
                normal
                    .as_deref()
                    .is_some_and(|text| text.contains("Example:"))
            );
        }
    }

    #[test]
    fn test_gpt_oss_analysis_call_does_not_swallow_commentary_handoff() {
        let mut parser = GptOssReasoningParser::new().expect("Failed to create parser");
        let text = "<|channel|>analysis to=functions.search <|constrain|>json<|message|>{\"q\":\"x\"}<|call|><|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{\"city\":\"SF\"}<|call|><|start|>assistant<|channel|>final<|message|>It is sunny.";
        let result = parser.detect_and_parse_reasoning(text, &[]);
        // An analysis-channel message WITH a `to=functions.` recipient is a
        // (malformed) directed tool call, not chain-of-thought: its args are NOT
        // leaked to reasoning_text.
        assert_eq!(result.reasoning_text, "");
        // Both directed tool-call envelopes (the analysis `search` call and the
        // commentary `get_weather` call) plus the final message are handed off
        // intact to the downstream tool-call jail — the analysis call no longer
        // swallows or strips the commentary handoff.
        assert_eq!(
            result.normal_text,
            "<|channel|>analysis to=functions.search <|constrain|>json<|message|>{\"q\":\"x\"}<|call|><|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{\"city\":\"SF\"}<|call|><|start|>assistant<|channel|>final<|message|>It is sunny."
        );
    }

    #[test]
    fn test_gpt_oss_recipientless_commentary_is_normal_text() {
        let mut parser = GptOssReasoningParser::new().expect("Failed to create parser");
        let text = "<|channel|>commentary<|message|>I will check that.<|end|><|start|>assistant<|channel|>final<|message|>Done.";
        let result = parser.detect_and_parse_reasoning(text, &[]);
        assert_eq!(result.reasoning_text, "");
        assert_eq!(result.normal_text, "I will check that.\nDone.");
    }

    #[test] // REASONING.stream.2.a, REASONING.batch.2.c, TOOLCALLING.harmony.1
    fn test_gpt_oss_reasoning_parser_streaming() {
        let mut parser = GptOssReasoningParser::new().expect("Failed to create parser");
        let chunks = vec![
            "<|channel|>",
            "analysis<|message|>The user asks a simple factual question: capital of Brazil.",
            " The answer is Brasília. No additional explanation needed.",
            "<|end|><|start|>assistant<|channel|>final<|message|>",
            "The capital of Brazil is Brasília.",
        ];
        let mut reasoning_text_incr = String::new();
        let mut normal_text_incr = String::new();
        for chunk in chunks {
            let result = parser.parse_reasoning_streaming_incremental(chunk, &[]);
            normal_text_incr.push_str(&result.normal_text);
            reasoning_text_incr.push_str(&result.reasoning_text);
        }
        assert!(normal_text_incr == "The capital of Brazil is Brasília.");
        assert!(
            reasoning_text_incr
                == "The user asks a simple factual question: capital of Brazil. The answer is Brasília. No additional explanation needed."
        );
    }

    #[test] // REASONING.stream.3.b
    fn test_gpt_oss_reasoning_parser_streaming_split_end_marker() {
        let mut parser = GptOssReasoningParser::new().expect("Failed to create parser");
        let chunks = vec![
            "<|channel|>analysis<|message|>thinking<|e",
            "nd|><|start|>assistant<|channel|>final<|message|>answer",
        ];
        let mut reasoning_text_incr = String::new();
        let mut normal_text_incr = String::new();
        for chunk in chunks {
            let result = parser.parse_reasoning_streaming_incremental(chunk, &[]);
            normal_text_incr.push_str(&result.normal_text);
            reasoning_text_incr.push_str(&result.reasoning_text);
        }
        assert_eq!(reasoning_text_incr, "thinking");
        assert_eq!(normal_text_incr, "answer");
    }

    #[test]
    fn test_gpt_oss_reasoning_parser_streaming_flushes_pending_suffix_on_finish() {
        let mut parser = GptOssReasoningParser::new().expect("Failed to create parser");
        let result = parser.parse_reasoning_streaming_incremental(
            "<|channel|>analysis<|message|>thinking<|e",
            &[],
        );
        let finish = parser.finish_reasoning_stream();
        assert_eq!(result.reasoning_text, "thinking");
        assert_eq!(finish.reasoning_text, "<|e");
        assert_eq!(finish.normal_text, "");
        let finish = parser.finish_reasoning_stream();
        assert_eq!(finish.reasoning_text, "");
        assert_eq!(finish.normal_text, "");

        let mut parser = GptOssReasoningParser::new().expect("Failed to create parser");
        let result = parser
            .parse_reasoning_streaming_incremental("<|channel|>final<|message|>answer<|e", &[]);
        let finish = parser.finish_reasoning_stream();
        assert_eq!(result.normal_text, "answer");
        assert_eq!(finish.normal_text, "<|e");
        assert_eq!(finish.reasoning_text, "");
    }

    #[test] // REASONING.stream.2.b
    fn test_gpt_oss_reasoning_parser_streaming_multiple_analysis_spans() {
        let mut parser = GptOssReasoningParser::new().expect("Failed to create parser");
        let chunks = vec![
            "<|channel|>analysis<|message|>first<|end|><|start|>assistant",
            "<|channel|>analysis<|message|>second<|end|><|start|>assistant",
            "<|channel|>final<|message|>done",
        ];
        let mut reasoning_text_incr = String::new();
        let mut normal_text_incr = String::new();
        for chunk in chunks {
            let result = parser.parse_reasoning_streaming_incremental(chunk, &[]);
            normal_text_incr.push_str(&result.normal_text);
            reasoning_text_incr.push_str(&result.reasoning_text);
        }
        assert_eq!(reasoning_text_incr, "first\nsecond");
        assert_eq!(normal_text_incr, "done");
    }

    #[test]
    fn test_gpt_oss_reasoning_parser_streaming_recipientless_commentary() {
        let mut parser = GptOssReasoningParser::new().expect("Failed to create parser");
        let chunks = vec![
            "<|channel|>commentary<|message|>I will check that.<|end|><|start|>assistant",
            "<|channel|>final<|message|>Done.",
        ];
        let mut reasoning_text_incr = String::new();
        let mut normal_text_incr = String::new();
        for chunk in chunks {
            let result = parser.parse_reasoning_streaming_incremental(chunk, &[]);
            normal_text_incr.push_str(&result.normal_text);
            reasoning_text_incr.push_str(&result.reasoning_text);
        }
        assert_eq!(reasoning_text_incr, "");
        assert_eq!(normal_text_incr, "I will check that.\nDone.");
    }

    #[test]
    fn test_gpt_oss_reasoning_parser_streaming_split_directed_commentary() {
        let mut parser = GptOssReasoningParser::new().expect("Failed to create parser");
        let chunks = vec![
            "<|channel|>commentary to=functions.get_",
            "weather <|constrain|>json<|message|>{\"city\":\"SF\"}<|call|>",
        ];
        let mut reasoning_text_incr = String::new();
        let mut normal_text_incr = String::new();
        for chunk in chunks {
            let result = parser.parse_reasoning_streaming_incremental(chunk, &[]);
            normal_text_incr.push_str(&result.normal_text);
            reasoning_text_incr.push_str(&result.reasoning_text);
        }
        assert_eq!(reasoning_text_incr, "");
        assert_eq!(
            normal_text_incr,
            "<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{\"city\":\"SF\"}<|call|>"
        );
    }

    #[test] // REASONING.stream.2.a, REASONING.batch.2.c, TOOLCALLING.harmony.1
    fn test_gpt_oss_reasoning_parser_streaming_chunked() {
        let mut parser = GptOssReasoningParser::new().expect("Failed to create parser");
        let enc = get_harmony_encoding()
            .as_ref()
            .expect("Failed to get encoding");
        let text = "<|channel|>analysis<|message|>The user asks a simple factual question: capital of Brazil. The answer is Brasília. No additional explanation needed.<|end|><|start|>assistant<|channel|>final<|message|>The capital of Brazil is Brasília.";
        let token_ids = enc.tokenizer().encode_with_special_tokens(text);
        let mut reasoning_text_incr = String::new();
        let mut normal_text_incr = String::new();

        let mut idx = 0;
        let chunk_size = 4;
        while idx < token_ids.len() {
            let end = (idx + chunk_size).min(token_ids.len());
            let result =
                parser.parse_reasoning_streaming_incremental("Test text", &token_ids[idx..end]);
            normal_text_incr.push_str(&result.normal_text);
            reasoning_text_incr.push_str(&result.reasoning_text);
            idx = end;
        }

        assert_eq!(normal_text_incr, "The capital of Brazil is Brasília.");
        assert_eq!(
            reasoning_text_incr,
            "The user asks a simple factual question: capital of Brazil. The answer is Brasília. No additional explanation needed."
        );
    }

    #[test] // REASONING.batch.3.a, TOOLCALLING.harmony.1
    fn test_gpt_oss_reasoning_parser_streaming_variable_length_chunks() {
        let text = "<|channel|>analysis<|message|>User asks: \"Hey, quick check: is everything up and running?\" We should check system health using the provided function get_system_health. Use function.<|end|><|start|>assistant<|channel|>commentary to=functions.get_system_health <|constrain|>json<|message|>{}";
        let enc = get_harmony_encoding()
            .as_ref()
            .expect("Failed to get encoding");
        let token_ids = enc.tokenizer().encode_with_special_tokens(text);

        // Send token one by one
        {
            let mut parser = GptOssReasoningParser::new().expect("Failed to create parser");
            let mut reasoning_text_incr = String::new();
            let mut normal_text_incr = String::new();
            for token in token_ids.iter() {
                let result = parser.parse_reasoning_streaming_incremental("", &[(*token)]);
                normal_text_incr.push_str(&result.normal_text);
                reasoning_text_incr.push_str(&result.reasoning_text);
            }
            assert_eq!(
                reasoning_text_incr,
                "User asks: \"Hey, quick check: is everything up and running?\" We should check system health using the provided function get_system_health. Use function."
            );
            // This input is a TRUNCATED tool call: it ends at `<|message|>{}` with
            // no `<|call|>` terminator. A complete directed tool call is handed off
            // to the jail in one piece (header + body + `<|call|>`) on the `<|call|>`
            // chunk; without a terminator there is no complete envelope to recover,
            // so the streaming parser emits no normal_text rather than leaking a
            // partial `<|channel|>…<|message|>` header.
            assert_eq!(normal_text_incr, "");
        }

        // Send token in chunks (chunking obtained from actual model output)
        {
            let mut parser = GptOssReasoningParser::new().expect("Failed to create parser");
            let mut reasoning_text_incr = String::new();
            let mut normal_text_incr = String::new();
            let chunk_tokens = [
                vec![200005],
                vec![35644, 200008, 1844, 31064, 25, 392, 25216, 11, 4853],
                vec![2371, 25, 382, 5519, 869, 326, 6788, 16842, 1416, 1757],
                vec![2371, 2420, 3230, 2360, 290, 5181, 1114, 717, 39303, 126214],
                vec![
                    13, 7649, 1114, 13, 200007, 200006, 173781, 200005, 12606, 815,
                ],
                vec![
                    316, 28, 44580, 775, 39303, 126214, 220, 200003, 4108, 200008,
                ],
                vec![12083],
            ];
            // Concatenate chunk tokens and verify they match original token_ids
            let concatenated: Vec<u32> = chunk_tokens.iter().flatten().copied().collect();
            assert_eq!(concatenated, token_ids);

            for token in chunk_tokens.iter() {
                let result = parser.parse_reasoning_streaming_incremental("", token);
                normal_text_incr.push_str(&result.normal_text);
                reasoning_text_incr.push_str(&result.reasoning_text);
            }
            assert_eq!(
                reasoning_text_incr,
                "User asks: \"Hey, quick check: is everything up and running?\" We should check system health using the provided function get_system_health. Use function."
            );
            // Same truncated-call contract as the token-by-token case above:
            // no <|call|> terminator → no complete envelope → no normal_text.
            assert_eq!(normal_text_incr, "");
        }
    }

    /// `<|call|>` and the following `final` text arrive in the **same chunk**.
    /// Before the fix, `reconstruct_directed_envelope` ran after the whole token
    /// loop and `rposition` would find the `final` channel's `<|channel|>` token
    /// (the last one in the buffer) instead of the tool call's, producing the wrong
    /// envelope.  The in-loop reconstruction fixes this by calling
    /// `reconstruct_directed_envelope` the moment `<|call|>` resets the parser,
    /// before any subsequent tokens are appended to the buffer.
    #[test] // TOOLCALLING.harmony.same-chunk-call-and-final
    fn test_gpt_oss_streaming_call_and_final_in_same_chunk() {
        let mut parser = GptOssReasoningParser::new().expect("Failed to create parser");
        let chunks = vec![
            "<|channel|>analysis to=functions.search code<|message|>",
            // `<|call|>` and `<|channel|>final` land in one chunk — this is the
            // scenario that the original post-loop reconstruction mis-handled.
            r#"{"q":"foo"}<|call|><|start|>assistant<|channel|>final<|message|>done"#,
        ];
        let mut normal_text = String::new();
        let mut reasoning_text = String::new();
        for chunk in chunks {
            let result = parser.parse_reasoning_streaming_incremental(chunk, &[]);
            normal_text.push_str(&result.normal_text);
            reasoning_text.push_str(&result.reasoning_text);
        }
        assert_eq!(reasoning_text, "");
        assert!(
            normal_text.contains("analysis to=functions.search"),
            "tool call envelope must be present: {normal_text:?}"
        );
        assert!(
            normal_text.contains("<|call|>"),
            "call terminator must be present: {normal_text:?}"
        );
        assert!(
            normal_text.ends_with("done"),
            "final text must follow tool call: {normal_text:?}"
        );
        let call_pos = normal_text.find("<|call|>").unwrap();
        let done_pos = normal_text.find("done").unwrap();
        assert!(
            call_pos < done_pos,
            "tool call must precede final text in output"
        );
    }

    /// Two directed tool calls arriving in a single chunk.  The original code had
    /// only one post-loop reconstruction path, so the second `<|call|>` clobbered
    /// the first.  The in-loop per-`<|call|>` reconstruction handles each
    /// terminator independently and preserves both envelopes.
    #[test] // TOOLCALLING.harmony.two-calls-one-chunk
    fn test_gpt_oss_streaming_two_calls_in_one_chunk() {
        let mut parser = GptOssReasoningParser::new().expect("Failed to create parser");
        let chunks = vec![
            // Both tool calls arrive together.
            "<|channel|>analysis to=functions.search code<|message|>{\"q\":\"x\"}<|call|><|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{\"city\":\"SF\"}<|call|>",
        ];
        let mut normal_text = String::new();
        let mut reasoning_text = String::new();
        for chunk in chunks {
            let result = parser.parse_reasoning_streaming_incremental(chunk, &[]);
            normal_text.push_str(&result.normal_text);
            reasoning_text.push_str(&result.reasoning_text);
        }
        assert_eq!(reasoning_text, "");
        assert!(
            normal_text.contains("analysis to=functions.search"),
            "first tool call must be present: {normal_text:?}"
        );
        assert!(
            normal_text.contains("commentary to=functions.get_weather"),
            "second tool call must be present: {normal_text:?}"
        );
        // First call must precede second call in the output.
        let first_pos = normal_text.find("search").unwrap();
        let second_pos = normal_text.find("get_weather").unwrap();
        assert!(first_pos < second_pos, "calls must appear in stream order");
    }

    #[test] // REASONING.stream.4.d
    fn test_gpt_oss_reasoning_parser_streaming_split_directed_analysis() {
        // Header arrives in chunk 1; body + <|call|> arrive in chunk 2.
        // This is the exact case last_directed_channel / reconstruct_directed_envelope
        // is designed for: the parser state resets on <|call|>, so the channel
        // and recipient must be persisted from the earlier chunk.
        let mut parser = GptOssReasoningParser::new().expect("Failed to create parser");
        let chunks = vec![
            "<|channel|>analysis to=functions.grep code<|message|>",
            r#"{"pattern":"foo"}<|call|>"#,
        ];
        let mut reasoning_text_incr = String::new();
        let mut normal_text_incr = String::new();
        for chunk in chunks {
            let result = parser.parse_reasoning_streaming_incremental(chunk, &[]);
            normal_text_incr.push_str(&result.normal_text);
            reasoning_text_incr.push_str(&result.reasoning_text);
        }
        assert_eq!(reasoning_text_incr, "");
        assert!(
            normal_text_incr.contains("analysis to=functions.grep"),
            "channel header must survive into normal_text: {normal_text_incr:?}"
        );
        assert!(
            normal_text_incr.ends_with("<|call|>"),
            "terminator must be present: {normal_text_incr:?}"
        );
    }

    // Malformed-harmony recovery (issue #237), cases A–D.
    //
    // A gpt-oss model that emits `<|channel|>final` followed by anything other
    // than `<|message|>` used to lose its entire final message: the frontend
    // answered `content: null` with `finish_reason: "stop"`, indistinguishable
    // from a legitimately empty answer. The four fixtures below are the control
    // (A), the two malformed shapes (B, C), and the contrast case where empty
    // content is genuinely correct (D).

    const ANALYSIS_PREFIX: &str =
        "<|channel|>analysis<|message|>User says hi. Answer briefly.<|end|><|start|>assistant";
    const EXPECTED_REASONING: &str = "User says hi. Answer briefly.";
    /// Case A: well-formed.
    const CASE_A: &str = "<|channel|>analysis<|message|>User says hi. Answer briefly.<|end|><|start|>assistant<|channel|>final<|message|>Hello!";
    /// Case B: stray token where `<|message|>` belongs. Mimics the observed
    /// token `164797` (`*)__`) in place of `200008` (`<|message|>`), which
    /// strands harmony in `StreamState::Header`.
    const CASE_B: &str = "<|channel|>analysis<|message|>User says hi. Answer briefly.<|end|><|start|>assistant<|channel|>final*)__Hello!";
    /// Case C: mistyped channel name, delimiter intact. Harmony parses this
    /// happily into `Content`; the loss is this parser refusing the channel.
    const CASE_C: &str = "<|channel|>analysis<|message|>User says hi. Answer briefly.<|end|><|start|>assistant<|channel|>finalx<|message|>Hello!";
    /// Case D: no final message at all. Empty content is the correct answer.
    const CASE_D: &str = "<|channel|>analysis<|message|>User says hi. Answer briefly.";

    /// Stream `text` in `chunk_size`-character chunks (0 = one whole chunk),
    /// finalize, and return the accumulated `(normal_text, reasoning_text)`.
    fn stream_with_finish(text: &str, chunk_size: usize) -> (String, String) {
        let mut parser = GptOssReasoningParser::new().expect("Failed to create parser");
        let mut normal = String::new();
        let mut reasoning = String::new();
        let chars: Vec<char> = text.chars().collect();
        let chunks: Vec<String> = if chunk_size == 0 {
            vec![text.to_string()]
        } else {
            chars
                .chunks(chunk_size)
                .map(|c| c.iter().collect::<String>())
                .collect()
        };
        for chunk in chunks {
            let result = parser.parse_reasoning_streaming_incremental(&chunk, &[]);
            normal.push_str(&result.normal_text);
            reasoning.push_str(&result.reasoning_text);
        }
        let finished = parser.finish_reasoning_stream();
        normal.push_str(&finished.normal_text);
        reasoning.push_str(&finished.reasoning_text);
        (normal, reasoning)
    }

    #[test] // issue #237
    fn test_gpt_oss_finish_recovers_final_without_message_delimiter() {
        // Every chunk size must recover: a chunk boundary inside the malformed
        // header must not change the outcome.
        for chunk_size in [1usize, 20, 0] {
            let (normal, reasoning) = stream_with_finish(CASE_B, chunk_size);
            assert!(
                normal.contains("Hello!"),
                "chunk_size={chunk_size}: final message must be recovered, got {normal:?}"
            );
            assert!(
                !normal.contains("<|"),
                "chunk_size={chunk_size}: recovery must not leak harmony markup, got {normal:?}"
            );
            assert_eq!(
                reasoning, EXPECTED_REASONING,
                "chunk_size={chunk_size}: reasoning must be unaffected"
            );
        }
    }

    #[test] // issue #237
    fn test_gpt_oss_finish_recovers_malformed_channel() {
        // Case C is a distinct failure mode from case B: harmony reaches
        // `Content` and decodes the text fine, but the channel name is one this
        // parser does not recognize, so no arm claimed the delta.
        for chunk_size in [1usize, 20, 0] {
            let (normal, reasoning) = stream_with_finish(CASE_C, chunk_size);
            assert_eq!(
                normal.trim(),
                "Hello!",
                "chunk_size={chunk_size}: content under an unknown channel must not be dropped"
            );
            assert_eq!(reasoning, EXPECTED_REASONING, "chunk_size={chunk_size}");
        }
    }

    #[test] // issue #237
    fn test_gpt_oss_finish_is_noop_for_analysis_only() {
        // The contrast case: the model really did stop after reasoning. An
        // empty `content` is the honest answer and recovery must not invent one.
        for chunk_size in [1usize, 20, 0] {
            let (normal, reasoning) = stream_with_finish(CASE_D, chunk_size);
            assert_eq!(
                normal, "",
                "chunk_size={chunk_size}: must not fabricate content, got {normal:?}"
            );
            assert_eq!(reasoning, EXPECTED_REASONING, "chunk_size={chunk_size}");
        }
    }

    #[test] // issue #237
    fn test_gpt_oss_finish_does_not_duplicate_already_emitted_final() {
        // `process_eos()` closes the `final` message on a well-formed stream.
        // Collecting from `messages()` there would re-emit text the caller has
        // already been handed, so the `Ok` path must contribute nothing.
        for chunk_size in [1usize, 20, 0] {
            let (normal, reasoning) = stream_with_finish(CASE_A, chunk_size);
            assert_eq!(
                normal, "Hello!",
                "chunk_size={chunk_size}: finish must add nothing already streamed"
            );
            assert_eq!(reasoning, EXPECTED_REASONING, "chunk_size={chunk_size}");
        }
    }

    #[test] // issue #237
    fn test_gpt_oss_finish_is_idempotent() {
        // The dynamo preprocessor finalizes from more than one path now, so a
        // second call must be inert rather than re-emitting recovered text.
        for text in [CASE_A, CASE_B, CASE_C, CASE_D] {
            let mut parser = GptOssReasoningParser::new().expect("Failed to create parser");
            let _ = parser.parse_reasoning_streaming_incremental(text, &[]);
            let _first = parser.finish_reasoning_stream();
            let second = parser.finish_reasoning_stream();
            assert_eq!(
                second.normal_text, "",
                "second finish must be empty: {text:?}"
            );
            assert_eq!(
                second.reasoning_text, "",
                "second finish must be empty: {text:?}"
            );
        }
    }

    #[test] // REASONING.batch.4 (issue #237)
    fn test_gpt_oss_malformed_header_batch_matches_streaming() {
        for text in [CASE_A, CASE_B, CASE_C, CASE_D] {
            let (stream_normal, stream_reasoning) = stream_with_finish(text, 0);
            let mut parser = GptOssReasoningParser::new().expect("Failed to create parser");
            let batch = parser.detect_and_parse_reasoning(text, &[]);
            assert_eq!(
                batch.normal_text, stream_normal,
                "batch/stream normal_text parity for {text:?}"
            );
            assert_eq!(
                batch.reasoning_text, stream_reasoning,
                "batch/stream reasoning_text parity for {text:?}"
            );
        }
    }

    #[test] // issue #237
    fn test_gpt_oss_finish_keeps_stranded_analysis_as_reasoning() {
        // A stranded `analysis` header holds reasoning, not the answer. Moving
        // it into `content` would show the model's chain of thought to the user.
        let text = "<|channel|>analysis*)__Thinking it over";
        for chunk_size in [1usize, 20, 0] {
            let (normal, reasoning) = stream_with_finish(text, chunk_size);
            assert_eq!(normal, "", "chunk_size={chunk_size}");
            assert_eq!(reasoning, "*)__Thinking it over", "chunk_size={chunk_size}");
        }
        let mut parser = GptOssReasoningParser::new().expect("Failed to create parser");
        let batch = parser.detect_and_parse_reasoning(text, &[]);
        assert_eq!(batch.normal_text, "");
        assert_eq!(batch.reasoning_text, "*)__Thinking it over");
    }

    #[test] // issue #237
    fn test_gpt_oss_finish_does_not_surface_stranded_tool_call_as_content() {
        // A directed tool-call header without `<|message|>` is a malformed tool
        // envelope. Its recipient and raw arguments must not reach `content`.
        let text = format!(
            "{ANALYSIS_PREFIX}<|channel|>commentary to=functions.get_weather <|constrain|>json{{\"location\":\"NYC\"}}<|call|>"
        );
        for chunk_size in [1usize, 20, 0] {
            let mut parser = GptOssReasoningParser::new().expect("Failed to create parser");
            let chars: Vec<char> = text.chars().collect();
            let step = if chunk_size == 0 {
                chars.len()
            } else {
                chunk_size
            };
            for chunk in chars.chunks(step) {
                let _ = parser
                    .parse_reasoning_streaming_incremental(&chunk.iter().collect::<String>(), &[]);
            }
            let finished = parser.finish_reasoning_stream();
            assert_eq!(
                finished.normal_text, "",
                "chunk_size={chunk_size}: stranded tool header must not be recovered as content"
            );
        }
    }

    #[test] // issue #237
    fn test_gpt_oss_unknown_channel_directed_payload_is_not_content() {
        // A mistyped channel with a `to=functions.*` recipient is a tool
        // envelope. Recovering it as the answer would hand the client raw tool
        // arguments, so the unknown-channel path must drop it just as the
        // stranded-header path refuses `to=` headers.
        let completed = format!(
            "{ANALYSIS_PREFIX}<|channel|>finalx to=functions.get_weather <|constrain|>json<|message|>{{\"location\":\"NYC\"}}<|call|>"
        );
        let truncated = format!(
            "{ANALYSIS_PREFIX}<|channel|>finalx to=functions.get_weather <|constrain|>json<|message|>{{\"location\":\"NYC\"}}"
        );
        for text in [completed.as_str(), truncated.as_str()] {
            // Only `finish` is asserted on: a chunk carrying `<|call|>` still
            // goes through the raw tool-parser handoff, which is unrelated.
            for chunk_size in [1usize, 20, 0] {
                let mut parser = GptOssReasoningParser::new().expect("Failed to create parser");
                let chars: Vec<char> = text.chars().collect();
                let step = if chunk_size == 0 {
                    chars.len()
                } else {
                    chunk_size
                };
                for chunk in chars.chunks(step) {
                    let _ = parser.parse_reasoning_streaming_incremental(
                        &chunk.iter().collect::<String>(),
                        &[],
                    );
                }
                let finished = parser.finish_reasoning_stream();
                assert_eq!(
                    finished.normal_text, "",
                    "chunk_size={chunk_size}: directed payload under an unknown channel must not become content: {text:?}"
                );
            }
            let mut parser = GptOssReasoningParser::new().expect("Failed to create parser");
            let batch = parser.detect_and_parse_reasoning(text, &[]);
            assert_eq!(
                batch.normal_text, "",
                "batch: directed payload under an unknown channel must not become content: {text:?}"
            );
            assert_eq!(batch.reasoning_text, EXPECTED_REASONING);
        }
    }

    #[test] // issue #237
    fn test_gpt_oss_recovery_keeps_leading_role_word_without_start() {
        // The parser is constructed with `Role::Assistant`, so it starts in
        // `Header` with no `<|start|>`. Plain output that happens to begin with
        // a role name must come back whole, not with that word stripped.
        for text in [
            "tools are configured per request.",
            "users can override this.",
            "system prompts are optional.",
            "developers own the template.",
        ] {
            for chunk_size in [1usize, 20, 0] {
                let (normal, reasoning) = stream_with_finish(text, chunk_size);
                assert_eq!(normal, text, "chunk_size={chunk_size}");
                assert_eq!(reasoning, "", "chunk_size={chunk_size}");
            }
            let mut parser = GptOssReasoningParser::new().expect("Failed to create parser");
            let batch = parser.detect_and_parse_reasoning(text, &[]);
            assert_eq!(batch.normal_text, text);
            assert_eq!(batch.reasoning_text, "");
        }
    }

    #[test] // issue #237
    fn test_gpt_oss_recovery_strips_role_after_start() {
        // After `<|start|>` the role word is header markup, not output.
        let text = format!("{ANALYSIS_PREFIX}*)__Hello!");
        for chunk_size in [1usize, 20, 0] {
            let (normal, reasoning) = stream_with_finish(&text, chunk_size);
            assert_eq!(normal, "*)__Hello!", "chunk_size={chunk_size}");
            assert_eq!(reasoning, EXPECTED_REASONING, "chunk_size={chunk_size}");
        }
    }

    #[test] // REASONING.batch.4 (issue #237)
    fn test_gpt_oss_batch_recovery_does_not_duplicate_raw_tool_handoff() {
        // With a directed call present, batch hands the raw text to the tool
        // parser. That text already holds the stranded `final` header, so
        // recovering it again would emit the answer twice.
        let text = "<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{\"location\":\"NYC\"}<|call|><|start|>assistant<|channel|>final*)__Done";
        let mut parser = GptOssReasoningParser::new().expect("Failed to create parser");
        let batch = parser.detect_and_parse_reasoning(text, &[]);
        assert_eq!(
            batch.normal_text.matches("*)__Done").count(),
            1,
            "recovered answer must appear exactly once: {:?}",
            batch.normal_text
        );
    }
}
