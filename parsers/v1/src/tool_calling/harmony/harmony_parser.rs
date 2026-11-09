// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::super::ToolDefinition;
use super::config::JsonParserConfig;
use super::payload::{HARMONY_MARKERS, HarmonyVisible};
use super::response::{CalledFunction, ToolCallResponse, ToolCallType};
use openai_harmony::chat::{Content, Role};
use openai_harmony::{
    HarmonyEncoding, HarmonyEncodingName, StreamableParser, load_harmony_encoding,
};
use regex::Regex;
use std::sync::OnceLock;
use uuid::Uuid;

static COMMENTARY_HEADER_PREFIX_REGEX: OnceLock<Regex> = OnceLock::new();
static COMMENTARY_BLOCK_CLEANUP_REGEX: OnceLock<Regex> = OnceLock::new();
static COMMENTARY_HEADER_CLEANUP_REGEX: OnceLock<Regex> = OnceLock::new();
static ANALYSIS_BLOCK_CLEANUP_REGEX: OnceLock<Regex> = OnceLock::new();
static FINAL_BLOCK_CLEANUP_REGEX: OnceLock<Regex> = OnceLock::new();
static SPECIAL_TOKEN_REGEX: OnceLock<Regex> = OnceLock::new();

// Locate only the routing prefix: native header metadata owns the role and
// complete recipient, including opaque names that are not regex identifiers.
const COMMENTARY_HEADER_PREFIX: &str = r"(?s)(?:(?:<\|start\|>assistant)?<\|channel\|>(?:commentary|analysis)[ \t\r\n\x0B\x0C]+to=functions\.|(?:<\|start\|>assistant\s+)?to=functions\.)";

fn commentary_header_prefix_regex() -> &'static Regex {
    COMMENTARY_HEADER_PREFIX_REGEX
        .get_or_init(|| Regex::new(COMMENTARY_HEADER_PREFIX).expect("header prefix regex"))
}

fn bounded_harmony_header(text: &str, mut start: usize) -> Option<std::ops::Range<usize>> {
    loop {
        let prefix = commentary_header_prefix_regex().find_at(text, start)?;
        let mut channel = prefix.as_str().contains("<|channel|>");
        for marker in special_token_regex().find_iter(&text[prefix.end()..]) {
            match marker.as_str() {
                "<|message|>" if channel => {
                    return Some(prefix.start()..prefix.end() + marker.end());
                }
                "<|channel|>" if !channel => channel = true,
                "<|start|>" | "<|channel|>" | "<|message|>" => break,
                _ => {}
            }
        }
        start = prefix.end();
    }
}

fn commentary_block_cleanup_regex() -> &'static Regex {
    COMMENTARY_BLOCK_CLEANUP_REGEX.get_or_init(|| {
        Regex::new(
            r"(?s)(?:<\|start\|>assistant)?<\|channel\|>commentary(?:\s+to=functions\.(?P<name>[\w.\-]+))?.*?<\|message\|>.*?(?:<\|(?:call|return)\|>|\z)",
        )
        .expect("commentary block cleanup regex")
    })
}

fn commentary_header_cleanup_regex() -> &'static Regex {
    COMMENTARY_HEADER_CLEANUP_REGEX.get_or_init(|| {
        Regex::new(
            r"(?s)(?:<\|start\|>assistant)?<\|channel\|>commentary(?:\s+to=functions\.(?P<name>[\w.\-]+))?.*\z",
        )
        .expect("commentary header cleanup regex")
    })
}

fn analysis_block_cleanup_regex() -> &'static Regex {
    ANALYSIS_BLOCK_CLEANUP_REGEX.get_or_init(|| {
        // Accepts both the spec-compliant reasoning form
        //   `<|channel|>analysis<|message|>...<|end|>`
        // and the gpt-oss model-bug directed-tool-call form (recovered by PR #10366)
        //   `<|channel|>analysis to=functions.X[ <|constrain|>json]<|message|>{args}<|call|>`
        Regex::new(
            r"(?s)(?:<\|start\|>assistant)?<\|channel\|>analysis(?:\s+to=functions\.(?P<name>[\w.\-]+))?.*?<\|message\|>(?P<body>.*?)(?:<\|(?:call|return)\|>|<\|end\|>|\z)",
        )
        .expect("analysis block cleanup regex")
    })
}

fn final_block_cleanup_regex() -> &'static Regex {
    FINAL_BLOCK_CLEANUP_REGEX.get_or_init(|| {
        Regex::new(r"(?s)(?:<\|start\|>assistant)?<\|channel\|>final<\|message\|>(?P<body>.*?)(?:<\|return\|>|<\|end\|>|\z)")
            .expect("final block cleanup regex")
    })
}

fn special_token_regex() -> &'static Regex {
    SPECIAL_TOKEN_REGEX.get_or_init(|| {
        let pattern = HARMONY_MARKERS
            .iter()
            .map(|marker| regex::escape(marker))
            .collect::<Vec<_>>()
            .join("|");
        Regex::new(&pattern).expect("special token cleanup regex")
    })
}

pub(crate) fn json_value_prefix_end(text: &str) -> Option<usize> {
    let mut json =
        serde_json::Deserializer::from_str(text).into_iter::<&serde_json::value::RawValue>();
    matches!(json.next(), Some(Ok(_))).then(|| json.byte_offset())
}

pub(crate) fn harmony_json_prelude(text: &str) -> bool {
    // Scalar words and numbers are also native implicit recipient names.
    // Structured values and quoted strings instead start the visible JSON grammar.
    matches!(
        text.trim_start().as_bytes().first(),
        Some(b'{' | b'[' | b'"')
    )
}

pub(crate) fn harmony_header_is_directed(header: &str) -> bool {
    header
        .split_whitespace()
        .any(|part| part.starts_with("to="))
}

// Directed malformed arguments retain raw recovery. Recipientless and projected
// visible JSON keep their markers as data even before the JSON value is complete.
pub(crate) fn harmony_markers(text: &str) -> impl Iterator<Item = (regex::Match<'_>, bool)> {
    let mut json_end = 0;
    let mut visible = Some(HarmonyVisible::default());
    let mut cursor = 0;
    let mut header_start = 0;
    let mut header_resume = 0;
    let mut header_pending = true;
    special_token_regex().find_iter(text).map(move |marker| {
        let mut visible_json = false;
        if let Some(body) = &mut visible {
            body.push_bytes(&text.as_bytes()[cursor..marker.start()]);
            visible_json = body.in_string();
            if visible_json {
                body.push_bytes(marker.as_str().as_bytes());
            }
        }
        let in_json = marker.start() < json_end || visible_json;
        if !in_json {
            match marker.as_str() {
                "<|start|>" => {
                    header_start = marker.start();
                    header_resume = marker.start();
                    header_pending = true;
                    visible = None;
                }
                "<|channel|>" => {
                    if !header_pending {
                        header_resume = header_start;
                        header_start = marker.start();
                        header_pending = true;
                    }
                    visible = None;
                }
                "<|message|>" => {
                    let directed = harmony_header_is_directed(&text[header_start..marker.start()])
                        || (header_resume < header_start
                            && std::iter::from_fn({
                                let span = &text[header_resume..marker.end()];
                                let mut start = 0;
                                move || {
                                    let header = bounded_harmony_header(span, start)?;
                                    start = header.end;
                                    Some(header)
                                }
                            })
                            .last()
                            .is_some_and(|header| header.end == marker.end() - header_resume));
                    visible = (!directed).then(HarmonyVisible::default);
                    json_end = if directed {
                        marker.end()
                            + json_value_prefix_end(&text[marker.end()..]).unwrap_or_default()
                    } else {
                        0
                    };
                    header_pending = false;
                    header_start = marker.end();
                    header_resume = marker.end();
                }
                "<|call|>" | "<|return|>" | "<|end|>" => {
                    visible = (!header_pending).then(HarmonyVisible::default);
                    header_start = marker.end();
                    header_resume = marker.end();
                    header_pending = true;
                    json_end = 0;
                }
                _ => {}
            }
        }
        cursor = marker.end();
        (marker, in_json)
    })
}

#[derive(Debug)]
pub(crate) struct HarmonyToolHeader {
    pub(crate) start: usize,
    pub(crate) end: usize,
    pub(crate) name: String,
    pub(crate) channel: Option<String>,
    pub(crate) recipient_first: bool,
}

pub(crate) fn parse_harmony_header(
    encoding: &HarmonyEncoding,
    tokens: &[u32],
    start_token: u32,
) -> Option<StreamableParser> {
    #[cfg(test)]
    HARMONY_HEADER_PARSES.with(|count| count.set(count.get() + 1));
    let mut header =
        StreamableParser::new(encoding.clone(), harmony_initial_role(tokens, start_token)).ok()?;
    for token in tokens {
        header.process(*token).ok()?;
    }
    Some(header)
}

fn harmony_role_word(text: &str) -> Option<std::ops::Range<usize>> {
    let role = text
        .strip_prefix("<|start|>")?
        .trim_start_matches(|ch: char| ch.is_ascii_whitespace());
    let start = text.len() - role.len();
    let word_end = role
        .find(|ch: char| ch.is_ascii_whitespace())
        .unwrap_or(role.len());
    let marker_end = special_token_regex()
        .find(role)
        .map_or(role.len(), |marker| marker.start());
    Some(start..start + word_end.min(marker_end))
}

fn parse_harmony_source_header(encoding: &HarmonyEncoding, text: &str) -> Option<StreamableParser> {
    let start = *encoding
        .tokenizer()
        .encode_with_special_tokens("<|start|>")
        .first()?;
    let mut tokens = encoding.tokenizer().encode_with_special_tokens(text);
    if !text.ends_with("<|message|>") {
        tokens.extend(
            encoding
                .tokenizer()
                .encode_with_special_tokens("<|message|>"),
        );
    }
    parse_harmony_header(encoding, &tokens, start)
}

pub(crate) fn harmony_header_ownership(
    encoding: &HarmonyEncoding,
    text: &str,
) -> (Option<Role>, bool) {
    let parse_role = |text: &str| parse_harmony_source_header(encoding, text)?.current_role();
    if let Some(role) = parse_role(text) {
        return (Some(role), true);
    }
    let role = harmony_role_word(text).and_then(|word| parse_role(&text[..word.end]));
    // Issue 237 also recovers raw text appended directly to the assistant role,
    // without any later header marker. It cannot establish a callable header.
    if role.is_none()
        && text.starts_with("<|start|>assistant")
        && special_token_regex()
            .find_iter(text)
            .all(|marker| marker.as_str() == "<|start|>")
    {
        return (Some(Role::Assistant), false);
    }
    (role, false)
}

pub(crate) fn harmony_header_role(encoding: &HarmonyEncoding, text: &str) -> Option<Role> {
    harmony_header_ownership(encoding, text).0
}

pub(crate) const HARMONY_KNOWN_CHANNELS: &[&str] = &["analysis", "commentary", "final"];

pub(crate) fn harmony_header_eof_is_metadata(encoding: &HarmonyEncoding, text: &str) -> bool {
    let mut has_header_marker = false;
    for (marker, quoted) in harmony_markers(text) {
        if quoted {
            continue;
        }
        if marker.as_str() == "<|message|>" {
            return false;
        }
        has_header_marker |= matches!(marker.as_str(), "<|start|>" | "<|channel|>");
    }
    if !has_header_marker {
        return false;
    }
    // Issue 237's recorded stray token replaced Message, leaving recoverable
    // prose joined to a known channel. An ordinary channel extension such as
    // finalx is metadata, not the visible suffix x.
    if text.rsplit_once("<|channel|>").is_some_and(|(_, channel)| {
        HARMONY_KNOWN_CHANNELS.iter().any(|known| {
            channel
                .strip_prefix(known)
                .is_some_and(|suffix| suffix.starts_with("*)__"))
        })
    }) {
        return false;
    }
    parse_harmony_source_header(encoding, text).is_some()
        || text
            .rsplit_once("<|channel|>")
            .is_some_and(|(prefix, channel)| {
                channel
                    .chars()
                    .next()
                    .is_none_or(|ch| ch.is_whitespace() || ch == '<')
                    && parse_harmony_source_header(encoding, &format!("{prefix}{channel}"))
                        .is_some()
            })
}

struct HarmonyRoleSpan {
    start: usize,
    end: usize,
    role: Option<Role>,
    header_end: Option<usize>,
    parsed: Option<StreamableParser>,
}

fn harmony_framed_roles(text: &str, encoding: Option<&HarmonyEncoding>) -> Vec<HarmonyRoleSpan> {
    let markers: Vec<_> = harmony_markers(text)
        .filter_map(|(marker, quoted)| (!quoted).then_some(marker))
        .collect();
    let mut spans = Vec::new();
    let mut active: Option<HarmonyRoleSpan> = None;
    let mut in_body = false;
    for (index, marker) in markers.iter().enumerate() {
        if marker.as_str() == "<|start|>" {
            let end = markers[index + 1..]
                .iter()
                .find(|next| matches!(next.as_str(), "<|message|>" | "<|start|>"));
            // An unconfirmed header inside a body remains raw argument data;
            // only its actual message delimiter can transfer role ownership.
            if in_body && end.is_none_or(|end| end.as_str() != "<|message|>") {
                continue;
            }
            if let Some(mut previous) = active.take() {
                previous.end = marker.start();
                spans.push(previous);
            }
            in_body = false;
            let header_end = end.map_or(text.len(), |end| {
                if end.as_str() == "<|message|>" {
                    end.end()
                } else {
                    end.start()
                }
            });
            // Named tool authors require the remaining header parts for native
            // role classification; do not classify only their first word.
            let source_header = &text[marker.start()..header_end];
            let parsed =
                encoding.and_then(|encoding| parse_harmony_source_header(encoding, source_header));
            let (role, native_complete) = match encoding {
                Some(_) if parsed.is_some() => (
                    parsed.as_ref().and_then(StreamableParser::current_role),
                    true,
                ),
                Some(encoding) => harmony_header_ownership(encoding, source_header),
                // Without the encoding, an exact native role word is sufficient
                // to preserve assistant prose, but unknown tool authors stay hidden.
                None => {
                    let role = harmony_role_word(source_header)
                        .and_then(|word| Role::try_from(&source_header[word]).ok());
                    let known_role = role.is_some();
                    (role, known_role)
                }
            };
            let competing_channel = markers[index + 1..]
                .iter()
                .take_while(|marker| marker.start() < header_end)
                .filter(|marker| marker.as_str() == "<|channel|>")
                .count()
                > 1;
            let complete = native_complete
                && !competing_channel
                && end.is_some_and(|end| end.as_str() == "<|message|>");
            active = Some(HarmonyRoleSpan {
                start: marker.start(),
                end: text.len(),
                role,
                header_end: complete.then_some(header_end),
                parsed,
            });
        } else if marker.as_str() == "<|message|>" {
            in_body = true;
        } else if matches!(marker.as_str(), "<|end|>" | "<|call|>" | "<|return|>") {
            let body_ended = in_body;
            in_body = false;
            if body_ended && let Some(mut previous) = active.take() {
                previous.end = marker.end();
                spans.push(previous);
            }
        }
    }
    if let Some(previous) = active {
        spans.push(previous);
    }
    spans
}

fn strip_nonassistant_harmony_messages(text: &str, encoding: Option<&HarmonyEncoding>) -> String {
    let mut output = String::new();
    let mut cursor = 0;
    for span in harmony_framed_roles(text, encoding) {
        if span.role != Some(Role::Assistant) {
            output.push_str(&text[cursor..span.start]);
            cursor = span.end;
        }
    }
    output.push_str(&text[cursor..]);
    output
}

// Explicit routing and native function recipients disambiguate header prefixes
// from the prose retained by legacy recovery before a later framed message.
fn harmony_unframed_prefix_is_header(prefix: &str, header: &StreamableParser) -> bool {
    header.current_role() == Some(Role::Assistant)
        && (prefix
            .split_ascii_whitespace()
            .next()
            .is_some_and(|word| word == "assistant" || word.starts_with("to="))
            || harmony_current_recipient(header)
                .is_some_and(|recipient| recipient.starts_with("functions.")))
}

fn harmony_bare_header_spans(
    text: &str,
    encoding: &HarmonyEncoding,
    roles: &[HarmonyRoleSpan],
) -> Vec<std::ops::Range<usize>> {
    let markers: Vec<_> = harmony_markers(text)
        .filter_map(|(marker, quoted)| (!quoted).then_some(marker))
        .collect();
    let start = encoding.tokenizer().encode_with_special_tokens("<|start|>")[0];
    let mut spans = Vec::new();
    let mut boundary = 0;
    let mut channel_less_start = Some(0);
    for (index, marker) in markers.iter().enumerate() {
        match marker.as_str() {
            "<|start|>" => channel_less_start = None,
            "<|call|>" | "<|return|>" | "<|end|>" => channel_less_start = Some(marker.end()),
            "<|message|>" => {
                if let Some(begin) = channel_less_start.take() {
                    let tokens = encoding
                        .tokenizer()
                        .encode_with_special_tokens(&text[begin..marker.end()]);
                    if parse_harmony_header(encoding, &tokens, start).is_some_and(|header| {
                        header.current_role() == Some(Role::Assistant)
                            && header.current_channel().is_none()
                            && harmony_is_function_call(
                                None,
                                harmony_current_recipient(&header).as_deref(),
                            )
                    }) {
                        spans.push(begin..marker.end());
                    }
                }
            }
            _ => {}
        }
        if marker.as_str() != "<|channel|>" {
            boundary = marker.end();
            continue;
        }
        let owner_index = roles.partition_point(|role| role.start <= marker.start());
        if owner_index > 0
            && roles[owner_index - 1]
                .header_end
                .is_some_and(|end| marker.start() < end)
        {
            continue;
        }
        let Some(next) = markers[index + 1..]
            .iter()
            .find(|next| matches!(next.as_str(), "<|start|>" | "<|channel|>" | "<|message|>"))
        else {
            continue;
        };
        if next.as_str() != "<|message|>" {
            boundary = next.start();
            continue;
        }
        let tokens = encoding
            .tokenizer()
            .encode_with_special_tokens(&text[boundary..next.end()]);
        let prefix_owned = parse_harmony_header(encoding, &tokens, start).is_some_and(|header| {
            harmony_unframed_prefix_is_header(&text[boundary..marker.start()], &header)
        });
        if prefix_owned {
            spans.push(boundary..next.end());
        } else {
            let tokens = encoding
                .tokenizer()
                .encode_with_special_tokens(&text[marker.start()..next.end()]);
            if parse_harmony_header(encoding, &tokens, start).is_some() {
                spans.push(marker.start()..next.end());
            }
        }
    }
    spans
}

pub(crate) fn harmony_tool_headers(
    text: &str,
    encoding: &HarmonyEncoding,
) -> impl Iterator<Item = HarmonyToolHeader> {
    let roles = harmony_framed_roles(text, Some(encoding));
    harmony_tool_headers_from_roles(text, encoding, &roles).into_iter()
}

fn harmony_tool_headers_from_roles(
    text: &str,
    encoding: &HarmonyEncoding,
    roles: &[HarmonyRoleSpan],
) -> Vec<HarmonyToolHeader> {
    let mut markers = harmony_markers(text).peekable();
    let mut spans = harmony_bare_header_spans(text, encoding, roles);
    spans.extend(
        roles
            .iter()
            .filter_map(|role| role.header_end.map(|end| role.start..end)),
    );
    let mut search_start = 0;
    while let Some(span) = bounded_harmony_header(text, search_start) {
        search_start = span.end;
        spans.push(span);
    }
    spans.sort_unstable_by_key(|span| (span.start, span.end));
    // Lexical recovery and native prefix discovery can name the same header;
    // the widest source span owns it once.
    let mut previous_end = None;
    spans.retain(|span| {
        if previous_end == Some(span.end) {
            false
        } else {
            previous_end = Some(span.end);
            true
        }
    });
    let mut spans = spans.into_iter();
    std::iter::from_fn(move || {
        loop {
            let span = spans.next()?;
            while markers
                .peek()
                .is_some_and(|(marker, _)| marker.start() < span.start)
            {
                markers.next();
            }
            if markers.peek().is_some_and(|(_, quoted)| *quoted) {
                continue;
            }
            let role_index = roles.partition_point(|role| role.start <= span.start);
            if role_index > 0 {
                let owner = &roles[role_index - 1];
                if span.start < owner.end && owner.role != Some(Role::Assistant) {
                    continue;
                }
            }
            let raw = &text[span.clone()];
            let start = *encoding
                .tokenizer()
                .encode_with_special_tokens("<|start|>")
                .first()?;
            let parsed;
            let header = if let Some(header) = roles
                .get(role_index.saturating_sub(1))
                .filter(|role| role.start == span.start && role.header_end == Some(span.end))
                .and_then(|role| role.parsed.as_ref())
            {
                header
            } else {
                let tokens = encoding.tokenizer().encode_with_special_tokens(raw);
                parsed = parse_harmony_header(encoding, &tokens, start);
                let Some(header) = parsed.as_ref() else {
                    continue;
                };
                header
            };
            if header.current_role() != Some(Role::Assistant) {
                continue;
            }
            let channel = header.current_channel();
            if !harmony_is_function_call(
                channel.as_deref(),
                harmony_current_recipient(header).as_deref(),
            ) {
                continue;
            }
            let Some(name) = harmony_current_recipient(header).and_then(|recipient| {
                recipient
                    .strip_prefix("functions.")
                    .filter(|name| !name.is_empty())
                    .map(str::to_owned)
            }) else {
                continue;
            };
            let recipient_first = raw.find("functions.") < raw.find("<|channel|>");
            return Some(HarmonyToolHeader {
                start: span.start,
                end: span.end,
                name,
                channel,
                recipient_first,
            });
        }
    })
    .collect()
}

pub(crate) fn harmony_initial_role(token_ids: &[u32], start_token_id: u32) -> Option<Role> {
    if token_ids.first() == Some(&start_token_id) {
        None
    } else {
        Some(Role::Assistant)
    }
}

#[derive(Debug, Default)]
pub(crate) struct HarmonyContentCursor {
    body_start: Option<usize>,
    candidate_start: Option<usize>,
}

impl HarmonyContentCursor {
    pub(crate) fn is_pending(&self) -> bool {
        self.candidate_start.is_some()
    }

    pub(crate) fn in_body(&self) -> bool {
        self.body_start.is_some()
    }

    pub(crate) fn body_start(&self) -> Option<usize> {
        self.body_start
    }
}

pub(crate) fn harmony_content_header_can_start(
    parser: &StreamableParser,
    framed: bool,
) -> anyhow::Result<bool> {
    // Admission waits for Message, when the prospective channel and recipient
    // can distinguish a final transition from a narrated tool-header example.
    Ok(framed || parser.current_role() == Some(Role::Assistant))
}

pub(crate) fn process_harmony_token(
    parser: &mut StreamableParser,
    encoding: &HarmonyEncoding,
    resynchronizing: &mut bool,
    cursor: &mut HarmonyContentCursor,
    token: u32,
    start_token: u32,
    before_reset: impl FnOnce(&StreamableParser, bool) -> anyhow::Result<()>,
) -> anyhow::Result<bool> {
    if *resynchronizing && token != start_token {
        return Ok(false);
    }
    #[derive(serde::Deserialize)]
    #[serde(tag = "state")]
    enum NativeState {
        ExpectStart,
        Header {
            header_tokens: Vec<u32>,
        },
        #[serde(other)]
        Content,
    }
    let special = encoding.tokenizer().is_special_token(token);
    let marker = special
        .then(|| encoding.tokenizer().decode_utf8([token]))
        .transpose()?;
    let channel_marker = marker.as_deref() == Some("<|channel|>");
    if let Some(body_start) = cursor.body_start
        && (token == start_token || channel_marker)
        && harmony_content_header_can_start(parser, token == start_token)?
    {
        if token == start_token {
            cursor.candidate_start = Some(parser.tokens().len());
        } else if cursor.candidate_start.is_none() {
            cursor.candidate_start = Some(body_start);
        }
    }
    if matches!(
        marker.as_deref(),
        Some("<|message|>" | "<|call|>" | "<|return|>" | "<|end|>")
    ) {
        let tokens = parser.tokens();
        if let (Some(body_start), Some(candidate_start)) =
            (cursor.body_start, cursor.candidate_start)
        {
            let framed = tokens.get(candidate_start) == Some(&start_token);
            if framed && marker.as_deref() != Some("<|message|>") {
                for ordinary in encoding
                    .tokenizer()
                    .encode_ordinary(marker.as_deref().expect("matched a control marker"))
                {
                    parser.process(ordinary)?;
                }
                return Ok(true);
            }
            let mut raw = encoding
                .tokenizer()
                .decode_utf8(&tokens[candidate_start..])?;
            raw.push_str("<|message|>");
            let tools: Vec<_> = harmony_tool_headers(&raw, encoding).collect();
            let previous_accepts = harmony_current_recipient(parser).is_some()
                || (parser.current_channel().as_deref() == Some("commentary")
                    && harmony_commentary_is_payload(
                        harmony_current_content_type(parser).as_deref(),
                        &parser.current_content()?,
                    ));
            let replacement = harmony_header_spans(&raw, encoding, &tools)
                .into_iter()
                .rev()
                .find(|span| span.end == raw.len() && (!framed || span.start == 0))
                .and_then(|span| {
                    parse_harmony_header(
                        encoding,
                        &encoding
                            .tokenizer()
                            .encode_with_special_tokens(&raw[span.clone()]),
                        start_token,
                    )
                    .map(|header| (span.start, header))
                })
                .filter(|(_, header)| {
                    framed
                        || previous_accepts
                        || (header.current_channel().as_deref() == Some("final")
                            && harmony_current_recipient(header).is_none())
                });
            if replacement.is_some() || (framed && marker.as_deref() == Some("<|message|>")) {
                if marker.as_deref() == Some("<|message|>") {
                    let mut previous = StreamableParser::new(
                        encoding.clone(),
                        harmony_initial_role(tokens, start_token),
                    )?;
                    for id in &tokens[..body_start] {
                        previous.process(*id)?;
                    }
                    let mut prefix = encoding
                        .tokenizer()
                        .decode_utf8(&tokens[body_start..candidate_start])?;
                    prefix.push_str(&raw[..replacement.as_ref().map_or(0, |(start, _)| *start)]);
                    for id in encoding.tokenizer().encode_ordinary(&prefix) {
                        previous.process(id)?;
                    }
                    let end = encoding.tokenizer().encode_with_special_tokens("<|end|>")[0];
                    previous.process(end)?;
                    before_reset(&previous, true)?;
                    if let Some((_, header)) = replacement {
                        *parser = header;
                        cursor.body_start = Some(parser.tokens().len());
                        cursor.candidate_start = None;
                        tracing::warn!(
                            why = "harmony_completed_replacement_header",
                            recovered_tokens = 1,
                            "Replacing unfinished content at its next complete native header"
                        );
                    } else {
                        *parser = StreamableParser::new(encoding.clone(), None)?;
                        *cursor = HarmonyContentCursor::default();
                        *resynchronizing = true;
                        tracing::warn!(
                            why = "harmony_invalid_replacement_header",
                            skipped_tokens = 1,
                            "Suppressing an invalid framed header until the next message start"
                        );
                    }
                } else {
                    // The native body still owns the bytes until the next
                    // message delimiter confirms the replacement header.
                    for ordinary in encoding
                        .tokenizer()
                        .encode_ordinary(marker.as_deref().expect("matched a control marker"))
                    {
                        parser.process(ordinary)?;
                    }
                }
                return Ok(true);
            }
            if marker.as_deref() == Some("<|message|>") {
                cursor.candidate_start = None;
            }
        }
    }
    let boundary = token == start_token || channel_marker;
    let after_stop = parser.tokens().last().is_some_and(|previous| {
        encoding.tokenizer().is_special_token(*previous)
            && encoding
                .tokenizer()
                .decode_bytes([*previous])
                .is_ok_and(|bytes| {
                    matches!(bytes.as_slice(), b"<|return|>" | b"<|call|>" | b"<|end|>")
                })
    });
    let replacement = if !*resynchronizing && (boundary || after_stop) {
        match serde_json::from_str::<NativeState>(&parser.state_json()?)? {
            NativeState::ExpectStart
                if token != start_token && after_stop && (!special || channel_marker) =>
            {
                Some(Some(Role::Assistant))
            }
            NativeState::Header { header_tokens } if boundary => {
                let has_channel = header_tokens.iter().any(|id| {
                    encoding.tokenizer().is_special_token(*id)
                        && encoding
                            .tokenizer()
                            .decode_bytes([*id])
                            .is_ok_and(|bytes| bytes == b"<|channel|>")
                });
                let opened_by_start = parser
                    .tokens()
                    .len()
                    .checked_sub(header_tokens.len() + 1)
                    .and_then(|index| parser.tokens().get(index))
                    == Some(&start_token);
                if token == start_token {
                    Some(None)
                } else if !has_channel
                    && !opened_by_start
                    && harmony_json_prelude(&encoding.tokenizer().decode_utf8(&header_tokens)?)
                {
                    Some(Some(Role::Assistant))
                } else if !has_channel {
                    None
                } else {
                    let inherited_role = parser.current_role();
                    let mut header =
                        StreamableParser::new(encoding.clone(), inherited_role.clone())?;
                    if inherited_role.is_none() {
                        header.process(start_token)?;
                    }
                    for id in header_tokens {
                        header.process(id)?;
                    }
                    let message = encoding
                        .tokenizer()
                        .encode_with_special_tokens("<|message|>");
                    let [message] = message.as_slice() else {
                        anyhow::bail!("Harmony message marker must encode as one control token");
                    };
                    header.process(*message)?;
                    Some(header.current_role())
                }
            }
            _ => None,
        }
    } else {
        None
    };
    if let Some(role) = replacement {
        before_reset(parser, true)?;
        *parser = StreamableParser::new(encoding.clone(), role)?;
        parser.process(token)?;
        *cursor = HarmonyContentCursor::default();
        tracing::warn!(
            why = "harmony_replacement_header",
            recovered_tokens = 1,
            "Recovering an assistant header at a message boundary"
        );
        return Ok(true);
    }
    if parser.process(token).is_err() {
        before_reset(parser, false)?;
        *parser = StreamableParser::new(encoding.clone(), None)?;
        *cursor = HarmonyContentCursor::default();
        *resynchronizing = true;
        tracing::warn!(
            why = "harmony_orphan_token_resync",
            skipped_tokens = 1,
            "Harmony parser is waiting for the next message start"
        );
        return Ok(false);
    }
    if marker.as_deref() == Some("<|message|>") && cursor.body_start.is_none() {
        cursor.body_start = Some(parser.tokens().len());
    } else if matches!(
        marker.as_deref(),
        Some("<|call|>" | "<|return|>" | "<|end|>")
    ) && parser.current_role().is_none()
    {
        *cursor = HarmonyContentCursor::default();
    }
    *resynchronizing = false;
    Ok(true)
}

fn push_unique(items: &mut Vec<String>, item: String) {
    if !items.iter().any(|existing| existing == &item) {
        items.push(item);
    }
}

fn record_special_tokens(text: &str, items: &mut Vec<String>) {
    for m in special_token_regex().find_iter(text) {
        push_unique(items, format!("special_token:{}", m.as_str()));
    }
}

struct HarmonyTextRegion {
    header: std::ops::Range<usize>,
    body: std::ops::Range<usize>,
    end: usize,
    visible: bool,
}

pub(crate) fn harmony_message_is_visible(
    channel: Option<&str>,
    recipient: Option<&str>,
    has_call_stop: bool,
    payload: bool,
) -> bool {
    channel == Some("final")
        || (channel.is_some_and(|channel| !matches!(channel, "analysis" | "commentary"))
            && recipient.is_none())
        || (channel == Some("commentary") && recipient.is_none() && !(has_call_stop && payload))
}

fn harmony_text_regions(text: &str, encoding: &HarmonyEncoding) -> Vec<HarmonyTextRegion> {
    let tools: Vec<_> = harmony_tool_headers(text, encoding).collect();
    let headers = harmony_header_spans(text, encoding, &tools);
    let ends: Vec<_> = harmony_body_markers(text, &headers)
        .into_iter()
        .filter(|marker| matches!(marker.as_str(), "<|call|>" | "<|return|>" | "<|end|>"))
        .collect();
    let start = encoding.tokenizer().encode_with_special_tokens("<|start|>")[0];
    let mut regions = Vec::new();
    for (index, header) in headers.iter().enumerate() {
        let tokens = encoding
            .tokenizer()
            .encode_with_special_tokens(&text[header.clone()]);
        let Some(parsed) = parse_harmony_header(encoding, &tokens, start) else {
            continue;
        };
        let limit = headers.get(index + 1).map_or(text.len(), |next| next.start);
        let stop = ends
            .get(ends.partition_point(|stop| stop.start() < header.end))
            .filter(|stop| stop.start() < limit);
        let body_end = stop.map_or(limit, |stop| stop.start());
        let end = stop.map_or(limit, |stop| stop.end());
        let recipient = harmony_current_recipient(&parsed);
        let content_type = harmony_current_content_type(&parsed);
        let visible = parsed.current_role() == Some(Role::Assistant)
            && harmony_message_is_visible(
                parsed.current_channel().as_deref(),
                recipient.as_deref(),
                stop.is_some_and(|stop| stop.as_str() == "<|call|>"),
                harmony_commentary_is_payload(content_type.as_deref(), &text[header.end..body_end]),
            );
        regions.push(HarmonyTextRegion {
            header: header.clone(),
            body: header.end..body_end,
            end,
            visible,
        });
    }
    regions
}

fn strip_harmony_protocol_from_normal_text(
    text: &str,
    reason: &'static str,
    encoding: Option<&HarmonyEncoding>,
) -> String {
    if encoding.is_some_and(|encoding| harmony_header_eof_is_metadata(encoding, text)) {
        tracing::warn!(
            family = "harmony",
            reason,
            why = "native_unfinished_header",
            original_len = text.len(),
            cleaned_len = 0,
            "stripped harmony protocol content from normal_text"
        );
        return String::new();
    }
    let owned = strip_nonassistant_harmony_messages(text, encoding);
    let text = owned.as_str();
    let mut stripped = Vec::new();
    let markers: Vec<_> = harmony_markers(text).collect();
    let action_stops = ["<|call|>", "<|return|>"];
    let analysis_stops = ["<|call|>", "<|return|>", "<|end|>"];
    let final_stops = ["<|return|>", "<|end|>"];
    let cleanups = [
        (
            commentary_block_cleanup_regex(),
            action_stops.as_slice(),
            "commentary_tool_call",
        ),
        (
            commentary_header_cleanup_regex(),
            &[],
            "commentary_tool_call_without_message",
        ),
        (
            analysis_block_cleanup_regex(),
            analysis_stops.as_slice(),
            "analysis_envelope",
        ),
        (
            final_block_cleanup_regex(),
            final_stops.as_slice(),
            "final_envelope",
        ),
    ];
    let native_regions =
        encoding.map_or_else(Vec::new, |encoding| harmony_text_regions(text, encoding));
    let native_owns = |position| {
        let index = native_regions.partition_point(|region| region.header.start <= position);
        index > 0 && position < native_regions[index - 1].end
    };
    let mut spans = Vec::new();
    if let Some(encoding) = encoding {
        for frame in harmony_framed_roles(text, Some(encoding)) {
            if frame.header_end.is_none()
                && harmony_header_eof_is_metadata(encoding, &text[frame.start..frame.end])
            {
                spans.push((
                    frame.start,
                    frame.end,
                    0,
                    "native_unfinished_header".to_string(),
                ));
            }
        }
    }
    for region in &native_regions {
        if region.visible {
            spans.push((
                region.header.start,
                region.header.end,
                0,
                "native_visible_header".to_string(),
            ));
            spans.push((
                region.body.end,
                region.end,
                0,
                "native_visible_end".to_string(),
            ));
        } else {
            spans.push((
                region.header.start,
                region.end,
                0,
                "native_nonvisible_message".to_string(),
            ));
        }
    }
    for prefix in commentary_header_prefix_regex().find_iter(text) {
        if native_owns(prefix.start()) {
            continue;
        }
        let first = markers.partition_point(|(marker, _)| marker.start() < prefix.start());
        if markers
            .get(first)
            .is_some_and(|(marker, in_json)| *in_json && marker.start() < prefix.end())
        {
            continue;
        }
        let after_prefix = markers.partition_point(|(marker, _)| marker.start() < prefix.end());
        let boundary = markers[after_prefix..].iter().find(|(marker, in_json)| {
            !in_json
                && marker.start() >= prefix.end()
                && matches!(marker.as_str(), "<|start|>" | "<|channel|>" | "<|message|>")
        });
        if boundary.is_none_or(|(marker, _)| marker.as_str() != "<|message|>") {
            let boundary_start = boundary.map_or(text.len(), |(marker, _)| marker.start());
            let end = commentary_header_prefix_regex()
                .find_at(text, prefix.end())
                .map_or(boundary_start, |next| next.start().min(boundary_start));
            spans.push((
                prefix.start(),
                end,
                0,
                "unfinished_directed_header".to_string(),
            ));
        }
    }

    // Keep JSON ownership anchored to the original message marker. Replacing a
    // final envelope first would expose its quoted markers to later cleanup passes.
    for (priority, (regex, endings, label)) in cleanups.into_iter().enumerate() {
        for caps in regex.captures_iter(text) {
            let matched = caps.get(0).expect("regex match has full span");
            if native_owns(matched.start()) {
                continue;
            }
            let start_index =
                markers.partition_point(|(marker, _)| marker.start() < matched.start());
            if markers
                .get(start_index)
                .is_some_and(|(marker, in_json)| marker.start() == matched.start() && *in_json)
            {
                continue;
            }
            let mut end = matched.end();
            let mut body_end = caps.name("body").map(|body| body.end());
            let end_index = markers.partition_point(|(marker, _)| marker.end() <= end);
            if end_index > 0
                && let (marker, true) = &markers[end_index - 1]
                && marker.end() == end
                && endings.contains(&marker.as_str())
            {
                let stop = markers[end_index..]
                    .iter()
                    .find(|(marker, in_json)| !in_json && endings.contains(&marker.as_str()));
                end = stop.map_or(text.len(), |(marker, _)| marker.end());
                body_end = Some(stop.map_or(text.len(), |(marker, _)| marker.start()));
            }
            let item = label.to_string();
            if label == "final_envelope" {
                let body = caps.name("body").expect("final envelope has body");
                spans.push((matched.start(), body.start(), priority, item.clone()));
                spans.push((
                    body_end.expect("final envelope has body end"),
                    end,
                    priority,
                    item,
                ));
            } else {
                spans.push((matched.start(), end, priority, item));
            }
        }
    }
    let mut stops = markers
        .iter()
        .filter(|(marker, in_json)| !in_json && action_stops.contains(&marker.as_str()))
        .peekable();
    for (marker, in_json) in &markers {
        if *in_json || native_owns(marker.start()) {
            continue;
        }
        if marker.as_str() == "<|message|>" {
            while stops
                .peek()
                .is_some_and(|(stop, _)| stop.start() < marker.end())
            {
                stops.next();
            }
            let end = stops.peek().map_or(text.len(), |(stop, _)| stop.end());
            spans.push((marker.start(), end, 4, "message_call_payload".to_string()));
        }
        spans.push((marker.start(), marker.end(), 5, "special_token".to_string()));
    }
    spans.sort_by_key(|(start, _, priority, _)| (*start, *priority));
    let mut cleaned = String::new();
    let mut cursor = 0;
    for (start, end, _, item) in spans {
        if start < cursor || start == end {
            continue;
        }
        cleaned.push_str(&text[cursor..start]);
        record_special_tokens(&text[start..end], &mut stripped);
        push_unique(&mut stripped, item);
        cursor = end;
    }
    cleaned.push_str(&text[cursor..]);

    if stripped.is_empty() {
        return text.to_string();
    }

    let cleaned = cleaned.trim().to_string();

    tracing::warn!(
        family = "harmony",
        reason,
        stripped = ?stripped,
        original_len = text.len(),
        cleaned_len = cleaned.len(),
        "stripped harmony protocol content from normal_text"
    );

    cleaned
}

fn serialize_harmony_arguments(raw_args: &str) -> String {
    let trimmed = raw_args.trim();
    if serde_json::from_str::<&serde_json::value::RawValue>(trimmed).is_ok() {
        crate::tool_calling::json::compact_json_lexically(trimmed)
    } else {
        trimmed.to_string()
    }
}

fn push_harmony_text(out: &mut String, content: &[Content]) {
    for item in content {
        if let Content::Text(text) = item {
            out.push_str(&text.text);
        }
    }
}

pub(crate) fn harmony_header_metadata<'a>(
    recipient: Option<&'a str>,
    content_type: Option<&'a str>,
) -> (Option<&'a str>, Option<&'a str>) {
    // Native Harmony treats a lone constraint as a recipient. An explicit `to=`
    // instead supplies a recipient and content type, and must remain directed.
    match (recipient, content_type) {
        (Some(recipient), None) if recipient.starts_with("<|constrain|>") => {
            (None, Some(recipient))
        }
        metadata => metadata,
    }
}

pub(crate) fn harmony_current_recipient(parser: &StreamableParser) -> Option<String> {
    let recipient = parser.current_recipient();
    if harmony_header_metadata(
        recipient.as_deref(),
        parser.current_content_type().as_deref(),
    )
    .0
    .is_none()
    {
        None
    } else {
        recipient
    }
}

pub(crate) fn harmony_is_function_call(channel: Option<&str>, recipient: Option<&str>) -> bool {
    matches!(channel, None | Some("analysis" | "commentary"))
        && recipient
            .and_then(|recipient| recipient.strip_prefix("functions."))
            .is_some_and(|name| !name.is_empty())
}

pub(crate) fn harmony_current_content_type(parser: &StreamableParser) -> Option<String> {
    let recipient = parser.current_recipient();
    let content_type = parser.current_content_type();
    harmony_header_metadata(recipient.as_deref(), content_type.as_deref())
        .1
        .map(str::to_owned)
}

pub(crate) fn harmony_commentary_is_payload(content_type: Option<&str>, text: &str) -> bool {
    content_type == Some("<|constrain|>json")
        || matches!(text.trim_start().as_bytes().first(), Some(b'{' | b'['))
}

#[cfg(test)]
thread_local! {
    static HARMONY_BOUNDARY_BYTES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static HARMONY_HEADER_PARSES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

pub(crate) fn harmony_header_spans(
    text: &str,
    encoding: &HarmonyEncoding,
    tools: &[HarmonyToolHeader],
) -> Vec<std::ops::Range<usize>> {
    let roles = harmony_framed_roles(text, Some(encoding));
    harmony_header_spans_from_roles(text, encoding, tools, &roles)
}

fn harmony_header_spans_from_roles(
    text: &str,
    encoding: &HarmonyEncoding,
    tools: &[HarmonyToolHeader],
    roles: &[HarmonyRoleSpan],
) -> Vec<std::ops::Range<usize>> {
    let mut spans: Vec<_> = roles
        .iter()
        .filter_map(|frame| frame.header_end.map(|end| frame.start..end))
        .collect();
    spans.extend(tools.iter().map(|header| header.start..header.end));
    spans.extend(harmony_bare_header_spans(text, encoding, roles));
    spans.sort_unstable_by_key(|span| (span.start, span.end));
    let mut owned: Vec<std::ops::Range<usize>> = Vec::new();
    for span in spans {
        if let Some(previous) = owned.last_mut()
            && span.start <= previous.end
        {
            previous.end = previous.end.max(span.end);
        } else {
            owned.push(span);
        }
    }
    owned
}

fn harmony_message_boundary(text: &str, encoding: &HarmonyEncoding) -> Option<usize> {
    #[cfg(test)]
    HARMONY_BOUNDARY_BYTES.with(|bytes| bytes.set(bytes.get() + text.len()));
    // A lone marker may be raw arguments; only native complete headers and
    // explicit message ends can interrupt the current call.
    let tools: Vec<_> = harmony_tool_headers(text, encoding).collect();
    let header = harmony_header_spans(text, encoding, &tools)
        .first()
        .map(|header| header.start);
    let end = harmony_markers(text).find_map(|(marker, quoted)| {
        (!quoted && marker.as_str() == "<|end|>").then_some(marker.start())
    });
    match (header, end) {
        (Some(header), Some(end)) => Some(header.min(end)),
        (Some(header), None) => Some(header),
        (None, end) => end,
    }
}

pub(crate) fn harmony_body_markers<'a>(
    text: &'a str,
    headers: &[std::ops::Range<usize>],
) -> Vec<regex::Match<'a>> {
    let mut header_index = 0;
    harmony_markers(text)
        .filter_map(|(marker, quoted)| {
            if quoted {
                return None;
            }
            while headers
                .get(header_index)
                .is_some_and(|header| header.end <= marker.start())
            {
                header_index += 1;
            }
            (!headers
                .get(header_index)
                .is_some_and(|header| header.start <= marker.start() && marker.end() <= header.end))
            .then_some(marker)
        })
        .collect()
}

pub(crate) fn harmony_body_action_markers<'a>(
    text: &'a str,
    headers: &[std::ops::Range<usize>],
) -> Vec<regex::Match<'a>> {
    harmony_body_markers(text, headers)
        .into_iter()
        .filter(|marker| matches!(marker.as_str(), "<|call|>" | "<|return|>"))
        .collect()
}

pub(crate) fn harmony_body_ending<'a>(
    text: &'a str,
    encoding: &HarmonyEncoding,
) -> Option<regex::Match<'a>> {
    let tools: Vec<_> = harmony_tool_headers(text, encoding).collect();
    let headers = harmony_header_spans(text, encoding, &tools);
    harmony_body_markers(text, &headers)
        .into_iter()
        .find(|marker| matches!(marker.as_str(), "<|call|>" | "<|return|>" | "<|end|>"))
}

/// Recover calls through bounded native headers when parsing the complete
/// completion fails (truncated JSON or competing messages, for example).
/// Returns (calls, residual_text) where residual_text is everything not
/// consumed by a matched commentary block — preserved so non-tool user-visible
/// spans aren't dropped.
fn extract_calls_from_headers(
    text: &str,
    encoding: &HarmonyEncoding,
) -> (Vec<ToolCallResponse>, String) {
    let mut out = Vec::new();
    let mut residual = String::new();
    let mut cursor = 0;
    let mut search_start = 0;
    let roles = harmony_framed_roles(text, Some(encoding));
    let headers = harmony_tool_headers_from_roles(text, encoding, &roles);
    let header_spans = harmony_header_spans_from_roles(text, encoding, &headers, &roles);
    let mut action_stops = harmony_body_action_markers(text, &header_spans)
        .into_iter()
        .peekable();
    for header in &headers {
        if header.start < search_start {
            continue;
        }
        let name = &header.name;
        while action_stops
            .peek()
            .is_some_and(|stop| stop.start() < header.end)
        {
            action_stops.next();
        }
        let Some(stop) = action_stops.peek().copied() else {
            break;
        };
        let raw_args = &text[header.end..stop.start()];
        // A later message's stop must not complete an earlier unfinished call.
        if let Some(boundary) = harmony_message_boundary(raw_args, encoding)
            && boundary < raw_args.len()
        {
            residual.push_str(&text[cursor..header.start]);
            // Discard this unfinished span before joining residual text, so its
            // header cannot swallow text after a subsequently recovered call.
            cursor = header.end + boundary;
            tracing::warn!(
                why = "harmony_unfinished_call_before_message",
                skipped_bytes = cursor - header.start,
                "Discarding an unfinished tool envelope before the next message"
            );
            search_start = cursor;
            continue;
        }
        residual.push_str(&text[cursor..header.start]);
        cursor = stop.end();
        search_start = cursor;
        action_stops.next();
        let args_json = serialize_harmony_arguments(raw_args);
        out.push(ToolCallResponse {
            id: format!("call-{}", Uuid::new_v4()),
            tp: ToolCallType::Function,
            function: CalledFunction {
                name: name.to_string(),
                arguments: args_json,
            },
        });
    }
    residual.push_str(&text[cursor..]);
    (out, residual.trim().to_string())
}

fn recover_harmony_calls(
    text: &str,
    encoding: &HarmonyEncoding,
) -> (Vec<ToolCallResponse>, Option<String>) {
    let (calls, residual) = extract_calls_from_headers(text, encoding);
    let normal_text = if calls.is_empty() {
        strip_harmony_protocol_from_normal_text(
            if residual == text.trim() {
                text
            } else {
                &residual
            },
            "parse_failed_no_recovered_calls",
            Some(encoding),
        )
    } else {
        strip_harmony_protocol_from_normal_text(
            &residual,
            "regex_recovery_residual",
            Some(encoding),
        )
    };
    (calls, Some(normal_text))
}

static GLOBAL_HARMONY_GPTOSS_ENCODING: tokio::sync::OnceCell<
    Result<HarmonyEncoding, anyhow::Error>,
> = tokio::sync::OnceCell::const_new();

pub async fn get_harmony_encoding() -> &'static Result<HarmonyEncoding, anyhow::Error> {
    GLOBAL_HARMONY_GPTOSS_ENCODING
        .get_or_init(|| async {
            tokio::task::spawn_blocking(|| {
                load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss)
            })
            .await
            .map_err(anyhow::Error::msg)
            .flatten()
        })
        .await
}

/// Extract complete Harmony tool calls and visible assistant text.
///
/// Native token parsing and bounded recovery retain directed commentary and
/// analysis calls ending in `<|call|>` or `<|return|>`. A message-only `<|end|>`
/// does not complete a call. Bare visible text survives reasoning handoff.
///
/// # Arguments
/// * `text` - The full Harmony-format string to be parsed, including required Harmony stop tokens.
///   Example:
///   `<|channel|>commentary to=functions.get_current_weather <|constrain|>json<|message|>{"location":"San Francisco"}<|call|>`
/// * `_config` - Parser configuration (currently unused but kept for API consistency)
///
/// # Returns
/// * `Ok((tool_calls, normal_text))` - Tuple containing extracted tool calls and any normal text
pub async fn parse_tool_calls_harmony_complete(
    text: &str,
    _config: &JsonParserConfig,
    _tools: Option<&[ToolDefinition]>,
) -> anyhow::Result<(Vec<ToolCallResponse>, Option<String>)> {
    // Reasoning extraction already removes visible message envelopes before
    // the batch tool parser runs; that handoff must remain visible content.
    let Some((first_marker, _)) = harmony_markers(text).find(|(_, in_json)| !in_json) else {
        return Ok((Vec::new(), Some(text.to_string())));
    };
    let prefix = &text[..first_marker.start()];
    // Native Header recovery may discard rejected header tokens; keep visible
    // prose outside that state, using native metadata for callable prefixes.
    let native_prefix = first_marker.as_str() != "<|start|>"
        && !harmony_json_prelude(prefix)
        && get_harmony_encoding().await.as_ref().is_ok_and(|encoding| {
            let end = harmony_markers(text)
                .find_map(|(marker, quoted)| {
                    (!quoted && marker.as_str() == "<|message|>").then_some(marker.end())
                })
                .unwrap_or(text.len());
            let mut raw = text[..end].to_string();
            if !raw.ends_with("<|message|>") {
                raw.push_str("<|message|>");
            }
            let tokens = encoding.tokenizer().encode_with_special_tokens(&raw);
            let start = encoding.tokenizer().encode_with_special_tokens("<|start|>")[0];
            parse_harmony_header(encoding, &tokens, start).is_some_and(|header| {
                // Legacy recovery retains prose before an unframed envelope
                // followed by a separate framed message. A standalone native
                // implicit recipient keeps its established suppression.
                harmony_unframed_prefix_is_header(prefix, &header)
                    || !harmony_markers(text)
                        .any(|(marker, quoted)| !quoted && marker.as_str() == "<|start|>")
            })
        });
    if !prefix.trim().is_empty() && !native_prefix {
        if first_marker.as_str() == "<|start|>"
            && !harmony_markers(text)
                .any(|(marker, quoted)| !quoted && marker.as_str() == "<|message|>")
        {
            // An established visible body owns a new Start until Message
            // confirms a header; reasoning handoff must not erase those bytes.
            let normal = get_harmony_encoding()
                .await
                .as_ref()
                .map_or(text, |encoding| {
                    &text[..harmony_body_ending(text, encoding)
                        .map_or(text.len(), |ending| ending.start())]
                });
            return Ok((Vec::new(), Some(normal.to_string())));
        }
        let (calls, normal) = parse_harmony_body(&text[first_marker.start()..]).await?;
        return Ok((
            calls,
            Some(format!("{prefix}{}", normal.unwrap_or_default())),
        ));
    }
    parse_harmony_body(text).await
}

async fn parse_harmony_body(text: &str) -> anyhow::Result<(Vec<ToolCallResponse>, Option<String>)> {
    let enc = match get_harmony_encoding().await.as_ref() {
        Ok(e) => e,
        Err(e) => {
            tracing::debug!("Failed to load harmony encoding: {e}. Tool calls will not be parsed.");
            let normal_text =
                strip_harmony_protocol_from_normal_text(text, "harmony_encoding_unavailable", None);
            return Ok((vec![], Some(normal_text)));
        }
    };

    if json_value_prefix_end(text).is_some()
        || harmony_markers(text)
            .next()
            .is_some_and(|(_, in_json)| in_json)
    {
        return Ok(recover_harmony_calls(text, enc));
    }
    // JSON-owned markers are payload bytes, not native Harmony control IDs.
    let mut tokens = Vec::new();
    let mut cursor = 0;
    for (marker, in_json) in harmony_markers(text) {
        if in_json {
            tokens.extend(
                enc.tokenizer()
                    .encode_with_special_tokens(&text[cursor..marker.start()]),
            );
            tokens.extend(enc.tokenizer().encode_ordinary(marker.as_str()));
            cursor = marker.end();
        }
    }
    tokens.extend(enc.tokenizer().encode_with_special_tokens(&text[cursor..]));
    let mut message_stops = Vec::new();
    let mut trailing_text = String::new();
    let parsed: anyhow::Result<_> = (|| {
        let start_tokens = enc.tokenizer().encode_with_special_tokens("<|start|>");
        let [start_token_id] = start_tokens.as_slice() else {
            anyhow::bail!("Harmony start marker must encode as one control token");
        };
        let mut parser =
            StreamableParser::new(enc.clone(), harmony_initial_role(&tokens, *start_token_id))?;
        let action_stops = enc.stop_tokens_for_assistant_actions()?;
        let call_stop = enc.tokenizer().encode_with_special_tokens("<|call|>");
        let mut resynchronizing = false;
        let mut content_cursor = HarmonyContentCursor::default();
        let mut retained = Vec::new();
        for token in tokens {
            if !process_harmony_token(
                &mut parser,
                enc,
                &mut resynchronizing,
                &mut content_cursor,
                token,
                *start_token_id,
                |parser, boundary_recovery| {
                    anyhow::ensure!(
                        boundary_recovery
                            || action_stops.contains(&token)
                            || enc.tokenizer().decode_utf8([token])? == "<|end|>",
                        "Harmony framing requires regex recovery"
                    );
                    retained.extend_from_slice(parser.messages());
                    Ok(())
                },
            )? {
                continue;
            }
            if retained.len() + parser.messages().len() > message_stops.len() {
                message_stops.push((
                    action_stops.contains(&token),
                    call_stop.as_slice() == [token],
                ));
            }
        }
        if let Err(error) = parser.process_eos() {
            if retained.is_empty() && parser.messages().is_empty() {
                return Err(error);
            }
            if let Some(recovered) = crate::reasoning::recover_stranded_header_text(&parser) {
                trailing_text = recovered.normal_text;
            }
            tracing::warn!(
                why = "harmony_unfinished_header_after_messages",
                completed_messages = retained.len() + parser.messages().len(),
                "Preserving completed Harmony messages at an unfinished header"
            );
        }
        retained.extend(parser.into_messages());
        message_stops.resize(retained.len(), (false, false));
        Ok((retained, content_cursor.is_pending()))
    })();
    let (messages, pending_content) = match parsed {
        Ok(messages) => messages,
        Err(e) => {
            tracing::debug!(
                "Failed to parse messages from completion tokens: {e}. Falling back to regex extraction."
            );
            // Recovery: harmony rejects parallel commentary blocks even when
            // every call is explicitly closed. Only EOF/truncated recovery is
            // gated, so streaming jails do not synthesize incomplete calls.
            // Harmony differs from generic JSON/XML recovery: a tool call is
            // complete only once `<|call|>` or `<|return|>` is present. Do not synthesize a
            // call from EOF; that would diverge from vLLM/openai_harmony.
            return Ok(recover_harmony_calls(text, enc));
        }
    };

    let mut res = Vec::with_capacity(messages.len());
    let mut normal_text = String::new();
    let source_has_action = harmony_markers(text)
        .any(|(marker, quoted)| !quoted && matches!(marker.as_str(), "<|call|>" | "<|return|>"));
    for (index, (message, (has_action_stop, has_call_stop))) in
        messages.iter().zip(message_stops).enumerate()
    {
        if message.author.role != Role::Assistant {
            continue;
        }

        let channel = message.channel.as_deref();
        let (recipient, content_type) = harmony_header_metadata(
            message.recipient.as_deref(),
            message.content_type.as_deref(),
        );
        let pending_body = if pending_content && index + 1 == messages.len() {
            let mut body = String::new();
            push_harmony_text(&mut body, &message.content);
            let ending = harmony_body_ending(&body, enc);
            Some((
                body[..ending.as_ref().map_or(body.len(), |ending| ending.start())].to_string(),
                ending.is_some_and(|ending| ending.as_str() == "<|call|>"),
            ))
        } else {
            None
        };

        // A native function recipient can omit its channel or use commentary
        // or analysis. Other channels keep their existing visibility policy.
        if harmony_is_function_call(channel, recipient) {
            if !has_action_stop {
                // A stop deferred while a new header was pending is still the
                // old call's ending if EOF never confirmed that new header.
                if pending_body.is_some() && source_has_action {
                    return Ok(recover_harmony_calls(text, enc));
                }
                continue;
            }

            let Some(fname) = message
                .recipient
                .as_ref()
                .and_then(|r| r.strip_prefix("functions."))
                .filter(|s| !s.is_empty())
                .map(|s| s.to_string())
            else {
                continue;
            };

            let Some(args_json) = message.content.first().and_then(|content| match content {
                Content::Text(text) => Some(serialize_harmony_arguments(&text.text)),
                _ => None,
            }) else {
                continue;
            };
            if harmony_message_boundary(&args_json, enc).is_some() {
                return Ok(recover_harmony_calls(text, enc));
            }

            res.push(ToolCallResponse {
                id: format!("call-{}", Uuid::new_v4()),
                tp: ToolCallType::Function,
                function: CalledFunction {
                    name: fname.to_string(),
                    arguments: args_json,
                },
            });
        } else if harmony_message_is_visible(
            channel,
            recipient,
            has_call_stop || pending_body.as_ref().is_some_and(|(_, call)| *call),
            message.content.iter().any(|content| match content {
                Content::Text(text) => harmony_commentary_is_payload(content_type, &text.text),
                _ => false,
            }),
        ) {
            // Final-channel answer text, or recipientless commentary that isn't a
            // tool-call payload, both surface as assistant content.
            if let Some((body, _)) = pending_body {
                normal_text.push_str(&body);
            } else {
                push_harmony_text(&mut normal_text, &message.content);
            }
        }
    }
    normal_text.push_str(&trailing_text);
    Ok((res, Some(normal_text)))
}

/// Short `<` / `<|` chunk tails are ambiguous between prose and the head of
/// a text-split marker; the jail holds them for the next chunk instead of
/// jailing immediately.
pub fn detect_tool_call_start_harmony(
    chunk: &str,
    config: &JsonParserConfig,
    strict: bool,
) -> bool {
    let trimmed = chunk.trim();
    if trimmed.is_empty() {
        return false;
    }

    let has_complete_end_token = config
        .tool_call_end_tokens
        .iter()
        .any(|token| !token.is_empty() && trimmed.contains(token));
    if has_complete_end_token {
        return true;
    }

    if strict {
        // Check for complete start tokens first
        let has_complete_token = config
            .tool_call_start_tokens
            .iter()
            .any(|token| !token.is_empty() && trimmed.contains(token));

        if has_complete_token {
            return true;
        }

        // Check for partial start tokens (streaming scenario)
        // This handles cases where start tokens are split across multiple chunks
        config.tool_call_start_tokens.iter().any(|token| {
            if token.is_empty() {
                return false;
            }
            // Check if the chunk could be a prefix of this start token
            // Handle Unicode character boundaries properly
            for i in 1..=token.chars().count() {
                if let Some(prefix) = token.chars().take(i).collect::<String>().get(..) {
                    let prefix_str = &prefix[..prefix.len()];
                    if trimmed == prefix_str || trimmed.ends_with(prefix_str) {
                        return true;
                    }
                }
            }
            false
        })
    } else {
        // Non-strict mode: check complete tokens and some heuristics
        let has_complete_token = config
            .tool_call_start_tokens
            .iter()
            .any(|token| !token.is_empty() && trimmed.contains(token));

        if has_complete_token {
            return true;
        }

        // Check for partial start tokens or known patterns
        let has_partial_token = config.tool_call_start_tokens.iter().any(|token| {
            if token.is_empty() {
                return false;
            }
            // Check if the chunk could be a prefix of this start token
            // Handle Unicode character boundaries properly
            for i in 1..=token.chars().count() {
                if let Some(prefix) = token.chars().take(i).collect::<String>().get(..) {
                    let prefix_str = &prefix[..prefix.len()];
                    if trimmed == prefix_str || trimmed.ends_with(prefix_str) {
                        return true;
                    }
                }
            }
            false
        });

        has_partial_token || trimmed.contains("<|channel|>")
    }
}

#[cfg(test)]
mod tests {
    #[tokio::test]
    async fn visible_json_prelude_preserves_exact_whitespace() {
        for padding in ["    ", "\t", "\n", " \n\t"] {
            for json in [r#"{"x":1}"#, "[1,2]", r#""雪""#] {
                for frame in ["", "<|start|>assistant"] {
                    let prefix = format!("{padding}{json}\n");
                    let expected = format!("{prefix}Done  \t\n");
                    let input =
                        format!("{prefix}{frame}<|channel|>final<|message|>Done  \t\n<|return|>");
                    let (calls, normal) =
                        parse_tool_calls_harmony_complete(&input, &Default::default(), None)
                            .await
                            .unwrap();
                    assert!(calls.is_empty(), "{input:?}");
                    assert_eq!(normal.as_deref(), Some(expected.as_str()), "{input:?}");
                    let (calls, raw) =
                        parse_tool_calls_harmony_complete(&expected, &Default::default(), None)
                            .await
                            .unwrap();
                    assert!(calls.is_empty());
                    assert_eq!(raw, normal, "raw control {input:?}");
                    let unfinished = format!("{prefix}<|start|>assistant<|channel|>final");
                    let (calls, raw) =
                        parse_tool_calls_harmony_complete(&unfinished, &Default::default(), None)
                            .await
                            .unwrap();
                    assert!(calls.is_empty());
                    assert_eq!(raw.as_deref(), Some(unfinished.as_str()));
                }
            }
        }
        for header in [
            "<|channel|>final",
            "<|start|>assistant<|channel|>final",
            "<|start|>assistant to=functions.lookup<|channel|>commentary",
        ] {
            let (calls, normal) =
                parse_tool_calls_harmony_complete(header, &Default::default(), None)
                    .await
                    .unwrap();
            assert!(calls.is_empty(), "{header}");
            assert_eq!(normal.as_deref(), Some(""), "{header}");
        }
    }

    #[tokio::test]
    async fn native_inventory_does_not_borrow_a_replacement_message_delimiter() {
        let encoding = get_harmony_encoding().await.as_ref().unwrap();
        for role in ["", "<|start|>assistant", "<|start|>user"] {
            for channel in ["analysis", "commentary"] {
                let input = format!(
                    "{role}<|channel|>{channel} to=functions.bad<|channel|>final<|message|>ANSWER<|return|>"
                );
                let calls: Vec<_> = harmony_tool_headers(&input, encoding).collect();
                assert!(calls.is_empty(), "{input}: {calls:?}");
                let forced = format!("{{\"literal\":\"<|start|>\"}}{input}");
                let (calls, normal) =
                    parse_tool_calls_harmony_complete(&forced, &Default::default(), None)
                        .await
                        .unwrap();
                assert!(calls.is_empty(), "{forced}: {calls:?}");
                assert_eq!(
                    normal.as_deref(),
                    Some(if role == "<|start|>user" {
                        r#"{"literal":"<|start|>"}"#
                    } else {
                        r#"{"literal":"<|start|>"}ANSWER"#
                    }),
                    "{forced}"
                );
            }
        }
    }

    #[tokio::test]
    async fn replacement_message_ending_releases_the_previous_role_owner() {
        let input = r#"{"literal":"<|start|>"}<|start|>user<|channel|>commentary to=functions.bad<|channel|>final<|message|>Private<|return|><|channel|>commentary to=functions.lookup<|message|>{}<|return|>"#;
        let (calls, normal) = parse_tool_calls_harmony_complete(input, &Default::default(), None)
            .await
            .unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "lookup");
        assert_eq!(calls[0].function.arguments, "{}");
        assert_eq!(normal.as_deref(), Some(r#"{"literal":"<|start|>"}"#));
    }

    #[tokio::test]
    async fn header_only_eof_does_not_surface_native_metadata() {
        for role in ["assistant", "user", "python to=browser"] {
            for channel in ["", "analysis", "commentary", "final", "finalx", "fi"] {
                for recipient in ["", " to=functions.lookup", " to=python"] {
                    if role != "assistant" && !recipient.is_empty() {
                        continue;
                    }
                    for frame in ["", "<|start|>"] {
                        if frame.is_empty() && role != "assistant" {
                            continue;
                        }
                        let prefix = if frame.is_empty() {
                            "".to_string()
                        } else {
                            format!("{frame}{role}")
                        };
                        let input = format!("{prefix}<|channel|>{channel}{recipient}");
                        let (calls, normal) =
                            parse_tool_calls_harmony_complete(&input, &Default::default(), None)
                                .await
                                .unwrap();
                        assert!(calls.is_empty(), "{input}: {calls:?}");
                        assert_eq!(normal.as_deref(), Some(""), "{input}");
                    }
                }
            }
        }
        for input in [
            "<|start|>assistant",
            "<|start|>assistant to=functions.lookup",
            "<|start|>assistant to=",
        ] {
            let (calls, normal) =
                parse_tool_calls_harmony_complete(input, &Default::default(), None)
                    .await
                    .unwrap();
            assert!(calls.is_empty(), "{input}");
            assert_eq!(normal.as_deref(), Some(""), "{input}");
        }
        for input in [
            "<|start|>assistant*)__Hello!",
            "<|start|>assistant<|channel|>final*)__Hello!",
            "<|channel|>analysis*)__Thinking it over",
            "ordinary prose",
        ] {
            let (calls, normal) =
                parse_tool_calls_harmony_complete(input, &Default::default(), None)
                    .await
                    .unwrap();
            assert!(calls.is_empty(), "{input}");
            assert!(
                normal.as_deref().is_some_and(|text| !text.is_empty()),
                "{input}"
            );
        }
    }

    #[tokio::test]
    async fn visible_native_eof_keeps_unconfirmed_headers_but_not_body_endings() {
        for channel in ["analysis", "commentary", "final", "finalx"] {
            for role in ["assistant", "user", "python to=browser"] {
                for ending in ["", "<|call|>", "<|return|>", "<|end|>"] {
                    let body = format!("Before<|start|>{role}<|channel|>final");
                    let input = format!("<|channel|>{channel}<|message|>{body}{ending}");
                    let (calls, normal) =
                        parse_tool_calls_harmony_complete(&input, &Default::default(), None)
                            .await
                            .unwrap();
                    assert!(calls.is_empty(), "{input}: {calls:?}");
                    assert_eq!(
                        normal.as_deref(),
                        Some(if channel == "analysis" {
                            ""
                        } else {
                            body.as_str()
                        }),
                        "{input}"
                    );
                    let (handoff_calls, handoff_normal) = parse_tool_calls_harmony_complete(
                        normal.as_deref().unwrap(),
                        &Default::default(),
                        None,
                    )
                    .await
                    .unwrap();
                    assert!(handoff_calls.is_empty(), "handoff {input}");
                    assert_eq!(handoff_normal, normal, "handoff {input}");
                }
            }
        }
        for ending in ["<|call|>", "<|return|>", "<|end|>"] {
            let body = r#"{"x":1}<|start|>assistant<|channel|>final"#;
            let input = format!("<|channel|>commentary<|message|>{body}{ending}");
            let (calls, normal) =
                parse_tool_calls_harmony_complete(&input, &Default::default(), None)
                    .await
                    .unwrap();
            assert!(calls.is_empty(), "{input}");
            assert_eq!(
                normal.as_deref(),
                Some(if ending == "<|call|>" { "" } else { body }),
                "{input}"
            );
        }
        let body = r#"{"literal":"<|start|>assistant<|channel|>final<|return|>"}"#;
        let input = format!("<|channel|>final<|message|>{body}<|return|>");
        let (calls, normal) = parse_tool_calls_harmony_complete(&input, &Default::default(), None)
            .await
            .unwrap();
        assert!(calls.is_empty());
        assert_eq!(normal.as_deref(), Some(body));
    }

    #[tokio::test]
    async fn visible_prelude_does_not_confirm_an_unfinished_header() {
        for header in [
            "<|start|>assistant to=functions.lookup<|channel|>commentary",
            "<|start|>assistant<|channel|>commentary to=functions.lookup",
            "<|start|>user<|channel|>final",
        ] {
            let (calls, normal) =
                parse_tool_calls_harmony_complete(header, &Default::default(), None)
                    .await
                    .unwrap();
            assert!(calls.is_empty(), "{header}");
            assert_eq!(normal.as_deref(), Some(""), "{header}");
            let input = format!("Before{header}");
            let (calls, normal) =
                parse_tool_calls_harmony_complete(&input, &Default::default(), None)
                    .await
                    .unwrap();
            assert!(calls.is_empty(), "{input}");
            assert_eq!(normal.as_deref(), Some(input.as_str()), "{input}");
        }
        let quoted = r#"{"literal":"<|start|>assistant<|channel|>final"}"#;
        let (calls, normal) = parse_tool_calls_harmony_complete(quoted, &Default::default(), None)
            .await
            .unwrap();
        assert!(calls.is_empty());
        assert_eq!(normal.as_deref(), Some(quoted));
        let input = "Before<|start|>user<|channel|>final<|message|>Private<|return|>";
        let (calls, normal) = parse_tool_calls_harmony_complete(input, &Default::default(), None)
            .await
            .unwrap();
        assert!(calls.is_empty());
        assert_eq!(normal.as_deref(), Some("Before"));
        let input = "Before<|start|>assistant<|channel|>commentary to=functions.lookup<|message|>{}<|return|>";
        let (calls, normal) = parse_tool_calls_harmony_complete(input, &Default::default(), None)
            .await
            .unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "lookup");
        assert_eq!(normal.as_deref(), Some("Before"));
    }

    #[test]
    fn cleanup_warning_bounds_request_derived_metadata() {
        // Tracing's callsite registry is process-global. Other parser tests
        // emit at this same callsite without a subscriber, so give the capture
        // its own registry while the rest of the suite remains parallel.
        const CHILD: &str = "DYNAMO_HARMONY_LOG_CAPTURE_CHILD";
        if std::env::var_os(CHILD).is_none() {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "tool_calling::harmony::harmony_parser::tests::cleanup_warning_bounds_request_derived_metadata", "--nocapture"])
                .env_clear()
                .env(CHILD, "1")
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "isolated cleanup warning contract failed: {}{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            return;
        }
        use std::sync::{Arc, Mutex};
        use tracing::{
            Event, Metadata, Subscriber,
            field::{Field, Visit},
            span::{Attributes, Id, Record},
        };

        #[derive(Clone)]
        struct WarningCapture(Arc<Mutex<Vec<String>>>);
        impl Subscriber for WarningCapture {
            fn enabled(&self, _: &Metadata<'_>) -> bool {
                true
            }
            fn new_span(&self, _: &Attributes<'_>) -> Id {
                Id::from_u64(1)
            }
            fn record(&self, _: &Id, _: &Record<'_>) {}
            fn record_follows_from(&self, _: &Id, _: &Id) {}
            fn enter(&self, _: &Id) {}
            fn exit(&self, _: &Id) {}
            fn event(&self, event: &Event<'_>) {
                struct Fields(String);
                impl Visit for Fields {
                    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
                        use std::fmt::Write;
                        write!(&mut self.0, "{}={value:?};", field.name()).unwrap();
                    }
                }
                let mut fields = Fields(String::new());
                event.record(&mut fields);
                self.0.lock().unwrap().push(fields.0);
            }
        }
        let captured = Arc::new(Mutex::new(Vec::new()));
        let mut input = "Before ".to_string();
        for index in 0..64 {
            let name = format!("REQUEST_PRIVATE_SENTINEL_{index}_{}", "x".repeat(2048));
            input.push_str(&format!(
                "<|channel|>commentary to=functions.{name}<|message|>{{}}<|return|>"
            ));
        }
        input.push_str(" After");
        // Vocabulary loading failure reaches this exact cleanup owner, but a
        // recorded model response cannot inject a missing encoding or inspect
        // tracing fields. Keep this observable log contract in the owner test.
        let normal = tracing::subscriber::with_default(WarningCapture(captured.clone()), || {
            strip_harmony_protocol_from_normal_text(&input, "harmony_encoding_unavailable", None)
        });
        assert_eq!(normal, "Before  After");
        let events = captured.lock().unwrap();
        assert_eq!(events.len(), 1);
        assert!(
            events[0].len() <= 1024,
            "cleanup warning emitted {} bytes for a {}-byte input",
            events[0].len(),
            input.len()
        );
        assert!(!events[0].contains("REQUEST_PRIVATE_SENTINEL"));
        assert!(events[0].contains("harmony_encoding_unavailable"));
        assert!(events[0].contains("commentary_tool_call"));
    }

    #[tokio::test]
    async fn numeric_argument_lexemes_survive_native_extraction() {
        for number in [
            "9007199254740993.0",
            "9007199254740993.1",
            "-9007199254740993.1",
            "1e-400",
            "1e400",
            "-0.0",
            "1.2300",
            "1e+20",
        ] {
            for arguments in [
                format!("{{\"value\":{number}}}"),
                format!("{{\"z\":\"雪 \\\" <|return|>\",\"a\":[{number},{{\"n\":{number}}}]}}"),
                format!("[{number},null,true]"),
            ] {
                for ending in ["<|call|>", "<|return|>"] {
                    let input = format!(
                        "<|start|>assistant<|channel|>commentary to=functions.f<|message|>{arguments}{ending}"
                    );
                    let (calls, normal) =
                        parse_tool_calls_harmony_complete(&input, &Default::default(), None)
                            .await
                            .unwrap();
                    assert_eq!(calls.len(), 1);
                    assert_eq!(calls[0].function.arguments, arguments, "{input}");
                    assert_eq!(normal.as_deref(), Some(""));
                }
            }
        }
        for malformed in ["", "{\"x\":", "[1,", "not json"] {
            assert_eq!(serialize_harmony_arguments(malformed), malformed);
        }
    }

    #[tokio::test]
    async fn quoted_json_prefix_keeps_surrounding_whitespace() {
        for json in [
            r#"{"x":"<|start|>"}"#,
            r#"["<|channel|>"]"#,
            r#""<|message|>""#,
        ] {
            for whitespace in [" \n", "    ", "\n\t"] {
                let prefix = format!("{whitespace}{json}\n");
                let (calls, normal) =
                    parse_tool_calls_harmony_complete(&prefix, &Default::default(), None)
                        .await
                        .unwrap();
                assert!(calls.is_empty());
                assert_eq!(normal.as_deref(), Some(prefix.as_str()));
                for frame in ["", "<|start|>assistant"] {
                    for ending in ["<|return|>", "<|call|>"] {
                        let input =
                            format!("{prefix}{frame}<|channel|>final<|message|>Done{ending}");
                        let (calls, normal) =
                            parse_tool_calls_harmony_complete(&input, &Default::default(), None)
                                .await
                                .unwrap();
                        assert!(calls.is_empty());
                        assert_eq!(normal, Some(format!("{prefix}Done")), "{input:?}");
                    }
                }
            }
        }
    }

    #[tokio::test]
    async fn framed_recovery_reuses_native_header_metadata() {
        for count in [2, 32, 64, 128] {
            let mut input = r#"{"literal":"<|start|>"}"#.to_string();
            for index in 0..count {
                input.push_str(&format!("<|start|>assistant<|channel|>commentary to=functions.f{index}<|message|>{{\"x\":1}}<|return|>"));
            }
            HARMONY_HEADER_PARSES.with(|value| value.set(0));
            let (calls, _) = parse_tool_calls_harmony_complete(&input, &Default::default(), None)
                .await
                .unwrap();
            assert_eq!(calls.len(), count);
            let parses = HARMONY_HEADER_PARSES.with(std::cell::Cell::get);
            // Unique names prevent a same-header cache from hiding repeated
            // native parsing across the shared request-local inventory.
            assert!(
                parses <= count + 4,
                "{count} calls reparsed {parses} headers"
            );
        }
    }

    #[tokio::test]
    async fn native_unframed_prefixes_keep_recipient_and_role_ownership() {
        use crate::{GptOssReasoningParser, ReasoningParser};

        let args = "{\"q\":\"雪\"}";
        let whitespace: Vec<_> = (0u8..=127)
            .filter(u8::is_ascii_whitespace)
            .map(char::from)
            .collect();
        let mut prefixes = vec!["functions.lookup".to_string()];
        for separator in whitespace {
            prefixes.push(format!("assistant{separator}to=functions.lookup"));
            prefixes.push(format!("assistant{separator}functions.lookup"));
        }
        for prefix in prefixes {
            for channel in ["analysis", "commentary"] {
                for ending in ["<|call|>", "<|return|>"] {
                    let input = format!("{prefix}<|channel|>{channel}<|message|>{args}{ending}");
                    let (calls, normal) =
                        parse_tool_calls_harmony_complete(&input, &Default::default(), None)
                            .await
                            .unwrap();
                    assert_eq!(calls.len(), 1, "direct {input}: {normal:?}");
                    assert_eq!(calls[0].function.name, "lookup", "{input}");
                    assert_eq!(calls[0].function.arguments, args, "{input}");
                    assert_eq!(normal.as_deref(), Some(""), "{input}");
                    let batch = GptOssReasoningParser::new()
                        .unwrap()
                        .detect_and_parse_reasoning(&input, &[]);
                    assert_eq!(batch.reasoning_text, "", "{input}");
                    let (calls, normal) = parse_tool_calls_harmony_complete(
                        &batch.normal_text,
                        &Default::default(),
                        None,
                    )
                    .await
                    .unwrap();
                    assert_eq!(calls.len(), 1, "batch {input}: {batch:?}");
                    assert_eq!(calls[0].function.arguments, args, "{input}");
                    assert_eq!(normal.as_deref(), Some(""), "{input}");
                }
            }
        }
        for separator in (0u8..=127).filter(u8::is_ascii_whitespace).map(char::from) {
            let input = format!("assistant{separator}<|channel|>final<|message|>Done<|return|>");
            let (calls, normal) =
                parse_tool_calls_harmony_complete(&input, &Default::default(), None)
                    .await
                    .unwrap();
            assert!(calls.is_empty(), "{input}");
            assert_eq!(normal.as_deref(), Some("Done"), "{input}");
        }
    }
    #[tokio::test]
    async fn native_unframed_prefixes_preserve_nonfunction_and_prose_controls() {
        for prefix in [
            "python",
            "assistant python",
            "to=python",
            "assistant to=python",
            "assistant\u{a0}functions.lookup",
            "assistant\u{a0}to=functions.lookup",
            "assistant\u{b}to=functions.lookup",
            "assistantx",
        ] {
            let input =
                format!("{prefix}<|channel|>commentary<|message|>{{\"q\":\"x\"}}<|return|>");
            let (calls, normal) =
                parse_tool_calls_harmony_complete(&input, &Default::default(), None)
                    .await
                    .unwrap();
            assert!(calls.is_empty(), "{input}: {calls:?}");
            assert_eq!(normal.as_deref(), Some(""), "{input}");
        }
        let input = "Before <|start|>assistant<|channel|>final<|message|>Done<|return|>";
        let (calls, normal) = parse_tool_calls_harmony_complete(input, &Default::default(), None)
            .await
            .unwrap();
        assert!(calls.is_empty());
        assert_eq!(normal.as_deref(), Some("Before Done"));
    }

    #[tokio::test]
    async fn native_unframed_prefixes_survive_each_text_and_token_split() {
        use crate::{GptOssReasoningParser, ReasoningParser};

        let encoding = get_harmony_encoding().await.as_ref().unwrap();
        for prefix in [
            "assistant\tto=functions.lookup",
            "functions.@f",
            "assistant functions.lookup",
        ] {
            for ending in ["<|call|>", "<|return|>"] {
                let input =
                    format!("{prefix}<|channel|>commentary<|message|>{{\"q\":\"雪\"}}{ending}");
                let tokens = encoding.tokenizer().encode_with_special_tokens(&input);
                let text_splits: Vec<_> = input
                    .char_indices()
                    .map(|(i, _)| i)
                    .chain([input.len()])
                    .collect();
                for token_mode in [false, true] {
                    let splits: Vec<_> = if token_mode {
                        (0..=tokens.len()).collect()
                    } else {
                        text_splits.clone()
                    };
                    for split in splits {
                        let mut parser = GptOssReasoningParser::new().unwrap();
                        let mut normal = String::new();
                        let mut reasoning = String::new();
                        for half in [false, true] {
                            let result = if token_mode {
                                parser.parse_reasoning_streaming_incremental(
                                    "",
                                    if half {
                                        &tokens[split..]
                                    } else {
                                        &tokens[..split]
                                    },
                                )
                            } else {
                                parser.parse_reasoning_streaming_incremental(
                                    if half {
                                        &input[split..]
                                    } else {
                                        &input[..split]
                                    },
                                    &[],
                                )
                            };
                            normal.push_str(&result.normal_text);
                            reasoning.push_str(&result.reasoning_text);
                        }
                        let tail = parser.finish_reasoning_stream();
                        normal.push_str(&tail.normal_text);
                        reasoning.push_str(&tail.reasoning_text);
                        let repeated = parser.finish_reasoning_stream();
                        assert_eq!(repeated.normal_text, "");
                        assert_eq!(repeated.reasoning_text, "");
                        assert_eq!(reasoning, "", "{token_mode}/{split}: {input}");
                        let (calls, visible) =
                            parse_tool_calls_harmony_complete(&normal, &Default::default(), None)
                                .await
                                .unwrap();
                        assert_eq!(calls.len(), 1, "{token_mode}/{split}: {input}: {normal}");
                        assert_eq!(
                            calls[0].function.name,
                            if prefix == "functions.@f" {
                                "@f"
                            } else {
                                "lookup"
                            }
                        );
                        assert_eq!(calls[0].function.arguments, "{\"q\":\"雪\"}");
                        assert_eq!(visible.as_deref(), Some(""));
                    }
                }
            }
        }
    }

    #[tokio::test]
    async fn recovery_projects_native_final_header_metadata() {
        let prefix = r#"{"visible":"<|start|>"}"#;
        for role in ["assistant", "user", "python"] {
            for frame in ["", "<|start|>"] {
                if frame.is_empty() && role != "assistant" {
                    continue;
                }
                for header_stop in ["<|call|>", "<|return|>"] {
                    for body_stop in ["<|call|>", "<|return|>"] {
                        for suffix in [
                            "",
                            "<|start|>assistant<|channel|>commentary to=functions.f/x<|message|>{\"x\":2}<|return|>",
                        ] {
                            let role_prefix = if frame.is_empty() {
                                String::new()
                            } else {
                                format!("{frame}{role}")
                            };
                            let input = format!(
                                "{prefix}<|channel|>commentary to=functions.bad<|message|>{{\"q\":{role_prefix}<|channel|>final {header_stop}<|message|>Done.{body_stop}{suffix}"
                            );
                            let (calls, normal) = parse_tool_calls_harmony_complete(
                                &input,
                                &Default::default(),
                                None,
                            )
                            .await
                            .unwrap();
                            assert_eq!(
                                normal.as_deref(),
                                Some(
                                    format!(
                                        "{prefix}{}",
                                        if role == "assistant" { "Done." } else { "" }
                                    )
                                    .as_str()
                                ),
                                "{input}: {calls:?}"
                            );
                            assert_eq!(calls.len(), usize::from(!suffix.is_empty()), "{input}");
                            if !suffix.is_empty() {
                                assert_eq!(calls[0].function.name, "f/x");
                                assert_eq!(calls[0].function.arguments, "{\"x\":2}");
                            }
                        }
                    }
                }
            }
        }
    }

    #[tokio::test]
    async fn recovery_stops_belong_to_native_bodies_not_headers() {
        let encoding = get_harmony_encoding().await.as_ref().unwrap();
        for (channel, recipient, tool) in [
            ("commentary", " to=functions.lookup", true),
            ("final", "", false),
            ("analysis", "", false),
            ("commentary", " to=python", false),
            ("other", "", false),
        ] {
            for frame in ["", "<|start|>assistant"] {
                for header_stop in ["<|call|>", "<|return|>"] {
                    for body_stop in ["<|call|>", "<|return|>"] {
                        for first_stop in ["", "<|return|>"] {
                            for later in [false, true] {
                                let body = if tool { "{\"x\":2}" } else { "Done" };
                                let second = format!(
                                    "{frame}<|channel|>{channel}{recipient} {header_stop}<|message|>{body}{body_stop}"
                                );
                                let suffix = if later {
                                    "<|start|>assistant<|channel|>commentary to=functions.last<|message|>{\"x\":3}<|return|>"
                                } else {
                                    ""
                                };
                                let input = format!(
                                    "<|channel|>commentary to=functions.first<|message|>{{\"x\":1}}{first_stop}{second}{suffix}"
                                );
                                let (calls, residual) =
                                    extract_calls_from_headers(&input, encoding);
                                let mut expected = Vec::new();
                                if !first_stop.is_empty() {
                                    expected.push(("first", "{\"x\":1}"));
                                }
                                if tool {
                                    expected.push(("lookup", "{\"x\":2}"));
                                }
                                if later {
                                    expected.push(("last", "{\"x\":3}"));
                                }
                                let actual: Vec<_> = calls
                                    .iter()
                                    .map(|call| {
                                        (
                                            call.function.name.as_str(),
                                            call.function.arguments.as_str(),
                                        )
                                    })
                                    .collect();
                                assert_eq!(actual, expected, "{input}");
                                assert_eq!(
                                    residual,
                                    if tool { "" } else { second.as_str() },
                                    "{input}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    // Unified requests assert parser events, not the amount of input rescanned.
    #[tokio::test]
    async fn recovery_boundary_work_is_bounded_by_argument_bytes() {
        let encoding = get_harmony_encoding().await.as_ref().unwrap();
        let prefix = r#"{"literal":"<|start|>"}"#;
        let arguments = r#"{"x":1}"#;
        for count in [2, 32, 64, 128] {
            let input = format!("{prefix}{}", format!("<|start|>assistant<|channel|>commentary to=functions.lookup<|message|>{arguments}<|return|>").repeat(count));
            HARMONY_BOUNDARY_BYTES.with(|bytes| bytes.set(0));
            let (calls, residual) = extract_calls_from_headers(&input, encoding);
            assert_eq!(calls.len(), count);
            for call in calls {
                assert_eq!(call.function.name, "lookup");
                assert_eq!(call.function.arguments, arguments);
            }
            assert_eq!(residual, prefix);
            let scanned = HARMONY_BOUNDARY_BYTES.with(std::cell::Cell::get);
            assert!(
                scanned <= count * arguments.len(),
                "{count} calls scanned {scanned} bytes outside the argument windows"
            );
        }
    }

    // Corpus requests cannot inject a native encoding-loader failure.
    #[test]
    fn encoding_unavailable_preserves_only_known_assistant_text() {
        for (input, expected) in [
            (
                "<|start|>assistant<|channel|>final<|message|>ANSWER<|return|>",
                "ANSWER",
            ),
            ("bare answer", "bare answer"),
            (
                "<|start|>user<|channel|>final<|message|>SHOULD_HIDE<|return|>",
                "",
            ),
            (
                "<|start|>system<|channel|>analysis<|message|>SHOULD_HIDE<|end|>",
                "",
            ),
            (
                "<|start|>python<|channel|>commentary to=functions.fake<|message|>{\"x\":1}<|call|>",
                "",
            ),
            (
                "<|start|>assistant\u{a0}user<|channel|>final<|message|>SHOULD_HIDE<|return|>",
                "",
            ),
            (
                "Before <|start|>assistant<|channel|>final<|message|>ANSWER<|return|> After",
                "Before ANSWER After",
            ),
        ] {
            assert_eq!(
                strip_harmony_protocol_from_normal_text(
                    input,
                    "harmony_encoding_unavailable",
                    None
                ),
                expected,
                "{input}"
            );
        }
    }

    #[tokio::test]
    async fn fallback_role_words_keep_native_boundaries() {
        for role in ["assistant<g", "assistant\u{a0}user"] {
            let input = format!("<|start|>{role}<|channel|>final<|message|>SHOULD_HIDE<|return|>");
            let (calls, normal) =
                parse_tool_calls_harmony_complete(&input, &Default::default(), None)
                    .await
                    .unwrap();
            assert!(calls.is_empty());
            assert_eq!(normal.as_deref(), Some(""), "{input}");
            assert_eq!(
                strip_harmony_protocol_from_normal_text(
                    &input,
                    "harmony_encoding_unavailable",
                    None
                ),
                "",
                "unavailable {input}"
            );
        }
    }

    #[tokio::test]
    async fn recovery_orphan_stop_keeps_completed_messages_at_header_eof() {
        for prefix in ["", "<|start|>assistant"] {
            for orphan in ["", "<|call|>", "<|return|>"] {
                for tail in [
                    "",
                    "<|start|>assistant<|channel|>final",
                    "<|start|>assistant<|channel|>analysis to=functions.lookup",
                ] {
                    let input =
                        format!("{prefix}<|channel|>commentary<|message|>one<|end|>{orphan}{tail}");
                    let (calls, normal) = super::parse_tool_calls_harmony_complete(
                        &input,
                        &super::JsonParserConfig::default(),
                        None,
                    )
                    .await
                    .unwrap();
                    assert!(calls.is_empty());
                    assert_eq!(normal.as_deref(), Some("one"), "{input}");
                }
            }
        }
        for recipient in ["", "browser", "functions.lookup"] {
            let input = format!("<|channel|>finalx to={recipient}<|message|>secret<|return|>");
            let (calls, normal) = super::parse_tool_calls_harmony_complete(
                &input,
                &super::JsonParserConfig::default(),
                None,
            )
            .await
            .unwrap();
            assert!(calls.is_empty());
            assert_eq!(normal.as_deref(), Some(""));
        }
    }

    #[tokio::test]
    async fn recovery_recipient_first_header_survives_malformed_predecessor() {
        let input = r#"<|channel|>commentary to=functions.bad<|message|>{"x":to=functions.lookup<|channel|>commentary<|message|>{"path":"ok"}<|return|>"#;
        let (calls, _) = super::parse_tool_calls_harmony_complete(
            input,
            &super::JsonParserConfig::default(),
            None,
        )
        .await
        .unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "lookup");
        assert_eq!(calls[0].function.arguments, r#"{"path":"ok"}"#);
    }

    #[tokio::test]
    async fn recovery_unfinished_recipient_first_header_does_not_leak() {
        for channel in ["commentary", "analysis"] {
            for ending in ["<|call|>", "<|return|>"] {
                for suffix in [
                    "",
                    "<|start|>assistant<|channel|>final<|message|>Done.<|return|>",
                ] {
                    let input = format!(
                        "<|start|>assistant to=functions.bad<|channel|>{channel}<|start|>assistant to=functions.lookup<|channel|>commentary<|message|>{{\"q\":\"x\"}}{ending}{suffix}"
                    );
                    let (calls, normal) = super::parse_tool_calls_harmony_complete(
                        &input,
                        &super::JsonParserConfig::default(),
                        None,
                    )
                    .await
                    .unwrap();
                    assert_eq!(calls.len(), 1);
                    assert_eq!(calls[0].function.name, "lookup");
                    assert_eq!(calls[0].function.arguments, r#"{"q":"x"}"#);
                    assert_eq!(
                        normal.as_deref(),
                        Some(if suffix.is_empty() { "" } else { "Done." }),
                        "{input}"
                    );
                }
            }
        }
    }

    use super::*;

    // Model-output fixtures cannot own or saturate a Tokio runtime blocking pool.
    #[test]
    fn warm_encoding_does_not_wait_for_the_blocking_pool() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .max_blocking_threads(1)
            .build()
            .unwrap();
        let cached = runtime.block_on(get_harmony_encoding());
        assert!(cached.is_ok());
        let (entered_tx, entered_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let blocker = runtime.spawn_blocking(move || {
            entered_tx.send(()).unwrap();
            release_rx.recv().unwrap();
        });
        entered_rx
            .recv_timeout(std::time::Duration::from_secs(10))
            .unwrap();
        let ready = runtime.block_on(async {
            let mut lookup = std::pin::pin!(get_harmony_encoding());
            let mut context = std::task::Context::from_waker(std::task::Waker::noop());
            matches!(
                std::future::Future::poll(lookup.as_mut(), &mut context),
                std::task::Poll::Ready(encoding) if std::ptr::eq(encoding, cached)
            )
        });
        release_tx.send(()).unwrap();
        runtime.block_on(blocker).unwrap();
        assert!(ready, "warm lookup waited behind unrelated blocking work");
    }

    fn extract_name_and_args(call: ToolCallResponse) -> (String, serde_json::Value) {
        let args: serde_json::Value = serde_json::from_str(&call.function.arguments).unwrap();
        (call.function.name, args)
    }

    // DEPRECATED(parser-fixture-duplicate): Duplicate of YAML fixture coverage: TOOLCALLING.batch.1 in tests/parity/toolcalling/fixtures/harmony/TOOLCALLING.batch.yaml.
    #[tokio::test] // TOOLCALLING.batch.1, TOOLCALLING.harmony.2
    async fn test_parse_tool_calls_harmony_complete_basic() {
        for ending in ["<|call|>", "<|return|>"] {
            for channel in ["commentary", "analysis"] {
                let text = r#"<|channel|>commentary to=functions.get_current_weather <|constrain|>json<|message|>{"format":"celsius","location":"San Francisco"}<|call|>"#;
                let text = text
                    .replace("<|call|>", ending)
                    .replace("commentary", channel);
                let (tool_calls, normal_content) =
                    parse_tool_calls_harmony_complete(&text, &Default::default(), None)
                        .await
                        .unwrap();
                assert_eq!(tool_calls.len(), 1, "ending={ending}");
                assert_eq!(normal_content, Some("".to_string()));
                let (name, args) = extract_name_and_args(tool_calls[0].clone());
                assert_eq!(name, "get_current_weather");
                assert_eq!(args["location"], "San Francisco");
                assert_eq!(args["format"], "celsius");
            }
        }
    }

    #[tokio::test]
    async fn test_harmony_quoted_action_stops_remain_arguments() {
        use crate::tool_calling::parsers::detect_and_parse_tool_call_with_recovery;

        for literal in HARMONY_MARKERS {
            for ending in ["<|call|>", "<|return|>"] {
                for channel in ["commentary", "analysis"] {
                    for prefix in ["", "<|start|>assistant", "PREFIX "] {
                        for value in [
                            format!("literal {literal}"),
                            format!("escaped \"quote {literal} and slash \\"),
                        ] {
                            let arguments = serde_json::json!({"x": value}).to_string();
                            let text = format!(
                                "{prefix}<|channel|>{channel} to=functions.lookup<|message|>{arguments}{ending}"
                            );
                            let (calls, normal) = detect_and_parse_tool_call_with_recovery(
                                &text,
                                Some("harmony"),
                                None,
                            )
                            .await
                            .unwrap();
                            assert_eq!(calls.len(), 1, "{text}");
                            assert_eq!(calls[0].function.name, "lookup", "{text}");
                            assert_eq!(calls[0].function.arguments, arguments, "{text}");
                            assert_eq!(
                                normal.as_deref(),
                                Some(if prefix == "PREFIX " { "PREFIX " } else { "" }),
                                "{text}"
                            );
                        }
                    }
                }
            }
        }
        for literal in ["<|call|>", "<|return|>"] {
            let text = format!(
                "<|channel|>commentary to=functions.lookup<|message|>{{\"x\":\"literal {literal}\"}}"
            );
            let (calls, normal) =
                detect_and_parse_tool_call_with_recovery(&text, Some("harmony"), None)
                    .await
                    .unwrap();
            assert!(
                calls.is_empty(),
                "a quoted marker cannot complete a call: {text}"
            );
            assert_eq!(normal.as_deref(), Some(""));
        }
    }

    #[tokio::test]
    async fn test_parse_harmony_requires_each_tool_message_to_end() {
        for ending in ["", "<|end|>"] {
            let text = format!(
                "<|channel|>commentary to=functions.lookup<|message|>{{\"city\":\"Paris\"}}{ending}<|start|>assistant<|channel|>final<|message|>Done.<|return|>"
            );
            let (calls, _) = parse_tool_calls_harmony_complete(&text, &Default::default(), None)
                .await
                .unwrap();
            assert!(
                calls.is_empty(),
                "ending={ending:?}: a later return must not close an earlier tool message"
            );
        }
    }

    #[tokio::test]
    async fn test_harmony_regex_resumes_after_unfinished_call() {
        for (prefix, unfinished_args) in [
            ("PREFIX ", "{}"),
            ("", r#"{"text":"<|call|>"}"#),
            ("", r#"{"text":"<|return|>"}"#),
        ] {
            for unfinished_ending in ["", "<|end|>"] {
                for ending in ["<|call|>", "<|return|>"] {
                    for header_stop in ["", "<|return|>"] {
                        for suffix in [
                            "Done.",
                            "<|start|>assistant<|channel|>final<|message|>Done.<|return|>",
                        ] {
                            let text = format!(
                                "{prefix}<|channel|>commentary to=functions.unfinished<|message|>{unfinished_args}{unfinished_ending}<|start|>assistant<|channel|>commentary to=functions.lookup {header_stop}<|message|>{{\"city\":\"Paris\"}}{ending}{suffix}"
                            );
                            let (calls, normal) =
                                parse_tool_calls_harmony_complete(&text, &Default::default(), None)
                                    .await
                                    .unwrap();
                            assert_eq!(calls.len(), 1);
                            let (name, arguments) = extract_name_and_args(calls[0].clone());
                            assert_eq!(name, "lookup");
                            assert_eq!(arguments, serde_json::json!({"city": "Paris"}));
                            assert_eq!(
                                normal.as_deref(),
                                Some(format!("{prefix}Done.").as_str()),
                                "{text}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[tokio::test]
    async fn recovery_header_scan_stays_with_the_next_recipient() {
        let partial = "<|channel|>commentary to=functions.incomplete ";
        for tail in ["", "<|call|>", "<|return|>", "<|end|>", "<meta>"] {
            let text = format!(
                "{}to=functions.lookup<|channel|>commentary {tail}<|message|>{{}}<|return|>",
                partial.repeat(1000)
            );
            let encoding = get_harmony_encoding().await.as_ref().unwrap();
            let headers: Vec<_> = harmony_tool_headers(&text, encoding).collect();
            assert_eq!(headers.len(), 1);
            assert_eq!(headers[0].name, "lookup");
        }
        let text = "<|channel|>commentary to=functions.bad<|channel|>commentary to=functions.lookup<|message|>{}<|return|>";
        let encoding = get_harmony_encoding().await.as_ref().unwrap();
        let headers: Vec<_> = harmony_tool_headers(text, encoding).collect();
        assert_eq!(headers.len(), 1);
        assert_eq!(headers[0].name, "lookup");
    }

    #[tokio::test]
    async fn test_harmony_regex_does_not_extract_quoted_headers() {
        let encoding = get_harmony_encoding().await.as_ref().unwrap();
        for outer_channel in ["final", "commentary", "analysis"] {
            for tool_channel in ["commentary", "analysis"] {
                for framing in ["", "<|start|>assistant"] {
                    for complete_header in [false, true] {
                        let literal = format!(
                            "escaped \"quote and slash \\ {framing}<|channel|>{tool_channel} to=functions.fake{}",
                            if complete_header { "<|message|>{}" } else { "" }
                        );
                        for body in [
                            serde_json::json!(literal),
                            serde_json::json!({"example": literal}),
                            serde_json::json!([literal]),
                        ] {
                            for ending in ["<|call|>", "<|return|>"] {
                                let quoted = format!(
                                    "{framing}<|channel|>{outer_channel}<|message|>{body}{ending}"
                                );
                                for suffix in [
                                    "",
                                    " trailing",
                                    "<|channel|>final<|message|>Done.<|return|>",
                                ] {
                                    let text = format!("{quoted}{suffix}");
                                    let (calls, residual) =
                                        extract_calls_from_headers(&text, encoding);
                                    assert!(calls.is_empty(), "{text}");
                                    assert_eq!(residual, text, "{text}");
                                }
                                for arguments in ["{}", "not json <|channel|> at all"] {
                                    let text = format!(
                                        "{quoted}<|start|>assistant<|channel|>{tool_channel} to=functions.real<|message|>{arguments}{ending} trailing"
                                    );
                                    let (calls, residual) =
                                        extract_calls_from_headers(&text, encoding);
                                    assert_eq!(calls.len(), 1, "{text}");
                                    assert_eq!(calls[0].function.name, "real", "{text}");
                                    assert_eq!(calls[0].function.arguments, arguments, "{text}");
                                    assert_eq!(residual, format!("{quoted} trailing"), "{text}");
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[tokio::test]
    async fn test_harmony_recovery_preserves_final_json_markers() {
        use crate::tool_calling::parsers::detect_and_parse_tool_call_with_recovery;

        for channel in ["commentary", "analysis"] {
            for complete_header in [false, true] {
                let literal = format!(
                    "<|channel|>{channel} to=functions.fake{} literal <|return|> <|call|>",
                    if complete_header { "<|message|>{}" } else { "" }
                );
                let body = serde_json::json!({"example": literal}).to_string();
                for framing in ["", "<|start|>assistant"] {
                    for (suffix, visible) in [
                        (" trailing", " trailing"),
                        ("<|channel|>final<|message|>Done.<|return|>", "Done."),
                    ] {
                        let text =
                            format!("{framing}<|channel|>final<|message|>{body}<|return|>{suffix}");
                        let (calls, normal) =
                            detect_and_parse_tool_call_with_recovery(&text, Some("harmony"), None)
                                .await
                                .unwrap();
                        assert!(calls.is_empty(), "{text}");
                        assert_eq!(
                            normal.as_deref(),
                            Some(format!("{body}{visible}").as_str()),
                            "{text}"
                        );
                    }
                }
            }
        }
    }

    #[tokio::test]
    async fn test_harmony_visible_json_survives_prior_messages_and_inline_prose() {
        use crate::tool_calling::parsers::detect_and_parse_tool_call_with_recovery;

        let literal =
            "escaped \" \\ <|channel|>commentary to=functions.fake<|message|>{}<|return|>";
        for body in [
            serde_json::json!(literal),
            serde_json::json!({"example": literal}),
            serde_json::json!([literal]),
        ] {
            let body = body.to_string();
            for prefix in ["", "Preamble\n", "{}\n"] {
                let text = format!("{prefix}{body}");
                let (calls, normal) =
                    detect_and_parse_tool_call_with_recovery(&text, Some("harmony"), None)
                        .await
                        .unwrap();
                assert!(calls.is_empty(), "{text}");
                assert_eq!(normal.as_deref(), Some(text.as_str()), "{text}");
            }
            if body.starts_with(['{', '[']) {
                let text = format!("<|channel|>final<|message|>Example: {body}<|return|>");
                let (calls, normal) =
                    detect_and_parse_tool_call_with_recovery(&text, Some("harmony"), None)
                        .await
                        .unwrap();
                assert!(calls.is_empty(), "{text}");
                assert_eq!(
                    normal.as_deref(),
                    Some(format!("Example: {body}").as_str()),
                    "{text}"
                );
            }
        }
    }

    // With an unclosed final string, an exact corpus oracle cannot distinguish
    // literal stops from terminal stops. Assert no executable call and keep the known prefix.
    #[tokio::test]
    async fn test_harmony_unclosed_final_json_never_fabricates_a_call() {
        for channel in ["commentary", "analysis"] {
            for ending in ["<|call|>", "<|return|>"] {
                for prefix in ["", "Example: "] {
                    let body = format!(
                        "{prefix}{{\"example\":\"<|channel|>{channel} to=functions.fake<|message|>{{}}{ending}"
                    );
                    let text = format!("<|channel|>final<|message|>{body}<|return|>");
                    let (calls, normal) =
                        parse_tool_calls_harmony_complete(&text, &Default::default(), None)
                            .await
                            .unwrap();
                    assert!(calls.is_empty(), "{text}");
                    assert!(
                        normal.as_deref().unwrap().starts_with(&body),
                        "{text}: {normal:?}"
                    );
                }
            }
        }
    }

    #[tokio::test]
    async fn test_recipientless_commentary_uses_only_its_own_unquoted_call_stop() {
        for literal in HARMONY_MARKERS {
            for body in [
                serde_json::json!({"x": literal}),
                serde_json::json!([literal]),
            ] {
                let body = body.to_string();
                for ending in ["<|end|>", "<|return|>", ""] {
                    for framing in ["", "<|start|>assistant"] {
                        let text =
                            format!("{framing}<|channel|>commentary<|message|>{body}{ending}");
                        let (calls, normal) =
                            parse_tool_calls_harmony_complete(&text, &Default::default(), None)
                                .await
                                .unwrap();
                        assert!(calls.is_empty(), "{text}");
                        assert_eq!(normal.as_deref(), Some(body.as_str()), "{text}");
                    }
                }
                let visible =
                    format!("<|start|>assistant<|channel|>commentary<|message|>{body}<|end|>");
                let invalid_call = "<|start|>assistant<|channel|>commentary<|message|>{\"not_visible\":true}<|call|>";
                for text in [
                    format!("{visible}{invalid_call}"),
                    format!("{invalid_call}{visible}"),
                ] {
                    let (calls, normal) =
                        parse_tool_calls_harmony_complete(&text, &Default::default(), None)
                            .await
                            .unwrap();
                    assert!(calls.is_empty(), "{text}");
                    assert_eq!(normal.as_deref(), Some(body.as_str()), "{text}");
                }
            }
        }
    }

    #[tokio::test]
    async fn test_harmony_regex_skips_stops_before_arguments() {
        let encoding = get_harmony_encoding().await.as_ref().unwrap();
        for ending in ["<|call|>", "<|return|>"] {
            let text = format!(
                "PREFIX {ending}<|channel|>commentary to=functions.lookup {ending}<|message|>{{}}{ending} SUFFIX"
            );
            let (calls, residual) = extract_calls_from_headers(&text, encoding);
            assert_eq!(calls.len(), 1);
            assert_eq!(calls[0].function.name, "lookup");
            assert_eq!(calls[0].function.arguments, "{}");
            assert_eq!(residual, format!("PREFIX {ending} SUFFIX"));
        }
    }

    #[tokio::test]
    async fn test_parse_tool_calls_harmony_dotted_name() {
        let text = r#"<|channel|>commentary to=functions.my-tool.v2 <|constrain|>json<|message|>{"location":"Beijing"}<|call|>"#;
        let (tool_calls, _) = parse_tool_calls_harmony_complete(text, &Default::default(), None)
            .await
            .unwrap();
        let (name, args) = extract_name_and_args(tool_calls[0].clone());
        assert_eq!(name, "my-tool.v2");
        assert_eq!(args["location"], "Beijing");
    }

    // DEPRECATED(parser-fixture-duplicate): Duplicate of YAML fixture coverage: TOOLCALLING.batch.4.d in tests/parity/toolcalling/fixtures/harmony/TOOLCALLING.batch.4.yaml.
    #[tokio::test] // TOOLCALLING.batch.4, TOOLCALLING.harmony.2
    async fn test_parse_tools_harmony_without_start_token() {
        let text = r#"<|channel|>analysis<|message|>Need to use function get_current_weather.<|end|><|message|>{"location":"San Francisco"}<|call|>"#;
        let (tool_calls, normal_content) =
            parse_tool_calls_harmony_complete(text, &Default::default(), None)
                .await
                .unwrap();
        assert_eq!(normal_content, Some("".to_string()));
        assert_eq!(tool_calls.len(), 0);
    }

    // DEPRECATED(parser-fixture-duplicate): Duplicate of YAML fixture coverage: TOOLCALLING.batch.7.d, TOOLCALLING.batch.8.a in tests/parity/toolcalling/fixtures/harmony/TOOLCALLING.batch.7.yaml, tests/parity/toolcalling/fixtures/harmony/TOOLCALLING.batch.8.yaml.
    #[tokio::test] // TOOLCALLING.batch.7, TOOLCALLING.batch.8, TOOLCALLING.harmony.2
    async fn test_parse_tool_calls_harmony_with_multi_args() {
        let text = r#"<|channel|>analysis<|message|>Need to use function get_current_weather.<|end|><|start|>assistant<|channel|>commentary to=functions.get_current_weather <|constrain|>json<|message|>{"location":"San Francisco", "unit":"fahrenheit"}<|call|>"#;
        let (tool_calls, normal_content) =
            parse_tool_calls_harmony_complete(text, &Default::default(), None)
                .await
                .unwrap();
        assert_eq!(normal_content, Some("".to_string()));
        assert_eq!(tool_calls.len(), 1);
        let (name, args) = extract_name_and_args(tool_calls[0].clone());
        assert_eq!(name, "get_current_weather");
        assert_eq!(args["location"], "San Francisco");
        assert_eq!(args["unit"], "fahrenheit");
    }

    // DEPRECATED(parser-fixture-duplicate): Duplicate of YAML fixture coverage: TOOLCALLING.batch.8.a in tests/parity/toolcalling/fixtures/harmony/TOOLCALLING.batch.8.yaml.
    #[tokio::test] // TOOLCALLING.batch.8, TOOLCALLING.batch.8, TOOLCALLING.harmony.2
    async fn test_parse_tool_calls_harmony_with_normal_text() {
        let text = r#"<|channel|>analysis<|message|>Need to use function get_current_weather.<|end|><|start|>assistant<|channel|>commentary to=functions.get_current_weather <|constrain|>json<|message|>{"location":"San Francisco"}<|call|>"#;
        let (tool_calls, normal_content) =
            parse_tool_calls_harmony_complete(text, &Default::default(), None)
                .await
                .unwrap();
        assert_eq!(normal_content, Some("".to_string()));
        assert_eq!(tool_calls.len(), 1);
        let (name, args) = extract_name_and_args(tool_calls[0].clone());
        assert_eq!(name, "get_current_weather");
        assert_eq!(args["location"], "San Francisco");
    }

    #[tokio::test]
    async fn completed_native_messages_keep_unframed_visible_tail() {
        for prefix in [
            "<|channel|>final<|message|>before<|return|>",
            "<|channel|>commentary to=functions.f/x<|message|>{}<|call|>",
        ] {
            let input = format!("{prefix}after");
            let (calls, normal) =
                parse_tool_calls_harmony_complete(&input, &Default::default(), None)
                    .await
                    .unwrap();
            if prefix.contains("functions.") {
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].function.name, "f/x");
                assert_eq!(calls[0].function.arguments, "{}");
                assert_eq!(normal.as_deref(), Some("after"));
            } else {
                assert!(calls.is_empty());
                assert_eq!(normal.as_deref(), Some("beforeafter"));
            }
        }
    }

    #[tokio::test] // TOOLCALLING.batch.3 — gpt-oss
    async fn test_parse_harmony_bare_text_survives_reasoning_handoff() {
        use crate::{GptOssReasoningParser, ReasoningParser};

        for text in [
            "Hello, how can I help you today?",
            "visible\ndone",
            "first\nsecond",
            " 雪 \n",
            "tools are available",
        ] {
            let mut reasoning = GptOssReasoningParser::new().unwrap();
            let handoff = reasoning.detect_and_parse_reasoning(
                &format!("<|channel|>final<|message|>{text}<|return|>"),
                &[],
            );
            assert_eq!(handoff.normal_text, text);
            assert_eq!(handoff.reasoning_text, "");
            let (calls, normal) =
                parse_tool_calls_harmony_complete(&handoff.normal_text, &Default::default(), None)
                    .await
                    .unwrap();
            assert!(calls.is_empty());
            assert_eq!(normal.as_deref(), Some(text));
        }
    }

    #[tokio::test] // TOOLCALLING.batch.3 — gpt-oss
    async fn test_parse_harmony_final_message_returns_normal_text() {
        let text = r#"<|channel|>final<|message|>Hello, how can I help you today?<|return|>"#;
        let (tool_calls, normal_content) =
            parse_tool_calls_harmony_complete(text, &Default::default(), None)
                .await
                .unwrap();
        assert!(tool_calls.is_empty());
        assert_eq!(
            normal_content,
            Some("Hello, how can I help you today?".to_string())
        );
    }

    #[tokio::test]
    async fn test_parse_harmony_tool_only_keeps_final_as_normal_text() {
        let text = r#"<|channel|>analysis<|message|>Need to check weather.<|end|><|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{"location":"NYC"}<|call|><|start|>assistant<|channel|>final<|message|>I checked the weather.<|return|>"#;
        let (tool_calls, normal_content) =
            parse_tool_calls_harmony_complete(text, &Default::default(), None)
                .await
                .unwrap();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(normal_content, Some("I checked the weather.".to_string()));
        let (name, args) = extract_name_and_args(tool_calls[0].clone());
        assert_eq!(name, "get_weather");
        assert_eq!(args["location"], "NYC");
    }

    #[tokio::test]
    async fn test_parse_harmony_drops_directed_non_function_commentary() {
        let text = r#"<|start|>assistant<|channel|>commentary to=browser.search <|constrain|>json<|message|>{"query":"secret weather"}<|call|><|start|>assistant<|channel|>final<|message|>Done.<|return|>"#;
        let (tool_calls, normal_content) =
            parse_tool_calls_harmony_complete(text, &Default::default(), None)
                .await
                .unwrap();

        assert!(tool_calls.is_empty());
        let normal = normal_content.unwrap_or_default();
        assert_eq!(normal, "Done.");
        assert!(!normal.contains("secret weather"));
        assert!(!normal.contains("browser.search"));
        assert!(!normal.contains("<|channel|>"));
    }

    #[tokio::test]
    async fn test_parse_harmony_drops_recipientless_tool_call_payload() {
        let text = r#"<|start|>assistant<|channel|>commentary <|constrain|>json<|message|>{"location":"NYC"}<|call|><|start|>assistant<|channel|>final<|message|>Done.<|return|>"#;
        let (tool_calls, normal_content) =
            parse_tool_calls_harmony_complete(text, &Default::default(), None)
                .await
                .unwrap();

        assert!(tool_calls.is_empty());
        let normal = normal_content.unwrap_or_default();
        assert_eq!(normal, "Done.");
        assert!(!normal.contains("NYC"));
        assert!(!normal.contains("<|channel|>"));
    }

    // Harmony's strict tokenizer rejects two back-to-back commentary
    // blocks, so EOF recovery falls back to regex extraction and pins that
    // both calls are surfaced without leaking the raw envelopes.
    // DEPRECATED(parser-fixture-duplicate): Duplicate of YAML fixture coverage: TOOLCALLING.batch.2.b in tests/parity/toolcalling/fixtures/harmony/TOOLCALLING.batch.2.yaml.
    #[tokio::test] // TOOLCALLING.batch.2 — gpt-oss
    async fn test_parse_harmony_multiple_calls_recovers() {
        for ending in ["<|call|>", "<|return|>"] {
            let text = r#"<|start|>assistant<|channel|>commentary to=functions.a <|constrain|>json<|message|>{"x":1}<|call|><|start|>assistant<|channel|>commentary to=functions.b <|constrain|>json<|message|>{"y":2}<|call|>"#;
            let text = text.replace("<|call|>", ending);
            let (tool_calls, _normal) =
                parse_tool_calls_harmony_complete(&text, &Default::default(), None)
                    .await
                    .unwrap();
            assert_eq!(tool_calls.len(), 2);
            let (n0, a0) = extract_name_and_args(tool_calls[0].clone());
            let (n1, a1) = extract_name_and_args(tool_calls[1].clone());
            assert_eq!(n0, "a");
            assert_eq!(a0["x"], 1);
            assert_eq!(n1, "b");
            assert_eq!(a1["y"], 2);
        }
    }

    // DEPRECATED(parser-fixture-duplicate): Duplicate of YAML fixture coverage: TOOLCALLING.batch.4.a in tests/parity/toolcalling/fixtures/harmony/TOOLCALLING.batch.4.yaml.
    #[tokio::test] // TOOLCALLING.batch.4 — gpt-oss
    async fn test_parse_harmony_malformed_json_preserves_raw_arguments() {
        for arguments in [
            "not json at all",
            "not json <|channel|> at all",
            "not json <|start|> at all",
            "not json <|message|> at all",
        ] {
            for ending in ["<|call|>", "<|return|>"] {
                let text = format!(
                    "<|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{arguments}{ending}"
                );
                let (tool_calls, _normal) =
                    parse_tool_calls_harmony_complete(&text, &Default::default(), None)
                        .await
                        .unwrap();
                assert_eq!(tool_calls.len(), 1, "{text}");
                assert_eq!(tool_calls[0].function.name, "get_weather");
                assert_eq!(tool_calls[0].function.arguments, arguments);
            }
        }
    }

    // DEPRECATED(parser-fixture-duplicate): Duplicate of YAML fixture coverage: TOOLCALLING.batch.4.b in tests/parity/toolcalling/fixtures/harmony/TOOLCALLING.batch.4.yaml.
    #[tokio::test] // TOOLCALLING.batch.4 — gpt-oss
    async fn test_parse_harmony_unterminated_json_preserves_raw_arguments() {
        let text = r#"<|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{"location":"NYC<|call|>"#;
        let (tool_calls, _normal) = parse_tool_calls_harmony_complete(
            text,
            &JsonParserConfig {
                allow_eof_recovery: true,
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].function.name, "get_weather");
        assert_eq!(tool_calls[0].function.arguments, r#"{"location":"NYC"#);
    }

    // Bare-envelope TOOLCALLING.batch.5: no preceding `analysis` block, no `<|call|>`
    // at the end. Harmony requires the explicit call stop token, so fallback
    // must not accept EOS as a synthetic close.
    // DEPRECATED(parser-fixture-duplicate): Duplicate of YAML fixture coverage: TOOLCALLING.batch.5.a in tests/parity/toolcalling/fixtures/harmony/TOOLCALLING.batch.5.yaml.
    #[tokio::test] // TOOLCALLING.batch.5 — gpt-oss
    async fn test_parse_harmony_bare_envelope_no_call_token_drops() {
        let text = r#"<|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{"location":"NYC"}"#;
        let (tool_calls, normal) = parse_tool_calls_harmony_complete(
            text,
            &JsonParserConfig {
                allow_eof_recovery: true,
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();
        assert!(tool_calls.is_empty());
        assert_eq!(normal, Some("".to_string()));
    }

    // The regex fallback must preserve user-visible non-tool spans (prose before
    // the call and suffix after `<|call|>`) as `normal_text`, not zero them.
    #[tokio::test]
    async fn test_parse_harmony_regex_fallback_preserves_residual_text() {
        let text = r#"PREFIX <|channel|>analysis<|message|>Need a tool.<|end|><|start|>assistant<|channel|>commentary to=functions.a <|constrain|>json<|message|>{"x":1}<|call|> SUFFIX"#;
        let (tool_calls, normal) = parse_tool_calls_harmony_complete(
            text,
            &JsonParserConfig {
                allow_eof_recovery: true,
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();
        assert_eq!(tool_calls.len(), 1);
        let normal = normal.unwrap_or_default();
        assert!(
            normal.contains("PREFIX"),
            "normal must keep prefix: {normal:?}"
        );
        assert!(
            normal.contains("SUFFIX"),
            "normal must keep suffix: {normal:?}"
        );
        assert!(
            !normal.contains("Need a tool."),
            "normal must not expose Harmony analysis text: {normal:?}"
        );
        assert!(
            !normal.contains("<|start|>"),
            "normal must not leak harmony start token: {normal:?}"
        );
        assert!(
            !normal.contains("assistant"),
            "normal must not leak harmony assistant marker: {normal:?}"
        );
    }

    #[tokio::test]
    async fn test_parse_harmony_parse_failure_strips_protocol_from_normal_text() {
        let text = r#"Before <|start|>assistant<|channel|>commentary <|constrain|>json<|message|>{"location":"NYC"<|call|> After"#;
        let (tool_calls, normal) =
            parse_tool_calls_harmony_complete(text, &Default::default(), None)
                .await
                .unwrap();
        assert!(tool_calls.is_empty());
        let normal = normal.unwrap_or_default();
        assert_eq!(normal, "Before  After");
        for token in [
            "<|start|>",
            "<|channel|>",
            "<|constrain|>",
            "<|message|>",
            "<|call|>",
        ] {
            assert!(
                !normal.contains(token),
                "normal_text must not leak {token}: {normal:?}"
            );
        }
        assert!(
            !normal.contains("commentary"),
            "normal_text must not leak tool-call metadata: {normal:?}"
        );
    }

    #[tokio::test]
    async fn test_parse_harmony_parse_failure_strips_analysis_body() {
        let text = r#"Before <|channel|>analysis<|message|>Need to call get_weather.<|end|><|start|>assistant<|channel|>commentary <|constrain|>json<|message|>{"location":"NYC"<|call|> After"#;
        let (tool_calls, normal) =
            parse_tool_calls_harmony_complete(text, &Default::default(), None)
                .await
                .unwrap();
        assert!(tool_calls.is_empty());
        let normal = normal.unwrap_or_default();
        assert_eq!(normal, "Before  After");
        assert!(
            !normal.contains("<|channel|>"),
            "normal_text must not leak Harmony protocol tokens: {normal:?}"
        );
        assert!(
            !normal.contains("Need to call get_weather."),
            "normal_text must not expose Harmony analysis text: {normal:?}"
        );
    }

    // DEPRECATED(parser-fixture-duplicate): Duplicate of YAML fixture coverage: TOOLCALLING.batch.5.a, TOOLCALLING.batch.8.a in tests/parity/toolcalling/fixtures/harmony/TOOLCALLING.batch.5.yaml, tests/parity/toolcalling/fixtures/harmony/TOOLCALLING.batch.8.yaml.
    #[tokio::test] // TOOLCALLING.batch.4, TOOLCALLING.batch.5, TOOLCALLING.harmony.2
    async fn test_parse_tool_calls_harmony_without_call_token() {
        let text = r#"<|channel|>analysis<|message|>We need to call get_weather function. The user asks "What's the weather like in San Francisco in Celsius?" So location: "San Francisco, CA" unit: "celsius". Let's call function.<|end|><|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{"location":"San Francisco, CA","unit":"celsius"}"#;
        let (tool_calls, normal_content) =
            parse_tool_calls_harmony_complete(text, &Default::default(), None)
                .await
                .unwrap();
        assert_eq!(normal_content, Some("".to_string()));
        assert!(tool_calls.is_empty());
    }

    /// Parser-level invariant: the harmony parser is byte-stable — it
    /// doesn't see `finish_reason` and produces the same output regardless
    /// of the upstream stream-end reason. Real PIPELINE.finish_reason coverage (stop /
    /// tool_calls / length mapping) lives in
    /// `lib/llm/tests/test_streaming_tool_parsers.rs` and belongs in the
    /// cross-parser finish_reason mapping work-item (tracked separately).
    #[tokio::test]
    async fn test_harmony_parser_output_independent_of_upstream_finish() {
        let text = r#"<|channel|>commentary to=functions.get_current_weather <|constrain|>json<|message|>{"location":"NYC"}<|call|>"#;
        let (tool_calls, _) = parse_tool_calls_harmony_complete(text, &Default::default(), None)
            .await
            .unwrap();
        assert_eq!(tool_calls.len(), 1);
    }

    /// TOOLCALLING.batch.6 — empty args. A no-arg harmony call (`{}`) must still surface
    /// the function name.
    // DEPRECATED(parser-fixture-duplicate): Duplicate of YAML fixture coverage: TOOLCALLING.batch.6.a in tests/parity/toolcalling/fixtures/harmony/TOOLCALLING.batch.6.yaml.
    #[tokio::test] // TOOLCALLING.batch.6 — gpt-oss
    async fn test_parse_harmony_empty_args() {
        let text = r#"<|channel|>commentary to=functions.current_time <|constrain|>json<|message|>{}<|call|>"#;
        let (tool_calls, _) = parse_tool_calls_harmony_complete(text, &Default::default(), None)
            .await
            .unwrap();
        assert_eq!(tool_calls.len(), 1);
        let (name, args) = extract_name_and_args(tool_calls[0].clone());
        assert_eq!(name, "current_time");
        assert_eq!(args, serde_json::json!({}));
    }

    /// TOOLCALLING.batch.9 — empty / null content variants. Truly-empty (zero bytes)
    /// and whitespace-only inputs must yield no tool calls. Unlike the
    /// XML/JSON parsers (which trim whitespace down to `Some("")`), the
    /// harmony parser passes the input verbatim through to normal_text —
    /// pin that distinction here.
    // DEPRECATED(parser-fixture-duplicate): Duplicate of YAML fixture coverage: TOOLCALLING.batch.9 in tests/parity/toolcalling/fixtures/harmony/TOOLCALLING.batch.yaml.
    #[tokio::test] // TOOLCALLING.batch.9 — gpt-oss
    async fn test_parse_harmony_empty_and_whitespace_inputs() {
        for input in &["", " ", "\n", "\t\n  \t"] {
            let (tool_calls, normal) =
                parse_tool_calls_harmony_complete(input, &Default::default(), None)
                    .await
                    .unwrap();
            assert!(
                tool_calls.is_empty(),
                "Empty/whitespace input must yield no calls (input={:?})",
                input
            );
            assert_eq!(
                normal.as_deref(),
                Some(*input),
                "harmony passes empty/whitespace input verbatim to normal_text (input={:?})",
                input
            );
        }
    }

    /// TOOLCALLING.batch.10 — duplicate calls (same function name twice). Two
    /// back-to-back commentary blocks for the same function. Pin
    /// parser-level behavior — both calls returned with distinct ids
    /// and distinct args.
    // DEPRECATED(parser-fixture-duplicate): Duplicate of YAML fixture coverage: TOOLCALLING.batch.10 in tests/parity/toolcalling/fixtures/harmony/TOOLCALLING.batch.yaml.
    #[tokio::test] // TOOLCALLING.batch.10 — gpt-oss
    async fn test_parse_harmony_duplicate_calls_same_name() {
        let text = r#"<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{"city":"NYC"}<|call|><|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{"city":"LA"}<|call|>"#;
        let (tool_calls, _) = parse_tool_calls_harmony_complete(text, &Default::default(), None)
            .await
            .unwrap();
        assert_eq!(
            tool_calls.len(),
            2,
            "Both duplicate-name calls must be returned"
        );
        assert_ne!(
            tool_calls[0].id, tool_calls[1].id,
            "Duplicate calls must have distinct ids"
        );
        let (name0, args0) = extract_name_and_args(tool_calls[0].clone());
        let (name1, args1) = extract_name_and_args(tool_calls[1].clone());
        assert_eq!(name0, "get_weather");
        assert_eq!(name1, "get_weather");
        assert_eq!(args0["city"], "NYC");
        assert_eq!(args1["city"], "LA");
    }

    /// Regression test: the harmony parser must produce ids that are unique
    /// not just within a single invocation but also across multiple invocations.
    /// `jail.rs` slices the raw stream at each `<|call|>` boundary and invokes
    /// the parser once per section. With index-based ids ("call-1", "call-2", ...)
    /// each per-section invocation would restart from "call-1", and the
    /// downstream OpenAI Chat Completions response would carry duplicate ids,
    /// which clients (and Dynamo's own E2E harness) reject as
    /// `duplicate_tool_ids`. Using `Uuid::new_v4()` at construction guarantees
    /// uniqueness independent of how many times the parser is invoked.
    #[tokio::test]
    async fn test_parse_tool_calls_harmony_unique_ids_across_invocations() {
        // Two independent invocations, each on a single tool-call section —
        // matches the jail's per-<|call|> emission pattern.
        let chunk_a = r#"<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{"city":"SF"}<|call|>"#;
        let chunk_b = r#"<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{"city":"NY"}<|call|>"#;

        let (calls_a, _) = parse_tool_calls_harmony_complete(chunk_a, &Default::default(), None)
            .await
            .unwrap();
        let (calls_b, _) = parse_tool_calls_harmony_complete(chunk_b, &Default::default(), None)
            .await
            .unwrap();

        assert_eq!(calls_a.len(), 1, "first chunk should produce 1 call");
        assert_eq!(calls_b.len(), 1, "second chunk should produce 1 call");
        assert_ne!(
            calls_a[0].id, calls_b[0].id,
            "Tool call ids must be unique across separate parser invocations \
             (jail invokes the parser once per <|call|> boundary; identical ids \
             would surface as duplicate_tool_ids in the OpenAI response)",
        );
    }
}

#[cfg(test)]
mod detect_parser_tests {
    use super::*;

    #[test] // helper
    fn test_detect_tool_call_start_harmony_chunk_with_tool_call_start_token() {
        let text = r#"<|start|>assistant<|channel|>commentary to=functions.get_current_weather <|constrain|>json"#;
        let config = JsonParserConfig {
            tool_call_start_tokens: vec!["<|start|>assistant<|channel|>commentary".to_string()],
            tool_call_end_tokens: vec!["<|call|>".to_string()],
            ..Default::default()
        };
        let result = detect_tool_call_start_harmony(text, &config, false);
        assert!(result);
    }

    #[test] // helper
    fn test_detect_tool_call_start_harmony_chunk_without_tool_call_start_token() {
        // This is a warkaround for now. Right now everything is treated as tool call start token.
        // We need to improve this in the future.
        let text = r#"<|channel|>commentary to=functions.get_current_weather"#;
        let config = JsonParserConfig {
            tool_call_start_tokens: vec!["<|start|>assistant<|channel|>commentary".to_string()],
            tool_call_end_tokens: vec!["<|call|>".to_string()],
            ..Default::default()
        };
        let result = detect_tool_call_start_harmony(text, &config, false);
        assert!(result);
    }

    #[tokio::test]
    async fn test_parse_harmony_orphan_call_marker_does_not_leak() {
        let config = JsonParserConfig {
            tool_call_start_tokens: vec!["<|start|>assistant<|channel|>commentary".to_string()],
            tool_call_end_tokens: vec!["<|call|>".to_string()],
            ..Default::default()
        };

        assert!(detect_tool_call_start_harmony("<|call|>", &config, false));
        assert!(detect_tool_call_start_harmony("<|call|>", &config, true));

        let (tool_calls, normal) = parse_tool_calls_harmony_complete("<|call|>", &config, None)
            .await
            .unwrap();
        assert!(tool_calls.is_empty());
        assert_eq!(normal, Some("".to_string()));
    }

    #[test] // helper, TOOLCALLING.stream.3
    fn test_detect_tool_call_start_harmony_partial_tokens() {
        // Test partial token detection for streaming scenarios
        let config = JsonParserConfig {
            tool_call_start_tokens: vec!["<|start|>assistant<|channel|>commentary".to_string()],
            tool_call_end_tokens: vec!["<|call|>".to_string()],
            ..Default::default()
        };

        // Test various partial prefixes in strict mode
        assert!(
            detect_tool_call_start_harmony("<", &config, true),
            "'<' should be detected as potential start"
        );
        assert!(
            detect_tool_call_start_harmony("<|", &config, true),
            "'<|' should be detected as potential start"
        );
        assert!(
            detect_tool_call_start_harmony("<|start|>", &config, true),
            "'<|start|>' should be detected as potential start"
        );
        assert!(
            detect_tool_call_start_harmony("<|start|>assistant", &config, true),
            "'<|start|>assistant' should be detected as potential start"
        );

        // Test that unrelated text is not detected in strict mode
        assert!(
            !detect_tool_call_start_harmony("hello world", &config, true),
            "'hello world' should not be detected in strict mode"
        );
        assert!(
            !detect_tool_call_start_harmony("xyz", &config, true),
            "'xyz' should not be detected in strict mode"
        );

        // Channel-specific split markers over the production start-token
        // pair must still be detected; the jail, not this detector, decides
        // whether a short tail is prose.
        let prod_config = JsonParserConfig {
            tool_call_start_tokens: vec![
                "<|start|>assistant<|channel|>commentary".to_string(),
                "<|channel|>commentary".to_string(),
            ],
            tool_call_end_tokens: vec!["<|call|>".to_string()],
            ..Default::default()
        };
        for chunk in ["<|c", "<|channel|>comm", "<|start|>assistant<|ch"] {
            assert!(
                detect_tool_call_start_harmony(chunk, &prod_config, true),
                "{chunk:?} is a potential split marker and must be detected"
            );
        }
    }

    // --- analysis-channel tool-call recovery (the gpt-oss-120b malformed form) ---

    #[tokio::test]
    async fn test_parse_harmony_analysis_code_tool_call_recovered() {
        // Exact form emitted by the gpt-oss-120b worker: channel=analysis, constraint=code.
        let text =
            r#"<|channel|>analysis to=functions.grep code<|message|>{"pattern":"foo"}<|call|>"#;
        let (calls, normal) =
            parse_tool_calls_harmony_complete(text, &JsonParserConfig::default(), None)
                .await
                .expect("parser should not error");
        assert_eq!(calls.len(), 1, "analysis/code tool call must be recovered");
        assert_eq!(calls[0].function.name, "grep");
        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["pattern"], "foo");
        assert_eq!(
            normal,
            Some("".to_string()),
            "no markup should leak into normal_text"
        );
    }

    #[tokio::test]
    async fn test_parse_harmony_analysis_no_constraint_tool_call_recovered() {
        // analysis channel with no constraint marker at all.
        let text = r#"<|channel|>analysis to=functions.bash<|message|>{"command":"ls"}<|call|>"#;
        let (calls, _normal) =
            parse_tool_calls_harmony_complete(text, &JsonParserConfig::default(), None)
                .await
                .expect("parser should not error");
        assert_eq!(
            calls.len(),
            1,
            "analysis tool call without constraint must be recovered"
        );
        assert_eq!(calls[0].function.name, "bash");
    }

    #[tokio::test]
    async fn test_parse_harmony_recipientless_analysis_is_not_tool_call() {
        // Recipientless analysis is pure reasoning — must never become a tool call
        // and must not leak markup into normal_text.
        let text = r#"<|channel|>analysis<|message|>Let me think about grep.<|end|><|channel|>final<|message|>Done.<|return|>"#;
        let (calls, normal) =
            parse_tool_calls_harmony_complete(text, &JsonParserConfig::default(), None)
                .await
                .expect("parser should not error");
        assert!(
            calls.is_empty(),
            "recipientless analysis must not produce a tool call"
        );
        assert_eq!(
            normal,
            Some("Done.".to_string()),
            "only the final-channel text should appear in normal_text"
        );
    }

    #[tokio::test]
    async fn test_parse_harmony_analysis_tool_call_without_stop_drops() {
        // Analysis tool call missing the required <|call|> terminator must be dropped.
        let text = r#"<|channel|>analysis to=functions.grep code<|message|>{"pattern":"foo"}"#;
        let (calls, _normal) =
            parse_tool_calls_harmony_complete(text, &JsonParserConfig::default(), None)
                .await
                .expect("parser should not error");
        assert!(
            calls.is_empty(),
            "analysis tool call without <|call|> terminator must be dropped"
        );
    }

    #[tokio::test]
    async fn test_parse_harmony_analysis_tool_call_then_final() {
        // Reasoning block + analysis tool call + final answer: one tool call recovered,
        // normal_text = only the final-channel content.
        let text = concat!(
            "<|channel|>analysis<|message|>think<|end|>",
            "<|start|>assistant<|channel|>analysis to=functions.get_weather code<|message|>",
            r#"{"location":"NYC"}<|call|>"#,
            "<|start|>assistant<|channel|>final<|message|>Checked.<|return|>",
        );
        let (calls, normal) =
            parse_tool_calls_harmony_complete(text, &JsonParserConfig::default(), None)
                .await
                .expect("parser should not error");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(normal, Some("Checked.".to_string()));
    }
}
