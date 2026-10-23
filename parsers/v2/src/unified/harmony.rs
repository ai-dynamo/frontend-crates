// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unified GPT-OSS parser over the incremental Harmony message reader.

use std::collections::HashMap;
use std::panic::{RefUnwindSafe, UnwindSafe};

use openai_harmony::chat::Role;
use openai_harmony::{
    HarmonyEncoding, HarmonyEncodingName, StreamableParser, load_harmony_encoding,
};
use serde_json::Value;

use crate::tool_calling::traits::{Result, Tool, ToolCallDelta};
use crate::unified::{
    GuidedChannel, GuidedGrammar, GuidedReasoning, GuidedRouted, NativeUnified, UnifiedParser,
    UnifiedParserInit, UnifiedParserInputCapabilities, UnifiedParserOutput,
    UnifiedParserStartingState, UnifiedToolOutputMode,
};

const START: &str = "<|start|>";
const CHANNEL: &str = "<|channel|>";
const FORMAT: &str = "<|constrain|>";
const MESSAGE: &str = "<|message|>";
const MESSAGE_END: &str = "<|end|>";
pub(crate) const TURN_END: &str = "<|return|>";
const HANDOFF: &str = "<|call|>";
const CONTROL_MARKERS: &[&str] = &[
    START,
    CHANNEL,
    FORMAT,
    MESSAGE,
    MESSAGE_END,
    TURN_END,
    HANDOFF,
];
const GUIDED_LITERAL_ESCAPE: char = '\u{e000}';

// Distinct suffixes keep adjacent '<' and literal escape characters reversible.
fn shield_guided_literals(text: &str) -> String {
    let mut shielded = String::with_capacity(text.len());
    for ch in text.chars() {
        if ch == GUIDED_LITERAL_ESCAPE {
            shielded.push(GUIDED_LITERAL_ESCAPE);
            shielded.push('e');
        } else if ch == '<' {
            shielded.push(GUIDED_LITERAL_ESCAPE);
            shielded.push('l');
        } else {
            shielded.push(ch);
        }
    }
    shielded
}

fn restore_guided_literals(text: &str) -> String {
    let mut restored = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == GUIDED_LITERAL_ESCAPE {
            match chars.peek() {
                Some('e') => {
                    chars.next();
                    restored.push(GUIDED_LITERAL_ESCAPE);
                }
                Some('l') => {
                    chars.next();
                    restored.push('<');
                }
                _ => restored.push(ch),
            }
        } else {
            restored.push(ch);
        }
    }
    restored
}

static GLOBAL_HARMONY_ENCODING: std::sync::OnceLock<std::result::Result<HarmonyEncoding, String>> =
    std::sync::OnceLock::new();

fn harmony_encoding() -> Result<&'static HarmonyEncoding> {
    GLOBAL_HARMONY_ENCODING
        .get_or_init(|| {
            std::thread::spawn(|| load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss))
                .join()
                .unwrap_or_else(|_| Err(anyhow::anyhow!("Harmony encoding loader thread panicked")))
                .map_err(|error| error.to_string())
        })
        .as_ref()
        .map_err(|error| anyhow::anyhow!("Harmony encoding unavailable: {error}"))
}

pub fn encode_harmony(text: &str) -> anyhow::Result<Vec<u32>> {
    Ok(harmony_encoding()?
        .tokenizer()
        .encode_with_special_tokens(text))
}

pub fn decode_harmony(token_ids: &[u32]) -> anyhow::Result<String> {
    Ok(harmony_encoding()?.tokenizer().decode_utf8(token_ids)?)
}

fn token_id(encoding: &HarmonyEncoding, marker: &str) -> Result<u32> {
    let ids = encoding.tokenizer().encode_with_special_tokens(marker);
    match ids.as_slice() {
        [id] => Ok(*id),
        _ => anyhow::bail!("Harmony marker {marker:?} encoded as {} tokens", ids.len()),
    }
}

fn take_utf8_prefix(pending: &mut Vec<u8>, incoming: &[u8]) -> (String, Option<usize>) {
    pending.extend_from_slice(incoming);
    match std::str::from_utf8(pending) {
        Ok(text) => {
            let text = text.to_string();
            pending.clear();
            (text, None)
        }
        Err(error) if error.error_len().is_none() => {
            let valid_up_to = error.valid_up_to();
            let text = std::str::from_utf8(&pending[..valid_up_to])
                .expect("UTF-8 decoder marked this prefix valid")
                .to_string();
            *pending = pending[valid_up_to..].to_vec();
            (text, None)
        }
        Err(error) => {
            let replaced_bytes = pending.len() - error.valid_up_to();
            let text = String::from_utf8_lossy(pending).into_owned();
            pending.clear();
            (text, Some(replaced_bytes))
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Route {
    Reasoning,
    Text,
    Tool(String),
    Suppress,
}

fn implicit_recipient_prefix(text: &str) -> bool {
    let trimmed = text.trim_start();
    trimmed.is_empty()
        || "to=".starts_with(trimmed)
        || trimmed.strip_prefix("to=").is_some_and(|recipient| {
            recipient
                .trim_end()
                .chars()
                .all(|ch| ch.is_alphanumeric() || matches!(ch, '_' | '.' | '-'))
        })
}

struct QuotedControlTail {
    tokens: Vec<u32>,
    content: String,
    utf8: Vec<u8>,
    boundary: crate::tool_calling::scan::JsonValueBoundary,
}

struct HarmonyRuntime {
    encoding: HarmonyEncoding,
    allow_implicit_channel_header: bool,
    legacy_tool_compatibility: bool,
    parser: StreamableParser,
    marker_ids: HashMap<u32, &'static str>,
    start_id: u32,
    message_id: u32,
    handoff_id: u32,
    inside_message: bool,
    in_body: bool,
    route: Option<Route>,
    call_content: String,
    call_cursor: Option<super::GuidedJsonCursor>,
    active_call_index: usize,
    call_json: crate::tool_calling::scan::JsonValueBoundary,
    quoted_control_tail: Option<QuotedControlTail>,
    next_call_index: usize,
    text_pending: String,
    outside_pending: Vec<u32>,
    finished: bool,
}

impl HarmonyRuntime {
    fn new(
        encoding: HarmonyEncoding,
        allow_implicit_channel_header: bool,
        legacy_tool_compatibility: bool,
    ) -> Result<Self> {
        let mut marker_ids = HashMap::new();
        for marker in CONTROL_MARKERS {
            marker_ids.insert(token_id(&encoding, marker)?, *marker);
        }
        Ok(Self {
            parser: StreamableParser::new(encoding.clone(), None)?,
            allow_implicit_channel_header,
            legacy_tool_compatibility,
            start_id: token_id(&encoding, START)?,
            message_id: token_id(&encoding, MESSAGE)?,
            handoff_id: token_id(&encoding, HANDOFF)?,
            encoding,
            marker_ids,
            inside_message: false,
            in_body: false,
            route: None,
            call_content: String::new(),
            call_cursor: None,
            active_call_index: 0,
            call_json: Default::default(),
            quoted_control_tail: None,
            next_call_index: 0,
            text_pending: String::new(),
            outside_pending: Vec::new(),
            finished: false,
        })
    }

    fn route_current_message(&self) -> Route {
        if self.parser.current_role() != Some(Role::Assistant) {
            return Route::Suppress;
        }
        if let Some(recipient) = self.parser.current_recipient() {
            if let Some(name) = recipient.strip_prefix("functions.")
                && !name.is_empty()
                && name
                    .chars()
                    .all(|ch| ch.is_alphanumeric() || matches!(ch, '_' | '.' | '-'))
            {
                // The legacy tool API only recognizes commentary calls. The
                // native Unified parser accepts recipient routing on any channel.
                return if self.legacy_tool_compatibility
                    && self.parser.current_channel().as_deref() != Some("commentary")
                {
                    Route::Suppress
                } else {
                    Route::Tool(name.to_string())
                };
            }
            // openai-harmony 0.0.8 reports a lone content-type field as a
            // recipient. It must not suppress a response_format final message.
            if !recipient.starts_with(FORMAT) {
                return Route::Suppress;
            }
        }
        match self.parser.current_channel().as_deref() {
            Some("analysis") => Route::Reasoning,
            Some("commentary") if self.legacy_tool_compatibility => Route::Suppress,
            _ => Route::Text,
        }
    }

    fn begin_prefilled_message(
        &mut self,
        starting_state: UnifiedParserStartingState,
    ) -> Result<()> {
        let channel = match starting_state {
            UnifiedParserStartingState::None => return Ok(()),
            UnifiedParserStartingState::Reasoning => "analysis",
            UnifiedParserStartingState::Response => "final",
        };
        let prefix = format!("{START}assistant{CHANNEL}{channel}{MESSAGE}");
        let ids = self
            .encoding
            .tokenizer()
            .encode_with_special_tokens(&prefix);
        for id in ids {
            self.process_token(id, &mut UnifiedParserOutput::default())?;
        }
        Ok(())
    }

    fn feed_text(&mut self, delta: &str, output: &mut UnifiedParserOutput) -> Result<()> {
        if self.finished {
            anyhow::bail!("cannot push to a finished Harmony parser");
        }
        self.text_pending.push_str(delta);
        let held =
            super::marker_prefix_suffix_len(&self.text_pending, CONTROL_MARKERS.iter().copied());
        let safe_len = self.text_pending.len().saturating_sub(held);
        if safe_len == 0 {
            return Ok(());
        }
        let safe = self.text_pending[..safe_len].to_string();
        self.text_pending.drain(..safe_len);
        let ids = self.encoding.tokenizer().encode_with_special_tokens(&safe);
        for id in ids {
            self.process_token(id, output)?;
        }
        Ok(())
    }

    fn feed_token_ids(
        &mut self,
        token_ids: &[u32],
        output: &mut UnifiedParserOutput,
    ) -> Result<()> {
        if self.finished {
            anyhow::bail!("cannot push to a finished Harmony parser");
        }
        for &id in token_ids {
            self.process_token(id, output)?;
        }
        Ok(())
    }

    fn decode_token_ids_for_guided(
        &self,
        token_ids: &[u32],
        pending_token_bytes: &mut Vec<u8>,
    ) -> Result<String> {
        let mut output = String::new();
        let mut ordinary = Vec::new();
        let flush_ordinary = |ordinary: &mut Vec<u32>,
                              output: &mut String,
                              pending: &mut Vec<u8>,
                              encoding: &HarmonyEncoding|
         -> Result<()> {
            if ordinary.is_empty() {
                return Ok(());
            }
            let bytes = encoding.tokenizer().decode_bytes(ordinary.as_slice())?;
            ordinary.clear();
            let (text, invalid_bytes) = take_utf8_prefix(pending, &bytes);
            if let Some(replaced_bytes) = invalid_bytes {
                tracing::warn!(
                    why = "invalid_harmony_token_utf8",
                    replaced_bytes,
                    "replaced an invalid Harmony token fragment"
                );
            }
            output.push_str(&shield_guided_literals(&text));
            Ok(())
        };
        let encoding = &self.encoding;
        for &id in token_ids {
            if let Some(marker) = self.marker_ids.get(&id) {
                flush_ordinary(&mut ordinary, &mut output, pending_token_bytes, encoding)?;
                if !pending_token_bytes.is_empty() {
                    tracing::warn!(
                        why = "incomplete_harmony_token_utf8",
                        skipped_bytes = pending_token_bytes.len(),
                        "discarded an incomplete Harmony token fragment before a control token"
                    );
                    pending_token_bytes.clear();
                }
                output.push_str(marker);
            } else {
                ordinary.push(id);
            }
        }
        flush_ordinary(&mut ordinary, &mut output, pending_token_bytes, encoding)?;
        Ok(output)
    }

    fn flush_outside_pending(&mut self, output: &mut UnifiedParserOutput) {
        if !self.outside_pending.is_empty() {
            match self.encoding.tokenizer().decode_utf8(&self.outside_pending) {
                Ok(text) => output.push_text(text),
                Err(_) => tracing::warn!(
                    why = "incomplete_harmony_utf8",
                    skipped_tokens = self.outside_pending.len(),
                    "discarded an incomplete UTF-8 fragment"
                ),
            }
            self.outside_pending.clear();
        }
    }

    fn process_token(&mut self, id: u32, output: &mut UnifiedParserOutput) -> Result<()> {
        if !self.inside_message {
            let marker = self.marker_ids.get(&id).copied();
            let has_recipient = self.allow_implicit_channel_header
                && matches!(marker, Some(CHANNEL | MESSAGE | FORMAT))
                && self
                    .encoding
                    .tokenizer()
                    .decode_utf8(&self.outside_pending)
                    .is_ok_and(|text| {
                        implicit_recipient_prefix(&text)
                            && text
                                .trim()
                                .strip_prefix("to=")
                                .is_some_and(|recipient| !recipient.is_empty())
                    });
            // Only a complete recipient belongs to an implicit header. Preserve
            // whitespace and incomplete prefixes before a channel boundary.
            if id == self.start_id || (marker == Some(CHANNEL) && !has_recipient) {
                self.flush_outside_pending(output);
            }
            if id == self.start_id {
                self.inside_message = true;
                self.process_parser_token(id, output)?;
            } else if self.allow_implicit_channel_header
                && (marker == Some(CHANNEL)
                    || (has_recipient && matches!(marker, Some(MESSAGE | FORMAT))))
            {
                let pending = std::mem::take(&mut self.outside_pending);
                let prefix = self
                    .encoding
                    .tokenizer()
                    .encode_with_special_tokens(&format!("{START}assistant"));
                for prefix_id in prefix {
                    self.process_token(prefix_id, output)?;
                }
                for pending_id in pending {
                    self.process_token(pending_id, output)?;
                }
                self.process_token(id, output)?;
            } else if self.marker_ids.contains_key(&id) {
                // An orphan control ends the pending prefix: bytes on its two
                // sides must never combine into a synthetic recipient header.
                self.flush_outside_pending(output);
                tracing::warn!(
                    why = "orphan_harmony_control",
                    skipped_tokens = 1,
                    "discarded an orphan Harmony control token"
                );
            } else {
                self.outside_pending.push(id);
                if let Ok(text) = self.encoding.tokenizer().decode_utf8(&self.outside_pending) {
                    if self.allow_implicit_channel_header && implicit_recipient_prefix(&text) {
                        return Ok(());
                    }
                    self.outside_pending.clear();
                    output.push_text(text);
                }
            }
            return Ok(());
        }

        if self.quoted_control_tail.is_some() {
            return self.buffer_quoted_control(id, output);
        }

        let marker_text = self.marker_ids.get(&id).copied();
        if !self.legacy_tool_compatibility
            && self.in_body
            && matches!(self.route, Some(Route::Tool(_)))
            && self.call_json.inside_string()
            && matches!(marker_text, Some(START | MESSAGE_END | TURN_END | HANDOFF))
        {
            // A control token can be quoted data or a boundary after malformed
            // JSON. Keep only that ambiguous tail until valid JSON or EOF settles it.
            self.quoted_control_tail = Some(QuotedControlTail {
                tokens: Vec::new(),
                content: String::new(),
                utf8: Vec::new(),
                boundary: self.call_json.clone(),
            });
            return self.buffer_quoted_control(id, output);
        }

        // The legacy API treated a real handoff token as the call boundary even
        // when malformed JSON left its string unterminated.
        if self.in_body
            && matches!(self.route, Some(Route::Tool(_)))
            && self.call_json.inside_string()
            && !(self.legacy_tool_compatibility && id == self.handoff_id)
            && let Some(marker) = marker_text
        {
            for ordinary_id in self.encoding.tokenizer().encode_ordinary(marker) {
                self.process_parser_token(ordinary_id, output)?;
            }
            return Ok(());
        }

        if id == self.start_id {
            tracing::warn!(
                why = "harmony_resync_at_start",
                recovered_tokens = 1,
                "restarted Harmony parsing at a new message boundary"
            );
            self.parser = StreamableParser::new(self.encoding.clone(), None)?;
            self.inside_message = true;
            self.in_body = false;
            self.route = None;
            self.call_content.clear();
            self.call_json = Default::default();
            self.process_parser_token(id, output)?;
            return Ok(());
        }

        self.process_parser_token(id, output)
    }

    fn process_parser_token(&mut self, id: u32, output: &mut UnifiedParserOutput) -> Result<()> {
        let prior_messages = self.parser.messages().len();
        if let Err(error) = self.parser.process(id) {
            if self.legacy_tool_compatibility
                && id == self.handoff_id
                && let Some(Route::Tool(name)) = self.route.clone()
            {
                self.finish_tool_call(&name, output);
            }
            tracing::warn!(why = "malformed_harmony_message", skipped_tokens = 1, error = %error, "discarded malformed Harmony input and resumed at the next message");
            self.parser = StreamableParser::new(self.encoding.clone(), None)?;
            self.inside_message = false;
            self.in_body = false;
            self.route = None;
            self.call_content.clear();
            self.call_json = Default::default();
            return Ok(());
        }

        if id == self.message_id && self.inside_message && !self.in_body {
            self.in_body = true;
            self.route = Some(self.route_current_message());
            self.call_cursor = None;
            if !self.legacy_tool_compatibility
                && let Some(Route::Tool(name)) = &self.route
            {
                self.active_call_index = self.next_call_index;
                self.next_call_index += 1;
                self.call_cursor = Some(super::GuidedJsonCursor::named(name));
            }
            self.call_content.clear();
            self.call_json = Default::default();
        }

        if self.in_body
            && let Some(delta) = self.parser.last_content_delta()?
        {
            match self.route.as_ref().unwrap_or(&Route::Suppress) {
                Route::Reasoning => output.push_reasoning(delta),
                Route::Text => output.push_text(delta),
                Route::Tool(_) => {
                    self.advance_call_json(&delta);
                    self.call_content.push_str(&delta);
                    if let Some(cursor) = &mut self.call_cursor {
                        let mut calls = Vec::new();
                        cursor.advance(&self.call_content, &mut calls);
                        for mut call in calls {
                            call.tool_index = self.active_call_index;
                            output.push_call(call);
                        }
                    }
                }
                Route::Suppress => {}
            }
        }

        let messages = self.parser.messages().len();
        if messages > prior_messages {
            if let Some(Route::Tool(name)) = self.route.clone() {
                if id == self.handoff_id
                    || (!self.legacy_tool_compatibility
                        && matches!(
                            self.marker_ids.get(&id).copied(),
                            Some(MESSAGE_END | TURN_END)
                        ))
                {
                    self.finish_tool_call(&name, output);
                } else {
                    tracing::warn!(
                        why = "tool_message_without_handoff",
                        skipped_bytes = self.call_content.len(),
                        "discarded an unterminated Harmony tool call"
                    );
                }
            }
            self.inside_message = false;
            self.in_body = false;
            self.route = None;
            self.call_content.clear();
            self.call_json = Default::default();
        }
        Ok(())
    }

    fn advance_call_json(&mut self, text: &str) {
        for character in text.chars() {
            self.call_json.advance(character);
        }
    }

    fn buffer_quoted_control(&mut self, id: u32, output: &mut UnifiedParserOutput) -> Result<()> {
        let tail = self
            .quoted_control_tail
            .as_mut()
            .expect("quoted tail exists");
        tail.tokens.push(id);
        let bytes = self.encoding.tokenizer().decode_bytes([id])?;
        let (text, invalid) = take_utf8_prefix(&mut tail.utf8, &bytes);
        if let Some(replaced_bytes) = invalid {
            tracing::warn!(
                why = "invalid_harmony_token_utf8",
                replaced_bytes,
                "replaced an invalid Harmony token fragment"
            );
        }
        let mut complete = false;
        for character in text.chars() {
            complete |= tail.boundary.advance(character);
        }
        tail.content.push_str(&text);
        if complete {
            let candidate = format!("{}{}", self.call_content, tail.content);
            if serde_json::from_str::<Value>(&candidate).is_ok() {
                let tail = self.quoted_control_tail.take().expect("quoted tail exists");
                for token in tail.tokens {
                    if let Some(marker) = self.marker_ids.get(&token) {
                        for ordinary in self.encoding.tokenizer().encode_ordinary(marker) {
                            self.process_parser_token(ordinary, output)?;
                        }
                    } else {
                        self.process_parser_token(token, output)?;
                    }
                }
            }
        }
        Ok(())
    }

    fn recover_quoted_control(&mut self, output: &mut UnifiedParserOutput) -> Result<()> {
        let Some(tail) = self.quoted_control_tail.take() else {
            return Ok(());
        };
        let mut candidate = None;
        let mut start = None;
        for (at, &token) in tail.tokens.iter().enumerate() {
            if token == self.start_id {
                candidate = Some(at);
            } else if token == self.message_id
                && let Some(header_start) = candidate.take()
                && let Ok(header) = self
                    .encoding
                    .tokenizer()
                    .decode_utf8(&tail.tokens[header_start + 1..at])
                && header
                    .split(|ch: char| ch.is_whitespace() || ch == '<')
                    .next()
                    == Some("assistant")
            {
                start = Some(header_start);
                break;
            }
        }
        tracing::warn!(
            why = "malformed_harmony_quoted_control",
            skipped_bytes = self.call_content.len(),
            recovered_tokens = start.map_or(0, |at| tail.tokens.len() - at),
            "settled an ambiguous Harmony boundary after malformed JSON"
        );
        self.parser = StreamableParser::new(self.encoding.clone(), None)?;
        self.inside_message = false;
        self.in_body = false;
        self.route = None;
        self.call_content.clear();
        self.call_cursor = None;
        self.call_json = Default::default();
        if let Some(start) = start {
            for id in tail.tokens[start..].iter().copied() {
                self.process_token(id, output)?;
            }
        }
        Ok(())
    }

    fn finish_tool_call(&mut self, name: &str, output: &mut UnifiedParserOutput) {
        let arguments = if self.call_content.trim().is_empty() {
            "{}".to_string()
        } else {
            match serde_json::from_str::<Value>(&self.call_content) {
                Ok(value) if self.legacy_tool_compatibility => {
                    serde_json::to_string(&value).unwrap_or_else(|_| self.call_content.clone())
                }
                Ok(_) => self.call_content.clone(),
                Err(_) if self.legacy_tool_compatibility => self.call_content.trim().to_string(),
                Err(error) => {
                    tracing::warn!(why = "invalid_harmony_tool_arguments", skipped_bytes = self.call_content.len(), error = %error, "discarded malformed Harmony tool arguments");
                    return;
                }
            }
        };
        if let Some(cursor) = &self.call_cursor {
            let committed = cursor.committed().first();
            output.push_call(ToolCallDelta {
                tool_index: self.active_call_index,
                name: committed.is_none().then(|| name.to_string()),
                arguments: committed
                    .map_or(arguments.as_str(), |_call| {
                        &self.call_content[cursor.released_end()..]
                    })
                    .to_string(),
                complete: true,
            });
        } else {
            output.push_call(ToolCallDelta {
                tool_index: self.next_call_index,
                name: Some(name.to_string()),
                arguments,
                complete: true,
            });
            self.next_call_index += 1;
        }
    }

    fn finish(&mut self, output: &mut UnifiedParserOutput) -> Result<()> {
        if self.finished {
            anyhow::bail!("cannot finish a Harmony parser twice");
        }
        if !self.text_pending.is_empty() {
            let pending = std::mem::take(&mut self.text_pending);
            let ids = self
                .encoding
                .tokenizer()
                .encode_with_special_tokens(&pending);
            for id in ids {
                self.process_token(id, output)?;
            }
        }
        while self.quoted_control_tail.is_some() {
            self.recover_quoted_control(output)?;
        }
        self.flush_outside_pending(output);
        if self.in_body && matches!(self.route, Some(Route::Tool(_))) {
            tracing::warn!(
                why = "truncated_harmony_tool_call",
                skipped_bytes = self.call_content.len(),
                "discarded a truncated Harmony tool call"
            );
        }
        if self.inside_message
            && !self.in_body
            && let Err(error) = self.parser.process_eos()
        {
            self.recover_final_header(output)?;
            tracing::warn!(why = "unterminated_harmony_header", error = %error,
                "finalized Harmony input with an incomplete header");
        }
        self.finished = true;
        Ok(())
    }

    fn recover_final_header(&self, output: &mut UnifiedParserOutput) -> Result<()> {
        #[derive(serde::Deserialize)]
        #[serde(tag = "state")]
        enum State {
            Header {
                header_tokens: Vec<u32>,
            },
            #[serde(other)]
            Other,
        }
        let State::Header { header_tokens } = serde_json::from_str(&self.parser.state_json()?)?
        else {
            return Ok(());
        };
        let mut header_bytes = self.encoding.tokenizer().decode_bytes(&header_tokens)?;
        let (header, _) = take_utf8_prefix(&mut header_bytes, &[]);
        // Only an undirected assistant final is safe to surface as visible text.
        let Some(mut text) = header.strip_prefix("assistant<|channel|>final") else {
            return Ok(());
        };
        if text.starts_with(|ch: char| ch.is_alphanumeric() || ch == '_')
            || text.trim_start().starts_with("to=")
        {
            return Ok(());
        }
        while let Some(prefix) = [TURN_END, MESSAGE_END, HANDOFF]
            .iter()
            .find_map(|marker| text.strip_suffix(marker))
        {
            text = prefix;
        }
        // Embedded controls make an unfinished header unsafe to recover as visible text.
        if CONTROL_MARKERS.iter().any(|marker| text.contains(marker)) {
            return Ok(());
        }
        output.push_text(text);
        Ok(())
    }

    fn take_pending_text(&mut self) -> String {
        let mut recovered = std::mem::take(&mut self.text_pending);
        if !self.outside_pending.is_empty()
            && let Ok(text) = self.encoding.tokenizer().decode_utf8(&self.outside_pending)
        {
            recovered.push_str(&text);
        }
        recovered
    }
}

struct HarmonyNative {
    encoding: Option<HarmonyEncoding>,
    init_error: Option<String>,
    runtime: Option<HarmonyRuntime>,
    init: UnifiedParserInit,
    pending_token_bytes: Vec<u8>,
    allow_implicit_channel_header: bool,
    legacy_tool_compatibility: bool,
}

impl HarmonyNative {
    fn new(allow_implicit_channel_header: bool, legacy_tool_compatibility: bool) -> Self {
        let (encoding, init_error) = match harmony_encoding() {
            Ok(encoding) => (Some(encoding.clone()), None),
            Err(error) => (None, Some(error.to_string())),
        };
        Self {
            encoding,
            init_error,
            runtime: None,
            init: UnifiedParserInit::default(),
            pending_token_bytes: Vec::new(),
            allow_implicit_channel_header,
            legacy_tool_compatibility,
        }
    }

    fn make_runtime(&self, init: &UnifiedParserInit) -> Result<HarmonyRuntime> {
        let encoding = self.encoding.as_ref().ok_or_else(|| {
            anyhow::anyhow!(
                "Harmony parser initialization failed: {}",
                self.init_error.as_deref().unwrap_or("encoding unavailable")
            )
        })?;
        let mut runtime = HarmonyRuntime::new(
            encoding.clone(),
            self.allow_implicit_channel_header,
            self.legacy_tool_compatibility,
        )?;
        if init.prompt_token_ids.is_empty() {
            runtime.begin_prefilled_message(init.starting_state)?;
        } else {
            let mut ignored = UnifiedParserOutput::default();
            for id in &init.prompt_token_ids {
                runtime.process_token(*id, &mut ignored)?;
            }
            runtime.next_call_index = 0;
        }
        Ok(runtime)
    }

    fn guided_prompt_prefill(&self, init: &UnifiedParserInit) -> Result<Option<String>> {
        if init.prompt_token_ids.is_empty() {
            return Ok(None);
        }
        let encoding = self.encoding.as_ref().ok_or_else(|| {
            anyhow::anyhow!(
                "Harmony token decoding unavailable: {}",
                self.init_error.as_deref().unwrap_or("encoding unavailable")
            )
        })?;
        let mut marker_ids = HashMap::new();
        for marker in CONTROL_MARKERS {
            marker_ids.insert(token_id(encoding, marker)?, *marker);
        }

        let mut tail = String::new();
        let mut tail_open = false;
        let mut header_open = false;
        let mut pending_bytes = Vec::new();
        for id in &init.prompt_token_ids {
            if let Some(marker) = marker_ids.get(id) {
                pending_bytes.clear();
                match *marker {
                    START => {
                        tail.clear();
                        tail.push_str(START);
                        tail_open = true;
                        header_open = true;
                    }
                    MESSAGE_END | TURN_END | HANDOFF => {
                        tail.clear();
                        tail_open = false;
                        header_open = false;
                    }
                    MESSAGE if tail_open && header_open => {
                        tail.push_str(MESSAGE);
                        header_open = false;
                    }
                    _ if tail_open && header_open => tail.push_str(marker),
                    _ => {}
                }
                continue;
            }

            let bytes = encoding
                .tokenizer()
                .decode_bytes(std::slice::from_ref(id))?;
            let (text, _) = take_utf8_prefix(&mut pending_bytes, &bytes);
            if tail_open && header_open {
                tail.push_str(&shield_guided_literals(&text));
            }
        }

        let is_assistant = tail
            .strip_prefix(START)
            .is_some_and(|header| header.trim_start().starts_with("assistant"));
        Ok((tail_open && is_assistant).then_some(tail))
    }

    fn guided_prefilled_header(&self, prefill: &str) -> Option<super::GuidedPrefilledHeader> {
        let Some((start, body_start, _)) = find_header(prefill) else {
            return Some(super::GuidedPrefilledHeader::Pending);
        };
        Some(super::GuidedPrefilledHeader::Complete {
            tool_name: tool_recipient(prefill, start, body_start),
            body_start,
        })
    }

    fn clear_pending_token_bytes_at_eof(&mut self) {
        if !self.pending_token_bytes.is_empty() {
            tracing::warn!(
                why = "incomplete_harmony_token_utf8_at_eof",
                skipped_bytes = self.pending_token_bytes.len(),
                "discarded an incomplete Harmony token fragment at end of stream"
            );
            self.pending_token_bytes.clear();
        }
    }
}

impl NativeUnified for HarmonyNative {
    fn preserve_special_tokens(&self) -> bool {
        true
    }

    fn input_capabilities(&self) -> UnifiedParserInputCapabilities {
        UnifiedParserInputCapabilities {
            text: true,
            token_ids: true,
        }
    }

    fn decode_token_ids(&mut self, token_ids: &[u32]) -> Result<String> {
        let encoding = self.encoding.as_ref().ok_or_else(|| {
            anyhow::anyhow!(
                "Harmony token decoding unavailable: {}",
                self.init_error.as_deref().unwrap_or("encoding unavailable")
            )
        })?;
        let incoming = encoding.tokenizer().decode_bytes(token_ids)?;
        let (text, invalid_bytes) = take_utf8_prefix(&mut self.pending_token_bytes, &incoming);
        if let Some(replaced_bytes) = invalid_bytes {
            tracing::warn!(
                why = "invalid_harmony_token_utf8",
                replaced_bytes,
                "replaced an invalid Harmony token fragment"
            );
        }
        Ok(text)
    }

    fn guided_reasoning(&self) -> Option<GuidedReasoning> {
        Some(GuidedReasoning::Channel(GuidedChannel {
            find_open: guided_reasoning_open,
            find_close: |haystack, _, state| guided_reasoning_close(haystack, state),
            find_turn_end: |haystack, _, _| guided_turn_end(haystack),
            find_stray: guided_stray_header,
            find_routing: guided_routing_header,
            find_transition: guided_content_header,
            holdback: guided_header_holdback,
            strip_text: guided_strip_text,
            reasoning_filter: None,
            competitors: CONTROL_MARKERS,
            close_markers: &[MESSAGE_END, TURN_END],
            response_literal_markers: CONTROL_MARKERS,
            header_pending: Some(guided_header_pending),
            envelope_owned: false,
        }))
    }

    fn guided_grammar(&self) -> GuidedGrammar {
        GuidedGrammar {
            control_markers: CONTROL_MARKERS
                .iter()
                .map(|marker| marker.to_string())
                .collect(),
            invoke_start: format!("{START}assistant{CHANNEL}commentary to=functions."),
            invoke_start_alternates: vec![format!("{START}assistant to=functions.")],
            invoke_end: HANDOFF.to_string(),
            native_body_start: Some((MESSAGE.to_string(), true)),
            invoke_boundary_factory: None,
            guided_prefix_policy: Some(guided_invoke_prefix),
            guided_prefix_factory: None,
        }
    }

    fn apply_native_init(&mut self, init: &UnifiedParserInit) -> Result<()> {
        let runtime = self.make_runtime(init)?;
        self.runtime = Some(runtime);
        self.init = init.clone();
        self.pending_token_bytes.clear();
        Ok(())
    }

    fn restore_native_state(&mut self, _starting_state: UnifiedParserStartingState) {}

    fn push_native(&mut self, delta: &str, output: &mut UnifiedParserOutput) -> Result<()> {
        self.runtime
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("Harmony parser is not initialized"))?
            .feed_text(delta, output)
    }

    fn push_token_ids_native(
        &mut self,
        token_ids: &[u32],
        output: &mut UnifiedParserOutput,
    ) -> Result<()> {
        self.runtime
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("Harmony parser is not initialized"))?
            .feed_token_ids(token_ids, output)
    }

    fn decode_token_ids_for_guided(&mut self, token_ids: &[u32]) -> Result<String> {
        self.runtime
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("Harmony parser is not initialized"))?
            .decode_token_ids_for_guided(token_ids, &mut self.pending_token_bytes)
    }

    fn guided_prompt_prefill(&self, init: &UnifiedParserInit) -> Result<Option<String>> {
        HarmonyNative::guided_prompt_prefill(self, init)
    }

    fn guided_prefilled_header(&self, prefill: &str) -> Option<super::GuidedPrefilledHeader> {
        HarmonyNative::guided_prefilled_header(self, prefill)
    }

    fn finish_guided_input(&mut self) -> Result<()> {
        self.clear_pending_token_bytes_at_eof();
        Ok(())
    }

    fn restore_guided_token_literals(&self, output: &mut UnifiedParserOutput) {
        for event in &mut output.events {
            match event {
                crate::unified::UnifiedParserEvent::Text(text)
                | crate::unified::UnifiedParserEvent::Reasoning(text) => {
                    *text = restore_guided_literals(text);
                }
                crate::unified::UnifiedParserEvent::ToolCall(call) => {
                    call.arguments = restore_guided_literals(&call.arguments);
                }
                crate::unified::UnifiedParserEvent::MessageStart(_)
                | crate::unified::UnifiedParserEvent::MessageEnd => {}
            }
        }
    }

    fn finish_native(&mut self, output: &mut UnifiedParserOutput) -> Result<()> {
        self.clear_pending_token_bytes_at_eof();
        self.runtime
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("Harmony parser is not initialized"))?
            .finish(output)
    }

    fn reset_native(&mut self) -> String {
        self.pending_token_bytes.clear();
        match self.make_runtime(&self.init) {
            Ok(runtime) => {
                let recovered = self
                    .runtime
                    .as_mut()
                    .map(HarmonyRuntime::take_pending_text)
                    .unwrap_or_default();
                self.runtime = Some(runtime);
                recovered
            }
            Err(error) => {
                tracing::warn!(why = "harmony_reset_failed", error = %error, "could not rebuild Harmony parser during reset");
                self.runtime = None;
                String::new()
            }
        }
    }
}

pub(crate) fn gpt_oss_unified(_tools: &[Tool]) -> Box<dyn UnifiedParser> {
    // Published GPT-OSS fixtures use channel-first headers without an explicit role.
    Box::new(GuidedRouted::new(HarmonyNative::new(true, false)))
}

// Native-only lifecycle keeps unrelated guided state out of the legacy auto-trait contract.
struct LegacyHarmonyUnified {
    native: HarmonyNative,
    started: bool,
    finished: bool,
}

impl UnifiedParser for LegacyHarmonyUnified {
    fn input_capabilities(&self) -> UnifiedParserInputCapabilities {
        self.native.input_capabilities()
    }

    fn preserve_special_tokens(&self) -> bool {
        self.native.preserve_special_tokens()
    }

    fn initialize_request(&mut self, init: UnifiedParserInit) -> Result<()> {
        if self.started || self.finished {
            anyhow::bail!("cannot initialize a unified parser after parsing has started");
        }
        if init.tool_output_mode != UnifiedToolOutputMode::Native {
            anyhow::bail!("legacy Harmony tool parsing does not support guided tool output");
        }
        self.native.apply_native_init(&init)
    }

    fn parse_into(&mut self, delta: &str, output: &mut UnifiedParserOutput) -> Result<()> {
        if self.finished {
            anyhow::bail!("cannot push to a finished unified parser");
        }
        self.started = true;
        self.native.push_native(delta, output)
    }

    fn parse_token_ids_into(
        &mut self,
        token_ids: &[u32],
        output: &mut UnifiedParserOutput,
    ) -> Result<()> {
        if self.finished {
            anyhow::bail!("cannot push to a finished unified parser");
        }
        self.started = true;
        self.native.push_token_ids_native(token_ids, output)
    }

    fn finish(&mut self) -> Result<UnifiedParserOutput> {
        if self.finished {
            anyhow::bail!("cannot finish a unified parser twice");
        }
        self.started = true;
        self.finished = true;
        let mut output = UnifiedParserOutput::default();
        self.native.finish_native(&mut output)?;
        Ok(output)
    }

    fn reset(&mut self) -> String {
        let recovered = self.native.reset_native();
        self.started = false;
        self.finished = false;
        recovered
    }
}

pub(crate) fn gpt_oss_legacy_tool_adapter()
-> Box<dyn UnifiedParser + Sync + UnwindSafe + RefUnwindSafe> {
    Box::new(LegacyHarmonyUnified {
        native: HarmonyNative::new(true, true),
        started: false,
        finished: false,
    })
}

fn find_header(haystack: &str) -> Option<(usize, usize, Option<String>)> {
    let start = haystack.find(START);
    let channel = haystack.find(CHANNEL);
    let mut header_start = start.into_iter().chain(channel).min()?;
    // A stripped role marker may leave the role word before a channel header;
    // unrelated visible prose before a bare channel must remain visible.
    if channel == Some(header_start)
        && haystack[..header_start]
            .trim_start()
            .starts_with("assistant")
    {
        header_start = 0;
    }
    let message_at = haystack[header_start..].find(MESSAGE)? + header_start;
    let header = &haystack[header_start..message_at];
    let channel = header.find(CHANNEL).map(|at| {
        let rest = &header[at + CHANNEL.len()..];
        rest.split(|ch: char| ch.is_whitespace() || ch == '<')
            .next()
            .unwrap_or_default()
            .to_string()
    });
    Some((header_start, message_at + MESSAGE.len(), channel))
}

fn tool_recipient(haystack: &str, start: usize, end: usize) -> Option<String> {
    haystack[start..end]
        .split(|ch: char| ch.is_whitespace() || ch == '<')
        .filter_map(|part| part.strip_prefix("to="))
        .find_map(|name| {
            let function = name.strip_prefix("functions.")?;
            (!function.is_empty()
                && function
                    .chars()
                    .all(|ch| ch.is_alphanumeric() || matches!(ch, '_' | '.' | '-')))
            .then(|| function.to_string())
        })
}

fn is_tool_header(haystack: &str, start: usize, end: usize) -> bool {
    tool_recipient(haystack, start, end).is_some()
}

fn guided_header(
    haystack: &str,
    state: super::GuidedChannelState,
) -> Option<(usize, usize, Option<String>)> {
    let offset = if state.scope.allows_bare_header() {
        0
    } else if state.scope == super::GuidedTurnScope::ResponsePrefill {
        return None;
    } else {
        // Closing reasoning spends the bare-header scope, not later framed messages.
        haystack.find(START)?
    };
    let (start, end, channel) = find_header(&haystack[offset..])?;
    if matches!(
        partial_invoke_prefix(&haystack[offset + start..]),
        super::GuidedPrefix::Match | super::GuidedPrefix::Strip(_)
    ) {
        return None;
    }
    Some((start + offset, end + offset, channel))
}

fn guided_reasoning_open(
    haystack: &str,
    _flush: bool,
    state: super::GuidedChannelState,
) -> Option<(usize, usize)> {
    guided_header(haystack, state).and_then(|(start, end, channel)| {
        (channel.as_deref() == Some("analysis")).then_some((start, end - start))
    })
}

fn guided_content_header(
    haystack: &str,
    _flush: bool,
    state: super::GuidedChannelState,
) -> Option<(usize, usize)> {
    let (start, end, channel) = guided_header(haystack, state)?;
    // Guided tool JSON can be the final message body, rather than visible prose.
    (!matches!(channel.as_deref(), Some("analysis" | "final"))
        && !is_tool_header(haystack, start, end))
    .then_some((start, end - start))
}

fn guided_routing_header(
    haystack: &str,
    flush: bool,
    state: super::GuidedChannelState,
) -> Option<(usize, usize)> {
    guided_reasoning_open(haystack, flush, state)
        .or_else(|| guided_content_header(haystack, flush, state))
}

fn guided_stray_header(
    haystack: &str,
    flush: bool,
    state: super::GuidedChannelState,
) -> Option<(usize, usize)> {
    if state.scope == super::GuidedTurnScope::ResponsePrefill {
        return None;
    }
    let Some(header) = guided_header(haystack, state) else {
        return flush
            .then(|| haystack.find(START).map(|at| (at, START.len())))
            .flatten();
    };
    if header.2.as_deref() == Some("analysis") || is_tool_header(haystack, header.0, header.1) {
        return None;
    }
    Some((header.0, header.1 - header.0))
}

fn guided_reasoning_close(
    haystack: &str,
    state: super::GuidedChannelState,
) -> Option<(usize, usize)> {
    // A final header ends reasoning without deciding whether its body is tool JSON.
    let final_header = (state.scope == super::GuidedTurnScope::InReasoning)
        .then(|| guided_header(haystack, state))
        .flatten()
        .filter(|(start, end, channel)| {
            channel.as_deref() == Some("final") && !is_tool_header(haystack, *start, *end)
        })
        .map(|(start, end, _)| (start, end - start));
    [MESSAGE_END, TURN_END]
        .iter()
        .filter_map(|marker| haystack.find(marker).map(|at| (at, marker.len())))
        .chain(final_header)
        .min_by_key(|(at, _)| *at)
}

fn guided_turn_end(haystack: &str) -> Option<(usize, usize)> {
    haystack.find(TURN_END).map(|at| (at, TURN_END.len()))
}

// A JSON opener or a new channel settles an unterminated tool header. Never
// borrow a later message delimiter to turn the following reasoning into its body.
fn partial_invoke_prefix(text: &str) -> super::GuidedPrefix {
    let starts = [
        "<|start|>assistant<|channel|>commentary to=functions.",
        "<|start|>assistant to=functions.",
    ];
    let Some(prefix) = starts.iter().find(|prefix| text.starts_with(**prefix)) else {
        return super::GuidedPrefix::NoMatch;
    };
    let rest = &text[prefix.len()..];
    let mut channel_seen = prefix.contains(CHANNEL);
    for (at, ch) in rest.char_indices() {
        let tail = &rest[at..];
        if tail.starts_with(MESSAGE) {
            return super::GuidedPrefix::NoMatch;
        }
        if matches!(ch, '{' | '[') {
            return super::GuidedPrefix::Match;
        }
        if [START, MESSAGE_END, TURN_END, HANDOFF]
            .iter()
            .any(|marker| tail.starts_with(marker))
        {
            return super::GuidedPrefix::Strip(prefix.len());
        }
        if tail.starts_with(CHANNEL) {
            if channel_seen {
                return super::GuidedPrefix::Strip(prefix.len());
            }
            channel_seen = true;
        }
    }
    super::GuidedPrefix::Pending
}

fn guided_invoke_prefix(context: super::GuidedPrefixContext<'_>) -> super::GuidedPrefix {
    partial_invoke_prefix(&context.text[context.at..])
}

fn resolved_partial_header(haystack: &str) -> bool {
    haystack.rfind(START).is_some_and(|at| {
        matches!(
            partial_invoke_prefix(&haystack[at..]),
            super::GuidedPrefix::Match | super::GuidedPrefix::Strip(_)
        )
    })
}

fn guided_header_holdback(haystack: &str, _state: super::GuidedChannelState) -> usize {
    if !resolved_partial_header(haystack)
        && let Some(at) = [START, CHANNEL]
            .iter()
            .filter_map(|marker| haystack.rfind(marker))
            .min()
        && !haystack[at..].contains(MESSAGE)
    {
        return haystack.len() - at;
    }
    super::marker_prefix_suffix_len(haystack, CONTROL_MARKERS.iter().copied())
}

fn guided_header_pending(haystack: &str, flush: bool) -> bool {
    !flush
        && !resolved_partial_header(haystack)
        && [START, CHANNEL].iter().any(|marker| {
            haystack
                .rfind(marker)
                .is_some_and(|at| !haystack[at..].contains(MESSAGE))
        })
}

fn guided_strip_text(text: &str) -> String {
    text.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::unified::{UnifiedEvent, UnifiedParserEvent, UnifiedParserExt};

    #[test]
    fn legacy_adapter_matches_routed_native_lifecycle() {
        let input = concat!(
            "<|start|>assistant to=functions.lookup<|channel|>commentary<|message|>",
            r#"{"q":"x"}<|call|>"#
        );
        let ids = encode_harmony(input).unwrap();
        for tokens in [false, true] {
            let mut legacy = gpt_oss_legacy_tool_adapter();
            let mut routed = GuidedRouted::new(HarmonyNative::new(true, true));
            for _ in 0..2 {
                legacy.initialize(&[]).unwrap();
                routed.initialize(&[]).unwrap();
                let (actual, expected) = if tokens {
                    (
                        legacy.push_tokens(&ids).unwrap(),
                        routed.push_tokens(&ids).unwrap(),
                    )
                } else {
                    (legacy.push(input).unwrap(), routed.push(input).unwrap())
                };
                assert_eq!(actual, expected);
                assert_eq!(
                    legacy.initialize(&[]).unwrap_err().to_string(),
                    routed.initialize(&[]).unwrap_err().to_string()
                );
                assert_eq!(legacy.finish().unwrap(), routed.finish().unwrap());
                assert_eq!(
                    legacy.finish().unwrap_err().to_string(),
                    routed.finish().unwrap_err().to_string()
                );
                assert_eq!(
                    legacy.push("").unwrap_err().to_string(),
                    routed.push("").unwrap_err().to_string()
                );
                assert_eq!(legacy.reset(), routed.reset());
            }
        }
    }
    fn parse_text(
        input: &str,
        starting_state: UnifiedParserStartingState,
    ) -> Vec<UnifiedParserEvent> {
        let mut parser = gpt_oss_unified(&[]);
        parser
            .initialize_request(UnifiedParserInit {
                starting_state,
                ..UnifiedParserInit::default()
            })
            .unwrap();
        let mut events = parser.push(input).unwrap();
        let finish = parser.finish();
        assert!(
            finish.is_ok(),
            "finish failed after events {events:?}: {finish:?}"
        );
        events.extend(finish.unwrap().events);
        events
    }

    fn parse_guided(
        chunks: &[&str],
        starting_state: UnifiedParserStartingState,
    ) -> Vec<UnifiedEvent> {
        let mut parser = gpt_oss_unified(&[]);
        parser
            .initialize_request(UnifiedParserInit {
                starting_state,
                tool_output_mode: crate::unified::UnifiedToolOutputMode::GuidedJson {
                    named_tool: None,
                },
                ..UnifiedParserInit::default()
            })
            .unwrap();
        let mut events = Vec::new();
        for chunk in chunks {
            let pushed = parser.push(chunk);
            assert!(
                pushed.is_ok(),
                "push failed after events {events:?}: {pushed:?}"
            );
            events.extend(pushed.unwrap());
        }
        let finish = parser.finish();
        assert!(
            finish.is_ok(),
            "finish failed after events {events:?}: {finish:?}"
        );
        events.extend(finish.unwrap().events);
        crate::unified::assemble(&events)
    }

    fn every_char_chunk(input: &str) -> Vec<&str> {
        input
            .char_indices()
            .map(|(start, _)| start)
            .chain(std::iter::once(input.len()))
            .collect::<Vec<_>>()
            .windows(2)
            .map(|pair| &input[pair[0]..pair[1]])
            .collect()
    }

    #[test]
    fn routes_ordered_reasoning_tool_and_content() {
        let events = parse_text(
            concat!(
                "<|start|>assistant<|channel|>analysis<|message|>think<|end|>",
                "<|start|>assistant<|channel|>commentary to=functions.lookup<|constrain|>json<|message|>{\"q\":\"x\"}<|call|>",
                "<|start|>assistant<|channel|>analysis<|message|>after<|end|>",
                "<|start|>assistant<|channel|>final<|message|>done<|return|>"
            ),
            UnifiedParserStartingState::None,
        );
        assert_eq!(
            crate::unified::assemble(&events),
            vec![
                crate::unified::UnifiedEvent::Reasoning {
                    text: "think".into()
                },
                crate::unified::UnifiedEvent::ToolCall {
                    name: "lookup".into(),
                    arguments: serde_json::json!({"q":"x"})
                },
                crate::unified::UnifiedEvent::Reasoning {
                    text: "after".into()
                },
                crate::unified::UnifiedEvent::Text {
                    text: "done".into()
                },
            ]
        );
    }

    #[test]
    fn implicit_assistant_channel_header_routes_reasoning() {
        let events = parse_text(
            "<|channel|>analysis<|message|>legacy thought<|end|>",
            UnifiedParserStartingState::None,
        );
        assert!(
            matches!(events.as_slice(), [UnifiedParserEvent::Reasoning(text)] if text == "legacy thought")
        );
    }

    #[test]
    fn malformed_final_header_recovers_only_unemitted_final_text() {
        let reasoning = "<|channel|>analysis<|message|>thought<|end|>";
        for (tail, text) in [
            ("<|start|>assistant<|channel|>final*)__Hello!", "*)__Hello!"),
            (
                "<|start|>assistant<|channel|>final*)__hé🙂<|return|>",
                "*)__hé🙂",
            ),
            (
                "<|start|>assistant<|channel|>finalx<|message|>Hello!",
                "Hello!",
            ),
            ("<|start|>assistant<|channel|>analysis*)__private", ""),
            (
                "<|start|>assistant to=functions.lookup<|channel|>final*)__private",
                "",
            ),
            ("<|start|>assistant<|channel|>final to=functions.lookup", ""),
            ("<|start|>assistant<|channel|>final<|constrain|>json", ""),
            ("<|start|>assistant<|channel|>final <|constrain|>json", ""),
            ("<|start|>assistant<|channel|>final\t<|constrain|>json", ""),
            ("<|start|>assistant<|channel|>final\n<|constrain|>json", ""),
            ("<|start|>assistant<|channel|>final  <|constrain|>json", ""),
            (
                "<|start|>assistant<|channel|>final<|channel|>analysis private",
                "",
            ),
            ("<|start|>assistant<|channel|>final safe<|end|>private", ""),
            (
                "<|start|>assistant<|channel|>final safe<|return|>private",
                "",
            ),
            ("<|start|>assistant<|channel|>final safe<|call|>private", ""),
        ] {
            let input = format!("{reasoning}{tail}");
            let mut expected = vec![UnifiedEvent::Reasoning {
                text: "thought".into(),
            }];
            if !text.is_empty() {
                expected.push(UnifiedEvent::Text { text: text.into() });
            }
            for seed in [false, true] {
                let mut parser = gpt_oss_unified(&[]);
                let init = UnifiedParserInit {
                    prompt_token_ids: if seed {
                        encode_harmony("<|start|>assistant").unwrap()
                    } else {
                        vec![]
                    },
                    ..Default::default()
                };
                for split in input.char_indices().map(|(at, _)| at).chain([input.len()]) {
                    parser.reset();
                    parser.initialize_request(init.clone()).unwrap();
                    let mut events = parser.push(&input[..split]).unwrap();
                    events.extend(parser.push(&input[split..]).unwrap());
                    events.extend(parser.finish().unwrap().events);
                    assert_eq!(
                        crate::unified::assemble(&events),
                        expected,
                        "text split={split} seed={seed} tail={tail}"
                    );
                }
                let ids = encode_harmony(&input).unwrap();
                for split in 0..=ids.len() {
                    parser.reset();
                    parser.initialize_request(init.clone()).unwrap();
                    let mut events = parser.push_tokens(&ids[..split]).unwrap();
                    events.extend(parser.push_tokens(&ids[split..]).unwrap());
                    events.extend(parser.finish().unwrap().events);
                    assert_eq!(
                        crate::unified::assemble(&events),
                        expected,
                        "token split={split} seed={seed} tail={tail}"
                    );
                }
            }
        }
    }

    #[test]
    fn malformed_final_header_preserves_text_before_partial_utf8_at_eof() {
        let mut ids = encode_harmony("<|start|>assistant<|channel|>final Hello ").unwrap();
        let character = encode_harmony("\u{1f468}").unwrap();
        assert!(character.len() > 1);
        ids.push(character[0]);
        assert!(decode_harmony(&ids).is_err());
        for chunk_size in [ids.len(), 1] {
            let mut parser = gpt_oss_unified(&[]);
            parser.initialize(&[]).unwrap();
            let mut events = Vec::new();
            for chunk in ids.chunks(chunk_size) {
                events.extend(parser.push_tokens(chunk).unwrap());
            }
            events.extend(parser.finish().unwrap().events);
            assert_eq!(
                crate::unified::assemble(&events),
                vec![UnifiedEvent::Text {
                    text: " Hello ".into()
                }]
            );
        }
    }

    #[test]
    fn truncated_tool_message_drops_prose_at_eof() {
        let events = parse_text(
            "<|start|>assistant to=functions.lookup<|channel|>commentary <|message|>still thinking about it",
            UnifiedParserStartingState::None,
        );
        assert!(
            events.is_empty(),
            "truncated tool body leaked as text: {events:?}"
        );
    }

    #[test]
    fn token_and_text_inputs_match_with_utf8_fragments() {
        let input = "<|start|>assistant<|channel|>analysis<|message|>hé🙂<|end|><|start|>assistant<|channel|>final<|message|>ok<|return|>";
        let mut text_parser = gpt_oss_unified(&[]);
        let mut token_parser = gpt_oss_unified(&[]);
        text_parser.initialize(&[]).unwrap();
        token_parser.initialize(&[]).unwrap();
        let expected = text_parser.parse_complete(input).unwrap();
        let ids = crate::tool_calling::harmony::encode_harmony(input).unwrap();
        let mut actual = Vec::new();
        for chunk in ids.chunks(1) {
            actual.extend(token_parser.push_tokens(chunk).unwrap());
        }
        actual.extend(token_parser.finish().unwrap().events);
        assert_eq!(expected, crate::unified::assemble(&actual));
    }

    #[test]
    fn ordinary_token_ids_cannot_create_harmony_tool_calls() {
        let prefix = "<|start|>assistant<|channel|>final<|message|>before ";
        let marker_like =
            "<|start|>assistant to=functions.fake<|channel|>commentary<|message|>{\"x\":1}<|call|>";
        let suffix = " after<|return|>";
        let encoding = harmony_encoding().unwrap();
        let mut ids = encoding.tokenizer().encode_with_special_tokens(prefix);
        ids.extend(encoding.tokenizer().encode_ordinary(marker_like));
        ids.extend(encoding.tokenizer().encode_with_special_tokens(suffix));

        let mut parser = gpt_oss_unified(&[]);
        parser.initialize(&[]).unwrap();
        let mut actual = Vec::new();
        for chunk in ids.chunks(1) {
            actual.extend(parser.push_tokens(chunk).unwrap());
        }
        actual.extend(parser.finish().unwrap().events);
        assert_eq!(
            crate::unified::assemble(&actual),
            vec![crate::unified::UnifiedEvent::Text {
                text: format!("before {marker_like} after"),
            }]
        );

        let mut guided = gpt_oss_unified(&[]);
        guided
            .initialize_request(UnifiedParserInit {
                tool_output_mode: crate::unified::UnifiedToolOutputMode::GuidedJson {
                    named_tool: None,
                },
                invalid_guided_payload: super::super::InvalidGuidedPayloadPolicy::RecoverAsText,
                ..UnifiedParserInit::default()
            })
            .unwrap();
        let mut guided_events = Vec::new();
        for chunk in ids.chunks(1) {
            guided_events.extend(guided.push_tokens(chunk).unwrap());
        }
        guided_events.extend(guided.finish().unwrap().events);
        assert_eq!(
            crate::unified::assemble(&guided_events),
            vec![crate::unified::UnifiedEvent::Text {
                text: format!("before {marker_like} after"),
            }]
        );
    }

    #[test]
    fn guided_literal_shielding_is_reversible_for_adjacent_escape_characters() {
        let alphabet = ['<', GUIDED_LITERAL_ESCAPE, 'e', 'l'];
        for length in 0..=4 {
            for mut pattern in 0..4usize.pow(length) {
                let input: String = (0..length)
                    .map(|_| {
                        let ch = alphabet[pattern % 4];
                        pattern /= 4;
                        ch
                    })
                    .collect();
                assert_eq!(
                    restore_guided_literals(&shield_guided_literals(&input)),
                    input
                );
            }
        }
    }

    #[test]
    fn guided_recovery_preserves_private_use_characters_for_text_and_tokens() {
        let input = format!(
            r#"[{{"name":"lookup","arguments":{{"value":"<{}"#,
            GUIDED_LITERAL_ESCAPE
        );
        let encoding = harmony_encoding().unwrap();
        let ids = encoding.tokenizer().encode_ordinary(&input);
        for tokens in [false, true] {
            let mut parser = gpt_oss_unified(&[]);
            let length = if tokens { ids.len() } else { input.len() };
            for at in 0..=length {
                if !tokens && !input.is_char_boundary(at) {
                    continue;
                }
                parser.reset();
                parser
                    .initialize_request(UnifiedParserInit {
                        tool_output_mode: crate::unified::UnifiedToolOutputMode::GuidedJson {
                            named_tool: None,
                        },
                        invalid_guided_payload:
                            super::super::InvalidGuidedPayloadPolicy::RecoverAsText,
                        ..Default::default()
                    })
                    .unwrap();
                let mut events = Vec::new();
                if tokens {
                    events.extend(parser.push_tokens(&ids[..at]).unwrap());
                    events.extend(parser.push_tokens(&ids[at..]).unwrap());
                } else {
                    events.extend(parser.push(&input[..at]).unwrap());
                    events.extend(parser.push(&input[at..]).unwrap());
                }
                events.extend(parser.finish().unwrap().events);
                assert_eq!(
                    crate::unified::assemble(&events),
                    vec![UnifiedEvent::Text {
                        text: input.clone()
                    }],
                    "tokens={tokens} split={at}"
                );
            }
        }
    }

    #[test]
    fn guided_token_ids_preserve_literal_private_use_escape_character() {
        for value in [
            format!("<{GUIDED_LITERAL_ESCAPE}"),
            format!("{GUIDED_LITERAL_ESCAPE}<"),
            format!("{GUIDED_LITERAL_ESCAPE}{GUIDED_LITERAL_ESCAPE}<<"),
        ] {
            let input = format!(r#"[{{"name":"lookup","arguments":{{"value":"{value}"}}}}]"#);
            let ids = harmony_encoding()
                .unwrap()
                .tokenizer()
                .encode_ordinary(&input);
            for policy in [
                super::super::InvalidGuidedPayloadPolicy::Reject,
                super::super::InvalidGuidedPayloadPolicy::RecoverAsText,
                super::super::InvalidGuidedPayloadPolicy::StreamBestEffort,
            ] {
                for tokens in [false, true] {
                    let mut parser = gpt_oss_unified(&[]);
                    parser
                        .initialize_request(UnifiedParserInit {
                            starting_state: UnifiedParserStartingState::Response,
                            tool_output_mode: crate::unified::UnifiedToolOutputMode::GuidedJson {
                                named_tool: None,
                            },
                            invalid_guided_payload: policy,
                            ..Default::default()
                        })
                        .unwrap();
                    let mut events = Vec::new();
                    if tokens {
                        for chunk in ids.chunks(1) {
                            events.extend(parser.push_tokens(chunk).unwrap());
                        }
                    } else {
                        for chunk in every_char_chunk(&input) {
                            events.extend(parser.push(chunk).unwrap());
                        }
                    }
                    let expected = vec![UnifiedEvent::ToolCall {
                        name: "lookup".into(),
                        arguments: serde_json::json!({"value": value}),
                    }];
                    assert_eq!(
                        crate::unified::assemble(&events),
                        expected,
                        "before finish tokens={tokens} policy={policy:?}"
                    );
                    events.extend(parser.finish().unwrap().events);
                    assert_eq!(crate::unified::assemble(&events), expected);
                }
            }
        }
    }

    #[test]
    fn guided_token_restoration_does_not_rewrite_prior_accumulated_events() {
        let first = format!(
            "{START}assistant{CHANNEL}analysis{MESSAGE}private {}{MESSAGE_END}",
            GUIDED_LITERAL_ESCAPE
        );
        let second = format!("{START}assistant{CHANNEL}final{MESSAGE}done{TURN_END}");
        let mut parser = gpt_oss_unified(&[]);
        parser
            .initialize_request(UnifiedParserInit {
                tool_output_mode: crate::unified::UnifiedToolOutputMode::GuidedJson {
                    named_tool: None,
                },
                ..UnifiedParserInit::default()
            })
            .unwrap();

        let mut output = UnifiedParserOutput::default();
        let first_ids = crate::tool_calling::harmony::encode_harmony(&first).unwrap();
        parser
            .parse_token_ids_into(&first_ids, &mut output)
            .unwrap();
        assert!(output.events.iter().any(
            |event| matches!(event, UnifiedParserEvent::Reasoning(text) if text.ends_with(GUIDED_LITERAL_ESCAPE))
        ));

        let second_ids = crate::tool_calling::harmony::encode_harmony(&second).unwrap();
        parser
            .parse_token_ids_into(&second_ids, &mut output)
            .unwrap();
        assert!(output.events.iter().any(
            |event| matches!(event, UnifiedParserEvent::Reasoning(text) if text == &format!("private {}", GUIDED_LITERAL_ESCAPE))
        ));
        assert!(!output.events.iter().any(
            |event| matches!(event, UnifiedParserEvent::Reasoning(text) if text == "private <")
        ));
    }

    #[test]
    fn prompt_prefilled_tool_headers_match_native_for_every_suffix_token_split() {
        let body = "{\"q\":\"x\"}<|call|>";
        for (prefix, suffix) in [
            (
                "<|start|>assistant<|channel|>commentary to=functions.lookup<|message|>",
                body,
            ),
            (
                "<|start|>assistant to=functions.lookup<|channel|>commentary<|message|>",
                body,
            ),
            (
                "<|start|>assistant",
                " to=functions.lookup<|channel|>commentary<|message|>{\"q\":\"x\"}<|call|>",
            ),
            (
                "<|start|>assistant to=functions.lookup",
                "<|channel|>commentary<|message|>{\"q\":\"x\"}<|call|>",
            ),
            (
                "<|start|>assistant<|channel|>commentary to=functions.lookup",
                "<|message|>{\"q\":\"x\"}<|call|>",
            ),
        ] {
            let prompt_token_ids = crate::tool_calling::harmony::encode_harmony(prefix).unwrap();
            let native_owner = HarmonyNative::new(true, false);
            let prefill = native_owner
                .guided_prompt_prefill(&UnifiedParserInit {
                    prompt_token_ids: prompt_token_ids.clone(),
                    ..UnifiedParserInit::default()
                })
                .unwrap();
            assert_eq!(prefill.as_deref(), Some(prefix));
            let suffix_ids = crate::tool_calling::harmony::encode_harmony(suffix).unwrap();
            let mut native = gpt_oss_unified(&[]);
            native
                .initialize_request(UnifiedParserInit {
                    prompt_token_ids: prompt_token_ids.clone(),
                    ..UnifiedParserInit::default()
                })
                .unwrap();
            let mut expected = native.push(suffix).unwrap();
            expected.extend(native.finish().unwrap().events);
            let expected = crate::unified::assemble(&expected);

            for split in 0..=suffix_ids.len() {
                let mut guided = GuidedRouted::new(HarmonyNative::new(true, false));
                guided
                    .initialize_request(UnifiedParserInit {
                        prompt_token_ids: prompt_token_ids.clone(),
                        tool_output_mode: crate::unified::UnifiedToolOutputMode::GuidedJson {
                            named_tool: None,
                        },
                        ..UnifiedParserInit::default()
                    })
                    .unwrap();
                assert_eq!(guided.guided_prefill.as_deref(), Some(prefix));
                let header_is_complete = prefix.ends_with(MESSAGE);
                assert_eq!(
                    guided.guided.as_ref().unwrap().named_tool.as_deref(),
                    header_is_complete.then_some("lookup")
                );
                assert_eq!(guided.guided_prefill_pending, !header_is_complete);
                let mut actual = Vec::new();
                actual.extend(guided.push_tokens(&suffix_ids[..split]).unwrap());
                actual.extend(guided.push_tokens(&suffix_ids[split..]).unwrap());
                actual.extend(guided.finish().unwrap().events);
                let state = guided.guided.as_ref().unwrap();
                assert_eq!(
                    crate::unified::assemble(&actual),
                    expected,
                    "prefill={prefix:?}, suffix token split={split}, input={:?}, json={:?}, native={:?}, emitted={}",
                    state.input,
                    state.json,
                    state.native_header.native,
                    state.payload_emitted
                );
            }

            for split in 0..=suffix.len() {
                let mut guided = GuidedRouted::new(HarmonyNative::new(true, false));
                guided
                    .initialize_request(UnifiedParserInit {
                        prompt_token_ids: prompt_token_ids.clone(),
                        tool_output_mode: crate::unified::UnifiedToolOutputMode::GuidedJson {
                            named_tool: None,
                        },
                        ..UnifiedParserInit::default()
                    })
                    .unwrap();
                let mut actual = guided.push(&suffix[..split]).unwrap();
                actual.extend(guided.push(&suffix[split..]).unwrap());
                actual.extend(guided.finish().unwrap().events);
                assert_eq!(
                    crate::unified::assemble(&actual),
                    expected,
                    "text prefill={prefix:?}, suffix byte split={split}"
                );
            }
        }
    }

    #[test]
    fn reset_recovers_generated_text_held_by_an_incomplete_prefilled_header() {
        let prefix = "<|start|>assistant to=functions.lookup";
        let generated = "<|channel|>commentary";
        let prompt_token_ids = crate::tool_calling::harmony::encode_harmony(prefix).unwrap();
        let init = || UnifiedParserInit {
            prompt_token_ids: prompt_token_ids.clone(),
            tool_output_mode: crate::unified::UnifiedToolOutputMode::GuidedJson {
                named_tool: None,
            },
            ..UnifiedParserInit::default()
        };

        for split in 0..=generated.len() {
            let mut text = GuidedRouted::new(HarmonyNative::new(true, false));
            text.initialize_request(init()).unwrap();
            assert!(text.push(&generated[..split]).unwrap().is_empty());
            assert!(text.push(&generated[split..]).unwrap().is_empty());
            assert_eq!(text.reset(), generated, "text split={split}");
        }

        let generated_token_ids = crate::tool_calling::harmony::encode_harmony(generated).unwrap();
        for split in 0..=generated_token_ids.len() {
            let mut tokens = GuidedRouted::new(HarmonyNative::new(true, false));
            tokens.initialize_request(init()).unwrap();
            assert!(
                tokens
                    .push_tokens(&generated_token_ids[..split])
                    .unwrap()
                    .is_empty()
            );
            assert!(
                tokens
                    .push_tokens(&generated_token_ids[split..])
                    .unwrap()
                    .is_empty()
            );
            assert_eq!(tokens.reset(), generated, "token split={split}");
        }
    }

    #[test]
    fn prefilled_tool_header_reset_reuses_selected_tool() {
        let prefix = "<|start|>assistant to=functions.lookup";
        let prompt_token_ids = crate::tool_calling::harmony::encode_harmony(prefix).unwrap();
        let suffix_ids = crate::tool_calling::harmony::encode_harmony(
            "<|channel|>commentary<|message|>{\"q\":\"x\"}<|call|>",
        )
        .unwrap();
        let mut parser = GuidedRouted::new(HarmonyNative::new(true, false));
        parser
            .initialize_request(UnifiedParserInit {
                prompt_token_ids,
                tool_output_mode: crate::unified::UnifiedToolOutputMode::GuidedJson {
                    named_tool: None,
                },
                ..UnifiedParserInit::default()
            })
            .unwrap();
        assert_eq!(parser.guided.as_ref().unwrap().named_tool, None);
        assert!(parser.guided_prefill_pending);

        let parse_suffix = |parser: &mut GuidedRouted<HarmonyNative>| {
            let mut events = parser.push_tokens(&suffix_ids).unwrap();
            events.extend(parser.finish().unwrap().events);
            crate::unified::assemble(&events)
        };
        let first = parse_suffix(&mut parser);
        let recovered = parser.reset();
        assert!(recovered.is_empty());
        assert_eq!(parser.guided.as_ref().unwrap().named_tool, None);
        assert!(parser.guided_prefill_pending);
        let second = parse_suffix(&mut parser);
        assert_eq!(second, first);
        assert_eq!(
            first,
            vec![UnifiedEvent::ToolCall {
                name: "lookup".to_string(),
                arguments: serde_json::json!({"q": "x"}),
            }]
        );
    }

    #[test]
    fn guided_token_reset_restores_shielded_bytes_before_reuse() {
        let partial = format!(r#"{{"value":"{}<"#, GUIDED_LITERAL_ESCAPE);
        let ids = harmony_encoding()
            .unwrap()
            .tokenizer()
            .encode_ordinary(&partial);
        let mut parser = gpt_oss_unified(&[]);
        parser
            .initialize_request(UnifiedParserInit {
                starting_state: UnifiedParserStartingState::Response,
                tool_output_mode: crate::unified::UnifiedToolOutputMode::GuidedJson {
                    named_tool: None,
                },
                ..UnifiedParserInit::default()
            })
            .unwrap();
        for chunk in ids.chunks(1) {
            parser.push_tokens(chunk).unwrap();
        }

        assert_eq!(parser.reset(), partial);
        let reuse = r#"[{"name":"lookup","arguments":{"q":"x"}}]"#;
        let reuse_ids = harmony_encoding()
            .unwrap()
            .tokenizer()
            .encode_ordinary(reuse);
        let mut events = Vec::new();
        for chunk in reuse_ids.chunks(1) {
            events.extend(parser.push_tokens(chunk).unwrap());
        }
        events.extend(parser.finish().unwrap().events);
        assert!(matches!(
            crate::unified::assemble(&events).as_slice(),
            [crate::unified::UnifiedEvent::ToolCall { .. }]
        ));
    }

    #[test]
    fn guided_token_finish_clears_incomplete_utf8_decoder_state() {
        let mut parser = GuidedRouted::new(HarmonyNative::new(true, false));
        parser
            .initialize_request(UnifiedParserInit {
                starting_state: UnifiedParserStartingState::Response,
                tool_output_mode: crate::unified::UnifiedToolOutputMode::GuidedJson {
                    named_tool: None,
                },
                invalid_guided_payload: crate::unified::InvalidGuidedPayloadPolicy::RecoverAsText,
                ..UnifiedParserInit::default()
            })
            .unwrap();
        let mut output = UnifiedParserOutput::default();
        parser.parse_token_ids_into(&[126], &mut output).unwrap();
        assert!(!parser.native.pending_token_bytes.is_empty());

        parser.finish().unwrap();
        assert!(parser.native.pending_token_bytes.is_empty());
    }

    #[test]
    fn guided_token_error_preserves_events_committed_before_rejection() {
        let input = concat!(
            "<|start|>assistant<|channel|>analysis<|message|>why<|end|>",
            r#"{"bad":1}"#
        );
        let init = UnifiedParserInit {
            tool_output_mode: crate::unified::UnifiedToolOutputMode::GuidedJson {
                named_tool: None,
            },
            invalid_guided_payload: crate::unified::InvalidGuidedPayloadPolicy::Reject,
            ..UnifiedParserInit::default()
        };
        let mut text_parser = gpt_oss_unified(&[]);
        text_parser.initialize_request(init.clone()).unwrap();
        let mut text_output = UnifiedParserOutput::default();
        let text_result = text_parser.parse_into(input, &mut text_output);

        let mut token_parser = gpt_oss_unified(&[]);
        token_parser.initialize_request(init).unwrap();
        let token_ids = crate::tool_calling::harmony::encode_harmony(input).unwrap();
        let mut token_output = UnifiedParserOutput::default();
        let token_result = token_parser.parse_token_ids_into(&token_ids, &mut token_output);

        assert!(text_result.is_err());
        assert!(token_result.is_err());
        assert_eq!(token_output.events, text_output.events);
    }

    #[test]
    fn token_input_emits_valid_prefix_before_a_split_utf8_character() {
        let mut pending = Vec::new();
        let (prefix, invalid) = take_utf8_prefix(&mut pending, b"prefix\xf0\x9f");
        assert_eq!(prefix, "prefix");
        assert_eq!(invalid, None);
        assert_eq!(pending, b"\xf0\x9f");

        let (suffix, invalid) = take_utf8_prefix(&mut pending, b"\x99\x82");
        assert_eq!(suffix, "🙂");
        assert_eq!(invalid, None);
        assert!(pending.is_empty());
    }

    #[test]
    fn every_text_split_matches_the_unsplit_ordered_stream() {
        let input = concat!(
            "<|start|>assistant<|channel|>analysis<|message|>check🙂<|end|>",
            "<|start|>assistant<|channel|>commentary to=functions.lookup<|message|>",
            r#"{"marker":"<|call|>","q":"hé"}"#,
            "<|call|><|start|>assistant<|channel|>final<|message|>done<|return|>"
        );
        let expected = parse_text(input, UnifiedParserStartingState::None);
        for split in input
            .char_indices()
            .map(|(index, _)| index)
            .chain(std::iter::once(input.len()))
        {
            let mut parser = gpt_oss_unified(&[]);
            parser.initialize(&[]).unwrap();
            let mut actual = parser.push(&input[..split]).unwrap();
            actual.extend(parser.push(&input[split..]).unwrap());
            actual.extend(parser.finish().unwrap().events);
            assert_eq!(
                crate::unified::assemble(&actual),
                crate::unified::assemble(&expected),
                "text split at byte {split} changed the ordered output"
            );
        }
    }

    #[test]
    fn unknown_tool_and_empty_arguments_are_preserved() {
        let events = parse_text(
            "<|start|>assistant<|channel|>commentary to=functions.not_offered<|message|><|call|>",
            UnifiedParserStartingState::None,
        );
        assert!(matches!(
            events.as_slice(),
            [UnifiedParserEvent::ToolCall(call)]
                if call.name.as_deref() == Some("not_offered") && call.arguments == "{}"
        ));
    }

    #[test]
    fn response_prefill_routes_content_without_a_redundant_header() {
        let events = parse_text("ready", UnifiedParserStartingState::Response);
        assert!(matches!(
            events.as_slice(),
            [UnifiedParserEvent::Text(text)] if text == "ready"
        ));
    }

    #[test]
    fn guided_json_routes_harmony_reasoning_before_bare_payload() {
        let input = concat!(
            "<|start|>assistant<|channel|>analysis<|message|>checking<|end|>",
            r#"[{"name":"get_weather","arguments":{"city":"Paris"}}]"#
        );
        let mut parser = gpt_oss_unified(&[]);
        parser
            .initialize_request(UnifiedParserInit {
                tool_output_mode: crate::unified::UnifiedToolOutputMode::GuidedJson {
                    named_tool: None,
                },
                ..UnifiedParserInit::default()
            })
            .unwrap();
        let mut events = parser.push(input).unwrap();
        events.extend(parser.finish().unwrap().events);
        assert!(matches!(
            events.as_slice(),
            [
                UnifiedParserEvent::Reasoning(reasoning),
                UnifiedParserEvent::ToolCall(call)
            ] if reasoning == "checking"
                && call.name.as_deref() == Some("get_weather")
                && call.arguments == r#"{"city":"Paris"}"#
        ));

        let mut token_parser = gpt_oss_unified(&[]);
        token_parser
            .initialize_request(UnifiedParserInit {
                tool_output_mode: crate::unified::UnifiedToolOutputMode::GuidedJson {
                    named_tool: None,
                },
                ..UnifiedParserInit::default()
            })
            .unwrap();
        let token_ids = crate::tool_calling::harmony::encode_harmony(input).unwrap();
        let mut token_events = Vec::new();
        for chunk in token_ids.chunks(1) {
            token_events.extend(token_parser.push_tokens(chunk).unwrap());
        }
        token_events.extend(token_parser.finish().unwrap().events);
        assert_eq!(token_events, events);
    }

    #[test]
    fn guided_json_holds_split_harmony_headers_until_routed() {
        let chunks = [
            START,
            "assistant",
            CHANNEL,
            "analysis",
            MESSAGE,
            "checking",
            MESSAGE_END,
            r#"[{"name":"get_weather","arguments":{"city":"Paris"}}]"#,
        ];
        let mut parser = gpt_oss_unified(&[]);
        parser
            .initialize_request(UnifiedParserInit {
                tool_output_mode: crate::unified::UnifiedToolOutputMode::GuidedJson {
                    named_tool: None,
                },
                ..UnifiedParserInit::default()
            })
            .unwrap();
        let mut events = Vec::new();
        for chunk in chunks {
            events.extend(parser.push(chunk).unwrap());
        }
        events.extend(parser.finish().unwrap().events);
        assert!(matches!(
            events.as_slice(),
            [UnifiedParserEvent::Reasoning(reasoning), UnifiedParserEvent::ToolCall(call)]
                if reasoning == "checking"
                    && call.name.as_deref() == Some("get_weather")
                    && call.arguments == r#"{"city":"Paris"}"#
        ));
    }

    #[test]
    fn guided_final_header_wraps_json_after_optional_reasoning() {
        let arguments = serde_json::json!({"literal_text":"assistantfinal"});
        for reasoning in [
            "",
            "<|channel|>analysis<|message|>checking<|end|>",
            "<|channel|>analysis<|message|>checking",
        ] {
            for named in [false, true] {
                let payload = if named {
                    arguments.clone()
                } else {
                    serde_json::json!([{"name":"record_literal","parameters":arguments}])
                };
                let input = format!(
                    "{reasoning}<|start|>assistant<|channel|>final<|message|>{payload}<|return|>"
                );
                let mut expected = Vec::new();
                if !reasoning.is_empty() {
                    expected.push(UnifiedEvent::Reasoning {
                        text: "checking".into(),
                    });
                }
                expected.push(UnifiedEvent::ToolCall {
                    name: "record_literal".into(),
                    arguments: arguments.clone(),
                });
                let ids = encode_harmony(&input).unwrap();
                for tokens in [false, true] {
                    let splits: Vec<_> = if tokens {
                        (0..=ids.len()).collect()
                    } else {
                        input
                            .char_indices()
                            .map(|(at, _)| at)
                            .chain([input.len()])
                            .collect()
                    };
                    let end = if tokens { ids.len() } else { input.len() };
                    let mut schedules: Vec<_> = splits.iter().map(|at| vec![0, *at, end]).collect();
                    schedules.push(splits);
                    for boundaries in schedules {
                        let mut parser = gpt_oss_unified(&[]);
                        parser
                            .initialize_request(UnifiedParserInit {
                                tool_output_mode: UnifiedToolOutputMode::GuidedJson {
                                    named_tool: named.then(|| "record_literal".into()),
                                },
                                invalid_guided_payload:
                                    crate::unified::InvalidGuidedPayloadPolicy::StreamBestEffort,
                                ..Default::default()
                            })
                            .unwrap();
                        let mut events = Vec::new();
                        for pair in boundaries.windows(2) {
                            events.extend(if tokens {
                                parser.push_tokens(&ids[pair[0]..pair[1]]).unwrap()
                            } else {
                                parser.push(&input[pair[0]..pair[1]]).unwrap()
                            });
                        }
                        events.extend(parser.finish().unwrap().events);
                        assert_eq!(
                            crate::unified::assemble(&events),
                            expected,
                            "tokens={tokens}, boundaries={boundaries:?}, named={named}, reasoning={reasoning:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn guided_final_header_ends_unterminated_reasoning_before_prose() {
        for header in [
            "<|start|>assistant<|channel|>final<|message|>",
            "<|channel|>final<|message|>",
        ] {
            let input = format!("<|channel|>analysis<|message|>thought{header}Hello!<|return|>");
            for chunks in [vec![input.as_str()], every_char_chunk(&input)] {
                let mut parser = gpt_oss_unified(&[]);
                parser
                    .initialize_request(UnifiedParserInit {
                        tool_output_mode: UnifiedToolOutputMode::GuidedJson { named_tool: None },
                        invalid_guided_payload:
                            crate::unified::InvalidGuidedPayloadPolicy::StreamBestEffort,
                        ..Default::default()
                    })
                    .unwrap();
                let mut events = Vec::new();
                for chunk in chunks {
                    events.extend(parser.push(chunk).unwrap());
                }
                events.extend(parser.finish().unwrap().events);
                assert_eq!(
                    crate::unified::assemble(&events),
                    vec![
                        UnifiedEvent::Reasoning {
                            text: "thought".into()
                        },
                        UnifiedEvent::Text {
                            text: "Hello!".into()
                        },
                    ]
                );
            }
        }
    }

    #[test]
    fn guided_native_harmony_envelope_is_suppressed_as_one_span() {
        let input = concat!(
            "<|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>",
            r#"{"city": "Paris"}"#,
            "<|call|>"
        );
        for policy in [
            super::super::InvalidGuidedPayloadPolicy::Reject,
            super::super::InvalidGuidedPayloadPolicy::RecoverAsText,
            super::super::InvalidGuidedPayloadPolicy::StreamBestEffort,
        ] {
            for named in [false, true] {
                let mut schedules = vec![vec![input], every_char_chunk(input)];
                schedules.extend(
                    input
                        .char_indices()
                        .map(|(at, _)| vec![&input[..at], &input[at..]]),
                );
                let mut parser = gpt_oss_unified(&[]);
                for chunks in schedules {
                    parser.reset();
                    parser
                        .initialize_request(UnifiedParserInit {
                            tool_output_mode: crate::unified::UnifiedToolOutputMode::GuidedJson {
                                named_tool: named.then(|| "get_weather".into()),
                            },
                            invalid_guided_payload: policy,
                            ..Default::default()
                        })
                        .unwrap();
                    for chunk in chunks {
                        assert!(parser.push(chunk).unwrap().is_empty());
                    }
                    let finish = parser.finish();
                    if policy == super::super::InvalidGuidedPayloadPolicy::Reject {
                        assert!(finish.unwrap_err().to_string().contains("Missing"));
                    } else {
                        assert!(finish.unwrap().events.is_empty());
                    }
                }
            }
        }
    }

    #[test]
    fn guided_native_json_owns_reasoning_markers_until_handoff() {
        let input = concat!(
            "<|start|>assistant<|channel|>analysis<|message|>before<|end|>",
            "<|start|>assistant<|channel|>commentary to=functions.f <|constrain|>json<|message|>",
            r#"{"x": "<|start|>assistant<|channel|>analysis<|message|>quoted<|end|>"}"#,
            "<|call|>",
            r#"[{"name": "get_weather", "arguments": {"city": "Paris"}}]"#
        );
        let chunks = every_char_chunk(input);
        let events = parse_guided(&chunks, UnifiedParserStartingState::None);
        assert!(
            matches!(
                events.as_slice(),
                [UnifiedEvent::Reasoning { text: reasoning }, UnifiedEvent::ToolCall { name, arguments }]
                    if reasoning == "before"
                        && name == "get_weather"
                        && *arguments == serde_json::json!({"city":"Paris"})
            ),
            "unexpected guided events: {events:?}"
        );
    }

    #[test]
    fn guided_reasoning_holds_recipient_first_harmony_headers_until_complete() {
        let input = concat!(
            "<|start|>assistant<|channel|>analysis<|message|>before<|end|>",
            "<|start|>assistant to=functions.f<|channel|>commentary <|constrain|>json<|message|>",
            r#"{"x": "<|start|>assistant<|channel|>redacted<|message|>quoted<|end|>"}"#,
            "<|call|>",
            r#"[{"name": "get_weather", "arguments": {"city": "Paris"}}]"#
        );
        let chunks = every_char_chunk(input);
        let events = parse_guided(&chunks, UnifiedParserStartingState::None);
        assert!(
            matches!(
                events.as_slice(),
                [UnifiedEvent::Reasoning { text: reasoning }, UnifiedEvent::ToolCall { name, arguments }]
                    if reasoning == "before"
                        && name == "get_weather"
                        && *arguments == serde_json::json!({"city":"Paris"})
            ),
            "unexpected guided events: {events:?}"
        );
    }

    #[test]
    fn guided_response_keeps_quoted_harmony_markers_literal() {
        let before = concat!(
            "I mean ",
            "<|start|>assistant<|channel|>redacted<|message|>self literal<|end|>",
            r#"[{"name": "get_weather", "arguments": {"city": "Paris"}}]"#
        );
        let after = concat!(
            r#"[{"name": "get_weather", "arguments": {"city": "Paris"}}]"#,
            "I mean ",
            "<|start|>assistant<|channel|>redacted<|message|>self literal<|end|>"
        );
        let response = UnifiedParserStartingState::Response;
        for input in [before, after] {
            let chunks = every_char_chunk(input);
            let events = parse_guided(&chunks, response);
            assert!(
                matches!(
                    events.as_slice(),
                    [UnifiedEvent::Text { text }, UnifiedEvent::ToolCall { name, .. }]
                        if input == before
                            && text == "I mean <|start|>assistant<|channel|>redacted<|message|>self literal<|end|>"
                            && name == "get_weather"
                ) || matches!(
                    events.as_slice(),
                    [UnifiedEvent::ToolCall { name, .. }, UnifiedEvent::Text { text }]
                        if input == after
                            && text == "I mean <|start|>assistant<|channel|>redacted<|message|>self literal<|end|>"
                            && name == "get_weather"
                ),
                "unexpected guided events for response input: {events:?}"
            );
        }
    }

    #[test]
    fn prefilled_reasoning_and_reset_reuse() {
        let mut parser = gpt_oss_unified(&[]);
        parser
            .initialize_request(UnifiedParserInit {
                starting_state: UnifiedParserStartingState::Reasoning,
                ..UnifiedParserInit::default()
            })
            .unwrap();
        assert!(
            matches!(parser.push("still").unwrap().as_slice(), [UnifiedParserEvent::Reasoning(text)] if text == "still")
        );
        parser.finish().unwrap();
        parser.reset();
        assert!(
            matches!(parser.push("again").unwrap().as_slice(), [UnifiedParserEvent::Reasoning(text)] if text == "again")
        );
    }

    #[test]
    fn orphan_handoff_preserves_prose_but_truncated_tool_body_does_not() {
        let orphan = format!("{HANDOFF}still thinking about it");
        let truncated = format!(
            "{START}assistant to=functions.lookup{CHANNEL}commentary{MESSAGE}{{\"q\":still thinking about it"
        );
        let mut parser = gpt_oss_unified(&[]);
        for (input, expected) in [
            (
                &orphan,
                vec![UnifiedEvent::Text {
                    text: "still thinking about it".into(),
                }],
            ),
            (&truncated, Vec::new()),
        ] {
            for split in input.char_indices().map(|(at, _)| at).chain([input.len()]) {
                parser.reset();
                parser.initialize(&[]).unwrap();
                let mut events = parser.push(&input[..split]).unwrap();
                events.extend(parser.push(&input[split..]).unwrap());
                events.extend(parser.finish().unwrap().events);
                assert_eq!(crate::unified::assemble(&events), expected, "split={split}");
            }
            let ids = crate::tool_calling::harmony::encode_harmony(input).unwrap();
            for split in 0..=ids.len() {
                parser.reset();
                parser.initialize(&[]).unwrap();
                let mut events = parser.push_tokens(&ids[..split]).unwrap();
                events.extend(parser.push_tokens(&ids[split..]).unwrap());
                events.extend(parser.finish().unwrap().events);
                assert_eq!(
                    crate::unified::assemble(&events),
                    expected,
                    "token split={split}"
                );
            }
        }
    }

    #[test]
    fn native_arguments_stream_before_handoff_and_complete_once() {
        let mut parser = gpt_oss_unified(&[]);
        parser.initialize(&[]).unwrap();
        let mut events = parser
            .push(concat!(
                "<|channel|>analysis to=functions.echo<|message|>",
                "{\"value\":\"first"
            ))
            .unwrap();
        assert!(events.iter().any(|event| matches!(event,
            UnifiedParserEvent::ToolCall(call) if call.name.as_deref() == Some("echo")
        )));
        assert!(events.iter().any(|event| matches!(event,
            UnifiedParserEvent::ToolCall(call) if call.arguments.contains("first") && !call.complete
        )));
        let second = parser.push(" second\"}").unwrap();
        assert!(second.iter().any(|event| matches!(event,
            UnifiedParserEvent::ToolCall(call) if call.arguments.contains("second") && !call.complete
        )));
        events.extend(second);
        events.extend(parser.push("<|call|>").unwrap());
        events.extend(parser.finish().unwrap().events);
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event,
                    UnifiedParserEvent::ToolCall(call) if call.complete
                ))
                .count(),
            1
        );
        assert_eq!(
            crate::unified::assemble(&events),
            vec![crate::unified::UnifiedEvent::ToolCall {
                name: "echo".into(),
                arguments: serde_json::json!({"value":"first second"}),
            }]
        );
    }

    #[test]
    fn tool_marker_inside_json_string_is_payload() {
        let events = parse_text(
            concat!(
                "<|start|>assistant<|channel|>commentary to=functions.lookup<|message|>",
                r#"{"marker":"<|call|>"}"#,
                "<|call|>"
            ),
            UnifiedParserStartingState::None,
        );
        assert_eq!(
            crate::unified::assemble(&events),
            vec![crate::unified::UnifiedEvent::ToolCall {
                name: "lookup".into(),
                arguments: serde_json::json!({"marker":"<|call|>"}),
            }]
        );
    }

    #[test]
    fn every_harmony_control_marker_inside_json_string_is_payload() {
        let markers = CONTROL_MARKERS.join("");
        let arguments = serde_json::json!({"markers": markers}).to_string();
        let input = format!(
            "<|start|>assistant to=functions.lookup<|channel|>commentary <|message|>{arguments}<|call|>"
        );
        let events = parse_text(&input, UnifiedParserStartingState::None);
        assert_eq!(
            crate::unified::assemble(&events),
            vec![crate::unified::UnifiedEvent::ToolCall {
                name: "lookup".into(),
                arguments: serde_json::from_str(&arguments).unwrap(),
            }]
        );
    }

    fn assert_native_splits(input: &str, expected: &[UnifiedEvent]) {
        assert_parser_splits(gpt_oss_unified(&[]), input, expected);
    }

    fn assert_parser_splits(
        mut parser: Box<dyn crate::unified::UnifiedParser>,
        input: &str,
        expected: &[UnifiedEvent],
    ) {
        for split in input.char_indices().map(|(at, _)| at).chain([input.len()]) {
            parser.reset();
            parser.initialize(&[]).unwrap();
            let mut events = parser.push(&input[..split]).unwrap();
            events.extend(parser.push(&input[split..]).unwrap());
            events.extend(parser.finish().unwrap().events);
            assert_eq!(
                crate::unified::assemble(&events),
                expected,
                "text split={split}"
            );
        }
        let ids = encode_harmony(input).unwrap();
        for split in 0..=ids.len() {
            parser.reset();
            parser.initialize(&[]).unwrap();
            let mut events = parser.push_tokens(&ids[..split]).unwrap();
            events.extend(parser.push_tokens(&ids[split..]).unwrap());
            events.extend(parser.finish().unwrap().events);
            assert_eq!(
                crate::unified::assemble(&events),
                expected,
                "token split={split}"
            );
        }
    }

    #[test]
    fn implicit_channel_headers_preserve_visible_separator_bytes() {
        for whitespace in [" ", "\t", "\r\n", " \n\t", "\u{2003}"] {
            for (header, body, end, event) in [
                (
                    "analysis",
                    "think",
                    MESSAGE_END,
                    UnifiedEvent::Reasoning {
                        text: "think".into(),
                    },
                ),
                (
                    "commentary to=functions.lookup",
                    r#"{"q":"x"}"#,
                    HANDOFF,
                    UnifiedEvent::ToolCall {
                        name: "lookup".into(),
                        arguments: serde_json::json!({"q":"x"}),
                    },
                ),
            ] {
                let input = format!("before{whitespace}{CHANNEL}{header}{MESSAGE}{body}{end}after");
                let expected = [
                    UnifiedEvent::Text {
                        text: format!("before{whitespace}"),
                    },
                    event,
                    UnifiedEvent::Text {
                        text: "after".into(),
                    },
                ];
                assert_native_splits(&input, &expected);
                assert_parser_splits(gpt_oss_legacy_tool_adapter(), &input, &expected);
            }
        }
    }

    #[test]
    fn orphan_controls_preserve_pending_text_and_break_recipient_prefixes() {
        for prefix in [" ", "\t", "\r\n", "t", "to", "to=", "to= "] {
            let channel_input = format!("{prefix}{CHANNEL}analysis{MESSAGE}think{MESSAGE_END}");
            assert_native_splits(
                &channel_input,
                &[
                    UnifiedEvent::Text {
                        text: prefix.into(),
                    },
                    UnifiedEvent::Reasoning {
                        text: "think".into(),
                    },
                ],
            );
            for marker in [MESSAGE, FORMAT, HANDOFF, MESSAGE_END, TURN_END] {
                let input = format!("{prefix}{marker}answer");
                assert_native_splits(
                    &input,
                    &[UnifiedEvent::Text {
                        text: format!("{prefix}answer"),
                    }],
                );
            }
        }
        let input = format!("to={HANDOFF}functions.lookup{MESSAGE}answer");
        assert_native_splits(
            &input,
            &[UnifiedEvent::Text {
                text: "to=functions.lookupanswer".into(),
            }],
        );
    }

    #[test]
    fn legacy_and_native_recipient_channel_routing() {
        for channel in [None, Some("commentary"), Some("analysis"), Some("final")] {
            for recipient in [
                None,
                Some("functions.lookup"),
                Some("browser"),
                Some("functions."),
            ] {
                for format in ["", " <|constrain|>json"] {
                    for legacy in [false, true] {
                        let recipient_header =
                            recipient.map(|r| format!(" to={r}")).unwrap_or_default();
                        let channel_header =
                            channel.map(|c| format!("{CHANNEL}{c}")).unwrap_or_default();
                        let input = format!(
                            "{START}assistant{recipient_header}{channel_header}{format}{MESSAGE}{{\"q\":\"x\"}}{HANDOFF}"
                        );
                        let expected = match recipient {
                            Some("functions.lookup")
                                if !legacy || channel == Some("commentary") =>
                            {
                                vec![UnifiedEvent::ToolCall {
                                    name: "lookup".into(),
                                    arguments: serde_json::json!({"q": "x"}),
                                }]
                            }
                            Some(_) => vec![],
                            None if channel == Some("analysis") => vec![UnifiedEvent::Reasoning {
                                text: r#"{"q":"x"}"#.into(),
                            }],
                            None if legacy && channel == Some("commentary") => vec![],
                            None => vec![UnifiedEvent::Text {
                                text: r#"{"q":"x"}"#.into(),
                            }],
                        };
                        let parser = if legacy {
                            gpt_oss_legacy_tool_adapter()
                        } else {
                            gpt_oss_unified(&[])
                        };
                        assert_parser_splits(parser, &input, &expected);
                    }
                }
            }
        }
    }

    #[test]
    fn legacy_commentary_without_recipient_stays_suppressed() {
        for content_type in [String::new(), format!(" {FORMAT}json")] {
            for end in [HANDOFF, MESSAGE_END, TURN_END] {
                let input = format!(
                    "{START}assistant{CHANNEL}commentary{content_type}{MESSAGE}{{\"location\":\"NYC\"}}{end}{START}assistant{CHANNEL}final{MESSAGE}answer{TURN_END}"
                );
                assert_parser_splits(
                    gpt_oss_legacy_tool_adapter(),
                    &input,
                    &[UnifiedEvent::Text {
                        text: "answer".into(),
                    }],
                );
            }
        }
    }

    #[test]
    fn native_argument_offsets_include_leading_whitespace() {
        for whitespace in ["", " ", "\t", "\r\n", "  \n\t"] {
            let input = format!(
                "{START}assistant{CHANNEL}commentary to=functions.lookup {FORMAT}json{MESSAGE}{whitespace}{{\"q\":\"x\"}}{HANDOFF}"
            );
            assert_native_splits(
                &input,
                &[UnifiedEvent::ToolCall {
                    name: "lookup".into(),
                    arguments: serde_json::json!({"q":"x"}),
                }],
            );
        }
    }

    #[test]
    fn content_type_header_retains_final_content() {
        let input =
            format!("{START}assistant{CHANNEL}final {FORMAT}json{MESSAGE}{{\"a\":1}}{TURN_END}");
        assert_native_splits(
            &input,
            &[UnifiedEvent::Text {
                text: "{\"a\":1}".into(),
            }],
        );
    }

    #[test]
    fn implicit_recipient_first_header_is_not_visible_text() {
        let input = format!(
            " to=functions.lookup{CHANNEL}commentary {FORMAT}json{MESSAGE}{{\"q\":\"x\"}}{HANDOFF}"
        );
        assert_native_splits(
            &input,
            &[UnifiedEvent::ToolCall {
                name: "lookup".into(),
                arguments: serde_json::json!({"q":"x"}),
            }],
        );
    }

    #[test]
    fn malformed_string_does_not_erase_following_message() {
        let input = format!(
            "{START}assistant to=functions.lookup {FORMAT}json{MESSAGE}{{\"q\":\"abc{HANDOFF}{START}assistant{CHANNEL}final{MESSAGE}Answer{TURN_END}"
        );
        assert_native_splits(
            &input,
            &[UnifiedEvent::Text {
                text: "Answer".into(),
            }],
        );
    }

    #[test]
    fn quoted_complete_headers_remain_argument_data() {
        for boundary in [START, MESSAGE_END, TURN_END, HANDOFF] {
            for prefix in ["", "escaped \\\"", "backslash \\\\"] {
                let value = format!(
                    "{prefix}é{boundary}{START}assistant{CHANNEL}final{MESSAGE}literal{TURN_END}끝"
                );
                let args = serde_json::json!({"before":"", "q":value, "after":true});
                let input = format!(
                    "{START}assistant to=functions.lookup{CHANNEL}commentary{MESSAGE}{args}{HANDOFF}"
                );
                assert_native_splits(
                    &input,
                    &[UnifiedEvent::ToolCall {
                        name: "lookup".into(),
                        arguments: args,
                    }],
                );
            }
        }
    }

    #[test]
    fn malformed_quoted_boundaries_preserve_later_assistant_channels() {
        for boundary in [START, MESSAGE_END, TURN_END, HANDOFF] {
            let next = if boundary == START { "" } else { START };
            let input = format!(
                "{START}assistant to=functions.lookup{MESSAGE}{{\"q\":\"abc{boundary}{next}assistant{CHANNEL}analysis{MESSAGE}checking{MESSAGE_END}{START}assistant{CHANNEL}final {FORMAT}json{MESSAGE}{{\"a\":1}}{TURN_END}"
            );
            assert_native_splits(
                &input,
                &[
                    UnifiedEvent::Reasoning {
                        text: "checking".into(),
                    },
                    UnifiedEvent::Text {
                        text: "{\"a\":1}".into(),
                    },
                ],
            );
        }
    }

    #[test]
    fn native_complete_message_boundaries_and_invalid_provisional_calls() {
        for end in [HANDOFF, MESSAGE_END, TURN_END] {
            let header = format!("{START}assistant to=functions.lookup{MESSAGE}");
            let mut parser = gpt_oss_unified(&[]);
            parser.initialize(&[]).unwrap();
            let first = parser.push(&format!("{header}{{\"q\":\"first")).unwrap();
            assert!(first.iter().any(|event| matches!(event, UnifiedParserEvent::ToolCall(call) if !call.complete && call.arguments.contains("first"))));
            let mut events = first;
            events.extend(parser.push(&format!(" second\"}}{end}")).unwrap());
            events.extend(parser.finish().unwrap().events);
            assert_eq!(
                crate::unified::assemble(&events),
                vec![UnifiedEvent::ToolCall {
                    name: "lookup".into(),
                    arguments: serde_json::json!({"q":"first second"})
                }]
            );
            assert_eq!(
                events
                    .iter()
                    .filter(
                        |event| matches!(event, UnifiedParserEvent::ToolCall(call) if call.complete)
                    )
                    .count(),
                1
            );
            let input = format!(
                "{header}{{\"q\":\"x\"}} oops{end}{START}assistant{CHANNEL}final{MESSAGE}done{TURN_END}"
            );
            assert_native_splits(
                &input,
                &[UnifiedEvent::Text {
                    text: "done".into(),
                }],
            );
            let invalid = parse_text(&input, UnifiedParserStartingState::None);
            assert!(invalid.iter().any(
                |event| matches!(event, UnifiedParserEvent::ToolCall(call) if !call.complete)
            ));
            assert!(invalid.iter().all(
                |event| !matches!(event, UnifiedParserEvent::ToolCall(call) if call.complete)
            ));
        }
    }

    #[test]
    fn quoted_argument_tail_handles_short_chunks_long_fields_and_reuse() {
        let marker = format!("{HANDOFF}{START}assistant{CHANNEL}final{MESSAGE}quoted{TURN_END}");
        let long = "quote \\\" slash \\\\ é🙂 ".repeat(128);
        for position in 0..3 {
            let mut fields = vec![
                serde_json::json!(marker).to_string(),
                "\"\"".into(),
                "true".into(),
            ];
            fields.insert(position, serde_json::json!(long).to_string());
            let body = format!(
                "{{{}}}",
                fields
                    .iter()
                    .enumerate()
                    .map(|(i, v)| format!("\"k{i}\":{v}"))
                    .collect::<Vec<_>>()
                    .join(",")
            );
            let input = format!("{START}assistant to=functions.lookup{MESSAGE}{body}{HANDOFF}");
            let expected = vec![UnifiedEvent::ToolCall {
                name: "lookup".into(),
                arguments: serde_json::from_str(&body).unwrap(),
            }];
            let mut parser = gpt_oss_unified(&[]);
            for chunk_size in [1, 17, 128] {
                parser.reset();
                parser.initialize(&[]).unwrap();
                let mut events = Vec::new();
                let chars = input
                    .char_indices()
                    .map(|(at, _)| at)
                    .chain([input.len()])
                    .collect::<Vec<_>>();
                for indexes in chars.windows(chunk_size + 1).step_by(chunk_size) {
                    events.extend(
                        parser
                            .push(&input[indexes[0]..*indexes.last().unwrap()])
                            .unwrap(),
                    );
                }
                let consumed = ((chars.len() - 1) / chunk_size) * chunk_size;
                if consumed < chars.len() - 1 {
                    events.extend(parser.push(&input[chars[consumed]..]).unwrap());
                }
                events.extend(parser.finish().unwrap().events);
                assert_eq!(
                    crate::unified::assemble(&events),
                    expected,
                    "position={position}, chars={chunk_size}"
                );
                parser.reset();
                parser.initialize(&[]).unwrap();
                let mut events = Vec::new();
                for ids in encode_harmony(&input).unwrap().chunks(chunk_size) {
                    events.extend(parser.push_tokens(ids).unwrap());
                }
                events.extend(parser.finish().unwrap().events);
                assert_eq!(
                    crate::unified::assemble(&events),
                    expected,
                    "position={position}, tokens={chunk_size}"
                );
            }
            let scalar = serde_json::json!(marker).to_string();
            assert_native_splits(
                &format!("{START}assistant to=functions.lookup{MESSAGE}{scalar}{HANDOFF}"),
                &[UnifiedEvent::ToolCall {
                    name: "lookup".into(),
                    arguments: serde_json::json!(marker),
                }],
            );
        }
    }

    #[test]
    fn recipient_looking_plaintext_resumes_before_finish() {
        // Assembled corpus GOLDEN cannot express this pre-finish progress deadline.
        for prose in [
            "today is clear",
            "to=functions.lookup ordinary prose",
            "to=functions.bad/name prose",
            " to=functions.lookup\nnow prose",
        ] {
            let mut parser = gpt_oss_unified(&[]);
            parser.initialize(&[]).unwrap();
            let mut events = Vec::new();
            for character in prose.chars() {
                events.extend(parser.push(&character.to_string()).unwrap());
            }
            assert_eq!(
                crate::unified::assemble(&events),
                vec![UnifiedEvent::Text { text: prose.into() }]
            );
            assert!(parser.finish().unwrap().events.is_empty());
        }
    }

    #[test]
    fn repeated_ambiguous_starts_recover_only_the_complete_following_header() {
        for count in [1, 64, 2048] {
            let input = format!(
                "{START}assistant to=functions.lookup{MESSAGE}{{\"q\":\"abc{}{START}assistant{CHANNEL}final{MESSAGE}Answer{TURN_END}",
                START.repeat(count)
            );
            assert_eq!(
                crate::unified::assemble(&parse_text(&input, UnifiedParserStartingState::None)),
                vec![UnifiedEvent::Text {
                    text: "Answer".into()
                }]
            );
        }
    }
}
