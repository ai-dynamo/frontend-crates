// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unified parser for the Kimi K3 XTML chat format.
//!
//! K3 wraps one assistant message into ordered `think`, `response`, and
//! `tools` channels built from `<|open|>`, `<|close|>`, and `<|sep|>`.
//! Generation normally starts inside a prompt-prefilled `think` or `response`
//! channel; [`UnifiedParser::initialize`] supplies that initial state.

mod structural_tag;

#[cfg(test)]
mod tests;

pub use structural_tag::KimiK3StructuralTagBuilder;

use serde_json::Value;
use winnow::ascii::{multispace0 as ws0, multispace1 as ws1};
use winnow::combinator::{alt, delimited, eof, preceded, repeat, seq, terminated};
use winnow::error::{ContextError, ErrMode, ModalResult, StrContext};
use winnow::prelude::*;
use winnow::stream::Partial;
use winnow::token::{literal, rest, take_till, take_until, take_while};

use super::scan::{MarkerScanState, parse_buffered_event, safe_text_len_mul, take_until_marker};
use super::{
    UnifiedParser, UnifiedParserOutput, UnifiedParserPrefill, UnifiedStructuralTagBuilder,
};
use crate::{Tool, ToolCallDelta};

pub const KIMI_K3_FAMILY: &str = "kimi_k3";

pub(crate) const OPEN: &str = "<|open|>";
pub(crate) const SEP: &str = "<|sep|>";
pub(crate) const END_OF_MSG: &str = "<|end_of_msg|>";

pub(crate) const THINK_OPEN: &str = "<|open|>think<|sep|>";
pub(crate) const THINK_CLOSE: &str = "<|close|>think<|sep|>";
pub(crate) const RESPONSE_OPEN: &str = "<|open|>response<|sep|>";
pub(crate) const RESPONSE_CLOSE: &str = "<|close|>response<|sep|>";
pub(crate) const TOOLS_OPEN: &str = "<|open|>tools<|sep|>";
pub(crate) const TOOLS_CLOSE: &str = "<|close|>tools<|sep|>";
pub(crate) const MESSAGE_CLOSE: &str = "<|close|>message<|sep|>";
pub(crate) const CALL_OPEN: &str = "<|open|>call";
pub(crate) const CALL_CLOSE: &str = "<|close|>call<|sep|>";
pub(crate) const ARG_OPEN: &str = "<|open|>argument";
pub(crate) const ARG_CLOSE: &str = "<|close|>argument<|sep|>";
pub(crate) const JSON_OPEN: &str = "<|open|>json";
pub(crate) const JSON_CLOSE: &str = "<|close|>json<|sep|>";

const IDLE_MARKERS: &[&str] = &[
    THINK_OPEN,
    RESPONSE_OPEN,
    TOOLS_OPEN,
    MESSAGE_CLOSE,
    END_OF_MSG,
];
const REASONING_MARKERS: &[&str] = &[THINK_CLOSE, END_OF_MSG];
const RESPONSE_MARKERS: &[&str] = &[RESPONSE_CLOSE, TOOLS_OPEN, MESSAGE_CLOSE, END_OF_MSG];
const EPILOGUE_MARKERS: &[&str] = &[TOOLS_OPEN, MESSAGE_CLOSE, END_OF_MSG];
const TOOLS_MARKERS: &[&str] = &[CALL_OPEN, TOOLS_CLOSE, MESSAGE_CLOSE, END_OF_MSG];

const COMPACT_MIN_CONSUMED: usize = 4096;

type KimiK3Input<'i> = Partial<&'i str>;

#[derive(Debug, Clone, PartialEq, Eq)]
enum KimiK3Event {
    Text { len: usize },
    Reasoning { len: usize },
    Skip,
    ThinkOpen,
    ThinkClose,
    ResponseOpen,
    ResponseClose,
    ToolsOpen,
    ToolsClose,
    MessageEnd,
    CallOpen { name: String, index: Option<String> },
    CallComplete { arguments: String },
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
enum KimiK3Mode {
    #[default]
    Idle,
    Reasoning,
    Response,
    Epilogue,
    Tools,
    Call {
        name: String,
        index: Option<String>,
        scan: MarkerScanState,
    },
    Done,
}

/// Ordered state machine for Kimi K3 reasoning, response, and tool channels.
pub struct KimiK3UnifiedParser {
    buffer: String,
    cursor: usize,
    mode: KimiK3Mode,
    call_ids: Vec<String>,
}

impl KimiK3UnifiedParser {
    pub fn new(_tools: &[Tool]) -> Self {
        Self {
            buffer: String::new(),
            cursor: 0,
            mode: KimiK3Mode::Idle,
            call_ids: Vec::new(),
        }
    }

    fn pending(&self) -> &str {
        &self.buffer[self.cursor..]
    }

    fn take_pending(&mut self) -> String {
        let pending = self.pending().to_string();
        self.buffer.clear();
        self.cursor = 0;
        pending
    }

    fn advance(&mut self, consumed: usize) {
        self.cursor += consumed;
        if self.cursor >= COMPACT_MIN_CONSUMED && self.cursor >= self.buffer.len() / 2 {
            self.buffer.drain(..self.cursor);
            self.cursor = 0;
        }
    }

    fn apply_event(
        &mut self,
        event: KimiK3Event,
        output: &mut UnifiedParserOutput,
    ) -> anyhow::Result<()> {
        match event {
            KimiK3Event::Text { len } => output.push_text(self.pending()[..len].to_string()),
            KimiK3Event::Reasoning { len } => {
                output.push_reasoning(self.pending()[..len].to_string());
            }
            KimiK3Event::Skip => {}
            KimiK3Event::ThinkOpen => self.mode = KimiK3Mode::Reasoning,
            KimiK3Event::ThinkClose => self.mode = KimiK3Mode::Idle,
            KimiK3Event::ResponseOpen => self.mode = KimiK3Mode::Response,
            KimiK3Event::ResponseClose => self.mode = KimiK3Mode::Epilogue,
            KimiK3Event::ToolsOpen => self.mode = KimiK3Mode::Tools,
            KimiK3Event::ToolsClose => self.mode = KimiK3Mode::Epilogue,
            KimiK3Event::MessageEnd => self.mode = KimiK3Mode::Done,
            KimiK3Event::CallOpen { name, index } => {
                self.mode = KimiK3Mode::Call {
                    name,
                    index,
                    scan: MarkerScanState::default(),
                };
            }
            KimiK3Event::CallComplete { arguments } => {
                let mode = std::mem::replace(&mut self.mode, KimiK3Mode::Tools);
                let KimiK3Mode::Call { name, index, .. } = mode else {
                    anyhow::bail!("Kimi K3 call completion without an active tool call");
                };
                if name.is_empty() {
                    return Ok(());
                }

                let tool_index = self.call_ids.len();
                self.call_ids
                    .push(tool_call_id_for(&name, index.as_deref()));
                output.push_call(ToolCallDelta {
                    tool_index,
                    name: Some(name),
                    arguments,
                });
            }
        }
        Ok(())
    }

    fn reset_state(&mut self) -> String {
        self.mode = KimiK3Mode::Idle;
        self.call_ids.clear();
        self.take_pending()
    }
}

impl UnifiedParser for KimiK3UnifiedParser {
    fn create(tools: &[Tool]) -> anyhow::Result<Box<dyn UnifiedParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self::new(tools)))
    }

    fn initialize(&mut self, prefill: UnifiedParserPrefill) -> anyhow::Result<()> {
        self.buffer.clear();
        self.cursor = 0;
        self.call_ids.clear();
        self.mode = match prefill {
            UnifiedParserPrefill::None => KimiK3Mode::Idle,
            UnifiedParserPrefill::Reasoning => KimiK3Mode::Reasoning,
            UnifiedParserPrefill::Response => KimiK3Mode::Response,
        };
        Ok(())
    }

    fn preserve_special_tokens(&self) -> bool {
        true
    }

    fn structural_tag_builder(&self) -> Option<&dyn UnifiedStructuralTagBuilder> {
        Some(&structural_tag::KIMI_K3_STRUCTURAL_TAG_BUILDER)
    }

    fn tool_call_id(&self, tool_index: usize) -> Option<&str> {
        self.call_ids.get(tool_index).map(String::as_str)
    }

    fn parse_into(&mut self, chunk: &str, output: &mut UnifiedParserOutput) -> anyhow::Result<()> {
        self.buffer.push_str(chunk);

        loop {
            let parsed = {
                let pending = &self.buffer[self.cursor..];
                let mode = &mut self.mode;
                parse_buffered_event(pending, |input| parse_next_kimi_k3_event(input, mode))?
            };
            let Some((event, consumed)) = parsed else {
                break;
            };
            self.apply_event(event, output)?;
            self.advance(consumed);
        }

        Ok(())
    }

    fn finish(&mut self) -> anyhow::Result<UnifiedParserOutput> {
        let mut output = UnifiedParserOutput::default();

        match &self.mode {
            KimiK3Mode::Idle | KimiK3Mode::Response => {
                output.push_text(self.take_pending());
            }
            KimiK3Mode::Reasoning => {
                output.push_reasoning(self.take_pending());
            }
            KimiK3Mode::Epilogue | KimiK3Mode::Done => {
                self.buffer.clear();
                self.cursor = 0;
            }
            KimiK3Mode::Tools if self.pending().is_empty() => {
                self.buffer.clear();
                self.cursor = 0;
            }
            KimiK3Mode::Tools | KimiK3Mode::Call { .. } => {
                anyhow::bail!("incomplete Kimi K3 tool call");
            }
        }

        self.mode = KimiK3Mode::Idle;
        Ok(output)
    }

    fn reset(&mut self) -> String {
        self.reset_state()
    }
}

fn tool_call_id_for(name: &str, index: Option<&str>) -> String {
    match index {
        None => name.to_string(),
        Some(raw) => match raw.parse::<i64>() {
            Ok(one_based) => format!("{name}:{}", one_based - 1),
            Err(_) => format!("{name}:{raw}"),
        },
    }
}

fn parse_next_kimi_k3_event(
    input: &mut KimiK3Input<'_>,
    mode: &mut KimiK3Mode,
) -> ModalResult<KimiK3Event> {
    match mode {
        KimiK3Mode::Idle => parse_idle_event(input),
        KimiK3Mode::Reasoning => parse_reasoning_event(input),
        KimiK3Mode::Response => parse_response_event(input),
        KimiK3Mode::Epilogue => parse_epilogue_event(input),
        KimiK3Mode::Tools => parse_tools_event(input),
        KimiK3Mode::Call { scan, .. } => call_body_event(input, scan),
        KimiK3Mode::Done => parse_done_event(input),
    }
}

fn parse_idle_event(input: &mut KimiK3Input<'_>) -> ModalResult<KimiK3Event> {
    alt((
        literal(THINK_OPEN).value(KimiK3Event::ThinkOpen),
        literal(RESPONSE_OPEN).value(KimiK3Event::ResponseOpen),
        literal(TOOLS_OPEN).value(KimiK3Event::ToolsOpen),
        message_end_event,
        safe_idle_text_event,
    ))
    .parse_next(input)
}

fn parse_reasoning_event(input: &mut KimiK3Input<'_>) -> ModalResult<KimiK3Event> {
    alt((
        literal(THINK_CLOSE).value(KimiK3Event::ThinkClose),
        literal(END_OF_MSG).value(KimiK3Event::MessageEnd),
        safe_reasoning_event,
    ))
    .parse_next(input)
}

fn parse_response_event(input: &mut KimiK3Input<'_>) -> ModalResult<KimiK3Event> {
    alt((
        literal(RESPONSE_CLOSE).value(KimiK3Event::ResponseClose),
        literal(TOOLS_OPEN).value(KimiK3Event::ToolsOpen),
        message_end_event,
        safe_response_text_event,
    ))
    .parse_next(input)
}

fn parse_epilogue_event(input: &mut KimiK3Input<'_>) -> ModalResult<KimiK3Event> {
    alt((
        literal(TOOLS_OPEN).value(KimiK3Event::ToolsOpen),
        message_end_event,
        skip_epilogue_noise_event,
    ))
    .parse_next(input)
}

fn parse_tools_event(input: &mut KimiK3Input<'_>) -> ModalResult<KimiK3Event> {
    alt((
        call_open_event,
        literal(TOOLS_CLOSE).value(KimiK3Event::ToolsClose),
        message_end_event,
        skip_tools_noise_event,
    ))
    .parse_next(input)
}

fn message_end_event(input: &mut KimiK3Input<'_>) -> ModalResult<KimiK3Event> {
    alt((literal(MESSAGE_CLOSE), literal(END_OF_MSG)))
        .value(KimiK3Event::MessageEnd)
        .parse_next(input)
}

fn parse_done_event(input: &mut KimiK3Input<'_>) -> ModalResult<KimiK3Event> {
    rest.value(KimiK3Event::Skip).parse_next(input)
}

fn safe_idle_text_event(input: &mut KimiK3Input<'_>) -> ModalResult<KimiK3Event> {
    safe_text_len_mul(input, IDLE_MARKERS).map(|len| KimiK3Event::Text { len })
}

fn safe_reasoning_event(input: &mut KimiK3Input<'_>) -> ModalResult<KimiK3Event> {
    safe_text_len_mul(input, REASONING_MARKERS).map(|len| KimiK3Event::Reasoning { len })
}

fn safe_response_text_event(input: &mut KimiK3Input<'_>) -> ModalResult<KimiK3Event> {
    safe_text_len_mul(input, RESPONSE_MARKERS).map(|len| KimiK3Event::Text { len })
}

fn skip_epilogue_noise_event(input: &mut KimiK3Input<'_>) -> ModalResult<KimiK3Event> {
    safe_text_len_mul(input, EPILOGUE_MARKERS).map(|_| KimiK3Event::Skip)
}

fn skip_tools_noise_event(input: &mut KimiK3Input<'_>) -> ModalResult<KimiK3Event> {
    safe_text_len_mul(input, TOOLS_MARKERS).map(|_| KimiK3Event::Skip)
}

fn call_open_event(input: &mut KimiK3Input<'_>) -> ModalResult<KimiK3Event> {
    let (attrs,) = seq!(
        _: literal(CALL_OPEN),
        take_until(0.., SEP),
        _: literal(SEP),
    )
    .parse_next(input)?;
    let attrs = parse_tag_attrs(attrs)?;

    Ok(KimiK3Event::CallOpen {
        name: attr_value(&attrs, "tool").unwrap_or_default().to_string(),
        index: attr_value(&attrs, "index")
            .filter(|index| !index.is_empty())
            .map(str::to_string),
    })
}

fn call_body_event(
    input: &mut KimiK3Input<'_>,
    scan: &mut MarkerScanState,
) -> ModalResult<KimiK3Event> {
    let (body,) = seq!(
        take_until_marker(CALL_CLOSE, scan),
        _: literal(CALL_CLOSE),
    )
    .parse_next(input)?;
    Ok(KimiK3Event::CallComplete {
        arguments: parse_call_arguments(body)?,
    })
}

fn parse_call_arguments(body: &str) -> ModalResult<String> {
    let mut input = body;
    terminated(
        delimited(ws0, alt((json_block_arguments, typed_arguments)), ws0),
        eof,
    )
    .parse_next(&mut input)
    .map_err(|_| xtml_error("Kimi K3 call body"))
}

fn json_block_arguments(input: &mut &str) -> ModalResult<String> {
    let (raw,) = seq!(
        _: literal(JSON_OPEN),
        _: take_until(0.., SEP),
        _: literal(SEP),
        take_until(0.., JSON_CLOSE),
        _: literal(JSON_CLOSE),
    )
    .parse_next(input)?;
    Ok(raw.to_string())
}

fn typed_arguments(input: &mut &str) -> ModalResult<String> {
    let pairs: Vec<(String, Value)> =
        repeat(0.., terminated(argument_block, ws0)).parse_next(input)?;
    serialize_argument_pairs(&pairs).map_err(|_| xtml_error("Kimi K3 arguments"))
}

fn serialize_argument_pairs(pairs: &[(String, Value)]) -> serde_json::Result<String> {
    let mut output = String::from("{");
    for (index, (key, value)) in pairs.iter().enumerate() {
        if index != 0 {
            output.push(',');
        }
        output.push_str(&serde_json::to_string(key)?);
        output.push(':');
        output.push_str(&serde_json::to_string(value)?);
    }
    output.push('}');
    Ok(output)
}

fn argument_block(input: &mut &str) -> ModalResult<(String, Value)> {
    let (attrs, raw) = seq!(
        _: literal(ARG_OPEN),
        take_until(0.., SEP),
        _: literal(SEP),
        take_until(0.., ARG_CLOSE),
        _: literal(ARG_CLOSE),
    )
    .parse_next(input)?;
    let attrs = parse_tag_attrs(attrs)?;

    let key = attr_value(&attrs, "key").unwrap_or_default().to_string();
    let arg_type = attr_value(&attrs, "type").unwrap_or("string");
    Ok((key, decode_argument_value(arg_type, raw)))
}

fn decode_argument_value(arg_type: &str, raw: &str) -> Value {
    if arg_type == "string" {
        return Value::String(raw.to_string());
    }
    serde_json::from_str(raw).unwrap_or_else(|_| Value::String(raw.to_string()))
}

fn parse_tag_attrs(attrs: &str) -> ModalResult<Vec<(String, String)>> {
    let mut input = attrs;
    terminated(repeat(0.., preceded(ws1, tag_attr)), (ws0, eof))
        .parse_next(&mut input)
        .map_err(|_| xtml_error("XTML tag attributes"))
}

fn tag_attr(input: &mut &str) -> ModalResult<(String, String)> {
    seq!(
        take_while(1.., |character: char| character.is_alphanumeric() || character == '_')
            .map(str::to_string),
        _: literal("=\""),
        take_till(0.., '"').map(unescape_attr_value),
        _: literal("\""),
    )
    .parse_next(input)
}

fn unescape_attr_value(value: &str) -> String {
    value.replace("&quot;", "\"").replace("&amp;", "&")
}

fn attr_value<'a>(attrs: &'a [(String, String)], key: &str) -> Option<&'a str> {
    attrs
        .iter()
        .find(|(name, _)| name == key)
        .map(|(_, value)| value.as_str())
}

fn xtml_error(label: &'static str) -> ErrMode<ContextError> {
    let mut error = ContextError::new();
    error.push(StrContext::Label(label));
    ErrMode::Cut(error)
}
