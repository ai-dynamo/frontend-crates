// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Muse Glimmer ATEM tool-call parser.
//!
//! The authoritative grammar is the model's shipped decode spec: the
//! `response_template` key in `tokenizer_config.json` at
//! <https://huggingface.co/meta-models/Muse-Glimmer-30B>. Engine reference
//! implementations: vLLM `muse_glimmer_tool_parser.py` (PR #51655) and SGLang
//! `muse_glimmer_detector.py` (parser name `muse`).
//!
//! Assistant output is a chain of recipient-routed messages:
//!
//! ```text
//! <|start|>assistant to=self<|message|>...reasoning...<|eom|>
//! <|start|>assistant to=get_weather<|message|>
//! <atem:function_calls>
//! <atem:invoke name="get_weather">
//! <atem:parameter name="city">Paris</atem:parameter>
//! </atem:invoke>
//! </atem:function_calls><|eom|>
//! <|start|>assistant to=user<|message|>...final answer...<|eot|>
//! ```
//!
//! The generation prompt ends with `<|start|>assistant`, so the FIRST message
//! of a turn arrives header-less (` to=self<|message|>` with no `<|start|>`).
//! `<|eom|>` closes a message with more to follow; `<|eot|>` ends the turn.
//! Recipient `self` is reasoning, `user` (or no recipient) is the final
//! answer, and anything else is a tool call whose body is ATEM XML. Parallel
//! calls arrive as separate `<|eom|>`-chained messages; one message may also
//! carry several `<atem:invoke>` blocks (`repeats: true` in the spec).
//!
//! Channel scoping is the core safety property: an `<atem:invoke>` quoted
//! inside a `to=self` or `to=user` body must NOT parse as a real call. This
//! parser therefore segments messages first and extracts calls only from
//! tool-recipient bodies, like both engine parsers. It deliberately does NOT
//! adopt vLLM's scan-everything fallback for fully unframed ATEM text: that
//! fallback exists to paper over `skip_special_tokens=true` upstream, and it
//! turns ATEM markup quoted in an unwrapped user body into a live call.
//! SGLang has no such fallback either.
//!
//! Pair with the `muse_glimmer` reasoning parser: the reasoning stage consumes
//! `to=self` / `to=user` channels and forwards tool channels verbatim (framing
//! intact, normalized to start with `<|start|>`) so the streaming jail can key
//! on `<|start|>` and this parser can strip the framing.

use std::sync::OnceLock;

use regex::Regex;
use uuid::Uuid;

use super::super::ToolDefinition;
use super::super::response::{CalledFunction, ToolCallResponse, ToolCallType};

pub(crate) const START: &str = "<|start|>";
pub(crate) const MESSAGE: &str = "<|message|>";
pub(crate) const EOM: &str = "<|eom|>";
pub(crate) const EOT: &str = "<|eot|>";

pub(crate) const FUNCTION_CALLS_OPEN: &str = "<atem:function_calls>";
pub(crate) const INVOKE_OPEN_PREFIX: &str = "<atem:invoke";
pub(crate) const INVOKE_CLOSE: &str = "</atem:invoke>";

pub(crate) const REASONING_RECIPIENT: &str = "self";
pub(crate) const USER_RECIPIENT: &str = "user";

/// Matches one complete ATEM invoke open tag and captures the tool name.
/// Mirrors the `tool_calls.open_pattern` in the model's `response_template`
/// (attributes may precede or follow `name`, as SGLang also allows).
fn invoke_open_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r#"<atem:invoke\b[^>]*?\bname="(?P<name>[^"]+)"[^>]*?>"#).unwrap())
}

/// Matches one complete ATEM parameter element and captures key + raw value.
/// Mirrors the `tag_pattern` in the model's `response_template`.
fn parameter_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r#"(?s)<atem:parameter\b[^>]*?\bname="(?P<key>[^"]+)"[^>]*?>(?P<value>.*?)</atem:parameter>"#,
        )
        .unwrap()
    })
}

/// One recipient-routed assistant message located inside a larger text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ChannelMessage<'a> {
    /// `None` for a bare `<|message|>` header with no `to=` recipient.
    pub recipient: Option<&'a str>,
    /// Byte offset where the header begins (`<|start|>`, `to=`, or `<|message|>`).
    pub header_start: usize,
    /// Byte offset of the first body byte (just past `<|message|>`).
    pub body_start: usize,
    /// Byte offset one past the last body byte.
    pub body_end: usize,
    /// Byte offset where scanning should resume (past the terminator when closed).
    pub next_pos: usize,
    /// Whether the body saw its own `<|eom|>` / `<|eot|>` terminator.
    pub closed: bool,
    /// Whether the body was cut at a bare `to=<rcpt><|message|>` header
    /// (missing-`<|eom|>` recovery), so the NEXT message may resolve bare.
    pub bare_cut: bool,
}

impl ChannelMessage<'_> {
    pub(crate) fn is_tool_channel(&self) -> bool {
        self.recipient
            .is_some_and(|r| r != REASONING_RECIPIENT && r != USER_RECIPIENT)
    }
}

/// A recipient is a run of non-whitespace, non-`<` characters (SGLang's
/// `to=([^\s<]+)`; broader than vLLM's `[A-Za-z0-9_.\-]+` so namespaced and
/// unicode tool names survive).
fn is_recipient_char(c: char) -> bool {
    !c.is_whitespace() && c != '<'
}

/// Resolve the header that ends at the `<|message|>` found at `msg_pos`.
///
/// Returns `(header_start, recipient)`. The recipient must immediately abut
/// `<|message|>` (vLLM's anchoring); an optional `<|start|>assistant` prefix
/// and the non-newline whitespace between prompt framing and `to=` are
/// absorbed into the header.
pub(crate) fn resolve_header(text: &str, msg_pos: usize) -> (usize, Option<&str>) {
    let before = &text[..msg_pos];

    // Maximal recipient-charactered run ending at `<|message|>`.
    let run_start = before
        .char_indices()
        .rev()
        .take_while(|(_, c)| is_recipient_char(*c))
        .last()
        .map(|(i, _)| i)
        .unwrap_or(msg_pos);
    let run = &before[run_start..];

    let (mut header_start, recipient) = match run.find("to=") {
        Some(rel) => {
            let recipient = &run[rel + 3..];
            if recipient.is_empty() {
                (msg_pos, None)
            } else {
                (run_start + rel, Some(recipient))
            }
        }
        None => (msg_pos, None),
    };

    // Absorb `<|start|>assistant` (with the whitespace the template renders
    // between its parts) into the header so it never leaks as body text. The
    // role word only counts when `<|start|>` really precedes it; prose that
    // happens to end in "assistant" stays prose and takes the bare-header
    // path below.
    let ws_start = |s: &str| s.len() - s.trim_end_matches(|c: char| c.is_whitespace()).len();
    let pre = &text[..header_start];
    let pre_trimmed = &pre[..pre.len() - ws_start(pre)];
    let framed_start = match pre_trimmed.strip_suffix("assistant") {
        Some(stripped) => {
            let stripped_trimmed = &stripped[..stripped.len() - ws_start(stripped)];
            stripped_trimmed
                .ends_with(START)
                .then(|| stripped_trimmed.len() - START.len())
        }
        // `<|start|><|message|>` / `<|start|>to=...` without the role word.
        None => pre_trimmed
            .ends_with(START)
            .then(|| pre_trimmed.len() - START.len()),
    };
    if let Some(start) = framed_start {
        header_start = start;
    } else if recipient.is_some() {
        // A bare `to=` header (the prompt consumed `<|start|>assistant`):
        // absorb the template's separating space so it does not leak.
        let gap = &text[..header_start];
        let gap_ws = gap.len() - gap.trim_end_matches([' ', '\t']).len();
        header_start -= gap_ws;
    }

    (header_start, recipient)
}

/// Find the next channel message at or after `from`.
///
/// The body ends at its `<|eom|>` / `<|eot|>` terminator, at the start of the
/// next `<|start|>`-framed header, or at end of text. A REASONING body also
/// ends at a bare `to=<rcpt><|message|>` header: the model has been observed
/// to leave its analysis channel without emitting `<|eom|>`, writing the tool
/// header directly, and a terminator-only scan would swallow the call.
///
/// `allow_bare` gates whether a header may resolve WITHOUT `<|start|>`
/// framing. Only the turn's first message (the prompt consumed
/// `<|start|>assistant`) and the message after a bare-cut reasoning body are
/// legitimately bare; a bare-looking header anywhere else is quoted text, and
/// resolving it would promote content into a live tool channel.
pub(crate) fn next_channel_message(
    text: &str,
    from: usize,
    allow_bare: bool,
) -> Option<ChannelMessage<'_>> {
    let msg_rel = text[from..].find(MESSAGE)?;
    let msg_pos = from + msg_rel;
    let (mut header_start, mut recipient) = resolve_header(text, msg_pos);
    let framed = text[header_start..].starts_with(START);
    if !framed && !allow_bare && recipient.is_some() {
        // Quoted bare header: keep the `to=...` text as body/prose and treat
        // the marker as a recipient-less content header.
        header_start = msg_pos;
        recipient = None;
    }
    let header_start = header_start.max(from);
    let body_start = msg_pos + MESSAGE.len();

    let mut body_end = text.len();
    let mut next_pos = text.len();
    let mut closed = false;

    for terminator in [EOM, EOT] {
        if let Some(rel) = text[body_start..body_end].find(terminator) {
            body_end = body_start + rel;
            next_pos = body_end + terminator.len();
            closed = true;
        }
    }

    // A body can never legitimately contain a framed header; cut there. The
    // spec's own `start_anchor` is `<|start|>assistant`: `<|start|>` is a
    // reserved special token, so its decoded form marks a REAL channel switch
    // (the model emitted the token), never quotable prose. vLLM segments
    // identically. Bare headers cut only reasoning bodies
    // (missing-<|eom|> recovery).
    let is_reasoning = recipient == Some(REASONING_RECIPIENT);
    let mut bare_cut = false;
    let body = &text[body_start..body_end];
    let start_pos = body.find(START);
    // The slice starts at the true body start, which is an anchor.
    let bare_pos = if is_reasoning {
        bare_header_pos(body, None)
    } else {
        None
    };
    let boundary = match (start_pos, bare_pos) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (a, b) => a.or(b),
    };
    if let Some(rel) = boundary {
        bare_cut = bare_pos == Some(rel) && start_pos != Some(rel);
        body_end = body_start + rel;
        next_pos = body_end;
        closed = false;
    }

    Some(ChannelMessage {
        recipient,
        header_start,
        body_start,
        body_end,
        next_pos,
        closed,
        bare_cut,
    })
}

/// Position of the first `to=<recipient><|message|>` run inside `body`.
///
/// The `to=` must start the body or follow whitespace: this scan bounds the
/// missing-`<|eom|>` RECOVERY heuristic, and the observed defect always
/// separates the abandoned reasoning from the header with whitespace. An
/// unanchored match would cut mid-word (`pota`|`to=...`) and promote
/// concatenated prose into a tool channel. Header RESOLUTION at real message
/// boundaries stays unanchored like both engines' regexes.
pub(crate) fn bare_header_pos(body: &str, prev: Option<char>) -> Option<usize> {
    let mut search = 0;
    while let Some(rel) = body[search..].find("to=") {
        let at = search + rel;
        let anchored = if at == 0 {
            // `prev` is the character right before `body`: None at a real
            // body start (which is an anchor), else the last drained byte of
            // the same body, so a mid-word `to=` split by a chunk boundary
            // stays unanchored.
            prev.is_none_or(|c| c.is_whitespace())
        } else {
            body[..at]
                .chars()
                .next_back()
                .is_some_and(|c| c.is_whitespace())
        };
        let after = &body[at + 3..];
        let rcpt_len = after
            .char_indices()
            .take_while(|(_, c)| is_recipient_char(*c))
            .map(|(i, c)| i + c.len_utf8())
            .last()
            .unwrap_or(0);
        if anchored && rcpt_len > 0 && after[rcpt_len..].starts_with(MESSAGE) {
            return Some(at);
        }
        search = at + 3;
    }
    None
}

/// The normalized wire framing this crate uses when the reasoning stage
/// forwards a tool channel: always `<|start|>assistant to=<rcpt><|message|>`,
/// even when the model's first message arrived header-less. The streaming
/// jail keys on `<|start|>`, so the spelling must be uniform.
pub(crate) fn normalized_header(recipient: &str) -> String {
    format!("{START}assistant to={recipient}{MESSAGE}")
}

/// Detect a complete or partial Muse Glimmer structural marker in a chunk, so
/// the streaming jail starts buffering before markup can leak as content.
pub fn detect_tool_call_start_muse_glimmer(chunk: &str) -> bool {
    const STARTS: [&str; 3] = [START, FUNCTION_CALLS_OPEN, INVOKE_OPEN_PREFIX];
    STARTS.iter().any(|marker| {
        chunk.contains(marker)
            || (1..marker.len())
                .any(|len| marker.is_char_boundary(len) && chunk.ends_with(&marker[..len]))
    })
}

/// End of the first complete jailed span, terminator included. The span
/// extends across CONTIGUOUS complete `<|eom|>`-chained tool channels: an
/// interval-batched backend delta can carry several parallel calls at once,
/// and splitting after the first would leave the trailing complete call
/// buffered past the terminal/usage chunks. Without a terminator the jail
/// keeps accumulating (finalize recovers complete invokes at end of stream).
pub fn find_tool_call_end_position_muse_glimmer(chunk: &str) -> Option<usize> {
    let first_end = [EOM, EOT]
        .iter()
        .filter_map(|terminator| chunk.find(terminator).map(|pos| pos + terminator.len()))
        .min()?;

    let mut end = first_end;
    loop {
        let rest = &chunk[end..];
        if !rest.starts_with(START) {
            break;
        }
        let Some(msg_rel) = rest.find(MESSAGE) else {
            break;
        };
        let (header_start, recipient) = resolve_header(rest, msg_rel);
        let Some(rcpt) = recipient else { break };
        if header_start != 0 || rcpt == REASONING_RECIPIENT || rcpt == USER_RECIPIENT {
            break;
        }
        let body_start = msg_rel + MESSAGE.len();
        let Some(term_end) = [EOM, EOT]
            .iter()
            .filter_map(|t| rest[body_start..].find(t).map(|p| body_start + p + t.len()))
            .min()
        else {
            break;
        };
        end += term_end;
    }
    Some(end)
}

/// Parse a complete Muse Glimmer output (or one span accumulated by the jail).
///
/// Returns `(calls, normal_text)`:
/// - tool-recipient bodies contribute their complete `<atem:invoke>` blocks;
/// - `to=user` and recipient-less bodies contribute their text, unwrapped;
/// - `to=self` bodies contribute nothing (the reasoning parser owns them —
///   same standalone contract as vLLM's tool parser; SGLang instead surfaces
///   them as normal text, but its serving path never hands them to the tool
///   detector either);
/// - framing markers never leak; orphan terminators in unframed text are
///   stripped.
pub fn try_tool_call_parse_muse_glimmer(
    message: &str,
    tools: Option<&[ToolDefinition]>,
) -> anyhow::Result<(Vec<ToolCallResponse>, Option<String>)> {
    let mut calls = Vec::new();
    let mut normal = String::new();
    let mut pos = 0;
    // The turn's first message arrives header-less; after that only a
    // bare-cut reasoning body legitimizes another bare header.
    let mut allow_bare = true;

    while let Some(msg) = next_channel_message(message, pos, allow_bare) {
        // Prose between messages (or before the first header) stays visible.
        push_stripped(&mut normal, &message[pos..msg.header_start]);

        let body = &message[msg.body_start..msg.body_end];
        if msg.is_tool_channel() {
            extract_invokes(body, tools, &mut calls);
        } else if msg.recipient != Some(REASONING_RECIPIENT) {
            normal.push_str(body);
        }
        allow_bare = msg.bare_cut;
        pos = msg.next_pos;
    }

    push_stripped(&mut normal, &message[pos..]);
    Ok((calls, Some(normal)))
}

/// Append text that lives outside any routed message, stripping orphan
/// framing markers so they never reach the client.
pub(crate) fn push_stripped(out: &mut String, text: &str) {
    if text.is_empty() {
        return;
    }
    let mut cleaned = text.to_string();
    for marker in [EOM, EOT, START, MESSAGE] {
        if cleaned.contains(marker) {
            cleaned = cleaned.replace(marker, "");
        }
    }
    out.push_str(&cleaned);
}

/// Extract every complete `<atem:invoke ...>...</atem:invoke>` block from one
/// tool-channel body. A block missing its close is dropped with a warning —
/// both engines require the literal close (truncation mid-call).
fn extract_invokes(
    body: &str,
    tools: Option<&[ToolDefinition]>,
    calls: &mut Vec<ToolCallResponse>,
) {
    let mut cursor = 0;
    while let Some(open) = invoke_open_re().captures(&body[cursor..]) {
        let whole = open.get(0).expect("regex match has group 0");
        let name_raw = open.name("name").expect("regex requires name").as_str();
        let args_start = cursor + whole.end();

        let Some(close_rel) = body[args_start..].find(INVOKE_CLOSE) else {
            tracing::warn!(
                why = "muse_glimmer_truncated_invoke",
                buffered_bytes = body.len() - (cursor + whole.start()),
                "dropping ATEM invoke without a closing </atem:invoke> (truncated tool call?)"
            );
            return;
        };

        let args_body = &body[args_start..args_start + close_rel];
        let mut arguments = serde_json::Map::new();
        for param in parameter_re().captures_iter(args_body) {
            let key = param.name("key").expect("regex requires key").as_str();
            let raw = param.name("value").expect("regex requires value").as_str();
            arguments.insert(key.to_string(), decode_value(raw));
        }

        let name = normalize_name(name_raw, tools);
        let uuid_simple = Uuid::new_v4().simple().to_string();
        calls.push(ToolCallResponse {
            id: format!("call_{}", &uuid_simple[..24]),
            tp: ToolCallType::Function,
            function: CalledFunction {
                name,
                arguments: serde_json::to_string(&arguments).unwrap_or_else(|_| "{}".to_string()),
            },
        });

        cursor = args_start + close_rel + INVOKE_CLOSE.len();
    }
}

/// JSON-decode a parameter value when possible, else keep the raw string —
/// the spec's `value_parser: json` with `allow_non_json: true`. The raw
/// fallback is byte-preserving (NOT trimmed): both engines keep surrounding
/// whitespace in non-JSON string values.
fn decode_value(raw: &str) -> serde_json::Value {
    serde_json::from_str(raw).unwrap_or_else(|_| serde_json::Value::String(raw.to_string()))
}

/// Collapse the chat template's doubled namespace (`get_weather.get_weather`
/// -> `get_weather`) when the collapsed name is a registered tool. Both
/// engines do this. Leaf-only matching (SGLang's extra step) is deliberately
/// NOT done: an emitted `weather.get` would silently dispatch a registered
/// `calendar.get`. Unknown names pass through unchanged with a warning
/// (vLLM's policy; v1 parsers never drop calls by tool name).
fn normalize_name(emitted: &str, tools: Option<&[ToolDefinition]>) -> String {
    let registered: Vec<&str> = tools
        .unwrap_or_default()
        .iter()
        .map(|t| t.name.as_str())
        .collect();
    if registered.is_empty() || registered.contains(&emitted) {
        return emitted.to_string();
    }
    if let Some((head, tail)) = emitted.split_once('.')
        && head == tail
        && registered.contains(&head)
    {
        return head.to_string();
    }
    tracing::warn!(
        emitted_name = emitted,
        "Muse Glimmer: emitted tool name does not match any registered tool; passing through unchanged"
    );
    emitted.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(call: &ToolCallResponse) -> serde_json::Value {
        serde_json::from_str(&call.function.arguments).unwrap()
    }

    fn parse(text: &str) -> (Vec<ToolCallResponse>, String) {
        let (calls, normal) = try_tool_call_parse_muse_glimmer(text, None).unwrap();
        (calls, normal.unwrap())
    }

    fn tool_defs(names: &[&str]) -> Vec<ToolDefinition> {
        names
            .iter()
            .map(|n| ToolDefinition {
                name: n.to_string(),
                parameters: None,
                strict: None,
            })
            .collect()
    }

    const SINGLE_CALL: &str = concat!(
        " to=get_weather<|message|><atem:function_calls>\n",
        "<atem:invoke name=\"get_weather\">\n",
        "<atem:parameter name=\"city\">Paris</atem:parameter>\n",
        "</atem:invoke>\n",
        "</atem:function_calls><|eom|>"
    );

    #[test]
    fn single_call_headerless_first_message() {
        let (calls, normal) = parse(SINGLE_CALL);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(args(&calls[0]), serde_json::json!({"city": "Paris"}));
        assert_eq!(normal, "");
    }

    #[test]
    fn parallel_calls_in_separate_eom_chained_messages() {
        let text = concat!(
            " to=get_weather<|message|><atem:function_calls>\n",
            "<atem:invoke name=\"get_weather\">\n",
            "<atem:parameter name=\"city\">Paris</atem:parameter>\n",
            "</atem:invoke>\n</atem:function_calls><|eom|>",
            "<|start|>assistant to=get_time<|message|><atem:function_calls>\n",
            "<atem:invoke name=\"get_time\">\n",
            "<atem:parameter name=\"timezone\">CET</atem:parameter>\n",
            "</atem:invoke>\n</atem:function_calls><|eom|>"
        );
        let (calls, normal) = parse(text);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[1].function.name, "get_time");
        assert_eq!(args(&calls[1]), serde_json::json!({"timezone": "CET"}));
        assert_eq!(normal, "");
    }

    #[test]
    fn multiple_invokes_in_one_message() {
        let text = concat!(
            " to=tools<|message|><atem:function_calls>\n",
            "<atem:invoke name=\"a\">\n",
            "<atem:parameter name=\"x\">1</atem:parameter>\n",
            "</atem:invoke>\n",
            "<atem:invoke name=\"b\">\n",
            "<atem:parameter name=\"y\">2</atem:parameter>\n",
            "</atem:invoke>\n",
            "</atem:function_calls><|eom|>"
        );
        let (calls, normal) = parse(text);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].function.name, "a");
        assert_eq!(args(&calls[0]), serde_json::json!({"x": 1}));
        assert_eq!(calls[1].function.name, "b");
        assert_eq!(args(&calls[1]), serde_json::json!({"y": 2}));
        assert_eq!(normal, "");
    }

    #[test]
    fn value_types_json_and_raw_fallback() {
        let text = concat!(
            " to=f<|message|><atem:invoke name=\"f\">",
            "<atem:parameter name=\"n\">3</atem:parameter>",
            "<atem:parameter name=\"flag\">true</atem:parameter>",
            "<atem:parameter name=\"none\">null</atem:parameter>",
            "<atem:parameter name=\"obj\">{\"k\": [1, 2]}</atem:parameter>",
            "<atem:parameter name=\"plain\">just words</atem:parameter>",
            "<atem:parameter name=\"jsonish\">{not json}</atem:parameter>",
            "</atem:invoke><|eom|>"
        );
        let (calls, _) = parse(text);
        assert_eq!(
            args(&calls[0]),
            serde_json::json!({
                "n": 3,
                "flag": true,
                "none": null,
                "obj": {"k": [1, 2]},
                "plain": "just words",
                "jsonish": "{not json}",
            })
        );
    }

    #[test]
    fn quoted_json_string_stays_string_and_raw_keeps_whitespace() {
        let text = concat!(
            " to=f<|message|><atem:invoke name=\"f\">",
            "<atem:parameter name=\"quoted\">\"true\"</atem:parameter>",
            "<atem:parameter name=\"padded\"> spaced out </atem:parameter>",
            "</atem:invoke><|eom|>"
        );
        let (calls, _) = parse(text);
        assert_eq!(
            args(&calls[0]),
            serde_json::json!({"quoted": "true", "padded": " spaced out "})
        );
    }

    #[test]
    fn reasoning_channel_never_parses_as_call() {
        let text = concat!(
            " to=self<|message|>I could call <atem:invoke name=\"fake\">",
            "<atem:parameter name=\"x\">1</atem:parameter></atem:invoke> here.<|eom|>",
            "<|start|>assistant to=user<|message|>No tool needed.<|eot|>"
        );
        let (calls, normal) = parse(text);
        assert!(calls.is_empty());
        assert_eq!(normal, "No tool needed.");
    }

    #[test]
    fn atem_in_user_channel_stays_text() {
        let text = concat!(
            " to=user<|message|>Example: <atem:invoke name=\"demo\">",
            "<atem:parameter name=\"x\">1</atem:parameter></atem:invoke><|eot|>"
        );
        let (calls, normal) = parse(text);
        assert!(calls.is_empty());
        assert!(normal.contains("<atem:invoke name=\"demo\">"));
    }

    #[test]
    fn unframed_atem_does_not_parse() {
        let text = concat!(
            "<atem:function_calls><atem:invoke name=\"f\">",
            "<atem:parameter name=\"x\">1</atem:parameter></atem:invoke></atem:function_calls>"
        );
        let (calls, normal) = parse(text);
        assert!(calls.is_empty());
        assert_eq!(normal, text);
    }

    #[test]
    fn plain_text_passthrough() {
        let (calls, normal) = parse("The capital of France is Paris.");
        assert!(calls.is_empty());
        assert_eq!(normal, "The capital of France is Paris.");
    }

    #[test]
    fn empty_and_whitespace_input() {
        let (calls, normal) = parse("");
        assert!(calls.is_empty());
        assert_eq!(normal, "");
        let (calls, normal) = parse("   ");
        assert!(calls.is_empty());
        assert_eq!(normal, "   ");
    }

    #[test]
    fn bare_message_header_is_content() {
        let (calls, normal) = parse("<|start|>assistant<|message|>Hello there.<|eot|>");
        assert!(calls.is_empty());
        assert_eq!(normal, "Hello there.");
    }

    #[test]
    fn reasoning_then_answer() {
        let text = concat!(
            " to=self<|message|>Think first.<|eom|>",
            "<|start|>assistant to=user<|message|>The answer is 4.<|eot|>"
        );
        let (calls, normal) = parse(text);
        assert!(calls.is_empty());
        assert_eq!(normal, "The answer is 4.");
    }

    #[test]
    fn reasoning_then_tool_call() {
        let text = concat!(
            " to=self<|message|>Need the weather.<|eom|>",
            "<|start|>assistant to=get_weather<|message|><atem:function_calls>\n",
            "<atem:invoke name=\"get_weather\">\n",
            "<atem:parameter name=\"city\">Paris</atem:parameter>\n",
            "</atem:invoke>\n</atem:function_calls><|eom|>"
        );
        let (calls, normal) = parse(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(normal, "");
    }

    #[test]
    fn framed_header_inside_user_body_is_a_real_channel_switch() {
        // `<|start|>` is a reserved special token: a framed header mid-answer
        // means the model actually switched channels (the spec's start_anchor),
        // so the tool channel that follows is live. vLLM segments identically;
        // only a BARE `to=` run is treated as quotable prose (see the test
        // above).
        let text = concat!(
            " to=user<|message|>Look: <|start|>assistant to=get_weather<|message|>",
            "<atem:invoke name=\"get_weather\"><atem:parameter name=\"city\">Nice</atem:parameter></atem:invoke><|eom|>"
        );
        let (calls, normal) = parse(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(normal, "Look: ");
    }

    #[test]
    fn quoted_bare_header_in_user_body_is_not_promoted_to_a_call() {
        // A `to=user` answer QUOTING a bare header + ATEM must stay text: the
        // missing-<|eom|> recovery applies to reasoning bodies only, so quoted
        // markup in an answer can never become a live call.
        let text = concat!(
            " to=user<|message|>Example: to=search<|message|>",
            "<atem:invoke name=\"search\"><atem:parameter name=\"q\">oops</atem:parameter></atem:invoke><|eot|>"
        );
        let (calls, normal) = parse(text);
        assert!(calls.is_empty());
        assert!(normal.contains("Example: to=search"));
        assert!(normal.contains("<atem:invoke name=\"search\">"));
    }

    #[test]
    fn concatenated_prose_around_to_is_not_a_recovery_boundary() {
        // `potato=...` must not cut the reasoning body mid-word; the recovery
        // heuristic requires the `to=` to start the body or follow whitespace.
        let text = concat!(
            " to=self<|message|>weird potato=get_weather<|message|>",
            "<atem:invoke name=\"get_weather\"></atem:invoke><|eom|>",
        );
        let (calls, normal) = parse(text);
        assert!(calls.is_empty());
        assert_eq!(normal, "");
    }

    #[test]
    fn missing_eom_before_tool_header_recovers_the_call() {
        // Observed model defect: the reasoning channel ends without <|eom|>
        // and the tool header follows directly.
        let text = concat!(
            " to=self<|message|>thinking to=get_weather<|message|>",
            "<atem:invoke name=\"get_weather\"></atem:invoke><|eom|>"
        );
        let (calls, normal) = parse(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(args(&calls[0]), serde_json::json!({}));
        assert_eq!(normal, "");
    }

    #[test]
    fn truncated_invoke_is_dropped() {
        let text = concat!(
            " to=get_weather<|message|><atem:function_calls>\n",
            "<atem:invoke name=\"get_weather\">\n",
            "<atem:parameter name=\"city\">Par"
        );
        let (calls, normal) = parse(text);
        assert!(calls.is_empty());
        assert_eq!(normal, "");
    }

    #[test]
    fn unterminated_message_with_complete_invoke_recovers() {
        let text = concat!(
            " to=get_weather<|message|><atem:function_calls>\n",
            "<atem:invoke name=\"get_weather\">\n",
            "<atem:parameter name=\"city\">Paris</atem:parameter>\n",
            "</atem:invoke>\n</atem:function_calls>"
        );
        let (calls, _) = parse(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
    }

    #[test]
    fn orphan_terminator_is_stripped_from_prose() {
        let (calls, normal) = parse("some prose<|eom|> more");
        assert!(calls.is_empty());
        assert_eq!(normal, "some prose more");
    }

    #[test]
    fn empty_arguments_object() {
        let text = " to=ping<|message|><atem:invoke name=\"ping\"></atem:invoke><|eom|>";
        let (calls, _) = parse(text);
        assert_eq!(calls[0].function.arguments, "{}");
    }

    #[test]
    fn multiline_and_unicode_values() {
        let text = concat!(
            " to=f<|message|><atem:invoke name=\"f\">",
            "<atem:parameter name=\"text\">line one\nline two — καλημέρα 你好</atem:parameter>",
            "</atem:invoke><|eom|>"
        );
        let (calls, _) = parse(text);
        assert_eq!(
            args(&calls[0]),
            serde_json::json!({"text": "line one\nline two — καλημέρα 你好"})
        );
    }

    #[test]
    fn namespaced_names_pass_through_and_doubled_collapses() {
        let tools = tool_defs(&["get_weather", "web.search"]);
        let text = concat!(
            " to=get_weather.get_weather<|message|>",
            "<atem:invoke name=\"get_weather.get_weather\"></atem:invoke><|eom|>",
            "<|start|>assistant to=web.search<|message|>",
            "<atem:invoke name=\"web.search\"></atem:invoke><|eom|>"
        );
        let (calls, _) = try_tool_call_parse_muse_glimmer(text, Some(&tools)).unwrap();
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[1].function.name, "web.search");
    }

    #[test]
    fn unknown_name_passes_through() {
        let tools = tool_defs(&["calendar.get"]);
        let text =
            " to=weather.get<|message|><atem:invoke name=\"weather.get\"></atem:invoke><|eom|>";
        let (calls, _) = try_tool_call_parse_muse_glimmer(text, Some(&tools)).unwrap();
        // No leaf matching: `weather.get` must not dispatch `calendar.get`.
        assert_eq!(calls[0].function.name, "weather.get");
    }

    #[test]
    fn recipient_and_invoke_name_disagreement_uses_invoke_name() {
        let text = " to=get_weather<|message|><atem:invoke name=\"get_time\"></atem:invoke><|eom|>";
        let (calls, _) = parse(text);
        assert_eq!(calls[0].function.name, "get_time");
    }

    #[test]
    fn eot_terminates_a_tool_channel_too() {
        let text = " to=f<|message|><atem:invoke name=\"f\"></atem:invoke><|eot|>";
        let (calls, _) = parse(text);
        assert_eq!(calls.len(), 1);
    }

    #[test]
    fn detect_matches_complete_and_partial_markers() {
        assert!(detect_tool_call_start_muse_glimmer("<|start|>assistant"));
        assert!(detect_tool_call_start_muse_glimmer(
            "text <atem:function_calls>"
        ));
        assert!(detect_tool_call_start_muse_glimmer("text <atem:inv"));
        assert!(detect_tool_call_start_muse_glimmer("text <|sta"));
        assert!(detect_tool_call_start_muse_glimmer("text <"));
        assert!(!detect_tool_call_start_muse_glimmer("plain text"));
    }

    #[test]
    fn end_position_is_first_terminator() {
        let text = " to=f<|message|>x<|eom|>trailing";
        let end = find_tool_call_end_position_muse_glimmer(text).unwrap();
        assert_eq!(&text[..end], " to=f<|message|>x<|eom|>");
        assert!(find_tool_call_end_position_muse_glimmer(" to=f<|message|>x").is_none());
    }

    #[test]
    fn crlf_and_unicode_survive_in_values_and_recipients() {
        let text = concat!(
            " to=天気.lookup<|message|><atem:invoke name=\"天気.lookup\">",
            "<atem:parameter name=\"città\">Rome\r\nItaly</atem:parameter>",
            "</atem:invoke><|eom|>"
        );
        let (calls, normal) = parse(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "天気.lookup");
        assert_eq!(
            args(&calls[0]),
            serde_json::json!({"città": "Rome\r\nItaly"})
        );
        assert_eq!(normal, "");
    }

    #[test]
    fn earliest_terminator_wins_when_both_present() {
        // <|eot|> before <|eom|>: the user body ends at <|eot|>; the trailing
        // <|eom|> is an orphan and never leaks.
        let (calls, normal) = parse(" to=user<|message|>done<|eot|><|eom|>");
        assert!(calls.is_empty());
        assert_eq!(normal, "done");
    }

    #[test]
    fn duplicate_parameter_keys_last_wins() {
        let text = concat!(
            " to=f<|message|><atem:invoke name=\"f\">",
            "<atem:parameter name=\"x\">1</atem:parameter>",
            "<atem:parameter name=\"x\">2</atem:parameter>",
            "</atem:invoke><|eom|>"
        );
        let (calls, _) = parse(text);
        assert_eq!(args(&calls[0]), serde_json::json!({"x": 2}));
    }

    #[test]
    fn invoke_markup_inside_value_truncates_like_engines() {
        // The regex reads the value to the FIRST close marker, exactly like
        // both engine parsers; markup-as-data truncates rather than escapes.
        let text = concat!(
            " to=f<|message|><atem:invoke name=\"f\">",
            "<atem:parameter name=\"x\">a</atem:parameter>b</atem:parameter>",
            "</atem:invoke><|eom|>"
        );
        let (calls, _) = parse(text);
        assert_eq!(args(&calls[0]), serde_json::json!({"x": "a"}));
    }

    #[test]
    fn prose_ending_in_assistant_before_a_bare_header_stays_prose() {
        // The role word absorbs into the header only behind a real
        // `<|start|>`; "my assistant" is ordinary prose and the separating
        // whitespace is absorbed by the bare-header path in batch and
        // streaming alike.
        let (calls, normal) = parse("my assistant  to=user<|message|>x<|eot|>");
        assert!(calls.is_empty());
        assert_eq!(normal, "my assistantx");
    }

    #[test]
    fn segmentation_resolves_headers() {
        let text = "<|start|>assistant to=self<|message|>a<|eom|>";
        let msg = next_channel_message(text, 0, true).unwrap();
        assert_eq!(msg.recipient, Some("self"));
        assert_eq!(msg.header_start, 0);
        assert!(msg.closed);

        let msg = next_channel_message(" to=user<|message|>hi<|eot|>", 0, true).unwrap();
        assert_eq!(msg.recipient, Some("user"));
        assert_eq!(msg.header_start, 0);

        let msg = next_channel_message("<|message|>hi<|eot|>", 0, true).unwrap();
        assert_eq!(msg.recipient, None);
    }
}
