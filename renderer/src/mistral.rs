// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native Mistral instruct prompt rendering for tekken tokenizers v13 and v15.
//!
//! Mistral checkpoints ship no Jinja template; mistral-common builds the prompt
//! from token ids, inserting structural markers as special tokens and encoding
//! message text as ordinary BPE. This renderer reproduces mistral-common's
//! request normalizer and `InstructTokenizerV13`/`V15` encoders as segments:
//! markers are control segments, and every piece mistral-common encodes in one
//! call is one ordinary segment, so text such as a literal `[INST]` in user
//! content never becomes prompt structure.
//!
//! An image renders as one `[IMG]` control segment. mistral-common expands it
//! to the image's full `[IMG]`/`[IMG_BREAK]`/`[IMG_END]` grid, which depends on
//! the image size; whoever loads the image must do that expansion.

use std::borrow::Cow;
use std::collections::HashMap;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize as _};
use serde_json::{Map, Value};

use crate::python::PyJsonFormatter;
use crate::{
    OAIChatLikeRequest, OAIPromptFormatter, PromptRenderError, RenderedPrompt, RenderedSegment,
};

const BOS: &str = "<s>";
const EOS: &str = "</s>";
const INST: &str = "[INST]";
const INST_END: &str = "[/INST]";
const AVAILABLE_TOOLS: &str = "[AVAILABLE_TOOLS]";
const AVAILABLE_TOOLS_END: &str = "[/AVAILABLE_TOOLS]";
const TOOL_RESULTS: &str = "[TOOL_RESULTS]";
const TOOL_RESULTS_END: &str = "[/TOOL_RESULTS]";
const TOOL_CALLS: &str = "[TOOL_CALLS]";
const ARGS: &str = "[ARGS]";
const SYSTEM_PROMPT: &str = "[SYSTEM_PROMPT]";
const SYSTEM_PROMPT_END: &str = "[/SYSTEM_PROMPT]";
const THINK: &str = "[THINK]";
const THINK_END: &str = "[/THINK]";
const MODEL_SETTINGS: &str = "[MODEL_SETTINGS]";
const MODEL_SETTINGS_END: &str = "[/MODEL_SETTINGS]";
const IMG: &str = "[IMG]";

/// Separator between text from consecutive same-role messages.
const MESSAGE_JOIN: &str = "\n\n";

/// Tekken tokenizer versions this renderer reproduces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MistralTokenizerVersion {
    V13,
    V15,
}

impl MistralTokenizerVersion {
    /// Map a tekken `config.version` number (`13` for `v13`) to a version.
    pub fn from_number(version: u32) -> Option<Self> {
        match version {
            13 => Some(Self::V13),
            15 => Some(Self::V15),
            _ => None,
        }
    }

    /// Separator between text chunks of one message (mistral-common's
    /// `_chunk_join_str`).
    fn chunk_join(self) -> &'static str {
        match self {
            Self::V13 => MESSAGE_JOIN,
            Self::V15 => "",
        }
    }
}

/// Allowed values and default of one enum model setting.
#[derive(Debug, Clone, Deserialize)]
struct EnumSetting {
    accepts_none: bool,
    default: Option<String>,
    values: Vec<String>,
}

/// The tekken file's `model_settings_builder`, which decides the
/// `[MODEL_SETTINGS]` block of v15 prompts.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct ModelSettingsBuilder {
    #[serde(default)]
    reasoning_effort: Option<EnumSetting>,
    /// Present when the model accepts a response JSON schema.
    #[serde(default)]
    json_schema: Option<Value>,
}

#[derive(Debug, Clone)]
pub struct MistralFormatter {
    version: MistralTokenizerVersion,
    settings: Option<ModelSettingsBuilder>,
}

impl MistralFormatter {
    /// Build a formatter from a tekken tokenizer's version number and its
    /// `model_settings_builder` section (required for v15).
    pub fn new(version: u32, model_settings_builder: Option<&Value>) -> Result<Self> {
        let version = MistralTokenizerVersion::from_number(version).with_context(|| {
            format!("Mistral renderer supports tekken v13 and v15, got v{version}")
        })?;
        let settings = match (version, model_settings_builder) {
            (MistralTokenizerVersion::V13, _) => None,
            (MistralTokenizerVersion::V15, Some(builder)) => Some(
                ModelSettingsBuilder::deserialize(builder)
                    .context("Invalid tekken model_settings_builder")?,
            ),
            (MistralTokenizerVersion::V15, None) => {
                anyhow::bail!("tekken v15 tokenizers need a model_settings_builder")
            }
        };
        Ok(Self { version, settings })
    }

    fn build_segments(&self, req: &dyn OAIChatLikeRequest) -> Result<Vec<RenderedSegment>> {
        let messages =
            crate::messages_to_json(req).context("Failed to convert Mistral messages")?;
        crate::reject_unsupported_partial_assistant(&messages)?;
        crate::reject_unsupported_message_tools(&messages, &[])?;
        let Value::Array(messages) = messages else {
            anyhow::bail!("Mistral messages must be an array");
        };
        let messages = messages
            .iter()
            .map(Message::from_openai)
            .collect::<Result<Vec<_>>>()?;
        validate_message_order(&messages)?;
        if req.should_add_generation_prompt()
            && messages.len() > 1
            && messages.last().is_some_and(|m| m.role == Role::Assistant)
        {
            return Err(invalid(
                "the last message is from the assistant; use continue_final_message to continue it",
            ));
        }

        // Prefer the client's tools: the Jinja backfill would add
        // `type`/`properties` that mistral-common does not.
        let tools = req
            .raw_tools()
            .or_else(|| req.tools())
            .map(|tools| serde_json::to_value(&tools))
            .transpose()
            .context("Failed to convert Mistral tools")?;
        let tools_json = render_tools(tools.as_ref())?;
        let settings_json = self.render_settings(req)?;
        let messages = aggregate_messages(messages, self.version);
        // Without a generation prompt, a final assistant turn is continued: it
        // renders as mistral-common's `prefix` message, open and without `</s>`.
        let continued = (!req.should_add_generation_prompt()
            && messages.last().is_some_and(|m| m.role == Role::Assistant))
        .then(|| messages.len() - 1);

        let mut out = Segments::default();
        out.control(BOS);
        let first_user = messages.iter().position(|m| m.role == Role::User);
        for (index, message) in messages.iter().enumerate() {
            match message.role {
                Role::User => {
                    if Some(index) == first_user {
                        if let Some(tools_json) = &tools_json {
                            out.control(AVAILABLE_TOOLS);
                            out.text(tools_json);
                            out.control(AVAILABLE_TOOLS_END);
                        }
                        if let Some(settings_json) = &settings_json {
                            out.control(MODEL_SETTINGS);
                            out.text(settings_json);
                            out.control(MODEL_SETTINGS_END);
                        }
                    }
                    out.control(INST);
                    out.content(&image_first(&message.content));
                    out.control(INST_END);
                }
                Role::System => {
                    if self.version == MistralTokenizerVersion::V15 && message.content.has_think() {
                        return Err(invalid(
                            "thinking content is not supported in system messages",
                        ));
                    }
                    out.control(SYSTEM_PROMPT);
                    out.content(&message.content);
                    out.control(SYSTEM_PROMPT_END);
                }
                Role::Assistant => {
                    if message.content.is_empty() && message.tool_calls.is_empty() {
                        return Err(invalid(
                            "assistant messages need content, reasoning or tool calls",
                        ));
                    }
                    match &message.content {
                        // mistral-common strips trailing spaces from plain string replies.
                        AggregatedContent::Text(text) => out.text(text.trim_end_matches(' ')),
                        chunks => out.content(chunks),
                    }
                    for call in &message.tool_calls {
                        out.control(TOOL_CALLS);
                        out.text(&call.name);
                        out.control(ARGS);
                        out.text(&encoded_arguments(&call.arguments));
                    }
                    if continued != Some(index) {
                        out.control(EOS);
                    }
                }
                Role::Tool => {
                    out.control(TOOL_RESULTS);
                    out.content(&message.content);
                    out.control(TOOL_RESULTS_END);
                }
            }
        }
        Ok(out.0)
    }

    /// Render the `[MODEL_SETTINGS]` JSON, or `None` when no setting applies.
    fn render_settings(&self, req: &dyn OAIChatLikeRequest) -> Result<Option<String>> {
        let Some(builder) = &self.settings else {
            // v13 prompts carry no settings; upstream vLLM ignores both fields.
            return Ok(None);
        };

        // Keys are emitted in sorted order, as mistral-common does.
        let mut settings = Map::new();
        let response_format = req
            .response_format()
            .map(|format| serde_json::to_value(&format))
            .transpose()
            .context("Failed to convert response_format")?;
        // Like mistral-common's `build_settings`, a field the tokenizer has no
        // builder for is dropped rather than rejected.
        if builder.json_schema.is_some()
            && let Some(schema) = response_schema(response_format.as_ref())
        {
            settings.insert("json_schema".to_string(), schema);
        }

        let effort = requested_reasoning_effort(req)?;
        match (&builder.reasoning_effort, effort) {
            (None, _) => {}
            (Some(setting), None) => {
                if !setting.accepts_none {
                    return Err(invalid("reasoning_effort must be set for this model"));
                }
                if let Some(default) = &setting.default {
                    settings.insert("reasoning_effort".to_string(), default.clone().into());
                }
            }
            (Some(setting), Some(effort)) => {
                if !setting.values.contains(&effort) {
                    return Err(invalid(format!(
                        "reasoning_effort must be one of {:?}, got {effort:?}",
                        setting.values
                    )));
                }
                settings.insert("reasoning_effort".to_string(), effort.into());
            }
        }

        Ok((!settings.is_empty()).then(|| python_json(&Value::Object(settings))))
    }
}

impl OAIPromptFormatter for MistralFormatter {
    fn supports_add_generation_prompt(&self) -> bool {
        true
    }

    fn renders_continuation_natively(&self) -> bool {
        true
    }

    fn render(&self, req: &dyn OAIChatLikeRequest) -> Result<String> {
        Ok(self.render_prompt(req)?.into_text())
    }

    fn render_prompt(&self, req: &dyn OAIChatLikeRequest) -> Result<RenderedPrompt> {
        Ok(RenderedPrompt::segmented(self.build_segments(req)?))
    }

    /// Tool results are reordered by call order, and their images with them.
    fn media_message_order(&self, req: &dyn OAIChatLikeRequest) -> Option<Vec<usize>> {
        let Value::Array(messages) = crate::messages_to_json(req).ok()? else {
            return None;
        };
        let tool_has_image = messages.iter().any(|message| {
            message.get("role").and_then(Value::as_str) == Some("tool")
                && message
                    .get("content")
                    .and_then(Value::as_array)
                    .is_some_and(|parts| parts.iter().any(is_image_part))
        });
        if !tool_has_image {
            return None;
        }
        let messages = messages
            .iter()
            .map(Message::from_openai)
            .collect::<Result<Vec<_>>>()
            .ok()?;
        let order = rendered_message_order(&messages);
        order
            .iter()
            .enumerate()
            .any(|(index, &source)| index != source)
            .then_some(order)
    }
}

fn is_image_part(part: &Value) -> bool {
    part.get("type").and_then(Value::as_str) == Some("image_url")
}

/// mistral-common's `_maybe_put_image_first` for user turns: an image that
/// follows only text moves to the front.
fn image_first(content: &AggregatedContent) -> Cow<'_, AggregatedContent> {
    match content {
        AggregatedContent::Chunks(chunks)
            if chunks.len() >= 2
                && matches!(chunks.last(), Some(Chunk::Image))
                && chunks[..chunks.len() - 1]
                    .iter()
                    .all(|chunk| matches!(chunk, Chunk::Text(_))) =>
        {
            let mut reordered = Vec::with_capacity(chunks.len());
            reordered.push(Chunk::Image);
            reordered.extend_from_slice(&chunks[..chunks.len() - 1]);
            Cow::Owned(AggregatedContent::Chunks(reordered))
        }
        _ => Cow::Borrowed(content),
    }
}

fn invalid(message: impl Into<String>) -> anyhow::Error {
    PromptRenderError::invalid_request(message).into()
}

/// `json.dumps(value, ensure_ascii=False)`.
fn python_json(value: &Value) -> String {
    let mut buf = Vec::new();
    let mut serializer = serde_json::Serializer::with_formatter(&mut buf, PyJsonFormatter);
    value
        .serialize(&mut serializer)
        .expect("serializing a JSON value into a Vec cannot fail");
    String::from_utf8(buf).expect("serde_json writes UTF-8")
}

#[derive(Default)]
struct Segments(Vec<RenderedSegment>);

impl Segments {
    fn control(&mut self, token: &str) {
        self.0.push(RenderedSegment::new(token, true));
    }

    /// One ordinary-text encode call; empty text encodes to nothing.
    fn text(&mut self, text: &str) {
        if !text.is_empty() {
            self.0.push(RenderedSegment::new(text, false));
        }
    }

    fn content(&mut self, content: &AggregatedContent) {
        match content {
            AggregatedContent::Text(text) => self.text(text),
            AggregatedContent::Chunks(chunks) => {
                for chunk in chunks {
                    match chunk {
                        Chunk::Text(text) => self.text(text),
                        Chunk::Think { text, closed } => {
                            self.control(THINK);
                            self.text(text);
                            if *closed {
                                self.control(THINK_END);
                            }
                        }
                        Chunk::Image => self.control(IMG),
                    }
                }
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Role {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, PartialEq)]
enum Chunk {
    Text(String),
    Think { text: String, closed: bool },
    Image,
}

#[derive(Debug, Clone, PartialEq)]
enum Content {
    Text(String),
    Chunks(Vec<Chunk>),
}

impl Content {
    /// Python truthiness: `None`, `""` and `[]` are all empty.
    fn is_empty(&self) -> bool {
        match self {
            Content::Text(text) => text.is_empty(),
            Content::Chunks(chunks) => chunks.is_empty(),
        }
    }
}

#[derive(Debug, Clone)]
struct ToolCall {
    id: Option<String>,
    name: String,
    arguments: String,
}

#[derive(Debug, Clone)]
struct Message {
    role: Role,
    content: Option<Content>,
    tool_calls: Vec<ToolCall>,
    tool_call_id: Option<String>,
}

impl Message {
    /// Convert one OpenAI message the way mistral-common's `from_openai` does.
    fn from_openai(message: &Value) -> Result<Self> {
        let role = match message.get("role").and_then(Value::as_str) {
            // OpenAI's `developer` role is treated as system; mistral-common
            // itself rejects the role.
            Some("system" | "developer") => Role::System,
            Some("user") => Role::User,
            Some("assistant") => Role::Assistant,
            Some("tool") => Role::Tool,
            other => return Err(invalid(format!("unsupported message role {other:?}"))),
        };
        let mut content = content_from_openai(message.get("content"), role)?;
        let mut tool_calls = Vec::new();

        if role == Role::Assistant {
            if let Some(calls) = message.get("tool_calls").filter(|calls| !calls.is_null()) {
                let calls = calls
                    .as_array()
                    .ok_or_else(|| invalid("assistant tool_calls must be a list"))?;
                for call in calls {
                    tool_calls.push(tool_call_from_openai(call)?);
                }
            }
            if let Some(reasoning) = reasoning_from_openai(message)? {
                if matches!(&content, Some(Content::Chunks(chunks))
                    if chunks.iter().any(|c| matches!(c, Chunk::Think { .. })))
                {
                    return Err(invalid(
                        "assistant messages cannot carry both thinking chunks and reasoning",
                    ));
                }
                let think = Chunk::Think {
                    text: reasoning,
                    closed: true,
                };
                content = Some(Content::Chunks(match content {
                    None => vec![think],
                    Some(Content::Text(text)) => vec![think, Chunk::Text(text)],
                    Some(Content::Chunks(mut chunks)) => {
                        chunks.insert(0, think);
                        chunks
                    }
                }));
            }
        }
        if role == Role::System && content.is_none() {
            return Err(invalid("system messages need content"));
        }

        let tool_call_id = message
            .get("tool_call_id")
            .and_then(Value::as_str)
            .map(str::to_string);
        Ok(Self {
            role,
            content,
            tool_calls,
            tool_call_id,
        })
    }
}

fn content_from_openai(content: Option<&Value>, role: Role) -> Result<Option<Content>> {
    match content {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(text)) => Ok(Some(Content::Text(text.clone()))),
        Some(Value::Array(parts)) => {
            let mut chunks = Vec::with_capacity(parts.len());
            for part in parts {
                match part.get("type").and_then(Value::as_str) {
                    Some("text") => {
                        let text = part.get("text").and_then(Value::as_str).unwrap_or_default();
                        chunks.push(Chunk::Text(text.to_string()));
                    }
                    Some("thinking") if matches!(role, Role::Assistant | Role::System) => {
                        let text = part
                            .get("thinking")
                            .and_then(Value::as_str)
                            .unwrap_or_default();
                        let closed = part.get("closed").and_then(Value::as_bool).unwrap_or(true);
                        chunks.push(Chunk::Think {
                            text: text.to_string(),
                            closed,
                        });
                    }
                    // mistral-common rejects images in system and assistant messages.
                    Some("image_url") if matches!(role, Role::User | Role::Tool) => {
                        let url = match part.get("image_url") {
                            Some(Value::String(url)) => Some(url.as_str()),
                            Some(image_url) => image_url.get("url").and_then(Value::as_str),
                            None => None,
                        };
                        if url.is_none_or(str::is_empty) {
                            return Err(invalid("image_url parts need a url"));
                        }
                        chunks.push(Chunk::Image);
                    }
                    other => {
                        return Err(invalid(format!(
                            "content part type {other:?} is not supported by the Mistral renderer"
                        )));
                    }
                }
            }
            Ok(Some(Content::Chunks(chunks)))
        }
        Some(other) => Err(invalid(format!(
            "message content must be a string or a list of parts, got {other}"
        ))),
    }
}

fn reasoning_from_openai(message: &Value) -> Result<Option<String>> {
    // Segmented reasoning flattens as the Jinja path does: non-empty segments
    // joined with newlines (`ReasoningContent::to_flat_string`).
    let field = |key: &str| match message.get(key)? {
        Value::String(text) => Some(text.clone()),
        Value::Array(segments) => Some(
            segments
                .iter()
                .filter_map(Value::as_str)
                .filter(|segment| !segment.is_empty())
                .collect::<Vec<_>>()
                .join("\n"),
        ),
        _ => None,
    };
    match (field("reasoning_content"), field("reasoning")) {
        (Some(a), Some(b)) if a != b => Err(invalid(
            "assistant `reasoning_content` and `reasoning` must be equal when both are set",
        )),
        (Some(text), _) | (None, Some(text)) => Ok(Some(text)),
        (None, None) => Ok(None),
    }
}

fn tool_call_from_openai(call: &Value) -> Result<ToolCall> {
    let function = call
        .get("function")
        .ok_or_else(|| invalid("assistant tool calls need a function"))?;
    let name = function
        .get("name")
        .and_then(Value::as_str)
        .ok_or_else(|| invalid("assistant tool calls need a function name"))?;
    let arguments = match function.get("arguments") {
        None | Some(Value::Null) => String::new(),
        Some(Value::String(arguments)) => arguments.clone(),
        Some(other) => python_json(other),
    };
    Ok(ToolCall {
        id: call.get("id").and_then(Value::as_str).map(str::to_string),
        name: name.to_string(),
        arguments,
    })
}

/// mistral-common's message-order rule (`MistralRequestValidator`).
fn validate_message_order(messages: &[Message]) -> Result<()> {
    match messages {
        [] => return Err(invalid("the conversation needs at least one message")),
        [only] if !matches!(only.role, Role::User | Role::System) => {
            return Err(invalid(
                "the conversation must start with a user or system message",
            ));
        }
        _ => {}
    }
    for pair in messages.windows(2) {
        let allowed = match pair[0].role {
            Role::System => matches!(pair[1].role, Role::User | Role::Assistant | Role::System),
            Role::User => matches!(pair[1].role, Role::Assistant | Role::System | Role::User),
            Role::Assistant => matches!(pair[1].role, Role::Assistant | Role::User | Role::Tool),
            Role::Tool => matches!(pair[1].role, Role::Assistant | Role::Tool | Role::User),
        };
        if !allowed {
            return Err(invalid(format!(
                "unexpected {:?} message after a {:?} message",
                pair[1].role, pair[0].role
            )));
        }
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq)]
enum AggregatedContent {
    Text(String),
    Chunks(Vec<Chunk>),
}

impl AggregatedContent {
    fn is_empty(&self) -> bool {
        match self {
            AggregatedContent::Text(text) => text.is_empty(),
            AggregatedContent::Chunks(chunks) => chunks.is_empty(),
        }
    }

    fn has_think(&self) -> bool {
        matches!(self, AggregatedContent::Chunks(chunks)
            if chunks.iter().any(|c| matches!(c, Chunk::Think { .. })))
    }
}

struct AggregatedMessage {
    role: Role,
    content: AggregatedContent,
    tool_calls: Vec<ToolCall>,
    tool_call_id: Option<String>,
}

/// mistral-common's `_aggregate_content_chunks_impl`: coalesce text across
/// messages (joined with `MESSAGE_JOIN`) and within one message (joined with
/// the version's chunk separator); thinking and image chunks stay separate.
fn aggregate_content<'a>(
    contents: impl IntoIterator<Item = Option<&'a Content>>,
    chunk_join: &str,
) -> AggregatedContent {
    let mut chunks = Vec::new();
    let mut text = String::new();
    let mut has_text = false;

    for content in contents {
        let needs_message_join = has_text;
        let Some(content) = content.filter(|content| !content.is_empty()) else {
            continue;
        };
        match content {
            Content::Text(part) => {
                if needs_message_join {
                    text.push_str(MESSAGE_JOIN);
                }
                text.push_str(part);
                has_text = true;
            }
            Content::Chunks(parts) => {
                let mut needs_message_join = needs_message_join;
                for part in parts {
                    match part {
                        Chunk::Text(part) if part.is_empty() => {}
                        Chunk::Text(part) => {
                            if has_text {
                                text.push_str(if needs_message_join {
                                    MESSAGE_JOIN
                                } else {
                                    chunk_join
                                });
                            }
                            text.push_str(part);
                            has_text = true;
                            needs_message_join = false;
                        }
                        other => {
                            if has_text {
                                chunks.push(Chunk::Text(std::mem::take(&mut text)));
                                has_text = false;
                            }
                            chunks.push(other.clone());
                        }
                    }
                }
            }
        }
    }

    if chunks.is_empty() {
        return AggregatedContent::Text(text);
    }
    if has_text {
        chunks.push(Chunk::Text(text));
    }
    AggregatedContent::Chunks(chunks)
}

/// mistral-common's `InstructRequestNormalizerV13/V15._aggregate_messages`.
fn aggregate_messages(
    messages: Vec<Message>,
    version: MistralTokenizerVersion,
) -> Vec<AggregatedMessage> {
    let chunk_join = version.chunk_join();
    let mut out: Vec<AggregatedMessage> = Vec::new();
    let mut latest_call_ids: Vec<String> = Vec::new();

    let mut groups: Vec<Vec<Message>> = Vec::new();
    for message in messages {
        match groups.last_mut() {
            Some(group) if group[0].role == message.role => group.push(message),
            _ => groups.push(vec![message]),
        }
    }

    for group in groups {
        let role = group[0].role;
        match role {
            Role::User => out.push(AggregatedMessage {
                role,
                content: aggregate_content(group.iter().map(|m| m.content.as_ref()), chunk_join),
                tool_calls: Vec::new(),
                tool_call_id: None,
            }),
            Role::Assistant => {
                let content =
                    aggregate_content(group.iter().map(|m| m.content.as_ref()), chunk_join);
                let tool_calls: Vec<ToolCall> = group
                    .into_iter()
                    .flat_map(|m| m.tool_calls)
                    .map(|call| ToolCall {
                        arguments: normalize_arguments(&call.arguments),
                        ..call
                    })
                    .collect();
                latest_call_ids.extend(
                    tool_calls
                        .iter()
                        .map(|call| call.id.clone().unwrap_or_else(|| "null".to_string())),
                );
                out.push(AggregatedMessage {
                    role,
                    content,
                    tool_calls,
                    tool_call_id: None,
                });
            }
            Role::System => {
                for message in group {
                    out.push(AggregatedMessage {
                        role,
                        content: aggregate_content([message.content.as_ref()], chunk_join),
                        tool_calls: Vec::new(),
                        tool_call_id: None,
                    });
                }
            }
            Role::Tool => {
                let mut tool_messages: Vec<AggregatedMessage> = group
                    .into_iter()
                    .map(|message| AggregatedMessage {
                        role,
                        content: aggregate_content([message.content.as_ref()], chunk_join),
                        tool_calls: Vec::new(),
                        tool_call_id: message.tool_call_id,
                    })
                    .collect();
                sort_tool_messages(&mut tool_messages, &latest_call_ids);
                out.extend(tool_messages);
                // Reordering only depends on the latest assistant tool calls.
                latest_call_ids.clear();
            }
        }
    }

    if out.is_empty() {
        out.push(AggregatedMessage {
            role: Role::User,
            content: AggregatedContent::Text(String::new()),
            tool_calls: Vec::new(),
            tool_call_id: None,
        });
    }
    out
}

/// Order tool results by the latest assistant call order, then by arrival
/// (mistral-common's `_inplace_sort_tool_messages`; later duplicates win).
fn sort_tool_messages(messages: &mut Vec<AggregatedMessage>, latest_call_ids: &[String]) {
    let keys = {
        let ids: Vec<Option<&str>> = messages
            .iter()
            .map(|message| message.tool_call_id.as_deref())
            .collect();
        tool_sort_keys(&ids, latest_call_ids)
    };
    let mut keyed: Vec<_> = keys.into_iter().zip(std::mem::take(messages)).collect();
    keyed.sort_by_key(|(key, _)| *key);
    messages.extend(keyed.into_iter().map(|(_, message)| message));
}

/// Sort keys of one run of tool results: the call's position among the
/// latest assistant tool calls, then the last arrival of its call id.
fn tool_sort_keys(ids: &[Option<&str>], latest_call_ids: &[String]) -> Vec<(usize, usize)> {
    let call_index: HashMap<&str, usize> = latest_call_ids
        .iter()
        .enumerate()
        .map(|(index, id)| (id.as_str(), index))
        .collect();
    let result_index: HashMap<Option<&str>, usize> = ids
        .iter()
        .enumerate()
        .map(|(index, id)| (*id, index))
        .collect();
    ids.iter()
        .map(|id| {
            (
                call_index
                    .get(id.unwrap_or("null"))
                    .copied()
                    .unwrap_or(usize::MAX),
                result_index[id],
            )
        })
        .collect()
}

/// Source message indices in rendered order, mirroring `aggregate_messages`:
/// only tool results move.
fn rendered_message_order(messages: &[Message]) -> Vec<usize> {
    let mut order = Vec::with_capacity(messages.len());
    let mut latest_call_ids: Vec<String> = Vec::new();
    let mut start = 0;
    while start < messages.len() {
        let role = messages[start].role;
        let end = messages[start..]
            .iter()
            .position(|message| message.role != role)
            .map_or(messages.len(), |len| start + len);
        let run = &messages[start..end];
        match role {
            Role::Assistant => {
                latest_call_ids.extend(
                    run.iter()
                        .flat_map(|message| &message.tool_calls)
                        .map(|call| call.id.clone().unwrap_or_else(|| "null".to_string())),
                );
                order.extend(start..end);
            }
            Role::Tool => {
                let ids: Vec<Option<&str>> = run
                    .iter()
                    .map(|message| message.tool_call_id.as_deref())
                    .collect();
                let mut keyed: Vec<_> = tool_sort_keys(&ids, &latest_call_ids)
                    .into_iter()
                    .zip(start..end)
                    .collect();
                keyed.sort_by_key(|(key, _)| *key);
                order.extend(keyed.into_iter().map(|(_, index)| index));
                latest_call_ids.clear();
            }
            Role::User | Role::System => order.extend(start..end),
        }
        start = end;
    }
    order
}

/// mistral-common's `_normalize_json_content` for tool-call arguments.
fn normalize_arguments(arguments: &str) -> String {
    if arguments.is_empty() {
        return "{}".to_string();
    }
    match serde_json::from_str::<Value>(arguments) {
        Ok(value) => python_json(&value),
        Err(_) => arguments.to_string(),
    }
}

/// The argument text the V13 encoder emits: parsed JSON re-dumped, or an
/// unparseable string dumped as a JSON string.
fn encoded_arguments(arguments: &str) -> String {
    match serde_json::from_str::<Value>(arguments) {
        Ok(value) => python_json(&value),
        Err(_) => python_json(&Value::String(arguments.to_string())),
    }
}

/// `json.dumps` of the tools as mistral-common's `Tool.model_dump` (without
/// `strict`), or `None` when there are no tools.
fn render_tools(tools: Option<&Value>) -> Result<Option<String>> {
    let Some(tools) = tools
        .and_then(Value::as_array)
        .filter(|tools| !tools.is_empty())
    else {
        return Ok(None);
    };
    let mut rendered = Vec::with_capacity(tools.len());
    for tool in tools {
        let function = tool
            .get("function")
            .ok_or_else(|| invalid("tools must be function tools"))?;
        let name = function
            .get("name")
            .and_then(Value::as_str)
            .ok_or_else(|| invalid("tool functions need a name"))?;
        let description = function
            .get("description")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let parameters = function
            .get("parameters")
            .filter(|parameters| !parameters.is_null())
            .cloned()
            .unwrap_or_else(|| Value::Object(Map::new()));
        let mut function = Map::new();
        function.insert("name".to_string(), name.into());
        function.insert("description".to_string(), description.into());
        function.insert("parameters".to_string(), parameters);
        let mut tool = Map::new();
        tool.insert("type".to_string(), "function".into());
        tool.insert("function".to_string(), Value::Object(function));
        rendered.push(Value::Object(tool));
    }
    Ok(Some(python_json(&Value::Array(rendered))))
}

/// The JSON schema a `response_format` asks for, following upstream vLLM's
/// `to_mistral_response_format`: `json_object` allows any object or array,
/// `json_schema` uses its schema, everything else adds nothing.
fn response_schema(response_format: Option<&Value>) -> Option<Value> {
    let response_format = response_format?;
    match response_format.get("type").and_then(Value::as_str)? {
        "json_object" => Some(serde_json::json!({
            "anyOf": [{"type": "object"}, {"type": "array"}]
        })),
        "json_schema" => response_format
            .get("json_schema")
            .and_then(|schema| schema.get("schema"))
            .filter(|schema| !schema.is_null())
            .cloned(),
        _ => None,
    }
}

/// `reasoning_effort` from `chat_template_args` (where Dynamo normalizes the
/// top-level field) or from the request's own top-level field.
fn requested_reasoning_effort(req: &dyn OAIChatLikeRequest) -> Result<Option<String>> {
    let value = match req
        .chat_template_args()
        .and_then(|args| args.get("reasoning_effort"))
    {
        Some(value) => Some(value.clone()),
        None => req
            .reasoning_effort()
            .map(|value| serde_json::to_value(&value))
            .transpose()
            .context("Failed to convert reasoning_effort")?,
    };
    match value {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(effort)) => Ok(Some(effort)),
        Some(other) => Err(invalid(format!(
            "reasoning_effort must be a string, got {other}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use minijinja::value::Value as MiniValue;
    use serde_json::json;

    struct Request {
        messages: Value,
        tools: Option<Value>,
        raw_tools: Option<Value>,
        response_format: Option<Value>,
        args: HashMap<String, Value>,
        add_generation_prompt: bool,
    }

    impl Request {
        fn new(messages: Value) -> Self {
            Self {
                messages,
                tools: None,
                raw_tools: None,
                response_format: None,
                args: HashMap::new(),
                add_generation_prompt: true,
            }
        }
    }

    impl OAIChatLikeRequest for Request {
        fn model(&self) -> String {
            "mistral".to_string()
        }

        fn messages(&self) -> MiniValue {
            MiniValue::from_serialize(&self.messages)
        }

        fn tools(&self) -> Option<MiniValue> {
            self.tools.as_ref().map(MiniValue::from_serialize)
        }

        fn raw_tools(&self) -> Option<MiniValue> {
            self.raw_tools.as_ref().map(MiniValue::from_serialize)
        }

        fn response_format(&self) -> Option<MiniValue> {
            self.response_format.as_ref().map(MiniValue::from_serialize)
        }

        fn should_add_generation_prompt(&self) -> bool {
            self.add_generation_prompt
        }

        fn chat_template_args(&self) -> Option<&HashMap<String, Value>> {
            Some(&self.args)
        }
    }

    fn settings_builder() -> Value {
        json!({
            "reasoning_effort": {"type": "enum", "accepts_none": true, "default": "none", "values": ["none", "high"]},
            "json_schema": {"type": "json_schema", "accepts_none": false, "default": null}
        })
    }

    fn v15() -> MistralFormatter {
        MistralFormatter::new(15, Some(&settings_builder())).unwrap()
    }

    fn math_tool() -> Value {
        json!([{
            "type": "function",
            "function": {
                "name": "math_interpreter",
                "description": "Get the value of an arithmetic expression.",
                "parameters": {"type": "object", "properties": {"expression": {"type": "string", "description": "Math expression."}}}
            }
        }])
    }

    /// mistral-common `tests/test_tokenizer_v15.py::EXPECTED_TEXT_V15`.
    #[test]
    fn matches_mistral_common_v15_golden_conversation() {
        let mut request = Request::new(json!([
            {"role": "system", "content": "S"},
            {"role": "user", "content": "U1"},
            {"role": "assistant", "content": "A1", "tool_calls": [
                {"id": "123456789", "type": "function", "function": {"name": "F1", "arguments": "{}"}},
                {"id": "999999999", "type": "function", "function": {"name": "F2", "arguments": "{}"}}
            ]},
            {"role": "tool", "tool_call_id": "123456789", "content": "R1"},
            {"role": "tool", "tool_call_id": "999999999", "content": "R2"},
            {"role": "assistant", "content": "A2"},
            {"role": "user", "content": "U2"}
        ]));
        request.tools = Some(math_tool());
        request
            .args
            .insert("reasoning_effort".to_string(), json!("high"));

        let rendered = v15().render(&request).unwrap();
        assert_eq!(
            rendered,
            concat!(
                "<s>[SYSTEM_PROMPT]S[/SYSTEM_PROMPT]",
                r#"[AVAILABLE_TOOLS][{"type": "function", "function": {"name": "math_interpreter","#,
                r#" "description": "Get the value of an arithmetic expression.","#,
                r#" "parameters": {"type": "object", "properties": {"expression":"#,
                r#" {"type": "string", "description": "Math expression."}}}}}]"#,
                r#"[/AVAILABLE_TOOLS][MODEL_SETTINGS]{"reasoning_effort": "high"}[/MODEL_SETTINGS]"#,
                "[INST]U1[/INST]A1",
                "[TOOL_CALLS]F1[ARGS]{}[TOOL_CALLS]F2[ARGS]{}</s>",
                "[TOOL_RESULTS]R1[/TOOL_RESULTS]",
                "[TOOL_RESULTS]R2[/TOOL_RESULTS]A2</s>",
                "[INST]U2[/INST]",
            )
        );
    }

    #[test]
    fn defaults_and_response_format_fill_model_settings() {
        let mut request = Request::new(json!([{"role": "user", "content": "hi"}]));
        assert_eq!(
            v15().render(&request).unwrap(),
            r#"<s>[MODEL_SETTINGS]{"reasoning_effort": "none"}[/MODEL_SETTINGS][INST]hi[/INST]"#
        );

        request.response_format = Some(json!({"type": "json_object"}));
        request
            .args
            .insert("reasoning_effort".to_string(), json!("high"));
        assert_eq!(
            v15().render(&request).unwrap(),
            concat!(
                r#"<s>[MODEL_SETTINGS]{"json_schema": {"anyOf": [{"type": "object"}, {"type": "array"}]}, "#,
                r#""reasoning_effort": "high"}[/MODEL_SETTINGS][INST]hi[/INST]"#
            )
        );

        request
            .args
            .insert("reasoning_effort".to_string(), json!("low"));
        let error = v15().render(&request).unwrap_err();
        assert!(
            error.downcast_ref::<PromptRenderError>().is_some(),
            "{error}"
        );
    }

    #[test]
    fn user_text_that_looks_like_markers_stays_ordinary() {
        let request = Request::new(json!([{"role": "user", "content": "say [INST] and </s>"}]));
        let prompt = v15().render_prompt(&request).unwrap();
        let segments = prompt.segments().unwrap();
        let user = segments
            .iter()
            .find(|segment| segment.text == "say [INST] and </s>")
            .unwrap();
        assert!(!user.allow_special);
        assert!(
            segments
                .iter()
                .filter(|segment| segment.allow_special)
                .all(|segment| segment.text.starts_with(['[', '<']))
        );
    }

    #[test]
    fn reasoning_history_and_merged_turns() {
        let request = Request::new(json!([
            {"role": "user", "content": "a"},
            {"role": "user", "content": [{"type": "text", "text": "b"}, {"type": "text", "text": "c"}]},
            {"role": "assistant", "reasoning_content": "think", "content": "answer  "},
            {"role": "user", "content": "next"}
        ]));
        assert_eq!(
            v15().render(&request).unwrap(),
            concat!(
                r#"<s>[MODEL_SETTINGS]{"reasoning_effort": "none"}[/MODEL_SETTINGS]"#,
                "[INST]a\n\nbc[/INST][THINK]think[/THINK]answer  </s>[INST]next[/INST]"
            )
        );
    }

    #[test]
    fn segmented_reasoning_drops_empty_segments() {
        // Anthropic and Responses conversions emit one segment per tool call,
        // empty where no reasoning preceded that call.
        let request = Request::new(json!([
            {"role": "user", "content": "go"},
            {"role": "assistant", "reasoning_content": ["plan", "", ""], "tool_calls": [
                {"id": "aaaaaaaaa", "type": "function", "function": {"name": "f", "arguments": "{}"}},
                {"id": "bbbbbbbbb", "type": "function", "function": {"name": "g", "arguments": "{}"}}
            ]},
            {"role": "tool", "tool_call_id": "aaaaaaaaa", "content": "A"},
            {"role": "tool", "tool_call_id": "bbbbbbbbb", "content": "B"}
        ]));
        let rendered = v15().render(&request).unwrap();
        assert!(
            rendered.contains("[INST]go[/INST][THINK]plan[/THINK][TOOL_CALLS]f"),
            "{rendered}"
        );
    }

    #[test]
    fn tool_results_follow_call_order_and_arguments_normalize() {
        let request = Request::new(json!([
            {"role": "user", "content": "go"},
            {"role": "assistant", "tool_calls": [
                {"id": "aaaaaaaaa", "type": "function", "function": {"name": "f", "arguments": "{\"x\":1,  \"y\":[2]}"}},
                {"id": "bbbbbbbbb", "type": "function", "function": {"name": "g", "arguments": ""}},
                {"id": "ccccccccc", "type": "function", "function": {"name": "h", "arguments": "not json"}}
            ]},
            {"role": "tool", "tool_call_id": "bbbbbbbbb", "content": "B"},
            {"role": "tool", "tool_call_id": "aaaaaaaaa", "content": "A"}
        ]));
        assert_eq!(
            v15().render(&request).unwrap(),
            concat!(
                r#"<s>[MODEL_SETTINGS]{"reasoning_effort": "none"}[/MODEL_SETTINGS][INST]go[/INST]"#,
                r#"[TOOL_CALLS]f[ARGS]{"x": 1, "y": [2]}[TOOL_CALLS]g[ARGS]{}"#,
                r#"[TOOL_CALLS]h[ARGS]"not json"</s>"#,
                "[TOOL_RESULTS]A[/TOOL_RESULTS][TOOL_RESULTS]B[/TOOL_RESULTS]",
            )
        );
    }

    #[test]
    fn v13_has_no_settings_and_joins_chunks_with_blank_lines() {
        let formatter = MistralFormatter::new(13, None).unwrap();
        let mut request = Request::new(json!([
            {"role": "user", "content": [{"type": "text", "text": "b"}, {"type": "text", "text": "c"}]}
        ]));
        request
            .args
            .insert("reasoning_effort".to_string(), json!("high"));
        assert_eq!(
            formatter.render(&request).unwrap(),
            "<s>[INST]b\n\nc[/INST]"
        );
    }

    #[test]
    fn rejects_invalid_conversations() {
        let cases = [
            json!([]),
            json!([{"role": "assistant", "content": "hi"}]),
            json!([{"role": "user", "content": "hi"}, {"role": "tool", "tool_call_id": "x", "content": "r"}]),
            json!([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]),
            json!([{"role": "system", "content": [{"type": "image_url", "image_url": {"url": "x"}}]},
                {"role": "user", "content": "hi"}]),
            json!([{"role": "user", "content": "hi"},
                {"role": "assistant", "content": [{"type": "image_url", "image_url": {"url": "x"}}]},
                {"role": "user", "content": "hi"}]),
            json!([{"role": "user", "content": [{"type": "image_url", "image_url": {}}]}]),
        ];
        for messages in cases {
            let error = v15().render(&Request::new(messages.clone())).unwrap_err();
            assert!(
                error.downcast_ref::<PromptRenderError>().is_some(),
                "{messages}: {error}"
            );
        }
    }

    fn image(url: &str) -> Value {
        json!({"type": "image_url", "image_url": {"url": url}})
    }

    fn text(text: &str) -> Value {
        json!({"type": "text", "text": text})
    }

    /// Placement follows mistral-common's normalizer and `_maybe_put_image_first`.
    #[test]
    fn images_render_as_control_placeholders() {
        let user = |content: Value| {
            let rendered = v15()
                .render(&Request::new(json!([{"role": "user", "content": content}])))
                .unwrap();
            rendered
                .strip_prefix(r#"<s>[MODEL_SETTINGS]{"reasoning_effort": "none"}[/MODEL_SETTINGS]"#)
                .unwrap()
                .to_string()
        };
        // An image after only text moves first.
        assert_eq!(
            user(json!([text("What is "), text("this?"), image("a")])),
            "[INST][IMG]What is this?[/INST]"
        );
        assert_eq!(
            user(json!([image("a"), text("What is this?")])),
            "[INST][IMG]What is this?[/INST]"
        );
        // Otherwise images stay in place and split the text.
        assert_eq!(
            user(json!([text("x"), image("a"), image("b")])),
            "[INST]x[IMG][IMG][/INST]"
        );
        assert_eq!(
            user(json!([text("x"), image("a"), text("y")])),
            "[INST]x[IMG]y[/INST]"
        );
        assert_eq!(
            user(json!([{"type": "image_url", "image_url": "data:image/png;base64,AA"}])),
            "[INST][IMG][/INST]"
        );

        let request =
            Request::new(json!([{"role": "user", "content": [text("[IMG]"), image("a")]}]));
        let prompt = v15().render_prompt(&request).unwrap();
        let segments = prompt.segments().unwrap();
        let placeholders: Vec<_> = segments
            .iter()
            .filter(|segment| segment.text == IMG)
            .map(|segment| segment.allow_special)
            .collect();
        // The literal user text stays ordinary; only the image is a control token.
        assert_eq!(placeholders, [true, false]);
    }

    #[test]
    fn merged_user_turns_put_a_trailing_image_first() {
        let request = Request::new(json!([
            {"role": "user", "content": "a"},
            {"role": "user", "content": [image("x")]}
        ]));
        assert!(
            v15()
                .render(&request)
                .unwrap()
                .ends_with("[INST][IMG]a[/INST]"),
            "{}",
            v15().render(&request).unwrap()
        );
        assert_eq!(v15().media_message_order(&request), None);
    }

    #[test]
    fn tool_result_images_follow_the_reordered_results() {
        let request = Request::new(json!([
            {"role": "user", "content": "go"},
            {"role": "assistant", "tool_calls": [
                {"id": "aaaaaaaaa", "type": "function", "function": {"name": "f", "arguments": "{}"}},
                {"id": "bbbbbbbbb", "type": "function", "function": {"name": "g", "arguments": "{}"}}
            ]},
            {"role": "tool", "tool_call_id": "bbbbbbbbb", "content": [text("B"), image("b")]},
            {"role": "tool", "tool_call_id": "aaaaaaaaa", "content": [image("a"), text("A")]}
        ]));
        assert!(
            v15().render(&request).unwrap().ends_with(concat!(
                "[TOOL_RESULTS][IMG]A[/TOOL_RESULTS]",
                "[TOOL_RESULTS]B[IMG][/TOOL_RESULTS]"
            )),
            "{}",
            v15().render(&request).unwrap()
        );
        assert_eq!(v15().media_message_order(&request), Some(vec![0, 1, 3, 2]));

        // Results already in call order keep arrival order.
        let in_order = Request::new(json!([
            {"role": "user", "content": "go"},
            {"role": "assistant", "tool_calls": [
                {"id": "aaaaaaaaa", "type": "function", "function": {"name": "f", "arguments": "{}"}}
            ]},
            {"role": "tool", "tool_call_id": "aaaaaaaaa", "content": [image("a")]}
        ]));
        assert_eq!(v15().media_message_order(&in_order), None);
    }

    #[test]
    fn continued_final_assistant_turn_is_an_open_prefix() {
        let mut request = Request::new(json!([
            {"role": "user", "content": "Give JSON"},
            {"role": "assistant", "content": "```json\n  "}
        ]));
        request.add_generation_prompt = false;
        // mistral-common `prefix=True`: trailing spaces stripped, newline kept, no `</s>`.
        let rendered = v15().render(&request).unwrap();
        assert!(
            rendered.ends_with("[INST]Give JSON[/INST]```json\n"),
            "{rendered}"
        );
    }

    #[test]
    fn client_tools_win_over_the_jinja_backfill() {
        let mut request = Request::new(json!([{"role": "user", "content": "now?"}]));
        request.raw_tools =
            Some(json!([{"type": "function", "function": {"name": "now", "parameters": {}}}]));
        request.tools = Some(json!([{"type": "function", "function": {
            "name": "now", "description": "", "parameters": {"type": "object", "properties": {}}
        }}]));
        let rendered = v15().render(&request).unwrap();
        assert!(
            rendered.contains(r#"{"name": "now", "description": "", "parameters": {}}"#),
            "{rendered}"
        );
    }

    #[test]
    fn rejects_unsupported_versions_and_missing_v15_builder() {
        assert!(MistralFormatter::new(11, None).is_err());
        assert!(MistralFormatter::new(15, None).is_err());
        assert!(MistralFormatter::new(15, Some(&json!({"unknown": {}}))).is_err());
    }
}
