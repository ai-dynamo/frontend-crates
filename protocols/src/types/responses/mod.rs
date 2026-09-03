// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Dynamo owns the Responses-API input-side type chain. Upstream async-openai
// is the source for everything else (output-side types, streaming events,
// individual tool-call payloads, etc.).
//
// The input chain is owned because upstream marks fields as required that
// real-world clients (OpenAI Agents SDK, Codex, etc.) routinely omit when
// round-tripping a prior assistant turn as input:
//   - `OutputMessage.id` / `.status` — omitted when echoing a previous output
//   - `OutputTextContent.annotations` — omitted when the part carried none
//   - `ReasoningItem.id` — omitted by Codex/OpenCode/agent SDKs on echo
// Upstream is slow to relax these (the sibling `ReasoningItem.id` fix landed in
// 64bit/async-openai#535, but after our pinned async-openai, so we mirror it
// locally as `InputReasoningItem`); OpenAI's own hosted API accepts the relaxed
// shapes on input regardless.
//
// This mirrors the pattern in `crate::types::chat` where Dynamo owns the
// request types it needs to extend or relax while re-exporting the rest of
// upstream's type library verbatim.
//
// Naming: the relaxed assistant-input message is `InputOutputMessage` (and
// `InputOutputMessageContent` / `InputOutputTextContent` for its content
// parts) to avoid colliding with upstream's `OutputMessage`, which remains the
// canonical type for *output-side* response construction (`OutputItem`,
// `Response.output`). `MessageItem`, `Item`, `InputItem`, `InputParam`, and
// `CreateResponse` are input-only and shadow upstream's same-named types
// without conflict.

use std::collections::HashMap;

use serde::{Deserialize, Serialize, de};

// Re-export all upstream response types (shared structures like ResponseUsage,
// tool-call item types, streaming events, etc.). The types we own below
// shadow their upstream counterparts where no dual-side conflict exists.
pub use async_openai::types::responses::*;

// Re-export upstream's pre-shadow `InputContent` under an explicit alias.
// Needed because `FunctionCallOutput::Content` and `EasyInputContent::ContentList`
// are non-owned upstream types that carry upstream's original `InputContent`
// inline, so downstream consumers occasionally need to name it alongside the
// Dynamo-owned shadow defined further down this module.
pub use async_openai::types::responses::InputContent as UpstreamInputContent;

// Re-export from parent module for backward compat.
pub use crate::types::ImageDetail;
pub use crate::types::ReasoningEffort;
pub use crate::types::ResponseFormatJsonSchema;

// Backward-compatible type aliases for Dynamo consumer code migration.
pub type Input = InputParam;
pub type PromptConfig = Prompt;
pub type TextConfig = ResponseTextParam;
pub type TextResponseFormat = TextResponseFormatConfiguration;

/// Stream of response events.
pub type ResponseStream = std::pin::Pin<
    Box<dyn futures::Stream<Item = Result<ResponseStreamEvent, crate::error::OpenAIError>> + Send>,
>;

/// Fields on upstream `Response` that the OpenResponses spec requires as
/// `T | null` but async-openai declares as `Option<T>` with
/// `skip_serializing_if = Option::is_none` — meaning `None` disappears from
/// the wire shape, where the spec wants an explicit `null`.
///
/// Colocated here (next to the upstream `Response` re-export) rather than in
/// `lib/llm/src/protocols/openai/responses/mod.rs` so that when upstream's
/// `Response` gains a new nullable-required field, the reviewer editing this
/// module is looking directly at the authoritative list. Keep sorted
/// alphabetically; entries must match serde field names on `Response` exactly.
///
/// Any field we unconditionally populate ourselves during response
/// construction (e.g. `metadata`, `parallel_tool_calls`, `temperature`,
/// `text`, `tool_choice`, `tools`, `top_p`, `top_logprobs`, `truncation`,
/// `service_tier`, `background`) is deliberately absent — it's always
/// present on the wire, so listing it here would be noise.
pub const SPEC_NULLABLE_REQUIRED_RESPONSE_FIELDS: &[&str] = &[
    "billing",
    "completed_at",
    "conversation",
    "error",
    "incomplete_details",
    "instructions",
    "max_output_tokens",
    "max_tool_calls",
    "previous_response_id",
    "prompt",
    "prompt_cache_key",
    "prompt_cache_retention",
    "reasoning",
    "safety_identifier",
    "usage",
];

// ---------------------------------------------------------------------------
// Input-side assistant message (relaxed vs upstream OutputMessage)
// ---------------------------------------------------------------------------

/// Deserialize `null` or a missing field as the default empty `Vec`. Plain
/// `#[serde(default)]` only fires when the field is absent; explicit `null`
/// would otherwise fail `Vec::deserialize`. Clients (notably some Agents SDK
/// variants) have been observed to send `"annotations": null`, so treat
/// omission and explicit null the same.
fn deserialize_null_as_empty_vec<'de, T, D>(deserializer: D) -> Result<Vec<T>, D::Error>
where
    T: Deserialize<'de>,
    D: serde::Deserializer<'de>,
{
    Option::<Vec<T>>::deserialize(deserializer).map(Option::unwrap_or_default)
}

/// Deserialize `null` or a missing field as `T::default()`. Scalar counterpart
/// to `deserialize_null_as_empty_vec` — plain `#[serde(default)]` rejects
/// explicit `null` because serde tries to deserialize the null into `T` and
/// fails. Real clients emit `null` for unset enum-ish fields (e.g. OpenAI
/// Agents SDK sending `"detail": null` on `input_image` parts).
fn deserialize_null_as_default<'de, T, D>(deserializer: D) -> Result<T, D::Error>
where
    T: Deserialize<'de> + Default,
    D: serde::Deserializer<'de>,
{
    Option::<T>::deserialize(deserializer).map(Option::unwrap_or_default)
}

/// Deserialize `tool_choice`, coercing the object form `{"type": "auto" |
/// "none" | "required", ...}` into the upstream `Mode` variant.
///
/// Upstream `ToolChoiceParam` only accepts `auto`/`none`/`required` as a bare
/// string; the object form is reserved for naming a *specific* tool
/// (`{"type": "function", "name": ...}`). But Anthropic-style clients (and
/// litellm forwarding them verbatim) express the mode as an object, e.g.
/// `{"type": "auto", "disable_parallel_tool_use": true}`. OpenAI's hosted API
/// treats `{"type": "auto"}` and the bare `"auto"` identically; we do the same.
/// Extra keys (e.g. `disable_parallel_tool_use`) are accepted and ignored —
/// there is no per-call parallel-tool-use toggle to honor.
///
/// Any value that is not a mode-typed object falls through to standard
/// `ToolChoiceParam` deserialization, so bare strings and specific-tool /
/// hosted-tool objects keep working unchanged.
fn deserialize_tool_choice<'de, D>(deserializer: D) -> Result<Option<ToolChoiceParam>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let Some(value) = Option::<serde_json::Value>::deserialize(deserializer)? else {
        return Ok(None);
    };
    if let Some(serde_json::Value::String(t)) = value.get("type") {
        let mode = match t.as_str() {
            "auto" => Some(ToolChoiceOptions::Auto),
            "none" => Some(ToolChoiceOptions::None),
            "required" => Some(ToolChoiceOptions::Required),
            _ => None,
        };
        if let Some(mode) = mode {
            return Ok(Some(ToolChoiceParam::Mode(mode)));
        }
    }
    ToolChoiceParam::deserialize(value)
        .map(Some)
        .map_err(serde::de::Error::custom)
}

/// Relaxed counterpart to upstream `OutputTextContent` for input-side content.
/// `annotations` tolerates both missing and explicit `null`; upstream requires
/// it to be a present non-null array.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct InputOutputTextContent {
    #[serde(default, deserialize_with = "deserialize_null_as_empty_vec")]
    pub annotations: Vec<Annotation>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<Vec<LogProb>>,
    pub text: String,
}

/// Content parts of a prior assistant message presented as input.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum InputOutputMessageContent {
    OutputText(InputOutputTextContent),
    Refusal(RefusalContent),
}

/// An assistant message echoed back as input for a subsequent turn. Relaxed
/// compared to upstream `OutputMessage`: `id`, `status`, and `content` are all
/// optional. Some clients send a bare assistant shell (`{"type":"message",
/// "role":"assistant"}`) with no `content` at all, usually on pure tool-call
/// turns; treat absent `content` as an empty vec, same way we treat a missing
/// `id`/`status`.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct InputOutputMessage {
    #[serde(default, deserialize_with = "deserialize_null_as_empty_vec")]
    pub content: Vec<InputOutputMessageContent>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub role: AssistantRole,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub phase: Option<MessagePhase>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<OutputStatus>,
}

// ---------------------------------------------------------------------------
// Input-side image / content / message (shadow upstream, relaxed shapes)
// ---------------------------------------------------------------------------

/// Relaxed counterpart to upstream `InputImageContent`. `detail` defaults to
/// `ImageDetail::Auto` when the client omits it — OpenAI's hosted API and the
/// OpenResponses spec both accept this shape, but upstream's struct marks
/// `detail` as required.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct InputImageContent {
    #[serde(default, deserialize_with = "deserialize_null_as_default")]
    pub detail: ImageDetail,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image_url: Option<String>,
}

/// Parts of an input message: text, image, or file. Mirrors upstream
/// `InputContent` but routes `InputImage` through the Dynamo-owned relaxed
/// `InputImageContent` above.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum InputContent {
    InputText(InputTextContent),
    InputImage(InputImageContent),
    InputFile(InputFileContent),
}

/// User / system / developer input message. Shadows upstream `InputMessage`
/// so we can route through the Dynamo-owned `InputContent` chain.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Default)]
pub struct InputMessage {
    pub content: Vec<InputContent>,
    pub role: InputRole,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<OutputStatus>,
}

/// Content for `EasyInputMessage`. Shadows upstream's same-named enum so the
/// `ContentList` arm carries Dynamo's relaxed `InputContent` (with optional
/// `detail` on `InputImageContent`) instead of upstream's strict variant.
///
/// Without this shadow, the `InputItem::EasyMessage` fallback in the untagged
/// `InputItem` enum is the only path that still routes through upstream's
/// strict types — so any spec-compliant client that omits `type: "message"`
/// on a multimodal message (the documented default) fails with
/// "data did not match any variant of untagged enum InputItem". See issue
/// #9468.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(untagged)]
pub enum EasyInputContent {
    /// Plain-text content. Tried first so `"content": "hi"` short-circuits.
    Text(String),
    /// Structured content list (text/image/file parts).
    ContentList(Vec<InputContent>),
}

impl Default for EasyInputContent {
    fn default() -> Self {
        Self::Text(String::new())
    }
}

/// A simplified message input — the spec-default shape when a client omits the
/// `type` discriminator. Shadows upstream `EasyInputMessage` so the `content`
/// field routes through Dynamo's relaxed `EasyInputContent` (and transitively
/// the relaxed `InputContent` / `InputImageContent`). Field set is identical to
/// upstream for drop-in compatibility with construction sites in lib/llm.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Default)]
pub struct EasyInputMessage {
    /// Type discriminator. Optional with default `MessageType::Message` —
    /// matches the OpenAI Responses spec and `openai-python`'s
    /// `EasyInputMessageParam` (`type: Literal["message"]`, non-Required).
    #[serde(default)]
    pub r#type: MessageType,
    pub role: Role,
    pub content: EasyInputContent,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub phase: Option<MessagePhase>,
}

// ---------------------------------------------------------------------------
// Input-side Item / Message / InputItem / InputParam (shadow upstream)
// ---------------------------------------------------------------------------

/// Message item within `Item`. Untagged; disambiguated by the `role` field:
/// the `Output` variant requires `role: "assistant"` (via `AssistantRole`,
/// which is a single-variant enum) and `Input` requires `role` in
/// `"user" | "system" | "developer"` (via `InputRole`). A payload with an
/// unknown role (e.g. `"tool"`) or a missing `role` produces the generic
/// untagged-enum error — callers are expected to send a valid role. If you
/// see the "data did not match any variant of untagged enum" failure on this
/// type, it is almost always a role mismatch.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(untagged)]
pub enum MessageItem {
    /// Prior assistant output echoed back (role: assistant). Tried first — its
    /// `role` constraint excludes user/system/developer inputs.
    Output(InputOutputMessage),
    /// User / system / developer input message.
    Input(InputMessage),
}

/// A reasoning item echoed back as input for a subsequent turn. Relaxed
/// compared to upstream `ReasoningItem`: `id` and `summary` are both optional.
///
/// Upstream marks `id` (and a present `summary` array) as required, but real
/// clients omit them when round-tripping a prior reasoning turn as input:
/// Codex / OpenCode / agent SDKs send `reasoning` items carrying only
/// `encrypted_content` (and sometimes a `summary`) with no `id`. OpenAI's own
/// hosted API accepts this; the OpenAPI spec is wrong. Upstream fixed `id` in
/// `64bit/async-openai#535` (merged after our pinned async-openai), so we
/// mirror that one-line relaxation here rather than chase a crate bump.
///
/// Named `InputReasoningItem` (not `ReasoningItem`) because upstream's
/// `ReasoningItem` is dual-side: it is the canonical output-side type in
/// `OutputItem::Reasoning(..)` / `Response.output`, which must stay strict.
/// Same naming discipline as `InputOutputMessage` vs `OutputMessage`.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct InputReasoningItem {
    /// Optional on input — upstream requires it; clients drop it on echo.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// Defaults to empty when absent — upstream requires a present array.
    #[serde(default)]
    pub summary: Vec<SummaryPart>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<Vec<ReasoningTextContent>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub encrypted_content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<OutputStatus>,
}

/// Private Codex wire shape, normalized to an existing user message.
#[derive(Deserialize)]
struct CodexAgentMessage {
    #[serde(default)]
    content: Option<CodexAgentMessageContent>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum CodexAgentMessageContent {
    Text(String),
    Parts(Vec<CodexAgentMessageInputContent>),
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum CodexAgentMessageInputContent {
    InputText(InputTextContent),
    EncryptedContent { encrypted_content: String },
}

/// Structured input/output item, discriminated by `type`. Mirrors upstream
/// variant-for-variant; only `Message` and `Reasoning` use owned types.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Item {
    Message(MessageItem),
    FileSearchCall(FileSearchToolCall),
    ComputerCall(ComputerToolCall),
    ComputerCallOutput(ComputerCallOutputItemParam),
    WebSearchCall(WebSearchToolCall),
    FunctionCall(FunctionToolCall),
    FunctionCallOutput(FunctionCallOutputItemParam),
    ToolSearchCall(ToolSearchCallItemParam),
    ToolSearchOutput(ToolSearchOutputItemParam),
    Reasoning(InputReasoningItem),
    Compaction(CompactionSummaryItemParam),
    ImageGenerationCall(ImageGenToolCall),
    CodeInterpreterCall(CodeInterpreterToolCall),
    LocalShellCall(LocalShellToolCall),
    LocalShellCallOutput(LocalShellToolCallOutput),
    ShellCall(FunctionShellCallItemParam),
    ShellCallOutput(FunctionShellCallOutputItemParam),
    ApplyPatchCall(ApplyPatchToolCallItemParam),
    ApplyPatchCallOutput(ApplyPatchToolCallOutputItemParam),
    McpListTools(MCPListTools),
    McpApprovalRequest(MCPApprovalRequest),
    McpApprovalResponse(MCPApprovalResponse),
    McpCall(MCPToolCall),
    CustomToolCallOutput(CustomToolCallOutput),
    CustomToolCall(CustomToolCall),
}

/// Single input item. Untagged; order matters (most specific first).
#[derive(Debug, Serialize, Clone, PartialEq)]
#[serde(untagged)]
pub enum InputItem {
    ItemReference(ItemReference),
    Item(Item),
    EasyMessage(EasyInputMessage),
}

#[derive(Deserialize)]
#[serde(untagged)]
enum InputItemWire {
    ItemReference(ItemReference),
    Item(Item),
    EasyMessage(EasyInputMessage),
}

impl<'de> Deserialize<'de> for InputItem {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = serde_json::Value::deserialize(deserializer)?;
        if value.get("type").and_then(serde_json::Value::as_str) == Some("agent_message") {
            let message = CodexAgentMessage::deserialize(value).map_err(de::Error::custom)?;
            return Ok(normalize_codex_agent_message(message));
        }

        match InputItemWire::deserialize(value).map_err(de::Error::custom)? {
            InputItemWire::ItemReference(item) => Ok(Self::ItemReference(item)),
            InputItemWire::Item(item) => Ok(Self::Item(item)),
            InputItemWire::EasyMessage(message) => Ok(Self::EasyMessage(message)),
        }
    }
}

fn normalize_codex_agent_message(message: CodexAgentMessage) -> InputItem {
    let content = match message.content {
        None => String::new(),
        Some(CodexAgentMessageContent::Text(text)) => text,
        Some(CodexAgentMessageContent::Parts(parts)) => parts
            .into_iter()
            .map(|part| match part {
                CodexAgentMessageInputContent::InputText(part) => part.text,
                CodexAgentMessageInputContent::EncryptedContent { encrypted_content } => {
                    encrypted_content
                }
            })
            .collect::<Vec<_>>()
            .join("\n"),
    };
    InputItem::EasyMessage(EasyInputMessage {
        r#type: MessageType::Message,
        role: Role::User,
        content: EasyInputContent::Text(content),
        phase: None,
    })
}

/// Input to a `POST /v1/responses` request.
#[derive(Debug, Serialize, Clone, PartialEq)]
#[serde(untagged)]
pub enum InputParam {
    Text(String),
    Items(Vec<InputItem>),
}

impl<'de> Deserialize<'de> for InputParam {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        match serde_json::Value::deserialize(deserializer)? {
            serde_json::Value::String(text) => Ok(Self::Text(text)),
            serde_json::Value::Array(items) => {
                serde_json::from_value(serde_json::Value::Array(items))
                    .map(Self::Items)
                    .map_err(de::Error::custom)
            }
            _ => Err(de::Error::custom(
                "input must be a string or an array of input items",
            )),
        }
    }
}

impl Default for InputParam {
    fn default() -> Self {
        Self::Text(String::new())
    }
}

// ---------------------------------------------------------------------------
// CreateResponse (owned, uses Dynamo-owned InputParam)
// ---------------------------------------------------------------------------

/// Request body for `POST /v1/responses`. Mirrors upstream `CreateResponse`
/// field-for-field but uses Dynamo-owned `InputParam`, which transitively
/// accepts the relaxed input shapes described in this module's header. All
/// other fields reference upstream types verbatim.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Default)]
pub struct CreateResponse {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub background: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub conversation: Option<ConversationParam>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include: Option<Vec<IncludeEnum>>,
    pub input: InputParam,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tool_calls: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub previous_response_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt: Option<Prompt>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_cache_key: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_cache_retention: Option<PromptCacheRetention>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<Reasoning>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub safety_identifier: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<ServiceTier>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub store: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<ResponseStreamOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<ResponseTextParam>,
    #[serde(
        default,
        deserialize_with = "deserialize_tool_choice",
        skip_serializing_if = "Option::is_none"
    )]
    pub tool_choice: Option<ToolChoiceParam>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub truncation: Option<Truncation>,
}

// ---------------------------------------------------------------------------
// CountInputTokens (`POST /v1/responses/input_tokens`)
// ---------------------------------------------------------------------------

/// The `object` discriminator on a [`CountInputTokensResponse`].
pub const RESPONSE_INPUT_TOKENS_OBJECT: &str = "response.input_tokens";

/// Request body for `POST /v1/responses/input_tokens`.
///
/// A subset of [`CreateResponse`] — only the fields that reach the rendered
/// prompt. This mirrors `AnthropicCountTokensRequest`, which is the same
/// subset-of-the-create-request shape for `POST /v1/messages/count_tokens`.
///
/// Two deliberate differences from `CreateResponse`: `input` defaults (the
/// count endpoint accepts a body without one, whereas creating a response
/// requires it), and unknown fields are ignored, so stateful parameters
/// Dynamo does not serve (`conversation`, `previous_response_id`) are accepted
/// and disregarded rather than rejected. This endpoint reports a pre-flight
/// estimate; it never generates, so there is nothing for them to affect.
///
/// Deserialization is forgiving in two further places — an explicit
/// `"input": null` and unrecognized tool shapes — for the same reason
/// `AnthropicTool` keeps every field but `name` optional: a pre-flight
/// estimate that rejects a body it could have scored is strictly worse than
/// one that scores it approximately. See [`deserialize_lenient_tools`].
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Default)]
pub struct CountInputTokensRequest {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// `#[serde(default)]` alone covers an absent `input`, but not an explicit
    /// `"input": null` — serde still hands that null to `InputParam`, whose
    /// deserializer rejects it. Here both mean "nothing to count".
    #[serde(default, deserialize_with = "deserialize_null_default_input")]
    pub input: InputParam,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "deserialize_lenient_tools"
    )]
    pub tools: Option<Vec<Tool>>,
}

fn deserialize_null_default_input<'de, D>(deserializer: D) -> Result<InputParam, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Ok(Option::<InputParam>::deserialize(deserializer)?.unwrap_or_default())
}

/// Drop tool entries that do not deserialize into a known [`Tool`], rather than
/// failing the whole request.
///
/// `Tool` is upstream's `#[serde(tag = "type")]` enum, so it models only the
/// tool types the pinned `async-openai` knows. A caller that forwards a tool in
/// a shape upstream does not model — a Chat-Completions-style `{"type":
/// "custom", "custom": {...}}`, or a tool type newer than the pin — would
/// otherwise get a 400 for a field that contributes almost nothing to the
/// estimate.
///
/// Dropping costs nothing: [`estimate_tool_len`] already scores every
/// non-function tool as 0, because `convert_tools` forwards only function tools
/// to the backend. An unparseable tool was going to be worth 0 either way; this
/// only decides whether the rest of the body still gets counted. That is the
/// same trade `estimate_tool_len` documents when it wildcards where
/// `measure_item` is exhaustive — `Tool` is upstream's type, and we carry
/// no obligation to mirror its variants.
fn deserialize_lenient_tools<'de, D>(deserializer: D) -> Result<Option<Vec<Tool>>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let Some(raw) = Option::<Vec<serde_json::Value>>::deserialize(deserializer)? else {
        return Ok(None);
    };
    Ok(Some(
        raw.into_iter()
            .filter_map(|tool| serde_json::from_value::<Tool>(tool).ok())
            .collect(),
    ))
}

/// Response body for `POST /v1/responses/input_tokens`.
///
/// `Deserialize` is derived where the Anthropic count response is
/// serialize-only: this body is round-tripped by the frontend's integration
/// tests, which assert on the parsed shape rather than on raw JSON.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct CountInputTokensResponse {
    /// Always [`RESPONSE_INPUT_TOKENS_OBJECT`]. Required by the OpenAI spec.
    pub object: String,
    pub input_tokens: u32,
}

impl CountInputTokensResponse {
    pub fn new(input_tokens: u32) -> Self {
        Self {
            object: RESPONSE_INPUT_TOKENS_OBJECT.to_string(),
            input_tokens,
        }
    }
}

impl CountInputTokensRequest {
    /// Estimate input token count using a `len/3` heuristic.
    ///
    /// Same contract as `AnthropicCountTokensRequest::estimate_tokens`: sum the
    /// character lengths of everything that reaches the prompt, divide by three,
    /// and never report zero for input that carried content.
    ///
    /// This is an estimate, not a tokenization. A frontend serving a
    /// backend that tokenizes for itself has no tokenizer loaded, so this
    /// endpoint has to be able to answer without one.
    pub fn estimate_tokens(&self) -> u32 {
        let mut total_len: usize = 0;

        // `instructions` and a top-level string `input` are not free-floating
        // text: the converter turns them into a system and a user chat message
        // respectively, exactly like the item messages below. Charge them the
        // same role markers, or the identical prompt scores differently
        // depending on which shape the caller used to express it.
        //
        // Both are skipped when empty, because an absent field is not a
        // message. `InputParam::default()` is `Text("")`, so a body with no
        // `input` at all lands here — and it must stay worth zero.
        if let Some(instructions) = &self.instructions.as_ref().filter(|text| !text.is_empty()) {
            total_len += role_len(Role::System) + instructions.len();
        }

        match &self.input {
            InputParam::Text(text) if text.is_empty() => {}
            InputParam::Text(text) => total_len += role_len(Role::User) + text.len(),
            InputParam::Items(items) => total_len += estimate_input_items_len(items),
        }

        if let Some(tools) = &self.tools {
            for tool in tools {
                total_len += estimate_tool_len(tool);
            }
        }

        let tokens = total_len / 3;
        if tokens == 0 && total_len > 0 {
            1
        } else {
            tokens as u32
        }
    }
}

/// Approximate character cost of a role marker, using the same constants
/// `AnthropicCountTokensRequest::estimate_tokens` applies for the same purpose.
fn role_len(role: Role) -> usize {
    match role {
        Role::User => 4,
        Role::Assistant => 9,
        Role::System => 6,
        Role::Developer => 9,
    }
}

fn input_role_len(role: InputRole) -> usize {
    match role {
        InputRole::User => 4,
        InputRole::System => 6,
        InputRole::Developer => 9,
    }
}

/// The `tool` role marker on a tool-result message.
///
/// `role_len` covers only the roles upstream's `Role` enum models; chat
/// completions' `tool` role has no variant there, so it gets its own constant
/// on the same basis the others use — the length of the role word.
const TOOL_ROLE_LEN: usize = 4;

/// What an input item does to the converter's pending assistant message.
enum GroupEffect {
    /// Opens or extends the pending assistant message, emitting no message of
    /// its own.
    Assistant,
    /// Flushes any pending assistant message and emits its own.
    Flush,
    /// Neither emits nor flushes — the converter skips it outright.
    Skip,
}

/// Sum the input items, mirroring `convert_input_items_to_messages` — including
/// its coalescing.
///
/// Assistant-side items do not each become a message. An echoed assistant
/// message, a function call, and a reasoning summary all push into one
/// `PendingAssistant`, which is flushed only by the next non-assistant item or
/// by the end of the list. So the assistant role marker is charged once per
/// flushed group, not once per item: two parallel function calls are one
/// assistant turn and cost one marker between them.
///
/// A per-item sum cannot express that, which is why this walks the list rather
/// than mapping over it.
fn estimate_input_items_len(items: &[InputItem]) -> usize {
    let mut total = 0;
    let mut assistant_open = false;

    for item in items {
        let (effect, len) = measure_input_item(item);
        total += len;
        match effect {
            GroupEffect::Assistant => {
                if !assistant_open {
                    assistant_open = true;
                    total += role_len(Role::Assistant);
                }
            }
            GroupEffect::Flush => assistant_open = false,
            GroupEffect::Skip => {}
        }
    }

    total
}

/// Measure one item and report what it does to the pending assistant group.
///
/// Assistant-side arms return content only: their role marker is the caller's
/// to add, once per group.
fn measure_input_item(item: &InputItem) -> (GroupEffect, usize) {
    match item {
        // A pointer to an item held server-side. The content it names is not in
        // this request, so there is nothing here to measure — and the converter
        // skips it without flushing, so it cannot split an assistant group.
        InputItem::ItemReference(_) => (GroupEffect::Skip, 0),
        InputItem::EasyMessage(message) => {
            let content = estimate_easy_content_len(&message.content);
            match message.role {
                // A prior assistant turn echoed back; coalesces like the strict
                // `MessageItem::Output` path.
                Role::Assistant => (GroupEffect::Assistant, content),
                role => (GroupEffect::Flush, role_len(role) + content),
            }
        }
        InputItem::Item(item) => measure_item(item),
    }
}

fn estimate_easy_content_len(content: &EasyInputContent) -> usize {
    match content {
        EasyInputContent::Text(text) => text.len(),
        EasyInputContent::ContentList(parts) => parts.iter().map(estimate_input_content_len).sum(),
    }
}

/// Only text parts are measured. An image or file contributes tokens as a
/// function of its decoded form, which a character count cannot model at all —
/// the Anthropic estimator skips non-text blocks for the same reason.
fn estimate_input_content_len(part: &InputContent) -> usize {
    match part {
        InputContent::InputText(text) => text.text.len(),
        InputContent::InputImage(_) | InputContent::InputFile(_) => 0,
    }
}

fn measure_item(item: &Item) -> (GroupEffect, usize) {
    match item {
        Item::Message(MessageItem::Input(message)) => (
            GroupEffect::Flush,
            input_role_len(message.role)
                + message
                    .content
                    .iter()
                    .map(estimate_input_content_len)
                    .sum::<usize>(),
        ),
        // Assistant-side: pushed into the pending message, so no role marker
        // here. See `estimate_input_items_len`.
        Item::Message(MessageItem::Output(message)) => (
            GroupEffect::Assistant,
            message
                .content
                .iter()
                .map(|part| match part {
                    InputOutputMessageContent::OutputText(text) => text.text.len(),
                    InputOutputMessageContent::Refusal(refusal) => refusal.refusal.len(),
                })
                .sum::<usize>(),
        ),
        // Rendered as a tool call on the pending assistant message. `call_id`
        // is excluded deliberately: it is correlation plumbing, not prompt
        // text, in the templates Dynamo renders.
        Item::FunctionCall(call) => (
            GroupEffect::Assistant,
            call.name.len() + call.arguments.len(),
        ),
        // Its own `tool`-role message, one per output, so it both flushes the
        // assistant group and carries a role marker of its own.
        Item::FunctionCallOutput(output) => (
            GroupEffect::Flush,
            TOOL_ROLE_LEN
                + match &output.output {
                    FunctionCallOutput::Text(text) => text.len(),
                    FunctionCallOutput::Content(parts) => parts
                        .iter()
                        .map(|part| match part {
                            UpstreamInputContent::InputText(text) => text.text.len(),
                            UpstreamInputContent::InputImage(_)
                            | UpstreamInputContent::InputFile(_) => 0,
                        })
                        .sum(),
                },
        ),
        // Only `summary` is measured, because only `summary` is rendered:
        // the converter joins the summary parts and drops the rest of the
        // item. `content` is excluded for that reason alone — if Dynamo
        // learns to render it, it needs to start counting here too.
        // `encrypted_content` is excluded on its own merits: it is an opaque
        // blob the model never sees as prompt text, and it is routinely far
        // larger than the reasoning it stands for.
        Item::Reasoning(reasoning) => (
            GroupEffect::Assistant,
            reasoning
                .summary
                .iter()
                .map(|part| match part {
                    SummaryPart::SummaryText(text) => text.text.len(),
                })
                .sum(),
        ),
        // Everything below contributes nothing, because nothing below reaches
        // the prompt: Dynamo's `convert_input_items_to_messages` flushes and
        // skips every one of these ("we do not have a faithful Chat
        // Completions mapping"). The arms above are exactly its handled set.
        // Measuring the serialized form instead would bill callers for JSON
        // scaffolding the model never sees: a bare `web_search_call` is 59
        // characters, or 19 phantom tokens, and agentic clients echo many
        // such items per turn.
        //
        // Listed out rather than wildcarded on purpose. `Item` is a shadow
        // enum that "mirrors upstream variant-for-variant" and has to be
        // extended by hand whenever upstream grows a variant (see CLAUDE.md,
        // "Owned input chain"). A `_` arm would let that new variant default
        // to zero silently; an exhaustive match turns it into a compile error
        // that forces a render-or-not decision here, which is the same
        // mechanism CLAUDE.md relies on to catch drift in `From` impls.
        Item::FileSearchCall(_)
        | Item::ComputerCall(_)
        | Item::ComputerCallOutput(_)
        | Item::WebSearchCall(_)
        | Item::ToolSearchCall(_)
        | Item::ToolSearchOutput(_)
        | Item::Compaction(_)
        | Item::ImageGenerationCall(_)
        | Item::CodeInterpreterCall(_)
        | Item::LocalShellCall(_)
        | Item::LocalShellCallOutput(_)
        | Item::ShellCall(_)
        | Item::ShellCallOutput(_)
        | Item::ApplyPatchCall(_)
        | Item::ApplyPatchCallOutput(_)
        | Item::McpListTools(_)
        | Item::McpApprovalRequest(_)
        | Item::McpApprovalResponse(_)
        | Item::McpCall(_)
        | Item::CustomToolCallOutput(_)
        | Item::CustomToolCall(_) => (GroupEffect::Flush, 0),
    }
}

/// Mirrors `convert_tools`: only function tools are forwarded to the backend,
/// namespaced ones flattened to their bare function members. Hosted tools
/// (web search, file search, computer use) are dropped there and so cost
/// nothing here.
///
/// Wildcarded where `measure_item` is exhaustive, and deliberately so:
/// `Tool` is upstream's type, not one of our shadows, so we carry no
/// obligation to mirror its variants. Pinning it exhaustively would only
/// break the build every time async-openai adds a hosted tool we would
/// score as zero anyway.
fn estimate_tool_len(tool: &Tool) -> usize {
    match tool {
        Tool::Function(function) => function_tool_len(
            &function.name,
            function.description.as_ref(),
            function.parameters.as_ref(),
        ),
        Tool::Namespace(namespace) => namespace
            .tools
            .iter()
            .map(|tool| match tool {
                // The namespace name is an origin marker used to detect
                // collisions, not prompt text — `push_function` forwards the
                // bare function name.
                NamespaceToolParamTool::Function(function) => function_tool_len(
                    &function.name,
                    function.description.as_ref(),
                    function.parameters.as_ref(),
                ),
                NamespaceToolParamTool::Custom(_) => 0,
            })
            .sum(),
        _ => 0,
    }
}

fn function_tool_len(
    name: &str,
    description: Option<&String>,
    parameters: Option<&serde_json::Value>,
) -> usize {
    name.len()
        + description.map_or(0, |description| description.len())
        + parameters.map_or(0, |schema| schema.to_string().len())
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- tool_choice object form (ai-dynamo/dynamo#10963 CASE 1) ----

    fn tool_choice_of(json: serde_json::Value) -> Option<ToolChoiceParam> {
        let req: CreateResponse = serde_json::from_value(serde_json::json!({
            "input": "hi",
            "tool_choice": json,
        }))
        .expect("CreateResponse should deserialize");
        req.tool_choice
    }

    #[test]
    fn tool_choice_mode_object_coerces_to_mode() {
        // Anthropic-style / litellm shape: a mode expressed as an object with
        // extra keys. Must coerce to the corresponding `Mode`, ignoring extras.
        assert_eq!(
            tool_choice_of(serde_json::json!({"type": "auto", "disable_parallel_tool_use": true})),
            Some(ToolChoiceParam::Mode(ToolChoiceOptions::Auto)),
        );
        assert_eq!(
            tool_choice_of(serde_json::json!({"type": "none"})),
            Some(ToolChoiceParam::Mode(ToolChoiceOptions::None)),
        );
        assert_eq!(
            tool_choice_of(serde_json::json!({"type": "required"})),
            Some(ToolChoiceParam::Mode(ToolChoiceOptions::Required)),
        );
    }

    #[test]
    fn tool_choice_bare_string_still_works() {
        assert_eq!(
            tool_choice_of(serde_json::json!("auto")),
            Some(ToolChoiceParam::Mode(ToolChoiceOptions::Auto)),
        );
    }

    #[test]
    fn tool_choice_specific_function_object_still_works() {
        // The object form naming a specific tool must NOT be swallowed by the
        // mode coercion — `type: "function"` is not a mode.
        match tool_choice_of(serde_json::json!({"type": "function", "name": "get_weather"})) {
            Some(ToolChoiceParam::Function(f)) => assert_eq!(f.name, "get_weather"),
            other => panic!("expected Function tool choice, got {other:?}"),
        }
    }

    #[test]
    fn tool_choice_absent_is_none() {
        let req: CreateResponse =
            serde_json::from_value(serde_json::json!({"input": "hi"})).unwrap();
        assert!(req.tool_choice.is_none());
    }

    // ---- reasoning item echoed back without id/summary (#10963 CASE 2) ----

    #[test]
    fn reasoning_input_without_id_deserializes() {
        // Codex / OpenCode / agent SDKs echo a reasoning item with no `id`.
        let json = serde_json::json!({
            "type": "reasoning",
            "summary": [{"type": "summary_text", "text": "thinking"}],
        });
        match serde_json::from_value::<InputItem>(json).expect("should deserialize") {
            InputItem::Item(Item::Reasoning(r)) => {
                assert!(r.id.is_none());
                assert_eq!(r.summary.len(), 1);
            }
            other => panic!("expected Item::Reasoning, got {other:?}"),
        }
    }

    #[test]
    fn reasoning_input_encrypted_without_id_or_summary_deserializes() {
        let json = serde_json::json!({
            "type": "reasoning",
            "encrypted_content": "AB==",
        });
        match serde_json::from_value::<InputItem>(json).expect("should deserialize") {
            InputItem::Item(Item::Reasoning(r)) => {
                assert!(r.id.is_none());
                assert!(r.summary.is_empty());
                assert_eq!(r.encrypted_content.as_deref(), Some("AB=="));
            }
            other => panic!("expected Item::Reasoning, got {other:?}"),
        }
    }

    #[test]
    fn reasoning_input_with_id_still_works() {
        let json = serde_json::json!({
            "type": "reasoning",
            "id": "rs_1",
            "summary": [{"type": "summary_text", "text": "x"}],
            "status": "completed",
        });
        match serde_json::from_value::<InputItem>(json).expect("should deserialize") {
            InputItem::Item(Item::Reasoning(r)) => assert_eq!(r.id.as_deref(), Some("rs_1")),
            other => panic!("expected Item::Reasoning, got {other:?}"),
        }
    }

    #[test]
    fn full_request_with_idless_reasoning_item_deserializes() {
        // The exact failure mode reported in #10963: a turn-2 `input` list
        // containing an echoed reasoning item that lost its `id`.
        let req: Result<CreateResponse, _> = serde_json::from_value(serde_json::json!({
            "model": "m",
            "input": [
                {"role": "user", "content": "hi"},
                {"type": "reasoning", "summary": [{"type": "summary_text", "text": "x"}]},
            ],
        }));
        assert!(
            req.is_ok(),
            "idless reasoning input should deserialize: {req:?}"
        );
    }

    #[test]
    fn codex_agent_message_normalizes_to_user_message() {
        let req: CreateResponse = serde_json::from_value(serde_json::json!({
            "input": [{
                "type": "agent_message",
                "author": "/root",
                "recipient": "/root/worker",
                "content": [
                    {"type": "input_text", "text": "First."},
                    {"type": "input_text", "text": "Second."},
                ],
            }],
        }))
        .expect("Codex agent message should deserialize");

        let InputParam::Items(items) = req.input else {
            panic!("expected items");
        };
        assert!(matches!(
            &items[0],
            InputItem::EasyMessage(EasyInputMessage {
                role: Role::User,
                content: EasyInputContent::Text(text),
                ..
            }) if text == "First.\nSecond."
        ));
    }

    #[test]
    fn codex_agent_message_string_content_normalizes_to_user_message() {
        let item: InputItem = serde_json::from_value(serde_json::json!({
            "type": "agent_message",
            "author": "/root",
            "recipient": "/root/worker",
            "content": "Return exactly OK.",
        }))
        .expect("Codex agent message with string content should deserialize");

        assert!(matches!(
            item,
            InputItem::EasyMessage(EasyInputMessage {
                content: EasyInputContent::Text(text),
                ..
            }) if text == "Return exactly OK."
        ));
    }

    #[test]
    fn codex_agent_message_normalizes_encrypted_content() {
        let req: CreateResponse = serde_json::from_value(serde_json::json!({
            "input": [{
                "type": "agent_message",
                "content": [
                    {"type": "input_text", "text": "Payload:"},
                    {"type": "encrypted_content", "encrypted_content": "Return exactly OK."},
                ],
            }],
        }))
        .expect("Codex agent message with encrypted content should deserialize");

        let InputParam::Items(items) = req.input else {
            panic!("expected items");
        };
        assert!(matches!(
            &items[0],
            InputItem::EasyMessage(EasyInputMessage {
                content: EasyInputContent::Text(text),
                ..
            }) if text == "Payload:\nReturn exactly OK."
        ));
    }

    #[test]
    fn codex_agent_message_missing_content_normalizes_empty() {
        let item: InputItem = serde_json::from_value(serde_json::json!({
            "type": "agent_message",
            "author": "/root",
            "recipient": "/root/worker",
        }))
        .expect("Codex agent message without content should deserialize");
        assert!(matches!(
            item,
            InputItem::EasyMessage(EasyInputMessage {
                content: EasyInputContent::Text(text),
                ..
            }) if text.is_empty()
        ));
    }

    #[test]
    fn codex_agent_message_null_content_normalizes_empty() {
        let item: InputItem = serde_json::from_value(serde_json::json!({
            "type": "agent_message",
            "author": "/root",
            "recipient": "/root/worker",
            "content": null,
        }))
        .expect("Codex agent message with null content should deserialize");
        assert!(matches!(
            item,
            InputItem::EasyMessage(EasyInputMessage {
                content: EasyInputContent::Text(text),
                ..
            }) if text.is_empty()
        ));
    }

    #[test]
    fn relaxed_assistant_message_without_id_or_status() {
        let json = serde_json::json!({
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "hi"}]
        });
        let item: InputItem = serde_json::from_value(json).unwrap();
        match item {
            InputItem::Item(Item::Message(MessageItem::Output(out))) => {
                assert_eq!(out.role, AssistantRole::Assistant);
                assert!(out.id.is_none());
                assert!(out.status.is_none());
            }
            other => panic!("expected Item::Message(Output), got {other:?}"),
        }
    }

    #[test]
    fn input_image_without_detail_defaults_to_auto() {
        let json = serde_json::json!({
            "type": "input_image",
            "image_url": "https://example.com/cat.jpg"
        });
        let content: InputContent = serde_json::from_value(json).unwrap();
        match content {
            InputContent::InputImage(img) => assert_eq!(img.detail, ImageDetail::Auto),
            other => panic!("expected InputImage, got {other:?}"),
        }
    }

    #[test]
    fn input_image_with_explicit_null_detail_defaults_to_auto() {
        let json = serde_json::json!({
            "type": "input_image",
            "image_url": "https://example.com/cat.jpg",
            "detail": null
        });
        let content: InputContent = serde_json::from_value(json).unwrap();
        match content {
            InputContent::InputImage(img) => assert_eq!(img.detail, ImageDetail::Auto),
            other => panic!("expected InputImage, got {other:?}"),
        }
    }

    #[test]
    fn assistant_message_without_content_field_deserializes() {
        // Bare assistant shell — no `content` field at all. Seen in real
        // Codex/Agents-SDK traffic on pure tool-call turns. `#[serde(default)]`
        // on `content` must accept omission and yield an empty vec.
        let json = serde_json::json!({
            "type": "message",
            "role": "assistant"
        });
        let item: InputItem = serde_json::from_value(json).unwrap();
        match item {
            InputItem::Item(Item::Message(MessageItem::Output(out))) => {
                assert_eq!(out.role, AssistantRole::Assistant);
                assert!(out.content.is_empty());
                assert!(out.id.is_none());
                assert!(out.status.is_none());
            }
            other => panic!("expected Item::Message(Output), got {other:?}"),
        }
    }

    #[test]
    fn assistant_message_with_explicit_null_content_deserializes() {
        // Mirrors the `annotations: null` case: some serializers emit JSON null
        // for absent fields instead of omitting them. `Vec::deserialize` rejects
        // null, so `content` also needs `deserialize_null_as_empty_vec`.
        let json = serde_json::json!({
            "type": "message",
            "role": "assistant",
            "content": null
        });
        let item: InputItem = serde_json::from_value(json).unwrap();
        match item {
            InputItem::Item(Item::Message(MessageItem::Output(out))) => {
                assert!(out.content.is_empty());
            }
            other => panic!("expected Item::Message(Output), got {other:?}"),
        }
    }

    #[test]
    fn mcp_call_item_deserializes() {
        // Guards against Item variant drift vs upstream — MCP item types were
        // added after the initial owned `Item` chain landed.
        let json = serde_json::json!({
            "type": "mcp_call",
            "id": "mcp_1",
            "server_label": "srv",
            "name": "t",
            "arguments": "{}"
        });
        let item: InputItem = serde_json::from_value(json).unwrap();
        assert!(matches!(item, InputItem::Item(Item::McpCall(_))));
    }

    #[test]
    fn strict_assistant_message_still_deserializes() {
        let json = serde_json::json!({
            "type": "message",
            "role": "assistant",
            "id": "msg_1",
            "status": "completed",
            "content": [{"type": "output_text", "text": "hi", "annotations": []}]
        });
        let item: InputItem = serde_json::from_value(json).unwrap();
        match item {
            InputItem::Item(Item::Message(MessageItem::Output(out))) => {
                assert_eq!(out.id.as_deref(), Some("msg_1"));
                assert_eq!(out.status, Some(OutputStatus::Completed));
            }
            other => panic!("expected Item::Message(Output), got {other:?}"),
        }
    }

    #[test]
    fn user_message_routes_to_input_variant() {
        let json = serde_json::json!({
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "hi"}]
        });
        let item: InputItem = serde_json::from_value(json).unwrap();
        assert!(matches!(
            item,
            InputItem::Item(Item::Message(MessageItem::Input(_)))
        ));
    }

    #[test]
    fn function_call_item_still_deserializes() {
        let json = serde_json::json!({
            "type": "function_call",
            "call_id": "c",
            "name": "f",
            "arguments": "{}"
        });
        let item: InputItem = serde_json::from_value(json).unwrap();
        assert!(matches!(item, InputItem::Item(Item::FunctionCall(_))));
    }

    #[test]
    fn easy_message_string_content_routes_to_easymessage() {
        let json = serde_json::json!({"role": "assistant", "content": "x"});
        let item: InputItem = serde_json::from_value(json).unwrap();
        assert!(matches!(item, InputItem::EasyMessage(_)));
    }

    #[test]
    fn output_text_without_annotations_defaults_empty() {
        let json = serde_json::json!({"type": "output_text", "text": "hi"});
        let part: InputOutputMessageContent = serde_json::from_value(json).unwrap();
        match part {
            InputOutputMessageContent::OutputText(t) => {
                assert!(t.annotations.is_empty());
            }
            _ => panic!("expected OutputText"),
        }
    }

    #[test]
    fn output_text_with_explicit_null_annotations_deserializes_as_empty() {
        // Some clients serialize absent fields as JSON null instead of omitting
        // them. `Vec::deserialize` would reject null; the custom deserializer
        // treats explicit null identically to a missing field.
        let json = serde_json::json!({"type": "output_text", "text": "hi", "annotations": null});
        let part: InputOutputMessageContent = serde_json::from_value(json).unwrap();
        match part {
            InputOutputMessageContent::OutputText(t) => {
                assert!(t.annotations.is_empty());
            }
            _ => panic!("expected OutputText"),
        }
    }

    #[test]
    fn assistant_message_with_explicit_null_id_and_status_deserializes() {
        // `Option<T>` natively accepts null as `None`, so these explicit-null
        // fields should flow through without a custom deserializer. This test
        // pins that behavior against accidental regressions (e.g. if someone
        // switches the field type away from `Option<_>`).
        let json = serde_json::json!({
            "type": "message",
            "role": "assistant",
            "id": null,
            "status": null,
            "content": [{"type": "output_text", "text": "hi", "annotations": null}]
        });
        let item: InputItem = serde_json::from_value(json).unwrap();
        match item {
            InputItem::Item(Item::Message(MessageItem::Output(out))) => {
                assert!(out.id.is_none());
                assert!(out.status.is_none());
                assert_eq!(out.content.len(), 1);
            }
            other => panic!("expected Item::Message(Output), got {other:?}"),
        }
    }

    #[test]
    fn create_response_roundtrip_with_relaxed_input() {
        let body = serde_json::json!({
            "model": "m",
            "input": [
                {"type": "message", "role": "user", "content": [
                    {"type": "input_text", "text": "hi"}
                ]},
                {"type": "function_call", "call_id": "c", "name": "f", "arguments": "{}"},
                {"type": "message", "role": "assistant", "content": [
                    {"type": "output_text", "text": "\n\n"}
                ]},
                {"type": "function_call_output", "call_id": "c", "output": "x"}
            ]
        });

        let req: CreateResponse = serde_json::from_value(body).unwrap();
        let items = match &req.input {
            InputParam::Items(items) => items,
            _ => panic!("expected Items"),
        };
        assert_eq!(items.len(), 4);
        assert!(matches!(
            items[2],
            InputItem::Item(Item::Message(MessageItem::Output(_)))
        ));
    }

    // ---- EasyInputMessage / multimodal-without-`type` regression coverage ----
    // See issue #9468. Before the EasyInputMessage/EasyInputContent shadow
    // landed, the `InputItem::EasyMessage` fallback still routed through
    // upstream's strict `InputImageContent` (required `detail`), so any
    // multimodal message that omitted the spec-default `type: "message"` would
    // fail with "data did not match any variant of untagged enum InputItem".

    #[test]
    fn easy_message_multimodal_without_type_routes_to_easymessage() {
        // AIPerf's pre-PR-931 payload shape: no top-level `type`, content is a
        // list containing an `input_image` part with no `detail`.
        let json = serde_json::json!({
            "role": "user",
            "content": [
                {"type": "input_image", "image_url": "data:image/png;base64,abc"}
            ]
        });
        let item: InputItem = serde_json::from_value(json).unwrap();
        match item {
            InputItem::EasyMessage(easy) => {
                assert_eq!(easy.role, Role::User);
                assert_eq!(easy.r#type, MessageType::Message);
                match easy.content {
                    EasyInputContent::ContentList(parts) => {
                        assert_eq!(parts.len(), 1);
                        match &parts[0] {
                            InputContent::InputImage(img) => {
                                assert_eq!(img.detail, ImageDetail::Auto);
                                assert_eq!(
                                    img.image_url.as_deref(),
                                    Some("data:image/png;base64,abc")
                                );
                            }
                            other => panic!("expected InputImage, got {other:?}"),
                        }
                    }
                    other => panic!("expected ContentList, got {other:?}"),
                }
            }
            other => panic!("expected EasyMessage, got {other:?}"),
        }
    }

    #[test]
    fn easy_message_multimodal_with_explicit_null_detail() {
        // Same shape as above but with `detail: null` — exercises the
        // null-as-default path on the relaxed `InputImageContent` reached via
        // the EasyMessage variant.
        let json = serde_json::json!({
            "role": "user",
            "content": [
                {"type": "input_image", "image_url": "data:image/png;base64,abc", "detail": null}
            ]
        });
        let item: InputItem = serde_json::from_value(json).unwrap();
        assert!(matches!(item, InputItem::EasyMessage(_)));
    }

    #[test]
    fn easy_message_assistant_multimodal_without_type() {
        // Mixed-turn shape AIPerf emits when the prior assistant turn carried
        // structured (non-string) content: role=assistant, content list, no
        // top-level `type`.
        let json = serde_json::json!({
            "role": "assistant",
            "content": [
                {"type": "input_text", "text": "ok"}
            ]
        });
        let item: InputItem = serde_json::from_value(json).unwrap();
        match item {
            InputItem::EasyMessage(easy) => {
                assert_eq!(easy.role, Role::Assistant);
            }
            other => panic!("expected EasyMessage(assistant), got {other:?}"),
        }
    }

    #[test]
    fn easy_message_text_only_without_type_unchanged() {
        // Regression guard: the pre-existing text-only path was already
        // working (no multimodal content -> never hit upstream's strict
        // `InputImageContent`). Pin it so a future glob-shadow change can't
        // break it.
        let json = serde_json::json!({"role": "user", "content": "Hello"});
        let item: InputItem = serde_json::from_value(json).unwrap();
        match item {
            InputItem::EasyMessage(easy) => {
                assert_eq!(easy.role, Role::User);
                assert!(matches!(easy.content, EasyInputContent::Text(ref s) if s == "Hello"));
            }
            other => panic!("expected EasyMessage(Text), got {other:?}"),
        }
    }

    #[test]
    fn easy_message_with_explicit_type_still_routes_to_item_message() {
        // AIPerf's post-PR-931 payload (with `type: "message"`) should still
        // hit the structured `Item::Message` path first — proving the existing
        // strict path didn't regress when EasyMessage was shadowed.
        let json = serde_json::json!({
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_image", "image_url": "data:image/png;base64,abc"}
            ]
        });
        let item: InputItem = serde_json::from_value(json).unwrap();
        match item {
            InputItem::Item(Item::Message(MessageItem::Input(msg))) => {
                assert_eq!(msg.role, InputRole::User);
                assert_eq!(msg.content.len(), 1);
            }
            other => panic!("expected Item::Message(Input), got {other:?}"),
        }
    }

    #[test]
    fn create_response_roundtrip_aiperf_pre_pr931_payload() {
        // End-to-end shape: the exact request body AIPerf was emitting before
        // PR-931 for a multi-turn multimodal conversation. Mirrors what the
        // HTTP frontend receives. Must deserialize without error and preserve
        // turn ordering.
        let body = serde_json::json!({
            "model": "Qwen/Qwen2-VL-2B-Instruct",
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Describe"},
                        {"type": "input_image", "image_url": "data:image/png;base64,abc"}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [{"type": "input_text", "text": "ok"}]
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Now describe a different one."}]
                }
            ]
        });
        let req: CreateResponse = serde_json::from_value(body).unwrap();
        let items = match &req.input {
            InputParam::Items(items) => items,
            _ => panic!("expected Items"),
        };
        assert_eq!(items.len(), 3);
        // All three turns must land as EasyMessage (no top-level `type`).
        for (idx, item) in items.iter().enumerate() {
            assert!(
                matches!(item, InputItem::EasyMessage(_)),
                "turn {idx} did not route to EasyMessage: {item:?}",
            );
        }
    }

    // ---- count input tokens (POST /v1/responses/input_tokens) ----

    fn count(body: serde_json::Value) -> u32 {
        serde_json::from_value::<CountInputTokensRequest>(body)
            .expect("count request should deserialize")
            .estimate_tokens()
    }

    #[test]
    fn count_tokens_plain_text_input() {
        // user role (4) + "Hello, world!" (13) == 17; 17 / 3 == 5.
        assert_eq!(
            count(serde_json::json!({"model": "m", "input": "Hello, world!"})),
            5
        );
    }

    #[test]
    fn count_tokens_input_is_optional() {
        // The count endpoint accepts a body without `input`, unlike CreateResponse.
        assert_eq!(count(serde_json::json!({"model": "m"})), 0);
    }

    #[test]
    fn count_tokens_empty_input_is_zero() {
        assert_eq!(count(serde_json::json!({"input": ""})), 0);
    }

    #[test]
    fn count_tokens_short_input_never_rounds_to_zero() {
        // Every item that reaches the prompt now carries a role marker of at
        // least 4, so no `input` can land under the rounding threshold. Tools
        // are the one remaining path: they are appended to the request rather
        // than rendered as a message, so they carry no marker. A one-character
        // function name is 1, and 1 / 3 == 0 — but content was present, so
        // report 1.
        assert_eq!(
            count(serde_json::json!({"tools": [{"type": "function", "name": "a"}]})),
            1
        );
        // For comparison, the shortest possible input clears the guard on its
        // role marker alone: user (4) + "Hi" (2) == 6; 6 / 3 == 2.
        assert_eq!(count(serde_json::json!({"input": "Hi"})), 2);
    }

    #[test]
    fn count_tokens_instructions_contribute() {
        // system role (6) + "You are helpful." (16)
        //   + user role (4) + "Hi" (2) == 28; 28 / 3 == 9.
        assert_eq!(
            count(serde_json::json!({"input": "Hi", "instructions": "You are helpful."})),
            9
        );
    }

    #[test]
    fn count_tokens_scores_the_two_spellings_of_a_prompt_identically() {
        // `TryFrom<NvCreateResponse>` turns a top-level string `input` into a
        // user message and `instructions` into a system message, so these two
        // bodies build the same chat request and must score the same. Counting
        // the top-level forms as bare text undercounted them by the role
        // markers the item forms were already charged.
        assert_eq!(
            count(serde_json::json!({"input": "Hello"})),
            count(serde_json::json!({"input": [{"role": "user", "content": "Hello"}]})),
        );
        assert_eq!(
            count(serde_json::json!({
                "input": "Hello",
                "instructions": "You are helpful."
            })),
            count(serde_json::json!({"input": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"}
            ]})),
        );
    }

    #[test]
    fn count_tokens_easy_message_counts_role_and_content() {
        // user role (4) + "Hello" (5) == 9; 9 / 3 == 3.
        assert_eq!(
            count(serde_json::json!({"input": [{"role": "user", "content": "Hello"}]})),
            3
        );
    }

    #[test]
    fn count_tokens_structured_input_message() {
        // user role (4) + "Hello" (5) == 9; 9 / 3 == 3.
        assert_eq!(
            count(serde_json::json!({"input": [{
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Hello"}],
            }]})),
            3
        );
    }

    #[test]
    fn count_tokens_function_call_counts_name_and_arguments() {
        // A function call is rendered as a tool call on an assistant message:
        // assistant role (9) + "get_weather" (11) + r#"{"city":"SF"}"# (13)
        // == 33; 33 / 3 == 11.
        assert_eq!(
            count(serde_json::json!({"input": [{
                "type": "function_call",
                "call_id": "call_1",
                "name": "get_weather",
                "arguments": r#"{"city":"SF"}"#,
            }]})),
            11
        );
    }

    #[test]
    fn count_tokens_charges_one_assistant_marker_per_coalesced_turn() {
        // `convert_input_items_to_messages` accumulates assistant-side items
        // into one `PendingAssistant`, so two parallel tool calls are a single
        // assistant message. Charging a marker per item would invent a turn
        // that never reaches the prompt.
        let one = serde_json::json!({"input": [
            {"type": "function_call", "call_id": "c1", "name": "aa", "arguments": ""}
        ]});
        let two = serde_json::json!({"input": [
            {"type": "function_call", "call_id": "c1", "name": "aa", "arguments": ""},
            {"type": "function_call", "call_id": "c2", "name": "bb", "arguments": ""}
        ]});
        // assistant (9) + "aa" (2) == 11 → 3; adding "bb" (2) == 13 → 4.
        // The marker is paid once, not twice.
        assert_eq!(count(one), 3);
        assert_eq!(count(two), 4);

        // Assistant text, reasoning, and a tool call in one turn: still one
        // marker across all three.
        let mixed = serde_json::json!({"input": [
            {"role": "assistant", "content": "aa"},
            {"type": "reasoning", "summary": [{"type": "summary_text", "text": "bb"}]},
            {"type": "function_call", "call_id": "c1", "name": "cc", "arguments": ""}
        ]});
        assert_eq!(count(mixed), 5); // 9 + 2 + 2 + 2 == 15 → 5
    }

    #[test]
    fn count_tokens_reopens_the_assistant_turn_after_a_flush() {
        // A tool result ends the assistant turn, so the assistant items after
        // it are a second turn and pay a second marker.
        let two_turns = serde_json::json!({"input": [
            {"type": "function_call", "call_id": "c1", "name": "aa", "arguments": ""},
            {"type": "function_call_output", "call_id": "c1", "output": ""},
            {"type": "function_call", "call_id": "c2", "name": "bb", "arguments": ""}
        ]});
        // assistant (9) + "aa" (2) + tool (4) + assistant (9) + "bb" (2)
        // == 26; 26 / 3 == 8.
        assert_eq!(count(two_turns), 8);
    }

    #[test]
    fn count_tokens_item_reference_does_not_split_an_assistant_turn() {
        // The converter skips item references without flushing, so one sitting
        // between two tool calls must not make them look like two turns.
        let split = serde_json::json!({"input": [
            {"type": "function_call", "call_id": "c1", "name": "aa", "arguments": ""},
            {"type": "item_reference", "id": "item_abc"},
            {"type": "function_call", "call_id": "c2", "name": "bb", "arguments": ""}
        ]});
        let unsplit = serde_json::json!({"input": [
            {"type": "function_call", "call_id": "c1", "name": "aa", "arguments": ""},
            {"type": "function_call", "call_id": "c2", "name": "bb", "arguments": ""}
        ]});
        assert_eq!(count(split), count(unsplit));
    }

    #[test]
    fn count_tokens_unsupported_item_splits_an_assistant_turn() {
        // The converter flushes before skipping an unsupported variant
        // precisely so a later function call cannot coalesce across it. The
        // estimate has to agree, or it undercounts the second turn's marker.
        let across = serde_json::json!({"input": [
            {"type": "function_call", "call_id": "c1", "name": "aa", "arguments": ""},
            {"type": "web_search_call", "id": "ws_1", "status": "completed"},
            {"type": "function_call", "call_id": "c2", "name": "bb", "arguments": ""}
        ]});
        // Two turns: 9 + 2 + 9 + 2 == 22; 22 / 3 == 7.
        assert_eq!(count(across), 7);
    }

    #[test]
    fn count_tokens_function_call_output_counts_text() {
        // Rendered as its own tool-role message: tool role (4) + "sunny" (5)
        // == 9; 9 / 3 == 3.
        assert_eq!(
            count(serde_json::json!({"input": [{
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "sunny",
            }]})),
            3
        );
    }

    #[test]
    fn count_tokens_tools_contribute() {
        // "get_weather" (11) + "Get weather" (11) + r#"{"type":"object"}"# (17)
        // == 39; 39 / 3 == 13.
        assert_eq!(
            count(serde_json::json!({
                "input": "",
                "tools": [{
                    "type": "function",
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object"},
                }],
            })),
            13
        );
    }

    #[test]
    fn count_tokens_images_contribute_nothing() {
        // Only the text part is measured; the image part cannot be estimated
        // from character length.
        let with_image = count(serde_json::json!({"input": [{
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Describe this"},
                {"type": "input_image", "image_url": "https://example.com/a-very-long-url.png"},
            ],
        }]}));
        let without_image = count(serde_json::json!({"input": [{
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "Describe this"}],
        }]}));
        assert_eq!(with_image, without_image);
    }

    #[test]
    fn count_tokens_dropped_item_variants_cost_nothing() {
        // Dynamo's `convert_input_items_to_messages` flushes and skips these,
        // so they never reach the prompt. Counting their serialized form would
        // bill callers for JSON scaffolding the model never sees.
        for item in [
            serde_json::json!({"type": "web_search_call", "id": "ws_1", "status": "completed"}),
            serde_json::json!({
                "type": "computer_call",
                "call_id": "c_1",
                "id": "cu_1",
                "action": {"type": "screenshot"},
                "pending_safety_checks": [],
                "status": "completed",
            }),
        ] {
            assert_eq!(
                count(serde_json::json!({ "input": [item.clone()] })),
                0,
                "dropped item variant should not be counted: {item}"
            );
        }
    }

    #[test]
    fn count_tokens_counts_exactly_the_variants_the_converter_renders() {
        // Guards the coupling documented on `measure_item`: the explicit
        // arms are meant to be the same set Dynamo's converter handles. Each
        // variant is asserted on its own — a single assertion over an array of
        // all four would still pass with three of the arms deleted. If dynamo
        // grows support for another variant, this test should gain a case.
        for item in [
            serde_json::json!({"role": "user", "content": "Hello"}),
            serde_json::json!({
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Hello"}],
            }),
            serde_json::json!({
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Hi", "annotations": []}],
            }),
            serde_json::json!({
                "type": "function_call",
                "call_id": "c1",
                "name": "get_weather",
                "arguments": "{}",
            }),
            serde_json::json!({"type": "function_call_output", "call_id": "c1", "output": "sunny"}),
            serde_json::json!({
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "thinking"}],
            }),
        ] {
            assert!(
                count(serde_json::json!({ "input": [item.clone()] })) > 0,
                "rendered variant should be counted: {item}"
            );
        }
    }

    #[test]
    fn count_tokens_reasoning_counts_summary_only() {
        // The converter joins `summary` and drops everything else on the item,
        // so `content` and `encrypted_content` must not inflate the estimate.
        let summary_only = serde_json::json!({"input": [{
            "type": "reasoning",
            "summary": [{"type": "summary_text", "text": "thinking"}],
        }]});
        let with_dropped_fields = serde_json::json!({"input": [{
            "type": "reasoning",
            "summary": [{"type": "summary_text", "text": "thinking"}],
            "content": [{"type": "reasoning_text", "text": "a much longer private chain of thought"}],
            "encrypted_content": "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
        }]});

        // Reasoning rides on the pending assistant message:
        // assistant role (9) + "thinking" (8) == 17; 17 / 3 == 5.
        assert_eq!(count(summary_only.clone()), 5);
        assert_eq!(count(with_dropped_fields), count(summary_only));
    }

    #[test]
    fn count_tokens_hosted_tools_cost_nothing() {
        // `convert_tools` forwards only function tools; hosted tools are
        // dropped, so they must not inflate the estimate.
        assert_eq!(
            count(serde_json::json!({
                "input": "",
                "tools": [{"type": "web_search"}],
            })),
            0
        );
    }

    #[test]
    fn count_tokens_namespaced_tools_count_their_functions() {
        // `convert_tools` flattens namespaces to their bare function members,
        // so the members count and the namespace name does not.
        // "get_weather" (11) + "Get weather" (11) + r#"{"type":"object"}"# (17)
        // == 39; 39 / 3 == 13 — the same as the un-namespaced tool above.
        assert_eq!(
            count(serde_json::json!({
                "input": "",
                "tools": [{
                    "type": "namespace",
                    "name": "weather_ns",
                    "description": "Weather tools",
                    "tools": [{
                        "type": "function",
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {"type": "object"},
                    }],
                }],
            })),
            13
        );
    }

    #[test]
    fn count_tokens_item_reference_contributes_nothing() {
        // The referenced content is held server-side and is not in this request.
        assert_eq!(
            count(serde_json::json!({"input": [{"type": "item_reference", "id": "msg_1"}]})),
            0
        );
    }

    #[test]
    fn count_tokens_ignores_unsupported_stateful_fields() {
        // `previous_response_id` / `conversation` are accepted and disregarded
        // rather than rejected — this endpoint only estimates.
        assert_eq!(
            count(serde_json::json!({
                "model": "m",
                "input": "Hello, world!",
                "previous_response_id": "resp_abc123",
                "conversation": {"id": "conv_1"},
            })),
            5
        );
    }

    #[test]
    fn count_tokens_deserializes_the_litellm_request_shape() {
        // The exact body LiteLLM's CountTokens handler sends:
        // {model, input, instructions?, tools?} with chat tools already
        // flattened into the Responses shape.
        let request: CountInputTokensRequest = serde_json::from_value(serde_json::json!({
            "model": "dynamo/deepseek-ai/deepseek-v4-pro-sglang",
            "input": [{"role": "user", "content": "Hello"}],
            "instructions": "You are helpful.",
            "tools": [{
                "type": "function",
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {"type": "object"},
            }],
        }))
        .expect("LiteLLM request shape should deserialize");

        assert_eq!(
            request.model.as_deref(),
            Some("dynamo/deepseek-ai/deepseek-v4-pro-sglang")
        );
        assert!(matches!(request.input, InputParam::Items(ref items) if items.len() == 1));
        assert!(request.estimate_tokens() > 0);
    }

    #[test]
    fn count_tokens_accepts_explicit_null_input() {
        // `#[serde(default)]` covers an absent `input`; this covers the null
        // an emitter produces when it always writes the key.
        assert_eq!(count(serde_json::json!({"model": "m", "input": null})), 0);
        assert_eq!(
            count(serde_json::json!({
                "model": "m",
                "input": null,
                "instructions": "You are helpful."
            })),
            7
        );
    }

    #[test]
    fn count_tokens_drops_unparseable_tools_instead_of_failing() {
        // A Chat-Completions-shaped `custom` tool: `Tool` models the Responses
        // shape (`{"type": "custom", "name": ...}`), not the nested chat one.
        // It must not take the whole request down with it.
        let request: CountInputTokensRequest = serde_json::from_value(serde_json::json!({
            "model": "m",
            "input": "Hello, world!",
            "tools": [{"type": "custom", "custom": {"name": "x"}}],
        }))
        .expect("an unparseable tool should be dropped, not rejected");
        assert_eq!(request.tools.as_deref(), Some(&[][..]));
        // Same count as the identical body with no `tools` key at all: the
        // dropped tool was worth 0 either way.
        assert_eq!(request.estimate_tokens(), 5);
    }

    #[test]
    fn count_tokens_keeps_parseable_tools_alongside_dropped_ones() {
        // Dropping is per-entry, not all-or-nothing: a good function tool in
        // the same array still gets counted.
        let request: CountInputTokensRequest = serde_json::from_value(serde_json::json!({
            "model": "m",
            "input": "Hello, world!",
            "tools": [
                {"type": "custom", "custom": {"name": "x"}},
                {"type": "function", "name": "get_weather", "description": "Get weather"},
            ],
        }))
        .expect("a mixed tool array should deserialize");
        assert_eq!(request.tools.as_ref().map(Vec::len), Some(1));
        assert!(
            request.estimate_tokens()
                > count(serde_json::json!({"model": "m", "input": "Hello, world!"}))
        );
    }

    #[test]
    fn count_tokens_distinguishes_absent_tools_from_empty_tools() {
        // `None` and `Some([])` both score 0, but the field must round-trip
        // its presence: `skip_serializing_if` relies on the distinction.
        let absent: CountInputTokensRequest =
            serde_json::from_value(serde_json::json!({"input": "hi"})).unwrap();
        assert_eq!(absent.tools, None);
        let empty: CountInputTokensRequest =
            serde_json::from_value(serde_json::json!({"input": "hi", "tools": []})).unwrap();
        assert_eq!(empty.tools.as_deref(), Some(&[][..]));
        let null: CountInputTokensRequest =
            serde_json::from_value(serde_json::json!({"input": "hi", "tools": null})).unwrap();
        assert_eq!(null.tools, None);
    }

    #[test]
    fn count_tokens_response_serializes_to_the_openai_shape() {
        assert_eq!(
            serde_json::to_value(CountInputTokensResponse::new(42)).unwrap(),
            serde_json::json!({"object": "response.input_tokens", "input_tokens": 42})
        );
    }
}
