// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Request rules around one template render: `continue_final_message`, the
//! generation-prompt rule, a raw prompt in place of the template, tool-call
//! argument normalization, and a render failure as a client error.

use anyhow::Result;
use minijinja::value::Value;

use super::tool_arguments::normalize_tool_call_arguments;
use crate::{
    OAIChatLikeRequest, OAIPromptFormatter, PromptInput, PromptRenderError, RenderedPrompt,
    RenderedSegment, TextInput, TokenInput,
};

/// Per-request switches for [`render_chat_prompt`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ChatPromptOptions {
    /// Parse historical `tool_calls[*].function.arguments` strings into maps
    /// before rendering. See
    /// [`should_normalize_tool_call_arguments`](super::should_normalize_tool_call_arguments).
    pub normalize_tool_call_args: bool,
    /// The request's `continue_final_message`: leave the last message open, as
    /// HuggingFace `apply_chat_template(continue_final_message=True)` does.
    pub continue_final_message: bool,
    /// The request asked for its raw prompt in place of the template
    /// (Dynamo's `nvext.use_raw_prompt`).
    pub use_raw_prompt: bool,
}

/// Whether the template should add a generation prompt.
///
/// vLLM / HF: `continue_final_message` leaves the last turn open, which is
/// incompatible with appending a new generation prompt. Otherwise the request's
/// `add_generation_prompt` applies, and it defaults to true.
pub fn should_add_generation_prompt(
    add_generation_prompt: Option<bool>,
    continue_final_message: Option<bool>,
) -> bool {
    if continue_final_message == Some(true) {
        return false;
    }
    add_generation_prompt.unwrap_or(true)
}

/// Render a chat request with the request rules applied.
///
/// Returns `Ok(None)` when the request is not a single text prompt, so there is
/// nothing to render. With `use_raw_prompt`, `raw_prompt` supplies the prompt;
/// if it returns `None`, the template renders as usual.
///
/// # Errors
///
/// Every failure is a [`PromptRenderError::InvalidRequest`], because rendering
/// is driven by the request: the Dynamo frontend returns it as a 400, matching
/// vLLM. A formatter that already returned a `PromptRenderError` keeps its
/// message; any other formatter error keeps its full cause chain.
pub fn render_chat_prompt(
    formatter: &dyn OAIPromptFormatter,
    request: &dyn OAIChatLikeRequest,
    options: ChatPromptOptions,
    raw_prompt: impl FnOnce() -> Option<String>,
) -> Result<Option<RenderedPrompt>, PromptRenderError> {
    let continue_final = options.continue_final_message;
    if continue_final
        && formatter.media_message_order(request).is_some_and(|order| {
            order
                .last()
                .is_some_and(|&source| source != order.len() - 1)
        })
    {
        return Err(PromptRenderError::invalid_request(
            "Cannot continue the final message because the prompt formatter moves it before other messages",
        ));
    }
    let Some(prompt) = render_text_prompt(formatter, request, options, raw_prompt)? else {
        return Ok(None);
    };
    if !continue_final {
        return Ok(Some(prompt));
    }
    apply_continue_final_message(prompt)
        .map_err(|error| PromptRenderError::invalid_request(format!("{error:#}")))
        .map(Some)
}

fn render_text_prompt(
    formatter: &dyn OAIPromptFormatter,
    request: &dyn OAIChatLikeRequest,
    options: ChatPromptOptions,
    raw_prompt: impl FnOnce() -> Option<String>,
) -> Result<Option<RenderedPrompt>, PromptRenderError> {
    if let PromptInput::Text(_) = request.prompt_input_type()
        && let Some(TextInput::Single(_)) = request.extract_text()
    {
        let formatted_prompt = if options.use_raw_prompt {
            match raw_prompt() {
                Some(prompt) => RenderedPrompt::text(prompt),
                None => {
                    tracing::warn!("Raw prompt requested but not available");
                    render_template(formatter, request, options)?
                }
            }
        } else {
            render_template(formatter, request, options)?
        };
        Ok(Some(formatted_prompt))
    } else {
        Ok(None)
    }
}

/// Render through the template. The messages are converted for the wrapper
/// only here, so a raw prompt or a request with nothing to render never pays
/// for that copy.
fn render_template(
    formatter: &dyn OAIPromptFormatter,
    request: &dyn OAIChatLikeRequest,
    options: ChatPromptOptions,
) -> Result<RenderedPrompt, PromptRenderError> {
    if !options.normalize_tool_call_args && !options.continue_final_message {
        return formatter
            .render_prompt(request)
            .map_err(map_prompt_render_error);
    }
    let request = NormalizedArgsRequest::new(
        request,
        options.normalize_tool_call_args,
        options.continue_final_message,
    )
    .map_err(map_prompt_render_error)?;
    formatter
        .render_prompt(&request)
        .map_err(map_prompt_render_error)
}

/// A misconfigured template can also fail here, for instance a `chat_template`
/// map that omits the `tool_use` key, so log the cause chain before it is
/// flattened into the client-facing message.
pub(super) fn map_prompt_render_error(error: anyhow::Error) -> PromptRenderError {
    tracing::debug!(?error, "Chat template rendering failed");
    let message = match error.downcast_ref::<PromptRenderError>() {
        Some(PromptRenderError::InvalidRequest(message)) => message.clone(),
        None => format!("{error:#}"),
    };
    PromptRenderError::invalid_request(message)
}

/// Wraps a request so the template sees its messages after the per-request
/// transforms: historical `function.arguments` parsed into maps (GLM-5.2 and
/// later), and HuggingFace's continue-final-message marker on the last
/// message.
///
/// The transformed messages start from the same JSON the renderer would read
/// without the wrapper ([`crate::messages_to_json`]), and the wrapper returns
/// them as its template-messages override, so every formatter sees them
/// whether it reads the override or `messages()`. `typed_messages` stays
/// `None`, because the typed slice cannot carry the transforms. Every other
/// method forwards to the inner request.
///
/// # Errors
///
/// [`NormalizedArgsRequest::new`] fails when the messages cannot be converted
/// to JSON. The caller returns that as an invalid request, as the formatter
/// would without the wrapper, though a formatter may word its own error
/// differently.
pub(super) struct NormalizedArgsRequest<'a> {
    inner: &'a dyn OAIChatLikeRequest,
    messages: serde_json::Value,
}

impl<'a> NormalizedArgsRequest<'a> {
    pub(super) fn new(
        inner: &'a dyn OAIChatLikeRequest,
        normalize_tool_call_args: bool,
        continue_final_message: bool,
    ) -> Result<Self> {
        let mut messages = crate::messages_to_json(inner)?;
        if normalize_tool_call_args && let Err(e) = normalize_tool_call_arguments(&mut messages) {
            tracing::error!(
                error = %e,
                "tool_call arguments normalization failed; template rendering may fail \
                 if it calls .items() on a string"
            );
        }
        if continue_final_message && let Err(e) = append_continue_final_message_tag(&mut messages) {
            tracing::debug!(
                error = %e,
                "continue_final_message marker not appended; truncation will reject the request"
            );
        }
        Ok(Self { inner, messages })
    }
}

impl OAIChatLikeRequest for NormalizedArgsRequest<'_> {
    fn model(&self) -> String {
        self.inner.model()
    }

    fn messages(&self) -> Value {
        Value::from_serialize(&self.messages)
    }

    fn template_messages_json_override(&self) -> Option<serde_json::Value> {
        Some(self.messages.clone())
    }

    fn tools(&self) -> Option<Value> {
        self.inner.tools()
    }

    fn template_tools(&self) -> Option<Value> {
        self.inner.template_tools()
    }

    fn tool_choice(&self) -> Option<Value> {
        self.inner.tool_choice()
    }

    fn response_format(&self) -> Option<Value> {
        self.inner.response_format()
    }

    fn reasoning_effort(&self) -> Option<Value> {
        self.inner.reasoning_effort()
    }

    fn should_add_generation_prompt(&self) -> bool {
        self.inner.should_add_generation_prompt()
    }

    fn chat_template_args(&self) -> Option<&std::collections::HashMap<String, serde_json::Value>> {
        self.inner.chat_template_args()
    }

    fn prompt_input_type(&self) -> PromptInput {
        self.inner.prompt_input_type()
    }

    fn extract_tokens(&self) -> Option<TokenInput> {
        self.inner.extract_tokens()
    }

    fn extract_text(&self) -> Option<TextInput> {
        self.inner.extract_text()
    }

    fn mm_processor_kwargs(&self) -> Option<&serde_json::Value> {
        self.inner.mm_processor_kwargs()
    }
}

const CONTINUE_FINAL_MESSAGE_NOT_FOUND: &str =
    "Unable to continue the final message because it was not found in the rendered chat template.";

/// HuggingFace `render_jinja_template` marker. Appended to last-message content
/// before render so truncation uniquely identifies the final turn even when the
/// template rewrites or repeats earlier text.
pub(crate) const CONTINUE_FINAL_MESSAGE_TAG: &str = "CONTINUE_FINAL_MESSAGE_TAG ";

/// Append [`CONTINUE_FINAL_MESSAGE_TAG`] to the last message's text content.
/// Array content uses the last non-empty `text` part, matching Transformers.
pub(crate) fn append_continue_final_message_tag(messages: &mut serde_json::Value) -> Result<()> {
    let Some(last) = messages.as_array_mut().and_then(|arr| arr.last_mut()) else {
        anyhow::bail!(CONTINUE_FINAL_MESSAGE_NOT_FOUND);
    };
    match last.get_mut("content") {
        Some(serde_json::Value::String(text)) if !text.is_empty() => {
            text.push_str(CONTINUE_FINAL_MESSAGE_TAG);
            Ok(())
        }
        Some(serde_json::Value::Array(parts)) => {
            for part in parts.iter_mut().rev() {
                if let Some(serde_json::Value::String(text)) = part.get_mut("text") {
                    if text.is_empty() {
                        continue;
                    }
                    text.push_str(CONTINUE_FINAL_MESSAGE_TAG);
                    return Ok(());
                }
            }
            anyhow::bail!(CONTINUE_FINAL_MESSAGE_NOT_FOUND);
        }
        _ => anyhow::bail!(CONTINUE_FINAL_MESSAGE_NOT_FOUND),
    }
}

/// HuggingFace `apply_chat_template(continue_final_message=True)`: after render,
/// drop the marker and any tokens after it so the prompt ends at the last
/// message text.
///
/// The renderer must have seen [`CONTINUE_FINAL_MESSAGE_TAG`] in the last
/// message. Searching for the raw last-message string cannot prove the match
/// belongs to the final turn (a template may rewrite or repeat earlier copies).
///
/// When the renderer returned segment boundaries (Kimi K3 XTML), keep them so
/// tokenization still uses `encode_segments` instead of flattening to plain text.
/// Missing marker, empty content, or a template that dropped the marker is an
/// invalid request rather than silently returning the closed prompt.
pub(crate) fn apply_continue_final_message(prompt: RenderedPrompt) -> Result<RenderedPrompt> {
    let rendered = prompt.as_str();
    let tag = CONTINUE_FINAL_MESSAGE_TAG;
    let tag_name = tag.trim_end();
    let Some(tag_loc) = rendered.rfind(tag_name) else {
        anyhow::bail!(CONTINUE_FINAL_MESSAGE_NOT_FOUND);
    };
    // Transformers: if the full tag (including trailing space) survived, cut
    // at the tag. If the template trimmed that space, also rstrip the prefix.
    let end = if rendered[tag_loc..].starts_with(tag) {
        tag_loc
    } else {
        rendered[..tag_loc].trim_end().len()
    };
    Ok(truncate_rendered_prompt(prompt, end))
}

/// Truncate `prompt` to the first `end` bytes of `as_str()`, preserving
/// `RenderedSegment` trust boundaries when they are present.
fn truncate_rendered_prompt(prompt: RenderedPrompt, end: usize) -> RenderedPrompt {
    let Some(segments) = prompt.segments() else {
        return RenderedPrompt::text(prompt.as_str()[..end].to_string());
    };
    let mut out = Vec::new();
    let mut offset = 0usize;
    for seg in segments {
        let next = offset + seg.text.len();
        if next <= end {
            if !seg.text.is_empty() {
                out.push(seg.clone());
            }
            offset = next;
            if offset == end {
                break;
            }
            continue;
        }
        if offset < end {
            let keep = end - offset;
            if keep > 0 && keep <= seg.text.len() && seg.text.is_char_boundary(keep) {
                out.push(RenderedSegment {
                    text: seg.text[..keep].to_string(),
                    allow_special: seg.allow_special,
                });
            }
        }
        break;
    }
    if out.is_empty() {
        RenderedPrompt::text(String::new())
    } else {
        RenderedPrompt::segmented(out)
    }
}
