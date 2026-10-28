// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A frozen copy of the Dynamo frontend's request rules, used as the oracle
//! for the parity tests.
//!
//! Each block is copied from ai-dynamo/dynamo at
//! `cffc4fa75c15017bcd32e62f7d0fe4bad468c35d` (the cited `path:line` ranges
//! match `45e30e07`). The edits are only what it takes to compile without
//! dynamo-llm: module paths, `pub(crate)` so the tests can call the copied
//! functions, and the changes marked `ORACLE:`. Do not change the copied
//! logic: the parity tests prove the moved code still matches it.
#![allow(dead_code, clippy::all)]

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;
use dynamo_renderer::{
    OAIChatLikeRequest, OAIPromptFormatter, PromptInput, PromptRenderError, RenderedPrompt,
    RenderedSegment, TextInput, may_be_fix_tool_schema,
};
use minijinja::value::Value;

// ORACLE: stand-ins for the dynamo-llm types that the copied code reads.

/// The fields of dynamo-llm's `NvCreateChatCompletionRequest` that the
/// request rules read.
#[derive(Clone, Debug, Default, serde::Deserialize)]
pub struct NvCreateChatCompletionRequest {
    #[serde(flatten)]
    pub inner: dynamo_protocols::types::CreateChatCompletionRequest,
    #[serde(default, alias = "chat_template_kwargs")]
    pub chat_template_args: Option<HashMap<String, serde_json::Value>>,
    #[serde(default)]
    pub thinking: Option<serde_json::Value>,
    #[serde(flatten)]
    pub common: CommonExt,
    #[serde(default)]
    pub nvext: Option<NvExt>,
}

#[derive(Clone, Debug, Default, serde::Deserialize)]
pub struct CommonExt {
    #[serde(default)]
    pub guided_json: Option<serde_json::Value>,
    #[serde(default)]
    pub guided_regex: Option<String>,
    #[serde(default)]
    pub guided_choice: Option<Vec<String>>,
    #[serde(default)]
    pub guided_grammar: Option<String>,
    #[serde(default)]
    pub add_generation_prompt: Option<bool>,
    #[serde(default)]
    pub continue_final_message: Option<bool>,
}

#[derive(Clone, Debug, Default, serde::Deserialize)]
pub struct NvExt {
    #[serde(default)]
    pub use_raw_prompt: Option<bool>,
}

/// dynamo-llm's `NvExtProvider`, reduced to what `apply_template_inner` reads.
pub trait NvExtProvider {
    fn nvext(&self) -> Option<&NvExt>;
    fn raw_prompt(&self) -> Option<String>;
}

/// dynamo-llm's `CommonExtProvider`, reduced to what `apply_template` reads.
pub trait CommonExtProvider {
    fn get_continue_final_message(&self) -> Option<bool>;
}

impl NvExtProvider for NvCreateChatCompletionRequest {
    fn nvext(&self) -> Option<&NvExt> {
        self.nvext.as_ref()
    }

    // dynamo lib/llm/src/protocols/openai/chat_completions.rs:447-449
    fn raw_prompt(&self) -> Option<String> {
        None
    }
}

impl CommonExtProvider for NvCreateChatCompletionRequest {
    fn get_continue_final_message(&self) -> Option<bool> {
        self.common.continue_final_message
    }
}

// dynamo lib/llm/src/preprocessor.rs:1698-1706 (the NvExtProvider forwarding)
impl<R: NvExtProvider> NvExtProvider for NormalizedArgsRequest<'_, R> {
    fn nvext(&self) -> Option<&NvExt> {
        self.inner.nvext()
    }

    fn raw_prompt(&self) -> Option<String> {
        self.inner.raw_prompt()
    }
}

/// dynamo-llm's 400 error: the parity tests compare its message.
#[derive(Debug)]
pub struct InvalidArgument(pub String);

impl std::fmt::Display for InvalidArgument {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for InvalidArgument {}

pub(crate) fn invalid_argument_error(message: impl Into<String>) -> anyhow::Error {
    InvalidArgument(message.into()).into()
}

/// The parts of `OpenAIPreprocessor` that the copied methods read.
pub struct OpenAIPreprocessor {
    pub formatter: Arc<dyn OAIPromptFormatter>,
    pub normalize_tool_call_args: bool,
    /// ORACLE: the result of `unified_parser::selected_family`, which did not move.
    pub selected_family: Option<&'static str>,
}

// ---- the chat request's OAIChatLikeRequest impl ----
// dynamo lib/llm/src/preprocessor/prompt.rs:236-298
impl OAIChatLikeRequest for NvCreateChatCompletionRequest {
    fn model(&self) -> String {
        self.inner.model.clone()
    }

    fn messages(&self) -> Value {
        let messages_json = serde_json::to_value(&self.inner.messages).unwrap();
        Value::from_serialize(&messages_json)
    }

    fn typed_messages(&self) -> Option<&[dynamo_protocols::types::ChatCompletionRequestMessage]> {
        Some(self.inner.messages.as_slice())
    }

    fn tools(&self) -> Option<Value> {
        if self.inner.tools.is_none() {
            None
        } else {
            // Try to fix the tool schema if it is missing type and properties
            Some(may_be_fix_tool_schema(
                serde_json::to_value(&self.inner.tools).unwrap(),
            )?)
        }
    }

    fn tool_choice(&self) -> Option<Value> {
        if self.inner.tool_choice.is_none() {
            None
        } else {
            Some(Value::from_serialize(&self.inner.tool_choice))
        }
    }

    fn response_format(&self) -> Option<Value> {
        self.inner
            .response_format
            .as_ref()
            .map(Value::from_serialize)
    }

    fn should_add_generation_prompt(&self) -> bool {
        // vLLM / HF: continue_final_message leaves the last turn open, which is
        // incompatible with appending a new generation prompt. Validation already
        // rejects the omitted/true add_generation_prompt combination; keep this
        // guard so internal callers that skip ValidateRequest still match HF.
        if self.common.continue_final_message == Some(true) {
            return false;
        }
        self.common.add_generation_prompt.unwrap_or(true)
    }

    fn extract_text(&self) -> Option<TextInput> {
        Some(TextInput::Single(String::new()))
    }

    fn chat_template_args(&self) -> Option<&std::collections::HashMap<String, serde_json::Value>> {
        self.chat_template_args.as_ref()
    }

    fn mm_processor_kwargs(&self) -> Option<&serde_json::Value> {
        self.inner.mm_processor_kwargs.as_ref()
    }
}

// ---- truthy ----
// dynamo lib/truthy/src/lib.rs:20-34
pub fn is_truthy(val: &str) -> bool {
    matches!(
        val.trim().to_lowercase().as_str(),
        "1" | "true" | "on" | "yes"
    )
}

/// Check if a string is falsey: `0 | false | off | no` or empty,
/// case-insensitive, surrounding whitespace ignored.
pub fn is_falsey(val: &str) -> bool {
    matches!(
        val.trim().to_lowercase().as_str(),
        "" | "0" | "false" | "off" | "no"
    )
}
// dynamo lib/truthy/src/lib.rs:68-78
pub fn parse_bool_opt(val: &str) -> Option<bool> {
    if is_truthy(val) {
        Some(true)
    } else if val.trim().is_empty() {
        None
    } else if is_falsey(val) {
        Some(false)
    } else {
        None
    }
}
// ---- tool arguments ----
// dynamo lib/llm/src/preprocessor/prompt.rs:41-234
/// Parse a JSON object string into a serde_json Map, preserving the exact
/// spelling of numbers that cannot be round-tripped through f64 without loss.
/// Numbers that survive f64 conversion unchanged are kept as `Value::Number`;
/// numbers that would be corrupted (e.g. integers with >15 significant digits)
/// are kept as `Value::String` so the original value is available to templates.
/// Returns `None` when the top-level JSON value is not an object.
pub(crate) fn parse_args_object(
    s: &str,
) -> anyhow::Result<Option<serde_json::Map<String, serde_json::Value>>> {
    // First pass: parse the top-level value to check it is an object.
    let top: serde_json::Value = serde_json::from_str(s)?;
    let top_obj = match top {
        serde_json::Value::Object(m) => m,
        _ => return Ok(None),
    };

    // Second pass: re-parse each entry's value via its RawValue to detect
    // numbers that serde_json would corrupt through f64.
    // We use a raw string-to-rawvalue helper to get the exact bytes.
    let mut out = serde_json::Map::with_capacity(top_obj.len());
    for (k, cooked) in top_obj {
        // For non-numbers, use the already-parsed value directly.
        if !cooked.is_number() {
            out.insert(k, cooked);
            continue;
        }
        // For numbers: find the raw token in the source string so we can
        // verify that serde_json's f64 representation matches the original.
        // We do this by re-serializing the cooked number and comparing.
        let cooked_str = cooked.to_string();
        if let Some(raw_token) = extract_value_token(s, &k) {
            let raw_trimmed = raw_token.trim();
            if raw_trimmed != cooked_str.as_str() {
                // Precision was lost (e.g. 30-digit integer → 1.23e+29).
                // Keep the original spelling as a string.
                out.insert(k, serde_json::Value::String(raw_trimmed.to_string()));
                continue;
            }
        }
        out.insert(k, cooked);
    }
    Ok(Some(out))
}

/// Extract the raw JSON token for `key` from a JSON object string `s`.
/// Returns `None` when the key is not found or parsing fails.
pub(crate) fn extract_value_token<'a>(s: &'a str, key: &str) -> Option<&'a str> {
    // Build the escaped key as it appears in JSON.
    let key_json = serde_json::to_string(key).ok()?;
    // Find `"key":` in the source.
    let needle = format!("{}:", key_json);
    let start = s.find(needle.as_str())?;
    let after_colon = s[start + needle.len()..].trim_start();
    // Measure the extent of the JSON token.
    let end = token_end(after_colon)?;
    let source_offset = s.len() - s[start + needle.len()..].len()
        + (after_colon.as_ptr() as usize - s[start + needle.len()..].as_ptr() as usize);
    Some(&s[source_offset..source_offset + end])
}

/// Return the byte length of the first JSON token in `s`.
pub(crate) fn token_end(s: &str) -> Option<usize> {
    let s = s.trim_start();
    let first = s.chars().next()?;
    match first {
        '"' => {
            let mut i = 1;
            let b = s.as_bytes();
            while i < b.len() {
                if b[i] == b'\\' {
                    i += 2;
                } else if b[i] == b'"' {
                    return Some(i + 1);
                } else {
                    i += 1;
                }
            }
            None
        }
        '{' | '[' => {
            // Skip balanced braces/brackets — good enough for flat objects.
            let (open, close) = if first == '{' {
                (b'{', b'}')
            } else {
                (b'[', b']')
            };
            let mut depth = 0i32;
            let mut in_str = false;
            let mut escape = false;
            for (i, &b) in s.as_bytes().iter().enumerate() {
                if escape {
                    escape = false;
                    continue;
                }
                if in_str {
                    if b == b'\\' {
                        escape = true;
                    } else if b == b'"' {
                        in_str = false;
                    }
                } else {
                    match b {
                        b'"' => in_str = true,
                        b if b == open => depth += 1,
                        b if b == close => {
                            depth -= 1;
                            if depth == 0 {
                                return Some(i + 1);
                            }
                        }
                        _ => {}
                    }
                }
            }
            None
        }
        _ => {
            // Number, bool, null — ends at first delimiter.
            let end = s
                .find(|c: char| c == ',' || c == '}' || c == ']' || c.is_whitespace())
                .unwrap_or(s.len());
            if end == 0 { None } else { Some(end) }
        }
    }
}

/// Parse `tool_calls[*].function.arguments` from JSON string to object before
/// handing messages to MiniJinja. Some chat templates (e.g. GLM-5.2) call
/// `.items()` on arguments and require a dict; the OpenAI wire format carries
/// arguments as a JSON string. This matches TRT-LLM's chat_utils.py behaviour.
///
/// Large integer values in arguments are preserved exactly because `dynamo-llm`
/// enables the `serde_json/arbitrary_precision` feature.
///
/// Returns an error if any tool_call's arguments string contains invalid JSON,
/// so callers can surface a 400-style error instead of fabricating empty args.
pub(crate) fn normalize_tool_call_arguments(
    messages_json: &mut serde_json::Value,
) -> anyhow::Result<()> {
    let Some(messages) = messages_json.as_array_mut() else {
        return Ok(());
    };
    for message in messages {
        let Some(tool_calls) = message
            .get_mut("tool_calls")
            .and_then(serde_json::Value::as_array_mut)
        else {
            continue;
        };
        for tc in tool_calls.iter_mut() {
            let Some(args_str) = tc.pointer("/function/arguments").and_then(|v| v.as_str()) else {
                continue;
            };
            // Empty string is a valid no-argument call; normalise to {}.
            if args_str.is_empty() {
                if let Some(obj) = tc
                    .get_mut("function")
                    .and_then(serde_json::Value::as_object_mut)
                {
                    obj.insert(
                        "arguments".to_string(),
                        serde_json::Value::Object(serde_json::Map::new()),
                    );
                }
                continue;
            }
            let value = match parse_args_object(args_str) {
                Ok(Some(map)) => serde_json::Value::Object(map),
                Ok(None) => {
                    // Scalar or array — GLM's .items() would panic at render time.
                    tracing::warn!(
                        args_len = args_str.len(),
                        "tool_call arguments parsed to a non-object; \
                         substituting {{}} for template safety"
                    );
                    serde_json::Value::Object(serde_json::Map::new())
                }
                Err(e) => {
                    anyhow::bail!(
                        "tool_call arguments are not valid JSON (len={}): {e}",
                        args_str.len()
                    );
                }
            };
            if let Some(obj) = tc
                .get_mut("function")
                .and_then(serde_json::Value::as_object_mut)
            {
                obj.insert("arguments".to_string(), value);
            }
        }
    }
    Ok(())
}
// ---- continue_final_message ----
// dynamo lib/llm/src/preprocessor/prompt.rs:489-589
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
pub(crate) fn truncate_rendered_prompt(prompt: RenderedPrompt, end: usize) -> RenderedPrompt {
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

// ---- guided decoding ----
// dynamo lib/llm/src/preprocessor/tool_choice.rs:264-275
/// Whether top-level `guided_*` fields install a grammar; mirrors
/// `GuidedDecodingOptions::from_optional`.
pub fn has_explicit_guided_decoding(request: &NvCreateChatCompletionRequest) -> bool {
    request.common.guided_json.is_some()
        || request.common.guided_regex.is_some()
        || request
            .common
            .guided_choice
            .as_ref()
            .is_some_and(|v| !v.is_empty())
        || request.common.guided_grammar.is_some()
}
// ---- request wrapper ----
// dynamo lib/llm/src/preprocessor.rs:1603-1670
/// Thin wrapper that prepares messages for MiniJinja. Normalizes historical
/// `function.arguments` when the model opts in (GLM-5.2), and appends
/// HuggingFace's unique continue-final-message marker when that flag is set.
/// Other trait methods delegate to the inner request, except typed_messages:
/// rendering must use the transformed messages rather than the original slice.
struct NormalizedArgsRequest<'a, R> {
    inner: &'a R,
    normalize_tool_call_args: bool,
    continue_final_message: bool,
}

impl<R: OAIChatLikeRequest> OAIChatLikeRequest for NormalizedArgsRequest<'_, R> {
    fn model(&self) -> String {
        self.inner.model()
    }

    fn messages(&self) -> minijinja::value::Value {
        let mut json = serde_json::to_value(self.inner.typed_messages().unwrap_or_default())
            .unwrap_or_default();
        if self.normalize_tool_call_args
            && let Err(e) = normalize_tool_call_arguments(&mut json)
        {
            tracing::error!(
                error = %e,
                "tool_call arguments normalization failed; template rendering may fail \
                 if it calls .items() on a string"
            );
        }
        if self.continue_final_message
            && let Err(e) = append_continue_final_message_tag(&mut json)
        {
            tracing::debug!(
                error = %e,
                "continue_final_message marker not appended; truncation will reject the request"
            );
        }
        minijinja::value::Value::from_serialize(&json)
    }

    fn tools(&self) -> Option<minijinja::value::Value> {
        self.inner.tools()
    }

    fn tool_choice(&self) -> Option<minijinja::value::Value> {
        self.inner.tool_choice()
    }

    fn response_format(&self) -> Option<minijinja::value::Value> {
        self.inner.response_format()
    }

    fn should_add_generation_prompt(&self) -> bool {
        self.inner.should_add_generation_prompt()
    }

    fn extract_text(&self) -> Option<TextInput> {
        self.inner.extract_text()
    }

    fn chat_template_args(&self) -> Option<&std::collections::HashMap<String, serde_json::Value>> {
        self.inner.chat_template_args()
    }

    fn mm_processor_kwargs(&self) -> Option<&serde_json::Value> {
        self.inner.mm_processor_kwargs()
    }
}
impl OpenAIPreprocessor {
    // dynamo lib/llm/src/preprocessor.rs:2125-2143
    pub(crate) fn has_request_thinking_control(
        chat_template_args: Option<&HashMap<String, serde_json::Value>>,
    ) -> bool {
        chat_template_args.is_some_and(|args| {
            [
                "thinking",
                "enable_thinking",
                "thinking_mode",
                "reasoning_effort",
            ]
            .iter()
            .any(|key| args.contains_key(*key))
        })
    }

    pub(crate) fn request_has_client_thinking_control(
        request: &NvCreateChatCompletionRequest,
    ) -> bool {
        request.thinking.is_some()
            || Self::has_request_thinking_control(request.chat_template_args.as_ref())
    }
    // dynamo lib/llm/src/preprocessor.rs:2161-2194
    pub(crate) fn apply_default_thinking_mode_value(
        default_mode: Option<&str>,
        chat_template_args: &mut Option<HashMap<String, serde_json::Value>>,
        thinking_control_from_client: bool,
    ) {
        if thinking_control_from_client {
            return;
        }
        let Some(default_mode) = default_mode else {
            return;
        };
        let enabled = match default_mode {
            "enabled" => true,
            "disabled" => false,
            other => {
                tracing::warn!(
                    default_thinking_mode = other,
                    "Ignoring invalid runtime_config default_thinking_mode; expected 'enabled' or 'disabled'"
                );
                return;
            }
        };

        let args = chat_template_args.get_or_insert_with(HashMap::new);
        args.insert("thinking".to_string(), serde_json::Value::Bool(enabled));
        args.insert(
            "enable_thinking".to_string(),
            serde_json::Value::Bool(enabled),
        );
        args.insert(
            "thinking_mode".to_string(),
            serde_json::Value::String(if enabled { "enabled" } else { "disabled" }.to_string()),
        );
    }
    // dynamo lib/llm/src/preprocessor.rs:2241-2248
    pub(crate) fn has_structured_response_format<R: OAIChatLikeRequest>(request: &R) -> bool {
        request.response_format().is_some_and(|format| {
            format
                .get_attr("type")
                .ok()
                .is_some_and(|kind| kind.as_str().is_some_and(|kind| kind != "text"))
        })
    }
    // dynamo lib/llm/src/preprocessor.rs:2328-2530
    pub(crate) fn normalize_thinking_arg_with_source(
        request: &mut NvCreateChatCompletionRequest,
        reasoning_parser: Option<&str>,
        tool_call_parser: Option<&str>,
        thinking_control_from_client: bool,
    ) {
        let constrained_generation = !thinking_control_from_client
            && Self::is_minimax_m3_family(reasoning_parser, tool_call_parser)
            && Self::has_minimax_m3_constrained_generation(request);
        Self::normalize_thinking_args_with_source(
            &mut request.chat_template_args,
            reasoning_parser,
            tool_call_parser,
            thinking_control_from_client,
            constrained_generation,
        );
    }

    pub(crate) fn normalize_thinking_args_with_source(
        chat_template_args: &mut Option<HashMap<String, serde_json::Value>>,
        reasoning_parser: Option<&str>,
        tool_call_parser: Option<&str>,
        thinking_control_from_client: bool,
        constrained_generation: bool,
    ) {
        let normalized =
            Self::normalize_thinking_aliases(chat_template_args.as_ref(), reasoning_parser);

        if Self::is_minimax_m3_family(reasoning_parser, tool_call_parser) {
            Self::normalize_minimax_m3_thinking_mode(
                chat_template_args,
                normalized,
                thinking_control_from_client,
                constrained_generation,
            );
            return;
        }

        Self::apply_normalized_thinking_aliases(chat_template_args, normalized);
    }

    pub(crate) fn normalize_thinking_aliases(
        chat_template_args: Option<&HashMap<String, serde_json::Value>>,
        reasoning_parser: Option<&str>,
    ) -> Option<bool> {
        chat_template_args
            .and_then(|args| {
                // Normalize public aliases in array order: `thinking`,
                // `enable_thinking`, then `thinking_mode`.
                // Preserve "adaptive" as tri-state by returning None.
                for key in ["thinking", "enable_thinking", "thinking_mode"] {
                    match args.get(key) {
                        Some(serde_json::Value::Bool(b)) => {
                            // `thinking_bool_from_args` ignores `thinking_mode`,
                            // so read that bool directly.
                            if key == "thinking_mode" {
                                return Some(*b);
                            }
                            return dynamo_renderer::thinking_bool_from_args(Some(args));
                        }
                        Some(serde_json::Value::String(value)) => {
                            // Parse `thinking_mode` explicitly so
                            // "enabled"/"disabled" and boolean spellings
                            // behave predictably. Preserve unrecognized strings
                            // such as "adaptive" or DeepSeek's "chat"/"thinking".
                            if key == "thinking_mode" {
                                if value.eq_ignore_ascii_case("enabled") {
                                    return Some(true);
                                }
                                if value.eq_ignore_ascii_case("disabled") {
                                    return Some(false);
                                }
                                if let Some(b) = parse_bool_opt(value) {
                                    return Some(b);
                                }
                                return None;
                            }
                            return Some(is_truthy(value));
                        }
                        Some(serde_json::Value::Number(value)) => {
                            if let Some(value) = value.as_f64() {
                                return Some(value != 0.0);
                            }
                        }
                        _ => {}
                    }
                }
                None
            })
            .or_else(|| {
                matches!(reasoning_parser, Some("kimi_k25" | "kimi_k3" | "kimi-k3")).then_some(true)
            })
    }

    pub(crate) fn normalize_minimax_m3_thinking_mode(
        chat_template_args: &mut Option<HashMap<String, serde_json::Value>>,
        normalized: Option<bool>,
        thinking_control_from_client: bool,
        constrained_generation: bool,
    ) {
        // MiniMax M3 defaults `thinking_mode` to "adaptive", which breaks
        // constrained JSON/schema and forced-tool generation. If the client
        // did not set thinking controls, force "disabled" for those requests.
        let explicit_thinking_mode_is_adaptive = chat_template_args
            .as_ref()
            .and_then(|args| args.get("thinking_mode"))
            .and_then(|v| v.as_str())
            .is_some_and(|s| s.eq_ignore_ascii_case("adaptive"));
        if !thinking_control_from_client && constrained_generation {
            let args = chat_template_args.get_or_insert_default();
            args.insert(
                "thinking_mode".to_string(),
                serde_json::Value::String("disabled".to_string()),
            );
            args.insert("thinking".to_string(), serde_json::Value::Bool(false));
            args.insert(
                "enable_thinking".to_string(),
                serde_json::Value::Bool(false),
            );
            return;
        }

        Self::apply_normalized_thinking_aliases(chat_template_args, normalized);

        let Some(normalized) = normalized else {
            return;
        };

        // Canonicalize for M3 because its template only treats exact
        // "disabled" as off. Preserve explicit "adaptive" as client intent.
        if !explicit_thinking_mode_is_adaptive {
            let args = chat_template_args.get_or_insert_default();
            args.insert(
                "thinking_mode".to_string(),
                serde_json::Value::String(
                    if normalized { "enabled" } else { "disabled" }.to_string(),
                ),
            );
        }
    }

    pub(crate) fn apply_normalized_thinking_aliases(
        chat_template_args: &mut Option<HashMap<String, serde_json::Value>>,
        normalized: Option<bool>,
    ) {
        let Some(normalized) = normalized else {
            return;
        };
        let args = chat_template_args.get_or_insert_default();
        args.insert("thinking".to_string(), serde_json::Value::Bool(normalized));
        args.insert(
            "enable_thinking".to_string(),
            serde_json::Value::Bool(normalized),
        );
    }

    /// True when the deployment renders MiniMax M3's `thinking_mode` template.
    /// Either parser is sufficient, and tool-call aliases include `_nom`.
    pub(crate) fn is_minimax_m3_family(
        reasoning_parser: Option<&str>,
        tool_call_parser: Option<&str>,
    ) -> bool {
        matches!(reasoning_parser, Some("minimax_m3") | Some("minimax-m3"))
            || matches!(
                tool_call_parser,
                Some("minimax_m3")
                    | Some("minimax-m3")
                    | Some("minimax_m3_nom")
                    | Some("minimax-m3-nom")
            )
    }

    /// A structured `response_format`, top-level `guided_*`, and a forced tool
    /// choice each constrain output from the first token.
    ///
    /// With no client thinking control, a request this returns true for
    /// renders thinking disabled (see `normalize_minimax_m3_thinking_mode`),
    /// even when the grammar itself contains `<mm:think>...</mm:think>`. The
    /// reasoning parser is then skipped, and a generated think block stays
    /// verbatim in `content`. A client that wants that block split into
    /// `reasoning_content` must send a thinking control that does not disable
    /// thinking, preferably an adaptive one (`thinking_mode: "adaptive"` or
    /// `thinking: {type: "adaptive"}`): its prompt prefills no marker for the
    /// grammar to repeat.
    pub(crate) fn has_minimax_m3_constrained_generation(
        request: &NvCreateChatCompletionRequest,
    ) -> bool {
        Self::has_structured_response_format(request)
            || Self::has_guided_tool_choice(request)
            || has_explicit_guided_decoding(request)
    }

    /// True when `tool_choice` forces a tool call.
    /// Shared by guided-output reasoning checks and the M3 default.
    pub(crate) fn has_guided_tool_choice<R: OAIChatLikeRequest>(request: &R) -> bool {
        request.tool_choice().is_some_and(|tool_choice| {
            match tool_choice.as_str() {
                Some("required") => true,
                Some(_) => false,
                // The only supported non-string tool choice is a named function.
                None => true,
            }
        })
    }

    // dynamo lib/llm/src/preprocessor.rs:3203-3219
    pub(crate) fn validate_model_request<R: OAIChatLikeRequest>(&self, request: &R) -> Result<()> {
        // ORACLE: `selected_family` stays in dynamo-llm; the test passes its result.
        if self.selected_family == Some("gpt_oss")
            && request
                .chat_template_args()
                .and_then(|args| args.get("reasoning_effort"))
                .and_then(serde_json::Value::as_str)
                == Some("none")
        {
            return Err(invalid_argument_error(
                "GPT-OSS does not support reasoning_effort=none; use low, medium, or high",
            ));
        }
        Ok(())
    }
    // dynamo lib/llm/src/preprocessor.rs:3484-3581
    /// Rendering is driven by the request, so its failures are reported as 400 rather than
    /// 500, matching vLLM. A misconfigured template can also fail here, for instance a
    /// `chat_template` map that omits the `tool_use` key, so log the cause chain before it
    /// is flattened into the client-facing message.
    pub(crate) fn map_prompt_render_error(error: anyhow::Error) -> anyhow::Error {
        tracing::debug!(?error, "Chat template rendering failed");
        let message = match error.downcast_ref::<PromptRenderError>() {
            Some(PromptRenderError::InvalidRequest(message)) => message.clone(),
            None => format!("{error:#}"),
        };
        invalid_argument_error(message)
    }

    pub fn apply_template<
        // ORACLE: only the providers this method reads.
        R: OAIChatLikeRequest + NvExtProvider + CommonExtProvider,
    >(
        &self,
        request: &R,
    ) -> Result<Option<RenderedPrompt>> {
        let continue_final = request.get_continue_final_message() == Some(true);
        if continue_final
            && self
                .formatter
                .media_message_order(request)
                .is_some_and(|order| {
                    order
                        .last()
                        .is_some_and(|&source| source != order.len() - 1)
                })
        {
            return Err(invalid_argument_error(
                "Cannot continue the final message because the prompt formatter moves it before other messages",
            ));
        }
        let formatted_prompt = if self.normalize_tool_call_args || continue_final {
            self.apply_template_inner(&NormalizedArgsRequest {
                inner: request,
                normalize_tool_call_args: self.normalize_tool_call_args,
                continue_final_message: continue_final,
            })?
        } else {
            self.apply_template_inner(request)?
        };
        let Some(prompt) = formatted_prompt else {
            return Ok(None);
        };
        if !continue_final {
            return Ok(Some(prompt));
        }
        apply_continue_final_message(prompt)
            .map_err(|error| invalid_argument_error(format!("{error:#}")))
            .map(Some)
    }

    pub(crate) fn apply_template_inner<
        // ORACLE: only the providers this method reads.
        R: OAIChatLikeRequest + NvExtProvider,
    >(
        &self,
        request: &R,
    ) -> Result<Option<RenderedPrompt>> {
        if let PromptInput::Text(_) = request.prompt_input_type()
            && let Some(TextInput::Single(_)) = request.extract_text()
        {
            let use_raw_prompt = request
                .nvext()
                .is_some_and(|ext| ext.use_raw_prompt.unwrap_or(false));

            let formatted_prompt = if use_raw_prompt {
                match request.raw_prompt() {
                    Some(prompt) => RenderedPrompt::text(prompt),
                    None => {
                        tracing::warn!("Raw prompt requested but not available");
                        self.formatter
                            .render_prompt(request)
                            .map_err(Self::map_prompt_render_error)?
                    }
                }
            } else {
                self.formatter
                    .render_prompt(request)
                    .map_err(Self::map_prompt_render_error)?
            };
            Ok(Some(formatted_prompt))
        } else {
            Ok(None)
        }
    }
}
