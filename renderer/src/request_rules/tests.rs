// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tests that moved with the request rules from Dynamo's `dynamo-llm`
//! (`lib/llm/src/preprocessor.rs` and `lib/llm/src/preprocessor/prompt.rs`).
//!
//! Each test keeps its Dynamo name and assertions. `TestRequest` stands in for
//! Dynamo's `NvCreateChatCompletionRequest`, and the helpers below call the
//! moved functions the way the Dynamo frontend does. Assertions about Dynamo's
//! stream-side reasoning gates stay in Dynamo, because that code did not move.

use std::collections::HashMap;
use std::sync::Arc;

use super::template::{
    CONTINUE_FINAL_MESSAGE_TAG, ChatPromptOptions, NormalizedArgsRequest,
    append_continue_final_message_tag, apply_continue_final_message, map_prompt_render_error,
    render_chat_prompt, should_add_generation_prompt,
};
use super::thinking::{self, ThinkingRules};
use super::tool_arguments::normalize_tool_call_arguments;
use crate::{
    OAIChatLikeRequest, OAIPromptFormatter, PromptRenderError, RenderedPrompt, RenderedSegment,
    TextInput,
};

/// The fields of Dynamo's `NvCreateChatCompletionRequest` that the request
/// rules read, around the OpenAI chat request.
#[derive(Clone, Debug, serde::Deserialize)]
struct TestRequest {
    #[serde(flatten)]
    inner: dynamo_protocols::types::CreateChatCompletionRequest,
    #[serde(default, alias = "chat_template_kwargs")]
    chat_template_args: Option<HashMap<String, serde_json::Value>>,
    #[serde(default)]
    thinking: Option<serde_json::Value>,
    #[serde(default)]
    add_generation_prompt: Option<bool>,
    #[serde(default)]
    continue_final_message: Option<bool>,
    #[serde(default)]
    guided_json: Option<serde_json::Value>,
    #[serde(default)]
    guided_regex: Option<String>,
    #[serde(default)]
    guided_choice: Option<Vec<String>>,
    #[serde(default)]
    guided_grammar: Option<String>,
}

impl OAIChatLikeRequest for TestRequest {
    fn model(&self) -> String {
        self.inner.model()
    }

    fn messages(&self) -> minijinja::value::Value {
        self.inner.messages()
    }

    fn typed_messages(&self) -> Option<&[dynamo_protocols::types::ChatCompletionRequestMessage]> {
        self.inner.typed_messages()
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
        should_add_generation_prompt(self.add_generation_prompt, self.continue_final_message)
    }

    fn extract_text(&self) -> Option<TextInput> {
        Some(TextInput::Single(String::new()))
    }

    fn chat_template_args(&self) -> Option<&HashMap<String, serde_json::Value>> {
        self.chat_template_args.as_ref()
    }
}

/// Dynamo's `tool_choice::has_explicit_guided_decoding`, which stays in Dynamo.
fn has_explicit_guided_decoding(request: &TestRequest) -> bool {
    request.guided_json.is_some()
        || request.guided_regex.is_some()
        || request
            .guided_choice
            .as_ref()
            .is_some_and(|v| !v.is_empty())
        || request.guided_grammar.is_some()
}

fn request_has_client_thinking_control(request: &TestRequest) -> bool {
    request.thinking.is_some()
        || thinking::has_request_thinking_control(request.chat_template_args.as_ref())
}

fn has_minimax_m3_constrained_generation(request: &TestRequest) -> bool {
    thinking::has_minimax_m3_constrained_generation(request, has_explicit_guided_decoding(request))
}

/// Dynamo's test-only `normalize_thinking_arg`: the thinking rules with no
/// deployment default.
fn normalize_thinking_arg(
    request: &mut TestRequest,
    reasoning_parser: Option<&str>,
    tool_call_parser: Option<&str>,
) {
    let constrained_output = has_minimax_m3_constrained_generation(request);
    ThinkingRules {
        reasoning_parser,
        tool_call_parser,
        default_thinking_mode: None,
    }
    .apply(
        &mut request.chat_template_args,
        request.thinking.is_some(),
        constrained_output,
    );
}

fn normalize_thinking_arg_with_source(
    request: &mut TestRequest,
    reasoning_parser: Option<&str>,
    tool_call_parser: Option<&str>,
    thinking_control_from_client: bool,
) {
    let constrained_generation = !thinking_control_from_client
        && thinking::is_minimax_m3_family(reasoning_parser, tool_call_parser)
        && has_minimax_m3_constrained_generation(request);
    thinking::normalize_thinking_args(
        &mut request.chat_template_args,
        reasoning_parser,
        tool_call_parser,
        thinking_control_from_client,
        constrained_generation,
    );
}

/// The deployment default is a plain string here; Dynamo reads it from
/// `runtime_config.runtime_data["default_thinking_mode"]`.
fn runtime_config_with_default_thinking_mode(mode: &'static str) -> &'static str {
    mode
}

fn apply_default_thinking_mode_from_runtime_config(mode: &str, request: &mut TestRequest) {
    let from_client = request_has_client_thinking_control(request);
    thinking::apply_default_thinking_mode(Some(mode), &mut request.chat_template_args, from_client);
}

fn chat_request_with_args(
    chat_template_args: Option<HashMap<String, serde_json::Value>>,
) -> TestRequest {
    let mut request: TestRequest = serde_json::from_value(serde_json::json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "hi"}]
    }))
    .unwrap();
    request.chat_template_args = chat_template_args;
    request
}

/// Render the way Dynamo's `apply_template` does for a chat request.
fn apply_template(
    formatter: &dyn OAIPromptFormatter,
    request: &TestRequest,
    normalize_tool_call_args: bool,
) -> Result<Option<RenderedPrompt>, PromptRenderError> {
    render_chat_prompt(
        formatter,
        request,
        ChatPromptOptions {
            normalize_tool_call_args,
            continue_final_message: request.continue_final_message == Some(true),
            use_raw_prompt: false,
        },
        || None,
    )
}

// ----- Moved from lib/llm/src/preprocessor/prompt.rs -----

fn make_tool_call_messages(arguments: &str) -> serde_json::Value {
    serde_json::json!([{
        "role": "assistant",
        "tool_calls": [{
            "id": "call_1",
            "type": "function",
            "function": { "name": "f", "arguments": arguments }
        }]
    }])
}

/// Regression: large integers must survive the string→object conversion
/// without being silently corrupted to an f64 approximation.
/// parse_args_object detects precision loss and falls back to Value::String,
/// so the original spelling is preserved for the MiniJinja template.
#[test]
fn normalize_preserves_large_integer() {
    let big_int = "123456789012345678901234567890";
    let mut msgs = make_tool_call_messages(&format!("{{\"id\": {big_int}}}"));
    normalize_tool_call_arguments(&mut msgs).unwrap();
    let id_slot = &msgs[0]["tool_calls"][0]["function"]["arguments"]["id"];
    // parse_args_object stores it as Value::String when f64 would lose precision.
    let preserved = id_slot
        .as_str()
        .map(|s| s.to_string())
        .unwrap_or_else(|| id_slot.to_string());
    assert_eq!(
        preserved, big_int,
        "large integer must not be corrupted by f64 round-trip"
    );
}

/// Empty arguments string is a valid no-argument call; must normalise to {}
/// without emitting a warning or returning an error.
#[test]
fn normalize_empty_string_becomes_empty_object() {
    let mut msgs = make_tool_call_messages("");
    normalize_tool_call_arguments(&mut msgs).unwrap();
    assert_eq!(
        msgs[0]["tool_calls"][0]["function"]["arguments"],
        serde_json::json!({}),
        "empty arguments string must normalise to an empty object"
    );
}

/// Malformed JSON in arguments must return an error, not substitute {}.
#[test]
fn normalize_malformed_json_returns_error() {
    let mut msgs = make_tool_call_messages("not-valid-json{");
    let result = normalize_tool_call_arguments(&mut msgs);
    assert!(
        result.is_err(),
        "malformed arguments JSON must return Err, not substitute {{}}"
    );
}

/// Well-formed object arguments pass through unchanged.
#[test]
fn normalize_valid_object_passes_through() {
    let mut msgs = make_tool_call_messages(r#"{"city": "Paris", "unit": "celsius"}"#);
    normalize_tool_call_arguments(&mut msgs).unwrap();
    assert_eq!(
        msgs[0]["tool_calls"][0]["function"]["arguments"]["city"],
        serde_json::json!("Paris")
    );
}

// --- Multi-turn opt-in behaviour tests ---
//
// These guard the boundary between JsonString (default, GPT-OSS / Harmony
// path) and JsonObject (GLM-5.2 path).  The normalize function is the
// shared primitive; NormalizedArgsRequest calls it only when the model
// opts in; other models never call it.

/// GLM-5.2 (JsonObject mode): historical arguments must be a parsed object
/// so the chat template can call .items() on them.
#[test]
fn glm52_historical_arguments_are_object() {
    let mut msgs = make_tool_call_messages(r#"{"location": "San Francisco", "unit": "celsius"}"#);
    normalize_tool_call_arguments(&mut msgs).unwrap();
    let args = &msgs[0]["tool_calls"][0]["function"]["arguments"];
    assert!(
        args.is_object(),
        "GLM-5.2 (JsonObject mode): arguments must be a JSON object, got: {args}"
    );
    assert_eq!(args["location"], serde_json::json!("San Francisco"));
}

/// GPT-OSS (JsonString mode, default): historical arguments must stay as
/// a raw JSON string so the Harmony parser can re-emit a structured call.
/// This test models the multi-turn scenario where the first tool call is
/// in conversation history and the second must be parsed correctly.
#[test]
fn gptoss_historical_arguments_remain_string() {
    // Without normalization (JsonString default), arguments stay as-is.
    let msgs = make_tool_call_messages(r#"{"location": "San Francisco", "unit": "celsius"}"#);
    let args = &msgs[0]["tool_calls"][0]["function"]["arguments"];
    assert!(
        args.is_string(),
        "GPT-OSS (JsonString mode): arguments must remain a JSON string, got: {args}"
    );
    // The raw string is preserved for the Harmony template to handle.
    assert_eq!(
        args.as_str().unwrap(),
        r#"{"location": "San Francisco", "unit": "celsius"}"#
    );
}

fn continue_rendered(prompt: crate::RenderedPrompt) -> crate::RenderedPrompt {
    apply_continue_final_message(prompt).unwrap()
}

#[test]
fn continue_final_message_truncates_after_last_assistant_text() {
    let rendered = continue_rendered(RenderedPrompt::text(format!(
        "user text<|im_end|>LLM-Native Interaction{}<|im_end|><|im_start|>assistant",
        CONTINUE_FINAL_MESSAGE_TAG
    )));
    assert_eq!(
        rendered.as_str(),
        "user text<|im_end|>LLM-Native Interaction"
    );
}

#[test]
fn continue_final_message_rstrips_when_template_trims_tag_space() {
    // Transformers: if the template drops the marker's trailing space, also
    // rstrip the prefix (the template likely trimmed message trailing space).
    let rendered = continue_rendered(RenderedPrompt::text(format!(
        "hello world{}extra",
        CONTINUE_FINAL_MESSAGE_TAG.trim_end()
    )));
    assert_eq!(rendered.as_str(), "hello world");
}

#[test]
fn continue_final_message_marker_survives_repeated_last_text() {
    let rendered = continue_rendered(RenderedPrompt::text(format!(
        "LLM-Native Interaction in the user turn. LLM-Native Interaction{}<|im_end|>",
        CONTINUE_FINAL_MESSAGE_TAG
    )));
    assert_eq!(
        rendered.as_str(),
        "LLM-Native Interaction in the user turn. LLM-Native Interaction"
    );
}

#[test]
fn continue_final_message_appends_tag_to_last_array_text_part() {
    let mut messages = serde_json::json!([{
        "role": "assistant",
        "content": [
            {"type": "text", "text": "ignored"},
            {"type": "text", "text": "Design"}
        ]
    }]);
    append_continue_final_message_tag(&mut messages).unwrap();
    assert_eq!(
        messages[0]["content"][1]["text"].as_str().unwrap(),
        format!("Design{}", CONTINUE_FINAL_MESSAGE_TAG)
    );
    assert_eq!(messages[0]["content"][0]["text"], "ignored");
}

#[test]
fn continue_final_message_appends_tag_to_developer_array_text_part() {
    let mut messages = serde_json::json!([{
        "role": "developer",
        "content": [
            {"type": "text", "text": "ignored"},
            {"type": "text", "text": "Design"}
        ]
    }]);
    append_continue_final_message_tag(&mut messages).unwrap();
    assert_eq!(
        messages[0]["content"][1]["text"].as_str().unwrap(),
        format!("Design{}", CONTINUE_FINAL_MESSAGE_TAG)
    );
}

#[test]
fn continue_final_message_continues_final_user_message() {
    let rendered = continue_rendered(RenderedPrompt::text(format!(
        "hello world{}extra",
        CONTINUE_FINAL_MESSAGE_TAG
    )));
    assert_eq!(rendered.as_str(), "hello world");
}

/// Raw turns `same / previous / same`; rendered last turn is `SAME`.
/// Searching for `same` would cut at the first copy.
#[test]
fn continue_final_message_marker_survives_rewritten_final_turn() {
    let rendered = continue_rendered(RenderedPrompt::text(format!(
        "same|PREVIOUS|SAME{}|closed",
        CONTINUE_FINAL_MESSAGE_TAG
    )));
    assert_eq!(rendered.as_str(), "same|PREVIOUS|SAME");
}

/// Whitespace-only last content must not match earlier structural spaces.
#[test]
fn continue_final_message_whitespace_only_does_not_match_structural_spaces() {
    let mut messages = serde_json::json!([
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "   "}
    ]);
    append_continue_final_message_tag(&mut messages).unwrap();
    assert_eq!(
        messages[1]["content"].as_str().unwrap(),
        format!("   {}", CONTINUE_FINAL_MESSAGE_TAG)
    );

    let rendered = continue_rendered(RenderedPrompt::text(format!(
        "hello   world   {}extra",
        CONTINUE_FINAL_MESSAGE_TAG
    )));
    assert_eq!(rendered.as_str(), "hello   world   ");
}

#[test]
fn continue_final_message_errors_when_marker_is_missing() {
    let err =
        apply_continue_final_message(RenderedPrompt::text("unchanged".to_string())).unwrap_err();
    assert!(
        err.to_string()
            .contains("not found in the rendered chat template"),
        "unexpected error: {err}"
    );
}

#[test]
fn continue_final_message_errors_when_final_content_is_empty() {
    let mut messages = serde_json::json!([{"role": "assistant", "content": ""}]);
    let err = append_continue_final_message_tag(&mut messages).unwrap_err();
    assert!(
        err.to_string()
            .contains("not found in the rendered chat template"),
        "unexpected error: {err}"
    );
}

#[test]
fn continue_final_message_preserves_segment_boundaries() {
    let rendered = continue_rendered(RenderedPrompt::segmented(vec![
        RenderedSegment {
            text: "<|im_start|>assistant\n".to_string(),
            allow_special: true,
        },
        RenderedSegment {
            text: format!("LLM-Native Interaction{}", CONTINUE_FINAL_MESSAGE_TAG),
            allow_special: false,
        },
        RenderedSegment {
            text: "<|im_end|>".to_string(),
            allow_special: true,
        },
        RenderedSegment {
            text: "<|im_start|>assistant\n".to_string(),
            allow_special: true,
        },
    ]));
    assert_eq!(
        rendered.as_str(),
        "<|im_start|>assistant\nLLM-Native Interaction"
    );
    let segments = rendered
        .segments()
        .expect("Kimi-style prompts must keep segment boundaries");
    assert_eq!(segments.len(), 2);
    assert!(segments[0].allow_special);
    assert_eq!(segments[0].text, "<|im_start|>assistant\n");
    assert!(!segments[1].allow_special);
    assert_eq!(segments[1].text, "LLM-Native Interaction");
}

#[test]
fn continue_final_message_truncates_inside_an_ordinary_segment() {
    let rendered = continue_rendered(RenderedPrompt::segmented(vec![
        RenderedSegment {
            text: "<ctrl>".to_string(),
            allow_special: true,
        },
        RenderedSegment {
            text: format!("hello world{}extra", CONTINUE_FINAL_MESSAGE_TAG),
            allow_special: false,
        },
    ]));
    assert_eq!(rendered.as_str(), "<ctrl>hello world");
    let segments = rendered
        .segments()
        .expect("truncated prompt stays segmented");
    assert_eq!(segments.len(), 2);
    assert!(segments[0].allow_special);
    assert_eq!(segments[1].text, "hello world");
    assert!(!segments[1].allow_special);
}

// ----- Moved from lib/llm/src/preprocessor.rs -----

#[test]
fn test_default_thinking_mode_disabled_adds_template_args() {
    let runtime_config = runtime_config_with_default_thinking_mode("disabled");
    let mut request = chat_request_with_args(None);

    apply_default_thinking_mode_from_runtime_config(runtime_config, &mut request);

    let args = request.chat_template_args.as_ref().unwrap();
    assert_eq!(args.get("thinking"), Some(&serde_json::json!(false)));
    assert_eq!(args.get("enable_thinking"), Some(&serde_json::json!(false)));
    assert_eq!(
        args.get("thinking_mode"),
        Some(&serde_json::json!("disabled"))
    );
}

#[test]
fn test_default_thinking_mode_enabled_adds_template_args() {
    let runtime_config = runtime_config_with_default_thinking_mode("enabled");
    let mut request = chat_request_with_args(None);

    apply_default_thinking_mode_from_runtime_config(runtime_config, &mut request);

    let args = request.chat_template_args.as_ref().unwrap();
    assert_eq!(args.get("thinking"), Some(&serde_json::json!(true)));
    assert_eq!(args.get("enable_thinking"), Some(&serde_json::json!(true)));
    assert_eq!(
        args.get("thinking_mode"),
        Some(&serde_json::json!("enabled"))
    );
}

#[test]
fn test_default_thinking_mode_does_not_override_request() {
    let runtime_config = runtime_config_with_default_thinking_mode("disabled");
    let mut request = chat_request_with_args(Some(HashMap::from([(
        "thinking_mode".to_string(),
        serde_json::json!("enabled"),
    )])));

    apply_default_thinking_mode_from_runtime_config(runtime_config, &mut request);

    let args = request.chat_template_args.as_ref().unwrap();
    assert_eq!(
        args.get("thinking_mode"),
        Some(&serde_json::json!("enabled"))
    );
    assert!(!args.contains_key("thinking"));
    assert!(!args.contains_key("enable_thinking"));
}

#[test]
fn test_default_thinking_mode_precedes_parser_implicit_default() {
    let runtime_config = runtime_config_with_default_thinking_mode("disabled");
    let mut request = chat_request_with_args(None);

    apply_default_thinking_mode_from_runtime_config(runtime_config, &mut request);
    normalize_thinking_arg(&mut request, Some("kimi_k25"), None);

    let args = request.chat_template_args.as_ref().unwrap();
    assert_eq!(args.get("thinking"), Some(&serde_json::json!(false)));
    assert_eq!(args.get("enable_thinking"), Some(&serde_json::json!(false)));
    assert_eq!(
        args.get("thinking_mode"),
        Some(&serde_json::json!("disabled"))
    );
}

#[test]
fn test_default_thinking_mode_does_not_override_reasoning_effort() {
    let runtime_config = runtime_config_with_default_thinking_mode("disabled");
    let mut request = chat_request_with_args(Some(HashMap::from([(
        "reasoning_effort".to_string(),
        serde_json::json!("high"),
    )])));

    apply_default_thinking_mode_from_runtime_config(runtime_config, &mut request);

    let args = request.chat_template_args.as_ref().unwrap();
    assert_eq!(
        args.get("reasoning_effort"),
        Some(&serde_json::json!("high"))
    );
    assert!(!args.contains_key("thinking"));
    assert!(!args.contains_key("enable_thinking"));
    assert!(!args.contains_key("thinking_mode"));
}

fn test_prompt_formatter(template: &str) -> Arc<dyn OAIPromptFormatter> {
    let template: crate::ChatTemplate = serde_json::from_value(serde_json::json!({
        "chat_template": template
    }))
    .unwrap();
    match crate::PromptFormatter::from_parts(template, crate::ContextMixins::default(), false)
        .unwrap()
    {
        crate::PromptFormatter::OAI(formatter) => formatter,
    }
}

#[test]
fn continue_final_message_leaves_last_assistant_open_on_llama_template() {
    // The chat template of Dynamo's
    // `tests/data/sample-models/mock-llama-3.1-8b-instruct`.
    let template: crate::ChatTemplate = serde_json::from_value(serde_json::json!({
            "chat_template": "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim %}{% if loop.first %}{% set content = bos_token + content %}{% endif %}{% if not loop.last %}{% set content = content + '<|eot_id|>'%}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}",
            "bos_token": "<|begin_of_text|>",
            "eos_token": "<|eot_id|>"
        }))
        .unwrap();
    let crate::PromptFormatter::OAI(formatter) =
        crate::PromptFormatter::from_parts(template, crate::ContextMixins::default(), false)
            .unwrap();

    let default_request: TestRequest = serde_json::from_value(serde_json::json!({
        "model": "test-model",
        "messages": [
            {"role": "user", "content": "Continue this sentence"},
            {"role": "assistant", "content": "LLM-Native Interaction"}
        ]
    }))
    .unwrap();
    let continue_request: TestRequest = serde_json::from_value(serde_json::json!({
        "model": "test-model",
        "messages": [
            {"role": "user", "content": "Continue this sentence"},
            {"role": "assistant", "content": "LLM-Native Interaction"}
        ],
        "add_generation_prompt": false,
        "continue_final_message": true
    }))
    .unwrap();

    let default_prompt = apply_template(formatter.as_ref(), &default_request, false)
        .unwrap()
        .unwrap();
    let continue_prompt = apply_template(formatter.as_ref(), &continue_request, false)
        .unwrap()
        .unwrap();

    assert!(
        default_prompt
            .as_str()
            .ends_with("<|start_header_id|>assistant<|end_header_id|>\n\n"),
        "default render should start a new assistant turn, got {:?}",
        default_prompt.as_str()
    );
    assert!(
        continue_prompt.as_str().ends_with("LLM-Native Interaction"),
        "continue_final_message should leave the last assistant open, got {:?}",
        continue_prompt.as_str()
    );
    assert!(
        !continue_prompt.as_str().contains(
            "LLM-Native Interaction<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        ),
        "continue_final_message must not close the last assistant and start a new turn, got {:?}",
        continue_prompt.as_str()
    );
}

#[test]
fn should_add_generation_prompt_defaults_true_and_continue_forces_false() {
    let unset: TestRequest = serde_json::from_value(serde_json::json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "hi"}]
    }))
    .unwrap();
    assert!(unset.should_add_generation_prompt());

    let explicit_false: TestRequest = serde_json::from_value(serde_json::json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "hi"}],
        "add_generation_prompt": false
    }))
    .unwrap();
    assert!(!explicit_false.should_add_generation_prompt());

    let continue_with_false: TestRequest = serde_json::from_value(serde_json::json!({
        "model": "test-model",
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "partial"}
        ],
        "add_generation_prompt": false,
        "continue_final_message": true
    }))
    .unwrap();
    assert!(!continue_with_false.should_add_generation_prompt());
}

/// Qwen-style templates close every turn, including the last assistant. Truncate
/// after render is what actually leaves the prefix open; Llama's mock template
/// already omits the last eot when `add_generation_prompt` is false.
const QWEN_STYLE_TEMPLATE: &str = "\
{%- for message in messages -%}\
{%- if message.role == 'user' -%}{{ '<|im_start|>user\n' + message.content + '<|im_end|>\n' }}\
{%- elif message.role == 'assistant' -%}{{ '<|im_start|>assistant\n' + message.content + '<|im_end|>\n' }}\
{%- endif -%}\
{%- endfor -%}\
{%- if add_generation_prompt -%}{{ '<|im_start|>assistant\n' }}{%- endif -%}";

fn continue_request() -> TestRequest {
    serde_json::from_value(serde_json::json!({
        "model": "test-model",
        "messages": [
            {"role": "user", "content": "Continue this sentence"},
            {"role": "assistant", "content": "LLM-Native Interaction"}
        ],
        "add_generation_prompt": false,
        "continue_final_message": true
    }))
    .unwrap()
}

#[test]
fn normalized_args_request_preserves_render_transformations() {
    let request: TestRequest = serde_json::from_value(serde_json::json!({
        "model": "test-model",
        "messages": [{"role": "assistant", "content": "prefix", "tool_calls": [{
            "id": "c1", "type": "function",
            "function": {"name": "lookup", "arguments": "{\"query\":\"hello\"}"}
        }]}]
    }))
    .unwrap();
    let normalized = NormalizedArgsRequest::new(&request, true, true).unwrap();
    assert!(request.typed_messages().is_some());
    assert!(normalized.typed_messages().is_none());

    let messages = serde_json::to_value(normalized.messages()).unwrap();
    assert_eq!(
        messages[0]["tool_calls"][0]["function"]["arguments"],
        serde_json::json!({"query": "hello"})
    );
    assert_eq!(
        messages[0]["content"],
        format!("prefix{CONTINUE_FINAL_MESSAGE_TAG}")
    );
    // Keep string arguments in the template so only the adapter can perform
    // this conversion; the renderer's own argument normalization is bypassed.
    let formatter = test_prompt_formatter(
        "{% for m in messages %}{% for call in m.tool_calls %}{% if call.function.arguments is string %}RAW{% else %}{{ call.function.arguments.query }}{% endif %}{% endfor %}{{ m.content }}{% endfor %}",
    );
    let rendered = formatter.render_prompt(&normalized).unwrap();
    assert_eq!(
        rendered.as_str(),
        format!("helloprefix{CONTINUE_FINAL_MESSAGE_TAG}")
    );
    assert_eq!(
        apply_continue_final_message(rendered).unwrap().as_str(),
        "helloprefix"
    );
}

fn render_with_continue_final_message(
    formatter: &dyn OAIPromptFormatter,
    request: &TestRequest,
) -> RenderedPrompt {
    let continue_final = request.continue_final_message == Some(true);
    let rendered = if continue_final {
        formatter
            .render_prompt(&NormalizedArgsRequest::new(request, false, true).unwrap())
            .unwrap()
    } else {
        formatter.render_prompt(request).unwrap()
    };
    if continue_final {
        apply_continue_final_message(rendered).unwrap()
    } else {
        rendered
    }
}

#[test]
fn continue_final_message_strips_qwen_style_closing_tokens() {
    let formatter = test_prompt_formatter(QWEN_STYLE_TEMPLATE);
    let default_request: TestRequest = serde_json::from_value(serde_json::json!({
        "model": "test-model",
        "messages": [
            {"role": "user", "content": "Continue this sentence"},
            {"role": "assistant", "content": "LLM-Native Interaction"}
        ]
    }))
    .unwrap();
    let closed_only: TestRequest = serde_json::from_value(serde_json::json!({
        "model": "test-model",
        "messages": [
            {"role": "user", "content": "Continue this sentence"},
            {"role": "assistant", "content": "LLM-Native Interaction"}
        ],
        "add_generation_prompt": false
    }))
    .unwrap();

    let default_prompt = render_with_continue_final_message(formatter.as_ref(), &default_request);
    let closed_prompt = render_with_continue_final_message(formatter.as_ref(), &closed_only);
    let continue_prompt =
        render_with_continue_final_message(formatter.as_ref(), &continue_request());

    assert!(
        default_prompt.as_str().ends_with("<|im_start|>assistant\n"),
        "default Qwen render should start a new assistant turn, got {:?}",
        default_prompt.as_str()
    );
    assert!(
        closed_prompt
            .as_str()
            .ends_with("LLM-Native Interaction<|im_end|>\n"),
        "add_generation_prompt=false alone must still close the last assistant, got {:?}",
        closed_prompt.as_str()
    );
    assert_eq!(
        continue_prompt.as_str(),
        "<|im_start|>user\nContinue this sentence<|im_end|>\n<|im_start|>assistant\nLLM-Native Interaction"
    );
}

/// Raw turns `same / previous / same`; the template uppercases the last
/// turn to `SAME`. Searching rendered text for `same` would cut at the
/// first copy. The HuggingFace marker must keep the full conversation.
#[test]
fn continue_final_message_marker_survives_rewritten_final_turn_in_template() {
    const TEMPLATE: &str = "\
{%- for message in messages -%}\
{%- if loop.last -%}{{ message.content | upper }}|{%- else -%}{{ message.content }}|{%- endif -%}\
{%- endfor -%}closed";
    let formatter = test_prompt_formatter(TEMPLATE);
    let request: TestRequest = serde_json::from_value(serde_json::json!({
        "model": "test-model",
        "messages": [
            {"role": "user", "content": "same"},
            {"role": "user", "content": "previous"},
            {"role": "assistant", "content": "same"}
        ],
        "add_generation_prompt": false,
        "continue_final_message": true
    }))
    .unwrap();

    let prompt = render_with_continue_final_message(formatter.as_ref(), &request);
    assert_eq!(prompt.as_str(), "same|previous|SAME");
    assert!(
        !prompt.as_str().contains("CONTINUE_FINAL_MESSAGE_TAG"),
        "marker must be stripped from the prompt sent to the model, got {:?}",
        prompt.as_str()
    );
}

#[test]

fn test_kimi_thinking_normalization_keeps_template_and_gates_in_sync() {
    let template: crate::ChatTemplate = serde_json::from_value(serde_json::json!({
            "chat_template": "{% if thinking == true and enable_thinking == true %}<think>{% elif thinking == false and enable_thinking == false %}<chat>{% else %}<mismatch>{% endif %}"
        }))
        .unwrap();
    let formatter =
        crate::PromptFormatter::from_parts(template, crate::ContextMixins::default(), false)
            .unwrap();
    let cases = [
        (serde_json::json!(true), true, "true"),
        (serde_json::json!("true"), true, "string true"),
        (serde_json::json!("TRUE"), true, "uppercase true"),
        (serde_json::json!("1"), true, "string one"),
        (serde_json::json!("yes"), true, "yes"),
        (serde_json::json!("on"), true, "on"),
        (serde_json::json!(1), true, "one"),
        (serde_json::json!(false), false, "false"),
        (serde_json::json!("false"), false, "string false"),
        (serde_json::json!("no"), false, "no"),
        (serde_json::json!("off"), false, "off"),
        (serde_json::json!(0), false, "zero"),
    ];

    let assert_case =
        |key: Option<&str>, value: Option<&serde_json::Value>, expected, description| {
            let mut request: TestRequest = serde_json::from_value(serde_json::json!({
                "messages": [{"role": "user", "content": "hello"}],
                "model": "moonshotai/Kimi-K2.5-Instruct"
            }))
            .unwrap();
            if let (Some(key), Some(value)) = (key, value) {
                request.chat_template_args = Some(std::collections::HashMap::from([(
                    key.to_string(),
                    value.clone(),
                )]));
            }

            normalize_thinking_arg(&mut request, Some("kimi_k25"), None);
            let args = request.chat_template_args.as_ref();
            assert_eq!(
                crate::thinking_bool_from_args(args),
                Some(expected),
                "renderer helper mismatch for {description}"
            );
            // The SGLang and postprocessor gate assertions stay in Dynamo.

            let rendered = match &formatter {
                crate::PromptFormatter::OAI(formatter) => formatter.render(&request).unwrap(),
            };
            assert_eq!(
                rendered,
                if expected { "<think>" } else { "<chat>" },
                "rendered prompt mismatch for {description}"
            );
        };

    for key in ["thinking", "enable_thinking"] {
        for (value, expected, description) in &cases {
            assert_case(Some(key), Some(value), *expected, *description);
        }
    }
    assert_case(None, None, true, "omitted Kimi default");

    for parser in ["kimi_k3", "kimi-k3"] {
        let mut request: TestRequest = serde_json::from_value(serde_json::json!({
            "messages": [{"role": "user", "content": "hello"}],
            "model": "moonshotai/Kimi-K3"
        }))
        .unwrap();
        normalize_thinking_arg(&mut request, Some(parser), None);
        assert_eq!(
            crate::thinking_bool_from_args(request.chat_template_args.as_ref()),
            Some(true),
            "{parser} should default to thinking mode",
        );
    }

    let mut conflicting_request: TestRequest = serde_json::from_value(serde_json::json!({
        "messages": [{"role": "user", "content": "hello"}],
        "model": "moonshotai/Kimi-K2.5-Instruct",
        "chat_template_kwargs": {
            "thinking": "true",
            "enable_thinking": false
        }
    }))
    .unwrap();
    normalize_thinking_arg(&mut conflicting_request, Some("kimi_k25"), None);
    let args = conflicting_request.chat_template_args.as_ref().unwrap();
    assert_eq!(args.get("thinking"), Some(&serde_json::Value::Bool(true)));
    assert_eq!(
        args.get("enable_thinking"),
        Some(&serde_json::Value::Bool(true))
    );
    let rendered = match &formatter {
        crate::PromptFormatter::OAI(formatter) => formatter.render(&conflicting_request).unwrap(),
    };
    assert_eq!(rendered, "<think>");
}

fn minimax_m3_request(body: serde_json::Value) -> TestRequest {
    let mut base = serde_json::json!({
        "model": "MiniMaxAI/MiniMax-M3",
        "messages": [{"role": "user", "content": "hi"}],
    });
    let base_obj = base.as_object_mut().unwrap();
    for (k, v) in body.as_object().unwrap() {
        base_obj.insert(k.clone(), v.clone());
    }
    serde_json::from_value(base).unwrap()
}

#[test]
fn test_normalize_thinking_arg_m3_defaults_disabled_for_json_schema() {
    for parser in ["minimax_m3", "minimax-m3"] {
        let mut request = minimax_m3_request(serde_json::json!({
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "s", "schema": {"type": "object"}}
            }
        }));
        normalize_thinking_arg(&mut request, Some(parser), None);
        let args = request.chat_template_args.as_ref().unwrap();
        assert_eq!(
            args.get("thinking_mode"),
            Some(&serde_json::json!("disabled"))
        );
        assert_eq!(args.get("thinking"), Some(&serde_json::json!(false)));
        assert_eq!(args.get("enable_thinking"), Some(&serde_json::json!(false)));
    }
}

#[test]
fn test_normalize_thinking_arg_m3_defaults_disabled_for_tool_choice_required() {
    let mut request = minimax_m3_request(serde_json::json!({
        "tools": [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}],
        "tool_choice": "required"
    }));
    normalize_thinking_arg(&mut request, Some("minimax_m3"), None);
    let args = request.chat_template_args.as_ref().unwrap();
    assert_eq!(
        args.get("thinking_mode"),
        Some(&serde_json::json!("disabled"))
    );
}

#[test]
fn test_normalize_thinking_arg_m3_defaults_disabled_for_named_tool_choice() {
    let mut request = minimax_m3_request(serde_json::json!({
        "tools": [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}],
        "tool_choice": {"type": "function", "function": {"name": "f"}}
    }));
    normalize_thinking_arg(&mut request, Some("minimax_m3"), None);
    let args = request.chat_template_args.as_ref().unwrap();
    assert_eq!(
        args.get("thinking_mode"),
        Some(&serde_json::json!("disabled"))
    );
}

#[test]
fn test_normalize_thinking_arg_m3_tool_call_only_deployment_gets_default() {
    // Tool-call-only M3 deployments still render the M3 template.
    // Guided-output requests therefore still need "adaptive" overridden.
    for tool_call_parser in [
        "minimax_m3",
        "minimax-m3",
        "minimax_m3_nom",
        "minimax-m3-nom",
    ] {
        let mut request = minimax_m3_request(serde_json::json!({
            "tools": [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}],
            "tool_choice": "required"
        }));
        normalize_thinking_arg(&mut request, None, Some(tool_call_parser));
        let args = request.chat_template_args.as_ref().unwrap();
        assert_eq!(
            args.get("thinking_mode"),
            Some(&serde_json::json!("disabled")),
            "tool_call_parser={tool_call_parser:?} + tool_choice=required with no reasoning parser must still default thinking_mode=disabled"
        );
    }
}

#[test]
fn test_normalize_thinking_arg_m3_preserves_client_adaptive() {
    let mut request = minimax_m3_request(serde_json::json!({
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "s", "schema": {"type": "object"}}
        },
        "chat_template_kwargs": {"thinking_mode": "adaptive"}
    }));
    normalize_thinking_arg(&mut request, Some("minimax_m3"), None);
    let args = request.chat_template_args.as_ref().unwrap();
    assert_eq!(
        args.get("thinking_mode"),
        Some(&serde_json::json!("adaptive"))
    );
    assert!(!args.contains_key("thinking"));
    assert!(!args.contains_key("enable_thinking"));
}

#[test]
fn test_normalize_thinking_arg_m3_bridges_thinking_bool_to_mode() {
    let mut request = minimax_m3_request(serde_json::json!({
        "chat_template_kwargs": {"thinking": true}
    }));
    normalize_thinking_arg(&mut request, Some("minimax_m3"), None);
    let args = request.chat_template_args.as_ref().unwrap();
    assert_eq!(args.get("thinking"), Some(&serde_json::json!(true)));
    assert_eq!(args.get("enable_thinking"), Some(&serde_json::json!(true)));
    assert_eq!(
        args.get("thinking_mode"),
        Some(&serde_json::json!("enabled"))
    );
}

#[test]
fn test_normalize_thinking_arg_m3_bridges_enable_thinking_to_mode() {
    let mut request = minimax_m3_request(serde_json::json!({
        "chat_template_kwargs": {"enable_thinking": false}
    }));
    normalize_thinking_arg(&mut request, Some("minimax_m3"), None);
    let args = request.chat_template_args.as_ref().unwrap();
    assert_eq!(args.get("thinking"), Some(&serde_json::json!(false)));
    assert_eq!(args.get("enable_thinking"), Some(&serde_json::json!(false)));
    assert_eq!(
        args.get("thinking_mode"),
        Some(&serde_json::json!("disabled"))
    );
}

#[test]
fn test_normalize_thinking_arg_thinking_mode_enabled_is_thinking_on() {
    // `thinking_mode` strings must bypass `is_truthy`, which would map
    // "enabled" to false. M3 inputs are canonicalized to exact values.
    for value in ["enabled", "ENABLED", "Enabled"] {
        let mut request = minimax_m3_request(serde_json::json!({
            "chat_template_kwargs": {"thinking_mode": value}
        }));
        normalize_thinking_arg(&mut request, Some("minimax_m3"), None);
        let args = request.chat_template_args.as_ref().unwrap();
        assert_eq!(
            args.get("thinking"),
            Some(&serde_json::json!(true)),
            "thinking_mode={value:?} must map to thinking=true"
        );
        assert_eq!(args.get("enable_thinking"), Some(&serde_json::json!(true)));
        assert_eq!(
            args.get("thinking_mode"),
            Some(&serde_json::json!("enabled"))
        );
    }
}

#[test]
fn test_normalize_thinking_arg_thinking_mode_disabled_is_thinking_off() {
    for value in ["disabled", "DISABLED"] {
        let mut request = minimax_m3_request(serde_json::json!({
            "chat_template_kwargs": {"thinking_mode": value}
        }));
        normalize_thinking_arg(&mut request, Some("minimax_m3"), None);
        let args = request.chat_template_args.as_ref().unwrap();
        assert_eq!(args.get("thinking"), Some(&serde_json::json!(false)));
        assert_eq!(args.get("enable_thinking"), Some(&serde_json::json!(false)));
        assert_eq!(
            args.get("thinking_mode"),
            Some(&serde_json::json!("disabled"))
        );
    }
}

#[test]
fn test_normalize_thinking_arg_m3_canonicalizes_non_canonical_thinking_mode() {
    // M3 only treats exact "disabled" as off.
    // Canonicalize falsy aliases so `thinking=false` and the template gate agree.
    for value in [
        serde_json::json!(false),
        serde_json::json!(0),
        serde_json::json!("false"),
        serde_json::json!("no"),
        serde_json::json!("off"),
    ] {
        let mut request = minimax_m3_request(serde_json::json!({
            "chat_template_kwargs": {"thinking_mode": value.clone()}
        }));
        normalize_thinking_arg(&mut request, Some("minimax_m3"), None);
        let args = request.chat_template_args.as_ref().unwrap();
        assert_eq!(
            args.get("thinking_mode"),
            Some(&serde_json::json!("disabled")),
            "input={value} must canonicalize to \"disabled\""
        );
        assert_eq!(args.get("thinking"), Some(&serde_json::json!(false)));
        assert_eq!(args.get("enable_thinking"), Some(&serde_json::json!(false)));
    }

    // Truthy non-canonical values must canonicalize to "enabled".
    for value in [
        serde_json::json!(true),
        serde_json::json!(1),
        serde_json::json!("true"),
        serde_json::json!("yes"),
        serde_json::json!("on"),
    ] {
        let mut request = minimax_m3_request(serde_json::json!({
            "chat_template_kwargs": {"thinking_mode": value.clone()}
        }));
        normalize_thinking_arg(&mut request, Some("minimax_m3"), None);
        let args = request.chat_template_args.as_ref().unwrap();
        assert_eq!(
            args.get("thinking_mode"),
            Some(&serde_json::json!("enabled")),
            "input={value} must canonicalize to \"enabled\""
        );
        assert_eq!(args.get("thinking"), Some(&serde_json::json!(true)));
        assert_eq!(args.get("enable_thinking"), Some(&serde_json::json!(true)));
    }
}

#[test]
fn test_normalize_thinking_arg_deepseek_thinking_mode_words_preserved() {
    // DeepSeek V3.1 uses `thinking_mode` values "chat" and "thinking".
    // Preserve them verbatim without writing boolean aliases.
    for value in ["thinking", "chat"] {
        let mut request: TestRequest = serde_json::from_value(serde_json::json!({
            "model": "deepseek/DeepSeek-V3.1",
            "messages": [{"role": "user", "content": "hi"}],
            "chat_template_kwargs": {"thinking_mode": value}
        }))
        .unwrap();
        normalize_thinking_arg(&mut request, Some("deepseek_v3_1"), None);
        let args = request.chat_template_args.as_ref().unwrap();
        assert_eq!(
            args.get("thinking_mode").and_then(|v| v.as_str()),
            Some(value),
            "DeepSeek {value:?} must survive verbatim"
        );
        assert!(
            !args.contains_key("thinking"),
            "no boolean coercion for DeepSeek {value:?}"
        );
        assert!(!args.contains_key("enable_thinking"));
    }
}

#[test]
fn test_normalize_thinking_arg_thinking_mode_not_written_for_non_m3_parsers() {
    // Only M3-family templates use the "enabled"/"disabled" vocabulary.
    // Other parsers may treat an injected value as out-of-vocabulary.
    for parser in [
        "deepseek_v3_1",
        "deepseek_v3_2",
        "minimax_m2",
        "qwen3",
        "kimi_k25",
        "kimi_k3",
    ] {
        let mut request: TestRequest = serde_json::from_value(serde_json::json!({
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}],
            "chat_template_kwargs": {"thinking": true}
        }))
        .unwrap();
        normalize_thinking_arg(&mut request, Some(parser), None);
        let args = request.chat_template_args.as_ref().unwrap();
        assert_eq!(args.get("thinking"), Some(&serde_json::json!(true)));
        assert_eq!(args.get("enable_thinking"), Some(&serde_json::json!(true)));
        assert!(
            !args.contains_key("thinking_mode"),
            "parser {parser:?} must not receive an injected thinking_mode value"
        );
    }
}

#[test]
fn test_normalize_thinking_arg_kimi_or_else_does_not_write_thinking_mode() {
    // Kimi parsers synthesize `Some(true)` with no client input.
    // Write boolean aliases only; Kimi does not read `thinking_mode`.
    for parser in ["kimi_k25", "kimi_k3", "kimi-k3"] {
        let mut request: TestRequest = serde_json::from_value(serde_json::json!({
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .unwrap();
        normalize_thinking_arg(&mut request, Some(parser), None);
        let args = request.chat_template_args.as_ref().unwrap();
        assert_eq!(args.get("thinking"), Some(&serde_json::json!(true)));
        assert!(!args.contains_key("thinking_mode"));
    }
}

#[test]
fn test_normalize_thinking_arg_m3_leaves_plain_chat_untouched() {
    let mut request = minimax_m3_request(serde_json::json!({}));
    normalize_thinking_arg(&mut request, Some("minimax_m3"), None);
    assert!(request.chat_template_args.is_none());
}

#[test]
fn test_normalize_thinking_arg_non_m3_adaptive_is_no_op() {
    // Preserve non-M3 `thinking_mode: "adaptive"` verbatim by leaving
    // normalization as `None`.
    let mut request: TestRequest = serde_json::from_value(serde_json::json!({
        "model": "test",
        "messages": [{"role": "user", "content": "hi"}],
        "chat_template_kwargs": {"thinking_mode": "adaptive"}
    }))
    .unwrap();
    normalize_thinking_arg(&mut request, Some("qwen3"), None);
    let args = request.chat_template_args.as_ref().unwrap();
    assert_eq!(
        args.get("thinking_mode"),
        Some(&serde_json::json!("adaptive"))
    );
    assert!(!args.contains_key("thinking"));
    assert!(!args.contains_key("enable_thinking"));
}

#[test]
fn test_normalize_thinking_arg_operator_default_forces_m3_constrained_disabled() {
    // Runtime defaults are operator policy, not client intent. When the
    // client sends no thinking control, constrained M3 requests force
    // "disabled" even if the operator default is "enabled" — otherwise
    // the deployment default silently propagates a mode M3 cannot honor
    // with a constrained decoder attached.
    let runtime_config = runtime_config_with_default_thinking_mode("enabled");
    let mut request = minimax_m3_request(serde_json::json!({
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "s", "schema": {"type": "object"}}
        }
    }));
    let original_args = request.chat_template_args.clone();
    // Dynamo's `request_disables_reasoning_with_family` runs the same
    // rules on a copy of the arguments; its gate stays in Dynamo.
    let mut copied_args = request.chat_template_args.clone();
    ThinkingRules {
        reasoning_parser: Some("minimax_m3"),
        tool_call_parser: None,
        default_thinking_mode: Some("enabled"),
    }
    .apply(
        &mut copied_args,
        request.thinking.is_some(),
        has_minimax_m3_constrained_generation(&request),
    );
    assert_eq!(
        copied_args
            .as_ref()
            .and_then(|args| args.get("thinking_mode")),
        Some(&serde_json::json!("disabled"))
    );
    assert_eq!(request.chat_template_args, original_args);
    let thinking_control_from_client = request_has_client_thinking_control(&request);
    assert!(
        !thinking_control_from_client,
        "no chat_template_kwargs → not client intent"
    );
    apply_default_thinking_mode_from_runtime_config(runtime_config, &mut request);
    normalize_thinking_arg_with_source(
        &mut request,
        Some("minimax_m3"),
        None,
        thinking_control_from_client,
    );
    let args = request.chat_template_args.as_ref().unwrap();
    assert_eq!(
        args.get("thinking_mode"),
        Some(&serde_json::json!("disabled"))
    );
    assert_eq!(args.get("thinking"), Some(&serde_json::json!(false)));
    assert_eq!(args.get("enable_thinking"), Some(&serde_json::json!(false)));
}

#[test]
fn test_normalize_thinking_arg_m3_preserves_explicit_thinking_for_constrained() {
    // Explicit client thinking intent is respected for constrained M3
    // requests. The runtime attempts the combination even though the
    // template's adaptive/enabled path is known to conflict with the
    // constrained decoder — this is the client's opt-in, and returning
    // possibly-degraded content is preferred to a hard reject at the
    // API boundary (which would break provider-compatibility probes
    // that expect the request to be attempted).
    for (body, expected_mode, desc) in [
        (
            serde_json::json!({
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "s", "schema": {"type": "object"}}
                },
                "chat_template_kwargs": {"thinking_mode": "enabled"}
            }),
            "enabled",
            "json_schema + explicit thinking_mode=enabled",
        ),
        (
            serde_json::json!({
                "tools": [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}],
                "tool_choice": "required",
                "chat_template_kwargs": {"thinking_mode": "adaptive"}
            }),
            "adaptive",
            "required tool + explicit thinking_mode=adaptive",
        ),
        (
            serde_json::json!({
                "tools": [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}],
                "tool_choice": {"type": "function", "function": {"name": "f"}},
                "chat_template_kwargs": {"thinking": true}
            }),
            "enabled",
            "named tool + explicit thinking=true",
        ),
    ] {
        let mut request = minimax_m3_request(body);
        let mut copied_args = request.chat_template_args.clone();
        ThinkingRules {
            reasoning_parser: Some("minimax_m3"),
            tool_call_parser: None,
            default_thinking_mode: Some("disabled"),
        }
        .apply(
            &mut copied_args,
            request.thinking.is_some(),
            has_minimax_m3_constrained_generation(&request),
        );
        assert_eq!(
            copied_args
                .as_ref()
                .and_then(|args| args.get("thinking_mode")),
            Some(&serde_json::json!(expected_mode)),
            "{desc}: the deployment default must not override client intent"
        );
        let thinking_control_from_client = request_has_client_thinking_control(&request);
        assert!(
            thinking_control_from_client,
            "{desc} must be classified as client thinking intent"
        );
        normalize_thinking_arg_with_source(
            &mut request,
            Some("minimax_m3"),
            None,
            thinking_control_from_client,
        );
        let args = request.chat_template_args.as_ref().unwrap();
        assert_eq!(
            args.get("thinking_mode"),
            Some(&serde_json::json!(expected_mode)),
            "{desc} must preserve explicit client thinking"
        );
    }
}

/// Top-level `guided_*` requests and the bytes a grammar-constrained
/// backend emits for them from token 0.
fn minimax_m3_explicit_guided_cases() -> Vec<(&'static str, serde_json::Value, &'static str)> {
    vec![
        (
            "guided_json",
            serde_json::json!({"guided_json": {
                "type": "object",
                "properties": {"city": {"type": "string"}, "country": {"type": "string"}},
                "required": ["city", "country"]
            }}),
            r#"{"city":"Paris","country":"France"}"#,
        ),
        (
            "guided_regex",
            serde_json::json!({"guided_regex": "Paris|London"}),
            "Paris",
        ),
        (
            "guided_choice",
            serde_json::json!({"guided_choice": ["Paris", "London"]}),
            "Paris",
        ),
        (
            "guided_grammar",
            serde_json::json!({"guided_grammar": "root ::= \"Paris\" | \"London\""}),
            "Paris",
        ),
    ]
}

#[test]
fn test_normalize_thinking_arg_m3_defaults_disabled_for_explicit_guided_decoding() {
    // Top-level `guided_*` installs the same backend grammar as a structured
    // `response_format`, so M3 must render it with thinking disabled too.
    for (reasoning_parser, tool_call_parser) in [
        (Some("minimax_m3"), None),
        (Some("minimax-m3"), None),
        (None, Some("minimax_m3")),
        (Some("minimax_m3"), Some("minimax_m3")),
    ] {
        for (field, body, _) in minimax_m3_explicit_guided_cases() {
            let mut request = minimax_m3_request(body);
            let desc = format!("{field} with {reasoning_parser:?}/{tool_call_parser:?}");
            assert!(
                has_minimax_m3_constrained_generation(&request),
                "{desc} is constrained generation"
            );
            // The reasoning-parser gate assertion stays in Dynamo.
            normalize_thinking_arg(&mut request, reasoning_parser, tool_call_parser);
            let args = request.chat_template_args.as_ref();
            assert_eq!(
                args.and_then(|args| args.get("thinking_mode")),
                Some(&serde_json::json!("disabled")),
                "{desc} must render the M3 template with thinking disabled"
            );
            assert_eq!(
                args.and_then(|args| args.get("thinking")),
                Some(&serde_json::json!(false))
            );
            assert_eq!(
                args.and_then(|args| args.get("enable_thinking")),
                Some(&serde_json::json!(false))
            );
        }
    }
}

#[test]
fn test_normalize_thinking_arg_m3_explicit_guided_decoding_scope() {
    // Fields that install no grammar leave plain M3 chat untouched.
    for body in [
        serde_json::json!({"guided_choice": []}),
        serde_json::json!({"guided_decoding_backend": "xgrammar"}),
        serde_json::json!({"guided_whitespace_pattern": "[ ]?"}),
    ] {
        let mut request = minimax_m3_request(body.clone());
        assert!(
            !has_minimax_m3_constrained_generation(&request),
            "{body} installs no grammar"
        );
        normalize_thinking_arg(&mut request, Some("minimax_m3"), None);
        assert!(request.chat_template_args.is_none(), "{body}");
    }

    // Explicit client thinking still wins, as it does for `response_format`.
    for (field, mut body, _) in minimax_m3_explicit_guided_cases() {
        body.as_object_mut().unwrap().insert(
            "chat_template_kwargs".to_string(),
            serde_json::json!({"thinking_mode": "enabled"}),
        );
        let mut request = minimax_m3_request(body);
        normalize_thinking_arg(&mut request, Some("minimax_m3"), None);
        assert_eq!(
            request
                .chat_template_args
                .as_ref()
                .and_then(|args| args.get("thinking_mode")),
            Some(&serde_json::json!("enabled")),
            "{field} + explicit thinking_mode=enabled must keep client intent"
        );
    }

    // Non-M3 parsers keep their existing policy for `guided_*`.
    for parser in ["qwen3", "minimax_m2", "deepseek_v3_1", "glm45"] {
        let mut request = minimax_m3_request(serde_json::json!({
            "guided_json": {"type": "object"}
        }));
        normalize_thinking_arg(&mut request, Some(parser), None);
        assert!(
            request.chat_template_args.is_none(),
            "parser {parser:?} must not receive M3 thinking defaults"
        );
    }
}

/// The generation-prompt tail of the MiniMax M3 chat template (its
/// `add_generation_prompt` block): only an exact `thinking_mode ==
/// "disabled"` prefills `</mm:think>` and only an exact `"enabled"`

#[test]
fn prompt_invalid_request_maps_to_invalid_argument() {
    let error = PromptRenderError::invalid_request("unsupported model parameter").into();
    let mapped = map_prompt_render_error(error);

    assert_eq!(
        mapped,
        PromptRenderError::invalid_request("unsupported model parameter")
    );
}

#[test]
fn ordinary_prompt_error_maps_to_invalid_argument() {
    let mapped = map_prompt_render_error(anyhow::anyhow!("template configuration failed"));

    assert_eq!(
        mapped,
        PromptRenderError::invalid_request("template configuration failed")
    );
}

// ----- Tests added with the move -----

#[test]
fn continue_final_message_rejects_reordered_tool_results() {
    // The `apply_template` half of Dynamo's test of the same name.
    let image = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==";
    let mut request: TestRequest = serde_json::from_value(serde_json::json!({
        "model":"alias", "continue_final_message":true, "add_generation_prompt":false,
        "messages":[
            {"role":"assistant","tool_calls":[
                {"id":"a","type":"function","function":{"name":"image","arguments":"{}"}},
                {"id":"b","type":"function","function":{"name":"image","arguments":"{}"}}
            ]},
            {"role":"tool","tool_call_id":"b","content":[
                {"type":"image_url","image_url":{"url":image}}, {"type":"text","text":"Result B"}
            ]},
            {"role":"tool","tool_call_id":"a","content":[
                {"type":"image_url","image_url":{"url":image}}, {"type":"text","text":"Result A"}
            ]}
        ]
    }))
    .unwrap();
    let formatter = crate::deepseek::v41::DeepSeekV41Formatter;
    let error = apply_template(&formatter, &request, false).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("Cannot continue the final message"),
        "{error}"
    );

    request.inner.messages.swap(1, 2);
    let ordered = apply_template(&formatter, &request, false)
        .unwrap()
        .unwrap();
    assert_eq!(ordered.as_str().matches("<｜deepseek_image｜>").count(), 2);
    assert!(ordered.as_str().ends_with("Result B"));
}

#[test]
fn gpt_oss_rejects_reasoning_effort_none() {
    let args =
        |effort: serde_json::Value| Some(HashMap::from([("reasoning_effort".to_string(), effort)]));
    let error = thinking::reject_unsupported_reasoning_effort(
        Some("gpt_oss"),
        args("none".into()).as_ref(),
    )
    .unwrap_err();
    assert_eq!(
        error.to_string(),
        "GPT-OSS does not support reasoning_effort=none; use low, medium, or high"
    );
    for effort in ["low", "medium", "high"] {
        assert!(
            thinking::reject_unsupported_reasoning_effort(
                Some("gpt_oss"),
                args(effort.into()).as_ref()
            )
            .is_ok()
        );
    }
    // Other families, a missing key, and a non-string value pass.
    for family in [None, Some("qwen3"), Some("deepseek_v4")] {
        assert!(
            thinking::reject_unsupported_reasoning_effort(family, args("none".into()).as_ref())
                .is_ok()
        );
    }
    assert!(thinking::reject_unsupported_reasoning_effort(Some("gpt_oss"), None).is_ok());
    assert!(
        thinking::reject_unsupported_reasoning_effort(
            Some("gpt_oss"),
            args(serde_json::Value::Null).as_ref()
        )
        .is_ok()
    );
}

#[test]
fn raw_prompt_replaces_the_template_when_present() {
    let formatter = test_prompt_formatter("{% for m in messages %}<{{ m.content }}>{% endfor %}");
    let request = chat_request_with_args(None);
    let options = ChatPromptOptions {
        use_raw_prompt: true,
        ..Default::default()
    };

    let raw = render_chat_prompt(formatter.as_ref(), &request, options, || {
        Some("raw text".to_string())
    })
    .unwrap()
    .unwrap();
    assert_eq!(raw.as_str(), "raw text");

    // A requested raw prompt that is not available falls back to the template.
    let fallback = render_chat_prompt(formatter.as_ref(), &request, options, || None)
        .unwrap()
        .unwrap();
    assert_eq!(fallback.as_str(), "<hi>");

    // Without `use_raw_prompt` the raw prompt is never read.
    let templated = render_chat_prompt(
        formatter.as_ref(),
        &request,
        ChatPromptOptions::default(),
        || panic!("raw prompt must not be read"),
    )
    .unwrap()
    .unwrap();
    assert_eq!(templated.as_str(), "<hi>");
}

/// A formatter that fails the way a misconfigured template does.
struct FailingFormatter;

impl OAIPromptFormatter for FailingFormatter {
    fn supports_add_generation_prompt(&self) -> bool {
        true
    }

    fn render(&self, _req: &dyn OAIChatLikeRequest) -> anyhow::Result<String> {
        Err(anyhow::anyhow!("missing tool_use key").context("template configuration failed"))
    }
}

#[test]
fn render_failure_is_an_invalid_request_with_the_cause_chain() {
    let request = chat_request_with_args(None);

    let error = render_chat_prompt(
        &FailingFormatter,
        &request,
        ChatPromptOptions::default(),
        || None,
    )
    .unwrap_err();
    assert_eq!(
        error,
        PromptRenderError::invalid_request("template configuration failed: missing tool_use key")
    );
}

#[test]
fn continue_final_message_with_empty_final_content_is_an_invalid_request() {
    let formatter = test_prompt_formatter("{% for m in messages %}{{ m.content }}{% endfor %}");
    let request: TestRequest = serde_json::from_value(serde_json::json!({
        "model": "test-model",
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": ""}
        ],
        "add_generation_prompt": false,
        "continue_final_message": true
    }))
    .unwrap();

    let error = apply_template(formatter.as_ref(), &request, false).unwrap_err();
    assert_eq!(
        error,
        PromptRenderError::invalid_request(
            "Unable to continue the final message because it was not found in the rendered chat template."
        )
    );
}

/// A request type with no single text prompt, such as pre-tokenized input.
struct TokenPromptRequest(TestRequest);

impl OAIChatLikeRequest for TokenPromptRequest {
    fn model(&self) -> String {
        self.0.model()
    }

    fn messages(&self) -> minijinja::value::Value {
        self.0.messages()
    }

    fn should_add_generation_prompt(&self) -> bool {
        self.0.should_add_generation_prompt()
    }

    fn prompt_input_type(&self) -> crate::PromptInput {
        crate::PromptInput::Tokens(crate::TokenInput::Single(vec![]))
    }

    fn extract_tokens(&self) -> Option<crate::TokenInput> {
        Some(crate::TokenInput::Single(vec![1, 2, 3]))
    }

    fn extract_text(&self) -> Option<TextInput> {
        None
    }
}

#[test]
fn token_prompt_renders_nothing_with_or_without_the_rules() {
    let formatter = test_prompt_formatter("{% for m in messages %}{{ m.content }}{% endfor %}");
    let request = TokenPromptRequest(continue_request());
    for normalize_tool_call_args in [false, true] {
        for continue_final_message in [false, true] {
            let options = ChatPromptOptions {
                normalize_tool_call_args,
                continue_final_message,
                use_raw_prompt: false,
            };
            assert_eq!(
                render_chat_prompt(formatter.as_ref(), &request, options, || None),
                Ok(None),
                "{options:?}"
            );
        }
    }
}

/// A request whose messages carry replay metadata that only the
/// template-messages override can express.
struct ReplayRequest {
    inner: TestRequest,
    replay: serde_json::Value,
}

impl OAIChatLikeRequest for ReplayRequest {
    fn model(&self) -> String {
        self.inner.model()
    }

    fn messages(&self) -> minijinja::value::Value {
        self.inner.messages()
    }

    fn template_messages_json_override(&self) -> Option<serde_json::Value> {
        Some(self.replay.clone())
    }

    fn typed_messages(&self) -> Option<&[dynamo_protocols::types::ChatCompletionRequestMessage]> {
        self.inner.typed_messages()
    }

    fn should_add_generation_prompt(&self) -> bool {
        self.inner.should_add_generation_prompt()
    }

    fn extract_text(&self) -> Option<TextInput> {
        self.inner.extract_text()
    }
}

/// The rules apply to the override messages, so a request that replays
/// metadata through the override keeps it, and still gets the argument
/// normalization and the continue marker.
#[test]
fn rules_apply_to_template_messages_override() {
    let request = ReplayRequest {
        inner: continue_request(),
        replay: serde_json::json!([
            {"role": "assistant", "content": "", "replay": "R1", "tool_calls": [{
                "id": "c1", "type": "function",
                "function": {"name": "lookup", "arguments": "{\"query\":\"hello\"}"}
            }]},
            {"role": "assistant", "content": "partial", "replay": "R2"}
        ]),
    };
    let formatter = test_prompt_formatter(
        "{% for m in messages %}[{{ m.replay }}]{% for call in m.tool_calls %}{% if call.function.arguments is string %}RAW{% else %}{{ call.function.arguments.query }}{% endif %}{% endfor %}{{ m.content }}|{% endfor %}closed",
    );
    let options = ChatPromptOptions {
        normalize_tool_call_args: true,
        continue_final_message: true,
        use_raw_prompt: false,
    };

    let prompt = render_chat_prompt(formatter.as_ref(), &request, options, || None)
        .unwrap()
        .unwrap();
    assert_eq!(prompt.as_str(), "[R1]hello|[R2]partial");
}

#[test]
fn glm47_or_json_object_format_normalizes_tool_call_arguments() {
    use super::tool_arguments::should_normalize_tool_call_arguments;

    assert!(should_normalize_tool_call_arguments(Some("glm47"), false));
    assert!(should_normalize_tool_call_arguments(None, true));
    assert!(should_normalize_tool_call_arguments(Some("hermes"), true));
    for parser in [None, Some("glm45"), Some("hermes"), Some("harmony")] {
        assert!(
            !should_normalize_tool_call_arguments(parser, false),
            "{parser:?}"
        );
    }
}

/// Client intent is read before the default is applied, as in Dynamo's
/// `generate`, so the default never counts as client intent.
#[test]
fn thinking_rules_read_client_intent_before_the_default() {
    let rules = ThinkingRules {
        reasoning_parser: Some("minimax_m3"),
        tool_call_parser: None,
        default_thinking_mode: Some("enabled"),
    };

    // No client control: the default applies, and a constrained M3 request
    // still renders with thinking disabled.
    let mut args = None;
    rules.apply(&mut args, false, true);
    let args = args.unwrap();
    assert_eq!(
        args.get("thinking_mode"),
        Some(&serde_json::json!("disabled"))
    );
    assert_eq!(args.get("thinking"), Some(&serde_json::json!(false)));

    // Without constrained output the default stands.
    let mut args = None;
    rules.apply(&mut args, false, false);
    let args = args.unwrap();
    assert_eq!(
        args.get("thinking_mode"),
        Some(&serde_json::json!("enabled"))
    );
    assert_eq!(args.get("enable_thinking"), Some(&serde_json::json!(true)));

    // A top-level thinking field is client intent: no default, no M3
    // constrained override, and no alias to normalize.
    let mut args = None;
    rules.apply(&mut args, true, true);
    assert_eq!(args, None);
}
