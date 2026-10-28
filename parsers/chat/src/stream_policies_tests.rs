// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! dynamo-llm's unit tests for the stream policies, moved with them.
//!
//! They run on stand-ins for dynamo's stream types, with the same field names:
//! dynamo-runtime's `Annotated` and dynamo-llm's
//! `NvCreateChatCompletionStreamResponse` and `LLMMetricAnnotation`. Tests that
//! also check the batch path, the v1 jail, or prompt rendering keep only their
//! stream-policy half here; the rest stays in dynamo.

use std::collections::HashMap;

use dynamo_parsers::chat_stream::ChatStreamItem;
use dynamo_protocols::types::{
    ChatChoiceStream, ChatCompletionMessageContent, ChatCompletionMessageToolCallChunk,
    ChatCompletionStreamResponseDelta, CompletionUsage, CreateChatCompletionRequest,
    CreateChatCompletionStreamResponse, FinishReason, FunctionCallStream, FunctionType, Role,
};
use futures::{StreamExt, stream};

use super::*;
use crate::guided::GuidedToolConstraint;
use crate::nonempty_content::stream_can_defer_all_output;
use crate::reasoning_off::{
    ChatStreamRequest, is_reasoning_disabled_by_request, request_disables_reasoning_with_family,
};
use crate::reasoning_usage::annotate_reasoning_usage;
use crate::stream_item::{AnnotationFields, ToolCallCompletion, UnifiedStreamChunk};
use crate::tool_call_policy::normalize_chat_stream_roles;
use crate::unified_stream::apply_stream_with_constraint;
use dynamo_parsers_v2::UnifiedParserStartingState;

/// dynamo-runtime's `Annotated`, with a string error.
#[derive(Clone, Debug)]
struct Annotated<R> {
    data: Option<R>,
    id: Option<String>,
    event: Option<String>,
    comment: Option<Vec<String>>,
    error: Option<String>,
}

impl<R> Annotated<R> {
    fn from_data(data: R) -> Self {
        Self {
            data: Some(data),
            id: None,
            event: None,
            comment: None,
            error: None,
        }
    }
}

/// The fields of dynamo-llm's `LLMMetricAnnotation` that these tests set.
#[derive(Clone, Debug, Default, PartialEq)]
struct LLMMetricAnnotation {
    output_tokens: usize,
    chunk_tokens: usize,
}

/// dynamo-llm's `NvCreateChatCompletionStreamResponse`.
#[derive(Clone, Debug)]
struct NvCreateChatCompletionStreamResponse {
    inner: CreateChatCompletionStreamResponse,
    nvext: Option<serde_json::Value>,
    prompt_logprobs: Option<std::sync::Arc<Vec<u32>>>,
    llm_metrics: Option<LLMMetricAnnotation>,
    tool_call_completion: Vec<ToolCallCompletion>,
}

type Item = Annotated<NvCreateChatCompletionStreamResponse>;

impl UnifiedStreamChunk for NvCreateChatCompletionStreamResponse {
    fn inner(&self) -> &CreateChatCompletionStreamResponse {
        &self.inner
    }

    fn inner_mut(&mut self) -> &mut CreateChatCompletionStreamResponse {
        &mut self.inner
    }

    fn tool_call_completion(&self) -> &[ToolCallCompletion] {
        &self.tool_call_completion
    }

    fn tool_call_completion_mut(&mut self) -> &mut Vec<ToolCallCompletion> {
        &mut self.tool_call_completion
    }

    fn clear_nvext(&mut self) {
        self.nvext = None;
    }

    fn clear_llm_metrics(&mut self) {
        self.llm_metrics = None;
    }

    fn scrub_synthetic(&mut self) {
        self.inner.usage = None;
        self.llm_metrics = None;
        self.tool_call_completion.clear();
        self.nvext = None;
        self.prompt_logprobs = None;
    }
}

impl UnifiedStreamItem for Item {
    type Chunk = NvCreateChatCompletionStreamResponse;
    type Annotation = AnnotationFields;

    fn is_error(&self) -> bool {
        self.event.as_deref() == Some("error")
    }

    fn into_parts(self) -> Result<(Self::Chunk, AnnotationFields), Self> {
        match self.data {
            Some(data) => Ok((
                data,
                AnnotationFields {
                    id: self.id,
                    event: self.event,
                    comment: self.comment,
                    error: self.error,
                },
            )),
            None => Err(self),
        }
    }

    fn from_parts(data: Self::Chunk, annotation: AnnotationFields) -> Self {
        Self {
            data: Some(data),
            id: annotation.id,
            event: annotation.event,
            comment: annotation.comment,
            error: annotation.error,
        }
    }
}

/// dynamo's `scrub_synthetic_chunk_metadata`.
impl ChatStreamItem for Item {
    fn is_error(&self) -> bool {
        self.event.as_deref() == Some("error")
    }

    fn chunk(&self) -> Option<&CreateChatCompletionStreamResponse> {
        self.data.as_ref().map(|data| &data.inner)
    }

    fn chunk_mut(&mut self) -> Option<&mut CreateChatCompletionStreamResponse> {
        self.data.as_mut().map(|data| &mut data.inner)
    }

    fn scrub_synthetic(&mut self) {
        self.event = None;
        self.comment = None;
        self.error = None;
        if let Some(data) = self.data.as_mut() {
            UnifiedStreamChunk::scrub_synthetic(data);
        }
    }
}

impl PolicyStreamItem for Item {
    fn has_error(&self) -> bool {
        self.error.is_some()
    }

    fn clear_id(&mut self) {
        self.id = None;
    }

    fn chunk_tokens(&self) -> usize {
        self.data
            .as_ref()
            .and_then(|data| data.llm_metrics.as_ref())
            .map_or(0, |metrics| metrics.chunk_tokens)
    }
}

fn chat_stream_chunk(index: u32, role: Option<Role>) -> Item {
    #[allow(deprecated)]
    let choice = ChatChoiceStream {
        index,
        delta: ChatCompletionStreamResponseDelta {
            role,
            content: Some(ChatCompletionMessageContent::Text("content".to_string())),
            tool_calls: None,
            function_call: None,
            refusal: None,
            reasoning_content: None,
        },
        finish_reason: None,
        logprobs: None,
    };
    Annotated::from_data(NvCreateChatCompletionStreamResponse {
        inner: CreateChatCompletionStreamResponse {
            id: "test".to_string(),
            choices: vec![choice],
            created: 0,
            model: "test".to_string(),
            system_fingerprint: None,
            object: "chat.completion.chunk".to_string(),
            usage: None,
            service_tier: None,
        },
        nvext: None,
        prompt_logprobs: None,
        llm_metrics: None,
        tool_call_completion: Vec::new(),
    })
}

fn reasoning_usage_chunk(
    reasoning: Option<&str>,
    content: Option<&str>,
    chunk_tokens: usize,
) -> Item {
    let mut chunk = chat_stream_chunk(0, None);
    let data = chunk.data.as_mut().unwrap();
    data.inner.choices[0].delta.reasoning_content = reasoning.map(str::to_string);
    data.inner.choices[0].delta.content = content
        .map(str::to_string)
        .map(ChatCompletionMessageContent::Text);
    data.llm_metrics = Some(LLMMetricAnnotation {
        chunk_tokens,
        ..Default::default()
    });
    chunk
}

fn reasoning_usage_trailer(completion_tokens: u32, backend_reasoning_tokens: Option<u32>) -> Item {
    let mut chunk = chat_stream_chunk(0, None);
    let data = chunk.data.as_mut().unwrap();
    data.inner.choices.clear();
    let mut usage = CompletionUsage {
        prompt_tokens: 5,
        completion_tokens,
        total_tokens: 5 + completion_tokens,
        prompt_tokens_details: None,
        completion_tokens_details: None,
    };
    if let Some(reasoning_tokens) = backend_reasoning_tokens {
        usage
            .completion_tokens_details
            .get_or_insert_default()
            .reasoning_tokens = Some(reasoning_tokens);
    }
    data.inner.usage = Some(usage);
    data.llm_metrics = Some(LLMMetricAnnotation {
        output_tokens: completion_tokens as usize,
        ..Default::default()
    });
    chunk
}

fn trailer_reasoning_tokens(output: &[Item]) -> Option<u32> {
    output
        .last()
        .and_then(|response| response.data.as_ref())
        .and_then(|chunk| chunk.inner.usage.as_ref())
        .expect("usage trailer")
        .completion_tokens_details
        .as_ref()
        .and_then(|details| details.reasoning_tokens)
}

#[tokio::test]
async fn reasoning_usage_estimator_stamps_shared_chat_usage() {
    let output = annotate_reasoning_usage(stream::iter(vec![
        reasoning_usage_chunk(Some("think"), None, 1),
        reasoning_usage_chunk(Some(" more"), None, 1),
        reasoning_usage_chunk(None, None, 1),
        reasoning_usage_chunk(None, Some("answer"), 1),
        reasoning_usage_trailer(4, None),
    ]))
    .collect::<Vec<_>>()
    .await;

    assert_eq!(trailer_reasoning_tokens(&output), Some(3));
}

#[tokio::test]
async fn reasoning_usage_estimator_documents_mixed_chunk_overcount() {
    let output = annotate_reasoning_usage(stream::iter(vec![
        reasoning_usage_chunk(Some("think"), Some("answer"), 4),
        reasoning_usage_trailer(4, None),
    ]))
    .collect::<Vec<_>>()
    .await;

    assert_eq!(
        trailer_reasoning_tokens(&output),
        Some(4),
        "the chunk-granular estimate intentionally attributes the mixed chunk to reasoning"
    );
}

#[tokio::test]
async fn reasoning_usage_estimator_preserves_positive_backend_count() {
    let output = annotate_reasoning_usage(stream::iter(vec![
        reasoning_usage_chunk(Some("two estimated tokens"), None, 2),
        reasoning_usage_trailer(2, Some(1)),
    ]))
    .collect::<Vec<_>>()
    .await;

    assert_eq!(trailer_reasoning_tokens(&output), Some(1));
}

/// The stream half of dynamo's
/// `disabled_unified_reasoning_keeps_interleaved_text_order_in_stream_and_batch`.
/// The batch half checks the same text and stays in dynamo with the batch path.
#[tokio::test]
async fn disabled_unified_reasoning_keeps_interleaved_text_order_in_stream() {
    for (family, input) in [
        (
            "kimi_k3",
            "<|open|>response<|sep|>before<|close|>response<|sep|><|open|>think<|sep|>hidden<|close|>think<|sep|><|open|>response<|sep|>after<|close|>response<|sep|>",
        ),
        (
            "muse_glimmer",
            "<|start|>assistant to=user<|message|>before<|eom|><|start|>assistant to=self<|message|>hidden<|eom|><|start|>assistant to=user<|message|>after<|eot|>",
        ),
    ] {
        for split in input.char_indices().map(|(offset, _)| offset) {
            let (first, second) = input.split_at(split);
            let mut terminal = chat_stream_chunk(0, None);
            terminal.data.as_mut().unwrap().inner.choices[0]
                .delta
                .content = Some(ChatCompletionMessageContent::Text(second.to_string()));
            let mut first_chunk = chat_stream_chunk(0, None);
            first_chunk.data.as_mut().unwrap().inner.choices[0]
                .delta
                .content = Some(ChatCompletionMessageContent::Text(first.to_string()));
            terminal.data.as_mut().unwrap().inner.choices[0].finish_reason =
                Some(FinishReason::Stop);
            let raw = apply_stream_with_constraint(
                stream::iter([first_chunk, terminal]),
                None,
                GuidedToolConstraint::None,
                UnifiedParserStartingState::None,
                family,
                true,
            );
            let output = apply_unified_response_policies(raw, true, false, true)
                .collect::<Vec<_>>()
                .await;
            let mut text = String::new();
            for choice in output
                .iter()
                .filter_map(|r| r.data.as_ref())
                .flat_map(|r| &r.inner.choices)
            {
                assert!(
                    choice.delta.reasoning_content.is_none(),
                    "{family} split={split}"
                );
                if let Some(ChatCompletionMessageContent::Text(t)) = &choice.delta.content {
                    text.push_str(t);
                }
            }
            assert_eq!(text, "beforehiddenafter", "{family} split={split}");
        }
    }
}

#[tokio::test]
async fn force_nonempty_deferral_preserves_reasoning_usage() {
    let reasoning = reasoning_usage_chunk(Some("deferred thought"), None, 3);
    let mut terminal = chat_stream_chunk(0, None);
    let terminal_choice = &mut terminal.data.as_mut().unwrap().inner.choices[0];
    terminal_choice.delta.content = None;
    terminal_choice.finish_reason = Some(FinishReason::Stop);

    let output = apply_unified_response_policies(
        stream::iter(vec![reasoning, terminal, reasoning_usage_trailer(3, None)]),
        true,
        true,
        false,
    )
    .collect::<Vec<_>>()
    .await;

    assert_eq!(
        trailer_reasoning_tokens(&output),
        Some(3),
        "deferral must not erase the parser's reasoning classification"
    );
    assert_eq!(
        output
            .iter()
            .filter_map(|response| response.data.as_ref())
            .filter(|chunk| chunk.llm_metrics.is_some())
            .count(),
        2,
        "one source metric and one usage-trailer metric must survive"
    );
}

#[tokio::test]
async fn test_normalize_chat_stream_roles_recovers_missing_first_role_per_choice() {
    let input = stream::iter(vec![
        // Mirrors a parser releasing buffered content without the role that
        // arrived on the original, swallowed chunk.
        chat_stream_chunk(0, None),
        chat_stream_chunk(1, None),
        chat_stream_chunk(0, Some(Role::Assistant)),
        chat_stream_chunk(1, Some(Role::Assistant)),
    ]);

    let output: Vec<_> = normalize_chat_stream_roles(input).collect().await;
    let roles: Vec<_> = output
        .iter()
        .map(|response| response.data.as_ref().unwrap().inner.choices[0].delta.role)
        .collect();

    assert_eq!(
        roles,
        vec![Some(Role::Assistant), Some(Role::Assistant), None, None,]
    );
}

/// The policy half of dynamo's `apply_kimi_k3_no_tools`: the jail (which stays
/// in dynamo) emitted a call for a request with no tools. The call is dropped
/// and the `tool_calls` finish becomes `stop`.
#[tokio::test]
async fn tool_call_policy_drops_calls_the_request_did_not_permit() {
    let mut call = chat_stream_chunk(0, Some(Role::Assistant));
    let choice = &mut call.data.as_mut().unwrap().inner.choices[0];
    choice.delta.content = None;
    choice.delta.tool_calls = Some(vec![ChatCompletionMessageToolCallChunk {
        index: 0,
        id: Some("call_1".to_string()),
        r#type: Some(FunctionType::Function),
        function: Some(FunctionCallStream {
            name: Some("lookup".to_string()),
            arguments: Some("{}".to_string()),
        }),
        ..Default::default()
    }]);
    choice.finish_reason = Some(FinishReason::ToolCalls);

    for (permitted, finish) in [(false, FinishReason::Stop), (true, FinishReason::ToolCalls)] {
        let output: Vec<_> = crate::tool_call_policy::apply_tool_call_response_policy(
            stream::iter([call.clone()]),
            permitted,
        )
        .collect()
        .await;
        let choice = &output[0].data.as_ref().unwrap().inner.choices[0];
        assert_eq!(choice.delta.tool_calls.is_some(), permitted);
        assert_eq!(choice.finish_reason, Some(finish));
    }
}

fn minimax_m3_request(
    body: serde_json::Value,
) -> (
    CreateChatCompletionRequest,
    Option<HashMap<String, serde_json::Value>>,
) {
    let mut base = serde_json::json!({
        "model": "MiniMaxAI/MiniMax-M3",
        "messages": [{"role": "user", "content": "hi"}],
    });
    let base_obj = base.as_object_mut().unwrap();
    let mut args = None;
    for (k, v) in body.as_object().unwrap() {
        if k == "chat_template_kwargs" {
            args = Some(serde_json::from_value(v.clone()).unwrap());
        } else {
            base_obj.insert(k.clone(), v.clone());
        }
    }
    (serde_json::from_value(base).unwrap(), args)
}

fn view<'a>(
    request: &'a (
        CreateChatCompletionRequest,
        Option<HashMap<String, serde_json::Value>>,
    ),
    explicit_guided_decoding: bool,
) -> ChatStreamRequest<'a> {
    ChatStreamRequest {
        request: &request.0,
        chat_template_args: request.1.as_ref(),
        client_thinking_field: false,
        explicit_guided_decoding,
    }
}

/// The reasoning-parser half of dynamo's
/// `test_normalize_thinking_arg_operator_default_forces_m3_constrained_disabled`.
/// Its rendering half moved with the request rules into dynamo-renderer.
#[test]
fn test_m3_operator_default_forces_constrained_reasoning_off() {
    let request = minimax_m3_request(serde_json::json!({
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "s", "schema": {"type": "object"}}
        }
    }));
    let original_args = request.1.clone();
    assert!(request_disables_reasoning_with_family(
        &view(&request, false),
        None,
        Some("minimax_m3"),
        Some("enabled"),
        Some("minimax_m3"),
    ));
    assert_eq!(request.1, original_args);
}

/// The reasoning-parser half of dynamo's
/// `test_normalize_thinking_arg_m3_preserves_explicit_thinking_for_constrained`.
#[test]
fn test_m3_explicit_thinking_keeps_constrained_reasoning_on() {
    for (body, desc) in [
        (
            serde_json::json!({
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "s", "schema": {"type": "object"}}
                },
                "chat_template_kwargs": {"thinking_mode": "enabled"}
            }),
            "json_schema + explicit thinking_mode=enabled",
        ),
        (
            serde_json::json!({
                "tools": [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}],
                "tool_choice": "required",
                "chat_template_kwargs": {"thinking_mode": "adaptive"}
            }),
            "required tool + explicit thinking_mode=adaptive",
        ),
        (
            serde_json::json!({
                "tools": [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}],
                "tool_choice": {"type": "function", "function": {"name": "f"}},
                "chat_template_kwargs": {"thinking": true}
            }),
            "named tool + explicit thinking=true",
        ),
    ] {
        let request = minimax_m3_request(body);
        assert!(
            !request_disables_reasoning_with_family(
                &view(&request, false),
                None,
                Some("minimax_m3"),
                Some("disabled"),
                Some("minimax_m3"),
            ),
            "{desc}"
        );
    }
}

/// The reasoning-parser half of dynamo's
/// `test_normalize_thinking_arg_m3_defaults_disabled_for_explicit_guided_decoding`.
/// dynamo reads the `guided_*` fields from its own request type; here the
/// caller passes `explicit_guided_decoding`.
#[test]
fn test_m3_explicit_guided_decoding_turns_reasoning_off() {
    for (reasoning_parser, tool_call_parser) in [
        (Some("minimax_m3"), None),
        (Some("minimax-m3"), None),
        (Some("minimax_m3"), Some("minimax_m3")),
    ] {
        let request = minimax_m3_request(serde_json::json!({}));
        let desc = format!("{reasoning_parser:?}/{tool_call_parser:?}");
        assert!(
            request_disables_reasoning_with_family(
                &view(&request, true),
                tool_call_parser,
                reasoning_parser,
                None,
                None,
            ),
            "{desc} must not run the reasoning parser"
        );
        assert!(
            !request_disables_reasoning_with_family(
                &view(&request, false),
                tool_call_parser,
                reasoning_parser,
                None,
                None,
            ),
            "{desc} without a grammar keeps the reasoning parser"
        );
    }
}

#[test]
fn muse_force_nonempty_stream_reports_that_it_can_defer_all_output() {
    let mut args = HashMap::new();
    args.insert(
        "force_nonempty_content".to_string(),
        serde_json::Value::Bool(true),
    );

    assert!(stream_can_defer_all_output(
        None,
        Some("muse_glimmer"),
        Some(&args),
        crate::version::ParserVersion::Auto,
    ));
}

/// dynamo's `responses_force_nonempty_request_requires_fallback_keep_alive`
/// (`http/service/openai.rs`), on the predicate it calls.
#[test]
fn responses_force_nonempty_request_requires_fallback_keep_alive() {
    let auto = crate::version::ParserVersion::Auto;
    let mut chat_template_args = HashMap::new();

    assert!(!stream_can_defer_all_output(
        Some("qwen3_coder"),
        Some("qwen3"),
        None,
        auto
    ));

    chat_template_args.insert(
        "force_nonempty_content".to_string(),
        serde_json::Value::Bool(false),
    );
    assert!(!stream_can_defer_all_output(
        Some("qwen3_coder"),
        Some("qwen3"),
        Some(&chat_template_args),
        auto
    ));

    chat_template_args.insert(
        "force_nonempty_content".to_string(),
        serde_json::Value::Bool(true),
    );
    assert!(stream_can_defer_all_output(
        Some("qwen3_coder"),
        Some("qwen3"),
        Some(&chat_template_args),
        auto
    ));

    assert!(stream_can_defer_all_output(
        Some("muse_glimmer"),
        None,
        Some(&chat_template_args),
        auto
    ));
}

#[test]
fn test_is_reasoning_disabled_by_request() {
    let thinking_true = {
        let mut m = std::collections::HashMap::new();
        m.insert("thinking".to_string(), serde_json::Value::Bool(true));
        m
    };
    let thinking_false = {
        let mut m = std::collections::HashMap::new();
        m.insert("thinking".to_string(), serde_json::Value::Bool(false));
        m
    };
    let enable_thinking_true = {
        let mut m = std::collections::HashMap::new();
        m.insert("enable_thinking".to_string(), serde_json::Value::Bool(true));
        m
    };
    let enable_thinking_false = {
        let mut m = std::collections::HashMap::new();
        m.insert(
            "enable_thinking".to_string(),
            serde_json::Value::Bool(false),
        );
        m
    };
    let force_nonempty_content_true = {
        let mut m = std::collections::HashMap::new();
        m.insert(
            "force_nonempty_content".to_string(),
            serde_json::Value::Bool(true),
        );
        m
    };
    let thinking_mode_chat = {
        let mut m = std::collections::HashMap::new();
        m.insert(
            "thinking_mode".to_string(),
            serde_json::Value::String("chat".to_string()),
        );
        m
    };
    let thinking_mode_thinking = {
        let mut m = std::collections::HashMap::new();
        m.insert(
            "thinking_mode".to_string(),
            serde_json::Value::String("thinking".to_string()),
        );
        m
    };
    let thinking_mode_disabled = {
        let mut m = std::collections::HashMap::new();
        m.insert(
            "thinking_mode".to_string(),
            serde_json::Value::String("disabled".to_string()),
        );
        m
    };
    let reasoning_effort_none = {
        let mut m = std::collections::HashMap::new();
        m.insert(
            "reasoning_effort".to_string(),
            serde_json::Value::String("none".to_string()),
        );
        m
    };
    let reasoning_effort_high = {
        let mut m = std::collections::HashMap::new();
        m.insert(
            "reasoning_effort".to_string(),
            serde_json::Value::String("high".to_string()),
        );
        m
    };
    let empty_args = std::collections::HashMap::new();

    // (parser, args, expected_disabled, description)
    let cases = [
        (
            Some("kimi_k25"),
            Some(&thinking_false),
            true,
            "kimi_k25 + thinking=false → disabled",
        ),
        (
            Some("kimi_k25"),
            Some(&thinking_true),
            false,
            "kimi_k25 + thinking=true → enabled",
        ),
        (
            Some("kimi_k25"),
            None,
            false,
            "kimi_k25 + no args → enabled",
        ),
        (
            Some("kimi_k25"),
            Some(&empty_args),
            false,
            "kimi_k25 + empty args → enabled",
        ),
        (
            Some("kimi_k3"),
            Some(&thinking_false),
            true,
            "kimi_k3 + thinking=false → disabled",
        ),
        (
            Some("kimi-k3"),
            Some(&thinking_true),
            false,
            "kimi-k3 + thinking=true → enabled",
        ),
        // deepseek_r1 uses "thinking" bool or "thinking_mode" string
        (
            Some("deepseek_r1"),
            Some(&thinking_false),
            true,
            "deepseek_r1 + thinking=false → disabled",
        ),
        (
            Some("deepseek_r1"),
            Some(&thinking_true),
            false,
            "deepseek_r1 + thinking=true → enabled",
        ),
        (
            Some("deepseek_r1"),
            Some(&thinking_mode_chat),
            true,
            "deepseek_r1 + thinking_mode=chat → disabled",
        ),
        (
            Some("deepseek_r1"),
            Some(&thinking_mode_thinking),
            false,
            "deepseek_r1 + thinking_mode=thinking → enabled",
        ),
        (
            Some("deepseek_r1"),
            None,
            false,
            "deepseek_r1 + no args → enabled",
        ),
        (
            Some("deepseek_r1"),
            Some(&empty_args),
            false,
            "deepseek_r1 + empty args → enabled",
        ),
        (
            Some("deepseek_v3"),
            None,
            true,
            "deepseek_v3 + no args → disabled",
        ),
        (
            Some("deepseek_v3"),
            Some(&thinking_true),
            false,
            "deepseek_v3 + thinking=true → enabled",
        ),
        (
            Some("deepseek_v3_1"),
            Some(&enable_thinking_true),
            false,
            "deepseek_v3_1 + enable_thinking=true → enabled",
        ),
        (
            Some("deepseek_v3_2"),
            None,
            false,
            "deepseek_v3_2 + no args → enabled",
        ),
        (
            Some("deepseek_v3_2"),
            Some(&thinking_false),
            true,
            "deepseek_v3_2 + thinking=false → disabled",
        ),
        (
            Some("minimax_m2"),
            Some(&thinking_false),
            true,
            "minimax_m2 + thinking=false → disabled",
        ),
        (
            Some("minimax_m2"),
            Some(&thinking_true),
            false,
            "minimax_m2 + thinking=true → enabled",
        ),
        (
            Some("minimax_m2"),
            None,
            false,
            "minimax_m2 + no args → enabled",
        ),
        (
            Some("basic"),
            Some(&thinking_false),
            false,
            "basic → never disabled",
        ),
        (
            None,
            Some(&thinking_false),
            false,
            "no parser → never disabled",
        ),
        // nemotron_nano uses "enable_thinking" key
        (
            Some("nemotron_nano"),
            Some(&enable_thinking_false),
            true,
            "nemotron_nano + enable_thinking=false → disabled",
        ),
        (
            Some("nemotron_nano"),
            Some(&enable_thinking_true),
            false,
            "nemotron_nano + enable_thinking=true → enabled",
        ),
        (
            Some("nemotron_nano"),
            None,
            false,
            "nemotron_nano + no args → enabled",
        ),
        (
            Some("nemotron_nano"),
            Some(&empty_args),
            false,
            "nemotron_nano + empty args → enabled",
        ),
        (
            Some("nemotron3"),
            Some(&force_nonempty_content_true),
            false,
            "nemotron3 + force_nonempty_content=true → NOT disabled (parser stays on)",
        ),
        (
            Some("nemotron_v3"),
            Some(&enable_thinking_false),
            true,
            "nemotron_v3 + enable_thinking=false → disabled",
        ),
        (
            Some("nemotron_v3"),
            Some(&force_nonempty_content_true),
            false,
            "nemotron_v3 + force_nonempty_content=true → NOT disabled (parser stays on)",
        ),
        // deepseek_v4 — same convention as deepseek_r1; verify all three aliases
        // (deepseek_v4 / deepseek-v4 / deepseekv4) plus both signal keys.
        (
            Some("deepseek_v4"),
            Some(&thinking_false),
            true,
            "deepseek_v4 + thinking=false → disabled",
        ),
        (
            Some("deepseek_v4"),
            Some(&thinking_true),
            false,
            "deepseek_v4 + thinking=true → enabled",
        ),
        (
            Some("deepseek_v4"),
            Some(&thinking_mode_chat),
            true,
            "deepseek_v4 + thinking_mode=chat → disabled",
        ),
        (
            Some("deepseek_v4"),
            Some(&thinking_mode_thinking),
            false,
            "deepseek_v4 + thinking_mode=thinking → enabled",
        ),
        (
            Some("deepseek_v4"),
            None,
            false,
            "deepseek_v4 + no args → enabled",
        ),
        (
            Some("deepseek-v4"),
            Some(&thinking_false),
            true,
            "deepseek-v4 (hyphen alias) + thinking=false → disabled",
        ),
        (
            Some("deepseekv4"),
            Some(&thinking_mode_chat),
            true,
            "deepseekv4 (joined alias) + thinking_mode=chat → disabled",
        ),
        (
            Some("deepseek_v4"),
            Some(&enable_thinking_false),
            true,
            "deepseek_v4 + enable_thinking=false → disabled (vLLM alias)",
        ),
        (
            Some("deepseek_v4"),
            Some(&enable_thinking_true),
            false,
            "deepseek_v4 + enable_thinking=true → enabled (vLLM alias)",
        ),
        (
            Some("gemma4"),
            Some(&enable_thinking_false),
            true,
            "gemma4 + enable_thinking=false → disabled",
        ),
        (
            Some("gemma4"),
            Some(&enable_thinking_true),
            false,
            "gemma4 + enable_thinking=true → enabled",
        ),
        (
            Some("gemma4"),
            None,
            true,
            "gemma4 + no args → disabled (reasoning is opt-in)",
        ),
        (
            Some("gemma-4"),
            Some(&enable_thinking_false),
            true,
            "gemma-4 (hyphen alias) + enable_thinking=false → disabled",
        ),
        (
            Some("gemma-4"),
            None,
            true,
            "gemma-4 (hyphen alias) + no args → disabled (reasoning is opt-in)",
        ),
        (Some("mistral"), None, true, "mistral + no args → disabled"),
        (
            Some("mistral"),
            Some(&reasoning_effort_none),
            true,
            "mistral + reasoning_effort=none → disabled",
        ),
        (
            Some("mistral"),
            Some(&reasoning_effort_high),
            false,
            "mistral + reasoning_effort=high → enabled",
        ),
        (
            Some("minimax_m3"),
            Some(&thinking_mode_disabled),
            true,
            "minimax_m3 + thinking_mode=disabled → disabled",
        ),
        (
            Some("minimax-m3"),
            Some(&thinking_mode_disabled),
            true,
            "minimax-m3 + thinking_mode=disabled → disabled",
        ),
        (
            Some("minimax_m3"),
            Some(&thinking_mode_thinking),
            false,
            "minimax_m3 + thinking_mode=thinking → enabled",
        ),
        (
            Some("minimax_m3"),
            None,
            false,
            "minimax_m3 + no args → enabled",
        ),
    ];

    // The disable decision no longer depends on streaming — `enable_thinking`
    // and the per-family signals decide it identically for both paths.
    for (parser, args, expected, desc) in cases {
        assert_eq!(
            is_reasoning_disabled_by_request(parser, args),
            expected,
            "FAILED: {desc}",
        );
    }

    // force_nonempty_content=true does NOT disable reasoning parsing for
    // either path: the parser stays on so reasoning is split from the answer
    // (no leak). Non-streaming additionally moves reasoning into content when
    // no content was generated (the aggregator); streaming skips that move.
    assert!(
        !is_reasoning_disabled_by_request(Some("nemotron3"), Some(&force_nonempty_content_true),),
        "nemotron3 + force_nonempty_content=true → NOT disabled",
    );
    assert!(
        !is_reasoning_disabled_by_request(Some("nemotron_v3"), Some(&force_nonempty_content_true),),
        "nemotron_v3 + force_nonempty_content=true → NOT disabled",
    );
    // enable_thinking=false disables entirely (user turned thinking off).
    assert!(
        is_reasoning_disabled_by_request(Some("nemotron3"), Some(&enable_thinking_false),),
        "nemotron3 + enable_thinking=false → disabled",
    );

    // The force_nonempty_content=true → NOT disabled behavior is what lets a
    // reasoning-only non-streaming turn surface reasoning as content; verify
    // the aggregator half in test_move_reasoning_to_content_when_empty.
}
