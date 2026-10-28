// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The recorder that wrote `stream_policies_parity.jsonl.gz`. This crate does not build it.
//!
//! It ran as a test inside dynamo-llm at ai-dynamo/dynamo `45e30e07`, as
//! `lib/llm/src/preprocessor/stream_policies_recorder.rs`, next to
//! `stream_policies_capture.rs`. `stream_policies_recorder.patch` adds both
//! modules to `preprocessor.rs`, and taps the v1 jail's input and output
//! inside `postprocessor_parsing_stream_with_constraint`, because the jail
//! stays in dynamo. The taps only copy items; they change no output.
//!
//! Run it once per `DYN_PARSER_VERSION` value, because dynamo-llm reads the
//! version once per process:
//!
//! ```text
//! git apply stream_policies_recorder.patch
//! PARITY_OUT=auto.jsonl cargo test -p dynamo-llm --lib stream_policies_recorder -- --ignored
//! DYN_PARSER_VERSION=1 PARITY_OUT=1.jsonl cargo test -p dynamo-llm --lib stream_policies_recorder -- --ignored
//! DYN_PARSER_VERSION=2 PARITY_OUT=2.jsonl cargo test -p dynamo-llm --lib stream_policies_recorder -- --ignored
//! python3 pack_stream_policies_parity.py stream_policies_parity.jsonl.gz auto.jsonl 1.jsonl 2.jsonl
//! ```

use super::*;
use crate::protocols::common::metrics::LLMMetricAnnotation;
use dynamo_protocols::types::{
    ChatChoiceStream, ChatCompletionMessageToolCallChunk, ChatCompletionStreamResponseDelta,
    CompletionUsage, CreateChatCompletionStreamResponse, FinishReason, FunctionCallStream,
    FunctionType, Role,
};
use dynamo_runtime::error::DynamoError;
use futures::stream;
use serde_json::{Map, Value, json};
use std::io::Write;
use std::sync::Mutex;

type Item = Annotated<NvCreateChatCompletionStreamResponse>;

#[derive(Default)]
struct JailTap {
    request: Value,
    input: Vec<Value>,
    input_ended: bool,
    output: Vec<Value>,
    called: bool,
}

static JAIL_TAP: Mutex<Option<JailTap>> = Mutex::new(None);

pub(super) fn tap_jail_request(request: Value) {
    if let Some(tap) = JAIL_TAP.lock().unwrap().as_mut() {
        tap.request = request;
    }
}

pub(super) fn tap_jail_input<S>(stream: S) -> Pin<Box<dyn Stream<Item = Item> + Send>>
where
    S: Stream<Item = Item> + Send + 'static,
{
    if let Some(tap) = JAIL_TAP.lock().unwrap().as_mut() {
        tap.called = true;
    }
    let end = stream::once(async {
        if let Some(tap) = JAIL_TAP.lock().unwrap().as_mut() {
            tap.input_ended = true;
        }
    })
    .filter_map(|_| async { None });
    Box::pin(
        stream
            .inspect(|item| {
                if let Some(tap) = JAIL_TAP.lock().unwrap().as_mut() {
                    tap.input.push(compact(item));
                }
            })
            .chain(end),
    )
}

pub(super) fn tap_jail_output<S>(stream: S) -> Pin<Box<dyn Stream<Item = Item> + Send>>
where
    S: Stream<Item = Item> + Send + 'static,
{
    Box::pin(stream.inspect(|item| {
        if let Some(tap) = JAIL_TAP.lock().unwrap().as_mut() {
            tap.output.push(compact(item));
        }
    }))
}

/// A deployment's parser pair. The first 8 are Rhino's families.
const PAIRS: &[(Option<&str>, Option<&str>)] = &[
    (None, Some("deepseek_v4")),
    (Some("deepseek_v4"), Some("deepseek_v4")),
    (Some("glm45"), Some("glm45")),
    (Some("glm47"), Some("glm45")),
    (Some("harmony"), Some("gpt_oss")),
    (Some("kimi_k2"), Some("basic")),
    (Some("minimax_m3"), Some("minimax_m3")),
    (Some("hermes"), Some("qwen3")),
    (None, Some("qwen3")),
    (None, None),
    (Some("hermes"), None),
    (None, Some("nemotron_nano")),
    (Some("hermes"), Some("nemotron_v3")),
    (None, Some("kimi_k25")),
    (None, Some("mistral")),
    (None, Some("deepseek_r1")),
    (None, Some("deepseek_v3")),
    (None, Some("minimax_m2")),
    (Some("gemma4"), Some("gemma4")),
    (Some("deepseek_v41"), Some("deepseek_v41")),
];

fn pair_name(pair: (Option<&str>, Option<&str>)) -> String {
    format!("{}+{}", pair.0.unwrap_or("-"), pair.1.unwrap_or("-"))
}

/// The reasoning opener and closer the stream uses for a parser pair.
fn reasoning_pair(pair: (Option<&str>, Option<&str>)) -> (&'static str, &'static str) {
    match pair {
        (Some("harmony"), _) | (_, Some("gpt_oss")) => (
            "<|channel|>analysis<|message|>",
            "<|end|><|start|>assistant<|channel|>final<|message|>",
        ),
        (_, Some("gemma4")) => ("<|channel>thought\n", "<channel|>"),
        (_, Some("minimax_m3")) => ("<mm:think>", "</mm:think>"),
        (_, Some("mistral")) => ("[THINK]", "[/THINK]"),
        _ => ("<think>", "</think>"),
    }
}

/// A native call to `get_weather(city="Paris")` in the pair's tool format.
fn native_call(pair: (Option<&str>, Option<&str>)) -> String {
    match pair {
        (Some("kimi_k2"), _) => "<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{\"city\": \"Paris\"}<|tool_call_end|><|tool_calls_section_end|>".to_string(),
        (Some("minimax_m3"), _) => "]<]minimax[>[<tool_call>\n]<]minimax[>[<invoke name=\"get_weather\">]<]minimax[>[<city>Paris]<]minimax[>[</city>]<]minimax[>[</invoke>\n]<]minimax[>[</tool_call>".to_string(),
        (Some("deepseek_v41"), _) => "<｜DSML｜ calls><｜DSML｜ invoke name=\"get_weather\"><｜DSML｜ parameter name=\"city\" string=\"true\">Paris</｜DSML｜ parameter></｜DSML｜ invoke></｜DSML｜ calls>".to_string(),
        (Some("deepseek_v4"), _) | (None, Some("deepseek_v4")) => "<｜DSML｜tool_calls><｜DSML｜invoke name=\"get_weather\"><｜DSML｜parameter name=\"city\" string=\"true\">Paris</｜DSML｜parameter></｜DSML｜invoke></｜DSML｜tool_calls>".to_string(),
        (Some("glm45" | "glm47"), _) => "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>".to_string(),
        (Some("harmony"), _) => "<|start|>assistant to=functions.get_weather<|channel|>commentary json<|message|>{\"city\":\"Paris\"}<|call|>".to_string(),
        (Some("gemma4"), _) => "<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>}<tool_call|>".to_string(),
        _ => "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>".to_string(),
    }
}

fn scenarios(pair: (Option<&str>, Option<&str>)) -> Vec<(&'static str, String)> {
    let (open, close) = reasoning_pair(pair);
    let call = native_call(pair);
    let final_open = if pair.1 == Some("gpt_oss") {
        "<|channel|>final<|message|>"
    } else {
        ""
    };
    vec![
        ("text", format!("{final_open}Hello, world. The answer is 42.")),
        ("reason_text", format!("{open}Think it over.{close}Answer here.")),
        ("reason_only", format!("{open}Only thinking here.{close}")),
        ("reason_unclosed", format!("{open}never closes")),
        ("prefilled_reason", format!("inner thought{close}Answer.")),
        ("bare_reason", "inner thought with no markers".to_string()),
        ("reason_ws", format!("{open}Think.{close}  \n")),
        ("whitespace", "  \n ".to_string()),
        ("reason_call", format!("{open}Need a tool.{close}{call}")),
        ("call_text", format!("{call}{final_open}Done.")),
        ("bare_json", "{\"city\": \"Paris\"}".to_string()),
        ("reason_blank", format!("{open} \n{close} ")),
        ("empty", String::new()),
    ]
}

fn split(text: &str, size: usize) -> Vec<String> {
    if size == 0 || text.is_empty() {
        return vec![text.to_string()];
    }
    let chars: Vec<char> = text.chars().collect();
    chars.chunks(size).map(|c| c.iter().collect()).collect()
}

#[allow(deprecated)]
fn choice(index: u32, text: Option<&str>, role: bool) -> ChatChoiceStream {
    ChatChoiceStream {
        index,
        delta: ChatCompletionStreamResponseDelta {
            role: role.then_some(Role::Assistant),
            content: text.map(|t| ChatCompletionMessageContent::Text(t.to_string())),
            tool_calls: None,
            function_call: None,
            refusal: None,
            reasoning_content: None,
        },
        finish_reason: None,
        logprobs: None,
    }
}

fn data_item(choices: Vec<ChatChoiceStream>) -> Item {
    Annotated::from_data(NvCreateChatCompletionStreamResponse {
        inner: CreateChatCompletionStreamResponse {
            id: "chatcmpl-parity".to_string(),
            choices,
            created: 1,
            model: "parity-model".to_string(),
            system_fingerprint: None,
            service_tier: None,
            object: "chat.completion.chunk".to_string(),
            usage: None,
        },
        nvext: None,
        llm_metrics: None,
        prompt_logprobs: None,
        tool_call_completion: Vec::new(),
    })
}

fn usage_trailer(chunk_tokens: Option<usize>, reasoning_tokens: Option<u32>) -> Item {
    let mut trailer = data_item(Vec::new());
    let data = trailer.data.as_mut().unwrap();
    data.inner.usage = Some(CompletionUsage {
        prompt_tokens: 7,
        completion_tokens: 9,
        total_tokens: 16,
        prompt_tokens_details: None,
        completion_tokens_details: reasoning_tokens.map(|reasoning_tokens| {
            dynamo_protocols::types::CompletionTokensDetails {
                reasoning_tokens: Some(reasoning_tokens),
                ..Default::default()
            }
        }),
    });
    if let Some(chunk_tokens) = chunk_tokens {
        data.llm_metrics = Some(LLMMetricAnnotation {
            chunk_tokens,
            ..Default::default()
        });
    }
    trailer
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Envelope {
    Plain,
    NoFinish,
    UsageTrailer,
    Metadata,
    TwoChoices,
    ErrorMid,
    ErrorFieldOnly,
    ErrorEventOnly,
    BackendReasoningTokens,
    FinishToolCalls,
    PartsContent,
    ParsedCall,
    ParsedMixed,
    ParsedPartialCall,
}

const ENVELOPES: &[(Envelope, &str)] = &[
    (Envelope::NoFinish, "no_finish"),
    (Envelope::UsageTrailer, "usage_trailer"),
    (Envelope::Metadata, "metadata"),
    (Envelope::TwoChoices, "two_choices"),
    (Envelope::ErrorMid, "error_mid"),
    (Envelope::ErrorFieldOnly, "error_field_only"),
    (Envelope::ErrorEventOnly, "error_event_only"),
    (Envelope::BackendReasoningTokens, "backend_reasoning_tokens"),
    (Envelope::FinishToolCalls, "finish_tool_calls"),
    (Envelope::PartsContent, "parts_content"),
    (Envelope::ParsedCall, "parsed_call"),
    (Envelope::ParsedMixed, "parsed_mixed"),
    (Envelope::ParsedPartialCall, "parsed_partial_call"),
];

fn build_input(text: &str, size: usize, envelope: Envelope, second: &str) -> Vec<Item> {
    let pieces = split(text, size);
    let second_pieces = split(second, size);
    let finish = match envelope {
        Envelope::NoFinish => None,
        Envelope::FinishToolCalls => Some(FinishReason::ToolCalls),
        _ => Some(FinishReason::Stop),
    };
    let rounds = if envelope == Envelope::TwoChoices {
        pieces.len().max(second_pieces.len())
    } else {
        pieces.len()
    };
    let mut items = Vec::new();
    for i in 0..rounds {
        let mut choices = Vec::new();
        if let Some(piece) = pieces.get(i) {
            let mut c = choice(0, Some(piece), i == 0);
            if i + 1 == pieces.len() {
                c.finish_reason = finish;
            }
            choices.push(c);
        }
        if envelope == Envelope::TwoChoices
            && let Some(piece) = second_pieces.get(i)
        {
            let mut c = choice(1, Some(piece), i == 0);
            if i + 1 == second_pieces.len() {
                c.finish_reason = finish;
            }
            choices.push(c);
        }
        let mut item = data_item(choices);
        let data = item.data.as_mut().unwrap();
        // Every chunk reports its token count, as the backend does.
        data.llm_metrics = Some(LLMMetricAnnotation {
            chunk_tokens: 1 + i % 3,
            ..Default::default()
        });
        if envelope == Envelope::Metadata {
            data.nvext = Some(json!({"seq": i}));
            if i == 0 {
                data.prompt_logprobs = Some(Arc::new(vec![None, None]));
            }
            if i % 2 == 0 {
                item.id = Some(format!("evt-{i}"));
                item.comment = Some(vec![format!("c-{i}")]);
            }
        }
        items.push(item);
        if i == rounds / 2 {
            match envelope {
                Envelope::ErrorMid => items.push(Annotated {
                    data: None,
                    id: Some("err-1".to_string()),
                    event: Some("error".to_string()),
                    comment: Some(vec!["boom".to_string()]),
                    error: Some(DynamoError::msg("boom")),
                }),
                Envelope::ErrorFieldOnly => items.push(Annotated {
                    data: None,
                    id: None,
                    event: None,
                    comment: None,
                    error: Some(DynamoError::msg("transport")),
                }),
                Envelope::ErrorEventOnly => items.push(Annotated {
                    data: None,
                    id: None,
                    event: Some("error".to_string()),
                    comment: Some(vec!["event only".to_string()]),
                    error: None,
                }),
                Envelope::PartsContent => {
                    let mut c = choice(0, None, false);
                    c.delta.content = Some(ChatCompletionMessageContent::Parts(vec![
                        serde_json::from_value(json!({"type": "text", "text": "part"}))
                            .unwrap(),
                    ]));
                    items.push(data_item(vec![c]));
                }
                Envelope::ParsedMixed => {
                    let mut c = choice(0, Some("visible"), false);
                    c.delta.reasoning_content = Some("parsed thought".to_string());
                    items.push(data_item(vec![c]));
                }
                Envelope::ParsedCall | Envelope::ParsedPartialCall => {
                    let arguments = if envelope == Envelope::ParsedCall {
                        "{\"city\":\"Oslo\"}"
                    } else {
                        "{\"city\":"
                    };
                    let mut c = choice(0, None, false);
                    c.delta.tool_calls = Some(vec![ChatCompletionMessageToolCallChunk {
                        index: 0,
                        id: Some("call_parsed".to_string()),
                        r#type: Some(FunctionType::Function),
                        function: Some(FunctionCallStream {
                            name: Some("get_weather".to_string()),
                            arguments: Some(arguments.to_string()),
                        }),
                    }]);
                    items.push(data_item(vec![c]));
                }
                _ => {}
            }
        }
    }
    match envelope {
        Envelope::UsageTrailer => items.push(usage_trailer(Some(4), None)),
        Envelope::Metadata => {
            let mut trailer = usage_trailer(Some(5), None);
            trailer.data.as_mut().unwrap().nvext = Some(json!({"trailer": true}));
            items.push(trailer);
        }
        Envelope::BackendReasoningTokens => items.push(usage_trailer(None, Some(3))),
        Envelope::ErrorMid | Envelope::ErrorFieldOnly | Envelope::ErrorEventOnly => {
            items.push(usage_trailer(Some(2), None))
        }
        _ => {}
    }
    items
}

fn strip_nulls(value: Value) -> Value {
    match value {
        Value::Object(map) => Value::Object(
            map.into_iter()
                .filter(|(_, v)| !v.is_null())
                .map(|(k, v)| (k, strip_nulls(v)))
                .collect(),
        ),
        Value::Array(items) => Value::Array(items.into_iter().map(strip_nulls).collect()),
        other => other,
    }
}

fn compact(item: &Item) -> Value {
    let mut out = Map::new();
    if let Some(id) = &item.id {
        out.insert("id".into(), json!(id));
    }
    if let Some(event) = &item.event {
        out.insert("event".into(), json!(event));
    }
    if let Some(comment) = &item.comment {
        out.insert("comment".into(), json!(comment));
    }
    if let Some(error) = &item.error {
        out.insert("error".into(), json!(error.message()));
    }
    if let Some(data) = &item.data {
        let mut inner = strip_nulls(serde_json::to_value(&data.inner).unwrap());
        let map = inner.as_object_mut().unwrap();
        for (key, default) in [
            ("id", json!("chatcmpl-parity")),
            ("object", json!("chat.completion.chunk")),
            ("created", json!(1)),
            ("model", json!("parity-model")),
        ] {
            if map.get(key) == Some(&default) {
                map.remove(key);
            }
        }
        if let Some(Value::Array(choices)) = map.get_mut("choices") {
            for choice in choices {
                if let Some(Value::Array(calls)) = choice.pointer_mut("/delta/tool_calls") {
                    for call in calls {
                        if call
                            .get("id")
                            .and_then(Value::as_str)
                            .is_some_and(|id| id != "call_parsed")
                        {
                            call["id"] = json!("call-*");
                        }
                    }
                }
            }
        }
        if let Some(nvext) = &data.nvext {
            map.insert("nvext".into(), nvext.clone());
        }
        if let Some(metrics) = &data.llm_metrics {
            map.insert("llm_metrics".into(), json!(metrics.chunk_tokens));
        }
        if let Some(logprobs) = &data.prompt_logprobs {
            map.insert("prompt_logprobs".into(), json!(logprobs.len()));
        }
        if !data.tool_call_completion.is_empty() {
            map.insert(
                "tool_call_completion".into(),
                serde_json::to_value(&data.tool_call_completion).unwrap(),
            );
        }
        out.insert("data".into(), inner);
    }
    Value::Object(out)
}

fn tools_json() -> Value {
    json!([{
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"]
            }
        }
    }])
}

/// One request shape: its name and the request JSON fields beside `messages` and `model`.
fn request_shapes() -> Vec<(&'static str, Value)> {
    vec![
        ("plain", json!({})),
        ("tools", json!({"tools": tools_json()})),
        ("tools_none", json!({"tools": tools_json(), "tool_choice": "none"})),
        (
            "tools_required",
            json!({"tools": tools_json(), "tool_choice": "required"}),
        ),
        (
            "tools_named",
            json!({"tools": tools_json(), "tool_choice": {"type": "function", "function": {"name": "get_weather"}}}),
        ),
        (
            "force_nonempty",
            json!({"chat_template_args": {"force_nonempty_content": true}}),
        ),
        (
            "force_nonempty_tools",
            json!({"tools": tools_json(), "chat_template_args": {"force_nonempty_content": true}}),
        ),
        (
            "thinking_off",
            json!({"chat_template_args": {"enable_thinking": false}}),
        ),
        (
            "thinking_off_force_nonempty",
            json!({"chat_template_args": {"enable_thinking": false, "force_nonempty_content": true}}),
        ),
        (
            "thinking_on",
            json!({"chat_template_args": {"thinking": true}}),
        ),
        (
            "thinking_mode_disabled",
            json!({"chat_template_args": {"thinking_mode": "disabled"}}),
        ),
        (
            "thinking_mode_chat",
            json!({"chat_template_args": {"thinking_mode": "chat"}}),
        ),
        (
            "reasoning_effort_none",
            json!({"chat_template_args": {"reasoning_effort": "none"}}),
        ),
        (
            "reasoning_effort_high",
            json!({"chat_template_args": {"reasoning_effort": "high"}}),
        ),
        ("top_level_thinking", json!({"thinking": {"type": "enabled"}})),
        (
            "structured",
            json!({"response_format": {"type": "json_object"}}),
        ),
        ("guided_regex", json!({"guided_regex": "Paris|London"})),
    ]
}

fn request(shape: &Value) -> NvCreateChatCompletionRequest {
    let mut body = json!({
        "model": "parity-model",
        "messages": [{"role": "user", "content": "Weather in Paris?"}],
        "stream": true,
    });
    for (key, value) in shape.as_object().unwrap() {
        body[key] = value.clone();
    }
    serde_json::from_value(body).unwrap()
}

fn preprocessor(pair: (Option<&str>, Option<&str>)) -> Arc<OpenAIPreprocessor> {
    let mut mdc = ModelDeploymentCard::load_from_disk(
        "tests/data/sample-models/mock-llama-3.1-8b-instruct",
        None,
    )
    .unwrap();
    mdc.runtime_config.tool_call_parser = pair.0.map(str::to_string);
    mdc.runtime_config.reasoning_parser = pair.1.map(str::to_string);
    OpenAIPreprocessor::new(mdc).unwrap()
}

fn route_json(route: &ToolProcessingRoute) -> Value {
    match route {
        ToolProcessingRoute::Unified(family) => json!({"unified": family}),
        ToolProcessingRoute::LegacyJail(parser) => json!({"legacy_jail": parser}),
        ToolProcessingRoute::PassThrough => json!("pass_through"),
    }
}

/// The request facts the moved code takes as inputs.
fn request_json(shape_name: &str, request: &NvCreateChatCompletionRequest) -> Value {
    json!({
        "shape": shape_name,
        "inner": serde_json::to_value(&request.inner).unwrap(),
        "chat_template_args": request.chat_template_args,
        "client_thinking_field": request.thinking.is_some(),
        "explicit_guided_decoding": crate::preprocessor::tool_choice::has_explicit_guided_decoding(request),
    })
}

/// Predicates that the HTTP layer and the aggregator also call.
fn predicate_records(version: &str) -> Vec<Value> {
    let mut records = Vec::new();
    for &pair in PAIRS {
        for (shape_name, shape) in request_shapes() {
            let request = request(&shape);
            let args = request.chat_template_args.as_ref();
            let unified_family =
                crate::protocols::openai::chat_completions::unified_parser::selected_batch_family(
                    pair.0, pair.1,
                );
            let disables: Vec<Value> = [None, Some("enabled"), Some("disabled"), Some("bogus")]
                .into_iter()
                .map(|default_mode| {
                    json!({
                        "default_thinking_mode": default_mode,
                        "with_family": OpenAIPreprocessor::request_disables_reasoning_with_family(
                            &request, pair.0, pair.1, default_mode, unified_family,
                        ),
                        "without_family": OpenAIPreprocessor::request_disables_reasoning_with_family(
                            &request, pair.0, pair.1, default_mode, None,
                        ),
                    })
                })
                .collect();
            records.push(json!({
                "kind": "predicates",
                "name": format!("{}/{shape_name}", pair_name(pair)),
                "version": version,
                "tool_call_parser": pair.0,
                "reasoning_parser": pair.1,
                "request": request_json(shape_name, &request),
                "unified_family": unified_family,
                "request_disables_reasoning": disables,
                "is_reasoning_disabled_by_request":
                    OpenAIPreprocessor::is_reasoning_disabled_by_request(pair.1, args),
                "wants_reasoning_as_content_when_empty":
                    OpenAIPreprocessor::wants_reasoning_as_content_when_empty(args),
                "stream_can_defer_all_output":
                    OpenAIPreprocessor::stream_can_defer_all_output(pair.0, pair.1, args),
            }));
        }
    }
    records
}

struct Case {
    name: String,
    pair: (Option<&'static str>, Option<&'static str>),
    shape: (&'static str, Value),
    prefill: bool,
    envelope: Envelope,
    input: Vec<Item>,
}

fn cases() -> Vec<Case> {
    let mut cases = Vec::new();
    for &pair in PAIRS {
        let shapes = request_shapes();
        for (scenario, text) in scenarios(pair) {
            for shape in &shapes {
                for prefill in [false, true] {
                    if prefill
                        && !scenario.starts_with("prefilled")
                        && !scenario.starts_with("bare_")
                    {
                        continue;
                    }
                    let sizes: &[usize] = if shape.0 == "plain" { &[0, 4] } else { &[4] };
                    for &size in sizes {
                        cases.push(Case {
                            name: format!(
                                "{}/{scenario}/{}/prefill={prefill}/size={size}",
                                pair_name(pair),
                                shape.0
                            ),
                            pair,
                            shape: shape.clone(),
                            prefill,
                            envelope: Envelope::Plain,
                            input: build_input(&text, size, Envelope::Plain, ""),
                        });
                    }
                }
            }
        }
        // Stream shapes, on the request shapes that turn on each policy.
        let (open, close) = reasoning_pair(pair);
        let reason_text = format!("{open}Think it over.{close}Answer here.");
        let reason_only = format!("{open}Only thinking.{close}");
        for &(envelope, envelope_name) in ENVELOPES {
            for shape in shapes.iter().filter(|(name, _)| {
                matches!(
                    *name,
                    "plain" | "tools" | "tools_none" | "force_nonempty" | "thinking_off"
                )
            }) {
                for (text_name, text) in [("reason_text", &reason_text), ("reason_only", &reason_only)]
                {
                    let second = if envelope == Envelope::TwoChoices {
                        reason_only.as_str()
                    } else {
                        ""
                    };
                    cases.push(Case {
                        name: format!(
                            "{}/{envelope_name}/{text_name}/{}",
                            pair_name(pair),
                            shape.0
                        ),
                        pair,
                        shape: shape.clone(),
                        prefill: false,
                        envelope,
                        input: build_input(text, 3, envelope, second),
                    });
                }
            }
        }
    }
    cases
}

#[tokio::test]
#[ignore = "scratch recorder"]
async fn stream_policies_recorder() {
    let Ok(path) = std::env::var("PARITY_OUT") else {
        return;
    };
    let version = std::env::var(env_llm::DYN_PARSER_VERSION).unwrap_or_else(|_| "auto".into());
    let mut file = std::fs::File::create(path).unwrap();
    let mut count = 0usize;
    for record in predicate_records(&version) {
        writeln!(file, "{}", serde_json::to_string(&record).unwrap()).unwrap();
        count += 1;
    }
    let mut preprocessors: HashMap<String, Arc<OpenAIPreprocessor>> = HashMap::new();
    for case in cases() {
        let pre = preprocessors
            .entry(pair_name(case.pair))
            .or_insert_with(|| preprocessor(case.pair))
            .clone();
        let request = request(&case.shape.1);
        let mut record = json!({
            "kind": "stream",
            "name": case.name,
            "version": version,
            "tool_call_parser": case.pair.0,
            "reasoning_parser": case.pair.1,
            "request": request_json(case.shape.0, &request),
            "prefill": case.prefill,
            "input": case.input.iter().map(compact).collect::<Vec<_>>(),
        });
        let constraint = match crate::preprocessor::tool_choice::guided_tool_constraint(
            &request,
            pre.tool_call_parser.as_deref(),
            pre.runtime_config.reasoning_parser.as_deref(),
            false,
        ) {
            Ok(constraint) => constraint,
            Err(error) => {
                record["constraint_error"] = json!(error.to_string());
                writeln!(file, "{}", serde_json::to_string(&record).unwrap()).unwrap();
                count += 1;
                continue;
            }
        };
        record["constraint"] = serde_json::to_value(&constraint).unwrap();
        let route = match pre.tool_processing_route(&request, &constraint) {
            Ok(route) => route,
            Err(error) => {
                record["route_error"] = json!(error.to_string());
                writeln!(file, "{}", serde_json::to_string(&record).unwrap()).unwrap();
                count += 1;
                continue;
            }
        };
        // dynamo's jail asserts that an item with an error is a terminal
        // error event, so this stream shape cannot reach the jail.
        if case.envelope == Envelope::ErrorFieldOnly && route.uses_legacy_jail() {
            continue;
        }
        record["route"] = route_json(&route);
        record["tool_definitions"] = json!(
            crate::preprocessor::tool_choice::effective_tool_definitions(&request.inner)
                .unwrap()
                .iter()
                .map(|tool| json!({
                    "name": tool.name,
                    "parameters": tool.parameters,
                    "strict": tool.strict,
                }))
                .collect::<Vec<_>>()
        );
        *JAIL_TAP.lock().unwrap() = Some(JailTap::default());
        let output = pre.postprocessor_parsing_stream_with_constraint(
            stream::iter(case.input),
            &request,
            PromptReasoningPrefill::from(case.prefill),
            constraint,
            route,
        );
        match output {
            Ok(output) => {
                let output: Vec<Item> = output.collect().await;
                let with_roles: Vec<Item> =
                    OpenAIPreprocessor::normalize_chat_stream_roles(stream::iter(output.clone()))
                        .collect()
                        .await;
                record["output"] = json!(output.iter().map(compact).collect::<Vec<_>>());
                record["output_with_roles"] =
                    json!(with_roles.iter().map(compact).collect::<Vec<_>>());
            }
            Err(error) => record["stream_error"] = json!(error.to_string()),
        }
        let tap = JAIL_TAP.lock().unwrap().take().unwrap();
        if tap.called {
            record["jail"] = json!({
                "request": tap.request,
                "input": tap.input,
                "input_ended": tap.input_ended,
                "output": tap.output,
            });
        }
        writeln!(file, "{}", serde_json::to_string(&record).unwrap()).unwrap();
        count += 1;
    }
    eprintln!("recorded {count} records, version={version}");
}
