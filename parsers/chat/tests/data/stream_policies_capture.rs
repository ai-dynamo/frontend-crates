// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The capture that wrote `dynamo_llm_tests_capture.jsonl.gz`. This crate does not build it.
//!
//! It ran inside dynamo-llm at ai-dynamo/dynamo `45e30e07`, as
//! `lib/llm/src/preprocessor/stream_policies_capture.rs`, added by
//! `stream_policies_recorder.patch`. With `STREAM_POLICIES_CAPTURE=<file>`
//! set, each `postprocessor_parsing_stream_with_constraint` call whose output
//! stream ends appends one JSON line: the parsers, the request facts, the
//! plan, the input, the jail's input and output, and the output. Without the
//! variable it changes nothing.
//!
//! ```text
//! git apply stream_policies_recorder.patch
//! export STREAM_POLICIES_CAPTURE=capture.jsonl
//! cargo test -p dynamo-llm --lib
//! cargo test -p dynamo-llm --test postprocessor_parsing_stream
//! cargo test -p dynamo-llm --test test_reasoning_parser
//! ```
//!
//! The fixture keeps each distinct call once, named after the test that made
//! it, with `#n` for its order among that test's calls.

use super::*;
use dynamo_runtime::protocols::annotated::Annotated;
use serde_json::{Map, Value, json};
use std::io::Write;
use std::sync::Mutex;

type Item = Annotated<NvCreateChatCompletionStreamResponse>;
pub(super) type Capture = Option<Arc<Mutex<Map<String, Value>>>>;

static FILE: Mutex<Option<std::fs::File>> = Mutex::new(None);

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

pub(super) fn compact(item: &Item) -> Value {
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

pub(super) fn start(
    pre: &OpenAIPreprocessor,
    request: &NvCreateChatCompletionRequest,
    prefill: &PromptReasoningPrefill,
    constraint: &crate::protocols::openai::GuidedToolConstraint,
    route: &ToolProcessingRoute,
) -> Capture {
    let path = std::env::var("STREAM_POLICIES_CAPTURE").ok()?;
    {
        let mut file = FILE.lock().unwrap();
        if file.is_none() {
            *file = Some(
                std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(path)
                    .unwrap(),
            );
        }
    }
    let tools = crate::preprocessor::tool_choice::effective_tool_definitions(&request.inner).ok()?;
    let version = match crate::protocols::openai::chat_completions::tool_parser_v2::selected_version()
        .ok()?
    {
        crate::protocols::openai::chat_completions::tool_parser_v2::ParserVersion::Auto => "auto",
        crate::protocols::openai::chat_completions::tool_parser_v2::ParserVersion::V1 => "1",
        crate::protocols::openai::chat_completions::tool_parser_v2::ParserVersion::V2 => "2",
    };
    let route = match route {
        ToolProcessingRoute::Unified(family) => json!({"unified": family}),
        ToolProcessingRoute::LegacyJail(parser) => json!({"legacy_jail": parser}),
        ToolProcessingRoute::PassThrough => json!("pass_through"),
    };
    let mut record = Map::new();
    record.insert("kind".into(), json!("stream"));
    record.insert(
        "name".into(),
        json!(std::thread::current().name().unwrap_or("?").to_string()),
    );
    record.insert("version".into(), json!(version));
    record.insert("tool_call_parser".into(), json!(pre.tool_call_parser));
    record.insert(
        "reasoning_parser".into(),
        json!(pre.runtime_config.reasoning_parser),
    );
    record.insert(
        "request".into(),
        json!({
            "inner": serde_json::to_value(&request.inner).unwrap(),
            "chat_template_args": request.chat_template_args,
            "client_thinking_field": request.thinking.is_some(),
            "explicit_guided_decoding":
                crate::preprocessor::tool_choice::has_explicit_guided_decoding(request),
        }),
    );
    record.insert(
        "prefill".into(),
        json!({"legacy": prefill.legacy, "unified": prefill.unified, "prefix": prefill.prefix}),
    );
    record.insert("constraint".into(), serde_json::to_value(constraint).unwrap());
    record.insert("route".into(), route);
    record.insert(
        "tool_definitions".into(),
        json!(
            tools
                .iter()
                .map(|tool| json!({
                    "name": tool.name,
                    "parameters": tool.parameters,
                    "strict": tool.strict,
                }))
                .collect::<Vec<_>>()
        ),
    );
    record.insert(
        "guided_tool_streaming_disabled".into(),
        json!(env_is_falsey(env_llm::DYN_ENABLE_GUIDED_TOOL_STREAMING)),
    );
    record.insert("input".into(), json!([]));
    record.insert("output".into(), json!([]));
    Some(Arc::new(Mutex::new(record)))
}

/// The arguments S10 passes to the v1 jail.
pub(super) fn jail_request_json(
    tool_call_parser: Option<&str>,
    tool_choice: Option<&dynamo_protocols::types::ChatCompletionToolChoiceOption>,
    tool_definitions: Option<&[dynamo_parsers::tool_calling::ToolDefinition]>,
    uses_tool_call_structural_tag: bool,
    guided_streaming: bool,
) -> Value {
    json!({
        "tool_call_parser": tool_call_parser,
        "tool_choice": tool_choice,
        "tool_definitions": tool_definitions.map(|tools| {
            tools
                .iter()
                .map(|tool| json!({
                    "name": tool.name,
                    "parameters": tool.parameters,
                    "strict": tool.strict,
                }))
                .collect::<Vec<_>>()
        }),
        "uses_tool_call_structural_tag": uses_tool_call_structural_tag,
        "guided_streaming": guided_streaming,
    })
}

pub(super) fn set(capture: &Capture, key: &str, value: Value) {
    if let Some(capture) = capture {
        capture.lock().unwrap().insert(key.to_string(), value);
    }
}

fn push(capture: &Capture, key: &str, item: &Item) {
    if let Some(capture) = capture {
        let mut record = capture.lock().unwrap();
        let list = record.entry(key.to_string()).or_insert_with(|| json!([]));
        list.as_array_mut().unwrap().push(compact(item));
    }
}

pub(super) fn tap<S>(capture: &Capture, key: &'static str, stream: S) -> Pin<Box<dyn Stream<Item = Item> + Send>>
where
    S: Stream<Item = Item> + Send + 'static,
{
    let Some(record) = capture else {
        return Box::pin(stream);
    };
    record
        .lock()
        .unwrap()
        .entry(key.to_string())
        .or_insert_with(|| json!([]));
    let tapped = capture.clone();
    let ended = capture.clone();
    let end = futures::stream::once(async move {
        if let Some(capture) = ended {
            capture
                .lock()
                .unwrap()
                .insert(format!("{key}_ended"), json!(true));
        }
    })
    .filter_map(|_| async { None });
    Box::pin(stream.inspect(move |item| push(&tapped, key, item)).chain(end))
}

/// Tap the output and write the record when the output ends.
pub(super) fn finish<S>(capture: Capture, stream: S) -> Pin<Box<dyn Stream<Item = Item> + Send>>
where
    S: Stream<Item = Item> + Send + 'static,
{
    let Some(_) = &capture else {
        return Box::pin(stream);
    };
    let tapped = capture.clone();
    let end = futures::stream::once(async move {
        if let Some(capture) = capture {
            let mut record = capture.lock().unwrap().clone();
            if let Some(input) = record.remove("jail_input") {
                let ended = record.remove("jail_input_ended").unwrap_or(json!(false));
                let output = record.remove("jail_output").unwrap_or(json!([]));
                let request = record.remove("jail_request").unwrap_or(Value::Null);
                record.insert(
                    "jail".into(),
                    json!({"request": request, "input": input, "input_ended": ended, "output": output}),
                );
            }
            record.remove("input_ended");
            record.remove("jail_output_ended");
            let line = serde_json::to_string(&Value::Object(record)).unwrap();
            if let Some(file) = FILE.lock().unwrap().as_mut() {
                writeln!(file, "{line}").unwrap();
            }
        }
    })
    .filter_map(|_| async { None });
    Box::pin(stream.inspect(move |item| push(&tapped, "output", item)).chain(end))
}
