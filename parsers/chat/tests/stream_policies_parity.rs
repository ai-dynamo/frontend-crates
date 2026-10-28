// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Parity with dynamo-llm's stream policies and its parse-stage composition.
//!
//! `tests/data/stream_policies_parity.jsonl.gz` holds inputs and the output
//! that dynamo-llm's own code gave for them, before the code moved here. It
//! was recorded at ai-dynamo/dynamo `45e30e07a4f4cef37ab7bfa739f7bee0c89b3057`
//! (dynamo-parsers 10.0.7, dynamo-parsers-v2 0.7.35) by
//! `tests/data/stream_policies_recorder.rs`. Each line is one record:
//!
//! - `predicates`: one parser pair and one request shape, with the answers of
//!   `request_disables_reasoning_with_family` (for each default thinking mode,
//!   with and without the unified family), `is_reasoning_disabled_by_request`,
//!   `wants_reasoning_as_content_when_empty`, and `stream_can_defer_all_output`.
//! - `stream`: one parser pair, one request shape, a prompt prefill, and an
//!   input stream, with the output of `postprocessor_parsing_stream_with_constraint`
//!   and of `normalize_chat_stream_roles` after it.
//!
//! The parser pairs are Rhino's 8 families plus the parsers whose rules these
//! policies name (Nemotron, Kimi K2.5, Mistral, DeepSeek R1 and V3, MiniMax
//! M2, Gemma 4, and no parser). The request shapes turn on each rule: tools,
//! `tool_choice: none`, `force_nonempty_content`, each thinking control, a
//! structured response, and a guided regex. The stream shapes add a usage
//! trailer, per-chunk metadata, two choices, terminal and transport errors, a
//! backend `reasoning_tokens` count, a `tool_calls` finish, content parts, and
//! an already parsed tool call.
//!
//! dynamo ran the recorder three times: with `DYN_PARSER_VERSION` unset, `1`,
//! and `2`. The file holds every record of the first run, and each record of
//! the other runs that differs from the first. Each record names its version.
//!
//! The v1 tool-call jail stays in dynamo, so the recorder also kept the
//! jail's arguments, what it received, and what it sent. This test passes a
//! jail that checks it gets the same arguments and input, and replays the same
//! output. So a `LegacyJail` record tests
//! every stage here around dynamo's jail.
//!
//! `tests/data/dynamo_llm_tests_capture.jsonl.gz` holds the same kind of
//! `stream` record, captured from dynamo-llm's own tests instead of a matrix:
//! every `postprocessor_parsing_stream_with_constraint` call that dynamo's
//! lib tests and its `postprocessor_parsing_stream` and `test_reasoning_parser`
//! suites made at `45e30e07`, with each call's prompt prefill as dynamo
//! computed it. A record's name is the test that made the call. The capture
//! code is `tests/data/stream_policies_capture.rs`.
//!
//! Chunks leave out the fixed envelope (`id`, `object`, `created`, `model`)
//! and null fields. dynamo's `llm_metrics` is written as its `chunk_tokens`
//! value, and `prompt_logprobs` as its length. Generated tool-call IDs are
//! written as `call-*`.

use std::collections::HashMap;
use std::io::Read;
use std::sync::{Arc, Mutex};

use dynamo_chat_parsing::chat_stream::{
    ChatStreamParsers, ChatStreamPlan, JailRequest, PolicyStreamItem, parse_chat_stream,
};
use dynamo_chat_parsing::families::selected_family;
use dynamo_chat_parsing::guided::GuidedToolConstraint;
use dynamo_chat_parsing::nonempty_content::{
    stream_can_defer_all_output, wants_reasoning_as_content_when_empty,
};
use dynamo_chat_parsing::prefill::{PromptReasoningPrefill, prompt_reasoning_prefill_for_parsers};
use dynamo_chat_parsing::reasoning_off::{
    ChatStreamRequest, is_reasoning_disabled_by_request, request_disables_reasoning_with_family,
};
use dynamo_chat_parsing::route::{ToolProcessingRoute, tool_processing_route};
use dynamo_chat_parsing::stream_item::{
    AnnotationFields, ToolCallCompletion, UnifiedStreamChunk, UnifiedStreamItem,
};
use dynamo_chat_parsing::tool_call_policy::normalize_chat_stream_roles;
use dynamo_chat_parsing::version::ParserVersion;
use dynamo_parsers::chat_stream::ChatStreamItem;
use dynamo_parsers::tool_calling::ToolDefinition;
use dynamo_protocols::types::{CreateChatCompletionRequest, CreateChatCompletionStreamResponse};
use futures::stream::BoxStream;
use futures::{StreamExt, stream};
use serde_json::{Map, Value, json};

/// The shape of dynamo-llm's `NvCreateChatCompletionStreamResponse`.
#[derive(Clone, Debug)]
struct ParityChunk {
    inner: CreateChatCompletionStreamResponse,
    nvext: Option<Value>,
    llm_metrics: Option<u64>,
    prompt_logprobs: Option<u64>,
    tool_call_completion: Vec<ToolCallCompletion>,
}

/// The shape of dynamo-runtime's `Annotated`.
#[derive(Clone, Debug)]
struct Item {
    data: Option<ParityChunk>,
    id: Option<String>,
    event: Option<String>,
    comment: Option<Vec<String>>,
    error: Option<String>,
}

impl UnifiedStreamChunk for ParityChunk {
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
    type Chunk = ParityChunk;
    type Annotation = AnnotationFields;

    fn is_error(&self) -> bool {
        self.event.as_deref() == Some("error")
    }

    fn into_parts(self) -> Result<(ParityChunk, AnnotationFields), Self> {
        let Item {
            data,
            id,
            event,
            comment,
            error,
        } = self;
        match data {
            Some(data) => Ok((
                data,
                AnnotationFields {
                    id,
                    event,
                    comment,
                    error,
                },
            )),
            None => Err(Item {
                data: None,
                id,
                event,
                comment,
                error,
            }),
        }
    }

    fn from_parts(data: ParityChunk, annotation: AnnotationFields) -> Self {
        Item {
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
            .and_then(|data| data.llm_metrics)
            .map_or(0, |tokens| tokens as usize)
    }
}

const ENVELOPE: [(&str, &str); 3] = [
    ("id", "chatcmpl-parity"),
    ("object", "chat.completion.chunk"),
    ("model", "parity-model"),
];

fn fixture() -> Vec<Value> {
    read_fixture(include_bytes!("data/stream_policies_parity.jsonl.gz"))
}

fn read_fixture(bytes: &[u8]) -> Vec<Value> {
    let mut text = String::new();
    flate2::read::GzDecoder::new(bytes)
        .read_to_string(&mut text)
        .expect("decompress fixture");
    text.lines()
        .map(|line| serde_json::from_str(line).expect("fixture line"))
        .collect()
}

fn to_item(value: &Value) -> Item {
    let text = |key: &str| value.get(key).and_then(Value::as_str).map(str::to_string);
    let data = value.get("data").map(|data| {
        let mut data = data.as_object().expect("data object").clone();
        let nvext = data.remove("nvext");
        let llm_metrics = data.remove("llm_metrics").and_then(|v| v.as_u64());
        let prompt_logprobs = data.remove("prompt_logprobs").and_then(|v| v.as_u64());
        let tool_call_completion = data
            .remove("tool_call_completion")
            .map(|v| serde_json::from_value(v).expect("completion evidence"))
            .unwrap_or_default();
        for (key, default) in ENVELOPE {
            data.entry(key).or_insert(json!(default));
        }
        data.entry("created").or_insert(json!(1));
        ParityChunk {
            inner: serde_json::from_value(Value::Object(data)).expect("chunk"),
            nvext,
            llm_metrics,
            prompt_logprobs,
            tool_call_completion,
        }
    });
    Item {
        data,
        id: text("id"),
        event: text("event"),
        comment: value
            .get("comment")
            .map(|v| serde_json::from_value(v.clone()).expect("comment")),
        error: text("error"),
    }
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

fn to_compact(item: &Item) -> Value {
    let mut out = Map::new();
    for (key, field) in [("id", &item.id), ("event", &item.event)] {
        if let Some(field) = field {
            out.insert(key.into(), json!(field));
        }
    }
    if let Some(comment) = &item.comment {
        out.insert("comment".into(), json!(comment));
    }
    if let Some(error) = &item.error {
        out.insert("error".into(), json!(error));
    }
    if let Some(data) = &item.data {
        let mut inner = strip_nulls(serde_json::to_value(&data.inner).expect("serialize chunk"));
        let map = inner.as_object_mut().expect("chunk object");
        for (key, default) in ENVELOPE {
            if map.get(key) == Some(&json!(default)) {
                map.remove(key);
            }
        }
        if map.get("created") == Some(&json!(1)) {
            map.remove("created");
        }
        if let Some(Value::Array(choices)) = map.get_mut("choices") {
            for choice in choices {
                let Some(Value::Array(calls)) = choice.pointer_mut("/delta/tool_calls") else {
                    continue;
                };
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
        if let Some(nvext) = &data.nvext {
            map.insert("nvext".into(), nvext.clone());
        }
        if let Some(metrics) = data.llm_metrics {
            map.insert("llm_metrics".into(), json!(metrics));
        }
        if let Some(logprobs) = data.prompt_logprobs {
            map.insert("prompt_logprobs".into(), json!(logprobs));
        }
        if !data.tool_call_completion.is_empty() {
            map.insert(
                "tool_call_completion".into(),
                serde_json::to_value(&data.tool_call_completion).expect("evidence"),
            );
        }
        out.insert("data".into(), inner);
    }
    Value::Object(out)
}

fn items(value: &Value) -> Vec<Item> {
    value
        .as_array()
        .expect("item list")
        .iter()
        .map(to_item)
        .collect()
}

fn version(record: &Value) -> ParserVersion {
    match record["version"].as_str() {
        Some("auto") => ParserVersion::Auto,
        Some("1") => ParserVersion::V1,
        Some("2") => ParserVersion::V2,
        other => panic!("unknown version {other:?}"),
    }
}

/// The request fields the record carries, owned.
struct RecordedRequest {
    inner: CreateChatCompletionRequest,
    chat_template_args: Option<HashMap<String, Value>>,
    client_thinking_field: bool,
    explicit_guided_decoding: bool,
}

impl RecordedRequest {
    fn new(value: &Value) -> Self {
        Self {
            inner: serde_json::from_value(value["inner"].clone()).expect("request"),
            chat_template_args: serde_json::from_value(value["chat_template_args"].clone())
                .expect("chat_template_args"),
            client_thinking_field: value["client_thinking_field"].as_bool().unwrap(),
            explicit_guided_decoding: value["explicit_guided_decoding"].as_bool().unwrap(),
        }
    }

    fn view(&self) -> ChatStreamRequest<'_> {
        ChatStreamRequest {
            request: &self.inner,
            chat_template_args: self.chat_template_args.as_ref(),
            client_thinking_field: self.client_thinking_field,
            explicit_guided_decoding: self.explicit_guided_decoding,
        }
    }
}

fn parsers(record: &Value) -> (Option<&str>, Option<&str>) {
    (
        record["tool_call_parser"].as_str(),
        record["reasoning_parser"].as_str(),
    )
}

/// Check one `predicates` record. Returns the mismatches.
fn check_predicates(record: &Value) -> Vec<String> {
    let (tool, reasoning) = parsers(record);
    let version = version(record);
    let request = RecordedRequest::new(&record["request"]);
    let view = request.view();
    let args = request.chat_template_args.as_ref();
    let unified_family = record["unified_family"].as_str();
    let mut mismatches = Vec::new();
    let mut check = |what: &str, got: Value, want: &Value| {
        if &got != want {
            mismatches.push(format!("{what}: got {got}, dynamo {want}"));
        }
    };
    check(
        "selected family",
        json!(selected_family(tool, reasoning, version)),
        &json!(unified_family),
    );
    for entry in record["request_disables_reasoning"].as_array().unwrap() {
        let default_mode = entry["default_thinking_mode"].as_str();
        check(
            &format!("request_disables_reasoning_with_family({default_mode:?}, family)"),
            json!(request_disables_reasoning_with_family(
                &view,
                tool,
                reasoning,
                default_mode,
                unified_family
            )),
            &entry["with_family"],
        );
        check(
            &format!("request_disables_reasoning_with_family({default_mode:?}, None)"),
            json!(request_disables_reasoning_with_family(
                &view,
                tool,
                reasoning,
                default_mode,
                None
            )),
            &entry["without_family"],
        );
    }
    check(
        "is_reasoning_disabled_by_request",
        json!(is_reasoning_disabled_by_request(reasoning, args)),
        &record["is_reasoning_disabled_by_request"],
    );
    check(
        "wants_reasoning_as_content_when_empty",
        json!(wants_reasoning_as_content_when_empty(args)),
        &record["wants_reasoning_as_content_when_empty"],
    );
    check(
        "stream_can_defer_all_output",
        json!(stream_can_defer_all_output(tool, reasoning, args, version)),
        &record["stream_can_defer_all_output"],
    );
    mismatches
}

fn route(value: &Value) -> ToolProcessingRoute {
    if let Some(family) = value.get("unified").and_then(Value::as_str) {
        let family = dynamo_chat_parsing::families::unified_family_names()
            .iter()
            .copied()
            .find(|name| *name == family)
            .unwrap_or_else(|| panic!("unknown family {family}"));
        return ToolProcessingRoute::Unified(family);
    }
    if let Some(parser) = value.get("legacy_jail") {
        return ToolProcessingRoute::LegacyJail(parser.as_str().map(str::to_string));
    }
    assert_eq!(value, &json!("pass_through"));
    ToolProcessingRoute::PassThrough
}

fn tool_definitions(value: &Value) -> Vec<ToolDefinition> {
    value
        .as_array()
        .expect("tool definitions")
        .iter()
        .map(|tool| ToolDefinition {
            name: tool["name"].as_str().unwrap().to_string(),
            parameters: tool.get("parameters").filter(|v| !v.is_null()).cloned(),
            strict: tool["strict"].as_bool(),
        })
        .collect()
}

/// [`JailRequest`] in the form the recorder wrote.
fn jail_request_json(request: &JailRequest) -> Value {
    json!({
        "tool_call_parser": request.tool_call_parser,
        "tool_choice": request.tool_choice,
        "tool_definitions": request.tool_definitions.as_ref().map(|tools| {
            tools
                .iter()
                .map(|tool| json!({
                    "name": tool.name,
                    "parameters": tool.parameters,
                    "strict": tool.strict,
                }))
                .collect::<Vec<_>>()
        }),
        "uses_tool_call_structural_tag": request.uses_tool_call_structural_tag,
        "guided_streaming": request.guided_streaming,
    })
}

/// A jail that checks it gets the arguments and the input that dynamo's jail
/// got, then sends what dynamo's jail sent.
fn recorded_jail(
    recorded: Option<Value>,
    calls: Arc<Mutex<Vec<String>>>,
) -> impl FnOnce(BoxStream<'static, Item>, JailRequest) -> BoxStream<'static, Item> {
    move |mut input, request| {
        let Some(recorded) = recorded else {
            calls
                .lock()
                .unwrap()
                .push(format!("jail called, dynamo did not call it: {request:?}"));
            return Box::pin(stream::empty());
        };
        let got = jail_request_json(&request);
        if got != recorded["request"] {
            calls.lock().unwrap().push(format!(
                "jail request: got {got}, dynamo {}",
                recorded["request"]
            ));
            return Box::pin(stream::empty());
        }
        Box::pin(async_stream::stream! {
            let expected = recorded["input"].as_array().unwrap().clone();
            for (index, want) in expected.iter().enumerate() {
                match input.next().await {
                    Some(item) if &to_compact(&item) == want => {}
                    Some(item) => {
                        calls.lock().unwrap().push(format!(
                            "jail input {index}: got {}, dynamo {want}",
                            to_compact(&item)
                        ));
                        return;
                    }
                    None => {
                        calls
                            .lock()
                            .unwrap()
                            .push(format!("jail input ended at {index}, dynamo had more"));
                        return;
                    }
                }
            }
            if recorded["input_ended"].as_bool() == Some(true)
                && let Some(item) = input.next().await
            {
                calls.lock().unwrap().push(format!(
                    "jail input has an extra item {}",
                    to_compact(&item)
                ));
                return;
            }
            for item in items(&recorded["output"]) {
                yield item;
            }
        })
    }
}

/// The prompt prefill a record names: a boolean, or the three facts dynamo
/// computed from the rendered prompt. The facts are rebuilt from a prompt
/// that ends the same way.
fn prefill(
    value: &Value,
    tool: Option<&str>,
    reasoning: Option<&str>,
) -> Result<PromptReasoningPrefill, String> {
    if let Some(prefill) = value.as_bool() {
        return Ok(PromptReasoningPrefill::from(prefill));
    }
    let want = (
        value["legacy"].as_bool().unwrap(),
        value["unified"].as_bool().unwrap(),
        value["prefix"].as_str(),
    );
    if want.0 == want.1 && want.2.is_none() {
        return Ok(PromptReasoningPrefill::from(want.0));
    }
    const HEADER: &str = "<|channel>thought\n";
    let mut endings = vec![
        "",
        "<think>",
        "<mm:think>",
        "<|open|>think<|sep|>",
        "assistant to=self<|message|>",
    ];
    endings.extend(("<|channel>".len()..HEADER.len()).map(|end| &HEADER[..end]));
    endings
        .into_iter()
        .map(|ending| prompt_reasoning_prefill_for_parsers(tool, reasoning, Some(ending)))
        .find(|got| (got.legacy(), got.unified(), got.prefix()) == want)
        .ok_or_else(|| format!("no prompt gives prefill {value}"))
}

/// Run one `stream` record. Returns the output and the output with roles, as
/// compact JSON, or the reason it could not run.
async fn run_stream(record: &Value) -> Result<(Value, Value), String> {
    let (tool, reasoning) = parsers(record);
    let version = version(record);
    let request = RecordedRequest::new(&record["request"]);
    let route = route(&record["route"]);
    // The route is the route crate's rule; check it gives dynamo's answer here too.
    let ours = tool_processing_route(tool, reasoning, &request.inner, version)
        .map_err(|error| format!("route: {error}"))?;
    if ours != route {
        return Err(format!("route: got {ours:?}, dynamo {route:?}"));
    }
    let constraint: GuidedToolConstraint =
        serde_json::from_value(record["constraint"].clone()).expect("constraint");
    let jail_errors = Arc::new(Mutex::new(Vec::new()));
    let output: Vec<Item> = parse_chat_stream(
        stream::iter(items(&record["input"])),
        ChatStreamParsers {
            tool_call_parser: tool,
            reasoning_parser: reasoning,
        },
        request.view(),
        ChatStreamPlan {
            route,
            guided_tool_constraint: constraint,
            prompt_reasoning: prefill(&record["prefill"], tool, reasoning)?,
            tool_definitions: tool_definitions(&record["tool_definitions"]),
            parser_version: version,
            guided_tool_streaming_disabled: record["guided_tool_streaming_disabled"]
                .as_bool()
                .unwrap_or(false),
        },
        recorded_jail(record.get("jail").cloned(), Arc::clone(&jail_errors)),
    )
    .collect()
    .await;
    let jail_errors = jail_errors.lock().unwrap().clone();
    if !jail_errors.is_empty() {
        return Err(jail_errors.join("; "));
    }
    let with_roles: Vec<Item> = normalize_chat_stream_roles(stream::iter(output.clone()))
        .collect()
        .await;
    Ok((
        json!(output.iter().map(to_compact).collect::<Vec<_>>()),
        json!(with_roles.iter().map(to_compact).collect::<Vec<_>>()),
    ))
}

#[test]
fn fixture_items_round_trip() {
    for record in fixture().iter().filter(|r| r["kind"] == "stream") {
        for key in ["input", "output"] {
            let Some(list) = record.get(key) else {
                continue;
            };
            for value in list.as_array().unwrap() {
                assert_eq!(&to_compact(&to_item(value)), value, "{}", record["name"]);
            }
        }
    }
}

#[tokio::test]
async fn moved_stream_policies_match_dynamo_llm() {
    let mut checked = HashMap::<&str, usize>::new();
    let mut skipped = 0usize;
    let mut failures = Vec::new();
    let fixture = fixture();
    for record in &fixture {
        let name = format!("{} [{}]", record["name"], record["version"]);
        match record["kind"].as_str() {
            Some("predicates") => {
                let mismatches = check_predicates(record);
                if !mismatches.is_empty() {
                    failures.push(format!("{name}: {}", mismatches.join("; ")));
                }
                *checked.entry("predicates").or_default() += 1;
            }
            Some("stream") => {
                // A request that dynamo refused before the stream: the route
                // and the guided constraint are other crates' rules.
                if record.get("route").is_none() || record.get("stream_error").is_some() {
                    skipped += 1;
                    continue;
                }
                match run_stream(record).await {
                    Ok((output, with_roles)) => {
                        if output != record["output"] {
                            failures.push(format!(
                                "{name}: output\n  got    {output}\n  dynamo {}",
                                record["output"]
                            ));
                        } else if record
                            .get("output_with_roles")
                            .is_some_and(|want| &with_roles != want)
                        {
                            failures.push(format!(
                                "{name}: roles\n  got    {with_roles}\n  dynamo {}",
                                record["output_with_roles"]
                            ));
                        }
                    }
                    Err(error) => failures.push(format!("{name}: {error}")),
                }
                let kind = if record.get("jail").is_some() {
                    "stream with jail"
                } else {
                    "stream"
                };
                *checked.entry(kind).or_default() += 1;
            }
            other => panic!("unknown record kind {other:?}"),
        }
    }
    eprintln!("checked {checked:?}, skipped {skipped} refused requests");
    assert!(checked["stream with jail"] > 0);
    assert!(
        failures.is_empty(),
        "{} of {} records differ from dynamo-llm:\n{}",
        failures.len(),
        fixture.len(),
        failures
            .iter()
            .take(20)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );
}

#[tokio::test]
async fn dynamo_llm_test_streams_match() {
    let fixture = read_fixture(include_bytes!("data/dynamo_llm_tests_capture.jsonl.gz"));
    let mut jail = 0usize;
    let mut failures = Vec::new();
    for record in &fixture {
        jail += usize::from(record.get("jail").is_some());
        match run_stream(record).await {
            Ok((output, _)) if output == record["output"] => {}
            Ok((output, _)) => failures.push(format!(
                "{}: output\n  got    {output}\n  dynamo {}",
                record["name"], record["output"]
            )),
            Err(error) => failures.push(format!("{}: {error}", record["name"])),
        }
    }
    eprintln!(
        "checked {} captured calls, {jail} through the jail",
        fixture.len()
    );
    assert!(jail > 0);
    assert!(
        failures.is_empty(),
        "{} of {} captured calls differ from dynamo-llm:\n{}",
        failures.len(),
        fixture.len(),
        failures
            .iter()
            .take(20)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );
}

/// The replayed jail must refuse arguments dynamo's jail did not get, or the
/// replay could not catch a wrong `JailRequest`.
#[tokio::test]
async fn recorded_jail_rejects_changed_guided_streaming() {
    let record = read_fixture(include_bytes!("data/dynamo_llm_tests_capture.jsonl.gz"))
        .into_iter()
        .find(|record| record["name"] == "minimax_m2_disabled_named_json_returns_tool_call#2")
        .expect("recorded guided-JSON jail call");
    let request = RecordedRequest::new(&record["request"]);
    let constraint: GuidedToolConstraint =
        serde_json::from_value(record["constraint"].clone()).unwrap();
    assert!(constraint.installs_guided_json());
    assert_eq!(record["guided_tool_streaming_disabled"], json!(false));

    // The recorded request is the control; only the release flag changes.
    for guided_streaming in [true, false] {
        let errors = Arc::new(Mutex::new(Vec::new()));
        let output: Vec<_> = recorded_jail(record.get("jail").cloned(), Arc::clone(&errors))(
            Box::pin(stream::iter(items(&record["jail"]["input"]))),
            JailRequest {
                tool_call_parser: record["route"]["legacy_jail"].as_str().map(str::to_string),
                tool_choice: request.inner.tool_choice.clone(),
                tool_definitions: Some(tool_definitions(&record["tool_definitions"])),
                uses_tool_call_structural_tag: constraint.uses_structural_tag(),
                guided_streaming,
            },
        )
        .collect()
        .await;

        if guided_streaming {
            assert!(errors.lock().unwrap().is_empty());
            assert_eq!(
                json!(output.iter().map(to_compact).collect::<Vec<_>>()),
                record["jail"]["output"]
            );
        } else {
            assert!(
                !errors.lock().unwrap().is_empty(),
                "replay accepted guided_streaming=false for a capture made with guided_streaming=true"
            );
        }
    }
}
