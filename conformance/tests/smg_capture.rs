// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Run every tool-calling, reasoning, and unified conformance input through the
//! published SMG Rust parsers.
//!
//! This is both a coverage test and the live capture producer used by
//! `conformance/utils/render_table_v2.sh`. The renderer sets the four input-root
//! variables and `SMG_CAPTURE_OUTPUT`; a normal `cargo test` run resolves the
//! committed fixture snapshot directly and only checks that every real input is
//! represented in the capture.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use openai_protocol::common::{Function, Tool};
use reasoning_parser::ParserFactory as ReasoningParserFactory;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tool_parser::ParserFactory as ToolParserFactory;

mod common;

const TOOL_PARSER_VERSION: &str = "1.6.0";
const REASONING_PARSER_VERSION: &str = "1.6.0";

#[derive(Debug, Deserialize)]
struct ToolFixture {
    family: String,
    #[serde(default)]
    mode: Option<String>,
    #[serde(default)]
    cases: BTreeMap<String, ToolCase>,
}

#[derive(Debug, Deserialize)]
struct ToolCase {
    #[serde(default)]
    model_text: Option<String>,
    #[serde(default)]
    tools: Vec<RawTool>,
    #[serde(default)]
    chunks: Vec<TextChunk>,
}

#[derive(Debug, Deserialize)]
struct RawTool {
    name: String,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    parameters: Option<Value>,
    #[serde(default)]
    strict: Option<bool>,
}

impl From<&RawTool> for Tool {
    fn from(raw: &RawTool) -> Self {
        Self {
            tool_type: "function".to_string(),
            function: Function {
                name: raw.name.clone(),
                description: raw.description.clone(),
                parameters: raw.parameters.clone().unwrap_or_else(|| json!({})),
                strict: raw.strict,
            },
        }
    }
}

#[derive(Debug, Deserialize)]
struct TextChunk {
    #[serde(default)]
    delta_text: String,
}

#[derive(Debug, Deserialize)]
struct ReasoningFixture {
    family: String,
    mode: String,
    #[serde(default)]
    cases: BTreeMap<String, ReasoningCase>,
}

#[derive(Debug, Deserialize)]
struct ReasoningCase {
    #[serde(default)]
    model_text: Option<String>,
    #[serde(default)]
    chunks: Option<Vec<String>>,
    #[serde(default)]
    force_reasoning: bool,
}

#[derive(Debug, Deserialize)]
struct UnifiedFixture {
    family: String,
    #[serde(default)]
    cases: BTreeMap<String, UnifiedCase>,
}

#[derive(Debug, Deserialize)]
struct UnifiedCase {
    #[serde(default)]
    scenario: Option<String>,
    #[serde(default)]
    chunks: Vec<TextChunk>,
    #[serde(default)]
    init: Value,
}

#[derive(Debug, Default, Serialize)]
struct ToolCaptures {
    batch: BTreeMap<String, Value>,
    stream_on_batch: BTreeMap<String, Value>,
    stream: BTreeMap<String, Value>,
}

#[derive(Debug, Default, Serialize)]
struct ReasoningCaptures {
    batch: BTreeMap<String, Value>,
    stream: BTreeMap<String, Value>,
}

#[derive(Debug, Default, Serialize)]
struct Coverage {
    tool_batch_inputs: usize,
    tool_stream_inputs: usize,
    reasoning_batch_inputs: usize,
    reasoning_stream_inputs: usize,
    unified_inputs: usize,
}

#[derive(Debug, Serialize)]
struct CaptureDoc {
    schema: &'static str,
    tool_parser_version: &'static str,
    reasoning_parser_version: &'static str,
    toolcalling: ToolCaptures,
    reasoning: ReasoningCaptures,
    unified: BTreeMap<String, Value>,
    coverage: Coverage,
}

fn tool_parser_name(family: &str) -> Option<&'static str> {
    match family {
        "deepseek_v3" => Some("deepseek"),
        "deepseek_v3_1" => Some("deepseek31"),
        "deepseek_v3_2" => Some("deepseek32"),
        "deepseek_v4" => Some("deepseek_v4"),
        "glm47" => Some("glm47_moe"),
        "hermes" | "qwen25" => Some("qwen"),
        "inkling" => Some("inkling"),
        "kimi_k2" => Some("kimik2"),
        "llama3_json" => Some("llama"),
        "minimax_m2" => Some("minimax_m2"),
        "mistral" => Some("mistral"),
        "nemotron_nano" | "qwen3_coder" => Some("qwen_xml"),
        "pythonic" => Some("pythonic"),
        _ => None,
    }
}

fn reasoning_parser_name(family: &str) -> Option<&'static str> {
    match family {
        "deepseek_r1" | "deepseek_v3" => Some("deepseek_r1"),
        "deepseek_v4" | "qwen3" => Some("qwen3"),
        "inkling" => Some("inkling"),
        "kimi" => Some("kimi"),
        "kimi_k25" => Some("kimi_k25"),
        "minimax_append_think" => Some("minimax"),
        "nemotron_deci" => Some("glm45"),
        _ => None,
    }
}

fn unified_parser_names(family: &str) -> Option<(&'static str, &'static str)> {
    match family {
        "qwen3" => Some(("qwen3", "qwen_xml")),
        "kimi_k2" => Some(("kimi_k25", "kimik2")),
        _ => None,
    }
}

fn case_key(family: &str, case_id: &str) -> String {
    format!("{family}:{case_id}")
}

fn unavailable(crate_name: &str, version: &str, family: &str) -> Value {
    json!({
        "unavailable": format!(
            "SMG {crate_name} {version} has no parser for conformance family '{family}'."
        )
    })
}

fn parser_error(parser: &str, error: impl std::fmt::Display) -> Value {
    json!({
        "parser": parser,
        "error": {"kind": "parser_error", "message": error.to_string()},
    })
}

fn collect_yaml(root: &Path) -> Vec<PathBuf> {
    let mut paths = Vec::new();
    common::collect_yaml(root, &mut paths);
    paths.sort();
    paths
}

fn decode_arguments(raw: &str) -> Value {
    serde_json::from_str(raw).unwrap_or_else(|_| Value::String(raw.to_string()))
}

#[derive(Debug, Default)]
struct ToolAssembly {
    names: BTreeMap<usize, String>,
    arguments: BTreeMap<usize, String>,
    normal_text: String,
}

impl ToolAssembly {
    fn push(&mut self, result: &tool_parser::StreamingParseResult) -> Vec<Value> {
        self.normal_text.push_str(&result.normal_text);
        result
            .calls
            .iter()
            .map(|call| {
                if let Some(name) = &call.name {
                    self.names
                        .entry(call.tool_index)
                        .or_default()
                        .push_str(name);
                }
                self.arguments
                    .entry(call.tool_index)
                    .or_default()
                    .push_str(&call.parameters);
                json!({
                    "index": call.tool_index,
                    "name": call.name,
                    "arguments": call.parameters,
                })
            })
            .collect()
    }

    fn calls(&self) -> Vec<Value> {
        self.names
            .iter()
            .map(|(index, name)| {
                let raw = self.arguments.get(index).map(String::as_str).unwrap_or("");
                json!({"name": name, "arguments": decode_arguments(raw)})
            })
            .collect()
    }
}

async fn capture_tool_batch(family: &str, text: &str, tools: &[Tool]) -> Value {
    let Some(parser_name) = tool_parser_name(family) else {
        return unavailable("tool-parser", TOOL_PARSER_VERSION, family);
    };
    let factory = ToolParserFactory::new();
    let Some(parser) = factory.registry().create_parser(parser_name) else {
        return unavailable("tool-parser", TOOL_PARSER_VERSION, family);
    };
    match parser.parse_complete_with_tools(text, tools).await {
        Ok((normal_text, calls)) => json!({
            "parser": parser_name,
            "calls": calls.into_iter().map(|call| json!({
                "name": call.function.name,
                "arguments": decode_arguments(&call.function.arguments),
            })).collect::<Vec<_>>(),
            "normal_text": normal_text,
        }),
        Err(error) => parser_error(parser_name, error),
    }
}

async fn capture_tool_stream(family: &str, chunks: &[&str], tools: &[Tool]) -> Value {
    let Some(parser_name) = tool_parser_name(family) else {
        return unavailable("tool-parser", TOOL_PARSER_VERSION, family);
    };
    let factory = ToolParserFactory::new();
    let Some(mut parser) = factory.registry().create_parser(parser_name) else {
        return unavailable("tool-parser", TOOL_PARSER_VERSION, family);
    };

    let mut assembly = ToolAssembly::default();
    let mut captured_chunks = Vec::with_capacity(chunks.len());
    for chunk in chunks {
        match parser.parse_incremental(chunk, tools).await {
            Ok(result) => {
                let deltas = assembly.push(&result);
                captured_chunks.push(json!({
                    "expected": deltas,
                    "normal_text": result.normal_text,
                }));
            }
            Err(error) => return parser_error(parser_name, error),
        }
    }

    // SMG exposes any argument suffix not emitted incrementally through this
    // explicit accessor (the ToolParser trait has no finish method). Treat it as
    // end-of-stream output and attach it to the final input chunk.
    if let Some(tail) = parser.get_unstreamed_tool_args()
        && !tail.is_empty()
    {
        let tail_result = tool_parser::StreamingParseResult {
            normal_text: String::new(),
            calls: tail,
        };
        let deltas = assembly.push(&tail_result);
        if let Some(last) = captured_chunks.last_mut() {
            if let Some(expected) = last.get_mut("expected").and_then(Value::as_array_mut) {
                expected.extend(deltas);
            }
        } else {
            captured_chunks.push(json!({"expected": deltas, "normal_text": ""}));
        }
    }

    json!({
        "parser": parser_name,
        "calls": assembly.calls(),
        "normal_text": assembly.normal_text,
        "chunks": captured_chunks,
    })
}

fn prepare_reasoning_parser(
    parser_name: &str,
    force_reasoning: bool,
) -> Option<Box<dyn reasoning_parser::ReasoningParser>> {
    let factory = ReasoningParserFactory::new();
    let mut parser = factory.registry().create_parser(parser_name)?;
    if force_reasoning {
        parser.mark_reasoning_started();
        parser.mark_think_start_stripped();
    }
    Some(parser)
}

fn capture_reasoning_batch(family: &str, case: &ReasoningCase, text: &str) -> Value {
    let Some(parser_name) = reasoning_parser_name(family) else {
        return unavailable("reasoning-parser", REASONING_PARSER_VERSION, family);
    };
    let Some(mut parser) = prepare_reasoning_parser(parser_name, case.force_reasoning) else {
        return unavailable("reasoning-parser", REASONING_PARSER_VERSION, family);
    };
    match parser.detect_and_parse_reasoning(text) {
        Ok(result) => json!({
            "parser": parser_name,
            "reasoning_text": result.reasoning_text,
            "normal_text": result.normal_text,
        }),
        Err(error) => parser_error(parser_name, error),
    }
}

fn capture_reasoning_stream(family: &str, case: &ReasoningCase, chunks: &[String]) -> Value {
    let Some(parser_name) = reasoning_parser_name(family) else {
        return unavailable("reasoning-parser", REASONING_PARSER_VERSION, family);
    };
    let Some(mut parser) = prepare_reasoning_parser(parser_name, case.force_reasoning) else {
        return unavailable("reasoning-parser", REASONING_PARSER_VERSION, family);
    };
    let mut reasoning_text = String::new();
    let mut normal_text = String::new();
    let mut captured_chunks = Vec::with_capacity(chunks.len());
    for chunk in chunks {
        match parser.parse_reasoning_streaming_incremental(chunk) {
            Ok(result) => {
                reasoning_text.push_str(&result.reasoning_text);
                normal_text.push_str(&result.normal_text);
                captured_chunks.push(json!({
                    "reasoning_text": result.reasoning_text,
                    "normal_text": result.normal_text,
                }));
            }
            Err(error) => return parser_error(parser_name, error),
        }
    }
    json!({
        "parser": parser_name,
        "reasoning_text": reasoning_text,
        "normal_text": normal_text,
        "chunks": captured_chunks,
    })
}

fn unified_tools() -> Vec<Tool> {
    [
        ("get_weather", "city"),
        ("f", "x"),
        ("g", "y"),
        ("run", "cmd"),
    ]
    .into_iter()
    .map(|(name, key)| Tool {
        tool_type: "function".to_string(),
        function: Function {
            name: name.to_string(),
            description: None,
            parameters: json!({"type": "object", "properties": {key: {"type": "string"}}}),
            strict: None,
        },
    })
    .collect()
}

fn append_text_event(events: &mut Vec<Value>, kind: &str, text: &str) {
    if text.is_empty() {
        return;
    }
    if let Some(last) = events.last_mut()
        && last.get("kind").and_then(Value::as_str) == Some(kind)
        && let Some(existing) = last.get("text").and_then(Value::as_str)
    {
        let combined = format!("{existing}{text}");
        *last.get_mut("text").expect("text field exists") = Value::String(combined);
        return;
    }
    events.push(json!({"kind": kind, "text": text}));
}

fn assemble_unified(chunks: &[Value]) -> Vec<Value> {
    let mut events: Vec<Value> = Vec::new();
    let mut current_tool: Option<usize> = None;
    for chunk in chunks {
        let deltas = chunk.get("expected").and_then(Value::as_array);
        for delta in deltas.into_iter().flatten() {
            match delta.get("kind").and_then(Value::as_str) {
                Some(kind @ ("reasoning" | "text")) => {
                    current_tool = None;
                    append_text_event(
                        &mut events,
                        kind,
                        delta.get("text").and_then(Value::as_str).unwrap_or(""),
                    );
                }
                Some("tool_call") => {
                    let name = delta.get("name").and_then(Value::as_str);
                    let arguments = delta.get("arguments").and_then(Value::as_str).unwrap_or("");
                    if let Some(name) = name {
                        events.push(json!({"kind": "tool_call", "name": name, "_raw": arguments}));
                        current_tool = Some(events.len() - 1);
                    } else if let Some(index) = current_tool {
                        let raw = events[index]
                            .get("_raw")
                            .and_then(Value::as_str)
                            .unwrap_or("");
                        let mut combined = raw.to_string();
                        combined.push_str(arguments);
                        events[index]["_raw"] = Value::String(combined);
                    }
                }
                _ => {}
            }
        }
    }
    for event in &mut events {
        if event.get("kind").and_then(Value::as_str) != Some("tool_call") {
            continue;
        }
        let raw = event.get("_raw").and_then(Value::as_str).unwrap_or("");
        event["arguments"] = if raw.trim().is_empty() {
            json!({})
        } else {
            decode_arguments(raw)
        };
        event
            .as_object_mut()
            .expect("event is an object")
            .remove("_raw");
    }
    events
}

async fn capture_unified(family: &str, case: &UnifiedCase) -> Value {
    let Some((reasoning_name, tool_name)) = unified_parser_names(family) else {
        return json!({
            "unavailable": format!(
                "SMG reasoning-parser/tool-parser 1.6.0 cannot form a Combined parser for family '{family}'."
            )
        });
    };
    let force_reasoning = case
        .init
        .get("prefill")
        .and_then(Value::as_str)
        .is_some_and(|value| value != "None");
    let Some(mut reasoning) = prepare_reasoning_parser(reasoning_name, force_reasoning) else {
        return unavailable("reasoning-parser", REASONING_PARSER_VERSION, family);
    };
    let tool_factory = ToolParserFactory::new();
    let Some(mut tool) = tool_factory.registry().create_parser(tool_name) else {
        return unavailable("tool-parser", TOOL_PARSER_VERSION, family);
    };
    let tools = unified_tools();
    let mut chunks = Vec::with_capacity(case.chunks.len());

    for chunk in &case.chunks {
        let reasoning_result =
            match reasoning.parse_reasoning_streaming_incremental(&chunk.delta_text) {
                Ok(result) => result,
                Err(error) => return parser_error(reasoning_name, error),
            };
        let mut deltas = Vec::new();
        if !reasoning_result.reasoning_text.is_empty() {
            deltas.push(json!({"kind": "reasoning", "text": reasoning_result.reasoning_text}));
        }
        if !reasoning_result.normal_text.is_empty() {
            match tool
                .parse_incremental(&reasoning_result.normal_text, &tools)
                .await
            {
                Ok(result) => {
                    if !result.normal_text.is_empty() {
                        deltas.push(json!({"kind": "text", "text": result.normal_text}));
                    }
                    deltas.extend(result.calls.into_iter().map(|call| {
                        json!({
                            "kind": "tool_call",
                            "name": call.name,
                            "arguments": call.parameters,
                        })
                    }));
                }
                Err(error) => return parser_error(tool_name, error),
            }
        }
        chunks.push(json!({"expected": deltas}));
    }

    if let Some(tail) = tool.get_unstreamed_tool_args()
        && !tail.is_empty()
    {
        let deltas: Vec<Value> = tail
            .into_iter()
            .map(|call| {
                json!({
                    "kind": "tool_call",
                    "name": call.name,
                    "arguments": call.parameters,
                })
            })
            .collect();
        if let Some(last) = chunks.last_mut() {
            last["expected"]
                .as_array_mut()
                .expect("expected is an array")
                .extend(deltas);
        } else {
            chunks.push(json!({"expected": deltas}));
        }
    }

    json!({
        "parser": format!("Combined({reasoning_name} + {tool_name})"),
        "assembled": assemble_unified(&chunks),
        "chunks": chunks,
    })
}

async fn capture_toolcalling(root: &Path, output: &mut ToolCaptures, coverage: &mut Coverage) {
    for path in collect_yaml(root) {
        let text = std::fs::read_to_string(&path).expect("read tool fixture");
        let fixture: ToolFixture = match serde_yaml::from_str(&text) {
            Ok(fixture) => fixture,
            Err(_) => continue,
        };
        let mode = fixture.mode.as_deref().unwrap_or("");
        for (case_id, case) in &fixture.cases {
            let key = case_key(&fixture.family, case_id);
            let tools: Vec<Tool> = case.tools.iter().map(Tool::from).collect();
            if mode == "batch"
                && let Some(model_text) = case.model_text.as_deref()
            {
                coverage.tool_batch_inputs += 1;
                output.batch.insert(
                    key.clone(),
                    capture_tool_batch(&fixture.family, model_text, &tools).await,
                );
                output.stream_on_batch.insert(
                    key,
                    capture_tool_stream(&fixture.family, &[model_text], &tools).await,
                );
            } else if matches!(mode, "stream" | "streamv2") && !case.chunks.is_empty() {
                coverage.tool_stream_inputs += 1;
                let chunks: Vec<&str> = case
                    .chunks
                    .iter()
                    .map(|chunk| chunk.delta_text.as_str())
                    .collect();
                output.stream.insert(
                    key,
                    capture_tool_stream(&fixture.family, &chunks, &tools).await,
                );
            }
        }
    }
}

fn capture_reasoning(root: &Path, output: &mut ReasoningCaptures, coverage: &mut Coverage) {
    for path in collect_yaml(root) {
        let text = std::fs::read_to_string(&path).expect("read reasoning fixture");
        let fixture: ReasoningFixture = match serde_yaml::from_str(&text) {
            Ok(fixture) => fixture,
            Err(_) => continue,
        };
        for (case_id, case) in &fixture.cases {
            let key = case_key(&fixture.family, case_id);
            match fixture.mode.as_str() {
                "batch" => {
                    if let Some(model_text) = case.model_text.as_deref() {
                        coverage.reasoning_batch_inputs += 1;
                        output.batch.insert(
                            key,
                            capture_reasoning_batch(&fixture.family, case, model_text),
                        );
                    }
                }
                "stream" => {
                    if let Some(chunks) = case.chunks.as_deref() {
                        coverage.reasoning_stream_inputs += 1;
                        output
                            .stream
                            .insert(key, capture_reasoning_stream(&fixture.family, case, chunks));
                    }
                }
                _ => {}
            }
        }
    }
}

async fn capture_unified_inputs(
    root: &Path,
    output: &mut BTreeMap<String, Value>,
    coverage: &mut Coverage,
) {
    for path in collect_yaml(root) {
        let text = std::fs::read_to_string(&path).expect("read unified fixture");
        let fixture: UnifiedFixture = match serde_yaml::from_str(&text) {
            Ok(fixture) => fixture,
            Err(_) => continue,
        };
        for (case_id, case) in &fixture.cases {
            if case.chunks.is_empty() {
                continue;
            }
            coverage.unified_inputs += 1;
            let scenario = case.scenario.as_deref().unwrap_or(case_id);
            output.insert(
                case_key(&fixture.family, scenario),
                capture_unified(&fixture.family, case).await,
            );
        }
    }
}

fn env_path(name: &str, fallback: impl FnOnce() -> PathBuf) -> PathBuf {
    std::env::var_os(name)
        .map(PathBuf::from)
        .unwrap_or_else(fallback)
}

#[tokio::test(flavor = "multi_thread")]
async fn smg_capture_covers_all_conformance_inputs() {
    let fixture_root = common::ensure_fixtures();
    let toolcalling = env_path("SMG_TOOLCALLING_FIXTURES", || {
        fixture_root.join("toolcalling/fixtures-batch-v1/inputs")
    });
    let toolcalling_stream = std::env::var_os("SMG_TOOLCALLING_STREAM_FIXTURES")
        .map(PathBuf::from)
        .unwrap_or_else(|| fixture_root.join("toolcalling/fixtures-stream-v2/inputs"));
    let reasoning = env_path("SMG_REASONING_FIXTURES", || {
        fixture_root.join("reasoning/fixtures-v1/inputs")
    });
    let unified = env_path("SMG_UNIFIED_FIXTURES", || {
        fixture_root.join("unified/inputs")
    });

    let mut tool_captures = ToolCaptures::default();
    let mut reasoning_captures = ReasoningCaptures::default();
    let mut unified_captures = BTreeMap::new();
    let mut coverage = Coverage::default();

    capture_toolcalling(&toolcalling, &mut tool_captures, &mut coverage).await;
    if toolcalling_stream != toolcalling {
        capture_toolcalling(&toolcalling_stream, &mut tool_captures, &mut coverage).await;
    }
    capture_reasoning(&reasoning, &mut reasoning_captures, &mut coverage);
    capture_unified_inputs(&unified, &mut unified_captures, &mut coverage).await;

    assert!(
        coverage.tool_batch_inputs > 0,
        "no tool batch inputs captured"
    );
    assert!(
        coverage.tool_stream_inputs > 0,
        "no tool stream inputs captured"
    );
    assert!(
        coverage.reasoning_batch_inputs > 0,
        "no reasoning batch inputs captured"
    );
    assert!(
        coverage.reasoning_stream_inputs > 0,
        "no reasoning stream inputs captured"
    );
    assert!(coverage.unified_inputs > 0, "no unified inputs captured");
    assert_eq!(tool_captures.batch.len(), coverage.tool_batch_inputs);
    assert_eq!(
        tool_captures.stream_on_batch.len(),
        coverage.tool_batch_inputs
    );
    assert_eq!(tool_captures.stream.len(), coverage.tool_stream_inputs);
    assert_eq!(
        reasoning_captures.batch.len(),
        coverage.reasoning_batch_inputs
    );
    assert_eq!(
        reasoning_captures.stream.len(),
        coverage.reasoning_stream_inputs
    );
    assert_eq!(unified_captures.len(), coverage.unified_inputs);

    let doc = CaptureDoc {
        schema: "smg-conformance/v1",
        tool_parser_version: TOOL_PARSER_VERSION,
        reasoning_parser_version: REASONING_PARSER_VERSION,
        toolcalling: tool_captures,
        reasoning: reasoning_captures,
        unified: unified_captures,
        coverage,
    };
    if let Some(output) = std::env::var_os("SMG_CAPTURE_OUTPUT") {
        let output = PathBuf::from(output);
        if let Some(parent) = output.parent() {
            std::fs::create_dir_all(parent).expect("create SMG capture output directory");
        }
        std::fs::write(
            &output,
            serde_json::to_vec_pretty(&doc).expect("serialize SMG capture"),
        )
        .unwrap_or_else(|error| panic!("write {}: {error}", output.display()));
        eprintln!("wrote {}", output.display());
    }
}
