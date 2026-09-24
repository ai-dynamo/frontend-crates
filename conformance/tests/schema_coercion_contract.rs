// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Executable A–H schema contracts. Run all rows, write observations, then fail the group.
//! No production-code changes or parser-derived expectations. See schema_suite/README.md.
use std::collections::BTreeMap;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::{LazyLock, Mutex};

use anyhow::{Context, Result, ensure};
use dynamo_parsers::tool_calling::{
    ToolChoice, ToolDefinition,
    jail::{Annotated, JailedStream},
    tools::try_tool_call_parse_aggregate_finalize,
};
use dynamo_parsers_v2::{
    InvalidGuidedPayload, InvalidGuidedPayloadPolicy, Tool, ToolCallDelta, UnifiedParserEvent,
    UnifiedParserExt, UnifiedParserInit, UnifiedParserStartingState, UnifiedToolOutputMode,
    create_tool_parser_for_family, create_unified_parser_for_family,
};
use dynamo_protocols::types::{
    ChatChoiceStream, ChatCompletionMessageContent, ChatCompletionStreamResponseDelta,
    CreateChatCompletionStreamResponse, Role,
};
use futures::{StreamExt, stream};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

static OUTPUT_LOCK: Mutex<()> = Mutex::new(());
fn suite_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("schema_suite")
}
fn python() -> String {
    std::env::var("SCHEMA_SUITE_PYTHON").unwrap_or_else(|_| "python3".into())
}
static CORPUS: LazyLock<Value> = LazyLock::new(|| {
    let output = Command::new(python())
        .arg(suite_dir().join("cases.py"))
        .output()
        .expect("case generator");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).expect("authored corpus JSON")
});
fn record(row: &Value) {
    if let Ok(path) = std::env::var("SCHEMA_SUITE_RESULTS") {
        let _lock = OUTPUT_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .expect("results file");
        writeln!(file, "{row}").expect("write result");
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
struct Call {
    name: String,
    arguments: String,
}
#[derive(Default, Debug, Serialize)]
struct Observation {
    ids: Vec<Option<String>>,
    typed_rejection: bool,
    calls: Vec<Call>,
    text: String,
    events: Vec<Value>,
    deltas: Vec<Value>,
    errors: Vec<String>,
    lifecycle: Vec<String>,
    early_arguments: bool,
}
fn str_at<'a>(v: &'a Value, key: &str) -> &'a str {
    v[key].as_str().unwrap_or("")
}
fn tools(c: &Value) -> Result<Vec<Tool>> {
    c["tools"]
        .as_array()
        .context("tools array")?
        .iter()
        .map(|v| {
            Ok(Tool {
                name: str_at(v, "name").into(),
                description: None,
                parameters: v.get("parameters").cloned().unwrap_or(Value::Null),
                strict: v["strict"].as_bool(),
            })
        })
        .collect()
}
fn v1_tools(c: &Value) -> Result<Vec<ToolDefinition>> {
    c["tools"]
        .as_array()
        .context("tools array")?
        .iter()
        .map(|v| {
            Ok(ToolDefinition {
                name: str_at(v, "name").into(),
                parameters: v.get("parameters").cloned(),
                strict: v["strict"].as_bool(),
            })
        })
        .collect()
}
fn partitions<'a>(input: &'a str, strategy: &str) -> Vec<Vec<&'a str>> {
    let boundaries: Vec<_> = input
        .char_indices()
        .map(|(i, _)| i)
        .chain([input.len()])
        .collect();
    match strategy {
        "whole" => vec![vec![input]],
        "all_splits" => boundaries
            .iter()
            .map(|&i| vec![&input[..i], &input[i..]])
            .collect(),
        "chars" => vec![boundaries.windows(2).map(|w| &input[w[0]..w[1]]).collect()],
        "seeded" => {
            let mut out = vec![""];
            let mut at = 0;
            let mut seed = 17_u32;
            while at + 1 < boundaries.len() {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                let next = (at + 1 + (seed as usize % 7)).min(boundaries.len() - 1);
                out.push(&input[boundaries[at]..boundaries[next]]);
                at = next;
            }
            out.push("");
            vec![out]
        }
        _ => panic!("unknown chunk strategy {strategy}"),
    }
}
fn check_deltas(deltas: &[ToolCallDelta], o: &mut Observation) {
    let mut accumulated: BTreeMap<usize, (String, String, bool)> = BTreeMap::new();
    let mut order = vec![];
    for d in deltas {
        o.deltas.push(json!({"index":d.tool_index,"name":d.name,"arguments":d.arguments,"complete":d.complete}));
        if !accumulated.contains_key(&d.tool_index) {
            order.push(d.tool_index);
        }
        let entry = accumulated.entry(d.tool_index).or_default();
        if entry.2 {
            o.lifecycle
                .push(format!("update after completion for {}", d.tool_index));
        }
        if let Some(n) = &d.name {
            if !entry.0.is_empty() {
                o.lifecycle
                    .push(format!("name repeated for {}", d.tool_index));
            }
            entry.0 = n.clone();
        }
        entry.1.push_str(&d.arguments);
        if d.complete {
            if entry.0.is_empty() {
                o.lifecycle.push("complete call has no name".into());
            }
            if serde_json::from_str::<Value>(&entry.1).is_err() {
                o.lifecycle
                    .push(format!("completed invalid JSON: {}", entry.1));
            }
            entry.2 = true;
        }
    }
    for index in order {
        let (name, arguments, complete) = accumulated.remove(&index).unwrap();
        if complete {
            o.calls.push(Call { name, arguments });
        }
    }
}
#[allow(deprecated)]
fn chunk(content: &str) -> Annotated<CreateChatCompletionStreamResponse> {
    Annotated {
        data: Some(CreateChatCompletionStreamResponse {
            id: "schema-suite".into(),
            created: 0,
            model: "unit-test".into(),
            object: "chat.completion.chunk".into(),
            system_fingerprint: None,
            usage: None,
            service_tier: None,
            choices: vec![ChatChoiceStream {
                index: 0,
                delta: ChatCompletionStreamResponseDelta {
                    role: Some(Role::Assistant),
                    content: Some(ChatCompletionMessageContent::Text(content.into())),
                    tool_calls: None,
                    function_call: None,
                    refusal: None,
                    reasoning_content: None,
                },
                finish_reason: None,
                logprobs: None,
            }],
        }),
        id: None,
        event: None,
        comment: None,
        error: None,
    }
}
async fn parse(c: &Value, surface: &str, chunks: &[&str], token_mode: bool) -> Result<Observation> {
    let family = str_at(c, "family");
    let input = str_at(c, "input");
    let selector = c["selector"].as_str().unwrap_or(family);
    let mut o = Observation::default();
    let ts1 = v1_tools(c)?;
    let ts2 = tools(c)?;
    let schemas1: Vec<_> = ts1.iter().map(|t| t.parameters.clone()).collect();
    let schemas2: Vec<_> = ts2.iter().map(|t| t.parameters.clone()).collect();
    if surface == "v1_batch" {
        let ts = &ts1;
        let (calls, text) =
            try_tool_call_parse_aggregate_finalize(input, Some(selector), Some(ts)).await?;
        o.ids = calls.iter().map(|c| Some(c.id.clone())).collect();
        o.calls = calls
            .into_iter()
            .map(|c| Call {
                name: c.function.name,
                arguments: c.function.arguments,
            })
            .collect();
        o.text = text.unwrap_or_default();
    } else if surface == "v1_jail" {
        let jail = JailedStream::builder()
            .tool_call_parser(selector)
            .tool_definitions(ts1.clone())
            .build();
        let out: Vec<_> = jail
            .apply_with_finish_reason(stream::iter(
                chunks.iter().map(|s| chunk(s)).collect::<Vec<_>>(),
            ))
            .collect()
            .await;
        let mut calls: BTreeMap<u32, Call> = BTreeMap::new();
        let mut ids: BTreeMap<u32, Option<String>> = BTreeMap::new();
        for item in out {
            if item.error.is_some() {
                o.errors.push(format!("{:?}", item.error));
            }
            if let Some(data) = item.data {
                for ch in data.choices {
                    if let Some(ChatCompletionMessageContent::Text(t)) = ch.delta.content {
                        o.text.push_str(&t);
                    }
                    if let Some(cs) = ch.delta.tool_calls {
                        for call in cs {
                            if call.id.is_some() {
                                ids.insert(call.index, call.id.clone());
                            }
                            if let Some(f) = call.function {
                                let e = calls.entry(call.index).or_insert(Call {
                                    name: String::new(),
                                    arguments: String::new(),
                                });
                                if let Some(n) = f.name {
                                    e.name.push_str(&n);
                                }
                                if let Some(a) = f.arguments {
                                    e.arguments.push_str(&a);
                                }
                            }
                        }
                    }
                }
            }
        }
        o.ids = calls
            .keys()
            .map(|i| ids.get(i).cloned().flatten())
            .collect();
        o.calls = calls.into_values().collect();
    } else if surface == "v2_tool" {
        let selector = if selector == "deepseek_v41" {
            "deepseek_v4"
        } else {
            selector
        };
        let mut p = create_tool_parser_for_family(selector, &ts2)?;
        let mut deltas = vec![];
        let mut fed = String::new();
        if token_mode {
            let ids = dynamo_parsers_v2::tool_calling::harmony::encode_harmony(input)?;
            for id in ids {
                let r = p.push_tokens(&[id])?;
                deltas.extend(r.calls);
                o.text.push_str(&r.normal_text);
            }
        } else {
            for s in chunks {
                fed.push_str(s);
                let r = match p.push(s) {
                    Ok(r) => r,
                    Err(e) => {
                        o.errors.push(e.to_string());
                        break;
                    }
                };
                if !fed.contains("</parameter>") && r.calls.iter().any(|d| !d.arguments.is_empty())
                {
                    o.early_arguments = true;
                }
                deltas.extend(r.calls);
                o.text.push_str(&r.normal_text);
            }
        }
        if o.errors.is_empty() {
            match p.finish() {
                Ok(r) => {
                    deltas.extend(r.calls);
                    o.text.push_str(&r.normal_text);
                }
                Err(e) => o.errors.push(e.to_string()),
            }
        }
        check_deltas(&deltas, &mut o);
        o.ids = (0..o.calls.len())
            .map(|i| p.tool_call_id(i).map(str::to_owned))
            .collect();
    } else {
        let mut p = create_unified_parser_for_family(selector, &ts2)?;
        let mode = match str_at(&c["init"], "mode") {
            "named" => UnifiedToolOutputMode::GuidedJson {
                named_tool: Some("f".into()),
            },
            "required" => UnifiedToolOutputMode::GuidedJson { named_tool: None },
            _ => UnifiedToolOutputMode::Native,
        };
        let init = UnifiedParserInit {
            starting_state: match str_at(&c["init"], "start") {
                "reasoning" => UnifiedParserStartingState::Reasoning,
                "response" => UnifiedParserStartingState::Response,
                _ => UnifiedParserStartingState::None,
            },
            tool_output_mode: mode,
            invalid_guided_payload: match str_at(&c["init"], "policy") {
                "recover" => InvalidGuidedPayloadPolicy::RecoverAsText,
                "stream" => InvalidGuidedPayloadPolicy::StreamBestEffort,
                _ => InvalidGuidedPayloadPolicy::Reject,
            },
            ..Default::default()
        };
        p.initialize_request(init.clone())?;
        if let Some(prefix) = c["reset_prefix"].as_str() {
            p.push(prefix)?;
            p.reset();
            p.initialize_request(init)?;
        }
        let mut events = vec![];
        let mut fed = String::new();
        for s in chunks {
            fed.push_str(s);
            match p.push(s) {
                Ok(r) => {
                    if !fed.contains("</parameter>") && r.iter().any(|e|matches!(e,UnifiedParserEvent::ToolCall(d) if !d.arguments.is_empty())){o.early_arguments=true;}
                    events.extend(r);
                }
                Err(e) => {
                    o.typed_rejection = e.downcast_ref::<InvalidGuidedPayload>().is_some();
                    o.errors.push(e.to_string());
                    break;
                }
            }
        }
        if o.errors.is_empty() {
            match p.finish() {
                Ok(r) => events.extend(r.events),
                Err(e) => {
                    o.typed_rejection = e.downcast_ref::<InvalidGuidedPayload>().is_some();
                    o.errors.push(e.to_string());
                }
            }
        }
        let ds: Vec<_> = events
            .iter()
            .filter_map(|e| {
                if let UnifiedParserEvent::ToolCall(d) = e {
                    Some(d.clone())
                } else {
                    None
                }
            })
            .collect();
        check_deltas(&ds, &mut o);
        o.ids = (0..o.calls.len())
            .map(|i| p.tool_call_id(i).map(str::to_owned))
            .collect();
        for e in &events {
            if let UnifiedParserEvent::Text(t) = e {
                o.text.push_str(t);
            }
        }
        o.events = serde_json::to_value(dynamo_parsers_v2::assemble(&events))?
            .as_array()
            .unwrap()
            .clone();
    }
    ensure!(
        schemas1 == ts1.iter().map(|t| t.parameters.clone()).collect::<Vec<_>>(),
        "v1 schema mutation"
    );
    ensure!(
        schemas2 == ts2.iter().map(|t| t.parameters.clone()).collect::<Vec<_>>(),
        "v2 schema mutation"
    );
    Ok(o)
}
fn semantic(v: Value) -> Value {
    match v {
        Value::Array(a) => Value::Array(a.into_iter().map(semantic).collect()),
        Value::Object(m) => Value::Object(m.into_iter().map(|(k, v)| (k, semantic(v))).collect()),
        Value::Number(ref n) if n.is_f64() => {
            let x = n.as_f64().unwrap();
            if x.fract() == 0.0 && x.abs() <= 9007199254740992.0 {
                json!(x as i64)
            } else {
                v
            }
        }
        _ => v,
    }
}
fn validate(c: &Value, o: &Observation) -> Result<()> {
    if c["expect_error"].as_bool() == Some(true) {
        ensure!(
            o.typed_rejection,
            "expected a typed guided-payload rejection, got {:?}",
            o.errors
        );
        ensure!(o.calls.is_empty(), "rejected request committed calls");
        return Ok(());
    }
    if let Some(message) = c["allow_error_contains"].as_str()
        && !o.errors.is_empty()
    {
        ensure!(
            o.errors.iter().all(|e| e.contains(message)),
            "unexpected parser error: {:?}",
            o.errors
        );
        ensure!(o.calls.is_empty(), "invalid input committed a call");
        return Ok(());
    }
    ensure!(o.errors.is_empty(), "parser errors: {:?}", o.errors);
    ensure!(
        o.lifecycle.is_empty(),
        "raw delta contract: {:?}",
        o.lifecycle
    );
    let expected: Vec<Call> = serde_json::from_value(c["expected"].clone())?;
    ensure!(
        o.calls.len() == expected.len(),
        "call count: actual {}, expected {}",
        o.calls.len(),
        expected.len()
    );
    for (got, want) in o.calls.iter().zip(&expected) {
        ensure!(
            got.name == want.name,
            "tool name: {:?} != {:?}",
            got.name,
            want.name
        );
        let g: Value =
            serde_json::from_str(&got.arguments).context("actual arguments invalid JSON")?;
        let w: Value = serde_json::from_str(&want.arguments)?;
        ensure!(
            semantic(g) == semantic(w),
            "typed arguments: {} != {}",
            got.arguments,
            want.arguments
        );
    }
    if let Some(ids) = c.get("expected_ids") {
        ensure!(json!(o.ids) == *ids, "native call IDs differ: {:?}", o.ids);
    }
    if let Some(exact) = c["raw_exact"].as_array() {
        ensure!(
            o.calls
                .iter()
                .map(|v| json!(v.arguments))
                .collect::<Vec<_>>()
                == *exact,
            "source order/raw argument spelling differs"
        );
    }
    if let Some(needles) = c["raw_contains"].as_array() {
        for n in needles {
            ensure!(
                o.calls
                    .iter()
                    .any(|v| v.arguments.contains(n.as_str().unwrap())),
                "numeric lexeme lost: {n}"
            );
        }
    }
    if c.get("events").is_some() {
        ensure!(
            semantic(json!(o.events)) == semantic(c["events"].clone()),
            "ordered events differ"
        );
    }
    if let Some(t) = c["expected_text"].as_str() {
        ensure!(o.text == t, "text recovery differs");
    }
    if let Some(early) = c["early"].as_str() {
        ensure!(
            o.early_arguments == (early == "string"),
            "eager emission: actual {}, expected {}",
            o.early_arguments,
            early
        );
    }
    Ok(())
}
async fn parser_group(group: &str) {
    let mut failed = 0;
    let mut total = 0;
    for c in CORPUS["cases"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|c| str_at(c, "group") == group)
    {
        let family = str_at(c, "family");
        if let Ok(filter) = std::env::var("SCHEMA_SUITE_FAMILY")
            && filter != family
        {
            continue;
        }
        for surface in ["v1_batch", "v1_jail", "v2_tool", "unified"] {
            if let Some(only) = c["surfaces"].as_array()
                && !only.iter().any(|s| s == surface)
            {
                continue;
            }
            let mut row = json!({"id":c["id"],"group":group,"family":family,"surface":surface,"input":c["input"],"tools":c["tools"],"expected":c["expected"],"contract":c["contract"]});
            row["case"] = c.clone();
            row["init"] = c["init"].clone();
            if CORPUS["capabilities"][family][surface] != true {
                row["status"] = json!("unavailable");
                row["reason"] = json!("No implementation in the pinned registry");
                record(&row);
                continue;
            }
            let mut checks = 0;
            let mut errors = 0;
            let mut examples = vec![];
            for strategy in c["strategies"].as_array().unwrap() {
                if surface == "v1_batch" && strategy != "whole" {
                    continue;
                }
                for (partition, chunks) in
                    partitions(str_at(c, "input"), strategy.as_str().unwrap())
                        .iter()
                        .enumerate()
                {
                    checks += 1;
                    let result = parse(c, surface, chunks, false).await;
                    if row.get("actual").is_none()
                        && let Ok(o) = &result
                    {
                        row["actual"] = json!(o);
                        row["sample_strategy"] = strategy.clone();
                    }
                    let validation = match &result {
                        Ok(o) => validate(c, o),
                        Err(e) => Err(anyhow::anyhow!(e.to_string())),
                    };
                    if let Err(e) = validation {
                        errors += 1;
                        if examples.len() < 3 {
                            examples.push(json!({"strategy":strategy,"partition":partition,"chunks":chunks,"reason":e.to_string(),"actual":result.ok()}));
                        }
                    }
                }
            }
            // Token-native Harmony is independently exercised, not approximated with strings.
            if family == "harmony" && surface == "v2_tool" {
                checks += 1;
                let result = parse(c, surface, &[], true).await;
                let validation = match &result {
                    Ok(o) => validate(c, o),
                    Err(e) => Err(anyhow::anyhow!(e.to_string())),
                };
                if let Err(e) = validation {
                    errors += 1;
                    if examples.len() < 3 {
                        examples.push(json!({"strategy":"token_ids","reason":e.to_string(),"actual":result.ok()}));
                    }
                }
            }
            if checks == 0 {
                row["status"] = json!("not_applicable");
                row["reason"] = json!("Streaming-only assertion");
                record(&row);
                continue;
            }
            total += 1;
            if errors > 0 {
                failed += 1;
            }
            row["status"] = json!(if errors == 0 { "pass" } else { "fail" });
            row["checks"] = json!(checks);
            row["failed_checks"] = json!(errors);
            row["failures"] = json!(examples);
            record(&row);
        }
    }
    assert!(total > 0, "{group}: no executable cases selected");
    assert_eq!(
        failed, 0,
        "{group}: {failed}/{total} case/surface rows failed (full evidence in SCHEMA_SUITE_RESULTS)"
    );
}
fn support_group(group: &str) {
    if matches!(group, "H05" | "H07" | "H08") {
        rust_harness_group(group);
        return;
    }
    let mut payload = json!({"group":group});
    if group.starts_with('G') {
        let output = Command::new(python())
            .arg(suite_dir().join("support.py"))
            .args(["grammar_cases", group])
            .output()
            .expect("grammar cases");
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let mut cases: Value = serde_json::from_slice(&output.stdout).expect("grammar case JSON");
        for c in cases.as_array_mut().unwrap() {
            use dynamo_parsers::tool_calling::structural_tag::{
                StructuralTagBuilder, StructuralTagSchemaMode, ToolCallFormatBuildContext,
            };
            let choice = match str_at(c, "choice") {
                "named" => ToolChoice::Named("f".into()),
                "required" => ToolChoice::Required,
                _ => ToolChoice::Auto,
            };
            let ts = vec![ToolDefinition {
                name: "f".into(),
                parameters: Some(c["schema"].clone()),
                strict: c["strict"].as_bool(),
            }];
            let builder = if str_at(c, "family") == "generic" {
                dynamo_parsers::tool_calling::parsers::get_tool_parser_map()["qwen3_coder"]
                    .structural_tag_builder
                    .clone()
                    .unwrap()
            } else {
                StructuralTagBuilder::KimiK3
            };
            let ctx = ToolCallFormatBuildContext {
                tool_choice: &choice,
                tools: &ts,
                parallel_tool_calls: c["parallel"].as_bool(),
                schema_mode: if c["schema_mode"] == "strict" {
                    StructuralTagSchemaMode::Strict
                } else {
                    StructuralTagSchemaMode::Auto
                },
                starts_in_reasoning: false,
            };
            match builder.build_tool_call_format(&ctx) {
                Ok(v) => c["grammar"] = json!(v),
                Err(e) => c["build_error"] = json!(e.to_string()),
            }
        }
        payload["cases"] = cases;
    } else {
        payload["capabilities"] = CORPUS["capabilities"].clone();
    }
    let py = if group.starts_with('G') {
        std::env::var("SCHEMA_SUITE_XGRAMMAR_PYTHON").unwrap_or_else(|_| python())
    } else {
        python()
    };
    let mut child = Command::new(py)
        .arg(suite_dir().join("support.py"))
        .arg("check")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("support runner");
    child
        .stdin
        .take()
        .unwrap()
        .write_all(payload.to_string().as_bytes())
        .unwrap();
    let output = child.wait_with_output().expect("support output");
    assert!(
        output.status.success(),
        "support process failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let rows: Vec<Value> = serde_json::from_slice(&output.stdout).expect("support JSON");
    assert!(!rows.is_empty(), "no support observations");
    let mut fails = 0;
    for mut row in rows {
        if row["status"] == "pass" && (group == "G14" || group == "H03") {
            let runtime = tokio::runtime::Runtime::new().unwrap();
            let c = if group == "G14" {
                json!({"family":"kimi_k3","tools":row["tools"],"input":row["parse_input"],"expected":[{"name":"f","arguments":row["parse_expected"].to_string()}]})
            } else {
                let ts: Vec<Tool> = serde_yaml::from_str(
                    &serde_yaml::to_string(&row["rust_roundtrip_tools"]).unwrap(),
                )
                .unwrap();
                assert_eq!(
                    serde_json::to_value(ts).unwrap(),
                    row["rust_roundtrip_tools"]
                );
                json!({"family":"glm47","tools":row["rust_roundtrip_tools"],"input":row["input"]["input"],"expected":[{"name":"f","arguments":"{\"x\":null}"}]})
            };
            let input = str_at(&c, "input");
            let surfaces: &[&str] = if group == "G14" {
                &["v1_batch", "v1_jail", "v2_tool", "unified"]
            } else {
                &["unified"]
            };
            let mut observations = vec![];
            for surface in surfaces {
                let result = runtime.block_on(parse(&c, surface, &[input], false));
                let validation = result
                    .as_ref()
                    .map_err(|e| anyhow::anyhow!(e.to_string()))
                    .and_then(|o| validate(&c, o));
                observations.push(json!({"surface":surface,"actual":result.as_ref().ok(),"reason":validation.as_ref().err().map(ToString::to_string)}));
                row["checks"] = json!(row["checks"].as_u64().unwrap_or(0) + 1);
                if let Err(e) = validation {
                    row["status"] = json!("fail");
                    row["failed_checks"] = json!(row["failed_checks"].as_u64().unwrap_or(0) + 1);
                    row["reason"] = json!(format!(
                        "{surface} accepted grammar/YAML payload parser roundtrip: {e}"
                    ));
                }
            }
            row["parser_roundtrip"] = json!(observations);
        }
        if row["status"] != "pass" {
            fails += 1;
        }
        record(&row);
    }
    assert_eq!(
        fails, 0,
        "{group}: {fails} contract failures/errors; see report"
    );
}

fn rust_harness_group(group: &str) {
    if group == "H05" {
        let mut failures = 0;
        for (family, surfaces) in CORPUS["capabilities"].as_object().unwrap() {
            for (surface, expected) in surfaces.as_object().unwrap() {
                let actual = match surface.as_str() {
                    "v1_batch" | "v1_jail" => {
                        dynamo_parsers::tool_calling::parsers::get_tool_parser_map()
                            .contains_key(family.as_str())
                    }
                    "v2_tool" => create_tool_parser_for_family(
                        if family == "deepseek_v41" {
                            "deepseek_v4"
                        } else {
                            family
                        },
                        &[],
                    )
                    .is_ok(),
                    _ => create_unified_parser_for_family(family, &[]).is_ok(),
                };
                let pass = json!(actual) == *expected;
                if !pass {
                    failures += 1;
                }
                record(
                    &json!({"id":format!("H05.{family}.{surface}"),"group":group,"family":family,"surface":"inventory","checked_surface":surface,"status":if pass{"pass"}else{"fail"},"expected_supported":expected,"actual_supported":actual,"checks":1}),
                );
            }
        }
        assert_eq!(failures, 0, "registry and coverage inventory differ");
    } else {
        let result = (|| -> Result<()> {
            if group == "H07" {
                let c = json!({"expected":[{"name":"f","arguments":"{\"x\":9007199254740993}"}],"raw_contains":["9007199254740993"]});
                let good = Observation {
                    calls: vec![Call {
                        name: "f".into(),
                        arguments: "{\"x\":9007199254740993}".into(),
                    }],
                    ..Default::default()
                };
                validate(&c, &good)?;
                let bad = Observation {
                    calls: vec![Call {
                        name: "f".into(),
                        arguments: "{\"x\":9007199254740992}".into(),
                    }],
                    ..Default::default()
                };
                ensure!(
                    validate(&c, &bad).is_err(),
                    "oracle missed numeric rounding"
                );
                let c = json!({"expected":[{"name":"f","arguments":"{\"z\":1,\"a\":2}"}],"raw_exact":["{\"z\":1,\"a\":2}"]});
                let bad = Observation {
                    calls: vec![Call {
                        name: "f".into(),
                        arguments: "{\"a\":2,\"z\":1}".into(),
                    }],
                    ..Default::default()
                };
                ensure!(validate(&c, &bad).is_err(), "oracle missed key reorder");
            } else {
                let bad = vec![ToolCallDelta {
                    tool_index: 0,
                    name: Some("f".into()),
                    arguments: "{bad".into(),
                    complete: true,
                }];
                let events: Vec<_> = bad
                    .iter()
                    .cloned()
                    .map(UnifiedParserEvent::ToolCall)
                    .collect();
                let mut o = Observation::default();
                check_deltas(&bad, &mut o);
                ensure!(
                    !o.lifecycle.is_empty(),
                    "raw validator missed invalid completion"
                );
                let assembled = dynamo_parsers_v2::assemble(&events);
                record(
                    &json!({"id":"H08.assembly_evidence","group":group,"family":"harness","surface":"oracle","status":"pass","checks":1,"raw":o.deltas,"assembled":assembled,"rejected_by_raw_validator":o.lifecycle}),
                );
                let duplicate = vec![
                    ToolCallDelta {
                        tool_index: 0,
                        name: Some("f".into()),
                        arguments: "{}".into(),
                        complete: true,
                    },
                    ToolCallDelta {
                        tool_index: 0,
                        name: None,
                        arguments: "".into(),
                        complete: true,
                    },
                ];
                let mut o = Observation::default();
                check_deltas(&duplicate, &mut o);
                ensure!(
                    !o.lifecycle.is_empty(),
                    "oracle missed duplicate completion"
                );
            }
            Ok(())
        })();
        record(
            &json!({"id":format!("{group}.oracle"),"group":group,"family":"harness","surface":"oracle","status":if result.is_ok(){"pass"}else{"fail"},"checks":1,"reason":result.as_ref().err().map(ToString::to_string)}),
        );
        result.unwrap();
    }
}

#[tokio::test]
async fn a01() {
    parser_group("A01").await;
}

#[tokio::test]
async fn a02() {
    parser_group("A02").await;
}

#[tokio::test]
async fn a03() {
    parser_group("A03").await;
}

#[tokio::test]
async fn a04() {
    parser_group("A04").await;
}

#[tokio::test]
async fn a05() {
    parser_group("A05").await;
}

#[tokio::test]
async fn a06() {
    parser_group("A06").await;
}

#[tokio::test]
async fn a07() {
    parser_group("A07").await;
}

#[tokio::test]
async fn a08() {
    parser_group("A08").await;
}

#[tokio::test]
async fn a09() {
    parser_group("A09").await;
}

#[tokio::test]
async fn a10() {
    parser_group("A10").await;
}

#[tokio::test]
async fn a11() {
    parser_group("A11").await;
}

#[tokio::test]
async fn a12() {
    parser_group("A12").await;
}

#[tokio::test]
async fn a13() {
    parser_group("A13").await;
}

#[tokio::test]
async fn a14() {
    parser_group("A14").await;
}

#[tokio::test]
async fn a15() {
    parser_group("A15").await;
}

#[tokio::test]
async fn a16() {
    parser_group("A16").await;
}

#[tokio::test]
async fn b01() {
    parser_group("B01").await;
}

#[tokio::test]
async fn b02() {
    parser_group("B02").await;
}

#[tokio::test]
async fn b03() {
    parser_group("B03").await;
}

#[tokio::test]
async fn b04() {
    parser_group("B04").await;
}

#[tokio::test]
async fn b05() {
    parser_group("B05").await;
}

#[tokio::test]
async fn b06() {
    parser_group("B06").await;
}

#[tokio::test]
async fn b07() {
    parser_group("B07").await;
}

#[tokio::test]
async fn b08() {
    parser_group("B08").await;
}

#[tokio::test]
async fn b09() {
    parser_group("B09").await;
}

#[tokio::test]
async fn b10() {
    parser_group("B10").await;
}

#[tokio::test]
async fn b11() {
    parser_group("B11").await;
}

#[tokio::test]
async fn b12() {
    parser_group("B12").await;
}

#[tokio::test]
async fn b13() {
    parser_group("B13").await;
}

#[tokio::test]
async fn b14() {
    parser_group("B14").await;
}

#[tokio::test]
async fn b15() {
    parser_group("B15").await;
}

#[tokio::test]
async fn b16() {
    parser_group("B16").await;
}

#[tokio::test]
async fn b17() {
    parser_group("B17").await;
}

#[tokio::test]
async fn b18() {
    parser_group("B18").await;
}

#[tokio::test]
async fn b19() {
    parser_group("B19").await;
}

#[tokio::test]
async fn b20() {
    parser_group("B20").await;
}

#[tokio::test]
async fn c01() {
    parser_group("C01").await;
}

#[tokio::test]
async fn c02() {
    parser_group("C02").await;
}

#[tokio::test]
async fn c03() {
    parser_group("C03").await;
}

#[tokio::test]
async fn c04() {
    parser_group("C04").await;
}

#[tokio::test]
async fn c05() {
    parser_group("C05").await;
}

#[tokio::test]
async fn c06() {
    parser_group("C06").await;
}

#[tokio::test]
async fn c07() {
    parser_group("C07").await;
}

#[tokio::test]
async fn c08() {
    parser_group("C08").await;
}

#[tokio::test]
async fn c09() {
    parser_group("C09").await;
}

#[tokio::test]
async fn c10() {
    parser_group("C10").await;
}

#[tokio::test]
async fn c11() {
    parser_group("C11").await;
}

#[tokio::test]
async fn c12() {
    parser_group("C12").await;
}

#[tokio::test]
async fn c13() {
    parser_group("C13").await;
}

#[tokio::test]
async fn c14() {
    parser_group("C14").await;
}

#[tokio::test]
async fn c15() {
    parser_group("C15").await;
}

#[tokio::test]
async fn d01() {
    parser_group("D01").await;
}

#[tokio::test]
async fn d02() {
    parser_group("D02").await;
}

#[tokio::test]
async fn d03() {
    parser_group("D03").await;
}

#[tokio::test]
async fn d04() {
    parser_group("D04").await;
}

#[tokio::test]
async fn d05() {
    parser_group("D05").await;
}

#[tokio::test]
async fn d06() {
    parser_group("D06").await;
}

#[tokio::test]
async fn d07() {
    parser_group("D07").await;
}

#[tokio::test]
async fn d08() {
    parser_group("D08").await;
}

#[tokio::test]
async fn d09() {
    parser_group("D09").await;
}

#[tokio::test]
async fn d10() {
    parser_group("D10").await;
}

#[tokio::test]
async fn d11() {
    parser_group("D11").await;
}

#[tokio::test]
async fn d12() {
    parser_group("D12").await;
}

#[tokio::test]
async fn d13() {
    parser_group("D13").await;
}

#[tokio::test]
async fn e01() {
    parser_group("E01").await;
}

#[tokio::test]
async fn e02() {
    parser_group("E02").await;
}

#[tokio::test]
async fn e03() {
    parser_group("E03").await;
}

#[tokio::test]
async fn e04() {
    parser_group("E04").await;
}

#[tokio::test]
async fn e05() {
    parser_group("E05").await;
}

#[tokio::test]
async fn e06() {
    parser_group("E06").await;
}

#[tokio::test]
async fn e07() {
    parser_group("E07").await;
}

#[tokio::test]
async fn e08() {
    parser_group("E08").await;
}

#[tokio::test]
async fn e09() {
    parser_group("E09").await;
}

#[tokio::test]
async fn e10() {
    parser_group("E10").await;
}

#[tokio::test]
async fn e11() {
    parser_group("E11").await;
}

#[tokio::test]
async fn e12() {
    parser_group("E12").await;
}

#[tokio::test]
async fn f01() {
    parser_group("F01").await;
}

#[tokio::test]
async fn f02() {
    parser_group("F02").await;
}

#[tokio::test]
async fn f03() {
    parser_group("F03").await;
}

#[tokio::test]
async fn f04() {
    parser_group("F04").await;
}

#[tokio::test]
async fn f05() {
    parser_group("F05").await;
}

#[tokio::test]
async fn f06() {
    parser_group("F06").await;
}

#[tokio::test]
async fn f07() {
    parser_group("F07").await;
}

#[tokio::test]
async fn f08() {
    parser_group("F08").await;
}

#[tokio::test]
async fn f09() {
    parser_group("F09").await;
}

#[tokio::test]
async fn f10() {
    parser_group("F10").await;
}

#[tokio::test]
async fn f11() {
    parser_group("F11").await;
}

#[tokio::test]
async fn f12() {
    parser_group("F12").await;
}

#[test]
fn g01() {
    support_group("G01");
}

#[test]
fn g02() {
    support_group("G02");
}

#[test]
fn g03() {
    support_group("G03");
}

#[test]
fn g04() {
    support_group("G04");
}

#[test]
fn g05() {
    support_group("G05");
}

#[test]
fn g06() {
    support_group("G06");
}

#[test]
fn g07() {
    support_group("G07");
}

#[test]
fn g08() {
    support_group("G08");
}

#[test]
fn g09() {
    support_group("G09");
}

#[test]
fn g10() {
    support_group("G10");
}

#[test]
fn g11() {
    support_group("G11");
}

#[test]
fn g12() {
    support_group("G12");
}

#[test]
fn g13() {
    support_group("G13");
}

#[test]
fn g14() {
    support_group("G14");
}

#[test]
fn h01() {
    support_group("H01");
}

#[test]
fn h02() {
    support_group("H02");
}

#[test]
fn h03() {
    support_group("H03");
}

#[test]
fn h04() {
    support_group("H04");
}

#[test]
fn h05() {
    support_group("H05");
}

#[test]
fn h06() {
    support_group("H06");
}

#[test]
fn h07() {
    support_group("H07");
}

#[test]
fn h08() {
    support_group("H08");
}
