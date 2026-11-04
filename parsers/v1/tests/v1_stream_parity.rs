// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Parity with dynamo-llm's v1 stream stages.
//!
//! `tests/data/v1_stream_parity.jsonl` holds input streams and the output that
//! dynamo-llm's own code gave for them, before these stages moved here. It was
//! recorded at ai-dynamo/dynamo `45e30e07a4f4cef37ab7bfa739f7bee0c89b3057`
//! (dynamo-parsers 10.0.7). Each line is one case:
//!
//! * `reasoning`: the reasoning stream, for each of the three v1 families
//!   (Kimi-Linear `basic`, MiniMax `minimax_m3`, Qwen3 `qwen3`) and each mix of
//!   options.
//! * `strip`: the leading-opener strip.
//! * `jail`: the jail under the terminal-error contract, with each family's
//!   tool parser (`kimi_k2`, `minimax_m3`, `hermes`).
//! * `v1_route`: the reasoning stream and then the jail, as dynamo runs them
//!   for these families.
//!
//! Chunks in the file leave out the fixed envelope (`id`, `object`, `created`,
//! `model`) and null fields. Generated tool-call IDs are written as `call-*`.
//! The test puts the envelope back on the input, and removes it from the
//! output the same way before it compares.

use dynamo_parsers::reasoning::stream::{
    ReasoningStreamOptions, parse_reasoning_content_from_stream,
    strip_leading_reasoning_start_from_stream,
};
use dynamo_parsers::tool_calling::ToolDefinition;
use dynamo_parsers::tool_calling::jail::{
    Annotated, apply_jail_terminal_error_contract, apply_tool_calling_jail,
};
use dynamo_protocols::types::CreateChatCompletionStreamResponse;
use futures::StreamExt;
use futures::stream::{self, BoxStream};
use serde_json::{Map, Value, json};

type Item = Annotated<CreateChatCompletionStreamResponse>;

const FIXTURE: &str = include_str!("data/v1_stream_parity.jsonl");

fn envelope() -> Map<String, Value> {
    let Value::Object(map) = json!({
        "id": "chatcmpl-parity",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "parity-model",
    }) else {
        unreachable!()
    };
    map
}

fn to_item(value: &Value) -> Item {
    let mut value = value.clone();
    if let Some(Value::Object(data)) = value.get_mut("data") {
        for (key, default) in envelope() {
            data.entry(key).or_insert(default);
        }
    }
    serde_json::from_value(value).expect("fixture item")
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

fn to_compact(item: Item) -> Value {
    let mut value = strip_nulls(serde_json::to_value(item).expect("serialize item"));
    if let Some(Value::Object(data)) = value.get_mut("data") {
        for (key, default) in envelope() {
            if data.get(&key) == Some(&default) {
                data.remove(&key);
            }
        }
        if let Some(Value::Array(choices)) = data.get_mut("choices") {
            for choice in choices {
                let Some(Value::Array(calls)) = choice.pointer_mut("/delta/tool_calls") else {
                    continue;
                };
                for call in calls {
                    if call
                        .get("id")
                        .and_then(Value::as_str)
                        .is_some_and(|id| id.starts_with("call-"))
                    {
                        call["id"] = json!("call-*");
                    }
                }
            }
        }
    }
    value
}

fn tools() -> Vec<ToolDefinition> {
    vec![ToolDefinition {
        name: "get_weather".into(),
        parameters: Some(json!({
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"]
        })),
        strict: None,
    }]
}

fn options(stage: &Value) -> ReasoningStreamOptions {
    let flag = |key: &str| stage["options"][key].as_bool().unwrap_or(false);
    ReasoningStreamOptions {
        prompt_injected_reasoning: flag("prompt_injected_reasoning"),
        bypass_bare_guided_json: flag("bypass_bare_guided_json"),
        defer_reasoning_for_nonempty_content: flag("defer_reasoning_for_nonempty_content"),
    }
}

fn jail(stage: &Value, input: BoxStream<'static, Item>) -> BoxStream<'static, Item> {
    let tool_call_parser = stage["tool_call_parser"].as_str().map(str::to_string);
    apply_jail_terminal_error_contract(input, move |s| {
        apply_tool_calling_jail(tool_call_parser, None, Some(tools()), false, s)
    })
    .boxed()
}

async fn run(stage: &Value, input: Vec<Item>) -> Vec<Item> {
    let input = stream::iter(input).boxed();
    let reasoning_parser = || stage["reasoning_parser"].as_str().unwrap().to_string();
    match stage["kind"].as_str().unwrap() {
        "reasoning" => {
            parse_reasoning_content_from_stream(input, reasoning_parser(), options(stage))
                .collect()
                .await
        }
        "v1_route" => {
            let reasoned =
                parse_reasoning_content_from_stream(input, reasoning_parser(), options(stage))
                    .boxed();
            jail(stage, reasoned).collect().await
        }
        "strip" => {
            let token = match stage["think_start_token"].as_str().unwrap() {
                "<think>" => "<think>",
                "[THINK]" => "[THINK]",
                other => panic!("unknown opener {other}"),
            };
            strip_leading_reasoning_start_from_stream(input, token)
                .collect()
                .await
        }
        "jail" => jail(stage, input).collect().await,
        other => panic!("unknown stage {other}"),
    }
}

#[tokio::test]
async fn v1_stream_stages_match_dynamo_llm() {
    let mut cases = 0;
    let mut failures = Vec::new();
    let mut families = std::collections::BTreeSet::new();
    for line in FIXTURE.lines().filter(|l| !l.trim().is_empty()) {
        let case: Value = serde_json::from_str(line).expect("fixture line");
        let name = case["name"].as_str().unwrap().to_string();
        if let Some(family) = name.split('/').nth(1) {
            families.insert(family.to_string());
        }
        let input = case["input"]
            .as_array()
            .unwrap()
            .iter()
            .map(to_item)
            .collect();
        let output: Vec<Value> = run(&case["stage"], input)
            .await
            .into_iter()
            .map(to_compact)
            .collect();
        if Value::Array(output.clone()) != case["output"] {
            failures.push(format!(
                "{name}\n  expected: {}\n  actual:   {}",
                case["output"],
                Value::Array(output)
            ));
        }
        cases += 1;
    }
    assert!(cases > 400, "fixture has only {cases} cases");
    for family in ["kimi_linear", "minimax", "qwen3"] {
        assert!(families.contains(family), "fixture lacks family {family}");
    }
    assert!(
        failures.is_empty(),
        "{} of {cases} cases differ from dynamo-llm:\n{}",
        failures.len(),
        failures.join("\n")
    );
}
