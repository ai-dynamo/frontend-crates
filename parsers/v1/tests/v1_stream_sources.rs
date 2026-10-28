// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Source recording through the v1 stream stages: the reasoning stream, the
//! leading-opener strip, and the jail, alone and as the v1 route.
//!
//! Each case of `tests/data/v1_stream_parity.jsonl` (dynamo-llm's recorded
//! streams) runs twice: on [`Annotated`] (the stock path, no slot) and on
//! [`SourcedChunk`] with the caller's ranges. Single-choice cases also run at
//! one character per chunk and at every two-way split of their content. Each
//! run checks:
//!
//! * Same output: the sourced run emits the chunks that the stock run emits.
//! * Verbatim and tiling: each output choice's ranges, read from that
//!   choice's input content, equal its `content` and its `reasoning_content`.
//! * Causality: no range goes past the bytes fed so far.

// A list with one range is the value meant here.
#![allow(clippy::single_range_in_vec_init)]

use std::collections::BTreeMap;
use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use dynamo_parsers::chat_stream::{ChatStreamItem, SourcedChunk};
use dynamo_parsers::reasoning::stream::{
    ReasoningStreamOptions, parse_reasoning_content_from_stream,
    strip_leading_reasoning_start_from_stream,
};
use dynamo_parsers::sources::{ChoiceSources, SourceSlot};
use dynamo_parsers::tool_calling::ToolDefinition;
use dynamo_parsers::tool_calling::jail::{
    Annotated, apply_jail_terminal_error_contract, apply_tool_calling_jail,
    apply_tool_calling_jail_sourced, tool_calling_jail_can_record_sources,
};
use dynamo_protocols::types::{ChatCompletionMessageContent, CreateChatCompletionStreamResponse};
use futures::StreamExt;
use futures::stream::{self, BoxStream};
use serde_json::{Value, json};

type Item = Annotated<CreateChatCompletionStreamResponse>;

const FIXTURE: &str = include_str!("data/v1_stream_parity.jsonl");

fn to_item(value: &Value) -> Item {
    let mut value = value.clone();
    if let Some(Value::Object(data)) = value.get_mut("data") {
        for (key, default) in [
            ("id", json!("chatcmpl-sources")),
            ("object", json!("chat.completion.chunk")),
            ("created", json!(1)),
            ("model", json!("sources-model")),
        ] {
            data.entry(key).or_insert(default);
        }
    }
    serde_json::from_value(value).expect("fixture item")
}

/// The stock output, with tool-call ids blanked (they are random).
fn shape(items: &[Item]) -> Vec<Value> {
    items
        .iter()
        .map(|item| {
            let mut value = serde_json::to_value(item).unwrap();
            if let Some(choices) = value["data"]["choices"].as_array_mut() {
                for choice in choices {
                    if let Some(calls) = choice["delta"]["tool_calls"].as_array_mut() {
                        for call in calls {
                            if call.get("id").is_some_and(|id| !id.is_null()) {
                                call["id"] = json!("id");
                            }
                        }
                    }
                }
            }
            value
        })
        .collect()
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

fn opener(stage: &Value) -> &'static str {
    match stage["think_start_token"].as_str().unwrap() {
        "<think>" => "<think>",
        "[THINK]" => "[THINK]",
        other => panic!("unknown opener {other}"),
    }
}

fn reasoning_parser(stage: &Value) -> String {
    stage["reasoning_parser"].as_str().unwrap().to_string()
}

fn tool_call_parser(stage: &Value) -> Option<String> {
    stage["tool_call_parser"].as_str().map(str::to_string)
}

async fn run_stock(stage: &Value, input: Vec<Item>) -> Vec<Item> {
    let input = stream::iter(input).boxed();
    let jail = |input: BoxStream<'static, Item>| {
        let parser = tool_call_parser(stage);
        apply_jail_terminal_error_contract(input, move |s| {
            apply_tool_calling_jail(parser, None, Some(tools()), false, s)
        })
        .boxed()
    };
    match stage["kind"].as_str().unwrap() {
        "reasoning" => {
            parse_reasoning_content_from_stream(input, reasoning_parser(stage), options(stage))
                .collect()
                .await
        }
        "v1_route" => {
            let reasoned =
                parse_reasoning_content_from_stream(input, reasoning_parser(stage), options(stage))
                    .boxed();
            jail(reasoned).collect().await
        }
        "strip" => {
            strip_leading_reasoning_start_from_stream(input, opener(stage))
                .collect()
                .await
        }
        "jail" => jail(input).collect().await,
        other => panic!("unknown stage {other}"),
    }
}

/// The sourced run, each output item paired with the number of input items
/// pulled when it was produced.
async fn run_sourced(stage: &Value, input: Vec<SourcedChunk>) -> Vec<(SourcedChunk, usize)> {
    let pulled = Arc::new(AtomicUsize::new(0));
    let counter = pulled.clone();
    let input = stream::iter(input)
        .inspect(move |_| {
            counter.fetch_add(1, Ordering::SeqCst);
        })
        .boxed();
    let jail = |input: BoxStream<'static, SourcedChunk>| {
        let parser = tool_call_parser(stage);
        apply_jail_terminal_error_contract(input, move |s| {
            apply_tool_calling_jail_sourced(parser, None, Some(tools()), false, s)
        })
        .boxed()
    };
    let output: BoxStream<'static, SourcedChunk> = match stage["kind"].as_str().unwrap() {
        "reasoning" => {
            parse_reasoning_content_from_stream(input, reasoning_parser(stage), options(stage))
                .boxed()
        }
        "v1_route" => jail(
            parse_reasoning_content_from_stream(input, reasoning_parser(stage), options(stage))
                .boxed(),
        ),
        "strip" => strip_leading_reasoning_start_from_stream(input, opener(stage)).boxed(),
        "jail" => jail(input),
        other => panic!("unknown stage {other}"),
    };
    output
        .map(|item| {
            let seen = pulled.load(Ordering::SeqCst);
            (item, seen)
        })
        .collect()
        .await
}

fn text_of(content: &Option<ChatCompletionMessageContent>) -> &str {
    match content {
        Some(ChatCompletionMessageContent::Text(text)) => text,
        _ => "",
    }
}

/// Gives each input item its slot: ranges over each choice's own content.
fn with_sources(input: &[Item]) -> Vec<SourcedChunk> {
    let mut offsets: BTreeMap<u32, usize> = BTreeMap::new();
    input
        .iter()
        .map(|item| {
            let sources = item.data.as_ref().map_or_else(Vec::new, |data| {
                data.choices
                    .iter()
                    .map(|choice| {
                        let offset = offsets.entry(choice.index).or_default();
                        let text = text_of(&choice.delta.content);
                        let start = *offset;
                        *offset += text.len();
                        ChoiceSources {
                            index: choice.index,
                            content: if text.is_empty() {
                                vec![]
                            } else {
                                vec![start..start + text.len()]
                            },
                            reasoning: vec![],
                        }
                    })
                    .collect()
            });
            SourcedChunk {
                chunk: item.clone(),
                sources: Some(sources),
            }
        })
        .collect()
}

fn read(input: &str, ranges: &[Range<usize>]) -> String {
    ranges.iter().map(|range| &input[range.clone()]).collect()
}

/// One input run: stock and sourced, compared and checked.
async fn check(stage: &Value, input: Vec<Item>, label: &str) {
    let stock = run_stock(stage, input.clone()).await;
    let sourced = run_sourced(stage, with_sources(&input)).await;
    let chunks: Vec<Item> = sourced.iter().map(|(item, _)| item.chunk.clone()).collect();
    assert_eq!(
        shape(&stock),
        shape(&chunks),
        "{label}: recording changed the output"
    );

    // Each choice's input content, and how much of it was fed after each item.
    let mut content_in: BTreeMap<u32, String> = BTreeMap::new();
    let mut fed_after: Vec<BTreeMap<u32, usize>> = Vec::new();
    for item in &input {
        if let Some(data) = &item.data {
            for choice in &data.choices {
                content_in
                    .entry(choice.index)
                    .or_default()
                    .push_str(text_of(&choice.delta.content));
            }
        }
        fed_after.push(
            content_in
                .iter()
                .map(|(index, text)| (*index, text.len()))
                .collect(),
        );
    }
    let records = tool_call_parser(stage)
        .is_none_or(|parser| tool_calling_jail_can_record_sources(Some(parser), None, false));
    for (item, pulled) in &sourced {
        let sources = item
            .sources()
            .unwrap_or_else(|| panic!("{label}: no slot on an output item"));
        let choices = item.chunk().map_or(&[][..], |data| &data.choices[..]);
        if !records {
            continue;
        }
        let fed = pulled
            .checked_sub(1)
            .and_then(|at| fed_after.get(at))
            .cloned()
            .unwrap_or_default();
        for choice in choices {
            let entry = sources
                .iter()
                .find(|entry| entry.index == choice.index)
                .unwrap_or_else(|| panic!("{label}: no source entry for choice {}", choice.index));
            let input = content_in.get(&choice.index).map_or("", String::as_str);
            assert_eq!(
                read(input, &entry.content),
                text_of(&choice.delta.content),
                "{label}: content does not tile"
            );
            assert_eq!(
                read(input, &entry.reasoning),
                choice.delta.reasoning_content.as_deref().unwrap_or(""),
                "{label}: reasoning does not tile"
            );
            let fed = fed.get(&choice.index).copied().unwrap_or(0);
            for range in entry.content.iter().chain(&entry.reasoning) {
                assert!(
                    range.end <= fed,
                    "{label}: {range:?} is after the {fed} bytes fed"
                );
            }
        }
    }
}

/// The same stream with its content cut differently: items before the first
/// content and after the last stay; the content items become `parts`.
fn resplit(input: &[Item], parts: &[&str]) -> Vec<Item> {
    let is_content = |item: &Item| {
        item.data.as_ref().is_some_and(|data| {
            data.choices
                .iter()
                .any(|choice| !text_of(&choice.delta.content).is_empty())
        })
    };
    let first = input.iter().position(is_content).unwrap();
    let last = input.iter().rposition(is_content).unwrap();
    let template = input[first].clone();
    let mut out: Vec<Item> = input[..first].to_vec();
    for part in parts.iter().filter(|part| !part.is_empty()) {
        let mut item = template.clone();
        let data = item.data.as_mut().unwrap();
        data.choices[0].delta.content = Some(ChatCompletionMessageContent::Text(part.to_string()));
        data.choices[0].finish_reason = None;
        out.push(item);
    }
    out.extend_from_slice(&input[last + 1..]);
    out
}

fn single_choice_text(input: &[Item]) -> Option<String> {
    let mut text = String::new();
    for item in input {
        let Some(data) = &item.data else { continue };
        if data.choices.len() > 1 || data.choices.iter().any(|choice| choice.index != 0) {
            return None;
        }
        for choice in &data.choices {
            // The finish chunk must carry no text, so the resplit keeps it whole.
            if choice.finish_reason.is_some() && !text_of(&choice.delta.content).is_empty() {
                return None;
            }
            if matches!(
                choice.delta.content,
                Some(ChatCompletionMessageContent::Parts(_))
            ) {
                return None;
            }
            text.push_str(text_of(&choice.delta.content));
        }
    }
    (!text.is_empty()).then_some(text)
}

#[tokio::test]
async fn every_recorded_stream_at_every_split() {
    let mut runs = 0;
    let mut cases = 0;
    for line in FIXTURE.lines() {
        let case: Value = serde_json::from_str(line).unwrap();
        let stage = &case["stage"];
        let name = case["name"].as_str().unwrap();
        let input: Vec<Item> = case["input"]
            .as_array()
            .unwrap()
            .iter()
            .map(to_item)
            .collect();
        check(stage, input.clone(), name).await;
        runs += 1;
        cases += 1;
        let Some(text) = single_choice_text(&input) else {
            continue;
        };
        let chars: Vec<String> = text.chars().map(String::from).collect();
        let chars: Vec<&str> = chars.iter().map(String::as_str).collect();
        check(stage, resplit(&input, &chars), &format!("{name} by char")).await;
        runs += 1;
        for at in (1..text.len()).filter(|at| text.is_char_boundary(*at)) {
            let label = format!("{name} split at {at}");
            check(stage, resplit(&input, &[&text[..at], &text[at..]]), &label).await;
            runs += 1;
        }
    }
    assert!(cases >= 521, "fixture shrank: {cases}");
    println!("{cases} recorded streams, {runs} runs per recording mode");
}

/// Items for one choice: a role chunk, `chunks` of content, and a stop chunk.
fn chat(chunks: &[&str]) -> Vec<Item> {
    let mut items = vec![to_item(
        &json!({"data": {"choices": [{"index": 0, "delta": {"role": "assistant"}}]}}),
    )];
    for chunk in chunks {
        items.push(to_item(
            &json!({"data": {"choices": [{"index": 0, "delta": {"content": chunk}}]}}),
        ));
    }
    items.push(to_item(
        &json!({"data": {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}}),
    ));
    items
}

/// The input offset of each emitted content byte of choice 0, in output order.
fn content_offsets(output: &[(SourcedChunk, usize)]) -> (String, Vec<usize>) {
    let mut text = String::new();
    let mut offsets = Vec::new();
    for (item, _) in output {
        let Some(data) = item.chunk() else { continue };
        for choice in &data.choices {
            let content = text_of(&choice.delta.content);
            if content.is_empty() {
                continue;
            }
            text.push_str(content);
            let entry = item
                .sources()
                .and_then(|all| all.iter().find(|entry| entry.index == choice.index))
                .expect("a source entry");
            offsets.extend(entry.content.iter().flat_map(|range| range.clone()));
        }
    }
    (text, offsets)
}

#[tokio::test]
async fn a_stop_string_in_the_answer_is_dated_by_its_own_bytes() {
    // The v1 route for Qwen3 (`qwen3` + `hermes`), one token-like chunk at a
    // time. The stop string also appears in the reasoning, so byte equality
    // alone cannot pick the right occurrence.
    let chunks = [
        "<think>", "STOP", " or not", "</think>", "The", " answer", " is", " STOP", " here.",
    ];
    let raw: String = chunks.concat();
    let stage = json!({"kind": "v1_route", "reasoning_parser": "qwen3", "tool_call_parser": "hermes", "options": {}});
    let input = chat(&chunks);
    check(&stage, input.clone(), "stop string").await;
    let output = run_sourced(&stage, with_sources(&input)).await;
    let (text, offsets) = content_offsets(&output);
    assert_eq!(text, "The answer is STOP here.");
    let at = text.find("STOP").unwrap();
    let source = raw.rfind("STOP").unwrap();
    assert_eq!(
        &offsets[at..at + 4],
        &[source, source + 1, source + 2, source + 3]
    );
}

#[tokio::test]
async fn text_without_ranges_gets_no_entry() {
    // The caller gives the second content chunk no ranges. From then on that
    // choice's text has no source, and the stages say so with no entry.
    let stage = json!({"kind": "v1_route", "reasoning_parser": "qwen3", "tool_call_parser": "hermes", "options": {}});
    let input = chat(&["<think>r</think>", "an", "swer"]);
    let mut sourced = with_sources(&input);
    sourced[2].sources = Some(vec![]);
    let output = run_sourced(&stage, sourced).await;
    let mut entries_after = 0;
    let mut texts_after = 0;
    let mut seen_gap = false;
    for (item, pulled) in &output {
        if *pulled >= 3 {
            seen_gap = true;
        }
        if !seen_gap {
            continue;
        }
        let Some(data) = item.chunk() else { continue };
        for choice in &data.choices {
            if !text_of(&choice.delta.content).is_empty() {
                texts_after += 1;
                entries_after += item
                    .sources()
                    .unwrap()
                    .iter()
                    .filter(|entry| entry.index == choice.index)
                    .count();
            }
        }
    }
    assert!(texts_after > 0);
    assert_eq!(entries_after, 0);
}

#[tokio::test]
async fn a_parser_that_cannot_record_gives_no_entries() {
    // GPT-OSS's v1 reasoning parser does not record.
    let stage = json!({"kind": "reasoning", "reasoning_parser": "gpt_oss", "options": {}});
    let input = chat(&["<|channel|>final<|message|>hi"]);
    let output = run_sourced(&stage, with_sources(&input)).await;
    for (item, _) in &output {
        let sources = item.sources().expect("a slot");
        let data = item.chunk().unwrap();
        for choice in &data.choices {
            if !text_of(&choice.delta.content).is_empty()
                || choice.delta.reasoning_content.is_some()
            {
                assert!(sources.iter().all(|entry| entry.index != choice.index));
            }
        }
    }
}

#[test]
fn the_jail_records_only_for_the_three_marker_parsers() {
    for parser in ["hermes", "kimi_k2", "minimax_m3"] {
        assert!(tool_calling_jail_can_record_sources(
            Some(parser.into()),
            None,
            false
        ));
    }
    assert!(!tool_calling_jail_can_record_sources(
        Some("harmony".into()),
        None,
        false
    ));
    let required = Some(dynamo_protocols::types::ChatCompletionToolChoiceOption::Required);
    assert!(!tool_calling_jail_can_record_sources(
        Some("hermes".into()),
        required,
        false
    ));
}
