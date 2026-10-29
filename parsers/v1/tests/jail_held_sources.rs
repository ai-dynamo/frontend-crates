// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The jail reports the input ranges of the text it still holds.
//!
//! While it records, each output item of the sourced jail carries `held`:
//! for each choice that still holds text when the item goes out, the input
//! ranges of that text. A caller that cuts text after the jail, such as a
//! stop string on the answer, needs it. The jail can hold answer text, such as
//! a possible tool-call opener, while later reasoning passes, so the
//! reasoning's own ranges do not tell the caller that earlier text is still to
//! come.

// A list with one range is the value meant here.
#![allow(clippy::single_range_in_vec_init)]

use dynamo_parsers::chat_stream::SourcedChunk;
use dynamo_parsers::sources::ChoiceSources;
use dynamo_parsers::tool_calling::ToolDefinition;
use dynamo_parsers::tool_calling::jail::{
    Annotated, apply_tool_calling_jail, apply_tool_calling_jail_sourced,
};
use dynamo_protocols::types::CreateChatCompletionStreamResponse;
use futures::StreamExt;
use futures::stream;
use serde_json::{Value, json};

type Item = Annotated<CreateChatCompletionStreamResponse>;

fn item(delta: Value, finish: Option<&str>) -> Item {
    serde_json::from_value(json!({"data": {
        "id": "chatcmpl-held",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "held-model",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }}))
    .expect("item")
}

fn tools() -> Option<Vec<ToolDefinition>> {
    Some(vec![ToolDefinition {
        name: "write_file".into(),
        parameters: Some(json!({
            "type": "object",
            "properties": {"text": {"type": "string"}}
        })),
        strict: None,
    }])
}

fn sources(
    content: Vec<std::ops::Range<usize>>,
    reasoning: Vec<std::ops::Range<usize>>,
) -> Vec<ChoiceSources> {
    vec![ChoiceSources {
        index: 0,
        content,
        reasoning,
    }]
}

/// The reply `a<tool<think>r`, as the reasoning stream hands it to the jail,
/// one token per item: `a`, `<`, `tool`, then the reasoning `r` (the marker's
/// bytes 6..13 went to the reasoning stream), then the finish.
fn reply() -> Vec<SourcedChunk> {
    vec![
        SourcedChunk {
            chunk: item(json!({"role": "assistant", "content": "a"}), None),
            sources: Some(sources(vec![0..1], vec![])),
            held: None,
        },
        SourcedChunk {
            chunk: item(json!({"content": "<"}), None),
            sources: Some(sources(vec![1..2], vec![])),
            held: None,
        },
        SourcedChunk {
            chunk: item(json!({"content": "tool"}), None),
            sources: Some(sources(vec![2..6], vec![])),
            held: None,
        },
        SourcedChunk {
            chunk: item(json!({"reasoning_content": "r"}), None),
            sources: Some(sources(vec![], vec![13..14])),
            held: None,
        },
        SourcedChunk {
            chunk: item(json!({}), Some("stop")),
            sources: Some(vec![]),
            held: None,
        },
    ]
}

async fn run(input: Vec<SourcedChunk>) -> Vec<SourcedChunk> {
    apply_tool_calling_jail_sourced(
        Some("hermes".into()),
        None,
        tools(),
        false,
        stream::iter(input),
    )
    .collect()
    .await
}

fn text(item: &SourcedChunk, field: &str) -> String {
    let value = serde_json::to_value(&item.chunk).expect("json");
    value["data"]["choices"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|choice| choice["delta"][field].as_str())
        .collect()
}

/// The item that lets the reasoning `r` through reports `<tool`, bytes 1..6,
/// as held answer text; the item that finally releases `<tool` reports
/// nothing held.
#[tokio::test]
async fn reasoning_past_held_answer_text_reports_the_held_text() {
    let output = run(reply()).await;
    let reasoning = output
        .iter()
        .find(|item| text(item, "reasoning_content") == "r")
        .expect("the reasoning goes out");
    assert_eq!(
        reasoning.held,
        Some(sources(vec![1..6], vec![])),
        "{output:#?}"
    );
    let released = output
        .iter()
        .find(|item| text(item, "content").contains("<tool"))
        .expect("the held text goes out at the end");
    assert_eq!(released.held, Some(vec![]), "{output:#?}");
    // The first item, `a`, goes out with nothing held.
    assert_eq!(text(&output[0], "content"), "a");
    assert_eq!(output[0].held, Some(vec![]));
}

/// Within one input item the jail can emit several items. An earlier one
/// still reports the text that a later one of the same input releases: it
/// was held when the earlier item went out.
#[tokio::test]
async fn an_item_reports_the_text_released_after_it_in_the_same_input() {
    let mut input = reply();
    // The reasoning and the finish in one input item.
    input.truncate(3);
    input.push(SourcedChunk {
        chunk: item(json!({"reasoning_content": "r"}), Some("stop")),
        sources: Some(sources(vec![], vec![13..14])),
        held: None,
    });
    let output = run(input).await;
    let at = |predicate: &dyn Fn(&SourcedChunk) -> bool| {
        output.iter().position(predicate).expect("item")
    };
    let reasoning = at(&|item| text(item, "reasoning_content") == "r");
    let released = at(&|item| text(item, "content").contains("<tool"));
    assert!(reasoning < released, "{output:#?}");
    let held = output[reasoning].held.as_ref().expect("held");
    assert!(
        held.iter().any(|choice| choice.content.contains(&(1..6))),
        "{output:#?}"
    );
    assert_eq!(output.last().expect("item").held, Some(vec![]));
}

/// Recording off, the jail reports nothing: the stock path is the same, and
/// a parser that cannot record gives `None`.
#[tokio::test]
async fn a_jail_that_does_not_record_reports_nothing() {
    let plain: Vec<Item> = apply_tool_calling_jail(
        Some("hermes".into()),
        None,
        tools(),
        false,
        stream::iter(reply().into_iter().map(|item| item.chunk)),
    )
    .collect()
    .await;
    let sourced = run(reply()).await;
    let chunks: Vec<_> = sourced
        .iter()
        .map(|item| serde_json::to_value(&item.chunk).expect("json"))
        .collect();
    let plain: Vec<_> = plain
        .iter()
        .map(|item| serde_json::to_value(item).expect("json"))
        .collect();
    assert_eq!(chunks, plain, "recording changed the output");

    let unrecorded: Vec<SourcedChunk> = apply_tool_calling_jail_sourced(
        Some("pythonic".into()),
        None,
        tools(),
        false,
        stream::iter(reply()),
    )
    .collect()
    .await;
    assert!(
        unrecorded.iter().all(|item| item.held.is_none()),
        "{unrecorded:#?}"
    );
}

#[tokio::test]
async fn review_f1_a_source_gap_does_not_invent_held_input_ranges() {
    let mut input = reply();
    for item in &mut input {
        for source in item.sources.as_mut().unwrap() {
            for range in source.content.iter_mut().chain(&mut source.reasoning) {
                range.start += 100;
                range.end += 100;
            }
        }
    }
    let control = run(input.clone()).await;
    let reasoning = control
        .iter()
        .find(|item| text(item, "reasoning_content") == "r")
        .unwrap();
    assert_eq!(reasoning.held, Some(sources(vec![101..106], vec![])));

    input[1].sources = None;
    let output = run(input).await;
    let reasoning = output
        .iter()
        .find(|item| text(item, "reasoning_content") == "r")
        .unwrap();
    assert_eq!(reasoning.sources, Some(vec![]));
    assert!(
        reasoning
            .held
            .as_ref()
            .unwrap()
            .iter()
            .flat_map(|entry| entry.content.iter().chain(&entry.reasoning))
            .all(|range| range.start >= 100),
        "held reports ranges outside the caller's input: {:?}",
        reasoning.held
    );
}
