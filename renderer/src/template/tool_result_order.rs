// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Rendered tool-result order for chat templates that reorder tool results.
//!
//! GLM-5.3-Flash's template puts each run of tool results into call order.
//! Media collection must follow the same order, or images in out-of-order
//! parallel results fill other results' placeholders.

use std::collections::HashSet;

use serde_json::Value;

use crate::OAIChatLikeRequest;

/// Whether a template contains GLM-5.3-Flash's tool-result sort block.
pub(super) fn sorts_tool_results(source: &str) -> bool {
    source.contains("ns_chk.can_sort") && source.contains("tc_id_exists")
}

enum Message<'a> {
    Assistant(Vec<Option<&'a str>>),
    Tool { id: Option<&'a str>, packed: bool },
    Other,
}

/// Source message indices in GLM's rendered order. Reads roles and IDs only,
/// so typed requests copy no image data.
pub(super) fn media_order(req: &dyn OAIChatLikeRequest) -> Option<Vec<usize>> {
    use dynamo_protocols::types::ChatCompletionRequestMessage as Typed;
    if let Some(messages) = req.typed_messages() {
        if !messages
            .iter()
            .any(|message| matches!(message, Typed::Tool(_)))
        {
            return None;
        }
        return Some(order(
            &messages
                .iter()
                .map(|message| match message {
                    Typed::Assistant(assistant) => Message::Assistant(
                        assistant
                            .tool_calls
                            .iter()
                            .flatten()
                            .map(|call| non_empty(&call.id))
                            .collect(),
                    ),
                    Typed::Tool(tool) => Message::Tool {
                        id: non_empty(&tool.tool_call_id),
                        packed: false,
                    },
                    _ => Message::Other,
                })
                .collect::<Vec<_>>(),
        ));
    }
    let messages = crate::messages_to_json(req).ok()?;
    let messages = messages.as_array()?;
    if !messages.iter().any(|message| message["role"] == "tool") {
        return None;
    }
    Some(order(
        &messages
            .iter()
            .map(|message| match message["role"].as_str() {
                Some("assistant") => Message::Assistant(
                    message["tool_calls"]
                        .as_array()
                        .into_iter()
                        .flatten()
                        .map(raw_id)
                        .collect(),
                ),
                Some("tool") => Message::Tool {
                    id: raw_id(message),
                    // The template's `is_list_of_outputs`.
                    packed: message["content"]
                        .get(0)
                        .is_some_and(|entry| entry.get("output").is_some()),
                },
                _ => Message::Other,
            })
            .collect::<Vec<_>>(),
    ))
}

fn non_empty(id: &str) -> Option<&str> {
    (!id.is_empty()).then_some(id)
}

/// The template's `id_of`: `tool_call_id`, else `id`, when non-empty.
fn raw_id(value: &Value) -> Option<&str> {
    ["tool_call_id", "id"]
        .into_iter()
        .find_map(|key| value[key].as_str().and_then(non_empty))
}

fn order(messages: &[Message]) -> Vec<usize> {
    let mut output = Vec::with_capacity(messages.len());
    let mut start = 0;
    while start < messages.len() {
        let is_tool = |index: &usize| matches!(messages[*index], Message::Tool { .. });
        if !is_tool(&start) {
            output.push(start);
            start += 1;
            continue;
        }
        let end = (start..messages.len())
            .find(|index| !is_tool(index))
            .unwrap_or(messages.len());
        let mut run: Vec<usize> = (start..end).collect();
        if let Some(keys) = call_positions(messages, start, end) {
            run.sort_by_key(|&index| keys[index - start]);
        }
        output.extend(run);
        start = end;
    }
    output
}

/// Call position of each result in `messages[start..end]`, or `None` when the
/// template's `can_sort` check fails and the run keeps arrival order.
fn call_positions(messages: &[Message], start: usize, end: usize) -> Option<Vec<usize>> {
    // The run must directly follow the assistant message with the calls.
    let Message::Assistant(calls) = &messages[start.checked_sub(1)?] else {
        return None;
    };
    let calls: Vec<&str> = calls.iter().copied().collect::<Option<_>>()?;
    let mut unique = HashSet::new();
    if calls.is_empty() || !calls.iter().all(|id| unique.insert(id)) {
        return None;
    }
    let mut seen = HashSet::new();
    messages[start..end]
        .iter()
        .map(|message| match message {
            // ponytail: packed `{id, output}` results sort inside one message,
            // which a message order can't express; keep arrival order.
            Message::Tool { packed: true, .. } => None,
            Message::Tool { id: Some(id), .. } if seen.insert(*id) => {
                calls.iter().position(|call| call == id)
            }
            _ => None,
        })
        .collect()
}
