// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;

use dynamo_parsers::reasoning::ReasoningParser;
use dynamo_parsers::tool_calling::jail::{Annotated, JailedStream};
use dynamo_parsers::tool_calling::ToolDefinition;
use dynamo_protocols::types::CreateChatCompletionStreamResponse;
use futures::StreamExt;
use serde_json::{json, Value};

fn chunk(
    content: &str,
    reasoning: &str,
    finish: Option<&str>,
) -> Annotated<CreateChatCompletionStreamResponse> {
    Annotated {
        data: Some(serde_json::from_value(json!({
            "id":"capture", "created":0, "model":"capture", "object":"chat.completion.chunk",
            "choices":[{"index":0,"delta":{"role":"assistant","content":content,"reasoning_content":reasoning},"finish_reason":finish}]
        })).expect("valid capture chunk")),
        id: None, event: None, comment: None, error: None,
    }
}

pub fn text_chunks(request: &Value) -> Result<Vec<&str>, String> {
    let rows = request["chunks"]
        .as_array()
        .ok_or("expected request chunks")?;
    let mut texts: Vec<&str> = rows
        .iter()
        .map(|row| {
            row["delta_text"]
                .as_str()
                .ok_or("expected literal chunk text")
        })
        .collect::<Result<_, _>>()?;
    let input = request["input"].as_str().ok_or("expected input")?;
    // A terminal-step row is a finish operation, never literal model text.
    if texts.last() == Some(&"‹finish›") && texts[..texts.len() - 1].concat() == input {
        texts.pop();
    }
    if texts.concat() != input
        || rows
            .iter()
            .any(|row| row.get("token_ids").is_some() || row.get("finish_reason").is_some())
    {
        return Err("jail capture requires an exact text chunk schedule".into());
    }
    Ok(texts)
}

pub async fn capture(
    mut parser: impl ReasoningParser,
    tool_name: &str,
    tools: Vec<ToolDefinition>,
    request: Value,
) -> Result<Value, String> {
    let texts = text_chunks(&request)?;
    let mut inputs = Vec::new();
    for text in texts {
        let delta = parser.parse_reasoning_streaming_incremental(text, &[]);
        inputs.push(chunk(&delta.normal_text, &delta.reasoning_text, None));
    }
    let tail = parser.finish_reasoning_stream();
    inputs.push(chunk(&tail.normal_text, &tail.reasoning_text, None));
    inputs.push(chunk("", "", request["finish_reason"].as_str()));
    let jail = JailedStream::builder()
        .tool_call_parser(tool_name)
        .tool_definitions(tools)
        .build();
    let output: Vec<_> = jail.apply(futures::stream::iter(inputs)).collect().await;
    let mut normal_text = String::new();
    let mut reasoning_text = String::new();
    let mut calls: BTreeMap<u32, (String, String)> = BTreeMap::new();
    for annotated in output {
        if let Some(error) = annotated.error {
            return Err(format!("jail error: {error:?}"));
        }
        let Some(response) = annotated.data else {
            continue;
        };
        // Protocol serialization keeps this adapter on the release's actual wire fields.
        let response = serde_json::to_value(response).map_err(|error| error.to_string())?;
        for choice in response["choices"]
            .as_array()
            .ok_or("missing jail choices")?
        {
            let delta = &choice["delta"];
            normal_text.push_str(delta["content"].as_str().unwrap_or_default());
            reasoning_text.push_str(delta["reasoning_content"].as_str().unwrap_or_default());
            if let Some(deltas) = delta["tool_calls"].as_array() {
                for call in deltas {
                    let index = call["index"].as_u64().ok_or("missing tool index")? as u32;
                    let (name, arguments) = calls.entry(index).or_default();
                    name.push_str(call["function"]["name"].as_str().unwrap_or_default());
                    arguments.push_str(call["function"]["arguments"].as_str().unwrap_or_default());
                }
            }
        }
    }
    Ok(
        json!({"reasoning_text":reasoning_text,"normal_text":normal_text,
        "calls":calls.into_values().map(|(name, arguments)| json!({"name":name,"arguments":serde_json::from_str::<Value>(&arguments).unwrap_or(Value::String(arguments))})).collect::<Vec<_>>()}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_parsers::reasoning::ReasoningParserType;

    #[test]
    fn terminal_schedule_is_an_operation_and_invalid_schedules_are_rejected() {
        let mut request = json!({"input":"abc","chunks":[{"delta_text":"a"},{"delta_text":"bc"},{"delta_text":"‹finish›"}]});
        assert_eq!(text_chunks(&request).unwrap(), vec!["a", "bc"]);
        request["chunks"][0]["token_ids"] = json!([1]);
        assert!(text_chunks(&request).is_err());
        request["chunks"] = json!([{"delta_text":"different"}]);
        assert!(text_chunks(&request).is_err());
    }

    #[tokio::test]
    async fn jail_preserves_final_call_order() {
        let text = "<think>work</think><tool_call>{\"name\":\"first\",\"arguments\":{}}</tool_call><tool_call>{\"name\":\"second\",\"arguments\":{}}</tool_call>";
        let request = json!({"input":text,"chunks":[{"delta_text":text}],"finish_reason":"stop"});
        let parser = ReasoningParserType::get_reasoning_parser_from_name("qwen3");
        let result = capture(parser, "hermes", vec![], request).await.unwrap();
        assert_eq!(result["reasoning_text"], "work");
        assert_eq!(result["calls"][0]["name"], "first");
        assert_eq!(result["calls"][1]["name"], "second");
    }

    #[tokio::test]
    async fn content_survives_every_split_and_length_termination() {
        let text = "answer";
        for split in 0..=text.len() {
            let request = json!({"input":text,"chunks":[{"delta_text":&text[..split]},{"delta_text":&text[split..]}],"finish_reason":"length"});
            let parser = ReasoningParserType::get_reasoning_parser_from_name("qwen3");
            let result = capture(parser, "qwen3_coder", vec![], request)
                .await
                .unwrap();
            assert_eq!(result["normal_text"], "answer", "split {split}");
            assert_eq!(result["reasoning_text"], "", "split {split}");
        }
    }
}
