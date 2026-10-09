// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Replay the shared DSML stream fixtures through v1's jailed stream. Its output
//! chunks coalesce input chunks, so compare the actual recorded output schedule.
use dynamo_parsers::tool_calling::ToolDefinition;
use dynamo_parsers::tool_calling::jail::{Annotated, JailedStream};
use dynamo_protocols::types::{
    ChatChoiceStream, ChatCompletionMessageContent, ChatCompletionStreamResponseDelta,
    CreateChatCompletionStreamResponse, FinishReason, Role,
};
use futures::StreamExt;
use serde_json::{Value, json};

mod common;

fn input_chunk(content: Option<String>) -> Annotated<CreateChatCompletionStreamResponse> {
    let finish_reason = content.is_none().then_some(FinishReason::Stop);
    #[allow(deprecated)]
    let choice = ChatChoiceStream {
        index: 0,
        delta: ChatCompletionStreamResponseDelta {
            role: Some(Role::Assistant),
            content: content.map(ChatCompletionMessageContent::Text),
            tool_calls: None,
            function_call: None,
            refusal: None,
            reasoning_content: None,
        },
        finish_reason,
        logprobs: None,
    };
    Annotated {
        data: Some(CreateChatCompletionStreamResponse {
            id: "fixture".into(),
            choices: vec![choice],
            created: 0,
            model: "fixture".into(),
            system_fingerprint: None,
            object: "chat.completion.chunk".into(),
            usage: None,
            service_tier: None,
        }),
        id: None,
        event: None,
        comment: None,
        error: None,
    }
}

fn canonical_arguments(value: &mut Value) {
    if let Some(raw) = value.get("arguments").and_then(Value::as_str)
        && let Ok(arguments) = serde_json::from_str::<Value>(raw)
    {
        value["arguments"] = Value::String(arguments.to_string());
    }
}

#[tokio::test]
async fn dsml_jail_matches_shared_stream_capture() {
    let root = common::ensure_fixtures().join("toolcalling/fixtures-stream-v1");
    let mut checked = 0;
    for family in ["deepseek_v3_2", "deepseek_v4"] {
        let rel = format!("{family}/TOOLCALLING.streamv1.dsml.yaml");
        let input: Value =
            serde_yaml::from_str(&std::fs::read_to_string(root.join("inputs").join(&rel)).unwrap())
                .unwrap();
        let capture: Value = serde_yaml::from_str(
            &std::fs::read_to_string(root.join("dynamo_v1-10.0.2").join(&rel)).unwrap(),
        )
        .unwrap();
        for (id, case) in input["cases"].as_object().unwrap() {
            let tools = case["tools"]
                .as_array()
                .unwrap()
                .iter()
                .map(|tool| ToolDefinition {
                    name: tool["name"].as_str().unwrap().into(),
                    parameters: tool.get("parameters").cloned(),
                    strict: tool.get("strict").and_then(Value::as_bool),
                })
                .collect();
            let jail = JailedStream::builder()
                .tool_call_parser(family)
                .tool_definitions(tools)
                .build();
            let mut chunks: Vec<_> = case["chunks"]
                .as_array()
                .unwrap()
                .iter()
                .map(|chunk| input_chunk(Some(chunk["delta_text"].as_str().unwrap().into())))
                .collect();
            chunks.push(input_chunk(None));
            let outputs: Vec<_> = jail.apply(futures::stream::iter(chunks)).collect().await;
            let mut observed = Vec::new();
            for output in outputs {
                let Some(response) = output.data else {
                    continue;
                };
                let Some(choice) = response.choices.into_iter().next() else {
                    continue;
                };
                let mut entry = json!({"expected": []});
                for call in choice.delta.tool_calls.unwrap_or_default() {
                    let mut delta = json!({"index": call.index});
                    if call.id.is_some() {
                        delta["id"] = json!(true);
                    }
                    if let Some(function) = call.function {
                        if let Some(name) = function.name {
                            delta["name"] = json!(name);
                        }
                        if let Some(arguments) = function.arguments {
                            delta["arguments"] = json!(arguments);
                        }
                    }
                    canonical_arguments(&mut delta);
                    entry["expected"].as_array_mut().unwrap().push(delta);
                }
                if let Some(ChatCompletionMessageContent::Text(text)) = choice.delta.content
                    && !text.is_empty()
                {
                    entry["normal_text"] = json!(text);
                }
                observed.push(entry);
            }
            assert_eq!(
                json!(observed),
                capture["cases"][id]["chunks"],
                "{family}/{id}"
            );
            checked += 1;
        }
    }
    assert_eq!(checked, 18, "both DSML families must replay all nine cases");
}
