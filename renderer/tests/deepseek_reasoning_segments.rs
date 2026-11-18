// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_protocols::types::CreateChatCompletionRequest;
use dynamo_renderer::{
    OAIPromptFormatter,
    deepseek::{v4::DeepSeekV4Formatter, v41::DeepSeekV41Formatter},
};
use serde_json::{Value, json};

fn request(reasoning: Value, with_tools: bool) -> CreateChatCompletionRequest {
    let mut body = json!({
        "model": "deepseek",
        "messages": [
            {"role": "user", "content": "Look up the weather."},
            {
                "role": "assistant", "content": null, "reasoning_content": reasoning,
                "tool_calls": [{
                    "id": "call_1", "type": "function",
                    "function": {"name": "weather", "arguments": "{}"}
                }]
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "Sunny"},
            {"role": "user", "content": "What about tomorrow?"}
        ]
    });
    if with_tools {
        body["tools"] = json!([{
            "type": "function",
            "function": {
                "name": "weather",
                "parameters": {"type": "object", "properties": {}}
            }
        }]);
    }
    serde_json::from_value(body).unwrap()
}

#[test]
fn tool_definitions_preserve_historical_reasoning_segments() {
    let v4 = DeepSeekV4Formatter::new_thinking();
    for formatter in [&v4 as &dyn OAIPromptFormatter, &DeepSeekV41Formatter] {
        let segmented = request(json!(["First reason.", "", "Second reason.", ""]), true);
        let flat = request(json!("First reason.\nSecond reason."), true);

        let prompt = formatter.render(&segmented).unwrap();
        assert_eq!(prompt, formatter.render(&flat).unwrap());
        assert!(prompt.contains("<think>First reason.\nSecond reason.</think>"));
    }
}

#[test]
fn default_drops_historical_reasoning_segments_without_tool_definitions() {
    let v4 = DeepSeekV4Formatter::new_thinking();
    for formatter in [&v4 as &dyn OAIPromptFormatter, &DeepSeekV41Formatter] {
        let segmented = request(json!(["First reason.", "", "Second reason.", ""]), false);
        let absent = request(Value::Null, false);

        let prompt = formatter.render(&segmented).unwrap();
        assert_eq!(prompt, formatter.render(&absent).unwrap());
        assert!(!prompt.contains("First reason."));
        assert!(!prompt.contains("Second reason."));
    }
}

#[test]
fn v41_rejects_literal_image_placeholders_in_reasoning_segments() {
    let segmented = request(json!(["First reason.", "<｜deepseek_image｜>", ""]), true);
    let error = DeepSeekV41Formatter.render(&segmented).unwrap_err();
    assert!(error.to_string().contains("literal image placeholders"));
}
