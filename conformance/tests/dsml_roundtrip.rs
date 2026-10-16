// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_parsers::tool_calling::parsers::detect_and_parse_tool_call_with_recovery;
use dynamo_parsers_v2::{
    UnifiedEvent, UnifiedParserExt, UnifiedParserInit, UnifiedParserStartingState, assemble,
    create_tool_parser_for_family, create_unified_parser_for_family,
};
use dynamo_renderer::deepseek::{common::ThinkingMode, v4, v32, v41};
use serde_json::{Value, json};

fn render(family: &str, content: &str, calls: &[Value]) -> String {
    let messages = [json!({
        "role": "assistant", "content": content, "tool_calls": calls,
    })];
    let encoded = match family {
        "deepseek_v3_2" => v32::encode_messages(&messages, ThinkingMode::Chat, false),
        "deepseek_v4" => v4::encode_messages(&messages, ThinkingMode::Chat, false),
        "deepseek_v41" => v41::encode_messages(&messages, ThinkingMode::Chat, true, 75),
        _ => unreachable!("test family"),
    }
    .expect("render assistant turn");
    encoded
        .strip_prefix(v4::tokens::BOS)
        .unwrap_or(&encoded)
        .strip_suffix(v4::tokens::EOS)
        .expect("assistant end marker")
        .to_string()
}

fn call(name: &str, arguments: Value) -> Value {
    json!({"type": "function", "function": {
        "name": name, "arguments": arguments.to_string(),
    }})
}

fn tools() -> Vec<Value> {
    vec![
        call(
            "weather",
            json!({"city": "東京", "days": 3, "options": [true, null]}),
        ),
        call("echo", json!({"text": "first\n\nsecond"})),
    ]
}

#[tokio::test]
async fn v1_render_parse_render_preserves_dsml_bytes() {
    for family in ["deepseek_v3_2", "deepseek_v4"] {
        for content in ["", "Checking.", "Checking.\n\n", "\n\n", " \t"] {
            for count in [1, 2] {
                let original_calls = tools();
                let generated = render(family, content, &original_calls[..count]);
                let (calls, parsed_content) =
                    detect_and_parse_tool_call_with_recovery(&generated, Some(family), None)
                        .await
                        .expect("parse generated turn");
                assert_eq!(parsed_content.as_deref(), Some(content), "{family}");
                let calls: Vec<_> = calls
                    .into_iter()
                    .map(|call| call_from_strings(&call.function.name, &call.function.arguments))
                    .collect();
                assert_eq!(calls.len(), count, "{family}");
                assert_eq!(render(family, content, &calls), generated, "{family}");
            }
        }
    }
}

fn call_from_strings(name: &str, arguments: &str) -> Value {
    call(
        name,
        serde_json::from_str(arguments).expect("JSON arguments"),
    )
}

#[test]
fn v2_render_parse_render_preserves_dsml_bytes() {
    for family in ["deepseek_v4", "deepseek_v41"] {
        for content in ["", "Checking.", "Checking.\n\n", "\n\n", " \t"] {
            for count in [1, 2] {
                let original_calls = tools();
                let generated = render(family, content, &original_calls[..count]);
                for characters in [false, true] {
                    let chunks: Vec<_> = if characters {
                        generated.chars().map(|ch| ch.to_string()).collect()
                    } else {
                        vec![generated.clone()]
                    };
                    let mut parser = create_unified_parser_for_family(family, &[]).unwrap();
                    parser
                        .initialize_request(UnifiedParserInit {
                            starting_state: UnifiedParserStartingState::Response,
                            ..Default::default()
                        })
                        .unwrap();
                    let mut events = Vec::new();
                    for chunk in &chunks {
                        events.extend(parser.push(chunk).unwrap());
                    }
                    events.extend(parser.finish().unwrap().events);
                    let mut parsed_content = String::new();
                    let mut calls = Vec::new();
                    for event in assemble(&events) {
                        match event {
                            UnifiedEvent::Text { text } => parsed_content.push_str(&text),
                            UnifiedEvent::ToolCall { name, arguments } => {
                                calls.push(call(&name, arguments))
                            }
                            UnifiedEvent::Reasoning { .. } => panic!("unexpected reasoning"),
                        }
                    }
                    assert_eq!(parsed_content, content, "{family}, characters={characters}");
                    assert_eq!(calls.len(), count, "{family}");
                    assert_eq!(
                        render(family, &parsed_content, &calls),
                        generated,
                        "{family}"
                    );

                    let mut legacy = create_tool_parser_for_family("deepseek_v4", &[]).unwrap();
                    let mut output = legacy.push("").unwrap();
                    for chunk in &chunks {
                        output.append(legacy.push(chunk).unwrap());
                    }
                    output.append(legacy.finish().unwrap());
                    let output = output.coalesce_calls();
                    let calls: Vec<_> = output
                        .calls
                        .into_iter()
                        .map(|call| {
                            call_from_strings(
                                call.name.as_deref().expect("call name"),
                                &call.arguments,
                            )
                        })
                        .collect();
                    assert_eq!(
                        render(family, &output.normal_text, &calls),
                        generated,
                        "legacy {family}"
                    );
                }
            }
        }
    }
}
