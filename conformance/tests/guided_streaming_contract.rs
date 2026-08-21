// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_parsers::tool_calling::jail::{Annotated, JailedStream};
use dynamo_parsers_v2::{
    InvalidGuidedPayloadPolicy, Tool, UnifiedParserEvent, UnifiedParserExt, UnifiedParserInit,
    UnifiedParserStartingState, UnifiedToolOutputMode, create_unified_parser_for_family,
};
use dynamo_protocols::types::{
    ChatChoiceStream, ChatCompletionMessageContent, ChatCompletionStreamResponseDelta,
    CreateChatCompletionStreamResponse, Role,
};
use futures::StreamExt;
use futures::stream;

struct Case {
    named: bool,
    payload: &'static str,
    arguments: &'static str,
}

fn v1_chunk(content: &str) -> Annotated<CreateChatCompletionStreamResponse> {
    #[allow(deprecated)]
    let choice = ChatChoiceStream {
        index: 0,
        delta: ChatCompletionStreamResponseDelta {
            role: Some(Role::Assistant),
            content: Some(ChatCompletionMessageContent::Text(content.to_string())),
            tool_calls: None,
            function_call: None,
            refusal: None,
            reasoning_content: None,
        },
        finish_reason: None,
        logprobs: None,
    };
    Annotated {
        data: Some(CreateChatCompletionStreamResponse {
            id: "guided-conformance".to_string(),
            choices: vec![choice],
            created: 0,
            model: "test".to_string(),
            system_fingerprint: None,
            object: "chat.completion.chunk".to_string(),
            usage: None,
            service_tier: None,
        }),
        id: None,
        event: None,
        comment: None,
        error: None,
    }
}

fn chunks(payload: &str) -> Vec<&str> {
    payload
        .char_indices()
        .map(|(at, ch)| &payload[at..at + ch.len_utf8()])
        .collect()
}

async fn v1(case: &Case) -> (usize, Vec<String>, String, String) {
    let builder = JailedStream::builder().guided_streaming(true);
    let jail = if case.named {
        builder.tool_choice_named("search".to_string()).build()
    } else {
        builder.tool_choice_required().build()
    };
    let input = chunks(case.payload).into_iter().map(v1_chunk);
    let output: Vec<_> = jail
        .apply_with_finish_reason(stream::iter(input))
        .collect()
        .await;

    let mut carrying = 0;
    let mut names = Vec::new();
    let mut arguments = String::new();
    let mut content = String::new();
    for response in output {
        let Some(data) = response.data else {
            continue;
        };
        for choice in data.choices {
            if let Some(ChatCompletionMessageContent::Text(text)) = choice.delta.content {
                content.push_str(&text);
            }
            if let Some(calls) = choice.delta.tool_calls {
                carrying += 1;
                for call in calls {
                    if let Some(function) = call.function {
                        if let Some(name) = function.name {
                            names.push(name);
                        }
                        if let Some(fragment) = function.arguments {
                            arguments.push_str(&fragment);
                        }
                    }
                }
            }
        }
    }
    (carrying, names, arguments, content)
}

fn v2(case: &Case) -> (usize, Vec<String>, String, String) {
    let tools = [Tool {
        name: "search".to_string(),
        description: None,
        parameters: serde_json::json!({"type":"object"}),
        strict: None,
    }];
    let mut parser = create_unified_parser_for_family("qwen3", &tools).unwrap();
    parser
        .initialize_request(UnifiedParserInit {
            prompt_token_ids: Vec::new(),
            starting_state: UnifiedParserStartingState::None,
            tool_output_mode: UnifiedToolOutputMode::GuidedJson {
                named_tool: case.named.then(|| "search".to_string()),
            },
            invalid_guided_payload: InvalidGuidedPayloadPolicy::StreamBestEffort,
        })
        .unwrap();

    let mut events = Vec::new();
    for chunk in chunks(case.payload) {
        events.extend(parser.push(chunk).unwrap());
    }
    events.extend(parser.finish().unwrap().events);

    let mut carrying = 0;
    let mut names = Vec::new();
    let mut arguments = String::new();
    let mut content = String::new();
    for event in events {
        match event {
            UnifiedParserEvent::ToolCall(call) => {
                carrying += 1;
                if let Some(name) = call.name {
                    names.push(name);
                }
                arguments.push_str(&call.arguments);
            }
            UnifiedParserEvent::Text(text) => content.push_str(&text),
            UnifiedParserEvent::Reasoning(_) => {}
        }
    }
    (carrying, names, arguments, content)
}

#[tokio::test]
async fn v1_and_v2_share_the_guided_streaming_contract() {
    let cases = [
        Case {
            named: false,
            payload: r#"[{"name":"search","parameters":{"query":"Rust","limit":10}}]"#,
            arguments: r#"{"query":"Rust","limit":10}"#,
        },
        Case {
            named: true,
            payload: "  {\"query\":\"Rust\",\"limit\":10}",
            arguments: r#"{"query":"Rust","limit":10}"#,
        },
    ];

    for case in cases {
        let v1 = v1(&case).await;
        let v2 = v2(&case);
        for (name, output) in [("v1", &v1), ("v2", &v2)] {
            assert_eq!(output.1, vec!["search".to_string()], "{name}");
            assert_eq!(output.2, case.arguments, "{name}");
            assert!(
                output.3.is_empty(),
                "{name} leaked JSON as text: {output:?}"
            );
            assert!(
                output.0 > 2,
                "{name} buffered instead of streaming: {output:?}"
            );
        }
    }
}

/// A truncated guided payload used to leak its call envelope into the assistant
/// message: the array shape delivered `},{"name":` as text and the single-call shape
/// delivered a bare `}`. The cursor releases argument bytes only up to the argument
/// object's own closing brace, so everything after it is envelope. Once streaming has
/// committed, that tail is never assistant text. v2 had the identical bug at a
/// different, pre-existing site (frontend-crates#195) - both generations are driven
/// here, the way the sibling contract test above does.
#[tokio::test]
async fn a_truncated_guided_payload_never_leaks_its_envelope_as_text() {
    let cases = [
        // Cut mid-envelope, just after a complete first call.
        Case {
            named: false,
            payload: r#"[{"name":"search","parameters":{"query":"Rust"}},{"name":"#,
            arguments: r#"{"query":"Rust"}"#,
        },
        // Cut before the array's own `]`.
        Case {
            named: false,
            payload: r#"[{"name":"search","parameters":{"query":"Rust"}}"#,
            arguments: r#"{"query":"Rust"}"#,
        },
    ];

    for case in cases {
        let v1 = v1(&case).await;
        let v2 = v2(&case);
        for (name, output) in [("v1", &v1), ("v2", &v2)] {
            assert_eq!(
                output.1,
                vec!["search".to_string()],
                "{name} {:?}",
                case.payload
            );
            assert_eq!(output.2, case.arguments, "{name} {:?}", case.payload);
            assert!(
                output.3.is_empty(),
                "{name} truncated payload {:?} leaked envelope bytes as text: {output:?}",
                case.payload
            );
            assert!(
                output.0 > 2,
                "{name} truncated payload {:?} buffered instead of streaming",
                case.payload
            );
        }
    }
}
