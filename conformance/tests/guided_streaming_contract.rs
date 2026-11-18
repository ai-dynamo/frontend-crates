// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_parsers_v2::{
    InvalidGuidedPayloadPolicy, Tool, UnifiedParserEvent, UnifiedParserExt, UnifiedParserInit,
    UnifiedParserStartingState, UnifiedToolOutputMode, create_unified_parser_for_family,
};
struct Case {
    named: bool,
    payload: &'static str,
    arguments: &'static str,
}

fn chunks(payload: &str) -> Vec<&str> {
    payload
        .char_indices()
        .map(|(at, ch)| &payload[at..at + ch.len_utf8()])
        .collect()
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
            UnifiedParserEvent::MessageStart(_) | UnifiedParserEvent::MessageEnd => {
                unreachable!("message phases are opt-in")
            }
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

#[test]
fn unified_guided_streaming_contract() {
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
        let output = v2(&case);

        assert_eq!(output.1, vec!["search".to_string()], "Unified v2");
        assert_eq!(output.2, case.arguments, "Unified v2");
        assert!(
            output.3.is_empty(),
            "Unified v2 leaked JSON as text: {output:?}"
        );
        assert!(
            output.0 > 2,
            "Unified v2 buffered instead of streaming: {output:?}"
        );
    }
}

/// A truncated guided payload used to leak its call envelope into the assistant
/// message: the array shape delivered `},{"name":` as text and the single-call shape
/// delivered a bare `}`. The cursor releases argument bytes only up to the argument
/// object's own closing brace, so everything after it is envelope. Once streaming has
/// committed, that tail must never become assistant text.
#[test]
fn a_truncated_guided_payload_never_leaks_its_envelope_as_text() {
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
        let output = v2(&case);

        assert_eq!(
            output.1,
            vec!["search".to_string()],
            "Unified v2 {:?}",
            case.payload
        );
        assert_eq!(output.2, case.arguments, "Unified v2 {:?}", case.payload);
        assert!(
            output.3.is_empty(),
            "Unified v2 truncated payload {:?} leaked envelope bytes as text: {output:?}",
            case.payload
        );
        assert!(
            output.0 > 2,
            "Unified v2 truncated payload {:?} buffered instead of streaming",
            case.payload
        );
    }
}
