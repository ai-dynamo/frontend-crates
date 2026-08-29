// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Gemma 4 guided-mode behavior through the public unified-parser factory.

use dynamo_parsers_v2::{
    InvalidGuidedPayloadPolicy, Tool, UnifiedEvent, UnifiedParserExt, UnifiedParserInit,
    UnifiedParserStartingState, UnifiedToolOutputMode, assemble,
};

fn weather_tools() -> Vec<Tool> {
    vec![Tool {
        name: "get_weather".into(),
        description: None,
        parameters: serde_json::json!({
            "type": "object",
            "properties": { "city": { "type": "string" } }
        }),
        strict: None,
    }]
}

fn assert_guided_at_every_split(
    input: &str,
    mode: UnifiedToolOutputMode,
    expected: &[UnifiedEvent],
) {
    let tools = weather_tools();
    let mut split_points = vec![None];
    split_points.extend(
        (1..input.len())
            .filter(|&split| input.is_char_boundary(split))
            .map(Some),
    );
    for split in split_points {
        let mut parser = dynamo_parsers_v2::create_unified_parser_for_family("gemma4", &tools)
            .expect("built-in Gemma 4 parser");
        parser
            .initialize_request(UnifiedParserInit {
                starting_state: UnifiedParserStartingState::None,
                tool_output_mode: mode.clone(),
                invalid_guided_payload: InvalidGuidedPayloadPolicy::RecoverAsText,
                ..UnifiedParserInit::default()
            })
            .expect("guided request");

        let chunks = split.map_or_else(|| vec![input], |at| vec![&input[..at], &input[at..]]);
        let mut deltas = Vec::new();
        for chunk in chunks {
            deltas.extend(parser.push(chunk).expect("push"));
        }
        deltas.extend(parser.finish().expect("finish").events);
        assert_eq!(assemble(&deltas), expected, "split {split:?}");
    }
}

fn guided_init(named_tool: Option<&str>, policy: InvalidGuidedPayloadPolicy) -> UnifiedParserInit {
    UnifiedParserInit {
        starting_state: UnifiedParserStartingState::None,
        tool_output_mode: UnifiedToolOutputMode::GuidedJson {
            named_tool: named_tool.map(str::to_owned),
        },
        invalid_guided_payload: policy,
        ..UnifiedParserInit::default()
    }
}

/// Gemma's lexical `call:` prefix is only structural when the grammar-aware
/// scanner accepts a call body. Guided mode must not strip the same bytes from
/// ordinary visible prose, even when the prefix is split across pushes. This
/// stays a public-factory regression instead of joining the unified corpus:
/// the corpus generator cannot add one family-specific guided input without
/// creating irrelevant rows for every guided family.
#[test]
fn preserves_ordinary_call_prefix_at_every_split() {
    let cases = [
        (
            "I will call: you tomorrow<|channel>thought\nchecking<channel|>[{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Paris\"}}]",
            "I will call: you tomorrow",
            "checking",
        ),
        (
            "prefix call:{\"x\":1}<|channel>thought\nwhy<channel|>[{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Paris\"}}]",
            "prefix call:{\"x\":1}",
            "why",
        ),
    ];
    for (input, visible, reasoning) in cases {
        let expected = vec![
            UnifiedEvent::Text {
                text: visible.into(),
            },
            UnifiedEvent::Reasoning {
                text: reasoning.into(),
            },
            UnifiedEvent::ToolCall {
                name: "get_weather".into(),
                arguments: serde_json::json!({"city": "Paris"}),
            },
        ];
        assert_guided_at_every_split(
            input,
            UnifiedToolOutputMode::GuidedJson { named_tool: None },
            &expected,
        );
    }
}

#[test]
fn consumes_valid_call_prefix_at_every_split() {
    let expected = [UnifiedEvent::ToolCall {
        name: "get_weather".into(),
        arguments: serde_json::json!({"city": "Paris"}),
    }];
    let cases = [
        (
            "call:[{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Paris\"}}]",
            UnifiedToolOutputMode::GuidedJson { named_tool: None },
        ),
        (
            "call:{\"city\":\"Paris\"}",
            UnifiedToolOutputMode::GuidedJson {
                named_tool: Some("get_weather".into()),
            },
        ),
    ];
    for (input, mode) in cases {
        assert_guided_at_every_split(input, mode, &expected);
    }
}

#[test]
fn strips_malformed_call_prefixes_without_losing_reasoning_at_every_split() {
    let cases = [
        (
            "call:<|channel>thought\nsecret<channel|>[{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Paris\"}}]",
            "secret",
        ),
        (
            "<|channel>thought\nI'll call call:get_weather<channel|>[{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Paris\"}}]",
            "I'll call get_weather",
        ),
    ];
    for (input, reasoning) in cases {
        let expected = vec![
            UnifiedEvent::Reasoning {
                text: reasoning.into(),
            },
            UnifiedEvent::ToolCall {
                name: "get_weather".into(),
                arguments: serde_json::json!({"city": "Paris"}),
            },
        ];
        assert_guided_at_every_split(
            input,
            UnifiedToolOutputMode::GuidedJson { named_tool: None },
            &expected,
        );
    }
}

/// A public factory parser can abandon an incomplete native-looking candidate,
/// then serve a fresh guided request without carrying scanner or cursor state
/// across requests.
#[test]
fn reset_after_partial_native_candidate_reuses_the_public_factory_for_guided_json() {
    let tools = weather_tools();
    let mut parser = dynamo_parsers_v2::create_unified_parser_for_family("gemma4", &tools)
        .expect("built-in Gemma 4 parser");
    parser
        .initialize_request(guided_init(None, InvalidGuidedPayloadPolicy::RecoverAsText))
        .expect("first guided request");

    assert!(
        parser
            .push("<|tool_call>call:get_weather{city:<|\"|>Par")
            .expect("partial native-looking candidate")
            .is_empty()
    );
    assert_eq!(
        parser.reset(),
        "call:get_weather{city:<|\"|>Par",
        "the recognized native wrapper is control markup, while its incomplete body is recovered"
    );

    parser
        .initialize_request(guided_init(
            Some("get_weather"),
            InvalidGuidedPayloadPolicy::RecoverAsText,
        ))
        .expect("fresh named guided request");
    let mut events = parser.push(r#"{"city":"Paris"}"#).expect("guided JSON");
    events.extend(parser.finish().expect("finish").events);
    assert_eq!(
        assemble(&events),
        vec![UnifiedEvent::ToolCall {
            name: "get_weather".into(),
            arguments: serde_json::json!({"city": "Paris"}),
        }]
    );
}

/// A rejected request setup must not poison a parser that is reset before a
/// valid guided request. This exercises the public lifecycle rather than the
/// generic router directly.
#[test]
fn rejected_guided_initialization_then_reset_allows_valid_initialization() {
    let tools = weather_tools();
    let mut parser = dynamo_parsers_v2::create_unified_parser_for_family("gemma4", &tools)
        .expect("built-in Gemma 4 parser");
    parser
        .initialize_request(UnifiedParserInit::default())
        .expect("native request");
    parser.push("visible").expect("start native request");

    assert!(
        parser
            .initialize_request(guided_init(None, InvalidGuidedPayloadPolicy::RecoverAsText))
            .is_err()
    );
    assert_eq!(parser.reset(), "");

    parser
        .initialize_request(guided_init(None, InvalidGuidedPayloadPolicy::RecoverAsText))
        .expect("valid guided request after rejected setup");
    let mut events = parser
        .push(r#"[{"name":"get_weather","arguments":{"city":"Paris"}}]"#)
        .expect("guided JSON");
    events.extend(parser.finish().expect("finish").events);
    assert_eq!(
        assemble(&events),
        vec![UnifiedEvent::ToolCall {
            name: "get_weather".into(),
            arguments: serde_json::json!({"city": "Paris"}),
        }]
    );
}

/// Native-looking markup is suppressed immediately. Once a real reasoning
/// channel closes, it must be emitted on that push, and guided JSON must commit
/// before finish under the explicitly streaming policy.
#[test]
fn malformed_native_envelope_then_reasoning_and_guided_json_emit_at_their_crossings() {
    let tools = weather_tools();
    let mut parser = dynamo_parsers_v2::create_unified_parser_for_family("gemma4", &tools)
        .expect("built-in Gemma 4 parser");
    parser
        .initialize_request(guided_init(
            None,
            InvalidGuidedPayloadPolicy::StreamBestEffort,
        ))
        .expect("streaming guided request");

    assert!(
        parser
            .push("<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>}<tool_call|>")
            .expect("native-looking envelope")
            .is_empty()
    );
    assert_eq!(
        parser
            .push("<|channel>thought\nchecking<channel|>")
            .expect("completed reasoning"),
        vec![dynamo_parsers_v2::UnifiedParserEvent::Reasoning(
            "checking".into()
        )],
        "reasoning must not wait for guided JSON or finish"
    );
    let json_events = parser
        .push(r#"[{"name":"get_weather","arguments":{"city":"Paris"}}]"#)
        .expect("completed guided JSON");
    assert!(
        json_events.iter().any(|event| matches!(
            event,
            dynamo_parsers_v2::UnifiedParserEvent::ToolCall(call)
                if call.name.as_deref() == Some("get_weather")
        )),
        "guided JSON must commit when its call crosses completion, before finish"
    );
    assert!(parser.finish().expect("finish").events.is_empty());
}

/// Kimi does not install Gemma's optional `call:` policy. Its existing guided
/// JSON input must still dispatch normally, proving the generic router treats an
/// absent policy as the former no-op behavior.
#[test]
fn kimi_guided_dispatch_is_unchanged_without_a_prefix_policy() {
    let tools = weather_tools();
    let mut parser = dynamo_parsers_v2::create_unified_parser_for_family("kimi_k2", &tools)
        .expect("built-in Kimi K2 parser");
    parser
        .initialize_request(UnifiedParserInit {
            starting_state: UnifiedParserStartingState::None,
            tool_output_mode: UnifiedToolOutputMode::GuidedJson { named_tool: None },
            invalid_guided_payload: InvalidGuidedPayloadPolicy::RecoverAsText,
            ..UnifiedParserInit::default()
        })
        .expect("guided request");

    let input = r#"[{"name":"get_weather","arguments":{"city":"Paris"}}]"#;
    let mut events = parser.push(input).expect("push");
    events.extend(parser.finish().expect("finish").events);
    assert_eq!(
        assemble(&events),
        vec![UnifiedEvent::ToolCall {
            name: "get_weather".into(),
            arguments: serde_json::json!({"city": "Paris"}),
        }]
    );
}
