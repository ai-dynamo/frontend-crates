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
