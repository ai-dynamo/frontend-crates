// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_parsers_v2::{
    Tool, UnifiedEvent, UnifiedParserExt, assemble, create_tool_parser_for_family,
    create_unified_parser_for_family,
};

fn tools() -> Vec<Tool> {
    ["get_weather", "commentary"]
        .into_iter()
        .map(|name| Tool {
            name: name.into(),
            description: None,
            parameters: serde_json::json!({
                "type": "object", "properties": {"city": {"type": "string"}}
            }),
            strict: None,
        })
        .collect()
}

const TURN: &str = concat!(
    "<|start|>assistant to=self<|message|>Check the weather.<|eom|>",
    "<|start|>assistant to=commentary<|message|>Checking.<|eom|>",
    "<|start|>assistant to=get_weather<|message|><atem:function_calls>",
    "<atem:invoke name=\"get_weather\"><atem:parameter name=\"city\">Paris</atem:parameter>",
    "</atem:invoke></atem:function_calls><|eom|>",
    "<|start|>assistant to=user<|message|>Done.<|eot|>"
);

#[test]
fn muse_spark_open_keeps_reasoning_commentary_calls_and_final_phases() {
    let mut legacy = create_unified_parser_for_family("muse_spark", &tools()).unwrap();
    legacy.preserve_message_phases(true);
    let expected = legacy.parse_complete(TURN).unwrap();
    assert!(
        expected
            .iter()
            .any(|e| matches!(e, UnifiedEvent::Reasoning { text } if text == "Check the weather."))
    );
    assert!(expected.iter().any(|e| matches!(e, UnifiedEvent::ToolCall { name, arguments } if name == "get_weather" && arguments["city"] == "Paris")));
    let phases: Vec<_> = expected
        .iter()
        .filter_map(|e| match e {
            UnifiedEvent::MessageStart { phase } => Some(*phase),
            _ => None,
        })
        .collect();
    assert_eq!(
        serde_json::to_value(phases).unwrap(),
        serde_json::json!(["commentary", "final_answer"])
    );

    let mut parser = create_unified_parser_for_family("muse_spark_open", &tools()).unwrap();
    parser.preserve_message_phases(true);
    assert_eq!(parser.parse_complete(TURN).unwrap(), expected);

    let mut parser = create_unified_parser_for_family("muse_spark_open", &tools()).unwrap();
    parser.preserve_message_phases(true);
    let mut deltas = Vec::new();
    for ch in TURN.chars() {
        deltas.extend(parser.push(&ch.to_string()).unwrap());
    }
    deltas.extend(parser.finish().unwrap());
    assert_eq!(assemble(&deltas), expected);
}

#[test]
fn muse_spark_open_tool_parser_keeps_commentary_out_of_tool_calls() {
    let mut legacy = create_tool_parser_for_family("muse_spark", &tools()).unwrap();
    let expected = legacy.parse_complete(TURN).unwrap();
    let mut parser = create_tool_parser_for_family("muse_spark_open", &tools()).unwrap();
    assert!(parser.preserve_special_tokens());
    let actual = parser.parse_complete(TURN).unwrap();
    assert_eq!(actual.normal_text, expected.normal_text);
    // The tool-only parser leaves reasoning separation to its caller.
    assert_eq!(actual.normal_text, "Check the weather.Checking.Done.");
    assert_eq!(actual.calls, expected.calls);
    assert_eq!(actual.calls.len(), 1);
    assert_eq!(actual.calls[0].name.as_deref(), Some("get_weather"));
}
