// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_parsers_v2::{
    Tool, UnifiedParserExt, create_tool_parser_for_family, create_unified_parser_for_family,
    encode_harmony, tool_arguments_raw,
};
use serde_json::json;

const FAMILIES: &[&str] = &[
    "glm47",
    "gemma4",
    "minimax_m3",
    "minimax_m2",
    "qwen3_coder",
    "deepseek_v4",
    "muse_glimmer",
    "kimi_k2",
    "harmony",
];

fn wire(family: &str, value: &str) -> String {
    match family {
        "glm47" => format!(
            "<tool_call>run<arg_key>value</arg_key><arg_value>{value}</arg_value></tool_call>"
        ),
        "gemma4" => format!(
            "<|tool_call>call:run{{value:{}}}<tool_call|>",
            value.replace("\"inner\"", "inner")
        ),
        "minimax_m3" => format!(
            "]<]minimax[>[<tool_call>]<]minimax[>[<invoke name=\"run\">]<]minimax[>[<value>{value}]<]minimax[>[</value>]<]minimax[>[</invoke>]<]minimax[>[</tool_call>"
        ),
        "minimax_m2" => format!(
            "<minimax:tool_call><invoke name=\"run\"><parameter name=\"value\">{value}</parameter></invoke></minimax:tool_call>"
        ),
        "qwen3_coder" => format!(
            "<tool_call><function=run><parameter=value>{value}</parameter></function></tool_call>"
        ),
        "deepseek_v4" => format!(
            "<｜DSML｜tool_calls><｜DSML｜invoke name=\"run\"><｜DSML｜parameter name=\"value\" string=\"false\">{value}</｜DSML｜parameter></｜DSML｜invoke></｜DSML｜tool_calls>"
        ),
        "muse_glimmer" => format!(
            "<|start|>assistant to=run<|message|><atem:function_calls><atem:invoke name=\"run\"><atem:parameter name=\"value\">{value}</atem:parameter></atem:invoke></atem:function_calls><|eom|>"
        ),
        "kimi_k2" => format!(
            "<|tool_calls_section_begin|><|tool_call_begin|>functions.run:0<|tool_call_argument_begin|>{{\"value\":{value}}}<|tool_call_end|><|tool_calls_section_end|>"
        ),
        "harmony" => format!(
            "<|channel|>commentary to=functions.run <|constrain|>json<|message|>{{\"value\":{value}}}<|call|>"
        ),
        _ => panic!("unknown family"),
    }
}

fn tools(kind: &str) -> [Tool; 1] {
    [Tool {
        name: "run".into(),
        description: None,
        parameters: json!({"type":"object","properties":{"value":{"type":kind}}}),
        strict: None,
    }]
}

fn parse(family: &str, input: &str, split: usize, kind: &str) -> String {
    let mut parser = create_tool_parser_for_family(family, &tools(kind)).unwrap();
    let mut arguments = String::new();
    let mut complete = false;
    for chunk in [&input[..split], &input[split..]] {
        for call in parser.push(chunk).unwrap().calls {
            arguments.push_str(&call.arguments);
            complete |= call.complete;
        }
    }
    for call in parser.finish().unwrap().calls {
        arguments.push_str(&call.arguments);
        complete |= call.complete;
    }
    assert!(complete, "{family} split {split}");
    arguments
}

#[test]
fn exact_numeric_arguments_at_every_split() {
    for number in [
        "9007199254740992.5",
        "-9007199254740993.25",
        "0.10000000000000000001",
        "1e-400",
        "9.0071992547409925e15",
        "1e999999999999999999999999999999",
    ] {
        for family in FAMILIES {
            for nested in [false, true] {
                let value = if nested {
                    format!("{{\"inner\":[{number}]}}")
                } else {
                    number.into()
                };
                let input = wire(family, &value);
                let expected = format!("{{\"value\":{value}}}");
                for split in (0..=input.len()).filter(|&s| input.is_char_boundary(s)) {
                    assert_eq!(
                        parse(
                            family,
                            &input,
                            split,
                            if nested { "object" } else { "number" }
                        ),
                        expected,
                        "{family} split {split}"
                    );
                }
            }
        }
    }
}

#[test]
fn integer_conversion_is_decimal_exact() {
    for family in ["glm47", "minimax_m3", "minimax_m2", "qwen3_coder"] {
        for (raw, expected) in [
            ("42.0", "42"),
            ("4.2e1", "42"),
            ("42.0000000000000001", "\"42.0000000000000001\""),
            ("1e-400", "\"1e-400\""),
            ("9007199254740993.0", "9007199254740993"),
        ] {
            let input = wire(family, raw);
            for split in (0..=input.len()).filter(|&s| input.is_char_boundary(s)) {
                assert_eq!(
                    parse(family, &input, split, "integer"),
                    format!("{{\"value\":{expected}}}"),
                    "{family} {raw}"
                );
            }
        }
        let input = wire(family, "42");
        assert_eq!(parse(family, &input, 0, "string"), "{\"value\":\"42\"}");
    }
}

#[test]
fn unified_keeps_exact_raw_arguments_separate_from_value_projection() {
    for (native, unified) in [
        ("qwen3_coder", "qwen3"),
        ("glm47", "glm47"),
        ("gemma4", "gemma4"),
        ("deepseek_v4", "deepseek_v4"),
        ("muse_glimmer", "muse_glimmer"),
        ("kimi_k2", "kimi_k2"),
    ] {
        let input = wire(native, "{\"inner\":[9007199254740992.5]}");
        for split in (0..=input.len()).filter(|&s| input.is_char_boundary(s)) {
            let mut parser = create_unified_parser_for_family(unified, &tools("object")).unwrap();
            let mut events = parser.push(&input[..split]).unwrap();
            events.extend(parser.push(&input[split..]).unwrap());
            events.extend(parser.finish().unwrap().events);
            assert_eq!(
                tool_arguments_raw(&events).get(&0).map(String::as_str),
                Some("{\"value\":{\"inner\":[9007199254740992.5]}}"),
                "{unified} split {split}"
            );
        }
    }
}

#[test]
fn harmony_tokens_preserve_nested_numbers() {
    let input = wire("harmony", "{\"inner\":[9007199254740992.5,1e-400]}");
    let ids = encode_harmony(&input).unwrap();
    for split in 0..=ids.len() {
        let mut parser = create_tool_parser_for_family("harmony", &[]).unwrap();
        let mut arguments = String::new();
        for chunk in [&ids[..split], &ids[split..]] {
            for call in parser.push_tokens(chunk).unwrap().calls {
                arguments.push_str(&call.arguments);
            }
        }
        for call in parser.finish().unwrap().calls {
            arguments.push_str(&call.arguments);
        }
        assert_eq!(
            arguments, "{\"value\":{\"inner\":[9007199254740992.5,1e-400]}}",
            "split {split}"
        );
    }
}

#[test]
fn minimax_nested_tags_preserve_numbers_through_aggregation() {
    let tools = [Tool {
        name: "run".into(),
        description: None,
        parameters: json!({"type":"object","properties":{"value":{
            "type":"object","properties":{
                "inner":{"type":"number"},
                "items":{"type":"array","items":{"type":"number"}}
            }
        }}}),
        strict: None,
    }];
    let input = wire(
        "minimax_m3",
        concat!(
            "]<]minimax[>[<inner>9007199254740992.5]<]minimax[>[</inner>",
            "]<]minimax[>[<inner>-9007199254740993.25]<]minimax[>[</inner>",
            "]<]minimax[>[<items>]<]minimax[>[<item>1e-400]<]minimax[>[</item>",
            "]<]minimax[>[<item>9.0071992547409925e15]<]minimax[>[</item>]<]minimax[>[</items>"
        ),
    );
    for split in (0..=input.len()).filter(|&s| input.is_char_boundary(s)) {
        let mut parser = create_tool_parser_for_family("minimax_m3", &tools).unwrap();
        let mut arguments = String::new();
        let mut complete = false;
        for chunk in [&input[..split], &input[split..]] {
            for call in parser.push(chunk).unwrap().calls {
                arguments.push_str(&call.arguments);
                complete |= call.complete;
            }
        }
        assert!(
            complete,
            "nested call must complete before finish at split {split}"
        );
        for call in parser.finish().unwrap().calls {
            arguments.push_str(&call.arguments);
        }
        assert_eq!(
            arguments,
            "{\"value\":{\"inner\":[9007199254740992.5,-9007199254740993.25],\"items\":[1e-400,9.0071992547409925e15]}}",
            "split {split}"
        );
    }
}
