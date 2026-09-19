// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_parsers_v2::{ToolParseResult, create_tool_parser_for_family};
use serde_json::{Value, json};

#[test]
fn tool_projection_preserves_reasoning_markers_and_parallel_calls() {
    let calls = concat!(
        "\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"get_weather\">\n",
        "<｜DSML｜ parameter name=\"city\" string=\"true\">杭州</｜DSML｜ parameter>\n",
        "<｜DSML｜ parameter name=\"count\" string=\"false\">42</｜DSML｜ parameter>\n",
        "</｜DSML｜ invoke>\n<｜DSML｜ invoke name=\"add\">\n",
        "<｜DSML｜ parameter name=\"x\" string=\"false\">1.5</｜DSML｜ parameter>\n",
        "<｜DSML｜ parameter name=\"y\" string=\"false\">2.25</｜DSML｜ parameter>\n",
        "</｜DSML｜ invoke>\n</｜DSML｜ calls>"
    );
    for prefix in [
        "Checking.",
        "<think>Plan.</think>Checking.",
        "Plan.</think>Checking.",
    ] {
        let input = format!("{prefix}{calls}");
        for split in input.char_indices().map(|(at, _)| at).chain([input.len()]) {
            let mut parser = create_tool_parser_for_family("deepseek_v41", &[]).unwrap();
            assert!(parser.preserve_special_tokens());
            let mut result = ToolParseResult::default();
            for chunk in [&input[..split], "", &input[split..]] {
                result.append(parser.push(chunk).unwrap());
            }
            result.append(parser.finish().unwrap());
            let result = result.coalesce_calls();
            assert_eq!(result.normal_text, format!("{prefix}\n\n"));
            assert_eq!(result.calls.len(), 2);
            for (index, (name, arguments)) in [
                ("get_weather", json!({"city": "杭州", "count": 42})),
                ("add", json!({"x": 1.5, "y": 2.25})),
            ]
            .into_iter()
            .enumerate()
            {
                let call = &result.calls[index];
                assert_eq!(call.tool_index, index);
                assert_eq!(call.name.as_deref(), Some(name));
                assert_eq!(
                    serde_json::from_str::<Value>(&call.arguments).unwrap(),
                    arguments
                );
                assert!(call.complete);
            }
        }
    }
}

#[test]
fn tool_projection_flushes_plain_text_without_completing_truncated_calls() {
    for suffix in [
        "<thi",
        "<｜DSML｜ calls><｜DSML｜ invoke name=\"run\"><｜DSML｜ parameter name=\"text\" string=\"true\">unfinished",
    ] {
        let mut parser = create_tool_parser_for_family("deepseek_v41", &[]).unwrap();
        let mut result = parser
            .push(&format!("<think>plan</think>{suffix}"))
            .unwrap();
        result.append(parser.finish().unwrap());
        let result = result.coalesce_calls();
        assert!(result.calls.is_empty());
        let expected = if suffix == "<thi" {
            "<think>plan</think><thi"
        } else {
            "<think>plan</think>"
        };
        assert_eq!(result.normal_text, expected);
    }
}
