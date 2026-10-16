// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Exercise the packaged MiniMax M3 downstream-boundary reasoning cases.
mod common;

use dynamo_parsers::reasoning::{ReasoningParser, ReasoningParserType};
use serde_json::Value;

#[test]
fn minimax_m3_required_reasoning_boundaries() {
    let path = common::ensure_fixtures()
        .join("reasoning/fixtures-v1/inputs/minimax_m3/REASONING.stream.4-boundaries.yaml");
    let fixture: Value = serde_yaml::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let cases = fixture["cases"].as_object().unwrap();
    assert_eq!(cases.len(), 3);
    for id in [
        "REASONING.stream.4.b",
        "REASONING.stream.4.c",
        "REASONING.stream.4.d",
    ] {
        let case = &cases[id];
        let mut parser = ReasoningParserType::get_reasoning_parser_from_name("minimax_m3");
        let mut reasoning = String::new();
        let mut normal = String::new();
        for chunk in case["chunks"].as_array().unwrap() {
            let result = parser.parse_reasoning_streaming_incremental(chunk.as_str().unwrap(), &[]);
            reasoning.push_str(&result.reasoning_text);
            normal.push_str(&result.normal_text);
        }
        let result = parser.finish_reasoning_stream();
        reasoning.push_str(&result.reasoning_text);
        normal.push_str(&result.normal_text);
        assert_eq!(
            reasoning,
            case["expected"]["dynamo_v1"]["reasoning_text"]
                .as_str()
                .unwrap(),
            "{id}"
        );
        assert_eq!(
            normal,
            case["expected"]["dynamo_v1"]["normal_text"]
                .as_str()
                .unwrap(),
            "{id}"
        );
    }
}
