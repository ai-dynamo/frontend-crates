// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

use dynamo_parsers_v2::{ToolParseResult, create_tool_parser_for_family};
use serde_json::{Value, json};

fn invoke(spaced: bool, name: &str, value: &str) -> String {
    let gap = if spaced { " " } else { "" };
    format!(
        "<｜DSML｜{gap}invoke name=\"{name}\"><｜DSML｜{gap}parameter name=\"value\" string=\"true\">{value}</｜DSML｜{gap}parameter><｜DSML｜{gap}parameter name=\"count\" string=\"false\">42</｜DSML｜{gap}parameter></｜DSML｜{gap}invoke>"
    )
}

fn wrap(spaced: bool, body: &str) -> String {
    let tag = if spaced { " calls" } else { "tool_calls" };
    format!("<｜DSML｜{tag}>{body}</｜DSML｜{tag}>")
}

fn parse(chunks: &[&str]) -> ToolParseResult {
    let mut parser = create_tool_parser_for_family("deepseek_v4", &[]).unwrap();
    assert!(parser.preserve_special_tokens());
    let mut result = ToolParseResult::default();
    for chunk in chunks {
        result.append(parser.push(chunk).unwrap());
    }
    result.append(parser.finish().unwrap());
    result.coalesce_calls()
}

fn every_partition(input: &str, check: impl Fn(ToolParseResult)) {
    check(parse(&[input]));
    let boundaries: Vec<_> = input
        .char_indices()
        .map(|(i, _)| i)
        .chain([input.len()])
        .collect();
    let chars: Vec<_> = boundaries.windows(2).map(|w| &input[w[0]..w[1]]).collect();
    check(parse(&chars));
    for split in boundaries {
        check(parse(&[&input[..split], "", &input[split..]]));
    }
}

#[test]
fn existing_selector_accepts_both_dialects_and_parallel_calls() {
    for spaced in [false, true] {
        let value = "杭州 café &amp; <think>literal</think> <｜DSML｜ calls> <｜DSML｜tool_calls>";
        let body = format!(
            "{}{}",
            invoke(spaced, "inspect", value),
            invoke(spaced, "done", "end")
        );
        for prefix in ["before ", "<think>plan</think>answer ", "plan</think>\n\n"] {
            let input = format!("{prefix}{}after", wrap(spaced, &body));
            every_partition(&input, |result| {
                assert_eq!(result.normal_text, format!("{prefix}after"));
                assert_eq!(result.calls.len(), 2);
                for (index, (name, value)) in [("inspect", value), ("done", "end")]
                    .into_iter()
                    .enumerate()
                {
                    let call = &result.calls[index];
                    assert_eq!(call.tool_index, index);
                    assert_eq!(call.name.as_deref(), Some(name));
                    assert!(call.complete);
                    assert_eq!(
                        serde_json::from_str::<Value>(&call.arguments).unwrap(),
                        json!({"value":value,"count":42})
                    );
                }
            });
        }
    }
}

#[test]
fn dialect_selection_resets_between_invocations_and_bare_calls() {
    for (first, second) in [(false, true), (true, false)] {
        let body = format!(
            "{}{}",
            invoke(first, "first", "a"),
            invoke(second, "second", "b")
        );
        for input in [
            body.clone(),
            wrap(first, &body),
            format!(
                "{}{}",
                wrap(first, &invoke(first, "first", "a")),
                wrap(second, &invoke(second, "second", "b"))
            ),
        ] {
            every_partition(&input, |result| {
                assert!(result.normal_text.is_empty());
                assert_eq!(result.calls.len(), 2);
                assert_eq!(result.calls[0].name.as_deref(), Some("first"));
                assert_eq!(result.calls[1].name.as_deref(), Some("second"));
            });
        }
    }
}

#[test]
fn incomplete_calls_stay_incomplete_and_literal_closing_markers_are_data() {
    for spaced in [false, true] {
        let gap = if spaced { " " } else { "" };
        let incomplete = format!(
            "prefix<｜DSML｜{gap}invoke name=\"inspect\"><｜DSML｜{gap}parameter name=\"value\" string=\"true\">unfinished"
        );
        every_partition(&incomplete, |result| {
            assert_eq!(result.normal_text, "prefix");
            assert!(result.calls.is_empty());
        });
        let value = "literal </｜DSML｜ invoke> and </｜DSML｜invoke> here";
        every_partition(&wrap(spaced, &invoke(spaced, "inspect", value)), |result| {
            assert_eq!(result.calls.len(), 1);
            assert_eq!(
                serde_json::from_str::<Value>(&result.calls[0].arguments).unwrap()["value"],
                value
            );
        });
    }
    every_partition("<think>plain</think><｜DSML", |result| {
        assert_eq!(result.normal_text, "<think>plain</think><｜DSML");
        assert!(result.calls.is_empty());
    });
}
