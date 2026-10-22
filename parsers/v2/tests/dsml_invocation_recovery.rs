// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_parsers_v2::{
    ToolParseResult, UnifiedEvent, UnifiedParserExt, UnifiedParserInit, assemble,
    create_tool_parser_for_family, create_unified_parser_for_family,
};
use serde_json::{Value, json};

fn invoke(name: &str, body: &str) -> String {
    format!("<｜DSML｜ invoke name=\"{name}\">{body}</｜DSML｜ invoke>")
}

fn parameter(name: &str, value: &str) -> String {
    format!("<｜DSML｜ parameter name=\"{name}\" string=\"true\">{value}</｜DSML｜ parameter>")
}

fn check(input: &str, expected: &[(&str, Value)]) {
    let ends: Vec<_> = input
        .char_indices()
        .map(|(at, _)| at)
        .chain([input.len()])
        .collect();
    let mut chunks: Vec<Vec<&str>> = ends
        .iter()
        .map(|&at| vec![&input[..at], &input[at..]])
        .collect();
    chunks.push(ends.windows(2).map(|w| &input[w[0]..w[1]]).collect());
    for chunks in chunks {
        let mut parser = create_unified_parser_for_family("deepseek_v41", &[]).unwrap();
        parser
            .initialize_request(UnifiedParserInit::default())
            .unwrap();
        let mut events = Vec::new();
        for chunk in &chunks {
            events.extend(parser.push(chunk).unwrap());
        }
        events.extend(parser.finish().unwrap().events);
        let wanted: Vec<_> = expected
            .iter()
            .map(|(name, args)| UnifiedEvent::ToolCall {
                name: (*name).into(),
                arguments: args.clone(),
            })
            .collect();
        assert_eq!(assemble(&events), wanted, "{chunks:?}");

        let mut legacy = create_tool_parser_for_family("deepseek_v4", &[]).unwrap();
        let mut output = ToolParseResult::default();
        for chunk in &chunks {
            output.append(legacy.push(chunk).unwrap());
        }
        output.append(legacy.finish().unwrap());
        let output = output.coalesce_calls();
        assert_eq!(output.calls.len(), expected.len(), "legacy {chunks:?}");
        for (index, (actual, (name, args))) in output.calls.iter().zip(expected).enumerate() {
            assert_eq!(actual.tool_index, index);
            assert_eq!(actual.name.as_deref(), Some(*name));
            assert_eq!(
                serde_json::from_str::<Value>(&actual.arguments).unwrap(),
                *args
            );
            assert!(actual.complete);
        }
    }
}

#[test]
fn redundant_parameter_closers_preserve_every_argument() {
    let close = "</｜DSML｜ parameter>";
    let body = format!(
        "{close}\n{}{close}\n{close}{}{close}",
        parameter("command", "printf 'Café🙂'"),
        parameter("author", "bot")
    );
    let input = format!("<｜DSML｜ calls>{}</｜DSML｜ calls>", invoke("Bash", &body));
    check(
        &input,
        &[("Bash", json!({"command":"printf 'Café🙂'", "author":"bot"}))],
    );
}

#[test]
fn supplied_bash_sample_with_extra_parameter_close_is_recovered() {
    let command = "cd /home/devpod/multica/repositories/_frontend_service_repos/fund-high-finance-55b46a2ba015/submit && sed -n '1,80p' FW-10586-technical-solution.md";
    let body = format!(
        "{}\n{}\n</｜DSML｜ parameter>\n{}",
        parameter("command", command),
        parameter("description", "Read migration doc"),
        parameter("author", "bot")
    );
    let input = format!("<｜DSML｜ calls>{}</｜DSML｜ calls>", invoke("Bash", &body));
    check(
        &input,
        &[(
            "Bash",
            json!({"command":command, "description":"Read migration doc", "author":"bot"}),
        )],
    );
}

#[test]
fn malformed_invokes_are_isolated_in_wrapped_and_bare_sequences() {
    for body in [
        "{bad}",
        "{\"a\":",
        "{} trailing junk",
        "plain text",
        "<｜DSML｜ parameter name=\"x\" string=\"maybe\">42</｜DSML｜ parameter>",
    ] {
        let calls = format!(
            "{}{}{}{}{}",
            invoke("first", "{}"),
            invoke("bad", body),
            invoke("", "{}"),
            invoke("bad_again", body),
            invoke("last", &parameter("value", "你好🙂"))
        );
        for wrapped in [true, false] {
            let input = if wrapped {
                format!("<｜DSML｜ calls>{calls}</｜DSML｜ calls>")
            } else {
                calls.clone()
            };
            check(
                &input,
                &[("first", json!({})), ("last", json!({"value":"你好🙂"}))],
            );
        }
    }
}

#[test]
fn failed_calls_in_separate_blocks_do_not_block_other_blocks() {
    let input = format!(
        "<｜DSML｜ calls>{}</｜DSML｜ calls><｜DSML｜ calls>{}</｜DSML｜ calls><｜DSML｜ calls>{}</｜DSML｜ calls>",
        invoke("first", "{}"),
        invoke("bad", "{bad}"),
        invoke("last", "{}")
    );
    check(&input, &[("first", json!({})), ("last", json!({}))]);
}

#[test]
fn later_call_is_emitted_without_waiting_for_outer_close() {
    let mut parser = create_unified_parser_for_family("deepseek_v41", &[]).unwrap();
    let bad = invoke(
        "bad",
        "<｜DSML｜ parameter name=\"x\" string=\"maybe\">42</｜DSML｜ parameter>",
    );
    let input = format!("<｜DSML｜ calls>{bad}{}", invoke("good", "{}"));
    assert_eq!(
        assemble(&parser.push(&input).unwrap()),
        vec![UnifiedEvent::ToolCall {
            name: "good".into(),
            arguments: json!({})
        }]
    );
    assert!(parser.push("</｜DSML｜ calls>").unwrap().is_empty());
    assert!(parser.finish().unwrap().events.is_empty());
}
