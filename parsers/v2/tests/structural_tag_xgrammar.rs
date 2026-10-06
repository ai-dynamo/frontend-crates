// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Executable native structural-tag contracts. Run explicitly with --ignored.
use dynamo_parsers_v2::structural_tag::{
    ReasoningBoundary, StructuralTagContext, StructuralTagOptions, StructuralTagSchemaMode,
    StructuralTagToolChoice,
};
use dynamo_parsers_v2::{
    Tool, UnifiedEvent, UnifiedParserExt, UnifiedParserInit, UnifiedParserStartingState,
    UnifiedToolOutputMode, assemble, create_unified_parser_for_family,
    structural_tag_builder_for_family,
};
use serde_json::{Value, json};
use std::{
    io::Write,
    process::{Command, Stdio},
};

struct Case {
    id: String,
    family: &'static str,
    grammar: Value,
    input: String,
    // None means grammar-negative, not parser-negative.
    expected: Option<Vec<UnifiedEvent>>,
    tools: Vec<Tool>,
    reasoning: bool,
}

fn tools() -> Vec<Tool> {
    ["weather", "other"]
        .into_iter()
        .map(|name| Tool {
            name: name.into(),
            description: None,
            strict: Some(true),
            parameters: json!({"type":"object", "properties":{
            "city":{"type":"string", "enum":["Paris", "東京"]},
            "count":{"type":"integer"}}, "required":["city", "count"],
            "additionalProperties":false}),
        })
        .collect()
}

// Hand-authored native syntax, independent of the grammar AST and parser output.
fn arguments(family: &str, city: &str, count: &str) -> String {
    match family {
        "deepseek_v4" => format!(
            "<｜DSML｜parameter name=\"city\" string=\"true\">{city}</｜DSML｜parameter>\n<｜DSML｜parameter name=\"count\" string=\"false\">{count}</｜DSML｜parameter>\n"
        ),
        "qwen3_coder" => format!(
            "<parameter=city>\n{city}\n</parameter>\n<parameter=count>\n{count}\n</parameter>"
        ),
        "glm47" => format!(
            "<arg_key>city</arg_key><arg_value>{city}</arg_value><arg_key>count</arg_key><arg_value>{count}</arg_value>"
        ),
        _ => unreachable!(),
    }
}

fn call(family: &str, name: &str, args: &str) -> String {
    match family {
        "deepseek_v4" => format!("<｜DSML｜invoke name=\"{name}\">\n{args}</｜DSML｜invoke>\n"),
        "qwen3_coder" => {
            format!("<tool_call>\n<function={name}>\n{args}\n</function>\n</tool_call>")
        }
        "glm47" => format!("<tool_call>{name}{args}</tool_call>"),
        _ => unreachable!(),
    }
}
fn block(family: &str, calls: &[String]) -> String {
    match family {
        "deepseek_v4" => format!(
            "\n\n<｜DSML｜tool_calls>\n{}</｜DSML｜tool_calls>",
            calls.concat()
        ),
        "qwen3_coder" => calls.join("\n"),
        "glm47" => calls.concat(),
        _ => unreachable!(),
    }
}
fn events(family: &str, city: &str, names: &[&str]) -> Vec<UnifiedEvent> {
    let mut result = Vec::new();
    // Current Unified extraction preserves the DSML block's leading separator.
    if family == "deepseek_v4" {
        result.push(UnifiedEvent::Text {
            text: "\n\n".into(),
        });
    }
    for (index, name) in names.iter().enumerate() {
        if family == "qwen3_coder" && index > 0 {
            result.push(UnifiedEvent::Text { text: "\n".into() });
        }
        result.push(UnifiedEvent::ToolCall {
            name: (*name).into(),
            arguments: json!({"city":city,"count":2}),
        });
    }
    result
}

fn cases() -> Vec<Case> {
    let mut cases = Vec::new();
    for family in ["deepseek_v4", "qwen3_coder", "glm47"] {
        let tools = tools();
        let builder = structural_tag_builder_for_family(family).unwrap();
        let base = StructuralTagContext {
            tool_choice: StructuralTagToolChoice::Required,
            tools: &tools,
            parallel_tool_calls: Some(false),
            schema_mode: StructuralTagSchemaMode::Auto,
            structured_output_schema: None,
            starts_in_reasoning: false,
        };
        let options = StructuralTagOptions::default();
        let one = block(
            family,
            &[call(family, "weather", &arguments(family, "Paris", "2"))],
        );
        let two = block(
            family,
            &[
                call(family, "weather", &arguments(family, "Paris", "2")),
                call(family, "other", &arguments(family, "Paris", "2")),
            ],
        );
        let mut add = |label: &str,
                       context: StructuralTagContext<'_>,
                       options: StructuralTagOptions,
                       input: String,
                       expected: Option<Vec<UnifiedEvent>>| {
            cases.push(Case {
                id: format!("{family}/{label}"),
                family,
                grammar: builder
                    .build_with_options(&context, &options)
                    .unwrap()
                    .expect("constrained request"),
                input,
                expected,
                tools: context.tools.to_vec(),
                reasoning: context.starts_in_reasoning
                    && options.reasoning_boundary == ReasoningBoundary::StructuralTag,
            });
        };
        for (label, choice) in [
            ("required", StructuralTagToolChoice::Required),
            ("named", StructuralTagToolChoice::Named("weather")),
            ("auto", StructuralTagToolChoice::Auto),
        ] {
            let context = StructuralTagContext {
                tool_choice: choice,
                ..base
            };
            add(
                label,
                context,
                options,
                one.clone(),
                Some(events(family, "Paris", &["weather"])),
            );
            add(
                &format!("{label}-truncated"),
                context,
                options,
                one[..one.len() - 1].into(),
                None,
            );
            add(
                &format!("{label}-wrong-name"),
                context,
                options,
                one.replace("weather", "unknown"),
                None,
            );
            add(
                &format!("{label}-parallel-disabled"),
                context,
                options,
                two.clone(),
                None,
            );
        }
        add("required-missing", base, options, String::new(), None);
        add("required-text", base, options, "hello".into(), None);
        add(
            "auto-text",
            StructuralTagContext {
                tool_choice: StructuralTagToolChoice::Auto,
                ..base
            },
            options,
            "hello".into(),
            Some(vec![UnifiedEvent::Text {
                text: "hello".into(),
            }]),
        );
        let none = StructuralTagContext {
            tool_choice: StructuralTagToolChoice::None,
            ..base
        };
        add(
            "none-text",
            none,
            options,
            "hello".into(),
            Some(vec![UnifiedEvent::Text {
                text: "hello".into(),
            }]),
        );
        add("none-call", none, options, one.clone(), None);
        add(
            "parallel",
            StructuralTagContext {
                parallel_tool_calls: Some(true),
                ..base
            },
            options,
            two.clone(),
            Some(events(family, "Paris", &["weather", "other"])),
        );
        add(
            "named-other",
            StructuralTagContext {
                tool_choice: StructuralTagToolChoice::Named("weather"),
                ..base
            },
            options,
            one.replace("weather", "other"),
            None,
        );
        for (label, args) in [
            ("enum", arguments(family, "London", "2")),
            ("type", arguments(family, "Paris", "\"bad\"")),
            ("required-property", String::new()),
            (
                "extra-property",
                format!(
                    "{}{}",
                    arguments(family, "Paris", "2"),
                    arguments(family, "Paris", "2")
                        .replace("city", "extra_city")
                        .replace("count", "extra_count")
                ),
            ),
        ] {
            add(
                label,
                base,
                options,
                block(family, &[call(family, "weather", &args)]),
                None,
            );
        }
        add(
            "unicode",
            base,
            options,
            block(
                family,
                &[call(family, "weather", &arguments(family, "東京", "2"))],
            ),
            Some(events(family, "東京", &["weather"])),
        );
        let suffix = if family == "qwen3_coder" { "\n\n" } else { "" };
        let reasoning = StructuralTagContext {
            starts_in_reasoning: true,
            ..base
        };
        let mut expected = vec![UnifiedEvent::Reasoning {
            text: "checking".into(),
        }];
        if family == "qwen3_coder" {
            expected.push(UnifiedEvent::Text {
                text: suffix.into(),
            });
        }
        expected.extend(events(family, "Paris", &["weather"]));
        add(
            "reasoning",
            reasoning,
            options,
            format!("checking</think>{suffix}{one}"),
            Some(expected),
        );
        add(
            "reasoning-unclosed",
            reasoning,
            options,
            "checking".into(),
            None,
        );
        add(
            "external-reasoning",
            reasoning,
            StructuralTagOptions {
                reasoning_boundary: ReasoningBoundary::External,
                ..options
            },
            one.clone(),
            Some(events(family, "Paris", &["weather"])),
        );
        let schema = json!({"type":"object","properties":{"ok":{"type":"boolean"}},"required":["ok"],"additionalProperties":false});
        let response = StructuralTagContext {
            tool_choice: StructuralTagToolChoice::Auto,
            structured_output_schema: Some(&schema),
            ..base
        };
        add(
            "response",
            response,
            options,
            "{\"ok\":true}".into(),
            Some(vec![UnifiedEvent::Text {
                text: "{\"ok\":true}".into(),
            }]),
        );
        add(
            "response-tool",
            response,
            options,
            one.clone(),
            Some(events(family, "Paris", &["weather"])),
        );
        add(
            "response-invalid",
            response,
            options,
            "{\"ok\":3}".into(),
            None,
        );
        add(
            "response-mixed",
            response,
            options,
            format!("{{\"ok\":true}}{one}"),
            None,
        );
        let loose_tools: Vec<_> = tools
            .iter()
            .cloned()
            .map(|mut tool| {
                tool.strict = Some(false);
                tool
            })
            .collect();
        let loose = StructuralTagContext {
            tools: &loose_tools,
            ..base
        };
        add(
            "auto-nonstrict",
            loose,
            options,
            one.replace("Paris", "London"),
            Some(events(family, "London", &["weather"])),
        );
        add(
            "forced-strict",
            StructuralTagContext {
                schema_mode: StructuralTagSchemaMode::Strict,
                ..loose
            },
            options,
            one.replace("Paris", "London"),
            None,
        );
        let null_tools: Vec<_> = tools
            .iter()
            .cloned()
            .map(|mut tool| {
                tool.parameters = Value::Null;
                tool
            })
            .collect();
        let mut null_expected = events(family, "London", &["weather"]);
        // Qwen/GLM use the schema to type XML values; DSML carries string flags.
        if family != "deepseek_v4"
            && let Some(UnifiedEvent::ToolCall { arguments, .. }) = null_expected.last_mut()
        {
            arguments["count"] = json!("2");
        }
        add(
            "null-schema",
            StructuralTagContext {
                tools: &null_tools,
                ..base
            },
            options,
            one.replace("Paris", "London"),
            Some(null_expected),
        );
        add(
            "any-order",
            base,
            StructuralTagOptions {
                tool_arguments_any_order: true,
                ..options
            },
            one.clone(),
            Some(events(family, "Paris", &["weather"])),
        );
        let args = arguments(family, "Paris", "2");
        let count_marker = match family {
            "deepseek_v4" => "<｜DSML｜parameter name=\"count\"",
            "qwen3_coder" => "\n<parameter=count>",
            "glm47" => "<arg_key>count",
            _ => unreachable!(),
        };
        let city_only = args.split_once(count_marker).unwrap().0;
        let missing_count = block(family, &[call(family, "weather", city_only)]);
        add(
            "ordered-missing-required",
            base,
            options,
            missing_count.clone(),
            None,
        );
        let any_order = StructuralTagOptions {
            tool_arguments_any_order: true,
            ..options
        };
        // XGrammar keeps a minimum entry count equal to the number of required
        // keys, but permits duplicates to satisfy it while a required key is absent.
        add(
            "any-order-too-few-entries",
            base,
            any_order,
            missing_count.clone(),
            None,
        );
        let separator = if family == "qwen3_coder" { "\n" } else { "" };
        let repeated_city = block(
            family,
            &[call(
                family,
                "weather",
                &format!("{city_only}{separator}{city_only}"),
            )],
        );
        let count_only = &args[city_only.len()..];
        let reversed = block(
            family,
            &[call(
                family,
                "weather",
                &format!(
                    "{}{separator}{city_only}",
                    count_only.trim_start_matches('\n')
                ),
            )],
        );
        add("ordered-reversed", base, options, reversed.clone(), None);
        add(
            "any-order-reversed",
            base,
            any_order,
            reversed,
            Some(events(family, "Paris", &["weather"])),
        );
        let mut empty_events = events(family, "Paris", &["weather"]);
        if let Some(UnifiedEvent::ToolCall { arguments, .. }) = empty_events.last_mut() {
            *arguments = json!({"city":"Paris"});
        }
        add(
            "any-order-duplicate-missing-required",
            base,
            StructuralTagOptions {
                tool_arguments_any_order: true,
                ..options
            },
            repeated_city,
            Some(empty_events),
        );
        let auto = StructuralTagContext {
            tool_choice: StructuralTagToolChoice::Auto,
            ..base
        };
        let bare_marker = match family {
            "deepseek_v4" => "<｜DSML｜invoke name=\"weather\">",
            "qwen3_coder" => "<function=weather>",
            "glm47" => "<arg_key>city</arg_key>",
            _ => unreachable!(),
        };
        add(
            "bare-call-marker",
            auto,
            options,
            format!("hello {bare_marker}"),
            None,
        );
        add(
            "excluded-marker",
            auto,
            options,
            "hello </think>".into(),
            None,
        );
        add(
            "allowed-marker",
            auto,
            StructuralTagOptions {
                exclude_special_tokens: Some(false),
                ..options
            },
            "hello </think>".into(),
            Some(vec![UnifiedEvent::Text {
                // Native extraction consumes the reasoning control marker.
                text: "hello ".into(),
            }]),
        );
    }
    cases
}

fn assert_parser(case: &Case) {
    let expected = case.expected.as_ref().unwrap();
    let boundaries: Vec<_> = case
        .input
        .char_indices()
        .map(|(i, _)| i)
        .chain([case.input.len()])
        .collect();
    let mut chunkings = vec![vec![case.input.as_str()]];
    chunkings.push(
        boundaries
            .windows(2)
            .map(|w| &case.input[w[0]..w[1]])
            .collect(),
    );
    for &at in &boundaries {
        chunkings.push(vec![&case.input[..at], &case.input[at..]]);
    }
    for (index, chunks) in chunkings.into_iter().enumerate() {
        let mut parser = create_unified_parser_for_family(case.family, &case.tools).unwrap();
        parser
            .initialize_request(UnifiedParserInit {
                starting_state: if case.reasoning {
                    UnifiedParserStartingState::Reasoning
                } else {
                    UnifiedParserStartingState::None
                },
                tool_output_mode: UnifiedToolOutputMode::Native,
                ..Default::default()
            })
            .unwrap();
        let mut deltas = Vec::new();
        for chunk in chunks {
            deltas.extend(parser.push(chunk).unwrap());
        }
        deltas.extend(parser.finish().unwrap().events);
        assert_eq!(
            &assemble(&deltas),
            expected,
            "parser {} chunking {index}",
            case.id
        );
    }
}

#[test]
#[ignore = "requires pinned Python environment; see tests/structural_tag/README.md"]
fn structural_tag_xgrammar() {
    let cases = cases();
    assert!(cases.len() >= 100, "empty or incomplete matrix");
    let requests: Vec<_> = cases
        .iter()
        .map(|c| json!({"id":c.id,"grammar":c.grammar,"input":c.input}))
        .collect();
    let python = std::env::var_os("XGRAMMAR_PYTHON").unwrap_or_else(|| "python3".into());
    let mut child = Command::new(python)
        .arg(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/structural_tag/worker.py"
        ))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("start XGrammar worker");
    let mut stdin = child.stdin.take().unwrap();
    // Writing in a thread avoids a pipe deadlock if the worker emits a large result.
    let writer = std::thread::spawn(move || {
        stdin
            .write_all(&serde_json::to_vec(&requests).unwrap())
            .unwrap();
    });
    let output = child.wait_with_output().unwrap();
    writer.join().unwrap();
    assert!(
        output.status.success(),
        "XGrammar worker failed: {}",
        output.status
    );
    let results: Vec<Value> = serde_json::from_slice(&output.stdout).expect("worker JSON");
    assert_eq!(results.len(), cases.len(), "worker skipped cases");
    let mut failures = Vec::new();
    for (case, result) in cases.iter().zip(results) {
        assert_eq!(result["id"], case.id);
        if case.id.ends_with("/named-truncated") {
            assert_eq!(
                result["accepted"], true,
                "negative control must be a valid prefix: {}",
                case.id
            );
            assert_eq!(
                result["complete"], false,
                "prefix acceptance is not completion"
            );
        }
        if result["complete"] != case.expected.is_some() {
            failures.push(format!("grammar {}: {result}", case.id));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));

    for case in cases.iter().filter(|c| c.expected.is_some()) {
        assert_parser(case);
    }
    eprintln!(
        "{} grammar cases passed; all positive cases passed Unified chunkings",
        cases.len()
    );
}

#[test]
fn authored_positive_samples_match_unified() {
    for case in cases().iter().filter(|case| case.expected.is_some()) {
        assert_parser(case);
    }
}

// Schema rejection is a generation constraint. Unified still extracts native
// syntax, including a city outside the declared enum.
#[test]
fn grammar_negative_schema_is_still_extractable() {
    for mut case in cases()
        .into_iter()
        .filter(|case| case.id.ends_with("/enum"))
    {
        assert!(case.expected.is_none());
        case.expected = Some(events(case.family, "London", &["weather"]));
        assert_parser(&case);
    }
}
