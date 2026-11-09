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
            "auto-parallel",
            StructuralTagContext {
                tool_choice: StructuralTagToolChoice::Auto,
                parallel_tool_calls: Some(true),
                ..base
            },
            options,
            two.clone(),
            Some(events(family, "Paris", &["weather", "other"])),
        );
        add(
            "auto-call-exclusions-disabled",
            StructuralTagContext {
                tool_choice: StructuralTagToolChoice::Auto,
                ..base
            },
            StructuralTagOptions {
                exclude_special_tokens: Some(false),
                ..options
            },
            one.clone(),
            Some(events(family, "Paris", &["weather"])),
        );
        if family == "qwen3_coder" {
            for (label, choice) in [
                (
                    "auto-missing-function-equals",
                    StructuralTagToolChoice::Auto,
                ),
                (
                    "required-missing-function-equals",
                    StructuralTagToolChoice::Required,
                ),
            ] {
                add(
                    label,
                    StructuralTagContext {
                        tool_choice: choice,
                        ..base
                    },
                    options,
                    one.replace("<function=", "<function"),
                    None,
                );
            }
            add(
                "parallel-missing-second-function-equals",
                StructuralTagContext {
                    parallel_tool_calls: Some(true),
                    ..base
                },
                options,
                two.replace("<function=other", "<functionother"),
                None,
            );
        }
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
#[ignore = "requires lean Python verifier; see tests/check_structural_tag_xgrammar.py"]
fn structural_tag_xgrammar() {
    let cases = cases();
    assert!(cases.len() >= 100, "empty or incomplete matrix");
    let requests: Vec<_> = cases
        .iter()
        .map(|c| json!({"id":c.id,"grammar":c.grammar,"input":c.input}))
        .collect();
    let results = run_xgrammar(requests);
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

fn run_xgrammar(requests: Vec<Value>) -> Vec<Value> {
    let expected_count = requests.len();
    let python = std::env::var_os("CONFORMANCE_GRAMMAR_PYTHON").unwrap_or_else(|| "python3".into());
    let mut child = Command::new(python)
        .arg(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/check_structural_tag_xgrammar.py"
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
    assert_eq!(results.len(), expected_count, "worker skipped cases");
    results
}

#[test]
#[ignore = "requires lean Python verifier; see tests/check_structural_tag_xgrammar.py"]
fn harmony_trigger_exclusions_xgrammar() {
    // The engine owns Harmony's <|call|> stop token, so it is intentionally
    // absent from the grammar terminals. These checks qualify the grammar;
    // the XML/DSML suite separately qualifies complete native parser inputs.
    let call = |name: &str| {
        format!(
            "<|channel|>commentary to=functions.{name}<|constrain|>json<|message|>{{\"city\":\"Paris\",\"count\":2}}"
        )
    };
    let one = call("weather");
    let two = format!("{one}<|start|>assistant{}", call("other"));
    let tools = tools();
    let mut requests = Vec::new();
    let mut expected = Vec::new();
    for family in ["harmony", "harmony_text", "gpt_oss"] {
        let builder = structural_tag_builder_for_family(family).unwrap();
        for excluded in [None, Some(false), Some(true)] {
            for (choice, mode) in [
                (StructuralTagToolChoice::Auto, "auto"),
                (StructuralTagToolChoice::Required, "required"),
                (StructuralTagToolChoice::Named("weather"), "named"),
            ] {
                for parallel in [false, true] {
                    let grammar = builder
                        .build_with_options(
                            &StructuralTagContext {
                                tool_choice: choice,
                                tools: &tools,
                                parallel_tool_calls: Some(parallel),
                                schema_mode: StructuralTagSchemaMode::Auto,
                                structured_output_schema: None,
                                starts_in_reasoning: false,
                            },
                            &StructuralTagOptions {
                                exclude_special_tokens: excluded,
                                reasoning_boundary: ReasoningBoundary::External,
                                ..Default::default()
                            },
                        )
                        .unwrap()
                        .unwrap();
                    let mut samples = vec![
                        ("single", one.clone(), true),
                        ("wrong-name", call("unknown"), false),
                        (
                            "missing-recipient-dot",
                            one.replace("to=functions.", "to=functions"),
                            false,
                        ),
                    ];
                    if mode != "named" {
                        samples.push(("parallel", two.clone(), parallel));
                    }
                    if mode == "auto" {
                        samples.push((
                            "commentary-text",
                            "<|channel|>commentary<|message|>Checking the weather.".into(),
                            true,
                        ));
                        samples.push((
                            "bare-recipient-marker",
                            "hello to=functions.weather".into(),
                            excluded != Some(true),
                        ));
                    }
                    for (sample, input, accepted) in samples {
                        let id = format!(
                            "{family}/{mode}/exclusions-{excluded:?}/parallel-{parallel}/{sample}"
                        );
                        requests.push(json!({"id":id,"grammar":grammar,"input":input}));
                        expected.push((id, accepted));
                    }
                }
            }
        }
    }
    let results = run_xgrammar(requests);
    let mut failures = Vec::new();
    for ((id, accepted), result) in expected.iter().zip(results) {
        assert_eq!(&result["id"], id);
        if result["complete"] != *accepted {
            failures.push(format!("grammar {id}: {result}"));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    eprintln!("{} Harmony grammar cases passed", expected.len());
}

fn assert_no_trigger_exclusion_conflict(node: &Value, family: &str) {
    match node {
        Value::Object(fields) => {
            if let (Some(triggers), Some(excludes)) = (
                fields.get("triggers").and_then(Value::as_array),
                fields.get("excludes").and_then(Value::as_array),
            ) {
                for trigger in triggers.iter().filter_map(Value::as_str) {
                    for excluded in excludes.iter().filter_map(Value::as_str) {
                        assert!(
                            excluded.is_empty() || !trigger.contains(excluded),
                            "{family}: trigger {trigger:?} contains free-text exclusion {excluded:?}"
                        );
                    }
                }
            }
            for child in fields.values() {
                assert_no_trigger_exclusion_conflict(child, family);
            }
        }
        Value::Array(elements) => {
            for child in elements {
                assert_no_trigger_exclusion_conflict(child, family);
            }
        }
        _ => {}
    }
}

#[test]
fn registered_builders_have_no_trigger_exclusion_conflicts() {
    let families: std::collections::BTreeSet<_> = dynamo_parsers_v2::REGISTERED_FAMILIES
        .iter()
        .chain(dynamo_parsers_v2::unified::builtin_unified_families())
        .copied()
        .collect();
    let tools = tools();
    let answer_schema = json!({"type":"object","properties":{"ok":{"type":"boolean"}},"required":["ok"],"additionalProperties":false});
    for family in families {
        let Some(builder) = structural_tag_builder_for_family(family) else {
            continue;
        };
        let mut family_tools = tools.clone();
        if matches!(family, "minimax_m3" | "minimax-m3") {
            // M3 declines non-ASCII enum literals for guided JSON fallback.
            // Exercise trigger exclusions with a schema its native grammar accepts.
            for tool in &mut family_tools {
                tool.parameters["properties"]["city"]["enum"] = json!(["Paris", "London"]);
            }
        }
        for choice in [
            StructuralTagToolChoice::Auto,
            StructuralTagToolChoice::Required,
        ] {
            for starts_in_reasoning in [false, true] {
                for schema in [None, Some(&answer_schema)] {
                    for boundary in [
                        ReasoningBoundary::StructuralTag,
                        ReasoningBoundary::External,
                    ] {
                        let result = builder.build_with_options(
                            &StructuralTagContext {
                                tool_choice: choice,
                                tools: &family_tools,
                                parallel_tool_calls: Some(true),
                                schema_mode: StructuralTagSchemaMode::Auto,
                                structured_output_schema: schema,
                                starts_in_reasoning,
                            },
                            &StructuralTagOptions {
                                exclude_special_tokens: Some(true),
                                reasoning_boundary: boundary,
                                ..Default::default()
                            },
                        );
                        // Spark deliberately requires grammar-owned reasoning.
                        if matches!(family, "muse_spark" | "muse_spark_open")
                            && boundary == ReasoningBoundary::External
                        {
                            assert!(
                                result.is_err(),
                                "{family}: external reasoning must be rejected"
                            );
                            continue;
                        }
                        let grammar = result.unwrap().unwrap();
                        assert_no_trigger_exclusion_conflict(&grammar, family);
                    }
                }
            }
        }
    }
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

const V41_BAR: char = '\u{FF5C}';

fn v41_tool(parameters: Value) -> Tool {
    Tool {
        name: "reply".into(),
        description: None,
        strict: Some(true),
        parameters,
    }
}

fn text_tool() -> Tool {
    v41_tool(json!({
        "type":"object",
        "properties":{"text":{"type":"string"}},
        "required":["text"],
        "additionalProperties":false
    }))
}

fn v41_grammar(
    tool: &Tool,
    choice: StructuralTagToolChoice<'_>,
    exclude_special_tokens: bool,
    parallel_tool_calls: Option<bool>,
    structured_output_schema: Option<&Value>,
) -> Value {
    structural_tag_builder_for_family("deepseek_v41")
        .expect("deepseek_v41 builder")
        .build_with_options(
            &StructuralTagContext {
                tools: std::slice::from_ref(tool),
                tool_choice: choice,
                parallel_tool_calls,
                schema_mode: StructuralTagSchemaMode::Strict,
                structured_output_schema,
                starts_in_reasoning: false,
            },
            &StructuralTagOptions {
                exclude_special_tokens: Some(exclude_special_tokens),
                ..Default::default()
            },
        )
        .unwrap()
        .expect("constrained request")
}

fn v41_parameter(name: &str, string: bool, value: &str) -> String {
    let bar = V41_BAR;
    format!(
        "<{bar}DSML{bar} parameter name=\"{name}\" string=\"{string}\">{value}</{bar}DSML{bar} parameter>\n"
    )
}

fn v41_invoke(name: &str, arguments: &str) -> String {
    let bar = V41_BAR;
    format!("<{bar}DSML{bar} invoke name=\"{name}\">\n{arguments}</{bar}DSML{bar} invoke>\n")
}

fn v41_block(calls: &str) -> String {
    let bar = V41_BAR;
    format!("\n\n<{bar}DSML{bar} calls>\n{calls}</{bar}DSML{bar} calls>")
}

fn v41_arguments(text: &str) -> String {
    v41_parameter("text", true, text)
}

fn v41_input(text: &str) -> String {
    v41_block(&v41_invoke("reply", &v41_arguments(text)))
}

fn v41_open_input(text: &str) -> String {
    let bar = V41_BAR;
    format!(
        "\n\n<{bar}DSML{bar} calls>\n<{bar}DSML{bar} invoke name=\"reply\">\n<{bar}DSML{bar} parameter name=\"text\" string=\"true\">{text}"
    )
}

fn named_argument_schema(grammar: &Value) -> &Value {
    let format = &grammar["format"];
    let tags = if format["type"] == "tag" && format["content"]["type"] == "tags_with_separator" {
        &format["content"]["tags"]
    } else {
        &format["elements"][1]["tags"]
    };
    &tags.as_array().expect("tool call tags")[0]["content"]
}

fn check_cases(results: &[Value], expected: &[(String, bool, bool)]) {
    assert_eq!(expected.len(), results.len(), "worker skipped cases");
    let mut failures = Vec::new();
    for ((id, accepted, complete), result) in expected.iter().zip(results) {
        assert_eq!(&result["id"], id);
        if result["accepted"] != *accepted || result["complete"] != *complete {
            failures.push(format!("{id}: {result}"));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

fn v4_grammar(tool: &Tool) -> Value {
    structural_tag_builder_for_family("deepseek_v4")
        .expect("deepseek_v4 builder")
        .build_with_options(
            &StructuralTagContext {
                tools: std::slice::from_ref(tool),
                tool_choice: StructuralTagToolChoice::Named("reply"),
                parallel_tool_calls: Some(false),
                schema_mode: StructuralTagSchemaMode::Strict,
                structured_output_schema: None,
                starts_in_reasoning: false,
            },
            &StructuralTagOptions {
                exclude_special_tokens: Some(true),
                ..Default::default()
            },
        )
        .unwrap()
        .expect("constrained request")
}

fn v4_input(text: &str) -> String {
    let bar = V41_BAR;
    format!(
        "\n\n<{bar}DSML{bar}tool_calls>\n<{bar}DSML{bar}invoke name=\"reply\">\n<{bar}DSML{bar}parameter name=\"text\" string=\"true\">{text}</{bar}DSML{bar}parameter>\n</{bar}DSML{bar}invoke>\n</{bar}DSML{bar}tool_calls>"
    )
}

#[test]
#[ignore = "requires lean Python verifier; see tests/check_structural_tag_xgrammar.py"]
fn v41_argument_exclusion_xgrammar() {
    let tool = text_tool();
    let mut requests = Vec::new();
    let mut expected = Vec::new();
    for (mode, choice) in [
        ("auto", StructuralTagToolChoice::Auto),
        ("required", StructuralTagToolChoice::Required),
        ("named", StructuralTagToolChoice::Named("reply")),
    ] {
        for guarded in [true, false] {
            let grammar = v41_grammar(&tool, choice, guarded, Some(false), None);
            for (sample, input, guarded_case, free_case) in [
                ("valid", v41_input("hello 東京"), (true, true), (true, true)),
                ("empty", v41_input(""), (true, true), (true, true)),
                // A valid prefix stays acceptable; nothing force-closes it at a token limit.
                (
                    "unfinished",
                    v41_open_input("hello</"),
                    (true, false),
                    (true, false),
                ),
                (
                    "malformed-double-space",
                    v41_input("hello</｜  parameter> world"),
                    (false, false),
                    (true, true),
                ),
                (
                    "wrong-close-name",
                    v41_input("hello</｜ unknown> world"),
                    (false, false),
                    (true, true),
                ),
                (
                    "invoke-close-in-string",
                    v41_input("hello</｜ invoke> world"),
                    (false, false),
                    (true, true),
                ),
                (
                    "literal-marker",
                    v41_input("literal </｜ marker"),
                    (false, false),
                    (true, true),
                ),
                ("unicode", v41_input("東京🙂"), (true, true), (true, true)),
            ] {
                let id = format!("v41/{mode}/guarded={guarded}/{sample}");
                requests.push(json!({"id":id, "grammar":grammar, "input":input}));
                expected.push((id, guarded, guarded_case, free_case));
            }
        }
    }
    let results = run_xgrammar(requests);
    let supports = results[0]["excludes_supported"]
        .as_bool()
        .expect("worker capability");
    let adjusted: Vec<_> = expected
        .into_iter()
        .map(|(id, guarded, guarded_case, free_case)| {
            let case = if guarded && supports {
                guarded_case
            } else {
                free_case
            };
            (id, case.0, case.1)
        })
        .collect();
    check_cases(&results, &adjusted);
    eprintln!(
        "V4.1 argument exclusion cases passed on XGrammar {} (excludes supported: {supports})",
        results[0]["backend_version"].as_str().unwrap_or("unknown")
    );
}

#[test]
#[ignore = "requires lean Python verifier; see tests/check_structural_tag_xgrammar.py"]
fn v41_exclusion_token_boundaries_xgrammar() {
    let tool = text_tool();
    let bar = V41_BAR;
    let prefix = [
        format!("\n\n<{bar}DSML{bar} calls>\n"),
        format!("<{bar}DSML{bar} invoke name=\"reply\">\n"),
        format!("<{bar}DSML{bar} parameter name=\"text\" string=\"true\">"),
    ];
    let parameter_end = format!("</{bar}DSML{bar} parameter>");
    let suffix = format!("</{bar}DSML{bar} invoke>\n</{bar}DSML{bar} calls>");
    let mut requests = Vec::new();
    let mut expected = Vec::new();
    for guarded in [true, false] {
        let grammar = v41_grammar(
            &tool,
            StructuralTagToolChoice::Named("reply"),
            guarded,
            Some(false),
            None,
        );
        let cases = [
            // A token may cross the content/structural boundary and contain the complete
            // valid terminator, including the reserved prefix.
            (
                "content-and-terminator",
                prefix
                    .iter()
                    .cloned()
                    .chain([format!("hello{parameter_end}\n")])
                    .chain([suffix.clone()])
                    .collect::<Vec<_>>(),
                (true, true),
                (true, true),
            ),
            // The valid terminator may start a new token at the content boundary.
            (
                "terminator-starts-token",
                prefix
                    .iter()
                    .cloned()
                    .chain([
                        "hello".to_string(),
                        format!("{parameter_end}\n"),
                        suffix.clone(),
                    ])
                    .collect::<Vec<_>>(),
                (true, true),
                (true, true),
            ),
            // The reserved prefix may span tokens; malformed spacing is still rejected.
            (
                "malformed-close-split",
                prefix
                    .iter()
                    .cloned()
                    .chain([
                        "hello</".to_string(),
                        format!("{bar}  parameter>"),
                        " world".to_string(),
                        format!("{parameter_end}\n"),
                        suffix.clone(),
                    ])
                    .collect::<Vec<_>>(),
                (false, false),
                (true, true),
            ),
            // A literal marker split across tokens is representable without the guard.
            (
                "literal-marker-split",
                prefix
                    .iter()
                    .cloned()
                    .chain([
                        "hello</".to_string(),
                        format!("{bar} marker"),
                        format!("{parameter_end}\n"),
                        suffix.clone(),
                    ])
                    .collect::<Vec<_>>(),
                (false, false),
                (true, true),
            ),
            // A token may end inside the valid terminator after `</`; XGrammar 0.2.8
            // accepts the completed closing marker with the exclusion active.
            (
                "terminator-split-after-slash",
                prefix
                    .iter()
                    .cloned()
                    .chain([
                        "hello</".to_string(),
                        format!("{bar}DSML{bar} parameter>\n"),
                        suffix.clone(),
                    ])
                    .collect::<Vec<_>>(),
                (true, true),
                (true, true),
            ),
            // A token may end after the reserved prefix; XGrammar 0.2.8 accepts the
            // completed closing marker with the exclusion active.
            (
                "terminator-split-after-prefix",
                prefix
                    .iter()
                    .cloned()
                    .chain([
                        format!("hello</{bar}"),
                        format!("DSML{bar} parameter>\n"),
                        suffix.clone(),
                    ])
                    .collect::<Vec<_>>(),
                (true, true),
                (true, true),
            ),
            // The reserved prefix may begin a valid structural ending.
            (
                "unfinished-terminator",
                prefix
                    .iter()
                    .cloned()
                    .chain([format!("hello</{bar}")])
                    .collect::<Vec<_>>(),
                (true, false),
                (true, false),
            ),
        ];
        for (sample, tokens, guarded_case, free_case) in cases {
            let id = format!("v41/tokens/guarded={guarded}/{sample}");
            requests.push(json!({"id":id, "grammar":grammar, "tokens":tokens}));
            expected.push((id, guarded, guarded_case, free_case));
        }
    }
    let results = run_xgrammar(requests);
    let supports = results[0]["excludes_supported"]
        .as_bool()
        .expect("worker capability");
    let adjusted: Vec<_> = expected
        .into_iter()
        .map(|(id, guarded, guarded_case, free_case)| {
            let case = if guarded && supports {
                guarded_case
            } else {
                free_case
            };
            (id, case.0, case.1)
        })
        .collect();
    check_cases(&results, &adjusted);
}

#[test]
#[ignore = "requires lean Python verifier; see tests/check_structural_tag_xgrammar.py"]
fn v41_exclusion_length_bounds_xgrammar() {
    let schema = json!({
        "type":"object",
        "properties":{"text":{"type":"string","minLength":2,"maxLength":3}},
        "required":["text"],
        "additionalProperties":false
    });
    let tool = v41_tool(schema);
    let guarded = v41_grammar(
        &tool,
        StructuralTagToolChoice::Named("reply"),
        true,
        Some(false),
        None,
    );
    let unguarded = v41_grammar(
        &tool,
        StructuralTagToolChoice::Named("reply"),
        false,
        Some(false),
        None,
    );
    assert_eq!(
        named_argument_schema(&guarded)["excludes"],
        json!(["</｜"]),
        "the guard must be emitted for the bounded schema"
    );
    assert_eq!(
        named_argument_schema(&guarded)["json_schema"]["properties"]["text"]["minLength"],
        2
    );
    assert!(named_argument_schema(&unguarded).get("excludes").is_none());

    let mut requests = Vec::new();
    let mut expected = Vec::new();
    for (label, grammar, guarded_mode) in [
        ("guarded", &guarded, true),
        ("unguarded", &unguarded, false),
    ] {
        for (sample, text, free_ok) in [
            ("empty", "", false),
            ("one", "a", false),
            ("two", "ab", true),
            ("three", "東京🙂", true),
            ("four", "abcd", false),
        ] {
            let id = format!("v41/bounds/{label}/{sample}");
            requests.push(json!({"id":id, "grammar":grammar, "input":v41_input(text)}));
            // XGrammar 0.2.8 drops minLength/maxLength when exclusions are active.
            expected.push((id, guarded_mode, true, free_ok));
        }
    }
    let results = run_xgrammar(requests);
    let supports = results[0]["excludes_supported"]
        .as_bool()
        .expect("worker capability");
    let adjusted: Vec<_> = expected
        .into_iter()
        .map(|(id, guarded_mode, guarded_ok, free_ok)| {
            let accepted = if guarded_mode && supports {
                guarded_ok
            } else {
                free_ok
            };
            (id, accepted, accepted)
        })
        .collect();
    check_cases(&results, &adjusted);
}

#[test]
#[ignore = "requires lean Python verifier; see tests/check_structural_tag_xgrammar.py"]
fn v41_argument_exclusion_does_not_leak_xgrammar() {
    let tool = text_tool();
    let v4 = v4_grammar(&tool);
    assert!(
        v4["format"]["content"]["tags"][0]["content"]
            .get("excludes")
            .is_none(),
        "V4 must not inherit V4.1 argument exclusions"
    );
    let v4_input = v4_input("hello</｜  parameter>world");

    let response_schema = json!({
        "type":"object",
        "properties":{"text":{"type":"string"}},
        "required":["text"],
        "additionalProperties":false
    });
    let tool = text_tool();
    let v41_response = v41_grammar(
        &tool,
        StructuralTagToolChoice::Auto,
        true,
        Some(false),
        Some(&response_schema),
    );
    assert!(
        v41_response["format"]["elements"][1]
            .get("excludes")
            .is_none(),
        "a structured-output response schema must not inherit argument exclusions"
    );
    let response_input = json!({"text":"literal </｜ marker"}).to_string();

    let requests = vec![
        json!({"id":"v4/argument-marker", "grammar":v4, "input":v4_input}),
        json!({"id":"v41/response-marker", "grammar":v41_response, "input":response_input}),
    ];
    let results = run_xgrammar(requests);
    check_cases(
        &results,
        &[
            ("v4/argument-marker".to_string(), true, true),
            ("v41/response-marker".to_string(), true, true),
        ],
    );
}
