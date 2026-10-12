// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Parse whole model outputs with the v2 Unified parser.
//!
//! Reads JSONL jobs on stdin, `{id, text, family, starting_state, tools}`. Writes a
//! header line naming the linked dynamo-parsers-v2 version, then one JSONL line per
//! job: its assembled events in model order, or `{id, error}` when the parser fails
//! or panics. A malformed job or an unknown family stops the run before any job is
//! parsed. `conformance/utils/src/parser_roundtrip` drives it.
//!
//! Each text is fed whole through `parse_into` and `finish` with native tool output,
//! so a pass here says nothing about chunk boundaries, guided tool choice or
//! Dynamo's serving adapter.

use std::io::{BufRead, BufWriter, Write};
use std::panic::catch_unwind;

use anyhow::{Context, Result, anyhow, ensure};
use dynamo_parsers_v2::{
    Tool, UnifiedEvent, UnifiedParserInit, UnifiedParserOutput, UnifiedParserStartingState,
    assemble_with_tool_indices, builtin_unified_families, canonical_unified_family,
    create_unified_parser_for_family,
};
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{Value, json};

const PARSER: &str = "dynamo-parsers-v2";
const SCHEMA: &str = "parser-roundtrip-events/v1";

/// The channel the rendered generation prompt left open; the caller decides it.
#[derive(Deserialize)]
#[serde(remote = "UnifiedParserStartingState", rename_all = "snake_case")]
enum StartingStateDef {
    None,
    Reasoning,
    Response,
}

/// An OpenAI tool object; the parser reads only its function.
#[derive(Deserialize)]
struct RequestTool {
    function: Function,
}

#[derive(Deserialize)]
struct Function {
    name: String,
    #[serde(default)]
    parameters: Value,
    strict: Option<bool>,
}

/// OpenAI tool objects as Dynamo's `to_v2_tools` flattens them: no description,
/// which no v2 parser reads, and missing parameters as null.
fn function_tools<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> std::result::Result<Vec<Tool>, D::Error> {
    let tools = Vec::<RequestTool>::deserialize(deserializer)?;
    let tool = |RequestTool { function }| Tool {
        name: function.name,
        description: None,
        parameters: function.parameters,
        strict: function.strict,
    };
    Ok(tools.into_iter().map(tool).collect())
}

#[derive(Deserialize)]
struct Job {
    id: String,
    text: String,
    family: String,
    #[serde(with = "StartingStateDef")]
    starting_state: UnifiedParserStartingState,
    #[serde(default, deserialize_with = "function_tools")]
    tools: Vec<Tool>,
}

#[derive(Debug, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum Event {
    Reasoning {
        text: String,
    },
    Text {
        text: String,
    },
    ToolCall {
        index: usize,
        name: String,
        /// The parser's argument string, keys in the order it wrote them; a
        /// `Value` would sort them.
        arguments: String,
        id: Option<String>,
    },
}

#[derive(Debug, Serialize)]
struct Parsed<'a> {
    id: &'a str,
    events: Vec<Event>,
    /// Calls that never completed, which the assembled events leave out.
    incomplete_calls: usize,
}

#[derive(Serialize)]
struct Failure<'a> {
    id: &'a str,
    error: String,
}

/// The version of `package` in a Cargo lock file; it must be listed once.
fn locked_version<'a>(lock: &'a str, package: &str) -> Result<&'a str> {
    let mut versions = lock.split("[[package]]").filter_map(|entry| {
        let field = |name: &str| {
            entry.lines().find_map(|line| {
                line.strip_prefix(name)?
                    .strip_prefix(" = \"")?
                    .strip_suffix('"')
            })
        };
        if field("name")? != package {
            return None;
        }
        field("version")
    });
    let version = versions
        .next()
        .with_context(|| format!("{package} is not in Cargo.lock"))?;
    ensure!(versions.next().is_none(), "ambiguous {package} package");
    Ok(version)
}

/// Every job on `input`. Fails on the first malformed job or unknown family.
fn read_jobs(input: impl BufRead) -> Result<Vec<Job>> {
    let mut jobs = Vec::new();
    for (index, line) in input.lines().enumerate() {
        let line = line.with_context(|| format!("job line {}", index + 1))?;
        if line.trim().is_empty() {
            continue;
        }
        // serde's message can quote the job's text, so report only where it failed.
        let job: Job = serde_json::from_str(&line).map_err(|error| {
            anyhow!(
                "job line {} is malformed: {:?} at column {}",
                index + 1,
                error.classify(),
                error.column()
            )
        })?;
        ensure!(
            canonical_unified_family(&job.family).is_some(),
            "job line {}: unknown family {:?}; built-in: {:?}",
            index + 1,
            job.family,
            builtin_unified_families()
        );
        jobs.push(job);
    }
    Ok(jobs)
}

fn parse(job: &Job) -> Result<Parsed<'_>> {
    let mut parser = create_unified_parser_for_family(&job.family, &job.tools)?;
    parser.initialize_request(UnifiedParserInit {
        starting_state: job.starting_state,
        ..UnifiedParserInit::default()
    })?;
    let mut output = UnifiedParserOutput::default();
    parser.parse_into(&job.text, &mut output)?;
    output.append(&mut parser.finish()?);

    let mut arguments = output.tool_arguments_raw();
    let events = assemble_with_tool_indices(&output.events)
        .into_iter()
        .map(|(index, event)| match event {
            UnifiedEvent::Reasoning { text } => Ok(Event::Reasoning { text }),
            UnifiedEvent::Text { text } => Ok(Event::Text { text }),
            UnifiedEvent::ToolCall { name, .. } => {
                let index = index.context("an assembled call has no tool index")?;
                Ok(Event::ToolCall {
                    index,
                    name,
                    arguments: arguments
                        .remove(&index)
                        .context("an assembled call has no argument bytes")?,
                    id: parser.tool_call_id(index).map(str::to_owned),
                })
            }
        })
        .collect::<Result<_>>()?;
    Ok(Parsed {
        id: &job.id,
        events,
        // Every opened call has an argument entry and each assembled call took its
        // own, so the rest belong to the incomplete calls `assemble` dropped.
        incomplete_calls: arguments.len(),
    })
}

/// Writes the job's result line. A parser error or panic belongs to its case, so
/// the remaining jobs still run.
fn write_result(out: &mut impl Write, job: &Job) -> Result<()> {
    let failure = |error: String| Failure { id: &job.id, error };
    match catch_unwind(|| parse(job)) {
        Ok(Ok(parsed)) => serde_json::to_writer(&mut *out, &parsed),
        Ok(Err(error)) => serde_json::to_writer(&mut *out, &failure(format!("{error:#}"))),
        Err(_) => serde_json::to_writer(&mut *out, &failure("parser panicked".to_owned())),
    }?;
    writeln!(out)?;
    Ok(())
}

fn main() -> Result<()> {
    // A panic payload can quote the case text, so report only where it happened.
    std::panic::set_hook(Box::new(|info| match info.location() {
        Some(location) => eprintln!("parser panicked at {location}"),
        None => eprintln!("parser panicked"),
    }));
    let version = locked_version(include_str!("../../Cargo.lock"), PARSER)?;
    let jobs = read_jobs(std::io::stdin().lock())?;
    let mut out = BufWriter::new(std::io::stdout().lock());
    let header = json!({"schema": SCHEMA, "parser": PARSER, "version": version});
    writeln!(out, "{header}")?;
    for job in &jobs {
        write_result(&mut out, job)?;
    }
    out.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_parsers_v2::{UnifiedParser, register_unified_parser};

    fn job(family: &str, starting_state: &str, text: &str, tools: Value) -> Job {
        serde_json::from_value(json!({
            "id": "case",
            "text": text,
            "family": family,
            "starting_state": starting_state,
            "tools": tools,
        }))
        .expect("the job is valid")
    }

    fn tool(name: &str, properties: Value) -> Value {
        json!({"type": "function", "function": {
            "name": name,
            "parameters": {"type": "object", "properties": properties},
        }})
    }

    fn weather() -> Value {
        json!([tool("get_weather", json!({"city": {"type": "string"}}))])
    }

    fn reasoning(text: &str) -> Event {
        Event::Reasoning { text: text.into() }
    }

    fn text(text: &str) -> Event {
        Event::Text { text: text.into() }
    }

    /// The events and the incomplete-call count the parser gives one job.
    fn parsed(
        family: &str,
        starting_state: &str,
        text: &str,
        tools: Value,
    ) -> Result<(Vec<Event>, usize)> {
        let job = job(family, starting_state, text, tools);
        let parsed = parse(&job)?;
        Ok((parsed.events, parsed.incomplete_calls))
    }

    #[test]
    fn deepseek_v4_call_after_a_prompt_opened_reasoning() -> Result<()> {
        // The text after a `<think>` generation prompt, as DeepSeek V4 writes a call.
        let dsml = "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"get_weather\">\n\
                    <｜DSML｜parameter name=\"city\" string=\"true\">Paris</｜DSML｜parameter>\n\
                    </｜DSML｜invoke>\n</｜DSML｜tool_calls>";
        let input = format!("Check the city.</think>Looking.{dsml}");
        let (events, incomplete) = parsed("deepseek_v4", "reasoning", &input, weather())?;
        let [
            first,
            second,
            Event::ToolCall {
                name, arguments, ..
            },
        ] = events.as_slice()
        else {
            anyhow::bail!("expected reasoning, text and a call: {events:?}");
        };
        ensure!(*first == reasoning("Check the city."));
        ensure!(*second == text("Looking."));
        ensure!(name == "get_weather");
        ensure!(serde_json::from_str::<Value>(arguments)? == json!({"city": "Paris"}));
        ensure!(incomplete == 0);
        // Without the starting state the same text holds no reasoning.
        let (events, _) = parsed("deepseek_v4", "none", &input, weather())?;
        let is_reasoning = |event: &Event| matches!(event, Event::Reasoning { .. });
        ensure!(!events.iter().any(is_reasoning), "{events:?}");
        Ok(())
    }

    #[test]
    fn qwen3_arguments_keep_their_key_order() -> Result<()> {
        let input = "<tool_call>\n<function=run>\n<parameter=zeta>\n1\n</parameter>\n\
                     <parameter=alpha>\nx\n</parameter>\n</function>\n</tool_call>";
        let properties = json!({"alpha": {"type": "string"}, "zeta": {"type": "integer"}});
        let (events, _) = parsed("qwen3", "none", input, json!([tool("run", properties)]))?;
        let [Event::ToolCall { arguments, .. }] = events.as_slice() else {
            anyhow::bail!("expected one call: {events:?}");
        };
        ensure!(serde_json::from_str::<Value>(arguments)? == json!({"zeta": 1, "alpha": "x"}));
        let (zeta, alpha) = (arguments.find("zeta"), arguments.find("alpha"));
        ensure!(zeta < alpha, "key order changed: {arguments}");
        Ok(())
    }

    #[test]
    fn incomplete_calls_are_counted_and_left_out() -> Result<()> {
        let input = "Hi.<tool_call>\n<function=get_weather>\n<parameter=city>\nPar";
        let (events, incomplete) = parsed("qwen3", "none", input, weather())?;
        ensure!(incomplete == 1, "{events:?}");
        let is_call = |event: &Event| matches!(event, Event::ToolCall { .. });
        ensure!(!events.iter().any(is_call), "{events:?}");
        Ok(())
    }

    #[test]
    fn a_bad_job_stops_the_run_without_quoting_its_text() -> Result<()> {
        let fields = r#""id": "a", "text": "secret""#;
        for (line, expected) in [
            // A data error's message would quote the value.
            (
                format!(r#"{{{fields}, "family": "qwen3", "starting_state": "secret"}}"#),
                "job line 1 is malformed",
            ),
            (
                format!(r#"{{{fields}, "family": "qwen3"}}"#),
                "job line 1 is malformed",
            ),
            (
                format!(r#"{{{fields}, "family": "nope", "starting_state": "none"}}"#),
                "job line 1: unknown family \"nope\"",
            ),
        ] {
            let error = match read_jobs(line.as_bytes()) {
                Ok(_) => anyhow::bail!("accepted {line}"),
                Err(error) => format!("{error:#}"),
            };
            ensure!(error.contains(expected), "{error}");
            ensure!(!error.contains("secret"), "{error}");
        }
        Ok(())
    }

    struct Panics;

    impl UnifiedParser for Panics {
        fn parse_into(&mut self, delta: &str, _: &mut UnifiedParserOutput) -> Result<()> {
            panic!("quoting {delta}")
        }

        fn finish(&mut self) -> Result<UnifiedParserOutput> {
            Ok(UnifiedParserOutput::default())
        }
    }

    #[test]
    fn a_panic_answers_its_case_with_an_error() -> Result<()> {
        register_unified_parser("roundtrip_panics", |_| Ok(Box::new(Panics)));
        let mut line = Vec::new();
        write_result(
            &mut line,
            &job("roundtrip_panics", "none", "secret", json!([])),
        )?;
        ensure!(
            serde_json::from_slice::<Value>(&line)?
                == json!({"id": "case", "error": "parser panicked"})
        );
        Ok(())
    }

    #[test]
    fn the_workspace_lock_names_the_parser_once() -> Result<()> {
        locked_version(include_str!("../../Cargo.lock"), PARSER)?;
        Ok(())
    }
}
