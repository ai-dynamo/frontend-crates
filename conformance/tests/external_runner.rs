// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeSet;

use anyhow::Result;
use dynamo_parsers_v2::{
    Tool, ToolCallDelta, UnifiedParser, UnifiedParserOutput, register_unified_parser,
    unregister_unified_parser,
};

use dynamo_conformance_fixtures_v2::{RunOptions, run_unified_family};

static SERIALIZE: std::sync::Mutex<()> = std::sync::Mutex::new(());

struct Restore(&'static str);

impl Drop for Restore {
    fn drop(&mut self) {
        unregister_unified_parser(self.0);
    }
}

#[derive(Default)]
struct WeatherParser {
    emitted: bool,
}

impl UnifiedParser for WeatherParser {
    fn parse_into(&mut self, _delta: &str, output: &mut UnifiedParserOutput) -> Result<()> {
        if !self.emitted {
            output.push_call(ToolCallDelta {
                tool_index: 0,
                name: Some("get_weather".to_owned()),
                arguments: r#"{"city":"Paris"}"#.to_owned(),
                complete: true,
            });
            self.emitted = true;
        }
        Ok(())
    }

    fn finish(&mut self) -> Result<UnifiedParserOutput> {
        Ok(UnifiedParserOutput::default())
    }
}

fn weather_factory(_tools: &[Tool]) -> Result<Box<dyn UnifiedParser>> {
    Ok(Box::new(WeatherParser::default()))
}

#[derive(Default)]
struct TextParser;

impl UnifiedParser for TextParser {
    fn parse_into(&mut self, delta: &str, output: &mut UnifiedParserOutput) -> Result<()> {
        output.push_text(delta);
        Ok(())
    }

    fn finish(&mut self) -> Result<UnifiedParserOutput> {
        Ok(UnifiedParserOutput::default())
    }
}

fn text_factory(_tools: &[Tool]) -> Result<Box<dyn UnifiedParser>> {
    Ok(Box::new(TextParser))
}

#[test]
fn external_factory_runs_authored_golden_and_streaming_checks() {
    let _serial = SERIALIZE.lock().unwrap_or_else(|error| error.into_inner());
    register_unified_parser("qwen3", weather_factory);
    let _restore = Restore("qwen3");
    let options = RunOptions {
        case_ids: Some(BTreeSet::from(["UNIFIED.1-tool_only".to_owned()])),
        check_stream_isolation: false,
        ..RunOptions::default()
    };

    let report = run_unified_family("qwen3", &options).unwrap();

    assert_eq!(report.selected, 1);
    assert_eq!(report.executed, 1);
    assert_eq!(report.passed, 1);
    assert!(report.failures.is_empty(), "{:?}", report.failures);
    assert_eq!(report.exit_code(false), 0);
}

#[test]
fn incorrect_external_factory_fails_and_streams_remain_isolated() {
    let _serial = SERIALIZE.lock().unwrap_or_else(|error| error.into_inner());
    register_unified_parser("qwen3", text_factory);
    let _restore = Restore("qwen3");
    let options = RunOptions {
        case_ids: Some(BTreeSet::from([
            "UNIFIED.1-tool_only".to_owned(),
            "UNIFIED.10-reason_only".to_owned(),
        ])),
        ..RunOptions::default()
    };

    let report = run_unified_family("qwen3", &options).unwrap();

    assert_eq!(report.selected, 2);
    assert_eq!(report.executed, 2);
    assert!(!report.failures.is_empty());
    assert_eq!(report.isolation_pairs, 1);
    assert!(
        report.isolation_failures.is_empty(),
        "{:?}",
        report.isolation_failures
    );
    assert_eq!(report.exit_code(false), 1);
}

#[test]
fn generation_constraints_are_reported_as_unsupported() {
    let options = RunOptions {
        case_ids: Some(BTreeSet::from([
            "UNIFIED.36-structured_response_after_reasoning".to_owned(),
        ])),
        check_every_split: false,
        check_stream_batch: false,
        check_stream_isolation: false,
    };

    let report = run_unified_family("qwen3", &options).unwrap();

    assert_eq!(report.selected, 1);
    assert_eq!(report.executed, 0);
    assert_eq!(report.unsupported.len(), 1);
    assert!(report.unsupported[0].reason.contains("structural-tag"));
    assert_eq!(report.exit_code(false), 0);
    assert_eq!(report.exit_code(true), 1);
}

#[test]
fn numeric_yaml_chunks_reach_the_glm_parser_without_changing_their_spelling() {
    let options = RunOptions {
        case_ids: Some(BTreeSet::from([
            "UNIFIED.7-arg_numeric_14_const_exponent".to_owned()
        ])),
        ..RunOptions::default()
    };

    let report = run_unified_family("glm47", &options).unwrap();

    assert_eq!(report.selected, 1);
    assert_eq!(report.executed, 1);
    assert_eq!(report.passed, 1, "{report:#?}");
}

#[test]
fn composed_schema_types_match_authored_golden_at_every_split() {
    let _serial = SERIALIZE.lock().unwrap_or_else(|error| error.into_inner());
    let options = RunOptions {
        case_ids: Some(BTreeSet::from([
            "UNIFIED.7-conditional_schema_number_types".to_owned(),
            "UNIFIED.7-not_keyword_type_complement".to_owned(),
        ])),
        ..RunOptions::default()
    };
    for family in ["beam", "glm47", "minimax_m3", "qwen3"] {
        let report = run_unified_family(family, &options).unwrap();
        assert_eq!(report.passed, 2, "{report:#?}");
        assert_eq!(report.exit_code(true), 0, "{report:#?}");
    }
}

#[test]
fn every_native_family_reports_every_active_case() {
    let _serial = SERIALIZE.lock().unwrap_or_else(|error| error.into_inner());
    let families: BTreeSet<_> = dynamo_parsers_v2::builtin_unified_families()
        .iter()
        .map(|family| dynamo_parsers_v2::canonical_unified_family(family).unwrap())
        .collect();
    let options = RunOptions {
        check_every_split: false,
        check_stream_isolation: false,
        ..RunOptions::default()
    };
    let mut errors = Vec::new();
    for family in families {
        match run_unified_family(family, &options) {
            Ok(report) => {
                println!(
                    "{family}: selected={} executed={} passed={} unsupported={} failures={}",
                    report.selected,
                    report.executed,
                    report.passed,
                    report.unsupported.len(),
                    report.failures.len()
                );
                // A parser mismatch is a report outcome. A loader or batch-parity
                // failure prevents an external author from evaluating their parser.
                let failed_cases: BTreeSet<_> = report
                    .failures
                    .iter()
                    .map(|failure| &failure.case_id)
                    .collect();
                assert_eq!(
                    report.executed + report.unsupported.len(),
                    report.selected,
                    "{report:#?}"
                );
                assert_eq!(
                    report.passed + failed_cases.len(),
                    report.executed,
                    "{report:#?}"
                );
                if report
                    .failures
                    .iter()
                    .any(|failure| failure.check != "golden")
                {
                    errors.push(format!("{report:#?}"));
                }
            }
            Err(error) => errors.push(format!("{family}: {error:#}")),
        }
    }
    assert!(errors.is_empty(), "{}", errors.join("\n"));
}
