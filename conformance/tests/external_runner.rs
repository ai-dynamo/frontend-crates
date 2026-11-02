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
