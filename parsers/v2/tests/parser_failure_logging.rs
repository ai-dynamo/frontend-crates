// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use dynamo_parsers_v2::{
    Tool, UnifiedEvent, UnifiedParserExt, UnifiedParserInit, assemble,
    create_unified_parser_for_family,
};
use tracing::{Event, Level, Subscriber, field::Visit};
use tracing_subscriber::{Layer, Registry, layer::Context, prelude::*};

#[derive(Debug)]
struct RecordedEvent {
    level: Level,
    fields: BTreeMap<String, String>,
}

impl Visit for RecordedEvent {
    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
        self.fields.insert(field.name().into(), value.into());
    }

    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        self.fields
            .insert(field.name().into(), format!("{value:?}"));
    }
}

#[derive(Clone, Default)]
struct Capture(Arc<Mutex<Vec<RecordedEvent>>>);

impl<S: Subscriber> Layer<S> for Capture {
    fn on_event(&self, event: &Event<'_>, _ctx: Context<'_, S>) {
        let mut recorded = RecordedEvent {
            level: *event.metadata().level(),
            fields: BTreeMap::new(),
        };
        event.record(&mut recorded);
        self.0.lock().unwrap().push(recorded);
    }
}

impl Capture {
    fn run(&self, f: impl FnOnce()) {
        tracing::subscriber::with_default(Registry::default().with(self.clone()), f);
    }

    fn assert_failure(&self, field: &str, expected: &str) {
        self.assert_family_failure("deepseek_v41", field, expected);
    }

    fn assert_family_failure(&self, family: &str, field: &str, expected: &str) {
        let events = self.0.lock().unwrap();
        assert!(events.iter().all(|e| e.level != Level::ERROR), "{events:?}");
        let errors: Vec<_> = events.iter().filter(|e| e.level == Level::WARN).collect();
        assert_eq!(errors.len(), 1, "{events:?}");
        assert_eq!(errors[0].fields["family"], family);
        assert_eq!(errors[0].fields["tool_index"], "0");
        assert!(errors[0].fields[field].contains(expected), "{events:?}");
    }

    fn assert_no_warnings(&self) {
        let events = self.0.lock().unwrap();
        assert!(
            events
                .iter()
                .all(|e| e.level != Level::WARN && e.level != Level::ERROR),
            "{events:?}"
        );
    }
}

fn invoke(name: &str, body: &str) -> String {
    format!("<｜DSML｜ invoke name=\"{name}\">{body}</｜DSML｜ invoke>")
}

fn wrapped(invoke: &str) -> String {
    format!("<｜DSML｜ calls>{invoke}</｜DSML｜ calls>")
}

#[test]
fn recovered_declared_tool_header_logs_warning_with_original_bytes() {
    let capture = Capture::default();
    let input = "<｜DSML｜ parameter  name=\"Bash\"><｜DSML｜ parameter name=\"command\">pwd</｜DSML｜ parameter>";
    capture.run(|| {
        let tools = [Tool {
            name: "Bash".into(),
            description: None,
            strict: None,
            parameters: serde_json::json!({}),
        }];
        let mut parser = create_unified_parser_for_family("deepseek_v41", &tools).unwrap();
        parser.push(input).unwrap();
        capture.assert_no_warnings();
        assert_eq!(parser.finish().unwrap().assembled().len(), 1);
    });
    let events = capture.0.lock().unwrap();
    assert!(events.iter().all(|event| event.level != Level::ERROR));
    let recovered: Vec<_> = events
        .iter()
        .filter(|event| {
            event.fields.get("stage").map(String::as_str) == Some("invoke_header_recovery")
        })
        .collect();
    assert_eq!(recovered.len(), 1);
    assert_eq!(recovered[0].level, Level::WARN);
    assert_eq!(recovered[0].fields["tool_name"], "Bash");
    assert_eq!(recovered[0].fields["missing_invoke_end"], "true");
    assert_eq!(recovered[0].fields["raw_output"], input);
}

#[test]
fn discarded_call_logs_warning_before_consuming_its_bytes() {
    let capture = Capture::default();
    capture.run(|| {
        let mut parser = create_unified_parser_for_family("deepseek_v41", &[]).unwrap();
        parser
            .initialize_request(UnifiedParserInit::default())
            .unwrap();
        let input = wrapped(&invoke("", ""));
        assert!(parser.push(&input).unwrap().is_empty());
        assert!(parser.reset().is_empty());
        capture.assert_failure(
            "raw_output",
            input
                .strip_prefix("<｜DSML｜ calls>")
                .unwrap()
                .strip_suffix("</｜DSML｜ calls>")
                .unwrap(),
        );
    });
    capture.assert_failure("error", "empty DeepSeek V4.1 tool name");
    assert_eq!(
        capture
            .0
            .lock()
            .unwrap()
            .iter()
            .find(|e| e.level == Level::WARN)
            .unwrap()
            .fields["stage"],
        "invoke"
    );
}

#[test]
fn invalid_flag_logs_only_when_the_invoke_closes() {
    let capture = Capture::default();
    capture.run(|| {
        let mut parser = create_unified_parser_for_family("deepseek_v41", &[]).unwrap();
        let input = wrapped(&invoke(
            "run",
            "<｜DSML｜ parameter name=\"x\" string=\"maybe\">x</｜DSML｜ parameter>",
        ));
        let end = input.rfind("</｜DSML｜ invoke>").unwrap();
        parser.push(&input[..end]).unwrap();
        capture.assert_no_warnings();
        parser.push(&input[end..]).unwrap();
        assert!(parser.finish().unwrap().events.is_empty());
        capture.assert_failure(
            "raw_output",
            input
                .strip_prefix("<｜DSML｜ calls>")
                .unwrap()
                .strip_suffix("</｜DSML｜ calls>")
                .unwrap(),
        );
    });
    capture.assert_failure("error", "invalid DeepSeek V4.1 parameter header");
    assert_eq!(
        capture
            .0
            .lock()
            .unwrap()
            .iter()
            .find(|e| e.level == Level::WARN)
            .unwrap()
            .fields["stage"],
        "invoke"
    );
}

#[test]
fn discarded_json_failure_logs_cause() {
    let capture = Capture::default();
    capture.run(|| {
        let mut parser = create_unified_parser_for_family("deepseek_v41", &[]).unwrap();
        let input = wrapped(&invoke("run", "{\"a\":"));
        let end = input.rfind("</｜DSML｜ invoke>").unwrap();
        parser.push(&input[..end]).unwrap();
        capture.assert_no_warnings();
        parser.push(&input[end..]).unwrap();
        assert!(parser.finish().unwrap().events.is_empty());
        capture.assert_failure(
            "raw_output",
            input
                .strip_prefix("<｜DSML｜ calls>")
                .unwrap()
                .strip_suffix("</｜DSML｜ calls>")
                .unwrap(),
        );
    });
    capture.assert_failure("error", "invalid DeepSeek V4.1 JSON arguments");
    let events = capture.0.lock().unwrap();
    let error = events.iter().find(|e| e.level == Level::WARN).unwrap();
    assert!(error.fields["error"].contains("EOF"));
    assert_eq!(error.fields["stage"], "invoke");
}

#[test]
fn discarded_deferred_call_logs_warning_and_keeps_later_call() {
    let capture = Capture::default();
    capture.run(|| {
        let mut parser = create_unified_parser_for_family("deepseek_v41", &[]).unwrap();
        let bad = invoke("bad", "{\"a\":");
        let input = wrapped(&(bad.clone() + &invoke("good", "{}")));
        let mut events = parser.push(&input).unwrap();
        events.extend(parser.finish().unwrap().events);
        assert_eq!(
            assemble(&events),
            vec![UnifiedEvent::ToolCall {
                name: "good".into(),
                arguments: serde_json::json!({}),
            }]
        );
    });
    capture.assert_failure("error", "invalid DeepSeek V4.1 JSON arguments");
    capture.assert_failure("raw_output", &invoke("bad", "{\"a\":"));
    let events = capture.0.lock().unwrap();
    let warning = events.iter().find(|e| e.level == Level::WARN).unwrap();
    assert!(!warning.fields["raw_output"].contains("good"));
}

#[test]
fn truncated_call_logs_warning_at_eof() {
    let capture = Capture::default();
    capture.run(|| {
        let mut parser = create_unified_parser_for_family("deepseek_v41", &[]).unwrap();
        assert!(
            parser
                .push("<｜DSML｜ calls><｜DSML｜ invoke name=\"run\">")
                .unwrap()
                .is_empty()
        );
        capture.assert_no_warnings();
        assert!(parser.finish().unwrap().events.is_empty());
    });
    capture.assert_failure("why", "deepseek_v41_incomplete_invoke");
    capture.assert_failure(
        "raw_output",
        "<｜DSML｜ calls><｜DSML｜ invoke name=\"run\">",
    );
}

#[test]
fn successful_recovery_does_not_log_warnings() {
    let capture = Capture::default();
    capture.run(|| {
        let mut parser = create_unified_parser_for_family("deepseek_v41", &[]).unwrap();
        let input = wrapped(&invoke(
            "run",
            concat!(
                "<｜DSML｜ parameter name=\"x\" string=\"false\">not JSON</｜DSML｜ parameter>",
                "<｜DSML｜ parameter name=\"x\">latest</｜DSML｜ parameter>",
                "<｜DSML｜ parameter name=\"enabled\" string=\"false\">false</｜DSML｜ parameter>",
            ),
        ));
        let mut events = Vec::new();
        for ch in input.chars() {
            events.extend(parser.push(ch.encode_utf8(&mut [0; 4])).unwrap());
        }
        events.extend(parser.finish().unwrap().events);
        assert_eq!(
            assemble(&events),
            vec![UnifiedEvent::ToolCall {
                name: "run".into(),
                arguments: serde_json::json!({"x": "latest", "enabled": false}),
            }]
        );
    });
    capture.assert_no_warnings();
}

#[test]
fn warning_preserves_long_unicode_input_across_chunks_without_truncation() {
    let capture = Capture::default();
    let payload = "第一行\n\"引号\"\\路径🙂".repeat(4096);
    let input = wrapped(&invoke(
        "run",
        &format!(
            "<｜DSML｜ parameter name=\"value\" string=\"maybe\">{payload}</｜DSML｜ parameter>"
        ),
    ));
    capture.run(|| {
        let mut parser = create_unified_parser_for_family("deepseek_v41", &[]).unwrap();
        let split = input.find("引号").unwrap() + "引".len();
        parser.push(&input[..split]).unwrap();
        capture.assert_no_warnings();
        parser.push(&input[split..]).unwrap();
        assert!(parser.finish().unwrap().events.is_empty());
        assert!(parser.reset().is_empty());
    });
    capture.assert_failure("error", "invalid DeepSeek V4.1 parameter header");
    let events = capture.0.lock().unwrap();
    let warning = events.iter().find(|e| e.level == Level::WARN).unwrap();
    assert_eq!(
        warning.fields["raw_output"],
        input
            .strip_prefix("<｜DSML｜ calls>")
            .unwrap()
            .strip_suffix("</｜DSML｜ calls>")
            .unwrap()
    );
}

#[test]
fn warning_after_a_valid_call_preserves_failed_content_and_index() {
    let capture = Capture::default();
    let failed = invoke("", "{}");
    capture.run(|| {
        let mut parser = create_unified_parser_for_family("deepseek_v41", &[]).unwrap();
        parser.push(&wrapped(&invoke("good", "{}"))).unwrap();
        assert!(parser.push(&failed).unwrap().is_empty());
    });
    let events = capture.0.lock().unwrap();
    assert!(events.iter().all(|e| e.level != Level::ERROR));
    let warnings: Vec<_> = events.iter().filter(|e| e.level == Level::WARN).collect();
    assert_eq!(warnings.len(), 1);
    assert_eq!(warnings[0].fields["tool_index"], "1");
    assert_eq!(warnings[0].fields["raw_output"], failed);
}

#[test]
fn legacy_entry_point_also_logs_failed_input_at_warn() {
    let capture = Capture::default();
    let input = wrapped(&invoke("", "{}"));
    capture.run(|| {
        let mut parser =
            dynamo_parsers_v2::create_tool_parser_for_family("deepseek_v4", &[]).unwrap();
        assert!(parser.push(&input).unwrap().calls.is_empty());
    });
    capture.assert_family_failure("deepseek_v41", "error", "empty DeepSeek V4.1 tool name");
    capture.assert_failure("raw_output", &invoke("", "{}"));
}

#[test]
fn file_subscriber_persists_failed_content_at_warn() {
    let path = std::env::temp_dir().join(format!("parser-failure-{}.log", uuid::Uuid::new_v4()));
    let file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)
        .unwrap();
    let layer = tracing_subscriber::fmt::layer()
        .with_ansi(false)
        .with_writer(Arc::new(file))
        .with_filter(tracing_subscriber::filter::LevelFilter::WARN);
    let input = wrapped(&invoke(
        "run",
        "<｜DSML｜ parameter name=\"value\" string=\"maybe\">persistent_payload_原文\nsecond_line</｜DSML｜ parameter>",
    ));
    tracing::subscriber::with_default(Registry::default().with(layer), || {
        let mut parser = create_unified_parser_for_family("deepseek_v41", &[]).unwrap();
        parser.push(&input).unwrap();
        assert!(parser.finish().unwrap().events.is_empty());
    });
    let persisted = std::fs::read_to_string(&path).unwrap();
    std::fs::remove_file(path).unwrap();
    assert!(persisted.contains("WARN"), "{persisted}");
    assert!(!persisted.contains("ERROR"), "{persisted}");
    assert!(persisted.contains("raw_output="), "{persisted}");
    assert!(persisted.contains("persistent_payload_原文"), "{persisted}");
    assert!(persisted.contains("second_line"), "{persisted}");
    assert!(persisted.contains("invalid DeepSeek V4.1 parameter header"));
}
