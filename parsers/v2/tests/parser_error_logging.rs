// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use dynamo_parsers_v2::{
    UnifiedEvent, UnifiedParserExt, UnifiedParserInit, assemble, create_unified_parser_for_family,
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
        let events = self.0.lock().unwrap();
        let errors: Vec<_> = events.iter().filter(|e| e.level == Level::ERROR).collect();
        assert_eq!(errors.len(), 1, "{events:?}");
        assert_eq!(errors[0].fields["family"], "deepseek_v41");
        assert_eq!(errors[0].fields["tool_index"], "0");
        assert!(errors[0].fields[field].contains(expected), "{events:?}");
    }

    fn assert_no_errors(&self) {
        let events = self.0.lock().unwrap();
        assert!(events.iter().all(|e| e.level != Level::ERROR), "{events:?}");
    }
}

fn invoke(name: &str, body: &str) -> String {
    format!("<｜DSML｜ invoke name=\"{name}\">{body}</｜DSML｜ invoke>")
}

fn wrapped(invoke: &str) -> String {
    format!("<｜DSML｜ calls>{invoke}</｜DSML｜ calls>")
}

#[test]
fn push_failure_logs_error_and_preserves_recovery_bytes() {
    let capture = Capture::default();
    capture.run(|| {
        let mut parser = create_unified_parser_for_family("deepseek_v41", &[]).unwrap();
        parser
            .initialize_request(UnifiedParserInit::default())
            .unwrap();
        let input = wrapped(&invoke("", ""));
        assert!(parser.push(&input).is_err());
        assert_eq!(parser.reset(), input);
    });
    capture.assert_failure("error", "empty DeepSeek V4.1 tool name");
    assert_eq!(
        capture
            .0
            .lock()
            .unwrap()
            .iter()
            .find(|e| e.level == Level::ERROR)
            .unwrap()
            .fields["stage"],
        "push"
    );
}

#[test]
fn invalid_flag_logs_only_when_finalized() {
    let capture = Capture::default();
    capture.run(|| {
        let mut parser = create_unified_parser_for_family("deepseek_v41", &[]).unwrap();
        let input = wrapped(&invoke(
            "run",
            "<｜DSML｜ parameter name=\"x\" string=\"maybe\">x</｜DSML｜ parameter>",
        ));
        parser.push(&input).unwrap();
        capture.assert_no_errors();
        assert!(parser.finish().is_err());
    });
    capture.assert_failure("error", "invalid DeepSeek V4.1 parameter header");
    assert_eq!(
        capture
            .0
            .lock()
            .unwrap()
            .iter()
            .find(|e| e.level == Level::ERROR)
            .unwrap()
            .fields["stage"],
        "finish"
    );
}

#[test]
fn deferred_json_failure_logs_cause_at_finish() {
    let capture = Capture::default();
    capture.run(|| {
        let mut parser = create_unified_parser_for_family("deepseek_v41", &[]).unwrap();
        parser.push(&wrapped(&invoke("run", "{\"a\":"))).unwrap();
        capture.assert_no_errors();
        assert!(parser.finish().is_err());
    });
    capture.assert_failure("error", "invalid DeepSeek V4.1 JSON arguments");
    let events = capture.0.lock().unwrap();
    let error = events.iter().find(|e| e.level == Level::ERROR).unwrap();
    assert!(error.fields["error"].contains("EOF"));
    assert_eq!(error.fields["stage"], "finish");
}

#[test]
fn discarded_deferred_call_logs_error_and_keeps_later_call() {
    let capture = Capture::default();
    capture.run(|| {
        let mut parser = create_unified_parser_for_family("deepseek_v41", &[]).unwrap();
        let input = wrapped(&(invoke("bad", "{\"a\":") + &invoke("good", "{}")));
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
}

#[test]
fn truncated_call_logs_error_at_eof() {
    let capture = Capture::default();
    capture.run(|| {
        let mut parser = create_unified_parser_for_family("deepseek_v41", &[]).unwrap();
        assert!(
            parser
                .push("<｜DSML｜ calls><｜DSML｜ invoke name=\"run\">")
                .unwrap()
                .is_empty()
        );
        capture.assert_no_errors();
        assert!(parser.finish().unwrap().events.is_empty());
    });
    capture.assert_failure("why", "deepseek_v41_incomplete_invoke");
}

#[test]
fn successful_recovery_does_not_log_errors() {
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
    capture.assert_no_errors();
}
