// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Vendor parsers and the Nemotron V3 tool-only-turn rule (issue #406).
//!
//! Its own test binary: registering a vendor parser for `nemotron_v3` replaces the
//! built-in for the whole process, which would race the tests in `blank_tool_turn.rs`.

use anyhow::Result;
use dynamo_parsers_v2::{
    Tool, ToolCallDelta, UnifiedEvent, UnifiedParser, UnifiedParserExt, UnifiedParserOutput,
    assemble, create_unified_parser_for_family,
};

/// A vendor parser that echoes text, emits a call for `CALL`, and fails on `ERR`.
struct ScriptedVendor {
    calls: usize,
}

impl UnifiedParser for ScriptedVendor {
    fn parse_into(&mut self, delta: &str, output: &mut UnifiedParserOutput) -> Result<()> {
        match delta {
            "ERR" => anyhow::bail!("scripted failure"),
            "CALL" => {
                output.push_call(ToolCallDelta {
                    tool_index: self.calls,
                    name: Some("enter_digit".into()),
                    arguments: "{\"digit\":\"1\"}".into(),
                    complete: true,
                });
                self.calls += 1;
            }
            _ => output.push_text(delta),
        }
        Ok(())
    }

    fn finish(&mut self) -> Result<UnifiedParserOutput> {
        Ok(UnifiedParserOutput::default())
    }
}

fn scripted_vendor(_tools: &[Tool]) -> Result<Box<dyn UnifiedParser>> {
    Ok(Box::new(ScriptedVendor { calls: 0 }))
}

fn text(value: &str) -> UnifiedEvent {
    UnifiedEvent::Text { text: value.into() }
}

fn call() -> UnifiedEvent {
    UnifiedEvent::ToolCall {
        name: "enter_digit".into(),
        arguments: serde_json::json!({"digit": "1"}),
    }
}

fn tool_only_turn(family: &str) -> Vec<UnifiedEvent> {
    let mut parser = create_unified_parser_for_family(family, &[]).unwrap();
    let mut events = Vec::new();
    for chunk in ["\n", "CALL", "\n"] {
        events.extend(parser.push(chunk).unwrap());
    }
    events.extend(parser.finish().unwrap().events);
    assemble(&events)
}

/// A vendor parser registered for Nemotron V3 gets the same rule as the built-in it
/// shadows, so routing never decides the answer. A vendor family of its own is
/// returned unwrapped.
#[test]
fn vendor_parsers_follow_their_family_and_errors_keep_committed_text() {
    let own_family = "blank_tool_turn_vendor";
    dynamo_parsers_v2::register_unified_parser(own_family, scripted_vendor);
    assert_eq!(tool_only_turn(own_family), [text("\n"), call(), text("\n")]);
    dynamo_parsers_v2::unregister_unified_parser(own_family);

    dynamo_parsers_v2::register_unified_parser("nemotron_v3", scripted_vendor);
    for family in ["nemotron_v3", "nemotron3"] {
        assert_eq!(tool_only_turn(family), [call()], "{family}");

        let mut parser = create_unified_parser_for_family(family, &[]).unwrap();
        let events = parser.parse_complete("\n").unwrap();
        assert_eq!(
            events,
            [text("\n")],
            "{family}: whitespace without a call is kept"
        );
    }

    // On `Err`, held whitespace was committed by the parser, so it reaches the caller.
    let mut parser = create_unified_parser_for_family("nemotron_v3", &[]).unwrap();
    let mut output = UnifiedParserOutput::default();
    parser.parse_into("\n", &mut output).unwrap();
    assert!(
        output.is_empty(),
        "whitespace is held before the turn decides"
    );
    assert!(parser.parse_into("ERR", &mut output).is_err());
    assert_eq!(assemble(&output.events), [text("\n")]);

    // `reset` starts a fresh stream: nothing held carries over.
    parser
        .parse_into(" ", &mut UnifiedParserOutput::default())
        .unwrap();
    parser.reset();
    let events = parser.parse_complete("Hi").unwrap();
    assert_eq!(events, [text("Hi")]);

    dynamo_parsers_v2::unregister_unified_parser("nemotron_v3");
}
