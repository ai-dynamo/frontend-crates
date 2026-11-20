// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// A Muse prose quotation cannot span a message boundary (`<|eom|>` or `<|eot|>`
// followed by `<|start|>`). Prose quotes are inert by default, so each test re-runs
// itself with `DYNAMO_PARSERS_DISABLE_QUOTE_AWARE_PARSING=0`, the compatibility
// mode where an open quote holds control markers literal.

#[path = "support/quote_policy.rs"]
mod quote_policy;

use dynamo_parsers_v2::{
    Tool, UnifiedEvent, UnifiedParserExt, assemble, create_unified_parser_for_family,
};

fn tools() -> Vec<Tool> {
    ["get_weather", "commentary"]
        .iter()
        .map(|name| Tool {
            name: (*name).to_string(),
            description: None,
            parameters: serde_json::json!({"type": "object", "properties": {"city": {"type": "string"}}}),
            strict: None,
        })
        .collect()
}

fn parse(family: &str, chunks: &[&str]) -> Vec<UnifiedEvent> {
    let mut parser = create_unified_parser_for_family(family, &tools()).unwrap();
    let mut events = Vec::new();
    for chunk in chunks {
        events.extend(parser.push(chunk).unwrap());
    }
    events.extend(parser.finish().unwrap().events);
    assemble(&events)
}

fn assert_every_chunking(family: &str, input: &str, expected: &[UnifiedEvent]) {
    assert_eq!(parse(family, &[input]), expected, "{family} whole input");
    let characters: Vec<String> = input.chars().map(String::from).collect();
    let characters: Vec<&str> = characters.iter().map(String::as_str).collect();
    assert_eq!(
        parse(family, &characters),
        expected,
        "{family} one character at a time"
    );
    for split in (1..input.len()).filter(|&at| input.is_char_boundary(at)) {
        assert_eq!(
            parse(family, &[&input[..split], &input[split..]]),
            expected,
            "{family} split at byte {split}"
        );
    }
}

fn reasoning(text: &str) -> UnifiedEvent {
    UnifiedEvent::Reasoning { text: text.into() }
}

fn text(text: &str) -> UnifiedEvent {
    UnifiedEvent::Text { text: text.into() }
}

fn call(name: &str, arguments: serde_json::Value) -> UnifiedEvent {
    UnifiedEvent::ToolCall {
        name: name.into(),
        arguments,
    }
}

fn tool_channel(name: &str, key: &str, value: &str) -> String {
    format!(
        "<|start|>assistant to={name}<|message|><atem:function_calls>\n\
         <atem:invoke name=\"{name}\">\n<atem:parameter name=\"{key}\">{value}</atem:parameter>\n\
         </atem:invoke>\n</atem:function_calls><|eom|>"
    )
}

#[test]
fn a_thought_ending_inside_an_open_quote_still_closes_at_eom() {
    if !quote_policy::quote_aware_process(
        "a_thought_ending_inside_an_open_quote_still_closes_at_eom",
    ) {
        return;
    }
    // An unmatched quote before `<|eom|>` must not hold it literal and pull the
    // following messages into the thought.
    for quote in ['`', '"', '\''] {
        let input = format!(
            "<|start|>assistant to=self<|message|>x {quote}<|eom|>\
             <|start|>assistant to=commentary<|message|>a {quote}b{quote}<|eom|>\
             <|start|>assistant to=user<|message|>Done.<|eot|>"
        );
        assert_every_chunking(
            "muse_spark",
            &input,
            &[
                reasoning(&format!("x {quote}")),
                text(&format!("a {quote}b{quote}Done.")),
            ],
        );
    }
}

#[test]
fn an_open_quote_in_content_does_not_swallow_the_next_call() {
    if !quote_policy::quote_aware_process("an_open_quote_in_content_does_not_swallow_the_next_call")
    {
        return;
    }
    // The later backtick would close the quote: without the boundary, the call and
    // the answer's opening became literal commentary text.
    let input = format!(
        "<|start|>assistant to=commentary<|message|>Use `x<|eom|>{}\
         <|start|>assistant to=user<|message|>Said `done`<|eot|>",
        tool_channel("get_weather", "city", "Paris")
    );
    assert_every_chunking(
        "muse_spark",
        &input,
        &[
            text("Use `x"),
            call("get_weather", serde_json::json!({"city": "Paris"})),
            text("Said `done`"),
        ],
    );
}

#[test]
fn an_unmatched_quote_does_not_span_a_glimmer_message_boundary() {
    if !quote_policy::quote_aware_process(
        "an_unmatched_quote_does_not_span_a_glimmer_message_boundary",
    ) {
        return;
    }
    // A quote left open at `<|eom|>` must not pair with a quote in the next message
    // and turn that message's framing into reasoning text.
    let input = format!(
        "<|start|>assistant to=self<|message|>x `<|eom|>{}\
         <|start|>assistant to=user<|message|>a `b`<|eot|>",
        tool_channel("get_weather", "city", "`Paris`")
    );
    assert_every_chunking(
        "muse_glimmer",
        &input,
        &[
            reasoning("x `"),
            call("get_weather", serde_json::json!({"city": "`Paris`"})),
            text("a `b`"),
        ],
    );
}

#[test]
fn a_quoted_terminator_before_a_header_reads_the_same_at_every_split() {
    if !quote_policy::quote_aware_process(
        "a_quoted_terminator_before_a_header_reads_the_same_at_every_split",
    ) {
        return;
    }
    // Missing-`<|eom|>` recovery with a quoted `<|eom|>`: the first terminator whose
    // next control marker is `<|start|>` is the boundary, whether or not that header
    // has arrived when the quote's closing character does. The quote opened before
    // that `<|eom|>` is unmatched there; the debris between the terminator and the
    // header stays visible.
    let input = "<|start|>assistant to=self<|message|>I said \"<|eom|>\" now\
                 <|start|>assistant to=user<|message|>Hi<|eot|>";
    for family in ["muse_spark", "muse_glimmer"] {
        assert_every_chunking(family, input, &[reasoning("I said \""), text("\" nowHi")]);
    }
}
