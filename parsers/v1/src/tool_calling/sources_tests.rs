// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Source recording in the batch tool parsers that the jail calls.
//!
//! The sourced variants are crate-internal, so these tests live in the crate.
//! For `hermes`, `kimi_k2`, and `minimax_m3`, on every corpus text and every
//! prefix of it (the jail parses prefixes while it waits for a call to end):
//!
//! * Same output: the sourced variant returns the public function's calls and
//!   normal text, with recording on and off.
//! * Verbatim and tiling: the input bytes of the ranges, in order, equal the
//!   normal text.

// A list with one range is the expected value in many cases here.
#![allow(clippy::single_range_in_vec_init)]

use std::ops::Range;

use serde::Deserialize;

use super::ToolCallResponse;
use super::parsers::{
    detect_and_parse_tool_call_sourced, detect_and_parse_tool_call_with_recovery_sourced,
};
use super::tools::{try_tool_call_parse_aggregate, try_tool_call_parse_aggregate_finalize};
use crate::sources::SourcedText;

/// The sourced parse that the jail calls: with recovery at finalize.
async fn try_tool_call_parse_aggregate_sourced(
    message: &str,
    parser: Option<&str>,
    tools: Option<&[super::ToolDefinition]>,
    finalize: bool,
    record: bool,
) -> anyhow::Result<(Vec<ToolCallResponse>, Option<SourcedText>)> {
    if finalize {
        detect_and_parse_tool_call_with_recovery_sourced(message, parser, tools, record).await
    } else {
        detect_and_parse_tool_call_sourced(message, parser, tools, record).await
    }
}

pub(crate) const CORPUS: &str = include_str!("../../tests/data/source_ranges_corpus.json");

/// The batch parsers on Rhino's three v1 routes.
pub(crate) const RECORDING_TOOL_PARSERS: &[&str] = &["hermes", "kimi_k2", "minimax_m3"];

#[derive(Deserialize)]
pub(crate) struct Corpus {
    pub(crate) tool: Vec<ToolCase>,
}

#[derive(Deserialize)]
pub(crate) struct ToolCase {
    pub(crate) name: String,
    pub(crate) parser: String,
    pub(crate) chunks: Vec<String>,
}

pub(crate) fn corpus() -> Corpus {
    serde_json::from_str(CORPUS).unwrap()
}

/// Hand-written inputs for each normal-text path: prose beside a call, an
/// orphan closer, a truncated call, a call recovered without its wrapper, and
/// a parse with no call.
pub(crate) fn targeted(parser: &str) -> Vec<&'static str> {
    match parser {
        "hermes" => vec![
            r#"STOP {"name":"f","arguments":{}}"#,
            r#"STOP <tool_call>{"name":"f","arguments":{}}</tool_call> after"#,
            r#"  lead <tool_call>{"name":"f","arguments":{"a":1}}</tool_call><tool_call>{"name":"g","arguments":{}}</tool_call>"#,
            r#"{"x":"a</tool_call>b"}"#,
            r#"text {"x":1}</tool_call>  </tool_call>"#,
            r#"keep this <tool_call>{"name":"f","argu"#,
            r#"<tool_call>not json</tool_call> tail"#,
            "plain prose, no call",
            "  ",
            "",
        ],
        "kimi_k2" => vec![
            "I will check. <|tool_calls_section_begin|><|tool_call_begin|>functions.f:0<|tool_call_argument_begin|>{\"a\":1}<|tool_call_end|><|tool_calls_section_end|> after",
            "x <|tool_calls_section_begin|>garbage<|tool_calls_section_end|> y",
            "a <|tool_calls_section_begin|><|tool_call_begin|>functions.f:0<|tool_call_argument_begin|>{}<|tool_call_end|><|tool_calls_section_end|> b <|tool_calls_section_begin|><|tool_call_begin|>functions.g:1<|tool_call_argument_begin|>{}<|tool_call_end|><|tool_calls_section_end|>",
            "bare <|tool_call_begin|>functions.f:0<|tool_call_argument_begin|>{}<|tool_call_end|> tail",
            "gap <|tool_call_begin|>functions.f:0<|tool_call_argument_begin|>{}<|tool_call_end|> mid <|tool_calls_section_begin|><|tool_call_begin|>functions.g:1<|tool_call_argument_begin|>{}<|tool_call_end|><|tool_calls_section_end|>",
            " <|tool_calls_section_begin|><|tool_call_begin|>functions.f:0<|tool_call_argument_begin|>{}<|tool_call_end|>",
            "plain prose, no call",
            "",
        ],
        "minimax_m3" => vec![
            "STOP \t]<]minimax[>[<invoke name=\"f\">]<]minimax[>[</invoke>",
            "Let me look. ]<]minimax[>[<tool_call>]<]minimax[>[<invoke name=\"f\">]<]minimax[>[<a>1]<]minimax[>[</a>]<]minimax[>[</invoke>]<]minimax[>[</tool_call> after",
            "pre ]<]minimax[>[<invoke name=\"f\">]<]minimax[>[</invoke> ]<]minimax[>[<tool_call>]<]minimax[>[<invoke name=\"g\">]<]minimax[>[</invoke>]<]minimax[>[</tool_call>",
            "orphan ]<]minimax[>[</tool_call> tail",
            "truncated ]<]minimax[>[<tool_call>]<]minimax[>[<invoke name=\"f\">",
            "plain prose, no call",
            "",
        ],
        _ => vec![],
    }
}

/// Every text to parse for `parser`: whole corpus streams, their prefixes at
/// each chunk boundary and each character, and the targeted inputs.
pub(crate) fn texts(parser: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut push_with_prefixes = |text: &str| {
        for at in (0..=text.len()).filter(|at| text.is_char_boundary(*at)) {
            out.push(text[..at].to_string());
        }
    };
    for case in corpus().tool.iter().filter(|case| case.parser == parser) {
        push_with_prefixes(&case.chunks.concat());
    }
    for text in targeted(parser) {
        push_with_prefixes(text);
    }
    out
}

fn shape(calls: &[ToolCallResponse]) -> Vec<(String, String)> {
    calls
        .iter()
        .map(|call| (call.function.name.clone(), call.function.arguments.clone()))
        .collect()
}

pub(crate) fn read(input: &str, ranges: &[Range<usize>]) -> String {
    ranges.iter().map(|range| &input[range.clone()]).collect()
}

#[tokio::test]
async fn batch_parsers_tile_their_normal_text() {
    let mut count = 0;
    for parser in RECORDING_TOOL_PARSERS {
        for text in texts(parser) {
            for finalize in [false, true] {
                let label = format!("{parser} finalize={finalize} {text:?}");
                let public = if finalize {
                    try_tool_call_parse_aggregate_finalize(&text, Some(parser), None).await
                } else {
                    try_tool_call_parse_aggregate(&text, Some(parser), None).await
                };
                let off = try_tool_call_parse_aggregate_sourced(
                    &text,
                    Some(parser),
                    None,
                    finalize,
                    false,
                )
                .await;
                let on = try_tool_call_parse_aggregate_sourced(
                    &text,
                    Some(parser),
                    None,
                    finalize,
                    true,
                )
                .await;
                let (Ok(public), Ok(off), Ok(on)) = (public, off, on) else {
                    panic!("{label}: a parse failed");
                };
                assert_eq!(shape(&public.0), shape(&off.0), "{label}");
                assert_eq!(shape(&public.0), shape(&on.0), "{label}");
                let off_text = off.1.as_ref().map(|normal| normal.text());
                let on_text = on.1.as_ref().map(|normal| normal.text());
                assert_eq!(public.1.as_deref(), off_text, "{label}");
                assert_eq!(public.1.as_deref(), on_text, "{label}");
                if let Some(off) = off.1 {
                    assert!(!off.spans().is_recording(), "{label}");
                }
                if let Some(on) = on.1 {
                    assert!(on.spans().is_recording(), "{label}: no sources");
                    assert_eq!(read(&text, on.spans().ranges()), on.text(), "{label}");
                }
                count += 1;
            }
        }
    }
    assert!(count > 10_000, "corpus shrank: {count}");
    println!("{count} parses per recording mode");
}

async fn normal_ranges(parser: &str, text: &str, finalize: bool) -> (usize, Vec<Range<usize>>) {
    let (calls, normal) =
        try_tool_call_parse_aggregate_sourced(text, Some(parser), None, finalize, true)
            .await
            .unwrap();
    (calls.len(), normal.unwrap().spans().ranges().to_vec())
}

#[tokio::test]
async fn text_beside_a_call_keeps_its_own_offsets() {
    // Review finding 1: Hermes prefix narration before a bare call.
    assert_eq!(
        normal_ranges("hermes", r#"STOP {"name":"f","arguments":{}}"#, false).await,
        (1, vec![0..4])
    );
    // The MiniMax M3 case from review round 2.
    let m3 = "STOP \t]<]minimax[>[<invoke name=\"f\">]<]minimax[>[</invoke>";
    assert_eq!(normal_ranges("minimax_m3", m3, true).await, (1, vec![0..4]));
    // Kimi K2 joins the text on both sides of a section that parses no call.
    let kimi = "x <|tool_calls_section_begin|>garbage<|tool_calls_section_end|> y";
    let after = kimi.len() - 2;
    assert_eq!(
        normal_ranges("kimi_k2", kimi, false).await,
        (0, vec![0..2, after..kimi.len()])
    );
    // A repeated word: the kept occurrence is the first one.
    let twice = r#"STOP <tool_call>{"name":"STOP","arguments":{}}</tool_call>"#;
    assert_eq!(normal_ranges("hermes", twice, false).await, (1, vec![0..4]));
}
