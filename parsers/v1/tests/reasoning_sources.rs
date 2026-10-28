// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Source recording in the v1 reasoning parsers.
//!
//! With recording on, a parser reports, for every byte it returns, the input
//! byte it was copied from. These tests check four rules on every corpus
//! stream at every split:
//!
//! * Same output: recording on and off return identical results.
//! * Verbatim and tiling: per channel, the input bytes of the runs, read in
//!   order, equal the returned text.
//! * Causality: no run goes past the bytes fed so far.
//! * Order: runs never go back in the input.
//!
//! Byte equality cannot tell two occurrences of the same bytes apart, so the
//! exact-offset cases at the end name the expected ranges.

use std::ops::Range;

use dynamo_parsers::ReasoningParser;
use dynamo_parsers::reasoning::ReasoningParserType;
use dynamo_parsers::sources::{SourceChannel, SourceRun};
use serde::Deserialize;

const CORPUS: &str = include_str!("data/source_ranges_corpus.json");

#[derive(Deserialize)]
struct Corpus {
    reasoning: Vec<ReasoningCase>,
}

#[derive(Deserialize)]
struct ReasoningCase {
    name: String,
    parser: String,
    prompt_injected_reasoning: bool,
    chunks: Vec<String>,
}

/// Every reasoning parser name that `BasicReasoningParser` serves.
const BASIC_PARSERS: &[&str] = &[
    "basic",
    "qwen3",
    "minimax_m3",
    "deepseek_r1",
    "deepseek_v4",
    "kimi",
    "kimi_k25",
    "kimi_k3",
    "mistral",
    "minimax_m2",
    "nemotron_v3",
    "nemotron_deci",
    "step3",
];

/// Reasoning parsers that do not record: they must say so.
const REFUSING_PARSERS: &[&str] = &[
    "gpt_oss",
    "granite",
    "gemma4",
    "inkling",
    "minimax_append_think",
];

fn make(name: &str, in_reasoning: bool) -> Box<dyn ReasoningParser> {
    let mut parser = ReasoningParserType::get_reasoning_parser_from_name(name);
    if in_reasoning {
        parser.set_in_reasoning(true);
    }
    Box::new(parser)
}

#[derive(Debug, PartialEq, Eq)]
struct Step {
    normal: String,
    reasoning: String,
}

/// Feeds `chunks`, then finishes. Returns each call's result and, when
/// recording, its runs.
fn run(
    name: &str,
    in_reasoning: bool,
    chunks: &[&str],
    record: bool,
) -> (Vec<Step>, Vec<Vec<SourceRun>>) {
    let mut parser = make(name, in_reasoning);
    if record {
        assert!(parser.record_sources(), "{name} refused to record");
    }
    let mut steps = Vec::new();
    let mut runs = Vec::new();
    let mut take = |parser: &mut Box<dyn ReasoningParser>, result: dynamo_parsers::ParserResult| {
        steps.push(Step {
            normal: result.normal_text,
            reasoning: result.reasoning_text,
        });
        let mut out = Vec::new();
        parser.take_sources(&mut out);
        runs.push(out);
    };
    for chunk in chunks {
        let result = parser.parse_reasoning_streaming_incremental(chunk, &[]);
        take(&mut parser, result);
    }
    let result = parser.finish_reasoning_stream();
    take(&mut parser, result);
    (steps, runs)
}

fn read(input: &str, runs: &[SourceRun], channel: SourceChannel) -> String {
    runs.iter()
        .filter(|run| run.channel == channel)
        .map(|run| &input[run.input.clone()])
        .collect()
}

/// Checks one stream: same output on and off, verbatim and tiling, causality,
/// and input order.
fn check(name: &str, in_reasoning: bool, chunks: &[&str], label: &str) {
    let (off, off_runs) = run(name, in_reasoning, chunks, false);
    let (on, on_runs) = run(name, in_reasoning, chunks, true);
    assert_eq!(off, on, "{label}: recording changed the output");
    assert!(
        off_runs.iter().all(Vec::is_empty),
        "{label}: recording off reported runs"
    );
    let input: String = chunks.concat();
    let mut fed = 0;
    let mut last_end = 0;
    for (index, (step, runs)) in on.iter().zip(&on_runs).enumerate() {
        fed += chunks.get(index).map_or(0, |chunk| chunk.len());
        assert_eq!(
            read(&input, runs, SourceChannel::Normal),
            step.normal,
            "{label}: call {index} normal text does not tile"
        );
        assert_eq!(
            read(&input, runs, SourceChannel::Reasoning),
            step.reasoning,
            "{label}: call {index} reasoning text does not tile"
        );
        for run in runs {
            assert!(
                run.input.end <= fed,
                "{label}: call {index} {run:?} is after the input fed"
            );
            assert!(
                run.input.start >= last_end,
                "{label}: call {index} {run:?} goes back"
            );
            assert!(!run.input.is_empty(), "{label}: empty run");
            last_end = run.input.end;
        }
    }
}

/// The recorded split, one character per chunk, and every two-way split.
fn splits(input: &str, recorded: &[String]) -> Vec<Vec<String>> {
    let mut out = vec![recorded.to_vec()];
    out.push(input.chars().map(String::from).collect());
    for at in (0..=input.len()).filter(|at| input.is_char_boundary(*at)) {
        out.push(vec![input[..at].to_string(), input[at..].to_string()]);
    }
    out
}

fn sweep(name: &str, in_reasoning: bool, recorded: &[String], label: &str) -> usize {
    let input = recorded.concat();
    let mut count = 0;
    for chunks in splits(&input, recorded) {
        let chunks: Vec<&str> = chunks.iter().map(String::as_str).collect();
        check(name, in_reasoning, &chunks, label);
        count += 1;
    }
    count
}

/// Inputs aimed at each buffer path: partial markers, the force-reasoning
/// splice, tool-start exits, dangling closers, tool parameters, and
/// multi-byte characters next to markers.
const TARGETED: &[&str] = &[
    "A<think>B</think>C",
    "<think>ok</think>ok",
    "\n<think>prefix splice</think>after",
    "plain text with no markers",
    "<think>open and never closed",
    "stray </think> closer then text",
    "<think>a</think>b<think>c</think>d",
    "<thin then not a marker",
    "reasoning</thi then not a closer</think>done",
    "<mm:think>m3</mm:think>answer",
    "prefill reasoning</mm:think>answer",
    "thinking]<]minimax[>[<tool_call>]<]minimax[>[<invoke name=\"f\">",
    "<think>pre<tool_call>{\"name\":\"f\"}</tool_call>",
    "<think>x<|tool_calls_section_begin|>y",
    "<tool_call><function=f><parameter=p></think><think></parameter></function></tool_call><think>r</think>",
    "◁think▷kimi◁/think▷done",
    "[THINK]mistral[/THINK]done",
    "<|open|>think<|sep|>k3<|close|>think<|sep|>answer<|open|>response",
    "<think>é</think>ü<",
    "<",
    "",
];

#[test]
fn every_corpus_stream_at_every_split() {
    let corpus: Corpus = serde_json::from_str(CORPUS).unwrap();
    let mut runs = 0;
    for case in &corpus.reasoning {
        let label = format!("{} [{}]", case.name, case.parser);
        runs += sweep(
            &case.parser,
            case.prompt_injected_reasoning,
            &case.chunks,
            &label,
        );
    }
    // Rhino's three v1 families, from dynamo's recorded streams.
    assert!(corpus.reasoning.len() >= 120, "corpus shrank");
    println!(
        "{} streams, {runs} runs per recording mode",
        corpus.reasoning.len()
    );
}

#[test]
fn every_basic_parser_on_targeted_inputs() {
    let mut runs = 0;
    for name in BASIC_PARSERS {
        for in_reasoning in [false, true] {
            for input in TARGETED {
                let label = format!("{name} in_reasoning={in_reasoning} {input:?}");
                runs += sweep(name, in_reasoning, &[input.to_string()], &label);
            }
        }
    }
    println!("{runs} runs per recording mode");
}

#[test]
fn parsers_that_cannot_record_say_so() {
    for name in BASIC_PARSERS {
        assert!(make(name, false).record_sources(), "{name}");
    }
    for name in REFUSING_PARSERS {
        let mut parser = make(name, false);
        assert!(!parser.record_sources(), "{name}");
        let result = parser.parse_reasoning_streaming_incremental("<think>a</think>b", &[]);
        let mut out = Vec::new();
        parser.take_sources(&mut out);
        assert!(out.is_empty(), "{name}: {result:?}");
    }
}

#[test]
fn recording_refuses_held_bytes_it_cannot_date() {
    let mut parser = make("basic", false);
    let held = parser.parse_reasoning_streaming_incremental("text </th", &[]);
    assert_eq!(held.normal_text, "text ");
    assert!(!parser.record_sources());

    // With nothing held, recording can start mid-stream; offsets still count
    // every byte this parser received.
    let mut parser = make("basic", false);
    parser.parse_reasoning_streaming_incremental("abc", &[]);
    assert!(parser.record_sources());
    parser.parse_reasoning_streaming_incremental("<think>r</think>", &[]);
    let mut out = Vec::new();
    parser.take_sources(&mut out);
    assert_eq!(out, vec![run_of(SourceChannel::Reasoning, 10..11)]);
}

#[test]
fn batch_parsing_records_nothing() {
    let mut parser = make("basic", false);
    assert!(parser.record_sources());
    let result = parser.detect_and_parse_reasoning("<think>r</think> n ", &[]);
    assert_eq!(result.normal_text, "n");
    let mut out = Vec::new();
    parser.take_sources(&mut out);
    assert!(out.is_empty());
}

fn run_of(channel: SourceChannel, input: Range<usize>) -> SourceRun {
    SourceRun { channel, input }
}

fn runs_of(name: &str, in_reasoning: bool, chunks: &[&str]) -> Vec<SourceRun> {
    run(name, in_reasoning, chunks, true).1.concat()
}

#[test]
fn interleaved_text_keeps_its_input_order() {
    // Review finding 2: the two fields lose their relative order, the runs keep it.
    use SourceChannel::{Normal, Reasoning};
    assert_eq!(
        runs_of("basic", false, &["A<think>B</think>C"]),
        vec![
            run_of(Normal, 0..1),
            run_of(Reasoning, 8..9),
            run_of(Normal, 17..18),
        ]
    );
}

#[test]
fn repeated_text_gets_the_offset_of_the_bytes_that_were_kept() {
    use SourceChannel::{Normal, Reasoning};
    assert_eq!(
        runs_of("basic", false, &["<think>ok</think>ok"]),
        vec![run_of(Reasoning, 7..9), run_of(Normal, 17..19)]
    );
    // A held `</th` that turns out to be text, released by the next call.
    assert_eq!(
        runs_of("basic", true, &["ok</th", "ok"]),
        vec![run_of(Reasoning, 0..2), run_of(Reasoning, 2..8)]
    );
    // MiniMax M3 holds a lone `<` and then releases it with the next chunk.
    assert_eq!(
        runs_of("minimax_m3", false, &["a<", "b</mm:think>c"]),
        vec![run_of(Reasoning, 0..3), run_of(Normal, 14..15)]
    );
    // The force-reasoning splice cuts a `<think>` out of the middle of the buffer.
    assert_eq!(
        runs_of("deepseek_r1", false, &["\n<think>r</think>n"]),
        vec![
            run_of(Reasoning, 0..1),
            run_of(Reasoning, 8..9),
            run_of(Normal, 17..18)
        ]
    );
}

#[test]
fn the_deferral_case_gets_exact_leaf_ranges() {
    // The v1 reasoning stream can hold `" "` back and release it after `R`
    // (force_nonempty_content), so stock order is `R`, then `" A"`. The leaf
    // dates each byte; the stage keeps its own order and never sorts.
    use SourceChannel::{Normal, Reasoning};
    let (_, per_call) = run(
        "basic",
        false,
        &[" ", "<think>", "R", "</think>", "A"],
        true,
    );
    assert_eq!(
        per_call,
        vec![
            vec![run_of(Normal, 0..1)],
            vec![],
            vec![run_of(Reasoning, 8..9)],
            vec![],
            vec![run_of(Normal, 17..18)],
            vec![],
        ]
    );
}
