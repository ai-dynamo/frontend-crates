// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Record the Dynamo v1 REASONING parser output over the reasoning fixture corpus.
//!
//! The other three corpora each had a recorder (`record_dynamo_batch`,
//! `record_dynamo_jail_stream`, `record_dynamo_stream`); reasoning had none, so a NEW
//! reasoning case could not get its `expected.dynamo_v1` block without hand-writing the
//! answer — which is how `gpt_oss` and `granite` ended up with empty inputs for
//! `REASONING.batch.6.b` / `REASONING.stream.2.c` instead of real coverage.
//!
//! Runs exactly what the parity fixtures record:
//!   batch  -> `detect_and_parse_reasoning(model_text, &[])`
//!   stream -> `parse_reasoning_streaming_incremental(chunk, &[])` per chunk, with the
//!             per-chunk results concatenated, which is what the fixture stores.
//!
//! JSON in (one family per invocation; the Python driver owns fixture I/O):
//!   {"family": "gpt_oss", "mode": "batch",
//!    "cases": {"REASONING.batch.6.b": {"model_text": "..."}}}
//!   {"family": "gpt_oss", "mode": "stream",
//!    "cases": {"REASONING.stream.2.c": {"chunks": ["...", "..."]}}}
//! JSON out:
//!   {"REASONING.batch.6.b": {"reasoning_text": "...", "normal_text": "..."}}
//!
//! Usage: cargo run -p dynamo-parsers --bin record_dynamo_reasoning -- <input.json>

use std::collections::BTreeMap;

use dynamo_parsers::reasoning::{ReasoningParser, ReasoningParserType};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Input {
    family: String,
    mode: String,
    cases: BTreeMap<String, CaseIn>,
}

#[derive(Deserialize)]
struct CaseIn {
    #[serde(default)]
    model_text: String,
    #[serde(default)]
    chunks: Vec<String>,
}

#[derive(Serialize)]
struct CaseOut {
    reasoning_text: String,
    normal_text: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .ok_or("usage: record_dynamo_reasoning <input.json>")?;
    let input: Input = serde_json::from_str(&std::fs::read_to_string(path)?)?;

    let mut out: BTreeMap<String, CaseOut> = BTreeMap::new();
    for (case_id, case) in input.cases {
        let result = record_case(&input.family, &input.mode, &case);
        out.insert(case_id, result);
    }

    println!("{}", serde_json::to_string(&out)?);
    Ok(())
}

fn record_case(family: &str, mode: &str, case: &CaseIn) -> CaseOut {
    // A fresh parser per case: these are independent turns, and several parsers keep
    // state across chunks (buffering a split marker) that must not leak between cases.
    let mut parser = ReasoningParserType::get_reasoning_parser_from_name(family);
    if mode != "stream" {
        let result = parser.detect_and_parse_reasoning(&case.model_text, &[]);
        return CaseOut {
            reasoning_text: result.reasoning_text,
            normal_text: result.normal_text,
        };
    }

    let mut reasoning = String::new();
    let mut normal = String::new();
    for chunk in &case.chunks {
        let result = parser.parse_reasoning_streaming_incremental(chunk, &[]);
        reasoning.push_str(&result.reasoning_text);
        normal.push_str(&result.normal_text);
    }
    // Streaming parsers may retain a partial marker or ambiguous prefix until EOF.
    // Captures must exercise the same finish path as serving and parity tests.
    let result = parser.finish_reasoning_stream();
    reasoning.push_str(&result.reasoning_text);
    normal.push_str(&result.normal_text);
    CaseOut {
        reasoning_text: reasoning,
        normal_text: normal,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stream_capture_flushes_buffered_minimax_m3_text_at_eof() {
        for text in ["plain text", "plain <"] {
            let result = record_case(
                "minimax_m3",
                "stream",
                &CaseIn {
                    model_text: String::new(),
                    chunks: vec![text.to_string()],
                },
            );
            assert_eq!(result.reasoning_text, "");
            assert_eq!(result.normal_text, text);
        }
    }
}
