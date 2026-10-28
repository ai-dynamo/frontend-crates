// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared harness for the source recording cost guards. It uses only API
//! that the base commit also has, so the recording-off guard runs there.

#![allow(dead_code)]

use std::time::{Duration, Instant};

use dynamo_parsers::tool_calling::jail::annotated::Annotated;
use dynamo_protocols::types::CreateChatCompletionStreamResponse;
use serde::Deserialize;

pub const CORPUS: &str = include_str!("../data/source_ranges_corpus.json");
const ROUNDS: usize = 15;

#[derive(Deserialize)]
pub struct Corpus {
    pub reasoning: Vec<Case>,
    pub tool: Vec<Case>,
}

#[derive(Deserialize)]
pub struct Case {
    pub parser: String,
    pub chunks: Vec<String>,
}

pub fn corpus() -> Corpus {
    serde_json::from_str(CORPUS).unwrap()
}

/// One character per chunk, as the endpoint feeds one token per chunk.
pub fn chars(case: &Case) -> Vec<String> {
    case.chunks.concat().chars().map(String::from).collect()
}

/// Median over the rounds of `run`, in nanoseconds per chunk.
pub fn median_ns_per_chunk(chunks: usize, mut run: impl FnMut()) -> f64 {
    let mut times: Vec<Duration> = (0..ROUNDS)
        .map(|_| {
            let start = Instant::now();
            run();
            start.elapsed()
        })
        .collect();
    times.sort();
    times[ROUNDS / 2].as_nanos() as f64 / chunks as f64
}

pub fn chunk(text: &str) -> Annotated<CreateChatCompletionStreamResponse> {
    Annotated {
        data: Some(
            serde_json::from_value(serde_json::json!({
                "id": "cost", "object": "chat.completion.chunk", "created": 0, "model": "m",
                "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": null}]
            }))
            .unwrap(),
        ),
        id: None,
        event: None,
        comment: None,
        error: None,
    }
}
