// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Render throughput probe.
//!
//! Measures nanoseconds per render for each [`RenderMode`] over one fixed
//! request, so a change to the render path can be shown not to slow it down.
//! Deliberately dependency-free: it reports raw timings rather than a
//! statistical verdict, and is meant to be run before and after a change on
//! the same machine with the same command.
//!
//! ```text
//! cargo run --release -p dynamo-renderer --example render_bench -- adaptive
//! ```
//!
//! Output is one line per round plus a summary line per mode:
//! `mode=<m> rounds=<n> min_ns=<n> median_ns=<n> renders_per_sec=<n>`.

use std::time::{Duration, Instant};

use dynamo_protocols::types::CreateChatCompletionRequest;
use dynamo_renderer::{ChatTemplate, ContextMixins, PromptFormatter, RenderMode};
use serde_json::json;

/// Conversation turns in the benchmark request.
const TURNS: usize = 16;
const WARMUP_RENDERS: usize = 200;
const RENDERS_PER_ROUND: usize = 2_000;
const ROUNDS: usize = 5;

fn template() -> ChatTemplate {
    serde_json::from_value(json!({
        "chat_template": concat!(
            "{% for m in messages %}",
            "<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n",
            "{% endfor %}",
            "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}",
        ),
        "bos_token": "<s>",
        "eos_token": "<|im_end|>",
        "unk_token": "<unk>",
    }))
    .expect("benchmark template is well formed")
}

fn request() -> CreateChatCompletionRequest {
    let mut messages = Vec::with_capacity(TURNS + 1);
    messages.push(json!({"role": "system", "content": "You are a helpful assistant."}));
    for turn in 0..TURNS {
        let text = format!(
            "turn {turn}: the quick brown fox jumps over the lazy dog, \
             then it does so again to make the content realistic in length"
        );
        if turn % 2 == 0 {
            messages.push(json!({"role": "user", "content": text}));
        } else {
            // Carry the fields the Adaptive path rewrites -- reasoning injection
            // and tool-call argument normalization -- so the two modes do
            // genuinely different work and a regression inside that block is
            // visible instead of being measured on no-op data.
            messages.push(json!({
                "role": "assistant",
                "content": text,
                "reasoning_content": format!("reasoning for turn {turn}"),
                "tool_calls": [{
                    "id": format!("call_{turn}"),
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "arguments": "{\"city\": \"Paris\", \"unit\": \"c\"}"
                    }
                }],
            }));
        }
    }
    serde_json::from_value(json!({"model": "bench", "messages": messages}))
        .expect("benchmark request is well formed")
}

fn mode_from_arg(arg: &str) -> RenderMode {
    match arg {
        "adaptive" => RenderMode::Adaptive,
        "verbatim" => RenderMode::Verbatim,
        other => panic!("unknown mode {other:?}; expected `adaptive` or `verbatim`"),
    }
}

fn main() {
    let requested: Vec<RenderMode> = std::env::args()
        .skip(1)
        .map(|arg| mode_from_arg(&arg))
        .collect();
    let modes = if requested.is_empty() {
        vec![RenderMode::Adaptive, RenderMode::Verbatim]
    } else {
        requested
    };

    let req = request();
    // Count the rendered bytes so the compiler cannot elide the work.
    let mut sink = 0usize;

    for mode in modes {
        // `PromptFormatter` has a single variant, so this pattern is irrefutable;
        // a `let ... else` here trips `irrefutable_let_patterns`, which CI denies.
        let PromptFormatter::OAI(formatter) =
            PromptFormatter::from_parts_with_mode(template(), ContextMixins::new(&[]), true, mode)
                .expect("formatter builds");

        for _ in 0..WARMUP_RENDERS {
            sink += formatter.render(&req).expect("render succeeds").len();
        }

        let mut per_round: Vec<Duration> = Vec::with_capacity(ROUNDS);
        for round in 0..ROUNDS {
            let started = Instant::now();
            for _ in 0..RENDERS_PER_ROUND {
                sink += formatter.render(&req).expect("render succeeds").len();
            }
            let elapsed = started.elapsed();
            per_round.push(elapsed);
            println!(
                "mode={mode:?} round={round} total_ns={} per_render_ns={}",
                elapsed.as_nanos(),
                elapsed.as_nanos() / RENDERS_PER_ROUND as u128,
            );
        }

        let mut sorted = per_round.clone();
        sorted.sort();
        let min = sorted.first().copied().unwrap_or_default();
        let median = sorted[sorted.len() / 2];
        println!(
            "mode={mode:?} rounds={ROUNDS} min_ns={} median_ns={} renders_per_sec={}",
            min.as_nanos() / RENDERS_PER_ROUND as u128,
            median.as_nanos() / RENDERS_PER_ROUND as u128,
            RENDERS_PER_ROUND as f64 / median.as_secs_f64(),
        );
    }

    // Keep the sink observable so the loop above cannot be optimised away.
    std::hint::black_box(sink);
}
