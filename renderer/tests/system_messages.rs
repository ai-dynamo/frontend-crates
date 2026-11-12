// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A `system` message after the first one must reach the rendered prompt.
//!
//! Agent clients send such messages mid-conversation: Claude Code sends system
//! reminders, and Codex sends developer messages that servers map to `system`. Some
//! templates read only the first system message and have no branch for a later one,
//! so the later message renders nothing and no error is raised. The renderer must
//! notice this at load time and demote the later message to a user turn in place.
//!
//! Fixtures, from Hugging Face:
//! - `gpt_oss_20b_chat_template.jinja`: `openai/gpt-oss-20b` `chat_template.jinja` at
//!   revision `6cee5e81ee83917806bbde320786a8fb61efebee`, plus a final newline.
//! - `qwen3_chat_template.jinja`: the `chat_template` of `Qwen/Qwen3-0.6B`
//!   `tokenizer_config.json` at revision `c1899de289a04d12100db370d81485cdf75e47ca`.
//!
//! The MiniMax M3 template drops later system messages too, but its license allows
//! only non-commercial use, so it is not in this repository. Download
//! `chat_template.jinja` from `MiniMaxAI/MiniMax-M3` at revision
//! `d95e30cb912fe3021bb9c2ba66a8ce60b6c937a9`, set `DYNAMO_RENDERER_MINIMAX_M3_TEMPLATE`
//! to its path, and run the ignored test.

use std::sync::Arc;

use dynamo_protocols::types::CreateChatCompletionRequest;
use dynamo_renderer::{ChatTemplate, ContextMixins, OAIPromptFormatter, PromptFormatter};
use serde_json::json;

const GPT_OSS: &str = include_str!("fixtures/gpt_oss_20b_chat_template.jinja");
const QWEN3: &str = include_str!("fixtures/qwen3_chat_template.jinja");
const MINIMAX_M3_ENV: &str = "DYNAMO_RENDERER_MINIMAX_M3_TEMPLATE";

const FIRST_SYSTEM: &str = "FIRST_SYSTEM_MARKER";
const LATER_SYSTEM: &str = "LATER_SYSTEM_MARKER";

/// What the request must render, in this order.
const ORDER: [&str; 5] = [
    FIRST_SYSTEM,
    "first question",
    "first answer",
    LATER_SYSTEM,
    "second question",
];

fn formatter(template: &str, bos: Option<&str>, eos: &str) -> Arc<dyn OAIPromptFormatter> {
    let config: ChatTemplate = serde_json::from_value(json!({
        "chat_template": template,
        "bos_token": bos,
        "eos_token": eos,
    }))
    .unwrap();
    let PromptFormatter::OAI(formatter) =
        PromptFormatter::from_parts(config, ContextMixins::new(&[]), false).unwrap();
    formatter
}

/// A system message, then a later one after an assistant reply.
fn later_system_request() -> CreateChatCompletionRequest {
    serde_json::from_value(json!({
        "model": "test",
        "messages": [
            {"role": "system", "content": FIRST_SYSTEM},
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "first answer"},
            {"role": "system", "content": LATER_SYSTEM},
            {"role": "user", "content": "second question"}
        ]
    }))
    .unwrap()
}

/// Fails unless each marker is in `prompt`, after the one before it.
fn assert_in_order(prompt: &str, markers: &[&str]) {
    let mut from = 0;
    for marker in markers {
        match prompt[from..].find(marker) {
            Some(at) => from += at + marker.len(),
            None => panic!("`{marker}` is missing or out of order in the prompt:\n{prompt}"),
        }
    }
}

#[test]
fn gpt_oss_demotes_later_system_message_to_user_turn() {
    let formatter = formatter(GPT_OSS, Some("<|startoftext|>"), "<|return|>");
    let prompt = formatter.render(&later_system_request()).unwrap();
    assert_in_order(&prompt, &ORDER);
    assert!(
        prompt.contains(&format!("<|start|>user<|message|>{LATER_SYSTEM}<|end|>")),
        "later system message must render as a user turn in place:\n{prompt}"
    );
}

/// Control: Qwen3 renders a later system message itself, so it must stay a system turn.
#[test]
fn qwen3_keeps_later_system_message_as_system_turn() {
    let formatter = formatter(QWEN3, None, "<|im_end|>");
    let prompt = formatter.render(&later_system_request()).unwrap();
    assert_in_order(&prompt, &ORDER);
    assert!(
        prompt.contains(&format!("<|im_start|>system\n{LATER_SYSTEM}<|im_end|>")),
        "later system message must stay a system turn:\n{prompt}"
    );
}

#[test]
#[ignore = "needs the MiniMax M3 template, which is not in this repository; set DYNAMO_RENDERER_MINIMAX_M3_TEMPLATE"]
fn minimax_m3_demotes_later_system_message_to_user_turn() {
    let path = std::env::var(MINIMAX_M3_ENV)
        .unwrap_or_else(|_| panic!("set {MINIMAX_M3_ENV} to the MiniMax M3 chat_template.jinja"));
    let template = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let formatter = formatter(&template, Some("]~b]"), "[e~[");
    let prompt = formatter.render(&later_system_request()).unwrap();
    assert_in_order(&prompt, &ORDER);
    assert!(
        prompt.contains(&format!("]~b]user\n{LATER_SYSTEM}[e~[")),
        "later system message must render as a user turn in place:\n{prompt}"
    );
}
