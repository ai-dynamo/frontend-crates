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
//! - `qwen2_5_vl_chat_template.jinja`: `Qwen/Qwen2.5-VL-7B-Instruct`
//!   `chat_template.json` at revision `cc594898137f460bfe9f0759e9844b3ce807cfb5`.
//! - `qwen3_vl_chat_template.jinja`: `Qwen/Qwen3-VL-8B-Instruct` `chat_template.json`
//!   at revision `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`.
//! - `glm5_chat_template.jinja`: `zai-org/GLM-5` `chat_template.jinja` at revision
//!   `c183ef8c61faee82855eca1ed9bb3a9a7ce3b0b2`.
//! - `kimi_linear_chat_template.jinja`: `moonshotai/Kimi-Linear-48B-A3B-Instruct`
//!   `chat_template.jinja` at revision `e1df551a447157d4658b573f9a695d57658590e9`.
//! - `deepseek_r1_chat_template.jinja`: the `chat_template` of `deepseek-ai/DeepSeek-R1`
//!   `tokenizer_config.json` at revision `56d4cbbb4d29f4355bab4b9a39ccb717a14ad5ad`.
//!
//! The Qwen and gpt-oss templates are Apache-2.0; GLM-5, Kimi-Linear, and
//! DeepSeek-R1 are MIT.
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
const QWEN2_5_VL: &str = include_str!("fixtures/qwen2_5_vl_chat_template.jinja");
const QWEN3_VL: &str = include_str!("fixtures/qwen3_vl_chat_template.jinja");
const GLM5: &str = include_str!("fixtures/glm5_chat_template.jinja");
const KIMI_LINEAR: &str = include_str!("fixtures/kimi_linear_chat_template.jinja");
const DEEPSEEK_R1: &str = include_str!("fixtures/deepseek_r1_chat_template.jinja");
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
    formatter_with_unk(template, bos, eos, None)
}

fn formatter_with_unk(
    template: &str,
    bos: Option<&str>,
    eos: &str,
    unk: Option<&str>,
) -> Arc<dyn OAIPromptFormatter> {
    let config: ChatTemplate = serde_json::from_value(json!({
        "chat_template": template,
        "bos_token": bos,
        "eos_token": eos,
        "unk_token": unk,
    }))
    .unwrap();
    let PromptFormatter::OAI(formatter) =
        PromptFormatter::from_parts(config, ContextMixins::new(&[]), false).unwrap();
    formatter
}

/// A system message, then a later one after an assistant reply.
fn later_system_request() -> CreateChatCompletionRequest {
    later_message_request("system")
}

fn later_message_request(later_role: &str) -> CreateChatCompletionRequest {
    serde_json::from_value(json!({
        "model": "test",
        "messages": [
            {"role": "system", "content": FIRST_SYSTEM},
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "first answer"},
            {"role": later_role, "content": LATER_SYSTEM},
            {"role": "user", "content": "second question"}
        ]
    }))
    .unwrap()
}

/// Whether the renderer demotes the later system message. The flag is private,
/// so it is read from behavior: demotion turns the message into a user turn before
/// the template runs, so the request renders exactly like the same request with
/// that message sent as a user turn only when the flag is on.
fn demotes_later_system(formatter: &dyn OAIPromptFormatter) -> bool {
    formatter.render(&later_message_request("system")).unwrap()
        == formatter.render(&later_message_request("user")).unwrap()
}

/// The renderer must demote the later message exactly when the template drops it,
/// and the message must reach the prompt either way. A template that keeps it may
/// move it: DeepSeek-R1 gathers every system message into one prompt at the start.
fn assert_classified(formatter: Arc<dyn OAIPromptFormatter>, template_drops: bool) {
    assert_eq!(demotes_later_system(formatter.as_ref()), template_drops);
    let prompt = formatter.render(&later_system_request()).unwrap();
    assert!(prompt.contains(LATER_SYSTEM), "{prompt}");
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

// Real templates: the renderer demotes a later system message exactly for the
// templates that drop it.

#[test]
fn real_template_gpt_oss_drops_later_system_and_is_demoted() {
    assert_classified(
        formatter(GPT_OSS, Some("<|startoftext|>"), "<|return|>"),
        true,
    );
}

#[test]
fn real_template_qwen3_vl_drops_later_system_and_is_demoted() {
    assert_classified(formatter(QWEN3_VL, None, "<|im_end|>"), true);
}

#[test]
fn real_template_qwen3_keeps_later_system_and_is_not_demoted() {
    assert_classified(formatter(QWEN3, None, "<|im_end|>"), false);
}

#[test]
fn real_template_qwen2_5_vl_keeps_later_system_and_is_not_demoted() {
    assert_classified(formatter(QWEN2_5_VL, None, "<|im_end|>"), false);
}

#[test]
fn real_template_glm5_keeps_later_system_and_is_not_demoted() {
    assert_classified(formatter(GLM5, None, "<|endoftext|>"), false);
}

#[test]
fn real_template_kimi_linear_keeps_later_system_and_is_not_demoted() {
    assert_classified(
        formatter_with_unk(KIMI_LINEAR, Some("[BOS]"), "[EOS]", Some("[UNK]")),
        false,
    );
}

#[test]
fn real_template_deepseek_r1_keeps_later_system_and_is_not_demoted() {
    assert_classified(
        formatter(
            DEEPSEEK_R1,
            Some("<\u{ff5c}begin\u{2581}of\u{2581}sentence\u{ff5c}>"),
            "<\u{ff5c}end\u{2581}of\u{2581}sentence\u{ff5c}>",
        ),
        false,
    );
}
