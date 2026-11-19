// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Request rules that the Dynamo frontend applies around prompt rendering.
//!
//! These rules decide what a request's prompt looks like beyond the template
//! itself: which thinking mode the template sees, whether the last message
//! stays open, whether the request's raw prompt replaces the template, and how
//! historical tool-call arguments reach the template. They live here so that
//! any frontend that renders with this crate gives a client the same prompt
//! as the Dynamo frontend.
//!
//! The rules read only the request trait, the chat-template argument map, and
//! plain values. A caller passes in what its own request type carries beyond
//! [`OAIChatLikeRequest`](crate::OAIChatLikeRequest): a top-level `thinking`
//! field, guided-decoding fields, `continue_final_message`, and the raw prompt.

mod template;
mod thinking;
mod tool_arguments;
mod truthy;

#[cfg(test)]
mod tests;

pub use template::{ChatPromptOptions, render_chat_prompt, should_add_generation_prompt};
pub use thinking::{
    ChatTemplateArgs, ThinkingRules, apply_default_thinking_mode, has_guided_tool_choice,
    has_minimax_m3_constrained_generation, has_request_thinking_control,
    has_structured_response_format, is_minimax_m3_family, normalize_thinking_args,
    reject_unsupported_reasoning_effort,
};
pub use tool_arguments::should_normalize_tool_call_arguments;
