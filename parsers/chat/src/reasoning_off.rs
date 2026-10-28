// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Whether a request turned reasoning off, and what happens to reasoning text
//! when it did.
//!
//! The answer depends on the deployment's parsers and on the thinking controls
//! the request sent. The Dynamo frontend uses it to skip the v1 reasoning
//! parser, to start a unified v2 parser outside reasoning, and to tell the
//! non-streaming aggregator to keep reasoning out of `reasoning_content`.

use dynamo_parsers::chat_stream::ChatStreamItem;
use dynamo_protocols::types::{ChatCompletionMessageContent, CreateChatCompletionRequest};
use dynamo_renderer::request_rules::{
    ChatTemplateArgs, ThinkingRules, has_minimax_m3_constrained_generation, is_minimax_m3_family,
};
use dynamo_renderer::thinking_bool_from_args;
use futures::{Stream, StreamExt};

/// What the stream rules read from one chat request.
///
/// The Dynamo frontend's request type carries more than the OpenAI request.
/// These fields hold the parts the rules read.
#[derive(Clone, Copy, Debug)]
pub struct ChatStreamRequest<'a> {
    /// The OpenAI request.
    pub request: &'a CreateChatCompletionRequest,
    /// The request's `chat_template_args`.
    pub chat_template_args: Option<&'a ChatTemplateArgs>,
    /// The request carries a thinking control outside `chat_template_args`.
    /// In Dynamo, this is the top-level `thinking` field.
    pub client_thinking_field: bool,
    /// The request's own guided-decoding fields install a grammar. In Dynamo,
    /// these are `guided_json`, `guided_regex`, a non-empty `guided_choice`,
    /// and `guided_grammar`.
    pub explicit_guided_decoding: bool,
}

/// Mirror dynamo-renderer's DeepSeek reasoning-toggle precedence.
fn deepseek_renderer_reasoning_enabled(
    chat_template_args: Option<&ChatTemplateArgs>,
    default_enabled: bool,
) -> bool {
    if let Some(enabled) = thinking_bool_from_args(chat_template_args) {
        return enabled;
    }
    if let Some(mode) = chat_template_args
        .and_then(|args| args.get("thinking_mode"))
        .and_then(serde_json::Value::as_str)
    {
        return match mode {
            "chat" => false,
            "thinking" => true,
            _ => default_enabled,
        };
    }
    default_enabled
}

fn mistral_reasoning_enabled(chat_template_args: Option<&ChatTemplateArgs>) -> bool {
    chat_template_args
        .and_then(|args| args.get("reasoning_effort"))
        .and_then(serde_json::Value::as_str)
        .is_some_and(|effort| effort != "none")
}

pub(crate) fn is_nemotron_force_reasoning(reasoning_parser: Option<&str>) -> bool {
    matches!(
        reasoning_parser,
        Some("nemotron_nano" | "nemotron3" | "nemotron_v3")
    )
}

/// Check if reasoning parsing should be disabled based on per-request parameters.
///
/// The thinking rules (the deployment's default mode, then the alias
/// normalization) run first, on a copy of `chat_template_args`, so this
/// decision sees the same mode the prompt was rendered with.
///
/// For a unified v2 family (`unified_family`): Gemma 4 reasoning is opt-in;
/// DeepSeek V4 follows the prompt renderer's mode; any other family turns
/// reasoning off on `thinking: false`. Otherwise
/// [`is_reasoning_disabled_by_request`] decides by the reasoning parser:
/// - kimi_k25/K3: disabled when chat_template_args contains "thinking": false.
/// - Nemotron force-reasoning aliases: disabled when chat_template_args
///   contains "enable_thinking": false. "force_nonempty_content": true does
///   NOT disable parsing (streaming or non-streaming): the parser stays on so
///   reasoning is split from the answer, and reasoning is surfaced as content
///   only when no content was generated.
/// - DeepSeek: follows the same effective mode used by the prompt renderer.
/// - Mistral: disabled unless `reasoning_effort` is present and not `none`.
/// - Gemma 4: reasoning is opt-in and disabled unless chat_template_args
///   explicitly enables thinking. Gemma 4's chat template injects reasoning
///   markers only when enable_thinking=true.
/// - MiniMax M3: disabled when chat_template_args contains
///   "thinking_mode": "disabled", matching SGLang's MiniMax M3 request
///   convention.
pub fn request_disables_reasoning_with_family(
    request: &ChatStreamRequest<'_>,
    tool_parser: Option<&str>,
    reasoning_parser: Option<&str>,
    default_thinking_mode: Option<&str>,
    unified_family: Option<&str>,
) -> bool {
    let mut chat_template_args = request.chat_template_args.cloned();
    // Only a MiniMax M3 deployment reads the constrained-output fact, so the
    // request is inspected only there.
    let constrained_output = is_minimax_m3_family(reasoning_parser, tool_parser)
        && has_minimax_m3_constrained_generation(request.request, request.explicit_guided_decoding);
    ThinkingRules {
        reasoning_parser,
        tool_call_parser: tool_parser,
        default_thinking_mode,
    }
    .apply(
        &mut chat_template_args,
        request.client_thinking_field,
        constrained_output,
    );
    // Unified owns both channels even when only a tool selector was configured.
    // The legacy split path retains its existing request-control behavior.
    if let Some(family) = unified_family {
        let thinking = thinking_bool_from_args(chat_template_args.as_ref());
        return match family {
            "gemma4" => thinking != Some(true),
            "deepseek_v4" | "deepseek_v41" => {
                !deepseek_renderer_reasoning_enabled(chat_template_args.as_ref(), true)
            }
            _ => thinking == Some(false),
        };
    }
    is_reasoning_disabled_by_request(reasoning_parser, chat_template_args.as_ref())
}

/// Whether the request's `chat_template_args` turn the v1 reasoning parser off.
///
/// Unlike [`request_disables_reasoning_with_family`], this reads the arguments
/// as they are, without the thinking rules. The Dynamo frontend's v1 stream
/// path calls it on the request's arguments after the request rules already
/// rewrote them for rendering.
pub fn is_reasoning_disabled_by_request(
    reasoning_parser: Option<&str>,
    chat_template_args: Option<&ChatTemplateArgs>,
) -> bool {
    match reasoning_parser {
        Some("kimi_k25" | "kimi_k3" | "kimi-k3") => {
            thinking_bool_from_args(chat_template_args) == Some(false)
        }
        parser if is_nemotron_force_reasoning(parser) => {
            // `enable_thinking=false` turns reasoning off entirely (streaming
            // and non-streaming). `force_nonempty_content=true` does NOT
            // disable parsing: keeping the parser on splits reasoning from the
            // answer, so a reasoning+answer turn no longer leaks the reasoning
            // text and `</think>` into `content`.
            //
            // The reasoning-*only* move (surface reasoning as content when the
            // answer is empty) happens on both paths, by different means.
            // Non-streaming uses the aggregator flag
            // `move_reasoning_to_content_when_empty`. Streaming cannot retract
            // a reasoning_content delta already sent, so it instead holds
            // reasoning back until it knows whether an answer follows. Both
            // end with the same contract: a reasoning-only turn surfaces its
            // text as `content`.
            thinking_bool_from_args(chat_template_args) == Some(false)
        }
        Some("deepseek_v3" | "deepseek_v3_1") => {
            !deepseek_renderer_reasoning_enabled(chat_template_args, false)
        }
        Some(
            "deepseek_r1" | "deepseek_v3_2" | "deepseek_v4" | "deepseek-v4" | "deepseekv4"
            | "deepseek_v41" | "minimax_m2",
        ) => !deepseek_renderer_reasoning_enabled(chat_template_args, true),
        Some("gemma4") | Some("gemma-4") => {
            thinking_bool_from_args(chat_template_args) != Some(true)
        }
        Some("mistral") => !mistral_reasoning_enabled(chat_template_args),
        Some("minimax_m3") | Some("minimax-m3") => {
            if let Some(args) = chat_template_args
                && let Some(mode) = args.get("thinking_mode").and_then(|v| v.as_str())
            {
                return mode == "disabled";
            }
            false
        }
        _ => false,
    }
}

/// When the request turned reasoning off, fold each choice's
/// `reasoning_content` into the front of its text `content`. A choice whose
/// content is a list of parts keeps both fields as they are.
pub(crate) fn fold_reasoning_into_content<I, S>(
    stream: S,
    reasoning_disabled: bool,
) -> impl Stream<Item = I> + Send
where
    I: ChatStreamItem,
    S: Stream<Item = I> + Send + 'static,
{
    stream.map(move |mut response| {
        if reasoning_disabled && let Some(data) = response.chunk_mut() {
            for choice in &mut data.choices {
                if let Some(reasoning) = choice.delta.reasoning_content.take() {
                    let text = match choice.delta.content.take() {
                        Some(ChatCompletionMessageContent::Text(text)) => text,
                        None => String::new(),
                        Some(other) => {
                            choice.delta.content = Some(other);
                            choice.delta.reasoning_content = Some(reasoning);
                            continue;
                        }
                    };
                    choice.delta.content = Some(ChatCompletionMessageContent::Text(format!(
                        "{reasoning}{text}"
                    )));
                }
            }
        }
        response
    })
}
