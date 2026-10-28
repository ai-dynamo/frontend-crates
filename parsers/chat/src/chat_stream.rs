// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The parse stage of a chat stream: from detokenized chunks to reasoning,
//! content, and tool-call chunks, with the request's stream policies.
//!
//! [`parse_chat_stream`] picks the stages for one request and chains them. A
//! request on a unified v2 route runs the v2 stream adapter, then the
//! tool-call policy, the reasoning-off fold, the `reasoning_tokens` estimate,
//! and the `force_nonempty_content` deferral. A request on the v1 route runs
//! dynamo-parsers' reasoning stream, the usage hold, the estimate, the v1
//! tool-call jail, and the tool-call policy.
//!
//! The v1 jail is a stage the caller passes in. Dynamo wraps the jail from
//! dynamo-parsers with its own per-chunk metadata handling, which stays in
//! Dynamo.
//!
//! The stages run on the caller's own stream item, so Dynamo keeps its
//! per-chunk fields. The item implements dynamo-parsers'
//! [`ChatStreamItem`], this crate's [`UnifiedStreamItem`], and
//! [`PolicyStreamItem`].

use dynamo_parsers::chat_stream::ChatStreamItem;
use dynamo_parsers::reasoning::stream::{
    ReasoningStreamOptions, guided_json_reasoning_start_token, is_force_reasoning_parser,
    is_minimax_m3_reasoning_parser, parse_reasoning_content_from_stream,
    skips_guided_json_when_prompt_injected, skips_structured_response_when_prompt_injected,
    strip_leading_reasoning_start_from_stream, structured_response_supports_sglang_reasoning_gate,
    supports_reasoning_before_guided_json,
};
use dynamo_parsers::tool_calling::ToolDefinition;
use dynamo_protocols::types::ChatCompletionToolChoiceOption;
use dynamo_renderer::request_rules::{has_structured_response_format, is_minimax_m3_family};
use futures::Stream;
use futures::stream::BoxStream;

use crate::guided::GuidedToolConstraint;
use crate::nonempty_content::{
    defer_unified_reasoning_for_nonempty_content, hold_usage_until_stream_end,
    wants_reasoning_as_content_when_empty,
};
use crate::prefill::PromptReasoningPrefill;
use crate::reasoning_off::{
    ChatStreamRequest, fold_reasoning_into_content, is_nemotron_force_reasoning,
    is_reasoning_disabled_by_request, request_disables_reasoning_with_family,
};
use crate::reasoning_usage::annotate_reasoning_usage;
use crate::route::ToolProcessingRoute;
use crate::stream_item::UnifiedStreamItem;
use crate::tool_call_policy::apply_tool_call_response_policy;
use crate::tools::tool_call_parsing_enabled;
use crate::unified_stream::{
    StreamPrefill, UnifiedRequestPolicy, apply_stream_with_policy, stream_prefill,
};
use crate::version::ParserVersion;

/// What the stream policies read from a stream item beyond
/// [`ChatStreamItem`].
///
/// [`ChatStreamItem::scrub_synthetic`] must clear the annotation's `event`,
/// `comment`, and `error`, and every per-chunk field: usage, token metrics,
/// tool-call completion evidence, the extension field, and prompt logprobs.
/// In Dynamo this is `scrub_synthetic_chunk_metadata`.
pub trait PolicyStreamItem: ChatStreamItem {
    /// Whether the annotation carries an error, whatever its event.
    ///
    /// The usage hold drops a pending usage trailer after such an item. The
    /// other stages end on [`ChatStreamItem::is_error`], which reads the
    /// event. In Dynamo, this is `Annotated::error.is_some()`.
    fn has_error(&self) -> bool;

    /// Clear the annotation's `id`. A synthetic chunk split off before its
    /// source chunk must not repeat the source's SSE event id.
    fn clear_id(&mut self);

    /// The number of generated tokens this chunk carries, or 0 when the
    /// chunk does not say. In Dynamo, this is `llm_metrics.chunk_tokens`.
    fn chunk_tokens(&self) -> usize;
}

/// The deployment's parser names.
#[derive(Clone, Copy, Debug, Default)]
pub struct ChatStreamParsers<'a> {
    /// The tool-call parser name, such as `"hermes"`.
    pub tool_call_parser: Option<&'a str>,
    /// The reasoning parser name, such as `"qwen3"`.
    pub reasoning_parser: Option<&'a str>,
}

/// The decisions the caller made for one request before its stream starts.
#[derive(Clone, Debug)]
pub struct ChatStreamPlan {
    /// The parser path, from [`tool_processing_route`](crate::route::tool_processing_route).
    pub route: ToolProcessingRoute,
    /// The tool-output grammar the request installed.
    pub guided_tool_constraint: GuidedToolConstraint,
    /// Whether the rendered prompt already opened a reasoning block.
    pub prompt_reasoning: PromptReasoningPrefill,
    /// The request's effective tools, from
    /// [`effective_tools`](crate::tools::effective_tools) and
    /// [`convert_tools`](crate::tools::convert_tools). Empty when it has none.
    pub tool_definitions: Vec<ToolDefinition>,
    /// The deployment's parser version (in Dynamo, `DYN_PARSER_VERSION`).
    pub parser_version: ParserVersion,
    /// The operator turned off streaming of guided tool calls (in Dynamo, a
    /// falsy `DYN_ENABLE_GUIDED_TOOL_STREAMING`).
    pub guided_tool_streaming_disabled: bool,
}

/// What the v1 tool-call jail needs for one request.
#[derive(Clone, Debug)]
pub struct JailRequest {
    /// The tool-call parser the jail runs. The route chose it.
    pub tool_call_parser: Option<String>,
    /// The request's `tool_choice`.
    pub tool_choice: Option<ChatCompletionToolChoiceOption>,
    /// The request's effective tools, or `None` when it has none.
    pub tool_definitions: Option<Vec<ToolDefinition>>,
    /// The tool-output grammar is a structural tag.
    pub uses_tool_call_structural_tag: bool,
    /// Release guided calls as they arrive instead of at the closing brace.
    pub guided_streaming: bool,
}

/// Whether a forced tool_choice's installed JSON grammar should release
/// calls incrementally, or fall back to buffer-to-completion.
///
/// The single owner of that decision, for both streaming paths that can act on
/// it: the v1 jail and the unified v2 adapter. [`parse_chat_stream`] decides it
/// once per request and hands the answer to whichever path runs.
fn guided_tool_streaming_release(installs_guided_json: bool, rollback_disabled: bool) -> bool {
    installs_guided_json && !rollback_disabled
}

/// Parse one request's stream of detokenized chat chunks.
///
/// The caller decided the route and the guided constraint before the request
/// reached the backend (see [`ChatStreamPlan`]). `jail` runs the v1 tool-call
/// jail on the stream it is given; only a [`ToolProcessingRoute::LegacyJail`]
/// request calls it.
///
/// The caller still adds the assistant role to each choice's first delta
/// afterwards, with
/// [`normalize_chat_stream_roles`](crate::tool_call_policy::normalize_chat_stream_roles).
pub fn parse_chat_stream<I, S, J, O>(
    stream: S,
    parsers: ChatStreamParsers<'_>,
    request: ChatStreamRequest<'_>,
    plan: ChatStreamPlan,
    jail: J,
) -> BoxStream<'static, I>
where
    I: PolicyStreamItem + UnifiedStreamItem,
    S: Stream<Item = I> + Send + 'static,
    J: FnOnce(BoxStream<'static, I>, JailRequest) -> O,
    O: Stream<Item = I> + Send + 'static,
{
    let ChatStreamPlan {
        route: tool_processing_route,
        guided_tool_constraint,
        prompt_reasoning: prompt_prefill,
        tool_definitions,
        parser_version,
        guided_tool_streaming_disabled,
    } = plan;
    let ChatStreamParsers {
        tool_call_parser,
        reasoning_parser,
    } = parsers;
    let prompt_injected_reasoning = prompt_prefill.for_route(&tool_processing_route);
    let uses_tool_call_structural_tag = guided_tool_constraint.uses_structural_tag();
    let defer_reasoning_for_nonempty_content =
        wants_reasoning_as_content_when_empty(request.chat_template_args);
    // Two different streaming paths can release a grammar-constrained tool call
    // before its payload closes: the unified (v2) adapter below and the jail near
    // the end of this function. Both must obey the same rollback lever, so the
    // decision is made ONCE, here, and handed to whichever path runs.
    let guided_tool_streaming = guided_tool_streaming_release(
        guided_tool_constraint.installs_guided_json(),
        guided_tool_streaming_disabled,
    );
    let tool_definitions = (!tool_definitions.is_empty()).then_some(tool_definitions);

    if let ToolProcessingRoute::Unified(family) = &tool_processing_route {
        let reasoning_disabled = request_disables_reasoning_with_family(
            &request,
            tool_call_parser,
            reasoning_parser,
            None,
            Some(family),
        );
        let unified: BoxStream<'static, I> = Box::pin(apply_stream_with_policy(
            stream,
            tool_definitions,
            guided_tool_constraint,
            StreamPrefill {
                starting_state: stream_prefill(family, prompt_injected_reasoning),
                prefix: prompt_prefill.prefix(),
            },
            family,
            guided_tool_streaming,
            UnifiedRequestPolicy {
                reasoning_disabled,
                structured_response: has_structured_response_format(request.request),
                parsers_v2_selected: parser_version == ParserVersion::V2,
            },
        ));
        return apply_unified_response_policies(
            unified,
            tool_call_parsing_enabled(request.request),
            defer_reasoning_for_nonempty_content,
            reasoning_disabled,
        );
    }

    // Guided output may be bare JSON or `reasoning</think>JSON`. Supported
    // parsers inspect the stream shape before deciding whether to parse it.
    let is_guided_tool_choice = matches!(
        request.request.tool_choice,
        Some(ChatCompletionToolChoiceOption::Required)
            | Some(ChatCompletionToolChoiceOption::Named(_))
    );
    let is_structured_response = has_structured_response_format(request.request);
    let is_minimax_m3_explicit_guided_output =
        is_minimax_m3_family(reasoning_parser, tool_call_parser)
            && request.explicit_guided_decoding;
    let is_guided_output =
        is_guided_tool_choice || is_structured_response || is_minimax_m3_explicit_guided_output;
    // Force parsers opt in by capability; prompt-seeded parsers opt in per request.
    // Structural-tag tool formats do not use this guided-JSON detection path.
    let inspect_force_reasoning_guided_output = is_guided_output
        && !uses_tool_call_structural_tag
        && supports_reasoning_before_guided_json(reasoning_parser);
    let inspect_prompt_injected_guided_output = is_guided_output
        && prompt_injected_reasoning
        && !uses_tool_call_structural_tag
        && (skips_guided_json_when_prompt_injected(reasoning_parser)
            || (is_structured_response
                && skips_structured_response_when_prompt_injected(reasoning_parser)));
    let inspect_unsupported_structured_response_reasoning_gate = is_structured_response
        && !uses_tool_call_structural_tag
        && !structured_response_supports_sglang_reasoning_gate(reasoning_parser);
    let bypass_reasoning_for_bare_guided_json = inspect_force_reasoning_guided_output
        || inspect_prompt_injected_guided_output
        || inspect_unsupported_structured_response_reasoning_gate;
    // Preserve the older bypass for force-reasoning parsers not yet opted in.
    let skip_reasoning_for_guided_json = is_guided_output
        && !uses_tool_call_structural_tag
        && is_force_reasoning_parser(reasoning_parser)
        && !inspect_force_reasoning_guided_output;

    let reasoning_disabled_by_request =
        is_reasoning_disabled_by_request(reasoning_parser, request.chat_template_args);

    // Try to parse reasoning content only if parser is configured.
    let should_parse_reasoning = reasoning_parser.is_some()
        && !reasoning_disabled_by_request
        && !skip_reasoning_for_guided_json;
    let should_strip_disabled_reasoning_start = reasoning_disabled_by_request
        && is_nemotron_force_reasoning(reasoning_parser)
        && !skip_reasoning_for_guided_json;
    let guided_reasoning_start_token =
        if should_parse_reasoning && bypass_reasoning_for_bare_guided_json {
            guided_json_reasoning_start_token(reasoning_parser)
        } else {
            None
        };

    // A bracket-prefixed reasoning opener is indistinguishable from JSON
    // until enough bytes arrive. Strip only a configured complete marker;
    // nonmatching bare JSON is restored unchanged for shape detection.
    let stream: BoxStream<'static, I> = if let Some(start_token) = guided_reasoning_start_token {
        Box::pin(strip_leading_reasoning_start_from_stream(
            stream,
            start_token,
        ))
    } else {
        Box::pin(stream)
    };

    // On MiniMax M3, a top-level `guided_*` grammar can own the output from
    // token 0 (unless a vLLM-native reasoning parser delays it), so a
    // prompt-prefilled `<mm:think>` does not prove the output is reasoning:
    // `guided_regex` "Paris" would land in `reasoning_content` with empty
    // `content`. Leave the parser unseeded so its own rules decide: a worker
    // that delays the grammar still ends reasoning with `</mm:think>`, which
    // dangling-end recovery splits as reasoning, and marker-free output is
    // decided as content when the parser is finalized.
    //
    // The cost is truncation under explicitly enabled thinking. If such a
    // worker's output ends before `</mm:think>` for any reason (a `length`
    // finish, a stop string, or a stream with no `finish_reason`), the held
    // reasoning has no end marker to split on, so finalization delivers it
    // as `content`, which does not match the grammar. Keeping it as
    // reasoning means seeding the parser only where the worker delays the
    // grammar. Workers publish that today for tool grammars only: vLLM sets
    // `tool_call_structural_tag_excludes_reasoning` when its engine-side
    // reasoning parser delays the grammar, and SGLang sets
    // `tool_call_structural_tag_reasoning_gate` when it honors a per-request
    // `require_reasoning`, which `guided_*` alone does not request.
    let seed_reasoning_parser = prompt_injected_reasoning
        && !(is_minimax_m3_reasoning_parser(reasoning_parser) && request.explicit_guided_decoding);

    // Only a force_nonempty_content request needs the deferral and the EOF
    // flush of a truncated `<think>` prefix; gating on the request keeps the
    // per-token clone and the finish_reasoning_stream() flush off every
    // other request's path (a MiniMax M3 parser is still finalized on its
    // terminal chunk, or at end of stream without one). Same predicate the
    // aggregator uses, so the streaming and non-streaming paths cannot
    // disagree.
    let stream: BoxStream<'static, I> = match reasoning_parser {
        Some(parser) if should_parse_reasoning => Box::pin(parse_reasoning_content_from_stream(
            stream,
            parser.to_string(),
            ReasoningStreamOptions {
                prompt_injected_reasoning: seed_reasoning_parser,
                bypass_bare_guided_json: bypass_reasoning_for_bare_guided_json,
                defer_reasoning_for_nonempty_content,
            },
        )),
        _ if should_strip_disabled_reasoning_start => {
            Box::pin(strip_leading_reasoning_start_from_stream(stream, "<think>"))
        }
        _ => stream,
    };
    // A MiniMax M3 parser can also release held text in an end-of-stream
    // chunk, so its usage is held too.
    let stream: BoxStream<'static, I> = if defer_reasoning_for_nonempty_content
        || should_strip_disabled_reasoning_start
        || (should_parse_reasoning && is_minimax_m3_reasoning_parser(reasoning_parser))
    {
        Box::pin(hold_usage_until_stream_end(stream))
    } else {
        stream
    };
    let stream: BoxStream<'static, I> = if should_parse_reasoning {
        Box::pin(annotate_reasoning_usage(stream))
    } else {
        stream
    };

    let transformed_stream: BoxStream<'static, I> = match tool_processing_route {
        ToolProcessingRoute::LegacyJail(effective_tool_call_parser) => {
            // A forced tool_choice installed a JSON grammar, so the jail may release
            // calls as they arrive instead of buffering to the closing brace. The
            // jail keeps its own native fallback, so a backend that ignores the
            // grammar (MiniMax M2 emits XML under `required`) still parses normally.
            // Same request-scoped decision the unified path above was given.
            Box::pin(jail(
                stream,
                JailRequest {
                    tool_call_parser: effective_tool_call_parser,
                    tool_choice: request.request.tool_choice.clone(),
                    tool_definitions,
                    uses_tool_call_structural_tag,
                    guided_streaming: guided_tool_streaming,
                },
            ))
        }
        ToolProcessingRoute::PassThrough => stream,
        ToolProcessingRoute::Unified(_) => {
            unreachable!("unified routes return before legacy response processing")
        }
    };

    apply_tool_call_response_policy(
        transformed_stream,
        tool_call_parsing_enabled(request.request),
    )
}

/// The policies after the unified v2 adapter, in Dynamo's order: the
/// tool-call policy, the reasoning-off fold, the `reasoning_tokens` estimate,
/// then the `force_nonempty_content` deferral and its usage hold.
fn apply_unified_response_policies<I, S>(
    stream: S,
    emit_tool_calls: bool,
    defer_reasoning_for_nonempty_content: bool,
    reasoning_disabled: bool,
) -> BoxStream<'static, I>
where
    I: PolicyStreamItem,
    S: Stream<Item = I> + Send + 'static,
{
    let stream = apply_tool_call_response_policy(stream, emit_tool_calls);
    let stream = fold_reasoning_into_content(stream, reasoning_disabled);
    // Observe parser classification before force_nonempty deferral removes the
    // reasoning delta. The annotated usage trailer can still be held below until
    // every deferred recovery chunk has been emitted.
    let stream: BoxStream<'static, I> = Box::pin(annotate_reasoning_usage(stream));
    let stream: BoxStream<'static, I> = if defer_reasoning_for_nonempty_content {
        Box::pin(defer_unified_reasoning_for_nonempty_content(stream))
    } else {
        stream
    };
    if defer_reasoning_for_nonempty_content {
        Box::pin(hold_usage_until_stream_end(stream))
    } else {
        stream
    }
}

#[cfg(test)]
#[path = "stream_policies_tests.rs"]
mod tests;
