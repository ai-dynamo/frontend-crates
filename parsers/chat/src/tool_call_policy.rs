// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The request's tool-call policy on parsed output, and the assistant role on
//! each choice's first delta.

use std::collections::HashSet;

use dynamo_parsers::chat_stream::ChatStreamItem;
use dynamo_protocols::types::{FinishReason, Role};
use futures::{Stream, StreamExt};

/// Enforce the request's tool-call policy after model-specific parsing.
///
/// Kimi K3 must keep its jail active even when the request does not permit
/// tools because the same parser unwraps ordinary XTML response channels.
/// The jail can still emit structured tool-call deltas while doing that
/// decoding, so parser activation alone is not an output-policy boundary.
/// Apply the policy to the shared stream before the HTTP streaming and
/// non-streaming paths diverge. This policy deliberately fails closed: when
/// a decoder consumes a tool-only turn, suppressing the unauthorized call
/// may leave an empty assistant turn with `finish_reason: stop`. Reconstructing
/// parser-specific wire markup as assistant content would leak internal
/// protocol tokens and could still be mistaken for an actionable call.
///
/// `tool_call_parsing_enabled` comes from
/// [`tool_call_parsing_enabled`](crate::tools::tool_call_parsing_enabled).
pub fn apply_tool_call_response_policy<I, S>(
    stream: S,
    tool_call_parsing_enabled: bool,
) -> futures::stream::BoxStream<'static, I>
where
    I: ChatStreamItem,
    S: Stream<Item = I> + Send + 'static,
{
    if tool_call_parsing_enabled {
        return Box::pin(stream);
    }

    Box::pin(stream.map(|mut response| {
        if let Some(data) = response.chunk_mut() {
            for choice in &mut data.choices {
                choice.delta.tool_calls = None;
                if choice.finish_reason == Some(FinishReason::ToolCalls) {
                    choice.finish_reason = Some(FinishReason::Stop);
                }
            }
        }
        response
    }))
}

/// Ensure the first emitted delta for each choice carries the assistant role.
///
/// This runs after reasoning and tool-call parsing so a parser that buffers the
/// original role-bearing delta cannot leave downstream consumers without a role.
/// Every later delta of the choice has no role.
pub fn normalize_chat_stream_roles<I, S>(stream: S) -> impl Stream<Item = I> + Send + 'static
where
    I: ChatStreamItem,
    S: Stream<Item = I> + Send + 'static,
{
    let mut role_emitted_choices = HashSet::new();

    stream.map(move |mut response| {
        if let Some(data) = response.chunk_mut() {
            for choice in &mut data.choices {
                choice.delta.role = role_emitted_choices
                    .insert(choice.index)
                    .then_some(Role::Assistant);
            }
        }
        response
    })
}
