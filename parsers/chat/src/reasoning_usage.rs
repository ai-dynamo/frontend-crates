// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The `reasoning_tokens` estimate in the usage chunk.

use std::collections::HashSet;

use dynamo_protocols::types::{
    ChatChoiceStream, CompletionUsage, CreateChatCompletionStreamResponse,
};
use futures::{Stream, StreamExt};

use crate::chat_stream::PolicyStreamItem;

/// Estimates reasoning-token usage from the parser-classified Chat Completion stream.
///
/// This is intentionally chunk-granular: if one decoded chunk contains both reasoning
/// and visible content, every token in that chunk is counted as reasoning. The estimate
/// therefore overcounts visible-content tokens in mixed chunks. A positive count supplied
/// by the backend remains authoritative.
#[derive(Debug, Default)]
struct ReasoningUsageEstimator {
    total: u32,
    active_choices: HashSet<u32>,
}

impl ReasoningUsageEstimator {
    fn observe(&mut self, chunk: &CreateChatCompletionStreamResponse, chunk_tokens: usize) {
        let token_count = chunk_tokens.try_into().unwrap_or(u32::MAX);

        let has_reasoning = chunk
            .choices
            .iter()
            .any(|choice| choice.delta.reasoning_content.is_some());
        let active_without_visible_output = chunk.choices.iter().any(|choice| {
            self.active_choices.contains(&choice.index) && !choice_has_visible_output(choice)
        });
        let usage_chunk_while_reasoning =
            chunk.choices.is_empty() && !self.active_choices.is_empty();

        if token_count > 0
            && (has_reasoning || active_without_visible_output || usage_chunk_while_reasoning)
        {
            self.total = self.total.saturating_add(token_count);
        }

        for choice in &chunk.choices {
            if choice.delta.reasoning_content.is_some() {
                self.active_choices.insert(choice.index);
            } else if choice_has_visible_output(choice) {
                self.active_choices.remove(&choice.index);
            }
        }
    }

    fn annotate(&self, usage: &mut CompletionUsage) {
        let details = usage.completion_tokens_details.get_or_insert_default();
        if details.reasoning_tokens.unwrap_or(0) == 0 {
            details.reasoning_tokens = Some(self.total);
        }
    }
}

fn choice_has_visible_output(choice: &ChatChoiceStream) -> bool {
    choice.delta.content.is_some()
        || choice
            .delta
            .tool_calls
            .as_ref()
            .is_some_and(|calls| !calls.is_empty())
}

/// Write the estimate into every chunk's `usage` as
/// `completion_tokens_details.reasoning_tokens`, unless the backend already
/// sent a positive count.
pub fn annotate_reasoning_usage<I, S>(stream: S) -> impl Stream<Item = I> + Send
where
    I: PolicyStreamItem,
    S: Stream<Item = I> + Send + 'static,
{
    let mut estimator = ReasoningUsageEstimator::default();
    stream.map(move |mut response| {
        let chunk_tokens = response.chunk_tokens();
        if let Some(chunk) = response.chunk_mut() {
            estimator.observe(chunk, chunk_tokens);
            if let Some(usage) = chunk.usage.as_mut() {
                estimator.annotate(usage);
            }
        }
        response
    })
}
