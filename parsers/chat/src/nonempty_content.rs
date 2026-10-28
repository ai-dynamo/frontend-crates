// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The `force_nonempty_content` contract on a stream, and the keep-alive
//! predicate that follows from it.
//!
//! A request with `chat_template_args.force_nonempty_content: true` promises
//! the client a non-empty `content`. When the parser classifies the whole turn
//! as reasoning, the stream holds the reasoning back and sends it as `content`
//! at the end. While it holds, the stream can send no data frames at all, so
//! the HTTP layer turns SSE keep-alive frames on for it.

use std::collections::HashMap;

use dynamo_parsers::chat_stream::ChatStreamItem;
use dynamo_protocols::types::ChatCompletionMessageContent;
use dynamo_renderer::request_rules::ChatTemplateArgs;
use futures::{Stream, StreamExt};

use crate::chat_stream::PolicyStreamItem;
use crate::families::selected_family;
use crate::reasoning_off::is_reasoning_disabled_by_request;
use crate::version::ParserVersion;

#[derive(Default)]
struct DeferredUnifiedChoice {
    pending_reasoning: String,
    pending_content: String,
    saw_visible_output: bool,
    drained: bool,
}

impl DeferredUnifiedChoice {
    fn release_split(&mut self) -> (Option<String>, Option<String>) {
        let content = std::mem::take(&mut self.pending_content);
        let reasoning = std::mem::take(&mut self.pending_reasoning);
        (
            (!content.is_empty()).then_some(content),
            (!reasoning.is_empty()).then_some(reasoning),
        )
    }

    fn drain(&mut self) -> (Option<String>, Option<String>) {
        if self.drained {
            return (None, None);
        }
        self.drained = true;
        if self.saw_visible_output {
            return self.release_split();
        }
        let reasoning = std::mem::take(&mut self.pending_reasoning);
        let content = std::mem::take(&mut self.pending_content);
        if !reasoning.trim().is_empty() {
            (Some(reasoning), None)
        } else {
            ((!content.is_empty()).then_some(content), None)
        }
    }
}

/// Whether a request should surface parsed `reasoning_content` as `content`
/// when no content was generated: any request carrying
/// `force_nonempty_content=true`.
///
/// Deliberately NOT keyed on the model or its reasoning parser. The flag is a
/// request-level contract — "this response will have non-empty content" — so
/// it is honored generically after parsing and before the response is sent,
/// rather than being reimplemented inside each model-specific parser. Keying
/// it on a parser allow-list means every new alias silently loses the
/// behavior until someone edits the list.
///
/// Note it rides in on `chat_template_args` but is NOT consumed by the chat
/// template: the Nemotron template never reads `force_nonempty_content`, and
/// rendering with it set produces a byte-identical prompt. It is a
/// serving-layer flag that upstream happens to transport through the template
/// kwargs, which is why the check belongs in postprocessing and not in a
/// parser. So a client can set it on any model, and doing so is an explicit
/// request for non-empty content — honoring it generically is the intent, not
/// an accident of where the flag is declared.
///
/// Drives both paths. The non-streaming aggregator takes it as
/// `move_reasoning_to_content_when_empty`, and the stream path holds reasoning
/// back so it reaches the same answer at the terminal chunk. Both must use
/// this one predicate or the two paths would disagree on the same input.
///
/// `enable_thinking` is intentionally not consulted here: when thinking is
/// off, reasoning parsing is disabled, so `reasoning_content` is always empty
/// and the move is vacuous rather than suppressed.
pub fn wants_reasoning_as_content_when_empty(
    chat_template_args: Option<&ChatTemplateArgs>,
) -> bool {
    chat_template_args.is_some_and(|args| {
        args.get("force_nonempty_content") == Some(&serde_json::Value::Bool(true))
    })
}

/// Whether this request's stream can withhold every data frame while it
/// buffers, which is the only reason to force SSE keep-alive frames on.
///
/// The buffering runs only when the stream parses reasoning, so a
/// `force_nonempty_content=true` request with reasoning disabled
/// (`enable_thinking=false`) defers nothing and gets no heartbeat. The
/// configured interval is opt-in because some OpenAI-compatible clients do
/// not ignore SSE comment frames.
///
/// Known gap: a guided-JSON bypass also suppresses the deferral, but it is
/// derived from guided-output inspection that is not available at the HTTP
/// layer, so a guided-JSON bypass still enables the heartbeat when nothing is
/// withheld. That direction is conservative — extra comment frames on an
/// opt-in path rather than a silent stream.
///
/// `version` is the deployment's parser version (in Dynamo,
/// `DYN_PARSER_VERSION`).
pub fn stream_can_defer_all_output(
    tool_call_parser: Option<&str>,
    reasoning_parser: Option<&str>,
    chat_template_args: Option<&ChatTemplateArgs>,
    version: ParserVersion,
) -> bool {
    // Unified families share reasoning-only output deferral with the legacy path.
    let has_reasoning_decoder = reasoning_parser.is_some()
        || selected_family(tool_call_parser, reasoning_parser, version).is_some();
    has_reasoning_decoder
        && wants_reasoning_as_content_when_empty(chat_template_args)
        && !is_reasoning_disabled_by_request(reasoning_parser, chat_template_args)
}

/// Apply the request-level non-empty-content contract after a unified parser has
/// already split reasoning, content, and tool calls.
pub(crate) fn defer_unified_reasoning_for_nonempty_content<I, S>(
    stream_in: S,
) -> impl Stream<Item = I> + Send
where
    I: PolicyStreamItem,
    S: Stream<Item = I> + Send + 'static,
{
    async_stream::stream! {
        let mut states: HashMap<u32, DeferredUnifiedChoice> = HashMap::new();
        let mut last_response: Option<I> = None;
        tokio::pin!(stream_in);

        while let Some(mut response) = stream_in.next().await {
            if ChatStreamItem::is_error(&response) {
                yield response;
                return;
            }
            let Some(data) = response.chunk_mut() else {
                yield response;
                continue;
            };
            let mut prefix_choices = Vec::new();

            for choice in &mut data.choices {
                let state = states.entry(choice.index).or_default();
                if choice.delta.content.is_some()
                    || choice.delta.reasoning_content.is_some()
                    || choice.delta.tool_calls.is_some()
                {
                    state.drained = false;
                }

                if !state.saw_visible_output
                    && let Some(reasoning) = choice.delta.reasoning_content.take()
                {
                    state.pending_reasoning.push_str(&reasoning);
                }

                let carries_parts = matches!(
                    choice.delta.content,
                    Some(ChatCompletionMessageContent::Parts(_))
                );
                match choice.delta.content.take() {
                    Some(ChatCompletionMessageContent::Text(text)) => {
                        if !state.saw_visible_output && text.trim().is_empty() {
                            state.pending_content.push_str(&text);
                        } else {
                            if !text.trim().is_empty() {
                                state.saw_visible_output = true;
                            }
                            let mut content = std::mem::take(&mut state.pending_content);
                            content.push_str(&text);
                            choice.delta.content =
                                Some(ChatCompletionMessageContent::Text(content));
                        }
                    }
                    Some(parts @ ChatCompletionMessageContent::Parts(_)) => {
                        choice.delta.content = Some(parts);
                    }
                    None => {}
                }

                let has_tool_calls = choice
                    .delta
                    .tool_calls
                    .as_ref()
                    .is_some_and(|calls| !calls.is_empty());
                if carries_parts || has_tool_calls {
                    state.saw_visible_output = true;
                }
                if carries_parts {
                    state.pending_content.clear();
                }

                if state.saw_visible_output {
                    let (content, reasoning) = state.release_split();
                    if let Some(reasoning) = reasoning {
                        let mut prefix = choice.clone();
                        prefix.delta.role = None;
                        prefix.delta.content = None;
                        prefix.delta.tool_calls = None;
                        prefix.delta.function_call = None;
                        prefix.delta.refusal = None;
                        prefix.delta.reasoning_content = Some(reasoning);
                        prefix.finish_reason = None;
                        prefix.logprobs = None;
                        prefix_choices.push(prefix);
                    }
                    if let Some(content) = content {
                        match choice.delta.content.as_mut() {
                            Some(ChatCompletionMessageContent::Text(existing)) => {
                                existing.insert_str(0, &content);
                            }
                            Some(ChatCompletionMessageContent::Parts(_)) | None => {
                                let mut prefix = choice.clone();
                                prefix.delta.role = None;
                                prefix.delta.content =
                                    Some(ChatCompletionMessageContent::Text(content));
                                prefix.delta.tool_calls = None;
                                prefix.delta.function_call = None;
                                prefix.delta.refusal = None;
                                prefix.delta.reasoning_content = None;
                                prefix.finish_reason = None;
                                prefix.logprobs = None;
                                prefix_choices.push(prefix);
                            }
                        }
                    }
                }

                if choice.finish_reason.is_some() {
                    let (content, reasoning) = state.drain();
                    if let Some(content) = content {
                        match choice.delta.content.as_mut() {
                            Some(ChatCompletionMessageContent::Text(existing)) => {
                                existing.push_str(&content);
                            }
                            Some(ChatCompletionMessageContent::Parts(_)) => {}
                            None => {
                                choice.delta.content =
                                    Some(ChatCompletionMessageContent::Text(content));
                            }
                        }
                    }
                    if let Some(reasoning) = reasoning {
                        choice
                            .delta
                            .reasoning_content
                            .get_or_insert_default()
                            .push_str(&reasoning);
                    }
                }
            }

            if !data.choices.is_empty() {
                last_response = Some(response.clone());
            }
            for prefix_choice in prefix_choices {
                let mut prefix_response = response.clone();
                prefix_response.scrub_synthetic();
                if let Some(prefix_data) = prefix_response.chunk_mut() {
                    prefix_data.choices = vec![prefix_choice];
                }
                prefix_response.clear_id();
                yield prefix_response;
            }
            yield response;
        }

        let mut indices: Vec<_> = states.keys().copied().collect();
        indices.sort_unstable();
        let flushed: Vec<_> = indices
            .into_iter()
            .filter_map(|index| {
                let (content, reasoning) = states.get_mut(&index)?.drain();
                (content.is_some() || reasoning.is_some())
                    .then_some((index, content, reasoning))
            })
            .collect();
        if !flushed.is_empty()
            && let Some(mut response) = last_response
            && {
                response.scrub_synthetic();
                response.chunk().is_some()
            }
            && let Some(data) = response.chunk_mut()
            && let Some(template) = data.choices.first().cloned()
        {
            data.choices = flushed
                .into_iter()
                .map(|(index, content, reasoning)| {
                    let mut choice = template.clone();
                    choice.index = index;
                    choice.delta.role = None;
                    choice.delta.content = content.map(ChatCompletionMessageContent::Text);
                    choice.delta.tool_calls = None;
                    choice.delta.function_call = None;
                    choice.delta.refusal = None;
                    choice.delta.reasoning_content = reasoning;
                    choice.finish_reason = None;
                    choice.logprobs = None;
                    choice
                })
                .collect();
            yield response;
        }
    }
}

/// Hold the trailing usage-only chunk until every parser recovery chunk has
/// been emitted. A truncated upstream stream can end without
/// `finish_reason`, so reasoning recovery happens at EOF; forwarding usage
/// immediately would put that recovered content after the chunk clients
/// treat as the stream trailer. A transport error discards the pending
/// trailer because the response did not complete successfully.
///
/// A transport error here is an item whose annotation carries an error
/// ([`PolicyStreamItem::has_error`]), whatever its event.
pub(crate) fn hold_usage_until_stream_end<I, S>(stream: S) -> impl Stream<Item = I> + Send
where
    I: PolicyStreamItem,
    S: Stream<Item = I> + Send + 'static,
{
    async_stream::stream! {
        tokio::pin!(stream);
        let mut pending_usage = None;
        let mut transport_failed = false;
        while let Some(response) = stream.next().await {
            if response.has_error() {
                transport_failed = true;
                pending_usage = None;
                yield response;
                continue;
            }
            let is_usage_only = response
                .chunk()
                .is_some_and(|data| data.choices.is_empty() && data.usage.is_some());
            if is_usage_only {
                if let Some(previous) = pending_usage.replace(response) {
                    yield previous;
                }
            } else {
                yield response;
            }
        }
        if !transport_failed && let Some(usage) = pending_usage {
            yield usage;
        }
    }
}
