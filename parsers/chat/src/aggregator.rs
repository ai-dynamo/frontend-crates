// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The non-streaming finalize: the chunks of a `stream: false` reply become one
//! chat completion.
//!
//! Moved from dynamo `lib/llm/src/protocols/openai/chat_completions/aggregator.rs`,
//! with the v2 tool-only batch parser and the guided-JSON conversion from
//! `tool_parser_v2.rs` beside it.
//!
//! [`DeltaAggregator`] folds each chunk's OpenAI part and tool-call completion
//! evidence. [`DeltaAggregator::finish`] then applies the rules: it re-parses the
//! text when the stream carried no tool call, drops cut-off and malformed calls,
//! applies `parallel_tool_calls: false`, and can move reasoning to content.
//!
//! Dynamo's chunks also carry `nvext` and prompt logprobs, which this crate cannot
//! name. Dynamo folds those itself around [`DeltaAggregator::push`], so the
//! aggregator takes only the OpenAI chunk.

use std::collections::{BTreeMap, HashMap};

use dynamo_parsers::tool_calling::{
    ToolCallResponse, ToolDefinition, try_tool_call_parse_aggregate_finalize,
};
use dynamo_protocols::types::{
    ChatCompletionMessageContent, ChatCompletionMessageToolCall, CreateChatCompletionRequest,
    CreateChatCompletionResponse, CreateChatCompletionStreamResponse, FunctionCall, FunctionType,
    ResponseFormat,
};

use crate::families::selected_family;
use crate::guided::GuidedToolConstraint;
use crate::nonempty_content::wants_reasoning_as_content_when_empty;
use crate::reasoning_off::{ChatStreamRequest, request_disables_reasoning_with_family};
use crate::route::{InvalidRequest, validate_tool_request_mode};
use crate::stream_item::ToolCallCompletion;
use crate::tools::tool_call_parsing_enabled;
use crate::unified_stream::UnifiedRequestPolicy;
use crate::version::{PARSER_VERSION_ENV, ParserVersion};

fn is_harmony_parser(parser: &str) -> bool {
    parser == "harmony"
}

fn is_kimi_k3_parser(parser: &str) -> bool {
    matches!(parser, "kimi_k3" | "kimi-k3")
}

fn contains_harmony_protocol(text: &str) -> bool {
    text.contains("<|channel|>")
}

/// Quote/escape-aware replacement for `dynamo_parsers::tool_calling::detect_tool_call_start`,
/// which does a naive substring search and false-positives on a marker token that only
/// appears as a quoted JSON string value (e.g. a tool argument literally containing the
/// text `<tool_call>`). Looks up the real per-family marker tokens for `parser` from
/// `dynamo_parsers`' own parser map (falling back to its "default" key exactly as
/// `detect_tool_call_start` does for `None`/empty), then checks each with
/// `unified_parser::first_unquoted_marker_position` so a marker embedded inside a quoted string
/// is not misread as native tool-call markup.
fn contains_native_tool_call_marker(content: &str, parser: &str) -> bool {
    crate::unified_stream::first_unquoted_native_tool_call_marker(content, parser).is_some()
}

/// Drops any recovered native-fallback calls that don't match the forced
/// `tool_name` from a `GuidedJsonNamed` constraint. Guided-JSON failure only
/// tells us the *shape* was wrong; it says nothing about which tool the
/// native markup named, so a malformed guided output that happens to embed
/// native markup for a different tool must never be handed back to a client
/// that pinned `tool_choice` to one specific function.
fn filter_calls_to_forced_tool_name(
    calls: Vec<dynamo_parsers::tool_calling::ToolCallResponse>,
    constraint: &GuidedToolConstraint,
) -> Vec<dynamo_parsers::tool_calling::ToolCallResponse> {
    let GuidedToolConstraint::GuidedJsonNamed { tool_name } = constraint else {
        return calls;
    };
    let (matched, dropped): (Vec<_>, Vec<_>) = calls
        .into_iter()
        .partition(|call| &call.function.name == tool_name);
    if !dropped.is_empty() {
        tracing::warn!(
            tool_name,
            dropped = dropped.len(),
            "dropped native-fallback tool call(s) whose name did not match the tool_choice-forced tool_name"
        );
    }
    matched
}

async fn parse_complete_tool_output(
    content: &str,
    parser: &str,
    constraint: &GuidedToolConstraint,
    tools: &[dynamo_parsers::tool_calling::ToolDefinition],
    version: ParserVersion,
) -> anyhow::Result<(
    Vec<dynamo_parsers::tool_calling::ToolCallResponse>,
    Option<String>,
)> {
    let parser = match parser {
        "deepseek-v4" | "deepseekv4" => "deepseek_v4",
        parser => parser,
    };
    if version == ParserVersion::V2 && !tool_parser_v2::supports_family(parser) {
        anyhow::bail!(
            "{PARSER_VERSION_ENV}=2 was requested, but parser {parser:?} has no compatible v2 implementation"
        );
    }
    if constraint.installs_guided_json() {
        match tool_parser_v2::parse_complete_guided_json(content, constraint) {
            Ok(calls) => return Ok((calls, Some(String::new()))),
            Err(guided_error) => {
                if !contains_native_tool_call_marker(content, parser) {
                    return Err(guided_error);
                }
                tracing::warn!(
                    parser,
                    why = "guided_json_reconstructed_but_native_markup_observed",
                    recovered_bytes = content.len(),
                    "falling back to the configured native tool parser"
                );
            }
        }
    }

    let result = if version == ParserVersion::V2 && tool_parser_v2::supports_family(parser) {
        tool_parser_v2::parse_complete(content, Some(tools), parser)
            .map(|(calls, normal)| (calls, Some(normal)))
    } else {
        try_tool_call_parse_aggregate_finalize(content, Some(parser), None).await
    };

    result.and_then(|(calls, normal)| {
        let filtered = filter_calls_to_forced_tool_name(calls, constraint);
        if filtered.is_empty()
            && matches!(
                constraint,
                GuidedToolConstraint::GuidedJsonNamed { .. }
            )
        {
            // Mirror the "no native markup found" behavior above: a fallback
            // that recovered nothing usable for the forced tool is treated
            // the same as a fallback that found nothing at all.
            return Err(anyhow::anyhow!(
                "native tool-call fallback recovered no call matching the tool_choice-forced tool name"
            ));
        }
        Ok((filtered, normal))
    })
}

/// What the finalize reads from the request and the deployment: dynamo's
/// `ParsingOptions` fields that the aggregator uses, and the parser version.
#[derive(Clone, Debug)]
pub struct AggregatorOptions {
    pub tool_call_parser: Option<String>,
    pub reasoning_parser: Option<String>,
    /// A disabled thinking request must not regain a reasoning channel during raw batch recovery.
    pub reasoning_disabled: bool,
    /// JSON response formatting is a content contract, separate from guided tool JSON.
    pub structured_response: bool,
    /// Some model parsers (currently Harmony) must still run during non-streaming
    /// aggregation to remove model-internal channel markup even when the request
    /// forbids tool calls. This flag suppresses any structured calls they discover.
    pub suppress_tool_calls: bool,
    /// Exact tool-output grammar installed while preprocessing this request.
    pub guided_tool_constraint: GuidedToolConstraint,
    /// The request's `parallel_tool_calls`. `Some(false)` caps each choice to one tool call.
    pub parallel_tool_calls: Option<bool>,
    /// When the message has no content and no tool calls, move the reasoning into
    /// `content`. Dynamo sets it for `force_nonempty_content=true` requests.
    pub move_reasoning_to_content_when_empty: bool,
    /// The request's tool schemas, so batch parsing can type arguments as the stream does.
    pub tools: Vec<ToolDefinition>,
    /// The deployment's parser version (`DYN_PARSER_VERSION` in dynamo).
    pub parser_version: ParserVersion,
}

impl Default for AggregatorOptions {
    /// No parsers, no request options, and [`ParserVersion::Auto`], which is dynamo's
    /// version when `DYN_PARSER_VERSION` is unset.
    fn default() -> Self {
        Self {
            tool_call_parser: None,
            reasoning_parser: None,
            reasoning_disabled: false,
            structured_response: false,
            suppress_tool_calls: false,
            guided_tool_constraint: GuidedToolConstraint::None,
            parallel_tool_calls: None,
            move_reasoning_to_content_when_empty: false,
            tools: Vec::new(),
            parser_version: ParserVersion::Auto,
        }
    }
}

impl AggregatorOptions {
    /// Enforce request-level tool-call permission while preserving independent
    /// reasoning parsing and any parser needed for whole-response decoding.
    /// `tool_call_parser` originates in model configuration, so HTTP handlers
    /// must narrow it to requests that actually permit tool calls. Whole-response
    /// decoders are retained because they also remove internal channel markup from
    /// ordinary content; `suppress_tool_calls` remains the output policy boundary.
    ///
    /// Moved from dynamo's `ParsingOptions` (`lib/llm/src/protocols/openai.rs`).
    pub fn with_tool_call_parsing_enabled(mut self, enabled: bool) -> Self {
        if !enabled {
            self.suppress_tool_calls = true;
            let whole_response_decoder = matches!(
                self.tool_call_parser.as_deref(),
                Some("harmony" | "kimi_k3" | "kimi-k3")
            ) || selected_family(
                self.tool_call_parser.as_deref(),
                self.reasoning_parser.as_deref(),
                self.parser_version,
            )
            .is_some();
            // Removing an incompatible tool parser must not manufacture a reasoning-only V2 route.
            // TODO: Separate pair identity from legacy decoder activation: with suppressed calls,
            // hermes + qwen3 must retain literal tool-call text in batch as streaming does.
            let reasoning_would_select_v2 =
                selected_family(None, self.reasoning_parser.as_deref(), self.parser_version)
                    .is_some();
            if !whole_response_decoder && !reasoning_would_select_v2 {
                self.tool_call_parser = None;
            }
            self.guided_tool_constraint = GuidedToolConstraint::None;
        }
        self
    }
}

impl AggregatorOptions {
    /// The options the Dynamo frontend's HTTP handlers build for `request`
    /// before they aggregate its reply: the deployment's parsers narrowed to
    /// what the request permits (`apply_request_tool_call_parsing_options`
    /// in dynamo-llm's `http/service.rs`), with the request's
    /// `parallel_tool_calls` and `force_nonempty_content`. The Chat
    /// Completions and Responses handlers use these as they are; the
    /// Messages handler then applies [`Self::for_messages`].
    ///
    /// `default_thinking_mode` is the deployment's, as the stream path reads
    /// it. The caller decides the tool-output grammar, as for the stream
    /// path's `ChatStreamPlan`, and passes it as `guided_tool_constraint`.
    /// `tool_definitions` are the request's effective tools, validated by
    /// the caller.
    pub fn for_request(
        request: &ChatStreamRequest<'_>,
        tool_call_parser: Option<&str>,
        reasoning_parser: Option<&str>,
        default_thinking_mode: Option<&str>,
        guided_tool_constraint: GuidedToolConstraint,
        tool_definitions: Vec<ToolDefinition>,
        parser_version: ParserVersion,
    ) -> Self {
        let unified_family = selected_family(tool_call_parser, reasoning_parser, parser_version);
        let reasoning_disabled = request_disables_reasoning_with_family(
            request,
            tool_call_parser,
            reasoning_parser,
            default_thinking_mode,
            unified_family,
        );
        let options = Self {
            tool_call_parser: tool_call_parser.map(str::to_owned),
            reasoning_parser: reasoning_parser.map(str::to_owned),
            reasoning_disabled,
            structured_response: has_structured_response_format(request.request),
            suppress_tool_calls: false,
            guided_tool_constraint,
            parallel_tool_calls: request.request.parallel_tool_calls,
            move_reasoning_to_content_when_empty: wants_reasoning_as_content_when_empty(
                request.chat_template_args,
            ),
            tools: Vec::new(),
            parser_version,
        };
        let mut options =
            options.with_tool_call_parsing_enabled(tool_call_parsing_enabled(request.request));
        options.tools = tool_definitions;
        options
    }

    /// The Messages handler's options: it aggregates without the
    /// `parallel_tool_calls` cap, because its converter keeps only the first
    /// tool block itself when the client set `disable_parallel_tool_use`.
    pub fn for_messages(mut self) -> Self {
        self.parallel_tool_calls = None;
        self
    }
}

/// Whether the request asks for a structured `response_format`: any but
/// `text`. Moved from dynamo-llm's
/// `OpenAIPreprocessor::has_structured_response_format`.
pub fn has_structured_response_format(request: &CreateChatCompletionRequest) -> bool {
    request
        .response_format
        .as_ref()
        .is_some_and(|format| !matches!(format, ResponseFormat::Text))
}

/// Aggregates the chunks of a chat completion stream into one response.
pub struct DeltaAggregator {
    /// Unique identifier for the chat completion.
    id: String,
    /// Model name used for the chat completion.
    model: String,
    /// Timestamp (Unix epoch) indicating when the response was created.
    created: u32,
    /// Optional usage statistics for the completion request.
    usage: Option<dynamo_protocols::types::CompletionUsage>,
    /// Optional system fingerprint for version tracking.
    system_fingerprint: Option<String>,
    /// Map of incremental response choices, keyed by index.
    choices: HashMap<u32, DeltaChoice>,
    /// Optional service tier information for the response.
    service_tier: Option<dynamo_protocols::types::ServiceTierResponse>,
}

/// Represents the accumulated state of a single chat choice during streaming aggregation.
#[derive(Debug)]
struct DeltaChoice {
    /// The index of the choice in the completion.
    index: u32,
    /// The accumulated text content for the choice.
    text: String,
    /// The role associated with this message (e.g., `system`, `user`, `assistant`).
    role: Option<dynamo_protocols::types::Role>,
    /// The reason the completion was finished (if applicable).
    finish_reason: Option<dynamo_protocols::types::FinishReason>,
    /// Optional log probabilities for the chat choice.
    logprobs: Option<dynamo_protocols::types::ChatChoiceLogprobs>,
    // Tool-call chunks accumulated in the order they arrived from the stream,
    // keyed by `index` so chunks that carry only argument fragments can be
    // merged into the entry created by the initial (id + name) chunk.
    // BTreeMap preserves deterministic iteration order on the index dimension.
    // See [`merge_tool_call_chunk`] for per-field merge semantics.
    // #8640: replaces the old `Option<Vec<ChatCompletionMessageToolCall>>`
    // which required id/name/arguments to all be set on the same chunk.
    tool_call_chunks: BTreeMap<u32, dynamo_protocols::types::ChatCompletionMessageToolCallChunk>,
    parser_call_completion: HashMap<u32, bool>,
    // Optional tool calls for the chat choice, populated *after* fold either
    // by finalizing `tool_call_chunks` above, or by
    // `try_tool_call_parse_aggregate_finalize` running against `text` for producers
    // that put tool calls in content rather than as structured chunks.
    tool_calls: Option<Vec<dynamo_protocols::types::ChatCompletionMessageToolCall>>,

    /// Optional reasoning content for the chat choice.
    reasoning_content: Option<String>,
    refusal: Option<String>,

    /// Accumulated content parts for multimodal responses
    content_parts: Vec<dynamo_protocols::types::ChatCompletionResponseContentPart>,
}

fn suppress_tool_call_output(choice: &mut DeltaChoice) {
    // Fail closed when the decoded turn contains only an unauthorized tool call.
    // We cannot safely reconstruct parser-specific wire markup as assistant text;
    // in that case content remains empty and the terminal reason becomes `stop`.
    choice.tool_calls = None;
    if choice.finish_reason == Some(dynamo_protocols::types::FinishReason::ToolCalls) {
        choice.finish_reason = Some(dynamo_protocols::types::FinishReason::Stop);
    }
}

/// A token-limit finish can interrupt any call that the upstream emitted as provisional.
/// Unified calls require complete JSON at every position. Legacy backend output
/// retains its existing final-call-only truncation check.
///
/// An empty argument string is the MOST truncated shape, not a parameterless call: the
/// limit landed after the name and before the first argument token. The wire schema does
/// permit omitted arguments, but nothing here can tell that apart from a call cut short,
/// so at the token limit it is dropped with every other unparseable tail.
fn drop_incomplete_length_tool_calls(choice: &mut DeltaChoice, unified: bool) {
    if choice.finish_reason != Some(dynamo_protocols::types::FinishReason::Length) {
        return;
    }

    let Some(tool_calls) = choice.tool_calls.as_mut() else {
        return;
    };
    let terminal_index = tool_calls.len().saturating_sub(1);
    let calls = std::mem::take(tool_calls);
    *tool_calls = calls
        .into_iter()
        .enumerate()
        .filter(|(index, tool_call)| {
            (!unified && *index != terminal_index)
                || serde_json::from_str::<serde_json::Value>(&tool_call.function.arguments).is_ok()
        })
        .map(|(_, tool_call)| tool_call)
        .collect();
    if tool_calls.is_empty() {
        choice.tool_calls = None;
    }
}

/// A stop or EOF can follow a streamed call whose last argument fragment is not
/// valid JSON. The adapter can normalize that stop to `ToolCalls` before aggregation,
/// so validate that terminal state too before exposing an executable final response.
/// An empty argument string remains valid when the upstream explicitly completed a
/// parameterless call with `ToolCalls`.
fn drop_malformed_terminal_tool_calls(
    choice: &mut DeltaChoice,
    require_explicit_tool_terminal: bool,
) {
    // Provisional streamed calls were rejected by wire index before finalization.
    // Shared batch parsing returns only completed calls, so no terminal is needed
    // to reconfirm them here. Legacy backend arguments retain their existing contract.
    if !require_explicit_tool_terminal {
        return;
    }

    if !matches!(
        choice.finish_reason,
        None | Some(dynamo_protocols::types::FinishReason::Stop)
            | Some(dynamo_protocols::types::FinishReason::ContentFilter)
            | Some(dynamo_protocols::types::FinishReason::ToolCalls)
    ) {
        return;
    }

    let explicit_parameterless_call =
        choice.finish_reason == Some(dynamo_protocols::types::FinishReason::ToolCalls);
    let Some(tool_calls) = choice.tool_calls.as_mut() else {
        return;
    };
    tool_calls.retain(|tool_call| {
        let arguments = &tool_call.function.arguments;
        (explicit_parameterless_call && arguments.is_empty())
            || serde_json::from_str::<serde_json::Value>(arguments).is_ok()
    });
    if tool_calls.is_empty() {
        choice.tool_calls = None;
        if choice.finish_reason == Some(dynamo_protocols::types::FinishReason::ToolCalls) {
            choice.finish_reason = Some(dynamo_protocols::types::FinishReason::Stop);
        }
    }
}

/// A failed structured parse at the output limit is not ordinary assistant prose.
/// Suppress the incomplete candidate rather than exposing parser markup or a partial
/// guided-JSON document as `content`. Plain text remains untouched unless the parser
/// actually identified a structured candidate.
fn suppress_incomplete_structured_content(
    choice: &mut DeltaChoice,
    parser: &str,
    constraint: &GuidedToolConstraint,
) {
    if choice.finish_reason != Some(dynamo_protocols::types::FinishReason::Length)
        || choice.text.is_empty()
    {
        return;
    }

    if let Some(marker_start) =
        crate::unified_stream::first_unquoted_structural_tool_call_marker(&choice.text, parser)
    {
        tracing::warn!(
            parser,
            suppressed_bytes = choice.text.len() - marker_start,
            "suppressing incomplete structured tool output on length finish"
        );
        choice.text.truncate(marker_start);
    } else if constraint.installs_guided_json() {
        tracing::warn!(
            parser,
            recovered_bytes = choice.text.len(),
            "suppressing incomplete guided structured output on length finish"
        );
        choice.text.clear();
    }
}

impl Default for DeltaAggregator {
    /// Provides a default implementation for `DeltaAggregator` by calling [`DeltaAggregator::new`].
    fn default() -> Self {
        Self::new()
    }
}

/// Merge an incoming chunk into the per-index accumulator.
///
/// #8640: the prior implementation required `id`, `name`, and `arguments`
/// all on the same chunk, and thus the argument-fragment deltas were dropped
/// and the client saw `arguments: ""`.
///
/// The fix here merges by `index` across deltas: `id`, `type`, `function.name`
/// first-wins; `function.arguments` concatenated across fragments. This matches
/// the OpenAI streaming spec and vLLM/SGLang hermes emission:
///
/// * delta 1: `{index, id, type, function: { name }}`
/// * delta 2..N: `{index, function: { arguments: "<fragment>" }}`
fn merge_tool_call_chunk(
    existing: &mut dynamo_protocols::types::ChatCompletionMessageToolCallChunk,
    incoming: dynamo_protocols::types::ChatCompletionMessageToolCallChunk,
) {
    if existing.id.is_none()
        && let Some(id) = incoming.id
    {
        existing.id = Some(id);
    }
    if existing.r#type.is_none()
        && let Some(ty) = incoming.r#type
    {
        existing.r#type = Some(ty);
    }
    let Some(incoming_fn) = incoming.function else {
        return;
    };
    match &mut existing.function {
        None => existing.function = Some(incoming_fn),
        Some(existing_fn) => {
            if existing_fn.name.is_none()
                && let Some(name) = incoming_fn.name
            {
                existing_fn.name = Some(name);
            }
            if let Some(args_fragment) = incoming_fn.arguments {
                existing_fn
                    .arguments
                    .get_or_insert_with(String::new)
                    .push_str(&args_fragment);
            }
        }
    }
}

/// Convert a fully merged chunk (post-merge accumulator state) to a finalized
/// `ChatCompletionMessageToolCall`. Returns `None` only if `id` or
/// `function.name` never arrived across any chunk — those are required by the
/// final OpenAI response schema. Missing `arguments` is legal (empty-args
/// tool calls) and becomes `""`. A warning is logged on drop so a producer
/// bug in upstream (e.g. vLLM / SGLang emitting fragments without ever
/// establishing the id+name opener) doesn't silently eat a tool call the
/// way the pre-fix code did.
fn finalize_merged_tool_chunk(
    chunk: dynamo_protocols::types::ChatCompletionMessageToolCallChunk,
) -> Option<dynamo_protocols::types::ChatCompletionMessageToolCall> {
    let index = chunk.index;
    let Some(id) = chunk.id else {
        tracing::warn!(
            tool_call_index = index,
            "dropping merged tool-call chunk: no `id` arrived across any delta"
        );
        return None;
    };
    let Some(function) = chunk.function else {
        tracing::warn!(
            tool_call_index = index,
            tool_call_id = %id,
            "dropping merged tool-call chunk: no `function` arrived across any delta"
        );
        return None;
    };
    let Some(name) = function.name else {
        tracing::warn!(
            tool_call_index = index,
            tool_call_id = %id,
            "dropping merged tool-call chunk: no `function.name` arrived across any delta"
        );
        return None;
    };
    Some(dynamo_protocols::types::ChatCompletionMessageToolCall {
        id,
        // Use the merged r#type if the stream carried one. Falls back to
        // `Function` — today the only variant in the OpenAI schema, but
        // threading the merged value keeps us forward-compat if variants
        // are added later and avoids dead state in `merge_tool_call_chunk`.
        r#type: chunk
            .r#type
            .unwrap_or(dynamo_protocols::types::FunctionType::Function),
        function: dynamo_protocols::types::FunctionCall {
            name,
            arguments: function.arguments.unwrap_or_default(),
        },
        custom: None,
    })
}

impl DeltaAggregator {
    /// Creates a new, empty [`DeltaAggregator`] instance.
    pub fn new() -> Self {
        Self {
            id: "".to_string(),
            model: "".to_string(),
            created: 0,
            usage: None,
            system_fingerprint: None,
            choices: HashMap::new(),
            service_tier: None,
        }
    }

    /// Fold one chunk, then the tool-call completion evidence that came with it.
    pub fn push(
        &mut self,
        delta: CreateChatCompletionStreamResponse,
        tool_call_completion: &[ToolCallCompletion],
    ) {
        self.id = delta.id;
        self.model = delta.model;
        self.created = delta.created;
        self.service_tier = delta.service_tier;

        // Aggregate usage statistics if available.
        if let Some(usage) = delta.usage {
            self.usage = Some(usage);
        }
        if let Some(system_fingerprint) = delta.system_fingerprint {
            self.system_fingerprint = Some(system_fingerprint);
        }

        // Aggregate choices incrementally.
        for choice in delta.choices {
            let choice_role = choice.delta.role;
            let state_choice = self.choices.entry(choice.index).or_insert(DeltaChoice {
                index: choice.index,
                text: "".to_string(),
                role: choice_role,
                finish_reason: None,
                logprobs: None,
                tool_call_chunks: BTreeMap::new(),
                parser_call_completion: HashMap::new(),
                tool_calls: None,
                reasoning_content: None,
                refusal: None,
                content_parts: Vec::new(),
            });

            if state_choice.role.is_none() {
                state_choice.role = choice_role;
            }

            // Handle content based on type
            if let Some(content) = &choice.delta.content {
                match content {
                    ChatCompletionMessageContent::Text(text) => {
                        state_choice.text.push_str(text);
                    }
                    ChatCompletionMessageContent::Parts(parts) => {
                        state_choice.content_parts.extend(parts.clone());
                    }
                }
            }

            if let Some(refusal) = &choice.delta.refusal {
                state_choice
                    .refusal
                    .get_or_insert_with(String::new)
                    .push_str(refusal);
            }

            if let Some(reasoning_content) = &choice.delta.reasoning_content {
                state_choice
                    .reasoning_content
                    .get_or_insert_with(String::new)
                    .push_str(reasoning_content);
            }

            // #8640: streaming producers split a single tool call across
            // multiple deltas (delta 1 = id + name; delta 2..N = argument
            // fragments), so we merge chunks into a per-index accumulator
            // here instead of treating each chunk as a complete tool call.
            // Finalization to `tool_calls` happens after the fold.
            if let Some(incoming_chunks) = choice.delta.tool_calls {
                for chunk in incoming_chunks {
                    let entry = state_choice
                        .tool_call_chunks
                        .entry(chunk.index)
                        .or_insert_with(|| {
                            dynamo_protocols::types::ChatCompletionMessageToolCallChunk {
                                index: chunk.index,
                                id: None,
                                r#type: None,
                                function: None,
                                custom: None,
                            }
                        });
                    merge_tool_call_chunk(entry, chunk);
                }
            }

            // Update finish reason if provided.
            if let Some(finish_reason) = choice.finish_reason {
                state_choice.finish_reason = Some(finish_reason);
            }

            // Update logprobs
            if let Some(logprobs) = &choice.logprobs {
                let state_lps = state_choice.logprobs.get_or_insert(
                    dynamo_protocols::types::ChatChoiceLogprobs {
                        content: None,
                        refusal: None,
                    },
                );
                if let Some(content_lps) = &logprobs.content {
                    state_lps
                        .content
                        .get_or_insert(Vec::new())
                        .extend(content_lps.clone());
                }
                if let Some(refusal_lps) = &logprobs.refusal {
                    state_lps
                        .refusal
                        .get_or_insert(Vec::new())
                        .extend(refusal_lps.clone());
                }
            }
        }
        // Completion can arrive without another argument or choice delta.
        // Apply it independently so the final call is not discarded.
        for evidence in tool_call_completion {
            if let Some(choice) = self.choices.get_mut(&evidence.choice_index) {
                choice
                    .parser_call_completion
                    .entry(evidence.tool_index)
                    .and_modify(|complete| *complete |= evidence.complete)
                    .or_insert(evidence.complete);
            }
        }
    }

    /// Build the response with the tool-call terminal policy dynamo's chat
    /// handler uses: [`Self::finish_with_tool_terminal_policy`] requires an
    /// explicit tool-call terminal when the parser pair selects a unified family
    /// and the deployment explicitly selected parser version 2.
    ///
    /// # Errors
    ///
    /// [`InvalidRequest`] when the parser version cannot serve the parser pair.
    pub async fn finish(
        self,
        parsing_options: &AggregatorOptions,
    ) -> Result<CreateChatCompletionResponse, InvalidRequest> {
        // V4.1 is default-on independently of this experiment. Its flag-off
        // backend-supplied arguments keep the established aggregation contract.
        let require_explicit_tool_terminal = selected_family(
            parsing_options.tool_call_parser.as_deref(),
            parsing_options.reasoning_parser.as_deref(),
            parsing_options.parser_version,
        )
        .is_some()
            && parsing_options.parser_version == ParserVersion::V2;
        self.finish_with_tool_terminal_policy(parsing_options, require_explicit_tool_terminal)
            .await
    }

    /// Build the response. `require_explicit_tool_terminal` drops a streamed call
    /// whose arguments are not valid JSON at a `stop`, `content_filter`,
    /// `tool_calls`, or missing finish, and checks every call at `length`.
    ///
    /// # Errors
    ///
    /// [`InvalidRequest`] when the parser version cannot serve the parser pair.
    pub async fn finish_with_tool_terminal_policy(
        self,
        parsing_options: &AggregatorOptions,
        require_explicit_tool_terminal: bool,
    ) -> Result<CreateChatCompletionResponse, InvalidRequest> {
        let mut aggregator = self;

        // #8640: finalize the per-index tool-call chunk accumulator into the
        // choice's `tool_calls` vector. Chunks missing id or name across the
        // whole stream are dropped here (they're not a valid tool call in the
        // final schema), but chunks missing only `arguments` get defaulted to
        // "" — the old code dropped those entirely.
        for choice in aggregator.choices.values_mut() {
            if choice.tool_call_chunks.is_empty() {
                continue;
            }
            let finalized: Vec<_> = std::mem::take(&mut choice.tool_call_chunks)
                .into_iter()
                // Explicit parser evidence owns provisional calls. Backend chunks have
                // no completion bit; the shared argument validation below handles them.
                .filter(|(index, _)| choice.parser_call_completion.get(index) != Some(&false))
                .filter_map(|(_, chunk)| finalize_merged_tool_chunk(chunk))
                .collect();
            // choice.tool_calls is always None at this point: or_insert
            // initializes it to None, try_tool_call_parse_aggregate_finalize
            // runs strictly after this loop. Unconditional assign is the only
            // reachable path; no merge-with-existing needed.
            if !finalized.is_empty() {
                choice.tool_calls = Some(finalized);
            }
        }

        // This is both a defense-in-depth check for structured deltas and a
        // prerequisite for whole-response decoders such as Harmony: clear any
        // already-structured calls before parsing so the decoder can still
        // inspect ordinary text below.
        if parsing_options.suppress_tool_calls {
            for choice in aggregator.choices.values_mut() {
                suppress_tool_call_output(choice);
            }
        }

        // One configured unified family owns raw batch output before legacy parsing.
        // Worker-parsed choices already carry tool calls and skip this safety net.
        let version = parsing_options.parser_version;
        let selected_unified_family = selected_family(
            parsing_options.tool_call_parser.as_deref(),
            parsing_options.reasoning_parser.as_deref(),
            version,
        );
        if selected_unified_family.is_none() {
            validate_tool_request_mode(
                version,
                parsing_options.tool_call_parser.as_deref(),
                parsing_options.reasoning_parser.as_deref(),
                !matches!(
                    parsing_options.guided_tool_constraint,
                    GuidedToolConstraint::None
                ),
            )
            .map_err(|error| InvalidRequest(error.to_string()))?;
        }
        if let Some(family) = selected_unified_family {
            for choice in aggregator.choices.values_mut() {
                if choice.text.is_empty()
                    || choice
                        .tool_calls
                        .as_ref()
                        .is_some_and(|calls| !calls.is_empty())
                {
                    continue;
                }
                match crate::unified_stream::parse_complete_with_policy(
                    family,
                    &choice.text,
                    &parsing_options.guided_tool_constraint,
                    &parsing_options.tools,
                    UnifiedRequestPolicy {
                        reasoning_disabled: parsing_options.reasoning_disabled,
                        structured_response: parsing_options.structured_response,
                        parsers_v2_selected: version == ParserVersion::V2,
                    },
                ) {
                    Ok(parsed) => {
                        if parsed.incomplete_native_call
                            && matches!(
                                choice.finish_reason,
                                Some(
                                    dynamo_protocols::types::FinishReason::Stop
                                        | dynamo_protocols::types::FinishReason::ToolCalls
                                )
                            )
                        {
                            choice.finish_reason =
                                Some(dynamo_protocols::types::FinishReason::Length);
                        }
                        choice.text = parsed.text;
                        if !parsed.reasoning.is_empty() && !parsing_options.reasoning_disabled {
                            choice
                                .reasoning_content
                                .get_or_insert_with(String::new)
                                .push_str(&parsed.reasoning);
                        }
                        if !parsed.tool_calls.is_empty() && !parsing_options.suppress_tool_calls {
                            choice.tool_calls = Some(parsed.tool_calls);
                            // OpenAI contract: a message carrying tool calls finishes
                            // with `ToolCalls`, not `Stop`.
                            if choice.finish_reason
                                == Some(dynamo_protocols::types::FinishReason::Stop)
                            {
                                choice.finish_reason =
                                    Some(dynamo_protocols::types::FinishReason::ToolCalls);
                            }
                        }
                    }
                    Err(error) => {
                        // Best-effort: the aggregated text is served as-is rather than
                        // failing a request the model already answered.
                        suppress_incomplete_structured_content(
                            choice,
                            family,
                            &parsing_options.guided_tool_constraint,
                        );
                        tracing::warn!(
                            error = %error,
                            family,
                            "failed to parse aggregated unified output; serving it unparsed"
                        );
                    }
                }
            }
        }

        if selected_unified_family.is_none() && parsing_options.tool_call_parser.is_some() {
            for choice in aggregator.choices.values_mut() {
                if choice
                    .tool_calls
                    .as_ref()
                    .is_some_and(|calls| !calls.is_empty())
                    || choice.text.is_empty()
                {
                    continue;
                }

                let Some(parser) = parsing_options.tool_call_parser.as_deref() else {
                    continue;
                };

                // V2 finish drops truncated values instead of guessing.
                // Guided JSON is handled above from the exact carried constraint.
                let parse_result = parse_complete_tool_output(
                    &choice.text,
                    parser,
                    &parsing_options.guided_tool_constraint,
                    &parsing_options.tools,
                    version,
                )
                .await;
                let (tool_calls, content) = match parse_result {
                    Ok(result) => result,
                    Err(error) => {
                        suppress_incomplete_structured_content(
                            choice,
                            parser,
                            &parsing_options.guided_tool_constraint,
                        );
                        tracing::debug!(
                            error = %error,
                            parser,
                            "failed to parse aggregated chat tool calls"
                        );
                        continue;
                    }
                };

                if !tool_calls.is_empty() {
                    choice.tool_calls = Some(
                        tool_calls
                            .into_iter()
                            .map(tool_call_response_to_protocol)
                            .collect(),
                    );
                    choice.text = content.unwrap_or_default();
                } else if (is_harmony_parser(parser) && contains_harmony_protocol(&choice.text))
                    || is_kimi_k3_parser(parser)
                {
                    choice.text = content.unwrap_or_default();
                } else if choice.finish_reason
                    == Some(dynamo_protocols::types::FinishReason::Length)
                {
                    suppress_incomplete_structured_content(
                        choice,
                        parser,
                        &parsing_options.guided_tool_constraint,
                    );
                }
            }
        }

        for choice in aggregator.choices.values_mut() {
            drop_incomplete_length_tool_calls(choice, require_explicit_tool_terminal);
            drop_malformed_terminal_tool_calls(choice, require_explicit_tool_terminal);
        }

        // A retained whole-response parser may discover a syntactically valid
        // call while removing model-internal channel markup. Parser activation
        // is not permission to expose that call; enforce the request policy
        // again after aggregate parsing.
        if parsing_options.suppress_tool_calls {
            for choice in aggregator.choices.values_mut() {
                suppress_tool_call_output(choice);
            }
        }

        // Enforce parallel_tool_calls == false as a universal post-parse fallback,
        // similar to vLLM's maybe_filter_parallel_tool_calls
        if parsing_options.parallel_tool_calls == Some(false) {
            for choice in aggregator.choices.values_mut() {
                if let Some(calls) = choice.tool_calls.as_mut()
                    && calls.len() > 1
                {
                    calls.truncate(1);
                }
            }
        }

        // for non-streaming `force_nonempty_content=true`
        // requests, a reasoning-only turn leaves `content` empty because all
        // output was split into `reasoning_content`. The chat template promised
        // non-empty content, so surface the reasoning as content. Only when no
        // content and no tool calls were produced; when content exists,
        // reasoning stays in `reasoning_content`.
        if parsing_options.move_reasoning_to_content_when_empty {
            for choice in aggregator.choices.values_mut() {
                let has_tool_calls = choice
                    .tool_calls
                    .as_ref()
                    .is_some_and(|calls| !calls.is_empty());
                // `content_parts` (multimodal) counts as content too — `From<DeltaChoice>`
                // prefers parts over `text`, so moving reasoning into `text` here would be
                // dropped. Only move when there is no content of either kind.
                // Whitespace-only text counts as empty — matching vLLM's
                // NemotronV3ReasoningParser (`not final_content.strip()`): a
                // trailing newline after `</think>` must not block the move and
                // leave semantically empty content.
                if choice.text.trim().is_empty()
                    && choice.content_parts.is_empty()
                    && !has_tool_calls
                    && choice
                        .reasoning_content
                        .as_deref()
                        .is_some_and(|r| !r.trim().is_empty())
                {
                    choice.text = choice.reasoning_content.take().unwrap_or_default();
                }
            }
        }

        // Extract aggregated choices and sort them by index.
        let mut choices: Vec<_> = aggregator
            .choices
            .into_values()
            .map(dynamo_protocols::types::ChatChoice::from)
            .collect();

        choices.sort_by_key(|a| a.index);

        // Construct the final response object.
        Ok(CreateChatCompletionResponse {
            id: aggregator.id,
            created: aggregator.created,
            usage: aggregator.usage,
            model: aggregator.model,
            object: "chat.completion".to_string(),
            system_fingerprint: aggregator.system_fingerprint,
            choices,
            service_tier: aggregator.service_tier,
        })
    }
}

#[allow(deprecated)]
impl From<DeltaChoice> for dynamo_protocols::types::ChatChoice {
    /// Converts a [`DeltaChoice`] into an [`dynamo_protocols::types::ChatChoice`].
    ///
    /// # Note
    /// The `function_call` field is deprecated.
    fn from(delta: DeltaChoice) -> Self {
        // A terminal reason that already explains why generation ended
        // outranks the tool-call rewrite. Only `Stop` and a missing reason are
        // rewritten, which is the same rule the streaming path applies in
        // `unified_parser::ChoiceState::normalize_finish_reason`; the two paths used to
        // disagree, so a call truncated at the token limit reached a non-streaming
        // caller labelled `tool_calls` with no sign it had been cut off.
        //
        // Preserving `Length` is also what makes the Responses conversion reachable:
        // `responses::chat_completion_to_response` keys `status: "incomplete"` and
        // `incomplete_details.reason: "max_output_tokens"` off this exact value.
        let has_tool_calls = delta
            .tool_calls
            .as_ref()
            .is_some_and(|calls| !calls.is_empty());
        let finish_reason = match delta.finish_reason {
            Some(dynamo_protocols::types::FinishReason::Stop) | None if has_tool_calls => {
                Some(dynamo_protocols::types::FinishReason::ToolCalls)
            }
            other => other,
        };

        // Determine content format based on what we accumulated
        let content = if !delta.content_parts.is_empty() {
            // Multimodal response with content parts
            Some(ChatCompletionMessageContent::Parts(delta.content_parts))
        } else if !delta.text.is_empty() {
            // Text-only response (backward compatible)
            Some(ChatCompletionMessageContent::Text(delta.text))
        } else {
            None
        };

        dynamo_protocols::types::ChatChoice {
            message: dynamo_protocols::types::ChatCompletionResponseMessage {
                role: delta
                    .role
                    .unwrap_or(dynamo_protocols::types::Role::Assistant),
                content,
                tool_calls: delta.tool_calls,
                refusal: delta.refusal,
                function_call: None,
                audio: None,
                reasoning_content: delta.reasoning_content,
            },
            index: delta.index,
            finish_reason,
            logprobs: delta.logprobs,
        }
    }
}

/// Map a parser's tool call onto the response type.
fn tool_call_response_to_protocol(parsed: ToolCallResponse) -> ChatCompletionMessageToolCall {
    ChatCompletionMessageToolCall {
        id: parsed.id,
        r#type: FunctionType::Function,
        function: FunctionCall {
            name: parsed.function.name,
            arguments: parsed.function.arguments,
        },
        custom: None,
    }
}

/// The v2 tool-only parser and the guided-JSON conversion, moved from dynamo
/// `chat_completions/tool_parser_v2.rs`.
mod tool_parser_v2 {
    use dynamo_parsers::tool_calling::{
        CalledFunction, ToolCallResponse, ToolCallType, ToolDefinition,
    };
    use dynamo_parsers_v2::create_tool_parser_for_family;
    use uuid::Uuid;

    use crate::guided::GuidedToolConstraint;
    use crate::unified_stream::to_v2_tools;

    fn v2_family(parser: &str) -> Option<&'static str> {
        match parser {
            "qwen3_coder" => Some("qwen3_coder"),
            "deepseek_v4" | "deepseek-v4" | "deepseekv4" => Some("deepseek_v4"),
            _ => None,
        }
    }

    /// Map engine-facing parser aliases onto the family keys used by
    /// `dynamo-parsers-v2`.
    pub(super) fn canonical_family(family: &str) -> &str {
        match family {
            "deepseek-v4" | "deepseekv4" => "deepseek_v4",
            family => family,
        }
    }

    /// Whether `family` has a v2 tool-only parser.
    pub(super) fn supports_family(family: &str) -> bool {
        v2_family(family).is_some()
    }

    /// Batch (non-streaming) path: run the whole response text through the `family` v2
    /// parser's complete lifecycle and map the coalesced calls back onto the v1
    /// `(tool_calls, normal_text)` tuple the aggregator consumes. No jail involved; a
    /// call truncated mid parameter value is dropped (returns zero calls, empty text).
    pub(super) fn parse_complete(
        content: &str,
        tools: Option<&[ToolDefinition]>,
        family: &str,
    ) -> anyhow::Result<(Vec<ToolCallResponse>, String)> {
        let v2_tools = to_v2_tools(tools);
        let mut parser = create_tool_parser_for_family(canonical_family(family), &v2_tools)?;
        let result = parser.parse_complete(content)?;

        let tool_calls = result
            .calls
            .into_iter()
            .map(|call| ToolCallResponse {
                id: format!("call-{}", Uuid::new_v4()),
                tp: ToolCallType::Function,
                function: CalledFunction {
                    name: call.name.unwrap_or_default(),
                    arguments: call.arguments,
                },
            })
            .collect();

        Ok((tool_calls, result.normal_text))
    }

    /// Convert the two guided-JSON response shapes into parser-native tool calls.
    ///
    /// Named guidance emits only the selected tool's argument object. Required guidance
    /// emits one call envelope or an array of envelopes. This conversion is family-neutral:
    /// This legacy conversion is used outside the shared unified route, whose parser
    /// consumes the same constraint directly and retains ordered reasoning spans.
    pub(super) fn parse_complete_guided_json(
        content: &str,
        constraint: &GuidedToolConstraint,
    ) -> anyhow::Result<Vec<ToolCallResponse>> {
        fn response(
            name: String,
            arguments: &serde_json::Value,
        ) -> anyhow::Result<ToolCallResponse> {
            Ok(ToolCallResponse {
                id: format!("call-{}", Uuid::new_v4()),
                tp: ToolCallType::Function,
                function: CalledFunction {
                    name,
                    arguments: serde_json::to_string(arguments)?,
                },
            })
        }

        let payload: serde_json::Value = serde_json::from_str(content.trim())?;
        match constraint {
            GuidedToolConstraint::GuidedJsonNamed { tool_name } => {
                anyhow::ensure!(
                    payload.is_object(),
                    "named guided payload must be an object"
                );
                Ok(vec![response(tool_name.clone(), &payload)?])
            }
            GuidedToolConstraint::GuidedJsonRequired => {
                let envelopes = match &payload {
                    serde_json::Value::Array(envelopes) => envelopes.as_slice(),
                    serde_json::Value::Object(_) => std::slice::from_ref(&payload),
                    _ => anyhow::bail!("required guided payload must be an object or array"),
                };
                envelopes
                    .iter()
                    .map(|envelope| {
                        let object = envelope.as_object().ok_or_else(|| {
                            anyhow::anyhow!("guided call envelope must be an object")
                        })?;
                        let name = object
                            .get("name")
                            .and_then(serde_json::Value::as_str)
                            .ok_or_else(|| {
                                anyhow::anyhow!("guided call envelope has no string name")
                            })?;
                        let arguments = object
                            .get("parameters")
                            .or_else(|| object.get("arguments"))
                            .ok_or_else(|| {
                                anyhow::anyhow!("guided call envelope has no arguments")
                            })?;
                        response(name.to_string(), arguments)
                    })
                    .collect()
            }
            GuidedToolConstraint::None | GuidedToolConstraint::StructuralTag => {
                anyhow::bail!("request did not install guided JSON")
            }
        }
    }
}

#[cfg(test)]
#[path = "aggregator_tests.rs"]
pub(crate) mod tests;
