// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `AggregatorOptions::for_request` and `for_messages` against a copy of the
//! Dynamo frontend's handler code at dynamo `37f2f0a712`, over a matrix of
//! parser pairs and requests.

use dynamo_chat_parsing::aggregator::AggregatorOptions;
use dynamo_chat_parsing::families::selected_family;
use dynamo_chat_parsing::guided::GuidedToolConstraint;
use dynamo_chat_parsing::nonempty_content::wants_reasoning_as_content_when_empty;
use dynamo_chat_parsing::reasoning_off::{
    ChatStreamRequest, request_disables_reasoning_with_family,
};
use dynamo_chat_parsing::tools::{convert_tools, effective_tools, tool_call_parsing_enabled};
use dynamo_chat_parsing::version::ParserVersion;
use dynamo_parsers::tool_calling::ToolDefinition;
use dynamo_protocols::types::CreateChatCompletionRequest;
use serde_json::{Map, Value, json};

/// The Dynamo frontend's code, copied as the oracle. Each change from the
/// stock code is marked `ORACLE CHANGE`.
mod oracle {
    use super::*;

    /// `OpenAIPreprocessor::has_structured_response_format`
    /// (`lib/llm/src/preprocessor.rs:2315`), over the request's
    /// `response_format` as `OAIChatLikeRequest::response_format` serializes it.
    /// ORACLE CHANGE: a `serde_json` value for minijinja's.
    fn has_structured_response_format(request: &CreateChatCompletionRequest) -> bool {
        request
            .response_format
            .as_ref()
            .map(|format| serde_json::to_value(format).expect("serialize"))
            .is_some_and(|format| {
                format
                    .get("type")
                    .and_then(Value::as_str)
                    .is_some_and(|kind| kind != "text")
            })
    }

    /// `apply_request_tool_call_parsing_options` (`lib/llm/src/http/service.rs:59`),
    /// then the handler's `with_parallel_tool_calls` and
    /// `with_move_reasoning_to_content_when_empty`
    /// (`lib/llm/src/http/service/openai.rs:3611-3623`, `:4325-4352`;
    /// `anthropic.rs:633-647`), and for Messages the aggregation options
    /// (`anthropic.rs:924-930`).
    ///
    /// ORACLE CHANGE: `AggregatorOptions` for `ParsingOptions` (its port);
    /// the guided constraint and the converted tools come in, as the
    /// structural-tag decision and the tool validation are the caller's;
    /// `selected_family` with the version for `selected_batch_family`, which
    /// reads `DYN_PARSER_VERSION`.
    #[allow(clippy::too_many_arguments)]
    pub fn options(
        request: &ChatStreamRequest<'_>,
        tool_call_parser: Option<&str>,
        reasoning_parser: Option<&str>,
        default_thinking_mode: Option<&str>,
        guided_tool_constraint: GuidedToolConstraint,
        converted_tools: Vec<ToolDefinition>,
        parser_version: ParserVersion,
        messages_disable_parallel_tool_use: Option<bool>,
    ) -> AggregatorOptions {
        let mut parsing_options = AggregatorOptions {
            tool_call_parser: tool_call_parser.map(str::to_owned),
            reasoning_parser: reasoning_parser.map(str::to_owned),
            parser_version,
            ..AggregatorOptions::default()
        };
        let unified_family = selected_family(
            parsing_options.tool_call_parser.as_deref(),
            parsing_options.reasoning_parser.as_deref(),
            parser_version,
        );
        parsing_options.reasoning_disabled = request_disables_reasoning_with_family(
            request,
            parsing_options.tool_call_parser.as_deref(),
            parsing_options.reasoning_parser.as_deref(),
            default_thinking_mode,
            unified_family,
        );
        parsing_options.structured_response = has_structured_response_format(request.request);
        let tool_call_parsing_enabled = tool_call_parsing_enabled(request.request);
        parsing_options.guided_tool_constraint = guided_tool_constraint;
        let mut parsing_options =
            parsing_options.with_tool_call_parsing_enabled(tool_call_parsing_enabled);
        parsing_options.tools = converted_tools;

        parsing_options.parallel_tool_calls = request.request.parallel_tool_calls;
        let move_reasoning_to_content_when_empty =
            wants_reasoning_as_content_when_empty(request.chat_template_args);
        parsing_options.move_reasoning_to_content_when_empty = move_reasoning_to_content_when_empty;

        if messages_disable_parallel_tool_use == Some(true) {
            parsing_options.parallel_tool_calls = None;
        }
        parsing_options
    }
}

const PARSERS: &[(Option<&str>, Option<&str>)] = &[
    (None, None),
    (Some("hermes"), Some("qwen3")),
    (Some("qwen3_coder"), Some("qwen3")),
    (None, Some("qwen3")),
    (Some("hermes"), None),
    (Some("harmony"), Some("gpt_oss")),
    (Some("kimi_k3"), Some("kimi_k3")),
    (Some("kimi_k2"), Some("kimi_k25")),
    (Some("glm47"), Some("glm45")),
    (Some("deepseek_v4"), Some("deepseek_v4")),
    (Some("gemma4"), Some("gemma4")),
    (Some("minimax_m3"), Some("minimax_m3")),
    (Some("mistral"), Some("mistral")),
];

fn tool() -> Value {
    json!({"type": "function", "function": {
        "name": "get_weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
    }})
}

/// The request's variants, each a set of fields over a base request.
fn request_fields() -> Vec<Map<String, Value>> {
    let tools = [None, Some(json!([tool()]))];
    let tool_choices = [
        None,
        Some(json!("auto")),
        Some(json!("none")),
        Some(json!("required")),
        Some(json!({"type": "function", "function": {"name": "get_weather"}})),
    ];
    let parallel = [None, Some(json!(true)), Some(json!(false))];
    let formats = [
        None,
        Some(json!({"type": "text"})),
        Some(json!({"type": "json_object"})),
    ];
    let mut out = Vec::new();
    for tools in &tools {
        for choice in &tool_choices {
            for parallel in &parallel {
                for format in &formats {
                    let mut fields = Map::new();
                    let mut set = |key: &str, value: &Option<Value>| {
                        if let Some(value) = value {
                            fields.insert(key.to_owned(), value.clone());
                        }
                    };
                    set("tools", tools);
                    set("tool_choice", choice);
                    set("parallel_tool_calls", parallel);
                    set("response_format", format);
                    out.push(fields);
                }
            }
        }
    }
    out
}

fn template_args() -> Vec<Option<Map<String, Value>>> {
    [
        None,
        Some(json!({"enable_thinking": false})),
        Some(json!({"enable_thinking": true})),
        Some(json!({"thinking": false})),
        Some(json!({"force_nonempty_content": true})),
        Some(json!({"thinking_mode": "disabled"})),
    ]
    .into_iter()
    .map(|args| args.map(|args| args.as_object().expect("object").clone()))
    .collect()
}

/// Every option the builder sets matches the stock handler's, for every
/// parser pair, request, template arguments, thinking field, default
/// thinking mode, and parser version, on Chat Completions, Responses, and
/// Messages.
#[test]
fn the_options_are_the_stock_handlers() {
    let mut cases = 0;
    let mut differences = Vec::new();
    for fields in request_fields() {
        let mut body = json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
        });
        body.as_object_mut().expect("object").extend(fields);
        let request: CreateChatCompletionRequest =
            serde_json::from_value(body.clone()).expect("a chat request");
        let tools = convert_tools(
            effective_tools(&request, |_| Ok(()))
                .expect("tools")
                .as_ref(),
        );
        for args in template_args() {
            let args = args.map(|args| args.into_iter().collect());
            for client_thinking_field in [false, true] {
                let view = ChatStreamRequest {
                    request: &request,
                    chat_template_args: args.as_ref(),
                    client_thinking_field,
                    explicit_guided_decoding: false,
                };
                for &(tool_parser, reasoning_parser) in PARSERS {
                    for default_thinking_mode in [None, Some("enabled"), Some("disabled")] {
                        for version in [ParserVersion::Auto, ParserVersion::V2] {
                            for messages in [None, Some(false), Some(true)] {
                                let mut ours = AggregatorOptions::for_request(
                                    &view,
                                    tool_parser,
                                    reasoning_parser,
                                    default_thinking_mode,
                                    GuidedToolConstraint::None,
                                    tools.clone(),
                                    version,
                                );
                                if messages == Some(true) {
                                    ours = ours.for_messages();
                                }
                                let stock = oracle::options(
                                    &view,
                                    tool_parser,
                                    reasoning_parser,
                                    default_thinking_mode,
                                    GuidedToolConstraint::None,
                                    tools.clone(),
                                    version,
                                    messages,
                                );
                                let (ours, stock) = (format!("{ours:?}"), format!("{stock:?}"));
                                if ours != stock {
                                    differences.push(format!(
                                        "{body} args {args:?} thinking field \
                                         {client_thinking_field} {tool_parser:?}/\
                                         {reasoning_parser:?} default {default_thinking_mode:?} \
                                         {version:?} messages {messages:?}:\n ours  {ours}\n stock {stock}"
                                    ));
                                }
                                cases += 1;
                            }
                        }
                    }
                }
            }
        }
    }
    assert!(
        differences.is_empty(),
        "{} of {cases}:\n{}",
        differences.len(),
        differences[..differences.len().min(5)].join("\n")
    );
    assert!(cases > 100_000, "{cases} cases");
}

/// `for_messages` drops the cap only; a Messages request whose client did
/// not set `disable_parallel_tool_use` has no cap to drop, so the stock
/// options and `for_messages` agree there too.
#[test]
fn for_messages_drops_only_the_parallel_cap() {
    let request: CreateChatCompletionRequest = serde_json::from_value(json!({
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [tool()],
        "parallel_tool_calls": false,
    }))
    .expect("a chat request");
    let view = ChatStreamRequest {
        request: &request,
        chat_template_args: None,
        client_thinking_field: false,
        explicit_guided_decoding: false,
    };
    let options = AggregatorOptions::for_request(
        &view,
        Some("hermes"),
        Some("qwen3"),
        None,
        GuidedToolConstraint::None,
        Vec::new(),
        ParserVersion::Auto,
    );
    assert_eq!(options.parallel_tool_calls, Some(false));
    let messages = options.clone().for_messages();
    assert_eq!(messages.parallel_tool_calls, None);
    assert_eq!(
        format!(
            "{:?}",
            AggregatorOptions {
                parallel_tool_calls: None,
                ..options
            }
        ),
        format!("{messages:?}")
    );
}
