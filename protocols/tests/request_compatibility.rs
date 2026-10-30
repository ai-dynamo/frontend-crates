// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Wire-level regressions for ai-dynamo/frontend-crates#299.
//! Backend capability filtering belongs in the serving adapter, not serde.

use dynamo_protocols::types::responses::{
    CreateResponse, TextResponseFormatConfiguration, Verbosity,
};
use dynamo_protocols::types::{CreateChatCompletionRequest, CreateCompletionRequest};
use serde_json::json;

#[test]
fn chat_custom_tools_preserve_openai_declarations_and_named_choice() {
    for format in [None, Some(json!({"type":"text"}))] {
        let mut custom = json!({"name":"run_python","description":"Execute Python"});
        if let Some(format) = format {
            custom["format"] = format;
        }
        let tool = json!({"type":"custom","custom":custom});
        let choice = json!({"type":"custom","custom":{"name":"run_python"}});
        let request: CreateChatCompletionRequest = serde_json::from_value(json!({
            "model":"test", "messages":[{"role":"user","content":"Calculate"}],
            "tools":[tool.clone()], "tool_choice":choice.clone()
        }))
        .expect("Chat Completions accepts OpenAI custom tools");
        let output = serde_json::to_value(&request).unwrap();
        assert_eq!(output["tools"][0], tool);
        assert_eq!(output["tool_choice"], choice);
    }
}

#[test]
fn chat_custom_tool_calls_preserve_freeform_history() {
    use dynamo_protocols::types::ChatCompletionMessageToolCall;
    let input = "*** Begin Patch\nUnicode: café 日本語\n\"quoted\" \\ backslash\n*** End Patch";
    let call = json!({"id":"call_1","type":"custom","custom":{"name":"apply_patch","input":input}});
    let parsed: ChatCompletionMessageToolCall = serde_json::from_value(call.clone()).unwrap();
    assert_eq!(serde_json::to_value(parsed).unwrap(), call);
    let request: CreateChatCompletionRequest = serde_json::from_value(json!({
        "model":"test", "messages":[
            {"role":"assistant","tool_calls":[call.clone()]},
            {"role":"tool","tool_call_id":"call_1","content":"Patch applied"}
        ]
    }))
    .unwrap();
    assert_eq!(
        serde_json::to_value(request).unwrap()["messages"][0]["tool_calls"][0],
        call
    );
}

#[test]
fn chat_custom_stream_chunks_preserve_input_and_omit_function() {
    use dynamo_protocols::types::ChatCompletionMessageToolCallChunk;
    for chunk in [
        json!({"index":0,"id":"call_1","type":"custom","custom":{"name":"run_python","input":"print("}}),
        json!({"index":0,"custom":{"input":"2 + 3)"}}),
    ] {
        let parsed: ChatCompletionMessageToolCallChunk =
            serde_json::from_value(chunk.clone()).unwrap();
        assert_eq!(serde_json::to_value(parsed).unwrap(), chunk);
    }
}

#[test]
fn chat_custom_shapes_reject_missing_or_non_string_payloads() {
    use dynamo_protocols::types::{ChatCompletionMessageToolCall, ChatCompletionTool};
    for tool in [
        json!({"type":"custom"}),
        json!({"type":"custom","custom":{}}),
    ] {
        assert!(serde_json::from_value::<ChatCompletionTool>(tool).is_err());
    }
    for input in [json!(null), json!(42), json!({"code":"print(1)"})] {
        assert!(
            serde_json::from_value::<ChatCompletionMessageToolCall>(json!({
                "type":"custom","id":"call_1","custom":{"name":"run_python","input":input}
            }))
            .is_err()
        );
    }
}

#[test]
fn chat_custom_tool_names_cannot_also_name_a_function() {
    let mut request: CreateChatCompletionRequest = serde_json::from_value(json!({
        "model":"test", "messages":[{"role":"user","content":"Look up"}],
        "tools":[{"type":"custom","custom":{"name":"lookup"}},
                 {"type":"function","function":{"name":"lookup","parameters":{"type":"object"}}}]
    }))
    .unwrap();
    // Response projection restores custom identity by name, so the kinds cannot share one.
    let error = request.normalize_custom_tools().unwrap_err();
    assert!(
        error.to_string().contains("both custom and function"),
        "{error}"
    );
}

/// A tool built in Rust may set only the wrapper's flag. Serialization
/// writes it into the function too, and writes a function's own flag once.
#[test]
fn chat_tool_deferral_alias_applies_when_built_in_rust() {
    use dynamo_protocols::types::{ChatCompletionTool, FunctionObject};
    for (function_flag, tool_flag, expected) in [
        (None, None, None),
        (Some(true), None, Some(true)),
        (None, Some(true), Some(true)),
        (None, Some(false), Some(false)),
        (Some(false), Some(true), Some(false)),
    ] {
        let tool = ChatCompletionTool {
            function: FunctionObject {
                name: "find_ticket".into(),
                defer_loading: function_flag,
                ..Default::default()
            },
            defer_loading: tool_flag,
            ..Default::default()
        };
        let output = serde_json::to_value(&tool).unwrap();
        let mut function = json!({"name": "find_ticket"});
        if let Some(flag) = expected {
            function["defer_loading"] = json!(flag);
        }
        let mut wire = json!({"type": "function", "function": function});
        if let Some(flag) = tool_flag {
            wire["defer_loading"] = json!(flag);
        }
        assert_eq!(
            output, wire,
            "function={function_flag:?}, wrapper={tool_flag:?}"
        );
        assert_eq!(
            serde_json::to_string(&tool)
                .unwrap()
                .matches("defer_loading")
                .count(),
            usize::from(expected.is_some()) + usize::from(tool_flag.is_some()),
        );
    }
}

#[test]
fn chat_tool_deferral_survives_request_round_trip() {
    for (function_flag, tool_flag, expected) in [
        (Some(true), None, true),
        (Some(false), None, false),
        (None, Some(true), true),
        (None, Some(false), false),
        (Some(false), Some(true), false),
        (Some(true), Some(false), true),
    ] {
        let mut tool = json!({
            "type": "function",
            "function": {"name": "find_ticket", "parameters": {"type": "object"}}
        });
        if let Some(flag) = function_flag {
            tool["function"]["defer_loading"] = json!(flag);
        }
        if let Some(flag) = tool_flag {
            tool["defer_loading"] = json!(flag);
        }
        let request: CreateChatCompletionRequest = serde_json::from_value(json!({
            "model": "example-model",
            "messages": [{"role": "user", "content": "Find a ticket"}],
            "tools": [tool]
        }))
        .unwrap();
        let output = serde_json::to_value(request.clone()).unwrap();
        assert_eq!(
            output["tools"][0]["function"]["defer_loading"],
            json!(expected),
            "function={function_flag:?}, wrapper={tool_flag:?}: deferral must reach rendering"
        );
        assert_eq!(
            output["tools"][0].get("defer_loading"),
            tool_flag.map(|flag| json!(flag)).as_ref()
        );
    }
}

#[test]
fn chat_tool_deferral_rejects_non_boolean_values() {
    for invalid in [json!("true"), json!(1), json!([]), json!({})] {
        for nested in [true, false] {
            let mut tool = json!({"type": "function", "function": {"name": "lookup"}});
            if nested {
                tool["function"]["defer_loading"] = invalid.clone();
            } else {
                tool["defer_loading"] = invalid.clone();
            }
            assert!(
                serde_json::from_value::<CreateChatCompletionRequest>(json!({
                    "model": "example-model", "messages": [], "tools": [tool]
                }))
                .is_err(),
                "invalid defer_loading must be rejected: {invalid}, nested={nested}"
            );
        }
    }
}

#[test]
fn chat_tool_deferral_omitted_and_null_keep_original_wire_shape() {
    let ordinary = json!({
        "type": "function",
        "function": {"name": "lookup", "description": "Look up a ticket", "strict": false}
    });
    for null_function in [false, true] {
        for null_tool in [false, true] {
            let mut input = ordinary.clone();
            if null_function {
                input["function"]["defer_loading"] = json!(null);
            }
            if null_tool {
                input["defer_loading"] = json!(null);
            }
            let tool: dynamo_protocols::types::ChatCompletionTool =
                serde_json::from_value(input).unwrap();
            assert_eq!(serde_json::to_value(tool).unwrap(), ordinary);
        }
    }
}

#[test]
fn chat_tool_deferral_builder_matches_wire_normalization() {
    use dynamo_protocols::types::{ChatCompletionToolArgs, FunctionObjectArgs};

    for function_flag in [None, Some(false), Some(true)] {
        let mut function = FunctionObjectArgs::default();
        function.name("lookup");
        if let Some(flag) = function_flag {
            function.defer_loading(flag);
        }
        let tool = ChatCompletionToolArgs::default()
            .function(function.build().unwrap())
            .defer_loading(true)
            .build()
            .unwrap();
        assert_eq!(
            tool.function.defer_loading,
            Some(function_flag.unwrap_or(true))
        );
    }
}

#[test]
fn chat_tool_deferral_serialization_handles_direct_construction_and_mutation() {
    use dynamo_protocols::types::{ChatCompletionTool, FunctionObject};

    for (function_flag, wrapper_flag, expected) in [
        (None, Some(true), Some(true)),
        (None, Some(false), Some(false)),
        (Some(false), Some(true), Some(false)),
        (Some(true), Some(false), Some(true)),
        (Some(false), None, Some(false)),
        (None, None, None),
    ] {
        let mut tool = ChatCompletionTool {
            function: FunctionObject {
                name: "lookup".into(),
                defer_loading: function_flag,
                ..Default::default()
            },
            defer_loading: wrapper_flag,
            ..Default::default()
        };
        let before = tool.clone();
        let wire = serde_json::to_value(&tool).unwrap();
        let text = serde_json::to_string(&tool).unwrap();
        assert_eq!(
            text.matches("\"defer_loading\":").count(),
            usize::from(expected.is_some()) + usize::from(wrapper_flag.is_some()),
            "each object must contain at most one defer_loading key"
        );
        assert_eq!(
            wire["function"].get("defer_loading"),
            expected.map(|value| json!(value)).as_ref(),
            "function={function_flag:?}, wrapper={wrapper_flag:?}"
        );
        assert_eq!(tool, before, "serialization must not mutate the tool");
        let decoded: ChatCompletionTool = serde_json::from_value(wire).unwrap();
        assert_eq!(decoded.function.defer_loading, expected);

        tool.function.defer_loading = None;
        tool.defer_loading = Some(false);
        assert_eq!(
            serde_json::to_value(&tool).unwrap()["function"]["defer_loading"],
            json!(false),
            "later mutation must use the same wrapper fallback"
        );
    }
}

#[test]
fn responses_verbosity_without_format_defaults_to_text() {
    for (wire_verbosity, verbosity) in [
        ("low", Verbosity::Low),
        ("medium", Verbosity::Medium),
        ("high", Verbosity::High),
    ] {
        let request: CreateResponse = serde_json::from_value(json!({
            "model": "example-model",
            "input": "Say hello",
            "text": {"verbosity": wire_verbosity}
        }))
        .unwrap();
        let text = request.text.as_ref().unwrap();
        assert_eq!(text.format, TextResponseFormatConfiguration::Text);
        assert_eq!(text.verbosity, Some(verbosity));
        assert_eq!(
            serde_json::to_value(&request).unwrap()["text"],
            json!({"format": {"type": "text"}, "verbosity": wire_verbosity})
        );
    }
}

#[test]
fn responses_verbosity_preserves_explicit_format() {
    for format in [
        json!({"type": "text"}),
        json!({"type": "json_object"}),
        json!({
            "type": "json_schema",
            "name": "answer",
            "strict": true,
            "schema": {
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
                "additionalProperties": false
            }
        }),
    ] {
        let request: CreateResponse = serde_json::from_value(json!({
            "model": "example-model",
            "input": "Return an answer",
            "text": {"verbosity": "high", "format": format}
        }))
        .unwrap();
        assert_eq!(
            request.text.as_ref().unwrap().verbosity,
            Some(Verbosity::High)
        );
        assert_eq!(
            serde_json::to_value(&request).unwrap()["text"],
            json!({
                "verbosity": "high", "format": format
            })
        );
    }
}

#[test]
fn completion_requests_accept_stream_options_independently_of_stream() {
    for stream in [
        None,
        Some(json!(null)),
        Some(json!(false)),
        Some(json!(true)),
    ] {
        for include_usage in [false, true] {
            for continuous_usage_stats in [false, true] {
                let options = json!({
                    "include_usage": include_usage,
                    "continuous_usage_stats": continuous_usage_stats
                });
                let mut wire = json!({
                    "model": "example-model",
                    "stream_options": options
                });
                if let Some(stream) = &stream {
                    wire["stream"] = stream.clone();
                }

                let mut chat_wire = wire.clone();
                chat_wire["messages"] = json!([{"role": "user", "content": "Say hello"}]);
                let chat: CreateChatCompletionRequest = serde_json::from_value(chat_wire).unwrap();

                wire["prompt"] = json!("Say hello");
                let completion: CreateCompletionRequest = serde_json::from_value(wire).unwrap();

                let expected_stream = stream.as_ref().and_then(serde_json::Value::as_bool);
                assert_eq!(chat.stream, expected_stream);
                assert_eq!(completion.stream, expected_stream);
                for actual in [chat.stream_options, completion.stream_options] {
                    let actual = actual.unwrap();
                    assert_eq!(actual.include_usage, include_usage);
                    assert_eq!(actual.continuous_usage_stats, continuous_usage_stats);
                }
                for serialized in [
                    serde_json::to_value(chat).unwrap(),
                    serde_json::to_value(completion).unwrap(),
                ] {
                    assert_eq!(serialized["stream_options"], options);
                    assert_eq!(
                        serialized
                            .get("stream")
                            .and_then(serde_json::Value::as_bool),
                        expected_stream
                    );
                }
            }
        }
    }
}
