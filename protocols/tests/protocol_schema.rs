// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![cfg(feature = "protocol-schema")]

use dynamo_protocols::types::*;
use serde::{Serialize, de::DeserializeOwned};
use serde_json::{Value, json};
use utoipa::{OpenApi, ToSchema};

#[derive(OpenApi)]
#[openapi(components(schemas(
    CreateChatCompletionRequest,
    CreateCompletionRequest,
    CreateCompletionResponse,
    CreateChatCompletionResponse,
    CreateChatCompletionStreamResponse
)))]
struct Contract;

/// Return the registered chat/completion schemas as one JSON OpenAPI document.
fn document() -> Value {
    serde_json::to_value(Contract::openapi()).unwrap()
}

/// Build a validator for `T`, which must be included in `document()`.
/// Panics if the generated schema cannot be compiled.
fn validator<T: ToSchema>() -> jsonschema::Validator {
    let mut doc = document();
    doc["$ref"] = json!(format!("#/components/schemas/{}", T::name()));
    jsonschema::validator_for(&doc).unwrap()
}

/// Parse `input` as `T`, then assert that its serialized form matches `T`'s schema.
/// Use for inputs that normalize during parsing; panics on conversion or validation failure.
fn canonical<T: ToSchema + DeserializeOwned + Serialize>(input: Value) {
    let parsed: T = serde_json::from_value(input).unwrap();
    let output = serde_json::to_value(parsed).unwrap();
    let schema = validator::<T>();
    let errors: Vec<_> = schema.iter_errors(&output).map(|e| e.to_string()).collect();
    assert!(errors.is_empty(), "{output}: {errors:?}");
}

#[test]
fn request_roots_are_distinct_from_upstream_and_have_real_fields() {
    let doc = document();
    let schemas = &doc["components"]["schemas"];
    assert_eq!(
        CreateChatCompletionRequest::name(),
        "dynamo_protocols.chat.CreateChatCompletionRequest"
    );
    let own = &schemas[CreateChatCompletionRequest::name().as_ref()];
    assert!(own["properties"].get("messages").is_some());
    assert!(own["properties"].get("mm_processor_kwargs").is_some());
    assert_eq!(own["required"], json!(["messages", "model"]));
    let completion = &schemas[CreateCompletionRequest::name().as_ref()];
    assert!(completion["properties"].get("prompt_embeds").is_some());
    assert!(completion["properties"].get("prompt").is_some());
}

#[test]
fn component_graph_resolves_every_reference() {
    fn walk(value: &Value, root: &Value) {
        match value {
            Value::Object(fields) => {
                if let Some(reference) = fields.get("$ref").and_then(Value::as_str) {
                    assert!(reference.starts_with("#/components/schemas/"));
                    assert!(root.pointer(&reference[1..]).is_some(), "{reference}");
                }
                for child in fields.values() {
                    walk(child, root);
                }
            }
            Value::Array(values) => {
                for child in values {
                    walk(child, root);
                }
            }
            _ => {}
        }
    }
    let doc = document();
    walk(&doc, &doc);
    for name in doc["components"]["schemas"].as_object().unwrap().keys() {
        assert!(
            name.starts_with("dynamo_protocols.") || name.starts_with("async_openai."),
            "{name}"
        );
    }
}

/// Upstream schemas must describe field shapes and include their nested definitions.
/// Check valid and invalid prompts, then validate a serialized completion response.
#[test]
fn dependency_types_describe_native_shapes_and_transitive_components() {
    let doc = document();
    let schemas = doc["components"]["schemas"].as_object().unwrap();
    for name in [
        "async_openai.Prompt",
        "async_openai.FunctionObject",
        "async_openai.ResponseFormatJsonSchema",
        "async_openai.CreateCompletionResponse",
        "async_openai.Choice",
        "async_openai.CompletionUsage",
        "async_openai.PromptTokensDetails",
        "async_openai.CompletionTokensDetails",
    ] {
        assert!(
            schemas.contains_key(name),
            "missing transitive component {name}"
        );
    }
    assert_eq!(
        schemas["async_openai.FunctionObject"]["properties"]["name"]["type"],
        "string"
    );
    assert_eq!(
        schemas["async_openai.CompletionUsage"]["properties"]["total_tokens"]["type"],
        "integer"
    );
    let completion = validator::<CreateCompletionRequest>();
    for prompt in [
        json!("hi"),
        json!(["hi"]),
        json!([1, 2]),
        json!([[1], [2]]),
        json!([]),
    ] {
        assert!(completion.is_valid(&json!({"model":"test", "prompt":prompt})));
    }
    for prompt in [json!(true), json!({"text":"hi"}), json!(["hi", 2])] {
        assert!(!completion.is_valid(&json!({"model":"test", "prompt":prompt})));
    }
    canonical::<CreateCompletionResponse>(json!({
        "id":"cmpl-1", "object":"text_completion", "created":1, "model":"test",
        "choices":[{"index":0,"text":"hi","finish_reason":"stop"}],
        "usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}
    }));
}

/// Schemas must retain field defaults, optional/nullable `n`, and numeric constraints.
/// Values outside serving limits still parse when their Rust types allow them;
/// these annotations must not become runtime validation.
#[test]
fn owned_request_metadata_preserves_standard_limits_and_dynamo_differences() {
    let doc = document();
    let stream_options = &doc["components"]["schemas"]["dynamo_protocols.chat.ChatCompletionStreamOptions"]
        ["properties"];
    for field in ["include_usage", "continuous_usage_stats"] {
        assert_eq!(stream_options[field]["default"], json!(false));
    }
    for name in [
        CreateChatCompletionRequest::name(),
        CreateCompletionRequest::name(),
    ] {
        let request = &doc["components"]["schemas"][name.as_ref()];
        let fields = &request["properties"];
        assert_eq!(fields["n"]["default"], json!(1), "{name}");
        let types = fields["n"]["type"].as_array().unwrap();
        assert!(types.contains(&json!("integer")) && types.contains(&json!("null")));
        let required = request["required"].as_array().unwrap();
        assert!(!required.contains(&json!("n")));
        for (field, min, max) in [
            ("temperature", 0, 2),
            ("presence_penalty", -2, 2),
            ("frequency_penalty", -2, 2),
            ("n", 1, 128),
        ] {
            assert_eq!(fields[field]["minimum"], json!(min), "{name}.{field}");
            assert_eq!(fields[field]["maximum"], json!(max), "{name}.{field}");
        }
        assert_eq!(fields["top_p"]["exclusiveMinimum"], json!(0));
        assert_eq!(fields["top_p"]["maximum"], json!(1));
        assert_eq!(
            fields["logit_bias"]["additionalProperties"]["type"],
            "number"
        );
        assert_eq!(
            fields["logit_bias"]["additionalProperties"]["minimum"],
            json!(-100)
        );
        assert_eq!(
            fields["logit_bias"]["additionalProperties"]["maximum"],
            json!(100)
        );
        // Backend/model defaults are not universal API promises.
        assert!(fields["temperature"].get("default").is_none());
    }
    let chat = validator::<CreateChatCompletionRequest>();
    let completion = validator::<CreateCompletionRequest>();
    for (field, good, bad) in [
        ("temperature", json!(2), json!(2.1)),
        ("top_p", json!(0.5), json!(0)),
        ("n", json!(128), json!(129)),
        ("presence_penalty", json!(-2), json!(-2.1)),
        ("frequency_penalty", json!(2), json!(2.1)),
        ("logit_bias", json!({"7":0.5}), json!({"7":101})),
    ] {
        for (value, valid) in [(good, true), (bad, false)] {
            let mut chat_wire = json!({"model":"test","messages":[{"role":"user","content":"hi"}]});
            let mut completion_wire = json!({"model":"test","prompt":"hi"});
            chat_wire[field] = value.clone();
            completion_wire[field] = value;
            assert_eq!(chat.is_valid(&chat_wire), valid, "{chat_wire}");
            assert_eq!(
                completion.is_valid(&completion_wire),
                valid,
                "{completion_wire}"
            );
            assert!(serde_json::from_value::<CreateChatCompletionRequest>(chat_wire).is_ok());
            assert!(serde_json::from_value::<CreateCompletionRequest>(completion_wire).is_ok());
        }
    }
    let chat_fields =
        &doc["components"]["schemas"][CreateChatCompletionRequest::name().as_ref()]["properties"];
    assert_eq!(chat_fields["top_logprobs"]["maximum"], json!(20));
    let completion_fields =
        &doc["components"]["schemas"][CreateCompletionRequest::name().as_ref()]["properties"];
    assert_eq!(completion_fields["logprobs"]["maximum"], json!(5));
    assert_eq!(completion_fields["best_of"]["maximum"], json!(20));
}

#[test]
fn named_and_string_tool_choices_have_no_variant_wrapper() {
    let schema = validator::<ChatCompletionToolChoiceOption>();
    for value in [
        json!("none"),
        json!("auto"),
        json!("required"),
        json!({"type":"function","function":{"name":"lookup"}}),
    ] {
        assert!(schema.is_valid(&value), "{value}");
        canonical::<ChatCompletionToolChoiceOption>(value);
    }
    assert!(!schema.is_valid(&json!({"Named":{"type":"function","function":{"name":"lookup"}}})));
    assert!(!schema.is_valid(&json!("invalid")));
}

#[test]
fn custom_deserializers_produce_schema_valid_canonical_messages() {
    // These accepted input forms normalize before serialization. The schema
    // describes the canonical shape, not every custom input/alias branch.
    canonical::<CreateChatCompletionRequest>(json!({
        "model":"test",
        "messages":[
            {"role":"system","tools":[{"name":"lookup"}]},
            {"role":"assistant","reasoning":"thinking","content":null,
             "tool_calls":[{"id":"call1","function":{"name":"lookup","arguments":{"q":"hi"}}}]}
        ],
        "stream_options":{"include_usage":null,"continuous_usage_stats":null}
    }));
}

#[test]
fn multimodal_request_and_completion_preserve_native_shapes() {
    canonical::<CreateChatCompletionRequest>(json!({
        "model":"test","messages":[{"role":"user","content":[
            {"type":"text","text":"describe"},
            {"type":"image_url","image_url":{"url":"https://example.com/image.png"}}
        ]}],"mm_processor_kwargs":{"size":256},"reasoning_effort":"high"
    }));
    canonical::<CreateCompletionRequest>(json!({
        "model":"test","prompt":[],"echo":true,"stream_options":{"include_usage":true}
    }));
    let schema = validator::<CreateCompletionRequest>();
    for echo in [json!(1), json!("true")] {
        let request = json!({"model":"test","prompt":"hi","echo":echo});
        assert!(!schema.is_valid(&request));
        assert!(serde_json::from_value::<CreateCompletionRequest>(request).is_err());
    }
}

#[test]
fn unary_and_stream_responses_retain_reasoning_and_usage() {
    canonical::<CreateChatCompletionResponse>(json!({
        "id":"chat-1","object":"chat.completion","created":1,"model":"test",
        "choices":[{"index":0,"message":{"role":"assistant","content":"hi",
                    "reasoning_content":"thought"},"finish_reason":"stop","logprobs":null}]
    }));
    canonical::<CreateChatCompletionStreamResponse>(json!({
        "id":"chat-1","object":"chat.completion.chunk","created":1,"model":"test",
        "choices":[{"index":0,"delta":{"content":"hi","reasoning_content":"thought"},
                    "finish_reason":null,"logprobs":null}]
    }));
}

#[test]
fn stop_schema_accepts_empty_arrays_and_preserves_variant_shapes() {
    let chat_schema = validator::<CreateChatCompletionRequest>();
    let completion_schema = validator::<CreateCompletionRequest>();
    for stop in [json!([]), json!("end"), json!(["end"]), json!([576])] {
        let chat = json!({
            "model":"test", "messages":[{"role":"user","content":"hi"}], "stop":stop
        });
        let completion = json!({"model":"test","prompt":"hi","stop":stop});
        assert!(chat_schema.is_valid(&chat), "{chat}");
        assert!(completion_schema.is_valid(&completion), "{completion}");
        canonical::<CreateChatCompletionRequest>(chat);
        canonical::<CreateCompletionRequest>(completion);
    }
    for stop in [json!(["end", 576]), json!([-1]), json!(true)] {
        let chat = json!({
            "model":"test", "messages":[{"role":"user","content":"hi"}], "stop":stop
        });
        let completion = json!({"model":"test","prompt":"hi","stop":stop});
        assert!(!chat_schema.is_valid(&chat), "{chat}");
        assert!(!completion_schema.is_valid(&completion), "{completion}");
        assert!(serde_json::from_value::<CreateChatCompletionRequest>(chat).is_err());
        assert!(serde_json::from_value::<CreateCompletionRequest>(completion).is_err());
    }
}

fn required_nullable_output<T: ToSchema + DeserializeOwned + Serialize>(
    input: Value,
    required_fields: &[(&str, &str)],
    omitted_fields: &[&str],
) {
    let parsed: T = serde_json::from_value(input).unwrap();
    let output = serde_json::to_value(parsed).unwrap();
    let schema = validator::<T>();
    assert!(schema.is_valid(&output), "{output}");
    for &(parent, field) in required_fields {
        assert_eq!(
            output.pointer(parent).unwrap().get(field),
            Some(&Value::Null)
        );
        let mut missing = output.clone();
        missing
            .pointer_mut(parent)
            .unwrap()
            .as_object_mut()
            .unwrap()
            .remove(field);
        assert!(
            !schema.is_valid(&missing),
            "missing {parent}/{field}: {missing}"
        );
        // Deserialization remains permissive; this schema describes output.
        assert!(serde_json::from_value::<T>(missing).is_ok());
    }
    for pointer in omitted_fields {
        assert!(output.pointer(pointer).is_none(), "{pointer}: {output}");
    }
}

#[test]
fn unary_response_requires_always_serialized_nullable_fields() {
    required_nullable_output::<CreateChatCompletionResponse>(
        json!({
            "id":"chat-1","object":"chat.completion","created":1,"model":"test",
            "choices":[{"index":0,"message":{"role":"assistant"}}]
        }),
        &[
            ("/choices/0/message", "content"),
            ("/choices/0/message", "refusal"),
            ("/choices/0", "finish_reason"),
            ("/choices/0", "logprobs"),
        ],
        &[
            "/choices/0/message/tool_calls",
            "/choices/0/message/reasoning_content",
            "/usage",
        ],
    );
}

#[test]
fn stream_response_requires_always_serialized_nullable_fields() {
    required_nullable_output::<CreateChatCompletionStreamResponse>(
        json!({
            "id":"chat-1","object":"chat.completion.chunk","created":1,"model":"test",
            "choices":[{"index":0,"delta":{}}]
        }),
        &[("/choices/0", "finish_reason"), ("/choices/0", "logprobs")],
        &[
            "/choices/0/delta/content",
            "/choices/0/delta/refusal",
            "/usage",
        ],
    );
}

#[test]
fn logprobs_require_nullable_content_and_refusal() {
    required_nullable_output::<ChatChoiceLogprobs>(
        json!({}),
        &[("", "content"), ("", "refusal")],
        &[],
    );
    canonical::<ChatChoiceLogprobs>(json!({"content":[],"refusal":[]}));
}

#[test]
fn token_logprobs_require_nullable_bytes_but_not_token_id() {
    required_nullable_output::<ChatCompletionTokenLogprob>(
        json!({"token":"a","logprob":-0.5,"top_logprobs":[]}),
        &[("", "bytes")],
        &["/token_id"],
    );
    for bytes in [json!(null), json!([]), json!([97])] {
        let token = json!({"token":"a","logprob":-0.5,"bytes":bytes,"top_logprobs":[]});
        canonical::<ChatCompletionTokenLogprob>(token.clone());
        let logprobs = json!({"content":[token.clone()],"refusal":[token]});
        canonical::<CreateChatCompletionResponse>(json!({
            "id":"chat-1","object":"chat.completion","created":1,"model":"test",
            "choices":[{"index":0,"message":{"role":"assistant"},"logprobs":logprobs}]
        }));
        canonical::<CreateChatCompletionStreamResponse>(json!({
            "id":"chat-1","object":"chat.completion.chunk","created":1,"model":"test",
            "choices":[{"index":0,"delta":{},"logprobs":logprobs}]
        }));
    }
}
