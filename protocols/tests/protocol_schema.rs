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
    CreateChatCompletionResponse,
    CreateChatCompletionStreamResponse
)))]
struct Contract;

fn document() -> Value {
    serde_json::to_value(Contract::openapi()).unwrap()
}

fn validator<T: ToSchema>() -> jsonschema::Validator {
    let mut doc = document();
    doc["$ref"] = json!(format!("#/components/schemas/{}", T::name()));
    jsonschema::validator_for(&doc).unwrap()
}

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

#[test]
fn unannotated_dependency_types_are_explicit_import_slots() {
    let doc = document();
    let schemas = doc["components"]["schemas"].as_object().unwrap();
    for (name, schema) in schemas {
        if let Some(short) = name.strip_prefix("async_openai.") {
            assert_eq!(schema["x-dynamo-schema-import"]["crate"], "async-openai");
            assert_eq!(schema["x-dynamo-schema-import"]["type"], short);
            let kinds = schema["type"].as_array().unwrap();
            assert!(
                !kinds.contains(&json!("null")),
                "Option<T> owns nullability"
            );
            assert!(
                kinds.len() > 1,
                "must not fabricate an object-only contract"
            );
        }
    }
    assert!(schemas.contains_key("async_openai.Prompt"));
    assert!(schemas.contains_key("async_openai.FunctionObject"));
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
