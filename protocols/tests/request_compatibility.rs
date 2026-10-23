// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Wire-level request compatibility regressions, including ai-dynamo/frontend-crates#299.
//! Backend capability filtering belongs in the serving adapter, not serde.

use dynamo_protocols::types::responses::{
    CreateResponse, TextResponseFormatConfiguration, Verbosity,
};
use dynamo_protocols::types::{CreateChatCompletionRequest, CreateCompletionRequest};
use serde_json::json;

/// Declaring a schema default must not fill in `n` during JSON parsing.
/// Omitted/null counts stay `None` and serialize without `n`; explicit counts,
/// including zero, are preserved. Run with and without `protocol-schema` enabled.
#[test]
fn choice_count_schema_default_does_not_inject_wire_values() {
    for n in [
        None,
        Some(json!(null)),
        Some(json!(0)),
        Some(json!(1)),
        Some(json!(2)),
    ] {
        let mut chat_wire = json!({
            "model": "example-model",
            "messages": [{"role": "user", "content": "Say hello"}]
        });
        let mut completion_wire = json!({"model": "example-model", "prompt": "Say hello"});
        if let Some(value) = &n {
            chat_wire["n"] = value.clone();
            completion_wire["n"] = value.clone();
        }
        let expected = n
            .as_ref()
            .and_then(serde_json::Value::as_u64)
            .map(|n| n as u8);
        let chat: CreateChatCompletionRequest = serde_json::from_value(chat_wire).unwrap();
        let completion: CreateCompletionRequest = serde_json::from_value(completion_wire).unwrap();
        assert_eq!(chat.n, expected);
        assert_eq!(completion.n, expected);
        for serialized in [
            serde_json::to_value(chat).unwrap(),
            serde_json::to_value(completion).unwrap(),
        ] {
            assert_eq!(serialized.get("n").cloned(), expected.map(|n| json!(n)));
        }
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
