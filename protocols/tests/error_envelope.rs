// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Wire shape of the OpenAI error envelope, as OpenAI SDKs read it and as
//! `WrappedError` / `ApiError` model it: `{"error": {message, type, param, code}}`
//! with `type` an OpenAI error class and `code` a string or null. A frontend that
//! emits this shape is classifiable by the official SDKs; the flat
//! `{message, type, code: <int>}` body is not (ai-dynamo/frontend-crates#208).

use dynamo_protocols::error::{ApiError, WrappedError};

const DOCUMENTED: &str = r#"{"error":{"message":"Invalid type for 'max_tokens': expected an integer, but got a string instead.","type":"invalid_request_error","param":"max_tokens","code":null}}"#;

#[test]
fn documented_envelope_round_trips() {
    let parsed: WrappedError = serde_json::from_str(DOCUMENTED).unwrap();
    assert_eq!(
        parsed.error.r#type.as_deref(),
        Some("invalid_request_error")
    );
    assert_eq!(parsed.error.param.as_deref(), Some("max_tokens"));
    assert_eq!(parsed.error.code, None);

    let reserialized: serde_json::Value = serde_json::to_value(&parsed).unwrap();
    let documented: serde_json::Value = serde_json::from_str(DOCUMENTED).unwrap();
    assert_eq!(reserialized, documented);
}

#[test]
fn code_is_a_string_or_null() {
    let coded: WrappedError = serde_json::from_str(
        r#"{"error":{"message":"The model `x` does not exist","type":"invalid_request_error","param":null,"code":"model_not_found"}}"#,
    )
    .unwrap();
    assert_eq!(coded.error.code.as_deref(), Some("model_not_found"));

    let integer_code = serde_json::from_str::<WrappedError>(
        r#"{"error":{"message":"Bad Request","type":"invalid_request_error","param":null,"code":400}}"#,
    );
    assert!(
        integer_code.is_err(),
        "an integer `code` is not part of the OpenAI contract and must not parse"
    );
}

#[test]
fn optional_fields_may_be_omitted() {
    let minimal: WrappedError = serde_json::from_str(
        r#"{"error":{"message":"boom","type":null,"param":null,"code":null}}"#,
    )
    .unwrap();
    assert_eq!(minimal.error.message, "boom");
    assert_eq!(minimal.error.r#type, None);
}

#[test]
fn flat_body_is_not_an_envelope() {
    // The shape #208 reports from the HTTP frontend: no `error` wrapper, HTTP
    // reason phrase as `type`, integer `code`. SDKs reading `body.error` see
    // nothing, and `WrappedError` rejects it for the same reason.
    let flat = serde_json::from_str::<WrappedError>(
        r#"{"message":"Failed to deserialize the JSON body into the target type","type":"Bad Request","code":400}"#,
    );
    assert!(flat.is_err());
}

#[test]
fn display_carries_type_param_and_code() {
    let error = ApiError {
        message: "Invalid value".to_string(),
        r#type: Some("invalid_request_error".to_string()),
        param: Some("temperature".to_string()),
        code: Some("invalid_value".to_string()),
    };
    assert_eq!(
        error.to_string(),
        "invalid_request_error: Invalid value (param: temperature) (code: invalid_value)"
    );
}
