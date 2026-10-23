// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_protocols::{
    normalization::{openai, sglang, vllm},
    types::CreateCompletionRequest,
    validation::{openai::OpenAIValidation, vllm::VllmValidation},
};
use serde_json::json;

const NORMALIZERS: [fn(&mut Option<u8>); 3] = [
    openai::normalize_completion_logprobs,
    vllm::normalize_completion_logprobs,
    sglang::normalize_completion_logprobs,
];

#[test]
fn omitted_and_null_logprobs_preserve_existing_parse_and_serialization() {
    for input in [
        json!({"model":"test", "prompt":"hello"}),
        json!({"model":"test", "prompt":"hello", "logprobs":null}),
    ] {
        for normalize in NORMALIZERS {
            let mut request: CreateCompletionRequest =
                serde_json::from_value(input.clone()).unwrap();
            assert_eq!(request.logprobs, None);
            let before = request.clone();
            normalize(&mut request.logprobs);
            assert_eq!(request, before);
            assert!(
                serde_json::to_value(&request)
                    .unwrap()
                    .get("logprobs")
                    .is_none()
            );
        }
    }
}

#[test]
fn explicit_counts_are_preserved_including_invalid_values() {
    for value in [0, 5, 6, 10, 20, 21, 255] {
        let input = json!({"model":"test", "prompt":"hello", "logprobs":value});
        let parsed: CreateCompletionRequest = serde_json::from_value(input).unwrap();
        for normalize in NORMALIZERS {
            let mut request = parsed.clone();
            normalize(&mut request.logprobs);
            assert_eq!(request, parsed);
            assert_eq!(serde_json::to_value(&request).unwrap()["logprobs"], value);
        }
        assert_eq!(parsed.logprobs, Some(value as u8));
        assert_eq!(OpenAIValidation::validate(&parsed).is_ok(), value <= 5);
        assert_eq!(VllmValidation::validate(&parsed).is_ok(), value <= 20);
    }
}

#[test]
fn preservation_does_not_widen_the_existing_unsigned_request_field() {
    for value in [-1, 256] {
        let input = json!({"model":"test", "prompt":"hello", "logprobs":value});
        assert!(serde_json::from_value::<CreateCompletionRequest>(input).is_err());
    }
}
