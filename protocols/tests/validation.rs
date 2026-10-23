// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_protocols::{
    types::{CreateChatCompletionRequest, CreateCompletionRequest},
    validation::{
        ValidationError,
        openai::{self, OpenAIValidation},
        sglang::{self, SglangValidation},
        vllm::{self, VllmValidation},
    },
};

type FloatValidator = fn(Option<f32>) -> Result<(), ValidationError>;
type CountValidator = fn(Option<u8>) -> Result<(), ValidationError>;
type ChatValidator = fn(&CreateChatCompletionRequest) -> Result<(), ValidationError>;
type CompletionValidator = fn(&CreateCompletionRequest) -> Result<(), ValidationError>;

const CHAT_VALIDATORS: [ChatValidator; 3] = [
    OpenAIValidation::validate,
    VllmValidation::validate,
    SglangValidation::validate,
];
const COMPLETION_VALIDATORS: [CompletionValidator; 3] = [
    OpenAIValidation::validate,
    VllmValidation::validate,
    SglangValidation::validate,
];

// ---------------------------------------------------------------------------
// Field helpers
// ---------------------------------------------------------------------------

#[test]
fn float_helpers_accept_none_and_bounds_and_reject_nonfinite_values() {
    let cases: [(FloatValidator, [f32; 3], [f32; 2]); 9] = [
        (openai::validate_temperature, [0.0, 1.0, 2.0], [-0.01, 2.01]),
        (openai::validate_top_p, [0.0, 0.5, 1.0], [-0.01, 1.01]),
        (
            openai::validate_frequency_penalty,
            [-2.0, 0.0, 2.0],
            [-2.01, 2.01],
        ),
        (
            openai::validate_presence_penalty,
            [-2.0, 0.0, 2.0],
            [-2.01, 2.01],
        ),
        (
            vllm::validate_top_p,
            [f32::MIN_POSITIVE, 0.5, 1.0],
            [-0.0, 1.01],
        ),
        (vllm::validate_min_p, [-0.0, 0.5, 1.0], [-0.01, 1.01]),
        (
            sglang::validate_temperature,
            [0.0, 3.0, f32::MAX],
            [-0.01, -1.0],
        ),
        (
            sglang::validate_top_p,
            [f32::MIN_POSITIVE, 0.5, 1.0],
            [0.0, -0.01],
        ),
        (sglang::validate_min_p, [0.0, 0.5, 1.0], [-0.01, 1.01]),
    ];
    for (validate, valid, invalid) in cases {
        assert!(validate(None).is_ok());
        for value in valid {
            assert!(validate(Some(value)).is_ok());
        }
        for value in invalid
            .into_iter()
            .chain([f32::NAN, f32::INFINITY, f32::NEG_INFINITY])
        {
            assert!(validate(Some(value)).is_err());
        }
    }
}

#[test]
fn top_p_lower_bound_is_inclusive_for_openai_and_exclusive_for_engines() {
    assert_eq!(openai::MIN_TOP_P, vllm::MIN_TOP_P_EXCLUSIVE);
    assert_eq!(vllm::MIN_TOP_P_EXCLUSIVE, sglang::MIN_TOP_P_EXCLUSIVE);
    assert!(openai::validate_top_p(Some(openai::MIN_TOP_P)).is_ok());
    assert!(vllm::validate_top_p(Some(vllm::MIN_TOP_P_EXCLUSIVE)).is_err());
    assert!(sglang::validate_top_p(Some(sglang::MIN_TOP_P_EXCLUSIVE)).is_err());
}

#[test]
fn count_helpers_accept_none_and_bounds_and_report_field_and_value() {
    let cases: [(CountValidator, &str, &str, u8, u8); 5] = [
        (
            openai::validate_completion_logprobs,
            "logprobs",
            "Logprobs must be between 0 and 5",
            0,
            5,
        ),
        (
            vllm::validate_completion_logprobs,
            "logprobs",
            "Logprobs must be between 0 and 20",
            0,
            20,
        ),
        (
            openai::validate_n,
            "n",
            "n must be between 1 and 128",
            1,
            128,
        ),
        (
            openai::validate_top_logprobs,
            "top_logprobs",
            "Top_logprobs must be between 0 and 20",
            0,
            20,
        ),
        (
            openai::validate_best_of,
            "best_of",
            "Best_of must be between 0 and 20",
            0,
            20,
        ),
    ];
    for (validate, field, message, minimum, maximum) in cases {
        assert!(validate(None).is_ok());
        for value in [minimum, maximum] {
            assert!(validate(Some(value)).is_ok());
        }
        for value in [maximum + 1, u8::MAX] {
            assert_eq!(
                validate(Some(value)).unwrap_err(),
                ValidationError {
                    field,
                    message: message.into(),
                    value: value.to_string(),
                }
            );
        }
    }
    assert_eq!(openai::validate_n(Some(0)).unwrap_err().value, "0");
}

#[test]
fn logit_bias_checks_the_callers_f32_value_without_clamping() {
    for value in [-100.0, -0.0, 0.0, 100.0] {
        assert!(openai::validate_logit_bias_value(value).is_ok());
    }
    for value in [-100.01, 100.01, f32::NAN, f32::NEG_INFINITY, f32::INFINITY] {
        assert_eq!(
            openai::validate_logit_bias_value(value).unwrap_err(),
            ValidationError {
                field: "logit_bias",
                message: "Logit bias must be between -100 and 100".into(),
                value: value.to_string(),
            }
        );
    }
}

// ---------------------------------------------------------------------------
// Request traits
// ---------------------------------------------------------------------------

#[test]
fn all_traits_accept_unspecified_values_without_mutation() {
    let chat = CreateChatCompletionRequest::default();
    let completion = CreateCompletionRequest::default();
    let chat_before = chat.clone();
    let completion_before = completion.clone();
    for validate in CHAT_VALIDATORS {
        assert!(validate(&chat).is_ok());
    }
    for validate in COMPLETION_VALIDATORS {
        assert!(validate(&completion).is_ok());
    }
    assert_eq!(chat, chat_before);
    assert_eq!(completion, completion_before);
}

#[test]
fn all_traits_return_the_first_error_in_order_without_mutation() {
    for validate in CHAT_VALIDATORS {
        let mut request = CreateChatCompletionRequest {
            temperature: Some(-1.0),
            top_p: Some(-1.0),
            frequency_penalty: Some(3.0),
            presence_penalty: Some(3.0),
            ..Default::default()
        };
        let before = request.clone();
        assert_eq!(validate(&request).unwrap_err().field, "temperature");
        assert_eq!(request, before);
        request.temperature = None;
        assert_eq!(validate(&request).unwrap_err().field, "top_p");
        request.top_p = None;
        assert_eq!(validate(&request).unwrap_err().field, "frequency_penalty");
        request.frequency_penalty = None;
        assert_eq!(validate(&request).unwrap_err().field, "presence_penalty");
        request.presence_penalty = None;
        assert!(validate(&request).is_ok());
    }
    for (index, validate) in COMPLETION_VALIDATORS.into_iter().enumerate() {
        let mut request = CreateCompletionRequest {
            temperature: Some(-1.0),
            top_p: Some(-1.0),
            frequency_penalty: Some(3.0),
            presence_penalty: Some(3.0),
            logprobs: Some(21),
            ..Default::default()
        };
        let before = request.clone();
        assert_eq!(validate(&request).unwrap_err().field, "temperature");
        assert_eq!(request, before);
        request.temperature = None;
        assert_eq!(validate(&request).unwrap_err().field, "top_p");
        request.top_p = None;
        assert_eq!(validate(&request).unwrap_err().field, "frequency_penalty");
        request.frequency_penalty = None;
        assert_eq!(validate(&request).unwrap_err().field, "presence_penalty");
        request.presence_penalty = None;
        if index < 2 {
            assert_eq!(validate(&request).unwrap_err().field, "logprobs");
        } else {
            assert!(validate(&request).is_ok()); // SGLang logprobs are uncovered.
        }
    }
}

#[test]
fn openai_trait_checks_counts_last_and_engines_leave_them_uncovered() {
    let mut chat = CreateChatCompletionRequest {
        presence_penalty: Some(3.0),
        n: Some(0),
        top_logprobs: Some(21),
        ..Default::default()
    };
    let before = chat.clone();
    assert_eq!(
        OpenAIValidation::validate(&chat).unwrap_err().field,
        "presence_penalty"
    );
    assert_eq!(chat, before);
    chat.presence_penalty = None;
    assert_eq!(OpenAIValidation::validate(&chat).unwrap_err().field, "n");
    assert!(VllmValidation::validate(&chat).is_ok());
    assert!(SglangValidation::validate(&chat).is_ok());
    chat.n = Some(1);
    assert_eq!(
        OpenAIValidation::validate(&chat).unwrap_err().field,
        "top_logprobs"
    );
    chat.top_logprobs = Some(20);
    chat.logprobs = Some(false); // The logprobs relationship is uncovered.
    assert!(OpenAIValidation::validate(&chat).is_ok());

    let mut completion = CreateCompletionRequest {
        logprobs: Some(6),
        n: Some(0),
        best_of: Some(21),
        ..Default::default()
    };
    let before = completion.clone();
    assert_eq!(
        OpenAIValidation::validate(&completion).unwrap_err().field,
        "logprobs"
    );
    assert_eq!(completion, before);
    completion.logprobs = None;
    assert_eq!(
        OpenAIValidation::validate(&completion).unwrap_err().field,
        "n"
    );
    assert!(VllmValidation::validate(&completion).is_ok());
    assert!(SglangValidation::validate(&completion).is_ok());
    completion.n = Some(1);
    assert_eq!(
        OpenAIValidation::validate(&completion).unwrap_err().field,
        "best_of"
    );
    completion.best_of = Some(0);
    completion.stream = Some(true); // Streaming relationships are uncovered.
    assert!(OpenAIValidation::validate(&completion).is_ok());
}

#[test]
fn engine_traits_apply_overrides_without_running_the_baseline_first() {
    let mut chat = CreateChatCompletionRequest::default();
    let mut completion = CreateCompletionRequest::default();

    chat.top_p = Some(0.0);
    completion.top_p = Some(0.0);
    assert!(OpenAIValidation::validate(&chat).is_ok());
    assert!(OpenAIValidation::validate(&completion).is_ok());
    for validate in &CHAT_VALIDATORS[1..] {
        assert_eq!(validate(&chat).unwrap_err().field, "top_p");
    }
    for validate in &COMPLETION_VALIDATORS[1..] {
        assert_eq!(validate(&completion).unwrap_err().field, "top_p");
    }

    chat.top_p = Some(0.5);
    completion.top_p = Some(0.5);
    chat.temperature = Some(3.0);
    completion.temperature = Some(3.0);
    assert!(SglangValidation::validate(&chat).is_ok());
    assert!(SglangValidation::validate(&completion).is_ok());
    for validate in &CHAT_VALIDATORS[..2] {
        assert_eq!(validate(&chat).unwrap_err().field, "temperature");
    }
    for validate in &COMPLETION_VALIDATORS[..2] {
        assert_eq!(validate(&completion).unwrap_err().field, "temperature");
    }

    completion.temperature = None;
    completion.logprobs = Some(21);
    assert_eq!(
        VllmValidation::validate(&completion).unwrap_err().field,
        "logprobs"
    );
    assert!(SglangValidation::validate(&completion).is_ok());
}

// ---------------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------------

#[test]
fn diagnostics_render_message_and_rejected_value_without_a_source() {
    let error = openai::validate_temperature(Some(2.5)).unwrap_err();
    assert_eq!(error.value, "2.5");
    assert_eq!(
        error.to_string(),
        format!("{}, got {}", error.message, error.value)
    );
    assert!(std::error::Error::source(&error).is_none());
    for (value, expected) in [
        (f32::NAN, "NaN"),
        (f32::INFINITY, "inf"),
        (f32::NEG_INFINITY, "-inf"),
    ] {
        assert_eq!(
            openai::validate_temperature(Some(value)).unwrap_err().value,
            expected
        );
    }
}

#[test]
fn diagnostic_text_matches_existing_wording() {
    assert_eq!(
        sglang::validate_temperature(Some(f32::NAN))
            .unwrap_err()
            .to_string(),
        "Temperature must be finite and at least 0, got NaN"
    );
    assert_eq!(
        openai::validate_top_p(Some(2.0)).unwrap_err().to_string(),
        "Top_p must be between 0 and 1, got 2",
    );
    assert_eq!(
        sglang::validate_top_p(Some(0.0)).unwrap_err().to_string(),
        "Top_p must be between 0 and 1, got 0"
    );
    assert_eq!(vllm::validate_top_p(Some(-0.0)).unwrap_err().value, "-0");
    let error = vllm::validate_min_p(Some(-0.5)).unwrap_err();
    assert_eq!(error.field, "min_p");
    assert_eq!(error.message, "Min_p must be between 0 and 1");
}
