// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_decisions::*;
use serde_json::json;

fn question() -> CanonicalQuestion {
    CanonicalQuestion {
        ordinal: 0,
        id: Some("opaque-id".into()),
        kind: QuestionKind::Choice,
        instructions: Some(json!("Choose")),
        candidates: vec![
            Candidate {
                value: ChoiceValue::Bool(true),
                description: None,
                label: None,
            },
            Candidate {
                value: ChoiceValue::String("true".into()),
                description: None,
                label: None,
            },
        ],
    }
}

#[test]
fn canonical_semantics_keep_typed_choices_and_ids_out_of_prompts() {
    let question = question();
    let alternate = CanonicalQuestion {
        id: Some("different-id".into()),
        ..question.clone()
    };
    let input = json!({"evidence":[1,2]});
    let rendered = render_question_prompt(&input, &question).unwrap();
    assert_eq!(
        rendered,
        render_question_prompt(&input, &alternate).unwrap()
    );
    assert_eq!(rendered.labels, ["A", "B"]);
    assert_ne!(question.candidates[0].value, question.candidates[1].value);
    assert!(!rendered.content.contains("opaque-id"));
}

#[test]
fn canonical_capabilities_reject_missing_evidence_and_backend_limits() {
    let request = CanonicalRequest {
        dialect: Dialect::OpenAi,
        model: "alias".into(),
        input: json!("evidence"),
        questions: vec![question()],
        temperature: 1.0,
        prompt_format_version: 1,
        chat_template_kwargs: Default::default(),
        safety_identifier: None,
    };
    let capabilities = Capabilities {
        max_questions: 1,
        max_candidates: 2,
        supports_predicate: true,
        supports_choice: true,
        supports_score: true,
        vocabulary_label_mass: true,
        measured_cache_reads: true,
        prompt_format_version: 1,
    };
    assert!(request.validate_capabilities(&capabilities).is_ok());
    for invalid in [
        Capabilities {
            measured_cache_reads: false,
            ..capabilities.clone()
        },
        Capabilities {
            max_candidates: 1,
            ..capabilities.clone()
        },
        Capabilities {
            max_questions: 0,
            ..capabilities.clone()
        },
        Capabilities {
            supports_choice: false,
            ..capabilities.clone()
        },
        Capabilities {
            prompt_format_version: 2,
            ..capabilities.clone()
        },
    ] {
        assert!(request.validate_capabilities(&invalid).is_err());
    }
    let native = CanonicalRequest {
        dialect: Dialect::SglangNative,
        ..request
    };
    assert!(
        native
            .validate_capabilities(&Capabilities {
                vocabulary_label_mass: false,
                ..capabilities
            })
            .is_err()
    );
}

#[test]
fn numerical_edge_cases_fail_closed() {
    for scores in [
        vec![],
        vec![f64::NAN],
        vec![f64::INFINITY],
        vec![0.1],
        vec![f64::NEG_INFINITY; 2],
    ] {
        assert!(reduce_vocab_logprobs(&scores, 1.0).is_err());
    }
    let d = reduce_vocab_logprobs(&[f64::NEG_INFINITY, -0.5], 1.0).unwrap();
    assert_eq!(d.probabilities, vec![0.0, 1.0]);
    assert!(reduce_vocab_logprobs(&[-0.01, -0.01], 1.0).is_err());
    assert_eq!(choice_confidence(&[0.5, 0.5]).unwrap(), 0.0);
    assert_eq!(score_confidence(&[0.5, 0.5]).unwrap(), 0.0);
    assert!(reduce_vocab_logprobs(&[-1.0], 0.0).is_err());
}

#[test]
fn float32_vocabulary_scores_allow_expected_rounding_only() {
    let distribution =
        reduce_vocab_logprobs(&[-0.6991652250289917, -0.6871650815010071], 1.0).unwrap();
    assert_eq!(distribution.label_mass, Some(1.0));
    assert!((distribution.probabilities[0] - 0.497).abs() < 1e-7);
    assert!(reduce_vocab_logprobs(&[0.50001_f64.ln(), 0.50001_f64.ln()], 1.0).is_err());
}

#[test]
fn temperature_changes_distribution_not_original_vocabulary_mass() {
    let scores = [0.72_f64.ln(), 0.08_f64.ln()];
    let cold = reduce_vocab_logprobs(&scores, 0.5).unwrap();
    let hot = reduce_vocab_logprobs(&scores, 2.0).unwrap();
    assert_eq!(cold.label_mass, hot.label_mass);
    assert!(cold.probabilities[0] > hot.probabilities[0]);
    assert!((choice_confidence(&[0.1, 0.2, 0.7]).unwrap() - 0.55).abs() < 1e-12);
    assert!((score_confidence(&[0.1, 0.2, 0.7]).unwrap() - 0.4).abs() < 1e-12);
    for t in [f64::NAN, f64::INFINITY, -1.0] {
        assert!(reduce_vocab_logprobs(&scores, t).is_err());
    }
}

#[test]
fn malformed_distributions_and_dialect_errors_fail_closed() {
    for distribution in [
        Distribution {
            probabilities: vec![0.2, 0.2],
            label_mass: None,
        },
        Distribution {
            probabilities: vec![f64::NAN],
            label_mass: None,
        },
        Distribution {
            probabilities: vec![1.0],
            label_mass: Some(1.1),
        },
    ] {
        assert!(
            distribution
                .validate(distribution.probabilities.len())
                .is_err()
        );
    }
    let distribution = Distribution {
        probabilities: vec![1.0],
        label_mass: None,
    };
    assert!(distribution.validate(2).is_err());
    assert_eq!(choice_confidence(&[1.0]).unwrap(), 1.0);
    assert_eq!(score_confidence(&[1.0]).unwrap(), 1.0);
    let native = DecisionError::validation(Dialect::SglangNative, "bad request");
    assert_eq!(
        native.response_body(),
        json!({"object":"error","message":"bad request","type":"invalid_request_error","param":null,"code":400})
    );
    assert_eq!(
        DecisionError::validation(Dialect::OpenAi, "bad request").response_body(),
        json!({"error":{"message":"bad request","type":"invalid_request_error","param":null,"code":"invalid_request_error"}})
    );
    assert_eq!(
        DecisionError::validation(Dialect::Jev, "bad request").status,
        422
    );
}
