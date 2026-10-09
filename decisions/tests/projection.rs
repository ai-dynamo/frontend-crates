// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_decisions::*;
use serde_json::{Value, json};

fn parse(value: Value, route: Route) -> CanonicalRequest {
    parse_request(&serde_json::to_vec(&value).unwrap(), route).unwrap()
}
fn usage() -> Usage {
    Usage {
        input_tokens: 100,
        output_tokens: 2,
        cached_tokens: Some(20),
        reasoning_tokens: 1,
    }
}
fn dist() -> Distribution {
    Distribution {
        probabilities: vec![0.9, 0.1],
        label_mass: Some(0.94),
    }
}

#[test]
fn complete_openai_response_preserves_order_labels_names_and_refusal() {
    let req = parse(
        json!({"model":"alias","input":"x","questions":[
            {"type":"predicate","name":"z","instructions":"ok?"},
            {"type":"score","name":"a","instructions":"rate","levels":[{"label":"Low"},{"label":"High"}]},
            {"type":"choice","name":"z","instructions":"pick","choices":[{"value":"billing"},{"value":"technical"}]},
            {"type":"predicate","instructions":"private?"}
        ]}),
        Route::Decisions,
    );
    let actual = project_response(
        &req,
        "canonical",
        &[
            QuestionOutcome::Answer(dist()),
            QuestionOutcome::Answer(dist()),
            QuestionOutcome::Answer(dist()),
            QuestionOutcome::Refusal("private".into()),
        ],
        &usage(),
    )
    .unwrap();
    let expected = json!({"model":"canonical","answers":[
        {"type":"predicate","name":"z","probability":0.9},
        {"type":"score","name":"a","score":0.1,"confidence":0.8,"probabilities":[{"value":0,"label":"Low","probability":0.9},{"value":1,"label":"High","probability":0.1}]},
        {"type":"choice","name":"z","choice":"billing","confidence":0.8,"probabilities":[{"value":"billing","probability":0.9},{"value":"technical","probability":0.1}]},
        {"type":"refusal","name":null}],
        "usage":{"input_tokens":100,"output_tokens":2,"total_tokens":102,"input_tokens_details":{"cached_tokens":20,"cache_write_tokens":0},"output_tokens_details":{"reasoning_tokens":1}}
    });
    assert_eq!(actual, expected);
    let _: protocols::openai::Response = serde_json::from_value(actual).unwrap();
}

#[test]
fn jev_predicate_score_fixtures_preserve_criteria() {
    let jev = parse(
        json!({"model":"jev-latest","state":{"text":"x"},"questions":{
            "p":{"type":"noul","instructions":"ok?","criteria":{"true":{"means":"yes"},"false":"no"}},
            "s":{"type":"score","instructions":"rate","criteria":[{"label":"Low"},{"label":"High"}]}
        }}),
        Route::SystemOne,
    );
    let outcomes = vec![
        QuestionOutcome::Answer(dist()),
        QuestionOutcome::Answer(dist()),
    ];
    let j = project_response(&jev, "canonical", &outcomes, &usage()).unwrap();
    assert_eq!(
        j["answers"]["p"],
        json!({"type":"noul","noul":0.9,"x_label_mass":0.94})
    );
    assert_eq!(
        j["answers"]["s"],
        json!({"type":"score","score":0.1,"confidence":0.8,"probabilities":{"0":0.9,"1":0.1},"legend":{"0":{"label":"Low"},"1":{"label":"High"}},"x_label_mass":0.94})
    );
    let _: protocols::systemone::Response = serde_json::from_value(j).unwrap();
}

#[test]
fn malformed_executor_results_and_counters_fail_closed() {
    let request = parse(
        json!({"model":"m","input":"x","questions":[{"type":"predicate","instructions":"ok?"}]}),
        Route::Decisions,
    );
    for d in [
        Distribution {
            probabilities: vec![0.9],
            label_mass: None,
        },
        Distribution {
            probabilities: vec![0.0, 0.0],
            label_mass: None,
        },
        Distribution {
            probabilities: vec![1.1, -0.1],
            label_mass: None,
        },
        Distribution {
            probabilities: vec![f64::NAN, 0.1],
            label_mass: None,
        },
        Distribution {
            label_mass: Some(1.1),
            ..dist()
        },
        Distribution {
            label_mass: Some(f64::NAN),
            ..dist()
        },
    ] {
        let error =
            project_response(&request, "m", &[QuestionOutcome::Answer(d)], &usage()).unwrap_err();
        assert_eq!(error.status, 502);
    }
    assert!(project_response(&request, "m", &[], &usage()).is_err());
    for counters in [
        Usage {
            input_tokens: u64::MAX,
            ..usage()
        },
        Usage {
            cached_tokens: Some(101),
            ..usage()
        },
        Usage {
            reasoning_tokens: 3,
            ..usage()
        },
        Usage {
            input_tokens: i32::MAX as u64,
            ..usage()
        },
    ] {
        assert!(
            project_response(&request, "m", &[QuestionOutcome::Answer(dist())], &counters).is_err()
        );
    }
}

#[test]
fn jev_refusal_is_explicit_and_missing_label_mass_is_not_invented() {
    let request = parse(
        json!({"model":"m","state":"x","questions":{"p":{"type":"noul","instructions":"ok?"}}}),
        Route::SystemOne,
    );
    let error = project_response(
        &request,
        "m",
        &[QuestionOutcome::Refusal("secret".into())],
        &usage(),
    )
    .unwrap_err();
    assert_eq!(error.status, 422);
    assert_eq!(error.code, "decision_refused");
    assert!(!error.response_body().to_string().contains("secret"));
    let response = project_response(
        &request,
        "m",
        &[QuestionOutcome::Answer(Distribution {
            label_mass: None,
            ..dist()
        })],
        &usage(),
    )
    .unwrap();
    assert!(response["answers"]["p"].get("x_label_mass").is_none());
}

#[test]
fn distribution_temperature_ties_and_confidence_are_independent() {
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
    assert_eq!(
        reduce_vocab_logprobs(&[-1.0, -1.0], f64::MIN_POSITIVE)
            .unwrap()
            .probabilities,
        vec![0.5, 0.5]
    );
    assert_eq!(
        reduce_vocab_logprobs(&[-10000.0, -10001.0], 1.0)
            .unwrap()
            .label_mass,
        Some(0.0)
    );
    let req = parse(
        json!({"model":"m","input":"x","questions":[{"type":"choice","instructions":"pick","choices":[{"value":"first"},{"value":"second"}]}]}),
        Route::Decisions,
    );
    let response = project_response(
        &req,
        "m",
        &[QuestionOutcome::Answer(Distribution {
            probabilities: vec![0.5, 0.5],
            label_mass: Some(0.8),
        })],
        &usage(),
    )
    .unwrap();
    assert_eq!(response["answers"][0]["choice"], "first");
}

#[test]
fn dialect_error_bodies_preserve_openai_and_jev_contracts() {
    let openai = DecisionError::validation(Dialect::OpenAi, "bad request");
    assert_eq!(
        openai.response_body(),
        json!({"error":{"message":"bad request","type":"invalid_request_error","param":null,"code":"invalid_request_error"}})
    );
    let jev = DecisionError::new(Dialect::Jev, 404, "model_not_found", "unknown model");
    assert_eq!(
        jev.response_body(),
        json!({"detail":"unknown model","code":"model_not_found"})
    );
}
