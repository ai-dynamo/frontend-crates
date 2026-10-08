// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_decisions::*;
use serde_json::json;

fn parse(body: serde_json::Value, route: Route) -> CanonicalRequest {
    parse_request(&serde_json::to_vec(&body).unwrap(), route).unwrap()
}

fn usage() -> Usage {
    Usage {
        input_tokens: 85,
        output_tokens: 0,
        cached_tokens: Some(0),
        reasoning_tokens: 0,
    }
}

#[test]
fn default_openai_preserves_typed_values_and_null_names() {
    let request = parse(
        json!({"model":"m","input":"test","questions":[
            {"type":"choice","instructions":"pick","choices":[{"value":true},{"value":"true"}]}
        ]}),
        Route::Decisions,
    );
    assert_eq!(request.dialect, Dialect::OpenAi);
    assert_ne!(
        request.questions[0].candidates[0].value,
        request.questions[0].candidates[1].value
    );
    let response = project_response(
        &request,
        "canonical",
        &[QuestionOutcome::Answer(
            reduce_vocab_logprobs(&[0.846_f64.ln(), 0.094_f64.ln()], 1.0).unwrap(),
        )],
        &usage(),
    )
    .unwrap();
    assert_eq!(response["answers"][0]["choice"], true);
    assert!(response["answers"][0]["name"].is_null());
    assert_eq!(response["model"], "canonical");
    assert_eq!(response["usage"]["total_tokens"], 85);
}

#[test]
fn native_selects_both_contracts_and_echoes_alias() {
    let request = parse(
        json!({"model":"alias","nvext":{"format":"sglang_native"},"input":"test","questions":[
            {"type":"choice","id":"d","question":"pick","options":[{"name":"a"},{"name":"b"}]}
        ]}),
        Route::Decisions,
    );
    let distribution = reduce_vocab_logprobs(&[0.846_f64.ln(), 0.094_f64.ln()], 1.0).unwrap();
    let response = project_response(
        &request,
        "canonical",
        &[QuestionOutcome::Answer(distribution)],
        &usage(),
    )
    .unwrap();
    assert_eq!(response["model"], "alias");
    assert_eq!(response["answers"]["d"]["choice"], "a");
    assert!((response["answers"]["d"]["label_mass"].as_f64().unwrap() - 0.94).abs() < 1e-12);
    assert!(response["answers"]["d"].get("confidence").is_none());
}

#[test]
fn jev_preserves_structured_state_order_and_singletons() {
    let request = parse_request(br#"{"model":"jev-latest","state":{"x":[1,2]},"questions":{"z":{"type":"choice","criteria":{"last":null}},"a":{"type":"score","criteria":[{"rubric":"ok"}]}}}"#,Route::SystemOne).unwrap();
    assert_eq!(request.model, "jev-latest");
    assert_eq!(request.questions[0].id.as_deref(), Some("z"));
    let distribution = Distribution {
        probabilities: vec![1.0],
        label_mass: None,
    };
    let response = project_response(
        &request,
        "m",
        &[
            QuestionOutcome::Answer(distribution.clone()),
            QuestionOutcome::Answer(distribution),
        ],
        &usage(),
    )
    .unwrap();
    assert_eq!(response["answers"]["z"]["confidence"], 1.0);
    assert_eq!(response["answers"]["a"]["score"], 0.0);
    assert_eq!(
        response["answers"]["a"]["legend"]["0"],
        json!({"rubric":"ok"})
    );
}

#[test]
fn actual_refusal_is_not_a_scored_answer() {
    let request = parse(
        json!({"model":"m","input":"x","questions":[{"type":"predicate","instructions":"safe?"}]}),
        Route::Decisions,
    );
    let response = project_response(
        &request,
        "m",
        &[QuestionOutcome::Refusal("internal reason".into())],
        &usage(),
    )
    .unwrap();
    assert_eq!(
        response["answers"][0],
        json!({"type":"refusal","name":null})
    );
    assert!(!response.to_string().contains("internal reason"));
}

#[test]
fn missing_cache_measurement_is_not_fabricated() {
    let request = parse(
        json!({"model":"m","input":"x","questions":[{"type":"predicate","instructions":"safe?"}]}),
        Route::Decisions,
    );
    let mut counters = usage();
    counters.cached_tokens = None;
    assert!(
        project_response(
            &request,
            "m",
            &[QuestionOutcome::Answer(Distribution {
                probabilities: vec![0.9, 0.1],
                label_mass: Some(0.8)
            })],
            &counters
        )
        .is_err()
    );
}
