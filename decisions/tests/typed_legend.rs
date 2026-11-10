// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_decisions::*;
use serde_json::{Value, json};

#[test]
fn jev_score_projection_rejects_missing_or_scalar_legend() {
    let payload = json!({"model":"m","state":"x","questions":{
        "s":{"type":"score","criteria":["low","high"]}
    }});
    let request = parse_request(&serde_json::to_vec(&payload).unwrap(), Route::SystemOne).unwrap();
    let outcomes = [QuestionOutcome::Answer(Distribution {
        probabilities: vec![0.4, 0.6],
        label_mass: None,
    })];
    let usage = Usage::default();
    for description in [None, Some(Value::Null), Some(json!(true)), Some(json!(7))] {
        let mut invalid = request.clone();
        invalid.questions[0].candidates[0].description = description;
        let error = project_response(&invalid, "m", &outcomes, &usage).unwrap_err();
        assert_eq!(error.code, "invalid_executor_result");
    }
    assert!(project_response(&request, "m", &outcomes, &usage).is_ok());
}
