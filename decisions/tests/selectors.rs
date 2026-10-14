// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_decisions::*;
use serde_json::json;

#[test]
fn phase_one_rejects_every_nvext_envelope_with_route_specific_errors() {
    for (route, body, status) in [
        (
            Route::Decisions,
            json!({"model":"m","input":"x","questions":[{"type":"predicate","instructions":"ok?"}]}),
            400,
        ),
        (
            Route::SystemOne,
            json!({"model":"m","state":"x","questions":{"q":{"type":"noul","instructions":"ok?"}}}),
            422,
        ),
    ] {
        assert!(parse_request(&serde_json::to_vec(&body).unwrap(), route).is_ok());
        for extension in [
            json!({}),
            json!({"format":"oai"}),
            json!({"format":"sglang_native"}),
            json!({"format":"unknown"}),
            json!(null),
            json!(true),
            json!([]),
        ] {
            let mut rejected = body.clone();
            rejected["nvext"] = extension;
            let error = parse_request(&serde_json::to_vec(&rejected).unwrap(), route)
                .expect_err("Phase 1 does not accept nvext");
            assert_eq!(error.status, status);
            assert_eq!(error.code, "invalid_request_error");
            assert_eq!(
                error.dialect,
                if route == Route::Decisions {
                    Dialect::OpenAi
                } else {
                    Dialect::Jev
                }
            );
        }
    }
}

#[test]
fn duplicate_keys_are_rejected_recursively() {
    for body in [
        r#"{"model":"a","model":"b"}"#,
        r#"{"state":{"private":"x","private":"y"},"questions":{}}"#,
    ] {
        assert!(parse_request(body.as_bytes(), Route::SystemOne).is_err());
    }
}

#[test]
fn route_determines_the_contract_without_schema_guessing() {
    let openai =
        br#"{"model":"m","input":"x","questions":[{"type":"predicate","instructions":"ok?"}]}"#;
    let jev =
        br#"{"model":"m","state":"x","questions":{"q":{"type":"noul","instructions":"ok?"}}}"#;
    assert_eq!(
        parse_request(openai, Route::Decisions).unwrap().dialect,
        Dialect::OpenAi
    );
    assert_eq!(
        parse_request(jev, Route::SystemOne).unwrap().dialect,
        Dialect::Jev
    );
    assert!(parse_request(openai, Route::SystemOne).is_err());
    assert!(parse_request(jev, Route::Decisions).is_err());
}

#[test]
fn native_payloads_and_mixed_schemas_are_not_accepted() {
    for question in [
        json!({"id":"q","type":"yes_no","question":"ok?"}),
        json!({"id":"q","type":"choice","question":"pick","options":[{"name":"a"},{"name":"b"}]}),
        json!({"id":"q","type":"score","question":"rate","levels":["low","high"]}),
        json!({"type":"predicate","instructions":"ok?","id":"q","question":"ok?"}),
    ] {
        let body = json!({"model":"m","input":"x","questions":[question]});
        let error =
            parse_request(&serde_json::to_vec(&body).unwrap(), Route::Decisions).unwrap_err();
        assert_eq!(error.dialect, Dialect::OpenAi);
        assert_eq!(error.status, 400);
        assert!(error.response_body()["error"].is_object());
    }
}
