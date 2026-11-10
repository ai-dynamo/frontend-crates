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
    for (route, valid, duplicate) in [
        (
            Route::SystemOne,
            r#"{"model":"m","state":"x","questions":{"q":{"type":"noul","instructions":"ok?"}}}"#,
            r#"{"model":"m","model":"m","state":"x","questions":{"q":{"type":"noul","instructions":"ok?"}}}"#,
        ),
        (
            Route::SystemOne,
            r#"{"model":"m","state":{"private":"x"},"questions":{"q":{"type":"noul","instructions":"ok?"}}}"#,
            r#"{"model":"m","state":{"private":"x","private":"y"},"questions":{"q":{"type":"noul","instructions":"ok?"}}}"#,
        ),
        (
            Route::SystemOne,
            r#"{"model":"m","state":"x","questions":{"q":{"type":"noul","instructions":"ok?"}}}"#,
            r#"{"model":"m","state":"x","questions":{"q":{"type":"noul","instructions":"ok?","\u0069nstructions":"ok?"}}}"#,
        ),
        (
            Route::SystemOne,
            r#"{"model":"m","state":"x","questions":{"q":{"type":"noul","instructions":"ok?"}},"metadata":{"trace":"x"}}"#,
            r#"{"model":"m","state":"x","questions":{"q":{"type":"noul","instructions":"ok?"}},"metadata":{"trace":"x","trace":"y"}}"#,
        ),
        (
            Route::Decisions,
            r#"{"model":"m","input":"x","questions":[{"type":"predicate","instructions":"ok?"}]}"#,
            r#"{"model":"m","\u006dodel":"m","input":"x","questions":[{"type":"predicate","instructions":"ok?"}]}"#,
        ),
        (
            Route::Decisions,
            r#"{"model":"m","input":"x","questions":[{"type":"predicate","instructions":"ok?"}]}"#,
            r#"{"model":"m","input":"x","questions":[{"type":"predicate","instructions":"ok?","instructions":"ok?"}]}"#,
        ),
    ] {
        assert!(parse_request(valid.as_bytes(), route).is_ok());
        let error = parse_request(duplicate.as_bytes(), route).unwrap_err();
        assert_eq!(error.status, 400);
        assert_eq!(error.code, "invalid_json");
    }
}

#[test]
fn jev_ignores_inert_root_extensions_but_keeps_question_validation_strict() {
    let baseline =
        json!({"model":"m","state":"x","questions":{"q":{"type":"noul","instructions":"ok?"}}});
    let expected =
        parse_request(&serde_json::to_vec(&baseline).unwrap(), Route::SystemOne).unwrap();
    let mut extended = baseline.clone();
    extended["trace_id"] = json!("opaque-trace");
    extended["metadata"] = json!({"context":[null, true, 42, {"nested":"value"}]});
    extended["input"] = json!(null);
    let parsed = parse_request(&serde_json::to_vec(&extended).unwrap(), Route::SystemOne).unwrap();
    assert_eq!(
        serde_json::to_value(parsed).unwrap(),
        serde_json::to_value(expected).unwrap()
    );

    for unknown in ["trace_id", "temperature"] {
        let mut nested = baseline.clone();
        nested["questions"]["q"][unknown] = json!(null);
        assert!(parse_request(&serde_json::to_vec(&nested).unwrap(), Route::SystemOne).is_err());
    }
    let openai = json!({"model":"m","input":"x","questions":[{"type":"predicate","instructions":"ok?"}],"metadata":{}});
    assert!(parse_request(&serde_json::to_vec(&openai).unwrap(), Route::Decisions).is_err());
}

#[test]
fn jev_rejects_reserved_root_controls_by_presence() {
    let baseline =
        json!({"model":"m","state":"x","questions":{"q":{"type":"noul","instructions":"ok?"}}});
    for control in [
        "nvext",
        "temperature",
        "prompt_format_version",
        "return_prompt_token_ids",
    ] {
        for value in [json!(null), json!(1), json!({}), json!(false)] {
            let mut body = baseline.clone();
            body[control] = value;
            let error =
                parse_request(&serde_json::to_vec(&body).unwrap(), Route::SystemOne).unwrap_err();
            assert_eq!(error.status, 422);
            assert_eq!(error.code, "invalid_request_error");
            assert!(error.response_body()["detail"].is_array());
        }
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
