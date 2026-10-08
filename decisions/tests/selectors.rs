// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_decisions::*;
use serde_json::json;

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
fn no_format_guessing_or_unknown_controls() {
    for format in [json!("other"), json!(true), json!(null)] {
        let body = json!({"model":"m","input":"x","questions":[],"nvext":{"format":format}});
        assert!(parse_request(&serde_json::to_vec(&body).unwrap(), Route::Decisions).is_err());
    }
    assert!(parse_request(br#"{"model":"m","input":"x","questions":[{"id":"q","type":"yes_no","question":"ok?"}]}"#,Route::Decisions).is_err());
    assert!(parse_request(br#"{"state":"x","nvext":{"format":"oai"},"questions":{"q":{"type":"noul","instructions":"ok?"}}}"#,Route::SystemOne).is_err());
}
