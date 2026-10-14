// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_decisions::*;
use serde_json::{Value, json};

fn parse(value: Value, route: Route) -> Result<CanonicalRequest, DecisionError> {
    parse_request(&serde_json::to_vec(&value).unwrap(), route)
}
fn oai() -> Value {
    json!({"model":"m","input":"x","questions":[{"type":"choice","instructions":"pick","choices":[{"value":true},{"value":"true"}]}]})
}

#[test]
fn question_names_remain_metadata_and_question_limits_are_enforced() {
    let mut value = oai();
    let second = value["questions"][0].clone();
    value["questions"][0]["name"] = json!("opaque name");
    value["questions"].as_array_mut().unwrap().push(second);
    let request = parse(value.clone(), Route::Decisions).unwrap();
    assert_eq!(
        render_question_prompt(&request.input, &request.questions[0]).unwrap(),
        render_question_prompt(&request.input, &request.questions[1]).unwrap()
    );
    let error = parse_request_with_options(
        &serde_json::to_vec(&value).unwrap(),
        Route::Decisions,
        ParseOptions { max_questions: 1 },
    )
    .unwrap_err();
    assert_eq!(error.status, 400);
}

#[test]
fn unsupported_images_and_jev_template_controls_fail_before_execution() {
    let mut value = oai();
    value["input"] = json!([{"role":"user","content":[{"type":"input_image","image_url":"data:image/png;base64,x"}]}]);
    assert_eq!(
        parse(value, Route::Decisions).unwrap_err().code,
        "unsupported_capability"
    );
    let jev =
        json!({"model":"m","state":"x","questions":{"p":{"type":"noul","instructions":"ok?"}}});
    for (key, control) in [
        ("images", json!(["image"])),
        ("chat_template_kwargs", json!({"enable_thinking":true})),
        ("chat_template_kwargs", json!({"arbitrary":false})),
    ] {
        let mut value = jev.clone();
        value[key] = control;
        let error = parse(value, Route::SystemOne).unwrap_err();
        assert_eq!(error.code, "unsupported_capability");
        assert_eq!(error.status, 422);
    }
    let mut value = jev;
    value["chat_template_kwargs"] = json!({"thinking":false,"enable_thinking":false});
    assert!(parse(value, Route::SystemOne).is_ok());
}

#[test]
fn optional_openai_strings_are_not_nullable() {
    for field in ["name", "instructions"] {
        let mut value = oai();
        value["questions"][0][field] = Value::Null;
        assert!(parse(value, Route::Decisions).is_err());
    }
    let mut value = oai();
    value["questions"][0]["choices"][0]["description"] = Value::Null;
    assert!(parse(value, Route::Decisions).is_err());
}

#[test]
fn strict_input_rejects_wrong_roles_mixed_shapes_and_empty_questions() {
    for input in [
        json!(true),
        json!(3),
        json!([]),
        json!([{"role":"assistant","content":"x"}]),
        json!([{"role":"user","content":[]}]),
    ] {
        let mut value = oai();
        value["input"] = input;
        assert!(parse(value, Route::Decisions).is_err());
    }
    let mut value = oai();
    value["questions"] = json!([]);
    assert!(parse(value, Route::Decisions).is_err());
    let mut value = oai();
    value["questions"][0]["question"] = json!("mixed");
    assert!(parse(value, Route::Decisions).is_err());
    let mut value = oai();
    value["model"] = json!(" ");
    assert!(parse(value, Route::Decisions).is_err());
    let mut value = oai();
    value["safety_identifier"] = json!("x".repeat(129));
    assert!(parse(value, Route::Decisions).is_err());
    let mut value = oai();
    value["questions"][0]["choices"] = json!([{"value":true},{"value":true}]);
    assert!(parse(value, Route::Decisions).is_err());
}

#[test]
fn jev_validation_is_sanitized_and_uses_422() {
    for question in [
        json!({"type":"noul"}),
        json!({"type":"noul","instructions":{}}),
        json!({"type":"choice","criteria":{}}),
        json!({"type":"score","criteria":[null]}),
    ] {
        let error = parse(
            json!({"model":"jev-latest","state":"private-secret","questions":{"secret":question}}),
            Route::SystemOne,
        )
        .unwrap_err();
        assert_eq!(error.status, 422);
        assert!(error.response_body()["detail"].is_array());
        assert!(!error.response_body().to_string().contains("secret"));
    }
    for body in [
        "null",
        "[]",
        "{} garbage",
        "{\"state\":NaN}",
        "{\"state\":1e500}",
    ] {
        assert!(parse_request(body.as_bytes(), Route::SystemOne).is_err());
    }
}

#[test]
fn model_capabilities_are_validated_before_dispatch() {
    let request = parse(oai(), Route::Decisions).unwrap();
    let capabilities = Capabilities {
        max_questions: 1,
        max_candidates: 2,
        supports_predicate: true,
        supports_choice: true,
        supports_score: true,
        measured_cache_reads: true,
        prompt_format_version: 1,
    };
    request.validate_capabilities(&capabilities).unwrap();
    for bad in [
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
        assert!(request.validate_capabilities(&bad).is_err());
    }
    let request = parse(oai(), Route::Decisions).unwrap();
    assert!(
        request
            .validate_capabilities(&Capabilities {
                measured_cache_reads: false,
                ..capabilities
            })
            .is_err()
    );
}

#[test]
fn jev_preserves_case_sensitive_names_and_structured_levels() {
    let request = parse(
        json!({"model":"jev-latest","state":"x","questions":{
            "q":{"type":"choice","criteria":{"A":null,"a":null}},
            "s":{"type":"score","criteria":["",{},[]]}
        }}),
        Route::SystemOne,
    )
    .unwrap();
    assert_eq!(
        request.questions[0].candidates[0].value,
        ChoiceValue::String("A".into())
    );
    assert_eq!(
        request.questions[0].candidates[1].value,
        ChoiceValue::String("a".into())
    );
    assert_eq!(request.questions[1].candidates.len(), 3);
}

#[test]
fn oversized_openai_strings_fail_schema_validation() {
    let oversized = "x".repeat(1_048_577);
    for field in ["instructions", "name"] {
        let mut value = oai();
        value["questions"][0][field] = json!(oversized);
        assert_eq!(parse(value, Route::Decisions).unwrap_err().status, 400);
    }
    let mut value = oai();
    value["questions"][0]["choices"][0]["description"] = json!(oversized);
    assert!(parse(value, Route::Decisions).is_err());
    let mut value = oai();
    value["model"] = json!(oversized);
    assert!(parse(value, Route::Decisions).is_err());
    let request = json!({"model":"m","input":"x","questions":[{"type":"score","instructions":"score","levels":[{"label":oversized},{"label":"ok"}]}]});
    assert!(parse(request, Route::Decisions).is_err());
}

#[test]
fn jev_requires_the_model_field_supplied_by_official_sdks() {
    let result = parse(
        json!({"state":"x","questions":{"p":{"type":"noul","instructions":"ok?"}}}),
        Route::SystemOne,
    );
    let error = result.expect_err("Jev model must be required");
    assert_eq!(error.status, 422);
}
