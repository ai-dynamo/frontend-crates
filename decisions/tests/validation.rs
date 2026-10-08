// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_decisions::*;
use serde_json::{Value, json};

fn parse(value: Value, route: Route) -> Result<CanonicalRequest, DecisionError> {
    parse_request(&serde_json::to_vec(&value).unwrap(), route)
}
fn native() -> Value {
    json!({"nvext":{"format":"sglang_native"},"input":"x","questions":[{"type":"choice","id":"q","question":"pick","options":[{"name":"a"},{"name":"b"}]}]})
}
fn oai() -> Value {
    json!({"model":"m","input":"x","questions":[{"type":"choice","instructions":"pick","choices":[{"value":true},{"value":"true"}]}]})
}

#[test]
fn native_casefold_and_blank_structured_text_match_reference() {
    let mut value = native();
    value["questions"][0]["options"] = json!([{"name":"Straße"},{"name":"STRASSE"}]);
    assert!(parse(value, Route::Decisions).is_err());
    for blank in [json!({}), json!([]), json!(" ")] {
        let mut value = native();
        value["input"] = blank;
        assert!(parse(value, Route::Decisions).is_err());
    }
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
fn unsupported_modalities_controls_and_versions_fail_before_execution() {
    for (key, value) in [
        ("images", json!(["image"])),
        ("return_prompt_token_ids", json!(true)),
        ("prompt_format_version", json!(2)),
        ("chat_template_kwargs", json!({"enable_thinking":true})),
        ("chat_template_kwargs", json!({"arbitrary":false})),
    ] {
        let mut request = native();
        request[key] = value;
        let error = parse(request, Route::Decisions).unwrap_err();
        assert_eq!(error.code, "unsupported_capability");
        assert_eq!(error.status, 400);
    }
    let mut value = oai();
    value["input"] = json!([{"role":"user","content":[{"type":"input_image","image_url":"data:image/png;base64,x"}]}]);
    assert_eq!(
        parse(value, Route::Decisions).unwrap_err().code,
        "unsupported_capability"
    );
    let mut value = native();
    value["chat_template_kwargs"] = json!({"thinking":false,"enable_thinking":false});
    assert!(parse(value, Route::Decisions).is_ok());
}

#[test]
fn native_schema_and_limits_are_independent() {
    for name in [" ", "bad\nname", "a\u{2028}b", "a\u{2029}b"] {
        let mut value = native();
        value["questions"][0]["options"][0]["name"] = json!(name);
        assert!(parse(value, Route::Decisions).is_err());
    }
    let mut value = native();
    value["questions"][0]["options"] = json!([{"name":"a"},{"name":" A "}]);
    assert!(parse(value, Route::Decisions).is_err());
    let mut value = native();
    let duplicate = value["questions"][0].clone();
    value["questions"].as_array_mut().unwrap().push(duplicate);
    assert!(parse(value, Route::Decisions).is_err());
    for temperature in [0.0, -1.0] {
        let mut value = native();
        value["temperature"] = json!(temperature);
        assert!(parse(value, Route::Decisions).is_err());
    }
    let mut value = native();
    value["questions"][0]["options"] = json!(
        (0..27)
            .map(|i| json!({"name":i.to_string()}))
            .collect::<Vec<_>>()
    );
    assert!(parse(value, Route::Decisions).is_err());
}

#[test]
fn question_ids_remain_exact_and_do_not_become_instructions() {
    let mut value = native();
    let mut second = value["questions"][0].clone();
    second["id"] = json!("Q");
    value["questions"].as_array_mut().unwrap().push(second);
    let request = parse(value, Route::Decisions).unwrap();
    assert_eq!(
        render_question_prompt(&request.input, &request.questions[0]).unwrap(),
        render_question_prompt(&request.input, &request.questions[1]).unwrap()
    );
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
fn extension_disable_is_explicit_and_does_not_disable_oai() {
    let opts = ParseOptions {
        extensions_enabled: false,
        max_questions: 1,
    };
    assert!(
        parse_request_with_options(
            &serde_json::to_vec(&native()).unwrap(),
            Route::Decisions,
            opts
        )
        .is_err()
    );
    let mut value = oai();
    value["nvext"] = json!({"format":"oai"});
    assert!(
        parse_request_with_options(&serde_json::to_vec(&value).unwrap(), Route::Decisions, opts)
            .is_ok()
    );
    for invalid in [json!([]), json!(null), json!({"format":"oai","other":true})] {
        value["nvext"] = invalid;
        assert!(parse(value.clone(), Route::Decisions).is_err());
    }
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
    let request = parse(native(), Route::Decisions).unwrap();
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
            vocabulary_label_mass: false,
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
fn jev_does_not_inherit_native_name_or_level_normalization() {
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
