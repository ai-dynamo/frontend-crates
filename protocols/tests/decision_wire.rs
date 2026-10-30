// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_protocols::types::decisions::{ChoiceValue, openai, systemone};
use serde_json::json;

#[test]
fn decision_choice_values_preserve_json_types() {
    assert_eq!(
        serde_json::from_value::<ChoiceValue>(json!(true)).unwrap(),
        ChoiceValue::Bool(true)
    );
    assert_eq!(
        serde_json::from_value::<ChoiceValue>(json!("true")).unwrap(),
        ChoiceValue::String("true".into())
    );
    for invalid in [json!(1), json!(null), json!([]), json!({})] {
        assert!(serde_json::from_value::<ChoiceValue>(invalid).is_err());
    }
}

#[test]
fn decision_openai_wire_rejects_native_fields_and_nullable_names() {
    let request = json!({"model":"m","input":"evidence","questions":[{"type":"predicate","instructions":"safe?"}]});
    let decoded: openai::Request = serde_json::from_value(request.clone()).unwrap();
    assert_eq!(serde_json::to_value(decoded).unwrap(), request);
    for question in [
        json!({"type":"predicate","instructions":"safe?","name":null}),
        json!({"type":"yes_no","question":"safe?","id":"q"}),
    ] {
        assert!(serde_json::from_value::<openai::Question>(question).is_err());
    }
}

#[test]
fn decision_jev_preserves_question_order_and_case_sensitive_values() {
    let jev: systemone::Request = serde_json::from_str(r#"{"model":"m","state":{"evidence":[1,2]},"questions":{"z":{"type":"noul"},"a":{"type":"choice","criteria":{"A":null,"a":null}}}}"#).unwrap();
    assert_eq!(
        jev.questions.keys().map(String::as_str).collect::<Vec<_>>(),
        ["z", "a"]
    );
    assert!(
        serde_json::from_value::<systemone::Request>(json!({"state":"x","questions":{}})).is_err()
    );
}

#[test]
fn decision_response_serialization_preserves_refusal_and_typed_choice() {
    assert_eq!(
        serde_json::to_value(openai::Answer::Refusal { name: None }).unwrap(),
        json!({"type":"refusal","name":null})
    );
    assert_eq!(
        serde_json::to_value(openai::ChoiceProbability {
            value: ChoiceValue::Bool(false),
            probability: 0.2
        })
        .unwrap(),
        json!({"value":false,"probability":0.2})
    );
    assert_eq!(
        serde_json::to_value(systemone::Answer::Noul {
            noul: 0.8,
            x_label_mass: None
        })
        .unwrap(),
        json!({"type":"noul","noul":0.8})
    );
}

#[cfg(feature = "protocol-schema")]
#[test]
fn decision_public_schemas_use_distinct_component_names() {
    use utoipa::{OpenApi, ToSchema};
    assert_eq!(openai::Request::name(), "DecisionOpenAiRequest");
    assert_eq!(systemone::Request::name(), "DecisionSystemOneRequest");

    #[derive(OpenApi)]
    #[openapi(components(schemas(
        openai::Request,
        openai::Response,
        systemone::Request,
        systemone::Response
    )))]
    struct Api;

    fn check_references(
        value: &serde_json::Value,
        schemas: &serde_json::Map<String, serde_json::Value>,
    ) {
        match value {
            serde_json::Value::Object(object) => {
                if let Some(name) = object.get("$ref").and_then(serde_json::Value::as_str) {
                    let name = name.strip_prefix("#/components/schemas/").unwrap();
                    assert!(
                        schemas.contains_key(name),
                        "missing schema component {name}"
                    );
                }
                for child in object.values() {
                    check_references(child, schemas);
                }
            }
            serde_json::Value::Array(array) => {
                for child in array {
                    check_references(child, schemas);
                }
            }
            _ => {}
        }
    }

    let document = serde_json::to_value(Api::openapi()).unwrap();
    let schemas = document["components"]["schemas"].as_object().unwrap();
    assert!(schemas.contains_key("DecisionSystemOneStructuredText"));
    check_references(&document, schemas);
}

#[test]
fn decision_openai_image_detail_and_message_type_follow_wire_constraints() {
    for detail in [
        json!("low"),
        json!("high"),
        json!("auto"),
        json!("original"),
        json!(null),
    ] {
        let part =
            json!({"type":"input_image","image_url":"data:image/png;base64,eA==","detail":detail});
        let decoded: openai::InputPart = serde_json::from_value(part.clone()).unwrap();
        assert_eq!(serde_json::to_value(decoded).unwrap(), part);
    }
    for detail in [json!("medium"), json!(true), json!(1), json!({}), json!([])] {
        let part = json!({"type":"input_image","image_url":"https://example.com/image.png","detail":detail});
        assert!(serde_json::from_value::<openai::InputPart>(part).is_err());
    }
    for message in [
        json!({"role":"user","content":"evidence"}),
        json!({"role":"user","content":"evidence","type":"message"}),
    ] {
        let decoded: openai::InputMessage = serde_json::from_value(message.clone()).unwrap();
        assert_eq!(serde_json::to_value(decoded).unwrap(), message);
    }
    for kind in [json!(null), json!("other"), json!(true)] {
        assert!(
            serde_json::from_value::<openai::InputMessage>(
                json!({"role":"user","content":"evidence","type":kind})
            )
            .is_err()
        );
    }
}

#[test]
fn decision_openai_optional_question_metadata_rejects_explicit_null() {
    for question in [
        json!({"type":"predicate","instructions":"safe?"}),
        json!({"type":"choice","instructions":"route?","choices":[{"value":true},{"value":"true"}]}),
        json!({"type":"score","instructions":"quality?","levels":[{"label":"low"},{"label":"high"}]}),
    ] {
        let decoded: openai::Question = serde_json::from_value(question.clone()).unwrap();
        assert_eq!(serde_json::to_value(decoded).unwrap(), question);
        for name in [json!(""), json!("duplicate")] {
            let mut named = question.clone();
            named["name"] = name;
            assert!(serde_json::from_value::<openai::Question>(named).is_ok());
        }
        let mut nullable = question;
        nullable["name"] = json!(null);
        assert!(serde_json::from_value::<openai::Question>(nullable).is_err());
    }
    for description in [json!(""), json!("description")] {
        assert!(
            serde_json::from_value::<openai::ChoiceOption>(
                json!({"value":false,"description":description})
            )
            .is_ok()
        );
        assert!(
            serde_json::from_value::<openai::ScoreLevel>(
                json!({"label":"level","description":description})
            )
            .is_ok()
        );
    }
    assert!(
        serde_json::from_value::<openai::ChoiceOption>(json!({"value":false,"description":null}))
            .is_err()
    );
    assert!(
        serde_json::from_value::<openai::ScoreLevel>(json!({"label":"level","description":null}))
            .is_err()
    );
}

#[test]
fn decision_jev_structured_text_restricts_only_top_level_json_types() {
    let structured = [
        json!("text"),
        json!({"nested":{"flag":true,"count":1,"nil":null}}),
        json!([1,true,null,{"z":2}]),
    ];
    for value in &structured {
        let typed = systemone::StructuredText::try_from(value.clone()).unwrap();
        assert_eq!(serde_json::Value::from(typed), *value);
        assert!(
            serde_json::from_value::<systemone::Request>(
                json!({"model":"m","state":value,"questions":{}})
            )
            .is_ok()
        );
        assert!(
            serde_json::from_value::<systemone::Question>(
                json!({"type":"score","criteria":[value]})
            )
            .is_ok()
        );
        assert!(serde_json::from_value::<systemone::Answer>(json!({"type":"score","score":0.0,"probabilities":{"0":1.0},"legend":{"0":value},"confidence":1.0})).is_ok());
    }
    for invalid in [json!(null), json!(1), json!(true)] {
        assert!(systemone::StructuredText::try_from(invalid.clone()).is_err());
        assert!(
            serde_json::from_value::<systemone::Request>(
                json!({"model":"m","state":invalid,"questions":{}})
            )
            .is_err()
        );
        assert!(
            serde_json::from_value::<systemone::Question>(
                json!({"type":"score","criteria":[invalid]})
            )
            .is_err()
        );
        assert!(serde_json::from_value::<systemone::Answer>(json!({"type":"score","score":0.0,"probabilities":{"0":1.0},"legend":{"0":invalid},"confidence":1.0})).is_err());
    }
    for value in structured.into_iter().chain([json!(null)]) {
        assert!(
            serde_json::from_value::<systemone::Question>(
                json!({"type":"noul","instructions":value,"criteria":{"true":value,"false":value}})
            )
            .is_ok()
        );
        assert!(
            serde_json::from_value::<systemone::Question>(
                json!({"type":"choice","instructions":value,"criteria":{"A":value}})
            )
            .is_ok()
        );
        assert!(
            serde_json::from_value::<systemone::Question>(
                json!({"type":"score","instructions":value,"criteria":["level"]})
            )
            .is_ok()
        );
    }
    for invalid in [json!(1), json!(true)] {
        for question in [
            json!({"type":"noul","instructions":invalid}),
            json!({"type":"noul","criteria":{"true":invalid}}),
            json!({"type":"choice","criteria":{"A":invalid}}),
            json!({"type":"score","instructions":invalid,"criteria":["level"]}),
        ] {
            assert!(serde_json::from_value::<systemone::Question>(question).is_err());
        }
    }
}

// Nested objects are `serde_json::Map`s, which keep their order only with
// serde_json's `preserve_order`.
#[cfg(feature = "preserve_order")]
#[test]
fn decision_jev_nested_object_order_survives_standalone_protocol_round_trip() {
    let raw = r#"{"model":"m","state":{"z":1,"a":{"last":true,"first":null}},"questions":{"z":{"type":"choice","criteria":{"B":{"z":1,"a":2},"A":null}},"a":{"type":"score","criteria":[{"z":1,"a":2}]}}}"#;
    let decoded: systemone::Request = serde_json::from_str(raw).unwrap();
    let encoded = serde_json::to_string(&decoded).unwrap();
    assert!(
        encoded.contains(r#""state":{"z":1,"a":{"last":true,"first":null}}"#),
        "{encoded}"
    );
    assert!(
        encoded.contains(r#""criteria":{"B":{"z":1,"a":2},"A":null}"#),
        "{encoded}"
    );
    assert!(
        encoded.contains(r#""criteria":[{"z":1,"a":2}]"#),
        "{encoded}"
    );
}

#[cfg(feature = "protocol-schema")]
fn schema_json<T: utoipa::PartialSchema>() -> serde_json::Value {
    serde_json::to_value(T::schema()).unwrap()
}

#[cfg(feature = "protocol-schema")]
fn assert_optional_nonnullable_property(schema: &serde_json::Value, field: &str, expected: usize) {
    fn visit<'a>(
        value: &'a serde_json::Value,
        field: &str,
        matches: &mut Vec<&'a serde_json::Value>,
    ) {
        match value {
            serde_json::Value::Object(object) => {
                if let Some(property) = object
                    .get("properties")
                    .and_then(|properties| properties.get(field))
                {
                    assert!(
                        !object
                            .get("required")
                            .and_then(serde_json::Value::as_array)
                            .is_some_and(|required| required.contains(&json!(field))),
                        "{field} must be optional: {value}"
                    );
                    matches.push(property);
                }
                for child in object.values() {
                    visit(child, field, matches);
                }
            }
            serde_json::Value::Array(values) => {
                for child in values {
                    visit(child, field, matches);
                }
            }
            _ => {}
        }
    }
    fn contains_null(value: &serde_json::Value) -> bool {
        match value {
            serde_json::Value::Object(object) => {
                object.get("nullable") == Some(&json!(true))
                    || object.get("type") == Some(&json!("null"))
                    || object
                        .get("type")
                        .and_then(serde_json::Value::as_array)
                        .is_some_and(|types| types.contains(&json!("null")))
                    || object.values().any(contains_null)
            }
            serde_json::Value::Array(values) => values.iter().any(contains_null),
            _ => false,
        }
    }
    let mut properties = Vec::new();
    visit(schema, field, &mut properties);
    assert_eq!(
        properties.len(),
        expected,
        "{field} missing from schema: {schema}"
    );
    for property in properties {
        assert!(
            !contains_null(property),
            "{field} must reject null: {property}"
        );
    }
}

#[cfg(feature = "protocol-schema")]
#[test]
fn decision_openai_optional_metadata_schema_matches_decoder() {
    assert_optional_nonnullable_property(&schema_json::<openai::InputMessage>(), "type", 1);
    assert_optional_nonnullable_property(&schema_json::<openai::Question>(), "name", 3);
    assert_optional_nonnullable_property(&schema_json::<openai::ChoiceOption>(), "description", 1);
    assert_optional_nonnullable_property(&schema_json::<openai::ScoreLevel>(), "description", 1);
}

#[test]
fn decision_flat_modules_retain_legacy_public_type_aliases() {
    let request = json!({"model":"m","input":"evidence","questions":[]});
    let flat: dynamo_protocols::types::decisions::Request =
        serde_json::from_value(request).unwrap();
    let _: openai::Request = flat;
    let request = json!({"model":"m","state":"evidence","questions":{}});
    let flat: dynamo_protocols::types::systemone::Request =
        serde_json::from_value(request).unwrap();
    let _: systemone::Request = flat;
}

#[cfg(feature = "protocol-schema")]
#[test]
fn decision_image_detail_schema_preserves_enum_and_nullability() {
    let detail = schema_json::<openai::ImageDetail>();
    assert_eq!(detail["type"], "string");
    assert_eq!(detail["enum"], json!(["low", "high", "auto", "original"]));
    let part = schema_json::<openai::InputPart>();
    let image = part["oneOf"]
        .as_array()
        .unwrap()
        .iter()
        .find(|variant| variant["properties"].get("detail").is_some())
        .unwrap();
    assert!(
        !image["required"]
            .as_array()
            .unwrap()
            .contains(&json!("detail"))
    );
    let alternatives = image["properties"]["detail"]["oneOf"].as_array().unwrap();
    assert!(
        alternatives
            .iter()
            .any(|alternative| alternative["type"] == "null")
    );
    assert!(
        alternatives
            .iter()
            .any(|alternative| alternative["$ref"]
                == "#/components/schemas/DecisionOpenAiImageDetail")
    );
}

#[cfg(feature = "protocol-schema")]
fn assert_nullable_reference(schema: &serde_json::Value, target: &str) {
    let alternatives = schema["oneOf"].as_array().unwrap();
    assert_eq!(alternatives.len(), 2, "{schema}");
    assert!(alternatives.contains(&json!({"type":"null"})), "{schema}");
    assert!(alternatives.contains(&json!({"$ref":target})), "{schema}");
}

#[cfg(feature = "protocol-schema")]
fn assert_jev_question_nullable_fields(question: &serde_json::Value) {
    let structured = "#/components/schemas/DecisionSystemOneStructuredText";
    for variant in question["oneOf"].as_array().unwrap() {
        assert_nullable_reference(&variant["properties"]["instructions"], structured);
        assert!(
            !variant["required"]
                .as_array()
                .is_some_and(|required| required.contains(&json!("instructions")))
        );
        let criteria = &variant["properties"]["criteria"];
        match criteria["type"].as_str() {
            Some("object") => {
                assert_nullable_reference(&criteria["additionalProperties"], structured)
            }
            Some("array") => assert_eq!(criteria["items"], json!({"$ref":structured})),
            _ => {
                assert_nullable_reference(
                    criteria,
                    "#/components/schemas/DecisionSystemOneNoulCriteria",
                );
                assert!(
                    !variant["required"]
                        .as_array()
                        .is_some_and(|required| required.contains(&json!("criteria")))
                );
            }
        }
    }
    let noul = schema_json::<systemone::NoulCriteria>();
    for label in ["true", "false"] {
        assert_nullable_reference(&noul["properties"][label], structured);
        assert!(
            !noul["required"]
                .as_array()
                .is_some_and(|required| required.contains(&json!(label)))
        );
    }
}

#[cfg(feature = "protocol-schema")]
#[test]
fn decision_jev_structured_text_schema_matches_top_level_decoder() {
    let structured = schema_json::<systemone::StructuredText>();
    let alternatives = structured["oneOf"].as_array().unwrap();
    assert_eq!(
        alternatives
            .iter()
            .map(|schema| &schema["type"])
            .collect::<Vec<_>>(),
        [&json!("string"), &json!("object"), &json!("array")]
    );
    assert_eq!(alternatives[2]["items"], json!({}));
    assert!(
        alternatives[1]["additionalProperties"] == json!({})
            || alternatives[1]["additionalProperties"] == json!(true)
    );
    let reference = json!({"$ref":"#/components/schemas/DecisionSystemOneStructuredText"});
    let request = schema_json::<systemone::Request>();
    assert_eq!(request["properties"]["state"], reference);
    assert!(
        request["required"]
            .as_array()
            .unwrap()
            .contains(&json!("state"))
    );
    let question = schema_json::<systemone::Question>();
    assert_jev_question_nullable_fields(&question);
    let score = question["oneOf"]
        .as_array()
        .unwrap()
        .iter()
        .find(|variant| variant["properties"]["criteria"]["type"] == "array")
        .unwrap();
    assert_eq!(score["properties"]["criteria"]["items"], reference);
    let answer = schema_json::<systemone::Answer>();
    let score = answer["oneOf"]
        .as_array()
        .unwrap()
        .iter()
        .find(|variant| variant["properties"].get("legend").is_some())
        .unwrap();
    assert_eq!(
        score["properties"]["legend"]["additionalProperties"],
        reference
    );
}
