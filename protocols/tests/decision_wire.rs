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

#[cfg(feature = "utoipa")]
#[test]
fn decision_public_schemas_use_distinct_component_names() {
    use utoipa::ToSchema;
    assert_eq!(openai::Request::name(), "DecisionOpenAiRequest");
    assert_eq!(systemone::Request::name(), "DecisionSystemOneRequest");
}
