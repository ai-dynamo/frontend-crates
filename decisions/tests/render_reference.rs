// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_decisions::*;
use serde_json::{Value, json};

fn question(kind: QuestionKind, instructions: Value, description: Value) -> CanonicalQuestion {
    CanonicalQuestion {
        ordinal: 0,
        id: None,
        kind,
        instructions: Some(instructions),
        candidates: ["first", "second"]
            .into_iter()
            .map(|value| Candidate {
                value: ChoiceValue::String(value.into()),
                description: Some(description.clone()),
                label: None,
            })
            .collect(),
    }
}

// Reference: SGLang 0b635266d4a09f8db2d12bdcb793b085199faa8f,
// serving_decisions.py render_text/render_question, Python compact json.dumps.
#[test]
fn structured_prompt_values_match_pinned_python_float_spelling() {
    let input: Value = serde_json::from_str(
        r#"{"tiny":1e-7,"small":1e-5,"boundary":0.0001,"fixed":1e15,"large":1e16,"negative":-0.0,"unicode":"é","nested":[1e-6,{"x":1e20}]}"#,
    )
    .unwrap();
    let expected = r#"{"tiny":1e-07,"small":1e-05,"boundary":0.0001,"fixed":1000000000000000.0,"large":1e+16,"negative":-0.0,"unicode":"é","nested":[1e-06,{"x":1e+20}]}"#;
    for kind in [
        QuestionKind::Choice,
        QuestionKind::Predicate,
        QuestionKind::Score,
    ] {
        let rendered =
            render_question_prompt(&input, &question(kind, input.clone(), input.clone())).unwrap();
        let instruction = if kind == QuestionKind::Predicate {
            format!("Is the following true? {expected}")
        } else {
            format!("Question: {expected}")
        };
        let descriptions = match kind {
            QuestionKind::Choice => format!("A: first - {expected}\nB: second - {expected}"),
            QuestionKind::Predicate => format!("yes: {expected}\nno: {expected}"),
            QuestionKind::Score => format!("0: {expected}\n1: {expected}"),
        };
        let ending = match kind {
            QuestionKind::Choice => "Answer with the letter of one option only.",
            QuestionKind::Predicate => "Answer with yes or no only.",
            QuestionKind::Score => "Answer with the number of one level only.",
        };
        assert_eq!(
            rendered.content,
            format!("{expected}\n\n{instruction}\n{descriptions}\n{ending}")
        );
    }
}

#[test]
fn predicate_keeps_nonempty_rendered_criteria_without_blank_filtering() {
    for (value, text) in [(json!({}), "{}"), (json!([]), "[]"), (json!(" \t"), " \t")] {
        let rendered = render_question_prompt(
            &json!("state"),
            &question(QuestionKind::Predicate, json!("Check evidence"), value),
        )
        .unwrap();
        assert_eq!(
            rendered.content,
            format!(
                "state\n\nIs the following true? Check evidence\nyes: {text}\nno: {text}\nAnswer with yes or no only."
            )
        );
    }
    for value in [Value::Null, json!("")] {
        let rendered = render_question_prompt(
            &json!("state"),
            &question(QuestionKind::Predicate, json!("Check evidence"), value),
        )
        .unwrap();
        assert_eq!(
            rendered.content,
            "state\n\nIs the following true? Check evidence\nAnswer with yes or no only."
        );
    }
}

#[test]
fn only_candidate_capacity_errors_have_the_index_capacity_marker() {
    let request = CanonicalRequest {
        dialect: Dialect::Jev,
        model: "alias".into(),
        input: json!("evidence"),
        questions: vec![question(QuestionKind::Choice, json!("Choose"), Value::Null)],
        temperature: 1.0,
        prompt_format_version: 1,
        chat_template_kwargs: Default::default(),
        safety_identifier: None,
    };
    let capabilities = Capabilities {
        max_questions: 1,
        max_candidates: 2,
        supports_predicate: true,
        supports_choice: true,
        supports_score: true,
        measured_cache_reads: true,
        prompt_format_version: 1,
    };
    for (kind, count, limit) in [
        (QuestionKind::Choice, 27, 26),
        (QuestionKind::Score, 11, 10),
    ] {
        let candidate_question = question(kind, json!("Choose"), Value::Null);
        let request = CanonicalRequest {
            questions: vec![CanonicalQuestion {
                candidates: (0..count)
                    .map(|ordinal| Candidate {
                        value: ChoiceValue::String(format!("option {ordinal}")),
                        description: None,
                        label: None,
                    })
                    .collect(),
                ..candidate_question
            }],
            ..request.clone()
        };
        let error = request
            .validate_capabilities(&Capabilities {
                max_candidates: limit,
                ..capabilities.clone()
            })
            .unwrap_err();
        assert_eq!(error.status, 422);
        assert!(error.message.contains("options per choice"));
    }
    for invalid in [
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
        let error = request.validate_capabilities(&invalid).unwrap_err();
        assert!(!error.message.contains("options per choice"));
    }
    let openai = CanonicalRequest {
        dialect: Dialect::OpenAi,
        ..request
    };
    let error = openai
        .validate_capabilities(&Capabilities {
            measured_cache_reads: false,
            ..capabilities
        })
        .unwrap_err();
    assert!(!error.message.contains("options per choice"));
}

#[test]
fn prompt_label_capacity_uses_marker_only_for_overflow() {
    for (kind, count) in [(QuestionKind::Choice, 256), (QuestionKind::Score, 11)] {
        let base = question(kind, json!("Choose"), Value::Null);
        let oversized = CanonicalQuestion {
            candidates: vec![base.candidates[0].clone(); count],
            ..base
        };
        let error = render_question_prompt(&json!("state"), &oversized).unwrap_err();
        assert!(error.message.contains("options per choice"));
    }
    let empty = CanonicalQuestion {
        candidates: vec![],
        ..question(QuestionKind::Choice, json!("Choose"), Value::Null)
    };
    let error = render_question_prompt(&json!("state"), &empty).unwrap_err();
    assert!(!error.message.contains("options per choice"));
}
