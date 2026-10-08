// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_decisions::*;
use serde_json::{Value, json};

fn parse(value: Value, route: Route) -> CanonicalRequest {
    parse_request(&serde_json::to_vec(&value).unwrap(), route).unwrap()
}

#[test]
fn choice_prompt_matches_reference_format_and_ignores_id() {
    let request = parse(
        json!({"model":"jev-latest","state":"evidence","questions":{"do not render this":{"type":"choice","instructions":"pick","criteria":{"billing":"invoices","technical":null}}}}),
        Route::SystemOne,
    );
    assert_eq!(render_question_prompt(&request.input,&request.questions[0]).unwrap(),RenderedQuestion{
        content:"evidence\n\nQuestion: pick\nA: billing - invoices\nB: technical\nAnswer with the letter of one option only.".into(),labels:vec!["A".into(),"B".into()]
    });
}

#[test]
fn typed_boolean_and_string_choices_render_distinctly() {
    let request = parse(
        json!({"model":"m","input":"x","questions":[{"type":"choice","instructions":"pick","choices":[{"value":true},{"value":"true"}]}]}),
        Route::Decisions,
    );
    let rendered = render_question_prompt(&request.input, &request.questions[0]).unwrap();
    assert!(rendered.content.contains("A: true\nB: \"true\""));
}

#[test]
fn score_labels_and_descriptions_survive_and_match_jev() {
    let oai = parse(
        json!({"model":"m","input":"x","questions":[{"type":"score","instructions":"rate","levels":[{"label":"Low"},{"label":"High","description":"excellent"}]}]}),
        Route::Decisions,
    );
    let jev = parse(
        json!({"model":"jev-latest","state":"x","questions":{"q":{"type":"score","instructions":"rate","criteria":["Low","High - excellent"]}}}),
        Route::SystemOne,
    );
    assert_eq!(
        render_question_prompt(&oai.input, &oai.questions[0]).unwrap(),
        render_question_prompt(&jev.input, &jev.questions[0]).unwrap()
    );
}

#[test]
fn structured_noul_without_instructions_never_uses_id() {
    let request = parse(
        json!({"model":"jev-latest","state":{"list":["a","b"]},"questions":{"secret":{"type":"noul","instructions":null,"criteria":{"true":{"ok":true},"false":"bad"}}}}),
        Route::SystemOne,
    );
    let rendered = render_question_prompt(&request.input, &request.questions[0]).unwrap();
    assert_eq!(
        rendered.content,
        "{\"list\":[\"a\",\"b\"]}\n\nIs the following true?\nyes: {\"ok\":true}\nno: bad\nAnswer with yes or no only."
    );
}

#[test]
fn message_boundaries_and_text_parts_remain_structured() {
    let request = parse(
        json!({"model":"m","input":[{"role":"user","content":"first"},{"role":"user","content":[{"type":"input_text","text":"second"},{"type":"input_text","text":"third"}]}],"questions":[{"type":"predicate","instructions":"ok?"}]}),
        Route::Decisions,
    );
    let rendered = render_question_prompt(&request.input, &request.questions[0]).unwrap();
    assert!(
        rendered
            .content
            .starts_with("[{\"role\":\"user\",\"content\":\"first\"},{")
    );
    assert!(rendered.content.contains("\"text\":\"second\""));
    assert!(rendered.content.contains("\"text\":\"third\""));
}

#[test]
fn wide_labels_are_verified_externally_not_silently_truncated() {
    let options: Vec<_> = (0..255)
        .map(|i| json!({"value":format!("choice{i}")}))
        .collect();
    let request = parse(
        json!({"model":"m","input":"x","questions":[{"type":"choice","instructions":"pick","choices":options}]}),
        Route::Decisions,
    );
    let rendered = render_question_prompt(&request.input, &request.questions[0]).unwrap();
    assert_eq!(rendered.labels.len(), 255);
    assert_eq!(rendered.labels[0], "AA");
    assert_eq!(rendered.labels[254], "JU");
    let mut q = request.questions[0].clone();
    q.candidates.clear();
    assert!(render_question_prompt(&request.input, &q).is_err());
}
