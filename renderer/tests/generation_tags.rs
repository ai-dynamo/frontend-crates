// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_protocols::types::CreateChatCompletionRequest;
use dynamo_renderer::{ChatTemplate, ContextMixins, PromptFormatter};
use serde::Deserialize;
use serde_json::json;

#[derive(Deserialize)]
struct ReferenceCases {
    cases: Vec<ReferenceCase>,
}

#[derive(Deserialize)]
struct ReferenceCase {
    name: String,
    template: String,
    expected: String,
}

#[test]
fn generation_tags_match_huggingface_rendering() {
    // Expected prompts were rendered with Transformers 5.12.1's AssistantTracker,
    // not by deleting tags: whitespace controls and block-local variables matter.
    let reference: ReferenceCases =
        serde_json::from_str(include_str!("fixtures/generation_tags.json")).unwrap();
    let request: CreateChatCompletionRequest = serde_json::from_value(json!({
        "model": "test",
        "messages": [{"role": "user", "content": "hello"}],
    }))
    .unwrap();

    for case in reference.cases {
        let config: ChatTemplate =
            serde_json::from_value(json!({"chat_template": case.template})).unwrap();
        let formatter = PromptFormatter::from_parts(config, ContextMixins::default(), false)
            .unwrap_or_else(|error| panic!("{}: {error:#}", case.name));
        let PromptFormatter::OAI(formatter) = formatter;
        let actual = formatter
            .render(&request)
            .unwrap_or_else(|error| panic!("{}: {error:#}", case.name));
        assert_eq!(actual, case.expected, "{}", case.name);
    }
}
