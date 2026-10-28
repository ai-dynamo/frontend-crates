// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A request whose messages cannot be converted to JSON must fail the same
//! way with the request rules as without them, not render an empty prompt.

use dynamo_renderer::request_rules::{ChatPromptOptions, render_chat_prompt};
use dynamo_renderer::{
    ChatTemplate, ContextMixins, OAIChatLikeRequest, PromptFormatter, PromptRenderError, TextInput,
};
use minijinja::Value;

struct Request {
    invalid_metadata: bool,
}

impl OAIChatLikeRequest for Request {
    fn model(&self) -> String {
        "m".to_string()
    }

    fn messages(&self) -> Value {
        let metadata = if self.invalid_metadata {
            Value::from_serialize(std::collections::BTreeMap::from([(
                vec![1, 2],
                "invalid JSON key",
            )]))
        } else {
            Value::from_serialize(serde_json::json!({"key": "value"}))
        };
        [minijinja::context! {
            role => "user",
            content => "retain this context",
            metadata => metadata,
        }]
        .into_iter()
        .collect()
    }

    fn should_add_generation_prompt(&self) -> bool {
        false
    }

    fn extract_text(&self) -> Option<TextInput> {
        Some(TextInput::Single(String::new()))
    }
}

#[test]
fn normalization_preserves_message_serialization_error() {
    let config: ChatTemplate = serde_json::from_value(serde_json::json!({
        "chat_template": "{% for m in messages %}{{ m.content }}{% endfor %}"
    }))
    .unwrap();
    let PromptFormatter::OAI(formatter) =
        PromptFormatter::from_parts(config, ContextMixins::default(), false).unwrap();

    let render = |request: &Request, normalize_tool_call_args| {
        render_chat_prompt(
            formatter.as_ref(),
            request,
            ChatPromptOptions {
                normalize_tool_call_args,
                ..Default::default()
            },
            || None,
        )
    };
    let valid = Request {
        invalid_metadata: false,
    };
    for normalize in [false, true] {
        assert_eq!(
            render(&valid, normalize).unwrap().unwrap().as_str(),
            "retain this context"
        );
    }

    let invalid = Request {
        invalid_metadata: true,
    };
    let plain = render(&invalid, false);
    assert_eq!(
        plain,
        Err(PromptRenderError::invalid_request(
            "Failed to convert messages to JSON: key must be a string"
        ))
    );
    assert_eq!(render(&invalid, true), plain);
}
