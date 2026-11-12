// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_protocols::types::CreateChatCompletionRequest;
use dynamo_renderer::{ChatTemplate, ContextMixins, PromptFormatter};
use serde_json::{Value, json};

fn render(template: &str, messages: &Value) -> String {
    let config: ChatTemplate = serde_json::from_value(json!({
        "chat_template": template,
    }))
    .unwrap();
    let PromptFormatter::OAI(formatter) =
        PromptFormatter::from_parts(config, ContextMixins::new(&[]), false).unwrap();
    let request: CreateChatCompletionRequest = serde_json::from_value(json!({
        "model": "test",
        "messages": messages,
    }))
    .unwrap();
    formatter.render(&request).unwrap()
}

#[test]
fn clipped_nonleading_content_keeps_supported_system_role() {
    let template = concat!(
        "{%- for m in messages -%}",
        "<{{ m.role }}>",
        "{% if loop.first %}{{ m.content }}{% else %}{{ m.content[:20] }}{% endif %}",
        "</{{ m.role }}>",
        "{%- endfor -%}"
    );
    let messages = json!([
        {"role": "system", "content": "Be helpful"},
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "answer"},
        {"role": "system", "content": "Be concise"},
        {"role": "user", "content": "next question"},
    ]);
    let expected = concat!(
        "<system>Be helpful</system>",
        "<user>question</user>",
        "<assistant>answer</assistant>",
        "<system>Be concise</system>",
        "<user>next question</user>"
    );

    let raw = minijinja::Environment::new()
        .render_str(template, minijinja::context! { messages => &messages })
        .unwrap();
    assert_eq!(raw, expected);
    assert_eq!(render(template, &messages), expected);
}

#[test]
fn inference_only_system_turn_keeps_supported_system_role() {
    let template = concat!(
        "{%- for m in messages -%}",
        "{%- if m.role != 'system' or loop.first or add_generation_prompt -%}",
        "<{{ m.role }}>{{ m.content }}</{{ m.role }}>",
        "{%- endif -%}",
        "{%- endfor -%}",
        "{%- if add_generation_prompt -%}<assistant>{%- endif -%}"
    );
    let messages = json!([
        {"role": "system", "content": "Be helpful"},
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "answer"},
        {"role": "system", "content": "Be concise"},
        {"role": "user", "content": "next question"},
    ]);
    let expected = concat!(
        "<system>Be helpful</system>",
        "<user>question</user>",
        "<assistant>answer</assistant>",
        "<system>Be concise</system>",
        "<user>next question</user>",
        "<assistant>"
    );

    let raw = minijinja::Environment::new()
        .render_str(
            template,
            minijinja::context! { messages => &messages, add_generation_prompt => true },
        )
        .unwrap();
    assert_eq!(raw, expected);
    assert_eq!(render(template, &messages), expected);
}
