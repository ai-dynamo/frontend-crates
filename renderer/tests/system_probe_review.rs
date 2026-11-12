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

#[test]
fn turn_numbers_do_not_hide_a_dropped_system_message() {
    let template = concat!(
        "{%- for m in messages -%}",
        "{%- if m.role == 'system' and loop.first -%}",
        "<system>{{ m.content }}</system>",
        "{%- elif m.role in ['user', 'assistant'] -%}",
        "[{{ loop.index0 }}]<{{ m.role }}>{{ m.content }}</{{ m.role }}>",
        "{%- endif -%}",
        "{%- endfor -%}"
    );
    let messages = json!([
        {"role": "system", "content": "Be helpful"},
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "answer"},
        {"role": "system", "content": "Be concise"},
        {"role": "user", "content": "next question"},
    ]);
    let raw = minijinja::Environment::new()
        .render_str(template, minijinja::context! { messages => &messages })
        .unwrap();
    assert!(raw.contains("<system>Be helpful</system>"));
    assert!(!raw.contains("Be concise"));
    assert!(raw.contains("[4]<user>next question</user>"));

    let rendered = render(template, &messages);
    assert!(
        rendered.contains("[3]<user>Be concise</user>"),
        "the later system content must reach a user turn in place: {rendered}"
    );
}

#[test]
fn separator_changes_do_not_invent_system_role_support() {
    let template = concat!(
        "{%- for m in messages -%}",
        "{%- if m.role in ['user', 'assistant'] -%}",
        "{%- if not loop.first -%}{{ '\n' }}{%- endif -%}",
        "<{{ m.role }}>{{ m.content }}</{{ m.role }}>",
        "{%- endif -%}",
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
        "\n<user>question</user>",
        "\n<assistant>answer</assistant>",
        "\n<user>next question</user>"
    );
    let raw = minijinja::Environment::new()
        .render_str(template, minijinja::context! { messages => &messages })
        .unwrap();
    assert_eq!(raw, expected);
    assert_eq!(render(template, &messages), expected);
}

#[test]
fn inference_settings_missing_from_probe_do_not_hide_dropped_system() {
    use dynamo_renderer::OAIChatLikeRequest;
    use std::collections::HashMap;

    struct Request {
        messages: Value,
        args: HashMap<String, Value>,
    }

    impl OAIChatLikeRequest for Request {
        fn model(&self) -> String {
            "test".to_string()
        }

        fn messages(&self) -> minijinja::Value {
            minijinja::Value::from_serialize(&self.messages)
        }

        fn should_add_generation_prompt(&self) -> bool {
            true
        }

        fn chat_template_args(&self) -> Option<&HashMap<String, Value>> {
            Some(&self.args)
        }
    }

    let template = concat!(
        "{%- if add_generation_prompt and generation_settings is not defined -%}",
        "{{ raise_exception('generation_settings is required for inference') }}",
        "{%- endif -%}",
        "{%- for m in messages -%}",
        "{%- if loop.first or m.role != 'system' -%}",
        "<{{ m.role }}>{{ m.content }}</{{ m.role }}>",
        "{%- endif -%}",
        "{%- endfor -%}",
        "{%- if add_generation_prompt -%}",
        "<assistant mode='{{ generation_settings.mode }}'>",
        "{%- endif -%}"
    );
    let config: ChatTemplate = serde_json::from_value(json!({"chat_template": template})).unwrap();
    let PromptFormatter::OAI(formatter) =
        PromptFormatter::from_parts(config, ContextMixins::new(&[]), false).unwrap();
    let request = Request {
        messages: json!([
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
            {"role": "system", "content": "Be concise"},
            {"role": "user", "content": "next question"},
        ]),
        args: HashMap::from([("generation_settings".to_string(), json!({"mode": "chat"}))]),
    };
    let rendered = formatter.render(&request).unwrap();
    assert!(rendered.ends_with("<assistant mode='chat'>"));
    assert!(
        rendered.contains("<user>Be concise</user>"),
        "the configured inference request must keep the later system content: {rendered}"
    );
}
