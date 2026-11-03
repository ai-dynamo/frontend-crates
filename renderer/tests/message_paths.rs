// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_protocols::types::{ChatCompletionRequestMessage, CreateChatCompletionRequest};
use dynamo_renderer::{
    ContextMixins, OAIChatLikeRequest, OAIPromptFormatter, PromptFormatter,
    deepseek::{v4::DeepSeekV4Formatter, v32::DeepSeekV32Formatter, v41::DeepSeekV41Formatter},
    kimi_k3::KimiK3Formatter,
};
use serde_json::json;
use std::{collections::HashMap, sync::Arc};

struct Request<'a> {
    inner: &'a CreateChatCompletionRequest,
    typed: bool,
    args: &'a HashMap<String, serde_json::Value>,
}

#[test]
fn typed_and_generic_content_normalization_match() {
    let array_template = "{% for m in messages %}{% for part in m.content %}{% if part.type == 'text' %}{{ part.text }}{% elif part.type == 'image' %}<image>{% endif %}{% endfor %}{% endfor %}";
    let phi_template = "{% for m in messages %}{{ '<|' + m.role + '|>' + m.content + '<|end|>' }}{% endfor %}<|assistant|>";
    let passthrough = "{% for m in messages %}{{ m.content }}{% endfor %}";
    let mixed = json!([
        {"type": "text", "text": "Before "},
        {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}},
        {"type": "text", "text": " after"}
    ]);
    for (template, content, expected) in [
        (array_template, json!("Hello 中文"), "Hello 中文"),
        (array_template, mixed.clone(), "Before <image> after"),
        (
            phi_template,
            mixed.clone(),
            "<|user|>Before <|image_1|> after<|end|><|assistant|>",
        ),
        (passthrough, mixed, "Before  after"),
    ] {
        let PromptFormatter::OAI(formatter) = PromptFormatter::from_parts(
            serde_json::from_value(json!({"chat_template": template})).unwrap(),
            ContextMixins::new(&[]),
            true,
        )
        .unwrap();
        let inner = serde_json::from_value(json!({
            "model": "test", "messages": [{"role": "user", "content": content}]
        }))
        .unwrap();
        let args = HashMap::new();
        let typed = Request {
            inner: &inner,
            typed: true,
            args: &args,
        };
        let generic = Request {
            typed: false,
            ..typed
        };
        let actual = formatter.render(&typed).unwrap();
        assert_eq!(actual, expected);
        assert_eq!(formatter.render(&generic).unwrap(), actual);
    }
}

impl OAIChatLikeRequest for Request<'_> {
    fn model(&self) -> String {
        self.inner.model()
    }
    fn messages(&self) -> minijinja::Value {
        assert!(!self.typed, "typed messages must skip MiniJinja conversion");
        self.inner.messages()
    }
    fn typed_messages(&self) -> Option<&[ChatCompletionRequestMessage]> {
        self.typed.then_some(self.inner.messages.as_slice())
    }
    fn tools(&self) -> Option<minijinja::Value> {
        self.inner.tools()
    }
    fn tool_choice(&self) -> Option<minijinja::Value> {
        self.inner.tool_choice()
    }
    fn response_format(&self) -> Option<minijinja::Value> {
        self.inner.response_format()
    }
    fn should_add_generation_prompt(&self) -> bool {
        true
    }
    fn chat_template_args(&self) -> Option<&HashMap<String, serde_json::Value>> {
        Some(self.args)
    }
}

#[test]
fn typed_and_generic_message_paths_match() {
    // Render the whole normalized context so field loss and changes in object
    // ordering are visible as well as ordinary message content.
    let PromptFormatter::OAI(jinja) = PromptFormatter::from_parts(
        serde_json::from_value(json!({
            "chat_template": "{{ messages | tojson }}{{ tools | tojson }}"
        }))
        .unwrap(),
        ContextMixins::new(&[]),
        true,
    )
    .unwrap();
    let formatters: Vec<(&str, Arc<dyn OAIPromptFormatter>, &[bool])> = vec![
        ("jinja", jinja, &[false]),
        (
            "v3.2",
            Arc::new(DeepSeekV32Formatter::new_thinking()),
            &[false, true],
        ),
        (
            "v4",
            Arc::new(DeepSeekV4Formatter::new_thinking()),
            &[false, true],
        ),
        ("v4.1", Arc::new(DeepSeekV41Formatter), &[false, true]),
        ("kimi", Arc::new(KimiK3Formatter::new(true)), &[false, true]),
    ];
    let requests = [
        json!({"model": "test", "messages": [
            {"role": "system", "content": "Keep Unicode 中文 🦀 and <special> text."},
            {"role": "user", "content": "  unchanged\n"}
        ]}),
        json!({"model": "test", "messages": [
            {"role": "user", "content": [{"type": "text", "text": "東京"}, {"type": "text", "text": "weather?"}]},
            {"role": "assistant", "content": null, "reasoning_content": "check",
             "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "weather", "arguments": "{\"city\":\"東京\"}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "sunny"},
            {"role": "user", "content": "explain"}
        ], "tools": [{"type": "function", "function": {"name": "weather", "parameters": {}}}]}),
    ];
    for body in requests {
        let inner: CreateChatCompletionRequest = serde_json::from_value(body).unwrap();
        for (name, formatter, thinking_values) in &formatters {
            for &thinking in *thinking_values {
                let args = HashMap::from([("thinking".into(), json!(thinking))]);
                let typed = Request {
                    inner: &inner,
                    typed: true,
                    args: &args,
                };
                let generic = Request {
                    typed: false,
                    ..typed
                };
                // RenderedPrompt equality also checks Kimi's segment trust flags.
                assert_eq!(
                    formatter.render_prompt(&typed).unwrap(),
                    formatter.render_prompt(&generic).unwrap(),
                    "{name}, thinking={thinking}"
                );
            }
        }
    }
}

// Representative consumer-owned Phi-style template with both string and array
// branches. Marker spellings are model-specific, never inferred by Dynamo.
const CUSTOM_TEMPLATE: &str = r"{%- for message in messages -%}
{{- '<|' ~ message.role ~ '|>' -}}
{%- if message.name is defined -%}{{- '[' ~ message.name ~ ']' -}}{%- endif -%}
{%- if message.tool_call_id is defined -%}{{- '[' ~ message.tool_call_id ~ ']' -}}{%- endif -%}
{%- if message.content is string -%}{{- message.content -}}
{%- else -%}{%- for part in message.content -%}
{%- if part.type == 'text' -%}{{- part.text -}}
{%- elif part.type == 'custom' -%}
{%- if part.fields is defined -%}{{- raise_exception('custom payload reached template') -}}{%- endif -%}
{%- if part.kind == 'chemistry' -%}{{- '<|chem_projected|>' -}}
{%- elif part.kind == 'biology' -%}{{- '<|bio_projected|>' -}}{%- endif -%}
{%- elif part.type == 'image' -%}{{- '<|image_1|>' -}}
{%- elif part.type == 'audio' -%}{{- '<|audio|>' -}}
{%- elif part.type == 'video' -%}{{- '<|video|>' -}}{%- endif -%}
{%- endfor -%}{%- endif -%}{{- '<|end|>' -}}
{%- endfor -%}
{%- if add_generation_prompt -%}{{- '<|assistant|>' -}}{%- endif -%}";

struct PlaceholderRequest(serde_json::Value);
impl OAIChatLikeRequest for PlaceholderRequest {
    fn model(&self) -> String {
        "custom-model".into()
    }
    fn messages(&self) -> minijinja::Value {
        minijinja::Value::from_serialize(&self.0)
    }
    fn tools(&self) -> Option<minijinja::Value> {
        None
    }
    fn tool_choice(&self) -> Option<minijinja::Value> {
        None
    }
    fn response_format(&self) -> Option<minijinja::Value> {
        None
    }
    fn should_add_generation_prompt(&self) -> bool {
        true
    }
}

#[test]
fn custom_parts_keep_positions_in_typed_and_payload_free_template_paths() {
    let PromptFormatter::OAI(formatter) = PromptFormatter::from_parts(
        serde_json::from_value(json!({"chat_template": CUSTOM_TEMPLATE})).unwrap(),
        ContextMixins::new(&[]),
        true,
    )
    .unwrap();
    let args = HashMap::new();
    for role in ["user", "tool"] {
        let mut message = json!({"role":role, "content":[
            {"type":"text", "text":"Does this structure "},
            {"type":"custom", "kind":"chemistry", "fields":{"data_base64":"SECRET_ONE"}},
            {"type":"text", "text":" match the diagram?"},
            {"type":"image_url", "image_url":{"url":"https://example.org/image.png"}},
            {"type":"custom", "kind":"biology", "fields":{"metadata":"SECRET_BIO"}},
            {"type":"custom", "kind":"chemistry", "fields":{"data_base64":"SECRET_TWO"}},
            {"type":"audio_url", "audio_url":{"url":"https://example.org/audio.wav"}},
            {"type":"video_url", "video_url":{"url":"https://example.org/video.mp4"}}
        ]});
        let metadata = if role == "user" {
            message["name"] = json!("alice");
            "alice"
        } else {
            message["tool_call_id"] = json!("call_7");
            "call_7"
        };
        let inner: CreateChatCompletionRequest = serde_json::from_value(json!({
            "model":"custom-model", "messages":[message.clone()]
        }))
        .unwrap();
        let expected = format!(
            "<|{role}|>[{metadata}]Does this structure <|chem_projected|> match the diagram?<|image_1|><|bio_projected|><|chem_projected|><|audio|><|video|><|end|><|assistant|>"
        );
        for typed in [false, true] {
            let request = Request {
                inner: &inner,
                typed,
                args: &args,
            };
            assert_eq!(formatter.render(&request).unwrap(), expected);
        }
        // Dynamo's template projection intentionally cannot deserialize as the
        // HTTP Custom type: fields is required on HTTP, absent in template input.
        for part in message["content"].as_array_mut().unwrap() {
            if part["type"] == "custom" {
                part.as_object_mut().unwrap().remove("fields");
            }
        }
        assert_eq!(
            formatter
                .render(&PlaceholderRequest(json!([message])))
                .unwrap(),
            expected
        );
    }
}
