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
    let formatters: Vec<(&str, Arc<dyn OAIPromptFormatter>)> = vec![
        ("jinja", jinja),
        ("v3.2", Arc::new(DeepSeekV32Formatter::new_thinking())),
        ("v4", Arc::new(DeepSeekV4Formatter::new_thinking())),
        ("v4.1", Arc::new(DeepSeekV41Formatter)),
        ("kimi", Arc::new(KimiK3Formatter::new(true))),
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
        for thinking in [false, true] {
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
            for (name, formatter) in &formatters {
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
