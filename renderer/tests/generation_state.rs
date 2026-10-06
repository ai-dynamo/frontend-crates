// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use dynamo_renderer::{
    ChatTemplate, ContextMixins, GenerationState, NoOpFormatter, OAIChatLikeRequest,
    OAIPromptFormatter, PromptFormatter, RenderedPrompt, RenderedSegment,
    deepseek::{v4::DeepSeekV4Formatter, v32::DeepSeekV32Formatter, v41::DeepSeekV41Formatter},
    inkling::InklingFormatter,
    kimi_k3::KimiK3Formatter,
};
use minijinja::Value as MiniValue;
use serde_json::{Value, json};

struct Request {
    messages: Value,
    args: HashMap<String, Value>,
    generation_prompt: bool,
    effort: Option<Value>,
    tools: Option<Value>,
    choice: Option<Value>,
    format: Option<Value>,
}

impl Request {
    fn new(messages: Value) -> Self {
        Self {
            messages,
            args: HashMap::new(),
            generation_prompt: true,
            effort: None,
            tools: None,
            choice: None,
            format: None,
        }
    }
}

impl OAIChatLikeRequest for Request {
    fn model(&self) -> String {
        "test".into()
    }
    fn messages(&self) -> MiniValue {
        MiniValue::from_serialize(&self.messages)
    }
    fn should_add_generation_prompt(&self) -> bool {
        self.generation_prompt
    }
    fn chat_template_args(&self) -> Option<&HashMap<String, Value>> {
        Some(&self.args)
    }
    fn reasoning_effort(&self) -> Option<MiniValue> {
        self.effort.as_ref().map(MiniValue::from_serialize)
    }
    fn tools(&self) -> Option<MiniValue> {
        self.tools.as_ref().map(MiniValue::from_serialize)
    }
    fn tool_choice(&self) -> Option<MiniValue> {
        self.choice.as_ref().map(MiniValue::from_serialize)
    }
    fn response_format(&self) -> Option<MiniValue> {
        self.format.as_ref().map(MiniValue::from_serialize)
    }
}

fn user_request() -> Request {
    Request::new(json!([{"role":"user", "content":"Hello"}]))
}

fn deepseek() -> Vec<Box<dyn OAIPromptFormatter>> {
    vec![
        Box::new(DeepSeekV32Formatter::new_thinking()),
        Box::new(DeepSeekV4Formatter::new_thinking()),
        Box::new(DeepSeekV41Formatter),
    ]
}

fn check(
    formatter: &dyn OAIPromptFormatter,
    request: &Request,
    state: GenerationState,
) -> RenderedPrompt {
    let prompt = formatter.render_prompt(request).unwrap();
    assert_eq!(prompt.generation_state(), state, "{}", prompt.as_str());
    assert_eq!(formatter.render(request).unwrap(), prompt.as_str());
    prompt
}

#[test]
fn existing_constructors_and_custom_formatters_make_no_state_claim() {
    let segments = vec![RenderedSegment::new("<think>", false)];
    for prompt in [
        RenderedPrompt::text("<think>".into()),
        RenderedPrompt::segmented(segments.clone()),
    ] {
        assert_eq!(prompt.generation_state(), GenerationState::Unknown);
        let known = prompt
            .clone()
            .with_generation_state(GenerationState::Reasoning);
        assert_ne!(prompt, known); // Equality includes the continuation contract.
        assert_eq!(prompt.as_str(), known.as_str());
        assert_eq!(prompt.segments(), known.segments());
        assert_eq!(
            known.with_generation_state(GenerationState::Unknown),
            prompt
        );
    }
    let request = Request::new(json!([{"role":"user", "content":"<think>"}]));
    check(&NoOpFormatter, &request, GenerationState::Unknown);
    let config: ChatTemplate =
        serde_json::from_value(json!({"chat_template":"{{ messages[0].content }}"})).unwrap();
    let PromptFormatter::OAI(formatter) =
        PromptFormatter::from_parts(config, ContextMixins::default(), false).unwrap();
    check(formatter.as_ref(), &request, GenerationState::Unknown);
}

#[test]
fn deepseek_reports_the_emitted_transition_even_without_generation_prompt_flag() {
    for formatter in deepseek() {
        let mut request = user_request();
        for add in [true, false] {
            request.generation_prompt = add;
            for thinking in [true, false] {
                request.args.insert("thinking".into(), json!(thinking));
                let state = if thinking {
                    GenerationState::Reasoning
                } else {
                    GenerationState::Response
                };
                let prompt = check(formatter.as_ref(), &request, state);
                assert!(
                    prompt
                        .as_str()
                        .ends_with(if thinking { "<think>" } else { "</think>" })
                );
            }
        }
    }
}

#[test]
fn top_level_effort_overrides_thinking_for_v4_and_v41() {
    for formatter in [
        Box::new(DeepSeekV4Formatter::new_thinking()) as Box<dyn OAIPromptFormatter>,
        Box::new(DeepSeekV41Formatter),
    ] {
        let mut request = user_request();
        request.args.insert("thinking".into(), json!(true));
        request
            .args
            .insert("reasoning_effort".into(), json!("high"));
        request.effort = Some(json!("none"));
        let prompt = check(formatter.as_ref(), &request, GenerationState::Response);
        assert!(prompt.as_str().ends_with("</think>"));
    }
}

#[test]
fn deepseek_closed_turns_and_unsupported_endings_are_unknown() {
    for formatter in deepseek() {
        for messages in [
            json!([]),
            json!([{"role":"system", "content":"<think>"}]),
            json!([{"role":"user", "content":"Hello"}, {"role":"assistant", "content":"Done", "reasoning_content":"Thought"}]),
        ] {
            check(
                formatter.as_ref(),
                &Request::new(messages),
                GenerationState::Unknown,
            );
        }
    }
}

#[test]
fn deepseek_tool_results_reopen_the_channel() {
    for formatter in deepseek() {
        let mut request = Request::new(json!([
            {"role":"user", "content":"Weather?"},
            {"role":"assistant", "content":"", "reasoning_content":"Check weather", "tool_calls":[{"id":"call_1", "type":"function", "function":{"name":"weather", "arguments":"{}"}}]},
            {"role":"tool", "tool_call_id":"call_1", "content":"Sunny"}
        ]));
        check(formatter.as_ref(), &request, GenerationState::Reasoning);
        request.args.insert("thinking".into(), json!(false));
        check(formatter.as_ref(), &request, GenerationState::Response);
    }
}

#[test]
fn deepseek_v32_incomplete_tool_results_are_unknown() {
    let request = Request::new(json!([
        {"role":"user", "content":"Weather?"},
        {"role":"assistant", "tool_calls":[
            {"id":"a", "type":"function", "function":{"name":"weather", "arguments":"{}"}},
            {"id":"b", "type":"function", "function":{"name":"weather", "arguments":"{}"}}
        ]},
        {"role":"tool", "tool_call_id":"a", "content":"Sunny"}
    ]));
    check(
        &DeepSeekV32Formatter::new_thinking(),
        &request,
        GenerationState::Unknown,
    );
}

#[test]
fn kimi_channels_and_partial_prefix_preserve_segments() {
    let formatter = KimiK3Formatter::new(false);
    let mut request = user_request();
    for thinking in [true, false] {
        request.args.insert("thinking".into(), json!(thinking));
        let prompt = check(
            &formatter,
            &request,
            if thinking {
                GenerationState::Reasoning
            } else {
                GenerationState::Response
            },
        );
        assert!(prompt.segments().is_some());
        assert!(prompt.as_str().ends_with(if thinking {
            "<|open|>think<|sep|>"
        } else {
            "<|open|>response<|sep|>"
        }));
    }
    request.args.insert("thinking".into(), json!(true));
    request.messages = json!([{"role":"user", "content":"Continue"}, {"role":"assistant", "content":"<|open|>think<|sep|>", "partial":true}]);
    for add in [true, false] {
        request.generation_prompt = add;
        let prompt = check(&formatter, &request, GenerationState::Response);
        let last = prompt.segments().unwrap().last().unwrap();
        assert_eq!(last.text, "<|open|>think<|sep|>");
        assert!(!last.allow_special);
    }
    request = user_request();
    request.generation_prompt = false;
    check(&formatter, &request, GenerationState::Unknown);
}

#[test]
fn kimi_named_tools_and_response_schema_use_actual_channel() {
    let formatter = KimiK3Formatter::new(false);
    let mut request = user_request();
    request.format = Some(
        json!({"type":"json_schema", "json_schema":{"name":"answer", "schema":{"type":"object"}}}),
    );
    check(&formatter, &request, GenerationState::Reasoning);
    request.tools = Some(
        json!([{"type":"function", "function":{"name":"weather", "parameters":{"type":"object"}}}]),
    );
    request.choice = Some(json!({"type":"function", "function":{"name":"weather"}}));
    check(&formatter, &request, GenerationState::Response);
}

#[test]
fn inkling_opens_an_assistant_without_selecting_a_channel() {
    let mut request = user_request();
    let prompt = check(&InklingFormatter, &request, GenerationState::Unopened);
    assert!(prompt.as_str().ends_with("<|message_model|>"));
    request.generation_prompt = false;
    check(&InklingFormatter, &request, GenerationState::Unknown);
}

#[test]
fn kimi_partial_response_metadata_initializes_unified_parser() {
    use dynamo_parsers_v2::{
        UnifiedEvent, UnifiedParserExt, UnifiedParserInit, UnifiedParserStartingState, assemble,
        create_unified_parser_for_family,
    };

    let formatter = KimiK3Formatter::new(false);
    let mut request = Request::new(json!([
        {"role": "user", "content": "Check the weather in Paris."},
        {"role": "assistant", "content": "I will ", "partial": true}
    ]));
    request.args.insert("thinking".into(), json!(true));
    request.tools = Some(json!([{
        "type": "function",
        "function": {
            "name": "weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"]
            }
        }
    }]));

    // Only generated bytes reach the parser. The response opening and the
    // partial text are already in the prompt, even though thinking is enabled.
    let completion = concat!(
        "check Paris.",
        "<|close|>response<|sep|>",
        "<|open|>tools<|sep|>",
        "<|open|>call tool=\"weather\" index=\"0\"<|sep|>",
        "<|open|>argument key=\"city\" type=\"string\"<|sep|>Paris",
        "<|close|>argument<|sep|>",
        "<|close|>call<|sep|>",
        "<|close|>tools<|sep|>",
        "<|close|>message<|sep|><|end_of_msg|>"
    );
    let expected = vec![
        UnifiedEvent::Text {
            text: "check Paris.".into(),
        },
        UnifiedEvent::ToolCall {
            name: "weather".into(),
            arguments: json!({"city": "Paris"}),
        },
    ];

    for add_generation_prompt in [true, false] {
        request.generation_prompt = add_generation_prompt;
        let prompt = formatter.render_prompt(&request).unwrap();
        assert!(prompt.as_str().ends_with("<|open|>response<|sep|>I will "));
        let starting_state = match prompt.generation_state() {
            GenerationState::Unopened => UnifiedParserStartingState::None,
            GenerationState::Reasoning => UnifiedParserStartingState::Reasoning,
            GenerationState::Response => UnifiedParserStartingState::Response,
            other => panic!("expected known native continuation state, got {other:?}"),
        };
        let init = UnifiedParserInit {
            starting_state,
            ..Default::default()
        };
        let mut parser = create_unified_parser_for_family("kimi_k3", &[]).unwrap();

        // Whole completion and every two-chunk split share the same contract.
        // Reinitialize the same parser to check request-state restoration too.
        for split in completion.char_indices().map(|(at, _)| at) {
            parser.reset();
            parser.initialize_request(init.clone()).unwrap();
            let mut events = parser.push(&completion[..split]).unwrap();
            events.extend(parser.push(&completion[split..]).unwrap());
            events.extend(parser.finish().unwrap().events);
            assert_eq!(
                assemble(&events),
                expected,
                "split={split}, add_generation_prompt={add_generation_prompt}"
            );
        }
        parser.reset();
        parser.initialize_request(init).unwrap();
        let mut events = Vec::new();
        for ch in completion.chars() {
            events.extend(parser.push(&ch.to_string()).unwrap());
        }
        events.extend(parser.finish().unwrap().events);
        assert_eq!(assemble(&events), expected, "one character per chunk");
    }
}
