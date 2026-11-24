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
        RenderedPrompt::segmented_with_pending(segments.clone(), 1),
    ] {
        assert_eq!(prompt.generation_state(), GenerationState::Unknown);
        let known = prompt
            .clone()
            .with_generation_state(GenerationState::Reasoning);
        assert_ne!(prompt, known); // Equality includes the continuation contract.
        assert_eq!(prompt.as_str(), known.as_str());
        assert_eq!(prompt.segments(), known.segments());
        assert_eq!(prompt.pending_segments(), known.pending_segments());
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
fn deepseek_reports_the_emitted_transition_with_generation_prompt_control() {
    for (formatter, honors_flag) in deepseek().into_iter().zip([false, false, true]) {
        let mut request = user_request();
        for add in [true, false] {
            request.generation_prompt = add;
            for thinking in [true, false] {
                request.args.insert("thinking".into(), json!(thinking));
                let suppressed = !add && honors_flag;
                let state = if suppressed {
                    GenerationState::Unknown
                } else if thinking {
                    GenerationState::Reasoning
                } else {
                    GenerationState::Response
                };
                let prompt = check(formatter.as_ref(), &request, state);
                if suppressed {
                    assert!(prompt.as_str().ends_with("Hello"));
                    continue;
                }
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
        assert_eq!(prompt.pending_segments(), 3);
        let pending: String = prompt
            .pending_encode_segments()
            .unwrap()
            .iter()
            .map(|segment| segment.text)
            .collect();
        assert_eq!(
            pending,
            if thinking {
                "<|open|>think<|sep|>"
            } else {
                "<|open|>response<|sep|>"
            }
        );
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
        assert_eq!(prompt.pending_segments(), 0);
        let last = prompt.segments().unwrap().last().unwrap();
        assert_eq!(last.text, "<|open|>think<|sep|>");
        assert!(!last.allow_special);
    }
    request = user_request();
    request.generation_prompt = false;
    let prompt = check(&formatter, &request, GenerationState::Unknown);
    assert_eq!(prompt.pending_segments(), 0);
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

    request.generation_prompt = false;
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
        assert_eq!(assemble(&events), expected, "split={split}");
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

#[test]
fn deepseek_v4_omits_only_the_default_effort_prefix() {
    let formatter = DeepSeekV4Formatter::new_thinking();
    let mut request = user_request();
    let default = check(&formatter, &request, GenerationState::Reasoning);
    request
        .args
        .insert("emit_default_thinking_effort".into(), json!(false));
    let omitted = check(&formatter, &request, GenerationState::Reasoning);
    assert_eq!(
        omitted.as_str(),
        "<｜begin▁of▁sentence｜><｜User｜>Hello<｜Assistant｜><think>"
    );
    request.effort = Some(json!("high"));
    assert_eq!(
        check(&formatter, &request, GenerationState::Reasoning),
        default
    );
    request.effort = Some(json!("none"));
    let chat = check(&formatter, &request, GenerationState::Response);
    assert!(chat.as_str().ends_with("</think>"));
}

#[test]
fn template_default_effort_preserves_explicit_effort_and_message_text() {
    let gpt_oss = r#"{%- macro build_system_message() -%}
        {%- if reasoning_effort is not defined %}{%- set reasoning_effort = "medium" %}{%- endif %}
        {{- "Reasoning: " + reasoning_effort + "\n\n" }}
        {%- endmacro -%}{{ build_system_message() }}"#;
    let beam = r#"{%- macro build_system_message(builtin_tool_definitions) -%}
        {%- if reasoning_effort is not defined or reasoning_effort is none %}{%- set reasoning_effort = 512 %}{%- endif %}
        {{- "\n\nReasoning effort: " + (reasoning_effort | string) }}
        {%- endmacro -%}{{ build_system_message([]) }}"#;
    let glm52 = r"[gMASK]<sop>
{%- set effective_reasoning_effort = 'high' if reasoning_effort is defined and reasoning_effort == 'high' else 'max' -%}
{%- if (enable_thinking is not defined or enable_thinking) and effective_reasoning_effort is not none -%}<|system|>Reasoning Effort: {{ effective_reasoning_effort | capitalize }}{%- endif -%}";
    let glm53 = r"[gMASK]<sop>
{%- set effective_reasoning_effort = reasoning_effort if reasoning_effort is defined and reasoning_effort in ['low', 'high'] else 'max' -%}
{%- if effective_reasoning_effort is not none -%}<|system|>Reasoning Effort: {{ effective_reasoning_effort | capitalize }}{%- endif -%}";
    let muse = r"{%- macro render_reasoning() -%}
        {%- set rs = reasoning_strength if reasoning_strength is defined and reasoning_strength else 'high' -%}
        {{- 'Reasoning strength: ' + rs + '.' -}}
        {%- endmacro -%}{{- '\n\n' -}} {{- render_reasoning() -}}";
    for (source, key, default, alternate, instruction, alternative) in [
        (
            gpt_oss,
            "reasoning_effort",
            json!("medium"),
            json!("low"),
            "Reasoning: medium\n\n",
            "Reasoning: low",
        ),
        (
            beam,
            "reasoning_effort",
            json!(512),
            json!(32),
            "\n\nReasoning effort: 512",
            "Reasoning effort: 32",
        ),
        (
            glm52,
            "reasoning_effort",
            json!("max"),
            json!("high"),
            "<|system|>Reasoning Effort: Max",
            "Reasoning Effort: High",
        ),
        (
            glm53,
            "reasoning_effort",
            json!("max"),
            json!("low"),
            "<|system|>Reasoning Effort: Max",
            "Reasoning Effort: Low",
        ),
        (
            muse,
            "reasoning_strength",
            json!("high"),
            json!("low"),
            "\n\nReasoning strength: high.",
            "Reasoning strength: low.",
        ),
    ] {
        let template = format!("{source}{{{{ messages[0].content }}}}<|assistant|>");
        let config: ChatTemplate =
            serde_json::from_value(json!({"chat_template": [
                {"name": "default", "template": template},
                {"name": "tool_use", "template": "{{ emit_default_thinking_effort }}: custom tool prompt"}
            ]})).unwrap();
        let PromptFormatter::OAI(formatter) =
            PromptFormatter::from_parts(config, ContextMixins::default(), false).unwrap();
        let mut req =
            Request::new(json!([{"role":"user","content":format!("Keep {instruction} here.")}]));
        let baseline = formatter.render(&req).unwrap();
        assert_eq!(baseline.matches(instruction).count(), 2, "{baseline:?}");
        req.args
            .insert("emit_default_thinking_effort".into(), json!(true));
        assert_eq!(formatter.render(&req).unwrap(), baseline);
        req.args
            .insert("emit_default_thinking_effort".into(), json!(false));
        assert_eq!(
            formatter.render(&req).unwrap(),
            baseline.replacen(instruction, "", 1)
        );
        req.args.insert(key.into(), default);
        assert_eq!(formatter.render(&req).unwrap(), baseline);
        req.args.insert(key.into(), alternate.clone());
        let explicit = formatter.render(&req).unwrap();
        assert!(explicit.contains(alternative), "{explicit:?}");
        req.args.remove(key);
        req.effort = Some(alternate.clone());
        assert_eq!(formatter.render(&req).unwrap(), explicit);
        req.effort = None;
        req.args.insert("reasoning_effort".into(), alternate);
        assert_eq!(formatter.render(&req).unwrap(), explicit);
        req.args.clear();
        req.args
            .insert("emit_default_thinking_effort".into(), json!("false"));
        assert!(
            formatter
                .render(&req)
                .unwrap_err()
                .to_string()
                .contains("must be a boolean")
        );
        req.tools = Some(json!([{"type":"function","function":{"name":"test","parameters":{}}}]));
        req.args
            .insert("emit_default_thinking_effort".into(), json!("custom"));
        assert_eq!(
            formatter.render(&req).unwrap(),
            "custom: custom tool prompt"
        );
    }
}
