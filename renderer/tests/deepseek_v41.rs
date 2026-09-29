// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_renderer::{OAIPromptFormatter, deepseek::v41::DeepSeekV41Formatter};
use serde_json::{Value, json};

struct Request {
    request: dynamo_protocols::types::CreateChatCompletionRequest,
    args: Option<std::collections::HashMap<String, Value>>,
}

struct RawRequest {
    messages: Value,
    tools: Option<Value>,
}

impl dynamo_renderer::OAIChatLikeRequest for Request {
    fn model(&self) -> String {
        self.request.model()
    }
    fn messages(&self) -> minijinja::Value {
        self.request.messages()
    }
    fn tools(&self) -> Option<minijinja::Value> {
        self.request.tools()
    }
    fn tool_choice(&self) -> Option<minijinja::Value> {
        self.request.tool_choice()
    }
    fn response_format(&self) -> Option<minijinja::Value> {
        self.request.response_format()
    }
    fn reasoning_effort(&self) -> Option<minijinja::Value> {
        self.request.reasoning_effort()
    }
    fn should_add_generation_prompt(&self) -> bool {
        true
    }
    fn chat_template_args(&self) -> Option<&std::collections::HashMap<String, Value>> {
        self.args.as_ref()
    }
}

impl dynamo_renderer::OAIChatLikeRequest for RawRequest {
    fn model(&self) -> String {
        "deepseek-v4.1".into()
    }
    fn messages(&self) -> minijinja::Value {
        minijinja::Value::from_serialize(&self.messages)
    }
    fn tools(&self) -> Option<minijinja::Value> {
        self.tools.as_ref().map(minijinja::Value::from_serialize)
    }
    fn should_add_generation_prompt(&self) -> bool {
        true
    }
}

fn render(fields: Value) -> anyhow::Result<String> {
    let mut request =
        json!({"model":"deepseek-v4.1", "messages":[{"role":"user","content":"Hello"}]});
    request
        .as_object_mut()
        .unwrap()
        .extend(fields.as_object().unwrap().clone());
    let args = request
        .get("chat_template_args")
        .or_else(|| request.get("chat_template_kwargs"))
        .map(|value| serde_json::from_value(value.clone()))
        .transpose()?;
    let request = Request {
        request: serde_json::from_value(request)?,
        args,
    };
    DeepSeekV41Formatter.render(&request)
}

#[test]
fn api_reasoning_effort_matches_the_reference_encoder() {
    for (effort, budget) in [("low", 50), ("high", 75), ("max", 100)] {
        let output = render(json!({"reasoning_effort":effort})).unwrap();
        assert!(output.starts_with(&format!(
            "<｜begin▁of▁sentence｜><｜System｜>Reasoning Effort: {budget} "
        )));
        assert!(output.ends_with("<｜Assistant｜><think>"));
    }
    assert!(render(json!({})).unwrap().contains("Reasoning Effort: 75 "));
    assert!(render(json!({"reasoning_effort":"xhigh"})).is_err());
    assert_eq!(
        render(json!({"reasoning_effort":"none"})).unwrap(),
        "<｜begin▁of▁sentence｜><｜User｜>Hello<｜Assistant｜></think>"
    );
}

#[test]
fn template_options_set_numeric_effort_and_disable_thinking() {
    for field in ["chat_template_args", "chat_template_kwargs"] {
        assert!(
            render(json!({(field):{"reasoning_effort":37}}))
                .unwrap()
                .contains("Reasoning Effort: 37 ")
        );
        let output = render(json!({(field):{"thinking":false}})).unwrap();
        assert!(!output.contains("Reasoning Effort:"));
        assert!(output.ends_with("</think>"));
        for invalid in [
            json!(0),
            json!(101),
            json!(1.5),
            json!(true),
            json!("xhigh"),
        ] {
            assert!(render(json!({(field):{"reasoning_effort":invalid}})).is_err());
        }
    }
}

#[test]
fn unsupported_media_and_literal_image_tokens_are_rejected() {
    let block = json!({"type":"input_audio","input_audio":{"data":"AAAA","format":"wav"}});
    assert!(render(json!({"messages":[{"role":"user","content":[block]}]})).is_err());
    assert!(
        render(json!({"messages":[{"role":"user","content":"<｜deepseek_image｜>"}]})).is_err()
    );
}

#[test]
fn formatter_rejects_unsupported_partial_assistant() {
    let error = render(json!({"messages":[
        {"role":"user","content":"Continue"},
        {"role":"assistant","content":"The answer is","partial":true}
    ]}))
    .unwrap_err();

    assert!(matches!(
        error.downcast_ref::<dynamo_renderer::PromptRenderError>(),
        Some(dynamo_renderer::PromptRenderError::InvalidRequest(message))
            if message.contains("`partial: true` is not supported")
    ));
}

#[test]
fn formatter_rejects_system_tools_before_top_level_injection() {
    let error = render(json!({
        "messages":[
            {"role":"system","tools":[{
                "name":"dynamic_tool",
                "parameters":{"type":"object"}
            }]},
            {"role":"user","content":"Use a tool"}
        ],
        "tools":[{
            "type":"function",
            "function":{
                "name":"top_level_tool",
                "parameters":{"type":"object"}
            }
        }]
    }))
    .unwrap_err();

    assert!(matches!(
        error.downcast_ref::<dynamo_renderer::PromptRenderError>(),
        Some(dynamo_renderer::PromptRenderError::InvalidRequest(message))
            if message.contains("message-level `tools`") && message.contains("system")
    ));
}

#[test]
fn formatter_preserves_developer_tools_with_top_level_tools() {
    let request = RawRequest {
        messages: json!([{
            "role": "developer",
            "content": "Use a tool",
            "tools": [{"type": "function", "function": {"name": "developer_tool"}}]
        }]),
        tools: Some(json!([
            {"type": "function", "function": {"name": "top_level_tool"}}
        ])),
    };

    let output = DeepSeekV41Formatter.render(&request).unwrap();

    assert!(output.contains("developer_tool"));
    assert!(output.contains("top_level_tool"));
}

#[test]
fn mid_system_preserves_generation_header_and_reasoning_cutoff() {
    let output = render(json!({"messages":[
        {"role":"user","content":"First"},
        {"role":"assistant","content":"Reply","reasoning_content":"Old reasoning"},
        {"role":"system","content":"Next instruction"}
    ]}))
    .unwrap();
    assert!(!output.contains("Old reasoning"));
    assert!(output.ends_with("<｜System｜>Next instruction<｜Assistant｜><think>"));
    assert!(!output.contains("<think></think>"));
}

#[test]
fn tool_choice_none_omits_schema_and_keeps_response_format() {
    let output = render(
        json!({"reasoning_effort":"none", "tool_choice":"none", "tools":[
        {"type":"function","function":{"name":"weather","parameters":{"type":"object"}}}
    ],"response_format":{"type":"json_object"}}),
    )
    .unwrap();
    assert!(!output.contains("weather"));
    assert!(output.contains("## Response Format:"));
}

#[test]
fn official_python_prompt_fixtures() {
    use dynamo_renderer::deepseek::{common::ThinkingMode, v41::encode_messages};
    let cases: Vec<Value> =
        serde_json::from_str(include_str!("fixtures/deepseek_v41.json")).unwrap();
    for case in cases {
        let mode = if case["thinking"].as_bool().unwrap() {
            ThinkingMode::Thinking
        } else {
            ThinkingMode::Chat
        };
        let output = encode_messages(
            case["messages"].as_array().unwrap(),
            mode,
            case["drop_thinking"].as_bool().unwrap(),
            case["effort"].as_u64().unwrap() as u8,
        )
        .unwrap();
        assert_eq!(
            output,
            case["expected"].as_str().unwrap(),
            "{}",
            case["name"]
        );
    }
}

#[test]
fn native_detection_selects_v41_before_v4_and_honors_model_type() {
    use dynamo_renderer::{PromptFormatter, deepseek_formatter_for};
    for (model_type, name, v41) in [
        (Some("deepseek_v41"), "alias", true),
        (None, "deepseek-v4.1-flash", true),
        (Some("deepseek_v4"), "deepseek-v4.1-flash", false),
        (None, "deepseek-v4-flash", false),
    ] {
        let PromptFormatter::OAI(formatter) =
            deepseek_formatter_for(&model_type.map(str::to_owned), name).unwrap();
        let request: dynamo_protocols::types::CreateChatCompletionRequest = serde_json::from_value(
            json!({"model":name,"messages":[{"role":"user","content":"Hello"}]}),
        )
        .unwrap();
        assert_eq!(
            formatter.render(&request).unwrap().contains("<｜System｜>"),
            v41
        );
    }
    assert!(deepseek_formatter_for(&Some("llama".into()), "deepseek-v4.1").is_none());
}

#[test]
fn absent_and_empty_tools_have_identical_prompts() {
    let plain = render(json!({"reasoning_effort":"none"})).unwrap();
    assert_eq!(
        render(json!({"reasoning_effort":"none","tools":[]})).unwrap(),
        plain
    );
    assert_eq!(render(json!({"reasoning_effort":"none","tool_choice":"none","tools":[{"type":"function","function":{"name":"lookup"}}]})).unwrap(), plain);
}

#[test]
fn fixed_generation_header_is_not_advertised_as_configurable() {
    assert!(!DeepSeekV41Formatter.supports_add_generation_prompt());
}

#[test]
fn raw_encoder_rejects_explicit_tool_namespaces() {
    use dynamo_renderer::deepseek::{common::ThinkingMode, v41::encode_messages};
    for message in [
        json!({"role":"system","content":"","tools":[{"namespace":"calendar","function":{"name":"lookup"}}]}),
        json!({"role":"system","content":"","tools":[{"function":{"namespace":{"name":"calendar"},"name":"lookup"}}]}),
        json!({"role":"assistant","tool_calls":[{"namespace":"calendar","function":{"name":"lookup","arguments":"{}"}}]}),
        json!({"role":"assistant","tool_calls":[{"function":{"namespace":"calendar","name":"lookup","arguments":"{}"}}]}),
    ] {
        let error = encode_messages(&[message], ThinkingMode::Thinking, true, 50).unwrap_err();
        assert!(error.to_string().contains("namespace"));
    }
}

#[test]
fn qualified_tool_names_remain_verbatim() {
    use dynamo_renderer::deepseek::{common::ThinkingMode, v41::encode_messages};
    let output = encode_messages(
        &[json!({"role":"assistant","tool_calls":[{"function":{"name":"calendar::lookup","arguments":"{}"}}]})],
        ThinkingMode::Thinking,
        true,
        50,
    ).unwrap();
    assert!(output.contains("name=\"calendar::lookup\""));
}

#[test]
fn normalized_effort_uses_the_api_mapping() {
    for (effort, budget) in [("low", 50), ("high", 75), ("max", 100)] {
        assert!(
            render(json!({"chat_template_args": {"reasoning_effort": effort}}))
                .unwrap()
                .contains(&format!("Reasoning Effort: {budget} "))
        );
    }
    let output = render(json!({
        "reasoning_effort": "low",
        "chat_template_args": {"reasoning_effort": "max"}
    }))
    .unwrap();
    assert!(output.contains("Reasoning Effort: 50 "));
    assert!(!output.contains("Reasoning Effort: 100 "));
    let output = render(json!({"reasoning_effort": "none", "chat_template_args": {"reasoning_effort": "none", "thinking": false}})).unwrap();
    assert!(!output.contains("Reasoning Effort:"));
    assert!(output.ends_with("</think>"));
}

#[test]
fn typed_api_image_messages_match_the_reference() {
    let cases: Vec<Value> =
        serde_json::from_str(include_str!("fixtures/deepseek_v41.json")).unwrap();
    // Effort mapping is covered separately; exercise each image shape in both modes.
    for case in cases.iter().filter(|case| {
        let name = case["name"].as_str().unwrap();
        name.starts_with("image-") && (name.ends_with("-chat") || name.ends_with("-high"))
    }) {
        let effort = if case["thinking"].as_bool().unwrap() {
            "high"
        } else {
            "none"
        };
        let output =
            render(json!({"messages": case["messages"], "reasoning_effort": effort})).unwrap();
        assert_eq!(
            output,
            case["expected"].as_str().unwrap(),
            "{}",
            case["name"]
        );
    }
}

#[test]
fn image_normalization_rejects_missing_sources_and_literal_markers() {
    use dynamo_renderer::deepseek::{common::ThinkingMode, v41::encode_messages};
    for message in [
        json!({"role":"user","content":[{"type":"image_url"}]}),
        json!({"role":"user","content":[{"type":"image_url","image_url":{"url":""}}]}),
        json!({"role":"user","content":[{"type":"text","text":"<｜deepseek_image｜>"}]}),
        json!({"role":"assistant","content":"", "reasoning_content":"<｜deepseek_image｜>"}),
        json!({"role":"user","content":[{"type":"video_url","video_url":{"url":"https://example.com/v.mp4"}}]}),
    ] {
        assert!(encode_messages(&[message], ThinkingMode::Chat, true, 75).is_err());
    }
}

#[test]
fn typed_cached_image_requests_validate_sources() {
    for role in ["user", "tool"] {
        for image_url in [json!(null), json!({"url":""})] {
            let output = render(json!({"messages":[{
                "role":role,"tool_call_id":"shot1","content":[
                    {"type":"image_url","image_url":image_url,"uuid":"photo-1"}
                ]
            }],"reasoning_effort":"none"}))
            .unwrap();
            assert_eq!(output.matches("<｜deepseek_image｜>").count(), 1);
        }
    }
    for part in [
        json!({"type":"image_url","image_url":{"url":""}}),
        json!({"type":"image_url","uuid":""}),
    ] {
        assert!(render(json!({"messages":[{"role":"user","content":[part]}]})).is_err());
    }
}

#[cfg(test)]
mod media_order_tests {
    use super::*;
    use dynamo_protocols::types::ChatCompletionRequestMessage as Message;
    use serde_json::{Value, json};

    fn order(messages: &[Message]) -> Vec<usize> {
        let request: dynamo_protocols::types::CreateChatCompletionRequest =
            serde_json::from_value(json!({"model":"alias","messages":messages})).unwrap();
        DeepSeekV41Formatter.media_message_order(&request).unwrap()
    }

    fn assistant(ids: &[&str]) -> Value {
        json!({"role":"assistant","tool_calls":ids.iter().map(|id| json!({
            "id":id,"type":"function","function":{"name":"image","arguments":"{}"}
        })).collect::<Vec<_>>()})
    }

    fn tool(id: &str, parts: usize) -> Value {
        json!({"role":"tool","tool_call_id":id,"content":(0..parts).map(|n|json!({
            "type":"image_url","image_url":{"url":format!("https://example.com/{id}/{n}")},"uuid":format!("{id}-{n}")
        })).collect::<Vec<_>>()})
    }

    fn check(messages: Vec<Value>, expected: &[usize]) {
        let messages: Vec<Message> = serde_json::from_value(json!(messages)).unwrap();
        assert_eq!(order(&messages), expected);
    }

    #[test]
    fn reversed_results_keep_each_image_group_and_uuid_together() {
        check(
            vec![assistant(&["a", "b"]), tool("b", 2), tool("a", 1)],
            &[0, 2, 1],
        );
        check(
            vec![assistant(&["a", "b"]), tool("a", 1), tool("b", 2)],
            &[0, 1, 2],
        );
    }

    #[test]
    fn user_images_stay_between_sorted_tool_result_slots() {
        let user = json!({"role":"user","content":[{"type":"image_url","uuid":"user-image"}]});
        check(
            vec![assistant(&["a", "b"]), tool("b", 2), user, tool("a", 1)],
            &[0, 3, 2, 1],
        );
    }

    #[test]
    fn raw_task_groups_keep_tool_results_with_their_user_turn() {
        for (ids, task, expected) in [
            (vec!["a", "b", "c"], json!("action"), vec![0, 1, 2, 3, 5, 4]),
            (vec!["a", "b", "c"], Value::Null, vec![0, 5, 2, 3, 4, 1]),
        ] {
            let mut messages = vec![
                assistant(&ids),
                tool(ids.last().unwrap(), 1),
                json!({"role":"user","content":"TASK_USER", "task":task}),
                json!({"role":"user","content":"NEXT_USER"}),
            ];
            messages.extend(ids[..ids.len() - 1].iter().rev().map(|id| tool(id, 1)));
            for (index, message) in messages.iter_mut().enumerate() {
                if let Some(parts) = message["content"].as_array_mut() {
                    parts.insert(0, json!({"type":"text","text":format!("RESULT[{index}]")}));
                }
            }
            let request = RawRequest {
                messages: json!(messages),
                tools: None,
            };
            let prompt = DeepSeekV41Formatter.render(&request).unwrap();
            let positions: Vec<_> = expected[1..]
                .iter()
                .map(|index| {
                    let label = match index {
                        2 => "TASK_USER".to_owned(),
                        3 => "NEXT_USER".to_owned(),
                        _ => format!("RESULT[{index}]"),
                    };
                    prompt.find(&label).unwrap()
                })
                .collect();
            assert!(
                positions.windows(2).all(|pair| pair[0] < pair[1]),
                "{task}: {prompt}"
            );
            assert_eq!(
                DeepSeekV41Formatter.media_message_order(&request),
                Some(expected)
            );
        }
    }

    #[test]
    fn separate_turns_use_their_own_call_order() {
        check(
            vec![
                assistant(&["a", "b"]),
                tool("b", 1),
                tool("a", 1),
                assistant(&["d", "c"]),
                tool("c", 1),
                tool("d", 1),
            ],
            &[0, 2, 1, 3, 5, 4],
        );
    }

    #[test]
    fn system_boundaries_do_not_merge_tool_result_groups() {
        check(
            vec![
                assistant(&["a", "b"]),
                tool("b", 1),
                json!({"role":"system","content":"New instruction"}),
                tool("a", 1),
            ],
            &[0, 1, 2, 3],
        );
    }

    #[test]
    fn missing_and_unknown_call_ids_keep_reference_stable_order() {
        check(vec![tool("b", 1), tool("a", 1)], &[0, 1]);
        check(
            vec![
                assistant(&["a", "b"]),
                tool("b", 1),
                tool("unknown", 1),
                tool("a", 1),
            ],
            &[0, 2, 3, 1],
        );
    }
    #[test]
    fn collection_order_matches_actual_renderer() {
        use dynamo_renderer::{OAIPromptFormatter, deepseek::v41::DeepSeekV41Formatter};
        let ids = ["c", "a", "b"];
        let mut raw = vec![
            assistant(&["a", "b", "c"]),
            tool(ids[0], 2),
            json!({"role":"user","content":[{"type":"image_url","image_url":{"url":"https://example.com/user"}}]}),
            tool(ids[1], 1),
            tool(ids[2], 2),
        ];
        let mut slot = 0;
        let mut message_slots = vec![Vec::new(); raw.len()];
        for (i, message) in raw.iter_mut().enumerate() {
            if let Some(parts) = message["content"].as_array_mut() {
                for part in std::mem::take(parts) {
                    if part["type"] == "image_url" {
                        let label = format!("IMAGE_SLOT[{slot}]");
                        message_slots[i].push(label.clone());
                        parts.push(json!({"type":"text","text":label}));
                        slot += 1;
                    }
                    parts.push(part);
                }
            }
        }
        // Labels beside each real image expose the formatter's placeholder order.
        let req: dynamo_protocols::types::CreateChatCompletionRequest = serde_json::from_value(
            json!({"model":"alias","messages":raw,"reasoning_effort":"none"}),
        )
        .unwrap();
        let raw_request = RawRequest {
            messages: json!(raw),
            tools: None,
        };
        for request in [
            &req as &dyn dynamo_renderer::OAIChatLikeRequest,
            &raw_request,
        ] {
            let prompt = DeepSeekV41Formatter.render(request).unwrap();
            assert_eq!(prompt.matches("<｜deepseek_image｜>").count(), slot);
            let actual: Vec<_> = DeepSeekV41Formatter
                .media_message_order(request)
                .unwrap()
                .into_iter()
                .flat_map(|i| message_slots[i].iter())
                .map(|label| prompt.find(label).unwrap())
                .collect();
            assert!(
                actual.windows(2).all(|pair| pair[0] < pair[1]),
                "{ids:?}: {actual:?}"
            );
        }
    }
}

#[test]
fn media_order_defaults_to_arrival_order_without_tools() {
    let request: dynamo_protocols::types::CreateChatCompletionRequest = serde_json::from_value(
        json!({"model":"alias","messages":[{"role":"user","content":"Hello"}]}),
    )
    .unwrap();
    assert!(
        dynamo_renderer::NoOpFormatter
            .media_message_order(&request)
            .is_none()
    );
    assert!(DeepSeekV41Formatter.media_message_order(&request).is_none());
    let raw = RawRequest {
        messages: json!([{"role":"user","content":"Hello"}]),
        tools: None,
    };
    assert!(DeepSeekV41Formatter.media_message_order(&raw).is_none());
}
