// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use dynamo_protocols::types::responses::{RequestTool, native_tools_for_chat_template};
use dynamo_renderer::{
    ChatTemplate, ContextMixins, OAIChatLikeRequest, OAIPromptFormatter, PromptFormatter,
};
use minijinja::Value;
use serde_json::json;

const TEMPLATE: &str = include_str!("fixtures/muse_spark_open_deferred_tools.jinja");

#[test]
fn client_search_is_a_function_in_the_native_template_view() {
    let tools: Vec<RequestTool> = serde_json::from_value(serde_json::json!([
        {"type":"tool_search","execution":"client","description":"Discover tools.",
         "parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}
    ])).unwrap();
    let native = native_tools_for_chat_template(&tools);
    assert_eq!(native.len(), 1);
    assert_eq!(native[0]["function"]["name"], "tool_search");
    assert_eq!(
        native[0]["function"]["parameters"]["required"],
        serde_json::json!(["query"])
    );
}

struct Request {
    native: Vec<RequestTool>,
    args: HashMap<String, serde_json::Value>,
    none: bool,
}

impl OAIChatLikeRequest for Request {
    fn model(&self) -> String {
        "muse_spark_open".into()
    }
    fn messages(&self) -> Value {
        Value::from_serialize(json!([{"role":"user","content":"Use the appropriate tool."}]))
    }
    fn tools(&self) -> Option<Value> {
        Some(Value::from_serialize(
            json!([{"type":"function","function":{"name":"read","description":"Read","parameters":{"type":"object","properties":{}}}}]),
        ))
    }
    fn template_tools(&self) -> Option<Value> {
        Some(Value::from_serialize(native_tools_for_chat_template(
            &self.native,
        )))
    }
    fn tool_choice(&self) -> Option<Value> {
        self.none.then(|| Value::from("none"))
    }
    fn should_add_generation_prompt(&self) -> bool {
        true
    }
    fn chat_template_args(&self) -> Option<&HashMap<String, serde_json::Value>> {
        Some(&self.args)
    }
}

fn formatter(source: serde_json::Value) -> std::sync::Arc<dyn OAIPromptFormatter> {
    let source = if source.is_object() {
        json!([source])
    } else {
        source
    };
    let config: ChatTemplate = serde_json::from_value(json!({"chat_template":source})).unwrap();
    let PromptFormatter::OAI(formatter) =
        PromptFormatter::from_parts(config, ContextMixins::new(&[]), true).unwrap();
    formatter
}

fn request(tools: serde_json::Value) -> Request {
    Request {
        native: serde_json::from_value(tools).unwrap(),
        args: HashMap::from([("current_date".into(), json!("2026-10-06"))]),
        none: false,
    }
}

#[test]
fn actual_template_renders_deferred_namespace_in_both_modes() {
    let f = formatter(json!(TEMPLATE));
    let mut request = request(
        json!([{"type":"namespace","name":"fs","description":"Files","defer_loading":true,"tools":[{"type":"function","name":"read","description":"Read"}]}]),
    );
    assert!(f.supports_native_tools(&request));
    let rendered = f.render(&request).unwrap();
    assert!(rendered.contains("\"has_deferred_functions\": true"));
    assert!(!rendered.contains("\"name\": \"fs.read\""));
    request
        .args
        .insert("deferred_render_mode".into(), json!("tool_name_only"));
    let rendered = f.render(&request).unwrap();
    assert!(rendered.contains("\"name\": \"fs.read\""));
    assert!(rendered.contains("\"defer_loading\": true"));
    assert!(!rendered.contains("\"parameters\""));
}

#[test]
fn mixed_explicit_member_flags_keep_only_eager_member_schema() {
    let f = formatter(json!(TEMPLATE));
    let request = request(
        json!([{"type":"namespace","name":"fs","description":"Files","defer_loading":true,"tools":[
            {"type":"function","name":"inherited"},
            {"type":"function","name":"deferred","defer_loading":true},
            {"type":"function","name":"eager","defer_loading":false}
        ]}]),
    );
    let rendered = f.render(&request).unwrap();
    assert!(rendered.contains("\"name\": \"fs.eager\""));
    assert!(!rendered.contains("\"name\": \"fs.inherited\""));
    assert!(!rendered.contains("\"name\": \"fs.deferred\""));
    assert!(rendered.contains("\"additionalProperties\": false"));
}

#[test]
fn legacy_template_and_comment_marker_keep_flat_tools() {
    let f = formatter(json!("{# deferred_render_mode #}{{ tools | tojson }}"));
    let request = request(
        json!([{"type":"namespace","name":"fs","description":"Files","tools":[{"type":"function","name":"read"}]}]),
    );
    assert!(!f.supports_native_tools(&request));
    let rendered = f.render(&request).unwrap();
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&rendered).unwrap()[0]["function"]["name"],
        "read"
    );
    assert!(!rendered.contains("namespace"));
}

#[test]
fn capability_is_checked_on_selected_variant_and_tool_choice_none() {
    let f = formatter(json!({"default":"{{ tools | tojson }}", "tool_use":TEMPLATE}));
    let mut request = request(
        json!([{"type":"namespace","name":"fs","description":"Files","tools":[{"type":"function","name":"read"}]}]),
    );
    assert!(f.supports_native_tools(&request));
    request.none = true;
    assert!(!f.supports_native_tools(&request));
    assert_eq!(f.render(&request).unwrap(), "null");
}

#[test]
fn custom_member_uses_named_freeform_string_schema() {
    let f = formatter(json!(TEMPLATE));
    let request = request(
        json!([{"type":"namespace","name":"fs","description":"Files","tools":[{"type":"custom","name":"patch"}]}]),
    );
    let rendered = f.render(&request).unwrap();
    assert!(rendered.contains("\"name\": \"fs.patch\""));
    assert!(rendered.contains("\"required\": [\"input\"]"));
}

#[test]
fn top_level_deferred_function_is_rendered_without_its_full_schema() {
    let f = formatter(json!(TEMPLATE));
    let request = request(
        json!([{"type":"function","name":"read","description":"Read","parameters":{"type":"object","properties":{"path":{"type":"string"}}},"defer_loading":true}]),
    );
    let rendered = f.render(&request).unwrap();
    assert!(rendered.contains("\"defer_loading\": true"));
    assert!(!rendered.contains("\"parameters\""));
    assert!(rendered.contains("[deferred: use tool_search to load full schema]"));
}

#[test]
fn null_member_flags_inherit_and_false_alone_keeps_training_template_semantics() {
    let f = formatter(json!(TEMPLATE));
    let request = request(
        json!([{"type":"namespace","name":"fs","defer_loading":true,"tools":[
            {"type":"function","name":"inherited","defer_loading":null},
            {"type":"function","name":"also_inherited","defer_loading":false}
        ]}]),
    );
    let rendered = f.render(&request).unwrap();
    // The reference template carves out eager members only for mixed explicit
    // true/false flags; absent/null and false alone do not meet that condition.
    assert!(rendered.contains("\"has_deferred_functions\": true"));
    assert!(!rendered.contains("\"parameters\""));
}

#[test]
fn marker_in_literals_prose_raw_blocks_or_other_identifiers_does_not_enable_native_tools() {
    let request = request(
        json!([{"type":"namespace","name":"fs","tools":[{"type":"function","name":"read"}]}]),
    );
    for source in [
        "deferred_render_mode {{ tools | tojson }}",
        "{{ 'deferred_render_mode' }}{{ tools | tojson }}",
        "{% raw %}{{ deferred_render_mode }}{% endraw %}{{ tools | tojson }}",
        "{{ other_deferred_render_mode | default('') }}{{ tools | tojson }}",
    ] {
        assert!(
            !formatter(json!(source)).supports_native_tools(&request),
            "{source}"
        );
    }
}
