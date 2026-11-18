// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

use dynamo_protocols::types::responses::{
    NamespaceToolParamTool, RequestTool, Tool, ToolSearchExecutionType, ToolSearchToolParam,
};
use serde_json::{Value, json};

pub use dynamo_protocols::types::{CUSTOM_TOOL_INPUT_ARG, custom_tool_parameters};

/// Function view for the client-owned search operation. Hosted search is not rendered.
pub fn client_tool_search_function(search: &ToolSearchToolParam) -> Option<Value> {
    (search.execution == Some(ToolSearchExecutionType::Client)).then(|| {
        json!({"type":"function", "function": {
            "name":"tool_search", "description":search.description.as_deref().unwrap_or("Search for tools and return their full definitions."),
            "parameters":search.parameters.clone().unwrap_or_else(|| json!({"type":"object","properties":{"query":{"type":"string"}},"required":["query"],"additionalProperties":false}))
        }})
    })
}

fn function(
    name: &str,
    description: Option<&str>,
    parameters: Option<&Value>,
    defer: Option<bool>,
) -> Value {
    let mut value = json!({"name":name, "description":description.unwrap_or(""),
        "parameters":parameters.cloned().unwrap_or_else(|| json!({"type":"object","properties":{}}))});
    if let Some(flag) = defer {
        value["defer_loading"] = Value::Bool(flag);
    }
    value
}

/// Native structures for templates that understand namespaces and deferral.
/// Hosted tools remain outside the function-template rendering surface.
pub fn native_tools_for_chat_template(tools: &[RequestTool]) -> Vec<Value> {
    tools.iter().filter_map(|tool| match &tool.inner {
        Tool::Function(f) => Some(json!({"type":"function", "function":
            function(&f.name, f.description.as_deref(), f.parameters.as_ref(), f.defer_loading)})),
        Tool::Custom(f) => Some(json!({"type":"function", "function":
            function(&f.name, f.description.as_deref(), Some(&custom_tool_parameters()), f.defer_loading)})),
        Tool::Namespace(ns) => {
            let members: Vec<_> = ns.tools.iter().map(|member| match member {
                NamespaceToolParamTool::Function(f) => function(&f.name,
                    f.description.as_deref().filter(|description| !description.is_empty()).or(Some(ns.description.as_str())), f.parameters.as_ref(), f.defer_loading),
                NamespaceToolParamTool::Custom(f) => function(&f.name,
                    f.description.as_deref().filter(|description| !description.is_empty()).or(Some(ns.description.as_str())), Some(&custom_tool_parameters()), f.defer_loading),
            }).collect();
            let mut value = json!({"type":"namespace", "name":ns.name, "description":ns.description, "tools":members});
            if let Some(flag) = tool.namespace_defer_loading { value["defer_loading"] = Value::Bool(flag); }
            Some(value)
        }
        Tool::ToolSearch(search) => client_tool_search_function(search),
        _ => None,
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn native_members_preserve_inheritance_and_custom_input_schema() {
        let tools: Vec<RequestTool> = serde_json::from_value(json!([{
            "type":"namespace","name":"fs","description":"Files","defer_loading":true,"tools":[
                {"type":"function","name":"read"},
                {"type":"function","name":"write","defer_loading":false},
                {"type":"custom","name":"patch","description":null,"format":{"type":"text"},"defer_loading":true}
            ]
        }])).unwrap();
        let native = native_tools_for_chat_template(&tools);
        assert_eq!(native[0]["defer_loading"], true);
        assert!(native[0]["tools"][0].get("defer_loading").is_none());
        assert_eq!(native[0]["tools"][1]["defer_loading"], false);
        assert_eq!(
            native[0]["tools"][2]["parameters"]["required"],
            json!([CUSTOM_TOOL_INPUT_ARG])
        );
        assert_eq!(native[0]["tools"][0]["name"], "read");
    }
}
