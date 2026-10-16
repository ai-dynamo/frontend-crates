// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

use serde::{Deserialize, Deserializer, Serialize, Serializer, de};
use serde_json::{Value, json};

use super::{NamespaceToolParamTool, Tool, ToolSearchExecutionType, ToolSearchToolParam};

/// Request-side tool definition. The pinned upstream namespace type lacks
/// `defer_loading`; keep that extension without forking hosted-tool types.
#[derive(Debug, Clone, PartialEq)]
pub struct RequestTool {
    pub inner: Tool,
    pub namespace_defer_loading: Option<bool>,
}

impl From<Tool> for RequestTool {
    fn from(inner: Tool) -> Self {
        Self {
            inner,
            namespace_defer_loading: None,
        }
    }
}

impl<'de> Deserialize<'de> for RequestTool {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let mut value = Value::deserialize(deserializer)?;
        fn normalize_custom(tool: &mut Value) {
            if tool.get("type").and_then(Value::as_str) == Some("custom")
                && tool.get("format").is_none()
            {
                tool["format"] = json!({"type":"text"});
            }
        }
        normalize_custom(&mut value);
        if value.get("type").and_then(Value::as_str) == Some("namespace") {
            if value.get("description").is_none_or(Value::is_null) {
                value["description"] = json!("");
            }
            if let Some(members) = value.get_mut("tools").and_then(Value::as_array_mut) {
                for member in members {
                    normalize_custom(member);
                }
            }
        }
        let namespace_defer_loading =
            if value.get("type").and_then(Value::as_str) == Some("namespace") {
                value
                    .get("defer_loading")
                    .cloned()
                    .map(serde_json::from_value::<Option<bool>>)
                    .transpose()
                    .map_err(de::Error::custom)?
                    .flatten()
            } else {
                None
            };
        let inner = serde_json::from_value(value).map_err(de::Error::custom)?;
        Ok(Self {
            inner,
            namespace_defer_loading,
        })
    }
}

impl Serialize for RequestTool {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut value = serde_json::to_value(&self.inner).map_err(serde::ser::Error::custom)?;
        if matches!(self.inner, Tool::Namespace(_)) {
            if let Some(flag) = self.namespace_defer_loading {
                value["defer_loading"] = Value::Bool(flag);
            }
        }
        value.serialize(serializer)
    }
}

/// Custom tools use a named string parameter in function-only chat templates.
pub const CUSTOM_TOOL_INPUT_ARG: &str = "input";

pub fn custom_tool_parameters() -> Value {
    json!({"type":"object", "properties": {CUSTOM_TOOL_INPUT_ARG: {
        "type":"string", "description":"The complete freeform input for this tool, passed through verbatim."
    }}, "required":[CUSTOM_TOOL_INPUT_ARG]})
}

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
    fn request_namespace_deferral_round_trips_without_defaulting_false() {
        for (input, expected) in [
            (None, None),
            (Some(Value::Null), None),
            (Some(json!(true)), Some(true)),
            (Some(json!(false)), Some(false)),
        ] {
            let mut value =
                json!({"type":"namespace","name":"fs","description":"Files","tools":[]});
            if let Some(flag) = input {
                value["defer_loading"] = flag;
            }
            let tool: RequestTool = serde_json::from_value(value).unwrap();
            assert_eq!(tool.namespace_defer_loading, expected);
            assert_eq!(
                serde_json::to_value(tool).unwrap().get("defer_loading"),
                expected.map(Value::Bool).as_ref()
            );
        }
    }

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

#[cfg(test)]
mod search_choice_tests {
    #[test]
    fn client_search_tool_choice_round_trips() {
        let value = serde_json::json!({"model":"/model","input":"Search.","tools":[{"type":"tool_search","execution":"client"}],"tool_choice":{"type":"tool_search"}});
        let request: super::super::CreateResponse = serde_json::from_value(value).unwrap();
        assert_eq!(
            serde_json::to_value(request).unwrap()["tool_choice"],
            serde_json::json!({"type":"tool_search"})
        );
    }
}
