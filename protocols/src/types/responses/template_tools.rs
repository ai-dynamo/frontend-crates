// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

use serde::{Deserialize, Deserializer, Serialize, Serializer, de};
use serde_json::{Value, json};

use super::Tool;

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
