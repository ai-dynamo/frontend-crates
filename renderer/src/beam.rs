// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Rendering view for Beam's `builtin_tools` template variable.
//!
//! Beam's chat template takes `chat_template_kwargs["builtin_tools"]` as the **names** of
//! tools from `tools=` to declare in the system `# Tools` block; every other tool is
//! declared in the developer block. Clients may still send a declaration object in place
//! of a name (the pre-1.0 contract, where the template rendered such objects directly):
//! it stands for its name, and when no tool of that name is declared under `tools` the
//! object itself is rendered as that tool. The result is a render-only view: the
//! request's own `tools` (validation, decoding constraints, tool policy) are never
//! modified, and names are checked against the declarations here so a typo fails the
//! request instead of silently losing the tool from the prompt.

use std::collections::HashMap;

use anyhow::{Result, bail};
use serde_json::{Value, json};

use crate::OAIChatLikeRequest;

const BUILTIN_TOOLS_KEY: &str = "builtin_tools";
/// Longest client-supplied name echoed back in an error message.
const MAX_ECHOED_NAME_CHARS: usize = 64;

/// What the template sees in place of the request's `tools` and `chat_template_args`.
pub struct BeamRendering {
    pub tools: Option<minijinja::value::Value>,
    pub chat_template_args: HashMap<String, Value>,
}

fn echo(name: &str) -> String {
    name.chars().take(MAX_ECHOED_NAME_CHARS).collect()
}

/// The name of an OpenAI tool (`{"function": {"name": ..}}`) or a flat declaration
/// (`{"name": ..}`).
fn declared_name(tool: &Value) -> Option<&str> {
    tool.get("function")
        .and_then(|function| function.get("name"))
        .or_else(|| tool.get("name"))
        .and_then(Value::as_str)
}

/// A declaration object as an entry of `tools`, which the template reads either way
/// (`tool.function if tool.function is defined else tool`).
fn as_function_tool(declaration: &Value) -> Value {
    if declaration.get("function").is_some() {
        declaration.clone()
    } else {
        json!({"type": "function", "function": declaration})
    }
}

/// Resolve `chat_template_args["builtin_tools"]` for rendering.
///
/// Returns `None` when the request has no `builtin_tools`. A name must match a declared
/// tool; a declaration object stands for its name and is appended to the rendered tools
/// when the request declares no tool of that name. Anything else is an invalid request
/// (the error message is safe to return to the client). When
/// `exclude_tools_when_tool_choice_none` is set and the request's `tool_choice` is
/// `"none"`, no builtin declarations are rendered.
pub fn builtin_tools_view<R: OAIChatLikeRequest + ?Sized>(
    request: &R,
    exclude_tools_when_tool_choice_none: bool,
) -> Result<Option<BeamRendering>> {
    let Some(args) = request.chat_template_args() else {
        return Ok(None);
    };
    let Some(builtin) = args.get(BUILTIN_TOOLS_KEY).filter(|value| !value.is_null()) else {
        return Ok(None);
    };
    let Some(entries) = builtin.as_array() else {
        bail!("`builtin_tools` must be an array of tool names or declarations");
    };

    let mut tools = match request
        .tools()
        .map(|tools| crate::python::template_json(&tools))
    {
        None | Some(Ok(Value::Null)) => Vec::new(),
        Some(Ok(Value::Array(tools))) => tools,
        Some(_) => bail!("`tools` must be an array"),
    };
    let declared_count = tools.len();

    let mut names: Vec<String> = Vec::with_capacity(entries.len());
    for entry in entries {
        let name = match entry {
            Value::String(name) => {
                if !tools
                    .iter()
                    .any(|tool| declared_name(tool) == Some(name.as_str()))
                {
                    bail!(
                        "`builtin_tools` entry `{}` is not declared in `tools`",
                        echo(name)
                    );
                }
                name.clone()
            }
            Value::Object(_) => {
                let Some(name) = declared_name(entry) else {
                    bail!("`builtin_tools` declarations must have a string `name`");
                };
                let name = name.to_string();
                // The template declares a named tool from `tools`; one the request did not
                // also declare there is rendered as given.
                if !tools
                    .iter()
                    .any(|tool| declared_name(tool) == Some(name.as_str()))
                {
                    tools.push(as_function_tool(entry));
                }
                name
            }
            _ => bail!("`builtin_tools` entries must be a tool name or a declaration object"),
        };
        if !names.contains(&name) {
            names.push(name);
        }
    }

    // The renderer omits ordinary tools for `tool_choice: "none"` under the runtime
    // policy; builtin declarations are ours to omit under the same policy.
    let tools_excluded = exclude_tools_when_tool_choice_none
        && request
            .tool_choice()
            .and_then(|choice| crate::python::template_json(&choice).ok())
            .is_some_and(|choice| choice.as_str() == Some("none"));
    if tools_excluded {
        names.clear();
        tools.truncate(declared_count);
    }

    let mut chat_template_args = args.clone();
    chat_template_args.insert(
        BUILTIN_TOOLS_KEY.to_string(),
        Value::Array(names.into_iter().map(Value::String).collect()),
    );
    let tools =
        (!tools.is_empty()).then(|| minijinja::value::Value::from_serialize(Value::Array(tools)));
    Ok(Some(BeamRendering {
        tools,
        chat_template_args,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use minijinja::value::Value as MiniValue;

    struct Request {
        tools: Option<Value>,
        tool_choice: Option<Value>,
        args: Option<HashMap<String, Value>>,
    }

    impl OAIChatLikeRequest for Request {
        fn model(&self) -> String {
            "beam".to_string()
        }

        fn messages(&self) -> MiniValue {
            MiniValue::from_serialize(json!([{"role": "user", "content": "hi"}]))
        }

        fn tools(&self) -> Option<MiniValue> {
            self.tools.as_ref().map(MiniValue::from_serialize)
        }

        fn tool_choice(&self) -> Option<MiniValue> {
            self.tool_choice.as_ref().map(MiniValue::from_serialize)
        }

        fn should_add_generation_prompt(&self) -> bool {
            true
        }

        fn chat_template_args(&self) -> Option<&HashMap<String, Value>> {
            self.args.as_ref()
        }
    }

    fn tool(name: &str, parameters: Value) -> Value {
        json!({"type": "function", "function": {
            "name": name, "description": format!("{name} tool"), "parameters": parameters
        }})
    }

    fn request(tools: Vec<Value>, builtin: Value) -> Request {
        Request {
            tools: Some(Value::Array(tools)),
            tool_choice: None,
            args: Some(HashMap::from([
                ("builtin_tools".to_string(), builtin),
                ("other".to_string(), json!(1)),
            ])),
        }
    }

    fn as_json(value: &MiniValue) -> Value {
        serde_json::to_value(value).unwrap()
    }

    fn object_schema() -> Value {
        json!({"type": "object", "properties": {"cmd": {"type": "string"}}})
    }

    #[test]
    fn names_pass_through_and_the_declared_tools_stay_whole() {
        let req = request(
            vec![
                tool("bash", json!({"type": "string"})),
                tool("weather", object_schema()),
            ],
            json!(["bash", "bash"]),
        );
        let all = as_json(&req.tools().unwrap());
        let view = builtin_tools_view(&req, true).unwrap().unwrap();
        assert_eq!(view.chat_template_args["builtin_tools"], json!(["bash"]));
        assert_eq!(view.chat_template_args["other"], json!(1));
        // The template selects the system-block tools from `tools` by name, so every
        // declared tool stays, in order.
        assert_eq!(as_json(&view.tools.unwrap()), all);
    }

    #[test]
    fn object_entry_stands_for_its_declared_tool() {
        let supplied =
            json!({"name": "bash", "description": "mine", "parameters": {"type": "string"}});
        let req = request(
            vec![
                tool("bash", object_schema()),
                tool("weather", object_schema()),
            ],
            json!([supplied, "bash"]),
        );
        let all = as_json(&req.tools().unwrap());
        let view = builtin_tools_view(&req, true).unwrap().unwrap();
        assert_eq!(view.chat_template_args["builtin_tools"], json!(["bash"]));
        // The declaration under `tools` is the one rendered; nothing is declared twice.
        assert_eq!(as_json(&view.tools.unwrap()), all);
    }

    #[test]
    fn object_entry_without_declared_tool_is_rendered_as_that_tool() {
        let supplied = json!({"name": "python", "parameters": {"type": "string"}});
        let req = request(
            vec![tool("weather", object_schema())],
            json!([supplied.clone()]),
        );
        let view = builtin_tools_view(&req, true).unwrap().unwrap();
        assert_eq!(view.chat_template_args["builtin_tools"], json!(["python"]));
        let rendered = as_json(&view.tools.unwrap());
        assert_eq!(rendered.as_array().unwrap().len(), 2);
        assert_eq!(rendered[0]["function"]["name"], json!("weather"));
        assert_eq!(
            rendered[1],
            json!({"type": "function", "function": supplied})
        );
    }

    #[test]
    fn invalid_entries_are_rejected_with_bounded_errors() {
        let long = "x".repeat(10_000);
        for builtin in [
            json!(["missing"]),
            json!([long]),
            json!([{"description": "no name"}]),
            json!([7]),
            json!("bash"),
        ] {
            let req = request(vec![tool("bash", object_schema())], builtin);
            let err = builtin_tools_view(&req, true).err().expect("must reject");
            assert!(err.to_string().len() < 300, "unbounded error: {err}");
        }
    }

    #[test]
    fn absent_or_null_builtin_tools_yield_no_view() {
        let plain = Request {
            tools: None,
            tool_choice: None,
            args: None,
        };
        assert!(builtin_tools_view(&plain, true).unwrap().is_none());
        let null = request(vec![tool("bash", object_schema())], Value::Null);
        assert!(builtin_tools_view(&null, true).unwrap().is_none());
    }

    #[test]
    fn tool_choice_none_clears_builtin_names_only_when_excluding() {
        let supplied = json!({"name": "python", "parameters": {"type": "string"}});
        for (choice, exclude, expect_declared) in [
            (Some("none"), true, false),
            (Some("none"), false, true),
            (Some("auto"), true, true),
            (Some("required"), true, true),
            (None, true, true),
        ] {
            let mut req = request(
                vec![
                    tool("bash", object_schema()),
                    tool("weather", object_schema()),
                ],
                json!(["bash", supplied.clone()]),
            );
            req.tool_choice = choice.map(|choice| json!(choice));
            let view = builtin_tools_view(&req, exclude).unwrap().unwrap();
            let names = view.chat_template_args["builtin_tools"].as_array().unwrap();
            assert_eq!(!names.is_empty(), expect_declared, "{choice:?} {exclude}");
            // The declared tools are left for the renderer's own policy; an appended
            // declaration goes with the names.
            let rendered = as_json(&view.tools.unwrap());
            assert_eq!(
                rendered.as_array().unwrap().len(),
                if expect_declared { 3 } else { 2 },
                "{choice:?} {exclude}"
            );
        }
    }
}
