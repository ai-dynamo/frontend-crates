// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The `tool_choice` module against a copy of dynamo-llm's code at dynamo
//! `37f2f0a712`, over generated tool lists, tool choices, and requests.
//!
//! dynamo pins `dynamo-protocols` 8.4.0 and `dynamo-parsers` 10.2.2. The chat types
//! these functions read did not change from 8.4.0 to this workspace's version, so
//! the oracle runs at dynamo's pins here.
//!
//! Outputs compare as serialized JSON text, not as `Value`s: with `preserve_order`
//! the map order is part of the grammar the engine receives, and `Value` equality
//! ignores it.

use std::borrow::Cow;
use std::collections::BTreeMap;

use dynamo_chat_parsing::guided::GuidedToolConstraint;
use dynamo_chat_parsing::tool_choice::{self, ToolChoiceGuidance};
use dynamo_chat_parsing::tools;
use dynamo_protocols::types::{
    ChatCompletionTool, ChatCompletionToolChoiceOption, CreateChatCompletionRequest,
};
use serde_json::{Value, json};

/// dynamo-llm's code, copied as the oracle. `lib/llm/src/protocols/openai/tools.rs`
/// is copied whole, less its tests; each change from the stock code is marked
/// `ORACLE CHANGE`.
mod oracle {
    use super::*;
    use dynamo_chat_parsing::tools::{convert_tool_choice, requires_native_tool_call_format};
    use dynamo_protocols::types::FunctionObject;
    use thiserror::Error;

    // ---- lib/llm/src/protocols/openai/tools.rs:8-318 ----
    // ORACLE CHANGE: `pub(crate)` items are `pub`; the `use` lines above replace the
    // file's own.
    /// Errors that can occur when deriving JSON schemas for tool_choice requests.
    #[derive(Debug, Error, PartialEq, Eq)]
    pub enum ToolChoiceError {
        #[error("tool_choice requires a matching `tools` array")]
        MissingTools,
        #[error("tool `{0}` was not provided in `tools`")]
        ToolNotFound(String),
        #[error("$defs for tool `{0}` must be an object")]
        InvalidDefinitionMap(String),
        #[error("tool_choice `required` needs at least one tool definition")]
        EmptyTools,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum ToolChoiceValidation<'a> {
        Unforced,
        Required,
        Named(&'a str),
    }

    /// The guided-decoding grammar selected for a forced tool choice.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub enum ToolChoiceGuidance {
        Json(Value),
        Regex(String),
    }

    /// Validate the forced-choice contract shared by wire and parser-facing tool types.
    pub fn validate_tool_choice_against_names<'a>(
        tool_choice: ToolChoiceValidation<'_>,
        tool_names: impl IntoIterator<Item = &'a str>,
    ) -> Result<(), ToolChoiceError> {
        let mut tool_names = tool_names.into_iter();
        match tool_choice {
            ToolChoiceValidation::Unforced => Ok(()),
            ToolChoiceValidation::Required if tool_names.next().is_none() => {
                Err(ToolChoiceError::EmptyTools)
            }
            ToolChoiceValidation::Named(name) if !tool_names.any(|tool_name| tool_name == name) => {
                Err(ToolChoiceError::ToolNotFound(name.to_string()))
            }
            ToolChoiceValidation::Required | ToolChoiceValidation::Named(_) => Ok(()),
        }
    }

    /// Validate an OpenAI `tool_choice` against the request's declared tools.
    pub fn validate_openai_tool_choice(
        tool_choice: Option<&ChatCompletionToolChoiceOption>,
        tools: Option<&[ChatCompletionTool]>,
    ) -> Result<(), ToolChoiceError> {
        let Some(tool_choice) = tool_choice else {
            return Ok(());
        };

        match tool_choice {
            ChatCompletionToolChoiceOption::None | ChatCompletionToolChoiceOption::Auto => Ok(()),
            ChatCompletionToolChoiceOption::Required => {
                let tools = tools.ok_or(ToolChoiceError::MissingTools)?;
                validate_tool_choice_against_names(
                    ToolChoiceValidation::Required,
                    tools.iter().map(|tool| tool.function.name.as_str()),
                )
            }
            ChatCompletionToolChoiceOption::Named(named) => {
                let tools = tools.ok_or(ToolChoiceError::MissingTools)?;
                validate_tool_choice_against_names(
                    ToolChoiceValidation::Named(&named.function.name),
                    tools.iter().map(|tool| tool.function.name.as_str()),
                )
            }
        }
    }

    /// Builds the guided-decoding grammar enforced for the given tool_choice/tools pair.
    pub fn get_tool_choice_guidance_from_tools(
        tool_choice: Option<&ChatCompletionToolChoiceOption>,
        tools: Option<&[ChatCompletionTool]>,
        parallel_tool_calls: Option<bool>,
    ) -> Result<Option<ToolChoiceGuidance>, ToolChoiceError> {
        let Some(choice) = tool_choice else {
            return Ok(None);
        };
        validate_openai_tool_choice(Some(choice), tools)?;

        match choice {
            ChatCompletionToolChoiceOption::None | ChatCompletionToolChoiceOption::Auto => Ok(None),
            ChatCompletionToolChoiceOption::Named(named) => {
                let tools = tools.ok_or(ToolChoiceError::MissingTools)?;
                let tool = find_tool(tools, &named.function.name)
                    .ok_or_else(|| ToolChoiceError::ToolNotFound(named.function.name.clone()))?;
                let parameters = clone_parameters(&tool.function);
                if admits_only_empty_object(&parameters) {
                    // JSON schemas allow whitespace between tokens. For the single-value `{}`
                    // schema, greedy decoding can therefore emit whitespace until max_tokens.
                    // Keep the named-tool constraint and make the only legal argument value exact.
                    return Ok(Some(ToolChoiceGuidance::Regex(r"\{\}".to_string())));
                }
                Ok(Some(ToolChoiceGuidance::Json(parameters)))
            }
            ChatCompletionToolChoiceOption::Required => {
                let tools = tools.ok_or(ToolChoiceError::MissingTools)?;
                build_required_schema(tools, parallel_tool_calls)
                    .map(ToolChoiceGuidance::Json)
                    .map(Some)
            }
        }
    }

    /// Builds the JSON-schema branch of the guided-decoding grammar.
    ///
    /// Callers that install guided decoding should use `get_tool_choice_guidance_from_tools`
    /// so named zero-argument tools retain their exact regex constraint.
    pub fn get_json_schema_from_tools(
        tool_choice: Option<&ChatCompletionToolChoiceOption>,
        tools: Option<&[ChatCompletionTool]>,
        parallel_tool_calls: Option<bool>,
    ) -> Result<Option<Value>, ToolChoiceError> {
        Ok(
            get_tool_choice_guidance_from_tools(tool_choice, tools, parallel_tool_calls)?.and_then(
                |guidance| match guidance {
                    ToolChoiceGuidance::Json(schema) => Some(schema),
                    ToolChoiceGuidance::Regex(_) => None,
                },
            ),
        )
    }

    fn find_tool<'a>(
        tools: &'a [ChatCompletionTool],
        name: &str,
    ) -> Option<&'a ChatCompletionTool> {
        tools.iter().find(|tool| tool.function.name == name)
    }

    /// True when `schema` admits exactly one document, the empty object `{}`.
    ///
    /// Only the fully closed, property-less object qualifies. Leaving `additionalProperties`
    /// unset admits other documents, and a schema with any property gives the grammar a
    /// required key to emit, so neither can stall. The keyword allowlist keeps an unfamiliar
    /// constraint from being read as "empty".
    fn admits_only_empty_object(schema: &Value) -> bool {
        let Value::Object(map) = schema else {
            return false;
        };
        if map.get("type").and_then(Value::as_str) != Some("object") {
            return false;
        }
        if map.get("additionalProperties") != Some(&Value::Bool(false)) {
            return false;
        }
        let properties_empty = match map.get("properties") {
            None => true,
            Some(Value::Object(properties)) => properties.is_empty(),
            Some(_) => false,
        };
        if !properties_empty {
            return false;
        }
        let required_empty = map
            .get("required")
            .is_none_or(|required| required.as_array().is_some_and(|list| list.is_empty()));
        if !required_empty {
            return false;
        }
        map.iter().all(|(key, value)| {
            matches!(
                key.as_str(),
                "type"
                    | "properties"
                    | "required"
                    | "additionalProperties"
                    | "title"
                    | "description"
                    | "$comment"
                    | "default"
                    | "deprecated"
                    | "examples"
                    | "readOnly"
                    | "writeOnly"
            ) || key == "minProperties" && value.as_u64() == Some(0)
                || key == "maxProperties" && value.as_u64().is_some()
        })
    }

    fn clone_parameters(function: &FunctionObject) -> Value {
        function
            .parameters
            .clone()
            .unwrap_or_else(|| json!({"type": "object", "properties": {}}))
    }

    /// Build an array of tool calls, preserving each parameter document's local definitions.
    /// Schemas with local references or `$defs` keep their own document root after nesting.
    fn build_required_schema(
        tools: &[ChatCompletionTool],
        parallel_tool_calls: Option<bool>,
    ) -> Result<Value, ToolChoiceError> {
        let mut defs = serde_json::Map::new();
        let mut any_of = Vec::with_capacity(tools.len());

        for (index, tool) in tools.iter().enumerate() {
            let mut schema = clone_parameters(&tool.function);
            if schema.get("$defs").is_some_and(|defs| !defs.is_object()) {
                return Err(ToolChoiceError::InvalidDefinitionMap(
                    tool.function.name.clone(),
                ));
            }
            let reference = format!("#/$defs/tool_{index}");
            if rebase_schema_refs(&mut schema, &reference) || schema.get("$defs").is_some() {
                defs.insert(format!("tool_{index}"), schema);
                schema = json!({"$ref": reference});
            }
            any_of.push(json!({
                "properties": {
                    "name": {
                        "type": "string",
                        "enum": [tool.function.name],
                    },
                    "parameters": schema,
                },
                "required": ["name", "parameters"],
            }));
        }

        // Build the top-level array schema with anyOf constraints
        let mut result = json!({
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "anyOf": any_of,
            },
        });

        // `parallel_tool_calls: false` is otherwise enforced only downstream, by discarding
        // tool indices above zero in the HTTP stream. That leaves the extra calls GENERATED
        // and observable by any consumer upstream of that filter, and it wastes the tokens
        // spent producing them. Constraining generation is the earlier, cheaper fix; the
        // HTTP filter stays as defense in depth.
        if parallel_tool_calls == Some(false)
            && let Value::Object(map) = &mut result
        {
            map.insert("maxItems".to_string(), json!(1));
        }

        if !defs.is_empty()
            && let Value::Object(map) = &mut result
        {
            map.insert("$defs".to_string(), Value::Object(defs));
        }

        Ok(result)
    }

    /// Relocate fragment-only JSON Pointer references with their parameter document.
    /// An explicit `$id` starts a separate resource; its references retain that base.
    fn rebase_schema_refs(schema: &mut Value, root: &str) -> bool {
        let mut rebased = false;
        match schema {
            Value::Array(schemas) => {
                for schema in schemas {
                    rebased |= rebase_schema_refs(schema, root);
                }
            }
            Value::Object(schema) => {
                if schema.contains_key("$id") {
                    return false;
                }
                if let Some(Value::String(reference)) = schema.get_mut("$ref")
                    && let Some(fragment) = reference.strip_prefix('#')
                    && (fragment.is_empty()
                        || fragment.starts_with('/')
                        || fragment
                            .get(..3)
                            .is_some_and(|prefix| prefix.eq_ignore_ascii_case("%2f")))
                {
                    *reference = format!("{root}{fragment}");
                    rebased = true;
                }
                for (keyword, value) in schema {
                    match keyword.as_str() {
                        "$defs" | "definitions" | "properties" | "patternProperties"
                        | "dependentSchemas" | "dependencies" => {
                            if let Value::Object(schemas) = value {
                                for schema in schemas.values_mut() {
                                    rebased |= rebase_schema_refs(schema, root);
                                }
                            }
                        }
                        "items"
                        | "prefixItems"
                        | "additionalItems"
                        | "contains"
                        | "additionalProperties"
                        | "unevaluatedProperties"
                        | "unevaluatedItems"
                        | "propertyNames"
                        | "allOf"
                        | "anyOf"
                        | "oneOf"
                        | "not"
                        | "if"
                        | "then"
                        | "else"
                        | "contentSchema" => rebased |= rebase_schema_refs(value, root),
                        _ => {}
                    }
                }
            }
            _ => {}
        }
        rebased
    }

    /// ORACLE CHANGE: the caller's request error is a plain message, as dynamo's
    /// `DynamoError` (`InvalidArgument`) displays it.
    fn invalid_argument(message: impl Into<String>, _public_message: Option<&str>) -> String {
        message.into()
    }

    /// `effective_tools` (`lib/llm/src/preprocessor/tool_choice.rs:314`).
    /// ORACLE CHANGE: the moved `effective_tools` with the caller's validator for
    /// `validate::validated_effective_tools`, which calls dynamo's `validate_tools`.
    /// It, and the moved `convert_tool_choice` and `requires_native_tool_call_format`
    /// the oracle also calls, match dynamo `37f2f0a712` line for line apart from
    /// visibility, that validator, and `..Default::default()` for fields dynamo sets
    /// to `None`.
    pub fn effective_tools<F>(
        request: &CreateChatCompletionRequest,
        validate: F,
    ) -> Result<Cow<'_, [ChatCompletionTool]>, String>
    where
        F: FnOnce(&[ChatCompletionTool]) -> anyhow::Result<()>,
    {
        tools::effective_tools(request, validate)
            .map_err(|error| invalid_argument(error.to_string(), None))
    }

    /// `guided_tool_constraint` (`lib/llm/src/preprocessor/tool_choice.rs:335`).
    /// ORACLE CHANGE: the `guided_*` check and the tool validator come in.
    pub fn guided_tool_constraint<F>(
        request: &CreateChatCompletionRequest,
        has_explicit_guided_decoding: bool,
        tool_call_parser: Option<&str>,
        reasoning_parser: Option<&str>,
        uses_structural_tag: bool,
        validate: F,
    ) -> Result<GuidedToolConstraint, String>
    where
        F: FnOnce(&[ChatCompletionTool]) -> anyhow::Result<()>,
    {
        if uses_structural_tag {
            return Ok(GuidedToolConstraint::StructuralTag);
        }
        let tools = effective_tools(request, validate)?;
        guided_tool_constraint_with_effective_tools(
            request,
            has_explicit_guided_decoding,
            tool_call_parser,
            reasoning_parser,
            false,
            tools.as_ref(),
        )
    }

    /// `guided_tool_constraint_with_effective_tools`
    /// (`lib/llm/src/preprocessor/tool_choice.rs:356`).
    /// ORACLE CHANGE: `has_explicit_guided_decoding(request)` comes in as a value;
    /// `request.inner.*` is `request.*`.
    pub fn guided_tool_constraint_with_effective_tools(
        request: &CreateChatCompletionRequest,
        has_explicit_guided_decoding: bool,
        tool_call_parser: Option<&str>,
        reasoning_parser: Option<&str>,
        uses_structural_tag: bool,
        tools: &[ChatCompletionTool],
    ) -> Result<GuidedToolConstraint, String> {
        if uses_structural_tag {
            return Ok(GuidedToolConstraint::StructuralTag);
        }
        let tool_choice = request
            .tool_choice
            .as_ref()
            .unwrap_or(&ChatCompletionToolChoiceOption::Auto);
        validate_openai_tool_choice(Some(tool_choice), Some(tools))
            .map_err(|error| invalid_argument(error.to_string(), None))?;
        let is_forced_tool_choice = matches!(
            tool_choice,
            ChatCompletionToolChoiceOption::Required | ChatCompletionToolChoiceOption::Named(_)
        );
        if !is_forced_tool_choice {
            return Ok(GuidedToolConstraint::None);
        }
        if has_explicit_guided_decoding {
            return Ok(GuidedToolConstraint::None);
        }
        let parser_tool_choice = convert_tool_choice(tool_choice);
        if requires_native_tool_call_format(tool_call_parser, &parser_tool_choice)
            || requires_native_tool_call_format(reasoning_parser, &parser_tool_choice)
        {
            return Ok(GuidedToolConstraint::None);
        }
        match get_tool_choice_guidance_from_tools(
            Some(tool_choice),
            Some(tools),
            request.parallel_tool_calls,
        ) {
            Ok(Some(_)) => Ok(installed_json_constraint(tool_choice)),
            Ok(None) => Ok(GuidedToolConstraint::None),
            Err(e) => Err(invalid_argument(e.to_string(), None)),
        }
    }

    /// `installed_json_constraint` (`lib/llm/src/preprocessor/tool_choice.rs:416`).
    fn installed_json_constraint(
        tool_choice: &ChatCompletionToolChoiceOption,
    ) -> GuidedToolConstraint {
        match tool_choice {
            ChatCompletionToolChoiceOption::Named(named) => GuidedToolConstraint::GuidedJsonNamed {
                tool_name: named.function.name.clone(),
            },
            ChatCompletionToolChoiceOption::Required => GuidedToolConstraint::GuidedJsonRequired,
            ChatCompletionToolChoiceOption::Auto | ChatCompletionToolChoiceOption::None => {
                GuidedToolConstraint::None
            }
        }
    }
}

// ---- The generated inputs ----

/// Parameter schemas, each aimed at one rule of the grammar builder.
fn schemas() -> Vec<Option<Value>> {
    let closed = || json!({"type": "object", "properties": {}, "additionalProperties": false});
    let with = |key: &str, value: Value| {
        let mut schema = closed();
        schema[key] = value;
        schema
    };
    let mut schemas = vec![
        None,
        Some(json!({
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"]
        })),
        Some(closed()),
        Some(with("required", json!([]))),
        Some(json!({"type": "object", "properties": {}})),
        Some(json!({"type": "object"})),
        Some(json!({"type": "object", "additionalProperties": false})),
        Some(json!({"type": "object", "properties": {}, "additionalProperties": true})),
        Some(json!({"type": "object", "properties": {}, "additionalProperties": {}})),
        Some(json!({"type": "string"})),
        Some(json!({"properties": {}, "additionalProperties": false})),
        Some(with("properties", json!([]))),
        Some(with("properties", json!({"x": {"type": "string"}}))),
        Some(with("required", json!(["x"]))),
        Some(with("required", json!("x"))),
        Some(with("minProperties", json!(1))),
        Some(with("minProperties", json!(-1))),
        Some(with("maxProperties", json!(0))),
        Some(with("maxProperties", json!(1.5))),
        Some(with("patternProperties", json!({}))),
        Some(with(
            "$schema",
            json!("https://json-schema.org/draft/2020-12/schema"),
        )),
        Some(json!(true)),
        Some(json!([])),
        Some(json!("object")),
        Some(json!({})),
    ];
    for (key, value) in [
        ("title", json!("t")),
        ("description", json!("d")),
        ("$comment", json!("c")),
        ("default", json!({})),
        ("deprecated", json!(true)),
        ("examples", json!([{}])),
        ("readOnly", json!(true)),
        ("writeOnly", json!(false)),
        ("minProperties", json!(0)),
        ("maxProperties", json!(3)),
    ] {
        schemas.push(Some(with(key, value)));
    }

    // Local references and definition maps.
    let value_ref =
        |reference: &str| json!({"type": "object", "properties": {"value": {"$ref": reference}}});
    for reference in [
        "#",
        "#/",
        "#/$defs/shared",
        "#/definitions/shared",
        "#/properties/other",
        "#%2F$defs%2Fshared",
        "#%2f$defs%2fshared",
        "#%2",
        "#%3Fx",
        "#anchor",
        "#/$defs/a~1b~0c",
        "urn:example:remote",
        "a%2Fremote",
        "other.json#/$defs/shared",
        "",
    ] {
        let mut schema = value_ref(reference);
        schema["$defs"] = json!({"shared": {"type": "string"}, "a/b~c": {"type": "number"}});
        schema["definitions"] = json!({"shared": {"type": "integer"}});
        schema["properties"]["other"] = json!({"type": "boolean"});
        schemas.push(Some(schema));
        schemas.push(Some(value_ref(reference)));
    }
    schemas.push(Some(
        json!({"type": "object", "properties": {"value": {"$ref": 7}}}),
    ));
    schemas.push(Some(json!({"type": "object", "$defs": {}})));
    schemas.push(Some(json!({"type": "object", "$defs": []})));
    schemas.push(Some(json!({"type": "object", "$defs": "shared"})));
    schemas.push(Some(json!({"type": "object", "$defs": null})));
    schemas.push(Some(json!({"type": "object", "definitions": []})));
    schemas.push(Some(
        json!({"type": "object", "$id": "urn:root", "properties": {"v": {"$ref": "#"}}}),
    ));
    schemas.push(Some(json!({
        "type": "object",
        "properties": {
            "scoped": {"$id": "urn:scoped", "$ref": "#/$defs/local", "$defs": {"local": {"type": "string"}}},
            "plain": {"$ref": "#/$defs/local"}
        },
        "$defs": {"local": {"type": "number"}}
    })));
    // A local reference under each keyword the rebase walks, and under keywords it
    // must leave alone.
    let reference = json!({"$ref": "#/$defs/shared"});
    for keyword in [
        "items",
        "prefixItems",
        "additionalItems",
        "contains",
        "additionalProperties",
        "unevaluatedProperties",
        "unevaluatedItems",
        "propertyNames",
        "allOf",
        "anyOf",
        "oneOf",
        "not",
        "if",
        "then",
        "else",
        "contentSchema",
        "const",
        "enum",
        "default",
        "examples",
        "x-custom",
    ] {
        for value in [
            reference.clone(),
            json!([reference.clone(), {"type": "null"}]),
        ] {
            schemas.push(Some(json!({"type": "object", keyword: value})));
        }
    }
    for keyword in [
        "$defs",
        "definitions",
        "properties",
        "patternProperties",
        "dependentSchemas",
        "dependencies",
    ] {
        schemas.push(Some(
            json!({"type": "object", keyword: {"k": reference.clone()}}),
        ));
        schemas.push(Some(
            json!({"type": "object", keyword: [reference.clone()]}),
        ));
    }
    schemas.push(Some(json!({
        "type": "object",
        "properties": {"deep": {"items": {"anyOf": [{"not": {"$ref": "#"}}]}}}
    })));
    schemas
}

const NAMES: [&str; 3] = ["get_weather", "lookup", "get_weather"];

fn tool(name: &str, parameters: &Option<Value>) -> Value {
    let mut function = json!({"name": name});
    if let Some(parameters) = parameters {
        function["parameters"] = parameters.clone();
    }
    json!({"type": "function", "function": function})
}

/// Top-level tool lists: `None` for an omitted field.
fn tool_lists() -> Vec<Option<Vec<Value>>> {
    let schemas = schemas();
    let n = schemas.len();
    let mut lists = vec![None, Some(vec![])];
    for (index, schema) in schemas.iter().enumerate() {
        lists.push(Some(vec![tool(NAMES[0], schema)]));
        lists.push(Some(vec![
            tool(NAMES[0], schema),
            tool(NAMES[1], &schemas[(index * 7 + 3) % n]),
        ]));
        // Three tools, with a duplicate name: a named choice takes the first.
        lists.push(Some(vec![
            tool(NAMES[1], &schemas[(index * 5 + 1) % n]),
            tool(NAMES[0], schema),
            tool(NAMES[2], &schemas[(index * 11 + 2) % n]),
        ]));
    }
    lists
}

fn choices() -> Vec<Option<Value>> {
    let named = |name: &str| json!({"type": "function", "function": {"name": name}});
    vec![
        None,
        Some(json!("none")),
        Some(json!("auto")),
        Some(json!("required")),
        Some(named(NAMES[0])),
        Some(named(NAMES[1])),
        Some(named("missing")),
        // Names match exactly, not ignoring case.
        Some(named("GET_WEATHER")),
    ]
}

const PARALLEL: [Option<bool>; 3] = [None, Some(true), Some(false)];

/// Messages with system-message tools, for the effective tool list.
fn message_sets() -> Vec<Value> {
    let user = json!({"role": "user", "content": "go"});
    let system = |tools: Value| json!({"role": "system", "content": "", "tools": tools});
    vec![
        json!([user.clone()]),
        json!([
            system(json!([{"name": "lookup", "parameters": {"type": "object"}}])),
            user.clone()
        ]),
        json!([
            system(
                json!([{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {}, "additionalProperties": false}}}])
            ),
            user.clone()
        ]),
        json!([
            system(
                json!([{"name": "lookup", "parameters": {"$defs": {"d": {"type": "string"}}, "properties": {"v": {"$ref": "#/$defs/d"}}}}])
            ),
            user.clone()
        ]),
        json!([system(json!(["not an object"])), user.clone()]),
        json!([system(json!([{"name": ""}])), user.clone()]),
        json!([system(json!([{"name": "has space"}])), user.clone()]),
    ]
}

fn request(
    messages: &Value,
    tools: &Option<Vec<Value>>,
    choice: &Option<Value>,
    parallel: Option<bool>,
) -> CreateChatCompletionRequest {
    let mut value = json!({"model": "m", "messages": messages});
    if let Some(tools) = tools {
        value["tools"] = json!(tools);
    }
    if let Some(choice) = choice {
        value["tool_choice"] = choice.clone();
    }
    if let Some(parallel) = parallel {
        value["parallel_tool_calls"] = json!(parallel);
    }
    serde_json::from_value(value).expect("generated request must deserialize")
}

/// A stand-in for the caller's tool validation, so its errors and their order show.
/// Its error has a cause, because dynamo shows only the top message.
fn validate(tools: &[ChatCompletionTool]) -> anyhow::Result<()> {
    for (index, tool) in tools.iter().enumerate() {
        if tool.function.name.contains(' ') {
            return Err(anyhow::anyhow!("a space is not allowed")
                .context(format!("Function at index {index} has an invalid name")));
        }
    }
    Ok(())
}

const PARSERS: [(Option<&str>, Option<&str>); 9] = [
    (None, None),
    (Some("hermes"), None),
    (Some("kimi_k2"), None),
    (None, Some("kimi_k2")),
    (Some("kimi_k3"), None),
    (None, Some("kimi-k3")),
    (Some("kimi-k3"), Some("qwen3")),
    (Some("kimi_k25"), Some("kimi_k25")),
    (Some("harmony"), Some("gpt_oss")),
];

// ---- Rendering ----

fn text(value: &Value) -> String {
    serde_json::to_string(value).expect("serialize")
}

fn guidance(result: Result<Option<ToolChoiceGuidance>, tool_choice::ToolChoiceError>) -> String {
    match result {
        Ok(None) => "none".into(),
        Ok(Some(ToolChoiceGuidance::Json(schema))) => format!("json {}", text(&schema)),
        Ok(Some(ToolChoiceGuidance::Regex(regex))) => format!("regex {regex}"),
        Err(error) => format!("error {error:?} {error}"),
    }
}

fn oracle_guidance(
    result: Result<Option<oracle::ToolChoiceGuidance>, oracle::ToolChoiceError>,
) -> String {
    match result {
        Ok(None) => "none".into(),
        Ok(Some(oracle::ToolChoiceGuidance::Json(schema))) => format!("json {}", text(&schema)),
        Ok(Some(oracle::ToolChoiceGuidance::Regex(regex))) => format!("regex {regex}"),
        Err(error) => format!("error {error:?} {error}"),
    }
}

fn schema(result: Result<Option<Value>, impl std::fmt::Debug + std::fmt::Display>) -> String {
    match result {
        Ok(None) => "none".into(),
        Ok(Some(schema)) => format!("json {}", text(&schema)),
        Err(error) => format!("error {error:?} {error}"),
    }
}

fn unit(result: Result<(), impl std::fmt::Debug + std::fmt::Display>) -> String {
    match result {
        Ok(()) => "ok".into(),
        Err(error) => format!("error {error:?} {error}"),
    }
}

fn constraint(result: Result<GuidedToolConstraint, String>) -> String {
    match result {
        Ok(constraint) => text(&serde_json::to_value(constraint).expect("serialize")),
        Err(error) => format!("error {error}"),
    }
}

/// Counts outcomes by kind, so a generator change that stops reaching a rule fails.
#[derive(Default)]
struct Coverage(BTreeMap<String, usize>);

impl Coverage {
    fn add(&mut self, rendered: &str) {
        let kind = if let Some(rest) = rendered.strip_prefix("error ") {
            format!(
                "error {}",
                rest.split([' ', '(']).next().unwrap_or_default()
            )
        } else if rendered.starts_with("json ") {
            let mut kind = String::from("json");
            for marker in ["\"$defs\"", "\"maxItems\"", "\"anyOf\"", "#/$defs/tool_"] {
                if rendered.contains(marker) {
                    kind.push(' ');
                    kind.push_str(marker);
                }
            }
            kind
        } else {
            rendered.split(' ').next().unwrap_or_default().to_string()
        };
        *self.0.entry(kind).or_default() += 1;
    }

    fn at_least(&self, kind: &str, min: usize) {
        let count = self.0.get(kind).copied().unwrap_or_default();
        assert!(
            count >= min,
            "only {count} outcomes of kind {kind:?}: {:#?}",
            self.0
        );
    }
}

fn parse_choice(choice: &Option<Value>) -> Option<ChatCompletionToolChoiceOption> {
    choice
        .as_ref()
        .map(|choice| serde_json::from_value(choice.clone()).expect("tool choice"))
}

fn parse_tools(tools: &Option<Vec<Value>>) -> Option<Vec<ChatCompletionTool>> {
    tools
        .as_ref()
        .map(|tools| serde_json::from_value(json!(tools)).expect("tools"))
}

/// The grammar builder and its checks, on every tool list, choice, and
/// `parallel_tool_calls` value.
#[test]
fn grammar_matches_dynamo_llm() {
    let mut coverage = Coverage::default();
    let mut rows = 0;
    let mut mismatches = Vec::new();
    for tools in tool_lists() {
        let parsed = parse_tools(&tools);
        let tools = parsed.as_deref();
        let names: Vec<&str> = tools
            .unwrap_or_default()
            .iter()
            .map(|tool| tool.function.name.as_str())
            .collect();
        for name in [
            None,
            Some(NAMES[0]),
            Some(NAMES[1]),
            Some("missing"),
            Some("GET_WEATHER"),
        ] {
            for required in [false, true] {
                let (got, want) = match (name, required) {
                    (None, false) => (
                        tool_choice::ToolChoiceValidation::Unforced,
                        oracle::ToolChoiceValidation::Unforced,
                    ),
                    (None, true) => (
                        tool_choice::ToolChoiceValidation::Required,
                        oracle::ToolChoiceValidation::Required,
                    ),
                    (Some(name), _) => (
                        tool_choice::ToolChoiceValidation::Named(name),
                        oracle::ToolChoiceValidation::Named(name),
                    ),
                };
                let got = unit(tool_choice::validate_tool_choice_against_names(
                    got,
                    names.iter().copied(),
                ));
                let want = unit(oracle::validate_tool_choice_against_names(
                    want,
                    names.iter().copied(),
                ));
                if got != want {
                    mismatches.push(format!(
                        "names {name:?} {names:?}\n  got {got}\n want {want}"
                    ));
                }
            }
        }
        for choice in choices() {
            let choice = parse_choice(&choice);
            let choice = choice.as_ref();
            let validated = unit(tool_choice::validate_openai_tool_choice(choice, tools));
            let expected = unit(oracle::validate_openai_tool_choice(choice, tools));
            if validated != expected {
                mismatches.push(format!(
                    "validate {choice:?} {tools:?}\n  got {validated}\n want {expected}"
                ));
            }
            for parallel in PARALLEL {
                rows += 1;
                let got = guidance(tool_choice::get_tool_choice_guidance_from_tools(
                    choice, tools, parallel,
                ));
                let want = oracle_guidance(oracle::get_tool_choice_guidance_from_tools(
                    choice, tools, parallel,
                ));
                coverage.add(&want);
                if got != want {
                    mismatches.push(format!(
                        "guidance {choice:?} {parallel:?} {tools:?}\n  got {got}\n want {want}"
                    ));
                }
                let got = schema(tool_choice::get_json_schema_from_tools(
                    choice, tools, parallel,
                ));
                let want = schema(oracle::get_json_schema_from_tools(choice, tools, parallel));
                if got != want {
                    mismatches.push(format!(
                        "json schema {choice:?} {parallel:?} {tools:?}\n  got {got}\n want {want}"
                    ));
                }
            }
        }
    }
    assert!(
        mismatches.is_empty(),
        "{} of {rows} rows differ from dynamo-llm; first: {}",
        mismatches.len(),
        mismatches[0]
    );
    eprintln!("grammar rows: {rows}; outcomes: {:#?}", coverage.0);
    for (kind, min) in [
        ("none", 100),
        ("regex", 50),
        ("json", 100),
        ("json \"maxItems\" \"anyOf\"", 50),
        ("json \"$defs\" \"maxItems\" \"anyOf\" #/$defs/tool_", 50),
        ("json \"$defs\" \"anyOf\" #/$defs/tool_", 50),
        ("error MissingTools", 10),
        ("error ToolNotFound", 100),
        ("error InvalidDefinitionMap", 10),
        ("error EmptyTools", 3),
    ] {
        coverage.at_least(kind, min);
    }
}

/// `guided_tool_constraint` and its effective-tools form, on every request, parser
/// pair, structural-tag decision, and `guided_*` value.
#[test]
fn constraint_matches_dynamo_llm() {
    let mut coverage = Coverage::default();
    let mut rows = 0;
    let mut mismatches = Vec::new();
    let tool_lists = tool_lists();
    let message_sets = message_sets();
    for (list_index, tools) in tool_lists.iter().enumerate() {
        for (message_index, messages) in message_sets.iter().enumerate() {
            // System-message tools on every seventh list keep the run short.
            if message_index > 0 && list_index % 7 != 0 {
                continue;
            }
            for choice in choices() {
                for parallel in PARALLEL {
                    let request = request(messages, tools, &choice, parallel);
                    for (tool_call_parser, reasoning_parser) in PARSERS {
                        for uses_structural_tag in [false, true] {
                            for explicit in [false, true] {
                                rows += 1;
                                let got = constraint(
                                    tool_choice::guided_tool_constraint(
                                        &request,
                                        explicit,
                                        tool_call_parser,
                                        reasoning_parser,
                                        uses_structural_tag,
                                        validate,
                                    )
                                    .map_err(|error| error.0),
                                );
                                let want = constraint(oracle::guided_tool_constraint(
                                    &request,
                                    explicit,
                                    tool_call_parser,
                                    reasoning_parser,
                                    uses_structural_tag,
                                    validate,
                                ));
                                coverage.add(&want);
                                if got != want {
                                    mismatches.push(format!(
                                        "constraint {tool_call_parser:?}/{reasoning_parser:?} st={uses_structural_tag} guided={explicit} {}\n  got {got}\n want {want}",
                                        serde_json::to_string(&request).unwrap()
                                    ));
                                }
                                // The effective-tools form, on the raw top-level list:
                                // it skips the merge and validation, so it sees lists
                                // `effective_tools` rejects.
                                let raw = request.tools.as_deref().unwrap_or_default();
                                let got = constraint(
                                    tool_choice::guided_tool_constraint_with_effective_tools(
                                        &request,
                                        explicit,
                                        tool_call_parser,
                                        reasoning_parser,
                                        uses_structural_tag,
                                        raw,
                                    )
                                    .map_err(|error| error.0),
                                );
                                let want = constraint(
                                    oracle::guided_tool_constraint_with_effective_tools(
                                        &request,
                                        explicit,
                                        tool_call_parser,
                                        reasoning_parser,
                                        uses_structural_tag,
                                        raw,
                                    ),
                                );
                                if got != want {
                                    mismatches.push(format!(
                                        "effective-tools constraint {tool_call_parser:?}/{reasoning_parser:?} st={uses_structural_tag} guided={explicit} {}\n  got {got}\n want {want}",
                                        serde_json::to_string(&request).unwrap()
                                    ));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    assert!(
        mismatches.is_empty(),
        "{} of {rows} rows differ from dynamo-llm; first: {}",
        mismatches.len(),
        mismatches[0]
    );
    eprintln!("constraint rows: {rows}; outcomes: {:#?}", coverage.0);
    for (kind, min) in [
        ("\"none\"", 1000),
        ("\"structural_tag\"", 1000),
        ("\"guided_json_required\"", 100),
        (
            "{\"guided_json_named\":{\"tool_name\":\"get_weather\"}}",
            100,
        ),
        ("{\"guided_json_named\":{\"tool_name\":\"lookup\"}}", 100),
        // `ToolChoiceError`s, by their first word.
        ("error tool", 1000),
        ("error tool_choice", 10),
        ("error $defs", 100),
        // Errors from `effective_tools` and the caller's validator.
        ("error tools", 100),
        ("error dynamic", 1000),
        ("error Function", 1000),
    ] {
        coverage.at_least(kind, min);
    }
}

/// The oracle's `effective_tools` errors keep dynamo's text: the top error only,
/// as `DynamoError` shows it, not the anyhow chain.
#[test]
fn effective_tools_errors_keep_dynamo_text() {
    let request = request(
        &json!([{"role": "user", "content": "go"}]),
        &Some(vec![]),
        &None,
        None,
    );
    let error = tool_choice::guided_tool_constraint(&request, false, None, None, false, validate)
        .expect_err("an explicit empty tools array is rejected");
    assert_eq!(
        error.0,
        "tools array cannot be empty; omit the field entirely to disable tool use"
    );
    assert_eq!(
        oracle::effective_tools(&request, validate).map(|_| ()),
        Err(error.0)
    );
}
