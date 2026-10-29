// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The grammar a forced `tool_choice` installs, and the guided-tool constraint a
//! request implies.
//!
//! Moved from dynamo `lib/llm/src/protocols/openai/tools.rs` (the grammar) and
//! `lib/llm/src/preprocessor/tool_choice.rs` (the constraint).

use dynamo_protocols::types::{
    ChatCompletionTool, ChatCompletionToolChoiceOption, CreateChatCompletionRequest, FunctionObject,
};
use serde_json::{Value, json};
use thiserror::Error;

use crate::guided::GuidedToolConstraint;
use crate::route::InvalidRequest;
use crate::tools::{convert_tool_choice, effective_tools, requires_native_tool_call_format};

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

/// A `tool_choice` reduced to what the forced-choice check needs, so the OpenAI wire
/// type and the parser crate's `ToolChoice` share one check.
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

fn find_tool<'a>(tools: &'a [ChatCompletionTool], name: &str) -> Option<&'a ChatCompletionTool> {
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

/// The guided-tool constraint a request implies, given what the structural-tag stage
/// already decided.
///
/// Direct postprocessor callers and remote frontends use this compatibility helper
/// when they did not run request preprocessing. The worker streaming path carries the
/// constraint its guided-decoding stage installed; topology-B aggregation may
/// reconstruct a structural-tag request as guided JSON, so its complete-output parser
/// also checks the generated wire shape.
///
/// `has_explicit_guided_decoding` says whether the request's own `guided_*` fields
/// install a grammar. Those fields are not on [`CreateChatCompletionRequest`], so the
/// caller reads them. `validate` is the caller's tool validation, as for
/// [`effective_tools`].
///
/// # Errors
///
/// [`InvalidRequest`] when [`effective_tools`] fails, or when a forced `tool_choice`
/// does not match the effective tools.
pub fn guided_tool_constraint<F>(
    request: &CreateChatCompletionRequest,
    has_explicit_guided_decoding: bool,
    tool_call_parser: Option<&str>,
    reasoning_parser: Option<&str>,
    uses_structural_tag: bool,
    validate: F,
) -> Result<GuidedToolConstraint, InvalidRequest>
where
    F: FnOnce(&[ChatCompletionTool]) -> anyhow::Result<()>,
{
    if uses_structural_tag {
        return Ok(GuidedToolConstraint::StructuralTag);
    }
    let tools =
        effective_tools(request, validate).map_err(|error| InvalidRequest(error.to_string()))?;
    guided_tool_constraint_with_effective_tools(
        request,
        has_explicit_guided_decoding,
        tool_call_parser,
        reasoning_parser,
        false,
        tools.as_ref(),
    )
}

/// Derive the guided-tool constraint from an effective tool set the caller has
/// already normalized and validated through [`effective_tools`].
///
/// # Errors
///
/// [`InvalidRequest`] when a forced `tool_choice` does not match `tools`.
pub fn guided_tool_constraint_with_effective_tools(
    request: &CreateChatCompletionRequest,
    has_explicit_guided_decoding: bool,
    tool_call_parser: Option<&str>,
    reasoning_parser: Option<&str>,
    uses_structural_tag: bool,
    tools: &[ChatCompletionTool],
) -> Result<GuidedToolConstraint, InvalidRequest> {
    if uses_structural_tag {
        return Ok(GuidedToolConstraint::StructuralTag);
    }
    let tool_choice = request
        .tool_choice
        .as_ref()
        .unwrap_or(&ChatCompletionToolChoiceOption::Auto);
    validate_openai_tool_choice(Some(tool_choice), Some(tools))
        .map_err(|error| InvalidRequest(error.to_string()))?;
    let is_forced_tool_choice = matches!(
        tool_choice,
        ChatCompletionToolChoiceOption::Required | ChatCompletionToolChoiceOption::Named(_)
    );
    // Only a forced choice installs a tool-level JSON schema. Auto/None leave any
    // JSON constraint to `response_format`, which governs assistant CONTENT.
    if !is_forced_tool_choice {
        return Ok(GuidedToolConstraint::None);
    }
    // Forced + explicit guided decoding is rejected at request time, so reaching here
    // with both means the request never ran.
    if has_explicit_guided_decoding {
        return Ok(GuidedToolConstraint::None);
    }
    let parser_tool_choice = convert_tool_choice(tool_choice);
    // Kimi forced requests use marker-delimited native formats that generic JSON
    // guidance cannot represent. Keep this compatibility path aligned with
    // request preprocessing for either configured parser slot.
    if requires_native_tool_call_format(tool_call_parser, &parser_tool_choice)
        || requires_native_tool_call_format(reasoning_parser, &parser_tool_choice)
    {
        return Ok(GuidedToolConstraint::None);
    }
    // Validate the forced choice against the actual tools the same way request
    // preprocessing does, instead of blindly installing a constraint for a
    // `tool_choice` that names a tool absent from `tools` (or an empty `tools`
    // list under `tool_choice: "required"`).
    match get_tool_choice_guidance_from_tools(
        Some(tool_choice),
        Some(tools),
        request.parallel_tool_calls,
    ) {
        Ok(Some(_)) => Ok(installed_json_constraint(tool_choice)),
        Ok(None) => Ok(GuidedToolConstraint::None),
        Err(e) => Err(InvalidRequest(e.to_string())),
    }
}

/// Map a forced `tool_choice` onto the guided-JSON parser constraint for its grammar.
///
/// Only reachable once `get_tool_choice_guidance_from_tools` produced either JSON or
/// regex guidance. It returns `None` for `Auto`/`None`, so those arms are unreachable
/// in practice and report [`GuidedToolConstraint::None`] rather than guessing.
fn installed_json_constraint(tool_choice: &ChatCompletionToolChoiceOption) -> GuidedToolConstraint {
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

#[cfg(test)]
#[path = "tool_choice_tests.rs"]
mod tests;
