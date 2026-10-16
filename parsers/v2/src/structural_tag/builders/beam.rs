// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Per-turn structural tag for the Beam channel grammar.
//!
//! Mirrors `tool_call_structural_tag` of the model builder's SGLang serving layer
//! (`response_template_structural_tag.py`), specialised to the Beam response template, so
//! both serving paths constrain the same grammar: a non-empty analysis channel, then
//! either a tool call on the channel wire
//! (` to=NAME<|channel|>analysis <|constrain|>xml<|message|>` with `rfl:parameter`
//! arguments, each held to its JSON-schema type) or, for automatic tool choice, the final
//! answer — free text, or a JSON document of the request's structured-output schema. Under automatic choice the call may also name an undeclared tool with free-form
//! arguments, as the serving layer forwards unknown tools. Arguments of a held tool come in
//! declared or alphabetical order, as Beam writes them sorted.
//!
//! The fixtures under `tests/data/structural_tags/beam/` were produced by the serving
//! layer's own builder (`BeamDetector.get_structural_tag`, `thinking_mode=True`,
//! `message_start=True`) and the tests assert equality with them. Three deliberate
//! deviations, each of which only falls back to loose (untyped) arguments for that tool:
//!
//! - a string `pattern` is not held: the serving layer checks the regex against XGrammar;
//! - only the string `format`s the serving layer spells out are held;
//! - a JSON value is held only when its schema uses keywords XGrammar compiles, where the
//!   serving layer compiles the schema to find out.
//!
//! The grammar covers the whole message, so the caller says where the rendered prompt
//! leaves the assistant message ([`PromptChannel`]): at its bare `<|start|>assistant`
//! header, or inside an open analysis or final channel (assistant prefill).

use serde_json::{Map, Value, json};

use super::{StructuralTagBuilder, TurnGrammar};
use crate::Tool;
use crate::structural_tag::{PromptChannel, StructuralTagContext, StructuralTagToolChoice};

/// The Beam family's builder: a turn grammar, see [`beam_turn_structural_tag`].
pub(crate) static BEAM: StructuralTagBuilder = StructuralTagBuilder::turn(&BeamTurnGrammar);

struct BeamTurnGrammar;

impl TurnGrammar for BeamTurnGrammar {
    fn build_turn(
        &self,
        context: &StructuralTagContext<'_>,
        channel: PromptChannel,
    ) -> anyhow::Result<Option<Value>> {
        Ok(beam_turn_structural_tag(
            context.tools,
            context.tool_choice,
            context.parallel_tool_calls,
            channel,
            context.structured_output_schema,
        ))
    }
}

// The Beam response template, as the serving layer's `beam.py` declares it.
const THINK_OPEN: &str = "<|channel|>analysis<|message|>";
const THINK_CLOSE: &str = "<|end|><|start|>assistant";
const ANSWER_OPEN: &str = "<|channel|>final<|message|>";
const ANSWER_CLOSERS: [&str; 3] = ["<|return|>", "<|end|>", "<|call|>"];
const CALL_CLOSERS: [&str; 2] = ["<|end|><|start|>assistant", "<|call|>"];
const CALL_OPEN_BEFORE: &str = " to=";
const CALL_OPEN_AFTER: &str = r"<\|channel\|>analysis <\|constrain\|>xml<\|message\|>";
/// The tool name is any run without whitespace or `<`: `[^\s<]+`.
const NAME_PATTERN: &str = r"[^\s<]+";
const NAME_NEGATED_CLASS: &str = r"\s<";
const PARAM_BEGIN: &str = "<rfl:parameter name=\"";
const PARAM_MIDDLE: &str = "\">\n";
const PARAM_END: &str = "\n</rfl:parameter>";
/// A value may also end where the next parameter opens.
const PARAM_FOLLOW: &str = "\n<rfl:parameter name=\"";
const KEY_PATTERN: &str = "[^\"]+";
/// Templates write at most a newline between parameters.
const SEPARATOR_PATTERN: &str = r"\n?";

/// Keywords that describe a schema without constraining its values.
const ANNOTATIONS: &[&str] = &[
    "title",
    "description",
    "default",
    "examples",
    "deprecated",
    "$comment",
    "$schema",
    "$id",
];
const OBJECT_KEYWORDS: &[&str] = &[
    "type",
    "properties",
    "required",
    "additionalProperties",
    "$defs",
    "definitions",
];
const JSON_TYPES: [&str; 7] = [
    "string", "number", "integer", "boolean", "array", "object", "null",
];
// Keyword families the serving layer infers a type from (`kimik3_structural_tag`).
const INFER_STRING_KEYWORDS: &[&str] = &["format", "maxLength", "minLength", "pattern"];
const INFER_NUMBER_KEYWORDS: &[&str] = &[
    "exclusiveMaximum",
    "exclusiveMinimum",
    "maximum",
    "minimum",
    "multipleOf",
];
const INFER_ARRAY_KEYWORDS: &[&str] = &[
    "contains",
    "items",
    "maxContains",
    "maxItems",
    "minContains",
    "minItems",
    "prefixItems",
    "uniqueItems",
];
const INFER_OBJECT_KEYWORDS: &[&str] = &[
    "additionalProperties",
    "dependentRequired",
    "dependentSchemas",
    "maxProperties",
    "minProperties",
    "patternProperties",
    "properties",
    "propertyNames",
    "required",
];
/// Keywords XGrammar's JSON-schema converter is known to accept. Anything else leaves the
/// value to free text, as an uncompilable schema would fail the request at the engine.
const XGRAMMAR_JSON_KEYWORDS: &[&str] = &[
    "type",
    "enum",
    "const",
    "properties",
    "required",
    "additionalProperties",
    "items",
    "prefixItems",
    "minItems",
    "maxItems",
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "minLength",
    "maxLength",
    "pattern",
    "format",
    "anyOf",
    "oneOf",
    "allOf",
    "$ref",
    "$defs",
    "definitions",
    "nullable",
];

/// The structural tag for one Beam turn, or `None` where the serving layer derives none: a
/// strict tool whose schema the XML arguments cannot hold, a tool name the template cannot
/// spell, a forced choice without tools, or a forced choice once the final channel is open.
pub(crate) fn beam_turn_structural_tag(
    tools: &[Tool],
    tool_choice: StructuralTagToolChoice<'_>,
    parallel_tool_calls: Option<bool>,
    channel: PromptChannel,
    answer_schema: Option<&Value>,
) -> Option<Value> {
    let answer = matches!(
        tool_choice,
        StructuralTagToolChoice::Auto | StructuralTagToolChoice::None
    );
    let named = matches!(tool_choice, StructuralTagToolChoice::Named(_));
    let selected: Vec<&Tool> = match tool_choice {
        StructuralTagToolChoice::None => Vec::new(),
        StructuralTagToolChoice::Named(name) => {
            tools.iter().filter(|tool| tool.name == name).collect()
        }
        StructuralTagToolChoice::Auto | StructuralTagToolChoice::Required => tools.iter().collect(),
    };
    let rest = turn_rest(
        &selected,
        answer,
        parallel_tool_calls != Some(false) && !named,
        matches!(tool_choice, StructuralTagToolChoice::Auto),
        answer_schema,
    )?;
    let format = match channel {
        PromptChannel::MessageStart => {
            let mut elements = vec![thinking_format()];
            elements.extend(rest);
            sequence(elements)
        }
        PromptChannel::OpenReasoning => {
            let mut elements = vec![any_text(&[THINK_CLOSE]), const_string(THINK_CLOSE)];
            elements.extend(rest);
            sequence(elements)
        }
        PromptChannel::OpenAnswer => {
            if !answer {
                return None;
            }
            match answer_schema {
                // The final channel is already open: only the JSON document remains.
                Some(schema) => sequence(vec![json_schema_format(schema)]),
                None => sequence(answer_text()),
            }
        }
    };
    Some(json!({"type": "structural_tag", "format": format}))
}

/// The output after the analysis: calls, the answer, or either.
fn turn_rest(
    tools: &[&Tool],
    answer: bool,
    parallel: bool,
    undeclared_names: bool,
    answer_schema: Option<&Value>,
) -> Option<Vec<Value>> {
    if tools.is_empty() {
        return answer.then(|| vec![reply_format(answer_schema)]);
    }
    let mut calls: Vec<(String, Value, bool)> = Vec::with_capacity(tools.len() + 1);
    for tool in tools {
        let name = &tool.name;
        let parameters = typed_parameters(&tool.parameters);
        let strict = tool.strict == Some(true);
        // `strict_arguments` holds every tool it can; a strict tool must be held.
        let mut arguments = match xml_arguments_format(&parameters, true) {
            Ok(arguments) => arguments,
            Err(Unencodable) if strict => return None,
            Err(Unencodable) => None,
        };
        let held = arguments.is_some();
        if !held && !strict {
            arguments = Some(any_parameters());
        }
        // A strict tool the arguments cannot hold leaves the turn unheld.
        let arguments = arguments?;
        if !name_encodable(name) {
            return None;
        }
        calls.push((escape_regex(name), arguments, held));
    }
    if undeclared_names {
        // Calls to tools that are not held all take the same arguments.
        let held_names: Vec<&str> = tools
            .iter()
            .zip(&calls)
            .filter(|(_, (_, _, held))| *held)
            .map(|(tool, _)| tool.name.as_str())
            .collect();
        let others = format!("(?:{})", other_names(&held_names));
        calls.retain(|(_, _, held)| *held);
        calls.push((others, any_parameters(), false));
    }
    let call = || {
        one_of(
            calls
                .iter()
                .map(|(name, arguments, _)| {
                    sequence(vec![
                        regex(&format!("{CALL_OPEN_BEFORE}{name}{CALL_OPEN_AFTER}")),
                        arguments.clone(),
                    ])
                })
                .collect(),
        )
    };
    let closers = const_strings(&CALL_CLOSERS);
    let mut elements = vec![call()];
    if parallel {
        elements.push(star(sequence(vec![closers.clone(), call()])));
    }
    // EOS can terminate a call without writing its closer.
    elements.push(optional(closers));
    if answer {
        elements = vec![one_of(vec![
            sequence(elements),
            reply_format(answer_schema),
        ])];
    }
    Some(elements)
}

/// A non-empty analysis channel; the grammar requires reasoning.
fn thinking_format() -> Value {
    // A first character that starts none of the closers.
    let nonempty = sequence(vec![regex("[^<]"), json!({"type": "any_text"})]);
    tag(THINK_OPEN, nonempty, &[THINK_CLOSE])
}

/// The final channel: free text, or a JSON document of `schema` (a structured answer),
/// which the parser serves whole, so no closer follows it.
fn reply_format(schema: Option<&Value>) -> Value {
    let mut elements = vec![const_string(ANSWER_OPEN)];
    match schema {
        Some(schema) => elements.push(json_schema_format(schema)),
        None => elements.extend(answer_text()),
    }
    sequence(elements)
}

fn json_schema_format(schema: &Value) -> Value {
    json!({"type": "json_schema", "json_schema": schema})
}

/// Whether the turn grammar holds `tool`'s arguments to its schema (every value framed
/// as the schema types it), rather than admitting any parameters as raw text. The Beam
/// parser frames values the same way under `ArgumentFraming::SchemaConstrained`.
pub(crate) fn beam_holds_tool(tool: &Tool) -> bool {
    matches!(
        xml_arguments_format(&typed_parameters(&tool.parameters), true),
        Ok(Some(_))
    )
}

/// Text that cannot close the answer early; like a call, the answer may end at a stop token
/// instead of a closer.
fn answer_text() -> Vec<Value> {
    vec![
        any_text(&ANSWER_CLOSERS),
        optional(const_strings(&ANSWER_CLOSERS)),
    ]
}

fn name_encodable(name: &str) -> bool {
    !name.is_empty() && !name.chars().any(|c| c.is_whitespace() || c == '<')
}

/// A regex for the names `[^\s<]+` matches besides `names`.
fn other_names(names: &[&str]) -> String {
    if names.is_empty() {
        return NAME_PATTERN.to_string();
    }
    #[derive(Default)]
    struct Node {
        children: Vec<(char, Node)>,
        terminal: bool,
    }
    let mut trie = Node::default();
    for name in names {
        let mut node = &mut trie;
        for c in name.chars() {
            let index = match node.children.iter().position(|(child, _)| *child == c) {
                Some(index) => index,
                None => {
                    node.children.push((c, Node::default()));
                    node.children.len() - 1
                }
            };
            node = &mut node.children[index].1;
        }
        node.terminal = true;
    }
    fn other(chars: &[char]) -> String {
        let mut sorted: Vec<char> = chars.to_vec();
        sorted.sort_unstable();
        let escaped: String = sorted.iter().map(|c| class_escape(*c)).collect();
        format!("[^{NAME_NEGATED_CLASS}{escaped}]")
    }
    /// The rest of a name past `node` that ends on none of the names.
    fn rest(node: &Node, first: bool) -> String {
        let chars: Vec<char> = node.children.iter().map(|(c, _)| *c).collect();
        let mut branches = vec![format!("{}{}*", other(&chars), other(&[]))];
        branches.extend(
            node.children.iter().map(|(c, child)| {
                format!("{}{}", escape_regex(&c.to_string()), rest(child, false))
            }),
        );
        let suffix = if first || node.terminal { "" } else { "?" };
        format!("(?:{}){suffix}", branches.join("|"))
    }
    rest(&trie, true)
}

/// Character classes in XGrammar regexes take whitespace only as escapes.
fn class_escape(c: char) -> String {
    match c {
        ' ' => " ".to_string(),
        '\t' => r"\t".to_string(),
        '\n' => r"\n".to_string(),
        '\r' => r"\r".to_string(),
        '\\' | ']' | '^' | '-' | '[' => format!("\\{c}"),
        _ => c.to_string(),
    }
}

pub(crate) fn escape_regex(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        if matches!(
            c,
            '\\' | '.' | '+' | '*' | '?' | '(' | ')' | '[' | ']' | '{' | '}' | '|' | '^' | '$'
        ) {
            out.push('\\');
        }
        out.push(c);
    }
    out
}

/// A tool name or parameter name the template cannot spell in a call.
struct Unencodable;

/// How one argument value is held.
#[derive(Clone, PartialEq)]
enum ValueFormat {
    /// Free text, which the parser reads as a string.
    Text,
    /// One of the string choices.
    Choices(Vec<String>),
    /// Raw text matching a regex.
    Regex(String),
    /// A JSON value of the schema.
    Json(Value),
}

/// The `rfl:parameter` body of a call to a tool with `parameters`: typed to the schema when
/// `strict`, else any parameters. `Ok(None)` where the schema cannot be held.
fn xml_arguments_format(
    parameters: &Map<String, Value>,
    strict: bool,
) -> Result<Option<Value>, Unencodable> {
    let properties = parameters.get("properties");
    let typed = match properties {
        None => true,
        Some(Value::Object(properties)) => properties.values().all(typed_by_parser),
        Some(_) => false,
    };
    // Other schema shapes cannot be typed by the XML parser.
    if strict
        && (!typed
            || parameters.keys().any(|key| {
                !ANNOTATIONS.contains(&key.as_str()) && !OBJECT_KEYWORDS.contains(&key.as_str())
            }))
    {
        return Ok(None);
    }
    let required: Vec<String> = match parameters.get("required") {
        Some(Value::Array(required)) => {
            if strict && properties.is_some() && !required.iter().all(Value::is_string) {
                // Required entries that are no names leave the tool to the JSON schema.
                return Ok(None);
            }
            required
                .iter()
                .filter_map(|key| key.as_str().map(str::to_string))
                .collect()
        }
        Some(other) if strict && properties.is_some() && truthy(other) => return Ok(None),
        _ => Vec::new(),
    };
    if !strict {
        return Ok(Some(any_parameters()));
    }
    let Some(properties) = properties.and_then(Value::as_object) else {
        return Ok(match parameters.get("additionalProperties") {
            Some(Value::Bool(false)) => Some(const_string("")),
            None | Some(Value::Bool(true)) => Some(any_parameters()),
            Some(Value::Object(extra)) if extra.is_empty() => Some(any_parameters()),
            Some(_) => None,
        });
    };
    // A declared key repeated as an extra one would bypass its schema.
    if parameters
        .get("additionalProperties")
        .is_some_and(|extra| *extra != Value::Bool(false))
    {
        return Ok(None);
    }
    let mut values: Vec<(String, ValueFormat)> = Vec::with_capacity(properties.len());
    for (key, schema) in properties {
        let Some(value) = value_format(schema, parameters) else {
            return Ok(None);
        };
        values.push((key.clone(), value));
    }
    for key in &required {
        if !properties.contains_key(key) {
            values.push((key.clone(), ValueFormat::Text));
        }
    }
    if values.is_empty() {
        return Ok(Some(const_string("")));
    }
    let mut more = Vec::with_capacity(values.len());
    let mut last = Vec::with_capacity(values.len());
    for (key, value) in &values {
        let (Some(before), Some(final_)) =
            (parameter(key, value, false)?, parameter(key, value, true)?)
        else {
            return Ok(None);
        };
        more.push(before);
        last.push(final_);
    }
    let keys: Vec<&str> = values.iter().map(|(key, _)| key.as_str()).collect();
    let ordered = |order: &[usize]| -> Value {
        let before: Vec<Value> = order
            .iter()
            .map(|&index| {
                if required.contains(&values[index].0) {
                    more[index].clone()
                } else {
                    optional(more[index].clone())
                }
            })
            .collect();
        // Parameters of `order[start..stop]` that end at one of them. Halving keeps the tag
        // O(n log n) in size and O(log n) deep.
        fn last_of(
            order: &[usize],
            before: &[Value],
            last: &[Value],
            start: usize,
            stop: usize,
        ) -> Value {
            if stop - start == 1 {
                return last[order[start]].clone();
            }
            let middle = (start + stop) / 2;
            let mut tail: Vec<Value> = before[start..middle].to_vec();
            tail.push(last_of(order, before, last, middle, stop));
            one_of(vec![
                last_of(order, before, last, start, middle),
                sequence(tail),
            ])
        }
        // The last parameter follows every required one.
        let first = order
            .iter()
            .enumerate()
            .filter(|&(_, &index)| required.contains(&values[index].0))
            .map(|(position, _)| position)
            .max()
            .unwrap_or(0);
        let mut elements: Vec<Value> = before[..first].to_vec();
        elements.push(last_of(order, &before, &last, first, order.len()));
        sequence(elements)
    };
    let declared: Vec<usize> = (0..values.len()).collect();
    let mut orders = vec![ordered(&declared)];
    let mut sorted = declared.clone();
    // Beam also writes arguments sorted by name.
    sorted.sort_by(|&a, &b| keys[a].cmp(keys[b]));
    if sorted != declared {
        orders.push(ordered(&sorted));
    }
    Ok(Some(arguments(one_of(orders), required.is_empty())))
}

/// `body` after the opener of its first parameter.
fn arguments(body: Value, optional_: bool) -> Value {
    let body = sequence(vec![
        regex(SEPARATOR_PATTERN),
        const_string(PARAM_BEGIN),
        body,
    ]);
    if optional_ { optional(body) } else { body }
}

/// Any parameters, each a key and free text.
fn any_parameters() -> Value {
    let any_parameter = |last: bool| {
        tag(
            "",
            sequence(vec![
                regex(KEY_PATTERN),
                const_string(PARAM_MIDDLE),
                value_text(),
            ]),
            &parameter_ends(last),
        )
    };
    arguments(
        sequence(vec![star(any_parameter(false)), any_parameter(true)]),
        true,
    )
}

/// Raw text has no escaping for a call closer, nor, where a value may end before the next
/// opener, for any value end.
fn value_text() -> Value {
    any_text(&[CALL_CLOSERS[0], CALL_CLOSERS[1], PARAM_FOLLOW, PARAM_END])
}

/// Each value but the last ends by opening the next parameter, whose tag starts after the
/// opener.
fn parameter_ends(last: bool) -> Vec<String> {
    if last {
        vec![PARAM_END.to_string()]
    } else {
        vec![
            format!("{PARAM_END}{PARAM_BEGIN}"),
            format!("{PARAM_END}\n{PARAM_BEGIN}"),
            PARAM_FOLLOW.to_string(),
        ]
    }
}

/// The tag of one named parameter, or `None` where a text choice cannot be framed.
fn parameter(key: &str, value: &ValueFormat, last: bool) -> Result<Option<Value>, Unencodable> {
    // The parser must read the key back from `<rfl:parameter name="KEY">`.
    if key.is_empty() || key.contains('"') || CALL_CLOSERS.iter().any(|closer| key.contains(closer))
    {
        return Err(Unencodable);
    }
    let begin = format!("{key}{PARAM_MIDDLE}");
    let ends = parameter_ends(last);
    let content = match value {
        ValueFormat::Text => value_text(),
        ValueFormat::Regex(pattern) => regex(pattern),
        ValueFormat::Json(schema) => json_schema_format(schema),
        ValueFormat::Choices(choices) => {
            // Optional framing whitespace must not change a text choice, and a choice must
            // not end the value or the call early.
            let framed = choices.iter().all(|choice| {
                !CALL_CLOSERS.iter().any(|closer| choice.contains(closer))
                    && !choice.contains(PARAM_END)
                    && !choice.contains(PARAM_FOLLOW)
            });
            if !framed {
                return Ok(None);
            }
            const_strings(&choices.iter().map(String::as_str).collect::<Vec<_>>())
        }
    };
    Ok(Some(tag(&begin, content, &ends)))
}

/// JSON for decoded values and raw text for string values.
fn value_format(schema: &Value, parameters: &Map<String, Value>) -> Option<ValueFormat> {
    if *schema == Value::Bool(false) {
        return None;
    }
    let types = declared_types(schema);
    if !types.is_empty() && !types.contains(&"string") {
        return json_format(json_value_schema(schema, parameters));
    }
    text_format(schema)
}

fn json_value_schema(schema: &Value, parameters: &Map<String, Value>) -> Value {
    let Value::Object(schema) = schema else {
        return json!({});
    };
    let mut merged = Map::new();
    for key in ["$defs", "definitions"] {
        if let Some(definitions) = parameters.get(key) {
            merged.insert(key.to_string(), definitions.clone());
        }
    }
    if schema.get("nullable").is_some_and(truthy) {
        // XGrammar ignores `nullable`.
        merged.insert(
            "anyOf".to_string(),
            json!([Value::Object(schema.clone()), {"type": "null"}]),
        );
    } else {
        for (key, value) in schema {
            merged.insert(key.clone(), value.clone());
        }
    }
    Value::Object(merged)
}

/// JSON unless the schema is unsupported. (No call closer is ambiguous with JSON syntax.)
fn json_format(schema: Value) -> Option<ValueFormat> {
    xgrammar_compiles(&schema).then_some(ValueFormat::Json(schema))
}

/// Whether XGrammar is expected to compile `schema`: every keyword is one it accepts and
/// every reference is local.
fn xgrammar_compiles(schema: &Value) -> bool {
    match schema {
        Value::Bool(_) => true,
        Value::Object(schema) => schema.iter().all(|(key, value)| {
            if ANNOTATIONS.contains(&key.as_str()) {
                return true;
            }
            if !XGRAMMAR_JSON_KEYWORDS.contains(&key.as_str()) {
                return false;
            }
            match key.as_str() {
                "$ref" => value
                    .as_str()
                    .is_some_and(|reference| reference.starts_with("#/")),
                "properties" | "$defs" | "definitions" => match value {
                    Value::Object(entries) => entries.values().all(xgrammar_compiles),
                    _ => false,
                },
                "items" | "additionalProperties" => match value {
                    Value::Array(items) => items.iter().all(xgrammar_compiles),
                    other => xgrammar_compiles(other),
                },
                "prefixItems" | "anyOf" | "oneOf" | "allOf" => match value {
                    Value::Array(items) => items.iter().all(xgrammar_compiles),
                    _ => false,
                },
                _ => true,
            }
        }),
        _ => false,
    }
}

/// Constrain raw text without changing its parsed value.
fn text_format(schema: &Value) -> Option<ValueFormat> {
    if *schema == Value::Bool(true) {
        return Some(ValueFormat::Text);
    }
    let Value::Object(object) = schema else {
        return None;
    };
    if object.contains_key("format") || object.contains_key("pattern") {
        return string_regex(object);
    }
    const ALLOWED: &[&str] = &["type", "nullable", "enum", "const", "anyOf", "oneOf"];
    if object
        .keys()
        .any(|key| !ANNOTATIONS.contains(&key.as_str()) && !ALLOWED.contains(&key.as_str()))
    {
        return None;
    }
    let mut branches: Vec<&Value> = Vec::new();
    for keyword in ["anyOf", "oneOf"] {
        match object.get(keyword) {
            Some(Value::Array(items)) => branches.extend(items),
            Some(other) if truthy(other) => return None,
            _ => {}
        }
    }
    if object.contains_key("enum") || object.contains_key("const") {
        let choices = string_choices(object)?;
        if !branches.is_empty() || declared_types(schema).iter().any(|t| *t != "string") {
            return None;
        }
        return Some(ValueFormat::Choices(choices));
    }
    if branches
        .iter()
        .any(|branch| text_format(branch) != Some(ValueFormat::Text))
    {
        return None;
    }
    Some(ValueFormat::Text)
}

/// Hold a known string `format` of a raw string. Like XGrammar, a format overrides a pattern
/// and drops length bounds. A bare `pattern` is left to free text (see the module docs).
fn string_regex(schema: &Map<String, Value>) -> Option<ValueFormat> {
    const ALLOWED: &[&str] = &["type", "format", "pattern", "minLength", "maxLength"];
    if schema
        .keys()
        .any(|key| !ANNOTATIONS.contains(&key.as_str()) && !ALLOWED.contains(&key.as_str()))
        || declared_types(&Value::Object(schema.clone())) != ["string"]
    {
        return None;
    }
    let format = schema.get("format")?.as_str()?;
    string_format_regex(format).map(ValueFormat::Regex)
}

/// Raw string forms of the formats XGrammar holds in JSON strings, as the serving layer
/// spells them.
fn string_format_regex(format: &str) -> Option<String> {
    const HOST_LABEL: &str = r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?";
    const DATE: &str = r"[0-9]{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12][0-9]|3[01])";
    const TIME: &str = r"(?:[01][0-9]|2[0-3]):[0-5][0-9]:(?:[0-5][0-9]|60)(?:\.[0-9]+)?(?:Z|[+-](?:[01][0-9]|2[0-3]):[0-5][0-9])";
    const OCTET: &str = r"(?:25[0-5]|2[0-4][0-9]|[01]?[0-9]?[0-9])";
    const ATOM: &str = r"[A-Za-z0-9_!#$%&'*+/=?^`{|}~-]+";
    const URI_CHARS: &str = r"A-Za-z0-9_.~!$&'()*+,;=";
    const ESCAPED: &str = r"%[0-9A-Fa-f]{2}";
    let path = format!(r"(?:[{URI_CHARS}:@-]|{ESCAPED})");
    let query = format!(r"(?:[{URI_CHARS}:@/?-]|{ESCAPED})*");
    Some(match format {
        "date" => DATE.to_string(),
        "time" => TIME.to_string(),
        "date-time" => format!("{DATE}T{TIME}"),
        "email" => format!(
            r"{ATOM}(?:\.{ATOM})*@[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?(?:\.[A-Za-z0-9][A-Za-z0-9-]*[A-Za-z0-9])*"
        ),
        "hostname" => format!(r"{HOST_LABEL}(?:\.{HOST_LABEL})*"),
        "ipv4" => format!(r"(?:{OCTET}\.){{3}}{OCTET}"),
        "uuid" => r"[0-9A-Fa-f]{8}(?:-[0-9A-Fa-f]{4}){3}-[0-9A-Fa-f]{12}".to_string(),
        "uri" => format!(
            r"[A-Za-z][A-Za-z+.-]*:(?://(?:(?:[{URI_CHARS}:-]|{ESCAPED})*@)?(?:[{URI_CHARS}-]|{ESCAPED})*(?::[0-9]*)?(?:/{path}*)*|/?(?:{path}+(?:/{path}*)*)?)(?:\?{query})?(?:#{query})?"
        ),
        _ => return None,
    })
}

/// Whether the parser reads schema values with their JSON type.
fn typed_by_parser(schema: &Value) -> bool {
    if *schema == Value::Bool(true) || !declared_types(schema).is_empty() {
        return true;
    }
    match schema {
        Value::Object(object) => {
            object.keys().all(|key| ANNOTATIONS.contains(&key.as_str()))
                || string_choices(object).is_some()
        }
        _ => false,
    }
}

/// The `const` or `enum` string choices of a schema.
fn string_choices(schema: &Map<String, Value>) -> Option<Vec<String>> {
    let choices: Vec<&Value> = match schema.get("const") {
        Some(value) => vec![value],
        None => match schema.get("enum") {
            Some(Value::Array(items)) => items.iter().collect(),
            _ => return None,
        },
    };
    if choices.is_empty() {
        return None;
    }
    choices
        .iter()
        .map(|choice| choice.as_str().map(str::to_string))
        .collect()
}

/// The types a schema declares, as the serving layer's response parser reads them: the
/// `type`, the types of `anyOf`/`oneOf` branches, and `null` for `nullable`.
fn declared_types(schema: &Value) -> Vec<&str> {
    let Value::Object(object) = schema else {
        return Vec::new();
    };
    let mut types = Vec::new();
    match object.get("type") {
        Some(Value::String(declared)) => types.push(declared.as_str()),
        Some(Value::Array(declared)) => types.extend(declared.iter().filter_map(Value::as_str)),
        _ => {}
    }
    for keyword in ["anyOf", "oneOf"] {
        if let Some(Value::Array(choices)) = object.get(keyword) {
            for choice in choices {
                types.extend(declared_types(choice));
            }
        }
    }
    if object.get("nullable").is_some_and(truthy) && !types.contains(&"null") {
        types.push("null");
    }
    types
}

/// Resolve annotated aliases and infer the types needed for argument decoding.
fn typed_parameters(parameters: &Value) -> Map<String, Value> {
    let parameters = match parameters {
        // Like the generic JSON schema, a tool without parameters takes none.
        Value::Object(object) if !object.is_empty() => object.clone(),
        _ => {
            let mut object = Map::new();
            object.insert("type".to_string(), json!("object"));
            object.insert("properties".to_string(), json!({}));
            object
        }
    };
    let Some(Value::Object(properties)) = parameters.get("properties") else {
        return parameters;
    };
    let typed: Map<String, Value> = properties
        .iter()
        .map(|(key, schema)| (key.clone(), typed(schema, &parameters, &[])))
        .collect();
    let mut parameters = parameters;
    parameters.insert("properties".to_string(), Value::Object(typed));
    parameters
}

fn typed(schema: &Value, root: &Map<String, Value>, seen: &[&str]) -> Value {
    let Value::Object(object) = schema else {
        return schema.clone();
    };
    let aliased: Vec<&str> = ["$ref", "allOf"]
        .into_iter()
        .filter(|key| object.contains_key(*key))
        .collect();
    let rest: Map<String, Value> = object
        .iter()
        .filter(|(key, _)| !aliased.contains(&key.as_str()))
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect();
    let mut target: Option<&Value> = None;
    let mut seen: Vec<&str> = seen.to_vec();
    if aliased.len() == 1
        && rest
            .keys()
            .all(|key| ANNOTATIONS.contains(&key.as_str()) || key == "nullable")
    {
        match (object.get("$ref"), object.get("allOf")) {
            (Some(Value::String(reference)), _) if !seen.contains(&reference.as_str()) => {
                target = resolve_local_ref(reference, root);
                seen.push(reference);
            }
            (_, Some(Value::Array(branches))) if branches.len() == 1 => {
                target = Some(&branches[0]);
            }
            _ => {}
        }
    }
    if let Some(Value::Object(target)) = target {
        let mut merged = target.clone();
        for (key, value) in rest {
            merged.insert(key, value);
        }
        return typed(&Value::Object(merged), root, &seen);
    }
    let mut object = object.clone();
    for keyword in ["anyOf", "oneOf"] {
        if let Some(Value::Array(items)) = object.get(keyword) {
            let items: Vec<Value> = items.iter().map(|item| typed(item, root, &seen)).collect();
            object.insert(keyword.to_string(), Value::Array(items));
        }
    }
    if !["type", "anyOf", "oneOf"]
        .iter()
        .any(|key| object.contains_key(*key))
    {
        let types = inferred_types(&Value::Object(object.clone()), root, &[]);
        if !types.is_empty() && types.len() < JSON_TYPES.len() {
            let declared = if types.len() == 1 {
                json!(types[0])
            } else {
                json!(types)
            };
            object.insert("type".to_string(), declared);
        }
    }
    Value::Object(object)
}

fn resolve_local_ref<'a>(reference: &str, root: &'a Map<String, Value>) -> Option<&'a Value> {
    let path = reference.strip_prefix("#/")?;
    let mut value: &Value = &Value::Null;
    let mut current: Option<&Map<String, Value>> = Some(root);
    for part in path.split('/') {
        let key = part.replace("~1", "/").replace("~0", "~");
        value = current?.get(&key)?;
        current = value.as_object();
    }
    matches!(value, Value::Bool(_) | Value::Object(_)).then_some(value)
}

/// The types the serving layer infers for a schema from its keywords
/// (`kimik3._schema_types`), in `JSON_TYPES` order; all of them when nothing constrains
/// the type.
fn inferred_types(schema: &Value, root: &Map<String, Value>, seen: &[&str]) -> Vec<&'static str> {
    let all = || JSON_TYPES.to_vec();
    let in_order = |allowed: &dyn Fn(&str) -> bool| -> Vec<&'static str> {
        JSON_TYPES.iter().copied().filter(|t| allowed(t)).collect()
    };
    let object = match schema {
        Value::Bool(false) => return Vec::new(),
        Value::Bool(true) => return all(),
        Value::Object(object) => object,
        _ => return all(),
    };
    if let Some(Value::String(reference)) = object.get("$ref")
        && !seen.contains(&reference.as_str())
        && let Some(target) = resolve_local_ref(reference, root)
    {
        let mut seen = seen.to_vec();
        seen.push(reference);
        return inferred_types(target, root, &seen);
    }
    match object.get("type") {
        Some(Value::String(declared)) => {
            return if JSON_TYPES.contains(&declared.as_str()) {
                in_order(&|t| t == declared)
            } else {
                all()
            };
        }
        Some(Value::Array(declared)) => {
            return in_order(&|t| declared.iter().any(|d| d.as_str() == Some(t)));
        }
        _ => {}
    }
    for keyword in ["anyOf", "oneOf"] {
        if let Some(Value::Array(options)) = object.get(keyword) {
            let mut union: Vec<&str> = Vec::new();
            for option in options {
                if matches!(option, Value::Bool(_) | Value::Object(_)) {
                    union.extend(inferred_types(option, root, seen));
                }
            }
            return in_order(&|t| union.contains(&t));
        }
    }
    if let Some(Value::Array(options)) = object.get("allOf") {
        let sets: Vec<Vec<&str>> = options
            .iter()
            .filter(|option| matches!(option, Value::Bool(_) | Value::Object(_)))
            .map(|option| inferred_types(option, root, seen))
            .filter(|types| types.len() != JSON_TYPES.len())
            .collect();
        if let Some((first, others)) = sets.split_first() {
            fn widen<'a>(types: &[&'a str]) -> Vec<&'a str> {
                let mut widened = types.to_vec();
                if types.contains(&"number") && !widened.contains(&"integer") {
                    widened.push("integer");
                }
                widened
            }
            let mut result = widen(first);
            for types in others {
                let types = widen(types);
                result.retain(|t| types.contains(t));
            }
            // number survives the intersection only if every branch allows it, making the
            // widened integer redundant rather than the other way.
            if result.contains(&"number") && result.contains(&"integer") {
                result.retain(|t| *t != "integer");
            }
            return in_order(&|t| result.contains(&t));
        }
    }
    if let Some(constant) = object.get("const") {
        return vec![json_type(constant)];
    }
    if let Some(Value::Array(choices)) = object.get("enum") {
        let found: Vec<&str> = choices.iter().map(json_type).collect();
        return in_order(&|t| found.contains(&t));
    }
    let has_any = |keywords: &[&str]| keywords.iter().any(|key| object.contains_key(*key));
    if has_any(INFER_OBJECT_KEYWORDS) {
        return vec!["object"];
    }
    if has_any(INFER_ARRAY_KEYWORDS) {
        return vec!["array"];
    }
    if has_any(INFER_STRING_KEYWORDS) {
        return vec!["string"];
    }
    if has_any(INFER_NUMBER_KEYWORDS) {
        return vec!["number"];
    }
    all()
}

fn json_type(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(number) if number.is_f64() => "number",
        Value::Number(_) => "integer",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

/// Python truthiness of a JSON value, which the serving layer's checks rely on.
fn truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(flag) => *flag,
        Value::Number(number) => number.as_f64() != Some(0.0),
        Value::String(text) => !text.is_empty(),
        Value::Array(items) => !items.is_empty(),
        Value::Object(entries) => !entries.is_empty(),
    }
}

// Structural-tag format primitives, emitted as the xgrammar JSON wire format directly:
// this grammar uses `star`, `optional`, `regex` and multi-ended `tag` nodes that the
// shared `wire` module does not model.

fn sequence(elements: Vec<Value>) -> Value {
    json!({"type": "sequence", "elements": elements})
}

fn one_of(mut formats: Vec<Value>) -> Value {
    if formats.len() == 1 {
        formats.pop().expect("one format")
    } else {
        json!({"type": "or", "elements": formats})
    }
}

fn optional(content: Value) -> Value {
    json!({"type": "optional", "content": content})
}

fn star(content: Value) -> Value {
    json!({"type": "star", "content": content})
}

fn tag(begin: &str, content: Value, end: &[impl AsRef<str>]) -> Value {
    let end: Vec<&str> = end.iter().map(AsRef::as_ref).collect();
    json!({"type": "tag", "begin": begin, "content": content, "end": end})
}

fn const_string(value: &str) -> Value {
    json!({"type": "const_string", "value": value})
}

fn const_strings(values: &[&str]) -> Value {
    one_of(values.iter().map(|value| const_string(value)).collect())
}

fn regex(pattern: &str) -> Value {
    json!({"type": "regex", "pattern": pattern})
}

fn any_text(excludes: &[&str]) -> Value {
    json!({"type": "any_text", "excludes": excludes})
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structural_tag::StructuralTagSchemaMode;

    /// Each fixture, as the serving layer's `BeamDetector.get_structural_tag` produced it
    /// for the tools, `tool_choice` and `parallel_tool_calls` recorded beside it in
    /// `tests/data/structural_tags/beam/README.index.json`.
    macro_rules! fixture {
        ($name:literal) => {
            serde_json::from_str::<Value>(include_str!(concat!(
                "../../../tests/data/structural_tags/beam/",
                $name,
                ".json"
            )))
            .expect("fixture is JSON")
        };
    }

    fn tool(name: &str, parameters: Option<Value>, strict: bool) -> Tool {
        Tool {
            name: name.to_string(),
            description: None,
            parameters: parameters.unwrap_or(Value::Null),
            strict: Some(strict),
        }
    }

    fn object(properties: Value, required: Option<Vec<&str>>) -> Value {
        let mut schema = json!({"type": "object", "properties": properties});
        if let Some(required) = required {
            schema["required"] = json!(required);
        }
        schema
    }

    fn weather() -> Tool {
        tool(
            "get_weather",
            Some(object(
                json!({"location": {"type": "string"}}),
                Some(vec!["location"]),
            )),
            false,
        )
    }

    fn noargs() -> Tool {
        tool("get_time", Some(object(json!({}), None)), false)
    }

    fn rich() -> Tool {
        tool(
            "book_table",
            Some(object(
                json!({
                    "city": {"type": "string", "description": "City"},
                    "guests": {"type": "integer"},
                    "budget": {"type": "number", "minimum": 0},
                    "outdoor": {"type": "boolean"},
                    "cuisine": {"type": "string", "enum": ["thai", "italian", "sushi"]},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "window": {"type": "object", "properties": {"start": {"type": "string"}, "end": {"type": "string"}}, "required": ["start"]},
                    "notes": {"type": "string"}
                }),
                Some(vec!["city", "guests"]),
            )),
            false,
        )
    }

    fn triangle() -> Tool {
        tool(
            "triangle_properties_get",
            Some(object(
                json!({
                    "side1": {"type": "integer"}, "side2": {"type": "integer"}, "side3": {"type": "integer"},
                    "get_area": {"type": "boolean"}, "get_perimeter": {"type": "boolean"}, "get_angles": {"type": "boolean"}
                }),
                Some(vec!["side1", "side2", "side3"]),
            )),
            false,
        )
    }

    fn loose_maxlength() -> Tool {
        tool(
            "short_text",
            Some(object(
                json!({"q": {"type": "string", "maxLength": 10}}),
                Some(vec!["q"]),
            )),
            false,
        )
    }

    fn turn(
        tools: &[Tool],
        tool_choice: StructuralTagToolChoice<'_>,
        parallel: bool,
    ) -> Option<Value> {
        beam_turn_structural_tag(
            tools,
            tool_choice,
            Some(parallel),
            PromptChannel::MessageStart,
            None,
        )
    }

    fn turn_json(
        tools: &[Tool],
        tool_choice: StructuralTagToolChoice<'_>,
        parallel: bool,
        schema: &Value,
    ) -> Option<Value> {
        beam_turn_structural_tag(
            tools,
            tool_choice,
            Some(parallel),
            PromptChannel::MessageStart,
            Some(schema),
        )
    }

    use StructuralTagToolChoice::{Auto, Named, None as NoTools, Required};

    #[test]
    fn plain_chat_and_tool_choice_none_hold_reasoning_then_the_answer() {
        assert_eq!(turn(&[], Auto, true).unwrap(), fixture!("chat_no_tools"));
        assert_eq!(
            turn(&[weather()], NoTools, true).unwrap(),
            fixture!("none_with_tools")
        );
    }

    #[test]
    fn automatic_choice_offers_held_calls_undeclared_calls_and_the_answer() {
        assert_eq!(
            turn(&[weather()], Auto, true).unwrap(),
            fixture!("auto_weather")
        );
        assert_eq!(
            turn(&[weather(), noargs()], Auto, true).unwrap(),
            fixture!("auto_weather_two")
        );
        assert_eq!(
            turn(&[weather()], Auto, false).unwrap(),
            fixture!("auto_weather_noparallel")
        );
    }

    #[test]
    fn forced_choices_drop_the_answer_and_named_choices_drop_parallel_calls() {
        assert_eq!(
            turn(&[weather()], Required, true).unwrap(),
            fixture!("required_weather")
        );
        assert_eq!(
            turn(&[weather()], Required, false).unwrap(),
            fixture!("required_weather_noparallel")
        );
        assert_eq!(
            turn(&[weather(), noargs()], Required, true).unwrap(),
            fixture!("required_weather_two")
        );
        assert_eq!(
            turn(&[weather(), noargs()], Named("get_weather"), true).unwrap(),
            fixture!("named_weather_of_two")
        );
        assert_eq!(
            turn(&[rich(), noargs(), loose_maxlength()], Required, true).unwrap(),
            fixture!("required_mixed_three")
        );
    }

    #[test]
    fn typed_arguments_come_in_declared_or_sorted_order() {
        assert_eq!(turn(&[rich()], Auto, true).unwrap(), fixture!("auto_rich"));
        assert_eq!(
            turn(&[rich()], Required, false).unwrap(),
            fixture!("required_rich_noparallel")
        );
        assert_eq!(
            turn(&[triangle()], Auto, true).unwrap(),
            fixture!("auto_triangle")
        );
        let sorted_same = tool(
            "alpha",
            Some(object(
                json!({"a": {"type": "string"}, "b": {"type": "integer"}}),
                Some(vec!["a"]),
            )),
            false,
        );
        assert_eq!(
            turn(&[sorted_same], Auto, true).unwrap(),
            fixture!("auto_sorted_same")
        );
        let optional_only = tool(
            "list_items",
            Some(object(
                json!({"limit": {"type": "integer"}, "cursor": {"type": "string"}}),
                None,
            )),
            false,
        );
        assert_eq!(
            turn(&[optional_only], Auto, true).unwrap(),
            fixture!("auto_optional_only")
        );
        let required_extra = tool(
            "send",
            Some(object(
                json!({"to": {"type": "string"}}),
                Some(vec!["to", "body"]),
            )),
            false,
        );
        assert_eq!(
            turn(&[required_extra], Auto, true).unwrap(),
            fixture!("auto_required_extra")
        );
    }

    #[test]
    fn schema_shapes_type_as_the_serving_layer_does() {
        let nullable_defs = tool(
            "configure",
            Some(json!({
                "type": "object",
                "$defs": {"Level": {"type": "integer", "minimum": 0, "maximum": 3}},
                "properties": {
                    "level": {"$ref": "#/$defs/Level"},
                    "retries": {"type": "integer", "nullable": true},
                    "mode": {"enum": ["fast", "slow"]},
                    "kind": {"const": "primary"},
                    "ratio": {"anyOf": [{"type": "number"}, {"type": "null"}]}
                },
                "required": ["level"]
            })),
            false,
        );
        assert_eq!(
            turn(&[nullable_defs], Auto, true).unwrap(),
            fixture!("auto_nullable_defs")
        );
        let anyof = tool(
            "find",
            Some(object(
                json!({"key": {"anyOf": [{"type": "integer"}, {"type": "string"}]}}),
                Some(vec!["key"]),
            )),
            false,
        );
        assert_eq!(
            turn(&[anyof], Auto, true).unwrap(),
            fixture!("auto_anyof_str_int")
        );
        let int_enum = tool(
            "set_level",
            Some(object(
                json!({"level": {"type": "integer", "enum": [1, 2, 3]}}),
                Some(vec!["level"]),
            )),
            false,
        );
        assert_eq!(
            turn(&[int_enum], Auto, true).unwrap(),
            fixture!("auto_int_enum")
        );
        let string_only = tool(
            "echo",
            Some(object(
                json!({"text": {}, "note": {"description": "free"}}),
                Some(vec!["text"]),
            )),
            false,
        );
        assert_eq!(
            turn(&[string_only], Auto, true).unwrap(),
            fixture!("auto_string_only_props")
        );
        let newline_enum = tool(
            "pick",
            Some(object(
                json!({"style": {"type": "string", "enum": ["plain", "two\nlines", " padded "]}}),
                Some(vec!["style"]),
            )),
            false,
        );
        assert_eq!(
            turn(&[newline_enum], Auto, true).unwrap(),
            fixture!("auto_newline_enum")
        );
    }

    #[test]
    fn tools_without_arguments_take_none() {
        assert_eq!(
            turn(&[tool("ping", None, false)], Auto, true).unwrap(),
            fixture!("auto_no_parameters")
        );
        let closed = tool(
            "nothing",
            Some(json!({"type": "object", "additionalProperties": false})),
            false,
        );
        assert_eq!(
            turn(&[closed], Auto, true).unwrap(),
            fixture!("auto_props_none_additional_false")
        );
    }

    #[test]
    fn unsupported_schemas_fall_back_to_loose_arguments() {
        // Every loose single-tool automatic tag is the same: the call names any tool.
        let loose = fixture!("auto_loose_additional");
        let additional = tool(
            "free_form",
            Some(
                json!({"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"], "additionalProperties": true}),
            ),
            false,
        );
        assert_eq!(turn(&[additional], Auto, true).unwrap(), loose);
        assert_eq!(
            turn(&[loose_maxlength()], Auto, true).unwrap(),
            fixture!("auto_loose_maxlength")
        );
        let quote_param = tool(
            "odd",
            Some(object(
                json!({"na\"me": {"type": "string"}}),
                Some(vec!["na\"me"]),
            )),
            false,
        );
        assert_eq!(
            turn(&[quote_param], Auto, true).unwrap(),
            fixture!("auto_quote_param")
        );
        // An unknown string format leaves the tool loose, as in the serving layer.
        let formats = tool(
            "schedule",
            Some(object(
                json!({
                    "on": {"type": "string", "format": "date"}, "email": {"type": "string", "format": "email"},
                    "id": {"type": "string", "format": "uuid"}, "label": {"type": "string", "format": "color"}
                }),
                Some(vec!["on"]),
            )),
            false,
        );
        assert_eq!(
            turn(&[formats], Auto, true).unwrap(),
            fixture!("auto_formats")
        );
        // Deviation: the serving layer holds this `pattern` as a regex; this port leaves it loose.
        let pattern = tool(
            "lookup",
            Some(object(
                json!({"code": {"type": "string", "pattern": "^[A-Z]{3}[0-9]{2}$"}}),
                Some(vec!["code"]),
            )),
            false,
        );
        let ours = turn(&[pattern], Auto, true).unwrap();
        assert_eq!(ours, loose);
        assert_ne!(ours, fixture!("auto_pattern"));
    }

    #[test]
    fn known_string_formats_are_held_as_regexes() {
        let dated = tool(
            "schedule",
            Some(object(
                json!({"on": {"type": "string", "format": "date"}}),
                Some(vec!["on"]),
            )),
            false,
        );
        let tag = turn(&[dated], Auto, true).unwrap();
        let text = serde_json::to_string(&tag).unwrap();
        assert!(text.contains(
            r#"{"type":"regex","pattern":"[0-9]{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12][0-9]|3[01])"}"#
        ));
        assert!(
            text.contains(r#"" to=schedule<"#),
            "a held tool is named in its call"
        );
    }

    #[test]
    fn unencodable_names_and_unholdable_strict_tools_derive_no_grammar() {
        let spaced = tool(
            "bad name",
            Some(object(json!({"q": {"type": "string"}}), Some(vec!["q"]))),
            false,
        );
        assert!(turn(&[spaced], Auto, true).is_none());
        let strict_short = tool(
            "strict_short",
            Some(object(
                json!({"q": {"type": "string", "maxLength": 10}}),
                Some(vec!["q"]),
            )),
            true,
        );
        assert!(turn(&[strict_short, weather()], Auto, true).is_none());
        // A strict tool the arguments can hold is fine.
        let strict_weather = tool(
            "get_weather",
            Some(object(
                json!({"location": {"type": "string"}}),
                Some(vec!["location"]),
            )),
            true,
        );
        assert_eq!(
            turn(&[strict_weather], Auto, true).unwrap(),
            fixture!("auto_weather")
        );
        // A forced choice needs a tool.
        assert!(turn(&[], Required, true).is_none());
    }

    #[test]
    fn open_channels_hold_only_the_rest_of_the_message() {
        let full = fixture!("auto_weather");
        let rest = full["format"]["elements"][1].clone();
        let reasoning =
            beam_turn_structural_tag(&[weather()], Auto, None, PromptChannel::OpenReasoning, None)
                .unwrap();
        assert_eq!(
            reasoning["format"],
            json!({"type": "sequence", "elements": [
                {"type": "any_text", "excludes": ["<|end|><|start|>assistant"]},
                {"type": "const_string", "value": "<|end|><|start|>assistant"},
                rest
            ]})
        );
        let answer =
            beam_turn_structural_tag(&[weather()], Auto, None, PromptChannel::OpenAnswer, None)
                .unwrap();
        assert_eq!(
            answer["format"],
            json!({"type": "sequence", "elements": [
                {"type": "any_text", "excludes": ["<|return|>", "<|end|>", "<|call|>"]},
                {"type": "optional", "content": {"type": "or", "elements": [
                    {"type": "const_string", "value": "<|return|>"},
                    {"type": "const_string", "value": "<|end|>"},
                    {"type": "const_string", "value": "<|call|>"}
                ]}}
            ]})
        );
        assert!(
            beam_turn_structural_tag(
                &[weather()],
                Required,
                None,
                PromptChannel::OpenAnswer,
                None
            )
            .is_none(),
            "no call can be forced once the final channel is open"
        );
    }

    #[test]
    fn the_registered_builder_builds_the_turn_and_the_json_answer() {
        let tools = [weather()];
        let context = StructuralTagContext {
            tool_choice: Auto,
            tools: &tools,
            parallel_tool_calls: None,
            schema_mode: StructuralTagSchemaMode::Auto,
            structured_output_schema: None,
            starts_in_reasoning: false,
        };
        assert!(BEAM.constrains_every_turn());
        assert_eq!(
            BEAM.build(&context).unwrap().unwrap(),
            fixture!("auto_weather")
        );
        assert_eq!(
            BEAM.build_turn(&context, PromptChannel::MessageStart)
                .unwrap()
                .unwrap(),
            fixture!("auto_weather")
        );
        let schema = json!({"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]});
        let structured = StructuralTagContext {
            structured_output_schema: Some(&schema),
            ..context
        };
        assert_eq!(
            BEAM.build(&structured).unwrap().unwrap(),
            fixture!("json_weather_auto")
        );
        assert_eq!(
            crate::tool_calling::structural_tag_builder_for_family("beam")
                .expect("beam registers its builder")
                .build(&context)
                .unwrap()
                .unwrap(),
            fixture!("auto_weather")
        );
    }

    #[test]
    fn json_answers_are_framed_in_the_final_channel() {
        let ok = json!({"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]});
        let any_object = json!({"type": "object"});
        assert_eq!(
            turn_json(&[], Auto, true, &ok).unwrap(),
            fixture!("json_no_tools")
        );
        assert_eq!(
            turn_json(&[], Auto, true, &any_object).unwrap(),
            fixture!("json_object_no_tools")
        );
        assert_eq!(
            turn_json(&[weather()], NoTools, true, &ok).unwrap(),
            fixture!("json_no_tools")
        );
        assert_eq!(
            turn_json(&[weather()], Auto, true, &ok).unwrap(),
            fixture!("json_weather_auto")
        );
        assert_eq!(
            turn_json(&[weather(), noargs()], Auto, true, &ok).unwrap(),
            fixture!("json_weather_two_auto")
        );
        assert_eq!(
            turn_json(&[weather()], Auto, false, &ok).unwrap(),
            fixture!("json_weather_noparallel")
        );
        let three_sides = tool(
            "triangle_properties_get",
            Some(object(
                json!({"side1": {"type": "integer"}, "side2": {"type": "integer"}, "side3": {"type": "integer"}}),
                Some(vec!["side1", "side2", "side3"]),
            )),
            false,
        );
        assert_eq!(
            turn_json(&[three_sides], Auto, true, &ok).unwrap(),
            fixture!("json_triangle_auto")
        );
        // A forced call has no answer, so the schema does not apply.
        assert_eq!(
            turn_json(&[weather()], Required, true, &ok).unwrap(),
            fixture!("required_weather")
        );
        // With the final channel already open only the document remains.
        let rest = beam_turn_structural_tag(
            &[weather()],
            Auto,
            None,
            PromptChannel::OpenAnswer,
            Some(&ok),
        )
        .unwrap();
        assert_eq!(
            rest["format"],
            json!({"type": "sequence", "elements": [{"type": "json_schema", "json_schema": ok}]})
        );
    }

    #[test]
    fn held_tools_are_the_ones_the_grammar_types() {
        assert!(beam_holds_tool(&weather()));
        assert!(beam_holds_tool(&rich()));
        assert!(beam_holds_tool(&noargs()));
        assert!(!beam_holds_tool(&loose_maxlength()));
        let additional = tool(
            "free_form",
            Some(
                json!({"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"], "additionalProperties": true}),
            ),
            false,
        );
        assert!(!beam_holds_tool(&additional));
    }

    #[test]
    fn other_names_spell_every_name_but_the_held_ones() {
        assert_eq!(other_names(&[]), NAME_PATTERN);
        let pattern = other_names(&["ab", "a"]);
        assert_eq!(
            pattern,
            r"(?:[^\s<a][^\s<]*|a(?:[^\s<b][^\s<]*|b(?:[^\s<][^\s<]*)))"
        );
        let re = regex::Regex::new(&format!("^{pattern}$")).unwrap();
        for other in ["b", "abc", "aa", "ba", "x"] {
            assert!(re.is_match(other), "{other:?} is another name");
        }
        for held in ["a", "ab"] {
            assert!(!re.is_match(held), "{held:?} is held");
        }
        assert!(!re.is_match("a b") && !re.is_match("a<b"));
        // The two-name trie the serving layer derives for the `auto_weather_two` fixture.
        assert_eq!(
            other_names(&["get_weather", "get_time"]),
            fixture!("auto_weather_two")["format"]["elements"][1]["elements"][0]["elements"][0]
                ["elements"][2]["elements"][0]["pattern"]
                .as_str()
                .unwrap()
                .strip_prefix(" to=(?:")
                .unwrap()
                .strip_suffix(&format!("){CALL_OPEN_AFTER}"))
                .unwrap()
        );
    }
}
