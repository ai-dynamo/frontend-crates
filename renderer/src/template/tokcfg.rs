// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//based on: https://github.com/EricLBuehler/mistral.rs/blob/d970bb5feb863acf8e8ec90de97e18221fb959f1/mistralrs-core/src/pipeline/chat_template.rs

use std::collections::HashMap;

use chrono::{DateTime, Local};
use either::Either;
use minijinja::{Error, ErrorKind, Output, State, Value, escape_formatter, value::Kwargs};
use serde::{Deserialize, Serialize};

use crate::python::{PyJsonFormatter, python_float_repr};

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
pub struct AddedTokensDecoder {
    __type: Option<String>,
    pub content: String,
    lstrip: bool,
    normalized: bool,
    rstrip: bool,
    single_word: bool,
    special: Option<bool>,
}

pub fn raise_exception(msg: String) -> Result<String, minijinja::Error> {
    Err(minijinja::Error::new(ErrorKind::InvalidOperation, msg))
}

#[derive(Debug, Deserialize)]
pub struct BeginEndUnkTok(
    #[serde(with = "either::serde_untagged")] pub Either<String, AddedTokensDecoder>,
);

/// Support older tool use patterns where the tool use template was separate from the default/chat template.
/// Modern patterns use a single template with a `tool_use` key, e.g.
///
/// ```jinja
/// {%- if tools is not none and tool_choice is not none %}
/// ```
#[derive(Debug, Deserialize)]
pub struct ChatTemplateValue(
    #[serde(with = "either::serde_untagged")] pub Either<String, Vec<HashMap<String, String>>>,
);

/// If present, pad_token is usually a single value. Deepseek R1 and it's distill's use a map.
#[allow(dead_code)]
#[derive(Debug, Deserialize)]
pub struct PadTokenValue(
    #[serde(with = "either::serde_untagged")] pub Either<String, AddedTokensDecoder>,
);

#[allow(dead_code)]
#[derive(Debug, Deserialize, Default)]
/// Template for chat models including bos/eos/unk as well as the chat template.
pub struct ChatTemplate {
    pub bos_token: Option<BeginEndUnkTok>,
    pub eos_token: Option<BeginEndUnkTok>,
    pub unk_token: Option<BeginEndUnkTok>,

    /// Jinja format [chat templating] for chat completion.
    ///
    /// [chat templating]: https://huggingface.co/docs/transformers/chat_templating
    pub chat_template: Option<ChatTemplateValue>,

    // future
    add_bos_token: Option<bool>,
    add_eos_token: Option<bool>,
    added_tokens_decoder: Option<HashMap<String, AddedTokensDecoder>>,
    additional_special_tokens: Option<Vec<String>>,
    clean_up_tokenization_spaces: Option<bool>,
    device_map: Option<String>,
    legacy: Option<bool>,
    model_max_length: Option<f64>,
    pad_token: Option<PadTokenValue>,
    sp_model_kwargs: Option<HashMap<String, String>>,
    spaces_between_special_tokens: Option<bool>,
    tokenizer_class: Option<String>,
    truncation_size: Option<String>,
    use_default_system_prompt: Option<bool>,
}

impl ChatTemplate {
    pub fn eos_tok(&self) -> Option<String> {
        match self.eos_token.as_ref()?.0 {
            Either::Left(ref lit) => Some(lit.clone()),
            Either::Right(ref added) => Some(added.content.clone()),
        }
    }

    pub fn bos_tok(&self) -> Option<String> {
        match self.bos_token.as_ref()?.0 {
            Either::Left(ref lit) => Some(lit.clone()),
            Either::Right(ref added) => Some(added.content.clone()),
        }
    }

    pub fn unk_tok(&self) -> Option<String> {
        match self.unk_token.as_ref()?.0 {
            Either::Left(ref lit) => Some(lit.clone()),
            Either::Right(ref added) => Some(added.content.clone()),
        }
    }
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
pub struct GenerationConfig {
    #[serde(with = "either::serde_untagged")]
    bos_token_id: Either<u32, Vec<u32>>,
    #[serde(with = "either::serde_untagged")]
    eos_token_id: Either<u32, Vec<u32>>,
}

/// Python `str` of a non-integer float (`1e-06`, `1e+16`, `nan`); minijinja never uses
/// an exponent (`0.000001`, `10000000000000000.0`) and prints `NaN`. `None` for every
/// other value.
fn python_float_str(value: &Value) -> Option<String> {
    if !value.is_number() || value.is_integer() {
        return None;
    }
    f64::try_from(value.clone()).ok().map(python_float_repr)
}

/// Output formatter for `{{ x }}`. HF renders through Python, which prints a float
/// with `str`; see [`python_float_str`]. Other values are unchanged. `~` concatenation
/// is a VM op that can't be overridden, and filters that stringify values themselves
/// (`join`, `replace`, ...) use minijinja's spelling, so `'x' ~ 1e-6` and
/// `[1e-6] | join` still render `0.000001`.
pub fn python_formatter(out: &mut Output, state: &State, value: &Value) -> Result<(), Error> {
    match python_float_str(value) {
        Some(text) => out.write_str(&text).map_err(Error::from),
        None => escape_formatter(out, state, value),
    }
}

/// The `string` filter with Python `str` floats, matching [`python_formatter`];
/// other values go to the builtin.
pub fn python_string(state: &State, value: &Value) -> Result<Value, Error> {
    match python_float_str(value) {
        Some(text) => Ok(Value::from(text)),
        None => minijinja::filters::string(state, value),
    }
}

/// Mirrors HF transformers' `tojson` filter, not stock Jinja2's. Transformers
/// overrides Jinja's HTML-safe `tojson` with plain
/// `json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators,
/// sort_keys=sort_keys)`, `ensure_ascii` defaulting to `False`, in its chat-template
/// environment, and vLLM/SGLang render through that — so chat templates (and the
/// models trained on their output) expect Python separators and **no** HTML
/// escaping (`'`, `<`, `>`, `&` stay literal).
pub fn tojson(value: Value, kwargs: Kwargs) -> Result<Value, Error> {
    // Python takes a space count or the indent string itself.
    let indent = kwargs
        .get::<Option<Value>>("indent")?
        .map(|indent| match indent.as_str() {
            Some(text) => Ok(text.to_owned()),
            None => i64::try_from(indent)
                .map(|n| " ".repeat(usize::try_from(n).unwrap_or(0)))
                .map_err(|_| {
                    Error::new(
                        ErrorKind::InvalidOperation,
                        "tojson indent must be an int or a string",
                    )
                }),
        })
        .transpose()?;
    let separators = kwargs.get::<Option<Vec<String>>>("separators")?;
    // Python defaults to `(", ", ": ")`, or `(",", ": ")` when `indent` is set.
    let (item_sep, key_sep) = match separators.as_deref() {
        Some([item, key]) => (item.as_str(), key.as_str()),
        Some(_) => {
            return Err(Error::new(
                ErrorKind::InvalidOperation,
                "tojson separators must be an (item, key) pair",
            ));
        }
        None if indent.is_some() => (",", ": "),
        None => (", ", ": "),
    };
    let flag = |key| -> Result<bool, Error> {
        Ok(kwargs
            .get::<Option<Value>>(key)?
            .is_some_and(|v| v.is_true()))
    };
    let sort_keys = flag("sort_keys")?;
    let formatter =
        PyJsonFormatter::new(item_sep, key_sep, indent.as_deref(), flag("ensure_ascii")?);
    let mut buf = Vec::new();
    let mut serializer = serde_json::Serializer::with_formatter(&mut buf, formatter);
    let result = if sort_keys {
        serde_json::to_value(&value).and_then(|mut sorted| {
            sorted.sort_all_objects();
            sorted.serialize(&mut serializer)
        })
    } else {
        value.serialize(&mut serializer)
    };
    result.map_err(|err| {
        Error::new(ErrorKind::BadSerialization, "cannot serialize to JSON").with_source(err)
    })?;
    String::from_utf8(buf)
        .map_err(|err| {
            Error::new(ErrorKind::BadSerialization, "cannot serialize to JSON").with_source(err)
        })
        .map(Value::from_safe_string)
}

/// Parse a JSON string into a structured value.
///
/// HuggingFace/transformers chat-template environments expose this filter, and several
/// published templates depend on it — e.g. Step-3.7-Flash's `tool_use` block applies it to
/// a tool call's `arguments`, which arrive as a JSON *string*, to iterate the decoded
/// object. Without it minijinja aborts the render with `unknown filter: fromjson`, which
/// fails every multi-turn tool-call request.
///
/// Values that are not strings pass through untouched, so templates that apply the filter
/// defensively to an already-decoded value keep rendering.
pub fn fromjson(value: Value) -> Result<Value, Error> {
    let Some(text) = value.as_str() else {
        return Ok(value);
    };
    let parsed: serde_json::Value = serde_json::from_str(text).map_err(|err| {
        Error::new(ErrorKind::InvalidOperation, "cannot parse JSON").with_source(err)
    })?;
    Ok(Value::from_serialize(&parsed))
}

pub fn strftime_now(format_str: &str) -> Result<Value, Error> {
    let local: DateTime<Local> = Local::now();
    Ok(Value::from_safe_string(
        local.format(format_str).to_string(),
    ))
}

#[cfg(test)]
mod fromjson_tests {
    use super::*;

    #[test]
    fn parses_json_object_string() {
        let out = fromjson(Value::from(r#"{"location":"San Francisco","n":3}"#)).unwrap();
        assert_eq!(
            out.get_attr("location").unwrap().as_str(),
            Some("San Francisco")
        );
        assert_eq!(out.get_attr("n").unwrap().to_string(), "3");
    }

    #[test]
    fn parses_json_array_string() {
        let out = fromjson(Value::from(r#"[1,2,3]"#)).unwrap();
        assert_eq!(out.len(), Some(3));
    }

    #[test]
    fn passes_through_non_string() {
        // Already-decoded values must survive a defensive `| fromjson`.
        let already = Value::from_serialize(serde_json::json!({"a": 1}));
        let out = fromjson(already).unwrap();
        assert_eq!(out.get_attr("a").unwrap().to_string(), "1");
    }

    #[test]
    fn errors_on_malformed_json() {
        assert!(fromjson(Value::from("{not json")).is_err());
    }

    /// Regression: renders the shape of Step-3.7-Flash's `tool_use` block, where a tool
    /// call's `arguments` arrive as a JSON string. Before the filter existed this failed
    /// with `unknown filter: fromjson`, 500-ing every multi-turn tool-call request.
    #[test]
    fn renders_tool_use_block_with_json_string_arguments() {
        let mut env = minijinja::Environment::new();
        env.add_filter("fromjson", fromjson);
        env.add_template(
            "tool_use",
            "{% for tc in tool_calls %}{% set a = tc.function.arguments | fromjson %}\
CALL {{ tc.function.name }} loc={{ a.location }} unit={{ a.unit }}{% endfor %}",
        )
        .unwrap();
        let rendered = env
            .get_template("tool_use")
            .unwrap()
            .render(minijinja::context! { tool_calls => serde_json::json!([{
                "function": {
                    "name": "get_weather",
                    "arguments": "{\"location\":\"San Francisco\",\"unit\":\"F\"}"
                }
            }])})
            .expect("tool_use template must render once `fromjson` is registered");
        assert_eq!(rendered, "CALL get_weather loc=San Francisco unit=F");
    }
}

/// Expected strings are HF transformers' `tojson` output for the same input and
/// kwargs, i.e. Python `json.dumps(x, ensure_ascii=False, ...)`.
#[cfg(test)]
mod tojson_tests {
    use super::*;
    use serde_json::json;

    fn try_render(tmpl: &str, x: serde_json::Value) -> Result<String, Error> {
        let mut env = minijinja::Environment::new();
        env.add_filter("tojson", tojson);
        env.render_str(tmpl, minijinja::context! { x })
    }

    fn render(tmpl: &str, x: serde_json::Value) -> String {
        try_render(tmpl, x).unwrap()
    }

    fn shape() -> serde_json::Value {
        json!({"a": [1, 2], "b": {"c": "é"}, "d": [], "e": {}})
    }

    /// Kimi K2.x `tool_declare` renders `tools | tojson(separators=(',', ':'))`.
    #[test]
    fn honours_separators() {
        let tools = json!([{"type": "function", "function": {"name": "get_weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}]);
        assert_eq!(
            render("{{ x | tojson(separators=(',', ':')) }}", tools),
            r#"[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}}]"#
        );
    }

    #[test]
    fn honours_separators_with_indent() {
        assert_eq!(
            render(
                "{{ x | tojson(indent=2, separators=(', ', ':')) }}",
                shape()
            ),
            "{\n  \"a\":[\n    1, \n    2\n  ], \n  \"b\":{\n    \"c\":\"é\"\n  }, \n  \"d\":[], \n  \"e\":{}\n}"
        );
    }

    /// Python takes the indent string itself as well as a space count.
    #[test]
    fn indent_accepts_a_string() {
        assert_eq!(
            render("{{ x | tojson(indent='\t') }}", json!({"a": [1]})),
            "{\n\t\"a\": [\n\t\t1\n\t]\n}"
        );
    }

    /// Python's `json.dumps` raises on these too.
    #[test]
    fn rejects_malformed_kwargs() {
        for tmpl in [
            "{{ x | tojson(separators=(',',)) }}",
            "{{ x | tojson(indent=[2]) }}",
        ] {
            let err = try_render(tmpl, shape()).unwrap_err();
            assert_eq!(err.kind(), ErrorKind::InvalidOperation, "{tmpl}");
        }
    }

    #[test]
    fn sort_keys_sorts_nested_maps() {
        let x = json!({"b": 1, "a": {"d": [{"z": 1, "y": 2}], "c": 3}});
        assert_eq!(
            render("{{ x | tojson(sort_keys=true) }}", x),
            r#"{"a": {"c": 3, "d": [{"y": 2, "z": 1}]}, "b": 1}"#
        );
    }

    #[test]
    fn ensure_ascii_escapes_like_python() {
        let x = json!({"clé": "café 😀 \u{7f} \n"});
        assert_eq!(
            render("{{ x | tojson(ensure_ascii=true) }}", x),
            r#"{"cl\u00e9": "caf\u00e9 \ud83d\ude00 \u007f \n"}"#
        );
    }

    #[test]
    fn defaults_unchanged() {
        let flat = r#"{"a": [1, 2], "b": {"c": "é"}, "d": [], "e": {}}"#;
        assert_eq!(render("{{ x | tojson }}", shape()), flat);
        assert_eq!(
            render(
                "{{ x | tojson(separators=none, ensure_ascii=false) }}",
                shape()
            ),
            flat
        );
        assert_eq!(
            render("{{ x | tojson(indent=2) }}", shape()),
            "{\n  \"a\": [\n    1,\n    2\n  ],\n  \"b\": {\n    \"c\": \"é\"\n  },\n  \"d\": [],\n  \"e\": {}\n}"
        );
    }

    /// `sort_keys` re-serializes through `serde_json::Value`; floats keep Python spelling.
    #[test]
    fn sort_keys_keeps_python_floats() {
        assert_eq!(
            render(
                "{{ x | tojson(separators=(',', ':'), sort_keys=true) }}",
                json!({"n": 1e16, "lr": 0.000001})
            ),
            r#"{"lr":1e-06,"n":1e+16}"#
        );
    }
}
