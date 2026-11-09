// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Mistral v11+/v13/v15 reasoning and tool calls on the UnifiedParser.
//!
//! Output starts in visible content. `[THINK]` opens reasoning and `[/THINK]`
//! closes it; a duplicate `[THINK]` or a stray `[/THINK]` is stripped. A
//! `[TOOL_CALLS]` inside a thought ends the thought for good, because nothing
//! after a call can return to reasoning. See `tool_calling::mistral` for the
//! call grammar. Guided tool output is refused: the grammar has no call
//! terminator for the guided reader to strip native markup up to. A forced
//! `tool_choice`, `auto` with tools, and a JSON schema for the reply use
//! [`ForcedToolGrammar`] instead, which keeps the native format this parser
//! reads.

use std::borrow::Cow;
use std::fmt::Write as _;

use serde_json::Value;

use crate::tool_calling::mistral::{ARGS, TOOL_CALLS, mistral_scanner};
use crate::tool_calling::scan::ReasoningSpec;
use crate::tool_calling::traits::Tool;
use crate::unified::{GuidedRouted, ScannerUnified, UnifiedParser};

const REASONING_START: &str = "[THINK]";
const REASONING_END: &str = "[/THINK]";
const MODEL_SETTINGS: &str = "[MODEL_SETTINGS]";

const ANY_OBJECT: &str = r#"{"type":"object"}"#;
const CLOSED_EMPTY_OBJECT: &str =
    r#"{"type":"object","properties":{},"additionalProperties":false}"#;

/// Build the Mistral unified parser for one stream.
pub(crate) fn mistral_unified(_tools: &[Tool]) -> Box<dyn UnifiedParser> {
    Box::new(GuidedRouted::new(ScannerUnified::new(
        mistral_scanner().with_reasoning(ReasoningSpec {
            start: REASONING_START,
            end: REASONING_END,
            forced_start: false,
            preserve_special_tokens: true,
            ..Default::default()
        }),
    )))
}

/// mistral-common's tool-call grammar (`GrammarFactory.get_lark_from_jinja`),
/// in llguidance lark.
///
/// Upstream vLLM constrains v11+ tekken tokenizers with this grammar, not with a
/// JSON schema. Forced choices need it because the prompt does not carry
/// `tool_choice`, so the grammar is what forces the call; under `auto` it keeps
/// every call's arguments a JSON object. A JSON schema for the reply joins the
/// same grammar as one more alternative, so a backend sees lark for every
/// constrained request. The output keeps the `[TOOL_CALLS]name[ARGS]{json}`
/// format this parser reads. Three differences: `%json` schemas are compact, the
/// think template is chosen whenever `[THINK]`/`[/THINK]` exist, where
/// mistral-common also requires tekken v13+, and a thought may precede a JSON
/// reply whenever `[MODEL_SETTINGS]` exists, where mistral-common requires tekken
/// v15+. Tokenizers with think tokens are v13+, and those with
/// `[MODEL_SETTINGS]` are v15+, in practice.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ForcedToolGrammar {
    tool_calls: u32,
    args: u32,
    /// `[THINK]` and `[/THINK]`. With both, a thought may precede the calls.
    think: Option<(u32, u32)>,
    /// `[MODEL_SETTINGS]` exists. With think tokens, a thought may also precede
    /// a JSON reply.
    has_model_settings: bool,
}

/// The `tool_choice` modes [`ForcedToolGrammar::render`] renders. mistral-common's
/// `none` mode, whose body is bare content, is not one of them; with a JSON
/// schema it is [`ForcedToolGrammar::render_json`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CallMode<'a> {
    /// Calls are optional.
    Auto,
    Required,
    Named(&'a str),
}

impl ForcedToolGrammar {
    /// Resolve the special-token ids the grammar names with llguidance's
    /// `<[id]>` syntax. `lookup` must return an id only for a special token.
    /// `None` when `[TOOL_CALLS]` or `[ARGS]` has none.
    pub fn resolve(lookup: impl Fn(&str) -> Option<u32>) -> Option<Self> {
        Some(Self {
            tool_calls: lookup(TOOL_CALLS)?,
            args: lookup(ARGS)?,
            think: lookup(REASONING_START).zip(lookup(REASONING_END)),
            has_model_settings: lookup(MODEL_SETTINGS).is_some(),
        })
    }

    /// The grammar for one `tool_choice` mode. `parallel_tool_calls` allows one
    /// or more calls; otherwise one. A `json_schema` also allows a reply that is
    /// only JSON matching it, in place of the calls, as mistral-common unions it.
    pub fn render(
        &self,
        mode: CallMode<'_>,
        tools: &[Tool],
        parallel_tool_calls: bool,
        json_schema: Option<&Value>,
    ) -> String {
        let (named, body) = match mode {
            CallMode::Auto => (None, Body::OptionalCalls),
            CallMode::Required => (None, Body::Calls),
            CallMode::Named(name) => (Some(name), Body::Calls),
        };
        let fcall = self.fcall(named, tools, parallel_tool_calls);
        self.grammar(body, &fcall, json_schema)
    }

    /// A reply that is only JSON matching `json_schema`, after an optional
    /// thought (mistral-common's `get_lark_for_json_schema`).
    pub fn render_json(&self, json_schema: &Value) -> String {
        self.grammar(Body::Content, "", Some(json_schema))
    }

    /// mistral-common's base and think templates. The `Content` body is the
    /// JSON-only grammar, which keeps the unreachable `body` and `fcalls` rules
    /// mistral-common renders for it. An empty schema is falsy in mistral-common,
    /// so it adds no JSON section, and a JSON-only grammar is then bare content.
    fn grammar(&self, body: Body, fcall: &str, json_schema: Option<&Value>) -> String {
        let json_schema = json_schema.filter(|schema| !is_empty_object(schema));
        let json_only = matches!(body, Body::Content);
        let mut grammar = String::new();
        match self.think {
            Some((begin, end)) => {
                match json_schema {
                    None => grammar.push_str("start: body\n"),
                    Some(schema) => {
                        let start = if json_only {
                            "json_grammar"
                        } else {
                            "body | json_grammar"
                        };
                        let think = if self.has_model_settings {
                            "think? "
                        } else {
                            ""
                        };
                        let _ = writeln!(
                            grammar,
                            "start: {start}\njson_grammar: {think}SAFE_WS? %json {schema}"
                        );
                    }
                }
                let body = match body {
                    Body::OptionalCalls => "think? (content | fcalls)",
                    Body::Calls => "think? fcalls",
                    Body::Content => "think? content",
                };
                let _ = write!(
                    grammar,
                    "body: {body}\nfcalls: content? {fcall}\n\n\
                     content: (/(.|\\n)+/)+\n\nthink: <[{begin}]> content <[{end}]>\n\n\
                     SAFE_WS: /[ \\t\\r\\n]{{1,8}}/"
                );
            }
            None => {
                match json_schema {
                    None => grammar.push_str("start: body\n"),
                    Some(schema) => {
                        let body = if json_only { "" } else { "body | " };
                        let _ = writeln!(grammar, "start: {body}SAFE_WS? %json {schema}");
                    }
                }
                let body = match body {
                    Body::OptionalCalls => "content | (content? fcalls)",
                    Body::Calls => "content? fcalls",
                    Body::Content => "content",
                };
                let _ = write!(
                    grammar,
                    "body: {body}\nfcalls: {fcall}\n\n\
                     content: (/(.|\\n)+/)+\n\nSAFE_WS: /[ \\t\\r\\n]{{1,8}}/"
                );
            }
        }
        grammar
    }

    /// mistral-common's `_convert_tool_calls`. Unless some tool is `strict`, any
    /// object is accepted, and an unnamed choice accepts any name.
    fn fcall(&self, named: Option<&str>, tools: &[Tool], parallel: bool) -> String {
        let entries: Vec<(String, Cow<'static, str>)> =
            if tools.iter().any(|tool| tool.strict == Some(true)) {
                tools
                    .iter()
                    .filter(|tool| named.is_none_or(|name| tool.name == name))
                    .map(|tool| (quoted(&tool.name), arguments_schema(tool)))
                    .collect()
            } else {
                let name = named.map_or_else(|| "/.+/".to_string(), quoted);
                vec![(name, Cow::Borrowed(ANY_OBJECT))]
            };
        let mut calls = entries
            .into_iter()
            .map(|(name, schema)| {
                format!(
                    "<[{}]> SAFE_WS? {name} <[{}]> SAFE_WS? %json {schema} SAFE_WS?",
                    self.tool_calls, self.args
                )
            })
            .collect::<Vec<_>>();
        let call = if calls.len() == 1 {
            calls.swap_remove(0)
        } else {
            calls
                .iter()
                .map(|call| format!("({call})"))
                .collect::<Vec<_>>()
                .join(" | ")
        };
        if parallel { format!("({call})+") } else { call }
    }
}

/// mistral-common's `mode` for the `body` rule.
#[derive(Clone, Copy)]
enum Body {
    /// `auto`: content, or calls after optional content.
    OptionalCalls,
    /// `required`, or a named tool.
    Calls,
    /// `none`: content only.
    Content,
}

fn is_empty_object(value: &Value) -> bool {
    value.as_object().is_some_and(|map| map.is_empty())
}

fn quoted(name: &str) -> String {
    Value::String(name.to_owned()).to_string()
}

/// A strict tool's declared parameters, or mistral-common's closed empty object
/// when it declares none. Other tools accept any object.
fn arguments_schema(tool: &Tool) -> Cow<'static, str> {
    match &tool.parameters {
        _ if tool.strict != Some(true) => Cow::Borrowed(ANY_OBJECT),
        schema @ Value::Object(map) if !map.is_empty() => Cow::Owned(schema.to_string()),
        _ => Cow::Borrowed(CLOSED_EMPTY_OBJECT),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::unified::{
        UnifiedEvent, UnifiedParserEvent, UnifiedParserExt, UnifiedParserInit,
        UnifiedParserStartingState, UnifiedToolOutputMode, assemble,
    };

    fn reasoning(text: &str) -> UnifiedEvent {
        UnifiedEvent::Reasoning { text: text.into() }
    }

    fn text(text: &str) -> UnifiedEvent {
        UnifiedEvent::Text { text: text.into() }
    }

    fn call(name: &str, arguments: serde_json::Value) -> UnifiedEvent {
        UnifiedEvent::ToolCall {
            name: name.into(),
            arguments,
        }
    }

    /// Raw events for `input` pushed as two chunks split at byte `at`.
    fn events_split_at(input: &str, at: usize) -> Vec<UnifiedParserEvent> {
        let mut parser = mistral_unified(&[]);
        let mut events = parser.push(&input[..at]).expect("push prefix");
        events.extend(parser.push(&input[at..]).expect("push suffix"));
        events.extend(parser.finish().expect("finish").events);
        events
    }

    /// The same assembled events at every two-chunk split point, and with one
    /// character per chunk.
    fn assert_every_split(input: &str, want: &[UnifiedEvent]) {
        for at in (0..=input.len()).filter(|&at| input.is_char_boundary(at)) {
            assert_eq!(
                assemble(&events_split_at(input, at)),
                want,
                "split at byte {at} of {input:?}"
            );
        }
        let mut parser = mistral_unified(&[]);
        let mut events = Vec::new();
        for ch in input.chars() {
            events.extend(parser.push(ch.encode_utf8(&mut [0; 4])).expect("push char"));
        }
        events.extend(parser.finish().expect("finish").events);
        assert_eq!(assemble(&events), want, "one char per chunk of {input:?}");
    }

    fn weather(city: &str) -> UnifiedEvent {
        call("get_weather", serde_json::json!({"city": city}))
    }

    #[test]
    fn reasoning_then_call() {
        assert_every_split(
            r#"[THINK]Need the weather.[/THINK][TOOL_CALLS]get_weather[ARGS]{"city": "Paris"}"#,
            &[reasoning("Need the weather."), weather("Paris")],
        );
    }

    /// Parallel calls repeat `[TOOL_CALLS]` with no end marker.
    #[test]
    fn content_then_parallel_calls() {
        assert_every_split(
            r#"Let me check.[TOOL_CALLS]get_weather[ARGS]{"city": "Paris"}[TOOL_CALLS]get_weather[ARGS]{"city": "Lyon"}"#,
            &[text("Let me check."), weather("Paris"), weather("Lyon")],
        );
    }

    /// `[TOOL_CALLS]` ends an open thought for good: nothing after a call can
    /// return to reasoning, so the rest of the thought is the call's leftover.
    #[test]
    fn call_inside_reasoning_ends_the_thought() {
        assert_every_split(
            r#"[THINK]need a tool[TOOL_CALLS]get_weather[ARGS]{"city": "Paris"}"#,
            &[reasoning("need a tool"), weather("Paris")],
        );
        assert_every_split(
            r#"[THINK]need a tool [TOOL_CALLS]get_weather[ARGS]{"city": "Paris"} more thought[/THINK]answer"#,
            &[reasoning("need a tool"), weather("Paris")],
        );
    }

    #[test]
    fn plain_answer_stays_content() {
        assert_every_split(
            "The answer is 42, no tools needed.",
            &[text("The answer is 42, no tools needed.")],
        );
    }

    /// Text after the JSON value is dropped (vllm-project/vllm#48975).
    #[test]
    fn text_after_the_arguments_is_dropped() {
        assert_every_split(
            r#"[TOOL_CALLS]get_weather[ARGS]{"city": "SF"}Estimating the weather..."#,
            &[weather("SF")],
        );
        assert_every_split(
            r#"[TOOL_CALLS]get_weather{"city": "SF"}Estimating {roughly}..."#,
            &[weather("SF")],
        );
    }

    /// Leftover text between calls, braces included, cannot start a call.
    #[test]
    fn text_between_calls_is_dropped() {
        assert_every_split(
            r#"[TOOL_CALLS]f[ARGS]{"x": 1} then {braces} [ARGS][TOOL_CALLS]g[ARGS]{"y": 2}"#,
            &[
                call("f", serde_json::json!({"x": 1})),
                call("g", serde_json::json!({"y": 2})),
            ],
        );
    }

    /// The calls of upstream vLLM's v11+ `test_extract_tool_calls` table.
    #[test]
    fn upstream_v11_table() {
        let add = || call("add_this_and_that", serde_json::json!({"a": 3.5, "b": 4}));
        let pair = || {
            [
                call("add", serde_json::json!({"a": 3.5, "b": 4})),
                call("multiply", serde_json::json!({"a": 3, "b": 6})),
            ]
        };
        let sf = || {
            call(
                "get_current_weather",
                serde_json::json!({"city": "San Francisco", "state": "CA", "unit": "celsius"}),
            )
        };
        let bash = || {
            call(
                "bash",
                serde_json::json!({"command": "print(\"hello world!\")\nre.compile(r'{}')"}),
            )
        };
        let cases: [(&str, Vec<UnifiedEvent>); 10] = [
            (
                r#"[TOOL_CALLS]add_this_and_that{"a": 3.5, "b": 4}"#,
                vec![add()],
            ),
            (
                r#"[TOOL_CALLS]get_current_weather{"city": "San Francisco", "state": "CA", "unit": "celsius"}"#,
                vec![sf()],
            ),
            (
                r#"[TOOL_CALLS]add{"a": 3.5, "b": 4}[TOOL_CALLS]multiply{"a": 3, "b": 6}"#,
                pair().into(),
            ),
            (
                r#"[TOOL_CALLS]add_this_and_that[ARGS]{"a": 3.5, "b": 4}"#,
                vec![add()],
            ),
            (
                r#"[TOOL_CALLS]get_current_weather[ARGS]{"city": "San Francisco", "state": "CA", "unit": "celsius"}"#,
                vec![sf()],
            ),
            (
                r#"[TOOL_CALLS]add[ARGS]{"a": 3.5, "b": 4}[TOOL_CALLS]multiply[ARGS]{"a": 3, "b": 6}"#,
                pair().into(),
            ),
            (
                r#"hi[TOOL_CALLS]add[ARGS]{"a": 1, "b": 2}"#,
                vec![text("hi"), call("add", serde_json::json!({"a": 1, "b": 2}))],
            ),
            (
                r#"hi{hi[TOOL_CALLS]bash{"command": "print(\"hello world!\")\nre.compile(r'{}')"}"#,
                vec![text("hi{hi"), bash()],
            ),
            (
                r#"[TOOL_CALLS]get_current_weather{"city": "San Francisco"}Estimating the weather..."#,
                vec![call(
                    "get_current_weather",
                    serde_json::json!({"city": "San Francisco"}),
                )],
            ),
            (
                r#"[TOOL_CALLS]get_current_weather[ARGS]{"city": "San Francisco"}Estimating the weather..."#,
                vec![call(
                    "get_current_weather",
                    serde_json::json!({"city": "San Francisco"}),
                )],
            ),
        ];
        for (input, want) in cases {
            assert_every_split(input, &want);
        }
    }

    /// A name with no arguments is a call with `{}`, as upstream vLLM emits it.
    #[test]
    fn missing_body_is_an_empty_object() {
        for input in [
            "[TOOL_CALLS]toolname_no_args",
            "[TOOL_CALLS]toolname_no_args[ARGS]",
            "[TOOL_CALLS]toolname_no_args[ARGS]  ",
        ] {
            assert_every_split(input, &[call("toolname_no_args", serde_json::json!({}))]);
        }
        assert_every_split(
            r#"[TOOL_CALLS]first[TOOL_CALLS]second[ARGS]{"x": 1}"#,
            &[
                call("first", serde_json::json!({})),
                call("second", serde_json::json!({"x": 1})),
            ],
        );
    }

    /// A repeated `[ARGS]`, or a `[CALL_ID]` after `[ARGS]`, does not open a
    /// JSON array; the body after it is the arguments.
    #[test]
    fn stray_separator_after_args() {
        assert_every_split(
            r#"[TOOL_CALLS]f[ARGS][ARGS]{"x": 1}[TOOL_CALLS]g[ARGS][CALL_ID]a1b2c3d4e{"y": 2}"#,
            &[
                call("f", serde_json::json!({"x": 1})),
                call("g", serde_json::json!({"y": 2})),
            ],
        );
    }

    /// Arguments keep the first JSON value even when it is not an object, and
    /// drop the text after it, as upstream vLLM's `_extract_args_json` does.
    #[test]
    fn non_object_arguments_keep_the_first_json_value() {
        assert_every_split(
            r#"[TOOL_CALLS]a[ARGS][1, 2][TOOL_CALLS]b[ARGS]42[TOOL_CALLS]c[ARGS]"done" trailing words"#,
            &[
                call("a", serde_json::json!([1, 2])),
                call("b", serde_json::json!(42)),
                call("c", serde_json::json!("done")),
            ],
        );
    }

    /// A name before the marker is content, and the call keeps an empty name
    /// so the malformed generation surfaces instead of disappearing.
    #[test]
    fn name_before_the_marker_is_content() {
        assert_every_split(
            r#"bash[TOOL_CALLS]{"command": "ls"}"#,
            &[text("bash"), call("", serde_json::json!({"command": "ls"}))],
        );
        assert_every_split(
            r#"bash[TOOL_CALLS]{"command": "ls"}[TOOL_CALLS]submit[ARGS]{"answer": {"nested": 1}}"#,
            &[
                text("bash"),
                call("", serde_json::json!({"command": "ls"})),
                call("submit", serde_json::json!({"answer": {"nested": 1}})),
            ],
        );
    }

    /// An unterminated JSON body at EOF is an incomplete call and is dropped
    /// (policy P2). Upstream vLLM instead emits the raw prefix as arguments.
    #[test]
    fn truncated_body_is_dropped() {
        assert_every_split(
            r#"hi{hi[TOOL_CALLS]bash{"command": "print(\"hello world!\")\nre.compile(r'{}')"#,
            &[text("hi{hi")],
        );
        assert_every_split(
            r#"[THINK]ok[/THINK][TOOL_CALLS]get_weather[ARGS]{"city": "Par"#,
            &[reasoning("ok")],
        );
        assert_every_split("bash[TOOL_CALLS]{not valid json at all", &[text("bash")]);
    }

    /// A malformed body ends at the next call; its raw bytes do not parse, so
    /// assembly gives `{}` (policy P3), and the later call survives.
    #[test]
    fn malformed_body_ends_at_the_next_call() {
        assert_every_split(
            r#"[TOOL_CALLS]bad[ARGS]{"x": [TOOL_CALLS]ok[ARGS]{"y": 2}"#,
            &[
                call("bad", serde_json::json!({})),
                call("ok", serde_json::json!({"y": 2})),
            ],
        );
        assert_every_split(
            r#"[TOOL_CALLS]bad[ARGS]{"x": 1]}[TOOL_CALLS]ok[ARGS]{}"#,
            &[
                call("bad", serde_json::json!({})),
                call("ok", serde_json::json!({})),
            ],
        );
    }

    /// A string that never closes swallows the later calls while streaming.
    /// At EOF the body ends at the first marker it swallowed, so they survive.
    #[test]
    fn unterminated_string_releases_later_calls_at_eof() {
        let echo = |value: &str| call("echo", serde_json::json!({"value": value}));
        assert_every_split(
            r#"[TOOL_CALLS]bad[ARGS]{"value": {"x":"unfinished[TOOL_CALLS]echo[ARGS]{"value": "é"}[TOOL_CALLS]echo[ARGS]{"value": "Café"}"#,
            &[call("bad", serde_json::json!({})), echo("é"), echo("Café")],
        );
    }

    /// A truncated body whose marker sits in a closed string keeps it as data:
    /// the quote parity is even, so no quote is missing and the call is dropped.
    #[test]
    fn truncated_body_keeps_a_quoted_marker_as_data() {
        assert_every_split(r#"[TOOL_CALLS]f[ARGS]{"x":"[TOOL_CALLS]g[ARGS]{}""#, &[]);
        assert_every_split(
            r#"ok[TOOL_CALLS]f[ARGS]{"x":"[TOOL_CALLS]g[ARGS]{}", "y": 1"#,
            &[text("ok")],
        );
    }

    /// Markers inside a JSON string are argument data (`I7`).
    #[test]
    fn markers_inside_a_string_argument_are_data() {
        let cmd = "git log [TOOL_CALLS] [ARGS] [/THINK] { --oneline";
        let input = format!("[TOOL_CALLS]run[ARGS]{}", serde_json::json!({"cmd": cmd}));
        assert_every_split(&input, &[call("run", serde_json::json!({"cmd": cmd}))]);
    }

    /// v11 may write `[CALL_ID]ID` between the name and the arguments.
    #[test]
    fn call_id_is_stripped() {
        assert_every_split(
            r#"[TOOL_CALLS]get_weather[CALL_ID]a1b2c3d4e[ARGS]{"city": "Paris"}"#,
            &[weather("Paris")],
        );
    }

    /// A name that is not a tool identifier is not a call: the unsupported
    /// pre-v11 array and prose after a bare opener are dropped, while an empty
    /// name still surfaces the malformed call.
    #[test]
    fn non_identifier_names_are_not_calls() {
        assert_every_split(
            r#"[TOOL_CALLS] [{"name": "add", "arguments": {"a": 1}}, {"name": "mul", "arguments": {"b": 2}}]"#,
            &[],
        );
        assert_every_split(
            "Sure.[TOOL_CALLS]Sure, here is the answer.",
            &[text("Sure.")],
        );
        assert_every_split(
            r#"[TOOL_CALLS]not a name[ARGS]{"x": 1}[TOOL_CALLS]get_weather[ARGS]{"city": "Paris"}"#,
            &[weather("Paris")],
        );
    }

    /// A valid v11 `[CALL_ID]` is the call's id; anything else is replaced.
    #[test]
    fn valid_call_id_is_kept() {
        let mut parser = mistral_unified(&[]);
        parser
            .parse_complete(
                r#"[TOOL_CALLS]f[CALL_ID]a1b2c3d4e[ARGS]{}[TOOL_CALLS]g[CALL_ID]bad id[ARGS]{}"#,
            )
            .expect("parse");
        assert_eq!(parser.tool_call_id(0), Some("a1b2c3d4e"));
        let minted = parser.tool_call_id(1).expect("minted id");
        assert_eq!(minted.len(), 9);
        assert_ne!(minted, "bad id");
    }

    #[test]
    fn repeated_opener_is_one_call() {
        assert_every_split(
            r#"[TOOL_CALLS][TOOL_CALLS]get_weather[ARGS]{"city": "Paris"}"#,
            &[weather("Paris")],
        );
    }

    #[test]
    fn opener_alone_emits_nothing() {
        assert_every_split("Sure.[TOOL_CALLS]", &[text("Sure.")]);
        assert_every_split("Sure.[TOOL_CALLS]   ", &[text("Sure.")]);
    }

    /// The reasoning cases of upstream vLLM's `test_mistral_reasoning_v13`.
    #[test]
    fn upstream_v13_reasoning_table() {
        let cases: [(&str, Vec<UnifiedEvent>); 8] = [
            ("[THINK]r[/THINK]c", vec![reasoning("r"), text("c")]),
            ("[THINK]r[/THINK]", vec![reasoning("r")]),
            (
                "[THINK]a\nb[/THINK]c\nd",
                vec![reasoning("a\nb"), text("c\nd")],
            ),
            ("hello world", vec![text("hello world")]),
            ("[THINK]r", vec![reasoning("r")]),
            ("[/THINK]This is the rest", vec![text("This is the rest")]),
            (
                "Before\n[THINK]r[/THINK]\nAfter",
                vec![text("Before\n"), reasoning("r"), text("\nAfter")],
            ),
            ("", vec![]),
        ];
        for (input, want) in cases {
            assert_every_split(input, &want);
        }
    }

    /// A duplicate `[THINK]` inside a thought and a stray `[/THINK]` outside one
    /// are both stripped.
    #[test]
    fn duplicate_and_stray_reasoning_markers_are_stripped() {
        assert_every_split(
            "[THINK]a[THINK]b[/THINK]c[/THINK]d",
            &[reasoning("ab"), text("cd")],
        );
    }

    /// Reasoning loses its trailing whitespace once the thought ends, and keeps
    /// it when the thought continues in an adjacent span.
    #[test]
    fn reasoning_is_right_stripped() {
        assert_every_split("[THINK]a \n[/THINK] b", &[reasoning("a"), text(" b")]);
        assert_every_split("[THINK]a  ", &[reasoning("a")]);
        assert_every_split("[THINK] \n[/THINK]b", &[text("b")]);
        assert_every_split(
            "[THINK]a [/THINK][THINK]b [/THINK]c",
            &[reasoning("a b"), text("c")],
        );
    }

    #[test]
    fn stray_argument_markers_outside_a_call_are_stripped() {
        assert_every_split("a[ARGS]b[CALL_ID]c", &[text("abc")]);
    }

    #[test]
    fn prefilled_reasoning_continues_the_thought() {
        let mut parser = mistral_unified(&[]);
        parser
            .initialize_request(UnifiedParserInit {
                starting_state: UnifiedParserStartingState::Reasoning,
                ..UnifiedParserInit::native(&[])
            })
            .expect("prefilled reasoning");
        let mut events = parser.push("still thinking [/THINK]done").expect("push");
        events.extend(parser.finish().expect("finish").events);
        assert_eq!(
            assemble(&events),
            [reasoning("still thinking"), text("done")]
        );
    }

    /// Guided output is refused before any state changes.
    #[test]
    fn guided_tool_output_is_refused() {
        let mut parser = mistral_unified(&[]);
        let guided = UnifiedParserInit {
            tool_output_mode: UnifiedToolOutputMode::GuidedJson { named_tool: None },
            ..UnifiedParserInit::native(&[])
        };
        assert!(parser.initialize_request(guided).is_err());
        let events = parser
            .parse_complete("[THINK]r[/THINK]c")
            .expect("native parse");
        assert_eq!(events, [reasoning("r"), text("c")]);
    }

    #[test]
    fn markers_require_special_tokens() {
        assert!(mistral_unified(&[]).preserve_special_tokens());
    }

    /// mistral-common accepts only 9-character alphanumeric call ids.
    #[test]
    fn tool_call_ids_are_nine_alphanumerics_per_call() {
        let mut parser = mistral_unified(&[]);
        parser
            .parse_complete(r#"[TOOL_CALLS]f[ARGS]{}[TOOL_CALLS]g[ARGS]{}"#)
            .expect("parse");
        let first = parser.tool_call_id(0).expect("first id").to_string();
        let second = parser.tool_call_id(1).expect("second id").to_string();
        for id in [&first, &second] {
            assert_eq!(id.len(), 9, "{id}");
            assert!(id.chars().all(|ch| ch.is_ascii_alphanumeric()), "{id}");
        }
        assert_ne!(first, second);
        assert_eq!(parser.tool_call_id(2), None);

        parser.reset();
        assert_eq!(parser.tool_call_id(0), None);
    }

    /// Each call streams as one complete update as soon as its JSON closes.
    #[test]
    fn a_call_streams_when_its_json_closes() {
        let mut parser = mistral_unified(&[]);
        assert_eq!(
            parser
                .push(r#"[TOOL_CALLS]get_weather[ARGS]{"city": "Par"#)
                .expect("push"),
            []
        );
        let events = parser.push(r#"is"}"#).expect("push");
        assert_eq!(assemble(&events), [weather("Paris")]);
    }

    /// Reasoning markers are special tokens, so prose quotation cannot make
    /// `[/THINK]` literal. An odd quote in a thought must not hold the closer
    /// until a later quote turns the answer into reasoning.
    #[test]
    fn odd_quote_in_reasoning_does_not_hold_the_closer() {
        assert_every_split(
            r#"[THINK]He said "hi[/THINK]The answer is "42"."#,
            &[reasoning(r#"He said "hi"#), text(r#"The answer is "42"."#)],
        );
    }

    /// The closer must reach the consumer as soon as it streams, not at EOF.
    #[test]
    fn odd_quote_in_reasoning_streams_the_answer_before_eof() {
        let mut parser = mistral_unified(&[]);
        let mut events = parser
            .push(r#"[THINK]He said "hi[/THINK]The answer"#)
            .expect("push");
        events.extend(parser.push(" is 42.").expect("push"));
        assert_eq!(
            assemble(&events),
            [reasoning(r#"He said "hi"#), text("The answer is 42.")]
        );
    }

    /// An odd quote in visible content must not hide the call that follows.
    #[test]
    fn odd_quote_in_content_does_not_hide_the_call() {
        assert_every_split(
            r#"He said "maybe[TOOL_CALLS]get_weather[ARGS]{"city": "Paris"}"#,
            &[
                text(r#"He said "maybe"#),
                call("get_weather", serde_json::json!({"city": "Paris"})),
            ],
        );
    }

    #[test]
    fn odd_quote_in_content_streams_the_call_before_eof() {
        let mut parser = mistral_unified(&[]);
        let events = parser
            .push(r#"He said "maybe[TOOL_CALLS]get_weather[ARGS]{"city": "Paris"}"#)
            .expect("push");
        assert_eq!(
            assemble(&events),
            [
                text(r#"He said "maybe"#),
                call("get_weather", serde_json::json!({"city": "Paris"})),
            ]
        );
    }
}

#[cfg(test)]
mod forced_tool_grammar_tests {
    use super::*;
    use serde_json::json;

    const THINK: ForcedToolGrammar = ForcedToolGrammar {
        tool_calls: 9,
        args: 32,
        think: Some((34, 35)),
        has_model_settings: true,
    };

    fn tool(name: &str, strict: Option<bool>, parameters: Value) -> Tool {
        Tool {
            name: name.to_string(),
            description: None,
            parameters,
            strict,
        }
    }

    fn weather(strict: Option<bool>) -> Tool {
        let schema = json!({"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]});
        tool("get_weather", strict, schema)
    }

    fn time(strict: Option<bool>) -> Tool {
        let schema = json!({"type": "object", "properties": {"zone": {"type": "string"}}});
        tool("get_time", strict, schema)
    }

    fn fcalls(grammar: &str) -> &str {
        grammar.lines().nth(2).expect("fcalls rule")
    }

    // The goldens below are mistral-common 1.13 output for the mock tekken v15
    // tokenizer, with compact `%json` schemas.

    #[test]
    fn think_template_matches_mistral_common() {
        let tools = [weather(None), time(None)];
        assert_eq!(
            THINK.render(CallMode::Required, &tools, true, None),
            r#"start: body
body: think? fcalls
fcalls: content? (<[9]> SAFE_WS? /.+/ <[32]> SAFE_WS? %json {"type":"object"} SAFE_WS?)+

content: (/(.|\n)+/)+

think: <[34]> content <[35]>

SAFE_WS: /[ \t\r\n]{1,8}/"#
        );
        assert_eq!(
            fcalls(&THINK.render(CallMode::Named("get_time"), &tools, true, None)),
            r#"fcalls: content? (<[9]> SAFE_WS? "get_time" <[32]> SAFE_WS? %json {"type":"object"} SAFE_WS?)+"#
        );
        // `auto` makes the calls optional and changes nothing else.
        assert_eq!(
            THINK.render(CallMode::Auto, &tools, true, None),
            r#"start: body
body: think? (content | fcalls)
fcalls: content? (<[9]> SAFE_WS? /.+/ <[32]> SAFE_WS? %json {"type":"object"} SAFE_WS?)+

content: (/(.|\n)+/)+

think: <[34]> content <[35]>

SAFE_WS: /[ \t\r\n]{1,8}/"#
        );
    }

    #[test]
    fn base_template_matches_mistral_common() {
        let base = ForcedToolGrammar {
            think: None,
            ..THINK
        };
        assert_eq!(
            base.render(CallMode::Required, &[weather(None)], false, None),
            r#"start: body
body: content? fcalls
fcalls: <[9]> SAFE_WS? /.+/ <[32]> SAFE_WS? %json {"type":"object"} SAFE_WS?

content: (/(.|\n)+/)+

SAFE_WS: /[ \t\r\n]{1,8}/"#
        );
        assert_eq!(
            base.render(CallMode::Auto, &[weather(None)], false, None),
            r#"start: body
body: content | (content? fcalls)
fcalls: <[9]> SAFE_WS? /.+/ <[32]> SAFE_WS? %json {"type":"object"} SAFE_WS?

content: (/(.|\n)+/)+

SAFE_WS: /[ \t\r\n]{1,8}/"#
        );
    }

    #[test]
    fn strict_tools_constrain_names_and_arguments() {
        let tools = [weather(Some(true)), time(Some(true))];
        assert_eq!(
            fcalls(&THINK.render(CallMode::Required, &tools, true, None)),
            concat!(
                r#"fcalls: content? ((<[9]> SAFE_WS? "get_weather" <[32]> SAFE_WS? %json {"type":"object","properties":{"city":{"type":"string"}},"required":["city"]} SAFE_WS?)"#,
                r#" | (<[9]> SAFE_WS? "get_time" <[32]> SAFE_WS? %json {"type":"object","properties":{"zone":{"type":"string"}}} SAFE_WS?))+"#,
            )
        );
        assert_eq!(
            fcalls(&THINK.render(CallMode::Auto, &tools, true, None)),
            fcalls(&THINK.render(CallMode::Required, &tools, true, None)),
            "auto constrains each call exactly as required does"
        );
        assert_eq!(
            fcalls(&THINK.render(CallMode::Named("get_weather"), &tools, false, None)),
            r#"fcalls: content? <[9]> SAFE_WS? "get_weather" <[32]> SAFE_WS? %json {"type":"object","properties":{"city":{"type":"string"}},"required":["city"]} SAFE_WS?"#
        );
    }

    /// One strict tool lists every tool by name; the others still accept any
    /// object. Without parallel calls mistral-common leaves the alternation
    /// unparenthesized, so `content?` binds to the first call only.
    #[test]
    fn mixed_strictness_lists_every_tool() {
        assert_eq!(
            fcalls(&THINK.render(
                CallMode::Required,
                &[weather(Some(true)), time(None)],
                false,
                None
            )),
            concat!(
                r#"fcalls: content? (<[9]> SAFE_WS? "get_weather" <[32]> SAFE_WS? %json {"type":"object","properties":{"city":{"type":"string"}},"required":["city"]} SAFE_WS?)"#,
                r#" | (<[9]> SAFE_WS? "get_time" <[32]> SAFE_WS? %json {"type":"object"} SAFE_WS?)"#,
            )
        );
        assert_eq!(
            fcalls(&THINK.render(
                CallMode::Required,
                &[tool("ping", Some(true), Value::Null)],
                false,
                None
            )),
            r#"fcalls: content? <[9]> SAFE_WS? "ping" <[32]> SAFE_WS? %json {"type":"object","properties":{},"additionalProperties":false} SAFE_WS?"#
        );
    }

    fn answer() -> Value {
        json!({"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]})
    }

    // The JSON goldens are mistral-common `c668fc2` (version string 1.13.0)
    // output for the mock tekken tokenizer as v15, as v13 (no model settings)
    // and as v11 (base template), with compact `%json` schemas.

    #[test]
    fn json_only_matches_mistral_common() {
        // mistral-common renders the unreachable `fcalls` rule with an empty
        // call list, so its line ends in a space.
        assert_eq!(
            THINK.render_json(&answer()),
            concat!(
                "start: json_grammar\n",
                r#"json_grammar: think? SAFE_WS? %json {"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"]}"#,
                "\nbody: think? content\nfcalls: content? \n\n",
                "content: (/(.|\\n)+/)+\n\nthink: <[34]> content <[35]>\n\n",
                r"SAFE_WS: /[ \t\r\n]{1,8}/",
            )
        );
        let v13 = ForcedToolGrammar {
            has_model_settings: false,
            ..THINK
        };
        assert_eq!(
            v13.render_json(&json!({"anyOf": [{"type": "object"}, {"type": "array"}]}))
                .lines()
                .nth(1),
            Some(r#"json_grammar: SAFE_WS? %json {"anyOf":[{"type":"object"},{"type":"array"}]}"#)
        );
        let base = ForcedToolGrammar {
            think: None,
            ..THINK
        };
        assert_eq!(
            base.render_json(&answer()),
            concat!(
                r#"start: SAFE_WS? %json {"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"]}"#,
                "\nbody: content\nfcalls: \n\n",
                "content: (/(.|\\n)+/)+\n\n",
                r"SAFE_WS: /[ \t\r\n]{1,8}/",
            )
        );
        // An empty schema adds no JSON section, leaving bare content.
        assert_eq!(
            THINK.render_json(&json!({})),
            concat!(
                "start: body\nbody: think? content\nfcalls: content? \n\n",
                "content: (/(.|\\n)+/)+\n\nthink: <[34]> content <[35]>\n\n",
                r"SAFE_WS: /[ \t\r\n]{1,8}/",
            )
        );
    }

    /// A reply schema becomes one more `start` alternative; the tool rules are
    /// unchanged, so a forced choice may answer with JSON instead of a call.
    #[test]
    fn json_union_matches_mistral_common() {
        let tools = [weather(None)];
        assert_eq!(
            THINK.render(CallMode::Auto, &tools, true, Some(&answer())),
            r#"start: body | json_grammar
json_grammar: think? SAFE_WS? %json {"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"]}
body: think? (content | fcalls)
fcalls: content? (<[9]> SAFE_WS? /.+/ <[32]> SAFE_WS? %json {"type":"object"} SAFE_WS?)+

content: (/(.|\n)+/)+

think: <[34]> content <[35]>

SAFE_WS: /[ \t\r\n]{1,8}/"#
        );
        let required = THINK.render(
            CallMode::Named("get_weather"),
            &tools,
            true,
            Some(&answer()),
        );
        assert_eq!(
            required.lines().take(4).collect::<Vec<_>>(),
            [
                "start: body | json_grammar",
                r#"json_grammar: think? SAFE_WS? %json {"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"]}"#,
                "body: think? fcalls",
                r#"fcalls: content? (<[9]> SAFE_WS? "get_weather" <[32]> SAFE_WS? %json {"type":"object"} SAFE_WS?)+"#,
            ]
        );
        let v13 = ForcedToolGrammar {
            has_model_settings: false,
            ..THINK
        };
        assert_eq!(
            v13.render(CallMode::Auto, &tools, true, Some(&answer()))
                .lines()
                .nth(1),
            Some(
                r#"json_grammar: SAFE_WS? %json {"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"]}"#
            )
        );
        let base = ForcedToolGrammar {
            think: None,
            ..THINK
        };
        assert_eq!(
            base.render(CallMode::Required, &tools, true, Some(&answer())),
            r#"start: body | SAFE_WS? %json {"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"]}
body: content? fcalls
fcalls: (<[9]> SAFE_WS? /.+/ <[32]> SAFE_WS? %json {"type":"object"} SAFE_WS?)+

content: (/(.|\n)+/)+

SAFE_WS: /[ \t\r\n]{1,8}/"#
        );
        // An empty schema adds no JSON alternative.
        assert_eq!(
            THINK.render(CallMode::Auto, &tools, true, Some(&json!({}))),
            THINK.render(CallMode::Auto, &tools, true, None)
        );
    }

    #[test]
    fn resolve_needs_the_call_markers_and_both_think_markers() {
        fn ids<'a>(pairs: &'a [(&'a str, u32)]) -> impl Fn(&str) -> Option<u32> + 'a {
            move |token| {
                pairs
                    .iter()
                    .find_map(|(name, id)| (*name == token).then_some(*id))
            }
        }
        assert_eq!(
            ForcedToolGrammar::resolve(ids(&[
                ("[TOOL_CALLS]", 9),
                ("[ARGS]", 32),
                ("[THINK]", 34),
                ("[/THINK]", 35),
                ("[MODEL_SETTINGS]", 36)
            ])),
            Some(THINK)
        );
        assert_eq!(
            ForcedToolGrammar::resolve(ids(&[
                ("[TOOL_CALLS]", 9),
                ("[ARGS]", 32),
                ("[THINK]", 34),
                ("[/THINK]", 35)
            ])),
            Some(ForcedToolGrammar {
                has_model_settings: false,
                ..THINK
            })
        );
        assert_eq!(
            ForcedToolGrammar::resolve(ids(&[
                ("[TOOL_CALLS]", 9),
                ("[ARGS]", 32),
                ("[THINK]", 34)
            ])),
            Some(ForcedToolGrammar {
                think: None,
                has_model_settings: false,
                ..THINK
            })
        );
        assert_eq!(
            ForcedToolGrammar::resolve(ids(&[
                ("[TOOL_CALLS]", 9),
                ("[THINK]", 34),
                ("[/THINK]", 35)
            ])),
            None
        );
    }
}
