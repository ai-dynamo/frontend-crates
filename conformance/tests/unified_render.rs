// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Compute the Dynamo v2 column LIVE, render the unified conformance tab, and
//! write the capture the packaged shard is built from.
//!
//! Columns: GOLDEN (authored oracle) | vLLM (captured) | Dynamo v2.
//!
//! The Dynamo column is a PER-FAMILY MIXTURE, the same shape as vLLM Rust's: the
//! native UnifiedParser where one exists, and the v1-reasoning + v2-tool split
//! everywhere else. The split is the "before" state — it parses ALL reasoning
//! first, so reasoning interleaved with tool calls loses its position. Every
//! current corpus family has a native UnifiedParser; a new split-only family
//! reintroduces the fallback.
//!
//! Output: `conformance/unified/unified_results.yaml`. The exploder and packager turn that
//! feed into the committed `dynamo_v2-<ver>` shard, which is what the
//! CONFORMANCE_v2.html tab actually reads — the tab never runs these parsers. The
//! `committed_dynamo_capture_matches_the_live_parsers` test below fails if that
//! shard drifts from the parsers.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use dynamo_parsers::{ReasoningParser, ReasoningParserType};
use dynamo_parsers_v2::{
    assemble, create_tool_parser_for_family, create_unified_parser_for_family,
};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::{Value, json};

mod common;
use common::unified_capture::{self as capture, chunk_input};

use common::{
    Init,
    known_unified_divergences::{self as divergences, Check, Expected},
    unified_tools as tools,
};

#[derive(Deserialize)]
struct GoldenFile {
    family: String,
    cases: BTreeMap<String, GoldenCase>,
}

#[derive(Deserialize)]
struct GoldenCase {
    description: String,
    #[serde(default)]
    policy: Vec<String>,
    #[serde(default)]
    init: Init,
    #[serde(default)]
    finish_reason: Option<String>,
    input: String,
    #[serde(default)]
    input_chunks: Option<Vec<String>>,
    #[serde(default)]
    tools: Option<serde_json::Value>,
    golden: Vec<Ev>,
    expect: BTreeMap<String, Expect>,
}

#[derive(Deserialize, Clone)]
struct Expect {
    verdict: String,
    #[serde(default)]
    class: Option<String>,
    #[serde(default)]
    note: Option<String>,
}

/// A unified event. PartialEq drives the golden comparison.
#[derive(Deserialize, serde::Serialize, Clone, Debug)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum Ev {
    MessageStart {
        phase: String,
    },
    MessageEnd,
    Reasoning {
        text: String,
    },
    Text {
        text: String,
    },
    ToolCall {
        name: String,
        #[serde(default = "default_raw_arguments")]
        arguments: ArgumentJson,
    },
}

impl PartialEq for Ev {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::MessageStart { phase: left }, Self::MessageStart { phase: right }) => {
                left == right
            }
            (Self::MessageEnd, Self::MessageEnd) => true,
            (Self::Reasoning { text: left }, Self::Reasoning { text: right })
            | (Self::Text { text: left }, Self::Text { text: right }) => left == right,
            (
                Self::ToolCall {
                    name: left_name,
                    arguments: left_arguments,
                },
                Self::ToolCall {
                    name: right_name,
                    arguments: right_arguments,
                },
            ) => left_name == right_name && left_arguments == right_arguments,
            _ => false,
        }
    }
}

#[derive(Clone, Debug)]
struct ArgumentJson(String);

impl PartialEq for ArgumentJson {
    fn eq(&self, other: &Self) -> bool {
        common::canonical_argument_json(&self.0) == common::canonical_argument_json(&other.0)
    }
}

impl Serialize for ArgumentJson {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        // YAML numbers pass through binary64 in the Python capture pipeline.
        // Keep the complete JSON argument text so nested decimals stay exact.
        serializer.serialize_str(&self.0)
    }
}

impl<'de> Deserialize<'de> for ArgumentJson {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = serde_json::Value::deserialize(deserializer)?;
        let raw = match value {
            serde_json::Value::String(encoded) => {
                if serde_json::value::RawValue::from_string(encoded.clone()).is_ok() {
                    encoded
                } else {
                    serde_json::to_string(&encoded).map_err(serde::de::Error::custom)?
                }
            }
            value => serde_json::to_string(&value).map_err(serde::de::Error::custom)?,
        };
        serde_json::value::RawValue::from_string(raw.clone()).map_err(serde::de::Error::custom)?;
        Ok(Self(raw))
    }
}

fn default_raw_arguments() -> ArgumentJson {
    ArgumentJson("{}".to_string())
}

fn raw_arguments(raw: &str) -> ArgumentJson {
    serde_json::value::RawValue::from_string(raw.to_string())
        .unwrap_or_else(|error| panic!("tool arguments must be valid JSON ({raw:?}): {error}"));
    ArgumentJson(raw.to_string())
}

impl Ev {
    fn render(&self) -> String {
        match self {
            Ev::MessageStart { phase } => format!("message_start({phase})"),
            Ev::MessageEnd => "message_end".into(),
            Ev::Reasoning { text } => format!("reasoning({text:?})"),
            Ev::Text { text } => format!("text({text:?})"),
            Ev::ToolCall { name, arguments } => {
                format!("tool_call({name}, {})", arguments.0)
            }
        }
    }
}

/// Map a corpus family to (v1 reasoning parser, v2 tool parser) for the SPLIT path.
/// Declared once in `parser_families.yaml` under `unified:`; see `common::unified_family`.
fn parsers_for(family: &str) -> (String, String) {
    let f = common::unified_family(family);
    (f.reasoning_parser, f.tool_parser)
}

/// Fold one tool-parser result into the event list, preserving text/call order
/// and coalescing per-`tool_index` deltas into one call.
fn feed(
    res: dynamo_parsers_v2::ToolParseResult,
    out: &mut Vec<Ev>,
    slots: &mut BTreeMap<usize, usize>,
    raw_args: &mut BTreeMap<usize, String>,
) {
    if !res.normal_text.is_empty() {
        if let Some(Ev::Text { text }) = out.last_mut() {
            text.push_str(&res.normal_text);
        } else {
            out.push(Ev::Text {
                text: res.normal_text,
            });
        }
    }
    for d in res.calls {
        let pos = *slots.entry(d.tool_index).or_insert_with(|| {
            out.push(Ev::ToolCall {
                name: d.name.clone().unwrap_or_default(),
                arguments: raw_arguments("null"),
            });
            out.len() - 1
        });
        if let Some(n) = &d.name
            && let Ev::ToolCall { name, .. } = &mut out[pos]
            && name.is_empty()
        {
            *name = n.clone();
        }
        raw_args
            .entry(d.tool_index)
            .or_default()
            .push_str(&d.arguments);
    }
}

impl From<dynamo_parsers_v2::UnifiedEvent> for Ev {
    fn from(e: dynamo_parsers_v2::UnifiedEvent) -> Self {
        match e {
            dynamo_parsers_v2::UnifiedEvent::MessageStart { phase } => Ev::MessageStart {
                phase: serde_json::to_value(phase)
                    .unwrap()
                    .as_str()
                    .unwrap()
                    .to_owned(),
            },
            dynamo_parsers_v2::UnifiedEvent::MessageEnd => Ev::MessageEnd,
            dynamo_parsers_v2::UnifiedEvent::Reasoning { text } => Ev::Reasoning { text },
            dynamo_parsers_v2::UnifiedEvent::Text { text } => Ev::Text { text },
            dynamo_parsers_v2::UnifiedEvent::ToolCall { name, arguments } => Ev::ToolCall {
                name,
                arguments: raw_arguments(
                    &serde_json::to_string(&arguments).expect("serialize projected arguments"),
                ),
            },
        }
    }
}

fn unified_delta_json(d: &dynamo_parsers_v2::UnifiedParserEvent) -> Value {
    match d {
        dynamo_parsers_v2::UnifiedParserEvent::MessageStart(phase) => {
            json!({"kind":"message_start","phase":phase})
        }
        dynamo_parsers_v2::UnifiedParserEvent::MessageEnd => json!({"kind":"message_end"}),
        dynamo_parsers_v2::UnifiedParserEvent::Reasoning(text) => {
            json!({"kind": "reasoning", "text": text})
        }
        dynamo_parsers_v2::UnifiedParserEvent::Text(text) => json!({"kind": "text", "text": text}),
        dynamo_parsers_v2::UnifiedParserEvent::ToolCall(c) => {
            json!({"kind": "tool_call", "name": c.name, "arguments": c.arguments, "complete": c.complete})
        }
    }
}

/// Compute the Dynamo v2 unified event list for one input.
///
/// A family with a native unified parser is parsed by one state machine per
/// stream owning reasoning + content + tool calls. Every current corpus family
/// takes this path; the split fallback remains for other inputs.
///
/// Both paths are driven from the SAME chunking as `dynamo_chunks`, so the
/// assembled row and the per-chunk rows in the popup describe one run.
fn dynamo_events_with_tools(
    family: &str,
    input: &str,
    init: &Init,
    tool_schemas: &[dynamo_parsers_v2::Tool],
) -> Vec<Ev> {
    if create_unified_parser_for_family(family, tool_schemas).is_ok() {
        return capture_native_case(family, &chunk_input(input), init, tool_schemas)
            .expect("native capture")
            .0;
    }

    let (reasoning_name, tool_family) = parsers_for(family);

    let mut rp = ReasoningParserType::get_reasoning_parser_from_name(&reasoning_name);
    let split = rp.detect_and_parse_reasoning(input, &[]);

    let mut out = Vec::new();
    if !split.reasoning_text.is_empty() {
        out.push(Ev::Reasoning {
            text: split.reasoning_text.clone(),
        });
    }

    let chunks: Vec<String> = split.normal_text.chars().map(|ch| ch.to_string()).collect();
    out.extend(tool_events_with_chunks(
        &tool_family,
        tool_schemas,
        chunks
            .iter()
            .map(|text| dynamo_parsers_v2::ToolParserInput::Text(text)),
    ));
    out
}

fn tool_events_with_chunks<'a>(
    tool_family: &str,
    tool_schemas: &[dynamo_parsers_v2::Tool],
    chunks: impl IntoIterator<Item = dynamo_parsers_v2::ToolParserInput<'a>>,
) -> Vec<Ev> {
    let mut tp = create_tool_parser_for_family(tool_family, tool_schemas)
        .unwrap_or_else(|e| panic!("create tool parser for `{tool_family}`: {e}"));
    let mut out = Vec::new();
    let mut slots: BTreeMap<usize, usize> = BTreeMap::new();
    let mut raw_args: BTreeMap<usize, String> = BTreeMap::new();
    for chunk in chunks {
        let r = tp
            .push_input(chunk)
            .unwrap_or_else(|e| panic!("push `{tool_family}`: {e}"));
        feed(r, &mut out, &mut slots, &mut raw_args);
    }
    let r = tp
        .finish()
        .unwrap_or_else(|e| panic!("finish `{tool_family}`: {e}"));
    feed(r, &mut out, &mut slots, &mut raw_args);

    for (ti, pos) in &slots {
        let raw = raw_args.get(ti).map(String::as_str).unwrap_or("");
        if let Ev::ToolCall { arguments, .. } = &mut out[*pos] {
            *arguments = raw_arguments(if raw.trim().is_empty() { "{}" } else { raw });
        }
    }
    out
}

#[test]
fn dotted_harmony_split_matches_golden_at_every_boundary() {
    use dynamo_parsers_v2::ToolParserInput::{Text, Tokens};
    let path = common::ensure_unified_golden().join(common::unified_family("gpt_oss").golden_spec);
    let file: GoldenFile = serde_yaml::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let mut checked = 0;
    for (id, case) in file.cases {
        if !id.starts_with("UNIFIED.dotted_name_") {
            continue;
        }
        let tools = common::unified_tools_for_schemas(case.tools.as_ref());
        let mut reasoning = ReasoningParserType::get_reasoning_parser_from_name("gpt_oss");
        let split = reasoning.detect_and_parse_reasoning(&case.input, &[]);
        assert!(split.reasoning_text.is_empty(), "{id}");
        assert_eq!(
            dynamo_events_with_tools("gpt_oss", &case.input, &case.init, &tools),
            case.golden,
            "{id}: native character input"
        );
        assert_eq!(
            tool_events_with_chunks("harmony", &tools, [Text(&split.normal_text)]),
            case.golden,
            "{id}: whole"
        );
        for offset in split
            .normal_text
            .char_indices()
            .map(|(offset, _)| offset)
            .chain(std::iter::once(split.normal_text.len()))
        {
            let (left, right) = split.normal_text.split_at(offset);
            assert_eq!(
                tool_events_with_chunks("harmony", &tools, [Text(left), Text(right)]),
                case.golden,
                "{id}: text split {offset}"
            );
        }
        let tokens = dynamo_parsers_v2::encode_harmony(&split.normal_text).unwrap();
        for offset in 0..=tokens.len() {
            let (left, right) = tokens.split_at(offset);
            assert_eq!(
                tool_events_with_chunks("harmony", &tools, [Tokens(left), Tokens(right)]),
                case.golden,
                "{id}: token split {offset}"
            );
        }
        checked += 1;
    }
    assert_eq!(checked, 4);
}

type NativeCapture = (Vec<Ev>, Vec<ChunkRow>, BTreeMap<usize, String>);

fn capture_native_case(
    family: &str,
    chunks: &[String],
    init: &Init,
    tool_schemas: &[dynamo_parsers_v2::Tool],
) -> Result<NativeCapture, capture::CaptureFailure> {
    let mut parser = create_unified_parser_for_family(family, tool_schemas)
        .unwrap_or_else(|e| panic!("create native parser {family}: {e}"));
    init.apply(&mut parser, family);
    capture::capture_step(
        init.validate_capture(family, tool_schemas, chunks),
        "structural tag",
    )?;
    let rows = capture::native_capture(&mut parser, chunks)?;
    let deltas: Vec<_> = rows.iter().flatten().cloned().collect();
    let raw_args = dynamo_parsers_v2::tool_arguments_raw(&deltas);
    let assembled: Vec<Ev> = assemble(&deltas).into_iter().map(Ev::from).collect();
    let rows = chunks
        .iter()
        .cloned()
        .chain(std::iter::once("‹finish›".into()))
        .zip(rows)
        .map(|(delta_text, deltas)| ChunkRow {
            delta_text,
            deltas: deltas.iter().map(unified_delta_json).collect(),
        })
        .collect();
    Ok((assembled, rows, raw_args))
}

#[test]
fn numeric_raw_arguments_match_authored_tokens_for_every_family() {
    let dir = common::ensure_unified_golden();
    let mut checked = 0usize;
    for entry in std::fs::read_dir(&dir).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().and_then(|extension| extension.to_str()) != Some("yaml") {
            continue;
        }
        let file: GoldenFile = serde_yaml::from_str(&std::fs::read_to_string(&path).unwrap())
            .unwrap_or_else(|error| panic!("{}: {error}", path.display()));
        for (id, case) in file.cases {
            if !id
                .strip_prefix("UNIFIED.")
                .is_some_and(|scenario| scenario.starts_with("arg_numeric_"))
            {
                continue;
            }
            let schemas = common::unified_tools_for_schemas(case.tools.as_ref());
            let chunks = capture::input_chunks(&case.input, case.input_chunks.as_deref());
            let (_assembled, _rows, raw) =
                capture_native_case(&file.family, &chunks, &case.init, &schemas)
                    .unwrap_or_else(|error| panic!("{id}: {error:?}"));
            let expected = if file.family == "beam"
                && case.tools.as_ref().is_some_and(|tools| {
                    tools
                        .as_array()
                        .and_then(|tools| tools.first())
                        .and_then(|tool| tool.pointer("/parameters/properties/value/type"))
                        .and_then(Value::as_array)
                        .is_some_and(|types| {
                            types.iter().any(|kind| kind.as_str() == Some("string"))
                        })
                }) {
                // Beam chooses the schema-permitted string branch for this union; compare its
                // exact spelling instead of the historical v1 numeric projection.
                let raw = case
                    .input
                    .split_once(r#"<rfl:parameter name="value">"#)
                    .and_then(|(_, value)| value.split_once("</rfl:parameter>"))
                    .map(|(value, _)| value.trim())
                    .unwrap_or_else(|| panic!("{id}: cannot recover authored numeric token"));
                format!(r#"{{"value":{}}}"#, serde_json::to_string(raw).unwrap())
            } else {
                case.golden
                    .iter()
                    .find_map(|event| match event {
                        Ev::ToolCall { arguments, .. } => Some(arguments.0.clone()),
                        _ => None,
                    })
                    .unwrap_or_else(|| panic!("{id}: numeric case has no golden tool call"))
            };
            let actual = raw
                .get(&0)
                .unwrap_or_else(|| panic!("{id}: no raw tool arguments"));
            assert_eq!(
                common::canonical_argument_json(actual),
                common::canonical_argument_json(&expected),
                "{id}"
            );
            checked += 1;
        }
    }
    assert!(checked > 0, "no authored numeric cases were checked");
}

/// One streaming chunk: the delta text fed, and the RAW per-chunk deltas Dynamo
/// emitted (reasoning/text/tool_call fragments, not coalesced) as JSON.
struct ChunkRow {
    delta_text: String,
    deltas: Vec<Value>,
}

fn tool_deltas(res: &dynamo_parsers_v2::ToolParseResult, out: &mut Vec<Value>) {
    if !res.normal_text.is_empty() {
        out.push(json!({"kind": "text", "text": res.normal_text}));
    }
    for c in &res.calls {
        out.push(json!({"kind": "tool_call", "name": c.name, "arguments": c.arguments}));
    }
}

/// Stream `input` through Dynamo's split pipeline CHUNK BY CHUNK, recording the
/// real per-chunk emitted deltas (v1 reasoning streaming incremental -> v2 tool
/// streaming push on the leftover content).
fn dynamo_chunks(family: &str, input: &str, init: &Init) -> Vec<ChunkRow> {
    dynamo_chunks_with_tools(family, input, init, &tools())
}

fn dynamo_chunks_with_tools(
    family: &str,
    input: &str,
    init: &Init,
    tool_schemas: &[dynamo_parsers_v2::Tool],
) -> Vec<ChunkRow> {
    if create_unified_parser_for_family(family, tool_schemas).is_ok() {
        return capture_native_case(family, &chunk_input(input), init, tool_schemas)
            .expect("native capture")
            .1;
    }

    let (reasoning_name, tool_family) = parsers_for(family);
    let mut rp = ReasoningParserType::get_reasoning_parser_from_name(&reasoning_name);
    let mut tp = create_tool_parser_for_family(&tool_family, tool_schemas)
        .unwrap_or_else(|e| panic!("create tool parser for `{tool_family}`: {e}"));

    let mut rows = Vec::new();
    for chunk in chunk_input(input) {
        let mut deltas: Vec<Value> = Vec::new();
        let rr = rp.parse_reasoning_streaming_incremental(&chunk, &[]);
        if !rr.reasoning_text.is_empty() {
            deltas.push(json!({"kind": "reasoning", "text": rr.reasoning_text}));
        }
        if !rr.normal_text.is_empty() {
            let tr = tp.push(&rr.normal_text).expect("split capture push failed");
            tool_deltas(&tr, &mut deltas);
        }
        rows.push(ChunkRow {
            delta_text: chunk,
            deltas,
        });
    }
    // Flush: reasoning tail -> tool -> finish.
    let mut tail: Vec<Value> = Vec::new();
    let rf = rp.finish_reasoning_stream();
    if !rf.reasoning_text.is_empty() {
        tail.push(json!({"kind": "reasoning", "text": rf.reasoning_text}));
    }
    if !rf.normal_text.is_empty() {
        let tr = tp
            .push(&rf.normal_text)
            .expect("split capture tail push failed");
        tool_deltas(&tr, &mut tail);
    }
    tool_deltas(
        &tp.finish().expect("split capture finish failed"),
        &mut tail,
    );
    rows.push(ChunkRow {
        delta_text: "‹finish›".to_string(),
        deltas: tail,
    });
    rows
}

#[test]
fn finish_is_part_of_the_stream_schedule_even_when_it_emits_nothing() {
    let rows = dynamo_chunks("qwen3", "plain response", &Init::default());
    let finish = rows.last().expect("finish row");
    assert_eq!(finish.delta_text, "‹finish›");
    assert!(
        finish.deltas.is_empty(),
        "fixture must exercise an empty finish"
    );
}

#[test]
fn nested_numeric_arguments_are_compared_without_binary64_rounding() {
    let raw = r#"{"nested":{"fraction":9007199254740992.5,"tiny":1e-400},"values":[9007199254740993,-9007199254740993]}"#;
    let expected = vec![Ev::ToolCall {
        name: "inspect".into(),
        arguments: raw_arguments(raw),
    }];
    let yaml = format!("kind: tool_call\nname: inspect\narguments: '{raw}'\n");
    let decoded: Ev = serde_yaml::from_str(&yaml).expect("decode raw numeric arguments");
    let exact = vec![decoded];
    let rounded = vec![Ev::ToolCall {
        name: "inspect".into(),
        arguments: raw_arguments(
            r#"{"nested":{"fraction":9007199254740992.0,"tiny":0.0},"values":[9007199254740992,-9007199254740992]}"#,
        ),
    }];

    assert_eq!(classify("qwen3", &expected, &exact), "MATCH");
    assert_eq!(classify("qwen3", &expected, &rounded), "ARG_MISMATCH");
    let serialized = serde_json::to_string(&exact).expect("serialize exact argument token");
    assert!(serialized.contains("9007199254740992.5"));
    assert!(serialized.contains("1e-400"));
    assert!(serialized.contains("9007199254740993"));
    let decoded_json: Vec<Ev> =
        serde_json::from_str(&serialized).expect("decode captured arguments");
    assert_eq!(decoded_json, exact);
    let yaml = common::json_to_yaml_value(&exact);
    let decoded_yaml: Vec<Ev> = serde_yaml::from_value(yaml).expect("decode packaged arguments");
    assert_eq!(decoded_yaml, exact);
    assert_eq!(
        raw_arguments(r#"{"number":4.2e1,"text":"42"}"#),
        raw_arguments(r#"{ "text":"42", "number":42.0 }"#)
    );
    assert_ne!(
        raw_arguments(r#"{"number":42}"#),
        raw_arguments(r#"{"number":"42"}"#)
    );
}

#[test]
fn yaml_numeric_arguments_keep_their_json_number_lexemes() {
    let yaml = "kind: tool_call\nname: inspect\narguments:\n  fraction: 3.5\n  text: \"3.5\"\n";
    let decoded: Ev = serde_yaml::from_str(yaml).expect("decode YAML numeric arguments");
    assert_eq!(
        decoded,
        Ev::ToolCall {
            name: "inspect".into(),
            arguments: raw_arguments(r#"{"fraction":3.5,"text":"3.5"}"#),
        }
    );
}

/// Classify a Dynamo divergence from the golden.
fn classify(family: &str, golden: &[Ev], got: &[Ev]) -> &'static str {
    if golden == got {
        return "MATCH";
    }
    // Control markup that leaked into a visible payload.
    const MARKERS: &[&str] = &[
        "<|",
        "|>",
        "<think>",
        "</think>",
        "◁",
        "<channel",
        "channel|>",
    ];
    // Per-family markup that leaks invisibly to MARKERS above. gemma4's channel opener
    // leaves `thought\n`. qwen3's tool envelope has NO `<|...|>` sentinels, so a
    // `<tool_call>...</tool_call>` leaking into reasoning_content is invisible to the
    // shared list — enumerate it (kimi's tool/section markers already contain `<|`/`|>`).
    // Declared per family in `parser_families.yaml` (`unified:` -> `leak_markers`),
    // because this markup is invisible to the shared MARKERS list above.
    let family_leak: Vec<String> = common::unified_family(family).leak_markers;
    let leaks = got.iter().any(|e| match e {
        Ev::Text { text } | Ev::Reasoning { text } => MARKERS
            .iter()
            .copied()
            .chain(family_leak.iter().map(String::as_str))
            .any(|m| text.contains(m)),
        Ev::ToolCall { .. } | Ev::MessageStart { .. } | Ev::MessageEnd => false,
    });
    if leaks {
        return "LEAK";
    }
    let reasoning = |evs: &[Ev]| {
        evs.iter()
            .filter(|e| matches!(e, Ev::Reasoning { .. }))
            .count()
    };
    if reasoning(got) < reasoning(golden) {
        return "MERGE";
    }
    // Tool calls line up by name but an argument value differs (e.g. a string arg
    // truncated at a marker-looking substring) -> ARG_MISMATCH.
    let calls = |evs: &[Ev]| -> Vec<(String, String)> {
        evs.iter()
            .filter_map(|e| match e {
                Ev::ToolCall { name, arguments } => Some((name.clone(), arguments.0.clone())),
                _ => None,
            })
            .collect()
    };
    let (gc, tc) = (calls(golden), calls(got));
    if gc.len() == tc.len()
        && gc.iter().zip(&tc).all(|(a, b)| a.0 == b.0)
        && gc.iter().zip(&tc).any(|(a, b)| a.1 != b.1)
    {
        return "ARG_MISMATCH";
    }
    // Same content (concatenated per kind), different order/boundaries -> ORDER;
    // content actually missing -> LOSS.
    let cat = |evs: &[Ev], want_reasoning: bool| -> String {
        evs.iter()
            .filter_map(|e| match e {
                Ev::Reasoning { text } if want_reasoning => Some(text.as_str()),
                Ev::Text { text } if !want_reasoning => Some(text.as_str()),
                _ => None,
            })
            .collect()
    };
    if gc == tc && cat(golden, true) == cat(got, true) && cat(golden, false) == cat(got, false) {
        return "ORDER";
    }
    "LOSS"
}

fn todo_for(class: &str) -> &'static str {
    match class {
        "MERGE" | "ORDER" => {
            "TODO: adopt the UnifiedParser. The split parses ALL reasoning first, so reasoning that occurs between or after tool calls is merged up front and loses its position. One state machine per stream (owning reasoning+content+tools) fixes this by construction."
        }
        "LOSS" => {
            "TODO: content/reasoning dropped by the split. UnifiedParser must preserve every segment in order."
        }
        _ => "TODO: unify reasoning + tool parsing into one ordered event stream.",
    }
}

fn esc(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

fn events_html(evs: &[Ev]) -> String {
    if evs.is_empty() {
        return "<i>(no events)</i>".to_string();
    }
    evs.iter()
        .map(|e| format!("<div>{}</div>", esc(&e.render())))
        .collect()
}

struct Cell {
    verdict: String, // "MATCH" or a divergence class
    tip: String,
}

fn cell(
    engine: &str,
    input: &str,
    golden: &[Ev],
    got_html: &str,
    verdict: &str,
    class: &str,
    extra: &str,
) -> Cell {
    let v = if verdict == "match" || class == "MATCH" {
        "MATCH"
    } else {
        class
    };
    let tip = format!(
        "<b>{engine}</b> — <b>{v}</b><hr><b>input</b><pre>{}</pre><b>golden</b>{}<b>{engine}</b>{}{}",
        esc(input),
        events_html(golden),
        got_html,
        extra,
    );
    Cell {
        verdict: v.to_string(),
        tip,
    }
}

#[test]
fn render_unified_conformance_html() {
    let capture_provenance = common::dynamo_capture_provenance(None);
    let known = divergences::load();
    let mut observed_dynamo = std::collections::BTreeSet::new();
    let mut observed_projection = std::collections::BTreeSet::new();
    let mut parity_failures = Vec::new();
    // The vLLM column is LIVE, not an expectation. `capture_vllm_rust_unified.py`
    // records the `vllm-parser` crate against this same corpus; reading it here is
    // what makes the column evidence instead of a claim.
    let vllm_live: BTreeMap<(String, String), Vec<Ev>> = {
        let froot = common::ensure_unified_fixtures().join("unified");
        let mut m = BTreeMap::new();
        // Shards key by TAXONOMY id (`UNIFIED.30-guided_json_named_tool`); the golden keys by SCENARIO
        // (`UNIFIED.guided_json_named_tool.qwen3`). The inputs shard carries both, so
        // it is the bridge — without it every cell reads NO-DATA while the capture
        // sits right there, which is how this first went wrong.
        let mut by_tax: BTreeMap<(String, String), String> = BTreeMap::new();
        for entry in glob_yaml(&froot.join("inputs")) {
            if let Ok(doc) = serde_yaml::from_str::<InputDoc>(
                &std::fs::read_to_string(&entry).unwrap_or_default(),
            ) {
                for (cid, c) in doc.cases {
                    by_tax.insert((doc.family.clone(), cid), c.scenario);
                }
            }
        }
        if let Some(d) = common::version_dirs_ascending(&froot, "vllm_rust-").pop() {
            for entry in glob_yaml(&d) {
                if let Ok(doc) = serde_yaml::from_str::<CaptureDoc>(
                    &std::fs::read_to_string(&entry).unwrap_or_default(),
                ) {
                    for (cid, c) in doc.cases {
                        if let Some(sc) = by_tax.get(&(doc.family.clone(), cid)) {
                            m.insert(
                                (doc.family.clone(), format!("UNIFIED.{sc}.{}", doc.family)),
                                c.assembled,
                            );
                        }
                    }
                }
            }
        }
        m
    };
    let dir = common::ensure_unified_golden();
    let mut files: Vec<GoldenFile> = Vec::new();
    for entry in std::fs::read_dir(&dir).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().and_then(|e| e.to_str()) != Some("yaml") {
            continue;
        }
        let text = std::fs::read_to_string(&path).unwrap();
        files.push(
            serde_yaml::from_str(&text).unwrap_or_else(|e| panic!("{}: {e}", path.display())),
        );
    }
    files.sort_by(|a, b| a.family.cmp(&b.family));
    let selected_families = common::capture_families();
    files.retain(|file| {
        selected_families
            .as_ref()
            .is_none_or(|selected| selected.contains(&file.family))
    });

    let mut rows = String::new();
    let mut dynamo_red = 0usize;
    let mut vllm_red = 0usize;
    let mut total = 0usize;
    // Capture tooling consumes this feed; the viewer prefers resolved JSON even in a checkout.
    let mut json_cases: Vec<Value> = Vec::new();

    for file in &files {
        // Split-only families remain in the Unified corpus for their legacy batch
        // histories, but they have no current native UnifiedParser capture to render.
        if !common::unified_family(&file.family).native {
            continue;
        }
        rows.push_str(&format!(
            "<tr class=fam><td colspan=4>{} &nbsp; <span class=sub>reasoning=`{}` · tool=`{}`</span></td></tr>",
            esc(&file.family),
            parsers_for(&file.family).0,
            parsers_for(&file.family).1,
        ));
        for (id, case) in &file.cases {
            total += 1;

            // Dynamo: live.
            let case_tools = common::unified_tools_for_schemas(case.tools.as_ref());
            let schedule = capture::input_chunks(&case.input, case.input_chunks.as_deref());
            let (got, captured_rows, error) =
                match capture_native_case(&file.family, &schedule, &case.init, &case_tools) {
                    Ok((got, rows, _raw)) => (got, rows, None),
                    Err(capture::CaptureFailure::Error(error)) => {
                        (Vec::new(), Vec::new(), Some(error))
                    }
                    Err(failure) => panic!("native capture unavailable: {failure:?}"),
                };
            let dclass = if error.is_some() {
                "ERROR"
            } else {
                classify(&file.family, &case.golden, &got)
            };
            let actual = error
                .as_ref()
                .map(|e| format!("ERROR: {e}"))
                .unwrap_or_else(|| got.iter().map(Ev::render).collect::<Vec<_>>().join("  |  "));
            // These entries pin the public Value projection separately from raw
            // argument bytes, which are checked by the numeric parser tests.
            let projection_matches =
                match divergences::expected(&known, &file.family, id, Check::Projection) {
                    Some(Expected::Projection(expected)) if actual == expected.actual => {
                        observed_projection.insert((file.family.clone(), id.clone()));
                        true
                    }
                    Some(Expected::Projection(expected)) => {
                        parity_failures.push(format!(
                    "{id}: known projection divergence changed\n expected: {}\n      got: {}",
                    expected.actual, actual
                    ));
                        false
                    }
                    _ => false,
                };
            if dclass != "MATCH" {
                match divergences::expected(&known, &file.family, id, Check::Golden) {
                    Some(Expected::Golden(expected)) if actual == expected.actual => {
                        observed_dynamo.insert((file.family.clone(), id.clone()));
                    }
                    Some(Expected::Golden(expected)) => parity_failures.push(format!(
                        "{id}: known golden divergence changed\n expected: {}\n      got: {}",
                        expected.actual, actual
                    )),
                    _ if projection_matches => {}
                    _ => parity_failures.push(format!(
                        "{id}: unexpected golden divergence\n   golden: {}\n   actual: {}",
                        case.golden
                            .iter()
                            .map(Ev::render)
                            .collect::<Vec<_>>()
                            .join("  |  "),
                        actual
                    )),
                }
            }
            eprintln!(
                "{id:44} dynamo={dclass:6} :: {}",
                got.iter().map(Ev::render).collect::<Vec<_>>().join("  |  ")
            );
            let chunk_feed: Vec<Value> = if error.is_some() {
                schedule
                    .iter()
                    .cloned()
                    .chain(std::iter::once("‹finish›".into()))
                    .map(|text| json!({"delta_text": text}))
                    .collect()
            } else {
                captured_rows
                    .into_iter()
                    .map(|r| json!({"delta_text":r.delta_text,"dynamo":r.deltas}))
                    .collect()
            };

            let scenario = id
                .strip_prefix("UNIFIED.")
                .and_then(|s| s.strip_suffix(&format!(".{}", file.family)))
                .unwrap_or(id.as_str());
            let vx = case.expect.get("vllm");
            json_cases.push(serde_json::json!({
                "id": id,
                "family": file.family,
                "scenario": scenario,
                "description": case.description,
                "policy": case.policy,
                "init": case.init.applied(),
                "finish_reason": case.finish_reason.clone().unwrap_or_else(|| "stop".to_string()),
                "input": case.input,
                "tools": common::unified_tool_schemas_for_case(case.tools.as_ref()),
                "golden": case.golden,
                "dynamo": got,
                "dynamo_error": error,
                "dynamo_verdict": dclass,
                "vllm_verdict": vx.map(|e| if e.verdict == "match" { "MATCH".to_string() } else { e.class.clone().unwrap_or_else(|| "DIVERGE".into()) }),
                "vllm_note": vx.and_then(|e| e.note.clone()),
                "policy_tags": case.policy,
                "chunks": chunk_feed,
            }));
            let policy = if case.policy.is_empty() {
                String::new()
            } else {
                format!("<div class=pol>policy: {}</div>", case.policy.join(", "))
            };
            let dtodo = if dclass == "MATCH" {
                String::new()
            } else {
                dynamo_red += 1;
                format!("<hr><div class=todo>{}</div>", esc(todo_for(dclass)))
            };
            let dcell = cell(
                "Dynamo today (native unified, LIVE)",
                &case.input,
                &case.golden,
                &events_html(&got),
                if dclass == "MATCH" {
                    "match"
                } else {
                    "diverge"
                },
                dclass,
                &format!("{policy}{dtodo}"),
            );

            // vLLM: expected (from golden expect.vllm).
            let vx = case.expect.get("vllm").cloned().unwrap_or(Expect {
                verdict: "match".into(),
                class: None,
                note: None,
            });
            let vclass = if vx.verdict == "match" {
                "MATCH".to_string()
            } else {
                vx.class.clone().unwrap_or_else(|| "DIVERGE".into())
            };
            if vclass != "MATCH" {
                vllm_red += 1;
            }
            let vnote = vx
                .note
                .map(|n| format!("<hr><div class=note>{}</div>", esc(&n)))
                .unwrap_or_default();
            let vlive = vllm_live.get(&(file.family.clone(), id.clone()));
            let (vgot_html, vverdict, vclass_final) = match vlive {
                Some(ev) => {
                    let matches = ev == &case.golden;
                    (
                        events_html(ev),
                        if matches { "match" } else { "diverge" },
                        if matches {
                            "MATCH".to_string()
                        } else {
                            "DIVERGE".to_string()
                        },
                    )
                }
                // No capture for this case: say so. Do NOT fall back to the authored
                // expectation dressed up as a result — that is how this column spent
                // its life claiming MATCH with nothing behind it.
                None => (
                    "<i>(no vLLM capture for this case)</i>".to_string(),
                    "diverge",
                    "NO-DATA".to_string(),
                ),
            };
            let vcell = cell(
                "vLLM Rust 0.25.1 (LIVE)",
                &case.input,
                &case.golden,
                &vgot_html,
                vverdict,
                &vclass_final,
                &vnote,
            );

            let gtip = format!(
                "<b>GOLDEN</b> (authored oracle){}<hr><b>input</b><pre>{}</pre>{}",
                policy,
                esc(&case.input),
                events_html(&case.golden),
            );

            rows.push_str(&format!(
                "<tr><td class=case>{}<div class=desc>{}</div></td>\
                 <td class='c gold'>golden<div class=tip>{}</div></td>\
                 <td class='c {}'>{}<div class=tip>{}</div></td>\
                 <td class='c {}'>{}<div class=tip>{}</div></td></tr>",
                esc(id),
                esc(&case.description),
                gtip,
                css(&vcell.verdict),
                label(&vcell.verdict),
                vcell.tip,
                css(&dcell.verdict),
                label(&dcell.verdict),
                dcell.tip,
            ));
        }
    }

    // Machine-readable capture feed consumed by explode_unified_fixtures.py.
    // YAML so it reads like the rest of the conformance fixture corpus. conformance/unified/
    // is the gitignored build tree — create it (a fresh checkout won't have it; the
    // committed data is the self-contained family and capture YAML under conformance/fixtures-unified-v2/).
    let yaml_out = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("unified/unified_results.yaml");
    assert_eq!(
        common::dynamo_capture_provenance(Some(capture_provenance["label"].as_str().unwrap())),
        capture_provenance,
        "capture source changed during the run"
    );
    std::fs::create_dir_all(yaml_out.parent().unwrap()).unwrap();
    let feed = serde_json::json!({
        "schema": "unified-results/v1",
        "capture_provenance": capture_provenance,
        "note": "GOLDEN = authored oracle. dynamo = LIVE native UnifiedParser output. vllm = documented expectation when a live capture is unavailable.",
        "cases": json_cases,
    });
    std::fs::write(&yaml_out, serde_json::to_string_pretty(&feed).unwrap()).unwrap();
    eprintln!(
        "wrote {} ({total} cases, dynamo_red={dynamo_red}, vllm_red={vllm_red})",
        yaml_out.display()
    );

    // Sanity: the harness computes REAL failures, not a strawman. vLLM's
    // documented-expectation column supplies that signal (`vllm_red` is large
    // and asserted elsewhere via the rendered legend), so this only needs to
    // pin Dynamo's own invariant.
    assert!(total >= 14, "expected the seed corpus");
    parity_failures.extend(divergences::reconcile(
        &known,
        Check::Golden,
        &observed_dynamo,
    ));
    parity_failures.extend(divergences::reconcile(
        &known,
        Check::Projection,
        &observed_projection,
    ));
    assert_eq!(
        dynamo_red,
        observed_dynamo.len() + observed_projection.len(),
        "every red report cell must match one exact allowlisted golden or projection divergence"
    );
    assert!(
        parity_failures.is_empty(),
        "unified report has unexpected or stale golden divergences:\n\n{}",
        parity_failures.join("\n\n")
    );
}

/// One committed `dynamo_v2-<ver>/<family>/<key>.yaml` capture.
#[derive(Deserialize)]
struct CaptureDoc {
    family: String,
    cases: BTreeMap<String, CaptureCase>,
}

#[derive(Deserialize)]
struct CaptureCase {
    #[serde(default)]
    error: Option<String>,
    #[serde(default)]
    assembled: Vec<Ev>,
    #[serde(default)]
    chunks: Vec<CaptureChunk>,
}

#[derive(Deserialize)]
struct CaptureChunk {
    #[serde(default)]
    expected: Vec<Value>,
}

/// The scenario slug for each committed case key, read from the `inputs/` shard.
#[derive(Deserialize)]
struct InputDoc {
    family: String,
    cases: BTreeMap<String, InputCase>,
}

#[derive(Deserialize)]
struct InputCase {
    #[serde(default)]
    scenario: String,
    #[serde(default)]
    input: String,
    /// The guard must re-run each case under the SAME configuration the shard was
    /// captured with; re-running everything under the default would report false
    /// drift for every prefilled / guided-JSON case.
    #[serde(default)]
    init: Init,
    #[serde(default)]
    tools: Option<Value>,
    #[serde(default)]
    chunks: Option<Vec<capture::InputChunk>>,
}

/// GUARD: the COMMITTED Dynamo capture must equal what the parsers produce NOW.
///
/// The Unified tab is rendered by Python from the committed shard — it never runs
/// the Rust parsers. So changing a parser without re-capturing leaves the page
/// showing the OLD behavior while every Rust test still passes, and the two
/// disagree silently. (That is exactly what happened when the unified parser
/// landed: `unified_parity` was 33/33 green while the page still drew the split.)
///
/// This closes that gap: touch a parser, and the capture must be regenerated.
#[test]
fn committed_dynamo_capture_matches_the_live_parsers() {
    let root = common::ensure_unified_fixtures().join("unified");
    validate_committed_dynamo_capture(&root);
}

fn validate_committed_dynamo_capture(root: &std::path::Path) {
    if !root.join("inputs").is_dir() {
        panic!(
            "no committed unified fixtures under {} — extract them first",
            root.display()
        );
    }
    // Advancing the parser build does not relabel retained evidence. Compare the
    // newest recorded snapshot with this build and reject changed live outputs.
    let capture_dir = common::latest_recorded_capture_dir(root, "dynamo_v2-")
        .expect("no committed dynamo_v2-<ver> capture dir");

    validate_selected_dynamo_capture(root, &capture_dir);
}

fn validate_selected_dynamo_capture(root: &std::path::Path, capture_dir: &std::path::Path) {
    let input_dirs = shared_overlay_dirs(root, "inputs");
    let native_families: BTreeSet<String> = common::unified_families()
        .into_iter()
        .filter_map(|(name, family)| family.native.then_some(name))
        .collect();
    let native_family_args: Vec<String> = native_families.iter().cloned().collect();
    let validation = common::capture_stimulus_command()
        .arg("--validate-current")
        .arg(capture_dir)
        .arg("--families")
        .args(&native_family_args)
        .args(["--format", "json"])
        .arg("--inputs")
        .args(&input_dirs)
        .output()
        .expect("validate current capture stimulus");
    assert!(
        validation.status.success(),
        "current capture stimulus validation failed: {}",
        String::from_utf8_lossy(&validation.stderr)
    );
    let captures: Vec<CaptureDoc> =
        serde_json::from_slice(&validation.stdout).expect("validated effective capture records");

    // key -> (family, scenario, input, init, tools), from the base inputs shard plus
    // PR-qualified sparse overlays. New cases must carry their input metadata in
    // the same overlay as the capture, rather than making the released shard mutable.
    type CaptureKey = (String, String);
    type CaptureMetadata = InputCase;
    let mut meta: BTreeMap<CaptureKey, CaptureMetadata> = BTreeMap::new();
    for input_dir in input_dirs {
        for entry in glob_yaml(&input_dir) {
            let doc: InputDoc = serde_yaml::from_str(&std::fs::read_to_string(&entry).unwrap())
                .unwrap_or_else(|e| panic!("{}: {e}", entry.display()));
            // Split-only inputs feed legacy batch histories, not this native v2 capture.
            if !native_families.contains(&doc.family) {
                continue;
            }
            for (key, case) in doc.cases {
                // A sparse overlay can rename a case while preserving its scenario.
                // The newer key replaces the released key for capture validation.
                meta.retain(|(family, _), previous| {
                    family != &doc.family || previous.scenario != case.scenario
                });
                // v1_split scenarios have committed Dynamo v1 captures and are
                // intentionally absent from the current Dynamo v2 capture.
                if case.scenario.starts_with("v1_split_") {
                    continue;
                }
                meta.insert((doc.family.clone(), key), case);
            }
        }
    }

    let mut stale: Vec<String> = Vec::new();
    let mut checked = 0usize;
    let capture_keys: std::collections::BTreeSet<(String, String)> = captures
        .iter()
        .flat_map(|doc| {
            doc.cases
                .keys()
                .map(|key| (doc.family.clone(), key.clone()))
        })
        .collect();
    for key in meta.keys() {
        if !capture_keys.contains(key) {
            stale.push(format!(
                "{} [{}] has no current Dynamo capture",
                key.0, key.1
            ));
        }
    }
    for doc in captures {
        for (key, committed) in doc.cases {
            let Some(metadata) = meta.get(&(doc.family.clone(), key.clone())) else {
                stale.push(format!(
                    "{} [{key}] has no input metadata in inputs or its PR overlays",
                    doc.family
                ));
                continue;
            };
            checked += 1;
            let InputCase {
                scenario,
                input,
                init,
                tools: tool_schemas,
                chunks,
            } = metadata;
            let id = format!("UNIFIED.{scenario}.{}", doc.family);

            let case_tools = common::unified_tools_for_schemas(tool_schemas.as_ref());
            let schedule = capture::replay_chunks(input, chunks.as_deref());
            let (live_assembled, live_rows, _raw) =
                match capture_native_case(&doc.family, &schedule, init, &case_tools) {
                    Ok(value) => {
                        if committed.error.is_some() {
                            stale.push(format!("{id}: recorded error no longer occurs"));
                            continue;
                        }
                        value
                    }
                    Err(capture::CaptureFailure::Error(error))
                        if committed.error.as_ref() == Some(&error) =>
                    {
                        continue;
                    }
                    Err(failure) => {
                        stale.push(format!("{id}: unexpected live error {failure:?}"));
                        continue;
                    }
                };
            if live_assembled != committed.assembled {
                stale.push(format!(
                    "{id} [{key}] assembled\n    committed: {}\n         live: {}",
                    committed
                        .assembled
                        .iter()
                        .map(Ev::render)
                        .collect::<Vec<_>>()
                        .join("  |  "),
                    live_assembled
                        .iter()
                        .map(Ev::render)
                        .collect::<Vec<_>>()
                        .join("  |  "),
                ));
                continue;
            }
            // The page assembles the Dynamo column from these per-chunk deltas, so
            // they have to be current too — not just the assembled list.
            let live_chunks: Vec<Vec<Value>> = live_rows.into_iter().map(|r| r.deltas).collect();
            let committed_chunks: Vec<Vec<Value>> =
                committed.chunks.into_iter().map(|c| c.expected).collect();
            if live_chunks != committed_chunks {
                stale.push(format!("{id} [{key}] per-chunk deltas differ"));
            }
        }
    }

    assert!(checked > 0, "no committed capture cases were compared");
    assert!(
        stale.is_empty(),
        "{} of {checked} latest recorded Dynamo cases differ from the live parser. \
         Verify the parser against authored inputs/GOLDEN and run latest parser tests. \
         If the intended output changed, request an administrator update of the affected latest YAML; \
         contributors do not recapture historical versions.\n\n{}",
        stale.len(),
        stale.join("\n\n"),
    );
}

#[test]
fn plain_capture_records_reach_live_guard() {
    let root = std::env::temp_dir().join(format!("dynamo-plain-capture-{}", std::process::id()));
    let base = root.join("dynamo_v2-0.6.0");
    std::fs::create_dir_all(root.join("dynamo_v2-0.5.0")).unwrap();
    std::fs::create_dir_all(root.join("dynamo_v2-999.0.0+historical")).unwrap();
    let write = |directory: &std::path::Path, key: &str, doc: &Value| {
        std::fs::create_dir_all(directory.join("gemma4")).unwrap();
        std::fs::write(
            directory.join(format!("gemma4/{key}.yaml")),
            serde_json::to_vec(doc).unwrap(),
        )
        .unwrap();
    };
    let mut captures = BTreeMap::new();
    for key in ["old", "retained", "added"] {
        let chunks = dynamo_chunks("gemma4", key, &Init::default());
        let input_chunks: Vec<Value> = chunks
            .iter()
            .map(|row| json!({"delta_text":row.delta_text}))
            .collect();
        let stimulus = json!({"input":key,
            "init":{"starting_state":"None","tool_output_mode":"Native","named_tool":null},
            "finish_reason":"stop","tools":common::unified_tool_schemas(),"chunks":input_chunks});
        let mut input = stimulus.clone();
        input["scenario"] = json!(key);
        write(
            &root.join("inputs"),
            key,
            &json!({"family":"gemma4","cases":{key:input}}),
        );
        let output_chunks: Vec<Value> = chunks
            .into_iter()
            .map(|row| json!({"expected":row.deltas}))
            .collect();
        captures.insert(key, json!({"family":"gemma4","cases":{key:{
            "capture_input":stimulus,"assembled":[{"kind":"text","text":key}],"chunks":output_chunks}}}));
    }
    write(&base, "old", &captures["old"]);
    write(&base, "retained", &captures["retained"]);
    write(&base, "added", &captures["added"]);
    validate_committed_dynamo_capture(&root);

    std::fs::remove_file(base.join("gemma4/old.yaml")).unwrap();
    assert!(std::panic::catch_unwind(|| validate_committed_dynamo_capture(&root)).is_err());
    write(&base, "old", &captures["old"]);
    let mut wrong = captures["added"].clone();
    wrong["cases"]["added"]["assembled"] = json!([{"kind":"text","text":"wrong"}]);
    write(&base, "added", &wrong);
    let failure = std::panic::catch_unwind(|| validate_committed_dynamo_capture(&root))
        .expect_err("the live comparison must read the plain capture record");
    assert!(
        failure
            .downcast_ref::<String>()
            .unwrap()
            .contains("assembled")
    );
    std::fs::remove_dir_all(&root).unwrap();
}

#[test]
fn current_plain_version_reaches_live_guard_and_binds_tools() {
    if std::env::var_os("DYNAMO_SOURCE_SNAPSHOT_TEST_CHILD").is_none() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "current_plain_version_reaches_live_guard_and_binds_tools",
                "--nocapture",
            ])
            .env("DYNAMO_SOURCE_SNAPSHOT_TEST_CHILD", "1")
            .env("CONFORMANCE_DYNAMO_V2_LABEL", "current")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        return;
    }
    let root = std::env::temp_dir().join(format!("dynamo-source-snapshot-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let provenance = common::dynamo_capture_provenance(Some("current"));
    assert!(provenance["label"].as_str().unwrap().contains("+source."));
    let base = root.join(format!(
        "dynamo_v2-{}",
        provenance["crate_version"].as_str().unwrap()
    ));
    let make = |key: &str, text: &str, assembled: Value| {
        let init = Init::default();
        let chunks = dynamo_chunks("gemma4", text, &init);
        let input_chunks: Vec<Value> = chunks
            .iter()
            .map(|row| json!({"delta_text": row.delta_text}))
            .collect();
        let stimulus = json!({"input": text, "init": {"starting_state":"None", "tool_output_mode":"Native", "named_tool":null},
            "finish_reason":"stop", "tools":common::unified_tool_schemas(), "chunks":input_chunks});
        let input = json!({"family":"gemma4", "cases":{key: {
            "scenario":key, "input":text, "init":stimulus["init"], "tools":stimulus["tools"],
            "finish_reason":"stop", "chunks":stimulus["chunks"]}}});
        let output_chunks: Vec<Value> = chunks
            .into_iter()
            .map(|row| json!({"expected":row.deltas}))
            .collect();
        let capture = json!({"family":"gemma4", "capture_provenance":provenance, "cases":{key:{
            "capture_input":stimulus, "assembled":assembled, "chunks":output_chunks}}});
        (input, capture)
    };
    let write = |directory: &std::path::Path, key: &str, doc: &Value| {
        std::fs::create_dir_all(directory.join("gemma4")).unwrap();
        std::fs::write(
            directory.join(format!("gemma4/{key}.yaml")),
            serde_json::to_vec(doc).unwrap(),
        )
        .unwrap();
    };
    let (old_input, old_capture) = make(
        "old",
        "old text",
        json!([{"kind":"text","text":"old text"}]),
    );
    write(&root.join("inputs"), "old", &old_input);
    write(&base, "old", &old_capture);
    validate_committed_dynamo_capture(&root);
    let (new_input, new_capture) = make(
        "added",
        "<|tool_call>call:f{x:<|\"|>1<|\"|>}<tool_call|>",
        json!([{"kind":"tool_call","name":"f","arguments":{"x":"1"}}]),
    );
    write(&root.join("inputs"), "added", &new_input);
    write(&base, "added", &new_capture);
    validate_committed_dynamo_capture(&root);
    std::fs::remove_file(root.join("inputs/gemma4/old.yaml")).unwrap();
    std::fs::remove_file(base.join("gemma4/old.yaml")).unwrap();
    validate_committed_dynamo_capture(&root);
    let mut wrong = new_capture.clone();
    let terminal = wrong["cases"]["added"]["chunks"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .flat_map(|chunk| chunk["expected"].as_array_mut().unwrap())
        .find(|delta| delta["complete"] == true)
        .expect("the captured call must contain a completion delta");
    terminal["complete"] = json!(false);
    write(&base, "added", &wrong);
    let failure = std::panic::catch_unwind(|| validate_committed_dynamo_capture(&root))
        .expect_err("changing only completion must fail the per-chunk comparison");
    assert!(
        failure
            .downcast_ref::<String>()
            .unwrap()
            .contains("per-chunk deltas differ")
    );
    let mut wrong = new_capture.clone();
    wrong["cases"]["added"]["capture_input"]["tools"] = json!([]);
    write(&base, "added", &wrong);
    assert!(std::panic::catch_unwind(|| validate_committed_dynamo_capture(&root)).is_err());
    write(&base, "added", &new_capture);
    let mut wrong_input = new_input;
    wrong_input["cases"]["added"]["tools"] = json!([]);
    write(&root.join("inputs"), "added", &wrong_input);
    assert!(std::panic::catch_unwind(|| validate_committed_dynamo_capture(&root)).is_err());
    std::fs::remove_dir_all(&root).unwrap();
}

/// Return the released shared shard followed by its PR-qualified sparse overlays.
/// The overlays are ordered by PR number and then patch number so a later patch is
/// authoritative if a branch intentionally extends its own metadata.
fn shared_overlay_dirs(root: &std::path::Path, base: &str) -> Vec<PathBuf> {
    let mut overlays: Vec<(u64, u64, PathBuf)> = std::fs::read_dir(root)
        .into_iter()
        .flatten()
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| path.is_dir())
        .filter_map(|path| {
            let name = path.file_name()?.to_str()?;
            let rest = name.strip_prefix(&format!("{base}+pr"))?;
            let (pr, patch) = rest.split_once(".patch")?;
            Some((pr.parse().ok()?, patch.parse().ok()?, path))
        })
        .collect();
    overlays.sort_by_key(|(pr, patch, _)| (*pr, *patch));

    let mut dirs = vec![root.join(base)];
    dirs.extend(overlays.into_iter().map(|(_, _, path)| path));
    dirs
}

#[test]
fn release_qualified_capture_order_preserves_current_and_released_owners() {
    let prefix = "dynamo_v2-";
    let older_release = common::version_capture_sort_key("dynamo_v2-0.3.2", prefix).unwrap();
    let current_release = common::version_capture_sort_key("dynamo_v2-0.4.0", prefix).unwrap();
    assert!(older_release < current_release);
    assert!(common::version_capture_sort_key("dynamo_v2-0.3.3.patch1", prefix).is_none());
}

#[test]
fn shared_input_overlays_are_folded_after_the_released_shard() {
    let root = std::env::temp_dir().join(format!(
        "unified-render-overlay-order-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(root.join("inputs")).unwrap();
    std::fs::create_dir_all(root.join("inputs+pr166.patch10")).unwrap();
    std::fs::create_dir_all(root.join("inputs+pr166.patch2")).unwrap();
    std::fs::create_dir_all(root.join("inputs+pr167.patch1")).unwrap();

    let names: Vec<String> = shared_overlay_dirs(&root, "inputs")
        .into_iter()
        .map(|path| path.file_name().unwrap().to_string_lossy().into_owned())
        .collect();
    assert_eq!(
        names,
        [
            "inputs",
            "inputs+pr166.patch2",
            "inputs+pr166.patch10",
            "inputs+pr167.patch1",
        ]
    );
    std::fs::remove_dir_all(root).unwrap();
}

fn glob_yaml(dir: &std::path::Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for fam in std::fs::read_dir(dir).into_iter().flatten().flatten() {
        let p = fam.path();
        if !p.is_dir() {
            continue;
        }
        for f in std::fs::read_dir(&p).into_iter().flatten().flatten() {
            let fp = f.path();
            if fp.extension().and_then(|e| e.to_str()) == Some("yaml") {
                out.push(fp);
            }
        }
    }
    out.sort();
    out
}

fn label(v: &str) -> String {
    if v == "MATCH" {
        "✓".to_string()
    } else {
        format!("✗ {v}")
    }
}
fn css(v: &str) -> &'static str {
    if v == "MATCH" { "MATCH" } else { "RED" }
}

#[test]
fn case_tools_control_capture_types_without_changing_legacy_defaults() {
    // Exercise the authored Python YAML path as well as Rust deserialization;
    // dropping tools during emit_yaml would silently restore the string default.
    let output = std::process::Command::new("python3")
        .arg("-c")
        .arg(
            r#"
import sys
sys.path.insert(0, sys.argv[1])
import gen_unified_golden as g
case = {"description":"per-case schema", "policy":[], "init":{},
        "finish_reason":"stop", "input":"", "golden":[], "expect":{},
        "tools":[{"name":"f", "parameters":{"type":"object",
                 "properties":{"x":{"type":"integer"}}}}]}
g.build_cases = lambda family: {"custom": case}
print(g.emit_yaml("glm47"))
"#,
        )
        .arg(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("utils/src"))
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let mut file: GoldenFile = serde_yaml::from_slice(&output.stdout).unwrap();
    let case = file.cases.remove("custom").unwrap();
    let input = "<tool_call>f<arg_key>x</arg_key><arg_value>42</arg_value></tool_call>";
    let custom = common::unified_tools_for_schemas(case.tools.as_ref());
    let expected = vec![Ev::ToolCall {
        name: "f".into(),
        arguments: raw_arguments(r#"{"x":42}"#),
    }];
    assert_eq!(
        dynamo_events_with_tools("glm47", input, &case.init, &custom),
        expected
    );
    assert_ne!(
        dynamo_events_with_tools("glm47", input, &case.init, &tools()),
        expected
    );
    let rows = dynamo_chunks_with_tools("glm47", input, &case.init, &custom);
    assert!(rows.iter().flat_map(|row| &row.deltas).any(|delta| {
        delta["arguments"]
            .as_str()
            .is_some_and(|args| args.contains("42"))
    }));
    let legacy: GoldenCase = serde_json::from_value(json!({
        "description": "legacy schema", "input": "", "golden": [], "expect": {}
    }))
    .unwrap();
    assert_eq!(
        common::unified_tool_schemas_for_case(legacy.tools.as_ref()),
        common::unified_tool_schemas()
    );
    let input_case: InputCase = serde_json::from_value(json!({"tools":case.tools})).unwrap();
    assert_eq!(input_case.tools, case.tools);
}
