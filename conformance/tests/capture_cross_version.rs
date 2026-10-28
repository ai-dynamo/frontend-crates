// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Run an OLDER build's unified parser against the CURRENT corpus.
//!
//! Every committed `dynamo_v2-<ver>/` shard holds only the cases that existed when it
//! was taken, so a case added later shows "no data" on the older column and the table
//! cannot say what the old parser WOULD have done with it. That is the interesting
//! question for a behavior change: not "did the old build pass the old tests" but
//! "what does the old build do with the new ones".
//!
//! Usage — check this file out into a worktree at the OLD commit, then run it there
//! pointed at the CURRENT corpus:
//!
//! ```text
//! git worktree add --detach /tmp/old <old-sha>
//! cp conformance/tests/capture_cross_version.rs /tmp/old/conformance/tests/
//! cd /tmp/old && \
//!   XVER_INPUTS=<repo>/conformance/unified/inputs \
//!   XVER_OUT=<repo>/conformance/unified/dynamo_v2-<version> \
//!   XVER_LABEL=<version> \
//!   cargo test -p dynamo-conformance-fixtures-v2 --test capture_cross_version -- --nocapture
//! ```
//!
//! Plain versions require source equality with the release tag. `XVER_LABEL=current`
//! records an unpublished source-qualified label and its fingerprint. Copy `common/`
//! and `conformance/utils/src/unified_tools.json` to the same relative paths in the
//! historical worktree: the common helper includes the shared schemas at compile time.
//! Set `CONFORMANCE_DYNAMO_PROVENANCE_SCRIPT` to the current checker.
//!
//! For old builds without request initialization, compile with
//! `RUSTFLAGS='--cfg conformance_legacy_init --check-cfg=cfg(conformance_legacy_init)'`.
//! Non-default requests are then recorded as unavailable, never as default-mode data.
//! Builds predating UnifiedParser instead use `--cfg conformance_split_only`.
//! Builds without ToolCallDelta::complete also require `--cfg conformance_legacy_terminal`.
//! Builds without the reasoning-aware structural-tag API use `--cfg conformance_no_structural_tag`.
//! Builds without message-phase events use `--cfg conformance_legacy_message_phases`.
//!
//! No-op unless `XVER_INPUTS` is set, so it costs nothing in a normal test run.

#![allow(unexpected_cfgs)] // Historical runner cfg is intentionally not a crate feature.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use dynamo_parsers::{ReasoningParser, ReasoningParserType};
use dynamo_parsers_v2::create_tool_parser_for_family;
#[cfg(not(conformance_split_only))]
use dynamo_parsers_v2::{
    REGISTERED_UNIFIED_FAMILIES, UnifiedEvent, assemble, create_unified_parser_for_family,
};
use serde_json::{Value, json};

mod common;
use common::unified_capture::{self as capture, CaptureFailure, capture_step, chunk_input};
use common::unified_tools as tools;

type Captured = (Vec<Vec<serde_yaml::Value>>, Vec<serde_yaml::Value>);

fn requested_chunks(
    value: &serde_yaml::Value,
    family: &str,
    case_id: &str,
) -> Option<Vec<capture::InputChunk>> {
    let has_numeric_delta = value.as_sequence().is_some_and(|rows| {
        rows.iter().any(|row| {
            row.as_mapping()
                .and_then(|mapping| mapping.get(serde_yaml::Value::String("delta_text".into())))
                .is_some_and(|delta| matches!(delta, serde_yaml::Value::Number(_)))
        })
    });
    if has_numeric_delta {
        eprintln!("[xver] {family}/{case_id}: numeric YAML delta_text; deriving chunks from input");
        return None;
    }
    Some(
        serde_yaml::from_value(value.clone())
            .unwrap_or_else(|error| panic!("requested chunks for {family}/{case_id}: {error}")),
    )
}

#[test]
fn numeric_yaml_chunk_text_uses_input_schedule() {
    let chunks: serde_yaml::Value = serde_yaml::from_str("- delta_text: 4.2e1").unwrap();
    assert!(requested_chunks(&chunks, "deepseek_v4", "numeric_exponent").is_none());
}

fn capture_record(
    result: Result<Captured, CaptureFailure>,
    init: &common::Init,
) -> serde_yaml::Value {
    let mut record = serde_yaml::Mapping::new();
    record.insert("init".into(), common::json_to_yaml_value(init));
    match result {
        Ok((rows, assembled)) => {
            record.insert("assembled".into(), serde_yaml::Value::Sequence(assembled));
            let chunks = rows
                .into_iter()
                .map(|expected| {
                    let mut row = serde_yaml::Mapping::new();
                    row.insert("expected".into(), serde_yaml::Value::Sequence(expected));
                    serde_yaml::Value::Mapping(row)
                })
                .collect();
            record.insert("chunks".into(), serde_yaml::Value::Sequence(chunks));
        }
        Err(CaptureFailure::Unavailable(reason)) => {
            record.insert("unavailable".into(), reason.into());
        }
        Err(CaptureFailure::Error(error)) => {
            record.insert("error".into(), error.into());
        }
    }
    serde_yaml::Value::Mapping(record)
}

fn requires_request_init(init: &common::Init) -> bool {
    !matches!(init.starting_state.as_str(), "" | "None")
        || !matches!(init.tool_output_mode.as_str(), "" | "Native")
        || init.named_tool.is_some()
        || init.structural_tag.is_some()
}

fn unavailable_init(init: &common::Init, native: bool, family: &str) -> bool {
    requires_request_init(init)
        && (!native || cfg!(any(conformance_legacy_init, conformance_split_only)))
        || cfg!(conformance_muse_guided_unsupported)
            && family == "muse_glimmer"
            && init.tool_output_mode == "GuidedJson"
}

#[cfg(not(any(conformance_legacy_init, conformance_split_only)))]
fn apply_init(
    parser: &mut Box<dyn dynamo_parsers_v2::UnifiedParser>,
    init: &common::Init,
) -> Result<(), CaptureFailure> {
    capture_step(init.try_apply(parser), "initialize_request")
}

#[cfg(all(conformance_legacy_init, not(conformance_split_only)))]
fn apply_init(
    _parser: &mut Box<dyn dynamo_parsers_v2::UnifiedParser>,
    init: &common::Init,
) -> Result<(), CaptureFailure> {
    assert!(
        !requires_request_init(init),
        "unsupported request reached legacy parser"
    );
    Ok(())
}

#[test]
fn capture_request_init_availability() {
    let guided = common::Init {
        tool_output_mode: "GuidedJson".into(),
        ..Default::default()
    };
    assert_eq!(
        unavailable_init(&guided, true, "muse_glimmer"),
        cfg!(any(
            conformance_muse_guided_unsupported,
            conformance_legacy_init,
            conformance_split_only
        ))
    );
    for init in [
        common::Init::default(),
        common::Init {
            starting_state: "None".into(),
            tool_output_mode: "Native".into(),
            named_tool: None,
            ..Default::default()
        },
    ] {
        assert!(!unavailable_init(&init, false, "gemma4"));
        assert!(!unavailable_init(&init, true, "gemma4"));
    }
    for init in [
        common::Init {
            starting_state: "Reasoning".into(),
            ..Default::default()
        },
        common::Init {
            starting_state: "Response".into(),
            ..Default::default()
        },
        common::Init {
            tool_output_mode: "GuidedJson".into(),
            ..Default::default()
        },
        common::Init {
            named_tool: Some("get_weather".into()),
            ..Default::default()
        },
    ] {
        assert!(unavailable_init(&init, false, "gemma4"));
        assert_eq!(
            unavailable_init(&init, true, "gemma4"),
            cfg!(any(conformance_legacy_init, conformance_split_only))
        );
    }
}

#[test]
fn capture_producer_records_init_and_identity() {
    // This exercises serialization and the producer loop, not just Init::apply:
    // omitting the apply call still passes parser-level initialization tests.
    let scratch =
        std::env::temp_dir().join(format!("dynamo-capture-provenance-{}", std::process::id()));
    let inputs = scratch.join("inputs/gemma4");
    std::fs::create_dir_all(&inputs).unwrap();
    let fixture = json!({"cases": {
        "guided": {"input": "{\"city\":\"Paris\"}", "init": {
            "tool_output_mode": "GuidedJson", "named_tool": "get_weather"
        }},
        "reasoning": {"input": "hidden<channel|>visible", "init": {"starting_state": "Reasoning"},
            "chunks": [{"delta_text":"hidden<channel|>visible"}, {"delta_text":""}, {"delta_text":"‹finish›"}]},
        "native": {"input": "plain response"}
    }});
    std::fs::write(
        inputs.join("init.yaml"),
        serde_yaml::to_string(&fixture).unwrap(),
    )
    .unwrap();
    let missing = scratch.join("inputs/not_a_parser");
    std::fs::create_dir_all(&missing).unwrap();
    std::fs::write(
        missing.join("missing.yaml"),
        "cases:\n  absent:\n    input: test\n",
    )
    .unwrap();
    let output = scratch.join("output");
    let result = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "capture_this_build_against_the_current_corpus",
            "--nocapture",
        ])
        .env("XVER_INPUTS", scratch.join("inputs"))
        .env("XVER_OUT", &output)
        .env("XVER_LABEL", "current")
        .env(
            "XVER_FAMILIES",
            Path::new(env!("CARGO_MANIFEST_DIR")).join("utils/src/parser_families.yaml"),
        )
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    );
    for key in ["guided", "reasoning", "native"] {
        let doc: Value = serde_yaml::from_str(
            &std::fs::read_to_string(output.join(format!("gemma4/{key}.yaml"))).unwrap(),
        )
        .unwrap();
        assert_eq!(doc["capture_provenance"]["kind"], "unpublished");
        assert_eq!(
            doc["captured_with"]["dynamo_v2"],
            doc["capture_provenance"]["label"]
        );
        assert!(
            doc["captured_with"]["dynamo_v2"]
                .as_str()
                .unwrap()
                .ends_with(doc["capture_provenance"]["source_sha256"].as_str().unwrap())
        );
        let case = &doc["cases"][key];
        assert_eq!(
            case["capture_input"]["input"],
            fixture["cases"][key]["input"]
        );
        if let Some(requested) = fixture["cases"][key].get("chunks") {
            assert_eq!(&case["capture_input"]["chunks"], requested);
            assert_eq!(
                case["chunks"].as_array().unwrap().len(),
                requested.as_array().unwrap().len()
            );
        }
        let expected_tools = fixture["cases"][key]
            .get("tools")
            .cloned()
            .unwrap_or_else(common::unified_tool_schemas);
        assert_eq!(case["capture_input"]["tools"], expected_tools);
        assert_eq!(
            case["capture_input"]["chunks"]
                .as_array()
                .unwrap()
                .last()
                .unwrap(),
            &json!({"delta_text":"‹finish›"})
        );
        if cfg!(any(conformance_legacy_init, conformance_split_only)) && key != "native" {
            assert!(case["unavailable"].as_str().is_some());
            assert!(case.get("assembled").is_none());
        } else {
            let expected = match key {
                "guided" => {
                    json!([{"kind":"tool_call", "name":"get_weather", "arguments":r#"{"city":"Paris"}"#}])
                }
                "reasoning" => {
                    json!([{"kind":"reasoning", "text":"hidden"}, {"kind":"text", "text":"visible"}])
                }
                _ => json!([{"kind":"text", "text":"plain response"}]),
            };
            assert_eq!(case["assembled"], expected, "{key}");
        }
    }
    let missing_doc: Value = serde_yaml::from_str(
        &std::fs::read_to_string(output.join("not_a_parser/absent.yaml")).unwrap(),
    )
    .unwrap();
    assert!(
        missing_doc["cases"]["absent"]["unavailable"]
            .as_str()
            .unwrap()
            .contains("no parser")
    );
    assert!(missing_doc["cases"]["absent"].get("assembled").is_none());
    std::fs::remove_dir_all(scratch).unwrap();
}

#[test]
fn capture_failures_are_not_successful_empty_outputs() {
    let init = common::Init::default();
    for stage in [
        "native push",
        "native finish",
        "split chunk push",
        "split reasoning tail push",
        "split chunk finish",
        "split assembled push",
        "split assembled finish",
    ] {
        let record = capture_record(
            capture_step::<Captured, _>(Err("probe failure"), stage),
            &init,
        );
        assert_eq!(
            record["error"].as_str(),
            Some(format!("{stage}: probe failure").as_str())
        );
        assert!(record.get("assembled").is_none());
        assert!(record.get("chunks").is_none());
    }
    let unavailable = split_path_capture_with_parsers("gemma4", "not_a_parser", "text", &tools())
        .map(|(rows, assembled)| (rows.into_iter().map(|_| Vec::new()).collect(), assembled));
    assert!(
        capture_record(unavailable, &init)["unavailable"]
            .as_str()
            .unwrap()
            .contains("not_a_parser")
    );
    let empty = capture_record(Ok((Vec::new(), Vec::new())), &init);
    assert!(empty["assembled"].as_sequence().unwrap().is_empty());
    assert!(empty.get("error").is_none());
}

#[cfg(not(any(conformance_legacy_init, conformance_split_only)))]
#[test]
fn native_parser_errors_reach_capture_records() {
    struct FailingParser {
        on_finish: bool,
    }
    impl dynamo_parsers_v2::UnifiedParser for FailingParser {
        fn parse_into(
            &mut self,
            _delta: &str,
            _output: &mut dynamo_parsers_v2::UnifiedParserOutput,
        ) -> anyhow::Result<()> {
            if self.on_finish {
                Ok(())
            } else {
                anyhow::bail!("push probe")
            }
        }
        fn finish(&mut self) -> anyhow::Result<dynamo_parsers_v2::UnifiedParserOutput> {
            anyhow::bail!("finish probe")
        }
    }
    for on_finish in [false, true] {
        let mut parser: Box<dyn dynamo_parsers_v2::UnifiedParser> =
            Box::new(FailingParser { on_finish });
        let init = common::Init::default();
        let record = capture_record(native_capture(&mut parser, &["text".into()], &init), &init);
        assert_eq!(
            record["error"].as_str().unwrap(),
            if on_finish {
                "native finish: finish probe"
            } else {
                "native push: push probe"
            }
        );
        assert!(record.get("assembled").is_none());
    }
}

#[test]
fn current_capture_selector_uses_plain_semantic_version() {
    let provenance = common::dynamo_capture_provenance(None);
    let root = std::env::temp_dir().join(format!("dynamo-current-selector-{}", std::process::id()));
    let selected = root.join(format!(
        "dynamo_v2-{}",
        provenance["crate_version"].as_str().unwrap()
    ));
    let historical = root.join("dynamo_v2-0.0.1");
    std::fs::create_dir_all(&historical).unwrap();
    assert!(
        std::panic::catch_unwind(|| common::version_dirs_ascending_with_current(
            &root,
            "dynamo_v2-",
            common::UNIFIED_DYNAMO_V2_CURRENT_CAPTURE,
        ))
        .is_err()
    );
    std::fs::create_dir_all(&selected).unwrap();
    let dirs = common::version_dirs_ascending_with_current(
        &root,
        "dynamo_v2-",
        common::UNIFIED_DYNAMO_V2_CURRENT_CAPTURE,
    );
    assert_eq!(dirs, vec![historical, selected]);
    std::fs::remove_dir_all(root).unwrap();
}

/// Corpus family -> (v1 reasoning parser, v2 tool parser) for the SPLIT path, read from
/// the `unified:` block of `parser_families.yaml`.
///
/// This used to be a hand-kept copy of `unified_render::parsers_for`, which is exactly
/// the duplication this file cannot afford: it gets copied into an OLDER worktree to run
/// against that build, so a stale copy would silently map a family to the wrong parser
/// there. `XVER_FAMILIES` points at the CURRENT manifest, so the old tree is driven by
/// today's declarations rather than whatever it happened to ship.
fn parsers_for(family: &str) -> Option<(String, String)> {
    // A MISSING ROW is a legitimate "this family is not declared" and returns None. A
    // missing, unreadable or malformed MANIFEST is not — it would make every split-path
    // family look undeclared and the run would report "no unified parser in this build"
    // while quietly capturing nothing. Those cases panic with the path in the message.
    let path = std::env::var("XVER_FAMILIES").unwrap_or_else(|_| {
        panic!("XVER_FAMILIES is required: point it at the CURRENT conformance/utils/src/parser_families.yaml")
    });
    let text =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("XVER_FAMILIES read {path}: {e}"));
    let doc: serde_yaml::Value =
        serde_yaml::from_str(&text).unwrap_or_else(|e| panic!("XVER_FAMILIES parse {path}: {e}"));
    let unified = doc
        .get("unified")
        .unwrap_or_else(|| panic!("{path} has no `unified:` block"));
    let row = unified.get(family)?; // not declared for the unified tab — genuinely skippable
    let get = |k: &str| {
        row.get(k)
            .and_then(|v| v.as_str())
            .unwrap_or_else(|| panic!("{path}: `unified.{family}` has no `{k}`"))
            .to_string()
    };
    Some((get("reasoning_parser"), get("tool_parser")))
}

fn tool_deltas(res: &dynamo_parsers_v2::ToolParseResult, out: &mut Vec<Value>) {
    if !res.normal_text.is_empty() {
        out.push(json!({"kind": "text", "text": res.normal_text}));
    }
    for c in &res.calls {
        out.push(tool_delta_json(c));
    }
}

fn tool_delta_json(c: &dynamo_parsers_v2::ToolCallDelta) -> Value {
    let mut value = json!({"kind": "tool_call", "name": c.name, "arguments": c.arguments});
    #[cfg(not(conformance_legacy_terminal))]
    {
        value["complete"] = json!(c.complete);
    }
    #[cfg(conformance_legacy_terminal)]
    {
        value["terminal_metadata_unavailable"] = json!("this build has no ToolCallDelta::complete");
    }
    value
}

#[cfg(not(conformance_legacy_terminal))]
#[test]
fn capture_preserves_tool_terminal_metadata() {
    for complete in [false, true] {
        let delta = dynamo_parsers_v2::ToolCallDelta {
            tool_index: 0,
            name: Some("f".into()),
            arguments: "{}".into(),
            complete,
        };
        assert_eq!(tool_delta_json(&delta)["complete"], json!(complete));
        #[cfg(not(conformance_split_only))]
        assert_eq!(
            delta_to_yaml(&dynamo_parsers_v2::UnifiedParserEvent::ToolCall(delta))["complete"],
            serde_yaml::Value::Bool(complete)
        );
    }
}

#[cfg(conformance_legacy_terminal)]
#[test]
fn capture_marks_legacy_terminal_metadata_unavailable() {
    let mut parser =
        create_tool_parser_for_family("qwen3_coder", &tools()).expect("qwen3 tool parser");
    let mut result = parser
        .push("<tool_call><function=f><parameter=x>ok</parameter></function></tool_call>")
        .expect("push");
    result.calls.extend(parser.finish().expect("finish").calls);
    assert!(!result.calls.is_empty());
    for delta in &result.calls {
        let record = tool_delta_json(delta);
        assert!(record.get("complete").is_none());
        assert!(record["terminal_metadata_unavailable"].as_str().is_some());
    }
}

fn native_family_registered(family: &str) -> bool {
    #[cfg(not(conformance_split_only))]
    {
        REGISTERED_UNIFIED_FAMILIES.contains(&family)
    }
    #[cfg(conformance_split_only)]
    {
        let _ = family;
        false
    }
}

fn family_registered(family: &str) -> bool {
    native_family_registered(family)
        || parsers_for(family).is_some_and(|(reason, tool)| {
            tool_family_registered(&tool)
                && dynamo_parsers::reasoning::get_available_reasoning_parsers()
                    .contains(&reason.as_str())
        })
}

fn tool_family_registered(family: &str) -> bool {
    #[cfg(not(conformance_legacy_registry))]
    {
        dynamo_parsers_v2::REGISTERED_FAMILIES.contains(&family)
    }
    #[cfg(conformance_legacy_registry)]
    {
        let names: Vec<String> =
            serde_json::from_str(&std::env::var("XVER_TOOL_FAMILIES").expect("artifact registry"))
                .expect("registry JSON");
        names.iter().any(|name| name == family)
    }
}

fn split_path_chunks_scheduled(
    reasoning_name: &str,
    tool_family: &str,
    chunks: &[String],
    tool_schemas: &[dynamo_parsers_v2::Tool],
) -> Result<Vec<Vec<Value>>, CaptureFailure> {
    if !tool_family_registered(tool_family)
        || !dynamo_parsers::reasoning::get_available_reasoning_parsers().contains(&reasoning_name)
    {
        return Err(CaptureFailure::Unavailable(format!(
            "no registered split pipeline {reasoning_name} + {tool_family}"
        )));
    }
    let mut rp = ReasoningParserType::get_reasoning_parser_from_name(reasoning_name);
    let mut tp = create_tool_parser_for_family(tool_family, tool_schemas).map_err(|error| {
        CaptureFailure::Error(format!("registered split parser {tool_family}: {error:#}"))
    })?;

    let mut rows = Vec::new();
    for chunk in chunks {
        let mut deltas: Vec<Value> = Vec::new();
        let rr = rp.parse_reasoning_streaming_incremental(chunk, &[]);
        if !rr.reasoning_text.is_empty() {
            deltas.push(json!({"kind": "reasoning", "text": rr.reasoning_text}));
        }
        if !rr.normal_text.is_empty() {
            let tr = capture_step(tp.push(&rr.normal_text), "split chunk push")?;
            tool_deltas(&tr, &mut deltas);
        }
        rows.push(deltas);
    }
    // Flush in the same order the live harness uses: reasoning tail -> tool -> finish.
    let mut tail: Vec<Value> = Vec::new();
    let rf = rp.finish_reasoning_stream();
    if !rf.reasoning_text.is_empty() {
        tail.push(json!({"kind": "reasoning", "text": rf.reasoning_text}));
    }
    if !rf.normal_text.is_empty() {
        let tr = capture_step(tp.push(&rf.normal_text), "split reasoning tail push")?;
        tool_deltas(&tr, &mut tail);
    }
    tool_deltas(&capture_step(tp.finish(), "split chunk finish")?, &mut tail);
    rows.push(tail);
    Ok(rows)
}

#[cfg(not(conformance_split_only))]
fn ev_to_yaml(ev: &UnifiedEvent) -> serde_yaml::Value {
    common::unified_event_to_yaml_value(ev)
}

/// Fold per-chunk deltas into assembled events, mirroring the page's `_assemble_stream`:
/// consecutive same-kind text/reasoning runs concatenate, and a tool_call delta carrying
/// a name opens a call while later nameless fragments append to its argument string.
fn fold_chunks(rows: &[Vec<Value>]) -> Vec<serde_yaml::Value> {
    let mut out: Vec<Value> = Vec::new();
    let mut raw: Vec<String> = Vec::new(); // argument text per open call, by position in `out`
    for row in rows {
        for d in row {
            let kind = d.get("kind").and_then(|k| k.as_str()).unwrap_or("");
            let text = d.get("text").and_then(|t| t.as_str()).unwrap_or("");
            match kind {
                "reasoning" | "text" => {
                    let same = out
                        .last()
                        .and_then(|l| l.get("kind"))
                        .and_then(|k| k.as_str())
                        == Some(kind);
                    if same {
                        let last = out.last_mut().unwrap();
                        let joined = format!("{}{}", last["text"].as_str().unwrap_or(""), text);
                        last["text"] = Value::String(joined);
                    } else {
                        out.push(json!({"kind": kind, "text": text}));
                        raw.push(String::new());
                    }
                }
                "tool_call" => {
                    let name = d.get("name").and_then(|n| n.as_str()).unwrap_or("");
                    let args = d.get("arguments").and_then(|a| a.as_str()).unwrap_or("");
                    let open_last = out
                        .last()
                        .and_then(|l| l.get("kind"))
                        .and_then(|k| k.as_str())
                        == Some("tool_call")
                        && name.is_empty();
                    if !open_last {
                        out.push(json!({"kind": "tool_call", "name": name, "arguments": ""}));
                        raw.push(String::new());
                    }
                    if let Some(r) = raw.last_mut() {
                        r.push_str(args);
                    }
                }
                _ => {}
            }
        }
    }
    // Argument text is a JSON fragment stream; parse once at the end, and keep it as a
    // string when it never became valid JSON rather than dropping what the parser said.
    for (i, ev) in out.iter_mut().enumerate() {
        if ev.get("kind").and_then(|k| k.as_str()) == Some("tool_call") {
            let r = raw.get(i).map(String::as_str).unwrap_or("");
            ev["arguments"] = if r.trim().is_empty() {
                json!({})
            } else {
                serde_json::from_str(r).unwrap_or_else(|_| Value::String(r.to_string()))
            };
        }
    }
    out.into_iter()
        .map(|v| common::unified_event_to_yaml_value(&v))
        .collect()
}

fn split_path_capture_with_parsers(
    reasoning_name: &str,
    tool_family: &str,
    input: &str,
    tool_schemas: &[dynamo_parsers_v2::Tool],
) -> Result<(Vec<Vec<Value>>, Vec<serde_yaml::Value>), CaptureFailure> {
    split_path_capture_with_schedule(
        reasoning_name,
        tool_family,
        input,
        &chunk_input(input),
        tool_schemas,
    )
}

fn split_path_capture_with_schedule(
    reasoning_name: &str,
    tool_family: &str,
    input: &str,
    chunks: &[String],
    tool_schemas: &[dynamo_parsers_v2::Tool],
) -> Result<(Vec<Vec<Value>>, Vec<serde_yaml::Value>), CaptureFailure> {
    let rows = split_path_chunks_scheduled(reasoning_name, tool_family, chunks, tool_schemas)?;

    // The split serving path assembles reasoning over the WHOLE input before it
    // streams the leftover through the tool parser. Its raw chunk evidence comes
    // from the incremental APIs, but folding those rows invents interleaving the
    // assembled contract cannot represent and rewrites released capture behavior.
    let mut rp = ReasoningParserType::get_reasoning_parser_from_name(reasoning_name);
    let split = rp.detect_and_parse_reasoning(input, &[]);
    let mut assembled_rows = Vec::new();
    if !split.reasoning_text.is_empty() {
        assembled_rows.push(vec![json!({
            "kind": "reasoning",
            "text": split.reasoning_text,
        })]);
    }
    let mut tp = create_tool_parser_for_family(tool_family, tool_schemas).map_err(|error| {
        CaptureFailure::Error(format!("registered split parser {tool_family}: {error:#}"))
    })?;
    for ch in split.normal_text.chars() {
        let mut buf = [0u8; 4];
        let mut deltas = Vec::new();
        tool_deltas(
            &capture_step(tp.push(ch.encode_utf8(&mut buf)), "split assembled push")?,
            &mut deltas,
        );
        if !deltas.is_empty() {
            assembled_rows.push(deltas);
        }
    }
    let mut tail = Vec::new();
    tool_deltas(
        &capture_step(tp.finish(), "split assembled finish")?,
        &mut tail,
    );
    if !tail.is_empty() {
        assembled_rows.push(tail);
    }
    let assembled = fold_chunks(&assembled_rows);
    Ok((rows, assembled))
}

#[cfg(not(conformance_split_only))]
fn native_capture(
    parser: &mut Box<dyn dynamo_parsers_v2::UnifiedParser>,
    chunks: &[String],
    init: &common::Init,
) -> Result<Captured, CaptureFailure> {
    apply_init(parser, init)?;
    let captured = capture::native_capture(parser, chunks)?;
    let deltas: Vec<_> = captured.iter().flatten().cloned().collect();
    let rows = captured
        .iter()
        .map(|row| row.iter().map(delta_to_yaml).collect())
        .collect();
    Ok((rows, assemble(&deltas).iter().map(ev_to_yaml).collect()))
}

fn capture_case_with_chunks(
    family: &str,
    input: &str,
    chunks: &[String],
    init: &common::Init,
    tool_schemas: &[dynamo_parsers_v2::Tool],
) -> Result<Captured, CaptureFailure> {
    #[cfg(not(conformance_split_only))]
    let native_error = match create_unified_parser_for_family(family, tool_schemas) {
        Ok(mut parser) => {
            if cfg!(conformance_no_structural_tag) && init.structural_tag.is_some() {
                return Err(CaptureFailure::Unavailable(
                    "this build has no reasoning-aware structural-tag request API".into(),
                ));
            }
            if unavailable_init(init, true, family) {
                return Err(CaptureFailure::Unavailable(
                    "this build cannot apply the requested initialization".into(),
                ));
            }
            #[cfg(not(conformance_legacy_init))]
            capture_step(
                init.validate_capture(family, tool_schemas, chunks),
                "structural tag",
            )?;
            return native_capture(&mut parser, chunks, init);
        }
        Err(error) if REGISTERED_UNIFIED_FAMILIES.contains(&family) => {
            return Err(CaptureFailure::Error(format!(
                "registered native parser construction: {error:#}"
            )));
        }
        Err(error) => format!("{error:#}"),
    };
    #[cfg(conformance_split_only)]
    let native_error = "this build predates UnifiedParser";
    let (reasoning, tool) = parsers_for(family).ok_or_else(|| {
        CaptureFailure::Unavailable(format!("no parser for {family}: {native_error}"))
    })?;
    if unavailable_init(init, false, family) {
        return Err(CaptureFailure::Unavailable(
            "split capture cannot apply the requested initialization".into(),
        ));
    }
    let (rows, assembled) =
        split_path_capture_with_schedule(&reasoning, &tool, input, chunks, tool_schemas)?;
    let rows = rows
        .into_iter()
        .map(|row| {
            row.into_iter()
                .map(|value| common::json_to_yaml_value(&value))
                .collect()
        })
        .collect();
    Ok((rows, assembled))
}

#[test]
fn split_capture_assembles_reasoning_over_the_whole_input() {
    let input = "<|channel>thought\nLook it up.<channel|><|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>}<tool_call|><|channel>thought\nNow answer.<channel|>It's 18C.";
    let (_, assembled) =
        split_path_capture_with_parsers("gemma4", "gemma4", input, &tools()).expect("split path");
    let want = vec![
        common::json_to_yaml_value(&json!({"kind": "reasoning", "text": "Look it up.Now answer."})),
        common::json_to_yaml_value(
            &json!({"kind": "tool_call", "name": "get_weather", "arguments": r#"{"city":"Paris"}"#}),
        ),
        common::json_to_yaml_value(&json!({"kind": "text", "text": "It's 18C."})),
    ];
    assert_eq!(assembled, want);
}

#[test]
fn json_to_yaml_preserves_large_integers_and_yaml_incompatible_floats() {
    let value: Value = serde_json::from_str(r#"{"value":9007199254740993,"tiny":1e-400}"#).unwrap();
    let yaml = common::json_to_yaml_value(&value);
    assert_eq!(yaml["value"].as_u64(), Some(9_007_199_254_740_993));
    assert_eq!(yaml["tiny"].as_str(), Some("1e-400"));
    for raw in ["0e1", "0.0e-400"] {
        let value: Value = serde_json::from_str(raw).unwrap();
        assert!(common::json_to_yaml_value(&value).is_number());
    }
}

#[test]
fn captured_event_arguments_round_trip_without_yaml_number_conversion() {
    for raw in [
        r#"{"fraction":9007199254740992.5,"tiny":1e-400,"huge":1e400}"#,
        r#"{"values":[9007199254740993,-9007199254740993],"text":"3.5"}"#,
        "null",
        r#""null""#,
        r#""[1]""#,
    ] {
        let arguments: Value = serde_json::from_str(raw).unwrap();
        let event = json!({"kind": "tool_call", "name": "inspect", "arguments": arguments});
        let yaml = common::unified_event_to_yaml_value(&event);
        let captured = yaml["arguments"].as_str().expect("raw JSON arguments");
        assert_eq!(serde_json::from_str::<Value>(captured).unwrap(), arguments);
    }
}

/// Per-chunk rows record RAW deltas, not assembled events — `arguments` stays the
/// literal fragment the parser emitted. Mirrors `unified_render::unified_delta_json`.
/// (Assembling per chunk instead produces a mapping and makes every case look changed.)
#[cfg(all(not(conformance_split_only), not(conformance_legacy_delta)))]
fn delta_to_yaml(d: &dynamo_parsers_v2::UnifiedParserEvent) -> serde_yaml::Value {
    let v = match d {
        #[cfg(not(conformance_legacy_message_phases))]
        dynamo_parsers_v2::UnifiedParserEvent::MessageStart(phase) => {
            serde_json::json!({"kind":"message_start","phase":phase})
        }
        #[cfg(not(conformance_legacy_message_phases))]
        dynamo_parsers_v2::UnifiedParserEvent::MessageEnd => {
            serde_json::json!({"kind":"message_end"})
        }
        dynamo_parsers_v2::UnifiedParserEvent::Reasoning(text) => {
            serde_json::json!({"kind": "reasoning", "text": text})
        }
        dynamo_parsers_v2::UnifiedParserEvent::Text(text) => {
            serde_json::json!({"kind": "text", "text": text})
        }
        dynamo_parsers_v2::UnifiedParserEvent::ToolCall(c) => tool_delta_json(c),
    };
    common::json_to_yaml_value(&v)
}

#[cfg(all(not(conformance_split_only), conformance_legacy_delta))]
fn delta_to_yaml(d: &dynamo_parsers_v2::UnifiedDelta) -> serde_yaml::Value {
    let value = match d {
        dynamo_parsers_v2::UnifiedDelta::Reasoning { text } => {
            json!({"kind": "reasoning", "text": text})
        }
        dynamo_parsers_v2::UnifiedDelta::Text { text } => json!({"kind": "text", "text": text}),
        dynamo_parsers_v2::UnifiedDelta::ToolCall(call) => tool_delta_json(call),
    };
    serde_yaml::to_value(value).expect("delta")
}

#[test]
fn capture_rejects_missing_or_malformed_input_fields() {
    let scratch =
        std::env::temp_dir().join(format!("dynamo-capture-invalid-{}", std::process::id()));
    for (index, fixture, reason) in [
        (0, "{}", "cases mapping"),
        (1, "cases: {1: {input: text}}", "case ID"),
        (2, "cases: {'': {input: text}}", "case ID"),
        (3, "cases: {probe: {}}", "string input"),
        (4, "cases: {probe: {input: 123}}", "string input"),
        (
            5,
            "cases: {probe: {input: text, tools: 123}}",
            "case tools schema",
        ),
        (
            6,
            "cases: {probe: {input: text, chunks: []}}",
            "authored input_chunks must reproduce input",
        ),
        (
            7,
            "cases: {probe: {input: text, finish_reason: 123}}",
            "string finish_reason",
        ),
    ] {
        let root = scratch.join(index.to_string());
        let inputs = root.join("inputs/gemma4");
        std::fs::create_dir_all(&inputs).unwrap();
        std::fs::write(inputs.join("invalid.yaml"), fixture).unwrap();
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "capture_this_build_against_the_current_corpus",
                "--nocapture",
            ])
            .env("XVER_INPUTS", root.join("inputs"))
            .env("XVER_OUT", root.join("output"))
            .env("XVER_LABEL", "current")
            .output()
            .unwrap();
        assert!(!output.status.success());
        assert!(
            String::from_utf8_lossy(&output.stderr).contains(reason),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(!root.join("output").exists());
    }
}

#[test]
fn capture_this_build_against_the_current_corpus() {
    let Ok(inputs_root) = std::env::var("XVER_INPUTS") else {
        return; // not a cross-version run
    };
    let out_root = PathBuf::from(std::env::var("XVER_OUT").expect("XVER_OUT"));
    let requested_label = std::env::var("XVER_LABEL").expect("XVER_LABEL");
    let provenance = common::dynamo_capture_provenance(Some(&requested_label));
    let label = provenance["label"].as_str().expect("capture label");

    let mut families: Vec<PathBuf> = std::fs::read_dir(Path::new(&inputs_root))
        .expect("inputs dir")
        .map(|e| e.expect("read inputs directory entry").path())
        .filter(|p| p.is_dir())
        .collect();
    families.sort();

    let mut total = 0usize;
    let selected_families = common::capture_families();
    if let Some(selected) = &selected_families {
        for family in selected {
            assert!(
                families
                    .iter()
                    .any(|path| path.file_name().is_some_and(|name| name == family.as_str())),
                "requested capture family has no input directory: {family}"
            );
        }
    }
    for fam_dir in families {
        let family = fam_dir.file_name().unwrap().to_string_lossy().to_string();
        if selected_families
            .as_ref()
            .is_some_and(|selected| !selected.contains(&family))
        {
            continue;
        }
        let mut cases: BTreeMap<String, serde_yaml::Value> = BTreeMap::new();
        let mut files: Vec<PathBuf> = std::fs::read_dir(&fam_dir)
            .expect("family dir")
            .map(|e| e.expect("read family directory entry").path())
            .filter(|p| p.extension().is_some_and(|x| x == "yaml"))
            .collect();
        files.sort();
        for fp in files {
            let doc: serde_yaml::Value =
                serde_yaml::from_str(&std::fs::read_to_string(&fp).expect("read")).expect("yaml");
            let case_map = doc
                .get("cases")
                .and_then(|c| c.as_mapping())
                .expect("input document requires a cases mapping");
            for (cid, cdoc) in case_map {
                let cid = cid
                    .as_str()
                    .filter(|s| !s.is_empty())
                    .expect("case ID must be a nonempty string")
                    .to_string();
                let input = cdoc
                    .get("input")
                    .and_then(|v| v.as_str())
                    .expect("case requires a string input");
                let init: common::Init = cdoc
                    .get("init")
                    .map(|value| serde_yaml::from_value(value.clone()).expect("case init"))
                    .unwrap_or_default();
                let finish_reason = cdoc
                    .get("finish_reason")
                    .map(|value| {
                        value
                            .as_str()
                            .expect("case requires a string finish_reason")
                    })
                    .unwrap_or("stop");
                let tool_schemas: Vec<dynamo_parsers_v2::Tool> = cdoc
                    .get("tools")
                    .map(|value| serde_yaml::from_value(value.clone()).expect("case tools schema"))
                    .unwrap_or_else(tools);
                let tool_schema_json = cdoc
                    .get("tools")
                    .map(|value| serde_json::to_value(value).expect("case tools schema JSON"));
                let tool_schema_json =
                    common::unified_tool_schemas_for_case(tool_schema_json.as_ref());
                let recorded = cdoc
                    .get("chunks")
                    .and_then(|value| requested_chunks(value, &family, &cid));
                let schedule = capture::replay_chunks(input, recorded.as_deref());
                let mut record = capture_record(
                    capture_case_with_chunks(&family, input, &schedule, &init, &tool_schemas),
                    &init,
                );
                let chunks: Vec<_> = schedule
                    .into_iter()
                    .chain(std::iter::once("‹finish›".to_string()))
                    .map(|text| serde_json::json!({"delta_text":text}))
                    .collect();
                let stimulus = serde_json::json!({
                    "input":input,
                    "init": init.applied(),
                    "finish_reason":finish_reason, "tools":tool_schema_json, "chunks":chunks,
                });
                record.as_mapping_mut().unwrap().insert(
                    "capture_input".into(),
                    common::json_to_yaml_value(&stimulus),
                );
                cases.insert(cid, record);
            }
        }
        let fam_out = out_root.join(&family);
        assert!(!cases.is_empty(), "capture family has no cases: {family}");
        std::fs::create_dir_all(&fam_out).expect("mkdir");
        for (cid, case) in &cases {
            let mut cw = serde_yaml::Mapping::new();
            cw.insert("dynamo_v2".into(), label.into());
            let mut one = serde_yaml::Mapping::new();
            one.insert(cid.clone().into(), case.clone());
            let mut doc = serde_yaml::Mapping::new();
            doc.insert("family".into(), family.clone().into());
            doc.insert("mode".into(), "unified".into());
            doc.insert("captured_with".into(), serde_yaml::Value::Mapping(cw));
            doc.insert(
                "capture_provenance".into(),
                common::json_to_yaml_value(&provenance),
            );
            doc.insert("cases".into(), serde_yaml::Value::Mapping(one));
            std::fs::write(
                fam_out.join(format!("{cid}.yaml")),
                serde_yaml::to_string(&serde_yaml::Value::Mapping(doc)).expect("emit"),
            )
            .expect("write");
            total += 1;
        }
        println!("[xver] {family}: {} cases", cases.len());
    }
    println!("[xver] wrote {total} case files to {}", out_root.display());
    assert!(total > 0, "captured nothing — check XVER_INPUTS layout");
    assert_eq!(
        common::dynamo_capture_provenance(Some(label)),
        provenance,
        "capture source changed during the run"
    );
}

#[test]
fn family_jobs_replay_terminal_tool_calls_without_accepting_chunk_metadata() {
    let document: Value = serde_yaml::from_str(
        &std::fs::read_to_string(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("fixtures-unified-v2/families/gemma4/inputs_and_golden.yaml"),
        )
        .unwrap(),
    )
    .unwrap();
    let case = document["cases"]
        .as_object()
        .unwrap()
        .values()
        .find(|case| case["scenario"] == "guided_json_bare_tool_header_recovers_inside_a_thought")
        .unwrap();
    let request = &case["request"];
    assert_eq!(request["finish_reason"], "tool_calls");
    let mut cases = Vec::new();
    for reason in ["stop", "length", "tool_calls", "unknown"] {
        let mut request = request.clone();
        request["finish_reason"] = json!(reason);
        cases.push(json!({"id":reason,"request":request}));
    }
    for field in ["token_ids", "finish_reason"] {
        let mut request = request.clone();
        request["chunks"][0][field] = if field == "token_ids" {
            json!([1])
        } else {
            json!("tool_calls")
        };
        cases.push(json!({"id":field,"request":request}));
    }
    let scratch = std::env::temp_dir().join(format!("dynamo-terminal-jobs-{}", std::process::id()));
    std::fs::create_dir_all(&scratch).unwrap();
    let jobs = scratch.join("jobs.json");
    let output = scratch.join("results.json");
    std::fs::write(
        &jobs,
        serde_json::to_vec(&json!([{"family":"gemma4","cases":cases}])).unwrap(),
    )
    .unwrap();
    let result = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "capture_family_jobs", "--nocapture"])
        .env("XVER_JOBS", jobs)
        .env("XVER_JSON_OUTPUT", &output)
        .env(
            "XVER_FAMILIES",
            Path::new(env!("CARGO_MANIFEST_DIR")).join("utils/src/parser_families.yaml"),
        )
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let results: Value = serde_json::from_slice(&std::fs::read(output).unwrap()).unwrap();
    for reason in ["stop", "length", "tool_calls"] {
        let record = &results["gemma4"][reason];
        assert!(record.get("unavailable").is_none(), "{reason}: {record}");
        let mut expected = case["golden"]["assembled"].clone();
        for event in expected.as_array_mut().unwrap() {
            if event["kind"] == "tool_call" && !event["arguments"].is_string() {
                event["arguments"] = json!(serde_json::to_string(&event["arguments"]).unwrap());
            }
        }
        assert_eq!(record["assembled"], expected);
        assert_eq!(record["chunks"], results["gemma4"]["stop"]["chunks"]);
        let mut expected_request = request.clone();
        expected_request["finish_reason"] = json!(reason);
        assert_eq!(record["capture_input"], expected_request);
    }
    for reason in ["unknown", "token_ids", "finish_reason"] {
        assert_eq!(
            results["gemma4"][reason]["capture_observation"]["unavailable"]["code"],
            "unsupported_request_mode"
        );
    }
}

#[test]
fn capture_family_jobs() {
    let Ok(job_path) = std::env::var("XVER_JOBS") else {
        return;
    };
    let jobs: Vec<Value> =
        serde_json::from_slice(&std::fs::read(job_path).expect("jobs")).expect("job JSON");
    let mut families = serde_json::Map::new();
    for job in jobs {
        let family = job["family"].as_str().expect("family");
        let mut results = serde_json::Map::new();
        for case in job["cases"].as_array().expect("cases") {
            let request = &case["request"];
            let input = request["input"].as_str().expect("input");
            let mut recorded = Vec::new();
            let mut supported = true;
            for row in request["chunks"].as_array().expect("chunks") {
                if row
                    .as_object()
                    .expect("chunk")
                    .keys()
                    .any(|key| key != "delta_text")
                {
                    supported = false;
                }
                recorded.push(capture::InputChunk {
                    delta_text: row["delta_text"].as_str().expect("chunk text").to_owned(),
                });
            }
            let chunks = capture::replay_chunks(input, Some(&recorded));
            let init: common::Init = serde_json::from_value(request["init"].clone()).expect("init");
            let schemas = common::unified_tools_for_schemas(Some(&request["tools"]));
            let result = if !supported
                || !matches!(
                    request["finish_reason"].as_str(),
                    Some("stop" | "length" | "tool_calls")
                ) {
                Err(CaptureFailure::Unavailable(
                    "this text adapter cannot replay token or per-chunk finish metadata".into(),
                ))
            } else {
                match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    capture_case_with_chunks(family, input, &chunks, &init, &schemas)
                })) {
                    Ok(result) => result,
                    Err(_) => Err(CaptureFailure::Error("registered parser panicked".into())),
                }
            };
            let mut record = serde_json::to_value(capture_record(result, &init)).expect("record");
            record["parser_path"] = json!(if native_family_registered(family) {
                "unified"
            } else {
                "split"
            });
            if let Some(detail) = record
                .get("unavailable")
                .and_then(Value::as_str)
                .map(str::to_owned)
            {
                record["capture_observation"] = json!({"unavailable": {
                    "code": if family_registered(family) { "unsupported_request_mode" } else { "unsupported_family" }, "detail": detail,
                }});
            }
            record["capture_input"] = request.clone();
            assert!(
                results
                    .insert(case["id"].as_str().expect("id").to_owned(), record)
                    .is_none()
            );
        }
        assert!(
            families
                .insert(family.to_owned(), Value::Object(results))
                .is_none()
        );
    }
    std::fs::write(
        std::env::var("XVER_JSON_OUTPUT").expect("output"),
        serde_json::to_vec(&families).expect("output JSON"),
    )
    .expect("write output");
}
