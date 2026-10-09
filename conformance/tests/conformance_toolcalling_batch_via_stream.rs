// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Live stream-on-batch output must match its pinned v2 capture exactly.
//! Differences from v1 additionally require a triaged parity note or an exact
//! baseline-defect observation. Neither kind changes the authored golden oracle.

use std::collections::{BTreeMap, BTreeSet};

mod common;
use common::{collect_yaml, fixture_name};

use dynamo_parsers_v2::{
    HarmonyToolStreamParser, Tool, ToolCallDelta, ToolParseResult, assemble_tool_calls,
    create_tool_parser_for_family,
};
use serde::Deserialize;
use serde_json::Value;

#[derive(Deserialize)]
struct Fixture {
    family: String,
    mode: String,
    #[serde(default)]
    cases: BTreeMap<String, Case>,
}

#[derive(Deserialize)]
struct Case {
    #[serde(default)]
    model_text: Option<String>,
    #[serde(default)]
    expected: Option<Expected>,
    #[serde(default)]
    tools: Vec<Tool>,
}

#[derive(Deserialize)]
struct Expected {
    dynamo_v1: EngineExpected,
}

#[derive(Deserialize)]
struct EngineExpected {
    #[serde(default)]
    calls: Vec<ExpCall>,
    #[serde(default)]
    normal_text: String,
}

#[derive(Deserialize)]
struct ExpCall {
    name: String,
    #[serde(default)]
    arguments: Value,
}

impl From<&EngineExpected> for EngineResult {
    fn from(expected: &EngineExpected) -> Self {
        Self {
            calls: expected
                .calls
                .iter()
                .map(|c| (c.name.clone(), c.arguments.clone()))
                .collect(),
            normal_text: expected.normal_text.clone(),
        }
    }
}

#[derive(Deserialize)]
struct StreamCapture {
    family: String,
    captured_with: BTreeMap<String, String>,
    cases: BTreeMap<String, StreamCaptureCase>,
}

#[derive(Deserialize)]
struct StreamCaptureCase {
    #[serde(default)]
    dynamo_v2: Option<Value>,
}

#[derive(Deserialize)]
struct PinnedExpected {
    calls: Vec<ExpCall>,
    normal_text: String,
}

fn pinned_result(block: &Value) -> Option<EngineResult> {
    let expected: PinnedExpected = serde_json::from_value(block.clone()).ok()?;
    Some(EngineResult {
        calls: expected
            .calls
            .into_iter()
            .map(|call| (call.name, call.arguments))
            .collect(),
        normal_text: expected.normal_text,
    })
}

#[derive(Debug)]
enum Exception {
    Parity,
    BaselineDefect(EngineResult),
}

type Notes = BTreeMap<String, BTreeMap<String, BTreeMap<String, String>>>;

fn exceptions(notes: &Notes) -> Result<BTreeMap<String, Exception>, String> {
    let mut result = BTreeMap::new();
    for (family, cases) in notes {
        for (cid, keys) in cases {
            let id = format!("{family}:{cid}");
            let parity = keys.get("stream_vs_batch");
            let defect = keys.get("baseline_defect");
            if parity.is_some() && defect.is_some() {
                return Err(format!(
                    "{id}: a baseline defect cannot also be an intended parity difference"
                ));
            }
            if let Some(note) = parity.or(defect)
                && note.trim().is_empty()
            {
                return Err(format!("{id}: empty divergence note"));
            }
            let exception = if let Some(note) = defect {
                if !note.starts_with("FIXME (v2 defect):") {
                    return Err(format!(
                        "{id}: baseline_defect must identify the v2 defect with a FIXME prefix"
                    ));
                }
                let actual = keys
                    .get("baseline_defect_actual")
                    .ok_or_else(|| format!("{id}: baseline defect lacks its exact observation"))?;
                let expected: Value = serde_json::from_str(actual)
                    .map_err(|e| format!("{id}: invalid baseline_defect_actual: {e}"))?;
                let result = pinned_result(&expected).ok_or_else(|| {
                    format!("{id}: baseline_defect_actual requires calls and normal_text")
                })?;
                Some(Exception::BaselineDefect(result))
            } else if parity.is_some() {
                Some(Exception::Parity)
            } else {
                None
            };
            if keys.contains_key("baseline_defect_actual") && defect.is_none() {
                return Err(format!("{id}: baseline_defect_actual has no defect note"));
            }
            if let Some(exception) = exception {
                result.insert(id, exception);
            }
        }
    }
    Ok(result)
}

fn check_result(
    got: &EngineResult,
    batch: &EngineResult,
    pinned: Option<&EngineResult>,
    exception: Option<&Exception>,
) -> Result<(), String> {
    let pinned = pinned.ok_or("missing pinned v2 batch-on-stream expectation")?;
    if got != pinned {
        return Err(format!(
            "live v2 output changed: got {got:?}, pinned {pinned:?}"
        ));
    }
    if let Some(Exception::BaselineDefect(actual)) = exception
        && got != actual
    {
        return Err(format!(
            "baseline defect changed: got {got:?}, observed {actual:?}"
        ));
    }
    if got == batch {
        if exception.is_some() {
            return Err("stale divergence entry: stream now agrees with batch".into());
        }
    } else if exception.is_none() {
        return Err(format!(
            "undocumented parity difference: stream {got:?}, batch {batch:?}"
        ));
    }
    Ok(())
}

fn reconcile(known: &BTreeMap<String, Exception>, observed: &BTreeSet<String>) -> Vec<String> {
    known
        .keys()
        .filter(|id| !observed.contains(*id))
        .map(|id| format!("{id}: unvisited divergence entry"))
        .collect()
}

fn capture_matches_version(capture: &StreamCapture, family: &str) -> Result<(), String> {
    let version = common::STREAM_DYNAMO_V2_CURRENT_CAPTURE
        .strip_prefix("dynamo_v2-")
        .unwrap();
    if capture.family != family
        || capture.captured_with.get("dynamo_v2").map(String::as_str) != Some(version)
    {
        return Err(format!(
            "pinned v2 capture must identify {family} at {version}"
        ));
    }
    Ok(())
}

#[test]
fn toolcalling_batch_via_stream_parity() {
    let fixture_root = common::ensure_fixtures().join("toolcalling");
    let batch_root = fixture_root.join("fixtures-batch-v1");
    let capture_root = fixture_root.join("fixtures-batch-on-stream-v1");
    let inputs_root = batch_root.join("inputs");
    let dyn_dirs = common::version_dirs_ascending(&batch_root, "dynamo_v1-");
    assert!(!dyn_dirs.is_empty(), "no v1 batch captures");
    let mut files = Vec::new();
    collect_yaml(&inputs_root, &mut files);
    files.sort();

    let kd_path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("toolcalling/known-divergences.yaml");
    let notes: Notes = serde_yaml::from_str(&std::fs::read_to_string(kd_path).unwrap()).unwrap();
    let known = exceptions(&notes).unwrap();
    let mut observed = BTreeSet::new();
    let mut failures = Vec::new();
    let mut total = 0usize;
    let mut consistent = 0usize;
    let mut documented = 0usize;
    let mut defects = 0usize;

    for path in &files {
        let mut fx: Fixture =
            serde_yaml::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        if fx.mode != "batch"
            || (fx.family != "harmony" && create_tool_parser_for_family(&fx.family, &[]).is_err())
        {
            continue;
        }
        let rel = path.strip_prefix(&inputs_root).unwrap();
        for dyn_dir in &dyn_dirs {
            if let Some(dfx) = std::fs::read_to_string(dyn_dir.join(rel))
                .ok()
                .and_then(|text| serde_yaml::from_str::<Fixture>(&text).ok())
            {
                for (cid, dcase) in dfx.cases {
                    if let (Some(case), Some(expected)) = (fx.cases.get_mut(&cid), dcase.expected) {
                        case.expected = Some(expected);
                    }
                }
            }
        }
        let capture_path = capture_root.join(rel);
        let capture = std::fs::read_to_string(&capture_path)
            .map_err(|e| e.to_string())
            .and_then(|text| {
                serde_yaml::from_str::<StreamCapture>(&text).map_err(|e| e.to_string())
            })
            .and_then(|capture| capture_matches_version(&capture, &fx.family).map(|()| capture));
        let capture = match capture {
            Ok(capture) => Some(capture),
            Err(e) => {
                // Placeholder files contain no executed v1 batch samples.
                if fx
                    .cases
                    .values()
                    .any(|case| case.model_text.is_some() && case.expected.is_some())
                {
                    failures.push(format!("{}: {e}", capture_path.display()));
                }
                None
            }
        };
        eprintln!("fixture {}", fixture_name(path));
        for (cid, case) in &fx.cases {
            let (Some(text), Some(expected)) = (&case.model_text, &case.expected) else {
                continue;
            };
            total += 1;
            let id = format!("{}:{cid}", fx.family);
            let got = match parse_stream_result(&fx.family, text, &case.tools) {
                Ok(result) => result,
                Err(e) => {
                    failures.push(format!("{id}: parser error: {e}"));
                    continue;
                }
            };
            let batch = EngineResult::from(&expected.dynamo_v1);
            let pinned = capture
                .as_ref()
                .and_then(|capture| capture.cases.get(cid))
                .and_then(|case| case.dynamo_v2.as_ref())
                .and_then(pinned_result);
            let exception = known.get(&id);
            if exception.is_some() {
                observed.insert(id.clone());
            }
            if got == batch {
                consistent += 1;
            }
            match exception {
                Some(Exception::Parity) => documented += 1,
                Some(Exception::BaselineDefect(_)) => {
                    defects += 1;
                    eprintln!(
                        "KNOWN BASELINE DEFECT {id}: {}",
                        notes[&fx.family][cid]["baseline_defect"]
                    );
                }
                None => {}
            }
            if let Err(e) = check_result(&got, &batch, pinned.as_ref(), exception) {
                failures.push(format!("{id}: {e}"));
            }
        }
    }
    failures.extend(reconcile(&known, &observed));
    eprintln!(
        "Dynamo stream-on-batch: {total} exact-capture checks; {consistent} agree with v1, {documented} documented differences, {defects} baseline defects"
    );
    assert!(total > 0, "no batch-on-stream cases executed");
    assert!(
        failures.is_empty(),
        "{} stream-on-batch gate failures:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[derive(Debug, PartialEq, Eq)]
struct EngineResult {
    calls: Vec<(String, Value)>,
    normal_text: String,
}

fn parse_stream_result(
    family: &str,
    text: &str,
    tools: &[Tool],
) -> Result<EngineResult, Box<dyn std::error::Error>> {
    if family == "harmony" {
        let mut parser = HarmonyToolStreamParser::new()?;
        let mut result = parser.parse_tool_call_streaming_text(text);
        let finish = parser.finish_tool_call_stream();
        result.normal_text.push_str(&finish.normal_text);
        result.tool_call_chunks.extend(finish.tool_call_chunks);
        return Ok(EngineResult {
            calls: assemble_tool_calls(&result.tool_call_chunks)
                .into_iter()
                .map(|(n, a)| {
                    let v = serde_json::from_str(&a).unwrap_or(Value::String(a));
                    (n, v)
                })
                .collect(),
            normal_text: result.normal_text,
        });
    }

    let mut parser = create_tool_parser_for_family(family, tools)?;
    let mut result = parser.push(text)?;
    result.append(parser.finish()?);
    Ok(EngineResult {
        normal_text: result.normal_text.clone(),
        calls: assemble_trait_calls(result),
    })
}

fn assemble_trait_calls(result: ToolParseResult) -> Vec<(String, Value)> {
    let mut names = BTreeMap::<usize, String>::new();
    let mut args = BTreeMap::<usize, String>::new();
    let mut complete = BTreeMap::<usize, bool>::new();
    for ToolCallDelta {
        tool_index,
        name,
        arguments,
        complete: call_complete,
    } in result.calls
    {
        *complete.entry(tool_index).or_default() |= call_complete;
        if let Some(name) = name {
            names.entry(tool_index).or_insert(name);
        }
        args.entry(tool_index).or_default().push_str(&arguments);
    }
    names
        .into_iter()
        .filter_map(|(idx, name)| {
            let raw = args.remove(&idx).unwrap_or_default();
            if complete.get(&idx) != Some(&true) {
                return None;
            }
            let value = serde_json::from_str(&raw).unwrap_or(Value::String(raw));
            Some((name, value))
        })
        .collect()
}

#[cfg(test)]
mod gate_tests {
    use super::*;
    fn result(value: Value, text: &str) -> EngineResult {
        EngineResult {
            calls: vec![("probe".into(), value)],
            normal_text: text.into(),
        }
    }

    #[test]
    fn documented_difference_still_rejects_changed_arguments_and_text() {
        let pinned = result(serde_json::json!({"value": 42}), "after");
        let batch = result(serde_json::json!({"value": "42"}), "");
        assert!(check_result(&pinned, &batch, Some(&pinned), Some(&Exception::Parity)).is_ok());
        for changed in [
            result(serde_json::json!({"value": "42"}), "after"),
            result(serde_json::json!({"value": 42}), "lost"),
        ] {
            assert!(
                check_result(&changed, &batch, Some(&pinned), Some(&Exception::Parity))
                    .unwrap_err()
                    .contains("live v2 output changed")
            );
        }
    }

    #[test]
    fn pinned_output_does_not_permit_unlisted_parity_difference() {
        let pinned = result(Value::from(42), "");
        let batch = result(Value::from("42"), "");
        assert!(
            check_result(&pinned, &batch, Some(&pinned), None)
                .unwrap_err()
                .contains("undocumented parity difference")
        );
        assert!(
            check_result(&pinned, &batch, None, Some(&Exception::Parity))
                .unwrap_err()
                .contains("missing pinned")
        );
        for partial in [
            serde_json::json!({}),
            serde_json::json!({"calls": []}),
            serde_json::json!({"normal_text": ""}),
        ] {
            assert!(
                pinned_result(&partial).is_none(),
                "partial capture became an empty expectation"
            );
        }
    }

    #[test]
    fn exact_baseline_defect_cannot_change_with_recaptured_expectations() {
        let actual = result(Value::from(1), "");
        let defect = Exception::BaselineDefect(result(Value::from(1), ""));
        let batch = EngineResult {
            calls: vec![],
            normal_text: String::new(),
        };
        assert!(check_result(&actual, &batch, Some(&actual), Some(&defect)).is_ok());
        let changed = result(Value::from(2), "");
        assert!(
            check_result(&changed, &batch, Some(&changed), Some(&defect))
                .unwrap_err()
                .contains("baseline defect changed")
        );
    }

    #[test]
    fn stale_and_unvisited_exceptions_fail() {
        let result = result(Value::from(42), "");
        assert!(
            check_result(&result, &result, Some(&result), Some(&Exception::Parity))
                .unwrap_err()
                .contains("stale")
        );
        let known = BTreeMap::from([("family:case".into(), Exception::Parity)]);
        assert_eq!(
            reconcile(&known, &BTreeSet::new()),
            ["family:case: unvisited divergence entry"]
        );
    }

    #[test]
    fn empty_unpinned_or_misclassified_notes_fail() {
        for keys in [
            BTreeMap::from([("stream_vs_batch".into(), " ".into())]),
            BTreeMap::from([(
                "baseline_defect".into(),
                "FIXME (v2 defect): empty name".into(),
            )]),
            BTreeMap::from([("baseline_defect_actual".into(), "{}".into())]),
            BTreeMap::from([
                ("stream_vs_batch".into(), "intended".into()),
                (
                    "baseline_defect".into(),
                    "FIXME (v2 defect): empty name".into(),
                ),
            ]),
        ] {
            assert!(
                exceptions(&BTreeMap::from([(
                    "family".into(),
                    BTreeMap::from([("case".into(), keys)])
                )]))
                .is_err()
            );
        }
    }

    #[test]
    fn capture_version_and_family_must_match() {
        let mut capture = StreamCapture {
            family: "glm47".into(),
            captured_with: BTreeMap::from([(
                "dynamo_v2".into(),
                common::STREAM_DYNAMO_V2_CURRENT_CAPTURE
                    .strip_prefix("dynamo_v2-")
                    .unwrap()
                    .into(),
            )]),
            cases: BTreeMap::new(),
        };
        assert!(capture_matches_version(&capture, "glm47").is_ok());
        assert!(capture_matches_version(&capture, "qwen3_coder").is_err());
        capture
            .captured_with
            .insert("dynamo_v2".into(), "0.7.19".into());
        assert!(capture_matches_version(&capture, "glm47").is_err());
    }
}
