// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use anyhow::{Context, Result, bail, ensure};
use dynamo_parsers_v2::{
    InvalidGuidedPayloadPolicy, Tool, UnifiedEvent, UnifiedParser, UnifiedParserEvent,
    UnifiedParserExt, UnifiedParserInit, UnifiedParserOutput, UnifiedParserStartingState,
    UnifiedToolOutputMode, assemble, assemble_with_tool_indices, create_unified_parser_for_family,
    tool_arguments_raw,
};
use serde::de::{DeserializeSeed, Error, IgnoredAny, MapAccess, Visitor};
use serde::{Deserialize, Deserializer};

use crate::arguments::{canonical_argument_json, captured_arguments};

#[derive(Debug, Clone)]
pub struct RunOptions {
    /// Run only these published case IDs. `None` selects every active case.
    pub case_ids: Option<BTreeSet<String>>,
    pub check_every_split: bool,
    pub check_stream_batch: bool,
    pub check_stream_isolation: bool,
}

impl Default for RunOptions {
    fn default() -> Self {
        Self {
            case_ids: None,
            check_every_split: true,
            check_stream_batch: true,
            check_stream_isolation: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CaseFailure {
    pub case_id: String,
    pub check: String,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CaseUnsupported {
    pub case_id: String,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunReport {
    pub family: String,
    pub selected: usize,
    pub executed: usize,
    pub passed: usize,
    pub unsupported: Vec<CaseUnsupported>,
    pub failures: Vec<CaseFailure>,
    pub isolation_pairs: usize,
    pub isolation_failures: Vec<CaseFailure>,
}

impl RunReport {
    /// Return a process status suitable for a CI wrapper.
    pub fn exit_code(&self, fail_on_unsupported: bool) -> i32 {
        if self.failures.is_empty()
            && self.isolation_failures.is_empty()
            && (!fail_on_unsupported || self.unsupported.is_empty())
        {
            0
        } else {
            1
        }
    }
}

#[derive(Deserialize)]
struct FamilyManifest {
    unified: BTreeMap<String, FamilyConfig>,
}

#[derive(Deserialize)]
struct FamilyConfig {
    #[serde(default)]
    native: bool,
}

#[derive(Deserialize)]
struct FamilyFixtures {
    cases: BTreeMap<String, CaseMetadata>,
}

#[derive(Deserialize)]
struct CaseMetadata {
    lifecycle: Option<String>,
    display_id: Option<String>,
}

#[derive(Deserialize)]
struct CaseFixture {
    request: RequestFixture,
    golden: GoldenFixture,
}

#[derive(Deserialize)]
struct RequestFixture {
    input: String,
    #[serde(default)]
    init: RequestInit,
    #[serde(default)]
    tools: Vec<Tool>,
    #[serde(default)]
    chunks: Vec<ChunkFixture>,
}

#[derive(Default, Deserialize)]
struct RequestInit {
    #[serde(default)]
    preserve_message_phases: bool,
    #[serde(default)]
    starting_state: String,
    #[serde(default)]
    tool_output_mode: String,
    #[serde(default)]
    named_tool: Option<String>,
    #[serde(default)]
    structural_tag: Option<serde_yaml::Value>,
}

#[derive(Deserialize)]
struct ChunkFixture {
    delta_text: String,
}

#[derive(Deserialize)]
struct GoldenFixture {
    assembled: Vec<UnifiedEvent>,
}

struct SelectedCases<'a>(&'a BTreeMap<String, String>);

impl<'de> DeserializeSeed<'de> for SelectedCases<'_> {
    type Value = BTreeMap<String, CaseFixture>;

    fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<Self::Value, D::Error> {
        deserializer.deserialize_map(self)
    }
}

impl<'de> Visitor<'de> for SelectedCases<'_> {
    type Value = BTreeMap<String, CaseFixture>;

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("authored cases")
    }

    fn visit_map<M: MapAccess<'de>>(self, mut map: M) -> Result<Self::Value, M::Error> {
        let mut cases = BTreeMap::new();
        while let Some(key) = map.next_key::<String>()? {
            if let Some(id) = self.0.get(&key) {
                let case = map.next_value::<CaseFixture>().map_err(|error| {
                    M::Error::custom(format!("parse active case {id}: {error}"))
                })?;
                cases.insert(key, case);
            } else {
                map.next_value::<IgnoredAny>()?;
            }
        }
        Ok(cases)
    }
}

struct SelectedFixtures<'a>(&'a BTreeMap<String, String>);

impl<'de> Visitor<'de> for SelectedFixtures<'_> {
    type Value = BTreeMap<String, CaseFixture>;

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("an authored family corpus")
    }

    fn visit_map<M: MapAccess<'de>>(self, mut map: M) -> Result<Self::Value, M::Error> {
        let mut cases = None;
        while let Some(key) = map.next_key::<String>()? {
            if key == "cases" {
                if cases.is_some() {
                    return Err(M::Error::duplicate_field("cases"));
                }
                cases = Some(map.next_value_seed(SelectedCases(self.0))?);
            } else {
                map.next_value::<IgnoredAny>()?;
            }
        }
        cases.ok_or_else(|| M::Error::missing_field("cases"))
    }
}

struct ActiveCase {
    id: String,
    input: String,
    chunks: Vec<String>,
    tools: Vec<Tool>,
    init: RequestInit,
    golden: Vec<UnifiedEvent>,
}

/// Run the active authored GOLDEN cases for one native family using the parser
/// factory currently registered under that family name.
///
/// A Rust parser can register itself with `dynamo_parsers_v2::register_unified_parser`
/// and call this function from its own test or executable. The runner reads the
/// repository's authored `inputs_and_golden.yaml` files; it does not add vendor
/// families to the built-in registry or execute recorded parser captures as an oracle.
pub fn run_unified_family(family: &str, options: &RunOptions) -> Result<RunReport> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let manifest: FamilyManifest =
        serde_yaml::from_str(include_str!("../utils/src/parser_families.yaml"))
            .context("parse parser_families.yaml")?;
    let config = manifest
        .unified
        .get(family)
        .with_context(|| format!("unknown Unified family `{family}`"))?;
    ensure!(
        config.native,
        "family `{family}` does not have a native Unified parser"
    );

    let fixture_path = root
        .join("fixtures-unified-v2/families")
        .join(family)
        .join("inputs_and_golden.yaml");
    let text = std::fs::read_to_string(&fixture_path)
        .with_context(|| format!("read authored corpus {}", fixture_path.display()))?;
    let active = load_active_cases(&text, options)
        .with_context(|| format!("parse authored corpus {}", fixture_path.display()))?;

    let mut report = RunReport {
        family: family.to_owned(),
        selected: active.len(),
        executed: 0,
        passed: 0,
        unsupported: Vec::new(),
        failures: Vec::new(),
        isolation_pairs: 0,
        isolation_failures: Vec::new(),
    };
    for case in &active {
        if let Some(reason) = unsupported_reason(case) {
            report.unsupported.push(CaseUnsupported {
                case_id: case.id.clone(),
                reason,
            });
            continue;
        }
        report.executed += 1;
        let before = report.failures.len();
        check_case(family, case, options, &mut report.failures);
        if report.failures.len() == before {
            report.passed += 1;
        }
    }
    if options.check_stream_isolation {
        check_isolation(family, &active, &mut report);
    }
    Ok(report)
}

fn load_active_cases(text: &str, options: &RunOptions) -> Result<Vec<ActiveCase>> {
    // Index only metadata first. A Value round trip would turn a plain `4.2e1`
    // chunk into a float, losing bytes before the parser ever sees its input.
    let fixtures: FamilyFixtures = serde_yaml::from_str(text)?;

    let mut selected_rows = BTreeMap::new();
    let mut known_active_ids = BTreeSet::new();
    for (key, metadata) in fixtures.cases {
        if metadata.lifecycle.as_deref() != Some("active") {
            continue;
        }
        let id = metadata
            .display_id
            .as_deref()
            .filter(|id| !id.is_empty())
            .unwrap_or(&key)
            .to_owned();
        known_active_ids.insert(id.clone());
        if options
            .case_ids
            .as_ref()
            .is_some_and(|selected| !selected.contains(&id))
        {
            continue;
        }
        selected_rows.insert(key, id);
    }
    ensure!(!selected_rows.is_empty(), "no active cases selected");
    if let Some(selected) = &options.case_ids {
        let missing: Vec<_> = selected.difference(&known_active_ids).cloned().collect();
        ensure!(
            missing.is_empty(),
            "unknown or inactive case IDs: {}",
            missing.join(", ")
        );
    }

    let cases = serde_yaml::Deserializer::from_str(text)
        .deserialize_map(SelectedFixtures(&selected_rows))?;
    let mut active = Vec::new();
    for (key, case) in cases {
        let id = selected_rows[&key].clone();
        let chunks = authored_chunks(&case.request.input, &case.request.chunks)
            .with_context(|| format!("{id}: invalid authored chunk schedule"))?;
        active.push(ActiveCase {
            id,
            input: case.request.input,
            chunks,
            tools: case.request.tools,
            init: case.request.init,
            golden: case.golden.assembled,
        });
    }
    Ok(active)
}

fn unsupported_reason(case: &ActiveCase) -> Option<String> {
    case.init.structural_tag.as_ref().map(|_| {
        "structural-tag generation constraints are not parser output and are not executed by this runner".to_owned()
    })
}

fn check_case(
    family: &str,
    case: &ActiveCase,
    options: &RunOptions,
    failures: &mut Vec<CaseFailure>,
) {
    let chunks: Vec<_> = case.chunks.iter().map(String::as_str).collect();
    let deltas = match run_stream(family, case, &chunks) {
        Ok(events) => events,
        Err(error) => {
            failures.push(failure(&case.id, "stream", format!("{error:#}")));
            return;
        }
    };
    let streamed = assemble_exact(&deltas);
    if !events_equal(&streamed, &case.golden) {
        failures.push(failure(
            &case.id,
            "golden",
            format!("expected {:?}, got {:?}", case.golden, streamed),
        ));
    }

    if options.check_every_split {
        let whole = match run_stream(family, case, &[case.input.as_str()]) {
            Ok(events) => assemble_exact(&events),
            Err(error) => {
                failures.push(failure(&case.id, "whole-input", format!("{error:#}")));
                return;
            }
        };
        for split in 0..=case.input.len() {
            if !case.input.is_char_boundary(split) {
                continue;
            }
            let chunks = [&case.input[..split], &case.input[split..]];
            match run_stream(family, case, &chunks) {
                Ok(events) if events_equal(&assemble_exact(&events), &whole) => {}
                Ok(events) => failures.push(failure(
                    &case.id,
                    "chunk-invariance",
                    format!(
                        "split at byte {split} produced {events:?}, whole input produced {whole:?}"
                    ),
                )),
                Err(error) => failures.push(failure(
                    &case.id,
                    "chunk-invariance",
                    format!("split at byte {split}: {error}"),
                )),
            }
        }
    }

    if options.check_stream_batch {
        match create_initialized_parser(family, case) {
            Ok(mut parser) => match parser.parse_complete(&case.input) {
                Ok(events) if events == assemble(&deltas) => {}
                Ok(events) => failures.push(failure(
                    &case.id,
                    "stream-batch",
                    format!("stream produced {streamed:?}, batch produced {events:?}"),
                )),
                Err(error) => failures.push(failure(
                    &case.id,
                    "stream-batch",
                    format!("batch parse failed: {error:#}"),
                )),
            },
            Err(error) => failures.push(failure(&case.id, "stream-batch", format!("{error:#}"))),
        }
    }
}

fn run_stream(family: &str, case: &ActiveCase, chunks: &[&str]) -> Result<Vec<UnifiedParserEvent>> {
    let mut parser = create_initialized_parser(family, case)?;
    let mut output = UnifiedParserOutput::default();
    for chunk in chunks {
        parser
            .parse_into(chunk, &mut output)
            .with_context(|| format!("parser rejected input chunk {chunk:?}"))?;
    }
    output.append(&mut parser.finish().context("parser finish failed")?);
    Ok(output.events)
}

fn assemble_exact(deltas: &[UnifiedParserEvent]) -> Vec<UnifiedEvent> {
    let raw = tool_arguments_raw(deltas);
    assemble_with_tool_indices(deltas)
        .into_iter()
        .map(|(index, mut event)| {
            if let Some(index) = index
                && let UnifiedEvent::ToolCall { arguments, .. } = &mut event
            {
                *arguments = serde_json::Value::String(captured_arguments(arguments, &raw[&index]));
            }
            event
        })
        .collect()
}

fn events_equal(left: &[UnifiedEvent], right: &[UnifiedEvent]) -> bool {
    let canonical = |arguments: &serde_json::Value| {
        let encoded = match arguments {
            serde_json::Value::String(raw)
                if serde_json::from_str::<Box<serde_json::value::RawValue>>(raw).is_ok() =>
            {
                raw.clone()
            }
            value => value.to_string(),
        };
        canonical_argument_json(encoded.trim())
    };
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(left, right)| match (left, right) {
                (
                    UnifiedEvent::ToolCall {
                        name: ln,
                        arguments: la,
                    },
                    UnifiedEvent::ToolCall {
                        name: rn,
                        arguments: ra,
                    },
                ) => ln == rn && canonical(la) == canonical(ra),
                _ => left == right,
            })
}

fn create_initialized_parser(family: &str, case: &ActiveCase) -> Result<Box<dyn UnifiedParser>> {
    let mut parser = create_unified_parser_for_family(family, &case.tools)
        .with_context(|| format!("create parser for `{family}`"))?;
    parser.preserve_message_phases(case.init.preserve_message_phases);
    parser
        .initialize_request(UnifiedParserInit {
            starting_state: starting_state(&case.init.starting_state)?,
            tool_output_mode: tool_output_mode(&case.init)?,
            invalid_guided_payload: InvalidGuidedPayloadPolicy::RecoverAsText,
            ..UnifiedParserInit::default()
        })
        .with_context(|| format!("initialize parser for `{}`", case.id))?;
    Ok(parser)
}

fn starting_state(value: &str) -> Result<UnifiedParserStartingState> {
    match value {
        "" | "None" => Ok(UnifiedParserStartingState::None),
        "Reasoning" => Ok(UnifiedParserStartingState::Reasoning),
        "Response" => Ok(UnifiedParserStartingState::Response),
        other => bail!("unsupported starting state `{other}`"),
    }
}

fn tool_output_mode(init: &RequestInit) -> Result<UnifiedToolOutputMode> {
    match init.tool_output_mode.as_str() {
        "" | "Native" => Ok(UnifiedToolOutputMode::Native),
        "GuidedJson" => Ok(UnifiedToolOutputMode::GuidedJson {
            named_tool: init.named_tool.clone(),
        }),
        other => bail!("unsupported tool output mode `{other}`"),
    }
}

fn authored_chunks(input: &str, rows: &[ChunkFixture]) -> Result<Vec<String>> {
    let mut chunks: Vec<_> = rows.iter().map(|row| row.delta_text.clone()).collect();
    if chunks.last().map(String::as_str) == Some("‹finish›")
        && chunks[..chunks.len() - 1].concat() == input
    {
        chunks.pop();
    }
    if chunks.is_empty() {
        chunks.push(input.to_owned());
    }
    ensure!(
        chunks.concat() == input,
        "chunks do not reproduce the authored input"
    );
    Ok(chunks)
}

fn check_isolation(family: &str, cases: &[ActiveCase], report: &mut RunReport) {
    for pair in cases.windows(2) {
        let [a, b] = pair else { continue };
        if unsupported_reason(a).is_some() || unsupported_reason(b).is_some() {
            continue;
        }
        report.isolation_pairs += 1;
        let chunks_a: Vec<_> = a.chunks.iter().map(String::as_str).collect();
        let chunks_b: Vec<_> = b.chunks.iter().map(String::as_str).collect();
        let solo_a = run_stream(family, a, &chunks_a);
        let solo_b = run_stream(family, b, &chunks_b);
        let mut pa = match create_initialized_parser(family, a) {
            Ok(parser) => parser,
            Err(error) => {
                report.isolation_failures.push(failure(
                    &a.id,
                    "stream-isolation",
                    format!("{error:#}"),
                ));
                continue;
            }
        };
        let mut pb = match create_initialized_parser(family, b) {
            Ok(parser) => parser,
            Err(error) => {
                report.isolation_failures.push(failure(
                    &b.id,
                    "stream-isolation",
                    format!("{error:#}"),
                ));
                continue;
            }
        };
        let (mut out_a, mut out_b) = (
            UnifiedParserOutput::default(),
            UnifiedParserOutput::default(),
        );
        let mut failed = false;
        for index in 0..chunks_a.len().max(chunks_b.len()) {
            if let Some(chunk) = chunks_a.get(index)
                && let Err(error) = pa.parse_into(chunk, &mut out_a)
            {
                report.isolation_failures.push(failure(
                    &a.id,
                    "stream-isolation",
                    format!("{error:#}"),
                ));
                failed = true;
                break;
            }
            if let Some(chunk) = chunks_b.get(index)
                && let Err(error) = pb.parse_into(chunk, &mut out_b)
            {
                report.isolation_failures.push(failure(
                    &b.id,
                    "stream-isolation",
                    format!("{error:#}"),
                ));
                failed = true;
                break;
            }
        }
        if failed {
            continue;
        }
        match pa.finish() {
            Ok(mut tail) => out_a.append(&mut tail),
            Err(error) => {
                report.isolation_failures.push(failure(
                    &a.id,
                    "stream-isolation",
                    format!("finish failed: {error:#}"),
                ));
                continue;
            }
        }
        match pb.finish() {
            Ok(mut tail) => out_b.append(&mut tail),
            Err(error) => {
                report.isolation_failures.push(failure(
                    &b.id,
                    "stream-isolation",
                    format!("finish failed: {error:#}"),
                ));
                continue;
            }
        }
        match solo_a {
            Ok(expected)
                if !events_equal(&assemble_exact(&out_a.events), &assemble_exact(&expected)) =>
            {
                report.isolation_failures.push(failure(
                    &a.id,
                    "stream-isolation",
                    format!("interleaving with {} changed output", b.id),
                ));
            }
            Err(error) => report.isolation_failures.push(failure(
                &a.id,
                "stream-isolation",
                format!("standalone output failed: {error:#}"),
            )),
            _ => {}
        }
        match solo_b {
            Ok(expected)
                if !events_equal(&assemble_exact(&out_b.events), &assemble_exact(&expected)) =>
            {
                report.isolation_failures.push(failure(
                    &b.id,
                    "stream-isolation",
                    format!("interleaving with {} changed output", a.id),
                ));
            }
            Err(error) => report.isolation_failures.push(failure(
                &b.id,
                "stream-isolation",
                format!("standalone output failed: {error:#}"),
            )),
            _ => {}
        }
    }
}

fn failure(case_id: &str, check: &str, message: impl Into<String>) -> CaseFailure {
    CaseFailure {
        case_id: case_id.to_owned(),
        check: check.to_owned(),
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_parsers_v2::{builtin_unified_families, canonical_unified_family};

    #[test]
    fn exact_argument_comparison_rejects_rounding_and_type_changes() {
        let call = |arguments| UnifiedEvent::ToolCall {
            name: "f".to_owned(),
            arguments,
        };
        assert!(events_equal(
            &[call(serde_json::json!({"n": 42}))],
            &[call(serde_json::Value::String(r#"{"n":4.2e1}"#.to_owned()))],
        ));
        for (left, right) in [
            (r#"{"n":9007199254740992.5}"#, r#"{"n":9007199254740992}"#),
            (r#"{"n":1e-400}"#, r#"{"n":0}"#),
            (r#"{"n":1e400}"#, r#"{"n":"1e400"}"#),
            (r#"{"n":42}"#, r#"{"n":"42"}"#),
        ] {
            assert!(
                !events_equal(
                    &[call(serde_json::Value::String(left.to_owned()))],
                    &[call(serde_json::Value::String(right.to_owned()))],
                ),
                "{left} versus {right}"
            );
        }
    }

    #[test]
    fn selected_text_keeps_scalar_spelling_and_skips_unselected_payloads() -> Result<()> {
        for scalar in [
            "4.2e1",
            "42.0",
            "-0",
            "001",
            "true",
            "null",
            "2026-11-03",
            "1e400",
        ] {
            let text = format!(
                "cases:
  retired:
    lifecycle: retired
    request: null
    golden: null
  unselected:
    lifecycle: active
    display_id: ignored
    request: {{chunks: [{{delta_text: [not, text]}}]}}
  selected:
    lifecycle: active
    display_id: selected
    request:
      input: '{scalar}'
      chunks: [{{delta_text: {scalar}}}]
    golden: {{assembled: []}}
"
            );
            let options = RunOptions {
                case_ids: Some(BTreeSet::from(["selected".to_owned()])),
                ..RunOptions::default()
            };
            let cases = load_active_cases(&text, &options)?;
            assert_eq!(cases.len(), 1);
            assert_eq!(cases[0].input, scalar);
            assert_eq!(cases[0].chunks, [scalar]);
            assert!(load_active_cases(&text, &RunOptions::default()).is_err());
        }
        Ok(())
    }

    #[test]
    fn every_builtin_family_loads_its_complete_active_corpus() -> Result<()> {
        let families: BTreeSet<_> = builtin_unified_families()
            .iter()
            .map(|family| canonical_unified_family(family).unwrap())
            .collect();
        for family in families {
            let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("fixtures-unified-v2/families")
                .join(family)
                .join("inputs_and_golden.yaml");
            let text = std::fs::read_to_string(path)?;
            let index: FamilyFixtures = serde_yaml::from_str(&text)?;
            let expected = index
                .cases
                .values()
                .filter(|case| case.lifecycle.as_deref() == Some("active"))
                .count();
            let cases = load_active_cases(&text, &RunOptions::default())
                .with_context(|| format!("family {family}"))?;
            assert_eq!(cases.len(), expected, "{family}");
            assert!(expected > 0, "{family}");
        }
        Ok(())
    }
}
