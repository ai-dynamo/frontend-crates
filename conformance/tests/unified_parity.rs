// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Current native conformance captures are compared with independently authored
//! GOLDEN as warnings. Recorded latest YAML agreement is enforced by
//! `unified_render`; request, lifecycle, and chunk invariants remain blocking.
//!
//! It asserts the invariants from `conformance/utils/lib/parsers/UNIFIED_CASES.md`
//! that are checkable from the single-stream corpus:
//!
//! * `I2` order preservation — the assembled list is compared order-sensitively
//! * `I5` chunk-invariance — the same input under five chunk splittings
//! * `I6` stream/batch parity — `parse_complete` against the streamed result
//! * `I4` per-stream isolation — two concurrent parsers do not see each other

mod common;
use common::unified_capture::{self as capture, chunk_input as chunk_markers};

use common::{
    Init,
    known_unified_divergences::{self as divergences, Check, Expected},
};

use std::collections::BTreeMap;

use dynamo_parsers_v2::{
    REGISTERED_UNIFIED_FAMILIES, UnifiedEvent, UnifiedParserExt, assemble,
    create_unified_parser_for_family,
};
use serde::Deserialize;

#[test]
fn prepared_golden_rejects_partial_snapshots_and_yields_to_alternate_sources() {
    assert!(common::prepared_unified_golden(None, false).is_none());
    let root = std::env::temp_dir().join(format!(
        "prepared-golden-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    assert!(common::prepared_unified_golden(Some(root.clone()), true).is_none());
    assert!(
        std::panic::catch_unwind(|| common::prepared_unified_golden(Some(root.clone()), false))
            .is_err()
    );
    std::fs::create_dir_all(&root).unwrap();
    assert!(
        std::panic::catch_unwind(|| common::prepared_unified_golden(Some(root.clone()), false))
            .is_err()
    );
    for family in common::unified_families().values() {
        std::fs::write(root.join(&family.golden_spec), "prepared fixture").unwrap();
    }
    assert_eq!(
        common::prepared_unified_golden(Some(root.clone()), false),
        Some(root.clone())
    );
    assert!(common::prepared_unified_golden(Some(root.clone()), true).is_none());
    let file = root.join(
        &common::unified_families()
            .values()
            .next()
            .unwrap()
            .golden_spec,
    );
    std::fs::remove_file(&file).unwrap();
    assert!(
        std::panic::catch_unwind(|| common::prepared_unified_golden(Some(root.clone()), false))
            .is_err()
    );
    std::fs::write(&file, "prepared fixture").unwrap();
    assert!(
        std::panic::catch_unwind(|| common::prepared_unified_golden(Some(file.clone()), false))
            .is_err()
    );
    std::fs::remove_dir_all(root).unwrap();
}

#[derive(Deserialize)]
struct GoldenFile {
    #[serde(default)]
    family: String,
    cases: BTreeMap<String, GoldenCase>,
}

#[derive(Deserialize)]
struct GoldenCase {
    input: String,
    #[serde(default)]
    input_chunks: Option<Vec<String>>,
    #[serde(default)]
    tools: Option<serde_json::Value>,
    golden: Vec<UnifiedEvent>,
    #[serde(skip)]
    raw_golden_arguments: Vec<String>,
    /// Request-scoped parser configuration, declared by the case. Shared with
    /// `unified_render` via `common::Init` so both harnesses configure a case
    /// identically; see that type for why it is declared and not inferred.
    #[serde(default)]
    init: Init,
}

fn load_native_golden() -> Vec<GoldenFile> {
    let dir = common::ensure_unified_golden();
    let mut files: Vec<GoldenFile> = Vec::new();
    for entry in std::fs::read_dir(&dir).unwrap_or_else(|e| panic!("read {}: {e}", dir.display())) {
        let path = entry.unwrap().path();
        if path.extension().and_then(|e| e.to_str()) != Some("yaml") {
            continue;
        }
        let text = std::fs::read_to_string(&path).unwrap();
        let raw: serde_yaml::Value =
            serde_yaml::from_str(&text).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let mut file: GoldenFile = serde_yaml::from_value(raw.clone())
            .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        if let Some(cases) = raw.get("cases").and_then(serde_yaml::Value::as_mapping) {
            for (id, value) in cases {
                let Some(id) = id.as_str() else { continue };
                let Some(case) = file.cases.get_mut(id) else {
                    continue;
                };
                case.raw_golden_arguments = value
                    .get("golden")
                    .and_then(serde_yaml::Value::as_sequence)
                    .into_iter()
                    .flatten()
                    .filter(|event| {
                        event.get("kind").and_then(serde_yaml::Value::as_str) == Some("tool_call")
                    })
                    .filter_map(|event| event.get("arguments").and_then(serde_yaml::Value::as_str))
                    .map(str::to_owned)
                    .collect();
            }
        }
        files.push(file);
    }
    let families = common::unified_families();
    files.retain(|file| families.get(&file.family).is_some_and(|row| row.native));
    files.sort_by(|a, b| a.family.cmp(&b.family));
    for file in &mut files {
        for case in file.cases.values_mut() {
            for event in &mut case.golden {
                if let UnifiedEvent::ToolCall { arguments, .. } = event {
                    *arguments = common::decoded_golden_arguments(arguments);
                }
            }
        }
    }
    files
}

#[test]
fn structural_response_case_checks_the_schema_before_parsing() {
    let mut families = Vec::new();
    for file in load_native_golden() {
        for (id, mut case) in file.cases {
            if !id.contains("structured_response_after_reasoning") {
                continue;
            }
            families.push(file.family.clone());
            let tools = common::unified_tools_for_schemas(case.tools.as_ref());
            let chunks = capture::input_chunks(&case.input, case.input_chunks.as_deref());
            assert!(
                events(&file.family, &chunks, &case.init, &tools).is_ok(),
                "{id}"
            );
            if file.family == "gpt_oss" {
                for invalid in [
                    "<|return|>".to_string(),
                    format!("{}extra", case.input),
                    format!("{}<|return|>", case.input),
                    case.input.replace("<|return|>", "<|call|>"),
                ] {
                    assert!(
                        events(
                            &file.family,
                            std::slice::from_ref(&invalid),
                            &case.init,
                            &tools,
                        )
                        .is_err(),
                        "{id}: accepted invalid stop sequence {invalid:?}"
                    );
                }
            }
            // Parsing the same stream still succeeds with a contradictory schema.
            // Only a real grammar check can reject this request before parsing.
            case.init.structural_tag.as_mut().unwrap().schema =
                serde_json::json!({"const": "impossible response"});
            assert!(
                events(&file.family, &chunks, &case.init, &tools).is_err(),
                "{id}: ignored the response schema"
            );
        }
    }
    families.sort();
    assert_eq!(
        families,
        [
            "beam",
            "deepseek_v4",
            "deepseek_v41",
            "glm47",
            "gpt_oss",
            "kimi_k2",
            "kimi_k3",
            "minimax_m3",
            "muse_spark",
            "qwen3"
        ]
    );
}

fn events(
    family: &str,
    chunks: &[String],
    init: &Init,
    tool_schemas: &[dynamo_parsers_v2::Tool],
) -> Result<Vec<UnifiedEvent>, String> {
    events_and_raw(family, chunks, init, tool_schemas).map(|(events, _)| events)
}

fn events_and_raw(
    family: &str,
    chunks: &[String],
    init: &Init,
    tool_schemas: &[dynamo_parsers_v2::Tool],
) -> Result<(Vec<UnifiedEvent>, std::collections::BTreeMap<usize, String>), String> {
    let mut parser = create_unified_parser_for_family(family, tool_schemas)
        .unwrap_or_else(|e| panic!("create unified parser for `{family}`: {e}"));
    init.apply(&mut parser, family);
    init.validate_capture(family, tool_schemas, chunks)
        .map_err(|error| format!("{error:#}"))?;

    let rows = capture::native_capture(&mut parser, chunks).map_err(|failure| match failure {
        capture::CaptureFailure::Error(error) => error,
        other => panic!("unexpected capture failure: {other:?}"),
    })?;
    let deltas = rows.into_iter().flatten().collect::<Vec<_>>();
    let raw_arguments = dynamo_parsers_v2::tool_arguments_raw(&deltas);
    Ok((assemble(&deltas), raw_arguments))
}

fn render(result: &Result<Vec<UnifiedEvent>, String>) -> String {
    let events = match result {
        Ok(events) => events,
        Err(error) => return format!("ERROR: {error}"),
    };
    events
        .iter()
        .map(|e| match e {
            UnifiedEvent::MessageStart { phase } => format!("message_start({phase:?})"),
            UnifiedEvent::MessageEnd => "message_end".into(),
            UnifiedEvent::Reasoning { text } => format!("reasoning({text:?})"),
            UnifiedEvent::Text { text } => format!("text({text:?})"),
            UnifiedEvent::ToolCall { name, arguments } => format!("tool_call({name}, {arguments})"),
        })
        .collect::<Vec<_>>()
        .join("  |  ")
}

/// Split into chunks of at most `n` chars (never mid-char).
fn chunk_every(input: &str, n: usize) -> Vec<String> {
    input
        .chars()
        .collect::<Vec<_>>()
        .chunks(n)
        .map(|c| c.iter().collect())
        .collect()
}

fn splittings(input: &str) -> Vec<(String, Vec<String>)> {
    vec![
        ("marker-aligned".into(), chunk_markers(input)),
        ("whole".into(), vec![input.to_string()]),
        ("1-char".into(), chunk_every(input, 1)),
        ("3-char".into(), chunk_every(input, 3)),
        ("7-char".into(), chunk_every(input, 7)),
    ]
}

fn extended_splittings(input: &str) -> Vec<(String, Vec<String>)> {
    const MAX_TWO_PART: usize = 24;
    const RANDOM_SPLITS: u64 = 4;
    let bounds: Vec<usize> = (1..input.len())
        .filter(|offset| input.is_char_boundary(*offset))
        .collect();
    let mut out = Vec::new();
    for &b in bounds
        .iter()
        .step_by(bounds.len().div_ceil(MAX_TWO_PART).max(1))
    {
        out.push((
            format!("split@{b}"),
            vec![input[..b].into(), input[b..].into()],
        ));
    }
    let markers = chunk_markers(input);
    for k in [2, 4, 8] {
        let grouped = markers.chunks(k).map(|c| c.concat()).collect();
        out.push((format!("marker-aligned-x{k}"), grouped));
    }
    if bounds.len() >= 2 {
        let mut state = input.len() as u64 ^ 0x9e37_79b9_7f4a_7c15;
        let mut next = |n: usize| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 33) as usize % n
        };
        for seed in 0..RANDOM_SPLITS {
            let mut cuts: Vec<usize> = (0..2 + next(5))
                .map(|_| bounds[next(bounds.len())])
                .collect();
            cuts.sort_unstable();
            cuts.dedup();
            let mut chunks = Vec::new();
            let mut start = 0;
            for end in cuts.into_iter().chain(std::iter::once(input.len())) {
                chunks.push(input[start..end].to_string());
                start = end;
            }
            out.push((format!("random-{seed}"), chunks));
        }
    }
    out
}

/// Quote-looking strings must survive every possible two-chunk boundary,
/// including boundaries inside their native escaping and framing.
#[test]
fn string_quote_controls_survive_every_split_for_every_family() {
    let mut covered = std::collections::BTreeSet::new();
    for file in load_native_golden().into_iter().filter(|file| {
        dynamo_parsers_v2::canonical_unified_family(&file.family) == Some(file.family.as_str())
    }) {
        let mut count = 0;
        for (id, case) in file.cases {
            if !id.contains("xml_string_type_") {
                continue;
            }
            count += 1;
            let tools = common::unified_tools_for_schemas(case.tools.as_ref());
            let expected = if id.contains("single_quoted") {
                "{'a':1}"
            } else {
                r#"{"a":1}"#
            };
            assert_eq!(
                case.golden,
                vec![UnifiedEvent::ToolCall {
                    name: "schema_probe".into(),
                    arguments: serde_json::json!({"value": expected}),
                }],
                "{id}: oracle must preserve the exact string"
            );
            for (schedule, chunks) in splittings(&case.input) {
                assert_eq!(
                    events(&file.family, &chunks, &case.init, &tools).unwrap(),
                    case.golden,
                    "{id}: {schedule}"
                );
            }
            for split in
                (0..=case.input.len()).filter(|offset| case.input.is_char_boundary(*offset))
            {
                let chunks = vec![case.input[..split].into(), case.input[split..].into()];
                assert_eq!(
                    events(&file.family, &chunks, &case.init, &tools).unwrap(),
                    case.golden,
                    "{id}: split {split}"
                );
            }
        }
        assert_eq!(count, 2, "{}: missing quote control", file.family);
        covered.insert(file.family);
    }
    assert_eq!(
        covered,
        REGISTERED_UNIFIED_FAMILIES
            .iter()
            .map(|family| dynamo_parsers_v2::canonical_unified_family(family)
                .unwrap()
                .to_string())
            .collect()
    );
}

/// Report assembled event disagreements without treating observations as the oracle.
#[test]
fn unified_parser_matches_the_golden_oracle() {
    let files = load_native_golden();
    assert!(
        !files.is_empty(),
        "no golden family has a unified parser — the surface is unproven"
    );

    let mut failures = Vec::new();
    let mut checked = 0usize;
    for file in &files {
        for (id, case) in &file.cases {
            checked += 1;
            let case_tools = common::unified_tools_for_schemas(case.tools.as_ref());
            let (got, raw_arguments) = match events_and_raw(
                &file.family,
                &capture::input_chunks(&case.input, case.input_chunks.as_deref()),
                &case.init,
                &case_tools,
            ) {
                Ok(result) => (Ok(result.0), Some(result.1)),
                Err(error) => (Err(error), None),
            };
            if id.starts_with("UNIFIED.arg_numeric_")
                && let Some(actual) = raw_arguments
                    .as_ref()
                    .filter(|_| !case.raw_golden_arguments.is_empty())
            {
                let expected = case
                    .raw_golden_arguments
                    .iter()
                    .enumerate()
                    .map(|(index, arguments)| (index, arguments.clone()))
                    .collect::<std::collections::BTreeMap<_, _>>();
                if actual.len() == expected.len()
                    && actual.iter().all(|(index, raw)| {
                        serde_json::value::RawValue::from_string(raw.clone()).is_ok()
                            && expected.get(index).is_some_and(|expected| {
                                common::canonical_argument_json(raw.trim())
                                    == common::canonical_argument_json(expected.trim())
                            })
                    })
                {
                    continue;
                }
            }
            if !got
                .as_ref()
                .is_ok_and(|events| common::json_values_equal(events, &case.golden))
            {
                failures.push(format!(
                    "{id}\n     input: {:?}\n    golden: {}\n   unified: {}",
                    case.input,
                    render(&Ok(case.golden.clone())),
                    render(&got),
                ));
            }
        }
    }

    capture::report_comparison("unified-golden", checked, &failures, false);
    assert!(checked >= 30, "expected the qwen3 corpus, got {checked}");
}

#[test]
fn golden_only_change_warns_through_real_corpus_execution() {
    let root = std::env::temp_dir().join(format!("golden-warning-{}", std::process::id()));
    let relative = "conformance/fixtures-unified-v2/families";
    let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    for family in std::fs::read_dir(manifest.join("fixtures-unified-v2/families")).unwrap() {
        let family = family.unwrap();
        let source = family.path().join("inputs_and_golden.yaml");
        if !source.is_file() {
            continue;
        }
        let destination = root.join(relative).join(family.file_name());
        std::fs::create_dir_all(&destination).unwrap();
        std::fs::copy(source, destination.join("inputs_and_golden.yaml")).unwrap();
    }
    let path = root
        .join(relative)
        .join("deepseek_v41/inputs_and_golden.yaml");
    // Preserve request scalar spellings and change only the first authored oracle.
    let source = std::fs::read_to_string(&path).unwrap();
    assert!(source.contains("          city: Paris\n"));
    std::fs::write(
        &path,
        source.replacen(
            "          city: Paris\n",
            "          city: capture-warning-regression\n",
            1,
        ),
    )
    .unwrap();
    let reports = root.join("reports");
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "unified_parser_matches_the_golden_oracle",
            "--nocapture",
        ])
        .env("FRONTEND_CRATES_ROOT", &root)
        .env("CONFORMANCE_CAPTURE_REPORT_DIR", &reports)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: serde_json::Value =
        serde_json::from_slice(&std::fs::read(reports.join("unified-golden.json")).unwrap())
            .unwrap();
    assert!(report["checked"].as_u64().unwrap() > 0);
    assert_eq!(report["blocking"], false);
    assert!(
        report["differences"]
            .as_array()
            .unwrap()
            .iter()
            .any(|difference| difference
                .as_str()
                .unwrap()
                .contains("capture-warning-regression"))
    );
}

/// I5: the assembled list must not depend on where chunk boundaries fall.
#[test]
fn unified_parser_is_chunk_invariant() {
    let known = divergences::load();
    let mut observed = divergences::ObservedChunkDivergences::new();
    let mut failures = Vec::new();
    for file in load_native_golden() {
        for (id, case) in &file.cases {
            let case_tools = common::unified_tools_for_schemas(case.tools.as_ref());
            let baseline = events(
                &file.family,
                std::slice::from_ref(&case.input),
                &case.init,
                &case_tools,
            );
            let baseline_rendered = render(&baseline);
            let expected = divergences::expected(&known, &file.family, id, Check::ChunkInvariance);
            let schedules = splittings(&case.input)
                .into_iter()
                .chain(extended_splittings(&case.input));
            for (label, chunks) in schedules {
                let got = events(&file.family, &chunks, &case.init, &case_tools);
                if let Some(Expected::ChunkInvariance(expected)) = &expected {
                    let seen = observed
                        .entry((file.family.clone(), id.clone()))
                        .or_default();
                    if let Err(detail) = expected.observe(&baseline_rendered, &render(&got), seen) {
                        failures.push(format!("{id} [{label}, {} chunks]: {detail}", chunks.len()));
                    }
                } else if got != baseline {
                    failures.push(format!(
                        "{id} [{label}, {} chunks]\n  whole: {}\n    got: {}",
                        chunks.len(),
                        baseline_rendered,
                        render(&got),
                    ));
                }
            }
        }
    }
    failures.extend(divergences::reconcile_chunk_invariance(&known, &observed));
    assert!(
        failures.is_empty(),
        "chunk splitting changed the assembled events ({}):\n\n{}",
        failures.len(),
        failures.join("\n\n"),
    );
}

#[test]
fn reference_argument_and_dotted_name_cases_preserve_results_at_every_split() {
    let reference_scenarios = [
        "glm_ref_object",
        "glm_ref_encoded_targets",
        "glm_ref_json_looking_strings",
        "glm_ref_scalar_types",
        "unused_reference_graph_parameter_types",
        "nullable_reference_alias_literals",
        "local_schema_id_preserves_type",
    ];
    let template_reference_scenarios = [
        "arg_json_null_ref",
        "arg_string_null_ref",
        "arg_null_mixed_labels",
    ];
    let dotted_scenarios = [
        "dotted_name_minimal",
        "dotted_name_mixed_punctuation",
        "dotted_name_multiple_segments",
        "dotted_name_shared_prefix",
    ];
    let dotted_native_families = [
        "beam",
        "deepseek_v4",
        "deepseek_v41",
        "gemma4",
        "glm47",
        "gpt_oss",
        "kimi_k2",
        "kimi_k3",
        "minimax_m3",
        "muse_glimmer",
        "muse_spark",
        "nemotron_v3",
        "qwen3",
    ];
    let mut checked = 0;
    let files = load_native_golden();
    for file in files {
        for (id, case) in &file.cases {
            if !reference_scenarios
                .iter()
                .chain(template_reference_scenarios.iter())
                .chain(dotted_scenarios.iter())
                .any(|scenario| id == &format!("UNIFIED.{scenario}.{}", file.family))
            {
                continue;
            }
            let tools = common::unified_tools_for_schemas(case.tools.as_ref());
            let whole = events(
                &file.family,
                std::slice::from_ref(&case.input),
                &case.init,
                &tools,
            );
            assert!(whole.is_ok(), "{id}: {}", render(&whole));
            if id.contains(".dotted_name_") {
                let characters = chunk_every(&case.input, 1);
                assert_eq!(
                    events(&file.family, &characters, &case.init, &tools),
                    whole,
                    "{id}: characters"
                );
            }
            for split in case
                .input
                .char_indices()
                .map(|(offset, _)| offset)
                .chain(std::iter::once(case.input.len()))
            {
                let chunks = vec![case.input[..split].into(), case.input[split..].into()];
                let divided = events(&file.family, &chunks, &case.init, &tools);
                assert_eq!(divided, whole, "{id} at byte {split}");
            }
            checked += 1;
        }
    }
    let native_families = common::unified_families()
        .values()
        .filter(|family| family.native)
        .count();
    let template_families = common::unified_families()
        .values()
        .filter(|family| family.native && family.scenario_profile.as_deref() != Some("native"))
        .count();
    assert_eq!(
        checked,
        native_families * reference_scenarios.len()
            + template_families * template_reference_scenarios.len()
            + dotted_native_families.len() * dotted_scenarios.len(),
    );
}

/// I6: parsing the whole output at once assembles to the streamed result.
#[test]
fn unified_parser_has_stream_batch_parity() {
    let known = divergences::load();
    let mut observed = std::collections::BTreeSet::new();
    let mut failures = Vec::new();
    for file in load_native_golden() {
        for (id, case) in &file.cases {
            let case_tools = common::unified_tools_for_schemas(case.tools.as_ref());
            let streamed = events(
                &file.family,
                &capture::input_chunks(&case.input, case.input_chunks.as_deref()),
                &case.init,
                &case_tools,
            );
            let mut parser = create_unified_parser_for_family(&file.family, &case_tools).unwrap();
            case.init.apply(&mut parser, id);

            let batch = parser
                .parse_complete(&case.input)
                .map_err(|e| format!("native push: {e:#}"));
            if batch != streamed {
                match divergences::expected(&known, &file.family, id, Check::StreamBatch) {
                    Some(Expected::StreamBatch(expected))
                        if render(&batch) == expected.batch
                            && render(&streamed) == expected.stream =>
                    {
                        observed.insert((file.family.clone(), id.clone()));
                    }
                    Some(Expected::StreamBatch(expected)) => failures.push(format!(
                        "{id}: known stream/batch divergence changed\n expected batch: {}\n expected stream: {}\n actual batch: {}\n actual stream: {}",
                        expected.batch,
                        expected.stream,
                        render(&batch),
                        render(&streamed),
                    )),
                    _ => failures.push(format!(
                        "{id}: batch and stream disagree\n   batch: {}\n  stream: {}",
                        render(&batch),
                        render(&streamed),
                    )),
                }
            }
        }
    }
    failures.extend(divergences::reconcile(
        &known,
        Check::StreamBatch,
        &observed,
    ));
    assert!(
        failures.is_empty(),
        "{} stream/batch discrepancies remain:\n\n{}",
        failures.len(),
        failures.join("\n\n")
    );
}

fn advance(
    parser: &mut Box<dyn dynamo_parsers_v2::UnifiedParser>,
    chunk: Option<&str>,
    state: &mut Result<Vec<dynamo_parsers_v2::UnifiedParserEvent>, String>,
) {
    if let Ok(deltas) = state {
        let result = match chunk {
            Some(chunk) => parser
                .push(chunk)
                .map_err(|e| format!("native push: {e:#}")),
            None => parser
                .finish()
                .map(|out| out.events)
                .map_err(|e| format!("native finish: {e:#}")),
        };
        match result {
            Ok(events) => deltas.extend(events),
            Err(error) => *state = Err(error),
        }
    }
}

/// I4: one parser per stream, so interleaving two streams cannot contaminate
/// either one.
#[test]
fn unified_parsers_are_isolated_per_stream() {
    for file in load_native_golden() {
        let cases: Vec<_> = file.cases.iter().collect();
        for pair in cases.windows(2) {
            let [(id_a, a), (id_b, b)] = pair else {
                continue;
            };
            let (ca, cb) = (
                capture::input_chunks(&a.input, a.input_chunks.as_deref()),
                capture::input_chunks(&b.input, b.input_chunks.as_deref()),
            );
            let tools_a = common::unified_tools_for_schemas(a.tools.as_ref());
            let tools_b = common::unified_tools_for_schemas(b.tools.as_ref());
            let solo_a = events(&file.family, &ca, &a.init, &tools_a);
            let solo_b = events(&file.family, &cb, &b.init, &tools_b);

            let mut pa = create_unified_parser_for_family(&file.family, &tools_a).unwrap();
            let mut pb = create_unified_parser_for_family(&file.family, &tools_b).unwrap();
            a.init.apply(&mut pa, id_a);
            b.init.apply(&mut pb, id_b);

            let (mut da, mut db) = (Ok(Vec::new()), Ok(Vec::new()));
            for i in 0..ca.len().max(cb.len()) {
                if let Some(c) = ca.get(i) {
                    advance(&mut pa, Some(c), &mut da);
                }
                if let Some(c) = cb.get(i) {
                    advance(&mut pb, Some(c), &mut db);
                }
            }
            advance(&mut pa, None, &mut da);
            advance(&mut pb, None, &mut db);

            assert_eq!(
                da.map(|d| assemble(&d)),
                solo_a,
                "{id_a}: interleaving with {id_b} changed its events"
            );
            assert_eq!(
                db.map(|d| assemble(&d)),
                solo_b,
                "{id_b}: interleaving with {id_a} changed its events"
            );
        }
    }
}

/// Guard the registry against drift, the same way the tool-only suite does.
#[test]
fn registered_unified_families_all_create() {
    for family in REGISTERED_UNIFIED_FAMILIES {
        create_unified_parser_for_family(family, &[]).unwrap_or_else(|e| {
            panic!("REGISTERED_UNIFIED_FAMILIES entry `{family}` does not create: {e}")
        });
    }
    assert!(
        !load_native_golden().is_empty(),
        "no golden family maps to a registered unified parser"
    );
}

/// The manifest and the parser registry must agree about which families are native.
///
/// These are two different systems — a YAML row read by the conformance harness, and a
/// `match` compiled into `dynamo-parsers-v2` — and nothing links them at compile time.
/// Before, five lists carried this and a family added to one but missed in another
/// failed loudly at best and silently lost coverage at worst. This is the one assertion
/// that keeps the single declaration honest, in both directions.
#[test]
fn manifest_and_parser_registry_agree_on_native_families() {
    let mut wrong = Vec::new();
    let declared: std::collections::BTreeSet<String> = common::unified_families()
        .iter()
        .filter(|(_, row)| row.native)
        .flat_map(|(family, row)| {
            [family.as_str(), row.registry_key(family)]
                .into_iter()
                .filter_map(dynamo_parsers_v2::canonical_unified_family)
                .map(str::to_string)
        })
        .collect();
    for (family, row) in common::unified_families() {
        let constructs = create_unified_parser_for_family(&family, &[]).is_ok();
        if row.native && !constructs {
            wrong.push(format!(
                "{family}: manifest says native, but create_unified_parser_for_family rejects it \
                 — add it to `unified_registry!` in parsers/v2/src/unified/mod.rs"
            ));
        }
        if !native_corpus_registration_valid(
            &family,
            row.native,
            constructs,
            dynamo_parsers_v2::canonical_unified_family(&family),
            &declared,
        ) && !row.native
        {
            wrong.push(format!(
                "{family}: a native UnifiedParser exists, but the manifest still says \
                 native: false — flip it in conformance/utils/src/parser_families.yaml"
            ));
        }
    }
    // ...and the other direction. Iterating manifest rows alone leaves a hole: a family
    // added to `unified_registry!` with NO manifest row is invisible here, constructs
    // fine, and silently gets no golden coverage — the exact failure this guard exists
    // to prevent, one level up.
    for registered in REGISTERED_UNIFIED_FAMILIES {
        let canonical =
            dynamo_parsers_v2::canonical_unified_family(registered).unwrap_or(registered);
        if !declared.contains(canonical) {
            wrong.push(format!(
                "{registered}: in `unified_registry!` but no native `unified:` row declares it \
                 — add one in conformance/utils/src/parser_families.yaml, or it gets no \
                 golden coverage"
            ));
        }
    }

    assert!(
        wrong.is_empty(),
        "manifest/registry disagree:\n  {}",
        wrong.join("\n  ")
    );
}

// Legacy split-pipeline corpora can share a registry alias with a native
// corpus. The canonical native owner must still declare and exercise its oracle.
fn native_corpus_registration_valid(
    family: &str,
    native: bool,
    constructs: bool,
    canonical: Option<&str>,
    declared: &std::collections::BTreeSet<String>,
) -> bool {
    if native {
        constructs
    } else {
        !constructs || canonical.is_some_and(|owner| owner != family && declared.contains(owner))
    }
}

#[test]
fn legacy_alias_never_replaces_a_native_oracle() {
    let declared = std::collections::BTreeSet::from(["gpt_oss".to_string()]);
    assert!(native_corpus_registration_valid(
        "harmony",
        false,
        true,
        Some("gpt_oss"),
        &declared
    ));
    assert!(!native_corpus_registration_valid(
        "gpt_oss",
        false,
        true,
        Some("gpt_oss"),
        &declared
    ));
    assert!(!native_corpus_registration_valid(
        "harmony",
        false,
        true,
        Some("gpt_oss"),
        &Default::default()
    ));
    assert!(!native_corpus_registration_valid(
        "harmony",
        true,
        false,
        Some("gpt_oss"),
        &declared
    ));
}

/// Compare only tool calls because the tool-only API has no separate reasoning channel.
#[test]
fn deepseek_tool_adapter_matches_both_native_golden_corpora() {
    use dynamo_parsers_v2::{ToolParseResult, create_tool_parser_for_family};

    let mut coverage = BTreeMap::<String, usize>::new();
    for file in load_native_golden()
        .into_iter()
        .filter(|file| matches!(file.family.as_str(), "deepseek_v4" | "deepseek_v41"))
    {
        for (id, case) in file.cases {
            if !matches!(case.init.tool_output_mode.as_str(), "" | "Native") {
                continue;
            }
            *coverage.entry(file.family.clone()).or_default() += 1;
            for (label, chunks) in splittings(&case.input) {
                let case_tools = common::unified_tools_for_schemas(case.tools.as_ref());
                let native = events_and_raw(&file.family, &chunks, &case.init, &case_tools).map(
                    |(events, raw)| {
                        let names = events
                            .into_iter()
                            .filter_map(|event| match event {
                                UnifiedEvent::ToolCall { name, .. } => Some(name),
                                _ => None,
                            })
                            .collect::<Vec<_>>();
                        assert_eq!(names.len(), raw.len(), "native raw argument inventory");
                        names
                            .into_iter()
                            .zip(raw.into_values())
                            .map(|(name, arguments)| {
                                (name, common::canonical_argument_json(&arguments))
                            })
                            .collect::<Vec<_>>()
                    },
                );
                let mut parser = create_tool_parser_for_family("deepseek_v4", &case_tools).unwrap();
                let mut result = ToolParseResult::default();
                let mut failure = None;
                for chunk in chunks {
                    match parser.push(&chunk) {
                        Ok(delta) => result.append(delta),
                        Err(error) => {
                            failure = Some(format!("native push: {error:#}"));
                            break;
                        }
                    }
                }
                if failure.is_none() {
                    match parser.finish() {
                        Ok(delta) => result.append(delta),
                        Err(error) => failure = Some(format!("native finish: {error:#}")),
                    }
                }
                let actual = failure.map_or_else(
                    || {
                        Ok(result
                            .coalesce_calls()
                            .calls
                            .into_iter()
                            .map(|call| {
                                (
                                    call.name.unwrap(),
                                    common::canonical_argument_json(&call.arguments),
                                )
                            })
                            .collect::<Vec<_>>())
                    },
                    Err,
                );
                assert_eq!(
                    actual, native,
                    "{} {id} {label}: tool adapter differs from native parser",
                    file.family
                );
            }
        }
    }
    for family in ["deepseek_v4", "deepseek_v41"] {
        assert!(
            coverage.get(family).copied().unwrap_or_default() > 0,
            "missing {family} corpus"
        );
    }
}
