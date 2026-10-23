// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The acceptance gate for the unified parser: every `UNIFIED.*` case of every
//! family that has a unified parser must assemble EXACTLY to the authored golden
//! event list, except for documented parser defects in `known-divergences.yaml`.
//!
//! `unified_schema_roundtrip` proves the corpus is well-formed and
//! `unified_render` draws it; undocumented or stale divergences fail CI here.
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
    files.retain(|file| has_unified_parser(&file.family));
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
            assert_eq!(
                events(&file.family, &chunks, &case.init, &tools).unwrap(),
                case.golden,
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
        ["deepseek_v4", "glm47", "gpt_oss", "muse_spark", "qwen3"]
    );
}

fn has_unified_parser(family: &str) -> bool {
    create_unified_parser_for_family(family, &[]).is_ok()
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

/// Quote-looking strings must survive every possible two-chunk boundary,
/// including boundaries inside their native escaping and framing.
#[test]
fn string_quote_controls_survive_every_split_for_every_family() {
    let mut covered = std::collections::BTreeSet::new();
    for file in load_native_golden() {
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

/// The gate: assembled events must equal the golden, exactly and in order.
#[test]
fn unified_parser_matches_the_golden_oracle() {
    let files = load_native_golden();
    let known = divergences::load();
    assert!(
        !files.is_empty(),
        "no golden family has a unified parser — the surface is unproven"
    );

    let mut failures = Vec::new();
    let mut observed = std::collections::BTreeSet::new();
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
                if actual == &expected {
                    continue;
                }
            }
            if !got
                .as_ref()
                .is_ok_and(|events| common::json_values_equal(events, &case.golden))
            {
                match divergences::expected(&known, &file.family, id, Check::Golden) {
                    Some(Expected::Golden(expected)) if render(&got) == expected.actual => {
                        observed.insert((file.family.clone(), id.clone()));
                    }
                    Some(Expected::Golden(expected)) => failures.push(format!(
                        "{id}: known golden divergence changed\n expected: {}\n      got: {}",
                        expected.actual,
                        render(&got),
                    )),
                    _ => failures.push(format!(
                        "{id}\n     input: {:?}\n    golden: {}\n   unified: {}",
                        case.input,
                        render(&Ok(case.golden.clone())),
                        render(&got),
                    )),
                }
            }
        }
    }

    failures.extend(divergences::reconcile(&known, Check::Golden, &observed));
    assert!(
        failures.is_empty(),
        "{} of {checked} unified cases diverge from the golden oracle:\n\n{}",
        failures.len(),
        failures.join("\n\n"),
    );
    assert!(checked >= 30, "expected the qwen3 corpus, got {checked}");
}

/// I5: the assembled list must not depend on where chunk boundaries fall.
#[test]
fn unified_parser_is_chunk_invariant() {
    let known = divergences::load();
    let mut observed = std::collections::BTreeSet::new();
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
            for (label, chunks) in splittings(&case.input) {
                let got = events(&file.family, &chunks, &case.init, &case_tools);
                if got != baseline {
                    match divergences::expected(
                        &known,
                        &file.family,
                        id,
                        Check::ChunkInvariance,
                    ) {
                        Some(Expected::ChunkInvariance(expected))
                            if render(&baseline) == expected.baseline
                                && render(&got) == expected.divergent =>
                        {
                            observed.insert((file.family.clone(), id.clone()));
                        }
                        Some(Expected::ChunkInvariance(expected)) => failures.push(format!(
                            "{id} [{label}, {} chunks]: known divergence changed\n expected baseline: {}\n expected split: {}\n actual baseline: {}\n actual split: {}",
                            chunks.len(),
                            expected.baseline,
                            expected.divergent,
                            render(&baseline),
                            render(&got),
                        )),
                        _ => failures.push(format!(
                            "{id} [{label}, {} chunks]\n  whole: {}\n    got: {}",
                            chunks.len(),
                            render(&baseline),
                            render(&got),
                        )),
                    }
                }
            }
        }
    }
    failures.extend(divergences::reconcile(
        &known,
        Check::ChunkInvariance,
        &observed,
    ));
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
                assert_eq!(whole.as_ref(), Ok(&case.golden), "{id}");
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
    assert_eq!(
        checked,
        native_families * reference_scenarios.len()
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
    for (family, row) in common::unified_families() {
        let constructs = create_unified_parser_for_family(&family, &[]).is_ok();
        if row.native && !constructs {
            wrong.push(format!(
                "{family}: manifest says native, but create_unified_parser_for_family rejects it \
                 — add it to `unified_registry!` in parsers/v2/src/unified/mod.rs"
            ));
        }
        if !row.native && constructs {
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

/// Compare only tool calls because the tool-only API has no separate reasoning channel.
#[test]
fn deepseek_tool_adapter_matches_both_native_golden_corpora() {
    use dynamo_parsers_v2::{ToolParseResult, create_tool_parser_for_family};

    let known = divergences::load();
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
            let expected: Vec<_> = case
                .golden
                .into_iter()
                .filter(|event| matches!(event, UnifiedEvent::ToolCall { .. }))
                .collect();
            for (label, chunks) in splittings(&case.input) {
                let case_tools = common::unified_tools_for_schemas(case.tools.as_ref());
                let mut parser = create_tool_parser_for_family("deepseek_v4", &case_tools).unwrap();
                let mut result = ToolParseResult::default();
                let mut failure = None;
                for chunk in chunks {
                    match parser.push(&chunk) {
                        Ok(delta) => result.append(delta),
                        Err(error) => {
                            failure = Some(format!("ERROR: native push: {error:#}"));
                            break;
                        }
                    }
                }
                if let Some(error) = failure {
                    let Some(Expected::Golden(expected_error)) =
                        divergences::expected(&known, &file.family, &id, Check::Golden)
                    else {
                        panic!("{id} {label}: unexpected {error}");
                    };
                    assert_eq!(error, expected_error.actual, "{id} {label}");
                    continue;
                }
                result.append(
                    parser
                        .finish()
                        .unwrap_or_else(|e| panic!("{id} {label}: {e}")),
                );
                let actual: Vec<_> = result
                    .coalesce_calls()
                    .calls
                    .into_iter()
                    .map(|call| UnifiedEvent::ToolCall {
                        name: call.name.unwrap(),
                        arguments: serde_json::from_str(&call.arguments).unwrap(),
                    })
                    .collect();
                assert_eq!(actual, expected, "{} {id} {label}", file.family);
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
