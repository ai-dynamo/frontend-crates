// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The acceptance gate for the unified parser: every `UNIFIED.*` case of every
//! family that has a unified parser must assemble EXACTLY to the authored golden
//! event list.
//!
//! `unified_schema_roundtrip` proves the corpus is well-formed and
//! `unified_render` draws it; this file is what fails CI when a parser is wrong.
//! It asserts the invariants from `conformance/utils/lib/parsers/UNIFIED_CASES.md`
//! that are checkable from the single-stream corpus:
//!
//! * `I2` order preservation — the assembled list is compared order-sensitively
//! * `I5` chunk-invariance — the same input under five chunk splittings
//! * `I6` stream/batch parity — `parse_complete` against the streamed result
//! * `I4` per-stream isolation — two concurrent parsers do not see each other

mod common;

use common::Init;

use std::collections::BTreeMap;

use dynamo_parsers_v2::{
    REGISTERED_UNIFIED_FAMILIES, Tool, UnifiedEvent, UnifiedParserExt, assemble,
    create_unified_parser_for_family,
};
use serde::Deserialize;
use serde_json::json;

#[derive(Deserialize)]
struct GoldenFile {
    family: String,
    cases: BTreeMap<String, GoldenCase>,
}

#[derive(Deserialize)]
struct GoldenCase {
    input: String,
    golden: Vec<UnifiedEvent>,
    /// Request-scoped parser configuration, declared by the case. Shared with
    /// `unified_render` via `common::Init` so both harnesses configure a case
    /// identically; see that type for why it is declared and not inferred.
    #[serde(default)]
    init: Init,
}

/// Tool schemas the corpus is written against (string params, so a value like
/// `1` stays the string `"1"` exactly as the golden records it). Mirrors
/// `tools()` in `unified_render.rs`.
fn tools() -> Vec<Tool> {
    let mk = |name: &str, key: &str| Tool {
        name: name.to_string(),
        description: None,
        parameters: json!({"type":"object","properties":{key:{"type":"string"}}}),
        strict: None,
    };
    vec![
        mk("get_weather", "city"),
        mk("f", "x"),
        mk("g", "y"),
        mk("run", "cmd"),
        mk("sum_values", "values"),
        mk("log", "note"),
    ]
}

fn load_golden() -> Vec<GoldenFile> {
    let dir = common::ensure_unified_golden();
    let mut files: Vec<GoldenFile> = Vec::new();
    for entry in std::fs::read_dir(&dir).unwrap_or_else(|e| panic!("read {}: {e}", dir.display())) {
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
    files
}

fn has_unified_parser(family: &str) -> bool {
    create_unified_parser_for_family(family, &[]).is_ok()
}

fn events(family: &str, chunks: &[String], init: &Init) -> Vec<UnifiedEvent> {
    let mut parser = create_unified_parser_for_family(family, &tools())
        .unwrap_or_else(|e| panic!("create unified parser for `{family}`: {e}"));
    init.apply(&mut parser, family);

    let mut deltas = Vec::new();
    for chunk in chunks {
        deltas.extend(parser.push(chunk).unwrap_or_else(|e| panic!("push: {e}")));
    }
    deltas.extend(parser.finish().unwrap_or_else(|e| panic!("finish: {e}")));
    assemble(&deltas)
}

fn render(events: &[UnifiedEvent]) -> String {
    events
        .iter()
        .map(|e| match e {
            UnifiedEvent::Reasoning { text } => format!("reasoning({text:?})"),
            UnifiedEvent::Text { text } => format!("text({text:?})"),
            UnifiedEvent::ToolCall { name, arguments } => format!("tool_call({name}, {arguments})"),
        })
        .collect::<Vec<_>>()
        .join("  |  ")
}

/// Marker-aligned chunking: each `<...>` control marker is its own chunk.
fn chunk_markers(input: &str) -> Vec<String> {
    let bytes = input.as_bytes();
    let mut chunks = Vec::new();
    let (mut i, mut text_start) = (0, 0);
    while i < bytes.len() {
        if bytes[i] == b'<' {
            if text_start < i {
                chunks.push(input[text_start..i].to_string());
            }
            let mut j = i + 1;
            while j < bytes.len() && bytes[j] != b'>' {
                j += 1;
            }
            let end = (j + 1).min(bytes.len());
            chunks.push(input[i..end].to_string());
            i = end;
            text_start = i;
        } else {
            i += 1;
        }
    }
    if text_start < bytes.len() {
        chunks.push(input[text_start..].to_string());
    }
    chunks
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

/// The gate: assembled events must equal the golden, exactly and in order.
#[test]
fn unified_parser_matches_the_golden_oracle() {
    let files = load_golden();
    let covered: Vec<&GoldenFile> = files
        .iter()
        .filter(|f| has_unified_parser(&f.family))
        .collect();
    assert!(
        !covered.is_empty(),
        "no golden family has a unified parser — the surface is unproven"
    );

    let mut failures = Vec::new();
    let mut checked = 0usize;
    for file in &covered {
        for (id, case) in &file.cases {
            checked += 1;
            let got = events(&file.family, &chunk_markers(&case.input), &case.init);
            if got != case.golden {
                failures.push(format!(
                    "{id}\n     input: {:?}\n    golden: {}\n   unified: {}",
                    case.input,
                    render(&case.golden),
                    render(&got),
                ));
            }
        }
    }

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
    let mut failures = Vec::new();
    for file in load_golden()
        .iter()
        .filter(|f| has_unified_parser(&f.family))
    {
        for (id, case) in &file.cases {
            let baseline = events(&file.family, std::slice::from_ref(&case.input), &case.init);
            for (label, chunks) in splittings(&case.input) {
                let got = events(&file.family, &chunks, &case.init);
                if got != baseline {
                    failures.push(format!(
                        "{id} [{label}, {} chunks]\n  whole: {}\n    got: {}",
                        chunks.len(),
                        render(&baseline),
                        render(&got),
                    ));
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "chunk splitting changed the assembled events ({}):\n\n{}",
        failures.len(),
        failures.join("\n\n"),
    );
}

/// I6: parsing the whole output at once assembles to the streamed result.
#[test]
fn unified_parser_has_stream_batch_parity() {
    for file in load_golden()
        .iter()
        .filter(|f| has_unified_parser(&f.family))
    {
        for (id, case) in &file.cases {
            let streamed = events(&file.family, &chunk_markers(&case.input), &case.init);
            let mut parser = create_unified_parser_for_family(&file.family, &tools()).unwrap();
            case.init.apply(&mut parser, id);

            let batch = parser
                .parse_complete(&case.input)
                .unwrap_or_else(|e| panic!("{id}: parse_complete: {e}"));
            assert_eq!(
                batch,
                streamed,
                "{id}: batch and stream disagree\n   batch: {}\n  stream: {}",
                render(&batch),
                render(&streamed),
            );
        }
    }
}

/// I4: one parser per stream, so interleaving two streams cannot contaminate
/// either one.
#[test]
fn unified_parsers_are_isolated_per_stream() {
    for file in load_golden()
        .iter()
        .filter(|f| has_unified_parser(&f.family))
    {
        let cases: Vec<_> = file.cases.iter().collect();
        for pair in cases.windows(2) {
            let [(id_a, a), (id_b, b)] = pair else {
                continue;
            };
            let (ca, cb) = (chunk_markers(&a.input), chunk_markers(&b.input));
            let solo_a = events(&file.family, &ca, &a.init);
            let solo_b = events(&file.family, &cb, &b.init);

            let mut pa = create_unified_parser_for_family(&file.family, &tools()).unwrap();
            let mut pb = create_unified_parser_for_family(&file.family, &tools()).unwrap();
            a.init.apply(&mut pa, id_a);
            b.init.apply(&mut pb, id_b);

            let (mut da, mut db) = (Vec::new(), Vec::new());
            for i in 0..ca.len().max(cb.len()) {
                if let Some(c) = ca.get(i) {
                    da.extend(pa.push(c).unwrap());
                }
                if let Some(c) = cb.get(i) {
                    db.extend(pb.push(c).unwrap());
                }
            }
            da.extend(pa.finish().unwrap().events);
            db.extend(pb.finish().unwrap().events);

            assert_eq!(
                assemble(&da),
                solo_a,
                "{id_a}: interleaving with {id_b} changed its events"
            );
            assert_eq!(
                assemble(&db),
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
        load_golden().iter().any(|f| has_unified_parser(&f.family)),
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
        .flat_map(|(family, row)| [family.clone(), row.registry_key(family).to_string()])
        .collect();
    for registered in REGISTERED_UNIFIED_FAMILIES {
        if !declared.contains(*registered) {
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
