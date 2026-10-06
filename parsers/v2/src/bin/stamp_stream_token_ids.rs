// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Stamp `delta_token_ids` into every chunk of every harmony stream fixture.
//!
//! Reads conformance/toolcalling/fixtures-stream-v1/harmony/TOOLCALLING.stream.*.yaml,
//! encodes the FULL concatenated delta_text per case with the gpt-oss harmony
//! tokenizer, then aligns those tokens back to individual chunks by tracking the
//! decoded byte cursor. The resulting per-chunk token ids form a valid token
//! sequence: special tokens like <|message|> that span a character-split boundary
//! are assigned to the earlier chunk rather than encoded as broken fragments.
//!
//! Usage (from repo root):
//!   cargo run -p dynamo-parsers-v2 --bin stamp_stream_token_ids

use std::path::PathBuf;

use dynamo_parsers_v2::{decode_harmony, encode_harmony};
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    cases: serde_yaml::Mapping,
}

#[derive(Deserialize)]
struct Case {
    #[serde(default)]
    chunks: Vec<Chunk>,
}

#[derive(Deserialize)]
struct Chunk {
    delta_text: String,
    #[serde(default)]
    delta_token_ids: Option<Vec<u32>>,
}

fn main() -> anyhow::Result<()> {
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        // this crate lives at parsers/v2, so the repo root is two levels up
        .parent()
        .and_then(|p| p.parent())
        .expect("parsers/v2 is two levels below the repo root")
        .to_path_buf();

    let args: Vec<String> = std::env::args().skip(1).collect();
    if args == ["--help"] {
        println!("usage: stamp_stream_token_ids [--input FILE]");
        return Ok(());
    }
    let mut files: Vec<PathBuf> = if args.len() == 2 && args[0] == "--input" {
        vec![PathBuf::from(&args[1])]
    } else {
        anyhow::ensure!(
            args.is_empty(),
            "usage: stamp_stream_token_ids [--input FILE]"
        );
        let root = repo_root.join("conformance/toolcalling/fixtures-stream-v1/harmony");
        if !root.exists() {
            return Ok(());
        }
        std::fs::read_dir(root)?
            .map(|entry| entry.map(|entry| entry.path()))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .filter(|path| {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| {
                        name.starts_with("TOOLCALLING.stream") && name.ends_with(".yaml")
                    })
            })
            .collect()
    };
    files.sort();
    for path in files {
        let src = std::fs::read_to_string(&path)?;
        let out = stamp_token_ids(&src)?;
        if out != src {
            std::fs::write(&path, out)?;
            println!("updated {}", path.display());
        }
    }

    Ok(())
}

/// Encode the full text for each case, align tokens to chunk boundaries, and
/// add token IDs to each parsed chunk while preserving its owning case and text.
/// Source edits avoid reserializing unrelated numeric schema and golden values.
fn stamp_token_ids(src: &str) -> anyhow::Result<String> {
    let fixture: Fixture = serde_yaml::from_str(src)?;
    let mut chunk_ids = Vec::new();
    for case_value in fixture.cases.values() {
        let case: Case = serde_yaml::from_value(case_value.clone())?;
        let ids_by_chunk = align_tokens_to_chunks(&case.chunks)?;
        chunk_ids.extend(
            case.chunks
                .iter()
                .zip(ids_by_chunk)
                .map(|(chunk, ids)| (chunk.delta_token_ids.is_some(), ids)),
        );
    }

    let lines = src.split_inclusive('\n').collect::<Vec<_>>();
    let mut insertions = Vec::new();
    let mut line_index = 0;
    let mut chunk_index = 0;
    let mut chunks_indent = None;
    let mut item_indent = None;
    while line_index < lines.len() {
        let content = lines[line_index].trim_end_matches(['\n', '\r']);
        let trimmed = content.trim_start();
        let indent = content.len() - trimmed.len();
        if trimmed == "chunks:" {
            chunks_indent = Some(indent);
            item_indent = None;
            line_index += 1;
            continue;
        }
        if let Some(parent_indent) = chunks_indent
            && !trimmed.is_empty()
            && !trimmed.starts_with('#')
            && indent <= parent_indent
            && !(indent == parent_indent && trimmed.starts_with("- "))
        {
            chunks_indent = None;
            item_indent = None;
        }
        if chunks_indent.is_none() || !trimmed.starts_with("- ") {
            line_index += 1;
            continue;
        }
        let expected_item_indent = *item_indent.get_or_insert(indent);
        if indent != expected_item_indent || !trimmed.starts_with("- delta_text:") {
            line_index += 1;
            continue;
        }
        anyhow::ensure!(
            chunk_index < chunk_ids.len(),
            "found more delta_text fields in YAML than parsed chunks"
        );
        let (already_stamped, ids) = &chunk_ids[chunk_index];
        chunk_index += 1;

        let field_indent = expected_item_indent + 2;
        let mut end = line_index + 1;
        while end < lines.len() {
            let next = lines[end].trim_end_matches(['\n', '\r']);
            if next.trim().is_empty() {
                end += 1;
                continue;
            }
            if next.len() - next.trim_start().len() <= field_indent {
                break;
            }
            end += 1;
        }
        if !already_stamped {
            let indent = " ".repeat(field_indent);
            insertions.push((
                end,
                format!("{indent}delta_token_ids: {}\n", ids_to_yaml_flow(ids)),
            ));
        }
        line_index = end;
    }
    anyhow::ensure!(
        chunk_index == chunk_ids.len(),
        "parsed chunk count does not match delta_text fields in YAML"
    );

    let mut out = String::with_capacity(src.len() + insertions.len() * 32);
    let mut insertion_index = 0;
    for line_index in 0..=lines.len() {
        while insertion_index < insertions.len() && insertions[insertion_index].0 == line_index {
            out.push_str(&insertions[insertion_index].1);
            insertion_index += 1;
        }
        if let Some(line) = lines.get(line_index) {
            out.push_str(line);
        }
    }
    Ok(out)
}

/// Encode the full concatenated text for a case and align the resulting token
/// ids back to individual chunks using a decoded-byte cursor.
///
/// Each token is assigned to the chunk whose cumulative byte boundary it first
/// crosses. A token that spans a chunk boundary (common for character-split
/// fixtures) is assigned entirely to the earlier chunk, giving that chunk a
/// slightly longer token span, but the total token sequence is valid.
fn align_tokens_to_chunks(chunks: &[Chunk]) -> anyhow::Result<Vec<Vec<u32>>> {
    // Cumulative byte lengths for each chunk.
    let cumulative_bytes: Vec<usize> = chunks
        .iter()
        .scan(0usize, |acc, c| {
            *acc += c.delta_text.len();
            Some(*acc)
        })
        .collect();

    let full_text: String = chunks.iter().map(|c| c.delta_text.as_str()).collect();
    let tokens = encode_harmony(&full_text)?;

    let mut result: Vec<Vec<u32>> = vec![Vec::new(); chunks.len()];
    let mut byte_cursor = 0usize;
    let mut chunk_idx = 0;
    let mut undecoded: Vec<u32> = Vec::new();

    for &token in &tokens {
        undecoded.push(token);

        // Accumulate tokens until they decode cleanly (handles partial UTF-8).
        let decoded = decode_harmony(&undecoded).unwrap_or_default();
        if decoded.is_empty() {
            continue;
        }
        byte_cursor += decoded.len();
        let drained = std::mem::take(&mut undecoded);

        // Advance chunk_idx to the chunk that contains byte_cursor.
        while chunk_idx + 1 < chunks.len() && byte_cursor > cumulative_bytes[chunk_idx] {
            chunk_idx += 1;
        }

        for t in drained {
            result[chunk_idx].push(t);
        }
    }

    // Flush any remaining undecoded tokens (shouldn't happen with valid UTF-8).
    for t in undecoded {
        result[chunk_idx.min(chunks.len() - 1)].push(t);
    }

    Ok(result)
}

/// Render token IDs as a YAML flow sequence: `[1, 2, 3]`.
fn ids_to_yaml_flow(ids: &[u32]) -> String {
    format!("{ids:?}")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_each_case_decodes_to_its_text(src: &str) {
        let stamped = stamp_token_ids(src).expect("stamp fixture");
        let fixture: serde_yaml::Value =
            serde_yaml::from_str(&stamped).expect("parse stamped fixture");
        let cases = fixture["cases"].as_mapping().expect("cases mapping");
        for (case_id, case) in cases {
            let chunks = case["chunks"].as_sequence().expect("chunks sequence");
            for chunk in chunks {
                let text = chunk["delta_text"].as_str().unwrap_or_default();
                let ids = chunk["delta_token_ids"]
                    .as_sequence()
                    .expect("stamped token IDs")
                    .iter()
                    .map(|id| id.as_u64().expect("numeric token ID") as u32)
                    .collect::<Vec<_>>();
                assert_eq!(
                    decode_harmony(&ids).expect("decode token IDs"),
                    text,
                    "{case_id:?}"
                );
            }
        }
        assert_eq!(stamp_token_ids(&stamped).expect("restamp fixture"), stamped);
    }

    #[test]
    fn stamps_cases_in_yaml_order_even_when_ids_are_unsorted() {
        assert_each_case_decodes_to_its_text(
            "family: harmony\nmode: streamv1\ncases:\n  z:\n    chunks:\n    - delta_text: Hello\n  a:\n    chunks:\n    - delta_text: World\n",
        );
    }

    #[test]
    fn preserves_multiline_chunk_text_when_stamping() {
        assert_each_case_decodes_to_its_text(
            "family: harmony\nmode: streamv1\ncases:\n  one:\n    chunks:\n    - delta_text: |-\n        hello\n        world\n",
        );
    }

    #[test]
    fn preserves_exact_unrelated_numeric_scalars() {
        let src = "family: harmony\ncases:\n  one:\n    tools:\n    - parameters: {const: 9007199254740992.5}\n    golden: {value: 9007199254740992.5}\n    chunks:\n    - delta_text: Hello\n";
        let stamped = stamp_token_ids(src).expect("stamp fixture");
        assert!(stamped.contains("const: 9007199254740992.5"));
        assert!(stamped.contains("value: 9007199254740992.5"));
        assert!(!stamped.contains("9007199254740992.0"));
    }

    #[test]
    fn rejects_malformed_case_chunk_and_text_types() {
        for src in [
            "cases: bad\n",
            "cases:\n  one:\n    chunks: bad\n",
            "cases:\n  one:\n    chunks:\n    - delta_text: 42\n",
        ] {
            assert!(
                stamp_token_ids(src).is_err(),
                "accepted malformed YAML: {src}"
            );
        }
    }
}
