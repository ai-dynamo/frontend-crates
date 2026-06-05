// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Stamp `delta_token_ids` into every chunk of every harmony stream fixture.
//!
//! Reads conformance/toolcalling/fixtures-stream-v2/harmony/TOOLCALLING.stream.*.yaml,
//! encodes the FULL concatenated delta_text per case with the gpt-oss harmony
//! tokenizer, then aligns those tokens back to individual chunks by tracking the
//! decoded byte cursor. The resulting per-chunk token ids form a valid token
//! sequence: special tokens like <|message|> that span a character-split boundary
//! are assigned to the earlier chunk rather than encoded as broken fragments.
//!
//! Usage (from repo root):
//!   cargo run -p dynamo-parsers-v2 --bin stamp_stream_token_ids

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::PathBuf;

use dynamo_parsers_v2::{decode_harmony, encode_harmony};
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    #[allow(dead_code)]
    family: String,
    #[allow(dead_code)]
    mode: Option<String>,
    #[serde(default)]
    cases: BTreeMap<String, Case>,
}

#[derive(Deserialize)]
struct Case {
    #[serde(default)]
    chunks: Vec<Chunk>,
}

#[derive(Deserialize)]
struct Chunk {
    #[serde(default)]
    delta_text: String,
}

fn main() -> anyhow::Result<()> {
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_path_buf();

    // Stamp the v2 stream overlay only. The v1 conformance corpus stays pristine
    // (dynamo-synced); the per-chunk token ids live outside the rsync path.
    let dirs = [repo_root.join("conformance/toolcalling/fixtures-stream-v2/harmony")];

    for root in &dirs {
        if !root.exists() {
            continue;
        }
        let mut files: Vec<_> = std::fs::read_dir(root)?
            .flatten()
            .map(|e| e.path())
            .filter(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .map(|n| n.starts_with("TOOLCALLING.stream") && n.ends_with(".yaml"))
                    .unwrap_or(false)
            })
            .collect();
        files.sort();

        for path in &files {
            let src = std::fs::read_to_string(path)?;
            let out = stamp_token_ids(&src)?;
            if out != src {
                std::fs::write(path, &out)?;
                println!("updated {}", path.display());
            } else {
                println!("no change {}", path.display());
            }
        }
    }
    Ok(())
}

/// Encode the full text for each case, align tokens to chunk boundaries, and
/// insert `delta_token_ids: [...]` lines after each `- delta_text:` line.
fn stamp_token_ids(src: &str) -> anyhow::Result<String> {
    // Parse YAML to extract chunk texts per case (preserving order).
    let fixture: Fixture = serde_yaml::from_str(src)?;

    // Build a flat, ordered list of per-chunk token id vectors — one entry per
    // `- delta_text:` line in the YAML, in document order.
    // BTreeMap sorts case ids by key, which matches the document order for these
    // fixtures.
    let mut all_chunk_ids: Vec<Vec<u32>> = Vec::new();
    for case in fixture.cases.values() {
        let chunk_ids = align_tokens_to_chunks(&case.chunks)?;
        all_chunk_ids.extend(chunk_ids);
    }

    // Line-oriented pass: insert `delta_token_ids:` after each `- delta_text:`.
    let mut out = String::with_capacity(src.len() + 512);
    let mut lines = src.lines().peekable();
    let mut chunk_cursor = 0usize;

    while let Some(line) = lines.next() {
        out.push_str(line);
        out.push('\n');

        let trimmed = line.trim_start();
        if !trimmed.starts_with("- delta_text:") {
            continue;
        }
        // Already stamped?
        if let Some(next) = lines.peek()
            && next.trim_start().starts_with("delta_token_ids:")
        {
            chunk_cursor += 1;
            continue;
        }

        let ids = &all_chunk_ids[chunk_cursor];
        chunk_cursor += 1;

        let indent_spaces = line.len() - line.trim_start().len();
        let id_indent = " ".repeat(indent_spaces + 2);
        let ids_str = ids_to_yaml_flow(ids);
        out.push_str(&id_indent);
        out.push_str("delta_token_ids: ");
        out.push_str(&ids_str);
        out.push('\n');
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

/// Render token ids as a YAML flow sequence: `[1, 2, 3]`
fn ids_to_yaml_flow(ids: &[u32]) -> String {
    if ids.is_empty() {
        return "[]".to_string();
    }
    let mut s = String::from("[");
    for (i, id) in ids.iter().enumerate() {
        if i > 0 {
            s.push_str(", ");
        }
        let _ = write!(s, "{id}");
    }
    s.push(']');
    s
}
