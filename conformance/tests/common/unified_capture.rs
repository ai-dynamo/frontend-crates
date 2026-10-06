// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde::Deserialize;

#[derive(Debug)]
pub enum CaptureFailure {
    Unavailable(String),
    Error(String),
}

pub fn capture_step<T, E: std::fmt::Display>(
    result: Result<T, E>,
    stage: &str,
) -> Result<T, CaptureFailure> {
    result.map_err(|error| CaptureFailure::Error(format!("{stage}: {error:#}")))
}

/// Tokenize an input into streaming chunks: each control marker (`<...>`, incl.
/// `<|...|>` / `<|"|>`) is its own chunk, and each run of text between markers is
/// a chunk. Generic across the gemma4 / qwen3 / kimi grammars.
pub fn chunk_input(input: &str) -> Vec<String> {
    let bytes = input.as_bytes();
    let mut chunks = Vec::new();
    let mut i = 0;
    let mut text_start = 0;
    while i < bytes.len() {
        if bytes[i] == b'<' {
            // flush any pending text run
            if text_start < i {
                chunks.push(input[text_start..i].to_string());
            }
            // consume through the matching '>'
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

pub fn input_chunks(input: &str, authored: Option<&[String]>) -> Vec<String> {
    match authored {
        Some(chunks) => {
            assert_eq!(
                chunks.concat(),
                input,
                "authored input_chunks must reproduce input"
            );
            chunks.to_vec()
        }
        None => chunk_input(input),
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InputChunk {
    pub delta_text: String,
}

pub fn replay_chunks(input: &str, recorded: Option<&[InputChunk]>) -> Vec<String> {
    match recorded {
        None => input_chunks(input, None),
        Some(rows) => {
            let (finish, rows) = rows
                .split_last()
                .expect("recorded schedule requires finish");
            assert_eq!(
                finish.delta_text, "‹finish›",
                "recorded schedule requires finish"
            );
            let chunks: Vec<_> = rows.iter().map(|row| row.delta_text.clone()).collect();
            input_chunks(input, Some(&chunks))
        }
    }
}

#[cfg(not(conformance_split_only))]
pub fn native_capture(
    parser: &mut Box<dyn dynamo_parsers_v2::UnifiedParser>,
    chunks: &[String],
) -> Result<Vec<Vec<dynamo_parsers_v2::UnifiedParserEvent>>, CaptureFailure> {
    use dynamo_parsers_v2::UnifiedParserExt;
    let mut rows = Vec::new();
    for chunk in chunks {
        rows.push(capture_step(parser.push(chunk), "native push")?);
    }
    rows.push(
        capture_step(parser.finish(), "native finish")?
            .into_iter()
            .collect(),
    );
    Ok(rows)
}

#[test]
fn authored_and_recorded_schedules_preserve_utf8_and_empty_chunks() {
    let input = "<call>éCafé</call>";
    let authored = vec![input.to_string(), String::new()];
    assert_eq!(input_chunks(input, Some(&authored)), authored);
    let mut rows: Vec<_> = authored
        .iter()
        .map(|s| InputChunk {
            delta_text: s.clone(),
        })
        .collect();
    rows.push(InputChunk {
        delta_text: "‹finish›".into(),
    });
    assert_eq!(replay_chunks(input, Some(&rows)), authored);
    assert_eq!(input_chunks(input, None), chunk_input(input));
    assert!(std::panic::catch_unwind(|| input_chunks(input, Some(&["wrong".into()]))).is_err());
    assert!(std::panic::catch_unwind(|| replay_chunks(input, Some(&rows[..1]))).is_err());
}
