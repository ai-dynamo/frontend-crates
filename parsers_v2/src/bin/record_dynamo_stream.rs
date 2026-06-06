// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Record Dynamo parser v2 per-chunk streaming emit into harmony stream fixtures.
//!
//! Reads conformance/toolcalling/fixtures-stream-v2/harmony/TOOLCALLING.stream.*.yaml, runs
//! HarmonyToolStreamParser over each case's chunks via BOTH input paths:
//!   - token path  (parse_tool_call_streaming_incremental) when a chunk carries
//!     delta_token_ids
//!   - text path   (parse_tool_call_streaming_text) otherwise
//!
//! Prints the per-chunk emitted deltas as JSON so they can be written into
//! `chunks[].expected.dynamo`.
//!
//! Output JSON: {case_id: [[{index,id,name,arguments}, ...], ...]}
//! Usage:
//!   cargo run -p dynamo-parsers-v2 --bin record_dynamo_stream -- <fixture.yaml>
//!
//! The binary name is legacy; the code under test is Dynamo parser v2.

use std::collections::BTreeMap;

use dynamo_parsers::tool_calling::ToolCallResponseChunk;
use dynamo_parsers_v2::HarmonyToolStreamParser;
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
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
    #[serde(default)]
    delta_token_ids: Vec<u32>,
}

fn main() -> anyhow::Result<()> {
    // Args: <fixture.yaml> [--text]
    //   default: token path (delta_token_ids per chunk, else text)
    //   --text : force the text path (parse_tool_call_streaming_text) per chunk
    let args: Vec<String> = std::env::args().skip(1).collect();
    let force_text = args.iter().any(|a| a == "--text");
    let path = args
        .iter()
        .find(|a| !a.starts_with("--"))
        .ok_or_else(|| anyhow::anyhow!("usage: record_dynamo_stream <fixture.yaml> [--text]"))?;
    let src = std::fs::read_to_string(path)?;
    let fx: Fixture = serde_yaml::from_str(&src)?;

    let mut out = String::from("{\n");
    let case_count = fx.cases.len();
    for (ci, (cid, case)) in fx.cases.iter().enumerate() {
        let mut parser = HarmonyToolStreamParser::new()?;
        let mut per_chunk_raw: Vec<Vec<ToolCallResponseChunk>> = Vec::new();
        for chunk in &case.chunks {
            let result = if force_text {
                parser.parse_tool_call_streaming_text(&chunk.delta_text)
            } else if !chunk.delta_token_ids.is_empty() {
                parser.parse_tool_call_streaming_incremental(&chunk.delta_token_ids)
            } else {
                parser.parse_tool_call_streaming_text(&chunk.delta_text)
            };
            per_chunk_raw.push(result.tool_call_chunks);
        }
        // Flush at finish (the text path can hold a suffix until EOS); attribute
        // the flushed deltas to the last chunk (which is the finish_reason chunk).
        let fin = parser.finish_tool_call_stream().tool_call_chunks;
        if !fin.is_empty() {
            if let Some(last) = per_chunk_raw.last_mut() {
                last.extend(fin);
            } else {
                per_chunk_raw.push(fin);
            }
        }
        let per_chunk: Vec<String> = per_chunk_raw
            .iter()
            .map(|deltas| {
                let rendered: Vec<String> = deltas
                    .iter()
                    .map(|c| {
                        let mut parts = vec![format!("\"index\": {}", c.index)];
                        if c.id.is_some() {
                            parts.push("\"id\": true".to_string());
                        }
                        if let Some(f) = &c.function {
                            if let Some(n) = &f.name {
                                parts.push(format!("\"name\": {}", json_str(n)));
                            }
                            if let Some(a) = &f.arguments {
                                parts.push(format!("\"arguments\": {}", json_str(a)));
                            }
                        }
                        format!("{{{}}}", parts.join(", "))
                    })
                    .collect();
                format!("[{}]", rendered.join(", "))
            })
            .collect();
        let trailing = if ci + 1 < case_count { "," } else { "" };
        out.push_str(&format!(
            "  {}: [{}]{}\n",
            json_str(cid),
            per_chunk.join(", "),
            trailing
        ));
    }
    out.push_str("}\n");
    print!("{out}");
    Ok(())
}

fn json_str(s: &str) -> String {
    serde_json::to_string(s).unwrap_or_else(|_| "\"\"".to_string())
}
