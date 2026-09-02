// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Record Dynamo parser v2 per-chunk streaming emit into stream fixtures.
//!
//! Reads conformance/toolcalling/fixtures-stream-v2/harmony/TOOLCALLING.stream.*.yaml, runs
//! HarmonyToolStreamParser over each case's chunks through one selected input path:
//!   - default: token path (parse_tool_call_streaming_incremental), using
//!     delta_token_ids only
//!   - --text: text path (parse_tool_call_streaming_text), using delta_text
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

use dynamo_parsers_v2::{
    HarmonyToolStreamParser, Tool, ToolCallDelta, ToolCallResponseChunk, ToolParseResult,
    ToolParserInput, create_tool_parser_for_family,
};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Fixture {
    family: String,
    #[serde(default)]
    cases: BTreeMap<String, Case>,
}

#[derive(Deserialize)]
struct Case {
    #[serde(default)]
    tools: Vec<Tool>,
    #[serde(default)]
    chunks: Vec<Chunk>,
}

#[derive(Deserialize)]
struct Chunk {
    #[serde(default)]
    delta_text: String,
    #[serde(default)]
    delta_token_ids: Vec<u32>,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Serialize)]
struct ChunkEmit {
    deltas: Vec<DeltaEmit>,
    normal_text: String,
}

#[derive(Serialize)]
struct DeltaEmit {
    index: usize,
    #[serde(skip_serializing_if = "is_false")]
    id: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    arguments: Option<String>,
    complete: bool,
}

fn main() -> anyhow::Result<()> {
    // Args: <fixture.yaml> [--text]
    //   default: token path (delta_token_ids per chunk)
    //   --text : force the text path (parse_tool_call_streaming_text) per chunk
    let args: Vec<String> = std::env::args().skip(1).collect();
    let force_text = args.iter().any(|a| a == "--text");
    let path = args
        .iter()
        .find(|a| !a.starts_with("--"))
        .ok_or_else(|| anyhow::anyhow!("usage: record_dynamo_stream <fixture.yaml> [--text]"))?;
    let src = std::fs::read_to_string(path)?;
    let fx: Fixture = serde_yaml::from_str(&src)?;

    let mut out = BTreeMap::new();
    for (cid, case) in &fx.cases {
        let per_chunk = if fx.family == "harmony" || fx.family == "harmony_text" {
            record_harmony(case, force_text)?
        } else {
            record_trait_parser(&fx.family, case)?
        };
        out.insert(cid.clone(), per_chunk);
    }
    println!("{}", serde_json::to_string_pretty(&out)?);
    Ok(())
}

fn record_harmony(case: &Case, force_text: bool) -> anyhow::Result<Vec<ChunkEmit>> {
    let mut parser = HarmonyToolStreamParser::new()?;
    let mut per_chunk: Vec<ChunkEmit> = Vec::new();
    for chunk in &case.chunks {
        let result = if force_text {
            parser.parse_tool_call_streaming_text(&chunk.delta_text)
        } else {
            parser.parse_tool_call_streaming_incremental(&chunk.delta_token_ids)
        };
        per_chunk.push(ChunkEmit {
            deltas: result
                .tool_call_chunks
                .into_iter()
                .map(delta_from_chunk)
                .collect(),
            normal_text: result.normal_text,
        });
    }
    let fin = parser.finish_tool_call_stream();
    append_finish(
        &mut per_chunk,
        fin.tool_call_chunks
            .into_iter()
            .map(delta_from_chunk)
            .collect(),
        &fin.normal_text,
    );
    Ok(per_chunk)
}

fn record_trait_parser(family: &str, case: &Case) -> anyhow::Result<Vec<ChunkEmit>> {
    let mut parser = create_tool_parser_for_family(family, &case.tools)?;
    let mut per_chunk = Vec::new();
    for chunk in &case.chunks {
        // B9: data-driven input — a token-native parser consumes token ids, others
        // text, via the shared push_input abstraction (no family-name branch).
        let input = if parser.prefers_tokens() {
            ToolParserInput::Tokens(&chunk.delta_token_ids)
        } else {
            ToolParserInput::Text(&chunk.delta_text)
        };
        let mut result = parser.push_input(input)?;
        if chunk.finish_reason.is_some() {
            result.append(parser.finish()?);
        }
        per_chunk.push(chunk_emit_from_parse_result(result));
    }
    Ok(per_chunk)
}

fn append_finish(per_chunk: &mut Vec<ChunkEmit>, deltas: Vec<DeltaEmit>, normal_text: &str) {
    if deltas.is_empty() && normal_text.is_empty() {
        return;
    }
    if let Some(last) = per_chunk.last_mut() {
        last.deltas.extend(deltas);
        last.normal_text.push_str(normal_text);
    } else {
        per_chunk.push(ChunkEmit {
            deltas,
            normal_text: normal_text.to_string(),
        });
    }
}

fn chunk_emit_from_parse_result(result: ToolParseResult) -> ChunkEmit {
    ChunkEmit {
        deltas: result
            .calls
            .into_iter()
            .map(delta_from_tool_delta)
            .collect(),
        normal_text: result.normal_text,
    }
}

fn delta_from_tool_delta(delta: ToolCallDelta) -> DeltaEmit {
    DeltaEmit {
        index: delta.tool_index,
        id: false,
        name: delta.name,
        arguments: Some(delta.arguments),
        complete: delta.complete,
    }
}

fn delta_from_chunk(chunk: ToolCallResponseChunk) -> DeltaEmit {
    DeltaEmit {
        index: chunk.index as usize,
        id: chunk.id.is_some(),
        name: chunk.function.as_ref().and_then(|f| f.name.clone()),
        arguments: chunk.function.as_ref().and_then(|f| f.arguments.clone()),
        complete: true,
    }
}

fn is_false(value: &bool) -> bool {
    !*value
}
