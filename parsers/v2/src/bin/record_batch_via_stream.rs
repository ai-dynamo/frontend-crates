// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Record Dynamo Rust stream parser results on BATCH samples, so the
//! parity generator (Python, can't run the Rust parser) can render the
//! "Stream parser on batch" tab. Feeds each batch fixture's full `model_text`
//! through the streaming parser (text path) + finish, and emits the assembled
//! calls per case as JSON.
//!
//! Output:
//! - `--family <family>`: {case_id: {"calls": [...], "normal_text": "..."}}
//! - no `--family`: {family: {case_id: {"calls": [...], "normal_text": "..."}}}
//!
//! Usage:
//!   cargo run -p dynamo-parsers-v2 --bin record_batch_via_stream -- --family deepseek_v4

use std::collections::BTreeMap;
use std::path::PathBuf;

use dynamo_parsers_v2::{
    HarmonyToolStreamParser, Tool, ToolCallDelta, ToolParseResult, ToolParserInput,
    assemble_tool_calls, create_tool_parser_for_family,
};
use serde::{Deserialize, Serialize};
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
    // Schema-dependent parsers (glm47, kimi_k2, qwen3_coder, minimax_m2, …) need
    // the tool schema to coerce argument types the way the v1 batch parser did.
    #[serde(default)]
    tools: Vec<Tool>,
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let family = args
        .windows(2)
        .find(|pair| pair[0] == "--family")
        .map(|pair| pair[1].clone());
    let families = family
        .as_ref()
        .map(|f| vec![f.as_str()])
        .unwrap_or_else(|| vec!["harmony", "deepseek_v4"]);

    // `--root <dir>`: a `<family>/TOOLCALLING.batch*.yaml` tree carrying
    // `model_text` + `tools` per case (e.g. `fixtures-batch-v1/inputs/` from the
    // fixture cache). Default: the legacy staged flat tree.
    let fixture_root = args
        .windows(2)
        .find(|pair| pair[0] == "--root")
        .map(|pair| PathBuf::from(&pair[1]))
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                // this crate lives at parsers/v2, so the repo root is two levels up
                .parent()
                .and_then(|p| p.parent())
                .expect("parsers/v2 is two levels below the repo root")
                .join("conformance/toolcalling/fixtures-v1")
        });

    let mut nested = BTreeMap::new();
    for family in families {
        nested.insert(
            family.to_string(),
            record_family(&fixture_root, family)
                .map_err(|e| anyhow::anyhow!("record {family}: {e}"))?,
        );
    }

    if let Some(family) = family {
        println!("{}", serde_json::to_string_pretty(&nested[&family])?);
    } else {
        println!("{}", serde_json::to_string_pretty(&nested)?);
    }
    Ok(())
}

fn record_family(
    root: &std::path::Path,
    family: &str,
) -> anyhow::Result<BTreeMap<String, CaseOut>> {
    let dir = root.join(family);
    let mut files: Vec<_> = std::fs::read_dir(&dir)?
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .map(|n| n.starts_with("TOOLCALLING.batch") && n.ends_with(".yaml"))
                .unwrap_or(false)
        })
        .collect();
    files.sort();

    let mut out = BTreeMap::new();
    for path in &files {
        let fx: Fixture = serde_yaml::from_str(&std::fs::read_to_string(path)?)?;
        if fx.family != family || fx.mode != "batch" {
            continue;
        }
        for (cid, case) in &fx.cases {
            let Some(text) = case.model_text.as_ref() else {
                continue;
            };
            out.insert(
                cid.clone(),
                CaseOut {
                    result: parse_result(family, text, &case.tools)?,
                },
            );
        }
    }
    Ok(out)
}

fn parse_result(family: &str, text: &str, tools: &[Tool]) -> anyhow::Result<CaseOutInner> {
    if family == "harmony" {
        let mut parser = HarmonyToolStreamParser::new()?;
        let mut result = parser.parse_tool_call_streaming_text(text);
        let finish = parser.finish_tool_call_stream();
        result.normal_text.push_str(&finish.normal_text);
        result.tool_call_chunks.extend(finish.tool_call_chunks);
        return Ok(CaseOutInner {
            calls: assemble_tool_calls(&result.tool_call_chunks)
                .into_iter()
                .map(|(name, arguments)| call_out(name, arguments))
                .collect(),
            normal_text: result.normal_text,
        });
    }

    let mut parser = create_tool_parser_for_family(family, tools)?;
    // B9: batch text fed through the shared push_input abstraction (always Text for
    // a complete batch sample) instead of the bare push() call.
    let mut result = parser.push_input(ToolParserInput::Text(text))?;
    result.append(parser.finish()?);
    Ok(CaseOutInner {
        calls: calls_from_parse_result(result.clone())
            .into_iter()
            .map(|(name, arguments)| call_out(name, arguments))
            .collect(),
        normal_text: result.normal_text,
    })
}

fn calls_from_parse_result(result: ToolParseResult) -> Vec<(String, String)> {
    let mut names = BTreeMap::<usize, String>::new();
    let mut args = BTreeMap::<usize, String>::new();
    for ToolCallDelta {
        tool_index,
        name,
        arguments,
    } in result.calls
    {
        if let Some(name) = name {
            names.entry(tool_index).or_default().push_str(&name);
        }
        args.entry(tool_index).or_default().push_str(&arguments);
    }
    names
        .into_iter()
        .map(|(idx, name)| (name, args.remove(&idx).unwrap_or_default()))
        .collect()
}

fn call_out(name: String, arguments: String) -> CallOut {
    let arguments = serde_json::from_str::<Value>(&arguments).unwrap_or(Value::String(arguments));
    CallOut { name, arguments }
}

#[derive(Serialize)]
struct CaseOut {
    #[serde(flatten)]
    result: CaseOutInner,
}

#[derive(Serialize)]
struct CaseOutInner {
    calls: Vec<CallOut>,
    normal_text: String,
}

#[derive(Serialize)]
struct CallOut {
    name: String,
    arguments: Value,
}
