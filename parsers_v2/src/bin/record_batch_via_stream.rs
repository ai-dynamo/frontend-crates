// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Record the harmony STREAM parser's result on every harmony BATCH sample, so the
//! parity generator (Python, can't run the Rust parser) can render the
//! "Stream parser on batch" tab. Feeds each batch fixture's full `model_text`
//! through the streaming parser (text path) + finish, and emits the assembled
//! calls per case as JSON.
//!
//! Output: {case_id: {"calls": [{"name", "arguments": <json|string>}, ...]}}
//! This is the DYNAMO layer only. Merge it with the per-engine vLLM/SGLang captures
//! (conformance/utils/capture_harmony_batch_stream.py) via merge_batch_stream.py to
//! produce the nested conformance/utils/harmony_batch_stream.json the table consumes.
//! Usage:
//!   cargo run -p dynamo-parsers-v2 --bin record_batch_via_stream \
//!     > /tmp/dynamo_batch_stream.json   # then: see merge_batch_stream.py

use std::collections::BTreeMap;
use std::path::PathBuf;

use dynamo_parsers_v2::{HarmonyToolStreamParser, assemble_tool_calls};
use serde::Deserialize;

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
}

fn main() -> anyhow::Result<()> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("conformance/toolcalling/fixtures/harmony");

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

    let mut entries: Vec<String> = Vec::new();
    for path in &files {
        let fx: Fixture = serde_yaml::from_str(&std::fs::read_to_string(path)?)?;
        if fx.family != "harmony" || fx.mode != "batch" {
            continue;
        }
        for (cid, case) in &fx.cases {
            let Some(text) = case.model_text.as_ref() else {
                continue;
            };
            let mut parser = HarmonyToolStreamParser::new()?;
            let mut all = parser.parse_tool_call_streaming_text(text).tool_call_chunks;
            all.extend(parser.finish_tool_call_stream().tool_call_chunks);
            let calls: Vec<String> = assemble_tool_calls(&all)
                .iter()
                .map(|(name, args)| {
                    // arguments: emit parsed JSON when valid, else the raw string.
                    let args_json = serde_json::from_str::<serde_json::Value>(args)
                        .unwrap_or_else(|_| serde_json::Value::String(args.clone()));
                    format!(
                        "{{\"name\": {}, \"arguments\": {}}}",
                        serde_json::to_string(name).unwrap(),
                        serde_json::to_string(&args_json).unwrap()
                    )
                })
                .collect();
            entries.push(format!(
                "  {}: {{\"calls\": [{}]}}",
                serde_json::to_string(cid).unwrap(),
                calls.join(", ")
            ));
        }
    }
    println!("{{\n{}\n}}", entries.join(",\n"));
    Ok(())
}
