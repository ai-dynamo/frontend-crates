// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Record the Dynamo v1 BATCH parser output over the batch fixture corpus, for
//! refreshing `fixtures-batch-v1/dynamo-<version>/` (`expected.dynamo`) after an
//! intentional v1 parser behavior change. Runs the same call the parity test and
//! dynamo's `parse_tool_calls_batch` PyO3 binding make:
//! `detect_and_parse_tool_call_with_recovery(text, family, tools)`.
//!
//! JSON in (one family per invocation; same YAML-free contract as
//! `record_dynamo_jail_stream` — the Python driver owns fixture I/O):
//!   {"family": "qwen25",
//!    "cases": {"TOOLCALLING.batch.8.a": {"model_text": "...",
//!                                        "tools": [{"name", "parameters"?, "strict"?}]}}}
//! JSON out (`arguments` decoded to a JSON value when possible, mirroring the
//! parity test's `decode_args`; malformed-body cases keep the raw string):
//!   {"TOOLCALLING.batch.8.a": {"calls": [{"name", "arguments"}], "normal_text": "..."}}
//! A case the parser errors on is omitted and the error goes to stderr.
//!
//! Usage: cargo run -p dynamo-parsers --bin record_dynamo_batch -- <input.json>

use std::collections::BTreeMap;

use dynamo_parsers::tool_calling::ToolDefinition;
use dynamo_parsers::tool_calling::parsers::detect_and_parse_tool_call_with_recovery;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Deserialize)]
struct Input {
    family: String,
    cases: BTreeMap<String, CaseIn>,
}

#[derive(Deserialize)]
struct CaseIn {
    model_text: String,
    #[serde(default)]
    tools: Vec<RawTool>,
}

// ToolDefinition does not derive Deserialize, so deserialize into this and build it.
#[derive(Deserialize)]
struct RawTool {
    name: String,
    #[serde(default)]
    parameters: Option<Value>,
    #[serde(default)]
    strict: Option<bool>,
}

#[derive(Serialize)]
struct CaseOut {
    calls: Vec<CallOut>,
    normal_text: String,
}

#[derive(Serialize)]
struct CallOut {
    name: String,
    arguments: Value,
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    let path = std::env::args()
        .nth(1)
        .ok_or_else(|| anyhow::anyhow!("usage: record_dynamo_batch <input.json>"))?;
    let input: Input = serde_json::from_str(&std::fs::read_to_string(&path)?)?;

    let mut out = BTreeMap::new();
    for (cid, case) in &input.cases {
        let tools: Vec<ToolDefinition> = case
            .tools
            .iter()
            .map(|t| ToolDefinition {
                name: t.name.clone(),
                parameters: t.parameters.clone(),
                strict: t.strict,
            })
            .collect();
        let tools_opt = (!tools.is_empty()).then_some(tools.as_slice());
        match detect_and_parse_tool_call_with_recovery(
            &case.model_text,
            Some(&input.family),
            tools_opt,
        )
        .await
        {
            Ok((calls, normal_text)) => {
                out.insert(
                    cid.clone(),
                    CaseOut {
                        calls: calls
                            .into_iter()
                            .map(|c| CallOut {
                                name: c.function.name.clone(),
                                arguments: serde_json::from_str(&c.function.arguments)
                                    .unwrap_or(Value::String(c.function.arguments)),
                            })
                            .collect(),
                        normal_text: normal_text.unwrap_or_default(),
                    },
                );
            }
            Err(e) => eprintln!("record_dynamo_batch: {} [{cid}]: {e}", input.family),
        }
    }
    println!("{}", serde_json::to_string_pretty(&out)?);
    Ok(())
}
