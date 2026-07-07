// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Record the Dynamo v1 streaming tool-call JAIL output over the streamv2 chunk
//! corpus, for the "Dynamo Rust v1 3.0.0 (stream)" conformance candidate.
//!
//! The v1 streaming path has no token-incremental parser; instead it buffers
//! ("jails") the model's output once a tool-call start looks imminent, then runs
//! the v1 batch parser on the released block. That machinery is `JailedStream`
//! (dynamo-parsers, moved here by DIS-2296). This recorder drives it over the same
//! per-chunk `delta_text` inputs the v2 stream tab uses.
//!
//! JSON in (one family per invocation):
//!   {"family": "hermes", "cases": {"TOOLCALLING.streamv2.1.a": ["delta1", "delta2", ...]}}
//! JSON out (per output chunk the jail emits — it coalesces, so the count differs
//! from the input; downstream assembles by concatenating per index):
//!   {"TOOLCALLING.streamv2.1.a": [{"deltas": [{"index", "id"?, "name"?, "arguments"?}], "normal_text"}]}
//!
//! Usage: cargo run -p dynamo-parsers --bin record_dynamo_jail_stream -- <input.json>

use std::collections::BTreeMap;

use dynamo_parsers::tool_calling::jail::{Annotated, JailedStream};
use dynamo_protocols::types::{
    ChatChoiceStream, ChatCompletionMessageContent, ChatCompletionStreamResponseDelta,
    CreateChatCompletionStreamResponse, FinishReason, Role,
};
use futures::StreamExt;
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Input {
    family: String,
    /// case id -> the per-chunk `delta_text` stream (same inputs as the v2 stream tab).
    cases: BTreeMap<String, Vec<String>>,
}

#[derive(Serialize)]
struct DeltaEmit {
    index: u32,
    #[serde(skip_serializing_if = "is_false")]
    id: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    arguments: Option<String>,
}

#[derive(Serialize)]
struct ChunkEmit {
    deltas: Vec<DeltaEmit>,
    normal_text: String,
}

fn is_false(b: &bool) -> bool {
    !*b
}

/// Build one input stream chunk carrying `content` as assistant delta text; a
/// `finish` chunk (content=None, finish_reason=Stop) closes the stream so the jail
/// flushes any accumulated block.
fn mock_chunk(
    content: Option<String>,
    finish: bool,
) -> Annotated<CreateChatCompletionStreamResponse> {
    #[allow(deprecated)]
    let choice = ChatChoiceStream {
        index: 0,
        delta: ChatCompletionStreamResponseDelta {
            role: Some(Role::Assistant),
            content: content.map(ChatCompletionMessageContent::Text),
            tool_calls: None,
            function_call: None,
            refusal: None,
            reasoning_content: None,
        },
        finish_reason: if finish {
            Some(FinishReason::Stop)
        } else {
            None
        },
        logprobs: None,
    };
    let response = CreateChatCompletionStreamResponse {
        id: "rec".to_string(),
        choices: vec![choice],
        created: 0,
        model: "rec".to_string(),
        system_fingerprint: None,
        object: "chat.completion.chunk".to_string(),
        usage: None,
        service_tier: None,
    };
    Annotated {
        data: Some(response),
        id: None,
        event: None,
        comment: None,
        error: None,
    }
}

async fn record_case(family: &str, chunks: &[String]) -> Vec<ChunkEmit> {
    let mut inputs: Vec<_> = chunks
        .iter()
        .map(|t| mock_chunk(Some(t.clone()), false))
        .collect();
    inputs.push(mock_chunk(None, true));

    let jail = JailedStream::builder().tool_call_parser(family).build();
    let out: Vec<Annotated<CreateChatCompletionStreamResponse>> =
        jail.apply(futures::stream::iter(inputs)).collect().await;

    let mut per_chunk = Vec::new();
    for a in out {
        let Some(resp) = a.data else { continue };
        let Some(choice) = resp.choices.into_iter().next() else {
            continue;
        };
        let normal_text = match choice.delta.content.as_ref() {
            Some(ChatCompletionMessageContent::Text(t)) => t.clone(),
            _ => String::new(),
        };
        let deltas = choice
            .delta
            .tool_calls
            .unwrap_or_default()
            .into_iter()
            .map(|tc| DeltaEmit {
                index: tc.index,
                id: tc.id.is_some(),
                name: tc.function.as_ref().and_then(|f| f.name.clone()),
                arguments: tc.function.as_ref().and_then(|f| f.arguments.clone()),
            })
            .collect();
        per_chunk.push(ChunkEmit {
            deltas,
            normal_text,
        });
    }
    per_chunk
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let path = std::env::args()
        .nth(1)
        .ok_or_else(|| anyhow::anyhow!("usage: record_dynamo_jail_stream <input.json>"))?;
    let input: Input = serde_json::from_str(&std::fs::read_to_string(&path)?)?;

    let mut out: BTreeMap<String, Vec<ChunkEmit>> = BTreeMap::new();
    for (cid, chunks) in &input.cases {
        out.insert(cid.clone(), record_case(&input.family, chunks).await);
    }
    println!("{}", serde_json::to_string_pretty(&out)?);
    Ok(())
}
