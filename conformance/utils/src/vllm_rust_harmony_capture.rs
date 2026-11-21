// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;
use std::io::Read;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::{Result, anyhow, ensure};
use futures::{StreamExt, executor::block_on, stream};
use openai_harmony::{HarmonyEncoding, HarmonyEncodingName, load_harmony_encoding};
use serde::Deserialize;
use serde_json::{Value, json};
use vllm_chat::{
    AssistantBlockKind, AssistantContentBlock, ChatEvent, ChatOutputProcessor, ChatRequest,
    ChatTool, ChatToolChoice, HarmonyChatOutputProcessor, ResolvedToolContext,
};
use vllm_llm::FinishReason;
use vllm_text::{DecodedText, DecodedTextEvent, Finished, SampledDelta};

#[derive(Deserialize)]
struct Job {
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    id: String,
    input: String,
    chunks: Vec<String>,
    tools: Vec<ChatTool>,
    terminal_step: bool,
    finish_reason: String,
}

fn token_chunks(encoding: &HarmonyEncoding, case: &Case) -> Result<Vec<Vec<u32>>> {
    let tokenizer = encoding.tokenizer();
    let mut tokens = tokenizer
        .encode_with_special_tokens(&case.input)
        .into_iter()
        .peekable();
    let mut offset = 0;
    let mut boundary = 0;
    let mut rows = Vec::new();
    let mut decoded = Vec::new();
    for chunk in &case.chunks {
        boundary += chunk.len();
        let mut row = Vec::new();
        while let Some(id) = tokens.peek().copied() {
            let bytes = tokenizer.decode_bytes([id])?;
            if offset + bytes.len() > boundary {
                break;
            }
            offset += bytes.len();
            decoded.extend(bytes);
            row.push(tokens.next().expect("peeked token"));
        }
        rows.push(row);
    }
    ensure!(
        tokens.next().is_none() && decoded == case.input.as_bytes(),
        "token projection differs from the authored input"
    );
    if case.terminal_step {
        rows.push(Vec::new());
    }
    ensure!(!rows.is_empty(), "the job requires a terminal chunk");
    Ok(rows)
}

async fn capture(encoding: &HarmonyEncoding, case: &Case) -> Result<Value> {
    let mut request = ChatRequest::for_test();
    request.tool_context = ResolvedToolContext::new(
        &[],
        case.tools.clone(),
        (!case.tools.is_empty()).then_some(ChatToolChoice::Auto),
        true,
    )?;
    let processor = Box::new(HarmonyChatOutputProcessor::new(&request)?);
    let token_chunks = token_chunks(encoding, case)?;
    let mut chunks = vec![Vec::new(); token_chunks.len()];
    let final_index = chunks.len() - 1;
    let updates = std::iter::once((
        0,
        DecodedTextEvent::Start {
            prompt_token_ids: Arc::from([]),
            prompt_logprobs: None,
        },
    ))
    .chain(
        token_chunks
            .into_iter()
            .enumerate()
            .map(move |(index, token_ids)| {
                let finished = (index == final_index).then(|| {
                    Box::new(Finished {
                        usage: Default::default(),
                        finish_reason: if case.finish_reason == "length" {
                            FinishReason::Length
                        } else {
                            FinishReason::stop_eos()
                        },
                        kv_transfer_params: None,
                        ec_transfer_params: None,
                        sampling_mask: None,
                        spec_decode_metrics: None,
                        weight_versions: None,
                        timestamps: Default::default(),
                    })
                });
                (
                    index,
                    DecodedTextEvent::TextDelta {
                        decoded: DecodedText::default(),
                        sampled: SampledDelta {
                            token_ids,
                            logprobs: None,
                        },
                        finished,
                    },
                )
            })
            .collect::<Vec<_>>(),
    );
    let cursor = Arc::new(AtomicUsize::new(0));
    let cursor_in = cursor.clone();
    let input = stream::iter(updates).map(move |(index, update)| {
        cursor_in.store(index, Ordering::Relaxed);
        Ok(update)
    });
    let mut output = processor.process(Box::pin(input))?;
    let mut terminal = None;
    while let Some(event) = output.next().await {
        let event = match event {
            Ok(event) => event,
            Err(error) => {
                return Ok(
                    json!({"error": format!("HarmonyOutputParsing: {error}"), "partial_chunks": chunks}),
                );
            }
        };
        let chunk = &mut chunks[cursor.load(Ordering::Relaxed)];
        match event {
            ChatEvent::BlockDelta {
                kind: AssistantBlockKind::Text,
                delta,
                ..
            } => chunk.push(json!({"kind": "text", "text": delta})),
            ChatEvent::BlockDelta {
                kind: AssistantBlockKind::Reasoning,
                delta,
                ..
            } => chunk.push(json!({"kind": "reasoning", "text": delta})),
            ChatEvent::ToolCallStart { name, .. } => {
                chunk.push(json!({"kind": "tool_call", "name": name, "arguments": ""}))
            }
            ChatEvent::ToolCallArgumentsDelta { delta, .. } => {
                chunk.push(json!({"kind": "tool_call", "name": null, "arguments": delta}))
            }
            ChatEvent::Done { message, .. } => terminal = Some(message),
            _ => {}
        }
    }
    let message =
        terminal.ok_or_else(|| anyhow!("Harmony processor omitted the terminal message"))?;
    let assembled: Vec<_> = message
        .content
        .into_iter()
        .map(|block| match block {
            AssistantContentBlock::Text { text } => json!({"kind": "text", "text": text}),
            AssistantContentBlock::Reasoning { text } => json!({"kind": "reasoning", "text": text}),
            AssistantContentBlock::ToolCall(call) => {
                let args =
                    serde_json::from_str(&call.arguments).unwrap_or(Value::String(call.arguments));
                json!({"kind": "tool_call", "name": call.name, "arguments": args})
            }
        })
        .collect();
    Ok(
        json!({"assembled": assembled, "chunks": chunks, "parser": "vLLM Rust (HarmonyChatOutputProcessor)"}),
    )
}

fn main() -> Result<()> {
    let mut input = String::new();
    std::io::stdin().read_to_string(&mut input)?;
    let job: Job = serde_json::from_str(&input)?;
    let encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss)?;
    let mut results = BTreeMap::new();
    for case in job.cases {
        results.insert(case.id.clone(), block_on(capture(&encoding, &case))?);
    }
    println!("{}", json!({"results": results}));
    Ok(())
}
