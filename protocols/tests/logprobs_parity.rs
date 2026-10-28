// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `dynamo_protocols::logprobs` against a copy of the Dynamo frontend's code
//! at dynamo `37f2f0a712`: `DeltaGenerator::create_logprobs`
//! (`lib/llm/src/protocols/openai/chat_completions/delta.rs:92-151`),
//! `convert_backend_top_logprobs` and `token_to_utf8_bytes`
//! (`lib/llm/src/protocols/openai.rs:290-344`). Each change from the stock
//! code is marked `ORACLE CHANGE`.

use dynamo_protocols::logprobs::{LogprobFormat, TopLogprobInput, chat_token_logprob};
use dynamo_protocols::types::{ChatChoiceLogprobs, ChatCompletionTokenLogprob, TopLogprobs};

/// ORACLE CHANGE: dynamo-llm's backend `TopLogprob`, with no `rank`, which
/// the conversion does not read.
#[derive(Clone, Debug)]
struct TopLogprob {
    token_id: u32,
    token: Option<String>,
    logprob: f64,
    bytes: Option<Vec<u8>>,
}

// lib/llm/src/protocols/openai.rs:290-296
fn token_to_utf8_bytes(token: &str) -> Option<Vec<u8>> {
    if token.is_empty() {
        None
    } else {
        Some(token.as_bytes().to_vec())
    }
}

// lib/llm/src/protocols/openai.rs:300-344
fn convert_backend_top_logprobs(
    top_lps: &[TopLogprob],
    selected_token: &str,
    selected_token_id: u32,
    selected_logprob: f32,
    return_tokens_as_token_ids: bool,
) -> Vec<TopLogprobs> {
    let mut found_selected = false;
    let mut result: Vec<TopLogprobs> = top_lps
        .iter()
        .map(|top_lp| {
            let tok = if return_tokens_as_token_ids {
                format!("token_id:{}", top_lp.token_id)
            } else {
                top_lp.token.clone().unwrap_or_default()
            };
            found_selected = found_selected || top_lp.token_id == selected_token_id;
            let bytes = if return_tokens_as_token_ids {
                token_to_utf8_bytes(&tok)
            } else {
                top_lp.bytes.clone().or_else(|| token_to_utf8_bytes(&tok))
            };
            TopLogprobs {
                token: tok,
                logprob: top_lp.logprob as f32,
                bytes,
            }
        })
        .collect();

    if !found_selected {
        let token = if return_tokens_as_token_ids {
            format!("token_id:{}", selected_token_id)
        } else {
            selected_token.to_string()
        };
        result.push(TopLogprobs {
            bytes: token_to_utf8_bytes(&token),
            token,
            logprob: selected_logprob,
        });
    }
    result
}

/// ORACLE CHANGE: the generator's options as arguments: `enable_logprobs`,
/// `return_tokens_as_token_ids`, and `suppress_top_logprobs`.
fn create_logprobs(
    enable_logprobs: bool,
    return_as_ids: bool,
    suppress_top_logprobs: bool,
    tokens: Vec<Option<String>>,
    token_ids: &[u32],
    logprobs: Option<Vec<f64>>,
    top_logprobs: Option<Vec<Vec<TopLogprob>>>,
) -> Option<ChatChoiceLogprobs> {
    if !enable_logprobs || logprobs.is_none() {
        return None;
    }

    let toks = tokens
        .into_iter()
        .zip(token_ids)
        .map(|(token, token_id)| (token.unwrap_or_default(), *token_id))
        .collect::<Vec<(String, u32)>>();
    let tok_lps = toks
        .iter()
        .zip(logprobs.unwrap())
        .map(|(_, lp)| lp as f32)
        .collect::<Vec<f32>>();

    let content = toks
        .iter()
        .zip(tok_lps)
        .enumerate()
        .map(|(index, ((t, tid), lp))| {
            let top_lps = top_logprobs
                .as_ref()
                .and_then(|positions| positions.get(index))
                .map(Vec::as_slice)
                .unwrap_or_default();
            let token_str = if return_as_ids {
                format!("token_id:{}", tid)
            } else {
                t.clone()
            };
            let converted = if suppress_top_logprobs {
                Vec::new()
            } else {
                convert_backend_top_logprobs(top_lps, t, *tid, lp, return_as_ids)
            };
            ChatCompletionTokenLogprob {
                token: token_str.clone(),
                logprob: lp,
                token_id: Some(*tid),
                bytes: token_to_utf8_bytes(&token_str),
                top_logprobs: converted,
            }
        })
        .collect();

    Some(ChatChoiceLogprobs {
        content: Some(content),
        refusal: None,
    })
}

fn candidate(token_id: u32, token: Option<&str>, logprob: f64, bytes: Option<&[u8]>) -> TopLogprob {
    TopLogprob {
        token_id,
        token: token.map(str::to_owned),
        logprob,
        bytes: bytes.map(<[u8]>::to_vec),
    }
}

/// Every token shape against every candidate list, in every format: one
/// entry each, as the stock frontend writes the entries of a backend output
/// of one token.
#[test]
fn every_entry_is_the_frontend_s() {
    let tokens = [Some("hello"), Some(""), None, Some("\u{fffd}"), Some("é")];
    let tops: Vec<Vec<TopLogprob>> = vec![
        vec![],
        vec![candidate(7, Some("hello"), -0.25, None)],
        vec![
            candidate(9, Some(" world"), -0.5, None),
            candidate(8, None, -1.5, None),
        ],
        vec![
            candidate(9, Some("x"), -0.5, Some(&[0xc3])),
            candidate(7, None, -0.75, None),
            candidate(8, Some(""), -2.0, None),
        ],
    ];
    let mut cases = 0;
    for token in tokens {
        for top in &tops {
            for tokens_as_ids in [false, true] {
                for no_top_logprobs in [false, true] {
                    let logprob = -0.75_f32;
                    let stock = create_logprobs(
                        true,
                        tokens_as_ids,
                        no_top_logprobs,
                        vec![token.map(str::to_owned)],
                        &[7],
                        Some(vec![f64::from(logprob)]),
                        Some(vec![top.clone()]),
                    )
                    .and_then(|logprobs| logprobs.content)
                    .expect("one entry");
                    let ours = chat_token_logprob(
                        token,
                        7,
                        logprob,
                        top.iter().map(|candidate| TopLogprobInput {
                            token_id: candidate.token_id,
                            token: candidate.token.as_deref(),
                            bytes: candidate.bytes.as_deref(),
                            logprob: candidate.logprob as f32,
                        }),
                        LogprobFormat {
                            tokens_as_ids,
                            no_top_logprobs,
                        },
                    );
                    assert_eq!(
                        vec![ours],
                        stock,
                        "{token:?}, {top:?}, as ids {tokens_as_ids}, none {no_top_logprobs}"
                    );
                    cases += 1;
                }
            }
        }
    }
    assert_eq!(cases, 80);
}

/// The control: a request that names no `top_logprobs` still gets the
/// chosen token as its one candidate, and `top_logprobs: 0` gets none.
#[test]
fn the_chosen_token_is_a_candidate_unless_none_are_asked_for() {
    let entry = |no_top_logprobs| {
        chat_token_logprob(
            Some("hi"),
            3,
            -0.5,
            [],
            LogprobFormat {
                tokens_as_ids: false,
                no_top_logprobs,
            },
        )
    };
    let with = entry(false);
    assert_eq!(with.top_logprobs.len(), 1);
    assert_eq!(with.top_logprobs[0].token, "hi");
    assert_eq!(with.top_logprobs[0].bytes.as_deref(), Some(&b"hi"[..]));
    assert!(entry(true).top_logprobs.is_empty());
}
