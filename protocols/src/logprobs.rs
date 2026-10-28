// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The log probability entries of a chat reply, as the Dynamo frontend
//! writes them (`DeltaGenerator::create_logprobs`).
//!
//! The frontend writes one entry per generated token. The token is its text
//! decoded alone (`dynamo_tokenizers::token_text`), or `""` when it has none,
//! such as a hidden stop token. The top candidates come from the backend,
//! and the chosen token is added to them when the backend left it out, so
//! even a request that names no `top_logprobs` gets one candidate. Only a
//! request with `top_logprobs: 0` gets none.

use crate::types::{ChatCompletionTokenLogprob, TopLogprobs};

/// A top candidate as the backend reports it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TopLogprobInput<'a> {
    /// The candidate's token ID.
    pub token_id: u32,
    /// Its text, or `None` when neither the backend nor the decode gives
    /// whole text (`dynamo_tokenizers::lossless_token_text`).
    pub token: Option<&'a str>,
    /// Its bytes, when the backend reports them.
    pub bytes: Option<&'a [u8]>,
    /// Its log probability.
    pub logprob: f32,
}

/// How the entries of one request are written.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LogprobFormat {
    /// Write each token as `token_id:<id>` (`return_tokens_as_token_ids`).
    pub tokens_as_ids: bool,
    /// Write no top candidates: the request set `top_logprobs` to 0.
    pub no_top_logprobs: bool,
}

/// The entry of one generated token: `token` is its own text, `token_id` its
/// ID, `logprob` its log probability, and `top` the backend's candidates.
pub fn chat_token_logprob<'a>(
    token: Option<&str>,
    token_id: u32,
    logprob: f32,
    top: impl IntoIterator<Item = TopLogprobInput<'a>>,
    format: LogprobFormat,
) -> ChatCompletionTokenLogprob {
    let text = token.unwrap_or_default();
    let token = if format.tokens_as_ids {
        format!("token_id:{token_id}")
    } else {
        text.to_owned()
    };
    let top_logprobs = if format.no_top_logprobs {
        Vec::new()
    } else {
        top_logprobs(top, text, token_id, logprob, format.tokens_as_ids)
    };
    ChatCompletionTokenLogprob {
        bytes: token_bytes(&token),
        token,
        logprob,
        token_id: Some(token_id),
        top_logprobs,
    }
}

/// The top candidates of a token: the backend's, then the chosen token when
/// they leave it out.
pub fn top_logprobs<'a>(
    top: impl IntoIterator<Item = TopLogprobInput<'a>>,
    selected_token: &str,
    selected_token_id: u32,
    selected_logprob: f32,
    tokens_as_ids: bool,
) -> Vec<TopLogprobs> {
    let mut found_selected = false;
    let mut result: Vec<TopLogprobs> = top
        .into_iter()
        .map(|candidate| {
            found_selected = found_selected || candidate.token_id == selected_token_id;
            let token = if tokens_as_ids {
                format!("token_id:{}", candidate.token_id)
            } else {
                candidate.token.unwrap_or_default().to_owned()
            };
            let bytes = if tokens_as_ids {
                token_bytes(&token)
            } else {
                candidate
                    .bytes
                    .map(<[u8]>::to_vec)
                    .or_else(|| token_bytes(&token))
            };
            TopLogprobs {
                token,
                logprob: candidate.logprob,
                bytes,
            }
        })
        .collect();
    if !found_selected {
        let token = if tokens_as_ids {
            format!("token_id:{selected_token_id}")
        } else {
            selected_token.to_owned()
        };
        result.push(TopLogprobs {
            bytes: token_bytes(&token),
            token,
            logprob: selected_logprob,
        });
    }
    result
}

/// The bytes of a token's text: `None` for no text.
pub fn token_bytes(token: &str) -> Option<Vec<u8>> {
    (!token.is_empty()).then(|| token.as_bytes().to_vec())
}
