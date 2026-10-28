// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The stop-token rules of the Dynamo frontend's decoder, over a
//! [`DecodeStream`], without its stop strings.
//!
//! From token `min_tokens` on:
//! - a hidden stop token ends the reply, flushes the stream, and adds no
//!   text of its own;
//! - a visible one, such as a tool-call end token that is also an
//!   end-of-sequence token, decodes, flushes the stream, and ends the reply.
//!
//! Before `min_tokens`, a stop token is ordinary text. A token both hidden
//! and visible is visible. With `no_stop_trim`, a hidden stop token decodes
//! as a visible one does. Once the reply ends, later tokens add no text: the
//! frontend drops them.
//!
//! The frontend also decodes each token on its own, for its log
//! probabilities. That costs a decode per token, so a decoder does it only
//! when built [`with_token_text`](StopTokenDecoder::with_token_text).
//! [`lossless_token_text`] is the text the frontend gives a top candidate.

use crate::{DecodeStream, Result, TokenIdType, traits};

/// The stop-token IDs of one request, as the frontend's stop conditions
/// carry them.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct StopTokenIds {
    /// The request's own stop token IDs (`stop_token_ids`). Hidden.
    pub user: Vec<TokenIdType>,
    /// The model's, such as its end-of-sequence IDs. Hidden.
    pub hidden: Vec<TokenIdType>,
    /// Stop tokens whose text the reply keeps.
    pub visible: Vec<TokenIdType>,
}

/// The stop token that ended a reply.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopTokenTrigger {
    /// A visible stop token.
    Visible(TokenIdType),
    /// One of the request's own stop tokens.
    User(TokenIdType),
    /// A hidden stop token that is not the request's.
    Hidden(TokenIdType),
}

/// What one token adds to a reply.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct StopTokenStep {
    /// The text the token completes, with the stream's held text when the
    /// token ends the reply.
    pub text: Option<String>,
    /// The stop token, when this token ends the reply.
    pub stop: Option<StopTokenTrigger>,
    /// The token's own text, for its log probability: [`token_text`], or
    /// the stream's text when that fails. `None` for a hidden stop token
    /// that ends the reply, and when the decoder is not built
    /// [`with_token_text`](StopTokenDecoder::with_token_text).
    pub token: Option<String>,
}

/// A [`DecodeStream`] with the frontend decoder's stop-token rules.
pub struct StopTokenDecoder {
    stream: DecodeStream,
    user: Vec<TokenIdType>,
    hidden: Vec<TokenIdType>,
    visible: Vec<TokenIdType>,
    min_tokens: u32,
    no_stop_trim: bool,
    token_text: bool,
    generated: u32,
    stopped: bool,
}

impl StopTokenDecoder {
    /// A decoder over `stream` that stops on `ids` from token `min_tokens` on.
    pub fn new(
        stream: DecodeStream,
        ids: StopTokenIds,
        min_tokens: u32,
        no_stop_trim: bool,
    ) -> Self {
        let StopTokenIds {
            user,
            hidden,
            visible,
        } = ids;
        let mut all_hidden = user.clone();
        for id in hidden {
            if !all_hidden.contains(&id) {
                all_hidden.push(id);
            }
        }
        Self {
            stream,
            user,
            hidden: all_hidden,
            visible,
            min_tokens,
            no_stop_trim,
            token_text: false,
            generated: 0,
            stopped: false,
        }
    }

    /// The same decoder, which also gives each token's own text
    /// ([`StopTokenStep::token`]): one more decode per token.
    pub fn with_token_text(mut self) -> Self {
        self.token_text = true;
        self
    }

    /// Decode the next generated token.
    ///
    /// # Errors
    ///
    /// When the stream cannot decode. The stream may hold the token already,
    /// so the reply cannot go on.
    pub fn step(&mut self, id: TokenIdType) -> Result<StopTokenStep> {
        if self.stopped {
            return Ok(StopTokenStep::default());
        }
        let below_min = self.generated < self.min_tokens;
        self.generated += 1;
        let hidden = self.hidden.contains(&id);
        let visible = self.visible.contains(&id);
        if !below_min && hidden && !visible && !self.no_stop_trim {
            self.stopped = true;
            let trigger = if self.user.contains(&id) {
                StopTokenTrigger::User(id)
            } else {
                StopTokenTrigger::Hidden(id)
            };
            return Ok(StopTokenStep {
                text: self.stream.finish()?,
                stop: Some(trigger),
                token: None,
            });
        }
        let mut text = self.stream.step(id)?;
        // Decoded alone: the stream's text can hold earlier tokens' bytes.
        let token = self
            .token_text
            .then(|| {
                token_text(
                    self.stream.tokenizer().as_ref(),
                    id,
                    self.stream.skip_special_tokens(),
                )
                .or_else(|| text.clone())
            })
            .flatten();
        if below_min || !(hidden || visible) {
            return Ok(StopTokenStep {
                text,
                stop: None,
                token,
            });
        }
        self.stopped = true;
        if let Some(tail) = self.stream.finish()? {
            text.get_or_insert_default().push_str(&tail);
        }
        let trigger = if visible {
            StopTokenTrigger::Visible(id)
        } else if self.user.contains(&id) {
            StopTokenTrigger::User(id)
        } else {
            StopTokenTrigger::Hidden(id)
        };
        Ok(StopTokenStep {
            text,
            stop: Some(trigger),
            token,
        })
    }

    /// The reply ended without a stop token: the text the stream still
    /// holds.
    ///
    /// # Errors
    ///
    /// When the stream cannot decode.
    pub fn finish(&mut self) -> Result<Option<String>> {
        if self.stopped {
            return Ok(None);
        }
        self.stopped = true;
        self.stream.finish()
    }

    /// The tokens stepped so far, until the reply ended.
    pub fn generated_tokens(&self) -> u32 {
        self.generated
    }

    /// Whether a stop token, or [`finish`](Self::finish), ended the reply.
    pub fn is_stopped(&self) -> bool {
        self.stopped
    }
}

/// The text of token `id` decoded alone, as the frontend writes the chosen
/// token in a log probability. A byte of a longer character decodes to
/// U+FFFD, which is kept: the tokenizer gives no way to its own bytes.
/// `None` when the decode fails.
pub fn token_text(
    tokenizer: &dyn traits::Tokenizer,
    id: TokenIdType,
    skip_special_tokens: bool,
) -> Option<String> {
    tokenizer
        .decode(&[id], skip_special_tokens)
        .ok()
        .map(String::from)
}

/// The text of token `id` decoded alone, as the frontend writes a top
/// candidate in a log probability: `None` when the decode fails, or is not
/// whole text (it ends in a partial character or holds U+FFFD).
pub fn lossless_token_text(
    tokenizer: &dyn traits::Tokenizer,
    id: TokenIdType,
    skip_special_tokens: bool,
) -> Option<String> {
    let decoded = tokenizer.decode(&[id], skip_special_tokens).ok()?;
    (decoded.is_complete() && !decoded.as_str().contains('\u{fffd}')).then(|| decoded.into())
}
