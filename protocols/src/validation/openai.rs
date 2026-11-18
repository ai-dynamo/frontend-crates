// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! OpenAI baseline contract.
//!
//! Sources: [Chat Completions](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create)
//! and [Completions](https://developers.openai.com/api/reference/resources/completions/methods/create),
//! read on 2026-10-21. These are documented bounds, not hosted-server probes.
//! All bounds are inclusive.

use super::{ValidationError, check};
use crate::types::{CreateChatCompletionRequest, CreateCompletionRequest};

/// Lower bound for `temperature`.
pub const MIN_TEMPERATURE: f32 = 0.0;
/// Upper bound for `temperature`.
pub const MAX_TEMPERATURE: f32 = 2.0;
/// Lower bound for `top_p`.
pub const MIN_TOP_P: f32 = 0.0;
/// Upper bound for `top_p`.
pub const MAX_TOP_P: f32 = 1.0;
/// Lower bound for `frequency_penalty`.
pub const MIN_FREQUENCY_PENALTY: f32 = -2.0;
/// Upper bound for `frequency_penalty`.
pub const MAX_FREQUENCY_PENALTY: f32 = 2.0;
/// Lower bound for `presence_penalty`.
pub const MIN_PRESENCE_PENALTY: f32 = -2.0;
/// Upper bound for `presence_penalty`.
pub const MAX_PRESENCE_PENALTY: f32 = 2.0;
/// Lower bound for the integer completion `logprobs`.
pub const MIN_LOGPROBS: u8 = 0;
/// Upper bound for the integer completion `logprobs`.
pub const MAX_LOGPROBS: u8 = 5;
/// Lower bound for chat `top_logprobs`.
pub const MIN_TOP_LOGPROBS: u8 = 0;
/// Upper bound for chat `top_logprobs`.
pub const MAX_TOP_LOGPROBS: u8 = 20;
/// Lower bound for `n`.
pub const MIN_N: u8 = 1;
/// Upper bound for `n`.
pub const MAX_N: u8 = 128;
/// Lower bound for completion `best_of`.
pub const MIN_BEST_OF: u8 = 0;
/// Upper bound for completion `best_of`.
pub const MAX_BEST_OF: u8 = 20;
/// Lower bound for one `logit_bias` value.
pub const MIN_LOGIT_BIAS: f32 = -100.0;
/// Upper bound for one `logit_bias` value.
pub const MAX_LOGIT_BIAS: f32 = 100.0;

/// Accepts `None` or `temperature` in [0, 2].
pub fn validate_temperature(value: Option<f32>) -> Result<(), ValidationError> {
    check(
        value,
        |v| (MIN_TEMPERATURE..=MAX_TEMPERATURE).contains(&v),
        "temperature",
        || format!("Temperature must be between {MIN_TEMPERATURE} and {MAX_TEMPERATURE}"),
    )
}

/// Accepts `None` or `top_p` in [0, 1].
pub fn validate_top_p(value: Option<f32>) -> Result<(), ValidationError> {
    check(
        value,
        |v| (MIN_TOP_P..=MAX_TOP_P).contains(&v),
        "top_p",
        || format!("Top_p must be between {MIN_TOP_P} and {MAX_TOP_P}"),
    )
}

/// Accepts `None` or `frequency_penalty` in [-2, 2].
pub fn validate_frequency_penalty(value: Option<f32>) -> Result<(), ValidationError> {
    check(
        value,
        |v| (MIN_FREQUENCY_PENALTY..=MAX_FREQUENCY_PENALTY).contains(&v),
        "frequency_penalty",
        || {
            format!(
                "Frequency penalty must be between {MIN_FREQUENCY_PENALTY} and {MAX_FREQUENCY_PENALTY}"
            )
        },
    )
}

/// Accepts `None` or `presence_penalty` in [-2, 2].
pub fn validate_presence_penalty(value: Option<f32>) -> Result<(), ValidationError> {
    check(
        value,
        |v| (MIN_PRESENCE_PENALTY..=MAX_PRESENCE_PENALTY).contains(&v),
        "presence_penalty",
        || {
            format!(
                "Presence penalty must be between {MIN_PRESENCE_PENALTY} and {MAX_PRESENCE_PENALTY}"
            )
        },
    )
}

/// Accepts `None` or the integer `/v1/completions` `logprobs` in 0..=5.
///
/// This is not chat's boolean `logprobs` field.
pub fn validate_completion_logprobs(value: Option<u8>) -> Result<(), ValidationError> {
    check(
        value,
        |v| (MIN_LOGPROBS..=MAX_LOGPROBS).contains(&v),
        "logprobs",
        || format!("Logprobs must be between {MIN_LOGPROBS} and {MAX_LOGPROBS}"),
    )
}

/// Accepts `None` or chat `top_logprobs` in 0..=20.
///
/// The relationship to chat's boolean `logprobs` field is not checked.
pub fn validate_top_logprobs(value: Option<u8>) -> Result<(), ValidationError> {
    check(
        value,
        |v| (MIN_TOP_LOGPROBS..=MAX_TOP_LOGPROBS).contains(&v),
        "top_logprobs",
        || format!("Top_logprobs must be between {MIN_TOP_LOGPROBS} and {MAX_TOP_LOGPROBS}"),
    )
}

/// Accepts `None` or `n` in 1..=128.
///
/// Batch totals and sampling interactions are not checked.
pub fn validate_n(value: Option<u8>) -> Result<(), ValidationError> {
    check(
        value,
        |v| (MIN_N..=MAX_N).contains(&v),
        "n",
        || format!("n must be between {MIN_N} and {MAX_N}"),
    )
}

/// Accepts `None` or completion `best_of` in 0..=20.
///
/// Checks only the count. Relationships to `n` and streaming are not checked.
pub fn validate_best_of(value: Option<u8>) -> Result<(), ValidationError> {
    check(
        value,
        |v| (MIN_BEST_OF..=MAX_BEST_OF).contains(&v),
        "best_of",
        || format!("Best_of must be between {MIN_BEST_OF} and {MAX_BEST_OF}"),
    )
}

/// Accepts one parsed `logit_bias` value in [-100, 100], without clamping.
///
/// Unlike the other helpers, this takes one map entry's value, not an `Option`.
/// The caller owns map traversal, token-ID policy, JSON number admission, and
/// precision conversion. The error value is this `f32`'s `Display`, not the JSON text.
pub fn validate_logit_bias_value(value: f32) -> Result<(), ValidationError> {
    check(
        Some(value),
        |v| (MIN_LOGIT_BIAS..=MAX_LOGIT_BIAS).contains(&v),
        "logit_bias",
        || format!("Logit bias must be between {MIN_LOGIT_BIAS} and {MAX_LOGIT_BIAS}"),
    )
}

/// Covered OpenAI request checks.
///
/// Order: `temperature`, `top_p`, `frequency_penalty`, `presence_penalty`,
/// completion `logprobs` (completions only), `n`, then chat `top_logprobs` or
/// completion `best_of`. All other fields and cross-field rules are uncovered.
pub trait OpenAIValidation {
    /// Returns the first covered failure, without mutation.
    fn validate(&self) -> Result<(), ValidationError>;
}

impl OpenAIValidation for CreateChatCompletionRequest {
    fn validate(&self) -> Result<(), ValidationError> {
        validate_temperature(self.temperature)?;
        validate_top_p(self.top_p)?;
        validate_frequency_penalty(self.frequency_penalty)?;
        validate_presence_penalty(self.presence_penalty)?;
        validate_n(self.n)?;
        validate_top_logprobs(self.top_logprobs)
    }
}

impl OpenAIValidation for CreateCompletionRequest {
    fn validate(&self) -> Result<(), ValidationError> {
        validate_temperature(self.temperature)?;
        validate_top_p(self.top_p)?;
        validate_frequency_penalty(self.frequency_penalty)?;
        validate_presence_penalty(self.presence_penalty)?;
        validate_completion_logprobs(self.logprobs)?;
        validate_n(self.n)?;
        validate_best_of(self.best_of)
    }
}
