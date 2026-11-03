// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! vLLM contract: OpenAI reuse, shared engine operations, and vLLM overrides.
//!
//! Source: vLLM v0.28.0 [sampling checks](https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/sampling_params.py)
//! and [model limits](https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/config/model.py).
//! This pin does not establish later revisions or HTTP conformance.
//! Tiny-temperature flooring and greedy rewrites are not implemented.

use super::{ValidationError, check};
use crate::types::{CreateChatCompletionRequest, CreateCompletionRequest};

pub use super::openai::{
    MAX_FREQUENCY_PENALTY, MAX_PRESENCE_PENALTY, MAX_TEMPERATURE, MAX_TOP_P, MIN_FREQUENCY_PENALTY,
    MIN_LOGPROBS, MIN_PRESENCE_PENALTY, MIN_TEMPERATURE, validate_frequency_penalty,
    validate_presence_penalty, validate_temperature,
};
// Same as SGLang; differs from OpenAI or is outside it.
pub use super::shared::{
    MAX_MIN_P, MIN_MIN_P, MIN_TOP_P_EXCLUSIVE, validate_min_p, validate_top_p,
};

/// Fixed library upper bound for completion `logprobs`.
pub const MAX_LOGPROBS: u8 = 20;

/// Accepts `None` or the integer `/v1/completions` `logprobs` in 0..=20.
///
/// This is a fixed library policy chosen by DEP #16992; it needs no model context.
/// Actual vLLM checks against the configurable `max_logprobs` (default 20) and
/// also accepts -1 for the full vocabulary, which the `u8` request field cannot hold.
pub fn validate_completion_logprobs(value: Option<u8>) -> Result<(), ValidationError> {
    check(
        value,
        |v| (MIN_LOGPROBS..=MAX_LOGPROBS).contains(&v),
        "logprobs",
        || format!("Logprobs must be between {MIN_LOGPROBS} and {MAX_LOGPROBS}"),
    )
}

/// Covered vLLM request checks.
///
/// Order: `temperature`, `top_p`, `frequency_penalty`, `presence_penalty`, then
/// completion `logprobs` (completions only). All other fields and cross-field
/// rules are uncovered.
pub trait VllmValidation {
    /// Returns the first covered failure, without mutation.
    fn validate(&self) -> Result<(), ValidationError>;
}

impl VllmValidation for CreateChatCompletionRequest {
    fn validate(&self) -> Result<(), ValidationError> {
        validate_temperature(self.temperature)?;
        validate_top_p(self.top_p)?;
        validate_frequency_penalty(self.frequency_penalty)?;
        validate_presence_penalty(self.presence_penalty)
    }
}

impl VllmValidation for CreateCompletionRequest {
    fn validate(&self) -> Result<(), ValidationError> {
        validate_temperature(self.temperature)?;
        validate_top_p(self.top_p)?;
        validate_frequency_penalty(self.frequency_penalty)?;
        validate_presence_penalty(self.presence_penalty)?;
        validate_completion_logprobs(self.logprobs)
    }
}
