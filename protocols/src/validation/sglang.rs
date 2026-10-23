// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! SGLang contract: OpenAI reuse, shared engine operations, and SGLang overrides.
//!
//! Source: SGLang `5569b14d` [sampling checks](https://github.com/sgl-project/sglang/blob/5569b14d26ffd94a8326889eca15a153e8f1b68b/python/sglang/srt/sampling/sampling_params.py),
//! [request conversion](https://github.com/sgl-project/sglang/blob/5569b14d26ffd94a8326889eca15a153e8f1b68b/python/sglang/srt/entrypoints/openai/protocol.py),
//! and [verification](https://github.com/sgl-project/sglang/blob/5569b14d26ffd94a8326889eca15a153e8f1b68b/python/sglang/srt/managers/tokenizer_manager.py).
//! This pin does not establish later revisions or HTTP conformance.
//! The engine's pre-validation greedy rewrite is not implemented.

use super::{ValidationError, check};
use crate::types::{CreateChatCompletionRequest, CreateCompletionRequest};

pub use super::openai::{
    MAX_FREQUENCY_PENALTY, MAX_PRESENCE_PENALTY, MAX_TOP_P, MIN_FREQUENCY_PENALTY,
    MIN_PRESENCE_PENALTY, MIN_TEMPERATURE, validate_frequency_penalty, validate_presence_penalty,
};
// Same as vLLM; differs from OpenAI or is outside it.
pub use super::shared::{
    MAX_MIN_P, MIN_MIN_P, MIN_TOP_P_EXCLUSIVE, validate_min_p, validate_top_p,
};

/// Accepts `None` or a finite `temperature` >= 0. SGLang has no upper bound.
pub fn validate_temperature(value: Option<f32>) -> Result<(), ValidationError> {
    check(
        value,
        |v| v.is_finite() && v >= MIN_TEMPERATURE,
        "temperature",
        || format!("Temperature must be finite and at least {MIN_TEMPERATURE}"),
    )
}

/// Covered SGLang request checks.
///
/// Order: `temperature`, `top_p`, `frequency_penalty`, `presence_penalty`.
/// All other fields and cross-field rules are uncovered.
pub trait SglangValidation {
    /// Returns the first covered failure, without mutation.
    fn validate(&self) -> Result<(), ValidationError>;
}

impl SglangValidation for CreateChatCompletionRequest {
    fn validate(&self) -> Result<(), ValidationError> {
        validate_temperature(self.temperature)?;
        validate_top_p(self.top_p)?;
        validate_frequency_penalty(self.frequency_penalty)?;
        validate_presence_penalty(self.presence_penalty)
    }
}

impl SglangValidation for CreateCompletionRequest {
    fn validate(&self) -> Result<(), ValidationError> {
        validate_temperature(self.temperature)?;
        validate_top_p(self.top_p)?;
        validate_frequency_penalty(self.frequency_penalty)?;
        validate_presence_penalty(self.presence_penalty)
    }
}
