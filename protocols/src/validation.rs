// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Opt-in, partial validation of parsed protocol values.
//!
//! Select one contract namespace: [`openai`], [`vllm`], or [`sglang`]. Each
//! namespace exposes field helpers and a request trait for the covered request
//! types. The engine namespaces reuse the OpenAI operation where the engine
//! matches it and define an override where it differs. Engine contracts are not
//! supersets of OpenAI, so do not run the OpenAI checks before an engine's checks.
//!
//! All operations in this module follow the same contract:
//!
//! - They are read-only. They never mutate, normalize, insert defaults, or dispatch.
//! - Field helpers are named `validate_<field>` and take the request field's
//!   parsed type. `None` (omitted or null at the typed boundary) always succeeds.
//! - Float helpers reject NaN and infinities. They check parsed `f32` values and do
//!   not reproduce JSON coercion, null admission, or backend HTTP diagnostics.
//! - Request traits run their covered checks in a documented order and return the
//!   first failure. Success does not establish complete endpoint admission.
//!
//! Coverage and known gaps are tracked in the temporary
//! [coverage tracker](https://github.com/ai-dynamo/frontend-crates/blob/main/protocols/src/validation/coverage.md).
//!
//! ```rust
//! use dynamo_protocols::{
//!     normalization,
//!     types::CreateCompletionRequest,
//!     validation::{openai::OpenAIValidation, vllm::VllmValidation},
//! };
//!
//! let mut request = CreateCompletionRequest {
//!     logprobs: Some(10),
//!     ..Default::default()
//! };
//! normalization::vllm::normalize_completion_logprobs(&mut request.logprobs);
//! // vLLM accepts 10 completion logprobs; the OpenAI baseline allows at most 5.
//! assert!(VllmValidation::validate(&request).is_ok());
//! assert!(OpenAIValidation::validate(&request).is_err());
//! assert_eq!(request.logprobs, Some(10));
//! ```
//!
//! A server still owns prompt, model, runtime, and other admission checks.

pub mod openai;
pub mod sglang;
pub mod vllm;

// Operations that vLLM and SGLang implement equivalently outside the OpenAI
// contract. Private: consumers reach them through an engine namespace.
mod shared;

/// A library diagnostic for the first failed covered check.
///
/// `Display` renders `"{message}, got {value}"`. The error carries no HTTP status
/// or backend exception metadata; consumers select their own transport mapping.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{message}, got {value}")]
pub struct ValidationError {
    /// The request field whose check failed.
    pub field: &'static str,
    /// Field label and constraint, without the rejected value.
    pub message: String,
    /// Rust `Display` of the rejected value, for example `NaN` or `-0`.
    /// This is not the original JSON text.
    pub value: String,
}

/// Accepts `None` or a value for which `accepted` returns true.
/// Builds the diagnostic only on failure.
fn check<T: Copy + std::fmt::Display>(
    value: Option<T>,
    accepted: impl FnOnce(T) -> bool,
    field: &'static str,
    message: impl FnOnce() -> String,
) -> Result<(), ValidationError> {
    match value {
        Some(value) if !accepted(value) => Err(ValidationError {
            field,
            message: message(),
            value: value.to_string(),
        }),
        _ => Ok(()),
    }
}
