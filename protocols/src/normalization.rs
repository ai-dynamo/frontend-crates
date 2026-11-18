// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Explicit normalization of parsed request fields, separate from validation.
//!
//! Namespaces mirror [`crate::validation`]: `openai` is the baseline and the
//! engine namespaces reuse it where their behavior matches. Only completion
//! `logprobs` is covered. Model defaults, sentinel rewrites, greedy
//! transformations, and request-wide normalization are not implemented.

pub mod openai {
    /// Preserves completion `logprobs` unchanged: no default is inserted for `None`,
    /// and explicit values, including invalid ones, are left for validation.
    ///
    /// Call after parsing and before the selected logprobs validator. Parsing already
    /// maps both a missing key and explicit null to `None`, and serialization omits
    /// the key for `None`; this operation cannot recover that distinction. It cannot
    /// fail and is idempotent. Later engine conversion owns the effective count.
    pub fn normalize_completion_logprobs(_value: &mut Option<u8>) {}
}

/// vLLM normalization; currently identical to the OpenAI baseline.
pub mod vllm {
    pub use super::openai::normalize_completion_logprobs;
}

/// SGLang normalization; currently identical to the OpenAI baseline.
pub mod sglang {
    pub use super::openai::normalize_completion_logprobs;
}
