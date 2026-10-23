// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Operations that both pinned engines implement equivalently, outside the
//! OpenAI baseline. Engine modules re-export these; see their module docs for
//! the source revisions. Matching numeric rules do not imply matching engine
//! HTTP null handling or error responses.

use super::openai::MAX_TOP_P;
use super::{ValidationError, check};

/// Exclusive lower bound for `top_p`; 0 itself is rejected.
/// OpenAI's baseline [`super::openai::MIN_TOP_P`] is inclusive instead.
pub const MIN_TOP_P_EXCLUSIVE: f32 = 0.0;
/// Inclusive lower bound for `min_p`.
pub const MIN_MIN_P: f32 = 0.0;
/// Inclusive upper bound for `min_p`.
pub const MAX_MIN_P: f32 = 1.0;

/// Accepts `None` or `top_p` in (0, 1].
pub fn validate_top_p(value: Option<f32>) -> Result<(), ValidationError> {
    check(
        value,
        |v| v > MIN_TOP_P_EXCLUSIVE && v <= MAX_TOP_P,
        "top_p",
        // Text kept identical to the existing Dynamo diagnostic.
        || format!("Top_p must be between {MIN_TOP_P_EXCLUSIVE} and {MAX_TOP_P}"),
    )
}

/// Accepts `None` or `min_p` in [0, 1].
///
/// The covered request types have no `min_p` field, so request traits do not call this.
pub fn validate_min_p(value: Option<f32>) -> Result<(), ValidationError> {
    check(
        value,
        |v| (MIN_MIN_P..=MAX_MIN_P).contains(&v),
        "min_p",
        || format!("Min_p must be between {MIN_MIN_P} and {MAX_MIN_P}"),
    )
}
