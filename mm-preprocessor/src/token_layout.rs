// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Token-layout mechanics for the preprocessing pipeline.
//!
//! Families describe their prompt geometry as a [`TokenLayout`] value
//! (`processor.rs`); [`apply_layout`] applies it mechanically. Expanding the
//! already-tokenized prompt means non-media tokens can never drift from a
//! retokenize.

use crate::processor::TokenLayout;

/// The expanded prompt plus, per media item (indexed as in the layout), the
/// inclusive `(start, end)` token range it occupies.
pub struct ExpandedPrompt {
    pub input_ids: Vec<i32>,
    pub offsets: Vec<(u32, u32)>,
}

/// Apply a family's [`TokenLayout`] to the original prompt, validating the
/// whole contract rather than just indexing safely:
/// * text ranges are in bounds, ascending, and non-overlapping;
/// * segments cover every source token exactly once — a forgotten tail
///   segment must not silently truncate the prompt;
/// * each of the `n_items` media items is placed exactly once;
/// * no item expands to zero tokens (no representable offset).
pub fn apply_layout(
    src: &[i32],
    layout: &TokenLayout,
    n_items: usize,
) -> Result<ExpandedPrompt, String> {
    let _ = (src, layout, n_items);
    todo!("validating single-pass expansion")
}

/// Build the simplest layout: each occurrence of `placeholder_id` in `ids`
/// becomes `counts[i]` copies (i-th occurrence ↔ i-th media item). Errs when
/// the occurrence count and `counts` disagree.
pub fn layout_by_placeholder(
    ids: &[i32],
    placeholder_id: i32,
    counts: &[usize],
) -> Result<TokenLayout, String> {
    let _ = (ids, placeholder_id, counts);
    todo!("qwen-style repeat layout")
}
