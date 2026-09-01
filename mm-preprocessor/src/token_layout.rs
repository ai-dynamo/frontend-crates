// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Token-layout mechanics for the preprocessing pipeline.
//!
//! Families describe their prompt geometry as a [`TokenLayout`] value
//! (`pipeline.rs`); [`apply_layout`] applies it mechanically. Expanding the
//! already-tokenized prompt means non-media tokens can never drift from a
//! retokenize.

use crate::pipeline::TokenLayout;

/// The expanded prompt plus, per media item (indexed as in the layout), the
/// inclusive `(start, end)` token range it occupies.
pub struct ExpandedPrompt {
    pub input_ids: Vec<i32>,
    pub offsets: Vec<(u32, u32)>,
}

/// Apply a family's [`TokenLayout`] to the original prompt.
///
/// The point of the layout being data is that a family cannot get expansion,
/// offsets, and positions out of sync, so this validates the whole contract
/// rather than just indexing safely:
/// * text ranges are in bounds, ascending, and non-overlapping;
/// * together with the media placeholders they cover every source token
///   exactly once — a family that forgets a tail segment must not silently
///   truncate the prompt;
/// * every one of the `n_items` media items is placed exactly once;
/// * no item expands to zero tokens (which would have no representable offset).
pub fn apply_layout(
    src: &[i32],
    layout: &TokenLayout,
    n_items: usize,
) -> Result<ExpandedPrompt, String> {
    let _ = (src, layout, n_items);
    todo!("PR2: validating single-pass expansion")
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
    todo!("PR2: qwen-style repeat layout")
}
