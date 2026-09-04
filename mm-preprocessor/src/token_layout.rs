// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Token-layout mechanics for the preprocessing pipeline.
//!
//! Families describe their prompt geometry as a [`TokenLayout`] value
//! (`processor.rs`); [`apply_layout`] applies it mechanically. Expanding the
//! already-tokenized prompt means non-media tokens can never drift from a
//! retokenize.

use crate::processor::TokenLayout;

/// The expanded prompt plus two views per media item (indexed as in the
/// layout): `offsets`, the inclusive `(start, end)` range of the item's whole
/// expansion, and `feature_ranges`, the sub-ranges of its `Feature` runs —
/// the consumer's scatter targets. For a bare-placeholder image the two
/// coincide; a structured expansion's `offsets` also cover literal scaffold
/// (wrappers, timestamps) that `feature_ranges` exclude.
pub struct ExpandedPrompt {
    pub input_ids: Vec<i32>,
    pub offsets: Vec<(u32, u32)>,
    pub feature_ranges: Vec<Vec<std::ops::Range<u32>>>,
}

/// Apply a family's [`TokenLayout`] to the original prompt, validating the
/// whole contract rather than just indexing safely:
/// * text and media `src` ranges are in bounds, ascending, and together
///   cover every source token exactly once — a forgotten tail segment must
///   not silently truncate the prompt;
/// * each of the `n_items` media items is placed exactly once;
/// * no item expands to zero feature tokens (no scatter target).
pub fn apply_layout(
    src: &[i32],
    layout: &TokenLayout,
    n_items: usize,
) -> crate::Result<ExpandedPrompt> {
    let _ = (src, layout, n_items);
    todo!("validating single-pass expansion")
}

/// Build the simplest layout: the i-th occurrence of `placeholder_id` in
/// `ids` becomes the i-th media item — one `Feature` run of `counts[i]`
/// copies replacing the occurrence. Errs when the occurrence count and
/// `counts` disagree.
pub fn layout_by_placeholder(
    ids: &[i32],
    placeholder_id: i32,
    counts: &[usize],
) -> crate::Result<TokenLayout> {
    let _ = (ids, placeholder_id, counts);
    todo!("qwen-style repeat layout")
}
