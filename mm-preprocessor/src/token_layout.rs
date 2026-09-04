// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Token-layout mechanics for the preprocessing pipeline.
//!
//! Families describe their prompt geometry as a [`TokenLayout`] value
//! (`processor.rs`); [`apply_layout`] applies it mechanically. Expanding the
//! already-tokenized prompt means non-media tokens can never drift from a
//! retokenize.

use crate::processor::TokenLayout;

/// The expanded prompt. `offsets` and `feature_ranges` are indexed by media
/// item, in layout order:
/// * `offsets` — inclusive `(start, end)` of the item's whole expansion;
/// * `feature_ranges` — where the engine puts the item's feature embeddings
///   (the `Feature` parts). For a plain image placeholder this is the whole
///   expansion; when the expansion also has `Literal` tokens (markers,
///   timestamps), those are skipped.
pub struct ExpandedPrompt {
    pub input_ids: Vec<i32>,
    pub offsets: Vec<(u32, u32)>,
    pub feature_ranges: Vec<Vec<std::ops::Range<u32>>>,
}

/// Apply a family's [`TokenLayout`] to the original prompt, and validate it
/// while expanding:
/// * the `Text` and `Media::src` ranges cover the original ids exactly once,
///   in order — nothing dropped, nothing duplicated;
/// * each of the `n_items` media items appears exactly once;
/// * every item has at least one feature token.
pub fn apply_layout(
    src: &[i32],
    layout: &TokenLayout,
    n_items: usize,
) -> crate::Result<ExpandedPrompt> {
    let _ = (src, layout, n_items);
    todo!("validating single-pass expansion")
}

/// Build the simplest layout: the i-th occurrence of `placeholder_id` in
/// `ids` becomes media item i, expanded to `counts[i]` copies of the
/// placeholder (a single `Feature` part). Errs if the number of occurrences
/// differs from `counts.len()`.
pub fn layout_by_placeholder(
    ids: &[i32],
    placeholder_id: i32,
    counts: &[usize],
) -> crate::Result<TokenLayout> {
    let _ = (ids, placeholder_id, counts);
    todo!("qwen-style repeat layout")
}
