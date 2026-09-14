// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Token-layout mechanics for the preprocessing pipeline.
//!
//! Families describe their prompt geometry as a [`TokenLayout`] value
//! (`processor.rs`); [`apply_layout`] applies it mechanically. Expanding the
//! already-tokenized prompt means non-media tokens can never drift from a
//! retokenize.

use crate::processor::{ExpansionPart, Segment, TokenLayout};
use crate::{MmError, Result};

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

/// Apply a family's [`TokenLayout`] to the original prompt. The i-th entry in
/// `feature_token_counts` is the number of embeddings produced by media item
/// i. The function validates while expanding:
/// * the `Text` and `Media::src` ranges cover the original ids exactly once,
///   in order — nothing dropped, nothing duplicated;
/// * each media item appears exactly once;
/// * each item's `Feature` parts contain exactly its expected number of
///   feature tokens.
pub fn apply_layout(
    src: &[i32],
    layout: &TokenLayout,
    feature_token_counts: &[usize],
) -> Result<ExpandedPrompt> {
    let n_items = feature_token_counts.len();
    let mut out = Vec::new();
    let mut offsets: Vec<Option<(u32, u32)>> = vec![None; n_items];
    let mut feature_ranges: Vec<Vec<std::ops::Range<u32>>> = vec![Vec::new(); n_items];
    // Source tokens consumed so far: `Text` copies a range, `Media` replaces
    // its `src` range.
    let mut consumed = 0usize;
    for segment in &layout.segments {
        match segment {
            Segment::Text(range) => {
                let text = src.get(range.clone()).ok_or_else(|| {
                    MmError::internal(format!("layout: text range {range:?} out of bounds"))
                })?;
                if range.start != consumed {
                    return Err(MmError::internal(format!(
                        "layout: text range {range:?} does not resume at source index {consumed}"
                    )));
                }
                consumed = range.end;
                out.extend_from_slice(text);
            }
            Segment::Media {
                item,
                src: replaced,
                expansion,
            } => {
                if replaced.start != consumed || replaced.end < replaced.start {
                    return Err(MmError::internal(format!(
                        "layout: media item {item} src range {replaced:?} does not resume at \
                         source index {consumed}"
                    )));
                }
                if replaced.end > src.len() {
                    return Err(MmError::internal(format!(
                        "layout: media item {item} src range {replaced:?} out of bounds"
                    )));
                }
                consumed = replaced.end;

                let start = out.len() as u32;
                let mut features = 0usize;
                let mut ranges = Vec::new();
                for part in expansion {
                    match part {
                        ExpansionPart::Feature { id, n } => {
                            let part_start = out.len() as u32;
                            out.resize(out.len() + n, *id);
                            features += n;
                            if *n > 0 {
                                ranges.push(part_start..out.len() as u32);
                            }
                        }
                        ExpansionPart::Literal(ids) => out.extend_from_slice(ids),
                    }
                }
                let n = out.len() as u32 - start;
                if n == 0 {
                    return Err(MmError::internal(format!(
                        "layout: media item {item} expands to zero tokens"
                    )));
                }
                let expected = *feature_token_counts.get(*item).ok_or_else(|| {
                    MmError::internal(format!("layout: media item {item} out of range"))
                })?;
                if features != expected {
                    return Err(MmError::internal(format!(
                        "layout: media item {item} expands to {features} feature token(s), \
                         expected {expected}"
                    )));
                }
                if offsets[*item].replace((start, start + n - 1)).is_some() {
                    return Err(MmError::internal(format!(
                        "layout: media item {item} placed twice"
                    )));
                }
                feature_ranges[*item] = ranges;
            }
        }
    }
    if consumed != src.len() {
        return Err(MmError::internal(format!(
            "layout: covers {consumed} of {} source token(s)",
            src.len()
        )));
    }
    let offsets = offsets
        .into_iter()
        .enumerate()
        .map(|(i, slot)| {
            slot.ok_or_else(|| MmError::internal(format!("layout: media item {i} not placed")))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(ExpandedPrompt {
        input_ids: out,
        offsets,
        feature_ranges,
    })
}

/// Build the simplest layout: the i-th occurrence of `placeholder_id` in
/// `ids` becomes media item i, expanded to `counts[i]` copies of the
/// placeholder (a single `Feature` part). Errs if the number of occurrences
/// differs from `counts.len()`.
pub fn layout_by_placeholder(
    ids: &[i32],
    placeholder_id: i32,
    counts: &[usize],
) -> Result<TokenLayout> {
    let found = ids.iter().filter(|&&id| id == placeholder_id).count();
    if found != counts.len() {
        return Err(MmError::invalid_input(format!(
            "prompt has {found} media placeholder(s) but {} media item(s)",
            counts.len()
        )));
    }
    let mut segments = Vec::new();
    let mut text_start = 0;
    let mut item = 0;
    for (pos, &id) in ids.iter().enumerate() {
        if id == placeholder_id {
            if text_start < pos {
                segments.push(Segment::Text(text_start..pos));
            }
            segments.push(Segment::Media {
                item,
                src: pos..pos + 1,
                expansion: vec![ExpansionPart::Feature {
                    id: placeholder_id,
                    n: counts[item],
                }],
            });
            item += 1;
            text_start = pos + 1;
        }
    }
    if text_start < ids.len() {
        segments.push(Segment::Text(text_start..ids.len()));
    }
    Ok(TokenLayout { segments })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn expand(ids: &[i32], placeholder: i32, counts: &[usize]) -> Result<ExpandedPrompt> {
        apply_layout(
            ids,
            &layout_by_placeholder(ids, placeholder, counts)?,
            counts,
        )
    }

    #[test]
    fn expands_in_order_with_inclusive_offsets() {
        // [7, PAD, 8, PAD, 9] with counts [2, 3]
        let e = expand(&[7, 1, 8, 1, 9], 1, &[2, 3]).unwrap();
        assert_eq!(e.input_ids, vec![7, 1, 1, 8, 1, 1, 1, 9]);
        assert_eq!(e.offsets, vec![(1, 2), (4, 6)]);
        // Feature-only expansions: the feature range is the whole expansion.
        assert_eq!(e.feature_ranges, vec![vec![1..3], vec![4..7]]);
    }

    #[test]
    fn count_mismatch_errs() {
        assert!(expand(&[7, 1, 9], 1, &[2, 3]).is_err());
        assert!(expand(&[7, 1, 1, 9], 1, &[2]).is_err());
    }

    #[test]
    fn zero_count_errs() {
        assert!(expand(&[7, 1, 9], 1, &[0]).is_err());
    }

    #[test]
    fn no_placeholders_no_items_ok() {
        let e = expand(&[7, 8], 1, &[]).unwrap();
        assert_eq!(e.input_ids, vec![7, 8]);
        assert!(e.offsets.is_empty());
    }

    /// A structured expansion mixing `Literal` markers with `Feature` tokens:
    /// the offsets cover the whole expansion, the feature ranges only the
    /// `Feature` parts.
    #[test]
    fn literal_parts_are_skipped_by_feature_ranges() {
        let layout = TokenLayout {
            segments: vec![
                Segment::Text(0..1),
                Segment::Media {
                    item: 0,
                    src: 1..2,
                    expansion: vec![
                        ExpansionPart::Literal(vec![90]),
                        ExpansionPart::Feature { id: 5, n: 2 },
                        ExpansionPart::Literal(vec![91]),
                    ],
                },
                Segment::Text(2..3),
            ],
        };
        let e = apply_layout(&[7, 1, 9], &layout, &[2]).unwrap();
        assert_eq!(e.input_ids, vec![7, 90, 5, 5, 91, 9]);
        assert_eq!(e.offsets, vec![(1, 4)]);
        assert_eq!(e.feature_ranges, vec![vec![2..4]]);
    }

    #[test]
    fn feature_count_mismatch_and_missing_placement_err() {
        let media = |n| Segment::Media {
            item: 0,
            src: 1..2,
            expansion: vec![ExpansionPart::Feature { id: 5, n }],
        };
        // The family's expansion must produce exactly the item's feature count.
        let wrong = TokenLayout {
            segments: vec![Segment::Text(0..1), media(3), Segment::Text(2..3)],
        };
        assert!(apply_layout(&[7, 1, 9], &wrong, &[2]).is_err());
        // Every item must be placed; ranges must be in bounds.
        let missing = TokenLayout {
            segments: vec![Segment::Text(0..3)],
        };
        assert!(apply_layout(&[7, 1, 9], &missing, &[2]).is_err());
        let out_of_bounds = TokenLayout {
            segments: vec![Segment::Text(0..4)],
        };
        assert!(apply_layout(&[7, 1, 9], &out_of_bounds, &[]).is_err());
    }

    /// A family that skips, repeats, or reorders source tokens would silently
    /// serve a truncated or scrambled prompt; the layout must reject it instead.
    #[test]
    fn incomplete_or_disordered_coverage_errs() {
        let media = || Segment::Media {
            item: 0,
            src: 1..2,
            expansion: vec![ExpansionPart::Feature { id: 5, n: 2 }],
        };
        let cases = [
            // Dropped tail: [7, PAD, 9] expanded without the trailing 9.
            vec![Segment::Text(0..1), media()],
            // Dropped head.
            vec![media(), Segment::Text(2..3)],
            // Gap in the middle (source index 1 never consumed).
            vec![Segment::Text(0..1), Segment::Text(2..3)],
            // Duplicated source span.
            vec![
                Segment::Text(0..1),
                Segment::Text(0..1),
                media(),
                Segment::Text(2..3),
            ],
            // Out of order.
            vec![Segment::Text(2..3), media(), Segment::Text(0..1)],
        ];
        for (i, segments) in cases.into_iter().enumerate() {
            let layout = TokenLayout { segments };
            assert!(
                apply_layout(&[7, 1, 9], &layout, &[2]).is_err(),
                "case {i} should be rejected"
            );
        }
    }
}
