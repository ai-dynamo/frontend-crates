// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Model-specific frame selection: which source frames a model's processor
//! keeps, from the source frame count and rate alone.

use crate::{MmError, Result};

/// The source frames GLM-5.3-Flash keeps, as SGLang's `glm_sample_frame_indices`
/// picks them: the first frame at or after each `1 / target_fps` boundary,
/// refilled by `np.linspace` when that gives too few or too many, duplicates
/// removed, and the last frame repeated when the count is odd.
///
/// It equals transformers' `Glm5NextVideoProcessor.sample_frames` except on
/// clips shorter than `1 / target_fps`, where HF keeps no frame and this keeps
/// the first frame twice.
pub fn glm_frame_indices(
    total_frames: u64,
    fps: f64,
    target_fps: f64,
    max_frames: u32,
) -> Result<Vec<u64>> {
    if total_frames == 0 {
        return Err(MmError::invalid_input("video has no frames"));
    }
    let duration = total_frames as f64 / fps;
    if !(fps.is_finite()
        && fps > 0.0
        && duration.is_finite()
        && target_fps.is_finite()
        && target_fps > 0.0)
        || max_frames == 0
    {
        return Err(MmError::invalid_input(
            "video fps, duration, target fps and max_frames must be positive and finite",
        ));
    }
    let extract = ((duration * target_fps) as u64)
        .min(u64::from(max_frames))
        .max(1);
    let (per_frame, interval, max_second) = (1.0 / fps, 1.0 / target_fps, duration.trunc());
    let reached = |index: u64, current: f64| index as f64 * per_frame >= current;
    let mut indices = Vec::new();
    let (mut next, mut current) = (0, 0.0);
    // Beyond `extract` picks, the linspace over the whole clip below replaces
    // them, so the walk stops there whatever the metadata claims.
    while indices.len() as u64 <= extract {
        let mut index = next.max(((current * fps).ceil() as u64).min(total_frames));
        while index > next && reached(index - 1, current) {
            index -= 1;
        }
        while index < total_frames && !reached(index, current) {
            index += 1;
        }
        if index == total_frames {
            break;
        }
        current += interval;
        indices.push(index);
        if current >= max_second {
            break;
        }
        next = index + 1;
    }
    let kept = indices.len() as u64;
    if kept < extract {
        indices = linspace(indices[0], indices[indices.len() - 1], extract);
    } else if kept > extract {
        indices = linspace(0, total_frames - 1, extract);
    }
    indices.dedup();
    if indices.len() % 2 == 1 {
        indices.push(indices[indices.len() - 1]);
    }
    Ok(indices)
}

/// `np.linspace(start, end, num, dtype=int)`: float64 steps, truncated.
fn linspace(start: u64, end: u64, num: u64) -> Vec<u64> {
    if num == 1 {
        return vec![start];
    }
    let step = (end - start) as f64 / (num - 1) as f64;
    (0..num)
        .map(|i| {
            if i == num - 1 {
                end
            } else {
                (i as f64 * step + start as f64) as u64
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn picks_one_frame_per_half_second() {
        let picked = glm_frame_indices(90, 30.0, 2.0, 2048).unwrap();
        assert_eq!(picked, [0, 15, 30, 45, 60, 75]);
    }

    #[test]
    fn refills_deduplicates_and_evens_the_count() {
        // 2.5 s: the loop stops at 2 s with 4 frames, linspace refills 5, and
        // the odd count repeats the last.
        let picked = glm_frame_indices(25, 10.0, 2.0, 2048).unwrap();
        assert_eq!(picked, [0, 3, 7, 11, 15, 15]);
        // 1 fps source: linspace repeats frames, which are then removed.
        assert_eq!(glm_frame_indices(4, 1.0, 2.0, 2048).unwrap(), [0, 1, 2, 3]);
        // Under 0.5 s: one frame, kept twice.
        assert_eq!(glm_frame_indices(6, 30.0, 2.0, 2048).unwrap(), [0, 0]);
        // max_frames caps the count: linspace over the whole clip.
        assert_eq!(
            glm_frame_indices(300, 30.0, 2.0, 4).unwrap(),
            [0, 99, 199, 299]
        );
    }

    #[test]
    fn jumps_between_boundaries_on_huge_metadata() {
        let picked = glm_frame_indices(1_000_000_000_000, 1e9, 2.0, 2048).unwrap();
        assert_eq!(
            picked,
            (0..2000).map(|i| i * 500_000_000).collect::<Vec<_>>()
        );
        // 1e12 boundaries: the walk stops past max_frames picks.
        let picked = glm_frame_indices(1_000_000_000_000, 1.0, 2.0, 2048).unwrap();
        assert_eq!(picked, linspace(0, 999_999_999_999, 2048));
    }

    /// Equality with SGLang's `glm_sample_frame_indices` on 197 cases; the
    /// table is written by `tests/fixtures/video/generate.py`.
    #[test]
    fn matches_sglang_sampler() {
        let table = include_str!("../../tests/fixtures/video/glm_frame_indices.txt");
        for line in table.lines() {
            let (case, picks) = line.split_once(": ").unwrap();
            let case: Vec<&str> = case.split(' ').collect();
            let picked = glm_frame_indices(
                case[0].parse().unwrap(),
                case[1].parse().unwrap(),
                case[2].parse().unwrap(),
                case[3].parse().unwrap(),
            )
            .unwrap();
            let want: Vec<u64> = picks.split(' ').map(|i| i.parse().unwrap()).collect();
            assert_eq!(picked, want, "{line}");
        }
        assert_eq!(table.lines().count(), 197);
    }

    #[test]
    fn rejects_empty_and_invalid_rates() {
        assert!(glm_frame_indices(0, 30.0, 2.0, 2048).is_err());
        assert!(glm_frame_indices(10, 0.0, 2.0, 2048).is_err());
        assert!(glm_frame_indices(10, f64::NAN, 2.0, 2048).is_err());
        // A subnormal fps gives an infinite duration.
        assert!(glm_frame_indices(1, 1e-309, 2.0, 2048).is_err());
        assert!(glm_frame_indices(10, 30.0, 0.0, 2048).is_err());
        assert!(glm_frame_indices(10, 30.0, 2.0, 0).is_err());
    }
}
