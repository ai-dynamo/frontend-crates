// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Reusable image transform primitives (flat HWC layout) that model-specific
//! processors compose: single-pass u8→f32 normalize, pad-to-grid, patch
//! extraction. Families with fused fast paths (e.g. qwen's normalize-inside-
//! patchify LUT) bypass these; they exist for families whose pipelines keep
//! the stages separate.

/// Normalize u8 RGB pixels to f32 in a single pass: `(pixel/255 - mean) / std`.
/// Writes into `out`, which must have length `h * w * 3`.
pub fn normalize_rgb_f32(
    rgb: &[u8],
    h: usize,
    w: usize,
    mean: &[f32; 3],
    std: &[f32; 3],
    out: &mut [f32],
) {
    let _ = (rgb, h, w, mean, std, out);
    todo!("PR2")
}

/// Pad an HWC image to a grid-aligned size, filling padded pixels with
/// `pad_value`. Returns the padded buffer and the new (height, width).
pub fn pad_to_grid(
    rgb_f32: &[f32],
    h: usize,
    w: usize,
    channels: usize,
    grid_h: usize,
    grid_w: usize,
    pad_value: &[f32],
) -> (Vec<f32>, usize, usize) {
    let _ = (rgb_f32, h, w, channels, grid_h, grid_w, pad_value);
    todo!("PR2")
}

/// Reshape a padded HWC image into patches of shape `[num_patches, ph, pw, C]`.
/// `h` and `w` must be divisible by `ph` and `pw` respectively.
pub fn extract_patches_hwc(
    data: &[f32],
    h: usize,
    w: usize,
    channels: usize,
    ph: usize,
    pw: usize,
) -> Vec<f32> {
    let _ = (data, h, w, channels, ph, pw);
    todo!("PR2")
}
