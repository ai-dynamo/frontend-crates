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
    debug_assert_eq!(rgb.len(), h * w * 3);
    debug_assert_eq!(out.len(), h * w * 3);
    let inv255 = 1.0f32 / 255.0;
    for i in 0..h * w {
        for c in 0..3 {
            let raw = rgb[i * 3 + c] as f32 * inv255;
            out[i * 3 + c] = (raw - mean[c]) / std[c];
        }
    }
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
    let new_h = h.div_ceil(grid_h) * grid_h;
    let new_w = w.div_ceil(grid_w) * grid_w;
    let mut out = vec![0.0f32; new_h * new_w * channels];
    for i in 0..new_h * new_w {
        for c in 0..channels {
            out[i * channels + c] = pad_value[c];
        }
    }
    for y in 0..h {
        let src_start = y * w * channels;
        let dst_start = y * new_w * channels;
        out[dst_start..dst_start + w * channels]
            .copy_from_slice(&rgb_f32[src_start..src_start + w * channels]);
    }
    (out, new_h, new_w)
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
    let nph = h / ph;
    let npw = w / pw;
    let patch_size = ph * pw * channels;
    let mut out = vec![0.0f32; nph * npw * patch_size];
    for i in 0..nph {
        for j in 0..npw {
            let patch_idx = i * npw + j;
            for y in 0..ph {
                let src_y = i * ph + y;
                let src_start = (src_y * w + j * pw) * channels;
                let dst_start = patch_idx * patch_size + y * pw * channels;
                out[dst_start..dst_start + pw * channels]
                    .copy_from_slice(&data[src_start..src_start + pw * channels]);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_applies_rescale_then_mean_and_std() {
        let rgb = [0u8, 128, 255, 255, 0, 128];
        let mut out = [0.0f32; 6];
        normalize_rgb_f32(&rgb, 1, 2, &[0.5; 3], &[0.5; 3], &mut out);
        let expect = |v: u8| (v as f32 / 255.0 - 0.5) / 0.5;
        assert_eq!(out, rgb.map(expect));
    }

    #[test]
    fn pad_to_grid_rounds_up_and_keeps_the_original_corner() {
        // 1x2 single-channel image padded to a 2x2 grid.
        let (out, h, w) = pad_to_grid(&[1.0, 2.0], 1, 2, 1, 2, 2, &[-1.0]);
        assert_eq!((h, w), (2, 2));
        assert_eq!(out, vec![1.0, 2.0, -1.0, -1.0]);
    }

    #[test]
    fn patches_are_extracted_row_major_within_each_patch() {
        // 2x4 single-channel image into 2x2 patches: two patches, side by side.
        let data: Vec<f32> = (0..8).map(|v| v as f32).collect();
        assert_eq!(
            extract_patches_hwc(&data, 2, 4, 1, 2, 2),
            vec![0.0, 1.0, 4.0, 5.0, 2.0, 3.0, 6.0, 7.0]
        );
    }
}
