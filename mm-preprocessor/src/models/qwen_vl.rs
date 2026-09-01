// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Qwen VL family (Qwen2-VL / 2.5-VL / 3-VL / 3.5) image processor.
//!
//! Pure-Rust equivalent of the HF `Qwen2VLImageProcessor` pipeline:
//! `smart_resize` → bicubic resize → rescale + normalize → patchify into
//! `[grid_h*grid_w, C*tps*ps*ps]` (HF flatten order: patches by
//! `(gh/m, gw/m, m, m)`, features by `(C, tps, ps, ps)`, temporal copies
//! duplicated for stills) — plus the image-only M-RoPE fast path.
//! All parameters come from the runtime spec; nothing is hardcoded per model.

use crate::image::resize;
use crate::pipeline::{
    DecodedMedia, Geometry, MmFamilyProcessor, PositionOutput, ProcessedItem, TokenLayout,
};

/// One media item's placement for M-RoPE: inclusive token range + patch grid.
pub struct MropeItem {
    pub start: u32,
    pub end: u32,
    pub grid: [u32; 3],
}

/// Resolved processor params, deserialized from the consumer-side spec JSON
/// (unknown fields like `family` are ignored here).
#[derive(Clone, Debug, serde::Deserialize)]
pub struct QwenVlSpec {
    pub image_token_id: i32,
    pub patch_size: usize,
    pub merge_size: usize,
    pub temporal_patch_size: usize,
    pub min_pixels: usize,
    pub max_pixels: usize,
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
    #[serde(default)]
    pub resample: Resampler,
}

/// The HF image processor the pipeline must match bit-exactly. Defaults to
/// the one a default server runs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Resampler {
    /// `Qwen2VLImageProcessor` / `…Fast` — torchvision on a uint8 tensor.
    #[default]
    AtenU8,
    /// `Qwen2VLImageProcessorPil`, behind `--disable-fast-image-processor`.
    Pil,
}

impl From<Resampler> for resize::Resample {
    fn from(r: Resampler) -> Self {
        match r {
            Resampler::AtenU8 => resize::Resample::AtenU8,
            Resampler::Pil => resize::Resample::Pil(resize::Filter::Bicubic),
        }
    }
}

pub struct QwenVlProcessor {
    // PR3 adds: the spec, plus a per-channel u8 → normalized-f32 lookup table
    // that rounds as the mirrored HF processor rounds (the slow path rescales
    // then normalizes; the fast path folds the rescale into mean/std first —
    // the two differ on 128 of the 256 u8 inputs, so picking the wrong form
    // silently costs bit-exactness).
    #[allow(dead_code)] // read from PR3
    spec: QwenVlSpec,
}

impl QwenVlProcessor {
    pub fn new(spec: QwenVlSpec) -> Result<Self, String> {
        if spec.patch_size == 0 || spec.merge_size == 0 || spec.temporal_patch_size == 0 {
            return Err("qwen_vl spec: sizes must be positive".into());
        }
        Ok(Self { spec })
    }

    pub fn from_spec_json(json: &str) -> Result<Self, String> {
        let spec: QwenVlSpec =
            serde_json::from_str(json).map_err(|e| format!("qwen_vl spec: {e}"))?;
        Self::new(spec)
    }
}

impl MmFamilyProcessor for QwenVlProcessor {
    fn process_item(&self, media: &DecodedMedia) -> Result<ProcessedItem, String> {
        let _ = media;
        // PR3: smart_resize → resize (skipped when dims already match) →
        // fused normalize+patchify (HF flatten order) → feature tensor
        // `[gh*gw, 3*tps*ps*ps]` + aux `image_grid_thw` + `Grid([1, gh, gw])`.
        todo!("PR3: HF Qwen2VLImageProcessor equivalent")
    }

    fn layout(&self, input_ids: &[i32], items: &[Geometry]) -> Result<TokenLayout, String> {
        let _ = (input_ids, items);
        // PR3: `t*h*w / merge_size²` tokens per image, via
        // `token_layout::layout_by_placeholder`.
        todo!("PR3: placeholder-repeat layout")
    }

    fn positions(
        &self,
        input_len: usize,
        offsets: &[(u32, u32)],
        items: &[Geometry],
    ) -> Result<PositionOutput, String> {
        let _ = (input_len, offsets, items);
        todo!("PR3: mrope_image_only over the item offsets/grids")
    }
}

/// The Qwen `smart_resize`: dims divisible by `factor`, total pixels within
/// `[min_pixels, max_pixels]`, aspect ratio preserved as closely as possible.
/// Matches the Python reference exactly, including `round()`'s
/// round-half-to-even; rejects (rather than reaches PIL with) the degenerate
/// case where a very thin image floors a side to 0.
pub fn smart_resize(
    height: usize,
    width: usize,
    factor: usize,
    min_pixels: usize,
    max_pixels: usize,
) -> Result<(usize, usize), String> {
    let _ = (height, width, factor, min_pixels, max_pixels);
    todo!("PR3: banker's-rounding resize targets")
}

/// Image-only M-RoPE fast path (the image branch of
/// `MRotaryEmbedding.get_rope_index`, identical across Qwen generations):
/// text runs sequentially on all three rows; each image spans `(t, h/m, w/m)`
/// index grids; positions advance by `max(t, h/m, w/m)` past an image.
/// Returns flattened row-major `[3, input_len]` positions and the delta
/// (`max + 1 - input_len`). `items` must be in prompt order.
pub fn mrope_image_only(
    input_len: usize,
    items: &[MropeItem],
    merge_size: usize,
) -> Result<(Vec<i64>, i64), String> {
    let _ = (input_len, items, merge_size);
    todo!("PR3: three-row position fill")
}
