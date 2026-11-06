// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Adapted from sgl-project/sglang PR #46563 (Apache-2.0), by Kan Wu and Zhihan Guo.

//! GLM-5.3-Flash images as SGLang's `Glm4vImageProcessor` runs transformers'
//! `Glm5NextImageProcessor`: `smart_to_rgb` decode, resize to the content size,
//! zero-pad to Dynamo's planned canvas, then Qwen-VL's normalize and patch order.

use crate::MmError;
use crate::image::decode::{DecodeLimits, decode_rgb_smart};
use crate::image::resize::resize_rgb;
use crate::models::glm5_next::{Glm5NextProcessor, Glm5NextSpec};
use crate::models::qwen_vl::{MropeItem, Resampler, mrope_image_only};
use crate::processor::{
    DecodedMedia, Geometry, MediaMetadata, MmFamilyProcessor, Modality, PositionOutput,
    ProcessedItem, Tensor, TensorData, TokenLayout,
};
use crate::token_layout::layout_by_placeholder;

/// Full image pipeline parameters, including the shared metadata planner.
#[derive(Clone, Debug, serde::Deserialize)]
pub struct Glm5NextImageSpec {
    #[serde(flatten)]
    pub planner: Glm5NextSpec,
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
    #[serde(default)]
    pub resample: Resampler,
}

pub struct Glm5NextImageProcessor {
    planner: Glm5NextProcessor,
    spec: Glm5NextImageSpec,
    /// Per-channel u8 to normalized f32, as Dynamo's Qwen-VL `normalize_lut`.
    lut: [[f32; 256]; 3],
}

impl Glm5NextImageProcessor {
    pub fn new(spec: Glm5NextImageSpec) -> Result<Self, MmError> {
        let planner = Glm5NextProcessor::new(spec.planner.clone())?;
        if spec
            .image_std
            .iter()
            .any(|std| !std.is_finite() || *std <= 0.0)
        {
            return Err(MmError::invalid_input(
                "glm5_next: image_std must be positive",
            ));
        }
        if spec.image_mean.iter().any(|mean| !mean.is_finite()) {
            return Err(MmError::invalid_input(
                "glm5_next: image_mean must be finite",
            ));
        }
        let lut = core::array::from_fn(|c| {
            let (mean, std) = (spec.image_mean[c], spec.image_std[c]);
            match spec.resample {
                Resampler::Pil => core::array::from_fn(|v| (v as f32 / 255.0 - mean) / std),
                // torchvision folds the rescale into mean and std first.
                Resampler::AtenU8 => {
                    core::array::from_fn(|v| (v as f32 - mean * 255.0) / (std * 255.0))
                }
            }
        });
        Ok(Self { planner, spec, lut })
    }

    /// `[grid_h * grid_w, 3 * tps * ps * ps]` in merged-block order, each still
    /// repeated over the temporal patch.
    fn patchify(&self, rgb: &[u8], h: usize, w: usize) -> Result<Vec<f32>, MmError> {
        let geometry = &self.spec.planner.geometry;
        let (ps, m, tps) = (
            geometry.patch_size as usize,
            geometry.merge_size as usize,
            geometry.temporal_patch_size as usize,
        );
        let (gh, gw) = (h / ps, w / ps);
        let len = h
            .checked_mul(w)
            .and_then(|n| n.checked_mul(3))
            .and_then(|n| n.checked_mul(tps))
            .ok_or_else(|| MmError::limit_exceeded("glm5_next: feature size overflows usize"))?;
        let mut out = Vec::new();
        out.try_reserve_exact(len)
            .map_err(|_| MmError::limit_exceeded("glm5_next: feature allocation failed"))?;
        for (bh, bw, mh, mw) in (0..gh / m).flat_map(|bh| {
            (0..gw / m).flat_map(move |bw| {
                (0..m).flat_map(move |mh| (0..m).map(move |mw| (bh, bw, mh, mw)))
            })
        }) {
            let (y0, x0) = ((bh * m + mh) * ps, (bw * m + mw) * ps);
            for (c, lut) in self.lut.iter().enumerate() {
                let start = out.len();
                for y in y0..y0 + ps {
                    let row = &rgb[(y * w + x0) * 3..(y * w + x0 + ps) * 3];
                    out.extend(row.iter().skip(c).step_by(3).map(|&v| lut[v as usize]));
                }
                for _ in 1..tps {
                    out.extend_from_within(start..start + ps * ps);
                }
            }
        }
        Ok(out)
    }
}

impl MmFamilyProcessor for Glm5NextImageProcessor {
    fn decode_image(&self, data: &[u8], limits: &DecodeLimits) -> crate::Result<DecodedMedia> {
        let (rgb, height, width) = decode_rgb_smart(data, limits)?;
        Ok(DecodedMedia::Image { rgb, height, width })
    }

    fn num_media_tokens(&self, media: &MediaMetadata) -> Result<usize, MmError> {
        self.planner.num_media_tokens(media)
    }

    fn layout_from_metadata(
        &self,
        input_ids: &[i32],
        media: &[MediaMetadata],
    ) -> Result<TokenLayout, MmError> {
        self.planner.layout_from_metadata(input_ids, media)
    }

    fn process_item(&self, media: &DecodedMedia) -> Result<ProcessedItem, MmError> {
        let DecodedMedia::Image { rgb, height, width } = media;
        let (h, w) = (*height, *width);
        let geometry = &self.spec.planner.geometry;
        let pixels = h.checked_mul(w).and_then(|n| n.checked_mul(3));
        if h == 0 || w == 0 || pixels != Some(rgb.len()) {
            return Err(MmError::invalid_input(
                "glm5_next: invalid RGB dimensions or buffer length",
            ));
        }
        let width = u32::try_from(w)
            .map_err(|_| MmError::limit_exceeded("glm5_next: width exceeds u32"))?;
        let height = u32::try_from(h)
            .map_err(|_| MmError::limit_exceeded("glm5_next: height exceeds u32"))?;
        let plan = self.planner.plan_image(width, height)?;
        let (canvas_h, canvas_w) = (plan.resized_height as usize, plan.resized_width as usize);

        // `Glm5NextImageProcessor.resize`: images above the token minimum are never upscaled.
        let factor =
            (geometry.patch_size * geometry.merge_size * geometry.patch_expand_factor) as u64;
        let mut scale = (canvas_h as f64 / h as f64).min(canvas_w as f64 / w as f64);
        if (h as u128 * w as u128)
            >= u128::from(factor) * u128::from(factor) * u128::from(geometry.min_image_tokens)
        {
            scale = scale.min(1.0);
        }
        let content =
            |side: usize, canvas: usize| ((side as f64 * scale).floor() as usize).clamp(1, canvas);
        let (content_h, content_w) = (content(h, canvas_h), content(w, canvas_w));
        let resized;
        let content_rgb = if (content_h, content_w) != (h, w) {
            resized = resize_rgb(rgb, h, w, content_h, content_w, self.spec.resample.into());
            &resized
        } else {
            rgb
        };
        let canvas_len = canvas_h
            .checked_mul(canvas_w)
            .and_then(|n| n.checked_mul(3))
            .ok_or_else(|| MmError::limit_exceeded("glm5_next: canvas size overflows usize"))?;
        let mut canvas = Vec::new();
        canvas
            .try_reserve_exact(canvas_len)
            .map_err(|_| MmError::limit_exceeded("glm5_next: canvas allocation failed"))?;
        canvas.resize(canvas_len, 0u8);
        for (y, row) in content_rgb.chunks_exact(content_w * 3).enumerate() {
            canvas[y * canvas_w * 3..][..content_w * 3].copy_from_slice(row);
        }

        let grid = [1, plan.grid_h, plan.grid_w];
        let patch_dim = (geometry.patch_size as usize)
            .checked_mul(geometry.patch_size as usize)
            .and_then(|n| n.checked_mul(geometry.temporal_patch_size as usize))
            .and_then(|n| n.checked_mul(3))
            .ok_or_else(|| MmError::limit_exceeded("glm5_next: patch dimension overflows usize"))?;
        Ok(ProcessedItem {
            modality: Modality::Image,
            feature_token_count: plan.num_image_tokens as usize,
            feature: Tensor {
                shape: vec![plan.grid_h as usize * plan.grid_w as usize, patch_dim],
                data: TensorData::F32(self.patchify(&canvas, canvas_h, canvas_w)?),
            },
            aux: vec![(
                "image_grid_thw".into(),
                Tensor {
                    shape: vec![3],
                    data: TensorData::I64(grid.iter().map(|&g| g as i64).collect()),
                },
            )],
            geometry: Some(Geometry::Grid(grid)),
        })
    }

    fn layout(&self, input_ids: &[i32], items: &[ProcessedItem]) -> Result<TokenLayout, MmError> {
        let counts: Vec<usize> = items.iter().map(|item| item.feature_token_count).collect();
        if self
            .spec
            .planner
            .video_token_id
            .is_some_and(|id| input_ids.contains(&id))
        {
            return Err(MmError::unsupported("glm5_next supports only images"));
        }
        layout_by_placeholder(input_ids, self.spec.planner.image_token_id, &counts)
    }

    /// `MRotaryEmbedding.get_rope_index_glm4v` on an image-only prompt.
    fn positions(
        &self,
        input_len: usize,
        offsets: &[(u32, u32)],
        items: &[ProcessedItem],
    ) -> Result<PositionOutput, MmError> {
        if offsets.len() != items.len() {
            return Err(MmError::invalid_input(
                "glm5_next: offsets and items differ in length",
            ));
        }
        let items: Vec<MropeItem> = (offsets.iter().zip(items))
            .map(|(&(start, end), item)| match item.geometry {
                Some(Geometry::Grid(grid)) => Ok(MropeItem { start, end, grid }),
                _ => Err(MmError::invalid_input("glm5_next: item without its grid")),
            })
            .collect::<Result<_, _>>()?;
        let merge_size = self.spec.planner.geometry.merge_size as usize;
        let (positions, delta) = mrope_image_only(input_len, &items, merge_size)?;
        Ok(PositionOutput::MRope { positions, delta })
    }
}
