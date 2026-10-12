// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Geometry math adapted from SGLang's deepseek_v41_image_processing.py (Apache-2.0).
// Pinned source: sgl-project/sglang@ffac53d779c08dcdab2d07e5e2a41dba83f0e65c.

//! Pixel-free DSV4.1 image geometry, shared by routers and encoders.
//!
//! The registered processor supports metadata-only counting and prompt layout.
//! It does not decode, resize pixels, or produce vision embeddings. Configuration
//! is supplied explicitly so both consumers use the same model revision's rules.

use crate::processor::{
    DecodedMedia, MediaMetadata, MmFamilyProcessor, ProcessedItem, TokenLayout,
};
use crate::{MmError, Result};

/// Resolved vision configuration, with the same field names as SGLang/HF.
#[derive(Clone, Debug, serde::Deserialize)]
pub struct DeepseekV41GeometrySpec {
    #[serde(alias = "patch_size")]
    pub vision_patch_size: u32,
    #[serde(alias = "downsample_ratio")]
    pub vision_downsample_ratio: u32,
    #[serde(alias = "min_pixels")]
    pub vision_min_pixels: u32,
    #[serde(alias = "max_image_tokens")]
    pub vision_max_n_token: u32,
    #[serde(alias = "max_wh_ratio")]
    pub vision_max_wh_ratio: Option<f64>,
}

/// Planning parameters resolved from the checkpoint, without serving defaults.
#[derive(Clone, Debug, serde::Deserialize)]
pub struct DeepseekV41Spec {
    pub image_token_id: i32,
    #[serde(flatten)]
    pub geometry: DeepseekV41GeometrySpec,
}

/// DSV4.1 metadata-only processor. Pixel preprocessing is explicitly unsupported.
///
/// ```
/// use dynamo_multimodal::{Result, registry, token_layout};
/// use dynamo_multimodal::processor::MediaMetadata;
/// # fn main() -> Result<()> {
/// // Synthetic settings: callers supply their checkpoint's resolved values.
/// let family = registry::processor_from_spec(r#"{
///     "family": "deepseek_v41", "image_token_id": 100,
///     "vision_patch_size": 14, "vision_downsample_ratio": 3,
///     "vision_min_pixels": 0, "vision_max_n_token": 1024
/// }"#)?;
/// let media = [MediaMetadata::Image { width: 84, height: 126 }];
/// let ids = [7, 100, 9];
/// let counts = media.iter().map(|item| family.num_media_tokens(item))
///     .collect::<Result<Vec<_>>>()?;
/// let layout = family.layout_from_metadata(&ids, &media)?;
/// let expanded = token_layout::apply_layout(&ids, &layout, &counts, 13)?;
/// assert_eq!(expanded.offsets, [(1, 11)]); // inclusive full spans
/// assert_eq!(expanded.feature_ranges[0][0], 1..12);
/// # Ok(())
/// # }
/// ```
pub struct DeepseekV41Processor {
    spec: DeepseekV41Spec,
}

impl DeepseekV41Processor {
    pub fn new(spec: DeepseekV41Spec) -> Result<Self> {
        spec.geometry.validate()?;
        if spec.image_token_id < 0 {
            return Err(MmError::invalid_input(
                "deepseek_v41: image_token_id must be nonnegative",
            ));
        }
        Ok(Self { spec })
    }

    pub fn plan_image(&self, width: u32, height: u32) -> Result<DeepseekV41ImagePlan> {
        plan_image_grid(width, height, &self.spec.geometry)
    }
}

impl MmFamilyProcessor for DeepseekV41Processor {
    fn num_media_tokens(&self, media: &MediaMetadata) -> Result<usize> {
        match media {
            MediaMetadata::Image { width, height } => {
                usize::try_from(self.plan_image(*width, *height)?.num_image_tokens)
                    .map_err(|_| MmError::limit_exceeded("deepseek_v41: token count exceeds usize"))
            }
            _ => Err(MmError::unsupported("deepseek_v41 supports only images")),
        }
    }

    fn layout_from_metadata(
        &self,
        input_ids: &[i32],
        media: &[MediaMetadata],
    ) -> Result<TokenLayout> {
        let counts = media
            .iter()
            .map(|item| self.num_media_tokens(item))
            .collect::<Result<Vec<_>>>()?;
        // All positions, including boundaries and row separators, use the same
        // vocabulary id. The encoder supplies embeddings for the entire span.
        crate::token_layout::layout_by_placeholder(input_ids, self.spec.image_token_id, &counts)
    }

    fn process_item(&self, _media: &DecodedMedia) -> Result<ProcessedItem> {
        Err(MmError::unsupported(
            "deepseek_v41 pixel preprocessing is not implemented; use metadata-only planning",
        ))
    }

    fn layout(&self, _input_ids: &[i32], _items: &[ProcessedItem]) -> Result<TokenLayout> {
        Err(MmError::unsupported(
            "deepseek_v41 processed-item layout is not implemented; use layout_from_metadata",
        ))
    }
}

/// Grids and full placeholder span for one image; all dimensions are height/width.
#[derive(Clone, Debug, PartialEq, Eq, serde::Deserialize)]
pub struct DeepseekV41ImagePlan {
    pub resized_height: u32,
    pub resized_width: u32,
    pub vit_height: u32,
    pub vit_width: u32,
    pub llm_height: u32,
    pub llm_width: u32,
    /// Includes a separator per row and two boundary positions, not just image features.
    pub num_image_tokens: u32,
}

impl DeepseekV41ImagePlan {
    /// SGLang's separate token types: START=0, IMAGE=1, ROW_SEPARATOR=2, END=3.
    /// These are not vocabulary ids: every expanded id is `image_token_id`.
    /// Iteration does not allocate; consumers choose whether to materialize it.
    pub fn token_types(&self) -> impl Iterator<Item = u8> + '_ {
        std::iter::once(0)
            .chain((0..self.llm_height).flat_map(|_| {
                std::iter::repeat_n(1, self.llm_width as usize).chain(std::iter::once(2))
            }))
            .chain(std::iter::once(3))
    }
}

impl DeepseekV41GeometrySpec {
    pub(crate) fn validate(&self) -> Result<()> {
        if self.vision_patch_size == 0 || self.vision_downsample_ratio == 0 {
            return Err(MmError::invalid_input(
                "deepseek_v41 geometry: patch size and downsample ratio must be positive",
            ));
        }
        // One grid cell, one row separator, and two boundaries is the smallest span.
        if self.vision_max_n_token < 4 {
            return Err(MmError::invalid_input(
                "deepseek_v41 geometry: token budget must be at least 4",
            ));
        }
        if let Some(ratio) = self.vision_max_wh_ratio
            && (!ratio.is_finite() || ratio <= 0.0)
        {
            return Err(MmError::invalid_input(
                "deepseek_v41 geometry: width/height cap must be finite and positive",
            ));
        }
        Ok(())
    }
}

/// Mirror SGLang's `plan_image_grid` from original dimensions, without reading pixels.
///
/// The width/height cap is intentionally asymmetric: it only caps wide images.
/// Minimum-pixel upscaling truncates each dimension before patch alignment, as
/// Python does. The returned span fits `vision_max_n_token` or planning fails.
pub fn plan_image_grid(
    width: u32,
    height: u32,
    spec: &DeepseekV41GeometrySpec,
) -> Result<DeepseekV41ImagePlan> {
    spec.validate()?;
    if width == 0 || height == 0 {
        return Err(MmError::invalid_input(
            "deepseek_v41 geometry: image dimensions must be positive",
        ));
    }
    let (mut width, mut height) = (f64::from(width), f64::from(height));
    if let Some(ratio) = spec.vision_max_wh_ratio
        && width > height * ratio
    {
        width = height * ratio;
    }
    let pixels = width * height;
    if pixels > 0.0 && pixels < f64::from(spec.vision_min_pixels) {
        let ratio = (f64::from(spec.vision_min_pixels) / pixels).sqrt();
        width = (width * ratio).trunc();
        height = (height * ratio).trunc();
    }
    if width <= 0.0 || height <= 0.0 || !width.is_finite() || !height.is_finite() {
        return Err(MmError::invalid_input(
            "deepseek_v41 geometry: minimum-pixel scaling produced invalid dimensions",
        ));
    }

    let p = f64::from(spec.vision_patch_size);
    let mut best_width = (width / p).ceil() * p;
    let mut best_height = (height / p).ceil() * p;
    let grid = |h: f64, w: f64| {
        let downsample = f64::from(spec.vision_downsample_ratio);
        (
            ((h / p).floor() / downsample).ceil(),
            ((w / p).floor() / downsample).ceil(),
        )
    };
    let token_count = |h: f64, w: f64| h * (w + 1.0) + 2.0;
    let (lh, lw) = grid(best_height, best_width);
    let budget = f64::from(spec.vision_max_n_token);
    if token_count(lh, lw) > budget {
        let r = height / width;
        let max_w = ((budget - 2.0) / r + 0.25).sqrt() - 0.5;
        let max_h = max_w * r;
        let cell = p * f64::from(spec.vision_downsample_ratio);
        (best_height, best_width) = if max_w < 1.0 {
            (((budget - 2.0) / 2.0).floor() * cell, cell)
        } else if max_h < 1.0 {
            (cell, (budget - 3.0) * cell)
        } else {
            let beta = (max_w.floor() * cell / width).min(max_h.floor() * cell / height);
            (
                (height * beta / p).floor() * p,
                (width * beta / p).floor() * p,
            )
        };
    }
    let (lh, lw) = grid(best_height, best_width);
    let count = token_count(lh, lw);
    if lh < 1.0 || lw < 1.0 || !count.is_finite() || count > budget {
        return Err(MmError::invalid_input(
            "deepseek_v41 geometry: resize plan cannot satisfy the token budget",
        ));
    }
    let dimension = |value: f64| -> Result<u32> {
        if !value.is_finite() || value < 1.0 || value > f64::from(u32::MAX) {
            return Err(MmError::limit_exceeded(
                "deepseek_v41 geometry: planned dimension exceeds u32 range",
            ));
        }
        Ok(value as u32)
    };
    Ok(DeepseekV41ImagePlan {
        resized_height: dimension(best_height)?,
        resized_width: dimension(best_width)?,
        vit_height: dimension((best_height / p).floor())?,
        vit_width: dimension((best_width / p).floor())?,
        llm_height: dimension(lh)?,
        llm_width: dimension(lw)?,
        num_image_tokens: dimension(count)?,
    })
}
