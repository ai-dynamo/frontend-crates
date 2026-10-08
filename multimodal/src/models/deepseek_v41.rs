// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Geometry math adapted from SGLang's deepseek_v41_image_processing.py (Apache-2.0).
// Pinned source: sgl-project/sglang@ffac53d779c08dcdab2d07e5e2a41dba83f0e65c.

//! Pixel-free DSV4.1 image geometry, shared by routers and encoders.
//!
//! This is a geometry planner, not a registered [`super::super::processor::MmFamilyProcessor`].
//! It does not decode, resize pixels, or produce vision embeddings. Configuration
//! is supplied explicitly so both consumers use the same model revision's rules.

use crate::{MmError, Result};

/// Resolved vision configuration, with the same field names as SGLang/HF.
#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub struct DeepseekV41GeometrySpec {
    pub vision_patch_size: u32,
    pub vision_downsample_ratio: u32,
    pub vision_min_pixels: u32,
    pub vision_max_n_token: u32,
    pub vision_max_wh_ratio: Option<f64>,
}

/// Grids and full placeholder span for one image; all dimensions are height/width.
#[derive(Clone, Debug, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
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

impl DeepseekV41GeometrySpec {
    fn validate(&self) -> Result<()> {
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
