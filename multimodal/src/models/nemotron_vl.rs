// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Size math adapted from SGLang's internvl_utils.py (Apache-2.0), pinned source
// sgl-project/sglang@a2ffd214696ea7144d3c237133a15df3315d900b, and from
// image_processing_nemotron_h_omni.py (Apache-2.0) in
// nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16@fb0ee0bc6b3f70342afd8d84cb80dbadbaa8a1fe.

//! Nemotron 3.5 Super VL (`nemotron_h_omni`) images.
//!
//! Each image becomes one dynamic-resolution tile, and its `<image>`
//! placeholder expands to `<img>`, one `<image>` per feature token, `</img>`.
//! SGLang's Python processor and the checkpoint's HF remote-code processor
//! pick different tile sizes and resize kernels, so the spec names the
//! reference to mirror ([`NemotronVlSizing`]).
//!
//! The tile size depends on the whole request, so `process_item` is
//! `Unsupported`: engines call [`MmFamilyProcessor::process_items`]. Routers
//! call `layout_from_metadata`, which is exact for the request; its `Feature`
//! parts, like [`NemotronVlProcessor::plan_images`], hold the feature counts.
//! `num_media_tokens` is the span of one image alone: its feature tokens and
//! the two markers.
//!
//! Video, audio and the InternVL tile path (`min_num_patches = 0`) are
//! unsupported.

use crate::image::resize::{Filter, Resample, resize_rgb};
use crate::processor::{
    DecodedMedia, ExpansionPart, MediaMetadata, MmFamilyProcessor, Modality, ProcessedItem,
    Segment, Tensor, TensorData, TokenLayout,
};
use crate::token_layout::layout_by_placeholder;
use crate::{MmError, Result};

/// The reference that decides tile sizes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NemotronVlSizing {
    /// SGLang's `NanoNemotronVLImageProcessor` on a prompt of token ids: the
    /// full pixel path, with the request budget `max_model_len - text tokens`.
    /// SGLang counts the text of a string prompt another way; that is not mirrored.
    Sglang,
    /// The HF remote-code image processor. Sizes and token counts only: its
    /// float resize kernel is not mirrored. vLLM shares the size formula but
    /// adds a text-aware request budget, which this mode does not apply.
    Hf,
}

/// Resolved parameters; `sizing` and `max_model_len` are the consumer's to set
/// for the server it routes to.
#[derive(Clone, Debug, PartialEq, serde::Deserialize)]
pub struct NemotronVlSpec {
    /// `<image>`: the chat-template placeholder and the expanded feature id.
    pub image_token_id: i32,
    /// `<img>`, written before each image's feature tokens.
    pub image_start_token_id: i32,
    /// `</img>`, written after them.
    pub image_end_token_id: i32,
    pub patch_size: u32,
    /// Pixel-shuffle factor, `1 / downsample_ratio`. Only 2 is supported.
    pub downsample_factor: u32,
    pub min_num_patches: u32,
    pub max_num_patches: u32,
    /// SGLang: `--context-length`, or 8192 when unset. HF: the image
    /// processor's `max_model_len`.
    pub max_model_len: u32,
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
    pub sizing: NemotronVlSizing,
}

/// Resize target and `<image>` count of one image.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NemotronVlImagePlan {
    pub resized_width: u32,
    pub resized_height: u32,
    pub num_image_tokens: u32,
}

/// Counts and layout for both references; pixels for `sizing = sglang`.
///
/// ```
/// use dynamo_multimodal::processor::{ExpansionPart, MediaMetadata, Segment};
/// use dynamo_multimodal::{Result, registry, token_layout};
/// # fn main() -> Result<()> {
/// // SGLang with `--context-length 16384`; ids from the checkpoint's tokenizer.json.
/// let family = registry::processor_from_spec(r#"{
///     "family": "nemotron_vl", "sizing": "sglang", "max_model_len": 16384,
///     "image_token_id": 18, "image_start_token_id": 19, "image_end_token_id": 20,
///     "patch_size": 16, "downsample_factor": 2,
///     "min_num_patches": 1024, "max_num_patches": 13312,
///     "image_mean": [0.48145466, 0.4578275, 0.40821073],
///     "image_std": [0.26862954, 0.26130258, 0.27577711]
/// }"#)?;
/// let media = [MediaMetadata::Image { width: 1280, height: 720 }];
/// let ids = [7, 18, 9];
/// let layout = family.layout_from_metadata(&ids, &media)?;
/// // The request's counts: the Feature part of each media segment.
/// let counts: Vec<usize> = layout.segments.iter().filter_map(|segment| match segment {
///     Segment::Media { expansion, .. } => Some(expansion.iter().map(|part| match part {
///         ExpansionPart::Feature { n, .. } => *n,
///         _ => 0,
///     }).sum()),
///     _ => None,
/// }).collect();
/// assert_eq!(counts, [880]); // 1280x704 tile; the HF processor gives 920
/// assert_eq!(family.num_media_tokens(&media[0])?, 882); // with <img> and </img>
/// let expanded = token_layout::apply_layout(&ids, &layout, &counts, 16384)?;
/// assert_eq!(expanded.offsets, [(1, 882)]); // <img> ... </img>
/// assert_eq!(expanded.feature_ranges[0][0], 2..882); // SGLang's offsets: (2, 881)
/// # Ok(())
/// # }
/// ```
pub struct NemotronVlProcessor {
    spec: NemotronVlSpec,
    /// Per-channel u8 to normalized bfloat16 bits.
    lut: [[u16; 256]; 3],
}

impl NemotronVlProcessor {
    pub fn new(spec: NemotronVlSpec) -> Result<Self> {
        if [
            spec.image_token_id,
            spec.image_start_token_id,
            spec.image_end_token_id,
        ]
        .iter()
        .any(|id| *id < 0)
        {
            return Err(MmError::invalid_input(
                "nemotron_vl: token ids must be nonnegative",
            ));
        }
        if spec.patch_size == 0 {
            return Err(MmError::invalid_input(
                "nemotron_vl: patch_size must be positive",
            ));
        }
        if spec.downsample_factor != 2 {
            return Err(MmError::unsupported(
                "nemotron_vl: only downsample_factor = 2 is supported",
            ));
        }
        if spec.sizing == NemotronVlSizing::Sglang && spec.max_model_len == 0 {
            return Err(MmError::invalid_input(
                "nemotron_vl: max_model_len must be positive; SGLang uses 8192 with no --context-length",
            ));
        }
        if spec.sizing == NemotronVlSizing::Sglang && spec.min_num_patches == 0 {
            return Err(MmError::unsupported(
                "nemotron_vl: min_num_patches = 0 selects SGLang's tile path, which is not implemented",
            ));
        }
        if spec
            .image_std
            .iter()
            .any(|std| !std.is_finite() || *std <= 0.0)
        {
            return Err(MmError::invalid_input(
                "nemotron_vl: image_std must be positive",
            ));
        }
        if spec.image_mean.iter().any(|mean| !mean.is_finite()) {
            return Err(MmError::invalid_input(
                "nemotron_vl: image_mean must be finite",
            ));
        }
        // torchvision `ToTensor` then `Normalize`, in f32.
        let lut = core::array::from_fn(|c| {
            let (mean, std) = (spec.image_mean[c], spec.image_std[c]);
            core::array::from_fn(|v| bf16_bits((v as f32 / 255.0 - mean) / std))
        });
        Ok(Self { spec, lut })
    }

    /// One plan per image of a request, in prompt order; `sizes` are
    /// `(width, height)`. SGLang sizing reads the text length off `input_ids`;
    /// HF sizing depends only on the number of images.
    pub fn plan_images(
        &self,
        input_ids: &[i32],
        sizes: &[(u32, u32)],
    ) -> Result<Vec<NemotronVlImagePlan>> {
        if sizes
            .iter()
            .any(|&(width, height)| width == 0 || height == 0)
        {
            return Err(MmError::invalid_input(
                "nemotron_vl: image dimensions must be positive",
            ));
        }
        match self.spec.sizing {
            NemotronVlSizing::Sglang => self.plan_sglang(input_ids, sizes),
            NemotronVlSizing::Hf => self.plan_hf(sizes),
        }
    }

    /// `compute_budgeted_image_sizes` under the processor's budget.
    fn plan_sglang(
        &self,
        input_ids: &[i32],
        sizes: &[(u32, u32)],
    ) -> Result<Vec<NemotronVlImagePlan>> {
        let spec = &self.spec;
        let text_tokens = input_ids
            .iter()
            .filter(|&&id| id != spec.image_token_id)
            .count();
        let budget = i64::from(spec.max_model_len) - text_tokens as i64;
        let downsample = i64::from(spec.downsample_factor);
        let mut max_patches = vec![i64::from(spec.max_num_patches); sizes.len()];
        let mut plans = Vec::new();
        // SGLang stops after 10 rounds and serves the last sizes, over budget or not.
        for _ in 0..10 {
            plans = sizes
                .iter()
                .zip(&max_patches)
                .map(|(&(width, height), &max)| self.plan_sglang_image(width, height, max as u64))
                .collect::<Result<Vec<_>>>()?;
            let total: i64 = plans
                .iter()
                .map(|plan| i64::from(plan.num_image_tokens))
                .sum();
            if total <= budget {
                break;
            }
            // Negative when the text alone is longer than the context.
            let scale = budget as f64 / total as f64;
            for (max, plan) in max_patches.iter_mut().zip(&plans) {
                let patches = i64::from(plan.num_image_tokens) * downsample * downsample;
                *max = ((patches as f64 * scale) as i64).max(i64::from(spec.min_num_patches));
            }
        }
        Ok(plans)
    }

    /// `compute_dynamic_image_size`.
    fn plan_sglang_image(
        &self,
        width: u32,
        height: u32,
        max_patches: u64,
    ) -> Result<NemotronVlImagePlan> {
        let spec = &self.spec;
        let downsample = u64::from(spec.downsample_factor);
        let min_patches = u64::from(spec.min_num_patches);
        let native_side = |pixels: u32| {
            ((f64::from(pixels) / f64::from(spec.patch_size)).round_ties_even() as u64).max(1)
        };
        let (native_w, native_h) = (native_side(width), native_side(height));
        let native = patch_count(native_w, native_h)?;
        let budget = native.min(max_patches).max(min_patches);
        let factor = (budget as f64 / native as f64).sqrt().min(1.0);
        let snap = |side: u64, scale: f64, round: fn(f64) -> f64| {
            ((round(side as f64 * scale / downsample as f64) as u64) * downsample).max(downsample)
        };
        let mut w = snap(native_w, factor, f64::round_ties_even);
        let mut h = snap(native_h, factor, f64::round_ties_even);
        if w * h < min_patches {
            let up = (min_patches as f64 / (w * h) as f64).sqrt();
            (w, h) = (snap(w, up, f64::ceil), snap(h, up, f64::ceil));
        }
        if w * h > max_patches {
            let down = (max_patches as f64 / (w * h) as f64).sqrt();
            (w, h) = (snap(w, down, f64::floor), snap(h, down, f64::floor));
        }
        self.plan(w, h)
    }

    /// `_preprocess`: every image of the call gets the same patch budget.
    fn plan_hf(&self, sizes: &[(u32, u32)]) -> Result<Vec<NemotronVlImagePlan>> {
        let spec = &self.spec;
        let downsample = i64::from(spec.downsample_factor);
        let min_patches = i64::from(spec.min_num_patches);
        let mut budget = ((i64::from(spec.max_model_len) - 4) * downsample * downsample)
            .max(min_patches.saturating_mul(sizes.len() as i64));
        if spec.max_num_patches > 0 {
            budget = budget.min(i64::from(spec.max_num_patches));
        }
        let budget = budget.max(min_patches) as u64;
        sizes
            .iter()
            .map(|&(width, height)| self.plan_hf_image(width, height, budget))
            .collect()
    }

    /// `_compute_target_patches`.
    fn plan_hf_image(&self, width: u32, height: u32, budget: u64) -> Result<NemotronVlImagePlan> {
        let spec = &self.spec;
        let downsample = u64::from(spec.downsample_factor);
        let min_patches = u64::from(spec.min_num_patches);
        let closest_side = |pixels: u32| {
            (f64::from(pixels) / f64::from(spec.patch_size) + 0.5).round_ties_even() as u64
        };
        let (closest_w, closest_h) = (closest_side(width), closest_side(height));
        let closest = patch_count(closest_w, closest_h)?;
        let factor = (budget as f64 / closest as f64).sqrt().min(1.0);
        let mut h = (factor * closest_h as f64).floor() as u64;
        let mut w = (factor * closest_w as f64).floor() as u64;
        // Python divides by `h * w` in the upscale, or resizes to an empty tile.
        if w == 0 || h == 0 {
            return Err(MmError::invalid_input(
                "nemotron_vl: the image is too thin for the HF processor's tile",
            ));
        }
        if budget > min_patches && w * h < min_patches {
            let up = (min_patches as f64 / (w * h) as f64).sqrt();
            h = (up * h as f64).ceil() as u64;
            w = (up * w as f64).ceil() as u64;
        }
        let snap = |side: u64, other: u64| match side % downsample {
            0 => side,
            rest if (side + downsample - rest) * other <= budget => side + downsample - rest,
            rest => (side - rest).max(downsample),
        };
        let h = snap(h, w);
        let w = snap(w, h);
        self.plan(w, h)
    }

    fn plan(&self, patches_w: u64, patches_h: u64) -> Result<NemotronVlImagePlan> {
        let spec = &self.spec;
        let limit = || MmError::limit_exceeded("nemotron_vl: planned tile exceeds u32 range");
        let pixels = |patches: u64| {
            patches
                .checked_mul(u64::from(spec.patch_size))
                .and_then(|pixels| u32::try_from(pixels).ok())
                .ok_or_else(limit)
        };
        let downsample = u64::from(spec.downsample_factor);
        Ok(NemotronVlImagePlan {
            resized_width: pixels(patches_w)?,
            resized_height: pixels(patches_h)?,
            num_image_tokens: u32::try_from(patches_w * patches_h / (downsample * downsample))
                .map_err(|_| limit())?,
        })
    }

    /// `resize_image_to_pixels` then `.to(torch.bfloat16)`: `[1, 3, H, W]`, planar.
    fn pixels(&self, media: &DecodedMedia, plan: &NemotronVlImagePlan) -> Result<ProcessedItem> {
        let DecodedMedia::Image { rgb, height, width } = media else {
            return Err(MmError::unsupported("nemotron_vl: expects an RGB image"));
        };
        let (out_h, out_w) = (plan.resized_height as usize, plan.resized_width as usize);
        let resized;
        let tile = if (out_h, out_w) != (*height, *width) {
            let resample = Resample::Pil(Filter::Bicubic);
            resized = resize_rgb(rgb, *height, *width, out_h, out_w, resample);
            &resized
        } else {
            rgb
        };
        let mut feature = Vec::new();
        feature
            .try_reserve_exact(tile.len())
            .map_err(|_| MmError::limit_exceeded("nemotron_vl: feature allocation failed"))?;
        for (channel, lut) in self.lut.iter().enumerate() {
            feature.extend(
                tile.iter()
                    .skip(channel)
                    .step_by(3)
                    .map(|&value| lut[value as usize]),
            );
        }
        Ok(ProcessedItem {
            modality: Modality::Image,
            feature_token_count: plan.num_image_tokens as usize,
            feature: Tensor {
                shape: vec![1, 3, out_h, out_w],
                data: TensorData::Bf16(feature),
            },
            aux: Vec::new(),
            geometry: None,
        })
    }

    fn expand(&self, input_ids: &[i32], counts: &[usize]) -> Result<TokenLayout> {
        let spec = &self.spec;
        let mut layout = layout_by_placeholder(input_ids, spec.image_token_id, counts)?;
        for segment in &mut layout.segments {
            if let Segment::Media { expansion, .. } = segment {
                expansion.insert(0, ExpansionPart::Literal(vec![spec.image_start_token_id]));
                expansion.push(ExpansionPart::Literal(vec![spec.image_end_token_id]));
            }
        }
        Ok(layout)
    }
}

impl MmFamilyProcessor for NemotronVlProcessor {
    /// One image alone with no text: its feature tokens and the two markers. It is the
    /// engine's span only while the request budget does not bind; the layout is exact.
    fn num_media_tokens(&self, media: &MediaMetadata) -> Result<usize> {
        match media {
            MediaMetadata::Image { width, height } => {
                let plan = self.plan_images(&[], &[(*width, *height)])?[0];
                Ok(plan.num_image_tokens as usize + 2)
            }
            _ => Err(MmError::unsupported("nemotron_vl supports only images")),
        }
    }

    fn layout_from_metadata(
        &self,
        input_ids: &[i32],
        media: &[MediaMetadata],
    ) -> Result<TokenLayout> {
        let sizes = media
            .iter()
            .map(|item| match item {
                MediaMetadata::Image { width, height } => Ok((*width, *height)),
                _ => Err(MmError::unsupported("nemotron_vl supports only images")),
            })
            .collect::<Result<Vec<_>>>()?;
        let counts: Vec<usize> = self
            .plan_images(input_ids, &sizes)?
            .iter()
            .map(|plan| plan.num_image_tokens as usize)
            .collect();
        self.expand(input_ids, &counts)
    }

    fn process_item(&self, _media: &DecodedMedia) -> Result<ProcessedItem> {
        Err(MmError::unsupported(
            "nemotron_vl sizes an image by its whole request; call process_items",
        ))
    }

    fn process_items(
        &self,
        input_ids: &[i32],
        media: &[DecodedMedia],
    ) -> Result<Vec<ProcessedItem>> {
        if self.spec.sizing != NemotronVlSizing::Sglang {
            return Err(MmError::unsupported(
                "nemotron_vl: hf sizing is metadata-only; pixel preprocessing needs sizing = sglang",
            ));
        }
        let placeholders = input_ids
            .iter()
            .filter(|&&id| id == self.spec.image_token_id)
            .count();
        if placeholders != media.len() {
            return Err(MmError::invalid_input(format!(
                "nemotron_vl: prompt has {placeholders} image placeholder(s) but {} image(s)",
                media.len()
            )));
        }
        let sizes = media
            .iter()
            .map(|item| {
                let DecodedMedia::Image { rgb, height, width } = item else {
                    return Err(MmError::unsupported("nemotron_vl: expects an RGB image"));
                };
                let pixels = height.checked_mul(*width).and_then(|n| n.checked_mul(3));
                match (u32::try_from(*width), u32::try_from(*height)) {
                    (Ok(width), Ok(height)) if pixels == Some(rgb.len()) => Ok((width, height)),
                    _ => Err(MmError::invalid_input(
                        "nemotron_vl: invalid RGB dimensions or buffer length",
                    )),
                }
            })
            .collect::<Result<Vec<_>>>()?;
        let plans = self.plan_images(input_ids, &sizes)?;
        media
            .iter()
            .zip(&plans)
            .map(|(item, plan)| self.pixels(item, plan))
            .collect()
    }

    fn layout(&self, input_ids: &[i32], items: &[ProcessedItem]) -> Result<TokenLayout> {
        let counts: Vec<usize> = items.iter().map(|item| item.feature_token_count).collect();
        self.expand(input_ids, &counts)
    }
}

/// Python's `int / int` equals the f64 division only while both ints convert
/// exactly, so larger patch counts fail closed.
fn patch_count(width: u64, height: u64) -> Result<u64> {
    width
        .checked_mul(height)
        .filter(|count| *count < 1 << 53)
        .ok_or_else(|| {
            MmError::limit_exceeded("nemotron_vl: image is too large for exact size planning")
        })
}

/// `torch.Tensor.to(torch.bfloat16)` on a non-NaN f32: round to nearest even.
fn bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_rounds_to_nearest_even() {
        for (f32_bits, expected) in [
            (0x3F80_8000, 0x3F80),
            (0x3F81_8000, 0x3F82),
            (0x3F80_8001, 0x3F81),
            (0x3F80_7FFF, 0x3F80),
            (0xBF80_8000, 0xBF80),
            (0xBF81_8000, 0xBF82),
            (0x0000_0000, 0x0000),
            (0x8000_0000, 0x8000),
            (0x3EAA_AAAB, 0x3EAB),
            (0x7F7F_FFFF, 0x7F80),
        ] {
            assert_eq!(
                bf16_bits(f32::from_bits(f32_bits)),
                expected,
                "{f32_bits:#x}"
            );
        }
    }
}
