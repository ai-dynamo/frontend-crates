// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Size math adapted from SGLang's internvl_utils.py (Apache-2.0), pinned source
// sgl-project/sglang@a2ffd214696ea7144d3c237133a15df3315d900b, and from
// image_processing_nemotron_h_omni.py (Apache-2.0) in
// nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16@fb0ee0bc6b3f70342afd8d84cb80dbadbaa8a1fe, with
// its processing_nemotron_h_omni.py. Video math adapted from SGLang's nano_nemotron_vl.py,
// internvl_utils.py and evs_core.py (Apache-2.0), pinned source
// sgl-project/sglang@25c1c439424d0d781d409c462e944cf6197b250c.

//! Nemotron 3.5 Super VL (`nemotron_h_omni`) images and videos.
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
//! `num_media_tokens` is `Unsupported` with `sglang` sizing, because an image's
//! span depends on the request; with `hf` sizing it is the feature tokens and
//! the two markers.
//!
//! A video's placeholder is `<image>` too, as the model sees it. Each group
//! of `temporal_patch_size` frames expands to its frame label, `<img>`, its
//! `<image>` run and `</img>`, with the groups joined by a newline. The host
//! samples the frames. Labels are text, so a video layout needs the host's
//! encoder: `layout_with_text` for processed items, or
//! [`NemotronVlProcessor::video_layout`] from metadata alone. A request holds
//! images or videos, not both.
//!
//! Audio and the InternVL tile path (`min_num_patches = 0`) are unsupported.

use crate::image::resize::{Filter, Resample, resize_rgb};
use crate::models::qwen_vl::{Resampler, normalize_lut};
use crate::processor::{
    Capabilities, DecodedMedia, ExpansionPart, Geometry, MediaMetadata, MmFamilyProcessor,
    Modality, ProcessedItem, Segment, Tensor, TensorData, TextEncoder, TokenLayout,
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
    /// Absent: images only.
    #[serde(default)]
    pub video: Option<NemotronVlVideoSpec>,
}

/// Video parameters. HF reads them from the processor files, SGLang from
/// `config.json`.
#[derive(Clone, Debug, PartialEq, serde::Deserialize)]
pub struct NemotronVlVideoSpec {
    /// Frames per group: HF's `video_temporal_patch_dim`, SGLang's
    /// `video_temporal_patch_size`. Must be at least 2.
    pub temporal_patch_size: u32,
    /// `video_target_num_patches`. SGLang reads it from `vision_config` in
    /// `config.json`, and with 0 makes each frame one square `image_size` tile.
    pub target_num_patches: u32,
    pub maintain_aspect_ratio: bool,
    /// SGLang's `vision_config.image_size`.
    pub image_size: u32,
    /// SGLang's `video_pruning_rate`. Above 0, the prompt holds only the
    /// feature tokens that EVS keeps; HF writes them all.
    pub pruning_rate: f64,
}

impl NemotronVlVideoSpec {
    pub(crate) fn validate(&self) -> Result<()> {
        if self.temporal_patch_size < 2 {
            return Err(MmError::unsupported(
                "nemotron_vl: video needs temporal_patch_size of at least 2",
            ));
        }
        if !(0.0..1.0).contains(&self.pruning_rate) {
            return Err(MmError::invalid_input(
                "nemotron_vl: pruning_rate must be in [0, 1)",
            ));
        }
        Ok(())
    }
}

/// Resize target and `<image>` count of one image.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NemotronVlImagePlan {
    pub resized_width: u32,
    pub resized_height: u32,
    pub num_image_tokens: u32,
}

/// Resize target of each frame of a video, and the `<image>` count of each
/// group of frames as the prompt holds it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NemotronVlVideoPlan {
    pub resized_width: u32,
    pub resized_height: u32,
    pub group_tokens: Vec<u32>,
}

/// One video of a request, for [`NemotronVlProcessor::video_layout`].
#[derive(Clone, Copy, Debug)]
pub struct NemotronVlVideo<'a> {
    pub width: u32,
    pub height: u32,
    /// Each sampled frame's index in the source video.
    pub frame_indices: &'a [u64],
    /// The source frame rate; 0 when unknown.
    pub fps: f64,
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
/// assert!(family.num_media_tokens(&media[0]).is_err()); // the request decides the span
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
        if let Some(video) = &spec.video {
            video.validate()?;
            if spec.sizing == NemotronVlSizing::Sglang
                && video.target_num_patches == 0
                && video.image_size / spec.patch_size < 2
            {
                return Err(MmError::invalid_input(
                    "nemotron_vl: image_size must hold at least 2 patches per side",
                ));
            }
        }
        // torchvision `ToTensor` then `Normalize`, in f32.
        let lut = core::array::from_fn(|c| {
            normalize_lut(Resampler::Pil, spec.image_mean[c], spec.image_std[c]).map(bf16_bits)
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
    /// `process_items` checks that `rgb` holds `width * height` pixels.
    fn pixels(
        &self,
        rgb: &[u8],
        size: (u32, u32),
        plan: &NemotronVlImagePlan,
    ) -> Result<ProcessedItem> {
        let (out_h, out_w) = (plan.resized_height as usize, plan.resized_width as usize);
        let mut feature = feature_buffer(Some(3 * out_h * out_w))?;
        let out_size = (plan.resized_width, plan.resized_height);
        self.append_pixels(rgb, size, out_size, &mut feature);
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

    /// One RGB frame resized as `resize_image_to_pixels` does, then
    /// `.to(torch.bfloat16)`, appended planar.
    fn append_pixels(
        &self,
        rgb: &[u8],
        (width, height): (u32, u32),
        (out_w, out_h): (u32, u32),
        feature: &mut Vec<u16>,
    ) {
        let (height, width) = (height as usize, width as usize);
        let (out_h, out_w) = (out_h as usize, out_w as usize);
        let resized;
        let tile = if (out_h, out_w) != (height, width) {
            let resample = Resample::Pil(Filter::Bicubic);
            resized = resize_rgb(rgb, height, width, out_h, out_w, resample);
            resized.as_slice()
        } else {
            rgb
        };
        for (channel, lut) in self.lut.iter().enumerate() {
            feature.extend(
                tile.iter()
                    .skip(channel)
                    .step_by(3)
                    .map(|&value| lut[value as usize]),
            );
        }
    }

    fn video(&self) -> Result<&NemotronVlVideoSpec> {
        self.spec
            .video
            .as_ref()
            .ok_or_else(|| MmError::unsupported("nemotron_vl: this spec has no video"))
    }

    /// Frame size and `<image>` counts of a video of `num_frames` sampled
    /// frames, each `width` x `height`.
    pub fn plan_video(
        &self,
        width: u32,
        height: u32,
        num_frames: usize,
    ) -> Result<NemotronVlVideoPlan> {
        let video = self.video()?;
        if width == 0 || height == 0 || num_frames == 0 {
            return Err(MmError::invalid_input(
                "nemotron_vl: a video needs frames with positive dimensions",
            ));
        }
        let target = f64::from(video.target_num_patches);
        let aspect = f64::from(width) / f64::from(height);
        let frame = match self.spec.sizing {
            NemotronVlSizing::Hf => {
                let (w, h) = hf_video_patches(video, aspect);
                self.plan(w, h)?
            }
            NemotronVlSizing::Sglang if video.target_num_patches > 0 => {
                // `get_video_target_size_and_feature_size`.
                let snap = |patches: f64| ((patches / 2.0).round_ties_even() as u64 * 2).max(2);
                let (w, h) = if video.maintain_aspect_ratio {
                    let h = (target / aspect.max(1e-6)).sqrt();
                    (snap(h * aspect), snap(h))
                } else {
                    let side = snap(target.sqrt().trunc());
                    (side, side)
                };
                self.plan(w, h)?
            }
            NemotronVlSizing::Sglang => {
                // One `image_size` tile; the prompt holds `rows * cols` of its tokens.
                let side = u64::from(video.image_size / self.spec.patch_size);
                let rows = (side * side / 4).isqrt();
                NemotronVlImagePlan {
                    resized_width: video.image_size,
                    resized_height: video.image_size,
                    num_image_tokens: u32::try_from(rows * rows).map_err(|_| {
                        MmError::limit_exceeded("nemotron_vl: video token count exceeds u32 range")
                    })?,
                }
            }
        };
        let groups = num_frames.div_ceil(video.temporal_patch_size as usize);
        let per_group = frame.num_image_tokens;
        let limit = || MmError::limit_exceeded("nemotron_vl: video token count exceeds u32 range");
        let total = u64::from(per_group)
            .checked_mul(groups as u64)
            .filter(|total| *total <= u64::from(u32::MAX))
            .ok_or_else(limit)?;
        let mut group_tokens = Vec::new();
        group_tokens
            .try_reserve_exact(groups)
            .map_err(|_| limit())?;
        group_tokens.resize(groups, per_group);
        if self.spec.sizing == NemotronVlSizing::Sglang && video.pruning_rate > 0.0 {
            // `tokens_per_frame`: EVS keeps this many in all, the remainder in the last group.
            let kept =
                ((total as f64 * (1.0 - video.pruning_rate)) as u64).max(u64::from(per_group));
            let base = kept / groups as u64;
            group_tokens.fill(base as u32);
            group_tokens[groups - 1] = (base + kept % groups as u64) as u32;
        }
        Ok(NemotronVlVideoPlan {
            resized_width: frame.resized_width,
            resized_height: frame.resized_height,
            group_tokens,
        })
    }

    /// Each frame's timestamp, `index * int(1000 / fps) / 1000` seconds. With
    /// fps 0, HF prints none and SGLang prints 0.
    fn frame_timestamps(&self, frame_indices: &[u64], fps: f64) -> Result<Vec<Option<f64>>> {
        if !(fps.is_finite() && fps >= 0.0) {
            return Err(MmError::invalid_input(
                "nemotron_vl: video fps must be finite and nonnegative",
            ));
        }
        if fps == 0.0 && self.spec.sizing == NemotronVlSizing::Hf {
            return Ok(vec![None; frame_indices.len()]);
        }
        let exact = |value: f64| value < (1u64 << 53) as f64;
        let ms = if fps > 0.0 {
            (1000.0 / fps).trunc()
        } else {
            0.0
        };
        frame_indices
            .iter()
            .map(|&index| {
                let total = index as f64 * ms;
                if exact(index as f64) && exact(ms) && exact(total) {
                    Ok(Some(total / 1000.0))
                } else {
                    Err(MmError::invalid_input(
                        "nemotron_vl: video frame timestamp is out of range",
                    ))
                }
            })
            .collect()
    }

    /// Per group: the frame label, `<img>`, the `<image>` run, `</img>`.
    fn video_expansion(
        &self,
        tokens: &[u32],
        timestamps: &[Option<f64>],
        text: &dyn TextEncoder,
    ) -> Result<Vec<ExpansionPart>> {
        let spec = &self.spec;
        let group = self.video()?.temporal_patch_size as usize;
        if tokens.len() != timestamps.len().div_ceil(group) {
            return Err(MmError::invalid_input(
                "nemotron_vl: video groups do not match its frames",
            ));
        }
        let mut parts = Vec::with_capacity(4 * tokens.len());
        for (index, &n) in tokens.iter().enumerate() {
            let first = index * group;
            let label = (first..(first + group).min(timestamps.len()))
                .map(|frame| {
                    let prefix = if frame == first { "Frame" } else { "frame" };
                    match timestamps[frame] {
                        Some(seconds) => {
                            format!("{prefix} {} sampled at {seconds:.2} seconds", frame + 1)
                        }
                        None => format!("{prefix} {}", frame + 1),
                    }
                })
                .collect::<Vec<_>>()
                .join(" and ");
            let separator = if index == 0 { "" } else { "\n" };
            parts.push(ExpansionPart::Literal(
                text.encode(&format!("{separator}{label}: "))?,
            ));
            parts.push(ExpansionPart::Literal(vec![spec.image_start_token_id]));
            parts.push(ExpansionPart::Feature {
                id: spec.image_token_id,
                n: n as usize,
            });
            parts.push(ExpansionPart::Literal(vec![spec.image_end_token_id]));
        }
        Ok(parts)
    }

    /// `process_mm_data_async` for one video: `[frames, 3, H, W]` bfloat16.
    fn process_video(
        &self,
        rgb: &[u8],
        (height, width): (usize, usize),
        frame_indices: &[u64],
        fps: f64,
    ) -> Result<ProcessedItem> {
        let invalid =
            || MmError::invalid_input("nemotron_vl: invalid video dimensions or buffer length");
        let num_frames = frame_indices.len();
        let size = (u32::try_from(width), u32::try_from(height));
        let (Ok(w), Ok(h)) = size else {
            return Err(invalid());
        };
        let plan = self.plan_video(w, h, num_frames)?;
        let frame_len = height.checked_mul(width).and_then(|n| n.checked_mul(3));
        let Some(frame_len) = frame_len.filter(|n| n.checked_mul(num_frames) == Some(rgb.len()))
        else {
            return Err(invalid());
        };
        let timestamps = self.frame_timestamps(frame_indices, fps)?;
        let (out_w, out_h) = (plan.resized_width, plan.resized_height);
        let shape = vec![num_frames, 3, out_h as usize, out_w as usize];
        let mut feature = feature_buffer(shape.iter().try_fold(1usize, |n, &d| n.checked_mul(d)))?;
        for frame in rgb.chunks_exact(frame_len) {
            self.append_pixels(frame, (w, h), (out_w, out_h), &mut feature);
        }
        Ok(ProcessedItem {
            modality: Modality::Video,
            feature_token_count: plan.group_tokens.iter().map(|&n| n as usize).sum(),
            feature: Tensor {
                shape,
                data: TensorData::Bf16(feature),
            },
            aux: Vec::new(),
            geometry: Some(Geometry::FrameGroups {
                tokens: plan.group_tokens,
                timestamps: timestamps.into_iter().flatten().collect(),
            }),
        })
    }

    /// The prompt layout of a request whose placeholders are all videos, from
    /// metadata alone, for both sizings.
    pub fn video_layout(
        &self,
        input_ids: &[i32],
        videos: &[NemotronVlVideo<'_>],
        text: &dyn TextEncoder,
    ) -> Result<TokenLayout> {
        let sizes: Vec<_> = (videos.iter())
            .map(|video| (video.width, video.height))
            .collect();
        self.check_video_sizes(&sizes)?;
        let mut counts = Vec::with_capacity(videos.len());
        let mut expansions = Vec::with_capacity(videos.len());
        for video in videos {
            let plan = self.plan_video(video.width, video.height, video.frame_indices.len())?;
            let timestamps = self.frame_timestamps(video.frame_indices, video.fps)?;
            counts.push(plan.group_tokens.iter().map(|&n| n as usize).sum());
            expansions.push(Some(self.video_expansion(
                &plan.group_tokens,
                &timestamps,
                text,
            )?));
        }
        self.expand(input_ids, &counts, expansions)
    }

    /// The HF processor expands only the first video of a prompt, and SGLang
    /// with `target_num_patches` sizes every video by the first one.
    fn check_video_sizes(&self, sizes: &[(u32, u32)]) -> Result<()> {
        let video = self.video()?;
        match self.spec.sizing {
            NemotronVlSizing::Hf if sizes.len() > 1 => Err(MmError::unsupported(
                "nemotron_vl: the HF processor expands only the first video of a prompt",
            )),
            NemotronVlSizing::Sglang
                if video.target_num_patches > 0 && sizes.iter().any(|size| *size != sizes[0]) =>
            {
                Err(MmError::unsupported(
                    "nemotron_vl: SGLang sizes every video by the first one; \
                     videos of different frame sizes are not supported",
                ))
            }
            _ => Ok(()),
        }
    }

    /// Wraps each image in `<img>` and `</img>`; `videos[item]` replaces a video's expansion.
    fn expand(
        &self,
        input_ids: &[i32],
        counts: &[usize],
        mut videos: Vec<Option<Vec<ExpansionPart>>>,
    ) -> Result<TokenLayout> {
        let spec = &self.spec;
        let mut layout = layout_by_placeholder(input_ids, spec.image_token_id, counts)?;
        for segment in &mut layout.segments {
            if let Segment::Media {
                item, expansion, ..
            } = segment
            {
                match videos.get_mut(*item).and_then(Option::take) {
                    Some(video) => *expansion = video,
                    None => {
                        expansion
                            .insert(0, ExpansionPart::Literal(vec![spec.image_start_token_id]));
                        expansion.push(ExpansionPart::Literal(vec![spec.image_end_token_id]));
                    }
                }
            }
        }
        Ok(layout)
    }
}

impl MmFamilyProcessor for NemotronVlProcessor {
    fn capabilities(&self) -> Capabilities {
        Capabilities::new(true, self.spec.video.is_some(), false)
    }

    fn num_media_tokens(&self, media: &MediaMetadata) -> Result<usize> {
        match media {
            MediaMetadata::Image { .. } if self.spec.sizing == NemotronVlSizing::Sglang => {
                Err(MmError::unsupported(
                    "nemotron_vl: with sglang sizing an image's span depends on the request; \
                     use layout_from_metadata",
                ))
            }
            MediaMetadata::Image { width, height } => {
                let plan = self.plan_images(&[], &[(*width, *height)])?[0];
                Ok(plan.num_image_tokens as usize + 2)
            }
            MediaMetadata::Video { .. } => Err(video_needs_labels()),
            _ => Err(MmError::unsupported(
                "nemotron_vl supports only images and videos",
            )),
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
                MediaMetadata::Video { .. } => Err(video_needs_labels()),
                _ => Err(MmError::unsupported(
                    "nemotron_vl supports only images and videos",
                )),
            })
            .collect::<Result<Vec<_>>>()?;
        let counts: Vec<usize> = self
            .plan_images(input_ids, &sizes)?
            .iter()
            .map(|plan| plan.num_image_tokens as usize)
            .collect();
        self.expand(input_ids, &counts, Vec::new())
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
                "nemotron_vl: prompt has {placeholders} image placeholder(s) but {} media item(s)",
                media.len()
            )));
        }
        let videos: Vec<_> = (media.iter())
            .filter_map(|item| match item {
                DecodedMedia::Video {
                    rgb,
                    height,
                    width,
                    frame_indices,
                    fps,
                } => Some((rgb, (*height, *width), frame_indices, *fps)),
                _ => None,
            })
            .collect();
        if !videos.is_empty() {
            if videos.len() < media.len() {
                return Err(MmError::unsupported(
                    "nemotron_vl: a request with both images and videos is not supported",
                ));
            }
            let sizes: Vec<_> = (videos.iter())
                .map(|&(_, (height, width), ..)| (width as u32, height as u32))
                .collect();
            self.check_video_sizes(&sizes)?;
            return (videos.into_iter())
                .map(|(rgb, size, frame_indices, fps)| {
                    self.process_video(rgb, size, frame_indices, fps)
                })
                .collect();
        }
        let images = media
            .iter()
            .map(|item| {
                let DecodedMedia::Image { rgb, height, width } = item else {
                    return Err(MmError::unsupported("nemotron_vl: expects an RGB image"));
                };
                let pixels = height.checked_mul(*width).and_then(|n| n.checked_mul(3));
                match (u32::try_from(*width), u32::try_from(*height)) {
                    (Ok(width), Ok(height)) if pixels == Some(rgb.len()) => {
                        Ok((rgb.as_slice(), (width, height)))
                    }
                    _ => Err(MmError::invalid_input(
                        "nemotron_vl: invalid RGB dimensions or buffer length",
                    )),
                }
            })
            .collect::<Result<Vec<_>>>()?;
        let sizes: Vec<_> = images.iter().map(|&(_, size)| size).collect();
        let plans = self.plan_images(input_ids, &sizes)?;
        images
            .iter()
            .zip(&plans)
            .map(|(&(rgb, size), plan)| self.pixels(rgb, size, plan))
            .collect()
    }

    fn layout(&self, input_ids: &[i32], items: &[ProcessedItem]) -> Result<TokenLayout> {
        if items.iter().any(|item| item.modality == Modality::Video) {
            return Err(MmError::invalid_input(
                "nemotron_vl: a video layout needs the host's text encoder; call layout_with_text",
            ));
        }
        let counts: Vec<usize> = items.iter().map(|item| item.feature_token_count).collect();
        self.expand(input_ids, &counts, Vec::new())
    }

    fn layout_with_text(
        &self,
        input_ids: &[i32],
        items: &[ProcessedItem],
        text: &dyn TextEncoder,
    ) -> Result<TokenLayout> {
        let videos = items
            .iter()
            .map(|item| match &item.geometry {
                Some(Geometry::FrameGroups { tokens, timestamps }) => {
                    let timestamps: Vec<_> = timestamps.iter().copied().map(Some).collect();
                    self.video_expansion(tokens, &timestamps, text).map(Some)
                }
                _ if item.modality == Modality::Video => Err(MmError::invalid_input(
                    "nemotron_vl: a video item needs its frame groups",
                )),
                _ => Ok(None),
            })
            .collect::<Result<Vec<_>>>()?;
        let counts: Vec<usize> = items.iter().map(|item| item.feature_token_count).collect();
        self.expand(input_ids, &counts, videos)
    }
}

fn video_needs_labels() -> MmError {
    MmError::unsupported(
        "nemotron_vl: a video's span holds its frame labels; use NemotronVlProcessor::video_layout",
    )
}

/// `_compute_target_patches_video`: `(w, h)` in patches.
fn hf_video_patches(video: &NemotronVlVideoSpec, aspect: f64) -> (u64, u64) {
    let target = u64::from(video.target_num_patches);
    if !video.maintain_aspect_ratio {
        let side = ((target as f64).sqrt() as u64 / 2 * 2).max(2);
        return (side, side);
    }
    let side = |patches: f64| (patches.sqrt().round_ties_even() as u64).max(1);
    let (w, h) = (side(target as f64 * aspect), side(target as f64 / aspect));
    let up = |patches: u64| patches + patches % 2;
    if up(w).saturating_mul(up(h)) <= target {
        (up(w), up(h))
    } else {
        ((w - w % 2).max(2), (h - h % 2).max(2))
    }
}

fn feature_buffer(len: Option<usize>) -> Result<Vec<u16>> {
    let mut feature = Vec::new();
    len.and_then(|len| feature.try_reserve_exact(len).ok())
        .ok_or_else(|| MmError::limit_exceeded("nemotron_vl: feature allocation failed"))?;
    Ok(feature)
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
