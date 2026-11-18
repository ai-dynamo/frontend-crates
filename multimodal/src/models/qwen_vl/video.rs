// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Qwen3-VL video: HF's `Qwen3VLVideoProcessor` sizing, video frames to
//! patches, and the `<t seconds>` expansion of `Qwen3VLProcessor`. Patchify
//! and M-RoPE are shared with images in the parent module.

use super::{MAX_RATIO, QwenVlProcessor, grid_tensor, patchify};
use crate::image::resize;
use crate::processor::{
    ExpansionPart, Geometry, Modality, ProcessedItem, Tensor, TensorData, TextEncoder,
};
use crate::video::VideoPixelBudget;
use crate::{MmError, Result};

impl QwenVlProcessor {
    /// `Qwen3VLVideoProcessor` with `do_sample_frames=False`, and the
    /// timestamps of `Qwen3VLProcessor._calculate_timestamps`.
    pub(super) fn process_video(
        &self,
        rgb: &[u8],
        (h, w): (usize, usize),
        frame_indices: &[u64],
        fps: f64,
    ) -> Result<ProcessedItem> {
        let (Some(video), Some(lut)) = (&self.spec.video, &self.video_lut) else {
            return Err(MmError::unsupported(
                "qwen_vl: this spec has no video processor",
            ));
        };
        let num_frames = frame_indices.len();
        let frame_len = h.checked_mul(w).and_then(|n| n.checked_mul(3));
        if frame_len.and_then(|n| n.checked_mul(num_frames)) != Some(rgb.len()) {
            return Err(MmError::invalid_input(
                "qwen_vl: video rgb length does not match frames * height * width * 3",
            ));
        }
        if !(fps.is_finite() && fps > 0.0) {
            return Err(MmError::invalid_input(
                "qwen_vl: video fps must be a positive finite number",
            ));
        }
        let sizing = &video.processor;
        let (th, tw) = if video.pre_resized {
            let factor = sizing.patch_size * sizing.merge_size;
            if h == 0 || w == 0 || h % factor != 0 || w % factor != 0 {
                return Err(MmError::invalid_input(format!(
                    "qwen_vl: pre-resized frame sides must be multiples of {factor}, got {h}x{w}"
                )));
            }
            (h, w)
        } else {
            sizing.resize(num_frames, h, w, &VideoPixelBudget::default())?
        };
        let frames: Vec<std::borrow::Cow<[u8]>> = rgb
            .chunks_exact(h * w * 3)
            .map(|frame| {
                if (th, tw) == (h, w) {
                    frame.into()
                } else {
                    resize::resize_rgb(frame, h, w, th, tw, resize::Resample::AtenU8).into()
                }
            })
            .collect();
        let (ps, m, tps) = (
            sizing.patch_size,
            sizing.merge_size,
            sizing.temporal_patch_size,
        );
        let padded = num_frames.div_ceil(tps) * tps;
        let padded_frames: Vec<&[u8]> = (0..padded)
            .map(|t| &*frames[t.min(num_frames - 1)])
            .collect();
        let grid = [(padded / tps) as u32, (th / ps) as u32, (tw / ps) as u32];
        let num_patches = grid.iter().map(|&g| g as usize).product::<usize>();
        Ok(ProcessedItem {
            modality: Modality::Video,
            feature_token_count: num_patches / (m * m),
            feature: Tensor {
                shape: vec![num_patches, 3 * tps * ps * ps],
                data: TensorData::F32(patchify(&padded_frames, (th, tw), (ps, m, tps), lut)),
            },
            aux: vec![grid_tensor("video_grid_thw", grid)],
            geometry: Some(Geometry::TimedGrid {
                grid,
                timestamps: frame_timestamps(frame_indices, fps, tps),
            }),
        })
    }

    /// `Qwen3VLProcessor.replace_video_token`, once per temporal patch.
    pub(super) fn video_expansion(
        &self,
        grid: &[u32; 3],
        timestamps: &[f64],
        text: &dyn TextEncoder,
    ) -> Result<Vec<ExpansionPart>> {
        let Some(video) = &self.spec.video else {
            return Err(MmError::invalid_input(
                "qwen_vl: video item for a spec without video",
            ));
        };
        let m = self.spec.merge_size;
        let per_patch = grid[1] as usize * grid[2] as usize / (m * m);
        let mut parts = Vec::with_capacity(4 * timestamps.len());
        for seconds in timestamps {
            parts.push(ExpansionPart::Literal(
                text.encode(&format!("<{seconds:.1} seconds>"))?,
            ));
            parts.push(ExpansionPart::Literal(vec![video.vision_start_token_id]));
            parts.push(ExpansionPart::Feature {
                id: video.video_token_id,
                n: per_patch,
            });
            parts.push(ExpansionPart::Literal(vec![video.vision_end_token_id]));
        }
        Ok(parts)
    }
}

/// `Qwen3VLProcessor._calculate_timestamps`.
fn frame_timestamps(frame_indices: &[u64], fps: f64, tps: usize) -> Vec<f64> {
    let last = frame_indices.len() - 1;
    let seconds = |i: usize| frame_indices[i.min(last)] as f64 / fps;
    (0..frame_indices.len().div_ceil(tps))
        .map(|g| (seconds(g * tps) + seconds(g * tps + tps - 1)) / 2.0)
        .collect()
}
/// HF's video `smart_resize` (`Qwen3VLVideoProcessor`): like [`smart_resize`](super::smart_resize)
/// but the pixel budget bounds the whole clip, `t_bar * h_bar * w_bar`, where
/// `t_bar` is `num_frames` rounded to a multiple of `temporal_factor`. Frames
/// smaller than `factor` are scaled up first, as HF does.
pub fn smart_video_resize(
    num_frames: usize,
    height: usize,
    width: usize,
    temporal_factor: usize,
    factor: usize,
    min_pixels: usize,
    max_pixels: usize,
) -> Result<(usize, usize)> {
    if factor == 0 || temporal_factor == 0 || min_pixels == 0 || min_pixels > max_pixels {
        return Err(MmError::invalid_input(
            "smart_video_resize: invalid factor or pixel bounds",
        ));
    }
    if factor > u32::MAX as usize || temporal_factor > u32::MAX as usize {
        return Err(MmError::invalid_input(
            "smart_video_resize: factor out of range",
        ));
    }
    if num_frames < temporal_factor {
        return Err(MmError::invalid_input(format!(
            "smart_video_resize: {num_frames} frames is fewer than temporal_factor {temporal_factor}"
        )));
    }
    if height == 0 || width == 0 {
        return Err(MmError::invalid_input("empty video frame"));
    }
    // Frame metadata can be untrusted; keep the u128 volumes below far from
    // overflow (and the f64 conversions exact enough).
    if height > u32::MAX as usize || width > u32::MAX as usize || num_frames > u32::MAX as usize {
        return Err(MmError::invalid_input(
            "smart_video_resize: frame count or dimensions out of range",
        ));
    }
    let f = factor as f64;
    let (mut h, mut w) = (height as f64, width as f64);
    if height < factor || width < factor {
        // `int(height * scale)`: truncate toward zero, as Python does.
        let scale = (f / h).max(f / w);
        h = (h * scale).trunc();
        w = (w * scale).trunc();
    }
    let ratio = h.max(w) / h.min(w);
    if ratio > MAX_RATIO {
        return Err(MmError::invalid_input(format!(
            "absolute aspect ratio must be smaller than {MAX_RATIO}, got {ratio}"
        )));
    }
    let mut h_bar = ((h / f).round_ties_even() * f) as usize;
    let mut w_bar = ((w / f).round_ties_even() * f) as usize;
    let t_bar =
        (num_frames as f64 / temporal_factor as f64).round_ties_even() as usize * temporal_factor;
    let clip = (t_bar as u128)
        .saturating_mul(h_bar as u128)
        .saturating_mul(w_bar as u128);
    // The unrounded clip volume; `h` and `w` are whole numbers here. HF divides
    // exact integers, so past 2^53 the f64 quotients below would round
    // differently: refuse rather than return a size that differs from HF.
    let volume = (num_frames as u128)
        .saturating_mul(h as u128)
        .saturating_mul(w as u128);
    if volume > 1 << 53 {
        return Err(MmError::invalid_input(
            "smart_video_resize: clip volume out of range",
        ));
    }
    if clip > max_pixels as u128 {
        let beta = (volume as f64 / max_pixels as f64).sqrt();
        h_bar = (((h / beta / f).floor() * f) as usize).max(factor);
        w_bar = (((w / beta / f).floor() * f) as usize).max(factor);
    } else if clip < min_pixels as u128 {
        let beta = (min_pixels as f64 / volume as f64).sqrt();
        h_bar = ((h * beta / f).ceil() * f) as usize;
        w_bar = ((w * beta / f).ceil() * f) as usize;
    }
    Ok((h_bar, w_bar))
}
/// The video processor type [`Qwen3VlVideo`] accepts.
const QWEN3_VL_VIDEO_PROCESSOR: &str = "Qwen3VLVideoProcessor";
/// Cosmos3-Edge's video processor: the same sizing, built for one temporal
/// patch (it refuses any other `temporal_patch_size`).
const COSMOS3_EDGE_VIDEO_PROCESSOR: &str = "Cosmos3EdgeVideoProcessor";

/// Video sizing of HF's `Qwen3VLVideoProcessor`, deserialized from the
/// model's `video_preprocessor_config.json` (unknown keys are ignored).
///
/// Deserializing fails unless `video_processor_type` is
/// `Qwen3VLVideoProcessor` (or `Cosmos3EdgeVideoProcessor` with
/// `temporal_patch_size` 1) and `size` gives both bounds. Video is never sized
/// with a model's image bounds, which can differ (Qwen3-VL-8B: 65,536 to
/// 16,777,216 pixels for images, 4,096 to 25,165,824 for a clip), or with the
/// rules of Qwen2-VL and Qwen2.5-VL, which size video differently.
#[derive(Clone, Debug, PartialEq, Eq, serde::Deserialize)]
#[serde(try_from = "RawQwen3VlVideo")]
pub struct Qwen3VlVideo {
    patch_size: usize,
    pub(super) merge_size: usize,
    temporal_patch_size: usize,
    /// `size["shortest_edge"]`: lower bound on the clip's pixels.
    min_pixels: usize,
    /// `size["longest_edge"]`: upper bound on the clip's pixels, and the
    /// ceiling for any request budget.
    max_pixels: usize,
}

/// The wire form of [`Qwen3VlVideo`], before validation.
#[derive(serde::Deserialize)]
struct RawQwen3VlVideo {
    video_processor_type: String,
    patch_size: usize,
    merge_size: usize,
    temporal_patch_size: usize,
    size: RawVideoSize,
    /// Must be absent or `true`: `false` skips resizing in HF.
    #[serde(default)]
    do_resize: Option<bool>,
    /// Must be absent or `false`: `true` caps frames at `max_video_tokens`,
    /// which this crate does not implement.
    #[serde(default)]
    cap_pixels_per_frame: Option<bool>,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct RawVideoSize {
    shortest_edge: usize,
    longest_edge: usize,
}

impl TryFrom<RawQwen3VlVideo> for Qwen3VlVideo {
    type Error = MmError;

    fn try_from(raw: RawQwen3VlVideo) -> Result<Self> {
        match raw.video_processor_type.as_str() {
            QWEN3_VL_VIDEO_PROCESSOR => {}
            COSMOS3_EDGE_VIDEO_PROCESSOR if raw.temporal_patch_size == 1 => {}
            other => {
                return Err(MmError::unsupported(format!(
                    "video processor {other} is not {QWEN3_VL_VIDEO_PROCESSOR} \
                     (or {COSMOS3_EDGE_VIDEO_PROCESSOR} with temporal_patch_size 1)"
                )));
            }
        }
        if raw.do_resize == Some(false) || raw.cap_pixels_per_frame == Some(true) {
            return Err(MmError::unsupported(
                "video processor configs with do_resize = false or cap_pixels_per_frame = true are not supported",
            ));
        }
        if raw.patch_size == 0 || raw.merge_size == 0 || raw.temporal_patch_size == 0 {
            return Err(MmError::invalid_input(
                "video processor patch, merge and temporal patch sizes must be at least 1",
            ));
        }
        // `smart_video_resize` refuses factors above u32::MAX; refuse them here
        // so a config that loads can always be resized.
        let factor = raw.patch_size.checked_mul(raw.merge_size);
        if factor.is_none_or(|f| f > u32::MAX as usize)
            || raw.temporal_patch_size > u32::MAX as usize
        {
            return Err(MmError::invalid_input(
                "video processor patch_size * merge_size and temporal_patch_size must fit in 32 bits",
            ));
        }
        let size = raw.size;
        if size.shortest_edge == 0 || size.shortest_edge > size.longest_edge {
            return Err(MmError::invalid_input(format!(
                "video processor size must have 0 < shortest_edge <= longest_edge, got {} and {}",
                size.shortest_edge, size.longest_edge
            )));
        }
        Ok(Self {
            patch_size: raw.patch_size,
            merge_size: raw.merge_size,
            temporal_patch_size: raw.temporal_patch_size,
            min_pixels: size.shortest_edge,
            max_pixels: size.longest_edge,
        })
    }
}

impl Qwen3VlVideo {
    /// The sizing of a `Qwen3VLVideoProcessor` with these values, checked as
    /// its deserialized config is (`min_pixels` and `max_pixels` are
    /// `size["shortest_edge"]` and `size["longest_edge"]`).
    pub fn new(
        patch_size: usize,
        merge_size: usize,
        temporal_patch_size: usize,
        min_pixels: usize,
        max_pixels: usize,
    ) -> Result<Self> {
        Self::try_from(RawQwen3VlVideo {
            video_processor_type: QWEN3_VL_VIDEO_PROCESSOR.to_string(),
            patch_size,
            merge_size,
            temporal_patch_size,
            size: RawVideoSize {
                shortest_edge: min_pixels,
                longest_edge: max_pixels,
            },
            do_resize: None,
            cap_pixels_per_frame: None,
        })
    }

    /// Target `(height, width)` for each of `num_frames` sampled frames, as
    /// HF's `Qwen3VLVideoProcessor` computes it.
    ///
    /// A request can only tighten the model's clip budget:
    /// `budget.total_pixels` is clamped to the model's maximum, so a payload
    /// cannot raise memory use above what the model config allows. The
    /// model's minimum is lowered to the effective budget when a small
    /// request would otherwise invert the bounds (HF does not validate this).
    /// `max_pixels_per_frame` follows HF's `cap_pixels_per_frame`: a frame gets
    /// at most the cap or its even share of the clip budget, whichever is
    /// smaller, but never less than `1.05 * min_pixels`, so the cap can be
    /// exceeded when it is below that floor. Unlike HF, the resulting clip
    /// budget is clamped to the model's maximum: with many frames, HF's floor
    /// can push it past `longest_edge`.
    ///
    /// The budget is the target of HF's algorithm, not a hard limit, and HF's
    /// result can exceed it: dimensions round to the nearest multiple of the
    /// factor, the frame count rounds to the temporal factor, a side never goes
    /// below the factor (thin frames), and too-small clips round up to the
    /// minimum. A budget under one `factor`-sized cell per frame is rejected.
    pub fn resize(
        &self,
        num_frames: usize,
        height: usize,
        width: usize,
        budget: &VideoPixelBudget,
    ) -> Result<(usize, usize)> {
        budget.validate()?;
        if num_frames == 0 {
            return Err(MmError::invalid_input("video has no sampled frames"));
        }
        // Checked for overflow when the config was read.
        let factor = self.patch_size * self.merge_size;
        let mut max_pixels = budget
            .total_pixels
            .map_or(self.max_pixels, |t| t.min(self.max_pixels));
        let cell = factor.saturating_mul(factor).saturating_mul(num_frames);
        if max_pixels < cell {
            return Err(MmError::invalid_input(format!(
                "video pixel budget {max_pixels} is below one {factor}x{factor} cell per frame for {num_frames} frames"
            )));
        }
        let min_pixels = self.min_pixels.min(max_pixels);
        if let Some(cap) = budget.max_pixels_per_frame {
            let floor = (min_pixels as f64 * 1.05) as usize;
            let per_frame = cap.min(max_pixels / num_frames).max(floor);
            // HF lets the floor push the clip past `longest_edge`; a request
            // never raises the model's maximum, so clamp it back.
            max_pixels = per_frame.saturating_mul(num_frames).min(self.max_pixels);
        }
        smart_video_resize(
            num_frames,
            height,
            width,
            self.temporal_patch_size,
            factor,
            min_pixels,
            max_pixels,
        )
    }
}

/// [`Qwen3VlVideo`] as the decoder's [`crate::video_decode::Fit`]: HF's
/// `Qwen3VLVideoProcessor` sizing under a request budget, resampled as the
/// torchvision-backed video processors do (bicubic on uint8 frames, before
/// rescaling). Cosmos3-Nano / Super ship that processor unchanged, and
/// Cosmos3-Edge the same sizing. Set
/// [`QwenVlVideoSpec::pre_resized`](super::QwenVlVideoSpec::pre_resized)
/// for its frames: HF resizes once, and a second resize under the model's
/// default budget can change the size.
#[cfg(feature = "video-decode")]
#[derive(Clone, Copy, Debug)]
pub struct Qwen3VlVideoFit<'a> {
    pub video: &'a Qwen3VlVideo,
    pub budget: VideoPixelBudget,
}

#[cfg(feature = "video-decode")]
impl crate::video_decode::Fit for Qwen3VlVideoFit<'_> {
    fn fit(&self, num_frames: usize, width: u32, height: u32) -> Result<(u32, u32)> {
        let (h, w) =
            self.video
                .resize(num_frames, height as usize, width as usize, &self.budget)?;
        let side = |v: usize| {
            u32::try_from(v).map_err(|_| MmError::invalid_input("resized frame side out of range"))
        };
        Ok((side(w)?, side(h)?))
    }

    fn filter(&self) -> resize::Resample {
        resize::Resample::AtenU8
    }
}
#[cfg(test)]
mod tests {
    use super::super::tests::tiny_spec;
    use super::super::{QwenVlSpec, QwenVlVideoSpec};
    use super::*;
    use crate::processor::{DecodedMedia, MmFamilyProcessor};

    /// Qwen3-VL-8B-Instruct's `video_preprocessor_config.json`, verbatim.
    const QWEN3_VL_8B_VIDEO_CONFIG: &str = r#"{
        "size": {"longest_edge": 25165824, "shortest_edge": 4096},
        "patch_size": 16,
        "temporal_patch_size": 2,
        "merge_size": 2,
        "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.5, 0.5, 0.5],
        "processor_class": "Qwen3VLProcessor",
        "video_processor_type": "Qwen3VLVideoProcessor"
    }"#;

    /// Qwen3-VL's factors with the given clip bounds.
    fn qwen3_video(min_pixels: usize, max_pixels: usize) -> Qwen3VlVideo {
        Qwen3VlVideo {
            patch_size: 16,
            merge_size: 2,
            temporal_patch_size: 2,
            min_pixels,
            max_pixels,
        }
    }

    fn default_video() -> Qwen3VlVideo {
        qwen3_video(65536, 16_777_216)
    }

    #[test]
    fn video_bounds_come_from_the_video_processor_config() {
        let video: Qwen3VlVideo = serde_json::from_str(QWEN3_VL_8B_VIDEO_CONFIG).unwrap();
        assert_eq!(video, qwen3_video(4096, 25_165_824));
        // HF's smart_resize for these bounds; the model's image bounds
        // (65,536 to 16,777,216) would give (512, 960) instead.
        let none = VideoPixelBudget::default();
        assert_eq!(video.resize(32, 1080, 1920, &none).unwrap(), (640, 1152));
    }

    #[test]
    fn other_video_processors_and_incomplete_configs_are_refused() {
        let config: serde_json::Value = serde_json::from_str(QWEN3_VL_8B_VIDEO_CONFIG).unwrap();
        let with = |key: &str, value: serde_json::Value| {
            let mut c = config.clone();
            c[key] = value;
            c
        };
        let without = |key: &str| {
            let mut c = config.clone();
            c.as_object_mut().unwrap().remove(key);
            c
        };
        let refused = [
            // Another generation's processor, and an image processor config.
            with("video_processor_type", "Qwen2VLVideoProcessor".into()),
            without("video_processor_type"),
            without("size"),
            with("size", serde_json::json!({"longest_edge": 25165824})),
            with(
                "size",
                serde_json::json!({"longest_edge": 4096, "shortest_edge": 8192}),
            ),
            with(
                "size",
                serde_json::json!({"longest_edge": 4096, "shortest_edge": 0}),
            ),
            with("merge_size", 0.into()),
            with("patch_size", usize::MAX.into()),
            // No overflow, but past the 32 bits smart_video_resize accepts:
            // the config would load and then never resize.
            with("patch_size", (1u64 << 31).into()),
            with("temporal_patch_size", (1u64 << 32).into()),
            // Settings that change HF's sizing in ways not implemented here.
            with("do_resize", false.into()),
            with("cap_pixels_per_frame", true.into()),
            with(
                "size",
                serde_json::json!({"longest_edge": 25165824, "shortest_edge": 4096, "max_pixels": 1}),
            ),
        ];
        for c in refused {
            assert!(
                serde_json::from_value::<Qwen3VlVideo>(c.clone()).is_err(),
                "{c}"
            );
        }
        // Their defaults, spelled out, are fine.
        let mut explicit = with("do_resize", true.into());
        explicit["cap_pixels_per_frame"] = false.into();
        assert!(serde_json::from_value::<Qwen3VlVideo>(explicit).is_ok());
        let r = serde_json::from_value::<Qwen3VlVideo>(with(
            "video_processor_type",
            "Qwen2VLVideoProcessor".into(),
        ));
        assert!(r.is_err_and(|e| e.to_string().contains("is not Qwen3VLVideoProcessor")));
    }

    #[test]
    fn cosmos3_edge_video_processor_is_the_same_sizing_with_one_temporal_patch() {
        // Cosmos3-Edge's video_preprocessor_config.json: HF's
        // Cosmos3EdgeVideoProcessor runs Qwen3-VL's smart_resize with
        // temporal_patch_size 1 (it refuses any other value).
        let edge: serde_json::Value = serde_json::json!({
            "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5],
            "merge_size": 2, "patch_size": 16, "processor_class": "Cosmos3EdgeProcessor",
            "return_metadata": true,
            "size": {"longest_edge": 25165824, "shortest_edge": 4096},
            "temporal_patch_size": 1, "video_processor_type": "Cosmos3EdgeVideoProcessor"
        });
        let video: Qwen3VlVideo = serde_json::from_value(edge.clone()).unwrap();
        assert_eq!(video.temporal_patch_size, 1);
        // 20 frames of 720p: 1280x704 as HF's formula gives (t_bar = 20).
        let none = VideoPixelBudget::default();
        assert_eq!(video.resize(20, 720, 1280, &none).unwrap(), (704, 1280));
        let mut two = edge;
        two["temporal_patch_size"] = 2.into();
        assert!(serde_json::from_value::<Qwen3VlVideo>(two).is_err());
    }

    #[test]
    fn video_request_budget_cannot_exceed_the_model_maximum() {
        let video = default_video();
        let huge = VideoPixelBudget {
            total_pixels: Some(usize::MAX),
            max_pixels_per_frame: None,
        };
        let base = video
            .resize(32, 2160, 3840, &VideoPixelBudget::default())
            .unwrap();
        assert_eq!(video.resize(32, 2160, 3840, &huge).unwrap(), base);
    }

    #[test]
    fn video_budget_below_one_cell_per_frame_is_rejected() {
        let video = default_video();
        let tiny = VideoPixelBudget {
            total_pixels: Some(1_000),
            max_pixels_per_frame: Some(10),
        };
        assert!(video.resize(8, 1080, 1920, &tiny).is_err());
        // Exactly one 32x32 cell per frame is the smallest accepted budget.
        let edge = VideoPixelBudget {
            total_pixels: Some(8 * 32 * 32),
            max_pixels_per_frame: None,
        };
        assert_eq!(video.resize(8, 1080, 1920, &edge).unwrap(), (32, 32));
    }

    #[test]
    fn video_budget_is_a_target_that_hf_rounding_can_overshoot() {
        // Pinned from HF's own function (see tests/video_golden.rs): the budget
        // steers the algorithm but is not enforced afterwards.
        let video = default_video();
        let one_point_three = VideoPixelBudget {
            total_pixels: Some(1_300_000),
            max_pixels_per_frame: None,
        };
        // 5 frames round to t_bar = 4, so 4 * 480 * 640 fits; the real clip is 5 frames.
        let (h, w) = video.resize(5, 480, 640, &one_point_three).unwrap();
        assert_eq!((h, w), (480, 640));
        assert!(5 * h * w > 1_300_000);
    }

    #[test]
    fn video_request_total_below_the_minimum_lowers_the_minimum() {
        // A 3 MP budget under a 4 MP minimum would invert the bounds; the
        // minimum drops to 3 MP. HF's smart_resize with min = max = 3,000,000
        // gives (640, 1152).
        let video = qwen3_video(4_000_000, 16_777_216);
        let budget = VideoPixelBudget {
            total_pixels: Some(3_000_000),
            max_pixels_per_frame: None,
        };
        assert_eq!(video.resize(4, 1080, 1920, &budget).unwrap(), (640, 1152));
        // Small frames take the scale-up branch, which reads the minimum:
        // HF gives (768, 1024) for a 3,000,000 minimum, (256, 320) for 1.
        assert_eq!(video.resize(4, 240, 320, &budget).unwrap(), (768, 1024));
    }

    #[test]
    fn a_per_frame_cap_never_raises_the_clip_past_the_model_maximum() {
        // 18,000 frames: HF's 1.05 * min floor (4,300 px a frame) times the
        // frame count is about 77M, three times Qwen3-VL-8B's 25,165,824.
        // Clamped, the cap changes nothing; HF's own result for these bounds
        // without a cap is (32, 32).
        let video: Qwen3VlVideo = serde_json::from_str(QWEN3_VL_8B_VIDEO_CONFIG).unwrap();
        let capped = VideoPixelBudget {
            total_pixels: None,
            max_pixels_per_frame: Some(1_000),
        };
        assert_eq!(video.resize(18_000, 1080, 1920, &capped).unwrap(), (32, 32));
    }

    #[test]
    fn video_resize_rejects_empty_input_and_bad_budgets() {
        let video = default_video();
        let none = VideoPixelBudget::default();
        assert!(video.resize(0, 480, 640, &none).is_err());
        assert!(video.resize(8, 0, 640, &none).is_err());
        let zero = VideoPixelBudget {
            total_pixels: Some(0),
            max_pixels_per_frame: None,
        };
        assert!(video.resize(8, 480, 640, &zero).is_err());
    }

    #[test]
    fn video_sampling_and_resize_compose() {
        // fps=0.1 on 5 s samples 1 frame (floor 1); HF's smart_resize needs at
        // least `temporal_factor` frames, so the pair errors cleanly instead of
        // producing a size for a clip the processor cannot patchify.
        use crate::video::{VideoOptions, resolve_num_frames};
        let options = VideoOptions {
            fps: Some(0.1),
            ..Default::default()
        };
        let n = resolve_num_frames(&options, 5.0, 150).unwrap();
        assert_eq!(n, 1);
        let none = VideoPixelBudget::default();
        assert!(default_video().resize(n as usize, 480, 640, &none).is_err());
    }

    #[test]
    fn smart_video_resize_rejects_out_of_range_input() {
        assert!(smart_video_resize(8, usize::MAX, 640, 2, 32, 1, 1 << 30).is_err());
        assert!(smart_video_resize(usize::MAX, 480, 640, 2, 32, 1, 1 << 30).is_err());
        // An enormous factor must not overflow the volume arithmetic.
        assert!(smart_video_resize(2, 1, 1, 2, usize::MAX, 1, usize::MAX).is_err());
        assert!(smart_video_resize(2, 1, 1, 2, u32::MAX as usize, 1, usize::MAX).is_err());
    }

    #[test]
    fn smart_video_resize_rejects_volumes_past_exact_f64_division() {
        // HF divides exact integers; beyond 2^53 f64 quotients can differ, so
        // this input (HF returns (60_000_000, 60_000_000)) is refused.
        assert!(
            smart_video_resize(3, 60_000_000, 60_000_000, 2, 32, 1, 10_799_999_999_999_997)
                .is_err()
        );
    }

    fn video_spec() -> QwenVlSpec {
        QwenVlSpec {
            video: Some(QwenVlVideoSpec {
                video_token_id: 2,
                vision_start_token_id: 3,
                vision_end_token_id: 4,
                processor: Qwen3VlVideo::new(2, 2, 2, 16, 1 << 20).unwrap(),
                pre_resized: false,
                image_mean: [0.0; 3],
                image_std: [1.0; 3],
            }),
            ..tiny_spec()
        }
    }

    fn video(frame_indices: Vec<u64>, fps: f64) -> DecodedMedia {
        DecodedMedia::Video {
            rgb: vec![9; frame_indices.len() * 4 * 4 * 3],
            height: 4,
            width: 4,
            frame_indices,
            fps,
        }
    }

    struct Bytes;

    impl TextEncoder for Bytes {
        fn encode(&self, text: &str) -> Result<Vec<i32>> {
            Ok(text.bytes().map(i32::from).collect())
        }
    }

    #[test]
    fn video_needs_a_video_spec_and_consistent_frames() {
        let images_only = QwenVlProcessor::new(tiny_spec()).unwrap();
        assert!(matches!(
            images_only.process_item(&video(vec![0, 1], 2.0)),
            Err(MmError::Unsupported { .. })
        ));

        let proc = QwenVlProcessor::new(video_spec()).unwrap();
        assert!(proc.capabilities().supports(Modality::Video));
        let short = DecodedMedia::Video {
            rgb: vec![9; 2 * 4 * 4 * 3 - 1],
            height: 4,
            width: 4,
            frame_indices: vec![0, 1],
            fps: 2.0,
        };
        for bad in [
            short,
            video(vec![], 2.0),
            video(vec![0, 1], 0.0),
            video(vec![0, 1], f64::NAN),
        ] {
            assert!(proc.process_item(&bad).is_err());
        }

        let mut spec = video_spec();
        spec.video.as_mut().unwrap().processor = Qwen3VlVideo::new(2, 1, 2, 16, 1 << 20).unwrap();
        assert!(QwenVlProcessor::new(spec).is_err(), "merge sizes differ");
    }

    /// HF's `do_resize=False`: two 32x32 frames under Qwen3-VL's 4,096-pixel
    /// clip minimum keep their size instead of growing to 64x64.
    #[test]
    fn pre_resized_frames_keep_their_size() {
        let mut spec = video_spec();
        spec.video.as_mut().unwrap().processor = qwen3_video(4096, 25_165_824);
        let frames = |side: usize| DecodedMedia::Video {
            rgb: vec![9; 2 * side * side * 3],
            height: side,
            width: side,
            frame_indices: vec![0, 1],
            fps: 2.0,
        };
        let grid = |spec: &QwenVlSpec, side: usize| {
            let item = QwenVlProcessor::new(spec.clone())
                .unwrap()
                .process_item(&frames(side))?;
            match item.geometry {
                Some(Geometry::TimedGrid { grid, .. }) => Ok(grid),
                _ => unreachable!(),
            }
        };
        assert_eq!(grid(&spec, 32).unwrap(), [1, 4, 4]);
        spec.video.as_mut().unwrap().pre_resized = true;
        assert_eq!(grid(&spec, 32).unwrap(), [1, 2, 2]);
        assert!(matches!(grid(&spec, 40), Err(MmError::InvalidInput { .. })));
    }

    /// A video layout needs the host's text encoder; items must come in
    /// placeholder order; a video's M-RoPE needs one nonempty range per
    /// temporal patch.
    #[test]
    fn video_layout_and_positions_refuse_malformed_input() {
        let proc = QwenVlProcessor::new(video_spec()).unwrap();
        let image = || {
            proc.process_item(&DecodedMedia::Image {
                rgb: vec![9; 4 * 4 * 3],
                height: 4,
                width: 4,
            })
            .unwrap()
        };
        let clip = || proc.process_item(&video(vec![0, 2, 4], 2.0)).unwrap();
        let prompt = [7, 1, 8, 2, 9];
        let items = [image(), clip()];
        assert_eq!(
            proc.placeholder_modalities(&prompt).unwrap(),
            [Modality::Image, Modality::Video]
        );
        assert!(proc.layout(&prompt, &items).is_err(), "no text encoder");
        assert!(
            proc.layout_with_text(&prompt, &[clip(), image()], &Bytes)
                .is_err()
        );
        let positions = |runs: &[&[(u32, u32)]]| {
            let ranges: Vec<Vec<_>> = (runs.iter())
                .map(|run| run.iter().map(|&(start, end)| start..end).collect())
                .collect();
            proc.positions_with_features(9, &[], &ranges, &items)
        };
        assert!(
            positions(&[&[(1, 2)], &[(3, 4)]]).is_err(),
            "one patch range"
        );
        assert!(
            positions(&[&[(1, 2)], &[(3, 4), (6, 6)]]).is_err(),
            "empty range"
        );
        assert!(positions(&[&[(1, 2)], &[(3, 4), (6, 7)]]).is_ok());
    }
}
