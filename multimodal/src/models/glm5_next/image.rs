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
use crate::models::qwen_vl::{MropeItem, Resampler, mrope_image_only, normalize_lut};
use crate::processor::{
    Capabilities, DecodedMedia, ExpansionPart, Geometry, MediaMetadata, MmFamilyProcessor,
    Modality, PositionOutput, ProcessedItem, Segment, Tensor, TensorData, TextEncoder, TokenLayout,
};

/// Full image pipeline parameters, including the shared metadata planner.
#[derive(Clone, Debug, PartialEq, serde::Deserialize)]
pub struct Glm5NextImageSpec {
    #[serde(flatten)]
    pub planner: Glm5NextSpec,
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
    /// PIL preprocessing requires the planner's `patch_expand_factor` to be 1.
    #[serde(default)]
    pub resample: Resampler,
}

pub struct Glm5NextImageProcessor {
    planner: Glm5NextProcessor,
    pub(super) spec: Glm5NextImageSpec,
    /// Per-channel u8 to normalized f32, as Dynamo's Qwen-VL `normalize_lut`.
    lut: [[f32; 256]; 3],
    /// The video processor's lookup; HF's video processor is torchvision-backed.
    pub(super) video_lut: Option<[[f32; 256]; 3]>,
}

fn check_mean_std(mean: &[f32; 3], std: &[f32; 3]) -> Result<(), MmError> {
    if std.iter().any(|std| !std.is_finite() || *std <= 0.0) {
        return Err(MmError::invalid_input(
            "glm5_next: image_std must be positive",
        ));
    }
    if mean.iter().any(|mean| !mean.is_finite()) {
        return Err(MmError::invalid_input(
            "glm5_next: image_mean must be finite",
        ));
    }
    Ok(())
}

impl Glm5NextImageProcessor {
    pub fn new(spec: Glm5NextImageSpec) -> Result<Self, MmError> {
        let planner = Glm5NextProcessor::new(spec.planner.clone())?;
        // The pinned HF PIL `_preprocess` ignores `patch_expand_factor` when
        // calling resize. Refuse specs that would disagree with the shared
        // metadata planner's canvas and token count.
        if spec.resample == Resampler::Pil && spec.planner.geometry.patch_expand_factor != 1 {
            return Err(MmError::unsupported(
                "glm5_next: PIL preprocessing requires patch_expand_factor = 1 to match the shared planner",
            ));
        }
        check_mean_std(&spec.image_mean, &spec.image_std)?;
        let video_lut = match &spec.planner.video {
            Some(video) => {
                check_mean_std(&video.image_mean, &video.image_std)?;
                Some(core::array::from_fn(|c| {
                    normalize_lut(Resampler::AtenU8, video.image_mean[c], video.image_std[c])
                }))
            }
            None => None,
        };
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
        Ok(Self {
            planner,
            spec,
            lut,
            video_lut,
        })
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
    fn capabilities(&self) -> Capabilities {
        Capabilities::new(true, self.spec.planner.video.is_some(), false)
    }

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

    fn process_items(
        &self,
        _input_ids: &[i32],
        media: &[DecodedMedia],
    ) -> Result<Vec<ProcessedItem>, MmError> {
        let videos = media
            .iter()
            .filter(|item| item.modality() == Modality::Video)
            .count();
        media
            .iter()
            .map(|item| self.process_media(item, videos))
            .collect()
    }

    fn process_item(&self, media: &DecodedMedia) -> Result<ProcessedItem, MmError> {
        if media.modality() == Modality::Video {
            return Err(MmError::unsupported(
                "glm5_next sizes a video by its request; call process_items",
            ));
        }
        self.process_media(media, 1)
    }

    fn layout(&self, input_ids: &[i32], items: &[ProcessedItem]) -> Result<TokenLayout, MmError> {
        self.layout_with_text(input_ids, items, &NoTextEncoder)
    }

    fn placeholder_modalities(&self, input_ids: &[i32]) -> Result<Vec<Modality>, MmError> {
        Ok(input_ids
            .iter()
            .filter_map(|&id| self.placeholder_modality(id))
            .collect())
    }

    fn layout_with_text(
        &self,
        input_ids: &[i32],
        items: &[ProcessedItem],
        text: &dyn TextEncoder,
    ) -> Result<TokenLayout, MmError> {
        let planner = &self.spec.planner;
        if planner.video.is_none()
            && planner
                .video_token_id
                .is_some_and(|id| input_ids.contains(&id))
        {
            return Err(MmError::unsupported("glm5_next supports only images"));
        }
        let found = input_ids
            .iter()
            .filter(|&&id| self.placeholder_modality(id).is_some())
            .count();
        if found != items.len() {
            return Err(MmError::invalid_input(format!(
                "glm5_next: prompt has {found} media placeholder(s) but {} media item(s)",
                items.len()
            )));
        }
        let mut segments = Vec::new();
        let (mut text_start, mut next) = (0, 0);
        for (pos, &id) in input_ids.iter().enumerate() {
            let Some(modality) = self.placeholder_modality(id) else {
                continue;
            };
            let item = &items[next];
            if item.modality != modality {
                return Err(MmError::invalid_input(format!(
                    "glm5_next: media placeholder {next} is {modality:?} but its item is {:?}",
                    item.modality
                )));
            }
            if text_start < pos {
                segments.push(Segment::Text(text_start..pos));
            }
            let expansion = match &item.geometry {
                Some(Geometry::TimedGrid { grid, timestamps }) => {
                    self.video_expansion(grid, timestamps, text)?
                }
                _ => vec![ExpansionPart::Feature {
                    id,
                    n: item.feature_token_count,
                }],
            };
            segments.push(Segment::Media {
                item: next,
                src: pos..pos + 1,
                expansion,
            });
            next += 1;
            text_start = pos + 1;
        }
        if text_start < input_ids.len() {
            segments.push(Segment::Text(text_start..input_ids.len()));
        }
        Ok(TokenLayout { segments })
    }

    /// `get_rope_index_glm4v`: an image is one grid, and each temporal group
    /// of a video is a `t = 1` grid with its timestamp text between groups.
    fn positions_with_features(
        &self,
        input_len: usize,
        _offsets: &[(u32, u32)],
        feature_ranges: &[Vec<std::ops::Range<u32>>],
        items: &[ProcessedItem],
    ) -> Result<PositionOutput, MmError> {
        if feature_ranges.len() != items.len() {
            return Err(MmError::invalid_input(
                "glm5_next: feature range and item counts differ",
            ));
        }
        let mut mrope_items = Vec::new();
        for (ranges, item) in feature_ranges.iter().zip(items) {
            let (grid, runs) = match &item.geometry {
                Some(Geometry::Grid(grid)) => (*grid, 1),
                Some(Geometry::TimedGrid { grid, .. }) => ([1, grid[1], grid[2]], grid[0] as usize),
                _ => return Err(MmError::invalid_input("glm5_next: item without its grid")),
            };
            if ranges.len() != runs || ranges.iter().any(|range| range.is_empty()) {
                return Err(MmError::invalid_input(format!(
                    "glm5_next: an item needs {runs} nonempty feature range(s), got {ranges:?}"
                )));
            }
            mrope_items.extend(ranges.iter().map(|range| MropeItem {
                start: range.start,
                end: range.end - 1,
                grid,
            }));
        }
        let merge_size = self.spec.planner.geometry.merge_size as usize;
        let (positions, delta) = mrope_image_only(input_len, &mrope_items, merge_size)?;
        Ok(PositionOutput::MRope { positions, delta })
    }

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
        let ranges: Vec<_> = offsets
            .iter()
            .map(|&(start, end)| std::iter::once(start..end.saturating_add(1)).collect())
            .collect();
        self.positions_with_features(input_len, offsets, &ranges, items)
    }
}

impl Glm5NextImageProcessor {
    fn placeholder_modality(&self, id: i32) -> Option<Modality> {
        let planner = &self.spec.planner;
        if id == planner.image_token_id {
            Some(Modality::Image)
        } else if planner.video_token_id == Some(id) {
            Some(Modality::Video)
        } else {
            None
        }
    }

    /// `videos` counts the request's videos, which split the video budget.
    fn process_media(&self, media: &DecodedMedia, videos: usize) -> Result<ProcessedItem, MmError> {
        if let DecodedMedia::Video {
            rgb,
            height,
            width,
            frame_indices,
            fps,
        } = media
        {
            return self.process_video(rgb, (*height, *width), frame_indices, *fps, videos);
        }
        let DecodedMedia::Image { rgb, height, width } = media else {
            return Err(MmError::unsupported(
                "glm5_next: expects an RGB image or video",
            ));
        };
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
}

/// The encoder behind [`Glm5NextImageProcessor`]'s `layout`: only a video's
/// expansion needs one.
struct NoTextEncoder;

impl TextEncoder for NoTextEncoder {
    fn encode(&self, _text: &str) -> crate::Result<Vec<i32>> {
        Err(MmError::invalid_input(
            "glm5_next: video layout needs a text encoder (layout_with_text)",
        ))
    }
}
