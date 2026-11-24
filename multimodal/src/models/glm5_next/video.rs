// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! GLM-5.3-Flash video as SGLang's GLM processor runs it: one canvas for the
//! clip, every frame stretched to it by torchvision v1's bicubic resize
//! (SGLang's `resize_mode` "resize"), then
//! `Glm5NextVideoProcessor`'s normalize and patch order, and
//! `Glm5NextProcessor`'s expansion of each temporal group into image tokens and
//! a timestamp.

use super::{Glm5NextVideoSpec, plan_video_grid};
use crate::image::resize::{Resample, resize_rgb};
use crate::models::qwen_vl::patchify;
use crate::processor::{
    ExpansionPart, Geometry, Modality, ProcessedItem, Tensor, TensorData, TextEncoder,
};
use crate::{MmError, Result};

/// What the video path reads from the pixel processor.
pub(super) struct VideoStage<'a> {
    pub(super) video: &'a Glm5NextVideoSpec,
    pub(super) lut: &'a [[f32; 256]; 3],
    pub(super) image_token_id: i32,
}

/// `videos` is the request's video count: SGLang gives each video an equal
/// share of `max_image_tokens` when there are several.
pub(super) fn process(
    stage: &VideoStage<'_>,
    rgb: &[u8],
    (h, w): (usize, usize),
    frame_indices: &[u64],
    fps: f64,
    videos: usize,
) -> Result<ProcessedItem> {
    let (video, lut) = (stage.video, stage.lut);
    let num_frames = frame_indices.len();
    if num_frames == 0 {
        return Err(MmError::invalid_input(
            "glm5_next: video has no sampled frames",
        ));
    }
    let frame_len = h.checked_mul(w).and_then(|n| n.checked_mul(3));
    if h == 0 || w == 0 || frame_len.and_then(|n| n.checked_mul(num_frames)) != Some(rgb.len()) {
        return Err(MmError::invalid_input(
            "glm5_next: video rgb length does not match frames * height * width * 3",
        ));
    }
    if !(fps.is_finite() && fps > 0.0) {
        return Err(MmError::invalid_input(
            "glm5_next: video fps must be a positive finite number",
        ));
    }
    let side = |v: usize| {
        u32::try_from(v).map_err(|_| MmError::limit_exceeded("glm5_next: video side exceeds u32"))
    };
    let frames_u32 = u32::try_from(num_frames)
        .map_err(|_| MmError::limit_exceeded("glm5_next: frame count exceeds u32"))?;
    let mut geometry = video.geometry.clone();
    if videos > 1 {
        let share = u32::try_from(videos).unwrap_or(u32::MAX);
        geometry.max_image_tokens = (geometry.max_image_tokens / share).max(1);
    }
    let plan = plan_video_grid(frames_u32, side(w)?, side(h)?, &geometry)?;
    let (th, tw) = (plan.resized_height as usize, plan.resized_width as usize);
    let frames: Vec<std::borrow::Cow<[u8]>> = rgb
        .chunks_exact(h * w * 3)
        .map(|frame| {
            if (th, tw) == (h, w) {
                frame.into()
            } else {
                resize_rgb(frame, h, w, th, tw, Resample::AtenF32Bicubic).into()
            }
        })
        .collect();
    let (ps, m, tps) = (
        geometry.patch_size as usize,
        geometry.merge_size as usize,
        geometry.temporal_patch_size as usize,
    );
    let padded: Vec<&[u8]> = (0..num_frames.div_ceil(tps) * tps)
        .map(|t| &*frames[t.min(num_frames - 1)])
        .collect();
    let grid = [plan.grid_t, plan.grid_h, plan.grid_w];
    let num_patches = grid.iter().map(|&g| g as usize).product::<usize>();
    // `Glm5NextProcessor.replace_video_token`: every second frame's time,
    // whatever the temporal patch size, padded with the last.
    let mut timestamps: Vec<f64> = frame_indices
        .iter()
        .step_by(2)
        .take(plan.grid_t as usize)
        .map(|&index| index as f64 / fps)
        .collect();
    let last = timestamps[timestamps.len() - 1];
    timestamps.resize(plan.grid_t as usize, last);
    Ok(ProcessedItem {
        modality: Modality::Video,
        feature_token_count: plan.grid_t as usize * plan.tokens_per_group as usize,
        feature: Tensor {
            shape: vec![num_patches, 3 * tps * ps * ps],
            data: TensorData::F32(patchify(&padded, (th, tw), (ps, m, tps), lut)),
        },
        aux: vec![(
            "video_grid_thw".into(),
            Tensor {
                shape: vec![3],
                data: TensorData::I64(grid.iter().map(|&g| i64::from(g)).collect()),
            },
        )],
        geometry: Some(Geometry::TimedGrid { grid, timestamps }),
    })
}

/// `<|begin_of_image|>` + `<|image|>` x N + `<|end_of_image|>` +
/// `"{t:.1f} seconds"` per temporal group.
pub(super) fn expansion(
    stage: &VideoStage<'_>,
    grid: &[u32; 3],
    timestamps: &[f64],
    text: &dyn TextEncoder,
) -> Result<Vec<ExpansionPart>> {
    let video = stage.video;
    let m = video.geometry.merge_size as usize;
    let per_group = grid[1] as usize * grid[2] as usize / (m * m);
    let mut parts = Vec::with_capacity(4 * timestamps.len());
    for seconds in timestamps {
        parts.push(ExpansionPart::Literal(vec![video.image_start_token_id]));
        parts.push(ExpansionPart::Feature {
            id: stage.image_token_id,
            n: per_group,
        });
        parts.push(ExpansionPart::Literal(vec![video.image_end_token_id]));
        parts.push(ExpansionPart::Literal(
            text.encode(&format!("{seconds:.1} seconds"))?,
        ));
    }
    Ok(parts)
}
