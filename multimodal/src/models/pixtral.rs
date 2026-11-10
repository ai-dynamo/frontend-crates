// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Sizing and expansion adapted from transformers' image_processing_pixtral.py and
// processing_pixtral.py (Apache-2.0), v5.19.0.

//! Pixtral images (Mistral Large 4) as transformers' `PixtralProcessor` runs
//! them on the CPU; SGLang's default tokenizer mode calls that processor. Fit
//! the long side into `longest_edge` with floor, stretch both sides up to a
//! multiple of `patch_size * spatial_merge_size`, normalize to planar
//! `[1, 3, H, W]`, and expand each `[IMG]` into rows of `[IMG]` closed by
//! `[IMG_BREAK]` / `[IMG_END]`.
//!
//! The pixel values are those of the CPU kernels: a GPU resize gives the same
//! sizes and ids but other values. mistral-common, which vLLM runs for this
//! model, sizes with round instead of floor and has its own pixel pipeline;
//! neither is implemented here.

use crate::image::resize::resize_rgb;
use crate::models::qwen_vl::{Resampler, normalize_lut};
use crate::processor::{
    DecodedMedia, ExpansionPart, Geometry, MediaMetadata, MmFamilyProcessor, ProcessedItem,
    Segment, Tensor, TensorData, TokenLayout,
};
use crate::token_layout::layout_by_placeholder;
use crate::{MmError, Result, execution};

/// Resolved processor params: ids from `params.json` `vision_encoder`, knobs
/// from the HF `PixtralProcessor` config.
#[derive(Clone, Debug, PartialEq, serde::Deserialize)]
pub struct PixtralSpec {
    /// `[IMG]`: the chat-template placeholder and every feature position.
    pub image_token_id: i32,
    /// `[IMG_BREAK]`: closes each grid row but the last.
    pub image_break_token_id: i32,
    /// `[IMG_END]`: closes the last grid row.
    pub image_end_token_id: i32,
    pub patch_size: u32,
    pub spatial_merge_size: u32,
    /// HF `size.longest_edge`.
    pub longest_edge: u32,
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
    #[serde(default)]
    pub resample: Resampler,
}

/// Mistral Large 4 image processor: HF `PixtralProcessor` pixels and prompt layout.
///
/// ```
/// use dynamo_multimodal::{Result, registry, token_layout};
/// use dynamo_multimodal::processor::{ExpansionPart, MediaMetadata, Segment};
/// # fn main() -> Result<()> {
/// // mistralai/Mistral-Large-4-1T-A52B: params.json ids, processor_config.json knobs.
/// let family = registry::processor_from_spec(r#"{
///     "family": "pixtral", "image_token_id": 10, "image_break_token_id": 12,
///     "image_end_token_id": 13, "patch_size": 14, "spatial_merge_size": 2,
///     "longest_edge": 1540, "image_mean": [0.48145466, 0.4578275, 0.40821073],
///     "image_std": [0.26862954, 0.26130258, 0.27577711]
/// }"#)?;
/// let media = [MediaMetadata::Image { width: 100, height: 60 }];
/// // <s> [INST] [IMG] text [/INST]: the chat template writes one [IMG] per image.
/// let ids = [1, 3, 10, 4420, 4];
/// // 84x112 pixels, 3 rows of 4 [IMG]; each row adds one closing id.
/// assert_eq!(family.num_media_tokens(&media[0])?, 15);
/// let layout = family.layout_from_metadata(&ids, &media)?;
/// // `apply_layout` takes the positions that get embeddings, not that span:
/// // without pixels, the sum of each item's `Feature` parts.
/// let counts: Vec<usize> = (layout.segments.iter())
///     .filter_map(|segment| match segment {
///         Segment::Media { expansion, .. } => Some(expansion.iter().map(|part| match part {
///             ExpansionPart::Feature { n, .. } => *n,
///             _ => 0,
///         }).sum()),
///         Segment::Text(_) => None,
///     })
///     .collect();
/// assert_eq!(counts, [12]);
/// let expanded = token_layout::apply_layout(&ids, &layout, &counts, 64)?;
/// assert_eq!(
///     expanded.input_ids,
///     [1, 3, 10, 10, 10, 10, 12, 10, 10, 10, 10, 12, 10, 10, 10, 10, 13, 4420, 4]
/// );
/// assert_eq!(expanded.offsets, [(2, 16)]);
/// assert_eq!(expanded.feature_ranges[0], [2..6, 7..11, 12..16]); // one run per row
/// # Ok(())
/// # }
/// ```
pub struct PixtralProcessor {
    spec: PixtralSpec,
    /// `patch_size * spatial_merge_size`: pixels per prompt token side.
    patch: u32,
    lut: [[f32; 256]; 3],
}

impl PixtralProcessor {
    pub fn new(spec: PixtralSpec) -> Result<Self> {
        let patch = spec
            .patch_size
            .checked_mul(spec.spatial_merge_size)
            .filter(|patch| *patch > 0 && spec.longest_edge > 0)
            .ok_or_else(|| {
                MmError::invalid_input(
                    "pixtral spec: patch_size, spatial_merge_size and longest_edge must be \
                     positive, and their product must fit u32",
                )
            })?;
        let ids = [
            spec.image_token_id,
            spec.image_break_token_id,
            spec.image_end_token_id,
        ];
        if ids.iter().any(|id| *id < 0) || ids[0] == ids[1] || ids[0] == ids[2] || ids[1] == ids[2]
        {
            return Err(MmError::invalid_input(
                "pixtral spec: the three image token ids must be nonnegative and distinct",
            ));
        }
        if spec
            .image_std
            .iter()
            .any(|std| !std.is_finite() || *std <= 0.0)
        {
            return Err(MmError::invalid_input(
                "pixtral spec: image_std values must be finite and positive",
            ));
        }
        if spec.image_mean.iter().any(|mean| !mean.is_finite()) {
            return Err(MmError::invalid_input(
                "pixtral spec: image_mean values must be finite",
            ));
        }
        let lut = core::array::from_fn(|c| {
            normalize_lut(spec.resample, spec.image_mean[c], spec.image_std[c])
        });
        Ok(Self { spec, patch, lut })
    }

    /// `(rows, cols)` of prompt tokens: HF's `get_resize_output_image_size`
    /// in units of `patch_size * spatial_merge_size` pixels.
    pub fn grid(&self, width: u32, height: u32) -> Result<(u32, u32)> {
        let longest = f64::from(self.spec.longest_edge);
        let (mut h, mut w) = (f64::from(height), f64::from(width));
        let ratio = (h / longest).max(w / longest);
        if ratio > 1.0 {
            h = (h / ratio).floor();
            w = (w / ratio).floor();
        }
        // HF plans a zero-sized resize here and its resize raises.
        if h < 1.0 || w < 1.0 {
            return Err(MmError::invalid_input(format!(
                "pixtral: a {width}x{height} image has a side of 0 pixels after fitting {} pixels",
                self.spec.longest_edge
            )));
        }
        Ok((
            (h as u32).div_ceil(self.patch),
            (w as u32).div_ceil(self.patch),
        ))
    }

    /// Each `[IMG]` becomes its item's grid: `cols` feature positions per row,
    /// then `[IMG_BREAK]`, with `[IMG_END]` closing the last row.
    fn expand(&self, input_ids: &[i32], grids: &[(u32, u32)]) -> Result<TokenLayout> {
        let image = self.spec.image_token_id;
        let mut layout = layout_by_placeholder(input_ids, image, &vec![0; grids.len()])?;
        for segment in &mut layout.segments {
            let Segment::Media {
                item, expansion, ..
            } = segment
            else {
                continue;
            };
            let (rows, cols) = grids[*item];
            *expansion = (1..=rows)
                .flat_map(|row| {
                    let close = if row == rows {
                        self.spec.image_end_token_id
                    } else {
                        self.spec.image_break_token_id
                    };
                    [
                        ExpansionPart::Feature {
                            id: image,
                            n: cols as usize,
                        },
                        ExpansionPart::Literal(vec![close]),
                    ]
                })
                .collect();
        }
        Ok(layout)
    }
}

impl MmFamilyProcessor for PixtralProcessor {
    /// The prompt span `rows * (cols + 1)` that the HF processor call writes;
    /// `rows * cols` of them take embeddings.
    fn num_media_tokens(&self, media: &MediaMetadata) -> Result<usize> {
        match media {
            MediaMetadata::Image { width, height } => {
                let (rows, cols) = self.grid(*width, *height)?;
                usize::try_from(u64::from(rows) * (u64::from(cols) + 1))
                    .map_err(|_| MmError::limit_exceeded("pixtral: token count exceeds usize"))
            }
            _ => Err(MmError::unsupported("pixtral supports only images")),
        }
    }

    fn layout_from_metadata(
        &self,
        input_ids: &[i32],
        media: &[MediaMetadata],
    ) -> Result<TokenLayout> {
        let grids = media
            .iter()
            .map(|item| match item {
                MediaMetadata::Image { width, height } => self.grid(*width, *height),
                _ => Err(MmError::unsupported("pixtral supports only images")),
            })
            .collect::<Result<Vec<_>>>()?;
        self.expand(input_ids, &grids)
    }

    fn process_item(&self, media: &DecodedMedia) -> Result<ProcessedItem> {
        let DecodedMedia::Image { rgb, height, width } = media else {
            return Err(MmError::unsupported("pixtral: expects an RGB image"));
        };
        let (h, w) = (*height, *width);
        if h.checked_mul(w).and_then(|n| n.checked_mul(3)) != Some(rgb.len()) {
            return Err(MmError::invalid_input(
                "pixtral: rgb length does not match height * width * 3",
            ));
        }
        let side = |n: usize| {
            u32::try_from(n).map_err(|_| MmError::limit_exceeded("pixtral: side exceeds u32"))
        };
        let (rows, cols) = self.grid(side(w)?, side(h)?)?;
        let patch = self.patch as usize;
        let (th, tw) = (rows as usize * patch, cols as usize * patch);
        let plane = th
            .checked_mul(tw)
            .filter(|plane| plane.checked_mul(3).is_some())
            .ok_or_else(|| MmError::limit_exceeded("pixtral: feature size overflows usize"))?;
        let mut pixel_values = Vec::new();
        pixel_values
            .try_reserve_exact(3 * plane)
            .map_err(|_| MmError::limit_exceeded("pixtral: feature allocation failed"))?;
        pixel_values.resize(3 * plane, 0.0f32);
        let resized;
        let data = if (th, tw) != (h, w) {
            resized = resize_rgb(rgb, h, w, th, tw, self.spec.resample.into());
            &resized
        } else {
            rgb.as_slice()
        };
        execution::for_chunks_mut(&mut pixel_values, plane, |c, channel| {
            for (value, pixel) in channel.iter_mut().zip(data.chunks_exact(3)) {
                *value = self.lut[c][pixel[c] as usize];
            }
        });
        Ok(ProcessedItem {
            modality: media.modality(),
            feature_token_count: rows as usize * cols as usize,
            feature: Tensor {
                shape: vec![1, 3, th, tw],
                data: TensorData::F32(pixel_values),
            },
            aux: vec![(
                "image_sizes".to_string(),
                Tensor {
                    shape: vec![1, 2],
                    data: TensorData::I64(vec![th as i64, tw as i64]),
                },
            )],
            geometry: Some(Geometry::Grid([1, rows, cols])),
        })
    }

    fn layout(&self, input_ids: &[i32], items: &[ProcessedItem]) -> Result<TokenLayout> {
        let grids = items
            .iter()
            .map(|item| match item.geometry {
                Some(Geometry::Grid([_, rows, cols])) => Ok((rows, cols)),
                _ => Err(MmError::invalid_input("pixtral: item is missing its grid")),
            })
            .collect::<Result<Vec<_>>>()?;
        self.expand(input_ids, &grids)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// mistralai/Mistral-Large-4-1T-A52B.
    fn spec() -> PixtralSpec {
        PixtralSpec {
            image_token_id: 10,
            image_break_token_id: 12,
            image_end_token_id: 13,
            patch_size: 14,
            spatial_merge_size: 2,
            longest_edge: 1540,
            image_mean: [0.481_454_66, 0.457_827_5, 0.408_210_73],
            image_std: [0.268_629_54, 0.261_302_6, 0.275_777_1],
            resample: Resampler::AtenU8,
        }
    }

    #[test]
    fn rejects_invalid_specs() {
        let invalid = |edit: fn(&mut PixtralSpec)| {
            let mut spec = spec();
            edit(&mut spec);
            assert!(matches!(
                PixtralProcessor::new(spec),
                Err(MmError::InvalidInput { .. })
            ));
        };
        invalid(|spec| spec.patch_size = 0);
        invalid(|spec| spec.spatial_merge_size = 0);
        invalid(|spec| spec.longest_edge = 0);
        invalid(|spec| spec.patch_size = u32::MAX);
        invalid(|spec| spec.image_token_id = -1);
        invalid(|spec| spec.image_break_token_id = 10);
        invalid(|spec| spec.image_end_token_id = 12);
        invalid(|spec| spec.image_std[1] = 0.0);
        invalid(|spec| spec.image_std[2] = f32::NAN);
        invalid(|spec| spec.image_mean[0] = f32::INFINITY);
    }
}
