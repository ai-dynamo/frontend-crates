// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Composite, normalize and patchify code adapted from sgl-project/sglang PR
// #34188 (`rust/sglang-mm/src/kimi_k3/mod.rs` at bd09a3d36c, Apache-2.0), by Kan Wu.

//! Kimi-K3 and K3.1 (MoonViT) images, as the checkpoint's
//! `kimi_k3_vision_processing.py` and `media_utils.py` give them on Pillow
//! 12.3, the path SGLang runs on CPU: PIL decode in the native mode, NaViT
//! sizing, PIL bicubic resize (RGBA premultiplied, as Pillow does), the
//! chessboard composite after the resize, zero pad, numpy's f64-through
//! normalize, and patchify to `(n_patches, 3, ps, ps)`.
//!
//! Every stage after the decode is bit-exact. The decode is bit-exact for
//! JPEG, where both sides run libjpeg-turbo. PNG, BMP, WebP and GIF decode
//! with Rust crates that are second implementations of what Pillow links:
//! the forms known to differ are `Unsupported`, and a form that is not known
//! would decode with other pixels and no error. See README section 7.
//!
//! The expansion prints each image's source size as text
//! (`<|media_begin|>image {w}x{h}<|media_content|>`, the pad run,
//! `<|media_end|>`), so the spec carries the token ids of that text; see
//! [`SizeTokens`].

use std::collections::HashSet;

use crate::image::decode::{DecodeLimits, decode_native};
use crate::image::resize::{self, Filter, Resample};
use crate::processor::{
    DecodedMedia, ExpansionPart, Geometry, MediaMetadata, MmFamilyProcessor, ProcessedItem,
    Segment, Tensor, TensorData, TokenLayout,
};
use crate::{MmError, Result, execution, token_layout};

/// f64 holds every integer below this exactly.
const EXACT_F64: u64 = 1 << 53;

/// The only `transparent_bg_config.pattern` the pipeline mirrors.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BgPattern {
    Chessboard,
}

/// The only `transparent_bg_fill_stage` the pipeline mirrors.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FillStage {
    AfterResize,
}

/// `media_utils.TransparentBgConfig`.
#[derive(Clone, Copy, Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TransparentBgConfig {
    pub pattern: BgPattern,
    pub chessboard_square_size: usize,
    pub chessboard_square_on_top_left: bool,
    pub chessboard_white_value: u8,
    pub chessboard_gray_value: u8,
}

impl TransparentBgConfig {
    fn value_at(&self, y: usize, x: usize) -> u8 {
        let s = self.chessboard_square_size;
        if (y / s + x / s) % 2 == usize::from(self.chessboard_square_on_top_left) {
            self.chessboard_gray_value
        } else {
            self.chessboard_white_value
        }
    }
}

/// Token ids of the text around an image's pad run. The tokenizer splits a
/// decimal number into groups of 3 digits from the left, and every 1 to 3
/// digit string is one token, so the header is a lookup.
///
/// As JSON: `begin` (`<|media_begin|>image `), `by` (`x`), `content`
/// (`<|media_content|>`), `end` (`<|media_end|>`) and `digits`, the ids of
/// `0`..`9`, then `00`..`99`, then `000`..`999`. The consumer builds the table
/// with the model's tokenizer and checks that rule; the crate can only check
/// the table's shape.
#[derive(Clone, Debug, serde::Deserialize)]
pub struct SizeTokens {
    begin: Vec<i32>,
    by: Vec<i32>,
    content: Vec<i32>,
    end: Vec<i32>,
    digits: Vec<i32>,
}

const DIGIT_STRINGS: usize = 10 + 100 + 1000;

impl SizeTokens {
    /// What holds for every table of this shape. A table without zero padding
    /// repeats ids (`05` as `5`).
    fn validate(&self) -> Result<()> {
        let distinct = self.digits.iter().collect::<HashSet<_>>().len();
        if self.digits.len() != DIGIT_STRINGS
            || distinct != DIGIT_STRINGS
            || [&self.begin, &self.by, &self.content, &self.end]
                .iter()
                .any(|ids| ids.is_empty())
        {
            return Err(MmError::invalid_input(
                "kimi_k3 spec: size_tokens needs 1110 distinct digit ids and nonempty marker ids",
            ));
        }
        Ok(())
    }

    fn header(&self, width: u32, height: u32) -> Vec<i32> {
        let mut ids = self.begin.clone();
        for (number, after) in [(width, &self.by), (height, &self.content)] {
            for group in number.to_string().as_bytes().chunks(3) {
                let value = group
                    .iter()
                    .fold(0, |value, digit| value * 10 + usize::from(digit - b'0'));
                ids.push(self.digits[[0, 10, 110][group.len() - 1] + value]);
            }
            ids.extend(after);
        }
        ids
    }
}

/// The `media_proc_cfg` image knobs, plus the pad token id and the size
/// header's token table.
#[derive(Clone, Debug, serde::Deserialize)]
pub struct KimiK3Spec {
    pub image_token_id: i32,
    pub patch_size: u32,
    pub merge_kernel_size: u32,
    pub in_patch_limit: u64,
    pub patch_limit_on_one_side: u32,
    pub image_mean: [f64; 3],
    pub image_std: [f64; 3],
    pub transparent_bg_config: TransparentBgConfig,
    /// Required: the checkpoint composites before the resize when it is absent.
    pub transparent_bg_fill_stage: FillStage,
    /// Must be null: a fixed token count is not mirrored.
    #[serde(default)]
    pub fixed_output_tokens: Option<u64>,
    /// Without it, pixels and [`KimiK3Processor::plan`] work, and the prompt
    /// expansion and its length are `Unsupported`.
    #[serde(default)]
    pub size_tokens: Option<SizeTokens>,
}

impl KimiK3Spec {
    /// Refuse what the pipeline cannot honor exactly.
    pub(crate) fn validate(&self) -> Result<()> {
        if self.fixed_output_tokens.is_some() {
            return Err(MmError::unsupported(
                "kimi_k3 spec: fixed_output_tokens is not mirrored",
            ));
        }
        let max_side = u64::from(self.patch_limit_on_one_side) * u64::from(self.patch_size);
        if self.patch_size == 0
            || self.merge_kernel_size == 0
            || !(1..=EXACT_F64).contains(&self.in_patch_limit)
            || !(1..=EXACT_F64).contains(&max_side)
        {
            return Err(MmError::invalid_input(
                "kimi_k3 spec: sizes and limits must be positive and below 2^53",
            ));
        }
        if self.image_std.iter().any(|s| !s.is_finite() || *s == 0.0)
            || self.image_mean.iter().any(|m| !m.is_finite())
        {
            return Err(MmError::invalid_input(
                "kimi_k3 spec: image_mean must be finite and image_std finite and nonzero",
            ));
        }
        if self.transparent_bg_config.chessboard_square_size == 0 {
            return Err(MmError::invalid_input(
                "kimi_k3 spec: chessboard square size must be positive",
            ));
        }
        self.size_tokens
            .as_ref()
            .map_or(Ok(()), SizeTokens::validate)
    }
}

/// `navit_resize_image`'s result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NavitPlan {
    pub new_width: u64,
    pub new_height: u64,
    pub pad_width: u64,
    pub pad_height: u64,
    /// The feature tokens: the length of the pad run.
    pub num_tokens: usize,
}

pub struct KimiK3Processor {
    spec: KimiK3Spec,
    lut: [[f32; 256]; 3],
}

/// `media_utils.normalize`: numpy runs each in-place f32 op against the f64
/// mean and inverse std in f64, then casts back.
fn normalize_lut(mean: f64, std: f64) -> [f32; 256] {
    let std_inv = 1.0 / std;
    core::array::from_fn(|v| {
        let x = (v as f64 / 255.0) as f32;
        let x = (f64::from(x) - mean) as f32;
        (f64::from(x) * std_inv) as f32
    })
}

/// `fill_transparent_bg_with` on HWC RGBA: an f32 blend truncated to u8.
fn composite(rgba: &[u8], h: usize, w: usize, bg: TransparentBgConfig) -> Vec<u8> {
    let mut out = vec![0u8; h * w * 3];
    execution::for_chunks_mut(&mut out, w * 3, |y, row| {
        let src = &rgba[y * w * 4..(y + 1) * w * 4];
        for (x, (o, p)) in row.chunks_exact_mut(3).zip(src.chunks_exact(4)).enumerate() {
            let alpha = f32::from(p[3]) / 255.0;
            let back = f32::from(bg.value_at(y, x));
            for (o, &c) in o.iter_mut().zip(&p[..3]) {
                *o = (alpha * f32::from(c) + (1.0 - alpha) * back) as u8;
            }
        }
    });
    out
}

fn side(value: usize) -> Result<u32> {
    u32::try_from(value).map_err(|_| MmError::limit_exceeded("kimi_k3: image side exceeds u32"))
}

impl KimiK3Processor {
    pub fn new(spec: KimiK3Spec) -> Result<Self> {
        spec.validate()?;
        let lut = core::array::from_fn(|c| normalize_lut(spec.image_mean[c], spec.image_std[c]));
        Ok(Self { spec, lut })
    }

    /// `navit_resize_image` from the source size alone.
    pub fn plan(&self, width: u32, height: u32) -> Result<NavitPlan> {
        if width == 0 || height == 0 {
            return Err(MmError::invalid_input(
                "kimi_k3: image dimensions must be positive",
            ));
        }
        let s = &self.spec;
        let (w, h, ps) = (u64::from(width), u64::from(height), u64::from(s.patch_size));
        let max_side = u64::from(s.patch_limit_on_one_side) * ps;
        // Python divides by the exact int product; an f64 product rounds first.
        let cells = (w / ps).max(1) * (h / ps).max(1);
        if cells > EXACT_F64 {
            return Err(MmError::limit_exceeded(
                "kimi_k3: image is too large for exact resize planning",
            ));
        }
        let scale = 1f64
            .min((s.in_patch_limit as f64 / cells as f64).sqrt())
            .min(max_side as f64 / w as f64)
            .min(max_side as f64 / h as f64);
        let fit = |v: u64| ((v as f64 * scale) as u64).clamp(1, max_side);
        let (new_width, new_height) = (fit(w), fit(h));
        let factor = u64::from(s.merge_kernel_size) * ps;
        let pad = |v: u64| (factor - v % factor) % factor;
        let (pad_width, pad_height) = (pad(new_width), pad(new_height));
        let rows = (new_height + pad_height) / factor;
        let cols = (new_width + pad_width) / factor;
        let limit = u64::from(s.patch_limit_on_one_side);
        let merge = u64::from(s.merge_kernel_size);
        if rows * merge > limit || cols * merge > limit {
            return Err(MmError::invalid_input(
                "kimi_k3: token grid exceeds patch_limit_on_one_side",
            ));
        }
        Ok(NavitPlan {
            new_width,
            new_height,
            pad_width,
            pad_height,
            num_tokens: usize::try_from(rows * cols)
                .map_err(|_| MmError::limit_exceeded("kimi_k3: token count exceeds usize"))?,
        })
    }

    /// An image's feature tokens, source width and height.
    fn image(&self, media: &MediaMetadata) -> Result<(usize, u32, u32)> {
        let MediaMetadata::Image { width, height } = *media else {
            return Err(MmError::unsupported("kimi_k3 supports only images"));
        };
        Ok((self.plan(width, height)?.num_tokens, width, height))
    }

    fn size_tokens(&self) -> Result<&SizeTokens> {
        self.spec.size_tokens.as_ref().ok_or_else(|| {
            MmError::unsupported(
                "kimi_k3: the prompt expansion prints the image size and needs size_tokens",
            )
        })
    }

    /// Zero-pad to the grid, normalize, and emit patches in row-major order.
    fn patchify(&self, rgb: &[u8], nh: usize, nw: usize, gh: usize, gw: usize) -> Result<Vec<f32>> {
        let ps = self.spec.patch_size as usize;
        let len = [gw, 3, ps, ps]
            .iter()
            .try_fold(gh, |len, &n| len.checked_mul(n))
            .ok_or_else(|| MmError::limit_exceeded("kimi_k3: feature size overflows usize"))?;
        let mut out = Vec::new();
        out.try_reserve_exact(len)
            .map_err(|_| MmError::limit_exceeded("kimi_k3: feature allocation failed"))?;
        out.resize(len, 0f32);
        let elems = 3 * ps * ps;
        execution::for_chunks_mut(&mut out, gw * elems, |i, row| {
            for (j, patch) in row.chunks_exact_mut(elems).enumerate() {
                for (c, plane) in patch.chunks_exact_mut(ps * ps).enumerate() {
                    for py in 0..ps {
                        let y = i * ps + py;
                        for px in 0..ps {
                            let x = j * ps + px;
                            let v = if y < nh && x < nw {
                                rgb[(y * nw + x) * 3 + c]
                            } else {
                                0
                            };
                            plane[py * ps + px] = self.lut[c][usize::from(v)];
                        }
                    }
                }
            }
        });
        Ok(out)
    }

    /// `items` hold each image's feature tokens, source width and height.
    fn expand(&self, input_ids: &[i32], items: &[(usize, u32, u32)]) -> Result<TokenLayout> {
        let tokens = self.size_tokens()?;
        let counts = items.iter().map(|item| item.0).collect::<Vec<_>>();
        let mut layout =
            token_layout::layout_by_placeholder(input_ids, self.spec.image_token_id, &counts)?;
        for segment in &mut layout.segments {
            if let Segment::Media {
                item, expansion, ..
            } = segment
            {
                let (_, width, height) = items[*item];
                expansion.insert(0, ExpansionPart::Literal(tokens.header(width, height)));
                expansion.push(ExpansionPart::Literal(tokens.end.clone()));
            }
        }
        Ok(layout)
    }
}

impl MmFamilyProcessor for KimiK3Processor {
    fn decode_image(&self, data: &[u8], limits: &DecodeLimits) -> Result<DecodedMedia> {
        let image = decode_native(data, limits)?;
        let (height, width) = (image.height, image.width);
        if !image.exact_resize {
            let plan = self.plan(side(width)?, side(height)?)?;
            if (plan.new_width, plan.new_height) != (width as u64, height as u64) {
                return Err(MmError::unsupported(
                    "kimi_k3: a palette, 1-bit, L with tRNS or CMYK source that needs a resize \
                     is not mirrored",
                ));
            }
        }
        Ok(if image.alpha {
            DecodedMedia::ImageRgba {
                rgba: image.pixels,
                height,
                width,
            }
        } else {
            DecodedMedia::Image {
                rgb: image.pixels,
                height,
                width,
            }
        })
    }

    /// The whole expansion: the size header, the pad run and the end token.
    fn num_media_tokens(&self, media: &MediaMetadata) -> Result<usize> {
        let (pads, width, height) = self.image(media)?;
        let tokens = self.size_tokens()?;
        (tokens.header(width, height).len() + tokens.end.len())
            .checked_add(pads)
            .ok_or_else(|| MmError::limit_exceeded("kimi_k3: prompt span exceeds usize"))
    }

    /// The pad run, which needs no `size_tokens`.
    fn num_feature_tokens(&self, media: &MediaMetadata) -> Result<usize> {
        Ok(self.image(media)?.0)
    }

    fn layout_from_metadata(
        &self,
        input_ids: &[i32],
        media: &[MediaMetadata],
    ) -> Result<TokenLayout> {
        let items = media
            .iter()
            .map(|item| self.image(item))
            .collect::<Result<Vec<_>>>()?;
        self.expand(input_ids, &items)
    }

    fn process_item(&self, media: &DecodedMedia) -> Result<ProcessedItem> {
        let (pixels, channels, h, w) = match media {
            DecodedMedia::Image { rgb, height, width } => (rgb, 3, *height, *width),
            DecodedMedia::ImageRgba {
                rgba,
                height,
                width,
            } => (rgba, 4, *height, *width),
        };
        if h.checked_mul(w).and_then(|n| n.checked_mul(channels)) != Some(pixels.len()) {
            return Err(MmError::invalid_input(
                "kimi_k3: pixel length does not match height * width * channels",
            ));
        }
        let (width, height) = (side(w)?, side(h)?);
        let plan = self.plan(width, height)?;
        let (nh, nw) = (plan.new_height as usize, plan.new_width as usize);
        let rgb = if channels == 4 {
            let resized = resize::resize_rgba(pixels, h, w, nh, nw);
            composite(&resized, nh, nw, self.spec.transparent_bg_config)
        } else {
            resize::resize_rgb(pixels, h, w, nh, nw, Resample::Pil(Filter::Bicubic))
        };
        let ps = self.spec.patch_size as usize;
        let gh = (nh + plan.pad_height as usize) / ps;
        let gw = (nw + plan.pad_width as usize) / ps;
        let grid = [1, gh as u32, gw as u32];
        Ok(ProcessedItem {
            modality: media.modality(),
            feature_token_count: plan.num_tokens,
            feature: Tensor {
                shape: vec![gh * gw, 3, ps, ps],
                data: TensorData::F32(self.patchify(&rgb, nh, nw, gh, gw)?),
            },
            aux: vec![(
                "image_grid_thw".to_string(),
                Tensor {
                    shape: vec![3],
                    data: TensorData::I64(grid.iter().map(|&g| i64::from(g)).collect()),
                },
            )],
            geometry: Some(Geometry::SourceSize { width, height }),
        })
    }

    fn layout(&self, input_ids: &[i32], items: &[ProcessedItem]) -> Result<TokenLayout> {
        let items = items
            .iter()
            .map(|item| match item.geometry {
                Some(Geometry::SourceSize { width, height }) => {
                    Ok((item.feature_token_count, width, height))
                }
                _ => Err(MmError::invalid_input(
                    "kimi_k3: item is missing its source size",
                )),
            })
            .collect::<Result<Vec<_>>>()?;
        self.expand(input_ids, &items)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(patch_limit_on_one_side: u32) -> KimiK3Spec {
        KimiK3Spec {
            image_token_id: 163605,
            patch_size: 14,
            merge_kernel_size: 2,
            in_patch_limit: 65536,
            patch_limit_on_one_side,
            image_mean: [0.5; 3],
            image_std: [0.5; 3],
            transparent_bg_config: TransparentBgConfig {
                pattern: BgPattern::Chessboard,
                chessboard_square_size: 8,
                chessboard_square_on_top_left: true,
                chessboard_white_value: 255,
                chessboard_gray_value: 180,
            },
            transparent_bg_fill_stage: FillStage::AfterResize,
            fixed_output_tokens: None,
            size_tokens: None,
        }
    }

    /// `media_utils.navit_resize_image` of the checkpoint, as
    /// `(new_width, new_height, pad_width, pad_height, num_tokens)`.
    #[test]
    fn plan_matches_navit_resize_image() {
        for (side, width, height, expected) in [
            (512, 1, 1, (1, 1, 27, 27, 1)),
            (512, 3, 5, (3, 5, 25, 23, 1)),
            (512, 300, 200, (300, 200, 8, 24, 88)),
            (512, 1023, 767, (1023, 767, 13, 17, 1036)),
            (512, 3584, 3584, (3584, 3584, 0, 0, 16384)),
            // One patch column over the budget stays: floor(3585 / 14) = 256.
            (512, 3585, 3584, (3585, 3584, 27, 0, 16512)),
            (512, 7168, 7168, (3584, 3584, 0, 0, 16384)),
            (512, 65535, 65535, (3584, 3584, 0, 0, 16384)),
            (512, 9000, 4000, (5386, 2393, 18, 15, 16598)),
            // The side cap, and `int()` truncation to 62 and 47.
            (512, 7300, 64, (7168, 62, 0, 22, 768)),
            (512, 60, 9000, (47, 7168, 9, 0, 512)),
            (512, 100000, 7, (7167, 1, 1, 27, 256)),
            (8, 300, 24, (112, 8, 0, 20, 4)),
            (16, 2, 1000, (1, 224, 27, 0, 8)),
        ] {
            let plan = KimiK3Processor::new(spec(side))
                .unwrap()
                .plan(width, height)
                .unwrap();
            assert_eq!(
                (
                    plan.new_width,
                    plan.new_height,
                    plan.pad_width,
                    plan.pad_height,
                    plan.num_tokens
                ),
                expected,
                "{width}x{height}, side limit {side}"
            );
        }
    }

    /// The checkpoint asserts here: 98 px pads to 4 tokens of 2 patches, over 7.
    #[test]
    fn plan_refuses_what_the_checkpoint_asserts_against_or_cannot_be_exact() {
        let odd = KimiK3Processor::new(spec(7)).unwrap();
        assert!(matches!(
            odd.plan(300, 24),
            Err(MmError::InvalidInput { .. })
        ));
        let shipped = KimiK3Processor::new(spec(512)).unwrap();
        assert!(matches!(
            shipped.plan(0, 1),
            Err(MmError::InvalidInput { .. })
        ));
        assert!(matches!(
            shipped.plan(u32::MAX, u32::MAX),
            Err(MmError::LimitExceeded { .. })
        ));
    }

    /// 300x200 expands to 7 header ids, 88 pads and the end token.
    #[test]
    fn num_media_tokens_is_the_prompt_span() {
        let image = MediaMetadata::Image {
            width: 300,
            height: 200,
        };
        let mut with_table = spec(512);
        with_table.size_tokens = Some(size_tokens());
        let family = KimiK3Processor::new(with_table).unwrap();
        assert_eq!(family.plan(300, 200).unwrap().num_tokens, 88);
        assert_eq!(family.num_media_tokens(&image).unwrap(), 96);
        assert_eq!(family.num_feature_tokens(&image).unwrap(), 88);
        for media in [
            MediaMetadata::Video {
                width: 28,
                height: 28,
                num_frames: 4,
            },
            MediaMetadata::Audio {
                num_samples: 16000,
                sample_rate: 16000,
            },
        ] {
            assert!(matches!(
                family.num_media_tokens(&media),
                Err(MmError::Unsupported { .. })
            ));
            assert!(matches!(
                family.num_feature_tokens(&media),
                Err(MmError::Unsupported { .. })
            ));
        }
    }

    /// Bit patterns of `media_utils.normalize` on u8 0, 1, 127, 128, 254, 255.
    #[test]
    fn normalize_goes_through_f64_as_numpy_does() {
        let bits = |mean, std| {
            let lut = normalize_lut(mean, std);
            [0, 1, 127, 128, 254, 255].map(|v: usize| lut[v].to_bits())
        };
        assert_eq!(
            bits(0.5, 0.5),
            [
                3212836864, 3212705278, 3145760896, 998277376, 1065221630, 1065353216
            ]
        );
        assert_eq!(
            bits(0.48145466, 0.26862954),
            [
                3219482844, 3219360384, 1031594142, 1033655846, 1073034981, 1073157442
            ]
        );
    }

    /// `fill_transparent_bg_with` on a 16x2 image of (200, 13, 255) over the
    /// shipped chessboard, one alpha per pixel.
    #[test]
    fn composite_matches_fill_transparent_bg_with() {
        let alpha = [
            0, 1, 2, 64, 127, 128, 129, 254, 255, 3, 77, 100, 150, 200, 250, 253,
        ];
        let rgba: Vec<u8> = (alpha.iter().chain(alpha.iter().rev()))
            .flat_map(|&a| [200, 13, 255, a])
            .collect();
        let expected = [
            255, 255, 255, 254, 254, 255, 254, 253, 255, 241, 194, 255, 227, 134, 255, 227, 133,
            255, 227, 132, 255, 200, 13, 255, 200, 13, 255, 180, 178, 180, 186, 129, 202, 187, 114,
            209, 191, 81, 224, 195, 49, 238, 199, 16, 253, 199, 14, 254, 200, 14, 255, 201, 17,
            255, 211, 65, 255, 222, 112, 255, 233, 160, 255, 238, 181, 254, 254, 252, 255, 200, 13,
            255, 199, 13, 254, 190, 95, 217, 190, 96, 217, 189, 96, 217, 185, 138, 198, 180, 178,
            180, 180, 179, 180, 180, 180, 180,
        ];
        assert_eq!(
            composite(&rgba, 2, 16, spec(512).transparent_bg_config),
            expected
        );
    }

    /// Markers, and digit string `k` of `0`..`9`, `00`..`99`, `000`..`999` as
    /// id `100 + k`.
    fn size_tokens() -> SizeTokens {
        SizeTokens {
            begin: vec![0, 3, 4],
            by: vec![5],
            content: vec![1],
            end: vec![2],
            digits: (100..1210).collect(),
        }
    }

    #[test]
    fn size_header_is_a_lookup_and_a_table_of_another_shape_is_refused() {
        let tokens = size_tokens();
        // "100", "05", "x", "7": groups of 3 from the left keep leading zeros.
        assert_eq!(
            tokens.header(10005, 7),
            [0, 3, 4, 100 + 110 + 100, 100 + 10 + 5, 5, 100 + 7, 1]
        );
        assert_eq!(tokens.header(u32::MAX, 1).len(), 3 + 4 + 1 + 1 + 1);

        // A table of another shape: short, built without zero padding (`05`
        // gets the id of `5`), or with an empty marker.
        let mut unpadded = tokens.digits.clone();
        unpadded[10 + 5] = unpadded[5];
        for broken in [
            SizeTokens {
                digits: tokens.digits[..1109].to_vec(),
                ..tokens.clone()
            },
            SizeTokens {
                digits: unpadded,
                ..tokens.clone()
            },
            SizeTokens {
                end: Vec::new(),
                ..tokens.clone()
            },
        ] {
            let mut spec = spec(512);
            spec.size_tokens = Some(broken);
            assert!(matches!(
                KimiK3Processor::new(spec),
                Err(MmError::InvalidInput { .. })
            ));
        }
    }

    #[test]
    fn invalid_specs_are_refused() {
        let invalid: [fn(&mut KimiK3Spec); 9] = [
            |spec| spec.patch_size = 0,
            |spec| spec.merge_kernel_size = 0,
            |spec| spec.patch_limit_on_one_side = 0,
            |spec| spec.in_patch_limit = 0,
            |spec| spec.in_patch_limit = (1 << 53) + 1,
            |spec| spec.image_std[1] = 0.0,
            |spec| spec.image_std[2] = f64::NAN,
            |spec| spec.image_mean[0] = f64::INFINITY,
            |spec| spec.transparent_bg_config.chessboard_square_size = 0,
        ];
        for change in invalid {
            let mut spec = spec(512);
            change(&mut spec);
            assert!(matches!(
                KimiK3Processor::new(spec),
                Err(MmError::InvalidInput { .. })
            ));
        }
        let mut fixed = spec(512);
        fixed.fixed_output_tokens = Some(64);
        assert!(matches!(
            KimiK3Processor::new(fixed),
            Err(MmError::Unsupported { .. })
        ));
    }

    /// A spec JSON carries the checkpoint's knobs; the ones the pipeline does
    /// not mirror are an error, not ignored.
    #[test]
    fn spec_json_refuses_the_knobs_the_pipeline_does_not_mirror() {
        let shipped = serde_json::json!({
            "image_token_id": 163605, "patch_size": 14, "merge_kernel_size": 2,
            "in_patch_limit": 65536, "patch_limit_on_one_side": 512,
            "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5],
            "transparent_bg_config": {
                "pattern": "chessboard", "chessboard_square_size": 8,
                "chessboard_square_on_top_left": true,
                "chessboard_white_value": 255, "chessboard_gray_value": 180
            },
            "transparent_bg_fill_stage": "after_resize",
            "fixed_output_tokens": null
        });
        let build = |spec: serde_json::Value| {
            serde_json::from_value::<KimiK3Spec>(spec)
                .map_err(|error| MmError::invalid_input_with_source("spec", error))
                .and_then(KimiK3Processor::new)
                .map(|_| ())
        };
        assert!(build(shipped.clone()).is_ok());

        let mut before_resize = shipped.clone();
        before_resize["transparent_bg_fill_stage"] = "before_resize".into();
        let mut no_stage = shipped.clone();
        no_stage
            .as_object_mut()
            .unwrap()
            .remove("transparent_bg_fill_stage");
        let mut white = shipped.clone();
        white["transparent_bg_config"]["pattern"] = "white".into();
        for spec in [before_resize, no_stage, white] {
            assert!(matches!(build(spec), Err(MmError::InvalidInput { .. })));
        }
        let mut fixed = shipped;
        fixed["fixed_output_tokens"] = 64.into();
        assert!(matches!(build(fixed), Err(MmError::Unsupported { .. })));
    }

    /// `patch_size` 2^31 passes the spec limits; its feature size does not fit.
    #[test]
    fn process_item_refuses_a_feature_size_that_overflows() {
        let mut huge = spec(512);
        huge.patch_size = 1 << 31;
        let family = KimiK3Processor::new(huge).unwrap();
        let pixel = DecodedMedia::Image {
            rgb: vec![0; 3],
            height: 1,
            width: 1,
        };
        assert!(matches!(
            family.process_item(&pixel),
            Err(MmError::LimitExceeded { .. })
        ));
    }

    #[test]
    fn process_item_refuses_a_wrong_buffer_length() {
        let family = KimiK3Processor::new(spec(512)).unwrap();
        for media in [
            DecodedMedia::Image {
                rgb: vec![0; 11],
                height: 2,
                width: 2,
            },
            DecodedMedia::ImageRgba {
                rgba: vec![0; 12],
                height: 2,
                width: 2,
            },
        ] {
            assert!(matches!(
                family.process_item(&media),
                Err(MmError::InvalidInput { .. })
            ));
        }
    }
}
