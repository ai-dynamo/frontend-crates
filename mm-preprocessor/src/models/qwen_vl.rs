// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Qwen VL family (Qwen2-VL / 2.5-VL / 3-VL / 3.5) image processor.
//!
//! Pure-Rust equivalent of the HF `Qwen2VLImageProcessor` pipeline:
//! `smart_resize` → bicubic resize → rescale + normalize → patchify into
//! `[grid_h*grid_w, C*tps*ps*ps]` in HF flatten order (patches ordered
//! `(gh/m, gw/m, m, m)`, features `(C, tps, ps, ps)`, temporal copies
//! duplicated for stills) — plus the image-only M-RoPE fast path. All
//! parameters come from the runtime spec.

use crate::image::resize;
use crate::processor::{
    DecodedMedia, Geometry, MediaMetadata, MmFamilyProcessor, PositionOutput, ProcessedItem,
    Tensor, TensorData, TokenLayout,
};
use crate::{MmError, Result, execution, token_layout};

const MAX_RATIO: f64 = 200.0;

/// One media item's placement for M-RoPE: inclusive token range + patch grid.
pub struct MropeItem {
    pub start: u32,
    pub end: u32,
    pub grid: [u32; 3],
}

/// Resolved processor params, deserialized from the consumer-side spec JSON
/// (unknown fields like `family` are ignored here).
#[derive(Clone, Debug, serde::Deserialize)]
pub struct QwenVlSpec {
    pub image_token_id: i32,
    pub patch_size: usize,
    pub merge_size: usize,
    pub temporal_patch_size: usize,
    pub min_pixels: usize,
    pub max_pixels: usize,
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
    #[serde(default)]
    pub resample: Resampler,
}

/// The HF image processor the pipeline must match bit-exactly. Defaults to
/// the one a default server runs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Resampler {
    /// `Qwen2VLImageProcessor` / `…Fast` — torchvision on a uint8 tensor.
    #[default]
    AtenU8,
    /// `Qwen2VLImageProcessorPil`, behind `--disable-fast-image-processor`.
    Pil,
}

impl From<Resampler> for resize::Resample {
    fn from(r: Resampler) -> Self {
        match r {
            Resampler::AtenU8 => resize::Resample::AtenU8,
            Resampler::Pil => resize::Resample::Pil(resize::Filter::Bicubic),
        }
    }
}

pub struct QwenVlProcessor {
    spec: QwenVlSpec,
    /// Per-channel u8 → normalized-f32 lookup; see [`normalize_lut`].
    lut: [[f32; 256]; 3],
}

/// `1 / rescale_factor`; the consumer's spec resolution must reject any
/// other factor.
const INV_RESCALE: f32 = 255.0;

/// u8 → normalized f32, rounded as the mirrored processor rounds. The slow one
/// rescales then normalizes; the fast one folds the rescale into mean/std first
/// (`_fuse_mean_std_and_rescale_factor`), which differs on 128 of the 256 inputs.
fn normalize_lut(resample: Resampler, mean: f32, std: f32) -> [f32; 256] {
    match resample {
        Resampler::Pil => core::array::from_fn(|v| (v as f32 / INV_RESCALE - mean) / std),
        Resampler::AtenU8 => {
            let (mean, std) = (mean * INV_RESCALE, std * INV_RESCALE);
            core::array::from_fn(|v| (v as f32 - mean) / std)
        }
    }
}

impl QwenVlProcessor {
    pub fn new(spec: QwenVlSpec) -> Result<Self> {
        if spec.patch_size == 0 || spec.merge_size == 0 || spec.temporal_patch_size == 0 {
            return Err(MmError::invalid_input(
                "qwen_vl spec: sizes must be positive",
            ));
        }
        if spec.min_pixels == 0 || spec.min_pixels > spec.max_pixels {
            return Err(MmError::invalid_input(
                "qwen_vl spec: min_pixels must be positive and no greater than max_pixels",
            ));
        }
        if spec
            .image_std
            .iter()
            .any(|std| !std.is_finite() || *std <= 0.0)
        {
            return Err(MmError::invalid_input(
                "qwen_vl spec: image_std values must be finite and positive",
            ));
        }
        if spec.image_mean.iter().any(|mean| !mean.is_finite()) {
            return Err(MmError::invalid_input(
                "qwen_vl spec: image_mean values must be finite",
            ));
        }
        let lut = core::array::from_fn(|c| {
            normalize_lut(spec.resample, spec.image_mean[c], spec.image_std[c])
        });
        Ok(Self { spec, lut })
    }

    pub fn from_spec_json(json: &str) -> Result<Self> {
        let spec: QwenVlSpec = serde_json::from_str(json)
            .map_err(|error| MmError::invalid_input_with_source("invalid qwen_vl spec", error))?;
        Self::new(spec)
    }

    fn factor(&self) -> usize {
        self.spec.patch_size * self.spec.merge_size
    }

    /// Patch grid of the resized image, `[t, h, w]` with `t = 1` for a still.
    fn grid(&self, resized_h: usize, resized_w: usize) -> [u32; 3] {
        let (gh, gw) = (
            resized_h / self.spec.patch_size,
            resized_w / self.spec.patch_size,
        );
        [1, gh as u32, gw as u32]
    }

    /// The ViT merges `merge_size²` patches per token.
    fn tokens_per_image(&self, grid: &[u32; 3]) -> usize {
        (grid[0] as usize * grid[1] as usize * grid[2] as usize)
            / (self.spec.merge_size * self.spec.merge_size)
    }

    /// Floats per flattened patch: `C * tps * ps * ps`.
    fn patch_dim(&self) -> usize {
        3 * self.spec.temporal_patch_size * self.spec.patch_size * self.spec.patch_size
    }

    /// HF flatten: patches ordered `(gh/m, gw/m, m, m)`, features `(C, tps,
    /// ps, ps)`; parallel over merged-block rows.
    fn patchify(&self, rgb: &[u8], h: usize, w: usize) -> Vec<f32> {
        let (ps, m, tps) = (
            self.spec.patch_size,
            self.spec.merge_size,
            self.spec.temporal_patch_size,
        );
        let (gh, gw) = (h / ps, w / ps);
        let dim = self.patch_dim();
        let block_row = gw * m * dim; // one merged-block row of patches
        let mut out = vec![0.0f32; gh * gw * dim];

        execution::for_chunks_mut(&mut out, block_row, |i, chunk| {
            let mut p = 0;
            for j in 0..gw / m {
                for mh in 0..m {
                    for mw in 0..m {
                        let y0 = (i * m + mh) * ps;
                        let x0 = (j * m + mw) * ps;
                        let patch = &mut chunk[p * dim..(p + 1) * dim];
                        for (c, ch) in patch.chunks_exact_mut(tps * ps * ps).enumerate() {
                            for py in 0..ps {
                                let src = ((y0 + py) * w + x0) * 3 + c;
                                for px in 0..ps {
                                    ch[py * ps + px] = self.lut[c][rgb[src + px * 3] as usize];
                                }
                            }
                            // Temporal copies of a still are duplicates.
                            let (t0, rest) = ch.split_at_mut(ps * ps);
                            for frame in rest.chunks_exact_mut(ps * ps) {
                                frame.copy_from_slice(t0);
                            }
                        }
                        p += 1;
                    }
                }
            }
        });
        out
    }
}

impl MmFamilyProcessor for QwenVlProcessor {
    fn num_media_tokens(&self, media: &MediaMetadata) -> Result<usize> {
        match media {
            MediaMetadata::Image { width, height } => {
                let (th, tw) = smart_resize(
                    *height as usize,
                    *width as usize,
                    self.factor(),
                    self.spec.min_pixels,
                    self.spec.max_pixels,
                )?;
                Ok(self.tokens_per_image(&self.grid(th, tw)))
            }
            _ => Err(MmError::unsupported(
                "qwen_vl: only image token accounting is supported",
            )),
        }
    }

    fn process_item(&self, media: &DecodedMedia) -> Result<ProcessedItem> {
        let DecodedMedia::Image { rgb, height, width } = media;
        let (h, w) = (*height, *width);
        if h.checked_mul(w).and_then(|n| n.checked_mul(3)) != Some(rgb.len()) {
            return Err(MmError::invalid_input(
                "qwen_vl: rgb length does not match height * width * 3",
            ));
        }
        let (th, tw) = smart_resize(
            h,
            w,
            self.factor(),
            self.spec.min_pixels,
            self.spec.max_pixels,
        )?;
        let resized;
        let data = if (th, tw) != (h, w) {
            resized = resize::resize_rgb(rgb, h, w, th, tw, self.spec.resample.into());
            &resized
        } else {
            rgb.as_slice()
        };
        let grid = self.grid(th, tw);
        let num_patches = grid[1] as usize * grid[2] as usize;
        Ok(ProcessedItem {
            modality: media.modality(),
            feature_token_count: self.tokens_per_image(&grid),
            feature: Tensor {
                shape: vec![num_patches, self.patch_dim()],
                data: TensorData::F32(self.patchify(data, th, tw)),
            },
            aux: vec![(
                "image_grid_thw".to_string(),
                Tensor {
                    shape: vec![3],
                    data: TensorData::I64(grid.iter().map(|&g| g as i64).collect()),
                },
            )],
            geometry: Some(Geometry::Grid(grid)),
        })
    }

    fn layout(&self, input_ids: &[i32], items: &[ProcessedItem]) -> Result<TokenLayout> {
        let counts = items
            .iter()
            .map(|item| item.feature_token_count)
            .collect::<Vec<_>>();
        token_layout::layout_by_placeholder(input_ids, self.spec.image_token_id, &counts)
    }

    fn positions(
        &self,
        input_len: usize,
        offsets: &[(u32, u32)],
        items: &[ProcessedItem],
    ) -> Result<PositionOutput> {
        if offsets.len() != items.len() {
            return Err(MmError::invalid_input(
                "qwen_vl: offset and item counts differ",
            ));
        }
        let mrope_items = offsets
            .iter()
            .zip(items)
            .map(|(&(start, end), item)| match &item.geometry {
                Some(Geometry::Grid(grid)) => Ok(MropeItem {
                    start,
                    end,
                    grid: *grid,
                }),
                None => Err(MmError::invalid_input("qwen_vl: item is missing its grid")),
            })
            .collect::<Result<Vec<_>>>()?;
        let (positions, delta) = mrope_image_only(input_len, &mrope_items, self.spec.merge_size)?;
        Ok(PositionOutput::MRope { positions, delta })
    }
}

/// HF's `smart_resize`: round to multiples of `factor`, then adjust toward
/// the pixel budget. Clamping thin images can exceed `max_pixels`.
pub fn smart_resize(
    height: usize,
    width: usize,
    factor: usize,
    min_pixels: usize,
    max_pixels: usize,
) -> Result<(usize, usize)> {
    if factor == 0 || min_pixels == 0 || min_pixels > max_pixels {
        return Err(MmError::invalid_input(
            "smart_resize: invalid factor or pixel bounds",
        ));
    }
    let (h, w) = (height as f64, width as f64);
    if height == 0 || width == 0 {
        return Err(MmError::invalid_input("empty image"));
    }
    let ratio = h.max(w) / h.min(w);
    if ratio > MAX_RATIO {
        return Err(MmError::invalid_input(format!(
            "absolute aspect ratio must be smaller than {MAX_RATIO}, got {ratio}"
        )));
    }
    let f = factor as f64;
    let mut h_bar = ((h / f).round_ties_even() * f) as usize;
    let mut w_bar = ((w / f).round_ties_even() * f) as usize;
    // Rounded metadata dimensions can have an area larger than usize::MAX.
    let pixels = h_bar as u128 * w_bar as u128;
    if pixels > max_pixels as u128 {
        let beta = (h * w / max_pixels as f64).sqrt();
        h_bar = (((h / beta / f).floor() * f) as usize).max(factor);
        w_bar = (((w / beta / f).floor() * f) as usize).max(factor);
    } else if pixels < min_pixels as u128 {
        let beta = (min_pixels as f64 / (h * w)).sqrt();
        h_bar = ((h * beta / f).ceil() * f) as usize;
        w_bar = ((w * beta / f).ceil() * f) as usize;
    }
    Ok((h_bar, w_bar))
}

/// Image-only M-RoPE (the image branch of `MRotaryEmbedding.get_rope_index`):
/// text runs sequentially on all three rows, each image spans `(t, h/m, w/m)`
/// index grids, and positions advance by the grid's max past an image.
/// Returns row-major `[3, input_len]` positions and the delta
/// (`max + 1 - input_len`). `items` must be in prompt order.
pub fn mrope_image_only(
    input_len: usize,
    items: &[MropeItem],
    merge_size: usize,
) -> Result<(Vec<i64>, i64)> {
    if merge_size == 0 {
        return Err(MmError::invalid_input("mrope: merge_size must be positive"));
    }
    let len = input_len;
    let mut pos = vec![0i64; 3 * len];
    let fill_text = |st: usize, n: usize, base: i64, pos: &mut [i64]| {
        for k in 0..n {
            let v = base + k as i64;
            pos[st + k] = v;
            pos[len + st + k] = v;
            pos[2 * len + st + k] = v;
        }
    };
    let mut st = 0usize;
    let mut next_pos = 0i64;
    for item in items {
        let (start, end) = (item.start as usize, item.end as usize);
        if start < st || end < start || end >= len {
            return Err(MmError::invalid_input(format!(
                "mrope: item range ({start},{end}) out of order/bounds"
            )));
        }
        fill_text(st, start - st, next_pos, &mut pos);
        next_pos += (start - st) as i64;

        if item.grid[0] != 1
            || !(item.grid[1] as usize).is_multiple_of(merge_size)
            || !(item.grid[2] as usize).is_multiple_of(merge_size)
        {
            return Err(MmError::invalid_input("mrope: invalid image grid"));
        }
        let gh = item.grid[1] as usize / merge_size;
        let gw = item.grid[2] as usize / merge_size;
        if gh * gw != end - start + 1 {
            return Err(MmError::invalid_input(
                "mrope: token span does not match grid",
            ));
        }
        for hi in 0..gh {
            for wi in 0..gw {
                let idx = start + hi * gw + wi;
                pos[idx] = next_pos;
                pos[len + idx] = next_pos + hi as i64;
                pos[2 * len + idx] = next_pos + wi as i64;
            }
        }
        next_pos += gh.max(gw) as i64;
        st = end + 1;
    }
    if st < len {
        fill_text(st, len - st, next_pos, &mut pos);
    }
    let max = pos.iter().copied().max().unwrap_or(-1);
    Ok((pos, max + 1 - len as i64))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::processor::Modality;

    fn valid_spec() -> QwenVlSpec {
        QwenVlSpec {
            image_token_id: 0,
            patch_size: 14,
            merge_size: 2,
            temporal_patch_size: 2,
            min_pixels: 56 * 56,
            max_pixels: 28 * 28 * 1280,
            image_mean: [0.481_454_66, 0.457_827_5, 0.408_210_73],
            image_std: [0.268_629_54, 0.261_302_6, 0.275_777_1],
            resample: Resampler::AtenU8,
        }
    }

    fn tiny_spec() -> QwenVlSpec {
        QwenVlSpec {
            image_token_id: 1,
            patch_size: 2,
            merge_size: 2,
            temporal_patch_size: 2,
            min_pixels: 4,
            max_pixels: 1 << 30,
            image_mean: [0.0; 3],
            image_std: [1.0; 3],
            resample: Resampler::default(),
        }
    }

    #[test]
    fn rejects_invalid_pixel_bounds() {
        for (min_pixels, max_pixels) in [(0, 1), (2, 1)] {
            let mut spec = valid_spec();
            spec.min_pixels = min_pixels;
            spec.max_pixels = max_pixels;

            assert!(matches!(
                QwenVlProcessor::new(spec),
                Err(MmError::InvalidInput { .. })
            ));
        }
    }

    #[test]
    fn rejects_invalid_image_std() {
        for invalid_std in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            let mut spec = valid_spec();
            spec.image_std[1] = invalid_std;

            assert!(matches!(
                QwenVlProcessor::new(spec),
                Err(MmError::InvalidInput { .. })
            ));
        }
    }

    #[test]
    fn rejects_non_finite_image_mean() {
        for invalid_mean in [f32::NAN, f32::INFINITY] {
            let mut spec = valid_spec();
            spec.image_mean[1] = invalid_mean;

            assert!(matches!(
                QwenVlProcessor::new(spec),
                Err(MmError::InvalidInput { .. })
            ));
        }
    }

    /// The fused and unfused normalize forms are not interchangeable: with
    /// mean = std = 0.5 they disagree on 128 of the 256 u8 inputs, so picking
    /// the wrong one silently costs bit-exactness with the HF processor.
    #[test]
    fn normalize_lut_differs_per_resampler() {
        let pil = normalize_lut(Resampler::Pil, 0.5, 0.5);
        let aten = normalize_lut(Resampler::AtenU8, 0.5, 0.5);
        assert_eq!(pil.iter().zip(aten).filter(|(p, a)| *p != a).count(), 128);
        // Both still span [-1, 1] — this is rounding, not a scale error.
        for lut in [pil, aten] {
            assert_eq!(lut[0], -1.0);
            assert_eq!(lut[255], 1.0);
        }
    }

    #[test]
    fn smart_resize_matches_python_reference() {
        // Values from the Python `smart_resize` (qwen_vl.py) run offline.
        assert_eq!(
            smart_resize(1365, 2048, 28, 3136, 12845056).unwrap(),
            (1372, 2044)
        );
        assert_eq!(
            smart_resize(100, 100, 28, 3136, 12845056).unwrap(),
            (112, 112)
        );
        // Downscale branch: 4000x3000 exceeds 1280*28*28 → floor_by_factor.
        assert_eq!(
            smart_resize(3000, 4000, 28, 3136, 1003520).unwrap(),
            (840, 1148)
        );
        // Upscale branch: tiny image below min_pixels → ceil_by_factor.
        assert_eq!(smart_resize(20, 20, 28, 3136, 12845056).unwrap(), (56, 56));
        // Qwen3.5 factors (patch 16 * merge 2, min 65536, max 16777216).
        assert_eq!(
            smart_resize(1365, 2048, 32, 65536, 16777216).unwrap(),
            (1376, 2048)
        );
        // Banker's rounding tie: 48/32 = 1.5 rounds to 2 (even), not 1.
        assert_eq!(smart_resize(4000, 48, 32, 4, 1 << 30).unwrap(), (4000, 64));
        // Extreme aspect ratio rejected.
        assert!(smart_resize(10000, 10, 28, 3136, 12845056).is_err());
    }

    #[test]
    fn smart_resize_thin_images_match_hf() {
        assert_eq!(smart_resize(10, 2000, 28, 3136, 3136).unwrap(), (28, 812));
        assert_eq!(smart_resize(28, 5600, 28, 3136, 3136).unwrap(), (28, 784));
    }

    #[test]
    fn num_media_tokens_handles_large_dimensions() {
        let proc = QwenVlProcessor::new(QwenVlSpec {
            patch_size: 16,
            min_pixels: 65536,
            max_pixels: 16777216,
            ..valid_spec()
        })
        .unwrap();
        // Rounding these dimensions to factor 32 gives an area of 2^64.
        // HF downsizes to 4096x4096, producing 16384 merged image tokens.
        assert_eq!(
            proc.num_media_tokens(&MediaMetadata::Image {
                width: u32::MAX,
                height: u32::MAX,
            })
            .unwrap(),
            16384
        );
    }

    /// The consumer's message layer gates modalities on what a family
    /// declares, so a family gaining video/audio support must not silently
    /// inherit the images-only default.
    #[test]
    fn qwen_declares_images_only() {
        let caps = QwenVlProcessor::new(tiny_spec()).unwrap().capabilities();
        assert!(caps.supports(Modality::Image));
        assert!(!caps.supports(Modality::Video) && !caps.supports(Modality::Audio));
    }

    /// Routers must get the exact expanded token count from header metadata
    /// alone — the §2.3 example: 100×76 → 112×84 → 6×8 grid → 12 tokens.
    #[test]
    fn num_media_tokens_matches_process_item() {
        let proc = QwenVlProcessor::new(valid_spec()).unwrap();
        let counted = proc
            .num_media_tokens(&MediaMetadata::Image {
                width: 100,
                height: 76,
            })
            .unwrap();
        assert_eq!(counted, 12);

        let item = proc
            .process_item(&DecodedMedia::Image {
                rgb: vec![7u8; 76 * 100 * 3],
                height: 76,
                width: 100,
            })
            .unwrap();
        assert_eq!(item.feature_token_count, counted);
        assert_eq!(item.feature.shape, vec![48, 1176]);
    }

    #[test]
    fn process_item_rejects_mismatched_rgb_length() {
        let proc = QwenVlProcessor::new(valid_spec()).unwrap();
        // 112x84 needs no resize, so a short buffer would reach patchify.
        for len in [112 * 84 * 3 - 1, 112 * 84 * 3 + 3] {
            assert!(matches!(
                proc.process_item(&DecodedMedia::Image {
                    rgb: vec![7u8; len],
                    height: 112,
                    width: 84,
                }),
                Err(MmError::InvalidInput { .. })
            ));
        }
    }

    #[test]
    fn positions_reject_unpaired_items() {
        let proc = QwenVlProcessor::new(valid_spec()).unwrap();
        let item = proc
            .process_item(&DecodedMedia::Image {
                rgb: vec![7u8; 76 * 100 * 3],
                height: 76,
                width: 100,
            })
            .unwrap();
        assert!(matches!(
            proc.positions(16, &[], &[item]),
            Err(MmError::InvalidInput { .. })
        ));
    }

    #[test]
    fn patchify_layout_matches_hf_order() {
        // 4x8 image, ps=2, m=2, tps=2 → gh=2, gw=4, dim=3*2*2*2=24.
        // Pixel value encodes its (y, x): v = y*16 + x*2 (fits u8).
        let (h, w) = (4usize, 8usize);
        let mut rgb = vec![0u8; h * w * 3];
        for y in 0..h {
            for x in 0..w {
                for c in 0..3 {
                    rgb[(y * w + x) * 3 + c] = (y * 16 + x * 2 + c) as u8;
                }
            }
        }
        let proc = QwenVlProcessor::new(tiny_spec()).unwrap();
        let pv = proc.patchify(&rgb, h, w);
        let dim = 24; // 3 * tps * ps * ps
        assert_eq!(pv.len(), 2 * 4 * dim);

        // Patch order (gh/m=1, gw/m=2, m, m): patch 0 = block(0,0) offset (0,0),
        // patch 1 = (0,0)+(0,1) → x0=2, patch 2 = (0,0)+(1,0) → y0=2,
        // patch 4 = block(0,1) → x0=4.
        let lut = |y: usize, x: usize, c: usize| ((y * 16 + x * 2 + c) as f32) / 255.0;
        // patch 1, channel 0, t=0, (py=0, px=0) → pixel (0, 2).
        assert_eq!(pv[dim], lut(0, 2, 0));
        // patch 2, channel 0, t=0, (0,0) → pixel (2, 0).
        assert_eq!(pv[2 * dim], lut(2, 0, 0));
        // patch 4, channel 0 → pixel (0, 4).
        assert_eq!(pv[4 * dim], lut(0, 4, 0));
        // Temporal duplicate: t=1 block equals t=0 block.
        let ps2 = 4; // ps*ps
        assert_eq!(pv[dim + ps2], pv[dim]);
        // Channel 1 block of patch 0 → same pixel, c=1.
        assert_eq!(pv[2 * ps2], lut(0, 0, 1)); // c stride = tps*ps*ps = 8
    }

    #[test]
    fn mrope_image_only_matches_reference() {
        // 3 text tokens, image of grid [1, 4, 6] (m=2 → 2x3 = 6 tokens), 2 text.
        // input: [T T T I I I I I I T T], len 11.
        let items = [MropeItem {
            start: 3,
            end: 8,
            grid: [1, 4, 6],
        }];
        let (pos, delta) = mrope_image_only(11, &items, 2).unwrap();
        let len = 11;
        // Text prefix 0..3: all rows 0,1,2.
        for k in 0..3 {
            assert_eq!(
                (pos[k], pos[len + k], pos[2 * len + k]),
                (k as i64, k as i64, k as i64)
            );
        }
        // Image tokens: t=0, h in 0..2, w in 0..3, +3 offset.
        assert_eq!((pos[3], pos[len + 3], pos[2 * len + 3]), (3, 3, 3));
        assert_eq!((pos[4], pos[len + 4], pos[2 * len + 4]), (3, 3, 4));
        assert_eq!((pos[6], pos[len + 6], pos[2 * len + 6]), (3, 4, 3));
        // Text tail resumes at 3 + max(1,2,3) = 6.
        assert_eq!((pos[9], pos[len + 9], pos[2 * len + 9]), (6, 6, 6));
        assert_eq!((pos[10], pos[len + 10], pos[2 * len + 10]), (7, 7, 7));
        // delta = max + 1 - len = 7 + 1 - 11.
        assert_eq!(delta, -3);
    }

    #[test]
    fn mrope_rejects_invalid_geometry() {
        for (start, end, grid, merge) in [
            (2, 1, [1, 2, 2], 2),
            (0, 0, [1, 2, 2], 0),
            (0, 0, [1, 3, 2], 2),
            (0, 1, [2, 2, 2], 2),
        ] {
            assert!(mrope_image_only(3, &[MropeItem { start, end, grid }], merge).is_err());
        }
    }
}
