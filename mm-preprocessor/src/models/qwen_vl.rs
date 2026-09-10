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

    fn tokens_per_image(&self, grid: &[u32; 3]) -> usize {
        (grid[0] as usize * grid[1] as usize * grid[2] as usize)
            / (self.spec.merge_size * self.spec.merge_size)
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
        let dim = 3 * tps * ps * ps;
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
                        for c in 0..3 {
                            let ch = &mut patch[c * tps * ps * ps..];
                            for py in 0..ps {
                                let src = ((y0 + py) * w + x0) * 3 + c;
                                for px in 0..ps {
                                    ch[py * ps + px] = self.lut[c][rgb[src + px * 3] as usize];
                                }
                            }
                            // Temporal copies of a still are duplicates.
                            let (t0, rest) = ch.split_at_mut(ps * ps);
                            for t in 0..tps - 1 {
                                rest[t * ps * ps..(t + 1) * ps * ps].copy_from_slice(t0);
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
                let (gh, gw) = (th / self.spec.patch_size, tw / self.spec.patch_size);
                Ok(self.tokens_per_image(&[1, gh as u32, gw as u32]))
            }
            _ => Err(MmError::unsupported(
                "qwen_vl: only image token accounting is supported",
            )),
        }
    }

    fn process_item(&self, media: &DecodedMedia) -> Result<ProcessedItem> {
        let DecodedMedia::Image { rgb, height, width } = media;
        let (h, w) = (*height, *width);
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
        let (gh, gw) = (th / self.spec.patch_size, tw / self.spec.patch_size);
        // `smart_resize` guarantees both: dims are positive and divisible by
        // `patch_size * merge_size`. `patchify` indexes on that (and the `dim`
        // division below needs a non-empty grid), so fail loudly rather than
        // panic if a future spec change breaks the guarantee.
        if gh == 0 || gw == 0 || gh % self.spec.merge_size != 0 || gw % self.spec.merge_size != 0 {
            return Err(MmError::internal(format!(
                "qwen_vl: patch grid {gh}x{gw} is empty or not a multiple of merge_size {}",
                self.spec.merge_size
            )));
        }
        let pixel_values = self.patchify(data, th, tw);
        let dim = pixel_values.len() / (gh * gw);
        let grid = [1, gh as u32, gw as u32];
        Ok(ProcessedItem {
            modality: media.modality(),
            feature_token_count: self.tokens_per_image(&grid),
            feature: Tensor {
                shape: vec![gh * gw, dim],
                data: TensorData::F32(pixel_values),
            },
            aux: vec![(
                "image_grid_thw".to_string(),
                Tensor {
                    shape: vec![3],
                    data: TensorData::I64(vec![1, gh as i64, gw as i64]),
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
        let mrope_items = offsets
            .iter()
            .zip(items)
            .map(|(&(start, end), item)| match &item.geometry {
                Some(Geometry::Grid(grid)) => Ok(MropeItem {
                    start,
                    end,
                    grid: *grid,
                }),
                None => Err(MmError::internal("qwen_vl: item is missing its grid")),
            })
            .collect::<Result<Vec<_>>>()?;
        let (positions, delta) = mrope_image_only(input_len, &mrope_items, self.spec.merge_size)?;
        Ok(PositionOutput::MRope { positions, delta })
    }
}

/// Python-`round()` (round-half-to-even), which `round_by_factor` relies on.
fn round_half_even(x: f64) -> f64 {
    if (x - x.trunc()).abs() == 0.5 {
        (x / 2.0).round() * 2.0
    } else {
        x.round()
    }
}

/// Qwen's `smart_resize`: dims divisible by `factor`, total pixels within
/// `[min_pixels, max_pixels]`, aspect ratio preserved as closely as possible.
/// Matches the Python reference exactly (including round-half-to-even);
/// `Err` when a very thin image would floor a side to 0.
pub fn smart_resize(
    height: usize,
    width: usize,
    factor: usize,
    min_pixels: usize,
    max_pixels: usize,
) -> Result<(usize, usize)> {
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
    let mut h_bar = ((round_half_even(h / f) * f) as usize).max(factor);
    let mut w_bar = ((round_half_even(w / f) * f) as usize).max(factor);
    if h_bar * w_bar > max_pixels {
        let beta = (h * w / max_pixels as f64).sqrt();
        h_bar = ((h / beta / f).floor() * f) as usize;
        w_bar = ((w / beta / f).floor() * f) as usize;
    } else if h_bar * w_bar < min_pixels {
        let beta = (min_pixels as f64 / (h * w)).sqrt();
        h_bar = ((h * beta / f).ceil() * f) as usize;
        w_bar = ((w * beta / f).ceil() * f) as usize;
    }
    // The downscale branch floors without a lower clamp (as Python does), so a
    // very thin image against a small `max_pixels` can floor a side to 0.
    // Python then fails inside PIL's resize; here it would reach the resize
    // coefficient math (overflow panic in debug, garbage in release) and the
    // `dim = len / (gh * gw)` division, so reject it as a request error.
    if h_bar == 0 || w_bar == 0 {
        return Err(MmError::invalid_input(format!(
            "smart_resize: {height}x{width} degenerates to {h_bar}x{w_bar} at \
             max_pixels={max_pixels}; image is too thin for this pixel budget"
        )));
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
        if start < st || end >= len {
            return Err(MmError::internal(format!(
                "mrope: item range ({start},{end}) out of order/bounds"
            )));
        }
        fill_text(st, start - st, next_pos, &mut pos);
        next_pos += (start - st) as i64;

        let t = item.grid[0] as usize;
        let gh = item.grid[1] as usize / merge_size;
        let gw = item.grid[2] as usize / merge_size;
        if t * gh * gw != end - start + 1 {
            return Err(MmError::internal("mrope: token span does not match grid"));
        }
        for ti in 0..t {
            for hi in 0..gh {
                for wi in 0..gw {
                    let idx = start + (ti * gh + hi) * gw + wi;
                    pos[idx] = next_pos + ti as i64;
                    pos[len + idx] = next_pos + hi as i64;
                    pos[2 * len + idx] = next_pos + wi as i64;
                }
            }
        }
        next_pos += (t.max(gh).max(gw)) as i64;
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

    /// A thin image against a small `max_pixels` floors one side to 0. That
    /// must reject the request, never reach the resize coefficient math and
    /// panic on a worker thread (`attempt to multiply with overflow`).
    #[test]
    fn degenerate_target_is_rejected_not_panicked() {
        // Aspect ratio 200 is exactly at MAX_RATIO, so it passes that guard;
        // 10 / beta then floors to 0 with factor 28.
        assert!(smart_resize(10, 2000, 28, 3136, 3136).is_err());

        let mut spec = tiny_spec();
        spec.patch_size = 14;
        spec.min_pixels = 3136;
        spec.max_pixels = 3136;
        let proc = QwenVlProcessor::new(spec).unwrap();
        let err = proc
            .process_item(&DecodedMedia::Image {
                rgb: vec![0u8; 10 * 2000 * 3],
                height: 10,
                width: 2000,
            })
            .err()
            .expect("degenerate geometry must be an Err, never a panic");
        assert!(
            err.to_string().contains("smart_resize"),
            "unexpected error: {err}"
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
}
