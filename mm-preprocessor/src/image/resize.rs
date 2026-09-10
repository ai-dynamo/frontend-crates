// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bit-exact resamplers: PIL's fixed-point kernels and torchvision's uint8
//! antialias path. Which one a family selects is part of its spec — the two
//! quantize weights differently (i32 vs per-axis i16), so their outputs are
//! NOT interchangeable at the byte level.

use crate::execution;

/// PIL's `PRECISION_BITS` for 8-bit images: weights quantized to i32.
const PIL_PRECISION_BITS: u32 = 32 - 8 - 2;

/// Resampling filters, bit-exact clones of PIL's kernels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Filter {
    /// support 3.0 — PIL `LANCZOS`.
    Lanczos,
    /// support 2.0, a = -0.5 — PIL `BICUBIC`.
    Bicubic,
}

/// A resampler reproduced bit-exactly. Both share PIL's geometry, kernels and
/// per-pass u8 rounding, and differ only in how the weights are quantized.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Resample {
    /// PIL `Image.resize`, i32 weights.
    Pil(Filter),
    /// ATen's uint8 antialias bicubic — torchvision `resize(antialias=True)` on
    /// a uint8 tensor. i16 weights, so it rounds unlike `Pil(Bicubic)`.
    AtenU8,
}

impl Resample {
    fn filter(self) -> Filter {
        match self {
            Resample::Pil(filter) => filter,
            Resample::AtenU8 => Filter::Bicubic,
        }
    }

    /// Fixed-point precision for one axis's already-normalized weights. ATen
    /// (`_compute_weights_precision`) takes the widest that stays inside i16.
    fn precision(self, weights: &[f64]) -> u32 {
        match self {
            Resample::Pil(_) => PIL_PRECISION_BITS,
            Resample::AtenU8 => {
                let wmax = weights.iter().fold(0.0f64, |m, w| m.max(w.abs()));
                (1..PIL_PRECISION_BITS)
                    .take_while(|&p| (0.5 + wmax * (1u64 << p) as f64) < (1 << 15) as f64)
                    .last()
                    .unwrap_or(1)
            }
        }
    }
}

impl Filter {
    fn support(self) -> f64 {
        match self {
            Filter::Lanczos => 3.0,
            Filter::Bicubic => 2.0,
        }
    }

    fn eval(self, x: f64) -> f64 {
        match self {
            Filter::Lanczos => lanczos(x),
            Filter::Bicubic => bicubic(x),
        }
    }
}

fn sinc(x: f64) -> f64 {
    if x == 0.0 {
        return 1.0;
    }
    let x = x * std::f64::consts::PI;
    x.sin() / x
}

fn lanczos(x: f64) -> f64 {
    if (-3.0..3.0).contains(&x) {
        sinc(x) * sinc(x / 3.0)
    } else {
        0.0
    }
}

fn bicubic(x: f64) -> f64 {
    const A: f64 = -0.5;
    let x = x.abs();
    if x < 1.0 {
        ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        (((x - 5.0) * x + 8.0) * x - 4.0) * A
    } else {
        0.0
    }
}

struct Coeffs {
    bounds: Vec<(usize, usize)>,
    kk: Vec<i32>,
    ksize: usize,
    prec: u32,
}

fn precompute_coeffs(in_size: usize, out_size: usize, resample: Resample) -> Coeffs {
    let filter = resample.filter();
    let scale = in_size as f64 / out_size as f64;
    let filterscale = if scale < 1.0 { 1.0 } else { scale };
    let support = filter.support() * filterscale;
    let ksize = support.ceil() as usize * 2 + 1;
    let ss = 1.0 / filterscale;

    let mut kkf = vec![0.0f64; out_size * ksize];
    let mut bounds = vec![(0usize, 0usize); out_size];
    for xx in 0..out_size {
        let center = (xx as f64 + 0.5) * scale;
        let mut xmin = (center - support + 0.5) as i32;
        if xmin < 0 {
            xmin = 0;
        }
        let mut xmax = (center + support + 0.5) as i32;
        if xmax > in_size as i32 {
            xmax = in_size as i32;
        }
        let count = (xmax - xmin) as usize;
        let k = &mut kkf[xx * ksize..(xx + 1) * ksize];
        let mut ww = 0.0f64;
        for (x, kv) in k[..count].iter_mut().enumerate() {
            let w = filter.eval((x as f64 + xmin as f64 - center + 0.5) * ss);
            *kv = w;
            ww += w;
        }
        if ww != 0.0 {
            for kv in k[..count].iter_mut() {
                *kv /= ww;
            }
        }
        bounds[xx] = (xmin as usize, count);
    }

    let prec = resample.precision(&kkf);
    let factor = (1i64 << prec) as f64;
    let kk = kkf
        .iter()
        .map(|&v| {
            if v < 0.0 {
                (-0.5 + v * factor) as i32
            } else {
                (0.5 + v * factor) as i32
            }
        })
        .collect();
    Coeffs {
        bounds,
        kk,
        ksize,
        prec,
    }
}

#[inline]
fn clip8(v: i32, prec: u32) -> u8 {
    if v >= 1 << (prec + 8) {
        255
    } else if v <= 0 {
        0
    } else {
        (v >> prec) as u8
    }
}

fn resample_horizontal(src: &[u8], h: usize, w: usize, out_w: usize, c: &Coeffs) -> Vec<u8> {
    let mut out = vec![0u8; h * out_w * 3];
    execution::for_chunks_mut(&mut out, out_w * 3, |y, row| {
        let src_row = &src[y * w * 3..(y + 1) * w * 3];
        for xx in 0..out_w {
            let (xmin, count) = c.bounds[xx];
            let k = &c.kk[xx * c.ksize..xx * c.ksize + count];
            let mut s = [1i32 << (c.prec - 1); 3];
            for (x, &coef) in k.iter().enumerate() {
                let p = (xmin + x) * 3;
                s[0] += src_row[p] as i32 * coef;
                s[1] += src_row[p + 1] as i32 * coef;
                s[2] += src_row[p + 2] as i32 * coef;
            }
            let o = xx * 3;
            row[o] = clip8(s[0], c.prec);
            row[o + 1] = clip8(s[1], c.prec);
            row[o + 2] = clip8(s[2], c.prec);
        }
    });
    out
}

fn resample_vertical(src: &[u8], w: usize, out_h: usize, c: &Coeffs) -> Vec<u8> {
    let mut out = vec![0u8; out_h * w * 3];
    execution::for_chunks_mut(&mut out, w * 3, |yy, row| {
        let (ymin, count) = c.bounds[yy];
        let k = &c.kk[yy * c.ksize..yy * c.ksize + count];
        for x in 0..w {
            let mut s = [1i32 << (c.prec - 1); 3];
            for (y, &coef) in k.iter().enumerate() {
                let p = ((ymin + y) * w + x) * 3;
                s[0] += src[p] as i32 * coef;
                s[1] += src[p + 1] as i32 * coef;
                s[2] += src[p + 2] as i32 * coef;
            }
            let o = x * 3;
            row[o] = clip8(s[0], c.prec);
            row[o + 1] = clip8(s[1], c.prec);
            row[o + 2] = clip8(s[2], c.prec);
        }
    });
    out
}

/// Separable resize of a flat HWC RGB buffer, bit-exact against `resample`.
///
/// Enters the fan-out pool once for both passes; the per-row `for_chunks_mut`
/// calls inside then reuse that entry rather than injecting a job per pass.
pub fn resize_rgb(
    src: &[u8],
    h: usize,
    w: usize,
    out_h: usize,
    out_w: usize,
    resample: Resample,
) -> Vec<u8> {
    execution::in_pool(move || resize_passes(src, h, w, out_h, out_w, resample))
}

fn resize_passes(
    src: &[u8],
    h: usize,
    w: usize,
    out_h: usize,
    out_w: usize,
    resample: Resample,
) -> Vec<u8> {
    // Per-axis coefficients — and, under `AtenU8`, a per-axis precision.
    let coeffs = |in_size, out_size| precompute_coeffs(in_size, out_size, resample);
    match (out_w != w, out_h != h) {
        (true, true) => {
            let tmp = resample_horizontal(src, h, w, out_w, &coeffs(w, out_w));
            resample_vertical(&tmp, out_w, out_h, &coeffs(h, out_h))
        }
        (true, false) => resample_horizontal(src, h, w, out_w, &coeffs(w, out_w)),
        (false, true) => resample_vertical(src, w, out_h, &coeffs(h, out_h)),
        (false, false) => src.to_vec(),
    }
}

pub fn resize_lanczos_rgb(src: &[u8], h: usize, w: usize, out_h: usize, out_w: usize) -> Vec<u8> {
    resize_rgb(src, h, w, out_h, out_w, Resample::Pil(Filter::Lanczos))
}

/// Long-edge rescale targeting `frac` of the long edge, optionally capped;
/// `(w, h)` in, `(w, h)` out.
pub fn scaled_dims(w: usize, h: usize, frac: Option<f64>, cap: Option<i64>) -> (usize, usize) {
    let Some(frac) = frac else {
        return (w, h);
    };
    let long_edge = w.max(h);
    if long_edge == 0 {
        return (w, h);
    }
    let mut target = long_edge as f64 * frac;
    if let Some(cap) = cap {
        let effective_cap = cap.max(long_edge as i64);
        target = target.min(effective_cap as f64);
    }
    let ratio = target / long_edge as f64;
    if ratio == 1.0 {
        return (w, h);
    }
    let scale = |v: usize| ((v as f64 * ratio + 0.5).floor() as i64).max(1) as usize;
    (scale(w), scale(h))
}

#[cfg(test)]
mod tests {
    use super::*;

    // A 6x8 RGB noise source and the outputs PIL and torchvision produce for
    // it, captured from the Python implementations this module mirrors. Noise
    // rather than a gradient: on a smooth ramp the two quantizations agree, so
    // a gradient fixture would pass even with the wrong weight precision.
    const H: usize = 6;
    const W: usize = 8;
    const OH: usize = 4;
    const OW: usize = 5;

    const SRC: [u8; 144] = [
        190, 59, 63, 127, 17, 131, 195, 223, 131, 117, 197, 23, 180, 231, 134, 169, 170, 223, 116,
        136, 248, 146, 177, 33, 145, 56, 122, 216, 106, 202, 86, 216, 83, 28, 146, 123, 147, 31,
        232, 241, 110, 121, 91, 155, 51, 23, 103, 231, 18, 219, 232, 245, 51, 29, 217, 145, 127, 1,
        230, 71, 16, 221, 60, 37, 244, 210, 22, 144, 160, 73, 69, 49, 53, 85, 14, 192, 25, 58, 139,
        237, 23, 131, 213, 62, 201, 68, 101, 141, 29, 40, 150, 12, 68, 176, 56, 46, 147, 67, 142,
        96, 120, 141, 81, 226, 121, 161, 49, 248, 19, 11, 62, 164, 189, 47, 93, 149, 229, 123, 215,
        145, 67, 109, 132, 15, 78, 102, 84, 96, 129, 53, 187, 251, 136, 93, 52, 105, 89, 164, 155,
        54, 64, 42, 79, 61,
    ];
    /// `Image.resize((5, 4), Image.BICUBIC)`.
    const PIL_BICUBIC: [u8; 60] = [
        173, 45, 116, 139, 174, 116, 132, 171, 116, 180, 144, 189, 101, 149, 136, 134, 119, 145,
        168, 150, 90, 41, 181, 107, 94, 164, 148, 54, 95, 108, 115, 72, 74, 144, 180, 77, 129, 115,
        98, 129, 81, 76, 143, 95, 102, 74, 92, 129, 69, 132, 154, 91, 98, 157, 125, 106, 106, 94,
        118, 110,
    ];
    /// `Image.resize((5, 4), Image.LANCZOS)`.
    const PIL_LANCZOS: [u8; 60] = [
        178, 39, 122, 130, 177, 119, 135, 168, 116, 189, 141, 196, 96, 154, 137, 138, 115, 146,
        172, 148, 81, 34, 187, 111, 100, 157, 146, 50, 88, 101, 116, 67, 71, 152, 183, 78, 128,
        123, 90, 121, 80, 78, 147, 100, 105, 73, 90, 126, 63, 136, 166, 92, 92, 163, 129, 103, 98,
        93, 124, 120,
    ];
    /// `torchvision resize([4, 5], BICUBIC, antialias=True)` on the uint8 tensor.
    const ATEN_U8: [u8; 60] = [
        173, 45, 116, 139, 174, 116, 132, 171, 116, 180, 144, 189, 101, 149, 136, 134, 119, 145,
        168, 150, 90, 41, 181, 107, 94, 164, 148, 54, 95, 108, 115, 72, 74, 143, 180, 77, 129, 115,
        98, 129, 81, 76, 143, 95, 102, 74, 92, 129, 69, 132, 154, 91, 98, 157, 125, 106, 106, 94,
        118, 110,
    ];
    /// `Image.resize((5, 6), Image.BICUBIC)` — the horizontal pass alone.
    const PIL_WIDTH_ONLY: [u8; 90] = [
        165, 35, 92, 162, 164, 113, 152, 223, 76, 159, 170, 230, 132, 158, 120, 179, 73, 157, 108,
        185, 117, 87, 89, 177, 198, 107, 117, 45, 127, 156, 112, 148, 150, 202, 127, 87, 6, 233,
        69, 29, 218, 180, 53, 95, 93, 109, 54, 32, 153, 184, 37, 164, 148, 80, 154, 22, 58, 165,
        37, 54, 126, 87, 140, 101, 178, 153, 91, 32, 149, 118, 149, 90, 112, 192, 185, 45, 94, 117,
        56, 107, 147, 95, 143, 156, 130, 76, 114, 87, 68, 61,
    ];
    /// `Image.resize((8, 4), Image.BICUBIC)` — the vertical pass alone.
    const PIL_HEIGHT_ONLY: [u8; 96] = [
        180, 51, 80, 157, 51, 164, 152, 223, 113, 87, 175, 60, 173, 154, 176, 203, 145, 185, 110,
        144, 176, 100, 153, 110, 56, 159, 182, 240, 68, 87, 168, 173, 102, 11, 206, 84, 74, 145,
        125, 112, 180, 168, 52, 137, 105, 59, 70, 113, 81, 90, 72, 167, 57, 79, 129, 231, 62, 138,
        157, 122, 122, 61, 79, 138, 102, 49, 120, 67, 137, 159, 113, 75, 99, 93, 140, 38, 97, 120,
        80, 141, 130, 93, 131, 255, 88, 61, 54, 127, 129, 123, 132, 92, 128, 68, 133, 95,
    ];

    #[test]
    fn matches_pil_and_torchvision_byte_for_byte() {
        let cases: [(Resample, usize, usize, &[u8]); 5] = [
            (Resample::Pil(Filter::Bicubic), OH, OW, &PIL_BICUBIC),
            (Resample::Pil(Filter::Lanczos), OH, OW, &PIL_LANCZOS),
            (Resample::AtenU8, OH, OW, &ATEN_U8),
            (Resample::Pil(Filter::Bicubic), H, OW, &PIL_WIDTH_ONLY),
            (Resample::Pil(Filter::Bicubic), OH, W, &PIL_HEIGHT_ONLY),
        ];
        for (resample, out_h, out_w, expected) in cases {
            let out = resize_rgb(&SRC, H, W, out_h, out_w, resample);
            assert_eq!(out, expected, "{resample:?} to {out_h}x{out_w}");
        }
        assert_eq!(
            resize_lanczos_rgb(&SRC, H, W, OH, OW),
            PIL_LANCZOS,
            "lanczos convenience wrapper"
        );
    }

    /// The two quantizations are not interchangeable — picking the wrong one
    /// for a family silently costs bit-exactness with its HF processor, and
    /// this is the byte that proves it on this fixture.
    #[test]
    fn pil_and_aten_bicubic_disagree() {
        let differing = PIL_BICUBIC
            .iter()
            .zip(ATEN_U8)
            .filter(|(pil, aten)| **pil != *aten)
            .count();
        assert_eq!(differing, 1);
    }

    #[test]
    fn identity_resize_copies_the_source() {
        assert_eq!(resize_rgb(&SRC, H, W, H, W, Resample::AtenU8), SRC);
    }

    #[test]
    fn scaled_dims_takes_the_long_edge_with_an_optional_cap() {
        // No fraction: untouched. Fraction alone: rounded half-up, floored at 1.
        assert_eq!(scaled_dims(80, 60, None, None), (80, 60));
        assert_eq!(scaled_dims(80, 60, Some(0.5), None), (40, 30));
        assert_eq!(scaled_dims(3, 1, Some(0.1), None), (1, 1));
        assert_eq!(scaled_dims(0, 0, Some(0.5), None), (0, 0));
        // The cap is raised to the source's own long edge, so it bounds
        // upscaling without ever forcing a downscale.
        assert_eq!(scaled_dims(80, 60, Some(2.0), None), (160, 120));
        assert_eq!(scaled_dims(80, 60, Some(2.0), Some(120)), (120, 90));
        assert_eq!(scaled_dims(80, 60, Some(0.9), Some(40)), (72, 54));
    }
}
