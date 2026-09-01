// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bit-exact resamplers: PIL's fixed-point kernels and torchvision's uint8
//! antialias path. Which one a family selects is part of its spec — the two
//! quantize weights differently (i32 vs per-axis i16), so their outputs are
//! NOT interchangeable at the byte level.

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

/// Separable resize of a flat HWC RGB buffer, bit-exact against `resample`.
pub fn resize_rgb(
    src: &[u8],
    h: usize,
    w: usize,
    out_h: usize,
    out_w: usize,
    resample: Resample,
) -> Vec<u8> {
    let _ = (src, h, w, out_h, out_w, resample);
    todo!("PR2: precomputed fixed-point coefficients, horizontal then vertical pass")
}

pub fn resize_lanczos_rgb(src: &[u8], h: usize, w: usize, out_h: usize, out_w: usize) -> Vec<u8> {
    resize_rgb(src, h, w, out_h, out_w, Resample::Pil(Filter::Lanczos))
}

/// Long-edge rescale targeting `frac` of the long edge, optionally capped;
/// `(w, h)` in, `(w, h)` out.
pub fn scaled_dims(w: usize, h: usize, frac: Option<f64>, cap: Option<i64>) -> (usize, usize) {
    let _ = (w, h, frac, cap);
    todo!("PR2: long-edge fraction with cap")
}
