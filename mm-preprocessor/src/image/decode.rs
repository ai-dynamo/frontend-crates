// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// Decode encoded image bytes (jpeg/png/webp/gif/bmp — the formats the Python
/// PIL path commonly accepts) to `(HWC u8 RGB, height, width)`.
///
/// Samples deeper than 8 bits are rejected: PIL clips to 255 where a u8
/// conversion would rescale, so refusing is the only bit-exact answer.
pub fn decode_rgb(data: &[u8]) -> crate::Result<(Vec<u8>, usize, usize)> {
    let _ = data;
    todo!("pure-Rust decoders via the `image` crate")
}

/// `(height, width)` from the encoded header alone — no pixel decode (PIL's
/// lazy `Image.open(...).size`). Pairs with
/// Supplies [`MediaMetadata::Image`](crate::processor::MediaMetadata::Image)
/// for pixel-free token accounting.
pub fn dimensions(data: &[u8]) -> crate::Result<(usize, usize)> {
    let _ = data;
    todo!("header-only probe via the `image` crate reader")
}
