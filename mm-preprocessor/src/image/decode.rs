// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// Decode encoded image bytes (jpeg/png/webp/gif/bmp — the formats the Python
/// PIL path commonly accepts) to `(HWC u8 RGB, height, width)`.
///
/// Samples deeper than 8 bits are rejected: PIL clips to 255 where a u8
/// conversion would rescale, so refusing is the only bit-exact answer.
pub fn decode_rgb(data: &[u8]) -> Result<(Vec<u8>, usize, usize), String> {
    let _ = data;
    todo!("PR2: pure-Rust decoders via the `image` crate")
}
