// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Adapted from sgl-project/sglang PR #46563 (Apache-2.0), by Kan Wu and Zhihan Guo.

//! EXIF-aware alpha compositing used by SGLang's `smart_to_rgb`.

use super::{DecodeLimits, decode_rgb_exif_transposed};
use crate::MmError;
use image::{DynamicImage, ImageDecoder, ImageFormat, ImageReader};
use std::io::Cursor;

/// `utils/common.py::smart_to_rgb`: EXIF transpose, then any alpha composited
/// onto a background that contrasts with the opaque edge pixels.
pub fn decode_rgb_smart(
    data: &[u8],
    limits: &DecodeLimits,
) -> Result<(Vec<u8>, usize, usize), MmError> {
    let (rgb, h, w) = decode_rgb_exif_transposed(data, limits)?;
    match alpha(data)? {
        Some(alpha) if alpha.len() == h * w => {
            let rgba: Vec<u8> = (rgb.chunks_exact(3).zip(alpha))
                .flat_map(|(pixel, a)| [pixel[0], pixel[1], pixel[2], a])
                .collect();
            Ok((composite(&rgba, h, w), h, w))
        }
        Some(_) => Err(MmError::invalid_input(
            "glm5_next: alpha and RGB decodes disagree",
        )),
        None => Ok((rgb, h, w)),
    }
}

/// The alpha of PIL's `convert("RGBA")` for an image `smart_to_rgb` composites
/// (RGBA, LA, or `transparency`), oriented as Dynamo orients the RGB. A BMP
/// has alpha, in PIL and in `image` alike, only with an alpha bitmask.
fn alpha(data: &[u8]) -> Result<Option<Vec<u8>>, MmError> {
    let invalid = |error: image::ImageError| MmError::invalid_input(error.to_string());
    let format = match image::guess_format(data) {
        Ok(ImageFormat::Gif) => return gif_alpha(data),
        Ok(format @ (ImageFormat::Png | ImageFormat::WebP | ImageFormat::Bmp)) => format,
        _ => return Ok(None),
    };
    let decoder = ImageReader::with_format(Cursor::new(data), format)
        .into_decoder()
        .map_err(invalid)?;
    if !decoder.color_type().has_alpha() {
        return Ok(None);
    }
    let mut image = DynamicImage::from_decoder(decoder).map_err(invalid)?;
    image.apply_orientation(super::exif_orientation(data)?);
    Ok(Some(
        image.into_rgba8().pixels().map(|pixel| pixel[3]).collect(),
    ))
}

/// PIL opens a GIF's first frame as `P` with the frame's transparent index as
/// `transparency`, and fills the canvas outside the frame with it (Dynamo's `GifImage`).
fn gif_alpha(data: &[u8]) -> Result<Option<Vec<u8>>, MmError> {
    let invalid = |error: gif::DecodingError| MmError::invalid_input(error.to_string());
    let mut options = gif::DecodeOptions::new();
    options.set_color_output(gif::ColorOutput::Indexed);
    let mut decoder = options.read_info(data).map_err(invalid)?;
    let (screen_w, screen_h) = (usize::from(decoder.width()), usize::from(decoder.height()));
    let Some(frame) = decoder.read_next_frame().map_err(invalid)? else {
        return Ok(None);
    };
    let Some(transparent) = frame.transparent else {
        return Ok(None);
    };
    let (top, left) = (usize::from(frame.top), usize::from(frame.left));
    let frame_w = usize::from(frame.width);
    let w = screen_w.max(left + frame_w);
    let h = screen_h.max(top + usize::from(frame.height));
    let mut alpha = vec![0u8; h * w];
    for (y, row) in frame.buffer.chunks_exact(frame_w).enumerate() {
        let start = (top + y) * w + left;
        for (a, &index) in alpha[start..start + frame_w].iter_mut().zip(row) {
            *a = if index == transparent { 0 } else { 255 };
        }
    }
    Ok(Some(alpha))
}

/// `background.paste(image, mask=alpha)`, with PIL's `BLEND` / `DIV255` rounding.
fn composite(rgba: &[u8], h: usize, w: usize) -> Vec<u8> {
    let alpha = |x: usize, y: usize| &rgba[(y * w + x) * 4..][..4];
    let rows = (0..w)
        .step_by((w / 20).max(1))
        .flat_map(|x| [(x, 0), (x, h - 1)]);
    let columns = (0..h)
        .step_by((h / 20).max(1))
        .flat_map(|y| [(0, y), (w - 1, y)]);
    let edges: Vec<&[u8]> = (rows.chain(columns))
        .map(|(x, y)| alpha(x, y))
        .filter(|pixel| pixel[3] > 128)
        .collect();
    let sum: usize = edges
        .iter()
        .flat_map(|pixel| &pixel[..3])
        .map(|&v| v as usize)
        .sum();
    let background: u32 = match edges.len() {
        0 => 255,
        n if sum > 128 * 3 * n => 32,
        _ => 240,
    };
    (rgba.chunks_exact(4))
        .flat_map(|pixel| {
            let a = pixel[3] as u32;
            pixel[..3].iter().map(move |&c| {
                let t = background * (255 - a) + c as u32 * a + 128;
                (((t >> 8) + t) >> 8) as u8
            })
        })
        .collect()
}
