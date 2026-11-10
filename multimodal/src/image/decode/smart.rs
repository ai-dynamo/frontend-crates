// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Adapted from sgl-project/sglang PR #46563 (Apache-2.0), by Kan Wu and Zhihan Guo.

//! EXIF-aware alpha compositing used by SGLang's `smart_to_rgb`.

use super::{DecodeLimits, GifImage, decode_rgb_exif_transposed};
use crate::MmError;
use image::{ImageFormat, ImageReader};
use std::io::Cursor;

/// `utils/common.py::smart_to_rgb`: EXIF transpose, then any alpha composited
/// onto a background that contrasts with the opaque edge pixels.
pub fn decode_rgb_smart(
    data: &[u8],
    limits: &DecodeLimits,
) -> Result<(Vec<u8>, usize, usize), MmError> {
    let format = super::format(data)?;
    match format {
        // Keep Pillow's JPEG backend, including its CMYK conversion.
        ImageFormat::Jpeg => return decode_rgb_exif_transposed(data, limits),
        ImageFormat::Gif => {
            let gif = GifImage::new(data)?;
            let (h, w) = gif.dimensions();
            limits.check(h, w)?;
            let rgb = if gif.frame.transparent.is_some() {
                composite(&gif.decode_rgba()?, h, w)
            } else {
                gif.decode_rgb()?
            };
            return Ok((rgb, h, w));
        }
        _ => {}
    }

    let (h, w) = super::dimensions(data)?;
    let orientation = super::exif_orientation(data)?;
    let (h, w) = super::oriented_dimensions(h, w, orientation);
    limits.check(h, w)?;
    // Retain alpha from this one pixel decode; do not decode again to recover
    // it after RGB conversion. A BMP has alpha only with an alpha bitmask.
    let mut image = super::decode_8bit(ImageReader::with_format(Cursor::new(data), format))?;
    image.apply_orientation(orientation);
    let rgb = if image.color().has_alpha() {
        composite(image.into_rgba8().as_raw(), h, w)
    } else {
        image.into_rgb8().into_raw()
    };
    Ok((rgb, h, w))
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

#[cfg(test)]
mod tests {
    use super::*;
    use image::DynamicImage;

    #[test]
    fn grayscale_and_rgba_alpha_use_the_same_pillow_compositing() {
        for (pixels, expected) in [
            ([200, 255, 20, 128], [200, 26]), // Bright edges select dark background.
            ([20, 255, 200, 128], [20, 220]), // Dark edges select light background.
            ([200, 0, 20, 128], [255, 137]),  // No opaque edges selects white.
        ] {
            let gray = image::GrayAlphaImage::from_raw(2, 1, pixels.to_vec()).unwrap();
            let gray = DynamicImage::ImageLumaA8(gray);
            let rgba = DynamicImage::ImageRgba8(gray.to_rgba8());
            for image in [gray, rgba] {
                let mut encoded = Cursor::new(Vec::new());
                image.write_to(&mut encoded, ImageFormat::Png).unwrap();
                let (rgb, h, w) =
                    decode_rgb_smart(encoded.get_ref(), &DecodeLimits::default()).unwrap();
                assert_eq!((h, w), (1, 2));
                assert_eq!(
                    rgb,
                    expected
                        .into_iter()
                        .flat_map(|v| [v; 3])
                        .collect::<Vec<_>>()
                );
            }
        }
    }

    #[test]
    fn transparent_gif_preserves_first_frame_offsets_and_expands_the_canvas() {
        let mut encoded = Vec::new();
        {
            let mut encoder =
                gif::Encoder::new(&mut encoded, 2, 2, &[1, 2, 3, 200, 200, 200]).unwrap();
            encoder
                .write_frame(&gif::Frame {
                    left: 1,
                    top: 1,
                    width: 2,
                    height: 1,
                    transparent: Some(0),
                    buffer: vec![1, 0].into(),
                    ..gif::Frame::default()
                })
                .unwrap();
        }
        let (rgb, h, w) = decode_rgb_smart(&encoded, &DecodeLimits::default()).unwrap();
        assert_eq!((h, w), (2, 3));
        let mut expected = vec![32; 2 * 3 * 3];
        expected[12..15].fill(200);
        assert_eq!(rgb, expected);
        assert!(matches!(
            decode_rgb_smart(
                &encoded,
                &DecodeLimits {
                    max_width: Some(2),
                    ..DecodeLimits::default()
                }
            ),
            Err(MmError::LimitExceeded { .. })
        ));
    }
}
