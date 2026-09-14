// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::{MmError, Result};

/// Decode encoded image bytes (jpeg/png/webp/gif/bmp — the formats the Python
/// PIL path commonly accepts) to `(HWC u8 RGB, height, width)`.
///
/// Samples deeper than 8 bits are rejected: PIL clips to 255 where a u8
/// conversion would rescale, so refusing is the only bit-exact answer.
pub fn decode_rgb(data: &[u8]) -> Result<(Vec<u8>, usize, usize)> {
    use image::ColorType;

    let img = image::load_from_memory(data)
        .map_err(|error| MmError::invalid_input_with_source("image decode failed", error))?;
    if !matches!(
        img.color(),
        ColorType::L8 | ColorType::La8 | ColorType::Rgb8 | ColorType::Rgba8
    ) {
        return Err(MmError::invalid_input(format!(
            "image decode: unsupported color {:?}",
            img.color()
        )));
    }
    let rgb = img.to_rgb8();
    let (w, h) = rgb.dimensions();
    Ok((rgb.into_raw(), h as usize, w as usize))
}

/// `(height, width)` from the encoded header alone — no pixel decode (PIL's
/// lazy `Image.open(...).size`). Supplies
/// [`MediaMetadata::Image`](crate::processor::MediaMetadata::Image)
/// for pixel-free token accounting.
pub fn dimensions(data: &[u8]) -> Result<(usize, usize)> {
    let reader = image::ImageReader::new(std::io::Cursor::new(data))
        .with_guessed_format()
        .map_err(|error| MmError::invalid_input_with_source("image probe failed", error))?;
    let (w, h) = reader
        .into_dimensions()
        .map_err(|error| MmError::invalid_input_with_source("image probe failed", error))?;
    Ok((h as usize, w as usize))
}

#[cfg(test)]
mod tests {
    use super::{decode_rgb, dimensions};
    use image::ImageFormat;

    fn encode(img: &image::DynamicImage, fmt: ImageFormat) -> Vec<u8> {
        let mut buf = std::io::Cursor::new(Vec::new());
        img.write_to(&mut buf, fmt).unwrap();
        buf.into_inner()
    }

    /// Formats the Python (PIL) path accepts must decode, not reject.
    #[test]
    fn decodes_webp_gif_bmp() {
        let img = image::DynamicImage::ImageRgb8(image::RgbImage::from_fn(6, 4, |x, y| {
            image::Rgb([x as u8 * 40, y as u8 * 60, 7])
        }));
        for fmt in [ImageFormat::WebP, ImageFormat::Gif, ImageFormat::Bmp] {
            let (rgb, h, w) = decode_rgb(&encode(&img, fmt)).unwrap();
            assert_eq!((h, w), (4, 6), "{fmt:?}");
            assert_eq!(rgb.len(), 4 * 6 * 3, "{fmt:?}");
            assert_eq!(dimensions(&encode(&img, fmt)).unwrap(), (4, 6), "{fmt:?}");
        }
    }

    /// Samples deeper than 8 bits stay rejected (PIL clips; we refuse).
    #[test]
    fn deep_png_rejected() {
        let img = image::DynamicImage::ImageRgb16(image::ImageBuffer::from_pixel(
            2,
            2,
            image::Rgb([65535u16, 0, 0]),
        ));
        let err = decode_rgb(&encode(&img, ImageFormat::Png)).err().unwrap();
        assert!(err.to_string().contains("unsupported color"), "{err}");
    }
}
