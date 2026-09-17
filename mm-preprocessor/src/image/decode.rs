// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::{MmError, Result};
use image::ImageFormat;

/// Caps checked against the encoded header before any pixel is decoded, so an
/// oversized input costs a header parse rather than an allocation. `None`
/// leaves that axis unbounded.
#[derive(Clone, Copy, Debug, Default)]
pub struct DecodeLimits {
    pub max_width: Option<usize>,
    pub max_height: Option<usize>,
    pub max_pixels: Option<usize>,
}

impl DecodeLimits {
    fn check(&self, h: usize, w: usize) -> Result<()> {
        let over = |cap: Option<usize>, value: usize| cap.is_some_and(|cap| value > cap);
        if over(self.max_width, w)
            || over(self.max_height, h)
            || over(self.max_pixels, h.saturating_mul(w))
        {
            return Err(MmError::limit_exceeded(format!(
                "image {w}x{h} exceeds decode limits {self:?}"
            )));
        }
        Ok(())
    }
}

/// Decode encoded image bytes (jpeg/png/webp/gif/bmp — the formats the Python
/// PIL path commonly accepts) to `(HWC u8 RGB, height, width)`, refusing
/// anything the header says is over `limits`.
///
/// JPEG goes through libjpeg-turbo, Pillow's own backend, with its default
/// accurate IDCT and fancy upsampling. Samples deeper than 8 bits are
/// rejected: PIL clips to 255 where a u8 conversion would rescale, so
/// refusing is the only bit-exact answer.
pub fn decode_rgb(data: &[u8], limits: &DecodeLimits) -> Result<(Vec<u8>, usize, usize)> {
    let (h, w) = dimensions(data)?;
    limits.check(h, w)?;
    let rgb = match format(data)? {
        ImageFormat::Jpeg => {
            turbojpeg::decompress(data, turbojpeg::PixelFormat::RGB)
                .map_err(invalid("jpeg decode failed"))?
                .pixels
        }
        fmt => {
            use image::ColorType;
            let img = image::load_from_memory_with_format(data, fmt)
                .map_err(invalid("image decode failed"))?;
            if !matches!(
                img.color(),
                ColorType::L8 | ColorType::La8 | ColorType::Rgb8 | ColorType::Rgba8
            ) {
                return Err(MmError::invalid_input(format!(
                    "image decode: unsupported color {:?}",
                    img.color()
                )));
            }
            img.into_rgb8().into_raw()
        }
    };
    Ok((rgb, h, w))
}

/// `(height, width)` from the encoded header alone — no pixel decode (PIL's
/// lazy `Image.open(...).size`). Supplies
/// [`MediaMetadata::Image`](crate::processor::MediaMetadata::Image)
/// for pixel-free token accounting.
pub fn dimensions(data: &[u8]) -> Result<(usize, usize)> {
    let (w, h) = match format(data)? {
        ImageFormat::Jpeg => {
            let header = turbojpeg::read_header(data).map_err(invalid("jpeg probe failed"))?;
            (header.width, header.height)
        }
        fmt => {
            let mut reader = image::ImageReader::new(std::io::Cursor::new(data));
            reader.set_format(fmt);
            let (w, h) = reader
                .into_dimensions()
                .map_err(invalid("image probe failed"))?;
            (w as usize, h as usize)
        }
    };
    Ok((h, w))
}

fn format(data: &[u8]) -> Result<ImageFormat> {
    image::guess_format(data).map_err(invalid("unrecognized image format"))
}

fn invalid<E>(context: &'static str) -> impl Fn(E) -> MmError
where
    E: std::error::Error + Send + Sync + 'static,
{
    move |error| MmError::invalid_input_with_source(context, error)
}

#[cfg(test)]
mod tests {
    use super::*;

    const JPEG: &[u8] = include_bytes!("../../tests/fixtures/decode/pillow_noise_13x9.jpg");
    /// `PIL.Image.open(JPEG).convert("RGB")`, Pillow 12 on libjpeg-turbo.
    const PILLOW_RGB: &[u8] = include_bytes!("../../tests/fixtures/decode/pillow_noise_13x9.rgb");

    fn encode(img: &image::DynamicImage, fmt: ImageFormat) -> Vec<u8> {
        let mut buf = std::io::Cursor::new(Vec::new());
        img.write_to(&mut buf, fmt).unwrap();
        buf.into_inner()
    }

    #[test]
    fn jpeg_matches_pillow_byte_for_byte() {
        let (rgb, h, w) = decode_rgb(JPEG, &DecodeLimits::default()).unwrap();
        assert_eq!((h, w), (9, 13));
        assert_eq!(rgb, PILLOW_RGB);
        assert_eq!(dimensions(JPEG).unwrap(), (9, 13));
    }

    /// Formats the Python (PIL) path accepts must decode, not reject.
    #[test]
    fn decodes_webp_gif_bmp() {
        let img = image::DynamicImage::ImageRgb8(image::RgbImage::from_fn(6, 4, |x, y| {
            image::Rgb([x as u8 * 40, y as u8 * 60, 7])
        }));
        for fmt in [ImageFormat::WebP, ImageFormat::Gif, ImageFormat::Bmp] {
            let (rgb, h, w) = decode_rgb(&encode(&img, fmt), &DecodeLimits::default()).unwrap();
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
        let err = decode_rgb(&encode(&img, ImageFormat::Png), &DecodeLimits::default())
            .err()
            .unwrap();
        assert!(err.to_string().contains("unsupported color"), "{err}");
    }

    #[test]
    fn limits_are_enforced_from_the_header() {
        let too_narrow = DecodeLimits {
            max_width: Some(12),
            ..DecodeLimits::default()
        };
        let too_many = DecodeLimits {
            max_pixels: Some(9 * 13 - 1),
            ..DecodeLimits::default()
        };
        for limits in [too_narrow, too_many] {
            assert!(matches!(
                decode_rgb(JPEG, &limits),
                Err(MmError::LimitExceeded { .. })
            ));
        }
        let exact = DecodeLimits {
            max_width: Some(13),
            max_height: Some(9),
            max_pixels: Some(9 * 13),
        };
        assert!(decode_rgb(JPEG, &exact).is_ok());
    }
}
