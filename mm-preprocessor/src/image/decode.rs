// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::{MmError, Result};
use image::ImageFormat;

/// Header-checked limits enforced before pixel decoding. `None` means unbounded.
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
        ImageFormat::Jpeg => decode_jpeg(data)?,
        ImageFormat::Gif => GifImage::new(data)?.decode_rgb()?,
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

fn decode_jpeg(data: &[u8]) -> Result<Vec<u8>> {
    use turbojpeg::{Colorspace, PixelFormat};
    let header = turbojpeg::read_header(data).map_err(invalid("jpeg probe failed"))?;
    let cmyk = matches!(header.colorspace, Colorspace::CMYK | Colorspace::YCCK);
    let pixels = turbojpeg::decompress(
        data,
        if cmyk {
            PixelFormat::CMYK
        } else {
            PixelFormat::RGB
        },
    )
    .map_err(invalid("jpeg decode failed"))?
    .pixels;
    if !cmyk {
        return Ok(pixels);
    }

    // Pillow reads JPEG CMYK as inverted ("CMYK;I"), then applies its rounded
    // CMYK-to-RGB conversion. With inverted samples this is round(C * K / 255).
    let mut rgb = Vec::with_capacity(pixels.len() / 4 * 3);
    for pixel in pixels.chunks_exact(4) {
        for &component in &pixel[..3] {
            rgb.push(((u32::from(component) * u32::from(pixel[3]) + 127) / 255) as u8);
        }
    }
    Ok(rgb)
}

struct GifImage<'a> {
    decoder: gif::Decoder<&'a [u8]>,
    frame: gif::Frame<'static>,
}

impl<'a> GifImage<'a> {
    fn new(data: &'a [u8]) -> Result<Self> {
        let mut options = gif::DecodeOptions::new();
        options.set_color_output(gif::ColorOutput::Indexed);
        let mut decoder = options
            .read_info(data)
            .map_err(invalid("gif probe failed"))?;
        let frame = decoder
            .next_frame_info()
            .map_err(invalid("gif probe failed"))?
            .ok_or_else(|| MmError::invalid_input("gif has no image frame"))?
            .clone();
        if frame.width == 0 || frame.height == 0 {
            return Err(MmError::invalid_input("gif has an empty image frame"));
        }
        Ok(Self { decoder, frame })
    }

    fn dimensions(&self) -> (usize, usize) {
        // Pillow expands the canvas when the first frame extends past it.
        (
            usize::from(self.decoder.height())
                .max(usize::from(self.frame.top) + usize::from(self.frame.height)),
            usize::from(self.decoder.width())
                .max(usize::from(self.frame.left) + usize::from(self.frame.width)),
        )
    }

    fn decode_rgb(mut self) -> Result<Vec<u8>> {
        let (h, w) = self.dimensions();
        let mut indices = vec![0; usize::from(self.frame.width) * usize::from(self.frame.height)];
        self.decoder
            .read_into_buffer(&mut indices)
            .map_err(invalid("gif decode failed"))?;
        let palette = self
            .decoder
            .palette()
            .map_err(invalid("gif palette missing"))?;
        let color = |index: u8| {
            let start = usize::from(index) * 3;
            palette
                .get(start..start + 3)
                .ok_or_else(|| MmError::invalid_input("gif palette index out of bounds"))
        };
        // Pillow fills the first canvas with the transparent index or zero,
        // ignoring the declared background. RGB conversion preserves its color.
        let background = color(self.frame.transparent.unwrap_or(0))?;
        let mut rgb = background.repeat(h * w);
        for (y, row) in indices
            .chunks_exact(usize::from(self.frame.width))
            .enumerate()
        {
            let start = ((y + usize::from(self.frame.top)) * w + usize::from(self.frame.left)) * 3;
            for (pixel, &index) in rgb[start..start + row.len() * 3]
                .chunks_exact_mut(3)
                .zip(row)
            {
                pixel.copy_from_slice(color(index)?);
            }
        }
        Ok(rgb)
    }
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
        ImageFormat::Gif => {
            let (h, w) = GifImage::new(data)?.dimensions();
            (w, h)
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

    #[test]
    fn cmyk_and_ycck_jpeg_match_pillow() {
        for (jpeg, expected) in [
            (
                include_bytes!("../../tests/fixtures/decode/pillow_cmyk_13x9.jpg").as_slice(),
                include_bytes!("../../tests/fixtures/decode/pillow_cmyk_13x9.rgb").as_slice(),
            ),
            (
                include_bytes!("../../tests/fixtures/decode/turbojpeg_ycck_13x9.jpg").as_slice(),
                include_bytes!("../../tests/fixtures/decode/turbojpeg_ycck_13x9.rgb").as_slice(),
            ),
        ] {
            let (rgb, h, w) = decode_rgb(jpeg, &DecodeLimits::default()).unwrap();
            assert_eq!((h, w), (9, 13));
            assert_eq!(rgb, expected);
        }
    }

    #[test]
    fn gif_first_frame_canvas_matches_pillow() {
        for (gif, expected) in [
            (
                include_bytes!("../../tests/fixtures/decode/pillow_offset_8x8.gif").as_slice(),
                include_bytes!("../../tests/fixtures/decode/pillow_offset_8x8.rgb").as_slice(),
            ),
            (
                include_bytes!("../../tests/fixtures/decode/pillow_transparent_offset_8x8.gif")
                    .as_slice(),
                include_bytes!("../../tests/fixtures/decode/pillow_transparent_offset_8x8.rgb")
                    .as_slice(),
            ),
            (
                include_bytes!("../../tests/fixtures/decode/pillow_local_offset_8x8.gif")
                    .as_slice(),
                include_bytes!("../../tests/fixtures/decode/pillow_local_offset_8x8.rgb")
                    .as_slice(),
            ),
        ] {
            let (rgb, h, w) = decode_rgb(gif, &DecodeLimits::default()).unwrap();
            assert_eq!((h, w), (8, 8));
            assert_eq!(dimensions(gif).unwrap(), (h, w));
            assert_eq!(rgb, expected);
        }
    }

    #[test]
    fn gif_limits_include_the_first_frame_extent_before_decoding() {
        let gif = include_bytes!("../../tests/fixtures/decode/pillow_expanded_6x6.gif");
        let expected = include_bytes!("../../tests/fixtures/decode/pillow_expanded_6x6.rgb");
        assert_eq!(dimensions(gif).unwrap(), (6, 6));
        let (rgb, h, w) = decode_rgb(gif, &DecodeLimits::default()).unwrap();
        assert_eq!((h, w), (6, 6));
        assert_eq!(rgb, expected);

        // Keep the logical screen, global palette, first frame descriptor and
        // LZW code size and first block length, but omit all compressed pixels.
        let header = &gif[..13 + 256 * 3 + 10 + 2];
        assert_eq!(dimensions(header).unwrap(), (6, 6));
        for limits in [
            DecodeLimits {
                max_width: Some(5),
                ..DecodeLimits::default()
            },
            DecodeLimits {
                max_height: Some(5),
                ..DecodeLimits::default()
            },
            DecodeLimits {
                max_pixels: Some(35),
                ..DecodeLimits::default()
            },
        ] {
            assert!(matches!(
                decode_rgb(header, &limits),
                Err(MmError::LimitExceeded { .. })
            ));
        }
    }

    /// Formats the Python (PIL) path accepts must decode, not reject.
    #[test]
    fn decodes_webp_bmp() {
        let img = image::DynamicImage::ImageRgb8(image::RgbImage::from_fn(6, 4, |x, y| {
            image::Rgb([x as u8 * 40, y as u8 * 60, 7])
        }));
        for fmt in [ImageFormat::WebP, ImageFormat::Bmp] {
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
