// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::{MmError, Result};
use image::metadata::Orientation;
use image::{ImageDecoder, ImageFormat};

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

/// Like [`decode_rgb`], but apply the EXIF orientation used by
/// `PIL.ImageOps.exif_transpose(Image.open(...)).convert("RGB")`.
///
/// Limits apply to transposed dimensions before pixel decoding. Pair with
/// [`dimensions_exif_transposed`] for token accounting. XMP-only orientation
/// is not supported.
pub fn decode_rgb_exif_transposed(
    data: &[u8],
    limits: &DecodeLimits,
) -> Result<(Vec<u8>, usize, usize)> {
    let (h, w) = dimensions(data)?;
    let orientation = exif_orientation(data)?;
    let (out_h, out_w) = oriented_dimensions(h, w, orientation);
    limits.check(out_h, out_w)?;
    let (rgb, _, _) = decode_rgb(data, &DecodeLimits::default())?;
    if orientation == Orientation::NoTransforms {
        return Ok((rgb, h, w));
    }
    let rgb = image::RgbImage::from_raw(w as u32, h as u32, rgb)
        .ok_or_else(|| MmError::invalid_input("image dimensions disagree with RGB buffer"))?;
    let mut img = image::DynamicImage::ImageRgb8(rgb);
    img.apply_orientation(orientation);
    Ok((img.into_rgb8().into_raw(), out_h, out_w))
}

/// Metadata-only dimensions after EXIF orientation, without decoding pixels.
///
/// Supports JPEG, PNG and WebP EXIF, including PNG tEXt/zTXt raw profiles.
/// PNG metadata may follow IDAT; compressed pixels are skipped. Legacy profile
/// text is capped at 1 MiB. Missing/malformed TIFF orientation tags are ignored;
/// container errors propagate. XMP-only orientation is not supported.
/// Encoded bytes remain unchanged.
pub fn dimensions_exif_transposed(data: &[u8]) -> Result<(usize, usize)> {
    let (h, w) = dimensions(data)?;
    Ok(oriented_dimensions(h, w, exif_orientation(data)?))
}

fn oriented_dimensions(h: usize, w: usize, orientation: Orientation) -> (usize, usize) {
    match orientation {
        Orientation::Rotate90
        | Orientation::Rotate270
        | Orientation::Rotate90FlipH
        | Orientation::Rotate270FlipH => (w, h),
        _ => (h, w),
    }
}

fn exif_orientation(data: &[u8]) -> Result<Orientation> {
    let exif = match format(data)? {
        // Read APP1 without linking a second JPEG pixel decoder.
        ImageFormat::Jpeg => {
            return Ok(jpeg_exif(data)
                .and_then(Orientation::from_exif_chunk)
                .unwrap_or(Orientation::NoTransforms));
        }
        ImageFormat::Png => png_exif(data)?,
        fmt @ ImageFormat::WebP => {
            let mut reader = image::ImageReader::new(std::io::Cursor::new(data));
            reader.set_format(fmt);
            reader
                .into_decoder()
                .map_err(invalid("image metadata probe failed"))?
                .exif_metadata()
                .map_err(invalid("image EXIF probe failed"))?
        }
        // GIF uses its first-frame canvas dimensions and has no EXIF.
        _ => None,
    };
    Ok(exif
        .as_deref()
        .and_then(|bytes| {
            // Pillow accepts both TIFF-only and JPEG-style EXIF WebP chunks.
            Orientation::from_exif_chunk(bytes.strip_prefix(b"Exif\0\0").unwrap_or(bytes))
        })
        .unwrap_or(Orientation::NoTransforms))
}

const PNG_EXIF_TEXT_LIMIT: usize = 1024 * 1024;
const PNG_EXIF_KEYWORD: &str = "Raw profile type exif";

fn png_exif(data: &[u8]) -> Result<Option<Vec<u8>>> {
    let mut decoder = png::Decoder::new(std::io::Cursor::new(data));
    decoder.set_ignore_iccp_chunk(true);
    let mut reader = decoder
        .read_info()
        .map_err(invalid("png metadata probe failed"))?;
    if let Some(exif) = &reader.info().exif_metadata {
        return Ok(Some(exif.to_vec()));
    }
    // Scan trailing metadata without inflating pixels. Header-only inputs
    // retain complete metadata read before EOF; other container errors fail.
    match reader.finish() {
        Ok(()) => {}
        Err(png::DecodingError::IoError(error))
            if error.kind() == std::io::ErrorKind::UnexpectedEof => {}
        Err(error) => return Err(invalid("png metadata probe failed")(error)),
    }
    png_raw_profile(reader.info())
}

fn png_raw_profile(info: &png::Info<'_>) -> Result<Option<Vec<u8>>> {
    let plain = info
        .uncompressed_latin1_text
        .iter()
        .rev()
        .find(|chunk| chunk.keyword == PNG_EXIF_KEYWORD);
    let compressed = info
        .compressed_latin1_text
        .iter()
        .rev()
        .find(|chunk| chunk.keyword == PNG_EXIF_KEYWORD);
    if plain.is_some() && compressed.is_some() {
        return Err(MmError::unsupported("png has competing EXIF text profiles"));
    }
    let text = if let Some(chunk) = plain {
        std::borrow::Cow::Borrowed(chunk.text.as_str())
    } else if let Some(chunk) = compressed {
        let mut chunk = chunk.clone();
        chunk
            .decompress_text_with_limit(PNG_EXIF_TEXT_LIMIT)
            .map_err(invalid("png EXIF text decode failed"))?;
        std::borrow::Cow::Owned(
            chunk
                .get_text()
                .map_err(invalid("png EXIF text decode failed"))?,
        )
    } else {
        return Ok(None);
    };
    if text.len() > PNG_EXIF_TEXT_LIMIT {
        return Err(MmError::limit_exceeded("png EXIF text exceeds 1 MiB"));
    }
    // Pillow ignores the three-line profile header and its declared byte count.
    let mut exif = Vec::new();
    let mut high = None;
    for byte in text.split('\n').skip(3).flat_map(str::bytes) {
        if byte.is_ascii_whitespace() {
            continue;
        }
        let nibble = match byte {
            b'0'..=b'9' => byte - b'0',
            b'a'..=b'f' => byte - b'a' + 10,
            b'A'..=b'F' => byte - b'A' + 10,
            _ => return Ok(None),
        };
        if let Some(previous) = high.take() {
            exif.push(previous * 16 + nibble);
        } else {
            high = Some(nibble);
        }
    }
    Ok(high.is_none().then_some(exif))
}

fn jpeg_exif(data: &[u8]) -> Option<&[u8]> {
    let mut remaining = data.strip_prefix(&[0xff, 0xd8])?;
    loop {
        // JPEG allows repeated 0xff fill bytes before a marker.
        remaining = remaining.strip_prefix(&[0xff])?;
        while remaining.first() == Some(&0xff) {
            remaining = &remaining[1..];
        }
        let (&marker, rest) = remaining.split_first()?;
        remaining = rest;
        match marker {
            0xda | 0xd9 => return None, // SOS/EOI: never inspect compressed pixels.
            0x01 | 0xd0..=0xd8 => continue, // Standalone markers have no length.
            _ => {}
        }
        let len = usize::from(u16::from_be_bytes(remaining.get(..2)?.try_into().ok()?));
        let payload = remaining.get(2..len)?;
        if marker == 0xe1
            && let Some(exif) = payload.strip_prefix(b"Exif\0\0")
        {
            return Some(exif);
        }
        remaining = remaining.get(len..)?;
    }
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
    use image::ImageEncoder;

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
        assert_eq!(
            decode_rgb_exif_transposed(JPEG, &DecodeLimits::default()).unwrap(),
            (rgb, h, w)
        );
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
            assert_eq!(dimensions_exif_transposed(gif).unwrap(), (h, w));
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

    fn exif(orientation: u16, big_endian: bool) -> Vec<u8> {
        let mut tiff = if big_endian {
            b"MM\0*\0\0\0\x08\0\x01\x01\x12\0\x03\0\0\0\x01".to_vec()
        } else {
            b"II*\0\x08\0\0\0\x01\0\x12\x01\x03\0\x01\0\0\0".to_vec()
        };
        tiff.extend(if big_endian {
            orientation.to_be_bytes()
        } else {
            orientation.to_le_bytes()
        });
        tiff.extend([0; 6]);
        tiff
    }

    fn with_exif(fmt: ImageFormat, exif: Vec<u8>) -> Vec<u8> {
        let pixels = (0..4)
            .flat_map(|y| (0..6).flat_map(move |x| [x * 40, y * 60, x + y * 10]))
            .collect::<Vec<u8>>();
        let mut bytes = Vec::new();
        match fmt {
            ImageFormat::Jpeg => {
                // Add metadata without re-encoding the existing JPEG fixture.
                bytes.extend(&JPEG[..2]);
                bytes.extend([0xff, 0xe1]);
                bytes.extend(((exif.len() + 8) as u16).to_be_bytes());
                bytes.extend(b"Exif\0\0");
                bytes.extend(exif);
                bytes.extend(&JPEG[2..]);
            }
            ImageFormat::Png => {
                let mut encoder = image::codecs::png::PngEncoder::new(&mut bytes);
                encoder.set_exif_metadata(exif).unwrap();
                encoder
                    .write_image(&pixels, 6, 4, image::ExtendedColorType::Rgb8)
                    .unwrap();
            }
            ImageFormat::WebP => {
                let mut encoder = image::codecs::webp::WebPEncoder::new_lossless(&mut bytes);
                encoder.set_exif_metadata(exif).unwrap();
                encoder
                    .write_image(&pixels, 6, 4, image::ExtendedColorType::Rgb8)
                    .unwrap();
            }
            _ => unreachable!(),
        }
        bytes
    }

    #[test]
    fn exif_dimensions_and_pixels_match_pillow() {
        for fmt in [ImageFormat::Jpeg, ImageFormat::Png, ImageFormat::WebP] {
            let (h, w, golden): (_, _, &[u8]) = if fmt == ImageFormat::Jpeg {
                (
                    9,
                    13,
                    include_bytes!("../../tests/fixtures/decode/pillow_exif_jpeg.rgb"),
                )
            } else {
                (
                    4,
                    6,
                    include_bytes!("../../tests/fixtures/decode/pillow_exif_lossless.rgb"),
                )
            };
            for big_endian in [false, true] {
                for tag in 0..=9 {
                    let bytes = with_exif(fmt, exif(tag, big_endian));
                    let expected_dimensions = if (5..=8).contains(&tag) {
                        (w, h)
                    } else {
                        (h, w)
                    };
                    let (rgb, out_h, out_w) =
                        decode_rgb_exif_transposed(&bytes, &DecodeLimits::default()).unwrap();
                    let index = if (1..=8).contains(&tag) { tag - 1 } else { 0 } as usize;
                    assert_eq!(
                        rgb,
                        golden[index * h * w * 3..(index + 1) * h * w * 3],
                        "{fmt:?}, tag={tag}, big_endian={big_endian}"
                    );
                    assert_eq!((out_h, out_w), expected_dimensions);
                    assert_eq!(
                        dimensions_exif_transposed(&bytes).unwrap(),
                        expected_dimensions
                    );
                }
            }
        }
    }

    #[test]
    fn prefixed_webp_and_malformed_exif() {
        let mut prefixed = b"Exif\0\0".to_vec();
        prefixed.extend(exif(6, false));
        assert_eq!(
            dimensions_exif_transposed(&with_exif(ImageFormat::WebP, prefixed)).unwrap(),
            (6, 4)
        );
        let mut bad_offset = exif(6, false);
        bad_offset[4..8].copy_from_slice(&u32::MAX.to_le_bytes());
        for fmt in [ImageFormat::Jpeg, ImageFormat::Png, ImageFormat::WebP] {
            for exif in [vec![], b"not TIFF".to_vec(), bad_offset.clone()] {
                let bytes = with_exif(fmt, exif);
                assert_eq!(
                    dimensions_exif_transposed(&bytes).unwrap(),
                    dimensions(&bytes).unwrap()
                );
            }
        }
    }

    #[test]
    fn oriented_limits_are_checked_before_pixels_are_decoded() {
        let mut bytes = with_exif(ImageFormat::Png, exif(6, false));
        let idat = bytes.windows(4).position(|chunk| chunk == b"IDAT").unwrap();
        bytes[idat + 4] = 0; // Broken compressed pixels; dimensions still work.
        assert!(decode_rgb(&bytes, &DecodeLimits::default()).is_err());
        assert_eq!(dimensions_exif_transposed(&bytes).unwrap(), (6, 4));
        let limits = DecodeLimits {
            max_height: Some(5),
            ..DecodeLimits::default()
        };
        assert!(matches!(
            decode_rgb_exif_transposed(&bytes, &limits),
            Err(MmError::LimitExceeded { .. })
        ));
        let valid = with_exif(ImageFormat::Png, exif(6, false));
        let limits = DecodeLimits {
            max_width: Some(4),
            max_height: Some(6),
            max_pixels: Some(24),
        };
        // The raw width is six; limits intentionally apply to the output width.
        assert!(decode_rgb_exif_transposed(&valid, &limits).is_ok());
    }

    #[test]
    fn jpeg_orientation_probe_stops_before_compressed_pixels() {
        let bytes = with_exif(ImageFormat::Jpeg, exif(6, false));
        let sos = bytes
            .windows(2)
            .position(|marker| marker == [0xff, 0xda])
            .unwrap();
        let len = usize::from(u16::from_be_bytes(
            bytes[sos + 2..sos + 4].try_into().unwrap(),
        ));
        let header = &bytes[..sos + 2 + len];
        assert_eq!(dimensions_exif_transposed(header).unwrap(), (13, 9));
        assert!(decode_rgb(header, &DecodeLimits::default()).is_err());
        // Opting into orientation must not change the raw JPEG APIs.
        assert_eq!(dimensions(&bytes).unwrap(), (9, 13));
        assert_eq!(
            decode_rgb(&bytes, &DecodeLimits::default()).unwrap().0,
            PILLOW_RGB
        );
    }

    fn png_with_profile(
        text: &str,
        compressed: bool,
        direct: bool,
        trailing: Option<bool>,
    ) -> Vec<u8> {
        let mut bytes = Vec::new();
        let mut encoder = png::Encoder::new(&mut bytes, 6, 4);
        encoder.set_color(png::ColorType::Rgb);
        if compressed {
            encoder
                .add_ztxt_chunk(PNG_EXIF_KEYWORD.into(), text.into())
                .unwrap();
        } else {
            encoder
                .add_text_chunk(PNG_EXIF_KEYWORD.into(), text.into())
                .unwrap();
        }
        let mut writer = encoder.write_header().unwrap();
        if direct {
            writer
                .write_chunk(png::chunk::eXIf, &exif(6, false))
                .unwrap();
        }
        writer.write_image_data(&[0; 72]).unwrap();
        if let Some(compressed) = trailing {
            let hex = exif(3, false)
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>();
            let profile = format!("\nexif\n0\n{hex}\n");
            if compressed {
                writer
                    .write_text_chunk(&png::text_metadata::ZTXtChunk::new(
                        PNG_EXIF_KEYWORD,
                        profile,
                    ))
                    .unwrap();
            } else {
                writer
                    .write_text_chunk(&png::text_metadata::TEXtChunk::new(
                        PNG_EXIF_KEYWORD,
                        profile,
                    ))
                    .unwrap();
            }
        }
        writer.finish().unwrap();
        bytes
    }

    #[test]
    fn legacy_png_profile_limits_and_malformed_text() {
        for compressed in [false, true] {
            let malformed = png_with_profile("\nexif\n1\nnot hex\n", compressed, false, None);
            assert_eq!(dimensions_exif_transposed(&malformed).unwrap(), (4, 6));
            let oversized = png_with_profile(
                &"0".repeat(PNG_EXIF_TEXT_LIMIT + 1),
                compressed,
                false,
                None,
            );
            assert!(dimensions_exif_transposed(&oversized).is_err());
        }
        let direct = png_with_profile("\nexif\n1\nnot hex\n", false, true, None);
        assert_eq!(dimensions_exif_transposed(&direct).unwrap(), (6, 4));
    }

    #[test]
    fn trailing_png_profiles_override_or_refuse_competing_encodings() {
        let hex = exif(6, false)
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        let profile = format!("\nexif\n0\n{hex}\n");
        let plain = png_with_profile(&profile, false, false, Some(false));
        assert_eq!(dimensions_exif_transposed(&plain).unwrap(), (4, 6));
        let competing = png_with_profile(&profile, false, false, Some(true));
        assert!(matches!(
            dimensions_exif_transposed(&competing),
            Err(MmError::Unsupported { .. })
        ));
    }

    #[test]
    fn legacy_png_profiles_do_not_need_compressed_pixels() {
        for bytes in [
            include_bytes!("../../tests/fixtures/decode/pillow_raw_exif_tEXt_before_idat.png")
                .as_slice(),
            include_bytes!("../../tests/fixtures/decode/pillow_raw_exif_zTXt_before_idat.png")
                .as_slice(),
        ] {
            let idat = bytes.windows(4).position(|chunk| chunk == b"IDAT").unwrap();
            let header = &bytes[..idat + 4];
            assert_eq!(dimensions_exif_transposed(header).unwrap(), (6, 4));
            assert!(decode_rgb(header, &DecodeLimits::default()).is_err());
        }
        let plain = encode(&image::DynamicImage::new_rgb8(6, 4), ImageFormat::Png);
        let idat = plain.windows(4).position(|chunk| chunk == b"IDAT").unwrap();
        assert_eq!(
            dimensions_exif_transposed(&plain[..idat + 4]).unwrap(),
            (4, 6)
        );
    }

    #[test]
    fn legacy_png_metadata_skips_corrupt_idat_before_checking_limits() {
        let hex = exif(6, false)
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        let profile = format!("\nexif\n0\n{hex}\n");
        for compressed in [false, true] {
            let mut bytes = Vec::new();
            let mut encoder = png::Encoder::new(&mut bytes, 6, 4);
            encoder.set_color(png::ColorType::Rgb);
            let mut writer = encoder.write_header().unwrap();
            // The writer supplies a correct CRC for deliberately invalid zlib.
            writer.write_chunk(png::chunk::IDAT, b"not zlib").unwrap();
            if compressed {
                writer
                    .write_text_chunk(&png::text_metadata::ZTXtChunk::new(
                        PNG_EXIF_KEYWORD,
                        &profile,
                    ))
                    .unwrap();
            } else {
                writer
                    .write_text_chunk(&png::text_metadata::TEXtChunk::new(
                        PNG_EXIF_KEYWORD,
                        &profile,
                    ))
                    .unwrap();
            }
            writer.finish().unwrap();
            assert!(decode_rgb(&bytes, &DecodeLimits::default()).is_err());
            assert_eq!(dimensions_exif_transposed(&bytes).unwrap(), (6, 4));
            let limits = DecodeLimits {
                max_height: Some(5),
                ..DecodeLimits::default()
            };
            assert!(matches!(
                decode_rgb_exif_transposed(&bytes, &limits),
                Err(MmError::LimitExceeded { .. })
            ));
        }
    }
}
