// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Decode as `PIL.Image.open` leaves an image, for pipelines that resize in
//! the native mode and convert afterwards.

use super::{DecodeLimits, GifImage, decode_8bit, decode_jpeg, dimensions, format, invalid};
use crate::{MmError, Result};
use image::ImageFormat;
use std::io::Cursor;

/// An image in the channels its PIL mode converts to, with no EXIF transpose.
pub(crate) struct NativeImage {
    /// HWC u8: straight-alpha RGBA when `alpha`, else RGB.
    pub(crate) pixels: Vec<u8>,
    /// PIL's mode is not RGB and has an alpha band or a `transparency` entry.
    pub(crate) alpha: bool,
    pub(crate) height: usize,
    pub(crate) width: usize,
    /// PIL resizes this source as RGB(A) or L(A), which `pixels` reproduce.
    /// False for palette, 1-bit, L with tRNS and CMYK sources.
    pub(crate) exact_resize: bool,
}

/// `Image.open(...)`, then `convert("RGBA")` when `alpha`, else `convert("RGB")`.
/// `Unsupported` for the forms known to differ from Pillow; an unknown one decodes.
pub(crate) fn decode_native(data: &[u8], limits: &DecodeLimits) -> Result<NativeImage> {
    let format = format(data)?;
    // The mode of a PNG, and the refusals a header decides, before any decode.
    let png = match format {
        // Pillow keeps a transparent index that a later extension clears.
        ImageFormat::Gif if gif_control_extensions(data) > 1 => {
            return Err(MmError::unsupported(
                "gif: more than one graphic control extension before the image is not mirrored",
            ));
        }
        ImageFormat::Jpeg | ImageFormat::Gif => None,
        // Pillow keeps a text chunk in `info` under its keyword, and reads
        // `interlace` back from there to choose the deinterlacing.
        ImageFormat::Png if png_interlace_text(data) => {
            return Err(MmError::unsupported(
                "png: a text chunk named interlace is not mirrored",
            ));
        }
        ImageFormat::Png => Some(png_mode(data)?),
        // Pillow reads palette BMP, RLE gaps and 16-bit fields in its own way.
        ImageFormat::Bmp if !bmp_truecolor(data) => {
            return Err(MmError::unsupported(
                "bmp: only uncompressed 24-bit and 32-bit files are mirrored",
            ));
        }
        // `image-webp` blends a lossless frame that asks for it; libwebp does
        // not blend a first frame.
        ImageFormat::WebP if webp_animated(data) => {
            return Err(MmError::unsupported(
                "webp: an animated image is not mirrored",
            ));
        }
        // libwebp adds the loop-filter deltas to a segment's filter level
        // before it clamps; `image-webp` clamps first, as RFC 6386 does.
        ImageFormat::WebP if webp_filter_level_unclamped(data) => {
            return Err(MmError::unsupported(
                "webp: a segment filter level outside 0..=63 with loop-filter deltas is not mirrored",
            ));
        }
        ImageFormat::Bmp | ImageFormat::WebP => None,
        // Other formats decode when a consumer's build enables them in `image`.
        other => {
            return Err(MmError::unsupported(format!(
                "{other:?} images are not mirrored"
            )));
        }
    };
    let (height, width) = dimensions(data)?;
    limits.check(height, width)?;
    let (pixels, alpha, exact_resize) = match format {
        ImageFormat::Jpeg => {
            let header = turbojpeg::read_header(data).map_err(invalid("jpeg probe failed"))?;
            let cmyk = matches!(
                header.colorspace,
                turbojpeg::Colorspace::CMYK | turbojpeg::Colorspace::YCCK
            );
            (decode_jpeg(data)?, false, !cmyk)
        }
        // Mode P, or L for a gray-ramp palette; neither is told apart here.
        ImageFormat::Gif => {
            let gif = GifImage::new(data)?;
            // Pillow paints a frame whose own table is an identity gray ramp
            // with the global table.
            let ramp = |table: &[u8]| {
                (table.chunks_exact(3).zip(0..=u8::MAX)).all(|(rgb, i)| rgb == [i; 3])
            };
            if gif.frame.palette.as_deref().is_some_and(ramp)
                && gif
                    .decoder
                    .global_palette()
                    .is_some_and(|table| !ramp(table))
            {
                return Err(MmError::unsupported(
                    "gif: a gray-ramp local color table beside a global one is not mirrored",
                ));
            }
            if gif.frame.transparent.is_some() {
                (gif.decode_rgba()?, true, false)
            } else {
                (gif.decode_rgb()?, false, false)
            }
        }
        fmt => {
            let image = decode_8bit(image::ImageReader::with_format(Cursor::new(data), fmt))?;
            let (alpha, exact_resize) = png.unwrap_or_else(|| (image.color().has_alpha(), true));
            let pixels = if alpha {
                image.into_rgba8().into_raw()
            } else {
                image.into_rgb8().into_raw()
            };
            (pixels, alpha, exact_resize)
        }
    };
    Ok(NativeImage {
        pixels,
        alpha,
        height,
        width,
        exact_resize,
    })
}

/// `(alpha, exact_resize)` from the PNG header, as Pillow's mode table reads it.
fn png_mode(data: &[u8]) -> Result<(bool, bool)> {
    use png::{BitDepth, ColorType};
    let mut decoder = png::Decoder::new(Cursor::new(data));
    decoder.set_ignore_iccp_chunk(true);
    let reader = decoder.read_info().map_err(invalid("png probe failed"))?;
    let info = reader.info();
    let trns = info.trns.is_some();
    Ok(match (info.color_type, info.bit_depth) {
        (ColorType::Rgba | ColorType::GrayscaleAlpha, _) => (true, true),
        // Mode RGB: the tRNS color stays in `info` and RGB sources ignore it.
        (ColorType::Rgb, _) => (false, true),
        // Pillow reads `FF* 00 FF* 0A` as one transparent entry: its pattern's
        // `$` also matches before a final newline byte.
        (ColorType::Indexed, _)
            if info.trns.as_deref().is_some_and(|alpha| {
                alpha.last() == Some(&b'\n')
                    && alpha.iter().filter(|&&a| a == 0).count() == 1
                    && alpha[..alpha.len() - 1].iter().all(|&a| a == 0 || a == 255)
            }) =>
        {
            return Err(MmError::unsupported(
                "png: this palette tRNS is not mirrored",
            ));
        }
        (ColorType::Indexed, _) => (trns, false),
        // Pillow compares the unscaled tRNS sample with the scaled pixels at 2
        // and 4 bits, and reads a 1-bit sample above 1 as white.
        (ColorType::Grayscale, depth)
            if info.trns.as_deref().is_some_and(|sample| match depth {
                BitDepth::Eight => false,
                BitDepth::One => sample != [0] && sample != [1],
                _ => true,
            }) =>
        {
            return Err(MmError::unsupported(
                "png: this grayscale tRNS sample is not mirrored",
            ));
        }
        (ColorType::Grayscale, BitDepth::Eight) => (trns, !trns),
        (ColorType::Grayscale, BitDepth::One) => (trns, false),
        (ColorType::Grayscale, _) => (false, true),
    })
}

/// A tEXt, zTXt or iTXt chunk before IDAT with the keyword `interlace`. Pillow
/// reads a chunk with no NUL as all keyword, so this does not parse the text.
fn png_interlace_text(data: &[u8]) -> bool {
    let mut rest = data.get(8..).unwrap_or_default();
    while let Some((head, body)) = rest.split_at_checked(8) {
        let size = u32::from_be_bytes([head[0], head[1], head[2], head[3]]) as usize;
        let Some(chunk) = body.get(..size) else {
            return false;
        };
        match &head[4..] {
            b"IDAT" => return false,
            b"tEXt" | b"zTXt" | b"iTXt"
                if chunk.split(|&byte| byte == 0).next() == Some(b"interlace".as_slice()) =>
            {
                return true;
            }
            _ => {}
        }
        rest = body.get(size.saturating_add(4)..).unwrap_or_default();
    }
    false
}

/// 24 or 32 bits per pixel, BI_RGB or BI_BITFIELDS, with a pixel offset. The
/// 12-byte core header has 16-bit fields and no compression field.
fn bmp_truecolor(data: &[u8]) -> bool {
    let u16_at = |at| {
        data.get(at..at + 2)
            .map(|b| u16::from_le_bytes([b[0], b[1]]))
    };
    let u32_at = |at| Some(u32::from(u16_at(at)?) | u32::from(u16_at(at + 2)?) << 16);
    let core = u32_at(14) == Some(12);
    let bits = u16_at(if core { 24 } else { 28 });
    let compression = if core { Some(0) } else { u32_at(30) };
    matches!(bits, Some(24 | 32)) && matches!(compression, Some(0 | 3)) && u32_at(10) != Some(0)
}

/// Graphic Control Extensions between the screen descriptor, with its color
/// table, and the first block that is not an extension.
fn gif_control_extensions(data: &[u8]) -> usize {
    let table = |flags: &u8| usize::from(flags >> 7) * (3 << ((flags & 7) + 1));
    let mut at = 13 + data.get(10).map_or(0, table);
    let mut controls = 0;
    while let Some([0x21, label]) = data.get(at..at + 2) {
        controls += usize::from(*label == 0xf9);
        at += 2;
        while let Some(&len) = data.get(at) {
            at += 1 + usize::from(len);
            if len == 0 {
                break;
            }
        }
    }
    controls
}

/// The VP8X flags byte after `RIFF....WEBPVP8X....`: animation is 0x02.
fn webp_animated(data: &[u8]) -> bool {
    data.get(12..16) == Some(b"VP8X") && data.get(20).is_some_and(|flags| flags & 0x02 != 0)
}

/// A lossy key frame where some segment's filter level is outside 0..=63 and
/// a loop-filter delta then moves it (RFC 6386 9.3 and 9.6).
fn webp_filter_level_unclamped(data: &[u8]) -> bool {
    // The `VP8 ` chunk follows the RIFF header, or other chunks of a VP8X file.
    let mut rest = data.get(12..).unwrap_or_default();
    let frame = loop {
        let Some((head, body)) = rest.split_at_checked(8) else {
            return false;
        };
        if &head[..4] == b"VP8 " {
            break body;
        }
        let size = u32::from_le_bytes([head[4], head[5], head[6], head[7]]) as usize;
        rest = body
            .get(size.saturating_add(size & 1)..)
            .unwrap_or_default();
    };
    // The first partition follows the frame tag, the start code and the size.
    let mut bits = Vp8Bits::new(frame.get(10..).unwrap_or_default());
    bits.literal(2);
    let (mut segment, mut absolute) = ([0; 4], false);
    let segments = bits.flag();
    if segments {
        let update_map = bits.flag();
        if bits.flag() {
            absolute = bits.flag();
            for _ in 0..4 {
                bits.signed(7);
            }
            segment = std::array::from_fn(|_| bits.signed(6));
        }
        for _ in 0..3 * u32::from(update_map) {
            if bits.flag() {
                bits.literal(8);
            }
        }
    }
    bits.flag();
    let level = bits.literal(6);
    bits.literal(3);
    if !(segments && level != 0 && bits.flag() && bits.flag()) {
        return false;
    }
    // A key frame uses the first reference delta and the first mode delta.
    let reference = bits.signed(6);
    for _ in 0..3 {
        bits.signed(6);
    }
    let mode = bits.signed(6);
    segment.iter().any(|&segment| {
        let base = if absolute { segment } else { level + segment };
        [reference, reference + mode]
            .iter()
            .any(|delta| (base + delta).clamp(0, 63) != (base.clamp(0, 63) + delta).clamp(0, 63))
    })
}

/// The VP8 boolean decoder (RFC 6386 7.3), for bits of probability 128.
struct Vp8Bits<'a> {
    bytes: std::slice::Iter<'a, u8>,
    value: u32,
    range: u32,
    shifts: u32,
}

impl<'a> Vp8Bits<'a> {
    fn new(data: &'a [u8]) -> Self {
        let mut bits = Self {
            bytes: data.iter(),
            value: 0,
            range: 255,
            shifts: 0,
        };
        bits.value = bits.byte() << 8 | bits.byte();
        bits
    }

    fn byte(&mut self) -> u32 {
        self.bytes.next().map_or(0, |&byte| u32::from(byte))
    }

    fn flag(&mut self) -> bool {
        let split = 1 + (((self.range - 1) * 128) >> 8);
        let set = self.value >= split << 8;
        if set {
            self.range -= split;
            self.value -= split << 8;
        } else {
            self.range = split;
        }
        while self.range < 128 {
            self.value <<= 1;
            self.range <<= 1;
            self.shifts += 1;
            if self.shifts == 8 {
                self.shifts = 0;
                self.value |= self.byte();
            }
        }
        set
    }

    fn literal(&mut self, bits: u32) -> i32 {
        (0..bits).fold(0, |value, _| value << 1 | i32::from(self.flag()))
    }

    /// A flag, then a magnitude and a sign. An absent value is 0.
    fn signed(&mut self, bits: u32) -> i32 {
        if !self.flag() {
            return 0;
        }
        let magnitude = self.literal(bits);
        if self.flag() { -magnitude } else { magnitude }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const JPEG: &[u8] = include_bytes!("../../../tests/fixtures/decode/pillow_noise_13x9.jpg");

    #[test]
    fn limits_are_checked_before_the_pixel_decode() {
        let exact = DecodeLimits {
            max_width: Some(13),
            max_height: Some(9),
            max_pixels: Some(9 * 13),
        };
        let image = decode_native(JPEG, &exact).unwrap();
        assert_eq!(
            (image.height, image.width, image.pixels.len()),
            (9, 13, 351)
        );
        assert!(!image.alpha && image.exact_resize);

        // The file cut after its scan header: a size, and no pixels to decode.
        let sos = JPEG.windows(2).position(|m| m == [0xff, 0xda]).unwrap();
        let scan_header = usize::from(u16::from_be_bytes([JPEG[sos + 2], JPEG[sos + 3]]));
        let header = &JPEG[..sos + 2 + scan_header];
        assert!(matches!(
            decode_native(header, &exact),
            Err(MmError::InvalidInput { .. })
        ));
        for limits in [
            DecodeLimits {
                max_width: Some(12),
                ..exact
            },
            DecodeLimits {
                max_height: Some(8),
                ..exact
            },
            DecodeLimits {
                max_pixels: Some(9 * 13 - 1),
                ..exact
            },
        ] {
            for data in [JPEG, header] {
                assert!(matches!(
                    decode_native(data, &limits),
                    Err(MmError::LimitExceeded { .. })
                ));
            }
        }
    }

    /// A 1x1 24-bit BMP, then the same file as a palette, RLE or 16-bit
    /// header, and with no pixel offset; then the 12-byte core header.
    #[test]
    fn bmp_admits_plain_truecolor_only() {
        let mut bmp = Vec::from(*b"BM");
        for field in [58u32, 0, 54, 40, 1, 1] {
            bmp.extend(field.to_le_bytes());
        }
        bmp.extend([1, 0, 24, 0]);
        bmp.extend([0; 24]);
        bmp.extend([30, 20, 10, 0]);
        let image = decode_native(&bmp, &DecodeLimits::default()).unwrap();
        assert_eq!((image.pixels, image.alpha), (vec![10, 20, 30], false));
        assert!(image.exact_resize);

        for (at, value) in [(28, 8), (28, 16), (30, 1), (10, 0)] {
            let mut refused = bmp.clone();
            refused[at] = value;
            assert!(
                matches!(
                    decode_native(&refused, &DecodeLimits::default()),
                    Err(MmError::Unsupported { .. })
                ),
                "byte {at} = {value}"
            );
        }

        let mut core = Vec::from(*b"BM");
        for field in [30u32, 0, 26, 12] {
            core.extend(field.to_le_bytes());
        }
        core.extend([1, 0, 1, 0, 1, 0, 24, 0, 30, 20, 10, 0]);
        let image = decode_native(&core, &DecodeLimits::default()).unwrap();
        assert_eq!((image.pixels, image.alpha), (vec![10, 20, 30], false));
        core[24] = 8;
        assert!(matches!(
            decode_native(&core, &DecodeLimits::default()),
            Err(MmError::Unsupported { .. })
        ));
    }

    /// The keyword decides, with or without its NUL, and only before IDAT.
    #[test]
    fn png_interlace_text_is_a_keyword_before_idat() {
        let png = |chunks: &[(&[u8; 4], &[u8])]| {
            let mut data = vec![0; 8];
            for (tag, body) in chunks {
                data.extend((body.len() as u32).to_be_bytes());
                data.extend(*tag);
                data.extend(*body);
                data.extend([0; 4]);
            }
            data
        };
        let header: (&[u8; 4], &[u8]) = (b"IHDR", &[0; 13]);
        for (chunk, expected) in [
            ((b"tEXt", &b"interlace\x001"[..]), true),
            ((b"tEXt", &b"interlace"[..]), true),
            ((b"zTXt", &b"interlace\0\0"[..]), true),
            ((b"iTXt", &b"interlace\0\0\0\0\0"[..]), true),
            ((b"tEXt", &b"Interlace\x001"[..]), false),
            ((b"tEXt", &b"interlaced\x001"[..]), false),
            ((b"tIME", &b"interlace\x001"[..]), false),
        ] {
            let before = png(&[header, chunk, (b"IDAT", &[])]);
            assert_eq!(png_interlace_text(&before), expected, "{chunk:?}");
            let after = png(&[header, (b"IDAT", &[]), chunk]);
            assert!(!png_interlace_text(&after), "{chunk:?} after IDAT");
        }
    }

    /// A format outside the five is refused by name, before any decoder runs.
    #[test]
    fn other_formats_are_unsupported() {
        let tiff = b"II*\0\x08\0\0\0\0\0\0\0\0\0\0\0";
        assert_eq!(image::guess_format(tiff).unwrap(), ImageFormat::Tiff);
        assert!(matches!(
            decode_native(tiff, &DecodeLimits::default()),
            Err(MmError::Unsupported { .. })
        ));
    }
}
