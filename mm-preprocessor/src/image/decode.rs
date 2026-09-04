// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub fn decode_rgb(data: &[u8]) -> Result<(Vec<u8>, usize, usize), String> {
    use image::ColorType;

    let img = image::load_from_memory(data).map_err(|e| format!("image decode: {e}"))?;
    // >8-bit samples: PIL clips to 255 where `to_rgb8` would rescale. Refuse
    // rather than silently diverge from the Python (PIL) pipeline.
    if !matches!(
        img.color(),
        ColorType::L8 | ColorType::La8 | ColorType::Rgb8 | ColorType::Rgba8
    ) {
        return Err(format!("image decode: unsupported color {:?}", img.color()));
    }
    let rgb = img.to_rgb8();
    let (w, h) = rgb.dimensions();
    Ok((rgb.into_raw(), h as usize, w as usize))
}

#[cfg(test)]
mod tests {
    use super::decode_rgb;
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
        assert!(err.contains("unsupported color"), "{err}");
    }
}
