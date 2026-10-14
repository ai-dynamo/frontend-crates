// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_multimodal::Result;
use dynamo_multimodal::image::decode::{
    DecodeLimits, decode_rgb_exif_transposed, dimensions, dimensions_exif_transposed,
};
use dynamo_multimodal::processor::MediaMetadata;
use dynamo_multimodal::registry::processor_from_spec;
use dynamo_multimodal::token_layout::apply_layout;

#[test]
fn legacy_png_exif_keeps_frontend_spans_and_decoded_geometry_aligned() -> Result<()> {
    // Synthetic geometry exposes the row-separator difference: 6*(4+1)+2
    // oriented positions versus 4*(6+1)+2 if EXIF orientation were missed.
    let processor = processor_from_spec(
        r#"{
            "family": "deepseek_v41", "image_token_id": 100,
            "vision_patch_size": 1, "vision_downsample_ratio": 1,
            "vision_min_pixels": 0, "vision_max_n_token": 1024
        }"#,
    )?;
    let golden = &include_bytes!("fixtures/decode/pillow_exif_lossless.rgb")[5 * 72..6 * 72];
    for encoded in [
        include_bytes!("fixtures/decode/pillow_raw_exif_tEXt_before_idat.png").as_slice(),
        include_bytes!("fixtures/decode/pillow_raw_exif_zTXt_before_idat.png").as_slice(),
        include_bytes!("fixtures/decode/pillow_raw_exif_tEXt_after_idat.png").as_slice(),
        include_bytes!("fixtures/decode/pillow_raw_exif_zTXt_after_idat.png").as_slice(),
    ] {
        assert_eq!(dimensions(encoded)?, (4, 6));
        let (height, width) = dimensions_exif_transposed(encoded)?;
        assert_eq!((height, width), (6, 4));
        let (rgb, decoded_height, decoded_width) =
            decode_rgb_exif_transposed(encoded, &DecodeLimits::default())?;
        assert_eq!((decoded_height, decoded_width), (height, width));
        assert_eq!(rgb, golden);

        let metadata = [MediaMetadata::Image {
            width: width as u32,
            height: height as u32,
        }];
        let counts = [processor.num_media_tokens(&metadata[0])?];
        assert_eq!(counts, [32]);
        assert_eq!(
            processor.num_media_tokens(&MediaMetadata::Image {
                width: 6,
                height: 4,
            })?,
            30
        );
        let input_ids = [7, 100, 9];
        let layout = processor.layout_from_metadata(&input_ids, &metadata)?;
        let expanded = apply_layout(&input_ids, &layout, &counts, 34)?;
        assert_eq!(
            expanded.input_ids,
            [vec![7], vec![100; 32], vec![9]].concat()
        );
        assert_eq!(expanded.offsets, [(1, 32)]);
        assert_eq!(expanded.feature_ranges[0].len(), 1);
        assert_eq!(expanded.feature_ranges[0][0], 1..33);
    }
    Ok(())
}
