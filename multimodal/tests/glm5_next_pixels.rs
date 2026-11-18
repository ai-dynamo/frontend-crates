// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! GLM pixel-path checks that the replay in `mm_parity.rs` does not make: spec
//! validation, resampler geometry, bad buffers and smart-decode limits.

mod common;

use dynamo_multimodal::image::decode::DecodeLimits;
use dynamo_multimodal::processor::MediaMetadata;
use dynamo_multimodal::registry::processor_from_spec;
use serde_json::{Value, json};

fn fixture_spec() -> Value {
    common::table("glm-5.3-flash.json")["cases"][0]["spec"].clone()
}

#[test]
fn invalid_pixel_specs_do_not_fall_back_to_the_planner() {
    for name in ["image_mean", "image_std"] {
        let mut spec = fixture_spec();
        spec.as_object_mut().unwrap().remove(name);
        assert!(
            processor_from_spec(&spec.to_string()).is_err(),
            "missing {name}"
        );
    }
    for value in [0.0, -1.0] {
        let mut spec = fixture_spec();
        spec["image_std"] = json!([value, 1.0, 1.0]);
        assert!(processor_from_spec(&spec.to_string()).is_err());
    }
}

#[test]
fn pil_refuses_patch_expansion_that_disagrees_with_the_shared_planner() {
    let mut spec = fixture_spec();
    spec["resample"] = json!("pil");
    spec["patch_expand_factor"] = json!(2);
    spec["min_image_tokens"] = json!(4);
    spec["max_image_tokens"] = json!(32);
    // For a 200x150 image, pinned HF PIL ignores the expansion factor and
    // produces grid [1, 10, 12] (30 tokens); the shared planner produces
    // [1, 12, 16] (48 tokens). Refuse the spec before either path is used.
    assert!(matches!(
        processor_from_spec(&spec.to_string()),
        Err(dynamo_multimodal::MmError::Unsupported { .. })
    ));
}

#[test]
fn supported_resamplers_keep_pixel_and_metadata_geometry_in_sync() {
    use dynamo_multimodal::processor::{DecodedMedia, Geometry};

    for (resample, expand, grid, tokens) in
        [("pil", 1, [1, 10, 12], 30), ("aten_u8", 2, [1, 12, 16], 48)]
    {
        let mut spec = fixture_spec();
        spec["resample"] = json!(resample);
        spec["patch_expand_factor"] = json!(expand);
        spec["min_image_tokens"] = json!(4);
        spec["max_image_tokens"] = json!(32);
        let family = processor_from_spec(&spec.to_string()).unwrap();
        let item = family
            .process_item(&DecodedMedia::Image {
                rgb: vec![42; 200 * 150 * 3],
                height: 150,
                width: 200,
            })
            .unwrap();
        let Some(Geometry::Grid(actual_grid)) = item.geometry else {
            panic!("expected an image grid");
        };
        assert_eq!(actual_grid, grid, "{resample}");
        assert_eq!(item.feature_token_count, tokens, "{resample}");
        assert_eq!(
            family
                .num_media_tokens(&MediaMetadata::Image {
                    width: 200,
                    height: 150,
                })
                .unwrap(),
            tokens,
            "{resample}"
        );
    }
}

#[test]
fn full_pipeline_rejects_bad_buffers_video_and_mismatched_offsets() {
    use dynamo_multimodal::processor::DecodedMedia;
    let family = processor_from_spec(&fixture_spec().to_string()).unwrap();
    for (height, width, rgb) in [(0, 1, vec![]), (1, 0, vec![]), (2, 2, vec![0; 11])] {
        assert!(
            family
                .process_item(&DecodedMedia::Image { rgb, height, width })
                .is_err()
        );
    }
    let item = family
        .process_item(&DecodedMedia::Image {
            rgb: vec![42; 3],
            height: 1,
            width: 1,
        })
        .unwrap();
    assert!(family.layout(&[154855, 154854], &[item]).is_err());
    assert!(family.positions(1, &[(0, 0)], &[]).is_err());
}

#[test]
fn smart_decode_checks_transposed_limits_before_decoding() {
    let family = processor_from_spec(&fixture_spec().to_string()).unwrap();
    for (name, oriented_height) in [
        ("exif_orientation6_64x48.jpg", 64),
        ("rgba_raw_exif6_56x48.png", 56),
    ] {
        let bytes = common::image(name);
        let limits = DecodeLimits {
            max_width: Some(48),
            max_height: Some(oriented_height),
            max_pixels: Some(48 * oriented_height),
        };
        assert!(family.decode_image(&bytes, &limits).is_ok(), "{name}");
        let limits = DecodeLimits {
            max_height: Some(oriented_height - 1),
            ..limits
        };
        assert!(
            matches!(
                family.decode_image(&bytes, &limits),
                Err(dynamo_multimodal::MmError::LimitExceeded { .. })
            ),
            "{name}"
        );
    }
}
