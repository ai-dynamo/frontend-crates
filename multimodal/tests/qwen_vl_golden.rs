// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The Qwen-VL values that `ProcessedItem` does not carry, against the
//! `stages` of `tests/fixtures/mm_parity/qwen2-vl.json`: each image's content
//! hash, and the resampler stage on its own, so a mismatch names the stage
//! instead of surfacing as an opaque `pixel_values` diff. `mm_parity.rs`
//! replays every output field.

mod common;

use dynamo_multimodal::content_hash_bytes;
use dynamo_multimodal::image::decode::{DecodeLimits, decode_rgb};
use dynamo_multimodal::image::resize;
use dynamo_multimodal::models::qwen_vl::{QwenVlSpec, smart_resize};

#[test]
fn content_hash_and_resampler_stage_match_the_reference() {
    let table = common::table("qwen2-vl.json");
    let cases = table["cases"].as_array().unwrap();
    let names: Vec<&str> = cases
        .iter()
        .map(|case| case["name"].as_str().unwrap())
        .collect();
    let all = [
        "aten_downscale",
        "aten_multi",
        "aten_upscale",
        "pil_round",
        "pil_tall",
    ];
    assert_eq!(names, all, "a dropped or renamed case");
    for case in cases {
        let name = &case["name"];
        let spec: QwenVlSpec = serde_json::from_value(case["spec"].clone()).unwrap();
        let stages = &case["stages"];
        for (i, image) in case["images"].as_array().unwrap().iter().enumerate() {
            let bytes = common::image(image.as_str().unwrap());
            assert_eq!(
                content_hash_bytes(&bytes),
                stages["content_hash"][i]
                    .as_str()
                    .unwrap()
                    .parse::<u64>()
                    .unwrap(),
                "{name}: hash[{i}]"
            );
            let (rgb, height, width) = decode_rgb(&bytes, &DecodeLimits::default()).unwrap();
            let (th, tw) = smart_resize(
                height,
                width,
                spec.patch_size * spec.merge_size,
                spec.min_pixels,
                spec.max_pixels,
            )
            .unwrap();
            let resized = resize::resize_rgb(&rgb, height, width, th, tw, spec.resample.into());
            assert_eq!(
                common::sha256(&resized),
                stages["resized_sha256"][i].as_str().unwrap(),
                "{name}: resized[{i}] ({height}x{width} -> {th}x{tw})"
            );
        }
    }
}
