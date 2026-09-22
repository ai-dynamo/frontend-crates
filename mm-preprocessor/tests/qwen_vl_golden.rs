// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! End-to-end golden replay: every output field of the README §2.2 composition
//! must be byte-identical to fixtures produced from the mirrored HF processor
//! (and `get_rope_index`) by `tests/fixtures/qwen_vl/generate.py`, which also
//! records the library versions it ran under. A systematic skew — wrong
//! resample filter, pass order, fused-vs-unfused normalize rounding, patch
//! order — still yields plausible-looking tensors; only bitwise comparison
//! catches it without a model in the loop.
//!
//! The resampler stage is pinned on its own (`resized_<i>.u8`) so a mismatch
//! names the stage instead of surfacing as an opaque `pixel_values` diff.

use dynamo_mm_preprocessor::image::decode::{DecodeLimits, decode_rgb};
use dynamo_mm_preprocessor::image::resize;
use dynamo_mm_preprocessor::models::qwen_vl::{QwenVlSpec, smart_resize};
use dynamo_mm_preprocessor::processor::{DecodedMedia, PositionOutput, TensorData};
use dynamo_mm_preprocessor::registry::processor_from_spec;
use dynamo_mm_preprocessor::{content_hash_bytes, token_layout};

/// Every fixture directory; a dropped or misnamed case fails by name.
const CASES: [&str; 5] = [
    "aten_downscale",
    "aten_multi",
    "aten_upscale",
    "pil_round",
    "pil_tall",
];

#[derive(serde::Deserialize)]
struct Case {
    spec: serde_json::Value,
    prompt_ids: Vec<i32>,
    input_ids: Vec<i32>,
    grids: Vec<[i64; 3]>,
    offsets: Vec<(u32, u32)>,
    /// Decimal strings: JSON numbers cannot carry a full u64.
    hashes: Vec<String>,
    mrope_delta: i64,
}

fn read(dir: &std::path::Path, name: &str) -> Vec<u8> {
    std::fs::read(dir.join(name)).unwrap_or_else(|e| panic!("{name}: {e}"))
}

#[test]
fn pipeline_output_matches_golden_fixtures() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen_vl");
    for name in CASES {
        let dir = root.join(name);
        let case: Case = serde_json::from_slice(&read(&dir, "case.json")).unwrap();

        let images = (0..)
            .map_while(|i| std::fs::read(dir.join(format!("input_{i}.png"))).ok())
            .collect::<Vec<_>>();
        let family = processor_from_spec(&case.spec.to_string()).unwrap();
        // The same knobs, typed, for replaying the resampler stage alone.
        let spec: QwenVlSpec = serde_json::from_value(case.spec.clone()).unwrap();

        let mut feature_bytes = Vec::new();
        let items = images
            .iter()
            .enumerate()
            .map(|(i, bytes)| {
                assert_eq!(
                    content_hash_bytes(bytes),
                    case.hashes[i].parse::<u64>().unwrap(),
                    "{name}: hash[{i}]"
                );
                let (rgb, height, width) = decode_rgb(bytes, &DecodeLimits::default()).unwrap();
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
                    resized,
                    read(&dir, &format!("resized_{i}.u8")),
                    "{name}: resized[{i}] ({height}x{width} -> {th}x{tw})"
                );
                let item = family
                    .process_item(&DecodedMedia::Image { rgb, height, width })
                    .unwrap_or_else(|e| panic!("{name}: {e}"));
                let (aux_name, grid) = &item.aux[0];
                assert_eq!(aux_name, "image_grid_thw", "{name}: aux[{i}]");
                let TensorData::I64(grid) = &grid.data else {
                    panic!("{name}: grid[{i}] dtype");
                };
                assert_eq!(grid[..], case.grids[i], "{name}: grid[{i}]");
                let TensorData::F32(pixel_values) = &item.feature.data else {
                    panic!("{name}: feature[{i}] dtype");
                };
                feature_bytes.extend(pixel_values.iter().flat_map(|v| v.to_le_bytes()));
                item
            })
            .collect::<Vec<_>>();
        assert_eq!(items.len(), case.grids.len(), "{name}: item count");
        assert_eq!(
            feature_bytes,
            read(&dir, "pixel_values.f32le"),
            "{name}: pixel_values bytes"
        );

        let layout = family.layout(&case.prompt_ids, &items).unwrap();
        let counts = items
            .iter()
            .map(|item| item.feature_token_count)
            .collect::<Vec<_>>();
        let expanded =
            token_layout::apply_layout(&case.prompt_ids, &layout, &counts, case.input_ids.len())
                .unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(expanded.input_ids, case.input_ids, "{name}: input_ids");
        assert_eq!(expanded.offsets, case.offsets, "{name}: offsets");
        // Image expansions are a single `Feature` part: each item's feature
        // range is exactly its whole (inclusive) offset span.
        for (i, &(start, end)) in expanded.offsets.iter().enumerate() {
            assert_eq!(
                expanded.feature_ranges[i],
                vec![start..end + 1],
                "{name}: feature_ranges[{i}]"
            );
        }

        let positions = family
            .positions(expanded.input_ids.len(), &expanded.offsets, &items)
            .unwrap();
        let PositionOutput::MRope { positions, delta } = positions else {
            panic!("{name}: expected M-RoPE");
        };
        let mrope_bytes: Vec<u8> = positions.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(
            mrope_bytes,
            read(&dir, "mrope.i64le"),
            "{name}: mrope bytes"
        );
        assert_eq!(delta, case.mrope_delta, "{name}: mrope delta");
    }
}
