// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! End-to-end golden replay: every output field of the README §2.2 composition
//! must be byte-identical to fixtures produced from the mirrored HF processor
//! (and `get_rope_index`) by SGLang's `generate_dynamo_golden.py`. A
//! systematic skew — wrong resample filter, fused-vs-unfused normalize
//! rounding, patch order — still yields plausible-looking tensors; only
//! bitwise comparison catches it without a model in the loop.

use dynamo_mm_preprocessor::processor::{DecodedMedia, PositionOutput, TensorData};
use dynamo_mm_preprocessor::registry::processor_from_spec;
use dynamo_mm_preprocessor::{content_hash_bytes, image::decode::decode_rgb, token_layout};

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
    let mut cases = 0;
    for entry in std::fs::read_dir(&root).expect("fixtures dir") {
        let dir = entry.unwrap().path();
        let case: Case = serde_json::from_slice(&read(&dir, "case.json")).unwrap();
        let name = dir.file_name().unwrap().to_string_lossy().into_owned();

        let images = (0..)
            .map_while(|i| std::fs::read(dir.join(format!("input_{i}.png"))).ok())
            .collect::<Vec<_>>();
        let family = processor_from_spec(&case.spec.to_string()).unwrap();

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
                let (rgb, height, width) = decode_rgb(bytes).unwrap();
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
        let expanded = token_layout::apply_layout(&case.prompt_ids, &layout, &counts)
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
        cases += 1;
    }
    assert!(cases >= 4, "expected fixtures under {}", root.display());
}
