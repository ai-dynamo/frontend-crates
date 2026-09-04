// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! End-to-end golden replay: every field of the driver output must be
//! byte-identical to fixtures produced from the mirrored HF processor (and
//! `get_rope_index`) by SGLang's `generate_dynamo_golden.py`. A systematic
//! skew — wrong resample filter, fused-vs-unfused normalize rounding, patch
//! order — still yields plausible-looking tensors; only bitwise comparison
//! catches it without a model in the loop.

use dynamo_mm_preprocessor::driver::{ImageSource, MmInput, process};
use dynamo_mm_preprocessor::pipeline::{PositionOutput, TensorData};
use dynamo_mm_preprocessor::registry::pipeline_from_spec;

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
fn driver_output_matches_golden_fixtures() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen_vl");
    let mut cases = 0;
    for entry in std::fs::read_dir(&root).expect("fixtures dir") {
        let dir = entry.unwrap().path();
        let case: Case = serde_json::from_slice(&read(&dir, "case.json")).unwrap();
        let name = dir.file_name().unwrap().to_string_lossy().into_owned();

        let images = (0..)
            .map_while(|i| {
                std::fs::read(dir.join(format!("input_{i}.png")))
                    .ok()
                    .map(ImageSource::Bytes)
            })
            .collect::<Vec<_>>();
        let family = pipeline_from_spec(&case.spec.to_string()).unwrap();
        let out = process(
            family.as_ref(),
            MmInput {
                text: None,
                input_ids: Some(case.prompt_ids.clone()),
                images,
            },
            |_| Err("fixtures carry input_ids".into()),
        )
        .unwrap_or_else(|e| panic!("{name}: {e}"));

        assert_eq!(out.input_ids, case.input_ids, "{name}: input_ids");
        assert_eq!(out.offsets, case.offsets, "{name}: offsets");
        assert_eq!(out.items.len(), case.grids.len(), "{name}: item count");

        let mut feature_bytes = Vec::new();
        for (i, item) in out.items.iter().enumerate() {
            assert_eq!(
                item.hash,
                case.hashes[i].parse::<u64>().unwrap(),
                "{name}: hash[{i}]"
            );
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
        }
        assert_eq!(
            feature_bytes,
            read(&dir, "pixel_values.f32le"),
            "{name}: pixel_values bytes"
        );

        let PositionOutput::MRope { positions, delta } = &out.positions else {
            panic!("{name}: expected M-RoPE");
        };
        let mrope_bytes: Vec<u8> = positions.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(
            mrope_bytes,
            read(&dir, "mrope.i64le"),
            "{name}: mrope bytes"
        );
        assert_eq!(*delta, case.mrope_delta, "{name}: mrope delta");
        cases += 1;
    }
    assert!(cases >= 4, "expected fixtures under {}", root.display());
}
