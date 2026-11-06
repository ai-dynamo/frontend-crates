// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Qwen3-VL video golden replay: each stage must hash as the fixtures
//! `tests/fixtures/qwen3_vl_video/generate.py` records from HF's
//! `Qwen3VLProcessor` (with Qwen3-VL-2B-Instruct's tokenizer) and
//! `Qwen3VLModel.get_rope_index`. A resized video's frames are pinned on
//! their own, so a mismatch names the stage.

use std::collections::HashMap;

use dynamo_multimodal::MmError;
use dynamo_multimodal::image::resize;
use dynamo_multimodal::models::qwen_vl::QwenVlSpec;
use dynamo_multimodal::processor::{
    DecodedMedia, Modality, PositionOutput, ProcessedItem, TensorData, TextEncoder,
};
use dynamo_multimodal::registry::processor_from_spec;
use dynamo_multimodal::token_layout;

const CASES: [&str; 4] = ["downscale_6f", "odd_3f", "tps3", "video_image_video"];

#[derive(serde::Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
enum Media {
    Image {
        shape: [usize; 2],
        seed: u64,
    },
    Video {
        shape: [usize; 3],
        seed: u64,
        frame_indices: Vec<u64>,
        fps: f64,
        resized_blake3: Option<String>,
    },
}

#[derive(serde::Deserialize)]
struct Case {
    spec: serde_json::Value,
    media: Vec<Media>,
    prompt_ids: Vec<i32>,
    timestamp_ids: HashMap<String, Vec<i32>>,
    input_ids: Vec<i32>,
    video_grids: Vec<Vec<i64>>,
    mrope_delta: i64,
    pixel_values_videos_blake3: String,
    mrope_blake3: String,
}

/// The tokenizer's ids for each timestamp text, as the generator recorded them.
struct Recorded(HashMap<String, Vec<i32>>);

impl TextEncoder for Recorded {
    fn encode(&self, text: &str) -> dynamo_multimodal::Result<Vec<i32>> {
        (self.0.get(text).cloned())
            .ok_or_else(|| MmError::invalid_input(format!("no recorded ids for {text:?}")))
    }
}

/// The generator's `noise`: byte `i` is `(((i + seed) * 2654435761) >> 20) & 255`.
fn noise(len: usize, seed: u64) -> Vec<u8> {
    (0..len as u64)
        .map(|i| ((i + seed).wrapping_mul(2_654_435_761) >> 20) as u8)
        .collect()
}

fn digest(bytes: &[u8]) -> String {
    blake3::hash(bytes).to_hex().to_string()
}

fn f32_bytes(item: &ProcessedItem) -> Vec<u8> {
    let TensorData::F32(values) = &item.feature.data else {
        panic!("feature dtype");
    };
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn grid(item: &ProcessedItem) -> Vec<i64> {
    let TensorData::I64(grid) = &item.aux[0].1.data else {
        panic!("grid dtype");
    };
    grid.clone()
}

#[test]
fn video_pipeline_matches_golden_fixtures() {
    let root =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen3_vl_video");
    for name in CASES {
        let json = std::fs::read(root.join(name).join("case.json")).unwrap();
        let case: Case = serde_json::from_slice(&json).unwrap();
        let family = processor_from_spec(&case.spec.to_string()).unwrap();
        let spec: QwenVlSpec = serde_json::from_value(case.spec.clone()).unwrap();
        let sizing = &spec.video.as_ref().unwrap().processor;

        let mut items = Vec::new();
        for media in &case.media {
            let item = match media {
                Media::Image {
                    shape: [h, w],
                    seed,
                } => family.process_item(&DecodedMedia::Image {
                    rgb: noise(h * w * 3, *seed),
                    height: *h,
                    width: *w,
                }),
                Media::Video {
                    shape: [t, h, w],
                    seed,
                    frame_indices,
                    fps,
                    resized_blake3,
                } => {
                    let (h, w) = (*h, *w);
                    let rgb = noise(t * h * w * 3, *seed);
                    if let Some(want) = resized_blake3 {
                        let (th, tw) = sizing
                            .resize(frame_indices.len(), h, w, &Default::default())
                            .unwrap();
                        let frames: Vec<u8> = rgb
                            .chunks_exact(h * w * 3)
                            .flat_map(|frame| {
                                resize::resize_rgb(frame, h, w, th, tw, resize::Resample::AtenU8)
                            })
                            .collect();
                        assert_eq!(&digest(&frames), want, "{name}: resized frames");
                    }
                    family.process_item(&DecodedMedia::Video {
                        rgb,
                        height: h,
                        width: w,
                        frame_indices: frame_indices.clone(),
                        fps: *fps,
                    })
                }
            };
            items.push(item.unwrap_or_else(|e| panic!("{name}: {e}")));
        }
        let videos: Vec<&ProcessedItem> = (items.iter())
            .filter(|item| item.modality == Modality::Video)
            .collect();
        let grids: Vec<Vec<i64>> = videos.iter().map(|item| grid(item)).collect();
        assert_eq!(grids, case.video_grids, "{name}: video_grid_thw");
        let features: Vec<u8> = videos.iter().flat_map(|item| f32_bytes(item)).collect();
        assert_eq!(
            digest(&features),
            case.pixel_values_videos_blake3,
            "{name}: pixel_values_videos"
        );

        let layout = family
            .layout_with_text(&case.prompt_ids, &items, &Recorded(case.timestamp_ids))
            .unwrap();
        let counts: Vec<usize> = items.iter().map(|item| item.feature_token_count).collect();
        let expanded = token_layout::apply_layout(&case.prompt_ids, &layout, &counts, usize::MAX)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(expanded.input_ids, case.input_ids, "{name}: input_ids");

        let PositionOutput::MRope { positions, delta } = family
            .positions_with_features(
                expanded.input_ids.len(),
                &expanded.offsets,
                &expanded.feature_ranges,
                &items,
            )
            .unwrap()
        else {
            panic!("{name}: expected M-RoPE");
        };
        let mrope: Vec<u8> = positions.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(digest(&mrope), case.mrope_blake3, "{name}: mrope");
        assert_eq!(delta, case.mrope_delta, "{name}: mrope delta");
    }
}
