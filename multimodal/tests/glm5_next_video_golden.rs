// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! GLM-5.3-Flash video parity: each stage must equal what
//! `tests/fixtures/glm5_next_video/generate.py` records from SGLang's GLM
//! processor, the engine that consumes these items. The fixtures also record
//! transformers' `Glm5NextProcessor` on the same clips; `KNOWN_GAPS` names the
//! fields where SGLang, and so this crate, differ from it.

use std::collections::HashMap;

use dynamo_multimodal::MmError;
use dynamo_multimodal::models::glm5_next::Glm5NextImageSpec;
use dynamo_multimodal::processor::{
    DecodedMedia, Modality, PositionOutput, ProcessedItem, TensorData, TextEncoder,
};
use dynamo_multimodal::registry::{
    ProcessorSpec, build_processor, glm5_next_pixel_spec, spec_from_hf_configs,
};
use dynamo_multimodal::token_layout;
use dynamo_multimodal::video::sample::glm_frame_indices;

/// Per case, the fields where SGLang differs from HF. SGLang resizes the
/// frames itself (stretched to the canvas, uint8 bicubic) before HF's
/// processor runs, so pixels differ everywhere; it splits the video budget
/// across a request's videos; and HF keeps no frame of a clip under 0.5 s.
const KNOWN_GAPS: [(&str, &[&str]); 7] = [
    ("fps30_3s", &["pixel_values_videos_blake3"]),
    ("odd_fill", &["pixel_values_videos_blake3"]),
    ("slow_source", &["pixel_values_videos_blake3"]),
    ("ntsc_tall", &["pixel_values_videos_blake3"]),
    ("short_clip", &["rejects"]),
    ("budget_cut", &["pixel_values_videos_blake3"]),
    (
        "video_image_video",
        &[
            "video_grids",
            "input_ids",
            "video_spans",
            "pixel_values_videos_blake3",
        ],
    ),
];

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
        fps: f64,
    },
}

#[derive(serde::Deserialize)]
struct Reference {
    sampled: Vec<Vec<u64>>,
    input_ids: Vec<i32>,
    video_grids: Vec<Vec<i64>>,
    video_spans: Vec<Vec<[u32; 2]>>,
    pixel_values_videos_blake3: String,
    mrope_blake3: String,
    mrope_delta: i64,
}

/// The checkpoint's `config.json` (the fields the resolver reads) and
/// `processor_config.json`.
#[derive(serde::Deserialize)]
struct Checkpoint {
    config: serde_json::Value,
    processor_config: serde_json::Value,
}

#[derive(serde::Deserialize)]
struct Case {
    checkpoint: Checkpoint,
    spec: serde_json::Value,
    media: Vec<Media>,
    prompt_ids: Vec<i32>,
    timestamp_ids: HashMap<String, Vec<i32>>,
    sglang: Reference,
    sglang_differs: Vec<String>,
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
fn video_pipeline_matches_sglang() {
    let root =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/glm5_next_video");
    for (name, gaps) in KNOWN_GAPS {
        let json = std::fs::read(root.join(name).join("case.json")).unwrap();
        let case: Case = serde_json::from_slice(&json).unwrap();
        assert_eq!(case.sglang_differs, gaps, "{name}: SGLang versus HF");
        let (config, pre) = (
            case.checkpoint.config.to_string(),
            case.checkpoint.processor_config.to_string(),
        );
        let want: Glm5NextImageSpec = serde_json::from_value(case.spec.clone()).unwrap();
        let Ok(ProcessorSpec::Glm5Next(planner)) = spec_from_hf_configs(&config, &pre) else {
            panic!("{name}: expected the glm5_next planner");
        };
        assert_eq!(planner, want.planner, "{name}: resolved planner");
        let pixels = glm5_next_pixel_spec(&config, &pre).unwrap();
        assert_eq!(pixels, want, "{name}: resolved pixel spec");
        let family = build_processor(ProcessorSpec::Glm5NextImage(pixels)).unwrap();
        let video = &case.spec["video"];
        let (target_fps, max_frames) = (
            video["fps"].as_f64().unwrap(),
            video["max_frames"].as_u64().unwrap() as u32,
        );

        let mut media = Vec::new();
        let mut sampled = Vec::new();
        for item in &case.media {
            media.push(match item {
                Media::Image {
                    shape: [h, w],
                    seed,
                } => DecodedMedia::Image {
                    rgb: noise(h * w * 3, *seed),
                    height: *h,
                    width: *w,
                },
                Media::Video {
                    shape: [t, h, w],
                    seed,
                    fps,
                } => {
                    let source = noise(t * h * w * 3, *seed);
                    let indices =
                        glm_frame_indices(*t as u64, *fps, target_fps, max_frames).unwrap();
                    let frame = h * w * 3;
                    let rgb = (indices.iter())
                        .flat_map(|&i| &source[i as usize * frame..][..frame])
                        .copied()
                        .collect();
                    sampled.push(indices.clone());
                    DecodedMedia::Video {
                        rgb,
                        height: *h,
                        width: *w,
                        frame_indices: indices,
                        fps: *fps,
                    }
                }
            });
        }
        for item in media.iter().filter(|m| m.modality() == Modality::Video) {
            assert!(
                matches!(family.process_item(item), Err(MmError::Unsupported { .. })),
                "{name}: a video needs its request (process_items)"
            );
        }
        let want = &case.sglang;
        assert_eq!(sampled, want.sampled, "{name}: sampled frames");

        let items = family
            .process_items(&case.prompt_ids, &media)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        let videos: Vec<&ProcessedItem> = (items.iter())
            .filter(|item| item.modality == Modality::Video)
            .collect();
        let grids: Vec<Vec<i64>> = videos.iter().map(|item| grid(item)).collect();
        assert_eq!(grids, want.video_grids, "{name}: video_grid_thw");
        let features: Vec<u8> = videos.iter().flat_map(|item| f32_bytes(item)).collect();
        assert_eq!(
            digest(&features),
            want.pixel_values_videos_blake3,
            "{name}: pixel_values_videos"
        );

        let layout = family
            .layout_with_text(&case.prompt_ids, &items, &Recorded(case.timestamp_ids))
            .unwrap();
        let counts: Vec<usize> = items.iter().map(|item| item.feature_token_count).collect();
        let expanded = token_layout::apply_layout(&case.prompt_ids, &layout, &counts, usize::MAX)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(expanded.input_ids, want.input_ids, "{name}: input_ids");
        let spans: Vec<Vec<[u32; 2]>> = (items.iter().zip(&expanded.feature_ranges))
            .filter(|(item, _)| item.modality == Modality::Video)
            .map(|(_, ranges)| ranges.iter().map(|r| [r.start, r.end - 1]).collect())
            .collect();
        assert_eq!(spans, want.video_spans, "{name}: video feature spans");

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
        assert_eq!(digest(&mrope), want.mrope_blake3, "{name}: mrope");
        assert_eq!(delta, want.mrope_delta, "{name}: mrope delta");
    }
}
