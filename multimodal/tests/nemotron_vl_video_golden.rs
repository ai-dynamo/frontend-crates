// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Nemotron 3.5 Super VL video golden replay against the fixtures
//! `tests/fixtures/nemotron_vl_video/generate.py` records: SGLang's
//! `NanoNemotronVLImageProcessor` for `sizing = sglang` (ids, item offsets,
//! bfloat16 features) and the checkpoint's remote-code processor for
//! `sizing = hf` (ids and frame size; the crate has no HF pixel path).

use std::collections::HashMap;

use dynamo_multimodal::MmError;
use dynamo_multimodal::models::nemotron_vl::{
    NemotronVlProcessor, NemotronVlSizing, NemotronVlSpec, NemotronVlVideo,
};
use dynamo_multimodal::processor::{
    DecodedMedia, MmFamilyProcessor, Modality, ProcessedItem, TensorData, TextEncoder,
};
use dynamo_multimodal::token_layout::{ExpandedPrompt, apply_layout};

const CASES: [&str; 14] = [
    "sglang_wide",
    "sglang_portrait_odd",
    "sglang_single_frame",
    "sglang_thin",
    "sglang_target_1024_wide",
    "sglang_target_1024_portrait_odd",
    "sglang_no_evs_portrait_odd",
    "sglang_two_videos",
    "sglang_target_1024_two_videos",
    "hf_wide",
    "hf_portrait_odd",
    "hf_single_frame",
    "hf_thin",
    "hf_no_fps_portrait_odd",
];

#[derive(serde::Deserialize)]
struct Media {
    shape: [usize; 3],
    seed: u64,
    frame_indices: Vec<u64>,
    fps: f64,
}

#[derive(serde::Deserialize)]
struct Case {
    spec: serde_json::Value,
    media: Vec<Media>,
    prompt_ids: Vec<i32>,
    label_ids: HashMap<String, Vec<i32>>,
    input_ids: Vec<i32>,
    offsets: Vec<[u32; 2]>,
    feature_shape: Vec<usize>,
    feature_dtype: String,
    feature_blake3: Option<String>,
}

/// The tokenizer's ids for each frame label, as the generator recorded them.
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

/// The frames the generator fed the reference: the clip's sampled frames.
fn frames(media: &Media) -> Vec<u8> {
    let [total, h, w] = media.shape;
    let clip = noise(total * h * w * 3, media.seed);
    (media.frame_indices.iter())
        .flat_map(|&i| clip.chunks_exact(h * w * 3).nth(i as usize).unwrap())
        .copied()
        .collect()
}

/// SGLang's one feature tensor: every video's frames, concatenated.
fn bf16_digest(items: &[ProcessedItem]) -> String {
    let bytes: Vec<u8> = (items.iter())
        .flat_map(|item| {
            let TensorData::Bf16(bits) = &item.feature.data else {
                panic!("feature dtype");
            };
            bits.iter().flat_map(|v| v.to_le_bytes())
        })
        .collect();
    blake3::hash(&bytes).to_hex().to_string()
}

fn check_prompt(name: &str, case: &Case, expanded: &ExpandedPrompt) {
    assert_eq!(expanded.input_ids, case.input_ids, "{name}: input_ids");
    let runs: Vec<[u32; 2]> = (expanded.feature_ranges.iter().flatten())
        .map(|range| [range.start, range.end - 1])
        .collect();
    assert_eq!(runs, case.offsets, "{name}: <image> runs");
}

#[test]
fn video_matches_golden_fixtures() {
    let root =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/nemotron_vl_video");
    for name in CASES {
        let json = std::fs::read(root.join(name).join("case.json")).unwrap();
        let case: Case = serde_json::from_slice(&json).unwrap();
        let spec: NemotronVlSpec = serde_json::from_value(case.spec.clone()).unwrap();
        let processor = NemotronVlProcessor::new(spec.clone()).unwrap();
        assert!(processor.capabilities().supports(Modality::Video));
        let text = Recorded(case.label_ids.clone());
        let plans: Vec<_> = (case.media.iter())
            .map(|media| {
                let [_, h, w] = media.shape;
                (processor.plan_video(w as u32, h as u32, media.frame_indices.len())).unwrap()
            })
            .collect();
        let num_frames: usize = case.media.iter().map(|m| m.frame_indices.len()).sum();
        for plan in &plans {
            let frame = [
                num_frames,
                3,
                plan.resized_height as usize,
                plan.resized_width as usize,
            ];
            assert_eq!(case.feature_shape, frame, "{name}: frame size");
        }
        let counts: Vec<usize> = (plans.iter())
            .map(|plan| plan.group_tokens.iter().map(|&n| n as usize).sum())
            .collect();

        let videos: Vec<_> = (case.media.iter())
            .map(|media| NemotronVlVideo {
                width: media.shape[2] as u32,
                height: media.shape[1] as u32,
                frame_indices: &media.frame_indices,
                fps: media.fps,
            })
            .collect();
        let layout = processor
            .video_layout(&case.prompt_ids, &videos, &text)
            .unwrap();
        let expanded = apply_layout(&case.prompt_ids, &layout, &counts, usize::MAX).unwrap();
        check_prompt(name, &case, &expanded);

        let media: Vec<_> = (case.media.iter())
            .map(|media| DecodedMedia::Video {
                rgb: frames(media),
                height: media.shape[1],
                width: media.shape[2],
                frame_indices: media.frame_indices.clone(),
                fps: media.fps,
            })
            .collect();
        match spec.sizing {
            NemotronVlSizing::Hf => {
                assert_eq!(case.feature_dtype, "float32", "{name}");
                assert!(matches!(
                    processor.process_items(&case.prompt_ids, &media),
                    Err(MmError::Unsupported { .. })
                ));
            }
            NemotronVlSizing::Sglang => {
                assert_eq!(case.feature_dtype, "bfloat16", "{name}");
                let items = processor
                    .process_items(&case.prompt_ids, &media)
                    .unwrap_or_else(|e| panic!("{name}: {e}"));
                assert_eq!(
                    Some(bf16_digest(&items)),
                    case.feature_blake3,
                    "{name}: feature"
                );
                let layout = processor
                    .layout_with_text(&case.prompt_ids, &items, &text)
                    .unwrap();
                let counts: Vec<usize> =
                    items.iter().map(|item| item.feature_token_count).collect();
                let expanded =
                    apply_layout(&case.prompt_ids, &layout, &counts, usize::MAX).unwrap();
                check_prompt(name, &case, &expanded);
            }
        }
    }
}

/// A video needs a video spec, the host's encoder for its labels, a buffer that
/// matches its frames, a usable fps, and a request without images.
#[test]
fn refuses_malformed_video_requests() {
    let root =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/nemotron_vl_video");
    let json = std::fs::read(root.join("sglang_wide/case.json")).unwrap();
    let case: Case = serde_json::from_slice(&json).unwrap();
    let spec: NemotronVlSpec = serde_json::from_value(case.spec).unwrap();
    let processor = NemotronVlProcessor::new(spec.clone()).unwrap();
    let video = |rgb: Vec<u8>, fps: f64| DecodedMedia::Video {
        rgb,
        height: 2,
        width: 2,
        frame_indices: vec![0, 1],
        fps,
    };
    let ids = [7, 18, 9];
    let ok = video(vec![0; 2 * 2 * 2 * 3], 30.0);
    let items = processor.process_items(&ids, &[ok]).unwrap();
    assert_eq!(items[0].feature.shape, [2, 3, 224, 224]);
    assert!(matches!(
        processor.layout(&ids, &items),
        Err(MmError::InvalidInput { .. })
    ));
    for bad in [
        video(vec![0; 23], 30.0),
        video(vec![0; 24], -1.0),
        video(vec![0; 24], f64::NAN),
    ] {
        assert!(matches!(
            processor.process_items(&ids, &[bad]),
            Err(MmError::InvalidInput { .. })
        ));
    }
    let image = DecodedMedia::Image {
        rgb: vec![0; 2 * 2 * 3],
        height: 2,
        width: 2,
    };
    assert!(matches!(
        processor.process_items(&[18, 18], &[video(vec![0; 24], 30.0), image]),
        Err(MmError::Unsupported { .. })
    ));

    let images_only = NemotronVlProcessor::new(NemotronVlSpec {
        video: None,
        ..spec.clone()
    })
    .unwrap();
    assert!(!images_only.capabilities().supports(Modality::Video));
    assert!(matches!(
        images_only.process_items(&ids, &[video(vec![0; 24], 30.0)]),
        Err(MmError::Unsupported { .. })
    ));
    let mut one_frame_groups = spec.clone();
    one_frame_groups.video.as_mut().unwrap().temporal_patch_size = 1;
    assert!(NemotronVlProcessor::new(one_frame_groups).is_err());

    let huge = processor.plan_video(224, 224, usize::MAX);
    assert!(matches!(huge, Err(MmError::LimitExceeded { .. })));
    let ungrouped = ProcessedItem {
        geometry: None,
        ..items.into_iter().next().unwrap()
    };
    assert!(matches!(
        processor.layout_with_text(&ids, &[ungrouped], &Recorded(case.label_ids)),
        Err(MmError::InvalidInput { .. })
    ));

    let size = |width: usize| DecodedMedia::Video {
        rgb: vec![0; 2 * width * 2 * 3],
        height: 2,
        width,
        frame_indices: vec![0, 1],
        fps: 30.0,
    };
    let two = [18, 7, 18];
    let mut target = spec.clone();
    target.video.as_mut().unwrap().target_num_patches = 1024;
    let target = NemotronVlProcessor::new(target).unwrap();
    assert_eq!(
        processor
            .process_items(&two, &[size(2), size(4)])
            .unwrap()
            .len(),
        2
    );
    assert!(target.process_items(&two, &[size(2), size(2)]).is_ok());
    assert!(matches!(
        target.process_items(&two, &[size(2), size(4)]),
        Err(MmError::Unsupported { .. })
    ));
    let clip = |width| NemotronVlVideo {
        width,
        height: 2,
        frame_indices: &[0, 1],
        fps: 30.0,
    };
    assert!(matches!(
        target.video_layout(&two, &[clip(2), clip(4)], &Recorded(HashMap::new())),
        Err(MmError::Unsupported { .. })
    ));
    let hf = NemotronVlProcessor::new(NemotronVlSpec {
        sizing: NemotronVlSizing::Hf,
        ..spec.clone()
    })
    .unwrap();
    assert!(matches!(
        hf.video_layout(&two, &[clip(2), clip(2)], &Recorded(HashMap::new())),
        Err(MmError::Unsupported { .. })
    ));
    assert!(matches!(
        hf.plan_video(224, 224, usize::MAX),
        Err(MmError::LimitExceeded { .. })
    ));
}
