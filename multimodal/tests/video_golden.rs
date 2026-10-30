// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Video resize parity: sizes must equal what HF's `smart_resize` (and the
//! `cap_pixels_per_frame` branch of `Qwen3VLVideoProcessor.resize`) return,
//! as recorded by `tests/fixtures/video/generate.py`.

use dynamo_multimodal::models::qwen_vl::{Qwen3VlVideo, smart_video_resize};
use dynamo_multimodal::video::VideoPixelBudget;

#[derive(serde::Deserialize)]
struct ResizeCase {
    num_frames: usize,
    height: usize,
    width: usize,
    temporal_factor: usize,
    factor: usize,
    min_pixels: usize,
    max_pixels: usize,
    expected: Option<[usize; 2]>,
}

#[derive(serde::Deserialize)]
struct BudgetCase {
    num_frames: usize,
    height: usize,
    width: usize,
    total_pixels: Option<usize>,
    max_pixels_per_frame: Option<usize>,
    expected: [usize; 2],
}

#[derive(serde::Deserialize)]
struct Fixture {
    video_config: Qwen3VlVideo,
    resize: Vec<ResizeCase>,
    budget: Vec<BudgetCase>,
}

fn fixture() -> Fixture {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/video/smart_resize.json");
    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}

#[test]
fn smart_video_resize_matches_hf() {
    for c in fixture().resize {
        let got = smart_video_resize(
            c.num_frames,
            c.height,
            c.width,
            c.temporal_factor,
            c.factor,
            c.min_pixels,
            c.max_pixels,
        )
        .ok()
        .map(|(h, w)| [h, w]);
        assert_eq!(got, c.expected, "{}x{}x{}", c.num_frames, c.height, c.width);
    }
}

#[test]
fn request_budgets_match_hf_cap_semantics() {
    let f = fixture();
    for c in f.budget {
        let budget = VideoPixelBudget {
            total_pixels: c.total_pixels,
            max_pixels_per_frame: c.max_pixels_per_frame,
        };
        let (h, w) = f
            .video_config
            .resize(c.num_frames, c.height, c.width, &budget)
            .unwrap();
        assert_eq!(
            [h, w],
            c.expected,
            "{} frames, total {:?}, cap {:?}",
            c.num_frames,
            c.total_pixels,
            c.max_pixels_per_frame
        );
    }
}
