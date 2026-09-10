// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Until a consumer arms the pool (`execution::init_pool`), this crate must
//! not own worker threads — with or without rayon linked: a serving engine
//! supplies concurrency across requests and pins its own cores, so a library
//! spawning pools behind its back would fight it.
//!
//! Guarded from the outside (thread count of the process) rather than by
//! inspecting the code, so it stays true no matter how the fan-out seam in
//! `execution` is refactored. Lives in its own test binary so nothing else
//! arms the pool in this process.

#![cfg(target_os = "linux")]

use dynamo_mm_preprocessor::processor::DecodedMedia;
use dynamo_mm_preprocessor::registry::processor_from_spec;

const SPEC: &str = r#"{"family":"qwen_vl","image_token_id":1,"patch_size":14,
    "merge_size":2,"temporal_patch_size":2,"min_pixels":3136,
    "max_pixels":12845056,"image_mean":[0.0,0.0,0.0],"image_std":[1.0,1.0,1.0]}"#;

fn thread_names() -> Vec<String> {
    std::fs::read_dir("/proc/self/task")
        .expect("procfs")
        .filter_map(|entry| {
            let comm = entry.ok()?.path().join("comm");
            Some(std::fs::read_to_string(comm).ok()?.trim().to_string())
        })
        .collect()
}

#[test]
fn processing_spawns_no_worker_threads_while_unarmed() {
    let before = thread_names().len();
    let family = processor_from_spec(SPEC).unwrap();

    // Two images through resize + patchify, so the fan-out seams are
    // exercised, not bypassed.
    for (h, w) in [(112usize, 112usize), (84, 140)] {
        let rgb = (0..h * w * 3).map(|v| v as u8).collect();
        let item = family
            .process_item(&DecodedMedia::Image {
                rgb,
                height: h,
                width: w,
            })
            .expect("processing should succeed");
        assert!(item.feature_token_count > 0);
    }

    let after = thread_names();
    let spawned: Vec<&String> = after.iter().filter(|t| t.starts_with("dyn-mm")).collect();
    assert!(
        spawned.is_empty(),
        "unarmed crate spawned worker threads: {spawned:?}"
    );
    assert_eq!(
        after.len(),
        before,
        "unarmed crate changed the process thread count: {after:?}"
    );
}
