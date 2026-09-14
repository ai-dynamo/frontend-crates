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

use dynamo_mm_preprocessor::image::resize::{Filter, Resample, resize_rgb};

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
fn resizing_spawns_no_worker_threads_while_unarmed() {
    let before = thread_names().len();

    // A two-axis resize enters both seams: `in_pool` around the passes, and
    // `for_chunks_mut` per row within each of them.
    let src: Vec<u8> = (0..48 * 64 * 3).map(|v| v as u8).collect();
    for resample in [Resample::AtenU8, Resample::Pil(Filter::Bicubic)] {
        let out = resize_rgb(&src, 48, 64, 24, 32, resample);
        assert_eq!(out.len(), 24 * 32 * 3);
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
