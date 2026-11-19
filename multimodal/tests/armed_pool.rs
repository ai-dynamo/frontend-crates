// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Arming the pool is process-global state, so everything about it lives in
//! this one test binary (its counterpart, the unarmed guard, has its own).

#![cfg(all(feature = "parallel", target_os = "linux"))]

use dynamo_multimodal::{MmError, execution};

#[test]
fn init_pool_arms_fanout_and_pins_the_thread_count() {
    execution::init_pool(2).unwrap();

    let mut buf = vec![0usize; 8];
    execution::for_chunks_mut(&mut buf, 2, |i, chunk| chunk.fill(i));
    assert_eq!(buf, [0, 0, 1, 1, 2, 2, 3, 3]);

    // Workers spawn asynchronously and name themselves once running; a
    // broadcast returns only after every worker has run it.
    execution::in_pool(|| rayon::broadcast(|_| ()));

    let named: Vec<String> = std::fs::read_dir("/proc/self/task")
        .expect("procfs")
        .filter_map(|entry| {
            let comm = entry.ok()?.path().join("comm");
            std::fs::read_to_string(comm).ok()
        })
        .map(|name| name.trim().to_string())
        .filter(|name| name.starts_with("dyn-mm"))
        .collect();
    assert_eq!(
        named.len(),
        2,
        "expected the armed pool's threads: {named:?}"
    );

    execution::init_pool(2).unwrap();
    assert!(matches!(
        execution::init_pool(3),
        Err(MmError::InvalidInput { .. })
    ));
}

#[cfg(feature = "video-decode")]
#[test]
fn armed_cpu_decode_matches_the_inline_decode() {
    use dynamo_multimodal::video_decode::{VideoDecodeLimits, decode_frames, sample_indices};
    execution::init_pool(2).unwrap();
    let limits = VideoDecodeLimits::default();
    for name in [
        "gop_vp8.webm",
        "gop_vp9.webm",
        "vp9_altref.webm",
        "vp9_invisible.webm",
    ] {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/video_decode")
            .join(name);
        let bytes = std::fs::read(&path).unwrap();
        let n = dynamo_multimodal::video_decode::probe(&bytes)
            .unwrap()
            .frame_count;
        for set in [
            (0..n).collect(),
            sample_indices(n, 4).unwrap(),
            vec![n / 2, n / 2, n - 1],
        ] {
            let armed = decode_frames(&bytes, &set, None, &limits).unwrap();
            // On a pool thread every helper runs inline.
            let inline = execution::in_pool(|| decode_frames(&bytes, &set, None, &limits)).unwrap();
            for k in 0..set.len() {
                assert!(
                    armed.frame(k) == inline.frame(k),
                    "{name}: frame {k} of {set:?}"
                );
            }
            assert_eq!(armed.timestamps(), inline.timestamps(), "{name} {set:?}");
        }
    }
}
