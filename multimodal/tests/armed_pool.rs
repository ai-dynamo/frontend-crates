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

    // A thread that a worker starts, such as CUDA's in the decode test below,
    // inherits the worker's name.
    let named: std::collections::BTreeSet<String> = std::fs::read_dir("/proc/self/task")
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

#[test]
fn ordered_fanout_consumes_in_input_order_and_stops_at_the_first_error() {
    execution::init_pool(2).unwrap();
    let mut seen = Vec::new();
    let r = execution::try_for_each_ordered(
        0..64usize,
        3,
        |i| {
            if i % 3 == 0 {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            if i == 40 { Err(i) } else { Ok(i * 2) }
        },
        |v| {
            seen.push(v);
            Ok(())
        },
    );
    assert_eq!(r, Err(40));
    assert_eq!(seen, (0..40).map(|i| i * 2).collect::<Vec<_>>());
    let r =
        execution::try_for_each_ordered(0..8usize, 2, Ok, |v| if v == 5 { Err(v) } else { Ok(()) });
    assert_eq!(r, Err(5));
}

#[test]
fn ordered_fanout_passes_a_panic_to_the_caller() {
    execution::init_pool(2).unwrap();
    let r = std::panic::catch_unwind(|| {
        execution::try_for_each_ordered(
            0..8usize,
            2,
            |i| {
                assert_ne!(i, 3);
                Ok::<_, ()>(i)
            },
            |_| Ok(()),
        )
    });
    assert!(r.is_err());
}

#[cfg(feature = "video-decode")]
#[test]
fn armed_decode_matches_the_inline_decode() {
    use dynamo_multimodal::image::resize::Resample;
    use dynamo_multimodal::video_decode::{
        Fit, FrameResize, VideoDecodeLimits, decode_frames, probe, sample_indices,
    };
    execution::init_pool(2).unwrap();
    let limits = VideoDecodeLimits::default();
    // NVDEC decodes these where it can, else libvpx: CUDA_VISIBLE_DEVICES=
    // runs the libvpx path on a GPU host.
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
        let info = probe(&bytes).unwrap();
        let n = info.frame_count;
        let half = FrameResize {
            width: info.width / 2,
            height: info.height / 2,
            filter: Resample::AtenU8,
        };
        for resize in [None, Some(half)] {
            for set in [
                (0..n).collect(),
                sample_indices(n, 4).unwrap(),
                vec![n / 2, n / 2, n - 1],
            ] {
                let decode = || {
                    let fit = resize.as_ref().map(|r| r as &dyn Fit);
                    decode_frames(&bytes, &set, fit, &limits).unwrap()
                };
                let armed = decode();
                // On a pool thread the decode sends no frame to the pool.
                let inline = execution::in_pool(decode);
                for k in 0..set.len() {
                    assert!(
                        armed.frame(k) == inline.frame(k),
                        "{name} {resize:?}: frame {k} of {set:?}"
                    );
                }
                assert_eq!(armed.timestamps(), inline.timestamps(), "{name} {set:?}");
            }
        }
    }
}
