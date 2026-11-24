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
        "gop_vp9.mp4",
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
                // On a pool thread the decode sends no frame or part to the pool.
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

#[cfg(feature = "video-decode")]
#[test]
fn an_armed_mp4_decode_fails_like_the_inline_decode() {
    use dynamo_multimodal::video_decode::{
        ByteSource, VideoDecodeLimits, decode_frames_from, probe,
    };
    use std::sync::atomic::{AtomicUsize, Ordering};
    /// Fails each read of byte `self.1` after the first: the demuxer reads
    /// every VP9 packet once to find the shown frames, and the decode reads it
    /// again.
    struct FailsOnRereadOf<'a>(&'a [u8], u64, AtomicUsize);
    impl ByteSource for FailsOnRereadOf<'_> {
        fn len(&self) -> u64 {
            self.0.len() as u64
        }
        fn read_at(&self, offset: u64, buf: &mut [u8]) -> std::io::Result<()> {
            if (offset..offset + buf.len() as u64).contains(&self.1)
                && self.2.fetch_add(1, Ordering::Relaxed) > 0
            {
                return Err(std::io::Error::other("read failed"));
            }
            self.0.read_at(offset, buf)
        }
    }
    execution::init_pool(2).unwrap();
    // NVDEC decodes it where it can, else libvpx in parts: CUDA_VISIBLE_DEVICES=
    // runs the parts on a GPU host.
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/video_decode/gop_vp9.mp4");
    let bytes = std::fs::read(path).unwrap();
    let info = probe(&bytes).unwrap();
    let all: Vec<u64> = (0..info.frame_count).collect();
    // The last byte of the media data, just before `moov`: in the last frame,
    // an inter frame of the last part.
    let at = bytes.windows(4).position(|w| w == b"moov").unwrap() as u64 - 5;
    let output = info.width as usize * info.height as usize * 3 * all.len();
    // The default room fits every part at once; no room leaves one thread.
    for max_output_bytes in [VideoDecodeLimits::default().max_output_bytes, output] {
        let mut limits = VideoDecodeLimits::default();
        limits.max_output_bytes = max_output_bytes;
        let decode = || {
            let source = FailsOnRereadOf(&bytes, at, AtomicUsize::new(0));
            let frames = decode_frames_from(&source, &all, None, &limits);
            frames.map(drop).unwrap_err().to_string()
        };
        assert_eq!(decode(), execution::in_pool(decode));
    }
}
