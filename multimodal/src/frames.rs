// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Sampled video frames and their JPEG encoding.
//!
//! [`VideoFrames`] is what a video decoder hands over: RGB frames plus the
//! timing a model needs. [`encode_jpeg_frames`] turns them into JPEG images for
//! a router whose backend accepts images but not video: each frame becomes an
//! image part carrying its [`JpegFrame::timestamp_secs`]. How that time is shown
//! to the model (an image-only model has no native sense of time) is the
//! consumer's choice; it differs between model adapters.
//!
//! The router owns transport. It base64-encodes `bytes` into a
//! `data:image/jpeg;base64,...` URL (or uploads them) and builds its own request
//! parts; this crate adds no HTTP or JSON types.

use crate::{MmError, Result};

/// Timing that travels with a set of frames.
#[derive(Clone, Debug)]
pub struct VideoTiming {
    /// Presentation time of each frame in seconds from the start of the video,
    /// one per frame, non-decreasing.
    pub timestamps: Vec<f64>,
    /// Frame rate of the source video.
    pub source_fps: f64,
    /// Duration of the source video in seconds.
    pub source_duration: f64,
}

/// Decoded, sampled frames of one video: `rgb` holds `frame_count` frames of
/// `height * width * 3` bytes, in presentation order.
///
/// Not `Clone`: it owns every decoded frame, so a copy would duplicate them
/// all. Share it with `Arc<VideoFrames>` instead.
#[derive(Debug)]
pub struct VideoFrames {
    width: usize,
    height: usize,
    frame_len: usize,
    rgb: Vec<u8>,
    timing: VideoTiming,
}

impl VideoFrames {
    /// Validate and wrap decoded frames.
    ///
    /// Errors if the buffer is not exactly one frame of `height * width * 3`
    /// bytes per timestamp, if a side is zero, if a timestamp is not finite and
    /// non-negative or decreases, if one lies more than a frame period past
    /// `source_duration` (container durations are rounded, so a small overshoot
    /// is allowed), or if `source_fps` or `source_duration` is not positive and
    /// finite.
    pub fn new(width: usize, height: usize, rgb: Vec<u8>, timing: VideoTiming) -> Result<Self> {
        if width == 0 || height == 0 {
            return Err(MmError::invalid_input("video frames have an empty side"));
        }
        let VideoTiming {
            timestamps,
            source_fps,
            source_duration,
        } = &timing;
        if timestamps.is_empty() {
            return Err(MmError::invalid_input("video has no frames"));
        }
        let frame_len = width
            .checked_mul(height)
            .and_then(|n| n.checked_mul(3))
            .ok_or_else(|| MmError::invalid_input("video frame size overflows"))?;
        if timestamps.len().checked_mul(frame_len) != Some(rgb.len()) {
            return Err(MmError::invalid_input(format!(
                "rgb buffer has {} bytes, expected {} frames of {frame_len}",
                rgb.len(),
                timestamps.len()
            )));
        }
        let positive = |v: f64| v.is_finite() && v > 0.0;
        if !positive(*source_fps) || !positive(*source_duration) {
            return Err(MmError::invalid_input(
                "source_fps and source_duration must be positive and finite",
            ));
        }
        let latest = source_duration + 1.0 / source_fps;
        let mut previous = 0.0;
        for &t in timestamps {
            if !t.is_finite() || t < previous || t > latest {
                return Err(MmError::invalid_input(format!(
                    "frame timestamp {t} is not finite, non-decreasing and within the video"
                )));
            }
            previous = t;
        }
        Ok(Self {
            width,
            height,
            frame_len,
            rgb,
            timing,
        })
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn frame_count(&self) -> usize {
        self.timing.timestamps.len()
    }

    /// Presentation time of each frame, in seconds.
    pub fn timestamps(&self) -> &[f64] {
        &self.timing.timestamps
    }

    pub fn source_fps(&self) -> f64 {
        self.timing.source_fps
    }

    pub fn source_duration(&self) -> f64 {
        self.timing.source_duration
    }

    /// All frames, concatenated: `[T, H, W, 3]` u8.
    pub fn rgb(&self) -> &[u8] {
        &self.rgb
    }

    /// One frame's `height * width * 3` bytes, or `None` past the last frame.
    pub fn frame(&self, index: usize) -> Option<&[u8]> {
        // `new` checked that `frame_count * frame_len` fits the buffer, so
        // this cannot overflow once `index` is in range.
        (index < self.frame_count())
            .then(|| &self.rgb[index * self.frame_len..(index + 1) * self.frame_len])
    }
}

/// One frame encoded as a JPEG, with its place in the video.
#[derive(Debug, PartialEq)]
#[non_exhaustive]
pub struct JpegFrame {
    pub bytes: Vec<u8>,
    /// Presentation time in seconds from the start of the video.
    pub timestamp_secs: f64,
}

/// JPEG quality used when a caller has no preference.
pub const DEFAULT_JPEG_QUALITY: u8 = 85;

/// libjpeg-turbo's largest supported image side, in pixels.
pub const MAX_JPEG_DIMENSION: usize = 65_500;

/// Encode every frame as a sequential (non-progressive) JPEG at `quality`
/// (1..=100) with 4:2:0 chroma subsampling.
///
/// This is CPU-bound and blocking: a router running it on an async runtime
/// should call it from a blocking thread. Peak memory is the decoded frames
/// plus every encoded JPEG, since all frames are returned at once.
pub fn encode_jpeg_frames(frames: &VideoFrames, quality: u8) -> Result<Vec<JpegFrame>> {
    if !(1..=100).contains(&quality) {
        return Err(MmError::invalid_input(format!(
            "jpeg quality must be in 1..=100, got {quality}"
        )));
    }
    if frames.width > MAX_JPEG_DIMENSION || frames.height > MAX_JPEG_DIMENSION {
        return Err(MmError::limit_exceeded(format!(
            "{}x{} frames exceed the {MAX_JPEG_DIMENSION} pixel jpeg limit",
            frames.width, frames.height
        )));
    }
    let mut compressor = turbojpeg::Compressor::new()
        .map_err(|e| MmError::internal_with_source("jpeg compressor init failed", e))?;
    let configure = |r: turbojpeg::Result<()>| {
        r.map_err(|e| MmError::internal_with_source("jpeg compressor setting rejected", e))
    };
    configure(compressor.set_quality(i32::from(quality)))?;
    configure(compressor.set_subsamp(turbojpeg::Subsamp::Sub2x2))?;
    configure(compressor.set_progressive(false))?;
    frames
        .rgb
        .chunks_exact(frames.frame_len)
        .zip(&frames.timing.timestamps)
        .map(|(pixels, &timestamp_secs)| {
            let image = turbojpeg::Image {
                pixels,
                width: frames.width,
                pitch: frames.width * 3,
                height: frames.height,
                format: turbojpeg::PixelFormat::RGB,
            };
            let bytes = compressor
                .compress_to_vec(image)
                .map_err(|e| MmError::internal_with_source("jpeg encode failed", e))?;
            Ok(JpegFrame {
                bytes,
                timestamp_secs,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::decode::{DecodeLimits, decode_rgb};

    fn timing(timestamps: Vec<f64>, duration: f64) -> VideoTiming {
        VideoTiming {
            timestamps,
            source_fps: 30.0,
            source_duration: duration,
        }
    }

    /// `n` frames of a gradient with three different channels, shifted by 40
    /// per frame, so a swapped channel or frame shows up in the error.
    fn gradient(n: usize, w: usize, h: usize) -> VideoFrames {
        let mut rgb = Vec::new();
        for f in 0..n {
            for _y in 0..h {
                for x in 0..w {
                    let v = ((x * 255 / (w - 1).max(1)) + f * 40) as u8;
                    rgb.extend_from_slice(&[v, (255 - v) / 2, 40 + (f as u8) * 30]);
                }
            }
        }
        let timestamps = (0..n).map(|i| i as f64 * 0.5).collect();
        VideoFrames::new(w, h, rgb, timing(timestamps, n as f64 * 0.5)).unwrap()
    }

    #[test]
    fn round_trips_every_frame_per_channel() {
        let frames = gradient(3, 64, 48);
        let jpegs = encode_jpeg_frames(&frames, 95).unwrap();
        assert_eq!(jpegs.len(), 3);
        for (i, jpeg) in jpegs.iter().enumerate() {
            assert_eq!(&jpeg.bytes[..2], &[0xFF, 0xD8], "JPEG SOI marker");
            let (rgb, h, w) = decode_rgb(&jpeg.bytes, &DecodeLimits::default()).unwrap();
            assert_eq!((h, w), (48, 64));
            let src = frames.frame(i).unwrap();
            for c in 0..3 {
                let sum: u64 = rgb
                    .iter()
                    .zip(src)
                    .skip(c)
                    .step_by(3)
                    .map(|(a, b)| u64::from(a.abs_diff(*b)))
                    .sum();
                let mean = sum / (rgb.len() as u64 / 3);
                assert!(mean < 8, "frame {i} channel {c}: mean error {mean}");
            }
        }
    }

    #[test]
    fn output_is_sequential_420() {
        let jpegs = encode_jpeg_frames(&gradient(1, 32, 32), 80).unwrap();
        let header = turbojpeg::read_header(&jpegs[0].bytes).unwrap();
        assert_eq!(header.subsamp, turbojpeg::Subsamp::Sub2x2);
        assert!(!header.is_progressive);
    }

    #[test]
    fn timestamps_travel_with_the_frames() {
        let jpegs = encode_jpeg_frames(&gradient(3, 16, 16), 80).unwrap();
        let t: Vec<f64> = jpegs.iter().map(|j| j.timestamp_secs).collect();
        assert_eq!(t, vec![0.0, 0.5, 1.0]);
    }

    #[test]
    fn lower_quality_is_smaller() {
        let frames = gradient(1, 128, 96);
        let high = encode_jpeg_frames(&frames, 95).unwrap();
        let low = encode_jpeg_frames(&frames, 10).unwrap();
        assert!(low[0].bytes.len() < high[0].bytes.len());
    }

    #[test]
    fn rejects_quality_outside_1_to_100() {
        let frames = gradient(1, 16, 16);
        assert!(encode_jpeg_frames(&frames, 0).is_err());
        assert!(encode_jpeg_frames(&frames, 101).is_err());
        assert!(encode_jpeg_frames(&frames, 1).is_ok());
        assert!(encode_jpeg_frames(&frames, 100).is_ok());
    }

    #[test]
    fn single_pixel_frame_encodes() {
        let frames = VideoFrames::new(1, 1, vec![10, 20, 30], timing(vec![0.0], 1.0)).unwrap();
        assert_eq!(encode_jpeg_frames(&frames, 90).unwrap().len(), 1);
    }

    #[test]
    fn frames_wider_than_libjpeg_allows_are_a_limit_error() {
        let w = MAX_JPEG_DIMENSION + 1;
        let frames = VideoFrames::new(w, 1, vec![0; w * 3], timing(vec![0.0], 1.0)).unwrap();
        let err = encode_jpeg_frames(&frames, 80).unwrap_err();
        assert!(matches!(err, MmError::LimitExceeded { .. }), "{err:?}");
    }

    #[test]
    fn rejects_malformed_frames() {
        let ok = |rgb: Vec<u8>, ts: Vec<f64>| VideoFrames::new(2, 2, rgb, timing(ts, 1.0));
        assert!(ok(vec![0; 12], vec![0.0]).is_ok());
        assert!(ok(vec![0; 11], vec![0.0]).is_err(), "short buffer");
        assert!(ok(vec![0; 24], vec![0.0]).is_err(), "frame count mismatch");
        assert!(ok(vec![], vec![]).is_err(), "no frames");
        assert!(ok(vec![0; 12], vec![f64::NAN]).is_err());
        assert!(ok(vec![0; 12], vec![-1.0]).is_err());
        assert!(VideoFrames::new(0, 2, vec![], timing(vec![0.0], 1.0)).is_err());
        assert!(VideoFrames::new(usize::MAX, 2, vec![], timing(vec![0.0], 1.0)).is_err());
        let bad = |fps: f64, dur: f64| VideoTiming {
            timestamps: vec![0.0],
            source_fps: fps,
            source_duration: dur,
        };
        assert!(VideoFrames::new(2, 2, vec![0; 12], bad(0.0, 1.0)).is_err());
        assert!(VideoFrames::new(2, 2, vec![0; 12], bad(30.0, f64::INFINITY)).is_err());
    }

    #[test]
    fn timestamps_must_be_ordered_and_inside_the_video() {
        let two = |a: f64, b: f64| VideoFrames::new(1, 1, vec![0; 6], timing(vec![a, b], 1.0));
        assert!(two(0.2, 0.8).is_ok());
        assert!(two(0.5, 0.5).is_ok(), "equal timestamps are not decreasing");
        assert!(two(0.8, 0.2).is_err(), "decreasing");
        // One frame period (1/30 s) of overshoot is allowed, not 100 s.
        assert!(two(0.0, 1.03).is_ok());
        assert!(two(0.0, 100.0).is_err());
    }

    #[test]
    fn frame_accessor_is_total() {
        let frames = gradient(2, 4, 4);
        assert_eq!(frames.frame(0).unwrap().len(), 48);
        assert!(frames.frame(1).is_some());
        assert!(frames.frame(2).is_none());
        // Indices whose byte offset would overflow must not panic or wrap.
        assert!(frames.frame(usize::MAX).is_none());
        assert!(frames.frame(1usize << (usize::BITS - 4)).is_none());
        assert_eq!(frames.rgb().len(), 96);
        assert_eq!(frames.frame_count(), 2);
        assert_eq!((frames.width(), frames.height()), (4, 4));
        assert_eq!(frames.timestamps(), &[0.0, 0.5]);
        assert_eq!(frames.source_fps(), 30.0);
        assert_eq!(frames.source_duration(), 1.0);
    }
}
