// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Per-request video options: frame sampling and pixel budgets.
//!
//! The names follow the OpenAI-compatible request surface that vLLM defines:
//! sampling options (`fps`, `max_frames`, `num_frames`) travel in
//! `media_io_kwargs.video`, pixel budgets (`total_pixels`,
//! `max_pixels_per_frame`) in `mm_processor_kwargs`. A router deserializes
//! each namespace into [`VideoOptions`] and [`VideoPixelBudget`]; the crate
//! turns them into a frame count ([`resolve_num_frames`]) and, per model
//! family, a target frame size (`QwenVlSpec::video_resize`).
//!
//! Nothing here decodes video. It is the arithmetic a decoder and a resizer
//! must agree on, kept pure so every consumer gets the same answer.

use crate::{MmError, Result};

/// Frame-sampling options (`media_io_kwargs.video`).
///
/// Unknown keys are ignored: the namespace is shared with the engine's own
/// loader options (`do_sample_frames`, `size`, `max_pixels`, ...), so a router
/// can deserialize the whole map. A misspelled key is therefore not an error.
///
/// `fps` wins over `num_frames`; `max_frames` caps either. With none set, every
/// frame is requested, still capped by `max_frames`.
///
/// Each field defaults to unset (`None`), which is what vLLM spells `-1`: all
/// frames, at the source frame rate. A literal `-1` in the request is accepted
/// and read as unset, so vLLM-shaped payloads deserialize unchanged.
#[derive(Clone, Debug, Default, PartialEq, serde::Deserialize)]
pub struct VideoOptions {
    /// Sample this many frames per second of video.
    #[serde(default, deserialize_with = "unset_if_minus_one_f64")]
    pub fps: Option<f64>,
    /// Upper bound on the number of sampled frames.
    #[serde(default, deserialize_with = "unset_if_minus_one_u64")]
    pub max_frames: Option<u64>,
    /// Sample exactly this many frames, evenly spaced.
    #[serde(default, deserialize_with = "unset_if_minus_one_u64")]
    pub num_frames: Option<u64>,
}

/// vLLM's `-1` (and JSON `null`) as unset; any other negative count is an error.
fn unset_if_minus_one_u64<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> std::result::Result<Option<u64>, D::Error> {
    use serde::de::Error;
    match <Option<i64> as serde::Deserialize>::deserialize(deserializer)? {
        None | Some(-1) => Ok(None),
        Some(n) => u64::try_from(n)
            .map(Some)
            .map_err(|_| D::Error::custom(format!("expected a frame count or -1, got {n}"))),
    }
}

/// vLLM's `-1` (and JSON `null`) as unset; other values are checked by
/// [`VideoOptions::validate`].
fn unset_if_minus_one_f64<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> std::result::Result<Option<f64>, D::Error> {
    let value = <Option<f64> as serde::Deserialize>::deserialize(deserializer)?;
    Ok(value.filter(|v| *v != -1.0))
}

/// Pixel budgets (`mm_processor_kwargs`), both optional.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Deserialize)]
pub struct VideoPixelBudget {
    /// Budget for the whole clip: all sampled frames share it, so more frames
    /// mean smaller frames. It can only lower the model's clip maximum, never
    /// raise it.
    #[serde(default)]
    pub total_pixels: Option<usize>,
    /// Cap on one frame's pixels. A frame also never gets more than its even
    /// share of the clip budget, and never less than `1.05 * min_pixels`, so
    /// the cap is exceeded when it is below that floor.
    #[serde(default)]
    pub max_pixels_per_frame: Option<usize>,
}

/// The model's clip-level pixel bounds (HF's video `size`:
/// `shortest_edge` / `longest_edge`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Deserialize)]
pub struct VideoBounds {
    /// Lower bound on the clip's pixels; small clips are scaled up to it.
    pub min_pixels: usize,
    /// Upper bound on the clip's pixels; the ceiling for any request budget.
    pub max_pixels: usize,
}

impl VideoOptions {
    /// Reject non-positive or non-finite values before they reach a decoder.
    pub fn validate(&self) -> Result<()> {
        if let Some(fps) = self.fps
            && !(fps.is_finite() && fps > 0.0)
        {
            return Err(MmError::invalid_input(format!(
                "video fps must be a positive finite number, got {fps}"
            )));
        }
        if self.max_frames == Some(0) || self.num_frames == Some(0) {
            return Err(MmError::invalid_input(
                "video max_frames and num_frames must be at least 1",
            ));
        }
        Ok(())
    }
}

impl VideoPixelBudget {
    /// Reject zero budgets.
    pub fn validate(&self) -> Result<()> {
        if self.total_pixels == Some(0) || self.max_pixels_per_frame == Some(0) {
            return Err(MmError::invalid_input(
                "video total_pixels and max_pixels_per_frame must be at least 1",
            ));
        }
        Ok(())
    }
}

/// How many frames to sample from a video of `duration_secs` seconds holding
/// `total_frames` frames.
///
/// Rules, in order: `fps` gives `trunc(duration * fps)`; otherwise
/// `num_frames`; otherwise every frame. The result is capped by `max_frames`
/// and is at least 1. Asking for more frames than the video holds is an error
/// rather than a silent clamp, so a caller notices a mismatched `fps`.
pub fn resolve_num_frames(
    options: &VideoOptions,
    duration_secs: f64,
    total_frames: u64,
) -> Result<u64> {
    options.validate()?;
    if total_frames == 0 {
        return Err(MmError::invalid_input("video has no frames"));
    }
    let requested = match options.fps {
        Some(fps) => {
            if !(duration_secs.is_finite() && duration_secs > 0.0) {
                return Err(MmError::invalid_input(
                    "video duration is required to sample by fps",
                ));
            }
            // Truncation matches the decoder this mirrors; the cast saturates.
            (duration_secs * fps) as u64
        }
        None => options.num_frames.unwrap_or(total_frames),
    };
    let requested = requested
        .min(options.max_frames.unwrap_or(requested))
        .max(1);
    if requested > total_frames {
        return Err(MmError::invalid_input(format!(
            "cannot sample {requested} frames from a video with {total_frames}"
        )));
    }
    Ok(requested)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opts(fps: Option<f64>, max_frames: Option<u64>, num_frames: Option<u64>) -> VideoOptions {
        VideoOptions {
            fps,
            max_frames,
            num_frames,
        }
    }

    #[test]
    fn fps_truncates_duration_times_fps() {
        // 10 s at 2 fps -> 20; 3.9 s at 2 fps -> trunc(7.8) = 7.
        assert_eq!(
            resolve_num_frames(&opts(Some(2.0), None, None), 10.0, 300).unwrap(),
            20
        );
        assert_eq!(
            resolve_num_frames(&opts(Some(2.0), None, None), 3.9, 300).unwrap(),
            7
        );
    }

    #[test]
    fn fps_wins_over_num_frames() {
        let o = opts(Some(1.0), None, Some(50));
        assert_eq!(resolve_num_frames(&o, 10.0, 300).unwrap(), 10);
    }

    #[test]
    fn num_frames_then_all_frames() {
        assert_eq!(
            resolve_num_frames(&opts(None, None, Some(8)), 10.0, 300).unwrap(),
            8
        );
        assert_eq!(
            resolve_num_frames(&opts(None, None, None), 10.0, 300).unwrap(),
            300
        );
    }

    #[test]
    fn max_frames_caps_every_mode() {
        assert_eq!(
            resolve_num_frames(&opts(Some(2.0), Some(5), None), 10.0, 300).unwrap(),
            5
        );
        assert_eq!(
            resolve_num_frames(&opts(None, Some(5), Some(8)), 10.0, 300).unwrap(),
            5
        );
        assert_eq!(
            resolve_num_frames(&opts(None, Some(5), None), 10.0, 300).unwrap(),
            5
        );
    }

    #[test]
    fn at_least_one_frame() {
        // trunc(0.4 s * 1 fps) = 0, raised to 1.
        assert_eq!(
            resolve_num_frames(&opts(Some(1.0), None, None), 0.4, 300).unwrap(),
            1
        );
    }

    #[test]
    fn more_frames_than_the_video_holds_is_an_error() {
        assert!(resolve_num_frames(&opts(Some(60.0), None, None), 10.0, 300).is_err());
        assert!(resolve_num_frames(&opts(None, None, Some(301)), 10.0, 300).is_err());
        // A max_frames cap below the total makes the same request valid.
        assert_eq!(
            resolve_num_frames(&opts(Some(60.0), Some(300), None), 10.0, 300).unwrap(),
            300
        );
    }

    #[test]
    fn rejects_invalid_options() {
        for o in [
            opts(Some(0.0), None, None),
            opts(Some(-1.0), None, None),
            opts(Some(f64::NAN), None, None),
            opts(Some(f64::INFINITY), None, None),
            opts(None, Some(0), None),
            opts(None, None, Some(0)),
        ] {
            assert!(resolve_num_frames(&o, 10.0, 300).is_err(), "{o:?}");
        }
        assert!(resolve_num_frames(&opts(None, None, None), 10.0, 0).is_err());
        assert!(resolve_num_frames(&opts(Some(1.0), None, None), 0.0, 300).is_err());
    }

    #[test]
    fn deserializes_realistic_payloads_ignoring_engine_keys() {
        let o: VideoOptions = serde_json::from_str(
            r#"{"fps": 2, "max_frames": 32, "do_sample_frames": true, "max_duration": 60}"#,
        )
        .unwrap();
        assert_eq!(o, opts(Some(2.0), Some(32), None));
        let b: VideoPixelBudget = serde_json::from_str(
            r#"{"total_pixels": 16777216, "size": {"longest_edge": 1}, "max_pixels": 5}"#,
        )
        .unwrap();
        assert_eq!(b.total_pixels, Some(16_777_216));
        assert_eq!(b.max_pixels_per_frame, None);
    }

    #[test]
    fn vllm_minus_one_means_unset() {
        let o: VideoOptions =
            serde_json::from_str(r#"{"fps": -1, "num_frames": -1, "max_frames": -1}"#).unwrap();
        assert_eq!(o, VideoOptions::default());
        let o: VideoOptions = serde_json::from_str(r#"{"fps": -1.0, "num_frames": null}"#).unwrap();
        assert_eq!(o, VideoOptions::default());
        // Unset everywhere means every frame, as vLLM's -1 does.
        assert_eq!(resolve_num_frames(&o, 10.0, 300).unwrap(), 300);
        // A real value next to a -1 still counts.
        let o: VideoOptions = serde_json::from_str(r#"{"fps": 2, "num_frames": -1}"#).unwrap();
        assert_eq!(o, opts(Some(2.0), None, None));
    }

    #[test]
    fn other_negative_counts_are_errors() {
        for bad in [r#"{"num_frames": -2}"#, r#"{"max_frames": -5}"#] {
            assert!(serde_json::from_str::<VideoOptions>(bad).is_err(), "{bad}");
        }
        // A negative fps other than -1 deserializes but fails validation.
        let o: VideoOptions = serde_json::from_str(r#"{"fps": -2}"#).unwrap();
        assert!(o.validate().is_err());
    }

    #[test]
    fn validate_is_public_and_rejects_zero_budgets() {
        assert!(opts(Some(0.0), None, None).validate().is_err());
        assert!(opts(Some(2.0), Some(8), None).validate().is_ok());
        let zero = VideoPixelBudget {
            total_pixels: Some(0),
            max_pixels_per_frame: None,
        };
        assert!(zero.validate().is_err());
        let zero = VideoPixelBudget {
            total_pixels: None,
            max_pixels_per_frame: Some(0),
        };
        assert!(zero.validate().is_err());
        assert!(VideoPixelBudget::default().validate().is_ok());
    }
}
