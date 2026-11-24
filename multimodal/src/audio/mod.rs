// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Transport-independent, owned PCM and experimental CPU audio features.
//!
//! A host can feed decoded PCM from any importer, including an owned SHM
//! snapshot. This module owns no fetcher, mapping, request lifecycle or threads.
//! Container decoding requires `audio-decode`; PCM/features require only `audio`.

#[cfg(feature = "audio-decode")]
pub mod decode;
pub mod features;
pub mod waveform;

use crate::{MmError, Result, processor::DecodedMedia};

/// Validate the canonical PCM carrier before copying, mixing or extracting.
/// At most stereo is supported: averaging an unspecified surround layout
/// would silently invent a downmix policy. Full scale is nominally [-1, 1],
/// but finite out-of-range samples are preserved, not clipped.
pub fn validate_pcm(samples: &[f32], sample_rate: u32, channels: u16) -> Result<usize> {
    if sample_rate == 0 || channels == 0 {
        return Err(MmError::invalid_input(
            "audio: rate and channels must be positive",
        ));
    }
    if channels > 2 {
        return Err(MmError::unsupported(
            "audio: only mono and stereo PCM are supported",
        ));
    }
    if samples.is_empty() || !samples.len().is_multiple_of(usize::from(channels)) {
        return Err(MmError::invalid_input(
            "audio: PCM must contain complete, nonempty frames",
        ));
    }
    if samples.iter().any(|sample| !sample.is_finite()) {
        return Err(MmError::invalid_input(
            "audio: PCM contains non-finite samples",
        ));
    }
    Ok(samples.len() / usize::from(channels))
}

/// Import explicitly described little-endian f32 PCM. `sample_frames` is
/// checked against the byte length before allocation. The returned carrier
/// owns its data; callers release their transport receipt after the copy.
pub fn from_f32le(
    bytes: &[u8],
    sample_rate: u32,
    channels: u16,
    sample_frames: u64,
    max_frames: usize,
) -> Result<DecodedMedia> {
    if sample_rate == 0 || channels == 0 {
        return Err(MmError::invalid_input(
            "audio: rate and channels must be positive",
        ));
    }
    if channels > 2 {
        return Err(MmError::unsupported(
            "audio: only mono and stereo PCM are supported",
        ));
    }
    let frames = usize::try_from(sample_frames)
        .map_err(|_| MmError::limit_exceeded("audio: frame count exceeds usize"))?;
    if frames > max_frames {
        return Err(MmError::limit_exceeded("audio: PCM frame limit exceeded"));
    }
    let expected = frames
        .checked_mul(usize::from(channels))
        .and_then(|n| n.checked_mul(4))
        .ok_or_else(|| MmError::limit_exceeded("audio: PCM byte count overflow"))?;
    if frames == 0 || expected != bytes.len() {
        return Err(MmError::invalid_input(
            "audio: f32le byte length does not match declared frames/channels",
        ));
    }
    let samples = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect::<Vec<_>>();
    validate_pcm(&samples, sample_rate, channels)?;
    Ok(DecodedMedia::Audio {
        samples,
        sample_rate,
        channels,
    })
}

/// Explicit equal-weight stereo downmix. Mixing in f64 avoids overflowing
/// finite f32 channel values; mono is copied unchanged. No resampling.
pub fn to_mono(samples: &[f32], sample_rate: u32, channels: u16) -> Result<Vec<f32>> {
    validate_pcm(samples, sample_rate, channels)?;
    Ok(if channels == 1 {
        samples.to_vec()
    } else {
        samples
            .chunks_exact(2)
            .map(|s| ((f64::from(s[0]) + f64::from(s[1])) * 0.5) as f32)
            .collect()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pcm_import_checks_shape_limits_and_values() {
        let bytes = [0.5f32, -0.5, 1., 0.]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        let DecodedMedia::Audio {
            samples, channels, ..
        } = from_f32le(&bytes, 16000, 2, 2, 2).unwrap()
        else {
            panic!()
        };
        assert_eq!(to_mono(&samples, 16000, channels).unwrap(), [0., 0.5]);
        assert!(matches!(
            from_f32le(&bytes, 16000, 2, 2, 1),
            Err(MmError::LimitExceeded { .. })
        ));
        assert!(from_f32le(&bytes, 16000, 1, 2, 2).is_err());
        assert!(from_f32le(&f32::NAN.to_le_bytes(), 16000, 1, 1, 1).is_err());
        assert!(from_f32le(&[], 0, 1, 0, 0).is_err());
        assert!(from_f32le(&bytes, 16000, 3, 1, 2).is_err());
    }
}
