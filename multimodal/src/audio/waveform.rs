// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Explicit channel/rate/length policy, independent of any model or transport.

use crate::{MmError, Result};
use rubato::{FftFixedInOut, Resampler};

/// CPU work and allocation bounds, also applied to already-decoded PCM.
#[derive(Clone, Copy, Debug)]
pub struct WaveformLimits {
    pub max_source_frames: usize,
    pub max_output_samples: usize,
}

impl Default for WaveformLimits {
    fn default() -> Self {
        Self {
            max_source_frames: 192_000 * 60,
            max_output_samples: 192_000 * 60,
        }
    }
}

/// Resampling is never implicit. Rubato's FFT backend has its own numerical
/// contract; it is not a bit-exact replacement for librosa/soxr or torchaudio.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResamplePolicy {
    Reject,
    RubatoFft,
}

/// Equal-weight mono/stereo conversion, followed by optional resampling,
/// truncation, then zero padding. Lengths count mono samples at the target
/// rate. `None` leaves that stage inactive; no model window is assumed.
#[derive(Clone, Copy, Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WaveformConfig {
    pub sample_rate: u32,
    pub resample: ResamplePolicy,
    pub max_samples: Option<usize>,
    pub pad_to_samples: Option<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AudioPlan {
    pub source_frames: usize,
    pub resampled_samples: usize,
    /// Samples retained after truncation, excluding right padding.
    pub valid_samples: usize,
    pub output_samples: usize,
    pub truncated: bool,
}

pub struct PreparedAudio {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub plan: AudioPlan,
}

/// Metadata-only planning. The same integer length rules drive processing.
/// Output length after resampling is `ceil(source_frames * dst_rate/src_rate)`;
/// filter latency is removed, and flushing never adds model-visible samples.
pub fn plan(
    source_frames: usize,
    source_rate: u32,
    config: &WaveformConfig,
    limits: &WaveformLimits,
) -> Result<AudioPlan> {
    if source_frames == 0 || source_rate == 0 || config.sample_rate == 0 {
        return Err(MmError::invalid_input(
            "audio: frame count and rates must be positive",
        ));
    }
    // Keep fixed-size FFT resampler scratch bounded even for coprime rates.
    if source_rate > 384_000 || config.sample_rate > 384_000 {
        return Err(MmError::unsupported(
            "audio: sample rates above 384 kHz are unsupported",
        ));
    }
    if source_frames > limits.max_source_frames {
        return Err(MmError::limit_exceeded(
            "audio: source frame limit exceeded",
        ));
    }
    if source_rate != config.sample_rate && config.resample == ResamplePolicy::Reject {
        return Err(MmError::unsupported(
            "audio: sample rate mismatch; select a resampling backend explicitly",
        ));
    }
    if config.max_samples == Some(0)
        || config.pad_to_samples == Some(0)
        || matches!((config.max_samples, config.pad_to_samples), (Some(max), Some(pad)) if pad > max)
    {
        return Err(MmError::invalid_input(
            "audio: length bounds must be positive and padding must not exceed truncation limit",
        ));
    }
    let resampled_samples = usize::try_from(
        (source_frames as u128 * u128::from(config.sample_rate)).div_ceil(u128::from(source_rate)),
    )
    .map_err(|_| MmError::limit_exceeded("audio: resampled length overflow"))?;
    // Bound intermediates, not just the model-visible truncated prefix.
    if resampled_samples > limits.max_output_samples {
        return Err(MmError::limit_exceeded(
            "audio: resampled sample limit exceeded",
        ));
    }
    let valid_samples = resampled_samples.min(config.max_samples.unwrap_or(usize::MAX));
    let output_samples = valid_samples.max(config.pad_to_samples.unwrap_or(0));
    if output_samples > limits.max_output_samples {
        return Err(MmError::limit_exceeded(
            "audio: padded sample limit exceeded",
        ));
    }
    Ok(AudioPlan {
        source_frames,
        resampled_samples,
        valid_samples,
        output_samples,
        truncated: valid_samples < resampled_samples,
    })
}

pub fn prepare(
    samples: &[f32],
    source_rate: u32,
    channels: u16,
    config: &WaveformConfig,
    limits: &WaveformLimits,
) -> Result<PreparedAudio> {
    // Check bounds before a full PCM scan or copy.
    let frames = samples
        .len()
        .checked_div(usize::from(channels))
        .ok_or_else(|| MmError::invalid_input("audio: zero channels"))?;
    let plan = plan(frames, source_rate, config, limits)?;
    let mono = super::to_mono(samples, source_rate, channels)?;
    let mut output = if source_rate == config.sample_rate {
        mono
    } else {
        resample(
            &mono,
            source_rate,
            config.sample_rate,
            plan.resampled_samples,
        )?
    };
    output.truncate(plan.valid_samples);
    output.resize(plan.output_samples, 0.0);
    Ok(PreparedAudio {
        samples: output,
        sample_rate: config.sample_rate,
        plan,
    })
}

fn resample(mono: &[f32], source_rate: u32, target_rate: u32, expected: usize) -> Result<Vec<f32>> {
    // A fixed input block of 1024 at e.g. 1 Hz -> 384 kHz would allocate
    // hundreds of millions of output samples. Bound the requested output
    // block too; rubato rounds to a multiple of the reduced rate ratio.
    let chunk_size =
        (4096u64 * u64::from(source_rate) / u64::from(target_rate)).clamp(1, 1024) as usize;
    let mut resampler =
        FftFixedInOut::<f32>::new(source_rate as usize, target_rate as usize, chunk_size, 1)
            .map_err(|e| {
                MmError::invalid_input_with_source("audio: cannot construct FFT resampler", e)
            })?;
    let chunk = resampler.input_frames_next();
    let delay = resampler.output_delay();
    let needed = expected
        .checked_add(delay)
        .ok_or_else(|| MmError::limit_exceeded("audio: resampler length overflow"))?;
    let mut collected = Vec::new();
    let capacity = needed
        .checked_add(resampler.output_frames_next())
        .ok_or_else(|| MmError::limit_exceeded("audio: resampler capacity overflow"))?;
    collected
        .try_reserve(capacity)
        .map_err(|e| MmError::internal_with_source("audio: cannot allocate resampler output", e))?;
    let mut input = vec![0.0; chunk];
    let mut consumed = 0;
    while collected.len() < needed {
        input.fill(0.0);
        let count = chunk.min(mono.len() - consumed);
        input[..count].copy_from_slice(&mono[consumed..consumed + count]);
        consumed += count;
        let output = resampler
            .process(&[&input], None)
            .map_err(|e| MmError::internal_with_source("audio: FFT resampling failed", e))?;
        collected.extend_from_slice(&output[0]);
    }
    let output = collected[delay..needed].to_vec();
    if output.iter().any(|sample| !sample.is_finite()) {
        return Err(MmError::invalid_input("audio: non-finite resampler output"));
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn config(rate: u32) -> WaveformConfig {
        WaveformConfig {
            sample_rate: rate,
            resample: ResamplePolicy::RubatoFft,
            max_samples: None,
            pad_to_samples: None,
        }
    }

    #[test]
    fn rate_length_and_padding_match_metadata_plan() {
        let limits = WaveformLimits::default();
        for (src, dst) in [
            (48000, 16000),
            (44100, 16000),
            (8000, 24000),
            (16000, 16000),
        ] {
            for n in [1, 31, 1024, 10003] {
                let samples = vec![0.0; n * 2];
                let output = prepare(&samples, src, 2, &config(dst), &limits).unwrap();
                assert_eq!(output.plan, plan(n, src, &config(dst), &limits).unwrap());
                assert_eq!(
                    output.samples.len(),
                    (n as u64 * u64::from(dst)).div_ceil(u64::from(src)) as usize
                );
                assert!(output.samples.iter().all(|&x| x == 0.0));
            }
        }
        let cfg = WaveformConfig {
            max_samples: Some(8),
            pad_to_samples: Some(8),
            ..config(16000)
        };
        let padded = prepare(&[0.5; 3], 16000, 1, &cfg, &limits).unwrap();
        assert_eq!(padded.samples, [0.5, 0.5, 0.5, 0., 0., 0., 0., 0.]);
        assert_eq!(padded.plan.valid_samples, 3);
        let truncated = prepare(&[0.5; 10], 16000, 1, &cfg, &limits).unwrap();
        assert!(truncated.plan.truncated);
        assert_eq!(truncated.samples, [0.5; 8]);
    }

    #[test]
    fn resampling_preserves_timing_and_rejects_aliasing() {
        let rate = 48000;
        let tone = |hz: f64| {
            (0..rate)
                .map(|i| (std::f64::consts::TAU * hz * f64::from(i) / f64::from(rate)).sin() as f32)
                .collect::<Vec<_>>()
        };
        let low = prepare(
            &tone(1000.),
            rate,
            1,
            &config(16000),
            &WaveformLimits::default(),
        )
        .unwrap();
        let high = prepare(
            &tone(12000.),
            rate,
            1,
            &config(16000),
            &WaveformLimits::default(),
        )
        .unwrap();
        let rms = |s: &[f32]| {
            (s[1000..15000]
                .iter()
                .map(|&x| f64::from(x).powi(2))
                .sum::<f64>()
                / 14000.)
                .sqrt()
        };
        assert!((rms(&low.samples) - 0.5f64.sqrt()).abs() < 0.002);
        assert!(rms(&high.samples) < 0.001);
        assert!(
            (low.samples[4004] - (std::f64::consts::TAU * 1000. * 4004. / 16000.).sin() as f32)
                .abs()
                < 0.005
        );
    }

    #[test]
    fn rejects_mismatch_and_budgets_before_work() {
        let limits = WaveformLimits {
            max_source_frames: 100,
            max_output_samples: 200,
        };
        assert!(plan(101, 16000, &config(16000), &limits).is_err());
        assert!(plan(100, 8000, &config(48000), &limits).is_err());
        let mut cfg = config(16000);
        cfg.resample = ResamplePolicy::Reject;
        assert!(matches!(
            plan(1, 48000, &cfg, &limits),
            Err(MmError::Unsupported { .. })
        ));
        assert!(prepare(&[0.], 16000, 0, &cfg, &limits).is_err());
    }
}
