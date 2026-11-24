// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Mel scales/filterbank and NumPy-compatible FFT rounding follow transformers
// 5.12.1 audio_utils.py (Hugging Face, Apache-2.0).

//! Configurable CPU STFT/power/Slaney-or-HTK log-mel extraction. There is no
//! model window, tokenizer, convolution length or scheduler policy here.
//! Floating-point output is experimental, not qualified bit-exact to HF/torch.

use crate::{MmError, Result};
use rustfft::{Fft, FftPlanner, num_complex::Complex};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CenterPadding {
    None,
    Reflect,
    Zero,
}

#[derive(Clone, Copy, Debug, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WindowFunction {
    HannPeriodic,
    HannSymmetric,
    Rectangular,
}

#[derive(Clone, Copy, Debug, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MelScale {
    Slaney,
    Htk,
}

/// HF's NumPy spectrogram stores the f64 FFT result as complex64 before
/// power/mel arithmetic. Other consumers may keep the full f64 result.
#[derive(Clone, Copy, Debug, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FftOutputPrecision {
    F32,
    F64,
}

/// All numerical policy is explicit. Extraction accepts an already-prepared
/// waveform; padding/truncation/resampling are separate stages in `waveform`.
/// Power is magnitude squared; the logarithm is base 10. Dynamic range is in
/// log10 units; the final transform is `(log_mel + output_offset)*output_scale`.
#[derive(Clone, Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LogMelConfig {
    pub sample_rate: u32,
    pub n_fft: usize,
    pub hop_length: usize,
    pub n_mels: usize,
    pub min_frequency: f64,
    pub max_frequency: f64,
    pub mel_scale: MelScale,
    pub slaney_normalize: bool,
    pub window: WindowFunction,
    pub center: CenterPadding,
    pub fft_output: FftOutputPrecision,
    pub drop_last_frame: bool,
    pub log_floor: f64,
    pub dynamic_range: Option<f32>,
    pub output_offset: f32,
    pub output_scale: f32,
}

/// Validated numerical policy and filterbank. Planning with it needs neither
/// PCM nor an FFT plan.
pub struct ValidatedLogMelConfig {
    config: LogMelConfig,
    filters: Vec<Vec<(usize, f64)>>,
}

impl ValidatedLogMelConfig {
    /// Plan frame geometry using the same validated policy as extraction.
    pub fn plan(
        &self,
        samples: usize,
        valid_samples: usize,
        limits: &FeatureLimits,
    ) -> Result<FeaturePlan> {
        self.config.plan_validated(samples, valid_samples, limits)
    }
}

/// Bounds checked before constructing a plan, allocating or running the FFT.
#[derive(Clone, Copy, Debug)]
pub struct FeatureLimits {
    pub max_samples: usize,
    pub max_frames: usize,
    pub max_values: usize,
}

impl Default for FeatureLimits {
    fn default() -> Self {
        Self {
            max_samples: 192_000 * 60,
            max_frames: 100_000,
            max_values: 16_000_000,
        }
    }
}

/// Pure feature geometry; masks here refer to frame *anchors* (`i*hop`),
/// not to full-window validity or a model encoder's convolution mask.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FeaturePlan {
    pub num_frames: usize,
    pub valid_frames: usize,
    pub num_mels: usize,
}

impl FeaturePlan {
    pub fn frame_anchor_mask(&self) -> Vec<i64> {
        (0..self.num_frames)
            .map(|i| i64::from(i < self.valid_frames))
            .collect()
    }
}

pub struct LogMelFeatures {
    /// Row-major `[num_mels, num_frames]`, without an implicit batch axis.
    pub data: Vec<f32>,
    pub plan: FeaturePlan,
}

/// Immutable FFT plan/window/filterbank, shareable between requests. Scratch
/// is local to an invocation; the extractor reads no environment or files.
pub struct LogMelExtractor {
    validated: ValidatedLogMelConfig,
    fft: Arc<dyn Fft<f64>>,
    window: Vec<f64>,
}

impl LogMelConfig {
    /// Validate settings and construct the mel filterbank without planning an
    /// FFT. A router and engine can both validate when resolving their policy.
    pub fn validate(self) -> Result<ValidatedLogMelConfig> {
        self.validate_parameters()?;
        let filters = build_mel_filters(&self)?;
        Ok(ValidatedLogMelConfig {
            config: self,
            filters,
        })
    }

    fn validate_parameters(&self) -> Result<()> {
        if self.sample_rate == 0
            || self.n_fft < 2
            || self.n_fft > 8192
            || self.hop_length == 0
            || self.n_mels == 0
            || self.n_mels > 512
        {
            return Err(MmError::invalid_input(
                "audio: invalid rate, FFT size (2..8192), hop or mel count (1..512)",
            ));
        }
        if !self.min_frequency.is_finite()
            || !self.max_frequency.is_finite()
            || self.min_frequency < 0.0
            || self.min_frequency >= self.max_frequency
            || self.max_frequency > f64::from(self.sample_rate) / 2.0
            || !self.log_floor.is_finite()
            || self.log_floor <= 0.0
            || !self.output_offset.is_finite()
            || !self.output_scale.is_finite()
            || self
                .dynamic_range
                .is_some_and(|r| !r.is_finite() || r <= 0.0)
        {
            return Err(MmError::invalid_input(
                "audio: invalid mel frequency, logarithm or output transform settings",
            ));
        }
        Ok(())
    }

    /// The same frame geometry used by extraction, available without samples.
    /// `valid_samples` excludes right-padding and must not exceed `samples`.
    /// For repeated planning, validate once and reuse [`ValidatedLogMelConfig`].
    pub fn plan(
        &self,
        samples: usize,
        valid_samples: usize,
        limits: &FeatureLimits,
    ) -> Result<FeaturePlan> {
        self.clone()
            .validate()?
            .plan(samples, valid_samples, limits)
    }

    fn plan_validated(
        &self,
        samples: usize,
        valid_samples: usize,
        limits: &FeatureLimits,
    ) -> Result<FeaturePlan> {
        if samples == 0 || valid_samples == 0 || valid_samples > samples {
            return Err(MmError::invalid_input(
                "audio: invalid waveform/valid sample counts",
            ));
        }
        if samples > limits.max_samples {
            return Err(MmError::limit_exceeded(
                "audio: feature sample limit exceeded",
            ));
        }
        let padding = match self.center {
            CenterPadding::None => 0,
            _ => 2 * (self.n_fft / 2),
        };
        let padded = samples
            .checked_add(padding)
            .ok_or_else(|| MmError::limit_exceeded("audio: centered sample count overflow"))?;
        let mut frames = padded
            .checked_sub(self.n_fft)
            .map(|n| n / self.hop_length + 1)
            .ok_or_else(|| {
                MmError::invalid_input("audio: waveform shorter than uncentered FFT window")
            })?;
        if self.drop_last_frame {
            frames -= 1;
        }
        if frames == 0 {
            return Err(MmError::invalid_input(
                "audio: feature policy produces no frames",
            ));
        }
        if frames > limits.max_frames
            || frames
                .checked_mul(self.n_mels)
                .is_none_or(|n| n > limits.max_values)
        {
            return Err(MmError::limit_exceeded(
                "audio: feature frame/value limit exceeded",
            ));
        }
        Ok(FeaturePlan {
            num_frames: frames,
            valid_frames: frames.min(valid_samples.div_ceil(self.hop_length)),
            num_mels: self.n_mels,
        })
    }
}

fn hz_to_mel(hz: f64, scale: MelScale) -> f64 {
    match scale {
        MelScale::Htk => 2595.0 * (1.0 + hz / 700.0).log10(),
        MelScale::Slaney if hz < 1000.0 => hz / (200.0 / 3.0),
        MelScale::Slaney => 15.0 + (hz / 1000.0).ln() / (6.4f64.ln() / 27.0),
    }
}

fn mel_to_hz(mel: f64, scale: MelScale) -> f64 {
    match scale {
        MelScale::Htk => 700.0 * (10.0f64.powf(mel / 2595.0) - 1.0),
        MelScale::Slaney if mel < 15.0 => mel * (200.0 / 3.0),
        MelScale::Slaney => 1000.0 * ((6.4f64.ln() / 27.0) * (mel - 15.0)).exp(),
    }
}

fn build_mel_filters(config: &LogMelConfig) -> Result<Vec<Vec<(usize, f64)>>> {
    let mel_min = hz_to_mel(config.min_frequency, config.mel_scale);
    let mel_max = hz_to_mel(config.max_frequency, config.mel_scale);
    let edges = (0..config.n_mels + 2)
        .map(|i| {
            mel_to_hz(
                mel_min + i as f64 * ((mel_max - mel_min) / (config.n_mels + 1) as f64),
                config.mel_scale,
            )
        })
        .collect::<Vec<_>>();
    let filters = (0..config.n_mels)
        .map(|m| {
            let (left, center, right) = (edges[m], edges[m + 1], edges[m + 2]);
            (0..config.n_fft / 2 + 1)
                .filter_map(|bin| {
                    let hz = bin as f64 * f64::from(config.sample_rate) / config.n_fft as f64;
                    let mut weight = ((hz - left) / (center - left))
                        .min((right - hz) / (right - center))
                        .max(0.0);
                    if config.slaney_normalize {
                        weight *= 2.0 / (right - left);
                    }
                    (weight > 0.0).then_some((bin, weight))
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    if filters.iter().any(Vec::is_empty) {
        return Err(MmError::invalid_input(
            "audio: mel filter has no FFT bins; reduce mel count or increase FFT size",
        ));
    }
    Ok(filters)
}

impl LogMelExtractor {
    pub fn new(config: LogMelConfig) -> Result<Self> {
        Ok(Self::from_validated(config.validate()?))
    }

    /// Build the FFT and window once from a previously validated policy.
    pub fn from_validated(validated: ValidatedLogMelConfig) -> Self {
        let config = &validated.config;
        let fft = FftPlanner::<f64>::new().plan_fft_forward(config.n_fft);
        let denominator = match config.window {
            WindowFunction::HannSymmetric => config.n_fft - 1,
            _ => config.n_fft,
        };
        let window = (0..config.n_fft)
            .map(|n| match config.window {
                WindowFunction::Rectangular => 1.0,
                _ => 0.5 - 0.5 * (std::f64::consts::TAU * n as f64 / denominator as f64).cos(),
            })
            .collect();
        Self {
            validated,
            fft,
            window,
        }
    }

    pub fn config(&self) -> &LogMelConfig {
        &self.validated.config
    }

    pub fn extract(
        &self,
        mono: &[f32],
        sample_rate: u32,
        valid_samples: usize,
        limits: &FeatureLimits,
    ) -> Result<LogMelFeatures> {
        let plan = self.validated.plan(mono.len(), valid_samples, limits)?;
        let cfg = self.config();
        if sample_rate != cfg.sample_rate {
            return Err(MmError::invalid_input(
                "audio: feature input sample rate does not match config",
            ));
        }
        super::validate_pcm(mono, sample_rate, 1)?;
        let mut buffer = vec![Complex::default(); cfg.n_fft];
        let mut scratch = vec![Complex::default(); self.fft.get_inplace_scratch_len()];
        let mut power = vec![0.0; cfg.n_fft / 2 + 1];
        let mut features = vec![0.0f32; cfg.n_mels * plan.num_frames];
        let left = match cfg.center {
            CenterPadding::None => 0,
            _ => cfg.n_fft / 2,
        };
        for frame in 0..plan.num_frames {
            for (i, (sample, &window)) in buffer.iter_mut().zip(&self.window).enumerate() {
                let index = (frame * cfg.hop_length + i) as i128 - left as i128;
                let value = if index >= 0 && index < mono.len() as i128 {
                    mono[index as usize]
                } else {
                    match cfg.center {
                        CenterPadding::Reflect if mono.len() > 1 => {
                            let period = 2 * (mono.len() as i128 - 1);
                            let folded = index.rem_euclid(period);
                            mono[folded.min(period - folded) as usize]
                        }
                        CenterPadding::Reflect => mono[0],
                        _ => 0.0,
                    }
                };
                *sample = Complex::new(f64::from(value) * window, 0.0);
            }
            self.fft.process_with_scratch(&mut buffer, &mut scratch);
            for (dst, value) in power.iter_mut().zip(&buffer) {
                let (re, im) = match cfg.fft_output {
                    FftOutputPrecision::F32 => {
                        (f64::from(value.re as f32), f64::from(value.im as f32))
                    }
                    FftOutputPrecision::F64 => (value.re, value.im),
                };
                *dst = re.hypot(im).powi(2);
            }
            for (mel, weights) in self.validated.filters.iter().enumerate() {
                let energy: f64 = weights
                    .iter()
                    .map(|&(bin, weight)| power[bin] * weight)
                    .sum();
                features[mel * plan.num_frames + frame] = energy.max(cfg.log_floor).log10() as f32;
            }
        }
        let max = features.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        for feature in &mut features {
            if let Some(range) = cfg.dynamic_range {
                *feature = feature.max(max - range);
            }
            *feature = (*feature + cfg.output_offset) * cfg.output_scale;
        }
        if features.iter().any(|v| !v.is_finite()) {
            return Err(MmError::invalid_input("audio: non-finite log-mel output"));
        }
        Ok(LogMelFeatures {
            data: features,
            plan,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    pub fn config() -> LogMelConfig {
        LogMelConfig {
            sample_rate: 16000,
            n_fft: 400,
            hop_length: 160,
            n_mels: 80,
            min_frequency: 0.,
            max_frequency: 8000.,
            mel_scale: MelScale::Slaney,
            slaney_normalize: true,
            window: WindowFunction::HannPeriodic,
            center: CenterPadding::Reflect,
            fft_output: FftOutputPrecision::F32,
            drop_last_frame: true,
            log_floor: 1e-10,
            dynamic_range: Some(8.),
            output_offset: 4.,
            output_scale: 0.25,
        }
    }

    #[test]
    fn silence_floor_and_frame_anchor_mask() {
        let extractor = LogMelExtractor::new(config()).unwrap();
        let output = extractor
            .extract(&[0.; 1600], 16000, 321, &FeatureLimits::default())
            .unwrap();
        assert_eq!(
            output.plan,
            FeaturePlan {
                num_frames: 10,
                valid_frames: 3,
                num_mels: 80
            }
        );
        assert_eq!(
            output.plan.frame_anchor_mask(),
            [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
        );
        assert!(output.data.iter().all(|&v| v == -1.5));
    }

    #[test]
    fn policies_change_geometry_without_model_assumptions() {
        let mut cfg = config();
        cfg.n_mels = 32;
        cfg.n_fft = 256;
        cfg.hop_length = 64;
        cfg.drop_last_frame = false;
        cfg.center = CenterPadding::None;
        let output = LogMelExtractor::new(cfg.clone())
            .unwrap()
            .extract(&[0.; 1024], 16000, 1024, &FeatureLimits::default())
            .unwrap();
        assert_eq!(output.plan.num_frames, 13);
        cfg.center = CenterPadding::Zero;
        assert_eq!(
            cfg.plan(1024, 1024, &FeatureLimits::default())
                .unwrap()
                .num_frames,
            17
        );
        assert!(cfg.plan(1024, 1025, &FeatureLimits::default()).is_err());
    }

    #[test]
    fn rejects_config_rate_and_allocation_limits() {
        let extractor = LogMelExtractor::new(config()).unwrap();
        assert!(
            extractor
                .extract(&[0.; 1600], 8000, 1600, &FeatureLimits::default())
                .is_err()
        );
        let limits = FeatureLimits {
            max_values: 1,
            ..FeatureLimits::default()
        };
        assert!(matches!(
            extractor.extract(&[0.; 1600], 16000, 1600, &limits),
            Err(MmError::LimitExceeded { .. })
        ));
        for value in [0., -1., f64::NAN, f64::INFINITY] {
            let mut cfg = config();
            cfg.log_floor = value;
            assert!(LogMelExtractor::new(cfg).is_err());
        }
    }

    #[test]
    fn planning_and_extraction_reject_empty_mel_filters() {
        let config = LogMelConfig {
            n_fft: 32,
            hop_length: 16,
            ..config()
        };
        assert!(matches!(
            config.plan(16000, 16000, &FeatureLimits::default()),
            Err(MmError::InvalidInput { .. })
        ));
        assert!(matches!(
            LogMelExtractor::new(config),
            Err(MmError::InvalidInput { .. })
        ));
    }

    #[test]
    fn odd_fft_single_sample_reflection_and_f64_are_well_defined() {
        let mut cfg = config();
        cfg.n_fft = 255;
        cfg.n_mels = 16;
        cfg.hop_length = 1;
        cfg.drop_last_frame = false;
        cfg.fft_output = FftOutputPrecision::F64;
        let output = LogMelExtractor::new(cfg)
            .unwrap()
            .extract(&[0.5], 16000, 1, &FeatureLimits::default())
            .unwrap();
        assert_eq!(output.plan.num_frames, 1);
        assert!(output.data.iter().all(|x| x.is_finite()));
    }
}
