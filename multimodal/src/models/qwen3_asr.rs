// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Experimental Qwen3-ASR profile for mono 16 kHz PCM. This profile rejects
//! clips longer than 30 seconds. For accepted clips, it follows SGLang's
//! Python Whisper feature extraction: pad to 30 seconds and count
//! `ceil(samples / 160)` valid frames. For a single clip, Hugging Face
//! `AutoProcessor` with `padding=True` can use `floor(samples / 160)` and
//! produce one fewer audio token at a chunk or stride boundary.
//! The caller provides resolved special token IDs and selects this profile
//! explicitly.

use crate::audio::features::{
    CenterPadding, FeatureLimits, FftOutputPrecision, LogMelConfig, LogMelExtractor, MelScale,
    WindowFunction,
};
use crate::audio::waveform::{self, ResamplePolicy, WaveformConfig, WaveformLimits};
use crate::processor::{
    Capabilities, DecodedMedia, MediaMetadata, MmFamilyProcessor, Modality, PositionOutput,
    ProcessedItem, Tensor, TensorData, TokenLayout,
};
use crate::{MmError, Result, token_layout::layout_by_placeholder};

const SAMPLE_RATE: u32 = 16_000;
const HOP: usize = 160;
const N_FFT: usize = 400;
const MELS: usize = 128;
pub const MAX_SAMPLES: usize = 30 * SAMPLE_RATE as usize;
const WAVEFORM: WaveformConfig = WaveformConfig {
    sample_rate: SAMPLE_RATE,
    resample: ResamplePolicy::Reject,
    max_samples: None,
    pad_to_samples: Some(MAX_SAMPLES),
};
const WAVEFORM_LIMITS: WaveformLimits = WaveformLimits {
    max_source_frames: MAX_SAMPLES,
    max_output_samples: MAX_SAMPLES,
};
const FEATURE_LIMITS: FeatureLimits = FeatureLimits {
    max_samples: MAX_SAMPLES,
    max_frames: MAX_SAMPLES / HOP,
    max_values: MELS * (MAX_SAMPLES / HOP),
};

#[derive(Clone, Copy, Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Qwen3AsrSpec {
    pub audio_token_id: i32,
    pub audio_start_id: i32,
    pub audio_end_id: i32,
}

pub struct Qwen3AsrProcessor {
    spec: Qwen3AsrSpec,
    extractor: LogMelExtractor,
}

impl Qwen3AsrProcessor {
    pub fn new(spec: Qwen3AsrSpec) -> Result<Self> {
        let [pad, start, end] = [spec.audio_token_id, spec.audio_start_id, spec.audio_end_id];
        if pad < 0 || start < 0 || end < 0 || pad == start || pad == end || start == end {
            return Err(MmError::invalid_input("qwen3_asr: invalid audio token IDs"));
        }
        let extractor = LogMelExtractor::new(LogMelConfig {
            sample_rate: SAMPLE_RATE,
            n_fft: N_FFT,
            hop_length: HOP,
            n_mels: MELS,
            min_frequency: 0.0,
            max_frequency: f64::from(SAMPLE_RATE) / 2.0,
            mel_scale: MelScale::Slaney,
            slaney_normalize: true,
            window: WindowFunction::HannPeriodic,
            center: CenterPadding::Reflect,
            fft_output: FftOutputPrecision::F32,
            drop_last_frame: true,
            log_floor: 1e-10,
            dynamic_range: Some(8.0),
            output_offset: 4.0,
            output_scale: 0.25,
        })?;
        Ok(Self { spec, extractor })
    }

    fn layout_with_counts(&self, ids: &[i32], counts: &[usize]) -> Result<TokenLayout> {
        let placeholder = [
            self.spec.audio_start_id,
            self.spec.audio_token_id,
            self.spec.audio_end_id,
        ];
        if ids.windows(3).filter(|span| *span == placeholder).count() != counts.len() {
            return Err(MmError::invalid_input(
                "qwen3_asr: expected one start/pad/end placeholder per audio item",
            ));
        }
        layout_by_placeholder(ids, self.spec.audio_token_id, counts)
    }
}

/// Each full 100-frame encoder chunk contributes 13 tokens. Three stride-2
/// convolutions reduce the last partial chunk using ceil division.
fn encoder_tokens(valid_samples: usize) -> usize {
    let frames = valid_samples.div_ceil(HOP);
    (frames / 100) * 13 + (frames % 100).div_ceil(8)
}

impl MmFamilyProcessor for Qwen3AsrProcessor {
    fn capabilities(&self) -> Capabilities {
        Capabilities::new(false, false, true)
    }

    fn num_media_tokens(&self, media: &MediaMetadata) -> Result<usize> {
        let MediaMetadata::Audio {
            num_samples,
            sample_rate,
        } = *media
        else {
            return Err(MmError::invalid_input("qwen3_asr: expected audio metadata"));
        };
        let samples = usize::try_from(num_samples)
            .map_err(|_| MmError::limit_exceeded("qwen3_asr: sample count overflow"))?;
        let waveform = waveform::plan(samples, sample_rate, &WAVEFORM, &WAVEFORM_LIMITS)?;
        Ok(encoder_tokens(waveform.valid_samples))
    }

    fn layout_from_metadata(&self, ids: &[i32], media: &[MediaMetadata]) -> Result<TokenLayout> {
        let counts = media
            .iter()
            .map(|item| self.num_media_tokens(item))
            .collect::<Result<Vec<_>>>()?;
        self.layout_with_counts(ids, &counts)
    }

    fn process_item(&self, media: &DecodedMedia) -> Result<ProcessedItem> {
        let DecodedMedia::Audio {
            samples,
            sample_rate: SAMPLE_RATE,
            channels: 1,
        } = media
        else {
            return Err(MmError::invalid_input(
                "qwen3_asr: expected mono 16 kHz PCM",
            ));
        };
        let waveform = waveform::prepare(samples, SAMPLE_RATE, 1, &WAVEFORM, &WAVEFORM_LIMITS)?;
        let features = self.extractor.extract(
            &waveform.samples,
            SAMPLE_RATE,
            waveform.plan.valid_samples,
            &FEATURE_LIMITS,
        )?;
        let frames = features.plan.num_frames;
        Ok(ProcessedItem {
            modality: Modality::Audio,
            feature_token_count: encoder_tokens(waveform.plan.valid_samples),
            feature: Tensor {
                shape: vec![1, MELS, frames],
                data: TensorData::F32(features.data),
            },
            aux: vec![(
                "feature_attention_mask".into(),
                Tensor {
                    shape: vec![1, frames],
                    data: TensorData::I64(features.plan.frame_anchor_mask()),
                },
            )],
            geometry: None,
        })
    }

    fn layout(&self, ids: &[i32], items: &[ProcessedItem]) -> Result<TokenLayout> {
        let counts = items
            .iter()
            .map(|item| item.feature_token_count)
            .collect::<Vec<_>>();
        self.layout_with_counts(ids, &counts)
    }

    fn positions(
        &self,
        input_len: usize,
        _offsets: &[(u32, u32)],
        _items: &[ProcessedItem],
    ) -> Result<PositionOutput> {
        Ok(PositionOutput::MRope {
            positions: (0..3).flat_map(|_| 0..input_len as i64).collect(),
            delta: 0,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expands_only_audio_feature_tokens_and_keeps_batch_shapes() {
        let processor = crate::registry::processor_from_spec(
            r#"{"family":"qwen3_asr","audio_token_id":2,"audio_start_id":1,"audio_end_id":3}"#,
        )
        .unwrap();
        let metadata = |num_samples| MediaMetadata::Audio {
            num_samples,
            sample_rate: SAMPLE_RATE,
        };
        for (samples, count) in [
            (1, 1),
            (HOP * 99, 13),
            (HOP * 100, 13),
            (HOP * 100 + 1, 14),
            (160_096, 131),
            (MAX_SAMPLES, 390),
        ] {
            assert_eq!(
                processor
                    .num_media_tokens(&metadata(samples as u64))
                    .unwrap(),
                count
            );
        }
        assert!(processor.num_media_tokens(&metadata(0)).is_err());
        assert!(
            processor
                .num_media_tokens(&metadata(MAX_SAMPLES as u64 + 1))
                .is_err()
        );
        let samples = HOP * 100 + 1;
        let item = processor
            .process_item(&DecodedMedia::Audio {
                samples: vec![0.0; samples],
                sample_rate: SAMPLE_RATE,
                channels: 1,
            })
            .unwrap();
        assert_eq!(item.feature_token_count, 14);
        assert_eq!(item.feature.shape, [1, 128, 3000]);
        let (name, mask) = &item.aux[0];
        assert_eq!(name, "feature_attention_mask");
        assert_eq!(mask.shape, [1, 3000]);
        let TensorData::I64(mask) = &mask.data else {
            panic!("expected an I64 mask")
        };
        assert!(mask[..101].iter().all(|&value| value == 1));
        assert!(mask[101..].iter().all(|&value| value == 0));
        let items = [item];
        assert!(processor.layout(&[7, 2, 9], &items).is_err());
        let ids = [7, 1, 2, 3, 9];
        let layout = processor.layout(&ids, &items).unwrap();
        let expanded = crate::token_layout::apply_layout(&ids, &layout, &[14], 32).unwrap();
        assert_eq!(expanded.offsets, [(2, 15)]);
        assert_eq!(expanded.input_ids[16..], [3, 9]);
        let metadata = [metadata(samples as u64)];
        assert!(
            processor
                .layout_from_metadata(&[7, 2, 9], &metadata)
                .is_err()
        );
        let plan = processor.layout_from_metadata(&ids, &metadata).unwrap();
        let planned = crate::token_layout::apply_layout(&ids, &plan, &[14], 32).unwrap();
        assert_eq!(planned.input_ids, expanded.input_ids);
        assert_eq!(planned.offsets, expanded.offsets);
        assert_eq!(planned.feature_ranges, expanded.feature_ranges);
        let PositionOutput::MRope { positions, delta } = processor
            .positions(expanded.input_ids.len(), &expanded.offsets, &items)
            .unwrap()
        else {
            panic!("expected audio M-RoPE positions");
        };
        assert_eq!(delta, 0);
        assert_eq!(positions, (0..18).cycle().take(54).collect::<Vec<_>>());
    }

    #[test]
    fn tone_features_match_the_sglang_whisper_reference() {
        let processor = Qwen3AsrProcessor::new(Qwen3AsrSpec {
            audio_token_id: 2,
            audio_start_id: 1,
            audio_end_id: 3,
        })
        .unwrap();
        let samples = (0..SAMPLE_RATE)
            .map(|i| {
                let phase = std::f64::consts::TAU * 440.0 * f64::from(i) / f64::from(SAMPLE_RATE);
                (0.5 * phase.sin()) as f32
            })
            .collect();
        let item = processor
            .process_item(&DecodedMedia::Audio {
                samples,
                sample_rate: SAMPLE_RATE,
                channels: 1,
            })
            .unwrap();
        let TensorData::F32(features) = &item.feature.data else {
            panic!("expected F32 features")
        };
        let frames = MAX_SAMPLES / HOP;
        // Transformers 5.19 WhisperFeatureExtractor on Torch CPU with SGLang's
        // 30-second padding. Include the reflected edge and padded silence.
        for (mel, frame, expected) in [
            (18, 0, 0.714_360_2),
            (18, 50, 1.485_354_1),
            (18, 99, 1.483_988_2),
            (18, 100, 1.334_928_5),
            (0, 50, -0.514_645_9),
            (18, 2000, -0.514_645_9),
        ] {
            let got = features[mel * frames + frame];
            assert!(
                (got - expected).abs() < 1e-4,
                "mel {mel}, frame {frame}: got {got}, expected {expected}"
            );
        }
    }
}
