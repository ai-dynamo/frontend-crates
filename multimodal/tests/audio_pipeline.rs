// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#![cfg(feature = "audio-decode")]

use dynamo_multimodal::{
    audio::{self, decode, features, waveform},
    processor::DecodedMedia,
};

#[test]
fn lossless_containers_and_pcm_feed_the_same_generic_pipeline() {
    let source = (0..4096)
        .flat_map(|i| [((i % 64) - 32) * 400, (((i + 16) % 64) - 32) * 400])
        .map(|i| i as f32 / 32768.)
        .collect::<Vec<_>>();
    let raw = source
        .iter()
        .flat_map(|s| s.to_le_bytes())
        .collect::<Vec<_>>();
    let decoded = audio::from_f32le(&raw, 22050, 2, 4096, 4096).unwrap();
    let wave_config = waveform::WaveformConfig {
        sample_rate: 16000,
        resample: waveform::ResamplePolicy::RubatoFft,
        max_samples: None,
        pad_to_samples: Some(4096),
    };
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/audio/log_mel.json")).unwrap();
    let feature_config: features::LogMelConfig =
        serde_json::from_value(fixture["cases"][0]["config"].clone()).unwrap();
    let extractor = features::LogMelExtractor::new(feature_config).unwrap();
    let process = |media: &DecodedMedia| {
        let DecodedMedia::Audio {
            samples,
            sample_rate,
            channels,
        } = media
        else {
            panic!()
        };
        let prepared = waveform::prepare(
            samples,
            *sample_rate,
            *channels,
            &wave_config,
            &waveform::WaveformLimits::default(),
        )
        .unwrap();
        let output = extractor
            .extract(
                &prepared.samples,
                prepared.sample_rate,
                prepared.plan.valid_samples,
                &features::FeatureLimits::default(),
            )
            .unwrap();
        (prepared.plan, output.plan, output.data)
    };
    let expected = process(&decoded);
    for bytes in [
        include_bytes!("fixtures/audio/stereo.wav").as_slice(),
        include_bytes!("fixtures/audio/stereo.flac").as_slice(),
    ] {
        let media = decode::decode(bytes, &decode::DecodeLimits::default()).unwrap();
        let DecodedMedia::Audio {
            samples,
            sample_rate,
            channels,
        } = &media
        else {
            panic!()
        };
        assert_eq!((*sample_rate, *channels), (22050, 2));
        assert_eq!(samples, &source);
        assert_eq!(process(&media), expected);
    }
}

#[test]
fn lossy_codecs_preserve_gapless_length_and_channels() {
    for (bytes, reference) in [
        (
            include_bytes!("fixtures/audio/stereo.mp3").as_slice(),
            include_bytes!("fixtures/audio/stereo.mp3.decoded.f32le").as_slice(),
        ),
        (
            include_bytes!("fixtures/audio/stereo.ogg").as_slice(),
            include_bytes!("fixtures/audio/stereo.ogg.decoded.f32le").as_slice(),
        ),
    ] {
        let media = decode::decode(bytes, &decode::DecodeLimits::default()).unwrap();
        let DecodedMedia::Audio {
            samples,
            sample_rate,
            channels,
        } = media
        else {
            panic!()
        };
        assert_eq!((sample_rate, channels, samples.len()), (22050, 2, 8192));
        assert_eq!(reference.len(), samples.len() * 4);
        let expected = reference
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()));
        let mut max_error = 0.0f32;
        for (&sample, expected) in samples.iter().zip(expected) {
            max_error = max_error.max((sample - expected).abs());
            // Independent libsndfile decoder, not the pre-encoding waveform.
            // This is explicitly a numerical check, not model qualification.
            assert!((sample - expected).abs() <= 2e-6);
        }
        eprintln!("lossy decode max_abs_error={max_error:e}");
    }
}
