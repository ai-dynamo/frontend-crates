// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! CPU integration probe. No tokenizer, model, network, Python or SHM needed.
//! cargo run -p dynamo-multimodal --features audio-decode --example audio_preprocess
//!     -- INPUT CONFIG.json OUTPUT.f32le

use dynamo_multimodal::audio::{self, decode, features, waveform};
use dynamo_multimodal::{MmError, processor::DecodedMedia};
use std::io::{BufWriter, Write};

#[derive(serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum Input {
    Encoded,
    F32le {
        sample_rate: u32,
        channels: u16,
        sample_frames: u64,
    },
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    input: Input,
    waveform: waveform::WaveformConfig,
    features: features::LogMelConfig,
}

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.len() != 3 {
        return Err("usage: audio_preprocess INPUT CONFIG.json OUTPUT.f32le".into());
    }
    let config: Config = serde_json::from_slice(&std::fs::read(&args[1])?)?;
    let limits = decode::DecodeLimits::default();
    if std::fs::metadata(&args[0])?.len() > limits.max_bytes as u64 {
        return Err(MmError::limit_exceeded("audio: encoded/PCM file exceeds byte limit").into());
    }
    let bytes = std::fs::read(&args[0])?;
    let media = match config.input {
        Input::Encoded => decode::decode(&bytes, &limits)?,
        Input::F32le {
            sample_rate,
            channels,
            sample_frames,
        } => audio::from_f32le(
            &bytes,
            sample_rate,
            channels,
            sample_frames,
            limits.max_frames,
        )?,
    };
    let DecodedMedia::Audio {
        samples,
        sample_rate,
        channels,
    } = media
    else {
        unreachable!()
    };
    let prepared = waveform::prepare(
        &samples,
        sample_rate,
        channels,
        &config.waveform,
        &waveform::WaveformLimits::default(),
    )?;
    let extractor = features::LogMelExtractor::new(config.features)?;
    let output = extractor.extract(
        &prepared.samples,
        prepared.sample_rate,
        prepared.plan.valid_samples,
        &features::FeatureLimits::default(),
    )?;
    let mut file = BufWriter::new(std::fs::File::create(&args[2])?);
    for value in &output.data {
        file.write_all(&value.to_le_bytes())?;
    }
    file.flush()?;
    let metadata = serde_json::json!({
        "source_rate": sample_rate, "source_channels": channels,
        "source_frames": prepared.plan.source_frames,
        "sample_rate": prepared.sample_rate,
        "resampled_samples": prepared.plan.resampled_samples,
        "valid_samples": prepared.plan.valid_samples,
        "output_samples": prepared.plan.output_samples,
        "truncated": prepared.plan.truncated,
        "shape": [output.plan.num_mels, output.plan.num_frames],
        "valid_frames": output.plan.valid_frames,
        "frame_anchor_mask": output.plan.frame_anchor_mask(),
    });
    println!("{}", serde_json::to_string_pretty(&metadata)?);
    Ok(())
}
