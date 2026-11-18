// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bounded, in-memory WAV/FLAC/MP3/Ogg-Vorbis decoding through Symphonia.
//! Preserves the source sample rate and mono/stereo interleaving. Unsupported
//! formats/rates are never passed to a Python fallback implicitly.

use crate::{MmError, Result, processor::DecodedMedia};
use std::io::Cursor;
use symphonia::core::{
    codecs::audio::AudioDecoderOptions,
    errors::Error,
    formats::{FormatOptions, TrackType, probe::Hint},
    io::MediaSourceStream,
    meta::MetadataOptions,
};

/// Per-item caps. Hosts also enforce whole-request budgets and admission.
#[derive(Clone, Copy, Debug)]
pub struct DecodeLimits {
    pub max_bytes: usize,
    /// Source frames (one sample per channel), before any model truncation.
    pub max_frames: usize,
}

impl Default for DecodeLimits {
    fn default() -> Self {
        Self {
            max_bytes: 32 * 1024 * 1024,
            max_frames: 192_000 * 60,
        }
    }
}

fn failure(context: &str, error: Error) -> MmError {
    match error {
        Error::Unsupported(_) => MmError::unsupported(format!("audio: {context}: {error}")),
        _ => MmError::invalid_input_with_source(format!("audio: {context}"), error),
    }
}

pub fn decode(data: &[u8], limits: &DecodeLimits) -> Result<DecodedMedia> {
    if data.len() > limits.max_bytes {
        return Err(MmError::limit_exceeded(
            "audio: encoded byte limit exceeded",
        ));
    }
    if data.is_empty() {
        return Err(MmError::invalid_input("audio: empty encoded input"));
    }
    let stream = MediaSourceStream::new(Box::new(Cursor::new(data.to_vec())), Default::default());
    let mut format = symphonia::default::get_probe()
        .probe(
            &Hint::new(),
            stream,
            FormatOptions::default(),
            MetadataOptions::default(),
        )
        .map_err(|e| failure("cannot probe container", e))?;
    let track = format
        .default_track(TrackType::Audio)
        .ok_or_else(|| MmError::unsupported("audio: container has no default audio track"))?;
    let track_id = track.id;
    let declared_frames = track.num_frames;
    if declared_frames.is_some_and(|n| n > limits.max_frames as u64) {
        return Err(MmError::limit_exceeded(
            "audio: declared frame limit exceeded",
        ));
    }
    let params = track
        .codec_params
        .as_ref()
        .and_then(|p| p.audio())
        .ok_or_else(|| MmError::unsupported("audio: missing codec parameters"))?;
    let mut decoder = symphonia::default::get_codecs()
        .make_audio_decoder(params, &AudioDecoderOptions::default())
        .map_err(|e| failure("unsupported audio codec", e))?;
    let mut samples = Vec::<f32>::new();
    let mut spec = None;
    while let Some(packet) = format
        .next_packet()
        .map_err(|e| failure("invalid packet", e))?
    {
        if packet.track_id != track_id {
            continue;
        }
        let decoded = decoder
            .decode(&packet)
            .map_err(|e| failure("cannot decode packet", e))?;
        let rate = decoded.spec().rate();
        let channels = decoded.spec().channels().count();
        if rate == 0 || channels == 0 {
            return Err(MmError::invalid_input("audio: invalid stream geometry"));
        }
        if channels > 2 {
            return Err(MmError::unsupported(
                "audio: only mono and stereo streams are supported",
            ));
        }
        if let Some(previous) = spec
            && previous != (rate, channels as u16)
        {
            return Err(MmError::unsupported(
                "audio: stream geometry changes between packets",
            ));
        }
        spec = Some((rate, channels as u16));
        let total = samples
            .len()
            .checked_add(decoded.samples_interleaved())
            .ok_or_else(|| MmError::limit_exceeded("audio: sample count overflow"))?;
        if total / channels > limits.max_frames {
            return Err(MmError::limit_exceeded(
                "audio: decoded frame limit exceeded",
            ));
        }
        let start = samples.len();
        samples
            .try_reserve(total - start)
            .map_err(|e| MmError::internal_with_source("audio: cannot allocate PCM", e))?;
        samples.resize(total, 0.0);
        decoded.copy_to_slice_interleaved(&mut samples[start..]);
    }
    let (sample_rate, channels) =
        spec.ok_or_else(|| MmError::invalid_input("audio: no decoded samples"))?;
    super::validate_pcm(&samples, sample_rate, channels)?;
    if declared_frames.is_some_and(|n| n != (samples.len() / usize::from(channels)) as u64) {
        return Err(MmError::invalid_input(
            "audio: decoded length does not match container frame count",
        ));
    }
    Ok(DecodedMedia::Audio {
        samples,
        sample_rate,
        channels,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wav(data: &[u8], channels: u16, format: u16, bits: u16) -> Vec<u8> {
        let mut out = b"RIFF".to_vec();
        out.extend_from_slice(&(36 + data.len() as u32).to_le_bytes());
        out.extend_from_slice(b"WAVEfmt \x10\x00\x00\x00");
        out.extend_from_slice(&format.to_le_bytes());
        out.extend_from_slice(&channels.to_le_bytes());
        out.extend_from_slice(&16000u32.to_le_bytes());
        let block = channels * (bits / 8);
        out.extend_from_slice(&(16000 * u32::from(block)).to_le_bytes());
        out.extend_from_slice(&block.to_le_bytes());
        out.extend_from_slice(&bits.to_le_bytes());
        out.extend_from_slice(b"data");
        out.extend_from_slice(&(data.len() as u32).to_le_bytes());
        out.extend_from_slice(data);
        out
    }

    #[test]
    fn wav_s16_and_f32_match_owned_pcm() {
        let pcm = [0i16, 16384, -16384, -32768, 32767, 0];
        let raw = pcm
            .into_iter()
            .flat_map(i16::to_le_bytes)
            .collect::<Vec<_>>();
        let DecodedMedia::Audio {
            samples,
            sample_rate,
            channels,
        } = decode(&wav(&raw, 2, 1, 16), &DecodeLimits::default()).unwrap()
        else {
            panic!()
        };
        assert_eq!(sample_rate, 16000);
        assert_eq!(channels, 2);
        let expected = pcm.map(|s| f32::from(s) / 32768.);
        assert_eq!(samples, expected);
        let floats = expected
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        let DecodedMedia::Audio { samples, .. } =
            decode(&wav(&floats, 2, 3, 32), &DecodeLimits::default()).unwrap()
        else {
            panic!()
        };
        assert_eq!(samples, expected);
    }

    #[test]
    fn encoded_decode_rejects_malformed_and_oversized_input() {
        let data = wav(&[0; 200], 1, 1, 16);
        assert!(matches!(
            decode(
                &data,
                &DecodeLimits {
                    max_bytes: 10,
                    max_frames: 100
                }
            ),
            Err(MmError::LimitExceeded { .. })
        ));
        assert!(matches!(
            decode(
                &data,
                &DecodeLimits {
                    max_bytes: 1000,
                    max_frames: 99
                }
            ),
            Err(MmError::LimitExceeded { .. })
        ));
        assert!(decode(&data[..data.len() - 2], &DecodeLimits::default()).is_err());
        assert!(decode(b"not audio", &DecodeLimits::default()).is_err());
        assert!(decode(&[], &DecodeLimits::default()).is_err());
        let nan = wav(&f32::NAN.to_le_bytes(), 1, 3, 32);
        assert!(decode(&nan, &DecodeLimits::default()).is_err());
    }
}
