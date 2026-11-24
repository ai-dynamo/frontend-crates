// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bounded, in-memory WAV/FLAC/MP3/Ogg-Vorbis decoding through Symphonia.
//! Preserves the source sample rate and mono/stereo interleaving. Unsupported
//! formats/rates are never passed to a Python fallback implicitly.

use crate::{MmError, Result, processor::DecodedMedia};
use symphonia::core::{
    codecs::audio::{AudioCodecParameters, AudioDecoderOptions, well_known as codecs},
    errors::Error,
    formats::{FormatOptions, TrackType, probe::Hint, well_known::FORMAT_ID_WAVE},
    io::{MediaSourceStream, ReadOnlySource},
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

fn wav_pcm_frame_bytes(params: &AudioCodecParameters) -> Result<usize> {
    // Use the stored sample width, which can differ from the valid bit count
    // in an extensible WAV (for example, 24 valid bits stored in 32 bits).
    let sample_bytes = match params.codec {
        codecs::CODEC_ID_PCM_U8 | codecs::CODEC_ID_PCM_ALAW | codecs::CODEC_ID_PCM_MULAW => 1,
        codecs::CODEC_ID_PCM_S16LE => 2,
        codecs::CODEC_ID_PCM_S24LE => 3,
        codecs::CODEC_ID_PCM_S32LE | codecs::CODEC_ID_PCM_F32LE => 4,
        codecs::CODEC_ID_PCM_F64LE => 8,
        _ => {
            return Err(MmError::unsupported(
                "audio: unsupported streamed WAV codec",
            ));
        }
    };
    let channels = params
        .channels
        .as_ref()
        .map_or(0, |channels| channels.count());
    if channels == 0 {
        return Err(MmError::invalid_input(
            "audio: streamed WAV has no channels",
        ));
    }
    Ok(sample_bytes * channels)
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
    // Decode sequentially. For a seekable MP3 without Xing/VBRI, Symphonia
    // estimates the frame count from the bitrate and uses it to trim output.
    // A read-only source skips that estimate but still honors gapless tags.
    let stream = MediaSourceStream::new(Box::new(ReadOnlySource::new(data)), Default::default());
    let mut format = symphonia::default::get_probe()
        .probe(
            &Hint::new(),
            stream,
            FormatOptions::default(),
            MetadataOptions::default(),
        )
        .map_err(|e| failure("cannot probe container", e))?;
    let is_wav = format.format_info().format == FORMAT_ID_WAVE;
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
    let streamed_wav_frame_bytes = if is_wav && declared_frames.is_none() {
        Some(wav_pcm_frame_bytes(params)?)
    } else {
        None
    };
    let mut decoder = symphonia::default::get_codecs()
        .make_audio_decoder(params, &AudioDecoderOptions::default())
        .map_err(|e| failure("unsupported audio codec", e))?;
    let mut samples = Vec::<f32>::new();
    let mut spec = None;
    loop {
        let packet = match format.next_packet() {
            Ok(Some(packet)) => packet,
            Ok(None) => break,
            // A WAV written to a pipe has no data length. The WAV reader
            // returns its last short packet before reporting EOF.
            Err(Error::IoError(e))
                if streamed_wav_frame_bytes.is_some()
                    && e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(e) => return Err(failure("invalid packet", e)),
        };
        if packet.track_id != track_id {
            continue;
        }
        if streamed_wav_frame_bytes.is_some_and(|size| !packet.data.len().is_multiple_of(size)) {
            return Err(MmError::invalid_input(
                "audio: streamed WAV contains an incomplete PCM frame",
            ));
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

    fn streamed_wav(data: &[u8], channels: u16, format: u16, bits: u16) -> Vec<u8> {
        let mut out = wav(data, channels, format, bits);
        out[4..8].fill(0xff);
        out[40..44].fill(0xff);
        out
    }

    #[test]
    fn streamed_wav_preserves_every_frame_and_applies_the_limit() {
        let frames = 93_680;
        let pcm = (0..frames * 2).map(|i| (i % 30063) as i16);
        let data = pcm.clone().flat_map(i16::to_le_bytes).collect::<Vec<_>>();
        let wav = streamed_wav(&data, 2, 1, 16);
        let limits = DecodeLimits {
            max_frames: frames,
            ..DecodeLimits::default()
        };
        let DecodedMedia::Audio {
            samples,
            sample_rate,
            channels,
        } = decode(&wav, &limits).unwrap()
        else {
            panic!()
        };
        assert_eq!(
            (samples.len(), sample_rate, channels),
            (frames * 2, 16000, 2)
        );
        assert!(samples.into_iter().eq(pcm.map(|v| f32::from(v) / 32768.)));
        assert!(matches!(
            decode(
                &wav,
                &DecodeLimits {
                    max_frames: frames - 1,
                    ..limits
                }
            ),
            Err(MmError::LimitExceeded { .. })
        ));
    }

    #[test]
    fn streamed_wav_rejects_incomplete_frames() {
        let limits = DecodeLimits::default();
        let pcm = streamed_wav(&[0; 12], 2, 1, 16);
        let float = streamed_wav(&[0; 12], 1, 3, 32);
        assert!(decode(&float, &limits).is_ok());
        for wav in [pcm, float] {
            assert!(matches!(
                decode(&wav[..wav.len() - 1], &limits),
                Err(MmError::InvalidInput { .. })
            ));
        }
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
