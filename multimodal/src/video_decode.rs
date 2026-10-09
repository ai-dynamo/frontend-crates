// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Video decoding to sampled RGB frames on NVIDIA GPUs, through NVDEC.
//!
//! [`probe`] reads a clip's container and reports its codec, coded size,
//! duration and frame count. [`decode_sampled`] decodes it and keeps
//! `num_frames` frames spread evenly over the clip, as [`VideoFrames`].
//!
//! Containers: MP4 (ISO BMFF) through `re_mp4`, WebM and Matroska through
//! `matroska-demuxer`. Codecs: H.264, HEVC, VP9 and AV1, 8-bit 4:2:0, as far as
//! the GPU's NVDEC supports them (checked at runtime).
//!
//! Decoding runs in NVIDIA's driver: `libcuda.so.1` and `libnvcuvid.so.1` are
//! loaded at runtime (see `nvdec`), so nothing is linked at build time and no
//! software decoder ships in this crate. Without a driver, or on a GPU whose
//! NVDEC does not support the stream, decoding returns
//! [`MmError::Unsupported`]. GPU 0 is used.
//!
//! YUV to RGB follows the bitstream's colour tags: BT.709, or BT.601 when the
//! bitstream says BT.601 or nothing (FFmpeg's default for untagged video), in
//! limited or full range. Other matrices are refused rather than guessed.
//! Container-level colour tags (MP4 `colr`, Matroska `Colour`) are not read.
//! Interlaced content is not deinterlaced.
//!
//! Decoding blocks the calling thread. Every frame up to the last sampled one
//! goes through the decoder, because frames depend on earlier frames; only the
//! sampled ones are copied off the GPU.

// NVDEC is Linux-only; elsewhere the decode path is compiled out.
#![cfg_attr(not(target_os = "linux"), allow(dead_code))]

use std::borrow::Cow;

use crate::frames::{VideoFrames, VideoTiming};
use crate::{MmError, Result};

#[cfg(target_os = "linux")]
mod nvdec;

/// A video codec found in a container.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum VideoCodec {
    Vp8,
    Vp9,
    Av1,
    H264,
    Hevc,
}

impl VideoCodec {
    fn name(self) -> &'static str {
        match self {
            Self::Vp8 => "VP8",
            Self::Vp9 => "VP9",
            Self::Av1 => "AV1",
            Self::H264 => "H.264",
            Self::Hevc => "HEVC",
        }
    }

    /// Why this crate cannot decode `self`, or `None` if NVDEC may (whether
    /// it does depends on the GPU, checked when decoding).
    fn unsupported_reason(self) -> Option<String> {
        match self {
            Self::Vp8 => Some("VP8 is not decoded by this crate".into()),
            _ => None,
        }
    }
}

/// What [`probe`] learns from the container, without decoding.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct VideoInfo {
    pub codec: VideoCodec,
    /// Coded frame size, from the codec's sample entry or track header (not
    /// the display size, which differs for non-square pixels).
    pub width: u32,
    pub height: u32,
    /// Seconds.
    pub duration: f64,
    /// Frames per second, from the frame count and duration.
    pub fps: f64,
    /// Frames in the first video track the container marks for display.
    pub frame_count: u64,
}

/// Bounds checked before anything large is allocated.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct VideoDecodeLimits {
    /// Largest frame accepted, from the container and from each keyframe
    /// header before it is decoded.
    pub max_width: u32,
    pub max_height: u32,
    /// Cap on the returned RGB frames, in bytes.
    pub max_output_bytes: usize,
}

impl Default for VideoDecodeLimits {
    /// 8192 x 8192 frames and 512 MiB of output, the decoded-frame budget
    /// `ai-dynamo/dynamo` uses for video.
    fn default() -> Self {
        Self {
            max_width: 8192,
            max_height: 8192,
            max_output_bytes: 512 << 20,
        }
    }
}

/// One compressed frame, in decode order.
struct Packet<'a> {
    data: Cow<'a, [u8]>,
    pts: f64,
    /// Whether the decoder will output a frame for this packet. `false` only
    /// for a VP9 packet whose frames are all hidden (`show_frame = 0`).
    displays: bool,
    /// Whether the output frame belongs to the video: `false` for a Matroska
    /// frame marked invisible and for a packet that displays nothing. Every
    /// packet is decoded, since later frames may reference it.
    shown: bool,
}

impl<'a> Packet<'a> {
    fn new(codec: VideoCodec, data: Cow<'a, [u8]>, pts: f64, invisible: bool) -> Result<Self> {
        let displays = codec != VideoCodec::Vp9 || vp9_packet_displays(&data)?;
        Ok(Self {
            data,
            pts,
            displays,
            shown: displays && !invisible,
        })
    }
}

/// A demuxed video track.
struct Track<'a> {
    codec: VideoCodec,
    width: u32,
    height: u32,
    duration: f64,
    /// The codec configuration record: `avcC` (H.264), `hvcC` (HEVC) or `av1C`
    /// (AV1), from the MP4 sample entry or Matroska `CodecPrivate`.
    config: Option<Vec<u8>>,
    packets: Vec<Packet<'a>>,
}

impl Track<'_> {
    fn info(&self) -> VideoInfo {
        let frame_count = self.packets.iter().filter(|p| p.shown).count() as u64;
        VideoInfo {
            codec: self.codec,
            width: self.width,
            height: self.height,
            duration: self.duration,
            fps: frame_count as f64 / self.duration,
            frame_count,
        }
    }
}

/// Read the container and describe its first video track.
pub fn probe(bytes: &[u8]) -> Result<VideoInfo> {
    Ok(demux(bytes, false)?.info())
}

/// Decode `bytes` and keep `num_frames` frames spread evenly over the clip:
/// frame indices `round(linspace(0, frame_count - 1, num_frames))`.
///
/// Errors: [`MmError::Unsupported`] without an NVIDIA driver, for VP8, for a
/// stream this GPU's NVDEC cannot decode, and for anything other than 8-bit
/// 4:2:0 in a supported colour matrix; [`MmError::LimitExceeded`] when a frame or the output exceeds
/// `limits`; [`MmError::InvalidInput`] for an unreadable container or
/// bitstream, zero frames requested, more frames than the clip shows, or a
/// resolution change mid-stream.
pub fn decode_sampled(
    bytes: &[u8],
    num_frames: u64,
    limits: &VideoDecodeLimits,
) -> Result<VideoFrames> {
    if num_frames == 0 {
        return Err(MmError::invalid_input("num_frames must be at least 1"));
    }
    // Refuses an undecodable codec before the packets are collected.
    let track = demux(bytes, true)?;
    let info = track.info();
    if num_frames > info.frame_count {
        return Err(MmError::invalid_input(format!(
            "cannot sample {num_frames} frames from a video with {}",
            info.frame_count
        )));
    }
    check_size(info.width, info.height, limits)?;
    let too_big = || {
        MmError::limit_exceeded(format!(
            "{num_frames} frames of {}x{} exceed the {} byte output limit",
            info.width, info.height, limits.max_output_bytes
        ))
    };
    let frame_len = (info.width as usize)
        .checked_mul(info.height as usize)
        .and_then(|n| n.checked_mul(3))
        .ok_or_else(too_big)?;
    frame_len
        .checked_mul(usize::try_from(num_frames).map_err(|_| too_big())?)
        .filter(|n| *n <= limits.max_output_bytes)
        .ok_or_else(too_big)?;
    let targets = sample_indices(info.frame_count, num_frames);

    let mut sampler = Sampler {
        targets: &targets,
        next: 0,
        shown: 0,
        width: info.width,
        height: info.height,
        limits: *limits,
        // Grows frame by frame: nothing large is reserved before decoding.
        rgb: Vec::new(),
        timestamps: Vec::with_capacity(targets.len()),
    };
    #[cfg(target_os = "linux")]
    nvdec::decode(&track, &mut sampler)?;
    #[cfg(not(target_os = "linux"))]
    return Err(MmError::unsupported(
        "NVDEC decoding is only built on Linux",
    ));
    if !sampler.done() {
        return Err(MmError::invalid_input(format!(
            "the video decoded to {} displayable frames, fewer than the {} it declares",
            sampler.shown, info.frame_count
        )));
    }
    VideoFrames::new(
        info.width as usize,
        info.height as usize,
        sampler.rgb,
        VideoTiming {
            timestamps: sampler.timestamps,
            source_fps: info.fps,
            source_duration: info.duration,
        },
    )
}

fn check_size(width: u32, height: u32, limits: &VideoDecodeLimits) -> Result<()> {
    if width == 0 || height == 0 {
        return Err(MmError::invalid_input(
            "video track has an empty frame size",
        ));
    }
    if width > limits.max_width || height > limits.max_height {
        return Err(MmError::limit_exceeded(format!(
            "{width}x{height} video exceeds the {}x{} decode limit",
            limits.max_width, limits.max_height
        )));
    }
    Ok(())
}

/// `round(linspace(0, total - 1, n))`, as the HF and Dynamo samplers pick frames.
fn sample_indices(total: u64, n: u64) -> Vec<u64> {
    if n == 1 {
        return vec![0];
    }
    let last = (total - 1) as f64;
    (0..n)
        .map(|i| (last * i as f64 / (n - 1) as f64).round_ties_even() as u64)
        .collect()
}

/// Collects the sampled frames as displayed frames arrive in order.
struct Sampler<'t> {
    targets: &'t [u64],
    next: usize,
    shown: u64,
    width: u32,
    height: u32,
    limits: VideoDecodeLimits,
    rgb: Vec<u8>,
    timestamps: Vec<f64>,
}

impl Sampler<'_> {
    fn done(&self) -> bool {
        self.next == self.targets.len()
    }

    /// Whether the next displayed frame is a sampled one.
    fn wants_next(&self) -> bool {
        self.targets.get(self.next) == Some(&self.shown)
    }

    /// Count a displayed frame that is not sampled.
    fn skip(&mut self) {
        self.shown += 1;
    }

    /// Account for one displayed frame; convert it if it is a sampled one
    /// (more than once if the index repeats).
    fn push(&mut self, picture: &Nv12<'_>, pts: f64) -> Result<()> {
        if (picture.width, picture.height) != (self.width, self.height) {
            return Err(MmError::invalid_input(format!(
                "frame size changed mid-stream from {}x{} to {}x{}",
                self.width, self.height, picture.width, picture.height
            )));
        }
        let index = self.shown;
        self.shown += 1;
        while self.targets.get(self.next) == Some(&index) {
            picture.to_rgb(&mut self.rgb);
            self.timestamps.push(pts);
            self.next += 1;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Demux
// ---------------------------------------------------------------------------

fn demux(bytes: &[u8], require_decodable: bool) -> Result<Track<'_>> {
    let track = if bytes.get(4..8) == Some(b"ftyp") {
        // re_mp4 trusts the sample tables: it expands their counts before we
        // can check anything (a 1 KB file can ask for 2^31 samples) and panics
        // on some inconsistencies. Bound the tables first; keep catching its
        // panics as a backstop. A caught panic still reaches the panic hook.
        validate_mp4_tables(bytes)?;
        std::panic::catch_unwind(|| demux_mp4(bytes, require_decodable))
            .unwrap_or_else(|_| Err(MmError::invalid_input("malformed MP4")))?
    } else if bytes.starts_with(&[0x1A, 0x45, 0xDF, 0xA3]) {
        demux_matroska(bytes, require_decodable)?
    } else {
        return Err(MmError::invalid_input(
            "unrecognised video container: expected MP4, WebM or Matroska",
        ));
    };
    if !track.packets.iter().any(|p| p.shown) {
        return Err(MmError::invalid_input("video track has no frames"));
    }
    if !(track.duration.is_finite() && track.duration > 0.0) {
        return Err(MmError::invalid_input("video duration is unknown or zero"));
    }
    Ok(track)
}

/// Refuse MP4 sample tables the file cannot back, before re_mp4 expands them.
///
/// Every sample needs at least one byte of the file, so the samples of all
/// tracks and fragments together may not outnumber the file's bytes; every
/// table must fit inside its box; and a track's duration may not end before
/// its last sample starts (re_mp4 subtracts the two).
fn validate_mp4_tables(bytes: &[u8]) -> Result<()> {
    let limit = bytes.len() as u64;
    let malformed = |what: &str| MmError::invalid_input(format!("malformed MP4: {what}"));
    // Running totals over the whole file, per kind of table re_mp4 expands:
    // sample sizes (stsz, stz2, trun), decode times (stts), offsets (ctts).
    let (mut sizes, mut times, mut offsets) = (0u64, 0u64, 0u64);
    let add = |total: &mut u64, count: u64| -> Result<()> {
        *total = total.saturating_add(count);
        if *total > limit {
            return Err(malformed("more samples than the file has bytes"));
        }
        Ok(())
    };
    let field = |body: &[u8], at: usize| -> Result<u64> {
        body.get(at..at + 4)
            .map(|b| u64::from(u32::from_be_bytes([b[0], b[1], b[2], b[3]])))
            .ok_or_else(|| malformed("truncated box"))
    };
    // Entries of `width` bytes after the version/flags and count fields.
    let table = |body: &[u8], width: u64| -> Result<u64> {
        let count = field(body, 4)?;
        if 8 + count * width > body.len() as u64 {
            return Err(malformed("table larger than its box"));
        }
        Ok(count)
    };
    let mut track_duration: Option<u64> = None;
    walk_mp4_boxes(bytes, 0, bytes.len(), 0, &mut |kind, body| {
        match kind {
            b"mdhd" => {
                let duration = match body.first() {
                    Some(1) => body
                        .get(24..32)
                        .map(|b| u64::from_be_bytes(b.try_into().unwrap_or_default())),
                    _ => field(body, 16).ok(),
                };
                track_duration = Some(duration.ok_or_else(|| malformed("truncated mdhd"))?);
            }
            b"stsz" => {
                let (size, count) = (field(body, 4)?, field(body, 8)?);
                add(&mut sizes, count)?;
                if size == 0 && 12 + count * 4 > body.len() as u64 {
                    return Err(malformed("sample size table"));
                }
            }
            b"stz2" | b"trun" => {
                add(
                    &mut sizes,
                    field(body, if kind == b"trun" { 4 } else { 8 })?,
                )?;
            }
            b"stts" | b"ctts" => {
                let entries = table(body, 8)?;
                let (mut samples, mut start, mut last) = (0u64, 0u128, 0u128);
                for i in 0..entries as usize {
                    let (n, delta) = (field(body, 8 + i * 8)?, field(body, 12 + i * 8)?);
                    samples += n;
                    if n > 0 {
                        start += u128::from(n) * u128::from(delta);
                        last = u128::from(delta);
                    }
                }
                add(
                    if kind == b"stts" {
                        &mut times
                    } else {
                        &mut offsets
                    },
                    samples,
                )?;
                if kind == b"stts"
                    && let Some(duration) = track_duration
                    && samples > 0
                    && start - last > u128::from(duration)
                {
                    return Err(malformed("track duration ends before its last sample"));
                }
            }
            b"stsc" => {
                for i in 0..table(body, 12)? as usize {
                    if field(body, 12 + i * 12)? > limit {
                        return Err(malformed("samples per chunk larger than the file"));
                    }
                }
            }
            b"stco" | b"stss" => {
                table(body, 4)?;
            }
            b"co64" => {
                table(body, 8)?;
            }
            _ => {}
        }
        Ok(())
    })
}

/// Call `visit` with the type and body of every leaf box, descending into the
/// container boxes that hold sample tables.
fn walk_mp4_boxes<F: FnMut(&[u8; 4], &[u8]) -> Result<()>>(
    bytes: &[u8],
    start: usize,
    end: usize,
    depth: u32,
    visit: &mut F,
) -> Result<()> {
    let malformed = || MmError::invalid_input("malformed MP4: box size");
    if depth > 16 {
        return Err(MmError::invalid_input(
            "malformed MP4: boxes nested too deeply",
        ));
    }
    let mut pos = start;
    while pos + 8 <= end {
        let word = |at: usize| u32::from_be_bytes(bytes[at..at + 4].try_into().unwrap_or_default());
        let kind: [u8; 4] = bytes[pos + 4..pos + 8].try_into().unwrap_or_default();
        let (header, size) = match word(pos) {
            0 => (8, (end - pos) as u64),
            1 => {
                let large = bytes.get(pos + 8..pos + 16).ok_or_else(malformed)?;
                (16, u64::from_be_bytes(large.try_into().unwrap_or_default()))
            }
            n => (8, u64::from(n)),
        };
        let size = usize::try_from(size)
            .ok()
            .filter(|s| *s >= header && pos.checked_add(*s).is_some_and(|e| e <= end))
            .ok_or_else(malformed)?;
        let body = (pos + header, pos + size);
        match &kind {
            b"moov" | b"trak" | b"mdia" | b"minf" | b"stbl" | b"moof" | b"traf" | b"mvex"
            | b"edts" => walk_mp4_boxes(bytes, body.0, body.1, depth + 1, visit)?,
            _ => visit(&kind, &bytes[body.0..body.1])?,
        }
        pos += size;
    }
    Ok(())
}

fn refuse_if_undecodable(codec: VideoCodec, require_decodable: bool) -> Result<()> {
    match codec.unsupported_reason() {
        Some(reason) if require_decodable => Err(MmError::unsupported(reason)),
        _ => Ok(()),
    }
}

fn demux_mp4(bytes: &[u8], require_decodable: bool) -> Result<Track<'_>> {
    use re_mp4::StsdBoxContent as Entry;
    let mp4 = re_mp4::Mp4::read_bytes(bytes)
        .map_err(|e| MmError::invalid_input_with_source("unreadable MP4", e))?;
    let track = mp4
        .tracks()
        .values()
        .find(|t| t.kind == Some(re_mp4::TrackKind::Video))
        .ok_or_else(|| MmError::invalid_input("MP4 has no video track"))?;
    // The sample entry carries the coded size; the track header's size is the
    // display size, which differs when pixels are not square.
    let (codec, width, height) = match &track.trak(&mp4).mdia.minf.stbl.stsd.contents {
        Entry::Vp08(b) => (VideoCodec::Vp8, b.width, b.height),
        Entry::Vp09(b) => (VideoCodec::Vp9, b.width, b.height),
        Entry::Av01(b) => (VideoCodec::Av1, b.width, b.height),
        Entry::Avc1(b) => (VideoCodec::H264, b.width, b.height),
        Entry::Hev1(b) | Entry::Hvc1(b) => (VideoCodec::Hevc, b.width, b.height),
        _ => {
            return Err(MmError::unsupported(format!(
                "unsupported MP4 video codec {}",
                track
                    .codec_string(&mp4)
                    .unwrap_or_else(|| "(unknown)".into())
            )));
        }
    };
    refuse_if_undecodable(codec, require_decodable)?;
    let config = match codec {
        VideoCodec::H264 | VideoCodec::Hevc | VideoCodec::Av1 => track.raw_codec_config(&mp4),
        _ => None,
    };
    let timescale = track.timescale.max(1) as f64;
    // Samples are in decode order, which is what the decoders need.
    let mut packets = Vec::with_capacity(track.samples.len());
    for sample in &track.samples {
        let data = usize::try_from(sample.offset)
            .ok()
            .zip(usize::try_from(sample.size).ok())
            .and_then(|(start, len)| bytes.get(start..start.checked_add(len)?))
            .ok_or_else(|| MmError::invalid_input("MP4 sample lies outside the file"))?;
        let pts = sample.composition_timestamp.max(0) as f64 / timescale;
        packets.push(Packet::new(codec, Cow::Borrowed(data), pts, false)?);
    }
    let mut duration = track.duration as f64 / timescale;
    if duration.is_nan() || duration <= 0.0 {
        duration = track
            .samples
            .iter()
            .map(|s| (s.composition_timestamp.max(0) as f64 + s.duration as f64) / timescale)
            .fold(0.0, f64::max);
    }
    Ok(Track {
        codec,
        width: u32::from(width),
        height: u32::from(height),
        duration,
        config,
        packets,
    })
}

fn demux_matroska(bytes: &[u8], require_decodable: bool) -> Result<Track<'_>> {
    use matroska_demuxer::{Frame, MatroskaFile, TrackType};
    let invalid = |e| MmError::invalid_input_with_source("unreadable WebM / Matroska", e);
    let mut file = MatroskaFile::open(std::io::Cursor::new(bytes)).map_err(invalid)?;
    let entry = file
        .tracks()
        .iter()
        .find(|t| t.track_type() == TrackType::Video)
        .ok_or_else(|| MmError::invalid_input("WebM / Matroska file has no video track"))?;
    let codec = match entry.codec_id() {
        "V_VP8" => VideoCodec::Vp8,
        "V_VP9" => VideoCodec::Vp9,
        "V_AV1" => VideoCodec::Av1,
        "V_MPEG4/ISO/AVC" => VideoCodec::H264,
        "V_MPEGH/ISO/HEVC" => VideoCodec::Hevc,
        other => {
            return Err(MmError::unsupported(format!(
                "unsupported Matroska video codec {other}"
            )));
        }
    };
    refuse_if_undecodable(codec, require_decodable)?;
    let size =
        |v: u64| u32::try_from(v).map_err(|_| MmError::invalid_input("video size out of range"));
    let (width, height) = entry
        .video()
        .map(|v| (v.pixel_width().get(), v.pixel_height().get()))
        .ok_or_else(|| MmError::invalid_input("Matroska video track has no size"))?;
    let (width, height) = (size(width)?, size(height)?);
    let number = entry.track_number().get();
    let config = entry.codec_private().map(<[u8]>::to_vec);
    // Timestamps and block durations are in units of `timestamp_scale`
    // nanoseconds; DefaultDuration is in nanoseconds.
    let scale = file.info().timestamp_scale().get() as f64 / 1e9;
    let default_duration = entry.default_duration().map(|ns| ns.get() as f64 / 1e9);
    let declared = file.info().duration().map(|d| d * scale);

    let mut packets = Vec::new();
    // The latest end time of a shown frame, when the file says how long frames
    // last (the last packet need not be the last frame when frames reorder).
    let mut last_end: Option<f64> = None;
    let mut frame = Frame::default();
    while file.next_frame(&mut frame).map_err(invalid)? {
        if frame.track != number {
            continue;
        }
        let pts = frame.timestamp as f64 * scale;
        let packet = Packet::new(
            codec,
            Cow::Owned(std::mem::take(&mut frame.data)),
            pts,
            frame.is_invisible,
        )?;
        if packet.shown {
            let end = frame
                .duration
                .map(|d| pts + d as f64 * scale)
                .or_else(|| default_duration.map(|d| pts + d));
            if let Some(end) = end {
                last_end = Some(last_end.map_or(end, |e: f64| e.max(end)));
            }
        }
        packets.push(packet);
    }
    // Report time from the first shown frame, as re_mp4 does for MP4, so a
    // clip that starts at 5 s has timestamps from 0 and a 1 s duration, not 6.
    let shown: Vec<f64> = packets.iter().filter(|p| p.shown).map(|p| p.pts).collect();
    let first = shown.iter().copied().fold(f64::INFINITY, f64::min);
    let first = if first.is_finite() { first } else { 0.0 };
    for packet in &mut packets {
        packet.pts = (packet.pts - first).max(0.0);
    }
    let last = shown.iter().copied().fold(first, f64::max) - first;
    // The segment duration and block end times are measured from the segment
    // start, so they shift with the timestamps.
    let duration = matroska_duration(
        declared.map(|d| d - first),
        last_end.map(|end| end - first),
        last,
        shown.len(),
    );
    Ok(Track {
        codec,
        width,
        height,
        duration,
        config,
        packets,
    })
}

/// A Matroska clip's duration from the segment duration or the latest frame
/// end the blocks give, whichever comes first and ends after the last shown
/// frame starts (at `last`). Otherwise the last frame is assumed to last as
/// long as the average gap between `frames` frames.
fn matroska_duration(
    declared: Option<f64>,
    last_end: Option<f64>,
    last: f64,
    frames: usize,
) -> f64 {
    declared
        .filter(|d| *d > last)
        .or(last_end.filter(|end| *end > last))
        .unwrap_or_else(|| last + last / frames.saturating_sub(1).max(1) as f64)
}

// ---------------------------------------------------------------------------
// YUV to RGB
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Matrix {
    Bt601,
    Bt709,
}

/// One decoded 8-bit 4:2:0 picture in NV12 layout: a luma plane, then one
/// plane of interleaved U/V samples, each with its own row stride.
struct Nv12<'a> {
    width: u32,
    height: u32,
    y: &'a [u8],
    uv: &'a [u8],
    y_stride: usize,
    uv_stride: usize,
    matrix: Matrix,
    full_range: bool,
}

impl Nv12<'_> {
    /// Append `height * width * 3` RGB bytes. Chroma is sampled by nearest
    /// neighbour (each 2x2 block shares one chroma sample).
    fn to_rgb(&self, out: &mut Vec<u8>) {
        // (Kr, Kb) per ITU-R BT.601 / BT.709.
        let (kr, kb) = match self.matrix {
            Matrix::Bt601 => (0.299_f32, 0.114_f32),
            Matrix::Bt709 => (0.2126_f32, 0.0722_f32),
        };
        let kg = 1.0 - kr - kb;
        let (y_scale, y_off, c_scale) = if self.full_range {
            (1.0_f32, 0.0_f32, 1.0_f32)
        } else {
            (255.0 / 219.0, 16.0, 255.0 / 224.0)
        };
        let clamp = |v: f32| v.round().clamp(0.0, 255.0) as u8;
        out.reserve(self.width as usize * self.height as usize * 3);
        for row in 0..self.height as usize {
            let y_row = &self.y[row * self.y_stride..];
            let uv_row = &self.uv[(row / 2) * self.uv_stride..];
            for (col, &luma) in y_row[..self.width as usize].iter().enumerate() {
                let c = (col / 2) * 2;
                let y = (f32::from(luma) - y_off) * y_scale;
                let u = (f32::from(uv_row[c]) - 128.0) * c_scale;
                let v = (f32::from(uv_row[c + 1]) - 128.0) * c_scale;
                let r = y + 2.0 * (1.0 - kr) * v;
                let b = y + 2.0 * (1.0 - kb) * u;
                let g = y - (2.0 * kb * (1.0 - kb) * u + 2.0 * kr * (1.0 - kr) * v) / kg;
                out.extend_from_slice(&[clamp(r), clamp(g), clamp(b)]);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Decoder input
// ---------------------------------------------------------------------------

/// The frames of a VP9 packet: a superframe's frames (its index sits in the
/// packet's last bytes), or the packet itself.
fn vp9_frames(packet: &[u8]) -> Vec<&[u8]> {
    let single = || vec![packet];
    let Some(&marker) = packet.last() else {
        return single();
    };
    if marker & 0xE0 != 0xC0 {
        return single();
    }
    let count = usize::from(marker & 0x07) + 1;
    let size_bytes = usize::from((marker >> 3) & 0x03) + 1;
    let index_len = 2 + size_bytes * count;
    let Some(index_start) = packet.len().checked_sub(index_len) else {
        return single();
    };
    if packet[index_start] != marker {
        return single();
    }
    let mut frames = Vec::with_capacity(count);
    let mut offset = 0;
    for i in 0..count {
        let field = &packet[index_start + 1 + i * size_bytes..][..size_bytes];
        let size = field
            .iter()
            .rev()
            .fold(0usize, |n, b| (n << 8) | usize::from(*b));
        match packet
            .get(offset..offset + size)
            .filter(|_| offset + size <= index_start)
        {
            Some(frame) => frames.push(frame),
            None => return single(),
        }
        offset += size;
    }
    frames
}

/// Whether a VP9 packet outputs a frame: one of its frames has `show_frame`
/// set. Unparseable headers count as shown, so the decoder's own checks
/// decide. Refused, because they break the one-output-per-packet mapping: a
/// packet that shows more than one frame, and `show_existing_frame` (NVDEC
/// outputs nothing for a standalone one).
fn vp9_packet_displays(packet: &[u8]) -> Result<bool> {
    let mut shown = 0;
    for frame in vp9_frames(packet) {
        let bit = |n: usize| frame.get(n / 8).map(|b| (b >> (7 - n % 8)) & 1);
        // (show_existing_frame, show_frame)
        let header = || -> Option<(bool, bool)> {
            if bit(0)? != 1 || bit(1)? != 0 {
                return None;
            }
            let profile = bit(2)? | (bit(3)? << 1);
            let pos = 4 + usize::from(profile == 3);
            if bit(pos)? == 1 {
                return Some((true, false));
            }
            Some((false, bit(pos + 2)? == 1)) // after frame_type
        };
        match header() {
            Some((true, _)) => {
                return Err(MmError::unsupported(
                    "VP9 show_existing_frame packets are not supported",
                ));
            }
            Some((false, show)) => shown += usize::from(show),
            None => shown += 1,
        }
    }
    if shown > 1 {
        return Err(MmError::unsupported(
            "VP9 packets that show more than one frame are not supported",
        ));
    }
    Ok(shown == 1)
}

/// The colour a VP9 keyframe declares: `Some(Ok((matrix, full_range)))` for a
/// keyframe, `None` for any other frame, and an error for a colour space this
/// crate does not convert (SMPTE 240M, BT.2020, sRGB). The first frame of a
/// superframe is the one read.
fn vp9_keyframe_colour(frame: &[u8]) -> Option<Result<(Matrix, bool)>> {
    // VP9 uncompressed header: frame_marker (2), profile low/high bits,
    // reserved bit for profile 3, show_existing_frame, frame_type,
    // show_frame, error_resilient_mode, then for a keyframe the sync code
    // (0x49 0x83 0x42) and the colour config.
    let bit = |n: usize| frame.get(n / 8).map(|b| (b >> (7 - n % 8)) & 1);
    if bit(0)? != 1 || bit(1)? != 0 {
        return None;
    }
    let profile = bit(2)? | (bit(3)? << 1);
    let mut pos = 4 + usize::from(profile == 3);
    if bit(pos)? == 1 {
        return None; // show_existing_frame
    }
    if bit(pos + 1)? != 0 {
        return None; // not a keyframe
    }
    pos += 4;
    let bytes =
        |at: usize| -> Option<u8> { (0..8).try_fold(0u8, |v, i| Some((v << 1) | bit(at + i)?)) };
    if (bytes(pos)?, bytes(pos + 8)?, bytes(pos + 16)?) != (0x49, 0x83, 0x42) {
        return None;
    }
    pos += 24 + usize::from(profile >= 2); // ten_or_twelve_bit
    let color_space = (bit(pos)? << 2) | (bit(pos + 1)? << 1) | bit(pos + 2)?;
    let full_range = color_space != 7 && bit(pos + 3)? == 1;
    Some(match color_space {
        0 | 1 | 3 => Ok((Matrix::Bt601, full_range)),
        2 => Ok((Matrix::Bt709, full_range)),
        other => Err(MmError::unsupported(format!(
            "VP9 colour space {other} is not supported"
        ))),
    })
}

/// Annex B start code.
const START_CODE: [u8; 4] = [0, 0, 0, 1];

/// Append each parameter set as an Annex B NAL unit, reading `count` sets of
/// (16-bit length, bytes) from `config` at `*pos`.
fn copy_parameter_sets(
    config: &[u8],
    pos: &mut usize,
    count: usize,
    out: &mut Vec<u8>,
) -> Result<()> {
    let malformed = || MmError::invalid_input("malformed codec configuration record");
    for _ in 0..count {
        let len = config.get(*pos..*pos + 2).ok_or_else(malformed)?;
        let len = usize::from(u16::from_be_bytes([len[0], len[1]]));
        let nal = config.get(*pos + 2..*pos + 2 + len).ok_or_else(malformed)?;
        out.extend_from_slice(&START_CODE);
        out.extend_from_slice(nal);
        *pos += 2 + len;
    }
    Ok(())
}

/// From an `avcC` record: its NAL length size and SPS/PPS as Annex B.
fn avcc_parameter_sets(avcc: &[u8]) -> Result<(usize, Vec<u8>)> {
    let malformed = || MmError::invalid_input("malformed avcC record");
    if avcc.len() < 7 || avcc[0] != 1 {
        return Err(malformed());
    }
    let length_size = usize::from(avcc[4] & 0x3) + 1;
    let mut out = Vec::new();
    let mut pos = 6;
    copy_parameter_sets(avcc, &mut pos, usize::from(avcc[5] & 0x1F), &mut out)?;
    let pps = *avcc.get(pos).ok_or_else(malformed)?;
    pos += 1;
    copy_parameter_sets(avcc, &mut pos, usize::from(pps), &mut out)?;
    Ok((length_size, out))
}

/// From an `hvcC` record: its NAL length size and VPS/SPS/PPS (and any other
/// arrays) as Annex B.
fn hvcc_parameter_sets(hvcc: &[u8]) -> Result<(usize, Vec<u8>)> {
    let malformed = || MmError::invalid_input("malformed hvcC record");
    if hvcc.len() < 23 {
        return Err(malformed());
    }
    let length_size = usize::from(hvcc[21] & 0x3) + 1;
    let mut out = Vec::new();
    let mut pos = 23;
    for _ in 0..hvcc[22] {
        // One byte of array flags and NAL type, then a 16-bit count.
        let count = hvcc.get(pos + 1..pos + 3).ok_or_else(malformed)?;
        let count = usize::from(u16::from_be_bytes([count[0], count[1]]));
        pos += 3;
        copy_parameter_sets(hvcc, &mut pos, count, &mut out)?;
    }
    Ok((length_size, out))
}

/// What goes in front of the first packet, and the NAL length size when the
/// samples are length-prefixed (H.264 / HEVC) and must become Annex B.
fn bitstream_prefix(track: &Track<'_>) -> Result<(Vec<u8>, Option<usize>)> {
    let config = track.config.as_deref();
    let missing =
        |what: &str| MmError::invalid_input(format!("{what} track has no codec configuration"));
    match track.codec {
        VideoCodec::H264 => {
            let (length_size, sets) = avcc_parameter_sets(config.ok_or_else(|| missing("H.264"))?)?;
            Ok((sets, Some(length_size)))
        }
        VideoCodec::Hevc => {
            let (length_size, sets) = hvcc_parameter_sets(config.ok_or_else(|| missing("HEVC"))?)?;
            Ok((sets, Some(length_size)))
        }
        // `av1C` is a 4-byte header followed by the sequence header OBUs, which
        // the first sample may omit.
        VideoCodec::Av1 => Ok((
            config.and_then(|c| c.get(4..)).unwrap_or_default().to_vec(),
            None,
        )),
        _ => Ok((Vec::new(), None)),
    }
}

/// Rewrite length-prefixed NAL units as Annex B, appending to `out`.
fn length_prefixed_to_annexb(sample: &[u8], length_size: usize, out: &mut Vec<u8>) -> Result<()> {
    let malformed = || MmError::invalid_input("malformed length-prefixed video sample");
    let mut pos = 0;
    while pos < sample.len() {
        let len = sample.get(pos..pos + length_size).ok_or_else(malformed)?;
        let len = len.iter().fold(0usize, |n, b| (n << 8) | usize::from(*b));
        pos += length_size;
        let nal = sample
            .get(pos..pos.checked_add(len).ok_or_else(malformed)?)
            .ok_or_else(malformed)?;
        out.extend_from_slice(&START_CODE);
        out.extend_from_slice(nal);
        pos += len;
    }
    Ok(())
}

/// The bytes to hand the decoder for `packet`: the prefix before the first
/// one, length-prefixed NAL units rewritten as Annex B. Borrows the packet
/// when nothing changes, otherwise builds into `buffer`.
fn to_decoder_input<'b>(
    packet: &'b Packet<'_>,
    prefix: &[u8],
    length_size: Option<usize>,
    first: bool,
    buffer: &'b mut Vec<u8>,
) -> Result<&'b [u8]> {
    if length_size.is_none() && (!first || prefix.is_empty()) {
        return Ok(&packet.data);
    }
    buffer.clear();
    if first {
        buffer.extend_from_slice(prefix);
    }
    match length_size {
        Some(size) => length_prefixed_to_annexb(&packet.data, size, buffer)?,
        None => buffer.extend_from_slice(&packet.data),
    }
    Ok(buffer)
}

#[cfg(test)]
mod tests {
    use super::*;

    // GPU tests are `#[ignore]`d: they need an NVIDIA driver. Run them with
    // `cargo test --features video-decode -- --ignored`.

    fn fixture(name: &str) -> Vec<u8> {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/video_decode")
            .join(name);
        std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
    }

    /// The solid colour of frame `i` in the 256x256 fixtures (see generate.py).
    fn color(i: u64) -> [u8; 3] {
        let i = i as u8;
        [40 + 16 * i, 200 - 12 * i, 90 + (i % 3) * 40]
    }

    fn mean(frame: &[u8], c: usize) -> u64 {
        let sum: u64 = frame.iter().skip(c).step_by(3).map(|&v| u64::from(v)).sum();
        sum / (frame.len() as u64 / 3)
    }

    fn assert_frame_color(frames: &VideoFrames, at: usize, expected: [u8; 3]) {
        let frame = frames.frame(at).unwrap();
        for (c, want) in expected.iter().enumerate() {
            let got = mean(frame, c);
            assert!(
                got.abs_diff(u64::from(*want)) <= 4,
                "frame {at} channel {c}: mean {got}, expected {want}"
            );
        }
    }

    fn track_of(name: &str) -> (Vec<u8>, VideoCodec, Option<Vec<u8>>) {
        let bytes = fixture(name);
        let (codec, config) = {
            let track = demux(&bytes, false).unwrap();
            (track.codec, track.config.clone())
        };
        (bytes, codec, config)
    }

    // -- container side: runs everywhere ----------------------------------

    #[test]
    fn sample_indices_spread_evenly() {
        assert_eq!(sample_indices(10, 1), vec![0]);
        assert_eq!(sample_indices(10, 2), vec![0, 9]);
        assert_eq!(sample_indices(10, 4), vec![0, 3, 6, 9]);
        assert_eq!(sample_indices(10, 10), (0..10).collect::<Vec<_>>());
        assert_eq!(sample_indices(5, 3), vec![0, 2, 4]);
        // linspace(0, 1, 3) = [0, 0.5, 1]; 0.5 rounds to even.
        assert_eq!(sample_indices(2, 3), vec![0, 0, 1]);
    }

    #[test]
    fn probes_every_fixture() {
        for (name, codec) in [
            ("vp8.webm", VideoCodec::Vp8),
            ("vp9.webm", VideoCodec::Vp9),
            ("vp9.mp4", VideoCodec::Vp9),
            ("av1.mp4", VideoCodec::Av1),
            ("h264.mp4", VideoCodec::H264),
            ("hevc.mp4", VideoCodec::Hevc),
        ] {
            let info = probe(&fixture(name)).unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(info.codec, codec, "{name}");
            assert_eq!((info.width, info.height), (256, 256), "{name}");
            assert_eq!(info.frame_count, 10, "{name}");
            assert!(
                (info.duration - 1.0).abs() < 0.05,
                "{name}: {}",
                info.duration
            );
            assert!((info.fps - 10.0).abs() < 0.5, "{name}: {}", info.fps);
        }
    }

    #[test]
    fn non_square_pixels_report_the_coded_size() {
        // The track header says 512x256 (the display size, 2:1 pixels); the
        // sample entry and the bitstream are 256x256.
        let info = probe(&fixture("vp9_sar.mp4")).unwrap();
        assert_eq!((info.width, info.height), (256, 256));
    }

    #[test]
    fn vp8_is_refused_before_decoding() {
        let err =
            decode_sampled(&fixture("vp8.webm"), 1, &VideoDecodeLimits::default()).unwrap_err();
        assert!(matches!(err, MmError::Unsupported { .. }), "{err:?}");
        assert!(format!("{err}").contains("VP8"), "{err}");
    }

    #[test]
    fn rejects_garbage_and_truncated_containers() {
        for bytes in [&b""[..], b"not a video at all", &[0u8; 64]] {
            let err = probe(bytes).unwrap_err();
            assert!(matches!(err, MmError::InvalidInput { .. }), "{err:?}");
        }
        for name in ["vp9.mp4", "vp9.webm"] {
            let r = probe(&fixture(name)[..12]);
            assert!(r.is_err(), "{name}: {r:?}");
        }
    }

    /// Patch the 32-bit big-endian field `offset` bytes after the first `fourcc`.
    fn patch_mp4(bytes: &mut [u8], fourcc: &[u8; 4], offset: usize, value: u32) {
        let at = bytes.windows(4).position(|w| w == fourcc).unwrap() + offset;
        bytes[at..at + 4].copy_from_slice(&value.to_be_bytes());
    }

    #[test]
    fn malformed_mp4_sample_tables_are_errors_not_panics() {
        // Each of these panics inside re_mp4 (index or arithmetic overflow);
        // `demux` must turn that into an error.
        for (fourcc, offset) in [(b"stco", 8), (b"stsc", 8)] {
            let mut bytes = fixture("vp9.mp4");
            patch_mp4(&mut bytes, fourcc, offset, 0);
            let r = probe(&bytes);
            assert!(
                matches!(r, Err(MmError::InvalidInput { .. })),
                "{fourcc:?}: {r:?}"
            );
        }
        // A track duration that ends before the last sample starts makes
        // re_mp4 subtract into an underflow.
        let mut bytes = fixture("vp9.mp4");
        // mdhd version 0: version/flags, creation, modification, timescale, duration.
        patch_mp4(&mut bytes, b"mdhd", 20, 1);
        let r = probe(&bytes);
        assert!(matches!(r, Err(MmError::InvalidInput { .. })), "{r:?}");
    }

    #[test]
    fn sample_counts_the_file_cannot_back_are_refused_up_front() {
        // A small file asking for 2^31 samples would make re_mp4 allocate
        // ~15 GB; the table check must refuse it first.
        for (fourcc, offset) in [(b"stsz", 12), (b"stts", 12)] {
            let mut bytes = fixture("vp9.mp4");
            patch_mp4(&mut bytes, fourcc, offset, 0x7FFF_FFFF);
            let started = std::time::Instant::now();
            let r = probe(&bytes);
            assert!(
                matches!(r, Err(MmError::InvalidInput { .. })),
                "{fourcc:?}: {r:?}"
            );
            assert!(started.elapsed() < std::time::Duration::from_secs(1));
        }
    }

    #[test]
    fn matroska_time_starts_at_the_first_frame() {
        let info = probe(&fixture("vp9_offset.webm")).unwrap();
        assert!((info.duration - 1.0).abs() < 0.05, "{}", info.duration);
        assert!((info.fps - 10.0).abs() < 0.5, "{}", info.fps);
    }

    #[test]
    fn a_truncated_file_is_an_error_not_a_short_result() {
        // Losing media data fails in the container, before any decoder runs.
        let full = fixture("h264.mp4");
        let r = decode_sampled(
            &full[..full.len() * 3 / 4],
            10,
            &VideoDecodeLimits::default(),
        );
        assert!(matches!(r, Err(MmError::InvalidInput { .. })), "{r:?}");
    }

    #[test]
    fn hidden_vp9_frames_do_not_count_as_displayed() {
        // Profile 0 headers: frame marker 10, profile 00, show_existing 0,
        // frame_type, show_frame, error_resilient.
        let shown = [0b1000_0011_u8, 0, 0];
        let hidden = [0b1000_0001_u8, 0, 0];
        let existing = [0b1000_1000_u8];
        assert!(vp9_packet_displays(&shown).unwrap());
        assert!(!vp9_packet_displays(&hidden).unwrap());
        // A superframe of a hidden frame then a shown one displays; one of two
        // hidden frames does not. Index: marker 0b110_00_001 (2 frames, 1-byte
        // sizes), sizes, marker.
        let superframe = |a: &[u8], b: &[u8]| {
            let marker = 0b1100_0001_u8;
            let mut p = [a, b].concat();
            p.extend_from_slice(&[marker, a.len() as u8, b.len() as u8, marker]);
            p
        };
        assert!(vp9_packet_displays(&superframe(&hidden, &shown)).unwrap());
        assert!(!vp9_packet_displays(&superframe(&hidden, &hidden)).unwrap());
        assert_eq!(vp9_frames(&superframe(&hidden, &shown)).len(), 2);
        // Packets that cannot map to exactly one output are refused.
        for packet in [existing.to_vec(), superframe(&shown, &shown)] {
            let r = vp9_packet_displays(&packet);
            assert!(matches!(r, Err(MmError::Unsupported { .. })), "{r:?}");
        }
        // Real clips: every packet displays, superframes included.
        for name in ["vp9.webm", "vp9_altref.webm"] {
            let bytes = fixture(name);
            let track = demux(&bytes, false).unwrap();
            assert!(
                track.packets.iter().all(|p| p.displays && p.shown),
                "{name}"
            );
        }
        let bytes = fixture("vp9_altref.webm");
        let track = demux(&bytes, false).unwrap();
        assert!(track.packets.iter().any(|p| vp9_frames(&p.data).len() > 1));
    }

    #[test]
    fn sample_counts_are_bounded_across_tracks() {
        // Two fixed-size stsz tables, each declaring 60% of the file's length
        // in samples: each fits on its own, together they do not.
        let stsz = |count: u32| {
            let mut b = 20u32.to_be_bytes().to_vec();
            b.extend_from_slice(b"stsz");
            b.extend_from_slice(&[0; 4]); // version, flags
            b.extend_from_slice(&1u32.to_be_bytes()); // every sample is 1 byte
            b.extend_from_slice(&count.to_be_bytes());
            b
        };
        let len = 1000u32;
        let free = |size: u32| {
            let mut b = size.to_be_bytes().to_vec();
            b.extend_from_slice(b"free");
            b.resize(size as usize, 0);
            b
        };
        let one = [stsz(600), free(len - 20)].concat();
        assert!(validate_mp4_tables(&one).is_ok());
        let two = [stsz(600), stsz(600), free(len - 40)].concat();
        let r = validate_mp4_tables(&two);
        assert!(matches!(r, Err(MmError::InvalidInput { .. })), "{r:?}");
    }

    #[test]
    fn matroska_duration_must_end_after_the_last_frame() {
        // Declared or block-derived ends are used when they pass the last
        // frame's start; otherwise the last gap is estimated.
        assert_eq!(matroska_duration(Some(2.0), Some(1.5), 1.0, 11), 2.0);
        assert_eq!(matroska_duration(None, Some(1.1), 1.0, 11), 1.1);
        // Only the first frame had a BlockDuration: its end (0.1 s) is
        // before the last frame (1.0 s), so it is not the clip's duration.
        assert!((matroska_duration(None, Some(0.1), 1.0, 11) - 1.1).abs() < 1e-9);
        assert!((matroska_duration(Some(0.5), None, 1.0, 11) - 1.1).abs() < 1e-9);
    }

    #[test]
    fn zero_frames_is_refused() {
        let r = decode_sampled(&fixture("h264.mp4"), 0, &VideoDecodeLimits::default());
        assert!(matches!(r, Err(MmError::InvalidInput { .. })), "{r:?}");
    }

    #[test]
    fn frame_counts_and_limits_are_checked_before_decoding() {
        // These fail before any driver call, so they hold without a GPU.
        let bytes = fixture("h264.mp4");
        let limits = VideoDecodeLimits::default();
        let r = decode_sampled(&bytes, 11, &limits);
        assert!(matches!(r, Err(MmError::InvalidInput { .. })), "{r:?}");
        let small = VideoDecodeLimits {
            max_width: 128,
            ..limits
        };
        let r = decode_sampled(&bytes, 1, &small);
        assert!(matches!(r, Err(MmError::LimitExceeded { .. })), "{r:?}");
        let tight = VideoDecodeLimits {
            max_output_bytes: 256 * 256 * 3,
            ..limits
        };
        let r = decode_sampled(&bytes, 2, &tight);
        assert!(matches!(r, Err(MmError::LimitExceeded { .. })), "{r:?}");
    }

    fn nv12_pixel(y: u8, u: u8, v: u8, matrix: Matrix, full_range: bool) -> Vec<u8> {
        let mut out = Vec::new();
        Nv12 {
            width: 1,
            height: 1,
            y: &[y],
            uv: &[u, v],
            y_stride: 1,
            uv_stride: 2,
            matrix,
            full_range,
        }
        .to_rgb(&mut out);
        out
    }

    #[test]
    fn nv12_to_rgb_matches_the_reference_matrices() {
        // Mid-grey in limited range stays grey.
        assert_eq!(
            nv12_pixel(126, 128, 128, Matrix::Bt601, false),
            vec![128, 128, 128]
        );
        // Pure red: BT.601 (Y=81, Cb=90, Cr=240); BT.709 (Y=63, Cb=102, Cr=240).
        for (y, u, v, m) in [(81, 90, 240, Matrix::Bt601), (63, 102, 240, Matrix::Bt709)] {
            let red = nv12_pixel(y, u, v, m, false);
            assert!(
                red[0] >= 250 && red[1] <= 4 && red[2] <= 4,
                "{m:?}: {red:?}"
            );
        }
        // Full range: Y is used as is.
        assert_eq!(
            nv12_pixel(200, 128, 128, Matrix::Bt601, true),
            vec![200, 200, 200]
        );
    }

    #[test]
    fn nv12_reads_interleaved_chroma_per_2x2_block() {
        // 4x2 picture: the left 2x2 block is red, the right one blue; a row
        // stride wider than the picture must be skipped.
        let y = [81, 81, 41, 41, 0, 0, 81, 81, 41, 41, 0, 0];
        let uv = [90, 240, 240, 110, 0, 0];
        let mut out = Vec::new();
        Nv12 {
            width: 4,
            height: 2,
            y: &y,
            uv: &uv,
            y_stride: 6,
            uv_stride: 6,
            matrix: Matrix::Bt601,
            full_range: false,
        }
        .to_rgb(&mut out);
        for row in 0..2 {
            for col in 0..4 {
                let px = &out[(row * 4 + col) * 3..][..3];
                if col < 2 {
                    assert!(px[0] > 240 && px[2] < 16, "red at {row},{col}: {px:?}");
                } else {
                    assert!(px[2] > 240 && px[0] < 16, "blue at {row},{col}: {px:?}");
                }
            }
        }
    }

    #[test]
    fn h264_parameter_sets_come_from_the_avcc_record() {
        let (_, codec, config) = track_of("h264.mp4");
        assert_eq!(codec, VideoCodec::H264);
        let (length_size, sets) = avcc_parameter_sets(&config.unwrap()).unwrap();
        assert_eq!(length_size, 4);
        // Annex B: an SPS (NAL type 7) then a PPS (8), each after a start code.
        assert_eq!(&sets[..4], &START_CODE);
        assert_eq!(sets[4] & 0x1F, 7);
        let pps = sets[4..].windows(4).position(|w| w == START_CODE).unwrap() + 8;
        assert_eq!(sets[pps] & 0x1F, 8);
        assert!(avcc_parameter_sets(&[1, 0, 0, 0, 0xFF, 0xE1, 0, 9]).is_err());
    }

    #[test]
    fn hevc_parameter_sets_come_from_the_hvcc_record() {
        let (_, codec, config) = track_of("hevc.mp4");
        assert_eq!(codec, VideoCodec::Hevc);
        let (length_size, sets) = hvcc_parameter_sets(&config.unwrap()).unwrap();
        assert_eq!(length_size, 4);
        // VPS (32), SPS (33) and PPS (34), each after a start code.
        let types: Vec<u8> = sets
            .windows(5)
            .filter(|w| w[..4] == START_CODE)
            .map(|w| (w[4] >> 1) & 0x3F)
            .collect();
        for t in [32, 33, 34] {
            assert!(types.contains(&t), "NAL type {t} missing from {types:?}");
        }
        assert!(hvcc_parameter_sets(&[0; 22]).is_err());
    }

    #[test]
    fn vp9_colour_is_read_from_the_keyframe_header() {
        let first = |name: &str| {
            let bytes = fixture(name);
            let track = demux(&bytes, false).unwrap();
            vp9_keyframe_colour(&track.packets[0].data).map(|r| r.unwrap())
        };
        assert_eq!(first("vp9_709.webm"), Some((Matrix::Bt709, false)));
        assert_eq!(first("vp9.webm"), Some((Matrix::Bt601, false)));
        // An inter frame carries no colour config.
        let bytes = fixture("vp9.webm");
        let track = demux(&bytes, false).unwrap();
        assert_eq!(
            vp9_keyframe_colour(&track.packets[1].data).map(|r| r.is_ok()),
            None
        );
        assert!(vp9_keyframe_colour(&[]).is_none());
    }

    #[test]
    fn length_prefixed_samples_become_annex_b() {
        let sample = [0, 0, 0, 2, 0x65, 0x88, 0, 0, 0, 1, 0x41];
        let mut out = Vec::new();
        length_prefixed_to_annexb(&sample, 4, &mut out).unwrap();
        assert_eq!(out, [0, 0, 0, 1, 0x65, 0x88, 0, 0, 0, 1, 0x41]);
        // A length running past the sample is an error, not a panic.
        assert!(length_prefixed_to_annexb(&[0, 0, 0, 9, 1], 4, &mut Vec::new()).is_err());
        assert!(length_prefixed_to_annexb(&[0, 0], 4, &mut Vec::new()).is_err());
    }

    #[test]
    fn the_first_packet_carries_the_parameter_sets() {
        let bytes = fixture("h264.mp4");
        let track = demux(&bytes, true).unwrap();
        let (prefix, length_size) = bitstream_prefix(&track).unwrap();
        let mut buffer = Vec::new();
        let first = to_decoder_input(&track.packets[0], &prefix, length_size, true, &mut buffer)
            .unwrap()
            .to_vec();
        assert!(first.starts_with(&prefix));
        let second =
            to_decoder_input(&track.packets[1], &prefix, length_size, false, &mut buffer).unwrap();
        assert!(second.starts_with(&START_CODE) && !second.starts_with(&prefix));
        // VP9 needs no rewriting: the packet is passed through untouched.
        let bytes = fixture("vp9.webm");
        let track = demux(&bytes, true).unwrap();
        let (prefix, length_size) = bitstream_prefix(&track).unwrap();
        let packet = &track.packets[1];
        let input = to_decoder_input(packet, &prefix, length_size, false, &mut buffer).unwrap();
        assert_eq!(input, &packet.data[..]);
    }

    // -- NVDEC: needs a GPU ------------------------------------------------

    #[test]
    #[ignore = "needs an NVIDIA GPU and driver (NVDEC)"]
    fn decodes_every_codec_in_display_order() {
        for name in [
            "h264.mp4",
            "hevc.mp4",
            "vp9.webm",
            "vp9.mp4",
            "av1.mp4",
            "vp9_sar.mp4",
        ] {
            let frames = decode_sampled(&fixture(name), 10, &VideoDecodeLimits::default())
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(frames.frame_count(), 10, "{name}");
            for i in 0..10 {
                assert_frame_color(&frames, i, color(i as u64));
            }
            let ts = frames.timestamps();
            assert!((ts[3] - 0.3).abs() < 0.01, "{name}: {ts:?}");
        }
    }

    #[test]
    #[ignore = "needs an NVIDIA GPU and driver (NVDEC)"]
    fn samples_evenly_and_keeps_their_timestamps() {
        for name in ["h264.mp4", "vp9.webm"] {
            let frames = decode_sampled(&fixture(name), 4, &VideoDecodeLimits::default()).unwrap();
            assert_eq!(frames.frame_count(), 4);
            for (at, index) in [0, 3, 6, 9].into_iter().enumerate() {
                assert_frame_color(&frames, at, color(index));
                assert!((frames.timestamps()[at] - index as f64 * 0.1).abs() < 0.01);
            }
        }
    }

    #[test]
    #[ignore = "needs an NVIDIA GPU and driver (NVDEC)"]
    fn h264_frame_matches_the_reference_conversion() {
        // h264_quad.rgb is a software decoder's planes for the same clip,
        // converted with the same formula; H.264 decoding is exact, so a pitch,
        // chroma-offset or layout mistake shows up directly.
        let frames =
            decode_sampled(&fixture("h264_quad.mp4"), 1, &VideoDecodeLimits::default()).unwrap();
        let expected = fixture("h264_quad.rgb");
        let got = frames.frame(0).unwrap();
        assert_eq!(got.len(), expected.len());
        let worst = got.iter().zip(&expected).map(|(a, b)| a.abs_diff(*b)).max();
        assert!(worst <= Some(1), "max channel difference {worst:?}");
    }

    #[test]
    #[ignore = "needs an NVIDIA GPU and driver (NVDEC)"]
    fn bt709_tagged_streams_convert_with_bt709() {
        // Encoded with the BT.709 matrix and tagged so; read as BT.601 the
        // colour would shift by more than 15 (generate.py checks that). VP9
        // reports its own colour codes, H.264 the H.273 ones.
        for name in ["vp9_709.webm", "h264_709.mp4"] {
            let frames = decode_sampled(&fixture(name), 1, &VideoDecodeLimits::default())
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_frame_color(&frames, 0, [220, 40, 200]);
        }
    }

    #[test]
    #[ignore = "needs an NVIDIA GPU and driver (NVDEC)"]
    fn invisible_frames_are_decoded_but_not_shown() {
        // vp9_invisible.webm is vp9_texture.webm with its third block flagged
        // invisible. The texture moves, so frames are predicted from their
        // predecessor: the shown frames must equal the original clip's frames
        // bit for bit, which only holds if the hidden frame was decoded.
        let limits = VideoDecodeLimits::default();
        let full = decode_sampled(&fixture("vp9_texture.webm"), 10, &limits).unwrap();
        let bytes = fixture("vp9_invisible.webm");
        assert_eq!(probe(&bytes).unwrap().frame_count, 9);
        let hidden = decode_sampled(&bytes, 9, &limits).unwrap();
        let shown: Vec<usize> = (0..10).filter(|i| *i != 2).collect();
        for (at, index) in shown.into_iter().enumerate() {
            assert!(
                hidden.frame(at) == full.frame(index),
                "shown frame {at} differs from original frame {index}"
            );
        }
    }

    #[test]
    #[ignore = "needs an NVIDIA GPU and driver (NVDEC)"]
    fn frames_larger_than_the_container_are_refused_before_allocation() {
        // A VP9 keyframe claiming 16384x16384 in a 256x256 container, and an
        // AV1 stream of 256x256 whose container claims 128x128.
        for name in ["vp9_huge.webm", "av1_small_container.mp4"] {
            let r = decode_sampled(&fixture(name), 1, &VideoDecodeLimits::default());
            assert!(
                matches!(r, Err(MmError::LimitExceeded { .. })),
                "{name}: {r:?}"
            );
        }
    }

    #[test]
    #[ignore = "needs an NVIDIA GPU and driver (NVDEC)"]
    fn matroska_timestamps_start_at_zero() {
        let frames = decode_sampled(
            &fixture("vp9_offset.webm"),
            2,
            &VideoDecodeLimits::default(),
        )
        .unwrap();
        assert_eq!(frames.timestamps()[0], 0.0);
        assert!(
            (frames.timestamps()[1] - 0.9).abs() < 0.01,
            "{:?}",
            frames.timestamps()
        );
    }

    #[test]
    #[ignore = "needs an NVIDIA GPU and driver (NVDEC)"]
    fn odd_sized_frames_match_the_reference_conversion() {
        // 130x129: NVDEC needs an even output size, so the frame is decoded
        // one row taller and cropped; the chroma plane starts after that even
        // row count. Compared with a software decoder's planes.
        let frames =
            decode_sampled(&fixture("vp9_odd.webm"), 1, &VideoDecodeLimits::default()).unwrap();
        assert_eq!((frames.width(), frames.height()), (130, 129));
        let expected = fixture("vp9_odd.rgb");
        let got = frames.frame(0).unwrap();
        assert_eq!(got.len(), expected.len());
        let worst = got.iter().zip(&expected).map(|(a, b)| a.abs_diff(*b)).max();
        assert!(worst <= Some(1), "max channel difference {worst:?}");
    }

    #[test]
    #[ignore = "needs an NVIDIA GPU and driver (NVDEC)"]
    fn alt_ref_superframes_keep_frames_and_timestamps_together() {
        // Frame i of vp9_altref.webm has a grey bar of 20 + 10 i (generate.py)
        // and starts at i / 10 s. Superframes carrying a hidden alt-ref make
        // NVDEC's timestamps slip by one, so every frame is checked against
        // its own timestamp.
        let frames = decode_sampled(
            &fixture("vp9_altref.webm"),
            20,
            &VideoDecodeLimits::default(),
        )
        .unwrap();
        for (i, ts) in frames.timestamps().iter().enumerate() {
            assert!(
                (ts - i as f64 * 0.1).abs() < 0.01,
                "{:?}",
                frames.timestamps()
            );
            let rgb = frames.frame(i).unwrap();
            let bar: Vec<u64> = (0..128)
                .flat_map(|y| (0..32).map(move |x| (y * 128 + x) * 3))
                .map(|at| u64::from(rgb[at]))
                .collect();
            let grey = bar.iter().sum::<u64>() / bar.len() as u64;
            assert!(grey.abs_diff(20 + 10 * i as u64) <= 3, "frame {i}: {grey}");
        }
    }

    #[test]
    #[ignore = "needs an NVIDIA GPU and driver (NVDEC)"]
    fn odd_sized_av1_is_refused() {
        // NVDEC would scale it to an even size rather than crop it.
        let r = decode_sampled(&fixture("av1_odd.mp4"), 1, &VideoDecodeLimits::default());
        assert!(matches!(r, Err(MmError::Unsupported { .. })), "{r:?}");
    }
}
