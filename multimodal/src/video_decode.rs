// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Video decoding to sampled RGB frames on NVIDIA GPUs through NVDEC, with
//! VP8 and VP9 also on the CPU through libvpx.
//!
//! [`probe`] reads a clip's container and reports its codec, coded size,
//! duration and frame count. [`decode_frames`] decodes it and keeps the frames
//! the caller names, each resized to one size if asked, as [`VideoFrames`];
//! [`decode_sampled`] is the common case of `num_frames` frames spread evenly
//! over the clip. All take the clip as a slice; the `_from` variants take any
//! [`ByteSource`], such as a file or a ranged download. A caller that needs
//! the clip's [`VideoInfo`] to choose its frames opens a [`Video`] once and
//! decodes from it, so the container is demuxed once, not once to probe and
//! again to decode.
//!
//! Which frames to keep and what size they should be are the caller's
//! decisions: a model's sampling rule and pixel budget turn into frame indices
//! and a [`Fit`], and this module applies them without knowing the model. The resize is the crate's bit-exact [`resize_rgb`], run on each
//! sampled frame as it is decoded, so full-size frames are never all held at
//! once.
//!
//! Containers: MP4 (ISO BMFF) through this crate's own demuxer (see `mp4`),
//! WebM and Matroska through `matroska-demuxer`. Codecs: H.264, HEVC, VP8, VP9
//! and AV1, 8-bit 4:2:0.
//!
//! Decoding runs in NVIDIA's driver when it can: `libcuda.so.1` and
//! `libnvcuvid.so.1` are loaded at runtime (see `nvdec`) and GPU 0 is used.
//! When NVDEC cannot decode a VP8 or VP9 stream (no driver, a GPU without the
//! codec, or VP8, which NVDEC is not used for), it is decoded on the calling
//! thread by libvpx, also loaded at runtime (see `vpx`). Nothing is linked at
//! build time and no decoder ships in this crate. Without a decoder for the
//! stream, decoding returns [`MmError::Unsupported`].
//!
//! 64-bit Linux only: the driver and libvpx structures are transcribed for
//! the 64-bit (LP64) layout, and their layout checks fail the build on 32-bit
//! Linux. On other operating systems decoding returns
//! [`MmError::Unsupported`].
//!
//! YUV to RGB follows the bitstream's colour tags: BT.709, or BT.601 when the
//! bitstream says BT.601 or nothing (FFmpeg's default for untagged video), in
//! limited or full range. Other matrices are refused rather than guessed.
//! Container-level colour tags (MP4 `colr`, Matroska `Colour`) are not read.
//! Interlaced content is not deinterlaced.
//!
//! Times are player times: the first shown frame is at
//! [`VideoInfo::start_time`], which an MP4 edit list's delay or a WebM /
//! Matroska clip's first timestamp sets. An MP4 edit list also hides the
//! frames it cuts; one that a player would have to seek within (rows that go
//! back, repeat or skip media, or a rate other than 1) is refused.
//!
//! Decoding blocks the calling thread. Frames depend on earlier frames, so a
//! sampled frame is decoded from the last keyframe before it; groups of
//! pictures that hold no sampled frame are skipped, never read or decoded
//! (keyframes are checked in the bitstream first, else the whole stream is
//! decoded). Only the sampled frames are copied off the GPU and converted.
//!
//! Memory: MP4 is read through the [`ByteSource`] as it is decoded, one packet
//! at a time, so only its index (sample tables) is held. WebM and Matroska
//! packets are read into memory while the file is demuxed, so those hold the
//! whole video next to the source. A VP8 or VP9 clip is also read once in full
//! before decoding, to find which packets display a frame and what colour VP9
//! declares.

// Decoding is Linux-only; elsewhere the decode path is compiled out.
#![cfg_attr(not(target_os = "linux"), allow(dead_code))]

use crate::frames::{VideoFrames, VideoTiming};
use crate::image::resize::{Resample, resize_rgb};
use crate::{MmError, Result};

mod matroska;
mod mp4;
#[cfg(target_os = "linux")]
mod nvdec;
#[cfg(target_os = "linux")]
mod seek;
mod source;
#[cfg(target_os = "linux")]
mod vpx;

pub use source::ByteSource;
use source::SourceReader;

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
    /// Player time of the first shown frame, in seconds: the delay an MP4 edit
    /// list puts before the media, or a WebM / Matroska clip's first
    /// timestamp; 0 when the video starts at once. Frame timestamps are
    /// player times, so the first is `start_time`.
    pub start_time: f64,
    /// Seconds the shown frames last, from `start_time`.
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
    /// Largest WebM / Matroska file accepted, in bytes: its demuxer allocates
    /// each block as it reads it and holds the video packets, so a file over
    /// this is refused before it is opened. While such a file is open the
    /// packets take about its size; the packet list (at most 2^23 entries)
    /// and the demuxer's index tables (read from at most 1 MiB of the file)
    /// come on top.
    /// MP4 is not bound by it (it keeps its sample tables and reads packets
    /// as they are decoded); `probe` applies the default.
    pub max_input_bytes: usize,
}

impl Default for VideoDecodeLimits {
    /// 8192 x 8192 frames, 512 MiB of output (the decoded-frame budget
    /// `ai-dynamo/dynamo` uses for video) and a 512 MiB WebM / Matroska file.
    fn default() -> Self {
        Self {
            max_width: 8192,
            max_height: 8192,
            max_output_bytes: 512 << 20,
            max_input_bytes: 512 << 20,
        }
    }
}

/// How frames are sized for a model: from the clip's frame size and the
/// number of frames kept, the one size every returned frame is resized to,
/// and the resampler to do it with. The decoder applies the answer without
/// knowing the model. [`FrameResize`] is a size chosen already;
/// `models::qwen_vl::Qwen3VlVideoFit` is HF's Qwen3-VL (and Cosmos3) rule
/// under a request budget; a consumer implements this for any other model.
pub trait Fit {
    /// `(width, height)` for each of `num_frames` frames of a `width` x
    /// `height` clip. Bounded by [`VideoDecodeLimits`] like the source.
    fn fit(&self, num_frames: usize, width: u32, height: u32) -> Result<(u32, u32)>;
    /// The resampler, which decides the bytes: PIL and torchvision differ.
    fn filter(&self) -> Resample;
}

/// A size every returned frame is resized to, chosen by the caller.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameResize {
    pub width: u32,
    pub height: u32,
    pub filter: Resample,
}

impl Fit for FrameResize {
    fn fit(&self, _num_frames: usize, _width: u32, _height: u32) -> Result<(u32, u32)> {
        Ok((self.width, self.height))
    }

    fn filter(&self) -> Resample {
        self.filter
    }
}

/// The most packets a track may have: `mp4::MAX_SAMPLES`, applied to Matroska
/// too, where packets are read into memory as the file is demuxed.
const MAX_PACKETS: usize = 1 << 23;

/// Where a packet's bytes are.
enum Payload {
    /// In the video source, read when the packet is decoded.
    Source { offset: u64, len: usize },
    /// Already read, by a demuxer that cannot say where they are.
    Owned(Vec<u8>),
}

/// One compressed frame, in decode order.
struct Packet {
    payload: Payload,
    pts: f64,
    /// Whether the decoder will output a frame for this packet. `false` only
    /// for a VP8 or VP9 packet whose frames are all hidden (`show_frame = 0`).
    displays: bool,
    /// Whether the output frame belongs to the video: `false` for a Matroska
    /// frame marked invisible and for a packet that displays nothing. Every
    /// packet a decode passes is decoded, since later frames may reference it.
    shown: bool,
    /// Whether the container marks the packet a keyframe, where a decode may
    /// start. Unchecked until a decode starts there (see `seek`).
    sync: bool,
}

impl Packet {
    fn new(payload: Payload, pts: f64, displays: bool, invisible: bool, sync: bool) -> Self {
        Self {
            payload,
            pts,
            displays,
            shown: displays && !invisible,
            sync,
        }
    }
}

/// A demuxed video track.
struct Track<'a> {
    codec: VideoCodec,
    width: u32,
    height: u32,
    start_time: f64,
    duration: f64,
    /// The codec configuration record: `avcC` (H.264), `hvcC` (HEVC) or `av1C`
    /// (AV1), from the MP4 sample entry or Matroska `CodecPrivate`.
    config: Option<Vec<u8>>,
    /// Whether H.264 / HEVC parameter sets may also travel in the packets: an
    /// MP4 `hev1` sample entry, or Matroska (`avc1` and `hvc1` keep them all
    /// in the configuration record).
    inband_parameter_sets: bool,
    packets: Vec<Packet>,
    /// What a VP9 track's keyframes declare, found while demuxing.
    vp9_colour: Vp9Colour,
    source: &'a dyn ByteSource,
}

impl Track<'_> {
    /// The bytes of `packet`: its own, or read from the source into `buffer`.
    fn packet_bytes<'b>(&'b self, packet: &'b Packet, buffer: &'b mut Vec<u8>) -> Result<&'b [u8]> {
        match &packet.payload {
            Payload::Owned(data) => Ok(data),
            Payload::Source { offset, len } => {
                buffer.resize(*len, 0);
                self.source
                    .read_at(*offset, buffer)
                    .map_err(|e| read_error(e, "a video packet"))?;
                Ok(buffer)
            }
        }
    }

    fn info(&self) -> VideoInfo {
        let frame_count = self.packets.iter().filter(|p| p.shown).count() as u64;
        VideoInfo {
            codec: self.codec,
            width: self.width,
            height: self.height,
            start_time: self.start_time,
            duration: self.duration,
            fps: frame_count as f64 / self.duration,
            frame_count,
        }
    }
}

/// Read the container and describe its first video track.
pub fn probe(bytes: &[u8]) -> Result<VideoInfo> {
    probe_from(&bytes)
}

/// [`probe`] for a video in any [`ByteSource`].
pub fn probe_from(source: &dyn ByteSource) -> Result<VideoInfo> {
    Ok(Video::open(source, &VideoDecodeLimits::default())?.info())
}

/// A video whose container has been read: its [`VideoInfo`], and the frames
/// it decodes to. Open once, decide which frames to keep from `info`, then
/// `decode`; [`probe`] followed by [`decode_frames`] reads the container
/// twice (for WebM / Matroska, every packet twice).
///
/// `limits.max_input_bytes` applies when the video is opened; the frame-size
/// and output limits apply to each decode.
pub struct Video<'a> {
    track: Track<'a>,
}

impl<'a> Video<'a> {
    /// Read the container of a video in any [`ByteSource`] (for one in
    /// memory, `&bytes`), which the decode reads packets from later. Errors
    /// as [`decode_frames_from`] before decoding: an unreadable container, or
    /// a source that ends before its declared length, is
    /// [`MmError::InvalidInput`]; a source whose reads fail
    /// [`MmError::Internal`]; a codec the crate does not handle
    /// [`MmError::Unsupported`]; a WebM / Matroska file over
    /// `limits.max_input_bytes` [`MmError::LimitExceeded`].
    pub fn open(source: &'a dyn ByteSource, limits: &VideoDecodeLimits) -> Result<Self> {
        Ok(Self {
            track: demux_with(source, limits.max_input_bytes)?,
        })
    }

    /// What [`probe`] reports.
    pub fn info(&self) -> VideoInfo {
        self.track.info()
    }

    /// [`decode_frames`] of this video.
    pub fn decode(
        &self,
        indices: &[u64],
        fit: Option<&dyn Fit>,
        limits: &VideoDecodeLimits,
    ) -> Result<VideoFrames> {
        check_indices(indices)?;
        decode_indices(&self.track, indices, fit, limits, decode_track)
    }

    /// [`decode_sampled`] of this video.
    pub fn decode_sampled(
        &self,
        num_frames: u64,
        limits: &VideoDecodeLimits,
    ) -> Result<VideoFrames> {
        check_count(num_frames)?;
        decode_count(&self.track, num_frames, limits, decode_track)
    }
}

/// Decode `bytes` and keep the shown frames at `indices` (counted from 0 in
/// display order, as [`VideoInfo::frame_count`] counts them), in that order,
/// each resized to the size `fit` gives, if any. An index may repeat; the
/// list must not decrease, since frames are kept as they are decoded.
///
/// Errors: [`MmError::Unsupported`] for a stream neither NVDEC (no driver, or
/// a GPU without the codec) nor, for VP8 and VP9, libvpx (not installed) can
/// decode here, and for anything other than 8-bit 4:2:0 in a supported colour
/// matrix; [`MmError::LimitExceeded`] when a source or resized frame, or the
/// output, exceeds `limits`; [`MmError::InvalidInput`] for an unreadable
/// container or bitstream, no indices, an index past the last frame or out of
/// order, an empty resize, or a resolution change mid-stream; and whatever
/// `fit` returns.
pub fn decode_frames(
    bytes: &[u8],
    indices: &[u64],
    fit: Option<&dyn Fit>,
    limits: &VideoDecodeLimits,
) -> Result<VideoFrames> {
    decode_frames_from(&bytes, indices, fit, limits)
}

/// [`decode_frames`] for a video in any [`ByteSource`]; same errors, plus
/// [`MmError::Internal`] when the source itself fails, whether while a packet,
/// an MP4 box or the Matroska structure is read. A source that ends before its
/// declared length, and a container the parser cannot read, are
/// [`MmError::InvalidInput`], as for a slice.
pub fn decode_frames_from(
    source: &dyn ByteSource,
    indices: &[u64],
    fit: Option<&dyn Fit>,
    limits: &VideoDecodeLimits,
) -> Result<VideoFrames> {
    decode_frames_with(source, indices, fit, limits, decode_track)
}

/// Decode `bytes` and keep `num_frames` frames spread evenly over the clip,
/// at the source size: [`decode_frames`] at indices
/// `round(linspace(0, frame_count - 1, num_frames))`, with its errors (zero
/// frames, more than the clip shows or more than the frame cap are
/// `InvalidInput`).
pub fn decode_sampled(
    bytes: &[u8],
    num_frames: u64,
    limits: &VideoDecodeLimits,
) -> Result<VideoFrames> {
    decode_sampled_from(&bytes, num_frames, limits)
}

/// [`decode_sampled`] for a video in any [`ByteSource`], with the errors of
/// [`decode_frames_from`].
pub fn decode_sampled_from(
    source: &dyn ByteSource,
    num_frames: u64,
    limits: &VideoDecodeLimits,
) -> Result<VideoFrames> {
    decode_sampled_with(source, num_frames, limits, decode_track)
}

/// A decoder: decodes a track and feeds its displayed frames to the sampler.
type Backend = fn(&Track<'_>, &mut Sampler<'_>) -> Result<()>;

/// NVDEC, or for VP8 and VP9 libvpx when NVDEC cannot decode the stream. A
/// broken or oversized stream is not decoded twice: only
/// [`MmError::Unsupported`] moves to the CPU.
#[cfg(target_os = "linux")]
fn decode_track(track: &Track<'_>, sampler: &mut Sampler<'_>) -> Result<()> {
    match nvdec::decode(track, sampler) {
        Err(MmError::Unsupported { message, .. })
            if matches!(track.codec, VideoCodec::Vp8 | VideoCodec::Vp9) =>
        {
            sampler.restart();
            vpx::decode(track, sampler).map_err(|e| match e {
                MmError::Unsupported { message: cpu, .. } => {
                    MmError::unsupported(format!("{message}; {cpu}"))
                }
                e => e,
            })
        }
        result => result,
    }
}

#[cfg(not(target_os = "linux"))]
fn decode_track(_: &Track<'_>, _: &mut Sampler<'_>) -> Result<()> {
    Err(MmError::unsupported(
        "video decoding is only built on Linux",
    ))
}

fn decode_sampled_with(
    source: &dyn ByteSource,
    num_frames: u64,
    limits: &VideoDecodeLimits,
    backend: Backend,
) -> Result<VideoFrames> {
    // Checked before the container is read: a bad count is a cheap error.
    check_count(num_frames)?;
    let track = demux_with(source, limits.max_input_bytes)?;
    decode_count(&track, num_frames, limits, backend)
}

/// What a frame count must satisfy before any video is known.
fn check_count(num_frames: u64) -> Result<()> {
    if num_frames == 0 {
        return Err(MmError::invalid_input("num_frames must be at least 1"));
    }
    // As for a list of indices: no video has more frames than this.
    if num_frames > MAX_PACKETS as u64 {
        return Err(MmError::invalid_input(format!(
            "more than {MAX_PACKETS} frames requested"
        )));
    }
    Ok(())
}

/// Decode `num_frames` evenly spaced frames of a demuxed track.
fn decode_count(
    track: &Track<'_>,
    num_frames: u64,
    limits: &VideoDecodeLimits,
    backend: Backend,
) -> Result<VideoFrames> {
    let targets = sample_indices(track.info().frame_count, num_frames)?;
    decode_demuxed(track, &targets, None, limits, backend)
}

fn decode_frames_with(
    source: &dyn ByteSource,
    indices: &[u64],
    fit: Option<&dyn Fit>,
    limits: &VideoDecodeLimits,
    backend: Backend,
) -> Result<VideoFrames> {
    // Checked before the container is read: a bad list is a cheap error.
    check_indices(indices)?;
    let track = demux_with(source, limits.max_input_bytes)?;
    decode_indices(&track, indices, fit, limits, backend)
}

/// What a list of frame indices must satisfy before any video is known.
fn check_indices(indices: &[u64]) -> Result<()> {
    if indices.is_empty() {
        return Err(MmError::invalid_input("no frame indices given"));
    }
    // Keeps the timestamps (8 bytes a frame) within reason whatever the
    // frame size: no video has more frames than this.
    if indices.len() > MAX_PACKETS {
        return Err(MmError::invalid_input(format!(
            "more than {MAX_PACKETS} frame indices"
        )));
    }
    if indices.windows(2).any(|pair| pair[1] < pair[0]) {
        return Err(MmError::invalid_input(
            "frame indices must not decrease (frames are kept in display order)",
        ));
    }
    Ok(())
}

/// Decode the frames at `indices` (already checked by [`check_indices`]) of a
/// demuxed track.
fn decode_indices(
    track: &Track<'_>,
    indices: &[u64],
    fit: Option<&dyn Fit>,
    limits: &VideoDecodeLimits,
    backend: Backend,
) -> Result<VideoFrames> {
    let frame_count = track.info().frame_count;
    if let Some(&last) = indices.last()
        && last >= frame_count
    {
        return Err(MmError::invalid_input(format!(
            "frame index {last} is past the last frame of a video with {frame_count}"
        )));
    }
    decode_demuxed(track, indices, fit, limits, backend)
}

/// Decode `track` and keep the shown frames at `targets`, resized if asked.
fn decode_demuxed(
    track: &Track<'_>,
    targets: &[u64],
    fit: Option<&dyn Fit>,
    limits: &VideoDecodeLimits,
    backend: Backend,
) -> Result<VideoFrames> {
    // Refused once here rather than by each decoder in turn.
    track.vp9_colour.get()?;
    let info = track.info();
    check_size(info.width, info.height, limits)?;
    // The size of every returned frame, asked of `fit` once the source size
    // is known and checked like the source.
    let resize = match fit {
        Some(fit) => {
            let (width, height) = fit.fit(targets.len(), info.width, info.height)?;
            check_size(width, height, limits).map_err(|e| match e {
                MmError::LimitExceeded { .. } => MmError::limit_exceeded(format!(
                    "{width}x{height} resize target exceeds the {}x{} decode limit",
                    limits.max_width, limits.max_height
                )),
                _ => MmError::invalid_input("resize target has an empty side"),
            })?;
            Some(FrameResize {
                width,
                height,
                filter: fit.filter(),
            })
        }
        None => None,
    };
    let (out_width, out_height) = resize.map_or((info.width, info.height), |r| (r.width, r.height));
    let too_big = || {
        MmError::limit_exceeded(format!(
            "{} frames of {out_width}x{out_height} exceed the {} byte output limit",
            targets.len(),
            limits.max_output_bytes
        ))
    };
    (out_width as usize)
        .checked_mul(out_height as usize)
        .and_then(|n| n.checked_mul(3))
        .and_then(|n| n.checked_mul(targets.len()))
        .filter(|n| *n <= limits.max_output_bytes)
        .ok_or_else(too_big)?;

    let mut sampler = Sampler {
        targets,
        next: 0,
        width: info.width,
        height: info.height,
        limits: *limits,
        resize,
        // Grow frame by frame: nothing large is reserved before decoding.
        rgb: Vec::new(),
        full: Vec::new(),
        timestamps: Vec::with_capacity(targets.len()),
    };
    backend(track, &mut sampler)?;
    if let Some(missing) = sampler.targets.get(sampler.next) {
        return Err(MmError::invalid_input(format!(
            "frame {missing} of the {} the video declares did not decode",
            info.frame_count
        )));
    }
    VideoFrames::new(
        out_width as usize,
        out_height as usize,
        sampler.rgb,
        VideoTiming {
            timestamps: sampler.timestamps,
            source_fps: info.fps,
            source_start: info.start_time,
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

/// The `n` frame indices [`decode_sampled`] keeps out of `total`:
/// `round(linspace(0, total - 1, n))`, ties to even, as the HF and Dynamo
/// samplers pick frames. `n` must be at least 1 and at most `total` (and the
/// frame cap), else [`MmError::InvalidInput`].
pub fn sample_indices(total: u64, n: u64) -> Result<Vec<u64>> {
    check_count(n)?;
    if n > total {
        return Err(MmError::invalid_input(format!(
            "cannot sample {n} frames from a video with {total}"
        )));
    }
    if n == 1 {
        return Ok(vec![0]);
    }
    let last = (total - 1) as f64;
    Ok((0..n)
        .map(|i| (last * i as f64 / (n - 1) as f64).round_ties_even() as u64)
        // `last` is rounded to an f64 (from 2^53 up), which may round up.
        .map(|index| index.min(total - 1))
        .collect())
}

/// Collects the sampled frames as displayed frames arrive, each with its
/// index among the video's shown frames. A decode that skips ahead (see
/// `seek`) never delivers a frame it skipped, and any frame it delivers that
/// is not the next sampled one is ignored.
struct Sampler<'t> {
    targets: &'t [u64],
    next: usize,
    width: u32,
    height: u32,
    limits: VideoDecodeLimits,
    /// The size every kept frame is resized to, if any.
    resize: Option<FrameResize>,
    rgb: Vec<u8>,
    /// One frame at the source size, when frames are resized.
    full: Vec<u8>,
    timestamps: Vec<f64>,
}

impl Sampler<'_> {
    fn done(&self) -> bool {
        self.next == self.targets.len()
    }

    /// Forget every frame, for a decode that starts over.
    fn restart(&mut self) {
        self.next = 0;
        self.rgb.clear();
        self.timestamps.clear();
    }

    /// Whether shown frame `index` is the next sampled one.
    fn wants(&self, index: u64) -> bool {
        self.targets.get(self.next) == Some(&index)
    }

    /// Take shown frame `index`: convert it, and resize it if asked, if it is
    /// the next sampled one (kept again if the index repeats).
    fn push(&mut self, picture: &Nv12<'_>, pts: f64, index: u64) -> Result<()> {
        if (picture.width, picture.height) != (self.width, self.height) {
            return Err(MmError::invalid_input(format!(
                "frame size changed mid-stream from {}x{} to {}x{}",
                self.width, self.height, picture.width, picture.height
            )));
        }
        if !self.wants(index) {
            return Ok(());
        }
        let start = self.rgb.len();
        match self.resize {
            None => picture.to_rgb(&mut self.rgb),
            Some(r) => {
                self.full.clear();
                picture.to_rgb(&mut self.full);
                let resized = resize_rgb(
                    &self.full,
                    self.height as usize,
                    self.width as usize,
                    r.height as usize,
                    r.width as usize,
                    r.filter,
                );
                self.rgb.extend_from_slice(&resized);
            }
        }
        let end = self.rgb.len();
        self.timestamps.push(pts);
        self.next += 1;
        while self.targets.get(self.next) == Some(&index) {
            self.rgb.extend_from_within(start..end);
            self.timestamps.push(pts);
            self.next += 1;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Demux
// ---------------------------------------------------------------------------

/// A failed read from the source: a short source is a bad file, anything else
/// is the source's own failure.
fn read_error(e: std::io::Error, what: &str) -> MmError {
    if e.kind() == std::io::ErrorKind::UnexpectedEof {
        MmError::invalid_input_with_source(format!("{what} lies outside the video"), e)
    } else {
        MmError::internal_with_source(format!("cannot read {what} from the video source"), e)
    }
}

/// [`demux_with`] under the default input limit.
#[cfg(test)]
fn demux(source: &dyn ByteSource) -> Result<Track<'_>> {
    demux_with(source, VideoDecodeLimits::default().max_input_bytes)
}

/// Read the container's video track. A WebM / Matroska file, whose packets
/// are held in memory, must be at most `max_input_bytes`; MP4 holds none.
fn demux_with(source: &dyn ByteSource, max_input_bytes: usize) -> Result<Track<'_>> {
    let mut magic = [0u8; 8];
    let magic = &mut magic[..source.len().min(8) as usize];
    source
        .read_at(0, magic)
        .map_err(|e| read_error(e, "the container header"))?;
    let track = if magic.get(4..8) == Some(b"ftyp") {
        demux_mp4(source)?
    } else if magic.starts_with(&[0x1A, 0x45, 0xDF, 0xA3]) {
        demux_matroska(source, max_input_bytes)?
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

/// Whether a VP8 or VP9 packet outputs a frame (`true` for other codecs), and
/// for VP9 the colour its keyframes declare.
fn packet_displays(codec: VideoCodec, packet: &[u8], vp9_colour: &mut Vp9Colour) -> Result<bool> {
    match codec {
        VideoCodec::Vp8 => Ok(vp8_packet_displays(packet)),
        VideoCodec::Vp9 => {
            vp9_colour.note(packet);
            vp9_packet_displays(packet)
        }
        _ => Ok(true),
    }
}

fn demux_mp4(source: &dyn ByteSource) -> Result<Track<'_>> {
    let mut movie = mp4::read(source)?;
    let codec = movie.codec;
    // Only VP8 and VP9 need a look inside the packets, for hidden frames and
    // colour, before the edit list is applied; the other codecs' packets are
    // not read until they are decoded.
    let mut vp9_colour = Vp9Colour::default();
    if matches!(codec, VideoCodec::Vp8 | VideoCodec::Vp9) {
        let mut scratch = Vec::new();
        for sample in &mut movie.samples {
            scratch.resize(sample.size as usize, 0);
            source
                .read_at(sample.offset, &mut scratch)
                .map_err(|e| read_error(e, "a video packet"))?;
            sample.displays = packet_displays(codec, &scratch, &mut vp9_colour)?;
        }
    }
    let video = movie.finish()?;
    let timescale = video.timescale as f64;
    // Samples are in decode order, which is what the decoders need.
    let mut packets = Vec::with_capacity(video.samples.len());
    for sample in &video.samples {
        let payload = Payload::Source {
            offset: sample.offset,
            len: sample.size as usize,
        };
        let pts = video.start_time + sample.cts as f64 / timescale;
        packets.push(Packet::new(
            payload,
            pts,
            sample.displays,
            !sample.shown,
            sample.sync,
        ));
    }
    Ok(Track {
        codec,
        width: video.width,
        height: video.height,
        start_time: video.start_time,
        duration: video.duration,
        config: match codec {
            VideoCodec::H264 | VideoCodec::Hevc | VideoCodec::Av1 => video.config,
            _ => None,
        },
        inband_parameter_sets: video.inband_parameter_sets,
        packets,
        vp9_colour,
        source,
    })
}

fn demux_matroska(source: &dyn ByteSource, max_input_bytes: usize) -> Result<Track<'_>> {
    use matroska_demuxer::{DemuxError, Frame, MatroskaFile, TrackType};
    let (reader, failure) = SourceReader::new(source);
    // The demuxer's own failures are bad input. A failure of the source under
    // it keeps its I/O meaning (a short source is bad input, anything else is
    // the source's own failure, `Internal`, as for MP4), whether the demuxer
    // passes it on or reports something else in its place (while opening, it
    // turns one into "no cluster found"): the reader remembers it.
    let invalid = |e: DemuxError| match (failure.take(), e) {
        (Some(io), _) | (None, DemuxError::IoError(io)) => {
            read_error(io, "the WebM / Matroska file")
        }
        (None, e) => MmError::invalid_input_with_source("unreadable WebM / Matroska", e),
    };
    // The demuxer allocates every block it reads, of any track, and every
    // binary or string field, checking each declared size only against the
    // bytes left in the file. Bounding the file before it is opened bounds
    // each of those payloads (its own bookkeeping it sizes as it goes).
    if source.len() > max_input_bytes as u64 {
        return Err(MmError::limit_exceeded(format!(
            "{} byte WebM / Matroska file exceeds the {max_input_bytes} byte input limit",
            source.len()
        )));
    }
    // What the demuxer would loop on, panic on or index without bound, while
    // opening the file or reading its frames.
    matroska::check(source)?;
    let mut file = MatroskaFile::open(reader).map_err(invalid)?;
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
    let mut vp9_colour = Vp9Colour::default();
    // The latest end time of a shown frame, when the file says how long frames
    // last (the last packet need not be the last frame when frames reorder).
    let mut last_end: Option<f64> = None;
    // Bytes of packets held so far. The demuxer checks each block's declared
    // size against the bytes left in the file before it allocates, so no
    // packet outgrows the file, and the packets of a file never outgrow it
    // either (kept explicit here, with the packet count under the MP4 sample
    // cap); the file itself is under `max_input_bytes`, checked above.
    let mut held: u64 = 0;
    let mut frame = Frame::default();
    while file.next_frame(&mut frame).map_err(invalid)? {
        if frame.track != number {
            continue;
        }
        held += frame.data.len() as u64;
        if packets.len() >= MAX_PACKETS || held > source.len() {
            return Err(MmError::invalid_input(
                "WebM / Matroska file has more packets, or packet bytes, than it can hold",
            ));
        }
        let pts = frame.timestamp as f64 * scale;
        let displays = packet_displays(codec, &frame.data, &mut vp9_colour)?;
        let packet = Packet::new(
            Payload::Owned(std::mem::take(&mut frame.data)),
            pts,
            displays,
            frame.is_invisible,
            // A Block in a BlockGroup has no keyframe flag: not a start.
            frame.is_keyframe == Some(true),
        );
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
    // The demuxer takes a read that fails on an element header for the end of
    // the file; the reader remembers it, so a source failure is never a
    // shorter video.
    if let Some(e) = failure.take() {
        return Err(read_error(e, "the WebM / Matroska file"));
    }
    // Timestamps stay player times: a clip whose first frame is at 5 s starts
    // at 5 s (`start_time`) and lasts 1 s, not 6.
    let shown: Vec<f64> = packets.iter().filter(|p| p.shown).map(|p| p.pts).collect();
    let first = shown.iter().copied().fold(f64::INFINITY, f64::min);
    let first = if first.is_finite() { first } else { 0.0 };
    if first < 0.0 {
        return Err(MmError::invalid_input("video starts before time 0"));
    }
    let last = shown.iter().copied().fold(first, f64::max) - first;
    // The segment duration and block end times are measured from the segment
    // start; the duration is measured from the first shown frame.
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
        start_time: first,
        duration,
        config,
        inband_parameter_sets: matches!(codec, VideoCodec::H264 | VideoCodec::Hevc),
        packets,
        vp9_colour,
        source,
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

/// A VP9 packet's superframe index, read as libvpx reads it: `None` without
/// one, else the frames in the sizes it lists (the last may run into the index
/// itself). A last byte that marks an index which is not there, and an index
/// whose frames overrun the packet, are invalid, as libvpx refuses them.
fn vp9_superframe(packet: &[u8]) -> Result<Option<Vec<&[u8]>>> {
    let Some(&marker) = packet.last() else {
        return Ok(None);
    };
    if marker & 0xE0 != 0xC0 {
        return Ok(None);
    }
    let invalid = || MmError::invalid_input("malformed VP9 superframe index");
    let count = usize::from(marker & 0x07) + 1;
    let size_bytes = usize::from((marker >> 3) & 0x03) + 1;
    let index_start = packet
        .len()
        .checked_sub(2 + size_bytes * count)
        .ok_or_else(invalid)?;
    if packet[index_start] != marker {
        return Err(invalid());
    }
    let mut frames = Vec::with_capacity(count);
    let mut offset = 0usize;
    for i in 0..count {
        let field = &packet[index_start + 1 + i * size_bytes..][..size_bytes];
        let size = field
            .iter()
            .rev()
            .fold(0usize, |n, b| (n << 8) | usize::from(*b));
        let end = offset.checked_add(size).ok_or_else(invalid)?;
        frames.push(packet.get(offset..end).ok_or_else(invalid)?);
        offset = end;
    }
    Ok(Some(frames))
}

/// The frames of a VP9 packet: a superframe's frames, or the packet itself.
fn vp9_frames(packet: &[u8]) -> Result<Vec<&[u8]>> {
    Ok(vp9_superframe(packet)?.unwrap_or_else(|| vec![packet]))
}

/// Whether a VP9 packet outputs a frame: one of its frames has `show_frame`
/// set. Unparseable headers count as shown, so the decoder's own checks
/// decide. Refused, because they break the one-output-per-packet mapping: a
/// packet that shows more than one frame, and `show_existing_frame` (NVDEC
/// outputs nothing for a standalone one), and a malformed superframe index.
fn vp9_packet_displays(packet: &[u8]) -> Result<bool> {
    let mut shown = 0;
    for frame in vp9_frames(packet)? {
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

/// The one colour a VP9 stream's keyframes declare. The decoder is created
/// with it, so it may not change mid-stream.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
enum Vp9Colour {
    #[default]
    Undeclared,
    Declared(Matrix, bool),
    /// A colour space this crate does not convert, by its VP9 code.
    Unsupported(u8),
    Changed,
}

impl Vp9Colour {
    /// Take in the colour each keyframe in `packet` declares, a superframe's
    /// later frames included. The first problem found stays. A packet whose
    /// superframe index is malformed is refused when its frames are counted.
    fn note(&mut self, packet: &[u8]) {
        let frames = vp9_frames(packet).unwrap_or_default();
        for found in frames.into_iter().filter_map(vp9_keyframe_colour) {
            *self = match (*self, found) {
                (Self::Unsupported(_) | Self::Changed, _) => return,
                (_, Err(space)) => Self::Unsupported(space),
                (Self::Declared(m, r), Ok(colour)) if (m, r) != colour => Self::Changed,
                (_, Ok((matrix, full_range))) => Self::Declared(matrix, full_range),
            };
        }
    }

    /// (matrix, full range) to decode with, `None` if no keyframe says.
    fn get(self) -> Result<Option<(Matrix, bool)>> {
        match self {
            Self::Undeclared => Ok(None),
            Self::Declared(matrix, full_range) => Ok(Some((matrix, full_range))),
            Self::Unsupported(space) => Err(MmError::unsupported(format!(
                "VP9 colour space {space} is not supported"
            ))),
            Self::Changed => Err(MmError::invalid_input("VP9 colour changed mid-stream")),
        }
    }
}

/// The colour a VP9 keyframe declares: `Some(Ok((matrix, full_range)))` for a
/// keyframe, `None` for any other frame, and `Some(Err(colour_space))` for a
/// colour space this crate does not convert (SMPTE 240M, BT.2020, sRGB).
fn vp9_keyframe_colour(frame: &[u8]) -> Option<std::result::Result<(Matrix, bool), u8>> {
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
        other => Err(other),
    })
}

/// The frame size a VP9 frame's uncompressed header gives explicitly (a
/// keyframe, an intra-only frame, or an inter frame that takes no reference's
/// size), or `None` when it gives none (`show_existing_frame`, or an inter
/// frame sized like a reference). A header that ends early or is malformed is
/// refused here rather than left to libvpx, so no size reaches it unchecked.
fn vp9_frame_size(frame: &[u8]) -> Result<Option<(u32, u32)>> {
    vp9_read_frame_size(frame)
        .ok_or_else(|| MmError::invalid_input("truncated or malformed VP9 frame header"))
}

/// [`vp9_frame_size`], with `None` for a header that cannot be read.
fn vp9_read_frame_size(frame: &[u8]) -> Option<Option<(u32, u32)>> {
    let mut pos = 0;
    let mut read = |n: usize| -> Option<u32> {
        (0..n).try_fold(0u32, |v, _| {
            let bit = (frame.get(pos / 8)? >> (7 - pos % 8)) & 1;
            pos += 1;
            Some((v << 1) | u32::from(bit))
        })
    };
    if read(2)? != 2 {
        return None; // frame_marker
    }
    let profile = read(1)? | (read(1)? << 1);
    if profile == 3 {
        read(1)?;
    }
    if read(1)? == 1 {
        return Some(None); // show_existing_frame
    }
    let key = read(1)? == 0;
    let show = read(1)? == 1;
    let error_resilient = read(1)? == 1;
    let intra_only = !key && !show && read(1)? == 1;
    if !key && !error_resilient {
        read(2)?; // reset_frame_context
    }
    if key || intra_only {
        if read(24)? != 0x49_83_42 {
            return None;
        }
        if key || profile > 0 {
            // color_config
            if profile >= 2 {
                read(1)?;
            }
            let rgb = read(3)? == 7;
            let subsampling = profile == 1 || profile == 3;
            read(match (rgb, subsampling) {
                (false, true) => 4,
                (false, false) | (true, true) => 1,
                (true, false) => 0,
            })?;
        }
        if intra_only {
            read(8)?; // refresh_frame_flags
        }
    } else {
        read(8 + 3 * 4)?; // refresh_frame_flags, ref_frame_idx and sign bias
        for _ in 0..3 {
            if read(1)? == 1 {
                return Some(None); // found_ref: the size of a reference
            }
        }
    }
    Some(Some((read(16)? + 1, read(16)? + 1)))
}

/// Whether a VP8 packet's frame header sets `show_frame`. An empty packet
/// counts as shown, so the decoder's own checks decide.
fn vp8_packet_displays(packet: &[u8]) -> bool {
    packet.first().is_none_or(|tag| tag & 0x10 != 0)
}

/// The frame size a VP8 keyframe declares, or `None` for an inter frame.
fn vp8_keyframe_size(packet: &[u8]) -> Option<(u32, u32)> {
    // Frame tag (keyframe when bit 0 is clear), start code, then 14-bit width
    // and height with 2 scaling bits each, little-endian.
    if packet.first()? & 1 != 0 || packet.get(3..6)? != [0x9d, 0x01, 0x2a] {
        return None;
    }
    let dimension = |at: usize| {
        Some(u32::from(
            u16::from_le_bytes([*packet.get(at)?, *packet.get(at + 1)?]) & 0x3FFF,
        ))
    };
    Some((dimension(6)?, dimension(8)?))
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

/// The bytes to hand the decoder for a packet whose bytes are `data`: the
/// prefix before the first one, length-prefixed NAL units rewritten as Annex B.
/// Borrows `data` when nothing changes, otherwise builds into `buffer`.
fn to_decoder_input<'b>(
    data: &'b [u8],
    prefix: &[u8],
    length_size: Option<usize>,
    first: bool,
    buffer: &'b mut Vec<u8>,
) -> Result<&'b [u8]> {
    if length_size.is_none() && (!first || prefix.is_empty()) {
        return Ok(data);
    }
    buffer.clear();
    if first {
        buffer.extend_from_slice(prefix);
    }
    match length_size {
        Some(size) => length_prefixed_to_annexb(data, size, buffer)?,
        None => buffer.extend_from_slice(data),
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

    /// The bytes of packet `i`, wherever the demuxer keeps them.
    fn packet_data(track: &Track<'_>, i: usize) -> Vec<u8> {
        track
            .packet_bytes(&track.packets[i], &mut Vec::new())
            .unwrap()
            .to_vec()
    }

    fn track_of(name: &str) -> (Vec<u8>, VideoCodec, Option<Vec<u8>>) {
        let bytes = fixture(name);
        let (codec, config) = {
            let track = demux(&bytes).unwrap();
            (track.codec, track.config.clone())
        };
        (bytes, codec, config)
    }

    // -- container side: runs everywhere ----------------------------------

    #[test]
    fn sample_indices_spread_evenly() {
        let sample = |total, n| sample_indices(total, n).unwrap();
        assert_eq!(sample(10, 1), vec![0]);
        assert_eq!(sample(10, 2), vec![0, 9]);
        assert_eq!(sample(10, 4), vec![0, 3, 6, 9]);
        assert_eq!(sample(10, 10), (0..10).collect::<Vec<_>>());
        assert_eq!(sample(5, 3), vec![0, 2, 4]);
        assert_eq!(sample(1, 1), vec![0]);
        // linspace(0, 5, 3) = [0, 2.5, 5]; 2.5 rounds to even.
        assert_eq!(sample(6, 3), vec![0, 2, 5]);
        // A count outside 1..=total, or past the frame cap, is refused before
        // anything is allocated.
        for (total, n) in [
            (10, 0),
            (0, 1),
            (10, 11),
            (0, 0),
            (u64::MAX, u64::MAX),
            (u64::MAX, MAX_PACKETS as u64 + 1),
        ] {
            let r = sample_indices(total, n);
            assert!(
                matches!(r, Err(MmError::InvalidInput { .. })),
                "{total} {n}: {r:?}"
            );
        }
        // Past 2^53 the f64 arithmetic rounds; an index still never reaches
        // `total`.
        assert_eq!(sample(u64::MAX, 2), vec![0, u64::MAX - 1]);
        assert_eq!(sample(1 << 54, 2), vec![0, (1 << 54) - 1]);
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
        // Tables that name no chunk or no chunk run: an error, never an
        // out-of-range index.
        for (fourcc, offset) in [(b"stco", 8), (b"stsc", 8)] {
            let mut bytes = fixture("vp9.mp4");
            patch_mp4(&mut bytes, fourcc, offset, 0);
            let r = probe(&bytes);
            assert!(
                matches!(r, Err(MmError::InvalidInput { .. })),
                "{fourcc:?}: {r:?}"
            );
        }
        // A track duration that ends before the last sample starts.
        let mut bytes = fixture("vp9.mp4");
        // mdhd version 0: version/flags, creation, modification, timescale, duration.
        patch_mp4(&mut bytes, b"mdhd", 20, 1);
        let r = probe(&bytes);
        assert!(matches!(r, Err(MmError::InvalidInput { .. })), "{r:?}");
    }

    #[test]
    fn sample_counts_the_file_cannot_back_are_refused_up_front() {
        // A small file asking for 2^31 samples is refused at once: by `stsz`
        // before anything is sized by its count, and by `stts` once its runs
        // outlast the samples.
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
    fn edit_lists_set_the_start_and_which_frames_show() {
        // ffprobe on the same files: start 1.5 s, 4 s, 40 frames; and start
        // 0, 10.5 s, 105 of 120 frames.
        let delay = probe(&fixture("delay.mp4")).unwrap();
        assert_eq!(delay.start_time, 1.5);
        assert!((delay.duration - 4.0).abs() < 1e-9, "{}", delay.duration);
        assert_eq!(delay.frame_count, 40);
        let cut = probe(&fixture("cut.mp4")).unwrap();
        assert_eq!(cut.start_time, 0.0);
        assert!((cut.duration - 10.5).abs() < 1e-9, "{}", cut.duration);
        assert_eq!(cut.frame_count, 105);
        let bytes = fixture("cut.mp4");
        let track = demux(&bytes).unwrap();
        assert_eq!(track.packets.len(), 120);
    }

    #[test]
    fn matroska_time_starts_at_the_first_frame() {
        let info = probe(&fixture("vp9_offset.webm")).unwrap();
        assert_eq!(info.start_time, 5.0);
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
        assert_eq!(vp9_frames(&superframe(&hidden, &shown)).unwrap().len(), 2);
        // libvpx lets the last frame run into the index; it refuses an index
        // whose frames overrun the packet, and a marker without its index.
        let mut long = superframe(&hidden, &shown);
        let at = long.len() - 2;
        long[at] += 4;
        assert_eq!(vp9_frames(&long).unwrap()[1].len(), shown.len() + 4);
        long[at] += 1;
        for packet in [long, vec![0x80, 0, 0b1100_0000]] {
            let r = vp9_frames(&packet);
            assert!(matches!(r, Err(MmError::InvalidInput { .. })), "{r:?}");
        }
        // Packets that cannot map to exactly one output are refused.
        for packet in [existing.to_vec(), superframe(&shown, &shown)] {
            let r = vp9_packet_displays(&packet);
            assert!(matches!(r, Err(MmError::Unsupported { .. })), "{r:?}");
        }
        // Real clips: every packet displays, superframes included.
        for name in ["vp9.webm", "vp9_altref.webm"] {
            let bytes = fixture(name);
            let track = demux(&bytes).unwrap();
            assert!(
                track.packets.iter().all(|p| p.displays && p.shown),
                "{name}"
            );
        }
        let bytes = fixture("vp9_altref.webm");
        let track = demux(&bytes).unwrap();
        assert!(
            (0..track.packets.len())
                .any(|i| vp9_frames(&packet_data(&track, i)).unwrap().len() > 1)
        );
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
            let track = demux(&bytes).unwrap();
            vp9_keyframe_colour(&packet_data(&track, 0)).map(|r| r.unwrap())
        };
        assert_eq!(first("vp9_709.webm"), Some((Matrix::Bt709, false)));
        assert_eq!(first("vp9.webm"), Some((Matrix::Bt601, false)));
        // An inter frame carries no colour config.
        let bytes = fixture("vp9.webm");
        let track = demux(&bytes).unwrap();
        assert_eq!(
            vp9_keyframe_colour(&packet_data(&track, 1)).map(|r| r.is_ok()),
            None
        );
        assert!(vp9_keyframe_colour(&[]).is_none());
        // A keyframe behind another frame in a superframe is read too, so a
        // colour change there is not missed.
        let first = |name: &str, i: usize| packet_data(&demux(&fixture(name)).unwrap(), i);
        let (inter, bt709) = (first("vp9.webm", 1), first("vp9_709.webm", 0));
        let mut superframe = [&inter[..], &bt709].concat();
        let marker = 0b1101_1001; // two frames, 4-byte sizes
        superframe.push(marker);
        for frame in [&inter, &bt709] {
            superframe.extend_from_slice(&u32::try_from(frame.len()).unwrap().to_le_bytes());
        }
        superframe.push(marker);
        let mut colour = Vp9Colour::default();
        colour.note(&first("vp9.webm", 0));
        assert_eq!(colour, Vp9Colour::Declared(Matrix::Bt601, false));
        colour.note(&superframe);
        assert_eq!(colour, Vp9Colour::Changed);
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
        let track = demux(&bytes).unwrap();
        let (prefix, length_size) = bitstream_prefix(&track).unwrap();
        let mut buffer = Vec::new();
        let first = to_decoder_input(
            &packet_data(&track, 0),
            &prefix,
            length_size,
            true,
            &mut buffer,
        )
        .unwrap()
        .to_vec();
        assert!(first.starts_with(&prefix));
        let second_packet = packet_data(&track, 1);
        let second =
            to_decoder_input(&second_packet, &prefix, length_size, false, &mut buffer).unwrap();
        assert!(second.starts_with(&START_CODE) && !second.starts_with(&prefix));
        // VP9 needs no rewriting: the packet is passed through untouched.
        let bytes = fixture("vp9.webm");
        let track = demux(&bytes).unwrap();
        let (prefix, length_size) = bitstream_prefix(&track).unwrap();
        let packet = packet_data(&track, 1);
        let input = to_decoder_input(&packet, &prefix, length_size, false, &mut buffer).unwrap();
        assert_eq!(input, &packet[..]);
    }

    /// A source that is not a slice and records every read.
    struct Recording<'a> {
        bytes: &'a [u8],
        reads: std::cell::RefCell<Vec<(u64, usize)>>,
    }

    impl<'a> Recording<'a> {
        fn new(bytes: &'a [u8]) -> Self {
            Self {
                bytes,
                reads: Default::default(),
            }
        }
    }

    impl ByteSource for Recording<'_> {
        fn len(&self) -> u64 {
            self.bytes.len() as u64
        }

        fn read_at(&self, offset: u64, buf: &mut [u8]) -> std::io::Result<()> {
            self.reads.borrow_mut().push((offset, buf.len()));
            (&self.bytes).read_at(offset, buf)
        }
    }

    /// The byte range of the top-level box `name` in an MP4.
    fn mp4_box_range(bytes: &[u8], name: &[u8; 4]) -> (u64, u64) {
        let mut pos = 0usize;
        while pos + 8 <= bytes.len() {
            // The fixtures use 32-bit box sizes.
            let size = u64::from(u32::from_be_bytes(bytes[pos..pos + 4].try_into().unwrap()));
            let header = 8;
            assert!(size >= header, "box size {size}");
            if &bytes[pos + 4..pos + 8] == name {
                return (pos as u64 + header, pos as u64 + size);
            }
            pos += size as usize;
        }
        panic!("no {} box", String::from_utf8_lossy(name));
    }

    #[test]
    fn an_mp4_is_probed_without_reading_its_media_data() {
        for name in ["h264.mp4", "hevc.mp4", "av1.mp4"] {
            let bytes = fixture(name);
            let source = Recording::new(&bytes);
            assert_eq!(
                probe_from(&source).unwrap(),
                probe(&bytes).unwrap(),
                "{name}"
            );
            let (start, end) = mp4_box_range(&bytes, b"mdat");
            let reads = source.reads.borrow();
            assert!(!reads.is_empty());
            assert!(
                reads
                    .iter()
                    .all(|&(at, len)| at + len as u64 <= start || at >= end),
                "{name}: read inside mdat {start}..{end}: {reads:?}"
            );
        }
    }

    #[test]
    fn every_container_probes_the_same_from_a_source() {
        for name in [
            "h264.mp4",
            "hevc.mp4",
            "av1.mp4",
            "vp9.webm",
            "vp9_altref.webm",
        ] {
            let bytes = fixture(name);
            assert_eq!(
                probe_from(&Recording::new(&bytes)).unwrap(),
                probe(&bytes).unwrap(),
                "{name}"
            );
        }
    }

    #[test]
    fn a_source_that_fails_is_an_internal_error_and_a_short_one_is_invalid() {
        struct Failing(u64);
        impl ByteSource for Failing {
            fn len(&self) -> u64 {
                self.0
            }
            fn read_at(&self, _: u64, _: &mut [u8]) -> std::io::Result<()> {
                Err(std::io::Error::other("disk on fire"))
            }
        }
        let r = probe_from(&Failing(1000));
        assert!(matches!(r, Err(MmError::Internal { .. })), "{r:?}");
        // A source that claims more bytes than it has.
        struct Short(Vec<u8>);
        impl ByteSource for Short {
            fn len(&self) -> u64 {
                self.0.len() as u64 + 100
            }
            fn read_at(&self, offset: u64, buf: &mut [u8]) -> std::io::Result<()> {
                (&self.0[..]).read_at(offset, buf)
            }
        }
        let r = probe_from(&Short(fixture("h264.mp4")));
        assert!(matches!(r, Err(MmError::InvalidInput { .. })), "{r:?}");
        // WebM / Matroska keeps the same meaning, mid-stream too: the source
        // serves the headers and fails in the packets.
        struct FailsAfter(Vec<u8>, u64);
        impl ByteSource for FailsAfter {
            fn len(&self) -> u64 {
                self.0.len() as u64
            }
            fn read_at(&self, offset: u64, buf: &mut [u8]) -> std::io::Result<()> {
                if offset + buf.len() as u64 > self.1 {
                    return Err(std::io::Error::other("disk on fire"));
                }
                (&self.0[..]).read_at(offset, buf)
            }
        }
        let webm = fixture("vp9.webm");
        let r = probe_from(&FailsAfter(webm.clone(), webm.len() as u64 / 2));
        assert!(matches!(r, Err(MmError::Internal { .. })), "{r:?}");
        let r = probe_from(&Short(webm.clone()));
        assert!(matches!(r, Err(MmError::InvalidInput { .. })), "{r:?}");
        let r = probe(&webm[..webm.len() / 2]);
        assert!(matches!(r, Err(MmError::InvalidInput { .. })), "{r:?}");
    }

    #[test]
    fn a_source_failure_anywhere_in_a_webm_is_internal() {
        // Wherever the failing read lands (the Cues at the end while opening,
        // an element header the demuxer would take for the end of the file,
        // or a packet), the result is the source's failure, never a shorter
        // clip or a "malformed file".
        struct FailsIn(Vec<u8>, std::ops::Range<u64>);
        impl ByteSource for FailsIn {
            fn len(&self) -> u64 {
                self.0.len() as u64
            }
            fn read_at(&self, offset: u64, buf: &mut [u8]) -> std::io::Result<()> {
                let end = offset + buf.len() as u64;
                if offset < self.1.end && end > self.1.start {
                    return Err(std::io::Error::other("disk on fire"));
                }
                (&self.0[..]).read_at(offset, buf)
            }
        }
        // Bytes the demuxer never reads (elements it seeks past) cannot fail
        // a read: those positions give the complete clip, nothing else.
        let bytes = fixture("vp9_altref.webm");
        let full = probe(&bytes).unwrap();
        for at in (0..bytes.len() as u64).step_by(5) {
            match probe_from(&FailsIn(bytes.clone(), at..at + 4)) {
                Err(MmError::Internal { .. }) => {}
                Ok(info) if info == full => {}
                r => panic!("at {at}: {r:?}"),
            }
        }
    }

    #[test]
    fn a_truncated_webm_is_never_a_shorter_video() {
        // The demuxer takes an I/O error on an element header for the end of
        // the file, so a cut there could pass as a shorter clip; a finalized
        // file's Segment size catches it first. Every cut of the file must be
        // an error, not a success with fewer frames.
        for name in ["vp9.webm", "vp8.webm", "vp9_altref.webm"] {
            let bytes = fixture(name);
            for cut in (1..bytes.len()).step_by(3) {
                let r = probe(&bytes[..cut]);
                assert!(
                    matches!(r, Err(MmError::InvalidInput { .. })),
                    "{name} cut at {cut}: {r:?}"
                );
            }
        }
    }

    /// `vp9.webm`, whose segment data starts at 48 with its SeekHead. The
    /// SeekHead's first two Seek entries keep a 4-byte SeekID at 59 and 73
    /// and a 1-byte SeekPosition at 66 and 80; a 98-byte Void follows at 111.
    fn vp9_webm_seek_head() -> Vec<u8> {
        let bytes = fixture("vp9.webm");
        assert_eq!(bytes[48..52], SEEK_HEAD_ID);
        assert_eq!(bytes[111], 0xEC);
        bytes
    }

    const SEEK_HEAD_ID: [u8; 4] = [0x11, 0x4D, 0x9B, 0x74];

    /// Point Seek entry `n` (0 or 1) of [`vp9_webm_seek_head`] at a SeekHead
    /// `position` bytes into the segment data.
    fn seek_to_seek_head(bytes: &mut [u8], n: usize, position: u8) {
        let (id, pos) = [(59, 66), (73, 80)][n];
        bytes[id..id + 4].copy_from_slice(&SEEK_HEAD_ID);
        bytes[pos] = position;
    }

    #[test]
    fn a_seek_head_that_loops_or_points_at_two_seek_heads_is_refused() {
        // matroska-demuxer 0.8.1 follows a SeekHead's reference to another
        // SeekHead with no record of those it has read: a SeekHead that
        // points at itself loops forever, and one that points at two
        // SeekHeads panics.
        let mut cycle = vp9_webm_seek_head();
        seek_to_seek_head(&mut cycle, 0, 0);
        let mut twice = cycle.clone();
        seek_to_seek_head(&mut twice, 1, 0);
        for (what, bytes) in [("a loop", cycle), ("two references", twice)] {
            let r = probe(&bytes);
            assert!(
                matches!(r, Err(MmError::InvalidInput { .. })),
                "{what}: {r:?}"
            );
        }
    }

    #[test]
    fn a_segment_that_starts_with_a_crc_32_element_is_refused() {
        // The demuxer skips such an element by reading it again, forever.
        // Here it replaces the SeekHead's header and runs to the Void.
        let mut bytes = vp9_webm_seek_head();
        bytes[48..50].copy_from_slice(&[0xBF, 0x80 | 61]);
        let r = probe(&bytes);
        assert!(matches!(r, Err(MmError::Unsupported { .. })), "{r:?}");
    }

    #[test]
    fn a_block_timestamp_the_demuxer_cannot_take_the_absolute_value_of_is_refused() {
        // The first SimpleBlock's data starts at 449: track 1, then its
        // 16-bit timestamp, -32768 here (the demuxer calls `abs` on it).
        let mut bytes = fixture("vp9.webm");
        assert_eq!(bytes[447..452], [0xA3, 0xC1, 0x81, 0x00, 0x00]);
        bytes[450..452].copy_from_slice(&i16::MIN.to_be_bytes());
        let r = probe(&bytes);
        assert!(matches!(r, Err(MmError::InvalidInput { .. })), "{r:?}");
    }

    #[test]
    fn a_seek_head_that_points_at_a_second_one_still_opens() {
        // Write a second SeekHead into the Void, holding the first entry
        // (Info, at 209, 161 into the segment), and point the first entry at
        // it (63 into the segment): the chain the format allows.
        let original = vp9_webm_seek_head();
        let mut bytes = original.clone();
        let seek = [
            &[0x4D, 0xBB, 0x8B, 0x53, 0xAB, 0x84][..],
            &[0x15, 0x49, 0xA9, 0x66, 0x53, 0xAC, 0x81, 161],
        ]
        .concat();
        let second = [&SEEK_HEAD_ID[..], &[0x80 | 14], &seek].concat();
        let void = [&[0xEC, 0x80 | 77][..], &[0; 77]].concat();
        bytes[111..209].copy_from_slice(&[second, void].concat());
        seek_to_seek_head(&mut bytes, 0, 63);
        assert_eq!(probe(&bytes).unwrap(), probe(&original).unwrap());
    }

    // -- NVDEC: needs a GPU ------------------------------------------------

    #[test]
    #[ignore = "needs an NVIDIA GPU and driver (NVDEC)"]
    fn a_source_decodes_like_a_slice_and_never_reads_the_whole_file() {
        for name in [
            "h264.mp4",
            "hevc.mp4",
            "av1.mp4",
            "vp9.webm",
            "vp9_altref.webm",
        ] {
            let bytes = fixture(name);
            let limits = VideoDecodeLimits::default();
            let want = decode_sampled(&bytes, 4, &limits).unwrap();
            let source = Recording::new(&bytes);
            let got = decode_sampled_from(&source, 4, &limits).unwrap();
            assert_eq!(got.rgb(), want.rgb(), "{name}");
            assert_eq!(got.timestamps(), want.timestamps(), "{name}");
            let biggest = source.reads.borrow().iter().map(|r| r.1).max().unwrap();
            if name.ends_with(".mp4") {
                assert!(biggest < bytes.len(), "{name}: one read of {biggest} bytes");
            }
        }
    }

    /// Decodes fixture `name` to `num_frames` frames with the default limits.
    type Decode = fn(&str, u64) -> Result<VideoFrames>;

    /// As `decode_sampled` picks the decoder.
    fn auto(name: &str, num_frames: u64) -> Result<VideoFrames> {
        decode_sampled(&fixture(name), num_frames, &VideoDecodeLimits::default())
    }

    fn assert_display_order(decode: Decode, names: &[&str]) {
        for name in names {
            let frames = decode(name, 10).unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(frames.frame_count(), 10, "{name}");
            for i in 0..10 {
                assert_frame_color(&frames, i, color(i as u64));
            }
            let ts = frames.timestamps();
            assert!((ts[3] - 0.3).abs() < 0.01, "{name}: {ts:?}");
        }
    }

    fn assert_even_samples(decode: Decode, names: &[&str]) {
        for name in names {
            let frames = decode(name, 4).unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(frames.frame_count(), 4);
            for (at, index) in [0, 3, 6, 9].into_iter().enumerate() {
                assert_frame_color(&frames, at, color(index));
                assert!((frames.timestamps()[at] - index as f64 * 0.1).abs() < 0.01);
            }
        }
    }

    #[test]
    #[ignore = "needs an NVIDIA GPU and driver (NVDEC)"]
    fn decodes_every_codec_in_display_order() {
        assert_display_order(
            auto,
            &[
                "h264.mp4",
                "hevc.mp4",
                "vp9.webm",
                "vp9.mp4",
                "av1.mp4",
                "vp9_sar.mp4",
            ],
        );
    }

    #[test]
    #[ignore = "needs an NVIDIA GPU and driver (NVDEC)"]
    fn samples_evenly_and_keeps_their_timestamps() {
        assert_even_samples(auto, &["h264.mp4", "vp9.webm"]);
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

    fn assert_bt709(decode: Decode, names: &[&str]) {
        // Encoded with the BT.709 matrix and tagged so; read as BT.601 the
        // colour would shift by more than 15 (generate.py checks that). VP9
        // reports its own colour codes, H.264 the H.273 ones.
        for name in names {
            let frames = decode(name, 1).unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_frame_color(&frames, 0, [220, 40, 200]);
        }
    }

    #[test]
    #[ignore = "needs an NVIDIA GPU and driver (NVDEC)"]
    fn bt709_tagged_streams_convert_with_bt709() {
        assert_bt709(auto, &["vp9_709.webm", "h264_709.mp4"]);
    }

    fn assert_invisible_frames_hidden(decode: Decode) {
        // vp9_invisible.webm is vp9_texture.webm with its third block flagged
        // invisible. The texture moves, so frames are predicted from their
        // predecessor: the shown frames must equal the original clip's frames
        // bit for bit, which only holds if the hidden frame was decoded.
        let full = decode("vp9_texture.webm", 10).unwrap();
        assert_eq!(
            probe(&fixture("vp9_invisible.webm")).unwrap().frame_count,
            9
        );
        let hidden = decode("vp9_invisible.webm", 9).unwrap();
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
    fn invisible_frames_are_decoded_but_not_shown() {
        assert_invisible_frames_hidden(auto);
    }

    fn assert_refused_before_allocation(decode: Decode, names: &[&str]) {
        for name in names {
            let r = decode(name, 1);
            assert!(
                matches!(r, Err(MmError::LimitExceeded { .. })),
                "{name}: {r:?}"
            );
        }
    }

    #[test]
    #[ignore = "needs an NVIDIA GPU and driver (NVDEC)"]
    fn frames_larger_than_the_container_are_refused_before_allocation() {
        // A VP9 keyframe claiming 16384x16384 in a 256x256 container, and an
        // AV1 stream of 256x256 whose container claims 128x128.
        assert_refused_before_allocation(auto, &["vp9_huge.webm", "av1_small_container.mp4"]);
    }

    fn assert_player_times(decode: Decode) {
        let frames = decode("vp9_offset.webm", 2).unwrap();
        assert_eq!(frames.source_start(), 5.0);
        assert_eq!(frames.timestamps()[0], 5.0);
        assert!(
            (frames.timestamps()[1] - 5.9).abs() < 0.01,
            "{:?}",
            frames.timestamps()
        );
    }

    #[test]
    #[ignore = "needs an NVIDIA GPU and driver (NVDEC)"]
    fn matroska_timestamps_are_player_times() {
        assert_player_times(auto);
    }

    /// The solid grey of frame `i` in delay.mp4 and cut.mp4.
    fn edit_grey(i: u64) -> [u8; 3] {
        let g = 16 + 2 * i as u8;
        [g, g, g]
    }

    #[test]
    #[ignore = "needs an NVIDIA GPU and driver (NVDEC)"]
    fn an_edit_list_delay_moves_the_frames_later() {
        let frames =
            decode_sampled(&fixture("delay.mp4"), 2, &VideoDecodeLimits::default()).unwrap();
        assert_eq!(frames.source_start(), 1.5);
        assert!((frames.source_duration() - 4.0).abs() < 1e-9);
        assert_eq!(frames.timestamps()[0], 1.5);
        assert!(
            (frames.timestamps()[1] - 5.4).abs() < 1e-9,
            "{:?}",
            frames.timestamps()
        );
        assert_frame_color(&frames, 0, edit_grey(0));
        assert_frame_color(&frames, 1, edit_grey(39));
    }

    #[test]
    #[ignore = "needs an NVIDIA GPU and driver (NVDEC)"]
    fn frames_an_edit_list_cuts_are_decoded_but_not_returned() {
        // The first 15 frames are references for the rest, never shown.
        let frames =
            decode_sampled(&fixture("cut.mp4"), 105, &VideoDecodeLimits::default()).unwrap();
        assert_eq!(frames.frame_count(), 105);
        for (at, t) in frames.timestamps().iter().enumerate() {
            assert!((t - at as f64 * 0.1).abs() < 1e-9, "{at}: {t}");
        }
        assert_frame_color(&frames, 0, edit_grey(15));
        assert_frame_color(&frames, 104, edit_grey(119));
    }

    fn assert_odd_size_matches_the_reference(decode: Decode) {
        let frames = decode("vp9_odd.webm", 1).unwrap();
        assert_eq!((frames.width(), frames.height()), (130, 129));
        let expected = fixture("vp9_odd.rgb");
        let got = frames.frame(0).unwrap();
        assert_eq!(got.len(), expected.len());
        let worst = got.iter().zip(&expected).map(|(a, b)| a.abs_diff(*b)).max();
        assert!(worst <= Some(1), "max channel difference {worst:?}");
    }

    #[test]
    #[ignore = "needs an NVIDIA GPU and driver (NVDEC)"]
    fn odd_sized_frames_match_the_reference_conversion() {
        // 130x129: NVDEC needs an even output size, so the frame is decoded
        // one row taller and cropped; the chroma plane starts after that even
        // row count. Compared with a software decoder's planes.
        assert_odd_size_matches_the_reference(auto);
    }

    fn assert_alt_refs_in_step(decode: Decode) {
        // Frame i of vp9_altref.webm has a grey bar of 20 + 10 i (generate.py)
        // and starts at i / 10 s. Superframes carrying a hidden alt-ref make
        // NVDEC's timestamps slip by one, so every frame is checked against
        // its own timestamp.
        let frames = decode("vp9_altref.webm", 20).unwrap();
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
    fn alt_ref_superframes_keep_frames_and_timestamps_together() {
        assert_alt_refs_in_step(auto);
    }

    // -- libvpx: needs libvpx installed (libvpx9 or the like) ----------------

    #[cfg(target_os = "linux")]
    mod libvpx {
        use super::*;
        use crate::image::resize::Filter;

        fn on(backend: Backend, name: &str, num_frames: u64) -> Result<VideoFrames> {
            let bytes = fixture(name);
            let limits = VideoDecodeLimits::default();
            decode_sampled_with(&bytes.as_slice(), num_frames, &limits, backend)
        }

        /// On the CPU, as when NVDEC cannot decode the stream.
        fn cpu(name: &str, num_frames: u64) -> Result<VideoFrames> {
            on(vpx::decode, name, num_frames)
        }

        #[test]
        fn vp8_and_vp9_decode_on_the_cpu() {
            let names = ["vp8.webm", "vp9.webm", "vp9.mp4", "vp9_sar.mp4"];
            assert_display_order(cpu, &names);
            assert_even_samples(cpu, &["vp8.webm", "vp9.webm"]);
            assert_bt709(cpu, &["vp9_709.webm"]);
            assert_odd_size_matches_the_reference(cpu);
            assert_player_times(cpu);
        }

        #[test]
        fn the_cpu_keeps_hidden_frames_out_of_the_output() {
            assert_invisible_frames_hidden(cpu);
            assert_alt_refs_in_step(cpu);
        }

        #[test]
        fn the_cpu_refuses_frames_larger_than_the_container_before_decoding() {
            assert_refused_before_allocation(cpu, &["vp9_huge.webm"]);
        }

        #[test]
        fn vp9_decodes_with_or_without_a_gpu() {
            // Through NVDEC on a GPU host; on one without (CI) through the
            // fallback to libvpx. Either way the frames are libvpx's.
            assert_display_order(auto, &["vp9.webm", "vp9.mp4"]);
            assert_alt_refs_in_step(auto);
            let frames = auto("vp9_texture.webm", 10).unwrap();
            assert!(frames.rgb() == cpu("vp9_texture.webm", 10).unwrap().rgb());
        }

        /// `decode_frames` on a fixture, through whichever decoder applies.
        fn frames_at(name: &str, indices: &[u64], fit: Option<&dyn Fit>) -> Result<VideoFrames> {
            decode_frames(&fixture(name), indices, fit, &VideoDecodeLimits::default())
        }

        /// Every frame of `frames` resized with `resize_rgb`, for comparison.
        fn resized_by_hand(
            frames: &VideoFrames,
            width: u32,
            height: u32,
            filter: Resample,
        ) -> Vec<u8> {
            (0..frames.frame_count())
                .flat_map(|i| {
                    resize_rgb(
                        frames.frame(i).unwrap(),
                        frames.height(),
                        frames.width(),
                        height as usize,
                        width as usize,
                        filter,
                    )
                })
                .collect()
        }

        #[test]
        fn the_caller_names_the_frames() {
            // Frame i of vp9.webm is color(i) at i / 10 s; an index may repeat.
            let frames = frames_at("vp9.webm", &[2, 5, 5, 9], None).unwrap();
            assert_eq!(frames.frame_count(), 4);
            for (at, index) in [2, 5, 5, 9].into_iter().enumerate() {
                assert_frame_color(&frames, at, color(index));
                assert!((frames.timestamps()[at] - index as f64 * 0.1).abs() < 0.01);
            }
            assert!(frames.frame(1) == frames.frame(2));
            // Evenly spaced sampling is the same call with linspace indices.
            let sampled = auto("vp9.webm", 4).unwrap();
            let named = frames_at("vp9.webm", &sample_indices(10, 4).unwrap(), None).unwrap();
            assert!(sampled.rgb() == named.rgb());
            assert_eq!(sampled.timestamps(), named.timestamps());
        }

        #[test]
        fn an_open_video_is_demuxed_once() {
            let limits = VideoDecodeLimits::default();
            let webm = fixture("vp9.webm");
            let source = Recording::new(&webm);
            let video = Video::open(&source, &limits).unwrap();
            assert_eq!(video.info(), probe(&webm).unwrap());
            // Opening is one probe's worth of reads, and WebM packets are
            // held once opened: decoding reads nothing more.
            let opened = source.reads.borrow().len();
            let probed = Recording::new(&webm);
            probe_from(&probed).unwrap();
            assert_eq!(opened, probed.reads.borrow().len());
            let named = video.decode(&[2, 5, 5, 9], None, &limits).unwrap();
            let sampled = video.decode_sampled(4, &limits).unwrap();
            assert_eq!(source.reads.borrow().len(), opened);
            let one_shot = frames_at("vp9.webm", &[2, 5, 5, 9], None).unwrap();
            assert!(named.rgb() == one_shot.rgb());
            assert_eq!(named.timestamps(), one_shot.timestamps());
            let one_shot = auto("vp9.webm", 4).unwrap();
            assert!(sampled.rgb() == one_shot.rgb());
            assert_eq!(sampled.timestamps(), one_shot.timestamps());
            // An MP4 keeps its packets in the source: opening reads the index
            // only, each decode reads packets, and nothing outside `mdat`.
            let mp4 = fixture("vp9.mp4");
            let source = Recording::new(&mp4);
            let video = Video::open(&source, &limits).unwrap();
            let (start, end) = mp4_box_range(&mp4, b"mdat");
            let opened = source.reads.borrow().len();
            let frames = video.decode(&[0, 9], None, &limits).unwrap();
            assert_eq!(frames.frame_count(), 2);
            let reads = source.reads.borrow();
            assert!(reads.len() > opened);
            assert!(
                reads[opened..]
                    .iter()
                    .all(|&(at, len)| at >= start && at + len as u64 <= end),
                "decode read outside mdat {start}..{end}: {:?}",
                &reads[opened..]
            );
            // The same checks as the free functions, from the same handle.
            let r = video.decode(&[3, 1], None, &limits);
            assert!(matches!(r, Err(MmError::InvalidInput { .. })), "{r:?}");
            let r = video.decode_sampled(0, &limits);
            assert!(matches!(r, Err(MmError::InvalidInput { .. })), "{r:?}");
            // A video in memory opens the same way.
            assert_eq!(Video::open(&webm, &limits).unwrap().info(), video.info());
        }

        #[test]
        fn frame_indices_are_checked_before_decoding() {
            let many = vec![0u64; MAX_PACKETS + 1];
            for indices in [&[][..], &[3, 1], &[10], &[0, 10], &many] {
                let r = frames_at("vp9.webm", indices, None);
                assert!(
                    matches!(r, Err(MmError::InvalidInput { .. })),
                    "{} indices: {r:?}",
                    indices.len()
                );
            }
        }

        #[test]
        fn frames_are_resized_as_resize_rgb_would() {
            // The decoder's per-frame resize must give the bytes a caller would
            // get resizing the full frames afterwards, for both kernels, down
            // and up, and from an odd source size.
            let cases = [
                ("vp9.webm", 100, 60, Resample::Pil(Filter::Bicubic)),
                ("vp9.webm", 100, 60, Resample::AtenU8),
                ("vp9.webm", 300, 300, Resample::AtenU8),
                ("vp9_odd.webm", 64, 64, Resample::AtenU8),
                ("vp8.webm", 128, 128, Resample::Pil(Filter::Bicubic)),
            ];
            for (name, width, height, filter) in cases {
                let n = probe(&fixture(name)).unwrap().frame_count;
                let indices: Vec<u64> = (0..n).collect();
                let full = frames_at(name, &indices, None).unwrap();
                let resize = FrameResize {
                    width,
                    height,
                    filter,
                };
                let got = frames_at(name, &indices, Some(&resize)).unwrap();
                assert_eq!(
                    (got.width(), got.height()),
                    (width as usize, height as usize),
                    "{name}"
                );
                assert!(
                    got.rgb() == resized_by_hand(&full, width, height, filter),
                    "{name} {width}x{height} {filter:?}: resized frames differ"
                );
                assert_eq!(got.timestamps(), full.timestamps(), "{name}");
            }
        }

        #[test]
        fn fit_is_asked_once_with_the_clip_and_frame_count() {
            // A caller's own rule: half the source size, told how many frames.
            struct Half(std::cell::Cell<Option<(usize, u32, u32)>>);
            impl Fit for Half {
                fn fit(&self, n: usize, w: u32, h: u32) -> Result<(u32, u32)> {
                    assert!(self.0.get().is_none(), "fit asked twice");
                    self.0.set(Some((n, w, h)));
                    Ok((w / 2, h / 2))
                }
                fn filter(&self) -> Resample {
                    Resample::AtenU8
                }
            }
            let half = Half(std::cell::Cell::new(None));
            let frames = frames_at("vp9.webm", &[0, 4, 9], Some(&half)).unwrap();
            assert_eq!(half.0.get(), Some((3, 256, 256)));
            assert_eq!((frames.width(), frames.height()), (128, 128));
            assert_frame_color(&frames, 1, color(4));
        }

        #[test]
        fn matroska_files_are_bounded_before_they_are_opened_but_mp4_is_not() {
            // 64 bytes: less than the 843-byte fixture.
            let tight = VideoDecodeLimits {
                max_input_bytes: 64,
                ..VideoDecodeLimits::default()
            };
            let webm = fixture("vp9.webm");
            let r = decode_frames_with(&webm.as_slice(), &[0], None, &tight, vpx::decode);
            assert!(matches!(r, Err(MmError::LimitExceeded { .. })), "{r:?}");
            // Refused before the demuxer opens the file, so before it can
            // allocate a block: a source over the default cap is read only
            // for its container header. The fixture's bytes stand in for the
            // start of a long file.
            struct Claims<'a> {
                bytes: &'a [u8],
                len: u64,
                reads: std::cell::RefCell<Vec<(u64, usize)>>,
            }
            impl ByteSource for Claims<'_> {
                fn len(&self) -> u64 {
                    self.len
                }
                fn read_at(&self, offset: u64, buf: &mut [u8]) -> std::io::Result<()> {
                    self.reads.borrow_mut().push((offset, buf.len()));
                    self.bytes.read_at(offset, buf)
                }
            }
            let long = Claims {
                bytes: &webm,
                len: VideoDecodeLimits::default().max_input_bytes as u64 + 1,
                reads: Default::default(),
            };
            let r = probe_from(&long);
            assert!(matches!(r, Err(MmError::LimitExceeded { .. })), "{r:?}");
            assert_eq!(*long.reads.borrow(), [(0, 8)]);
            // The cap is inclusive: a file of exactly the cap opens.
            let exact = |max_input_bytes: usize| {
                let limits = VideoDecodeLimits {
                    max_input_bytes,
                    ..VideoDecodeLimits::default()
                };
                Video::open(&webm, &limits).map(|v| v.info().frame_count)
            };
            assert_eq!(exact(webm.len()).unwrap(), 10);
            let r = exact(webm.len() - 1);
            assert!(matches!(r, Err(MmError::LimitExceeded { .. })), "{r:?}");
            // The same limit does not touch an MP4, whose packets stay in the
            // source (it may still fail for want of a GPU, differently).
            let r = decode_frames(&fixture("h264.mp4"), &[0], None, &tight);
            assert!(!matches!(r, Err(MmError::LimitExceeded { .. })), "{r:?}");
            // `probe` applies the default, which this file is far under.
            assert_eq!(probe(&webm).unwrap().frame_count, 10);
        }

        #[test]
        fn resize_targets_and_output_are_bounded() {
            let bicubic = Resample::Pil(Filter::Bicubic);
            let at = |w, h, limits: &VideoDecodeLimits| {
                let resize = FrameResize {
                    width: w,
                    height: h,
                    filter: bicubic,
                };
                decode_frames(&fixture("vp9.webm"), &[0, 9], Some(&resize), limits)
            };
            let limits = VideoDecodeLimits::default();
            let r = at(0, 64, &limits);
            assert!(matches!(r, Err(MmError::InvalidInput { .. })), "{r:?}");
            let r = at(limits.max_width + 1, 64, &limits);
            assert!(matches!(r, Err(MmError::LimitExceeded { .. })), "{r:?}");
            // The output limit applies to the resized frames: two 2x2 frames
            // fit in 24 bytes where two full frames would not.
            let tight = VideoDecodeLimits {
                max_output_bytes: 2 * 2 * 2 * 3,
                ..limits
            };
            assert_eq!(at(2, 2, &tight).unwrap().frame_count(), 2);
            let r = decode_frames(&fixture("vp9.webm"), &[0, 9], None, &tight);
            assert!(matches!(r, Err(MmError::LimitExceeded { .. })), "{r:?}");
        }

        #[test]
        fn qwen3_vl_and_cosmos3_size_frames_as_hf_does() {
            // Cosmos3-Nano's video_preprocessor_config.json (Qwen3VLVideoProcessor,
            // patch 16, merge 2, temporal 2, 4,096..25,165,824 pixels per clip).
            let nano: crate::models::qwen_vl::Qwen3VlVideo = serde_json::from_str(
                r#"{"size": {"longest_edge": 25165824, "shortest_edge": 4096},
                    "patch_size": 16, "temporal_patch_size": 2, "merge_size": 2,
                    "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5],
                    "processor_class": "Qwen3VLProcessor",
                    "video_processor_type": "Qwen3VLVideoProcessor"}"#,
            )
            .unwrap();
            // Ten 256x256 frames under a 163,840-pixel clip budget: HF's
            // smart_resize scales by sqrt(655,360 / 163,840) = 2 and floors to
            // the 32-pixel factor, so 128x128.
            let fit = crate::models::qwen_vl::Qwen3VlVideoFit {
                video: &nano,
                budget: crate::video::VideoPixelBudget {
                    total_pixels: Some(10 * 128 * 128),
                    max_pixels_per_frame: None,
                },
            };
            let indices: Vec<u64> = (0..10).collect();
            let frames = frames_at("vp9.webm", &indices, Some(&fit)).unwrap();
            assert_eq!((frames.width(), frames.height()), (128, 128));
            let full = frames_at("vp9.webm", &indices, None).unwrap();
            assert!(frames.rgb() == resized_by_hand(&full, 128, 128, Resample::AtenU8));
            // An odd count rounds to the temporal patch (9 frames count as 10
            // against the budget) and a repeated index counts as a frame, as
            // HF sees the sampled frames; the scale still comes out at 2.
            let nine = frames_at("vp9.webm", &indices[..9], Some(&fit)).unwrap();
            assert_eq!((nine.width(), nine.height()), (128, 128));
            let repeated =
                frames_at("vp9.webm", &[0, 0, 5, 5, 9, 9, 9, 9, 9, 9], Some(&fit)).unwrap();
            assert_eq!((repeated.width(), repeated.height()), (128, 128));
            assert!(repeated.frame(0) == frames.frame(0) && repeated.frame(9) == frames.frame(9));
            // Without a budget the clip is far under the model's maximum, so
            // 256 (a multiple of 32) stays.
            let fit = crate::models::qwen_vl::Qwen3VlVideoFit {
                video: &nano,
                budget: crate::video::VideoPixelBudget::default(),
            };
            let frames = frames_at("vp9.webm", &indices, Some(&fit)).unwrap();
            assert_eq!((frames.width(), frames.height()), (256, 256));
            assert!(frames.rgb() == full.rgb());
        }

        #[test]
        fn vp8_decodes_where_nvdec_cannot() {
            // NVDEC is not used for VP8, so this falls back with or without a GPU.
            let frames = auto("vp8.webm", 10).unwrap();
            assert!(frames.rgb() == cpu("vp8.webm", 10).unwrap().rgb());
        }

        #[test]
        fn corrupt_vp8_and_vp9_are_errors_not_crashes() {
            // libvpx runs in this process: damaged frames must come back as an
            // error or as frames, never as a crash. The packets are damaged
            // past their first bytes, so most damage lands in frame data.
            for name in ["vp8.webm", "vp9.webm", "vp9_altref.webm", "vp9_odd.webm"] {
                let bytes = fixture(name);
                let info = probe(&bytes).unwrap();
                let targets: Vec<u64> = (0..info.frame_count).collect();
                for seed in 0..64u64 {
                    let mut track = demux(&bytes).unwrap();
                    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
                    for _ in 0..8 {
                        state ^= state << 13;
                        state ^= state >> 7;
                        state ^= state << 17;
                        let i = (state as usize) % track.packets.len();
                        if let Payload::Owned(data) = &mut track.packets[i].payload
                            && data.len() > 8
                        {
                            let at = 8 + (state >> 16) as usize % (data.len() - 8);
                            data[at] ^= (state >> 40) as u8 | 1;
                        }
                    }
                    let mut sampler = Sampler {
                        targets: &targets,
                        next: 0,
                        width: info.width,
                        height: info.height,
                        limits: VideoDecodeLimits::default(),
                        resize: None,
                        rgb: Vec::new(),
                        full: Vec::new(),
                        timestamps: Vec::new(),
                    };
                    let _ = vpx::decode(&track, &mut sampler);
                }
            }
        }

        #[test]
        fn other_codecs_are_not_decoded_on_the_cpu() {
            let r = cpu("h264.mp4", 1);
            assert!(matches!(r, Err(MmError::Unsupported { .. })), "{r:?}");
        }

        #[test]
        #[ignore = "needs an NVIDIA GPU and driver (NVDEC)"]
        fn nvdec_and_the_cpu_decode_vp9_identically() {
            // VP9 decoding is exact, and both decoders share the YUV-to-RGB step.
            for name in [
                "vp9.webm",
                "vp9_texture.webm",
                "vp9_altref.webm",
                "vp9_odd.webm",
                "vp9_709.webm",
            ] {
                let n = probe(&fixture(name)).unwrap().frame_count;
                let gpu = on(nvdec::decode, name, n).unwrap();
                let cpu = cpu(name, n).unwrap();
                assert!(gpu.rgb() == cpu.rgb(), "{name}: frames differ");
                assert_eq!(gpu.timestamps(), cpu.timestamps(), "{name}");
            }
        }
    }

    #[test]
    fn frame_sizes_are_read_from_vp8_and_vp9_headers() {
        let packet = |name: &str, i: usize| packet_data(&demux(&fixture(name)).unwrap(), i);
        let size = |p: &[u8]| vp9_frame_size(p).unwrap();
        assert_eq!(size(&packet("vp9.webm", 0)), Some((256, 256)));
        assert_eq!(size(&packet("vp9_huge.webm", 0)), Some((16384, 16384)));
        // An inter frame takes a reference's size.
        assert_eq!(size(&packet("vp9.webm", 1)), None);
        assert!(vp9_frame_size(&[]).is_err());
        assert_eq!(vp8_keyframe_size(&packet("vp8.webm", 0)), Some((256, 256)));
        assert_eq!(vp8_keyframe_size(&packet("vp8.webm", 1)), None);
    }

    #[test]
    fn hidden_vp8_frames_do_not_count_as_displayed() {
        let bytes = fixture("vp8.webm");
        let track = demux(&bytes).unwrap();
        assert!(track.packets.iter().all(|p| p.displays));
        let mut hidden = packet_data(&track, 1);
        hidden[0] &= !0x10; // show_frame
        assert!(!vp8_packet_displays(&hidden));
        assert!(vp8_packet_displays(&[]));
    }

    #[test]
    #[ignore = "needs an NVIDIA GPU and driver (NVDEC)"]
    fn odd_sized_av1_is_refused() {
        // NVDEC would scale it to an even size rather than crop it.
        let r = decode_sampled(&fixture("av1_odd.mp4"), 1, &VideoDecodeLimits::default());
        assert!(matches!(r, Err(MmError::Unsupported { .. })), "{r:?}");
    }

    // -- Keyframe seeking: the same frames as decoding everything ------------

    #[cfg(target_os = "linux")]
    mod seeking {
        use super::*;

        /// Every frame on its own, and sets of frames, decode to the same bytes
        /// and times with seeking as in one decode of the whole stream.
        fn assert_seeking_changes_nothing(backend: Backend, name: &str) {
            let bytes = fixture(name);
            let limits = VideoDecodeLimits::default();
            let n = probe(&bytes).unwrap().frame_count;
            let decode = |indices: &[u64], seek: bool| {
                seek::DISABLED.set(!seek);
                let frames = decode_frames_with(&bytes.as_slice(), indices, None, &limits, backend);
                seek::DISABLED.set(false);
                frames.unwrap_or_else(|e| panic!("{name} {indices:?}: {e}"))
            };
            let all: Vec<u64> = (0..n).collect();
            let whole = decode(&all, false);
            // Checked against the source too, not only against the whole
            // decode: frame i of a gop_* clip has grey level `16 + 5 i` in
            // its left 32 columns (see generate.py).
            if name.starts_with("gop_") {
                for i in 0..n {
                    let grey = band_grey(whole.frame(i as usize).unwrap(), whole.width());
                    assert!(
                        grey.abs_diff(16 + 5 * i) <= 6,
                        "{name} frame {i}: grey {grey}"
                    );
                }
            }
            let mut sets: Vec<Vec<u64>> = (0..n).map(|i| vec![i]).collect();
            sets.extend([
                sample_indices(n, 4).unwrap(),
                sample_indices(n, n.min(16)).unwrap(),
                vec![0, n - 1],
                vec![n / 2, n / 2, n - 1],
                all.clone(),
            ]);
            for set in sets {
                let frames = decode(&set, true);
                for (k, &i) in set.iter().enumerate() {
                    assert!(
                        frames.frame(k) == whole.frame(i as usize),
                        "{name}: frame {i} differs when decoding {set:?}"
                    );
                    assert_eq!(
                        frames.timestamps()[k],
                        whole.timestamps()[i as usize],
                        "{name}: frame {i} time, decoding {set:?}"
                    );
                }
            }
        }

        /// The mean of the left 32 columns of an RGB frame `width` wide.
        fn band_grey(frame: &[u8], width: usize) -> u64 {
            let rows = frame.chunks(width * 3);
            let count = rows.len() as u64 * 32 * 3;
            let sum: u64 = rows
                .flat_map(|row| &row[..32 * 3])
                .map(|&v| u64::from(v))
                .sum();
            sum / count
        }

        #[test]
        #[ignore = "needs an NVIDIA GPU and driver (NVDEC)"]
        fn nvdec_frames_are_the_same_with_seeking() {
            for name in [
                "gop_h264.mp4",
                "gop_frag.mp4",
                "gop_h264_open.mp4",
                "gop_hevc.mp4",
                "gop_hev1.mp4",
                "gop_vp9.webm",
                "gop_av1.mp4",
                "h264.mp4",
                "hevc.mp4",
                "av1.mp4",
                "cut.mp4",
                "delay.mp4",
                "vp9_altref.webm",
                "vp9_invisible.webm",
            ] {
                assert_seeking_changes_nothing(nvdec::decode, name);
            }
        }

        #[test]
        #[ignore = "needs an NVIDIA GPU and driver (NVDEC)"]
        fn a_stream_that_starts_at_a_cra_picture_leaves_out_its_rasl_pictures() {
            // gop_hevc.mp4 from its CRA picture at frame 24, as a cut or a
            // segment would start: the two RASL pictures after it reference
            // frames that are gone. Shown, they cannot be decoded; hidden (as
            // an edit list from the CRA picture hides them), every frame is
            // the one a decode of the whole clip gives, with seeking or not.
            let bytes = fixture("gop_hevc.mp4");
            let slice = bytes.as_slice();
            let limits = VideoDecodeLimits::default();
            let full = demux(&slice).unwrap();
            let cra = (0..full.packets.len())
                .filter(|&i| full.packets[i].sync)
                .nth(3)
                .unwrap();
            let all: Vec<u64> = (0..full.info().frame_count).collect();
            let whole = decode_demuxed(&full, &all, None, &limits, nvdec::decode).unwrap();
            let originals = seek::Order::new(&full).shown[cra..].to_vec();
            for hidden in [false, true] {
                let mut track = demux(&slice).unwrap();
                track.packets.drain(..cra);
                if hidden {
                    track.packets[1].shown = false;
                    track.packets[2].shown = false;
                }
                let order = seek::Order::new(&track);
                for seek in [false, true] {
                    seek::DISABLED.set(!seek);
                    for k in 0..track.info().frame_count {
                        let packet = (0..track.packets.len())
                            .find(|&i| order.shown[i] == Some(k))
                            .unwrap();
                        let r = decode_demuxed(&track, &[k], None, &limits, nvdec::decode);
                        if packet == 1 || packet == 2 {
                            assert!(r.is_err(), "RASL picture {k} decoded");
                            continue;
                        }
                        let frames = r.unwrap_or_else(|e| panic!("frame {k}: {e}"));
                        let original = originals[packet].unwrap() as usize;
                        assert!(
                            frames.frame(0) == whole.frame(original),
                            "frame {k} (hidden {hidden}, seek {seek}) is not frame {original}"
                        );
                        assert_eq!(frames.timestamps()[0], whole.timestamps()[original]);
                    }
                    seek::DISABLED.set(false);
                }
            }
        }

        #[test]
        fn cpu_frames_are_the_same_with_seeking() {
            for name in [
                "gop_vp8.webm",
                "gop_vp9.webm",
                "vp9_altref.webm",
                "vp9_invisible.webm",
            ] {
                assert_seeking_changes_nothing(vpx::decode, name);
            }
        }
    }
}
