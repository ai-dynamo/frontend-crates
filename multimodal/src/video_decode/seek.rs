// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Keyframe seeking: which packets to decode to reach the sampled frames.
//!
//! A keyframe needs no earlier packet, so a sampled frame needs only the
//! packets from the last keyframe before it. A decode is planned as runs of
//! packets, each starting at a keyframe, and the packets between runs are
//! never decoded (in MP4 never read either, unless the track may carry
//! parameter sets in its samples; see below). Only whole groups of pictures
//! without a sampled frame are skipped: neighbouring groups that both hold
//! one decode straight through.
//!
//! The container says where the keyframes are (`stss` and fragment flags in
//! MP4, the block keyframe flag in Matroska), but that is input like any other.
//! Each keyframe a run starts at is first checked in its bitstream: an H.264
//! IDR picture, an HEVC IRAP picture, a VP8 or VP9 key frame, or a shown AV1
//! key frame. FFmpeg trusts the container here and relies on its decoders to
//! hold back frames decoded from a wrong start; NVDEC does not report that, so
//! the check is made before decoding. If any start fails it, the plan is the
//! whole stream, decoded as without seeking: seeking changes how much is
//! decoded, never which frames come out.
//!
//! An HEVC CRA picture can be followed in decode order by RASL pictures that
//! display before it and reference the group before. A sampled frame that
//! displays before its keyframe, or is a RASL picture, starts its run one
//! keyframe earlier. RASL pictures that cannot be decoded are not fed to the
//! decoder at all (see [`Leading`]), in a run from the first packet too.
//!
//! H.264 and HEVC parameter sets can also travel in the samples (an MP4 `hev1`
//! sample entry, or Matroska). A run that starts later is preceded by the
//! parameter sets of every packet before it, so the decoder starts the run
//! with the state a whole decode would have.

use std::collections::HashMap;

use super::{
    START_CODE, Track, VideoCodec, bitstream_prefix, vp8_keyframe_size, vp9_frames,
    vp9_keyframe_colour,
};
use crate::{MmError, Result};

/// Most bytes of in-band parameter sets, as Annex B, carried to the runs
/// together (each run gets its own copy); past it the plan is the whole
/// stream. A stream's sets are typically a few hundred bytes.
const MAX_PARAMETER_SET_BYTES: usize = 1 << 20;

/// The packets that display a frame, in display order, and each packet's place
/// in it.
pub(super) struct Order {
    /// Packets that display a frame, in display order: sorted by time for
    /// H.264 and HEVC, whose frames reorder, else in decode order.
    pub displayed: Vec<usize>,
    /// Each packet's position in `displayed` (0 for one that displays nothing).
    pub rank: Vec<usize>,
    /// Each packet's index among the shown frames, which the sampler counts
    /// in; `None` for a packet that shows nothing.
    pub shown: Vec<Option<u64>>,
}

impl Order {
    pub fn new(track: &Track<'_>) -> Self {
        let packets = &track.packets;
        let mut displayed: Vec<usize> = (0..packets.len())
            .filter(|&i| packets[i].displays)
            .collect();
        if reorders(track.codec) {
            displayed.sort_by(|&a, &b| packets[a].pts.total_cmp(&packets[b].pts));
        }
        let mut rank = vec![0; packets.len()];
        let mut shown = vec![None; packets.len()];
        let mut next = 0;
        for (r, &i) in displayed.iter().enumerate() {
            rank[i] = r;
            if packets[i].shown {
                shown[i] = Some(next);
                next += 1;
            }
        }
        Self {
            displayed,
            rank,
            shown,
        }
    }
}

/// Packets `start..end`, decoded from a fresh decoder state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Run {
    pub start: usize,
    pub end: usize,
    /// In-band parameter sets of the packets before `start`, as Annex B, to
    /// hand the decoder before the run's first packet.
    pub parameter_sets: Vec<u8>,
}

fn reorders(codec: VideoCodec) -> bool {
    matches!(codec, VideoCodec::H264 | VideoCodec::Hevc)
}

#[cfg(test)]
thread_local! {
    /// Tests compare a seeking decode with a whole one on the same thread.
    pub(super) static DISABLED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

fn enabled() -> bool {
    #[cfg(test)]
    if DISABLED.get() {
        return false;
    }
    true
}

/// The runs that decode the shown frames `targets` (non-decreasing indices).
pub(super) fn plan(track: &Track<'_>, order: &Order, targets: &[u64]) -> Result<Vec<Run>> {
    let packets = &track.packets;
    let whole = || {
        vec![Run {
            start: 0,
            end: packets.len(),
            parameter_sets: Vec::new(),
        }]
    };
    if !enabled() || targets.is_empty() {
        return Ok(whole());
    }
    let by_shown: Vec<usize> = order
        .displayed
        .iter()
        .copied()
        .filter(|&i| packets[i].shown)
        .collect();
    let keys: Vec<usize> = (0..packets.len()).filter(|&i| packets[i].sync).collect();
    let mut nal = NalReader::new(track);
    let mut runs: Vec<Run> = Vec::new();
    let mut last_packet = None;
    for &target in targets {
        let packet = *usize::try_from(target)
            .ok()
            .and_then(|t| by_shown.get(t))
            .ok_or_else(|| MmError::internal("sampled frame past the shown frames"))?;
        if last_packet == Some(packet) {
            continue;
        }
        last_packet = Some(packet);
        // The last keyframe at or before the packet in decode order, or one
        // earlier when the frame displays before that keyframe or is an HEVC
        // RASL picture; the first packet when there is none.
        let at = keys.partition_point(|&s| s <= packet);
        let mut leading = |key: usize| -> Result<bool> {
            Ok(reorders(track.codec)
                && (packets[key].pts > packets[packet].pts
                    || (track.codec == VideoCodec::Hevc
                        && nal.first_vcl(packet)?.is_some_and(hevc_rasl))))
        };
        let start = match at.checked_sub(1).map(|k| (k, keys[k])) {
            None => 0,
            Some((k, key)) if leading(key)? => match k.checked_sub(1).map(|k| keys[k]) {
                None => 0,
                // Still after the frame: timestamps that do not follow the
                // pictures. Not worth seeking in.
                Some(earlier) if packets[earlier].pts > packets[packet].pts => {
                    return Ok(whole());
                }
                Some(earlier) => earlier,
            },
            Some((_, key)) => key,
        };
        // Join the run before when no whole group lies between them: this
        // one meets or overlaps it (it can start earlier, when its frame
        // displays before its keyframe), or starts at the first keyframe
        // after it, so only the rest of that run's group is in between.
        let mut run = Run {
            start,
            end: packet + 1,
            parameter_sets: Vec::new(),
        };
        while let Some(before) = runs.last() {
            let next_key = keys
                .get(keys.partition_point(|&s| s < before.end))
                .copied()
                .unwrap_or(usize::MAX);
            if run.start > before.end && run.start > next_key {
                break;
            }
            run.start = run.start.min(before.start);
            run.end = run.end.max(before.end);
            runs.pop();
        }
        runs.push(run);
    }
    // Check each keyframe a run starts at, once the runs are final.
    for run in &runs {
        if run.start != 0 && !nal.is_key(run.start)? {
            return Ok(whole());
        }
    }
    if nal.inband && runs.iter().any(|run| run.start != 0) {
        let mut sets = ParameterSets::default();
        let mut scanned = 0;
        let mut carried = 0usize;
        for run in &mut runs {
            for index in scanned..run.start {
                if !nal.parameter_sets(index, &mut sets)? {
                    return Ok(whole());
                }
            }
            scanned = scanned.max(run.start);
            carried = carried.saturating_add(sets.bytes);
            if carried > MAX_PARAMETER_SET_BYTES {
                return Ok(whole());
            }
            run.parameter_sets = sets.annexb();
        }
    }
    Ok(runs)
}

/// The H.264 / HEVC parameter set NAL units seen so far, each kept once, in
/// the order of its last appearance (a later one replaces an earlier one with
/// the same id, so the decoder must see it last).
#[derive(Default)]
struct ParameterSets {
    /// Each NAL unit, and when it last appeared.
    last: HashMap<Vec<u8>, usize>,
    added: usize,
    /// Their size as Annex B.
    bytes: usize,
}

impl ParameterSets {
    /// Keep `nal`; `false`, keeping nothing, when it would take the sets past
    /// the cap.
    fn add(&mut self, nal: &[u8]) -> bool {
        self.added += 1;
        if let Some(at) = self.last.get_mut(nal) {
            *at = self.added;
            return true;
        }
        let bytes = self.bytes + START_CODE.len() + nal.len();
        if bytes > MAX_PARAMETER_SET_BYTES {
            return false;
        }
        self.bytes = bytes;
        self.last.insert(nal.to_vec(), self.added);
        true
    }

    fn annexb(&self) -> Vec<u8> {
        let mut nals: Vec<_> = self.last.iter().collect();
        nals.sort_unstable_by_key(|&(_, &at)| at);
        let mut out = Vec::with_capacity(self.bytes);
        for (nal, _) in nals {
            out.extend_from_slice(&START_CODE);
            out.extend_from_slice(nal);
        }
        out
    }
}

/// Reads packets' bitstream headers for planning.
struct NalReader<'t, 'p> {
    track: &'t Track<'p>,
    /// The NAL length size of H.264 and HEVC samples; `None` if the codec
    /// configuration does not parse (no keyframe passes then).
    length_size: Option<usize>,
    /// Whether H.264 / HEVC parameter sets may travel in the samples.
    inband: bool,
    /// Whether the AV1 sequence header in the codec configuration is a still
    /// picture's reduced one, where every frame is a key frame.
    av1_reduced: bool,
    buffer: Vec<u8>,
}

impl<'t, 'p> NalReader<'t, 'p> {
    fn new(track: &'t Track<'p>) -> Self {
        let length_size = bitstream_prefix(track).ok().and_then(|(_, size)| size);
        let av1_reduced = track.codec == VideoCodec::Av1
            && track
                .config
                .as_deref()
                .and_then(|c| c.get(4..))
                .is_some_and(|obus| av1_reduced_header(obus).unwrap_or(false));
        Self {
            track,
            length_size,
            inband: track.inband_parameter_sets && reorders(track.codec),
            av1_reduced,
            buffer: Vec::new(),
        }
    }

    fn bytes(&mut self, index: usize) -> Result<&[u8]> {
        self.track
            .packet_bytes(&self.track.packets[index], &mut self.buffer)
    }

    /// The type of packet `index`'s first coded-slice NAL unit (H.264, HEVC).
    fn first_vcl(&mut self, index: usize) -> Result<Option<u8>> {
        let (codec, length_size) = (self.track.codec, self.length_size);
        let data = self.bytes(index)?;
        Ok(length_size.and_then(|size| first_vcl_nal(codec, data, size)))
    }

    fn is_key(&mut self, index: usize) -> Result<bool> {
        let (codec, av1_reduced) = (self.track.codec, self.av1_reduced);
        if reorders(codec) {
            let kind = self.first_vcl(index)?;
            return Ok(match codec {
                VideoCodec::H264 => kind == Some(5),
                _ => kind.is_some_and(hevc_irap),
            });
        }
        let data = self.bytes(index)?;
        Ok(match codec {
            VideoCodec::Vp8 => vp8_keyframe_size(data).is_some(),
            VideoCodec::Vp9 => vp9_frames(data)
                .ok()
                .and_then(|frames| frames.first().copied())
                .is_some_and(|frame| vp9_keyframe_colour(frame).is_some()),
            VideoCodec::Av1 => av1_key_frame(data, av1_reduced),
            VideoCodec::H264 | VideoCodec::Hevc => false,
        })
    }

    /// Add packet `index`'s parameter set NAL units to `sets`; `false` once
    /// they are past the cap.
    fn parameter_sets(&mut self, index: usize, sets: &mut ParameterSets) -> Result<bool> {
        let (codec, length_size) = (self.track.codec, self.length_size);
        let Some(size) = length_size else {
            return Ok(true);
        };
        for nal in nals(self.bytes(index)?, size) {
            if nal.first().is_some_and(|&h| parameter_set(codec, h)) && !sets.add(nal) {
                return Ok(false);
            }
        }
        Ok(true)
    }
}

/// Decides which HEVC RASL pictures a run leaves out: those whose CRA is the
/// run's first picture, or whose IRAP is a BLA. The decoder discards them
/// (they reference pictures before it, which it never had), but NVDEC still
/// hands their timestamps to the frames after them, so they are not fed. No
/// later picture references them. Other codecs keep every packet.
pub(super) struct Leading {
    length_size: Option<usize>,
    /// Whether the last IRAP picture discards its RASL pictures.
    discards: bool,
}

impl Leading {
    pub fn new(track: &Track<'_>) -> Self {
        let length_size = match track.codec {
            VideoCodec::Hevc => bitstream_prefix(track).ok().and_then(|(_, size)| size),
            _ => None,
        };
        Self {
            length_size,
            discards: false,
        }
    }

    /// Whether to leave out packet `data`, the run's first when `first`.
    pub fn skips(&mut self, data: &[u8], first: bool) -> bool {
        let Some(kind) = self
            .length_size
            .and_then(|size| first_vcl_nal(VideoCodec::Hevc, data, size))
        else {
            return false;
        };
        if hevc_irap(kind) {
            self.discards = (first && kind == HEVC_CRA) || (16..=18).contains(&kind);
            return false;
        }
        if first {
            self.discards = false;
        }
        self.discards && hevc_rasl(kind)
    }
}

const HEVC_CRA: u8 = 21;

/// BLA, IDR and CRA pictures: decoding can start at them.
fn hevc_irap(kind: u8) -> bool {
    (16..=21).contains(&kind)
}

fn hevc_rasl(kind: u8) -> bool {
    matches!(kind, 8 | 9)
}

/// Whether a NAL unit header is a parameter set's: H.264 SPS and PPS, HEVC
/// VPS, SPS and PPS.
fn parameter_set(codec: VideoCodec, header: u8) -> bool {
    match codec {
        VideoCodec::H264 => matches!(header & 0x1f, 7 | 8),
        _ => matches!((header >> 1) & 0x3f, 32..=34),
    }
}

/// The NAL units of a length-prefixed sample, up to the first that does not
/// parse.
fn nals(sample: &[u8], length_size: usize) -> impl Iterator<Item = &[u8]> {
    let mut pos = 0usize;
    std::iter::from_fn(move || {
        let len = sample
            .get(pos..pos.checked_add(length_size)?)?
            .iter()
            .fold(0usize, |n, b| (n << 8) | usize::from(*b));
        pos += length_size;
        let nal = sample.get(pos..pos.checked_add(len)?)?;
        pos += len;
        Some(nal)
    })
}

/// The type of the first coded-slice NAL unit in a length-prefixed H.264 or
/// HEVC sample.
fn first_vcl_nal(codec: VideoCodec, sample: &[u8], length_size: usize) -> Option<u8> {
    nals(sample, length_size).find_map(|nal| {
        let header = *nal.first()?;
        match codec {
            VideoCodec::H264 => Some(header & 0x1f).filter(|kind| (1..=5).contains(kind)),
            _ => Some((header >> 1) & 0x3f).filter(|&kind| kind < 32),
        }
    })
}

/// The OBUs of an AV1 temporal unit, each as its type and payload. Stops at
/// the first one that does not parse.
fn av1_obus(data: &[u8]) -> impl Iterator<Item = (u8, &[u8])> {
    let mut pos = 0;
    std::iter::from_fn(move || {
        let header = *data.get(pos)?;
        let kind = (header >> 3) & 0xf;
        let extension = header & 0x4 != 0;
        let has_size = header & 0x2 != 0;
        pos += 1 + usize::from(extension);
        let size = if has_size {
            let mut size = 0usize;
            let mut i = 0;
            loop {
                let byte = *data.get(pos)?;
                size |= usize::from(byte & 0x7f) << (7 * i);
                pos += 1;
                i += 1;
                if byte & 0x80 == 0 {
                    break;
                }
                if i == 8 {
                    return None;
                }
            }
            size
        } else {
            data.len().checked_sub(pos)?
        };
        let payload = data.get(pos..pos.checked_add(size)?)?;
        pos += size;
        Some((kind, payload))
    })
}

const OBU_SEQUENCE_HEADER: u8 = 1;
const OBU_FRAME_HEADER: u8 = 3;
const OBU_FRAME: u8 = 6;

/// `reduced_still_picture_header` of the first sequence header among `obus`.
fn av1_reduced_header(obus: &[u8]) -> Option<bool> {
    av1_obus(obus)
        .find(|&(kind, _)| kind == OBU_SEQUENCE_HEADER)
        // seq_profile (3 bits), still_picture (1), reduced_still_picture_header (1).
        .and_then(|(_, payload)| payload.first().map(|b| b & 0x08 != 0))
}

/// Whether an AV1 temporal unit's first frame is a shown key frame. A sequence
/// header in the unit overrides the configuration's (`reduced`).
fn av1_key_frame(data: &[u8], reduced: bool) -> bool {
    let mut reduced = reduced;
    for (kind, payload) in av1_obus(data) {
        match kind {
            OBU_SEQUENCE_HEADER => {
                reduced = payload.first().is_some_and(|b| b & 0x08 != 0);
            }
            OBU_FRAME_HEADER | OBU_FRAME => {
                if reduced {
                    return true;
                }
                // show_existing_frame (1 bit), frame_type (2), show_frame (1).
                return payload
                    .first()
                    .is_some_and(|b| b & 0x80 == 0 && (b >> 5) & 0x3 == 0 && b & 0x10 != 0);
            }
            _ => {}
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use super::super::{ByteSource, Packet, Payload, demux};
    use super::*;

    fn fixture(name: &str) -> Vec<u8> {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/video_decode")
            .join(name);
        std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
    }

    fn keys(track: &Track<'_>) -> Vec<usize> {
        (0..track.packets.len())
            .filter(|&i| track.packets[i].sync)
            .collect()
    }

    fn packet_of(track: &Track<'_>, order: &Order, target: u64) -> usize {
        (0..track.packets.len())
            .find(|&i| order.shown[i] == Some(target))
            .unwrap()
    }

    /// The `(start, end)` of the runs that decode `targets`, and the packet
    /// each target is in.
    fn plan_of(track: &Track<'_>, targets: &[u64]) -> (Vec<(usize, usize)>, Vec<usize>) {
        let order = Order::new(track);
        let packets = targets
            .iter()
            .map(|&t| packet_of(track, &order, t))
            .collect();
        let runs = plan(track, &order, targets).unwrap();
        (runs.iter().map(|r| (r.start, r.end)).collect(), packets)
    }

    fn spans_of(bytes: &[u8], targets: &[u64]) -> (Vec<(usize, usize)>, Vec<usize>) {
        plan_of(&demux(&bytes).unwrap(), targets)
    }

    fn whole(bytes: &[u8]) -> Vec<(usize, usize)> {
        vec![(0, demux(&bytes).unwrap().packets.len())]
    }

    // The clips gop_*.* hold forty frames with a keyframe every eight; see
    // generate.py.

    #[test]
    fn containers_mark_keyframes_that_the_bitstream_confirms() {
        for name in [
            "gop_h264.mp4",
            "gop_frag.mp4",
            "gop_hevc.mp4",
            "gop_hev1.mp4",
            "gop_vp8.webm",
            "gop_vp9.webm",
            "gop_av1.mp4",
        ] {
            let bytes = fixture(name);
            let track = demux(&bytes).unwrap();
            let keys = keys(&track);
            assert_eq!(keys.len(), 5, "{name}: {keys:?}");
            let mut nal = NalReader::new(&track);
            for i in 0..track.packets.len() {
                assert_eq!(
                    nal.is_key(i).unwrap(),
                    keys.contains(&i),
                    "{name} packet {i}"
                );
            }
        }
        // Open-GOP H.264 marks I pictures that are not IDR pictures.
        let bytes = fixture("gop_h264_open.mp4");
        let track = demux(&bytes).unwrap();
        let mut nal = NalReader::new(&track);
        let confirmed: Vec<bool> = keys(&track)
            .iter()
            .map(|&k| nal.is_key(k).unwrap())
            .collect();
        assert_eq!(confirmed, [true, false, false, false, false]);
    }

    #[test]
    fn mp4_without_stss_marks_every_sample_and_an_empty_one_none() {
        let fixture_bytes = fixture("gop_h264.mp4");
        let at = fixture_bytes.windows(4).position(|w| w == b"stss").unwrap();
        let sync = |bytes: &[u8]| -> Vec<bool> {
            demux(&bytes)
                .unwrap()
                .packets
                .iter()
                .map(|p| p.sync)
                .collect()
        };
        // Without `stss` every sample is a keyframe.
        let mut bytes = fixture_bytes.clone();
        bytes[at..at + 4].copy_from_slice(b"free");
        assert!(sync(&bytes).iter().all(|&s| s));
        // An empty one: none.
        let mut bytes = fixture_bytes.clone();
        bytes[at + 8..at + 12].copy_from_slice(&0u32.to_be_bytes());
        assert!(sync(&bytes).iter().all(|&s| !s));
    }

    #[test]
    fn groups_without_sampled_frames_are_skipped() {
        let bytes = fixture("gop_h264.mp4");
        let (runs, packets) = spans_of(&bytes, &[0, 39]);
        assert_eq!(runs, [(0, packets[0] + 1), (32, packets[1] + 1)]);
        // Frames in neighbouring groups decode straight through.
        let (runs, packets) = spans_of(&bytes, &[12, 20]);
        assert_eq!(runs, [(8, packets[1] + 1)]);
        // Repeated and nearby frames share their run.
        let (runs, _) = spans_of(&bytes, &[9, 9, 10]);
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0].0, 8);
    }

    #[test]
    fn a_rasl_frame_starts_a_keyframe_earlier() {
        // gop_hevc.mp4: the CRA picture at frame 24 is decoded before RASL
        // pictures 22 and 23, which reference the group before it.
        let bytes = fixture("gop_hevc.mp4");
        let keys = keys(&demux(&bytes.as_slice()).unwrap());
        let (runs, packets) = spans_of(&bytes, &[22]);
        assert!(packets[0] > keys[3], "frame 22 follows the CRA picture");
        assert_eq!(runs, [(keys[2], packets[0] + 1)]);
        // Also when its time says otherwise: the bitstream decides.
        let slice = bytes.as_slice();
        let mut track = demux(&slice).unwrap();
        let packet = packets[0];
        track.packets[packet].pts = track.packets[keys[3]].pts + 0.01;
        assert_eq!(plan_of(&track, &[22]).0[0].0, keys[2]);
        // Frame 24 itself starts at its own keyframe.
        let (runs, _) = spans_of(&bytes, &[24]);
        assert_eq!(runs[0].0, keys[3]);
    }

    #[test]
    fn rasl_frames_are_left_out_only_after_a_cra_the_run_starts_at() {
        let bytes = fixture("gop_hevc.mp4");
        let track = demux(&bytes).unwrap();
        let keys = keys(&track);
        let skipped = |start: usize| -> Vec<usize> {
            let mut leading = Leading::new(&track);
            (start..track.packets.len())
                .filter(|&i| {
                    let data = track
                        .packet_bytes(&track.packets[i], &mut Vec::new())
                        .unwrap()
                        .to_vec();
                    leading.skips(&data, i == start)
                })
                .collect()
        };
        // From the CRA at frame 24: its RASL pictures (frames 22, 23).
        assert_eq!(skipped(keys[3]), [keys[3] + 1, keys[3] + 2]);
        // From an earlier keyframe they decode.
        assert!(skipped(keys[2]).is_empty());
        assert!(skipped(0).is_empty());
    }

    #[test]
    fn keyframes_are_confirmed_after_runs_join() {
        // gop_h264_open.mp4: only the first keyframe is an IDR picture. A run
        // that joins the first one never starts at the others.
        let bytes = fixture("gop_h264_open.mp4");
        let (runs, packets) = spans_of(&bytes, &[3, 12]);
        assert_eq!(runs, [(0, packets[1] + 1)]);
        // One that starts at an I picture turns seeking off.
        assert_eq!(spans_of(&bytes, &[39]).0, whole(&bytes));
        // So does an stss that names an inter frame (sample 10 for 9).
        let mut bytes = fixture("gop_h264.mp4");
        let at = bytes.windows(4).position(|w| w == b"stss").unwrap() + 16;
        assert_eq!(bytes[at..at + 4], 9u32.to_be_bytes());
        bytes[at..at + 4].copy_from_slice(&10u32.to_be_bytes());
        assert_eq!(spans_of(&bytes, &[12]).0, whole(&bytes));
        // And timestamps that put both candidate keyframes after the frame.
        let bytes = fixture("gop_h264.mp4");
        let slice = bytes.as_slice();
        let mut track = demux(&slice).unwrap();
        let keys = keys(&track);
        for &k in &keys[1..3] {
            track.packets[k].pts = 1e9;
        }
        let n = track.packets.len();
        assert_eq!(plan_of(&track, &[20]).0, [(0, n)]);
        // Disabled for tests that compare with a whole decode.
        DISABLED.set(true);
        let (runs, _) = spans_of(&bytes, &[39]);
        DISABLED.set(false);
        assert_eq!(runs, whole(&bytes));
    }

    /// A source that counts its reads.
    struct Counted<'a> {
        bytes: &'a [u8],
        reads: Cell<usize>,
    }

    impl ByteSource for Counted<'_> {
        fn len(&self) -> u64 {
            self.bytes.len() as u64
        }

        fn read_at(&self, offset: u64, buf: &mut [u8]) -> std::io::Result<()> {
            self.reads.set(self.reads.get() + 1);
            ByteSource::read_at(&self.bytes, offset, buf)
        }
    }

    #[test]
    fn each_keyframe_is_read_once_and_skipped_groups_never() {
        let bytes = fixture("gop_h264.mp4");
        let source = Counted {
            bytes: &bytes,
            reads: Cell::new(0),
        };
        let track = demux(&source).unwrap();
        let order = Order::new(&track);
        source.reads.set(0);
        plan(&track, &order, &[12; 40]).unwrap();
        assert_eq!(source.reads.get(), 1);
        source.reads.set(0);
        plan(&track, &order, &[1, 20, 39]).unwrap();
        assert_eq!(source.reads.get(), 2, "keyframes 16 and 32");
    }

    #[test]
    fn a_later_run_gets_the_parameter_sets_sent_before_it() {
        let parameter_sets = |name: &str, targets: &[u64]| -> Vec<Vec<u8>> {
            let bytes = fixture(name);
            let track = demux(&bytes).unwrap();
            let order = Order::new(&track);
            plan(&track, &order, targets)
                .unwrap()
                .into_iter()
                .map(|run| run.parameter_sets)
                .collect()
        };
        // gop_hev1.mp4 repeats VPS, SPS and PPS at every keyframe: a run from
        // frame 32 gets them once each.
        let sets = parameter_sets("gop_hev1.mp4", &[0, 39]);
        assert!(sets[0].is_empty());
        let nal_types: Vec<u8> = sets[1]
            .windows(5)
            .filter(|w| w[..4] == [0, 0, 0, 1])
            .map(|w| (w[4] >> 1) & 0x3f)
            .collect();
        assert_eq!(nal_types, [32, 33, 34]);
        // hvc1 keeps them all in the configuration record.
        assert!(
            parameter_sets("gop_hevc.mp4", &[39])
                .iter()
                .all(Vec::is_empty)
        );
    }

    /// The track of `gop_hev1.mp4` (in `source`) with `count` made-up packets
    /// at 30 fps, each a slice led by the NAL units `sets` gives for it: an
    /// IDR picture every eight packets, else a trailing picture.
    fn hevc_track<'a>(
        source: &'a dyn ByteSource,
        count: usize,
        sets: impl Fn(usize) -> Vec<Vec<u8>>,
    ) -> Track<'a> {
        let mut track = demux(source).unwrap();
        assert!(track.inband_parameter_sets);
        assert_eq!(bitstream_prefix(&track).unwrap().1, Some(4));
        let prefixed = |nal: &[u8]| [&(nal.len() as u32).to_be_bytes()[..], nal].concat();
        track.packets = (0..count)
            .map(|i| {
                let key = i % 8 == 0;
                let slice = if key {
                    [0x26, 0x01, 0x80]
                } else {
                    [0x02, 0x01, 0x80]
                };
                let mut data: Vec<u8> = sets(i).iter().flat_map(|n| prefixed(n)).collect();
                data.extend(prefixed(&slice));
                Packet::new(Payload::Owned(data), i as f64 / 30.0, true, false, key)
            })
            .collect();
        track
    }

    fn spans(track: &Track<'_>, targets: &[u64]) -> Vec<(usize, usize)> {
        let order = Order::new(track);
        let runs = plan(track, &order, targets).unwrap();
        runs.iter().map(|r| (r.start, r.end)).collect()
    }

    #[test]
    fn many_different_parameter_sets_stop_the_scan_at_the_cap() {
        // A different PPS in every packet: past the cap after about 100,000,
        // found as each is added (adding is not a search of those before).
        let bytes = fixture("gop_hev1.mp4");
        let source = bytes.as_slice();
        let count = 1 << 17;
        let pps = |i: usize| vec![[&[0x44, 0x01][..], &(i as u32).to_be_bytes()].concat()];
        let track = hevc_track(&source, count, pps);
        assert_eq!(spans(&track, &[count as u64 - 1]), [(0, count)]);
    }

    #[test]
    fn parameter_sets_are_capped_across_the_runs_they_are_copied_to() {
        // One 400 KB PPS in the first packet, copied to every later run.
        let bytes = fixture("gop_hev1.mp4");
        let source = bytes.as_slice();
        let big = [&[0x44, 0x01][..], &[7; 400_000]].concat();
        let track = hevc_track(
            &source,
            64,
            |i| {
                if i == 0 { vec![big.clone()] } else { vec![] }
            },
        );
        assert_eq!(spans(&track, &[20, 44]), [(16, 21), (40, 45)]);
        assert_eq!(spans(&track, &[20, 44, 60]), [(0, 64)]);
    }

    #[test]
    fn av1_key_frames_are_read_from_obus() {
        // OBU_FRAME with show_existing_frame 0, frame_type KEY_FRAME, show_frame 1.
        let key = [0x32, 0x01, 0x10];
        assert!(av1_key_frame(&key, false));
        // An inter frame, and a hidden key frame.
        assert!(!av1_key_frame(&[0x32, 0x01, 0x30], false));
        assert!(!av1_key_frame(&[0x32, 0x01, 0x00], false));
        // Every frame of a reduced still-picture header is a key frame.
        assert!(av1_key_frame(&[0x32, 0x01, 0x30], true));
        // A truncated unit is not.
        assert!(!av1_key_frame(&[0x32, 0x05, 0x10], false));
        assert!(!av1_key_frame(&[], false));
    }
}
