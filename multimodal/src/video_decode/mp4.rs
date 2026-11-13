// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! MP4 (ISO BMFF) demuxing: the sample table of the first video track.
//!
//! Only what the decoder needs is read, and everything else is skipped by box
//! size, never parsed: `udta`, `meta`, `emsg`, sample groups, `sdtp`, `stps`,
//! audio and text tracks.
//!
//! Keyframes are read as FFmpeg reads them: the samples `stss` lists, every
//! sample when there is no `stss`, and in a fragment those whose flags do not
//! say "not a sync sample" or "depends on others". They only tell the decoder
//! where it may start (see `seek`), which checks each one in the bitstream
//! first, so an `stss` that does not parse turns seeking off instead of
//! refusing a file that decodes.
//!
//! Times are player times. The video track's edit list (`edts/elst`) decides
//! which frames show and when: frames outside its window are decoded (later
//! frames may reference them) but not shown, the window starts at time 0, and
//! a leading empty edit delays it (`start_time`). Lists a player would need to
//! seek within are refused; see [`window`]. Without an edit list, time starts
//! at the first shown frame.
//!
//! Input is untrusted, so the parser fails closed. A box that is too small,
//! overshoots its parent or nests wrongly, bytes after the last box, a table
//! its box cannot hold, tables that disagree on the sample count, an index out
//! of range, arithmetic that overflows, a fragment whose decode time goes back
//! and a sample outside the media data are all [`MmError::InvalidInput`].
//! Anything valid but not handled (several sample descriptions, media data in
//! another file, encrypted entries, `stz2`, unknown `tfhd` / `trun` flags) is
//! [`MmError::Unsupported`]. No loop here can fail to advance, nothing is
//! indexed without a bounds check and sizes come from checked arithmetic, so no
//! input panics or hangs.
//!
//! Boxes nest to a fixed depth (`moov/trak/mdia/minf/stbl`), so the parser is
//! not recursive. Only `moov` and `moof` bodies are read, each into memory.
//! What a file may cost is capped whatever it declares: top-level boxes
//! ([`MAX_BOXES`]), a movie box ([`MAX_BOX_BODY`]), samples ([`MAX_SAMPLES`])
//! and a sample ([`MAX_SAMPLE_SIZE`]). More boxes or a larger sample than
//! the cap is [`MmError::LimitExceeded`]; a movie box or a sample count past
//! its cap is malformed, as no file this parser accepts holds one.

use std::collections::HashSet;

use super::{ByteSource, VideoCodec, read_error};
use crate::{MmError, Result};

/// Most top-level boxes: each costs a header read, and each `mdat` a 16-byte
/// entry in the media list, before any sample cap applies, so a file costs
/// at most this many reads and 16 MiB whatever its length. A fragment is two
/// boxes (four with CMAF's `styp` and `sidx`), so this is 2^19 fragments: a
/// fragment for every 16 samples at [`MAX_SAMPLES`], or two to five hours of
/// one frame per fragment at 30 fps.
const MAX_BOXES: u64 = 1 << 20;
/// Largest `moov` or `moof` body read into memory.
const MAX_BOX_BODY: u64 = 256 << 20;
/// Most samples in a track: a sample table costs memory per sample.
const MAX_SAMPLES: u64 = 1 << 23;
/// Largest sample: a packet is read into memory whole before it is decoded.
const MAX_SAMPLE_SIZE: u32 = 64 << 20;

/// One compressed frame of the video track, in decode order.
pub(super) struct Sample {
    pub offset: u64,
    pub size: u32,
    pub duration: u32,
    /// Display time in media timescale units, counted from the start of what
    /// the track shows: the edit list's window, or else the first frame. A
    /// shown frame that starts before the window is moved to 0; a hidden one
    /// keeps its place (negative before the window), so display order holds.
    pub cts: i64,
    /// Whether the decoder outputs a frame for the sample: set by the caller
    /// before [`Movie::finish`] (only a VP8 or VP9 packet may output nothing). A
    /// sample that outputs nothing takes no place in display order.
    pub displays: bool,
    /// Whether the frame displays inside the edit list's window. Hidden
    /// frames are still decoded: later frames may reference them.
    pub shown: bool,
    /// Whether the container marks the sample a keyframe (a sync sample).
    pub sync: bool,
}

/// The video track once [`Movie::finish`] has applied the edit list. Its
/// codec is [`Movie::codec`].
pub(super) struct Mp4Video {
    /// Coded size, from the sample entry.
    pub width: u32,
    pub height: u32,
    /// The payload of `avcC`, `hvcC` or `av1C`.
    pub config: Option<Vec<u8>>,
    /// Whether the sample entry lets parameter sets travel in the samples
    /// (`hev1`; `avc1` and `hvc1` keep them in the configuration record).
    pub inband_parameter_sets: bool,
    /// Media timescale: units per second.
    pub timescale: u64,
    /// Seconds before the first shown frame: an edit list's leading empty
    /// edit, else 0. A sample's time is `start_time + cts / timescale`.
    pub start_time: f64,
    /// Seconds the shown frames last.
    pub duration: f64,
    pub samples: Vec<Sample>,
}

fn malformed(what: &str) -> MmError {
    MmError::invalid_input(format!("malformed MP4: {what}"))
}

fn bytes<const N: usize>(data: &[u8], at: usize) -> Result<[u8; N]> {
    at.checked_add(N)
        .and_then(|end| data.get(at..end))
        .and_then(|b| b.try_into().ok())
        .ok_or_else(|| malformed("box shorter than its fields"))
}

fn be16(data: &[u8], at: usize) -> Result<u16> {
    bytes(data, at).map(u16::from_be_bytes)
}

fn be32(data: &[u8], at: usize) -> Result<u32> {
    bytes(data, at).map(u32::from_be_bytes)
}

fn be64(data: &[u8], at: usize) -> Result<u64> {
    bytes(data, at).map(u64::from_be_bytes)
}

fn index(value: u32) -> usize {
    usize::try_from(value).unwrap_or(usize::MAX)
}

/// Version, flags and the rest of a full box's body.
fn full_box<'a>(body: &'a [u8], allowed: &[u8], name: &str) -> Result<(u8, u32, &'a [u8])> {
    let head = bytes::<4>(body, 0)?;
    if !allowed.contains(&head[0]) {
        return Err(MmError::unsupported(format!(
            "MP4 {name} box version {}",
            head[0]
        )));
    }
    Ok((head[0], u32::from_be_bytes(head) & 0x00FF_FFFF, &body[4..]))
}

/// The entries of a table (`u32` count, then `entry` bytes each) in `rest`.
fn table<'a>(rest: &'a [u8], entry: usize, name: &str) -> Result<(usize, &'a [u8])> {
    let count = index(be32(rest, 0)?);
    count
        .checked_mul(entry)
        .and_then(|len| rest.get(4..4usize.checked_add(len)?))
        .map(|entries| (count, entries))
        .ok_or_else(|| malformed(&format!("{name} table larger than its box")))
}

fn need<'a>(slot: Option<&'a [u8]>, name: &str) -> Result<&'a [u8]> {
    slot.ok_or_else(|| malformed(&format!("sample table without {name}")))
}

fn set_once<T>(slot: &mut Option<T>, value: T, name: &str) -> Result<()> {
    if slot.replace(value).is_some() {
        return Err(malformed(&format!("more than one {name} box")));
    }
    Ok(())
}

/// The boxes inside `data`, as (type, body). Every box takes at least its
/// 8-byte header, so iteration ends; a bad size ends it with one error.
struct Boxes<'a> {
    data: &'a [u8],
    pos: usize,
    failed: bool,
}

impl<'a> Boxes<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            pos: 0,
            failed: false,
        }
    }
}

impl<'a> Iterator for Boxes<'a> {
    type Item = Result<([u8; 4], &'a [u8])>;

    fn next(&mut self) -> Option<Self::Item> {
        let head = &self.data[self.pos..];
        if self.failed || is_end(head) {
            return None;
        }
        let parsed = (|| {
            let (header, size) = match be32(head, 0)? {
                // Size 0 (to the end of the file) belongs at the top level.
                0 => return Err(malformed("box size")),
                1 => (16, be64(head, 8)?),
                n => (8, u64::from(n)),
            };
            if size < header || size > head.len() as u64 {
                return Err(malformed("box size"));
            }
            let kind = bytes::<4>(head, 4)?;
            // `size <= remaining`, so both fit in usize.
            Ok((kind, header as usize, size as usize))
        })();
        Some(match parsed {
            Ok((kind, header, size)) => {
                let body = &self.data[self.pos + header..self.pos + size];
                self.pos += size;
                Ok((kind, body))
            }
            Err(e) => {
                self.failed = true;
                Err(e)
            }
        })
    }
}

/// Whether `rest` holds no more boxes: it is empty, or QuickTime's 32-bit
/// zero that may end a list of boxes. Any other bytes are a box.
fn is_end(rest: &[u8]) -> bool {
    rest.is_empty() || rest == [0; 4]
}

/// Read the first video track of the MP4 in `source`. The caller marks which
/// samples display a frame, then calls [`Movie::finish`].
pub(super) fn read(source: &dyn ByteSource) -> Result<Movie> {
    let len = source.len();
    let mut movie: Option<Movie> = None;
    // The media data boxes' bodies, in file order.
    let mut media = Vec::new();
    let mut pos = 0u64;
    let mut boxes = 0u64;
    let header_read = |at: u64, buf: &mut [u8]| {
        source
            .read_at(at, buf)
            .map_err(|e| read_error(e, "an MP4 box header"))
    };
    while pos < len {
        let remaining = len - pos;
        // The 8-byte header, plus 8 more when the size is 64-bit (`1`): never
        // any of a box body, which may be gigabytes of media data.
        let mut head = [0u8; 16];
        if remaining < 8 {
            // `remaining < 8`, so it fits in usize.
            let tail = &mut head[..remaining as usize];
            header_read(pos, tail)?;
            if is_end(tail) {
                break;
            }
            return Err(malformed("bytes after the last box"));
        }
        boxes += 1;
        if boxes > MAX_BOXES {
            return Err(MmError::limit_exceeded(format!(
                "MP4 with more than {MAX_BOXES} top-level boxes"
            )));
        }
        header_read(pos, &mut head[..8])?;
        let (header, size) = match u32::from_be_bytes([head[0], head[1], head[2], head[3]]) {
            0 => (8, remaining),
            1 if remaining >= 16 => {
                header_read(pos + 8, &mut head[8..16])?;
                (16, u64::from_be_bytes(bytes(&head, 8)?))
            }
            1 => return Err(malformed("box size")),
            n => (8, u64::from(n)),
        };
        if size < header || size > remaining {
            return Err(malformed("box size"));
        }
        let kind = [head[4], head[5], head[6], head[7]];
        if &kind == b"mdat" {
            media.push((pos + header, pos + size));
        }
        if matches!(&kind, b"moov" | b"moof") {
            if size - header > MAX_BOX_BODY {
                return Err(malformed("movie box too large"));
            }
            let mut body = vec![0u8; usize::try_from(size - header).unwrap_or(usize::MAX)];
            source
                .read_at(pos + header, &mut body)
                .map_err(|e| read_error(e, "an MP4 box"))?;
            if &kind == b"moov" {
                if movie.is_some() {
                    return Err(malformed("more than one moov box"));
                }
                movie = Some(Movie::new(&body, len)?);
            } else {
                movie
                    .as_mut()
                    .ok_or_else(|| malformed("movie fragment before the moov box"))?
                    .add_fragment(pos, &body)?;
            }
        }
        pos += size;
    }
    let movie = movie.ok_or_else(|| malformed("no moov box"))?;
    // Each sample lies within one media data box: not in the movie's boxes,
    // and not in another file.
    for sample in &movie.samples {
        let after = media.partition_point(|&(start, _)| start <= sample.offset);
        let inside = after
            .checked_sub(1)
            .and_then(|i| media.get(i))
            .zip(sample.offset.checked_add(u64::from(sample.size)))
            .is_some_and(|(&(_, end), sample_end)| sample_end <= end);
        if !inside {
            return Err(malformed("sample lies outside the media data"));
        }
    }
    Ok(movie)
}

/// What the `trak` boxes say about a track, before its tables are read.
struct TrakHead<'a> {
    id: u32,
    timescale: u32,
    duration: u64,
    handler: [u8; 4],
    stbl: Option<&'a [u8]>,
    elst: Option<&'a [u8]>,
    dref: Option<&'a [u8]>,
}

fn trak_head(trak: &[u8]) -> Result<TrakHead<'_>> {
    let (mut id, mut mdhd, mut handler, mut stbl) = (None, None, None, None);
    let (mut elst, mut dref) = (None, None);
    for child in Boxes::new(trak) {
        let (kind, body) = child?;
        match &kind {
            b"tkhd" => {
                let (version, _, rest) = full_box(body, &[0, 1], "tkhd")?;
                let at = if version == 1 { 16 } else { 8 };
                set_once(&mut id, be32(rest, at)?, "tkhd")?;
            }
            b"edts" => {
                for child in Boxes::new(body) {
                    let (kind, body) = child?;
                    if &kind == b"elst" {
                        set_once(&mut elst, body, "elst")?;
                    }
                }
            }
            b"mdia" => {
                for child in Boxes::new(body) {
                    let (kind, body) = child?;
                    match &kind {
                        b"mdhd" => {
                            let (version, _, rest) = full_box(body, &[0, 1], "mdhd")?;
                            // All ones, like 0, means the duration is unknown.
                            let (timescale, duration) = if version == 1 {
                                (
                                    be32(rest, 16)?,
                                    Some(be64(rest, 20)?).filter(|&d| d != u64::MAX),
                                )
                            } else {
                                let d = be32(rest, 12)?;
                                (be32(rest, 8)?, (d != u32::MAX).then_some(u64::from(d)))
                            };
                            set_once(&mut mdhd, (timescale, duration.unwrap_or(0)), "mdhd")?;
                        }
                        b"hdlr" => {
                            let (_, _, rest) = full_box(body, &[0], "hdlr")?;
                            set_once(&mut handler, bytes::<4>(rest, 4)?, "hdlr")?;
                        }
                        b"minf" => {
                            for child in Boxes::new(body) {
                                let (kind, body) = child?;
                                match &kind {
                                    b"stbl" => set_once(&mut stbl, body, "stbl")?,
                                    b"dinf" => {
                                        for child in Boxes::new(body) {
                                            let (kind, body) = child?;
                                            if &kind == b"dref" {
                                                set_once(&mut dref, body, "dref")?;
                                            }
                                        }
                                    }
                                    _ => {}
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }
    let (timescale, duration) = mdhd.ok_or_else(|| malformed("track without mdhd"))?;
    Ok(TrakHead {
        id: id.ok_or_else(|| malformed("track without tkhd"))?,
        timescale,
        duration,
        handler: handler.unwrap_or_default(),
        stbl,
        elst,
        dref,
    })
}

/// Refuse media data that is not in this file. The sample entry names entry
/// `entry` of `dref`, which must be the first and flag itself as this file. A
/// track without `dref` has nowhere else to point, so its data is here.
fn check_data_reference(dref: Option<&[u8]>, entry: u16) -> Result<()> {
    if entry != 1 {
        return Err(MmError::unsupported(
            "MP4 data reference other than the first",
        ));
    }
    let Some(dref) = dref else {
        return Ok(());
    };
    let (_, _, rest) = full_box(dref, &[0], "dref")?;
    // The count is read first: a `dref` cut after its version and flags has
    // no count to slice past.
    let count = index(be32(rest, 0)?);
    let entries = Boxes::new(&rest[4..]).collect::<Result<Vec<_>>>()?;
    if entries.is_empty() || entries.len() != count {
        return Err(malformed("dref entries do not match their count"));
    }
    // `url ` or `urn ` (or QuickTime's `alis`) with flag 1: the media data
    // is in this file.
    let (kind, entry) = entries[0];
    let (_, flags, _) = full_box(entry, &[0], "data reference")?;
    if !matches!(&kind, b"url " | b"urn " | b"alis") || flags & 1 == 0 {
        return Err(MmError::unsupported("MP4 media data in another file"));
    }
    Ok(())
}

/// The video track and its fragment defaults, growing as `moof` boxes arrive.
pub(super) struct Movie {
    pub codec: VideoCodec,
    width: u32,
    height: u32,
    config: Option<Vec<u8>>,
    inband_parameter_sets: bool,
    track_id: u32,
    timescale: u32,
    duration: u64,
    default_duration: u32,
    default_size: u32,
    default_flags: u32,
    /// The video track's edit list, the movie timescale its durations use,
    /// and whether the file is fragmented.
    edits: Vec<Edit>,
    movie_timescale: u32,
    fragmented: bool,
    pub samples: Vec<Sample>,
    /// The sample with the latest composition time.
    last_displayed: Option<usize>,
    /// Decode time after the last sample, where a fragment without `tfdt`
    /// continues.
    next_dts: i64,
    file_len: u64,
}

impl Movie {
    fn new(moov: &[u8], file_len: u64) -> Result<Self> {
        let (mut traks, mut ids) = (Vec::new(), HashSet::new());
        let mut trexs = Vec::new();
        let (mut movie_timescale, mut fragmented) = (None, false);
        for child in Boxes::new(moov) {
            let (kind, body) = child?;
            match &kind {
                b"mvhd" => {
                    let (version, _, rest) = full_box(body, &[0, 1], "mvhd")?;
                    let at = if version == 1 { 16 } else { 8 };
                    set_once(&mut movie_timescale, be32(rest, at)?, "mvhd")?;
                }
                b"trak" => {
                    let head = trak_head(body)?;
                    if !ids.insert(head.id) {
                        return Err(malformed("two tracks with one id"));
                    }
                    traks.push(head);
                }
                b"mvex" => {
                    fragmented = true;
                    for child in Boxes::new(body) {
                        let (kind, body) = child?;
                        if &kind == b"trex" {
                            trexs.push(body);
                        }
                    }
                }
                _ => {}
            }
        }
        let trak = traks
            .iter()
            .filter(|t| &t.handler == b"vide")
            .min_by_key(|t| t.id)
            .ok_or_else(|| malformed("MP4 has no video track"))?;
        if trak.timescale == 0 {
            return Err(malformed("zero timescale"));
        }
        let stbl = trak
            .stbl
            .ok_or_else(|| malformed("video track without a sample table"))?;
        let mut movie = Self {
            codec: VideoCodec::H264,
            width: 0,
            height: 0,
            config: None,
            inband_parameter_sets: false,
            track_id: trak.id,
            timescale: trak.timescale,
            duration: trak.duration,
            default_duration: 0,
            default_size: 0,
            default_flags: 0,
            edits: trak.elst.map(edits).transpose()?.unwrap_or_default(),
            movie_timescale: movie_timescale.unwrap_or(0),
            fragmented,
            samples: Vec::new(),
            last_displayed: None,
            next_dts: 0,
            file_len,
        };
        let mut trex = None;
        for body in trexs {
            let (_, _, rest) = full_box(body, &[0], "trex")?;
            if be32(rest, 0)? == trak.id {
                set_once(&mut trex, (), "trex for the video track")?;
                if be32(rest, 4)? != 1 {
                    return Err(MmError::unsupported("MP4 with several sample descriptions"));
                }
                movie.default_duration = be32(rest, 8)?;
                movie.default_size = be32(rest, 12)?;
                movie.default_flags = be32(rest, 16)?;
            }
        }
        movie.read_stbl(stbl, trak.dref)?;
        Ok(movie)
    }

    /// The sample entry, then the samples the sample table lists.
    fn read_stbl(&mut self, stbl: &[u8], dref: Option<&[u8]>) -> Result<()> {
        let (mut stsd, mut stts, mut ctts, mut stsz) = (None, None, None, None);
        let (mut stsc, mut stco, mut co64) = (None, None, None);
        let mut stss = Vec::new();
        for child in Boxes::new(stbl) {
            let (kind, body) = child?;
            match &kind {
                b"stsd" => set_once(&mut stsd, body, "stsd")?,
                b"stts" => set_once(&mut stts, body, "stts")?,
                b"ctts" => set_once(&mut ctts, body, "ctts")?,
                b"stsz" => set_once(&mut stsz, body, "stsz")?,
                b"stsc" => set_once(&mut stsc, body, "stsc")?,
                b"stco" => set_once(&mut stco, body, "stco")?,
                b"co64" => set_once(&mut co64, body, "co64")?,
                b"stss" => stss.push(body),
                b"stz2" => {
                    return Err(MmError::unsupported("MP4 compact sample sizes (stz2)"));
                }
                _ => {}
            }
        }
        self.read_sample_entry(need(stsd, "stsd")?, dref)?;
        let (stts, stsz, stsc) = (
            need(stts, "stts")?,
            need(stsz, "stsz")?,
            need(stsc, "stsc")?,
        );
        let chunks = match (stco, co64) {
            (Some(_), Some(_)) => return Err(malformed("both stco and co64")),
            (Some(body), None) => Chunks::new(body, 4, "stco")?,
            (None, Some(body)) => Chunks::new(body, 8, "co64")?,
            (None, None) => return Err(malformed("sample table without stco or co64")),
        };
        self.read_samples(stts, ctts, stsz, stsc, &chunks)?;
        if !stss.is_empty() {
            self.mark_sync(&stss);
        }
        Ok(())
    }

    /// Keep only the keyframes `stss` lists (FFmpeg's reading: an empty one
    /// leaves none, so decoding starts at the first sample). Entries that do
    /// not parse or name no sample, and every entry of a table given twice,
    /// are dropped, which can only stop the decoder from skipping ahead.
    fn mark_sync(&mut self, stss: &[&[u8]]) {
        for sample in &mut self.samples {
            sample.sync = false;
        }
        let entries = match stss {
            [stss] => full_box(stss, &[0], "stss")
                .and_then(|(_, _, rest)| table(rest, 4, "stss"))
                .map_or(&[][..], |(_, entries)| entries),
            _ => &[],
        };
        for entry in entries.chunks_exact(4) {
            let number = u32::from_be_bytes([entry[0], entry[1], entry[2], entry[3]]);
            if let Some(sample) = index(number)
                .checked_sub(1)
                .and_then(|i| self.samples.get_mut(i))
            {
                sample.sync = true;
            }
        }
    }

    fn read_sample_entry(&mut self, stsd: &[u8], dref: Option<&[u8]>) -> Result<()> {
        let (_, _, rest) = full_box(stsd, &[0], "stsd")?;
        match be32(rest, 0)? {
            1 => {}
            0 => return Err(malformed("stsd without an entry")),
            _ => return Err(MmError::unsupported("MP4 with several sample descriptions")),
        }
        let mut entries = Boxes::new(&rest[4..]);
        let (kind, body) = entries
            .next()
            .ok_or_else(|| malformed("stsd without an entry"))??;
        if let Some(extra) = entries.next() {
            extra?;
            return Err(malformed("stsd holds more entries than it counts"));
        }
        let (codec, config_kind): (_, Option<&[u8; 4]>) = match &kind {
            b"avc1" => (VideoCodec::H264, Some(b"avcC")),
            b"hvc1" | b"hev1" => (VideoCodec::Hevc, Some(b"hvcC")),
            b"av01" => (VideoCodec::Av1, Some(b"av1C")),
            b"vp09" => (VideoCodec::Vp9, None),
            b"vp08" => (VideoCodec::Vp8, None),
            other => {
                return Err(MmError::unsupported(format!(
                    "unsupported MP4 video codec {}",
                    other.escape_ascii()
                )));
            }
        };
        // VisualSampleEntry: 78 bytes, then the codec's own boxes.
        let children = body
            .get(78..)
            .ok_or_else(|| malformed("short sample entry"))?;
        check_data_reference(dref, be16(body, 6)?)?;
        self.codec = codec;
        self.inband_parameter_sets = &kind == b"hev1";
        self.width = u32::from(be16(body, 24)?);
        self.height = u32::from(be16(body, 26)?);
        let mut config = None;
        for child in Boxes::new(children) {
            let (kind, body) = child?;
            if Some(&kind) == config_kind {
                set_once(&mut config, body.to_vec(), "codec configuration")?;
            }
        }
        if config_kind.is_some() && config.is_none() {
            return Err(malformed("sample entry without a codec configuration"));
        }
        self.config = config;
        Ok(())
    }

    fn read_samples(
        &mut self,
        stts: &[u8],
        ctts: Option<&[u8]>,
        stsz: &[u8],
        stsc: &[u8],
        chunks: &Chunks<'_>,
    ) -> Result<()> {
        let (_, _, stsz) = full_box(stsz, &[0], "stsz")?;
        let (constant, count) = (be32(stsz, 0)?, be32(stsz, 4)?);
        let sizes = if constant == 0 {
            table(&stsz[4..], 4, "stsz")?.1
        } else {
            &[]
        };
        let count = u64::from(count);
        if count > MAX_SAMPLES.min(self.file_len) {
            return Err(malformed("more samples than the file can hold"));
        }
        let count = usize::try_from(count).unwrap_or(usize::MAX);

        let (_, _, stts) = full_box(stts, &[0], "stts")?;
        let mut deltas = Runs::new(table(stts, 8, "stts")?.1);
        // Offsets are read as signed in both versions, as FFmpeg reads them:
        // muxers write negative offsets into version 0, and an unsigned one
        // of 2^31 or more would be a delay of over six hours at 90 kHz.
        let mut offsets = match ctts {
            Some(body) => {
                let (_, _, rest) = full_box(body, &[0, 1], "ctts")?;
                Some(Runs::new(table(rest, 8, "ctts")?.1))
            }
            None => None,
        };

        let (_, _, stsc) = full_box(stsc, &[0], "stsc")?;
        let (run_count, runs) = table(stsc, 12, "stsc")?;
        let run = |i: usize| -> Result<(u64, usize)> {
            let at = i * 12;
            if be32(runs, at + 8)? != 1 {
                return Err(malformed("sample description index other than 1"));
            }
            let per_chunk = index(be32(runs, at + 4)?);
            if per_chunk == 0 {
                return Err(malformed("chunk without samples"));
            }
            Ok((u64::from(be32(runs, at)?), per_chunk))
        };

        if run_count == 0 && chunks.count > 0 {
            return Err(malformed("chunks without samples"));
        }
        self.samples.reserve(count);
        let (mut dts, mut last_dts) = (0i64, 0i64);
        for i in 0..run_count {
            let (first, per_chunk) = run(i)?;
            // The first run starts at chunk 1; each later one after the last.
            if (i == 0 && first != 1) || first == 0 {
                return Err(malformed("chunk runs do not start at chunk 1"));
            }
            if first > chunks.count as u64 {
                return Err(malformed("chunk index out of range"));
            }
            let end = if i + 1 < run_count {
                let next = run(i + 1)?.0;
                if next <= first {
                    return Err(malformed("chunk runs out of order"));
                }
                next - 1
            } else {
                chunks.count as u64
            };
            for chunk in first..=end {
                let mut offset = chunks.offset(chunk)?;
                for _ in 0..per_chunk {
                    if self.samples.len() == count {
                        return Err(malformed("chunk table covers more samples than stsz"));
                    }
                    let n = self.samples.len();
                    let size = if constant == 0 {
                        be32(sizes, n * 4)?
                    } else {
                        constant
                    };
                    let duration = deltas
                        .take()
                        .ok_or_else(|| malformed("stts covers fewer samples than stsz"))?;
                    let shift = match offsets.as_mut() {
                        Some(runs) => i64::from(
                            runs.take()
                                .ok_or_else(|| malformed("ctts covers fewer samples than stsz"))?
                                as i32,
                        ),
                        None => 0,
                    };
                    self.push(offset, size, duration, dts, shift, true)?;
                    last_dts = dts;
                    dts = dts
                        .checked_add(i64::from(duration))
                        .ok_or_else(|| malformed("decode time overflow"))?;
                    offset = offset
                        .checked_add(u64::from(size))
                        .ok_or_else(|| malformed("sample offset overflow"))?;
                }
            }
        }
        if self.samples.len() < count {
            return Err(malformed("chunk table covers fewer samples than stsz"));
        }
        if deltas.take().is_some() || offsets.as_mut().is_some_and(|o| o.take().is_some()) {
            return Err(malformed("stts or ctts covers more samples than stsz"));
        }
        // The media cannot end before its last sample starts.
        if self.duration != 0 && u64::try_from(last_dts).unwrap_or(u64::MAX) > self.duration {
            return Err(malformed("track duration ends before its last sample"));
        }
        self.next_dts = dts;
        Ok(())
    }

    /// Append a sample, refusing one with no bytes or too many. [`read`]
    /// checks that it lies within the media data, once every box is known.
    fn push(
        &mut self,
        offset: u64,
        size: u32,
        duration: u32,
        dts: i64,
        shift: i64,
        sync: bool,
    ) -> Result<()> {
        if size == 0 {
            return Err(malformed("empty sample"));
        }
        if size > MAX_SAMPLE_SIZE {
            return Err(MmError::limit_exceeded(format!(
                "MP4 sample of {size} bytes exceeds the {MAX_SAMPLE_SIZE} byte limit"
            )));
        }
        let cts = dts
            .checked_add(shift)
            .ok_or_else(|| malformed("composition time overflow"))?;
        self.samples.push(Sample {
            offset,
            size,
            duration,
            cts,
            displays: true,
            shown: true,
            sync,
        });
        let latest = self.last_displayed.and_then(|i| self.samples.get(i));
        if latest.is_none_or(|latest| cts >= latest.cts) {
            self.last_displayed = Some(self.samples.len() - 1);
        }
        Ok(())
    }

    /// Append the samples of the video track's `traf` boxes in a `moof` that
    /// starts at `moof_start`.
    fn add_fragment(&mut self, moof_start: u64, moof: &[u8]) -> Result<()> {
        let mut traf_index = 0;
        for child in Boxes::new(moof) {
            let (kind, traf) = child?;
            if &kind != b"traf" {
                continue;
            }
            self.add_traf(moof_start, traf_index, traf)?;
            traf_index += 1;
        }
        Ok(())
    }

    fn add_traf(&mut self, moof_start: u64, traf_index: usize, traf: &[u8]) -> Result<()> {
        // Not 0x1_0000, duration-is-empty: a fragment of no samples that only
        // moves time on.
        const KNOWN_TFHD: u32 = 0x1 | 0x2 | 0x8 | 0x10 | 0x20 | 0x2_0000;
        const BASE_OFFSET: u32 = 0x1;
        const DESCRIPTION: u32 = 0x2;
        const DURATION: u32 = 0x8;
        const SIZE: u32 = 0x10;
        const FLAGS: u32 = 0x20;
        const BASE_IS_MOOF: u32 = 0x2_0000;

        let (mut tfhd, mut tfdt, mut truns) = (None, None, Vec::new());
        for child in Boxes::new(traf) {
            let (kind, body) = child?;
            match &kind {
                b"tfhd" => set_once(&mut tfhd, body, "tfhd")?,
                b"tfdt" => set_once(&mut tfdt, body, "tfdt")?,
                b"trun" => truns.push(body),
                _ => {}
            }
        }
        let tfhd = tfhd.ok_or_else(|| malformed("traf without tfhd"))?;
        let (_, flags, rest) = full_box(tfhd, &[0], "tfhd")?;
        // Other tracks' fragments are skipped, whatever they hold.
        if be32(rest, 0)? != self.track_id {
            return Ok(());
        }
        if flags & !KNOWN_TFHD != 0 {
            return Err(MmError::unsupported(format!("MP4 tfhd flags {flags:#x}")));
        }
        // The optional fields, in order.
        let mut at = 4;
        let mut field = |flag: u32, wide: bool| -> Result<Option<u64>> {
            if flags & flag == 0 {
                return Ok(None);
            }
            let value = if wide {
                be64(rest, at)?
            } else {
                u64::from(be32(rest, at)?)
            };
            at += if wide { 8 } else { 4 };
            Ok(Some(value))
        };
        let explicit_base = field(BASE_OFFSET, true)?;
        if field(DESCRIPTION, false)?.is_some_and(|index| index != 1) {
            return Err(MmError::unsupported("MP4 with several sample descriptions"));
        }
        let default_duration = field(DURATION, false)?.map_or(self.default_duration, |v| v as u32);
        let default_size = field(SIZE, false)?.map_or(self.default_size, |v| v as u32);
        let default_flags = field(FLAGS, false)?.map_or(self.default_flags, |v| v as u32);

        let base = match explicit_base {
            Some(base) => base,
            None if flags & BASE_IS_MOOF != 0 || traf_index == 0 => moof_start,
            // After the previous traf's data, which belongs to another track.
            None => {
                return Err(MmError::unsupported(
                    "MP4 fragment without a base data offset",
                ));
            }
        };

        let mut dts = match tfdt {
            Some(body) => {
                let (version, _, rest) = full_box(body, &[0, 1], "tfdt")?;
                let time = if version == 1 {
                    be64(rest, 0)?
                } else {
                    u64::from(be32(rest, 0)?)
                };
                let time = i64::try_from(time).map_err(|_| malformed("tfdt out of range"))?;
                self.continue_at(time)?;
                time
            }
            None => self.next_dts,
        };
        let mut cursor = base;
        for body in truns {
            cursor = self.add_run(
                body,
                base,
                cursor,
                &mut dts,
                (default_duration, default_size, default_flags),
            )?;
        }
        self.next_dts = dts;
        Ok(())
    }

    /// Move decode time to a fragment's `tfdt`. It may not go back over the
    /// samples so far; a gap after them lengthens the last one in display
    /// order.
    fn continue_at(&mut self, time: i64) -> Result<()> {
        let gap = time
            .checked_sub(self.next_dts)
            .filter(|gap| *gap >= 0)
            .ok_or_else(|| malformed("fragment decode time goes back"))?;
        let last = self.last_displayed.and_then(|i| self.samples.get_mut(i));
        if let Some(last) = last.filter(|_| gap > 0) {
            last.duration = u32::try_from(gap)
                .ok()
                .and_then(|gap| last.duration.checked_add(gap))
                .ok_or_else(|| malformed("gap between fragments too long"))?;
        }
        self.next_dts = time;
        Ok(())
    }

    /// Append the samples of one `trun`; returns where the next run's data
    /// starts, after this run's.
    fn add_run(
        &mut self,
        trun: &[u8],
        base: u64,
        cursor: u64,
        dts: &mut i64,
        (default_duration, default_size, default_flags): (u32, u32, u32),
    ) -> Result<u64> {
        const KNOWN_TRUN: u32 = 0x1 | 0x4 | 0x100 | 0x200 | 0x400 | 0x800;
        const DATA_OFFSET: u32 = 0x1;
        const FIRST_FLAGS: u32 = 0x4;
        const DURATION: u32 = 0x100;
        const SIZE: u32 = 0x200;
        const FLAGS: u32 = 0x400;
        const CTS: u32 = 0x800;
        // Sample flags that rule out starting a decode at the sample, as in
        // FFmpeg: `sample_is_non_sync_sample`, and `sample_depends_on` = 1.
        const NOT_SYNC: u32 = 0x1_0000 | 0x100_0000;

        let (_, flags, rest) = full_box(trun, &[0, 1], "trun")?;
        if flags & !KNOWN_TRUN != 0 {
            return Err(MmError::unsupported(format!("MP4 trun flags {flags:#x}")));
        }
        let count = u64::from(be32(rest, 0)?);
        let mut at = 4;
        let mut cursor = cursor;
        if flags & DATA_OFFSET != 0 {
            let delta = i64::from(i32::from_be_bytes(bytes(rest, at)?));
            cursor = base
                .checked_add_signed(delta)
                .ok_or_else(|| malformed("trun data offset out of range"))?;
            at += 4;
        }
        let mut first_flags = None;
        if flags & FIRST_FLAGS != 0 {
            first_flags = Some(be32(rest, at)?);
            at += 4;
        }
        let record = 4 * (flags & (DURATION | SIZE | FLAGS | CTS)).count_ones() as usize;
        let body = rest.get(at..).unwrap_or_default();
        if record != 0 && count > (body.len() / record) as u64 {
            return Err(malformed("trun lists more samples than its box holds"));
        }
        let total = self.samples.len() as u64 + count;
        if total > MAX_SAMPLES.min(self.file_len) {
            return Err(malformed("more samples than the file can hold"));
        }
        let mut next = at;
        for i in 0..count {
            let mut take = |flag: u32| -> Result<Option<u32>> {
                if flags & flag == 0 {
                    return Ok(None);
                }
                let value = be32(rest, next)?;
                next += 4;
                Ok(Some(value))
            };
            let duration = take(DURATION)?.unwrap_or(default_duration);
            let size = take(SIZE)?.unwrap_or(default_size);
            let sample_flags = take(FLAGS)?;
            // As FFmpeg reads them: a sample's own flags, else the run's
            // first-sample flags for its first, else the defaults.
            let sample_flags = sample_flags
                .or(first_flags.filter(|_| i == 0))
                .unwrap_or(default_flags);
            // Signed in both versions, as for `ctts`.
            let shift = take(CTS)?.map_or(0, |v| i64::from(v as i32));
            self.push(
                cursor,
                size,
                duration,
                *dts,
                shift,
                sample_flags & NOT_SYNC == 0,
            )?;
            cursor = cursor
                .checked_add(u64::from(size))
                .ok_or_else(|| malformed("sample offset overflow"))?;
            *dts = dts
                .checked_add(i64::from(duration))
                .ok_or_else(|| malformed("decode time overflow"))?;
        }
        Ok(cursor)
    }

    /// Apply the edit list (see [`window`]), or without one start time at the
    /// first frame that displays, as players do; then settle the duration.
    pub(super) fn finish(mut self) -> Result<Mp4Video> {
        let timescale = f64::from(self.timescale);
        let overflow = || malformed("composition time overflow");
        let Some(window) = window(
            &self.edits,
            self.movie_timescale,
            self.timescale,
            self.fragmented,
        )?
        else {
            let shown = || self.samples.iter().filter(|s| s.displays);
            let first = shown().map(|s| s.cts).min().unwrap_or(0);
            // How long the shown frames last, from the first: as with an edit
            // list, the frames decide, not `mdhd`.
            let mut end = 0i128;
            for s in shown() {
                end = end.max(i128::from(s.cts) - i128::from(first) + i128::from(s.duration));
            }
            for sample in &mut self.samples {
                sample.cts = sample.cts.checked_sub(first).ok_or_else(overflow)?;
                sample.shown = sample.displays;
            }
            return Ok(self.into_video(0.0, end as f64 / timescale));
        };

        // Each frame shows from its time until the next frame in display order
        // (the last for its sample duration). Compare in media units times the
        // movie timescale, where both timescales' values are exact integers.
        let (movie, media) = (i128::from(self.movie_timescale), i128::from(self.timescale));
        let from = i128::from(window.start) * movie;
        let until = window
            .length
            .map_or(i128::MAX, |length| from + i128::from(length) * media);
        // A sample that outputs no frame is hidden and takes no place.
        for sample in &mut self.samples {
            if !sample.displays {
                sample.shown = false;
                sample.cts = sample.cts.checked_sub(window.start).ok_or_else(overflow)?;
            }
        }
        let mut order: Vec<usize> = (0..self.samples.len())
            .filter(|&i| self.samples[i].displays)
            .collect();
        order.sort_by_key(|&i| (self.samples[i].cts, i));
        let mut end_shown = from;
        for (k, &i) in order.iter().enumerate() {
            let sample = &self.samples[i];
            let next = order.get(k + 1).map_or_else(
                || i128::from(sample.cts) + i128::from(sample.duration),
                |&j| i128::from(self.samples[j].cts),
            );
            let (start, end) = (i128::from(sample.cts) * movie, next * movie);
            // Durations in the movie timescale are rounded, so a sliver of a
            // frame (one movie tick or less) at an edge does not show it,
            // unless it is the whole frame or the whole window.
            let overlap = end.min(until) - start.max(from);
            let whole = overlap == end - start || overlap == until - from;
            let shown = overlap > 0 && (overlap > media || whole);
            if shown {
                end_shown = end_shown.max(end.min(until));
            }
            let sample = &mut self.samples[i];
            sample.shown = shown;
            let cts = if shown {
                sample.cts.max(window.start)
            } else {
                sample.cts
            };
            sample.cts = cts.checked_sub(window.start).ok_or_else(overflow)?;
        }
        // Time starts at the first shown frame: at the window's start, unless
        // the window starts before the media does.
        let first = self.samples.iter().filter(|s| s.shown).map(|s| s.cts).min();
        let first = first.unwrap_or(0);
        for sample in &mut self.samples {
            sample.cts = sample.cts.checked_sub(first).ok_or_else(overflow)?;
        }
        let start_time = window.delay as f64 / movie as f64 + first as f64 / media as f64;
        let duration =
            (end_shown - from - i128::from(first) * movie) as f64 / (movie * media) as f64;
        Ok(self.into_video(start_time, duration))
    }

    fn into_video(self, start_time: f64, duration: f64) -> Mp4Video {
        Mp4Video {
            width: self.width,
            height: self.height,
            config: self.config,
            inband_parameter_sets: self.inband_parameter_sets,
            timescale: u64::from(self.timescale),
            start_time,
            duration,
            samples: self.samples,
        }
    }
}

/// One row of an edit list: show `duration` (movie timescale) of the media
/// from `media_time` (media timescale; -1 for an empty edit, which shows
/// nothing) at `rate` (16.16 fixed point).
struct Edit {
    duration: u64,
    media_time: i64,
    rate: u32,
}

fn edits(elst: &[u8]) -> Result<Vec<Edit>> {
    let (version, _, rest) = full_box(elst, &[0, 1], "elst")?;
    let width = if version == 1 { 20 } else { 12 };
    let (count, entries) = table(rest, width, "elst")?;
    (0..count)
        .map(|i| {
            let at = i * width;
            Ok(if version == 1 {
                Edit {
                    duration: be64(entries, at)?,
                    media_time: be64(entries, at + 8)? as i64,
                    rate: be32(entries, at + 16)?,
                }
            } else {
                Edit {
                    duration: u64::from(be32(entries, at)?),
                    media_time: i64::from(be32(entries, at + 4)? as i32),
                    rate: be32(entries, at + 8)?,
                }
            })
        })
        .collect()
}

/// What an edit list shows: after `delay` (movie timescale) of nothing, the
/// media from `start` (media timescale) for `length` (movie timescale; `None`
/// to the end).
struct Window {
    delay: u64,
    start: i64,
    length: Option<u64>,
}

/// The window of an edit list, or `None` for no edit list. Accepted: one row
/// of media, optionally after one empty row (a delay); rows of media that
/// continue one another are joined into one. Refused: rows that go back,
/// repeat or skip media, an empty row anywhere but first, and a rate other
/// than 1. A zero duration means "to the end" only on the last row of a
/// fragmented file, whose length the movie header cannot know.
fn window(edits: &[Edit], movie: u32, media: u32, fragmented: bool) -> Result<Option<Window>> {
    const RATE_ONE: u32 = 0x1_0000;
    let refuse = |what: &str| MmError::unsupported(format!("MP4 edit list {what}"));
    let Some(head) = edits.first() else {
        return Ok(None);
    };
    if movie == 0 {
        return Err(malformed("edit list without a movie timescale"));
    }
    let (delay, rows) = if head.media_time == -1 {
        if head.rate != RATE_ONE {
            return Err(refuse("with a rate other than 1"));
        }
        if head.duration == 0 {
            return Err(refuse("with an edit of no duration"));
        }
        (head.duration, &edits[1..])
    } else {
        (0, edits)
    };
    let Some(first) = rows.first() else {
        return Err(refuse("that shows no media"));
    };
    let (movie, media_scale) = (i128::from(movie), i128::from(media));
    let mut length = Some(0u64);
    let (mut end, mut last_time) = (0i128, None);
    for (i, row) in rows.iter().enumerate() {
        match row.media_time {
            -1 => return Err(refuse("with an empty edit after the start")),
            time if time < 0 => return Err(malformed("edit media time")),
            _ => {}
        }
        if row.rate != RATE_ONE {
            return Err(refuse("with a rate other than 1"));
        }
        // Where this row starts, in media units times the movie timescale. It
        // must start after the last row did, where that one ended to within
        // one movie tick (`media` here), as edit durations are rounded.
        let at = i128::from(row.media_time) * movie;
        if let Some(last) = last_time
            && (row.media_time <= last || (at - end).abs() > media_scale)
        {
            return Err(refuse("that goes back, repeats or skips media"));
        }
        last_time = Some(row.media_time);
        if row.duration == 0 {
            if i + 1 == rows.len() && fragmented {
                length = None;
                break;
            }
            return Err(refuse("with an edit of no duration"));
        }
        length = Some(
            length
                .and_then(|l| l.checked_add(row.duration))
                .ok_or_else(|| malformed("edit list duration overflow"))?,
        );
        end = at + i128::from(row.duration) * media_scale;
    }
    Ok(Some(Window {
        delay,
        start: first.media_time,
        length,
    }))
}

/// Chunk offsets: `stco` (32-bit entries) or `co64` (64-bit).
struct Chunks<'a> {
    count: usize,
    entries: &'a [u8],
    wide: bool,
}

impl<'a> Chunks<'a> {
    fn new(body: &'a [u8], width: usize, name: &str) -> Result<Self> {
        let (_, _, rest) = full_box(body, &[0], name)?;
        let (count, entries) = table(rest, width, name)?;
        Ok(Self {
            count,
            entries,
            wide: width == 8,
        })
    }

    /// The offset of chunk `number`, counted from 1.
    fn offset(&self, number: u64) -> Result<u64> {
        let at = usize::try_from(number.saturating_sub(1))
            .ok()
            .filter(|at| number > 0 && *at < self.count)
            .ok_or_else(|| malformed("chunk index out of range"))?;
        if self.wide {
            be64(self.entries, at * 8)
        } else {
            be32(self.entries, at * 4).map(u64::from)
        }
    }
}

/// Run-length encoded per-sample values: (count, value) pairs.
struct Runs<'a> {
    entries: &'a [u8],
    next: usize,
    left: u32,
    value: u32,
}

impl<'a> Runs<'a> {
    fn new(entries: &'a [u8]) -> Self {
        Self {
            entries,
            next: 0,
            left: 0,
            value: 0,
        }
    }

    /// The value of the next sample, or `None` once every run is used up.
    fn take(&mut self) -> Option<u32> {
        while self.left == 0 {
            let entry = self.entries.get(self.next * 8..self.next * 8 + 8)?;
            self.left = u32::from_be_bytes([entry[0], entry[1], entry[2], entry[3]]);
            self.value = u32::from_be_bytes([entry[4], entry[5], entry[6], entry[7]]);
            self.next += 1;
        }
        self.left -= 1;
        Some(self.value)
    }
}

#[cfg(test)]
mod tests;
