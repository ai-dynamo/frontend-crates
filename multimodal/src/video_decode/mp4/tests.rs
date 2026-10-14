// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

fn parse(bytes: &[u8]) -> Result<Mp4Video> {
    read(&bytes)?.finish()
}

fn bx(kind: &[u8; 4], body: &[u8]) -> Vec<u8> {
    let mut b = (body.len() as u32 + 8).to_be_bytes().to_vec();
    b.extend_from_slice(kind);
    b.extend_from_slice(body);
    b
}

fn full(kind: &[u8; 4], version: u8, flags: u32, body: &[u8]) -> Vec<u8> {
    let mut b = vec![version];
    b.extend_from_slice(&flags.to_be_bytes()[1..]);
    b.extend_from_slice(body);
    bx(kind, &b)
}

fn words(values: &[u32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_be_bytes()).collect()
}

fn cat(parts: &[&[u8]]) -> Vec<u8> {
    parts.concat()
}

/// A B-frame clip of five frames at 30000 units per second. The first shown
/// frame is the first sample, which displays at 1000: composition time starts
/// there and shifts to 0.
const SIZES: [u32; 5] = [10, 20, 30, 40, 50];
const DURATION: u32 = 1000;
const SHIFTS: [i32; 5] = [1000, 2000, 0, 2000, 0];
/// Display times after the shift to the first shown frame.
const CTS: [i64; 5] = [0, 2000, 1000, 4000, 3000];
const FTYP: [u8; 16] = *b"\0\0\0\x10ftypisom\0\0\0\0";

fn entry(kind: &[u8; 4], config: Option<&[u8; 4]>) -> Vec<u8> {
    let mut body = vec![0u8; 78];
    // The data reference: the first, this file.
    body[6..8].copy_from_slice(&1u16.to_be_bytes());
    body[24..26].copy_from_slice(&64u16.to_be_bytes());
    body[26..28].copy_from_slice(&48u16.to_be_bytes());
    if let Some(config) = config {
        body.extend_from_slice(&bx(config, &[1, 2, 3, 4, 5]));
    }
    full(b"stsd", 0, 0, &cat(&[&words(&[1]), &bx(kind, &body)]))
}

/// The boxes of one `stbl`, to be changed one at a time.
struct Tables {
    stsd: Vec<u8>,
    stts: Vec<u8>,
    ctts: Option<Vec<u8>>,
    stsz: Vec<u8>,
    stsc: Vec<u8>,
    stco: Vec<u8>,
    extra: Vec<u8>,
}

impl Tables {
    /// Two samples per chunk, one in the last; `first` is where the media
    /// data starts.
    fn plain(first: u32) -> Self {
        let mut offsets = Vec::new();
        let mut at = first;
        for pair in SIZES.chunks(2) {
            offsets.push(at);
            at += pair.iter().sum::<u32>();
        }
        let mut ctts = words(&[SHIFTS.len() as u32]);
        for shift in SHIFTS {
            ctts.extend(words(&[1, shift as u32]));
        }
        Self {
            stsd: entry(b"avc1", Some(b"avcC")),
            stts: full(b"stts", 0, 0, &words(&[1, 5, DURATION])),
            ctts: Some(full(b"ctts", 0, 0, &ctts)),
            stsz: full(b"stsz", 0, 0, &cat(&[&words(&[0, 5]), &words(&SIZES)])),
            stsc: full(b"stsc", 0, 0, &words(&[2, 1, 2, 1, 3, 1, 1])),
            stco: full(
                b"stco",
                0,
                0,
                &cat(&[&words(&[offsets.len() as u32]), &words(&offsets)]),
            ),
            extra: Vec::new(),
        }
    }

    fn stbl(&self) -> Vec<u8> {
        bx(
            b"stbl",
            &cat(&[
                &self.stsd,
                &self.stts,
                self.ctts.as_deref().unwrap_or_default(),
                &self.stsz,
                &self.stsc,
                &self.stco,
                &self.extra,
            ]),
        )
    }
}

/// Extra boxes placed in a track, the movie and at the top level.
#[derive(Default)]
struct Extras {
    trak: Vec<u8>,
    minf: Vec<u8>,
    moov: Vec<u8>,
    top: Vec<u8>,
}

/// A track; `minf` holds the boxes of its `minf`, the `stbl` among them.
fn trak(id: u32, handler: &[u8; 4], duration: u32, minf: &[u8], extra: &[u8]) -> Vec<u8> {
    let tkhd = full(b"tkhd", 0, 3, &cat(&[&words(&[0, 0, id]), &[0u8; 68]]));
    let mdhd = full(b"mdhd", 0, 0, &words(&[0, 0, 30000, duration, 0]));
    let hdlr = full(b"hdlr", 0, 0, &cat(&[&words(&[0]), handler, &[0u8; 13]]));
    let minf = bx(b"minf", minf);
    let mdia = bx(b"mdia", &cat(&[&mdhd, &hdlr, &minf]));
    bx(b"trak", &cat(&[&tkhd, &mdia, extra]))
}

/// A plain file: ftyp, mdat, moov. Returns the file and where the media data
/// starts.
fn plain_file(
    tables: impl FnOnce(u32) -> Tables,
    wide_mdat: bool,
    extras: &Extras,
) -> (Vec<u8>, u32) {
    let payload = vec![7u8; SIZES.iter().sum::<u32>() as usize];
    let mdat = if wide_mdat {
        let mut b = 1u32.to_be_bytes().to_vec();
        b.extend_from_slice(b"mdat");
        b.extend_from_slice(&(payload.len() as u64 + 16).to_be_bytes());
        b.extend_from_slice(&payload);
        b
    } else {
        bx(b"mdat", &payload)
    };
    let first = (FTYP.len() + mdat.len() - payload.len()) as u32;
    let tables = tables(first);
    let minf = cat(&[&tables.stbl(), &extras.minf]);
    let video = trak(1, b"vide", DURATION * 5, &minf, &extras.trak);
    let moov = bx(b"moov", &cat(&[&mvhd(), &video, &extras.moov]));
    (cat(&[&FTYP, &mdat, &moov, &extras.top]), first)
}

/// The movie header: a timescale of 600, which edit durations use.
fn mvhd() -> Vec<u8> {
    full(
        b"mvhd",
        0,
        0,
        &cat(&[&words(&[0, 0, MOVIE_TIMESCALE, 0]), &[0u8; 80]]),
    )
}

const MOVIE_TIMESCALE: u32 = 600;

fn plain() -> Vec<u8> {
    plain_file(Tables::plain, false, &Extras::default()).0
}

fn samples_of(video: &Mp4Video, first: u64) -> Vec<(u64, u32, u32, i64, bool)> {
    video
        .samples
        .iter()
        .map(|s| (s.offset - first, s.size, s.duration, s.cts, s.shown))
        .collect()
}

/// The same clip as one fragment, with sizes and times in the `trun`. Returns
/// the file and where the media data starts.
fn fragmented() -> (Vec<u8>, u64) {
    fragmented_with(&[])
}

/// [`fragmented`] with `trak_extra` added to its track.
fn fragmented_with(trak_extra: &[u8]) -> (Vec<u8>, u64) {
    let moov = fragmented_moov(&[full(b"trex", 0, 0, &words(&[1, 1, 0, 0, 0]))], trak_extra);
    let moof = |data_offset: u32| {
        let tfhd = full(b"tfhd", 0, 0x2_0000, &words(&[1]));
        let tfdt = full(b"tfdt", 1, 0, &0u64.to_be_bytes());
        let mut run = words(&[5, data_offset]);
        for (size, shift) in SIZES.iter().zip(SHIFTS) {
            run.extend(words(&[DURATION, *size, shift as u32]));
        }
        let trun = full(b"trun", 0, 0x1 | 0x100 | 0x200 | 0x800, &run);
        bx(
            b"moof",
            &cat(&[
                &full(b"mfhd", 0, 0, &words(&[1])),
                &bx(b"traf", &cat(&[&tfhd, &tfdt, &trun])),
            ]),
        )
    };
    let moof_len = moof(0).len() as u32;
    let payload = vec![7u8; SIZES.iter().sum::<u32>() as usize];
    let file = cat(&[&FTYP, &moov, &moof(moof_len + 8), &bx(b"mdat", &payload)]);
    let first = (FTYP.len() + moov.len()) as u64 + u64::from(moof_len) + 8;
    (file, first)
}

/// The `moov` of a fragmented file: a video track with empty tables, then
/// `mvex` holding `trexs`.
fn fragmented_moov(trexs: &[Vec<u8>], trak_extra: &[u8]) -> Vec<u8> {
    let empty = Tables {
        stsd: entry(b"avc1", Some(b"avcC")),
        stts: full(b"stts", 0, 0, &words(&[0])),
        ctts: None,
        stsz: full(b"stsz", 0, 0, &words(&[0, 0])),
        stsc: full(b"stsc", 0, 0, &words(&[0])),
        stco: full(b"stco", 0, 0, &words(&[0])),
        extra: Vec::new(),
    };
    bx(
        b"moov",
        &cat(&[
            &mvhd(),
            &trak(1, b"vide", 0, &empty.stbl(), trak_extra),
            &bx(b"mvex", &trexs.concat()),
        ]),
    )
}

fn position(bytes: &[u8], fourcc: &[u8; 4]) -> usize {
    bytes.windows(4).position(|w| w == fourcc).unwrap()
}

#[test]
fn a_plain_table_lists_the_samples_in_decode_order_from_the_first_shown_frame() {
    let (bytes, first) = plain_file(Tables::plain, false, &Extras::default());
    assert_eq!(read(&bytes.as_slice()).unwrap().codec, VideoCodec::H264);
    let video = parse(&bytes).unwrap();
    assert_eq!((video.width, video.height), (64, 48));
    assert_eq!(video.config.as_deref(), Some(&[1, 2, 3, 4, 5][..]));
    assert_eq!(video.timescale, 30000);
    assert_eq!((video.start_time, video.duration), (0.0, 5000.0 / 30000.0));
    let mut at = 0u64;
    let want: Vec<_> = SIZES
        .iter()
        .zip(CTS)
        .map(|(size, cts)| {
            let row = (at, *size, DURATION, cts, true);
            at += u64::from(*size);
            row
        })
        .collect();
    assert_eq!(samples_of(&video, u64::from(first)), want);
}

#[test]
fn a_fragment_gives_the_same_samples_as_the_plain_table() {
    let (plain, plain_first) = plain_file(Tables::plain, false, &Extras::default());
    let (fragments, fragment_first) = fragmented();
    let plain = parse(&plain).unwrap();
    let fragments = parse(&fragments).unwrap();
    assert_eq!(
        samples_of(&plain, u64::from(plain_first)),
        samples_of(&fragments, fragment_first)
    );
    // `mdhd` says zero for a fragmented file, so the samples give the end.
    assert_eq!(fragments.duration, plain.duration);
}

#[test]
fn media_data_with_a_64_bit_size_or_running_to_the_end_is_found() {
    let (wide, first) = plain_file(Tables::plain, true, &Extras::default());
    let video = parse(&wide).unwrap();
    assert_eq!(video.samples[0].offset, u64::from(first));
    // Size 0 on the last box: it runs to the end of the file.
    let (mut open, first) = fragmented();
    let at = open.len() - SIZES.iter().sum::<u32>() as usize - 8;
    assert_eq!(&open[at + 4..at + 8], b"mdat");
    open[at..at + 4].copy_from_slice(&0u32.to_be_bytes());
    let video = parse(&open).unwrap();
    assert_eq!(video.samples[0].offset, first);
}

#[test]
fn boxes_that_are_skipped_are_never_parsed() {
    // A QuickTime `udta` ends with four zero bytes: a child of size 0, which
    // made the re_mp4 crate loop forever. These boxes hold bytes no parser
    // could follow, so only skipping them by size lets the file through.
    let junk = [0u8, 0, 0, 0, 0xFF, 0xFF, 0xFF, 0xFF, 1, 2, 3];
    let (want, first) = plain_file(Tables::plain, false, &Extras::default());
    let want = samples_of(&parse(&want).unwrap(), u64::from(first));
    let in_stbl = |first| {
        let mut tables = Tables::plain(first);
        tables.extra = cat(&[
            &bx(b"sgpd", &junk),
            &bx(b"sdtp", &junk),
            &bx(b"stss", &junk),
        ]);
        tables
    };
    let (bytes, first) = plain_file(in_stbl, false, &Extras::default());
    assert_eq!(samples_of(&parse(&bytes).unwrap(), u64::from(first)), want);
    for kind in [
        b"udta", b"meta", b"free", b"skip", b"uuid", b"emsg", b"sidx",
    ] {
        let extras = Extras {
            trak: bx(kind, &junk),
            minf: bx(kind, &junk),
            moov: cat(&[&bx(kind, &junk), &bx(b"udta", &[0, 0, 0, 0])]),
            top: cat(&[&bx(kind, &junk), &bx(kind, &[0, 0, 0, 0])]),
        };
        let (bytes, first) = plain_file(Tables::plain, false, &extras);
        let got = samples_of(&parse(&bytes).unwrap(), u64::from(first));
        assert_eq!(got, want, "{}", kind.escape_ascii());
    }
}

fn assert_invalid(bytes: &[u8], why: &str) {
    let r = parse(bytes);
    assert!(
        matches!(r, Err(MmError::InvalidInput { .. })),
        "{why}: {:?}",
        r.map(|_| ())
    );
}

fn assert_unsupported(bytes: &[u8], why: &str) {
    let r = parse(bytes);
    assert!(
        matches!(r, Err(MmError::Unsupported { .. })),
        "{why}: {:?}",
        r.map(|_| ())
    );
}

fn with(change: impl FnOnce(&mut Tables)) -> Vec<u8> {
    let tables = |first| {
        let mut tables = Tables::plain(first);
        change(&mut tables);
        tables
    };
    plain_file(tables, false, &Extras::default()).0
}

fn table(kind: &[u8; 4], body: &[u32]) -> Vec<u8> {
    full(kind, 0, 0, &words(body))
}

#[test]
fn tables_that_disagree_or_cannot_be_followed_are_invalid() {
    let d = DURATION;
    assert_invalid(
        &with(|t| t.stts = table(b"stts", &[1, 4, d])),
        "stts too short",
    );
    assert_invalid(
        &with(|t| t.stts = table(b"stts", &[1, 6, d])),
        "stts too long",
    );
    assert_invalid(
        &with(|t| t.stts = table(b"stts", &[1, 0x7FFF_FFFF, 1])),
        "stts count",
    );
    assert_invalid(
        &with(|t| t.stts = table(b"stts", &[9, 5, 1])),
        "stts past its box",
    );
    assert_invalid(
        &with(|t| t.ctts = Some(table(b"ctts", &[1, 4, 0]))),
        "ctts too short",
    );
    assert_invalid(
        &with(|t| t.stsz = table(b"stsz", &[0, 0x7FFF_FFFF])),
        "stsz count past the file",
    );
    assert_invalid(
        &with(|t| t.stsz = table(b"stsz", &[0, 5, 1, 2])),
        "stsz past its box",
    );
    assert_invalid(
        &with(|t| t.stsz = table(b"stsz", &[0, 5, 10, 20, 0, 40, 50])),
        "an empty sample",
    );
    assert_invalid(&with(|t| t.stsc = table(b"stsc", &[0])), "no chunk runs");
    assert_invalid(
        &with(|t| t.stsc = table(b"stsc", &[1, 1, 0, 1])),
        "no samples per chunk",
    );
    assert_invalid(
        &with(|t| t.stsc = table(b"stsc", &[1, 2, 2, 1])),
        "first run after chunk 1",
    );
    assert_invalid(
        &with(|t| t.stsc = table(b"stsc", &[2, 1, 1, 1, 1, 1, 1])),
        "two runs from one chunk",
    );
    assert_invalid(
        &with(|t| t.stsc = table(b"stsc", &[1, 1, 2, 2])),
        "description 2",
    );
    assert_invalid(&with(|t| t.stco = table(b"stco", &[0])), "no chunks");
    assert_invalid(
        &with(|t| t.stco = table(b"stco", &[1, 24])),
        "too few chunks",
    );
    assert_invalid(
        &with(|t| t.stco = table(b"stco", &[3, 24, 54, 0xFFFF_FFF0])),
        "a chunk outside the file",
    );
    assert_invalid(&with(|t| t.extra = table(b"co64", &[0])), "stco and co64");
    assert_invalid(&with(|t| t.extra = t.stsz.clone()), "two stsz");
}

#[test]
fn a_media_duration_that_ends_before_the_last_sample_is_invalid() {
    let mut bytes = plain();
    // mdhd: version/flags, creation, modification, timescale, duration.
    let at = position(&bytes, b"mdhd") + 20;
    bytes[at..at + 4].copy_from_slice(&1u32.to_be_bytes());
    assert_invalid(&bytes, "duration 1");
    // Zero means unknown: it is taken from the samples.
    bytes[at..at + 4].copy_from_slice(&0u32.to_be_bytes());
    assert_eq!(parse(&bytes).unwrap().duration, 5000.0 / 30000.0);
}

#[test]
fn boxes_that_do_not_fit_are_invalid() {
    let bytes = plain();
    let at = position(&bytes, b"stts") - 4;
    // Size 0 (to the end of the file) is for top-level boxes only.
    for size in [0u32, 1, 4, 7, 0xFFFF_FFFF, 0x7FFF_FFFF] {
        let mut bad = bytes.clone();
        bad[at..at + 4].copy_from_slice(&size.to_be_bytes());
        assert_invalid(&bad, &format!("size {size}"));
    }
    // A top-level box past the end of the file, and a short 64-bit size.
    let mut bad = bytes.clone();
    let moov_at = position(&bad, b"moov") - 4;
    bad[moov_at..moov_at + 4].copy_from_slice(&0xFFFF_FFF0u32.to_be_bytes());
    assert_invalid(&bad, "moov past the file");
    assert_invalid(&cat(&[&FTYP, b"\0\0\0\x01mdat"]), "short 64-bit size");
}

/// A file of nothing but empty boxes of one kind, as many as `len` holds,
/// made up as it is read: no byte of it exists before a read asks for it.
struct EmptyBoxes {
    kind: &'static [u8; 4],
    len: u64,
    reads: std::cell::Cell<u64>,
}

impl ByteSource for EmptyBoxes {
    fn len(&self) -> u64 {
        self.len
    }
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> std::io::Result<()> {
        assert!(offset + buf.len() as u64 <= self.len, "read past the end");
        self.reads.set(self.reads.get() + 1);
        let header = bx(self.kind, &[]);
        for (i, b) in buf.iter_mut().enumerate() {
            *b = header[(offset as usize + i) % 8];
        }
        Ok(())
    }
}

#[test]
fn a_file_of_many_tiny_boxes_is_refused_after_a_bounded_number_of_reads() {
    for kind in [b"mdat", b"free"] {
        // One box too many: refused at the cap, with one header read per box
        // allowed and nothing read for the rest of the file.
        let over = EmptyBoxes {
            kind,
            len: (MAX_BOXES + 1) * 8,
            reads: Default::default(),
        };
        let r = read(&over).map(|_| ());
        assert!(matches!(r, Err(MmError::LimitExceeded { .. })), "{r:?}");
        assert_eq!(over.reads.get(), MAX_BOXES);
        // Exactly the cap is walked to the end (and then has no moov box).
        let at = EmptyBoxes {
            kind,
            len: MAX_BOXES * 8,
            reads: Default::default(),
        };
        let r = read(&at).map(|_| ());
        assert!(matches!(r, Err(MmError::InvalidInput { .. })), "{r:?}");
        assert_eq!(at.reads.get(), MAX_BOXES);
    }
}

#[test]
fn a_movie_needs_one_moov_and_a_video_track() {
    let bytes = plain();
    let moov_at = position(&bytes, b"moov") - 4;
    assert_invalid(&bytes[..moov_at], "no moov");
    assert_invalid(&cat(&[&bytes, &bytes[moov_at..]]), "two moov");
    let mut audio = bytes.clone();
    let at = position(&audio, b"vide");
    audio[at..at + 4].copy_from_slice(b"soun");
    assert_invalid(&audio, "audio only");
}

#[test]
fn what_is_valid_but_not_handled_is_unsupported() {
    let two = full(b"stsd", 0, 0, &cat(&[&words(&[2]), &[0u8; 16]]));
    assert_unsupported(&with(|t| t.stsd = two), "two sample descriptions");
    assert_unsupported(&with(|t| t.stsd = entry(b"mp4v", None)), "MPEG-4 part 2");
    assert_unsupported(&with(|t| t.stsd = entry(b"encv", None)), "encrypted entry");
    assert_unsupported(&with(|t| t.extra = bx(b"stz2", &[0; 12])), "stz2");
    let v1 = full(b"stsz", 1, 0, &words(&[0, 0]));
    assert_unsupported(&with(|t| t.stsz = v1), "stsz version 1");
    // VP8 parses, so probing reports it; only decoding refuses it.
    let vp8 = with(|t| t.stsd = entry(b"vp08", None));
    assert_eq!(read(&vp8.as_slice()).unwrap().codec, VideoCodec::Vp8);
    parse(&vp8).unwrap();
    // A codec that needs its configuration record, without one.
    assert_invalid(
        &with(|t| t.stsd = entry(b"hvc1", None)),
        "hvc1 without hvcC",
    );
}

#[test]
fn a_fragment_that_cannot_be_placed_or_read_is_refused() {
    let (bytes, _) = fragmented();
    // A flag no part of this parser handles, in each box that has flags.
    for (name, flag) in [(b"trun", 0x1_0000u32), (b"tfhd", 0x4), (b"tfhd", 0x1_0000)] {
        let mut bad = bytes.clone();
        let at = position(&bad, name) + 5;
        let flags = u32::from_be_bytes([0, bad[at], bad[at + 1], bad[at + 2]]);
        bad[at..at + 3].copy_from_slice(&(flags | flag).to_be_bytes()[1..]);
        assert_unsupported(&bad, &format!("{} flags", name.escape_ascii()));
    }
    // A fragment before the moov.
    let moov_at = position(&bytes, b"moov") - 4;
    let moof_at = position(&bytes, b"moof") - 4;
    let reordered = cat(&[
        &bytes[..moov_at],
        &bytes[moof_at..],
        &bytes[moov_at..moof_at],
    ]);
    assert_invalid(&reordered, "moof before moov");
    // Media data that stops short.
    assert_invalid(&bytes[..bytes.len() - 1], "short mdat");
}

/// Every MP4 here parses, and no prefix or one-byte change of it panics or
/// hangs: the sweep runs on its own thread, and a hang fails the test.
#[test]
fn no_truncation_or_corruption_of_a_file_panics_or_hangs() {
    let (done, finished) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        sweep();
        done.send(()).unwrap();
    });
    // A panic in the sweep drops `done`: an error, not a timeout.
    match finished.recv_timeout(std::time::Duration::from_secs(120)) {
        Ok(()) => {}
        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => panic!("the sweep hangs"),
        Err(_) => panic!("the sweep panicked"),
    }
}

fn sweep() {
    let mut files = vec![plain(), fragmented().0];
    for name in ["h264.mp4", "hevc.mp4", "av1.mp4", "vp9.mp4", "vp9_sar.mp4"] {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/video_decode")
            .join(name);
        files.push(std::fs::read(path).unwrap());
    }
    for (n, bytes) in files.into_iter().enumerate() {
        let full = parse(&bytes).unwrap().samples.len();
        assert!(full > 0);
        for end in 0..bytes.len() {
            // A prefix is an error. The one exception: a fragmented file cut
            // between fragments is a valid, shorter file (nothing in it says
            // more is coming); the demuxer refuses it if it has no frames.
            if let Ok(video) = parse(&bytes[..end]) {
                assert!(
                    n == 1 && video.samples.len() < full,
                    "prefix of {end} bytes"
                );
            }
        }
        let mut mutated = bytes.clone();
        for at in 0..bytes.len() {
            for value in [0x00, 0x01, 0x7F, 0x80, 0xFF, bytes[at] ^ 0x10] {
                mutated[at] = value;
                let _ = parse(&mutated);
            }
            mutated[at] = bytes[at];
        }
    }
}

const ONE: u32 = 0x1_0000;

/// The clip with an edit list: rows of (duration in the movie timescale,
/// media time, rate), version 0.
fn edited(rows: &[(u32, i32, u32)]) -> Result<Mp4Video> {
    let mut body = words(&[rows.len() as u32]);
    for (duration, time, rate) in rows {
        body.extend(words(&[*duration, *time as u32, *rate]));
    }
    let extras = Extras {
        trak: bx(b"edts", &full(b"elst", 0, 0, &body)),
        ..Extras::default()
    };
    parse(&plain_file(Tables::plain, false, &extras).0)
}

/// (time, shown) of each sample in decode order. In display order the raw
/// times are 1000 (sample 0), 2000 (2), 3000 (1), 4000 (4), 5000 (3), each
/// frame 1000 long; one movie tick is 50 media units.
fn times(video: &Mp4Video) -> Vec<(i64, bool)> {
    video.samples.iter().map(|s| (s.cts, s.shown)).collect()
}

#[test]
fn an_edit_that_only_skips_the_b_frame_delay_changes_nothing() {
    let none = parse(&plain()).unwrap();
    let rows: [&[(u32, i32, u32)]; 3] = [
        &[],
        &[(100, 1000, ONE)],
        // Two rows that continue one another are one.
        &[(40, 1000, ONE), (60, 3000, ONE)],
    ];
    for rows in rows {
        let video = edited(rows).unwrap();
        assert_eq!(times(&video), times(&none), "{rows:?}");
        assert_eq!((video.start_time, video.duration), (0.0, none.duration));
    }
}

#[test]
fn an_edit_hides_the_frames_outside_it_and_times_start_at_its_start() {
    // From 3000 for 3000 units (60 ticks): the last three frames.
    let video = edited(&[(60, 3000, ONE)]).unwrap();
    let want = [
        (-2000, false),
        (0, true),
        (-1000, false),
        (2000, true),
        (1000, true),
    ];
    assert_eq!(times(&video), want);
    assert_eq!((video.start_time, video.duration), (0.0, 3000.0 / 30000.0));
    // From 1000 for 2000 units: the first two.
    let video = edited(&[(40, 1000, ONE)]).unwrap();
    let want = [
        (0, true),
        (2000, false),
        (1000, true),
        (4000, false),
        (3000, false),
    ];
    assert_eq!(times(&video), want);
    assert_eq!(video.duration, 2000.0 / 30000.0);
    // Past the media: nothing shows (the demuxer then refuses the clip).
    let video = edited(&[(20, 100_000, ONE)]).unwrap();
    assert!(video.samples.iter().all(|s| !s.shown));
}

#[test]
fn a_frame_cut_by_the_edge_shows_unless_only_a_sliver_is_left() {
    // From 1500: the first frame shows for its last 500 units, moved to 0.
    let video = edited(&[(80, 1500, ONE)]).unwrap();
    let want = [
        (0, true),
        (1500, true),
        (500, true),
        (3500, true),
        (2500, true),
    ];
    assert_eq!(times(&video), want);
    assert_eq!(video.duration, 4000.0 / 30000.0);
    // From 1990: 10 units of it are left, less than a movie tick, so time
    // starts at the next frame, 10 units into the window.
    let video = edited(&[(100, 1990, ONE)]).unwrap();
    assert_eq!(times(&video)[0], (-1000, false));
    assert_eq!(video.start_time, 10.0 / 30000.0);
    // A window of one tick inside a frame shows that frame.
    let video = edited(&[(1, 1500, ONE)]).unwrap();
    assert_eq!(times(&video)[0], (0, true));
    assert_eq!(video.duration, 50.0 / 30000.0);
}

#[test]
fn a_window_that_starts_before_the_media_starts_time_at_the_first_frame() {
    // From 0 to 6000, the media starting at 1000: the frames keep their
    // place, and the first one is at start_time.
    let video = edited(&[(120, 0, ONE)]).unwrap();
    let none = parse(&plain()).unwrap();
    assert_eq!(times(&video), times(&none));
    assert_eq!(video.start_time, 1000.0 / 30000.0);
    assert_eq!(video.duration, none.duration);
}

#[test]
fn rows_that_continue_to_within_a_movie_tick_are_joined() {
    // 33 ticks are 1650 units; the next row starts 10 units later.
    let video = edited(&[(33, 1000, ONE), (67, 2660, ONE)]).unwrap();
    assert_eq!(times(&video), times(&parse(&plain()).unwrap()));
}

#[test]
fn an_empty_edit_first_delays_the_start() {
    // 1.5 s of nothing (900 ticks), then the clip from its first frame.
    let video = edited(&[(900, -1, ONE), (100, 1000, ONE)]).unwrap();
    assert_eq!(times(&video), times(&parse(&plain()).unwrap()));
    assert_eq!((video.start_time, video.duration), (1.5, 5000.0 / 30000.0));
    // The same list in version 1 (64-bit fields).
    let mut body = words(&[2]);
    for (duration, time) in [(900u64, -1i64), (100, 1000)] {
        body.extend(duration.to_be_bytes());
        body.extend(time.to_be_bytes());
        body.extend(words(&[ONE]));
    }
    let extras = Extras {
        trak: bx(b"edts", &full(b"elst", 1, 0, &body)),
        ..Extras::default()
    };
    let wide = parse(&plain_file(Tables::plain, false, &extras).0).unwrap();
    assert_eq!(times(&wide), times(&video));
    assert_eq!(wide.start_time, 1.5);
}

#[test]
fn edit_lists_that_reorder_repeat_skip_pause_or_change_speed_are_refused() {
    let rows: [&[(u32, i32, u32)]; 13] = [
        &[(40, 3000, ONE), (40, 1000, ONE)],
        &[(40, 1000, ONE), (40, 1000, ONE)],
        // Skips 2000..3000.
        &[(20, 1000, ONE), (20, 3000, ONE)],
        &[(40, 1000, ONE), (20, -1, ONE), (40, 3000, ONE)],
        &[(100, 1000, ONE), (20, -1, ONE)],
        &[(20, -1, ONE), (20, -1, ONE), (100, 1000, ONE)],
        &[(20, -1, ONE)],
        &[(100, 1000, 2 * ONE)],
        &[(100, 1000, 0)],
        // No duration outside a fragmented file.
        &[(0, 1000, ONE)],
        // The leading empty row is held to the same rules.
        &[(20, -1, 0), (100, 1000, ONE)],
        &[(0, -1, ONE), (100, 1000, ONE)],
        // Rows within a tick of one another, but repeating.
        &[(1, 1000, ONE), (1, 1000, ONE)],
    ];
    for rows in rows {
        let r = edited(rows);
        assert!(
            matches!(r, Err(MmError::Unsupported { .. })),
            "{rows:?}: {:?}",
            r.map(|_| ())
        );
    }
    assert_invalid_edit(edited(&[(100, -2, ONE)]), "media time -2");
    // Edit durations need the movie timescale.
    let extras = Extras {
        trak: bx(b"edts", &full(b"elst", 0, 0, &words(&[1, 100, 1000, ONE]))),
        ..Extras::default()
    };
    let mut bytes = plain_file(Tables::plain, false, &extras).0;
    let at = position(&bytes, b"mvhd");
    bytes[at..at + 4].copy_from_slice(b"free");
    assert_invalid(&bytes, "no mvhd");
}

fn assert_invalid_edit(r: Result<Mp4Video>, why: &str) {
    assert!(
        matches!(r, Err(MmError::InvalidInput { .. })),
        "{why}: {:?}",
        r.map(|_| ())
    );
}

#[test]
fn a_fragmented_file_may_leave_its_last_edit_open() {
    // Duration 0 on the last row: to the end of the fragments.
    let elst = full(b"elst", 0, 0, &words(&[1, 0, 1000, ONE]));
    let (bytes, first) = fragmented_with(&bx(b"edts", &elst));
    let (plain, plain_first) = plain_file(Tables::plain, false, &Extras::default());
    let (open, plain) = (parse(&bytes).unwrap(), parse(&plain).unwrap());
    assert_eq!(
        samples_of(&open, first),
        samples_of(&plain, u64::from(plain_first))
    );
    assert_eq!(open.duration, plain.duration);
}

/// The chunk offsets of a `stco` built by [`table`].
fn stco_offsets(stco: &[u8]) -> Vec<u32> {
    stco[16..]
        .chunks(4)
        .map(|w| u32::from_be_bytes(w.try_into().unwrap()))
        .collect()
}

/// A `stco` of `offsets`.
fn stco(offsets: &[u32]) -> Vec<u8> {
    full(
        b"stco",
        0,
        0,
        &cat(&[&words(&[offsets.len() as u32]), &words(offsets)]),
    )
}

#[test]
fn other_table_layouts_give_the_same_samples() {
    let (bytes, first) = plain_file(Tables::plain, false, &Extras::default());
    let want = samples_of(&parse(&bytes).unwrap(), u64::from(first));
    let same = |change: &dyn Fn(&mut Tables)| {
        let (bytes, first) = plain_file(
            |at| {
                let mut tables = Tables::plain(at);
                change(&mut tables);
                tables
            },
            false,
            &Extras::default(),
        );
        samples_of(&parse(&bytes).unwrap(), u64::from(first))
    };
    // 64-bit chunk offsets.
    let co64 = |t: &mut Tables| {
        let offsets = stco_offsets(&t.stco);
        let mut body = words(&[offsets.len() as u32]);
        for offset in offsets {
            body.extend(u64::from(offset).to_be_bytes());
        }
        t.stco = full(b"co64", 0, 0, &body);
    };
    assert_eq!(same(&co64), want, "co64");
    // One sample per chunk.
    let single = |t: &mut Tables| {
        let mut at = stco_offsets(&t.stco)[0];
        let mut offsets = Vec::new();
        for size in SIZES {
            offsets.push(at);
            at += size;
        }
        t.stsc = table(b"stsc", &[1, 1, 1, 1]);
        t.stco = stco(&offsets);
    };
    assert_eq!(same(&single), want, "one sample per chunk");
    // Composition offsets that go negative, in version 1 and (as muxers
    // write them) version 0: only the origin moves, and it is shifted away.
    for version in [0, 1] {
        let negative = move |t: &mut Tables| {
            let mut body = words(&[SHIFTS.len() as u32]);
            for shift in SHIFTS {
                body.extend(words(&[1, (shift - 1000) as u32]));
            }
            t.ctts = Some(full(b"ctts", version, 0, &body));
        };
        assert_eq!(same(&negative), want, "ctts version {version}");
    }
    // One size for every sample: each chunk's samples follow one another.
    let got = same(&|t: &mut Tables| t.stsz = table(b"stsz", &[10, 5]));
    let offsets: Vec<_> = got.iter().map(|s| (s.0, s.1)).collect();
    assert_eq!(offsets, [(0, 10), (10, 10), (30, 10), (40, 10), (100, 10)]);
}

#[test]
fn tables_that_describe_more_samples_than_stsz_are_invalid() {
    // Three chunks of two samples: six, for five sizes.
    assert_invalid(
        &with(|t| t.stsc = table(b"stsc", &[1, 1, 2, 1])),
        "six samples in the chunks",
    );
    // No samples, while stts still lists five.
    assert_invalid(
        &with(|t| {
            t.stsz = table(b"stsz", &[0, 0]);
            t.stsc = table(b"stsc", &[0]);
            t.stco = table(b"stco", &[0]);
        }),
        "stts without samples",
    );
}

#[test]
fn a_sample_must_lie_in_the_media_data_of_this_file() {
    // The first chunk at offset 0: inside the ftyp.
    assert_invalid(
        &with(|t| {
            let mut offsets = stco_offsets(&t.stco);
            offsets[0] = 0;
            t.stco = stco(&offsets);
        }),
        "a sample in the ftyp",
    );
    // A dref of `count` with one entry of `kind` and `flags`.
    let dref = |count: u32, kind: &[u8; 4], flags: u32| {
        let entry = full(kind, 0, flags, &[]);
        let dref = full(b"dref", 0, 0, &cat(&[&words(&[count]), &entry]));
        let extras = Extras {
            minf: bx(b"dinf", &dref),
            ..Extras::default()
        };
        parse(&plain_file(Tables::plain, false, &extras).0).map(|_| ())
    };
    // Flag 1 on `url `, `urn ` or QuickTime's `alis`: the data is here.
    for kind in [b"url ", b"urn ", b"alis"] {
        dref(1, kind, 1).unwrap();
    }
    for (r, why) in [
        (dref(1, b"url ", 0), "data in another file"),
        (dref(1, b"free", 1), "an unknown data reference"),
    ] {
        assert!(
            matches!(r, Err(MmError::Unsupported { .. })),
            "{why}: {r:?}"
        );
    }
    let r = dref(2, b"url ", 1);
    assert!(matches!(r, Err(MmError::InvalidInput { .. })), "{r:?}");
    // A dref cut after its version and flags, or after its count, is invalid
    // (the first used to be sliced past its end).
    let count_only = words(&[1]);
    for body in [&[][..], &count_only[..]] {
        let extras = Extras {
            minf: bx(b"dinf", &full(b"dref", 0, 0, body)),
            ..Extras::default()
        };
        let r = parse(&plain_file(Tables::plain, false, &extras).0).map(|_| ());
        assert!(matches!(r, Err(MmError::InvalidInput { .. })), "{r:?}");
    }
    let mut second = plain();
    let at = position(&second, b"avc1") + 4 + 6;
    second[at..at + 2].copy_from_slice(&2u16.to_be_bytes());
    assert_unsupported(&second, "the second data reference");
}

#[test]
fn a_sample_larger_than_the_limit_is_refused_before_it_is_read() {
    let huge = MAX_SAMPLE_SIZE + 1;
    let r = parse(&with(|t| {
        t.stsz = table(b"stsz", &[0, 5, 10, 20, huge, 40, 50]);
    }));
    assert!(
        matches!(r, Err(MmError::LimitExceeded { .. })),
        "{:?}",
        r.map(|_| ())
    );
}

#[test]
fn an_unknown_media_duration_is_taken_from_the_samples() {
    let mut bytes = plain();
    let at = position(&bytes, b"mdhd") + 20;
    bytes[at..at + 4].copy_from_slice(&u32::MAX.to_be_bytes());
    assert_eq!(parse(&bytes).unwrap().duration, 5000.0 / 30000.0);
}

#[test]
fn framing_after_the_last_box_or_entry_is_checked() {
    let bytes = plain();
    // A 32-bit zero (as QuickTime ends a list of boxes) may end the file;
    // other short tails may not.
    parse(&cat(&[&bytes, &[0, 0, 0, 0]])).unwrap();
    assert_invalid(&cat(&[&bytes, &[1, 2, 3, 4, 5, 6, 7]]), "seven bytes");
    assert_invalid(&cat(&[&bytes, &[0, 0, 0]]), "three zero bytes");
    // An entry beyond the count, two configurations, and a VP9 entry too
    // short for a visual sample entry.
    let stsd = entry(b"avc1", Some(b"avcC"));
    let avc1 = &stsd[16..];
    let two = full(b"stsd", 0, 0, &cat(&[&words(&[1]), avc1, avc1]));
    assert_invalid(&with(|t| t.stsd = two), "an uncounted entry");
    let mut body = avc1[8..].to_vec();
    body.extend_from_slice(&bx(b"avcC", &[9]));
    let twice = full(b"stsd", 0, 0, &cat(&[&words(&[1]), &bx(b"avc1", &body)]));
    assert_invalid(&with(|t| t.stsd = twice), "two avcC");
    let mut short = vec![0u8; 28];
    short[7] = 1;
    let vp9 = full(b"stsd", 0, 0, &cat(&[&words(&[1]), &bx(b"vp09", &short)]));
    assert_invalid(&with(|t| t.stsd = vp9), "short vp09 entry");
}

#[test]
fn a_frame_that_displays_nothing_does_not_set_the_start() {
    // As a hidden VP9 frame: the first sample (raw time 1000) outputs nothing,
    // so time starts at the next frame to display (2000).
    let mut movie = read(&plain().as_slice()).unwrap();
    movie.samples[0].displays = false;
    let video = movie.finish().unwrap();
    // Four frames from 2000: 4000 units, not what mdhd says.
    assert_eq!(video.duration, 4000.0 / 30000.0);
    let want = [
        (-1000, false),
        (1000, true),
        (0, true),
        (3000, true),
        (2000, true),
    ];
    assert_eq!(times(&video), want);
}

/// Stands for the offset from a `moof` to its media data in [`fragments`].
const DATA: u32 = u32::MAX;

/// One `moof` and its `mdat`: `tfhd` flags and fields after the track id, an
/// optional `tfdt`, the `trun`s as (flags, fields from the sample count on),
/// and the length of the media data.
struct Fragment {
    tfhd: (u32, Vec<u32>),
    tfdt: Option<u64>,
    truns: Vec<(u32, Vec<u32>)>,
    data: usize,
}

/// A fragmented file with `trexs` and `fragments`; returns it and where each
/// fragment's media data starts.
fn fragments(trexs: &[Vec<u8>], fragments: &[Fragment]) -> (Vec<u8>, Vec<u64>) {
    let mut file = cat(&[&FTYP, &fragmented_moov(trexs, &[])]);
    let mut starts = Vec::new();
    for f in fragments {
        let moof = |data: u32| {
            let tfhd = full(
                b"tfhd",
                0,
                0x2_0000 | f.tfhd.0,
                &cat(&[&words(&[1]), &words(&f.tfhd.1)]),
            );
            let tfdt = f
                .tfdt
                .map(|t| full(b"tfdt", 1, 0, &t.to_be_bytes()))
                .unwrap_or_default();
            let truns: Vec<u8> = f
                .truns
                .iter()
                .flat_map(|(flags, fields)| {
                    let fields: Vec<_> = fields
                        .iter()
                        .map(|&v| if v == DATA { data } else { v })
                        .collect();
                    full(b"trun", 0, *flags, &words(&fields))
                })
                .collect();
            bx(
                b"moof",
                &cat(&[
                    &full(b"mfhd", 0, 0, &words(&[1])),
                    &bx(b"traf", &cat(&[&tfhd, &tfdt, &truns])),
                ]),
            )
        };
        let len = moof(0).len() as u32;
        starts.push(file.len() as u64 + u64::from(len) + 8);
        file.extend(moof(len + 8));
        file.extend(bx(b"mdat", &vec![7u8; f.data]));
    }
    (file, starts)
}

fn trex(fields: &[u32]) -> Vec<u8> {
    full(b"trex", 0, 0, &words(fields))
}

/// Two fragments: two samples of 20 bytes (the `tfhd` default size), then
/// two runs of one 10-byte sample (the `trex` default), at `tfdts`.
fn two_fragments(tfdts: [Option<u64>; 2], trexs: &[Vec<u8>]) -> (Vec<u8>, Vec<u64>) {
    let first = Fragment {
        tfhd: (0x10, vec![20]),
        tfdt: tfdts[0],
        truns: vec![(0x1, vec![2, DATA])],
        data: 40,
    };
    let second = Fragment {
        tfhd: (0, vec![]),
        tfdt: tfdts[1],
        truns: vec![(0x1, vec![1, DATA]), (0, vec![1])],
        data: 20,
    };
    fragments(trexs, &[first, second])
}

#[test]
fn fragment_defaults_runs_and_decode_times_are_followed() {
    // trex: track 1, description 1, 1000 units, 10 bytes, no flags.
    let defaults = [trex(&[1, 1, 1000, 10, 0])];
    let (bytes, starts) = two_fragments([None, None], &defaults);
    let video = parse(&bytes).unwrap();
    let got: Vec<_> = video
        .samples
        .iter()
        .map(|s| (s.offset, s.size, s.duration, s.cts))
        .collect();
    let (a, b) = (starts[0], starts[1]);
    let want = [
        (a, 20, 1000, 0),
        (a + 20, 20, 1000, 1000),
        (b, 10, 1000, 2000),
        (b + 10, 10, 1000, 3000),
    ];
    assert_eq!(got, want);
    assert_eq!(video.duration, 4000.0 / 30000.0);

    // A later tfdt leaves a gap: the sample before it lasts until then.
    let (bytes, _) = two_fragments([Some(0), Some(5000)], &defaults);
    let video = parse(&bytes).unwrap();
    let got: Vec<_> = video.samples.iter().map(|s| (s.duration, s.cts)).collect();
    assert_eq!(got, [(1000, 0), (4000, 1000), (1000, 5000), (1000, 6000)]);
    assert_eq!(video.duration, 7000.0 / 30000.0);
    // So does an empty fragment at the end.
    let first = Fragment {
        tfhd: (0, vec![]),
        tfdt: Some(0),
        truns: vec![(0x1, vec![1, DATA])],
        data: 10,
    };
    let empty = Fragment {
        tfhd: (0, vec![]),
        tfdt: Some(9000),
        truns: vec![],
        data: 0,
    };
    let (bytes, _) = fragments(&defaults, &[first, empty]);
    assert_eq!(parse(&bytes).unwrap().duration, 9000.0 / 30000.0);
}

#[test]
fn fragments_that_go_back_or_have_unclear_defaults_are_refused() {
    let defaults = [trex(&[1, 1, 1000, 10, 0])];
    assert_invalid(
        &two_fragments([Some(0), Some(1000)], &defaults).0,
        "tfdt goes back",
    );
    assert_unsupported(
        &two_fragments([None, None], &[trex(&[1, 2, 1000, 10, 0])]).0,
        "trex names description 2",
    );
    let twice = [defaults[0].clone(), defaults[0].clone()];
    assert_invalid(&two_fragments([None, None], &twice).0, "two trex");
}

#[test]
fn a_gap_after_reordered_frames_lengthens_the_last_one_shown() {
    // Three samples of 1000 units displayed at 0, 2000 and 1000: the one at
    // 2000 is last on screen, and the gap to 6000 is added to it.
    let defaults = [trex(&[1, 1, 1000, 10, 0])];
    let frames = Fragment {
        tfhd: (0, vec![]),
        tfdt: Some(0),
        truns: vec![(0x1 | 0x800, vec![3, DATA, 0, 1000, (-1000i32) as u32])],
        data: 30,
    };
    let empty = Fragment {
        tfhd: (0, vec![]),
        tfdt: Some(6000),
        truns: vec![],
        data: 0,
    };
    let (bytes, _) = fragments(&defaults, &[frames, empty]);
    let video = parse(&bytes).unwrap();
    let got: Vec<_> = video.samples.iter().map(|s| (s.duration, s.cts)).collect();
    assert_eq!(got, [(1000, 0), (4000, 2000), (1000, 1000)]);
    assert_eq!(video.duration, 6000.0 / 30000.0);
}

/// A file whose one `moof` holds a `traf` of track 2 (`duration-is-empty`,
/// no runs), then the video's with `video_flags` and one 10-byte sample.
fn after_another_track(video_flags: u32) -> Vec<u8> {
    let other = bx(
        b"traf",
        &full(b"tfhd", 0, 0x2_0000 | 0x1_0000, &words(&[2])),
    );
    let moof = |data: u32| {
        let tfhd = full(b"tfhd", 0, video_flags, &words(&[1]));
        let trun = full(b"trun", 0, 0x1, &words(&[1, data]));
        bx(
            b"moof",
            &cat(&[
                &full(b"mfhd", 0, 0, &words(&[1])),
                &other,
                &bx(b"traf", &cat(&[&tfhd, &trun])),
            ]),
        )
    };
    let len = moof(0).len() as u32;
    let moov = fragmented_moov(&[trex(&[1, 1, 1000, 10, 0])], &[]);
    cat(&[&FTYP, &moov, &moof(len + 8), &bx(b"mdat", &[7u8; 10])])
}

#[test]
fn fragments_of_other_tracks_are_skipped_but_their_data_is_not_followed() {
    // Flags this parser refuses, on another track, do not matter.
    assert_eq!(
        parse(&after_another_track(0x2_0000)).unwrap().samples.len(),
        1
    );
    // Without default-base-is-moof, the video's data would follow the other
    // track's, which is not tracked.
    assert_unsupported(&after_another_track(0), "inherited base offset");
}

fn sync_of(video: &Mp4Video) -> Vec<bool> {
    video.samples.iter().map(|s| s.sync).collect()
}

#[test]
fn keyframes_come_from_stss_as_ffmpeg_reads_it() {
    let stss = |numbers: &[u32]| {
        full(
            b"stss",
            0,
            0,
            &cat(&[&words(&[numbers.len() as u32]), &words(numbers)]),
        )
    };
    let sync = |extra: Vec<u8>| sync_of(&parse(&with(|t| t.extra = extra)).unwrap());
    // Without stss every sample is a keyframe.
    assert_eq!(sync_of(&parse(&plain()).unwrap()), [true; 5]);
    assert_eq!(sync(stss(&[1, 4])), [true, false, false, true, false]);
    // An empty one: none. Entries that name no sample are dropped.
    assert_eq!(sync(stss(&[])), [false; 5]);
    assert_eq!(sync(stss(&[0, 3, 9])), [false, false, true, false, false]);
    // A table given twice, or one its box cannot hold: none, and the file
    // still reads (it decoded before keyframes were read).
    assert_eq!(sync([stss(&[1]), stss(&[1])].concat()), [false; 5]);
    assert_eq!(sync(full(b"stss", 0, 0, &words(&[2, 1]))), [false; 5]);
}

/// Which samples of [`fragmented`] are keyframes, given the `trex` default
/// flags, the `tfhd` default flags, the `trun` first-sample flags and each
/// sample's own flags.
fn fragment_sync(
    trex: u32,
    tfhd: Option<u32>,
    first: Option<u32>,
    own: Option<[u32; 5]>,
) -> Vec<bool> {
    let moov = fragmented_moov(&[full(b"trex", 0, 0, &words(&[1, 1, 0, 0, trex]))], &[]);
    let moof = |data_offset: u32| {
        let (mut tfhd_flags, mut tfhd_body) = (0x2_0000, words(&[1]));
        if let Some(flags) = tfhd {
            tfhd_flags |= 0x20;
            tfhd_body.extend(words(&[flags]));
        }
        let tfhd = full(b"tfhd", 0, tfhd_flags, &tfhd_body);
        let tfdt = full(b"tfdt", 1, 0, &0u64.to_be_bytes());
        let mut trun_flags = 0x1 | 0x100 | 0x200 | 0x800;
        let mut run = words(&[5, data_offset]);
        if let Some(flags) = first {
            trun_flags |= 0x4;
            run.extend(words(&[flags]));
        }
        if own.is_some() {
            trun_flags |= 0x400;
        }
        for (i, (size, shift)) in SIZES.iter().zip(SHIFTS).enumerate() {
            // Duration, size, flags, composition offset.
            run.extend(words(&[DURATION, *size]));
            if let Some(own) = own {
                run.extend(words(&[own[i]]));
            }
            run.extend(words(&[shift as u32]));
        }
        let trun = full(b"trun", 0, trun_flags, &run);
        bx(
            b"moof",
            &cat(&[
                &full(b"mfhd", 0, 0, &words(&[1])),
                &bx(b"traf", &cat(&[&tfhd, &tfdt, &trun])),
            ]),
        )
    };
    let moof_len = moof(0).len() as u32;
    let payload = vec![7u8; SIZES.iter().sum::<u32>() as usize];
    let file = cat(&[&FTYP, &moov, &moof(moof_len + 8), &bx(b"mdat", &payload)]);
    sync_of(&parse(&file).unwrap())
}

#[test]
fn fragment_keyframes_come_from_sample_flags_as_ffmpeg_reads_them() {
    const NOT_SYNC: u32 = 0x1_0000;
    const DEPENDS: u32 = 0x100_0000;
    assert_eq!(fragment_sync(0, None, None, None), [true; 5]);
    assert_eq!(fragment_sync(NOT_SYNC, None, None, None), [false; 5]);
    assert_eq!(fragment_sync(0, Some(DEPENDS), None, None), [false; 5]);
    // tfhd's defaults replace trex's; the first-sample flags, the first's.
    assert_eq!(fragment_sync(NOT_SYNC, Some(0), None, None), [true; 5]);
    assert_eq!(
        fragment_sync(NOT_SYNC, None, Some(0), None),
        [true, false, false, false, false]
    );
    // A sample's own flags win over the first-sample flags.
    assert_eq!(
        fragment_sync(
            NOT_SYNC,
            None,
            Some(NOT_SYNC),
            Some([0, NOT_SYNC, DEPENDS, 0, NOT_SYNC])
        ),
        [true, false, false, true, false]
    );
}
