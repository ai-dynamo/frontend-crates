// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

const INFO: u32 = TABLES[0];
const CUES: u32 = TABLES[2];
const CHAPTERS: u32 = TABLES[3];
const TAGS: u32 = TABLES[4];
const TAG: u32 = 0x7373;
const CUE_POINT: u32 = 0xBB;
const CUE_TRACK_POSITIONS: u32 = 0xB7;
const CHAPTER_ATOM: u32 = 0xB6;
const VOID: u32 = 0xEC;

/// An element with an 8-byte size.
fn el(id: u32, body: &[u8]) -> Vec<u8> {
    let id = id.to_be_bytes();
    let skip = id.iter().position(|&b| b != 0).unwrap();
    let mut out = id[skip..].to_vec();
    out.push(0x01);
    out.extend_from_slice(&(body.len() as u64).to_be_bytes()[1..]);
    out.extend_from_slice(body);
    out
}

/// An element of unknown size.
fn open_el(id: u32, body: &[u8]) -> Vec<u8> {
    let id = id.to_be_bytes();
    let skip = id.iter().position(|&b| b != 0).unwrap();
    [&id[skip..], &[0xFF], body].concat()
}

fn uint(id: u32, value: u64) -> Vec<u8> {
    el(id, &value.to_be_bytes())
}

fn seek(id: u32, position: u64) -> Vec<u8> {
    el(
        SEEK,
        &[uint(SEEK_ID, id.into()), uint(SEEK_POSITION, position)].concat(),
    )
}

fn seek_head(seeks: &[Vec<u8>]) -> Vec<u8> {
    el(SEEK_HEAD, &seeks.concat())
}

/// A Tags element of `bytes` (at least 9) bytes of data: one Void.
fn tags(bytes: u64) -> Vec<u8> {
    el(TAGS, &el(VOID, &vec![0; bytes as usize - 9]))
}

/// The EBML header's data: DocType, DocTypeVersion, DocTypeReadVersion, as
/// the demuxer requires.
fn ebml_fields() -> Vec<u8> {
    [el(0x4282, b"webm"), uint(0x4287, 4), uint(0x4285, 2)].concat()
}

const EBML_DATA: u64 = 14 + 18 + 18;
/// Where the segment's data starts.
const BASE: u64 = 12 + EBML_DATA + 5;

/// A file whose segment, of unknown size, holds `children`.
fn file(children: &[Vec<u8>]) -> Vec<u8> {
    let ebml = el(EBML, &ebml_fields());
    let bytes = [
        &ebml[..],
        &SEGMENT.to_be_bytes(),
        &[0xFF],
        &children.concat(),
    ]
    .concat();
    assert_eq!(bytes.len() as u64, BASE + children.concat().len() as u64);
    bytes
}

fn run(bytes: &[u8]) -> Result<()> {
    check(&bytes)
}

fn assert_invalid(bytes: &[u8], why: &str) {
    let r = run(bytes);
    assert!(
        matches!(r, Err(MmError::InvalidInput { .. })),
        "{why}: {r:?}"
    );
}

fn assert_limit(bytes: &[u8], why: &str) {
    let r = run(bytes);
    assert!(
        matches!(r, Err(MmError::LimitExceeded { .. })),
        "{why}: {r:?}"
    );
}

#[test]
fn seek_heads_are_followed_once_each() {
    // A chain: the first SeekHead points at a second, which indexes Info.
    let probe = seek_head(&[seek(SEEK_HEAD, 0)]);
    let to_second = probe.len() as u64;
    let first = seek_head(&[seek(SEEK_HEAD, to_second)]);
    assert_eq!(first.len(), probe.len());
    let second = seek_head(&[seek(INFO, 0)]);
    run(&file(&[first.clone(), second])).unwrap();
    // The second leads back to the first, or a SeekHead to itself.
    assert_invalid(
        &file(&[first.clone(), seek_head(&[seek(SEEK_HEAD, 0)])]),
        "a loop of two",
    );
    assert_invalid(&file(&[seek_head(&[seek(SEEK_HEAD, 0)])]), "a loop of one");
    // One SeekHead that references two.
    let two = |at: u64| seek_head(&[seek(SEEK_HEAD, at), seek(SEEK_HEAD, at + 100)]);
    let to_next = two(0).len() as u64;
    let mut bytes = file(&[two(to_next), seek_head(&[])]);
    bytes.resize(bytes.len() + 100 - seek_head(&[]).len(), 0);
    bytes.extend_from_slice(&seek_head(&[]));
    assert_invalid(&bytes, "two references");
    // A reference to something else is the demuxer's to refuse.
    run(&file(&[seek_head(&[seek(SEEK_HEAD, 1000)])])).unwrap();
}

#[test]
fn a_segment_that_starts_with_a_crc_32_element_is_unsupported() {
    let r = run(&file(&[el(CRC_32, &[0; 4]), seek_head(&[])]));
    assert!(matches!(r, Err(MmError::Unsupported { .. })), "{r:?}");
    // Anywhere else, the demuxer reads past it.
    run(&file(&[seek_head(&[]), el(CRC_32, &[0; 4])])).unwrap();
}

#[test]
fn positions_the_demuxer_cannot_add_to_the_segment_offset_are_refused() {
    let last = u64::MAX - BASE;
    run(&file(&[seek_head(&[seek(INFO, last)])])).unwrap();
    assert_invalid(&file(&[seek_head(&[seek(INFO, last + 1)])]), "Seek");
    let cues = |position: u64| {
        let fields = uint(CUE_CLUSTER_POSITION, position);
        el(CUES, &el(CUE_POINT, &el(CUE_TRACK_POSITIONS, &fields)))
    };
    run(&file(&[cues(last)])).unwrap();
    assert_invalid(&file(&[cues(last + 1)]), "Cue");
}

#[test]
fn index_elements_are_bounded_wherever_the_demuxer_reads_them() {
    // With no Seek, every top-level element of a kind it indexes counts,
    // after the EBML header; bytes with clear top bits before an ID are
    // skipped, as the demuxer skips them.
    let at_cap = MAX_TABLE_BYTES - EBML_DATA;
    run(&file(&[vec![0; 3], tags(at_cap)])).unwrap();
    assert_limit(&file(&[vec![0; 3], tags(at_cap + 1)]), "top level");
    assert_limit(&file(&[tags(at_cap / 2), tags(at_cap / 2 + 1)]), "two Tags");
    // With a Seek, what it points at counts, even inside another element...
    let probe = seek_head(&[seek(TAGS, 0)]);
    let inside_void = probe.len() as u64 + 9;
    let pointing = seek_head(&[seek(TAGS, inside_void)]);
    let big = tags(at_cap + 1);
    assert_limit(&file(&[pointing, el(VOID, &big)]), "inside a Void");
    // ...and nothing it does not point at.
    run(&file(&[seek_head(&[seek(INFO, 0)]), big])).unwrap();
}

#[test]
fn what_the_demuxer_indexes_must_nest_inside_its_declared_size() {
    // An EBML header must have a known size, and the segment comes right
    // after it: an unknown-size child would stop the demuxer's walk early
    // and it would look for the segment there instead. This is the case a
    // review found to reach the CRC-32 loop past an earlier version.
    let header = [&ebml_fields()[..], &open_el(VOID, &[])].concat();
    let crc = el(CRC_32, &[0; 4]);
    let bytes = [&open_el(EBML, &header)[..], &open_el(SEGMENT, &crc)].concat();
    assert_invalid(&bytes, "EBML header of unknown size");
    let bytes = [&el(EBML, &header)[..], &open_el(SEGMENT, &crc)].concat();
    assert_invalid(&bytes, "unknown-size child");
    // A child that runs past its parent, which the demuxer would follow.
    let mut ebml = el(EBML, &[&ebml_fields()[..], &el(VOID, &[0; 8])].concat());
    // The Void's 8-byte size, before its 8 bytes of data: 108 instead of 8.
    let void_size = ebml.len() - 16;
    ebml[void_size..void_size + 8].copy_from_slice(&[0x01, 0, 0, 0, 0, 0, 0, 108]);
    assert_invalid(&[ebml, vec![0; 100]].concat(), "EBML child past its end");
    // The same in a SeekHead (a Seek covering the rest of the file), and a
    // Tag that covers more than its Tags, at any depth.
    let mut sh = seek_head(&[seek(INFO, 0)]);
    sh[12 + 2] = 0x01;
    sh[12 + 3..12 + 10].copy_from_slice(&[0, 0, 0, 0, 0, 0x10, 0]);
    assert_invalid(
        &file(&[sh, [0xEC, 0x80].repeat(4096)]),
        "Seek past its SeekHead",
    );
    let mut tags = el(TAGS, &el(TAG, &[]));
    let tag_size = tags.len() - 7;
    tags[tag_size..].copy_from_slice(&[0, 0, 0, 0, 0, 0, 200]);
    assert_invalid(&file(&[tags, vec![0; 200]]), "Tag past its Tags");
    // Zero bytes are skipped before an ID: inside an element, past its end.
    let tags = el(TAGS, &[&el(TAG, &[])[..], &[0, 0]].concat());
    assert_invalid(&file(&[tags, el(TAG, &[])]), "padding inside Tags");
}

#[test]
fn deep_nesting_is_walked_without_recursion() {
    // Chapter atoms 40,000 deep, under the cap: no stack is used per level.
    // Each header is 9 bytes, so the atom k levels from the bottom holds 9k.
    let depth = 40_000u64;
    let atoms: Vec<u8> = (0..depth)
        .rev()
        .flat_map(|k| [CHAPTER_ATOM as u8].into_iter().chain(size8(9 * k)))
        .collect();
    run(&file(&[el(CHAPTERS, &atoms)])).unwrap();
}

/// An 8-byte EBML size.
fn size8(size: u64) -> [u8; 8] {
    let mut out = size.to_be_bytes();
    out[0] = 0x01;
    out
}

#[test]
fn a_seek_must_point_at_an_element() {
    // The demuxer skips bytes with clear top bits to find one; each Seek
    // would scan them again.
    let probe = seek_head(&[seek(TAGS, 0)]);
    let at = probe.len() as u64;
    let tags = el(TAGS, &el(TAG, &[]));
    run(&file(&[seek_head(&[seek(TAGS, at)]), tags.clone()])).unwrap();
    assert_invalid(
        &file(&[seek_head(&[seek(TAGS, at)]), vec![0; 64], tags]),
        "Seek into padding",
    );
}

#[test]
fn block_timestamps_the_demuxer_cannot_add_are_refused() {
    let block = |time: i16| {
        el(
            SIMPLE_BLOCK,
            &[&[0x81][..], &time.to_be_bytes(), &[0x80, 1]].concat(),
        )
    };
    let cluster = |time: u64, block_time: i16| {
        open_el(
            CLUSTER,
            &[uint(TIMESTAMP, time), block(block_time)].concat(),
        )
    };
    run(&file(&[cluster(0, i16::MIN + 1)])).unwrap();
    assert_invalid(&file(&[cluster(0, i16::MIN)]), "abs of -32768");
    run(&file(&[cluster(u64::MAX, 0)])).unwrap();
    run(&file(&[cluster(u64::MAX, -1)])).unwrap();
    assert_invalid(&file(&[cluster(u64::MAX, 1)]), "past u64::MAX");
    run(&file(&[cluster(u64::MAX - 1, 1)])).unwrap();
}

#[test]
fn blocks_the_demuxer_cannot_step_past_are_refused() {
    // A SimpleBlock of track 1: timestamp 0, flags, then the data.
    let block =
        |flags: u8, rest: &[u8]| el(SIMPLE_BLOCK, &[&[0x81, 0, 0, flags][..], rest].concat());
    let cluster = |blocks: &[Vec<u8>]| open_el(CLUSTER, &blocks.concat());
    run(&file(&[cluster(&[block(0x80, &[1, 2, 3])])])).unwrap();
    // Fixed-size lacing: 3 frames of 2 bytes, or 7 bytes that do not divide.
    run(&file(&[cluster(&[block(0x84, &[2, 0, 0, 0, 0, 0, 0])])])).unwrap();
    assert_invalid(
        &file(&[cluster(&[block(0x84, &[2, 0, 0, 0, 0, 0, 0, 0])])]),
        "fixed lace with a remainder",
    );
    // A BlockGroup of unknown size, in a cluster the demuxer reaches.
    let group = [open_el(BLOCK_GROUP, &el(BLOCK, &[0x81, 0, 0, 0, 1]))];
    assert_invalid(&file(&[cluster(&group)]), "BlockGroup of unknown size");
    // Through a Seek to the cluster, as well as without one.
    let probe = seek_head(&[seek(CLUSTER, 0)]);
    let at = probe.len() as u64;
    let bytes = file(&[seek_head(&[seek(CLUSTER, at)]), cluster(&group)]);
    assert_invalid(&bytes, "through a Seek");
}
