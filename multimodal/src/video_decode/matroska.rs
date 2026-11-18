// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What `matroska-demuxer` (0.8.1) cannot be trusted with, checked before it
//! opens a WebM / Matroska file.
//!
//! Opening a file, the demuxer reads the elements it indexes (the EBML
//! header, SeekHeads, Info, Tracks, Cues, Chapters and Tags) whole into
//! element tables, which in measurements took up to 76 bytes of memory for
//! each byte of them. It does not hold a child to its parent's end, so a child
//! can run on past it and be read too. It follows a SeekHead's reference to
//! another SeekHead with no record of those it has read: a SeekHead that leads
//! back to one already read loops forever, and one with two such references
//! panics. It skips a CRC-32 element at the start of the segment by reading
//! the same element again, forever. It adds the segment's offset to each Seek
//! and Cue position unchecked, a BlockGroup's size to its position, and a
//! block's timestamp to its cluster's (and takes the absolute value of an
//! `i16`), which panic where overflow is checked.
//!
//! So rather than follow the demuxer's leniencies, this holds what it reads to
//! the format. Every element it indexes, and the EBML header, must be a tree
//! of known-size elements, each inside its parent, at every depth: then the
//! demuxer reads exactly those bytes, and they are counted against
//! [`MAX_TABLE_BYTES`]. A Seek must point at an element, not at bytes the
//! demuxer would skip to find one. Also refused are a loop of SeekHeads, a
//! SeekHead that references two, a segment that starts with a CRC-32 element,
//! positions and timestamps that overflow, a BlockGroup of unknown size and a
//! fixed-size lace that does not divide its block (the demuxer would resume
//! inside the block). Elements of a kind it indexes are checked wherever it
//! may find them, so some files are refused for an element it would not have
//! read. Where the demuxer would fail on its own, the check stops and leaves
//! the error to it.
//!
//! What the demuxer reads outside these elements it reads a frame or an
//! element at a time, each no larger than the file.

use std::collections::HashSet;

use super::{ByteSource, read_error};
use crate::{MmError, Result};

/// Most bytes of the elements the demuxer reads into tables when it opens a
/// file. Its tables were measured at up to 76 bytes of memory for each of
/// these bytes (a Tags of empty Tag elements, the worst of the kinds tried;
/// about 80 MiB at this limit). In files from common muxers the Cues are the
/// largest, about 20 bytes a cue: 1 MiB holds about 50,000 cues, over 14 hours
/// at one a second.
const MAX_TABLE_BYTES: u64 = 1 << 20;

const EBML: u32 = 0x1A45_DFA3;
const SEGMENT: u32 = 0x1853_8067;
const SEEK_HEAD: u32 = 0x114D_9B74;
const SEEK: u32 = 0x4DBB;
const SEEK_ID: u32 = 0x53AB;
const SEEK_POSITION: u32 = 0x53AC;
const CRC_32: u32 = 0xBF;
const CLUSTER: u32 = 0x1F43_B675;
const BLOCK_GROUP: u32 = 0xA0;
const BLOCK: u32 = 0xA1;
const SIMPLE_BLOCK: u32 = 0xA3;
const CUE_CLUSTER_POSITION: u32 = 0xF1;
const TIMESTAMP: u32 = 0xE7;
/// The elements, besides the EBML header and SeekHeads, the demuxer indexes
/// whole: Info, Tracks, Cues, Chapters and Tags.
const TABLES: [u32; 5] = [
    0x1549_A966,
    0x1654_AE6B,
    0x1C53_BB6B,
    0x1043_A770,
    0x1254_C367,
];
/// The elements the demuxer's ID table types as masters (`element_id.rs`).
const MASTERS: [u32; 31] = [
    EBML,
    SEGMENT,
    SEEK_HEAD,
    SEEK,
    0x1549_A966,
    CLUSTER,
    BLOCK_GROUP,
    0x75A1,
    0xA6,
    0x1654_AE6B,
    0xAE,
    0xE0,
    0xE1,
    0x6D80,
    0x6240,
    0x5034,
    0x5035,
    0x47E7,
    0x55B0,
    0x55D0,
    0x1C53_BB6B,
    0xBB,
    0xB7,
    0x1043_A770,
    0x45B9,
    0xB6,
    0x80,
    0x1254_C367,
    0x7373,
    0x63C0,
    0x67C8,
];

/// An element header.
#[derive(Clone, Copy)]
struct Element {
    id: u32,
    /// Where its data starts.
    data: u64,
    /// Its data size; `None` when unknown.
    size: Option<u64>,
}

impl Element {
    /// Where the element after it starts, as the demuxer moves on: past the
    /// data, or into it when the size is unknown.
    fn next(&self) -> u64 {
        self.data.saturating_add(self.size.unwrap_or(0))
    }
}

fn malformed(what: &str) -> MmError {
    MmError::invalid_input(format!("malformed WebM / Matroska: {what}"))
}

/// Refuse what `matroska-demuxer` would loop on, panic on or read without
/// bound in the WebM / Matroska file in `source`. Anything else it would
/// refuse on its own is left to it.
pub(super) fn check(source: &dyn ByteSource) -> Result<()> {
    let mut walk = Walk {
        source,
        len: source.len(),
        tables: 0,
    };
    let Some(ebml) = walk.element(0)? else {
        return Ok(());
    };
    if ebml.id != EBML {
        return Ok(());
    }
    // Its children end where it does, so the demuxer looks for the segment
    // right after it.
    walk.tree(&ebml, None)?;
    let Some(segment) = walk.element(ebml.next())? else {
        return Ok(());
    };
    if segment.id != SEGMENT {
        return Ok(());
    }
    let base = segment.data;

    // The SeekHeads, from the one the segment starts with, each followed
    // once, and the elements their Seeks point at.
    let mut next = match walk.element(base)? {
        Some(first) if first.id == CRC_32 => {
            return Err(MmError::unsupported(
                "WebM / Matroska segment that starts with a CRC-32 element",
            ));
        }
        Some(first) if first.id == SEEK_HEAD => Some(first),
        _ => None,
    };
    let mut seen = HashSet::new();
    let mut indexes_any = false;
    let mut cluster = None;
    while let Some(seek_head) = next.take() {
        if !seen.insert(seek_head.data) {
            return Err(malformed("SeekHeads that lead back to one already read"));
        }
        walk.tree(&seek_head, Some(base))?;
        let mut at = seek_head.data;
        while at < seek_head.next() {
            // Inside the tree just checked.
            let Some(seek) = walk.element(at)? else { break };
            at = seek.next();
            if seek.id != SEEK {
                continue;
            }
            let Some(entry) = walk.seek_entry(&seek)? else {
                // The demuxer fails on this Seek.
                return Ok(());
            };
            let Some((id, position)) = entry else {
                continue;
            };
            let target = base
                .checked_add(position)
                .ok_or_else(|| malformed("Seek position out of range"))?;
            if id != SEEK_HEAD {
                indexes_any = true;
                if id == CLUSTER {
                    cluster = Some(target);
                }
                walk.table(id, target, base)?;
                continue;
            }
            match walk.element_at(target)? {
                Some(found) if found.id == SEEK_HEAD => {
                    if next.replace(found).is_some() {
                        return Err(malformed("SeekHead that references two SeekHeads"));
                    }
                }
                // The demuxer fails on a reference to anything else.
                _ => return Ok(()),
            }
        }
    }
    // Without a Seek the demuxer indexes the segment's top-level elements as
    // it finds them, and without one to the first cluster it finds that too.
    if !indexes_any || cluster.is_none() {
        let mut at = base;
        while let Some(element) = walk.element(at)? {
            if !indexes_any && TABLES.contains(&element.id) {
                walk.tree(&element, Some(base))?;
            }
            if element.id == CLUSTER && cluster.is_none() {
                cluster = Some(at);
            }
            at = element.next();
        }
    }
    match cluster {
        Some(cluster) => walk.blocks(cluster),
        None => Ok(()),
    }
}

struct Walk<'a> {
    source: &'a dyn ByteSource,
    len: u64,
    /// Bytes of the trees the demuxer reads into tables, so far.
    tables: u64,
}

impl Walk<'_> {
    fn read(&self, at: u64, buf: &mut [u8]) -> Result<()> {
        self.source
            .read_at(at, buf)
            .map_err(|e| read_error(e, "a WebM / Matroska element"))
    }

    /// The element header at `at`, read as the demuxer reads it (bytes whose
    /// top four bits are clear are skipped before an ID; only a one-byte size
    /// of all ones is unknown); `None` where none can be read.
    fn element(&self, mut at: u64) -> Result<Option<Element>> {
        let mut head = [0u8; 12];
        loop {
            let n = self.len.saturating_sub(at).min(head.len() as u64) as usize;
            if n == 0 {
                return Ok(None);
            }
            let head = &mut head[..n];
            self.read(at, head)?;
            let Some(skip) = head.iter().position(|&b| b & 0xF0 != 0) else {
                at += n as u64;
                continue;
            };
            if skip > 0 {
                at += skip as u64;
                continue;
            }
            let id_len = head[0].leading_zeros() as usize + 1;
            let Some(id) = head.get(..id_len) else {
                return Ok(None);
            };
            let id = id.iter().fold(0u32, |v, &b| (v << 8) | u32::from(b));
            let Some(&first) = head.get(id_len) else {
                return Ok(None);
            };
            let size_len = first.leading_zeros() as usize + 1;
            if size_len > 8 {
                return Ok(None);
            }
            let Some(size) = head.get(id_len..id_len + size_len) else {
                return Ok(None);
            };
            let mask = if size_len == 8 { 0 } else { 0xFF >> size_len };
            let size = size[1..]
                .iter()
                .fold(u64::from(first & mask), |v, &b| (v << 8) | u64::from(b));
            return Ok(Some(Element {
                id,
                data: at + (id_len + size_len) as u64,
                size: (first != 0xFF).then_some(size),
            }));
        }
    }

    /// Hold `root` to the format and count it: known sizes, every child
    /// inside its parent, at every depth, and every Cue position addable to
    /// `base`. The walk is not recursive, so nesting cannot exhaust the stack.
    fn tree(&mut self, root: &Element, base: Option<u64>) -> Result<()> {
        let end = root
            .size
            .and_then(|size| root.data.checked_add(size))
            .filter(|&end| end <= self.len)
            .ok_or_else(|| malformed("an indexed element of unknown size or past the end"))?;
        self.tables = self.tables.saturating_add(end - root.data);
        if self.tables > MAX_TABLE_BYTES {
            return Err(MmError::limit_exceeded(format!(
                "WebM / Matroska index elements exceed the {MAX_TABLE_BYTES} byte limit"
            )));
        }
        let mut ends = vec![end];
        let mut at = root.data;
        while let Some(&end) = ends.last() {
            if at >= end {
                ends.pop();
                continue;
            }
            let child = self
                .element(at)?
                .ok_or_else(|| malformed("an unreadable element"))?;
            let child_end = child
                .size
                .and_then(|size| child.data.checked_add(size))
                .filter(|&child_end| child_end <= end)
                .ok_or_else(|| malformed("an element that runs past its parent"))?;
            if child.id == CUE_CLUSTER_POSITION
                && let Some(base) = base
                && self
                    .unsigned(&child)?
                    .is_some_and(|p| p.checked_add(base).is_none())
            {
                return Err(malformed("Cue position out of range"));
            }
            if MASTERS.contains(&child.id) {
                ends.push(child_end);
                at = child.data;
            } else {
                at = child_end;
            }
        }
        Ok(())
    }

    /// The element of kind `id` a Seek points the demuxer at, if it is one.
    fn table(&mut self, id: u32, at: u64, base: u64) -> Result<()> {
        if !TABLES.contains(&id) {
            return Ok(());
        }
        match self.element_at(at)? {
            Some(element) if element.id == id => self.tree(&element, Some(base)),
            // The demuxer fails on anything else there.
            _ => Ok(()),
        }
    }

    /// An unsigned integer element's value; `None` when the demuxer cannot
    /// read one (more than 8 bytes, or past the end).
    fn unsigned(&self, element: &Element) -> Result<Option<u64>> {
        let Some(size) = element.size.filter(|&s| s <= 8 && s <= self.len) else {
            return Ok(None);
        };
        if element.data > self.len - size {
            return Ok(None);
        }
        let mut bytes = [0u8; 8];
        let bytes = &mut bytes[..size as usize];
        self.read(element.data, bytes)?;
        Ok(Some(bytes.iter().fold(0, |v, &b| (v << 8) | u64::from(b))))
    }

    /// A Seek's target as the demuxer reads it: `None` when it fails on the
    /// Seek, `Some(None)` when it skips it (no ID or position, or an ID too
    /// long), else the ID and position. The first of each field counts.
    #[allow(clippy::option_option)]
    fn seek_entry(&self, seek: &Element) -> Result<Option<Option<(u32, u64)>>> {
        let (mut id, mut position) = (None, None);
        let mut at = seek.data;
        // Inside a SeekHead held to the format.
        while at < seek.next() {
            let Some(field) = self.element(at)? else {
                break;
            };
            at = field.next();
            let slot = match field.id {
                SEEK_ID => &mut id,
                SEEK_POSITION => &mut position,
                _ => continue,
            };
            let Some(value) = self.unsigned(&field)? else {
                return Ok(None);
            };
            slot.get_or_insert(value);
        }
        Ok(Some(id.zip(position).and_then(|(id, position)| {
            u32::try_from(id).ok().map(|id| (id, position))
        })))
    }

    /// The element at `at`, which a Seek points at: it must start there, not
    /// after bytes the demuxer would skip (and each Seek would scan again).
    fn element_at(&self, at: u64) -> Result<Option<Element>> {
        let mut first = [0u8];
        if at >= self.len {
            return Ok(None);
        }
        self.read(at, &mut first)?;
        if first[0] & 0xF0 == 0 {
            return Err(malformed("Seek that points at no element"));
        }
        self.element(at)
    }

    /// Walk the blocks from the first cluster as the demuxer reads frames:
    /// into clusters, BlockGroups and elements of unknown size, past the rest,
    /// keeping the cluster timestamp as it does. Refuse a BlockGroup of
    /// unknown size, and blocks it cannot step past or time.
    fn blocks(&self, mut at: u64) -> Result<()> {
        let mut cluster_time = 0;
        while let Some(element) = self.element(at)? {
            match (element.id, element.size) {
                (BLOCK_GROUP, None) => return Err(malformed("BlockGroup of unknown size")),
                (CLUSTER | BLOCK_GROUP, _) | (_, None) => at = element.data,
                (SIMPLE_BLOCK | BLOCK, Some(size)) => {
                    self.block(element.data, size, cluster_time)?;
                    at = element.next();
                }
                (TIMESTAMP, Some(_)) => {
                    cluster_time = self.unsigned(&element)?.unwrap_or(cluster_time);
                    at = element.next();
                }
                _ => at = element.next(),
            }
        }
        Ok(())
    }

    /// Refuse the block at `data` of `size` bytes if its timestamp overflows
    /// against `cluster_time`, or it is a fixed-size lace that does not
    /// divide its data (the demuxer would resume inside the block). A header
    /// that does not read is the demuxer's to refuse.
    fn block(&self, data: u64, size: u64, cluster_time: u64) -> Result<()> {
        // Track number (a size-like VINT of up to 8 bytes), a 16-bit signed
        // timestamp, flags and, when laced, the frame count.
        let mut head = [0u8; 12];
        let n = self.len.saturating_sub(data).min(12) as usize;
        let head = &mut head[..n];
        self.read(data, head)?;
        let Some(&first) = head.first() else {
            return Ok(());
        };
        let track_len = first.leading_zeros() as usize + 1;
        let Some(&[high, low]) = head.get(track_len..track_len + 2) else {
            return Ok(());
        };
        let time = i16::from_be_bytes([high, low]);
        let overflows = match u64::try_from(time) {
            Ok(after) => cluster_time.checked_add(after).is_none(),
            Err(_) => time == i16::MIN,
        };
        if overflows {
            return Err(malformed("block timestamp out of range"));
        }
        let (Some(&flags), Some(&count)) = (head.get(track_len + 2), head.get(track_len + 3))
        else {
            return Ok(());
        };
        let frames = u64::from(count) + 1;
        let data_size = size.saturating_sub(track_len as u64 + 4);
        if (flags >> 1) & 0b11 == 0b10 && !data_size.is_multiple_of(frames) {
            return Err(malformed("fixed-size lace that does not divide its block"));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
