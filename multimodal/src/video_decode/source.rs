// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Where the bytes of an encoded video come from.

use std::io::{self, Read, Seek, SeekFrom};

/// Random access to an encoded video of known length.
///
/// The decoder reads the container's index through it, then each compressed
/// frame at the moment it is decoded, so a source backed by a file or an HTTP
/// range request never has to hold the whole video in memory. Reads are small
/// and frequent: a source with per-request cost (a network source) should
/// cache blocks itself; nothing here buffers for it.
pub trait ByteSource {
    /// Total size in bytes.
    fn len(&self) -> u64;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Fill `buf` with the bytes starting at `offset`. Callers keep
    /// `offset + buf.len() <= self.len()`; reading past the end is an error.
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<()>;
}

/// A video already in memory: what [`decode_sampled`](super::decode_sampled)
/// and [`probe`](super::probe) wrap their slice in.
impl ByteSource for &[u8] {
    fn len(&self) -> u64 {
        <[u8]>::len(self) as u64
    }

    fn read_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        usize::try_from(offset)
            .ok()
            .and_then(|start| self.get(start..start.checked_add(buf.len())?))
            .map(|bytes| buf.copy_from_slice(bytes))
            .ok_or_else(|| io::Error::from(io::ErrorKind::UnexpectedEof))
    }
}

impl ByteSource for Vec<u8> {
    fn len(&self) -> u64 {
        Vec::len(self) as u64
    }

    fn read_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        self.as_slice().read_at(offset, buf)
    }
}

/// `Read + Seek` over a [`ByteSource`], for the container parsers.
pub(super) struct SourceReader<'a> {
    source: &'a dyn ByteSource,
    pos: u64,
}

impl<'a> SourceReader<'a> {
    pub(super) fn new(source: &'a dyn ByteSource) -> Self {
        Self { source, pos: 0 }
    }
}

impl Read for SourceReader<'_> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let remaining = self.source.len().saturating_sub(self.pos);
        let n = usize::try_from(remaining).map_or(buf.len(), |r| r.min(buf.len()));
        if n > 0 {
            self.source.read_at(self.pos, &mut buf[..n])?;
            self.pos += n as u64;
        }
        Ok(n)
    }
}

impl Seek for SourceReader<'_> {
    fn seek(&mut self, to: SeekFrom) -> io::Result<u64> {
        let (base, delta) = match to {
            SeekFrom::Start(n) => (0, i128::from(n)),
            SeekFrom::End(d) => (i128::from(self.source.len()), i128::from(d)),
            SeekFrom::Current(d) => (i128::from(self.pos), i128::from(d)),
        };
        self.pos = u64::try_from(base + delta).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "seek to a negative or oversized position",
            )
        })?;
        Ok(self.pos)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slices_read_at_any_offset_and_refuse_past_the_end() {
        let data = [1u8, 2, 3, 4, 5];
        let source = &data[..];
        let mut buf = [0u8; 2];
        source.read_at(3, &mut buf).unwrap();
        assert_eq!(buf, [4, 5]);
        assert_eq!(
            source.read_at(4, &mut buf).unwrap_err().kind(),
            io::ErrorKind::UnexpectedEof
        );
        assert_eq!(
            source.read_at(u64::MAX, &mut buf).unwrap_err().kind(),
            io::ErrorKind::UnexpectedEof
        );
        source.read_at(5, &mut []).unwrap();
        assert_eq!(ByteSource::len(&source), 5);
        assert!(!source.is_empty());
    }

    #[test]
    fn the_reader_behaves_like_a_cursor() {
        let data = (0u8..10).collect::<Vec<_>>();
        let source = data.as_slice();
        let mut reader = SourceReader::new(&source);
        let mut cursor = io::Cursor::new(data.as_slice());
        let mut a = [0u8; 4];
        let mut b = [0u8; 4];
        for to in [
            SeekFrom::Start(2),
            SeekFrom::Current(3),
            SeekFrom::End(-3),
            SeekFrom::Current(-1),
            SeekFrom::Start(20),
        ] {
            assert_eq!(reader.seek(to).unwrap(), cursor.seek(to).unwrap());
            assert_eq!(reader.read(&mut a).unwrap(), cursor.read(&mut b).unwrap());
            assert_eq!(a, b);
        }
        // Past the end reads nothing; before the start is an error.
        assert_eq!(reader.read(&mut a).unwrap(), 0);
        assert!(reader.seek(SeekFrom::Current(-1000)).is_err());
    }
}
