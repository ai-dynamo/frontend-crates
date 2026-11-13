// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Where the bytes of an encoded video come from.

use std::cell::RefCell;
use std::io::{self, Read, Seek, SeekFrom};
use std::rc::Rc;

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
    failure: ReadFailure,
}

/// The first failed read of a [`SourceReader`], kept outside it: a parser
/// that owns the reader may take a failed read for the end of the file
/// (`matroska-demuxer` does, on an element header), so the caller checks
/// this after parsing rather than trusting the parser to report it.
#[derive(Clone, Default)]
pub(super) struct ReadFailure(Rc<RefCell<Option<io::Error>>>);

impl ReadFailure {
    /// The first failed read, if any, once.
    pub(super) fn take(&self) -> Option<io::Error> {
        self.0.borrow_mut().take()
    }
}

impl<'a> SourceReader<'a> {
    pub(super) fn new(source: &'a dyn ByteSource) -> (Self, ReadFailure) {
        let failure = ReadFailure::default();
        let reader = Self {
            source,
            pos: 0,
            failure: failure.clone(),
        };
        (reader, failure)
    }
}

impl Read for SourceReader<'_> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let remaining = self.source.len().saturating_sub(self.pos);
        let n = usize::try_from(remaining).map_or(buf.len(), |r| r.min(buf.len()));
        if n > 0 {
            if let Err(e) = self.source.read_at(self.pos, &mut buf[..n]) {
                // `Interrupted` is not a failure: `read_exact` retries it.
                // `io::Error` is not `Clone`: keep an equivalent one.
                if e.kind() != io::ErrorKind::Interrupted {
                    self.failure
                        .0
                        .borrow_mut()
                        .get_or_insert_with(|| io::Error::new(e.kind(), e.to_string()));
                }
                return Err(e);
            }
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
        let (mut reader, failure) = SourceReader::new(&source);
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
        assert!(failure.take().is_none());
    }

    #[test]
    fn a_failed_read_is_remembered_for_the_caller() {
        struct Failing;
        impl ByteSource for Failing {
            fn len(&self) -> u64 {
                10
            }
            fn read_at(&self, _: u64, _: &mut [u8]) -> io::Result<()> {
                Err(io::Error::other("disk on fire"))
            }
        }
        let (mut reader, failure) = SourceReader::new(&Failing);
        let mut buf = [0u8; 4];
        let err = reader.read(&mut buf).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::Other);
        let kept = failure.take().unwrap();
        assert_eq!(
            (kept.kind(), kept.to_string()),
            (err.kind(), err.to_string())
        );
        assert!(failure.take().is_none());
        // An interrupted read is retried by `read_exact`, not remembered.
        struct InterruptsOnce(std::cell::Cell<bool>);
        impl ByteSource for InterruptsOnce {
            fn len(&self) -> u64 {
                4
            }
            fn read_at(&self, _: u64, buf: &mut [u8]) -> io::Result<()> {
                if self.0.replace(false) {
                    return Err(io::Error::from(io::ErrorKind::Interrupted));
                }
                buf.fill(7);
                Ok(())
            }
        }
        let source = InterruptsOnce(std::cell::Cell::new(true));
        let (mut reader, failure) = SourceReader::new(&source);
        reader.read_exact(&mut buf).unwrap();
        assert_eq!(buf, [7; 4]);
        assert!(failure.take().is_none());
    }
}
