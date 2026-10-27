// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Where the text of a stream chunk came from.
//!
//! A caller that feeds raw model text through the stream stages can ask each
//! stage to report, for every text byte it emits, the input byte it was copied
//! from. The ranges are in the caller's own offsets over the text it fed. They
//! date bytes; they never reorder them. A chunk type opts in through
//! [`SourceSlot`]. The defaults carry nothing, so a chunk type that does not
//! override them records nothing and costs nothing.

use std::ops::Range;

/// Input byte ranges, in the caller's text stream, for one choice of one chunk.
///
/// The ranges of `content` tile that choice's `content` text in order, and the
/// ranges of `reasoning` tile its `reasoning_content`. Each range's input bytes
/// equal the text bytes it stands for.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ChoiceSources {
    pub index: u32,
    pub content: Vec<Range<usize>>,
    pub reasoning: Vec<Range<usize>>,
}

/// A chunk that can carry the sources of its text. The defaults carry none.
pub trait SourceSlot {
    /// The sources of this chunk's text, or `None` when this chunk type records none.
    fn sources(&self) -> Option<&[ChoiceSources]> {
        None
    }

    /// The same, for in-place rewrite.
    fn sources_mut(&mut self) -> Option<&mut Vec<ChoiceSources>> {
        None
    }
}
