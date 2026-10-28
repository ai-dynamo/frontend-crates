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

/// The text of one choice of a chunk, as [`match_choice_sources`] needs it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChoiceText {
    pub index: u32,
    /// Byte length of the choice's text `content` (0 for none or parts).
    pub content_len: usize,
    /// Byte length of its `reasoning_content`.
    pub reasoning_len: usize,
}

impl ChoiceText {
    fn has_text(&self) -> bool {
        self.content_len + self.reasoning_len > 0
    }
}

/// Matches a chunk's source entries to its choices, in choice order.
///
/// Entries pair, in order, with the choices that share their index: with all
/// of those choices when the counts are equal, or else with those that have
/// text when that count is equal. Any other count is ambiguous and pairs
/// nothing. An entry whose ranges do not cover its choice's text lengths
/// exactly is not used. A choice gets `None` when no entry pairs with it. A
/// choice with text and `None` has no known source, and a stage treats it as
/// unsourced from then on.
pub fn match_choice_sources(
    entries: Vec<ChoiceSources>,
    choices: &[ChoiceText],
) -> Vec<Option<ChoiceSources>> {
    let mut out: Vec<Option<ChoiceSources>> = vec![None; choices.len()];
    let mut by_index: std::collections::BTreeMap<u32, std::collections::VecDeque<ChoiceSources>> =
        std::collections::BTreeMap::new();
    for entry in entries {
        by_index.entry(entry.index).or_default().push_back(entry);
    }
    for (index, mut group) in by_index {
        let members: Vec<usize> = (0..choices.len())
            .filter(|&at| choices[at].index == index)
            .collect();
        let with_text: Vec<usize> = members
            .iter()
            .copied()
            .filter(|&at| choices[at].has_text())
            .collect();
        let paired = if group.len() == members.len() {
            members
        } else if group.len() == with_text.len() {
            with_text
        } else {
            continue;
        };
        for at in paired {
            let entry = group.pop_front().expect("one entry per paired choice");
            let len =
                |ranges: &[Range<usize>]| ranges.iter().map(|range| range.len()).sum::<usize>();
            if len(&entry.content) == choices[at].content_len
                && len(&entry.reasoning) == choices[at].reasoning_len
            {
                out[at] = Some(entry);
            }
        }
    }
    out
}

/// Which field of a reasoning parser's result a run of text went to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SourceChannel {
    /// [`ParserResult::normal_text`](crate::ParserResult::normal_text).
    Normal,
    /// [`ParserResult::reasoning_text`](crate::ParserResult::reasoning_text).
    Reasoning,
}

/// One run of returned text and the input bytes it was copied from.
///
/// A reasoning parser returns its two fields without their relative order, and
/// one call can hold several transitions: `A<think>B</think>C` returns normal
/// `AC` and reasoning `B`. The runs keep that order. Read in order, the runs
/// of one channel tile that channel's text.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SourceRun {
    pub channel: SourceChannel,
    pub input: Range<usize>,
}

/// Appends `input` to `runs`, joining it to the last run when both are in one
/// channel and the input is contiguous.
pub(crate) fn push_run(runs: &mut Vec<SourceRun>, channel: SourceChannel, input: Range<usize>) {
    if input.is_empty() {
        return;
    }
    if let Some(last) = runs.last_mut()
        && last.channel == channel
        && last.input.end == input.start
    {
        last.input.end = input.end;
        return;
    }
    runs.push(SourceRun { channel, input });
}

/// The input byte ranges of one held or emitted string, in order.
///
/// The ranges tile the string: their lengths add up to its byte length, and
/// byte `i` of the string came from the input byte that this list maps it to.
/// `None` means recording is off. Every operation then does nothing, so the
/// stock path pays one `None` check per edit and allocates nothing.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Spans(Option<Vec<Range<usize>>>);

impl Spans {
    /// No recording.
    pub(crate) const OFF: Spans = Spans(None);

    /// An empty list when `on`, else [`Spans::OFF`].
    pub(crate) fn recording(on: bool) -> Self {
        Spans(on.then(Vec::new))
    }

    /// One contiguous range, when `on`.
    pub(crate) fn contiguous(on: bool, input: Range<usize>) -> Self {
        let mut spans = Self::recording(on);
        spans.push(input);
        spans
    }

    pub(crate) fn is_recording(&self) -> bool {
        self.0.is_some()
    }

    /// The ranges, or none when recording is off.
    pub(crate) fn ranges(&self) -> &[Range<usize>] {
        self.0.as_deref().unwrap_or(&[])
    }

    pub(crate) fn into_ranges(self) -> Vec<Range<usize>> {
        self.0.unwrap_or_default()
    }

    /// The number of string bytes these ranges cover.
    pub(crate) fn len(&self) -> usize {
        self.ranges().iter().map(|r| r.len()).sum()
    }

    /// Appends the source of bytes added at the end of the string.
    pub(crate) fn push(&mut self, input: Range<usize>) {
        let Some(ranges) = self.0.as_mut() else {
            return;
        };
        if input.is_empty() {
            return;
        }
        if let Some(last) = ranges.last_mut()
            && last.end == input.start
        {
            last.end = input.end;
            return;
        }
        ranges.push(input);
    }

    /// Appends the sources of a string appended at the end of this one.
    pub(crate) fn append(&mut self, other: &Spans) {
        if self.0.is_none() {
            return;
        }
        for range in other.ranges() {
            self.push(range.clone());
        }
    }

    /// Puts the sources of a string inserted at the front before these.
    pub(crate) fn prepend(&mut self, mut front: Spans) {
        if self.0.is_none() {
            return;
        }
        front.0.get_or_insert_with(Vec::new);
        front.append(self);
        *self = front;
    }

    /// The sources of string bytes `at`.
    pub(crate) fn slice(&self, at: Range<usize>) -> Spans {
        let Some(ranges) = self.0.as_ref() else {
            return Spans::OFF;
        };
        let mut out = Spans(Some(Vec::new()));
        let mut offset = 0;
        for range in ranges {
            let (start, end) = (offset, offset + range.len());
            offset = end;
            if end <= at.start {
                continue;
            }
            if start >= at.end {
                break;
            }
            let from = at.start.max(start) - start;
            let to = at.end.min(end) - start;
            out.push(range.start + from..range.start + to);
        }
        debug_assert_eq!(out.len(), at.len().min(offset.saturating_sub(at.start)));
        out
    }

    /// Drops the sources of the first `n` string bytes.
    pub(crate) fn drain_front(&mut self, n: usize) {
        if self.0.is_none() || n == 0 {
            return;
        }
        let total = self.len();
        *self = self.slice(n.min(total)..total);
    }

    /// Splits off the sources of string bytes `at..`, as `String::split_off`.
    pub(crate) fn split_off(&mut self, at: usize) -> Spans {
        if self.0.is_none() {
            return Spans::OFF;
        }
        let total = self.len();
        let tail = self.slice(at..total);
        *self = self.slice(0..at);
        tail
    }

    /// Drops the sources of string bytes `at`, which were cut out of the middle.
    pub(crate) fn remove(&mut self, at: Range<usize>) {
        if self.0.is_none() || at.is_empty() {
            return;
        }
        let total = self.len();
        let tail = self.slice(at.end..total);
        *self = self.slice(0..at.start);
        self.append(&tail);
    }

    pub(crate) fn clear(&mut self) {
        if let Some(ranges) = self.0.as_mut() {
            ranges.clear();
        }
    }

    /// Maps `local`, ranges over the string these spans describe, to input
    /// ranges.
    pub(crate) fn map(&self, local: &Spans) -> Spans {
        if self.0.is_none() {
            return Spans::OFF;
        }
        debug_assert!(local.is_recording(), "mapping text that recorded no ranges");
        let mut out = Spans::recording(true);
        for range in local.ranges() {
            out.append(&self.slice(range.clone()));
        }
        out
    }

    /// Takes the ranges out, leaving an empty list that still records.
    pub(crate) fn take(&mut self) -> Spans {
        match self.0.as_mut() {
            Some(ranges) => Spans(Some(std::mem::take(ranges))),
            None => Spans::OFF,
        }
    }
}

/// The byte range of `part` inside `whole`, when `part` is a subslice of it.
///
/// Parsers build their output from subslices (`trim`, `&s[a..b]`, `split_at`)
/// of the text they were given, so the slice's address gives its input offset
/// exactly. Equal bytes elsewhere in the input cannot be confused with it.
pub(crate) fn subslice_range(whole: &str, part: &str) -> Range<usize> {
    let start = (part.as_ptr() as usize).wrapping_sub(whole.as_ptr() as usize);
    assert!(
        start <= whole.len() && part.len() <= whole.len() - start,
        "subslice_range: not a subslice"
    );
    start..start + part.len()
}

/// Text a parser built from its input, with the input range of each byte.
///
/// Unrecorded text is a bare `String`, so `Option<SourcedText>` is as small as
/// `Option<String>`. The jail's per-chunk completion check holds a parse
/// result across an await, so its size is paid on every jailed chunk; recorded
/// text and its ranges sit behind one box instead.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum SourcedText {
    Plain(String),
    Sourced(Box<(String, Spans)>),
}

/// The sources of [`SourcedText::Plain`].
static NO_SPANS: Spans = Spans::OFF;

impl Default for SourcedText {
    fn default() -> Self {
        Self::Plain(String::new())
    }
}

impl SourcedText {
    fn new(text: String, spans: Spans) -> Self {
        if spans.is_recording() {
            Self::Sourced(Box::new((text, spans)))
        } else {
            Self::Plain(text)
        }
    }

    pub(crate) fn empty(record: bool) -> Self {
        Self::new(String::new(), Spans::recording(record))
    }

    /// Text from a parser that does not record its sources.
    pub(crate) fn unsourced(text: String) -> Self {
        Self::Plain(text)
    }

    /// `part`, a subslice of `base`.
    pub(crate) fn slice(base: &str, part: &str, record: bool) -> Self {
        if !record {
            return Self::Plain(part.to_string());
        }
        Self::new(
            part.to_string(),
            Spans::contiguous(true, subslice_range(base, part)),
        )
    }

    /// `parts`, subslices of `base`, joined.
    pub(crate) fn join(base: &str, parts: &[&str], record: bool) -> Self {
        if !record {
            return Self::Plain(parts.concat());
        }
        let mut spans = Spans::recording(true);
        for part in parts {
            spans.push(subslice_range(base, part));
        }
        Self::new(parts.concat(), spans)
    }

    pub(crate) fn text(&self) -> &str {
        match self {
            Self::Plain(text) => text,
            Self::Sourced(sourced) => &sourced.0,
        }
    }

    /// The sources of the text, [`Spans::OFF`] when it recorded none.
    pub(crate) fn spans(&self) -> &Spans {
        match self {
            Self::Plain(_) => &NO_SPANS,
            Self::Sourced(sourced) => &sourced.1,
        }
    }

    pub(crate) fn into_text(self) -> String {
        match self {
            Self::Plain(text) => text,
            Self::Sourced(sourced) => sourced.0,
        }
    }

    /// Cuts every occurrence of `pattern` out, as `str::replace(pattern, "")`.
    pub(crate) fn remove_all(&mut self, pattern: &str) {
        match self {
            Self::Plain(text) => *text = text.replace(pattern, ""),
            Self::Sourced(sourced) => {
                let (text, spans) = &mut **sourced;
                let cuts: Vec<usize> = text.match_indices(pattern).map(|(at, _)| at).collect();
                for at in cuts.into_iter().rev() {
                    spans.remove(at..at + pattern.len());
                }
                *text = text.replace(pattern, "");
            }
        }
    }

    /// Keeps bytes `at` of the text.
    fn keep(self, at: Range<usize>) -> Self {
        if at.start == 0 && at.end == self.text().len() {
            return self;
        }
        match self {
            Self::Plain(text) => Self::Plain(text[at].to_string()),
            Self::Sourced(mut sourced) => {
                let (text, spans) = &mut *sourced;
                *spans = spans.slice(at.clone());
                *text = text[at].to_string();
                Self::Sourced(sourced)
            }
        }
    }

    pub(crate) fn trim(self) -> Self {
        let text = self.text();
        let start = text.len() - text.trim_start().len();
        let end = text.trim_end().len().max(start);
        self.keep(start..end)
    }

    /// Keeps the first `at` bytes, then trims their end.
    pub(crate) fn truncate_trim_end(self, at: usize) -> Self {
        let end = self.text()[..at].trim_end().len();
        self.keep(0..end)
    }
}

#[cfg(test)]
// A list with one range is the expected value in many cases here.
#[allow(clippy::single_range_in_vec_init)]
mod tests {
    use super::*;

    fn spans(ranges: &[Range<usize>]) -> Spans {
        let mut spans = Spans::recording(true);
        for range in ranges {
            spans.push(range.clone());
        }
        spans
    }

    #[test]
    fn push_joins_contiguous_ranges() {
        assert_eq!(spans(&[0..3, 3..5, 9..10]).ranges(), &[0..5, 9..10]);
    }

    #[test]
    fn slice_maps_string_bytes_to_input_bytes() {
        // String bytes 0..5 came from input 10..15, bytes 5..8 from input 30..33.
        let s = spans(&[10..15, 30..33]);
        assert_eq!(s.slice(0..8).ranges(), &[10..15, 30..33]);
        assert_eq!(s.slice(3..6).ranges(), &[13..15, 30..31]);
        assert_eq!(s.slice(5..5).ranges(), &[] as &[Range<usize>]);
        assert_eq!(s.slice(6..8).ranges(), &[31..33]);
    }

    #[test]
    fn edits_keep_the_sources_of_the_kept_bytes() {
        // The review case: `{"x":"a</tool_call>b"}` with the closer cut out.
        let mut s = spans(&[0..22]);
        s.remove(7..19);
        assert_eq!(s.ranges(), &[0..7, 19..22]);

        let mut s = spans(&[0..4, 10..14]);
        let tail = s.split_off(2);
        assert_eq!(s.ranges(), &[0..2]);
        assert_eq!(tail.ranges(), &[2..4, 10..14]);

        let mut s = spans(&[10..14]);
        s.prepend(spans(&[0..2]));
        assert_eq!(s.ranges(), &[0..2, 10..14]);

        let mut s = spans(&[0..4, 10..14]);
        s.drain_front(5);
        assert_eq!(s.ranges(), &[11..14]);
    }

    #[test]
    fn recording_off_does_nothing() {
        let mut s = Spans::OFF;
        s.push(0..4);
        s.append(&spans(&[0..1]));
        s.prepend(spans(&[0..1]));
        s.remove(0..1);
        assert!(!s.is_recording());
        assert!(!s.split_off(0).is_recording());
        assert!(!s.slice(0..0).is_recording());
        assert_eq!(s.len(), 0);
    }

    #[test]
    fn subslice_range_reads_the_slice_address() {
        let text = "STOP STOP";
        assert_eq!(subslice_range(text, &text[5..]), 5..9);
        assert_eq!(subslice_range(text, text[..4].trim()), 0..4);
        assert_eq!(subslice_range(text, &text[9..]), 9..9);
    }

    #[test]
    #[should_panic(expected = "not a subslice")]
    fn subslice_range_refuses_a_copy() {
        let text = "STOP";
        let copy = text.to_string();
        subslice_range(text, &copy);
    }

    #[test]
    fn sourced_text_trims_and_joins() {
        let base = "  a b  |  c  ";
        let joined = SourcedText::join(base, &[&base[..7], &base[8..]], true);
        assert_eq!(joined.text(), "  a b    c  ");
        let trimmed = joined.trim();
        assert_eq!(trimmed.text(), "a b    c");
        assert_eq!(trimmed.spans().ranges(), &[2..7, 8..11]);
        let cut = SourcedText::slice(base, base, true).truncate_trim_end(7);
        assert_eq!((cut.text(), cut.spans().ranges()), ("  a b", &[0..5][..]));
        let blank = SourcedText::slice(base, &base[..2], true).trim();
        assert_eq!((blank.text(), blank.spans().len()), ("", 0));
        assert!(
            !SourcedText::join(base, &[base], false)
                .spans()
                .is_recording()
        );
    }

    #[test]
    fn remove_all_cuts_each_occurrence() {
        let base = "a</t>b</t></t>c";
        let mut text = SourcedText::slice(base, base, true);
        text.remove_all("</t>");
        assert_eq!(text.text(), "abc");
        assert_eq!(text.spans().ranges(), &[0..1, 5..6, 14..15]);
        let parent = spans(&[100..104, 200..211]);
        assert_eq!(
            parent.map(text.spans()).ranges(),
            &[100..101, 201..202, 210..211]
        );
    }

    #[test]
    fn choice_sources_pair_by_index_and_order() {
        let text = |index, content_len, reasoning_len| ChoiceText {
            index,
            content_len,
            reasoning_len,
        };
        let entry = |index, content: Vec<Range<usize>>| ChoiceSources {
            index,
            content,
            reasoning: vec![],
        };
        // One entry per choice: positional within the index.
        let got = match_choice_sources(
            vec![
                entry(0, vec![0..1]),
                entry(1, vec![5..7]),
                entry(0, vec![1..3]),
            ],
            &[text(0, 1, 0), text(0, 2, 0), text(1, 2, 0)],
        );
        assert_eq!(got[0].as_ref().unwrap().content, vec![0..1]);
        assert_eq!(got[1].as_ref().unwrap().content, vec![1..3]);
        assert_eq!(got[2].as_ref().unwrap().content, vec![5..7]);
        // A textless duplicate without an entry does not take the text's entry.
        let got = match_choice_sources(vec![entry(0, vec![4..5])], &[text(0, 0, 0), text(0, 1, 0)]);
        assert!(got[0].is_none());
        assert_eq!(got[1].as_ref().unwrap().content, vec![4..5]);
        // Ambiguous counts pair nothing; a wrong length is not used.
        let got = match_choice_sources(vec![entry(0, vec![0..1])], &[text(0, 1, 0), text(0, 1, 0)]);
        assert!(got.iter().all(Option::is_none));
        let got = match_choice_sources(vec![entry(0, vec![0..2])], &[text(0, 1, 0)]);
        assert!(got[0].is_none());
    }

    #[test]
    fn push_run_joins_one_channel() {
        let mut runs = Vec::new();
        push_run(&mut runs, SourceChannel::Normal, 0..1);
        push_run(&mut runs, SourceChannel::Normal, 1..2);
        push_run(&mut runs, SourceChannel::Reasoning, 2..3);
        push_run(&mut runs, SourceChannel::Normal, 3..3);
        assert_eq!(
            runs,
            vec![
                SourceRun {
                    channel: SourceChannel::Normal,
                    input: 0..2
                },
                SourceRun {
                    channel: SourceChannel::Reasoning,
                    input: 2..3
                },
            ]
        );
    }
}
