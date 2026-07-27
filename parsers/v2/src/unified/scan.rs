// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Streaming marker scanners shared by unified parsers.

use winnow::Parser;
use winnow::error::{ContextError, ErrMode, ModalResult, Needed};
use winnow::stream::{Offset, Partial, Stream};

pub(super) fn partial_prefix_len(buffer: &str, marker: &str) -> usize {
    let Some(first_byte) = marker.as_bytes().first().copied() else {
        return 0;
    };

    let max_len = buffer.len().min(marker.len().saturating_sub(1));
    let tail_start = buffer.len() - max_len;
    for index in tail_start..buffer.len() {
        if buffer.as_bytes()[index] != first_byte {
            continue;
        }
        let len = buffer.len() - index;
        if buffer.is_char_boundary(index)
            && marker.is_char_boundary(len)
            && marker.as_bytes()[..len] == buffer.as_bytes()[index..]
        {
            return len;
        }
    }
    0
}

pub(super) fn safe_text_len_mul(input: &mut Partial<&str>, markers: &[&str]) -> ModalResult<usize> {
    let text = **input;
    if text.is_empty() {
        return incomplete();
    }

    if let Some(start) = markers.iter().filter_map(|marker| text.find(marker)).min() {
        input.next_slice(start);
        return Ok(start);
    }

    let keep = markers
        .iter()
        .map(|marker| partial_prefix_len(text, marker))
        .max()
        .unwrap_or(0);
    let emit = text.len().saturating_sub(keep);
    if emit == 0 {
        return incomplete();
    }
    input.next_slice(emit);
    Ok(emit)
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(super) struct MarkerScanState {
    scan_start: usize,
}

impl MarkerScanState {
    fn reset(&mut self) {
        self.scan_start = 0;
    }
}

pub(super) fn take_until_marker<'i, 'a>(
    marker: &'a str,
    state: &'a mut MarkerScanState,
) -> impl Parser<Partial<&'i str>, &'i str, ErrMode<ContextError>> + 'a {
    move |input: &mut Partial<&'i str>| {
        let text = **input;
        if text.is_empty() {
            return incomplete();
        }

        let mut scan_start = state.scan_start.min(text.len());
        while !text.is_char_boundary(scan_start) {
            scan_start -= 1;
        }

        if let Some(offset) = text[scan_start..].find(marker) {
            let marker_start = scan_start + offset;
            let body = &text[..marker_start];
            input.next_slice(marker_start);
            state.reset();
            return Ok(body);
        }

        state.scan_start = text.len() - partial_prefix_len(text, marker);
        incomplete()
    }
}

pub(super) fn parse_buffered_event<E>(
    buffer: &str,
    parse: impl FnOnce(&mut Partial<&str>) -> ModalResult<E>,
) -> anyhow::Result<Option<(E, usize)>> {
    let mut input = Partial::new(buffer);
    let checkpoint = input.checkpoint();
    let event = match parse(&mut input) {
        Ok(event) => event,
        Err(ErrMode::Incomplete(_)) => return Ok(None),
        Err(ErrMode::Backtrack(error) | ErrMode::Cut(error)) => {
            let snippet = buffer
                .char_indices()
                .nth(80)
                .map_or(buffer, |(index, _)| &buffer[..index]);
            anyhow::bail!("unified parser failed near {snippet:?}: {error}");
        }
    };
    let consumed = input.offset_from(&checkpoint);
    if consumed == 0 {
        return Ok(None);
    }
    Ok(Some((event, consumed)))
}

fn incomplete<T>() -> ModalResult<T> {
    Err(ErrMode::Incomplete(Needed::Unknown))
}
