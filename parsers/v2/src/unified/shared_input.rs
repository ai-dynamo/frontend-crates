// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Immutable consumed input shared by continuations, followed by a private tail.

use std::borrow::{Borrow, Cow};
use std::ops::Range;
use std::sync::Arc;

pub(super) trait Fragment: ToOwned<Owned: Default + Clone> + 'static {
    fn empty() -> &'static Self;
    fn len(&self) -> usize;
    fn slice(&self, range: Range<usize>) -> &Self;
    fn append(target: &mut Self::Owned, input: &Self);
}

impl Fragment for str {
    fn empty() -> &'static Self {
        ""
    }
    fn len(&self) -> usize {
        str::len(self)
    }
    fn slice(&self, range: Range<usize>) -> &Self {
        &self[range]
    }
    fn append(target: &mut String, input: &Self) {
        target.push_str(input);
    }
}

impl Fragment for [u32] {
    fn empty() -> &'static Self {
        &[]
    }
    fn len(&self) -> usize {
        <[u32]>::len(self)
    }
    fn slice(&self, range: Range<usize>) -> &Self {
        &self[range]
    }
    fn append(target: &mut Vec<u32>, input: &Self) {
        target.extend_from_slice(input);
    }
}

pub(super) struct SharedInput<F: Fragment + ?Sized> {
    prefix: Option<Arc<F>>,
    tail: F::Owned,
}

impl<F: Fragment + ?Sized> Default for SharedInput<F> {
    fn default() -> Self {
        Self {
            prefix: None,
            tail: F::Owned::default(),
        }
    }
}

impl<F: Fragment + ?Sized> Clone for SharedInput<F> {
    fn clone(&self) -> Self {
        Self {
            prefix: self.prefix.clone(),
            tail: self.tail.clone(),
        }
    }
}

impl<F: Fragment + ?Sized> SharedInput<F> {
    pub(super) fn view(&self) -> InputView<'_, F> {
        InputView {
            prefix: self.prefix.as_deref().unwrap_or(F::empty()),
            tail: self.tail.borrow(),
        }
    }

    pub(super) fn len(&self) -> usize {
        self.view().len()
    }

    pub(super) fn clear(&mut self) {
        *self = Self::default();
    }

    pub(super) fn append(&mut self, input: &F) {
        F::append(&mut self.tail, input);
    }

    /// Freeze once at a checkpoint. Clones then share consumed input; appending
    /// generated data never calls `Arc::make_mut` on the consumed allocation.
    pub(super) fn checkpoint(&mut self) -> Self
    where
        Arc<F>: From<F::Owned>,
    {
        if self.tail.borrow().len() != 0 {
            let owned = if let Some(prefix) = &self.prefix {
                let mut joined = prefix.as_ref().to_owned();
                F::append(&mut joined, self.tail.borrow());
                self.tail = F::Owned::default();
                joined
            } else {
                std::mem::take(&mut self.tail)
            };
            self.prefix = Some(Arc::from(owned));
        }
        Self {
            prefix: self.prefix.clone(),
            tail: F::Owned::default(),
        }
    }
}

pub(super) struct InputView<'a, F: Fragment + ?Sized> {
    prefix: &'a F,
    tail: &'a F,
}

impl<F: Fragment + ?Sized> Copy for InputView<'_, F> {}
impl<F: Fragment + ?Sized> Clone for InputView<'_, F> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, F: Fragment + ?Sized> InputView<'a, F> {
    pub(super) fn len(self) -> usize {
        self.prefix.len() + self.tail.len()
    }

    pub(super) fn slice(self, range: Range<usize>) -> Cow<'a, F> {
        let boundary = self.prefix.len();
        assert!(range.start <= range.end && range.end <= self.len());
        if range.end <= boundary {
            Cow::Borrowed(self.prefix.slice(range))
        } else if range.start >= boundary {
            Cow::Borrowed(
                self.tail
                    .slice(range.start - boundary..range.end - boundary),
            )
        } else {
            let mut value = self.prefix.slice(range.start..boundary).to_owned();
            F::append(&mut value, self.tail.slice(0..range.end - boundary));
            Cow::Owned(value)
        }
    }
}

impl<'a> From<&'a str> for InputView<'a, str> {
    fn from(tail: &'a str) -> Self {
        Self { prefix: "", tail }
    }
}

impl<'a> InputView<'a, str> {
    pub(super) fn chars_from(self, start: usize) -> impl Iterator<Item = (usize, char)> + 'a {
        let boundary = self.prefix.len();
        let prefix_start = start.min(boundary);
        let tail_start = start.saturating_sub(boundary);
        self.prefix[prefix_start..]
            .char_indices()
            .map(move |(offset, ch)| (prefix_start + offset, ch))
            .chain(
                self.tail[tail_start..]
                    .char_indices()
                    .map(move |(offset, ch)| (boundary + tail_start + offset, ch)),
            )
    }

    pub(super) fn reader(self) -> impl std::io::Read + 'a {
        use std::io::Read;
        self.prefix.as_bytes().chain(self.tail.as_bytes())
    }

    pub(super) fn is_whitespace(self) -> bool {
        self.prefix.trim().is_empty() && self.tail.trim().is_empty()
    }
}

impl SharedInput<[u32]> {
    pub(super) fn push(&mut self, token: u32) {
        self.tail.push(token);
    }

    pub(super) fn iter(&self) -> impl Iterator<Item = &u32> {
        self.view().prefix.iter().chain(self.view().tail.iter())
    }
}

impl std::fmt::Display for SharedInput<str> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.view().prefix)?;
        formatter.write_str(self.view().tail)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;

    #[test]
    fn first_append_keeps_prefix_shared_and_choices_independent() {
        let mut input = SharedInput::<str>::default();
        input.append(&"hello ".repeat(100_000));
        let snapshot = input.checkpoint();
        let mut left = snapshot.clone();
        let mut right = snapshot.clone();
        left.append("left");
        right.append("right");
        assert!(Arc::ptr_eq(
            left.prefix.as_ref().unwrap(),
            right.prefix.as_ref().unwrap()
        ));
        assert_eq!(left.view().slice(600_000..600_004), "left");
        assert_eq!(right.view().slice(600_000..600_005), "right");
        assert_eq!(input.len(), 600_000);
    }

    #[test]
    fn view_preserves_offsets_and_cross_segment_literals() {
        let mut input = SharedInput::<str>::default();
        input.append("a漢");
        input.checkpoint();
        input.append("🙂z");
        assert_eq!(
            input.view().chars_from(1).collect::<Vec<_>>(),
            vec![(1, '漢'), (4, '🙂'), (8, 'z')]
        );
        assert_eq!(input.view().slice(1..8), "漢🙂");
        let mut bytes = String::new();
        input.view().reader().read_to_string(&mut bytes).unwrap();
        assert_eq!(bytes, "a漢🙂z");
    }

    #[test]
    fn unresolved_token_history_uses_the_same_shared_owner() {
        let mut input = SharedInput::<[u32]>::default();
        input.append(&[1, 2, 3]);
        let mut fork = input.checkpoint();
        fork.push(4);
        assert_eq!(fork.view().slice(1..4).as_ref(), &[2, 3, 4]);
        assert_eq!(input.iter().copied().collect::<Vec<_>>(), vec![1, 2, 3]);
        input.clear();
        assert_eq!(input.len(), 0);
    }
}
