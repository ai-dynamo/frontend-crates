// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Deterministic per-choice interleave schedule core (DIS-2381).
//!
//! Invariant this module exists to test:
//!
//! ```text
//!   demux(parse(interleave(A@0, B@1))) == (parse(A), parse(B))
//! ```
//!
//! A streaming tool-calling parser (v1 jail or a v2 `ToolParser`) must keep its
//! state isolated *per `choice.index`*. When a caller asks for `n>1` completions
//! the transport interleaves the choices' deltas onto one wire in an arbitrary,
//! runtime-dependent order. If the parser keys its buffer/marker/jail state off
//! anything other than `choice.index` (e.g. one shared accumulator, or arrival
//! order), one choice's partial marker or JSON leaks into another — producing
//! empty `tool_calls`, duplicated calls, or leaked markup in `content`.
//!
//! Single-choice fixtures can never catch this: with one choice there is no
//! second stream to bleed into, so a shared accumulator looks identical to a
//! per-choice one. This module builds the missing second stream deterministically
//! (no RNG) so the regression lanes can demux by `choice.index` and prove each
//! choice's output is byte-for-byte what it would be if it had run alone.
//!
//! This file is intentionally dependency-free (only `std`). Both the v1 jail lane
//! (`parsers/v1/tests/jail_interleave.rs`) and the v2 conformance lane
//! (`conformance/tests/parity_toolcalling_stream_interleave.rs`) `#[path]`-include
//! it so the schedule logic has exactly one source of truth across crates.

use std::collections::BTreeMap;

/// How to merge `k` per-choice item sequences onto one tagged wire.
///
/// Every variant is a pure function of the inputs — no RNG, no clock. The same
/// `(sequences, schedule)` always yields the same tagged stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Schedule {
    /// Column-major merge: round `r` emits item `r` of choice 0, then choice 1,
    /// ... skipping any choice already exhausted. The simplest fair interleave.
    RoundRobin,
    /// Choice `i` is delayed by `offset * i` rounds, so higher-index choices
    /// "start late". `FirstByteOffset(2)` on a pair delays choice 1 by two rounds:
    /// its first delta lands in the same round as choice 0's third, so choice 0
    /// has already streamed several deltas (opening/continuing a tool-call marker)
    /// before choice 1 produces a byte.
    FirstByteOffset(usize),
    /// Split each of `victim`'s deltas at `num/den` of its length and emit the
    /// sibling's same-round delta *between* the halves. This lands a foreign
    /// delta on a mid-delta boundary — the boundary a shared accumulator is most
    /// likely to corrupt (partial marker / partial JSON straddling the split).
    /// Requires exactly two choices.
    ///
    /// `victim` and the ratio are both parameters because a fixed
    /// "always split choice 0, always at the midpoint" schedule cannot see a
    /// parser that only breaks when the SIBLING is the one being split, or one
    /// that breaks at a non-midpoint boundary such as `<tool_ | call>`.
    BoundarySplit { victim: u32, num: usize, den: usize },
}

impl Schedule {
    /// Human label for failure messages ("RoundRobin", "FirstByteOffset(2)", ...).
    pub fn label(&self) -> String {
        match self {
            Schedule::RoundRobin => "RoundRobin".to_string(),
            Schedule::FirstByteOffset(n) => format!("FirstByteOffset({n})"),
            Schedule::BoundarySplit { victim, num, den } => {
                format!("BoundarySplit(ch{victim}@{num}/{den})")
            }
        }
    }
}

/// An item that `BoundarySplit` can cut in two. Concatenating the two halves
/// must reproduce the original exactly (checked by the roundtrip test).
pub trait Splittable: Clone {
    /// Split at `num/den` of the item's length. Returns `None` when the item
    /// cannot be split there (too small, or the ratio lands on an edge) — the
    /// caller then emits it whole.
    fn split_at_ratio(&self, num: usize, den: usize) -> Option<(Self, Self)>;
}

impl Splittable for String {
    fn split_at_ratio(&self, num: usize, den: usize) -> Option<(Self, Self)> {
        if self.chars().count() < 2 || den == 0 {
            return None;
        }
        // Split on a char boundary at/after the target byte so we never cut a
        // multi-byte UTF-8 sequence (fixtures include emoji and CJK markers).
        //
        // Clamp to at least byte 1 first: a small ratio on a short item rounds the
        // target down to 0, which used to return `None` and emit the item WHOLE —
        // so a split schedule was counted as exercised while behaving exactly like
        // a non-splitting one. Any item of >= 2 chars must genuinely split.
        let mut idx = ((self.len() * num) / den).max(1);
        while idx < self.len() && !self.is_char_boundary(idx) {
            idx += 1;
        }
        // The forward scan runs off the end when the target lands inside the FINAL
        // character — every chunk ending in a multi-byte char (emoji, CJK) hit this
        // and was emitted WHOLE, so the split schedule silently behaved like a
        // non-splitting one for it. Fall back to the last interior boundary.
        if idx >= self.len() {
            idx = self.len() - 1;
            while idx > 0 && !self.is_char_boundary(idx) {
                idx -= 1;
            }
        }
        if idx == 0 || idx >= self.len() {
            return None;
        }
        Some((self[..idx].to_string(), self[idx..].to_string()))
    }
}

impl Splittable for Vec<u32> {
    fn split_at_ratio(&self, num: usize, den: usize) -> Option<(Self, Self)> {
        if self.len() < 2 || den == 0 {
            return None;
        }
        let at = ((self.len() * num) / den).clamp(1, self.len() - 1);
        Some((self[..at].to_vec(), self[at..].to_vec()))
    }
}

/// Merge `sequences` (indexed by `choice.index`, so `sequences[i]` is choice `i`)
/// into a single `(choice_index, item)` stream ordered per `schedule`.
///
/// Pure and deterministic. `BoundarySplit` requires `sequences.len() == 2`.
pub fn interleave_items<T: Splittable>(sequences: &[Vec<T>], schedule: Schedule) -> Vec<(u32, T)> {
    match schedule {
        Schedule::RoundRobin => offset_merge(sequences, 0),
        Schedule::FirstByteOffset(n) => offset_merge(sequences, n),
        Schedule::BoundarySplit { victim, num, den } => {
            assert_eq!(
                sequences.len(),
                2,
                "BoundarySplit is defined for exactly two choices"
            );
            assert!(victim < 2, "BoundarySplit victim must be choice 0 or 1");
            boundary_split(&sequences[0], &sequences[1], victim, num, den)
        }
    }
}

/// Round-major merge where choice `i` is delayed by `offset * i` rounds.
/// `offset == 0` is a plain round-robin.
fn offset_merge<T: Clone>(sequences: &[Vec<T>], offset: usize) -> Vec<(u32, T)> {
    let mut out = Vec::new();
    let last_round = sequences
        .iter()
        .enumerate()
        .map(|(i, s)| s.len() + offset * i)
        .max()
        .unwrap_or(0);
    for round in 0..last_round {
        for (i, seq) in sequences.iter().enumerate() {
            let delay = offset * i;
            if round < delay {
                continue;
            }
            let local = round - delay;
            if local < seq.len() {
                out.push((i as u32, seq[local].clone()));
            }
        }
    }
    out
}

/// `BoundarySplit`: for each round, cut the VICTIM choice's delta at `num/den`
/// and drop the sibling's same-round delta between the halves.
fn boundary_split<T: Splittable>(
    a: &[T],
    b: &[T],
    victim: u32,
    num: usize,
    den: usize,
) -> Vec<(u32, T)> {
    let (vic, sib) = if victim == 0 { (a, b) } else { (b, a) };
    let sib_idx = 1 - victim;
    let mut out = Vec::new();
    let rounds = vic.len().max(sib.len());
    for r in 0..rounds {
        if r < vic.len() {
            match vic[r].split_at_ratio(num, den) {
                Some((head, tail)) => {
                    out.push((victim, head));
                    if r < sib.len() {
                        out.push((sib_idx, sib[r].clone()));
                    }
                    out.push((victim, tail));
                }
                None => {
                    out.push((victim, vic[r].clone()));
                    if r < sib.len() {
                        out.push((sib_idx, sib[r].clone()));
                    }
                }
            }
        } else if r < sib.len() {
            out.push((sib_idx, sib[r].clone()));
        }
    }
    out
}

/// Group a tagged stream back into per-`choice.index` item sequences.
///
/// Demux is by tag (`choice.index`), never by arrival order — the whole point is
/// that a correct parser is order-independent within a choice but must not mix
/// choices.
pub fn demux_items<T: Clone>(tagged: &[(u32, T)]) -> BTreeMap<u32, Vec<T>> {
    let mut out: BTreeMap<u32, Vec<T>> = BTreeMap::new();
    for (index, item) in tagged {
        out.entry(*index).or_default().push(item.clone());
    }
    out
}

// ── Schedule-core roundtrip tests ───────────────────────────────────────────
//
// These are compiled + run wherever this file is `#[path]`-included (the v1 jail
// lane). They prove the merge itself is lossless before any parser is involved:
// de-interleaving each schedule's output by `choice.index` recovers each input's
// bytes exactly. If these fail, a downstream parity failure is the schedule's
// fault, not the parser's.

#[cfg(test)]
mod schedule_core_roundtrip {
    use super::*;

    fn concat_strings(items: &[String]) -> String {
        items.concat()
    }

    /// Byte-exact concatenation roundtrip for every schedule.
    #[test]
    fn concat_roundtrip_all_schedules() {
        let a: Vec<String> = [
            "<tool_call>",
            "{\"name\":\"get_",
            "weather\"}",
            "</tool_call>",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let b: Vec<String> = ["plain ", "content ", "只是文本 ", "🧪 done"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let seqs = vec![a.clone(), b.clone()];

        for schedule in [
            Schedule::RoundRobin,
            Schedule::FirstByteOffset(1),
            Schedule::FirstByteOffset(3),
            Schedule::BoundarySplit {
                victim: 0,
                num: 1,
                den: 2,
            },
            Schedule::BoundarySplit {
                victim: 1,
                num: 1,
                den: 3,
            },
        ] {
            let tagged = interleave_items(&seqs, schedule);
            let demuxed = demux_items(&tagged);
            assert_eq!(
                concat_strings(&demuxed[&0]),
                concat_strings(&a),
                "{}: choice 0 bytes not recovered",
                schedule.label()
            );
            assert_eq!(
                concat_strings(&demuxed[&1]),
                concat_strings(&b),
                "{}: choice 1 bytes not recovered",
                schedule.label()
            );
        }
    }

    /// Non-splitting schedules preserve chunk boundaries item-for-item, not just
    /// the concatenation.
    #[test]
    fn item_for_item_roundtrip_non_splitting() {
        let a: Vec<String> = ["a0", "a1", "a2"].iter().map(|s| s.to_string()).collect();
        let b: Vec<String> = ["b0", "b1"].iter().map(|s| s.to_string()).collect();
        let seqs = vec![a.clone(), b.clone()];

        for schedule in [Schedule::RoundRobin, Schedule::FirstByteOffset(2)] {
            let demuxed = demux_items(&interleave_items(&seqs, schedule));
            assert_eq!(demuxed[&0], a, "{}: choice 0 chunks", schedule.label());
            assert_eq!(demuxed[&1], b, "{}: choice 1 chunks", schedule.label());
        }
    }

    /// FirstByteOffset actually delays the higher-index choice: choice 1's first
    /// delta only appears after choice 0 has been streaming for `offset` rounds
    /// (choice 0's round-`offset` item linearizes just before it, hence
    /// `offset + 1` choice-0 items precede choice 1's first).
    #[test]
    fn first_byte_offset_delays_second_choice() {
        let a: Vec<String> = (0..4).map(|i| format!("a{i}")).collect();
        let b: Vec<String> = (0..4).map(|i| format!("b{i}")).collect();
        let offset = 2;
        let tagged = interleave_items(&[a, b], Schedule::FirstByteOffset(offset));
        let first_b_pos = tagged.iter().position(|(idx, _)| *idx == 1).unwrap();
        let zero_before = tagged[..first_b_pos]
            .iter()
            .filter(|(idx, _)| *idx == 0)
            .count();
        assert_eq!(
            zero_before,
            offset + 1,
            "choice 1 must not start until choice 0 has streamed {offset} rounds"
        );
    }

    /// Roundtrip holds for token sequences too (the Harmony token-native path).
    #[test]
    fn concat_roundtrip_tokens() {
        let a: Vec<Vec<u32>> = vec![vec![1, 2, 3], vec![4], vec![5, 6]];
        let b: Vec<Vec<u32>> = vec![vec![7, 8], vec![9, 10, 11]];
        let seqs = vec![a.clone(), b.clone()];
        let flat = |v: &[Vec<u32>]| v.iter().flatten().copied().collect::<Vec<u32>>();

        for schedule in [
            Schedule::RoundRobin,
            Schedule::FirstByteOffset(1),
            Schedule::BoundarySplit {
                victim: 0,
                num: 1,
                den: 2,
            },
            Schedule::BoundarySplit {
                victim: 1,
                num: 1,
                den: 3,
            },
        ] {
            let demuxed = demux_items(&interleave_items(&seqs, schedule));
            assert_eq!(
                flat(&demuxed[&0]),
                flat(&a),
                "{} tokens ch0",
                schedule.label()
            );
            assert_eq!(
                flat(&demuxed[&1]),
                flat(&b),
                "{} tokens ch1",
                schedule.label()
            );
        }
    }

    /// Every item of >= 2 chars must ACTUALLY split at every ratio, including
    /// when the target byte lands inside a trailing multi-byte character. A
    /// silent `None` here would let a split schedule be counted as exercised
    /// while behaving exactly like a non-splitting one.
    #[test]
    fn split_at_ratio_never_silently_declines() {
        for s in [
            "ab",
            "xy",
            "  ",
            " \t ",
            "ab€",
            "x只",
            "hi🧪",
            "只是文本",
            "a🧪",
        ] {
            for (num, den) in [(1usize, 4usize), (1, 3), (1, 2), (3, 4)] {
                let got = s.to_string().split_at_ratio(num, den);
                let (head, tail) = got.unwrap_or_else(|| {
                    panic!(
                        "{s:?} @{num}/{den}: refused to split a {}-char item",
                        s.chars().count()
                    )
                });
                assert!(
                    !head.is_empty() && !tail.is_empty(),
                    "{s:?} @{num}/{den}: empty half"
                );
                assert_eq!(
                    format!("{head}{tail}"),
                    s,
                    "{s:?} @{num}/{den}: not lossless"
                );
            }
        }
    }

    /// k=3 round-robin merges three choices losslessly.
    #[test]
    fn round_robin_three_choices() {
        let a: Vec<String> = vec!["a0".into(), "a1".into()];
        let b: Vec<String> = vec!["b0".into(), "b1".into(), "b2".into()];
        let c: Vec<String> = vec!["c0".into()];
        let demuxed = demux_items(&interleave_items(
            &[a.clone(), b.clone(), c.clone()],
            Schedule::RoundRobin,
        ));
        assert_eq!(demuxed[&0], a);
        assert_eq!(demuxed[&1], b);
        assert_eq!(demuxed[&2], c);
    }
}
