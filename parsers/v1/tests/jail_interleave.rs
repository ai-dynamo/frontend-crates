// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Per-choice isolation regression lane for the v1 jail (DIS-2381 step 3).
//!
//! Invariant under test:
//!
//! ```text
//!   demux(parse(interleave(A@0, B@1))) == (parse(A), parse(B))
//! ```
//!
//! `JailedStream` keys its jail/marker/tool-call state off `choice.index` via
//! `ChoiceJailStateCollection`. A regression that shared one state across indices
//! (or routed every delta to index 0) would still pass every existing test,
//! because all existing jail tests are effectively single-choice: even
//! `test_multiple_choices_independent_jailing` delivers each choice's deltas in
//! its own slot of a packed multi-choice chunk, so a shared accumulator that
//! processed choices in a fixed order could still look correct.
//!
//! This lane instead runs two independent choices through ONE `JailedStream` with
//! their deltas *interleaved on the wire* under several deterministic schedules
//! (round-robin, first-byte offset, mid-delta boundary split — see
//! `common/interleave.rs`). It then demuxes the emitted chunks by `choice.index`
//! and asserts each choice's assembled result (tool calls, normal text, finish
//! handling) is byte-for-byte what that choice produced running ALONE. If jail
//! state leaks across choices, one choice's partial marker/JSON corrupts the
//! other and the demuxed assembly diverges from its solo golden.
//!
//! Two shapes get dedicated coverage because a per-choice bug can hide from the
//! ordinary pairs entirely (both were found and closed in the sibling PR
//! ai-dynamo/dynamo#11563):
//!
//!   * **Whitespace-only deltas** — the "undecided" arm, where a delta is neither
//!     clearly content nor the start of a marker. Covered by the
//!     `hermes_open_jail_x_ws_*` pairs, which hold choice 0's jail OPEN across the
//!     rounds in which the sibling's whitespace arrives; that open-jail window is
//!     the only one in which a shared undecided buffer is observable.
//!   * **Packed multi-choice chunks with non-contiguous, unsorted indices** —
//!     see `jail_interleave_packed_chunks_non_contiguous_indices`.
//!
//! Golden = the solo run (never a hand-authored n>1 expectation). Because the
//! jail reassembles content across arbitrary delta boundaries, the assembled
//! result is invariant to where a delta is split, so the same solo golden is
//! valid for every schedule.
//!
//! # Known limitation of the solo golden
//!
//! The solo golden is computed in THIS process, so it cannot detect a defect that
//! corrupts the interleaved run and the solo run identically — process-global
//! state such as a `OnceLock`/`lazy_static` cache, or anything memoised on first
//! use. Both sides would be wrong in the same way and still compare equal.
//! (rmccorm4 demonstrated this class on PR #135 against the v2 lane; the v2 lane
//! closes it by also asserting against the corpus's recorded `dynamo_v2` capture,
//! which was produced in an earlier process.)
//!
//! That escape hatch is NOT closed here: these pairs are hand-authored, so there
//! is no recorded capture to anchor to. What this lane does prove is per-choice
//! isolation; a shared-global defect in the v1 jail would need either a recorded
//! v1 corpus or a fresh-process baseline to catch.
//!
//! # What each dimension of the matrix exists for
//!
//! * **Both role assignments** (`AB` / `BA`) — the schedules give choice 0 the
//!   first slot of every round, so a single ordering never exercises the higher
//!   index arriving first.
//! * **`BoundarySplit` over both victims and several ratios** — a fixed
//!   "split choice 0 at the midpoint" cannot see a parser that breaks only when
//!   the sibling is split, or one that breaks at a boundary like `<tool_ | call>`.
//! * **Staggered terminators** — each choice's `finish_reason` rides its OWN last
//!   delta, so a shorter choice terminates MID-WIRE while its sibling streams on.
//!   A defect that tears down sibling state when the first choice finishes is
//!   simply unreachable without that.
//! * **`emission_profile`** — final totals cannot see a defect that merely DELAYS
//!   a choice's output while a sibling is live.
//! * **Modes and families** — `Mode::ToolChoice*` start the jail already jailed
//!   (`JailMode::Immediate`), a different state machine from the marker path, and
//!   a second marker family guards against Hermes-only scanner assumptions.

#[path = "common/interleave.rs"]
mod interleave;

use dynamo_parsers::tool_calling::jail::{Annotated, JailedStream};
use dynamo_protocols::types::{
    ChatChoiceStream, ChatCompletionMessageContent, ChatCompletionStreamResponseDelta,
    CreateChatCompletionStreamResponse, FinishReason, Role,
};
use futures::{StreamExt, stream};
use interleave::{Schedule, demux_items, interleave_items};
use std::collections::BTreeMap;

/// Build a single-choice content chunk tagged with `index` (mirrors the shape of
/// `create_mock_response_chunk` in `jail.rs`).
fn single_choice_chunk(
    content: &str,
    index: u32,
    finish_reason: Option<FinishReason>,
) -> Annotated<CreateChatCompletionStreamResponse> {
    #[allow(deprecated)]
    let choice = ChatChoiceStream {
        index,
        delta: ChatCompletionStreamResponseDelta {
            role: Some(Role::Assistant),
            content: Some(ChatCompletionMessageContent::Text(content.to_string())),
            tool_calls: None,
            function_call: None,
            refusal: None,
            reasoning_content: None,
        },
        finish_reason,
        logprobs: None,
    };
    Annotated {
        data: Some(CreateChatCompletionStreamResponse {
            id: "jail-interleave".to_string(),
            choices: vec![choice],
            created: 0,
            model: "test-model".to_string(),
            system_fingerprint: None,
            object: "chat.completion.chunk".to_string(),
            usage: None,
            service_tier: None,
        }),
        id: None,
        event: None,
        comment: None,
        error: None,
    }
}

/// Assembled per-choice view of jail output: the parts a real n>1 client sees.
#[derive(Debug, PartialEq)]
struct Assembled {
    /// `(name, arguments)` per tool call, accumulated by tool-call index.
    tool_calls: Vec<(String, String)>,
    /// Concatenated non-tool-call content.
    normal_text: String,
    /// Terminal finish reason the jail attributed to this choice, if any.
    finish: Option<FinishReason>,
    /// STREAMING SHAPE: after each chunk this choice emits, the running
    /// `(tool-call count, normal-text length)`.
    ///
    /// Final totals alone cannot see a defect that DELAYS a choice's output —
    /// e.g. a parser that stalls choice 0 the moment choice 1 appears and
    /// releases the correct bytes only at finalize. Totals match, so an
    /// assembled-only comparison passes while a real client sees choice 0 go
    /// silent for the whole of choice 1's stream. The profile makes emission
    /// timing part of the compared value: a correct per-choice implementation
    /// emits the same chunks in the same order whether or not a sibling is
    /// interleaved, so this must equal the solo run's profile exactly.
    emission_profile: Vec<(usize, usize)>,
}

/// Assemble one choice's emitted chunks (already demuxed by `choice.index`).
fn assemble(chunks: &[&ChatChoiceStream]) -> Assembled {
    let mut normal_text = String::new();
    let mut names: BTreeMap<u32, String> = BTreeMap::new();
    let mut args: BTreeMap<u32, String> = BTreeMap::new();
    let mut order: Vec<u32> = Vec::new();
    let mut finish: Option<FinishReason> = None;
    let mut emission_profile: Vec<(usize, usize)> = Vec::new();
    let mut deltas_so_far = 0usize;

    for choice in chunks {
        if let Some(ChatCompletionMessageContent::Text(text)) = choice.delta.content.as_ref() {
            normal_text.push_str(text);
        }
        if let Some(calls) = choice.delta.tool_calls.as_ref() {
            for call in calls {
                deltas_so_far += 1;
                if !order.contains(&call.index) {
                    order.push(call.index);
                }
                if let Some(function) = call.function.as_ref() {
                    if let Some(name) = function.name.as_ref() {
                        names.entry(call.index).or_default().push_str(name);
                    }
                    if let Some(a) = function.arguments.as_ref() {
                        args.entry(call.index).or_default().push_str(a);
                    }
                }
            }
        }
        if let Some(reason) = choice.finish_reason {
            finish = Some(reason);
        }
        // One profile sample per emitted chunk for this choice.
        //
        // Counts emitted DELTAS and accumulated argument bytes, NOT distinct call
        // indices: an index count only moves when a brand-new call first appears,
        // so a stall that delays later argument fragments of an already-started
        // call would leave every sample identical — precisely the shape this field
        // exists to catch. (v2 samples its delta count for the same reason.)
        let bytes_so_far = args.values().map(String::len).sum::<usize>() + normal_text.len();
        emission_profile.push((deltas_so_far, bytes_so_far));
    }

    let tool_calls = order
        .into_iter()
        .map(|idx| {
            (
                names.get(&idx).cloned().unwrap_or_default(),
                args.get(&idx).cloned().unwrap_or_default(),
            )
        })
        .collect();

    Assembled {
        tool_calls,
        normal_text,
        finish,
        emission_profile,
    }
}

/// Build one chunk carrying SEVERAL choices at once (the `Packed` emission shape
/// a real `n>1` engine produces). `entries` is `(choice.index, content)`; indices
/// must be distinct within a chunk, but need not be sorted or contiguous.
fn packed_chunk(
    entries: &[(u32, String, Option<FinishReason>)],
) -> Annotated<CreateChatCompletionStreamResponse> {
    #[allow(deprecated)]
    let choices: Vec<ChatChoiceStream> = entries
        .iter()
        .map(|(index, content, finish)| ChatChoiceStream {
            index: *index,
            delta: ChatCompletionStreamResponseDelta {
                role: Some(Role::Assistant),
                content: Some(ChatCompletionMessageContent::Text(content.clone())),
                tool_calls: None,
                function_call: None,
                refusal: None,
                reasoning_content: None,
            },
            finish_reason: *finish,
            logprobs: None,
        })
        .collect();
    Annotated {
        data: Some(CreateChatCompletionStreamResponse {
            id: "jail-interleave-packed".to_string(),
            choices,
            created: 0,
            model: "test-model".to_string(),
            system_fingerprint: None,
            object: "chat.completion.chunk".to_string(),
            usage: None,
            service_tier: None,
        }),
        id: None,
        event: None,
        comment: None,
        error: None,
    }
}

/// Run one choice's deltas solo through a fresh `JailedStream` and assemble the
/// single-choice output. This is the golden the interleaved run must reproduce.
///
/// `index` is the `choice.index` the solo run uses. A correct jail keys state off
/// the index but its *output* must not depend on the index's value, so goldens are
/// taken at the same index the interleaved run uses — that way a divergence is
/// attributable to cross-choice leakage, not to the index value itself.
async fn solo_at(parser: &str, mode: Mode, deltas: &[String], index: u32) -> Assembled {
    let last = deltas.len().saturating_sub(1);
    let chunks: Vec<_> = deltas
        .iter()
        .enumerate()
        .map(|(i, d)| {
            // Real upstream terminator on this choice's OWN last delta.
            let fin = (i == last).then_some(FinishReason::Stop);
            single_choice_chunk(d, index, fin)
        })
        .collect();
    let results: Vec<_> = build_jail(parser, mode)
        .apply_with_finish_reason(stream::iter(chunks))
        .collect()
        .await;
    let choices: Vec<&ChatChoiceStream> = results
        .iter()
        .filter_map(|r| r.data.as_ref())
        .flat_map(|d| d.choices.iter())
        .collect();
    assemble(&choices)
}

/// Feed an interleaved multi-choice stream through ONE `JailedStream`, then demux
/// the emitted chunks by `choice.index` and assemble each choice separately.
async fn interleaved_by_choice(
    parser: &str,
    mode: Mode,
    sequences: &[Vec<String>],
    schedule: Schedule,
) -> BTreeMap<u32, Assembled> {
    let tagged = interleave_items(sequences, schedule);
    // STAGGERED TERMINATION: each choice's terminator rides its OWN last delta,
    // which for a shorter choice lands MID-WIRE while its sibling is still
    // streaming. A defect that tears down every choice's state when the first
    // choice finishes is only reachable this way; hanging every terminator off
    // the end of the wire (or omitting them, as this lane used to) hides it.
    let mut remaining: BTreeMap<u32, usize> = BTreeMap::new();
    for (index, _) in &tagged {
        *remaining.entry(*index).or_default() += 1;
    }
    let mut seen: BTreeMap<u32, usize> = BTreeMap::new();
    let chunks: Vec<_> = tagged
        .iter()
        .map(|(index, content)| {
            let n = seen.entry(*index).or_default();
            *n += 1;
            let fin = (*n == remaining[index]).then_some(FinishReason::Stop);
            single_choice_chunk(content, *index, fin)
        })
        .collect();
    let results: Vec<_> = build_jail(parser, mode)
        .apply_with_finish_reason(stream::iter(chunks))
        .collect()
        .await;

    // Demux by `choice.index` (NEVER by arrival order): flat_map every choice out
    // of every emitted chunk — the jail may pack or split, so we must not assume
    // one choice per chunk.
    let mut per_choice: BTreeMap<u32, Vec<&ChatChoiceStream>> = BTreeMap::new();
    for choice in results
        .iter()
        .filter_map(|r| r.data.as_ref())
        .flat_map(|d| d.choices.iter())
    {
        per_choice.entry(choice.index).or_default().push(choice);
    }
    per_choice
        .into_iter()
        .map(|(index, chunks)| (index, assemble(&chunks)))
        .collect()
}

/// Jail configuration under test. `MarkerBased` is the default streaming path;
/// the `ToolChoice*` variants start the jail ALREADY jailed (`JailMode::Immediate`),
/// which is a different per-choice state machine — `get_or_create_state` is
/// called with `starts_jailed = true`, so a defect that shares state only on that
/// path is invisible to a marker-based-only lane.
#[derive(Clone, Copy, PartialEq)]
enum Mode {
    MarkerBased,
    ToolChoiceRequired,
    ToolChoiceNamed(&'static str),
}

impl Mode {
    fn label(&self) -> &'static str {
        match self {
            Mode::MarkerBased => "marker",
            Mode::ToolChoiceRequired => "required",
            Mode::ToolChoiceNamed(_) => "named",
        }
    }
}

/// Build a `JailedStream` for `parser` in `mode`.
fn build_jail(parser: &str, mode: Mode) -> JailedStream {
    let b = JailedStream::builder().tool_call_parser(parser);
    match mode {
        Mode::MarkerBased => b.build(),
        Mode::ToolChoiceRequired => b.tool_choice_required().build(),
        Mode::ToolChoiceNamed(name) => b.tool_choice_named(name.to_string()).build(),
    }
}

/// A named divergent pair: two choices that produce structurally different output
/// through the same parser, so a shared accumulator visibly corrupts one.
struct Pair {
    name: &'static str,
    parser: &'static str,
    mode: Mode,
    sequences: Vec<Vec<String>>,
}

fn s(parts: &[&str]) -> Vec<String> {
    parts.iter().map(|p| p.to_string()).collect()
}

fn divergent_pairs() -> Vec<Pair> {
    vec![
        // (tool call) x (plain content only): the classic n>1 leak — a jailed
        // tool call in choice 0 must not swallow choice 1's plain prose.
        Pair {
            name: "hermes_toolcall_x_plain",
            parser: "hermes",
            mode: Mode::MarkerBased,
            sequences: vec![
                s(&[
                    "Let me check. ",
                    r#"<tool_call>{"name":"get_weather","arguments":{"city":"Paris"}}</tool_call>"#,
                    " done.",
                ]),
                s(&["Just ", "plain ", "prose, no tools here."]),
            ],
        },
        // (two different tool-call shapes): both choices emit tool calls but with
        // different names/args — a shared buffer would cross-contaminate arguments.
        Pair {
            name: "hermes_two_distinct_calls",
            parser: "hermes",
            mode: Mode::MarkerBased,
            sequences: vec![
                s(&[
                    r#"<tool_call>{"name":"get_weather","#,
                    r#""arguments":{"city":"Paris"}}</tool_call>"#,
                ]),
                s(&[
                    r#"<tool_call>{"name":"get_time","#,
                    r#""arguments":{"tz":"UTC"}}</tool_call>"#,
                ]),
            ],
        },
        // (opening-marker-split-across-deltas) x (bare content): choice 0's
        // `<tool_call>` marker is split across delta boundaries; a shared partial
        // marker buffer would jail choice 1's bare content by mistake.
        Pair {
            name: "hermes_split_marker_x_bare",
            parser: "hermes",
            mode: Mode::MarkerBased,
            sequences: vec![
                s(&[
                    "<tool",
                    "_call>",
                    r#"{"name":"lookup","arguments":{"q":"cats"}}"#,
                    "</tool_call>",
                ]),
                s(&["bare content only, ", "never jailed"]),
            ],
        },
        // (open jail) x (whitespace-only deltas, then prose): choice 1's leading
        // deltas are pure whitespace, which the jail cannot yet classify as content
        // or as the start of a marker — the "undecided" bypass arm. Choice 0's jail
        // is deliberately held OPEN across those rounds (its marker and JSON are
        // split across deltas), so the whitespace arrives mid-jail. That open-jail
        // window is the only one in which a shared undecided buffer is observable:
        // the whitespace is swallowed into choice 0's accumulation and choice 1's
        // prose loses its leading run. The sibling PR ai-dynamo/dynamo#11563 closed
        // exactly this arm on the preprocessor side.
        Pair {
            name: "hermes_open_jail_x_ws_then_prose",
            parser: "hermes",
            mode: Mode::MarkerBased,
            sequences: vec![
                s(&[
                    "<tool_call>",
                    r#"{"name":"ws_a","#,
                    r#""arguments":{"k":1}}"#,
                    "</tool_call>",
                ]),
                s(&["   ", "\n\n", " \t ", "plain tail after whitespace"]),
            ],
        },
        // (open jail, marker split across deltas) x (whitespace for the WHOLE
        // stream): the same undecided arm at its extreme — choice 1 never emits a
        // non-whitespace byte, so a shared undecided buffer makes choice 1 vanish
        // from the output entirely rather than merely losing a leading run.
        Pair {
            name: "hermes_open_jail_x_ws_only",
            parser: "hermes",
            mode: Mode::MarkerBased,
            sequences: vec![
                s(&[
                    "<tool",
                    "_call>",
                    r#"{"name":"ws_c","arguments":{"k":3}}"#,
                    "</tool_call>",
                ]),
                s(&[" ", "  \n  ", "\t"]),
            ],
        },
        // A SECOND marker family. Every pair above is Hermes, so a defect in
        // shared scanner/marker state that only manifests for another family's
        // markers would never show up.
        Pair {
            name: "nemotron_deci_call_x_plain",
            parser: "nemotron_deci",
            mode: Mode::MarkerBased,
            sequences: vec![
                s(&[
                    "<TOOLCALL>",
                    r#"[{"name": "get_weather", "arguments": {"city": "Paris"}}]"#,
                    "</TOOLCALL>",
                ]),
                s(&["plain prose ", "for nemotron, ", "never jailed"]),
            ],
        },
        // tool_choice=required: the jail starts ALREADY jailed
        // (`JailMode::Immediate`), a different per-choice state machine from the
        // marker path — `get_or_create_state(index, starts_jailed = true)`.
        Pair {
            name: "required_mode_two_calls",
            parser: "hermes",
            mode: Mode::ToolChoiceRequired,
            sequences: vec![
                s(&[
                    r#"[{"name": "get_weather", "#,
                    r#""parameters": {"city": "Paris"}}]"#,
                ]),
                s(&[
                    r#"[{"name": "get_time", "#,
                    r#""parameters": {"tz": "UTC"}}]"#,
                ]),
            ],
        },
        // tool_choice=named: Immediate mode with a SingleObject format.
        Pair {
            name: "named_mode_two_objects",
            parser: "hermes",
            mode: Mode::ToolChoiceNamed("get_weather"),
            sequences: vec![
                s(&[r#"{"city": "#, r#""Paris"}"#]),
                s(&[r#"{"city": "#, r#""Tokyo"}"#]),
            ],
        },
    ]
}

/// Schedules for a k=2 pair. `BoundarySplit` is swept over BOTH victims and
/// several split ratios: a fixed "split choice 0 at the midpoint" cannot see a
/// parser that only breaks when the sibling is split, or one that breaks at a
/// non-midpoint boundary such as `<tool_ | call>`.
fn schedules_k2() -> Vec<Schedule> {
    let mut v = vec![
        Schedule::RoundRobin,
        Schedule::FirstByteOffset(1),
        Schedule::FirstByteOffset(2),
    ];
    for victim in [0u32, 1] {
        for (num, den) in [(1usize, 4usize), (1, 2), (3, 4)] {
            v.push(Schedule::BoundarySplit { victim, num, den });
        }
    }
    v
}

/// Whether a schedule can be meaningfully applied to `sequences`. Un-applicable
/// shapes are logged as skipped rather than silently counted as passing.
fn applicable(sequences: &[Vec<String>], schedule: Schedule) -> Result<(), String> {
    if sequences.iter().any(|s| s.is_empty()) {
        return Err("a choice has no deltas".to_string());
    }
    match schedule {
        Schedule::BoundarySplit { victim, .. } => {
            if sequences.len() != 2 {
                return Err("BoundarySplit requires exactly two choices".to_string());
            }
            if sequences[victim as usize]
                .iter()
                .all(|d| d.chars().count() < 2)
            {
                return Err(format!("choice {victim} has no splittable delta"));
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

#[tokio::test]
async fn jail_interleave_preserves_per_choice_isolation() {
    let mut ran = 0usize;
    let mut skipped = 0usize;
    let mut failures: Vec<String> = Vec::new();

    for pair in divergent_pairs() {
        assert_eq!(pair.sequences.len(), 2, "k=2 pairs only in this loop");
        // Both ROLE ASSIGNMENTS. The schedules give choice 0 the first slot of
        // every round, so running only (A@0, B@1) never exercises B arriving
        // first — a router that mishandles the higher-index choice leading would
        // pass. Swapping is the cheapest way to cover both arrival orders.
        for (role, sequences) in [
            (
                "AB",
                vec![pair.sequences[0].clone(), pair.sequences[1].clone()],
            ),
            (
                "BA",
                vec![pair.sequences[1].clone(), pair.sequences[0].clone()],
            ),
        ] {
            for schedule in schedules_k2() {
                if let Err(reason) = applicable(&sequences, schedule) {
                    eprintln!(
                        "SKIP pair={}/{} mode={} schedule={}: {reason}",
                        pair.name,
                        role,
                        pair.mode.label(),
                        schedule.label()
                    );
                    skipped += 1;
                    continue;
                }
                ran += 1;

                // Golden = the solo run of THIS choice's DEMUXED subsequence, not
                // of the original sequence. A splitting schedule re-chunks the
                // victim's own deltas, so the victim legitimately emits more
                // chunks than an unsplit run would; comparing against the original
                // sequence would flag that as a divergence. Feeding solo the same
                // post-split deltas keeps `emission_profile` an apples-to-apples
                // comparison whose ONLY remaining variable is the sibling's
                // presence on the wire — which is exactly the property under test.
                let items = demux_items(&interleave_items(&sequences, schedule));
                let golden_a = solo_at(pair.parser, pair.mode, items.get(&0).unwrap(), 0).await;
                let golden_b = solo_at(pair.parser, pair.mode, items.get(&1).unwrap(), 1).await;

                let demuxed =
                    interleaved_by_choice(pair.parser, pair.mode, &sequences, schedule).await;
                for (index, golden) in [(0u32, &golden_a), (1u32, &golden_b)] {
                    match demuxed.get(&index) {
                        Some(got) if got == golden => {}
                        Some(got) => failures.push(format!(
                            "schedule={} pair={}/{} mode={} choice={index} diverged:\n     got  {got:?}\n     want {golden:?}",
                            schedule.label(),
                            pair.name,
                            role,
                            pair.mode.label(),
                        )),
                        None => failures.push(format!(
                            "schedule={} pair={}/{} mode={} choice={index}: no output demuxed",
                            schedule.label(),
                            pair.name,
                            role,
                            pair.mode.label(),
                        )),
                    }
                }
            }
        }
    }

    eprintln!("v1 jail interleave: {ran} schedule-pairs ran, {skipped} skipped");
    // 8 pairs x 2 roles x 9 schedules, minus shapes with no splittable victim.
    // Guards against a pair, role, mode or schedule silently dropping out and
    // leaving a vacuously-passing matrix.
    assert!(
        ran >= 120,
        "expected the divergent-pair matrix to run, got {ran}"
    );
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// Group a tagged stream into PACKED chunks: consecutive items are merged into one
/// chunk until an index would repeat, then the chunk is flushed. Round-robin input
/// therefore yields chunks carrying every choice at once — the `Packed` emission
/// shape, rather than one choice per chunk.
fn pack_rounds(tagged: &[(u32, String)]) -> Vec<Vec<(u32, String)>> {
    let mut chunks: Vec<Vec<(u32, String)>> = Vec::new();
    let mut current: Vec<(u32, String)> = Vec::new();
    for (index, item) in tagged {
        if current.iter().any(|(i, _)| i == index) {
            chunks.push(std::mem::take(&mut current));
        }
        current.push((*index, item.clone()));
    }
    if !current.is_empty() {
        chunks.push(current);
    }
    chunks
}

/// Packed multi-choice chunks with NON-CONTIGUOUS, UNSORTED `choice.index` values.
///
/// Two blind spots this closes, both found and closed in the sibling PR
/// ai-dynamo/dynamo#11563:
///
/// 1. **Packed chunks.** Every other lane here emits one choice per chunk, so a
///    jail that processed only `choices[0]` of each chunk, or that carried
///    per-chunk (rather than per-choice) scratch state across the inner loop,
///    would still pass.
/// 2. **Non-contiguous / unsorted indices.** `ChoiceJailStateCollection` stores
///    states in a `Vec` kept sorted by index and looks them up with
///    `binary_search_by_key`. With the usual contiguous `0..k` arriving in order,
///    a state keyed by *vector position* is indistinguishable from one keyed by
///    `choice.index` — position and index coincide. Here the indices are
///    `[u32::MAX, 7, 2]` and they arrive in descending order, so every new state
///    inserts at position 0 and shifts its predecessors: position and index now
///    disagree for every choice, and a position-keyed lookup returns another
///    choice's jail buffer. `u32::MAX` additionally pins the boundary value.
#[tokio::test]
async fn jail_interleave_packed_chunks_non_contiguous_indices() {
    let parser = "hermes";
    // Deliberately unsorted and non-contiguous, including the u32 boundary.
    const INDICES: [u32; 3] = [u32::MAX, 7, 2];

    let sequences = vec![
        s(&[
            "checking ",
            r#"<tool_call>{"name":"max_idx_call","arguments":{"who":"max"}}</tool_call>"#,
            " done",
        ]),
        s(&["plain prose for seven, ", "never jailed"]),
        s(&[
            "<tool",
            "_call>",
            r#"{"name":"two_idx_call","arguments":{"who":"two"}}"#,
            "</tool_call>",
        ]),
    ];

    for schedule in [
        Schedule::RoundRobin,
        Schedule::FirstByteOffset(1),
        Schedule::FirstByteOffset(2),
    ] {
        // Golden: each choice alone, at the SAME index it uses interleaved.
        let mut goldens: BTreeMap<u32, Assembled> = BTreeMap::new();
        for (pos, seq) in sequences.iter().enumerate() {
            goldens.insert(
                INDICES[pos],
                solo_at(parser, Mode::MarkerBased, seq, INDICES[pos]).await,
            );
        }

        // Interleave by position, then remap position -> real choice.index.
        let tagged: Vec<(u32, String)> = interleave_items(&sequences, schedule)
            .into_iter()
            .map(|(pos, item)| (INDICES[pos as usize], item))
            .collect();

        let packed = pack_rounds(&tagged);
        assert!(
            packed.iter().any(|c| c.len() > 1),
            "{}: expected at least one genuinely packed multi-choice chunk",
            schedule.label()
        );

        // Staggered terminators here too: each index's terminator rides its own
        // last delta, which for the shorter choices lands mid-wire.
        let mut totals: BTreeMap<u32, usize> = BTreeMap::new();
        for (idx, _) in &tagged {
            *totals.entry(*idx).or_default() += 1;
        }
        let mut seen: BTreeMap<u32, usize> = BTreeMap::new();
        let chunks: Vec<_> = packed
            .iter()
            .map(|group| {
                let entries: Vec<(u32, String, Option<FinishReason>)> = group
                    .iter()
                    .map(|(idx, item)| {
                        let n = seen.entry(*idx).or_default();
                        *n += 1;
                        let fin = (*n == totals[idx]).then_some(FinishReason::Stop);
                        (*idx, item.clone(), fin)
                    })
                    .collect();
                packed_chunk(&entries)
            })
            .collect();
        let results: Vec<_> = build_jail(parser, Mode::MarkerBased)
            .apply_with_finish_reason(stream::iter(chunks))
            .collect()
            .await;

        // Demux by `choice.index`, never by arrival order or slot position.
        let mut per_choice: BTreeMap<u32, Vec<&ChatChoiceStream>> = BTreeMap::new();
        for choice in results
            .iter()
            .filter_map(|r| r.data.as_ref())
            .flat_map(|d| d.choices.iter())
        {
            per_choice.entry(choice.index).or_default().push(choice);
        }

        let mut failures = Vec::new();
        for index in INDICES {
            let golden = &goldens[&index];
            match per_choice.get(&index) {
                Some(chunks) => {
                    let got = assemble(chunks);
                    if &got != golden {
                        failures.push(format!(
                            "schedule={} packed/non-contiguous choice={index} diverged:\n     got  {got:?}\n     want {golden:?}",
                            schedule.label()
                        ));
                    }
                }
                None => failures.push(format!(
                    "schedule={} packed/non-contiguous choice={index}: no output demuxed",
                    schedule.label()
                )),
            }
        }
        // No index may be invented that was never sent (a position-keyed bug can
        // emit under a neighbour's index).
        for index in per_choice.keys() {
            assert!(
                INDICES.contains(index),
                "{}: jail emitted unknown choice.index {index}",
                schedule.label()
            );
        }
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }
}

/// One k=3 round-robin case: three choices (tool call / plain / tool call) share
/// one `JailedStream`; each must demux to its own solo golden.
#[tokio::test]
async fn jail_interleave_three_choices_round_robin() {
    let parser = "hermes";
    let sequences = vec![
        s(&[
            r#"<tool_call>{"name":"a_call","arguments":{"x":1}}</tool_call>"#,
            " after a",
        ]),
        s(&["plain ", "middle ", "content"]),
        s(&[
            "prefix ",
            r#"<tool_call>{"name":"c_call","arguments":{"y":2}}</tool_call>"#,
        ]),
    ];

    let goldens: Vec<Assembled> = {
        let mut v = Vec::new();
        // Golden at the SLOT each shape occupies, not always slot 0: comparing a
        // slot-0 reference against slots 1 and 2 would misattribute (or hide) a
        // defect whose behaviour depends on the choice index.
        for (i, seq) in sequences.iter().enumerate() {
            v.push(solo_at(parser, Mode::MarkerBased, seq, i as u32).await);
        }
        v
    };

    let demuxed =
        interleaved_by_choice(parser, Mode::MarkerBased, &sequences, Schedule::RoundRobin).await;
    let mut failures = Vec::new();
    for (index, golden) in goldens.iter().enumerate() {
        let index = index as u32;
        match demuxed.get(&index) {
            Some(got) if got == golden => {}
            Some(got) => failures.push(format!(
                "k=3 RoundRobin choice={index} diverged:\n     got  {got:?}\n     want {golden:?}"
            )),
            None => failures.push(format!("k=3 RoundRobin choice={index}: no output demuxed")),
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
