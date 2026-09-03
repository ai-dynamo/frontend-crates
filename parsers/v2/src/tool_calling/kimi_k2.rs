// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Streaming tool-call parser for Kimi K2.
//!
//! Kimi K2 emits tool calls as
//!   `<|tool_calls_section_begin|>`
//!     `<|tool_call_begin|>functions.NAME:IDX<|tool_call_argument_begin|>{JSON}<|tool_call_end|>`
//!     ... (one or more calls)
//!   `<|tool_calls_section_end|>`
//! The model may also emit singular section variants
//! (`<|tool_call_section_begin|>` / `<|tool_call_section_end|>`), and may drop
//! `section_end` entirely on max_tokens / EOS truncation.
//!
//! The streaming concern (buffering, chunk-split marker safety, normal_text
//! suppression) is owned by the shared [`scan::WrappedBlockScanner`]; the K2
//! grammar maps onto it with section variants as multi-token block markers,
//! the inner `call_end`/`argument_begin` markers as extra orphan markers, and
//! two K2-specific spec fields: the suppression latch engages after every
//! in-section call parse even when the call is malformed (`InvokeLatch::Always`),
//! and a call whose `call_end` never arrives before the section close is
//! dropped rather than swallowing the fence (`drop_invoke_crossing_block_end`).
//!
//! The per-call typing (function-id parsing, JSON validation, raw-string
//! fallback for malformed args) is delegated to the v1 batch parser
//! `try_tool_call_parse_kimi_k2` driven by the same `KimiK2ParserConfig`
//! `dynamo_parsers` uses for batch parsing, so a streamed call matches exactly
//! what the batch parser produces. A complete call is wrapped in the section
//! markers before delegating so the v1 parser always takes its normal section
//! path.
//!
//! The per-call arguments are already a JSON object string, so no key-order
//! reserialization is needed (unlike the XML families): the v1 parser
//! round-trips compact JSON byte-for-byte and falls back to the raw string for
//! malformed payloads, which is exactly what the fixtures expect.

use crate::tool_calling::scan::{
    BareRecoveryLatch, InvokeBoundaryFactory, InvokeEmitter, InvokeLatch, InvokeScan,
    WrappedBlockScanner, WrappedBlockSpec, find_first_outside_strings, json_value_end,
};
use crate::tool_calling::v1core::{
    KimiK2ParserConfig, ToolDefinition, try_tool_call_parse_kimi_k2,
};

use crate::tool_calling::traits::{Tool, ToolCallDelta, ToolParseResult, ToolParser};

// Mirror `KimiK2ParserConfig::default()` (the only config `kimi_k2_scanner`
// ever builds). `InvokeScan`'s hooks are plain `fn` pointers, not closures, so
// they cannot borrow a per-instance config; hardcoding the same defaults here
// is the existing pattern other `invoke_scan` families (e.g. gemma4) follow.
const CALL_START: &str = "<|tool_call_begin|>";
const CALL_END: &str = "<|tool_call_end|>";
const ARGUMENT_BEGIN: &str = "<|tool_call_argument_begin|>";

// Mirrors `KimiK2ParserConfig::default().section_end_variants` for the same
// reason as the consts above -- `kimi_invoke_end` needs to recognize a real
// section close to distinguish it from genuine EOS truncation (see its use
// below).
const SECTION_END_PLURAL: &str = "<|tool_calls_section_end|>";
const SECTION_END_SINGULAR: &str = "<|tool_call_section_end|>";

const FUNCTIONS_PREFIX: &str = "functions.";

/// Bytes valid in a `NAME:IDX` identifier, per `get_id_regex`'s `[\w.\-]+`.
fn ident_char(c: char) -> bool {
    c.is_alphanumeric() || matches!(c, '.' | '_' | '-')
}

/// Result of scanning for a `NAME:IDX` id at the start of a buffer.
enum NativeId {
    /// A complete id, with its length. At least one digit follows `:` --
    /// more digits streaming in later would only extend it, and every
    /// length already satisfies `\d+`, so there is no ambiguity to wait out.
    Complete(usize),
    /// What's buffered so far could still grow into a complete id (more
    /// name bytes, the `:` itself, or its digits) with more input.
    Pending,
    /// Terminates in a way that rules out `NAME:IDX` ever matching here.
    None,
}

/// Scan `text` for a complete `NAME:IDX` id (mirrors `get_id_regex`'s
/// `[\w.\-]+:\d+`), distinguishing "never going to match" from "not
/// determinable yet from what's buffered" -- the latter must wait for more
/// input rather than being treated as a bare, unindexed name. The
/// batch-path regex this mirrors is permissive by design (an unindexed name
/// is still valid prose, not a malformed id), which is exactly why this
/// cannot default to `None` just because a terminator hasn't streamed yet.
fn native_id_len(text: &str, flush: bool) -> NativeId {
    let ident_len = match text.find(|c: char| !ident_char(c)) {
        Some(i) => i,
        None if flush => text.len(),
        None => return NativeId::Pending,
    };
    if ident_len == 0 {
        return NativeId::None;
    }
    let rest = &text[ident_len..];
    let Some(after_colon) = rest.strip_prefix(':') else {
        return if !flush && rest.is_empty() {
            NativeId::Pending
        } else {
            NativeId::None
        };
    };
    match after_colon.find(|c: char| !c.is_ascii_digit()) {
        Some(0) => NativeId::None, // `:` immediately followed by a non-digit
        Some(d) => NativeId::Complete(ident_len + 1 + d),
        None if after_colon.is_empty() => {
            if flush {
                NativeId::None
            } else {
                NativeId::Pending
            }
        }
        None => NativeId::Complete(ident_len + 1 + after_colon.len()),
    }
}

/// Locate the end of one Kimi invoke (`call_start .. call_end`) by finding
/// where its JSON argument body actually closes first, rather than searching
/// the raw buffer for `call_end` from byte zero.
///
/// Two things follow from that ordering:
/// - A `call_end`-looking byte sequence embedded INSIDE the JSON string
///   argument is data, not the real closer (`UNIFIED.7.b`,
///   `arg_marker_in_string`) — the naive whole-buffer search matched the
///   embedded copy first and truncated the argument there.
/// - At true EOF (`flush`), a body whose JSON is syntactically complete but
///   whose `call_end` never streamed (max_tokens / EOS) is recoverable
///   (`UNIFIED.5.b`, `tool_no_close`, the same best-effort-recovery contract
///   as policy P2) instead of being dropped as if it were genuinely
///   truncated. `K2Emitter` synthesizes the missing closer before typing it.
///   BUT only for `tool_index == 0`: the captured batch contract
///   (`TOOLCALLING.batch.5.d`/`TOOLCALLING.streamv2.5.d`, independently
///   pinned against the real `parse_section_block` regex and its own
///   streaming golden capture) drops a later call that never gets its own
///   `call_end`, while still recovering an incomplete FIRST call at EOF
///   (`TOOLCALLING.batch.5.a`/`UNIFIED.tool_no_close`). This mirrors that
///   observed asymmetry; it is not a claim about model reliability beyond
///   what these two fixtures establish. `tool_index` is the same monotonic
///   per-stream counter `WrappedBlockScanner` already tracks for
///   `tool_call_id`.
fn kimi_invoke_end(text: &str, flush: bool, tool_index: usize) -> Option<usize> {
    // The first `argument_begin` in `text` only belongs to THIS invoke
    // (the one starting at byte 0) if nothing closes the invoke before it.
    // Model output is probabilistic and can violate its own grammar --
    // a bare `NAME:IDX<|tool_call_end|>` with no argument section at all,
    // immediately followed by a real second invoke that DOES have one. An
    // unbounded search matched the second invoke's `argument_begin` to the
    // first invoke's span, merging both into one string and silently
    // dropping the first call. Already-buffered bytes before a found
    // `argument_begin` can't be invalidated by more input streaming in
    // later, so this bound is safe to apply immediately, not just at
    // `flush`.
    let args_at = text.find(ARGUMENT_BEGIN).and_then(|pos| {
        // Either sibling marker before the found `argument_begin` proves it
        // belongs to a LATER invoke, not this one: a `call_end` means this
        // invoke already closed with no argument section; a second
        // `call_start` means a new invoke opened before this one ever
        // reached its own `argument_begin` (this one has neither a
        // `call_end` NOR an `argument_begin` of its own). Checking only the
        // `call_end` half left the `call_end`-less variant of the same
        // malformed shape unguarded -- currently masked by the downstream
        // regex's own forgiving `captures_iter` and the JSON-argument
        // branch's `CALL_START` bound below, not by this check actually
        // being correct, so a future change to either of those could silently
        // revive the merge.
        let belongs_to_later_invoke =
            text[..pos].contains(CALL_END) || text[CALL_START.len()..pos].contains(CALL_START);
        if belongs_to_later_invoke && flush {
            // Logged only at `flush` (this check re-runs on every call while
            // streaming, but the bare-close case can only finish resolving
            // once no more input is coming -- see the `remainder` check
            // below) so this fires exactly once per malformed invoke, not
            // once per push.
            tracing::warn!(
                why = "kimi_k2_invoke_closed_before_argument_begin",
                "stream dropped a bare invoke with no argument section of its own; \
                 a later argument_begin belongs to a different invoke"
            );
        }
        (!belongs_to_later_invoke).then_some(pos + ARGUMENT_BEGIN.len())
    });
    // No `argument_begin` at all: this isn't (yet, or ever) a well-formed
    // `call_start .. argument_begin .. json .. call_end` invoke -- e.g. the
    // guided-decoding native-markup-leak scenarios, where the buffer holds
    // `call_start` grammar tokens but never a real argument section.
    let Some(args_at) = args_at else {
        // A `call_end` found here is NOT reliable evidence until `flush`:
        // streaming only ever APPENDS bytes, so a legitimate `argument_begin`
        // that hasn't arrived YET can still turn up later and take priority
        // over this reading. Committing early made the result depend on
        // where the chunk boundary happened to land -- the exact same bytes
        // parsed to a dropped call in one push and a leaked raw-JSON `Text`
        // in two, for identical final input. Only trust this reading once
        // no more input is coming.
        //
        // Same bound as the `argument_begin` check above: a `call_end`
        // preceded by a SECOND `call_start` belongs to a later invoke, not
        // this one (this invoke never closed at all before the next one
        // opened) -- trusting it merged both spans the same way an
        // unbounded `argument_begin` search did.
        if flush
            && let Some(end) = text.find(CALL_END).map(|pos| pos + CALL_END.len())
            && !text[CALL_START.len()..end].contains(CALL_START)
        {
            return Some(end);
        }
        // No `argument_begin` AND no `call_end` (or not `flush` yet): not a
        // well-formed native invoke, and never going to become one from more
        // `call_end` bytes arriving -- e.g. a narrated `<|tool_call_begin|>`
        // header the model wrote while guided decoding actually constrained
        // the payload, with
        // nothing after it but bare JSON, a reasoning marker, or truncated
        // header text (`guided_json_*_bare_opener`,
        // `guided_json_narrated_prefix_inside_reasoning`,
        // `guided_json_stray_prefix_before_reasoning`). Waiting forever for
        // delimiters that will never come left the whole header + payload
        // leaking as visible text.
        //
        // Bound the invoke to the STRUCTURAL part only: the literal
        // `functions.` prefix, plus a complete `NAME:IDX` id when one
        // actually follows it. A bare name with no index
        // (`functions.get_weather` narrated inside a thought,
        // `guided_json_narrated_prefix_inside_reasoning`) is prose the model
        // wrote, not a real id -- swallowing it as control markup drops it
        // from the reasoning text the golden oracle expects it to survive
        // in. Whatever isn't consumed here -- JSON, `<think>`, a bare name,
        // or nothing -- is scanned fresh on its own terms.
        let after_start = &text[CALL_START.len()..];
        let Some(after_prefix) = after_start.strip_prefix(FUNCTIONS_PREFIX) else {
            // Not (yet) the literal `functions.` prefix. If what's buffered
            // is a proper prefix of it, more input could still complete the
            // match -- wait rather than deciding early.
            if !flush
                && after_start.len() < FUNCTIONS_PREFIX.len()
                && FUNCTIONS_PREFIX.starts_with(after_start)
            {
                return None;
            }
            // Genuinely not the expected shape: nothing here is header
            // markup, so bound the invoke to the bare opener marker itself.
            return Some(CALL_START.len());
        };
        let end = match native_id_len(after_prefix, flush) {
            NativeId::Complete(id_len) => CALL_START.len() + FUNCTIONS_PREFIX.len() + id_len,
            NativeId::Pending => return None,
            NativeId::None => CALL_START.len() + FUNCTIONS_PREFIX.len(),
        };
        // The byte right after `end` may be the start of a real
        // `argument_begin` or `call_end` that just hasn't finished
        // streaming (both begin with `<`, which never matches
        // `functions.`/`ident_char`). Both searches above already proved
        // neither marker exists in full yet, so a match here can only be a
        // genuine partial -- wait for it rather than prematurely bounding
        // the header.
        let remainder = &text[end..];
        // Reviewer-caught regression: the Kimi batch grammar permits `\s*`
        // between `NAME:IDX` and `argument_begin` (the regex in
        // `get_tool_call_regex` matches `\s*` there too), but the two
        // holdback checks below used `remainder` verbatim -- a chunk split
        // landing right after that permitted whitespace (`remainder == " "`)
        // matched neither marker's prefix, so this function committed to
        // `Some(end)` one push early, before `argument_begin` streamed in.
        // The caller then bounds the invoke to a header-only span with no
        // argument section at all, and the real call is silently lost
        // (`K2Emitter` can't parse a header with no `argument_begin`).
        // Deciding over the whitespace-trimmed view (never emitting or
        // discarding that whitespace -- it stays buffered either way, since
        // `end` doesn't move) closes this without weakening the check for
        // any non-whitespace byte.
        let structural_remainder = remainder.trim_start();
        // A COMPLETE `call_end` right here is the one case that is still
        // NOT settled: it could be this invoke's own (no-args) close, or it
        // could be a premature echo that a real `argument_begin` further
        // downstream will supersede once more input streams in -- streaming
        // only ever appends, so that later marker cannot be ruled out yet.
        // Read this the same way the pure `call_end`-only branch above does:
        // trust it only once nothing more is coming (`flush`). Same class of
        // bug either commit destroyed -- reading it early made the outcome
        // depend on the chunk boundary instead of the bytes.
        if !flush && structural_remainder.starts_with(CALL_END) {
            return None;
        }
        if !flush
            && [ARGUMENT_BEGIN, CALL_END].iter().any(|marker| {
                structural_remainder.len() < marker.len()
                    && marker.starts_with(structural_remainder)
            })
        {
            return None;
        }
        return Some(end);
    };
    // From here the shape has a real `argument_begin`, so ownership of the
    // closer search transfers to the JSON boundary (`I7`) when the argument
    // body IS balanced JSON -- never fall back to a raw literal search of
    // the (possibly still-streaming) buffer in that case, which would
    // re-match a `call_end`-looking byte sequence still sitting inside the
    // not-yet-closed argument string.
    let after_args = &text[args_at..];
    // `json_value_end` only proves bracket/quote NESTING is balanced, not
    // that the bytes are valid JSON (`{not-json}` reads as "balanced" --
    // braces match, zero quotes to mistrack -- but isn't a legal JSON
    // value; likewise `{<|tool_calls_section_end|>}` balances even though
    // its "content" is a section-end marker, not JSON). Every downstream
    // branch below this point (the well-formed `call_end` search, the
    // best-effort EOF recovery) assumed a `json_value_end` success meant
    // "this is real JSON" and never re-checked -- a bracket-balanced-but-
    // invalid body could still slip through, its embedded section-end
    // marker never even scanned for, since the code trusted `json_len` as
    // the argument's real boundary. Validating HERE, before any of those
    // branches run, means an invalid body falls through to the SAME
    // malformed/raw-string fallback below (with its own intervening-
    // section-end guard) instead of taking the well-formed path at all --
    // one owner for "is this argument actually usable as JSON", not a
    // patchwork of per-branch checks.
    let valid_json_len = json_value_end(after_args).filter(|&json_len| {
        serde_json::from_str::<serde_json::Value>(&after_args[..json_len]).is_ok()
    });
    let Some(json_len) = valid_json_len else {
        // `json_value_end` returning `None` does NOT mean "malformed" --
        // most of the time it means "not balanced YET", e.g. a chunk split
        // lands mid-string with a `call_end`-looking byte sequence sitting
        // inside the still-open quote (`UNIFIED.7.b`, `arg_marker_in_string`).
        // Falling back to a raw `call_end` search there re-matches that
        // EMBEDDED fake closer and truncates the argument -- exactly the I7
        // corruption this whole JSON-boundary approach exists to prevent.
        // Only at true EOF, once no more input can possibly arrive to
        // balance it, is "never resolves to JSON" a safe conclusion.
        //
        // At that point `parse_section_block` (the batch-mode typing layer
        // this module's own doc promises byte-parity with) has a
        // raw-string fallback for exactly this: when `serde_json::from_str`
        // fails, it ships the raw text verbatim instead of rejecting the
        // call. Propagating `None` unconditionally skipped that fallback
        // entirely -- the whole invoke never reached the typing layer, so
        // the SAME bytes that recover as a call with a raw-string argument
        // in batch mode silently vanished in streaming mode. If the
        // family's own literal `call_end` is already present, bound the
        // invoke there (same `call_start` bound as every other closer
        // search in this function) and let the raw text through to that
        // fallback, rather than deciding here that it can never be
        // recovered.
        // In malformed input, quote state is not a trustworthy owner across
        // invoke boundaries. An unmatched quote in this call can hide the
        // next call's opener, then a second unmatched quote can restore the
        // scanner state and make that later call's closer look structural.
        // Treat the earliest raw opener as a hard damage boundary before
        // accepting any quote-aware closer; valid JSON took the branch below
        // and therefore keeps marker-looking bytes inside strings as data.
        let next_call_start = after_args.find(CALL_START);
        let structural_call_end = find_first_outside_strings(after_args, [CALL_END])
            .map(|(position, _)| position)
            .filter(|position| next_call_start.is_none_or(|next| *position < next));
        let raw_call_end = after_args.find(CALL_END);
        let bounded_raw_call_end =
            raw_call_end.filter(|position| next_call_start.is_some_and(|next| *position < next));
        let (rel, markers_are_structural) = match structural_call_end {
            Some(position) => (position, true),
            // A later raw opener makes the earlier raw closer stable before
            // EOF: future bytes belong to the next invoke and cannot turn
            // this closer into quoted data for the current one.
            None if bounded_raw_call_end.is_some() => (bounded_raw_call_end?, false),
            None if flush => (raw_call_end?, false),
            None => return None,
        };
        if next_call_start.is_none_or(|next| rel < next) {
            // Mirror the well-formed-JSON sibling's section-end guard
            // below: a real section-end marker occurring before this
            // literal `call_end` means the model explicitly closed the
            // whole tool_calls section without ever giving THIS call its
            // own `call_end` ("mismatched fences"), same as the sibling
            // case, just discovered via the malformed/raw-string path
            // instead of the well-formed-JSON path. Recovering here would
            // swallow the section-end marker into this invoke's own
            // malformed argument and hide the section boundary from every
            // downstream check that trusts this returned position --
            // reproduced directly: `{"location": "unterminated<section_end>`
            // followed by a literal `call_end` recovered a call whose raw
            // argument absorbed the section-end marker as text.
            let before_call_end = &after_args[..rel];
            let section_end_intervenes = if markers_are_structural {
                find_first_outside_strings(
                    before_call_end,
                    [SECTION_END_PLURAL, SECTION_END_SINGULAR],
                )
                .is_some()
            } else {
                [SECTION_END_PLURAL, SECTION_END_SINGULAR]
                    .iter()
                    .any(|marker| before_call_end.contains(marker))
            };
            if !section_end_intervenes {
                return Some(args_at + rel + CALL_END.len());
            }
        }
        return None;
    };
    let json_end = args_at + json_len;
    let after_json = &text[json_end..];
    if let Some(rel) = after_json.find(CALL_END) {
        // Bound the search: a new invoke opening before this one's own
        // closer means this invoke never closed. Reaching past the new
        // opener to grab some LATER invoke's `call_end` merged both calls'
        // bytes into one corrupted invoke and silently dropped the second
        // call entirely. Fall through to the same best-effort recovery the
        // missing-closer case already uses, so the first call still ships
        // (JSON is complete) and the second is scanned as its own invoke.
        if after_json.find(CALL_START).is_none_or(|next| rel < next) {
            return Some(json_end + rel + CALL_END.len());
        }
    }
    // Best-effort recovery (`UNIFIED.5.b`, policy P2 sibling): the argument
    // body is syntactically complete but the model stopped before emitting
    // the closer. Only at true EOF -- otherwise wait for more input.
    //
    // But NOT when a real section-end marker follows instead of more input
    // running out: that's not truncation, it's the model explicitly closing
    // the whole tool_calls section without ever giving THIS call its own
    // `call_end` -- "mismatched fences" (`TOOLCALLING.batch.4.d`, sourced
    // from vLLM's own kimi_k2 parser tests). `parse_section_block`'s regex
    // (the batch-mode typing layer this module promises byte-parity with)
    // has no fallback for a missing `call_end` regardless of what follows
    // it, so recovering here would ship a call batch mode drops -- exactly
    // the divergence `conformance_toolcalling_batch_via_stream` caught.
    if flush {
        let trimmed_after_json = after_json.trim_start();
        if [SECTION_END_PLURAL, SECTION_END_SINGULAR]
            .iter()
            .any(|marker| trimmed_after_json.starts_with(marker))
        {
            return None;
        }
    }
    // `tool_index == 0` only (see the doc comment above): a later call with
    // no evidence it ever closes is a malformed shape, not truncation, once
    // an earlier call in the same response DID close correctly. `json_len`
    // is already proven valid JSON at this point (the hoisted
    // `serde_json::from_str` check above `valid_json_len` covers this
    // whole function, not just this one branch), so no separate
    // JSON-validity check is needed here.
    (flush && tool_index == 0).then_some(json_end)
}

/// Kimi's `call_start` marker is unambiguous wherever it appears; every
/// occurrence opens a real invoke (same effective behavior as the
/// marker-only path this hook replaces).
fn kimi_invoke_opens(_text: &str, _at: usize) -> bool {
    true
}

/// No additional holdback beyond the generic marker holdback: `call_end` and
/// `argument_begin` are already in `holdback_markers`, which already retains
/// a partial marker split across a chunk boundary.
fn kimi_invoke_holdback(_text: &str) -> usize {
    0
}

const KIMI_INVOKE_SCAN: InvokeScan = InvokeScan {
    end: kimi_invoke_end,
    opens: kimi_invoke_opens,
    holdback: kimi_invoke_holdback,
    resync: None,
};

fn spec(config: &KimiK2ParserConfig) -> WrappedBlockSpec {
    // Orphan markers: inner markers (`call_end`, `argument_begin`) and every
    // section-end variant only appear legitimately inside an open section;
    // outside one they are stray grammar markup to be stripped. Mirrors the v1
    // batch parser's `first_orphan_kimi_marker_index` (minus `call_start`,
    // which the bare-call recovery path already opens).
    let mut orphan_markers = vec![config.call_end.clone(), config.argument_begin.clone()];
    orphan_markers.extend(config.section_end_variants.clone());

    // Every grammar marker that must never be split-leaked as normal_text.
    let mut holdback_markers = config.section_start_variants.clone();
    holdback_markers.extend(config.section_end_variants.clone());
    holdback_markers.push(config.call_start.clone());
    holdback_markers.push(config.call_end.clone());
    holdback_markers.push(config.argument_begin.clone());

    WrappedBlockSpec {
        family: "kimi_k2",
        block_starts: config.section_start_variants.clone(),
        block_ends: config.section_end_variants.clone(),
        invoke_start: config.call_start.clone(),
        invoke_end: config.call_end.clone(),
        orphan_markers,
        holdback_markers,
        bare_recovery_latch: BareRecoveryLatch::Set,
        invoke_latch: InvokeLatch::Always,
        drop_invoke_crossing_block_end: true,
        // Every wrapped family's markers are special tokens today.
        preserve_special_tokens: true,
        invoke_boundary_factory: Some(InvokeBoundaryFactory::stateless(KIMI_INVOKE_SCAN)),
    }
}

#[cfg(test)]
mod boundary_tests {
    use super::*;

    #[test]
    fn kimi_exposes_its_stateless_boundary_as_family_metadata() {
        let scanner = kimi_k2_scanner(&[]);
        let factory = scanner
            .invoke_boundary_factory()
            .expect("Kimi needs grammar-aware boundary callbacks");
        let boundary = factory.create();

        assert!(boundary.opens(CALL_START, 0));
        assert_eq!(boundary.holdback("ordinary text"), 0);
    }
}

/// Value-typing hook: wraps one complete
/// `<|tool_call_begin|>...<|tool_call_end|>` call in the section markers so
/// the v1 parser takes its normal section path, then emits `name` + JSON
/// `arguments` as one delta.
pub(crate) struct K2Emitter {
    config: KimiK2ParserConfig,
    tools: Vec<ToolDefinition>,
    /// Native `functions.NAME:IDX` id per `tool_index`, for
    /// [`InvokeEmitter::tool_call_id`]. The v1core parser already extracts
    /// this id (`ToolCallResponse::id`) to resolve the function name; Kimi's
    /// envelope is the only wrapped grammar that NAMES the call this way, so
    /// this is the one family that needs to remember it past `parse_invoke`.
    native_ids: Vec<Option<String>>,
}

impl InvokeEmitter for K2Emitter {
    fn parse_invoke(
        &mut self,
        invoke: &str,
        tool_index: usize,
    ) -> anyhow::Result<Option<ToolCallDelta>> {
        // `kimi_invoke_end` may hand back a call recovered at EOF whose JSON
        // body is complete but whose `call_end` never streamed (`UNIFIED.5.b`).
        // Normalize it here: the regex-based v1 parser requires the literal
        // closer to delimit the arguments capture, so synthesize it rather
        // than re-feeding the raw, still-unclosed bytes.
        let synthesized;
        let invoke = if invoke.ends_with(self.config.call_end.as_str()) {
            invoke
        } else {
            synthesized = format!("{invoke}{}", self.config.call_end);
            synthesized.as_str()
        };
        let wrapped = format!(
            "{}{}{}",
            self.config.section_start, invoke, self.config.section_end
        );
        let (calls, _content) =
            try_tool_call_parse_kimi_k2(&wrapped, &self.config, Some(&self.tools))?;
        let Some(parsed) = calls.into_iter().next() else {
            return Ok(None);
        };
        // `tool_index` is assigned by the caller in emission order (0, 1, 2,
        // ...), so a plain positional slot is enough — pad rather than
        // index-assign, since a dropped/malformed invoke ahead of this one
        // (`Ok(None)` above) never reserves a slot for itself.
        if self.native_ids.len() <= tool_index {
            self.native_ids.resize(tool_index + 1, None);
        }
        self.native_ids[tool_index] = Some(parsed.id);
        Ok(Some(ToolCallDelta {
            tool_index,
            name: Some(parsed.function.name),
            arguments: parsed.function.arguments,
        }))
    }

    fn tool_call_id(&self, tool_index: usize) -> Option<&str> {
        self.native_ids.get(tool_index)?.as_deref()
    }

    fn reset(&mut self) {
        self.native_ids.clear();
    }
}

/// Stream parser for Kimi K2 tool calls.
pub struct KimiK2ToolStreamParser {
    scanner: WrappedBlockScanner<K2Emitter>,
}

/// Build the Kimi K2 marker scanner for one stream.
///
/// Extracted so the tool-only parser and the unified adapter share ONE scanner
/// construction. Two constructions would be two grammars that drift.
pub(crate) fn kimi_k2_scanner(tools: &[Tool]) -> WrappedBlockScanner<K2Emitter> {
    let config = KimiK2ParserConfig::default();
    WrappedBlockScanner::new(
        spec(&config),
        K2Emitter {
            config,
            tools: tools.iter().map(ToolDefinition::from).collect(),
            native_ids: Vec::new(),
        },
    )
}

impl KimiK2ToolStreamParser {
    pub fn new(tools: &[Tool]) -> Self {
        Self {
            scanner: kimi_k2_scanner(tools),
        }
    }
}

impl ToolParser for KimiK2ToolStreamParser {
    fn create(tools: &[Tool]) -> anyhow::Result<Box<dyn ToolParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self::new(tools)))
    }

    fn preserve_special_tokens(&self) -> bool {
        self.scanner.preserve_special_tokens()
    }

    fn push(&mut self, chunk: &str) -> anyhow::Result<ToolParseResult> {
        self.scanner.push(chunk)
    }

    fn finish(&mut self) -> anyhow::Result<ToolParseResult> {
        self.scanner.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn weather_tools() -> Vec<Tool> {
        vec![Tool {
            name: "get_weather".to_string(),
            description: None,
            parameters: serde_json::json!({
                "type": "object",
                "properties": { "location": { "type": "string" } }
            }),
            strict: None,
        }]
    }

    fn parse_chunks(tools: &[Tool], chunks: &[&str]) -> ToolParseResult {
        let mut parser = KimiK2ToolStreamParser::new(tools);
        let mut out = ToolParseResult::default();
        for chunk in chunks {
            out.append(parser.push(chunk).expect("push"));
        }
        out.append(parser.finish().expect("finish"));
        out
    }

    #[test]
    fn hardcoded_markers_mirror_the_config_default() {
        // `CALL_START`/`CALL_END`/`ARGUMENT_BEGIN`/`SECTION_END_PLURAL`/
        // `SECTION_END_SINGULAR` all exist only because `InvokeScan`'s hooks
        // are plain `fn` pointers and cannot borrow a per-instance config
        // (see the comment on `CALL_START` above). `KimiK2ParserConfig::
        // default()` is the one real owner of these strings; this test is
        // the parity check that fails loudly if a future config change
        // silently stops matching these mirrors, instead of `kimi_invoke_end`
        // quietly scanning for the wrong bytes.
        let config = KimiK2ParserConfig::default();
        assert_eq!(config.call_start, CALL_START);
        assert_eq!(config.call_end, CALL_END);
        assert_eq!(config.argument_begin, ARGUMENT_BEGIN);
        assert_eq!(
            config.section_end_variants,
            vec![
                SECTION_END_PLURAL.to_string(),
                SECTION_END_SINGULAR.to_string()
            ],
        );
    }

    #[test]
    fn section_end_marker_embedded_in_a_string_argument_is_data_not_a_boundary() {
        // A well-formed call whose own JSON string argument happens to
        // contain the literal bytes of the section-end marker (e.g. echoing
        // a shell command) must NOT be mistaken for the real block boundary.
        // The shared `drop_invoke_crossing_block_end` safety net used a raw,
        // non-string-aware search that matched this embedded copy and
        // dropped the whole call, leaking its JSON tail as garbage text and
        // corrupting the `tool_index` of the following call (`next_index`
        // never advanced for the dropped one).
        let out = parse_chunks(
            &weather_tools(),
            &[
                "<|tool_calls_section_begin|><|tool_call_begin|>functions.run:0<|tool_call_argument_begin|>",
                "{\"cmd\":\"echo <|tool_calls_section_end|>\"}<|tool_call_end|>",
                "<|tool_call_begin|>functions.get_weather:1<|tool_call_argument_begin|>{\"location\":\"NYC\"}<|tool_call_end|><|tool_calls_section_end|>",
            ],
        );
        assert_eq!(out.normal_text, "");
        let merged = out.coalesce_calls();
        assert_eq!(merged.calls.len(), 2);
        assert_eq!(merged.calls[0].tool_index, 0);
        assert_eq!(merged.calls[0].name.as_deref(), Some("run"));
        assert_eq!(
            merged.calls[0].arguments,
            r#"{"cmd":"echo <|tool_calls_section_end|>"}"#
        );
        assert_eq!(merged.calls[1].tool_index, 1);
        assert_eq!(merged.calls[1].name.as_deref(), Some("get_weather"));
        assert_eq!(merged.calls[1].arguments, r#"{"location":"NYC"}"#);
    }

    #[test]
    fn emits_complete_call_on_close() {
        let out = parse_chunks(
            &weather_tools(),
            &[
                "<|tool_calls_section_begin|><|tool_call_begin|>",
                "functions.get_weather:0<|tool_call_argument_begin|>",
                "{\"location\":\"NYC\"}<|tool_call_end|><|tool_calls_section_end|>",
            ],
        );
        assert_eq!(out.normal_text, "");
        let merged = out.coalesce_calls();
        assert_eq!(merged.calls.len(), 1);
        assert_eq!(merged.calls[0].tool_index, 0);
        assert_eq!(merged.calls[0].name.as_deref(), Some("get_weather"));
        assert_eq!(merged.calls[0].arguments, r#"{"location":"NYC"}"#);
    }

    /// Direct unit test on the boundary finder itself, bypassing the
    /// scanner/typing pipeline entirely -- the pipeline-level symptom of this
    /// bug is easy to mask by accident (the downstream regex's own
    /// `captures_iter` happens to skip a malformed prefix and still find the
    /// valid second call on its own, and content between two invokes in one
    /// section is suppressed by unrelated, correct `InvokeLatch::Always`
    /// behavior regardless of this fix). The boundary VALUE is the actual
    /// contract: a bare `NAME:IDX<|tool_call_end|>` invoke with no
    /// `argument_begin` at all must bound to itself, not reach across a
    /// second invoke's `call_start` to grab that invoke's `argument_begin`.
    #[test]
    fn invoke_end_does_not_reach_across_a_bare_close_to_a_later_argument_begin() {
        let text = "<|tool_call_begin|>functions.run:0<|tool_call_end|><|tool_call_begin|>functions.get_weather:1<|tool_call_argument_begin|>{\"city\": \"Paris\"}<|tool_call_end|>";
        // The correct boundary is the bare invoke's OWN call_end, not just
        // its id -- that call_end genuinely belongs to it, and stopping
        // there (rather than reaching into the second invoke) is what makes
        // `find(CALL_END)` in the fallback above correctly resolve this case
        // on its own, without ever needing the native-id path below it.
        let bare_invoke_with_own_close = "<|tool_call_begin|>functions.run:0<|tool_call_end|>";
        assert_eq!(
            kimi_invoke_end(text, true, 0),
            Some(bare_invoke_with_own_close.len()),
            "must bound to the bare invoke's own close, not span through the second invoke's call_end"
        );
    }

    /// Sibling of the test above: a bare invoke that has NEITHER its own
    /// `argument_begin` NOR its own `call_end` -- its id text runs straight
    /// into a second invoke's `call_start`. The `argument_begin` bound only
    /// checked for an intervening `call_end`; this shape has none, so it was
    /// unguarded and the second invoke's `argument_begin`/JSON/`call_end`
    /// still got attributed to the first, merging both spans -- masked from
    /// producing a visibly wrong result only by the downstream regex's
    /// forgiving `captures_iter` and the JSON-argument branch's own
    /// `CALL_START` bound, not by this check being correct.
    #[test]
    fn invoke_end_does_not_reach_across_a_bare_open_with_no_close_at_all() {
        let text = "<|tool_call_begin|>functions.run:0<|tool_call_begin|>functions.get_weather:1<|tool_call_argument_begin|>{\"city\": \"Paris\"}<|tool_call_end|>";
        let bare_invoke_id_only = "<|tool_call_begin|>functions.run:0";
        assert_eq!(
            kimi_invoke_end(text, true, 0),
            Some(bare_invoke_id_only.len()),
            "must bound to the bare invoke's id alone (it has no close of its own), \
             not span through the second invoke's call_end"
        );
    }

    #[test]
    fn emits_two_calls_in_one_section() {
        let tools = vec![
            Tool {
                name: "get_weather".to_string(),
                description: None,
                parameters: serde_json::json!({"type": "object"}),
                strict: None,
            },
            Tool {
                name: "get_time".to_string(),
                description: None,
                parameters: serde_json::json!({"type": "object"}),
                strict: None,
            },
        ];
        let out = parse_chunks(
            &tools,
            &[
                "<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{\"location\":\"NYC\"}<|tool_call_end|>",
                "<|tool_call_begin|>functions.get_time:1<|tool_call_argument_begin|>{\"timezone\":\"EST\"}<|tool_call_end|><|tool_calls_section_end|>",
            ],
        );
        let merged = out.coalesce_calls();
        assert_eq!(merged.calls.len(), 2);
        assert_eq!(merged.calls[0].name.as_deref(), Some("get_weather"));
        assert_eq!(merged.calls[0].arguments, r#"{"location":"NYC"}"#);
        assert_eq!(merged.calls[1].name.as_deref(), Some("get_time"));
        assert_eq!(merged.calls[1].arguments, r#"{"timezone":"EST"}"#);
    }

    #[test]
    fn preserves_prefix_text_before_section() {
        let out = parse_chunks(
            &weather_tools(),
            &[
                "I will",
                " check the weather. <|tool_calls_section_begin|>",
                "<|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{\"location\":\"NYC\"}<|tool_call_end|><|tool_calls_section_end|>",
            ],
        );
        assert_eq!(out.normal_text, "I will check the weather. ");
        assert_eq!(out.coalesce_calls().calls.len(), 1);
    }

    #[test]
    fn preserves_post_section_narration() {
        let out = parse_chunks(
            &weather_tools(),
            &[
                "<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{\"location\":\"NYC\"}<|tool_call_end|><|tool_calls_section_end|>",
                " Done.",
            ],
        );
        // In-section markup is suppressed; post-section narration is preserved
        // verbatim once the section closes (v1 batch parity, cases 8.b/8.c).
        assert_eq!(out.normal_text, " Done.");
        assert_eq!(out.coalesce_calls().calls.len(), 1);
    }

    #[test]
    fn preserves_inter_section_narration() {
        // Two sections separated by narration (case 8.d): the prefix and the
        // inter-section text both flow into normal_text; both calls are emitted.
        let tools = vec![
            Tool {
                name: "get_weather".to_string(),
                description: None,
                parameters: serde_json::json!({"type": "object"}),
                strict: None,
            },
            Tool {
                name: "get_time".to_string(),
                description: None,
                parameters: serde_json::json!({"type": "object"}),
                strict: None,
            },
        ];
        let out = parse_chunks(
            &tools,
            &[
                "First. <|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{\"location\":\"NYC\"}<|tool_call_end|><|tool_calls_section_end|>",
                " Then. <|tool_calls_section_begin|><|tool_call_begin|>functions.get_time:1<|tool_call_argument_begin|>{\"timezone\":\"EST\"}<|tool_call_end|><|tool_calls_section_end|>",
            ],
        );
        assert_eq!(out.normal_text, "First.  Then. ");
        let merged = out.coalesce_calls();
        assert_eq!(merged.calls.len(), 2);
        assert_eq!(merged.calls[0].name.as_deref(), Some("get_weather"));
        assert_eq!(merged.calls[1].name.as_deref(), Some("get_time"));
    }

    #[test]
    fn holds_back_marker_split_across_every_char() {
        // Worst case: the whole input arrives one fragment at a time, splitting
        // every grammar marker. No partial marker may leak into normal_text.
        let full = "<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{\"location\":\"NYC\"}<|tool_call_end|><|tool_calls_section_end|>";
        let chunks: Vec<&str> = full
            .as_bytes()
            .chunks(3)
            .map(|c| std::str::from_utf8(c).unwrap())
            .collect();
        let out = parse_chunks(&weather_tools(), &chunks);
        assert_eq!(out.normal_text, "");
        let merged = out.coalesce_calls();
        assert_eq!(merged.calls.len(), 1);
        assert_eq!(merged.calls[0].name.as_deref(), Some("get_weather"));
        assert_eq!(merged.calls[0].arguments, r#"{"location":"NYC"}"#);
    }

    #[test]
    fn suppresses_truncated_call_at_eof() {
        // Section + call header streamed, but no call_end / section_end before
        // EOF. The truncated call is dropped and no markup leaks.
        let out = parse_chunks(
            &weather_tools(),
            &[
                "<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>",
                "{\"location\":\"NY",
            ],
        );
        assert_eq!(out.normal_text, "");
        assert!(out.calls.is_empty());
    }

    #[test]
    fn mismatched_fences_drop_the_call_instead_of_recovering_it() {
        // `TOOLCALLING.batch.4.d` (sourced from vLLM's own kimi_k2 parser
        // tests): the JSON body is syntactically complete, but the model
        // closes the whole tool_calls section (`section_end`) without ever
        // giving this call its own `call_end`. This is NOT the same as
        // running out of tokens mid-call (`suppresses_truncated_call_at_eof`)
        // or completing right at EOF with nothing else following
        // (`UNIFIED.tool_no_close`, `TOOLCALLING.streamv2.5.a`) -- a real
        // section-end marker DOES follow, so the model had more to say and
        // chose not to close this call. Batch mode's regex requires a
        // literal `call_end` unconditionally and drops it; streaming must
        // match, or `conformance_toolcalling_batch_via_stream` diverges.
        let out = parse_chunks(
            &weather_tools(),
            &[
                "<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>",
                "{\"location\":\"NYC\"}<|tool_calls_section_end|>",
            ],
        );
        assert_eq!(out.normal_text, "");
        assert!(out.calls.is_empty());
    }

    /// Reviewer-caught regression, sibling of the test above: dropping a
    /// call with mismatched fences is correct, but the ABOVE test ends
    /// right after `section_end` and so cannot detect a different bug --
    /// `WrappedBlockScanner::drain` used to treat `invoke_end_at`
    /// returning `None` at flush as ALWAYS "genuinely incomplete", clearing
    /// the entire remaining buffer including any real visible text that
    /// followed the section-end marker. Batch mode's regex-based
    /// extraction correctly preserves that trailing text; streaming
    /// dropped it too. The correct result is an empty call list AND the
    /// visible suffix preserved as `normal_text`.
    #[test]
    fn mismatched_fences_preserve_text_after_section_end() {
        let out = parse_chunks(
            &weather_tools(),
            &[
                "<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{}<|tool_calls_section_end|>Visible answer",
            ],
        );
        assert!(out.calls.is_empty());
        assert_eq!(out.normal_text, "Visible answer");
    }

    /// Reviewer-caught regression: a bracket-balanced but JSON-GRAMMAR-INVALID
    /// body (`{not-json}` -- braces match, but it's not legal JSON: no
    /// quoted key, no colon, no value) with NO `call_end` anywhere in the
    /// buffer used to still recover a call at EOF, because `json_value_end`
    /// only proves bracket/quote nesting is balanced, not that the bytes
    /// parse as JSON. `parse_section_block`'s regex (batch mode) requires a
    /// literal `call_end` to match anything at all and so drops this input
    /// outright -- reproduced directly: streaming shipped a call with
    /// `arguments: "{not-json}"` while batch mode produced zero calls.
    #[test]
    fn eof_recovery_requires_actually_valid_json_not_just_balanced_brackets() {
        let text = "<|tool_call_begin|>functions.run:0<|tool_call_argument_begin|>{not-json}";
        assert_eq!(
            kimi_invoke_end(text, true, 0),
            None,
            "a bracket-balanced but grammatically invalid body with no call_end evidence \
             at all must not be recovered -- batch mode's regex can never match it either"
        );
    }

    /// Sibling positive control: the SAME shape but with genuinely valid
    /// JSON still recovers normally -- this fix must not regress the
    /// existing `UNIFIED.5.b`/`tool_no_close` best-effort recovery contract.
    #[test]
    fn eof_recovery_still_recovers_genuinely_valid_json_with_no_call_end() {
        let text =
            "<|tool_call_begin|>functions.run:0<|tool_call_argument_begin|>{\"city\": \"Paris\"}";
        assert_eq!(
            kimi_invoke_end(text, true, 0),
            Some(text.len()),
            "valid JSON with no call_end at true EOF must still be recovered"
        );
    }

    /// Reviewer-caught regression: a malformed (odd quote count) argument
    /// followed by a real section-end marker, with a literal `call_end`
    /// only appearing AFTER that section-end, used to still recover a call
    /// whose raw-string argument swallowed the section-end marker as text
    /// -- the same "mismatched fences" shape the well-formed-JSON path
    /// already guards against (see the `flush` block above this function's
    /// EOF-recovery gate), just reached through the malformed/raw-string
    /// fallback instead, which was missing the equivalent guard.
    #[test]
    fn malformed_argument_recovery_also_respects_an_intervening_section_end() {
        let text = "<|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{\"location\": \"unterminated<|tool_calls_section_end|><|tool_call_end|>";
        assert_eq!(
            kimi_invoke_end(text, true, 0),
            None,
            "a real section-end marker occurring before the only available call_end \
             means this call never got its own closer -- must drop, not swallow the \
             section-end into a malformed raw-string argument"
        );
    }

    /// Sibling positive control: the SAME malformed-argument raw-string
    /// fallback still recovers normally when no section-end marker
    /// intervenes -- this fix must not regress the raw-string recovery
    /// contract `malformed_non_json_arguments_still_ship_the_call_instead_of_vanishing`
    /// already covers end-to-end.
    #[test]
    fn malformed_argument_recovery_still_works_without_an_intervening_section_end() {
        let text = "<|tool_call_begin|>functions.run:0<|tool_call_argument_begin|>bad\"arg<|tool_call_end|>";
        assert_eq!(
            kimi_invoke_end(text, true, 0),
            Some(text.len()),
            "a malformed raw-string argument with its own real call_end and no \
             intervening section-end must still be recovered"
        );
    }

    #[test]
    fn closed_malformed_arguments_emit_on_the_closing_chunk() {
        let mut parser = KimiK2ToolStreamParser::new(&weather_tools());
        assert!(
            parser
                .push("<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>")
                .unwrap()
                .calls
                .is_empty()
        );

        let emitted = parser
            .push("{\"location\":\"NYC\"<|tool_call_end|><|tool_calls_section_end|>")
            .unwrap();
        assert_eq!(emitted.calls.len(), 1);
        assert_eq!(emitted.calls[0].arguments, r#"{"location":"NYC""#);
        assert!(parser.finish().unwrap().calls.is_empty());
    }

    #[test]
    fn closed_malformed_arguments_emit_before_finish_at_every_valid_utf8_split() {
        let input = "<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{\"location\":\"München\"<|tool_call_end|><|tool_calls_section_end|>";
        for split in (0..=input.len()).filter(|&index| input.is_char_boundary(index)) {
            let mut parser = KimiK2ToolStreamParser::new(&weather_tools());
            let mut emitted = parser.push(&input[..split]).unwrap();
            emitted.append(parser.push(&input[split..]).unwrap());
            assert_eq!(
                emitted.calls.len(),
                1,
                "split at byte {split} must emit once the closer is available"
            );
            assert_eq!(emitted.calls[0].arguments, r#"{"location":"München""#);
            assert!(
                parser.finish().unwrap().calls.is_empty(),
                "split at byte {split} must not defer the call until finish"
            );
        }
    }

    fn assert_closed_malformed_quoted_marker_emits_on_the_closing_chunk(marker: &str) {
        let arguments = format!(r#"{{"location" "München {marker} literal"}}"#);
        let mut parser = KimiK2ToolStreamParser::new(&weather_tools());
        assert!(
            parser
                .push("<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>")
                .unwrap()
                .calls
                .is_empty()
        );

        let emitted = parser
            .push(&format!("{arguments}{CALL_END}{SECTION_END_PLURAL}"))
            .unwrap();
        assert_eq!(emitted.calls.len(), 1);
        assert_eq!(emitted.calls[0].arguments, arguments);
        assert!(parser.finish().unwrap().calls.is_empty());
    }

    #[test]
    fn malformed_quoted_call_end_is_data_and_emits_on_the_closing_chunk() {
        assert_closed_malformed_quoted_marker_emits_on_the_closing_chunk(CALL_END);
    }

    #[test]
    fn malformed_quoted_section_end_is_data_and_emits_on_the_closing_chunk() {
        assert_closed_malformed_quoted_marker_emits_on_the_closing_chunk(SECTION_END_PLURAL);
    }

    #[test]
    fn closed_malformed_quoted_markers_emit_before_finish_at_every_valid_utf8_split() {
        let header = "<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>";
        for marker in [CALL_END, SECTION_END_PLURAL] {
            let arguments = format!(r#"{{"location" "München {marker} literal"}}"#);
            let input = format!("{header}{arguments}{CALL_END}{SECTION_END_PLURAL}");
            for split in (0..=input.len()).filter(|&index| input.is_char_boundary(index)) {
                let mut parser = KimiK2ToolStreamParser::new(&weather_tools());
                let mut emitted = parser.push(&input[..split]).unwrap();
                emitted.append(parser.push(&input[split..]).unwrap());
                assert_eq!(
                    emitted.calls.len(),
                    1,
                    "marker {marker:?}, split at byte {split} must emit once the closer is available"
                );
                assert_eq!(emitted.calls[0].arguments, arguments);
                assert!(
                    parser.finish().unwrap().calls.is_empty(),
                    "marker {marker:?}, split at byte {split} must not defer the call until finish"
                );
            }
        }
    }

    #[test]
    fn mismatched_fences_singular_section_variant_also_drops_the_call() {
        // Same shape as above, through the singular `<|tool_call_section_end|>`
        // variant -- the guard checks both `section_end_variants`, not just
        // the plural default.
        let out = parse_chunks(
            &weather_tools(),
            &[
                "<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{\"location\":\"NYC\"}<|tool_call_section_end|>",
            ],
        );
        assert_eq!(out.normal_text, "");
        assert!(out.calls.is_empty());
    }

    #[test]
    fn multi_call_mismatched_fences_keeps_the_closed_call_drops_the_open_one() {
        // `TOOLCALLING.batch.5.d`-adjacent shape: a properly closed first
        // call followed by a second call whose JSON is complete but whose
        // `call_end` is missing, with a real section_end right after (unlike
        // `.5.d`, which has no section_end at all and is a genuine-truncation
        // case handled by `known-divergences.yaml`, not this guard). The
        // first call must still ship; only the malformed second is dropped.
        let out = parse_chunks(
            &weather_tools(),
            &[
                "<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{\"location\":\"Boston\"}<|tool_call_end|>",
                "<|tool_call_begin|>functions.get_weather:1<|tool_call_argument_begin|>{\"location\":\"New York\"}<|tool_calls_section_end|>",
            ],
        );
        assert_eq!(out.normal_text, "");
        let merged = out.coalesce_calls();
        assert_eq!(merged.calls.len(), 1);
        assert_eq!(merged.calls[0].arguments, r#"{"location":"Boston"}"#);
    }

    #[test]
    fn strips_orphan_call_end_outside_section() {
        // A complete orphan `call_end` with no open section is stray double-close
        // markup: it must be stripped, never leaked, and the surrounding genuine
        // prose preserved (v1 `first_orphan_kimi_marker_index` parity).
        let out = parse_chunks(
            &weather_tools(),
            &["Here you go.", "<|tool_call_end|>", "All set."],
        );
        assert_eq!(out.normal_text, "Here you go.All set.");
        assert!(out.calls.is_empty());
    }

    #[test]
    fn strips_orphan_argument_begin_outside_section() {
        let out = parse_chunks(
            &weather_tools(),
            &["Here you go.", "<|tool_call_argument_begin|>", "All set."],
        );
        assert_eq!(out.normal_text, "Here you go.All set.");
        assert!(out.calls.is_empty());
    }

    #[test]
    fn strips_orphan_section_end_outside_section() {
        let out = parse_chunks(
            &weather_tools(),
            &["Here you go.", "<|tool_calls_section_end|>", "All set."],
        );
        assert_eq!(out.normal_text, "Here you go.All set.");
        assert!(out.calls.is_empty());
    }

    #[test]
    fn recovers_complete_bare_call_without_section() {
        let out = parse_chunks(
            &weather_tools(),
            &[
                "<|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>",
                "{\"location\":\"NYC\"}<|tool_call_end|>",
            ],
        );
        assert_eq!(out.normal_text, "");
        let merged = out.coalesce_calls();
        assert_eq!(merged.calls.len(), 1);
        assert_eq!(merged.calls[0].name.as_deref(), Some("get_weather"));
        assert_eq!(merged.calls[0].arguments, r#"{"location":"NYC"}"#);
    }
}
