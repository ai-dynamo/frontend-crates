// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Per-choice isolation sweep for Dynamo parser v2 (DIS-2381 step 3).
//!
//! Invariant under test, applied across the whole streamv2 corpus:
//!
//! ```text
//!   demux(parse(interleave(A@0, B@1))) == (parse(A), parse(B))
//! ```
//!
//! `parity_toolcalling_stream.rs` proves each case parses correctly on its own.
//! It cannot prove a `ToolParser` keeps its state isolated per `choice.index`,
//! because every fixture is single-choice. When a real `n>1` caller multiplexes
//! several completions onto one wire, it must hand each choice's delta to that
//! choice's own parser instance; a host that shared one parser across choices
//! would splice one choice's partial marker/JSON into another.
//!
//! This lane builds the missing multi-choice stream from the existing corpus: it
//! pairs cases *within a family* (same tools/parser), interleaves them under the
//! deterministic schedules from `parsers/v1/tests/common/interleave.rs` (reused
//! directly — one source of truth across crates), routes each tagged delta to a
//! per-`choice.index` parser via a `ChoiceRouter`, then demuxes each choice.
//!
//! # Two oracles, and why the solo run alone is not enough
//!
//! Each demuxed choice is compared against TWO goldens:
//!
//! 1. **Recorded `dynamo_v2` capture (primary, absolute).** The corpus's own
//!    per-chunk `expected:` / `normal_text`, folded to the assembled shape, with
//!    `arguments` kept as the RAW recorded string.
//! 2. **Solo run of that choice's demuxed subsequence (secondary, relative).**
//!
//! The solo run alone is NOT a sufficient oracle, and this is not hypothetical.
//! It is produced in the same process as the interleaved run, so any
//! process-global state — a `OnceLock`/`lazy_static` cache, anything memoised on
//! the first tool set — corrupts BOTH sides identically and they still compare
//! equal while both are wrong. rmccorm4 demonstrated exactly this on PR #135 by
//! caching the first tool configuration in a `OnceLock`: this lane passed all
//! 1088 pair-schedules while canonical parity reported real failures. Running the
//! solos first does not help — that changes which side seeds the cache, not the
//! fact that they share it. Only a golden recorded in an EARLIER process closes
//! it, hence oracle 1.
//!
//! Oracle 2 is kept for two reasons. It localises a divergence to cross-choice
//! leakage rather than to whole-corpus drift, and it is the ONLY oracle that can
//! see emission TIMING: the recorded corpus stores totals, so oracle 1 compares
//! totals alone. A parser that stalls a choice while a sibling is live and
//! releases the correct bytes at `finish` has identical totals and is invisible
//! to oracle 1; `EngineResult::emission_profile`, compared against the solo run,
//! is what catches it.
//!
//! KNOWN LIMIT of oracle 2: it is produced in this process, so a defect that
//! poisons state PERMANENTLY and process-globally (rather than only while two
//! parsers coexist) degrades the solo run identically and compares equal. That
//! is the same class oracle 1 exists to cover for totals; for timing there is no
//! recorded baseline to anchor to, and closing it would need a baseline captured
//! in a separate process. Coexistence-scoped stalls — the realistic per-request
//! bug — are caught.
//!
//! `BoundarySplit` re-chunks the input while the recorded capture describes the
//! ORIGINAL chunking, so oracle 1 is only meaningful there while the parser's
//! assembled output is chunking-invariant. `sweep_toolcalling_stream.rs` enforces
//! exactly that against `conformance/toolcalling/known-chunking-divergences.yaml`
//! (empty today). This lane consults the SAME allow-list and drops oracle 1 for
//! an allow-listed case under `BoundarySplit`, so adding an entry there can never
//! turn into a spurious isolation failure here. Oracle 2 still applies, because
//! the solo golden is built from the same re-chunked subsequence.

#[path = "../../parsers/v1/tests/common/interleave.rs"]
mod interleave;

mod common;

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::time::Instant;

use common::{
    STREAM_DYNAMO_V2_CURRENT_CAPTURE, collect_yaml, ensure_fixtures,
    version_dirs_ascending_with_current,
};
use dynamo_parsers_v2::{
    Tool, ToolCallDelta, ToolParser, ToolParserInput, create_tool_parser_for_family,
};
use interleave::{Schedule, demux_items, interleave_items};
use serde::Deserialize;

// ── Family registry (single source of truth) ────────────────────────────────

/// Rows of `conformance/utils/src/parser_families.yaml`. A family is exercised
/// here iff it has a non-null `dynamo_v2` id — derived, never hardcoded, so
/// registering a new v2 family auto-enrolls it.
#[derive(Deserialize)]
struct Registry {
    families: BTreeMap<String, FamilyRow>,
}

#[derive(Deserialize)]
struct FamilyRow {
    dynamo_v2: Option<String>,
    /// `tokens` (Harmony token-native path) or `text`.
    preferred_input: String,
}

/// The chunking allow-list `sweep_toolcalling_stream.rs` enforces against.
/// Loaded here so `BoundarySplit` can drop the recorded oracle for exactly the
/// cases whose assembled output is known to depend on where the stream is split.
fn load_chunking_allowlist() -> BTreeMap<String, BTreeMap<String, String>> {
    let path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("toolcalling/known-chunking-divergences.yaml");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("{}: read error: {e}", path.display()));
    serde_yaml::from_str(&text).unwrap_or_else(|e| panic!("{}: parse error: {e}", path.display()))
}

fn load_registry() -> Registry {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("utils/src/parser_families.yaml");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("{}: read error: {e}", path.display()));
    serde_yaml::from_str(&text).unwrap_or_else(|e| panic!("{}: parse error: {e}", path.display()))
}

// ── Fixture schema ───────────────────────────────────────────────────────────
//
// Inputs (shared per-chunk deltas) live in `inputs/`; Dynamo's recorded output
// lives in `dynamo_v2-<ver>/`. Both are loaded: the recorded output is the
// PRIMARY oracle (see the module header on why the solo run alone is not enough).

#[derive(Deserialize)]
struct Fixture {
    /// The fixture's OWN family key, as `conformance_toolcalling_stream.rs` and
    /// `sweep_toolcalling_stream.rs` both read it. Deriving it from the parent
    /// directory instead would diverge from those lanes the moment a fixture sits
    /// deeper than `inputs/<family>/`, since `collect_yaml` recurses — and every
    /// consequence of a wrong key here FAILS OPEN (the family looks like
    /// `dynamo_v2=null` and is skipped, or the chunking allow-list never matches).
    family: String,
    #[serde(default)]
    mode: Option<String>,
    #[serde(default)]
    cases: BTreeMap<String, Case>,
}

#[derive(Deserialize, Clone)]
struct Case {
    #[serde(default)]
    tools: Vec<Tool>,
    #[serde(default)]
    chunks: Vec<Chunk>,
}

#[derive(Deserialize, Clone)]
struct Chunk {
    #[serde(default)]
    delta_text: String,
    #[serde(default)]
    delta_token_ids: Vec<u32>,
    /// Deserialized only so a MID-STREAM terminator can be detected and the case
    /// skipped. The canonical lane calls `finish()` at the chunk carrying
    /// `finish_reason` and keeps pushing afterwards; this lane finishes once, at
    /// end of stream. Those agree only while `finish_reason` sits on the last
    /// chunk (true for all 272 enabled-family cases today). A fixture that
    /// terminated mid-stream would assemble differently here and diverge from the
    /// recorded oracle for reasons unrelated to per-choice isolation — so such a
    /// case is skipped and named instead of silently mis-compared.
    #[serde(default)]
    finish_reason: Option<String>,
}

impl Case {
    /// Why this case cannot be compared against the recorded oracle, if so.
    ///
    /// Two shapes are excluded, and BOTH must be, because this lane finishes each
    /// choice exactly once at the end of its own stream:
    ///
    /// * a terminator before the last chunk — the canonical lane calls `finish()`
    ///   there and keeps pushing, which assembles differently;
    /// * NO terminator at all — the recorded reference was captured without the
    ///   parser's closing output, so the closing output this lane produces would
    ///   be reported as per-choice leakage when nothing about multiple choices is
    ///   wrong.
    ///
    /// All 272 enabled-family cases carry exactly one terminator, on the last
    /// chunk, so neither exclusion fires today.
    fn unterminated_reason(&self) -> Option<&'static str> {
        if self.chunks.is_empty() {
            return Some("no chunks");
        }
        let last = self.chunks.len() - 1;
        if self
            .chunks
            .iter()
            .enumerate()
            .any(|(i, c)| c.finish_reason.is_some() && i != last)
        {
            return Some("finish_reason mid-stream");
        }
        if self.chunks[last].finish_reason.is_none() {
            return Some("no finish_reason on the last chunk");
        }
        None
    }
}

// ── Recorded dynamo_v2 output (the absolute oracle) ──────────────────────────
//
// Same shape the canonical `conformance_toolcalling_stream.rs` overlay uses.
// Capture dirs fold ASCENDING (latest wins per case), and a `unavailable`
// entry means the v2 parser cannot handle that case — such cases are skipped
// and NAMED, never silently counted as passing.

#[derive(Deserialize)]
struct DynFixture {
    #[serde(default)]
    cases: BTreeMap<String, DynCase>,
}

#[derive(Deserialize)]
struct DynCase {
    #[serde(default)]
    unavailable: Option<String>,
    #[serde(default)]
    chunks: Vec<DynChunk>,
}

#[derive(Deserialize)]
struct DynChunk {
    #[serde(default)]
    expected: Vec<FixtureDelta>,
    #[serde(default)]
    normal_text: Option<String>,
}

#[derive(Deserialize)]
struct FixtureDelta {
    index: u32,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
    /// Historical captures predate explicit completion and are complete by
    /// definition; only newly recorded provisional deltas set this to false.
    #[serde(default = "default_complete")]
    complete: bool,
}

fn default_complete() -> bool {
    true
}

/// One case's recorded dynamo_v2 output, already folded to the assembled shape
/// this lane compares against. `None` = the case is unavailable for v2.
type RecordedCases = BTreeMap<String, Option<EngineResult>>;

/// Fold every `dynamo_v2-<ver>/<rel>` capture into the assembled per-case
/// expectation.
///
/// Folding is PER CHUNK INDEX, matching `merge_dynamo` in the canonical
/// `conformance_toolcalling_stream.rs`. Replacing the whole case instead would
/// diverge from canonical the moment a newer capture recorded FEWER chunks than
/// an older one for the same case: canonical keeps the older capture's
/// expectations for the trailing chunks, a whole-case replace would silently
/// truncate the oracle and report a spurious divergence. No committed capture
/// does that today, but the two lanes must not disagree about what the corpus
/// means.
///
/// `input_chunks` maps case id -> the INPUT's chunk count. Recorded chunks beyond
/// that are DROPPED, matching canonical `merge_dynamo`, which only writes into
/// `case.chunks.get_mut(i)`. Keeping them would let a stale longer recording of a
/// since-shortened fixture demand output that no longer exists, and this lane
/// would report it as an isolation divergence.
fn load_recorded(
    dyn_dirs: &[PathBuf],
    rel: &Path,
    input_chunks: &BTreeMap<String, usize>,
) -> RecordedCases {
    // Per case: the merged per-chunk overlay PLUS a separate unavailability flag.
    //
    // The flag must be tracked ALONGSIDE the chunks, never by replacing them with
    // `None`: canonical `merge_dynamo` only sets/clears `unavailable` and leaves
    // the merged chunks in place. Collapsing the case to `None` on an
    // intermediate `unavailable` capture destroys everything merged so far, so
    // the sequence old(chunks 0..n) -> mid(unavailable) -> new(chunk 0 only)
    // would yield new-chunk-0 alone here while canonical yields
    // new-chunk-0 + old-chunks-1..n. No committed capture history has that shape
    // today, but the two lanes must not disagree about what the corpus means.
    #[derive(Default)]
    struct Merged {
        chunks: Vec<DynChunk>,
        unavailable: bool,
    }
    let mut merged: BTreeMap<String, Merged> = BTreeMap::new();
    for dir in dyn_dirs {
        let dfp = dir.join(rel);
        // A missing overlay is benign; any other I/O error must surface.
        let text = match std::fs::read_to_string(&dfp) {
            Ok(t) => t,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
            Err(e) => panic!("{}: dynamo overlay read error: {e}", dfp.display()),
        };
        let fx: DynFixture = serde_yaml::from_str(&text)
            .unwrap_or_else(|e| panic!("{}: dynamo overlay parse error: {e}", dfp.display()));
        for (cid, dcase) in fx.cases {
            let cid_key = cid.clone();
            let slot = merged.entry(cid).or_default();
            if dcase.unavailable.is_some() {
                // Flag it, but KEEP the chunks merged so far.
                slot.unavailable = true;
                continue;
            }
            // A later capture that supplies expectations CLEARS the flag without
            // discarding what earlier captures recorded.
            slot.unavailable = false;
            // Bound by the INPUT's chunk count, exactly as merge_dynamo does.
            let limit = input_chunks.get(&cid_key).copied().unwrap_or(0);
            for (i, dchunk) in dcase.chunks.into_iter().enumerate() {
                if i >= limit {
                    break;
                }
                if i < slot.chunks.len() {
                    slot.chunks[i] = dchunk;
                } else {
                    slot.chunks.push(dchunk);
                }
            }
        }
    }

    merged
        .into_iter()
        .map(|(cid, m)| {
            let assembled = (!m.unavailable).then(|| assemble_recorded(&m.chunks));
            (cid, assembled)
        })
        .collect()
}

/// Concatenate a case's merged per-chunk capture into the assembled shape this
/// lane compares against.
fn assemble_recorded(chunks: &[DynChunk]) -> EngineResult {
    let mut names: BTreeMap<u32, String> = BTreeMap::new();
    let mut args: BTreeMap<u32, String> = BTreeMap::new();
    let mut complete: BTreeMap<u32, bool> = BTreeMap::new();
    let mut order: Vec<u32> = Vec::new();
    let mut normal_text = String::new();
    for chunk in chunks {
        for d in &chunk.expected {
            if !order.contains(&d.index) {
                order.push(d.index);
            }
            if let Some(n) = &d.name {
                names.entry(d.index).or_default().push_str(n);
            }
            if let Some(a) = &d.arguments {
                args.entry(d.index).or_default().push_str(a);
            }
            *complete.entry(d.index).or_default() |= d.complete;
        }
        if let Some(nt) = &chunk.normal_text {
            normal_text.push_str(nt);
        }
    }
    let calls = order
        .into_iter()
        .filter_map(|i| {
            if complete.get(&i) != Some(&true) {
                return None;
            }
            Some((
                names.get(&i).cloned().unwrap_or_default(),
                args.get(&i).cloned().unwrap_or_default(),
            ))
        })
        .collect();
    EngineResult {
        calls,
        normal_text,
        emission_profile: Vec::new(),
    }
}

// ── Assembled per-choice result ──────────────────────────────────────────────

/// Assembled per-choice output. `arguments` stays a RAW string, never re-parsed
/// into a `Value`: the corpus records it verbatim (`{"location":"NYC"}`), and
/// round-tripping through `serde_json` would normalise key order and whitespace,
/// hiding a formatting divergence the recorded oracle is there to catch.
#[derive(Debug, PartialEq, Clone, Default)]
struct EngineResult {
    calls: Vec<(String, String)>,
    normal_text: String,
    /// STREAMING SHAPE: running `(delta count, normal-text length)` sampled after
    /// each `push_input` for this choice, plus once after `finish`.
    ///
    /// Totals alone cannot see a defect that DELAYS a choice's output. A parser
    /// that stalls as soon as a second instance exists and releases the correct
    /// bytes at `finish` produces identical totals while a real client watches
    /// choice 0 go silent for the whole of choice 1's stream. Sampling per push
    /// puts emission timing INSIDE the compared value.
    ///
    /// Empty when the value came from the recorded corpus, which has no timing
    /// information — the recorded oracle stays a totals-only comparison and the
    /// profile is checked against the solo run.
    emission_profile: Vec<(usize, usize)>,
}

/// Fold a choice's emitted tool-call deltas + normal text into assembled calls.
/// `tool_index` is parser-local; each choice owns its parser so indices never
/// collide across choices.
fn assemble(deltas: &[ToolCallDelta], normal_text: String) -> EngineResult {
    let mut names: BTreeMap<usize, String> = BTreeMap::new();
    let mut args: BTreeMap<usize, String> = BTreeMap::new();
    let mut complete: BTreeMap<usize, bool> = BTreeMap::new();
    let mut order: Vec<usize> = Vec::new();
    for d in deltas {
        if !order.contains(&d.tool_index) {
            order.push(d.tool_index);
        }
        if let Some(name) = &d.name {
            names.entry(d.tool_index).or_default().push_str(name);
        }
        args.entry(d.tool_index).or_default().push_str(&d.arguments);
        *complete.entry(d.tool_index).or_default() |= d.complete;
    }
    let calls = order
        .into_iter()
        .filter_map(|idx| {
            if complete.get(&idx) != Some(&true) {
                return None;
            }
            Some((
                names.get(&idx).cloned().unwrap_or_default(),
                args.get(&idx).cloned().unwrap_or_default(),
            ))
        })
        .collect();
    EngineResult {
        calls,
        normal_text,
        emission_profile: Vec::new(),
    }
}

#[test]
fn assemble_omits_incomplete_tool_deltas() {
    let result = assemble(
        &[ToolCallDelta {
            tool_index: 0,
            name: Some("get_weather".into()),
            arguments: r#"{"city":"Par"#.into(),
            complete: false,
        }],
        String::new(),
    );
    assert!(result.calls.is_empty());
}

// ── ChoiceRouter: one parser instance per choice.index ───────────────────────

/// Routes tagged deltas to a per-`choice.index` `ToolParser`, exactly what an
/// `n>1` caller must do: a completion's deltas always reach the parser holding
/// that completion's state, never a sibling's. Building one parser per index on
/// first sight (with that index's tools) mirrors real per-request construction.
struct ChoiceRouter {
    /// The REGISTERED parser id (`parser_families.yaml` -> `dynamo_v2`), not the
    /// fixture directory name. The registry explicitly allows the two to differ
    /// (`harmony_text: {dynamo_v2: harmony}`); constructing from the directory
    /// name only works by accident where the factory also matches the literal
    /// key, and would bail for any future row whose id differs.
    parser_id: String,
    parsers: HashMap<u32, Box<dyn ToolParser>>,
    deltas: HashMap<u32, Vec<ToolCallDelta>>,
    normal: HashMap<u32, String>,
    /// Per choice: running `(delta count, normal-text length)` sampled after each
    /// push and once after that choice's finish.
    profile: HashMap<u32, Vec<(usize, usize)>>,
    finished: HashMap<u32, bool>,
}

impl ChoiceRouter {
    fn new(parser_id: &str) -> Self {
        Self {
            parser_id: parser_id.to_string(),
            parsers: HashMap::new(),
            deltas: HashMap::new(),
            normal: HashMap::new(),
            profile: HashMap::new(),
            finished: HashMap::new(),
        }
    }

    fn push(
        &mut self,
        index: u32,
        tools: &[Tool],
        input: ToolParserInput<'_>,
    ) -> anyhow::Result<()> {
        let res = {
            let parser_id = &self.parser_id;
            let parser = match self.parsers.entry(index) {
                std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
                std::collections::hash_map::Entry::Vacant(e) => {
                    e.insert(create_tool_parser_for_family(parser_id, tools)?)
                }
            };
            parser.push_input(input)?
        };
        self.normal
            .entry(index)
            .or_default()
            .push_str(&res.normal_text);
        self.deltas.entry(index).or_default().extend(res.calls);
        self.sample(index);
        Ok(())
    }

    fn sample(&mut self, index: u32) {
        let calls = self.deltas.get(&index).map_or(0, Vec::len);
        let text = self.normal.get(&index).map_or(0, String::len);
        self.profile.entry(index).or_default().push((calls, text));
    }

    /// Finish ONE choice, at the point its own stream ends. Idempotent.
    fn finish_choice(&mut self, index: u32) -> anyhow::Result<()> {
        if self.finished.get(&index).copied().unwrap_or(false) {
            return Ok(());
        }
        self.finished.insert(index, true);
        if let Some(parser) = self.parsers.get_mut(&index) {
            let res = parser.finish()?;
            self.normal
                .entry(index)
                .or_default()
                .push_str(&res.normal_text);
            self.deltas.entry(index).or_default().extend(res.calls);
        }
        self.sample(index);
        Ok(())
    }

    fn finish(&mut self) -> anyhow::Result<()> {
        let indices: Vec<u32> = self.parsers.keys().copied().collect();
        for index in indices {
            self.finish_choice(index)?;
        }
        Ok(())
    }

    fn assembled(&self, index: u32) -> EngineResult {
        let mut r = assemble(
            self.deltas.get(&index).map(Vec::as_slice).unwrap_or(&[]),
            self.normal.get(&index).cloned().unwrap_or_default(),
        );
        r.emission_profile = self.profile.get(&index).cloned().unwrap_or_default();
        r
    }
}

/// Run one choice's items solo through a fresh single parser (the golden).
fn solo<T>(
    parser_id: &str,
    tools: &[Tool],
    items: &[T],
    to_input: fn(&T) -> ToolParserInput<'_>,
) -> anyhow::Result<EngineResult> {
    let mut router = ChoiceRouter::new(parser_id);
    for item in items {
        router.push(0, tools, to_input(item))?;
    }
    router.finish()?;
    Ok(router.assembled(0))
}

// ── Generic pair check over one item representation ──────────────────────────

#[allow(clippy::too_many_arguments)]
fn check_pair<T: interleave::Splittable>(
    parser_id: &str,
    tools_a: &[Tool],
    seq_a: &[T],
    tools_b: &[Tool],
    seq_b: &[T],
    schedule: Schedule,
    to_input: fn(&T) -> ToolParserInput<'_>,
    label: &str,
    recorded: [&EngineResult; 2],
    chunking_divergent: [bool; 2],
    chunking_skips: &mut Vec<String>,
    failures: &mut Vec<String>,
) -> anyhow::Result<()> {
    let sequences = vec![seq_a.to_vec(), seq_b.to_vec()];
    let tagged = interleave_items(&sequences, schedule);
    let demuxed = demux_items(&tagged);
    let tools_by_index = [tools_a, tools_b];

    // Interleaved run through ONE router (per-choice parsers).
    //
    // STAGGERED TERMINATION: each choice is finished at the point ITS OWN stream
    // ends, which for the shorter choice lands mid-wire while the sibling is
    // still streaming. Finishing every parser together at the end of the wire —
    // as this lane used to — cannot reach a defect that tears down sibling state
    // when the first choice terminates.
    let mut remaining: BTreeMap<u32, usize> = BTreeMap::new();
    for (index, _) in &tagged {
        *remaining.entry(*index).or_default() += 1;
    }
    let mut router = ChoiceRouter::new(parser_id);
    let mut interleaved_err: Option<anyhow::Error> = None;
    'wire: {
        for (index, item) in &tagged {
            if let Err(e) = router.push(*index, tools_by_index[*index as usize], to_input(item)) {
                interleaved_err = Some(e);
                break 'wire;
            }
            let left = remaining.get_mut(index).unwrap();
            *left -= 1;
            if *left == 0
                && let Err(e) = router.finish_choice(*index)
            {
                interleaved_err = Some(e);
                break 'wire;
            }
        }
        if let Err(e) = router.finish() {
            interleaved_err = Some(e);
        }
    }

    // A parser error on the interleaved wire is NOT automatically a skip.
    //
    // Counting it as one is fail-open on the exact defect this lane exists to
    // find: an error that appears ONLY while two choices are live is cross-choice
    // interference, and skipping it lets the suite exit green over the real thing.
    // So when the wire errors, re-run each choice ALONE on the same items. If both
    // succeed solo, the error is caused by the sibling's presence -> FAILURE. If a
    // solo run errors too, the parser simply cannot handle that shape -> the error
    // propagates to the caller and is logged as a named skip.
    if let Some(e) = interleaved_err {
        // Drop the interleaved parsers FIRST. The probe must be a genuine
        // single-parser run; leaving the wire's parsers alive would let a
        // coexistence-triggered defect break the probe too and disguise itself as
        // an unsupported shape — the precise fail-open being closed here.
        drop(router);
        let mut solo_ok = true;
        for index in [0u32, 1] {
            let items = demuxed.get(&index).cloned().unwrap_or_default();
            if solo(parser_id, tools_by_index[index as usize], &items, to_input).is_err() {
                solo_ok = false;
            }
        }
        if solo_ok {
            failures.push(format!(
                "schedule={} {label}: interleaved run FAILED but both choices parse alone \
                 -- this is cross-choice interference, not an unsupported shape: {e}",
                schedule.label(),
            ));
            return Ok(());
        }
        return Err(e);
    }

    for index in [0u32, 1] {
        let got = router.assembled(index);

        // `BoundarySplit` RE-CHUNKS the input while the recorded capture describes
        // the ORIGINAL chunking, so oracle 1 only means something there while this
        // case's assembled output is chunking-invariant.
        //
        // The allow-list alone is NOT a sufficient guard for that. It is populated
        // by `sweep_toolcalling_stream.rs`, whose invariance check is WEAKER than
        // the comparison here: the sweep normalises arguments through
        // `serde_json::from_str` and keys calls by name, while this lane compares
        // the RAW argument string and orders calls by first appearance. A parser
        // whose chunking sensitivity is confined to argument whitespace or key
        // order, or to a call that never emits a name, would pass the sweep — so no
        // allow-list entry would ever be added — and then fail HERE, reported as a
        // per-choice isolation divergence it is not.
        //
        // So decide chunking-invariance BY THIS LANE'S OWN COMPARISON: run this
        // choice solo at the original chunking and at the split chunking. If those
        // disagree the case is chunking-sensitive by the standard actually being
        // applied, and oracle 1 is skipped and NAMED rather than misattributed.
        // The allow-list is still honoured as a cheap short-circuit.
        let mut recorded_applies = true;
        if matches!(schedule, Schedule::BoundarySplit { .. }) {
            if chunking_divergent[index as usize] {
                // NAME it: an allow-listed case dropped from oracle 1 silently would
                // contradict the summary's claim that every skip is reported.
                recorded_applies = false;
                chunking_skips.push(format!(
                    "{label} choice={index} schedule={}: allow-listed in \
                     known-chunking-divergences.yaml; oracle 1 skipped",
                    schedule.label()
                ));
            } else {
                let split_items = demuxed.get(&index).cloned().unwrap_or_default();
                let orig_items = sequences[index as usize].clone();
                let at_split = solo(
                    parser_id,
                    tools_by_index[index as usize],
                    &split_items,
                    to_input,
                )?;
                let at_orig = solo(
                    parser_id,
                    tools_by_index[index as usize],
                    &orig_items,
                    to_input,
                )?;
                // Totals only: profiles differ by construction when the input is
                // re-chunked, which is not what is being decided here.
                let totals = |r: &EngineResult| (r.calls.clone(), r.normal_text.clone());
                if totals(&at_split) != totals(&at_orig) {
                    recorded_applies = false;
                    chunking_skips.push(format!(
                        "{label} choice={index} schedule={}: output is chunking-dependent \
                         (solo differs between original and split chunking); oracle 1 \
                         skipped, this is NOT an isolation failure",
                        schedule.label()
                    ));
                }
            }
        }

        // PRIMARY (absolute) oracle: the recorded dynamo_v2 capture for this case.
        // Anchoring to a value produced in an EARLIER process is what makes this
        // lane immune to shared-global corruption — an in-process golden would be
        // corrupted identically and still compare equal.
        // The recorded corpus has no timing information, so oracle 1 compares
        // TOTALS only; the streaming profile is checked against the solo run below.
        let got_totals = EngineResult {
            emission_profile: Vec::new(),
            ..got.clone()
        };
        if recorded_applies && &got_totals != recorded[index as usize] {
            let want = recorded[index as usize];
            failures.push(format!(
                "schedule={} {label} choice={index} diverged from recorded dynamo_v2:\n     got  {got_totals:?}\n     want {want:?}",
                schedule.label(),
            ));
            // Already failing against the absolute oracle; the relative one adds
            // nothing but noise for this choice.
            continue;
        }

        // SECONDARY (relative) oracle: the solo run of this choice's demuxed
        // subsequence. Weaker than the recorded capture, but it is the check that
        // localises a divergence to cross-choice leakage rather than to a
        // whole-corpus drift, and it covers the split-boundary dimension the
        // recorded capture has no chunking for.
        let items = demuxed.get(&index).cloned().unwrap_or_default();
        let golden = solo(parser_id, tools_by_index[index as usize], &items, to_input)?;
        if got != golden {
            failures.push(format!(
                "schedule={} {label} choice={index} diverged from solo golden:\n     got  {got:?}\n     want {golden:?}",
                schedule.label(),
            ));
        }
    }
    Ok(())
}

// `&String` (not `&str`) so it can be passed as a fn pointer where the generic
// item type is `String` — same reason as `token_input` below.
#[allow(clippy::ptr_arg)]
fn text_input(s: &String) -> ToolParserInput<'_> {
    ToolParserInput::Text(s.as_str())
}

#[allow(clippy::ptr_arg)]
fn token_input(v: &Vec<u32>) -> ToolParserInput<'_> {
    ToolParserInput::Tokens(v.as_slice())
}

// ── Deterministic within-family pairing ──────────────────────────────────────

/// Adjacent pairs + first-with-last over sorted case IDs. Budget-bounded (not
/// all-pairs); deterministic (no RNG).
fn pairs_for(ids: &[String]) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    for i in 0..ids.len().saturating_sub(1) {
        out.push((i, i + 1));
    }
    if ids.len() >= 3 {
        out.push((0, ids.len() - 1));
    }
    out
}

/// `BoundarySplit` is swept over both victims: a schedule that only ever splits
/// choice 0 cannot see a parser that breaks when the SIBLING is the split one.
/// Two ratios (not just the midpoint) cover non-midpoint boundaries such as
/// `<tool_ | call>`; the corpus is large enough that a full split-point sweep
/// belongs in the small hand-authored v1 lane rather than here.
fn schedules() -> Vec<Schedule> {
    vec![
        Schedule::RoundRobin,
        Schedule::FirstByteOffset(1),
        Schedule::FirstByteOffset(2),
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
    ]
}

// ── Test ─────────────────────────────────────────────────────────────────────

#[test]
fn toolcalling_stream_interleave_isolation() {
    let started = Instant::now();
    let registry = load_registry();
    let allowlist = load_chunking_allowlist();
    let enabled: BTreeMap<&str, &FamilyRow> = registry
        .families
        .iter()
        .filter(|(_, row)| row.dynamo_v2.is_some())
        .map(|(k, v)| (k.as_str(), v))
        .collect();

    let sv2 = ensure_fixtures().join("toolcalling/fixtures-stream-v2");
    let inputs_root = sv2.join("inputs");
    assert!(inputs_root.is_dir(), "missing {}", inputs_root.display());

    // Capture history for the v2 parser, folded ascending (latest wins per case)
    // via the shared helper the canonical parity test uses.
    let dyn_dirs =
        version_dirs_ascending_with_current(&sv2, "dynamo_v2-", STREAM_DYNAMO_V2_CURRENT_CAPTURE);
    assert!(
        !dyn_dirs.is_empty(),
        "no dynamo_v2-<version> dir under {}",
        sv2.display()
    );

    // Discover fixture families (input subdirs) and load every case per family,
    // alongside its recorded dynamo_v2 output.
    let mut families: BTreeMap<String, BTreeMap<String, Case>> = BTreeMap::new();
    let mut recorded: BTreeMap<String, RecordedCases> = BTreeMap::new();
    let mut files = Vec::new();
    collect_yaml(&inputs_root, &mut files);
    files.sort();
    for path in &files {
        let yaml = std::fs::read_to_string(path).unwrap();
        let fx: Fixture = match serde_yaml::from_str(&yaml) {
            Ok(f) => f,
            Err(e) => panic!("{}: YAML parse error: {e}", path.display()),
        };
        if !matches!(fx.mode.as_deref(), Some("stream" | "streamv2")) {
            continue;
        }
        let family = fx.family.clone();
        let rel = path
            .strip_prefix(&inputs_root)
            .expect("fixture under inputs root");
        let input_chunks: BTreeMap<String, usize> = fx
            .cases
            .iter()
            .map(|(cid, c)| (cid.clone(), c.chunks.len()))
            .collect();
        recorded
            .entry(family.clone())
            .or_default()
            .extend(load_recorded(&dyn_dirs, rel, &input_chunks));
        families.entry(family).or_default().extend(fx.cases);
    }

    let mut ran_pairs = 0usize;
    let mut skipped_cases: Vec<String> = Vec::new();
    let mut ran_families: Vec<String> = Vec::new();
    let mut skipped_families: Vec<String> = Vec::new();
    let mut errored_cases = 0usize;
    let mut failures: Vec<String> = Vec::new();
    let mut chunking_skips: Vec<String> = Vec::new();

    for (family, cases) in &families {
        let Some(row) = enabled.get(family.as_str()) else {
            skipped_families.push(format!("{family} ({} cases, dynamo_v2=null)", cases.len()));
            continue;
        };
        let use_tokens = row.preferred_input == "tokens";
        // Construct from the REGISTERED id, not the fixture directory name.
        let parser_id = row
            .dynamo_v2
            .as_deref()
            .expect("enabled families have a dynamo_v2 id");
        // HARD FAIL, not a skip: if an enabled family cannot even be constructed,
        // every one of its pairs would error and be counted as a skip, the family
        // would still be printed as exercised, and the run would exit 0 having
        // silently tested nothing for it.
        if let Err(e) = create_tool_parser_for_family(parser_id, &[]) {
            panic!(
                "enabled family {family} (dynamo_v2={parser_id}) cannot be constructed: {e}\n\
                 An enabled family must be testable; fix the registry or the factory."
            );
        }
        ran_families.push(family.clone());
        let mut family_pairs = 0usize;

        let fam_recorded = recorded.get(family.as_str());
        // Only cases WITH a recorded dynamo_v2 expectation can be checked against
        // the absolute oracle. A case that is missing or explicitly `unavailable`
        // is skipped and named — never silently paired against a weaker check.
        let ids: Vec<String> = cases
            .keys()
            .filter(|id| {
                if let Some(why) = cases[id.as_str()].unterminated_reason() {
                    skipped_cases.push(format!("{family}/{id} ({why})"));
                    return false;
                }
                match fam_recorded.and_then(|r| r.get(id.as_str())) {
                    Some(Some(_)) => true,
                    Some(None) => {
                        skipped_cases.push(format!("{family}/{id} (unavailable.dynamo_v2)"));
                        false
                    }
                    None => {
                        skipped_cases.push(format!("{family}/{id} (no recorded dynamo_v2)"));
                        false
                    }
                }
            })
            .cloned()
            .collect(); // BTreeMap => sorted
        for (i, j) in pairs_for(&ids) {
            let ca = &cases[&ids[i]];
            let cb = &cases[&ids[j]];
            let label = format!("{family} {}x{}", ids[i], ids[j]);
            let divergent = [
                allowlist
                    .get(family.as_str())
                    .is_some_and(|c| c.contains_key(&ids[i])),
                allowlist
                    .get(family.as_str())
                    .is_some_and(|c| c.contains_key(&ids[j])),
            ];
            let exp_fwd = [
                fam_recorded.unwrap()[&ids[i]].as_ref().unwrap(),
                fam_recorded.unwrap()[&ids[j]].as_ref().unwrap(),
            ];
            // Both ROLE ASSIGNMENTS: the schedules always give choice 0 the first
            // slot of a round, so running only (a@0, b@1) never exercises b
            // arriving first.
            for (role, ca, cb, exp) in [
                ("AB", ca, cb, exp_fwd),
                ("BA", cb, ca, [exp_fwd[1], exp_fwd[0]]),
            ] {
                let divergent = if role == "AB" {
                    divergent
                } else {
                    [divergent[1], divergent[0]]
                };
                let label = format!("{label}/{role}");
                for schedule in schedules() {
                    let result = if use_tokens {
                        let sa: Vec<Vec<u32>> = ca
                            .chunks
                            .iter()
                            .map(|c| c.delta_token_ids.clone())
                            .collect();
                        let sb: Vec<Vec<u32>> = cb
                            .chunks
                            .iter()
                            .map(|c| c.delta_token_ids.clone())
                            .collect();
                        check_pair(
                            parser_id,
                            &ca.tools,
                            &sa,
                            &cb.tools,
                            &sb,
                            schedule,
                            token_input,
                            &label,
                            exp,
                            divergent,
                            &mut chunking_skips,
                            &mut failures,
                        )
                    } else {
                        let sa: Vec<String> =
                            ca.chunks.iter().map(|c| c.delta_text.clone()).collect();
                        let sb: Vec<String> =
                            cb.chunks.iter().map(|c| c.delta_text.clone()).collect();
                        check_pair(
                            parser_id,
                            &ca.tools,
                            &sa,
                            &cb.tools,
                            &sb,
                            schedule,
                            text_input,
                            &label,
                            exp,
                            divergent,
                            &mut chunking_skips,
                            &mut failures,
                        )
                    };
                    match result {
                        Ok(()) => {
                            ran_pairs += 1;
                            family_pairs += 1;
                        }
                        // A parser error on a specific shape is logged + skipped, not
                        // silently passed — this lane is about isolation, not making
                        // every corpus shape parseable.
                        Err(e) => {
                            errored_cases += 1;
                            eprintln!(
                                "SKIP {label} schedule={}: parser error: {e}",
                                schedule.label()
                            );
                        }
                    }
                }
            }
        }
        // A family that HAD pairs but ran none is a coverage hole and fails.
        // A family with fewer than two pairable cases cannot form a pair at all —
        // a legitimate corpus state (a newly registered v2 family with one fixture,
        // or all-but-one case marked unavailable), so it is a NAMED skip instead of
        // panicking the whole suite.
        if pairs_for(&ids).is_empty() {
            skipped_cases.push(format!(
                "{family}: only {} pairable case(s), cannot form a pair",
                ids.len()
            ));
        } else {
            assert!(
                family_pairs > 0,
                "enabled family {family} had pairable cases but ran 0 pair-schedules"
            );
        }
    }

    eprintln!(
        "v2 interleave isolation: {ran_pairs} pair-schedules over families [{}]",
        ran_families.join(", ")
    );
    for s in &skipped_families {
        eprintln!("SKIP family {s}");
    }
    // Never silently narrow coverage: a case without a usable recorded dynamo_v2
    // expectation is dropped from the pairing set, so say which and why.
    for s in &skipped_cases {
        eprintln!("SKIP case {s}");
    }
    // Chunking-dependent cases are dropped from oracle 1 only, and always named.
    for s in &chunking_skips {
        eprintln!("SKIP oracle1 {s}");
    }
    eprintln!(
        "skipped {} families, {} cases without a recorded dynamo_v2 expectation, \
         {errored_cases} pair-schedules errored; elapsed {:.1}s",
        skipped_families.len(),
        skipped_cases.len(),
        started.elapsed().as_secs_f64()
    );

    assert!(ran_pairs > 0, "no enabled families exercised");
    if !failures.is_empty() {
        for f in &failures {
            eprintln!("FAIL {f}");
        }
        panic!("{} pair-schedules diverged", failures.len());
    }
}

// ── Guards for the two folding/termination classes ──────────────────────────
//
// These pin the behaviour Devin flagged on PR #135. Neither condition occurs in
// the committed corpus today, so without these the guards would be unexercised
// and could rot silently.

#[cfg(test)]
mod recorded_fold_guards {
    use super::*;

    fn write(dir: &Path, name: &str, body: &str) {
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(dir.join(name), body).unwrap();
    }

    /// A NEWER capture recording FEWER chunks must not truncate the oracle: the
    /// older capture's trailing chunks survive, matching `merge_dynamo`'s
    /// per-chunk-index overlay in the canonical lane.
    #[test]
    fn newer_capture_with_fewer_chunks_does_not_truncate() {
        let root = std::env::temp_dir().join(format!("fold-guard-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let old = root.join("dynamo_v2-0.1.1/fam");
        let new = root.join("dynamo_v2-0.1.2/fam");
        // Old capture: name in chunk 0, args in chunk 1, trailing text in chunk 2.
        write(
            &old,
            "F.yaml",
            "cases:\n  c1:\n    chunks:\n    - expected:\n      - index: 0\n        name: get_weather\n    - expected:\n      - index: 0\n        arguments: '{\"a\":1}'\n    - expected: []\n      normal_text: 'tail'\n",
        );
        // New capture: re-records ONLY chunk 0.
        write(
            &new,
            "F.yaml",
            "cases:\n  c1:\n    chunks:\n    - expected:\n      - index: 0\n        name: get_weather_v2\n",
        );

        let dirs = vec![root.join("dynamo_v2-0.1.1"), root.join("dynamo_v2-0.1.2")];
        let inputs = BTreeMap::from([("c1".to_string(), 8usize)]);
        let got = load_recorded(&dirs, Path::new("fam/F.yaml"), &inputs);
        let case = got.get("c1").unwrap().as_ref().unwrap();

        // Chunk 0 comes from the newer capture; chunks 1-2 survive from the older.
        assert_eq!(
            case.calls,
            vec![("get_weather_v2".to_string(), "{\"a\":1}".to_string())],
            "newer capture must override chunk 0 but not drop the older trailing chunks"
        );
        assert_eq!(
            case.normal_text, "tail",
            "trailing normal_text was truncated"
        );
        std::fs::remove_dir_all(&root).unwrap();
    }

    /// old(full chunks) -> mid(`unavailable`) -> new(chunk 0 only): the newest
    /// capture re-enables the case, and the older TRAILING chunks must survive the
    /// intermediate unavailability. Collapsing the case to `None` on the mid
    /// capture would drop them and disagree with canonical `merge_dynamo`.
    #[test]
    fn unavailable_then_reenabled_keeps_older_trailing_chunks() {
        let root = std::env::temp_dir().join(format!("fold-guard-reenable-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        write(
            &root.join("dynamo_v2-0.1.1/fam"),
            "F.yaml",
            "cases:\n  c1:\n    chunks:\n    - expected:\n      - index: 0\n        name: old_name\n    - expected:\n      - index: 0\n        arguments: '{\"a\":1}'\n    - expected: []\n      normal_text: 'tail'\n",
        );
        write(
            &root.join("dynamo_v2-0.1.2/fam"),
            "F.yaml",
            "cases:\n  c1:\n    unavailable: 'temporarily unsupported'\n",
        );
        write(
            &root.join("dynamo_v2-0.1.3/fam"),
            "F.yaml",
            "cases:\n  c1:\n    chunks:\n    - expected:\n      - index: 0\n        name: new_name\n",
        );
        let dirs = vec![
            root.join("dynamo_v2-0.1.1"),
            root.join("dynamo_v2-0.1.2"),
            root.join("dynamo_v2-0.1.3"),
        ];
        let inputs = BTreeMap::from([("c1".to_string(), 8usize)]);
        let got = load_recorded(&dirs, Path::new("fam/F.yaml"), &inputs);
        let case = got
            .get("c1")
            .unwrap()
            .as_ref()
            .expect("newest capture re-enables the case");
        assert_eq!(
            case.calls,
            vec![("new_name".to_string(), "{\"a\":1}".to_string())],
            "intermediate `unavailable` must not destroy older trailing chunks"
        );
        assert_eq!(case.normal_text, "tail", "trailing normal_text was dropped");
        std::fs::remove_dir_all(&root).unwrap();
    }

    /// An `unavailable` marker in the newest capture wins over older expectations.
    #[test]
    fn newest_unavailable_marks_case_unusable() {
        let root = std::env::temp_dir().join(format!("fold-guard-unavail-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        write(
            &root.join("dynamo_v2-0.1.1/fam"),
            "F.yaml",
            "cases:\n  c1:\n    chunks:\n    - expected:\n      - index: 0\n        name: n\n",
        );
        write(
            &root.join("dynamo_v2-0.1.2/fam"),
            "F.yaml",
            "cases:\n  c1:\n    unavailable: 'token parser cannot split this'\n",
        );
        let dirs = vec![root.join("dynamo_v2-0.1.1"), root.join("dynamo_v2-0.1.2")];
        let inputs = BTreeMap::from([("c1".to_string(), 8usize)]);
        let got = load_recorded(&dirs, Path::new("fam/F.yaml"), &inputs);
        assert!(
            got.get("c1").unwrap().is_none(),
            "newest unavailable must win over an older recorded expectation"
        );
        std::fs::remove_dir_all(&root).unwrap();
    }

    fn case_with_finish_at(idx: usize, n: usize) -> Case {
        Case {
            tools: Vec::new(),
            chunks: (0..n)
                .map(|i| Chunk {
                    delta_text: String::new(),
                    delta_token_ids: Vec::new(),
                    finish_reason: (i == idx).then(|| "stop".to_string()),
                })
                .collect(),
        }
    }

    /// A stale recording LONGER than the current input must be truncated, not
    /// kept: otherwise the oracle demands output the trimmed fixture can no
    /// longer produce, and this lane reports it as an isolation divergence.
    #[test]
    fn recorded_chunks_beyond_the_input_are_dropped() {
        let root = std::env::temp_dir().join(format!("fold-guard-trunc-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        write(
            &root.join("dynamo_v2-0.1.1/fam"),
            "F.yaml",
            "cases:\n  c1:\n    chunks:\n    - expected:\n      - index: 0\n        name: kept\n    - expected:\n      - index: 1\n        name: dropped\n      normal_text: 'gone'\n",
        );
        let dirs = vec![root.join("dynamo_v2-0.1.1")];
        // Input has been trimmed to ONE chunk; the second recorded chunk is stale.
        let inputs = BTreeMap::from([("c1".to_string(), 1usize)]);
        let got = load_recorded(&dirs, Path::new("fam/F.yaml"), &inputs);
        let case = got.get("c1").unwrap().as_ref().unwrap();
        assert_eq!(
            case.calls,
            vec![("kept".to_string(), String::new())],
            "recorded chunks beyond the input length must be dropped"
        );
        assert_eq!(
            case.normal_text, "",
            "stale trailing normal_text must be dropped"
        );
        std::fs::remove_dir_all(&root).unwrap();
    }

    /// A terminator on the last chunk is the normal shape and must NOT be skipped;
    /// one before it must be, since this lane finishes only at end of stream.
    #[test]
    fn mid_stream_finish_reason_is_detected() {
        assert!(
            case_with_finish_at(2, 3).unterminated_reason().is_none(),
            "finish_reason on the last chunk is the normal shape"
        );
        assert_eq!(
            case_with_finish_at(0, 3).unterminated_reason(),
            Some("finish_reason mid-stream"),
            "finish_reason before the last chunk must be detected"
        );
        assert_eq!(
            case_with_finish_at(1, 3).unterminated_reason(),
            Some("finish_reason mid-stream"),
            "finish_reason mid-stream must be detected"
        );
        // A case that never signals an end is excluded too: the recorded
        // reference for it was captured without the parser's closing output, so
        // comparing this lane's closing output against it would surface as a
        // bogus per-choice leakage failure.
        assert_eq!(
            case_with_finish_at(99, 3).unterminated_reason(),
            Some("no finish_reason on the last chunk"),
            "a case with no terminator at all must be excluded"
        );
        assert_eq!(
            case_with_finish_at(99, 0).unterminated_reason(),
            Some("no chunks"),
            "an empty case must be excluded"
        );
    }
}
