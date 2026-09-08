// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Incremental lexer over one guided `required` tool-call payload.
//!
//! # Why this is one type
//!
//! Streaming a call before its payload has closed needs four facts that only a
//! JSON lexer can answer: which call element the scan is inside, that element's
//! decoded `name`, where its argument OBJECT begins, and which bytes of that
//! object are safe to release. An earlier revision answered them with three
//! independent `str::find` helpers, and each one was wrong in its own way —
//! `"name"` was matched anywhere in the payload (so `{"x":"name","arguments":…}`
//! emitted a call named `arguments`), `_` decoded to the literal characters
//! `u005f`, the `parameters` alias was not recognised at all, and every helper
//! rescanned the whole accumulated buffer on every push, making a
//! character-at-a-time stream quadratic.
//!
//! Those are not four bugs, they are one: lexical state with no owner. This type
//! is that owner. It holds the string/escape state, the brace depth, the current
//! element index, the decoded name, and the released-byte offset in one place, and
//! it advances strictly FORWARD — each call consumes only the bytes appended since
//! the last one, so driving a payload one character at a time costs O(n) total.
//!
//! # What it deliberately does not do
//!
//! It does not validate. `parse_required_guided_calls` remains the only judge of
//! whether a payload is a legal call set, and the completion path still runs it.
//! The cursor's single question is narrower: *has enough arrived that a name and
//! an argument OBJECT can no longer turn into something else?* Requiring the
//! argument value to open with `{` is what makes that safe — a `null`, a string,
//! a number or an array never reaches the commit point, so the shapes the
//! contract voids are never put on the wire in the first place.
//!
//! # The two payload shapes
//!
//! `tool_choice` decides what the backend is constrained to emit, so it decides
//! what the cursor is lexing:
//!
//! - **required** ([`GuidedJsonCursor::new`]) — an array (or a bare object) of
//!   `{"name": …, "arguments": …}` envelopes. The name has to be *found*, and the
//!   commit point is the `{` that opens that envelope's argument object.
//! - **named** ([`GuidedJsonCursor::named`]) — the chosen tool's BARE argument
//!   object, with no envelope at all. The name is not in the payload: the request
//!   fixed it before the first byte arrived. So the commit point is the payload's
//!   own opening `{`, and the whole payload from that brace to its matching `}`
//!   is the argument set.
//!
//! Both modes keep the same rule about what may go on the wire — the first
//! non-whitespace byte of the argument set must be `{`. A named payload that opens
//! as a string, a number, `null` or an array is not an argument set, cannot be
//! bound to the tool, and stays on the buffered path that already surfaces it as
//! text. Committing it early would be the same unwithdrawable mistake the required
//! mode refuses to make.

use crate::tool_calling::traits::ToolCallDelta;

/// The two spellings a required-choice call may use for its argument object.
///
/// Both are accepted by `parse_required_guided_calls`; a call carrying BOTH is
/// ambiguous and that function voids it. The cursor has to know both spellings for
/// the same reason — recognising only `arguments` silently streamed nothing for a
/// `parameters` call, and the assembled result arrived with empty arguments.
const ARGUMENT_ALIASES: [&str; 2] = ["arguments", "parameters"];

/// Which payload shape the cursor is lexing.
///
/// A mode rather than an `Option<String>` threaded through every method: the two
/// shapes disagree about where the name comes from and where the argument object
/// starts, and those two facts are the whole difference. Keeping them in one field
/// means `reset` restores the mode too — a cursor that forgot it was named would
/// silently lex the next request's arguments as a required envelope.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Mode {
    /// `tool_choice=required`: envelopes carrying their own `name`.
    Required,
    /// `tool_choice={"type":"function","function":{"name":…}}`: bare arguments for
    /// the tool the request already named.
    Named { name: String },
}

/// Where the scanner sits inside a call object.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Slot {
    /// Between `{` or `,` and the next key.
    Key,
    /// A key literal closed; waiting for its `:`.
    Colon,
    /// Past the `:`; the next token is that key's value.
    Value,
}

/// A call the cursor has already put on the wire.
///
/// `released` is a byte count into the call's RAW argument object — the same bytes
/// `parse_required_guided_calls` will later hand back as that call's arguments — so
/// the completion path can emit exactly the remainder without re-deriving spans.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommittedCall {
    pub index: usize,
    pub name: String,
    pub released: usize,
    /// A second argument alias appeared AFTER this call was committed.
    ///
    /// `parse_required_guided_calls` voids an ambiguous call, but this one is
    /// already on the wire and a fragment cannot be unsaid. The completion path
    /// warns rather than pretending it can retract.
    pub ambiguous: bool,
}

/// Per-element scan state, reset at every call boundary.
#[derive(Debug, Default, Clone)]
struct Element {
    name: Option<String>,
    /// Offset of the `{` opening the argument object.
    args_start: Option<usize>,
    /// Offset just past the argument object's `}`, once it closed.
    args_end: Option<usize>,
    /// The scan is inside the argument object right now.
    in_args: bool,
    /// Both aliases appeared.
    ambiguous: bool,
    /// An alias appeared whose value was not an object.
    ///
    /// Blocks the commit even if the OTHER alias later supplies a good object:
    /// that shape is ambiguous too, and the buffered path voids it correctly.
    non_object_alias: bool,
    committed: bool,
    /// Bytes of the argument object already released, relative to `args_start`.
    released: usize,
}

/// Forward-only lexer over one guided payload.
#[derive(Debug)]
pub struct GuidedJsonCursor {
    /// Which payload shape is being lexed; fixed at construction by `tool_choice`.
    mode: Mode,
    /// Bytes already lexed. The scan never revisits them.
    scanned: usize,
    depth: i32,
    in_string: bool,
    escaped: bool,
    /// Opening quote of a key or name literal the cursor needs. The raw slice is
    /// decoded once, when its closing quote arrives; JSON escape semantics stay
    /// owned by `serde_json`.
    literal_start: Option<usize>,
    /// Depth at which a call object's keys sit: 2 under an array root, 1 under a
    /// bare object root.
    key_depth: i32,
    root_seen: bool,
    /// The payload does not open as a call shape, so nothing may ever commit.
    disabled: bool,
    slot: Slot,
    pending_key: Option<String>,
    element: Element,
    index: usize,
    committed: Vec<CommittedCall>,
    /// Absolute end, in the fed payload's coordinates, of the last byte released
    /// as a call fragment. Recovery may only emit bytes at or after this offset:
    /// anything before it is already on the wire and cannot be unsaid.
    released_end: usize,
}

impl Default for GuidedJsonCursor {
    fn default() -> Self {
        Self::new()
    }
}

impl GuidedJsonCursor {
    /// A cursor for a `required` payload: envelopes that carry their own names.
    pub fn new() -> Self {
        Self::in_mode(Mode::Required)
    }

    /// A cursor for a NAMED choice: the payload is `tool_name`'s bare argument
    /// object, and the name rides the first delta because the request already
    /// fixed it.
    pub fn named(tool_name: impl Into<String>) -> Self {
        Self::in_mode(Mode::Named {
            name: tool_name.into(),
        })
    }

    fn in_mode(mode: Mode) -> Self {
        // The named mode's name is known NOW, before any byte arrives, so it is
        // seeded here rather than discovered by the lexer. Everything downstream —
        // `maybe_commit`, `CommittedCall`, `flush` — then works on both shapes
        // unchanged, which is why there is one commit rule and not two.
        let element = Element {
            name: match &mode {
                Mode::Required => None,
                Mode::Named { name } => Some(name.clone()),
            },
            ..Element::default()
        };
        Self {
            mode,
            scanned: 0,
            depth: 0,
            in_string: false,
            escaped: false,
            literal_start: None,
            key_depth: 1,
            root_seen: false,
            disabled: false,
            slot: Slot::Key,
            pending_key: None,
            element,
            index: 0,
            committed: Vec::new(),
            released_end: 0,
        }
    }

    /// Back to the start of a payload — in the SAME mode. `tool_choice` belongs to
    /// the request, not to the payload, so a reset between chunks (or between
    /// requests, where the caller re-initialises) must not turn a named cursor into
    /// a required one.
    pub fn reset(&mut self) {
        *self = Self::in_mode(self.mode.clone());
    }

    /// Calls already on the wire, in emission order.
    pub fn committed(&self) -> &[CommittedCall] {
        &self.committed
    }

    pub fn has_committed(&self) -> bool {
        !self.committed.is_empty()
    }

    /// Absolute end of the bytes already released as call fragments, in the
    /// coordinates of the payload passed to [`Self::advance`].
    pub fn released_end(&self) -> usize {
        self.released_end
    }

    /// Lex the bytes appended since the last advance, emitting whatever became
    /// safe to commit.
    ///
    /// `payload` is the WHOLE accumulated payload, not just the new chunk — the
    /// cursor slices it from `scanned`, so the caller does not have to track which
    /// bytes it has already handed over.
    pub fn advance(&mut self, payload: &str, out: &mut Vec<ToolCallDelta>) {
        if self.disabled || payload.len() <= self.scanned {
            // Truncation (a reset mid-stream) would make the retained offsets lie.
            // The caller resets the cursor with the payload; nothing to do here.
            return;
        }

        let mut cut = self.scanned;
        for (relative, ch) in payload[self.scanned..].char_indices() {
            let at = self.scanned + relative;
            cut = at + ch.len_utf8();
            self.step(payload, at, cut, ch, out);
            if self.disabled {
                break;
            }
        }
        self.scanned = cut;
        self.maybe_commit(out);
        self.flush(payload, cut, out);
    }

    /// One character of the lex. `cut` is the offset just past `ch`, so a flush
    /// triggered mid-advance still sees the bytes this character contributed.
    fn step(
        &mut self,
        payload: &str,
        at: usize,
        cut: usize,
        ch: char,
        out: &mut Vec<ToolCallDelta>,
    ) {
        if let Mode::Named { .. } = self.mode {
            self.step_named(payload, at, cut, ch, out);
            return;
        }
        if self.in_string {
            self.step_in_string(payload, at, ch);
            return;
        }

        match ch {
            '"' => {
                if !self.root_seen {
                    // A required payload opens with `[` or `{`; a bare string is not
                    // a call shape and the buffered path owns it.
                    self.disabled = true;
                    return;
                }
                self.in_string = true;
                // Capture only the literals the cursor needs: every key at call-key
                // depth, and the `name` value. Capturing every string in the payload
                // would allocate once per argument value for nothing.
                let wanted = self.depth == self.key_depth
                    && match self.slot {
                        Slot::Key => true,
                        Slot::Value => self.pending_key.as_deref() == Some("name"),
                        Slot::Colon => false,
                    };
                self.literal_start = wanted.then_some(at);
            }
            '{' | '[' => {
                if !self.root_seen {
                    self.root_seen = true;
                    self.key_depth = if ch == '[' { 2 } else { 1 };
                }
                let opening_args = ch == '{'
                    && self.depth == self.key_depth
                    && self.slot == Slot::Value
                    && self.is_alias(self.pending_key.as_deref());
                let starts_element = self.depth == self.key_depth - 1 && ch == '{';
                self.depth += 1;
                if opening_args {
                    if self.element.args_start.is_some() {
                        self.element.ambiguous = true;
                    } else {
                        self.element.args_start = Some(at);
                        self.element.in_args = true;
                    }
                } else if starts_element {
                    self.slot = Slot::Key;
                    self.pending_key = None;
                } else if self.depth == self.key_depth
                    && self.slot == Slot::Value
                    && self.pending_key.as_deref() == Some("name")
                {
                    // `"name"` bound to a non-string: never commits.
                    self.element.non_object_alias = true;
                }
                self.maybe_commit(out);
            }
            '}' | ']' => {
                let closing_args =
                    self.element.in_args && self.depth == self.key_depth + 1 && ch == '}';
                let closes_element = self.depth == self.key_depth && ch == '}';
                self.depth -= 1;
                if closing_args {
                    self.element.args_end = Some(at + 1);
                    self.element.in_args = false;
                    self.maybe_commit(out);
                    self.flush(payload, cut, out);
                } else if closes_element {
                    // A name that closed AFTER its arguments only becomes committable
                    // here, so commit before the final flush or its bytes are lost
                    // with the element.
                    self.maybe_commit(out);
                    self.flush(payload, at, out);
                    self.finish_element();
                }
            }
            ':' if self.depth == self.key_depth && self.slot == Slot::Colon => {
                self.slot = Slot::Value;
            }
            ',' if self.depth == self.key_depth => {
                self.slot = Slot::Key;
                self.pending_key = None;
            }
            _ if ch.is_whitespace() => {}
            _ => {
                if !self.root_seen {
                    self.disabled = true;
                    return;
                }
                if self.depth == self.key_depth && self.slot == Slot::Value {
                    // A bare literal (`null`, a number, `true`) bound to an argument
                    // alias or to `name`: neither can ever become an object, so this
                    // element must stay on the buffered path.
                    if self.is_alias(self.pending_key.as_deref())
                        || self.pending_key.as_deref() == Some("name")
                    {
                        self.element.non_object_alias = true;
                    }
                }
            }
        }
    }

    /// One character of the lex for a NAMED choice.
    ///
    /// There is no envelope to walk: the payload IS the argument object, so this
    /// only has to find its opening `{`, track brace depth through strings, and
    /// stop at the matching `}`. It shares `Element`, `maybe_commit` and `flush`
    /// with the required mode so the commit rule and the released-byte accounting
    /// have exactly one implementation.
    fn step_named(
        &mut self,
        payload: &str,
        at: usize,
        cut: usize,
        ch: char,
        out: &mut Vec<ToolCallDelta>,
    ) {
        if self.in_string {
            // `literal_start` is never set in this mode, so `close_literal` is a
            // no-op: the only reason to track strings here is that a brace or a
            // quote inside one must not move the depth.
            self.step_in_string(payload, at, ch);
            return;
        }

        if self.element.args_start.is_none() {
            // Before the object opener. Whitespace is structural; anything that is
            // not `{` means the payload is not an argument set at all, and the
            // buffered path owns it.
            if ch.is_whitespace() {
                return;
            }
            if ch != '{' {
                self.disabled = true;
                return;
            }
            self.element.args_start = Some(at);
            self.element.in_args = true;
            self.depth = 1;
            self.root_seen = true;
            // The name is already known, so the commit lands on this brace — the
            // earliest instant the payload is provably an argument set.
            self.maybe_commit(out);
            return;
        }

        if self.element.args_end.is_some() {
            // Past the object's `}`. Trailing bytes belong to the buffer's tail
            // handling, never to the arguments.
            return;
        }

        match ch {
            '"' => self.in_string = true,
            '{' | '[' => self.depth += 1,
            '}' | ']' => {
                // Depth back to zero ends the argument set. A MISMATCHED closer
                // ends it here too: the payload is malformed, the buffered path
                // will say so, and the alternative is releasing every byte after
                // it as arguments because the real close never arrives.
                self.depth -= 1;
                if self.depth == 0 {
                    self.element.args_end = Some(at + ch.len_utf8());
                    self.element.in_args = false;
                    self.flush(payload, cut, out);
                }
            }
            _ => {}
        }
    }

    /// One character while inside a string literal.
    fn step_in_string(&mut self, payload: &str, at: usize, ch: char) {
        if self.escaped {
            self.escaped = false;
            return;
        }
        match ch {
            '\\' => self.escaped = true,
            '"' => {
                self.in_string = false;
                self.close_literal(payload, at);
            }
            _ => {}
        }
    }

    /// A captured literal closed: it is either this element's key or its name.
    fn close_literal(&mut self, payload: &str, end: usize) {
        let Some(start) = self.literal_start.take() else {
            return;
        };
        let Ok(literal) = serde_json::from_str::<String>(&payload[start..=end]) else {
            return;
        };
        match self.slot {
            Slot::Key => {
                if self.is_alias(Some(literal.as_str())) && self.element.args_start.is_some() {
                    self.element.ambiguous = true;
                }
                self.pending_key = Some(literal);
                self.slot = Slot::Colon;
            }
            Slot::Value => {
                if self.pending_key.as_deref() == Some("name") {
                    self.element.name = Some(literal);
                }
            }
            Slot::Colon => {}
        }
    }

    fn is_alias(&self, key: Option<&str>) -> bool {
        key.is_some_and(|key| ARGUMENT_ALIASES.contains(&key))
    }

    /// Put the current element on the wire once it can no longer change shape.
    ///
    /// Both facts are required: the name has closed, AND the argument value has
    /// been seen to open with `{`. Committing on the name alone is what let a
    /// `null`, a string and a non-object argument set reach the caller as calls the
    /// contract says must become text.
    fn maybe_commit(&mut self, out: &mut Vec<ToolCallDelta>) {
        if self.element.committed
            || self.element.ambiguous
            || self.element.non_object_alias
            || self.element.args_start.is_none()
        {
            return;
        }
        let Some(name) = self.element.name.clone() else {
            return;
        };
        self.element.committed = true;
        self.committed.push(CommittedCall {
            index: self.index,
            name: name.clone(),
            released: 0,
            ambiguous: false,
        });
        out.push(ToolCallDelta {
            tool_index: self.index,
            name: Some(name),
            arguments: String::new(),
            complete: false,
        });
    }

    /// Release argument bytes the current element has accumulated since the last
    /// fragment.
    fn flush(&mut self, payload: &str, cut: usize, out: &mut Vec<ToolCallDelta>) {
        if !self.element.committed {
            return;
        }
        let Some(start) = self.element.args_start else {
            return;
        };
        // Never past the argument object's own close: the bytes after it belong to
        // the call envelope, not to the arguments.
        let bound = self.element.args_end.unwrap_or(cut).min(cut);
        let from = start + self.element.released;
        if bound <= from {
            return;
        }
        let fragment = payload[from..bound].to_string();
        self.element.released = bound - start;
        self.released_end = self.released_end.max(bound);
        if let Some(record) = self.committed.last_mut() {
            record.released = self.element.released;
            record.ambiguous = self.element.ambiguous;
        }
        out.push(ToolCallDelta {
            tool_index: self.index,
            name: None,
            arguments: fragment,
            complete: false,
        });
    }

    /// The current call object closed; move to the next element.
    fn finish_element(&mut self) {
        if self.element.committed
            && let Some(record) = self.committed.last_mut()
        {
            record.ambiguous = self.element.ambiguous;
        }
        // ALWAYS advance, including for an element the cursor could not commit. The
        // index is the element's position in the payload array, and the completion
        // path matches committed records against `serde`'s element order — skipping
        // an uncommittable element here would slide every later call onto the wrong
        // index.
        self.index += 1;
        self.element = Element::default();
        self.slot = Slot::Key;
        self.pending_key = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Drive a payload one character at a time and collect every delta.
    fn stream(payload: &str) -> Vec<ToolCallDelta> {
        let mut cursor = GuidedJsonCursor::new();
        let mut out = Vec::new();
        let mut seen = String::new();
        for ch in payload.chars() {
            seen.push(ch);
            cursor.advance(&seen, &mut out);
        }
        out
    }

    /// Drive a NAMED payload one character at a time and collect every delta.
    fn stream_named(tool: &str, payload: &str) -> Vec<ToolCallDelta> {
        let mut cursor = GuidedJsonCursor::named(tool);
        let mut out = Vec::new();
        let mut seen = String::new();
        for ch in payload.chars() {
            seen.push(ch);
            cursor.advance(&seen, &mut out);
        }
        out
    }

    /// Names carried, in order.
    fn names(deltas: &[ToolCallDelta]) -> Vec<(usize, String)> {
        deltas
            .iter()
            .filter_map(|d| d.name.clone().map(|n| (d.tool_index, n)))
            .collect()
    }

    /// Argument bytes reassembled per tool index.
    fn arguments(deltas: &[ToolCallDelta]) -> Vec<(usize, String)> {
        let mut joined: Vec<(usize, String)> = Vec::new();
        for delta in deltas {
            if delta.arguments.is_empty() {
                continue;
            }
            match joined
                .iter_mut()
                .find(|(index, _)| *index == delta.tool_index)
            {
                Some((_, text)) => text.push_str(&delta.arguments),
                None => joined.push((delta.tool_index, delta.arguments.clone())),
            }
        }
        joined
    }

    #[test]
    fn a_name_key_is_a_key_not_any_occurrence_of_the_word() {
        // The old `str::find("\"name\"")` matched this VALUE and then read the next
        // string as the function name, emitting a call named `arguments`.
        let deltas = stream(r#"[{"x":"name","arguments":{"city":"Paris"}}]"#);
        assert_eq!(names(&deltas), Vec::<(usize, String)>::new());
        assert!(
            arguments(&deltas).is_empty(),
            "a nameless element must not stream"
        );
    }

    #[test]
    fn escaped_names_decode() {
        // NOT a raw literal: the payload must carry the six characters
        // `_`, which JSON decodes to `_`. The old decoder pushed the
        // escape's payload characters verbatim and produced `getu005fweather`.
        let payload = "[{\"name\":\"get\\u005fweather\",\"arguments\":{}}]";
        assert!(
            payload.contains("\\u005f"),
            "the escape must survive to the test"
        );
        assert_eq!(
            names(&stream(payload)),
            vec![(0, "get_weather".to_string())]
        );

        let deltas = stream(r#"[{"name":"a\"b\\c\nd","arguments":{}}]"#);
        assert_eq!(names(&deltas), vec![(0, "a\"b\\c\nd".to_string())]);
    }

    #[test]
    fn surrogate_pairs_decode_to_one_character() {
        // Escaped, as a JSON encoder outside the BMP actually emits it. Decoding
        // each half alone yields two replacement characters.
        let payload = "[{\"name\":\"a\\ud83d\\ude00b\",\"arguments\":{}}]";
        assert!(
            payload.contains("\\ud83d"),
            "the escape must survive to the test"
        );
        assert_eq!(
            names(&stream(payload)),
            vec![(0, "a\u{1F600}b".to_string())]
        );
    }

    #[test]
    fn the_parameters_alias_streams_too() {
        let deltas = stream(r#"[{"name":"get_weather","parameters":{"city":"Tokyo"}}]"#);
        assert_eq!(names(&deltas), vec![(0, "get_weather".to_string())]);
        assert_eq!(
            arguments(&deltas),
            vec![(0, r#"{"city":"Tokyo"}"#.to_string())]
        );
    }

    #[test]
    fn parallel_calls_get_distinct_indices() {
        let deltas =
            stream(r#"[{"name":"a","arguments":{"x":1}},{"name":"b","arguments":{"y":2}}]"#);
        assert_eq!(
            names(&deltas),
            vec![(0, "a".to_string()), (1, "b".to_string())]
        );
        assert_eq!(
            arguments(&deltas),
            vec![(0, r#"{"x":1}"#.to_string()), (1, r#"{"y":2}"#.to_string())]
        );
    }

    #[test]
    fn arguments_reassemble_byte_for_byte_at_every_split() {
        let payload = r#"[{"name":"f","arguments":{"a":"x y","b":[1,2],"c":{"d":"}"}}}]"#;
        let expected = r#"{"a":"x y","b":[1,2],"c":{"d":"}"}}"#;
        let deltas = stream(payload);
        assert_eq!(arguments(&deltas), vec![(0, expected.to_string())]);
    }

    #[test]
    fn a_brace_inside_an_argument_string_does_not_close_the_object() {
        let deltas = stream(r#"[{"name":"f","arguments":{"s":"}}]"}}]"#);
        assert_eq!(arguments(&deltas), vec![(0, r#"{"s":"}}]"}"#.to_string())]);
    }

    #[test]
    fn non_object_arguments_never_commit() {
        for payload in [
            r#"[{"name":"f","arguments":"just a string"}]"#,
            r#"[{"name":"f","arguments":null}]"#,
            r#"[{"name":"f","arguments":[1,2]}]"#,
            r#"[{"name":"f","arguments":7}]"#,
        ] {
            let deltas = stream(payload);
            assert!(
                deltas.is_empty(),
                "{payload} committed a call the contract voids: {deltas:?}"
            );
        }
    }

    #[test]
    fn a_parameterless_call_stays_on_the_buffered_path() {
        // No argument key at all means no arguments, which is VALID — but there is
        // no object opener to commit on, so the buffered path emits it with `{}`.
        let deltas = stream(r#"[{"name":"f"}]"#);
        assert!(deltas.is_empty());
    }

    #[test]
    fn both_aliases_present_never_commits() {
        let deltas = stream(r#"[{"name":"f","arguments":{"a":1},"parameters":{"b":2}}]"#);
        assert!(
            names(&deltas).is_empty() || deltas.iter().all(|d| d.tool_index == 0),
            "ambiguity must not produce a second call"
        );
    }

    #[test]
    fn a_name_after_its_arguments_still_commits() {
        let deltas = stream(r#"[{"arguments":{"city":"Paris"},"name":"get_weather"}]"#);
        assert_eq!(names(&deltas), vec![(0, "get_weather".to_string())]);
        assert_eq!(
            arguments(&deltas),
            vec![(0, r#"{"city":"Paris"}"#.to_string())]
        );
    }

    #[test]
    fn a_bare_object_payload_is_one_call() {
        let deltas = stream(r#"{"name":"f","arguments":{"x":1}}"#);
        assert_eq!(names(&deltas), vec![(0, "f".to_string())]);
        assert_eq!(arguments(&deltas), vec![(0, r#"{"x":1}"#.to_string())]);
    }

    #[test]
    fn a_non_call_payload_disables_the_cursor() {
        assert!(stream(r#""just a string""#).is_empty());
        assert!(stream("42").is_empty());
    }

    /// Byte at which the first name-carrying delta appeared, feeding one character
    /// at a time.
    fn name_arrives_at(payload: &str) -> usize {
        let mut cursor = GuidedJsonCursor::new();
        let mut out = Vec::new();
        let mut seen = String::new();
        for (index, ch) in payload.char_indices() {
            seen.push(ch);
            let before = out.len();
            cursor.advance(&seen, &mut out);
            if out[before..].iter().any(|d| d.name.is_some()) {
                return index;
            }
        }
        panic!("no name delta for {payload:?}");
    }

    #[test]
    fn the_name_lands_at_the_argument_opener_not_at_the_close() {
        // The commit point IS the `{` that opens the arguments — that is what makes
        // a non-object argument set unstreamable. So the win is not a fixed
        // fraction of the payload; it is everything the model still has to generate
        // AFTER that brace, which is the whole argument body.
        let payload = r#"[{"name":"get_weather","arguments":{"city":"Paris"}}]"#;
        let opener = payload
            .find(r#""arguments":{"#)
            .expect("an argument opener")
            + r#""arguments":"#.len();
        assert_eq!(name_arrives_at(payload), opener);
        assert!(name_arrives_at(payload) < payload.len() - 1);
    }

    #[test]
    fn a_large_argument_body_is_almost_entirely_streamed() {
        // The reported shape: a call whose arguments take seconds to generate. The
        // buffered path emits nothing until the final byte; here the name is out
        // after the first few percent and the body follows as fragments.
        let body: String = (0..200)
            .map(|i| format!(r#""k{i}":"v{i}","#))
            .collect::<String>();
        let payload = format!(r#"[{{"name":"f","arguments":{{{body}"last":"x"}}}}]"#);
        let at = name_arrives_at(&payload);
        assert!(
            at * 20 < payload.len(),
            "name arrived at byte {at} of {} — over 5% of the payload",
            payload.len()
        );

        // And the body really does arrive in pieces, not one burst.
        let deltas = stream(&payload);
        let frames = deltas.iter().filter(|d| !d.arguments.is_empty()).count();
        assert!(frames > 100, "arguments arrived in {frames} frame(s)");
    }

    #[test]
    fn multi_byte_argument_content_is_never_split_mid_character() {
        let payload = r#"[{"name":"f","arguments":{"city":"東京","emoji":"😀"}}]"#;
        let deltas = stream(payload);
        assert_eq!(
            arguments(&deltas),
            vec![(0, r#"{"city":"東京","emoji":"😀"}"#.to_string())]
        );
        for delta in &deltas {
            assert!(std::str::from_utf8(delta.arguments.as_bytes()).is_ok());
        }
    }

    #[test]
    fn whole_input_and_every_split_agree() {
        let payload = r#"[{"name":"a","arguments":{"x":"1"}},{"name":"b","arguments":{"y":"2"}}]"#;
        let whole = {
            let mut cursor = GuidedJsonCursor::new();
            let mut out = Vec::new();
            cursor.advance(payload, &mut out);
            out
        };
        let per_char = stream(payload);
        assert_eq!(names(&whole), names(&per_char));
        assert_eq!(arguments(&whole), arguments(&per_char));

        for split in 1..payload.len() {
            if !payload.is_char_boundary(split) {
                continue;
            }
            let mut cursor = GuidedJsonCursor::new();
            let mut out = Vec::new();
            cursor.advance(&payload[..split], &mut out);
            cursor.advance(payload, &mut out);
            assert_eq!(names(&out), names(&whole), "names differ at split {split}");
            assert_eq!(
                arguments(&out),
                arguments(&whole),
                "arguments differ at split {split}"
            );
        }
    }

    #[test]
    fn committed_records_track_released_bytes() {
        let payload = r#"[{"name":"f","arguments":{"x":1}}]"#;
        let mut cursor = GuidedJsonCursor::new();
        let mut out = Vec::new();
        cursor.advance(payload, &mut out);
        let committed = cursor.committed();
        assert_eq!(committed.len(), 1);
        assert_eq!(committed[0].index, 0);
        assert_eq!(committed[0].name, "f");
        assert_eq!(committed[0].released, r#"{"x":1}"#.len());
    }

    // ---- named choice: the payload is the tool's BARE argument object ----

    #[test]
    fn a_named_payload_streams_its_bare_arguments() {
        let payload = r#"{"city":"Paris","unit":"c"}"#;
        let deltas = stream_named("get_weather", payload);
        assert_eq!(names(&deltas), vec![(0, "get_weather".to_string())]);
        assert_eq!(arguments(&deltas), vec![(0, payload.to_string())]);
        let frames = deltas.iter().filter(|d| !d.arguments.is_empty()).count();
        assert!(frames > 1, "arguments arrived in {frames} frame(s)");
    }

    #[test]
    fn a_named_name_rides_only_the_first_delta() {
        let deltas = stream_named("get_weather", r#"{"city":"Paris"}"#);
        let carrying: Vec<usize> = deltas
            .iter()
            .enumerate()
            .filter(|(_, d)| d.name.is_some())
            .map(|(at, _)| at)
            .collect();
        assert_eq!(carrying, vec![0], "the name must ride exactly one delta");
        assert!(
            deltas[0].arguments.is_empty(),
            "the commit frame carries the name, not bytes: {:?}",
            deltas[0]
        );
    }

    #[test]
    fn a_named_payload_that_is_not_an_object_never_commits() {
        for payload in [
            r#""just a string""#,
            "42",
            "null",
            "[1,2]",
            "true",
            r#"  "leading whitespace then a string""#,
        ] {
            let deltas = stream_named("get_weather", payload);
            assert!(
                deltas.is_empty(),
                "{payload} committed a shape that is not an argument set: {deltas:?}"
            );
        }
    }

    #[test]
    fn a_named_payload_skips_whitespace_before_its_opener() {
        let deltas = stream_named("get_weather", "  \n\t{\"city\":\"Paris\"}");
        assert_eq!(names(&deltas), vec![(0, "get_weather".to_string())]);
        assert_eq!(
            arguments(&deltas),
            vec![(0, r#"{"city":"Paris"}"#.to_string())],
            "leading whitespace is structural, not an argument byte"
        );
    }

    #[test]
    fn a_named_brace_inside_a_string_does_not_close_the_object() {
        let deltas = stream_named("f", r#"{"s":"}{[]"}"#);
        assert_eq!(arguments(&deltas), vec![(0, r#"{"s":"}{[]"}"#.to_string())]);
    }

    #[test]
    fn a_named_payload_never_releases_past_its_closing_brace() {
        // Bytes after the object belong to the buffer's tail handling. Releasing
        // them would put non-argument bytes into the tool's argument set.
        let payload = r#"{"city":"Paris"} trailing"#;
        let deltas = stream_named("get_weather", payload);
        assert_eq!(
            arguments(&deltas),
            vec![(0, r#"{"city":"Paris"}"#.to_string())]
        );
        let mut cursor = GuidedJsonCursor::named("get_weather");
        let mut out = Vec::new();
        cursor.advance(payload, &mut out);
        assert_eq!(cursor.released_end(), r#"{"city":"Paris"}"#.len());
    }

    #[test]
    fn named_committed_records_track_released_bytes() {
        let payload = r#"{"x":1}"#;
        let mut cursor = GuidedJsonCursor::named("f");
        let mut out = Vec::new();
        cursor.advance(payload, &mut out);
        let committed = cursor.committed();
        assert_eq!(committed.len(), 1, "a named choice is exactly one call");
        assert_eq!(committed[0].index, 0);
        assert_eq!(committed[0].name, "f");
        assert_eq!(committed[0].released, payload.len());
        assert!(!committed[0].ambiguous);
    }

    #[test]
    fn a_named_reset_stays_named() {
        // `tool_choice` belongs to the request. A reset that forgot the mode would
        // lex the next payload as a required envelope and stream nothing.
        let mut cursor = GuidedJsonCursor::named("f");
        let mut out = Vec::new();
        cursor.advance(r#"{"x":1}"#, &mut out);
        cursor.reset();
        out.clear();
        cursor.advance(r#"{"y":2}"#, &mut out);
        assert_eq!(names(&out), vec![(0, "f".to_string())]);
        assert_eq!(arguments(&out), vec![(0, r#"{"y":2}"#.to_string())]);
    }

    #[test]
    fn named_whole_input_and_every_split_agree() {
        // Multi-byte content on purpose: `advance` takes `&str`, so a split can only
        // land on a char boundary, and the released fragments must never cut one.
        let payload = r#"{"city":"東京","emoji":"😀","note":"a\"b"}"#;
        let whole = {
            let mut cursor = GuidedJsonCursor::named("get_weather");
            let mut out = Vec::new();
            cursor.advance(payload, &mut out);
            out
        };
        assert_eq!(arguments(&whole), vec![(0, payload.to_string())]);

        let per_char = stream_named("get_weather", payload);
        assert_eq!(names(&per_char), names(&whole));
        assert_eq!(arguments(&per_char), arguments(&whole));

        for split in 1..payload.len() {
            if !payload.is_char_boundary(split) {
                continue;
            }
            let mut cursor = GuidedJsonCursor::named("get_weather");
            let mut out = Vec::new();
            cursor.advance(&payload[..split], &mut out);
            cursor.advance(payload, &mut out);
            assert_eq!(names(&out), names(&whole), "names differ at split {split}");
            assert_eq!(
                arguments(&out),
                arguments(&whole),
                "arguments differ at split {split}"
            );
            for delta in &out {
                assert!(
                    std::str::from_utf8(delta.arguments.as_bytes()).is_ok(),
                    "split {split} released a fragment that is not valid UTF-8"
                );
            }
        }
    }

    #[test]
    fn a_truncated_named_payload_releases_only_what_arrived() {
        let payload = r#"{"city":"Par"#;
        let deltas = stream_named("get_weather", payload);
        assert_eq!(names(&deltas), vec![(0, "get_weather".to_string())]);
        assert_eq!(arguments(&deltas), vec![(0, payload.to_string())]);
    }

    #[test]
    fn the_required_mode_still_needs_a_name_from_the_payload() {
        // The named seed must not leak into the required mode: a nameless required
        // element still may not commit.
        assert!(stream(r#"[{"arguments":{"a":1}}]"#).is_empty());
    }
}
