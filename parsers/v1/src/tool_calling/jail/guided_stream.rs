// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Incremental lexer over a guided (`tool_choice`) tool-call payload.
//!
//! # Why a lexer and not a few `str::find` calls
//!
//! Streaming a guided call before its payload has closed needs four facts that
//! only a JSON lexer can answer: which array element the scan is inside, that
//! element's decoded `name`, where its argument OBJECT begins, and which bytes of
//! that object are safe to put on the wire. Substring search answers none of them
//! correctly — `"name"` occurs as a VALUE in `{"x":"name","parameters":{}}`,
//! `_` is six characters that must decode to one, the `parameters` spelling
//! is as legal as `arguments`, and rescanning the whole accumulated buffer on
//! every chunk makes a token-at-a-time stream quadratic.
//!
//! [`GuidedStreamCursor`] owns that state in one place and advances strictly
//! FORWARD: every call consumes only the bytes appended since the last one, the
//! same "scan only what is new" discipline as `JsonCompletionProgress` in this
//! module's parent.
//!
//! # What it deliberately does not do
//!
//! It does not validate. The jail's buffered parse remains the only judge of
//! whether a payload is a legal call set. The cursor answers one narrower
//! question: *has enough arrived that a name and an argument OBJECT can no longer
//! turn into something else?* Requiring the argument value to open with `{` is
//! what makes an early emission safe — `null`, a string, a number and an array
//! never reach the commit point, so a shape the buffered path would void is never
//! put on the wire.
//!
//! # The one thing it cannot promise
//!
//! An emitted byte cannot be retracted. An element that names BOTH `arguments`
//! and `parameters` is ambiguous, and the cursor refuses to commit it whenever
//! the second spelling appears before the commit point (which is every ordering
//! except one: the alias that opens the object arriving first AND the name having
//! already closed). In that one ordering the call is already on the wire; the
//! cursor then blocks the element — no further argument bytes, no second call —
//! rather than pretending it can unsay them.

use std::collections::BTreeMap;

use super::ToolChoiceFormat;

/// The two spellings a guided call may use for its argument object. A call
/// carrying both is ambiguous — see the module header.
const ARGUMENT_ALIASES: [&str; 2] = ["arguments", "parameters"];

/// One fragment the cursor is willing to stand behind.
///
/// `name` is carried exactly once per call, on the delta that commits it; every
/// later delta for that call carries argument bytes and `name: None`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct GuidedDelta {
    pub tool_index: usize,
    pub name: Option<String>,
    pub arguments: String,
}

/// One call whose opening delta has already reached the client.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct StreamedCall {
    /// Argument bytes released so far, in their original wire representation.
    pub arguments: String,
}

/// What the payload is shaped like, derived from the jail's [`ToolChoiceFormat`].
#[derive(Debug, Clone, PartialEq, Eq)]
enum Mode {
    /// `tool_choice=required`: `[{"name":…,"arguments":{…}}, …]`.
    Array,
    /// `tool_choice=<named>`: the payload IS the bare argument object; the name
    /// is known up front and never appears in the bytes.
    Single { tool_name: String },
}

/// Where the scan sits inside a call object (array mode only).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Slot {
    /// Between `{` or `,` and the next key.
    Key,
    /// A key literal closed; waiting for its `:`.
    Colon,
    /// Past the `:`; the next token is that key's value.
    Value,
}

/// Per-element scan state. Reset at every call boundary.
#[derive(Debug, Default, Clone)]
struct Element {
    name: Option<String>,
    /// Offset of the `{` that opened the argument object.
    args_start: Option<usize>,
    /// Offset just past the argument object's closing `}`, once it arrived.
    args_end: Option<usize>,
    /// The scan is inside the argument object right now.
    in_args: bool,
    /// An argument alias key has been seen for this element.
    alias_seen: bool,
    /// This element may never commit, and may never release another byte:
    /// either an alias bound a non-object value, or both spellings appeared.
    blocked: bool,
    committed: bool,
    /// Argument bytes already released, relative to `args_start`.
    released: usize,
}

/// Forward-only lexer over one guided payload.
#[derive(Debug, Clone)]
pub(crate) struct GuidedStreamCursor {
    mode: Mode,
    /// Bytes already lexed. The scan never revisits them.
    scanned: usize,
    depth: i32,
    in_string: bool,
    escape: bool,
    /// Opening quote of a literal the cursor needs (a call key, or the `name`
    /// value). Decoding happens once, when the closing quote arrives, so JSON
    /// escape semantics stay owned by `serde_json`.
    literal_start: Option<usize>,
    /// Depth at which call keys sit: 2 under an array root, 1 under an object
    /// root. Array mode only.
    key_depth: i32,
    root_seen: bool,
    /// The payload does not open as a call shape; nothing may ever commit.
    disabled: bool,
    slot: Slot,
    pending_key: Option<String>,
    element: Element,
    index: usize,
    /// Calls already put on the wire, keyed in the cursor's source-index space.
    streamed: BTreeMap<usize, StreamedCall>,
}

impl GuidedStreamCursor {
    pub(crate) fn new(format: &ToolChoiceFormat) -> Self {
        let mode = match format {
            ToolChoiceFormat::ArrayOfTools => Mode::Array,
            ToolChoiceFormat::SingleObject { tool_name } => Mode::Single {
                tool_name: tool_name.clone(),
            },
        };
        Self::with_mode(mode)
    }

    fn with_mode(mode: Mode) -> Self {
        Self {
            mode,
            scanned: 0,
            depth: 0,
            in_string: false,
            escape: false,
            literal_start: None,
            key_depth: 1,
            root_seen: false,
            disabled: false,
            slot: Slot::Key,
            pending_key: None,
            element: Element::default(),
            index: 0,
            streamed: BTreeMap::new(),
        }
    }

    pub(crate) fn reset(&mut self) {
        let mode = self.mode.clone();
        *self = Self::with_mode(mode);
    }

    pub(crate) fn streamed(&self) -> &BTreeMap<usize, StreamedCall> {
        &self.streamed
    }

    /// Lex the bytes appended since the last advance and emit whatever became
    /// safe to send.
    ///
    /// `payload` is the WHOLE accumulated payload, not just the new chunk: the
    /// cursor slices it from its own `scanned` offset, so the caller never has to
    /// track which bytes it already handed over. The buffer must only ever grow —
    /// shrinking it would make every retained offset lie, so a shorter payload is
    /// treated as "nothing new" and the caller is expected to [`Self::reset`].
    pub(crate) fn advance(&mut self, payload: &str, out: &mut Vec<GuidedDelta>) {
        if self.disabled || payload.len() <= self.scanned {
            return;
        }

        let mut cut = self.scanned;
        for (relative, ch) in payload[self.scanned..].char_indices() {
            let at = self.scanned + relative;
            cut = at + ch.len_utf8();
            match self.mode {
                Mode::Array => self.step_array(payload, at, cut, ch, out),
                Mode::Single { .. } => self.step_single(at, ch),
            }
            if self.disabled {
                break;
            }
        }
        self.scanned = cut;
        self.maybe_commit(out);
        self.flush(payload, cut, out);
    }

    // ----- single-object mode -------------------------------------------------

    /// One character of a bare argument object. Everything before the first `{`
    /// is skipped: guided decoding can prefix whitespace, and only the object
    /// itself is the arguments.
    fn step_single(&mut self, at: usize, ch: char) {
        if !self.root_seen {
            if ch == '{' {
                self.root_seen = true;
                self.depth = 1;
                self.element.args_start = Some(at);
            } else if !ch.is_whitespace() {
                // Only WHITESPACE may precede a bare argument object. Anything else
                // means this payload is not one - the backend ignored the grammar and
                // emitted its native markup. Scanning on for a later `{` would stream
                // a brace from inside that markup as the arguments (MiniMax M2 emits
                // `<parameter name="location">San Francisco {CA}</parameter>` under a
                // named tool_choice). Disable instead and leave it to the jail's
                // native fallback, which is why that fallback still exists.
                self.disabled = true;
            }
            return;
        }
        if self.element.args_end.is_some() {
            // The object closed; trailing bytes are not arguments.
            return;
        }
        if self.in_string {
            self.step_in_string_raw(ch);
            return;
        }
        match ch {
            '"' => self.in_string = true,
            '{' | '[' => self.depth += 1,
            '}' | ']' => {
                self.depth -= 1;
                if self.depth == 0 {
                    self.element.args_end = Some(at + ch.len_utf8());
                }
            }
            _ => {}
        }
    }

    // ----- array mode ---------------------------------------------------------

    /// One character of the array lex. `cut` is the offset just past `ch`, so a
    /// flush triggered mid-advance still sees the bytes this character added.
    fn step_array(
        &mut self,
        payload: &str,
        at: usize,
        cut: usize,
        ch: char,
        out: &mut Vec<GuidedDelta>,
    ) {
        if self.in_string {
            self.step_in_string(payload, at, ch);
            return;
        }

        match ch {
            '"' => {
                if !self.root_seen {
                    // A guided `required` payload opens with `[` or `{`; a bare
                    // string is not a call shape.
                    self.disabled = true;
                    return;
                }
                self.in_string = true;
                // Capture only the literals that matter: keys at call-key depth,
                // and the `name` value. Capturing every string would allocate
                // once per argument value for nothing.
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
                let opens_args = ch == '{'
                    && self.depth == self.key_depth
                    && self.slot == Slot::Value
                    && is_alias(self.pending_key.as_deref());
                let opens_element = ch == '{' && self.depth == self.key_depth - 1;
                let name_is_not_a_string = self.depth == self.key_depth
                    && self.slot == Slot::Value
                    && self.pending_key.as_deref() == Some("name");
                self.depth += 1;
                if opens_args {
                    if self.element.args_start.is_some() {
                        self.element.blocked = true;
                    } else {
                        self.element.args_start = Some(at);
                        self.element.in_args = true;
                    }
                } else if opens_element {
                    self.slot = Slot::Key;
                    self.pending_key = None;
                } else if name_is_not_a_string {
                    self.element.blocked = true;
                }
                self.maybe_commit(out);
            }
            '}' | ']' => {
                let closes_args =
                    ch == '}' && self.element.in_args && self.depth == self.key_depth + 1;
                let closes_element = ch == '}' && self.depth == self.key_depth;
                self.depth -= 1;
                if closes_args {
                    self.element.args_end = Some(at + 1);
                    self.element.in_args = false;
                    self.maybe_commit(out);
                    self.flush(payload, cut, out);
                } else if closes_element {
                    // A `name` that closed AFTER its argument object only becomes
                    // committable here, so commit before the final flush or the
                    // element's bytes leave with it.
                    self.maybe_commit(out);
                    self.flush(payload, cut, out);
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
                if self.depth == self.key_depth
                    && self.slot == Slot::Value
                    && (is_alias(self.pending_key.as_deref())
                        || self.pending_key.as_deref() == Some("name"))
                {
                    // A bare literal (`null`, a number, `true`) bound to an
                    // argument alias or to `name`: neither can become an object,
                    // so this element stays off the wire.
                    self.element.blocked = true;
                }
            }
        }
    }

    /// One character inside a string literal the cursor may want to decode.
    fn step_in_string(&mut self, payload: &str, at: usize, ch: char) {
        if self.escape {
            self.escape = false;
            return;
        }
        match ch {
            '\\' => self.escape = true,
            '"' => {
                self.in_string = false;
                self.close_literal(payload, at);
            }
            _ => {}
        }
    }

    /// One character inside a string literal whose content is irrelevant.
    fn step_in_string_raw(&mut self, ch: char) {
        if self.escape {
            self.escape = false;
            return;
        }
        match ch {
            '\\' => self.escape = true,
            '"' => self.in_string = false,
            _ => {}
        }
    }

    /// A captured literal closed: it is this element's key, or its name.
    fn close_literal(&mut self, payload: &str, end: usize) {
        let Some(start) = self.literal_start.take() else {
            return;
        };
        // `serde_json` owns escape semantics, including `\uXXXX` and surrogate
        // pairs. Hand-rolling that decode is how `_` became `u005f`.
        let Ok(literal) = serde_json::from_str::<String>(&payload[start..=end]) else {
            return;
        };
        match self.slot {
            Slot::Key => {
                if is_alias(Some(literal.as_str())) {
                    if self.element.alias_seen {
                        // Both spellings in one element: ambiguous.
                        self.element.blocked = true;
                    }
                    self.element.alias_seen = true;
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

    // ----- emission -----------------------------------------------------------

    /// Put the current call on the wire once its shape can no longer change.
    ///
    /// Both facts are required: the name is known, AND the argument value has
    /// been seen to open with `{`. Committing on the name alone is what would let
    /// `"arguments": null` reach the client as a call.
    fn maybe_commit(&mut self, out: &mut Vec<GuidedDelta>) {
        if self.element.committed || self.element.blocked || self.element.args_start.is_none() {
            return;
        }
        let name = match &self.mode {
            Mode::Single { tool_name } => tool_name.clone(),
            Mode::Array => match &self.element.name {
                Some(name) => name.clone(),
                None => return,
            },
        };
        self.element.committed = true;
        self.streamed.insert(
            self.index,
            StreamedCall {
                arguments: String::new(),
            },
        );
        out.push(GuidedDelta {
            tool_index: self.index,
            name: Some(name),
            arguments: String::new(),
        });
    }

    /// Release argument bytes accumulated since the last fragment, RAW and
    /// byte-exact — the client reassembles the source object, so re-encoding
    /// here would change bytes the buffered path would have emitted verbatim.
    fn flush(&mut self, payload: &str, cut: usize, out: &mut Vec<GuidedDelta>) {
        if !self.element.committed || self.element.blocked {
            return;
        }
        let Some(start) = self.element.args_start else {
            return;
        };
        // Never past the argument object's own `}`: the bytes after it belong to
        // the call envelope, not to the arguments.
        let bound = self.element.args_end.unwrap_or(cut).min(cut);
        let from = start + self.element.released;
        if bound <= from {
            return;
        }
        let fragment = payload[from..bound].to_string();
        self.element.released = bound - start;
        // `maybe_commit` is the only writer and inserts at this same index, so a miss
        // is impossible today. Assert it in tests rather than panicking a live stream:
        // a future commit site that breaks the invariant should cost one fragment, not
        // the request.
        let Some(record) = self.streamed.get_mut(&self.index) else {
            debug_assert!(
                false,
                "a committed element must have a streamed-call record at index {}",
                self.index
            );
            tracing::error!(
                tool_index = self.index,
                "guided streaming committed an element with no streamed-call record; \
                 dropping the fragment"
            );
            return;
        };
        record.arguments.push_str(&fragment);
        out.push(GuidedDelta {
            tool_index: self.index,
            name: None,
            arguments: fragment,
        });
    }

    /// The current call object closed; move to the next element.
    fn finish_element(&mut self) {
        // ALWAYS advance, including for an element that never committed: the
        // index is the element's position in the array, and skipping one here
        // would slide every later call onto the wrong index.
        self.index += 1;
        self.element = Element::default();
        self.slot = Slot::Key;
        self.pending_key = None;
    }
}

fn is_alias(key: Option<&str>) -> bool {
    key.is_some_and(|key| ARGUMENT_ALIASES.contains(&key))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn array_cursor() -> GuidedStreamCursor {
        GuidedStreamCursor::new(&ToolChoiceFormat::ArrayOfTools)
    }

    fn named_cursor(tool_name: &str) -> GuidedStreamCursor {
        GuidedStreamCursor::new(&ToolChoiceFormat::SingleObject {
            tool_name: tool_name.to_string(),
        })
    }

    /// Drive a payload one character at a time.
    fn stream(mut cursor: GuidedStreamCursor, payload: &str) -> Vec<GuidedDelta> {
        let mut out = Vec::new();
        let mut seen = String::new();
        for ch in payload.chars() {
            seen.push(ch);
            cursor.advance(&seen, &mut out);
        }
        out
    }

    /// Feed the whole payload in one call.
    fn whole(mut cursor: GuidedStreamCursor, payload: &str) -> Vec<GuidedDelta> {
        let mut out = Vec::new();
        cursor.advance(payload, &mut out);
        out
    }

    /// Names carried, in emission order.
    fn names(deltas: &[GuidedDelta]) -> Vec<(usize, String)> {
        deltas
            .iter()
            .filter_map(|d| d.name.clone().map(|n| (d.tool_index, n)))
            .collect()
    }

    /// Argument bytes reassembled per tool index, in first-seen order.
    fn arguments(deltas: &[GuidedDelta]) -> Vec<(usize, String)> {
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

    // ----- required (array) mode ---------------------------------------------

    #[test]
    fn required_splits_the_name_from_its_arguments() {
        let payload = r#"[{"name":"get_weather","arguments":{"city":"Paris","unit":"c"}}]"#;
        let expected = r#"{"city":"Paris","unit":"c"}"#;
        let deltas = stream(array_cursor(), payload);

        assert_eq!(names(&deltas), vec![(0, "get_weather".to_string())]);
        // The name-carrying delta is the commit, and it carries no bytes.
        let first = deltas.first().expect("a commit delta");
        assert_eq!(first.name.as_deref(), Some("get_weather"));
        assert!(first.arguments.is_empty());
        assert_eq!(arguments(&deltas), vec![(0, expected.to_string())]);
        // Byte-for-byte against the source slice, not against a re-encode.
        let start = payload.find(expected).expect("the argument object");
        assert_eq!(
            arguments(&deltas)[0].1,
            payload[start..start + expected.len()]
        );
    }

    #[test]
    fn required_accepts_the_parameters_spelling() {
        let deltas = stream(
            array_cursor(),
            r#"[{"name":"get_weather","parameters":{"city":"Tokyo"}}]"#,
        );
        assert_eq!(names(&deltas), vec![(0, "get_weather".to_string())]);
        assert_eq!(
            arguments(&deltas),
            vec![(0, r#"{"city":"Tokyo"}"#.to_string())]
        );
    }

    #[test]
    fn two_calls_in_one_array_get_distinct_indices() {
        let payload = r#"[{"name":"a","arguments":{"x":1}},{"name":"b","parameters":{"y":[1,2]}}]"#;
        let deltas = stream(array_cursor(), payload);
        assert_eq!(
            names(&deltas),
            vec![(0, "a".to_string()), (1, "b".to_string())]
        );
        assert_eq!(
            arguments(&deltas),
            vec![
                (0, r#"{"x":1}"#.to_string()),
                (1, r#"{"y":[1,2]}"#.to_string())
            ]
        );
        // The whole-payload path must agree with the character-at-a-time path.
        let at_once = whole(array_cursor(), payload);
        assert_eq!(names(&at_once), names(&deltas));
        assert_eq!(arguments(&at_once), arguments(&deltas));
    }

    #[test]
    fn a_non_object_argument_value_is_never_committed() {
        for payload in [
            r#"[{"name":"f","arguments":"just a string"}]"#,
            r#"[{"name":"f","arguments":null}]"#,
            r#"[{"name":"f","arguments":7}]"#,
            r#"[{"name":"f","arguments":[1,2]}]"#,
            r#"[{"name":"f","parameters":null}]"#,
            // No argument key at all: valid, but there is no object to open on.
            r#"[{"name":"f"}]"#,
        ] {
            assert!(
                stream(array_cursor(), payload).is_empty(),
                "{payload} put a call on the wire that has no argument object"
            );
            assert!(whole(array_cursor(), payload).is_empty(), "{payload}");
        }
    }

    #[test]
    fn both_argument_aliases_in_one_element_do_not_commit() {
        // Ambiguity visible before the commit point: nothing is ever emitted.
        for payload in [
            r#"[{"parameters":{"b":2},"arguments":{"a":1},"name":"f"}]"#,
            r#"[{"name":"f","parameters":null,"arguments":{"a":1}}]"#,
            r#"[{"name":"f","arguments":[1,2],"parameters":{"b":2}}]"#,
        ] {
            assert!(
                stream(array_cursor(), payload).is_empty(),
                "{payload} committed an ambiguous element"
            );
            assert!(whole(array_cursor(), payload).is_empty(), "{payload}");
        }

        // The orderings where the FIRST alias completes a name-plus-object pair
        // before the second key exists: that call is already on the wire and a
        // fragment cannot be unsaid. The cursor must then release NOTHING
        // further — no byte of the second object, and never a second call.
        for (payload, first_object) in [
            (
                r#"[{"name":"f","arguments":{"a":1},"parameters":{"b":2}}]"#,
                r#"{"a":1}"#,
            ),
            (
                r#"[{"parameters":{"b":2},"name":"f","arguments":{"a":1}}]"#,
                r#"{"b":2}"#,
            ),
        ] {
            let deltas = stream(array_cursor(), payload);
            assert_eq!(names(&deltas), vec![(0, "f".to_string())], "{payload}");
            assert_eq!(
                arguments(&deltas),
                vec![(0, first_object.to_string())],
                "{payload} released bytes from the ambiguous second object"
            );
        }
    }

    #[test]
    fn the_word_name_as_a_value_is_not_the_call_name() {
        // Substring search matched this VALUE and then read the next string as
        // the function name, emitting a call named `arguments`.
        for payload in [
            r#"[{"x":"name","parameters":{}}]"#,
            r#"[{"x":"name","arguments":{"city":"Paris"}}]"#,
            r#"[{"arguments":{"name":"not the call name"}}]"#,
        ] {
            let deltas = stream(array_cursor(), payload);
            assert!(
                deltas.is_empty(),
                "{payload} invented a name: {deltas:?} — a nameless element must not stream"
            );
        }
    }

    #[test]
    fn escapes_in_the_name_decode() {
        // NOT a raw string literal: the payload carries the six characters
        // `_`, which JSON decodes to `_`. A hand-rolled decoder produced
        // `getu005fweather`.
        let payload = "[{\"name\":\"get\\u005fweather\",\"arguments\":{}}]";
        assert!(
            payload.contains("\\u005f"),
            "the escape must reach the lexer"
        );
        assert_eq!(
            names(&stream(array_cursor(), payload)),
            vec![(0, "get_weather".to_string())]
        );

        // Escaped quotes and backslashes inside the name.
        let payload = r#"[{"name":"a\"b\\c\nd","arguments":{}}]"#;
        assert_eq!(
            names(&stream(array_cursor(), payload)),
            vec![(0, "a\"b\\c\nd".to_string())]
        );

        // A surrogate pair, as an encoder outside the BMP actually emits it.
        let payload = "[{\"name\":\"a\\ud83d\\ude00b\",\"arguments\":{}}]";
        assert!(
            payload.contains("\\ud83d"),
            "the escape must reach the lexer"
        );
        assert_eq!(
            names(&stream(array_cursor(), payload)),
            vec![(0, "a\u{1F600}b".to_string())]
        );
    }

    #[test]
    fn a_brace_inside_an_argument_string_does_not_close_the_object() {
        let payload = r#"[{"name":"f","arguments":{"s":"}}]","t":"\""}}]"#;
        let deltas = stream(array_cursor(), payload);
        assert_eq!(
            arguments(&deltas),
            vec![(0, r#"{"s":"}}]","t":"\""}"#.to_string())]
        );
    }

    #[test]
    fn a_name_that_closes_after_its_arguments_still_commits() {
        let payload = r#"[{"arguments":{"city":"Paris"},"name":"get_weather"}]"#;
        let deltas = stream(array_cursor(), payload);
        assert_eq!(names(&deltas), vec![(0, "get_weather".to_string())]);
        assert_eq!(
            arguments(&deltas),
            vec![(0, r#"{"city":"Paris"}"#.to_string())]
        );
    }

    #[test]
    fn a_payload_that_is_not_a_call_shape_emits_nothing() {
        assert!(stream(array_cursor(), r#""just a string""#).is_empty());
        assert!(stream(array_cursor(), "42").is_empty());
    }

    #[test]
    fn reset_returns_the_cursor_to_a_fresh_stream() {
        let mut cursor = array_cursor();
        let mut out = Vec::new();
        cursor.advance(r#"[{"name":"a","arguments":{"x":1}}]"#, &mut out);
        assert_eq!(names(&out), vec![(0, "a".to_string())]);

        cursor.reset();
        let mut second = Vec::new();
        cursor.advance(r#"[{"name":"b","arguments":{"y":2}}]"#, &mut second);
        assert_eq!(names(&second), vec![(0, "b".to_string())]);
        assert_eq!(arguments(&second), vec![(0, r#"{"y":2}"#.to_string())]);
    }

    // ----- named (single-object) mode ----------------------------------------

    #[test]
    fn named_carries_the_name_on_the_first_delta_only() {
        let payload = r#"{"city":"Paris","unit":"c"}"#;
        let deltas = stream(named_cursor("get_weather"), payload);

        let first = deltas.first().expect("a commit delta");
        assert_eq!(first.name.as_deref(), Some("get_weather"));
        assert!(first.arguments.is_empty());
        assert!(
            deltas[1..].iter().all(|d| d.name.is_none()),
            "the name rode more than the first delta: {deltas:?}"
        );
        assert_eq!(arguments(&deltas), vec![(0, payload.to_string())]);
    }

    #[test]
    fn named_skips_whitespace_before_the_object() {
        let payload = "  \n\t{\"city\":\"Paris\"}";
        let deltas = stream(named_cursor("get_weather"), payload);
        assert_eq!(names(&deltas), vec![(0, "get_weather".to_string())]);
        assert_eq!(
            arguments(&deltas),
            vec![(0, r#"{"city":"Paris"}"#.to_string())]
        );
    }

    #[test]
    fn named_never_releases_past_the_closing_brace() {
        // Trailing bytes after the object are not arguments.
        let payload = "{\"a\":1}\n\ntrailing";
        let deltas = stream(named_cursor("f"), payload);
        assert_eq!(arguments(&deltas), vec![(0, r#"{"a":1}"#.to_string())]);
    }

    #[test]
    fn named_handles_braces_inside_strings() {
        let payload = r#"{"s":"}{ \" }","n":{"deep":[1,{"x":"}"}]}}"#;
        let deltas = stream(named_cursor("f"), payload);
        assert_eq!(arguments(&deltas), vec![(0, payload.to_string())]);
    }

    // ----- split-boundary sweeps (both modes) --------------------------------

    /// Feed `payload` as `[..split]` then the whole thing, for EVERY valid UTF-8
    /// split, and assert the reassembly and the fragment boundaries.
    fn assert_every_split(build: impl Fn() -> GuidedStreamCursor, payload: &str, expected: &str) {
        for split in 0..=payload.len() {
            if !payload.is_char_boundary(split) {
                continue;
            }
            let mut cursor = build();
            let mut out = Vec::new();
            cursor.advance(&payload[..split], &mut out);
            cursor.advance(payload, &mut out);

            let joined = arguments(&out);
            assert_eq!(
                joined,
                vec![(0, expected.to_string())],
                "reassembly differs at split {split}"
            );

            // No fragment may start or end mid-character.
            let mut offset = 0usize;
            for delta in out.iter().filter(|d| !d.arguments.is_empty()) {
                assert!(
                    expected.is_char_boundary(offset),
                    "fragment starts mid-character at {offset} (split {split})"
                );
                offset += delta.arguments.len();
                assert!(
                    expected.is_char_boundary(offset),
                    "fragment ends mid-character at {offset} (split {split})"
                );
            }
            assert_eq!(offset, expected.len());
        }
    }

    #[test]
    fn required_survives_every_char_boundary_split() {
        let payload = r#"[{"name":"f","arguments":{"city":"東京","emoji":"😀","q":"a\"b"}}]"#;
        let expected = r#"{"city":"東京","emoji":"😀","q":"a\"b"}"#;
        assert!(
            payload.chars().any(|c| c.len_utf8() > 1),
            "the sweep needs a multi-byte character"
        );
        assert_every_split(array_cursor, payload, expected);
    }

    #[test]
    fn named_survives_every_char_boundary_split() {
        let payload = r#"{"city":"東京","emoji":"😀","q":"a\"b"}"#;
        assert!(payload.chars().any(|c| c.len_utf8() > 1));
        assert_every_split(|| named_cursor("get_weather"), payload, payload);
    }

    // ----- long payloads (both modes) ----------------------------------------

    /// A >2000-byte argument object with nesting, escaped quotes and a literal
    /// `}` inside a string.
    fn long_arguments() -> String {
        let mut body = String::from(r#"{"note":"a \" quote and a } brace","items":["#);
        for i in 0..120 {
            if i > 0 {
                body.push(',');
            }
            body.push_str(&format!(
                r#"{{"k{i}":"v{i} }} \" x","nested":{{"deep":[{i},"文字"]}}}}"#
            ));
        }
        body.push_str(r#"],"last":"x"}"#);
        assert!(body.len() > 2000, "body was only {} bytes", body.len());
        body
    }

    /// Feed a payload in `chunk`-character steps.
    fn stream_chunks(
        mut cursor: GuidedStreamCursor,
        payload: &str,
        chunk: usize,
    ) -> Vec<GuidedDelta> {
        let mut out = Vec::new();
        let mut seen = String::new();
        for (n, ch) in payload.chars().enumerate() {
            seen.push(ch);
            if (n + 1) % chunk == 0 {
                cursor.advance(&seen, &mut out);
            }
        }
        cursor.advance(payload, &mut out);
        out
    }

    #[test]
    fn required_streams_a_long_payload_in_many_fragments() {
        let expected = long_arguments();
        let payload = format!(r#"[{{"name":"f","arguments":{expected}}}]"#);
        let deltas = stream_chunks(array_cursor(), &payload, 7);

        assert_eq!(names(&deltas), vec![(0, "f".to_string())]);
        assert_eq!(arguments(&deltas), vec![(0, expected.clone())]);
        let fragments = deltas.iter().filter(|d| !d.arguments.is_empty()).count();
        assert!(
            fragments > 100,
            "arguments arrived in {fragments} fragment(s), not a stream"
        );
        // And the whole-payload path produces the same bytes in one go.
        assert_eq!(
            arguments(&whole(array_cursor(), &payload)),
            vec![(0, expected)]
        );
    }

    #[test]
    fn named_native_markup_with_an_inner_brace_never_commits() {
        // A brace inside native markup is not the start of an argument object.
        let payload = "<minimax:tool_call><invoke name=\"get_weather\">\
<parameter name=\"location\">San Francisco {CA}</parameter></invoke></minimax:tool_call>";
        let mut cursor = GuidedStreamCursor::new(&ToolChoiceFormat::SingleObject {
            tool_name: "get_weather".to_string(),
        });
        let mut out = Vec::new();
        cursor.advance(payload, &mut out);
        assert!(
            out.is_empty(),
            "native markup must not be streamed as arguments, got {out:?}"
        );
    }

    #[test]
    fn named_streams_a_long_payload_in_many_fragments() {
        let payload = long_arguments();
        let deltas = stream_chunks(named_cursor("f"), &payload, 7);

        assert_eq!(names(&deltas), vec![(0, "f".to_string())]);
        assert_eq!(arguments(&deltas), vec![(0, payload.clone())]);
        let fragments = deltas.iter().filter(|d| !d.arguments.is_empty()).count();
        assert!(
            fragments > 100,
            "arguments arrived in {fragments} fragment(s), not a stream"
        );
    }
}
