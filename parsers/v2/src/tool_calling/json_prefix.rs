// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#[derive(Debug, Default)]
pub(crate) struct JsonPrefixState {
    frames: Vec<JsonFrame>,
    string: Option<JsonStringKind>,
    string_escaped: bool,
    unicode_digits: u8,
    primitive: Option<JsonPrimitive>,
    pub(crate) invalid: bool,
    pub(crate) complete: bool,
}

#[derive(Debug)]
enum JsonFrame {
    Object(JsonObjectState),
    Array(JsonArrayState),
}

#[derive(Debug)]
enum JsonObjectState {
    KeyOrEnd,
    Key,
    Colon,
    Value,
    CommaOrEnd,
}

#[derive(Debug)]
enum JsonArrayState {
    ValueOrEnd,
    Value,
    CommaOrEnd,
}

#[derive(Debug, Clone, Copy)]
enum JsonStringKind {
    Key,
    Value,
}

#[derive(Debug)]
enum JsonPrimitive {
    Literal {
        expected: &'static [u8],
        next: usize,
    },
    Number(JsonNumberState),
}

#[derive(Debug)]
enum JsonNumberState {
    Sign,
    IntegerZero,
    Integer,
    FractionStart,
    Fraction,
    ExponentStart,
    ExponentSign,
    Exponent,
}

fn is_json_whitespace(ch: char) -> bool {
    matches!(ch, ' ' | '\t' | '\n' | '\r')
}

impl JsonPrefixState {
    pub(crate) fn new(opener: char) -> Self {
        let mut state = Self::default();
        state.start_container(opener);
        state
    }

    pub(crate) fn in_string(&self) -> bool {
        self.string.is_some()
    }

    fn start_container(&mut self, opener: char) {
        match opener {
            '{' => self
                .frames
                .push(JsonFrame::Object(JsonObjectState::KeyOrEnd)),
            '[' => self
                .frames
                .push(JsonFrame::Array(JsonArrayState::ValueOrEnd)),
            _ => self.invalid = true,
        }
    }

    pub(crate) fn consume(&mut self, ch: char) {
        if self.invalid {
            return;
        }
        if self.complete {
            if !is_json_whitespace(ch) {
                self.invalid = true;
            }
            return;
        }
        if self.string.is_some() {
            self.consume_string(ch);
            return;
        }
        if self.primitive.is_some() {
            self.consume_primitive(ch);
            return;
        }

        let Some(frame) = self.frames.last() else {
            self.invalid = true;
            return;
        };
        match frame {
            JsonFrame::Object(state) => match *state {
                JsonObjectState::KeyOrEnd => {
                    if ch == '}' {
                        self.close_container();
                    } else if ch == '"' {
                        self.string = Some(JsonStringKind::Key);
                    } else if !is_json_whitespace(ch) {
                        self.invalid = true;
                    }
                }
                JsonObjectState::Key => {
                    if ch == '"' {
                        self.string = Some(JsonStringKind::Key);
                    } else if !is_json_whitespace(ch) {
                        self.invalid = true;
                    }
                }
                JsonObjectState::Colon => {
                    if ch == ':' {
                        self.set_object_state(JsonObjectState::Value);
                    } else if !is_json_whitespace(ch) {
                        self.invalid = true;
                    }
                }
                JsonObjectState::Value => self.start_value(ch),
                JsonObjectState::CommaOrEnd => {
                    if ch == ',' {
                        self.set_object_state(JsonObjectState::Key);
                    } else if ch == '}' {
                        self.close_container();
                    } else if !is_json_whitespace(ch) {
                        self.invalid = true;
                    }
                }
            },
            JsonFrame::Array(state) => match *state {
                JsonArrayState::ValueOrEnd => {
                    if ch == ']' {
                        self.close_container();
                    } else {
                        self.start_value(ch);
                    }
                }
                JsonArrayState::Value => self.start_value(ch),
                JsonArrayState::CommaOrEnd => {
                    if ch == ',' {
                        self.set_array_state(JsonArrayState::Value);
                    } else if ch == ']' {
                        self.close_container();
                    } else if !is_json_whitespace(ch) {
                        self.invalid = true;
                    }
                }
            },
        }
    }

    fn set_object_state(&mut self, state: JsonObjectState) {
        if let Some(JsonFrame::Object(current)) = self.frames.last_mut() {
            *current = state;
        } else {
            self.invalid = true;
        }
    }

    fn set_array_state(&mut self, state: JsonArrayState) {
        if let Some(JsonFrame::Array(current)) = self.frames.last_mut() {
            *current = state;
        } else {
            self.invalid = true;
        }
    }

    fn consume_string(&mut self, ch: char) {
        if self.unicode_digits > 0 {
            if ch.is_ascii_hexdigit() {
                self.unicode_digits -= 1;
            } else {
                self.invalid = true;
            }
            return;
        }
        if self.string_escaped {
            self.string_escaped = false;
            if ch == 'u' {
                self.unicode_digits = 4;
            } else if !matches!(ch, '"' | '\\' | '/' | 'b' | 'f' | 'n' | 'r' | 't') {
                self.invalid = true;
            }
            return;
        }
        if ch == '\\' {
            self.string_escaped = true;
        } else if ch == '"' {
            let Some(kind) = self.string.take() else {
                self.invalid = true;
                return;
            };
            match kind {
                JsonStringKind::Key => {
                    self.set_object_state(JsonObjectState::Colon);
                }
                JsonStringKind::Value => self.value_complete(),
            }
        } else if ch <= '\u{1f}' {
            self.invalid = true;
        }
    }

    fn consume_primitive(&mut self, ch: char) {
        let delimiter = is_json_whitespace(ch) || matches!(ch, ',' | '}' | ']');
        let Some(primitive) = self.primitive.take() else {
            self.invalid = true;
            return;
        };
        match primitive {
            JsonPrimitive::Literal { expected, next } => {
                if next == expected.len() && delimiter {
                    self.value_complete();
                    self.consume(ch);
                } else if next < expected.len()
                    && ch.is_ascii()
                    && Some(ch as u8) == expected.get(next).copied()
                {
                    self.primitive = Some(JsonPrimitive::Literal {
                        expected,
                        next: next + 1,
                    });
                } else {
                    self.invalid = true;
                }
            }
            JsonPrimitive::Number(mut state) => {
                if delimiter
                    && matches!(
                        state,
                        JsonNumberState::IntegerZero
                            | JsonNumberState::Integer
                            | JsonNumberState::Fraction
                            | JsonNumberState::Exponent
                    )
                {
                    self.value_complete();
                    self.consume(ch);
                } else if let Some(next) = Self::next_number_state(&state, ch) {
                    state = next;
                    self.primitive = Some(JsonPrimitive::Number(state));
                } else {
                    self.invalid = true;
                }
            }
        }
    }

    fn next_number_state(state: &JsonNumberState, ch: char) -> Option<JsonNumberState> {
        match state {
            JsonNumberState::Sign if ch == '0' => Some(JsonNumberState::IntegerZero),
            JsonNumberState::Sign if ch.is_ascii_digit() => Some(JsonNumberState::Integer),
            JsonNumberState::IntegerZero if ch == '.' => Some(JsonNumberState::FractionStart),
            JsonNumberState::IntegerZero if matches!(ch, 'e' | 'E') => {
                Some(JsonNumberState::ExponentStart)
            }
            JsonNumberState::Integer if ch.is_ascii_digit() => Some(JsonNumberState::Integer),
            JsonNumberState::Integer if ch == '.' => Some(JsonNumberState::FractionStart),
            JsonNumberState::Integer if matches!(ch, 'e' | 'E') => {
                Some(JsonNumberState::ExponentStart)
            }
            JsonNumberState::FractionStart if ch.is_ascii_digit() => {
                Some(JsonNumberState::Fraction)
            }
            JsonNumberState::Fraction if ch.is_ascii_digit() => Some(JsonNumberState::Fraction),
            JsonNumberState::Fraction if matches!(ch, 'e' | 'E') => {
                Some(JsonNumberState::ExponentStart)
            }
            JsonNumberState::ExponentStart if matches!(ch, '+' | '-') => {
                Some(JsonNumberState::ExponentSign)
            }
            JsonNumberState::ExponentStart if ch.is_ascii_digit() => {
                Some(JsonNumberState::Exponent)
            }
            JsonNumberState::ExponentSign if ch.is_ascii_digit() => Some(JsonNumberState::Exponent),
            JsonNumberState::Exponent if ch.is_ascii_digit() => Some(JsonNumberState::Exponent),
            _ => None,
        }
    }

    fn start_value(&mut self, ch: char) {
        match ch {
            '{' | '[' => self.start_container(ch),
            '"' => self.string = Some(JsonStringKind::Value),
            't' => {
                self.primitive = Some(JsonPrimitive::Literal {
                    expected: b"true",
                    next: 1,
                })
            }
            'f' => {
                self.primitive = Some(JsonPrimitive::Literal {
                    expected: b"false",
                    next: 1,
                })
            }
            'n' => {
                self.primitive = Some(JsonPrimitive::Literal {
                    expected: b"null",
                    next: 1,
                })
            }
            '-' => self.primitive = Some(JsonPrimitive::Number(JsonNumberState::Sign)),
            '0' => self.primitive = Some(JsonPrimitive::Number(JsonNumberState::IntegerZero)),
            ch if ch.is_ascii_digit() => {
                self.primitive = Some(JsonPrimitive::Number(JsonNumberState::Integer))
            }
            _ if is_json_whitespace(ch) => {}
            _ => self.invalid = true,
        }
    }

    fn value_complete(&mut self) {
        let Some(frame) = self.frames.last_mut() else {
            self.complete = true;
            return;
        };
        match frame {
            JsonFrame::Object(state) => *state = JsonObjectState::CommaOrEnd,
            JsonFrame::Array(state) => *state = JsonArrayState::CommaOrEnd,
        }
    }

    fn close_container(&mut self) {
        self.frames.pop();
        self.value_complete();
    }
}
