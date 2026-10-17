// SPDX-FileCopyrightText: Copyright (c) 2026 Reflection AI. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Python value semantics the Beam reference parser relies on.
//!
//! Copied from the model builder's frozen reference (`py.rs`) so Beam argument typing
//! is byte-identical to the reference: `str.strip()`, `int()`, `float()`,
//! `json.loads()` and `json.dumps(ensure_ascii=False)` edge cases decide what a client
//! receives. This is an independent copy of the one in `dynamo-parsers-v2`; the two
//! crates do not depend on each other.
//!
//! Two behaviours are deliberately not reproduced, because they would put invalid JSON
//! into `function.arguments`: `json.loads` accepting `NaN` / `Infinity` / numbers that
//! overflow to infinity, and lone surrogate escapes. Those inputs fail to decode, which
//! makes the caller keep the raw text.

#![allow(dead_code)]

use std::fmt::Write as _;

/// First code point of every run of ten Unicode decimal digits (`unicodedata.decimal`),
/// Unicode 15.0.0 as shipped with CPython 3.12.
const DECIMAL_ZEROS: [u32; 68] = [
    0x30, 0x660, 0x6F0, 0x7C0, 0x966, 0x9E6, 0xA66, 0xAE6, 0xB66, 0xBE6, 0xC66, 0xCE6, 0xD66,
    0xDE6, 0xE50, 0xED0, 0xF20, 0x1040, 0x1090, 0x17E0, 0x1810, 0x1946, 0x19D0, 0x1A80, 0x1A90,
    0x1B50, 0x1BB0, 0x1C40, 0x1C50, 0xA620, 0xA8D0, 0xA900, 0xA9D0, 0xA9F0, 0xAA50, 0xABF0, 0xFF10,
    0x104A0, 0x10D30, 0x11066, 0x110F0, 0x11136, 0x111D0, 0x112F0, 0x11450, 0x114D0, 0x11650,
    0x116C0, 0x11730, 0x118E0, 0x11950, 0x11C50, 0x11D50, 0x11DA0, 0x11F50, 0x16A60, 0x16AC0,
    0x16B50, 0x1D7CE, 0x1D7D8, 0x1D7E2, 0x1D7EC, 0x1D7F6, 0x1E140, 0x1E2F0, 0x1E4F0, 0x1E950,
    0x1FBF0,
];

/// `int()` refuses strings with more digits than this.
const MAX_STR_DIGITS: usize = 4300;

/// Nesting depth past which JSON decoding fails.
const MAX_JSON_DEPTH: usize = 900;

/// `str.isspace()` for one character.
pub(crate) fn is_space(c: char) -> bool {
    matches!(
        c,
        '\t' | '\n'
            | '\x0b'
            | '\x0c'
            | '\r'
            | '\x1c'..='\x1f'
            | ' '
            | '\u{85}'
            | '\u{a0}'
            | '\u{1680}'
            | '\u{2000}'..='\u{200a}'
            | '\u{2028}'
            | '\u{2029}'
            | '\u{202f}'
            | '\u{205f}'
            | '\u{3000}'
    )
}

/// `str.strip()` with no arguments.
pub(crate) fn strip(s: &str) -> &str {
    s.trim_matches(is_space)
}

fn decimal_value(c: char) -> Option<u32> {
    let code = c as u32;
    let index = DECIMAL_ZEROS.partition_point(|&zero| zero <= code);
    let zero = *DECIMAL_ZEROS.get(index.checked_sub(1)?)?;
    (code - zero < 10).then_some(code - zero)
}

fn transform_decimal_and_space(s: &str) -> Option<String> {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        if c.is_ascii() {
            out.push(c);
        } else if is_space(c) {
            out.push(' ');
        } else {
            out.push(char::from_digit(decimal_value(c)?, 10)?);
        }
    }
    Some(out)
}

fn is_c_space(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | b'\x0b' | b'\x0c' | b'\r')
}

fn trim_c_space(s: &str) -> &str {
    s.trim_matches(|c: char| c.is_ascii() && is_c_space(c as u8))
}

/// An exact Python integer, kept as its canonical decimal digits.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PyInt {
    negative: bool,
    digits: String,
}

impl PyInt {
    fn new(negative: bool, digits: &str) -> Self {
        let trimmed = digits.trim_start_matches('0');
        if trimmed.is_empty() {
            Self {
                negative: false,
                digits: "0".to_string(),
            }
        } else {
            Self {
                negative,
                digits: trimmed.to_string(),
            }
        }
    }

    pub(crate) fn from_i64(value: i64) -> Self {
        Self::new(value < 0, &value.unsigned_abs().to_string())
    }

    fn from_integral_f64(value: f64) -> Self {
        debug_assert!(value.is_finite() && value.fract() == 0.0);
        let bits = value.to_bits();
        let negative = bits >> 63 == 1;
        let exponent = ((bits >> 52) & 0x7ff) as i32;
        let fraction = bits & ((1u64 << 52) - 1);
        if exponent == 0 {
            return Self::new(false, "0");
        }
        let mantissa = fraction | (1u64 << 52);
        let shift = exponent - 1075;
        if shift <= 0 {
            return Self::new(negative, &(mantissa >> (-shift) as u32).to_string());
        }
        let mut limbs: Vec<u64> = vec![mantissa % 1_000_000_000, mantissa / 1_000_000_000];
        for _ in 0..shift {
            let mut carry = 0u64;
            for limb in limbs.iter_mut() {
                let doubled = *limb * 2 + carry;
                *limb = doubled % 1_000_000_000;
                carry = doubled / 1_000_000_000;
            }
            if carry > 0 {
                limbs.push(carry);
            }
        }
        let mut digits = String::new();
        let mut iter = limbs.iter().rev().skip_while(|&&l| l == 0);
        if let Some(first) = iter.next() {
            write!(digits, "{first}").unwrap();
        }
        for limb in iter {
            write!(digits, "{limb:09}").unwrap();
        }
        Self::new(negative, &digits)
    }

    pub(crate) fn is_zero(&self) -> bool {
        self.digits == "0"
    }
}

impl std::fmt::Display for PyInt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.negative {
            f.write_str("-")?;
        }
        f.write_str(&self.digits)
    }
}

/// An ordered Python `dict` with string keys. Assigning an existing key keeps its
/// position and replaces its value, as `dict.__setitem__` does.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct PyDict(Vec<(String, PyValue)>);

impl PyDict {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn insert(&mut self, key: String, value: PyValue) {
        if let Some(slot) = self.0.iter_mut().find(|(k, _)| *k == key) {
            slot.1 = value;
        } else {
            self.0.push((key, value));
        }
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = (&String, &PyValue)> {
        self.0.iter().map(|(k, v)| (k, v))
    }

    pub(crate) fn len(&self) -> usize {
        self.0.len()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

/// A value as the reference parser holds it.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum PyValue {
    None,
    Bool(bool),
    Int(PyInt),
    Float(f64),
    Str(String),
    List(Vec<PyValue>),
    Dict(PyDict),
}

impl PyValue {
    /// Python truthiness.
    pub(crate) fn is_truthy(&self) -> bool {
        match self {
            PyValue::None => false,
            PyValue::Bool(b) => *b,
            PyValue::Int(i) => !i.is_zero(),
            PyValue::Float(f) => *f != 0.0,
            PyValue::Str(s) => !s.is_empty(),
            PyValue::List(l) => !l.is_empty(),
            PyValue::Dict(d) => !d.is_empty(),
        }
    }

    pub(crate) fn from_json(value: &serde_json::Value) -> Self {
        match value {
            serde_json::Value::Null => PyValue::None,
            serde_json::Value::Bool(b) => PyValue::Bool(*b),
            serde_json::Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    PyValue::Int(PyInt::from_i64(i))
                } else if let Some(u) = n.as_u64() {
                    PyValue::Int(PyInt::new(false, &u.to_string()))
                } else {
                    PyValue::Float(n.as_f64().unwrap_or(f64::NAN))
                }
            }
            serde_json::Value::String(s) => PyValue::Str(s.clone()),
            serde_json::Value::Array(items) => {
                PyValue::List(items.iter().map(PyValue::from_json).collect())
            }
            serde_json::Value::Object(map) => {
                let mut dict = PyDict::new();
                for (k, v) in map {
                    dict.insert(k.clone(), PyValue::from_json(v));
                }
                PyValue::Dict(dict)
            }
        }
    }

    /// `json.dumps(self, ensure_ascii=False)`.
    pub(crate) fn dumps(&self) -> String {
        let mut out = String::new();
        dump_into(self, &mut out);
        out
    }
}

fn dump_into(value: &PyValue, out: &mut String) {
    match value {
        PyValue::None => out.push_str("null"),
        PyValue::Bool(true) => out.push_str("true"),
        PyValue::Bool(false) => out.push_str("false"),
        PyValue::Int(i) => write!(out, "{i}").unwrap(),
        PyValue::Float(f) => out.push_str(&float_json(*f)),
        PyValue::Str(s) => dump_str(s, out),
        PyValue::List(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                dump_into(item, out);
            }
            out.push(']');
        }
        PyValue::Dict(dict) => {
            out.push('{');
            for (i, (k, v)) in dict.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                dump_str(k, out);
                out.push_str(": ");
                dump_into(v, out);
            }
            out.push('}');
        }
    }
}

pub(crate) fn dump_str(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\x08' => out.push_str("\\b"),
            '\x0c' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => write!(out, "\\u{:04x}", c as u32).unwrap(),
            c => out.push(c),
        }
    }
    out.push('"');
}

fn float_json(f: f64) -> String {
    if f.is_nan() {
        "NaN".to_string()
    } else if f.is_infinite() {
        if f > 0.0 { "Infinity" } else { "-Infinity" }.to_string()
    } else {
        float_repr(f)
    }
}

/// `float.__repr__` for a finite float.
pub(crate) fn float_repr(f: f64) -> String {
    debug_assert!(f.is_finite());
    let sci = format!("{:e}", f.abs());
    let (mantissa, exponent) = sci.split_once('e').expect("`{:e}` has an exponent");
    let exponent: i32 = exponent.parse().expect("`{:e}` exponent is an integer");
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    let sign = if f.is_sign_negative() { "-" } else { "" };
    if f == 0.0 {
        return format!("{sign}0.0");
    }
    let decpt = exponent + 1;
    if decpt <= -4 || decpt > 16 {
        let mut out = String::from(sign);
        out.push_str(&digits[..1]);
        if digits.len() > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        let exp = decpt - 1;
        write!(out, "e{}{:02}", if exp < 0 { '-' } else { '+' }, exp.abs()).unwrap();
        out
    } else if decpt <= 0 {
        format!("{sign}0.{}{digits}", "0".repeat((-decpt) as usize))
    } else if decpt as usize >= digits.len() {
        format!(
            "{sign}{digits}{}.0",
            "0".repeat(decpt as usize - digits.len())
        )
    } else {
        let (int, frac) = digits.split_at(decpt as usize);
        format!("{sign}{int}.{frac}")
    }
}

/// `int(s)` in base 10.
pub(crate) fn parse_int(s: &str) -> Option<PyInt> {
    let ascii = transform_decimal_and_space(s)?;
    let body = trim_c_space(&ascii);
    let (negative, digits) = match body.as_bytes().first()? {
        b'-' => (true, &body[1..]),
        b'+' => (false, &body[1..]),
        _ => (false, body),
    };
    let clean = strip_digit_underscores(digits)?;
    if clean.is_empty() || clean.len() > MAX_STR_DIGITS {
        return None;
    }
    Some(PyInt::new(negative, &clean))
}

fn strip_digit_underscores(s: &str) -> Option<String> {
    let bytes = s.as_bytes();
    let mut out = String::with_capacity(s.len());
    for (i, &b) in bytes.iter().enumerate() {
        match b {
            b'0'..=b'9' => out.push(b as char),
            b'_' if i > 0
                && bytes[i - 1].is_ascii_digit()
                && bytes.get(i + 1).is_some_and(u8::is_ascii_digit) =>
            {
                continue;
            }
            _ => return None,
        }
    }
    Some(out)
}

/// `float(s)`.
pub(crate) fn parse_float(s: &str) -> Option<f64> {
    let ascii = transform_decimal_and_space(s)?;
    let body = trim_c_space(&ascii);
    let clean = if body.contains('_') {
        let bytes = body.as_bytes();
        let mut out = String::with_capacity(body.len());
        for (i, &b) in bytes.iter().enumerate() {
            if b == b'_' {
                if i == 0 || !bytes[i - 1].is_ascii_digit() {
                    return None;
                }
            } else {
                if i > 0 && bytes[i - 1] == b'_' && !b.is_ascii_digit() {
                    return None;
                }
                out.push(b as char);
            }
        }
        if bytes.last() == Some(&b'_') {
            return None;
        }
        out
    } else {
        body.to_string()
    };
    parse_float_literal(&clean)
}

fn parse_float_literal(s: &str) -> Option<f64> {
    let unsigned = s.strip_prefix(['+', '-']).unwrap_or(s);
    let lower = unsigned.to_ascii_lowercase();
    if matches!(lower.as_str(), "inf" | "infinity" | "nan") {
        let value = if lower == "nan" {
            f64::NAN
        } else {
            f64::INFINITY
        };
        return Some(if s.starts_with('-') { -value } else { value });
    }
    let bytes = unsigned.as_bytes();
    let mut i = 0;
    let int_digits = bytes.iter().take_while(|b| b.is_ascii_digit()).count();
    i += int_digits;
    let mut frac_digits = 0;
    if bytes.get(i) == Some(&b'.') {
        i += 1;
        frac_digits = bytes[i..].iter().take_while(|b| b.is_ascii_digit()).count();
        i += frac_digits;
    }
    if int_digits + frac_digits == 0 {
        return None;
    }
    if matches!(bytes.get(i), Some(b'e' | b'E')) {
        let mut j = i + 1;
        if matches!(bytes.get(j), Some(b'+' | b'-')) {
            j += 1;
        }
        let exp_digits = bytes[j..].iter().take_while(|b| b.is_ascii_digit()).count();
        if exp_digits == 0 {
            return None;
        }
        i = j + exp_digits;
    }
    if i != bytes.len() {
        return None;
    }
    s.parse::<f64>().ok()
}

/// `int(f)` for a float with no fractional part.
pub(crate) fn int_from_float(f: f64) -> PyInt {
    PyInt::from_integral_f64(f)
}

/// `json.loads(s)`, except as noted in the module docs.
pub(crate) fn json_loads(s: &str) -> Option<PyValue> {
    if s.starts_with('\u{feff}') {
        return None;
    }
    let mut reader = JsonReader {
        s,
        pos: 0,
        depth: 0,
    };
    reader.skip_ws();
    let value = reader.value()?;
    reader.skip_ws();
    (reader.pos == s.len()).then_some(value)
}

struct JsonReader<'a> {
    s: &'a str,
    pos: usize,
    depth: usize,
}

impl JsonReader<'_> {
    fn peek(&self) -> Option<u8> {
        self.s.as_bytes().get(self.pos).copied()
    }

    fn skip_ws(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.pos += 1;
        }
    }

    fn eat(&mut self, literal: &str) -> bool {
        if self.s[self.pos..].starts_with(literal) {
            self.pos += literal.len();
            true
        } else {
            false
        }
    }

    fn value(&mut self) -> Option<PyValue> {
        match self.peek()? {
            b'"' => self.string().map(PyValue::Str),
            b'{' => self.nested(Self::object),
            b'[' => self.nested(Self::array),
            b'n' if self.eat("null") => Some(PyValue::None),
            b't' if self.eat("true") => Some(PyValue::Bool(true)),
            b'f' if self.eat("false") => Some(PyValue::Bool(false)),
            _ => self.number(),
        }
    }

    fn nested(&mut self, parse: fn(&mut Self) -> Option<PyValue>) -> Option<PyValue> {
        self.depth += 1;
        if self.depth > MAX_JSON_DEPTH {
            return None;
        }
        let value = parse(self);
        self.depth -= 1;
        value
    }

    fn object(&mut self) -> Option<PyValue> {
        self.pos += 1;
        let mut dict = PyDict::new();
        self.skip_ws();
        if self.peek()? == b'}' {
            self.pos += 1;
            return Some(PyValue::Dict(dict));
        }
        loop {
            if self.peek()? != b'"' {
                return None;
            }
            let key = self.string()?;
            self.skip_ws();
            if self.peek()? != b':' {
                return None;
            }
            self.pos += 1;
            self.skip_ws();
            let value = self.value()?;
            dict.insert(key, value);
            self.skip_ws();
            match self.peek()? {
                b',' => {
                    self.pos += 1;
                    self.skip_ws();
                }
                b'}' => {
                    self.pos += 1;
                    return Some(PyValue::Dict(dict));
                }
                _ => return None,
            }
        }
    }

    fn array(&mut self) -> Option<PyValue> {
        self.pos += 1;
        let mut items = Vec::new();
        self.skip_ws();
        if self.peek()? == b']' {
            self.pos += 1;
            return Some(PyValue::List(items));
        }
        loop {
            items.push(self.value()?);
            self.skip_ws();
            match self.peek()? {
                b',' => {
                    self.pos += 1;
                    self.skip_ws();
                }
                b']' => {
                    self.pos += 1;
                    return Some(PyValue::List(items));
                }
                _ => return None,
            }
        }
    }

    fn string(&mut self) -> Option<String> {
        self.pos += 1;
        let mut out = String::new();
        loop {
            let rest = &self.s[self.pos..];
            let c = rest.chars().next()?;
            self.pos += c.len_utf8();
            match c {
                '"' => return Some(out),
                '\\' => {
                    let escape = self.peek()?;
                    self.pos += 1;
                    match escape {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\x08'),
                        b'f' => out.push('\x0c'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let unit = self.hex4()?;
                            let code = if (0xD800..0xDC00).contains(&unit) {
                                if !self.s[self.pos..].starts_with("\\u") {
                                    return None;
                                }
                                self.pos += 2;
                                let low = self.hex4()?;
                                if !(0xDC00..0xE000).contains(&low) {
                                    return None;
                                }
                                0x10000 + ((unit - 0xD800) << 10) + (low - 0xDC00)
                            } else {
                                unit
                            };
                            out.push(char::from_u32(code)?);
                        }
                        _ => return None,
                    }
                }
                c if (c as u32) < 0x20 => return None,
                c => out.push(c),
            }
        }
    }

    fn hex4(&mut self) -> Option<u32> {
        let digits = self.s.get(self.pos..self.pos + 4)?;
        if !digits.bytes().all(|b| b.is_ascii_hexdigit()) {
            return None;
        }
        self.pos += 4;
        u32::from_str_radix(digits, 16).ok()
    }

    /// `-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][-+]?[0-9]+)?`, ASCII digits only.
    fn number(&mut self) -> Option<PyValue> {
        let bytes = self.s.as_bytes();
        let start = self.pos;
        let mut i = start;
        if bytes.get(i) == Some(&b'-') {
            i += 1;
        }
        match bytes.get(i)? {
            b'0' => i += 1,
            b'1'..=b'9' => {
                while bytes.get(i).is_some_and(u8::is_ascii_digit) {
                    i += 1;
                }
            }
            _ => return None,
        }
        let mut is_float = false;
        if bytes.get(i) == Some(&b'.') && bytes.get(i + 1).is_some_and(u8::is_ascii_digit) {
            is_float = true;
            i += 1;
            while bytes.get(i).is_some_and(u8::is_ascii_digit) {
                i += 1;
            }
        }
        if matches!(bytes.get(i), Some(b'e' | b'E')) {
            let mut j = i + 1;
            if matches!(bytes.get(j), Some(b'+' | b'-')) {
                j += 1;
            }
            if bytes.get(j).is_some_and(u8::is_ascii_digit) {
                is_float = true;
                i = j;
                while bytes.get(i).is_some_and(u8::is_ascii_digit) {
                    i += 1;
                }
            }
        }
        let text = &self.s[start..i];
        self.pos = i;
        if is_float {
            let value: f64 = text.parse().ok()?;
            value.is_finite().then_some(PyValue::Float(value))
        } else {
            parse_int(text).map(PyValue::Int)
        }
    }
}
