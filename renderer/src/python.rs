// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Python value spellings. Reference encoders and HF chat templates render through
//! Python, so the prompt text they produce uses Python's `json.dumps` and `str`.

use std::io;

use serde_json::ser::Formatter;

/// Recover serde_json numbers after MiniJinja's serializer has represented
/// arbitrary-precision Number structs as maps. Keep this at the JSON boundary;
/// canonical request values never pass through MiniJinja's bounded floats.
pub(crate) fn template_json<T: serde::Serialize>(
    value: T,
) -> serde_json::Result<serde_json::Value> {
    serde_json::from_value(serde_json::to_value(value)?)
}

/// MiniJinja doesn't understand serde_json's arbitrary-precision Number struct.
/// Construct its native numeric values explicitly, retaining the Python float
/// behavior of the template engine rather than exposing private Number fields.
pub(crate) fn template_value(value: &serde_json::Value) -> minijinja::Value {
    use serde_json::Value;
    match value {
        Value::Null => minijinja::Value::from(()),
        Value::Bool(value) => minijinja::Value::from(*value),
        Value::String(value) => minijinja::Value::from(value.clone()),
        Value::Number(value) => {
            if let Some(value) = value.as_i64() {
                minijinja::Value::from(value)
            } else if let Some(value) = value.as_u64() {
                minijinja::Value::from(value)
            } else {
                minijinja::Value::from(value.to_string().parse::<f64>().expect("valid JSON number"))
            }
        }
        Value::Array(values) => values.iter().map(template_value).collect(),
        Value::Object(values) => values
            .iter()
            .map(|(key, value)| (key.clone(), template_value(value)))
            .collect(),
    }
}

/// Python's `repr(float)`, which `json.dumps` and `str` use: the shortest round-trip
/// digits, fixed notation for exponents in [-4, 16) (always with a fraction, `100.0`),
/// otherwise scientific with a signed, two-digit exponent (`1e-06`, `1e+16`).
/// Non-finite values are `nan`, `inf`, and `-inf`.
pub(crate) fn python_float_repr(value: f64) -> String {
    // serde_json's ryu picks the same shortest digits as Python, halfway ties
    // included (`{:e}` breaks some differently); only the notation is respelled.
    let Some(shortest) = serde_json::Number::from_f64(value) else {
        let non_finite = if value.is_nan() {
            "nan"
        } else if value < 0.0 {
            "-inf"
        } else {
            "inf"
        };
        return non_finite.to_string();
    };
    python_number_repr(&shortest.to_string())
}

/// Preserve arbitrary-precision digits while applying Python's float notation.
fn python_number_repr(shortest: &str) -> String {
    if !shortest.contains(['.', 'e', 'E']) {
        return shortest.to_owned();
    }
    let (sign, unsigned) = match shortest.strip_prefix('-') {
        Some(unsigned) => ("-", unsigned),
        None => ("", shortest),
    };
    let (mantissa, exponent) = match unsigned.split_once(['e', 'E']) {
        Some((mantissa, exponent)) => match exponent.parse::<i64>() {
            Ok(exponent) => (mantissa, exponent),
            Err(_) => return shortest.to_owned(),
        },
        None => (unsigned, 0),
    };
    let (integer, fraction) = mantissa.split_once('.').unwrap_or((mantissa, ""));
    let all_digits = format!("{integer}{fraction}");
    let significant = all_digits.trim_start_matches('0');
    let digits = significant.trim_end_matches('0');
    if digits.is_empty() {
        return format!("{sign}0.0");
    }
    let leading_zeros = (all_digits.len() - significant.len()) as i64;
    let Some(exponent) = exponent.checked_add(integer.len() as i64 - 1 - leading_zeros) else {
        return shortest.to_owned();
    };
    if (-4..16).contains(&exponent) {
        let point = exponent + 1;
        let fixed = if point <= 0 {
            format!("0.{}{digits}", "0".repeat(point.unsigned_abs() as usize))
        } else if point as usize >= digits.len() {
            format!("{digits}{}.0", "0".repeat(point as usize - digits.len()))
        } else {
            format!(
                "{}.{}",
                &digits[..point as usize],
                &digits[point as usize..]
            )
        };
        format!("{sign}{fixed}")
    } else {
        let (head, tail) = digits.split_at(1);
        let mantissa = if tail.is_empty() {
            head.to_string()
        } else {
            format!("{head}.{tail}")
        };
        let exponent_sign = if exponent < 0 { '-' } else { '+' };
        format!(
            "{sign}{mantissa}e{exponent_sign}{:02}",
            exponent.unsigned_abs()
        )
    }
}

/// Writes a float as Python `json.dumps` does, instead of serde_json's `1e-6`/`0.00001`.
/// serde_json writes non-finite floats as `null` before reaching a formatter, so the
/// guard here is only defensive; Python would write `NaN`/`Infinity`, which request
/// JSON can't carry.
fn write_python_float<W: ?Sized + io::Write>(writer: &mut W, value: f64) -> io::Result<()> {
    if !value.is_finite() {
        return writer.write_all(b"null");
    }
    writer.write_all(python_float_repr(value).as_bytes())
}

/// Formatter matching Python `json.dumps` output for the given `separators`, `indent`
/// and `ensure_ascii`, with Python float spelling. Python's default separators are
/// `", "` and `": "`; serde_json's `CompactFormatter` writes `","`/`":"` instead, and
/// chat templates embed these strings directly into the prompt, so the separator
/// choice is model-visible.
pub(crate) struct PyJsonFormatter<'a> {
    item_sep: &'a str,
    key_sep: &'a str,
    /// Python `indent`: each item starts on a new line, indented by one copy of this
    /// per nesting level.
    indent: Option<&'a str>,
    level: usize,
    has_value: bool,
    ensure_ascii: bool,
}

impl<'a> PyJsonFormatter<'a> {
    pub(crate) fn new(
        item_sep: &'a str,
        key_sep: &'a str,
        indent: Option<&'a str>,
        ensure_ascii: bool,
    ) -> Self {
        Self {
            item_sep,
            key_sep,
            indent,
            level: 0,
            has_value: false,
            ensure_ascii,
        }
    }

    fn newline_indent<W: ?Sized + io::Write>(&self, writer: &mut W) -> io::Result<()> {
        if let Some(indent) = self.indent {
            writer.write_all(b"\n")?;
            for _ in 0..self.level {
                writer.write_all(indent.as_bytes())?;
            }
        }
        Ok(())
    }

    fn open<W: ?Sized + io::Write>(&mut self, writer: &mut W, bracket: &[u8]) -> io::Result<()> {
        self.level += 1;
        self.has_value = false;
        writer.write_all(bracket)
    }

    fn close<W: ?Sized + io::Write>(&mut self, writer: &mut W, bracket: &[u8]) -> io::Result<()> {
        self.level -= 1;
        if self.has_value {
            self.newline_indent(writer)?;
        }
        writer.write_all(bracket)
    }

    fn begin_item<W: ?Sized + io::Write>(&mut self, writer: &mut W, first: bool) -> io::Result<()> {
        if !first {
            writer.write_all(self.item_sep.as_bytes())?;
        }
        self.newline_indent(writer)
    }
}

/// Python's `json.dumps(x, ensure_ascii=False)`.
impl Default for PyJsonFormatter<'static> {
    fn default() -> Self {
        Self::new(", ", ": ", None, false)
    }
}

impl Formatter for PyJsonFormatter<'_> {
    fn write_f64<W: ?Sized + io::Write>(&mut self, writer: &mut W, value: f64) -> io::Result<()> {
        write_python_float(writer, value)
    }

    fn write_number_str<W: ?Sized + io::Write>(
        &mut self,
        writer: &mut W,
        value: &str,
    ) -> io::Result<()> {
        writer.write_all(python_number_repr(value).as_bytes())
    }

    fn begin_array<W: ?Sized + io::Write>(&mut self, writer: &mut W) -> io::Result<()> {
        self.open(writer, b"[")
    }

    fn end_array<W: ?Sized + io::Write>(&mut self, writer: &mut W) -> io::Result<()> {
        self.close(writer, b"]")
    }

    fn begin_array_value<W: ?Sized + io::Write>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> io::Result<()> {
        self.begin_item(writer, first)
    }

    fn end_array_value<W: ?Sized + io::Write>(&mut self, _writer: &mut W) -> io::Result<()> {
        self.has_value = true;
        Ok(())
    }

    fn begin_object<W: ?Sized + io::Write>(&mut self, writer: &mut W) -> io::Result<()> {
        self.open(writer, b"{")
    }

    fn end_object<W: ?Sized + io::Write>(&mut self, writer: &mut W) -> io::Result<()> {
        self.close(writer, b"}")
    }

    fn begin_object_key<W: ?Sized + io::Write>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> io::Result<()> {
        self.begin_item(writer, first)
    }

    fn begin_object_value<W: ?Sized + io::Write>(&mut self, writer: &mut W) -> io::Result<()> {
        writer.write_all(self.key_sep.as_bytes())
    }

    fn end_object_value<W: ?Sized + io::Write>(&mut self, _writer: &mut W) -> io::Result<()> {
        self.has_value = true;
        Ok(())
    }

    // serde_json already escaped control characters, `"` and `\`. Python's
    // `ensure_ascii` additionally escapes everything outside `' '..='~'` as
    // `\uXXXX`, using UTF-16 surrogate pairs above the BMP.
    fn write_string_fragment<W: ?Sized + io::Write>(
        &mut self,
        writer: &mut W,
        fragment: &str,
    ) -> io::Result<()> {
        if !self.ensure_ascii {
            return writer.write_all(fragment.as_bytes());
        }
        let mut start = 0;
        for (i, c) in fragment.char_indices() {
            if !(' '..='~').contains(&c) {
                writer.write_all(&fragment.as_bytes()[start..i])?;
                for unit in c.encode_utf16(&mut [0; 2]) {
                    write!(writer, "\\u{unit:04x}")?;
                }
                start = i + c.len_utf8();
            }
        }
        writer.write_all(&fragment.as_bytes()[start..])
    }
}

/// Python's `str.isspace` for one character: Rust's `char::is_whitespace` plus
/// U+001C..U+001F, which Python also counts as whitespace.
pub(crate) fn is_python_space(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

/// Python's `str.split()` with no separator: whitespace runs separate the words,
/// and after `maxsplit` splits the rest is one word, its trailing whitespace kept.
pub(crate) fn python_split_whitespace(s: &str, maxsplit: Option<usize>) -> Vec<&str> {
    let mut words = Vec::new();
    let mut rest = s;
    loop {
        rest = rest.trim_start_matches(is_python_space);
        if rest.is_empty() {
            return words;
        }
        if maxsplit == Some(words.len()) {
            words.push(rest);
            return words;
        }
        let end = rest.find(is_python_space).unwrap_or(rest.len());
        words.push(&rest[..end]);
        rest = &rest[end..];
    }
}

/// Python's `str.splitlines`: `\r\n` and each of `\n \r \v \f \x1c \x1d \x1e \x85
/// U+2028 U+2029` end a line, kept with `keepends`.
pub(crate) fn python_splitlines(s: &str, keepends: bool) -> Vec<&str> {
    let mut lines = Vec::new();
    let mut start = 0;
    let mut chars = s.char_indices().peekable();
    while let Some((i, c)) = chars.next() {
        if !matches!(
            c,
            '\n' | '\r'
                | '\u{b}'
                | '\u{c}'
                | '\u{1c}'
                | '\u{1d}'
                | '\u{1e}'
                | '\u{85}'
                | '\u{2028}'
                | '\u{2029}'
        ) {
            continue;
        }
        let mut end = i + c.len_utf8();
        if c == '\r' && chars.next_if(|&(_, next)| next == '\n').is_some() {
            end += 1;
        }
        lines.push(&s[start..if keepends { end } else { i }]);
        start = end;
    }
    if start < s.len() {
        lines.push(&s[start..]);
    }
    lines
}

/// Python's `str.find` (or `str.rfind` with `reverse`): the character index of `sub`
/// in `s[start:end]`, or -1. Rust's `str::find` returns a byte offset, but templates
/// slice strings by character, so the two disagree on non-ASCII text.
pub(crate) fn python_find(
    s: &str,
    sub: &str,
    start: Option<i64>,
    end: Option<i64>,
    reverse: bool,
) -> i64 {
    let len = s.chars().count() as i64;
    // CPython's ADJUST_INDICES: negative indices count from the end, `end` is capped
    // at the length, and `start` past the end finds nothing, not even `''`.
    let adjust = |index: i64| {
        if index < 0 {
            (index + len).max(0)
        } else {
            index
        }
    };
    let start = start.map_or(0, adjust);
    let end = end.map_or(len, adjust).min(len);
    if end - start < sub.chars().count() as i64 {
        return -1;
    }
    let byte_offset = |index: i64| {
        s.char_indices()
            .nth(index as usize)
            .map_or(s.len(), |(offset, _)| offset)
    };
    let (byte_start, byte_end) = (byte_offset(start), byte_offset(end));
    let window = &s[byte_start..byte_end];
    let found = if reverse {
        window.rfind(sub)
    } else {
        window.find(sub)
    };
    found.map_or(-1, |offset| start + window[..offset].chars().count() as i64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn template_boundaries_keep_numbers_numeric() {
        let json = serde_json::json!({"values": [1, 1e-7, 1e15, -0.0]});
        let template = template_value(&json);
        let values = template.get_attr("values").unwrap();
        for value in values.try_iter().unwrap() {
            assert!(value.is_number());
        }
        assert_eq!(template_json(template).unwrap(), json);
        let externally_serialized = minijinja::Value::from_serialize(&json);
        assert_eq!(template_json(externally_serialized).unwrap(), json);
    }

    #[test]
    fn number_spelling_preserves_exact_digits() {
        for (input, expected) in [
            ("1e-7", "1e-07"),
            ("1e15", "1000000000000000.0"),
            ("1E-6", "1e-06"),
            ("-0.0", "-0.0"),
            ("9007199254740993", "9007199254740993"),
            ("9007199254740993.25", "9007199254740993.25"),
            ("1e-400", "1e-400"),
            ("1e99999999999999999999", "1e99999999999999999999"),
        ] {
            let mut compact = Vec::new();
            PyJsonFormatter::default()
                .write_number_str(&mut compact, input)
                .unwrap();
            let mut indented = Vec::new();
            PyJsonFormatter::new(",", ": ", Some("  "), false)
                .write_number_str(&mut indented, input)
                .unwrap();
            assert_eq!(String::from_utf8(compact).unwrap(), expected);
            assert_eq!(String::from_utf8(indented).unwrap(), expected);
        }
    }

    #[test]
    fn floats_match_python_repr() {
        for (value, python) in [
            (0.000001, "1e-06"),
            (0.0001, "0.0001"),
            (0.00001234, "1.234e-05"),
            (1e16, "1e+16"),
            (1e15, "1000000000000000.0"),
            (1.5e-7, "1.5e-07"),
            (2.5, "2.5"),
            (100.0, "100.0"),
            (0.1, "0.1"),
            (-0.0, "-0.0"),
            (0.0, "0.0"),
            (-123.456, "-123.456"),
            (1.7976931348623157e308, "1.7976931348623157e+308"),
            (5e-324, "5e-324"),
            // Exactly halfway between two shortest candidates.
            (1e15 + 0.25, "1000000000000000.2"),
            (1e14 + 0.125, "100000000000000.12"),
            (f64::NAN, "nan"),
            (f64::INFINITY, "inf"),
            (f64::NEG_INFINITY, "-inf"),
        ] {
            assert_eq!(python_float_repr(value), python, "{value:?}");
        }
    }

    /// Expected values from CPython 3.12's `str.find()` and `str.rfind()`.
    #[test]
    fn find_matches_python() {
        let s = "\u{e9}x\u{4f60}\u{597d}x<tool_call>";
        for (sub, start, end, expected) in [
            ("<tool_call>", None, None, (5, 5)),
            ("x", None, None, (1, 4)),
            ("x", Some(-2), None, (-1, -1)),
            ("x", Some(1), Some(-1), (1, 4)),
            ("x", Some(-100), Some(100), (1, 4)),
            ("", None, None, (0, 16)),
            ("", Some(17), None, (-1, -1)),
            ("", Some(-3), Some(-5), (-1, -1)),
        ] {
            let found = (
                python_find(s, sub, start, end, false),
                python_find(s, sub, start, end, true),
            );
            assert_eq!(found, expected, "{sub:?} {start:?} {end:?}");
        }
    }

    /// Expected values from CPython 3.12's `str.split()`, `str.splitlines()` and `str.isspace()`.
    #[test]
    fn whitespace_matches_python() {
        let all_breaks = "a\rb\r\nc\u{b}d\u{c}e\u{1c}f\u{1d}g\u{1e}h\u{85}i\u{2028}j\u{2029}k\nl\n";
        let split: &[(&str, Option<usize>, &[&str])] = &[
            ("\u{1c} answer \u{1f}", None, &["answer"]),
            ("  a  b \u{1c} c\u{1d}", None, &["a", "b", "c"]),
            ("  a  b \u{1c} c\u{1d}", Some(1), &["a", "b \u{1c} c\u{1d}"]),
            ("  lead and trail  ", Some(1), &["lead", "and trail  "]),
            ("  lead and trail  ", Some(0), &["lead and trail  "]),
            ("\u{85}\u{2028}x\u{2029}\u{3000}", None, &["x"]),
            ("\u{1c}\u{1d}\u{1e}\u{1f}", None, &[]),
            ("", None, &[]),
        ];
        for &(s, maxsplit, words) in split {
            assert_eq!(
                python_split_whitespace(s, maxsplit),
                words,
                "{s:?} {maxsplit:?}"
            );
        }
        let splitlines: &[(&str, &[&str], &[&str])] = &[
            (
                all_breaks,
                &["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l"],
                &[
                    "a\r",
                    "b\r\n",
                    "c\u{b}",
                    "d\u{c}",
                    "e\u{1c}",
                    "f\u{1d}",
                    "g\u{1e}",
                    "h\u{85}",
                    "i\u{2028}",
                    "j\u{2029}",
                    "k\n",
                    "l\n",
                ],
            ),
            (
                "\u{1c} answer \u{1f}",
                &["", " answer \u{1f}"],
                &["\u{1c}", " answer \u{1f}"],
            ),
            ("plain text", &["plain text"], &["plain text"]),
            ("", &[], &[]),
        ];
        for &(s, lines, kept) in splitlines {
            assert_eq!(python_splitlines(s, false), lines, "{s:?}");
            assert_eq!(python_splitlines(s, true), kept, "{s:?}");
        }
        assert!("\u{1c}\u{1d}\u{1e}\u{1f}".chars().all(is_python_space));
        assert!(!"\u{200b}".chars().all(is_python_space));
    }
}
