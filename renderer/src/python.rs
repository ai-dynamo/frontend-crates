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

/// Formatter matching Python `json.dumps` default output: `", "` and `": "`
/// separators, and Python float spelling. serde_json's `CompactFormatter` writes
/// `","`/`":"` instead, and chat templates embed these strings directly into the
/// prompt, so the separator choice is model-visible.
pub(crate) struct PyJsonFormatter;

impl Formatter for PyJsonFormatter {
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

    fn begin_array_value<W>(&mut self, writer: &mut W, first: bool) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        if !first {
            writer.write_all(b", ")?;
        }
        Ok(())
    }

    fn begin_object_key<W>(&mut self, writer: &mut W, first: bool) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        if !first {
            writer.write_all(b", ")?;
        }
        Ok(())
    }

    fn begin_object_value<W>(&mut self, writer: &mut W) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        writer.write_all(b": ")
    }
}

/// Adds Python float spelling to another formatter, e.g. `PrettyFormatter` for
/// `json.dumps(indent=n)`. Only the array, object, and separator hooks are delegated
/// to the wrapped formatter; every other hook uses serde_json's default.
pub(crate) struct PyFloats<F>(pub(crate) F);

impl<F: Formatter> Formatter for PyFloats<F> {
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
        self.0.begin_array(writer)
    }

    fn end_array<W: ?Sized + io::Write>(&mut self, writer: &mut W) -> io::Result<()> {
        self.0.end_array(writer)
    }

    fn begin_array_value<W: ?Sized + io::Write>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> io::Result<()> {
        self.0.begin_array_value(writer, first)
    }

    fn end_array_value<W: ?Sized + io::Write>(&mut self, writer: &mut W) -> io::Result<()> {
        self.0.end_array_value(writer)
    }

    fn begin_object<W: ?Sized + io::Write>(&mut self, writer: &mut W) -> io::Result<()> {
        self.0.begin_object(writer)
    }

    fn end_object<W: ?Sized + io::Write>(&mut self, writer: &mut W) -> io::Result<()> {
        self.0.end_object(writer)
    }

    fn begin_object_key<W: ?Sized + io::Write>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> io::Result<()> {
        self.0.begin_object_key(writer, first)
    }

    fn end_object_key<W: ?Sized + io::Write>(&mut self, writer: &mut W) -> io::Result<()> {
        self.0.end_object_key(writer)
    }

    fn begin_object_value<W: ?Sized + io::Write>(&mut self, writer: &mut W) -> io::Result<()> {
        self.0.begin_object_value(writer)
    }

    fn end_object_value<W: ?Sized + io::Write>(&mut self, writer: &mut W) -> io::Result<()> {
        self.0.end_object_value(writer)
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
            PyJsonFormatter
                .write_number_str(&mut compact, input)
                .unwrap();
            let mut wrapped = Vec::new();
            PyFloats(serde_json::ser::CompactFormatter)
                .write_number_str(&mut wrapped, input)
                .unwrap();
            assert_eq!(String::from_utf8(compact).unwrap(), expected);
            assert_eq!(String::from_utf8(wrapped).unwrap(), expected);
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
