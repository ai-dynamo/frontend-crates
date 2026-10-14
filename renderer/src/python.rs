// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Python value spellings. Reference encoders and HF chat templates render through
//! Python, so the prompt text they produce uses Python's `json.dumps` and `str`.

use std::io;

use serde_json::ser::Formatter;

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
    let shortest = shortest.to_string();
    let (sign, unsigned) = match shortest.strip_prefix('-') {
        Some(unsigned) => ("-", unsigned),
        None => ("", shortest.as_str()),
    };
    let (mantissa, exponent) = match unsigned.split_once('e') {
        Some((mantissa, exponent)) => (mantissa, exponent.parse().expect("ryu exponent")),
        None => (unsigned, 0),
    };
    let (integer, fraction) = mantissa.split_once('.').unwrap_or((mantissa, ""));
    let all_digits = format!("{integer}{fraction}");
    let significant = all_digits.trim_start_matches('0');
    let digits = significant.trim_end_matches('0');
    if digits.is_empty() {
        return format!("{sign}0.0");
    }
    let leading_zeros = (all_digits.len() - significant.len()) as i32;
    let exponent: i32 = exponent + integer.len() as i32 - 1 - leading_zeros;
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
