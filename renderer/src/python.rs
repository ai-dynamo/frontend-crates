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

/// Formatter matching Python `json.dumps` default output: `", "` and `": "`
/// separators, and Python float spelling. serde_json's `CompactFormatter` writes
/// `","`/`":"` instead, and chat templates embed these strings directly into the
/// prompt, so the separator choice is model-visible.
pub(crate) struct PyJsonFormatter;

impl Formatter for PyJsonFormatter {
    fn write_f64<W: ?Sized + io::Write>(&mut self, writer: &mut W, value: f64) -> io::Result<()> {
        write_python_float(writer, value)
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
