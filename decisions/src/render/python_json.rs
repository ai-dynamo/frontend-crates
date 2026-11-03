// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::io;

use serde::Serialize;
use serde_json::{Value, ser::Formatter};

/// Compact Python json.dumps(..., ensure_ascii=False) for model-visible text.
pub(super) fn to_string(value: &Value) -> String {
    let mut bytes = Vec::new();
    let mut serializer = serde_json::Serializer::with_formatter(&mut bytes, PythonCompact);
    value
        .serialize(&mut serializer)
        .expect("serializing a JSON value to memory cannot fail");
    String::from_utf8(bytes).expect("JSON serialization produces UTF-8")
}

struct PythonCompact;

impl Formatter for PythonCompact {
    fn write_f64<W: ?Sized + io::Write>(&mut self, writer: &mut W, value: f64) -> io::Result<()> {
        // Preserve shortest-roundtrip digits with Python's notation boundaries
        // and signed, two-digit exponents; other hooks retain compact JSON.
        let shortest = serde_json::Number::from_f64(value)
            .expect("JSON numbers are finite")
            .to_string();
        self.write_number_str(writer, &shortest)
    }

    fn write_number_str<W: ?Sized + io::Write>(
        &mut self,
        writer: &mut W,
        shortest: &str,
    ) -> io::Result<()> {
        // arbitrary_precision sends Number values through this hook, including
        // floats. Respell their digits directly so no precision is lost to f64.
        if !shortest.contains(['.', 'e', 'E']) {
            return writer.write_all(shortest.as_bytes());
        }
        let (sign, unsigned) = shortest
            .strip_prefix('-')
            .map_or(("", shortest), |unsigned| ("-", unsigned));
        let (mantissa, exponent) = match unsigned.split_once(['e', 'E']) {
            Some((m, e)) => match e.parse::<i64>() {
                Ok(e) => (m, e),
                Err(_) => return writer.write_all(shortest.as_bytes()),
            },
            None => (unsigned, 0),
        };
        let (integer, fraction) = mantissa.split_once('.').unwrap_or((mantissa, ""));
        let all_digits = format!("{integer}{fraction}");
        let significant = all_digits.trim_start_matches('0');
        let digits = significant.trim_end_matches('0');
        if digits.is_empty() {
            return write!(writer, "{sign}0.0");
        }
        let leading_zeros = all_digits.len() - significant.len();
        let Some(exponent) = exponent.checked_add(integer.len() as i64 - 1 - leading_zeros as i64)
        else {
            return writer.write_all(shortest.as_bytes());
        };
        let representation = if (-4..16).contains(&exponent) {
            fixed_float(digits, (exponent + 1) as i32)
        } else {
            let (head, tail) = digits.split_at(1);
            let fraction = if tail.is_empty() {
                String::new()
            } else {
                format!(".{tail}")
            };
            let exponent_sign = if exponent < 0 { '-' } else { '+' };
            format!(
                "{head}{fraction}e{exponent_sign}{:02}",
                exponent.unsigned_abs()
            )
        };
        write!(writer, "{sign}{representation}")
    }
}

fn fixed_float(digits: &str, point: i32) -> String {
    if point <= 0 {
        format!("0.{}{digits}", "0".repeat(point.unsigned_abs() as usize))
    } else if point as usize >= digits.len() {
        format!("{digits}{}.0", "0".repeat(point as usize - digits.len()))
    } else {
        format!(
            "{}.{}",
            &digits[..point as usize],
            &digits[point as usize..]
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numeric_text_keeps_exact_digits() {
        for (input, expected) in [
            ("1e-7", "1e-07"),
            ("1e15", "1000000000000000.0"),
            ("9007199254740993.25", "9007199254740993.25"),
            ("1e-400", "1e-400"),
            ("-0.0", "-0.0"),
            ("1e99999999999999999999", "1e99999999999999999999"),
        ] {
            let mut bytes = Vec::new();
            PythonCompact.write_number_str(&mut bytes, input).unwrap();
            assert_eq!(String::from_utf8(bytes).unwrap(), expected);
        }
    }
}
