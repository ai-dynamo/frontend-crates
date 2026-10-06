// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub(crate) mod schema;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use serde_json::value::RawValue;

pub(crate) type Object = indexmap::IndexMap<String, ParsedValue>;

#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub(crate) enum ParsedValue {
    Null,
    Bool(bool),
    String(String),
    Number(Box<RawValue>),
    Array(Vec<ParsedValue>),
    Object(Object),
}

impl From<Value> for ParsedValue {
    fn from(value: Value) -> Self {
        match value {
            Value::Null => Self::Null,
            Value::Bool(v) => Self::Bool(v),
            Value::String(v) => Self::String(v),
            Value::Number(v) => {
                Self::Number(RawValue::from_string(v.to_string()).expect("JSON number"))
            }
            Value::Array(v) => Self::Array(v.into_iter().map(Self::from).collect()),
            Value::Object(v) => {
                Self::Object(v.into_iter().map(|(k, v)| (k, Self::from(v))).collect())
            }
        }
    }
}

impl<'de> Deserialize<'de> for ParsedValue {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let raw = Box::<RawValue>::deserialize(deserializer)?;
        parse_at_depth(raw.get(), 0).map_err(serde::de::Error::custom)
    }
}

pub(crate) fn parse_json(text: &str) -> serde_json::Result<ParsedValue> {
    parse_at_depth(text, 0)
}

fn parse_at_depth(text: &str, depth: usize) -> serde_json::Result<ParsedValue> {
    if depth >= 128 {
        return Err(<serde_json::Error as serde::de::Error>::custom(
            "argument nesting limit exceeded",
        ));
    }
    let raw = validated_raw(text)?;
    let text = raw.get();
    Ok(match text.as_bytes()[0] {
        b'{' => {
            let entries: indexmap::IndexMap<String, Box<RawValue>> = serde_json::from_str(text)?;
            ParsedValue::Object(
                entries
                    .into_iter()
                    .map(|(k, v)| Ok((k, parse_at_depth(v.get(), depth + 1)?)))
                    .collect::<serde_json::Result<_>>()?,
            )
        }
        b'[' => {
            let values: Vec<Box<RawValue>> = serde_json::from_str(text)?;
            ParsedValue::Array(
                values
                    .into_iter()
                    .map(|v| parse_at_depth(v.get(), depth + 1))
                    .collect::<serde_json::Result<_>>()?,
            )
        }
        b'"' => ParsedValue::String(serde_json::from_str(text)?),
        b't' => ParsedValue::Bool(true),
        b'f' => ParsedValue::Bool(false),
        b'n' => ParsedValue::Null,
        _ => ParsedValue::Number(raw),
    })
}

fn validated_raw(text: &str) -> serde_json::Result<Box<RawValue>> {
    let raw: Box<RawValue> = serde_json::from_str(text)?;
    let mut depth = 0usize;
    let mut quoted = false;
    let mut escaped = false;
    // Validate every subtree before duplicate keys can discard a value. RawValue
    // validates number grammar without imposing f64 range, but has no depth cap.
    for byte in raw.get().bytes() {
        if quoted {
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                quoted = false;
            }
        } else {
            match byte {
                b'"' => quoted = true,
                b'{' | b'[' => {
                    depth += 1;
                    if depth >= 128 {
                        return Err(<serde_json::Error as serde::de::Error>::custom(
                            "argument nesting limit exceeded",
                        ));
                    }
                }
                b'}' | b']' => depth -= 1,
                _ => {}
            }
        }
    }
    Ok(raw)
}

pub(crate) fn valid_json(text: &str) -> bool {
    validated_raw(text).is_ok()
}

pub(crate) fn parse_object(text: &str) -> serde_json::Result<Object> {
    match parse_json(text)? {
        ParsedValue::Object(object) => Ok(object),
        _ => Err(<serde_json::Error as serde::de::Error>::custom(
            "expected argument object",
        )),
    }
}

pub(crate) fn is_integer_literal(value: &str) -> bool {
    let value = value.strip_prefix('-').unwrap_or(value);
    !value.is_empty() && value.bytes().all(|b| b.is_ascii_digit())
}

pub(crate) fn raw_number_literal(value: &str) -> Option<ParsedValue> {
    if !value.starts_with(|ch: char| ch == '-' || ch.is_ascii_digit()) {
        return None;
    }
    RawValue::from_string(value.to_string())
        .ok()
        .map(ParsedValue::Number)
}

pub(crate) fn coerce_integer_literal(value: &str) -> Option<ParsedValue> {
    if !is_integer_literal(value) {
        return None;
    }
    if let Ok(n) = value.parse::<i64>() {
        return raw_number_literal(&n.to_string());
    }
    raw_number_literal(value)
}

// Saturating the exponent bounds classification by token size, never its magnitude.
pub(crate) fn coerce_integral_number(value: &str) -> Option<ParsedValue> {
    let value = value.trim();
    if is_integer_literal(value) {
        return coerce_integer_literal(value);
    }
    let raw = raw_number_literal(value)?;
    let unsigned = value.strip_prefix('-').unwrap_or(value);
    let (mantissa, exponent) = unsigned.split_once(['e', 'E']).unwrap_or((unsigned, "0"));
    let negative_exponent = exponent.starts_with('-');
    let magnitude = exponent
        .trim_start_matches(['+', '-'])
        .bytes()
        .fold(0i64, |n, b| {
            n.saturating_mul(10).saturating_add(i64::from(b - b'0'))
        });
    let exponent = if negative_exponent {
        -magnitude
    } else {
        magnitude
    };
    let (whole, fraction) = mantissa.split_once('.').unwrap_or((mantissa, ""));
    let digits = format!("{whole}{fraction}");
    let significant = digits.trim_start_matches('0').trim_end_matches('0');
    if significant.is_empty() {
        return coerce_integer_literal("0");
    }
    let trailing = digits.len() - digits.trim_end_matches('0').len();
    let zeros = exponent
        .saturating_sub(fraction.len() as i64)
        .saturating_add(trailing as i64);
    if zeros < 0 {
        return None;
    }
    if zeros > 20 || significant.len() > 20 - zeros as usize {
        return Some(raw);
    }
    let sign = if value.starts_with('-') { "-" } else { "" };
    coerce_integer_literal(&format!(
        "{sign}{significant}{}",
        "0".repeat(zeros as usize)
    ))
}

#[derive(Clone, Copy)]
pub(crate) enum NumberSpelling {
    Preserve,
    NormalizeIntegral,
}

pub(crate) fn coerce_number_value(value: &str) -> Option<ParsedValue> {
    number_value(value, NumberSpelling::NormalizeIntegral)
}

pub(crate) fn number_value(value: &str, spelling: NumberSpelling) -> Option<ParsedValue> {
    if matches!(spelling, NumberSpelling::Preserve)
        && let Some(raw) = raw_number_literal(value)
    {
        return Some(raw);
    }
    if value.starts_with("+-") {
        return None;
    }
    if let Some(number) = coerce_integral_number(value).or_else(|| raw_number_literal(value)) {
        return Some(number);
    }
    // XML dialects historically accept decimal spellings outside JSON's grammar.
    let unsigned = value.strip_prefix('+').unwrap_or(value);
    let (sign, unsigned) = unsigned
        .strip_prefix('-')
        .map_or(("", unsigned), |v| ("-", v));
    let (mantissa, exponent) = unsigned
        .split_once(['e', 'E'])
        .map_or((unsigned, None), |(m, e)| (m, Some(e)));
    let (whole, fraction) = mantissa
        .split_once('.')
        .map_or((mantissa, None), |(w, f)| (w, Some(f)));
    if whole.is_empty() && fraction.is_none_or(str::is_empty) {
        return None;
    }
    if !whole.bytes().all(|b| b.is_ascii_digit())
        || fraction.is_some_and(|f| !f.bytes().all(|b| b.is_ascii_digit()))
    {
        return None;
    }
    let whole = whole.trim_start_matches('0');
    let mut normalized = format!("{sign}{}", if whole.is_empty() { "0" } else { whole });
    if let Some(fraction) = fraction.filter(|f| !f.is_empty()) {
        normalized.push('.');
        normalized.push_str(fraction);
    }
    if let Some(exponent) = exponent {
        normalized.push('e');
        normalized.push_str(exponent);
    }
    coerce_integral_number(&normalized).or_else(|| raw_number_literal(&normalized))
}

pub(crate) fn html_unescape(s: &str) -> String {
    s.replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&amp;", "&")
        .replace("&quot;", "\"")
        .replace("&#x27;", "'")
        .replace("&#39;", "'")
}

#[cfg(test)]
impl PartialEq<Value> for ParsedValue {
    fn eq(&self, other: &Value) -> bool {
        serde_json::from_str::<Value>(&serde_json::to_string(self).unwrap()).unwrap() == *other
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numeric_grammar_and_bounded_integrality() {
        for raw in ["+1", ".5", "-.5", "1.", "0001.5"] {
            assert!(coerce_number_value(raw).is_some(), "{raw}");
            assert!(raw_number_literal(raw).is_none(), "{raw}");
        }
        for raw in [
            "+-1", "--1", "1e", "1e+", "NaN", "inf", "true", "[]", "{}", "\"42\"",
        ] {
            assert!(coerce_number_value(raw).is_none(), "{raw}");
            assert!(raw_number_literal(raw).is_none(), "{raw}");
        }
        for (raw, expected) in [
            ("1e1 ", "10"),
            ("42.0", "42"),
            ("4.2e1", "42"),
            ("0e-999999999999999999999", "0"),
            ("1e999999999999999999999", "1e999999999999999999999"),
        ] {
            assert_eq!(
                serde_json::to_string(&coerce_integral_number(raw).unwrap()).unwrap(),
                expected
            );
        }
        for raw in ["42.0000000000000001", "1e-400", "1e-999999999999999999999"] {
            assert!(coerce_integral_number(raw).is_none());
            assert_eq!(
                serde_json::to_string(&number_value(raw, NumberSpelling::Preserve).unwrap())
                    .unwrap(),
                raw
            );
        }
        for raw in ["-0", "-0.0", "-0e0"] {
            assert_eq!(
                serde_json::to_string(&number_value(raw, NumberSpelling::Preserve).unwrap())
                    .unwrap(),
                raw
            );
        }
    }

    #[test]
    fn validation_and_conversion_share_depth_limits_before_duplicate_reduction() {
        for depth in [126, 127, 128, 129] {
            let nested = format!("{}0{}", "[".repeat(depth), "]".repeat(depth));
            for raw in [
                nested.clone(),
                format!("{{\"a\":{nested},\"a\":0}}"),
                format!("{{\"a\":{nested}}}"),
            ] {
                let baseline = serde_json::from_str::<Value>(&raw).is_ok();
                assert_eq!(valid_json(&raw), baseline, "depth {depth}");
                assert_eq!(parse_json(&raw).is_ok(), baseline, "depth {depth}");
            }
        }
    }

    #[test]
    fn exact_json_retains_order_and_duplicate_value_policy() {
        let input = r#"{"z":9007199254740992.5,"a":1e-400,"z":9007199254740993.25}"#;
        assert_eq!(
            serde_json::to_string(&parse_json(input).unwrap()).unwrap(),
            r#"{"z":9007199254740993.25,"a":1e-400}"#
        );
    }
}
