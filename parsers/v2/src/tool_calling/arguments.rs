// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde::de::{MapAccess, Visitor};
use serde::ser::SerializeMap;
use serde::{Deserializer, Serialize};
use serde_json::Value;
use serde_json::value::RawValue;
use std::collections::BTreeMap;

mod schema;
pub(crate) use schema::SchemaFacts;
#[cfg(test)]
pub(crate) use schema::schema_type_match;
#[cfg(test)]
pub(crate) use schema::{MAX_SCHEMA_DEPTH, MAX_SCHEMA_WORK};
pub(crate) use schema::{
    has_unsupported_schema_ref_scope, resolve_local_schema_ref, schema_has_type,
    schema_permits_null,
};

#[derive(Debug)]
pub(crate) enum ParsedValue {
    Json(Value),
    RawNumber(Box<RawValue>),
    RawJson(Box<RawValue>),
    Array(Vec<ParsedValue>),
    Object(ParsedObject),
}

// DSML, Gemma, M3 and literal JSON retain serde_json::Map's key and removal
// order. Only key markers enter Value; argument values keep their exact tokens.
#[derive(Debug, Default)]
pub(crate) struct ParsedObject {
    keys: serde_json::Map<String, Value>,
    values: BTreeMap<String, ParsedValue>,
}

impl ParsedObject {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn insert(&mut self, key: String, value: ParsedValue) -> Option<ParsedValue> {
        self.keys.insert(key.clone(), Value::Null);
        self.values.insert(key, value)
    }

    pub(crate) fn remove(&mut self, key: &str) -> Option<ParsedValue> {
        self.keys.remove(key);
        self.values.remove(key)
    }

    pub(crate) fn contains_key(&self, key: &str) -> bool {
        self.values.contains_key(key)
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

impl FromIterator<(String, ParsedValue)> for ParsedObject {
    fn from_iter<T: IntoIterator<Item = (String, ParsedValue)>>(items: T) -> Self {
        let mut object = Self::new();
        for (key, value) in items {
            object.insert(key, value);
        }
        object
    }
}

impl Serialize for ParsedObject {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut object = serializer.serialize_map(Some(self.values.len()))?;
        for key in self.keys.keys() {
            object.serialize_entry(key, &self.values[key])?;
        }
        object.end()
    }
}

struct RawFields;

impl<'de> Visitor<'de> for RawFields {
    type Value = Vec<(String, Box<RawValue>)>;

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("a JSON object")
    }

    fn visit_map<M>(self, mut map: M) -> Result<Self::Value, M::Error>
    where
        M: MapAccess<'de>,
    {
        let mut fields = Vec::new();
        while let Some(field) = map.next_entry()? {
            fields.push(field);
        }
        Ok(fields)
    }
}

fn raw_object_fields(value: &str) -> Option<Vec<(String, Box<RawValue>)>> {
    let mut deserializer = serde_json::Deserializer::from_str(value);
    let fields = deserializer.deserialize_map(RawFields).ok()?;
    deserializer.end().ok()?;
    Some(fields)
}

impl PartialEq<Value> for ParsedValue {
    fn eq(&self, other: &Value) -> bool {
        self.to_value() == *other
    }
}

impl std::fmt::Display for ParsedValue {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(
            &self
                .json_text()
                .expect("ParsedValue must serialize as JSON"),
        )
    }
}

impl From<Value> for ParsedValue {
    fn from(value: Value) -> Self {
        Self::Json(value)
    }
}

impl Serialize for ParsedValue {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self {
            Self::Json(value) => value.serialize(serializer),
            Self::RawNumber(raw) => raw.serialize(serializer),
            Self::RawJson(raw) => raw.serialize(serializer),
            Self::Array(items) => items.serialize(serializer),
            Self::Object(fields) => fields.serialize(serializer),
        }
    }
}

impl ParsedValue {
    pub(crate) fn json_text(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }

    pub(crate) fn to_value(&self) -> Value {
        serde_json::from_str(
            &self
                .json_text()
                .expect("ParsedValue must serialize as JSON"),
        )
        .expect("ParsedValue serialization must be valid JSON")
    }

    #[cfg(test)]
    pub(crate) fn as_object(&self) -> Option<&BTreeMap<String, ParsedValue>> {
        match self {
            Self::Object(fields) => Some(&fields.values),
            _ => None,
        }
    }
}

pub(crate) fn is_integer_literal(value: &str) -> bool {
    let value = value.strip_prefix('-').unwrap_or(value);
    !value.is_empty() && value.bytes().all(|b| b.is_ascii_digit())
}

pub(crate) fn raw_number_literal(value: &str) -> Option<ParsedValue> {
    if !is_json_number(value) {
        return None;
    }
    RawValue::from_string(value.to_string())
        .ok()
        .map(ParsedValue::RawNumber)
}

pub(crate) fn raw_json_literal(value: &str) -> Option<ParsedValue> {
    RawValue::from_string(value.to_string())
        .ok()
        .map(ParsedValue::RawJson)
}

pub(crate) fn parsed_json_object(value: &str) -> Option<ParsedValue> {
    let parsed = parsed_json_literal(value)?;
    matches!(&parsed, ParsedValue::Object(_)).then_some(parsed)
}

pub(crate) fn parsed_json_literal(value: &str) -> Option<ParsedValue> {
    RawValue::from_string(value.to_string()).ok()?;
    // RawValue skips serde_json's nesting limit; retain that limit before building
    // recursive nodes so serialization and drop remain stack bounded.
    parsed_json_literal_with_budget(value, 128)
}

fn parsed_json_literal_with_budget(value: &str, budget: usize) -> Option<ParsedValue> {
    let value = value.trim();
    match value.as_bytes().first()? {
        b'{' => {
            let budget = budget.checked_sub(1).filter(|budget| *budget > 0)?;
            let fields = raw_object_fields(value)?;
            fields
                .into_iter()
                .map(|(key, raw)| {
                    parsed_json_literal_with_budget(raw.get(), budget).map(|value| (key, value))
                })
                .collect::<Option<ParsedObject>>()
                .map(ParsedValue::Object)
        }
        b'[' => {
            let budget = budget.checked_sub(1).filter(|budget| *budget > 0)?;
            serde_json::from_str::<Vec<Box<RawValue>>>(value)
                .ok()?
                .into_iter()
                .map(|raw| parsed_json_literal_with_budget(raw.get(), budget))
                .collect::<Option<Vec<_>>>()
                .map(ParsedValue::Array)
        }
        b'"' | b't' | b'f' | b'n' => serde_json::from_str(value).ok().map(ParsedValue::Json),
        _ => raw_number_literal(value),
    }
}

fn is_json_number(value: &str) -> bool {
    let bytes = value.as_bytes();
    let mut cursor = usize::from(bytes.first() == Some(&b'-'));
    if cursor == bytes.len() {
        return false;
    }
    match bytes[cursor] {
        b'0' => cursor += 1,
        b'1'..=b'9' => {
            cursor += 1;
            while bytes.get(cursor).is_some_and(u8::is_ascii_digit) {
                cursor += 1;
            }
        }
        _ => return false,
    }
    if bytes.get(cursor) == Some(&b'.') {
        cursor += 1;
        let start = cursor;
        while bytes.get(cursor).is_some_and(u8::is_ascii_digit) {
            cursor += 1;
        }
        if cursor == start {
            return false;
        }
    }
    if matches!(bytes.get(cursor), Some(b'e' | b'E')) {
        cursor += 1;
        if matches!(bytes.get(cursor), Some(b'+' | b'-')) {
            cursor += 1;
        }
        let start = cursor;
        while bytes.get(cursor).is_some_and(u8::is_ascii_digit) {
            cursor += 1;
        }
        if cursor == start {
            return false;
        }
    }
    cursor == bytes.len()
}

pub(crate) fn normalize_decimal_spelling(value: &str) -> Option<ParsedValue> {
    let (sign, unsigned) = if let Some(unsigned) = value.strip_prefix('+') {
        ("", unsigned)
    } else if let Some(unsigned) = value.strip_prefix('-') {
        ("-", unsigned)
    } else {
        ("", value)
    };
    let (mantissa, exponent) = match unsigned.split_once(['e', 'E']) {
        Some((mantissa, exponent)) => (mantissa, Some(("e", exponent))),
        None => (unsigned, None),
    };
    let (whole, fraction) = match mantissa.split_once('.') {
        Some((whole, fraction)) => (whole, Some(fraction)),
        None => (mantissa, None),
    };
    if whole.is_empty() && fraction.is_none_or(str::is_empty) {
        return None;
    }
    let whole = if whole.is_empty() { "0" } else { whole };
    if whole.bytes().any(|byte| !byte.is_ascii_digit())
        || fraction.is_some_and(|digits| !digits.bytes().all(|byte| byte.is_ascii_digit()))
        || exponent.is_some_and(|(_, value)| {
            let value = value.strip_prefix(['+', '-']).unwrap_or(value);
            value.is_empty() || !value.bytes().all(|byte| byte.is_ascii_digit())
        })
    {
        return None;
    }
    let whole = whole.trim_start_matches('0');
    let whole = if whole.is_empty() { "0" } else { whole };
    let mut normalized = format!("{sign}{whole}");
    if let Some(fraction) = fraction {
        normalized.push('.');
        normalized.push_str(if fraction.is_empty() { "0" } else { fraction });
    }
    if let Some((marker, exponent)) = exponent {
        normalized.push_str(marker);
        normalized.push_str(exponent);
    }
    raw_number_literal(&normalized)
}

pub(crate) fn coerce_integer_literal(value: &str) -> Option<ParsedValue> {
    if !is_integer_literal(value) {
        return None;
    }

    if let Ok(n) = value.parse::<i64>() {
        return Some(Value::Number(n.into()).into());
    }
    Some(raw_number_literal(value).unwrap_or_else(|| Value::String(value.to_string()).into()))
}

pub(crate) fn coerce_integral_number(value: &str) -> Option<ParsedValue> {
    if is_integer_literal(value) {
        return coerce_integer_literal(value);
    }
    let Some(raw) = raw_number_literal(value) else {
        let normalized = normalize_decimal_spelling(value)?.json_text().ok()?;
        if normalized == value {
            return None;
        }
        return coerce_integral_number(&normalized);
    };
    let unsigned = value.strip_prefix('-').unwrap_or(value);
    let (mantissa, exponent_text) = match unsigned.split_once(['e', 'E']) {
        Some((mantissa, exponent)) => (mantissa, Some(exponent)),
        None => (unsigned, None),
    };
    let mantissa_digits = mantissa.bytes().filter(|byte| *byte != b'.');
    if mantissa_digits.clone().all(|byte| byte == b'0') {
        return Some(Value::Number(0.into()).into());
    }
    let exponent = match exponent_text {
        Some(exponent) => match exponent.parse::<i64>() {
            Ok(exponent) => exponent,
            Err(_) if !exponent.starts_with('-') => return Some(raw),
            Err(_) => return None,
        },
        None => 0,
    };
    let (whole, fraction) = mantissa.split_once('.').unwrap_or((mantissa, ""));
    if whole.is_empty()
        || !whole
            .bytes()
            .chain(fraction.bytes())
            .all(|b| b.is_ascii_digit())
    {
        return None;
    }
    let digits = format!("{whole}{fraction}");
    let significant = digits.trim_start_matches('0').trim_end_matches('0');
    let trailing = digits.len() - digits.trim_end_matches('0').len();
    let zeros = exponent
        .checked_sub(i64::try_from(fraction.len()).ok()?)?
        .checked_add(i64::try_from(trailing).ok()?)?;
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

#[cfg(test)]
mod tests {
    use super::{
        ParsedObject, ParsedValue, coerce_integral_number, normalize_decimal_spelling,
        parsed_json_literal, raw_number_literal,
    };

    #[test]
    fn object_mutations_follow_serde_map_order() {
        let mut actual = ParsedObject::new();
        let mut expected = serde_json::Map::new();
        for key in ["z", "b", "a", "b", "c"] {
            let value = serde_json::Value::String(format!("value-{key}"));
            assert_eq!(
                actual
                    .insert(key.into(), value.clone().into())
                    .map(|v| v.to_value()),
                expected.insert(key.into(), value)
            );
        }
        for key in ["b", "missing", "z"] {
            assert_eq!(
                actual.remove(key).map(|v| v.to_value()),
                expected.remove(key)
            );
            let value = serde_json::Value::String(format!("reinserted-{key}"));
            actual.insert(key.into(), value.clone().into());
            expected.insert(key.into(), value);
            assert_eq!(
                ParsedValue::Object(actual).json_text().unwrap(),
                serde_json::to_string(&expected).unwrap()
            );
            actual = expected
                .clone()
                .into_iter()
                .map(|(k, v)| (k, v.into()))
                .collect();
        }
    }

    #[test]
    fn literal_objects_decode_strings_and_replace_duplicate_keys() {
        let input = r#"{"z":{"b":1e-400,"a":9007199254740992.5},"a":"old","\u0061":"line\n\"quoted\"","z":{"c":4.2e1,"b":1e9999}}"#;
        let expected = serde_json::from_str::<serde_json::Value>(
            r#"{"z":{"b":"old","a":"old"},"a":"old","\u0061":"line\n\"quoted\"","z":{"c":"EXP","b":"HUGE"}}"#,
        ).unwrap();
        let expected = serde_json::to_string(&expected)
            .unwrap()
            .replace("\"EXP\"", "4.2e1")
            .replace("\"HUGE\"", "1e9999");
        assert_eq!(
            parsed_json_literal(input).unwrap().json_text().unwrap(),
            expected
        );
    }

    #[test]
    fn decimal_normalization_rejects_tokens_without_digits() {
        assert!(normalize_decimal_spelling(".").is_none());
        assert!(coerce_integral_number(".").is_none());
        assert!(normalize_decimal_spelling("+.5").is_some());
    }

    #[test]
    fn numeric_tokens_are_validated_without_binary_float_conversion() {
        for token in ["9007199254740992.5", "1e-400", "1e999999999999999999999"] {
            assert!(raw_number_literal(token).is_some(), "{token}");
        }
        for token in ["true", "null", "\"42\"", "+1", ".5", "01"] {
            assert!(raw_number_literal(token).is_none(), "{token}");
        }
        assert_eq!(
            coerce_integral_number("1e999999999999999999999")
                .expect("large positive exponent remains a number")
                .json_text()
                .unwrap(),
            "1e999999999999999999999"
        );
        assert!(coerce_integral_number("1e-999999999999999999999").is_none());
        assert_eq!(
            coerce_integral_number("0e-999999999999999999999")
                .expect("zero is integral for every exponent")
                .json_text()
                .unwrap(),
            "0"
        );
    }

    #[test]
    fn compact_json_preserves_number_tokens_and_utf8_strings() {
        let compact = parsed_json_literal(
            "{ \"city\": \"München\", \"amount\": 9007199254740992.5, \"tiny\": 1e-400 }",
        )
        .expect("valid JSON object");
        assert_eq!(
            compact.json_text().unwrap(),
            serde_json::to_string(
                &serde_json::json!({"city": "München", "amount": "EXACT", "tiny": "TINY"})
            )
            .unwrap()
            .replace("\"EXACT\"", "9007199254740992.5")
            .replace("\"TINY\"", "1e-400")
        );
    }

    #[test]
    fn sorted_json_retains_serde_nesting_limit() {
        for depth in 0..=130 {
            for (open, close) in [("[", "]"), ("{\"item\":", "}"), ("[{\"item\":", "}]")] {
                let input = format!("{}0{}", open.repeat(depth), close.repeat(depth));
                assert_eq!(
                    parsed_json_literal(&input).is_some(),
                    serde_json::from_str::<serde_json::Value>(&input).is_ok(),
                    "depth={depth}, open={open}"
                );
            }
        }
        let input = format!("{}0{}", "[".repeat(3000), "]".repeat(3000));
        assert!(parsed_json_literal(&input).is_none());
        let quoted = serde_json::to_string(&input).unwrap();
        assert_eq!(
            parsed_json_literal(&quoted).unwrap().json_text().unwrap(),
            quoted
        );
    }

    #[test]
    fn sorted_json_preserves_nested_number_tokens() {
        let value = parsed_json_literal(r#"{"z":{"b":1e-400,"a":9007199254740992.5},"a":[4.2e1]}"#)
            .expect("valid JSON object");
        assert_eq!(
            value.json_text().unwrap(),
            serde_json::to_string(
                &serde_json::from_str::<serde_json::Value>(
                    r#"{"z":{"b":"TINY","a":"EXACT"},"a":["EXP"]}"#,
                )
                .unwrap()
            )
            .unwrap()
            .replace("\"EXACT\"", "9007199254740992.5")
            .replace("\"TINY\"", "1e-400")
            .replace("\"EXP\"", "4.2e1")
        );
        assert!(parsed_json_literal(r#"{"value":NaN}"#).is_none());
    }
}
