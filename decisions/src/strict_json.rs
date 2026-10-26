// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde::de::{self, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer};
use serde_json::{Map, Number, Value};
use std::fmt;

/// Recursive duplicate-key detection must precede deserialization into a Value.
struct UniqueValue(Value);

impl<'de> Deserialize<'de> for UniqueValue {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_any(UniqueVisitor)
    }
}

struct UniqueVisitor;
impl<'de> Visitor<'de> for UniqueVisitor {
    type Value = UniqueValue;
    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("JSON with unique object keys")
    }
    fn visit_bool<E: de::Error>(self, v: bool) -> Result<Self::Value, E> {
        Ok(UniqueValue(Value::Bool(v)))
    }
    fn visit_i64<E: de::Error>(self, v: i64) -> Result<Self::Value, E> {
        Ok(UniqueValue(v.into()))
    }
    fn visit_u64<E: de::Error>(self, v: u64) -> Result<Self::Value, E> {
        Ok(UniqueValue(v.into()))
    }
    fn visit_f64<E: de::Error>(self, v: f64) -> Result<Self::Value, E> {
        Number::from_f64(v)
            .map(|v| UniqueValue(Value::Number(v)))
            .ok_or_else(|| E::custom("nonfinite number"))
    }
    fn visit_str<E: de::Error>(self, v: &str) -> Result<Self::Value, E> {
        Ok(UniqueValue(Value::String(v.into())))
    }
    fn visit_string<E: de::Error>(self, v: String) -> Result<Self::Value, E> {
        Ok(UniqueValue(Value::String(v)))
    }
    fn visit_unit<E: de::Error>(self) -> Result<Self::Value, E> {
        Ok(UniqueValue(Value::Null))
    }
    fn visit_none<E: de::Error>(self) -> Result<Self::Value, E> {
        Ok(UniqueValue(Value::Null))
    }
    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
        let mut values = Vec::new();
        while let Some(value) = seq.next_element::<UniqueValue>()? {
            values.push(value.0);
        }
        Ok(UniqueValue(Value::Array(values)))
    }
    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
        let mut values = Map::new();
        while let Some(key) = map.next_key::<String>()? {
            if values.contains_key(&key) {
                return Err(de::Error::custom("duplicate object key"));
            }
            values.insert(key, map.next_value::<UniqueValue>()?.0);
        }
        Ok(UniqueValue(Value::Object(values)))
    }
}

pub(crate) fn parse(bytes: &[u8]) -> Result<Value, serde_json::Error> {
    // The duplicate-key visitor validates structure only: arbitrary_precision
    // represents decimals as synthetic maps to custom visitors. Let Value's
    // own deserializer recover numbers after validation rather than exposing
    // that implementation detail as request data.
    serde_json::from_slice::<UniqueValue>(bytes)?;
    serde_json::from_slice(bytes)
}

#[cfg(test)]
mod tests {
    use super::parse;

    #[test]
    fn numbers_keep_their_types_and_duplicate_keys_stay_rejected() {
        for text in ["1.25", "1e-7", "9007199254740993", "[1.25, {\"n\": 1e-7}]"] {
            let expected: serde_json::Value = serde_json::from_str(text).unwrap();
            assert_eq!(parse(text.as_bytes()).unwrap(), expected, "{text}");
        }
        assert!(parse(br#"{"n":1.25,"n":2.5}"#).is_err());
        assert!(parse(br#"{"nested":{"n":1.25,"n":2.5}}"#).is_err());
    }
}
