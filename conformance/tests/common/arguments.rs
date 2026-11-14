// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;

pub fn canonical_argument_json(raw: &str) -> String {
    match raw.as_bytes().first() {
        Some(b'{') => {
            let fields: BTreeMap<String, Box<serde_json::value::RawValue>> =
                serde_json::from_str(raw).expect("validated object arguments");
            let fields = fields
                .into_iter()
                .map(|(key, value)| {
                    format!(
                        "{}={}",
                        serde_json::to_string(&key).expect("serialize object key"),
                        canonical_argument_json(value.get()),
                    )
                })
                .collect::<Vec<_>>();
            format!("object:[{}]", fields.join(","))
        }
        Some(b'[') => {
            let items: Vec<Box<serde_json::value::RawValue>> =
                serde_json::from_str(raw).expect("validated array arguments");
            let items = items
                .iter()
                .map(|item| canonical_argument_json(item.get()))
                .collect::<Vec<_>>();
            format!("array:[{}]", items.join(","))
        }
        Some(b'"') => {
            let value: String = serde_json::from_str(raw).expect("validated string argument");
            format!(
                "string:{}",
                serde_json::to_string(&value).expect("serialize string")
            )
        }
        Some(b't') => "boolean:true".to_string(),
        Some(b'f') => "boolean:false".to_string(),
        Some(b'n') => "null".to_string(),
        Some(b'-' | b'0'..=b'9') => canonical_argument_number(raw),
        _ => unreachable!("RawValue only accepts JSON"),
    }
}

fn canonical_argument_number(raw: &str) -> String {
    let (negative, unsigned) = match raw.strip_prefix('-') {
        Some(unsigned) => (true, unsigned),
        None => (false, raw),
    };
    let (mantissa, exponent) = match unsigned.split_once(['e', 'E']) {
        Some((mantissa, exponent)) => match exponent.parse::<i64>() {
            Ok(exponent) => (mantissa, exponent),
            Err(_) => return format!("number-token:{raw}"),
        },
        None => (unsigned, 0),
    };
    let (whole, fraction) = mantissa.split_once('.').unwrap_or((mantissa, ""));
    let digits = format!("{whole}{fraction}");
    let digits = digits.trim_start_matches('0');
    if digits.is_empty() {
        return "number:0".to_string();
    }
    let trailing_zeroes = digits.len() - digits.trim_end_matches('0').len();
    let significant = digits.trim_end_matches('0');
    let adjustment = match i64::try_from(trailing_zeroes)
        .ok()
        .zip(i64::try_from(fraction.len()).ok())
        .and_then(|(trailing, fractional)| trailing.checked_sub(fractional))
    {
        Some(adjustment) => adjustment,
        None => return format!("number-token:{raw}"),
    };
    let Some(exponent) = exponent.checked_add(adjustment) else {
        return format!("number-token:{raw}");
    };
    format!(
        "number:{}:{}:{exponent}",
        if negative { "-" } else { "+" },
        significant
    )
}

pub fn captured_arguments(projected: &serde_json::Value, raw: &str) -> String {
    if serde_json::value::RawValue::from_string(raw.to_owned()).is_ok()
        && raw_matches_projection(projected, raw.trim())
    {
        raw.to_owned()
    } else {
        serde_json::to_string(projected).expect("assembled arguments serialize")
    }
}

pub fn raw_matches_projection(projected: &serde_json::Value, raw: &str) -> bool {
    use serde_json::Value;
    use serde_json::value::RawValue;
    match (raw.as_bytes().first(), projected) {
        (Some(b'{'), Value::Object(actual)) => {
            let fields: std::collections::BTreeMap<String, Box<RawValue>> =
                serde_json::from_str(raw).expect("validated object arguments");
            actual.len() == fields.len()
                && fields.iter().all(|(key, value)| {
                    actual
                        .get(key)
                        .is_some_and(|actual| raw_matches_projection(actual, value.get()))
                })
        }
        (Some(b'['), Value::Array(actual)) => {
            let items: Vec<Box<RawValue>> =
                serde_json::from_str(raw).expect("validated array arguments");
            actual.len() == items.len()
                && actual
                    .iter()
                    .zip(items.iter())
                    .all(|(actual, value)| raw_matches_projection(actual, value.get()))
        }
        (Some(b'-' | b'0'..=b'9'), _) => match serde_json::from_str::<Value>(raw) {
            Ok(value) => value == *projected,
            // Raw reconstruction may restore an overflow token, but never an
            // object or sibling field discarded by the typed assembler.
            Err(_) => projected.as_str() == Some(raw),
        },
        _ => serde_json::from_str::<Value>(raw).is_ok_and(|value| value == *projected),
    }
}
