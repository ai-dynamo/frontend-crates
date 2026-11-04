// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::normalization::{Result, Value};
use serde_yaml::{Mapping, Value as Yaml};
use sha2::{Digest, Sha256};

impl Value for Yaml {
    fn object() -> Self {
        Self::Mapping(Mapping::new())
    }
    fn text(value: &str) -> Self {
        Self::String(value.into())
    }
    fn integer(value: u64) -> Self {
        Self::Number(value.into())
    }
    fn array(values: Vec<Self>) -> Self {
        Self::Sequence(values)
    }
    fn null() -> Self {
        Self::Null
    }
    fn is_null(&self) -> bool {
        Yaml::is_null(self)
    }
    fn as_integer(&self) -> Option<u64> {
        self.as_u64()
    }
    fn as_bool(&self) -> Option<bool> {
        Yaml::as_bool(self)
    }
    fn as_text(&self) -> Result<&str> {
        self.as_str().ok_or_else(|| "expected a string".into())
    }
    fn entries(&self) -> Result<Vec<(&str, &Self)>> {
        self.as_mapping()
            .ok_or("expected a mapping")?
            .iter()
            .map(|(key, value)| Ok((key.as_text()?, value)))
            .collect()
    }
    fn items(&self) -> Result<&[Self]> {
        self.as_sequence()
            .map(Vec::as_slice)
            .ok_or_else(|| "expected an array".into())
    }
    fn get(&self, name: &str) -> Option<&Self> {
        Yaml::get(self, name)
    }
    fn set(&mut self, name: &str, value: Self) -> Result<()> {
        self.as_mapping_mut()
            .ok_or("expected a mapping")?
            .insert(Self::text(name), value);
        Ok(())
    }
    fn digest(&self) -> Result<String> {
        Ok(format!(
            "{:x}",
            Sha256::digest(request_json(self)?.as_bytes())
        ))
    }
}

fn request_json(value: &Yaml) -> Result<String> {
    Ok(match value {
        Yaml::Null => "null".into(),
        Yaml::Bool(value) => value.to_string(),
        Yaml::Number(value) if value.is_f64() => {
            let number = value.as_f64().ok_or("invalid floating point number")?;
            if number.is_nan() {
                "NaN".into()
            } else if number.is_infinite() {
                if number.is_sign_negative() {
                    "-Infinity"
                } else {
                    "Infinity"
                }
                .into()
            } else {
                let repr = format!("{number:?}");
                if let Some((mantissa, exponent)) = repr.split_once('e') {
                    let exponent: i32 = exponent.parse().map_err(|_| "invalid exponent")?;
                    format!("{mantissa}e{exponent:+03}")
                } else {
                    repr
                }
            }
        }
        Yaml::Number(value) => value.to_string(),
        Yaml::String(value) => serde_json::to_string(value).map_err(|e| e.to_string())?,
        Yaml::Sequence(values) => format!(
            "[{}]",
            values
                .iter()
                .map(request_json)
                .collect::<Result<Vec<_>>>()?
                .join(",")
        ),
        Yaml::Mapping(values) => {
            let sorted = values
                .iter()
                .map(|(key, value)| Ok((key.as_text()?, value)))
                .collect::<Result<std::collections::BTreeMap<_, _>>>()?;
            format!(
                "{{{}}}",
                sorted
                    .into_iter()
                    .map(|(key, value)| {
                        Ok(format!(
                            "{}:{}",
                            serde_json::to_string(key).map_err(|e| e.to_string())?,
                            request_json(value)?
                        ))
                    })
                    .collect::<Result<Vec<_>>>()?
                    .join(",")
            )
        }
        Yaml::Tagged(_) => return Err("non JSON request value".into()),
    })
}
