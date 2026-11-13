// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Deterministic publication bytes and bounded gzip validation.

use anyhow::{Result, ensure};
use flate2::{Compression, GzBuilder, bufread::GzDecoder};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    io::{Read, Write},
};

/// Compact UTF-8 JSON always ends in exactly one terminal newline.
pub fn json_bytes(value: &Value) -> Result<Vec<u8>> {
    let mut bytes = serde_json::to_vec(value)?;
    bytes.push(b'\n');
    Ok(bytes)
}

/// Stable identity bytes ignore JSON object insertion order and escape Unicode.
pub fn canonical_json_bytes(value: &Value) -> Result<Vec<u8>> {
    fn sorted(value: &Value) -> Value {
        match value {
            Value::Object(map) => Value::Object(
                map.iter()
                    .map(|(k, v)| (k.clone(), sorted(v)))
                    .collect::<BTreeMap<_, _>>()
                    .into_iter()
                    .collect(),
            ),
            Value::Array(values) => Value::Array(values.iter().map(sorted).collect()),
            other => other.clone(),
        }
    }
    let text = serde_json::to_string(&sorted(value))?;
    let mut escaped = String::new();
    for ch in text.chars() {
        if ch.is_ascii() {
            escaped.push(ch);
        } else {
            for unit in ch.encode_utf16(&mut [0; 2]) {
                escaped.push_str(&format!("\\u{unit:04x}"));
            }
        }
    }
    Ok(escaped.into_bytes())
}

/// Hash the exact bytes represented by a resource descriptor.
pub fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// Use an explicit timestamp and OS byte, independent of host wall time.
pub fn gzip(bytes: &[u8]) -> Result<Vec<u8>> {
    gzip_with_level(bytes, 9)
}

/// Checkout resources trade compression time for size; production keeps level 9.
pub fn gzip_with_level(bytes: &[u8], level: u32) -> Result<Vec<u8>> {
    let mut encoder = GzBuilder::new()
        .mtime(0)
        .operating_system(3)
        .write(Vec::new(), Compression::new(level));
    encoder.write_all(bytes)?;
    Ok(encoder.finish()?)
}

/// Validate a compressed resource and recover at most the admitted byte count.
pub fn gunzip(bytes: &[u8], expected_sha256: &str, maximum_decoded: usize) -> Result<Vec<u8>> {
    ensure!(
        sha256(bytes) == expected_sha256,
        "compressed resource hash mismatch"
    );
    let mut decoded = Vec::new();
    let mut decoder = GzDecoder::new(bytes);
    (&mut decoder)
        .take(
            u64::try_from(maximum_decoded)?
                .checked_add(1)
                .ok_or_else(|| anyhow::anyhow!("decoded size overflow"))?,
        )
        .read_to_end(&mut decoded)?;
    ensure!(
        decoded.len() <= maximum_decoded,
        "decoded resource exceeds admitted size"
    );
    ensure!(
        decoder.get_ref().is_empty(),
        "trailing gzip bytes or members"
    );
    Ok(decoded)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn exact_json_newline_and_deterministic_gzip() -> Result<()> {
        let bytes = json_bytes(&json!({"value":["str","sample"]}))?;
        assert_eq!(bytes, b"{\"value\":[\"str\",\"sample\"]}\n");
        let first = gzip(&bytes)?;
        let second = gzip(&bytes)?;
        assert_eq!(first, second);
        assert_eq!(&first[4..8], &[0; 4]);
        assert_eq!(gunzip(&first, &sha256(&first), bytes.len())?, bytes);
        assert!(gunzip(&first, &sha256(&first), bytes.len() - 1).is_err());
        Ok(())
    }

    #[test]
    fn gzip_rejects_trailing_bytes_and_additional_members() -> Result<()> {
        let source = b"hello";
        let envelope = gzip(source)?;
        for trailer in [
            b"garbage".to_vec(),
            gzip(b"not-json")?,
            gzip(&vec![0; 4096])?,
        ] {
            let mut corrupt = envelope.clone();
            corrupt.extend(trailer);
            assert!(gunzip(&corrupt, &sha256(&corrupt), source.len()).is_err());
        }
        Ok(())
    }
}
