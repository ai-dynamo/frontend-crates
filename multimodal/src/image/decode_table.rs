// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Replay `tests/fixtures/mm_parity/decode.json`, which `corpus.py` records
//! from Pillow: each corpus image through each decode entry point, and the
//! RGBA resize rows.

use super::decode::{
    DecodeLimits, decode_native, decode_rgb, decode_rgb_exif_transposed, decode_rgb_smart,
};
use super::resize::resize_rgba;
use crate::{MmError, Result};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

const COLUMNS: [&str; 4] = [
    "decode_rgb",
    "decode_rgb_exif_transposed",
    "decode_rgb_smart",
    "decode_native",
];

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/mm_parity")
}

fn table() -> Value {
    serde_json::from_str(&std::fs::read_to_string(fixtures().join("decode.json")).unwrap()).unwrap()
}

fn image(name: &str) -> Vec<u8> {
    std::fs::read(fixtures().join("images").join(name)).unwrap()
}

fn kind(error: &MmError) -> &'static str {
    match error {
        MmError::Unsupported { .. } => "Unsupported",
        MmError::InvalidInput { .. } => "InvalidInput",
        MmError::LimitExceeded { .. } => "LimitExceeded",
        _ => "Internal",
    }
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn pixels(key: &str, pixels: &[u8], height: usize, width: usize) -> Value {
    json!({key: sha256(pixels), "width": width, "height": height})
}

/// `NativeImage::exact_resize` as its doc states it, from the PIL mode: an L
/// source decodes to RGBA only with a `transparency` entry.
fn exact_resize(mode: &str, alpha: bool) -> bool {
    match mode {
        "P" | "1" | "CMYK" => false,
        "L" => !alpha,
        _ => true,
    }
}

/// The fields `corpus.py` records from Pillow, computed from the crate's
/// output, with `exact_resize` in place of the PIL mode.
fn decode(column: &str, data: &[u8]) -> Result<Value> {
    let limits = DecodeLimits::default();
    let rgb =
        |(rgb, height, width): (Vec<u8>, usize, usize)| pixels("rgb_sha256", &rgb, height, width);
    Ok(match column {
        "decode_rgb" => rgb(decode_rgb(data, &limits)?),
        "decode_rgb_exif_transposed" => rgb(decode_rgb_exif_transposed(data, &limits)?),
        "decode_rgb_smart" => rgb(decode_rgb_smart(data, &limits)?),
        "decode_native" => {
            let image = decode_native(data, &limits)?;
            let key = if image.alpha {
                "rgba_sha256"
            } else {
                "rgb_sha256"
            };
            let mut cell = pixels(key, &image.pixels, image.height, image.width);
            cell["exact_resize"] = image.exact_resize.into();
            cell
        }
        other => panic!("unknown column {other}"),
    })
}

/// A cell holds Pillow's output, the error kind of a refusal, or Pillow's
/// output with a `diverges` reason: the crate must keep decoding other pixels
/// at Pillow's size.
fn check(cell: &Value, actual: Result<Value>) -> std::result::Result<(), String> {
    let mut pillow = cell.clone();
    let fields = pillow.as_object_mut().unwrap();
    let diverges = fields.remove("diverges");
    if let Some(mode) = fields.remove("mode") {
        let alpha = fields.contains_key("rgba_sha256");
        fields.insert(
            "exact_resize".into(),
            exact_resize(mode.as_str().unwrap(), alpha).into(),
        );
    }
    match (actual, cell["rejected"].as_str()) {
        (Err(error), Some(expected)) if kind(&error) == expected => Ok(()),
        (Err(error), expected) => Err(format!(
            "expected {}, got {error}",
            expected.unwrap_or("pixels")
        )),
        (Ok(actual), Some(expected)) => Err(format!("expected {expected}, decoded {actual}")),
        (Ok(actual), None) => match (diverges, actual == pillow) {
            (None, true) => Ok(()),
            (None, false) => Err(format!("{actual} differs from Pillow's {pillow}")),
            (Some(_), true) => Err("now matches Pillow; drop its diverges".into()),
            (Some(_), false)
                if actual["width"] != pillow["width"] || actual["height"] != pillow["height"] =>
            {
                Err(format!("{actual} is not Pillow's size {pillow}"))
            }
            (Some(_), false) => Ok(()),
        },
    }
}

#[test]
fn decode_entry_points_match_pillow() {
    let table = table();
    let rows = table["images"].as_object().unwrap();
    let files: BTreeSet<String> = std::fs::read_dir(fixtures().join("images"))
        .unwrap()
        .map(|entry| entry.unwrap().file_name().into_string().unwrap())
        .collect();
    assert_eq!(
        files,
        rows.keys().cloned().collect(),
        "a row per corpus image"
    );
    let mut failures = Vec::new();
    for (name, row) in rows {
        let data = image(name);
        let row = row.as_object().unwrap();
        assert_eq!(
            row.keys().map(String::as_str).collect::<BTreeSet<_>>(),
            BTreeSet::from(COLUMNS),
            "{name}"
        );
        for (column, cell) in row {
            if let Err(error) = check(cell, decode(column, &data)) {
                failures.push(format!("{name} {column}: {error}"));
            }
        }
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn rgba_resize_matches_pillow() {
    let table = table();
    let rows = table["resize_rgba"].as_array().unwrap();
    assert!(!rows.is_empty());
    for row in rows {
        let data = image(row["image"].as_str().unwrap());
        let source = decode_native(&data, &DecodeLimits::default()).unwrap();
        assert!(source.alpha, "{row}");
        let size = |i: usize| row["size"][i].as_u64().unwrap() as usize;
        let rgba = resize_rgba(
            &source.pixels,
            source.height,
            source.width,
            size(1),
            size(0),
        );
        assert_eq!(sha256(&rgba), row["rgba_sha256"], "{row}");
    }
}
