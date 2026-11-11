// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Adapted from sgl-project/sglang `rust/sglang-processor/tests/mm_parity.rs` (Apache-2.0).

//! Every model table in `tests/fixtures/mm_parity/` replays through the public
//! API as its `target` framework ran, bit for bit. `generate.py` there records
//! the tables; its README has the cell contract. With `MM_PARITY_RECORD_CRATE`
//! set to a directory, the test writes the crate's column there for
//! `generate.py --crate`.

mod common;

use std::path::PathBuf;

use dynamo_multimodal::image::decode::DecodeLimits;
use dynamo_multimodal::processor::{
    DecodedMedia, MediaMetadata, PositionOutput, ProcessedItem, TensorData,
};
use dynamo_multimodal::token_layout::{self, ExpandedPrompt};
use dynamo_multimodal::{MmError, registry::processor_from_spec};
use serde_json::{Value, json};

const CRATE: &str = "dynamo-multimodal";

#[test]
fn tables_match_their_reference() {
    let tables = tables();
    assert!(!tables.is_empty());
    let record = std::env::var_os("MM_PARITY_RECORD_CRATE").map(PathBuf::from);
    let mut failures = Vec::new();
    for path in &tables {
        let (table_failures, cells) = check(&common::read_json(path), record.is_none());
        failures.extend(table_failures);
        if let Some(dir) = &record {
            let column = json!({"version": env!("CARGO_PKG_VERSION"), "cases": cells});
            std::fs::write(dir.join(path.file_name().unwrap()), column.to_string()).unwrap();
        }
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

/// Every `*.json` but `decode.json`, the decode stage table.
fn tables() -> Vec<PathBuf> {
    let mut tables: Vec<PathBuf> = (std::fs::read_dir(common::mm_parity()).unwrap())
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "json"))
        .filter(|path| !path.ends_with("decode.json"))
        .collect();
    tables.sort();
    tables
}

/// The failures, and the crate's cell for each case.
fn check(fixture: &Value, check_cells: bool) -> (Vec<String>, serde_json::Map<String, Value>) {
    let model = fixture["model"].as_str().unwrap();
    let cases = fixture["cases"].as_array().unwrap();
    assert!(!cases.is_empty(), "{model}: no cases");
    let mut failures = Vec::new();
    let mut cells = serde_json::Map::new();
    for case in cases {
        let name = format!("{model} {}", case["name"]);
        let outcome = replay(fixture, case);
        let cell = crate_cell(fixture, case, &outcome);
        if check_cells && case["frameworks"][CRATE] != cell {
            failures.push(format!(
                "{name}: the crate gives {cell}; run generate.py --crate"
            ));
        }
        cells.insert(case["name"].as_str().unwrap().to_string(), cell);
        let case = match target(fixture, case) {
            Ok(case) => case,
            Err(failure) => {
                failures.push(format!("{name}: {failure}"));
                continue;
            }
        };
        // `rust_error` is the kind of the crate's refusal: required when the
        // reference refuses too, optional on a known gap only the crate refuses.
        let kind = &case["rust_error"];
        let refusal = match &outcome {
            Ok(_) if kind.is_string() => Err(format!("accepts; the cell records a {kind} refusal")),
            Err(error)
                if (kind.is_string() || case.get("error").is_some())
                    && *kind != error_kind(error) =>
            {
                Err(format!("refuses with {error}; the cell records {kind}"))
            }
            _ => Ok(()),
        };
        if let Err(failure) = refusal {
            failures.push(format!("{name}: {failure}"));
            continue;
        }
        match (matches(&case, outcome), case["known_gap"].as_str()) {
            (Ok(()), None) => {}
            (Err(error), None) => failures.push(format!("{name}: {error}")),
            (Err(error), Some(gap)) => eprintln!("{name}: known gap: {gap} ({error})"),
            (Ok(()), Some(_)) => failures.push(format!("{name} now matches; drop its known_gap")),
        }
    }
    (failures, cells)
}

/// The case as the table's `target` framework ran it: `expected` with the
/// target's differing fields, or the target's refusal.
fn target(fixture: &Value, case: &Value) -> Result<Value, String> {
    let target = fixture["target"].as_str().unwrap();
    if target == "hf" {
        return Ok(case.clone());
    }
    let cell = &case["frameworks"][target];
    let mut case = case.clone();
    if let Some(error) = cell.get("rejects") {
        case.as_object_mut().unwrap().remove("expected");
        case["error"] = error.clone();
    } else if let Some(differs) = cell["differs"].as_array() {
        if case.as_object_mut().unwrap().remove("error").is_some() {
            // The baseline refuses and the target accepts: the cell holds every field.
            let mut expected = cell.clone();
            expected.as_object_mut().unwrap().remove("differs");
            case["expected"] = expected;
        } else {
            for field in differs.iter().map(|field| field.as_str().unwrap()) {
                case["expected"][field] = cell[field].clone();
            }
        }
    } else if cell.get("=").is_none() && cell.get("omits").is_none() {
        return Err(format!("the target {target} has no result: {cell}"));
    }
    Ok(case)
}

/// The crate's cell against the HF baseline, in the form `generate.py` gives
/// every framework's. When the baseline refuses, it holds the target cell's fields.
fn crate_cell(fixture: &Value, case: &Value, outcome: &dynamo_multimodal::Result<Value>) -> Value {
    let actual = match outcome {
        Err(error) => return json!({"rejects": error_kind(error)}),
        Ok(actual) => actual,
    };
    let Some(expected) = case.get("expected") else {
        let mut cell = json!({"differs": ["error"]});
        let target = &case["frameworks"][fixture["target"].as_str().unwrap()];
        for field in target.as_object().into_iter().flat_map(|cell| cell.keys()) {
            if field != "differs" {
                cell[field] = actual[field].clone();
            }
        }
        return cell;
    };
    let (mut differs, mut omits) = (Vec::new(), Vec::new());
    for (field, value) in expected.as_object().unwrap() {
        match actual.get(field) {
            None => omits.push(field),
            Some(got) if got != value => differs.push(field),
            Some(_) => {}
        }
    }
    // `generate.py` sorts both. A workspace build turns on serde_json's
    // `preserve_order`, which iterates `expected` in the table's order.
    differs.sort();
    omits.sort();
    let mut cell = json!({});
    if !differs.is_empty() {
        cell["differs"] = json!(differs);
    }
    for field in differs {
        cell[field] = actual[field].clone();
    }
    if !omits.is_empty() {
        cell["omits"] = json!(omits);
    }
    if cell.as_object().unwrap().is_empty() {
        return json!({"=": true});
    }
    cell
}

/// `Ok` when the crate gives what the reference gave: the expected fields, or a refusal.
fn matches(case: &Value, outcome: dynamo_multimodal::Result<Value>) -> Result<(), String> {
    match (outcome, case.get("expected")) {
        (Ok(actual), Some(expected)) => compare(&actual, expected),
        (Err(error), Some(_)) => Err(error.to_string()),
        (Ok(_), None) => Err(format!("the reference rejects it: {}", case["error"])),
        (Err(_), None) => Ok(()),
    }
}

/// The fields `generate.py` records, computed through the public API.
fn replay(fixture: &Value, case: &Value) -> dynamo_multimodal::Result<Value> {
    // `spec_shared` holds the spec fields every case has, such as a token table.
    let mut spec = fixture.get("spec_shared").cloned().unwrap_or(json!({}));
    (spec.as_object_mut().unwrap()).extend(case["spec"].as_object().unwrap().clone());
    let family = processor_from_spec(&spec.to_string())?;
    let input_ids: Vec<i32> = (case["input_ids"].as_array().unwrap().iter())
        .map(|id| id.as_i64().unwrap() as i32)
        .collect();

    let media = (case["images"].as_array().unwrap().iter())
        .map(|name| {
            let bytes = common::image(name.as_str().unwrap());
            family.decode_image(&bytes, &DecodeLimits::default())
        })
        .collect::<dynamo_multimodal::Result<Vec<_>>>()?;
    let items = family.process_items(&input_ids, &media)?;
    let layout = family.layout(&input_ids, &items)?;
    let features: Vec<usize> = items.iter().map(|item| item.feature_token_count).collect();
    let expanded = token_layout::apply_layout(&input_ids, &layout, &features, usize::MAX)?;
    for (media, (features, literals)) in media.iter().zip(common::layout_counts(&layout)) {
        let (DecodedMedia::Image { height, width, .. }
        | DecodedMedia::ImageRgba { height, width, .. }) = media
        else {
            unreachable!("the tables hold images")
        };
        let metadata = MediaMetadata::Image {
            width: *width as u32,
            height: *height as u32,
        };
        // A router counts an item's whole span from its metadata alone.
        let tokens = family.num_media_tokens(&metadata)?;
        if tokens != features + literals {
            return Err(MmError::internal(format!(
                "num_media_tokens {tokens} != the item's {features} features + {literals} literals"
            )));
        }
    }
    let positions = family.positions(expanded.input_ids.len(), &expanded.offsets, &items)?;
    Ok(summary(&expanded, &items, &positions))
}

fn summary(
    expanded: &ExpandedPrompt,
    items: &[ProcessedItem],
    positions: &PositionOutput,
) -> Value {
    // Each item's inclusive feature spans; `offsets` when every item has one.
    let item_offsets: Vec<Vec<[u32; 2]>> = (expanded.feature_ranges.iter())
        .map(|ranges| {
            ranges
                .iter()
                .map(|range| [range.start, range.end - 1])
                .collect()
        })
        .collect();
    let offsets: Option<Vec<[u32; 2]>> = (item_offsets.iter())
        .map(|spans| match spans[..] {
            [span] => Some(span),
            _ => None,
        })
        .collect();
    let (feature_dtypes, feature_sha256): (Vec<_>, Vec<_>) = (items.iter())
        .map(|item| match &item.feature.data {
            TensorData::F32(values) => (
                "float32",
                common::sha256_le(values.iter().map(|v| v.to_le_bytes())),
            ),
            _ => panic!("a feature dtype the tables do not record"),
        })
        .unzip();
    let mut summary = json!({
        "input_ids_sha256": common::sha256_le(expanded.input_ids.iter().map(|id| id.to_le_bytes())),
        "num_input_ids": expanded.input_ids.len(),
        "offsets": offsets,
        "item_offsets": item_offsets,
        "feature_shapes": items.iter().map(|item| &item.feature.shape).collect::<Vec<_>>(),
        "feature_dtypes": feature_dtypes,
        "feature_sha256": feature_sha256,
        "mrope_sha256": null,
        "mrope_delta": null,
    });
    // Model kwargs such as `image_grid_thw`, by name: one value per item.
    for (name, _) in items.first().map_or(&[][..], |item| &item.aux[..]) {
        summary[name] = (items.iter())
            .map(|item| {
                let (_, tensor) = (item.aux.iter())
                    .find(|(aux, _)| aux == name)
                    .expect("every item has the same kwargs");
                let TensorData::I64(values) = &tensor.data else {
                    panic!("{name} is i64");
                };
                json!(values)
            })
            .collect();
    }
    if let PositionOutput::MRope { positions, delta } = positions {
        summary["mrope_sha256"] =
            json!(common::sha256_le(positions.iter().map(|p| p.to_le_bytes())));
        summary["mrope_delta"] = json!(delta);
    }
    summary
}

/// The fields the table records; the summary may hold more.
fn compare(actual: &Value, expected: &Value) -> Result<(), String> {
    let mismatches: Vec<String> = (expected.as_object().unwrap().iter())
        .filter(|(field, value)| actual[field.as_str()] != **value)
        .map(|(field, value)| {
            format!(
                "{field}: rust {} != reference {value}",
                actual[field.as_str()]
            )
        })
        .collect();
    if mismatches.is_empty() {
        Ok(())
    } else {
        Err(mismatches.join("; "))
    }
}

fn error_kind(error: &MmError) -> &'static str {
    match error {
        MmError::Unsupported { .. } => "Unsupported",
        MmError::InvalidInput { .. } => "InvalidInput",
        MmError::LimitExceeded { .. } => "LimitExceeded",
        MmError::Internal { .. } => "Internal",
        _ => panic!("an MmError kind the tables do not record: {error}"),
    }
}
