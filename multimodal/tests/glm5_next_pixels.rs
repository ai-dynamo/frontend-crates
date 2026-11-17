// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Adapted from sgl-project/sglang PR #46563 (Apache-2.0).

//! Replay the Python CPU oracle from SGLang PR #46563 through the public registry.

use std::path::{Path, PathBuf};

use dynamo_multimodal::image::decode::DecodeLimits;
use dynamo_multimodal::processor::{MediaMetadata, ProcessedItem};
use dynamo_multimodal::processor::{PositionOutput, TensorData};
use dynamo_multimodal::{registry::processor_from_spec, token_layout};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/glm5_next/pixels")
}

fn images() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/mm_parity/images")
}

#[test]
fn fixtures_match_sglang() {
    let fixture =
        serde_json::from_str(&std::fs::read_to_string(fixtures().join("cases.json")).unwrap())
            .unwrap();
    let failures = check(&fixture);
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

fn check(fixture: &Value) -> Vec<String> {
    let model = fixture["model"].as_str().unwrap();
    let mut failures = Vec::new();
    for case in fixture["cases"].as_array().unwrap() {
        let name = format!("{model} {}", case["name"]);
        // Same contract as tests/parity.rs: a known gap must keep failing.
        match (check_case(case), case["known_gap"].as_str()) {
            (Ok(()), None) => {}
            (Err(error), None) => failures.push(format!("{name}: {error}")),
            (Err(_), Some(gap)) => eprintln!("{name}: known gap: {gap}"),
            (Ok(()), Some(_)) => {
                failures.push(format!("{name} now matches SGLang; drop its known_gap"))
            }
        }
    }
    failures
}

fn check_case(case: &Value) -> Result<(), String> {
    let family = processor_from_spec(&case["spec"].to_string()).map_err(|e| e.to_string())?;
    let images: Vec<Vec<u8>> = (case["images"].as_array().unwrap().iter())
        .map(|name| std::fs::read(images().join(name.as_str().unwrap())).unwrap())
        .collect();
    let images: Vec<&[u8]> = images.iter().map(Vec::as_slice).collect();
    let input_ids: Vec<i32> = (case["input_ids"].as_array().unwrap().iter())
        .map(|id| id.as_i64().unwrap() as i32)
        .collect();

    let output = (|| -> dynamo_multimodal::Result<MmOutput> {
        let items = images
            .iter()
            .map(|bytes| {
                let media = family.decode_image(bytes, &DecodeLimits::default())?;
                let item = family.process_item(&media)?;
                let dynamo_multimodal::processor::DecodedMedia::Image { height, width, .. } = media
                else {
                    unreachable!()
                };
                let metadata = MediaMetadata::Image {
                    width: width as u32,
                    height: height as u32,
                };
                assert_eq!(
                    item.feature_token_count,
                    family.num_media_tokens(&metadata)?
                );
                Ok(item)
            })
            .collect::<dynamo_multimodal::Result<Vec<_>>>()?;
        let counts = items
            .iter()
            .map(|i| i.feature_token_count)
            .collect::<Vec<_>>();
        let layout = family.layout(&input_ids, &items)?;
        let expanded = token_layout::apply_layout(&input_ids, &layout, &counts, usize::MAX)?;
        let positions = family.positions(expanded.input_ids.len(), &expanded.offsets, &items)?;
        Ok(MmOutput {
            input_ids: expanded.input_ids,
            items,
            offsets: expanded.offsets,
            positions,
        })
    })();
    match (output, case.get("error")) {
        (Ok(output), None) => compare(&summary(&output), &case["expected"]),
        (Err(_), Some(_)) => Ok(()),
        (Ok(_), Some(error)) => Err(format!("SGLang rejects this case: {error}")),
        (Err(error), None) => Err(error.to_string()),
    }
}

struct MmOutput {
    input_ids: Vec<i32>,
    items: Vec<ProcessedItem>,
    offsets: Vec<(u32, u32)>,
    positions: PositionOutput,
}

/// The fields `generate_mm_parity.py` records, computed from the Rust output.
fn summary(output: &MmOutput) -> Value {
    let features = output.items.iter().map(|item| match &item.feature.data {
        TensorData::F32(values) => values,
        _ => panic!("features are f32"),
    });
    let mut summary = json!({
        "input_ids_sha256": sha256(output.input_ids.iter().flat_map(|id| id.to_le_bytes())),
        "num_input_ids": output.input_ids.len(),
        "offsets": output.offsets,
        "feature_shapes": output.items.iter().map(|item| &item.feature.shape).collect::<Vec<_>>(),
        "feature_sha256": features
            .map(|values| sha256(values.iter().flat_map(|v| v.to_le_bytes())))
            .collect::<Vec<_>>(),
    });
    // Model kwargs such as `image_grid_thw`, one value per item.
    for (name, _) in output.items.first().map_or(&[][..], |item| &item.aux[..]) {
        let values: Vec<Value> = (output.items.iter())
            .map(|item| {
                let (_, tensor) = (item.aux.iter())
                    .find(|(aux, _)| aux == name)
                    .expect("every item has the same aux");
                let TensorData::I64(values) = &tensor.data else {
                    panic!("{name} is i64");
                };
                json!(values)
            })
            .collect();
        summary[name] = json!(values);
    }
    if let PositionOutput::MRope { positions, delta } = &output.positions {
        summary["mrope_sha256"] = json!(sha256(positions.iter().flat_map(|p| p.to_le_bytes())));
        summary["mrope_delta"] = json!(delta);
    }
    summary
}

fn compare(actual: &Value, expected: &Value) -> Result<(), String> {
    let mismatches: Vec<String> = (expected.as_object().unwrap().iter())
        .filter(|(field, value)| actual[field.as_str()] != **value)
        .map(|(field, value)| format!("{field}: rust {} != python {value}", actual[field.as_str()]))
        .collect();
    if mismatches.is_empty() {
        Ok(())
    } else {
        Err(mismatches.join("; "))
    }
}

fn sha256(bytes: impl Iterator<Item = u8>) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes.collect::<Vec<u8>>());
    format!("{:x}", hasher.finalize())
}

fn fixture_spec() -> Value {
    let fixture: Value =
        serde_json::from_str(&std::fs::read_to_string(fixtures().join("cases.json")).unwrap())
            .unwrap();
    fixture["cases"][0]["spec"].clone()
}

#[test]
fn invalid_pixel_specs_do_not_fall_back_to_the_planner() {
    for name in ["image_mean", "image_std"] {
        let mut spec = fixture_spec();
        spec.as_object_mut().unwrap().remove(name);
        assert!(
            processor_from_spec(&spec.to_string()).is_err(),
            "missing {name}"
        );
    }
    for value in [0.0, -1.0] {
        let mut spec = fixture_spec();
        spec["image_std"] = json!([value, 1.0, 1.0]);
        assert!(processor_from_spec(&spec.to_string()).is_err());
    }
}

#[test]
fn pil_refuses_patch_expansion_that_disagrees_with_the_shared_planner() {
    let mut spec = fixture_spec();
    spec["resample"] = json!("pil");
    spec["patch_expand_factor"] = json!(2);
    spec["min_image_tokens"] = json!(4);
    spec["max_image_tokens"] = json!(32);
    // For a 200x150 image, pinned HF PIL ignores the expansion factor and
    // produces grid [1, 10, 12] (30 tokens); the shared planner produces
    // [1, 12, 16] (48 tokens). Refuse the spec before either path is used.
    assert!(matches!(
        processor_from_spec(&spec.to_string()),
        Err(dynamo_multimodal::MmError::Unsupported { .. })
    ));
}

#[test]
fn supported_resamplers_keep_pixel_and_metadata_geometry_in_sync() {
    use dynamo_multimodal::processor::{DecodedMedia, Geometry};

    for (resample, expand, grid, tokens) in
        [("pil", 1, [1, 10, 12], 30), ("aten_u8", 2, [1, 12, 16], 48)]
    {
        let mut spec = fixture_spec();
        spec["resample"] = json!(resample);
        spec["patch_expand_factor"] = json!(expand);
        spec["min_image_tokens"] = json!(4);
        spec["max_image_tokens"] = json!(32);
        let family = processor_from_spec(&spec.to_string()).unwrap();
        let item = family
            .process_item(&DecodedMedia::Image {
                rgb: vec![42; 200 * 150 * 3],
                height: 150,
                width: 200,
            })
            .unwrap();
        let Some(Geometry::Grid(actual_grid)) = item.geometry else {
            panic!("expected an image grid");
        };
        assert_eq!(actual_grid, grid, "{resample}");
        assert_eq!(item.feature_token_count, tokens, "{resample}");
        assert_eq!(
            family
                .num_media_tokens(&MediaMetadata::Image {
                    width: 200,
                    height: 150,
                })
                .unwrap(),
            tokens,
            "{resample}"
        );
    }
}

#[test]
fn full_pipeline_rejects_bad_buffers_video_and_mismatched_offsets() {
    use dynamo_multimodal::processor::DecodedMedia;
    let family = processor_from_spec(&fixture_spec().to_string()).unwrap();
    for (height, width, rgb) in [(0, 1, vec![]), (1, 0, vec![]), (2, 2, vec![0; 11])] {
        assert!(
            family
                .process_item(&DecodedMedia::Image { rgb, height, width })
                .is_err()
        );
    }
    let item = family
        .process_item(&DecodedMedia::Image {
            rgb: vec![42; 3],
            height: 1,
            width: 1,
        })
        .unwrap();
    assert!(family.layout(&[154855, 154854], &[item]).is_err());
    assert!(family.positions(1, &[(0, 0)], &[]).is_err());
}

#[test]
fn smart_decode_checks_transposed_limits_before_decoding() {
    let family = processor_from_spec(&fixture_spec().to_string()).unwrap();
    for (name, oriented_height) in [
        ("exif_orientation6_64x48.jpg", 64),
        ("rgba_raw_exif6_56x48.png", 56),
    ] {
        let bytes = std::fs::read(images().join(name)).unwrap();
        let limits = DecodeLimits {
            max_width: Some(48),
            max_height: Some(oriented_height),
            max_pixels: Some(48 * oriented_height),
        };
        assert!(family.decode_image(&bytes, &limits).is_ok(), "{name}");
        let limits = DecodeLimits {
            max_height: Some(oriented_height - 1),
            ..limits
        };
        assert!(
            matches!(
                family.decode_image(&bytes, &limits),
                Err(dynamo_multimodal::MmError::LimitExceeded { .. })
            ),
            "{name}"
        );
    }
}
