// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What the parity table `tests/fixtures/mm_parity/kimi-k3.json` cannot hold:
//! the prompt expansion from sizes alone, two drawn images (7300x64 and
//! 9000x4000) at the checkpoint's limits, both checkpoints' configs, and a
//! size sweep.
//! `tests/fixtures/kimi_k3/generate.py` records them from SGLang's Python CPU
//! path and the checkpoint's own processor, and stops on a difference.

mod common;

use std::path::Path;

use dynamo_multimodal::image::decode::DecodeLimits;
use dynamo_multimodal::processor::{DecodedMedia, MediaMetadata, ProcessedItem, TensorData};
use dynamo_multimodal::registry::{ProcessorSpec, processor_from_spec, spec_from_hf_configs};
use dynamo_multimodal::{MmError, token_layout};
use serde_json::{Value, json};

fn fixture(name: &str) -> Value {
    common::read_json(
        &Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/kimi_k3")
            .join(name),
    )
}

/// The spec with the table's token table for the size header.
fn with_size_tokens(spec: &Value) -> Value {
    let mut spec = spec.clone();
    spec["size_tokens"] = common::table("kimi-k3.json")["spec_shared"]["size_tokens"].clone();
    spec
}

#[test]
fn layouts_and_draws_match_sglang_cpu_path() {
    check(false);
}

/// 9000x4000 at the checkpoint's limits: about 30 s in an unoptimized build.
#[test]
#[ignore = "checkpoint-scale input; run with --release -- --ignored"]
fn checkpoint_scale_draws_match_sglang_cpu_path() {
    check(true);
}

fn check(checkpoint_scale: bool) {
    let cases = fixture("family.json")["cases"].as_array().unwrap().clone();
    let cases: Vec<&Value> = (cases.iter())
        .filter(|case| case["checkpoint_scale"].as_bool().unwrap_or(false) == checkpoint_scale)
        .collect();
    assert!(!cases.is_empty());
    let failures: Vec<String> = (cases.iter())
        .filter_map(|case| {
            let name = &case["name"];
            match replay(case) {
                Ok(actual) => compare(&actual, &case["expected"])
                    .err()
                    .map(|error| format!("{name}: {error}")),
                Err(error) => Some(format!("{name}: {error}")),
            }
        })
        .collect();
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

/// The fields `generate.py` records, computed from the Rust output.
fn replay(case: &Value) -> dynamo_multimodal::Result<Value> {
    let family = processor_from_spec(&with_size_tokens(&case["spec"]).to_string())?;
    let input_ids: Vec<i32> = (case["input_ids"].as_array().unwrap().iter())
        .map(|id| id.as_i64().unwrap() as i32)
        .collect();
    let media: Vec<DecodedMedia> = (case["draw"].as_array().into_iter().flatten())
        .map(draw)
        .collect();
    let mut metadata = Vec::new();
    for media in &media {
        let (DecodedMedia::Image { height, width, .. }
        | DecodedMedia::ImageRgba { height, width, .. }) = media
        else {
            unreachable!()
        };
        metadata.push(MediaMetadata::Image {
            width: *width as u32,
            height: *height as u32,
        });
    }
    for size in case["sizes"].as_array().into_iter().flatten() {
        metadata.push(MediaMetadata::Image {
            width: size[0].as_u64().unwrap() as u32,
            height: size[1].as_u64().unwrap() as u32,
        });
    }
    let items = (media.iter())
        .map(|media| family.process_item(media))
        .collect::<dynamo_multimodal::Result<Vec<_>>>()?;

    let layout = if items.is_empty() {
        family.layout_from_metadata(&input_ids, &metadata)?
    } else {
        family.layout(&input_ids, &items)?
    };
    let counts: Vec<usize> = (common::layout_counts(&layout).iter())
        .map(|&(features, _)| features)
        .collect();
    let expanded = token_layout::apply_layout(&input_ids, &layout, &counts, usize::MAX)?;
    let spans = (metadata.iter())
        .map(|image| family.num_media_tokens(image))
        .collect::<dynamo_multimodal::Result<Vec<_>>>()?;
    let mut summary = json!({
        "input_ids_sha256": common::sha256_le(expanded.input_ids.iter().map(|id| id.to_le_bytes())),
        "num_input_ids": expanded.input_ids.len(),
        // Python's offsets are the pad run, without the header and end token.
        "offsets": expanded.feature_ranges.iter().map(|ranges| {
            assert_eq!(ranges.len(), 1);
            [ranges[0].start, ranges[0].end - 1]
        }).collect::<Vec<_>>(),
        "spans": spans,
    });
    if !items.is_empty() {
        summary["image_grid_thw"] = items.iter().map(grid).collect();
        summary["feature_shapes"] = json!(
            items
                .iter()
                .map(|item| &item.feature.shape)
                .collect::<Vec<_>>()
        );
        summary["feature_sha256"] = (items.iter())
            .map(|item| {
                let TensorData::F32(values) = &item.feature.data else {
                    panic!("features are f32");
                };
                json!(common::sha256_le(values.iter().map(|v| v.to_le_bytes())))
            })
            .collect();
    }
    Ok(summary)
}

fn grid(item: &ProcessedItem) -> Value {
    let [(name, tensor)] = &item.aux[..] else {
        panic!("one model kwarg per item");
    };
    assert_eq!(
        (name.as_str(), &tensor.shape[..]),
        ("image_grid_thw", &[3][..])
    );
    let TensorData::I64(values) = &tensor.data else {
        panic!("image_grid_thw is i64");
    };
    json!(values)
}

/// `corpus.py::draw`: integer noise, with fully transparent and fully opaque
/// cells in the alpha band.
fn draw(recipe: &Value) -> DecodedMedia {
    let size = |key: &str| recipe[key].as_u64().unwrap() as usize;
    let (width, height, seed) = (size("width"), size("height"), size("seed") as u64);
    let bands = match recipe["kind"].as_str().unwrap() {
        "rgb" => 3,
        "rgba" => 4,
        other => panic!("unknown draw kind {other}"),
    };
    let mut pixels: Vec<u8> = (0..(height * width * bands) as u64)
        .map(|i| {
            let v = i
                .wrapping_add(seed.wrapping_mul(0x9E37_79B9))
                .wrapping_mul(2_654_435_761);
            ((v & 0xFFFF_FFFF) >> 24) as u8
        })
        .collect();
    if bands == 3 {
        return DecodedMedia::Image {
            rgb: pixels,
            height,
            width,
        };
    }
    for (p, pixel) in pixels.chunks_exact_mut(4).enumerate() {
        match (p % width / 7 + p / width / 5) % 3 {
            0 => pixel[3] = 0,
            1 => pixel[3] = 255,
            _ => {}
        }
    }
    DecodedMedia::ImageRgba {
        rgba: pixels,
        height,
        width,
    }
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

/// Both checkpoints ship one image processor config. It resolves to the spec
/// of every shipped-limit case of the table, and so do the files
/// `save_pretrained` writes for the image processor and for the whole processor.
#[test]
fn both_checkpoints_resolve_to_the_table_spec() {
    let ProcessorSpec::KimiK3(shipped) =
        serde_json::from_value(common::table("kimi-k3.json")["cases"][0]["spec"].clone()).unwrap()
    else {
        panic!("expected a Kimi-K3 spec");
    };
    let fixture = fixture("family.json");
    let checkpoints = fixture["checkpoints"].as_object().unwrap();
    assert_eq!(
        checkpoints.keys().collect::<Vec<_>>(),
        ["moonshotai/Kimi-K3", "moonshotai/Kimi-K3.1-Pro"]
    );
    for (model, files) in checkpoints {
        let resaved = &files["resaved"];
        assert_eq!(
            resaved["preprocessor_config"]["image_processor_type"],
            "KimiK3VisionProcessor"
        );
        assert_eq!(
            resaved["processor_config"]["processor_class"],
            "KimiK3Processor"
        );
        for pre in [
            &files["preprocessor_config"],
            &resaved["preprocessor_config"],
            &resaved["processor_config"],
        ] {
            let ProcessorSpec::KimiK3(spec) =
                spec_from_hf_configs(&files["config"].to_string(), &pre.to_string()).unwrap()
            else {
                panic!("expected a Kimi-K3 spec");
            };
            assert!(spec.size_tokens.is_none());
            assert_eq!(format!("{spec:?}"), format!("{shipped:?}"), "{model}");
        }
    }
}

/// A resolved spec has no token table: pixels work, the expansion does not.
#[test]
fn without_size_tokens_pixels_work_and_the_expansion_is_unsupported() {
    let table = common::table("kimi-k3.json");
    let case = &table["cases"][0];
    assert_eq!(case["images"], json!(["rgb_96x80.png"]));
    let family = processor_from_spec(&case["spec"].to_string()).unwrap();
    let media = family
        .decode_image(&common::image("rgb_96x80.png"), &DecodeLimits::default())
        .unwrap();
    let item = family.process_item(&media).unwrap();
    assert_eq!(json!([grid(&item)]), case["expected"]["image_grid_thw"]);
    let image = MediaMetadata::Image {
        width: 96,
        height: 80,
    };
    for result in [
        family.num_media_tokens(&image).map(|_| ()),
        family.layout_from_metadata(&[163605], &[image]).map(|_| ()),
        family.layout(&[163605], &[item]).map(|_| ()),
    ] {
        let Err(error @ MmError::Unsupported { .. }) = result else {
            panic!("the expansion needs the token table");
        };
        assert!(error.to_string().contains("size_tokens"), "{error}");
    }
}

/// `[width, height, num_tokens, new_width, new_height, span]`: the
/// checkpoint's `navit_resize_image`, and the prompt span with the real
/// tokenizer's header. The crate refuses the sizes it cannot plan exactly.
#[test]
fn plan_and_prompt_span_match_the_references_over_a_size_sweep() {
    use dynamo_multimodal::models::kimi_k3::{KimiK3Processor, KimiK3Spec};

    let counts = fixture("counts.json");
    let family = processor_from_spec(&with_size_tokens(&counts["spec"]).to_string()).unwrap();
    let spec: KimiK3Spec = serde_json::from_value(counts["spec"].clone()).unwrap();
    let typed = KimiK3Processor::new(spec).unwrap();
    let rows = counts["rows"].as_array().unwrap();
    let mut refused = 0;
    for row in rows {
        let [width, height, tokens, new_width, new_height, span] =
            core::array::from_fn(|i| row[i].as_u64().unwrap());
        let media = MediaMetadata::Image {
            width: width as u32,
            height: height as u32,
        };
        if (width / 14).max(1) * (height / 14).max(1) > 1 << 53 {
            refused += 1;
            assert!(matches!(
                family.num_media_tokens(&media),
                Err(MmError::LimitExceeded { .. })
            ));
            continue;
        }
        let plan = typed.plan(width as u32, height as u32).unwrap();
        assert_eq!(
            (plan.num_tokens as u64, plan.new_width, plan.new_height),
            (tokens, new_width, new_height),
            "{width}x{height}"
        );
        assert_eq!(
            family.num_media_tokens(&media).unwrap() as u64,
            span,
            "{width}x{height}"
        );
    }
    assert_eq!((rows.len(), refused), (3325, 18));
}
